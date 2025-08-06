# src/retrievers/graphrag/graphRAG_retriever.py
import os
import argparse
from typing import List
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
import re

from .query_expansion2 import Stage1Understanding
from .hybrid_retriever2 import hybrid_candidates_boosted


load_dotenv()

# -------- Lucene-safe sanitization for BM25 --------
_SPECIALS_PATTERN = re.compile(r'[+\-!(){}\[\]^"~*?:\\/]')

def lucene_sanitize(text: str) -> str:
    """
    Make a user query safe for Neo4j's db.index.fulltext.queryNodes by
    removing Lucene special operators and normalizing whitespace.
    Also removes the two-token operators '&&' and '||'.
    """
    if not text:
        return ""
    # 1) flatten whitespace/newlines
    s = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    # 2) remove two-token operators
    s = re.sub(r'\s*&&\s*', ' ', s)
    s = re.sub(r'\s*\|\|\s*', ' ', s)
    # 3) remove single-char operators
    s = _SPECIALS_PATTERN.sub(' ', s)
    # 4) collapse spaces
    s = " ".join(s.split())
    # 5) fallback: if everything vanished, keep alphanumerics
    if not s:
        tokens = re.findall(r'\w+', text)
        s = " ".join(tokens)
    return s

def require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing env var: {name}")
    return v

NEO4J_URI      = require("NEO4J_URI")
NEO4J_USERNAME = require("NEO4J_USERNAME")
NEO4J_PASSWORD = require("NEO4J_PASSWORD")

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
_EMBED_MODEL = "text-embedding-ada-002"

# --------- Plug in your embedding function here ---------
# You can point this to OpenAI/HF/etc. For now, placeholder.
def embed(text: str) -> List[float]:
    """
    Return a 1536-d embedding using the same model your chunks used.
    """
    # OpenAI recommends stripping newlines
    text = (text or "").replace("\n", " ").strip()
    resp = _openai.embeddings.create(model=_EMBED_MODEL, input=text)
    return resp.data[0].embedding


def retrieve(query: str):
    # Stage 1
    s1 = Stage1Understanding()
    try:
        s1_out = s1.process(query)
    finally:
        s1.close()

    # Prepare filters for Stage 2 (issuer optional)
    filters = {"year": s1_out.filters.get("year"),
               "period": s1_out.filters.get("period"),
               "issuer": None}

    # Embed expanded text
    q_emb = embed(s1_out.expandedText)

    # Stage 2
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        # Use a Lucene-safe string for BM25; keep the original text for embeddings
        bm25_query = lucene_sanitize(s1_out.expandedText)
        #ids_for_gate = s1_out.entityIds_hi + s1_out.entityIds_med
        rows = hybrid_candidates_boosted(
            driver,
            expanded_text=bm25_query,
            q_embedding=q_emb,
            entity_ids_hi=s1_out.entityIds_hi,
            entity_ids_med=s1_out.entityIds_med,
            N1=1000,            # BM25 pool
            N2=1000,            # Vector pool
            top_k=10,                # <-- return top 5 chunks
            alpha=0.5, beta=0.35,
            gamma_hi=0.10, gamma_med=0.05
        )

    finally:
        driver.close()

    # TODO: Cross-encoder re-rank 'rows' here (optional)
    return {
        "stage1": s1_out.__dict__,
        "candidates": rows
    }

#if __name__ == "__main__":
#    ap = argparse.ArgumentParser()
#    ap.add_argument("--q", required=True, help="User query")
#    args = ap.parse_args()

#    result = retrieve(args.q)
#    # Pretty print minimal fields
#    print("\n=== Stage 1 ===")
#    print(result["stage1"])
#    print("\n=== Stage 2 candidates (top 5) ===")
#    for r in result["candidates"]:
#        print({k: r.get(k) for k in ["chunkId","fusedScore","entities"]})
#        print("TEXT:", (r.get("text") or "")[:400], "\n")

