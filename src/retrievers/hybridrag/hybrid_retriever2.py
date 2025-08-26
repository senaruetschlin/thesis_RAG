# src/retrievers/graphrag/hybrid_retrieval.py
from typing import List, Dict, Any
from neo4j import Driver
import re

# Index names in your DB
CHUNK_TEXT_FT = "chunk_text_ft"   # FULLTEXT on :Chunk(text)
CHUNK_VEC_IX  = "chunkIndex"      # VECTOR on :Chunk(embedding)

# -----------------------
# Main Hybrid Retrieval
# -----------------------

# full_return_retrieval.py

CYPHER_HYBRID_BOOSTED = f"""
CALL (){{
  WITH $expandedText AS q
  CALL db.index.fulltext.queryNodes('{CHUNK_TEXT_FT}', q) YIELD node, score
  WHERE node:Chunk
  RETURN elementId(node) AS cid, score AS bm25, 0.0 AS vec
  LIMIT $N1

  UNION ALL

  WITH $qEmbedding AS emb
  CALL db.index.vector.queryNodes('{CHUNK_VEC_IX}', $N2, emb) YIELD node, score
  WHERE node:Chunk
  RETURN elementId(node) AS cid, 0.0 AS bm25, score AS vec
}}

WITH cid, bm25, vec
WITH cid, max(bm25) AS bm25, max(vec) AS vec
WITH collect({{cid:cid, bm25:bm25, vec:vec}}) AS rows
WITH rows,
     reduce(maxB=0.0, r IN rows | CASE WHEN r.bm25>maxB THEN r.bm25 ELSE maxB END) AS maxBM25,
     reduce(maxV=0.0, r IN rows | CASE WHEN r.vec>maxV THEN r.vec ELSE maxV END) AS maxVec
UNWIND rows AS r
WITH r.cid AS cid,
     CASE WHEN maxBM25>0 THEN r.bm25/maxBM25 ELSE 0 END AS bm25N,
     CASE WHEN maxVec >0  THEN r.vec /maxVec  ELSE 0 END AS vecN

MATCH (c:Chunk) WHERE elementId(c)=cid
OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
WITH c, bm25N, vecN, collect(DISTINCT e.id) AS ents
WITH c,
     bm25N, vecN,
     CASE WHEN size([x IN ents WHERE x IN $entityIds_hi])  > 0 THEN 1.0 ELSE 0.0 END AS hiOverlap,
     CASE WHEN size([x IN ents WHERE x IN $entityIds_med]) > 0 THEN 1.0 ELSE 0.0 END AS medOverlap

WITH c,
     $alpha*vecN + $beta*bm25N + $gamma_hi*hiOverlap + $gamma_med*medOverlap AS fused


ORDER BY fused DESC
LIMIT $M_pool   // <- ensure this is M_pool (pool size for Python-side dedup)

OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
RETURN c.id   AS chunkId,
       c.text AS text,
       fused  AS fusedScore,
       collect(DISTINCT e.id) AS entities
ORDER BY fusedScore DESC
"""


# -----------------------------
# Python-side similarity dedup
# -----------------------------

_word_re = re.compile(r"\b\w+\b", re.UNICODE)

def _token_set(text: str) -> set:
    if not text:
        return set()
    # lower-case, keep word characters only, split to tokens, convert to set
    return set(m.group(0).lower() for m in _word_re.finditer(text))

def jaccard_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0

# Alternative (order-sensitive) similarity:
# import difflib
# def difflib_similarity(a: str, b: str) -> float:
#     return difflib.SequenceMatcher(None, a or "", b or "").ratio()

def dedup_by_similarity(rows: List[Dict[str, Any]],
                        top_k: int,
                        threshold: float = 0.80,
                        sim_fn = jaccard_similarity) -> List[Dict[str, Any]]:
    """
    Keep rows in descending fusedScore order, but drop any row whose text is
    >= threshold similar to any already-kept row. Stop when we have top_k.
    """
    kept: List[Dict[str, Any]] = []
    texts: List[str] = []
    for r in rows:
        t = r.get("text") or ""
        # compute similarity against already kept items
        too_similar = False
        for t_kept in texts:
            if sim_fn(t, t_kept) >= threshold:
                too_similar = True
                break
        if not too_similar:
            kept.append(r)
            texts.append(t)
            if len(kept) >= top_k:
                break
    return kept

def hybrid_candidates_boosted(driver: Driver,
                              expanded_text: str,
                              q_embedding: List[float],
                              entity_ids_hi: List[str],
                              entity_ids_med: List[str],
                              N1: int = 1000,
                              N2: int = 1000,
                              top_k: int  = 5,
                              M_pool: int = 100,   # retrieve a generous pool before dedup
                              alpha: float = 0.5,
                              beta: float = 0.35,
                              gamma_hi: float = 0.10,
                              gamma_med: float = 0.05,
                              dedup_threshold: float = 0.80) -> List[Dict[str, Any]]:
    """
    1) Retrieve a pool (M_pool) using BM25 + vector + (hi/med) boost (ungated).
    2) Deduplicate rows by text similarity (>= dedup_threshold) and keep top_k.
    """
    params = dict(
        expandedText=expanded_text,
        qEmbedding=q_embedding,
        N1=N1, N2=N2,
        M_pool=M_pool,
        alpha=alpha,
        beta=beta,
        gamma_hi=gamma_hi,
        gamma_med=gamma_med,
        entityIds_hi=entity_ids_hi,
        entityIds_med=entity_ids_med
    )
    with driver.session() as sess:
        pool = sess.run(CYPHER_HYBRID_BOOSTED, **params).data()

    # pool is already sorted by fusedScore desc; apply similarity-dedup, keep top_k
    return dedup_by_similarity(pool, top_k=top_k, threshold=dedup_threshold)