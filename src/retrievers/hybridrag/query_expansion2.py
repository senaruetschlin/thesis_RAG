import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import spacy
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()  # loads .env into the environment if present

def require(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


# ---------------------------
# Config (override via env)
# ---------------------------

NEO4J_URI = require("NEO4J_URI")
NEO4J_USERNAME = require("NEO4J_USERNAME")      
NEO4J_PASSWORD = require("NEO4J_PASSWORD")


SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
ENTITY_FT_INDEX = os.getenv("ENTITY_FT_INDEX", "entity_name_ft")  # created once as shown above
EL_FT_TOPK = int(os.getenv("EL_FT_TOPK", "3"))

# How many FT hits to consider per mention
EL_FT_TOPK = int(os.getenv("EL_FT_TOPK", "3"))
# Treat the top FT hit as HIGH?
EL_FT_TOP1_AS_HIGH = os.getenv("EL_FT_TOP1_AS_HIGH", "1") == "1"
# Optional: require a minimum FT score to accept as HIGH (0 disables)
EL_FT_MIN_HIGH_SCORE = float(os.getenv("EL_FT_MIN_HIGH_SCORE", "0"))
# How many "contains-token" ids to add as MEDIUM
EL_CONTAIN_TOPK = int(os.getenv("EL_CONTAIN_TOPK", "5"))


# ---------------------------
# Helpers
# ---------------------------

def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", s.replace("\u00a0", " ")).strip().lower()

def _normalize_org(s: str) -> str:
    """Light canonicalization for company names."""
    s = s.lower()
    # drop common suffixes; feel free to extend this list
    s = re.sub(r"\b(corp(oration)?|inc(orporated)?|ltd|co|company)\b\.?", "", s)
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Basic metric lexicon to catch things spaCy may miss (extend as needed)
METRIC_TERMS = {
    "eps", "earnings per share", "net income", "revenue", "ebit", "ebitda",
    "operating income", "operating expenses", "interest expense", "cash flow",
    "free cash flow", "gross margin", "operating margin", "roe", "roa"
}

YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b", re.IGNORECASE)
Q_PATTERN = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)
FY_PATTERN = re.compile(r"\bFY\b", re.IGNORECASE)

@dataclass
class Stage1Output:
    expandedText: str           # here: just the original query (no aliases)
    entityIds_hi: List[str]
    entityIds_med: List[str]
    filters: Dict[str, Any]


class Stage1Understanding:
    """
    Stage 1: NER + Entity Linking (no alias terms).
    - Exact match on :Entity{id}  -> high confidence
    - Full-text fallback on :Entity(id) -> medium confidence
    """

    def __init__(self,
                 neo4j_uri: str = NEO4J_URI,
                 neo4j_user: str = NEO4J_USERNAME,
                 neo4j_password: str = NEO4J_PASSWORD,
                 spacy_model: str = SPACY_MODEL):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.nlp = spacy.load(spacy_model)

    # -------- NER --------

    def ner(self, q: str) -> List[Dict[str, str]]:
        """Return list of spans with coarse types for ORG / DATE / FIN_METRIC / TICKER."""
        doc = self.nlp(q)
        spans: List[Dict[str, str]] = []

        # 1) spaCy entities
        for ent in doc.ents:
            label = ent.label_.upper()
            txt = ent.text.strip()
            if label in {"ORG"}:
                typ = "ORG"
            elif label in {"DATE"}:
                typ = "DATE"
            elif label in {"MONEY", "PERCENT"}:
                typ = label
            else:
                typ = label
            spans.append({"text": txt, "type": typ})

        # 2) Very light metric/ticker detection (complements spaCy)
        tokens = [t.text for t in doc]
        q_lower = q.lower()

        # Metrics by lexicon presence
        for term in METRIC_TERMS:
            if term in q_lower:
                spans.append({"text": term, "type": "FIN_METRIC"})

        # Ticker heuristic: all‑caps 1–5 letters (naive—adjust as needed) and abbreviated financial terms
        for t in tokens:
            if 1 <= len(t) <= 5 and t.isupper() and t.isalpha():
                spans.append({"text": t, "type": "TICKER"})

        # Deduplicate spans (text, type)
        seen = set()
        uniq = []
        for s in spans:
            key = (_norm(s["text"]), s["type"])
            if key not in seen:
                seen.add(key)
                uniq.append(s)
        return uniq

    # -------- EL queries --------

    @staticmethod
    def _cypher_exact_entity() -> str:
        return """
        MATCH (e:Entity)
        WHERE toLower(e.id) = toLower($m)
        RETURN e.id AS id
        """

    @staticmethod
    def _cypher_entity_fulltext(index_name: str) -> str:
        return f"""
        CALL db.index.fulltext.queryNodes('{index_name}', $m)
        YIELD node, score
        RETURN node.id AS id, score
        ORDER BY score DESC
        LIMIT $k
        """

    @staticmethod
    def _cypher_entity_contains_token() -> str:
        # Case-insensitive whole-word match on e.id via regex
        return """
        MATCH (e:Entity)
        WHERE toLower(e.id) =~ $regex
        RETURN e.id AS id
        LIMIT $k
        """
    



    def resolve_mention(self, mention: str) -> Tuple[Set[str], Set[str]]:
        """
        Return (high_conf_ids, med_conf_ids) for a single mention.

        - HIGH: exact id match (case-insensitive); and optionally the TOP-1 full-text hit
            (if any, and above an optional score threshold).
        - MED:  remaining full-text hits (excluding HIGH) plus ids whose id contains the
            mention as a WHOLE WORD (regex), excluding HIGH again.
        """
        hi: Set[str] = set()
        med: Set[str] = set()

        m = (mention or "").strip()
        if not m:
            return hi, med

        m_norm = _normalize_org(m)

        with self.driver.session() as sess:
            # 1) HIGH: exact match on id
            hi.update(sess.run(self._cypher_exact_entity(), m=m).value("id"))

            if not hi and m_norm and m_norm != m:
                hi.update(sess.run(self._cypher_exact_entity(), m=m_norm).value("id"))

            # 2) FULL-TEXT: gather top-K hits and optionally promote top-1 to HIGH
            ft_rows = []
            if ENTITY_FT_INDEX:
                ft_rows = sess.run(self._cypher_entity_fulltext(ENTITY_FT_INDEX),
                                m=m, k=EL_FT_TOPK).data()
                if m_norm and m_norm != m:
                    ft_rows += sess.run(self._cypher_entity_fulltext(ENTITY_FT_INDEX),
                                        m=m_norm, k=EL_FT_TOPK).data()

            # Normalize rows to (id, score), keep best score per id
            ft_hits = []
            for r in ft_rows:
                # rows can come back as {'id':..., 'score':...} or [id, score] depending on driver use
                if isinstance(r, dict):
                    _id, _score = r.get("id") or r.get("node.id"), r.get("score", 0.0)
                else:
                    _id, _score = r[0], (r[1] if len(r) > 1 else 0.0)
                if _id:
                    ft_hits.append((_id, float(_score)))

            # Dedup while keeping best score
            best = {}
            for _id, sc in ft_hits:
                best[_id] = max(sc, best.get(_id, 0.0))
            ft_hits = sorted(best.items(), key=lambda x: x[1], reverse=True)

            # Promote FT top-1 to HIGH (if enabled & above threshold)
            if ft_hits and EL_FT_TOP1_AS_HIGH:
                top_id, top_score = ft_hits[0]
                if top_id not in hi and (EL_FT_MIN_HIGH_SCORE <= 0 or top_score >= EL_FT_MIN_HIGH_SCORE):
                    hi.add(top_id)
                # Remaining FT hits → MED
                for _id, _ in ft_hits[1:]:
                    if _id not in hi:
                        med.add(_id)
            else:
                for _id, _ in ft_hits:
                    if _id not in hi:
                        med.add(_id)

            # 3) WHOLE-WORD CONTAINS on id (MEDIUM)
            # Build a case-insensitive word-boundary regex, e.g., (?i).*\bgm\b.*
            mention_token = re.escape(m.lower())
            regex = f"(?i).*\\b{mention_token}\\b.*"
            contain_rows = sess.run(
                self._cypher_entity_contains_token(),
                regex=regex,
                k=EL_CONTAIN_TOPK
            ).value("id")
            for _id in contain_rows:
                if _id not in hi:
                    med.add(_id)

        return hi, med



    # -------- Filters --------

    def parse_filters(self, q: str, spans: List[Dict[str, str]]) -> Dict[str, Any]:
        years = {int(m.group(0)) for m in YEAR_PATTERN.finditer(q)}
        period = None
        mq = Q_PATTERN.search(q)
        if mq:
            period = f"Q{mq.group(1)}"
        elif FY_PATTERN.search(q):
            period = "FY"
        issuer_texts = [s["text"] for s in spans if s["type"] == "ORG"]
        return {
            "year": sorted(years) if years else None,
            "period": period,
            "issuer_texts": issuer_texts or None
        }

    # -------- Main API --------

    def process(self, query: str) -> Stage1Output:
        spans = self.ner(query)

        hi_ids: Set[str] = set()
        med_ids: Set[str] = set()

        # Resolve orgs / metrics / tickers
        for s in spans:
            if s["type"] in {"ORG", "FIN_METRIC", "TICKER"}:
                hi, med = self.resolve_mention(s["text"])
                hi_ids.update(hi)
                med_ids.update({x for x in med if x not in hi_ids})

        # Use issuer_texts (if any) through EL too
        filters = self.parse_filters(query, spans)
        for it in (filters.get("issuer_texts") or []):
            hi, med = self.resolve_mention(it)
            hi_ids.update(hi)
            med_ids.update({x for x in med if x not in hi_ids})

        # No alias expansion → expandedText is just the original query
        expanded = query

        return Stage1Output(
            expandedText=expanded,
            entityIds_hi=sorted(hi_ids),
            entityIds_med=sorted(med_ids - hi_ids),
            filters=filters
        )

    def close(self):
        self.driver.close()


# ---------------------------
# Example usage
# ---------------------------

if __name__ == "__main__":
    """
    Before running:
      export NEO4J_URI="neo4j+s://<db-id>.databases.neo4j.io"
      export NEO4J_USER="neo4j"
      export NEO4J_PASSWORD="*****"

    One-time (in Neo4j) for EL fallback:
      CALL db.index.fulltext.createNodeIndex('entity_name_ft', ['Entity'], ['id']);
    """
    s1 = Stage1Understanding()
    try:
        q = "What was Apple's EPS in FY 2023?"
        out = s1.process(q)
        print("expandedText:", out.expandedText)
        print("entityIds_hi:", out.entityIds_hi)
        print("entityIds_med:", out.entityIds_med)
        print("filters:", out.filters)
    finally:
        s1.close()
