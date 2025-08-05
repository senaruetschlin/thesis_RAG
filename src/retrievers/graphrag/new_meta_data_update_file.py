import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph

# ========================
# CONFIG
# ========================
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise RuntimeError("Missing Neo4j connection info in .env")


CHUNK_META_JSONL = "/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/existing_embeddings_with_meta_data.jsonl"
LOG_PATH         = "/Users/christel/Desktop/Thesis/thesis_repo/graph_build_metadata.log"

# ========================
# LOGGING
# ========================
Path(Path(LOG_PATH).parent).mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("metadata_update")
logger.setLevel(logging.INFO)

fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

if not logger.handlers:
    logger.addHandler(fh)
    logger.addHandler(ch)

logger.info("=== Metadata update started ===")

# ========================
# CONNECT TO GRAPH
# ========================
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)

# ========================
# Cypher Queries
# ========================
CHECK_EXISTENCE_Q = """
MATCH (c:Chunk {id: $chunk_id})-[:PART_OF]->(d:Document)
RETURN count(*) AS cnt
"""

UPDATE_META_Q = """
MATCH (c:Chunk {id: $chunk_id})-[:PART_OF]->(d:Document)
SET  d.source    = coalesce($source,    d.source),
     d.source_id = coalesce($source_id, d.source_id),
     d.ticker    = coalesce($ticker,    d.ticker),
     d.year      = coalesce(toInteger($year), d.year),
     c.page      = coalesce(toInteger($page), c.page),
     c.has_metadata = any(v IN [d.ticker, d.year, c.page] WHERE v IS NOT NULL)
"""

# ========================
# LOAD AND APPLY METADATA
# ========================
updated = 0
skipped = 0

if not Path(CHUNK_META_JSONL).exists():
    logger.error(f"Metadata file not found: {CHUNK_META_JSONL}")
    raise SystemExit(1)

with open(CHUNK_META_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            meta = json.loads(line)
            chunk_id = meta.get("chunk_id")
            if not chunk_id:
                continue

            # Only update if node exists
            exists = graph.query(CHECK_EXISTENCE_Q, {"chunk_id": chunk_id})
            if exists[0]["cnt"] == 0:
                skipped += 1
                logger.warning(f"Skipping (not found): {chunk_id}")
                continue

            graph.query(
                UPDATE_META_Q,
                {
                    "chunk_id": chunk_id,
                    "source": meta.get("source"),
                    "source_id": meta.get("source_id"),
                    "ticker": meta.get("ticker"),
                    "year": meta.get("year"),
                    "page": meta.get("page"),
                }
            )
            updated += 1
            logger.info(f"Updated metadata: {chunk_id}")

        except Exception as e:
            logger.error(f"Failed to update {chunk_id}: {e}")

logger.info(f"=== Metadata update finished ===")
logger.info(f"Total updated: {updated} | Skipped (not found): {skipped}")
print(f"Updated: {updated} | Skipped: {skipped}")
print(f"Logs written to: {LOG_PATH}")
