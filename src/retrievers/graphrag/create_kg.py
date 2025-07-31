import os
import json
import logging
import signal
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.graphs.graph_document import Node, Relationship
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

# === Setup Logging ===
logging.basicConfig(filename="graph_build.log", level=logging.INFO, format="%(asctime)s %(message)s")

# === Load Environment Variables ===
load_dotenv()

# === Config ===
EMBEDDINGS_JSON = "/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/embedded_chunks.json"
CHUNK_VECTOR_PROPERTY = "embedding"
TIMEOUT_SECONDS = 300

# === Init LLM ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_llm():
    return ChatOpenAI(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model_name="gpt-4o-mini"
    )

llm = init_llm()

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# === Timeout Utilities ===
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, timeout_handler)

# === Enforce Constraints ===
def setup_constraints():
    constraints = [
        "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT unique_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
    ]
    for constraint in constraints:
        graph.query(constraint)

setup_constraints()

doc_transformer = LLMGraphTransformer(llm=llm)

# === Load Precomputed Embeddings ===
with open(EMBEDDINGS_JSON, "r") as f:
    embedded_chunks = json.load(f)

start_from = 4637

# === Insert Pipeline ===
for item in embedded_chunks:
    chunk_id = item["chunk_id"]
    row_index = item["row_index"]
    text = item["text"]
    embedding = item["embedding"]

    if row_index < start_from:
        continue

    existing = graph.query(
        """
        MATCH (c:Chunk {id: $chunk_id})
        RETURN c.embedding IS NOT NULL AS has_embedding
        """,
        {"chunk_id": chunk_id}
    )

    if existing and existing[0]["has_embedding"]:
        logging.info(f"Skipping (already in DB): {chunk_id}")
        continue

    try:
        logging.info(f"Inserting chunk: {chunk_id}")

        # === Insert Chunk + Document ===
        signal.alarm(TIMEOUT_SECONDS)
        try:
            graph.query("""
                MERGE (d:Document {id: $row_index})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                MERGE (d)<-[:PART_OF]-(c)
                WITH c
                CALL db.create.setNodeVectorProperty(c, $embedding_property, $embedding)
            """, {
                "row_index": row_index,
                "chunk_id": chunk_id,
                "text": text,
                "embedding": embedding,
                "embedding_property": CHUNK_VECTOR_PROPERTY
            })
            signal.alarm(0)
        except TimeoutException:
            logging.error(f"Timeout inserting chunk {chunk_id}")
            continue

        # === Graph Extraction ===
        signal.alarm(TIMEOUT_SECONDS)
        try:
            fake_doc = Document(page_content=text, metadata={"row_index": row_index})
            graph_docs = doc_transformer.convert_to_graph_documents([fake_doc])
            signal.alarm(0)
        except TimeoutException:
            logging.error(f"Timeout transforming graph for chunk {chunk_id}")
            continue

        # === Add Graph Documents ===
        signal.alarm(TIMEOUT_SECONDS)
        try:
            for graph_doc in graph_docs:
                chunk_node = Node(id=chunk_id, type="Chunk")
                for node in graph_doc.nodes:
                    node.type = "Entity"
                    node.properties["original_type"] = node.properties.get("type", "Unknown")
                    graph_doc.relationships.append(
                        Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                    )

            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel="Entity",
                include_source=True
            )
            signal.alarm(0)
        except TimeoutException:
            logging.error(f"Timeout adding graph documents for chunk {chunk_id}")
            continue

    except Exception as e:
        logging.error(f"Failed chunk {chunk_id}: {str(e)}")

# === Vector Index Creation (if missing) ===
graph.query(f"""
    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
    FOR (c:Chunk) ON (c.{CHUNK_VECTOR_PROPERTY})
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }}
    }}
""")

logging.info("All pre-embedded chunks successfully processed!")