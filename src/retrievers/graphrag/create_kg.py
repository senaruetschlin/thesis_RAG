import os
import json
import logging
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_community.graphs.graph_document import Node, Relationship
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

# === Setup Logging ===
logging.basicConfig(filename="graph_build.log", level=logging.INFO, format="%(asctime)s %(message)s")

# === Load environment variables ===
load_dotenv()

# === Configuration ===
DOCS_PATH = "/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/unique_contexts_filtered.json"
RESUME_FROM_ROW = 0  
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

# === Retry-enabled LLM Init ===
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def init_llm():
    return ChatOpenAI(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        model_name="gpt-4o-mini"
    )

llm = init_llm()

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

# === Ensure Neo4j Constraints ===
def setup_constraints():
    constraints = [
        "CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT unique_document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE"
    ]
    for constraint in constraints:
        graph.query(constraint)

setup_constraints()

doc_transformer = LLMGraphTransformer(llm=llm)

# === Load Dataset ===
with open(DOCS_PATH) as f:
    data = json.load(f)

# === Create LangChain Document objects ===
docs = [
    Document(
        page_content=context.strip(),
        metadata={"row_index": idx}
    )
    for idx, context in enumerate(data)
    if isinstance(context, str) and context.strip()
]

# === Text Splitter ===
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# === Processing Pipeline ===
for doc in docs:
    row_index = doc.metadata["row_index"]
    
    if row_index < RESUME_FROM_ROW:
        continue

    chunks = text_splitter.split_documents([doc])

    for local_idx, chunk in enumerate(chunks):
        chunk_id = f"row_{row_index}_chunk_{local_idx}"

        # === Check if chunk exists and has embedding ===
        existing = graph.query(
            """
            MATCH (c:Chunk {id: $chunk_id})
            RETURN exists(c.embedding) AS has_embedding
            """,
            {"chunk_id": chunk_id}
        )

        if existing and existing[0]["has_embedding"]:
            logging.info(f"✅ Skipping chunk (already embedded): {chunk_id}")
            continue

        logging.info(f"🚧 Processing chunk: {chunk_id}")

        try:
            # === Embedding Generation ===
            chunk_embedding = embedding_provider.embed_query(chunk.page_content)

            # === Insert Chunk and Document ===
            graph.query("""
                MERGE (d:Document {id: $row_index})
                MERGE (c:Chunk {id: $chunk_id})
                SET c.text = $text
                MERGE (d)<-[:PART_OF]-(c)
                WITH c
                CALL db.create.setNodeVectorProperty(c, 'embedding', $embedding)
            """, {
                "row_index": row_index,
                "chunk_id": chunk_id,
                "text": chunk.page_content,
                "embedding": chunk_embedding
            })

            # === Graph Transformer Extraction ===
            graph_docs = doc_transformer.convert_to_graph_documents([chunk])

            for graph_doc in graph_docs:
                chunk_node = Node(id=chunk_id, type="Chunk")
                for node in graph_doc.nodes:
                    node.type = "Entity"
                    node.properties["original_type"] = node.properties.get("type", "Unknown")
                    graph_doc.relationships.append(
                        Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                    )

            # === Insert Extracted Graph ===
            graph.add_graph_documents(
                graph_docs,
                baseEntityLabel="Entity",
                include_source=True
            )

        except Exception as e:
            logging.error(f"❌ Failed processing chunk {chunk_id}: {str(e)}")

# === Create Vector Index (if missing) ===
graph.query("""
    CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
    FOR (c:Chunk) ON (c.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }
    }
""")

logging.info("🎉 All processing completed successfully!")
