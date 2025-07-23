import os
import json
from dotenv import load_dotenv

from langchain_community.graphs.graph_document import Node, Relationship
from langchain_community.document_loaders import PyPDFLoader  # still unused, safe to remove
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Load environment variables
load_dotenv()

# === Paths ===
DOCS_PATH = "/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/unique_contexts_filtered.json"

# === Setup OpenAI + Neo4j ===
llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'), 
    model_name="gpt-4o-mini"
)

embedding_provider = OpenAIEmbeddings(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model="text-embedding-ada-002"
)

graph = Neo4jGraph(
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD')
)

doc_transformer = LLMGraphTransformer(
    llm=llm
    # Optionally limit extracted types:
    # allowed_nodes=["Concept", "Technology", "Metric"],
    # allowed_relationships=["HAS", "DESCRIBES", "RELATES_TO"]
)

# === Load Dataset ===
with open(DOCS_PATH) as f:
    data = json.load(f)

# === Create LangChain Document objects ===
docs = []
for idx, item in enumerate(data):
    context = item.get("context", [])  # list of strings
    combined_context = "\n".join(context)
    if combined_context.strip():  # skip empty ones
        doc = Document(
            page_content=combined_context,
            metadata={"row_index": idx}
        )
        docs.append(doc)

# === Chunk Splitter (applied per doc below) ===
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1500,
    chunk_overlap=200
)

# === Chunk, Embed, and Add to Neo4j ===
for doc in docs:
    row_index = doc.metadata["row_index"]
    chunks = text_splitter.split_documents([doc])

    for local_idx, chunk in enumerate(chunks):
        chunk_id = f"row_{row_index}_chunk_{local_idx}"
        print(f"Processing: {chunk_id}")

        # Embed chunk
        chunk_embedding = embedding_provider.embed_query(chunk.page_content)

        # Insert Document + Chunk into Neo4j
        properties = {
            "row_index": row_index,
            "chunk_id": chunk_id,
            "text": chunk.page_content,
            "embedding": chunk_embedding
        }

        graph.query("""
            MERGE (d:Document {id: $row_index})
            MERGE (c:Chunk {id: $chunk_id})
            SET c.text = $text
            MERGE (d)<-[:PART_OF]-(c)
            WITH c
            CALL db.create.setNodeVectorProperty(c, 'textEmbedding', $embedding)
        """, properties)

        # Extract graph documents from chunk
        graph_docs = doc_transformer.convert_to_graph_documents([chunk])

        # Link Chunk to Entities
        for graph_doc in graph_docs:
            chunk_node = Node(id=chunk_id, type="Chunk")
            for node in graph_doc.nodes:
                graph_doc.relationships.append(
                    Relationship(source=chunk_node, target=node, type="HAS_ENTITY")
                )

        # Add extracted graph documents to Neo4j
        graph.add_graph_documents(graph_docs)

# === Vector Index Creation ===
graph.query("""
    CREATE VECTOR INDEX `chunkVector`
    IF NOT EXISTS
    FOR (c: Chunk) ON (c.textEmbedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 1536,
            `vector.similarity_function`: 'cosine'
        }
    };
""")
# please don't delete
