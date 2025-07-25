from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from neo4j_retriever import get_neo4j_vector_retriever
import os

# Initialize LLM
llm = ChatOpenAI(
    model_name="gpt-4o",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Load retriever
retriever = get_neo4j_vector_retriever()

# Setup RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Ask a question
query = "What is the impact of AI on data privacy?"
response = rag_chain.run(query)
print(response)
