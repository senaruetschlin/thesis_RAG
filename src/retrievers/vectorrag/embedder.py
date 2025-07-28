import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

def init_embedder():
    load_dotenv()
    return OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model="text-embedding-ada-002"
    )