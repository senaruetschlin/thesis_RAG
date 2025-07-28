# vectorrag/index_faiss.py
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings  # ← add this
from typing import List, Optional
import json
import numpy as np


def build_faiss_index(
    docs: List[Document], 
    embedder, 
    save_path: Optional[str] = None
) -> FAISS:
    """
    Build a FAISS index from a list of LangChain Documents and save it optionally.

    Args:
        docs (List[Document]): Chunked documents.
        embedder: An embedding model (e.g., OpenAIEmbeddings).
        save_path (str, optional): Directory path to save the FAISS index.

    Returns:
        FAISS: The created FAISS vectorstore object.
    """
    vectorstore = FAISS.from_documents(docs, embedder)
    if save_path:
        vectorstore.save_local(save_path)
    return vectorstore


def build_faiss_index_from_json(json_path: str, embedder: Embeddings, save_path: Optional[str] = None) -> FAISS:
    with open(json_path, "r") as f:
        data = json.load(f)

    texts = [item["text"] for item in data]
    embeddings = [item["embedding"] for item in data]
    metadatas = [{"chunk_id": item["chunk_id"], "row_index": item["row_index"]} for item in data]

    # FAISS expects float32
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index using LangChain
    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, embeddings)),
        embedding=embedder,
        metadatas=metadatas
    )

    if save_path:
        vectorstore.save_local(save_path)

    return vectorstore

def load_faiss_index(
    load_path: str, 
    embedder
) -> FAISS:
    """
    Load an existing FAISS index from disk.

    Args:
        load_path (str): Path where the FAISS index is stored.
        embedder: The same embedding model used to build the index.

    Returns:
        FAISS: The loaded FAISS vectorstore object.
    """
    return FAISS.load_local(load_path, embedder)