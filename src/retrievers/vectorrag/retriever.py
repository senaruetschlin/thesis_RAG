import faiss
import numpy as np
from pathlib import Path
from vectorrag.reranker import CrossEncoderReranker

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # or IndexFlatIP for dot product
    index.add(embeddings.cpu().numpy())
    return index

def search(index, query_embedding, top_k=5):
    query_np = np.array(query_embedding.cpu())
    D, I = index.search(query_np, top_k)
    return I[0], D[0]

def save_index(index, path="data/embeddings/faiss_index.index"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))
    print(f"FAISS index saved to {path}")

def load_index(path="data/embeddings/faiss_index.index"):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at {path}")
    print(f"Loaded FAISS index from {path}")
    return faiss.read_index(str(path))

def rerank_search(query: str, query_embedding, index, all_documents, top_k=10, rerank_k=5, return_scores=False):
    """
    Performs FAISS retrieval and reranks using a cross-encoder.
    
    Parameters:
        query (str): Natural language query.
        query_embedding (Tensor): Vector embedding of the query.
        index (faiss.Index): FAISS index object.
        all_documents (List[str]): Full list of documents aligned with FAISS index.
        top_k (int): Number of documents to retrieve from FAISS.
        rerank_k (int): Number of top documents to return after reranking.
        return_scores (bool): Whether to return relevance scores.

    Returns:
        List[str]: Top reranked documents.
        List[float] (optional): Relevance scores from reranker.
    """
    # Step 1: Retrieve top_k indices from FAISS
    indices, distances = search(index, query_embedding, top_k=top_k)
    retrieved_docs = [all_documents[i] for i in indices]

    # Step 2: Rerank using cross-encoder
    reranker = CrossEncoderReranker()
    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.model.predict(pairs)  # returns list of floats

    # Step 3: Sort by descending score
    scored_pairs = sorted(zip(scores, retrieved_docs), reverse=True)
    reranked_docs = [doc for score, doc in scored_pairs[:rerank_k]]
    reranked_scores = [score for score, doc in scored_pairs[:rerank_k]]

    if return_scores:
        return reranked_docs, reranked_scores
    return reranked_docs