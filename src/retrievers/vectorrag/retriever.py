import numpy as np
from vectorrag.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()

def faiss_search(index, query_embedding, top_k=5):
    query_np = np.array(query_embedding.cpu())
    D, I = index.search(query_np, top_k)
    return I[0], D[0]

def rerank_search(query, query_embedding, index, all_documents, top_k=10, rerank_k=5, return_scores=False):
    indices, distances = faiss_search(index, query_embedding, top_k)
    retrieved_docs = [all_documents[i] for i in indices]

    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.model.predict(pairs)

    scored_pairs = sorted(zip(scores, retrieved_docs), reverse=True)
    reranked_docs = [doc for score, doc in scored_pairs[:rerank_k]]
    reranked_scores = [score for score, doc in scored_pairs[:rerank_k]]

    return (reranked_docs, reranked_scores) if return_scores else reranked_docs