import numpy as np
from .reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()

#def faiss_search(index, query_embedding, top_k=5):
#    # Ensure numpy, 2D, float32
#    query_np = np.atleast_2d(query_embedding.detach().cpu().numpy().astype('float32'))
#    D, I = index.index.search(query_np, top_k)  # Use raw FAISS object (index.index is OK if you wrapped)
#    return I[0], D[0]

def faiss_search(index, query_embedding, top_k=5):
    # Ensure numpy, 2D, float32
    query_np = np.atleast_2d(query_embedding.detach().cpu().numpy().astype('float32'))
    D, I = index.search(query_np, top_k)   # ✅ use index.search directly
    return I[0], D[0]



def rerank_search(query, query_embedding, index, all_documents, top_k=10, rerank_k=5, return_scores=False):
    indices, distances = faiss_search(index, query_embedding, top_k)
    retrieved_docs = [all_documents[i] for i in indices]

    pairs = [(query, doc) for doc in retrieved_docs]
    scores = reranker.model.predict(pairs)  # assumes higher is better

    scored_pairs = sorted(zip(scores, retrieved_docs), key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for score, doc in scored_pairs[:rerank_k]]
    reranked_scores = [score for score, doc in scored_pairs[:rerank_k]]

    return (reranked_docs, reranked_scores) if return_scores else reranked_docs
