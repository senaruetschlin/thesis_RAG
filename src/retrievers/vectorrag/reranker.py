# vectorrag/reranker.py

from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def predict(self, query_doc_pairs: list[tuple[str, str]]) -> list[float]:
        """
        Predict relevance scores for (query, document) pairs.
        Returns a list of float scores (higher = more relevant).
        """
        return self.model.predict(query_doc_pairs)

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[str]:
        """
        Rerank documents based on CrossEncoder scores.
        """
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.predict(pairs)
        ranked = [doc for _, doc in sorted(zip(scores, documents), key=lambda x: x[0], reverse=True)]
        return ranked[:top_k]