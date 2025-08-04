from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-electra-base", batch_size: int = 16):
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size

    def predict(self, query_doc_pairs: list[tuple[str, str]]) -> list[float]:
        all_scores = []
        for batch in chunked(query_doc_pairs, self.batch_size):
            batch_scores = self.model.predict(batch)
            all_scores.extend(batch_scores)
        return all_scores

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        pairs = [(query, doc["text"]) for doc in documents]
        scores = self.predict(pairs)

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = score

        ranked = sorted(documents, key=lambda d: d["rerank_score"], reverse=True)
        return ranked[:top_k]