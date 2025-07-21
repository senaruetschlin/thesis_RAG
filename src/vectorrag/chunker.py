# chunker.py

import ast

def chunk_finder_sample(sample):
    chunks = []

    context = sample.get("context", [])

    if isinstance(context, str):
        try:
            context = ast.literal_eval(context)
        except Exception:
            context = []  # fallback if parsing fails

    if isinstance(context, list):
        for segment in context:
            if isinstance(segment, str) and segment.strip():
                chunks.append(f"[Text] {segment.strip()}")

    return chunks


def chunk_all(data):
    all_chunks = []
    chunk_metadata = []

    for sample in data:
        chunks = chunk_finder_sample(sample)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append({
                "id": sample.get("ID"),
                "source": sample.get("source"),
                "question": sample.get("question"),
                "answer": sample.get("answer")
            })

    return all_chunks, chunk_metadata