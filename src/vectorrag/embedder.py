from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
model.max_seq_length = 512

if torch.backends.mps.is_available():
    model.to("mps")
    print("Using MPS")
else:
    print("Using CPU")

def embed_passages(passages: list, batch_size: int = 64):
    embeddings = []
    total = len(passages)
    print(f"Embedding {total} passages in batches of {batch_size}...")
    for i in tqdm(range(0, total, batch_size), desc="🔄 Embedding progress"):
        batch = passages[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings.cpu())
    return torch.cat(embeddings)

def embed_query(query: str):
    prefixed = "Represent this sentence for searching relevant passages: " + query
    return model.encode([prefixed], convert_to_tensor=True)