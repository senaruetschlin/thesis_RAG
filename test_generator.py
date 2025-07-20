import json
import time
from src.generator import ChatGPTGenerator
from src.evaluation import evaluate_rag

# Load merged dataset
with open("data_processed/merged_dataset.json") as f:
    merged_dataset = json.load(f)

# Pick one example for initial test
sample = merged_dataset[0]
question = sample["question"]
context = sample["context_text"]

# Load system prompt
try:
    with open("generator_prompt.txt", "r") as f:
        system_prompt = f.read()
except FileNotFoundError:
    system_prompt = "You are a precise financial assistant. Base answers strictly on context provided."

# Instantiate generator
generator = ChatGPTGenerator()

# Generate answer for the first sample
answer = generator.generate(
    question=question,
    retrieved_docs=context,
    system_prompt=system_prompt
)

print("Question:", question)
print("Context (first 2 segments):", context[:2])
print("Generated Answer:", answer)
print("True Answer:", sample["answer"])

# --- Evaluation on first 5 samples ---
predictions = []
references = []
latencies = []

for sample in merged_dataset[:5]:
    question = sample["question"]
    context = sample["context_text"]
    true_answer = sample["answer"]

    start = time.time()
    generated_answer = generator.generate(
        question=question,
        retrieved_docs=context,
        system_prompt=system_prompt
    )
    end = time.time()
    latencies.append(end - start)

    predictions.append({
        "answer": generated_answer,
        "contexts": context
    })
    references.append({
        "answer": true_answer,
        "contexts": context
    })

metrics = [
    "context_precision",
    "context_recall",
    "faithfulness",
    "answer_accuracy",
    "string_presence",
    "latency"
    # "nvidia_metrics",  # Uncomment if available
]

results = evaluate_rag(
    predictions,
    references,
    metrics,
    retriever_latency=latencies
)

print("\nEvaluation Results:")
for k, v in results.items():
    print(f"{k}: {v}")
