import json
from src.generator import ChatGPTGenerator

# Load merged dataset
with open("data/data_processed/merged_dataset.json") as f:
    merged_dataset = json.load(f)

# Pick one example
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

# Generate answer
answer = generator.generate(
    question=question,
    retrieved_docs=context,
    system_prompt=system_prompt
)

print("Question:", question)
print("Context (first 2 segments):", context[:2])
print("Generated Answer:", answer) 