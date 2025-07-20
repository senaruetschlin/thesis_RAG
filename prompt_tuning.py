import json
import time
from src.generator import ChatGPTGenerator
from src.evaluation import evaluate_rag

def prompt_tuning(
    train_path="data_processed/merged_train.json",
    prompt_path="generator_prompt.txt",
    num_samples=10,  # Set to None to use all samples
    metrics=None
):
    # Load train set
    with open(train_path) as f:
        train_data = json.load(f)
    if num_samples:
        train_data = train_data[:num_samples]

    # Load or edit prompt
    try:
        with open(prompt_path, "r") as f:
            system_prompt = f.read()
    except FileNotFoundError:
        system_prompt = "You are a precise financial assistant. Base answers strictly on context provided."
        print(f"Prompt file {prompt_path} not found. Using default prompt.")

    print("\nCurrent system prompt:\n", system_prompt)
    print("\nYou can edit the prompt in", prompt_path, "and rerun this script to try a new prompt.")

    # Instantiate generator
    generator = ChatGPTGenerator()

    # Generate predictions and collect latency
    predictions = []
    references = []
    latencies = []

    for sample in train_data:
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

    # Default metrics if not provided
    if metrics is None:
        metrics = [
            "context_precision",
            "context_recall",
            "faithfulness",
            "answer_accuracy",
            "string_presence",
            "latency"
        ]

    # Evaluate
    results = evaluate_rag(
        predictions,
        references,
        metrics,
        retriever_latency=latencies
    )

    print("\nEvaluation Results for current prompt:")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    prompt_tuning()