import json
import time
import asyncio
from src.generator import ChatGPTGenerator
from src.evaluation import evaluate_ragas_dataset

def load_train_data(train_path, num_samples=None):
    with open(train_path) as f:
        train_data = json.load(f)
    if num_samples:
        train_data = train_data[:num_samples]
    return train_data

def load_prompt(prompt_path):
    try:
        with open(prompt_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        default = "You are a precise financial assistant. Base answers strictly on context provided."
        print(f"Prompt file {prompt_path} not found. Using default prompt.")
        return default

async def prompt_tuning(
    train_path="/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/Train_Val_Test/df_test.json",
    prompt_path="/Users/christel/Desktop/Thesis/thesis_repo/src/generator_prompt.txt",
    num_samples=20
):
    # Load train set
    train_data = load_train_data(train_path, num_samples)

    # Load or edit prompt
    system_prompt = load_prompt(prompt_path)
    print("\nCurrent system prompt:\n", system_prompt)
    print("\nYou can edit the prompt in", prompt_path, "and rerun this script to try a new prompt.")

    # Instantiate generator
    generator = ChatGPTGenerator()

    # Generate predictions and collect latency
    dataset = []
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

        dataset.append({
            "user_input": question,
            "retrieved_contexts": context,
            "response": generated_answer,
            "reference": true_answer
        })

    # Only apply the specified metrics
    metrics_list = ["answer_accuracy", "string_presence"]

    # Evaluate using the new evaluation function (now async)
    results = await evaluate_ragas_dataset(dataset, metrics_list=metrics_list)

    print("\nEvaluation Results for current prompt:")
    print(results)
    print("\nSample-wise details:")

    for i, sample in enumerate(dataset):
        print(f"Sample {i+1}:")
        print(f"  Question: {sample['user_input']}")
        print(f"  Response: {sample['response']}")
        print(f"  True Answer: {sample['reference']}")
        print("-" * 40)

if __name__ == "__main__":
    asyncio.run(prompt_tuning())