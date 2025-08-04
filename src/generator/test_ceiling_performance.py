import json
import time
import asyncio
from generator.generator import ChatGPTGenerator
from evaluation import evaluate_ragas_dataset

def load_train_data(train_path, num_samples=None):
    with open(train_path) as f:
        train_data = json.load(f)
    if num_samples:
        train_data = train_data[:num_samples]
    return train_data

def get_gold_context(sample):
    """
    Return the gold context directly from the 'gold_context' column.
    Considers only FinQA and ConvFinQA examples as FinDER does not provide the best context.
    """
    gold_context = sample.get("gold_context", "")
    if not gold_context:
        print(f"Warning: No gold context found for question: {sample.get('question', 'Unknown')}")
    return gold_context if isinstance(gold_context, str) else str(gold_context)

async def test_ceiling_performance(
    train_path="/Users/alex/Documents/Data Science Master/thesis_RAG/notebooks/filtered_gold_eval_dataset.json",
    num_samples=10
):
    """
    Test ceiling performance using gold context, limited to FinQA-only samples.
    """
    # Load and filter dataset
    train_data = load_train_data(train_path)
    train_data = [s for s in train_data if s.get("source") == "FinQA" or s.get("source") == "ConvFinQA"]

    # Apply sample limit *after* filtering
    if num_samples:
        train_data = train_data[num_samples:num_samples+10]

    # Instantiate generator (prompts now handled internally)
    generator = ChatGPTGenerator()

    dataset = []
    print("\nProcessing samples...")

    for i, sample in enumerate(train_data, 1):
        question = sample["question"]
        true_answer = sample["answer"]
        source = sample.get("source", "Unknown")

        relevant_context = get_gold_context(sample)
        

        start = time.time()
        generated_answer = generator.generate(
            question=question,
            retrieved_docs=[relevant_context],
        )
        end = time.time()

        dataset.append({
            "user_input": question,
            "retrieved_contexts": [relevant_context],
            "response": generated_answer,
            "reference": true_answer
        })

        # Print progress
        print(f"\nSample {i}:")
        print(f"Source: {source}")
        print(f"Question: {question}")
        print(f"Gold Context: {relevant_context}")
        print(f"Generated Answer: {generated_answer}")
        print(f"True Answer: {true_answer}")
        print("-" * 80)

    # Evaluate with RAGAS
    metrics_list = ["answer_accuracy", "string_presence"]
    print("\nEvaluating ceiling performance...")
    results = await evaluate_ragas_dataset(dataset, metrics_list=metrics_list)

    print("\nCeiling Performance Results:")
    print(results)

    # Additional stats
    total_samples = len(dataset)
    exact_matches = sum(1 for sample in dataset if sample['response'].strip() == sample['reference'].strip())

    print("\nAdditional Statistics:")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Matches: {exact_matches}")
    print(f"Exact Match Rate: {exact_matches / total_samples:.2%}")

if __name__ == "__main__":
    asyncio.run(test_ceiling_performance())