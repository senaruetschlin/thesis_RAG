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

def get_relevant_context(sample):
    """
    Extract only the relevant context using gold_text_id and gold_table_row.
    """
    relevant_contexts = []
    
    # Handle text contexts from 'gold_text_id'
    if 'gold_text_id' in sample and sample.get('gold_text_id'):
        text_ids = sample['gold_text_id']
        context_list = sample.get('context_text', [])

        if not isinstance(context_list, list):
             context_list = [str(context_list)]

        if isinstance(text_ids, str):
            text_ids = [text_ids]

        for text_id in text_ids:
            idx = -1
            if text_id.startswith('T'):
                try:
                    idx = int(text_id.replace('T', '')) - 1
                except (ValueError, IndexError):
                    idx = -1
            elif text_id.startswith('ref_'):
                try:
                    idx = int(text_id.replace('ref_', ''))
                except (ValueError, IndexError):
                    idx = -1
            
            if 0 <= idx < len(context_list):
                relevant_contexts.append(context_list[idx])

    # Handle table contexts from 'gold_table_row'
    if 'gold_table_row' in sample and sample.get('gold_table_row'):
        row_indices = sample['gold_table_row']
        table_list = sample.get('context_table', [])

        if not isinstance(table_list, list):
             table_list = []

        if isinstance(row_indices, int):
             row_indices = [row_indices]

        for row_idx in row_indices:
             if isinstance(row_idx, int) and 0 <= row_idx < len(table_list):
                  row_str = " | ".join(map(str, table_list[row_idx]))
                  relevant_contexts.append(f"Table Row {row_idx}: {row_str}")


    if not relevant_contexts:
        print(f"Warning: No relevant context found for question: {sample.get('question', 'Unknown')}")
        print(f"  gold_text_id: {sample.get('gold_text_id')}")
        print(f"  gold_table_row: {sample.get('gold_table_row')}")
        print(f"  context_text len: {len(sample.get('context_text', [])) if isinstance(sample.get('context_text', list)) else 'N/A'}")
        print(f"  context_table len: {len(sample.get('context_table', [])) if isinstance(sample.get('context_table', list)) else 'N/A'}")
        return ""
    
    return "\n\n".join(relevant_contexts)

async def test_ceiling_performance(
    train_path="/Users/christel/Desktop/Thesis/thesis_repo/data/data_processed/Train_Val_Test/df_test.json",
    prompt_path="/Users/christel/Desktop/Thesis/thesis_repo/src/generator_prompt.txt",
    num_samples=10
):
    """
    Test the ceiling performance of the generator by using only relevant context.
    """
    # Load train set
    train_data = load_train_data(train_path, num_samples)

    # Load prompt
    system_prompt = load_prompt(prompt_path)
    print("\nCurrent system prompt:\n", system_prompt)

    # Instantiate generator
    generator = ChatGPTGenerator()

    # Generate predictions with only relevant context
    dataset = []
    print("\nProcessing samples...")
    
    for i, sample in enumerate(train_data, 1):
        question = sample["question"]
        true_answer = sample["answer"]

        # Get only relevant context using gold annotations
        relevant_context = get_relevant_context(sample)
        
        # Generate answer with relevant context
        start = time.time()
        generated_answer = generator.generate(
            question=question,
            retrieved_docs=relevant_context,
            system_prompt=system_prompt
        )
        end = time.time()

        dataset.append({
            "user_input": question,
            "retrieved_contexts": relevant_context,
            "response": generated_answer,
            "reference": true_answer
        })

        # Print progress
        print(f"\nSample {i}:")
        print(f"Question: {question}")
        print(f"Relevant Context: {relevant_context}")
        print(f"Generated Answer: {generated_answer}")
        print(f"True Answer: {true_answer}")
        print("-" * 80)

    # Evaluate using specified metrics
    metrics_list = ["answer_accuracy", "string_presence"]
    print("\nEvaluating ceiling performance...")
    results = await evaluate_ragas_dataset(dataset, metrics_list=metrics_list)
    
    print("\nCeiling Performance Results:")
    print(results)
    
    # Calculate and print additional statistics
    total_samples = len(dataset)
    exact_matches = sum(1 for sample in dataset if sample['response'].strip() == sample['reference'].strip())
    
    print("\nAdditional Statistics:")
    print(f"Total Samples: {total_samples}")
    print(f"Exact Matches: {exact_matches}")
    print(f"Exact Match Rate: {exact_matches/total_samples:.2%}")

if __name__ == "__main__":
    asyncio.run(test_ceiling_performance()) 