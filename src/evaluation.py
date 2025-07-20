import time
from typing import List, Dict, Any

from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_accuracy,
    string_presence,
    nvidia_entity_recall,  # ← example real import
)
from ragas import set_llm
from ragas.llms import OpenAI, Vllm

# Map metric names to functions
RAGAS_METRICS = {
    "context_precision": context_precision,
    "context_recall": context_recall,
    "faithfulness": faithfulness,
    "answer_accuracy": answer_accuracy,
    "string_presence": string_presence,
    "nvidia_entity_recall": nvidia_entity_recall,
}

def evaluate_rag(
    predictions: List[Dict[str, Any]],
    references: List[Dict[str, Any]],
    metrics: List[str],
    retriever_latency: List[float] = None,
    retriever_cost: List[float] = None,
    llm: Any = None,
) -> Dict[str, Any]:
    if llm is not None:
        set_llm(llm)
        print(f"[RAGAS] Using LLM: {llm}")
    else:
        print("[RAGAS] Using default LLM")

    results: Dict[str, Any] = {}

    # Compute RAGAS metrics
    for name in metrics:
        if name in RAGAS_METRICS:
            func = RAGAS_METRICS[name]
            try:
                results[name] = func(predictions, references)
            except Exception as e:
                results[name] = f"Error: {e}"
        elif name == "latency":
            if retriever_latency:
                results["latency_mean"] = sum(retriever_latency) / len(retriever_latency)
                results["latency_max"] = max(retriever_latency)
                results["latency_min"] = min(retriever_latency)
            else:
                results.update(latency_mean=None, latency_max=None, latency_min=None)
        elif name == "cost":
            if retriever_cost:
                results["cost_total"] = sum(retriever_cost)
                results["cost_mean"] = sum(retriever_cost) / len(retriever_cost)
            else:
                results.update(cost_total=None, cost_mean=None)
        else:
            raise ValueError(f"Unknown metric: {name}")

    return results

# Example usage
if __name__ == "__main__":
    # Dummy data for demonstration
    predictions = [
        {"answer": "42", "contexts": ["The answer is 42."]},
        {"answer": "blue", "contexts": ["The sky is blue."]},
    ]
    references = [
        {"answer": "42", "contexts": ["The answer is 42."]},
        {"answer": "blue", "contexts": ["The sky is blue."]},
    ]
    metrics = [
        "context_precision",
        "context_recall",
        "faithfulness",
        "answer_accuracy",
        "string_presence",
        "nvidia_metrics",
        "latency",
        "cost",
    ]
    retriever_latency = [0.12, 0.15]
    retriever_cost = [0.01, 0.01]

    # Example: Use OpenAI GPT-3.5 for evaluation
    openai_llm = OpenAI(model="gpt-3.5-turbo", api_key="sk-...your-key...")
    results = evaluate_rag(predictions, references, metrics, retriever_latency, retriever_cost, llm=openai_llm)
    for k, v in results.items():
        print(f"{k}: {v}")

    # Example: Use local vllm for evaluation (uncomment to use)
    # vllm_llm = Vllm(base_url="http://localhost:8000/v1", model="Fin-R1")
    # results = evaluate_rag(predictions, references, metrics, retriever_latency, retriever_cost, llm=vllm_llm)
    # for k, v in results.items():
    #     print(f"{k}: {v}") 