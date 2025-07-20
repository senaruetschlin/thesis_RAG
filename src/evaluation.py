import asyncio
from typing import List, Dict, Any, Optional

from ragas import EvaluationDataset, evaluate
import os
from dotenv import load_dotenv

async def evaluate_ragas_dataset(
    dataset: List[Dict[str, Any]],
    metrics_list: Optional[List[str]] = None,
    llm_model: str = "gpt-4o",
    llm_type: str = "openai",  # 'openai' or 'vllm'
    vllm_base_url: str = "http://localhost:8000/v1"
):
    """
    Evaluate a dataset using RAGAS with the new API.

    Args:
        dataset: List of dicts with keys: user_input, retrieved_contexts, response, reference
        metrics_list: List of metric names to compute (see available_metrics below). If None, all are used.
        llm_model: Model name (e.g., 'gpt-4o', 'gpt-3.5-turbo', 'Fin-R1')
        llm_type: 'openai' (default) or 'vllm'
        vllm_base_url: Base URL for vllm server (if using vllm)
    Returns:
        Dictionary of metric results
    """
    # Metric mapping
    from ragas.metrics import (
        LLMContextPrecisionWithReference,
        NonLLMContextPrecisionWithReference,
        LLMContextRecall,
        NonLLMContextRecall,
        ContextEntityRecall,
        FaithfulnesswithHHEM,
        AnswerAccuracy,
        StringPresence,
    )
    available_metrics = {
        "context_precision_llm": LLMContextPrecisionWithReference,
        "context_precision_nonllm": NonLLMContextPrecisionWithReference,
        "context_recall_llm": LLMContextRecall,
        "context_recall_nonllm": NonLLMContextRecall,
        "context_entity_recall": ContextEntityRecall,
        "faithfulness": FaithfulnesswithHHEM,
        "answer_accuracy": AnswerAccuracy,
        "string_presence": StringPresence,
    }
    # If no metrics_list, use all
    if metrics_list is None:
        metrics_list = list(available_metrics.keys())
    # Instantiate metrics
    metrics = [available_metrics[name]() for name in metrics_list if name in available_metrics]
    if not metrics:
        raise ValueError("No valid metrics selected.")

    load_dotenv()
    if llm_type == "openai":
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in .env file.")
        llm = ChatOpenAI(model=llm_model, api_key=OPENAI_API_KEY)
        evaluator_llm = LangchainLLMWrapper(llm)
    elif llm_type == "vllm":
        from langchain_community.llms import VLLMOpenAI
        from ragas.llms import LangchainLLMWrapper
        llm = VLLMOpenAI(model=llm_model, base_url=vllm_base_url)
        evaluator_llm = LangchainLLMWrapper(llm)
    else:
        raise ValueError(f"Unknown llm_type: {llm_type}")

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    return evaluate(
        dataset=evaluation_dataset,
        metrics=metrics,
        llm=evaluator_llm
    )

# Example usage
if __name__ == "__main__":
    import asyncio
    # Example dataset
    dataset = [
        {
            "user_input": "Who introduced the theory of relativity?",
            "retrieved_contexts": ["Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity."],
            "response": "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity.",
            "reference": "Albert Einstein proposed the theory of relativity, which transformed our understanding of time, space, and gravity."
        },
        {
            "user_input": "Who was the first computer programmer?",
            "retrieved_contexts": ["Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."],
            "response": "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine.",
            "reference": "Ada Lovelace is regarded as the first computer programmer for her work on Charles Babbage's early mechanical computer, the Analytical Engine."
        }
    ]
    # Use all metrics by default
    results = asyncio.run(evaluate_ragas_dataset(dataset))
    print(results)
    # Or select specific metrics
    results2 = asyncio.run(evaluate_ragas_dataset(dataset, metrics_list=["context_precision_llm", "faithfulness"]))
    print(results2) 