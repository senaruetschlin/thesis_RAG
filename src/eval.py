from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Tuple
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# === Retrieval metrics for FinQA and ConFinQA with reference present ===
def context_precision_with_reference(
    dataset: List[Dict],
    embedder,
    top_k_context_key: str = "retrieved_contexts",
    gold_context_key: str = "reference_contexts",
    method: str = "embedding",  # "exact", "embedding", or "both"
    sim_threshold: float = 0.85
) -> pd.DataFrame:
    """
    Evaluates context precision via exact match or embedding-based similarity.

    Args:
        dataset: List of QA samples with user_input, retrieved_contexts, reference_contexts
        embedder: an object with .embed_documents() method
        method: "exact", "embedding", or "both"
        sim_threshold: similarity threshold for embedding-based match

    Returns:
        pd.DataFrame with precision scores and detailed diagnostics
    """
    results = []

    for sample in tqdm(dataset):
        question = sample.get("user_input", "N/A")
        retrieved = sample.get(top_k_context_key, [])
        gold = sample.get(gold_context_key, [])

        retrieved = [r.lower().strip() for r in retrieved if isinstance(r, str)]
        gold = [g.lower().strip() for g in gold if isinstance(g, str)]

        em_hits = 0
        emb_hits = 0

        # === Exact Match ===
        if method in {"exact", "both"}:
            em_hits = sum(any(g in r for g in gold) for r in retrieved)

        # === Embedding Similarity ===
        if method in {"embedding", "both"} and gold and retrieved:
            try:
                gold_emb = np.array(embedder.embed_documents(gold)).astype("float32")
                retrieved_emb = np.array(embedder.embed_documents(retrieved)).astype("float32")
                sim_matrix = cosine_similarity(retrieved_emb, gold_emb)
                emb_hits = sum(np.any(sim_row >= sim_threshold) for sim_row in sim_matrix)
            except Exception as e:
                print(f"Embedding error for question: {question}\n{e}")

        # Pick which metric to use for precision
        total = len(retrieved)
        precision_em = em_hits / total if total else 0
        precision_emb = emb_hits / total if total else 0

        results.append({
            "question": question,
            "total_retrieved": total,
            "precision_em": precision_em,
            "precision_emb": precision_emb,
            "irrelevant_em": [r for r in retrieved if not any(g in r for g in gold)],
            "irrelevant_emb": [r for idx, r in enumerate(retrieved)
                               if method in {"embedding", "both"} and
                               (idx >= len(sim_matrix) or not np.any(sim_matrix[idx] >= sim_threshold))]
        })

    return pd.DataFrame(results)
 
def context_recall_with_reference(
    dataset: List[Dict],
    embedder,
    top_k_context_key: str = "retrieved_contexts",
    gold_context_key: str = "reference_contexts",
    method: str = "embedding",  # "exact", "embedding", or "both"
    sim_threshold: float = 0.85
) -> pd.DataFrame:
    """
    Calculates the per-gold context recall. Output shows what fraction of the gold/reference contexts were matched by at least one of the retrieved results.
    Evaluates context recall via exact match or embedding-based similarity.
 
    Args:
        dataset: List of QA samples with user_input, retrieved_contexts, reference_contexts
        embedder: an object with .embed_documents() method (e.g., OpenAIEmbedder or SentenceTransformer)
        method: "exact", "embedding", or "both"
        sim_threshold: similarity threshold for embedding-based match
 
    Returns:
        pd.DataFrame with recall scores and detailed diagnostics
    """
    results = []
 
    for sample in tqdm(dataset):
        question = sample.get("user_input", "N/A")
        retrieved = sample.get(top_k_context_key, [])
        gold = sample.get(gold_context_key, [])
 
        retrieved = [r.lower().strip() for r in retrieved if isinstance(r, str)]
        gold = [g.lower().strip() for g in gold if isinstance(g, str)]
 
        em_hits = 0
        emb_hits = 0
 
        # === Exact Match ===
        if method in {"exact", "both"}:
            em_hits = sum(any(g in r for r in retrieved) for g in gold)
 
        # === Embedding Similarity ===
        if method in {"embedding", "both"} and gold and retrieved:
            try:
                gold_emb = np.array(embedder.embed_documents(gold)).astype("float32")
                retrieved_emb = np.array(embedder.embed_documents(retrieved)).astype("float32")
                sim_matrix = cosine_similarity(gold_emb, retrieved_emb)
                emb_hits = sum(np.any(sim_row >= sim_threshold) for sim_row in sim_matrix)
            except Exception as e:
                print(f"Embedding error for question: {question}\n{e}")
 
        # Pick which metric to use for recall
        total = len(gold)
        recall_em = em_hits / total if total else 0
        recall_emb = emb_hits / total if total else 0
 
        results.append({
            "question": question,
            "total_gold": total,
            "recall_em": recall_em,
            "recall_emb": recall_emb,
            "missing_em": [g for g in gold if not any(g in r for r in retrieved)],
            "missing_emb": [g for idx, g in enumerate(gold)
                            if method in {"embedding", "both"} and
                            (idx >= len(sim_matrix) or not np.any(sim_matrix[idx] >= sim_threshold))]
        })
 
    return pd.DataFrame(results)
 
def compute_f1_from_precision_recall_dfs(
    precision_df: pd.DataFrame,
    recall_df: pd.DataFrame,
    precision_col: str = "precision_emb",
    recall_col: str = "recall_emb"
) -> pd.DataFrame:
    """
    Combines precision and recall DataFrames to compute F1 score per sample.

    Args:
        precision_df: Output of context_precision_with_reference()
        recall_df: Output of context_recall_with_reference()
        precision_col: Column name for precision score
        recall_col: Column name for recall score

    Returns:
        A merged DataFrame with F1 score added
    """
    # Join on 'question'
    merged = pd.merge(precision_df, recall_df, on="question", suffixes=("_prec", "_rec"))

    def safe_f1(p, r):
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    merged["f1"] = merged.apply(lambda row: safe_f1(row[precision_col], row[recall_col]), axis=1)

    return merged
 
def context_entity_recall(
    dataset: List[Dict],
    nlp,
    top_k_context_key: str = "retrieved_contexts",
    gold_context_key: str = "reference_contexts"
) -> pd.DataFrame:
    """
    Computes entity-level recall: proportion of gold named entities found in retrieved contexts.
 
    Args:
        dataset: List of samples with user_input, retrieved_contexts, and reference_contexts
        nlp: spaCy language model for NER
        top_k_context_key: Key for retrieved contexts
        gold_context_key: Key for reference (gold) contexts
 
    Returns:
        pd.DataFrame with entity recall metrics and diagnostics
    """
    results = []
 
    for sample in tqdm(dataset):
        question = sample.get("user_input", "N/A")
        retrieved = sample.get(top_k_context_key, [])
        gold = sample.get(gold_context_key, [])
 
        # Join all contexts into one string each
        retrieved_text = " ".join([r for r in retrieved if isinstance(r, str)])
        gold_text = " ".join([g for g in gold if isinstance(g, str)])
 
        # Run NER
        retrieved_doc = nlp(retrieved_text)
        gold_doc = nlp(gold_text)
 
        # Extract unique entities (as strings)
        retrieved_ents = set(ent.text.strip().lower() for ent in retrieved_doc.ents)
        gold_ents = set(ent.text.strip().lower() for ent in gold_doc.ents)
 
        # Calculate overlap
        common_ents = retrieved_ents & gold_ents
        total_gold_ents = len(gold_ents)
        recall = len(common_ents) / total_gold_ents if total_gold_ents else 0
 
        results.append({
            "question": question,
            "retrieved_entities": list(retrieved_ents),
            "gold_entities": list(gold_ents),
            "common_entities": list(common_ents),
            "num_common_entities": len(common_ents),
            "total_gold_entities": total_gold_ents,
            "entity_recall": recall
        })
 
    return pd.DataFrame(results)
 
def compute_ndcg(
    dataset: List[Dict],
    embedder = None,
    top_k_context_key: str = "retrieved_contexts",
    gold_context_key: str = "reference_contexts",
    method: str = "embedding",  # "exact" or "embedding"
    sim_threshold: float = 0.85,
    k: int = 10
) -> pd.DataFrame:
    """
    Compute nDCG@k for retrieved contexts.
 
    Returns:
        DataFrame with nDCG@k per sample and average
    """
    results = []
 
    for sample in tqdm(dataset):
        question = sample.get("user_input", "N/A")
        retrieved = sample.get(top_k_context_key, [])[:k]
        gold = sample.get(gold_context_key, [])
 
        retrieved = [r.lower().strip() for r in retrieved if isinstance(r, str)]
        gold = [g.lower().strip() for g in gold if isinstance(g, str)]
 
        relevance = [0] * len(retrieved)  # Relevance at each rank
 
        if method == "exact":
            for i, r in enumerate(retrieved):
                if any(g in r for g in gold):
                    relevance[i] = 1
 
        elif method == "embedding" and embedder and gold and retrieved:
            try:
                gold_emb = np.array(embedder.embed_documents(gold)).astype("float32")
                retrieved_emb = np.array(embedder.embed_documents(retrieved)).astype("float32")
                sim_matrix = cosine_similarity(retrieved_emb, gold_emb)
                for i, sim_row in enumerate(sim_matrix):
                    if np.any(sim_row >= sim_threshold):
                        relevance[i] = 1
            except Exception as e:
                print(f"Embedding error for question: {question}\n{e}")
 
        # === Compute DCG and nDCG ===
        dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(relevance))  # +2 because log2(i+1) with 0-indexing
        ideal_relevance = sorted(relevance, reverse=True)
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevance))
 
        ndcg = dcg / idcg if idcg > 0 else 0.0
 
        results.append({
            "question": question,
            "dcg": dcg,
            "idcg": idcg,
            "ndcg@{}".format(k): ndcg
        })
 
    df = pd.DataFrame(results)
    df["mean_ndcg"] = df["ndcg@{}".format(k)].mean()
    return df

# === Retrieval metrics for FinDER without reference present ===
def proxy_retrieval_accuracy(faithfulness: float, accuracy: float) -> float:
    if faithfulness < 0.8 or accuracy < 0.8:
        return 0.0  # not enough confidence
    # Combine as minimum (conservative)
    return min(faithfulness, accuracy)


# === Generator metrics ===
def evaluate_faithfulness(response: str, context: str, model: str = "gpt-4o") -> Tuple[float, str]:
    """
    Uses an LLM to evaluate the faithfulness of a response with respect to the context.
    
    Args:
        response (str): The generated answer or response.
        context (str): The retrieved or reference context (e.g., chunks from your KG).
        model (str): OpenAI model to use (default is 'gpt-4o').
        
    Returns:
        Tuple[float, str]: Faithfulness score (0.0 to 1.0), and justification or explanation.
    """

    system_prompt = (
        "You are an evaluator that checks factual consistency between a generated answer "
        "and its source context. A response is considered *faithful* if all of its claims "
        "can be supported by the context. Return a JSON with a numeric score (0.0 to 1.0) "
        "and a brief justification."
    )

    user_prompt = f"""
[Context]
{context}

[Response]
{response}

Does the response faithfully reflect the context? Only use information supported by the context to judge this.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # Try to parse the model's output
    import json
    try:
        output_text = response.choices[0].message.content.strip()
        parsed = json.loads(output_text)
        score = float(parsed.get("score", 0.0))
        explanation = parsed.get("explanation", output_text)
        return score, explanation
    except Exception as e:
        return 0.0, f"Failed to parse LLM output: {str(e)}\nRaw output:\n{output_text}"
 

def answer_accuracy(dataset, generator_output_key="generated_answer", gold_key="reference_answer", question_key="question", judge_llm=None):
    """
    Evaluate answer accuracy using an LLM-as-a-judge approach.
    """
    results = []

    for sample in tqdm(dataset):
        generated = sample.get(generator_output_key, "").strip()
        gold = sample.get(gold_key, "").strip()
        question = sample.get(question_key, "").strip()

        if not generated or not gold:
            results.append({"question": question, "correct": False})
            continue

        # Format the prompt
        prompt = f"""You are evaluating the correctness of an answer.

Question:
{question}

Gold reference answer:
{gold}

Generated answer:
{generated}

Does the generated answer correctly and sufficiently answer the question, in a way that matches the gold reference? Answer with "Yes" or "No" only."""

        # Call your LLM (e.g., judge_llm.predict or .invoke)
        try:
            response = judge_llm.predict(prompt).strip().lower()
            correct = "yes" in response
        except Exception as e:
            print(f"Evaluation failed for question: {question}\n{e}")
            correct = False

        results.append({"question": question, "correct": correct})

    df = pd.DataFrame(results)
    df["accuracy"] = df["correct"].astype(int)
    df["mean_accuracy"] = df["accuracy"].mean()
    return df



def evaluate_rag_model(
    dataset: List[Dict],
    embedder,
    nlp,
    judge_llm,
    k: int = 10,
    sim_threshold: float = 0.85
) -> Dict[str, pd.DataFrame]:
    """
    Evaluates a RAG model across different sources (FinQA, ConvFinQA, FinDER).

    Returns:
        A dictionary of evaluation results as DataFrames, grouped by metric type.
    """
    # Separate dataset by source
    has_reference = [s for s in dataset if s.get("source") in {"FinQA", "ConvFinQA"}]
    no_reference  = [s for s in dataset if s.get("source") == "FinDER"]

    results = {}

    # === Retrieval metrics with reference (FinQA, ConvFinQA) ===
    if has_reference:
        print("Evaluating FinQA/ConvFinQA samples...")
        prec_df = context_precision_with_reference(has_reference, embedder, method="embedding", sim_threshold=sim_threshold)
        rec_df = context_recall_with_reference(has_reference, embedder, method="embedding", sim_threshold=sim_threshold)
        f1_df   = compute_f1_from_precision_recall_dfs(prec_df, rec_df)
        ent_df  = context_entity_recall(has_reference, nlp)
        ndcg_df = compute_ndcg(has_reference, embedder=embedder, k=k)

        results["precision"] = prec_df
        results["recall"] = rec_df
        results["f1"] = f1_df
        results["entity_recall"] = ent_df
        results["ndcg"] = ndcg_df

    # === Retrieval metric without reference (FinDER) ===
    if no_reference:
        print("Evaluating FinDER samples using proxy retrieval accuracy...")

        proxy_rows = []
        for sample in tqdm(no_reference):
            response = sample.get("generated_answer", "")
            context = " ".join(sample.get("retrieved_contexts", []))
            faithfulness_score, _ = evaluate_faithfulness(response, context)
            sample["faithfulness"] = faithfulness_score

            # Use LLM-as-judge accuracy
            acc_df = answer_accuracy([sample], judge_llm=judge_llm)
            acc = acc_df["accuracy"].iloc[0]

            proxy_acc = proxy_retrieval_accuracy(faithfulness_score, acc)
            proxy_rows.append({
                "question": sample.get("user_input", ""),
                "faithfulness": faithfulness_score,
                "accuracy": acc,
                "proxy_retrieval_accuracy": proxy_acc
            })

        results["finder_proxy_accuracy"] = pd.DataFrame(proxy_rows)

    # === Generator accuracy across all datasets ===
    print("Evaluating generated answer accuracy...")
    accuracy_df = answer_accuracy(dataset, judge_llm=judge_llm)
    results["answer_accuracy"] = accuracy_df

    return results
