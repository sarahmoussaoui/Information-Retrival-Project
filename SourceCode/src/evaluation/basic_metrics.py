"""Precision, recall, F1, and related metrics."""


def _extract_doc_ids(retrieved):
    """
    Extract document IDs from retrieved results.
    Handles both formats:
    - List of doc_ids: [1, 2, 3] or ["1", "2", "3"]
    - List of dicts: [{"doc_id": "1", "score": 0.5}, ...]
    
    Args:
        retrieved: List of doc_ids or list of dicts with "doc_id" key
        
    Returns:
        List of doc_ids (as strings for consistency)
    """
    if not retrieved:
        return []
    
    # Check if first element is a dict
    if isinstance(retrieved[0], dict):
        return [str(item["doc_id"]) for item in retrieved]
    else:
        return [str(doc_id) for doc_id in retrieved]


def _to_set(doc_list):
    """Convert list of doc_ids to set, handling string/int conversion."""
    return set(str(doc_id) for doc_id in doc_list)


def precision(retrieved, relevant, k=40):
    """
    Precision (cours):
    nombre de documents pertinents sélectionnés
    -------------------------------------------
    nombre total de documents sélectionnés

    Ici, les documents sélectionnés = top-k
    """
    if k <= 0:
        return 0.0

    retrieved_ids = _extract_doc_ids(retrieved)[:k]
    if not retrieved_ids:
        return 0.0

    relevant_set = set(map(str, relevant))

    tp = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)

    return tp / len(retrieved_ids)


def recall(retrieved, relevant, k=40):
    """
    Recall (cours):
    nombre de documents pertinents sélectionnés
    -------------------------------------------
    nombre total de documents pertinents
    """
    if not relevant:
        return 0.0

    retrieved_ids = _extract_doc_ids(retrieved)[:k]
    relevant_set = set(map(str, relevant))

    tp = sum(1 for doc_id in retrieved_ids if doc_id in relevant_set)

    return tp / len(relevant_set)


def f1_score(precision_val, recall_val):
    """
    Calculate F1-score: 2 * (precision * recall) / (precision + recall)
    
    Args:
        precision_val: Precision value (float)
        recall_val: Recall value (float)
        
    Returns:
        float: F1-score (0.0 to 1.0)
    """
    if precision_val + recall_val == 0:
        return 0.0
    return 2 * precision_val * recall_val / (precision_val + recall_val)


def precision_at_k(retrieved, relevant, k):
    """
    Calculate Precision at K: precision considering only top K retrieved documents
    
    Args:
        retrieved: List of retrieved document IDs (or list of dicts with "doc_id")
        relevant: Set or list of relevant document IDs
        k: Number of top documents to consider
        
    Returns:
        float: Precision@K score (0.0 to 1.0)
    """
    if k <= 0:
        return 0.0
    
    retrieved_ids = _extract_doc_ids(retrieved)
    retrieved_k = retrieved_ids[:k]
    
    if len(retrieved_k) == 0:
        return 0.0
    
    relevant_set = _to_set(relevant)
    retrieved_k_set = _to_set(retrieved_k)
    
    # True positives in top K
    tp = len(retrieved_k_set & relevant_set)
    
    return tp / k


def r_precision(retrieved, relevant):
    """
    Calculate R-Precision: precision at rank R, where R = |relevant|
    This is the precision when considering only the top R retrieved documents
    
    Args:
        retrieved: List of retrieved document IDs (or list of dicts with "doc_id")
        relevant: Set or list of relevant document IDs
        
    Returns:
        float: R-Precision score (0.0 to 1.0)
    """
    R = len(relevant)
    if R == 0:
        return 0.0
    
    retrieved_ids = _extract_doc_ids(retrieved)
    retrieved_R = retrieved_ids[:R]
    
    if len(retrieved_R) == 0:
        return 0.0
    
    relevant_set = _to_set(relevant)
    retrieved_R_set = _to_set(retrieved_R)
    
    # True positives in top R
    tp = len(retrieved_R_set & relevant_set)
    
    return tp / R



"""
Test script to evaluate basic metrics on Results JSON files using MED.REL (qrels).

This script:
1. Loads all result JSON files from Results/ directory
2. Loads qrels (relevance judgments) from data/processed/parse_preprocess/qrels.json
3. Computes basic metrics (Precision, Recall, F1, P@5, P@10, R-Precision) for each query
4. Displays results and optionally saves to CSV
"""

import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add src to path to import evaluation modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# from evaluation.basic_metrics import (
#     precision, recall, f1_score, 
#     precision_at_k, r_precision
# )



def load_results_json(results_path):
    """Load a results JSON file."""
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_qrels(qrels_path):
    """Load qrels (relevance judgments) JSON file."""
    with open(qrels_path, 'r', encoding='utf-8') as f:
        qrels = json.load(f)
    # Convert string keys to int for consistency
    return {int(k): v for k, v in qrels.items()}


def evaluate_model(results_data, qrels):
    """
    Evaluate a model's results against qrels.
    
    Args:
        results_data: Dict with "model" and "queries" keys
        qrels: Dict mapping query_id to list of relevant doc_ids
        
    Returns:
        Dict mapping query_id to metrics
    """
    model_name = results_data.get("model", "Unknown")
    queries_data = results_data.get("queries", {})
    
    evaluation = {}
    
    for query_id_str, retrieved_docs in queries_data.items():
        query_id = int(query_id_str)
        
        # Get relevant documents for this query
        if query_id not in qrels:
            print(f"Warning: Query {query_id} not found in qrels, skipping...")
            continue
        
        relevant = qrels[query_id]
        
        # Compute all basic metrics
        p = precision(retrieved_docs, relevant)
        r = recall(retrieved_docs, relevant)
        f1 = f1_score(p, r)
        p5 = precision_at_k(retrieved_docs, relevant, 5)
        p10 = precision_at_k(retrieved_docs, relevant, 10)
        rp = r_precision(retrieved_docs, relevant)
        
        evaluation[query_id] = {
            "Precision": p,
            "Recall": r,
            "F1": f1,
            "P@5": p5,
            "P@10": p10,
            "R-Precision": rp
        }
    
    return evaluation


def main():
    # Paths
    project_root = Path(__file__).parent.parent.parent
    print(project_root)
    results_dir = project_root / "Results"
    qrels_path = project_root / "data" / "processed" / "parse_preprocess" / "qrels.json"
    
    # Check if paths exist
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    if not qrels_path.exists():
        print(f"Error: Qrels file not found at {qrels_path}")
        return
    
    # Load qrels
    print("Loading qrels (relevance judgments)...")
    qrels = load_qrels(qrels_path)
    print(f"Loaded relevance judgments for {len(qrels)} queries\n")
    
    # Find all JSON files in Results directory
    result_files = list(results_dir.glob("*.json"))
    
    if not result_files:
        print(f"No JSON files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files to evaluate\n")
    print("=" * 80)
    
    # Evaluate each model
    all_results = {}
    
    for result_file in sorted(result_files):
        model_name = result_file.stem
        print(f"\nEvaluating: {model_name}")
        print("-" * 80)
        
        try:
            # Load results
            results_data = load_results_json(result_file)
            
            # Evaluate
            evaluation = evaluate_model(results_data, qrels)
            
            if not evaluation:
                print("No valid evaluations computed.")
                continue
            
            # Create DataFrame for this model
            df = pd.DataFrame.from_dict(evaluation, orient='index')
            df.index.name = "Query"
            
            # Compute averages
            avg_metrics = df.mean()
            
            # Display results
            print(f"\nPer-query metrics:")
            print(df.to_string())
            
            print(f"\nAverage metrics:")
            print(f"  Precision:    {avg_metrics['Precision']:.4f}")
            print(f"  Recall:       {avg_metrics['Recall']:.4f}")
            print(f"  F1-Score:     {avg_metrics['F1']:.4f}")
            print(f"  P@5:          {avg_metrics['P@5']:.4f}")
            print(f"  P@10:         {avg_metrics['P@10']:.4f}")
            print(f"  R-Precision:  {avg_metrics['R-Precision']:.4f}")
            
            # Store for summary
            all_results[model_name] = {
                "per_query": evaluation,
                "averages": avg_metrics.to_dict()
            }
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Create summary comparison
    if all_results:
        print("\n" + "=" * 80)
        print("SUMMARY: Average Metrics Across All Models")
        print("=" * 80)
        
        summary_data = {
            model: metrics["averages"]
            for model, metrics in all_results.items()
        }
        summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
        summary_df = summary_df.sort_values('F1', ascending=False)
        
        print("\n" + summary_df.to_string())
        
        # Save per-model JSON files and one summary JSON (simple format)
        output_dir = project_root / "evaluation_results" / "basic_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write per-model JSON files
        summary_models = {}
        for model_name, metrics in all_results.items():
            # Convert per-query metric keys to the simple format expected
            per_query_converted = {}
            for q, v in metrics["per_query"].items():
                per_query_converted[str(q)] = {
                    "precision": float(v.get("Precision", 0.0)),
                    "recall": float(v.get("Recall", 0.0)),
                    "f1_score": float(v.get("F1", 0.0)),
                    "precision_at_5": float(v.get("P@5", 0.0)),
                    "precision_at_10": float(v.get("P@10", 0.0)),
                    "r_precision": float(v.get("R-Precision", 0.0))
                }

            # Overall metrics
            df_detail = pd.DataFrame.from_dict(metrics["per_query"], orient='index')
            total_q = len(df_detail)
            mean_p = float(df_detail["Precision"].mean()) if total_q > 0 else 0.0
            mean_r = float(df_detail["Recall"].mean()) if total_q > 0 else 0.0
            mean_f1 = float(df_detail["F1"].mean()) if total_q > 0 else 0.0
            mean_p5 = float(df_detail["P@5"].mean()) if total_q > 0 else 0.0
            mean_p10 = float(df_detail["P@10"].mean()) if total_q > 0 else 0.0
            mean_rp = float(df_detail["R-Precision"].mean()) if total_q > 0 else 0.0

            overall = {
                "mean_precision": mean_p,
                "mean_recall": mean_r,
                "mean_f1_score": mean_f1,
                "mean_precision_at_5": mean_p5,
                "mean_precision_at_10": mean_p10,
                "mean_r_precision": mean_rp,
                "total_queries_evaluated": total_q
            }

            model_json = {
                "model_name": model_name,
                "query_metrics": per_query_converted,
                "overall_metrics": overall
            }

            out_file = output_dir / f"{model_name}_basic_metrics.json"
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(model_json, f, indent=2, ensure_ascii=False)
            print(f"Detailed results for {model_name} saved to: {out_file}")

            summary_models[model_name] = overall

        # Write summary JSON with ranking by mean_f1_score
        ranking = sorted(summary_models.keys(), key=lambda m: summary_models[m].get("mean_f1_score", 0.0), reverse=True)
        summary_json = {
            "ranking_by_F1": ranking,
            "models": summary_models
        }
        summary_file = output_dir / "basic_metrics_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_json, f, indent=2, ensure_ascii=False)
        print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()




