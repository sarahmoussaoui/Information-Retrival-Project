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

from evaluation.basic_metrics import (
    precision, recall, f1_score, 
    precision_at_k, r_precision
)



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
        
        # Save summary to CSV
        output_dir = Path(__file__).parent / "outputs" / "basic_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        summary_csv = output_dir / "basic_metrics_summary.csv"
        summary_df.to_csv(summary_csv)
        print(f"\nSummary saved to: {summary_csv}")
        
        # Save detailed results per model
        for model_name, metrics in all_results.items():
            detail_df = pd.DataFrame.from_dict(metrics["per_query"], orient='index')
            detail_df.index.name = "Query"
            detail_csv = output_dir / f"{model_name}_basic_metrics.csv"
            detail_df.to_csv(detail_csv)
            print(f"Detailed results for {model_name} saved to: {detail_csv}")


if __name__ == "__main__":
    main()

