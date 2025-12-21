"""UI components for Streamlit/PyQt."""

import json
from pathlib import Path
import os


def load_queries():
    """Load processed queries."""
    queries_path = Path("data/processed/parse_preprocess/queries_processed.json")
    
    if not queries_path.exists():
        return {}
    
    with open(queries_path, 'r') as f:
        return json.load(f)


def load_documents():
    """Load processed documents."""
    docs_path = Path("data/processed/parse_preprocess/docs_processed.json")
    
    if not docs_path.exists():
        return {}
    
    with open(docs_path, 'r') as f:
        return json.load(f)


def load_results(model_name):
    """Load retrieval results for a specific model."""
    results_path = Path(f"Results/{model_name}.json")
    
    if not results_path.exists():
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)


def load_metrics(model_name):
    """Load evaluation metrics for a specific model."""
    metrics_path = Path(f"evaluation_results/evaluation_results_dcg_ndcg_gain/{model_name}_metrics.json")
    
    if not metrics_path.exists():
        return None
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def get_available_models():
    """Get list of available models."""
    results_dir = Path("Results")
    
    if not results_dir.exists():
        return []
    
    models = []
    for file in results_dir.glob("*.json"):
        if file.stem not in ["evaluation_Avancée_results"]:
            models.append(file.stem)
    
    return sorted(models)


def get_query_ids(queries):
    """Get sorted list of query IDs."""
    if not queries:
        return list(range(1, 31))  # Default 30 queries
    
    return sorted([int(q) for q in queries.keys()])
