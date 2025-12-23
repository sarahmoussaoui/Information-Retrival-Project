"""UI components for Streamlit/PyQt."""

import json
from pathlib import Path
import os


def load_queries():
    """Load processed queries."""
    queries_path = Path("data/processed/parse_preprocess/queries_processed.json")
    
    if not queries_path.exists():
        print(f"Warning: {queries_path} not found")
        return {}
    
    try:
        with open(queries_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading queries: {e}")
        return {}


def load_documents():
    """Load processed documents."""
    docs_path = Path("data/processed/parse_preprocess/docs_processed.json")
    
    if not docs_path.exists():
        print(f"Warning: {docs_path} not found")
        return {}
    
    try:
        with open(docs_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading documents: {e}")
        return {}


def load_results(model_name):
    """Load retrieval results for a specific model."""
    results_path = Path(f"Results/{model_name}.json")
    
    if not results_path.exists():
        print(f"Warning: {results_path} not found")
        return None
    
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results for {model_name}: {e}")
        return None


def load_metrics(model_name):
    """Load evaluation metrics for a specific model."""
    metrics_path = Path(f"evaluation_results/evaluation_results_dcg_ndcg_gain/{model_name}_metrics.json")
    
    if not metrics_path.exists():
        print(f"Warning: {metrics_path} not found")
        return None
    
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics for {model_name}: {e}")
        return None


def get_available_models():
    """Get list of available models."""
    results_dir = Path("Results")
    
    if not results_dir.exists():
        print(f"Warning: {results_dir} not found")
        return ["BM25", "VSM_Cosine", "LM_MLE"]  # Default models
    
    try:
        models = []
        for file in results_dir.glob("*.json"):
            if file.stem not in ["evaluation_Avancée_results"]:
                models.append(file.stem)
        
        return sorted(models) if models else ["BM25", "VSM_Cosine", "LM_MLE"]
    except Exception as e:
        print(f"Error getting models: {e}")
        return ["BM25", "VSM_Cosine", "LM_MLE"]


def get_query_ids(queries):
    """Get sorted list of query IDs."""
    if not queries:
        return list(range(1, 31))  # Default 30 queries
    
    try:
        return sorted([int(q) for q in queries.keys()])
    except Exception as e:
        print(f"Error getting query IDs: {e}")
        return list(range(1, 31))
