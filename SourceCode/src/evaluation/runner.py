"""
runner.py
==========

Run all IR metrics on query results.

This script computes:
- Basic metrics: precision, recall, F1, precision@k, R-precision, average precision, reciprocal rank
- Advanced metrics: MAP, interpolated MAP, PR curves
- Gain metrics: DCG, nDCG, CG, CG@k, DCG@k, nDCG@k
- Optional: gain comparison between two systems
"""

import math
from basic_metrics import (
    precision, recall, f1_score, precision_at_k, r_precision,
    reciprocal_rank, average_precision, get_precision_recall_points,
    interpolate_precision
)
from advanced_metrics import (
    map_score, interpolated_map, get_pr_curve_data, get_interpolated_pr_curve
)
from dcg_ndcg_gain import (
    dcg_at_k, ndcg_at_k, get_relevance_vector, compare_systems_with_gain
)


def run_all(runs, qrels, k_values=(5, 10), relevance_scores=None):
    """
    Run all metrics for given query runs and qrels.
    
    Parameters
    ----------
    runs : dict
        Either {query_id: ranked_doc_ids} or {model_name: {query_id: ranked_doc_ids}}
    qrels : dict
        {query_id: relevant_doc_ids}
    k_values : tuple
        List of k values for precision@k, CG@k, DCG@k, nDCG@k
    relevance_scores : dict, optional
        {query_id: {doc_id: relevance_score}} for DCG/nDCG calculations
    
    Returns
    -------
    dict
        Nested dictionary of metrics per query (and per model if multiple models)
    """
    results = {}

    # check if we have multiple models or a single run
    multi_model = any(isinstance(v, dict) for v in runs.values())

    if multi_model:
        # multiple models
        for model_name, model_runs in runs.items():
            print(f"\nComputing metrics for model: {model_name}")
            results[model_name] = _compute_all_metrics(model_runs, qrels, k_values, relevance_scores)
    else:
        # single run
        results = _compute_all_metrics(runs, qrels, k_values, relevance_scores)

    return results


def _compute_all_metrics(runs, qrels, k_values, relevance_scores):
    """
    Helper function: compute all metrics for a single model's runs.
    """
    model_metrics = {}

    for query_id, ranked_docs in runs.items():
        if query_id not in qrels:
            model_metrics[query_id] = {}
            continue

        relevant_docs = qrels[query_id]

        # ----- Basic metrics -----
        metrics = {
            'precision': precision(ranked_docs, relevant_docs),
            'recall': recall(ranked_docs, relevant_docs),
            'f1': f1_score(ranked_docs, relevant_docs),
            'r_precision': r_precision(ranked_docs, relevant_docs),
            'reciprocal_rank': reciprocal_rank(ranked_docs, relevant_docs),
            'average_precision': average_precision(ranked_docs, relevant_docs)
        }

        # precision@k
        for k in k_values:
            metrics[f'precision@{k}'] = precision_at_k(ranked_docs, relevant_docs, k)

        # ----- Gain metrics -----
        if relevance_scores and query_id in relevance_scores:
            rel_scores = relevance_scores[query_id]

            # convert ranked list to relevance vector
            relevance_vector = [rel_scores.get(str(doc), 0) for doc in ranked_docs]

            # DCG/nDCG@default 20
            metrics['DCG@20'] = dcg_at_k(relevance_vector, k=20)
            metrics['nDCG@20'] = ndcg_at_k(relevance_vector, k=20)

            # Cumulative Gain (CG)
            metrics['CG'] = sum(relevance_vector)

            # CG@k, DCG@k, nDCG@k for specified k_values
            for k in k_values:
                metrics[f'CG@{k}'] = sum(relevance_vector[:k])
                metrics[f'DCG@{k}'] = dcg_at_k(relevance_vector[:k], k)
                metrics[f'nDCG@{k}'] = ndcg_at_k(relevance_vector[:k], k)

        # ----- PR curve -----
        # fixed: pass query_id and wrap ranked_docs/relevant_docs in dicts
        recalls, precisions = get_pr_curve_data(
            runs={query_id: ranked_docs},
            qrels={query_id: relevant_docs},
            query_id=query_id
        )
        metrics['pr_curve'] = {'recalls': recalls, 'precisions': precisions}

        # interpolated PR curve
        interp_recalls, interp_precisions = get_interpolated_pr_curve(
            {query_id: ranked_docs}, {query_id: relevant_docs}
        )
        metrics['interpolated_pr_curve'] = {'recalls': interp_recalls, 'precisions': interp_precisions}

        # save metrics for this query
        model_metrics[query_id] = metrics

    # ----- Overall metrics -----
    model_metrics['MAP'] = map_score(runs, qrels)
    model_metrics['interpolated_MAP'] = interpolated_map(runs, qrels)

    return model_metrics


# ----- Example usage -----
if __name__ == "__main__":
    # Example query runs for two models
    bm25_runs = {
        '1': ['d1', 'd2', 'd3'],
        '2': ['d3', 'd4', 'd2']
    }

    vsm_runs = {
        '1': ['d3', 'd1', 'd2'],
        '2': ['d2', 'd3', 'd4']
    }

    # Example relevant documents
    example_qrels = {
        '1': ['d1', 'd3'],
        '2': ['d2']
    }

    # Optional relevance scores for DCG/nDCG
    example_relevance_scores = {
        '1': {'d1': 3, 'd2': 0, 'd3': 2},
        '2': {'d2': 3, 'd3': 1, 'd4': 2}
    }

    # Dictionary of models (runs)
    all_model_runs = {
        'BM25': bm25_runs,
        'VSM': vsm_runs
    }

    # Run all metrics
    metrics = run_all(
        runs=all_model_runs,                 # <-- changed from model_runs to runs
        qrels=example_qrels,
        k_values=(5, 10),
        relevance_scores=example_relevance_scores
    )

    # Compare systems with gain (BM25 vs VSM)
    gain_results = compare_systems_with_gain(
        baseline_runs=bm25_runs,
        comparison_runs=vsm_runs,
        qrels=example_qrels
    )

    # Pretty-print results
    import json
    print("=== Metrics per model ===")
    print(json.dumps(metrics, indent=2))
    print("\n=== Gain comparison (BM25 vs VSM) ===")
    print(json.dumps(gain_results, indent=2))
