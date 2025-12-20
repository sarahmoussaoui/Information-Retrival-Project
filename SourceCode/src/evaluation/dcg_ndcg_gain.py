"""DCG/nDCG and gain metrics."""

import math


def dcg_at_k(relevance_scores, k=20):
    """
    calculate DCG at rank k
    formula from slides: DCG@k = rel_1 + sum(rel_i / log2(i)) for i=2 to k
    
    the first doc doesnt get discounted, rest do
    """
    if not relevance_scores:
        return 0.0
    
    # only look at top k results
    relevance_scores = relevance_scores[:k]
    
    # first document -> no discount
    dcg = float(relevance_scores[0])
    
    # rest of documents -> apply log discount
    for i in range(1, len(relevance_scores)):
        if relevance_scores[i] > 0:
            dcg += relevance_scores[i] / math.log2(i + 1)
    
    return dcg


def ndcg_at_k(relevance_scores, k=20):
    """
    normalized DCG at rank k
    nDCG@k = DCG@k / IDCG@k
    
    IDCG is the ideal DCG (best possible ranking)
    so nDCG tells us how close we are to perfect
    """
    actual_dcg = dcg_at_k(relevance_scores, k)
    
    # ideal DCG - sort relevance scores descending
    ideal_relevance = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance, k)
    
    # avoid division by zero
    if ideal_dcg == 0:
        return 0.0
    
    return actual_dcg / ideal_dcg


def calculate_gain_percentage(baseline_score, comparison_score):
    """
    calculate gain percentage between two scores
    Gain(%) = (comparison - baseline) / baseline * 100
    
    for lab5: compare systems using nDCG@20 on first 10 queries
    positive gain means comparison is better
    """
    if baseline_score == 0:
        if comparison_score > 0:
            return float('inf')  # infinit gain
        else:
            return 0.0
    
    gain = ((comparison_score - baseline_score) / baseline_score) * 100
    return gain


def get_relevance_vector(ranked_list, relevant_docs):
    """
    convert ranked list to relevance vector
    1 if doc is relevant, 0 otherwise
    
    this is what we feed to DCG/nDCG functions
    """
    relevant_set = set(str(d) for d in relevant_docs)
    relevance_vector = []
    
    for doc_id in ranked_list:
        if str(doc_id) in relevant_set:
            relevance_vector.append(1)
        else:
            relevance_vector.append(0)
    
    return relevance_vector


def evaluate_dcg_ndcg(runs, qrels, k=20):
    """
    evaluate DCG@k and nDCG@k for all queries
    returns a dict with results for each query
    """
    results = {}
    
    for query_id in runs:
        if query_id in qrels:
            ranked_list = runs[query_id]
            relevant_docs = qrels[query_id]
            
            # convert to relevance vector (1s and 0s)
            relevance_vector = get_relevance_vector(ranked_list, relevant_docs)
            
            # calculate metrics
            dcg = dcg_at_k(relevance_vector, k)
            ndcg = ndcg_at_k(relevance_vector, k)
            
            results[query_id] = {
                f'dcg@{k}': dcg,
                f'ndcg@{k}': ndcg
            }
    
    return results


def calculate_mean_dcg_ndcg(dcg_ndcg_results, k=20):
    """
    calculate mean DCG@k and nDCG@k across all queries
    just takes the average
    """
    if not dcg_ndcg_results:
        return 0.0, 0.0
    
    dcg_key = f'dcg@{k}'
    ndcg_key = f'ndcg@{k}'
    
    dcg_values = [r[dcg_key] for r in dcg_ndcg_results.values() if dcg_key in r]
    ndcg_values = [r[ndcg_key] for r in dcg_ndcg_results.values() if ndcg_key in r]
    
    mean_dcg = sum(dcg_values) / len(dcg_values) if dcg_values else 0.0
    mean_ndcg = sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0
    
    return mean_dcg, mean_ndcg


def compare_systems_with_gain(baseline_runs, comparison_runs, qrels, k=20, num_queries=10):
    """
    compare two systems using nDCG@k on first num_queries queries
    calculates gain percentage for each query
    
    for lab5: num_queries=10 (I1 to I10), k=20
    """
    gains = {}
    
    # check first 10 queries
    for i in range(1, num_queries + 1):
        query_id = str(i)
        
        if query_id in baseline_runs and query_id in comparison_runs and query_id in qrels:
            # get relevance vectors for both systems
            baseline_rel = get_relevance_vector(baseline_runs[query_id], qrels[query_id])
            comparison_rel = get_relevance_vector(comparison_runs[query_id], qrels[query_id])
            
            # calculate nDCG@k for both
            baseline_ndcg = ndcg_at_k(baseline_rel, k)
            comparison_ndcg = ndcg_at_k(comparison_rel, k)
            
            # calculate gain
            gain = calculate_gain_percentage(baseline_ndcg, comparison_ndcg)
            gains[query_id] = {
                'baseline_ndcg': baseline_ndcg,
                'comparison_ndcg': comparison_ndcg,
                'gain_percent': gain
            }
    
    # calculate mean gain (skip infinite values)
    valid_gains = [g['gain_percent'] for g in gains.values() 
                   if g['gain_percent'] != float('inf') and not math.isnan(g['gain_percent'])]
    
    mean_gain = sum(valid_gains) / len(valid_gains) if valid_gains else 0.0
    
    return {
        'per_query_gains': gains,
        'mean_gain_percent': mean_gain,
        'num_queries_compared': len(gains)
    }
