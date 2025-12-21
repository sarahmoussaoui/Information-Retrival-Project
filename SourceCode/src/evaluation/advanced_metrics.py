# MAP, interpolated MAP, PR curves

from basic_metrics import average_precision, get_precision_recall_points, interpolate_precision


def map_score(runs, qrels):
    """
    calculate mean average precision (MAP)
    mean of average precision scores across all queries
    """
    if not runs or not qrels:
        return 0.0
    
    ap_scores = []
    
    for query_id in runs:
        if query_id in qrels:
            ranked_docs = runs[query_id]
            relevant_docs = qrels[query_id]
            ap = average_precision(ranked_docs, relevant_docs)
            ap_scores.append(ap)
    
    if not ap_scores:
        return 0.0
    
    return sum(ap_scores) / len(ap_scores)


def interpolated_map(runs, qrels, recall_levels=None):
    """
    calculate interpolated MAP
    average interpolated precision across queries at standard recall levels
    returns dict {recall_level: precision}
    """
    if recall_levels is None:
        recall_levels = [i * 0.1 for i in range(11)]
    
    if not runs or not qrels:
        return {r: 0.0 for r in recall_levels}
    
    # store interpolated precisions per recall level
    interpolated_precisions = {r: [] for r in recall_levels}
    
    for query_id in runs:
        if query_id in qrels:
            ranked_docs = runs[query_id]
            relevant_docs = qrels[query_id]
            
            pr_points = get_precision_recall_points(ranked_docs, relevant_docs)
            interp_prec = interpolate_precision(pr_points, recall_levels)
            
            for r in recall_levels:
                interpolated_precisions[r].append(interp_prec[r])
    
    # average across queries
    mean_interp_map = {}
    for r in recall_levels:
        values = interpolated_precisions[r]
        mean_interp_map[r] = sum(values) / len(values) if values else 0.0
    
    return mean_interp_map


def get_pr_curve_data(runs, qrels, query_id):
    """
    get precision-recall points for one query
    returns two lists: recalls and precisions
    """
    if query_id not in runs or query_id not in qrels:
        return [], []
    
    ranked_docs = runs[query_id]
    relevant_docs = qrels[query_id]
    
    pr_points = get_precision_recall_points(ranked_docs, relevant_docs)
    
    if not pr_points:
        return [], []
    
    recalls = [point[0] for point in pr_points]
    precisions = [point[1] for point in pr_points]
    
    return recalls, precisions


def get_interpolated_pr_curve(runs, qrels, query_id=None):
    """
    get interpolated precision-recall curve
    if query_id is given, return curve for that query
    otherwise return average across all queries
    """
    recall_levels = [i * 0.1 for i in range(11)]
    
    if query_id is not None:
        if query_id not in runs or query_id not in qrels:
            return recall_levels, [0.0] * len(recall_levels)
        
        pr_points = get_precision_recall_points(runs[query_id], qrels[query_id])
        interp_prec = interpolate_precision(pr_points, recall_levels)
        precisions = [interp_prec[r] for r in recall_levels]
        return recall_levels, precisions
    
    else:
        interp_map_values = interpolated_map(runs, qrels, recall_levels)
        precisions = [interp_map_values[r] for r in recall_levels]
        return recall_levels, precisions
