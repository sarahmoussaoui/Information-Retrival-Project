"""Precision, recall, F1, and related metrics."""

def precision(retrieved, relevant):
    """
    calculate precision
    basically how many of the retrieved docs are actually relevant
    """
    if not retrieved:
        return 0.0
    
    # convert to sets for easier comparison
    retrieved_set = set(str(d) for d in retrieved)
    relevant_set = set(str(d) for d in relevant)
    
    relevant_retrieved = len(retrieved_set & relevant_set)
    return relevant_retrieved / len(retrieved_set)


def recall(retrieved, relevant):
    """
    calculate recall
    how many relevant docs did we actualy find
    """
    if not relevant:
        return 0.0
    
    retrieved_set = set(str(d) for d in retrieved)
    relevant_set = set(str(d) for d in relevant)
    
    relevant_retrieved = len(retrieved_set & relevant_set)
    return relevant_retrieved / len(relevant_set)


def f1_score(retrieved, relevant):
    """
    f1 score - harmonic mean of precision and recall
    formula: 2 * (P * R) / (P + R)
    """
    p = precision(retrieved, relevant)
    r = recall(retrieved, relevant)
    
    if p + r == 0:
        return 0.0
    
    return 2 * p * r / (p + r)


def precision_at_k(ranked_list, relevant, k):
    """
    precision at rank k
    for lab5 we need k=5 and k=10
    """
    if k <= 0 or not ranked_list:
        return 0.0
    
    # just take the top k docs
    top_k_docs = ranked_list[:k]
    return precision(top_k_docs, relevant)


def r_precision(ranked_list, relevant):
    """
    r-precision = precision at R
    where R is the number of relevant docs
    """
    r = len(relevant)
    if r == 0:
        return 0.0
    
    return precision_at_k(ranked_list, relevant, r)


def reciprocal_rank(ranked_list, relevant):
    """
    reciprocal rank - 1/rank of first relevant doc
    used in MRR calculation
    """
    relevant_set = set(str(d) for d in relevant)
    
    # find first relevant document
    for rank, doc_id in enumerate(ranked_list, start=1):
        if str(doc_id) in relevant_set:
            return 1.0 / rank
    
    # no relevant doc found
    return 0.0


def average_precision(ranked_list, relevant):
    """
    calculate average precision for one query
    this is used to compute MAP later
    
    AP = (1/R) * sum of precisions at each relevant doc position
    """
    relevant_set = set(str(d) for d in relevant)
    num_relevant = len(relevant_set)
    
    if num_relevant == 0:
        return 0.0
    
    precision_sum = 0.0
    num_relevant_seen = 0
    
    # go through ranked list
    for rank, doc_id in enumerate(ranked_list, start=1):
        if str(doc_id) in relevant_set:
            num_relevant_seen += 1
            # precision at this rank
            precision_at_rank = num_relevant_seen / rank
            precision_sum += precision_at_rank
    
    # average over total relevant docs
    ap = precision_sum / num_relevant
    return ap


def get_precision_recall_points(ranked_list, relevant):
    """
    get precision and recall at each relevant document
    used for drawing PR curves
    """
    relevant_set = set(str(d) for d in relevant)
    num_relevant = len(relevant_set)
    
    if num_relevant == 0:
        return []
    
    pr_points = []
    num_relevant_seen = 0
    
    for rank, doc_id in enumerate(ranked_list, start=1):
        if str(doc_id) in relevant_set:
            num_relevant_seen += 1
            # calculate precision and recall at this point
            prec = num_relevant_seen / rank
            rec = num_relevant_seen / num_relevant
            pr_points.append((rec, prec))
    
    return pr_points


def interpolate_precision(pr_points, recall_levels=None):
    """
    interpolated precision at standard recall levels
    for each recall level, take max precision at that recall or higher
    this makes the curve smoother
    """
    if recall_levels is None:
        # standard 11 points: 0.0, 0.1, 0.2, ... 1.0
        recall_levels = [i * 0.1 for i in range(11)]
    
    if not pr_points:
        return {r: 0.0 for r in recall_levels}
    
    interpolated = {}
    
    for recall_level in recall_levels:
        # find max precision at this recall level or higher
        max_prec = 0.0
        for rec, prec in pr_points:
            if rec >= recall_level:
                max_prec = max(max_prec, prec)
        
        interpolated[recall_level] = max_prec
    
    return interpolated
