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


def precision(retrieved, relevant):
    """
    Calculate precision: TP / (TP + FP) = TP / |retrieved|
    
    Args:
        retrieved: List of retrieved document IDs (or list of dicts with "doc_id")
        relevant: Set or list of relevant document IDs
        
    Returns:
        float: Precision score (0.0 to 1.0)
    """
    if len(retrieved) == 0:
        return 0.0
    
    retrieved_ids = _extract_doc_ids(retrieved)
    relevant_set = _to_set(relevant)
    retrieved_set = _to_set(retrieved_ids)
    
    # True positives: documents that are both retrieved and relevant
    tp = len(retrieved_set & relevant_set)
    
    return tp / len(retrieved_set)


def recall(retrieved, relevant):
    """
    Calculate recall: TP / (TP + FN) = TP / |relevant|
    
    Args:
        retrieved: List of retrieved document IDs (or list of dicts with "doc_id")
        relevant: Set or list of relevant document IDs
        
    Returns:
        float: Recall score (0.0 to 1.0)
    """
    if len(relevant) == 0:
        return 0.0
    
    retrieved_ids = _extract_doc_ids(retrieved)
    relevant_set = _to_set(relevant)
    retrieved_set = _to_set(retrieved_ids)
    
    # True positives: documents that are both retrieved and relevant
    tp = len(retrieved_set & relevant_set)
    
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

