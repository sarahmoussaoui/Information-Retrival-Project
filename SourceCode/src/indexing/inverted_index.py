"""Build and query an inverted index with TF-IDF weights."""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from .weighting import tf_idf


def build_inverted_index(docs: Dict[int, List[str]]):
    """Construct an inverted index mapping terms to TF-IDF-weighted postings.

    Args:
        docs: Mapping of doc_id -> list of preprocessed tokens

    Returns:
        index: dict[str, list[tuple[int, float]]] where each term maps to
               a postings list of (doc_id, tf-idf weight)

    Example:
        Input docs (preprocessed):
            {
                1: ["plasma", "plasma", "glucos"],
                7: ["glucos"],
                42: ["plasma"]
            }

        Output inverted index (weights shown conceptually):
            {
                "glucos": [(1, w11), (7, w71)],
                "plasma": [(1, w12), (42, w421)]
            }
    """

    index: Dict[str, List[Tuple[int, float]]] = defaultdict(list)

    # Precompute per-document tf_max and global document frequencies (df)
    N = len(docs)
    doc_tf_max: Dict[int, int] = {}
    doc_term_counts: Dict[int, Counter] = {}
    df_counter = Counter()

    for doc_id, tokens in docs.items():
        counts = Counter(tokens)
        doc_term_counts[doc_id] = counts
        doc_tf_max[doc_id] = max(counts.values()) if counts else 0
        # Document frequency: count presence (once per doc) for each term
        for term in counts.keys():
            df_counter[term] += 1

    # Build postings with tf-idf weights
    for doc_id, counts in doc_term_counts.items():
        tf_max = doc_tf_max[doc_id]
        for term, tf_raw in counts.items():
            df = df_counter[term]
            weight = tf_idf(tf_raw, tf_max, df, N)
            index[term].append((doc_id, float(weight)))

    # Sort postings by doc_id for deterministic ordering
    for term in index:
        index[term].sort(key=lambda x: x[0])

    return dict(index)
