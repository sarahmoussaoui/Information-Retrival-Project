"""Build and query an inverted index."""

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def build_inverted_index(docs: Dict[int, List[str]]):
    """Construct an inverted index mapping terms to postings.

    Args:
        docs: Mapping of doc_id -> list of preprocessed tokens

    Returns:
        index: dict[str, list[tuple[int, int]]] where each term maps to
               a postings list of (doc_id, term_frequency)

    Example:
        Input docs (preprocessed):
            {
                1: ["plasma", "plasma", "glucos"],
                7: ["glucos"],
                42: ["plasma"]
            }

        Output inverted index:
            {
                "glucos": [(1, 1), (7, 1)],
                "plasma": [(1, 2), (42, 1)]
            }
    """

    index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

    for doc_id, tokens in docs.items():
        counts = Counter(tokens)
        for term, tf in counts.items():
            index[term].append((doc_id, tf))

    # Sort postings by doc_id for deterministic ordering
    for term in index:
        index[term].sort(key=lambda x: x[0])

    return dict(index)
