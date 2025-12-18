"""Document-term matrix construction and storage."""

from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


def build_matrix(tokenized_docs: Dict[int, List[str]]) -> Tuple[csr_matrix, Dict[str, int], Dict[int, int]]:
    """Build a sparse document-term matrix from tokenized docs.

    Args:
        tokenized_docs: Mapping of doc_id -> list of tokens

    Returns:
        matrix: csr_matrix of shape (n_docs, n_terms)
        vocab: term -> column index
        doc_index: doc_id -> row index
    """

    # Assign row indices to doc_ids
    doc_ids = sorted(tokenized_docs.keys())
    doc_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Build vocabulary
    vocab = {}
    current_col = 0

    rows = []
    cols = []
    data = []

    for doc_id in doc_ids:
        tokens = tokenized_docs[doc_id]
        term_counts = {}
        for term in tokens:
            if term not in vocab:
                vocab[term] = current_col
                current_col += 1
            term_counts[term] = term_counts.get(term, 0) + 1

        row = doc_index[doc_id]
        for term, count in term_counts.items():
            rows.append(row)
            cols.append(vocab[term])
            data.append(count)

    n_rows = len(doc_ids)
    n_cols = len(vocab)
    matrix = csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float64)

    return matrix, vocab, doc_index


def build_binary_matrix(
    tokenized_docs: Dict[int, List[str]],
    vocab: Dict[str, int],
    doc_index: Dict[int, int],
) -> csr_matrix:
    """Build a binary presence/absence document-term matrix.

    Uses the provided `vocab` (term -> col) and `doc_index` (doc_id -> row)
    to ensure the same layout as the TF matrix.

    Args:
        tokenized_docs: Mapping of doc_id -> list of tokens
        vocab: Term vocabulary mapping used for columns
        doc_index: Document index mapping used for rows

    Returns:
        csr_matrix of shape (n_docs, n_terms) with 1.0 if term occurs in doc, else 0.0
    """

    n_rows = len(doc_index)
    n_cols = len(vocab)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []

    for doc_id, row in doc_index.items():
        tokens = tokenized_docs.get(doc_id, [])
        if not tokens:
            continue
        seen = set()
        for term in tokens:
            if term in vocab and term not in seen:
                rows.append(row)
                cols.append(vocab[term])
                data.append(1.0)
                seen.add(term)

    return csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.float64)
