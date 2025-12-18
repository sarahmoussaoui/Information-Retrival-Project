"""TF and TF-IDF weighting helpers."""

def tf(term_freq: int) -> float:
    return float(term_freq)


def tf_idf(term_freq: int, doc_freq: int, n_docs: int) -> float:
    """Compute TF-IDF with log scaling."""
    raise NotImplementedError
