"""TF and TF-IDF weighting helpers."""

import math


def tf_norm(term_freq: int, tf_max: int) -> float:
    """Normalized term frequency: tf / tf_max."""

    if tf_max <= 0:
        return 0.0
    return float(term_freq) / float(tf_max)


def tf_idf(term_freq: int, tf_max: int, doc_freq: int, n_docs: int) -> float:
    """Compute TF-IDF with log10 scaling: tf_norm * log10((N / df) + 1)."""

    if n_docs <= 0:
        return 0.0
    tf_component = tf_norm(term_freq, tf_max)
    idf_component = math.log10((n_docs / doc_freq) + 1) if doc_freq > 0 else 0.0
    return tf_component * idf_component
