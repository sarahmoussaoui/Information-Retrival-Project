"""Vector Space Model (VSM) with Cosine Similarity"""

import numpy as np
import json
import os


# =========================
# VSM RANKING (SINGLE QUERY)
# =========================
def vsm_rank(
    query_terms,
    doc_term_matrix,
    vocab
):
    """
    Vector Space Model ranking with cosine similarity.
    """

    # Build query vector
    q_vec = np.zeros(len(vocab))
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    for term in query_terms:
        if term in term_to_idx:
            q_vec[term_to_idx[term]] = 1  # binary weight

    # Build document vectors
    doc_vectors = {}
    for doc, term_weights in doc_term_matrix.items():
        vec = np.zeros(len(vocab))
        for term, weight in term_weights.items():
            if term in term_to_idx:
                vec[term_to_idx[term]] = weight
        doc_vectors[doc] = vec

    # Compute cosine similarity
    scores = {}
    q_norm = np.linalg.norm(q_vec)
    for doc, d_vec in doc_vectors.items():
        d_norm = np.linalg.norm(d_vec)
        if q_norm == 0 or d_norm == 0:
            scores[doc] = 0.0
        else:
            scores[doc] = np.dot(q_vec, d_vec) / (q_norm * d_norm)

    # Sort documents descending
    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs


# =========================
# RUN FULL PIPELINE
# =========================
def run_vsm(
    queries_terms,
    doc_term_matrix,
    vocab,
    output_dir="results"
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = "VSM_Cosine"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = vsm_rank(
            query_terms=terms,
            doc_term_matrix=doc_term_matrix,
            vocab=vocab
        )

        results["queries"][qid] = [
            {"doc_id": doc_id, "score": float(score)}
            for doc_id, score in ranking
        ]

    output_path = os.path.join(
        output_dir,
        f"{model_name}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] VSM Cosine results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# doc_term_matrix : dict
#   -> { doc_id: {term: weight, ...}, ... } (TF-IDF)
#
# vocab : list of all terms in collection


# =========================
# RUN
# =========================
run_vsm(
    queries_terms=queries_terms,
    doc_term_matrix=doc_term_matrix,
    vocab=vocab,
    output_dir="results"
)
