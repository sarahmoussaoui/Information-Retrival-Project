"""Language Model with Laplace (Add-1) Smoothing"""

import math
import json
import os


# =========================
# LAPLACE LM (SINGLE QUERY)
# =========================
def laplace_rank(
    query_terms,
    doc_term_counts,
    vocab_size: int
):
    """
    Language Model with Laplace smoothing (log-domain)
    """

    scores = {}

    for doc_id, tf_doc in doc_term_counts.items():
        doc_length = sum(tf_doc.values())
        log_score = 0.0

        for term in query_terms:
            tf = tf_doc.get(term, 0)
            prob = (tf + 1) / (doc_length + vocab_size)

            # Always > 0, but keep safe log
            log_score += math.log(prob)

        scores[doc_id] = log_score

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_laplace_lm(
    queries_terms,
    doc_term_counts,
    vocab_size: int,
    output_dir="SourceCode/src/models/individual_models/individual_results"
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = "LM_Laplace"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = laplace_rank(
            query_terms=terms,
            doc_term_counts=doc_term_counts,
            vocab_size=vocab_size
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

    print(f"[OK] Laplace LM results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# doc_term_counts : dict
#   -> { doc_id: {term: tf, ...}, ... }
#
# vocab_size : int
#   -> size of global vocabulary


# =========================
# RUN
# =========================
run_laplace_lm(
    queries_terms=queries_terms,
    doc_term_counts=doc_term_counts,
    vocab_size=vocab_size,
    output_dir="results"
)
