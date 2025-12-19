"""Language Model with Maximum Likelihood Estimation (MLE)"""

import math
import json
import os


# =========================
# MLE LM (SINGLE QUERY)
# =========================
def mle_rank(
    query_terms,
    doc_term_counts
):
    """
    Language Model with Maximum Likelihood Estimation (log-domain)
    """

    scores = {}

    for doc_id, tf_doc in doc_term_counts.items():
        doc_length = sum(tf_doc.values())
        log_score = 0.0
        zero_prob = False

        for term in query_terms:
            tf = tf_doc.get(term, 0)

            if tf == 0 or doc_length == 0:
                # Unseen term → zero probability
                zero_prob = True
                break

            prob = tf / doc_length
            log_score += math.log10(prob)

        # Assign very small score if any term unseen
        scores[doc_id] = float("-inf") if zero_prob else log_score

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_mle_lm(
    queries_terms,
    doc_term_counts,
    output_dir="SourceCode/src/models/individual_models/individual_results"
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = "LM_MLE"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = mle_rank(
            query_terms=terms,
            doc_term_counts=doc_term_counts
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

    print(f"[OK] MLE LM results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# doc_term_counts : dict
#   -> { doc_id: {term: tf, ...}, ... }


# =========================
# RUN
# =========================
run_mle_lm(
    queries_terms=queries_terms,
    doc_term_counts=doc_term_counts,
    output_dir="results"
)
