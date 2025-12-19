"""Language Model with Jelinek-Mercer Smoothing"""

import math
import json
import os


# =========================
# JELINEK–MERCER (SINGLE QUERY)
# =========================
def jm_rank(
    query_terms,
    doc_term_counts,
    collection_model,
    lamb: float = 0.2
):
    """
    Language Model with Jelinek-Mercer smoothing (log-domain)
    """

    cf = collection_model["cf"]
    collection_length = collection_model["collection_length"]

    scores = {}

    for doc_id, tf_doc in doc_term_counts.items():
        doc_length = sum(tf_doc.values())
        log_score = 0.0

        for term in query_terms:
            # P_ml(w|D)
            p_doc = tf_doc.get(term, 0) / doc_length if doc_length > 0 else 0.0

            # P_ml(w|C)
            p_coll = cf.get(term, 0) / collection_length

            prob = lamb * p_doc + (1 - lamb) * p_coll

            # Numerical stability
            log_score += math.log(prob if prob > 0 else 1e-10)

        scores[doc_id] = log_score

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_jelinek_mercer_lm(
    queries_terms,
    doc_term_counts,
    collection_model,
    lamb: float = 0.2,
    output_dir="SourceCode/src/models/individual_models/individual_results"
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"LM_JelinekMercer_lambda_{lamb}"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = jm_rank(
            query_terms=terms,
            doc_term_counts=doc_term_counts,
            collection_model=collection_model,
            lamb=lamb
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

    print(f"[OK] Jelinek–Mercer LM results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# doc_term_counts : dict
#   -> { doc_id: {term: tf, ...}, ... }
#
# collection_model : dict with:
#   collection_model["cf"]
#   collection_model["collection_length"]


# =========================
# RUN
# =========================
run_jelinek_mercer_lm(
    queries_terms=queries_terms,
    doc_term_counts=doc_term_counts,
    collection_model=collection_model,
    lamb=0.2,
    output_dir="results"
)
