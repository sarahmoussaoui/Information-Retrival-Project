"""Language Model with Dirichlet Prior Smoothing """

import math
import json
import os


# =========================
# DIRICHLET LM (SINGLE QUERY)
# =========================
def dirichlet_rank(
    query_terms,
    doc_term_counts,
    collection_model,
    mu: float = 2000
):
    """
    Language Model with Dirichlet prior smoothing (log-domain)
    """

    cf = collection_model["cf"]
    collection_length = collection_model["collection_length"]

    scores = {}

    for doc_id, tf_doc in doc_term_counts.items():
        doc_length = sum(tf_doc.values())
        log_score = 0.0

        for term in query_terms:
            tf = tf_doc.get(term, 0)
            p_coll = cf.get(term, 0) / collection_length

            prob = (tf + mu * p_coll) / (doc_length + mu)

            # Numerical safety
            log_score += math.log(prob if prob > 0 else 1e-10)

        scores[doc_id] = log_score

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_dirichlet_lm(
    queries_terms,
    doc_term_counts,
    collection_model,
    mu: float = 2000,
    output_dir="SourceCode/src/models/individual_models/individual_results"
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"LM_Dirichlet_mu_{mu}"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = dirichlet_rank(
            query_terms=terms,
            doc_term_counts=doc_term_counts,
            collection_model=collection_model,
            mu=mu
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

    print(f"[OK] Dirichlet LM results saved to: {output_path}")


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
run_dirichlet_lm(
    queries_terms=queries_terms,
    doc_term_counts=doc_term_counts,
    collection_model=collection_model,
    mu=2000,
    output_dir="results"
)
