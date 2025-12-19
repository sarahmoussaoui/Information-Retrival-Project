"""Extended Binary Independence Retrieval"""

import math
import json
import os


# =========================
# EXTENDED BIR (SINGLE QUERY)
# =========================
def extended_bir_rank(
    query_terms,
    index,
    use_relevance=False,
    query_id=None
):
    """
    Extended Binary Independence Retrieval Model
    """
    N = index["N"]
    documents = index["documents"]
    binary_matrix = index["binary_matrix"]
    tfidf = index["tfidf"]
    qrels = index.get("qrels", {})

    relevant_docs = []
    R = 0

    if use_relevance:
        if query_id is None or query_id not in qrels:
            raise ValueError(
                "query_id and qrels required for Extended BIR with relevance"
            )
        relevant_docs = qrels[query_id]
        R = len(relevant_docs)

    scores = {}

    for doc in documents:
        rsv = 0.0

        for term in query_terms:
            w_ij = tfidf.get(term, {}).get(doc, 0.0)
            if w_ij == 0:
                continue

            n = sum(binary_matrix.get(term, {}).values())

            if not use_relevance:
                idf_prob = math.log((N - n + 0.5) / (n + 0.5))
                rsv += w_ij * idf_prob
            else:
                r = sum(
                    1 for d in relevant_docs
                    if binary_matrix.get(term, {}).get(d, 0) == 1
                )

                numerator = (r + 0.5) * (N - R - n + r + 0.5)
                denominator = (n - r + 0.5) * (R - r + 0.5)

                prob_weight = math.log(numerator / denominator)
                rsv += w_ij * prob_weight

        scores[doc] = rsv

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_extended_bir(
    queries_terms,
    index,
    output_dir="results",
    use_relevance=False
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = (
        "BIR_extended_with_relevance"
        if use_relevance
        else "BIR_extended_no_relevance"
    )

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = extended_bir_rank(
            query_terms=terms,
            index=index,
            use_relevance=use_relevance,
            query_id=qid
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

    print(f"[OK] {model_name} results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# index : dict with:
#   index["N"]
#   index["documents"]
#   index["binary_matrix"]
#   index["tfidf"]
#   index["qrels"]


# =========================
# RUN BOTH MODES
# =========================
run_extended_bir(
    queries_terms=queries_terms,
    index=index,
    output_dir="results",
    use_relevance=False
)

run_extended_bir(
    queries_terms=queries_terms,
    index=index,
    output_dir="results",
    use_relevance=True
)
