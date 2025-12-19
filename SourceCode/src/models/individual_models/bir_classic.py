"""Binary Independence Retrieval (Classic)"""

import math
import json
import os


# =========================
# BIR RANKING (SINGLE QUERY)
# =========================
def bir_rank(
    query_terms,
    index,
    use_relevance=False,
    query_id=None
):
    """
    Classic Binary Independence Retrieval
    """
    N = index["N"]
    documents = index["documents"]
    binary_matrix = index["binary_matrix"]
    qrels = index.get("qrels", {})

    relevant_docs = []
    R = 0

    if use_relevance:
        if query_id is None or query_id not in qrels:
            raise ValueError("query_id and qrels required for BIR with relevance")
        relevant_docs = qrels[query_id]
        R = len(relevant_docs)

    scores = {}

    for doc in documents:
        rsv = 0.0

        for term in query_terms:
            if term not in binary_matrix:
                continue

            if binary_matrix[term].get(doc, 0) == 1:
                n = sum(binary_matrix[term].values())

                if not use_relevance:
                    rsv += math.log((N - n + 0.5) / (n + 0.5))
                else:
                    r = sum(
                        1 for d in relevant_docs
                        if binary_matrix[term].get(d, 0) == 1
                    )

                    numerator = (r + 0.5) * (N - R - n + r + 0.5)
                    denominator = (n - r + 0.5) * (R - r + 0.5)

                    rsv += math.log(numerator / denominator)

        scores[doc] = rsv

    # Sort descending
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_bir(
    queries_terms,
    index,
    output_dir="results",
    use_relevance=False
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = (
        "BIR_classic_with_relevance"
        if use_relevance
        else "BIR_classic_no_relevance"
    )

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = bir_rank(
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
#   index["N"]              -> number of documents
#   index["documents"]      -> list of doc IDs
#   index["binary_matrix"]  -> {term: {doc_id: 0/1}}
#   index["qrels"]          -> {query_id: [relevant_doc_ids]}


# =========================
# RUN BOTH MODES
# =========================
run_bir(
    queries_terms=queries_terms,
    index=index,
    output_dir="results",
    use_relevance=False
)

run_bir(
    queries_terms=queries_terms,
    index=index,
    output_dir="results",
    use_relevance=True
)
