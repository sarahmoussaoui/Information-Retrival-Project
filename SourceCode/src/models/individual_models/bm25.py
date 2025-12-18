"""BM25 Ranking"""

import math
import json
import os


# =========================
# BM25 RANKING (SINGLE QUERY)
# =========================
def bm25_rank(
    query_terms,
    index,
    k1=1.2,
    b=0.75
):
    N = index["N"]
    documents = index["documents"]
    tf = index["tf"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avg_doc_length"]
    binary_matrix = index["binary_matrix"]

    scores = {}

    for doc in documents:
        score = 0.0
        dl = doc_lengths[doc]

        for term in query_terms:
            tf_td = tf.get(term, {}).get(doc, 0)
            if tf_td == 0:
                continue

            n = sum(binary_matrix.get(term, {}).values())
            idf = math.log((N - n + 0.5) / (n + 0.5))

            norm = 1 - b + b * (dl / avgdl)
            tf_component = (tf_td * (k1 + 1)) / (tf_td + k1 * norm)

            score += idf * tf_component

        scores[doc] = score

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking


# =========================
# RUN FULL PIPELINE
# =========================
def run_bm25(
    queries_terms,
    index,
    output_dir="results",
    k1=1.2,
    b=0.75
):
    os.makedirs(output_dir, exist_ok=True)

    model_name = f"BM25_k1_{k1}_b_{b}"

    results = {
        "model": model_name,
        "queries": {}
    }

    for qid, terms in queries_terms.items():
        ranking = bm25_rank(
            query_terms=terms,
            index=index,
            k1=k1,
            b=b
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

    print(f"[OK] BM25 results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# queries_terms : dict
#   -> { "I1": ["term1", "term2"], ... }
#
# index : dict with:
#   index["N"]
#   index["documents"]
#   index["tf"]
#   index["doc_lengths"]
#   index["avg_doc_length"]
#   index["binary_matrix"]


# =========================
# RUN
# =========================
run_bm25(
    queries_terms=queries_terms,
    index=index,
    output_dir="results",
    k1=1.2,
    b=0.75
)
