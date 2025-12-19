"""Latent Semantic Indexing (LSI) with k=100 topics"""

import numpy as np
import json
import os


# =========================
# TRAINING
# =========================
def train_lsi(matrix, k: int = 100):
    U, s, VT = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)

    Uk = U[:, :k]
    Sk = S[:k, :k]
    VTk = VT[:k, :]

    Sk_inv = np.linalg.inv(Sk)
    M = Uk @ Sk_inv

    return Uk, Sk, VTk, M


# =========================
# SIMILARITY
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)


# =========================
# RANKING
# =========================
def lsi_rank(query_tfidf, doc_ids, VTk, M):
    q_lsi = query_tfidf @ M
    doc_vectors = VTk.T

    scores = []
    for doc_id, doc_vec in zip(doc_ids, doc_vectors):
        score = cosine_similarity(q_lsi, doc_vec)
        scores.append({"doc_id": doc_id, "score": float(score)})

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


# =========================
# MAIN PIPELINE
# =========================
def run_lsi(
    tfidf_matrix,
    queries_tfidf,
    doc_ids,
    output_dir="results",
    k=100
):
    os.makedirs(output_dir, exist_ok=True)

    # Train LSI
    _, _, VTk, M = train_lsi(tfidf_matrix, k=k)

    results = {
        "model": f"LSI_k{k}",
        "queries": {}
    }

    # Rank documents for each query
    for qid, q_vec in queries_tfidf.items():
        results["queries"][qid] = lsi_rank(
            query_tfidf=q_vec,
            doc_ids=doc_ids,
            VTk=VTk,
            M=M
        )

    # Save results
    output_path = os.path.join(output_dir, f"lsi_k{k}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] LSI results saved to: {output_path}")


# =========================
# EXPECTED INPUTS
# =========================
# tfidf_matrix : np.ndarray  -> (terms x docs)
# queries_tfidf : dict       -> {query_id: np.ndarray (terms,)}
# doc_ids : list             -> ["MED1", "MED2", ...]


# =========================
# RUN
# =========================
run_lsi(
    tfidf_matrix=tfidf_matrix,
    queries_tfidf=queries_tfidf,
    doc_ids=doc_ids,
    output_dir="results",
    k=100
)
