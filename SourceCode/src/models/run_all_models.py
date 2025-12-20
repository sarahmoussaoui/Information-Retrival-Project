"""
Master Script to Run All IR Models for All Queries
Produces JSON results per model in 'results/' folder
"""

import os
import json
import numpy as np
from models import (
    vsm_rank,
    train_lsi, lsi_rank,
    bir_rank,
    ext_bir_rank,
    bm25_score,
    mle_score,
    laplace_score,
    jm_score,
    dirichlet_score
)

import json
import math

# queries_terms : dict { query_id: [term, ...] }
with open(r"SourceCode\data\processed\parse_preprocess\queries_processed.json") as f:
    queries_terms = json.load(f)

# qrels : dict { query_id: [doc_id, ...] }
with open(r"SourceCode/data/processed/parse_preprocess/qrels.json") as f:
    qrels = json.load(f)

with open(r"SourceCode/data/processed/build_tf_idf_stats/doc_freq.json") as f:
    doc_freq = json.load(f)

with open(r"SourceCode/data/processed/build_index/doc_index.json") as f:
    doc_index = json.load(f)

with open(r"SourceCode/data/processed/build_tf_idf_stats/n_docs.json") as f:
    N = json.load(f)["n_docs"]

# vocab : list of terms
with open(r"SourceCode/data/processed/build_index/vocab.json") as f:
    vocab_dict = json.load(f) 

vocab = list(vocab_dict.keys())

with open(r"SourceCode/data/processed/build_index/doc_lengths.json") as f:
    doc_lengths = json.load(f)

with open(r"SourceCode/data/processed/build_index/avg_doc_length.json") as f:
    avg_doc_length = json.load(f)["avg_doc_length"]

##
with open(r"SourceCode/data/processed/build_tf_idf_stats/doc_tf_norm.json") as f:
    doc_tf_norm = json.load(f)

with open(r"SourceCode/data/processed/build_tf_idf_stats/collection_tf.json") as f: # should i use normalized or not ?
    collection_tf = json.load(f)

# tf : dict { term: {doc_id: tf} }
tf = {}

for doc_id, terms in doc_tf_norm.items():
    for term, freq in terms.items():
        tf.setdefault(term, {})[doc_id] = freq

# binary_matrix : dict { term: {doc_id: 0/1} }
binary_matrix = {
    term: {doc_id: 1 for doc_id in docs}
    for term, docs in tf.items()
}

# doc_term_counts : dict { doc_id: {term: tf} }
doc_term_counts = doc_tf_norm            

# doc_term_matrix : dict { doc_id: {term: tfidf} }
doc_term_matrix = {}

for doc_id, terms in doc_tf_norm.items():
    doc_term_matrix[doc_id] = {}

    for term, tf_norm in terms.items():
        idf = math.log10((N / doc_freq[term])+1) 
        doc_term_matrix[doc_id][term] = tf_norm * idf

collection_model = {
    "cf": collection_tf,
    "collection_length": sum(collection_tf.values())
}


output_dir = "./SourceCode/Results"
os.makedirs(output_dir, exist_ok=True)


def save_results(model_name, results):
    path = os.path.join(output_dir, f"{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] {model_name} saved to {path}")


# =========================
# 1️⃣ VSM
# =========================
vsm_results = {"model": "VSM_Cosine", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = vsm_rank(terms, doc_term_matrix, vocab)
    vsm_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("VSM_Cosine", vsm_results)


# =========================
# 2️⃣ LSI
# =========================
matrix = np.array([list(doc_term_matrix[d].get(t,0) for t in vocab) for d in doc_term_matrix]).T
Uk, Sk, VTk, M = train_lsi(matrix, k=100)
lsi_results = {"model": "LSI_k100", "queries": {}}
doc_ids = list(doc_term_matrix.keys())
for qid, terms in queries_terms.items():
    # build query tfidf vector
    q_vec = np.zeros(len(vocab))
    for t in terms:
        if t in vocab:
            q_vec[vocab.index(t)] = 1
    ranking = lsi_rank(q_vec, doc_ids, VTk, M)
    lsi_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("LSI_k100", lsi_results)


# =========================
# 3️⃣ Classic BIR (without and with relevance)
# =========================
for use_rel in [False, True]:
    model_name = "BIR_with_relevance" if use_rel else "BIR_no_relevance"
    bir_results = {"model": model_name, "queries": {}}
    for qid, terms in queries_terms.items():
        ranking = bir_rank(terms, index={"N": N, "documents": list(doc_term_counts.keys()),
                                         "binary_matrix": binary_matrix, "qrels": qrels},
                           use_relevance=use_rel, query_id=qid)
        bir_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
    save_results(model_name, bir_results)


# =========================
# 4️⃣ Extended BIR (without and with relevance)
# =========================
for use_rel in [False, True]:
    model_name = "ExtendedBIR_with_relevance" if use_rel else "ExtendedBIR_no_relevance"
    ext_bir_results = {"model": model_name, "queries": {}}
    for qid, terms in queries_terms.items():
        ranking = ext_bir_rank(terms, index={"N": N, "documents": list(doc_term_counts.keys()),
                                             "binary_matrix": binary_matrix, "tfidf": doc_term_matrix,
                                             "qrels": qrels},
                               use_relevance=use_rel, query_id=qid)
        ext_bir_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
    save_results(model_name, ext_bir_results)


# =========================
# 5️⃣ BM25
# =========================
bm25_results = {"model": "BM25", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = bm25_score(terms, index={"N": N, "documents": list(doc_term_counts.keys()),
                                       "tf": tf, "doc_lengths": doc_lengths,
                                       "avg_doc_length": avg_doc_length,
                                       "binary_matrix": binary_matrix})
    bm25_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("BM25", bm25_results)


# =========================
# 6️⃣ LM MLE
# =========================
mle_results = {"model": "LM_MLE", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = mle_score(terms, doc_term_counts)
    mle_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("LM_MLE", mle_results)


# =========================
# 7️⃣ LM Laplace
# =========================
laplace_results = {"model": "LM_Laplace", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = laplace_score(terms, doc_term_counts, vocab_size=len(vocab))
    laplace_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("LM_Laplace", laplace_results)


# =========================
# 8️⃣ LM Jelinek-Mercer
# =========================
jm_results = {"model": "LM_JelinekMercer", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = jm_score(terms, doc_term_counts, collection_model, lamb=0.2)
    jm_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("LM_JelinekMercer", jm_results)


# =========================
# 9️⃣ LM Dirichlet
# =========================
dirichlet_results = {"model": "LM_Dirichlet", "queries": {}}
for qid, terms in queries_terms.items():
    ranking = dirichlet_score(terms, doc_term_counts, collection_model)
    dirichlet_results["queries"][qid] = [{"doc_id": d, "score": float(s)} for d, s in ranking]
save_results("LM_Dirichlet", dirichlet_results)


print("[ALL DONE] All models executed and JSON results saved in 'Results/' folder.")