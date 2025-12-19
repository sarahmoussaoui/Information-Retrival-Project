"""All IR Models for Lab 5 – ready to import
  rank → emphasizes that the function produces a ranking directly.
  score → emphasizes that the function computes a score for each document (ranking is done after scoring).
"""

import numpy as np
import math

# =========================
# 1️⃣ Vector Space Model
# =========================
def vsm_rank(query, doc_term_matrix, vocab):
    q_vec = np.zeros(len(vocab))
    term_to_idx = {term: i for i, term in enumerate(vocab)}
    for term in query:
        if term in term_to_idx:
            q_vec[term_to_idx[term]] = 1  # binary weight

    doc_vectors = {}
    for doc, term_weights in doc_term_matrix.items():
        vec = np.zeros(len(vocab))
        for term, weight in term_weights.items():
            if term in term_to_idx:
                vec[term_to_idx[term]] = weight
        doc_vectors[doc] = vec

    scores = {}
    q_norm = np.linalg.norm(q_vec)
    for doc, d_vec in doc_vectors.items():
        d_norm = np.linalg.norm(d_vec)
        scores[doc] = np.dot(q_vec, d_vec)/(q_norm*d_norm + 1e-10) if q_norm*d_norm != 0 else 0.0

    ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

# =========================
# 2️⃣ LSI
# =========================
def train_lsi(matrix, k=100):
    U, s, VT = np.linalg.svd(matrix, full_matrices=False)
    S = np.diag(s)
    Uk = U[:, :k]
    Sk = S[:k, :k]
    VTk = VT[:k, :]
    Sk_inv = np.linalg.inv(Sk)
    M = Uk @ Sk_inv
    return Uk, Sk, VTk, M

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def lsi_rank(query_vec, doc_ids, VTk, M):
    q_lsi = query_vec @ M
    doc_vectors = VTk.T
    scores = []
    for doc_id, doc_vec in zip(doc_ids, doc_vectors):
        score = cosine_similarity(q_lsi, doc_vec)
        scores.append((doc_id, float(score)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores

# =========================
# 3️⃣ Classic BIR
# =========================
def bir_rank(query_terms, index, use_relevance=False, query_id=None):
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
            if term not in binary_matrix: continue
            if binary_matrix[term].get(doc, 0) == 1:
                n = sum(binary_matrix[term].values())
                if not use_relevance:
                    rsv += math.log((N-n+0.5)/(n+0.5))
                else:
                    r = sum(1 for d in relevant_docs if binary_matrix[term].get(d,0)==1)
                    numerator = (r+0.5)*(N-R-n+r+0.5)
                    denominator = (n-r+0.5)*(R-r+0.5)
                    rsv += math.log(numerator/denominator)
        scores[doc] = rsv
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking

# =========================
# 4️⃣ Extended BIR
# =========================
def ext_bir_rank(query_terms, index, use_relevance=False, query_id=None):
    N = index["N"]
    documents = index["documents"]
    binary_matrix = index["binary_matrix"]
    tfidf = index["tfidf"]
    qrels = index.get("qrels", {})
    relevant_docs = []
    R = 0
    if use_relevance:
        if query_id is None or query_id not in qrels:
            raise ValueError("query_id and qrels required for Extended BIR with relevance")
        relevant_docs = qrels[query_id]
        R = len(relevant_docs)
    scores = {}
    for doc in documents:
        rsv = 0.0
        for term in query_terms:
            w_ij = tfidf.get(term, {}).get(doc, 0.0)
            if w_ij == 0: continue
            n = sum(binary_matrix.get(term, {}).values())
            if not use_relevance:
                rsv += w_ij * math.log((N-n+0.5)/(n+0.5))
            else:
                r = sum(1 for d in relevant_docs if binary_matrix.get(term, {}).get(d,0)==1)
                numerator = (r+0.5)*(N-R-n+r+0.5)
                denominator = (n-r+0.5)*(R-r+0.5)
                rsv += w_ij * math.log(numerator/denominator)
        scores[doc] = rsv
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking

# =========================
# 5️⃣ BM25
# =========================
def bm25_score(query_terms, index, k1=1.2, b=0.75):
    N = index["N"]
    documents = index["documents"]
    tf = index["tf"]
    doc_lengths = index["doc_lengths"]
    avgdl = index["avg_doc_length"]
    binary_matrix = index["binary_matrix"]
    scores = {}
    for doc in documents:
        score_val = 0.0
        dl = doc_lengths[doc]
        for term in query_terms:
            tf_td = tf.get(term, {}).get(doc, 0)
            if tf_td==0: continue
            n = sum(binary_matrix.get(term, {}).values())
            idf = math.log((N-n+0.5)/(n+0.5))
            norm = 1 - b + b * (dl/avgdl)
            tf_component = tf_td*(k1+1)/(tf_td + k1*norm)
            score_val += idf*tf_component
        scores[doc] = score_val
    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranking

# =========================
# 6️⃣ Language Model – MLE
# =========================
def mle_score(query_terms, doc_term_counts):
    scores = {}
    for doc_id, tf_doc in doc_term_counts.items():
        dl = sum(tf_doc.values())
        log_score = 0.0
        zero_prob = False
        for term in query_terms:
            tf = tf_doc.get(term,0)
            if tf==0 or dl==0:
                zero_prob = True
                break
            log_score += math.log(tf/dl)
        scores[doc_id] = float("-inf") if zero_prob else log_score
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return ranking

# =========================
# 7️⃣ Language Model – Laplace
# =========================
def laplace_score(query_terms, doc_term_counts, vocab_size):
    scores = {}
    for doc_id, tf_doc in doc_term_counts.items():
        dl = sum(tf_doc.values())
        log_score = 0.0
        for term in query_terms:
            tf = tf_doc.get(term,0)
            log_score += math.log((tf+1)/(dl+vocab_size))
        scores[doc_id] = log_score
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return ranking

# =========================
# 8️⃣ Language Model – Jelinek-Mercer
# =========================
def jm_score(query_terms, doc_term_counts, collection_model, lamb=0.2):
    scores = {}
    cf = collection_model["cf"]
    collection_length = collection_model["collection_length"]
    for doc_id, tf_doc in doc_term_counts.items():
        dl = sum(tf_doc.values())
        log_score = 0.0
        for term in query_terms:
            p_doc = tf_doc.get(term,0)/dl if dl>0 else 0.0
            p_coll = cf.get(term,0)/collection_length
            prob = lamb*p_doc + (1-lamb)*p_coll
            log_score += math.log(prob if prob>0 else 1e-10)
        scores[doc_id] = log_score
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return ranking

# =========================
# 9️⃣ Language Model – Dirichlet
# =========================
def dirichlet_score(query_terms, doc_term_counts, collection_model, mu=2000):
    scores = {}
    cf = collection_model["cf"]
    collection_length = collection_model["collection_length"]
    for doc_id, tf_doc in doc_term_counts.items():
        dl = sum(tf_doc.values())
        log_score = 0.0
        for term in query_terms:
            tf = tf_doc.get(term,0)
            p_coll = cf.get(term,0)/collection_length
            prob = (tf + mu*p_coll)/(dl + mu)
            log_score += math.log(prob if prob>0 else 1e-10)
        scores[doc_id] = log_score
    ranking = sorted(scores.items(), key=lambda x:x[1], reverse=True)
    return ranking
