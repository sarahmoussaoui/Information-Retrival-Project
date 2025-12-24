import json
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# LOADERS
# =========================================================

def load_model_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_rankings = {}
    for qid, docs in data["queries"].items():
        all_rankings[str(qid)] = [str(d["doc_id"]) for d in docs]

    return all_rankings


def load_qrels(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        str(qid): set(str(doc_id) for doc_id in docs)
        for qid, docs in data.items()
    }


# =========================================================
# METRICS
# =========================================================

def precision_recall_curve(ranked_docs, relevant_docs):
    precisions, recalls = [], []
    retrieved_relevant = 0
    total_relevant = len(relevant_docs)

    for k, doc_id in enumerate(ranked_docs, start=1):
        if doc_id in relevant_docs:
            retrieved_relevant += 1
        precisions.append(retrieved_relevant / k)
        recalls.append(retrieved_relevant / total_relevant if total_relevant else 0)

    return recalls, precisions


def interpolated_pr_curve(recalls, precisions):
    recall_levels = np.linspace(0, 1, 11)
    interp_precisions = []

    for r in recall_levels:
        candidates = [p for rec, p in zip(recalls, precisions) if rec >= r]
        interp_precisions.append(max(candidates) if candidates else 0)

    return recall_levels, interp_precisions


def precision_at_k(ranked_docs, relevant_docs, k):
    return sum(1 for d in ranked_docs[:k] if d in relevant_docs) / k


def r_precision(ranked_docs, relevant_docs):
    R = len(relevant_docs)
    if R == 0:
        return 0
    return sum(1 for d in ranked_docs[:R] if d in relevant_docs) / R


def reciprocal_rank(ranked_docs, relevant_docs):
    for rank, doc in enumerate(ranked_docs, start=1):
        if doc in relevant_docs:
            return 1 / rank
    return 0


def average_precision(ranked_docs, relevant_docs):
    hits, score = 0, 0.0
    for k, doc in enumerate(ranked_docs, start=1):
        if doc in relevant_docs:
            hits += 1
            score += hits / k
    return score / len(relevant_docs) if relevant_docs else 0


def mean_average_precision(all_rankings, all_rels):
    return np.mean([
        average_precision(all_rankings[q], all_rels[q])
        for q in all_rankings
    ])


def mean_reciprocal_rank(all_rankings, all_rels):
    return np.mean([
        reciprocal_rank(all_rankings[q], all_rels[q])
        for q in all_rankings
    ])


def mean_pr_curve(all_rankings, all_rels):
    recall_levels = np.linspace(0, 1, 11)
    avg_precisions = np.zeros(11)

    for q in all_rankings:
        recalls, precisions = precision_recall_curve(all_rankings[q], all_rels[q])
        for i, r in enumerate(recall_levels):
            for rec, p in zip(recalls, precisions):
                if rec >= r:
                    avg_precisions[i] += p
                    break

    return recall_levels, avg_precisions / len(all_rankings)


def mean_interpolated_pr_curve(all_rankings, all_rels):
    recall_levels = np.linspace(0, 1, 11)
    avg_precisions = np.zeros(11)

    for q in all_rankings:
        recalls, precisions = precision_recall_curve(all_rankings[q], all_rels[q])
        _, interp = interpolated_pr_curve(recalls, precisions)
        avg_precisions += np.array(interp)

    return recall_levels, avg_precisions / len(all_rankings)

# =========================================================
#(FOR run_all)
# =========================================================

def map_score(all_rankings, all_rels):
    """
    Mean Average Precision (MAP)
    """
    ap_scores = []
    for qid in all_rankings:
        ranked_docs = all_rankings[qid]
        relevant_docs = all_rels.get(qid, set())
        ap_scores.append(average_precision(ranked_docs, relevant_docs))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def interpolated_map(all_rankings, all_rels):
    """
    Interpolated MAP (11-point interpolation)
    """
    recall_levels = np.linspace(0, 1, 11)
    interp_scores = []

    for qid in all_rankings:
        ranked_docs = all_rankings[qid]
        relevant_docs = all_rels.get(qid, set())

        recalls, precisions = precision_recall_curve(ranked_docs, relevant_docs)
        _, interp_precisions = interpolated_pr_curve(recalls, precisions)
        interp_scores.append(np.mean(interp_precisions))

    return float(np.mean(interp_scores)) if interp_scores else 0.0


def get_pr_curve_data(runs, qrels, query_id):
    """
    PR curve for ONE query (used in run_all)
    """
    ranked_docs = runs[query_id]
    relevant_docs = qrels.get(query_id, set())
    recalls, precisions = precision_recall_curve(ranked_docs, relevant_docs)
    return recalls, precisions


def get_interpolated_pr_curve(runs, qrels):
    """
    Interpolated PR curve (mean over queries)
    """
    recall_levels = np.linspace(0, 1, 11)
    avg_precisions = np.zeros(11)

    for qid in runs:
        ranked_docs = runs[qid]
        relevant_docs = qrels.get(qid, set())
        recalls, precisions = precision_recall_curve(ranked_docs, relevant_docs)
        _, interp = interpolated_pr_curve(recalls, precisions)
        avg_precisions += np.array(interp)

    avg_precisions /= len(runs)
    return recall_levels.tolist(), avg_precisions.tolist()

# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(model_name, all_rankings, all_rels, plot=True):
    print(f"\n====== {model_name} ======")

    per_query = {}

    for qid in all_rankings:
        ranked_docs = all_rankings[qid]
        relevant_docs = all_rels.get(qid, set())

        rr = reciprocal_rank(ranked_docs, relevant_docs)
        r_prec = r_precision(ranked_docs, relevant_docs)

        per_query[qid] = {
            "RR": float(rr),
            "R-Precision": float(r_prec),
            "P@5": float(precision_at_k(ranked_docs, relevant_docs, 5)),
            "P@10": float(precision_at_k(ranked_docs, relevant_docs, 10)),
            "AP": float(average_precision(ranked_docs, relevant_docs))
        }

    # ===== Moyennes globales =====
    map_score = np.mean([per_query[q]["AP"] for q in per_query])
    mrr_score = np.mean([per_query[q]["RR"] for q in per_query])
    p5 = np.mean([per_query[q]["P@5"] for q in per_query])
    p10 = np.mean([per_query[q]["P@10"] for q in per_query])
    r_prec_mean = np.mean([per_query[q]["R-Precision"] for q in per_query])

    print(f"MAP  = {map_score:.4f}")
    print(f"MRR  = {mrr_score:.4f}")
    print(f"P@5  = {p5:.4f}")
    print(f"P@10 = {p10:.4f}")
    print(f"R-P  = {r_prec_mean:.4f}")

    # ===== Courbes PR =====
    r1, p1 = mean_pr_curve(all_rankings, all_rels)
    r2, p2 = mean_interpolated_pr_curve(all_rankings, all_rels)

    if plot:
        plt.figure()
        plt.plot(r1, p1, marker="o")
        plt.title(f"PR Curve – {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.show()

        plt.figure()
        plt.plot(r2, p2, marker="o")
        plt.title(f"Interpolated PR Curve – {model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid()
        plt.show()

    return {
        "Global": {
            "MAP": float(map_score),
            "MRR": float(mrr_score),
            "P@5": float(p5),
            "P@10": float(p10),
            "R-Precision": float(r_prec_mean)
        },
        "PerQuery": per_query,
        "PR_Curve": {
            "recall": r1.tolist(),
            "precision": p1.tolist()
        },
        "Interpolated_PR_Curve": {
            "recall": r2.tolist(),
            "precision": p2.tolist()
        }
    }



# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    model_files = {
    "BIR_no_relevance": r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\BIR_no_relevance.json",
    "BIR_with_relevance":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\BIR_with_relevance.json",
    "ExtendedBIR_no_relevance":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\ExtendedBIR_no_relevance.json",
    "ExtendedBIR_with_relevance":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\ExtendedBIR_with_relevance.json",
    "BM25":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\BM25.json",
    "VSM_Cosine":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\VSM_Cosine.json",
    "LSI_k100":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\LSI_k100.json",
    "LM_MLE":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\LM_MLE.json",
    "LM_Laplace":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\LM_Laplace.json",
    "LM_JelinekMercer":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\LM_JelinekMercer.json",
    "LM_Dirichlet":  r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\Results\LM_Dirichlet.json"
}


    all_rels = load_qrels(r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\Information-Retrival-Project-main\Information-Retrival-Project-main\SourceCode\data\processed\parse_preprocess\qrels.json")
    # Stocker les résultats finaux (pour tableau comparatif)
    final_results = {}

    for model_name, model_path in model_files.items():
        all_rankings = load_model_results(model_path)
        final_results[model_name] = evaluate_model(model_name, all_rankings, all_rels)

    # Sauvegarde JSON
    with open(r"C:\Users\X-LARGE\Desktop\M2 SII\TPs\RI\lab 6\evaluation_results.json", "w") as f:
        json.dump(final_results, f, indent=4)


    print("\n Résultats sauvegardés dans evaluation_results.json")
