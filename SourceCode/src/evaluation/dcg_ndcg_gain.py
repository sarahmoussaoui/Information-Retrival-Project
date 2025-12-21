import numpy as np
import math
import json
from collections import defaultdict
import os
import math
import random



def dcg_at_k(relevance_scores, k=20, shuffle=True):
    """
    Calcule le DCG@k (Discounted Cumulative Gain).

    :param relevance_scores: Liste des scores de pertinence
    :param k: cutoff
    :param shuffle: si True, mélange aléatoirement les scores
    :return: DCG@k
    """
    scores = relevance_scores.copy()
    scores = scores[:k]
    # print("avant : ",scores)
    if shuffle:
        random.shuffle(scores)
    # print("apres : ",scores)


    if not scores:
        return 0.0

    # Premier élément sans discount
    dcg = scores[0]

    # Discount logarithmique
    for i, rel in enumerate(scores[1:], start=2):
        if rel > 0:
            dcg += rel / math.log2(i)

    return dcg


def ndcg_at_k(relevance_scores, k=20):
    """
    Calcule le nDCG@k (Normalized Discounted Cumulative Gain).
    
    :param relevance_scores: Liste des scores de pertinence
    :param k: Nombre de résultats à considérer
    :return: Valeur du nDCG@k
    """
    # DCG actuel
    dcg = dcg_at_k(relevance_scores, k)
    
    # DCG idéal : scores triés par ordre décroissant
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k,shuffle=False)
    
    # Éviter la division par zéro
    if ideal_dcg == 0:
        return 0.0
    
    return dcg / ideal_dcg

def calculate_gain_percentage(score1, score2):
    """
    Calcule le gain (%) entre deux séries de scores nDCG@20.
    
    :param score1: Liste des metriques pour un modele A 
    :param score2: Liste des metriques pour un modele B 
    :return: Liste des gains (%) pour chaque requête
    """
    gains = []
    
    for score1_val, score2_val in zip(score1, score2):
        if score1_val == 0:
            if score2_val > 0:
                gains.append(float('inf'))  # Gain infini (de 0 à >0)
            else:
                gains.append(0.0)
        else:
            gain = ((score2_val - score1_val) / score1_val) * 100
            gains.append(gain)
    
    return gains


def extract_scores_from_model_results(model_results, top_k=20):
    """
    Extrait uniquement les scores du modèle 
    :param model_results: Liste de dicts [{"doc_id": "180", "score": 3.97}, ...]
    :param top_k: Nombre de résultats à considérer
    :return: Liste des scores normalisés
    """
    top_results = model_results[:top_k]
    # Extraire juste les scores de similarité/pertinence
    scores = [result["score"] for result in top_results]
    return scores

def evaluate_single_model(model_data, model_name, output_dir="metrics"):
    """
    Évalue un seul modèle avec seulement DCG et nDCG
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {
        "model_name": model_name,
        "query_metrics": {},
        "overall_metrics": {}
    }
    
    # Pour chaque requête
    for query_id, query_results in model_data["queries"].items():
        # Extraire les scores du modèle 
        model_scores = extract_scores_from_model_results(query_results, top_k=20)
        
        # Calculer DCG et nDCG sur les scores du modèle
        # Pour nDCG, on considère que les scores du modèle sont des "scores de pertinence"
        dcg_20 = dcg_at_k(model_scores, k=20)
        ndcg_20 = ndcg_at_k(model_scores, k=20)
        
        # Ajouter au dictionnaire
        metrics["query_metrics"][query_id] = {
            "dcg@20": float(dcg_20),
            "ndcg@20": float(ndcg_20),
            "model_scores": model_scores[:20]  # Garder les 20 premiers scores
        }
    
    # Calculer les moyennes
    all_ndcg = [q["ndcg@20"] for q in metrics["query_metrics"].values()]
    all_dcg = [q["dcg@20"] for q in metrics["query_metrics"].values()]
    
    metrics["overall_metrics"] = {
        "mean_ndcg@20": float(sum(all_ndcg) / len(all_ndcg) if all_ndcg else 0),
        "mean_dcg@20": float(sum(all_dcg) / len(all_dcg) if all_dcg else 0),
        "total_queries_evaluated": len(metrics["query_metrics"])
    }
    
    # Sauvegarder en JSON
    output_file = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f" Métriques sauvegardées pour {model_name}: {output_file}")
    
    return metrics

def evaluate_all_models(json_paths, output_dir="metrics"):
    """
    Évalue tous les modèles à partir des fichiers JSON
    """
    all_metrics = {}
    
    for json_path in json_paths:
        # Charger les résultats du modèle
        with open(json_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        
        # Extraire le nom du modèle
        model_name = model_data.get("model", os.path.basename(json_path).replace('.json', ''))
        
        print(f"\n{'='*50}")
        print(f"Évaluation du modèle: {model_name}")
        print(f"Fichier: {json_path}")
        
        # Évaluer le modèle
        metrics = evaluate_single_model(
            model_data, 
            model_name, 
            output_dir
        )
        
        all_metrics[model_name] = metrics
        
        # Afficher un résumé
        print(f"  nDCG@20 moyen: {metrics['overall_metrics']['mean_ndcg@20']:.4f}")
        print(f"  DCG@20 moyen: {metrics['overall_metrics']['mean_dcg@20']:.4f}")
    
    # Sauvegarder toutes les métriques
    all_metrics_file = os.path.join(output_dir, "all_models_comparison.json")
    with open(all_metrics_file, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f" Toutes les métriques sauvegardées dans: {output_dir}/")
    print(f" Fichier de comparaison: {all_metrics_file}")
    
    return all_metrics

def create_comparison_report(all_metrics, output_dir="metrics"):
    """
    Crée un rapport de comparaison entre tous les modèles
    """
    comparison = {
        "model_comparison": {},
        "ranking_by_ndcg": [],
        "ranking_by_dcg": [],
        "gains_comparison": {}
    }
    
    model_names = list(all_metrics.keys())
    
    # Comparaison des métriques globales
    for model_name, metrics in all_metrics.items():
        comparison["model_comparison"][model_name] = {
            "mean_ndcg@20": metrics["overall_metrics"]["mean_ndcg@20"],
            "mean_dcg@20": metrics["overall_metrics"]["mean_dcg@20"]
        }
    
    # Classement par nDCG
    sorted_by_ndcg = sorted(
        model_names,
        key=lambda x: all_metrics[x]["overall_metrics"]["mean_ndcg@20"],
        reverse=True
    )
    comparison["ranking_by_ndcg"] = [
        {
            "rank": i+1,
            "model": model, 
            "mean_ndcg@20": all_metrics[model]["overall_metrics"]["mean_ndcg@20"]
        }
        for i, model in enumerate(sorted_by_ndcg)
    ]
    
    # Classement par DCG
    sorted_by_dcg = sorted(
        model_names,
        key=lambda x: all_metrics[x]["overall_metrics"]["mean_dcg@20"],
        reverse=True
    )
    comparison["ranking_by_dcg"] = [
        {
            "rank": i+1,
            "model": model, 
            "mean_dcg@20": all_metrics[model]["overall_metrics"]["mean_dcg@20"]
        }
        for i, model in enumerate(sorted_by_dcg)
    ]
    
    # Calcul des gains (%) pour les 10 premières requêtes (nDCG)
    for i, model_a in enumerate(model_names):
        for model_b in model_names[i+1:]:
            key = f"{model_a}_vs_{model_b}"
            
            # Récupérer les nDCG@20 pour les requêtes 1 à 10
            ndcg_a = []
            ndcg_b = []
            
            for q in range(1, 11):
                query_id = str(q)
                if query_id in all_metrics[model_a]["query_metrics"]:
                    ndcg_a.append(all_metrics[model_a]["query_metrics"][query_id]["ndcg@20"])
                if query_id in all_metrics[model_b]["query_metrics"]:
                    ndcg_b.append(all_metrics[model_b]["query_metrics"][query_id]["ndcg@20"])
            
            # Calculer les gains
            gains = calculate_gain_percentage(ndcg_a, ndcg_b)
            
            comparison["gains_comparison"][key] = {
                "model_a": model_a,
                "model_b": model_b,
                "mean_ndcg_a": sum(ndcg_a) / len(ndcg_a) if ndcg_a else 0,
                "mean_ndcg_b": sum(ndcg_b) / len(ndcg_b) if ndcg_b else 0,
                "gains_per_query": {
                    f"I{q}": (float('inf') if gains[q-1] == float('inf') else float(gains[q-1])) 
                    for q in range(1, len(gains)+1)
                },
                "mean_gain": sum(g for g in gains if isinstance(g, (int, float)) and not math.isinf(g)) / len(gains) if gains else 0
            }
    
    # Sauvegarder le rapport
    report_file = os.path.join(output_dir, "comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f" Rapport de comparaison sauvegardé: {report_file}")
    
    return comparison

# ==========================================
# UTILISATION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    # 1. Liste de tous tes fichiers JSON de résultats
    MODEL_JSON_PATHS = [
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\BIR_no_relevance.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\BIR_with_relevance.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\BM25.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\ExtendedBIR_no_relevance.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\ExtendedBIR_with_relevance.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\LM_Dirichlet.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\LM_Laplace.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\LM_JelinekMercer.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\LM_MLE.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\LSI_k100.json",
        r"C:\Users\dsuhs\Desktop\SII\RI\Projet\Information-Retrival-Project\SourceCode\Results\VSM_Cosine.json",        
    ]
    
    # 2. Évaluer tous les modèles
    print(" DÉBUT DE L'ÉVALUATION")
    print(f"Nombre de modèles à évaluer: {len(MODEL_JSON_PATHS)}")
    
    # Créer le dossier pour les résultats
    OUTPUT_DIR = "evaluation_results_jdid"
    
    # Évaluer tous les modèles
    all_metrics = evaluate_all_models(
        MODEL_JSON_PATHS,
        output_dir=OUTPUT_DIR
    )
    
    # 3. Créer un rapport de comparaison
    print("\n CRÉATION DU RAPPORT DE COMPARAISON")
    comparison_report = create_comparison_report(all_metrics, OUTPUT_DIR)
    
    # 4. Afficher un résumé dans la console
    print("\n" + "="*60)
    print(" CLASSEMENT FINAL DES MODÈLES")
    print("="*60)
    
    print("\nPar nDCG@20 moyen:")
    for entry in comparison_report["ranking_by_ndcg"]:
        print(f"  {entry['rank']}. {entry['model']}: {entry['mean_ndcg@20']:.4f}")
    
    print("\nPar DCG@20 moyen:")
    for entry in comparison_report["ranking_by_dcg"]:
        print(f"  {entry['rank']}. {entry['model']}: {entry['mean_dcg@20']:.4f}")
    
    print("\n Évaluation terminée avec succès!")
    print(f" Tous les résultats sont dans: {OUTPUT_DIR}/")