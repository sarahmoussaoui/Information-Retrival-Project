import json
import os
import numpy as np

def load_qrels_json(qrels_path):
    """
    Charge les jugements de pertinence depuis ton fichier qrels.json
    Format: {"1": [13, 14, 15, ...], "2": [80, 90, ...], ...}
    Tous les documents listés ont pertinence = 1
    """
    print(f"Chargement de {qrels_path}...")
    with open(qrels_path, 'r') as f:
        data = json.load(f)
    
    relevance = {}
    
    # Ton format: {"1": [13, 14, 15, ...], ...}
    for query_id, doc_list in data.items():
        query_id = str(query_id)
        if query_id not in relevance:
            relevance[query_id] = {}
        
        # Tous les documents dans la liste ont pertinence = 1
        for doc_id in doc_list:
            relevance[query_id][str(doc_id)] = 1  # Score de pertinence = 1
    
    print(f"  {len(relevance)} requêtes chargées")
    total_docs = sum(len(docs) for docs in relevance.values())
    print(f"  {total_docs} documents pertinents (tous score=1)")
    
    return relevance

def create_ltr_dataset_with_relevance(model_json_paths, qrels_path, output_path="ltr_supervised_dataset.json"):
    """
    Crée un dataset LTR avec labels de pertinence (tous = 1 pour les documents pertinents)
    inclure TOUS les documents non pertinents (label=0)
    """
    # Charger les jugements de pertinence
    relevance = load_qrels_json(qrels_path)
    
    if not relevance:
        print("ERREUR: Aucun jugement de pertinence chargé!")
        return []
    
    # Charger tous les modèles
    all_models_data = {}
    model_names = []
    
    print("\nChargement des modèles...")
    for path in model_json_paths:
        if os.path.exists(path):
            model_name = os.path.basename(path).replace('.json', '')
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    model_data = json.load(f)
                all_models_data[model_name] = model_data
                model_names.append(model_name)
                print(f"  [OK] {model_name}")
            except Exception as e:
                print(f"  [ERREUR] {model_name}: {e}")
        else:
            print(f"  [NON TROUVE] {path}")
    
    if not all_models_data:
        print("ERREUR: Aucun modèle chargé!")
        return []
    
    # Créer le dataset
    ltr_dataset = []
    
    print("\nCréation du dataset LTR avec labels...")
    
    # Pour chaque requête dans les jugements
    for query_id in relevance.keys():
        # Documents pertinents pour cette requête
        relevant_docs = relevance[query_id]
        
        # Pour chaque document pertinent
        for doc_id, rel_score in relevant_docs.items():
            # Extraire les features = scores de tous les modèles
            features = []
            found_in_any_model = False
            
            for model_name in model_names:
                score = 0.0
                model_data = all_models_data[model_name]
                
                # Chercher le score dans les résultats du modèle
                if query_id in model_data.get("queries", {}):
                    for result in model_data["queries"][query_id]:
                        if str(result.get("doc_id", "")) == str(doc_id):
                            score = float(result.get("score", 0.0))
                            if score > 0:
                                found_in_any_model = True
                            break
                
                features.append(score)
            
            # Ajouter au dataset seulement si trouvé dans au moins un modèle
            if found_in_any_model:
                ltr_dataset.append({
                    "query_id": query_id,
                    "doc_id": doc_id,
                    "features": features,
                    "relevance_label": int(rel_score),  # Toujours 1 pour les documents pertinents
                    "model_names": model_names
                })
    
    # Ajouter TOUS les documents NON pertinents (label=0)
    print("\nAjout de TOUS les exemples non pertinents...")
    
    # Pour chaque requête dans tous les modèles
    for query_id in relevance.keys():
        # Ensemble de tous les documents déjà traités pour cette requête
        processed_docs = set()
        for sample in ltr_dataset:
            if sample["query_id"] == query_id:
                processed_docs.add(sample["doc_id"])
        
        # Parcourir tous les modèles pour collecter tous les documents uniques
        all_docs_for_query = set()
        for model_name, model_data in all_models_data.items():
            if query_id in model_data.get("queries", {}):
                for result in model_data["queries"][query_id]:
                    doc_id = str(result.get("doc_id", ""))
                    if doc_id:
                        all_docs_for_query.add(doc_id)
        
        # Pour chaque document unique qui n'est pas déjà dans le dataset
        for doc_id in all_docs_for_query:
            if doc_id not in processed_docs:
                # Vérifier si ce document n'est PAS dans la liste pertinente
                is_relevant = False
                if query_id in relevance:
                    # Vérifier si le document est dans les jugements de pertinence
                    if doc_id in relevance[query_id]:
                        is_relevant = True
                
                # Si le document n'est pas pertinent, l'ajouter
                if not is_relevant:
                    # Extraire les features
                    features = []
                    found_in_any_model = False
                    
                    for model_name in model_names:
                        score = 0.0
                        model_data = all_models_data[model_name]
                        
                        if query_id in model_data.get("queries", {}):
                            for result in model_data["queries"][query_id]:
                                if str(result.get("doc_id", "")) == str(doc_id):
                                    score = float(result.get("score", 0.0))
                                    if score > 0:
                                        found_in_any_model = True
                                    break
                        
                        features.append(score)
                    
                    # Ajouter comme exemple non pertinent
                    if found_in_any_model:
                        ltr_dataset.append({
                            "query_id": query_id,
                            "doc_id": doc_id,
                            "features": features,
                            "relevance_label": 0,  # Non pertinent
                            "model_names": model_names
                        })
    
    # Statistiques
    print(f"\nDataset créé: {len(ltr_dataset)} échantillons")
    
    if ltr_dataset:
        # Distribution des labels
        label_counts = {0: 0, 1: 0}
        for sample in ltr_dataset:
            label = sample["relevance_label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dataset": ltr_dataset,
                "feature_names": model_names,
                "total_samples": len(ltr_dataset),
                "relevance_distribution": label_counts
            }, f, indent=2)
        
        print(f"\n[SUCCES] Dataset LTR sauvegardé: {output_path}")
        print(f"   Échantillons: {len(ltr_dataset)}")
        print(f"   Distribution des labels:")
        for rel in [1, 0]:
            count = label_counts.get(rel, 0)
            percentage = (count / len(ltr_dataset) * 100) if ltr_dataset else 0
            if rel == 1:
                print(f"     Pertinents (1): {count} ({percentage:.1f}%)")
            else:
                print(f"     Non pertinents (0): {count} ({percentage:.1f}%)")
    
    else:
        print("[ATTENTION] Dataset vide! Vérifie tes données.")
    
    return ltr_dataset
# def train_ltr_classification_model(dataset_path):
#     """
#     Entraîne un modèle LTR de classification (pertinent vs non pertinent)
#     """
#     print("\n" + "="*60)
#     print("ENTRAÎNEMENT DU MODÈLE LTR (CLASSIFICATION)")
#     print("="*60)
    
#     # Charger le dataset
#     with open(dataset_path, 'r') as f:
#         data = json.load(f)
    
#     # Préparer X (features) et y (labels)
#     X = []
#     y = []
    
#     for sample in data["dataset"]:
#         X.append(sample["features"])
#         y.append(sample["relevance_label"])
    
#     X = np.array(X)
#     y = np.array(y)
    
#     print(f"X shape: {X.shape}")  # (n_samples, n_models)
#     print(f"y shape: {y.shape}")  # (n_samples,)
    
#     if len(X) == 0:
#         print("ERREUR: Dataset vide!")
#         return None
    
#     # Entraîner un modèle de classification
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.model_selection import train_test_split
#     from sklearn.metrics import classification_report, accuracy_score
    
#     # Séparation train/test
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y
#     )
    
#     print(f"\nDonnées d'entraînement: {X_train.shape[0]} échantillons")
#     print(f"Données de test: {X_test.shape[0]} échantillons")
    
#     # Entraîner
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(X_train, y_train)
    
#     # Prédictions
#     y_pred = model.predict(X_test)
    
#     # Évaluation
#     print(f"\nRésultats de classification:")
#     print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
#     print(f"  Rapport de classification:")
#     print(classification_report(y_test, y_pred, target_names=['Non pertinent', 'Pertinent']))
    
#     # Importance des features
#     print(f"\nImportance des modèles (coefficients):")
#     feature_names = data["feature_names"]
#     for i, (name, coef) in enumerate(zip(feature_names, model.coef_[0])):
#         print(f"  {i+1:2d}. {name:30s}: {coef:.6f}")
    
#     return model, data["feature_names"]

# def rank_documents_with_ltr(model, feature_names, query_results_dict):
#     """
#     Classe des documents avec le modèle LTR
    
#     :param query_results_dict: dict {model_name: [{"doc_id": "...", "score": ...}, ...]}
#     :return: documents classés par score LTR décroissant
#     """
#     # Créer un mapping doc_id -> scores de tous les modèles
#     doc_scores = {}
    
#     # Pour chaque document dans les résultats
#     for model_name, results in query_results_dict.items():
#         if model_name in feature_names:
#             model_idx = feature_names.index(model_name)
#             for result in results:
#                 doc_id = str(result["doc_id"])
#                 if doc_id not in doc_scores:
#                     doc_scores[doc_id] = [0.0] * len(feature_names)
#                 doc_scores[doc_id][model_idx] = float(result["score"])
    
#     # Prédire le score LTR pour chaque document
#     ranked_docs = []
#     for doc_id, features in doc_scores.items():
#         # Pour LogisticRegression, on veut la probabilité d'être pertinent
#         prob_pertinent = model.predict_proba([features])[0][1]  # Probabilité classe 1
#         ranked_docs.append({
#             "doc_id": doc_id,
#             "ltr_score": prob_pertinent,
#             "model_scores": dict(zip(feature_names, features))
#         })
    
#     # Trier par score LTR décroissant
#     ranked_docs.sort(key=lambda x: x["ltr_score"], reverse=True)
    
#     return ranked_docs

# ==========================================
# UTILISATION PRINCIPALE
# ==========================================

if __name__ == "__main__":
    # Tes chemins
    MODEL_JSON_PATHS = [
        r"SourceCode\Results\BIR_no_relevance.json",
        r"SourceCode\Results\BIR_with_relevance.json",
        r"SourceCode\Results\BM25.json",
        r"SourceCode\Results\ExtendedBIR_no_relevance.json",
        r"SourceCode\Results\ExtendedBIR_with_relevance.json",
        r"SourceCode\Results\LM_Dirichlet.json",
        r"SourceCode\Results\LM_Laplace.json",
        r"SourceCode\Results\LM_JelinekMercer.json",
        r"SourceCode\Results\LM_MLE.json",
        r"SourceCode\Results\LSI_k100.json",
        r"SourceCode\Results\VSM_Cosine.json",
    ]
    
    # Chemin vers qrels.json
    QRELS_PATH = r"SourceCode\data\processed\parse_preprocess\qrels.json"
    
    OUTPUT_PATH = "ltr_supervised_dataset.json"
    
    # 1. Créer le dataset
    print("="*60)
    print("CREATION DU DATASET LTR")
    print("="*60)
    
    dataset = create_ltr_dataset_with_relevance(
        MODEL_JSON_PATHS, 
        QRELS_PATH, 
        OUTPUT_PATH
    )
    
    # if dataset:
    #     # 2. Entraîner le modèle LTR (classification binaire)
    #     model, feature_names = train_ltr_classification_model(OUTPUT_PATH)
        
    #     # 3. Exemple d'utilisation pour ranking
    #     print("\n" + "="*60)
    #     print("EXEMPLE DE RANKING AVEC LTR")
    #     print("="*60)
        
    #     # Charger les résultats d'un modèle pour une requête
    #     example_query_id = "1"
        
    #     # Construire query_results_dict pour cette requête
    #     query_results_dict = {}
    #     for path in MODEL_JSON_PATHS[:3]:  # Prendre juste 3 modèles pour l'exemple
    #         model_name = os.path.basename(path).replace('.json', '')
    #         with open(path, 'r') as f:
    #             model_data = json.load(f)
    #         if example_query_id in model_data["queries"]:
    #             query_results_dict[model_name] = model_data["queries"][example_query_id][:10]  # 10 premiers
        
    #     if query_results_dict:
    #         # Classer avec LTR
    #         ranked_docs = rank_documents_with_ltr(model, feature_names, query_results_dict)
            
    #         print(f"\nClassement LTR pour la requête {example_query_id}:")
    #         for i, doc in enumerate(ranked_docs[:5], 1):
    #             print(f"{i}. Doc {doc['doc_id']} - Score LTR: {doc['ltr_score']:.4f}")
                
    #             # Afficher les scores des modèles individuels
    #             print(f"   Scores: ", end="")
    #             for model_name in feature_names[:3]:  # Afficher juste 3 modèles
    #                 if model_name in doc['model_scores']:
    #                     score = doc['model_scores'][model_name]
    #                     if score > 0:
    #                         print(f"{model_name}: {score:.2f} ", end="")
    #             print()