import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os

class RobustPointwiseLTR:
    """
    LTR Pointwise robuste avec gestion des overflow et déséquilibre
    """
    
    def __init__(self, n_iter=1000, reg_lambda=0.01, class_weight=None):
        self.n_iter = n_iter
        self.reg_lambda = reg_lambda
        self.class_weight = class_weight
        self.weights = None
        self.bias = None
        self.X_mean = None
        self.X_std = None
        self.loss_history = []
    
    def _safe_sigmoid(self, x):
        """Sigmoid numericalement stable"""
        x = np.clip(x, -50, 50)  # Éviter overflow
        return 1 / (1 + np.exp(-x))
    
    def _safe_log_loss(self, y_true, y_pred):
        """Log loss numericalement stable"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def _normalize_features(self, X, fit=True):
        """Normalisation robuste"""
        if fit:
            self.X_mean = X.mean(axis=0)
            self.X_std = X.std(axis=0)
            self.X_std[self.X_std == 0] = 1  # Éviter division par 0
        return (X - self.X_mean) / self.X_std
    
    def train(self, X, y, verbose=True):
        """Entraîne le modèle avec descente de gradient robuste"""
        # 1. Sous-échantillonnage pour équilibrer (CRUCIAL)
        print("Sous-échantillonnage pour équilibrer les classes...")
        
        # Indices des classes
        idx_class_0 = np.where(y == 0)[0]
        idx_class_1 = np.where(y == 1)[0]
        
        # Garder tous les exemples de classe 1
        # Prendre un échantillon aléatoire de classe 0 (même taille que classe 1)
        n_class_1 = len(idx_class_1)
        n_samples = min(len(idx_class_0), n_class_1 * 2)  # Ratio 2:1
        
        np.random.seed(42)
        idx_class_0_sampled = np.random.choice(idx_class_0, n_samples, replace=False)
        
        # Combiner les indices
        idx_balanced = np.concatenate([idx_class_0_sampled, idx_class_1])
        np.random.shuffle(idx_balanced)
        
        X_balanced = X[idx_balanced]
        y_balanced = y[idx_balanced]
        
        print(f"Avant équilibrage: {len(y)} échantillons (classe 1: {n_class_1})")
        print(f"Après équilibrage: {len(y_balanced)} échantillons")
        print(f"Nouvelle distribution: Classe 0: {sum(y_balanced == 0)}, Classe 1: {sum(y_balanced == 1)}")
        
        # 2. Normalisation
        X_norm = self._normalize_features(X_balanced, fit=True)
        
        # Ajouter biais
        X_bias = np.c_[X_norm, np.ones(X_norm.shape[0])]
        m, n = X_bias.shape
        
        # 3. Initialisation avec petites valeurs
        self.weights = np.random.randn(n - 1) * 0.01
        self.bias = 0.0
        
        # 4. Descente de gradient avec learning rate adaptatif
        learning_rate = 0.1
        best_loss = float('inf')
        patience = 0
        max_patience = 100
        
        for i in range(self.n_iter):
            # Calcul forward
            z = X_norm.dot(self.weights) + self.bias
            y_pred = self._safe_sigmoid(z)
            
            # Calcul du gradient
            error = y_pred - y_balanced
            grad_w = (1/m) * X_norm.T.dot(error) + (self.reg_lambda/m) * self.weights
            grad_b = (1/m) * np.sum(error)
            
            # Mise à jour avec momentum
            self.weights -= learning_rate * grad_w
            self.bias -= learning_rate * grad_b
            
            # Calcul de la loss
            loss = self._safe_log_loss(y_balanced, y_pred)
            loss += (self.reg_lambda/(2*m)) * np.sum(self.weights**2)
            
            self.loss_history.append(loss)
            
            # Ajustement du learning rate
            if loss < best_loss:
                best_loss = loss
                patience = 0
            else:
                patience += 1
                if patience >= max_patience:
                    learning_rate *= 0.5
                    patience = 0
            
            # Affichage
            if verbose and (i % 100 == 0 or i == self.n_iter - 1):
                print(f"Iteration {i}: Loss = {loss:.6f}, LR = {learning_rate:.6f}")
            
            # Arrêt précoce
            if learning_rate < 1e-6:
                print(f"Arrêt précoce à l'itération {i}")
                break
        
        if verbose:
            print(f"Entraînement terminé - Loss finale: {self.loss_history[-1]:.6f}")
    
    def predict_proba(self, X):
        """Retourne les probabilités"""
        if self.weights is None:
            raise ValueError("Modèle non entraîné")
        
        X_norm = (X - self.X_mean) / self.X_std
        z = X_norm.dot(self.weights) + self.bias
        return self._safe_sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Prédiction binaire"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int), proba
    
    def get_model_importance(self, model_names):
        """Importance des modèles"""
        importance = np.abs(self.weights)
        return {name: imp for name, imp in zip(model_names, importance)}

def load_and_prepare_data(dataset_path):
    """Charge et prépare les données"""
    print(f"Chargement du dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    dataset = data["dataset"]
    model_names = data["feature_names"]
    
    # Organiser par requête
    queries_data = {}
    for sample in dataset:
        query_id = sample["query_id"]
        if query_id not in queries_data:
            queries_data[query_id] = []
        
        queries_data[query_id].append({
            "doc_id": sample["doc_id"],
            "features": sample["features"],
            "relevance_label": sample["relevance_label"]
        })
    
    print(f"  {len(dataset)} échantillons")
    print(f"  {len(queries_data)} requêtes uniques")
    print(f"  {len(model_names)} modèles")
    
    return dataset, queries_data, model_names


def split_by_queries_from_dataset(dataset, test_size=0.2, random_state=42):
    """
    Split TRAIN / TEST par requêtes à partir du dataset brut
    """
    # 1. Extraire les query_id uniques
    query_ids = list({sample["query_id"] for sample in dataset})

    # 2. Split des requêtes
    train_q, test_q = train_test_split(
        query_ids,
        test_size=test_size,
        random_state=random_state
    )

    train_q = set(train_q)
    test_q = set(test_q)

    # 3. Construire les ensembles
    X_train, y_train = [], []
    X_test, y_test = [], []

    for sample in dataset:
        if sample["query_id"] in train_q:
            X_train.append(sample["features"])
            y_train.append(sample["relevance_label"])
        else:
            X_test.append(sample["features"])
            y_test.append(sample["relevance_label"])

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_test), np.array(y_test),
        train_q, test_q
    )

def train_ltr_model_simple(dataset, model_names):
    """Entraîne le modèle LTR de manière robuste"""
    print("\n" + "="*60)
    print("ENTRAÎNEMENT DU MODÈLE LTR (VERSION ROBUSTE)")
    print("="*60)
    
    # Préparer X et y
    X = np.array([sample["features"] for sample in dataset])
    y = np.array([sample["relevance_label"] for sample in dataset])
    
    # Statistiques
    unique, counts = np.unique(y, return_counts=True)
    print(f"Distribution des labels:")
    for label, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"  Classe {label}: {count} ({percentage:.1f}%)")
    
    # Split train/test (sur données originales pour évaluation réaliste)
    X_train, y_train, X_test, y_test, train_q, test_q = split_by_queries_from_dataset(
    dataset,
    test_size=0.2,
    random_state=42
    )

    print(f"Requêtes train: {len(train_q)}")
    print(f"Requêtes test:  {len(test_q)}")
    print(f"Docs train:    {len(y_train)}")
    print(f"Docs test:     {len(y_test)}")


    # Entraîner le modèle
    model = RobustPointwiseLTR(
        n_iter=500,
        reg_lambda=0.1,
        class_weight=None
    )
    
    model.train(X_train, y_train, verbose=True)
    
    # Évaluation
    y_pred, y_prob = model.predict(X_test, threshold=0.5)
    
    print(f"\nÉvaluation sur le test set:")
    print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
    
    # Rapport détaillé
    print("\nRapport de classification:")
    print(classification_report(y_test, y_pred, 
                                target_names=['Non pertinent', 'Pertinent']))
    
    # Importance des modèles
    importance = model.get_model_importance(model_names)
    print(f"\nTop 5 modèles les plus importants:")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
    for model_name, imp in sorted_imp:
        print(f"  {model_name}: {imp:.6f}")
    
    return model

def rank_all_queries(model, queries_data, model_names):
    """Classe tous les documents pour toutes les requêtes"""
    print("\n" + "="*60)
    print("RANKING DE TOUTES LES REQUÊTES")
    print("="*60)
    
    all_rankings = {}
    
    for query_id, docs in queries_data.items():
        if len(docs) == 0:
            continue
        
        # Extraire features
        features = np.array([doc["features"] for doc in docs])
        doc_ids = [doc["doc_id"] for doc in docs]
        true_labels = [doc["relevance_label"] for doc in docs]
        
        # Calculer les scores LTR
        probabilities = model.predict_proba(features)
        
        # Créer et trier les résultats
        scored_docs = []
        for i in range(len(docs)):
            scored_docs.append({
                'score': float(probabilities[i]),
                'doc_id': doc_ids[i],
                'label': true_labels[i]
            })
        
        # Trier par score décroissant
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        
        # Ajouter les rangs
        results = []
        for rank, item in enumerate(scored_docs, 1):
            results.append({
                'doc_id': item['doc_id'],
                'ltr_score': item['score'],
                'relevance_label': item['label'],
                'rank': rank
            })
        
        all_rankings[query_id] = results
        
        # Progression
        if int(query_id) % 3 == 0:
            print(f"  Requête {query_id}: {len(results)} documents classés")
    
    print(f"\nRanking terminé pour {len(all_rankings)} requêtes")
    return all_rankings

def save_all_results(all_rankings, model_names, model, output_dir="ltr_results_robust"):
    """Sauvegarde tous les résultats"""
    print("\n" + "="*60)
    print("SAUVEGARDE DES RÉSULTATS")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Fichier principal
    main_output = {
        "model": "Robust_Pointwise_LTR",
        "total_queries": len(all_rankings),
        "feature_models": model_names,
        "model_weights": [float(w) for w in model.weights] if model.weights is not None else [],
        "model_bias": float(model.bias) if model.bias is not None else 0.0,
        "rankings": all_rankings
    }
    
    main_file = os.path.join(output_dir, "ltr_all_rankings.json")
    with open(main_file, 'w', encoding='utf-8') as f:
        json.dump(main_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fichier principal: {main_file}")
    
    # 2. Fichier par requête (simplifié)
    for query_id, rankings in all_rankings.items():
        query_file = os.path.join(output_dir, f"query_{query_id}.json")
        query_output = {
            "query_id": query_id,
            "total_documents": len(rankings),
            "top_20": rankings[:20]
        }
        
        with open(query_file, 'w', encoding='utf-8') as f:
            json.dump(query_output, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Fichiers individuels: {len(all_rankings)} fichiers créés")
    
    # 3. Statistiques
    stats = {
        "total_queries": len(all_rankings),
        "total_documents": sum(len(r) for r in all_rankings.values()),
        "queries_with_results": len([r for r in all_rankings.values() if len(r) > 0])
    }
    
    stats_file = os.path.join(output_dir, "statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Statistiques: {stats_file}")
    
    return main_file

def main():
    """Fonction principale"""
    
    DATASET_PATH = r"C:\Users\dsuhs\Desktop\SII\RI\Projet\ltr_supervised_dataset.json"
    OUTPUT_DIR = "ltr_final_results_v2"
    
    print("="*70)
    print("LTR ROBUSTE - AVEC GESTION DU DÉSÉQUILIBRE")
    print("="*70)
    
    try:
        # 1. Charger les données
        dataset, queries_data, model_names = load_and_prepare_data(DATASET_PATH)
        
        # 2. Entraîner le modèle
        model = train_ltr_model_simple(dataset, model_names)
        
        # 3. Ranker toutes les requêtes
        all_rankings = rank_all_queries(model, queries_data, model_names)
        
        # 4. Sauvegarder
        main_file = save_all_results(all_rankings, model_names, model, OUTPUT_DIR)
        
        # 5. Afficher un exemple
        print(f"\nExemple - Requête 1 (top 5):")
        if "1" in all_rankings:
            for i, r in enumerate(all_rankings["1"][:5], 1):
                pertinence = "✓" if r["relevance_label"] == 1 else "✗"
                print(f"  {i}. Doc {r['doc_id']} - Score: {r['ltr_score']:.4f} {pertinence}")
        
        print(f"\n" + "="*70)
        print("TERMINÉ ! Résultats sauvegardés.")
        print("="*70)
        
    except Exception as e:
        print(f"\n[ERREUR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





# import numpy as np
# from sklearn.linear_model import Ridge
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split

# class PointwiseLTR:
#     """
#     Learning to Rank - Approche Pointwise avec Régression
#     """
    
#     def __init__(self, model_type='ridge'):
#         """
#         :param model_type: 'ridge' (linéaire) ou 'rf' (random forest)
#         """
#         if model_type == 'ridge':
#             self.model = Ridge(alpha=1.0)
#         elif model_type == 'rf':
#             self.model = RandomForestRegressor(n_estimators=100, random_state=42)
#         else:
#             raise ValueError("model_type doit être 'ridge' ou 'rf'")
    
#     def prepare_features(self, query_doc_features, relevance_scores):
#         """
#         Prépare les données pour l'entraînement
        
#         :param query_doc_features: Liste de listes, chaque sous-liste = features pour (requête, document)
#         :param relevance_scores: Liste des scores de pertinence correspondants
#         :return: X (features), y (labels)
#         """
#         X = np.array(query_doc_features)
#         y = np.array(relevance_scores)
#         return X, y
    
#     def train(self, X, y):
#         """Entraîne le modèle"""
#         self.model.fit(X, y)
    
#     def predict(self, X):
#         """Prédit les scores pour de nouvelles paires (requête, document)"""
#         return self.model.predict(X)
    
#     def rank_documents(self, query_features_list):
#         """
#         Classe les documents pour une requête
        
#         :param query_features_list: Liste de features pour chaque document candidat
#         :return: indices des documents classés par ordre décroissant de score
#         """
#         scores = self.predict(query_features_list)
#         ranked_indices = np.argsort(scores)[::-1]  # Tri décroissant
#         return ranked_indices, scores[ranked_indices]

# # ==========================================
# # EXEMPLE D'UTILISATION
# # ==========================================

# if __name__ == "__main__":
#     # 1. Créer des données d'exemple
#     # Suppose que pour chaque (requête, document), tu as extrait 5 features
#     # Ex: [score_VSM, score_BM25, score_LM, TF, doc_length]
#     X_train = [
#         [0.85, 0.92, 0.78, 15, 200],  # Requête 1, Doc 1
#         [0.45, 0.60, 0.35, 8,  150],  # Requête 1, Doc 2
#         [0.75, 0.88, 0.72, 12, 180],  # Requête 2, Doc 1
#         [0.30, 0.40, 0.25, 5,  120],  # Requête 2, Doc 2
#         # ... plus de données
#     ]
    
#     # Scores de pertinence réels (de MED.REL: 0,1,2,3)
#     y_train = [3, 1, 2, 0]
    
#     # 2. Initialiser et entraîner le LTR
#     ltr = PointwiseLTR(model_type='ridge')  # ou 'rf' pour Random Forest
    
#     X, y = ltr.prepare_features(X_train, y_train)
#     ltr.train(X, y)
    
#     print("Modèle LTR entraîné ✓")
    
#     # 3. Utiliser pour ranking
#     # Nouvelles paires (requête, document) à classer
#     new_query_features = [
#         [0.80, 0.90, 0.75, 14, 190],  # Doc candidat 1
#         [0.50, 0.65, 0.40, 9,  160],  # Doc candidat 2
#         [0.20, 0.30, 0.15, 3,  100],  # Doc candidat 3
#     ]
    
#     ranked_indices, ranked_scores = ltr.rank_documents(new_query_features)
    
#     print("\nDocuments classés :")
#     for i, (idx, score) in enumerate(zip(ranked_indices, ranked_scores), 1):
#         print(f"{i}. Document {idx+1} - Score LTR: {score:.4f}")
    
#     # 4. Intégration simple avec tes modèles existants
#     def get_ltr_features_for_query(query_id, doc_ids, model_scores_dict):
#         """
#         Récupère les features pour le LTR à partir des scores des modèles existants
        
#         :param query_id: ID de la requête
#         :param doc_ids: Liste d'IDs de documents
#         :param model_scores_dict: Dict {'VSM': scores, 'BM25': scores, ...}
#         :return: Liste de features pour chaque document
#         """
#         features_list = []
        
#         for doc_id in doc_ids:
#             features = []
#             # Ajoute les scores de chaque modèle
#             for model_name in ['VSM', 'BM25', 'LM_Dirichlet', 'BIR']:
#                 if model_name in model_scores_dict:
#                     features.append(model_scores_dict[model_name][doc_id])
            
#             # Ajoute d'autres features si disponibles
#             # features.append(tf_value)
#             # features.append(doc_length)
            
#             features_list.append(features)
        
#         return features_list

# # Exemple d'utilisation avec tes modèles
# # model_scores = {
# #     'VSM': {doc1: 0.8, doc2: 0.5, ...},
# #     'BM25': {doc1: 0.9, doc2: 0.6, ...},
# #     'LM_Dirichlet': {doc1: 0.7, doc2: 0.4, ...},
# # }
# # 
# # features = get_ltr_features_for_query('I1', ['doc1', 'doc2', 'doc3'], model_scores)
# # ranked_docs = ltr.rank_documents(features)