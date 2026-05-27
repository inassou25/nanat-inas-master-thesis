# Prédiction du churn e-commerce par enrichissement RAG des avis clients

Mémoire de master — Code source complet du pipeline expérimental.

## Problématique

Dans quelle mesure une extraction sémantique structurée via un pipeline RAG améliore-t-elle la prédiction du churn client par rapport aux données structurées seules ou aux embeddings contextuels ?

## Données

Dataset **Olist** (e-commerce brésilien) — disponible sur [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Les fichiers de données ne sont pas inclus dans ce repo.

Variable cible : `churn` = client n'ayant pas passé de commande dans les 6 mois suivant son dernier achat.

## Architecture du projet

```
01_data_preparation/    → Split temporel strict (70/15/15)
02_rag_pipeline/        → Knowledge base + extraction RAG (GPT-4o mini + FAISS)
03_models/              → Entraînement et sélection des hyperparamètres
04_evaluation/          → Évaluation sur le test set, tests statistiques
05_figures/             → Figures du mémoire
utils/                  → Vérifications et diagnostics
```

## Les 4 méthodes comparées

| Méthode | Description | Algorithmes |
|---------|-------------|-------------|
| **M1** | Features structurées seules (RFM, livraison, paiement) | LR, XGBoost |
| **M2** | M1 + embeddings de reviews (`paraphrase-multilingual-MiniLM-L12-v2`) | LR, XGBoost |
| **M3** | M1 + scores RAG sur 4 dimensions (livraison, qualité, service, prix) | LR, XGBoost |
| **M4** | Zero-shot GPT-4o mini (probabilité de churn directe) | — |

M3 et M4 utilisent l'API OpenAI. M3 est entraîné sur 1 500 observations (contrainte budget API).

## Pipeline RAG (M3)

Pour chaque avis client :
1. **Retrieval** : FAISS récupère les *k* passages les plus proches depuis la base de connaissances académique (`knowledge_base.py`)
2. **Generation** : GPT-4o mini score l'avis sur 4 dimensions [0, 1] en s'appuyant sur le contexte récupéré
3. Ces 4 scores + 4 indicateurs binaires ("dimension mentionnée") deviennent des features de classification

La base de connaissances contient 43 passages issus de 6 articles académiques sur la satisfaction e-commerce (Masyhuri 2022, Al-Ayed 2022, Nguyen & Chanut 2018, De Caigny et al. 2020, Tirunillai & Tellis 2014, Archak et al. 2011).

## Ordre d'exécution

```bash
# 1. Préparer les données
python 01_data_preparation/split_temporel.py

# 2. Valider et calibrer le pipeline RAG
python 02_rag_pipeline/sensitivity_k.py        # Choisir k optimal
python 02_rag_pipeline/validation_rag.py       # Vérifier la cohérence des scores

# 3. Entraîner les modèles (grid search sur val)
python 03_models/train_val_m1m2.py             # M1 et M2
python 03_models/train_val_m3.py               # M3
python 03_models/seuils_optimaux.py            # Seuils F1 optimaux + M4 zero-shot

# 4. Évaluer sur le test set
python 04_evaluation/evaluation_test_final.py  # Résultats principaux
python 04_evaluation/controle_taille_m2.py     # Contrôle : M2 sur 1500 obs
python 04_evaluation/robustesse_regression.py  # Test de robustesse (Wilcoxon)

# 5. Générer les figures
python 05_figures/distribution_notes.py
python 05_figures/learning_curves_m1m2.py
python 05_figures/feature_importance.py
python 05_figures/calibration_curve.py
```

## Installation

```bash
pip install -r requirements.txt
```

Clé API OpenAI requise pour M3 et M4 :

```bash
export OPENAI_API_KEY="votre_clé"
```

## Métriques d'évaluation

- **PR-AUC** (métrique principale) — adaptée aux classes déséquilibrées (~19% churn)
- ROC-AUC, F1, Précision, Rappel
- IC à 95% par bootstrap (1 000 itérations)
- Tests de permutation pour la significativité des différences entre modèles

## Structure des fichiers de données attendus

Les scripts s'attendent à trouver les données dans le même dossier que le script, via la variable `save_path`. Modifier cette variable en tête de chaque script selon votre arborescence locale.
