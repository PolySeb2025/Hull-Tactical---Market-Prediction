# Hull Tactical - Market Prediction 📈🤖

Ce projet implémente un pipeline de Machine Learning avancé pour la prédiction de mouvements de marché (Market Prediction). Il s'appuie sur des techniques de finance quantitative pour traiter des séries temporelles non stationnaires et prédire les rendements futurs.

## 🚀 Fonctionnalités Clés

* **Différenciation Fractionnaire (FFD) :** Transformation des séries temporelles pour les rendre stationnaires tout en préservant la mémoire à long terme, implémentée dans `src/features.py`.
* **Ensemble Modeling :** Utilisation combinée de **XGBoost** et **LightGBM** pour la classification.
* **Inférence Temps Réel :** Une classe `Model` optimisée pour la production, capable de calculer les features FFD de manière incrémentale sur des flux de données "live".
* **Feature Engineering Avancé :** Sélection automatique des 20 meilleures features pour éviter le surapprentissage.
* **Analyse de Stationnarité :** Tests Augmented Dickey-Fuller (ADF) pour valider la robustesse des features.

## 📂 Structure du Projet

```text
hull_mldp_project/
├── data/
│   ├── raw/                # Données brutes (train.csv, test.csv)
│   └── processed/          # Données transformées (parquet)
├── notebooks/
│   ├── exploration.ipynb   # Analyse exploratoire et tests ADF
│   ├── features_ffd.ipynb  # Création des features fractionnaires
│   ├── labeling.ipynb      # Création des cibles (Targets)
│   └── training2.ipynb     # Pipeline final : Split, Sélection, Ensemble Learning
├── src/
│   ├── features.py         # Algorithmes FFD (Fixed Window)
│   ├── labeling.py         # Logique de labeling
│   └── sampling.py         # Poids d'échantillonnage (Uniqueness)
└── submission/
    ├── model.py            # Script d'inférence pour la production
    ├── features_list.pkl   # Liste des features retenues
    └── model_lpd.pkl       # Modèle sérialisé
```

## 🛠️ Installation

Assurez-vous d'avoir Python 3.10+ et installez les dépendances nécessaires :


```bash
pip install pandas numpy scikit-learn xgboost lightgbm statsmodels pyarrow joblib matplotlib seaborn
```

## ⚡ Workflow

Pour reproduire le pipeline complet, exécutez les notebooks dans l'ordre suivant :

    Exploration (exploration.ipynb) : Chargement des données et analyse de la stationnarité des variables brutes.

    Feature Engineering : Calcul des features FFD pour capturer la mémoire du marché sans sacrifier la stationnarité.

    Entraînement (training3.ipynb) :

        Calcul des poids d'échantillons.

        Sélection de features.

        Entraînement de l'ensemble (XGBoost + LightGBM).

        Génération du fichier de soumission.

## 🧠 Détails du Modèle

Le modèle final (submission/model.py) est conçu pour être déployé. Il intègre une mémoire historique (self.history) permettant de recalculer
les indicateurs techniques complexes (comme le FFD) à chaque nouveau point de donnée reçu, garantissant une cohérence parfaite entre l'entraînement et l'inférence.



