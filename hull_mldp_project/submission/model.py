import os
import joblib
import pandas as pd
import numpy as np

# --- 1. CONFIGURATION ---
# On charge les objets sauvegardés
# (Assure-toi que les fichiers .pkl sont dans le même dossier lors de la soumission)
MODEL_PATH = "model_lpd.pkl"
FEATURES_PATH = "features_list.pkl"

class Model:
    def __init__(self):
        # Chargement du modèle
        self.model = joblib.load(MODEL_PATH)
        self.features = joblib.load(FEATURES_PATH)
        
        # MEMOIRE pour le FFD
        # Pour calculer le FFD d'aujourd'hui, on a besoin des prix d'hier, avant-hier...
        # On va stocker l'historique récent ici.
        self.history = pd.DataFrame()
        self.min_history_len = 50 # Fenêtre minimale pour le FFD

    def frac_diff_ffd_realtime(self, new_value, column_name, d=0.1, thres=1e-5):
        """
        Calcul du FFD incrémental.
        """
        # Récupérer l'historique pour cette colonne
        if column_name not in self.history.columns:
            return np.nan
            
        series = self.history[column_name].copy()
        
        # Ajouter la nouvelle valeur temporairement
        series = pd.concat([series, pd.Series([new_value])], ignore_index=True)
        
        # Calculer les poids (Optimisation possible: pré-calculer les poids)
        w, k = [1.], 1
        while True:
            w_ = -w[-1] / k * (d - k + 1)
            if abs(w_) < thres: break
            w.append(w_)
            k += 1
        w = np.array(w[::-1]).reshape(-1, 1)
        width = len(w) - 1
        
        if len(series) < width + 1:
            return np.nan # Pas assez d'historique
            
        # Produit scalaire sur la fin
        # On prend juste la dernière fenêtre
        window = series.values[-(width+1):]
        return np.dot(w.T, window)[0]

    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fonction appelée par Kaggle à chaque nouveau jour.
        input_df contient une seule ligne (ou un petit batch).
        """
        # 1. Mise à jour de l'historique avec les nouvelles données brutes
        # On concatène les nouvelles lignes à notre mémoire
        self.history = pd.concat([self.history, input_df], axis=0)
        
        # On ne garde que les X derniers jours pour ne pas exploser la RAM
        if len(self.history) > 200:
            self.history = self.history.iloc[-200:]
            
        # 2. Construction des Features (Feature Engineering)
        # On doit recréer EXACTEMENT les colonnes utilisées lors du training
        
        current_row = input_df.iloc[-1].copy() # On prédit pour la dernière ligne reçue
        features_dict = {}
        
        for feat in self.features:
            # Cas 1: C'est une feature FFD (ex: 'P10_ffd')
            if '_ffd' in feat:
                original_col = feat.replace('_ffd', '')
                val = self.frac_diff_ffd_realtime(current_row[original_col], original_col, d=0.1)
                
                # Si on n'a pas assez d'historique (début de la simulation), on remplace par 0
                if np.isnan(val): val = 0.0 
                features_dict[feat] = val
                
            # Cas 2: C'est une feature normale (ex: 'M1')
            else:
                features_dict[feat] = current_row.get(feat, 0.0)
        
        # 3. Création du DataFrame pour le modèle
        X_live = pd.DataFrame([features_dict])
        
        # S'assurer que l'ordre des colonnes est respecté
        X_live = X_live[self.features]
        
        # 4. Prédiction
        # XGBoost renvoie [0, 1] ou proba. Kaggle attend souvent un score continu.
        # On renvoie la probabilité d'être classe 1.
        prediction = self.model.predict_proba(X_live)[:, 1][0]
        
        return prediction