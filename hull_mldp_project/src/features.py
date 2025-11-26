# src/features.py
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def get_weights_ffd(d, thres):
    """
    Calcule les poids w pour la différenciation fractionnaire (Fixed Window).
    d: Ordre de différenciation (ex: 0.4)
    thres: Seuil de coupure pour les poids négligeables (ex: 1e-5)
    """
    w, k = [1.], 1
    while True:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)

def frac_diff_ffd(series, d, thres=1e-5):
    """
    Applique la différenciation fractionnaire (FFD) sur une série pandas.
    Gère les valeurs manquantes via ffill avant calcul.
    """
    # 1. Gestion des NaNs (Critique en finance)
    series = series.ffill().dropna()
    
    # 2. Calcul des poids
    w = get_weights_ffd(d, thres)
    width = len(w) - 1
    
    # 3. Application des poids (Rolling dot product)
    # C'est ici que la magie opère : on applique la fenêtre glissante
    df = {}
    name = series.name
    series_val = series.values
    
    # On itère, mais optimisé (c'est lent en pure python, ici c'est acceptable)
    # Pour de la prod massive, on utiliserait numpy stride tricks
    output = []
    
    # On commence après la "fenêtre de mémoire" (width)
    if len(series) < width:
        return pd.Series(index=series.index, data=np.nan) # Pas assez de data

    for i in range(width, len(series)):
        # Le produit scalaire entre les poids et la fenêtre de prix passée
        val = np.dot(w.T, series_val[i-width:i+1])[0]
        output.append(val)
        
    # On remet l'index temporel correct (on a perdu les 'width' premières données)
    return pd.Series(data=output, index=series.index[width:], name=name)

def find_min_d(series, thres=1e-5, p_val_thres=0.05):
    """
    Trouve le d minimum pour rendre la série stationnaire.
    """
    # On teste d de 0.0 à 1.0 par pas de 0.1
    possible_ds = np.linspace(0, 1, 11)
    
    for d in possible_ds:
        if d == 0: 
            # Test sur brut
            diff_series = series.dropna()
        else:
            diff_series = frac_diff_ffd(series, d, thres).dropna()
        
        # Test ADF
        if len(diff_series) < 20: continue # Trop court
        p_val = adfuller(diff_series, maxlag=1, regression='c', autolag=None)[1]
        
        if p_val < p_val_thres:
            return d, p_val # Trouvé !
            
    return 1.0, 0.0 # Pire cas: on différencie totalement (rendement classique)