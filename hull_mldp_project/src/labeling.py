# src/labeling.py
import pandas as pd
import numpy as np

def get_daily_vol(close, span0=100):
    """
    SNIPPET 3.1: Calcule la volatilité quotidienne dynamique (EWM).
    close: Série de prix (ou cumulative returns)
    span0: Fenêtre de la moyenne mobile exponentielle
    """
    # Calcul des rendements journaliers : P(t) / P(t-1) - 1
    df0 = close.pct_change() 
    # Ecart-type mobile exponentiel
    df0 = df0.ewm(span=span0).std()
    return df0

def apply_pt_sl_on_t1(close, events, pt_sl, molecule):
    """
    SNIPPET 3.2 (Interne): Applique Stop Loss / Profit Taking
    """
    events_ = events.loc[molecule]
    out = events_[['t1']].copy(deep=True)
    
    if pt_sl[0] > 0: pt = pt_sl[0] * events_['trgt']
    else: pt = pd.Series(index=events.index) # NaNs

    if pt_sl[1] > 0: sl = -pt_sl[1] * events_['trgt']
    else: sl = pd.Series(index=events.index) # NaNs

    # --- CORRECTION ROBUSTE ---
    # On remplace les NaNs par le dernier index DISPONIBLE
    max_idx = close.index[-1]
    
    # On itère. Pandas va peut-être nous donner des floats (à cause des NaNs précédents),
    # mais on s'en fiche car on va caster brutalement dans la boucle.
    loop_series = events_['t1'].fillna(max_idx)

    for loc, t1 in loop_series.items():
        # ICI: On force la conversion en entier pur Python
        # C'est ça qui corrige le TypeError "slice indexing"
        loc_safe = int(loc)
        t1_safe = int(t1)

        # On utilise les versions "safe" pour le slicing
        df0 = close[loc_safe:t1_safe] # Chemin des prix
        
        # Calcul des rendements
        df0 = (df0 / close[loc_safe] - 1) * events_.at[loc, 'side'] 
        
        # Date du premier touché
        out.loc[loc, 'sl'] = df0[df0 < sl[loc]].index.min()
        out.loc[loc, 'pt'] = df0[df0 > pt[loc]].index.min()
        
    return out
        
    return out

def get_events(close, t_events, pt_sl, trgt, min_ret, num_threads=1, t1=False, side=None):
    """
    SNIPPET 3.3 / 3.6: Trouve l'heure de la première barrière touchée.
    Cette fonction est simplifiée pour le single-thread ici (plus simple pour commencer).
    
    close: Série de prix
    t_events: Index des événements (dates où on veut labelliser)
    pt_sl: Liste [Multiplicateur Haut, Multiplicateur Bas]
    trgt: Série de volatilité (cible dynamique)
    min_ret: Rendement minimum pour considérer un trade
    t1: Barrière verticale (temps max)
    side: (Optionnel) Sens du trade pour Meta-Labeling
    """
    
    # 1. Filtrer les cibles trop faibles
    trgt = trgt.loc[t_events]
    trgt = trgt[trgt > min_ret]
    
    # 2. Gérer la barrière verticale (t1)
    if t1 is False: 
        t1 = pd.Series(pd.NaT, index=t_events)
    
    # 3. Préparer l'objet events
    if side is None: 
        side_ = pd.Series(1., index=trgt.index)
        pt_sl_ = [pt_sl[0], pt_sl[0]]
    else: 
        side_ = side.loc[trgt.index]
        pt_sl_ = pt_sl[:2]
        
    events = pd.concat({'t1': t1, 'trgt': trgt, 'side': side_}, axis=1).dropna(subset=['trgt'])
    
    # --- CORRECTION ICI ---
    # On s'assure que l'index de events est bien du même type que l'index de close
    # Si close.index est Int64, events.index doit l'être aussi.
    events.index = events.index.astype(close.index.dtype)
    
    # On s'assure que la colonne t1 contient des valeurs du même type que l'index
    # Sauf les NaNs bien sûr.
    # Si t1 contient des floats (ex: 6978.0), on les convertit en entiers pour matcher l'index
    if pd.api.types.is_float_dtype(events['t1']):
         # On remplit temporairement les NaN pour pouvoir convertir en int, puis on remet NaN
         events['t1'] = events['t1'].fillna(-1).astype(close.index.dtype)
         events.loc[events['t1'] == -1, 't1'] = np.nan

    # 4. Appliquer Triple Barrière
    res = apply_pt_sl_on_t1(close, events, pt_sl_, events.index)
    
    events['t1'] = res.dropna(how='all').min(axis=1)
    if side is None: events = events.drop('side', axis=1)
        
    return events

def get_bins(events, close):
    """
    SNIPPET 3.5 / 3.7: Génère les labels finaux (0 ou 1)
    """
    # 1. Aligner les prix
    events_ = events.dropna(subset=['t1'])
    px = events_.index.union(events_['t1'].values).drop_duplicates()
    px = close.reindex(px, method='bfill')
    
    # 2. Créer l'objet de sortie
    out = pd.DataFrame(index=events_.index)
    
    # --- CORRECTION ICI : ON GARDE T1 et TRGT ---
    out['t1'] = events_['t1']     # CRUCIAL pour le Chapitre 4
    out['trgt'] = events_['trgt'] # Utile pour debug
    # ---------------------------------------------
    
    out['ret'] = px.loc[events_['t1'].values].values / px.loc[events_.index] - 1
    
    # Si on a une info de 'side' (Meta-Labeling)
    if 'side' in events_: out['ret'] *= events_['side']
        
    # Signe du résultat
    out['bin'] = np.sign(out['ret'])
    
    # Pour le Meta-Labeling: Si ret <= 0, alors label 0.
    if 'side' in events_: out.loc[out['ret'] <= 0, 'bin'] = 0
        
    return out