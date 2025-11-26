# src/sampling.py
import pandas as pd
import numpy as np

def get_num_co_events(close_idx, t1, molecule):
    """
    SNIPPET 4.1: Calcule le nombre d'événements concurrents (chevauchement) pour chaque barre.
    """
    # 1. Trouver les événements qui concernent cette période
    t1 = t1.fillna(close_idx[-1]) 
    t1 = t1[t1 >= molecule[0]]
    t1 = t1.loc[:t1[molecule].max()]
    
    # 2. Compter les chevauchements
    iloc = close_idx.searchsorted(np.array([t1.index[0], t1.max()]))
    count = pd.Series(0, index=close_idx[iloc[0]:iloc[1]+1])
    
    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.
        
    return count.loc[molecule[0]:t1[molecule].max()]

def get_sample_tw(t1, num_co_events, molecule):
    """
    SNIPPET 4.2: Calcule l'unicité moyenne d'un événement sur sa durée de vie.
    """
    wght = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[wght.index].items():
        # Poids = 1 / Nombre de concurrents
        wght.loc[t_in] = (1. / num_co_events.loc[t_in:t_out]).mean()
    return wght

def get_sample_weights(t1, close, events, molecule):
    """
    COMBINAISON SNIPPET 4.2 + 4.10: 
    Calcule les poids finaux (Unicité * Retour Absolu).
    """
    # 1. Calculer la concurrence (Overlap)
    num_co_events = get_num_co_events(close.index, t1, molecule)
    num_co_events = num_co_events.loc[~num_co_events.index.duplicated(keep='last')]
    num_co_events = num_co_events.reindex(close.index).fillna(0)
    
    # 2. Calculer l'unicité moyenne (tW)
    # Note: Dans une version multiprocessing, on découperait 'molecule'. Ici on fait tout d'un coup.
    uniq_w = get_sample_tw(t1, num_co_events, molecule)
    
    # 3. Pondération par Rendement Absolu (Snippet 4.10)
    # On donne plus de poids aux gros mouvements.
    ret = np.log(close).diff() # Log returns
    
    final_w = pd.Series(index=molecule)
    for t_in, t_out in t1.loc[final_w.index].items():
        # Somme des rendements absolus divisée par la concurrence
        # Si un gros mouvement est partagé par 10 trades, chacun reçoit 1/10 du crédit.
        final_w.loc[t_in] = (ret.loc[t_in:t_out] / num_co_events.loc[t_in:t_out]).sum()
        
    return final_w.abs()