# src/nuees_dynamiques/initialization.py

import numpy as np
from .distances import euclidean_distance_squared


def init_random(X, k, seed=None):
    """
    Initialisation aléatoire : choisit k points distincts parmi X.

    Paramètres :
        X : np.ndarray (n, d)
        k : int, nombre de centres
        seed : int ou None

    Retour :
        centers : np.ndarray (k, d)
    """
    n = X.shape[0]
    if k <= 0:
        raise ValueError("k doit être strictement positif.")
    if k > n:
        raise ValueError(f"k ({k}) ne peut pas dépasser le nombre de points ({n}).")

    rng = np.random.RandomState(seed)
    indices = rng.choice(n, size=k, replace=False)
    centers = X[indices].copy()
    return centers


def init_kmpp(X, k, seed=None, distance_fn=euclidean_distance_squared):
    """
    Initialisation k-means++ :
    - le premier centre est choisi uniformément au hasard parmi X ;
    - chaque centre suivant est choisi avec une probabilité proportionnelle
      à la distance² au centre le plus proche déjà sélectionné.

    Paramètres :
        X : np.ndarray (n, d)
        k : int
        seed : int ou None
        distance_fn : fonction de distance (x, c) -> float

    Retour :
        centers : np.ndarray (k, d)
    """
    n = X.shape[0]
    if k <= 0:
        raise ValueError("k doit être strictement positif.")
    if k > n:
        raise ValueError(f"k ({k}) ne peut pas dépasser le nombre de points ({n}).")

    rng = np.random.RandomState(seed)

    # 1) Premier centre : tirage uniforme
    first_idx = rng.randint(0, n)
    centers = [X[first_idx].copy()]

    # 2) Centres suivants
    for _ in range(1, k):
        # Calcul de la distance² de chaque point au centre le plus proche déjà choisi
        d2 = np.zeros(n, dtype=float)
        for i in range(n):
            best = distance_fn(X[i], centers[0])
            for c in centers[1:]:
                dist = distance_fn(X[i], c)
                if dist < best:
                    best = dist
            d2[i] = best

        # Probabilités proportionnelles à d2
        total = np.sum(d2)
        if total == 0.0:
            # Tous les points sont confondus avec les centres : on prend un point au hasard
            idx = rng.randint(0, n)
        else:
            probs = d2 / total
            idx = rng.choice(n, p=probs)

        centers.append(X[idx].copy())

    return np.array(centers)


def init_uniform(X, k, seed=None):
    """
    Initialisation uniforme : k centres répartis aléatoirement
    dans la boîte englobante des données.

    Paramètres :
        X : np.ndarray (n, d)
        k : int
        seed : int ou None

    Retour :
        centers : np.ndarray (k, d)
    """
    if k <= 0:
        raise ValueError("k doit être strictement positif.")

    rng = np.random.RandomState(seed)
    d = X.shape[1]
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    centers = np.zeros((k, d), dtype=float)
    for j in range(k):
        for l in range(d):
            centers[j, l] = rng.uniform(mins[l], maxs[l])

    return centers


def get_initialization_function(name):
    """
    Retourne la fonction d'initialisation correspondant au nom donné.
    """
    name = name.lower()
    if name in ("random", "aleatoire", "aléatoire"):
        return init_random
    elif name in ("kmpp", "kmeans++", "k-means++"):
        return init_kmpp
    elif name in ("uniform", "uniforme"):
        return init_uniform
    else:
        raise ValueError(f"Méthode d'initialisation inconnue : {name}")