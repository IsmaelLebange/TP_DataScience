# src/nuees_dynamiques/distances.py

import numpy as np


def euclidean_distance(x, c):
    """
    Distance euclidienne entre deux vecteurs x et c.
    Implémentation naïve avec boucle.
    """
    if len(x) != len(c):
        raise ValueError("x et c doivent avoir la même dimension.")
    s = 0.0
    for l in range(len(x)):
        diff = x[l] - c[l]
        s += diff * diff
    return np.sqrt(s)


def euclidean_distance_squared(x, c):
    """
    Distance euclidienne au carré entre x et c.
    Plus rapide car on évite la racine carrée.
    """
    if len(x) != len(c):
        raise ValueError("x et c doivent avoir la même dimension.")
    s = 0.0
    for l in range(len(x)):
        diff = x[l] - c[l]
        s += diff * diff
    return s


def manhattan_distance(x, c):
    """
    Distance de Manhattan (L1) entre x et c.
    """
    if len(x) != len(c):
        raise ValueError("x et c doivent avoir la même dimension.")
    s = 0.0
    for l in range(len(x)):
        s += abs(x[l] - c[l])
    return s


def minkowski_distance(x, c, p=3):
    """
    Distance de Minkowski d'ordre p.
    p=1 → Manhattan, p=2 → Euclidienne.
    """
    if len(x) != len(c):
        raise ValueError("x et c doivent avoir la même dimension.")
    if p <= 0:
        raise ValueError("p doit être strictement positif.")
    s = 0.0
    for l in range(len(x)):
        s += abs(x[l] - c[l]) ** p
    return s ** (1.0 / p)


def get_distance_function(name):
    """
    Retourne la fonction de distance correspondant au nom donné.
    """
    name = name.lower()
    if name in ("euclidienne", "euclidean", "l2"):
        return euclidean_distance
    elif name in ("euclidienne_carree", "euclidean_squared", "l2_squared"):
        return euclidean_distance_squared
    elif name in ("manhattan", "l1", "cityblock"):
        return manhattan_distance
    elif name in ("minkowski",):
        return minkowski_distance
    else:
        raise ValueError(f"Distance inconnue : {name}")

def compute_distance_matrix(X, centers, distance_fn=euclidean_distance_squared):
    """
    Calcule la matrice des distances entre chaque point de X et chaque centre.
    
    Paramètres :
        X : np.ndarray (n, d)
        centers : np.ndarray (k, d)
        distance_fn : fonction prenant (x, c) et renvoyant un float
    
    Retour :
        D : np.ndarray (n, k) où D[i, j] = distance(X[i], centers[j])
    """
    n = X.shape[0]
    k = centers.shape[0]
    D = np.zeros((n, k), dtype=float)

    for i in range(n):
        for j in range(k):
            D[i, j] = distance_fn(X[i], centers[j])

    return D

def closest_center(x, centers, distance_fn=euclidean_distance_squared):
    """
    Retourne l'indice du centre le plus proche de x.
    En cas d'égalité, retourne le plus petit indice.
    """
    best_idx = 0
    best_dist = distance_fn(x, centers[0])
    for j in range(1, centers.shape[0]):
        d = distance_fn(x, centers[j])
        if d < best_dist:
            best_dist = d
            best_idx = j
    return best_idx