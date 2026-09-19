# src/nuees_dynamiques/assignment.py

import numpy as np
from .distances import euclidean_distance_squared


def assign_clusters(X, centers, distance_fn=euclidean_distance_squared):
    """
    Affecte chaque point de X au centre le plus proche.

    Paramètres :
        X : np.ndarray (n, d)
        centers : np.ndarray (k, d)
        distance_fn : fonction de distance (x, c) -> float

    Retour :
        labels : np.ndarray (n,) d'entiers dans [0, k-1]
    """
    n = X.shape[0]
    k = centers.shape[0]

    if n == 0:
        raise ValueError("X est vide.")
    if k == 0:
        raise ValueError("centers est vide.")

    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_idx = 0
        best_dist = distance_fn(X[i], centers[0])

        for j in range(1, k):
            d = distance_fn(X[i], centers[j])
            if d < best_dist:
                best_dist = d
                best_idx = j

        labels[i] = best_idx

    return labels


def count_cluster_sizes(labels, k):
    """
    Compte le nombre de points dans chaque cluster.

    Paramètres :
        labels : np.ndarray (n,)
        k : int

    Retour :
        sizes : np.ndarray (k,) d'entiers
    """
    sizes = np.zeros(k, dtype=int)
    for lab in labels:
        sizes[lab] += 1
    return sizes


def get_cluster_members(X, labels, cluster_id):
    """
    Retourne les points appartenant au cluster cluster_id.

    Paramètres :
        X : np.ndarray (n, d)
        labels : np.ndarray (n,)
        cluster_id : int

    Retour :
        members : np.ndarray (m, d)
    """
    indices = np.where(labels == cluster_id)[0]
    return X[indices]