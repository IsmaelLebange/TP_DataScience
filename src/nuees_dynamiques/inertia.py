# src/nuees_dynamiques/inertia.py

import numpy as np
from .distances import euclidean_distance_squared


def compute_inertia(X, labels, centers, distance_fn=euclidean_distance_squared):
    """
    Calcule l'inertie intra-classe W.

    Paramètres :
        X : np.ndarray (n, d)
        labels : np.ndarray (n,)
        centers : np.ndarray (k, d)
        distance_fn : fonction de distance (x, c) -> float

    Retour :
        W : float
    """
    n = X.shape[0]
    W = 0.0

    for i in range(n):
        c = centers[labels[i]]
        W += distance_fn(X[i], c)

    return W


def inertia_converged(old_inertia, new_inertia, tol=1e-4):
    """
    Vérifie si l'inertie a suffisamment peu varié pour considérer
    que l'algorithme a convergé.

    Retour :
        converged : bool
        variation : float
    """
    variation = abs(old_inertia - new_inertia)
    converged = variation < tol
    return converged, variation