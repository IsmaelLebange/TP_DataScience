# src/nuees_dynamiques/update.py

import numpy as np
from .distances import euclidean_distance_squared


def update_prototypes_centroide(X, labels, k, old_prototypes,
                                distance_fn=euclidean_distance_squared):
    """
    Mise à jour des prototypes par la méthode du centroïde.
    Chaque prototype = moyenne des points de son cluster.
    """
    n, d = X.shape
    new_prototypes = np.zeros((k, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) == 0:
            empty_clusters.append(j)
            if old_prototypes is not None:
                worst_idx = 0
                worst_dist = -1.0
                for i in range(n):
                    dist = distance_fn(X[i], old_prototypes[j])
                    if dist > worst_dist:
                        worst_dist = dist
                        worst_idx = i
                new_prototypes[j] = X[worst_idx].copy()
            else:
                new_prototypes[j] = np.zeros(d, dtype=float)
        else:
            mean = np.zeros(d, dtype=float)
            for i in indices:
                for l in range(d):
                    mean[l] += X[i, l]
            mean /= len(indices)
            new_prototypes[j] = mean

    return new_prototypes, empty_clusters


def update_prototypes_mediane(X, labels, k, old_prototypes,
                              distance_fn=euclidean_distance_squared):
    """
    Mise à jour des prototypes par la méthode de la médiane.
    Chaque prototype = médiane coordonnée par coordonnée des points du cluster.
    Cohérent avec la distance de Manhattan (L1).
    """
    n, d = X.shape
    new_prototypes = np.zeros((k, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) == 0:
            empty_clusters.append(j)
            if old_prototypes is not None:
                worst_idx = 0
                worst_dist = -1.0
                for i in range(n):
                    dist = distance_fn(X[i], old_prototypes[j])
                    if dist > worst_dist:
                        worst_dist = dist
                        worst_idx = i
                new_prototypes[j] = X[worst_idx].copy()
            else:
                new_prototypes[j] = np.zeros(d, dtype=float)
        else:
            median = np.zeros(d, dtype=float)
            for l in range(d):
                column = [X[i, l] for i in indices]
                column_sorted = sorted(column)
                m = len(column_sorted)
                if m % 2 == 0:
                    median[l] = 0.5 * (column_sorted[m // 2 - 1] +
                                       column_sorted[m // 2])
                else:
                    median[l] = column_sorted[m // 2]
            new_prototypes[j] = median

    return new_prototypes, empty_clusters


def update_prototypes_medoide(X, labels, k, old_prototypes,
                              distance_fn=euclidean_distance_squared):
    """
    Mise à jour des prototypes par la méthode du médoïde.
    Chaque prototype = le point du cluster qui minimise la somme
    des distances aux autres points du même cluster.
    """
    n, d = X.shape
    new_prototypes = np.zeros((k, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) == 0:
            empty_clusters.append(j)
            if old_prototypes is not None:
                worst_idx = 0
                worst_dist = -1.0
                for i in range(n):
                    dist = distance_fn(X[i], old_prototypes[j])
                    if dist > worst_dist:
                        worst_dist = dist
                        worst_idx = i
                new_prototypes[j] = X[worst_idx].copy()
            else:
                new_prototypes[j] = np.zeros(d, dtype=float)
        else:
            best_idx = indices[0]
            best_cost = float("inf")
            for i in indices:
                cost = 0.0
                for m in indices:
                    cost += distance_fn(X[i], X[m])
                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
            new_prototypes[j] = X[best_idx].copy()

    return new_prototypes, empty_clusters


def update_prototypes(X, labels, k, old_prototypes,
                      prototype_type="centroide",
                      distance_fn=euclidean_distance_squared):
    """
    Dispatcher : choisit la méthode de mise à jour selon prototype_type.
    """
    pt = prototype_type.lower()
    if pt in ("centroide", "centroïde", "mean", "moyenne"):
        return update_prototypes_centroide(
            X, labels, k, old_prototypes, distance_fn
        )
    elif pt in ("mediane", "médiane", "median"):
        return update_prototypes_mediane(
            X, labels, k, old_prototypes, distance_fn
        )
    elif pt in ("medoide", "médoïde", "medoid"):
        return update_prototypes_medoide(
            X, labels, k, old_prototypes, distance_fn
        )
    else:
        raise ValueError(f"prototype_type inconnu : {prototype_type}")


def prototypes_have_moved(old_prototypes, new_prototypes, tol=1e-4):
    """
    Vérifie si les prototypes ont bougé au-delà de la tolérance.
    """
    k, d = old_prototypes.shape
    max_shift = 0.0
    for j in range(k):
        s = 0.0
        for l in range(d):
            diff = old_prototypes[j, l] - new_prototypes[j, l]
            s += diff * diff
        s = np.sqrt(s)
        if s > max_shift:
            max_shift = s

    moved = max_shift >= tol
    return moved, max_shift