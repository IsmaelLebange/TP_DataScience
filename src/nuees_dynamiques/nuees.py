# src/nuees_dynamiques/nuees.py

import numpy as np
from .distances import (
    euclidean_distance_squared, euclidean_distance, manhattan_distance
)


# =============================================================
# TYPE "POINT"
# =============================================================

def update_nuee_point(X, labels, k, old_nuees,
                      prototype_type="centroide",
                      distance_fn=euclidean_distance_squared):
    """Nuée = un point (centroïde, médiane ou médoïde)."""
    from .update import update_prototypes
    return update_prototypes(
        X, labels, k, old_nuees,
        prototype_type=prototype_type,
        distance_fn=distance_fn
    )


def assign_nuee_point(X, nuees, distance_fn=euclidean_distance_squared):
    """Affectation : distance au point le plus proche."""
    from .assignment import assign_clusters
    return assign_clusters(X, nuees, distance_fn=distance_fn)


# =============================================================
# TYPE "ENSEMBLE DE POINTS REPRÉSENTATIFS"
# =============================================================

def update_nuee_ensemble_points(X, labels, k, old_nuees,
                                m=3,
                                distance_fn=euclidean_distance_squared):
    """
    Nuée = un ensemble de m points représentatifs par cluster.
    Naïvement : on prend les m points les plus proches du centroïde.
    """
    n, d = X.shape
    new_nuees = np.zeros((k, m, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) == 0:
            empty_clusters.append(j)
            for r in range(m):
                new_nuees[j, r] = old_nuees[j, r] if old_nuees is not None else np.zeros(d)
            continue

        # Centroïde du cluster
        centroid = np.zeros(d, dtype=float)
        for i in indices:
            for l in range(d):
                centroid[l] += X[i, l]
        centroid /= len(indices)

        # Trier les points par distance au centroïde
        dists = [(i, distance_fn(X[i], centroid)) for i in indices]
        dists.sort(key=lambda t: t[1])

        # Prendre les m plus proches (ou tous si moins de m)
        for r in range(m):
            if r < len(dists):
                new_nuees[j, r] = X[dists[r][0]].copy()
            else:
                new_nuees[j, r] = centroid.copy()

    return new_nuees, empty_clusters


def assign_nuee_ensemble_points(X, nuees, distance_fn=euclidean_distance_squared):
    """
    Affectation : pour chaque point, distance minimale à l'un des représentants.
    """
    n = X.shape[0]
    k, m, d = nuees.shape
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            for r in range(m):
                dist = distance_fn(X[i], nuees[j, r])
                if dist < best_d:
                    best_d = dist
                    best_j = j
        labels[i] = best_j

    return labels


# =============================================================
# TYPE "DISTRIBUTION" (gaussienne)
# =============================================================

def update_nuee_distribution(X, labels, k, old_nuees,
                             reg=1e-6):
    """
    Nuée = une gaussienne (moyenne + covariance).
    Naïf : calcul de la moyenne et de la covariance empirique.
    """
    n, d = X.shape
    new_means = np.zeros((k, d), dtype=float)
    new_covs = np.zeros((k, d, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) == 0:
            empty_clusters.append(j)
            if old_nuees is not None:
                new_means[j] = old_nuees["means"][j]
                new_covs[j] = old_nuees["covs"][j]
            else:
                new_means[j] = np.zeros(d)
                new_covs[j] = np.eye(d)
            continue

        # Moyenne
        mean = np.zeros(d, dtype=float)
        for i in indices:
            for l in range(d):
                mean[l] += X[i, l]
        mean /= len(indices)
        new_means[j] = mean

        # Covariance
        cov = np.zeros((d, d), dtype=float)
        for i in indices:
            diff = X[i] - mean
            for a in range(d):
                for b in range(d):
                    cov[a, b] += diff[a] * diff[b]
        if len(indices) > 1:
            cov /= (len(indices) - 1)
        else:
            cov = np.eye(d)
        # Régularisation pour éviter les matrices singulières
        cov += reg * np.eye(d)
        new_covs[j] = cov

    return {"means": new_means, "covs": new_covs}, empty_clusters


def _mahalanobis(x, mean, cov_inv):
    diff = x - mean
    return float(diff @ cov_inv @ diff)


def assign_nuee_distribution(X, nuees):
    """Affectation : distance de Mahalanobis minimale."""
    n = X.shape[0]
    k = nuees["means"].shape[0]
    labels = np.zeros(n, dtype=int)

    # Pré-calculer les inverses
    invs = []
    for j in range(k):
        try:
            inv = np.linalg.inv(nuees["covs"][j])
        except np.linalg.LinAlgError:
            inv = np.eye(nuees["means"].shape[1])
        invs.append(inv)

    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            d = _mahalanobis(X[i], nuees["means"][j], invs[j])
            if d < best_d:
                best_d = d
                best_j = j
        labels[i] = best_j

    return labels


# =============================================================
# TYPE "AXE FACTORIEL"
# =============================================================

def update_nuee_axe(X, labels, k, old_nuees):
    """
    Nuée = un axe (point + direction) = 1er axe principal du cluster.
    Naïf : ACP locale via la matrice de covariance.
    """
    n, d = X.shape
    new_points = np.zeros((k, d), dtype=float)
    new_dirs = np.zeros((k, d), dtype=float)
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]

        if len(indices) < 2:
            empty_clusters.append(j)
            new_points[j] = old_nuees["points"][j] if old_nuees else np.zeros(d)
            new_dirs[j] = old_nuees["dirs"][j] if old_nuees else np.zeros(d)
            continue

        # Moyenne
        mean = np.zeros(d, dtype=float)
        for i in indices:
            for l in range(d):
                mean[l] += X[i, l]
        mean /= len(indices)

        # Covariance
        cov = np.zeros((d, d), dtype=float)
        for i in indices:
            diff = X[i] - mean
            for a in range(d):
                for b in range(d):
                    cov[a, b] += diff[a] * diff[b]
        cov /= len(indices)

        # 1er vecteur propre
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]  # plus grande valeur propre

        new_points[j] = mean
        new_dirs[j] = direction

    return {"points": new_points, "dirs": new_dirs}, empty_clusters


def _dist_to_axis(x, point, direction):
    """Distance orthogonale d'un point à une droite."""
    diff = x - point
    proj = diff @ direction
    orth = diff - proj * direction
    return float(orth @ orth)


def assign_nuee_axe(X, nuees):
    n = X.shape[0]
    k = nuees["points"].shape[0]
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            d = _dist_to_axis(X[i], nuees["points"][j], nuees["dirs"][j])
            if d < best_d:
                best_d = d
                best_j = j
        labels[i] = best_j

    return labels


# =============================================================
# TYPE "STRUCTURE" (stub + naïve)
# =============================================================

def update_nuee_structure(X, labels, k, old_nuees):
    """
    Nuée = une structure représentative (graphe/arbre).
    Naïf : on prend un sous-ensemble de points formant un chemin
    reliant les points extrêmes du cluster (approximation).
    """
    n, d = X.shape
    new_nuees = np.zeros((k, 2, d), dtype=float)  # 2 extrémités
    empty_clusters = []

    for j in range(k):
        indices = [i for i in range(n) if labels[i] == j]
        if len(indices) == 0:
            empty_clusters.append(j)
            if old_nuees is not None:
                new_nuees[j] = old_nuees[j]
            else:
                new_nuees[j] = np.zeros((2, d))
            continue

        # Trouver les 2 points les plus éloignés (extrémités)
        best_pair = (indices[0], indices[0])
        best_d = -1.0
        for a in indices:
            for b in indices:
                if a < b:
                    diff = X[a] - X[b]
                    d2 = float(diff @ diff)
                    if d2 > best_d:
                        best_d = d2
                        best_pair = (a, b)
        new_nuees[j, 0] = X[best_pair[0]].copy()
        new_nuees[j, 1] = X[best_pair[1]].copy()

    return new_nuees, empty_clusters


def assign_nuee_structure(X, nuees):
    """
    Affectation : distance minimale au segment [p1, p2] de chaque structure.
    """
    n = X.shape[0]
    k = nuees.shape[0]
    labels = np.zeros(n, dtype=int)

    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            p1 = nuees[j, 0]
            p2 = nuees[j, 1]
            # distance point-segment
            seg = p2 - p1
            seg_norm = float(seg @ seg)
            if seg_norm == 0:
                d2 = float((X[i] - p1) @ (X[i] - p1))
            else:
                t = float((X[i] - p1) @ seg) / seg_norm
                t = max(0.0, min(1.0, t))
                proj = p1 + t * seg
                d2 = float((X[i] - proj) @ (X[i] - proj))
            if d2 < best_d:
                best_d = d2
                best_j = j
        labels[i] = best_j

    return labels


# =============================================================
# DISPATCHER GLOBAL
# =============================================================

NUEES_TYPES = [
    "point",
    "ensemble_points",
    "distribution",
    "axe_factoriel",
    "structure",
]


def update_nuees(X, labels, k, old_nuees, nuee_type="point",
                 prototype_type="centroide",
                 distance_fn=euclidean_distance_squared,
                 **kwargs):
    nt = nuee_type.lower()
    if nt == "point":
        return update_nuee_point(X, labels, k, old_nuees,
                                 prototype_type=prototype_type,
                                 distance_fn=distance_fn)
    elif nt == "ensemble_points":
        return update_nuee_ensemble_points(X, labels, k, old_nuees,
                                           m=kwargs.get("m", 3),
                                           distance_fn=distance_fn)
    elif nt == "distribution":
        return update_nuee_distribution(X, labels, k, old_nuees)
    elif nt == "axe_factoriel":
        return update_nuee_axe(X, labels, k, old_nuees)
    elif nt == "structure":
        return update_nuee_structure(X, labels, k, old_nuees)
    else:
        raise ValueError(f"nuee_type inconnu : {nuee_type}")


def assign_nuees(X, nuees, nuee_type="point",
                 distance_fn=euclidean_distance_squared):
    nt = nuee_type.lower()
    if nt == "point":
        return assign_nuee_point(X, nuees, distance_fn=distance_fn)
    elif nt == "ensemble_points":
        return assign_nuee_ensemble_points(X, nuees, distance_fn=distance_fn)
    elif nt == "distribution":
        return assign_nuee_distribution(X, nuees)
    elif nt == "axe_factoriel":
        return assign_nuee_axe(X, nuees)
    elif nt == "structure":
        return assign_nuee_structure(X, nuees)
    else:
        raise ValueError(f"nuee_type inconnu : {nuee_type}")