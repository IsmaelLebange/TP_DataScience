# src/nuees_dynamiques/algorithm.py

import numpy as np

from .distances import euclidean_distance_squared
from .initialization import init_random, init_kmpp, init_uniform
from .nuees import update_nuees, assign_nuees, NUEES_TYPES
from .inertia import compute_inertia
from .utils import validate_X, validate_labels


def _get_init_fn(name):
    name = name.lower()
    if name in ("random", "aleatoire", "aléatoire"):
        return init_random
    elif name in ("kmpp", "kmeans++", "k-means++"):
        return init_kmpp
    elif name in ("uniform", "uniforme"):
        return init_uniform
    raise ValueError(f"Méthode d'initialisation inconnue : {name}")


def nuees_dynamiques(X, k, max_iter=100, tol=1e-4, seed=None,
                     init_method="kmpp",
                     nuee_type="point",
                     prototype_type="centroide",
                     distance_fn=euclidean_distance_squared,
                     verbose=False,
                     **kwargs):
    """
    Algorithme des nuées dynamiques (générique).

    nuee_type :
        - "point"             → k-means / k-médoïdes / médiane
        - "ensemble_points"   → m points représentatifs par nuée
        - "distribution"      → gaussienne (moyenne + covariance)
        - "axe_factoriel"     → droite/axe principal
        - "structure"         → segment représentatif

    Le cas "point" ramène l'algorithme à k-means (centroïde)
    ou k-médoïdes (médoïde), selon prototype_type.
    """
    validate_X(X)
    n, d = X.shape
    if k <= 0:
        raise ValueError("k doit être strictement positif.")
    if k > n:
        raise ValueError(f"k ({k}) ne peut pas dépasser n ({n}).")
    if nuee_type not in NUEES_TYPES:
        raise ValueError(f"nuee_type doit être dans {NUEES_TYPES}")

    # 1) Initialisation (points uniquement)
    init_fn = _get_init_fn(init_method)
    points = init_fn(X, k, seed=seed)

    # Adapter les nuées au type choisi
    if nuee_type == "point":
        nuees = points
    elif nuee_type == "ensemble_points":
        m = kwargs.get("m", 3)
        nuees = np.zeros((k, m, d), dtype=float)
        for j in range(k):
            for r in range(m):
                nuees[j, r] = points[j]
    elif nuee_type == "distribution":
        nuees = {
            "means": points.copy(),
            "covs": np.array([np.eye(d) for _ in range(k)])
        }
    elif nuee_type == "axe_factoriel":
        nuees = {
            "points": points.copy(),
            "dirs": np.zeros((k, d), dtype=float)
        }
        # Direction initiale : vecteur aléatoire unitaire
        rng = np.random.RandomState(seed)
        for j in range(k):
            v = rng.normal(size=d)
            v /= np.linalg.norm(v) + 1e-12
            nuees["dirs"][j] = v
    elif nuee_type == "structure":
        nuees = np.zeros((k, 2, d), dtype=float)
        for j in range(k):
            nuees[j, 0] = points[j]
            nuees[j, 1] = points[j]

    history = []
    empty_clusters_history = []
    stop_reason = "none"
    labels = None

    # 2) Boucle
    for iteration in range(1, max_iter + 1):
        # a) Affectation
        labels = assign_nuees(X, nuees, nuee_type=nuee_type,
                              distance_fn=distance_fn)
        validate_labels(labels, n, k)

        # b) Mise à jour
        new_nuees, empty_clusters = update_nuees(
            X, labels, k, nuees,
            nuee_type=nuee_type,
            prototype_type=prototype_type,
            distance_fn=distance_fn,
            **kwargs
        )
        empty_clusters_history.append(empty_clusters)

        # c) Inertie (toujours calculée par rapport aux centroïdes
        #    pour rester comparable entre variantes)
        centroids = np.zeros((k, d), dtype=float)
        for j in range(k):
            idx = [i for i in range(n) if labels[i] == j]
            if len(idx) == 0:
                centroids[j] = 0.0
            else:
                for i in idx:
                    centroids[j] += X[i]
                centroids[j] /= len(idx)
        inertia = compute_inertia(X, labels, centroids, distance_fn=distance_fn)
        history.append(inertia)

        if verbose:
            print(f"[Itération {iteration:03d}] inertie = {inertia:.6f}")

        # d) Convergence : on s'arrête si inertie stable OU max_iter
        if iteration >= max_iter:
            stop_reason = "max_iter"
            break
        if len(history) >= 2:
            if abs(history[-1] - history[-2]) < tol:
                stop_reason = "inertia"
                break

        nuees = new_nuees

    return {
        "labels": labels,
        "nuees": nuees,
        "history": history,
        "n_iter": len(history),
        "stop_reason": stop_reason,
        "final_inertia": inertia,
        "empty_clusters_history": empty_clusters_history,
        "nuee_type": nuee_type,
        "prototype_type": prototype_type,
    }