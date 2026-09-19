# src/nuees_dynamiques/evaluation.py

import numpy as np
from .distances import euclidean_distance, euclidean_distance_squared


def silhouette_score(X, labels, distance_fn=euclidean_distance):
    """
    Calcule le score de silhouette moyen (naïf).

    Pour chaque point i :
        a(i) = distance moyenne aux autres points de son cluster
        b(i) = min sur les autres clusters de la distance moyenne
               aux points de ce cluster
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
    Le score global est la moyenne des s(i).

    Interprétation :
        proche de  1 → clusters bien séparés
        proche de  0 → clusters qui se chevauchent
        proche de -1 → points probablement mal classés

    Paramètres :
        X : np.ndarray (n, d)
        labels : np.ndarray (n,)
        distance_fn : fonction (x, y) -> float

    Retour :
        float
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    k = len(unique_labels)

    if k < 2:
        raise ValueError("Silhouette nécessite au moins 2 clusters.")
    if k == n:
        raise ValueError("Silhouette indéfinie si chaque point est son propre cluster.")

    # Regrouper les indices par cluster
    clusters = {lab: [] for lab in unique_labels}
    for i in range(n):
        clusters[labels[i]].append(i)

    s_values = np.zeros(n, dtype=float)

    for i in range(n):
        lab_i = labels[i]
        same = clusters[lab_i]

        # a(i) : distance moyenne aux autres points du même cluster
        if len(same) <= 1:
            # cluster singleton : on définit a(i) = 0
            a_i = 0.0
        else:
            total = 0.0
            for j in same:
                if j != i:
                    total += distance_fn(X[i], X[j])
            a_i = total / (len(same) - 1)

        # b(i) : min sur les autres clusters de la distance moyenne
        b_i = float("inf")
        for lab_other, members in clusters.items():
            if lab_other == lab_i:
                continue
            if len(members) == 0:
                continue
            total = 0.0
            for j in members:
                total += distance_fn(X[i], X[j])
            avg = total / len(members)
            if avg < b_i:
                b_i = avg

        # s(i)
        denom = max(a_i, b_i)
        if denom == 0.0:
            s_values[i] = 0.0
        else:
            s_values[i] = (b_i - a_i) / denom

    return float(np.mean(s_values))


def davies_bouldin_score(X, labels, prototypes,
                         distance_fn=euclidean_distance):
    """
    Calcule l'indice de Davies-Bouldin (naïf).

    Pour chaque cluster i :
        S_i = dispersion intra-cluster =
              distance moyenne des points au prototype
        Pour chaque autre cluster j :
            M_ij = distance entre prototypes i et j
            R_ij = (S_i + S_j) / M_ij
        DB_i = max_j R_ij
    DB = moyenne des DB_i.

    Interprétation :
        plus petit = mieux (0 = parfait)

    Paramètres :
        X : np.ndarray (n, d)
        labels : np.ndarray (n,)
        prototypes : np.ndarray (k, d)
        distance_fn : fonction (x, y) -> float

    Retour :
        float
    """
    n = X.shape[0]
    k = prototypes.shape[0]

    # Regrouper les indices par cluster
    clusters = {j: [] for j in range(k)}
    for i in range(n):
        clusters[labels[i]].append(i)

    # Dispersion intra-cluster
    S = np.zeros(k, dtype=float)
    for j in range(k):
        members = clusters[j]
        if len(members) == 0:
            S[j] = 0.0
            continue
        total = 0.0
        for i in members:
            total += distance_fn(X[i], prototypes[j])
        S[j] = total / len(members)

    # Calcul de DB
    db_values = np.zeros(k, dtype=float)
    for i in range(k):
        max_r = 0.0
        for j in range(k):
            if i == j:
                continue
            m_ij = distance_fn(prototypes[i], prototypes[j])
            if m_ij == 0.0:
                r_ij = 0.0
            else:
                r_ij = (S[i] + S[j]) / m_ij
            if r_ij > max_r:
                max_r = r_ij
        db_values[i] = max_r

    return float(np.mean(db_values))


def elbow_curve(X, k_values, max_iter=100, tol=1e-4, seed=42,
                init_method="kmpp", prototype_type="centroide",
                distance_fn=euclidean_distance_squared,
                verbose=False):
    """
    Calcule l'inertie finale pour plusieurs valeurs de k (courbe du coude).

    Paramètres :
        X : np.ndarray (n, d)
        k_values : list[int]
        (autres) : passés à nuees_dynamiques

    Retour :
        dict { k : inertia_finale }
    """
    from .algorithm import nuees_dynamiques

    results = {}
    for k in k_values:
        if verbose:
            print(f"→ k = {k}")
        res = nuees_dynamiques(
            X, k=k, max_iter=max_iter, tol=tol, seed=seed,
            init_method=init_method, prototype_type=prototype_type,
            distance_fn=distance_fn, verbose=False
        )
        results[k] = res["final_inertia"]
    return results