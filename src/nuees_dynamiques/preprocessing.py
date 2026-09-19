# src/nuees_dynamiques/preprocessing.py

import numpy as np

def handle_missing_values(X, strategy="mean"):
    """
    Remplace les valeurs manquantes (NaN) selon la stratégie choisie.
    Stratégies : 'mean', 'median', 'zero', 'drop'.
    """
    if strategy == "drop":
        mask = ~np.isnan(X).any(axis=1)
        X_clean = X[mask]
        if X_clean.shape[0] == 0:
            raise ValueError("Toutes les lignes contiennent des NaN.")
        return X_clean

    X_clean = X.copy()

    if strategy == "mean":
        for j in range(X_clean.shape[1]):
            col = X_clean[:, j]
            mask = np.isnan(col)
            if np.any(mask):
                mean_val = np.nanmean(col)
                col[mask] = mean_val

    elif strategy == "median":
        for j in range(X_clean.shape[1]):
            col = X_clean[:, j]
            mask = np.isnan(col)
            if np.any(mask):
                median_val = np.nanmedian(col)
                col[mask] = median_val

    elif strategy == "zero":
        X_clean = np.nan_to_num(X_clean, nan=0.0)

    else:
        raise ValueError(f"Stratégie inconnue : {strategy}")

    return X_clean


def standardize(X):
    """
    Standardisation (z-score) : chaque colonne a moyenne 0 et écart-type 1.
    """
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    # Éviter la division par zéro
    stds[stds == 0] = 1.0

    return (X - means) / stds


def normalize_minmax(X):
    """
    Normalisation Min-Max : chaque colonne est ramenée dans [0, 1].
    """
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    return (X - mins) / ranges


def generate_synthetic_data(n_per_cluster=50, k=3, d=2, seed=42, spread=0.5):
    """
    Génère un jeu de données synthétique avec k clusters bien séparés.
    Retourne X (n, d) et les vrais labels (pour vérification uniquement).
    """
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for j in range(k):
        center = rng.uniform(-5, 5, size=d)
        points = rng.normal(loc=center, scale=spread, size=(n_per_cluster, d))
        X.append(points)
        y.extend([j] * n_per_cluster)
    X = np.vstack(X)
    y = np.array(y, dtype=int)

    # Mélanger les points
    idx = rng.permutation(len(X))
    return X[idx], y[idx]