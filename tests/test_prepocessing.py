from src.nuees_dynamiques.preprocessing import (
    generate_synthetic_data, standardize, normalize_minmax, handle_missing_values
)
import numpy as np

# Génération
X, y = generate_synthetic_data(n_per_cluster=30, k=3, d=2)
print("Forme de X :", X.shape)
print("Labels uniques :", np.unique(y))

# Introduction volontaire de NaN
X_nan = X.copy()
X_nan[0, 0] = np.nan
X_nan[5, 1] = np.nan

# Traitement
X_filled = handle_missing_values(X_nan, strategy="mean")
print("NaN restants après remplissage :", np.isnan(X_filled).sum())

# Standardisation
X_std = standardize(X_filled)
print("Moyenne par colonne :", np.mean(X_std, axis=0))
print("Écart-type par colonne :", np.std(X_std, axis=0))

# Normalisation min-max
X_mm = normalize_minmax(X_filled)
print("Min par colonne :", np.min(X_mm, axis=0))
print("Max par colonne :", np.max(X_mm, axis=0))