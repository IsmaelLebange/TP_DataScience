import numpy as np
from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.initialization import (
    init_random, init_kmpp, init_uniform
)

# Données synthétiques
X, y = generate_synthetic_data(n_per_cluster=30, k=3, d=2, seed=42)
print("Forme de X :", X.shape)

k = 3

# 1) Initialisation aléatoire
c_rand = init_random(X, k, seed=42)
print("init_random shape :", c_rand.shape)
print("Centres aléatoires :\n", c_rand)

# 2) Initialisation k-means++
c_kmpp = init_kmpp(X, k, seed=42)
print("init_kmpp shape :", c_kmpp.shape)
print("Centres k-means++ :\n", c_kmpp)

# 3) Initialisation uniforme
c_unif = init_uniform(X, k, seed=42)
print("init_uniform shape :", c_unif.shape)
print("Centres uniformes :\n", c_unif)

# Reproductibilité
c_rand2 = init_random(X, k, seed=42)
print("Reproductibilité init_random :", np.allclose(c_rand, c_rand2))

c_kmpp2 = init_kmpp(X, k, seed=42)
print("Reproductibilité init_kmpp :", np.allclose(c_kmpp, c_kmpp2))

# Cas limites
try:
    init_random(X, k=1000)
except ValueError as e:
    print("Erreur attendue (k > n) :", e)

try:
    init_kmpp(X, k=0)
except ValueError as e:
    print("Erreur attendue (k <= 0) :", e)