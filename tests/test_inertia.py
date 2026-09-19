import numpy as np
from src.nuees_dynamiques.inertia import compute_inertia, inertia_converged
from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.initialization import init_kmpp
from src.nuees_dynamiques.assignment import assign_clusters

from src.nuees_dynamiques.update import compute_centers

# Cas simple : 3 points, 2 centres
X = np.array([
    [0.0, 0.0],
    [2.0, 2.0],
    [10.0, 10.0],
])
labels = np.array([0, 0, 1])
centers = np.array([[1.0, 1.0], [10.0, 10.0]])

W = compute_inertia(X, labels, centers)
print("Inertie W =", W)
# Attendu :
# point 0 → distance² à [1,1] = 1 + 1 = 2
# point 1 → distance² à [1,1] = 1 + 1 = 2
# point 2 → distance² à [10,10] = 0
# W = 2 + 2 + 0 = 4

# Test de convergence de l'inertie
old_W = 4.0
new_W = 3.999
conv, var = inertia_converged(old_W, new_W, tol=1e-4)
print("Convergé ?", conv, "| variation =", var)
# Attendu : variation = 0.001 → non convergé

old_W = 4.0
new_W = 3.99999
conv, var = inertia_converged(old_W, new_W, tol=1e-4)
print("Convergé ?", conv, "| variation =", var)
# Attendu : variation = 1e-5 → convergé

# Test sur données synthétiques
X2, y2 = generate_synthetic_data(n_per_cluster=30, k=3, d=2, seed=42)
c2 = init_kmpp(X2, k=3, seed=42)
labels2 = assign_clusters(X2, c2)
W2 = compute_inertia(X2, labels2, c2)
print("Inertie sur données synthétiques :", W2)
print("Forme des labels :", labels2.shape)
print("Forme des centres :", c2.shape)



# Itération 1
labels_1 = assign_clusters(X2, c2)
W_1 = compute_inertia(X2, labels_1, c2)

# Mise à jour des centres
c3, _ = compute_centers(X2, labels_1, k=3, old_centers=c2)

# Itération 2
labels_2 = assign_clusters(X2, c3)
W_2 = compute_inertia(X2, labels_2, c3)

print("W avant mise à jour :", W_1)
print("W après mise à jour :", W_2)
print("W a diminué ?", W_2 <= W_1)