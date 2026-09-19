import numpy as np
from src.nuees_dynamiques.assignment import (
    assign_clusters, count_cluster_sizes, get_cluster_members
)

# Cas simple : 3 points, 2 centres
X = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [10.0, 10.0],
])
centers = np.array([
    [0.5, 0.5],
    [9.5, 9.5],
])

labels = assign_clusters(X, centers)
print("Labels :", labels)                    # [0, 0, 1]
print("Tailles :", count_cluster_sizes(labels, 2))  # [2, 1]

members_0 = get_cluster_members(X, labels, 0)
print("Membres cluster 0 :\n", members_0)

members_1 = get_cluster_members(X, labels, 1)
print("Membres cluster 1 :\n", members_1)

# Cas avec égalité
X2 = np.array([[1.0, 0.0]])
centers2 = np.array([
    [0.0, 0.0],
    [2.0, 0.0],
])
labels2 = assign_clusters(X2, centers2)
print("Égalité → label =", labels2[0])  # doit être 0 (plus petit indice)

# Test sur données synthétiques
from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.initialization import init_kmpp

X3, y3 = generate_synthetic_data(n_per_cluster=30, k=3, d=2, seed=42)
c3 = init_kmpp(X3, k=3, seed=42)
labels3 = assign_clusters(X3, c3)
print("Forme des labels :", labels3.shape)
print("Valeurs uniques :", np.unique(labels3))
print("Tailles des clusters :", count_cluster_sizes(labels3, 3))