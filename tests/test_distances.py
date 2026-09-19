import numpy as np
from src.nuees_dynamiques.distances import (
    euclidean_distance,
    euclidean_distance_squared,
    manhattan_distance,
    minkowski_distance,
    compute_distance_matrix,
    closest_center,
)

x = np.array([0.0, 0.0])
c1 = np.array([3.0, 4.0])
c2 = np.array([1.0, 1.0])

print("Euclidienne (x, c1) =", euclidean_distance(x, c1))          # 5.0
print("Euclidienne² (x, c1) =", euclidean_distance_squared(x, c1)) # 25.0
print("Manhattan (x, c1) =", manhattan_distance(x, c1))            # 7.0
print("Minkowski p=3 (x, c1) =", minkowski_distance(x, c1, p=3))   # ~4.497

X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 1.0]])
centers = np.array([[1.0, 1.0], [3.0, 4.0]])

D = compute_distance_matrix(X, centers)
print("Matrice de distances :\n", D)

idx = closest_center(x, centers)
print("Centre le plus proche de x :", idx)  # doit être 0 (centre [1,1])