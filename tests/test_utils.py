import numpy as np
from src.nuees_dynamiques.utils import set_seed, check_array, print_progress, validate_labels, validate_centers, validate_X


X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
labels = np.array([0, 1, 0])
centers = np.array([[2.0, 3.0], [4.0, 5.0]])

print(validate_X(X))          # True
print(validate_labels(labels, 3, 2))  # True
print(validate_centers(centers, 2, 2)) # True