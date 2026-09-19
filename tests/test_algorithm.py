# tests/test_algorithm.py

import numpy as np
from itertools import permutations
from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.algorithm import nuees_dynamiques


def clustering_accuracy(y_true, y_pred, k):
    """Précision à permutation près (vérification uniquement)."""
    best = 0.0
    for perm in permutations(range(k)):
        acc = np.mean([perm[y_pred[i]] == y_true[i]
                       for i in range(len(y_true))])
        best = max(best, acc)
    return best


def test_centroide():
    X, y_true = generate_synthetic_data(n_per_cluster=50, k=3, d=2, seed=42)
    result = nuees_dynamiques(
        X, k=3, max_iter=100, tol=1e-6, seed=42,
        prototype_type="centroide", verbose=True
    )
    print("n_iter =", result["n_iter"],
          "| stop =", result["stop_reason"],
          "| W =", result["final_inertia"])
    acc = clustering_accuracy(y_true, result["labels"], 3)
    print("Précision =", acc)
    assert acc > 0.95
    print("✅ centroide OK\n")


def test_medoide():
    X, y_true = generate_synthetic_data(n_per_cluster=50, k=3, d=2, seed=42)
    result = nuees_dynamiques(
        X, k=3, max_iter=100, tol=1e-6, seed=42,
        prototype_type="medoide", verbose=True
    )
    print("n_iter =", result["n_iter"],
          "| stop =", result["stop_reason"],
          "| W =", result["final_inertia"])
    acc = clustering_accuracy(y_true, result["labels"], 3)
    print("Précision =", acc)
    assert acc > 0.85
    print("✅ medoide OK\n")


def test_mediane():
    X, y_true = generate_synthetic_data(n_per_cluster=50, k=3, d=2, seed=42)
    result = nuees_dynamiques(
        X, k=3, max_iter=100, tol=1e-6, seed=42,
        prototype_type="mediane", verbose=True
    )
    print("n_iter =", result["n_iter"],
          "| stop =", result["stop_reason"],
          "| W =", result["final_inertia"])
    acc = clustering_accuracy(y_true, result["labels"], 3)
    print("Précision =", acc)
    assert acc > 0.85
    print("✅ mediane OK\n")


def test_k_equals_1():
    X, _ = generate_synthetic_data(n_per_cluster=30, k=3, d=2, seed=1)
    result = nuees_dynamiques(X, k=1, max_iter=50, seed=1,
                              init_method="random")
    assert result["prototypes"].shape == (1, 2)
    assert np.all(result["labels"] == 0)
    print("✅ k=1 OK\n")


def test_max_iter_reached():
    X, _ = generate_synthetic_data(n_per_cluster=50, k=5, d=2, seed=7)
    result = nuees_dynamiques(X, k=5, max_iter=2, tol=1e-12, seed=7)
    assert result["n_iter"] <= 2
    print("✅ max_iter OK\n")


def test_init_methods():
    X, _ = generate_synthetic_data(n_per_cluster=30, k=3, d=2, seed=3)
    for method in ("random", "kmpp", "uniform"):
        result = nuees_dynamiques(X, k=3, max_iter=50, seed=3,
                                  init_method=method)
        assert result["prototypes"].shape == (3, 2)
        assert result["n_iter"] >= 1
        print(f"✅ init_method = {method} OK")
    print()


if __name__ == "__main__":
    test_centroide()
    test_medoide()
    test_mediane()
    test_k_equals_1()
    test_max_iter_reached()
    test_init_methods()
    print("🎉 Tous les tests de algorithm.py sont OK")