# scripts/export_datasets.py

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.nuees_dynamiques.preprocessing import generate_synthetic_data


def export_synthetic():
    os.makedirs("data/synthetic", exist_ok=True)

    configs = [
        ("test_3clusters_2d", dict(n_per_cluster=60, k=3, d=2,
                                    seed=42, spread=0.7)),
        ("test_5clusters_2d", dict(n_per_cluster=40, k=5, d=2,
                                    seed=7, spread=0.6)),
        ("test_3clusters_3d", dict(n_per_cluster=60, k=3, d=3,
                                    seed=13, spread=0.7)),
    ]

    for name, cfg in configs:
        X, y = generate_synthetic_data(**cfg)
        d = X.shape[1]
        header = ",".join([f"x{i+1}" for i in range(d)])
        path = f"data/synthetic/{name}.csv"
        np.savetxt(path, X, delimiter=",", header=header, comments="")
        print(f"✅ {path} → {X.shape}")


def export_real():
    os.makedirs("data/real", exist_ok=True)

    try:
        from sklearn.datasets import load_iris, load_wine
    except ImportError:
        print("⚠️  sklearn non installé. Installer avec : pip install scikit-learn")
        print("   (uniquement pour l'export des données, pas pour l'algo)")
        return

    # Iris
    iris = load_iris()
    X_iris = iris.data
    header = ",".join(iris.feature_names).replace(" (cm)", "")
    path = "data/real/iris.csv"
    np.savetxt(path, X_iris, delimiter=",", header=header, comments="")
    print(f"✅ {path} → {X_iris.shape}")

    # Wine (optionnel, pour montrer la généralité)
    wine = load_wine()
    X_wine = wine.data
    header = ",".join(wine.feature_names)
    path = "data/real/wine.csv"
    np.savetxt(path, X_wine, delimiter=",", header=header, comments="")
    print(f"✅ {path} → {X_wine.shape}")


if __name__ == "__main__":
    print("=== Export synthétique ===")
    export_synthetic()
    print("\n=== Export réel ===")
    export_real()
    print("\n🎉 Datasets prêts dans data/")