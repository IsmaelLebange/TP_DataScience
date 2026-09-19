# scripts/visualize.py

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")))

from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.algorithm import nuees_dynamiques
from src.nuees_dynamiques.visualization import (
    plot_nuees_2d, plot_convergence, plot_cluster_sizes
)
from src.nuees_dynamiques.distances import euclidean_distance_squared


FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)


NUEES_TYPES = ["point", "ensemble_points",
               "distribution", "axe_factoriel", "structure"]


def run_all(X, k=3, seed=42):
    for nt in NUEES_TYPES:
        print(f"\n=== nuée = {nt} ===")
        try:
            res = nuees_dynamiques(
                X, k=k, max_iter=100, tol=1e-6, seed=seed,
                init_method="kmpp",
                nuee_type=nt,
                prototype_type="centroide",
                distance_fn=euclidean_distance_squared,
                verbose=False,
                m=3,
            )
        except Exception as e:
            print(f"  ⚠️  Échec : {e}")
            continue

        out = os.path.join(FIG_DIR, f"nuee_{nt}.png")
        try:
            plot_nuees_2d(X, res["labels"], res["nuees"],
                          nuee_type=nt, output_path=out)
        except Exception as e:
            print(f"  ⚠️  Plot échoué : {e}")

        print(f"  n_iter = {res['n_iter']} | "
              f"stop = {res['stop_reason']} | "
              f"W = {res['final_inertia']:.4f}")

        # Convergence (commune à tous)
        plot_convergence(
            res["history"],
            title=f"Convergence — nuée {nt}",
            output_path=os.path.join(FIG_DIR, f"conv_{nt}.png")
        )


def main():
    X, _ = generate_synthetic_data(
        n_per_cluster=60, k=3, d=2, seed=42, spread=0.7
    )
    print(f"Données : {X.shape}")

    run_all(X, k=3)

    # Tailles pour le cas point (référence)
    res = nuees_dynamiques(
        X, k=3, max_iter=100, tol=1e-6, seed=42,
        nuee_type="point", prototype_type="centroide",
        distance_fn=euclidean_distance_squared
    )
    plot_cluster_sizes(
        res["labels"], k=3,
        title="Tailles des clusters — nuée point",
        output_path=os.path.join(FIG_DIR, "sizes_point.png")
    )

    print(f"\n🎉 Figures dans : {FIG_DIR}")


if __name__ == "__main__":
    main()