# scripts/evaluate.py

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nuees_dynamiques.preprocessing import generate_synthetic_data
from src.nuees_dynamiques.algorithm import nuees_dynamiques
from src.nuees_dynamiques.evaluation import (
    silhouette_score, davies_bouldin_score, elbow_curve
)
from src.nuees_dynamiques.distances import (
    euclidean_distance, euclidean_distance_squared, manhattan_distance
)


RESULTS_DIR = "results/evaluation"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_variants(X, k, seed=42):
    """
    Lance les nuées dynamiques avec différentes variantes
    et retourne un tableau comparatif.
    """
    variants = [
        ("centroide + euclidienne²", "centroide",
         euclidean_distance_squared, euclidean_distance),
        ("mediane + manhattan", "mediane",
         manhattan_distance, manhattan_distance),
        ("medoide + euclidienne²", "medoide",
         euclidean_distance_squared, euclidean_distance),
        ("medoide + manhattan", "medoide",
         manhattan_distance, manhattan_distance),
    ]

    rows = []
    for name, proto, dist_fit, dist_eval in variants:
        print(f"\n=== {name} ===")
        res = nuees_dynamiques(
            X, k=k, max_iter=100, tol=1e-6, seed=seed,
            init_method="kmpp", prototype_type=proto,
            distance_fn=dist_fit, verbose=False
        )
        sil = silhouette_score(X, res["labels"], distance_fn=dist_eval)
        db = davies_bouldin_score(X, res["labels"], res["prototypes"],
                                  distance_fn=dist_eval)
        rows.append({
            "variante": name,
            "inertie": res["final_inertia"],
            "silhouette": sil,
            "davies_bouldin": db,
            "n_iter": res["n_iter"],
            "stop": res["stop_reason"],
        })
        print(f"  inertie        = {res['final_inertia']:.4f}")
        print(f"  silhouette     = {sil:.4f}")
        print(f"  davies_bouldin = {db:.4f}")
        print(f"  n_iter         = {res['n_iter']} ({res['stop_reason']})")

    return rows


def print_table(rows):
    print("\n" + "=" * 90)
    print("COMPARAISON DES VARIANTES")
    print("=" * 90)
    header = f"{'Variante':<30} {'Inertie':>10} {'Silhouette':>12} {'Davies-Bouldin':>15}"
    print(header)
    print("-" * 90)
    for r in rows:
        print(f"{r['variante']:<30} {r['inertie']:>10.4f} "
              f"{r['silhouette']:>12.4f} {r['davies_bouldin']:>15.4f}")
    print("=" * 90)


def plot_elbow(X, k_values, output_path):
    print("\nCalcul de la courbe du coude...")
    results = elbow_curve(X, k_values, seed=42, verbose=True)

    ks = sorted(results.keys())
    inertias = [results[k] for k in ks]

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o", linewidth=2, color="steelblue")
    plt.title("Courbe du coude — inertie vs nombre de classes")
    plt.xlabel("Nombre de classes k")
    plt.ylabel("Inertie finale W")
    plt.grid(True, alpha=0.3)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def plot_silhouette_comparison(rows, output_path):
    names = [r["variante"] for r in rows]
    sils = [r["silhouette"] for r in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(names, sils, color="mediumseagreen")
    plt.title("Silhouette par variante (plus grand = mieux)")
    plt.xlabel("Silhouette")
    plt.axvline(0, color="black", linewidth=0.8)
    for bar, v in zip(bars, sils):
        plt.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def plot_db_comparison(rows, output_path):
    names = [r["variante"] for r in rows]
    dbs = [r["davies_bouldin"] for r in rows]

    plt.figure(figsize=(10, 5))
    bars = plt.barh(names, dbs, color="indianred")
    plt.title("Davies-Bouldin par variante (plus petit = mieux)")
    plt.xlabel("Davies-Bouldin")
    for bar, v in zip(bars, dbs):
        plt.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def main():
    # 1) Données synthétiques
    X, y_true = generate_synthetic_data(
        n_per_cluster=60, k=3, d=2, seed=42, spread=0.7
    )
    print(f"Données : {X.shape}")

    # 2) Comparaison des variantes
    rows = run_variants(X, k=3)
    print_table(rows)

    # 3) Figures
    plot_silhouette_comparison(
        rows, os.path.join(RESULTS_DIR, "silhouette_variants.png")
    )
    plot_db_comparison(
        rows, os.path.join(RESULTS_DIR, "davies_bouldin_variants.png")
    )

    # 4) Courbe du coude
    plot_elbow(
        X, k_values=[1, 2, 3, 4, 5, 6, 7, 8],
        output_path=os.path.join(RESULTS_DIR, "elbow_curve.png")
    )

    print("\n🎉 Évaluation terminée. Figures dans :", RESULTS_DIR)


if __name__ == "__main__":
    main()