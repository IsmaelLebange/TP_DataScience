# src/nuees_dynamiques/pipeline.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .preprocessing import generate_synthetic_data
from .algorithm import nuees_dynamiques
from .evaluation import silhouette_score, davies_bouldin_score
from .visualization import (
    plot_nuees_2d, plot_convergence, plot_cluster_sizes
)
from .distances import (
    euclidean_distance_squared, euclidean_distance, manhattan_distance
)


NUEES_TYPES = ["point", "ensemble_points",
               "distribution", "axe_factoriel", "structure"]


# =============================================================
# MODE SINGLE
# =============================================================

def run_single(X, k, nuee_type="point", prototype_type="centroide",
               distance_fn=euclidean_distance_squared,
               max_iter=100, tol=1e-4, seed=42,
               init_method="kmpp", m=3,
               output_dir="results/single", verbose=False):
    """
    Exécute une seule expérience et sauvegarde les résultats.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n🚀 Nuées dynamiques — single run")
    print(f"   k={k}, nuee_type={nuee_type}, prototype={prototype_type}")

    result = nuees_dynamiques(
        X, k=k, max_iter=max_iter, tol=tol, seed=seed,
        init_method=init_method, nuee_type=nuee_type,
        prototype_type=prototype_type,
        distance_fn=distance_fn, verbose=verbose, m=m,
    )

    # Sauvegardes
    np.save(os.path.join(output_dir, "labels.npy"), result["labels"])
    np.save(os.path.join(output_dir, "inertia.npy"),
            np.array(result["final_inertia"]))

    # Résumé texte
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"k: {k}\n")
        f.write(f"nuee_type: {nuee_type}\n")
        f.write(f"prototype_type: {prototype_type}\n")
        f.write(f"n_iter: {result['n_iter']}\n")
        f.write(f"stop_reason: {result['stop_reason']}\n")
        f.write(f"final_inertia: {result['final_inertia']:.6f}\n")
        f.write(f"history: {result['history']}\n")

    print(f"   n_iter = {result['n_iter']} | "
          f"stop = {result['stop_reason']} | "
          f"W = {result['final_inertia']:.4f}")

    # Figures si 2D
    if X.shape[1] >= 2:
        try:
            plot_nuees_2d(
                X, result["labels"], result["nuees"],
                nuee_type=nuee_type,
                title=f"Nuées dynamiques — {nuee_type} (k={k})",
                output_path=os.path.join(output_dir, "clusters.png")
            )
            plot_convergence(
                result["history"],
                title=f"Convergence — {nuee_type}",
                output_path=os.path.join(output_dir, "convergence.png")
            )
            plot_cluster_sizes(
                result["labels"], k=k,
                title=f"Tailles des clusters — {nuee_type}",
                output_path=os.path.join(output_dir, "sizes.png")
            )
        except Exception as e:
            print(f"   ⚠️  Figures non générées : {e}")

    print(f"📁 Résultats dans : {output_dir}")
    return result


# =============================================================
# MODE FULL (pipeline complet)
# =============================================================

def _run_one_variant(X, k, nuee_type, prototype_type, distance_fn,
                     distance_eval, seed, output_dir, verbose=False):
    """Lance une variante et retourne ses métriques."""
    try:
        res = nuees_dynamiques(
            X, k=k, max_iter=100, tol=1e-6, seed=seed,
            init_method="kmpp", nuee_type=nuee_type,
            prototype_type=prototype_type,
            distance_fn=distance_fn, verbose=False, m=3,
        )
        sil = silhouette_score(X, res["labels"], distance_fn=distance_eval)
        try:
            db = davies_bouldin_score(X, res["labels"],
                                      res["nuees"]["means"]
                                      if nuee_type == "distribution"
                                      else res["nuees"],
                                      distance_fn=distance_eval)
        except Exception:
            db = float("nan")

        return {
            "nuee_type": nuee_type,
            "prototype_type": prototype_type,
            "inertia": res["final_inertia"],
            "silhouette": sil,
            "davies_bouldin": db,
            "n_iter": res["n_iter"],
            "stop_reason": res["stop_reason"],
            "labels": res["labels"],
            "nuees": res["nuees"],
            "history": res["history"],
        }
    except Exception as e:
        print(f"   ⚠️  {nuee_type} / {prototype_type} a échoué : {e}")
        return None


def run_full(output_dir="results/full", seed=42, verbose=True):
    """
    Pipeline complet :
    1. Génère des données synthétiques (2D, 3D, 5 clusters)
    2. Lance toutes les variantes des nuées dynamiques
    3. Évalue (silhouette, Davies-Bouldin)
    4. Génère les figures
    5. Produit un résumé
    """
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("PIPELINE COMPLET — NUÉES DYNAMIQUES")
    print("=" * 70)

    # -------------------------------------------------
    # 1. Jeux de données synthétiques
    # -------------------------------------------------
    print("\n[1/5] Génération des données synthétiques...")
    datasets = {
        "2D_k3": generate_synthetic_data(n_per_cluster=60, k=3, d=2,
                                          seed=seed, spread=0.7),
        "2D_k5": generate_synthetic_data(n_per_cluster=40, k=5, d=2,
                                          seed=seed + 1, spread=0.6),
        "3D_k3": generate_synthetic_data(n_per_cluster=60, k=3, d=3,
                                          seed=seed + 2, spread=0.7),
    }
    for name, (X, y) in datasets.items():
        print(f"   {name} : {X.shape}")

    # -------------------------------------------------
    # 2. Variantes à tester
    # -------------------------------------------------
    variants = [
        ("point", "centroide", euclidean_distance_squared, euclidean_distance),
        ("point", "mediane", manhattan_distance, manhattan_distance),
        ("point", "medoide", euclidean_distance_squared, euclidean_distance),
        ("ensemble_points", "centroide", euclidean_distance_squared, euclidean_distance),
        ("distribution", "centroide", euclidean_distance_squared, euclidean_distance),
        ("axe_factoriel", "centroide", euclidean_distance_squared, euclidean_distance),
        ("structure", "centroide", euclidean_distance_squared, euclidean_distance),
    ]

    # -------------------------------------------------
    # 3. Exécution sur 2D_k3 (référence)
    # -------------------------------------------------
    print("\n[2/5] Exécution sur 2D_k3...")
    X_ref, _ = datasets["2D_k3"]
    k_ref = 3
    rows = []

    for nt, pt, dfit, deval in variants:
        label = f"{nt}/{pt}" if nt == "point" else nt
        print(f"   → {label}")
        r = _run_one_variant(X_ref, k_ref, nt, pt, dfit, deval,
                             seed, fig_dir, verbose=False)
        if r is None:
            continue
        rows.append(r)
        print(f"      W={r['inertia']:.2f} | "
              f"Sil={r['silhouette']:.3f} | "
              f"DB={r['davies_bouldin']:.3f}")

    # -------------------------------------------------
    # 4. Figures
    # -------------------------------------------------
    print("\n[3/5] Génération des figures...")
    for r in rows:
        nt = r["nuee_type"]
        try:
            plot_nuees_2d(
                X_ref, r["labels"], r["nuees"], nuee_type=nt,
                title=f"Nuées — {nt} ({r['prototype_type']})",
                output_path=os.path.join(
                    fig_dir, f"nuee_{nt}_{r['prototype_type']}.png"
                )
            )
            plot_convergence(
                r["history"],
                title=f"Convergence — {nt} ({r['prototype_type']})",
                output_path=os.path.join(
                    fig_dir, f"conv_{nt}_{r['prototype_type']}.png"
                )
            )
        except Exception as e:
            print(f"   ⚠️  Figure {nt} échouée : {e}")

    # Comparaison silhouette & DB
    labels_txt = [f"{r['nuee_type']}/{r['prototype_type']}" for r in rows]
    sils = [r["silhouette"] for r in rows]
    dbs = [r["davies_bouldin"] for r in rows]

    plt.figure(figsize=(10, 5))
    plt.barh(labels_txt, sils, color="mediumseagreen")
    plt.title("Silhouette par variante (plus grand = mieux)")
    plt.xlabel("Silhouette")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "compare_silhouette.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.barh(labels_txt, dbs, color="indianred")
    plt.title("Davies-Bouldin par variante (plus petit = mieux)")
    plt.xlabel("Davies-Bouldin")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "compare_db.png"), dpi=150)
    plt.close()

    # Courbe du coude (nuée point / centroïde)
    print("   → Courbe du coude...")
    ks = [1, 2, 3, 4, 5, 6, 7, 8]
    inertias = []
    for k_val in ks:
        r = nuees_dynamiques(
            X_ref, k=k_val, max_iter=100, tol=1e-6, seed=seed,
            init_method="kmpp", nuee_type="point",
            prototype_type="centroide",
            distance_fn=euclidean_distance_squared, verbose=False
        )
        inertias.append(r["final_inertia"])
    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o", color="steelblue", linewidth=2)
    plt.title("Courbe du coude — inertie vs k")
    plt.xlabel("k")
    plt.ylabel("Inertie W")
    plt.grid(True, alpha=0.3)
    plt.xticks(ks)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "elbow_curve.png"), dpi=150)
    plt.close()

    # -------------------------------------------------
    # 5. Résumé final
    # -------------------------------------------------
    print("\n[4/5] Résumé des résultats :")
    print("-" * 80)
    print(f"{'Variante':<30}{'Inertie':>10}{'Silhouette':>14}{'Davies-Bouldin':>16}")
    print("-" * 80)
    for r in rows:
        label = f"{r['nuee_type']}/{r['prototype_type']}"
        print(f"{label:<30}{r['inertia']:>10.2f}"
              f"{r['silhouette']:>14.3f}{r['davies_bouldin']:>16.3f}")
    print("-" * 80)

    # Meilleure variante (silhouette max)
    if rows:
        best = max(rows, key=lambda r: r["silhouette"])
        print(f"\n🏆 Meilleure variante : "
              f"{best['nuee_type']}/{best['prototype_type']} "
              f"(silhouette = {best['silhouette']:.3f})")

    # Sauvegarde du résumé
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write("PIPELINE COMPLET — NUÉES DYNAMIQUES\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"seed: {seed}\n")
        f.write(f"jeux: {list(datasets.keys())}\n\n")
        f.write("Résultats sur 2D_k3 :\n")
        for r in rows:
            label = f"{r['nuee_type']}/{r['prototype_type']}"
            f.write(f"  {label:<30} "
                    f"W={r['inertia']:.4f} "
                    f"Sil={r['silhouette']:.4f} "
                    f"DB={r['davies_bouldin']:.4f} "
                    f"n_iter={r['n_iter']}\n")

    print(f"\n[5/5] 📁 Résultats dans : {output_dir}")
    print(f"      📊 Figures dans  : {fig_dir}")
    print(f"      📝 Résumé dans   : {output_dir}/summary.txt")
    print("\n✅ Pipeline complet terminé.")

    return {
        "rows": rows,
        "datasets": datasets,
        "output_dir": output_dir,
    }