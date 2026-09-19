# main.py

import argparse
import os
import sys
import numpy as np

from src.nuees_dynamiques.io import load_csv, save_history
from src.nuees_dynamiques.preprocessing import (
    handle_missing_values, standardize, normalize_minmax
)
from src.nuees_dynamiques.distances import (
    euclidean_distance_squared, euclidean_distance,
    manhattan_distance, minkowski_distance
)
from src.nuees_dynamiques.pipeline import run_single, run_full


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Algorithme des nuées dynamiques (from scratch).\n"
            "  • Sans --data ni --k : mode FULL (génère tout automatiquement).\n"
            "  • Avec --data et --k : mode SINGLE (une expérience)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # === Mode single ===
    parser.add_argument("--data", type=str, default=None,
                        help="Chemin du CSV (mode single).")
    parser.add_argument("--k", type=int, default=None,
                        help="Nombre de classes (mode single).")

    # === Paramètres communs ===
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init", type=str, default="kmpp",
                        choices=["random", "kmpp", "uniform"])
    parser.add_argument("--nuee-type", type=str, default="point",
                        choices=["point", "ensemble_points",
                                 "distribution", "axe_factoriel",
                                 "structure"])
    parser.add_argument("--m", type=int, default=3,
                        help="Nombre de représentants (ensemble_points).")
    parser.add_argument("--prototype", type=str, default="centroide",
                        choices=["centroide", "mediane", "medoide"])
    parser.add_argument("--distance", type=str, default="euclidienne_carree",
                        choices=["euclidienne", "euclidienne_carree",
                                 "manhattan", "minkowski"])
    parser.add_argument("--minkowski-p", type=float, default=3.0)
    parser.add_argument("--missing", type=str, default="mean",
                        choices=["mean", "median", "zero", "drop"])
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Dossier de sortie "
                             "(défaut : results/single ou results/full).")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()


def get_distance_fn(name, minkowski_p=3.0):
    name = name.lower()
    if name in ("euclidienne", "euclidean", "l2"):
        return euclidean_distance
    elif name in ("euclidienne_carree", "euclidean_squared", "l2_squared"):
        return euclidean_distance_squared
    elif name in ("manhattan", "l1", "cityblock"):
        return manhattan_distance
    elif name in ("minkowski",):
        return lambda x, c: minkowski_distance(x, c, p=minkowski_p)
    raise ValueError(f"Distance inconnue : {name}")


def main():
    args = parse_args()

    # ---------------------------------------------
    # Choix automatique du mode
    # ---------------------------------------------
    single_mode = (args.data is not None and args.k is not None)

    if not single_mode:
        # Mode FULL : tout automatique
        output_dir = args.output or "results/full"
        print("=" * 70)
        print("MODE FULL — exécution automatique du pipeline complet")
        print("=" * 70)
        run_full(output_dir=output_dir, seed=args.seed, verbose=args.verbose)
        return

    # ---------------------------------------------
    # Mode SINGLE
    # ---------------------------------------------
    output_dir = args.output or "results/single"
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 Chargement : {args.data}")
    X = load_csv(args.data)
    print(f"   → {X.shape[0]} points, {X.shape[1]} dimensions")

    if np.any(np.isnan(X)):
        print(f"🧹 NaN → stratégie : {args.missing}")
        X = handle_missing_values(X, strategy=args.missing)

    if args.standardize:
        print("📏 Standardisation")
        X = standardize(X)
    elif args.normalize:
        print("📏 Normalisation Min-Max")
        X = normalize_minmax(X)

    distance_fn = get_distance_fn(args.distance, args.minkowski_p)

    run_single(
        X, k=args.k,
        nuee_type=args.nuee_type,
        prototype_type=args.prototype,
        distance_fn=distance_fn,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        init_method=args.init,
        m=args.m,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    # Sauvegarde de l'historique au format CSV
    # (récupéré dans run_single via la valeur retournée)
    # Pour rester simple, on relit summary.txt et on régénère le CSV si besoin.
    # (facultatif)
    print(f"\n✅ Mode single terminé. Résultats dans : {output_dir}")


if __name__ == "__main__":
    main()