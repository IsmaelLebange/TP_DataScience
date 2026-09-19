# src/nuees_dynamiques/visualization.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _colors(k):
    return plt.cm.tab10(np.linspace(0, 1, max(k, 10)))


# =============================================================
# 2D : nuée = point
# =============================================================

def plot_nuee_point_2d(X, labels, nuees, title="Nuées dynamiques — 2D",
                       output_path=None):
    k = nuees.shape[0]
    colors = _colors(k)

    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=30, alpha=0.6,
                    color=colors[j], label=f"Cluster {j} ({len(idx)})")
    plt.scatter(nuees[:, 0], nuees[:, 1],
                s=250, marker="X", edgecolor="black",
                linewidth=2, color="red", zorder=5, label="Prototypes")
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


# =============================================================
# 2D : nuée = ensemble de points représentatifs
# =============================================================

def plot_nuee_ensemble_points_2d(X, labels, nuees,
                                 title="Nuées dynamiques — ensemble de points",
                                 output_path=None):
    k, m, d = nuees.shape
    colors = _colors(k)

    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        plt.scatter(nuees[j, :, 0], nuees[j, :, 1],
                    s=200, marker="X", edgecolor="black",
                    linewidth=2, color=colors[j], zorder=5)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


# =============================================================
# 2D : nuée = distribution (gaussienne)
# =============================================================

def _ellipse_from_cov(mean, cov, n_std=2.0, n_points=100):
    """Génère une ellipse à partir d'une gaussienne 2D."""
    eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    t = np.linspace(0, 2 * np.pi, n_points)
    ellipse = np.array([width / 2 * np.cos(t),
                        height / 2 * np.sin(t)])
    R = np.array([[np.cos(np.radians(angle)),
                   -np.sin(np.radians(angle))],
                  [np.sin(np.radians(angle)),
                   np.cos(np.radians(angle))]])
    ellipse = R @ ellipse
    return ellipse + mean[:2, None]


def plot_nuee_distribution_2d(X, labels, nuees,
                              title="Nuées dynamiques — distributions",
                              output_path=None):
    means = nuees["means"]
    covs = nuees["covs"]
    k = means.shape[0]
    colors = _colors(k)

    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.5,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        ell = _ellipse_from_cov(means[j], covs[j], n_std=2.0)
        plt.plot(ell[0], ell[1], color=colors[j], linewidth=2)
        plt.scatter(means[j, 0], means[j, 1],
                    s=150, marker="X", color=colors[j],
                    edgecolor="black", zorder=5)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


# =============================================================
# 2D : nuée = axe factoriel
# =============================================================

def plot_nuee_axe_2d(X, labels, nuees,
                     title="Nuées dynamiques — axes factoriels",
                     output_path=None, line_length=5.0):
    pts = nuees["points"]
    dirs = nuees["dirs"]
    k = pts.shape[0]
    colors = _colors(k)

    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        p = pts[j]
        d = dirs[j]
        p1 = p - line_length * d
        p2 = p + line_length * d
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=colors[j], linewidth=2.5)
        plt.scatter(p[0], p[1], s=150, marker="X",
                    color=colors[j], edgecolor="black", zorder=5)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


# =============================================================
# 2D : nuée = structure (segment)
# =============================================================

def plot_nuee_structure_2d(X, labels, nuees,
                           title="Nuées dynamiques — structures",
                           output_path=None):
    k = nuees.shape[0]
    colors = _colors(k)

    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        p1 = nuees[j, 0]
        p2 = nuees[j, 1]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=colors[j], linewidth=2.5)
        plt.scatter([p1[0], p2[0]], [p1[1], p2[1]],
                    s=150, marker="X", color=colors[j],
                    edgecolor="black", zorder=5)
    plt.title(title)
    plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(loc="best", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


# =============================================================
# Dispatcher global
# =============================================================

def plot_nuees_2d(X, labels, nuees, nuee_type="point",
                  title=None, output_path=None):
    if title is None:
        title = f"Nuées dynamiques — {nuee_type} (2D)"
    nt = nuee_type.lower()
    if nt == "point":
        plot_nuee_point_2d(X, labels, nuees, title=title,
                           output_path=output_path)
    elif nt == "ensemble_points":
        plot_nuee_ensemble_points_2d(X, labels, nuees, title=title,
                                     output_path=output_path)
    elif nt == "distribution":
        plot_nuee_distribution_2d(X, labels, nuees, title=title,
                                  output_path=output_path)
    elif nt == "axe_factoriel":
        plot_nuee_axe_2d(X, labels, nuees, title=title,
                         output_path=output_path)
    elif nt == "structure":
        plot_nuee_structure_2d(X, labels, nuees, title=title,
                               output_path=output_path)
    else:
        raise ValueError(f"nuee_type inconnu : {nuee_type}")


# =============================================================
# Courbes et histogrammes
# =============================================================

def plot_convergence(history, title="Convergence — inertie vs itérations",
                     output_path=None):
    iterations = list(range(1, len(history) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, history, marker="o", linewidth=2, color="steelblue")
    plt.title(title)
    plt.xlabel("Itération")
    plt.ylabel("Inertie W")
    plt.grid(True, alpha=0.3)
    plt.xticks(iterations)
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()


def plot_cluster_sizes(labels, k, title="Tailles des clusters",
                       output_path=None):
    sizes = np.zeros(k, dtype=int)
    for lab in labels:
        sizes[lab] += 1
    plt.figure(figsize=(7, 5))
    bars = plt.bar(range(k), sizes, color="mediumseagreen")
    plt.title(title)
    plt.xlabel("Cluster"); plt.ylabel("Nombre de points")
    plt.xticks(range(k))
    for bar, v in zip(bars, sizes):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                 str(v), ha="center")
    plt.tight_layout()
    if output_path:
        _ensure_dir(output_path)
        plt.savefig(output_path, dpi=150)
        print(f"Figure sauvegardée : {output_path}")
    plt.close()