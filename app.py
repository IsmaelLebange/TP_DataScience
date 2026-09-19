# app.py

import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.nuees_dynamiques.io import load_csv
from src.nuees_dynamiques.preprocessing import (
    handle_missing_values, standardize, normalize_minmax
)
from src.nuees_dynamiques.distances import (
    euclidean_distance_squared, euclidean_distance,
    manhattan_distance
)
from src.nuees_dynamiques.algorithm import nuees_dynamiques
from src.nuees_dynamiques.evaluation import (
    silhouette_score, davies_bouldin_score
)


class NueesApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Nuées dynamiques — Interface")
        self.geometry("1100x750")
        self.minsize(950, 650)

        self.X = None          # données chargées
        self.result = None     # dernier résultat
        self.distance_fn = euclidean_distance_squared

        self._build_ui()

    # -------------------------------------------------
    # Construction de l'interface
    # -------------------------------------------------
    def _build_ui(self):
        # ---- Panneau gauche : paramètres ----
        left = ttk.Frame(self, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Fichier CSV",
                  font=("TkDefaultFont", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 5))

        self.var_file = tk.StringVar(value="Aucun fichier")
        ttk.Entry(left, textvariable=self.var_file, width=32,
                  state="readonly").grid(row=1, column=0, sticky="we")
        ttk.Button(left, text="Choisir...",
                   command=self.on_choose_file).grid(
            row=1, column=1, padx=(5, 0))

        ttk.Separator(left, orient="horizontal").grid(
            row=2, column=0, columnspan=2, sticky="we", pady=10)

        # k
        ttk.Label(left, text="Nombre de classes (k)").grid(
            row=3, column=0, sticky="w")
        self.var_k = tk.IntVar(value=3)
        ttk.Spinbox(left, from_=1, to=50, textvariable=self.var_k,
                    width=8).grid(row=3, column=1, sticky="e")

        # Type de nuée
        ttk.Label(left, text="Type de nuée").grid(
            row=4, column=0, sticky="w", pady=(5, 0))
        self.var_nuee = tk.StringVar(value="point")
        ttk.Combobox(left, textvariable=self.var_nuee, state="readonly",
                     values=["point", "ensemble_points", "distribution",
                             "axe_factoriel", "structure"],
                     width=20).grid(row=4, column=1, sticky="e",
                                    pady=(5, 0))

        # Prototype
        ttk.Label(left, text="Prototype (si nuée=point)").grid(
            row=5, column=0, sticky="w", pady=(5, 0))
        self.var_proto = tk.StringVar(value="centroide")
        ttk.Combobox(left, textvariable=self.var_proto, state="readonly",
                     values=["centroide", "mediane", "medoide"],
                     width=20).grid(row=5, column=1, sticky="e",
                                    pady=(5, 0))

        # Distance
        ttk.Label(left, text="Distance").grid(
            row=6, column=0, sticky="w", pady=(5, 0))
        self.var_dist = tk.StringVar(value="euclidienne_carree")
        ttk.Combobox(left, textvariable=self.var_dist, state="readonly",
                     values=["euclidienne", "euclidienne_carree",
                             "manhattan"],
                     width=20).grid(row=6, column=1, sticky="e",
                                    pady=(5, 0))

        # Init
        ttk.Label(left, text="Initialisation").grid(
            row=7, column=0, sticky="w", pady=(5, 0))
        self.var_init = tk.StringVar(value="kmpp")
        ttk.Combobox(left, textvariable=self.var_init, state="readonly",
                     values=["random", "kmpp", "uniform"],
                     width=20).grid(row=7, column=1, sticky="e",
                                    pady=(5, 0))

        # Max iter
        ttk.Label(left, text="Max itérations").grid(
            row=8, column=0, sticky="w", pady=(5, 0))
        self.var_maxiter = tk.IntVar(value=100)
        ttk.Spinbox(left, from_=1, to=1000,
                    textvariable=self.var_maxiter, width=8).grid(
            row=8, column=1, sticky="e", pady=(5, 0))

        # Tol
        ttk.Label(left, text="Tolérance").grid(
            row=9, column=0, sticky="w", pady=(5, 0))
        self.var_tol = tk.StringVar(value="1e-4")
        ttk.Entry(left, textvariable=self.var_tol, width=10).grid(
            row=9, column=1, sticky="e", pady=(5, 0))

        # Seed
        ttk.Label(left, text="Graine (seed)").grid(
            row=10, column=0, sticky="w", pady=(5, 0))
        self.var_seed = tk.IntVar(value=42)
        ttk.Spinbox(left, from_=0, to=99999,
                    textvariable=self.var_seed, width=8).grid(
            row=10, column=1, sticky="e", pady=(5, 0))

        ttk.Separator(left, orient="horizontal").grid(
            row=11, column=0, columnspan=2, sticky="we", pady=10)

        # Prétraitement
        self.var_std = tk.BooleanVar(value=True)
        self.var_norm = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Standardiser (z-score)",
                        variable=self.var_std).grid(
            row=12, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(left, text="Normaliser (min-max)",
                        variable=self.var_norm).grid(
            row=13, column=0, columnspan=2, sticky="w")

        ttk.Separator(left, orient="horizontal").grid(
            row=14, column=0, columnspan=2, sticky="we", pady=10)

        # Boutons
        self.btn_run = ttk.Button(left, text="🚀 Lancer",
                                  command=self.on_run)
        self.btn_run.grid(row=15, column=0, columnspan=2,
                          sticky="we", pady=(0, 5))
        ttk.Button(left, text="💾 Sauvegarder les résultats",
                   command=self.on_save).grid(
            row=16, column=0, columnspan=2, sticky="we")

        # ---- Panneau droit : résultats ----
        right = ttk.Frame(self, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Zone texte de résultats
        ttk.Label(right, text="Résultats",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.text = tk.Text(right, height=10, wrap="word")
        self.text.pack(fill=tk.X, pady=(5, 10))
        self.text.configure(state="disabled")

        # Figure
        ttk.Label(right, text="Visualisation",
                  font=("TkDefaultFont", 10, "bold")).pack(anchor="w")

        self.fig = Figure(figsize=(6, 4.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Barre de statut
        self.status = tk.StringVar(value="Prêt.")
        ttk.Label(self, textvariable=self.status,
                  relief="sunken", anchor="w").pack(
            side=tk.BOTTOM, fill=tk.X)

    # -------------------------------------------------
    # Callbacks
    # -------------------------------------------------
    def on_choose_file(self):
        path = filedialog.askopenfilename(
            title="Choisir un fichier CSV",
            filetypes=[("CSV", "*.csv"), ("Tous les fichiers", "*.*")],
        )
        if not path:
            return
        try:
            X = load_csv(path)
            if np.any(np.isnan(X)):
                X = handle_missing_values(X, strategy="mean")
            self.X = X
            self.var_file.set(os.path.basename(path))
            self.status.set(
                f"Fichier chargé : {X.shape[0]} points, {X.shape[1]} dim."
            )
        except Exception as e:
            messagebox.showerror("Erreur de chargement", str(e))

    def _get_distance_fn(self, name):
        if name == "euclidienne":
            return euclidean_distance
        elif name == "euclidienne_carree":
            return euclidean_distance_squared
        elif name == "manhattan":
            return manhattan_distance
        raise ValueError(name)

    def on_run(self):
        if self.X is None:
            messagebox.showwarning("Aucune donnée",
                                   "Choisis d'abord un fichier CSV.")
            return

        # Désactiver le bouton pour éviter double-clic
        self.btn_run.configure(state="disabled")
        self.status.set("Exécution en cours...")
        self._clear_text()

        # Lancer dans un thread pour ne pas figer l'UI
        threading.Thread(target=self._run_algorithm, daemon=True).start()

    def _run_algorithm(self):
        try:
            X = self.X.copy()

            # Prétraitement
            if self.var_std.get():
                X = standardize(X)
            elif self.var_norm.get():
                X = normalize_minmax(X)

            k = int(self.var_k.get())
            nuee_type = self.var_nuee.get()
            prototype = self.var_proto.get()
            distance_fn = self._get_distance_fn(self.var_dist.get())
            init_method = self.var_init.get()
            max_iter = int(self.var_maxiter.get())
            tol = float(self.var_tol.get())
            seed = int(self.var_seed.get())

            # Lancement
            res = nuees_dynamiques(
                X, k=k, max_iter=max_iter, tol=tol, seed=seed,
                init_method=init_method, nuee_type=nuee_type,
                prototype_type=prototype,
                distance_fn=distance_fn, verbose=False, m=3,
            )

            # Métriques
            try:
                sil = silhouette_score(X, res["labels"],
                                       distance_fn=distance_fn)
            except Exception:
                sil = float("nan")

            try:
                # Prototypes pour DB
                if nuee_type == "distribution":
                    protos = res["nuees"]["means"]
                elif nuee_type == "axe_factoriel":
                    protos = res["nuees"]["points"]
                elif nuee_type == "structure":
                    protos = res["nuees"][:, 0]
                elif nuee_type == "ensemble_points":
                    protos = res["nuees"][:, 0]
                else:
                    protos = res["nuees"]
                db = davies_bouldin_score(X, res["labels"], protos,
                                          distance_fn=distance_fn)
            except Exception:
                db = float("nan")

            self.result = {"X": X, "res": res, "sil": sil, "db": db,
                           "nuee_type": nuee_type}

            # Mise à jour UI depuis le thread principal
            self.after(0, self._update_ui)

        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _update_ui(self):
        r = self.result
        res = r["res"]
        labels = res["labels"]
        k = int(self.var_k.get())

        # Texte
        lines = []
        lines.append(f"Type de nuée     : {r['nuee_type']}")
        lines.append(f"Prototype        : {self.var_proto.get()}")
        lines.append(f"Distance         : {self.var_dist.get()}")
        lines.append(f"Itérations       : {res['n_iter']}")
        lines.append(f"Raison d'arrêt   : {res['stop_reason']}")
        lines.append(f"Inertie finale   : {res['final_inertia']:.4f}")
        if len(res["history"]) > 0:
            lines.append(f"Inertie initiale : {res['history'][0]:.4f}")
        lines.append(f"Silhouette       : {r['sil']:.4f}")
        lines.append(f"Davies-Bouldin   : {r['db']:.4f}")
        lines.append("")
        lines.append("Tailles des clusters :")
        sizes = np.zeros(k, dtype=int)
        for lab in labels:
            sizes[lab] += 1
        for j in range(k):
            lines.append(f"  Cluster {j} : {sizes[j]} points")
        self._set_text("\n".join(lines))

        # Figure : 2D ou 3D (ou 1D)
        self._draw_figure(r["X"], labels, res)

        self.status.set("Terminé.")
        self.btn_run.configure(state="normal")

    def _on_error(self, msg):
        self.status.set("Erreur.")
        self.btn_run.configure(state="normal")
        messagebox.showerror("Erreur", msg)

    def _draw_figure(self, X, labels, res):
        self.ax.clear()

        n, d = X.shape
        k = len(np.unique(labels))
        colors = plt_colors(k)

        if d >= 2:
            for j in range(k):
                idx = np.where(labels == j)[0]
                if len(idx) == 0:
                    continue
                self.ax.scatter(X[idx, 0], X[idx, 1],
                                s=20, alpha=0.6, color=colors[j],
                                label=f"C{j}")
            # Prototypes selon le type
            nt = res.get("nuee_type", "point")
            try:
                if nt == "point":
                    self.ax.scatter(res["nuees"][:, 0], res["nuees"][:, 1],
                                    s=180, marker="X", edgecolor="black",
                                    color="red", zorder=5)
                elif nt == "ensemble_points":
                    for j in range(res["nuees"].shape[0]):
                        self.ax.scatter(res["nuees"][j, :, 0],
                                        res["nuees"][j, :, 1],
                                        s=120, marker="X",
                                        edgecolor="black",
                                        color=colors[j], zorder=5)
                elif nt == "distribution":
                    means = res["nuees"]["means"]
                    self.ax.scatter(means[:, 0], means[:, 1],
                                    s=180, marker="X", edgecolor="black",
                                    color="red", zorder=5)
                elif nt == "axe_factoriel":
                    pts = res["nuees"]["points"]
                    self.ax.scatter(pts[:, 0], pts[:, 1],
                                    s=180, marker="X", edgecolor="black",
                                    color="red", zorder=5)
                elif nt == "structure":
                    for j in range(res["nuees"].shape[0]):
                        p1 = res["nuees"][j, 0]
                        p2 = res["nuees"][j, 1]
                        self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                     color=colors[j], linewidth=2)
            except Exception:
                pass

            self.ax.set_xlabel("x1")
            self.ax.set_ylabel("x2")
            self.ax.set_title("Clusters (2 premières dimensions)")
        else:
            self.ax.text(0.5, 0.5, "Données 1D — pas de visualisation",
                         ha="center", va="center",
                         transform=self.ax.transAxes)

        self.ax.legend(loc="best", fontsize=8)
        self.ax.grid(True, alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw()

    def _set_text(self, content):
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")

    def _clear_text(self):
        self._set_text("")

    # -------------------------------------------------
    # Sauvegarde
    # -------------------------------------------------
    def on_save(self):
        if self.result is None:
            messagebox.showinfo("Rien à sauvegarder",
                                "Lance d'abord une exécution.")
            return
        out = filedialog.askdirectory(title="Dossier de sortie")
        if not out:
            return
        try:
            res = self.result["res"]
            np.save(os.path.join(out, "labels.npy"), res["labels"])
            np.save(os.path.join(out, "inertia.npy"),
                    np.array(res["final_inertia"]))
            self.fig.savefig(os.path.join(out, "clusters.png"), dpi=150)

            with open(os.path.join(out, "summary.txt"), "w") as f:
                f.write(f"nuee_type: {self.var_nuee.get()}\n")
                f.write(f"prototype: {self.var_proto.get()}\n")
                f.write(f"distance: {self.var_dist.get()}\n")
                f.write(f"k: {self.var_k.get()}\n")
                f.write(f"n_iter: {res['n_iter']}\n")
                f.write(f"stop_reason: {res['stop_reason']}\n")
                f.write(f"final_inertia: {res['final_inertia']:.6f}\n")
                f.write(f"silhouette: {self.result['sil']:.6f}\n")
                f.write(f"davies_bouldin: {self.result['db']:.6f}\n")
                f.write(f"history: {res['history']}\n")

            messagebox.showinfo("Sauvegarde OK",
                                f"Résultats sauvegardés dans :\n{out}")
        except Exception as e:
            messagebox.showerror("Erreur de sauvegarde", str(e))


def plt_colors(k):
    import matplotlib.pyplot as plt
    return plt.cm.tab10(np.linspace(0, 1, max(k, 10)))


if __name__ == "__main__":
    app = NueesApp()
    app.mainloop()