# Nuées dynamiques (from scratch)

Implémentation **from scratch** de l'algorithme des **nuées dynamiques** de Diday, dans le cadre d'un TP de Science des Données.

Le projet couvre le **cadre général** des nuées dynamiques et propose **cinq types de nuées** au choix :

| Type de nuée      | Description                                           | Cas particulier              |
| ----------------- | ----------------------------------------------------- | ---------------------------- |
| `point`           | Un prototype ponctuel (centroïde, médiane ou médoïde) | → **k-means** (centroïde)    |
| `ensemble_points` | m points représentatifs par nuée                      | Extension                    |
| `distribution`    | Gaussienne (moyenne + covariance)                     | → Mélange gaussien simplifié |
| `axe_factoriel`   | Droite principale (ACP locale)                        | Extension                    |
| `structure`       | Segment représentatif                                 | Extension                    |

Le cas `point` avec un **centroïde** ramène l'algorithme à **k-means**. Les autres cas illustrent la généralité du cadre de Diday.

---

## Structure du projet

```text
nuees_dynamiques/
├── data/
│   ├── synthetic/          # jeux synthétiques (committés)
│   ├── real/               # Iris, Wine (committés)
│   └── raw/                # datasets externes (prof, etc.)
│
├── src/nuees_dynamiques/
│   ├── io.py               # chargement / sauvegarde CSV
│   ├── preprocessing.py    # normalisation, standardisation, NaN
│   ├── distances.py        # euclidienne, Manhattan, Minkowski
│   ├── initialization.py   # random, k-means++, uniforme
│   ├── assignment.py       # affectation des points
│   ├── update.py           # mise à jour des prototypes (point)
│   ├── nuees.py            # gestion des 5 types de nuées
│   ├── inertia.py          # critère W
│   ├── convergence.py      # critères d'arrêt
│   ├── algorithm.py        # boucle principale générique
│   ├── evaluation.py       # silhouette, Davies-Bouldin, coude
│   ├── visualization.py    # figures 2D / 3D
│   ├── pipeline.py         # orchestration (single / full)
│   └── utils.py            # validation
│
├── tests/                  # tests unitaires
├── scripts/
│   ├── export_datasets.py  # génère les CSV (synthétiques + réels)
│   ├── evaluate.py         # comparaison des variantes
│   └── visualize.py        # figures
│
├── results/                # sorties générées
├── main.py                 # point d'entrée CLI (2 modes)
├── app.py                  # interface graphique Tkinter
├── requirements.txt
└── README.md
```

---

## Installation

Prérequis : Python ≥ 3.8.

```bash
git clone <url_du_depot>
cd nuees_dynamiques

python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

`requirements.txt` :

```txt
numpy
matplotlib
pytest
```

> Tkinter est fourni avec Python (sur Linux minimal : `sudo apt install python3-tk`).
>
> `scikit-learn` est utilisé **uniquement** par `scripts/export_datasets.py` pour exporter Iris et Wine en CSV. Il n'est pas requis pour lancer l'algorithme.

---

## Datasets

Trois sources disponibles :

* `data/synthetic/` — jeux synthétiques générés par nos soins :

  * `test_3clusters_2d.csv`
  * `test_5clusters_2d.csv`
  * `test_3clusters_3d.csv`
* `data/real/` — datasets réels exportés depuis sklearn :

  * `iris.csv`
  * `wine.csv`
* `data/raw/` — emplacement libre pour vos propres datasets.

Pour (re)générer les CSV :

```bash
pip install scikit-learn
python -m scripts.export_datasets
```

Format attendu pour un CSV externe : header en première ligne, séparateur virgule, colonnes numériques, valeurs manquantes = `NaN` ou vide.

---

## Utilisation

### Interface graphique (recommandé)

```bash
python app.py
```

Permet de :

* charger n'importe quel CSV (bouton **Choisir...**) ;
* régler k, type de nuée, prototype, distance, init, max iter, tol, seed ;
* cocher standardisation / normalisation ;
* lancer l'algorithme ;
* voir le résumé (inertie, silhouette, Davies-Bouldin, tailles) ;
* voir la figure des clusters avec les prototypes ;
* sauvegarder les résultats (labels, inertie, figure, résumé).

### Ligne de commande — mode FULL

Sans arguments, lance tout le pipeline (génération des données, 7 variantes, évaluation, figures, résumé) :

```bash
python main.py
```

Sortie dans `results/full/` :

```text
results/full/
├── figures/
│   ├── nuee_point_centroide.png
│   ├── nuee_distribution_centroide.png
│   ├── conv_point_centroide.png
│   ├── compare_silhouette.png
│   ├── compare_db.png
│   └── elbow_curve.png
└── summary.txt
```

### Ligne de commande — mode SINGLE

Une seule expérience paramétrée :

```bash
python main.py --data data/real/iris.csv --k 3 \
    --nuee-type point --prototype centroide \
    --standardize --verbose
```

Exemple complet :

```bash
python main.py \
  --data data/synthetic/test_3clusters_2d.csv \
  --k 3 \
  --nuee-type point \
  --prototype centroide \
  --distance euclidienne_carree \
  --init kmpp \
  --max-iter 100 \
  --tol 1e-6 \
  --seed 42 \
  --standardize \
  --output results/single \
  --verbose
```

Sortie dans `results/single/` :

```text
results/single/
├── labels.npy
├── inertia.npy
├── summary.txt
├── clusters.png
├── convergence.png
└── sizes.png
```

---

## Arguments CLI

| Argument        | Type  | Défaut               | Description                                                              |
| --------------- | ----- | -------------------- | ------------------------------------------------------------------------ |
| `--data`        | str   | —                    | Chemin du CSV (mode single)                                              |
| `--k`           | int   | —                    | Nombre de classes (mode single)                                          |
| `--max-iter`    | int   | 100                  | Nombre maximal d'itérations                                              |
| `--tol`         | float | 1e-4                 | Seuil de convergence                                                     |
| `--seed`        | int   | 42                   | Graine aléatoire                                                         |
| `--init`        | str   | `kmpp`               | `random`, `kmpp`, `uniform`                                              |
| `--nuee-type`   | str   | `point`              | `point`, `ensemble_points`, `distribution`, `axe_factoriel`, `structure` |
| `--m`           | int   | 3                    | Nombre de représentants (`ensemble_points`)                              |
| `--prototype`   | str   | `centroide`          | `centroide`, `mediane`, `medoide` (si nuée=point)                        |
| `--distance`    | str   | `euclidienne_carree` | `euclidienne`, `euclidienne_carree`, `manhattan`, `minkowski`            |
| `--minkowski-p` | float | 3.0                  | Paramètre p pour Minkowski                                               |
| `--missing`     | str   | `mean`               | `mean`, `median`, `zero`, `drop`                                         |
| `--standardize` | flag  | False                | Standardisation z-score                                                  |
| `--normalize`   | flag  | False                | Normalisation Min-Max                                                    |
| `--output`      | str   | `results/...`        | Dossier de sortie                                                        |
| `--verbose`     | flag  | False                | Afficher la progression                                                  |

Aide :

```bash
python main.py --help
```

---

## Tests

```bash
python -m tests.test_preprocessing
python -m tests.test_distances
python -m tests.test_initialization
python -m tests.test_assignment
python -m tests.test_update
python -m tests.test_inertia
python -m tests.test_convergence
python -m tests.test_algorithm
```

Ou :

```bash
pytest tests/ -v
```

---

## Scripts auxiliaires

```bash
python -m scripts.export_datasets   # génère les CSV
python -m scripts.evaluate          # comparaison des variantes
python -m scripts.visualize         # figures des 5 types de nuées
```

---

## Cadre théorique

L'algorithme alterne deux étapes :

1. **Affectation** — chaque point va vers la nuée la plus proche.
2. **Mise à jour** — chaque nuée est recalculée à partir de ses points.

On minimise l'inertie intra-classe :

```text
W = somme_j somme_{x dans C_j} d²(x, nuée_j)
```

Types de nuées supportés :

| Type                | Représentation        | Distance adaptée |
| ------------------- | --------------------- | ---------------- |
| `point` (centroïde) | vecteur moyen         | euclidienne²     |
| `point` (médiane)   | médiane coordonnée    | Manhattan        |
| `point` (médoïde)   | point réel            | quelconque       |
| `ensemble_points`   | m points              | euclidienne      |
| `distribution`      | (moyenne, covariance) | Mahalanobis      |
| `axe_factoriel`     | (point, direction)    | orthogonale      |
| `structure`         | (p1, p2)              | point-segment    |

---

## Résultats (jeu 2D, 3 clusters)

| Variante          | Inertie | Silhouette | Davies-Bouldin |
| ----------------- | ------- | ---------- | -------------- |
| point / centroide | 145.93  | **0.668**  | **0.467**      |
| point / medoide   | 148.87  | 0.668      | 0.495          |
| point / mediane   | 181.28  | 0.644      | 0.602          |
| ensemble_points   | 152.10  | 0.651      | 0.512          |
| distribution      | 145.93  | 0.668      | 0.467          |
| axe_factoriel     | 155.44  | 0.638      | 0.541          |
| structure         | 162.10  | 0.612      | 0.583          |

La variante `point / centroide` (→ k-means) donne les meilleurs résultats sur des clusters compacts et isotropes, ce qui est attendu.

---

## Limites

* Implémentation naïve (boucles explicites, pas de vectorisation avancée).
* Complexité en `O(T · n · k · d)` par itération.
* Pas de gestion du big data ni de calcul distribué.
* `axe_factoriel` et `structure` sont des versions simplifiées.
* `distribution` n'utilise pas l'algorithme EM complet.

## Perspectives

* Optimisations : vectorisation NumPy, kd-tree, Elkan, mini-batch.
* Extensions : mélanges gaussiens avec EM, nuées floues (fuzzy), nuées symboliques.
* Déploiement : API REST, package PyPI.

---

## Références

* Diday, E. (1971). *Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques.* Revue de Statistique Appliquée.
* MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.*
* Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.*
* Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
* Davies, D. L., & Bouldin, D. W. (1979). *A Cluster Separation Measure.*

---

## Auteur

**Ismaël Lebange** — TP Science des Données

## Licence

Projet académique — usage pédagogique uniquement.
