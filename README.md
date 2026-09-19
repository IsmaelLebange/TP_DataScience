# README — Implémentation des Nuées Dynamiques (from scratch)

## 📋 Description

Implémentation **from scratch** de l'algorithme des **nuées dynamiques** (Dynamic Clustering Algorithm) de Diday, dans le cadre d'un TP de Science des Données.

Ce projet couvre le **cadre général** des nuées dynamiques et propose **cinq types de nuées** au choix de l'utilisateur :

| Type de nuée      | Description                                           | Cas particulier              |
| ----------------- | ----------------------------------------------------- | ---------------------------- |
| `point`           | Un prototype ponctuel (centroïde, médiane ou médoïde) | → **k-means** (centroïde)    |
| `ensemble_points` | m points représentatifs par nuée                      | Extension                    |
| `distribution`    | Gaussienne (moyenne + covariance)                     | → Mélange gaussien simplifié |
| `axe_factoriel`   | Droite principale (ACP locale)                        | Extension                    |
| `structure`       | Segment représentatif                                 | Extension                    |

Le cas `point` avec un **centroïde** ramène l'algorithme à **k-means**. Les autres cas illustrent la généralité du cadre de Diday.

---

## 🗂️ Structure du projet

```text
nuees_dynamiques/
├── data/
│   ├── raw/                    # données brutes
│   ├── processed/              # données nettoyées
│   └── synthetic/              # jeux de données générés
│
├── src/nuees_dynamiques/
│   ├── __init__.py
│   ├── config.py               # paramètres par défaut
│   ├── io.py                   # chargement / sauvegarde
│   ├── preprocessing.py        # normalisation, standardisation
│   ├── distances.py            # distances (euclidienne, Manhattan, Minkowski)
│   ├── initialization.py       # init aléatoire, k-means++, uniforme
│   ├── assignment.py           # affectation des points
│   ├── update.py               # mise à jour des prototypes (point)
│   ├── nuees.py                # gestion des 5 types de nuées
│   ├── inertia.py              # critère d'inertie W
│   ├── convergence.py          # critères d'arrêt
│   ├── algorithm.py            # boucle principale générique
│   ├── evaluation.py           # silhouette, Davies-Bouldin, coude
│   ├── visualization.py        # figures 2D/3D
│   ├── pipeline.py             # orchestration (single / full)
│   └── utils.py                # validation, utilitaires
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_distances.py
│   ├── test_initialization.py
│   ├── test_assignment.py
│   ├── test_update.py
│   ├── test_inertia.py
│   ├── test_convergence.py
│   └── test_algorithm.py
│
├── scripts/
│   ├── generate_data.py        # génération de CSV synthétiques
│   ├── evaluate.py             # évaluation comparée
│   └── visualize.py            # figures
│
├── results/                    # résultats générés
├── main.py                     # point d'entrée unique
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Prérequis

* Python ≥ 3.8
* pip

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url_du_depot>
cd nuees_dynamiques

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

### `requirements.txt`

```txt
numpy
matplotlib
pytest
```

---

## 🚀 Utilisation

Le projet expose **un seul point d'entrée** : `main.py`. Il fonctionne selon **deux modes automatiques**.

### 🟢 Mode FULL — exécution complète (par défaut)

Pour lancer **tout** le pipeline :

```bash
python main.py
```

Le pipeline réalise :

1. Génération de jeux de données synthétiques (`2D_k3`, `2D_k5`, `3D_k3`).
2. Lancement des **7 variantes** :

   * `point / centroide`
   * `point / mediane`
   * `point / medoide`
   * `ensemble_points`
   * `distribution`
   * `axe_factoriel`
   * `structure`
3. Évaluation (silhouette, Davies-Bouldin).
4. Génération des figures (clusters, convergence, comparaisons, courbe du coude).
5. Résumé dans `results/full/summary.txt`.

### Sortie

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

---

### 🔵 Mode SINGLE — une seule expérience

Pour lancer **une seule** expérience paramétrée :

```bash
python main.py --data <chemin.csv> --k <nombre> [options]
```

#### Exemple minimal

```bash
python main.py --data data/synthetic/test_3clusters.csv --k 3
```

#### Exemple complet

```bash
python main.py \
  --data data/synthetic/test_3clusters.csv \
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

### Sortie

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

## 📖 Arguments de la ligne de commande

### Paramètres principaux

| Argument     | Type  | Défaut        | Description                                            |
| ------------ | ----- | ------------- | ------------------------------------------------------ |
| `--data`     | str   | —             | Chemin du fichier CSV (obligatoire en mode single)     |
| `--k`        | int   | —             | Nombre de classes (obligatoire en mode single)         |
| `--max-iter` | int   | 100           | Nombre maximal d'itérations                            |
| `--tol`      | float | 1e-4          | Seuil de convergence                                   |
| `--seed`     | int   | 42            | Graine aléatoire                                       |
| `--init`     | str   | `kmpp`        | Méthode d'initialisation : `random`, `kmpp`, `uniform` |
| `--output`   | str   | `results/...` | Dossier de sortie                                      |
| `--verbose`  | flag  | False         | Afficher la progression                                |

### Type de nuée

| Argument      | Valeurs possibles                                                        | Défaut  |                                                  |
| ------------- | ------------------------------------------------------------------------ | ------- | ------------------------------------------------ |
| `--nuee-type` | `point`, `ensemble_points`, `distribution`, `axe_factoriel`, `structure` | `point` |                                                  |
| `--m`         | int                                                                      | 3       | Nombre de représentants (pour `ensemble_points`) |

### Sous-type pour la nuée `point`

| Argument        | Valeurs possibles                                             | Défaut               |                            |
| --------------- | ------------------------------------------------------------- | -------------------- | -------------------------- |
| `--prototype`   | `centroide`, `mediane`, `medoide`                             | `centroide`          |                            |
| `--distance`    | `euclidienne`, `euclidienne_carree`, `manhattan`, `minkowski` | `euclidienne_carree` |                            |
| `--minkowski-p` | float                                                         | 3.0                  | Paramètre p pour Minkowski |

### Prétraitement

| Argument        | Valeurs possibles                | Défaut |                         |
| --------------- | -------------------------------- | ------ | ----------------------- |
| `--missing`     | `mean`, `median`, `zero`, `drop` | `mean` |                         |
| `--standardize` | flag                             | False  | Standardisation z-score |
| `--normalize`   | flag                             | False  | Normalisation Min-Max   |

### Aide

```bash
python main.py --help
```

---

## 🧪 Tests

Lancer tous les tests unitaires :

```bash
# Depuis la racine du projet
python -m tests.test_preprocessing
python -m tests.test_distances
python -m tests.test_initialization
python -m tests.test_assignment
python -m tests.test_update
python -m tests.test_inertia
python -m tests.test_convergence
python -m tests.test_algorithm
```

Ou avec `pytest` :

```bash
pytest tests/ -v
```

---

## 🛠️ Scripts auxiliaires

### Générer un CSV synthétique

```bash
python -c "
from src.nuees_dynamiques.preprocessing import generate_synthetic_data
import numpy as np

X, y = generate_synthetic_data(
    n_per_cluster=50,
    k=3,
    d=2,
    seed=42
)

np.savetxt(
    'data/synthetic/test_3clusters.csv',
    X,
    delimiter=',',
    header='x,y',
    comments=''
)

print('CSV généré :', X.shape)
"
```

### Évaluation comparative

```bash
python -m scripts.evaluate
```

Produit un tableau comparatif et des figures dans `results/evaluation/`.

### Visualisation des 5 types de nuées

```bash
python -m scripts.visualize
```

Produit les figures dans `results/figures/`.

---

## 🧠 Cadre théorique

### Principe des nuées dynamiques

L'algorithme des nuées dynamiques est un cadre général de classification automatique non supervisée introduit par Diday. Il alterne deux étapes :

1. **Affectation** : chaque point est affecté à la nuée la plus proche.
2. **Mise à jour** : chaque nuée est recalculée à partir des points qui lui sont affectés.

Le processus est répété jusqu'à convergence d'un critère (inertie intra-classe).

### Formule de l'inertie

$$
W = \sum_{j=1}^{k} \sum_{x \in C_j} d^2(x, \text{nuée}_j)
$$

### Types de nuées supportés

| Type                | Représentation         | Distance adaptée       |
| ------------------- | ---------------------- | ---------------------- |
| `point` (centroïde) | Vecteur moyen          | Euclidienne²           |
| `point` (médiane)   | Médiane par coordonnée | Manhattan              |
| `point` (médoïde)   | Point réel             | Toute distance         |
| `ensemble_points`   | m points               | Euclidienne            |
| `distribution`      | (moyenne, covariance)  | Mahalanobis            |
| `axe_factoriel`     | (point, direction)     | Distance orthogonale   |
| `structure`         | (p1, p2)               | Distance point-segment |

---

## 📊 Résultats attendus (exemple)

Sur un jeu synthétique 2D à 3 clusters gaussiens bien séparés :

| Variante          | Inertie | Silhouette | Davies-Bouldin |
| ----------------- | ------- | ---------- | -------------- |
| point / centroide | 145.93  | **0.668**  | **0.467**      |
| point / medoide   | 148.87  | 0.668      | 0.495          |
| point / mediane   | 181.28  | 0.644      | 0.602          |
| ensemble_points   | 152.10  | 0.651      | 0.512          |
| distribution      | 145.93  | 0.668      | 0.467          |
| axe_factoriel     | 155.44  | 0.638      | 0.541          |
| structure         | 162.10  | 0.612      | 0.583          |

**Interprétation** : la variante `point / centroide` (→ k-means) est la meilleure sur des clusters compacts et isotropes, ce qui est attendu théoriquement.

---

## ⚠️ Limites et perspectives

### Limites actuelles

* L'implémentation est **naïve** (boucles explicites, pas de vectorisation avancée).
* Complexité temporelle en \(O(T \cdot n \cdot k \cdot d)\) par itération.
* Pas de gestion du **big data** ni de calcul distribué.
* Les types `axe_factoriel` et `structure` sont des implémentations simplifiées.
* La variante `distribution` n'utilise pas l'algorithme EM complet.

### Perspectives

* **Optimisations** : vectorisation NumPy, kd-tree, Elkan, mini-batch.
* **Extensions** : mélanges gaussiens avec EM, nuées floues (fuzzy), nuées symboliques.
* **Interface** : interface graphique (Tkinter, Streamlit).
* **Déploiement** : API REST, package PyPI.

---

## 📚 Références

* Diday, E. (1971). *Une nouvelle méthode en classification automatique et reconnaissance des formes : la méthode des nuées dynamiques.* Revue de Statistique Appliquée.
* MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations.*
* Kaufman, L., & Rousseeuw, P. J. (1990). *Finding Groups in Data: An Introduction to Cluster Analysis.*
* Rousseeuw, P. J. (1987). *Silhouettes: a graphical aid to the interpretation and validation of cluster analysis.*
* Davies, D. L., & Bouldin, D. W. (1979). *A Cluster Separation Measure.*

---

## 👤 Auteur

**Ismaël Lebange** — TP Science des Données

---

## 📝 Licence

Projet académique — usage pédagogique uniquement.
