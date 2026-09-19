import csv
import numpy as np
import os



def load_csv(path, delimiter=",", skip_header=True):
    """
    Charge un CSV numérique.
    Tolère : lignes vides, valeurs manquantes ('' ou 'NaN'), BOM UTF-8.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable : {path}")

    data = []
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if skip_header and i == 0:
                continue
            # Ignorer les lignes vides
            if len(row) == 0 or all(c.strip() == "" for c in row):
                continue
            parsed = []
            for value in row:
                v = value.strip()
                if v == "" or v.lower() == "nan":
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(v))
                    except ValueError:
                        raise ValueError(
                            f"Ligne {i+1} : valeur non numérique '{value}'"
                        )
            data.append(parsed)

    if len(data) == 0:
        raise ValueError("Le CSV est vide (aucune ligne de données).")

    # Vérifier la cohérence des colonnes (à partir des données, pas du header)
    n_cols = len(data[0])
    for i, row in enumerate(data):
        if len(row) != n_cols:
            raise ValueError(
                f"Ligne {i+1} : {len(row)} colonnes au lieu de {n_cols}. "
                f"Vérifie ton CSV (séparateur, header, lignes vides)."
            )

    return np.array(data, dtype=float)
def save_results(labels,centers,inertia,output_dir):
    os.makedirs(output_dir,exist_ok=True)

    np.save(os.path.join(output_dir,"labels.npy"),labels)
    np.save(os.path.join(output_dir,"centers.npy"),centers)

    with open(os.path.join(output_dir,"inertia.txt"),"w") as f:
        f.write(f"{inertia:.6f}\n")
    print(f"Resultats sauvegardés dans {output_dir}")

def save_history(history,output_dir):
    os.makedirs(output_dir,exist_ok=True)
    path=os.path.join(output_dir,"history.csv")
    with open(path,"W",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["iteration","inertia"])
        for i,val in enumerate(history,start=1):
            writer.writerow([i,f"{val:.6f}"])
    print(f"Historique sauvegardé dans {path}")

    