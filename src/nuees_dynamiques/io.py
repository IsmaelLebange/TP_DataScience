import csv
import numpy as np
import os

def load_csv(path,delimiter=",", skip_header=True):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    data= []
    with open(path,"r",newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if skip_header and i == 0:
                continue
            if len(row) == 0:
                continue
            parsed_row = []
            for item in row:
                item=item.strip()
                if item == "" or item.lower() == "nan":
                    parsed_row.append(np.nan)
                else:
                    try:
                        parsed_row.append(float(item))
                    except ValueError:
                        parsed_row.append(item)
            data.append(parsed_row)
    if len(data) == 0:
        raise ValueError(f"The file {path} is empty or contains only headers.")

    n_cols = len(data[0])
    for i, row in enumerate(data):
        if len(row) != n_cols:
            raise ValueError(f"Row {i+1} in the file {path} does not have the same number of columns as the first row.")
    return np.array(data,dtype=float)

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

    