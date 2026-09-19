import numpy as np

def set_seed(seed):
    np.random.seed(seed)

def check_array(X):
    if not isinstance(X,np.ndarray):
        raise TypeError("X doit etre un tableau numpy")
    if X.ndim != 2:
        raise ValueError("X doit etre un tableay 2D (n_points,n_dimensions)")
    if not np.issubdtype(X.dtype,np.number):
        raise TypeError("X doit contenir uniquement des nombres")
    return True

def print_progress(iteration,inertia):
    print(f"Iteration{iteration:03d} | Inertie = {inertia:.6f}")

def validate_labels(labels,n,k):
    if not isinstance(labels,np.ndarray):
        raise TypeError("labels doit etre un tableau numpy")
    if labels.shape != (n,):
        raise ValueError(f"labels doit avoir la forme ({n},) mais a la forme {labels.shape}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("labels doit contenir uniquement des entiers")
    if np.any((labels < 0) | (labels >= k)):
        raise ValueError(f"labels doit contenir des entiers entre 0 et {k-1}")
    return True

def validate_centers(centers,k,d):
    if not isinstance(centers,np.ndarray):
        raise TypeError("centers doit etre un tableau numpy")
    if centers.shape != (k,d):
        raise ValueError(f"centers doit avoir la forme ({k},{d}) mais a la forme {centers.shape}")
    if not np.issubdtype(centers.dtype, np.number):
        raise TypeError("centers doit contenir uniquement des nombres")
    return True

def validate_X(X):
    check_array(X)
    if np.any(np.isnan(X)):
        raise ValueError("X contient des valeurs manquantes (NaN)")
    if np.any(np.isinf(X)):
        raise ValueError("X contient des valeurs infinies")
    return True

