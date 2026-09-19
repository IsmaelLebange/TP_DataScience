# src/nuees_dynamiques/convergence.py

from .update import prototypes_have_moved
from .inertia import inertia_converged


def check_convergence(old_prototypes, new_prototypes,
                      old_inertia, new_inertia,
                      iteration, max_iter,
                      tol=1e-4):
    """
    Vérifie si l'algorithme doit s'arrêter.

    Retour :
        stop : bool
        reason : str parmi {'prototypes', 'inertia', 'max_iter', 'none'}
        info : dict avec détails (max_shift, inertia_variation)
    """
    info = {"max_shift": None, "inertia_variation": None}

    if iteration >= max_iter:
        return True, "max_iter", info

    if old_prototypes is not None:
        moved, shift = prototypes_have_moved(
            old_prototypes, new_prototypes, tol
        )
        info["max_shift"] = shift
        if not moved:
            return True, "prototypes", info

    if old_inertia is not None:
        conv, var = inertia_converged(old_inertia, new_inertia, tol)
        info["inertia_variation"] = var
        if conv:
            return True, "inertia", info

    return False, "none", info