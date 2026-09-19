# tests/test_update.py

import numpy as np
from src.nuees_dynamiques.update import (
    update_prototypes, prototypes_have_moved
)


def test_centroide():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = np.array([0, 0, 1])
    old_protos = np.array([[1.0, 1.0], [9.0, 9.0]])

    new_protos, empties = update_prototypes(
        X, labels, k=2, old_prototypes=old_protos,
        prototype_type="centroide"
    )
    print("Centroïde :\n", new_protos)
    print("Vides :", empties)
    assert np.allclose(new_protos, [[1.0, 1.0], [10.0, 10.0]])
    assert empties == []
    print("✅ test_centroide OK")


def test_mediane():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = np.array([0, 0, 1])
    old_protos = np.array([[1.0, 1.0], [9.0, 9.0]])

    new_protos, empties = update_prototypes(
        X, labels, k=2, old_prototypes=old_protos,
        prototype_type="mediane"
    )
    print("Médiane :\n", new_protos)
    # Cluster 0 : [0,0] et [2,2] → médiane = [1,1]
    # Cluster 1 : [10,10] → médiane = [10,10]
    assert np.allclose(new_protos, [[1.0, 1.0], [10.0, 10.0]])
    print("✅ test_mediane OK")


def test_medoide():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = np.array([0, 0, 1])
    old_protos = np.array([[1.0, 1.0], [9.0, 9.0]])

    new_protos, empties = update_prototypes(
        X, labels, k=2, old_prototypes=old_protos,
        prototype_type="medoide"
    )
    print("Médoïde :\n", new_protos)
    # Cluster 0 : médoïde parmi [0,0] et [2,2]
    #   coût de [0,0] = d²([0,0],[0,0]) + d²([0,0],[2,2]) = 0 + 8 = 8
    #   coût de [2,2] = d²([2,2],[0,0]) + d²([2,2],[2,2]) = 8 + 0 = 8
    #   égalité → on garde le premier trouvé = [0,0]
    # Cluster 1 : [10,10]
    assert new_protos[1].tolist() == [10.0, 10.0]
    assert new_protos[0].tolist() in ([0.0, 0.0], [2.0, 2.0])
    print("✅ test_medoide OK")


def test_cluster_vide():
    X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0]])
    labels = np.array([0, 0, 0])  # cluster 1 vide
    old_protos = np.array([[1.0, 1.0], [9.0, 9.0]])

    new_protos, empties = update_prototypes(
        X, labels, k=2, old_prototypes=old_protos,
        prototype_type="centroide"
    )
    print("Cluster vide :\n", new_protos)
    print("Vides détectés :", empties)
    assert 1 in empties
    # Le point le plus éloigné de [9,9] est [0,0]
    assert new_protos[1].tolist() == [0.0, 0.0]
    print("✅ test_cluster_vide OK")


def test_prototypes_have_moved():
    old_p = np.array([[0.0, 0.0], [1.0, 1.0]])
    new_p = np.array([[0.5, 0.5], [1.5, 1.5]])
    moved, shift = prototypes_have_moved(old_p, new_p, tol=1e-4)
    print("Bougé ?", moved, "| shift =", shift)
    assert moved is True
    assert shift > 0.0
    print("✅ test_prototypes_have_moved OK")


if __name__ == "__main__":
    test_centroide()
    test_mediane()
    test_medoide()
    test_cluster_vide()
    test_prototypes_have_moved()
    print("\n🎉 Tous les tests de update.py sont OK")