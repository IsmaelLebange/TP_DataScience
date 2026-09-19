import numpy as np
from src.nuees_dynamiques.convergence import check_convergence

# Cas 1 : centres identiques → stop 'centers'
c1 = np.array([[0.0, 0.0], [1.0, 1.0]])
c2 = np.array([[0.0, 0.0], [1.0, 1.0]])
stop, reason, info = check_convergence(c1, c2, 10.0, 10.0, iteration=1, max_iter=100)
print("Cas 1 :", stop, reason, info)
# Attendu : True, 'centers'

# Cas 2 : centres qui bougent mais inertie stable → stop 'inertia'
c1 = np.array([[0.0, 0.0], [1.0, 1.0]])
c2 = np.array([[0.5, 0.5], [1.5, 1.5]])
stop, reason, info = check_convergence(c1, c2, 10.0, 10.000001, iteration=1, max_iter=100)
print("Cas 2 :", stop, reason, info)
# Attendu : True, 'inertia'

# Cas 3 : rien ne converge → continue
c1 = np.array([[0.0, 0.0], [1.0, 1.0]])
c2 = np.array([[5.0, 5.0], [6.0, 6.0]])
stop, reason, info = check_convergence(c1, c2, 10.0, 5.0, iteration=1, max_iter=100)
print("Cas 3 :", stop, reason, info)
# Attendu : False, 'none'

# Cas 4 : max_iter atteint
stop, reason, info = check_convergence(c1, c2, 10.0, 5.0, iteration=100, max_iter=100)
print("Cas 4 :", stop, reason, info)
# Attendu : True, 'max_iter'

# Cas 5 : première itération (pas de old_centers, pas de old_inertia)
stop, reason, info = check_convergence(None, c2, None, 5.0, iteration=1, max_iter=100)
print("Cas 5 :", stop, reason, info)
# Attendu : False, 'none'