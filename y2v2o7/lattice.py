"""Pyrochlore lattice geometry and DM vectors for Y2V2O7."""

import numpy as np

a_cub = 9.89      # Å
J_FM  = 8.22      # meV
S_val = 0.5

# Primitive FCC lattice vectors
a1 = (a_cub / 2) * np.array([0, 1, 1], float)
a2 = (a_cub / 2) * np.array([1, 0, 1], float)
a3 = (a_cub / 2) * np.array([1, 1, 0], float)

r_sub = np.array([[0, 0, 0], [1/4, 1/4, 0],
                  [1/4, 0, 1/4], [0, 1/4, 1/4]]) * a_cub
n_sub = 4

d_NN  = a_cub / (2 * np.sqrt(2))
d_NNN = a_cub * np.sqrt(6) / 4   # ≈ 6.056 Å


def find_bonds(d_target, n_max, tol):
    """All (i, j, cell offset, bond vector) with |r_j - r_i| ≈ d_target."""
    bonds = []
    for i in range(n_sub):
        for j in range(n_sub):
            for n1 in range(-n_max, n_max + 1):
                for n2 in range(-n_max, n_max + 1):
                    for n3 in range(-n_max, n_max + 1):
                        if i == j and n1 == n2 == n3 == 0:
                            continue
                        rj = r_sub[j] + n1*a1 + n2*a2 + n3*a3
                        if abs(np.linalg.norm(rj - r_sub[i]) - d_target) < tol:
                            bonds.append((i, j, np.array([n1, n2, n3]),
                                          rj - r_sub[i]))
    return bonds


bonds     = find_bonds(d_NN, 1, 0.1)
nnn_bonds = find_bonds(d_NNN, 2, 0.08)


def _nearest_tet_centre(r_mid):
    c_up   = (a_cub / 8) * np.array([1, 1, 1], float)
    c_dn   = (a_cub / 8) * np.array([3, 3, 3], float)
    shifts = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1],
                       [1,1,0],[1,0,1],[0,1,1],[1,1,1],
                       [-1,0,0],[0,-1,0],[0,0,-1],
                       [-1,-1,0],[-1,0,-1],[0,-1,-1]], float)
    best, bd = None, np.inf
    for s in shifts:
        R = (a_cub / 2) * s
        for c in (c_up, c_dn):
            d = np.linalg.norm(R + c - r_mid)
            if d < bd:
                bd, best = d, R + c.copy()
    return best


# Moriya-rule DM unit vectors, one per NN bond
dm_unit = []
for (i, j, dl, dv) in bonds:
    d_hat = dv / np.linalg.norm(dv)
    r_mid = r_sub[i] + 0.5 * dv
    c = _nearest_tet_centre(r_mid)
    a_vec = c - r_mid
    a_nrm = np.linalg.norm(a_vec)
    if a_nrm < 1e-10:
        dm_unit.append(np.zeros(3))
        continue
    dm = np.cross(a_vec / a_nrm, d_hat)
    dm_nrm = np.linalg.norm(dm)
    dm_unit.append(dm / dm_nrm if dm_nrm > 1e-10 else np.zeros(3))

n_hat = np.array([1, 1, 1]) / np.sqrt(3)   # magnetisation axis

# Reciprocal lattice
V  = np.dot(a1, np.cross(a2, a3))
b1 = 2*np.pi * np.cross(a2, a3) / V
b2 = 2*np.pi * np.cross(a3, a1) / V
b3 = 2*np.pi * np.cross(a1, a2) / V

G_pt = np.array([0,   0,   0  ])
X_pt = np.array([1/2, 0,   1/2])
W_pt = np.array([1/2, 1/4, 3/4])
L_pt = np.array([1/2, 1/2, 1/2])


def frac2cart(hkl):
    return hkl[0]*b1 + hkl[1]*b2 + hkl[2]*b3
