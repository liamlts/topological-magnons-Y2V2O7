import numpy as np

from y2v2o7.lattice import (a1, a2, a3, b1, b2, b3, bonds, nnn_bonds, dm_unit,
                            n_hat, d_NN, d_NNN, n_sub)


def test_bond_counts():
    # Pyrochlore: 6 nearest and 12 next-nearest neighbours per site.
    assert len(bonds) == 6 * n_sub
    assert len(nnn_bonds) == 12 * n_sub


def test_bond_lengths():
    assert np.allclose([np.linalg.norm(b[3]) for b in bonds], d_NN)
    assert np.allclose([np.linalg.norm(b[3]) for b in nnn_bonds], d_NNN)


def test_reciprocal_lattice():
    A = np.array([a1, a2, a3])
    B = np.array([b1, b2, b3])
    assert np.allclose(A @ B.T, 2 * np.pi * np.eye(3))


def test_dm_vectors_moriya_rules():
    dm = np.array(dm_unit)
    dv = np.array([b[3] for b in bonds])
    assert np.allclose(np.linalg.norm(dm, axis=1), 1.0)
    # D_ij ⊥ r_ij, and D_ji = −D_ij for each reversed bond.
    assert np.allclose(np.sum(dm * dv, axis=1), 0.0)
    for b, (i, j, dl, v) in enumerate(bonds):
        rev = [c for c, (i2, j2, _, v2) in enumerate(bonds)
               if i2 == j and j2 == i and np.allclose(v2, -v)]
        assert len(rev) == 1
        assert np.allclose(dm[rev[0]], -dm[b])
    # Projection on [111] takes only the values 0, ±√(2/3).
    proj = np.unique(np.round(np.abs(dm @ n_hat), 6))
    assert np.allclose(proj, [0.0, np.sqrt(2 / 3)])
