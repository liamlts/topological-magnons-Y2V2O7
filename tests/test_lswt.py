import numpy as np
import pytest
from scipy.linalg import eigh

from y2v2o7.lattice import J_FM, S_val, L_pt, X_pt, W_pt, frac2cart
from y2v2o7.lswt import (build_Hq, build_Hq_J2, build_H_slab,
                         berry_curvature_vec, berry_curvature_all,
                         chern_number_sphere, bands_along, find_crossings,
                         fit_cone_velocities, kp_projection,
                         representative_crossing, c2_bose)

D_Y2V2O7 = 0.32 * J_FM
JS = J_FM * S_val
Q_GENERIC = np.array([0.1, -0.2, 0.3])


def bands(q, D):
    return np.sort(eigh(build_Hq(q, D), eigvals_only=True))


@pytest.fixture(scope="module")
def weyl_point():
    """Weyl crossing on Γ→L at D/J = 0.32."""
    qL = frac2cart(L_pt)
    t = np.linspace(0, 1, 2000)
    om = bands_along(qL, D_Y2V2O7, len(t))[None]
    crossings = find_crossings(om, t, [0.32], qL)
    fit_cone_velocities(crossings, om, t, np.linalg.norm(qL))
    return representative_crossing(crossings, 0.32)


def test_hamiltonian_hermitian():
    for D in (0.0, D_Y2V2O7):
        H = build_Hq(Q_GENERIC, D)
        assert np.allclose(H, H.conj().T)
    H = build_Hq_J2(Q_GENERIC, D_Y2V2O7, 0.1 * J_FM)
    assert np.allclose(H, H.conj().T)


@pytest.mark.parametrize("q", [np.zeros(3), frac2cart(X_pt), frac2cart(W_pt),
                               0.37 * frac2cart(L_pt), Q_GENERIC])
def test_heisenberg_limit_flat_bands(q):
    # Pure Heisenberg pyrochlore FM: two dispersionless bands at 8JS.
    assert np.allclose(bands(q, 0.0)[2:], 8 * JS)


def test_goldstone_mode():
    for D in (0.0, D_Y2V2O7):
        assert abs(bands(np.zeros(3), D)[0]) < 1e-10


def test_ferromagnet_is_stable():
    rng = np.random.default_rng(0)
    for q in rng.uniform(-1, 1, (200, 3)):
        assert bands(q, D_Y2V2O7)[0] > 0


def test_dm_time_reversal():
    # Flipping D is equivalent to q → −q with complex conjugation.
    assert np.allclose(build_Hq(Q_GENERIC, -D_Y2V2O7),
                       build_Hq(-Q_GENERIC, D_Y2V2O7).conj())
    for n in range(4):
        assert np.allclose(berry_curvature_vec(Q_GENERIC, n, -D_Y2V2O7),
                           -berry_curvature_vec(-Q_GENERIC, n, D_Y2V2O7))


def test_berry_curvature_sums_to_zero():
    _, Om = berry_curvature_all(Q_GENERIC, D_Y2V2O7)
    assert np.allclose(Om.sum(axis=0), 0.0, atol=1e-10)


def test_time_reversal_without_dm():
    # build_Hq puts Bloch phases on cell offsets only, so Ω(q) is not zero
    # pointwise at D = 0, but time reversal still forces Ω(−q) = −Ω(q).
    _, Om_p = berry_curvature_all(Q_GENERIC, 0.0)
    _, Om_m = berry_curvature_all(-Q_GENERIC, 0.0)
    assert np.allclose(Om_m, -Om_p, atol=1e-8)


def test_weyl_point_energy(weyl_point):
    assert weyl_point is not None
    assert weyl_point['bands'] == (1, 2)
    assert weyl_point['omega'] == pytest.approx(29.16, abs=0.05)
    assert 0 < weyl_point['t'] < 1


def test_weyl_chirality_and_chern(weyl_point):
    q, (lo, hi) = weyl_point['q'], weyl_point['bands']
    chi, _ = kp_projection(q, (lo, hi), D_Y2V2O7)
    assert chi == +1
    kw = dict(N_theta=12, N_phi=24)
    C_lo = chern_number_sphere(q, 0.055, lo, D_Y2V2O7, **kw)
    C_hi = chern_number_sphere(q, 0.055, hi, D_Y2V2O7, **kw)
    C_flip = chern_number_sphere(q, 0.055, lo, -D_Y2V2O7, **kw)
    assert C_lo == pytest.approx(+1, abs=0.02)
    assert C_hi == pytest.approx(-1, abs=0.02)
    assert C_flip == pytest.approx(-1, abs=0.02)


def test_slab_is_hermitian_and_within_bulk_band():
    Hs = build_H_slab(np.zeros(3), D_Y2V2O7, 6)
    assert Hs.shape == (24, 24)
    assert np.allclose(Hs, Hs.conj().T)
    ev = eigh(Hs, eigvals_only=True)
    bulk_max = max(bands(q, D_Y2V2O7)[-1]
                   for q in np.random.default_rng(1).uniform(-1, 1, (300, 3)))
    assert ev.max() <= bulk_max + 1e-6


def test_c2_bose_limits():
    assert c2_bose(0.0)[0] == 0.0
    assert c2_bose(1e6)[0] == pytest.approx(np.pi**2 / 3, rel=1e-5)
    rho = np.logspace(-6, 4, 50)
    assert np.all(np.diff(c2_bose(rho)) > 0)
