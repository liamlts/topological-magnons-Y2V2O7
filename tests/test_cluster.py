import numpy as np
import pytest

from y2v2o7.cluster import cf_trigonal_d, gauss_convolve

DELTA = 0.030


def test_trigonal_field_structure():
    H = cf_trigonal_d(DELTA)
    assert H.shape == (10, 10)
    assert np.allclose(H, H.conj().T)
    assert abs(np.trace(H)) < 1e-12
    # Spin-diagonal with identical ↑↑ and ↓↓ blocks.
    assert np.allclose(H[0::2, 1::2], 0)
    assert np.allclose(H[0::2, 0::2], H[1::2, 1::2])


def test_trigonal_field_spectrum():
    # −(Δ/3)(m_n² − 2) for m_n = 0, ±1, ±2, each spin-doubled.
    ev = np.linalg.eigvalsh(cf_trigonal_d(DELTA)) / DELTA
    assert np.allclose(ev, [-2/3]*4 + [1/3]*4 + [2/3]*2)
    assert np.allclose(cf_trigonal_d(2 * DELTA), 2 * cf_trigonal_d(DELTA))


def test_t2g_splitting_with_cubic_field():
    edrixs = pytest.importorskip("edrixs")
    ev = np.linalg.eigvalsh(edrixs.cf_cubic_d(1.9) + cf_trigonal_d(DELTA))
    t2g = ev[:6]
    # eg' doublet (4 spin-orbitals) below the a1g singlet by ≈ Δ.
    assert np.allclose(t2g[:4], t2g[0])
    assert np.allclose(t2g[4:], t2g[4])
    assert t2g[4] - t2g[0] == pytest.approx(DELTA, rel=0.05)


def test_gauss_convolve_area_and_width():
    x = np.linspace(-1, 1, 2001)
    dx = x[1] - x[0]
    s1, s2 = 0.01, 0.02
    y = np.exp(-0.5 * (x / s1) ** 2)
    out = gauss_convolve(y, s2, dx)
    assert out.sum() == pytest.approx(y.sum(), rel=1e-10)
    # Gaussian widths add in quadrature.
    var = np.sum(out * x**2) / out.sum()
    assert np.sqrt(var) == pytest.approx(np.hypot(s1, s2), rel=1e-3)
