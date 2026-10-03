"""Single-ion helpers shared by the EDRIXS cluster scripts (no edrixs import)."""

import numpy as np


def cf_trigonal_d(delta_trig):
    """
    Trigonal D3d crystal field for d-orbitals with C3 axis along [111].

    H = −(Δ/3) × [(L·n̂)² − L(L+1)/3],  n̂ = [1,1,1]/√3

    Within t2g (once a cubic 10Dq separates t2g from eg) this puts a1g above
    eg' by Δ for Δ > 0.
    EDRIXS spin-orbital basis: (m=-2,↑), (m=-2,↓), …, (m=+2,↑), (m=+2,↓)
    i.e. m increases with index, spin interleaved as (m,↑) then (m,↓).
    """
    l = 2
    m_vals = np.arange(-l, l + 1, dtype=float)   # [-2,-1,0,1,2]
    n = len(m_vals)

    # L operators in the 5-dim orbital space (m from -l to +l)
    Lp = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        m = m_vals[i]
        Lp[i + 1, i] = np.sqrt((l - m) * (l + m + 1))   # L+|m⟩ → |m+1⟩
    Lm = Lp.conj().T
    Lx = (Lp + Lm) / 2.0
    Ly = (Lp - Lm) / (2j)
    Lz = np.diag(m_vals.astype(complex))

    LN = (Lx + Ly + Lz) / np.sqrt(3.0)
    # Within the t2g subspace the full l=2 operator gives 3× the t2g splitting,
    # so divide by 3.  Negative sign: positive delta_trig → a1g above eg' (compressed
    # pyrochlore octahedron, consistent with Y2V2O7 DFT/optical data).
    H5 = -(delta_trig / 3.0) * (LN @ LN - l * (l + 1) / 3.0 * np.eye(n, dtype=complex))

    # Expand to 10×10 (orbital acts on spin space as identity)
    H10 = np.zeros((10, 10), dtype=complex)
    for i in range(n):
        for j in range(n):
            H10[2 * i,     2 * j    ] = H5[i, j]   # ↑↑
            H10[2 * i + 1, 2 * j + 1] = H5[i, j]   # ↓↓
    return H10


def gauss_convolve(spec, sigma_eV, dE_eV):
    """Convolve spectrum with Gaussian of given σ (eV) and grid spacing dE."""
    hw = int(4 * sigma_eV / dE_eV) + 1
    k  = np.arange(-hw, hw + 1) * dE_eV
    kernel = np.exp(-0.5 * (k / sigma_eV) ** 2)
    kernel /= kernel.sum()
    return np.convolve(spec, kernel, mode='same')
