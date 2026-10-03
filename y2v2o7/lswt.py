"""Linear spin-wave theory, Berry curvature and Weyl-point analysis."""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from scipy.special import spence   # spence(z) = Li₂(1-z)

from .lattice import (J_FM, S_val, a1, a2, n_sub, bonds, nnn_bonds,
                      dm_unit, n_hat, a3)

kB = 0.08617   # meV/K

PAULI = [np.array([[0,1],[1,0]], complex),
         np.array([[0,-1j],[1j,0]], complex),
         np.array([[1,0],[0,-1]], complex)]


def build_Hq(qvec, D_val):
    """LSWT Hamiltonian for pyrochlore FM with DM interaction.

    DM vectors are projected onto the [111] magnetisation axis (n_hat).
    This is exact for the q=0 ordered state; finite-q canting corrections
    enter at O(D²/J²) and are neglected here.
    """
    H = np.zeros((n_sub, n_sub), dtype=complex)
    for b, (i, j, dl, dv) in enumerate(bonds):
        Dn  = D_val * np.dot(dm_unit[b], n_hat)
        dlc = dl[0]*a1 + dl[1]*a2 + dl[2]*a3
        ph  = np.exp(1j * np.dot(qvec, dlc))
        H[i, i] += S_val * (J_FM + Dn)
        H[i, j] -= S_val * (J_FM + 1j*Dn) * ph
    return H


def build_Hq_J2(qvec, D_val, J2_val):
    """LSWT Hamiltonian with NN DM (D_val) and NNN isotropic exchange (J2_val)."""
    H = build_Hq(qvec, D_val).copy()
    for (i, j, dl_int, dv) in nnn_bonds:
        dlc = dl_int[0]*a1 + dl_int[1]*a2 + dl_int[2]*a3
        ph  = np.exp(1j * np.dot(qvec, dlc))
        H[i, i] += S_val * J2_val
        H[i, j] -= S_val * J2_val * ph
    return H


def build_H_slab(kpar3, D_val, N_layers):
    """
    Slab Hamiltonian for stacking along a₃.  kpar3 is a 3-component Cartesian
    vector with zero component along a₃ (in-plane k).  Layer index = m₃ coeff.
    """
    Hs = np.zeros((4*N_layers, 4*N_layers), dtype=complex)
    for b, (i, j, dl_int, dv) in enumerate(bonds):
        n1, n2, n3 = dl_int
        delta_l = n3           # stacking direction = a₃ coefficient
        dl_inplane = n1*a1 + n2*a2   # in-plane part (a₃ has zero contribution here)
        ph_xy = np.exp(1j * np.dot(kpar3, dl_inplane))
        Dn    = D_val * np.dot(dm_unit[b], n_hat)
        for l in range(N_layers):
            l2 = l + delta_l
            if 0 <= l2 < N_layers:
                Hs[4*l + i, 4*l + i]   += S_val * (J_FM + Dn)
                Hs[4*l + i, 4*l2 + j]  -= S_val * (J_FM + 1j*Dn) * ph_xy
    return Hs


# ── Berry curvature ──────────────────────────────────────────────────────────

def dHdk(qvec, alpha, D_val, dq=2e-5):
    """Central-difference ∂H/∂k_α."""
    ea = np.zeros(3); ea[alpha] = dq
    return (build_Hq(qvec + ea, D_val) - build_Hq(qvec - ea, D_val)) / (2*dq)


def berry_curvature_all(qvec, D_val):
    """Band energies and Berry curvature vectors Ω_n(k) for every band (Å²).

    Returns (ev, Omega) with Omega of shape (n_sub, 3).
    """
    ev, vcs = eigh(build_Hq(qvec, D_val))
    dH = [dHdk(qvec, a, D_val) for a in range(3)]
    Omega = np.zeros((n_sub, 3))
    for n in range(n_sub):
        psi_n = vcs[:, n]
        for m in range(n_sub):
            if m == n:
                continue
            dE = ev[m] - ev[n]
            if abs(dE) < 1e-10:
                continue
            psi_m = vcs[:, m]
            for ci, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
                mna = psi_n.conj() @ dH[a] @ psi_m
                mnb = psi_m.conj() @ dH[b] @ psi_n
                Omega[n, ci] += -2.0 * np.imag(mna * mnb) / dE**2
    return ev, Omega


def berry_curvature_vec(qvec, band_idx, D_val):
    """Berry curvature vector Ω_n(k) via Kubo formula (units: Å²)."""
    return berry_curvature_all(qvec, D_val)[1][band_idx]


def chern_number_sphere(q_W, r_sphere, band_idx, D_val, N_theta=20, N_phi=40):
    """Chern number by integrating Berry curvature over a sphere of radius r_sphere."""
    th_e = np.linspace(0, np.pi, N_theta + 1)
    phi  = np.linspace(0, 2*np.pi, N_phi, endpoint=False)
    dt   = np.pi / N_theta
    dp   = 2*np.pi / N_phi
    C    = 0.0
    for it in range(N_theta):
        tm = 0.5 * (th_e[it] + th_e[it + 1])
        st = np.sin(tm)
        for p in phi:
            n_h = np.array([st * np.cos(p), st * np.sin(p), np.cos(tm)])
            k = q_W + r_sphere * n_h
            Om = berry_curvature_vec(k, band_idx, D_val)
            C += np.dot(Om, n_h) * r_sphere**2 * st * dt * dp
    return C / (2 * np.pi)


# ── Weyl crossings along a line ──────────────────────────────────────────────

def bands_along(q_end, D_val, n_pts):
    """Sorted band energies on the straight line Γ → q_end, shape (n_sub, n_pts)."""
    t  = np.linspace(0, 1, n_pts)
    qs = np.outer(t, q_end)
    om = np.zeros((n_sub, n_pts))
    for iq in range(n_pts):
        om[:, iq] = np.sort(np.real(eigh(build_Hq(qs[iq], D_val),
                                         eigvals_only=True)))
    return om


def find_crossings(om, t, dj_vals, q_end, thr=0.05):
    """Local gap minima below thr (meV) between adjacent bands.

    om has shape (len(dj_vals), n_sub, len(t)); crossings at D/J = 0 are skipped.
    """
    crossings = []
    for idj, dj in enumerate(dj_vals):
        if dj < 0.01:
            continue
        for b in range(n_sub - 1):
            gap = om[idj, b+1, :] - om[idj, b, :]
            for iq in range(1, len(t) - 1):
                if (abs(gap[iq]) < abs(gap[iq-1]) and
                        abs(gap[iq]) < abs(gap[iq+1]) and
                        abs(gap[iq]) < thr):
                    tc = t[iq]
                    oc = 0.5*(om[idj, b, iq] + om[idj, b+1, iq])
                    dup = any(abs(p['t'] - tc) < 0.01
                              and p['dj_idx'] == idj
                              and p['bands'] == (b, b+1)
                              for p in crossings)
                    if not dup:
                        crossings.append(dict(dj=dj, dj_idx=idj, bands=(b, b+1),
                                              t=tc, q=tc*q_end, omega=oc))
    return crossings


def _cone_hi(dq, w0, v): return w0 + v*np.abs(dq)
def _cone_lo(dq, w0, v): return w0 - v*np.abs(dq)


def fit_cone_velocities(crossings, om, t, q_len, win=0.08):
    """Fit |dq| cones to both bands at each crossing; sets 'vW', 'vhi', 'vlo'."""
    for wc in crossings:
        idj = wc['dj_idx']
        b0, b1_ = wc['bands']
        mask = np.abs(t - wc['t']) < win
        if mask.sum() < 20:
            wc['vW'] = np.nan; continue
        dq = (t[mask] - wc['t']) * q_len
        try:
            ph, _ = curve_fit(_cone_hi, dq, om[idj, b1_, mask],
                              p0=[wc['omega'], 50.], maxfev=5000)
            vhi = abs(ph[1])
        except Exception:
            vhi = np.nan
        try:
            pl, _ = curve_fit(_cone_lo, dq, om[idj, b0, mask],
                              p0=[wc['omega'], 50.], maxfev=5000)
            vlo = abs(pl[1])
        except Exception:
            vlo = np.nan
        wc['vW']  = 0.5*(vhi + vlo) if not (np.isnan(vhi) or np.isnan(vlo)) else np.nan
        wc['vhi'] = vhi
        wc['vlo'] = vlo


def kp_projection(q, bands, D_val, dq=1e-5):
    """Löwdin-project ∂H/∂k onto a band pair.

    Returns (chi, V_ax): the Weyl chirality sign(det v) and the three
    projected 2×2 velocity matrices.
    """
    b0, b1_ = bands
    _, vcs = eigh(build_Hq(q, D_val))
    P = vcs[:, [b0, b1_]]
    V_ax = []
    for e in [np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])]:
        dH = (build_Hq(q + dq*e, D_val) -
              build_Hq(q - dq*e, D_val)) / (2*dq)
        V_ax.append(P.conj().T @ dH @ P)
    vt = np.array([[np.real(np.trace(p @ Va)) / 2
                    for p in PAULI] for Va in V_ax])
    return int(np.sign(np.linalg.det(vt))), V_ax


def representative_crossing(crossings, target_dj=0.32):
    """Crossing with a successful cone fit whose D/J is closest to target_dj."""
    repr_wc = None
    for wc in crossings:
        if not np.isnan(wc.get('vW', np.nan)):
            if repr_wc is None or abs(wc['dj'] - target_dj) < abs(repr_wc['dj'] - target_dj):
                repr_wc = wc
    return repr_wc


# ── Thermal factors ──────────────────────────────────────────────────────────

def bose(E, T):
    if T < 0.1:
        return np.ones_like(E)
    x = E / (kB * T)
    return np.where(x > 500, 1, np.where(x < 1e-10, 1/np.maximum(x, 1e-30),
                                          1/(1 - np.exp(-x))))


def c2_bose(rho):
    """
    c₂(ρ) thermal weight for bosons (Matsumoto-Murakami 2011):
      c₂(ρ) = (1+ρ)(ln((1+ρ)/ρ))² − (ln ρ)² − 2 Li₂(−ρ)
    Li₂(−ρ) = spence(1+ρ)  [scipy convention: spence(z) = Li₂(1−z)]
    """
    rho = np.atleast_1d(np.asarray(rho, float))
    out = np.zeros_like(rho)
    ok  = rho > 1e-8
    r   = rho[ok]
    out[ok] = ((1+r) * np.log((1+r)/r)**2
               - np.log(r)**2
               - 2.0 * spence(1.0 + r))
    return out


def ff2_V4(Qmag):
    """Squared V⁴⁺ magnetic form factor (dipole approximation)."""
    s = Qmag / (4*np.pi)
    s2 = s**2
    return (0.0635*np.exp(-12.6861*s2) + 0.3033*np.exp(-5.4669*s2)
            + 0.6507*np.exp(-2.1724*s2) - 0.0176)**2
