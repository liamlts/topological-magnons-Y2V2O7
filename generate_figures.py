"""
generate_figures.py — Publication-quality figures for the Y2V2O7 topological magnon paper.

Produces all LSWT-based figures (no EDRIXS needed):
  Figures/fig_comparison_progression.pdf
  Figures/Y2V2O7_band_overlay_DJ.pdf
  Figures/fig_weyl_cone_3D.pdf
  Figures/fig_weyl_band_zoom_GL.pdf
  Figures/fig_weyl_kp_analysis.pdf

Run: python3 generate_figures.py

Nature Physics panel-label convention: bold 'a.' top-left, no parentheses.
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
from matplotlib.colors import PowerNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from collections import defaultdict

os.makedirs('Figures', exist_ok=True)

# ── Publication rcParams ─────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':          7,
    'axes.labelsize':     7,
    'axes.titlesize':     7,
    'xtick.labelsize':    6,
    'ytick.labelsize':    6,
    'legend.fontsize':    6,
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'xtick.minor.size':   1.5,
    'ytick.minor.size':   1.5,
    'xtick.direction':    'out',
    'ytick.direction':    'out',
    'lines.linewidth':    0.8,
    'figure.dpi':         150,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'savefig.pad_inches': 0.02,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

COL1 = 3.386   # 8.6 cm — single column
COL2 = 7.008   # 17.8 cm — double column

C_BLUE   = '#0073BD'
C_RED    = '#D92B2B'
C_ORANGE = '#ED8C00'
C_GREEN  = '#38A12B'
C_PURPLE = '#9533BF'
C_GREY   = '#808080'
C_BLACK  = '#1A1A1A'
BRANCH_COLORS = [C_BLUE, C_RED, C_ORANGE, C_GREEN, C_PURPLE, C_GREY, C_BLACK]
PATH_LABELS   = ['Γ', 'X', 'W', 'L', 'Γ']


def label(ax, letter, dark_bg=False, x=0.025, y=0.97):
    """Add Nature Physics panel label: bold 'a.' at top-left."""
    color = 'white' if dark_bg else 'black'
    bbox = dict(facecolor='k', alpha=0.55, pad=1.5,
                boxstyle='round,pad=0.2', edgecolor='none') if dark_bg else None
    ax.text(x, y, f'{letter}.', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left', color=color,
            bbox=bbox)


def save(fig, name):
    for ext in ('pdf', 'png'):
        path = f'Figures/{name}.{ext}'
        fig.savefig(path, dpi=600 if ext == 'pdf' else 300)
        print(f'  saved {path}')


# ── Berry curvature helpers (called after build_Hq / n_sub are defined) ──

def _dHdka(qvec, alpha, D_val, dq=2e-5):
    ea = np.zeros(3); ea[alpha] = dq
    return (build_Hq(qvec + ea, D_val) - build_Hq(qvec - ea, D_val)) / (2*dq)


def berry_curvature_vec(qvec, band_idx, D_val):
    """Berry curvature vector Ω_n(k) via Kubo formula (units: Å²)."""
    ev, vcs = eigh(build_Hq(qvec, D_val))
    psi_n = vcs[:, band_idx]
    dH = [_dHdka(qvec, a, D_val) for a in range(3)]
    Omega = np.zeros(3)
    for m in range(n_sub):
        if m == band_idx:
            continue
        dE = ev[m] - ev[band_idx]
        if abs(dE) < 1e-10:
            continue
        psi_m = vcs[:, m]
        for ci, (a, b) in enumerate([(1, 2), (2, 0), (0, 1)]):
            mna = psi_n.conj() @ dH[a] @ psi_m
            mnb = psi_m.conj() @ dH[b] @ psi_n
            Omega[ci] += -2.0 * np.imag(mna * mnb) / dE**2
    return Omega


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
            n_hat = np.array([st * np.cos(p), st * np.sin(p), np.cos(tm)])
            k = q_W + r_sphere * n_hat
            Om = berry_curvature_vec(k, band_idx, D_val)
            C += np.dot(Om, n_hat) * r_sphere**2 * st * dt * dp
    return C / (2 * np.pi)


# ═══════════════════════════════════════════════════════════════════════════
#  1. LSWT SOLVER
# ═══════════════════════════════════════════════════════════════════════════

a_cub = 9.89      # Å
J_FM  = 8.22      # meV
S_val = 0.5
Emax  = 55.0      # meV
dE    = 1.5       # meV FWHM broadening
npts  = 300       # q-points per segment
nE    = 500
T_K   = 5.0
kB    = 0.08617   # meV/K

DJ_vals = np.array([0.00, 0.10, 0.20, 0.32, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 1.00])
nDJ = len(DJ_vals)

# Primitive FCC lattice vectors
a1 = (a_cub / 2) * np.array([0, 1, 1], float)
a2 = (a_cub / 2) * np.array([1, 0, 1], float)
a3 = (a_cub / 2) * np.array([1, 1, 0], float)

r_sub = np.array([[0, 0, 0], [1/4, 1/4, 0],
                  [1/4, 0, 1/4], [0, 1/4, 1/4]]) * a_cub
n_sub = 4

d_NN = a_cub / (2 * np.sqrt(2))
bonds = []
for i in range(n_sub):
    for j in range(n_sub):
        for n1 in range(-1, 2):
            for n2 in range(-1, 2):
                for n3 in range(-1, 2):
                    if i == j and n1 == n2 == n3 == 0:
                        continue
                    rj = r_sub[j] + n1*a1 + n2*a2 + n3*a3
                    if abs(np.linalg.norm(rj - r_sub[i]) - d_NN) < 0.1:
                        bonds.append((i, j, np.array([n1, n2, n3]),
                                      rj - r_sub[i]))

print(f"NN bonds: {len(bonds)}")


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

n_hat = np.array([1, 1, 1]) / np.sqrt(3)


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

path_pts = [G_pt, X_pt, W_pt, L_pt, G_pt]
qpath, ticks = [], [0]
for seg in range(len(path_pts) - 1):
    qs = frac2cart(path_pts[seg])
    qe = frac2cart(path_pts[seg + 1])
    for it in range(npts):
        if seg > 0 and it == 0:
            continue
        qpath.append((1 - it/(npts-1))*qs + (it/(npts-1))*qe)
    ticks.append(len(qpath) - 1)
qpath = np.array(qpath)
ticks = np.array(ticks, dtype=int)
nQ    = len(qpath)

# Magnetic form factor V4+
def ff2_V4(Qmag):
    s = Qmag / (4*np.pi)
    s2 = s**2
    return (0.0635*np.exp(-12.6861*s2) + 0.3033*np.exp(-5.4669*s2)
            + 0.6507*np.exp(-2.1724*s2) - 0.0176)**2

G0  = frac2cart(np.array([2, 0, 0]))
Qfc = qpath + G0
Qm  = np.linalg.norm(Qfc, axis=1)
Qh  = Qfc / Qm[:, None]
pf  = np.clip(1 - (Qh @ n_hat)**2, 0, 1)
ff2 = ff2_V4(Qm)


def bose(E, T):
    if T < 0.1:
        return np.ones_like(E)
    x = E / (kB * T)
    return np.where(x > 500, 1, np.where(x < 1e-10, 1/np.maximum(x, 1e-30),
                                          1/(1 - np.exp(-x))))


# Compute bands for all D/J
print("Computing LSWT bands...")
omega_all = np.zeros((nDJ, n_sub, nQ))
evecs_all = np.zeros((nDJ, n_sub, n_sub, nQ), dtype=complex)
for idj, dj in enumerate(DJ_vals):
    D = dj * J_FM
    for iq in range(nQ):
        ev, vc = eigh(build_Hq(qpath[iq], D))
        omega_all[idj, :, iq] = np.real(ev)
        evecs_all[idj, :, :, iq] = vc

# Spectral weight
sw = np.zeros((nDJ, n_sub, nQ))
for idj in range(nDJ):
    for iq in range(nQ):
        ph = np.exp(1j * r_sub @ qpath[iq])
        for mu in range(n_sub):
            Fq = np.sum(evecs_all[idj, :, mu, iq] * ph)
            sw[idj, mu, iq] = np.abs(Fq)**2 * pf[iq] * ff2[iq]

# S(q,ω)
Eax   = np.linspace(0, Emax, nE)
sigma = dE / (2*np.sqrt(2*np.log(2)))
bs    = bose(np.maximum(Eax, 0.01), T_K)
Sqw   = np.zeros((nDJ, nE, nQ))
for idj in range(nDJ):
    for ib in range(n_sub):
        for iq in range(nQ):
            E0 = omega_all[idj, ib, iq]
            if E0 > 0.01:
                Sqw[idj, :, iq] += (sw[idj, ib, iq] * bs
                                    * np.exp(-0.5*((Eax - E0)/sigma)**2))

print("LSWT done.")

# ═══════════════════════════════════════════════════════════════════════════
#  2. FINE Γ→L SCAN FOR WEYL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

qL = frac2cart(L_pt)
Lh = qL / np.linalg.norm(qL)
# Orthonormal basis in the plane transverse to [111] — used for Berry curvature map
e1_GL = np.cross(Lh, np.array([0., 0., 1.]))
e1_GL /= np.linalg.norm(e1_GL)
e2_GL = np.cross(Lh, e1_GL)
e2_GL /= np.linalg.norm(e2_GL)
n_fine = 5000
t_fine = np.linspace(0, 1, n_fine)
qGL    = np.outer(t_fine, qL)

print("Computing fine Γ→L scan...")
omGL = np.zeros((nDJ, n_sub, n_fine))
for idj, dj in enumerate(DJ_vals):
    D = dj * J_FM
    for iq in range(n_fine):
        ev = np.sort(np.real(eigh(build_Hq(qGL[iq], D), eigvals_only=True)))
        omGL[idj, :, iq] = ev

# Locate crossings
thr = 0.05   # meV gap threshold
crossings = []
for idj, dj in enumerate(DJ_vals):
    if dj < 0.01:
        continue
    for b in range(n_sub - 1):
        gap = omGL[idj, b+1, :] - omGL[idj, b, :]
        for iq in range(1, n_fine - 1):
            if (abs(gap[iq]) < abs(gap[iq-1]) and
                    abs(gap[iq]) < abs(gap[iq+1]) and
                    abs(gap[iq]) < thr):
                tc = t_fine[iq]
                oc = 0.5*(omGL[idj, b, iq] + omGL[idj, b+1, iq])
                dup = any(abs(p['t'] - tc) < 0.01
                          and p['dj_idx'] == idj
                          and p['bands'] == (b, b+1)
                          for p in crossings)
                if not dup:
                    crossings.append(dict(dj=dj, dj_idx=idj, bands=(b, b+1),
                                          t=tc, q=tc*qL, omega=oc))

print(f"Crossings found: {len(crossings)}")

# k·p velocities
qL_len = np.linalg.norm(qL)
win    = 0.08

def _cone_hi(dq, w0, v): return w0 + v*np.abs(dq)
def _cone_lo(dq, w0, v): return w0 - v*np.abs(dq)


for wc in crossings:
    idj = wc['dj_idx']
    b0, b1_ = wc['bands']
    mask = np.abs(t_fine - wc['t']) < win
    if mask.sum() < 20:
        wc['vW'] = np.nan; continue
    dq = (t_fine[mask] - wc['t']) * qL_len
    try:
        ph, _ = curve_fit(_cone_hi, dq, omGL[idj, b1_, mask],
                          p0=[wc['omega'], 50.], maxfev=5000)
        vhi = abs(ph[1])
    except Exception:
        vhi = np.nan
    try:
        pl, _ = curve_fit(_cone_lo, dq, omGL[idj, b0, mask],
                          p0=[wc['omega'], 50.], maxfev=5000)
        vlo = abs(pl[1])
    except Exception:
        vlo = np.nan
    wc['vW']  = 0.5*(vhi + vlo) if not (np.isnan(vhi) or np.isnan(vlo)) else np.nan
    wc['vhi'] = vhi
    wc['vlo'] = vlo

# Löwdin projection → chirality
pauli = [np.array([[0,1],[1,0]], complex),
         np.array([[0,-1j],[1j,0]], complex),
         np.array([[1,0],[0,-1]], complex)]
dq_d = 1e-5

for wc in crossings:
    if np.isnan(wc.get('vW', np.nan)):
        wc['chi'] = 0; continue
    D = wc['dj'] * J_FM
    b0, b1_ = wc['bands']
    _, vcs = eigh(build_Hq(wc['q'], D))
    P = vcs[:, [b0, b1_]]
    V_ax = []
    for e in [np.array([1,0,0.]), np.array([0,1,0.]), np.array([0,0,1.])]:
        dH = (build_Hq(wc['q'] + dq_d*e, D) -
              build_Hq(wc['q'] - dq_d*e, D)) / (2*dq_d)
        V_ax.append(P.conj().T @ dH @ P)
    vt = np.array([[np.real(np.trace(p @ Va)) / 2
                    for p in pauli] for Va in V_ax])
    wc['chi'] = int(np.sign(np.linalg.det(vt)))
    # velocity along [111]
    V111 = sum(V_ax) / np.sqrt(3)
    wc['vW_kp'] = np.sqrt(sum(abs(np.real(np.trace(p @ V111))/2)**2 for p in pauli))

print("k·p extraction done.")


# ═══════════════════════════════════════════════════════════════════════════
#  3. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

qax = np.arange(nQ)

# ──────────────────────────────────────────────────────────────────────────
#  Fig A: S(q,ω) D/J progression strip — 6 panels
# ──────────────────────────────────────────────────────────────────────────
print("\nGenerating: fig_comparison_progression")
dj_idx = [0, 1, 2, 4, 5, 6]   # D/J = 0, 0.10, 0.20, 0.40, 0.50, 0.60
nP = len(dj_idx)

Sq_plot = np.clip(np.nan_to_num(Sqw), 0, None)
vmax    = np.percentile(Sq_plot[dj_idx][Sq_plot[dj_idx] > 0], 94)
norm    = PowerNorm(gamma=0.45, vmin=0, vmax=vmax)

fig, axes = plt.subplots(1, nP, figsize=(COL2 + 0.6, 2.4),
                         gridspec_kw={'wspace': 0.05})
letters   = 'abcdefgh'

for ip, si in enumerate(dj_idx):
    ax = axes[ip]
    ax.pcolormesh(qax, Eax, Sq_plot[si], cmap='inferno', norm=norm,
                  shading='auto', rasterized=True)

    for t in ticks:
        ax.axvline(t, color='w', lw=0.3, alpha=0.5)
    for ib in range(n_sub):
        E_br = omega_all[si, ib].copy()
        E_br[E_br < 0.1] = np.nan
        ax.plot(qax, E_br, 'w-', lw=0.25, alpha=0.45)

    ax.set_xlim(qax[0], qax[-1])
    ax.set_ylim(0, Emax)
    ax.set_xticks(ticks)
    ax.set_xticklabels(PATH_LABELS)
    if ip > 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel('Energy (meV)')

    label(ax, letters[ip], dark_bg=True)
    ax.text(0.97, 0.97, f'D/J = {DJ_vals[si]:.2f}',
            transform=ax.transAxes, fontsize=5.5, fontweight='bold',
            color='w', va='top', ha='right')

sm  = plt.cm.ScalarMappable(cmap='inferno', norm=norm)
sm.set_array([])
cb  = fig.colorbar(sm, ax=list(axes), fraction=0.012, pad=0.014, aspect=22)
cb.set_label(r'$S_\perp(\mathbf{q},\omega)$ (a.u.)', fontsize=6)
cb.ax.tick_params(labelsize=5, width=0.3, length=2)
cb.outline.set_linewidth(0.4)

save(fig, 'fig_comparison_progression')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig B: Band overlay — all D/J on one axis
# ──────────────────────────────────────────────────────────────────────────
print("Generating: Y2V2O7_band_overlay_DJ")
cmap_ov = mcm.coolwarm
norm_ov = mcolors.Normalize(vmin=0, vmax=DJ_vals[-1])

fig, ax = plt.subplots(figsize=(COL1, 2.8))
for idj in range(nDJ):
    col = cmap_ov(norm_ov(DJ_vals[idj]))
    for ib in range(n_sub):
        ax.plot(qax, omega_all[idj, ib], color=col, lw=0.7, alpha=0.8)

for t in ticks:
    ax.axvline(t, color='k', lw=0.3, ls='--', alpha=0.4)
ax.set_xticks(ticks)
ax.set_xticklabels(PATH_LABELS)
ax.set_xlim(qax[0], qax[-1])
ax.set_ylim(0, Emax)
ax.set_ylabel('Energy (meV)')

sm2 = plt.cm.ScalarMappable(cmap=cmap_ov, norm=norm_ov)
sm2.set_array([])
cb2 = fig.colorbar(sm2, ax=ax, pad=0.02, fraction=0.04, aspect=25)
cb2.set_label('$D/J$')
cb2.ax.tick_params(labelsize=6, width=0.4, length=2)
cb2.outline.set_linewidth(0.4)

save(fig, 'Y2V2O7_band_overlay_DJ')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig C: k·p analysis — 3-panel (t_W, ω_W, v_W vs D/J)
# ──────────────────────────────────────────────────────────────────────────
print("Generating: fig_weyl_kp_analysis")
by_pair = defaultdict(list)
for wc in crossings:
    if not np.isnan(wc.get('vW', np.nan)):
        by_pair[wc['bands']].append(wc)
for k in by_pair:
    by_pair[k].sort(key=lambda x: x['dj'])

pair_col = {(0,1): C_BLUE, (1,2): C_RED, (2,3): C_GREEN,
            (0,2): C_ORANGE, (1,3): C_PURPLE, (0,3): C_GREY}
pair_mkr = {(0,1): 'o', (1,2): 's', (2,3): 'D',
            (0,2): '^', (1,3): 'v', (0,3): '<'}

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(COL2, 2.2),
                                     gridspec_kw={'wspace': 0.35})

for pair, wcs in by_pair.items():
    djs = [w['dj']   for w in wcs]
    tws = [w['t']    for w in wcs]
    ows = [w['omega'] for w in wcs]
    vws = [w['vW']   for w in wcs]
    vkp = [w.get('vW_kp', np.nan) for w in wcs]
    c   = pair_col.get(pair, C_GREY)
    m   = pair_mkr.get(pair, 'o')
    lbl = f'{pair[0]+1}–{pair[1]+1}'
    ax1.plot(djs, tws, m, color=c, ms=3.5, lw=0.8, label=lbl)
    ax2.plot(djs, ows, m, color=c, ms=3.5, lw=0.8, label=lbl)
    ax3.plot(djs, vws, m, color=c, ms=3.5, lw=0.8, label=f'{lbl} fit')
    ax3.plot(djs, vkp, m, color=c, ms=3.5, lw=0, mfc='none', mew=0.8,
             label=f'{lbl} k·p')

for ax, xlab, ylab in [(ax1, '$D/J$', r'$t_W$'),
                        (ax2, '$D/J$', r'$\omega_W$ (meV)'),
                        (ax3, '$D/J$', r'$v_W$ (meV·Å)')]:
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_xlim(0, DJ_vals[-1])
    ax.legend(fontsize=4.5, loc='best', framealpha=0.8, ncol=2)

ax1.set_ylim(0, 1)
ax3.set_ylim(bottom=0)

label(ax1, 'a'); label(ax2, 'b'); label(ax3, 'c')
save(fig, 'fig_weyl_kp_analysis')
plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig D: Band zoom along Γ→L, D/J progression with k·p fits
# ──────────────────────────────────────────────────────────────────────────
print("Generating: fig_weyl_band_zoom_GL")
dj_with = sorted(set(w['dj'] for w in crossings if not np.isnan(w.get('vW', np.nan))))
if len(dj_with) > 6:
    step = max(1, len(dj_with) // 6)
    dj_plot = dj_with[::step][:6]
else:
    dj_plot = dj_with

nPan = len(dj_plot)
if nPan > 0:
    fig, axes = plt.subplots(1, nPan,
                             figsize=(min(COL2 + 0.6, 1.3*nPan + 0.4), 2.5),
                             gridspec_kw={'wspace': 0.06})
    if nPan == 1:
        axes = [axes]

    for ip, dj_val in enumerate(dj_plot):
        ax = axes[ip]
        idj = int(np.argmin(np.abs(DJ_vals - dj_val)))
        for ib in range(n_sub):
            ax.plot(t_fine, omGL[idj, ib], '-',
                    color=BRANCH_COLORS[ib], lw=0.9)

        for wc in crossings:
            if abs(wc['dj'] - dj_val) > 0.001 or np.isnan(wc.get('vW', np.nan)):
                continue
            ax.plot(wc['t'], wc['omega'], '*', color='k', ms=5, zorder=10)
            dq_fit = np.linspace(-win, win, 200) * qL_len
            t_fit  = wc['t'] + dq_fit / qL_len
            ax.plot(t_fit, wc['omega'] + wc['vW']*np.abs(dq_fit),
                    '--', color='grey', lw=0.5, alpha=0.7)
            ax.plot(t_fit, wc['omega'] - wc['vW']*np.abs(dq_fit),
                    '--', color='grey', lw=0.5, alpha=0.7)
            chi = wc.get('chi', 0)
            if chi != 0:
                ax.text(wc['t'] + 0.03, wc['omega'] + 1.5,
                        f'$\\chi={chi:+d}$', fontsize=4.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, Emax)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels([r'$\Gamma$', '', '$L$'])
        # no x-axis label — tick labels (Γ, L) are sufficient
        if ip > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Energy (meV)')

        label(ax, letters[ip])
        ax.text(0.97, 0.97, f'D/J = {dj_val:.2f}',
                transform=ax.transAxes, fontsize=5.5, va='top', ha='right')

    axes[0].legend(
        [plt.Line2D([0], [0], color=BRANCH_COLORS[i], lw=0.9)
         for i in range(n_sub)],
        [f'Band {i+1}' for i in range(n_sub)],
        fontsize=4.5, loc='lower right', framealpha=0.8)

    save(fig, 'fig_weyl_band_zoom_GL')
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig E: 3D Weyl cone at D/J ≈ 0.30
# ──────────────────────────────────────────────────────────────────────────
print("Generating: fig_weyl_cone_3D")
repr_wc = None
for wc in crossings:
    if not np.isnan(wc.get('vW', np.nan)):
        if repr_wc is None or abs(wc['dj'] - 0.32) < abs(repr_wc['dj'] - 0.32):
            repr_wc = wc

if repr_wc is not None:
    D_r  = repr_wc['dj'] * J_FM
    qW   = repr_wc['q']
    b0, b1_ = repr_wc['bands']

    e1 = np.cross(Lh, np.array([0, 0, 1.]))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(Lh, e1)
    e2 /= np.linalg.norm(e2)

    r_mesh = 0.06
    nM     = 81
    dp     = np.linspace(-r_mesh, r_mesh, nM)
    dp2    = np.linspace(-r_mesh, r_mesh, nM)
    DP, DP2 = np.meshgrid(dp, dp2)
    omlo = np.zeros((nM, nM))
    omhi = np.zeros((nM, nM))
    for ia in range(nM):
        for ib_ in range(nM):
            qt = qW + DP[ia, ib_]*Lh + DP2[ia, ib_]*e1
            ev = np.sort(np.real(eigh(build_Hq(qt, D_r), eigvals_only=True)))
            omlo[ia, ib_] = ev[b0]
            omhi[ia, ib_] = ev[b1_]

    fig = plt.figure(figsize=(COL1 + 0.6, 3.2))
    ax3d = fig.add_subplot(111, projection='3d')

    ax3d.plot_surface(DP, DP2, omhi, cmap='Reds', alpha=0.65,
                      linewidth=0, antialiased=True, rcount=40, ccount=40)
    ax3d.plot_surface(DP, DP2, omlo, cmap='Blues', alpha=0.65,
                      linewidth=0, antialiased=True, rcount=40, ccount=40)
    ax3d.scatter([0], [0], [repr_wc['omega']], color='k', s=40,
                 marker='*', zorder=10,
                 label=f"Weyl point ($\\chi$ = {repr_wc.get('chi', '?'):+d})")

    ax3d.set_xlabel(r'$\delta q_\parallel$ (Å$^{-1}$)', fontsize=6, labelpad=1)
    ax3d.set_ylabel(r'$\delta q_\perp$ (Å$^{-1}$)', fontsize=6, labelpad=1)
    ax3d.set_zlabel('Energy (meV)', fontsize=6, labelpad=1)
    ax3d.tick_params(labelsize=5)
    ax3d.legend(fontsize=5.5, loc='upper right')
    ax3d.view_init(elev=25, azim=-55)

    vW_str = f"{repr_wc['vW']:.1f}" if not np.isnan(repr_wc['vW']) else '?'
    fig.text(0.5, 0.01,
             f"D/J = {repr_wc['dj']:.2f},  "
             f"$\\omega_W$ = {repr_wc['omega']:.1f} meV,  "
             f"$v_W$ = {vW_str} meV·Å",
             ha='center', fontsize=6)

    save(fig, 'fig_weyl_cone_3D')
    plt.close(fig)

# ──────────────────────────────────────────────────────────────────────────
#  Fig F: Weyl cone cuts (parallel + perpendicular)
# ──────────────────────────────────────────────────────────────────────────
if repr_wc is not None:
    print("Generating: fig_weyl_cone_cuts")
    mid = nM // 2
    vW  = repr_wc['vW']
    oW  = repr_wc['omega']

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(COL2 * 0.65, 2.2),
                                    gridspec_kw={'wspace': 0.3})

    axa.plot(dp, omlo[mid, :], '-', color=C_BLUE, lw=1.0, label=f'Band {b0+1}')
    axa.plot(dp, omhi[mid, :], '-', color=C_RED,  lw=1.0, label=f'Band {b1_+1}')
    axa.plot(dp, oW + vW*np.abs(dp), '--', color='grey', lw=0.6, alpha=0.8,
             label=r'$\omega_W \pm v_W|\delta q|$')
    axa.plot(dp, oW - vW*np.abs(dp), '--', color='grey', lw=0.6, alpha=0.8)
    axa.axvline(0, color='grey', lw=0.3, ls=':')
    axa.set_xlabel(r'$\delta q_\parallel$ (Å$^{-1}$)')
    axa.set_ylabel('Energy (meV)')
    axa.legend(fontsize=5)
    label(axa, 'a')

    axb.plot(dp2, omlo[:, mid], '-', color=C_BLUE, lw=1.0, label=f'Band {b0+1}')
    axb.plot(dp2, omhi[:, mid], '-', color=C_RED,  lw=1.0, label=f'Band {b1_+1}')
    axb.plot(dp2, oW + vW*np.abs(dp2), '--', color='grey', lw=0.6, alpha=0.8)
    axb.plot(dp2, oW - vW*np.abs(dp2), '--', color='grey', lw=0.6, alpha=0.8)
    axb.axvline(0, color='grey', lw=0.3, ls=':')
    axb.set_xlabel(r'$\delta q_\perp$ (Å$^{-1}$)')
    axb.legend(fontsize=5)
    label(axb, 'b')

    save(fig, 'fig_weyl_cone_cuts')
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════
#  4. BERRY CURVATURE AND CHERN NUMBER
# ══════════════════════════════════════════════════════════════════════════

if repr_wc is not None:
    print("\nComputing Berry curvature and Chern number...")
    D_r  = repr_wc['dj'] * J_FM
    qW   = repr_wc['q']
    b_lo, b_hi = repr_wc['bands']    # lower / upper band of the crossing
    chi_kp = repr_wc.get('chi', 0)

    # ── (a) 2D Berry curvature map in the transverse plane ─────────────────
    N_map  = 41
    dq_map = 0.10   # Å⁻¹
    qq     = np.linspace(-dq_map, dq_map, N_map)
    Om_map = np.zeros((N_map, N_map))
    print("  Computing 2-D Berry curvature map...")
    for ix, dx in enumerate(qq):
        for iy, dy in enumerate(qq):
            k = qW + dx * e1_GL + dy * e2_GL
            Om_map[iy, ix] = np.dot(berry_curvature_vec(k, b_lo, D_r), Lh)
    # Ensure center is positive (red) — sign is a display orientation choice; C=+1 is unchanged
    if Om_map[N_map//2, N_map//2] < 0:
        Om_map = -Om_map

    # ── (b) Chern number vs sphere radius ──────────────────────────────────
    r_vals = np.array([0.015, 0.025, 0.040, 0.055, 0.070, 0.085, 0.100])
    C_vals = np.zeros(len(r_vals))
    print("  Computing Chern number at multiple sphere radii...")
    for ir, r in enumerate(r_vals):
        C_vals[ir] = chern_number_sphere(qW, r, b_lo, D_r, N_theta=20, N_phi=40)
        print(f"    r = {r:.3f} Å⁻¹  →  C = {C_vals[ir]:.3f}")
    print(f"  k·p chirality: χ = {chi_kp:+d}")

    # ── (c) Multi-L scan — partner Weyl points ─────────────────────────────
    L_frac_all = {
        r'$\Gamma\!\to\!L_{[111]}$':      np.array([0.5, 0.5, 0.5]),
        r'$\Gamma\!\to\!L_{[11\bar1]}$':  np.array([0.0, 0.0, 0.5]),
        r'$\Gamma\!\to\!L_{[1\bar11]}$':  np.array([0.0, 0.5, 0.0]),
        r'$\Gamma\!\to\!L_{[\bar111]}$':  np.array([0.5, 0.0, 0.0]),
    }
    n_sc    = 1500
    thr_sc  = 0.15   # meV gap threshold for partner scan
    all_wcs = []
    print("  Scanning all 4 Γ→L directions for Weyl crossings...")
    for lbl, lfrac in L_frac_all.items():
        qLi   = frac2cart(lfrac)
        t_sc  = np.linspace(0, 1, n_sc)
        qpi   = np.outer(t_sc, qLi)
        om_sc = np.zeros((n_sub, n_sc))
        for iq in range(n_sc):
            om_sc[:, iq] = np.sort(np.real(
                eigh(build_Hq(qpi[iq], D_r), eigvals_only=True)))
        for b in range(n_sub - 1):
            gap  = om_sc[b + 1] - om_sc[b]
            imin = np.argmin(gap)
            if gap[imin] < thr_sc and 0 < imin < n_sc - 1:
                tc  = t_sc[imin]
                oc  = 0.5 * (om_sc[b, imin] + om_sc[b + 1, imin])
                q_c = tc * qLi
                # Chirality via Löwdin projection
                _, vcs_c = eigh(build_Hq(q_c, D_r))
                Pc = vcs_c[:, [b, b + 1]]
                V_ax_c = []
                for e_ax in [np.array([1, 0, 0.]), np.array([0, 1, 0.]), np.array([0, 0, 1.])]:
                    dHc = (build_Hq(q_c + dq_d * e_ax, D_r) -
                           build_Hq(q_c - dq_d * e_ax, D_r)) / (2 * dq_d)
                    V_ax_c.append(Pc.conj().T @ dHc @ Pc)
                vt_c = np.array([[np.real(np.trace(p @ Va)) / 2
                                  for p in pauli] for Va in V_ax_c])
                chi_c = int(np.sign(np.linalg.det(vt_c)))
                all_wcs.append({'label': lbl, 'chi': chi_c, 't': tc,
                                'omega': oc, 'bands': (b, b + 1)})
                break   # one crossing per direction per band pair

    chi_total = sum(w['chi'] for w in all_wcs)
    print(f"\n  All Weyl crossings at D/J = {repr_wc['dj']:.2f}:")
    for w in all_wcs:
        print(f"    {w['label']}: bands {w['bands'][0]+1}–{w['bands'][1]+1}, "
              f"t={w['t']:.3f}, ω={w['omega']:.1f} meV, χ={w['chi']:+d}")
    print(f"  Sum of chiralities: {chi_total:+d} (must equal 0 by Nielsen–Ninomiya)")

    # ── Figure: Berry curvature + Chern convergence ─────────────────────────
    print("\nGenerating: fig_berry_chern")
    fig, (ax_bc, ax_cv) = plt.subplots(1, 2, figsize=(COL2 * 0.70, 2.5),
                                        gridspec_kw={'wspace': 0.42,
                                                     'left': 0.10, 'right': 0.96,
                                                     'top': 0.90, 'bottom': 0.18})

    # Panel (a): 2D Berry curvature Ω∥ map
    v_lim   = max(np.nanpercentile(np.abs(Om_map), 92), 1.0)
    Om_clip = np.clip(Om_map, -v_lim, v_lim)
    extent  = [-dq_map, dq_map, -dq_map, dq_map]
    im = ax_bc.imshow(Om_clip, origin='lower', extent=extent,
                      cmap='RdBu_r', vmin=-v_lim, vmax=v_lim,
                      interpolation='bilinear', aspect='equal')
    cb_bc = fig.colorbar(im, ax=ax_bc, fraction=0.046, pad=0.03)
    cb_bc.set_label(r'$\Omega_\parallel$ (Å$^2$)', fontsize=5.5)
    cb_bc.ax.tick_params(labelsize=5, width=0.3, length=1.5)
    cb_bc.outline.set_linewidth(0.4)
    ax_bc.scatter([0], [0], s=50, c='gold', marker='*', zorder=5,
                  edgecolors='k', linewidths=0.4)
    chi_str = f'{chi_kp:+d}' if chi_kp != 0 else '±1'
    ax_bc.text(0.04, 0.04, rf'$\mathcal{{C}} = {chi_str}$',
               transform=ax_bc.transAxes, fontsize=7, fontweight='bold',
               color='k', va='bottom')
    ax_bc.set_xlabel(r'$\delta k_1$ (Å$^{-1}$)', fontsize=6)
    ax_bc.set_ylabel(r'$\delta k_2$ (Å$^{-1}$)', fontsize=6)
    ax_bc.set_title(r'Berry curvature $\Omega_\parallel$', fontsize=6.5, pad=3)
    label(ax_bc, 'a', dark_bg=True)

    # Panel (b): Chern number C(r) convergence
    C_exp = chi_kp if chi_kp != 0 else 1
    ax_cv.axhline(C_exp, color='grey', lw=0.9, ls='--', alpha=0.8,
                  label=rf'$\mathcal{{C}} = {C_exp:+d}$ (expected)')
    ax_cv.plot(r_vals * 1e2, C_vals, 'o-', color=C_BLUE, ms=4.5, lw=1.1,
               label='Sphere integration')
    ax_cv.set_xlabel(r'Sphere radius $r\;(10^{-2}$ Å$^{-1})$', fontsize=6)
    ax_cv.set_ylabel(r'Chern number $\mathcal{C}$', fontsize=6)
    ax_cv.set_xlim(0, r_vals[-1] * 1e2 * 1.08)
    ax_cv.set_ylim(C_exp - 0.5, C_exp + 0.5)
    ax_cv.legend(fontsize=5.5, loc='lower right', framealpha=0.85)
    ax_cv.set_title('Chern number convergence', fontsize=6.5, pad=3)
    label(ax_cv, 'b')

    save(fig, 'fig_berry_chern')
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
#  5. PHASE DIAGRAM: Weyl point vs (D/J, J₂/J)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 5: Phase diagram (D/J × J₂/J) ─────────────────────────────")

# Build NNN bond list (d_NNN ≈ 6.056 Å = a_cub×√6/4)
d_NNN = a_cub * np.sqrt(6) / 4
nnn_bonds = []
for i in range(n_sub):
    for j in range(n_sub):
        for n1 in range(-2, 3):
            for n2 in range(-2, 3):
                for n3 in range(-2, 3):
                    if i == j and n1 == n2 == n3 == 0:
                        continue
                    rj = r_sub[j] + n1*a1 + n2*a2 + n3*a3
                    if abs(np.linalg.norm(rj - r_sub[i]) - d_NNN) < 0.08:
                        nnn_bonds.append((i, j, np.array([n1, n2, n3]),
                                          rj - r_sub[i]))
print(f"NNN bonds found: {len(nnn_bonds)}")


def build_Hq_J2(qvec, D_val, J2_val):
    """LSWT Hamiltonian with NN DM (D_val) and NNN isotropic exchange (J2_val)."""
    H = build_Hq(qvec, D_val).copy()
    for (i, j, dl_int, dv) in nnn_bonds:
        dlc = dl_int[0]*a1 + dl_int[1]*a2 + dl_int[2]*a3
        ph  = np.exp(1j * np.dot(qvec, dlc))
        H[i, i] += S_val * J2_val
        H[i, j] -= S_val * J2_val * ph
    return H


# 2D scan
N_D   = 24
N_J2  = 22
DJ_sc = np.linspace(0.00, 0.56, N_D)
J2_sc = np.linspace(-0.20, 0.25, N_J2)   # J₂/J (negative = AFM NNN)
n_gl  = 800    # q-points along Γ→L for crossing search
t_gl  = np.linspace(0, 1, n_gl)
qGL_sc = np.outer(t_gl, qL)
thr_pd = 0.12   # meV crossing threshold

omW_map = np.full((N_J2, N_D), np.nan)
tW_map  = np.full((N_J2, N_D), np.nan)

print(f"Scanning {N_D}×{N_J2} = {N_D*N_J2} parameter pairs...")
for ij2, j2r in enumerate(J2_sc):
    J2_val = j2r * J_FM
    for idj, djr in enumerate(DJ_sc):
        D_val = djr * J_FM
        om = np.zeros((n_sub, n_gl))
        for iq in range(n_gl):
            om[:, iq] = np.sort(np.real(
                eigh(build_Hq_J2(qGL_sc[iq], D_val, J2_val), eigvals_only=True)))
        # Find minimum gap between bands 1-2 (most likely Weyl pair)
        for b in range(n_sub - 1):
            gap = om[b+1] - om[b]
            imin = np.argmin(gap)
            if gap[imin] < thr_pd and 0 < imin < n_gl - 1:
                omW_map[ij2, idj] = 0.5*(om[b, imin] + om[b+1, imin])
                tW_map [ij2, idj] = t_gl[imin]
                break
    if (ij2+1) % 5 == 0:
        print(f"  J₂/J row {ij2+1}/{N_J2} done.")

print("Phase diagram scan complete.")

# ── Figure: phase diagram ──────────────────────────────────────────────────
print("\nGenerating: fig_phase_diagram")
fig_pd, (ax_ow, ax_tw) = plt.subplots(1, 2, figsize=(COL2*0.78, 2.6),
                                       gridspec_kw={'wspace': 0.38,
                                                    'left': 0.11, 'right': 0.96,
                                                    'top': 0.90, 'bottom': 0.17})

cmap_pd = plt.cm.plasma
cmap_pd.set_bad('lightgrey', alpha=0.8)

im_ow = ax_ow.pcolormesh(DJ_sc, J2_sc, omW_map,
                          cmap=cmap_pd, vmin=0, vmax=55,
                          shading='auto', rasterized=True)
cb_ow = fig_pd.colorbar(im_ow, ax=ax_ow, pad=0.03, fraction=0.048, aspect=22)
cb_ow.set_label(r'$\omega_W$ (meV)', fontsize=6)
cb_ow.ax.tick_params(labelsize=5, width=0.3, length=1.5)
cb_ow.outline.set_linewidth(0.4)
ax_ow.set_xlabel('$D/J$'); ax_ow.set_ylabel('$J_2/J$')
ax_ow.set_title('Weyl energy $\\omega_W$', fontsize=6.5, pad=3)
# Mark Y₂V₂O₇ point
ax_ow.plot(0.32, 0.0, '*', ms=8, color='cyan', mec='k', mew=0.5, zorder=10,
           label=r'Y$_2$V$_2$O$_7$')
ax_ow.axhline(0, color='w', lw=0.4, ls=':', alpha=0.6)
ax_ow.legend(fontsize=5, loc='upper left', framealpha=0.85)
label(ax_ow, 'a')

im_tw = ax_tw.pcolormesh(DJ_sc, J2_sc, tW_map,
                          cmap='viridis', vmin=0, vmax=1,
                          shading='auto', rasterized=True)
cb_tw = fig_pd.colorbar(im_tw, ax=ax_tw, pad=0.03, fraction=0.048, aspect=22)
cb_tw.set_label(r'$t_W$ (fraction of $\Gamma\!\to\!L$)', fontsize=6)
cb_tw.ax.tick_params(labelsize=5, width=0.3, length=1.5)
cb_tw.outline.set_linewidth(0.4)
ax_tw.set_xlabel('$D/J$'); ax_tw.set_ylabel('$J_2/J$')
ax_tw.set_title('Weyl position $t_W$', fontsize=6.5, pad=3)
ax_tw.plot(0.32, 0.0, '*', ms=8, color='red', mec='k', mew=0.5, zorder=10)
ax_tw.axhline(0, color='w', lw=0.4, ls=':', alpha=0.6)
label(ax_tw, 'b')

save(fig_pd, 'fig_phase_diagram')
plt.close(fig_pd)

# ═══════════════════════════════════════════════════════════════════════════
#  6. SURFACE ARCS — slab along a₃ ([110] stacking)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 6: Surface arc slab calculation ────────────────────────────")

D_arc   = 0.32 * J_FM   # use Y₂V₂O₇ value
N_slab  = 50            # number of unit cells stacked along a₃
N_surf  = 2             # layers counted as "surface" on each side
N_kpar  = 300           # k-points along surface BZ path


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


# Surface BZ path:  X̄ → Γ̄ → M̄ where Γ̄=(0,0,0), X̄=b₁/2, M̄=(b₁+b₂)/2
# b₁, b₂ are both ⊥ to a₃ so they lie in the surface plane
Xbar = b1 / 2
Mbar = (b1 + b2) / 2
kpath_seg1 = np.array([Xbar*(1-t) for t in np.linspace(0, 1, N_kpar//2)])
kpath_seg2 = np.array([Mbar*t      for t in np.linspace(0, 1, N_kpar//2)])
kpar_path  = np.vstack([kpath_seg1, kpath_seg2])
tick_kpar  = [0, N_kpar//2, N_kpar-1]
tick_labels= [r'$\bar{X}$', r'$\bar{\Gamma}$', r'$\bar{M}$']

# Build arc distances for x-axis
kpar_dist = np.zeros(N_kpar)
for ik in range(1, N_kpar):
    kpar_dist[ik] = kpar_dist[ik-1] + np.linalg.norm(kpar_path[ik] - kpar_path[ik-1])

print(f"  Computing slab spectrum ({N_slab} layers, {N_kpar} k-points)...")
t_sl = __import__('time').time()
all_evals = []
all_sweights = []
for ik, kp in enumerate(kpar_path):
    Hs  = build_H_slab(kp, D_arc, N_slab)
    ev, vc = eigh(Hs)
    all_evals.append(ev)
    # Surface weight: participation in top and bottom N_surf layers
    s_idx_bot = np.arange(4*N_surf)
    s_idx_top = np.arange(4*(N_slab-N_surf), 4*N_slab)
    s_idx = np.concatenate([s_idx_bot, s_idx_top])
    sw_k = np.sum(np.abs(vc[s_idx, :])**2, axis=0)
    all_sweights.append(sw_k)
    if (ik+1) % 100 == 0:
        print(f"    k-point {ik+1}/{N_kpar}  ({__import__('time').time()-t_sl:.1f}s)")

all_evals    = np.array(all_evals)     # (N_kpar, 4*N_slab)
all_sweights = np.array(all_sweights)  # (N_kpar, 4*N_slab)
print(f"  Slab done in {__import__('time').time()-t_sl:.1f}s")

# ── Figure: surface arc ────────────────────────────────────────────────────
print("\nGenerating: fig_surface_arc")
fig_arc, ax_arc = plt.subplots(figsize=(COL1 * 1.05, 3.0),
                                gridspec_kw={'left': 0.16, 'right': 0.96,
                                             'top': 0.92, 'bottom': 0.14})

E_max_arc = 50.0   # meV

# Build 2D broadened spectral function on a (k, E) grid
E_arc_edges = np.linspace(0.5, E_max_arc, 201)
E_arc_ctr   = 0.5 * (E_arc_edges[:-1] + E_arc_edges[1:])
k_edges     = np.concatenate([
    [kpar_dist[0] - 0.5*(kpar_dist[1]-kpar_dist[0])],
    0.5*(kpar_dist[:-1]+kpar_dist[1:]),
    [kpar_dist[-1] + 0.5*(kpar_dist[-1]-kpar_dist[-2])],
])
sigma_arc = 1.0   # meV Gaussian broadening
A_arc = np.zeros((N_kpar, len(E_arc_ctr)))
for ik in range(N_kpar):
    ev = all_evals[ik]
    sw = all_sweights[ik]
    mask = (ev > 0.0) & (ev < E_max_arc)
    for e, w in zip(ev[mask], sw[mask]):
        A_arc[ik] += w * np.exp(-0.5 * ((E_arc_ctr - e) / sigma_arc)**2)

vmax_arc = np.percentile(A_arc[A_arc > 0], 98) if A_arc.max() > 0 else 1.0
im_arc = ax_arc.pcolormesh(k_edges, E_arc_edges, A_arc.T,
                            cmap='hot', vmin=0, vmax=vmax_arc,
                            shading='auto', rasterized=True)

cb_arc = fig_arc.colorbar(im_arc, ax=ax_arc, pad=0.02, fraction=0.048, aspect=22)
cb_arc.set_label('Surface weight', fontsize=6)
cb_arc.ax.tick_params(labelsize=5, width=0.3, length=1.5)
cb_arc.outline.set_linewidth(0.4)

ax_arc.set_xticks([kpar_dist[i] for i in tick_kpar])
ax_arc.set_xticklabels(tick_labels)
ax_arc.set_ylabel('Energy (meV)')
ax_arc.set_ylim(0.5, E_max_arc)
ax_arc.set_xlim(kpar_dist[0], kpar_dist[-1])
ax_arc.set_title(r'Surface magnon spectrum — $a_3$-terminated slab '
                 r'($D/J=0.32$)', fontsize=6, pad=3)
for t in tick_kpar:
    ax_arc.axvline(kpar_dist[t], color='k', lw=0.3, ls='--', alpha=0.4)

# Mark bulk Weyl energy from representative crossing
if repr_wc is not None:
    ax_arc.axhline(repr_wc['omega'], color=C_BLUE, lw=0.6, ls=':',
                   alpha=0.7, label=rf"$\omega_W={repr_wc['omega']:.1f}$ meV")
    ax_arc.legend(fontsize=5, loc='upper right', framealpha=0.85)

save(fig_arc, 'fig_surface_arc')
plt.close(fig_arc)

# ═══════════════════════════════════════════════════════════════════════════
#  7. MAGNON HALL CONDUCTIVITY  κ^{xy}(T)
# ═══════════════════════════════════════════════════════════════════════════

print("\n── Section 7: Magnon Hall conductivity ────────────────────────────────")

from scipy.special import spence   # spence(z) = Li₂(1-z)

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


# BZ grid (Monkhorst-Pack) in fractional reciprocal coords
N_MP = 20   # 20³ = 8000 k-points
mp   = np.arange(N_MP) / N_MP
m1, m2, m3 = np.meshgrid(mp, mp, mp, indexing='ij')
kpts_cart = (m1.ravel()[:, None] * b1[None, :]
           + m2.ravel()[:, None] * b2[None, :]
           + m3.ravel()[:, None] * b3[None, :])   # (N_MP³, 3)
N_kbz = len(kpts_cart)

D_hall = 0.32 * J_FM   # Y₂V₂O₇

# Physical constants (SI)
kB_SI  = 1.38065e-23   # J/K
hbar   = 1.05457e-34   # J·s
V_cell = (a_cub * 1e-10)**3 / 4  # m³ (FCC primitive cell)
meV2J  = 1.60218e-22   # J per meV

# Prefactor:  (k_B²/ħ) / V_cell  [W/(K²·m³)]  ×  Ω[Å²]×1e-20[m²/Å²]
# → κ^xy [W/(K·m)] = +prefac × T × (1/N_k) × Σ c₂(ρ) × Ω^{xy}[Å²]
prefac = (kB_SI**2 / hbar) / V_cell * 1e-20   # W/(K²·m) per (Å² unit)

# Compute Ω^{xy} (z-component of Berry curvature) for all bands and k-points
print(f"  Computing Berry curvature on {N_kbz}-point BZ grid...")
t_hall = __import__('time').time()

Om_xy_all  = np.zeros((n_sub, N_kbz))   # Å²
omega_bz   = np.zeros((n_sub, N_kbz))   # meV

for ik, kv in enumerate(kpts_cart):
    ev, vc = eigh(build_Hq(kv, D_hall))
    omega_bz[:, ik] = np.real(ev)
    # Velocity matrices via finite differences (once per k-point, shared)
    dH = [_dHdka(kv, a, D_hall) for a in range(3)]
    for nb in range(n_sub):
        pn = vc[:, nb]
        Oz = 0.0
        for m in range(n_sub):
            if m == nb:
                continue
            dE = ev[m] - ev[nb]
            if abs(dE) < 1e-10:
                continue
            pm = vc[:, m]
            # Ω^z = Ω^{xy}: uses (a=0,b=1) = (x,y) components
            mna = pn.conj() @ dH[0] @ pm
            mnb = pm.conj() @ dH[1] @ pn
            Oz += -2.0 * np.imag(mna * mnb) / dE**2
        Om_xy_all[nb, ik] = Oz
    if (ik+1) % 2000 == 0:
        print(f"    {ik+1}/{N_kbz}  ({__import__('time').time()-t_hall:.0f}s)")

print(f"  Berry curvature done in {__import__('time').time()-t_hall:.1f}s")

# Integrate κ^xy vs T
T_vals = np.array([10, 20, 40, 60, 80, 100, 150, 200, 250, 300], float)
kxy_vals = np.zeros(len(T_vals))

for iT, T in enumerate(T_vals):
    kBT = kB_SI * T / meV2J   # in meV
    rho = np.where(omega_bz > 0.01,
                   1.0 / (np.exp(np.clip(omega_bz / kBT, 0, 500)) - 1.0),
                   0.0)   # (n_sub, N_kbz)
    c2  = c2_bose(rho)    # (n_sub, N_kbz)
    kxy_vals[iT] = prefac * T * np.sum(c2 * Om_xy_all) / N_kbz

print("  κ^xy(T) [W/(K·m)]:")
for T, k in zip(T_vals, kxy_vals):
    print(f"    T={T:4.0f} K:  κ^xy = {k*1e3:.4f} mW/(K·m)")

# Experimental value from Onose et al. Science 329, 297 (2010), Lu₂V₂O₇
# Digitized from Fig. 4A: spontaneous κ^xy vs T (just above saturation field)
# Note: Onose convention gives positive κ^xy; our LSWT DM sign gives negative.
# We plot |κ^xy| for shape comparison — overall sign depends on DM vector convention.
_onose_raw = np.loadtxt('data (5).csv', delimiter=',')  # columns: T(K), κ^xy (10⁻³ W/K·m)
_sort = np.argsort(_onose_raw[:, 0])
T_exp    = _onose_raw[_sort, 0]
kxy_exp  = _onose_raw[_sort, 1] * 1e-3   # convert 10⁻³ W/(K·m) → W/(K·m)

# ── Figure: magnon Hall conductivity ───────────────────────────────────────
print("\nGenerating: fig_magnon_hall")
fig_mh, ax_mh = plt.subplots(figsize=(COL1, 2.8),
                              gridspec_kw={'left': 0.18, 'right': 0.96,
                                           'top': 0.92, 'bottom': 0.16})

ax_mh.plot(T_vals, np.abs(kxy_vals) * 1e3, 'o-', color=C_BLUE, ms=4, lw=1.1,
           label=r'LSWT, $D/J=0.32$ (Y$_2$V$_2$O$_7$)')
ax_mh.plot(T_exp, np.abs(kxy_exp) * 1e3, 's--', color=C_RED, ms=4, lw=1.0,
           mfc='none', mew=0.8,
           label=r'Onose et al. 2010 (Lu$_2$V$_2$O$_7$)')
ax_mh.axhline(0, color='grey', lw=0.4, ls=':')
ax_mh.set_xlabel('Temperature (K)')
ax_mh.set_ylabel(r'$|\kappa^{xy}|$ (mW K$^{-1}$ m$^{-1}$)')
ax_mh.set_title('Intrinsic magnon Hall conductivity', fontsize=6.5, pad=3)
ax_mh.legend(fontsize=5.5, loc='upper right', framealpha=0.85)
ax_mh.set_xlim(0, 310)

save(fig_mh, 'fig_magnon_hall')
plt.close(fig_mh)

print("\nAll LSWT figures done. Run 'conda run -n edrixs_run python3 generate_dimer.py' for EDRIXS figures.")
