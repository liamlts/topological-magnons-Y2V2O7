"""
generate_dimer_anderson.py
==========================
Two-Impurity Cluster Anderson Model (TICAM) for a V–V dimer at the V L_3 edge.

Extends `generate_dimer_pure.py` by embedding the two V sites in a discrete
O 2p bath that hybridises with the V t2g manifold.  Captures the physics
that is MISSING from the pure dimer:

  * Covalent V–O hybridisation → ground-state admixture  |d²⟩ + α|d³L⁻¹⟩
  * Dynamic core-hole screening: the intermediate state pulls ligand
    charge to the core-hole site  (d³ → d⁴L⁻¹ process)
  * Charge-transfer satellites in XAS and RIXS
  * Renormalised effective Hubbard U (Zaanen–Sawatzky–Allen shift)

Bath topology
-------------
We include a **single bridging O 2p orbital** shared between the two V
sites (one spatial × two spin), coupled to each V t2g manifold with a
spin-diagonal, orbital-isotropic hybridisation V_pd.  This is the
minimal TICAM geometry that reproduces the charge-transfer physics
visible in the V L_3 RIXS intermediate state: covalent admixture
|d²⟩ + α|d³L⁻¹⟩ in the ground state, charge-transfer satellites in
XAS and RIXS, and dynamic core-hole screening.

The bath does *not* generate a realistic V–V superexchange on its own:
a single shared bath orbital hybridises only with the a1g symmetric
combination of t2g orbitals, so the bath-mediated exchange scales as
V_pd⁴ / (Δ_CT+U)² in one channel and is negligible at physical V_pd.
A direct V–V hopping t_direct (inherited from the pure-dimer model)
therefore remains the primary J generator; the bath adds the
CT / covalency physics on top.

Total Hilbert space (26 spin-orbitals):
  [0:6]    V t2g  site A   (3 spatial × 2 spin, real cubic basis)
  [6:12]   V t2g  site B
  [12:14]  bridging O 2p   (1 spatial × 2 spin)
  [14:20]  V 2p core site A
  [20:26]  V 2p core site B

Initial state  : 4 e in 14 valence orbs × (12,12) core = C(14,4) = 1001
Intermediate   : 5 valence e × C(6,5)·C(6,6) core = 2002 × 6 = 12012 per site

Both are dense-diagonalisable via `scipy.linalg.eigh`.

Parameters beyond the pure dimer
--------------------------------
  Delta_CT   charge-transfer gap  (ε_d − ε_b) ≈ 4 eV for V⁴⁺ oxide
  V_pd       V t2g ↔ bath hybridisation (≈ 0.3–1.0 eV for V oxides)
  t_direct   direct V–V t2g hopping; set to the pure-dimer value
             77 meV so J ≈ 4t²/U matches the Y₂V₂O₇ magnon scale.

Outputs
-------
  Figures/fig_dimer_anderson.{pdf,png}                — low-E publication view
  Figures/fig_dimer_anderson_highE.{pdf,png}          — d–d + CT satellites
  Figures/fig_dimer_anderson_state_analysis.{pdf,png} — per-eigenstate
      ⟨S²_A⟩, ⟨S²_B⟩, ⟨S²_tot⟩, ⟨S_z⟩, ⟨N⟩, ⟨n_L⟩
  Console : singlet–triplet gap, GS covalency fraction, L₃ resonance.
"""
import os
import time
import numpy as np
import scipy
import scipy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from scipy.ndimage import convolve1d
from scipy.signal  import find_peaks
import edrixs

os.makedirs('Figures', exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  Plot style (match pure dimer figures)
# ══════════════════════════════════════════════════════════════════════════════
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'xtick.minor.width': 0.5, 'ytick.minor.width': 0.5,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'lines.linewidth': 1.1,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
COL1, COL2 = 3.386, 7.008
C_LV   = '#1f4e9c'
C_LH   = '#c93a3a'
C_CONS = '#111111'
C_FLIP = '#1f7a3e'
C_ACC  = '#6a3d9a'


def label(ax, letter, dark_bg=False, x=0.018, y=0.975):
    c = 'white' if dark_bg else 'black'
    ax.text(x, y, f'{letter}', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top', ha='left', color=c)


def gauss_convolve(spec, sigma_eV, dE_eV):
    hw = int(4 * sigma_eV / dE_eV) + 1
    k = np.arange(-hw, hw + 1) * dE_eV
    kernel = np.exp(-0.5 * (k / sigma_eV) ** 2)
    kernel /= kernel.sum()
    return np.convolve(spec, kernel, mode='same')


# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
# --- V 3d single-site parameters (shared with the pure-dimer script) --------
zeta_d_i   = 0.030            # eV   V 3d spin-orbit coupling
delta_trig = 0.010            # eV   trigonal D3d splitting within t2g
U_d        = 3.0              # eV   d–d Hubbard U
J_H        = 0.68             # eV   Hund's coupling
t_direct   = 0.077            # eV   direct V–V t2g–t2g hopping; primary
                               #      J generator via 4t²/U (see docstring)
core_v     = 4.0              # eV   static 2p-core → 3d attraction U_dc

# --- Anderson bath parameters -----------------------------------------------
Delta_CT   = 4.0              # eV   charge-transfer gap  ε_d − ε_b
V_pd       = 0.40             # eV   V t2g ↔ bridging-O 2p hybridisation;
                               #      sets GS covalency ⟨n_L⟩ ≈ 4 % and
                               #      drives CT satellites in the XAS/RIXS
                               #      intermediate state

F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(U_d, J_H)


def _atom_scalar(value):
    """Some edrixs atom-data entries are length-1 arrays; coerce to float."""
    return float(value[0] if hasattr(value, '__len__') else value)


atom    = edrixs.utils.get_atom_data('V', '3d', 1, edge='L3')
zeta_p  = _atom_scalar(atom['c_soc'])
gamma_c = _atom_scalar(atom['gamma_c'])
sn      = dict(atom['slater_n'])
SLATER_PD_SCALE = 0.8
F2_pd = sn.get('F2_12', 6.759) * SLATER_PD_SCALE
G1_pd = sn.get('G1_12', 5.014) * SLATER_PD_SCALE
G3_pd = sn.get('G3_12', 2.853) * SLATER_PD_SCALE
F0_pd = edrixs.get_F0('dp', G1_pd, G3_pd)

# --- Experimental geometry & broadening -------------------------------------
thin  = np.radians(15.0)       # grazing incidence
thout = np.radians(135.0)      # backscattering
gamma_f   = 0.002              # eV   final-state half-width (Lorentzian)
res_FWHM  = 0.030              # eV   instrument resolution (FWHM, Gaussian)
sigma_res = res_FWHM / (2 * np.sqrt(2 * np.log(2)))
T_K       = 10.0               # K    base temperature (Boltzmann ensemble)

# --- XAS / RIXS absolute-energy anchors -------------------------------------
L3_EDGE_EV = 515.5              # V L3 main-edge absolute energy (experiment)
MAP_EMIN, MAP_EMAX = 512.5, 516.5
N_MAP_POINTS = 60

# J scale from the direct V–V hopping (bath-mediated exchange is negligible
# because the shared bath orbital couples only to the a1g t2g combination).
J_estimate = 4.0 * (t_direct ** 2) / U_d * 1e3
print("═" * 72)
print("  V–V DIMER  ×  TWO-IMPURITY CLUSTER ANDERSON MODEL")
print("═" * 72)
print(f"  V 3d :    ζ_d = {zeta_d_i*1e3:.0f} meV   Δ_trig = {delta_trig*1e3:.0f} meV   "
      f"U = {U_d:.1f} eV   J_H = {J_H:.2f} eV")
print(f"  Bath :    Δ_CT = {Delta_CT:.2f} eV   V_pd = {V_pd:.3f} eV   "
      f"t_direct = {t_direct*1e3:.0f} meV")
print(f"  Core :    ζ_p = {zeta_p:.3f} eV   U_dc = {core_v:.1f} eV   "
      f"Γ_c = {gamma_c:.3f} eV")
print(f"  Expected J via 4t²/U    : {J_estimate:.2f} meV "
      f"(from t_direct; bath contribution <0.1 meV)")
print("═" * 72)


# ══════════════════════════════════════════════════════════════════════════════
#  ORBITAL LAYOUT  (26 spin-orbitals)
# ══════════════════════════════════════════════════════════════════════════════
# [0:6]    t2g A      (real cubic basis: dxy, dyz, dzx × ↑,↓)
# [6:12]   t2g B
# [12:14]  bridging O 2p bath  (1 spatial × 2 spin:  b↑, b↓)
# [14:20]  2p core A
# [20:26]  2p core B
NORBS  = 26
TA_SL  = slice(0, 6)
TB_SL  = slice(6, 12)
BATH_SL = slice(12, 14)
CA_SL  = slice(14, 20)
CB_SL  = slice(20, 26)


# ══════════════════════════════════════════════════════════════════════════════
#  COULOMB TENSORS  (V 3d intra-site Slater; 2p-3d multipole in intermediate)
# ══════════════════════════════════════════════════════════════════════════════
umat_t2g_c = edrixs.get_umat_slater('t2g', F0_d, F2_d, F4_d)
umat_t2g   = edrixs.transform_utensor(umat_t2g_c, edrixs.tmat_c2r('t2g', True))

params_t2gp = [0.0, 0.0, 0.0, F0_pd, F2_pd, G1_pd, G3_pd, 0.0, 0.0]
umat_t2gp_c = edrixs.get_umat_slater('t2gp', *params_t2gp)
t12 = np.zeros((12, 12), dtype=complex)
t12[0:6, 0:6]   = edrixs.tmat_c2r('t2g', True)
t12[6:12, 6:12] = edrixs.tmat_c2r('p', True)
umat_t2gp = edrixs.transform_utensor(umat_t2gp_c, t12)

# Initial state Coulomb: V t2g intra-site only (bath non-interacting)
umat_i = np.zeros((NORBS,) * 4, dtype=complex)
umat_i[0:6, 0:6, 0:6, 0:6]   = umat_t2g
umat_i[6:12, 6:12, 6:12, 6:12] = umat_t2g

# Intermediate: add 2p-3d multipole on the core-hole site only
umat_n = np.zeros((2, NORBS, NORBS, NORBS, NORBS), dtype=complex)
for s in range(2):
    umat_n[s] += umat_i

# 12-orbital indx for embedding p-d Slater:
# first 6 indices = t2g on that site, last 6 = core on that site
indx = np.array([
    [0,  1,  2,  3,  4,  5,  14, 15, 16, 17, 18, 19],   # site A (t2g_A + core_A)
    [6,  7,  8,  9, 10, 11,  20, 21, 22, 23, 24, 25],   # site B
])
print("Embedding 2p–3d Slater integrals ...")
nz_uidx = np.argwhere(umat_t2gp != 0.0)
for s in range(2):
    for ijkl in nz_uidx:
        i, j, k, l = ijkl
        umat_n[s, indx[s, i], indx[s, j], indx[s, k], indx[s, l]] \
            += umat_t2gp[i, j, k, l]


# ══════════════════════════════════════════════════════════════════════════════
#  ONE-BODY HAMILTONIAN
# ══════════════════════════════════════════════════════════════════════════════
emat_i = np.zeros((NORBS, NORBS), dtype=complex)
emat_n = np.zeros((2, NORBS, NORBS), dtype=complex)

# --- V 3d spin–orbit coupling (both sites, all states) ---
soc_d = edrixs.cb_op(edrixs.atom_hsoc('t2g', zeta_d_i), edrixs.tmat_c2r('t2g', True))
soc_p = edrixs.cb_op(edrixs.atom_hsoc('p',   zeta_p),   edrixs.tmat_c2r('p',   True))
emat_i[TA_SL, TA_SL] += soc_d
emat_i[TB_SL, TB_SL] += soc_d
for s in range(2):
    emat_n[s, TA_SL, TA_SL] += soc_d
    emat_n[s, TB_SL, TB_SL] += soc_d
emat_n[0, CA_SL, CA_SL] += soc_p
emat_n[1, CB_SL, CB_SL] += soc_p

# --- Trigonal D₃d splitting in t2g (on both V sites) ---
H_trig3 = (delta_trig / 3.0) * (np.ones((3, 3)) - np.eye(3))
H_trig6 = np.zeros((6, 6), dtype=complex)
for i in range(3):
    for j in range(3):
        H_trig6[2*i,     2*j    ] = H_trig3[i, j]
        H_trig6[2*i + 1, 2*j + 1] = H_trig3[i, j]
emat_i[TA_SL, TA_SL] += H_trig6
emat_i[TB_SL, TB_SL] += H_trig6
for s in range(2):
    emat_n[s, TA_SL, TA_SL] += H_trig6
    emat_n[s, TB_SL, TB_SL] += H_trig6

# --- Direct V–V hopping (diagonal t2g A ↔ t2g B, both manifolds) ---
for i in range(6):
    emat_i[i,     6 + i] += -t_direct
    emat_i[6 + i, i    ] += -t_direct
    for s in range(2):
        emat_n[s, i,     6 + i] += -t_direct
        emat_n[s, 6 + i, i    ] += -t_direct

# --- Bridging-bath on-site energy: ε_b = −Δ_CT (V d taken as reference 0) ---
for b in range(12, 14):
    emat_i[b, b] -= Delta_CT
    for s in range(2):
        emat_n[s, b, b] -= Delta_CT

# --- V t2g ↔ bath hybridisation (spin-diagonal, orbital-isotropic V_pd) ---
# Bath spin index: 12 = b↑, 13 = b↓.
# t2g spin index follows tmat_c2r('t2g', True): orb0↑, orb0↓, orb1↑, orb1↓, orb2↑, orb2↓
#   → spin-↑ sites on (0, 2, 4),  spin-↓ sites on (1, 3, 5) within each 6-block.
for spin in range(2):
    b_idx = 12 + spin
    for alpha in range(3):
        dA = 2 * alpha + spin          # t2g A, spin σ
        dB = 6 + 2 * alpha + spin      # t2g B, spin σ
        emat_i[b_idx, dA] += V_pd
        emat_i[dA, b_idx] += V_pd
        emat_i[b_idx, dB] += V_pd
        emat_i[dB, b_idx] += V_pd
        for s in range(2):
            emat_n[s, b_idx, dA] += V_pd
            emat_n[s, dA, b_idx] += V_pd
            emat_n[s, b_idx, dB] += V_pd
            emat_n[s, dB, b_idx] += V_pd

# --- Static core-hole potential on the d orbitals of the core-hole site ---
for i in range(0, 6):
    emat_n[0, i, i] -= core_v
for i in range(6, 12):
    emat_n[1, i, i] -= core_v


# ══════════════════════════════════════════════════════════════════════════════
#  FOCK BASES
# ══════════════════════════════════════════════════════════════════════════════
# Initial : 4 electrons in 14 valence orbs (t2g_A:6 + t2g_B:6 + bath:2),
#           core fully filled (12 e in 12 orbs)
# Intermediate (core-hole site s):
#           5 e in 14 valence orbs, 5 e in core(s), 6 e in other core
print("\nBuilding Fock bases ...")
basis_i = edrixs.get_fock_bin_by_N(14, 4, 12, 12)
# In get_fock_bin_by_N the orbital layout inside the returned array must match
# our global layout.  Partition (14, 12) → [valence 14 orbs] [core 12 orbs].
# Our global layout is t2g_A(6) + t2g_B(6) + bath(2) + coreA(6) + coreB(6),
# which is exactly (14 val) + (12 core) so basis_i column j maps directly to
# global orbital j.
basis_n = [
    edrixs.get_fock_bin_by_N(14, 5, 6, 5, 6, 6),   # core hole on A
    edrixs.get_fock_bin_by_N(14, 5, 6, 6, 6, 5),   # core hole on B
]
ncfgs_i = len(basis_i)
ncfgs_n = len(basis_n[0])
print(f"  Initial       : {ncfgs_i} configurations")
print(f"  Intermediate  : {ncfgs_n} per core-hole site")

# Sanity checks on Fock-basis geometry.  These guard against silently
# miscounting electrons or mixing up the (valence | core) partition if the
# orbital layout above ever drifts out of sync with get_fock_bin_by_N.
assert ncfgs_i == 1001,  f"initial basis size {ncfgs_i} != 1001"
assert ncfgs_n == 12012, f"intermediate basis size {ncfgs_n} != 12012"
for cfg in basis_i[:8]:
    assert sum(cfg[0:14]) == 4,  "initial state: 4 e in valence expected"
    assert sum(cfg[14:26]) == 12, "initial state: full 2p core expected"
for cfg in basis_n[0][:4]:
    assert sum(cfg[0:14]) == 5, "intermediate: 5 e in valence expected"
    assert sum(cfg[14:20]) == 5, "intermediate: 1 hole in core A expected"
    assert sum(cfg[20:26]) == 6, "intermediate: full core B expected"


# ══════════════════════════════════════════════════════════════════════════════
#  DIAGONALISE H_i AND H_n
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding H_i (dense) ...")
t0 = time.time()
hmat_i  = edrixs.two_fermion(emat_i, basis_i, basis_i)
hmat_i += edrixs.four_fermion(umat_i, basis_i)
eval_i, evec_i = scipy.linalg.eigh(hmat_i)
del hmat_i
print(f"  done in {time.time() - t0:.1f} s")
E_rel = (eval_i - eval_i[0]) * 1000.0
print(f"  lowest levels (meV above GS):  {np.round(E_rel[:8], 3)}")
print(f"  → emergent singlet–triplet gap : {E_rel[1]:.3f} meV")

print("\nBuilding H_n per core-hole site (dense) ...")
eval_n = np.zeros((2, ncfgs_n))
evec_n = np.zeros((2, ncfgs_n, ncfgs_n), dtype=complex)
for s in range(2):
    t0 = time.time()
    hmat  = edrixs.two_fermion(emat_n[s], basis_n[s], basis_n[s])
    hmat += edrixs.four_fermion(umat_n[s], basis_n[s])
    eval_n[s], evec_n[s] = scipy.linalg.eigh(hmat)
    del hmat
    print(f"  site {s}: {time.time() - t0:.1f} s  "
          f"(E_n range {eval_n[s].min():.2f} → {eval_n[s].max():.2f} eV)")


# ══════════════════════════════════════════════════════════════════════════════
#  STATE ANALYSIS  (cf. He et al. 2024, CrI3_AIM_Fortran.ipynb)
# ------------------------------------------------------------------------------
# For every eigenstate of H_i we compute the diagonal expectation values of a
# set of site-resolved operators:
#   • <N_A>, <N_B>       electron count on V t2g manifold (site A / site B)
#   • <N_bath>           bath occupation, so L-hole weight = 2 − <N_bath>
#   • <S²_A>, <S²_B>     site spin-squared  →  local spin character
#   • <S²_tot>, <S_z>    total dimer spin quantum numbers
# This classifies each eigenstate by its charge ( d^n / d^(n+1)L^−1 content )
# and spin (singlet / triplet / higher multiplet).  Because we have the full
# eigenvectors from dense eigh we compute the expectation values directly,
# avoiding the density-matrix formalism used by edrixs' Fortran backend.
# ══════════════════════════════════════════════════════════════════════════════

def _site_spin_mats_1e(site_slice):
    """(Sx, Sy, Sz) one-body matrices acting on the 6-orb t2g block at site_slice.
    Ordering within the block is (orb0↑, orb0↓, orb1↑, orb1↓, orb2↑, orb2↓),
    matching edrixs.tmat_c2r('t2g', True)."""
    Sx = np.zeros((NORBS, NORBS), dtype=complex)
    Sy = np.zeros((NORBS, NORBS), dtype=complex)
    Sz = np.zeros((NORBS, NORBS), dtype=complex)
    base = site_slice.start
    for alpha in range(3):
        up = base + 2 * alpha
        dn = up + 1
        Sz[up, up] += 0.5;   Sz[dn, dn] -= 0.5
        Sx[up, dn] += 0.5;   Sx[dn, up] += 0.5
        Sy[up, dn] += -0.5j; Sy[dn, up] += +0.5j
    return Sx, Sy, Sz


N_ANA = min(400, ncfgs_i)
print(f"\nState analysis: computing site-resolved observables on the lowest "
      f"{N_ANA} eigenstates ...")
t0 = time.time()

# --- one-body operators ------------------------------------------------------
SxA1, SyA1, SzA1 = _site_spin_mats_1e(TA_SL)
SxB1, SyB1, SzB1 = _site_spin_mats_1e(TB_SL)

# --- many-body versions on basis_i -------------------------------------------
def _mb(op1e):
    return edrixs.two_fermion(op1e, basis_i, basis_i)

SxA = _mb(SxA1);  SyA = _mb(SyA1);  SzA = _mb(SzA1)
SxB = _mb(SxB1);  SyB = _mb(SyB1);  SzB = _mb(SzB1)
S2A  = SxA @ SxA + SyA @ SyA + SzA @ SzA
S2B  = SxB @ SxB + SyB @ SyB + SzB @ SzB
SxT  = SxA + SxB;  SyT = SyA + SyB;  SzT = SzA + SzB
S2T  = SxT @ SxT + SyT @ SyT + SzT @ SzT

V = evec_i[:, :N_ANA]

def _diag_expect(Op, V):
    return np.real(np.einsum('ji,jk,ki->i', np.conj(V), Op, V))

S2A_exp = _diag_expect(S2A, V)
S2B_exp = _diag_expect(S2B, V)
S2T_exp = _diag_expect(S2T, V)
SzT_exp = _diag_expect(SzT, V)

# --- number operators (diagonal in the Fock basis) ---------------------------
n_A_cfg    = np.zeros(ncfgs_i)
n_B_cfg    = np.zeros(ncfgs_i)
n_bath_cfg = np.zeros(ncfgs_i)
for a, cfg in enumerate(basis_i):
    n_A_cfg[a]    = sum(cfg[0:6])
    n_B_cfg[a]    = sum(cfg[6:12])
    n_bath_cfg[a] = sum(cfg[12:14])

w = np.abs(V) ** 2
N_A_exp    = (w * n_A_cfg[:, None]).sum(axis=0)
N_B_exp    = (w * n_B_cfg[:, None]).sum(axis=0)
N_bath_exp = (w * n_bath_cfg[:, None]).sum(axis=0)
L_hole_exp = 2.0 - N_bath_exp

# keep for later reporting
n_bath_diag = n_bath_cfg
L_fraction  = L_hole_exp[0]
print(f"  done in {time.time() - t0:.1f} s")
print(f"\n  Ground-state bath-hole weight  〈n_L〉 = {L_fraction*100:.2f} %   "
      f"(covalency admixture of |d³L⁻¹⟩)")

E_rel_ana = (eval_i[:N_ANA] - eval_i[0]) * 1000.0

print("\n  Lowest 12 eigenstates (state character):")
print("    #    E(meV)   <S²_A>  <S²_B>  <S²_tot>  <S_z>   <N_A>  <N_B>   L-hole")
for i in range(min(12, N_ANA)):
    print(f"   {i:2d}  {E_rel_ana[i]:8.3f}  {S2A_exp[i]:6.3f}  {S2B_exp[i]:6.3f}"
          f"   {S2T_exp[i]:6.3f}  {SzT_exp[i]:+6.3f}  "
          f"{N_A_exp[i]:5.3f}  {N_B_exp[i]:5.3f}  {L_hole_exp[i]*100:5.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITION OPERATORS  (V 2p → V 3d dipole, Cartesian)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding transition operators ...")
tran_ptod = edrixs.get_trans_oper('t2gp')
for j in range(3):
    tran_ptod[j] = edrixs.cb_op2(tran_ptod[j],
                                 edrixs.tmat_c2r('t2g', True),
                                 edrixs.tmat_c2r('p',   True))

dipole_n = np.zeros((2, 3, NORBS, NORBS), dtype=complex)
# site A: t2g_A ← core_A
dipole_n[0, :, 0:6,  14:20] = tran_ptod
# site B: t2g_B ← core_B
dipole_n[1, :, 6:12, 20:26] = tran_ptod

trop_n = np.zeros((2, 3, ncfgs_n, ncfgs_i), dtype=complex)
for s in range(2):
    for j in range(3):
        Tmb = edrixs.two_fermion(dipole_n[s, j], basis_n[s], basis_i)
        trop_n[s, j] = edrixs.cb_op2(Tmb, evec_n[s], evec_i)


# ══════════════════════════════════════════════════════════════════════════════
#  GROUND-STATE ENSEMBLE (Boltzmann at T)
# ══════════════════════════════════════════════════════════════════════════════
gs_list = list(range(min(4, ncfgs_i)))
gs_dist = edrixs.boltz_dist([eval_i[i] for i in gs_list], T_K)
print(f"\nGS list: {gs_list}   Boltzmann weights: {np.round(gs_dist, 4)}")


# ══════════════════════════════════════════════════════════════════════════════
#  CARTESIAN POLARIZATION VECTORS
# ══════════════════════════════════════════════════════════════════════════════
ki_hat   = np.array([ np.cos(thin),  0.0, -np.sin(thin)])
kout_hat = np.array([ np.cos(np.pi - thout), 0.0, np.sin(thout)])
e_LV      = np.array([0.0, 1.0, 0.0])
e_LH_in   = np.cross(ki_hat,   e_LV);  e_LH_in  /= np.linalg.norm(e_LH_in)
e_LH_out  = np.cross(kout_hat, e_LV);  e_LH_out /= np.linalg.norm(e_LH_out)


def xas_polarized(omi_rel_vals, e_in):
    x = np.zeros(len(omi_rel_vals))
    for s in range(2):
        for ig, igs in enumerate(gs_list):
            amp = sum(e_in[j] * trop_n[s, j][:, igs] for j in range(3))
            amp2 = np.abs(amp) ** 2
            dE   = eval_n[s] - eval_i[igs]
            diff = omi_rel_vals[:, None] - dE[None, :]
            L    = gamma_c / (diff ** 2 + gamma_c ** 2) / np.pi
            x += gs_dist[ig] * (L @ amp2)
    return x


# ── XAS ──────────────────────────────────────────────────────────────────────
print("\nComputing XAS (LV, LH, iso) ...")
ominc_rel = np.linspace(-14.0, 10.0, 1400)
xas_LV = xas_polarized(ominc_rel, e_LV)
xas_LH = xas_polarized(ominc_rel, e_LH_in)
xas_iso = (xas_LV + xas_LH +
           xas_polarized(ominc_rel, np.array([1.0, 0.0, 0.0]))) / 3.0

dE_rel       = float(ominc_rel[1] - ominc_rel[0])
xas_LV_conv  = gauss_convolve(xas_LV,  sigma_res, dE_rel)
xas_LH_conv  = gauss_convolve(xas_LH,  sigma_res, dE_rel)
xas_iso_conv = gauss_convolve(xas_iso, sigma_res, dE_rel)

omi_res_rel = ominc_rel[np.argmax(xas_iso_conv)]
omi_shift   = L3_EDGE_EV - omi_res_rel
ominc_xas   = ominc_rel + omi_shift
E_res       = omi_res_rel + omi_shift
print(f"  XAS main peak:  ω_rel = {omi_res_rel:.3f} eV  →  E_res = {E_res:.3f} eV")


# ══════════════════════════════════════════════════════════════════════════════
#  RIXS  (Kramers–Heisenberg, Cartesian polarisations)
# ══════════════════════════════════════════════════════════════════════════════
# Extended eloss grid: -100 meV to 6 eV to capture CT satellites if any
eloss_eV  = np.linspace(-0.100, 6.000, 12200)
eloss_meV = eloss_eV * 1000.0


def ffg_cartesian(omi_rel):
    F = np.zeros((3, 3, ncfgs_i, len(gs_list)), dtype=complex)
    for s in range(2):
        abs_trans = trop_n[s][:, :, gs_list]
        emi_trans = np.conj(np.transpose(trop_n[s], (0, 2, 1)))
        F += edrixs.scattering_mat(
            eval_i, eval_n[s], abs_trans, emi_trans, omi_rel, gamma_c
        )
    return F


def rixs_channel(F_tot, e_in, e_out):
    I      = np.zeros(len(eloss_eV))
    amp2_s = np.zeros(ncfgs_i)
    for ig, igs in enumerate(gs_list):
        F_scalar = np.einsum('j,jkf,k->f', e_out, F_tot[:, :, :, ig], e_in)
        amp2     = np.abs(F_scalar) ** 2
        amp2_s  += amp2 * gs_dist[ig]
        dEf      = eval_i - eval_i[igs]
        L = gamma_f / np.pi / ((eloss_eV[None, :] - dEf[:, None]) ** 2 + gamma_f ** 2)
        I += gs_dist[ig] * (amp2[:, None] * L).sum(axis=0)
    return I, amp2_s


print("\nComputing RIXS at resonance (σσ, σπ, πσ, ππ channels) ...")
t0 = time.time()
F_res = ffg_cartesian(omi_res_rel)

I_ss, s_ss = rixs_channel(F_res, e_LV,    e_LV)
I_sp, s_sp = rixs_channel(F_res, e_LV,    e_LH_out)
I_ps, s_ps = rixs_channel(F_res, e_LH_in, e_LV)
I_pp, s_pp = rixs_channel(F_res, e_LH_in, e_LH_out)

I_cons   = I_ss + I_pp
I_flip   = I_sp + I_ps
I_LV_iso = I_ss + I_sp
I_LH_iso = I_ps + I_pp
I_all    = I_cons + I_flip

s_cons = s_ss + s_pp
s_flip = s_sp + s_ps
s_LV   = s_ss + s_sp
s_LH   = s_ps + s_pp
print(f"  done in {time.time() - t0:.1f} s")

dE_el    = float(eloss_eV[1] - eloss_eV[0])
I_cons_r = gauss_convolve(I_cons,   sigma_res, dE_el)
I_flip_r = gauss_convolve(I_flip,   sigma_res, dE_el)
I_LV_r   = gauss_convolve(I_LV_iso, sigma_res, dE_el)
I_LH_r   = gauss_convolve(I_LH_iso, sigma_res, dE_el)
I_all_r  = gauss_convolve(I_all,    sigma_res, dE_el)
eloss_sticks_meV = (eval_i - eval_i[gs_list[0]]) * 1000.0


# ── 2D RIXS map (incident-energy scan, isotropic in/out) ────────────────────
ominc_scan     = np.linspace(MAP_EMIN, MAP_EMAX, N_MAP_POINTS)
ominc_scan_rel = ominc_scan - omi_shift
print(f"\nComputing 2D RIXS map ({MAP_EMIN:.1f} → {MAP_EMAX:.1f} eV, "
      f"{N_MAP_POINTS} incident energies, iso in/out) ...")
t0 = time.time()
I2d = np.zeros((N_MAP_POINTS, len(eloss_eV)))
for io, om in enumerate(ominc_scan_rel):
    F_om = ffg_cartesian(om)
    for e_in in (e_LV, e_LH_in):
        for e_out in (e_LV, e_LH_out):
            I_ch, _ = rixs_channel(F_om, e_in, e_out)
            I2d[io] += I_ch
print(f"  done in {time.time() - t0:.1f} s   peak = {np.max(I2d):.3e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES  (same layout as pure dimer: XAS + RIXS cut + 2D map)
# ══════════════════════════════════════════════════════════════════════════════
sig_inc  = 0.08
dE_inc   = float(ominc_scan[1] - ominc_scan[0])
hw_inc   = int(4 * sig_inc / dE_inc) + 1
ki       = np.arange(-hw_inc, hw_inc + 1) * dE_inc
kern_inc = np.exp(-0.5 * (ki / sig_inc) ** 2); kern_inc /= kern_inc.sum()
I2d_s    = convolve1d(I2d, kern_inc, axis=0, mode='nearest')


def _draw_sticks_below(ax, e_meV, amp, color, y0, depth, thresh,
                       lw=0.6, skip_elastic=True):
    if skip_elastic:
        inel = amp[np.abs(e_meV) > 0.5]
        amax = inel.max() if inel.size > 0 and inel.max() > 0 else 1.0
    else:
        amax = amp.max() if amp.max() > 0 else 1.0
    for e, a in zip(e_meV, amp):
        if skip_elastic and abs(e) < 0.5:
            continue
        if a > thresh * amax:
            ax.vlines(e, y0 - depth * min(a / amax, 1.0), y0,
                      color=color, lw=lw, alpha=0.9)


def make_figure(xlim_meV, ylim_map_meV, fname_base, tag,
                norm_window_meV=None, show_J_marker=True):
    print(f"\nGenerating figure [{tag}]:  eloss {xlim_meV} meV")
    fig = plt.figure(figsize=(COL2, 5.8))
    gs  = GridSpec(2, 2, hspace=0.42, wspace=0.26,
                   left=0.075, right=0.965, top=0.955, bottom=0.095,
                   height_ratios=[1.0, 1.05])
    ax_xas  = fig.add_subplot(gs[0, 0])
    ax_rixs = fig.add_subplot(gs[0, 1])
    ax_map  = fig.add_subplot(gs[1, :])

    # --- Panel a: XAS ---
    xas_norm = max(np.max(xas_LV_conv), np.max(xas_LH_conv), 1e-30)
    ax_xas.plot(ominc_xas, xas_LV_conv / xas_norm, color=C_LV, lw=1.3,
                label=r'LV ($\sigma$)')
    ax_xas.plot(ominc_xas, xas_LH_conv / xas_norm, color=C_LH, lw=1.3,
                label=r'LH ($\pi$)')
    ax_xas.axvline(E_res, color='0.35', lw=0.6, ls='--')
    ax_xas.text(E_res + 0.06, 0.96, r'$E_{\rm res}$', fontsize=7, color='0.2',
                va='top')
    ax_xas.set_xlabel('Incident energy (eV)')
    ax_xas.set_ylabel('XAS intensity (norm.)')
    ax_xas.set_xlim(512.0, 517.0)
    ax_xas.set_ylim(0.0, 1.08)
    ax_xas.minorticks_on()
    ax_xas.legend(loc='upper left', frameon=False, handlelength=1.8)
    label(ax_xas, 'a')

    # --- Panel b: RIXS ---
    if norm_window_meV is not None:
        w0, w1 = norm_window_meV
        nmask = (eloss_meV >= w0) & (eloss_meV <= w1)
    else:
        vis_mask = (eloss_meV >= xlim_meV[0]) & (eloss_meV <= xlim_meV[1])
        nmask    = vis_mask & (np.abs(eloss_meV) > 15.0)
    if nmask.sum() > 0:
        norm_R = max(np.max((I_LV_r + I_LH_r)[nmask]),
                     np.max((I_cons_r + I_flip_r)[nmask]), 1e-30)
    else:
        norm_R = max(np.max(I_LV_r + I_LH_r), 1e-30)

    ax_rixs.fill_between(eloss_meV, 0.0, I_LV_r / norm_R,
                         color=C_LV, alpha=0.10, linewidth=0)
    ax_rixs.fill_between(eloss_meV, 0.0, I_LH_r / norm_R,
                         color=C_LH, alpha=0.10, linewidth=0)
    ax_rixs.plot(eloss_meV, I_LV_r / norm_R, color=C_LV, lw=1.1,
                 label=r'LV in, iso out')
    ax_rixs.plot(eloss_meV, I_LH_r / norm_R, color=C_LH, lw=1.1,
                 label=r'LH in, iso out')
    ax_rixs.plot(eloss_meV, I_cons_r / norm_R, color=C_CONS, lw=1.4,
                 label=r'conserving ($\sigma\sigma{+}\pi\pi$)')
    ax_rixs.plot(eloss_meV, I_flip_r / norm_R, color=C_FLIP, lw=1.4, ls='--',
                 dashes=(4, 1.8),
                 label=r'spin-flip ($\sigma\pi{+}\pi\sigma$)')

    win_mask = (eloss_sticks_meV > xlim_meV[0]) & (eloss_sticks_meV < xlim_meV[1])
    e_w      = eloss_sticks_meV[win_mask]
    s_cons_w = s_cons[win_mask]
    s_flip_w = s_flip[win_mask]
    y_curve_max = 1.10
    depth_c = 0.16 * y_curve_max
    depth_f = 0.16 * y_curve_max
    y0_cons = 0.0
    y0_flip = -depth_c - 0.02
    y_bot   = y0_flip - depth_f - 0.01
    _draw_sticks_below(ax_rixs, e_w, s_cons_w, color=C_CONS,
                       y0=y0_cons, depth=depth_c, thresh=1e-3, lw=1.0)
    _draw_sticks_below(ax_rixs, e_w, s_flip_w, color=C_FLIP,
                       y0=y0_flip, depth=depth_f, thresh=1e-3, lw=1.0)
    ax_rixs.axhline(y0_cons, color='0.75', lw=0.4)
    ax_rixs.axhline(y0_flip, color='0.85', lw=0.35, ls=':')
    x_lbl = xlim_meV[0] + 0.02 * (xlim_meV[1] - xlim_meV[0])
    ax_rixs.text(x_lbl, y0_cons - depth_c * 0.55, 'conserving',
                 fontsize=6, color=C_CONS, ha='left', va='center',
                 bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.85))
    ax_rixs.text(x_lbl, y0_flip - depth_f * 0.55, 'spin-flip',
                 fontsize=6, color=C_FLIP, ha='left', va='center',
                 bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.85))

    if show_J_marker and xlim_meV[1] < 500:
        ax_rixs.axvline(E_rel[1], color=C_ACC, lw=0.6, ls=':', alpha=0.8)
        ax_rixs.annotate(f'$J$ = {E_rel[1]:.2f} meV',
                         xy=(E_rel[1], y_curve_max * 0.55),
                         xytext=(E_rel[1] + 0.10 * (xlim_meV[1] - xlim_meV[0]),
                                 y_curve_max * 0.78),
                         fontsize=7.5, color='0.12',
                         arrowprops=dict(arrowstyle='-', color=C_ACC,
                                         lw=0.6, alpha=0.7))

    ax_rixs.set_xlabel('Energy loss (meV)')
    ax_rixs.set_ylabel('RIXS intensity (norm.)')
    ax_rixs.set_xlim(*xlim_meV)
    ax_rixs.set_ylim(y_bot, y_curve_max * 1.12)
    ax_rixs.minorticks_on()
    ax_rixs.set_yticks([v for v in ax_rixs.get_yticks() if v >= 0.0])
    ax_rixs.legend(loc='upper right', frameon=False, ncol=1,
                   handlelength=1.8, borderaxespad=0.3, fontsize=6.5)
    label(ax_rixs, 'b')

    # --- Panel c: 2D map ---
    ymin, ymax = ylim_map_meV
    disp_mask  = (eloss_meV > ymin - 1.0) & (eloss_meV < ymax + 1.0)
    vmax       = np.percentile(I2d_s[:, disp_mask], 99.2)
    pcm        = ax_map.pcolormesh(
        ominc_scan, eloss_meV[disp_mask], I2d_s[:, disp_mask].T,
        cmap='inferno', norm=PowerNorm(gamma=0.40, vmin=0, vmax=vmax),
        shading='auto', rasterized=True,
    )
    ax_map.axvline(E_res, color='w', lw=0.8, ls='--', alpha=0.8)
    ax_map.text(E_res + 0.03, ymax - 0.03 * (ymax - ymin),
                r'$E_{\rm res}$', fontsize=7.5, color='w', va='top')

    I_sum_r   = I_LV_r + I_LH_r
    pk_mask   = (eloss_meV > max(20.0, ymin + 5.0)) & (eloss_meV < ymax - 1.0)
    I_masked  = np.where(pk_mask, I_sum_r, 0.0)
    pks, props = find_peaks(I_masked, height=np.max(I_masked) * 0.03,
                            distance=30)
    peak_meV  = eloss_meV[pks]
    peak_h    = props['peak_heights']
    order     = np.argsort(-peak_h)
    top_peaks = peak_meV[order][:3]
    markers   = []
    if ymin < E_rel[1] < ymax:
        markers.append((E_rel[1], r'$J$'))
    for i, E_pk in enumerate(top_peaks):
        markers.append((float(E_pk), rf'SOC$_{{{i+1}}}$'))
    for E_pk, lab in markers:
        ax_map.axhline(E_pk, color='w', lw=0.45, ls=':', alpha=0.6)
        ax_map.text(ominc_scan[-1] - 0.05, E_pk + 0.008 * (ymax - ymin),
                    lab, fontsize=7.5, color='w', ha='right', va='bottom',
                    fontweight='bold')

    ax_map.set_xlabel('Incident energy (eV)')
    ax_map.set_ylabel('Energy loss (meV)')
    ax_map.set_xlim(ominc_scan[0], ominc_scan[-1])
    ax_map.set_ylim(ymin, ymax)
    ax_map.minorticks_on()
    cb = fig.colorbar(pcm, ax=ax_map, fraction=0.022, pad=0.012, aspect=28)
    cb.set_label('RIXS intensity (arb.)', fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, width=0.5, length=2.0)
    cb.outline.set_linewidth(0.6)
    label(ax_map, 'c', dark_bg=True)

    for ext in ('pdf', 'png'):
        p = f'Figures/{fname_base}.{ext}'
        fig.savefig(p, dpi=600 if ext == 'pdf' else 400)
        print(f'  saved {p}')
    plt.close(fig)


# Low-E view (magnon + SOC d-d): 0–100 meV
make_figure(xlim_meV=(-15.0, 100.0),
            ylim_map_meV=(-15.0, 100.0),
            fname_base='fig_dimer_anderson',
            tag='low-E')

# High-E view: extend to 6 eV to reveal d-d multiplets + charge-transfer
# satellites that are only present in the Anderson model.
make_figure(xlim_meV=(-150.0, 6000.0),
            ylim_map_meV=(-150.0, 6000.0),
            fname_base='fig_dimer_anderson_highE',
            tag='high-E',
            norm_window_meV=(1500.0, 6000.0),
            show_J_marker=False)


# ══════════════════════════════════════════════════════════════════════════════
#  STATE-ANALYSIS FIGURE
#  3-panel plot (cf. He et al. 2024, CrI3 Fig. 4 / notebook cells 17 & 27):
#    a) <S²_tot>, <S²_A>+<S²_B>  vs energy loss  → magnon & local-spin character
#    b) <S_z_tot>                vs energy loss  → total-m_s quantum number
#    c) <N_A>, <N_B>, L-hole     vs energy loss  → d-d vs charge-transfer content
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating state-analysis figure ...")
E_cut_eV = 6.0
sel = E_rel_ana / 1000.0 <= E_cut_eV
E_sel = E_rel_ana[sel] / 1000.0     # eV for this plot
fig, axs = plt.subplots(3, 1, figsize=(COL2, 5.8), sharex=True,
                        gridspec_kw=dict(left=0.09, right=0.98, top=0.97,
                                         bottom=0.08, hspace=0.12))
# stick-plot style: scatter eigenstate points over a faint spectrum backdrop
def _scatter(ax, y, color, lbl, marker='o', ms=3.0, alpha=0.85):
    ax.scatter(E_sel, y[sel], s=ms**2, facecolor=color, edgecolor='none',
               alpha=alpha, label=lbl)

ax = axs[0]
_scatter(ax, S2T_exp,            C_CONS, r'$\langle S^{2}_{\rm tot}\rangle$')
_scatter(ax, S2A_exp + S2B_exp,  C_LV,   r'$\langle S^{2}_A\rangle + \langle S^{2}_B\rangle$',
         marker='s')
for y_ref, txt in [(0.0, 'singlet'), (0.75, r'$s_A{\cdot}s_B=3/4$'),
                   (1.5, r'$\langle S^{2}_A+S^{2}_B\rangle=3/2$'),
                   (2.0, 'triplet')]:
    ax.axhline(y_ref, color='0.80', lw=0.4, ls=':')
ax.set_ylabel(r'$\langle S^{2}\rangle$')
ax.set_ylim(-0.2, 6.5)
ax.legend(loc='upper right', frameon=False, ncol=2, fontsize=6.8,
          handlelength=1.4)
ax.minorticks_on()
label(ax, 'a')

ax = axs[1]
_scatter(ax, SzT_exp, C_CONS, r'$\langle S^{z}_{\rm tot}\rangle$')
ax.axhline(0, color='0.80', lw=0.4, ls=':')
ax.set_ylabel(r'$\langle S^{z}_{\rm tot}\rangle$')
ax.set_ylim(-1.3, 1.3)
ax.legend(loc='upper right', frameon=False, fontsize=6.8, handlelength=1.4)
ax.minorticks_on()
label(ax, 'b')

ax = axs[2]
_scatter(ax, N_A_exp,                C_LV,   r'$\langle N_A\rangle$')
_scatter(ax, N_B_exp,                C_LH,   r'$\langle N_B\rangle$', marker='s')
_scatter(ax, L_hole_exp,             C_ACC,  r'$\langle n_L\rangle = 2-\langle N_{\rm bath}\rangle$',
         marker='^')
for y_ref in (0.0, 1.0, 2.0):
    ax.axhline(y_ref, color='0.80', lw=0.4, ls=':')
ax.set_xlabel('Energy loss (eV)')
ax.set_ylabel('Occupation (electrons)')
ax.set_ylim(-0.05, 2.2)
ax.set_xlim(-0.1, E_cut_eV)
ax.legend(loc='upper right', frameon=False, fontsize=6.8, handlelength=1.4)
ax.minorticks_on()
label(ax, 'c')

for ext in ('pdf', 'png'):
    p = f'Figures/fig_dimer_anderson_state_analysis.{ext}'
    fig.savefig(p, dpi=600 if ext == 'pdf' else 400)
    print(f'  saved {p}')
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  TICAM RESULTS")
print("═" * 72)
print(f"  Singlet–triplet gap J       : {E_rel[1]:.3f} meV")
print(f"  Ground-state covalency |α|² : {L_fraction*100:.2f} %  "
      f"(fraction of |d³L⁻¹⟩ in GS)")
print(f"  Main L₃ resonance (absolute): {E_res:.3f} eV")
print(f"  Anderson bath parameters    : Δ_CT={Delta_CT:.2f} eV, V_pd={V_pd:.3f} eV")
print(f"  Hilbert size (initial)      : {ncfgs_i}")
print(f"  Hilbert size (intermediate) : {ncfgs_n} per core-hole site")
print("═" * 72)
