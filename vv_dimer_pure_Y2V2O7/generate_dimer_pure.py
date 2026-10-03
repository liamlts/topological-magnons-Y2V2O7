"""
generate_dimer_pure.py — Pure V-V dimer RIXS (no phenomenological peaks).

Follows the edrixs two-site cluster example closely:
  examples/more/RIXS/Ba3InIr2O9/t2g_two_site_cluster/{run_ed.py, run_rixs.py}

Adapted to V4+ (d1 per site) at the L3 edge (~515 eV) with Y2V2O7-like
parameters:
  * trigonal D3d crystal field within the t2g manifold (Δ_trig ≈ 30 meV)
  * 3d spin-orbit coupling (ζ_d ≈ 30 meV)
  * Hubbard U on V (Ud = 3 eV, JH = 0.68 eV)
  * diagonal t2g–t2g hopping between the two V sites (t ≈ 77 meV),
    giving superexchange J ≈ 4t²/U ≈ 8 meV — matches Y2V2O7 magnon bandwidth.
  * Full Slater 2p–3d Coulomb plus static core-hole potential.

Hilbert space (t2g + 2p projection per site, 2 sites, 24 spin-orbitals):
  Initial state:            C(12, 2) × C(12, 12) = 66 configurations
  Intermediate (hole on s): C(12, 3) × C(6, 5) × C(6, 6) = 1320

Outputs  Figures/fig_dimer_pure.pdf / .png.
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
import edrixs

os.makedirs('Figures', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
#  Plot style
# ──────────────────────────────────────────────────────────────────────────────
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
# Publication palette: colour-blind friendly
C_LV    = '#1f4e9c'    # LV (σ) — dark blue
C_LH    = '#c93a3a'    # LH (π) — red
C_CONS  = '#111111'    # conserving — near-black
C_FLIP  = '#1f7a3e'    # spin-flip — green
C_ACC   = '#6a3d9a'    # accent (feature markers)


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
# Y2V2O7: V4+ 3d1 on the pyrochlore B-site. 10Dq ≈ 1.9 eV pushes eg far above
# t2g, so we truncate to the t2g manifold for the low-energy RIXS calculation.
zeta_d_i   = 0.030    # eV   3d spin-orbit coupling (V)
delta_trig = 0.010    # eV   trigonal D3d splitting within t2g (Y2V2O7: weak)
U_d        = 3.0      # eV   Hubbard U on V
J_H        = 0.68     # eV   Hund coupling on V 3d
t_hop      = 0.077    # eV   inter-site diagonal t2g–t2g hopping
                       #      (J_sup ≈ 4t²/U ≈ 8.0 meV — matches Y2V2O7)
core_v     = 4.0      # eV   static 2p-core → 3d attractive potential

F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(U_d, J_H)

atom   = edrixs.utils.get_atom_data('V', '3d', 1, edge='L3')
_zp    = atom['c_soc']
zeta_p = float(_zp[0] if hasattr(_zp, '__len__') else _zp)
_gc    = atom['gamma_c']
gamma_c = float(_gc[0] if hasattr(_gc, '__len__') else _gc)      # HWHM
sn     = dict(atom['slater_n'])
F2_pd  = sn.get('F2_12', 6.759) * 0.8
G1_pd  = sn.get('G1_12', 5.014) * 0.8
G3_pd  = sn.get('G3_12', 2.853) * 0.8
F0_pd  = edrixs.get_F0('dp', G1_pd, G3_pd)

# Geometry: Diamond I21, 2θ ≈ 150° horizontal scattering
thin  = np.radians(15.0)
thout = np.radians(135.0)
phi   = 0.0

# Broadenings / temperature
gamma_f   = 0.002                                    # eV, RIXS final-state HWHM
res_FWHM  = 0.030                                    # eV, I21 resolution FWHM
sigma_res = res_FWHM / (2 * np.sqrt(2 * np.log(2)))
T_K       = 10.0

print(f"V4+ dimer  zeta_d={zeta_d_i*1e3:.0f} meV  Δ_trig={delta_trig*1e3:.0f} meV  "
      f"U={U_d:.1f} eV  JH={J_H:.2f} eV  t={t_hop*1e3:.0f} meV")
print(f"p-d:   F0={F0_pd:.3f}  F2={F2_pd:.3f}  G1={G1_pd:.3f}  G3={G3_pd:.3f} eV")
print(f"ζ_p = {zeta_p:.3f} eV,  Γ_c = {gamma_c:.3f} eV (HWHM)")
print(f"J_sup = 4t²/U = {4*t_hop**2/U_d*1e3:.2f} meV (expected dimer gap)")


# ══════════════════════════════════════════════════════════════════════════════
#  HAMILTONIAN  (follows Ba3InIr2O9/t2g_two_site_cluster/run_ed.py)
#  norbs = 24:  [0:6]  = t2g site A (spin-orbital; 2 spins × 3 t2g)
#               [6:12] = t2g site B
#               [12:18]= 2p  site A
#               [18:24]= 2p  site B
# ══════════════════════════════════════════════════════════════════════════════
norbs = 24

# ── Intra-site Coulomb tensors, transformed to real cubic basis ──────────────
umat_t2g_c = edrixs.get_umat_slater('t2g', F0_d, F2_d, F4_d)
umat_t2g   = edrixs.transform_utensor(umat_t2g_c, edrixs.tmat_c2r('t2g', True))

params_t2gp = [0.0, 0.0, 0.0,     # F^k d-d   (already in umat_t2g above)
               F0_pd, F2_pd,      # F^k p-d
               G1_pd, G3_pd,      # G^k p-d
               0.0, 0.0]          # F^k p-p
umat_t2gp_c = edrixs.get_umat_slater('t2gp', *params_t2gp)
t12 = np.zeros((12, 12), dtype=complex)
t12[0:6, 0:6]   = edrixs.tmat_c2r('t2g', True)
t12[6:12, 6:12] = edrixs.tmat_c2r('p', True)
umat_t2gp = edrixs.transform_utensor(umat_t2gp_c, t12)

# ── Embed into full 24-orbital tensors ───────────────────────────────────────
umat_i = np.zeros((norbs,) * 4, dtype=complex)
umat_n = np.zeros((2, norbs, norbs, norbs, norbs), dtype=complex)

umat_i[0:6,   0:6,   0:6,   0:6]   = umat_t2g
umat_i[6:12,  6:12,  6:12,  6:12]  = umat_t2g
for s in range(2):
    umat_n[s] += umat_i

indx = np.array([[0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17],     # site A
                 [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]])  # site B
print("Embedding 2p–3d Slater integrals ...")
for s in range(2):
    for i in range(12):
        for j in range(12):
            for k in range(12):
                for l in range(12):
                    v = umat_t2gp[i, j, k, l]
                    if v != 0.0:
                        umat_n[s, indx[s, i], indx[s, j],
                                  indx[s, k], indx[s, l]] += v

# ── One-body Hamiltonian: SOC + trigonal CF + inter-site hopping ─────────────
emat_i = np.zeros((norbs, norbs), dtype=complex)
emat_n = np.zeros((2, norbs, norbs), dtype=complex)

soc_d = edrixs.cb_op(edrixs.atom_hsoc('t2g', zeta_d_i), edrixs.tmat_c2r('t2g', True))
soc_p = edrixs.cb_op(edrixs.atom_hsoc('p',   zeta_p),   edrixs.tmat_c2r('p',   True))
emat_i[0:6,  0:6 ] += soc_d
emat_i[6:12, 6:12] += soc_d
for s in range(2):
    emat_n[s, 0:6,   0:6  ] += soc_d
    emat_n[s, 6:12,  6:12 ] += soc_d
emat_n[0, 12:18, 12:18] += soc_p
emat_n[1, 18:24, 18:24] += soc_p

# Trigonal D3d splitting in the real cubic t2g basis {dxy, dyz, dzx}:
#   H_trig = (Δ/3)(J₃ − I₃)    →   +2Δ/3 (a1g),  −Δ/3 (eg' doublet)
# Basis order after tmat_c2r('t2g', True): [orb_0↑, orb_0↓, orb_1↑, orb_1↓, orb_2↑, orb_2↓]
H_trig3 = (delta_trig / 3.0) * (np.ones((3, 3)) - np.eye(3))
H_trig6 = np.zeros((6, 6), dtype=complex)
for i in range(3):
    for j in range(3):
        H_trig6[2*i,     2*j    ] = H_trig3[i, j]
        H_trig6[2*i + 1, 2*j + 1] = H_trig3[i, j]
emat_i[0:6,  0:6 ] += H_trig6
emat_i[6:12, 6:12] += H_trig6
for s in range(2):
    emat_n[s, 0:6,  0:6 ] += H_trig6
    emat_n[s, 6:12, 6:12] += H_trig6

# Diagonal inter-site t2g–t2g hopping: each orbital on A hops to the same
# orbital on B with amplitude -t (spin conserving).  Sufficient for a single
# superexchange scale J ≈ 4t²/U.
for i in range(6):
    emat_i[i,     6 + i] += -t_hop
    emat_i[6 + i, i    ] += -t_hop
    for s in range(2):
        emat_n[s, i,     6 + i] += -t_hop
        emat_n[s, 6 + i, i    ] += -t_hop

# Static core-hole potential on the d orbitals of the core-hole site
for i in range(0, 6):
    emat_n[0, i, i] -= core_v
for i in range(6, 12):
    emat_n[1, i, i] -= core_v


# ══════════════════════════════════════════════════════════════════════════════
#  FOCK BASIS & DIAGONALIZATION
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Fock bases ...")
basis_i = edrixs.get_fock_bin_by_N(12, 2, 12, 12)
basis_n = [edrixs.get_fock_bin_by_N(12, 3, 6, 5, 6, 6),
           edrixs.get_fock_bin_by_N(12, 3, 6, 6, 6, 5)]
ncfgs_i = len(basis_i)
ncfgs_n = len(basis_n[0])
print(f"  Initial: {ncfgs_i}   Intermediate: {ncfgs_n} per site")

print("Building H_i ...")
t0 = time.time()
hmat_i  = edrixs.two_fermion(emat_i, basis_i, basis_i)
hmat_i += edrixs.four_fermion(umat_i, basis_i)
eval_i, evec_i = scipy.linalg.eigh(hmat_i)
del hmat_i
E_rel = (eval_i - eval_i[0]) * 1000.0
print(f"  done in {time.time() - t0:.1f} s")
print(f"  lowest levels (meV above GS): {np.round(E_rel[:8], 3)}")
print(f"  → singlet–triplet gap ≈ {E_rel[1]:.2f} meV")

print("Building H_n per core-hole site ...")
eval_n = np.zeros((2, ncfgs_n))
evec_n = np.zeros((2, ncfgs_n, ncfgs_n), dtype=complex)
for s in range(2):
    t0 = time.time()
    hmat  = edrixs.two_fermion(emat_n[s], basis_n[s], basis_n[s])
    hmat += edrixs.four_fermion(umat_n[s], basis_n[s])
    eval_n[s], evec_n[s] = scipy.linalg.eigh(hmat)
    del hmat
    print(f"  site {s}: {time.time() - t0:.1f} s")


# ══════════════════════════════════════════════════════════════════════════════
#  TRANSITION OPERATORS  (t2gp dipole, Cartesian x,y,z)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding transition operators ...")
tran_ptod = edrixs.get_trans_oper('t2gp')               # (3, 6, 6), Cartesian
for i in range(3):
    tran_ptod[i] = edrixs.cb_op2(tran_ptod[i],
                                 edrixs.tmat_c2r('t2g', True),
                                 edrixs.tmat_c2r('p',   True))

dipole_n = np.zeros((2, 3, norbs, norbs), dtype=complex)
dipole_n[0, :, 0:6,  12:18] = tran_ptod
dipole_n[1, :, 6:12, 18:24] = tran_ptod

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
#  CARTESIAN POLARIZATION VECTORS (LV, LH) for XAS and RIXS
# ══════════════════════════════════════════════════════════════════════════════
# Horizontal scattering plane = xz, sample surface = xy.
# Incoming beam ki at grazing angle thin below surface:
ki_hat   = np.array([ np.cos(thin),  0.0, -np.sin(thin)])
# Outgoing beam kout: 2θ = thin + thout measured on the same side
kout_hat = np.array([ np.cos(np.pi - thout), 0.0, np.sin(thout)])
# LV = σ: perpendicular to scattering plane (lab ŷ) — same for in and out
e_LV      = np.array([0.0, 1.0, 0.0])
# LH = π: in the scattering plane, perpendicular to k
_cross_in  = np.cross(ki_hat,   e_LV);  e_LH_in  = _cross_in  / np.linalg.norm(_cross_in)
_cross_out = np.cross(kout_hat, e_LV);  e_LH_out = _cross_out / np.linalg.norm(_cross_out)
print(f"\ne_LV      = {e_LV}")
print(f"e_LH_in   = {np.round(e_LH_in, 3)}")
print(f"e_LH_out  = {np.round(e_LH_out, 3)}")


def xas_polarized(omi_rel_vals, e_in):
    """XAS intensity with incoming Cartesian polarization e_in (3-vector)."""
    x = np.zeros(len(omi_rel_vals))
    for s in range(2):
        for ig, igs in enumerate(gs_list):
            amp = sum(e_in[j] * trop_n[s, j][:, igs] for j in range(3))   # (ncfgs_n,)
            amp2 = np.abs(amp) ** 2
            dE   = eval_n[s] - eval_i[igs]
            diff = omi_rel_vals[:, None] - dE[None, :]
            L    = gamma_c / (diff ** 2 + gamma_c ** 2) / np.pi
            x += gs_dist[ig] * (L @ amp2)
    return x


# ── XAS ──────────────────────────────────────────────────────────────────────
print("\nComputing XAS (LV, LH, isotropic) ...")
ominc_rel = np.linspace(-12.0, 8.0, 1200)   # covers 512–529 eV absolute
xas_LV    = xas_polarized(ominc_rel, e_LV)
xas_LH    = xas_polarized(ominc_rel, e_LH_in)
xas_iso   = (xas_LV + xas_LH + xas_polarized(ominc_rel, np.array([1.0, 0.0, 0.0]))) / 3.0

dE_rel       = float(ominc_rel[1] - ominc_rel[0])
xas_LV_conv  = gauss_convolve(xas_LV,  sigma_res, dE_rel)
xas_LH_conv  = gauss_convolve(xas_LH,  sigma_res, dE_rel)
xas_iso_conv = gauss_convolve(xas_iso, sigma_res, dE_rel)

omi_res_rel = ominc_rel[np.argmax(xas_iso_conv)]
L3_edge_eV  = 515.5
omi_shift   = L3_edge_eV - omi_res_rel
ominc_xas   = ominc_rel + omi_shift
E_res       = omi_res_rel + omi_shift
print(f"  XAS peak at ω_rel = {omi_res_rel:.3f} eV  →  E_res = {E_res:.3f} eV")


# ══════════════════════════════════════════════════════════════════════════════
#  RIXS  (Kramers-Heisenberg via edrixs.scattering_mat, Cartesian)
# ══════════════════════════════════════════════════════════════════════════════
# Single eloss grid from below-elastic out to 5 eV so the same calculation
# feeds both the low-energy (magnon + SOC) and high-energy (d–d multiplet)
# views.  Uniform 0.5 meV step.
eloss_eV  = np.linspace(-0.030, 5.000, 10060)
eloss_meV = eloss_eV * 1000.0


def ffg_cartesian(omi_rel):
    """F[j_emi, k_abs, f, g] via KH sum over intermediate states, summed over sites."""
    F = np.zeros((3, 3, ncfgs_i, len(gs_list)), dtype=complex)
    for s in range(2):
        abs_trans = trop_n[s][:, :, gs_list]
        emi_trans = np.conj(np.transpose(trop_n[s], (0, 2, 1)))
        F += edrixs.scattering_mat(
            eval_i, eval_n[s], abs_trans, emi_trans, omi_rel, gamma_c
        )
    return F


def rixs_channel(F_tot, e_in, e_out):
    """Single polarization channel (e_in, e_out): return broadened spectrum I(eloss)
    and stick amplitudes (|F|² summed over ground states)."""
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


# ── RIXS at resonance: 4 polarization channels ──────────────────────────────
print("\nComputing RIXS at resonance (σσ, σπ, πσ, ππ channels) ...")
t0 = time.time()
F_res = ffg_cartesian(omi_res_rel)                # (3, 3, ncfgs_i, n_gs)

I_ss, s_ss = rixs_channel(F_res, e_LV,    e_LV)       # σσ  (LV→LV) conserving
I_sp, s_sp = rixs_channel(F_res, e_LV,    e_LH_out)   # σπ  (LV→LH) spin-flip
I_ps, s_ps = rixs_channel(F_res, e_LH_in, e_LV)       # πσ  (LH→LV) spin-flip
I_pp, s_pp = rixs_channel(F_res, e_LH_in, e_LH_out)   # ππ  (LH→LH) conserving

I_cons  = I_ss + I_pp                                  # conserving total
I_flip  = I_sp + I_ps                                  # spin-flip total
I_LV_iso = I_ss + I_sp                                 # LV in, iso out
I_LH_iso = I_ps + I_pp                                 # LH in, iso out
I_all    = I_cons + I_flip

s_cons  = s_ss + s_pp
s_flip  = s_sp + s_ps
s_LV    = s_ss + s_sp
s_LH    = s_ps + s_pp
print(f"  done in {time.time() - t0:.1f} s")

# Resolution-broaden for display
dE_el       = float(eloss_eV[1] - eloss_eV[0])
I_cons_r    = gauss_convolve(I_cons,   sigma_res, dE_el)
I_flip_r    = gauss_convolve(I_flip,   sigma_res, dE_el)
I_LV_r      = gauss_convolve(I_LV_iso, sigma_res, dE_el)
I_LH_r      = gauss_convolve(I_LH_iso, sigma_res, dE_el)
I_all_r     = gauss_convolve(I_all,    sigma_res, dE_el)
eloss_sticks_meV = (eval_i - eval_i[gs_list[0]]) * 1000.0


# ── 2D RIXS map: 512.5 → 516.5 eV incident energy ───────────────────────────
MAP_EMIN, MAP_EMAX = 512.5, 516.5
n_map_pts      = 60
ominc_scan     = np.linspace(MAP_EMIN, MAP_EMAX, n_map_pts)
ominc_scan_rel = ominc_scan - omi_shift
print(f"\nComputing 2D RIXS map ({MAP_EMIN:.1f} → {MAP_EMAX:.1f} eV, "
      f"{n_map_pts} incident energies, iso in, iso out) ...")
t0 = time.time()
I2d = np.zeros((n_map_pts, len(eloss_eV)))
for io, om in enumerate(ominc_scan_rel):
    F_om = ffg_cartesian(om)
    for e_in in (e_LV, e_LH_in):
        for e_out in (e_LV, e_LH_out):
            I_ch, _ = rixs_channel(F_om, e_in, e_out)
            I2d[io] += I_ch
print(f"  done in {time.time() - t0:.1f} s   peak = {np.max(I2d):.3e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE(S) — two versions: low-E (up to ~200 meV) and high-E (up to 5 eV)
# ══════════════════════════════════════════════════════════════════════════════
from scipy.ndimage import convolve1d
from scipy.signal  import find_peaks

# Map smoothing along incident-energy axis (shared by both figures)
sig_inc  = 0.08
dE_inc   = float(ominc_scan[1] - ominc_scan[0])
hw_inc   = int(4 * sig_inc / dE_inc) + 1
ki       = np.arange(-hw_inc, hw_inc + 1) * dE_inc
kern_inc = np.exp(-0.5 * (ki / sig_inc) ** 2); kern_inc /= kern_inc.sum()
I2d_s    = convolve1d(I2d, kern_inc, axis=0, mode='nearest')

# Panel-b normalisation is computed inside make_figure over the visible window
# so both low-E and high-E figures have useful dynamic range (elastic peak is
# always ~2 orders above everything; clip it if it would wash out features).


def _draw_sticks_below(ax, e_meV, amp, color, y0, depth, thresh,
                       lw=0.6, skip_elastic=True):
    """Hang sticks downward from baseline y0 to y0-depth*(amp/amax).
    Normalisation uses only non-elastic amplitudes so inelastic sticks
    stay visible even when the elastic channel dominates."""
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
    """Build a publication-quality 3-panel figure (XAS, RIXS, 2D map)."""
    print(f"\nGenerating figure [{tag}]:  eloss range {xlim_meV} meV")
    fig = plt.figure(figsize=(COL2, 5.8))
    gs  = GridSpec(2, 2, hspace=0.42, wspace=0.26,
                   left=0.075, right=0.965, top=0.955, bottom=0.095,
                   height_ratios=[1.0, 1.05])
    ax_xas  = fig.add_subplot(gs[0, 0])
    ax_rixs = fig.add_subplot(gs[0, 1])
    ax_map  = fig.add_subplot(gs[1, :])

    # ─── Panel a: Polarized XAS 512–517 eV ───────────────────────────────────
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

    # ─── Panel b: RIXS at resonance: polarimetric + LV/LH-iso + sticks ───────
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

    # LV/LH-iso: faint filled area for context
    ax_rixs.fill_between(eloss_meV, 0.0, I_LV_r / norm_R,
                         color=C_LV, alpha=0.10, linewidth=0)
    ax_rixs.fill_between(eloss_meV, 0.0, I_LH_r / norm_R,
                         color=C_LH, alpha=0.10, linewidth=0)
    ax_rixs.plot(eloss_meV, I_LV_r / norm_R, color=C_LV, lw=1.1,
                 label=r'LV in, iso out')
    ax_rixs.plot(eloss_meV, I_LH_r / norm_R, color=C_LH, lw=1.1,
                 label=r'LH in, iso out')
    # Polarimetric (bold)
    ax_rixs.plot(eloss_meV, I_cons_r / norm_R, color=C_CONS, lw=1.4,
                 label=r'conserving ($\sigma\sigma{+}\pi\pi$)')
    ax_rixs.plot(eloss_meV, I_flip_r / norm_R, color=C_FLIP, lw=1.4, ls='--',
                 dashes=(4, 1.8),
                 label=r'spin-flip ($\sigma\pi{+}\pi\sigma$)')

    # Sticks below baseline
    win_mask = (eloss_sticks_meV > xlim_meV[0]) & (eloss_sticks_meV < xlim_meV[1])
    e_w      = eloss_sticks_meV[win_mask]
    s_cons_w = s_cons[win_mask]; s_flip_w = s_flip[win_mask]
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
    # stick labels on left
    x_lbl = xlim_meV[0] + 0.02 * (xlim_meV[1] - xlim_meV[0])
    ax_rixs.text(x_lbl, y0_cons - depth_c * 0.55,
                 'conserving', fontsize=6, color=C_CONS,
                 ha='left', va='center',
                 bbox=dict(fc='white', ec='none', pad=0.8, alpha=0.85))
    ax_rixs.text(x_lbl, y0_flip - depth_f * 0.55,
                 'spin-flip', fontsize=6, color=C_FLIP,
                 ha='left', va='center',
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
    # Hide negative tick labels (sticks region) for clean y-axis
    ax_rixs.set_yticks([v for v in ax_rixs.get_yticks() if v >= 0.0])
    ax_rixs.legend(loc='upper right', frameon=False, ncol=1,
                   handlelength=1.8, borderaxespad=0.3, fontsize=6.5)
    label(ax_rixs, 'b')

    # ─── Panel c: 2D RIXS map (incident vs eloss) ────────────────────────────
    ymin, ymax = ylim_map_meV
    disp_mask  = (eloss_meV > ymin - 1.0) & (eloss_meV < ymax + 1.0)
    vmax       = np.percentile(I2d_s[:, disp_mask], 99.2)
    pcm        = ax_map.pcolormesh(
        ominc_scan, eloss_meV[disp_mask], I2d_s[:, disp_mask].T,
        cmap='inferno', norm=PowerNorm(gamma=0.40, vmin=0, vmax=vmax),
        shading='auto', rasterized=True,
    )
    ax_map.axvline(E_res, color='w', lw=0.8, ls='--', alpha=0.8)
    ax_map.text(E_res + 0.03, ymax - 0.03 * (ymax - ymin), r'$E_{\rm res}$',
                fontsize=7.5, color='w', va='top', ha='left')

    # Feature markers: J + strongest inelastic peaks
    I_sum_r  = I_LV_r + I_LH_r
    pk_mask  = (eloss_meV > max(20.0, ymin + 5.0)) & (eloss_meV < ymax - 1.0)
    I_masked = np.where(pk_mask, I_sum_r, 0.0)
    pks, props = find_peaks(I_masked, height=np.max(I_masked) * 0.03,
                            distance=30)
    peak_meV = eloss_meV[pks]
    peak_h   = props['peak_heights']
    order    = np.argsort(-peak_h)
    top_peaks = peak_meV[order][:3]
    markers  = []
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
    # Colorbar
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


# Low-energy publication view: map and cuts zoomed to 100 meV
make_figure(xlim_meV=(-15.0, 100.0),
            ylim_map_meV=(-15.0, 100.0),
            fname_base='fig_dimer_pure',
            tag='low-E')

# High-energy view: full d-d multiplet structure up to 5 eV.
# Panel b normalised to the 1.5–5 eV window so d-d multiplets are resolved.
make_figure(xlim_meV=(-100.0, 5000.0),
            ylim_map_meV=(-100.0, 5000.0),
            fname_base='fig_dimer_pure_highE',
            tag='high-E',
            norm_window_meV=(1500.0, 5000.0),
            show_J_marker=False)

print("\nDone.")
print(f"  Dimer singlet–triplet gap (meV): {E_rel[1]:.3f}   (target 4t²/U ≈ {4*t_hop**2/U_d*1e3:.2f})")
print(f"  XAS peak (E_res): {E_res:.3f} eV")
print(f"  All features emerge from Kramers-Heisenberg + exact ED.")
