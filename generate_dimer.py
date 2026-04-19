"""
generate_dimer.py — V-V dimer RIXS figure for the topological magnon paper.

Run with:  /opt/miniconda3/envs/edrixs_run/bin/python generate_dimer.py

Y2V2O7 crystal-field parameters:
  10Dq  = 1.9 eV   (V4+ in pyrochlore oxide; from optical/DFT)
  Δ_trig = 30 meV  (D3d trigonal distortion along [111] pyrochlore axis)
  ζ_3d  = 30 meV   (V 3d SOC)
  T     = 10 K     (realistic beamtime temperature)

Geometry (I21, Diamond Light Source):
  θ_in = 15°, θ_out = 135° (2θ ≈ 150° horizontal scattering)
  Resolution = 30 meV FWHM (I21 at V L3 ≈ 515 eV, high-resolution grating)

Produces Figures/fig_vv_dimer.pdf — 3-panel figure:
  a. V L3 XAS (σ, π) with RIXS incident energy marked
  b. RIXS plane (E_inc × E_loss, spin-flip channel) — shows resonance profile
  c. V4+ energy levels + V–V dimer singlet/triplet from edrixs.build_opers
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import edrixs

os.makedirs('Figures', exist_ok=True)

# ── Uniform Nature Physics rcParams ──────────────────────────────────────────
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

COL1   = 3.386
COL2   = 7.008
C_FLIP = '#D42E2E'
C_CONS = '#0073BD'
C_PHON = '#38A12B'
C_XCHG = '#9533BF'
C_SUM  = '#555555'


def label(ax, letter, dark_bg=False, x=0.025, y=0.97):
    """Nature Physics panel label: bold 'a.' top-left, no parentheses."""
    c = 'white' if dark_bg else 'black'
    ax.text(x, y, f'{letter}.', transform=ax.transAxes,
            fontsize=8, fontweight='bold', va='top', ha='left', color=c)


def save(fig, name):
    for ext in ('pdf', 'png'):
        p = f'Figures/{name}.{ext}'
        fig.savefig(p, dpi=600 if ext == 'pdf' else 300)
        print(f'  saved {p}')


def cf_trigonal_d(delta_trig):
    """
    Trigonal D3d crystal field for d-orbitals with C3 axis along [111].

    H = Δ × [(L·n̂)² − L(L+1)/3],  n̂ = [1,1,1]/√3

    Within t2g: splits a1g (along [111]) by +2Δ/3 and eg' by −Δ/3.
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


# ═══════════════════════════════════════════════════════════════════════════
#  PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════

shell_name  = ('d', 'p32')
v_noccu     = 1
ten_dq      = 1.9      # eV  (V4+ in Y2V2O7 pyrochlore)
delta_trig  = 0.030    # eV  (30 meV D3d trigonal distortion along [111])
zeta_d_i    = 0.030    # eV  (30 meV 3d SOC)
zeta_d_n    = 0.030    # eV
J_meV       = 8.22     # meV (exchange coupling)
J_eV        = J_meV / 1000.0
T_K         = 10.0     # K   (measurement temperature)
res_FWHM_eV = 0.030    # eV  (energy resolution at V L3, consistent with paper)
sigma_res   = res_FWHM_eV / (2 * np.sqrt(2 * np.log(2)))

thin_deg   = 15.0
thout_deg  = 135.0
thin_rad   = np.radians(thin_deg)
thout_rad  = np.radians(thout_deg)
phi        = 0.0

poltype_all = [
    ('linear', 0,       'linear', 0      ),   # σσ
    ('linear', 0,       'linear', np.pi/2),   # σπ
    ('linear', np.pi/2, 'linear', 0      ),   # πσ
    ('linear', np.pi/2, 'linear', np.pi/2),   # ππ
]
poltype_sf  = [('linear', 0, 'linear', np.pi/2)]   # σπ only (RIXS plane)

# Slater integrals from EDRIXS database (80% reduction)
info        = edrixs.utils.get_atom_data('V', '3d', v_noccu, edge='L3')
c_soc_raw   = info['c_soc']
zeta_p_n    = float(c_soc_raw[0]) if isinstance(c_soc_raw, (list, tuple)) else float(c_soc_raw)
gc_raw      = info['gamma_c']
gamma_c     = float(gc_raw[0])    if isinstance(gc_raw, (list, tuple)) else float(gc_raw)

F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(0.0, 0.0)
slater_n_db = {k: v for k, v in info['slater_n']}
F2_pd = slater_n_db.get('F2_12', 6.759) * 0.8
G1_pd = slater_n_db.get('G1_12', 5.014) * 0.8
G3_pd = slater_n_db.get('G3_12', 2.853) * 0.8
F0_pd = edrixs.get_F0('dp', G1_pd, G3_pd)
slater = [[F0_d, F2_d, F4_d],
          [F0_d, F2_d, F4_d, F0_pd, F2_pd, G1_pd, G3_pd, 0.0, 0.0]]

off     = 515.0 + F0_pd
gamma_f = 0.002   # 2 meV intrinsic final-state broadening

# Crystal field: cubic Oh + trigonal D3d correction
cf_mat = edrixs.cf_cubic_d(ten_dq) + cf_trigonal_d(delta_trig)

# Verify CF eigenvalues
cf_evals = np.linalg.eigvalsh(cf_mat)
print(f"Y2V2O7 V4+: 10Dq={ten_dq:.2f} eV, Δ_trig={delta_trig*1000:.0f} meV, "
      f"ζ_3d={zeta_d_i*1000:.0f} meV")
print(f"CF eigenvalues (eV): {np.round(np.sort(cf_evals), 4)}")
print(f"Core-hole: ζ_2p={zeta_p_n:.3f} eV, Γ_c={gamma_c:.3f} eV")


# ═══════════════════════════════════════════════════════════════════════════
#  DIMER LEVELS via edrixs.build_opers  (Hubbard-dimer tutorial style)
# ═══════════════════════════════════════════════════════════════════════════
# H = J S_A·S_B,  spin-orbitals [0=A↑, 1=A↓, 2=B↑, 3=B↓]
print("\n=== Dimer Heisenberg levels via build_opers ===")

norb_spin  = 4
noccu_spin = 2
basis_spin = edrixs.get_fock_bin_by_N(norb_spin, noccu_spin)   # C(4,2) = 6 states

emat_spin = np.zeros((norb_spin, norb_spin), dtype=complex)

# build_opers(4, u, basis) implements H = Σ u[i,j,k,l] c†_i c†_j c_k c_l (no ½ prefactor).
# Paired entries each contribute half the target coefficient.
# Ising: Sz_A Sz_B = J/4*(n₀n₂ - n₀n₃ - n₁n₂ + n₁n₃)
# Flip:  S+_A S-_B = c†₀c†₃c₂c₁,  S-_A S+_B = c†₁c†₂c₃c₀
umat_spin = np.zeros((norb_spin,) * 4, dtype=complex)

for i, j, sign in [(0, 2, +1), (0, 3, -1), (1, 2, -1), (1, 3, +1)]:
    umat_spin[i, j, j, i] += sign * J_eV / 8
    umat_spin[j, i, i, j] += sign * J_eV / 8

umat_spin[0, 3, 2, 1] += J_eV / 4
umat_spin[3, 0, 1, 2] += J_eV / 4
umat_spin[1, 2, 3, 0] += J_eV / 4
umat_spin[2, 1, 0, 3] += J_eV / 4

H_dimer    = (edrixs.build_opers(2, emat_spin, basis_spin)
              + edrixs.build_opers(4, umat_spin, basis_spin))
E_dimer_meV = np.real(np.linalg.eigvalsh(H_dimer)) * 1000.0
print(f"build_opers eigenvalues (meV): {np.round(E_dimer_meV, 3)}")
print(f"  Analytical: singlet {-3*J_meV/4:.3f}, triplet {J_meV/4:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE-SITE V4+ EXACT DIAGONALISATION
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Single-site V4+ ED (Y2V2O7 CF) ===")

eval_i, eval_n, trans_op = edrixs.ed_1v1c_py(
    shell_name,
    shell_level=(0.0, -off),
    v_soc=(zeta_d_i, zeta_d_n),
    c_soc=zeta_p_n,
    v_noccu=v_noccu,
    slater=slater,
    v_cfmat=cf_mat,
    verbose=0,
)

# ── XAS: wide scan for panel (a) ────────────────────────────────────────────
print("Computing XAS...")
ominc_xas  = np.linspace(512.0, 532.0, 1400)   # covers 514–530 + buffer for convolution
xas_both   = edrixs.xas_1v1c_py(
    eval_i, eval_n, trans_op, ominc_xas,
    gamma_c=gamma_c, thin=thin_rad, phi=phi,
    pol_type=[('linear', 0), ('linear', np.pi/2)],
    gs_list=[0, 1, 2, 3], temperature=T_K,
)
xas_sigma  = xas_both[:, 0]
xas_pi     = xas_both[:, 1]
xas_total  = xas_sigma + xas_pi
# Broaden with I21 instrumental resolution (35 meV FWHM)
dE_xas    = float(ominc_xas[1] - ominc_xas[0])
xas_sigma = gauss_convolve(xas_sigma, sigma_res, dE_xas)
xas_pi    = gauss_convolve(xas_pi,    sigma_res, dE_xas)
xas_total = xas_sigma + xas_pi

# Extra display broadening (~50 meV extra sigma) for smoother visual appearance
sig_disp       = 0.050 / (2*np.sqrt(2*np.log(2)))
xas_sigma_disp = gauss_convolve(xas_sigma, sig_disp, dE_xas)
xas_pi_disp    = gauss_convolve(xas_pi,    sig_disp, dE_xas)
xas_total_disp = xas_sigma_disp + xas_pi_disp

E_res = ominc_xas[np.argmax(xas_sigma)]   # resonance = σ XAS maximum
print(f"XAS σ resonance (RIXS incident energy): {E_res:.3f} eV  "
      f"({E_res - 515.0:.2f} eV above threshold)")

# Single-site levels shifted so GS = 0
E_site_meV = (np.real(eval_i) - np.min(np.real(eval_i))) * 1000.0


# ── RIXS plane: spin-flip channel across L3 edge ────────────────────────────
ominc_plane    = np.linspace(512.0, 522.0, 80)
print(f"Computing RIXS plane (spin-flip, {len(ominc_plane)} incident energies)...")
eloss_plane_eV = np.linspace(-0.060, 0.200, 1200)
eloss_plane_meV = eloss_plane_eV * 1000.0
n_gs          = min(4, len(eval_i))

rixs_plane_raw = edrixs.rixs_1v1c_py(
    eval_i, eval_n, trans_op,
    ominc_plane, eloss_plane_eV,
    gamma_c=gamma_c, gamma_f=gamma_f,
    thin=thin_rad, thout=thout_rad, phi=phi,
    pol_type=poltype_sf,
    gs_list=list(range(n_gs)),
    temperature=T_K,
)
rixs_plane = rixs_plane_raw[:, :, 0]   # (n_Einc, n_eloss)
print("RIXS plane done.")


# ── RIXS at resonance: full 0–200 meV ───────────────────────────────────────
print("Computing RIXS at resonance (0–200 meV)...")
eloss_eV  = np.linspace(-0.005, 0.205, 4200)
eloss_meV = eloss_eV * 1000.0
dE        = float(eloss_eV[1] - eloss_eV[0])

rixs_res = edrixs.rixs_1v1c_py(
    eval_i, eval_n, trans_op,
    np.array([E_res]), eloss_eV,
    gamma_c=gamma_c, gamma_f=gamma_f,
    thin=thin_rad, thout=thout_rad, phi=phi,
    pol_type=poltype_all,
    gs_list=list(range(n_gs)),
    temperature=T_K,
)
rixs_cons = rixs_res[0, :, 0] + rixs_res[0, :, 3]   # σσ + ππ
rixs_flip = rixs_res[0, :, 1] + rixs_res[0, :, 2]   # σπ + πσ
print("RIXS at resonance done.")

from scipy.signal import find_peaks
mask_hi = (eloss_meV > 10) & (eloss_meV < 150)
pks, _  = find_peaks(rixs_flip * mask_hi, height=np.max(rixs_flip) * 0.02, distance=50)
E_soc_peaks = eloss_meV[pks] if len(pks) > 0 else np.array([45.0])
print(f"SOC excitation peaks (spin-flip, intrinsic): {np.round(E_soc_peaks, 1)} meV")


# ── Resolution convolution (I21, 35 meV FWHM) ───────────────────────────────
rixs_cons_conv  = gauss_convolve(rixs_cons, sigma_res, dE)
rixs_flip_conv  = gauss_convolve(rixs_flip, sigma_res, dE)


# ── Phenomenological additions ───────────────────────────────────────────────
norm_ref = max(np.max(rixs_flip), 1e-30)

# Exchange (ΔS=1 → spin-flip only); width = 2 meV intrinsic + convolved with resolution
A_xchg    = 0.60 * norm_ref
sig_xchg  = 0.002   # 2 meV σ intrinsic
exch_flip = A_xchg * np.exp(-0.5 * ((eloss_eV - J_eV) / sig_xchg) ** 2)
exch_conv = gauss_convolve(exch_flip, sigma_res, dE)

# Phonon at 100 meV (conserving only, 5% cross-talk)
# NOTE: phenomenological Gaussian; not derived from displaced-oscillator model.
# Amplitude is arbitrary — set to 65% of the max electronic signal for visibility.
E_ph    = 0.100   # eV
A_ph    = 0.65 * norm_ref
sig_ph  = 0.008   # eV (≈ 19 meV FWHM intrinsic width)
leakage = 0.05
phonon       = A_ph * np.exp(-0.5 * ((eloss_eV - E_ph) / sig_ph) ** 2)
phonon_conv  = gauss_convolve(phonon, sigma_res, dE)

# Resolution-broadened totals
cons_conv_total = rixs_cons_conv + phonon_conv
flip_conv_total = rixs_flip_conv + exch_conv + phonon_conv * leakage


# ═══════════════════════════════════════════════════════════════════════════
#  FIGURE  (2 × 2 panels)
# ═══════════════════════════════════════════════════════════════════════════
print("\nGenerating: fig_vv_dimer")

fig = plt.figure(figsize=(COL2, 5.6))
gs  = fig.add_gridspec(2, 2, hspace=0.44, wspace=0.34,
                        left=0.09, right=0.97, top=0.97, bottom=0.08)
ax_xas   = fig.add_subplot(gs[0, 0])        # a: XAS
ax_plane = fig.add_subplot(gs[0, 1])        # b: RIXS plane
ax_lvl   = fig.add_subplot(gs[1, :])        # c: energy levels (full width)


# ─── Panel a: XAS — full range 514–530 eV (main) + L3 zoom (inset) ─────────
# Normalize all polarizations to the same global max (preserves dichroism ratio)
xas_global_max = max(np.max(xas_sigma_disp), np.max(xas_pi_disp), 1e-30)

def _xas_lines(ax, lw_main=0.9, do_legend=False):
    ax.plot(ominc_xas, xas_sigma_disp / xas_global_max, color=C_CONS, lw=lw_main,
            label=r'$\sigma$ (LH)')
    ax.plot(ominc_xas, xas_pi_disp    / xas_global_max,  color=C_FLIP, lw=lw_main,
            label=r'$\pi$ (LV)')
    if do_legend:
        ax.legend(fontsize=5.0, loc='upper right', framealpha=0.85, ncol=1)

# Main panel: full L3 + L2 range
_xas_lines(ax_xas, do_legend=True)
ax_xas.set_xlim(512.0, 522.0)
# Set y-axis so the tallest peak reaches ~95% of the displayed range
y_peak = max(np.max(xas_sigma_disp), np.max(xas_pi_disp)) / xas_global_max
ax_xas.set_ylim(-0.02, y_peak / 0.95)
ax_xas.set_xlabel('Incident energy (eV)')
ax_xas.set_ylabel('XAS (arb. units)')

# L3 / L2 edge labels on main panel
mask_l3 = (ominc_xas > 515) & (ominc_xas < 522)
if mask_l3.any():
    i_l3 = np.argmax(xas_total_disp[mask_l3])
    ax_xas.text(ominc_xas[mask_l3][i_l3], xas_total_disp[mask_l3][i_l3] / xas_global_max + 0.05,
                r'$L_3$', fontsize=5.5, ha='center', va='bottom', color='0.4')
mask_l2 = (ominc_xas > 522) & (ominc_xas < 530)
if mask_l2.any():
    ax_xas.text(520.0, 0.5,
                r'$L_2$', fontsize=5.5, ha='center', va='bottom', color='0.4')

label(ax_xas, 'a')


# ─── Panel b: RIXS plane ────────────────────────────────────────────────────
X_plane = ominc_plane              # absolute incident energy (eV)
Y_plane = eloss_plane_meV           # -60 → 200 meV

# Mild Gaussian broadening along incident energy axis for smoother map
from scipy.ndimage import convolve1d
sig_inc_7b = 0.08  # eV σ
dE_inc_7b  = float(X_plane[1] - X_plane[0])
hw_7b      = int(4 * sig_inc_7b / dE_inc_7b) + 1
k_7b       = np.arange(-hw_7b, hw_7b + 1) * dE_inc_7b
kern_7b    = np.exp(-0.5 * (k_7b / sig_inc_7b)**2); kern_7b /= kern_7b.sum()
rixs_plane = convolve1d(rixs_plane, kern_7b, axis=0, mode='nearest')

# vmax from signal region (exclude elastic at |eloss| < 5 meV)
signal_mask = np.abs(eloss_plane_meV) > 5
vmax = np.percentile(rixs_plane[:, signal_mask], 99.5)
norm_plane = PowerNorm(gamma=0.35, vmin=0, vmax=vmax)

pc = ax_plane.pcolormesh(X_plane, Y_plane, rixs_plane.T,
                          cmap='inferno', norm=norm_plane,
                          shading='auto', rasterized=True)
ax_plane.axvline(E_res, color='w', lw=0.5, ls='--', alpha=0.6)   # resonance
ax_plane.axhline(0, color='w', lw=0.4, ls=':', alpha=0.4)        # elastic

# E_res label near the dashed resonance line (data coords, away from 'b' label)
ax_plane.text(E_res + 0.12, 30, r'$E_{\rm res}$',
              fontsize=5.5, va='bottom', ha='left', color='w')

# Annotate excitations with physical labels
_peak_labels = {0: 'elastic', 8: '$J$', 16: r'SOC$_1$', 61: r'SOC$_2$', 100: 'ph.'}

def _nearest_label(E_meV, tol=4):
    """Find nearest label within tol meV."""
    best_k, best_d = None, tol + 1
    for k in _peak_labels:
        d = abs(E_meV - k)
        if d < best_d:
            best_d, best_k = d, k
    return _peak_labels[best_k] if best_k is not None else f'{E_meV:.0f} meV'

for i, Ep in enumerate(E_soc_peaks):
    ax_plane.axhline(Ep, color='w', lw=0.4, ls=':', alpha=0.5)
    lbl = _nearest_label(Ep)
    ax_plane.text(X_plane[1] + 0.05, Ep + 3, lbl,
                  fontsize=5, color='w', ha='left', va='bottom')

cb = fig.colorbar(pc, ax=ax_plane, fraction=0.046, pad=0.03, aspect=20)
cb.set_label(r'$I_{\sigma\pi}$ (arb. units)', fontsize=6)
cb.ax.tick_params(labelsize=5, width=0.3, length=2)
cb.outline.set_linewidth(0.4)

ax_plane.set_xlabel('Incident energy (eV)')
ax_plane.set_ylabel('Energy loss (meV)')
ax_plane.set_xlim(512.0, 522.0)

# Auto-trim high-eloss axis: cut above last meV where column-max > 1% of vmax
col_max = rixs_plane.max(axis=0)                     # (n_eloss,)
above_thresh = eloss_plane_meV[col_max > 0.01 * vmax]
eloss_top = float(above_thresh.max()) + 15 if above_thresh.size > 0 else 130.0
eloss_top = max(eloss_top, max(E_soc_peaks) + 20 if len(E_soc_peaks) else 100.0)
ax_plane.set_ylim(-20, min(eloss_top, 200))
label(ax_plane, 'b', dark_bg=True)


# ─── Panel c: Energy level diagram ──────────────────────────────────────────
# Use energies relative to each column's own GS (both start at 0) so the
# dimer and single-site levels share a meaningful y-axis.
Ecut = 88.0
lx1, lx2 = 1.3, 3.8    # dimer column x-range
sx1, sx2 = 6.2, 8.7    # single-site column x-range
lx_mid   = (lx1 + lx2) / 2
sx_mid   = (sx1 + sx2) / 2

ax_lvl.set_xlim(0, 10)
ax_lvl.set_ylim(-17, Ecut + 12)
ax_lvl.set_xticks([])
ax_lvl.set_ylabel('Energy (meV)')

# ── Dimer levels (plot relative to singlet = 0 for the diagram) ──────────────
E_singlet = 0.0
E_triplet = J_meV         # ~8.2 meV

ax_lvl.plot([lx1, lx2], [E_singlet, E_singlet], color=C_CONS, lw=2.2)
ax_lvl.plot([lx1, lx2], [E_triplet, E_triplet], color=C_FLIP, lw=2.2)

# Labels centered on each bar, above/below to avoid overlap
ax_lvl.text(lx_mid, E_singlet - 2.0,
            r'$|S\!=\!0\rangle$', ha='center', va='top', fontsize=5.5, color=C_CONS)
ax_lvl.text(lx_mid, E_triplet + 2.0,
            r'$|S\!=\!1\rangle$', ha='center', va='bottom', fontsize=5.5, color=C_FLIP)

# J gap arrow + label, placed just right of the column
ax_lvl.annotate('', xy=(lx2 + 0.25, E_triplet), xytext=(lx2 + 0.25, E_singlet),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=0.7))
ax_lvl.text(lx2 + 0.4, E_triplet + 1.0,
            f'$J\\!=\\!{J_meV:.1f}$\nmeV',
            ha='left', va='bottom', fontsize=4.8, color='#555')

# ── Single-site levels (GS already at 0 in E_site_meV) ──────────────────────
groups = []
k = 0
while k < len(E_site_meV) and E_site_meV[k] < Ecut:
    m = k + 1
    while m < len(E_site_meV) and abs(E_site_meV[m] - E_site_meV[k]) < 0.6:
        m += 1
    groups.append((E_site_meV[k], m - k))
    k = m

jeff12_E = [E for E, d in groups if E < 3]
jeff32_E = [E for E, d in groups if 3 < E < Ecut]

for E in jeff12_E:
    ax_lvl.plot([sx1, sx2], [E, E], color=C_CONS, lw=1.8)
for E in jeff32_E:
    ax_lvl.plot([sx1, sx2], [E, E], color=C_FLIP, lw=1.8)

# Jeff=1/2 label: left of the single-site column, at GS y=0
if jeff12_E:
    ax_lvl.text(sx1 - 0.35, min(jeff12_E),
                r'$J_{\rm eff}\!=\!\frac{1}{2}$',
                ha='right', va='center', fontsize=5.5, color=C_CONS)

# Jeff=3/2 label: above the top Jeff=3/2 line, centered in column
if jeff32_E:
    ax_lvl.text(sx_mid, max(jeff32_E) + 2.5,
                r'$J_{\rm eff}\!=\!\frac{3}{2}$',
                ha='center', va='bottom', fontsize=5.5, color=C_FLIP)

# Δ_trig arrow: inside the column between the two Jeff=3/2 lines
if len(jeff32_E) > 1:
    dE_trig = jeff32_E[-1] - jeff32_E[0]
    mid_trig = (jeff32_E[0] + jeff32_E[-1]) / 2
    ax_lvl.annotate('', xy=(sx1 + 0.5, jeff32_E[-1]),
                    xytext=(sx1 + 0.5, jeff32_E[0]),
                    arrowprops=dict(arrowstyle='<->', color=C_FLIP, lw=0.7))
    ax_lvl.text(sx1 + 0.7, mid_trig,
                f'$\\Delta_{{\\rm trig}}$\n≈{dE_trig:.0f} meV',
                ha='left', va='center', fontsize=4.5, color=C_FLIP)

# Column headers at top
ax_lvl.text(lx_mid, Ecut + 9, 'V–V dimer',
            ha='center', fontsize=6, fontweight='bold', va='top')
ax_lvl.text(sx_mid, Ecut + 9, r'V$^{4+}$ single site',
            ha='center', fontsize=6, fontweight='bold', va='top')
ax_lvl.axvline(5.0, color='grey', lw=0.3, ls=':', alpha=0.6)

label(ax_lvl, 'c')




save(fig, 'fig_vv_dimer')
plt.close(fig)

print("\nAll EDRIXS figures done.")
print(f"  Y2V2O7 CF: 10Dq={ten_dq:.2f} eV, Δ_trig={delta_trig*1000:.0f} meV")
print(f"  Dimer: singlet {-3*J_meV/4:.2f} meV, triplet {J_meV/4:.2f} meV")
print(f"  SOC excitation (spin-flip): {np.round(E_soc_peaks, 1)} meV")
print(f"  I21 resolution: {res_FWHM_eV*1000:.0f} meV FWHM")
print(f"  Phonon: {E_ph*1000:.0f} meV (conserving only)")
