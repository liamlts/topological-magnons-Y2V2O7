"""
generate_xas_overview.py — V L2,3 XAS overview figure (500–530 eV).

Runs separate ED calculations for the L3 (2p3/2) and L2 (2p1/2) edges of
V4+ in Y2V2O7 and combines them on a single broadscale panel.

L3 (2p3/2): gamma_c ≈ 0.165 eV (narrow)
L2 (2p1/2): gamma_c ≈ 0.500 eV (broad — Coster-Kronig Auger channel)
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import edrixs

os.makedirs('Figures', exist_ok=True)

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

COL1   = 3.386   # single-column width (inches)
C_FLIP = '#D42E2E'
C_CONS = '#0073BD'
C_SUM  = '#555555'


def cf_trigonal_d(delta_trig):
    l = 2
    m_vals = np.arange(-l, l + 1, dtype=float)
    n = len(m_vals)
    Lp = np.zeros((n, n), dtype=complex)
    for i in range(n - 1):
        m = m_vals[i]
        Lp[i + 1, i] = np.sqrt((l - m) * (l + m + 1))
    Lm = Lp.conj().T
    Lx = (Lp + Lm) / 2.0
    Ly = (Lp - Lm) / (2j)
    Lz = np.diag(m_vals.astype(complex))
    LN = (Lx + Ly + Lz) / np.sqrt(3.0)
    H5 = -(delta_trig / 3.0) * (LN @ LN - l * (l + 1) / 3.0 * np.eye(n, dtype=complex))
    H10 = np.zeros((10, 10), dtype=complex)
    for i in range(n):
        for j in range(n):
            H10[2*i,   2*j  ] = H5[i, j]
            H10[2*i+1, 2*j+1] = H5[i, j]
    return H10


def gauss_convolve(spec, sigma_eV, dE_eV):
    hw = int(4 * sigma_eV / dE_eV) + 1
    k  = np.arange(-hw, hw + 1) * dE_eV
    kernel = np.exp(-0.5 * (k / sigma_eV) ** 2)
    kernel /= kernel.sum()
    return np.convolve(spec, kernel, mode='same')


def save(fig, name):
    for ext in ('pdf', 'png'):
        p = f'Figures/{name}.{ext}'
        fig.savefig(p, dpi=600 if ext == 'pdf' else 300)
        print(f'  saved {p}')


# ── Parameters ────────────────────────────────────────────────────────────────
v_noccu     = 1
ten_dq      = 1.9        # eV
delta_trig  = 0.030      # eV
zeta_d_i    = 0.030      # eV
zeta_d_n    = 0.030      # eV
T_K         = 10.0
res_FWHM_eV = 0.030
sigma_res   = res_FWHM_eV / (2 * np.sqrt(2 * np.log(2)))

thin_rad = np.radians(15.0)
phi      = 0.0

# ── Slater integrals from EDRIXS database (80% reduction) ────────────────────
info_l3    = edrixs.utils.get_atom_data('V', '3d', v_noccu, edge='L3')
c_soc_raw  = info_l3['c_soc']
zeta_p_n   = float(c_soc_raw[0]) if isinstance(c_soc_raw, (list, tuple)) else float(c_soc_raw)
gc_raw     = info_l3['gamma_c']
gamma_c_l3 = float(gc_raw[0]) if isinstance(gc_raw, (list, tuple)) else float(gc_raw)

F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(0.0, 0.0)
slater_n_db = {k: v for k, v in info_l3['slater_n']}
F2_pd = slater_n_db.get('F2_12', 6.759) * 0.8
G1_pd = slater_n_db.get('G1_12', 5.014) * 0.8
G3_pd = slater_n_db.get('G3_12', 2.853) * 0.8
F0_pd = edrixs.get_F0('dp', G1_pd, G3_pd)
slater = [[F0_d, F2_d, F4_d],
          [F0_d, F2_d, F4_d, F0_pd, F2_pd, G1_pd, G3_pd, 0.0, 0.0]]

# Edge energies: L3 onset ~515 eV, L2 onset ~522 eV (V 2p1/2–2p3/2 splitting ~7 eV)
off_l3    = 515.0 + F0_pd
off_l2    = 522.0 + F0_pd
gamma_c_l2 = 0.50  # L2 broader due to Coster-Kronig Auger decay

cf_mat = edrixs.cf_cubic_d(ten_dq) + cf_trigonal_d(delta_trig)

# ── L3 ED ─────────────────────────────────────────────────────────────────────
print("Running L3 ED...")
eval_i_l3, eval_n_l3, trans_op_l3 = edrixs.ed_1v1c_py(
    ('d', 'p32'),
    shell_level=(0.0, -off_l3),
    v_soc=(zeta_d_i, zeta_d_n),
    c_soc=zeta_p_n,
    v_noccu=v_noccu,
    slater=slater,
    v_cfmat=cf_mat,
    verbose=0,
)

# ── L2 ED ─────────────────────────────────────────────────────────────────────
print("Running L2 ED...")
eval_i_l2, eval_n_l2, trans_op_l2 = edrixs.ed_1v1c_py(
    ('d', 'p12'),
    shell_level=(0.0, -off_l2),
    v_soc=(zeta_d_i, zeta_d_n),
    c_soc=zeta_p_n,
    v_noccu=v_noccu,
    slater=slater,
    v_cfmat=cf_mat,
    verbose=0,
)

# ── XAS over 498–532 eV (buffer for convolution, display 500–530) ────────────
ominc = np.linspace(498.0, 532.0, 2400)
n_gs  = min(4, len(eval_i_l3))
dE    = float(ominc[1] - ominc[0])
pol   = [('linear', 0), ('linear', np.pi/2)]   # σ, π

print("Computing L3 XAS...")
raw_l3 = edrixs.xas_1v1c_py(
    eval_i_l3, eval_n_l3, trans_op_l3, ominc,
    gamma_c=gamma_c_l3, thin=thin_rad, phi=phi,
    pol_type=pol, gs_list=list(range(n_gs)), temperature=T_K,
)

print("Computing L2 XAS...")
raw_l2 = edrixs.xas_1v1c_py(
    eval_i_l2, eval_n_l2, trans_op_l2, ominc,
    gamma_c=gamma_c_l2, thin=thin_rad, phi=phi,
    pol_type=pol, gs_list=list(range(n_gs)), temperature=T_K,
)

# Combine L2 + L3
xas_sig = raw_l3[:, 0] + raw_l2[:, 0]
xas_pi  = raw_l3[:, 1] + raw_l2[:, 1]
xas_tot = xas_sig + xas_pi

# Broaden with I21 resolution
xas_sig = gauss_convolve(xas_sig, sigma_res, dE)
xas_pi  = gauss_convolve(xas_pi,  sigma_res, dE)
xas_tot = xas_sig + xas_pi

# Find L3 resonance from broadened L3-only σ
l3_sig_b = gauss_convolve(raw_l3[:, 0], sigma_res, dE)
E_res = ominc[np.argmax(l3_sig_b)]
print(f"L3 resonance (E_RIXS): {E_res:.2f} eV")

# Normalise all channels to the same global maximum (preserves dichroism)
mask_l3 = (ominc > 513) & (ominc < 522)
global_max = max(np.max(xas_sig), np.max(xas_pi), 1e-30)

# ── Figure ────────────────────────────────────────────────────────────────────
print("\nGenerating: fig_xas_overview")
fig, ax = plt.subplots(figsize=(COL1, 2.5),
                        gridspec_kw={'left': 0.14, 'right': 0.97,
                                     'top': 0.97, 'bottom': 0.18})

ax.plot(ominc, xas_sig / global_max, color=C_CONS, lw=1.1, label=r'$\sigma$ (LH)')
ax.plot(ominc, xas_pi  / global_max, color=C_FLIP, lw=1.1, label=r'$\pi$ (LV)')
ax.plot(ominc, xas_tot / global_max, color=C_SUM,  lw=0.7, ls='--',
        label='Total', alpha=0.85)

# E_RIXS marker
ax.axvline(E_res, color='k', lw=0.7, ls=':')
ax.text(E_res + 0.15, 0.93, r'$E_{\rm RIXS}$',
        transform=ax.get_xaxis_transform(), fontsize=5.5, va='top', color='k')

# L3 edge label: above t2g peak
i_l3 = np.argmax(xas_tot[mask_l3])
E_l3_peak = ominc[mask_l3][i_l3]
ax.text(E_l3_peak, xas_tot[mask_l3][i_l3] / global_max + 0.05,
        r'$L_3$', ha='center', va='bottom', fontsize=6.5, color=C_SUM, fontweight='bold')

# t2g and eg sub-labels within L3
ax.text(E_res - 0.10, np.interp(E_res, ominc, xas_sig) / global_max + 0.06,
        r'$t_{2g}$', ha='right', va='bottom', fontsize=5, color=C_SUM, style='italic')
mask_eg = (ominc > E_res + 0.8) & (ominc < E_res + 3.0)
if mask_eg.any():
    i_eg = np.argmax(xas_tot[mask_eg])
    ax.text(ominc[mask_eg][i_eg],
            xas_tot[mask_eg][i_eg] / global_max + 0.05,
            r'$e_g$', ha='center', va='bottom', fontsize=5, color=C_SUM, style='italic')

# L2 edge label — positioned at fixed location for clarity
mask_l2 = (ominc > 521) & (ominc < 530)
if mask_l2.any():
    i_l2 = np.argmax(xas_tot[mask_l2])
    E_l2_peak = ominc[mask_l2][i_l2]
    ax.text(520.0, 0.5,
            r'$L_2$', ha='center', va='bottom', fontsize=6.5, color=C_SUM, fontweight='bold')
    print(f"L2 peak: {E_l2_peak:.2f} eV")

ax.set_xlabel('Incident energy (eV)')
ax.set_ylabel('XAS (arb. units)')
ax.set_xlim(500, 530)
# Restrict y-axis so tallest peak is ~95% of displayed range
y_peak = max(np.max(xas_tot / global_max), 1.0)
ax.set_ylim(-0.03, y_peak / 0.95)
ax.legend(fontsize=5.5, loc='upper left', framealpha=0.85, ncol=1)

save(fig, 'fig_xas_overview')
plt.close(fig)
print("Done.")
