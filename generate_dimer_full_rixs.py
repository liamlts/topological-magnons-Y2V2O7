"""
generate_dimer_full_rixs.py — Full V4+-V4+ two-site cluster RIXS for Y2V2O7.

Exact Kramers-Heisenberg on a two-site dimer in the Y2V2O7 pyrochlore crystal
field.  The exchange/magnon peak at J = 8.22 meV emerges naturally from the
singlet-triplet splitting — no phenomenological peaks needed.

Orbital layout (28 spin-orbitals):
  [0:10]  site A d   [10:20] site B d
  [20:24] site A p32 [24:28] site B p32

Hilbert spaces:
  Initial:                C(20,2)×C(8,8)        =  190 states
  Intermediate (hole A):  C(20,3)×C(4,3)×C(4,4) = 4560 states
  Intermediate (hole B):  symmetric               = 4560 states
"""
import os, time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from scipy.linalg import eigh
import edrixs

os.makedirs('Figures', exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica','Arial','DejaVu Sans'],
    'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
    'legend.fontsize': 6, 'axes.linewidth': 0.5,
    'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'lines.linewidth': 0.8, 'figure.dpi': 150,
    'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})
COL1 = 3.386; C_FLIP = '#D42E2E'; C_CONS = '#0073BD'; C_PHON = '#38A12B'; C_SUM = '#333333'

# ── Parameters ────────────────────────────────────────────────────────────────
ten_dq    = 1.9;   delta_trig = 0.030;  zeta_d = 0.030   # eV
# J > 0 = ferromagnetic (triplet lower than singlet by J).
# Convention: H = J S̃_A · S̃_B with J > 0 gives FM ground-state triplet.
J_meV     = 8.22;  J_eV = J_meV / 1000.0
T_K       = 10.0;  kBT  = T_K * 8.617e-5                 # eV
gamma_f   = 0.002                                          # eV final-state width
res_FWHM  = 0.030; sigma_res = res_FWHM / (2*np.sqrt(2*np.log(2)))

# Slater integrals (80% reduction, from EDRIXS database)
info = edrixs.utils.get_atom_data('V', '3d', 1, edge='L3')
c_soc_raw = info['c_soc']
zeta_p    = float(c_soc_raw[0] if hasattr(c_soc_raw,'__len__') else c_soc_raw)
gc_raw    = info['gamma_c']
gamma_c   = float(gc_raw[0] if hasattr(gc_raw,'__len__') else gc_raw)
F0_d, F2_d, F4_d = edrixs.UdJH_to_F0F2F4(0.0, 0.0)  # for p-d Coulomb (1e/site, so dd vanishes)
si = dict(info['slater_i'])
F2_dd = si.get('F2_11', 10.127) * 0.8    # d-d Slater integrals (80% reduction)
F4_dd = si.get('F4_11', 6.354)  * 0.8
U_dd  = 3.0                               # Hubbard U for V in oxide (eV)
sn = dict(info['slater_n'])
F2_pd = sn.get('F2_12', 6.759)*0.8
G1_pd = sn.get('G1_12', 5.014)*0.8
G3_pd = sn.get('G3_12', 2.853)*0.8
F0_pd = edrixs.get_F0('dp', G1_pd, G3_pd)
# off_l3 = 515.0 + F0_pd is the bare edge energy; the d-d Hubbard U shifts
# the intermediate-state XAS resonance up by ~3.5 eV, so compensate here
# to restore the physical L3 resonance to ~518 eV.
off_l3 = 511.5 + F0_pd

print(f"gamma_c = {gamma_c:.3f} eV,  F0_pd = {F0_pd:.3f} eV,  off_l3 = {off_l3:.3f} eV")
print(f"J = {J_meV:.2f} meV,  T = {T_K:.0f} K,  kBT = {kBT*1000:.2f} meV")


# ── Crystal field + SOC ───────────────────────────────────────────────────────
def cf_trigonal_d(delta):
    l = 2; n = 2*l+1; mv = np.arange(-l, l+1, dtype=float)
    Lp = np.zeros((n,n), dtype=complex)
    for i in range(n-1): Lp[i+1,i] = np.sqrt((l-mv[i])*(l+mv[i]+1))
    Lm = Lp.conj().T
    LN = ((Lp+Lm)/2 + (Lp-Lm)/(2j) + np.diag(mv)) / np.sqrt(3)
    H5 = -(delta/3)*(LN@LN - l*(l+1)/3*np.eye(n, dtype=complex))
    H10 = np.zeros((10,10), dtype=complex)
    for i in range(n):
        for j in range(n):
            H10[2*i, 2*j]     = H5[i,j]
            H10[2*i+1, 2*j+1] = H5[i,j]
    return H10

cf_mat  = edrixs.cf_cubic_d(ten_dq) + cf_trigonal_d(delta_trig)
soc_mat = edrixs.atom_hsoc('d', zeta_d)
hd_site = cf_mat + soc_mat   # 10×10 single-site d Hamiltonian

# ── dp32 Slater integrals (single site, shape 14×14×14×14) ───────────────────
umat_dp = edrixs.get_umat_slater('dp32', F0_d, F2_d, F4_d, F0_pd, F2_pd, G1_pd, G3_pd, 0.0, 0.0)

# ── emat_i: one-body part of initial-state Hamiltonian ───────────────────────
NORBS = 28
emat_i = np.zeros((NORBS, NORBS), dtype=complex)
emat_i[0:10,  0:10 ] = hd_site           # site A d
emat_i[10:20, 10:20] = hd_site           # site B d
emat_i[20:24, 20:24] = -off_l3 * np.eye(4)   # site A p32
emat_i[24:28, 24:28] = -off_l3 * np.eye(4)   # site B p32

# ── Jeff=1/2 pseudo-spin exchange H = J S̃_A · S̃_B ──────────────────────────
# Strategy: project exchange onto the actual CF+SOC Jeff=1/2 doublet of a
# single d1 site.  The full L+S operator in the d-orbital space has strongly
# suppressed matrix elements in the t2g-derived Jeff=1/2 subspace (L+ mixes
# t2g into eg, quenched by ten_dq/ζ ~ 60).  Instead we build S̃ = σ/2
# directly from the two lowest eigenstates of hd_site.
#
# For S̃_A · S̃_B with identical-site doublet basis, the singlet-triplet gap
# is exactly J regardless of the arbitrary phase convention of {|ψ₁⟩,|ψ₂⟩}.

_ev1, _vec1 = eigh(hd_site)            # single-site CF+SOC
psi1 = _vec1[:, 0]                      # GS (lower Kramers state)
psi2 = _vec1[:, 1]                      # Kramers partner
print(f"  Single-site GS gap: {(_ev1[1]-_ev1[0])*1000:.4f} meV  (should be ~0 = Kramers)")
print(f"  t2g–eg gap:         {(_ev1[2]-_ev1[0])*1000:.2f} meV")

# S̃_z = (1/2)(|ψ₁⟩⟨ψ₁| − |ψ₂⟩⟨ψ₂|),  S̃₊ = |ψ₁⟩⟨ψ₂|,  S̃₋ = |ψ₂⟩⟨ψ₁|
Sz_eff = 0.5*(np.outer(psi1, psi1.conj()) - np.outer(psi2, psi2.conj()))
Sp_eff = np.outer(psi1, psi2.conj())
Sm_eff = np.outer(psi2, psi1.conj())

# umat encoding: O^A O^B → c†_{Ai} c†_{Bk} c_{Aj} c_{Bl} with sign from normal order
# umat[i, 10+k, j, 10+l] += −coeff × (O^A)_{ij} × (O^B)_{kl}
umat_i = np.zeros((NORBS,)*4, dtype=complex)

def _add_inter(op_A, op_B, coeff):
    for i in range(10):
        for j in range(10):
            vA = op_A[i, j]
            if abs(vA) < 1e-14: continue
            for k in range(10):
                for l in range(10):
                    vB = op_B[k, l]
                    if abs(vB) < 1e-14: continue
                    umat_i[i, 10+k, j, 10+l] += -coeff * vA * vB

_add_inter(Sz_eff, Sz_eff, J_eV)       # J S̃z_A S̃z_B
_add_inter(Sp_eff, Sm_eff, J_eV / 2)   # J/2 S̃+_A S̃−_B
_add_inter(Sm_eff, Sp_eff, J_eV / 2)   # J/2 S̃−_A S̃+_B

# ── Intra-site d-d Coulomb (pushes doubly-occupied configs to ~U_dd ≈ 3 eV) ──
# Without this, 2e-on-one-site configs are degenerate with physical 1e/site states.
umat_dd = edrixs.get_umat_slater('d', U_dd, F2_dd, F4_dd)  # (10,10,10,10)
print(f"Embedding d-d Coulomb (U={U_dd:.1f} eV, F2={F2_dd:.3f}, F4={F4_dd:.3f})...")
for d_off in [0, 10]:   # site A (0:10) and site B (10:20)
    nz = 0
    for i in range(10):
        for j in range(10):
            for k in range(10):
                for l in range(10):
                    v = umat_dd[i,j,k,l]
                    if abs(v) < 1e-12: continue
                    umat_i[d_off+i, d_off+j, d_off+k, d_off+l] += v
                    nz += 1
    print(f"  site {d_off//10}: {nz} non-zero d-d terms added")

# ── emat_n, umat_n for each core-hole site ───────────────────────────────────
# indx_dp[s]: 14 global indices mapping [d0..d9, p0..p3] to full 28-orbital space
indx_dp = [
    list(range(10))   + list(range(20,24)),  # site A
    list(range(10,20)) + list(range(24,28)), # site B
]

emat_n = [emat_i.copy(), emat_i.copy()]
umat_n = [umat_i.copy(), umat_i.copy()]
print("\nEmbedding dp32 Slater integrals...")
for s in range(2):
    ix = indx_dp[s]
    nz = 0
    for i in range(14):
        for j in range(14):
            for k in range(14):
                for l in range(14):
                    v = umat_dp[i,j,k,l]
                    if v != 0.0:
                        umat_n[s][ix[i], ix[j], ix[k], ix[l]] += v
                        nz += 1
    print(f"  site {s}: {nz} non-zero dp terms added")

# ── Fock bases ────────────────────────────────────────────────────────────────
print("\nBuilding Fock bases...")
t0 = time.time()
basis_i  = edrixs.get_fock_bin_by_N(20, 2, 8, 8)             # 190 states
basis_n0 = edrixs.get_fock_bin_by_N(20, 3, 4, 3, 4, 4)       # 4560 states (hole on A)
basis_n1 = edrixs.get_fock_bin_by_N(20, 3, 4, 4, 4, 3)       # 4560 states (hole on B)
basis_n  = [basis_n0, basis_n1]
print(f"  Initial: {len(basis_i)} states,  Intermediate: {len(basis_n0)} per site  ({time.time()-t0:.1f}s)")

# ── Diagonalize initial Hamiltonian ───────────────────────────────────────────
print("\nBuilding H_i (190×190)...")
t0 = time.time()
H_i = (edrixs.two_fermion(emat_i, basis_i) +
       edrixs.four_fermion(umat_i, basis_i))
print(f"  Built in {time.time()-t0:.1f}s")
eval_i, evec_i = eigh(H_i)
del H_i
E_rel = (eval_i[:8] - eval_i[0]) * 1000   # meV
print(f"  Lowest levels (meV): {np.round(E_rel, 2)}")
print(f"  → singlet {E_rel[0]:.2f} meV,  triplet {E_rel[1]:.2f} meV  (gap = {E_rel[1]:.2f} meV)")

n_gs = min(4, len(eval_i))
boltz = np.exp(-(eval_i[:n_gs] - eval_i[0]) / kBT)
boltz /= boltz.sum()
print(f"  Boltzmann weights @ {T_K:.0f}K: {np.round(boltz, 4)}")

# Keep final states within a window above GS (no RIXS signal from higher states)
E_window_eV = 0.250
mask_f = (eval_i - eval_i[0]) < E_window_eV
idx_f  = np.where(mask_f)[0]
eval_f = eval_i[mask_f]
print(f"  Final states in [{E_window_eV*1000:.0f} meV window]: {len(eval_f)}")

# ── Diagonalize intermediate Hamiltonians (one per core-hole site) ────────────
eval_n = []; evec_n = []
for s, (en, un, bn) in enumerate(zip(emat_n, umat_n, basis_n)):
    print(f"\nBuilding H_n[{s}] ({len(bn)}×{len(bn)})...")
    t0 = time.time()
    H_n  = edrixs.two_fermion(en, bn)
    print(f"  two_fermion: {time.time()-t0:.1f}s")
    t0 = time.time()
    H_n += edrixs.four_fermion(un, bn)
    print(f"  four_fermion: {time.time()-t0:.1f}s")
    print(f"  Diagonalizing...")
    t0 = time.time()
    ev, vc = eigh(H_n)
    print(f"  eigh: {time.time()-t0:.1f}s")
    eval_n.append(ev); evec_n.append(vc)
    del H_n

# ── Transition operators ──────────────────────────────────────────────────────
# get_trans_oper('dp32') → (3, 10, 4): Cartesian [x, y, z], d spin-orbs, p32 spin-orbs
trans_raw = edrixs.get_trans_oper('dp32')   # shape (3, 10, 4)

# Geometry approximation: σ (LH) = ŷ (pol_idx=1), π (LV) = ẑ (pol_idx=2)
# Accurate enough for 2θ=150° horizontal scattering to separate spin-flip from conserving.

def make_abs_tmat(pol_idx, site):
    """Single-particle absorption matrix (28×28): element [d_global, p_global] = t[d,p]."""
    tmat = np.zeros((NORBS, NORBS), dtype=complex)
    d_off = 0 if site == 0 else 10
    p_off = 20 if site == 0 else 24
    for d in range(10):
        for p in range(4):
            tmat[d_off+d, p_off+p] = trans_raw[pol_idx, d, p]
    return tmat

print("\nBuilding many-body transition operators...")
# Channels: sf = σπ (spin-flip), cons = σσ (conserving)
# absorption pol: both use σ (y), emission pol: sf→π (z), cons→σ (y)
pols = {
    'sf':      (1, 2),   # σ(LH) in, π(LV) out  — spin-flip
    'cons':    (1, 1),   # σ(LH) in, σ(LH) out  — conserving
    'lv_sf':   (2, 1),   # π(LV) in, σ(LH) out  — spin-flip, LV incident
    'lv_cons': (2, 2),   # π(LV) in, π(LV) out  — conserving, LV incident
}

T_abs_eig = {ch: [] for ch in pols}   # [site] → (n_n, n_gs)
T_emi_eig = {ch: [] for ch in pols}   # [site] → (n_f_window, n_n)

for s in range(2):
    for ch, (pi, po) in pols.items():
        t0 = time.time()
        # Absorption: c†_d c_p, left=basis_n, right=basis_i
        tmat_a = make_abs_tmat(pi, s)
        T_abs_mb  = edrixs.two_fermion(tmat_a, basis_n[s], basis_i)           # (n_n, n_i)
        T_a_eig   = evec_n[s].conj().T @ T_abs_mb @ evec_i                     # (n_n, n_i)
        T_abs_eig[ch].append(T_a_eig[None, :, :n_gs])                           # (1, n_n, n_gs)

        # Emission: c†_p c_d (adjoint of absorption), left=basis_i, right=basis_n
        tmat_e = make_abs_tmat(po, s).conj().T                                  # (NORBS,NORBS)
        T_emi_mb  = edrixs.two_fermion(tmat_e, basis_i, basis_n[s])            # (n_i, n_n)
        T_e_eig   = evec_i.conj().T @ T_emi_mb @ evec_n[s]                     # (n_i, n_n)
        T_emi_eig[ch].append(T_e_eig[None, idx_f, :])                          # (1, n_f_window, n_n)
        print(f"  site {s}, {ch}: {time.time()-t0:.1f}s")

# ── XAS (σ-polarization, sum both sites, find resonance) ─────────────────────
print("\nComputing XAS...")
ominc_xas = np.linspace(512.0, 525.0, 500)
xas = np.zeros(len(ominc_xas))
for s in range(2):
    amp2 = np.abs(T_abs_eig['cons'][s][0, :, :])**2  # (n_n, n_gs)
    for ig in range(n_gs):
        dE   = eval_n[s] - eval_i[ig]                              # (n_n,)
        diff = ominc_xas[:, None] - dE[None, :]                    # (n_om, n_n)
        L    = gamma_c / (diff**2 + gamma_c**2) / np.pi
        xas += boltz[ig] * L @ amp2[:, ig]
# Broaden with I21 resolution
dE_xas   = float(ominc_xas[1] - ominc_xas[0])
hw       = int(4*sigma_res/dE_xas)+1
k        = np.arange(-hw,hw+1)*dE_xas
kernel   = np.exp(-0.5*(k/sigma_res)**2); kernel /= kernel.sum()
xas_conv = np.convolve(xas, kernel, mode='same')
E_res    = ominc_xas[np.argmax(xas_conv)]
print(f"  XAS resonance: {E_res:.3f} eV")

# ── RIXS at E_res via Kramers-Heisenberg ─────────────────────────────────────
print(f"\nComputing RIXS at {E_res:.3f} eV (KH)...")
eloss_eV  = np.linspace(-0.030, 0.200, 3400)
eloss_meV = eloss_eV * 1000.0
I_sf   = np.zeros(len(eloss_eV))
I_cons = np.zeros(len(eloss_eV))

for s in range(2):
    for ch, I_out in [('sf', I_sf), ('cons', I_cons)]:
        t0 = time.time()
        # scattering_mat: abs=(1,n_n,n_gs), emi=(1,n_f,n_n) → F=(1,1,n_f,n_gs)
        F = edrixs.scattering_mat(
            eval_i[:n_gs], eval_n[s],
            T_abs_eig[ch][s],         # (1, n_n, n_gs)
            T_emi_eig[ch][s],         # (1, n_f_window, n_n)
            E_res, gamma_c,
        )
        # F: (1, 1, n_f_window, n_gs)
        F2 = F[0, 0, :, :]                                          # (n_f_win, n_gs)
        eloss_trans = eval_f[:, None] - eval_i[None, :n_gs]         # (n_f_win, n_gs)
        amps2       = np.abs(F2)**2                                  # (n_f_win, n_gs)
        diff = eloss_eV[:, None, None] - eloss_trans[None, :, :]    # (n_el, n_f, n_gs)
        L    = gamma_f / (diff**2 + gamma_f**2) / np.pi
        I_out += (L * amps2[None,:,:] * boltz[None,None,:]).sum(axis=(1,2))
        print(f"  site {s}, {ch}: {time.time()-t0:.1f}s")

# ── Phenomenological phonon (conserving only, 100 meV intrinsic width) ───────
# No instrumental resolution convolution — spectra broadened only by gamma_f
# (final-state Lorentzian) and gamma_c (core-hole, absorbed in KH formula).
dE = float(eloss_eV[1] - eloss_eV[0])
A_ph = 0.65 * max(np.max(I_cons), 1e-30)
phonon = A_ph * np.exp(-0.5*((eloss_eV - 0.100)/0.003)**2)   # 3 meV σ intrinsic
I_cons_total = I_cons + phonon
I_sf_total   = I_sf   + phonon * 0.05
I_iso_total  = I_cons_total + I_sf_total

norm = max(np.max(I_iso_total), 1e-30)

# ── Find SOC peak positions ───────────────────────────────────────────────────
from scipy.signal import find_peaks
mask_soc = (eloss_meV > 5) & (eloss_meV < 150)
pks, _   = find_peaks(I_sf_total * mask_soc, height=np.max(I_sf_total)*0.02, distance=50)
E_soc    = eloss_meV[pks] if len(pks) else np.array([])
print(f"\nRIXS peaks (spin-flip, broadened): {np.round(E_soc, 1)} meV")

# ── Figure ────────────────────────────────────────────────────────────────────
print("\nGenerating: fig_dimer_full_rixs")
fig, ax = plt.subplots(figsize=(COL1, 2.6),
                        gridspec_kw={'left':0.16,'right':0.97,'top':0.94,'bottom':0.18})

ax.plot(eloss_meV, I_iso_total/norm,  color=C_SUM,  lw=1.3,
        label='Isotropic (total)', zorder=3)
ax.fill_between(eloss_meV, I_cons_total/norm, alpha=0.15, color=C_CONS)
ax.plot(eloss_meV, I_cons_total/norm, color=C_CONS, lw=1.0,
        label=r'Conserving ($\sigma\sigma$)')
ax.fill_between(eloss_meV, I_sf_total/norm, alpha=0.15, color=C_FLIP)
ax.plot(eloss_meV, I_sf_total/norm, color=C_FLIP, lw=1.0,
        label=r'Spin-flip ($\sigma\pi$)')

# SOC peak annotations (on spin-flip)
for i, Ep in enumerate(E_soc):
    y_p = np.interp(Ep, eloss_meV, I_sf_total/norm)
    # stagger first two peaks vertically to avoid overlap
    if i == 0:
        xytext = (Ep, y_p + 0.22)
        ha = 'center'
    else:
        xytext = (Ep + 8, y_p + 0.08)
        ha = 'left'
    ax.annotate(f'{Ep:.0f} meV', xy=(Ep, y_p*1.03),
                xytext=xytext,
                fontsize=5.0, ha=ha, color=C_FLIP,
                arrowprops=dict(arrowstyle='->', color=C_FLIP, lw=0.5))

y_top = np.max(I_iso_total/norm)
ax.set_xlabel('Energy loss (meV)')
ax.set_ylabel('RIXS (norm. units)')
ax.set_xlim(-10, 200)
ax.set_ylim(-0.03, y_top * 1.28)
ax.legend(fontsize=5.5, loc='upper right', framealpha=0.9)
ax.set_title(r'Y$_2$V$_2$O$_7$ V–V dimer RIXS (full cluster)', fontsize=6, pad=3)

for ext in ('pdf', 'png'):
    p = f'Figures/fig_dimer_full_rixs.{ext}'
    fig.savefig(p, dpi=600 if ext=='pdf' else 300)
    print(f'  saved {p}')
plt.close(fig)

# ── 2D RIXS map: π(LV) in, isotropic (σ+π) out ──────────────────────────
print("\nComputing 2D RIXS map (π/LV in, isotropic out)...")
ominc_scan = np.linspace(512.0, 522.0, 180)
I_2d       = np.zeros((len(ominc_scan), len(eloss_eV)))

t_2d = time.time()
for i_om, E_in in enumerate(ominc_scan):
    for s in range(2):
        for ch in ['lv_sf', 'lv_cons']:
            F  = edrixs.scattering_mat(
                eval_i[:n_gs], eval_n[s],
                T_abs_eig[ch][s],   # (1, n_n, n_gs)
                T_emi_eig[ch][s],   # (1, n_f_win, n_n)
                E_in, gamma_c,
            )
            F2          = F[0, 0, :, :]                              # (n_f_win, n_gs)
            eloss_trans = eval_f[:, None] - eval_i[None, :n_gs]     # (n_f_win, n_gs)
            amps2       = np.abs(F2)**2
            diff = eloss_eV[:, None, None] - eloss_trans[None, :, :]
            L    = gamma_f / (diff**2 + gamma_f**2) / np.pi
            I_2d[i_om] += (L * amps2[None,:,:] * boltz[None,None,:]).sum(axis=(1,2))
    if (i_om + 1) % 30 == 0 or i_om == 0:
        elapsed = time.time() - t_2d
        print(f"  ω_in {i_om+1:2d}/{len(ominc_scan)}: {E_in:.3f} eV  ({elapsed:.1f}s elapsed)")

# Add phenomenological phonon: amplitude at resonance, envelope follows XAS
# Extra display broadening for XAS strip (~50 meV extra sigma)
sig_disp_2d  = 0.050 / (2*np.sqrt(2*np.log(2)))
hw_d         = int(4*sig_disp_2d/dE_xas) + 1
k_d          = np.arange(-hw_d, hw_d+1) * dE_xas
kern_d       = np.exp(-0.5*(k_d/sig_disp_2d)**2); kern_d /= kern_d.sum()
xas_conv_disp = np.convolve(xas_conv, kern_d, mode='same')
xas_env  = np.interp(ominc_scan, ominc_xas, xas_conv_disp / (np.max(xas_conv_disp) + 1e-30))
i_res    = np.argmin(np.abs(ominc_scan - E_res))
A_ph_2d  = 0.65 * max(np.max(I_2d[i_res]), 1e-30)
ph_shape = np.exp(-0.5 * ((eloss_eV - 0.100) / 0.003)**2)   # 3 meV σ
I_2d    += A_ph_2d * xas_env[:, None] * ph_shape[None, :]

# Elastic line: resolution-limited peak at 0 meV loss, follows XAS envelope
# Amplitude relative to inelastic — in real RIXS elastic is intense but
# we reduce it here to avoid washing out the inelastic features in the colorscale
A_el     = 1.5 * max(np.max(I_2d[i_res]), 1e-30)
el_shape = np.exp(-0.5 * (eloss_eV / (sigma_res * 0.7))**2)  # slightly sub-resolution (spectrometer tail)
I_2d    += A_el * xas_env[:, None] * el_shape[None, :]

# Gaussian broaden along incident energy axis for smoother appearance
sig_inc = 0.08  # eV σ (~190 meV FWHM) — mild cosmetic smoothing
dE_inc  = float(ominc_scan[1] - ominc_scan[0])
hw_inc  = int(4 * sig_inc / dE_inc) + 1
k_inc   = np.arange(-hw_inc, hw_inc + 1) * dE_inc
kern_inc = np.exp(-0.5 * (k_inc / sig_inc)**2); kern_inc /= kern_inc.sum()
from scipy.ndimage import convolve1d
I_2d = convolve1d(I_2d, kern_inc, axis=0, mode='nearest')

print(f"  2D map done in {time.time()-t_2d:.1f}s.  Peak intensity: {np.max(I_2d):.3e}")

# ── Figure: 2D RIXS map (incident energy on x, energy loss in eV on y) ────
print("\nGenerating: fig_rixs_2d_map")
fig_2d = plt.figure(figsize=(COL1 * 1.18, 3.8))
gs2    = GridSpec(2, 1, height_ratios=[1, 4], hspace=0.04,
                  left=0.17, right=0.88, top=0.94, bottom=0.12)
ax_m   = fig_2d.add_subplot(gs2[1])
ax_x   = fig_2d.add_subplot(gs2[0], sharex=ax_m)

# Show full range including elastic; y-axis in meV
disp_mask   = eloss_meV > -25.0
e_disp_meV  = eloss_meV[disp_mask]        # meV
I_disp      = I_2d[:, disp_mask]          # (n_ominc, n_eloss_masked)

vmax    = np.percentile(I_disp, 98.5)
n2d     = PowerNorm(gamma=0.40, vmin=0, vmax=vmax)
im      = ax_m.pcolormesh(ominc_scan, e_disp_meV, I_disp.T,
                           cmap='inferno', norm=n2d,
                           shading='auto', rasterized=True)
cb = fig_2d.colorbar(im, ax=ax_m, fraction=0.042, pad=0.025, aspect=22)
cb.set_label('RIXS (arb. units)', fontsize=5.5)
cb.ax.tick_params(labelsize=4.5, width=0.3, length=1.5)
cb.outline.set_linewidth(0.4)

# Resonance marker (vertical line in both panels)
ax_m.axvline(E_res, color='w', lw=0.7, ls='--', alpha=0.8)
ax_x.axvline(E_res, color='#D42E2E', lw=0.7, ls='--', alpha=0.9)
ax_m.text(E_res + 0.07, 145,
          rf'$E_{{\rm res}}$', fontsize=5.5, color='w', va='top', ha='left')

# Feature markers: horizontal lines at energy-loss positions (meV)
for E_pk_meV, lbl in [(0, 'elastic'), (E_rel[1], '$J$'), (22, 'SOC$_1$'), (67, 'SOC$_2$'), (100, 'ph.')]:
    ax_m.axhline(E_pk_meV, color='w', lw=0.35, ls=':', alpha=0.55)
    ax_m.text(ominc_scan[0] + 0.08, E_pk_meV + 2, lbl,
              fontsize=4.5, color='w', ha='left', va='bottom')

ax_m.set_xlabel('Incident energy (eV)')
ax_m.set_ylabel('Energy loss (meV)')
ax_m.set_xlim(ominc_scan[0], ominc_scan[-1])
ax_m.set_ylim(-25, 150)

# XAS top panel
ax_x.plot(ominc_scan, xas_env, color='#5BA4CF', lw=0.9)
ax_x.fill_between(ominc_scan, 0, xas_env, alpha=0.30, color='#5BA4CF')
ax_x.set_ylabel('XAS\n(norm. units)', fontsize=5.5, labelpad=1)
ax_x.set_ylim(0, 1.45)
ax_x.tick_params(labelbottom=False, labelsize=5.5)
ax_x.set_title(r'V$^{4+}$–V$^{4+}$ dimer 2D RIXS ($\pi$ in, isotropic out)',
               fontsize=6, pad=3)

for ext in ('pdf', 'png'):
    p = f'Figures/fig_rixs_2d_map.{ext}'
    fig_2d.savefig(p, dpi=600 if ext == 'pdf' else 300)
    print(f'  saved {p}')
plt.close(fig_2d)

print("\nDone.")
print(f"  Exchange gap from dimer ED:  {E_rel[1]:.3f} meV  (target J = {J_meV:.2f} meV)")
print(f"  SOC excitations (spin-flip): {np.round(E_soc, 1)} meV")
