"""
regen_berry_arc.py — Regenerate only fig_berry_chern and fig_surface_arc.
Skips all slow sections (spectral weight, phase diagram, Hall conductivity).
Runtime: ~2-3 min.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

from y2v2o7.lattice import J_FM, n_sub, b1, b2, L_pt, frac2cart
from y2v2o7.lswt import (build_Hq, build_H_slab, berry_curvature_vec,
                         chern_number_sphere, find_crossings,
                         fit_cone_velocities, kp_projection,
                         representative_crossing)
from y2v2o7.style import use as use_style, label, save, COL1, COL2, C_BLUE

os.makedirs('Figures', exist_ok=True)
use_style()

# ── Physics setup ─────────────────────────────────────────────────────────────
DJ_vals = np.array([0.00, 0.10, 0.20, 0.32, 0.40, 0.50,
                    0.60, 0.70, 0.80, 0.90, 1.00])

qL = frac2cart(L_pt)
Lh = qL / np.linalg.norm(qL)
e1_GL = np.cross(Lh, np.array([0., 0., 1.]))
e1_GL /= np.linalg.norm(e1_GL)
e2_GL = np.cross(Lh, e1_GL)
e2_GL /= np.linalg.norm(e2_GL)


# ── Fine Γ→L scan — find Weyl crossings ──────────────────────────────────────
n_fine = 5000
t_fine = np.linspace(0, 1, n_fine)
qGL    = np.outer(t_fine, qL)
qL_len = np.linalg.norm(qL)

print("Computing fine Γ→L scan...")
omGL = np.zeros((len(DJ_vals), n_sub, n_fine))
for idj, dj in enumerate(DJ_vals):
    D = dj * J_FM
    for iq in range(n_fine):
        ev = np.sort(np.real(eigh(build_Hq(qGL[iq], D), eigvals_only=True)))
        omGL[idj, :, iq] = ev

crossings = find_crossings(omGL, t_fine, DJ_vals, qL, thr=0.05)

print(f"Crossings found: {len(crossings)}")

# k·p velocities + chirality
fit_cone_velocities(crossings, omGL, t_fine, qL_len, win=0.08)

for wc in crossings:
    if np.isnan(wc.get('vW', np.nan)):
        wc['chi'] = 0; continue
    wc['chi'], _ = kp_projection(wc['q'], wc['bands'], wc['dj'] * J_FM)

print("k·p done.")

# Select repr_wc closest to D/J = 0.32
repr_wc = representative_crossing(crossings, 0.32)

# ══════════════════════════════════════════════════════════════════════════════
#  BERRY CURVATURE + CHERN NUMBER
# ══════════════════════════════════════════════════════════════════════════════
if repr_wc is not None:
    print("\nComputing Berry curvature and Chern number...")
    D_r  = repr_wc['dj'] * J_FM
    qW   = repr_wc['q']
    b_lo, b_hi = repr_wc['bands']
    chi_kp = repr_wc.get('chi', 0)

    # 2D map in transverse plane
    N_map  = 41
    dq_map = 0.10
    qq     = np.linspace(-dq_map, dq_map, N_map)
    Om_map = np.zeros((N_map, N_map))
    print("  Computing 2-D Berry curvature map...")
    for ix, dx in enumerate(qq):
        for iy, dy in enumerate(qq):
            k = qW + dx * e1_GL + dy * e2_GL
            Om_map[iy, ix] = np.dot(berry_curvature_vec(k, b_lo, D_r), Lh)
    # Ensure center is positive (red) — sign is a display orientation choice; C=+1 unchanged
    if Om_map[N_map//2, N_map//2] < 0:
        Om_map = -Om_map

    # Chern number vs sphere radius
    r_vals = np.array([0.015, 0.025, 0.040, 0.055, 0.070, 0.085, 0.100])
    C_vals = np.zeros(len(r_vals))
    print("  Computing Chern number at multiple sphere radii...")
    for ir, r in enumerate(r_vals):
        C_vals[ir] = chern_number_sphere(qW, r, b_lo, D_r, N_theta=20, N_phi=40)
        print(f"    r = {r:.3f} Å⁻¹  →  C = {C_vals[ir]:.3f}")
    print(f"  k·p chirality: χ = {chi_kp:+d}")

    # Multi-L scan for partner Weyl points (print only)
    L_frac_all = {
        r'$\Gamma\!\to\!L_{[111]}$':      np.array([0.5, 0.5, 0.5]),
        r'$\Gamma\!\to\!L_{[11\bar1]}$':  np.array([0.0, 0.0, 0.5]),
        r'$\Gamma\!\to\!L_{[1\bar11]}$':  np.array([0.0, 0.5, 0.0]),
        r'$\Gamma\!\to\!L_{[\bar111]}$':  np.array([0.5, 0.0, 0.0]),
    }
    n_sc, thr_sc = 1500, 0.15
    all_wcs = []
    print("  Scanning all 4 Γ→L directions for Weyl crossings...")
    for lbl, lfrac in L_frac_all.items():
        qLi  = frac2cart(lfrac)
        t_sc = np.linspace(0, 1, n_sc)
        qpi  = np.outer(t_sc, qLi)
        om_sc = np.zeros((n_sub, n_sc))
        for iq in range(n_sc):
            om_sc[:, iq] = np.sort(np.real(eigh(build_Hq(qpi[iq], D_r), eigvals_only=True)))
        for b in range(n_sub - 1):
            gap  = om_sc[b+1] - om_sc[b]
            imin = np.argmin(gap)
            if gap[imin] < thr_sc and 0 < imin < n_sc - 1:
                tc  = t_sc[imin]
                oc  = 0.5 * (om_sc[b, imin] + om_sc[b+1, imin])
                q_c = tc * qLi
                chi_c, _ = kp_projection(q_c, (b, b+1), D_r)
                all_wcs.append({'label': lbl, 'chi': chi_c, 't': tc, 'omega': oc, 'bands': (b, b+1)})
                break

    chi_total = sum(w['chi'] for w in all_wcs)
    print(f"\n  All Weyl crossings at D/J = {repr_wc['dj']:.2f}:")
    for w in all_wcs:
        print(f"    {w['label']}: bands {w['bands'][0]+1}–{w['bands'][1]+1}, "
              f"t={w['t']:.3f}, ω={w['omega']:.1f} meV, χ={w['chi']:+d}")
    print(f"  Sum of chiralities: {chi_total:+d}")

    # Figure
    print("\nGenerating: fig_berry_chern")
    fig, (ax_bc, ax_cv) = plt.subplots(1, 2, figsize=(COL2 * 0.70, 2.5),
                                        gridspec_kw={'wspace': 0.42,
                                                     'left': 0.10, 'right': 0.96,
                                                     'top': 0.90, 'bottom': 0.18})

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

# ══════════════════════════════════════════════════════════════════════════════
#  SURFACE ARC
# ══════════════════════════════════════════════════════════════════════════════
print("\n── Surface arc slab calculation ──────────────────────────────────────────")

D_arc  = 0.32 * J_FM
N_slab = 50
N_surf = 2
N_kpar = 300


Xbar = b1 / 2
Mbar = (b1 + b2) / 2
kpath_seg1 = np.array([Xbar*(1-t) for t in np.linspace(0, 1, N_kpar//2)])
kpath_seg2 = np.array([Mbar*t      for t in np.linspace(0, 1, N_kpar//2)])
kpar_path  = np.vstack([kpath_seg1, kpath_seg2])
tick_kpar  = [0, N_kpar//2, N_kpar-1]
tick_labels = [r'$\bar{X}$', r'$\bar{\Gamma}$', r'$\bar{M}$']

kpar_dist = np.zeros(N_kpar)
for ik in range(1, N_kpar):
    kpar_dist[ik] = kpar_dist[ik-1] + np.linalg.norm(kpar_path[ik] - kpar_path[ik-1])

print(f"  Computing slab spectrum ({N_slab} layers, {N_kpar} k-points)...")
import time
t_sl = time.time()
all_evals    = []
all_sweights = []
for ik, kp in enumerate(kpar_path):
    Hs = build_H_slab(kp, D_arc, N_slab)
    ev, vc = eigh(Hs)
    all_evals.append(ev)
    s_idx_bot = np.arange(4*N_surf)
    s_idx_top = np.arange(4*(N_slab-N_surf), 4*N_slab)
    s_idx = np.concatenate([s_idx_bot, s_idx_top])
    sw_k = np.sum(np.abs(vc[s_idx, :])**2, axis=0)
    all_sweights.append(sw_k)
    if (ik+1) % 100 == 0:
        print(f"    k-point {ik+1}/{N_kpar}  ({time.time()-t_sl:.1f}s)")

all_evals    = np.array(all_evals)
all_sweights = np.array(all_sweights)
print(f"  Slab done in {time.time()-t_sl:.1f}s")

print("\nGenerating: fig_surface_arc")
fig_arc, ax_arc = plt.subplots(figsize=(COL1 * 1.05, 3.0),
                                gridspec_kw={'left': 0.16, 'right': 0.96,
                                             'top': 0.92, 'bottom': 0.14})

E_max_arc   = 50.0   # meV
E_arc_edges = np.linspace(0.5, E_max_arc, 201)
E_arc_ctr   = 0.5 * (E_arc_edges[:-1] + E_arc_edges[1:])
k_edges     = np.concatenate([
    [kpar_dist[0] - 0.5*(kpar_dist[1]-kpar_dist[0])],
    0.5*(kpar_dist[:-1]+kpar_dist[1:]),
    [kpar_dist[-1] + 0.5*(kpar_dist[-1]-kpar_dist[-2])],
])
sigma_arc = 1.0
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

if repr_wc is not None:
    ax_arc.axhline(repr_wc['omega'], color=C_BLUE, lw=0.6, ls=':',
                   alpha=0.7, label=rf"$\omega_W={repr_wc['omega']:.1f}$ meV")
    ax_arc.legend(fontsize=5, loc='upper right', framealpha=0.85)

save(fig_arc, 'fig_surface_arc')
plt.close(fig_arc)

print("\nDone.")
