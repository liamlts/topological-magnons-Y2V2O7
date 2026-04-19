# Paper: Topological Magnon Band Structure in Pyrochlore Magnets

## Figure generation

### LSWT figures (no EDRIXS needed)
```bash
cd paper/
/opt/miniconda3/envs/edrixs_run/bin/python generate_figures.py
```
Produces in `Figures/`:
- `fig_comparison_progression.pdf` — S(q,ω) D/J progression strip (6 panels, a–f)
- `Y2V2O7_band_overlay_DJ.pdf` — all D/J overlaid on one axis
- `fig_weyl_kp_analysis.pdf` — t_W, ω_W, v_W vs D/J (3 panels, a–c)
- `fig_weyl_band_zoom_GL.pdf` — Γ→L zoom with k·p fits (panels a–f)
- `fig_weyl_cone_3D.pdf` — 3D Weyl cone surface plot
- `fig_weyl_cone_cuts.pdf` — parallel + perpendicular cone cuts (panels a, b)

### EDRIXS V-V dimer figure
```bash
/opt/miniconda3/envs/edrixs_run/bin/python generate_dimer.py
```
Produces:
- `fig_vv_dimer.pdf` — 3-panel dimer RIXS figure (panels a, b, c)

## Figures still needed (external tools)
- `Figures/betterpyro.pdf` — pyrochlore crystal structure (VESTA or CrystalMaker)
- `Figures/pyrochlore_bz.pdf` — FCC BZ with Weyl points (adapted from Mook 2016)

## Panel label convention
All figures use **Nature Physics style**: bold `a.` at top-left of each panel
(not `(a)` with parentheses).

## Key physical results
- Weyl crossings detected for D/J ∈ [0.10, 0.80] along Γ→L
- Band 1–2 (acoustic-optical) crossing: appears at t_W ≈ 0.3–0.9
- k·p fit and Löwdin projection agree within 5–10%
- SOC excitation at 45.7 meV confirmed in spin-flip RIXS channel (EDRIXS)
- Phonon at 100 meV cleanly separated from magnetic excitations by polarimetry

## Robustness notes
- Weyl crossing detection: 5000-point fine grid, gap threshold 0.05 meV
- k·p fit window: ±8% of Γ→L path
- LSWT bands computed for D/J = 0, 0.1, ..., 1.0 (11 values)
- Broadening: 1.5 meV FWHM Gaussian throughout
- T = 5 K Bose factor included in S(q,ω)
