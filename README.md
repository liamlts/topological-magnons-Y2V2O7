# Topological Weyl Magnons in Y₂V₂O₇: Code and Data

Code repository for the paper:
**"Topological Weyl Magnons in Y₂V₂O₇: Polarimetric RIXS Signatures and Thermal Hall Response"**

## Overview

This repository contains all computational scripts to reproduce the figures in the manuscript.
Two independent calculations are performed:

1. **Linear Spin-Wave Theory (LSWT)** — Magnon band structure, Berry curvature, Chern numbers,
   surface arcs, phase diagram, and thermal Hall conductivity for the pyrochlore ferromagnet.
2. **Cluster Exact Diagonalization (EDRIXS)** — Resonant inelastic X-ray scattering cross-sections
   for single-site V⁴⁺ and two-site V⁴⁺–V⁴⁺ dimer models using the Kramers–Heisenberg formalism.

## Requirements

```
Python >= 3.9
numpy >= 1.22
scipy >= 1.8
matplotlib >= 3.5
edrixs >= 0.0.4   (for RIXS calculations only)
```

Install via:
```bash
conda create -n topo_magnon python=3.10 numpy scipy matplotlib
conda activate topo_magnon
pip install edrixs
```

Or use the provided `environment.yml`:
```bash
conda env create -f environment.yml
conda activate topo_magnon
```

## Scripts

| Script | Description | Figures produced |
|--------|-------------|-----------------|
| `generate_figures.py` | LSWT band structure, Weyl cones, Berry curvature, phase diagram, surface arcs, magnon Hall | `fig_comparison_progression`, `Y2V2O7_band_overlay_DJ`, `fig_weyl_kp_analysis`, `fig_weyl_band_zoom_GL`, `fig_weyl_cone_3D`, `fig_weyl_cone_cuts`, `fig_berry_chern`, `fig_phase_diagram`, `fig_surface_arc`, `fig_magnon_hall` |
| `regen_berry_arc.py` | Standalone Berry curvature + surface arc (faster rerun) | `fig_berry_chern`, `fig_surface_arc` |
| `generate_dimer.py` | Single-site V⁴⁺ EDRIXS + phenomenological dimer RIXS | `fig_vv_dimer` |
| `generate_dimer_full_rixs.py` | Full two-site cluster ED RIXS (Kramers–Heisenberg) | `fig_dimer_full_rixs`, `fig_rixs_2d_map` |
| `generate_xas_overview.py` | V L₂,₃-edge XAS with σ and π polarizations | `fig_xas_overview` |

## Running

Generate all figures:
```bash
python generate_figures.py          # ~2 min (LSWT, no GPU)
python generate_xas_overview.py     # ~5 s
python generate_dimer.py            # ~10 s
python generate_dimer_full_rixs.py  # ~5 min (large ED)
```

All output goes to `Figures/`.

## Physical Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| J (NN exchange) | 8.22 meV | INS, Lu₂V₂O₇ (Mena et al. 2014) |
| D/J (DM ratio) | 0.32 | Moriya theory estimate |
| S | 1/2 | V⁴⁺ (d¹) |
| a (lattice constant) | 9.89 Å | XRD |
| 10Dq | 1.9 eV | V L-edge XAS fitting |
| ζ₃d | 30 meV | Atomic tables |
| ζ₂p | 4.65 eV | Atomic tables |

## Citation

If you use this code, please cite the associated paper.

## License

MIT License. See `LICENSE` for details.
