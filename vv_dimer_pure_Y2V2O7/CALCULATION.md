# How the V–V dimer RIXS calculation works

This document explains what `generate_dimer_pure.py` actually computes and
how it differs from the simpler edrixs workflows people usually start with
(single ion with crystal field, and Anderson-impurity / DMFT-style models).

## 1. What we are modelling

A single V⁴⁺–V⁴⁺ bond from the Y₂V₂O₇ pyrochlore.  Two V ions carry one 3d
electron each (nominal 3d¹ per site, Jeff=½ moment at low T) and are coupled
by superexchange J through the intervening oxygen.  We simulate V L₃-edge
RIXS (~515 eV) in exact diagonalisation.

Two electrons, two sites, real Slater–Coulomb, real 2p–3d dipole matrix
elements, and Kramers–Heisenberg scattering.  No model spin Hamiltonian, no
projected pseudospins, no phenomenological Gaussian at ΔE = J.

### Parameters (Y₂V₂O₇-like, can be tuned at the top of the script)

| symbol         | value      | meaning                                        |
|----------------|------------|------------------------------------------------|
| ζ_d            | 30 meV     | V 3d spin–orbit coupling                       |
| Δ_trig         | 10 meV     | D₃d trigonal splitting of t₂g (a₁g vs eg′)    |
| U              | 3.0 eV     | on-site Hubbard U (V)                          |
| J_H            | 0.68 eV    | Hund's coupling (V 3d)                         |
| F², G¹, G³ (pd)| edrixs × 0.8 | 2p–3d Slater integrals (atomic × 80 %)      |
| U_dc           | 4 eV       | static 2p core-hole → 3d attraction            |
| t              | 77 meV     | inter-site diagonal t₂g–t₂g hopping            |

`t` is the only inter-site term.  The singlet–triplet splitting J is
**not** put in by hand — it emerges as the eigenvalue of the full cluster
Hamiltonian and comes out near 4t²/U ≈ 8 meV (with SOC-mixed orbital
content pulling the true dimer gap to ~5 meV when Δ_trig is small).

## 2. Structure of the script

The layout mirrors the official edrixs two-site example
`examples/more/RIXS/Ba3InIr2O9/t2g_two_site_cluster/`
(`run_ed.py` + `run_rixs.py` merged into one file).

1. **Build Slater–Coulomb tensors** in the real cubic basis
   * `edrixs.get_umat_slater('t2g', ...)` for intra-site d–d
   * `edrixs.get_umat_slater('t2gp', ...)` for p–d in the core-hole state
   * `edrixs.transform_utensor(..., tmat_c2r(...))` to the real basis
2. **Embed** into 24-orbital arrays `umat_i` and `umat_n[s]`
   * layout: `[0:6]` t₂g site A, `[6:12]` t₂g site B, `[12:18]` 2p site A,
     `[18:24]` 2p site B; `indx[s]` maps the 12-orbital intra-site index
     into the 24-orbital global index for site s
3. **Build the one-body matrix** `emat_i`
   * d-SOC from `edrixs.atom_hsoc('t2g', ζ_d)` (rotated with `cb_op`)
   * trigonal: H_trig = (Δ/3)(J₃−I₃) gives a₁g at +2Δ/3, eg′ at −Δ/3
   * inter-site diagonal hopping −t between matching orbitals on A/B
4. **Core-hole variants** `emat_n[s]`, `umat_n[s]` add
   * 2p SOC on the core-hole site only (huge ζ_p ≈ 4.6 eV splits L₃/L₂)
   * p–d Slater integrals on that site
   * static 2p-hole attraction `−U_dc` on the local d orbitals
5. **Fock bases** with `get_fock_bin_by_N`
   * initial   : `C(12, 2) × C(12, 12)   = 66` configurations
   * intermed. : `C(12, 3) × C(6, 5) × C(6, 6) = 1320` per core-hole site
6. **Exact diagonalisation** with `scipy.linalg.eigh`
7. **Transition operators** from `edrixs.get_trans_oper('t2gp')`, rotated
   to the real cubic basis and sandwiched between ED eigenvectors with
   `edrixs.cb_op2` (gives `trop_n[s, j, ncfgs_n, ncfgs_i]`, Cartesian j ∈
   {x, y, z})
8. **XAS** computed directly from `trop_n` with a polarization vector
   `e_in`, summed over ground-state ensemble (Boltzmann)
9. **RIXS** via `edrixs.scattering_mat` (Kramers–Heisenberg sum over
   intermediate states); outer products with Cartesian `e_in`, `e_out`
   vectors to project any linear polarization channel
10. **Figure** — XAS (LV, LH), resonant RIXS with sticks, and 2D map

## 3. What is NOT in the calculation

- No phenomenological "exchange Gaussian" at ΔE = J
- No phonon peak (no electron-lattice coupling in the Hamiltonian)
- No ad-hoc pseudospin projection — Jeff=½ character is only what the
  trigonal+SOC eigenvectors actually produce
- No V–O hybridisation as an explicit bath (charge-transfer states are
  not included; the small core-hole excitation does not couple to O 2p)
- No long-range order or magnetic structure; the dimer is an isolated cluster

## 4. How this differs from a single-ion calculation

The standard edrixs single-ion workflow (e.g. `ed_1v1c_py`,
`rixs_1v1c_py`) solves one V⁴⁺ atom with 10-orbital d shell + 6 (or 4)
orbital 2p shell:

- Hilbert space: `C(10, 1) × C(6, 6) = 10` initial, `C(10, 2) × C(6, 5) =
  270` intermediate — trivial.
- Physics captured: Hund's rules, crystal field, spin–orbit, multiplets,
  intrinsic d-d excitations, the atomic L₃ lineshape.
- Physics **not** captured: anything involving two ions.  In particular
  there is no singlet–triplet splitting, no magnetic exchange, no
  superexchange peak in RIXS.  Spin-flip RIXS in a single-site model can
  only flip an isolated spin to an essentially degenerate partner; it
  cannot resolve the magnon energy, because no inter-site coupling exists.

A two-site cluster is the **minimum** model that produces a
singlet–triplet splitting.  This is why, to get a real "magnon-like"
peak at J meV from exact diagonalisation, you have to build the dimer.

The dimer is still small enough that everything diagonalises on a laptop
in ~20 s, which makes it ideal for exploring Y₂V₂O₇-style physics before
committing to bigger calculations (tetrahedron, Anderson impurity, DMFT).

## 5. How this differs from a single-impurity Anderson model

edrixs also supports impurity calculations where one correlated V site is
embedded in a non-interacting bath of discrete bath levels that represent
the rest of the crystal (`fit_hyb.py`, `solvers.py`, `ed_*_fort.py`).
That class of calculation captures:

- Charge-transfer and screening (V 3d ↔ O 2p covalency)
- Realistic core-hole screening → non-Lorentzian L₃ lineshape
- Feshbach-like resonances and back-action of the bath on final states

and is typically solved with Lanczos / Krylov / parallel Fortran solvers.

It does **not** natively give you the spin dynamics of the neighbour
V ions, because the bath in an impurity model is an **effective**
hybridisation mean field — the dimer partner is replaced by a sum of
non-interacting orbitals fit to a hybridisation function Δ(ω).  You
recover dynamical single-site spectra (XAS, RIXS on-site d-d) very
accurately, but you lose the coherent two-spin singlet/triplet physics
that lives in the exact dimer.

Our calculation is the opposite trade-off:

| aspect                    | single ion | 2-site cluster (this)  | Anderson impurity |
|---------------------------|:----------:|:----------------------:|:-----------------:|
| on-site multiplets        | yes        | yes                    | yes               |
| intrinsic d-d excitations | yes        | yes                    | yes               |
| intersite magnon / J peak | no         | yes                    | no (mean field)   |
| charge-transfer screening | no         | no                     | yes               |
| system size scaling       | trivial    | `C(12,2)·C(12,12) = 66`| large (bath fit)  |
| realistic L₃ lineshape    | partial    | partial                | yes               |

## 6. Polarization conventions used in the figure

- **Scattering geometry**: horizontal scattering plane (xz), sample surface
  in xy, `thin = 15°` from surface, `thout = 135°` from surface
  (2θ = 150°).
- **LV (σ)**: electric field perpendicular to the scattering plane,
  e = ŷ = (0, 1, 0).  Same vector for in and out.
- **LH (π)**: electric field in the scattering plane, perpendicular to k.
  For in and out we build `ki × ŷ` and `kout × ŷ`, normalised.
- "Isotropic out" in panel b means the outgoing *intensity* is summed
  incoherently over two orthogonal outgoing polarisations (LV_out,
  LH_out) — this is what you get from a standard RIXS detector that
  does not resolve outgoing polarization.

## 7. Reading the figure

- **Panel a** — XAS with LV and LH incidence from 512 to 517 eV.  The
  small linear dichroism (LV vs LH) is a consequence of the trigonal
  D₃d field on the final-state multiplets.
- **Panel b** — RIXS at resonance (E_res ≈ 515.5 eV).  The blue / red
  curves are resolution-broadened (30 meV FWHM) spectra for LV and LH
  incidence; the sticks above show the 66 unbroadened final-state
  amplitudes (elastic stick omitted so the inelastic structure is
  visible).  The J label sits on the lowest inelastic cluster — the
  triplet.  Higher groups are d-d excitations driven by SOC + trigonal.
- **Panel c** — 2D RIXS map over 512 → 517 eV with LV+LH incidence and
  isotropic out.  `E_res` is the vertical dashed line; the J
  horizontal stripe is faint and sits just above the elastic band,
  while the SOC d-d band near 45–70 meV is prominent and tracks the
  L₃ resonance.

## 8. Knobs worth turning

| If you want…                         | change                              |
|--------------------------------------|-------------------------------------|
| larger J                             | increase `t_hop` (≈ √(J·U/4))       |
| pure Jeff=½ ground state             | `zeta_d_i = 0.03` with `delta_trig → 0` |
| stronger trigonal (more dichroism)   | raise `delta_trig` back toward 30 meV |
| L₂ edge as well                      | widen `ominc_rel` and MAP_EMAX past 520 eV |
| different colorimeter                | `local_axis` rotation in `dipole_polvec_rixs` |
| tighter RIXS resolution              | reduce `res_FWHM`                    |
