# Provenance

Phase 1 of the turbulence-cascade project: build and validate a
pseudo-spectral DNS of forced homogeneous isotropic turbulence and produce
independent velocity snapshots for downstream structure-function analysis.
Structure functions, scaling-exponent fits, and phenomenology comparisons
are explicitly **not** part of this phase.

> Note: the task brief said this file existed as a template skeleton; the
> repo's initial commit contained only LICENSE, so it was written from
> scratch (see NOTES.md).

## Code

- Repository: real-limoges/cascade, branch `claude/turbulence-dns-solver-tu0n0x`
- Code version used for the production run: see `git_sha` in
  `artifacts/production_summary.json` and in every snapshot's metadata
  (TO FILL after run completes: <sha>)
- Solver: `dns/` — incompressible Navier-Stokes in Fourier space on a
  [0,2π)³ periodic box; rotational form u×ω with Leray projection;
  2/3-rule cube dealiasing (kc = N//3); classical explicit RK4 with
  adaptive CFL time step (CFL = 0.8, viscous bound enforced);
  deterministic fixed-power forcing f̂ = (P/2E_f)û on 0 < |k| ≤ 2.5.
- Stages (each re-runnable from the repo root):
  1. `python -m stages.stage1_feasibility` — a priori resolution check
  2. `python -m stages.stage2_sanity` — solver validation vs known behavior
  3. `python -m stages.stage3_production` — forced run + snapshots
     (resumes from `data/checkpoint.npz` if present)
  4. `python -m stages.stage4_validate` — stationarity/resolution/spectrum
     gates → `artifacts/resolution_validation.json`

## Physical and numerical parameters (production)

| parameter | value |
|---|---|
| grid | 128³, box [0,2π)³ |
| dealiased max wavenumber kc | 42 (2/3 rule) |
| viscosity ν | 0.0085 |
| forcing | fixed power P = 0.3 on 0 < \|k\| ≤ 2.5 (⟨ε⟩ = P at stationarity) |
| time stepping | RK4, adaptive dt = 0.8·Δx/max(\|u₁\|+\|u₂\|+\|u₃\|) |
| initial condition | random solenoidal field, E(k) ∝ k⁴exp(−2(k/k_p)²), E₀=1.5, k_p=3, seed 20260702 |
| spin-up discarded | 10 large-eddy turnovers (τ = ∫dt/T_eddy, T_eddy = L/u′) |
| snapshots | 40, spaced 1.0 T_eddy |
| precision | float32 (validated against float64: `artifacts/precision_preflight.json`) |

## Measured results (stage 4 gate)

TO FILL after run completes — Re_λ, k_max·η, ⟨ε⟩/P, spectrum slope,
validation verdict. Source of truth: `artifacts/resolution_validation.json`.

## Environment

- Linux 6.18.5 container, 4 CPU cores, 15 GB RAM
- Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pyfftw 0.15.1 (FFTW,
  FFTW_MEASURE plans, 4 threads), matplotlib for diagnostic plots
- Install: `pip install numpy scipy pyfftw matplotlib`

## Reproducibility

- All stage parameters live in `dns/params.py`; the IC seed is fixed.
  Bitwise reproducibility is not guaranteed across FFTW builds/thread
  counts (floating-point reduction order), but all validation gates are
  statistical and robust to that.
- The snapshot dataset (`data/snapshots/`, ~300 MB) is committed to the
  repo because the compute container is ephemeral; format documented in
  `artifacts/SCHEMA.md`, loader in `dns/snapshot_io.py`.
