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
- Solver: `dns/` — incompressible Navier-Stokes in Fourier space on a
  [0,2π)³ periodic box; rotational form u×ω with Leray projection;
  2/3-rule cube dealiasing (kc = N//3); classical explicit RK4 with
  adaptive CFL time step (CFL = 0.8, viscous bound enforced);
  deterministic fixed-power forcing f̂ = (P/2E_f)û on 0 < |k| ≤ 1.5;
  state re-projected and k=0 mode zeroed every step (see "Incident").
- Stages (each re-runnable from the repo root):
  1. `python -m stages.stage1_feasibility` — a priori resolution check
  2. `python -m stages.stage2_sanity` — solver validation vs known behavior
  3. `python -m stages.stage3_production` — forced run + snapshots
     (resumes from `data/checkpoint.npz` if present)
  4. `python -m stages.stage4_validate` — stationarity/resolution/spectrum/
     solenoidality gates → `artifacts/resolution_validation.json`
- Code versions for the validated dataset: snapshots 0–25 were produced by
  the solver at commit `14676a8` (pre-fix; verified clean, E_div/E ≤ 5e-5);
  snapshots 26–47 and the final validation at commit `4b6447d`
  (divergence fix). Every snapshot's metadata records its `git_sha`.

## Physical and numerical parameters (validated production run, "run 2")

| parameter | value |
|---|---|
| grid | 128³, box [0,2π)³ |
| dealiased max wavenumber kc | 42 (2/3 rule) |
| viscosity ν | 0.0085 |
| forcing | fixed power P = 0.3 on 0 < \|k\| ≤ 1.5 (18 modes; ⟨ε⟩ = P at stationarity) |
| time stepping | RK4, adaptive dt = 0.8·Δx/max(\|u₁\|+\|u₂\|+\|u₃\|), state re-projected each step |
| initial condition | random solenoidal field, E(k) ∝ k⁴exp(−2(k/k_p)²), E₀=1.5, k_p=2.5, seed 20260702 |
| spin-up discarded | 10 large-eddy turnovers (τ = ∫dt/T_eddy, T_eddy = L/u′) |
| snapshots | 48, spaced 1.0 T_eddy, first at τ = 11 |
| precision | float32 (validated against float64: `artifacts/precision_preflight.json`) |

## Measured results (stage 4 gate — artifacts/resolution_validation.json)

All four gates **passed**; sampling window = 48.0 large-eddy turnovers:

| quantity | value |
|---|---|
| ⟨ε⟩/P − 1 (time-weighted) | −0.22% (gate: \|·\| ≤ 10%) |
| energy drift over window | 4.6% of ⟨E⟩ (gate ≤ 10%) |
| k_max·η (measured ε) | 1.590 (gate ≥ 1.5) |
| spectrum slope, a priori band k∈[3,6] | −1.612 ± 0.03 SEM (gate −5/3 ± 0.25) |
| max snapshot E_div/E | 4.6e-5 (gate < 1e-4) |
| Re_λ | 69.1 |
| E, u′, L, T_eddy | 1.350, 0.949, 1.549, 1.633 |
| η, λ_Taylor | 0.0378, 0.619 |

**Honest caveats** (also inside the artifacts): at Re_λ ≈ 69 the −5/3 band
is a short approximate scaling range (< half a decade, k∈[3,6]), partially
supported by the spectral bottleneck — not an asymptotic inertial range;
the spectrum is visibly curved (slope −2.05 over k∈[3,8], −2.69 over
k∈[4,10]). Component anisotropy of the energy is up to 13% because only
18 large-scale modes are forced. Consecutive snapshots retain whole-field
correlation ≈ 0.39 (large-scale dominated) at 1 T_eddy spacing.

An earlier configuration ("run 1", forcing 0<|k|≤2.5) is archived under
`artifacts/run1_kf2.5/` + `data/snapshots_run1_kf2.5/`: it passed
stationarity and resolution but **failed** the −5/3 gate (slope −2.70,
Re_λ=56) — with k_max·η ≥ 1.5 at 128³ its scale window for inertial
scaling was empty. See `artifacts/run1_kf2.5/divergence_audit.json`
before reusing its snapshots (29/40 clean).

## Incident record (see NOTES.md for the full diagnosis)

The original solver never re-projected the state; floating-point roundoff
seeds a divergent velocity component that fixed-power forcing amplifies
exponentially (rate ≈ P/2E_f). Run 2 was stationary for ~46 turnovers,
then blew up into a spurious-energy runaway. The instability was
diagnosed from the energy budget (implied injection 3.6×P,
precision- and dt-independent), confirmed by exponential E_div growth
(2e-12 → 1.2), and fixed by per-step re-projection + k=0 zeroing, with a
stage-2 regression test and a stage-4 solenoidality gate added.
Contaminated snapshots were deleted; the run resumed from the last
verified-clean snapshot (index 25, re-projected). The dataset therefore
splices at t = 58.24: same physical trajectory, minus an O(5e-5)
spurious component.

## Snapshot dataset (data/snapshots/, committed)

- 48 files `snap_0000.npz … snap_0047.npz`, ~7.5 MB each (~360 MB total):
  dealiased spectral coefficients, complex64 — format in
  `artifacts/SCHEMA.md`, loader `dns.snapshot_io.load_velocity`.
- **Why 48 is enough for downstream structure functions**: box/L ≈ 4.1
  (L=1.55), so each snapshot holds ≈ 65 independent integral-scale
  volumes → ≈ 3100 independent large-scale samples, and >10⁸ point pairs
  per separation for spatial averaging. That is adequate for structure
  functions up to order ~6 at Re_λ≈69 (low orders converge much earlier);
  snapshot-to-snapshot spacing of 1 T_eddy is the standard
  approximate-independence criterion. High-order (p ≳ 8) tails would need
  more data; that is a phase-2 concern.

## Environment

- Linux 6.18.5 container (Claude Code remote), 4 CPU cores, 15 GB RAM
- Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pyfftw 0.15.1 (FFTW,
  FFTW_MEASURE plans, 4 threads), matplotlib for diagnostic plots
- Install: `pip install numpy scipy pyfftw matplotlib`

## Reproducibility

- All stage parameters live in `dns/params.py`; the IC seed is fixed.
  Bitwise reproducibility is not guaranteed across FFTW builds/thread
  counts (floating-point reduction order), but all validation gates are
  statistical and robust to that.
- The snapshot dataset is committed because the compute container is
  ephemeral. Wall cost to regenerate from scratch: ≈ 4 h on 4 cores.
