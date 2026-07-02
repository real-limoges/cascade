# Running notes — pseudo-spectral DNS of homogeneous isotropic turbulence

Chronological working notes for the solver-build-and-validate phase.
Structure functions / scaling exponents are explicitly out of scope here.

## 2026-07-02 — setup

- **Template discrepancy**: the task description says PROVENANCE.md and
  artifacts/SCHEMA.md exist as skeletons in this repo. They do not — the
  initial commit contains only LICENSE (checked `git ls-tree` on both
  branches). Created both files from scratch instead.
- Environment: 4 cores, 15 GB RAM, Python 3.11. Installed numpy 2.4.6,
  scipy 1.17.1, pyfftw 0.15.1. pyfftw with FFTW_MEASURE: 4.7 ms per 128^3
  r2c transform (scipy: 24 ms) → pyfftw chosen, scipy fallback kept.

## Design decisions

- **Formulation**: rotational form u×ω with Leray projection; with 2/3-rule
  cube truncation (kc = N//3) the dealiased nonlinear term is an exact
  Galerkin truncation, so it conserves energy to roundoff — used as a
  correctness gate, and it means the discrete energy balance
  dE/dt = inj − ε is exact for the truncated system.
- **Time stepping**: classical explicit RK4, adaptive dt from
  CFL·dx/max(|u₁|+|u₂|+|u₃|) with CFL=0.8. Viscous term explicit — at
  production parameters ν·(3kc²)·dt ≈ 0.25, far inside the RK4 real-axis
  stability bound (~2.79), so an integrating factor is unnecessary
  complexity. compute_dt() enforces the viscous bound anyway.
- **Forcing**: deterministic fixed-power forcing f̂ = (P/2E_f)·û on
  0 < |k| ≤ 2.5. Injects energy at *exactly* rate P every instant, so
  stationarity ⟺ ⟨ε⟩ = P — an explicit, falsifiable balance used as the
  stationarity diagnostic. Chosen over stochastic (OU) forcing for
  reproducibility and over pure linear (Lundgren) forcing to confine input
  to large scales. Chosen over decaying turbulence because downstream
  structure-function statistics want many statistically identical
  snapshots — decay would make every snapshot a different Reynolds number.
- **Stationarity is the claim to verify, not assume**: production run
  tracks E(t), ε(t) every step; stage 4 tests (a) ⟨ε⟩/P − 1, (b) linear
  drift of E over the sampling window, (c) block-to-block variability.

## Feasibility (stage 1, analytic — artifacts/feasibility.json)

- 128³, ν=0.0085, P=0.3 → η=(ν³/P)^¼=0.0378 (exact if stationary),
  k_max·η = 42·0.0378 = **1.59 ≥ 1.5** ✓.
- Predicted Re_λ ≈ 73 (using ε=C_ε u′³/L, C_ε=0.45, L≈1.4). **Honest
  limitation**: at Re_λ~73 there is no asymptotic inertial range — expect
  a <half-decade approximate −5/3 band around k∈[4,10], flattered by the
  spectral bottleneck. 192³/256³ at the same k_max·η would give Re_λ≈97/117
  and only ~0.2/0.3 decades — not worth 5–10× wall time on 4 cores. Stayed
  at 128³ per the task's guidance; the −5/3 gate is therefore evaluated on
  a short scaling range and reported as such.

## Sanity validation (stage 2 — artifacts/sanity_checks.json)

All passed (float64):
- Single-mode exact viscous decay u=sin(2z)x̂: rel. error 1.7e-12 vs
  exp(−νk²t) (RK4 time-integration error only).
- Nonlinear-term energy conservation: relative transfer 5e-17 (roundoff).
- Unforced 64³ decay (ν=0.02): |ΔE + ∫ε dt|/E₀ = 8.1e-5 (< 1e-3 gate,
  limited by trapezoid quadrature of the diagnostic, not the solver);
  E(t) strictly monotonic; max|∇·u| ~ 1e-15 (normalized).

## Performance

- Naive numpy implementation: 1.55 s/step at 128³ (memory-bound temporaries).
  Rewrote hot path: in-place ops, FFT plans bound to aligned buffers
  (FFTW multi-D c2r destroys input → state copied to scratch only there),
  forcing via precomputed band indices. float32 production mode → 0.58
  s/step. Production run ~1.5–2 h wall.
- float32 justified by stage-3 preflight: 25 RK4 steps from identical IC in
  float32 vs float64 must agree in E and ε to < 1e-4 (see
  artifacts/precision_preflight.json). Sanity stage remains float64.

## Production run (stage 3)

- Launched 128³, ν=0.0085, P=0.3, seed 20260702; spin-up 10 large-eddy
  turnovers (τ = ∫dt/T_eddy with T_eddy = L/u′ updated every 50 steps),
  then 40 snapshots at 1.0 T_eddy spacing.
- **Snapshot-count reasoning**: at Re_λ~73, L≈1.3 in a 2π box, each
  snapshot holds ≈(2π/L)³ ≈ 100 quasi-independent integral-scale volumes;
  40 snapshots spaced ≥1 turnover ≈ 4000 independent large-scale samples
  and >10⁸ point pairs per separation — adequate for structure functions
  up to order ~6 at this Re; low orders converge much earlier. 1-turnover
  spacing is the standard "approximately independent" criterion (velocity
  autocorrelation time ≈ T_eddy).
- Snapshots stored as compact dealiased spectral coefficients (complex64,
  ~7.5 MB each vs 25 MB physical float32) — lossless w.r.t. the dealiased
  state at float32 precision; loader provided in dns/snapshot_io.py.
