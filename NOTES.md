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

## Fresh-context audit (subagent, before stage 4 ran)

An independent fresh-context agent re-derived the numerics with its own
implementations at N=24. **No confirmed math errors**: RK4 in-place
algebra (order 4.01 measured), rfft Parseval weights (3e-16), curl and
projection signs (9e-16), dealias-cube invariance through forced RK4
steps (exactly 0 outside the cube), exact-P forcing injection
(0.300000000000000, 1.9e-16), snapshot pack/unpack negative-k mapping
(bit-exact roundtrip), and the stage-1 arithmetic (1.5887, Re_λ 73.2)
all verified. Findings, all minor and fixed before stage 4 ran:
- stage 4 used unweighted sample means over adaptively-spaced steps →
  switched to time-weighted (trapezoid) means, dt-weighted drift fit, and
  equal-*time* blocks.
- IC normalized the model spectrum before truncating at kc (negligible
  here, latent elsewhere) → normalize over retained shells.
- precision preflight computed dt separately per precision (1e-7 relative
  mismatch, contrary to the code comment) → dt now shared.
- 7-point slope fit had no uncertainty → added per-snapshot slope spread.
- Disclosed, unchanged: snapshot 0 is taken exactly at the spin-up
  boundary (least-decorrelated sample; downstream can drop it), and the
  k_max·η margin over the 1.5 gate is thin if ⟨ε⟩ overshoots P.

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

## Mid-run reality check (at spin-up end, t=13.9)

Instantaneous state from the checkpoint: E=1.05, ε=0.295≈P ✓,
k_max·η=1.59 ✓, but **Re_λ=54, not the predicted 73** — the stage-1
assumptions (L≈1.4, C_ε≈0.45) were optimistic; measured L=1.28,
C_ε≈0.64. Cross-check against literature: standard 128³ runs quoting
Re_λ≈70 (e.g. Gotoh et al. 2002) resolve only k_max·η≈1.0; insisting on
1.5 costs a factor (1.5/1.0)^(2/3)≈1.3 in Re_λ → ≈54. Our number is
*consistent*, not a solver problem.
- Consequence: the spectrum has **no constant-slope segment**. Local
  log-log slope: −1.75 over k∈[3,8], −2.75 over k∈[4,10] (the a priori
  gate band, which extends into the dissipation roll-off). Compensated
  magnitudes over k∈[3,9] are 1.3–2.5 ≈ C_K — Kolmogorov-like amplitude,
  but the −5/3±0.25 gate on k∈[4,10] will very likely FAIL. Stage 4 now
  reports slopes over several bands to make the band-sensitivity explicit;
  the a priori gate band is kept as the gate (no goalpost-moving).
- Escalation analysis using *measured* u′ (not stage-1 assumptions):
  192³ at matched k_max·η gives Re_λ≈71 for ~9 h — poor value. Forcing
  only 0<|k|≤1.5 at 128³ raises L→≈2.3, Re_λ→≈80 for ~3 h at the same
  k_max·η, at the cost of box/L≈2.7 (large-scale confinement — accepted
  practice in classic intermittency DNS, e.g. Vincent & Meneguzzi). That
  is the sensible escalation if the gate fails; larger L also widens the
  physical-space inertial window (L/η 34 → 61), which is what the
  downstream structure-function phase actually needs.
- Decision: let the current run finish (valid stationary dataset either
  way), apply the gate honestly, then escalate.

## 2026-07-03 — Run 1 (k_f=2.5) verdict: stationary and resolved, −5/3 gate FAILED

Full results: artifacts/run1_kf2.5/resolution_validation.json.
- Stationarity PASSED: 39 turnovers sampled, time-weighted ⟨ε⟩/P−1 =
  **+0.4%**, E-drift 1.8% of mean over the window, 4-block ε means within
  ±3% of P, component anisotropy ≤ 4.4%. This is an explicit diagnostic,
  not "it looked flat".
- Resolution PASSED: k_max·η = 1.587 (measured ε).
- Spectrum gate FAILED: slope −2.70±0.02 on the a priori band k∈[4,10]
  vs target −5/3±0.25. Band sweep: −1.55 (k3–6, forcing-contaminated),
  −2.01 (k3–8), −3.07 (k5–12). Measured Re_λ = 55.9, L=1.29, u′=0.85.
- Consecutive-snapshot whole-field correlation ρ ≈ 0.37 at 1 T_eddy
  spacing (large-scale dominated), consistent with "approximately
  independent" sampling.
- **Root cause, now demonstrated not just suspected**: k_max·η ≥ 1.5 at
  128³ fixes η = 0.038, so the dissipative roll-off begins at
  k ≈ 0.13/η ≈ 3.4, while forcing influence extends to k ≈ 2·k_f ≈ 5.
  The window where −5/3 could live, [2k_f, 0.13/η], is empty. The two
  requirements (k_max·η ≥ 1.5 at 128³ and a visible −5/3 range) are
  mutually incompatible in this box regardless of forcing details.
  Escalations that fix it genuinely (256³ at k_f≤1.5: Re_λ≈125, real
  short inertial range, k_max·η=1.5) cost ~2 days of wall time here.
  128³ with k_f≤1.5 keeps all constraints and raises Re_λ to ≈80,
  L/η 34→61, but the defensible-band slope is still projected ≈ −1.75
  to −1.95 (edge of tolerance) — better dataset, uncertain gate.

## Run 2 launched (k_f ≤ 1.5, 48 snapshots)

Attempted to put the fork (accept run 1 / k_f=1.5 rerun / relax k_max·η /
256³) to the user; the question tool failed with a transport error and the
user said to continue, so proceeded with the recommended option — the only
one that keeps every stated constraint while materially improving the
dataset. Changes for run 2, all made *before* launching:
- k_force 2.5 → 1.5 (18 forced modes, k²∈{1,2}); ic_kp 2.5; seed unchanged.
- 48 snapshots instead of 40: box/L drops to ≈2.7, so each snapshot holds
  ~20 (not ~100) independent integral volumes; 48×20 ≈ 10³ independent
  large-scale samples. Box confinement at box/L≈2.7 is the standard cost
  of forcing at the lowest shells (cf. classic intermittency DNS); it
  mainly degrades statistics at separations r ≳ L, which downstream
  should avoid anyway.
- First snapshot now one spacing after the spin-up boundary.
- A priori spectrum gate band for run 2, declared before the run: k∈[3,6]
  (above forcing influence 2k_f=3; kη ≤ 0.23). Multi-band slopes still
  reported. Updated feasibility (measured C_ε=0.64, L≈2.3): Re_λ ≈ 81,
  k_max·η = 1.589, strict −5/3 window [3, 4.0] — still nearly empty, so
  the honest expectation remains a marginal gate outcome.

## INCIDENT: forced-divergence instability (run 2, t ≈ 70–88) — found, diagnosed, fixed

Run 2 was statistically stationary through τ≈46 (E≈1.25–1.45, ε≈0.3,
budget closed to 0.7%), then energy grew monotonically to E=4.2 by t=87
with implied injection up to 3.6×P — impossible for fixed-power forcing
in exact arithmetic, so a numerical energy source.

Diagnosis chain (all measured, in artifacts/… and this repo's history):
1. Budget decomposition at the runaway state (float64): forcing injection
   exactly 0.300 ✓, viscous −ε ✓, but nonlinear "conservation" sum
   = **+0.648** — the rotational-form identity Σ c*·P[F(u×ω)] = ∫u·(u×ω)=0
   holds only for exactly solenoidal states.
2. Re-running 150 steps from the runaway checkpoint in float64 and at
   dt/2 reproduced the residual to 5 digits → not precision, not
   time-integration: the implemented RHS at that state creates energy.
3. The state had acquired a large divergent component and a mean flow
   |U|≈1.5: divergence energy across snapshots grew exponentially,
   2.0e-12 (snap 0) → 1.2 (snap 47), rate ≈ α = P/(2E_f).
   Mechanism: RK4's floating-point roundoff seeds a divergent component;
   the state is never re-projected; fixed-power forcing f = α·û_band
   amplifies the *full* band amplitude — divergent part included — at
   net rate α − νk² > 0. From float32 seed (~1e-7) this reaches O(1) at
   t≈70, exactly as observed. float64 would only delay onset (~t≈150);
   run 1 ended (59 τ) before its latent instability emerged (its
   snapshots are unaffected at their timescale).
4. Fix: re-project the state and zero the k=0 mode after every RK4 step
   (enforcing invariants of the continuous system); divergence monitor +
   abort in stage 3; new stage-2 regression test (3000-step forced
   float32 run must keep E_div/E < 1e-9 — passes with the fix; the
   production run itself is the demonstration that it fails without).
5. Data handling: snapshots 0–25 verified clean (E_div/E ≤ 5e-5 at snap
   25, ~1e-9 at snap 10); snapshots 26–47 deleted as contaminated;
   time series truncated to t ≤ 58.24; production resumed from snapshot
   25's re-projected state with the fixed solver to collect the
   remaining 22 snapshots. The splice is documented here and in
   PROVENANCE.md; pre- and post-splice samples are the same physical
   trajectory up to removal of an O(5e-5) spurious component.
