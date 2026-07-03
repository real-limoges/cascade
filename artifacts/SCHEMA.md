# Artifact schema

All artifacts are produced by the numbered stages under `stages/` (each is
re-runnable via `python -m stages.stageN_<name>` from the repo root).
JSON numbers are SI-free code units: box side 2π, integer wavenumbers,
unit density.

## artifacts/feasibility.json  (stage 1)

A priori resolution / Reynolds-number check for the chosen `(N, nu, P)`.

| field | meaning |
|---|---|
| `chosen` | predictions for the production parameters: `eta_predicted` = (ν³/P)^¼ (exact under stationarity, since fixed-power forcing makes ⟨ε⟩=P), `kmax_eta_predicted` (kc = N//3), `Re_lambda_predicted`, `inertial_range_k` and `inertial_range_decades` (crude bounds: above the forcing band, below kη≈0.15) |
| `alternatives_same_resolution_margin` | same predictions for 192³/256³ at matched k_max·η |
| `gate_kmax_eta` | `{threshold, value, passed}` |
| `honest_summary` | plain-language statement of what this resolution can and cannot show |

## artifacts/sanity_checks.json  (stage 2)

Solver-correctness checks against known behavior, all with explicit
thresholds and `passed` flags: `exact_viscous_decay` (single-mode
exp(−νk²t)), `nonlinear_conservation` (Galerkin energy conservation of the
dealiased rotational nonlinear term), `decay_energy_budget` (unforced 64³:
|ΔE + ∫ε dt|/E₀, monotonicity, normalized max divergence), `all_passed`.

## artifacts/precision_preflight.json  (stage 3)

float32-vs-float64 agreement of E and ε after `nsteps` identical RK4 steps
from the production IC; gates the use of single precision in production.

## artifacts/production_timeseries.npz  (stage 3)

Per-step arrays over the whole run (spin-up + sampling): `t`, `dt`, `E`
(total energy), `eps` (dissipation 2ν∑k²E(k)), `Ef` (energy in forcing
band), `umax` (max |u₁|+|u₂|+|u₃|, CFL input), `tau` (elapsed large-eddy
turnovers ∫dt/T_eddy).

## artifacts/production_summary.json  (stage 3)

Run bookkeeping: `git_sha`, final step/time/τ, `spinup_end` (t, step, τ),
`n_snapshots`, full parameter dict, wall time.

## artifacts/resolution_validation.json  (stage 4) — the validation gate

| field | meaning |
|---|---|
| `stationarity` | sampling-window diagnostics (time-weighted): `eps_over_P_minus_1` (energy balance; gated), `energy_drift_fraction` (dt-weighted linear E-drift over the window / mean; gated), 4 equal-time-block means of E and ε, `half_window_z` (reported) |
| `measured` | snapshot/window measurements: E, ε, u′, η, **kmax_eta** (gated ≥ 1.5), integral scale L, Taylor scale λ, **Re_lambda**, T_eddy, turnovers sampled |
| `spectrum_gate` | log-log slope of snapshot-averaged E(k) over the a priori `fit_band_k` (k∈[3,6] for the k_f=1.5 run); gated at −5/3 ± 0.25; `slope_by_band` shows the band-sensitivity of the curved spectrum; `slope_sem`/`slope_snapshot_std` from per-snapshot fits; `caveat` states the honest extent of the scaling range |
| `isotropy_and_independence` | component-energy fractions; consecutive-snapshot velocity correlation (supports snapshot-independence claim) |
| `snapshot_solenoidality` | max/median E_div/E over snapshots; **gated < 1e-4** (regression guard for the forced-divergence instability — see NOTES.md) |
| `gates`, `validated` | individual gate booleans and their conjunction |

## artifacts/run1_kf2.5/  (archive of the first production configuration)

Same artifact set for run 1 (forcing 0<|k|≤2.5, 40 snapshots): passed
stationarity/resolution, failed the −5/3 gate (slope −2.70, Re_λ=56).
`divergence_audit.json` (added post-hoc) lists per-snapshot E_div/E and
the clean subset (indices 0–28) — its `resolution_validation.json`
predates the divergence discovery; filter before any reuse. Snapshots in
`data/snapshots_run1_kf2.5/`.

## artifacts/energy_spectrum.json  (stage 4)

Snapshot-averaged shell spectrum: `k` (integer shells), `E_k_snapshot_mean`,
`compensated_k53_over_eps23` = E(k)k^{5/3}ε^{−2/3} (index-aligned with `k`;
entry 0 is a placeholder 0 for the k=0 shell).

## artifacts/timeseries.png, artifacts/spectrum.png  (stage 4)

Diagnostic plots: E(t) and ε(t) vs the injection rate with the spin-up
boundary marked; E(k) with the fitted slope and the Kolmogorov-compensated
spectrum.

## data/snapshots/snap_NNNN.npz  (stage 3; committed — the container is
ephemeral, and these are the deliverable dataset for the next phase)

Velocity snapshots for downstream structure-function statistics. Each file:

- `coeffs`: complex64 array, shape `(3, 2·kc+1, 2·kc+1, kc+1)` — the
  dealiased Fourier coefficients c_k (u(x)=Σ c_k e^{ik·x}) in numpy rfft
  ordering restricted to |k_i| ≤ kc (axis 1/2 index j maps to k = j for
  j ≤ kc, k = j − (2kc+1) otherwise; axis 3 is k_z = 0..kc). Lossless
  w.r.t. the float32 simulation state.
- `meta`: JSON string — snapshot index, t, step, τ, N, kc, ν, P, seed,
  git SHA, and instantaneous integral quantities (E, ε, u′, η, L, λ,
  Re_λ, T_eddy, k_max·η).
- The validated run-2 set has 48 snapshots spaced 1.0 T_eddy; indices
  0–25 predate the divergence fix (verified clean, E_div/E ≤ 5e-5),
  26–47 were sampled after resuming from snapshot 25's re-projected
  state (E_div/E ~ 1e-12). The trajectory splices at t = 58.24.

Recover physical velocity with
`dns.snapshot_io.load_velocity(path, N)` → `(3, N, N, N)` float64.
