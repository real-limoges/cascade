"""Single source of truth for run parameters. All stages import from here."""

# Production (forced, statistically stationary) run
PRODUCTION = {
    "N": 128,                 # grid points per direction
    "nu": 0.0085,             # kinematic viscosity
    "forcing_power": 0.3,     # fixed energy-injection rate P (=> <eps> = P)
    "k_force": 1.5,           # forcing band: 0 < |k| <= k_force
                              # (run 1 used 2.5; confined to the lowest
                              # shells to maximise L and Re_lambda at fixed
                              # kmax*eta — see NOTES.md)
    "cfl": 0.8,               # RK4 advective CFL number
    "ic_energy": 1.5,         # initial total kinetic energy
    "ic_kp": 2.5,             # initial spectrum peak wavenumber
    "seed": 20260702,         # RNG seed for the initial condition
    "spinup_eddy_times": 10.0,   # large-eddy turnovers discarded before sampling
    "n_snapshots": 48,           # snapshots collected for downstream statistics
                                 # (box/L ~ 2.7 at k_force=1.5 gives ~20
                                 # independent integral volumes per snapshot,
                                 # vs ~100 in run 1, so sample more of them)
    "snapshot_spacing_eddy": 1.0,  # spacing between snapshots, in T_eddy
    "fft_threads": 4,
    "dtype": "float32",   # working precision of the production run; the
                          # float64/float32 agreement preflight is in stage 3
}

# Sanity-check (unforced decay) run
SANITY = {
    "N": 64,
    "nu": 0.02,
    "ic_energy": 1.0,
    "ic_kp": 6.0,
    "seed": 777,
    "t_end": 2.0,
    "cfl": 0.6,
    "fft_threads": 4,
}

# Feasibility / validation thresholds
GATES = {
    "kmax_eta_min": 1.5,
    "spectrum_slope_target": -5.0 / 3.0,
    "spectrum_slope_tol": 0.25,
    "eps_balance_tol": 0.10,      # |<eps>/P - 1| during sampling window
    "energy_drift_tol": 0.10,     # linear E-drift over sampling window / <E>
    "decay_budget_tol": 1e-3,     # |dE + int(eps)dt| / E0 in the decay test
}
