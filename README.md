# cascade — DNS of homogeneous isotropic turbulence (phase 1)

A validated pseudo-spectral direct numerical simulation of forced
homogeneous isotropic turbulence at 128³ (Re_λ = 69, k_max·η = 1.59),
plus 48 independent velocity snapshots for downstream structure-function
analysis.

Start with the **executive summary in [PROVENANCE.md](PROVENANCE.md)** —
it covers the validated result, the honest caveats, and the one solver
bug found and fixed along the way. Then:

- `artifacts/resolution_validation.json` — the four validation gates
  (stationarity, resolution, spectrum slope, solenoidality), all passing
- `artifacts/SCHEMA.md` — what every artifact and snapshot file contains
- `NOTES.md` — chronological working notes, including the full
  forced-divergence incident diagnosis
- `stages/` — the four re-runnable pipeline stages (feasibility, sanity
  validation, production, validation gate)
- `dns/` — the solver; `dns/snapshot_io.load_velocity(path, N)` loads a
  snapshot as a physical velocity field

Requires `pip install numpy scipy pyfftw matplotlib` (Python 3.11).
