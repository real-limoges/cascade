"""Stage 3: forced production run — spin-up to stationarity, then snapshot
sampling for downstream structure-function statistics.

- Fixed-power forcing (P = 0.3) on 0 < |k| <= 2.5; at stationarity <eps> = P.
- Elapsed time is tracked in units of the instantaneous large-eddy turnover
  time T = L/u' (tau = int dt / T): spin-up discards the first
  `spinup_eddy_times` turnovers, then snapshots are written every
  `snapshot_spacing_eddy` turnovers until `n_snapshots` are collected.
- Full time series of (t, dt, E, eps, band energy, umax) is saved to
  artifacts/production_timeseries.npz for the stage-4 stationarity test.
- Checkpoints to data/checkpoint.npz every 1000 steps; re-running this
  stage resumes from the checkpoint.

Run:  python -m stages.stage3_production
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dns.spectral import SpectralGrid, load_wisdom, save_wisdom
from dns.solver import Solver
from dns.ic import isotropic_field
from dns.params import PRODUCTION
from dns.snapshot_io import save_snapshot

CKPT = "data/checkpoint.npz"
WISDOM = "data/fftw_wisdom.pkl"
SNAPDIR = "data/snapshots"
TS_PATH = "artifacts/production_timeseries.npz"


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def preflight_precision(p, nsteps=25):
    """Advance the same IC in float32 and float64 for a few dozen steps and
    require the energy/dissipation trajectories to agree closely; guards the
    choice of single precision for the production run."""
    results = {}
    dt = None
    for dt_name in ("float64", "float32"):
        load_wisdom(WISDOM)
        g = SpectralGrid(p["N"], threads=p["fft_threads"],
                         planner="FFTW_MEASURE", dtype=dt_name)
        save_wisdom(WISDOM)
        s = Solver(g, nu=p["nu"], forcing_power=p["forcing_power"],
                   k_force=p["k_force"], cfl=p["cfl"])
        s.c[:] = isotropic_field(g, E0=p["ic_energy"], kp=p["ic_kp"],
                                 seed=p["seed"]).astype(g.cdt)
        s.rhs(s.c, out=s._k1, measure_umax=True)
        if dt is None:
            dt = s.compute_dt()  # from the float64 pass; shared so both
                                 # trajectories are compared at equal times
        for _ in range(nsteps):
            s.step(dt)
        E, eps = g.energy_and_dissipation(s.c, s.nu)
        results[dt_name] = (E, eps)
        del s, g
    E64, e64 = results["float64"]
    E32, e32 = results["float32"]
    rel_E = abs(E32 - E64) / E64
    rel_e = abs(e32 - e64) / e64
    report = {
        "nsteps": nsteps, "E_float64": E64, "E_float32": E32,
        "eps_float64": e64, "eps_float32": e32,
        "rel_diff_E": rel_E, "rel_diff_eps": rel_e,
        "threshold": 1e-4, "passed": bool(rel_E < 1e-4 and rel_e < 1e-4),
    }
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/precision_preflight.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[preflight] float32 vs float64 after {nsteps} steps: "
          f"dE={rel_E:.2e} deps={rel_e:.2e} passed={report['passed']}",
          flush=True)
    if not report["passed"]:
        sys.exit("PRECISION PREFLIGHT FAILED: float32 diverges from float64")


def main():
    p = PRODUCTION
    os.makedirs("data", exist_ok=True)
    os.makedirs(SNAPDIR, exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    if not os.path.exists(CKPT):
        preflight_precision(p)

    load_wisdom(WISDOM)
    t0 = time.time()
    g = SpectralGrid(p["N"], threads=p["fft_threads"], planner="FFTW_MEASURE",
                     dtype=p["dtype"])
    save_wisdom(WISDOM)
    print(f"[setup] FFT planning done in {time.time()-t0:.1f}s", flush=True)

    s = Solver(g, nu=p["nu"], forcing_power=p["forcing_power"],
               k_force=p["k_force"], cfl=p["cfl"])

    series = {k: [] for k in ("t", "dt", "E", "eps", "Ef", "umax", "tau")}
    tau = 0.0            # elapsed large-eddy turnovers
    n_snaps = 0
    # first snapshot one full spacing *after* the spin-up boundary, so no
    # sample sits exactly at the least-decorrelated instant (audit finding)
    next_snap_tau = p["spinup_eddy_times"] + p["snapshot_spacing_eddy"]
    spinup_end = {"t": None, "step": None, "tau": p["spinup_eddy_times"]}

    if os.path.exists(CKPT):
        d = np.load(CKPT)
        s.c[:] = d["c"]
        s.t = float(d["t"]); s.step_count = int(d["step"])
        tau = float(d["tau"]); n_snaps = int(d["n_snaps"])
        next_snap_tau = float(d["next_snap_tau"])
        if d["spinup_end_t"] >= 0:
            spinup_end["t"] = float(d["spinup_end_t"])
            spinup_end["step"] = int(d["spinup_end_step"])
        ts_old = np.load(TS_PATH)
        for k in series:
            series[k] = list(ts_old[k])
        print(f"[resume] step={s.step_count} t={s.t:.2f} tau={tau:.2f} "
              f"snaps={n_snaps}", flush=True)
    else:
        s.c[:] = isotropic_field(g, E0=p["ic_energy"], kp=p["ic_kp"],
                                 seed=p["seed"])

    def save_ts():
        np.savez(TS_PATH, **{k: np.asarray(v) for k, v in series.items()})

    def save_ckpt():
        np.savez(
            CKPT, c=s.c, t=s.t, step=s.step_count, tau=tau,
            n_snaps=n_snaps, next_snap_tau=next_snap_tau,
            spinup_end_t=-1.0 if spinup_end["t"] is None else spinup_end["t"],
            spinup_end_step=-1 if spinup_end["step"] is None
            else spinup_end["step"],
        )
        save_ts()

    # prime umax for CFL
    s.rhs(s.c, out=s._k1, measure_umax=True)
    iq = s.integral_quantities()
    T_eddy = iq["T_eddy"]
    sha = git_sha()
    wall0 = time.time()

    while n_snaps < p["n_snapshots"]:
        dt = s.compute_dt()
        s.step(dt)
        tau += dt / T_eddy

        E = g.energy(s.c)
        eps = g.dissipation(s.c, s.nu)
        if not np.isfinite(E):
            save_ts()
            sys.exit(f"ABORT: non-finite energy at step {s.step_count}")
        if s.step_count % 200 == 0:
            ediv = s.divergence_energy()
            if ediv > 1e-8 * E:
                save_ts()
                sys.exit(f"ABORT: divergent-component energy {ediv:.3e} "
                         f"(E={E:.3e}) at step {s.step_count}")
        series["t"].append(s.t); series["dt"].append(dt)
        series["E"].append(E); series["eps"].append(eps)
        series["Ef"].append(s.band_energy(s.c))
        series["umax"].append(s._umax); series["tau"].append(tau)

        if s.step_count % 50 == 0:
            iq = s.integral_quantities()
            T_eddy = iq["T_eddy"]

        if spinup_end["t"] is None and tau >= p["spinup_eddy_times"]:
            spinup_end["t"] = s.t
            spinup_end["step"] = s.step_count
            print(f"[spinup done] t={s.t:.2f} step={s.step_count} "
                  f"Re_l={iq['Re_lambda']:.1f} kmax_eta={iq['kmax_eta']:.2f}",
                  flush=True)

        if tau >= next_snap_tau:
            iq = s.integral_quantities()
            T_eddy = iq["T_eddy"]
            meta = {
                "index": n_snaps, "t": s.t, "step": s.step_count,
                "tau_eddy": tau, "N": p["N"], "kc": g.kc, "nu": p["nu"],
                "forcing_power": p["forcing_power"], "k_force": p["k_force"],
                "seed": p["seed"], "git_sha": sha,
                **{k: float(v) for k, v in iq.items()},
            }
            path = os.path.join(SNAPDIR, f"snap_{n_snaps:04d}.npz")
            save_snapshot(path, s.c, g.kc, meta)
            n_snaps += 1
            next_snap_tau += p["snapshot_spacing_eddy"]
            rate = (time.time() - wall0) / max(s.step_count, 1)
            print(f"[snap {n_snaps:2d}/{p['n_snapshots']}] t={s.t:.2f} "
                  f"tau={tau:.1f} E={E:.3f} eps={eps:.3f} "
                  f"Re_l={iq['Re_lambda']:.1f} kmax_eta={iq['kmax_eta']:.2f} "
                  f"({rate:.2f}s/step)", flush=True)

        if s.step_count % 1000 == 0:
            save_ckpt()

    save_ckpt()
    summary = {
        "completed": True, "git_sha": sha,
        "final_step": s.step_count, "final_t": s.t, "final_tau": tau,
        "spinup_end": spinup_end, "n_snapshots": n_snaps,
        "params": p,
        "wall_seconds": time.time() - wall0,
    }
    with open("artifacts/production_summary.json", "w") as f:
        json.dump(summary, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(summary, indent=2,
                     default=lambda o: o.item() if hasattr(o, "item") else str(o)))


if __name__ == "__main__":
    main()
