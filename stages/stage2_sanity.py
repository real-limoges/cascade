"""Stage 2: solver sanity validation against known behavior (no forcing).

Four checks, each with an explicit pass threshold:

1. exact_viscous_decay   A single Fourier mode u = sin(2z) x_hat is an
                         exact solution decaying as exp(-nu k^2 t) (its
                         nonlinear term is a pure gradient, removed by the
                         projection). Compares against the closed form.
2. nonlinear_conservation  With 2/3 dealiasing the rotational-form
                         nonlinear term is a Galerkin truncation and must
                         conserve energy to roundoff: sum Re(c* . N(c)) ~ 0.
3. decay_energy_budget   Unforced 64^3 turbulence: the truncated system
                         satisfies dE/dt = -eps(t) exactly, so
                         E(t_end) - E(0) must equal -int eps dt to time-
                         integration accuracy. Also requires monotonic decay.
4. incompressibility     max |div u| stays at roundoff through the run.

Writes artifacts/sanity_checks.json; exits nonzero if any check fails.

Run:  python -m stages.stage2_sanity
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dns.spectral import SpectralGrid
from dns.solver import Solver
from dns.ic import isotropic_field
from dns.params import SANITY, GATES


def check_exact_viscous_decay():
    N, nu = 32, 0.1
    g = SpectralGrid(N, planner="FFTW_ESTIMATE")
    s = Solver(g, nu=nu)
    z = np.arange(N) * 2 * np.pi / N
    u0 = np.zeros((3, N, N, N))
    u0[0] = np.sin(2 * z)[None, None, :]
    s.c[:] = g.fwd3(u0)
    E0 = g.energy(s.c)
    dt, nsteps = 0.01, 100
    for _ in range(nsteps):
        s.step(dt)
    E1 = g.energy(s.c)
    exact = E0 * np.exp(-2.0 * nu * 4.0 * nsteps * dt)
    err = abs(E1 - exact) / exact
    return {"rel_error": err, "threshold": 1e-8, "passed": err < 1e-8}


def check_nonlinear_conservation(g, c):
    s = Solver(g, nu=0.0)
    out = np.empty_like(c)
    s.rhs(c, out=out)
    transfer = float(np.sum(g.w * (c.conj() * out).real))
    rel = abs(transfer) / g.energy(c)
    return {"rel_transfer": rel, "threshold": 1e-12, "passed": rel < 1e-12}


def check_decay(g, c0):
    p = SANITY
    s = Solver(g, nu=p["nu"], cfl=p["cfl"])
    s.c[:] = c0
    E0 = g.energy(s.c)
    ts, Es, eps_s = [0.0], [E0], [g.dissipation(s.c, p["nu"])]
    # prime umax for the CFL controller
    s.rhs(s.c, out=s._k1, measure_umax=True)
    max_div = 0.0
    while s.t < p["t_end"]:
        dt = min(s.compute_dt(), p["t_end"] - s.t)
        s.step(dt)
        ts.append(s.t)
        Es.append(g.energy(s.c))
        eps_s.append(g.dissipation(s.c, p["nu"]))
        if s.step_count % 20 == 0:
            max_div = max(max_div, g.max_divergence(s.c))
    ts, Es, eps_s = map(np.asarray, (ts, Es, eps_s))
    budget_err = abs((Es[-1] - Es[0]) + np.trapezoid(eps_s, ts)) / E0
    monotonic = bool(np.all(np.diff(Es) < 0))
    urms = np.sqrt(2 * Es[-1] / 3)
    div_rel = max_div / (urms * g.kc)
    return {
        "N": p["N"], "nu": p["nu"], "t_end": p["t_end"],
        "n_steps": int(s.step_count),
        "E_initial": float(E0), "E_final": float(Es[-1]),
        "energy_budget_rel_error": float(budget_err),
        "budget_threshold": GATES["decay_budget_tol"],
        "monotonic_decay": monotonic,
        "max_div_normalized": float(div_rel),
        "div_threshold": 1e-10,
        "passed": bool(
            budget_err < GATES["decay_budget_tol"]
            and monotonic
            and div_rel < 1e-10
        ),
    }


def check_forced_divergence_control():
    """Regression test for the forced-divergence instability: fixed-power
    forcing amplifies any divergent component of the band modes at rate
    ~P/(2 E_f), so roundoff-seeded divergence grows exponentially unless
    the state is re-projected every step. Run a float32 forced run at 32^3
    for 3000 steps and require the divergent-component energy to stay at
    the roundoff floor."""
    N = 32
    g = SpectralGrid(N, planner="FFTW_ESTIMATE", dtype="float32")
    s = Solver(g, nu=0.02, forcing_power=0.3, k_force=1.5, cfl=0.8)
    s.c[:] = isotropic_field(g, E0=1.3, kp=2.5, seed=99)
    s.rhs(s.c, out=s._k1, measure_umax=True)
    max_rel = 0.0
    for _ in range(3000):
        s.step(s.compute_dt())
        if s.step_count % 200 == 0:
            max_rel = max(max_rel,
                          s.divergence_energy() / g.energy(s.c))
    return {
        "n_steps": 3000, "max_Ediv_over_E": max_rel,
        "threshold": 1e-9, "passed": bool(max_rel < 1e-9),
    }


def main():
    p = SANITY
    g = SpectralGrid(p["N"], threads=p["fft_threads"], planner="FFTW_MEASURE")
    c0 = isotropic_field(g, E0=p["ic_energy"], kp=p["ic_kp"], seed=p["seed"])

    results = {
        "exact_viscous_decay": check_exact_viscous_decay(),
        "nonlinear_conservation": check_nonlinear_conservation(g, c0),
        "decay_energy_budget": check_decay(g, c0),
        "forced_divergence_control": check_forced_divergence_control(),
    }
    results["all_passed"] = all(r["passed"] for r in results.values())

    os.makedirs("artifacts", exist_ok=True)
    tojson = lambda o: o.item() if hasattr(o, "item") else str(o)
    with open("artifacts/sanity_checks.json", "w") as f:
        json.dump(results, f, indent=2, default=tojson)
    print(json.dumps(results, indent=2, default=tojson))
    if not results["all_passed"]:
        sys.exit("SANITY CHECKS FAILED")


if __name__ == "__main__":
    main()
