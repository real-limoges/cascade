"""Stage 4: validation gate for the production run.

Explicit, quantitative diagnostics — this stage decides whether the run is
usable, and says so in artifacts/resolution_validation.json:

A. Statistical stationarity over the sampling window (post spin-up):
   A1. energy balance:      |<eps>/P - 1| <= 10%   (P is injected exactly)
   A2. energy drift:        |linear-fit slope of E(t) * window| / <E> <= 10%
   A3. block statistics:    4-block means of E and eps reported, plus a
       two-sample z-like statistic comparing first/second half means
       against the block-to-block scatter (reported, not gated — with ~4
       independent blocks it has little power, but it makes the
       fluctuation scale explicit).
B. Resolution: k_max*eta >= 1.5 using <eps> measured over the window.
C. Spectrum: snapshot-averaged E(k); log-log slope over the a priori
   inertial band k in [4, 10] must be within +-0.25 of -5/3. The
   compensated spectrum is written out so the (short) extent of the
   scaling range is visible rather than implied.
D. Isotropy (reported): component energy ratios; snapshot-to-snapshot
   velocity correlation (supports the snapshot-independence claim).

Also writes artifacts/energy_spectrum.json and diagnostic plots
artifacts/spectrum.png, artifacts/timeseries.png.

Run:  python -m stages.stage4_validate
"""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dns.params import PRODUCTION, GATES
from dns.spectral import SpectralGrid
from dns.snapshot_io import load_snapshot, unpack

SNAPDIR = "data/snapshots"
TOJSON = lambda o: o.item() if hasattr(o, "item") else str(o)


def stationarity(ts, spinup_t, P):
    m = ts["t"] > spinup_t
    t, E, eps = ts["t"][m], ts["E"][m], ts["eps"][m]
    window = t[-1] - t[0]
    Em, epsm = E.mean(), eps.mean()
    bal = epsm / P - 1.0

    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    slope = np.linalg.lstsq(A, E, rcond=None)[0][0]
    drift = abs(slope) * window / Em

    nb = 4
    blocks = np.array_split(np.arange(t.size), nb)
    bE = np.array([E[b].mean() for b in blocks])
    beps = np.array([eps[b].mean() for b in blocks])
    h1, h2 = E[: t.size // 2].mean(), E[t.size // 2 :].mean()
    sigma_block = bE.std(ddof=1)
    z_halves = abs(h2 - h1) / (sigma_block if sigma_block > 0 else np.inf)

    return {
        "window_t": [float(t[0]), float(t[-1])],
        "window_eddy_turnovers": float(ts["tau"][m][-1] - ts["tau"][m][0]),
        "mean_E": Em, "mean_eps": epsm,
        "eps_over_P_minus_1": bal,
        "eps_balance_tol": GATES["eps_balance_tol"],
        "energy_drift_fraction": drift,
        "energy_drift_tol": GATES["energy_drift_tol"],
        "block_means_E": bE.tolist(), "block_means_eps": beps.tolist(),
        "block_scatter_E_rel": float(sigma_block / Em),
        "half_window_z": float(z_halves),
        "passed": bool(
            abs(bal) <= GATES["eps_balance_tol"]
            and drift <= GATES["energy_drift_tol"]
        ),
    }


def snapshot_stats(paths, p):
    g = SpectralGrid(p["N"], threads=p["fft_threads"],
                     planner="FFTW_ESTIMATE", dtype="float64")
    Ek_sum = None
    metas, comp_E, corr = [], [], []
    prev = None
    for path in paths:
        coeffs, meta = load_snapshot(path)
        c = unpack(coeffs, p["N"])
        metas.append(meta)
        Ek = g.spectrum(c)
        Ek_sum = Ek if Ek_sum is None else Ek_sum + Ek
        # component energies (isotropy)
        e = [
            0.5 * float(np.sum(g.w * (c[i].real**2 + c[i].imag**2)))
            for i in range(3)
        ]
        comp_E.append(e)
        # correlation with previous snapshot: <u_n . u_{n+1}> / sqrt(<u_n^2><u_{n+1}^2>)
        if prev is not None:
            num = float(np.sum(g.w * (c * prev.conj()).real))
            den = 2.0 * np.sqrt(
                g.energy(c) * g.energy(prev)
            )
            corr.append(num / den)
        prev = c
    Ek_mean = Ek_sum / len(paths)
    return g, Ek_mean, metas, np.array(comp_E), np.array(corr)


def main():
    p = PRODUCTION
    ts = dict(np.load("artifacts/production_timeseries.npz"))
    with open("artifacts/production_summary.json") as f:
        summary = json.load(f)
    spinup_t = summary["spinup_end"]["t"]
    P = p["forcing_power"]

    # --- A: stationarity
    stat = stationarity(ts, spinup_t, P)

    # --- snapshots
    paths = sorted(glob.glob(os.path.join(SNAPDIR, "snap_*.npz")))
    g, Ek, metas, comp_E, corr = snapshot_stats(paths, p)
    k = np.arange(len(Ek), dtype=np.float64)

    # --- B: resolution from measured dissipation
    nu = p["nu"]
    eps_m = stat["mean_eps"]
    eta = (nu**3 / eps_m) ** 0.25
    kmax_eta = g.kc * eta
    E_tot = float(Ek.sum())
    urms = np.sqrt(2 * E_tot / 3)
    lam = np.sqrt(15 * nu / eps_m) * urms
    re_lambda = urms * lam / nu
    L_int = 3 * np.pi / (4 * E_tot) * float(np.sum(Ek[1:] / k[1:]))

    # --- C: spectrum slope over the a priori band
    lo, hi = 4, 10
    band = (k >= lo) & (k <= hi) & (Ek > 0)
    slope, icept = np.polyfit(np.log(k[band]), np.log(Ek[band]), 1)
    target, tol = GATES["spectrum_slope_target"], GATES["spectrum_slope_tol"]
    slope_ok = abs(slope - target) <= tol
    # Kolmogorov-compensated spectrum for the record
    comp = Ek[1:] * k[1:] ** (5 / 3) / eps_m ** (2 / 3)

    # --- D: isotropy + snapshot independence
    comp_ratio = (comp_E / comp_E.sum(axis=1, keepdims=True)).mean(axis=0)
    iso = {
        "mean_component_energy_fractions": comp_ratio.tolist(),
        "max_anisotropy_pct": float(
            100 * np.max(np.abs(comp_ratio - 1 / 3) / (1 / 3))
        ),
        "consecutive_snapshot_velocity_correlation": {
            "mean": float(corr.mean()), "max": float(corr.max()),
            "note": "whole-field correlation is dominated by the forced "
                    "large scales; inertial-range increments decorrelate "
                    "faster than this",
        },
    }

    gates = {
        "stationarity": stat["passed"],
        "kmax_eta": bool(kmax_eta >= GATES["kmax_eta_min"]),
        "spectrum_slope": bool(slope_ok),
    }
    result = {
        "params": p,
        "git_sha": summary["git_sha"],
        "n_snapshots": len(paths),
        "snapshot_spacing_eddy_turnovers": p["snapshot_spacing_eddy"],
        "stationarity": stat,
        "measured": {
            "E": E_tot, "eps": eps_m, "urms": urms, "eta": eta,
            "kmax_eta": kmax_eta, "L_integral": L_int,
            "lambda_taylor": lam, "Re_lambda": re_lambda,
            "T_eddy": L_int / urms,
            "eddy_turnovers_sampled": stat["window_eddy_turnovers"],
        },
        "spectrum_gate": {
            "fit_band_k": [lo, hi],
            "fitted_slope": float(slope),
            "target": target, "tolerance": tol, "passed": slope_ok,
            "caveat": "at Re_lambda ~ 70-90 this is a short approximate "
                      "scaling range (< half a decade), partially supported "
                      "by the spectral bottleneck; it is NOT an asymptotic "
                      "inertial range",
        },
        "isotropy_and_independence": iso,
        "gates": gates,
        "validated": bool(all(gates.values())),
    }

    with open("artifacts/resolution_validation.json", "w") as f:
        json.dump(result, f, indent=2, default=TOJSON)
    with open("artifacts/energy_spectrum.json", "w") as f:
        json.dump(
            {"k": k.tolist(), "E_k_snapshot_mean": Ek.tolist(),
             "compensated_k53_over_eps23": [0.0] + comp.tolist(),
             "n_snapshots": len(paths)},
            f, indent=2, default=TOJSON,
        )

    plots(ts, spinup_t, k, Ek, comp, eps_m, eta, lo, hi, slope, icept, P)
    print(json.dumps(result, indent=2, default=TOJSON))
    if not result["validated"]:
        sys.exit("VALIDATION GATE FAILED — see artifacts/resolution_validation.json")


def plots(ts, spinup_t, k, Ek, comp, eps_m, eta, lo, hi, slope, icept, P):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(ts["t"], ts["E"], lw=0.8)
    axes[0].axvline(spinup_t, color="k", ls="--", lw=0.8, label="spin-up end")
    axes[0].set_ylabel("E(t)")
    axes[0].legend()
    axes[1].plot(ts["t"], ts["eps"], lw=0.8, label=r"$\varepsilon(t)$")
    axes[1].axhline(P, color="r", ls="--", lw=0.8, label="P (injection)")
    axes[1].axvline(spinup_t, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("t")
    axes[1].set_ylabel(r"$\varepsilon(t)$")
    axes[1].legend()
    fig.suptitle("Production run: energy and dissipation")
    fig.tight_layout()
    fig.savefig("artifacts/timeseries.png", dpi=120)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    m = (k > 0) & (Ek > 0)
    axes[0].loglog(k[m], Ek[m], "-o", ms=2.5)
    kk = np.linspace(lo, hi, 50)
    axes[0].loglog(kk, np.exp(icept) * kk**slope, "r--",
                   label=f"fit slope {slope:.2f}")
    axes[0].loglog(kk, np.exp(icept) * kk ** (-5 / 3) * (lo**slope / lo ** (-5 / 3)),
                   "g:", label="-5/3")
    axes[0].axvspan(lo, hi, alpha=0.1, color="r")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("E(k)")
    axes[0].legend()
    axes[1].semilogx(k[1:][Ek[1:] > 0] * eta, comp[Ek[1:] > 0], "-o", ms=2.5)
    axes[1].axhline(1.6, color="g", ls=":", label="C_K = 1.6")
    axes[1].set_xlabel(r"$k\eta$")
    axes[1].set_ylabel(r"$E(k)k^{5/3}\varepsilon^{-2/3}$")
    axes[1].legend()
    fig.suptitle("Snapshot-averaged energy spectrum")
    fig.tight_layout()
    fig.savefig("artifacts/spectrum.png", dpi=120)


if __name__ == "__main__":
    main()
