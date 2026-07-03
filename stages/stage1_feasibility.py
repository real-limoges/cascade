"""Stage 1: resolution / Reynolds-number feasibility check (analytic, no DNS).

For the chosen (N, nu, P) this computes the *a priori* Kolmogorov scale
implied by stationarity (<eps> = P exactly for fixed-power forcing), the
resulting k_max*eta, and estimates of Re_lambda and the inertial-range
width. Writes artifacts/feasibility.json and fails loudly if the
resolution gate cannot be met.

Run:  python -m stages.stage1_feasibility
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dns.params import PRODUCTION, GATES

C_EPS = 0.64   # dissipation coefficient eps ~ C_eps u'^3 / L
               # (updated to the value MEASURED in run 1; 0.45 was the
               # a priori guess and overpredicted Re_lambda by ~30%)
L_EST = 2.3    # integral-scale estimate for forcing at |k|<=1.5 in a 2pi box
               # (run 1, forced at |k|<=2.5, measured L=1.29)


def feasibility(N, nu, P, k_force=1.5):
    kc = N // 3                      # 2/3-rule max retained wavenumber
    eta = (nu**3 / P) ** 0.25        # exact at stationarity since <eps>=P
    kmax_eta = kc * eta
    urms = (P * L_EST / C_EPS) ** (1.0 / 3.0)
    lam = (15.0 * nu / P) ** 0.5 * urms
    re_lambda = urms * lam / nu
    Re_L = urms * L_EST / nu
    # crude inertial-range bounds: above the forcing influence, below the
    # onset of the dissipative roll-off (k*eta ~ 0.15)
    k_lo = 2.0 * k_force
    k_hi = 0.15 / eta
    return {
        "N": N, "kc": kc, "nu": nu, "forcing_power": P,
        "eta_predicted": eta, "kmax_eta_predicted": kmax_eta,
        "urms_predicted": urms, "Re_lambda_predicted": re_lambda,
        "Re_integral_predicted": Re_L,
        "inertial_range_k": [k_lo, max(k_hi, k_lo)],
        "inertial_range_decades": max(0.0, __import__("math").log10(max(k_hi / k_lo, 1.0))),
        "assumptions": {"C_eps": C_EPS, "L_integral_estimate": L_EST},
    }


def main():
    p = PRODUCTION
    report = {"chosen": feasibility(p["N"], p["nu"], p["forcing_power"])}

    # what would larger grids buy, holding kmax*eta ~ its 128^3 value?
    alternatives = []
    for N_alt in (192, 256):
        kc_alt = N_alt // 3
        # pick nu so that kmax*eta matches the chosen run's kmax*eta
        target = report["chosen"]["kmax_eta_predicted"]
        nu_alt = ((target / kc_alt) ** 4 * p["forcing_power"]) ** (1.0 / 3.0)
        alternatives.append(feasibility(N_alt, nu_alt, p["forcing_power"]))
    report["alternatives_same_resolution_margin"] = alternatives

    ok = report["chosen"]["kmax_eta_predicted"] >= GATES["kmax_eta_min"]
    report["gate_kmax_eta"] = {
        "threshold": GATES["kmax_eta_min"],
        "value": report["chosen"]["kmax_eta_predicted"],
        "passed": ok,
    }
    klo, khi = report["chosen"]["inertial_range_k"]
    report["honest_summary"] = (
        f"At {p['N']}^3 with nu={p['nu']}, P={p['forcing_power']}: "
        f"k_max*eta = {report['chosen']['kmax_eta_predicted']:.3f}, "
        f"predicted Re_lambda ~ {report['chosen']['Re_lambda_predicted']:.0f}. "
        "At this Reynolds number there is no asymptotic inertial range: the "
        f"strict -5/3 window is only k in [{klo:.1f}, {khi:.1f}]; any fitted "
        "scaling range is short (< half a decade) and partially supported by "
        "the spectral bottleneck. A decade of true inertial range would need "
        "Re_lambda >~ 300 (>= 512^3), which is not feasible on this hardware."
    )

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/feasibility.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    if not ok:
        sys.exit("FEASIBILITY GATE FAILED: k_max*eta < threshold")


if __name__ == "__main__":
    main()
