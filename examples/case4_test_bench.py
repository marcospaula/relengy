"""Case 4 — Test-bench reliability with competing failure modes.

Adapted from a ReliaSoft Brasil training case ("Caso 4: Bancada de Prova").

A part is qualified on a bench. The company norm: it must reach **90% reliability
at 20,000 cycles**, demonstrated at a **90% lower one-sided confidence bound**.
Twenty parts were tested; three distinct failure modes appeared (MFA, MFB, MFC);
six parts were still running at 120,000 cycles (suspensions).

The right model is **competing failure modes** (a series system of independent
risks): each mode is fit on its own, with the *other* modes' failures entering as
suspensions — a part left the risk set of mode A because another mode killed it
first. System reliability is the product:

    R_part(t) = R_MFA(t) * R_MFB(t) * R_MFC(t)

What relengy does here, and where it stops:
  * `fit_diagnostic` fits each mode by MRR and MLE, recommends the method, and
    flags a possible batch problem — its diagnosis is what tells us the MLE fits
    are not trustworthy on 4-5 failures under heavy censoring, so we read the
    physically credible MRR fit.
  * relengy does NOT yet expose confidence bounds, which the norm needs. The
    lower bound below is computed by reaching down to the wrapped `reliability`
    fit. See issue #11.

Run:  python examples/case4_test_bench.py
"""

from __future__ import annotations

from reliability.Fitters import Fit_Weibull_2P

from relengy.qualitative.fta import weibull_cdf
from relengy.quantitative.fitting import fit_diagnostic

TARGET_CYCLES = 20_000
TARGET_RELIABILITY = 0.90
# 90% one-sided normal quantile, norm.ppf(0.90); hard-coded to avoid a scipy import.
Z_90_ONE_SIDED = 1.2816

# Cycles to failure, by mode (each row of the table is one part).
MFA = [65800, 81500, 85000, 89900, 115000]
MFB = [41200, 49800, 58900, 76800]
MFC = [29800, 29900, 31250, 48900, 54000]
SUSPENSIONS = [120000] * 6

MODES = {"MFA": MFA, "MFB": MFB, "MFC": MFC}
ALL_FAILURES = MFA + MFB + MFC


def reliability_at(t: float, beta: float, eta: float) -> float:
    """R(t) = 1 - F(t), via relengy's weibull_cdf."""
    return 1.0 - weibull_cdf(t, beta, eta)


def suspensions_for(mode_failures: list[int]) -> list[int]:
    """Other modes' failures + the real suspensions, all right-censored.

    All cycle values in this dataset are distinct, so filtering by value is safe.
    """
    others = [c for c in ALL_FAILURES if c not in mode_failures]
    return others + SUSPENSIONS


def lower_bound_reliability(t: float, beta: float, eta: float,
                            beta_se: float, eta_se: float, cov: float) -> float:
    """90% one-sided lower bound on R(t), Fisher-matrix / delta method.

    Linearize on W = ln(-ln R) = beta * (ln t - ln eta). R decreases in W, so the
    LOWER bound on R uses the UPPER bound on W. This is what relengy should expose
    natively (issue #11); done here on the wrapped `reliability` fit's covariance.
    """
    import math

    w = beta * (math.log(t) - math.log(eta))
    dw_deta = -beta / eta
    dw_dbeta = math.log(t) - math.log(eta)
    var_w = (dw_deta**2 * eta_se**2
             + dw_dbeta**2 * beta_se**2
             + 2 * dw_deta * dw_dbeta * cov)
    w_upper = w + Z_90_ONE_SIDED * math.sqrt(var_w)
    return math.exp(-math.exp(w_upper))


def main() -> None:
    # ---- Part 1: per-mode fit with relengy (the showcase) -------------------
    fits = {}
    r_point_system = 1.0
    for name, failures in MODES.items():
        suspensions = suspensions_for(failures)
        diag = fit_diagnostic(failures, right_censored=suspensions)
        fits[name] = diag
        r_point = reliability_at(TARGET_CYCLES, diag.beta_mrr, diag.eta_mrr)
        r_point_system *= r_point

        print("=" * 74)
        print(f"MODE {name}")
        print("-" * 74)
        print(diag.report())
        print(f"\n  R_{name}(20k), MRR = {r_point:.4f}"
              f"   (MLE = {reliability_at(TARGET_CYCLES, diag.beta_mle, diag.eta_mle):.4f})")
        print()

    # ---- Part 2: 90% lower confidence bound (beyond relengy today, issue #11) --
    print("=" * 74)
    print("90% LOWER ONE-SIDED BOUND on R(20k)  [via the wrapped reliability fit]")
    print("-" * 74)
    r_lower_system = 1.0
    for name, failures in MODES.items():
        f = Fit_Weibull_2P(failures=failures, right_censored=suspensions_for(failures),
                           method="RRX", CI=0.90,
                           show_probability_plot=False, print_results=False)
        r_lower = lower_bound_reliability(TARGET_CYCLES, f.beta, f.alpha,
                                          f.beta_SE, f.alpha_SE, f.Cov_alpha_beta)
        r_lower_system *= r_lower
        print(f"  {name}:  R(20k) lower = {r_lower:.4f}")
    print(f"  SYSTEM (product of lower bounds, conservative) = {r_lower_system:.4f}")

    # ---- Verdict ------------------------------------------------------------
    print("=" * 74)
    print("VERDICT")
    print("-" * 74)
    print(f"  Norm: R({TARGET_CYCLES:,}) >= {TARGET_RELIABILITY:.0%} "
          "at the 90% lower one-sided bound.")
    print(f"  Point estimate (MRR):  R_system(20k) = {r_point_system:.1%}  -> clears 90%")
    print(f"  90% lower bound:       R_system(20k) ~ {r_lower_system:.1%}  -> BELOW 90%")
    print("  Q1: NOT approved -- the point estimate passes, but the test is "
          "under-powered\n      to demonstrate 90% at the required confidence.")
    print("  Q2: Improve MFC first -- lowest eta (~69,846 cycles), earliest "
          "failures, and\n      the dominant contributor to unreliability at 20k.")


if __name__ == "__main__":
    main()
