"""RS401 — Minimum durability for the whole population (sample != population).

Adapted from a ReliaSoft Brasil RS401 (Life Data Analysis) course example; the
source problem cites the Brazilian standard NBR 6742.

An item is fatigue-tested on a bench. Five specimens fail at:

    10,263 · 12,187 · 16,908 · 18,042 · 23,271 cycles

The spec: a **minimum durability of 8,000 cycles for the ENTIRE population**.
Every one of the five specimens outlasted 8,000 (the smallest is 10,263), so the
naive reading is "approve". That reading is wrong: it is about the *sample*. The
spec is about the *population*, and the question is what fraction of the
population is predicted to fail before 8,000 cycles — i.e. F(8,000).

This example also cross-checks relengy against the course's hand-drawn Weibull
paper solution, and shows why the graphical read-offs are internally inconsistent
while the numerical fit is not.

Run:  python examples/rs401_min_durability.py
"""

from __future__ import annotations

import numpy as np

from relengy.qualitative.fta import weibull_cdf
from relengy.quantitative.fitting import fit_diagnostic
from relengy.quantitative.ranks import bernard_median_rank

SPEC_CYCLES = 8000
FAILURES = [10263, 12187, 16908, 18042, 23271]

# The course's graphical read-offs (Weibull paper, by eye).
COURSE_BETA = 3.0
COURSE_ETA = 16000
COURSE_F_READ = 0.06  # F(8000) read straight off the plotted line


def reliability_and_unreliability(t: float, beta: float, eta: float) -> tuple[float, float]:
    f = weibull_cdf(t, beta, eta)
    return 1.0 - f, f


def main() -> None:
    diag = fit_diagnostic(FAILURES)  # 5 failures, no suspensions

    # ---- Part 1: the fit and the population question ------------------------
    print("=" * 74)
    print("FIT (relengy)")
    print("-" * 74)
    print(diag.report())
    r_mrr, f_mrr = reliability_and_unreliability(SPEC_CYCLES, diag.beta_mrr, diag.eta_mrr)
    print(f"\n  Smallest observed failure: {min(FAILURES):,} cycles "
          f"(all five cleared {SPEC_CYCLES:,}).")
    print(f"  But the fitted population gives F({SPEC_CYCLES:,}) = {f_mrr:.1%} "
          f"(R = {r_mrr:.1%}):")
    print(f"  ~{f_mrr:.0%} of the population is predicted to fail before the spec.")

    # ---- Part 2: cross-check against the course's graphical solution --------
    print("\n" + "=" * 74)
    print("CROSS-CHECK vs the course's Weibull-paper solution")
    print("-" * 74)

    print("  Median ranks — relengy bernard_median_rank vs the course table:")
    ranks = bernard_median_rank(np.arange(1, len(FAILURES) + 1), len(FAILURES)) * 100
    course_ranks = [12.96, 31.48, 50.00, 68.51, 87.03]
    for i, (t, a, b) in enumerate(zip(FAILURES, ranks, course_ranks), 1):
        print(f"    {i}deg {t:>6,}:  relengy {a:6.2f}%   course {b:6.2f}%")

    print("\n  Parameters and F(8000):")
    f_course_params = weibull_cdf(SPEC_CYCLES, COURSE_BETA, COURSE_ETA)
    print(f"    course graphical   beta={COURSE_BETA:.2f}  eta={COURSE_ETA:,}"
          f"  -> recomputes F = {f_course_params:.1%}")
    print(f"    course read F off the plot                       -> F = {COURSE_F_READ:.0%}")
    print(f"    relengy RRX        beta={diag.beta_mrr:.2f}  eta={diag.eta_mrr:,.0f}"
          f"  -> F = {f_mrr:.1%}")
    print("  The graphical read-offs don't reproduce each other: beta=3.0 & eta=16000")
    print(f"  recompute to {f_course_params:.0%}, not the {COURSE_F_READ:.0%} read off the line.")
    print("  relengy's single (beta, eta) is self-consistent and lands on the ~6-7%")
    print("  the plot pointed to. Same decision, exact numbers.")

    # ---- Verdict ------------------------------------------------------------
    print("\n" + "=" * 74)
    print("VERDICT")
    print("-" * 74)
    print(f"  Spec: minimum durability of {SPEC_CYCLES:,} cycles for the WHOLE population,")
    print("        i.e. essentially F(8000) = 0.")
    print(f"  Result: F({SPEC_CYCLES:,}) = {f_mrr:.1%} != 0  ->  REJECT.")
    print("  The sample passing is not the population meeting the spec.")
    print()
    print("  Engineering nuance (why the answer is really risk-based):")
    print("   - F=0 exactly is impossible for a 2P Weibull, so a purely literal read")
    print("     always rejects. The real call weighs risk = frequency (~7%) x severity;")
    print("     a non-critical, low-cost consequence could be accepted.")
    print("   - Fatigue often has a minimum life: if a 3P Weibull with threshold")
    print("     gamma > 8000 fits, then F(8000) = 0 and you would approve. Five points")
    print("     can't settle it, but it is the key physical question.")
    print("   - n=5 is tiny: the upper bound on F(8000) is far higher, so a")
    print("     conservative norm pushes toward reject / gather more data (see issue #11).")


if __name__ == "__main__":
    main()
