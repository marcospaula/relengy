"""Reliability growth: Crow-AMSAA (NHPP) and Duane.

For **repairable systems** under development. Do not confuse this with life data
analysis: here the times are *cumulative* on a single system (or fleet) being
corrected as it runs, not the ages of independent items until they fail.

⚠ **The Crow-AMSAA β is NOT the Weibull β.** Same symbol, opposite meaning:

| β | Weibull (LDA) | Crow-AMSAA (NHPP) |
|---|---|---|
| β < 1 | infant mortality (bad) | **reliability growing** (good) |
| β = 1 | random failures | homogeneous Poisson: no growth |
| β > 1 | wear-out | **reliability worsening** |

The failure intensity is ρ(T) = λ·β·T^(β−1). With β < 1 it decreases — the system
is improving. This is the most common confusion when reading the two chapters
back to back.

Estimators (ReliaWiki, *Crow-AMSAA (NHPP)*, the MLE and "Biasing and Unbiasing of
Beta" sections), checked against the wiki's own worked example:

    β̂ = n / (n·ln T* − Σ ln Tᵢ)          same formula for both test types
    λ̂ = n / T*^β

where T* = Tₙ if the test ends on a failure, T* = the final time if it ends on time.

Unbiasing (this is where they differ):

    TIME-terminated test:     β̄ = (N−1)/N · β̂
    FAILURE-terminated test:  β̄ = (N−2)/(N−1) · β̂
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Termination = Literal["failure", "time"]


@dataclass
class CrowAMSAA:
    beta: float
    lam: float
    n_failures: int
    t_end: float
    termination: Termination
    unbiased: bool

    # -- failure intensity -------------------------------------------------

    def instantaneous_intensity(self, t: float | None = None) -> float:
        """ρ(T) = λ·β·T^(β−1). The failure rate *right now*."""
        t = self.t_end if t is None else t
        return self.lam * self.beta * t ** (self.beta - 1.0)

    def cumulative_intensity(self, t: float | None = None) -> float:
        """λ_c(T) = λ·T^(β−1). The historical average, not the current rate."""
        t = self.t_end if t is None else t
        return self.lam * t ** (self.beta - 1.0)

    def expected_failures(self, t: float | None = None) -> float:
        """E[N(T)] = λ·T^β"""
        t = self.t_end if t is None else t
        return self.lam * t**self.beta

    # -- MTBF --------------------------------------------------------------

    def instantaneous_mtbf(self, t: float | None = None) -> float:
        """The MTBF the system has RIGHT NOW. This is the one you report to the customer."""
        return 1.0 / self.instantaneous_intensity(t)

    def cumulative_mtbf(self, t: float | None = None) -> float:
        """Average MTBF since the start of the test. Always optimistic? No: under
        growth (β<1) it is PESSIMISTIC — it carries the old failures."""
        return 1.0 / self.cumulative_intensity(t)

    # -- reading -----------------------------------------------------------

    def verdict(self) -> str:
        if self.beta < 1.0:
            return f"reliability IMPROVING (beta={self.beta:.4f} < 1)"
        if self.beta == 1.0:
            return "no growth: homogeneous Poisson process (beta = 1)"
        return f"reliability DETERIORATING (beta={self.beta:.4f} > 1)"

    def time_to_reach_mtbf(self, target_mtbf: float) -> float:
        """How much test time until the instantaneous MTBF reaches the target.

        Only meaningful with β < 1 (growing). Assumes the current growth rate
        holds — an extrapolation, and the handbook warns it tends to saturate.
        """
        if self.beta >= 1.0:
            raise ValueError(
                f"beta = {self.beta:.4f} >= 1: the system is not improving, "
                "the instantaneous MTBF never reaches the target"
            )
        # 1/(λβT^(β−1)) = target  =>  T^(1−β) = target·λ·β
        return (target_mtbf * self.lam * self.beta) ** (1.0 / (1.0 - self.beta))

    def report(self) -> str:
        return "\n".join([
            f"n = {self.n_failures} failures, {self.termination}-terminated test at T = {self.t_end:g}",
            f"beta   = {self.beta:.4f}{' (unbiased)' if self.unbiased else ' (MLE, biased)'}",
            f"lambda = {self.lam:.4f}",
            f"{self.verdict()}",
            f"instantaneous MTBF at T = {self.instantaneous_mtbf():.1f}",
            f"cumulative    MTBF at T = {self.cumulative_mtbf():.1f}",
        ])


def crow_amsaa(
    failure_times: Sequence[float],
    t_end: float | None = None,
    termination: Termination = "failure",
    unbiased: bool = False,
) -> CrowAMSAA:
    """Fit the Crow-AMSAA model by maximum likelihood.

    `failure_times` are **cumulative** test times, increasing.
    `t_end` is only needed if `termination='time'` (and must be > the last time).
    """
    t = np.asarray(failure_times, dtype=float)
    n = t.size
    if n < 2:
        raise ValueError("Crow-AMSAA needs at least 2 failures")
    if np.any(t <= 0):
        raise ValueError("cumulative times must be positive")
    if np.any(np.diff(t) <= 0):
        raise ValueError("cumulative times must be strictly increasing")

    if termination == "failure":
        if t_end is not None and not math.isclose(t_end, t[-1]):
            raise ValueError(
                "failure-terminated test: t_end is the last failure time "
                f"({t[-1]:g}), not {t_end:g}"
            )
        t_star = float(t[-1])
    elif termination == "time":
        if t_end is None:
            raise ValueError("a time-terminated test requires t_end")
        if t_end < t[-1]:
            raise ValueError(f"t_end ({t_end:g}) cannot be smaller than the last failure ({t[-1]:g})")
        t_star = float(t_end)
    else:
        raise ValueError(f"invalid termination: {termination!r}")

    denom = n * math.log(t_star) - float(np.sum(np.log(t)))
    if denom <= 0:
        raise ValueError("non-positive denominator: check the cumulative times")
    beta = n / denom

    if unbiased:
        # ReliaWiki, "Biasing and Unbiasing of Beta"
        if termination == "failure" and n < 3:
            # the (N-2)/(N-1) factor is zero at N=2, silently returning beta=0
            raise ValueError(
                "unbiased beta for a failure-terminated test needs at least 3 "
                "failures (the (N-2)/(N-1) correction collapses to 0 at N=2)"
            )
        beta *= (n - 1) / n if termination == "time" else (n - 2) / (n - 1)

    lam = n / t_star**beta
    return CrowAMSAA(beta, lam, n, t_star, termination, unbiased)


# ---------------------------------------------------------------------------
# Duane
# ---------------------------------------------------------------------------

@dataclass
class Duane:
    alpha: float   # growth rate (slope on log-log)
    b: float       # intercept
    n_failures: int

    def cumulative_mtbf(self, t: float) -> float:
        """m_c(T) = b·T^α"""
        return self.b * t**self.alpha

    def instantaneous_mtbf(self, t: float) -> float:
        """m_i = m_c / (1 − α). Diverges as α → 1."""
        if self.alpha >= 1.0:
            raise ValueError(f"alpha = {self.alpha:.4f} >= 1: instantaneous MTBF is undefined")
        return self.cumulative_mtbf(t) / (1.0 - self.alpha)


def duane(failure_times: Sequence[float]) -> Duane:
    """Duane by least-squares regression on log-log paper.

    Duane observed that the **cumulative** MTBF against cumulative time falls on a
    straight line in log-log. It is the same phenomenon as Crow-AMSAA, fit by
    regression instead of likelihood — hence `alpha ≈ 1 − beta`.

    **Descriptive, not inferential.** The OLS here fits cumulative-MTBF points
    that are serially correlated: each point carries all the earlier times, so it
    shares data with the next one. That violates the residual independence OLS
    assumes. The fit gives a **point** estimate of the growth trend — not
    confidence intervals or hypothesis tests. For inference, use `crow_amsaa`
    alongside (same phenomenon, by likelihood).
    """
    t = np.asarray(failure_times, dtype=float)
    if t.size < 2:
        raise ValueError("Duane needs at least 2 failures")
    if np.any(np.diff(t) <= 0):
        raise ValueError("cumulative times must be strictly increasing")

    n = np.arange(1, t.size + 1)
    mtbf_c = t / n                       # observed cumulative MTBF
    slope, intercept = np.polyfit(np.log(t), np.log(mtbf_c), 1)
    return Duane(alpha=float(slope), b=float(math.exp(intercept)), n_failures=t.size)
