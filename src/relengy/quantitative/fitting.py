"""Weibull fitting: method choice and diagnosis.

Two sources that **do not say exactly the same thing**. The difference matters.

**Abernethy** (New Weibull Handbook, 5.3.3) is categorical:
    "For engineers, the author recommends MRR as best practice [...]
     MLE is not recommended."

**ReliaSoft / ReliaWiki** (Parameter Estimation) qualifies:
    "our recommendation is to use rank regression techniques when the sample
     sizes are small and without heavy censoring. When heavy or uneven censoring
     is present, when a high proportion of interval data is present and/or when
     the sample size is sufficient, MLE should be preferred."

And it adds the reason both distrust MLE on a small sample:
    "MLE estimates of the shape parameter for the Weibull distribution are badly
     biased for small sample sizes, and the effect can be increased depending on
     the amount of censoring."

**The stance taken here:** MRR (RRX) is the default, as Abernethy wants — but it
fails exactly where ReliaSoft points, because MRR depends on *plotting positions*
and, under heavy or uneven censoring, the adjusted ranks carry little
information. In those cases MLE wins. `recommend_method()` decides and explains.

Running MRR and MLE together stays diagnostic: a beta_MLE much smaller than
beta_MRR suggests a batch problem, not a numerical error.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

from reliability.Fitters import Fit_Weibull_2P

# Ratio beta_mle/beta_mrr below which we suspect a batch problem. The handbook
# says "much less" without fixing a number; 0.75 is OUR choice, conservative, to
# flag rather than to conclude. Calibrate it against your own Weibull Library.
BATCH_SUSPICION_RATIO = 0.75

# "Small sample" = <= 20 failures. This number is from the handbook (and from
# Fulton & Tarum's slides, which cite it explicitly).
SMALL_SAMPLE_FAILURES = 20

# "Heavy censoring" — neither Abernethy nor ReliaWiki fixes a number; both say
# "heavy". 0.5 (half or more of the records are suspensions) is OUR choice.
# Adjust it if your Weibull Library points to a different turning point.
HEAVY_CENSORING_FRACTION = 0.5

# Band around beta = 1 for the "random failures" regime. A beta from a real fit
# never lands exactly on 1.0, so an exact equality would never trip that branch.
# ±0.05 treats beta as statistically indistinguishable from 1 within the typical
# noise of an MRR — a flag, not a formal test.
RANDOM_FAILURE_BAND = 0.05


@dataclass
class MethodRecommendation:
    method: str          # 'RRX' or 'MLE'
    reason: str
    caution: str | None = None

    def __str__(self) -> str:
        s = f"{self.method}: {self.reason}"
        return f"{s}\n  CAUTION: {self.caution}" if self.caution else s


def recommend_method(
    n_failures: int,
    n_censored: int = 0,
    uneven_censoring: bool = False,
    interval_data: bool = False,
) -> MethodRecommendation:
    """Choose between MRR (RRX) and MLE, reconciling Abernethy and ReliaSoft.

    `uneven_censoring`: the suspensions cluster (e.g. all at the end of the test)
    instead of spreading among the failures. ReliaWiki treats this as just as
    decisive as heavy censoring — and it is your judgment call, not something to
    infer from the counts.

    `interval_data`: there was periodic inspection; you only know the failure
    happened *between* two inspections. MRR does not handle this well.
    """
    total = n_failures + n_censored
    frac = n_censored / total if total else 0.0
    small = n_failures <= SMALL_SAMPLE_FAILURES
    heavy = frac >= HEAVY_CENSORING_FRACTION

    if n_failures == 0:
        return MethodRecommendation(
            "MLE",
            "no observed failures; MRR is impossible (there are no ranks to fit)",
            caution="consider Weibayes (beta held fixed) — handbook ch. 6",
        )

    if interval_data:
        return MethodRecommendation(
            "MLE",
            "interval data (periodic inspection): MRR needs exact failure times",
        )

    if heavy or uneven_censoring:
        why = []
        if heavy:
            why.append(f"heavy censoring ({frac:.0%} suspensions)")
        if uneven_censoring:
            why.append("uneven censoring")
        rec = MethodRecommendation(
            "MLE",
            " and ".join(why) + ": the MRR plotting positions lose information",
        )
        if small:
            rec.caution = (
                f"small sample ({n_failures} failures): the MLE beta is notoriously "
                "biased here. Use a reduced bias adjustment (RBA) — the `predictr` "
                "package implements one — or Weibayes."
            )
        return rec

    if small:
        return MethodRecommendation(
            "RRX",
            f"small sample ({n_failures} failures) and light censoring: "
            "exactly the case where Abernethy and ReliaSoft agree",
        )

    return MethodRecommendation(
        "RRX",
        "light censoring; MRR is the handbook's best practice and gives you the plot",
        caution="with large n and no censoring, MLE is valid too — if the two "
                "disagree sharply, investigate a batch problem",
    )


@dataclass
class WeibullDiagnosis:
    beta_mrr: float
    eta_mrr: float
    beta_mle: float
    eta_mle: float
    n_failures: int
    n_censored: int

    @property
    def ratio(self) -> float:
        return self.beta_mle / self.beta_mrr

    @property
    def small_sample(self) -> bool:
        """<= 20 failures is a 'small sample' by the handbook's standard."""
        return self.n_failures <= SMALL_SAMPLE_FAILURES

    @property
    def censoring_fraction(self) -> float:
        total = self.n_failures + self.n_censored
        return self.n_censored / total if total else 0.0

    def recommendation(self, uneven_censoring: bool = False,
                       interval_data: bool = False) -> MethodRecommendation:
        return recommend_method(
            self.n_failures, self.n_censored, uneven_censoring, interval_data
        )

    @property
    def batch_suspected(self) -> bool:
        return self.ratio < BATCH_SUSPICION_RATIO

    @property
    def batch_margin(self) -> float:
        """Signed distance from the ratio to the batch-suspicion cutoff.

        `BATCH_SUSPICION_RATIO - ratio`: positive when `batch_suspected` (the
        ratio is BELOW the cutoff — and the larger it is, the stronger the
        signal), negative when there is headroom. It exists so that 0.74 and
        0.20 do not sound equally conclusive just for tripping the same boolean:
        it surfaces *how far* past the threshold — the judgment the boolean hides.
        """
        return BATCH_SUSPICION_RATIO - self.ratio

    def regime(self) -> str:
        """Physical reading of beta (handbook 2.13-2.16), from the MRR."""
        b = self.beta_mrr
        if b < 1.0 - RANDOM_FAILURE_BAND:
            return "infant mortality (beta < 1)"
        if b <= 1.0 + RANDOM_FAILURE_BAND:
            return "random failures (beta ~ 1)"
        if b < 4.0:
            return "early wear-out (1 < beta < 4)"
        return "rapid old-age wear-out (beta > 4)"

    def report(self, uneven_censoring: bool = False,
               interval_data: bool = False) -> str:
        rec = self.recommendation(uneven_censoring, interval_data)
        lines = [
            f"n = {self.n_failures} failures, {self.n_censored} suspensions "
            f"({self.censoring_fraction:.0%} censored)",
            f"MRR: beta = {self.beta_mrr:.4f}  eta = {self.eta_mrr:.2f}",
            f"MLE: beta = {self.beta_mle:.4f}  eta = {self.eta_mle:.2f}",
            f"ratio beta_mle/beta_mrr = {self.ratio:.3f}  "
            f"({abs(self.batch_margin):.3f} "
            f"{'below' if self.batch_margin > 0 else 'above'} "
            f"the {BATCH_SUSPICION_RATIO} batch line)",
            f"regime: {self.regime()}",
            "",
            f"RECOMMENDED METHOD -> {rec}",
        ]
        if self.batch_suspected:
            lines.append(
                f"ALERT: MLE/MRR beta ratio {self.ratio:.3f} sits {self.batch_margin:.3f} "
                f"past the {BATCH_SUSPICION_RATIO} line -> suspected batch problem "
                "(handbook 5.3.3 / 3.9). The wider this gap, the stronger the signal; "
                "investigate a mixture of lots before trusting the fit."
            )
        if self.small_sample:
            lines.append(
                "NOTE: small sample (<= 20 failures). Consider Weibayes, with beta "
                "taken from your Weibull Library (handbook ch. 6)."
            )
        return "\n".join(lines)


def fit_diagnostic(failures, right_censored=None) -> WeibullDiagnosis:
    """Fit a 2P Weibull by MRR and by MLE and return the pair for comparison.

    Requires at least 2 failures. With 0 or 1 failure a two-parameter Weibull
    cannot be fit by regression (no line through fewer than two points) and MLE
    is unreliable — that is the case for Weibayes (fixed beta), not for this
    diagnosis. See `recommend_method`, which handles n=0 explicitly.
    """
    failures = list(failures)
    rc = list(right_censored) if right_censored is not None else None
    if len(failures) < 2:
        raise ValueError(
            f"fit_diagnostic needs at least 2 failures; got {len(failures)}. "
            "A two-parameter Weibull cannot be fit by rank regression (no line "
            "through fewer than two points) and MLE is unreliable with 0-1 "
            "failures. Hold beta from your Weibull Library and use Weibayes "
            "(handbook ch. 6); see recommend_method() for the choice."
        )
    common = dict(
        failures=list(failures),
        right_censored=rc,
        show_probability_plot=False,
        print_results=False,
    )
    mrr = Fit_Weibull_2P(method="RRX", **common)
    mle = Fit_Weibull_2P(method="MLE", **common)

    return WeibullDiagnosis(
        beta_mrr=mrr.beta,
        eta_mrr=mrr.alpha,  # the `reliability` package calls eta "alpha"
        beta_mle=mle.beta,
        eta_mle=mle.alpha,
        n_failures=len(common["failures"]),
        n_censored=len(rc) if rc else 0,
    )


def r_for_confidence(confidence: float) -> float:
    """Assumed failures for a zero-failure Weibayes bound at `confidence` (ch. 6.3).

    r = -ln(1 - C): the first failure is assumed imminent, so C=0.632 gives r=1
    (the classic lower bound) and C=0.90 gives r=2.303. A larger r is more
    conservative — it pushes eta down.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}")
    return -log(1.0 - confidence)


def weibayes(times, beta: float, r: float = 1.0) -> float:
    """Weibayes: one-parameter Weibull with `beta` assumed (handbook Eq. 6-1).

        eta = [ sum(t_i ** beta) / r ] ** (1 / beta)

    `times` are ALL units on test — failures AND suspensions; every one enters the
    sum. `beta` must come from prior knowledge of the SAME failure mode (your
    Weibull Library); if the mechanism changed, the assumption — and the answer —
    do not hold (handbook 6.6).

    Two regimes (handbook 6.3-6.4):
      - with failures: pass ``r = number of failures``. eta is then the MLE of eta
        given beta, and (being invariant under transformation) so are its B-lives.
      - zero failures: ``r=1`` gives the lower bound on eta at 63.2% confidence,
        the first failure assumed imminent. For another level pass
        ``r=r_for_confidence(C)``.

    Returns eta, in the same units as `times`.
    """
    t = [float(x) for x in times]
    if not t:
        raise ValueError("times is empty")
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0; got {beta}")
    if r <= 0.0:
        raise ValueError(f"r must be > 0 (zero-failure tests use r=1; see 6.3); got {r}")
    return (sum(x ** beta for x in t) / r) ** (1.0 / beta)
