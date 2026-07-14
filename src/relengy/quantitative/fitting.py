"""Ajuste Weibull: escolha de método e diagnóstico.

Duas fontes, que **não dizem exatamente a mesma coisa**. A diferença importa.

**Abernethy** (New Weibull Handbook, 5.3.3) é categórico:
    "For engineers, the author recommends MRR as best practice [...]
     MLE is not recommended."

**ReliaSoft / ReliaWiki** (Parameter Estimation) qualifica:
    "our recommendation is to use rank regression techniques when the sample
     sizes are small and without heavy censoring. When heavy or uneven censoring
     is present, when a high proportion of interval data is present and/or when
     the sample size is sufficient, MLE should be preferred."

E acrescenta o motivo pelo qual ambos desconfiam do MLE em amostra pequena:
    "MLE estimates of the shape parameter for the Weibull distribution are badly
     biased for small sample sizes, and the effect can be increased depending on
     the amount of censoring."

**Síntese adotada aqui:** MRR (RRX) é o padrão, como quer Abernethy — mas ele
falha justamente onde o ReliaSoft aponta, porque MRR depende de *plotting
positions* e, sob censura pesada ou desigual, os ranks ajustados carregam pouca
informação. Nesses casos o MLE ganha. `recommend_method()` decide e explica.

Rodar MRR e MLE juntos continua sendo diagnóstico: beta_MLE muito menor que
beta_MRR sugere problema de lote (batch problem), não erro numérico.
"""

from __future__ import annotations

from dataclasses import dataclass

from reliability.Fitters import Fit_Weibull_2P

# Limiar da razão beta_mle/beta_mrr abaixo do qual suspeitamos de batch problem.
# O handbook diz "much less" sem fixar número; 0.75 é escolha NOSSA, conservadora,
# para sinalizar e não para concluir. Calibre com a sua Weibull Library.
BATCH_SUSPICION_RATIO = 0.75

# "Amostra pequena" = <= 20 falhas. Este número é do handbook (e dos slides de
# Fulton & Tarum, que o citam explicitamente).
SMALL_SAMPLE_FAILURES = 20

# "Censura pesada" — nem Abernethy nem o ReliaWiki fixam um número; ambos dizem
# "heavy". 0.5 (metade ou mais dos registros são suspensões) é escolha NOSSA.
# Ajuste se a sua Weibull Library indicar outro ponto de virada.
HEAVY_CENSORING_FRACTION = 0.5

# Banda em torno de beta = 1 para o regime de "falhas aleatórias". Um beta de um
# ajuste real nunca cai exatamente em 1.0, então uma igualdade exata jamais
# dispararia esse ramo. ±0.05 trata o beta como estatisticamente indistinguível
# de 1 dentro do ruído típico de um MRR — sinalização, não teste formal.
RANDOM_FAILURE_BAND = 0.05


@dataclass
class MethodRecommendation:
    method: str          # 'RRX' ou 'MLE'
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
    """Escolhe entre MRR (RRX) e MLE, reconciliando Abernethy e ReliaSoft.

    `uneven_censoring`: as suspensões se concentram (p.ex. todas no fim do teste)
    em vez de se espalhar entre as falhas. O ReliaWiki trata isso como tão
    decisivo quanto a censura pesada — e o julgamento é seu, não dá para inferir
    das contagens.

    `interval_data`: houve inspeção periódica, você sabe apenas que a falha
    ocorreu *entre* duas inspeções. MRR não lida bem com isso.
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
        """<= 20 falhas é 'amostra pequena' pelo padrão do handbook."""
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
        """Distância (com sinal) da razão até o corte de suspeita de batch.

        `BATCH_SUSPICION_RATIO - ratio`: positiva quando `batch_suspected` (a
        razão está ABAIXO do corte — e quanto maior, mais forte o sinal),
        negativa quando há folga. Existe para que 0.74 e 0.20 não soem
        igualmente conclusivos só por dispararem o mesmo booleano: expõe o
        *quão longe* do limiar, que é o julgamento que o booleano esconde.
        """
        return BATCH_SUSPICION_RATIO - self.ratio

    def regime(self) -> str:
        """Leitura física de beta (handbook 2.13-2.16), a partir do MRR."""
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
        linhas = [
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
            linhas.append(
                f"ALERT: MLE/MRR beta ratio {self.ratio:.3f} sits {self.batch_margin:.3f} "
                f"past the {BATCH_SUSPICION_RATIO} line -> suspected batch problem "
                "(handbook 5.3.3 / 3.9). The wider this gap, the stronger the signal; "
                "investigate a mixture of lots before trusting the fit."
            )
        if self.small_sample:
            linhas.append(
                "NOTE: small sample (<= 20 failures). Consider Weibayes, with beta "
                "taken from your Weibull Library (handbook ch. 6)."
            )
        return "\n".join(linhas)


def fit_diagnostic(failures, right_censored=None) -> WeibullDiagnosis:
    """Ajusta Weibull 2P por MRR e por MLE e devolve o par para comparação.

    Exige ao menos 2 falhas. Com 0 ou 1 falha uma Weibull de 2 parâmetros não se
    ajusta por regressão (não há reta por menos de dois pontos) e o MLE é pouco
    confiável — é o caso de Weibayes (beta fixo), não deste diagnóstico. Ver
    `recommend_method`, que trata n=0 explicitamente.
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
        eta_mrr=mrr.alpha,  # `reliability` chama eta de alpha
        beta_mle=mle.beta,
        eta_mle=mle.alpha,
        n_failures=len(common["failures"]),
        n_censored=len(rc) if rc else 0,
    )
