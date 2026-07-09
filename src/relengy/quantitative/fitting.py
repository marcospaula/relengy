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


@dataclass
class MethodRecommendation:
    method: str          # 'RRX' ou 'MLE'
    reason: str
    caution: str | None = None

    def __str__(self) -> str:
        s = f"{self.method}: {self.reason}"
        return f"{s}\n  CUIDADO: {self.caution}" if self.caution else s


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
            "sem falhas observadas; MRR é impossível (não há ranks a ajustar)",
            caution="considere Weibayes (beta fixado) — handbook cap. 6",
        )

    if interval_data:
        return MethodRecommendation(
            "MLE",
            "dados intervalares (inspeção): MRR depende de tempos exatos de falha",
        )

    if heavy or uneven_censoring:
        motivo = []
        if heavy:
            motivo.append(f"censura pesada ({frac:.0%} suspensões)")
        if uneven_censoring:
            motivo.append("censura desigual")
        rec = MethodRecommendation(
            "MLE",
            " e ".join(motivo) + ": as plotting positions do MRR perdem informação",
        )
        if small:
            rec.caution = (
                f"amostra pequena ({n_failures} falhas): o beta do MLE é "
                "notoriamente enviesado aqui. Use correção de viés (RBA) — "
                "o pacote `predictr` implementa — ou Weibayes."
            )
        return rec

    if small:
        return MethodRecommendation(
            "RRX",
            f"amostra pequena ({n_failures} falhas) e censura leve: "
            "exatamente o caso em que Abernethy e ReliaSoft concordam",
        )

    return MethodRecommendation(
        "RRX",
        "censura leve; MRR é a best practice do handbook e dá o gráfico",
        caution="com n grande e sem censura, MLE também é válido — "
                "se discordarem muito, investigue batch problem",
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

    def regime(self) -> str:
        """Leitura física de beta (handbook 2.13-2.16), a partir do MRR."""
        b = self.beta_mrr
        if b < 1.0:
            return "mortalidade infantil (beta < 1)"
        if b == 1.0:
            return "falhas aleatorias (beta = 1)"
        if b < 4.0:
            return "desgaste inicial (1 < beta < 4)"
        return "desgaste de velhice, rapido (beta > 4)"

    def report(self, uneven_censoring: bool = False,
               interval_data: bool = False) -> str:
        rec = self.recommendation(uneven_censoring, interval_data)
        linhas = [
            f"n = {self.n_failures} falhas, {self.n_censored} suspensoes "
            f"({self.censoring_fraction:.0%} censurado)",
            f"MRR: beta = {self.beta_mrr:.4f}  eta = {self.eta_mrr:.2f}",
            f"MLE: beta = {self.beta_mle:.4f}  eta = {self.eta_mle:.2f}",
            f"razao beta_mle/beta_mrr = {self.ratio:.3f}",
            f"regime: {self.regime()}",
            "",
            f"METODO RECOMENDADO -> {rec}",
        ]
        if self.batch_suspected:
            linhas.append(
                "ALERTA: beta do MLE bem abaixo do MRR -> suspeita de batch problem "
                "(handbook 5.3.3 / 3.9). Investigue mistura de lotes antes de usar o ajuste."
            )
        if self.small_sample:
            linhas.append(
                "NOTA: amostra pequena (<= 20 falhas). Considere Weibayes com beta "
                "vindo da Weibull Library (handbook cap. 6)."
            )
        return "\n".join(linhas)


def fit_diagnostic(failures, right_censored=None) -> WeibullDiagnosis:
    """Ajusta Weibull 2P por MRR e por MLE e devolve o par para comparação."""
    rc = list(right_censored) if right_censored is not None else None
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
