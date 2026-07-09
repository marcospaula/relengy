"""Crescimento de confiabilidade: Crow-AMSAA (NHPP) e Duane.

Para **sistemas reparáveis** em desenvolvimento. Não confunda com análise de dados
de vida: aqui os tempos são *acumulados* de um único sistema (ou frota) que vai
sendo corrigido, não idades de itens independentes até falhar.

⚠ **O β do Crow-AMSAA NÃO é o β da Weibull.** Mesmo símbolo, significado oposto:

| β | Weibull (LDA) | Crow-AMSAA (NHPP) |
|---|---|---|
| β < 1 | mortalidade infantil (ruim) | **confiabilidade crescendo** (bom) |
| β = 1 | falhas aleatórias | Poisson homogêneo: sem crescimento |
| β > 1 | desgaste | **confiabilidade piorando** |

A intensidade de falha é ρ(T) = λ·β·T^(β−1). Com β < 1 ela decresce — o sistema
está melhorando. É a confusão mais comum ao ler os dois capítulos em sequência.

Estimadores (ReliaWiki, *Crow-AMSAA (NHPP)*, seções de MLE e "Biasing and
Unbiasing of Beta"), verificados contra o exemplo resolvido do próprio wiki:

    β̂ = n / (n·ln T* − Σ ln Tᵢ)          mesma fórmula nos dois tipos de teste
    λ̂ = n / T*^β

onde T* = Tₙ se o teste termina na falha, T* = tempo final se termina no tempo.

Desviesamento (aí sim difere):

    teste terminado por TEMPO:  β̄ = (N−1)/N · β̂
    teste terminado por FALHA:  β̄ = (N−2)/(N−1) · β̂
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

    # -- intensidade de falha ---------------------------------------------

    def instantaneous_intensity(self, t: float | None = None) -> float:
        """ρ(T) = λ·β·T^(β−1). A taxa de falha *agora*."""
        t = self.t_end if t is None else t
        return self.lam * self.beta * t ** (self.beta - 1.0)

    def cumulative_intensity(self, t: float | None = None) -> float:
        """λ_c(T) = λ·T^(β−1). A média histórica, não a taxa atual."""
        t = self.t_end if t is None else t
        return self.lam * t ** (self.beta - 1.0)

    def expected_failures(self, t: float | None = None) -> float:
        """E[N(T)] = λ·T^β"""
        t = self.t_end if t is None else t
        return self.lam * t**self.beta

    # -- MTBF --------------------------------------------------------------

    def instantaneous_mtbf(self, t: float | None = None) -> float:
        """O MTBF que o sistema tem AGORA. É este que se reporta ao cliente."""
        return 1.0 / self.instantaneous_intensity(t)

    def cumulative_mtbf(self, t: float | None = None) -> float:
        """MTBF médio desde o início do teste. Sempre otimista? Não: com
        crescimento (β<1) ele é PESSIMISTA — carrega as falhas antigas."""
        return 1.0 / self.cumulative_intensity(t)

    # -- leitura -----------------------------------------------------------

    def verdict(self) -> str:
        if self.beta < 1.0:
            return f"reliability IMPROVING (beta={self.beta:.4f} < 1)"
        if self.beta == 1.0:
            return "no growth: homogeneous Poisson process (beta = 1)"
        return f"reliability DETERIORATING (beta={self.beta:.4f} > 1)"

    def time_to_reach_mtbf(self, target_mtbf: float) -> float:
        """Quanto tempo de teste até o MTBF instantâneo atingir o alvo.

        Só faz sentido com β < 1 (crescendo). Assume que a taxa de crescimento
        atual se mantém — extrapolação, e o handbook avisa que ela costuma saturar.
        """
        if self.beta >= 1.0:
            raise ValueError(
                f"beta = {self.beta:.4f} >= 1: the system is not improving, "
                "the instantaneous MTBF never reaches the target"
            )
        # 1/(λβT^(β−1)) = alvo  =>  T^(1−β) = alvo·λ·β
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
    """Ajusta o modelo Crow-AMSAA por máxima verossimilhança.

    `failure_times` são tempos **acumulados** de teste, crescentes.
    `t_end` só é necessário se `termination='time'` (e deve ser > último tempo).
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
        beta *= (n - 1) / n if termination == "time" else (n - 2) / (n - 1)

    lam = n / t_star**beta
    return CrowAMSAA(beta, lam, n, t_star, termination, unbiased)


# ---------------------------------------------------------------------------
# Duane
# ---------------------------------------------------------------------------

@dataclass
class Duane:
    alpha: float   # taxa de crescimento (inclinação no log-log)
    b: float       # intercepto
    n_failures: int

    def cumulative_mtbf(self, t: float) -> float:
        """m_c(T) = b·T^α"""
        return self.b * t**self.alpha

    def instantaneous_mtbf(self, t: float) -> float:
        """m_i = m_c / (1 − α). Diverge quando α → 1."""
        if self.alpha >= 1.0:
            raise ValueError(f"alpha = {self.alpha:.4f} >= 1: instantaneous MTBF is undefined")
        return self.cumulative_mtbf(t) / (1.0 - self.alpha)


def duane(failure_times: Sequence[float]) -> Duane:
    """Duane por regressão nos mínimos quadrados em papel log-log.

    Duane observou que o MTBF **cumulativo** contra o tempo cumulativo cai numa
    reta em log-log. É o mesmo fenômeno do Crow-AMSAA, ajustado por regressão em
    vez de verossimilhança — daí `alpha ≈ 1 − beta`.
    """
    t = np.asarray(failure_times, dtype=float)
    if t.size < 2:
        raise ValueError("Duane needs at least 2 failures")
    if np.any(np.diff(t) <= 0):
        raise ValueError("cumulative times must be strictly increasing")

    n = np.arange(1, t.size + 1)
    mtbf_c = t / n                       # MTBF cumulativo observado
    slope, intercept = np.polyfit(np.log(t), np.log(mtbf_c), 1)
    return Duane(alpha=float(slope), b=float(math.exp(intercept)), n_failures=t.size)
