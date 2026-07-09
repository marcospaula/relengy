"""RBD — Reliability Block Diagrams e medidas de importância.

Lacuna real: o pacote `reliability` cobre LDA, ALT e sistemas reparáveis, mas
**não tem RBD**. Este módulo preenche.

Configurações (ReliaWiki, *RBDs and Analytical System Reliability*):

    Série     R = Π Rᵢ                     todos precisam funcionar
    Paralelo  R = 1 − Π (1 − Rᵢ)           basta um funcionar
    k-de-n    ao menos k dos n funcionam   (paralelo é 1-de-n; série é n-de-n)

**Importância de Birnbaum** = ∂R_sys/∂Rᵢ — quanto a confiabilidade do sistema
muda por unidade de mudança no componente i. Calculada exatamente por decomposição
pivotal, sem derivada numérica:

    R_sys = Rᵢ·R_sys(Rᵢ=1) + (1−Rᵢ)·R_sys(Rᵢ=0)
    ⟹ ∂R_sys/∂Rᵢ = R_sys(Rᵢ=1) − R_sys(Rᵢ=0)

Note que Birnbaum **não depende de Rᵢ**. Um componente já muito confiável pode ter
Birnbaum alto: ele mede o *potencial* estrutural, não onde está o risco atual. Para
priorizar melhoria use `criticality_importance`, que pondera pelo estado atual.

⚠ Assume componentes **independentes**, como toda álgebra de RBD. Sob causa comum
o resultado deixa de valer, e a saída é modelar a dependência explicitamente —
com uma rede bayesiana, ou com um fator de causa comum (beta-factor).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class Node:
    """Base. Um RBD é uma árvore de Series/Parallel/KooN com folhas Component."""

    def reliability(self, r: dict[str, float]) -> float:
        raise NotImplementedError

    def components(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Component(Node):
    name: str

    def reliability(self, r: dict[str, float]) -> float:
        try:
            return r[self.name]
        except KeyError:
            raise KeyError(f"sem confiabilidade para o componente {self.name!r}") from None

    def components(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Series(Node):
    children: tuple[Node, ...]

    def __init__(self, *children: Node):
        if not children:
            raise ValueError("Series precisa de ao menos um bloco")
        object.__setattr__(self, "children", tuple(children))

    def reliability(self, r: dict[str, float]) -> float:
        out = 1.0
        for c in self.children:
            out *= c.reliability(r)
        return out

    def components(self) -> set[str]:
        return set().union(*(c.components() for c in self.children))


@dataclass(frozen=True)
class Parallel(Node):
    children: tuple[Node, ...]

    def __init__(self, *children: Node):
        if not children:
            raise ValueError("Parallel precisa de ao menos um bloco")
        object.__setattr__(self, "children", tuple(children))

    def reliability(self, r: dict[str, float]) -> float:
        out = 1.0
        for c in self.children:
            out *= 1.0 - c.reliability(r)
        return 1.0 - out

    def components(self) -> set[str]:
        return set().union(*(c.components() for c in self.children))


@dataclass(frozen=True)
class KooN(Node):
    """k-de-n: o sistema funciona se ao menos `k` dos `n` blocos funcionarem.

    Para blocos com confiabilidades diferentes não há fórmula fechada simples;
    usamos programação dinâmica sobre a distribuição do número de blocos ativos
    (Poisson-binomial). Exato, O(n²).
    """

    k: int
    children: tuple[Node, ...]

    def __init__(self, k: int, *children: Node):
        n = len(children)
        if n == 0:
            raise ValueError("KooN precisa de blocos")
        if not 1 <= k <= n:
            raise ValueError(f"k={k} fora de 1..{n}")
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "children", tuple(children))

    def reliability(self, r: dict[str, float]) -> float:
        probs = [c.reliability(r) for c in self.children]
        # dist[j] = P(exatamente j blocos funcionando)
        dist = [1.0]
        for p in probs:
            nxt = [0.0] * (len(dist) + 1)
            for j, d in enumerate(dist):
                nxt[j] += d * (1.0 - p)
                nxt[j + 1] += d * p
            dist = nxt
        return sum(dist[self.k:])

    def components(self) -> set[str]:
        return set().union(*(c.components() for c in self.children))


# ---------------------------------------------------------------------------
# Medidas de importância
# ---------------------------------------------------------------------------

def _with(r: dict[str, float], name: str, value: float) -> dict[str, float]:
    d = dict(r)
    d[name] = value
    return d


def birnbaum_importance(system: Node, r: dict[str, float]) -> dict[str, float]:
    """∂R_sys/∂Rᵢ por decomposição pivotal. Exato, sem derivada numérica.

    Não depende de Rᵢ. Mede potencial estrutural, não risco atual.
    """
    _validate(system, r)
    return {
        name: system.reliability(_with(r, name, 1.0))
        - system.reliability(_with(r, name, 0.0))
        for name in sorted(system.components())
    }


def criticality_importance(system: Node, r: dict[str, float]) -> dict[str, float]:
    """I_C(i) = I_B(i) · (1 − Rᵢ) / (1 − R_sys).

    Probabilidade de que o componente i esteja falhado **e** seja o responsável
    pela falha do sistema, dado que o sistema falhou. É a medida certa para
    priorizar onde investir: pondera o potencial (Birnbaum) pelo estado atual.
    """
    _validate(system, r)
    r_sys = system.reliability(r)
    if math.isclose(r_sys, 1.0):
        raise ValueError(
            "R_sys = 1: nenhum componente pode ser crítico; a importância de "
            "criticidade é indefinida (divisão por zero)"
        )
    ib = birnbaum_importance(system, r)
    return {n: ib[n] * (1.0 - r[n]) / (1.0 - r_sys) for n in sorted(ib)}


def _validate(system: Node, r: dict[str, float]) -> None:
    missing = system.components() - set(r)
    if missing:
        raise KeyError(f"sem confiabilidade para: {sorted(missing)}")
    bad = {n: v for n, v in r.items() if not 0.0 <= v <= 1.0}
    if bad:
        raise ValueError(f"confiabilidades fora de [0,1]: {bad}")


# ---------------------------------------------------------------------------
# Ponte com a Weibull Library
# ---------------------------------------------------------------------------

def weibull_reliability(t: float, beta: float, eta: float) -> float:
    """R(t) = e^(−(t/η)^β). Alimenta o RBD a partir dos ajustes."""
    if t < 0:
        raise ValueError("t < 0")
    if beta <= 0 or eta <= 0:
        raise ValueError("beta e eta devem ser positivos")
    return math.exp(-((t / eta) ** beta))


def reliabilities_at(t: float, params: dict[str, tuple[float, float]]) -> dict[str, float]:
    """Converte {componente: (beta, eta)} em {componente: R(t)}."""
    return {n: weibull_reliability(t, b, e) for n, (b, e) in params.items()}


def mission_reliability(system: Node, t: float,
                        params: dict[str, tuple[float, float]]) -> float:
    """R_sys(t) a partir dos β/η de cada componente."""
    return system.reliability(reliabilities_at(t, params))


def series_of(names: Sequence[str]) -> Series:
    return Series(*(Component(n) for n in names))


def parallel_of(names: Sequence[str]) -> Parallel:
    return Parallel(*(Component(n) for n in names))
