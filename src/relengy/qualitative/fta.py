"""Fault Tree Analysis: minimal cut sets e probabilidade do evento de topo.

A ponte entre qualitativo e quantitativo. A árvore é qualitativa (estrutura lógica
de como as falhas se combinam); os minimal cut sets são qualitativos (quais
combinações mínimas causam o topo); a probabilidade do topo é quantitativa e pode
ser alimentada por Weibull: P(basic event) = F(t) = 1 - exp(-(t/eta)^beta).

Implementado sobre estruturas nativas: um FTA é pequeno o bastante para não
justificar dependência pesada, e a álgebra booleana aqui é direta.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

Gate = Literal["AND", "OR"]


@dataclass(frozen=True)
class Node:
    name: str


@dataclass
class FaultTree:
    """Árvore definida por um dicionário: nó -> (gate, [filhos]).

    Folhas (basic events) simplesmente não aparecem como chave.
    """

    top: str
    gates: dict[str, tuple[Gate, list[str]]]

    def basic_events(self) -> set[str]:
        internal = set(self.gates)
        children = {c for _, kids in self.gates.values() for c in kids}
        return children - internal

    def _cut_sets(self, node: str) -> list[frozenset[str]]:
        if node not in self.gates:
            return [frozenset({node})]

        gate, children = self.gates[node]
        child_sets = [self._cut_sets(c) for c in children]

        if gate == "OR":
            # União: qualquer filho sozinho causa o pai.
            return [cs for sets in child_sets for cs in sets]
        if gate == "AND":
            # Produto cartesiano: precisa de todos os filhos simultaneamente.
            return [frozenset().union(*combo) for combo in product(*child_sets)]
        raise ValueError(f"unknown gate: {gate!r}")

    def minimal_cut_sets(self) -> list[frozenset[str]]:
        """Cut sets após remover os supersets (minimalização por absorção)."""
        cuts = self._cut_sets(self.top)
        # Ordena por tamanho: um set só pode ser absorvido por um menor ou igual.
        cuts = sorted(set(cuts), key=len)
        minimal: list[frozenset[str]] = []
        for c in cuts:
            if not any(m <= c for m in minimal):
                minimal.append(c)
        return minimal

    def top_probability(self, probs: dict[str, float], exact: bool = True) -> float:
        """Probabilidade do evento de topo, dados os basic events INDEPENDENTES.

        A independência é suposição, não fato. Itens do mesmo lote, sob o mesmo
        ambiente ou mantidos pela mesma equipe falham juntos. Sob causa comum,
        este resultado **subestima** o topo num gate AND e **superestima** num OR.
        Para modelar dependência é preciso sair da álgebra de FTA — uma rede
        bayesiana sobre os mesmos eventos reproduz exatamente este valor quando
        não há causa comum, e diverge dele quando há.

        `exact=True` usa inclusão-exclusão sobre os minimal cut sets — correto,
        porém exponencial no número de cut sets.
        `exact=False` usa a rare-event approximation (soma das probabilidades dos
        cut sets), que SUPERESTIMA o topo e só é aceitável com probabilidades
        pequenas. Ela pode passar de 1.0; nesse caso o resultado é truncado e o
        valor deve ser tratado como um limite superior sem significado.
        """
        missing = self.basic_events() - set(probs)
        if missing:
            raise ValueError(f"no probability given for basic events: {sorted(missing)}")

        mcs = self.minimal_cut_sets()

        def p_cut(cut: frozenset[str]) -> float:
            p = 1.0
            for e in cut:
                p *= probs[e]
            return p

        if not exact:
            return min(1.0, sum(p_cut(c) for c in mcs))

        # Inclusão-exclusão: P(U Ci) = sum P(Ci) - sum P(Ci n Cj) + ...
        total = 0.0
        n = len(mcs)
        for mask in range(1, 1 << n):
            union: set[str] = set()
            bits = 0
            for i in range(n):
                if mask >> i & 1:
                    union |= mcs[i]
                    bits += 1
            p = 1.0
            for e in union:
                p *= probs[e]
            total += p if bits % 2 else -p
        return total


def weibull_cdf(t: float, beta: float, eta: float) -> float:
    """F(t) para alimentar basic events a partir da Weibull Library."""
    import math

    return 1.0 - math.exp(-((t / eta) ** beta))
