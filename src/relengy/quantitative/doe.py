"""DOE — Delineamento de Experimentos.

Convenção: fatores em **unidades codificadas ±1**, sempre. Não 0/1.
Com ±1 as colunas de efeitos principais e interações são ortogonais, os efeitos
são estimados independentemente, e o efeito de um fator é simplesmente a diferença
entre a média no nível alto e a média no nível baixo. `pyDOE3.fullfact` devolve
0/1, então convertemos.

Relação que confunde muita gente:

    efeito = 2 x coeficiente da regressão

O efeito é a mudança em y ao ir de -1 a +1 (duas unidades codificadas); o
coeficiente é a mudança por unidade. Ambos são reportados por `effects_table`.

Conexão com confiabilidade: DOE planeja **ensaios acelerados de vida (ALT)** —
quais combinações de estresse (temperatura, carga, tensão) rodar, e com que
resolução as interações podem ser separadas dos efeitos principais.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from pyDOE3 import ccdesign, fracfact, ff2n, pbdesign


# ---------------------------------------------------------------------------
# Geradores de matriz de projeto
# ---------------------------------------------------------------------------

def full_factorial(names: list[str]) -> pd.DataFrame:
    """Fatorial completo 2^k em ±1. Custa 2^k corridas."""
    m = ff2n(len(names))
    return pd.DataFrame(m, columns=names)


def fractional_factorial(generator: str) -> pd.DataFrame:
    """Fatorial fracionário 2^(k-p) a partir da string geradora do pyDOE3.

    Ex.: `fractional_factorial("a b c abc")` dá um 2^(4-1) — 8 corridas, 4 fatores,
    onde o 4º fator é confundido (aliased) com a interação tripla ABC.

    Fracionar não é de graça: você troca corridas por **confundimento**. Use
    `resolution()` para saber o que ficou misturado com o quê.
    """
    m = fracfact(generator)
    names = generator.split()
    return pd.DataFrame(m, columns=[f"x{i+1}" if len(n) > 1 else n
                                     for i, n in enumerate(names)])


def plackett_burman(n_factors: int) -> pd.DataFrame:
    """Plackett-Burman: triagem de muitos fatores com poucas corridas.

    Resolução III — efeitos principais confundidos com interações de 2 fatores.
    Serve para **peneirar** quais fatores importam, nunca para modelar interações.
    """
    m = pbdesign(n_factors)
    return pd.DataFrame(m[:, :n_factors], columns=[f"x{i+1}" for i in range(n_factors)])


def central_composite(n_factors: int, alpha: str = "orthogonal",
                      face: str = "circumscribed") -> pd.DataFrame:
    """Central Composite Design (RSM) — permite ajustar curvatura (termos quadráticos).

    Fatorial 2^k não detecta curvatura: com apenas dois níveis, todo fator é uma
    reta. O CCD adiciona pontos axiais e centrais para estimar o termo quadrático.
    """
    m = ccdesign(n_factors, center=(1, 1), alpha=alpha, face=face)
    return pd.DataFrame(m, columns=[f"x{i+1}" for i in range(n_factors)])


# ---------------------------------------------------------------------------
# Análise de efeitos
# ---------------------------------------------------------------------------

@dataclass
class Effect:
    term: str
    effect: float
    coefficient: float
    order: int


def _interaction_column(design: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    out = np.ones(len(design))
    for c in cols:
        out = out * design[c].to_numpy()
    return out


def effects_table(design: pd.DataFrame, response, max_order: int = 2) -> pd.DataFrame:
    """Efeitos principais e interações até `max_order`, em design ±1 balanceado.

    effect = média(y | coluna = +1) - média(y | coluna = -1)
    coefficient = effect / 2
    """
    y = np.asarray(response, dtype=float)
    if len(y) != len(design):
        raise ValueError(f"response has {len(y)} values, design has {len(design)} runs")

    vals = np.unique(design.to_numpy())
    if not np.all(np.isin(vals, [-1.0, 1.0])):
        raise ValueError(
            "design must be in coded units (±1); "
            f"found levels {vals.tolist()}. Use full_factorial()/code()."
        )

    rows: list[Effect] = []
    names = list(design.columns)
    for order in range(1, max_order + 1):
        for combo in combinations(names, order):
            col = _interaction_column(design, combo)
            hi, lo = y[col > 0], y[col < 0]
            if len(hi) == 0 or len(lo) == 0:
                continue  # coluna degenerada (confundida com a média)
            eff = hi.mean() - lo.mean()
            rows.append(Effect(":".join(combo), eff, eff / 2, order))

    df = pd.DataFrame(rows)
    df["abs_effect"] = df.effect.abs()
    return df.sort_values("abs_effect", ascending=False).drop(columns="abs_effect")


def code(values, low: float, high: float) -> np.ndarray:
    """Converte unidades naturais para ±1. O centro vira 0."""
    values = np.asarray(values, dtype=float)
    if high == low:
        raise ValueError("high and low cannot be equal")
    mid, half = (high + low) / 2.0, (high - low) / 2.0
    return (values - mid) / half


def decode(coded, low: float, high: float) -> np.ndarray:
    """Inverso de `code`."""
    coded = np.asarray(coded, dtype=float)
    mid, half = (high + low) / 2.0, (high - low) / 2.0
    return coded * half + mid


def pareto_of_effects(effects: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Ordena por |efeito|. É a leitura padrão: poucos fatores dominam."""
    return effects.reindex(effects.effect.abs().sort_values(ascending=False).index).head(top)


def is_orthogonal(design: pd.DataFrame) -> bool:
    """Colunas ortogonais <=> X'X é diagonal. Se não for, os efeitos se contaminam."""
    x = design.to_numpy(dtype=float)
    gram = x.T @ x
    return np.allclose(gram, np.diag(np.diag(gram)))
