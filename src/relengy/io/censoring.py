"""Conversão da convenção de dados do handbook/SuperSMITH para os pacotes Python.

Abernethy e o SuperSMITH usam uma coluna única onde o sinal carrega a semântica:
valor positivo é uma falha, valor negativo é uma suspensão (censura à direita).
Os pacotes Python (`reliability`, `predictr`, `surpyval`) querem duas listas.
"""

from __future__ import annotations

import numpy as np


def split_signed(values) -> tuple[np.ndarray, np.ndarray]:
    """Separa a convenção de coluna única com sinal em (falhas, suspensões).

    Suspensões chegam negativas e retornam com o valor absoluto.

    >>> f, s = split_signed([504, -91, 1200, -300])
    >>> f.tolist(), s.tolist()
    ([504.0, 1200.0], [91.0, 300.0])
    """
    arr = np.asarray(values, dtype=float)

    if np.any(arr == 0):
        raise ValueError(
            "age zero is ambiguous in this convention: the sign cannot tell a "
            "failure from a suspension at t=0. Handle those records explicitly."
        )
    if np.any(np.isnan(arr)):
        raise ValueError("NaN values present; clean the data before converting.")

    failures = arr[arr > 0]
    censored = -arr[arr < 0]
    return failures, censored


def to_signed(failures, right_censored=()) -> np.ndarray:
    """Operação inversa: volta para a coluna única com sinal."""
    f = np.asarray(failures, dtype=float)
    c = np.asarray(right_censored, dtype=float)
    if np.any(f <= 0) or np.any(c <= 0):
        raise ValueError("ages must be strictly positive.")
    return np.concatenate([f, -c])
