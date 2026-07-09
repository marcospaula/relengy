"""Median ranks com suspensões — New Weibull Handbook, cap. 2.9-2.11.

Bernard (eq. 2-6):   median rank = (i - 0.3) / (N + 0.4)
Auth/Johnson (2-5):  adjusted rank = [rev * prev + (N + 1)] / (rev + 1)

Onde `rev` é o reverse rank e `prev` o adjusted rank da falha anterior.
Com suspensões, `i` na equação de Bernard é o *adjusted* rank, não o rank bruto.
"""

from __future__ import annotations

import numpy as np


def bernard_median_rank(i, n: int) -> np.ndarray:
    """Aproximação de Bernard. Exata a ~1% para n=5 e ~0.1% para n=50."""
    return (np.asarray(i, dtype=float) - 0.3) / (n + 0.4)


def adjusted_ranks(failures, right_censored=()) -> tuple[np.ndarray, np.ndarray]:
    """Ranks ajustados de Auth/Johnson para dados com suspensões.

    Retorna (idades_de_falha_ordenadas, adjusted_ranks). Suspensões não recebem
    rank — elas apenas empurram os ranks das falhas subsequentes para cima, que é
    o mecanismo pelo qual "suspensions increase eta" (handbook 2.11).
    """
    f = np.asarray(failures, dtype=float)
    c = np.asarray(right_censored, dtype=float)
    n = f.size + c.size

    # Ordena tudo junto, marcando o que é falha.
    ages = np.concatenate([f, c])
    is_failure = np.concatenate([np.ones(f.size, bool), np.zeros(c.size, bool)])
    order = np.argsort(ages, kind="stable")
    ages, is_failure = ages[order], is_failure[order]

    out_ages, out_ranks = [], []
    prev = 0.0
    for pos, (age, failed) in enumerate(zip(ages, is_failure)):
        if not failed:
            continue
        reverse_rank = n - pos          # pos é 0-indexed; reverse rank é 1-indexed
        adj = (reverse_rank * prev + (n + 1)) / (reverse_rank + 1)
        out_ages.append(age)
        out_ranks.append(adj)
        prev = adj

    return np.array(out_ages), np.array(out_ranks)


def median_ranks_with_suspensions(failures, right_censored=()):
    """Conveniência: devolve (idades, median ranks em fração) prontos para plotar."""
    n = len(failures) + len(right_censored)
    ages, adj = adjusted_ranks(failures, right_censored)
    return ages, bernard_median_rank(adj, n)
