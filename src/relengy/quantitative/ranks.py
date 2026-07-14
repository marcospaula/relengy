"""Median ranks with suspensions — New Weibull Handbook, ch. 2.9-2.11.

Bernard (eq. 2-6):   median rank = (i - 0.3) / (N + 0.4)
Auth/Johnson (2-5):  adjusted rank = [rev * prev + (N + 1)] / (rev + 1)

`rev` is the reverse rank and `prev` the adjusted rank of the previous failure.
With suspensions, `i` in Bernard's equation is the *adjusted* rank, not the raw
rank.
"""

from __future__ import annotations

import numpy as np


def bernard_median_rank(i, n: int) -> np.ndarray:
    """Bernard's approximation. Accurate to ~1% for n=5 and ~0.1% for n=50."""
    return (np.asarray(i, dtype=float) - 0.3) / (n + 0.4)


def adjusted_ranks(failures, right_censored=()) -> tuple[np.ndarray, np.ndarray]:
    """Auth/Johnson adjusted ranks for data with suspensions.

    Returns (sorted failure ages, adjusted ranks). Suspensions get no rank of
    their own — they only push the ranks of the failures that follow them
    upward, which is the mechanism behind "suspensions increase eta"
    (handbook 2.11).
    """
    f = np.asarray(failures, dtype=float)
    c = np.asarray(right_censored, dtype=float)
    n = f.size + c.size

    # Sort everything together, marking which entries are failures.
    ages = np.concatenate([f, c])
    is_failure = np.concatenate([np.ones(f.size, bool), np.zeros(c.size, bool)])
    order = np.argsort(ages, kind="stable")
    ages, is_failure = ages[order], is_failure[order]

    out_ages, out_ranks = [], []
    prev = 0.0
    for pos, (age, failed) in enumerate(zip(ages, is_failure)):
        if not failed:
            continue
        reverse_rank = n - pos          # pos is 0-indexed; the reverse rank is 1-indexed
        adj = (reverse_rank * prev + (n + 1)) / (reverse_rank + 1)
        out_ages.append(age)
        out_ranks.append(adj)
        prev = adj

    return np.array(out_ages), np.array(out_ranks)


def median_ranks_with_suspensions(failures, right_censored=()):
    """Convenience: returns (ages, median ranks as a fraction) ready to plot."""
    n = len(failures) + len(right_censored)
    ages, adj = adjusted_ranks(failures, right_censored)
    return ages, bernard_median_rank(adj, n)
