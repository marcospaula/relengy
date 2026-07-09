"""Valida os median ranks contra a Tabela 2-3 do New Weibull Handbook (p. 2-7).

Dados "RIVET FAILURES": suspensões em 10, 45, 100; falhas em 30, 49, 82, 90, 96.
N = 8. O livro publica os adjusted ranks e os median ranks; usamos como gabarito.
"""

import numpy as np
import pytest

from relengy.io.censoring import split_signed
from relengy.quantitative.ranks import (
    adjusted_ranks,
    bernard_median_rank,
    median_ranks_with_suspensions,
)

FAILURES = [30, 49, 82, 90, 96]
SUSPENSIONS = [10, 45, 100]

# Tabela 2-3, coluna "Adjusted Rank (i)"
EXPECTED_ADJ = [1.125, 2.438, 3.750, 5.063, 6.375]

# Tabela 2-3, coluna "Median Rank", como IMPRESSA no livro.
# O segundo valor (25.5) está errado no livro: com adjusted rank 2.4375 e N=8,
# Bernard dá (2.4375 - 0.3)/8.4 = 25.446%, que arredonda para 25.4 e não 25.5.
# Os outros quatro conferem. Erratum tipográfico (ou ruído de OCR: 4 lido como 5).
# Mantemos o valor do livro visível e testamos contra o valor correto.
BOOK_MR_PCT = [9.8, 25.5, 41.1, 56.7, 72.3]
CORRECT_MR_PCT = [9.8, 25.4, 41.1, 56.7, 72.3]
BOOK_ERRATUM_INDEX = 1


def test_adjusted_ranks_match_table_2_3():
    ages, adj = adjusted_ranks(FAILURES, SUSPENSIONS)
    assert ages.tolist() == FAILURES
    np.testing.assert_allclose(adj, EXPECTED_ADJ, atol=1e-3)


def test_median_ranks_match_table_2_3():
    _, mr = median_ranks_with_suspensions(FAILURES, SUSPENSIONS)
    np.testing.assert_allclose(mr * 100, CORRECT_MR_PCT, atol=0.05)


def test_book_table_2_3_has_a_rounding_erratum():
    """Fixa a discrepância conhecida, para não a 'redescobrirmos' como bug nosso."""
    _, mr = median_ranks_with_suspensions(FAILURES, SUSPENSIONS)
    i = BOOK_ERRATUM_INDEX
    ours = mr[i] * 100
    assert ours == pytest.approx(25.446, abs=1e-3)
    assert round(ours, 1) == CORRECT_MR_PCT[i] == 25.4
    assert BOOK_MR_PCT[i] == 25.5  # o que está impresso, e que não reproduzimos


def test_bernard_worked_example_from_page_2_8():
    """O livro: adjusted rank 1.125, N=8 -> 9.82%."""
    assert bernard_median_rank(1.125, 8) * 100 == pytest.approx(9.82, abs=0.01)


def test_suspensions_do_not_shift_ranks_before_they_occur():
    """Handbook 2.9: 'suspended items do not affect rank numbers until after they occur.'

    A suspensão em 10 precede toda falha, então ela NÃO deve deixar o primeiro
    adjusted rank em 1.0 — ela conta em N e no reverse rank. Já uma suspensão
    posterior a todas as falhas não pode mexer em nada antes dela.
    """
    _, adj_com = adjusted_ranks([30, 49], [100])
    _, adj_sem = adjusted_ranks([30, 49], [])
    # A suspensão em 100 ocorre depois das duas falhas: ranks das falhas
    # permanecem 1 e 2 (inalterados), apesar de N mudar de 2 para 3.
    np.testing.assert_allclose(adj_com, [1.0, 2.0], atol=1e-9)
    np.testing.assert_allclose(adj_sem, [1.0, 2.0], atol=1e-9)


def test_split_signed_roundtrip():
    f, s = split_signed([504, -91, 1200, -300])
    assert f.tolist() == [504.0, 1200.0]
    assert s.tolist() == [91.0, 300.0]


def test_split_signed_rejects_zero():
    with pytest.raises(ValueError, match="ambiguous"):
        split_signed([100, 0, -50])
