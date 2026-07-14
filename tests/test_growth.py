"""Validado contra o exemplo resolvido do ReliaWiki, Crow-AMSAA (NHPP).

22 falhas, teste terminado por falha em T = 620 h.
Gabarito publicado: sum ln Ti = 105.6355, beta = 0.6142, lambda = 0.4239,
lambda_i(620) = 0.0217906 falhas/h, MTBF instantaneo = 46 h.
"""
import math
import pytest

from relengy.quantitative.growth import CrowAMSAA, crow_amsaa, duane

WIKI = [2.7, 10.3, 12.5, 30.6, 57.0, 61.3, 80.0, 109.5, 125.0, 128.6, 143.8,
        167.9, 229.2, 296.7, 320.6, 328.2, 366.2, 396.7, 421.1, 438.2, 501.2, 620.0]


def test_sum_of_logs_matches_wiki():
    assert sum(math.log(t) for t in WIKI) == pytest.approx(105.6355, abs=1e-4)

def test_beta_matches_wiki():
    ca = crow_amsaa(WIKI)
    assert ca.beta == pytest.approx(0.6142, abs=1e-4)

def test_lambda_matches_wiki():
    assert crow_amsaa(WIKI).lam == pytest.approx(0.4239, abs=1e-4)

def test_instantaneous_intensity_matches_wiki():
    # o wiki calcula com beta arredondado (0.6142); por isso a tolerancia
    assert crow_amsaa(WIKI).instantaneous_intensity() == pytest.approx(0.0217906, abs=1e-5)

def test_instantaneous_mtbf_matches_wiki():
    assert crow_amsaa(WIKI).instantaneous_mtbf() == pytest.approx(46.0, abs=0.5)

def test_beta_below_one_means_growth_not_infant_mortality():
    """O beta do Crow-AMSAA e o oposto do beta da Weibull."""
    ca = crow_amsaa(WIKI)
    assert ca.beta < 1
    assert "IMPROVING" in ca.verdict()

def test_cumulative_mtbf_is_pessimistic_under_growth():
    """Com beta<1 o MTBF cumulativo carrega as falhas antigas: subestima o atual."""
    ca = crow_amsaa(WIKI)
    assert ca.cumulative_mtbf() < ca.instantaneous_mtbf()

def test_expected_failures_recovers_n_at_t_end():
    """E[N(T*)] = lambda*T*^beta = n, por construcao do MLE."""
    ca = crow_amsaa(WIKI)
    assert ca.expected_failures() == pytest.approx(len(WIKI), rel=1e-9)


# ---- desviesamento: as duas formulas sao diferentes ----

def test_unbiased_failure_terminated_uses_n_minus_2_over_n_minus_1():
    b = crow_amsaa(WIKI).beta
    u = crow_amsaa(WIKI, unbiased=True).beta
    n = len(WIKI)
    assert u == pytest.approx(b * (n - 2) / (n - 1))

def test_unbiased_time_terminated_uses_n_minus_1_over_n():
    b = crow_amsaa(WIKI, t_end=700, termination="time").beta
    u = crow_amsaa(WIKI, t_end=700, termination="time", unbiased=True).beta
    n = len(WIKI)
    assert u == pytest.approx(b * (n - 1) / n)

def test_unbiased_beta_is_smaller():
    assert crow_amsaa(WIKI, unbiased=True).beta < crow_amsaa(WIKI).beta


def test_unbiased_failure_terminated_needs_three_failures():
    """N=2 zera o fator (N-2)/(N-1) -> beta=0 silencioso; o guard rejeita."""
    with pytest.raises(ValueError, match="at least 3"):
        crow_amsaa([10, 20], unbiased=True)
    assert crow_amsaa([10, 20, 30], unbiased=True).beta > 0          # N=3 ja passa
    # time-terminated com N=2 continua valido (fator (N-1)/N = 1/2, nao zera)
    assert crow_amsaa([10, 20], t_end=30, termination="time", unbiased=True).beta > 0


# ---- time terminated ----

def test_time_terminated_needs_t_end():
    with pytest.raises(ValueError, match="requires t_end"):
        crow_amsaa(WIKI, termination="time")

def test_time_terminated_t_end_before_last_failure_rejected():
    with pytest.raises(ValueError, match="cannot be smaller"):
        crow_amsaa(WIKI, t_end=500, termination="time")

def test_failure_terminated_rejects_inconsistent_t_end():
    with pytest.raises(ValueError, match="last failure time"):
        crow_amsaa(WIKI, t_end=700, termination="failure")

def test_longer_test_without_new_failures_lowers_beta():
    """Rodar mais tempo sem falhar e evidencia de melhora."""
    assert crow_amsaa(WIKI, t_end=900, termination="time").beta < crow_amsaa(WIKI).beta


# ---- extrapolacao ----

def test_time_to_reach_target_mtbf():
    ca = crow_amsaa(WIKI)
    t = ca.time_to_reach_mtbf(60.0)
    assert t > ca.t_end                                    # precisa testar mais
    assert ca.instantaneous_mtbf(t) == pytest.approx(60.0)  # e chega la

def test_time_to_reach_mtbf_impossible_when_deteriorating():
    ca = CrowAMSAA(beta=1.4, lam=0.01, n_failures=10, t_end=100,
                   termination="failure", unbiased=False)
    with pytest.raises(ValueError, match="not improving"):
        ca.time_to_reach_mtbf(500)


# ---- validacao de entrada ----

def test_rejects_non_increasing_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        crow_amsaa([10, 5, 20])

def test_rejects_too_few_failures():
    with pytest.raises(ValueError, match="at least 2"):
        crow_amsaa([10])


# ---- Duane ----

def test_duane_alpha_is_roughly_one_minus_crow_beta():
    d = duane(WIKI)
    ca = crow_amsaa(WIKI)
    assert d.alpha == pytest.approx(1 - ca.beta, abs=0.05)

def test_duane_instantaneous_exceeds_cumulative():
    d = duane(WIKI)
    assert d.instantaneous_mtbf(620) > d.cumulative_mtbf(620)

def test_duane_instantaneous_undefined_when_alpha_ge_one():
    d = duane(WIKI)
    d.alpha = 1.0
    with pytest.raises(ValueError, match="undefined"):
        d.instantaneous_mtbf(620)
