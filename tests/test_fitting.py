"""A escolha de metodo reconcilia Abernethy (MRR sempre) com ReliaSoft
(MLE sob censura pesada/desigual ou dados intervalares)."""
import pytest

from relengy.quantitative.fitting import (
    HEAVY_CENSORING_FRACTION, RANDOM_FAILURE_BAND, SMALL_SAMPLE_FAILURES,
    WeibullDiagnosis, fit_diagnostic, recommend_method,
)


def _diag(beta_mrr: float) -> WeibullDiagnosis:
    """Diagnostico minimo so para exercitar regime() com um beta controlado."""
    return WeibullDiagnosis(beta_mrr=beta_mrr, eta_mrr=100.0, beta_mle=beta_mrr,
                            eta_mle=100.0, n_failures=10, n_censored=0)


def _diag_ratio(ratio: float) -> WeibullDiagnosis:
    """Diagnostico com razao beta_mle/beta_mrr controlada, para o sinal de batch."""
    return WeibullDiagnosis(beta_mrr=2.0, eta_mrr=100.0, beta_mle=2.0 * ratio,
                            eta_mle=100.0, n_failures=10, n_censored=0)

def test_small_sample_light_censoring_uses_rrx():
    """O caso em que as duas fontes concordam: MRR."""
    r = recommend_method(n_failures=8, n_censored=1)
    assert r.method == "RRX"
    assert "agree" in r.reason

def test_heavy_censoring_flips_to_mle():
    """Onde Abernethy ('MLE not recommended') e ReliaSoft divergem: ReliaSoft ganha."""
    r = recommend_method(n_failures=3, n_censored=17)   # 85% censurado
    assert r.method == "MLE"
    assert "heavy censoring" in r.reason

def test_heavy_censoring_with_small_sample_warns_about_mle_bias():
    r = recommend_method(n_failures=3, n_censored=17)
    assert r.caution is not None
    assert "biased" in r.caution
    assert "predictr" in r.caution or "Weibayes" in r.caution

def test_uneven_censoring_flips_to_mle_even_when_light():
    """Censura desigual e julgamento do analista, nao se infere das contagens."""
    r = recommend_method(n_failures=10, n_censored=1, uneven_censoring=True)
    assert r.method == "MLE"
    assert "uneven censoring" in r.reason

def test_interval_data_forces_mle():
    r = recommend_method(n_failures=10, n_censored=0, interval_data=True)
    assert r.method == "MLE"
    assert "interval data" in r.reason

def test_zero_failures_cannot_use_rank_regression():
    r = recommend_method(n_failures=0, n_censored=25)
    assert r.method == "MLE"
    assert "Weibayes" in (r.caution or "")

def test_threshold_boundary_is_inclusive():
    """Exatamente 50% de censura ja conta como pesada."""
    assert HEAVY_CENSORING_FRACTION == 0.5
    assert recommend_method(10, 10).method == "MLE"     # 50% -> pesada
    assert recommend_method(10, 9).method == "RRX"      # 47% -> leve

def test_small_sample_boundary():
    assert SMALL_SAMPLE_FAILURES == 20
    assert "small sample" in recommend_method(20, 0).reason
    assert "small sample" not in recommend_method(21, 0).reason

def test_diagnosis_exposes_censoring_fraction_and_recommendation():
    d = fit_diagnostic([30, 49, 82, 90, 96], [10, 45, 100])
    assert d.censoring_fraction == pytest.approx(3 / 8)
    assert d.recommendation().method == "RRX"          # 37.5% -> leve
    assert "RECOMMENDED METHOD" in d.report()

def test_report_changes_when_interval_data_declared():
    d = fit_diagnostic([30, 49, 82, 90, 96], [10, 45, 100])
    assert "RRX" in d.report()
    assert "MLE" in d.report(interval_data=True).split("RECOMMENDED METHOD")[1]

@pytest.mark.parametrize("failures", [[], [50]])
def test_fit_diagnostic_needs_two_failures_and_points_to_weibayes(failures):
    """0 ou 1 falha: erro de dominio (Weibayes), nao erro opaco da dependencia."""
    with pytest.raises(ValueError, match="at least 2 failures"):
        fit_diagnostic(failures, right_censored=[100, 100])
    with pytest.raises(ValueError, match="Weibayes"):
        fit_diagnostic(failures)


def test_regime_uses_a_band_around_one_not_exact_equality():
    """beta de um ajuste real nunca cai em 1.0 exato: o ramo 'random' e uma banda."""
    assert RANDOM_FAILURE_BAND == 0.05
    assert "infant mortality" in _diag(0.90).regime()      # abaixo da banda
    assert "random failures" in _diag(0.95).regime()       # borda inferior
    assert "random failures" in _diag(1.00).regime()
    assert "random failures" in _diag(1.05).regime()       # borda superior (inclusiva)
    assert "early wear-out" in _diag(1.10).regime()        # acima da banda
    assert "rapid old-age" in _diag(5.0).regime()


def test_batch_margin_exposes_signed_distance_to_the_cutoff():
    """O booleano esconde o grau; batch_margin mostra o quao longe do corte."""
    blatant = _diag_ratio(0.20)
    marginal = _diag_ratio(0.74)
    clean = _diag_ratio(0.90)

    assert blatant.batch_suspected and marginal.batch_suspected
    assert not clean.batch_suspected

    # positivo quando suspeito, e maior = mais forte
    assert blatant.batch_margin == pytest.approx(0.55)
    assert marginal.batch_margin == pytest.approx(0.01)
    assert blatant.batch_margin > marginal.batch_margin
    # negativo quando ha folga
    assert clean.batch_margin == pytest.approx(-0.15)


def test_report_shows_how_far_past_the_batch_line_not_just_the_flag():
    """0.74 e 0.20 disparam o mesmo flag, mas o report os distingue pela distancia."""
    marginal = _diag_ratio(0.74).report()
    blatant = _diag_ratio(0.20).report()

    assert "0.010" in marginal and "0.75" in marginal     # limitrofe: 0.01 do corte
    assert "0.550" in blatant                              # gritante: 0.55 do corte
    assert marginal != blatant                             # nao soam iguais

    # sem suspeita, a linha da razao ainda diz o lado (folga acima do corte)
    assert "above the 0.75 batch line" in _diag_ratio(0.90).report()
