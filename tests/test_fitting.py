"""A escolha de metodo reconcilia Abernethy (MRR sempre) com ReliaSoft
(MLE sob censura pesada/desigual ou dados intervalares)."""
import pytest

from relengy.quantitative.fitting import (
    HEAVY_CENSORING_FRACTION, SMALL_SAMPLE_FAILURES, fit_diagnostic, recommend_method,
)

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
