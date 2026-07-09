import math
import pytest

from relengy.quantitative.rbd import (
    Component, KooN, Parallel, Series, birnbaum_importance,
    criticality_importance, mission_reliability, parallel_of, series_of,
    weibull_reliability,
)

A, B, C = Component("A"), Component("B"), Component("C")
R = {"A": 0.9, "B": 0.8, "C": 0.7}


# ---- identidades fundamentais ----

def test_series_is_the_product():
    assert Series(A, B, C).reliability(R) == pytest.approx(0.9 * 0.8 * 0.7)

def test_parallel_is_one_minus_product_of_unreliabilities():
    assert Parallel(A, B, C).reliability(R) == pytest.approx(1 - 0.1 * 0.2 * 0.3)

def test_series_is_never_better_than_its_worst_component():
    assert Series(A, B, C).reliability(R) <= min(R.values())

def test_parallel_is_never_worse_than_its_best_component():
    assert Parallel(A, B, C).reliability(R) >= max(R.values())

def test_koon_with_k_equals_1_is_parallel():
    assert KooN(1, A, B, C).reliability(R) == pytest.approx(Parallel(A, B, C).reliability(R))

def test_koon_with_k_equals_n_is_series():
    assert KooN(3, A, B, C).reliability(R) == pytest.approx(Series(A, B, C).reliability(R))

def test_2oo3_identical_matches_closed_form():
    """R = 3R^2 - 2R^3 para tres componentes identicos."""
    p = 0.9
    r = {"A": p, "B": p, "C": p}
    assert KooN(2, A, B, C).reliability(r) == pytest.approx(3 * p**2 - 2 * p**3)

def test_koon_distinct_reliabilities_by_enumeration():
    """2-de-3 com R distintos: soma das probabilidades de >= 2 funcionando."""
    ra, rb, rc = 0.9, 0.8, 0.7
    exato = (ra*rb*rc                      # 3 ok
             + ra*rb*(1-rc) + ra*(1-rb)*rc + (1-ra)*rb*rc)   # exatamente 2
    assert KooN(2, A, B, C).reliability(R) == pytest.approx(exato)

def test_nested_structure():
    """A em serie com (B paralelo C)."""
    sys = Series(A, Parallel(B, C))
    assert sys.reliability(R) == pytest.approx(0.9 * (1 - 0.2 * 0.3))


# ---- importancia de Birnbaum ----

def test_birnbaum_of_series_is_product_of_the_others():
    ib = birnbaum_importance(Series(A, B, C), R)
    assert ib["A"] == pytest.approx(0.8 * 0.7)
    assert ib["B"] == pytest.approx(0.9 * 0.7)
    assert ib["C"] == pytest.approx(0.9 * 0.8)

def test_birnbaum_of_parallel_is_product_of_others_unreliabilities():
    ib = birnbaum_importance(Parallel(A, B, C), R)
    assert ib["A"] == pytest.approx(0.2 * 0.3)
    assert ib["C"] == pytest.approx(0.1 * 0.2)

def test_in_series_the_worst_component_is_the_most_important():
    ib = birnbaum_importance(Series(A, B, C), R)
    assert max(ib, key=ib.get) == "C"        # C tem o menor R

def test_in_parallel_the_best_component_is_the_most_important():
    ib = birnbaum_importance(Parallel(A, B, C), R)
    assert max(ib, key=ib.get) == "A"        # A tem o maior R

def test_birnbaum_does_not_depend_on_own_reliability():
    """Muda R_A: a importancia de Birnbaum de A nao muda."""
    sys = Series(A, B, C)
    i1 = birnbaum_importance(sys, {**R, "A": 0.99})["A"]
    i2 = birnbaum_importance(sys, {**R, "A": 0.10})["A"]
    assert i1 == pytest.approx(i2)

def test_birnbaum_matches_numerical_derivative():
    """Validacao independente: derivada numerica de R_sys em relacao a R_A."""
    sys = Series(A, Parallel(B, C))
    h = 1e-6
    hi = sys.reliability({**R, "A": R["A"] + h})
    lo = sys.reliability({**R, "A": R["A"] - h})
    assert birnbaum_importance(sys, R)["A"] == pytest.approx((hi - lo) / (2 * h), abs=1e-6)


# ---- importancia de criticidade ----

def test_criticality_weights_birnbaum_by_current_unreliability():
    sys = Series(A, B, C)
    ib = birnbaum_importance(sys, R)
    ic = criticality_importance(sys, R)
    r_sys = sys.reliability(R)
    assert ic["B"] == pytest.approx(ib["B"] * (1 - R["B"]) / (1 - r_sys))

def test_criticality_can_reorder_priorities_versus_birnbaum():
    """Birnbaum aponta potencial; criticidade aponta onde o risco esta agora."""
    sys = Parallel(A, B, C)
    ib = birnbaum_importance(sys, R)
    ic = criticality_importance(sys, R)
    assert max(ib, key=ib.get) == "A"     # o mais confiavel tem maior potencial
    assert max(ic, key=ic.get) == "C"     # mas o risco real esta no pior

def test_criticality_undefined_for_perfect_system():
    with pytest.raises(ValueError, match="R_sys = 1"):
        criticality_importance(Series(A, B), {"A": 1.0, "B": 1.0})


# ---- validacao ----

def test_missing_component_reliability():
    with pytest.raises(KeyError, match="no reliability"):
        Series(A, B).reliability({"A": 0.9})

def test_reliability_outside_unit_interval_rejected():
    with pytest.raises(ValueError, match="outside"):
        birnbaum_importance(Series(A), {"A": 1.5})

def test_koon_k_out_of_range():
    with pytest.raises(ValueError, match="outside"):
        KooN(4, A, B, C)

def test_empty_series_rejected():
    with pytest.raises(ValueError, match="at least one"):
        Series()


# ---- ponte com Weibull ----

def test_weibull_reliability_at_eta_is_exp_minus_one():
    assert weibull_reliability(100, beta=2.0, eta=100) == pytest.approx(math.exp(-1))
    assert weibull_reliability(100, beta=0.5, eta=100) == pytest.approx(math.exp(-1))

def test_mission_reliability_from_weibull_params():
    sys = series_of(["A", "B"])
    params = {"A": (2.0, 1000.0), "B": (1.5, 2000.0)}
    esperado = weibull_reliability(500, 2.0, 1000) * weibull_reliability(500, 1.5, 2000)
    assert mission_reliability(sys, 500, params) == pytest.approx(esperado)

def test_helpers_build_flat_structures():
    assert series_of(["A", "B"]).reliability(R) == pytest.approx(0.72)
    assert parallel_of(["A", "B"]).reliability(R) == pytest.approx(0.98)
