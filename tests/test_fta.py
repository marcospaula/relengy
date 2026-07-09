import pytest

from relengy.qualitative.fta import FaultTree, weibull_cdf


def simple_tree() -> FaultTree:
    """TOP = A OR (B AND C).  Minimal cut sets: {A}, {B,C}."""
    return FaultTree(
        top="TOP",
        gates={"TOP": ("OR", ["A", "G1"]), "G1": ("AND", ["B", "C"])},
    )


def test_basic_events():
    assert simple_tree().basic_events() == {"A", "B", "C"}


def test_minimal_cut_sets():
    mcs = simple_tree().minimal_cut_sets()
    assert sorted(map(sorted, mcs)) == [["A"], ["B", "C"]]


def test_absorption_removes_supersets():
    """TOP = A OR (A AND B).  {A,B} é superset de {A}: some na minimalização."""
    ft = FaultTree(top="TOP", gates={"TOP": ("OR", ["A", "G1"]), "G1": ("AND", ["A", "B"])})
    assert ft.minimal_cut_sets() == [frozenset({"A"})]


def test_top_probability_exact_matches_hand_calculation():
    """P = P(A) + P(B)P(C) - P(A)P(B)P(C) = 0.1 + 0.06 - 0.006 = 0.154"""
    probs = {"A": 0.1, "B": 0.2, "C": 0.3}
    assert simple_tree().top_probability(probs) == pytest.approx(0.154)


def test_rare_event_approximation_overestimates():
    probs = {"A": 0.1, "B": 0.2, "C": 0.3}
    ft = simple_tree()
    exact = ft.top_probability(probs, exact=True)
    rare = ft.top_probability(probs, exact=False)
    assert rare == pytest.approx(0.16)
    assert rare > exact  # a aproximação é sempre um limite superior


def test_pure_and_gate():
    ft = FaultTree(top="TOP", gates={"TOP": ("AND", ["A", "B"])})
    assert ft.minimal_cut_sets() == [frozenset({"A", "B"})]
    assert ft.top_probability({"A": 0.5, "B": 0.4}) == pytest.approx(0.2)


def test_pure_or_gate():
    ft = FaultTree(top="TOP", gates={"TOP": ("OR", ["A", "B"])})
    # 1 - (1-0.5)(1-0.4) = 0.7
    assert ft.top_probability({"A": 0.5, "B": 0.4}) == pytest.approx(0.7)


def test_missing_probability_is_an_error():
    with pytest.raises(ValueError, match="sem probabilidade"):
        simple_tree().top_probability({"A": 0.1, "B": 0.2})


def test_weibull_cdf_feeds_basic_events():
    # Em t = eta, F = 1 - e^-1 = 0.6321, independente de beta.
    assert weibull_cdf(100, beta=2.0, eta=100) == pytest.approx(0.632120, abs=1e-5)
    assert weibull_cdf(100, beta=0.5, eta=100) == pytest.approx(0.632120, abs=1e-5)
