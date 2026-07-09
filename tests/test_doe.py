import numpy as np
import pytest

from relengy.quantitative.doe import (
    code, decode, effects_table, full_factorial, is_orthogonal,
    plackett_burman, pareto_of_effects,
)

def test_full_factorial_is_coded_pm1_and_orthogonal():
    d = full_factorial(["A", "B", "C"])
    assert d.shape == (8, 3)
    assert set(np.unique(d.to_numpy())) == {-1.0, 1.0}
    assert is_orthogonal(d)

def test_effects_recover_known_model():
    """y = 10 + 3A - 2B + 1*AB  =>  efeitos A=6, B=-4, AB=2 (efeito = 2*coef)."""
    d = full_factorial(["A", "B"])
    A, B = d.A.to_numpy(), d.B.to_numpy()
    y = 10 + 3 * A - 2 * B + 1 * (A * B)

    e = effects_table(d, y, max_order=2).set_index("term")
    assert e.loc["A", "effect"] == pytest.approx(6.0)
    assert e.loc["B", "effect"] == pytest.approx(-4.0)
    assert e.loc["A:B", "effect"] == pytest.approx(2.0)
    # coeficiente = efeito / 2, ou seja, os betas originais
    assert e.loc["A", "coefficient"] == pytest.approx(3.0)
    assert e.loc["B", "coefficient"] == pytest.approx(-2.0)

def test_effects_rejects_uncoded_design():
    import pandas as pd
    d = pd.DataFrame({"A": [0, 1, 0, 1], "B": [0, 0, 1, 1]})
    with pytest.raises(ValueError, match="±1"):
        effects_table(d, [1, 2, 3, 4])

def test_effects_rejects_length_mismatch():
    d = full_factorial(["A", "B"])
    with pytest.raises(ValueError, match="resposta tem"):
        effects_table(d, [1, 2, 3])

def test_pareto_orders_by_absolute_effect():
    d = full_factorial(["A", "B"])
    A, B = d.A.to_numpy(), d.B.to_numpy()
    y = 10 + 0.5 * A - 9 * B
    p = pareto_of_effects(effects_table(d, y))
    assert p.iloc[0].term == "B"          # |−18| domina
    assert abs(p.iloc[0].effect) > abs(p.iloc[1].effect)

def test_code_decode_roundtrip():
    # temperatura de 50 a 150 C -> centro 100
    assert code([50, 100, 150], 50, 150).tolist() == [-1.0, 0.0, 1.0]
    np.testing.assert_allclose(decode([-1, 0, 1], 50, 150), [50, 100, 150])

def test_plackett_burman_is_orthogonal_screening():
    d = plackett_burman(4)
    assert set(np.unique(d.to_numpy())) == {-1.0, 1.0}
    assert is_orthogonal(d)
