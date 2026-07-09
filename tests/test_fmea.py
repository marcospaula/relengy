import pandas as pd
import pytest

from relengy.qualitative.fmea import (
    ap_table_template, failure_mode_criticality, item_criticality,
    score_ap, score_rpn, validate, validate_ap_table,
)

def base_df():
    return pd.DataFrame({
        "item": ["bomba", "bomba", "valvula"],
        "failure_mode": ["vazamento", "travamento", "nao abre"],
        "effect": ["perda de fluido", "parada", "sem fluxo"],
        "cause": ["selo", "rolamento", "mola"],
        "control": ["inspecao", "vibracao", "teste"],
        "severity": [7, 9, 4],
        "occurrence": [3, 2, 5],
        "detection": [5, 8, 3],
    })

def test_validate_rejects_out_of_scale():
    df = base_df()
    df.loc[0, "severity"] = 11
    with pytest.raises(ValueError, match="outside the"):
        validate(df)

def test_validate_reports_all_problems_at_once():
    df = base_df().drop(columns=["cause"])
    df.loc[0, "detection"] = 99
    with pytest.raises(ValueError) as e:
        validate(df)
    assert "missing columns" in str(e.value) and "outside the" in str(e.value)

def test_rpn_and_severity_override():
    out = score_rpn(base_df())
    r = out.set_index("failure_mode")
    assert r.loc["vazamento", "rpn"] == 7 * 3 * 5   # 105 -> acima do limiar
    assert r.loc["travamento", "rpn"] == 9 * 2 * 8  # 144
    # travamento tem S=9: severidade dispara acao mesmo se RPN fosse baixo
    assert r.loc["travamento", "driver"] == "RPN + severidade"
    # valvula: RPN = 60, S = 4 -> sem acao
    assert not r.loc["nao abre", "action_required"]

def test_ap_template_is_complete_and_empty():
    t = ap_table_template()
    assert len(t) == 1000
    assert (t.ap == "").all()

def test_validate_ap_table_catches_incomplete():
    with pytest.raises(ValueError, match="incomplete"):
        validate_ap_table({(1, 1, 1): "L"})

def test_validate_ap_table_catches_non_monotonic():
    # tabela toda "L" exceto um ponto onde subir D reduz a prioridade
    table = {k: "H" for k in __import__("itertools").product(range(1, 11), repeat=3)}
    table[(5, 5, 6)] = "L"   # subir D de 5 para 6 derruba H -> L
    with pytest.raises(ValueError, match="not monotonic"):
        validate_ap_table(table)

def test_score_ap_uses_lookup_not_formula():
    table = {k: "L" for k in __import__("itertools").product(range(1, 11), repeat=3)}
    table[(9, 2, 8)] = "H"
    out = score_ap(base_df(), table)
    assert out.iloc[0].failure_mode == "travamento"   # o unico H vem primeiro
    assert out.iloc[0].ap == "H"

def test_mil_std_cm_formula():
    # Cm = beta * alpha * lambda_p * t
    assert failure_mode_criticality(0.5, 0.4, 1e-5, 1000) == pytest.approx(2e-3)

def test_mil_std_cm_rejects_bad_beta():
    with pytest.raises(ValueError, match="beta"):
        failure_mode_criticality(1.5, 0.4, 1e-5, 1000)

def test_item_criticality_sums_within_severity_category():
    df = pd.DataFrame({
        "item": ["bomba", "bomba"],
        "severity_category": ["II", "II"],
        "beta": [1.0, 0.5],
        "alpha": [0.7, 0.3],
        "lambda_p": [1e-5, 1e-5],
        "t": [100, 100],
    })
    out = item_criticality(df)
    # Cr = 1.0*0.7*1e-5*100 + 0.5*0.3*1e-5*100 = 7e-4 + 1.5e-4 = 8.5e-4
    assert out.iloc[0].cr == pytest.approx(8.5e-4)

def test_item_criticality_rejects_alpha_not_summing_to_one():
    df = pd.DataFrame({
        "item": ["bomba", "bomba"], "severity_category": ["II", "II"],
        "beta": [1.0, 1.0], "alpha": [0.7, 0.7], "lambda_p": [1e-5, 1e-5], "t": [100, 100],
    })
    with pytest.raises(ValueError, match="do not sum to 1"):
        item_criticality(df)

def test_item_criticality_rejects_bad_category():
    df = pd.DataFrame({
        "item": ["b"], "severity_category": ["V"],
        "beta": [1.0], "alpha": [1.0], "lambda_p": [1e-5], "t": [100],
    })
    with pytest.raises(ValueError, match="invalid categories"):
        item_criticality(df)
