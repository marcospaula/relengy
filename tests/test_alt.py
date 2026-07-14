import math
import pytest

from relengy.quantitative.alt import (
    BOLTZMANN_EV_PER_K, B_to_activation_energy, activation_energy_to_B,
    arrhenius_af, arrhenius_life, celsius_to_kelvin, coffin_manson_af,
    equivalent_time_table, eyring_af, eyring_life, extrapolate_life,
    fitter_name, hallberg_peck_af, ipl_af, ipl_life, required_test_time,
)

# ---- o mapeamento de nomes, que nao e obvio ----

def test_arrhenius_is_called_exponential_in_the_package():
    assert fitter_name("Arrhenius") == "Fit_Weibull_Exponential"

def test_ipl_is_called_power():
    assert fitter_name("Inverse Power Law") == "Fit_Weibull_Power"
    assert fitter_name("IPL", "Lognormal") == "Fit_Lognormal_Power"

def test_temperature_humidity_is_dual_exponential():
    assert fitter_name("Temperature-Humidity") == "Fit_Weibull_Dual_Exponential"

def test_unknown_model_rejected():
    with pytest.raises(KeyError, match="unknown model"):
        fitter_name("Coffin-Manson")

def test_the_fitters_actually_exist_in_reliability():
    """O mapeamento nao vale nada se os nomes nao existirem de fato."""
    from reliability import ALT_fitters
    for model in ["Arrhenius", "Eyring", "IPL", "Temperature-Humidity",
                  "Temperature-NonThermal", "Dual Power"]:
        name = fitter_name(model)
        assert hasattr(ALT_fitters, name), f"{name} nao existe em ALT_fitters"


# ---- guarda de Kelvin: o erro classico ----

def test_celsius_passed_as_kelvin_warns():
    with pytest.warns(UserWarning, match="Celsius"):
        arrhenius_life(85.0, c=1.0, b=5000.0)      # 85 "graus" -> 85 K = -188 C

def test_kelvin_conversion():
    assert celsius_to_kelvin(0) == pytest.approx(273.15)
    assert celsius_to_kelvin(85) == pytest.approx(358.15)

def test_zero_kelvin_rejected():
    with pytest.raises(ValueError, match="impossible"):
        arrhenius_life(0.0, 1.0, 5000.0)


# ---- energia de ativacao ----

def test_activation_energy_roundtrip():
    ea = 0.7
    assert B_to_activation_energy(activation_energy_to_B(ea)) == pytest.approx(ea)

def test_B_from_typical_activation_energy():
    # Ea = 0.7 eV -> B = 0.7 / 8.617e-5 ~ 8123 K
    assert activation_energy_to_B(0.7) == pytest.approx(0.7 / BOLTZMANN_EV_PER_K)
    assert activation_energy_to_B(0.7) == pytest.approx(8123.0, rel=1e-3)


# ---- fatores de aceleracao ----

def test_arrhenius_af_hand_calculation():
    """Ea=0.7 eV, uso 40 C (313.15 K), ensaio 125 C (398.15 K)."""
    vu, va = celsius_to_kelvin(40), celsius_to_kelvin(125)
    b = activation_energy_to_B(0.7)
    esperado = math.exp(b * (1 / vu - 1 / va))
    assert arrhenius_af(vu, va, ea_ev=0.7) == pytest.approx(esperado)
    assert arrhenius_af(vu, va, b=b) == pytest.approx(esperado)
    assert arrhenius_af(vu, va, ea_ev=0.7) > 100      # aceleracao forte

def test_arrhenius_af_equals_life_ratio():
    """AF = L_uso / L_acelerado, por definicao."""
    vu, va, b, c = 313.15, 398.15, 8000.0, 1.0
    assert arrhenius_af(vu, va, b=b) == pytest.approx(
        arrhenius_life(vu, c, b) / arrhenius_life(va, c, b))

def test_arrhenius_af_needs_exactly_one_of_b_or_ea():
    with pytest.raises(ValueError, match="exactly one"):
        arrhenius_af(313.15, 398.15)
    with pytest.raises(ValueError, match="exactly one"):
        arrhenius_af(313.15, 398.15, b=8000.0, ea_ev=0.7)

def test_arrhenius_af_rejects_non_accelerating_stress():
    with pytest.raises(ValueError, match="must be HIGHER"):
        arrhenius_af(398.15, 313.15, ea_ev=0.7)

def test_ipl_af_hand_calculation():
    """AF = (V_acc/V_uso)^n = (20/10)^3 = 8"""
    assert ipl_af(10.0, 20.0, n=3.0) == pytest.approx(8.0)

def test_ipl_af_independent_of_K():
    assert ipl_af(10, 20, 3) == pytest.approx(
        ipl_life(10, k=0.5, n=3) / ipl_life(20, k=0.5, n=3))
    assert ipl_af(10, 20, 3) == pytest.approx(
        ipl_life(10, k=99.0, n=3) / ipl_life(20, k=99.0, n=3))

def test_eyring_af_equals_life_ratio():
    vu, va, b = 313.15, 398.15, 8000.0
    assert eyring_af(vu, va, b) == pytest.approx(
        eyring_life(vu, a=1.0, b=b) / eyring_life(va, a=1.0, b=b))

def test_eyring_accelerates_more_than_arrhenius_for_same_B():
    """O fator linear V_acc/V_uso > 1 amplifica o exponencial."""
    vu, va, b = 313.15, 398.15, 8000.0
    assert eyring_af(vu, va, b) > arrhenius_af(vu, va, b=b)


# ---- uso pratico ----

def test_extrapolate_life():
    assert extrapolate_life(life_accelerated=100.0, af=50.0) == pytest.approx(5000.0)

def test_required_test_time_is_the_inverse_of_extrapolate():
    assert required_test_time(5000.0, af=50.0) == pytest.approx(100.0)

def test_af_below_one_rejected():
    with pytest.raises(ValueError, match="accelerated nothing"):
        extrapolate_life(100.0, af=0.5)


# ---- Coffin-Manson (fadiga termomecanica) ----

def test_coffin_manson_hand_calculation():
    """(ΔT_acc/ΔT_uso)^m = (205/76)^4 ~ 52.9. Bate com 54750/1034 = 52.95 cls."""
    assert coffin_manson_af(76.0, 205.0, m=4.0) == pytest.approx((205 / 76) ** 4)
    assert coffin_manson_af(76.0, 205.0, m=4.0) == pytest.approx(52.95, rel=1e-3)

def test_coffin_manson_rejects_non_accelerating_amplitude():
    with pytest.raises(ValueError, match="must be LARGER"):
        coffin_manson_af(205.0, 76.0, m=4.0)

def test_coffin_manson_rejects_bad_inputs():
    with pytest.raises(ValueError, match="positive"):
        coffin_manson_af(-1.0, 205.0, m=4.0)
    with pytest.raises(ValueError, match="exponent m"):
        coffin_manson_af(76.0, 205.0, m=0.0)


# ---- Hallberg-Peck (temperatura + umidade) ----

def test_hallberg_peck_factorizes_into_humidity_times_arrhenius():
    vu, va = 305.15, 358.15
    rh_factor = (0.9 / 0.5) ** 3.0
    assert hallberg_peck_af(vu, va, 0.5, 0.9, p=3.0, ea_ev=0.8) == pytest.approx(
        rh_factor * arrhenius_af(vu, va, ea_ev=0.8))

def test_hallberg_peck_percent_or_fraction_same_ratio():
    """RH so aparece como razao: 0.5/0.9 == 50/90."""
    vu, va = 305.15, 358.15
    assert hallberg_peck_af(vu, va, 0.5, 0.9, p=3.0, ea_ev=0.8) == pytest.approx(
        hallberg_peck_af(vu, va, 50.0, 90.0, p=3.0, ea_ev=0.8))

def test_hallberg_peck_humidity_only_accelerates_more():
    vu, va = 305.15, 358.15
    assert hallberg_peck_af(vu, va, 0.5, 0.9, p=3.0, ea_ev=0.8) > arrhenius_af(
        vu, va, ea_ev=0.8)

def test_hallberg_peck_rejects_non_accelerating_humidity():
    with pytest.raises(ValueError, match="must be HIGHER"):
        hallberg_peck_af(305.15, 358.15, 0.9, 0.5, p=3.0, ea_ev=0.8)


# ---- tabela de tempo equivalente ----

def test_equivalent_time_table_matches_deck_htol_and_coffin_manson():
    """Reproduz duas linhas do estudo: HTOL (Arrhenius) e termomecanico (CM)."""
    tab = equivalent_time_table([
        {"mecanismo": "operacao (HTOL)", "model": "arrhenius", "ea_ev": 0.7,
         "v_use": celsius_to_kelvin(87), "v_accelerated": celsius_to_kelvin(125),
         "target_life_use": 12000.0},
        {"mecanismo": "termomecanico (TC)", "model": "coffin_manson", "m": 4.0,
         "dt_use": 76.0, "dt_accelerated": 205.0,
         "target_life_use": 54750.0, "unit": "ciclos"},
    ])
    assert tab[0]["af"] == pytest.approx(8.6, rel=1e-2)
    assert tab[0]["tempo_ensaio_equivalente"] == pytest.approx(1393.0, rel=1e-2)
    assert tab[1]["af"] == pytest.approx(52.95, rel=1e-3)
    assert tab[1]["tempo_ensaio_equivalente"] == pytest.approx(1034.0, rel=1e-2)
    assert tab[1]["unit"] == "ciclos"

def test_equivalent_time_table_rejects_unknown_model():
    with pytest.raises(KeyError, match="unknown model"):
        equivalent_time_table([{"model": "weibull", "target_life_use": 1.0}])
