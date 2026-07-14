"""Ensaios acelerados de vida (ALT): relações vida-estresse e fator de aceleração.

Equações verificadas contra o ReliaWiki arquivado em `library/reliawiki/alt/`.

    Arrhenius  L(V) = C · e^(B/V)            V em KELVIN, estresse térmico
    Eyring     L(V) = (1/V) · e^(−(A − B/V)) V em KELVIN, estresse térmico
    IPL        L(V) = 1 / (K · V^n)          estresse não-térmico (fadiga, tensão)

Fator de aceleração — quanto o ensaio "adianta" a vida:

    AF = L_uso / L_acelerado

    Arrhenius: AF = e^(B·(1/V_uso − 1/V_acc))
    IPL:       AF = (V_acc / V_uso)^n

No Arrhenius, B = Ea/k, com Ea a energia de ativação (eV) e k a constante de
Boltzmann. Ea é propriedade do *mecanismo* de falha, não do componente: dois
componentes distintos que degradam pelo mesmo mecanismo compartilham Ea, e o
mesmo componente sob dois mecanismos não tem um Ea só.

Além das relações vida-estresse contínuas acima (que casam com os fitters do
pacote `reliability`), há dois modelos de AF **por mecanismo de dano** que se
calculam direto, sem ajustar curva — cada um vale para um mecanismo específico:

    Coffin-Manson  AF = (ΔT_acc / ΔT_uso)^m     fadiga termomecânica (CICLOS)
    Hallberg-Peck  AF = (RH_acc/RH_uso)^p · e^(B·(1/T_uso − 1/T_acc))   corrosão (umidade)

O AF só é físico quando o **mecanismo** do modelo é o mecanismo real. Pedir "a
energia de ativação" ou empilhar umidade sem esse mecanismo ocorrer em campo são
armadilhas clássicas de ALT (ver a review-list dos 7 pitfalls no projeto de
análises). `equivalent_time_table` consolida vários mecanismos numa tabela de
tempo de ensaio equivalente por condição de carga.

---

**Os nomes do pacote `reliability` não são os nomes da literatura.** Isto custa
horas a quem procura `Fit_Weibull_Arrhenius` e não acha. Ver `MODEL_MAP`.
"""

from __future__ import annotations

import math
import warnings

# CODATA 2018. O ReliaWiki cita 8.6173303e-5 (CODATA 2014); a diferença é
# irrelevante para engenharia, mas registramos a fonte.
BOLTZMANN_EV_PER_K = 8.617333262e-5

# Abaixo disto, quase certamente alguém passou Celsius achando que era Kelvin.
# 200 K = −73 °C: nenhum ALT térmico usual opera abaixo disso.
_SUSPICIOUS_KELVIN = 200.0


# ---------------------------------------------------------------------------
# Nomes: ReliaWiki / literatura  ->  pacote `reliability`
# ---------------------------------------------------------------------------

MODEL_MAP: dict[str, str] = {
    "Arrhenius": "Exponential",             # Fit_Weibull_Exponential
    "Eyring": "Eyring",                     # Fit_Weibull_Eyring
    "Inverse Power Law": "Power",           # Fit_Weibull_Power
    "IPL": "Power",
    "Temperature-Humidity": "Dual_Exponential",
    "Temperature-NonThermal": "Power_Exponential",
    "Dual Power": "Dual_Power",             # dois estresses não-térmicos
}

STRESS_KIND: dict[str, str] = {
    "Arrhenius": "1 thermal (Kelvin)",
    "Eyring": "1 thermal (Kelvin)",
    "Inverse Power Law": "1 non-thermal",
    "Temperature-Humidity": "2 thermal",
    "Temperature-NonThermal": "thermal (stress_1) + non-thermal (stress_2)",
    "Dual Power": "2 non-thermal",
}


def fitter_name(model: str, distribution: str = "Weibull") -> str:
    """Nome da função no pacote `reliability` para um modelo da literatura.

    >>> fitter_name("Arrhenius")
    'Fit_Weibull_Exponential'
    >>> fitter_name("IPL", "Lognormal")
    'Fit_Lognormal_Power'
    """
    if model not in MODEL_MAP:
        raise KeyError(f"unknown model: {model!r}. Known: {sorted(MODEL_MAP)}")
    if distribution not in ("Weibull", "Lognormal", "Normal", "Exponential"):
        raise ValueError(f"unsupported distribution: {distribution!r}")
    return f"Fit_{distribution}_{MODEL_MAP[model]}"


# ---------------------------------------------------------------------------
# Temperatura
# ---------------------------------------------------------------------------

def celsius_to_kelvin(c: float) -> float:
    return c + 273.15


def _check_kelvin(*temps: float) -> None:
    for t in temps:
        if t <= 0:
            raise ValueError(f"temperature {t} <= 0 K is impossible")
        if t < _SUSPICIOUS_KELVIN:
            warnings.warn(
                f"temperature {t} K = {t - 273.15:.1f} C. That looks like Celsius "
                "passed as Kelvin. Arrhenius and Eyring require KELVIN — use "
                "celsius_to_kelvin(). The error is silent and the AF comes out "
                "wrong by orders of magnitude.",
                UserWarning,
                stacklevel=3,
            )


def activation_energy_to_B(ea_ev: float) -> float:
    """B = Ea / k. Ea em elétron-volts."""
    if ea_ev <= 0:
        raise ValueError("activation energy must be positive")
    return ea_ev / BOLTZMANN_EV_PER_K


def B_to_activation_energy(b: float) -> float:
    """Inverso: Ea = B · k. Útil para sanidade — Ea típico fica entre 0.3 e 1.5 eV."""
    return b * BOLTZMANN_EV_PER_K


# ---------------------------------------------------------------------------
# Relações vida-estresse
# ---------------------------------------------------------------------------

def arrhenius_life(v: float, c: float, b: float) -> float:
    """L(V) = C·e^(B/V), V em Kelvin."""
    _check_kelvin(v)
    return c * math.exp(b / v)


def eyring_life(v: float, a: float, b: float) -> float:
    """L(V) = (1/V)·e^(−(A − B/V)), V em Kelvin."""
    _check_kelvin(v)
    return (1.0 / v) * math.exp(-(a - b / v))


def ipl_life(v: float, k: float, n: float) -> float:
    """L(V) = 1/(K·V^n). V é estresse não-térmico (tensão, carga, pressão)."""
    if v <= 0 or k <= 0:
        raise ValueError("V and K must be positive")
    return 1.0 / (k * v**n)


# ---------------------------------------------------------------------------
# Fatores de aceleração
# ---------------------------------------------------------------------------

def arrhenius_af(v_use: float, v_accelerated: float, *,
                 b: float | None = None, ea_ev: float | None = None) -> float:
    """AF = e^(B·(1/V_uso − 1/V_acc)). Informe B **ou** a energia de ativação."""
    if (b is None) == (ea_ev is None):
        raise ValueError("give exactly one of: b or ea_ev")
    _check_kelvin(v_use, v_accelerated)
    if v_accelerated <= v_use:
        raise ValueError(
            f"the accelerated stress ({v_accelerated} K) must be HIGHER than the "
            f"use stress ({v_use} K); otherwise there is no acceleration"
        )
    b = activation_energy_to_B(ea_ev) if b is None else b
    return math.exp(b * (1.0 / v_use - 1.0 / v_accelerated))


def ipl_af(v_use: float, v_accelerated: float, n: float) -> float:
    """AF = (V_acc / V_uso)^n. Não depende de K."""
    if v_use <= 0 or v_accelerated <= 0:
        raise ValueError("stresses must be positive")
    if v_accelerated <= v_use:
        raise ValueError("the accelerated stress must be higher than the use stress")
    if n <= 0:
        raise ValueError("n must be positive (life falls as stress rises)")
    return (v_accelerated / v_use) ** n


def eyring_af(v_use: float, v_accelerated: float, b: float) -> float:
    """AF = (V_acc/V_uso) · e^(B·(1/V_uso − 1/V_acc)).

    Difere do Arrhenius pelo fator linear V_acc/V_uso, que é > 1. Logo, para o
    mesmo B, o Eyring acelera **mais** que o Arrhenius. Isso importa ao comparar
    os dois modelos ajustados aos mesmos dados: um B menor no Eyring pode
    representar a mesma física.
    """
    _check_kelvin(v_use, v_accelerated)
    if v_accelerated <= v_use:
        raise ValueError("the accelerated stress must be higher than the use stress")
    return (v_accelerated / v_use) * math.exp(b * (1.0 / v_use - 1.0 / v_accelerated))


def coffin_manson_af(dt_use: float, dt_accelerated: float, m: float) -> float:
    """AF = (ΔT_acc / ΔT_uso)^m — fadiga termomecânica (ciclagem térmica).

    O AF conta em **ciclos**, não em horas: quantos ciclos de campo um ciclo de
    ensaio "vale". ΔT é a *amplitude* do ciclo térmico, não a temperatura.

    O expoente m é do mecanismo de trinca: ~1 para materiais dúcteis, ~2 para
    soldas Sn-Pb, ~4 para trincas em ligas duras, até ~9 para frágeis. Chutar m é
    chutar o AF — como no Arrhenius com Ea.

    Este é o Coffin-Manson "puro". A extensão de Norris-Landzberg acrescenta um
    termo de frequência de ciclagem e um Arrhenius na temperatura máxima do ciclo;
    quando a frequência de ensaio e de uso diferem muito, é ela que se deve usar.

    >>> round(coffin_manson_af(76.0, 205.0, m=4.0), 1)   # ΔT_uso=76, ΔT_acc=205 C
    52.9
    """
    if dt_use <= 0 or dt_accelerated <= 0:
        raise ValueError("thermal-cycle amplitudes ΔT must be positive")
    if dt_accelerated <= dt_use:
        raise ValueError(
            f"the accelerated ΔT ({dt_accelerated}) must be LARGER than the use ΔT "
            f"({dt_use}); otherwise there is no acceleration"
        )
    if m <= 0:
        raise ValueError("the Coffin-Manson exponent m must be positive")
    return (dt_accelerated / dt_use) ** m


def hallberg_peck_af(v_use: float, v_accelerated: float,
                     rh_use: float, rh_accelerated: float, *, p: float,
                     b: float | None = None, ea_ev: float | None = None) -> float:
    """AF = (RH_acc/RH_uso)^p · e^(B·(1/V_uso − 1/V_acc)) — corrosão por umidade.

    É o termo de umidade de Peck multiplicando o Arrhenius de temperatura (V em
    KELVIN). Informe B **ou** a energia de ativação, como em `arrhenius_af`. O
    expoente de Peck p fica tipicamente ~2.7–3 (ex.: corrosão de bond pad).

    RH pode entrar como fração (0.85) ou percentual (85): como só aparece a razão
    RH_acc/RH_uso, a unidade cancela — desde que as duas venham na mesma unidade.

    Só use este modelo (dois estressores) se o mecanismo real **depende de
    umidade**. Somar umidade "por garantia" acelera por um caminho que pode não
    ocorrer em campo (pitfall clássico).

    >>> # fatoriza: parte de umidade × parte de Arrhenius
    >>> af = hallberg_peck_af(305.15, 358.15, 0.5, 0.9, p=3.0, ea_ev=0.8)
    >>> af > arrhenius_af(305.15, 358.15, ea_ev=0.8)   # umidade acelera ainda mais
    True
    """
    if rh_use <= 0 or rh_accelerated <= 0:
        raise ValueError("relative humidities must be positive")
    if rh_accelerated <= rh_use:
        raise ValueError(
            f"the accelerated RH ({rh_accelerated}) must be HIGHER than the use RH "
            f"({rh_use}); otherwise the humidity term does not accelerate"
        )
    if p <= 0:
        raise ValueError("the Peck exponent p must be positive")
    humidity = (rh_accelerated / rh_use) ** p
    return humidity * arrhenius_af(v_use, v_accelerated, b=b, ea_ev=ea_ev)


def extrapolate_life(life_accelerated: float, af: float) -> float:
    """Vida na condição de uso = vida acelerada × AF."""
    if af < 1.0:
        raise ValueError(f"AF = {af} < 1: the test accelerated nothing")
    return life_accelerated * af


def required_test_time(target_life_use: float, af: float) -> float:
    """Quanto tempo de ensaio acelerado equivale a uma vida-alvo em uso.

    Nome deliberadamente NÃO iniciado por `test_`: o pytest coleta qualquer
    função com esse prefixo, mesmo em módulos de produção importados por testes.
    """
    if af < 1.0:
        raise ValueError("AF < 1")
    return target_life_use / af


# ---------------------------------------------------------------------------
# Tabela de tempo equivalente por condição de carga
# ---------------------------------------------------------------------------

def _af_from_spec(spec: dict) -> float:
    """Calcula o AF de uma linha da tabela conforme o modelo do mecanismo."""
    model = spec["model"]
    if model == "arrhenius":
        return arrhenius_af(spec["v_use"], spec["v_accelerated"],
                            b=spec.get("b"), ea_ev=spec.get("ea_ev"))
    if model == "eyring":
        return eyring_af(spec["v_use"], spec["v_accelerated"], spec["b"])
    if model == "ipl":
        return ipl_af(spec["v_use"], spec["v_accelerated"], spec["n"])
    if model == "coffin_manson":
        return coffin_manson_af(spec["dt_use"], spec["dt_accelerated"], spec["m"])
    if model == "hallberg_peck":
        return hallberg_peck_af(spec["v_use"], spec["v_accelerated"],
                                spec["rh_use"], spec["rh_accelerated"],
                                p=spec["p"], b=spec.get("b"), ea_ev=spec.get("ea_ev"))
    raise KeyError(
        f"unknown model {model!r}; expected one of: arrhenius, eyring, ipl, "
        "coffin_manson, hallberg_peck"
    )


def equivalent_time_table(loadings: list[dict]) -> list[dict]:
    """Consolida vários mecanismos numa tabela de **tempo de ensaio equivalente**.

    Cada `loading` descreve um mecanismo de dano com seu modelo, a condição de uso
    e a de ensaio, e a vida-alvo em uso. A função devolve, por linha, o AF e o
    tempo de ensaio que reproduz aquela vida-alvo (`vida_alvo_uso / AF`).

    É a materialização do "guardar o resultado em tempo equivalente por condição
    de carga numa tabela dedicada": permite comparar mecanismos lado a lado e ver
    qual **domina o plano de ensaio** (o de maior tempo equivalente).

    Chaves esperadas em cada `loading`:
        mecanismo    rótulo livre (ex.: "operação (HTOL)")
        model        arrhenius | eyring | ipl | coffin_manson | hallberg_peck
        target_life_use   vida-alvo em uso (h para térmico/umidade; ciclos p/ CM)
        unit         (opcional) unidade da vida — só rótulo; default "h"
        + os parâmetros do modelo (ver `_af_from_spec` e as funções de AF)

    Retorna uma lista de dicts com: mecanismo, model, af, target_life_use,
    tempo_ensaio_equivalente, unit.

    >>> tab = equivalent_time_table([
    ...     {"mecanismo": "operação", "model": "arrhenius", "ea_ev": 0.7,
    ...      "v_use": celsius_to_kelvin(87), "v_accelerated": celsius_to_kelvin(125),
    ...      "target_life_use": 12000.0},
    ...     {"mecanismo": "termomecânico", "model": "coffin_manson", "m": 4.0,
    ...      "dt_use": 76.0, "dt_accelerated": 205.0,
    ...      "target_life_use": 54750.0, "unit": "ciclos"},
    ... ])
    >>> round(tab[0]["af"], 1), round(tab[0]["tempo_ensaio_equivalente"])
    (8.6, 1394)
    >>> round(tab[1]["af"], 1), round(tab[1]["tempo_ensaio_equivalente"])
    (52.9, 1034)
    """
    tabela = []
    for spec in loadings:
        af = _af_from_spec(spec)
        alvo = spec["target_life_use"]
        tabela.append({
            "mecanismo": spec.get("mecanismo", spec["model"]),
            "model": spec["model"],
            "af": af,
            "target_life_use": alvo,
            "tempo_ensaio_equivalente": required_test_time(alvo, af),
            "unit": spec.get("unit", "h"),
        })
    return tabela
