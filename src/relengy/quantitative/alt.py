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
    "Arrhenius": "1 térmico (Kelvin)",
    "Eyring": "1 térmico (Kelvin)",
    "Inverse Power Law": "1 não-térmico",
    "Temperature-Humidity": "2 térmicos",
    "Temperature-NonThermal": "térmico (stress_1) + não-térmico (stress_2)",
    "Dual Power": "2 não-térmicos",
}


def fitter_name(model: str, distribution: str = "Weibull") -> str:
    """Nome da função no pacote `reliability` para um modelo da literatura.

    >>> fitter_name("Arrhenius")
    'Fit_Weibull_Exponential'
    >>> fitter_name("IPL", "Lognormal")
    'Fit_Lognormal_Power'
    """
    if model not in MODEL_MAP:
        raise KeyError(f"modelo desconhecido: {model!r}. Conhecidos: {sorted(MODEL_MAP)}")
    if distribution not in ("Weibull", "Lognormal", "Normal", "Exponential"):
        raise ValueError(f"distribuição não suportada: {distribution!r}")
    return f"Fit_{distribution}_{MODEL_MAP[model]}"


# ---------------------------------------------------------------------------
# Temperatura
# ---------------------------------------------------------------------------

def celsius_to_kelvin(c: float) -> float:
    return c + 273.15


def _check_kelvin(*temps: float) -> None:
    for t in temps:
        if t <= 0:
            raise ValueError(f"temperatura {t} <= 0 K é impossível")
        if t < _SUSPICIOUS_KELVIN:
            warnings.warn(
                f"temperatura {t} K = {t - 273.15:.1f} °C. Isso parece Celsius "
                "passado como Kelvin. Arrhenius e Eyring exigem KELVIN — use "
                "celsius_to_kelvin(). O erro é silencioso e o AF sai errado por "
                "ordens de grandeza.",
                UserWarning,
                stacklevel=3,
            )


def activation_energy_to_B(ea_ev: float) -> float:
    """B = Ea / k. Ea em elétron-volts."""
    if ea_ev <= 0:
        raise ValueError("energia de ativação deve ser positiva")
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
        raise ValueError("V e K devem ser positivos")
    return 1.0 / (k * v**n)


# ---------------------------------------------------------------------------
# Fatores de aceleração
# ---------------------------------------------------------------------------

def arrhenius_af(v_use: float, v_accelerated: float, *,
                 b: float | None = None, ea_ev: float | None = None) -> float:
    """AF = e^(B·(1/V_uso − 1/V_acc)). Informe B **ou** a energia de ativação."""
    if (b is None) == (ea_ev is None):
        raise ValueError("informe exatamente um: b ou ea_ev")
    _check_kelvin(v_use, v_accelerated)
    if v_accelerated <= v_use:
        raise ValueError(
            f"estresse acelerado ({v_accelerated} K) deve ser MAIOR que o de uso "
            f"({v_use} K); caso contrário não há aceleração"
        )
    b = activation_energy_to_B(ea_ev) if b is None else b
    return math.exp(b * (1.0 / v_use - 1.0 / v_accelerated))


def ipl_af(v_use: float, v_accelerated: float, n: float) -> float:
    """AF = (V_acc / V_uso)^n. Não depende de K."""
    if v_use <= 0 or v_accelerated <= 0:
        raise ValueError("estresses devem ser positivos")
    if v_accelerated <= v_use:
        raise ValueError("estresse acelerado deve ser maior que o de uso")
    if n <= 0:
        raise ValueError("n deve ser positivo (vida cai com o estresse)")
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
        raise ValueError("estresse acelerado deve ser maior que o de uso")
    return (v_accelerated / v_use) * math.exp(b * (1.0 / v_use - 1.0 / v_accelerated))


def extrapolate_life(life_accelerated: float, af: float) -> float:
    """Vida na condição de uso = vida acelerada × AF."""
    if af < 1.0:
        raise ValueError(f"AF = {af} < 1: o ensaio não acelerou nada")
    return life_accelerated * af


def required_test_time(target_life_use: float, af: float) -> float:
    """Quanto tempo de ensaio acelerado equivale a uma vida-alvo em uso.

    Nome deliberadamente NÃO iniciado por `test_`: o pytest coleta qualquer
    função com esse prefixo, mesmo em módulos de produção importados por testes.
    """
    if af < 1.0:
        raise ValueError("AF < 1")
    return target_life_use / af
