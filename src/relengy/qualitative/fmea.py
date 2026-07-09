"""FMEA / FMECA — três metodologias, deliberadamente separadas.

Não existe pacote maduro de FMEA no PyPI. Os nomes plausíveis são homônimos de
outros domínios (`fmea` é criptografia; `pyrca` é estrutura de concreto). FMEA é,
na prática, uma planilha com regras — o valor está em validar o schema e calcular
os índices de forma consistente e auditável.

Três esquemas de priorização, que NÃO são intercambiáveis:

1. RPN = S x O x D  (SAE J1739 / AIAG 4ª ed. — LEGADO)
   Criticado com razão: 1x10x10 = 10x10x1 = 100, mas severidade 1 é irrelevante e
   severidade 10 indetectável é crítica. Mantido para compatibilidade histórica.

2. AP — Action Priority (AIAG-VDA 1ª ed., 2019 — ATUAL)
   Substitui o RPN por uma tabela de decisão S x O x D -> {H, M, L}. NÃO é uma
   fórmula. Ver `action_priority()`: a tabela é carregada de arquivo, não
   embutida aqui. Ver a docstring de `load_ap_table` para o porquê.

3. Criticality Number — MIL-STD-1629A, Task 102 (aeroespacial / defesa)
   Quantitativo, não ordinal. Verificado contra o texto do próprio padrão em
   `library/standards/mil_std_1629a.txt`, secoes 3.2.1.3 a 3.2.1.7:

       Cm = beta * alpha * lambda_p * t          (por modo de falha)
       Cr = sum(Cm) sobre os modos do item       (por item)

   onde beta = probabilidade condicional de perda de missão,
        alpha = failure mode ratio (fração da taxa de falha devida a este modo),
        lambda_p = taxa de falha da peça,
        t = duração da fase da missão (horas ou ciclos).

   Este é o elo com o lado quantitativo: lambda_p pode vir da Weibull Library.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema comum
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "item",
    "failure_mode",
    "effect",
    "cause",
    "control",
    "severity",     # S, 1-10
    "occurrence",   # O, 1-10
    "detection",    # D, 1-10 (10 = não detectável)
]

SCALE_COLUMNS = ["severity", "occurrence", "detection"]

ActionPriority = Literal["H", "M", "L"]

# MIL-STD-1629A, 4.4.3 — classificação de severidade (categorias, não 1-10).
MIL_SEVERITY_CATEGORIES = {
    "I": "Catastrophic - may cause death or loss of the weapon system",
    "II": "Critical - severe injury, major damage, results in mission loss",
    "III": "Marginal - minor injury or damage, mission delay or degradation",
    "IV": "Minor - no injury or damage; causes unscheduled maintenance",
}


@dataclass
class FMEAConfig:
    """Limiares do RPN legado. Ajuste ao padrão da sua organização."""

    rpn_threshold: int = 100
    severity_critical: int = 9
    scale_min: int = 1
    scale_max: int = 10


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, cfg: FMEAConfig | None = None) -> pd.DataFrame:
    """Valida schema e escalas. Levanta ValueError listando todos os problemas."""
    cfg = cfg or FMEAConfig()
    problems: list[str] = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    for col in SCALE_COLUMNS:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any():
            problems.append(f"'{col}' is not numeric on rows {df.index[s.isna()].tolist()}")
            continue
        out = ~s.between(cfg.scale_min, cfg.scale_max)
        if out.any():
            problems.append(
                f"'{col}' outside the {cfg.scale_min}-{cfg.scale_max} scale "
                f"on rows {df.index[out].tolist()}"
            )

    if problems:
        raise ValueError("invalid FMEA:\n  - " + "\n  - ".join(problems))
    return df


# ---------------------------------------------------------------------------
# 1. RPN legado (SAE J1739)
# ---------------------------------------------------------------------------

def score_rpn(df: pd.DataFrame, cfg: FMEAConfig | None = None) -> pd.DataFrame:
    """RPN clássico. LEGADO: prefira `score_ap` para trabalho novo."""
    cfg = cfg or FMEAConfig()
    validate(df, cfg)

    out = df.copy()
    out["rpn"] = out.severity * out.occurrence * out.detection
    out["criticality_so"] = out.severity * out.occurrence  # ignora detecção

    out["action_required"] = (out.rpn >= cfg.rpn_threshold) | (
        out.severity >= cfg.severity_critical
    )
    out["driver"] = "—"
    hi_rpn = out.rpn >= cfg.rpn_threshold
    hi_sev = out.severity >= cfg.severity_critical
    out.loc[hi_rpn, "driver"] = "RPN"
    out.loc[hi_sev, "driver"] = "severidade"
    out.loc[hi_rpn & hi_sev, "driver"] = "RPN + severidade"

    return out.sort_values(
        ["action_required", "severity", "rpn"], ascending=[False, False, False]
    )


# ---------------------------------------------------------------------------
# 2. AIAG-VDA Action Priority
# ---------------------------------------------------------------------------

def ap_table_template() -> pd.DataFrame:
    """CSV vazio com as 1000 combinações S x O x D, coluna `ap` em branco."""
    rows = product(range(1, 11), repeat=3)
    df = pd.DataFrame(rows, columns=["severity", "occurrence", "detection"])
    df["ap"] = ""
    return df


def load_ap_table(path: str | Path) -> dict[tuple[int, int, int], str]:
    """Carrega a tabela AP de um CSV com colunas severity, occurrence, detection, ap.

    Por que a tabela NÃO está embutida neste arquivo:

      - Ela tem 1000 células e é publicada no manual AIAG-VDA FMEA (1ª ed., 2019),
        que é material licenciado. Transcrevê-la de memória é irresponsável:
        uma única célula errada inverte a prioridade de uma ação de segurança.
      - Organizações frequentemente customizam a tabela.

    Preencha `templates/fmea/aiag_vda_ap_table.csv` a partir do seu exemplar do
    manual. `validate_ap_table` checa completude e monotonicidade depois.
    """
    df = pd.read_csv(path)
    need = {"severity", "occurrence", "detection", "ap"}
    if not need.issubset(df.columns):
        raise ValueError(f"the AP table CSV needs the columns {sorted(need)}")

    df = df[df.ap.astype(str).str.strip() != ""]
    table = {}
    for r in df.itertuples(index=False):
        ap = str(r.ap).strip().upper()
        if ap not in {"H", "M", "L"}:
            raise ValueError(f"invalid AP {ap!r} at S={r.severity} O={r.occurrence} D={r.detection}")
        table[(int(r.severity), int(r.occurrence), int(r.detection))] = ap
    return table


def validate_ap_table(table: dict[tuple[int, int, int], str]) -> None:
    """Checa completude (1000 células) e monotonicidade.

    Monotonicidade: aumentar S, O ou D — mantendo os outros dois fixos — nunca
    pode DIMINUIR a prioridade de ação. Isso não prova que a tabela é a do
    manual, mas pega erros de transcrição que trocam H por L.
    """
    rank = {"L": 0, "M": 1, "H": 2}
    expected = set(product(range(1, 11), repeat=3))
    missing = expected - set(table)
    if missing:
        raise ValueError(f"incomplete AP table: {len(missing)} cells missing, e.g. {sorted(missing)[:5]}")

    violations = []
    for s, o, d in expected:
        for ds, do, dd in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            nxt = (s + ds, o + do, d + dd)
            if nxt in table and rank[table[nxt]] < rank[table[(s, o, d)]]:
                violations.append(((s, o, d), table[(s, o, d)], nxt, table[nxt]))
    if violations:
        raise ValueError(
            f"AP table is not monotonic at {len(violations)} pontos, ex.: {violations[:3]}"
        )


def score_ap(df: pd.DataFrame, table: dict[tuple[int, int, int], str]) -> pd.DataFrame:
    """Atribui AP (H/M/L) por linha, via lookup. Ordena por prioridade."""
    validate(df)
    out = df.copy()

    def lookup(r) -> str:
        key = (int(r.severity), int(r.occurrence), int(r.detection))
        if key not in table:
            raise KeyError(f"S/O/D combination missing from the AP table: {key}")
        return table[key]

    out["ap"] = out.apply(lookup, axis=1)
    order = {"H": 0, "M": 1, "L": 2}
    out["_k"] = out.ap.map(order)
    return out.sort_values(["_k", "severity"], ascending=[True, False]).drop(columns="_k")


# ---------------------------------------------------------------------------
# 3. MIL-STD-1629A Criticality (Task 102)
# ---------------------------------------------------------------------------

def failure_mode_criticality(beta, alpha, lambda_p, t):
    """Cm = beta * alpha * lambda_p * t   (MIL-STD-1629A, 3.2.1.6).

    beta     : probabilidade condicional de perda de missão, 0..1
    alpha    : failure mode ratio, 0..1 (soma dos modos de um item = 1)
    lambda_p : taxa de falha da peça (falhas por hora ou por ciclo)
    t        : duração da fase da missão (mesma unidade de lambda_p)
    """
    beta = np.asarray(beta, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError("beta must lie in [0, 1]")
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha (failure mode ratio) must lie in [0, 1]")
    return beta * alpha * np.asarray(lambda_p, dtype=float) * np.asarray(t, dtype=float)


def item_criticality(df: pd.DataFrame, group: str = "item") -> pd.DataFrame:
    """Cr = soma dos Cm dos modos do item, por classificação de severidade.

    Espera colunas: item, severity_category, beta, alpha, lambda_p, t.
    O padrão soma Cm dentro de uma MESMA categoria de severidade — somar entre
    categorias mistura consequências incomparáveis.
    """
    need = {"item", "severity_category", "beta", "alpha", "lambda_p", "t"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing columns: {sorted(need - set(df.columns))}")

    bad = df.loc[~df.severity_category.isin(MIL_SEVERITY_CATEGORIES), "severity_category"]
    if len(bad):
        raise ValueError(f"invalid categories {bad.unique().tolist()}; use I, II, III, IV")

    out = df.copy()
    out["cm"] = failure_mode_criticality(out.beta, out.alpha, out.lambda_p, out.t)

    # Checagem física: os alphas dos modos de um item devem somar 1.
    sums = out.groupby(group).alpha.sum()
    off = sums[(sums - 1.0).abs() > 1e-6]
    if len(off):
        raise ValueError(
            "failure mode ratios (alpha) do not sum to 1 for: "
            + ", ".join(f"{k} (soma={v:.4f})" for k, v in off.items())
        )

    return (
        out.groupby([group, "severity_category"], as_index=False)
        .cm.sum()
        .rename(columns={"cm": "cr"})
        .sort_values("cr", ascending=False)
    )


def template() -> pd.DataFrame:
    """Planilha FMEA vazia com o schema correto."""
    return pd.DataFrame(columns=REQUIRED_COLUMNS)
