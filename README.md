# relengy

**Reliability engineering in Python — with the method decisions written down where the code makes them.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Tests](https://img.shields.io/badge/tests-117%20passing-brightgreen)

Life data analysis, reliability growth, accelerated testing, reliability block
diagrams, FMEA and fault trees.

Most reliability code answers *how*. This library also answers *why this way*:
where two authorities disagree — and in reliability they disagree often — the
disagreement is quoted in the docstring, and the code takes a documented side.

---

## Why another one

The excellent [`reliability`](https://reliability.readthedocs.io/) package fits
distributions. `relengy` sits one layer up, on the decisions that surround the fit:

- **Which estimation method?** Abernethy (*The New Weibull Handbook*, 5.3.3) says
  median rank regression, flatly: *"MLE is not recommended."* ReliaSoft qualifies:
  MLE when the sample is large, or censoring is heavy or uneven. Both are right about
  different data. `recommend_method()` decides, and tells you why.
- **Is my sample even one population?** A weak fit is a hypothesis about the model,
  not a verdict on the data.
- **Does the number survive contact with the physics?** A fitted initial thickness of
  54,631 mm is a belt 54 metres thick. The library is built by someone who once
  printed that number and did not look at it.

---

## Install

```bash
pip install relengy                # core: fitting, ranks, growth, alt, rbd, fmea, fta
pip install "relengy[doe]"         # + design of experiments (pyDOE3)
pip install "relengy[docx]"        # + Word report output (python-docx)
pip install "relengy[all]"
```

Core dependencies are deliberately few: `numpy`, `pandas`, `reliability`. A library
should not drag JupyterLab into the environment of someone who only wants to fit a
Weibull.

---

## Quick start

### Fit a Weibull, and be told which method to trust

```python
from relengy.quantitative.fitting import fit_diagnostic

failures      = [30, 49, 82, 90, 96]
right_censored = [100, 100, 100]

d = fit_diagnostic(failures, right_censored)
print(d.regime())                    # 'desgaste inicial (1 < beta < 4)'
print(d.recommendation().method)     # 'RRX'
print(d.recommendation().reason)     # the reasoning, and whose it is
print(d.report())                    # the whole diagnosis, in prose
```

`d.batch_suspected` is the one worth staring at: when the MLE shape parameter comes
in far below the regression one, that is not numerical noise — it is a hint that the
sample mixes more than one population.

### Is this repairable system getting better or worse?

```python
from relengy.quantitative.growth import crow_amsaa

# cumulative operating times of successive failures on one system
fit = crow_amsaa([100, 180, 320, 610, 1200], termination="failure")

print(fit.beta)                    # < 1 → improving,  > 1 → deteriorating
print(fit.instantaneous_mtbf())    # the MTBF you have now
print(fit.cumulative_mtbf())       # the average of the past — a different question
print(fit.verdict())
```

### A fault tree that refuses to lie to you

```python
from relengy.qualitative.fta import FaultTree

ft = FaultTree(
    top="TOP",
    gates={"TOP": ("OR", ["power_loss", "G1"]),
           "G1":  ("AND", ["pump_fails", "standby_fails"])},
)

print(ft.minimal_cut_sets())   # [frozenset({'power_loss'}), frozenset({'pump_fails', 'standby_fails'})]
print(ft.top_probability({"power_loss": 0.01, "pump_fails": 0.02, "standby_fails": 0.05}))
```

The docstring for that last one tells you, in as many words, that independence is an
assumption and not a fact — items from the same batch, in the same environment,
maintained by the same crew fail together — and that under common cause this number
underestimates an AND gate and overestimates an OR gate.

---

## What's inside

| Module | What it does |
|---|---|
| `quantitative.fitting` | Weibull fitting; RRX / RRY / MLE, and which one to use |
| `quantitative.ranks` | Median ranks, Benard's approximation, adjusted ranks under suspensions |
| `quantitative.growth` | Crow-AMSAA / Duane; time- and failure-terminated, biased and unbiased |
| `quantitative.alt` | Accelerated life testing: Arrhenius, inverse power law, acceleration factor |
| `quantitative.rbd` | Reliability block diagrams: series, parallel, k-out-of-n; Birnbaum and criticality importance |
| `quantitative.doe` | Full and fractional factorials, Plackett-Burman, central composite |
| `qualitative.fmea` | FMEA/FMECA as tabular data with a schema, not as a spreadsheet |
| `qualitative.fta` | Fault trees, minimal cut sets, exact and rare-event top probability |
| `io.censoring` | Right-censored data, without silently dropping the suspensions |
| `report` | Describe the analysis once; render Markdown and Word from the same source |

---

## Status and scope

**v0.1 — alpha.** 117 tests passing. The API may still move.

Deliberately not published yet: a knowledge-graph module (failure mechanisms as
first-class nodes) and failure-mode similarity. They work, but the design has not
settled, and an unstable module in a small library costs more than it gives.

**On language:** identifiers and the public API are English. Docstrings — and the
explanatory strings the library returns, such as `regime()`, `verdict()` and
`recommendation().reason` — are still in Portuguese. They are dense technical
arguments rather than comments, and translating them badly would be worse than
leaving them for now. You can see this in the first example above.

Translation is happening in two passes: the returned strings first (a small,
bounded set), the docstrings module by module after. Contributions welcome — that
is a good first issue.

---

## Method references

- Abernethy, R. B. *The New Weibull Handbook*, 5th ed.
- ReliaSoft ReliaWiki — Life Data Analysis, ALT, Reliability Growth, System Analysis, DOE.
- MIL-STD-1629A (FMECA), ISO 14224 (taxonomy of failure data).

Reading notes on where these sources conflict, and how this library resolves the
conflict, are in [`docs/abernethy-weibull-notes.md`](docs/abernethy-weibull-notes.md).

---

## License

MIT. See [LICENSE](LICENSE).

Built by [Marcos de Paula](https://www.linkedin.com/in/marcospaula/) — reliability
engineer, 20+ years across aerospace, mining and subsea oil & gas.
