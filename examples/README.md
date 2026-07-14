# Examples

Worked reliability cases solved with `relengy`.

## Case 4 — Test-bench reliability with competing failure modes

`case4_test_bench.py` · *adapted from a ReliaSoft Brasil training case ("Caso 4:
Bancada de Prova").*

### The problem

A part is qualified on a test bench. Company norm: it must reach **90% reliability
at 20,000 cycles**, demonstrated at a **90% lower one-sided confidence bound**.

Twenty parts were tested. Three distinct failure modes appeared (MFA, MFB, MFC);
six parts were still running at 120,000 cycles (suspensions).

| Mode | Cycles to failure | Count |
|------|-------------------|-------|
| MFA  | 65,800 · 81,500 · 85,000 · 89,900 · 115,000 | 5 |
| MFB  | 41,200 · 49,800 · 58,900 · 76,800 | 4 |
| MFC  | 29,800 · 29,900 · 31,250 · 48,900 · 54,000 | 5 |
| —    | 120,000 (suspended) | 6 |

**Questions.** (1) Would you approve the part? (2) Which failure mode would you
improve first, and why?

### The method: competing failure modes

The three modes are **independent competing risks**, so you do **not** fit one
Weibull to the 14 pooled times. Each mode is fit on its own, with the *other*
modes' failures entering as **suspensions** (a part left the risk set of one mode
because another mode failed it first). The part is a **series system**:

```
R_part(t) = R_MFA(t) · R_MFB(t) · R_MFC(t)
```

### What `relengy` contributes

Per mode, `fit_diagnostic` runs MRR and MLE together, recommends a method, and
flags a possible batch problem. Here that diagnosis is decisive: with only 4–5
failures per mode under 75–80% censoring, it (a) recommends MLE for the heavy
censoring, but (b) **cautions** that small-sample MLE is badly biased, and (c)
raises a **batch-problem alert** because MLE and MRR disagree sharply. Together
those say *don't trust the MLE fit* — and indeed the MLE fit for MFC is
physically impossible (η ≈ 266,000 cycles with failures at 30–54k). So we read
the credible **MRR** fit.

| Mode | β (MRR) | η (MRR) | R(20k) point | R(20k) 90% lower |
|------|--------:|--------:|-------------:|-----------------:|
| MFA  | 4.32 | 121,202 | 0.9996 | 0.989 |
| MFB  | 3.13 | 107,499 | 0.9948 | 0.879 |
| MFC  | 3.41 | **69,846** | 0.9860 | 0.982 |
| **System** | | | **98.1%** | **~85%** |

### Answers

**1) Approve? No.** The point estimate clears the bar (R_system(20k) = 98.1% by
MRR, 93.5% by MLE), but the norm asks for the **90% lower one-sided bound**, and
there the system falls to **~85%** (MRR) / ~78% (MLE) — below 90%. A single mode,
MFB, already sits at 87.9%, and the system is the product of all three. The test
is **under-powered**: 14 failures spread over three competing modes cannot
*demonstrate* 90% at 90% confidence. Verdict: do not approve; gather more data
(more parts / more cycles) before qualifying.

**2) Improve MFC first.** It has the lowest characteristic life (η ≈ 69,846), the
earliest failures (starting ~29,800 cycles, closest to the 20,000 target), and it
contributes roughly **71% of the system's unreliability** at 20k. Raising MFC's
life moves the system the most. (Note: on the *lower bound*, MFB drops the most —
but that reflects its smaller sample, i.e. a need for more **data**, not a worse
physical mode. Engineering effort goes to the dominant real risk, MFC.)

### The gap this case surfaced

`relengy` returns point estimates only; the 90% lower bound above is computed by
reaching down to the wrapped `reliability` fit (Fisher-matrix / delta method). The
norm's question hinges on that bound — so exposing confidence bounds natively is
tracked in **issue #11**.

### Run it

```bash
python examples/case4_test_bench.py
```
