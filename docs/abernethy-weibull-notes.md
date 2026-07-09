# The New Weibull Handbook (Abernethy, 5ª ed.) — notas de trabalho

**Fonte:** `library/books/Robert-B-Abernethy-The-New-Weibull-Handbook-R-B-Abernethy-2006.pdf`
**Texto pesquisável:** `library/books/abernethy-nwh-5ed.txt` (~21.5k linhas)

> **Aviso sobre o OCR.** O PDF foi digitalizado com Acrobat Paper Capture. O texto tem
> ruído sistemático: `cOlTection` → *correction*, `fonnula` → *formula*, `Benard` → *Bernard*,
> `11 is` → *It is*, e **`to` frequentemente significa `t₀`**. Ao citar o livro, confira a
> página no PDF. Ao fazer `grep`, use padrões tolerantes.

---

## Como este livro é usado neste projeto

O handbook é a referência normativa para o lado **quantitativo**. Os slides de
Fulton & Tarum (`library/slides/fulton-tarum-2021.txt`) seguem a **mesma numeração de
capítulos** e servem como resumo executivo — use-os para localizar o tema, depois
vá ao livro para o detalhe.

## Mapa dos capítulos

| Cap. | Tema | Onde importa aqui |
|---|---|---|
| 1 | Visão geral da análise Weibull | — |
| 2 | Plotagem e interpretação (median ranks, β) | `notebooks/quantitative/` |
| 3 | Dados sujos, "bad Weibulls", correção t₀, mistura de modos | Triagem de dados |
| 4 | Previsão de falhas = análise de risco | Forecast de peças/garantia |
| 5 | MLE, MRR, e o **RBA** (reduced bias adjustment) | **Define o método padrão** |
| 6 | Weibayes e testes de substanciação; zero-failure; acelerado | Amostras pequenas |
| 7 | Estimativas intervalares (bounds) | Incerteza |
| 8 | Modelos relacionados: binomial, Poisson, exponencial, Kaplan-Meier | — |
| 9 | Crow-AMSAA, garantia, custo de ciclo de vida (LCC) | Crescimento de confiabilidade |
| 10 | Síntese + fluxograma de best practice | Checklist final |

---

## Decisões técnicas que o livro impõe ao projeto

### 1. O método de ajuste padrão é MRR (X on Y), não MLE

Citação literal (linhas 6177–6183 do texto extraído):

> "Which is the best practice, MLE with RBA factors or Median Rank Regression, X on Y?
> For engineers, the author recommends **MRR as best practice** as it provides a good
> graphic plot and is the simplest method. […] **MLE is not recommended.**"

Consequência prática: no pacote `reliability`, isso significa `method='RRX'`
(rank regression on X) — **não** o `method='MLE'` que é o padrão da biblioteca.

```python
from reliability.Fitters import Fit_Weibull_2P
fit = Fit_Weibull_2P(failures=dados, method='RRX')   # best practice do Abernethy
```

### 2. Rodar MRR e MLE-RBA juntos é um diagnóstico, não redundância

> "Why not use both methods, MRR and MLE-RBA? If the results are similar this is
> additional evidence that the fit is good. **If the MLE-RBA beta is much less than the
> MRR beta, there may be a batch problem.**"

Isto é, a *discordância* entre os dois métodos é sinal físico — indica **problema de
lote**. Reaparece na lista de sintomas de "bad Weibull" (linha 11702):
*"The MLE beta is shallow compared to the MRR beta. (Section 5.3.3)"*
seguido de *"steep line followed by shallow line indicates a batch problem"*.

O pacote `predictr` implementa MRR **com correção de viés**, cobrindo o RBA do cap. 5.

### 3. Median rank via aproximação de Bernard

Equação (2-6), linha 1782:

```
Median Rank de Bernard = (i - 0.3) / (N + 0.4)
```

Precisão declarada (linha 1779): **~1% para N = 5 e 0.1% para N = 50**.

Com **suspensões** (dados censurados à direita), o rank `i` não é o rank bruto: usa-se
o *adjusted rank* pela fórmula de Auth/Leonard Johnson (eq. 2-5, linha 1752):

```
Adjusted Rank = [ (Reverse Rank) × (Previous Adjusted Rank) + (N + 1) ] / (Reverse Rank + 1)
```

e só então o adjusted rank entra em Bernard. Exemplo do livro (linha 1795): adjusted
rank 1.125 com N=8 → `(1.125 - 0.3) × 100 / (8 + 0.4)` = **9.82%**.

> Suspensões **aumentam η** (seção 2.11). Ignorá-las enviesa a vida para baixo.

### 4. Interpretação de β (cap. 2.13–2.16) — a leitura física

| β | Significado | Regime da curva da banheira |
|---|---|---|
| β < 1 | Mortalidade infantil | Falhas decrescentes |
| β = 1 | Falhas aleatórias (exponencial) | Taxa constante |
| 1 < β < 4 | Desgaste inicial (*early wear out*) | Crescente |
| β > 4 | Desgaste de velhice, rápido (*old age*) | Fortemente crescente |

**Cuidado (2.17):** modos de falha Weibull podem estar *"covered"* — mascarados por
outro modo dominante. Um β limpo não prova modo único.

### 5. Amostra pequena → Weibayes

Fulton & Tarum são categóricos (slides, linhas 1019–1025):

> "Weibayes (1-Parameter Weibull) = Combination of Weibull and Bayesian.
> **Weibayes is the best solution for small samples.** Weibayes requires input of an
> appropriate Beta slope value."

Amostra "pequena" = **≤ 20 ocorrências** (padrão do handbook, citado no slide 8).

O β de entrada vem de resultados anteriores em equipamento similar falhando do mesmo
modo. É exatamente para isso que existe a **Weibull Library** (ver abaixo).

### 6. A "Weibull Library" é a ideia central, não um extra

Slide final (linha 1024):

> "Save the results from every Weibull solution to start and maintain your
> 'Weibull Library' >>> **MOST IMPORTANT IDEA FOR WE!**"

Por isso este projeto tem `weibull_library/` como estrutura de primeira classe:
cada análise concluída deposita ali seu β, η, modo de falha e contexto, alimentando os
Weibayes futuros. Cap. 6.19 do handbook ("Weibull Libraries and Lessons Learned").

---

## Convenção de dados (do handbook e do software SuperSMITH)

Entrada em coluna única:
- **valor positivo** = falha (*occurrence*)
- **valor negativo** = suspensão (*right-censored*), ex.: `-91`

Os pacotes Python separam em duas listas (`failures=`, `right_censored=`).
Ver `src/relengy/io/` para o conversor.

---

## Pontos ainda não destilados

Lidos apenas o índice, o prefácio, e as seções 2.10, 5.3.3 e o wrap-up dos slides.
**Não** foram lidos em profundidade — abrir sob demanda:

- Cap. 3: correção t₀ e discriminação Weibull-3P vs Lognormal (requer ≥ 20 falhas)
- Cap. 4: cálculo de failure forecast (com e sem reposição), intervalo ótimo de troca
- Cap. 5.7: **Dauser shift** (tempos de suspensão desconhecidos)
- Cap. 5.8: 5 opções para dados de inspeção/intervalares (Probit, Kaplan-Meier, MLE intervalar)
- Cap. 6.10: planos de teste *zero-failure*; 6.14: *sudden death*
- Cap. 7.3: bounds — beta-binomial, Fisher matrix, likelihood ratio, pivotal Monte Carlo
- Cap. 9: Crow-AMSAA (soluções IEC 61649 vs regressão)
- Cap. 10.10: fluxograma de best practice — **vale transcrever como checklist**
