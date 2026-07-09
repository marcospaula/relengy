← [Home](Home.md)

# Glossário

**Adjusted rank** — rank corrigido pela presença de suspensões (fórmula de
Auth/Leonard Johnson). Entra na aproximação de Bernard no lugar do rank bruto.

**ALT** — *Accelerated Life Testing*. Ensaio sob estresse elevado (temperatura,
carga, tensão) para induzir falhas mais rápido, extrapolando para condição normal
via Arrhenius, Eyring, Inverse Power Law.

**AP** — *Action Priority*. Substitui o RPN no AIAG-VDA (2019). Tabela de decisão
S×O×D → {H, M, L}, não fórmula.

**B life** — B10 é a idade em que 10% da população falhou. B50 é a mediana (≠ η).

**Batch problem** — lotes com qualidade distinta misturados na amostra. Sinal:
β_MLE ≪ β_MRR, ou reta íngreme seguida de reta rasa no gráfico.

**Bernard (aproximação de)** — median rank ≈ (i − 0.3)/(N + 0.4). Precisa a ~1%
para N = 5 e 0.1% para N = 50.

**β (beta)** — parâmetro de **forma** da Weibull. Diz *como* o item falha.
Ver [Interpretação do β](Interpretacao-do-Beta.md).

**Bowtie** — diagrama que junta uma árvore de falhas (causas) e uma árvore de
eventos (consequências) em torno de um evento de topo, com barreiras nos dois lados.

**Censura à direita** → *suspensão*.

**Censura intervalar** — sabe-se apenas que a falha ocorreu entre duas inspeções.
Força MLE.

**Crow-AMSAA (NHPP)** — modelo de crescimento de confiabilidade para sistemas
reparáveis. Processo de Poisson não-homogêneo. Também chamado Duane quando na
forma gráfica original.

**Cm / Cr** — *criticality numbers* do MIL-STD-1629A. `Cm = β·α·λp·t` por modo;
`Cr = Σ Cm` por item. Aqui o β é probabilidade condicional de perda de missão —
**não** é o β da Weibull. Nome colidente, conceitos distintos.

**Cut set (minimal)** — menor combinação de eventos básicos cuja ocorrência
simultânea causa o evento de topo de uma FTA.

**Dauser shift** — correção para suspensões com tempos desconhecidos (handbook 5.7).

**Dogleg** — joelho no gráfico Weibull. Indica mistura de modos de falha.

**η (eta)** — vida **característica**. A idade em que **63,2%** da população
falhou, para qualquer β.

**FMECA** — FMEA + análise de criticidade (o "C"). MIL-STD-1629A.

**λp (lambda_p)** — taxa de falha da peça. λ = 1/η **somente** se β = 1.

**MRR / RRX / RRY** — *Median Rank Regression*. RRX regride X em Y ("X on Y" no
handbook); RRY o inverso. Dão resultados diferentes.

**MLE-RBA** — MLE com *Reduced Bias Adjustment*. Corrige o viés do β do MLE em
amostra pequena (handbook 5.3). Implementado no pacote `predictr`.

**Mortalidade infantil** — β < 1, taxa de falha decrescente. Manutenção preventiva
por tempo **piora** a situação.

**RPN** — *Risk Priority Number* = S × O × D. Legado; ver [FMEA](FMEA-e-FMECA.md).

**Suspensão** — item que **não** falhou até a idade observada (censura à direita).
Notação do SuperSMITH: valor negativo. Suspensões **aumentam η**.

**Sudden death** — plano de teste em que grupos são testados até a primeira falha
de cada grupo (handbook 6.14).

**t₀** — parâmetro de posição da Weibull 3P. Positivo = período garantido sem
falha; negativo = peças "velhas" (envelhecimento de prateleira).

**Weibayes** — Weibull de 1 parâmetro: β fixado, só η estimado. A resposta para
amostra pequena. Requer β de fonte externa → Weibull Library.
