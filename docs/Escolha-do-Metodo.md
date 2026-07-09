← [Home](Home.md)

# Escolha do método: MRR ou MLE?

As duas referências centrais do projeto **não dizem a mesma coisa**. Isso não é
detalhe acadêmico: muda o β que você reporta.

## Abernethy — categórico

> "For engineers, the author recommends MRR as best practice […] **MLE is not
> recommended.**"
> — *The New Weibull Handbook*, 5.3.3

## ReliaWiki (ReliaSoft) — condicional

> "our recommendation is to use **rank regression** when the sample sizes are
> **small and without heavy censoring**. When **heavy or uneven censoring** is
> present, when a high proportion of **interval data** is present and/or when the
> **sample size is sufficient**, **MLE should be preferred**."
> — *Parameter Estimation*

## Onde concordam

Ambos desconfiam do MLE em amostra pequena, e o ReliaWiki explica por quê:

> "MLE estimates of the shape parameter for the Weibull distribution are **badly
> biased for small sample sizes**, and the effect can be increased depending on
> the amount of censoring."

E ambos reconhecem o que o MLE faz melhor:

> "MLE can handle suspensions and interval data better than rank regression […]
> It can also provide estimates with **one or no observed failures**, which rank
> regression cannot do."

## Síntese

O "MLE is not recommended" de Abernethy vale para o **caso dele**: tempos exatos
de falha, censura moderada. Ele não cobre censura pesada nem dados de inspeção.
MRR depende de *plotting positions*; sob censura pesada, os ranks ajustados
carregam pouca informação, e aí o MLE ganha.

| Situação | Método | Por quê |
|---|---|---|
| 0 falhas | MLE / Weibayes | MRR não tem ranks para ajustar |
| Dados intervalares | MLE | MRR exige tempos exatos |
| Censura ≥ 50% ou desigual | MLE | plotting positions perdem informação |
| ↑ **e** amostra pequena | MLE + **RBA** | β do MLE enviesado; use `predictr` ou Weibayes |
| Amostra pequena, censura leve | **RRX** | as duas fontes concordam |
| Amostra grande, censura leve | RRX | best practice; MLE também serve |

```python
from relengy.quantitative.fitting import recommend_method

recommend_method(n_failures=8, n_censored=1)
# RRX: amostra pequena (8 falhas) e censura leve: exatamente o caso em que
#      Abernethy e ReliaSoft concordam

recommend_method(n_failures=3, n_censored=17)
# MLE: censura pesada (85% suspensões): as plotting positions do MRR perdem informação
#   CUIDADO: amostra pequena (3 falhas): o beta do MLE é notoriamente enviesado
#   aqui. Use correção de viés (RBA) — o pacote `predictr` implementa — ou Weibayes.
```

## Duas ressalvas honestas

**Os limiares são nossos.** Nem Abernethy nem o ReliaWiki fixam um número para
"heavy censoring" — ambos dizem apenas *"heavy"*. Os 50% estão em
`HEAVY_CENSORING_FRACTION`, constante nomeada, para você calibrar contra a sua
Weibull Library. O mesmo vale para `BATCH_SUSPICION_RATIO = 0.75`.

**`uneven_censoring` não se infere das contagens.** Vinte suspensões concentradas
no fim do teste são qualitativamente diferentes de vinte espalhadas entre as
falhas — e a razão censurados/total é idêntica nos dois casos. É julgamento do
analista, por isso é um parâmetro explícito e não um cálculo.

## No pacote `reliability`

O padrão da biblioteca é `method='MLE'`. Passe explicitamente:

```python
from reliability.Fitters import Fit_Weibull_2P
Fit_Weibull_2P(failures=f, right_censored=c, method='RRX')
```

`RRX` = rank regression on X (o que Abernethy chama de "X on Y"). `RRY` regride na
outra direção e dá resultado diferente — o handbook (5.4) discute quando cada uma.
