← [Home](Home.md)

# Interpretação do β (e do η)

## β — o parâmetro de forma

Diz **como** o item falha. É a leitura física do ajuste (handbook 2.13–2.16).

| β | Nome | Regime | O que fazer |
|---|---|---|---|
| β < 1 | Mortalidade infantil | taxa de falha **decrescente** | burn-in, revisar processo/montagem. Trocar peça por peça nova **piora**. |
| β = 1 | Falhas aleatórias | taxa **constante** (exponencial) | manutenção preventiva por tempo **não ajuda**. Buscar causa externa. |
| 1 < β < 4 | Desgaste inicial | crescente | troca preventiva começa a fazer sentido |
| β > 4 | Velhice, desgaste rápido | fortemente crescente | há vida bem definida; troca programada é eficaz |

A consequência prática mais cara: **com β ≤ 1, manutenção preventiva baseada em
tempo é contraproducente.** Trocar um item na metade da vida introduz um item novo
com taxa de falha *maior*. Boa parte dos planos de manutenção erra aqui.

## η — a vida característica

É a idade em que **63,2%** da população falhou, sempre — independente de β.

Porque F(η) = 1 − e^(−1) = 0,632. Não é a média, nem a mediana.

```python
from relengy.qualitative.fta import weibull_cdf
weibull_cdf(100, beta=2.0, eta=100)   # 0.632
weibull_cdf(100, beta=0.5, eta=100)   # 0.632  — mesmo valor
```

## Armadilhas

**Um β limpo não prova modo único.** Modos podem estar *"covered"* — mascarados
por um dominante (handbook 2.17). Se há razão física para dois mecanismos,
procure o dogleg.

**β é do modo de falha, não do equipamento.** Uma bomba tem β diferente para
vazamento de selo e para falha de rolamento. Ajustar os dois juntos produz um β
intermediário que não descreve nenhum dos dois.

**Suspensões aumentam η, e afetam pouco β** (handbook 2.11). Se η parece baixo
demais, verifique se as suspensões entraram.

**Compare β do mesmo método.** β obtido por MLE não é comparável a β por MRR —
e a divergência entre eles é o sinal de batch problem, não ruído. Ver
[Escolha do método](Escolha-do-Metodo.md).

## "B lives"

B10 = idade em que 10% da população falhou. B1, B10, B50 são a linguagem de
especificação de vida em rolamentos e aeronáutica. B50 ≠ η (B50 é a mediana).
