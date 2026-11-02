# Protocolo M6: carteira aprendida no patrimônio depois de IR (`tax-portfolio-v1`)

Registrado em 2026-11-01, **antes de o modelo ser rodado nos dados reais**. Até aqui, o programa só rodou em mercados sintéticos, para conferir que funciona.

O SHA-256 deste arquivo está na regra do protocolo, dentro de [`tax_portfolio.krs`](tax_portfolio.krs), e o livro-razão (`results/holdout_ledger.csv`) grava o hash do protocolo na primeira execução. Se este texto mudar, o hash não bate.

## Pergunta

Uma carteira long-only cujos parâmetros são treinados por gradiente **no patrimônio depois de custos e do IR de pessoa física** bate, na B3, as alternativas do mesmo universo? As comparações são o equal-weight e o momentum, todas no simulador realista do laboratório, com custos reais e IR reais.

O que há de novo é o objetivo de treino. As regras do IR entram no gradiente:
- isenção de R$ 20 mil de vendas no mês;
- 15% sobre o ganho líquido;
- prejuízo compensado nos meses seguintes;
- custo médio.

Assim o modelo pode aprender a negociar mais devagar para ficar isento, ou a sair do mercado.

## Dados e universo

- **Fonte:** dados do laboratório `quant-b3`: retornos mensais totais, universo NEFIN (`b3.eligible()`) e CDI.
- **Ativo vivo no mês m:**
  - elegível no NEFIN;
  - negociou em todos os 12 meses de m − 12 a m − 1.
- **Meses com menos de 30 ativos vivos** ficam sem sinal e sem carteira.
- **Sinais do mês m**, calculados com dados até o fim de m − 1. Cada um vira z-score entre os ativos vivos, limitado a ±3:
  1. momentum 12-2: retorno acumulado de m − 12 a m − 2;
  2. reversão: retorno de m − 1;
  3. volatilidade: desvio dos retornos mensais de m − 12 a m − 1.
- **Tendência do mercado:** retorno de 12 meses do equal-weight dos vivos menos o CDI de 12 meses.

## Modelo

- **Parâmetros:** θ = (v ∈ ℝ³, a, b, c).
- **Alvo do mês:** `t = σ(a + b·tendência) · softmax(z·v)` sobre os ativos vivos. O resto fica no CDI.
- **Velocidade:** a carteira anda uma fração σ(c) do caminho entre as posições atuais (depois dos retornos do mês anterior) e o alvo.
- **Teto:** se a soma dos pesos passar de 0,99, todos são reduzidos na mesma proporção. Custo e IR saem do resto.
- **Valores iniciais:** v = 0 (equal-weight dos vivos), a = 2, b = 0, c = 0.

## Treino (walk-forward)

- **Quando e com que dados:** todo janeiro, a partir de 2006-01, o modelo é reajustado com todos os meses de 2001-01 até o mês anterior.
  - A primeira vez, em 2006-01, com 200 passos de Adam (taxa 0,05).
  - Depois, 60 passos a cada janeiro, partindo do θ anterior.
- **Objetivo:** maximizar o log-crescimento médio mensal do patrimônio depois de custos e IR num simulador suave, menos 0,001·‖v‖². O simulador começa com R$ 100 mil e:
  - cobra custo de 25 bps por lado;
  - aplica o IR de 15% sobre o ganho líquido realizado, com custo médio;
  - isenta os meses com vendas até R$ 20 mil (degrau suavizado, escala de R$ 500, só dentro do `grad`);
  - compensa o prejuízo nos meses seguintes.
- **Prova de causalidade:** o verificador temporal da Kairós (M3) prova na compilação que nenhuma decisão usa dados do próprio mês ou posteriores. As duas variantes com vazamento (treinar até m, usar o retorno de m) dão E0301.

## Avaliação na pesquisa (2006-01 a 2026-10)

- **Simulador:** o do laboratório (`b3.evaluate`), com custos reais, IR de pessoa física, R$ 100 mil e 1 pregão de atraso.
- **Comparações, no mesmo simulador e nas mesmas datas:**
  - equal-weight dos ativos vivos, rebalanceado todo mês;
  - momentum 12-2: terço de cima dos vivos, pesos iguais.
- **Registro:** as três avaliações entram no registro de tentativas do laboratório (`results/registro_testes.csv`), junto com todas as anteriores. O DSR do modelo desconta todas.

## Holdout

- **Período:** de 2026-11 em diante. É o futuro: a busca do laboratório já abriu 2019-01..2026-10.
- **Abertura:** uma vez só (`open_once`), quando houver pelo menos **24 meses** (por volta de 2028-11).
- **Cerca:** até lá, nenhum programa Kairós enxerga esses meses.

## Regra de aceitação

O modelo só é aceito se valerem as duas condições:

1. **Pesquisa:** DSR ≥ 90% contando todas as tentativas do registro, **e** CAGR depois de custos e IR maior que o do equal-weight.
2. **Holdout:** alpha CAPM > 0 **e** PSR ≥ 90% (probabilidade de o Sharpe ser positivo).

Se a condição 1 falhar, o resultado da pesquisa já diz que não há evidência de alpha, e o holdout só confirma ou não a direção.

## O que não pode mudar

Nada acima: sinais, universo, datas, modelo, objetivo, passos, taxas, custos e regra.

Qualquer mudança é **outro protocolo**, com outro nome e com holdout começando depois dos meses que já tiverem sido vistos.
