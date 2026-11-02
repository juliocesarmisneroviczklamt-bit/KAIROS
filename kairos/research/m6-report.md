# M6: resultado da pesquisa (`tax-portfolio-v1`)

O protocolo está em [`m6-protocol.md`](m6-protocol.md) (SHA-256 `d44da677…`).

- **Pré-registro:** o protocolo e o programa [`tax_portfolio.krs`](tax_portfolio.krs) foram publicados no commit `52a186d`, antes de qualquer execução nos dados reais.
- **Execução nos dados reais:** uma única, em 2026-11-01, com dados até 2026-10. Levou 5 minutos.
- **Livro-razão:** o protocolo foi registrado com hash `ec112afd…`.

## Veredito: rejeitado pela regra pré-registrada

A condição 1 exigia DSR ≥ 90% **e** CAGR acima do equal-weight. O modelo bateu o equal-weight, mas o DSR ficou em **0,8%**: não há evidência de alpha.

O holdout (2026-11 em diante) continua reservado e será aberto uma vez, com pelo menos 24 meses de dados (por volta de 2028-11). Pelo protocolo, ele só vai confirmar ou não a direção.

## Números da pesquisa

As três avaliações rodaram no simulador realista do laboratório:
- **Período:** 2006-01 a 2026-10 (250 meses).
- **Custos e IR:** custos reais (emolumentos, meio spread, impacto) e IR de pessoa física.
- **Capital e execução:** R$ 100 mil, com 1 pregão de atraso.

| carteira | CAGR depois de custos e IR | Sharpe sobre o CDI | maxDD | alpha CAPM a.a. (t) | giro por ano | DSR |
|---|---|---|---|---|---|---|
| **modelo** | **7,50%** | −0,02 | −44,3% | −1,5% (−0,79) | **0,58** | 0,8% (14 tentativas) |
| equal-weight dos vivos | 5,52% | −0,05 | −54,3% | −2,7% (−1,22) | 1,28 | 0,5% (15) |
| momentum 12-2 | 9,74% | 0,10 | −47,7% | +0,8% (0,32) | 5,04 | 3,5% (16) |

O CDI rendeu **10,36% ao ano** no mesmo período, bruto (cerca de 8,8% depois do IR de renda fixa). Nenhuma das três carteiras de ações o bateu depois de impostos: 2006–2026 foi um período ruim para a bolsa brasileira.

Cada avaliação entrou no registro de tentativas do laboratório (`results/registro_testes.csv`), que passou de 13 para 16 linhas. O DSR de cada uma desconta todas as tentativas feitas até ela.

## O que o modelo aprendeu

Os parâmetros, reajustados todo janeiro com os meses anteriores:

| ano | momentum | reversão (mês anterior) | volatilidade | fração investida | velocidade |
|---|---|---|---|---|---|
| 2006 | +0,96 | +0,19 | −0,26 | 99,8% | 8,7% |
| 2010 | +0,20 | +0,16 | −0,25 | ≈100% | 6,6% |
| 2016 | +0,49 | +0,21 | −0,54 | ≈100% | 7,1% |
| 2021 | +0,95 | +0,74 | +0,08 | ≈100% | 5,0% |
| 2026 | +0,50 | +0,28 | −0,24 | ≈100% | 6,6% |

- **Negociar devagar.** A velocidade ficou entre 5% e 9% do caminho até o alvo por mês, e o giro caiu para 0,58 por ano, contra 5,04 do momentum. É o efeito que só um objetivo com custos e IR dentro do gradiente produz: o modelo prefere carregar posições a pagar custo e imposto.
  - Foi isso que o fez bater o equal-weight (+2,0 pontos por ano, com maxDD menor).
  - Não foi suficiente para bater o CDI.
- **Baixa volatilidade e momentum.** O peso em volatilidade foi negativo até 2018, preferindo ações menos voláteis, e quase zero depois. O momentum foi positivo o tempo todo, mais forte no começo e depois de 2019.
- **Nunca saiu do mercado.** A fração investida ficou em ≈100%: o sinal de tendência não aprendeu a ir para o CDI. Numa década em que o CDI bateu a bolsa, isso custou caro.

## O que este marco mostra

- **A linguagem fez o trabalho que prometeu:**
  - o treino walk-forward foi provado causal na compilação: as variantes com vazamento dão E0301;
  - o gradiente atravessou as regras do IR, e o comportamento de negociar devagar surgiu sozinho;
  - as três avaliações viraram tentativas no registro, com o DSR descontado automaticamente;
  - o holdout ficou reservado no livro-razão antes de existir.
- **O modelo não encontrou alpha.** Pela regra registrada antes, ele é rejeitado.
  - Com 16 tentativas no registro, o melhor Sharpe que o acaso daria é 0,50 ao ano (`quant-b3 registro`), e os três ficaram entre −0,05 e 0,10.
  - Mesmo o momentum, a melhor das três carteiras, fica com DSR de 3,5%.

Isso é o resultado honesto. Ele combina com tudo o que o laboratório já tinha encontrado: nenhum long-only aprovado na B3 depois de custos e impostos.

A próxima melhoria não está na arquitetura do modelo, e sim em informação nova: dados fundamentalistas, eventos, fluxo. Qualquer ideia nova precisa de outro protocolo, com holdout depois dos meses já vistos (hoje, a partir de 2026-11).
