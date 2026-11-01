# quant-b3

Laboratório em Rust pra procurar alpha na B3 **sem se enganar**.

A regra do projeto: antes de acreditar numa ideia nova, o motor precisa reproduzir um efeito que já é conhecido. O primeiro teste é replicar o fator de momentum (WML) do NEFIN-USP.

## Pipeline

```
quant-b3 fetch-cotahist   # COTAHIST da B3, 1995..hoje (todas as ações, inclusive as que saíram da bolsa)
quant-b3 ingest           # parse da largura fixa -> data/processed/bars.bin
quant-b3 fetch-events     # proventos, desdobramentos, grupamentos, cisões e subscrições (API da B3, com cache)
quant-b3 fetch-macro      # CDI diário (BCB/SGS 12) + fatores NEFIN
quant-b3 build            # retorno total diário ajustado -> data/processed/panel.bin
quant-b3 momentum         # replica o WML do NEFIN + long-only com custos + regressão de fatores
quant-b3 momentum-risco   # estudo A: momentum com controle de volatilidade
quant-b3 realista         # estudo B: spread real, impacto, atraso de execução e IR de pessoa física
quant-b3 small-caps       # estudo C: segmento abaixo do filtro de liquidez, com holdout e capacidade
quant-b3 registro         # todos os testes feitos, descontados pelo número de tentativas (DSR)
quant-b3 busca            # busca de alpha seguindo results/protocolo_busca.md (registrado antes de rodar)
quant-b3 paper iniciar    # congela as regras do paper trading e mostra as ordens do 1º mês
quant-b3 paper rodar      # baixa dados novos (B3/BCB), atualiza as carteiras simuladas (paper.cmd faz isso)
quant-b3 paper status     # situação do paper trading sem baixar nada
quant-b3 inspect PETR4 --from 20080420 --to 20080430   # audita um papel
quant-b3 all              # tudo em sequência
```

Binário: `target\release\quant-b3.exe` (compile com `cargo build --release`).

## Fontes

| dado | fonte | observação |
|---|---|---|
| preços | B3, séries históricas COTAHIST | sem ajuste; inclui empresas deslistadas, o que evita viés de sobrevivência |
| eventos | B3, `listedCompaniesProxy` (GetListedSupplementCompany / GetListedCashDividends) | inclui empresas canceladas (Ambev antiga, Eletropaulo, Telemar...) |
| CDI | BCB SGS série 12 | taxa diária |
| fatores | NEFIN-USP | Mkt, SMB, HML, WML, IML; versões EW e VW, diário e mensal |

## Como o retorno total é calculado

Para cada par de pregões consecutivos em que o papel negociou:

```
r = P_t * M / P_{t-1} - 1 + Y
```

- `M` multiplica a quantidade de ações por desdobramento, bonificação ou grupamento, com data-com em `[t-1, t)`.
- `Y` é a soma, como fração do preço, de: proventos em dinheiro (valor dividido pelo fechamento na data-com informado pela B3), cisões (ações recebidas, precificadas no primeiro pregão da empresa nova) e direitos de subscrição (valor teórico).
- A lista de eventos da B3 é **incompleta** (ex.: falta o grupamento de 2017 da Gafisa). Por isso, um salto de preço sem evento é tratado como desdobramento/grupamento só se:
  - aparecer um marcador **novo** `EB`/`EG` no ESPECI naquele pregão. Com razão redonda, usa a razão; com razão quebrada, neutraliza o dia; ou
  - a razão for enorme e exata (≥10×, erro < 3%).
- **Volume não serve de prova.** Depois de um crash o mercado negocia os mesmos R$ a preço menor, então a quantidade sobe como num desdobramento. A primeira versão confundiu a fraude da Americanas (AMER3, −77% em 12/01/2023) com um desdobramento 1:4.
- Evento da B3 que não aparece no preço é descartado. Fator da B3 incompatível com o salto observado é corrigido pela razão redonda.
- Filtro final: dia com >+200% ou <−90% sem explicação é neutralizado (fica só o provento).
- Tudo isso fica registrado em `results/diag/` (`splits_inferidos.csv`, `movimentos_extremos.csv`, `residuos_eventos.csv`, `extremos_no_universo.csv`).

Trocas de código (VVAR3 → VIIA3 etc.) são encadeadas por ISIN.

## Regras anti-autoengano

1. **Replicar antes de inventar.** Se o motor não reproduz um fator conhecido, ele tem bug.
2. **Sem viés de sobrevivência.** Empresas que morreram ficam na base.
3. **Sem look-ahead.** O universo do ano t usa só dados de t-1. O sinal do mês t usa só dados até t-1.
4. **Custos sempre.** Giro × custo por lado. Em small cap, slippage come o edge.
5. **Holdout.** Para ideias novas, pesquise em 2001–2018 e olhe 2019–hoje uma vez só, no fim.
6. **Conte quantas ideias testou.** Mil testes geram ~50 "significativos" a 5% por acaso.
7. **Alpha é o que sobra depois dos fatores.** Regressão contra Mkt/SMB/HML/IML/WML do NEFIN com erros de Newey-West.

## Teste 1: momentum (rodado em 2026-11-01)

Universo estilo NEFIN: classe mais negociada, >80% dos pregões com mais de R$ 500 mil, listada antes de dezembro de t−1. Terciles por retorno de t−12 a t−2.

**Validação do motor** (contra o WML do NEFIN, equal-weighted): correlação mensal 0,936 e diária 0,910. Mercado EW contra mercado NEFIN: 0,929.

| WML long-short, sem custo | nosso | NEFIN |
|---|---|---|
| 2001–2010 | +10,0% a.a. (t=1,26) | +8,9% (t=1,21) |
| 2011–2020 | +15,0% (t=2,39) | +17,8% (t=2,76) |
| 2021–hoje | +19,3% (t=2,98) | +17,1% (t=2,83) |
| 2001–hoje | +14,0% (t=3,29) | +14,2% (t=3,47) |

Momentum não é explicado por mercado, tamanho, valor e iliquidez: alpha de +14,1% a.a., t=4,27.

**Long-only nos vencedores** (2001–2026, pesos iguais, rebalanceamento mensal, giro de 43%/mês, **antes de IR**):

| custo por lado | CAGR | vol | maxDD | excesso sobre o universo EW |
|---|---|---|---|---|
| 0 bps | 16,6% | 22,6% | −48% | +5,9% a.a. (t=2,97) |
| 25 bps | 15,1% | 22,6% | −49% | +4,6% (t=2,32) |
| 50 bps | 13,6% | 22,6% | −49% | +3,4% (t=1,67) |
| 100 bps | 10,8% | 22,6% | −51% | +0,8% (t=0,39) |

No mesmo período: CDI 12,0%, mercado NEFIN 13,1%, universo EW 9,5%.

Com 25 bps, o alpha de 4 fatores é +2,4% a.a. (t=1,31), **não significativo**. Com o WML na regressão vira −3,2% (beta WML 0,39, t=12). É o prêmio do fator momentum, que já é conhecido, não alpha novo. O custo de execução decide se ele chega no bolso.

## Fase 2 (rodada em 2026-11-01)

Os parâmetros de cada estudo foram fixados no código antes da primeira rodada (constantes no topo de `risk.rs`, `realistic.rs` e `smallcaps.rs`).

### A. Momentum com controle de risco (`momentum-risco`)

O WML é escalado por 12% / vol dos últimos 126 pregões, com alavancagem máxima de 2x (Barroso & Santa-Clara 2015).

| 2001–2026 | Sharpe | maxDD | pior mês | assimetria |
|---|---|---|---|---|
| WML original | 0,70 | −54% | −33% | −1,31 |
| WML escalado | **0,98** (t=4,6) | **−34%** | **−18%** | −0,77 |

Melhora nos dois subperíodos.

No **long-only**, o controle (exposição = min(1, 15%/vol), resto no CDI) corta a queda máxima de −49% pra −30%, mas **piora o Sharpe** sobre o CDI: 0,22 → 0,15.

### B. Execução realista + IR (`realista`)

Custo real = emolumentos (0,03%) + meio spread real do fechamento + impacto 0,5·σ·√(ordem/volume). No universo líquido isso dá ~25 bps por lado. Um pregão de atraso custa ~0,2 ponto ao ano.

IR com as regras de hoje (15%, isenção até R$ 20 mil/mês em vendas, compensação de prejuízo, JCP 15% retido). CAGR 2001–2026 depois de custos e IR, com o capital em reais de hoje:

| capital | momentum long-only | ETF de índice (15% no resgate) | CDI (15% no resgate) |
|---|---|---|---|
| R$ 50 mil | 13,6% | 12,4% | 11,3% |
| R$ 300 mil | 12,6% | 12,4% | 11,3% |
| R$ 3 milhões | 12,0% | 12,4% | 11,3% |

Com capital pequeno, a isenção de R$ 20 mil é o que faz a diferença. Com capital médio, momentum empata com o ETF depois do IR. Com capital grande, perde por causa do impacto e de não ter isenção.

### C. Small caps (`small-caps`)

Segmento: fora do filtro NEFIN, negociada em >80% dos pregões, volume mediano ≥ R$ 50 mil, preço ≥ R$ 1. Hipóteses registradas antes de rodar; pesquisa 2001–2018, holdout 2019–hoje.

- O custo real nas small caps é de **1,1% a 1,3% por lado**, contra 0,23% no universo líquido.
- H1 (momentum small > momentum líquido): −1,5% a.a. (t=−0,4) na pesquisa. **Não passou.**
- H2 (prêmio de iliquidez): +1,4% a.a. (t=0,5) na pesquisa, −1,4% no holdout. **Não passou.**

### Registro de testes (`registro`)

6 testes. O melhor de 6 sem habilidade daria Sharpe ~0,50 por acaso.

| teste | DSR |
|---|---|
| WML escalado | **98%** |
| WML long-short | **82%** |
| todas as versões long-only e de small caps | < 10% |

**O que sobrevive é o long-short, que exige vender a descoberto (aluguel de ações). Isso ainda não foi modelado.**

## Busca de alpha (2026-11-01)

O protocolo foi registrado **antes** de rodar: `results/protocolo_busca.md`, SHA-256 `e83eb427…`, conferido pelo próprio programa. São 7 candidatos long-only da literatura, com custo real e 1 pregão de atraso. A medida de alpha é o CAPM contra o mercado NEFIN.

| | MOM | RESMOM | MAX52 | LOWVOL | DY | MOM+LOWVOL | TENDÊNCIA |
|---|---|---|---|---|---|---|---|
| alpha 2001–18 (t) | 0,24 | 0,75 | −0,05 | 0,04 | 0,19 | 0,51 | 1,25 |
| IR 2001–18 | 0,05 | 0,16 | −0,01 | 0,01 | 0,04 | 0,13 | 0,28 |
| IR holdout 2019–26 | 0,27 | 0,09 | 0,09 | 0,49 | 0,30 | 0,51 | −0,72 |

**Nenhum candidato teve alpha com t > 2 na pesquisa, então nenhum alpha foi encontrado.**

- O de maior IR (TENDÊNCIA) caiu no holdout.
- Escolher agora quem foi bem no holdout (MOM+LOWVOL, LOWVOL) seria vício de seleção: o holdout já foi usado.

## Paper trading (desde 2026-11)

Como a busca não validou nada, o paper trading virou o teste fora da amostra que sobrou. Ele acompanha **os 7 candidatos ao mesmo tempo**, com regras congeladas em `results/paper/protocolo_paper.md` (SHA-256 `54360ca0…`):

- R$ 100 mil por estratégia;
- ordens no fechamento do 1º pregão do mês, com custo real e IR de pessoa física;
- caixa no CDI;
- TENDÊNCIA usa o BOVA11.

A cada pregão, um script local agendado (`paper.cmd`, fora do repositório porque tem o caminho da máquina) roda `quant-b3 paper rodar`. Ele baixa os dados da B3 e do BCB, recalcula tudo com o mesmo simulador do backtest e **só acrescenta** ao registro. Se um pregão já registrado mudar no recálculo, isso gera aviso.

Arquivos em `results/paper/`: `*_diario.csv`, `*_operacoes.csv`, `*_ir.csv`, `status.txt`, `log.txt` e `ordens_AAAAMM.txt`. Nenhuma estratégia é escolhida antes de 24 meses.

As pastas `data/` (dados baixados, ~1 GB) e `results/` (relatórios e registros) são geradas localmente e não vão para o repositório.

## Linguagem Kairós (M1 e M2 prontos)

O workspace tem um segundo crate, `kairos/`. É uma linguagem de programação própria para pesquisa e IA no mercado financeiro, que usa este laboratório como fonte de dados e como juiz. Ver `kairos/README.md` e a spec em `kairos/docs/spec.md`:
- tipos temporais: usar dado do futuro é erro de compilação (M3);
- holdout de uso único (M5);
- diferenciação nativa através de IR e custo (M4);
- tensores com eixos nomeados (pronto, M2);
- alto e baixo nível na mesma linguagem.

O módulo `b3` da linguagem lê `data/processed/panel.bin` e chama o simulador deste laboratório. O momentum escrito em Kairós (`kairos/examples/momentum.krs`) reproduz bit a bit o long-only do estudo A (`kairos/tests/b3_bridge.rs`).

## Limitações conhecidas

- Long-short assume venda a descoberto sem custo. Na prática há taxa de aluguel (BTC), margem e perdedoras difíceis de alugar.
- IR: as regras atuais foram aplicadas ao período todo (antes de 2005 a alíquota era outra). Não modela a retenção de 10% sobre dividendos acima de R$ 50 mil/mês por empresa, que vale desde 2026.
- O "ETF de índice" é o mercado value-weighted do NEFIN, sem taxa de administração.

- Proventos antigos (anos 90) podem estar incompletos na API da B3.
- Cisões só são precificadas quando a empresa nova negocia na B3 como ação. Se vier como BDR (ex.: Éxito/GPA 2023), cai no diagnóstico.
- Imposto de renda não é modelado: 15% sobre ganho em swing trade, isenção de R$ 20 mil/mês em vendas.
- Retornos de deslistagem: depois do último pregão o papel rende zero. Não há dado do valor efetivamente recebido em OPA/incorporação.

## Toolchain (esta máquina)

O Rust usa o target `x86_64-pc-windows-gnu`, sem MSVC. A configuração global fica em `%USERPROFILE%\.cargo\config.toml`:
- linker embutido do Rust (`link-self-contained=yes`);
- `dlltool` do WinLibs/MinGW-w64, instalado via winget.
