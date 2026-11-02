# KAIROS

[![CI](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/actions/workflows/ci.yml/badge.svg)](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/juliocesarmisneroviczklamt-bit/KAIROS)](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/releases)

**Kairós** é uma linguagem de programação criada do zero para pesquisa e IA no mercado financeiro. *Kairós* (καιρός) é grego para "o momento certo". A garantia central da linguagem é que cada decisão só enxerga o que já era conhecido quando foi tomada.

O repositório tem duas partes que trabalham juntas:

| pasta | o que é |
|---|---|
| [`kairos/`](kairos/) | **A linguagem**: léxico, parser, verificador de tipos com inferência, interpretador, REPL e biblioteca padrão escrita em Kairós. Ver o [README](kairos/README.md) e a [especificação](kairos/docs/spec.md). |
| [`src/`](src/) | **quant-b3**, laboratório em Rust que fornece os dados da B3 e julga as estratégias: simulador realista com custos e IR de pessoa física, registro de testes com Deflated Sharpe e paper trading. Ver a [documentação](docs/quant-b3.md). |

## O que torna a Kairós diferente

1. **Tipos temporais** (M3, pronto). Usar dado do futuro numa decisão é erro de compilação: o compilador sabe quando cada valor fica conhecido.
2. **Holdout de uso único** (M5, pronto). O protocolo declarado no programa esconde o holdout de todos os programas até ele ser aberto, uma única vez. Cada avaliação entra sozinha no registro de tentativas, com Deflated Sharpe.
3. **Diferenciação nativa** (M4, pronto). `grad` é parte da linguagem, inclusive através de regras discretas, via relaxações declaradas (`smooth_max`, `smooth_step`). O motor de IA (MLP, Adam) é escrito em Kairós.
4. **Tensores com eixos nomeados** (M2, pronto). Misturar eixos é erro de compilação.

Nenhuma linguagem cria alpha sozinha. A proposta é tornar impossíveis, ou pelo menos visíveis, os erros que fabricam alpha falso.

## Estado: v0.14.0 (M14)

| marco | entrega | estado |
|---|---|---|
| M1 | núcleo: tipos, inferência, `match`, closures, `inout`, interpretador, REPL | ✅ |
| M2 | tensores com eixos nomeados + módulo `b3` sobre os dados reais | ✅ |
| M3 | tipos temporais: usar dado do futuro numa decisão não compila | ✅ |
| M4 | `grad` nativo + motor de IA escrito em Kairós (MLP, Adam genérico) | ✅ |
| M5 | holdout de uso único + protocolos + registro de testes e DSR automáticos | ✅ |
| M6 | primeiro modelo inédito (carteira treinada no patrimônio depois de IR), pré-registrado | ✅ rejeitado pela própria regra ([relatório](kairos/research/m6-report.md)); holdout reservado até 2028-11 |
| M7 | back-end nativo (Cranelift) + `kairos-bench` | ✅ 1,20× Rust na média geométrica, com os mesmos bits |
| M8 | o laboratório escrito em Kairós: universo, estatística, carteiras e os 5 estudos | ✅ cada estudo imprime o relatório do Rust linha por linha |
| M9 | arquivos, texto, mapas, internet e a biblioteca padrão escrita em Kairós (JSON, CSV, datas) | ✅ registro de tentativas e coleta macro em Kairós, iguais ao Rust |
| M10 | a base de dados montada em Kairós: COTAHIST, eventos da B3, painel | ✅ `bars.bin`, cache de eventos e `panel.bin` iguais aos do Rust, byte a byte |
| M11 | o simulador (custos, impacto, IR) escrito em Kairós | ✅ `b3.evaluate`/`b3.simulate` rodam [`std/sim.krs`](kairos/std/sim.krs), com os mesmos bits do simulador Rust |
| M12 | auto-hospedagem: o compilador da Kairós escrito em Kairós | ✅ o front-end inteiro em [`kairos/compiler/`](kairos/compiler), igual ao Rust nó a nó em todos os arquivos; compila a si mesmo (a imagem é um ponto fixo) e seus programas rodam com `kairos exec` |
| M13 | o back-end em Kairós: o compilador escrito em Kairós gera código (WebAssembly) | ✅ [`kw.krs`](kairos/compiler/kw.krs) gera módulos WASI com a saída e os erros do interpretador; o compilador inteiro vira WebAssembly e, assim, compila a si mesmo no mesmo módulo, byte a byte; o runtime dos módulos é escrito em Kairós ([`rt.krs`](kairos/compiler/rt.krs)): contagem de referências, floats escritos e lidos como no Rust, a `libm`, Unicode e arquivos |
| M14 | o back-end WebAssembly completo: closures, tensores, a ponte `b3`, os holdouts e o `grad` | ✅ o `momentum`, o modelo do M6 e o MLP compilados para WebAssembly dão os mesmos números e gravam o mesmo registro de testes e o mesmo livro-razão que o interpretador; a ponte `b3` ([`rt_b3.krs`](kairos/compiler/rt_b3.krs)) e a fita do `grad` ([`rt_diff.krs`](kairos/compiler/rt_diff.krs)) são escritas em Kairós |
| M15 | o interpretador escrito em Kairós | em andamento: [`interp.krs`](kairos/compiler/interp.krs) roda o HIR do front-end em Kairós com a saída e os erros do interpretador Rust; compilado para WebAssembly, roda programas Kairós sem nenhum Rust |
| — | paper trading em Kairós | aguarda decisão: os dados de nov/2026 em diante são o holdout reservado do M6 |

## Exemplo: momentum na B3

O laço central de [`kairos/examples/momentum.krs`](kairos/examples/momentum.krs):

```krs
for m in max(12, b3.month_index(200101))..n {
    let alive = elig[month: m] and not is_nan(ret[month: m - 1])
    let signal = prod(1.0 + fill_nan(ret[month: m - 12 ..= m - 2], 0.0), over: month) - 1.0
    let s = where(alive, signal, nan)
    let cnt = int(nan_count(s, over: asset))
    if cnt < 15 { continue }
    let k = int(round(float(cnt) / 3.0))
    weights[month: m] = where(top_k(s, k, over: asset), 1.0 / float(k), 0.0)
}
let ev = b3.evaluate(weights, cost_bps: 25.0, lag: 0)
```

O programa roda em ~0,3 s sobre 31 anos de B3 e reproduz **bit a bit** os números da mesma carteira calculada em Rust pelo laboratório. Quem confere é [`kairos/tests/b3_bridge.rs`](kairos/tests/b3_bridge.rs).

| métrica (2001–2026, 25 bps por lado) | Kairós = Rust |
|---|---|
| CAGR | 14,63% a.a. |
| Sharpe sobre o CDI | 0,22 |
| maxDD | −48,6% |
| alpha CAPM | +2,3% a.a. (t = 1,14, não significativo) |

Trocando `m - 2` por `m`, o programa passa a usar o mês que ainda não terminou, e **não compila**:

```text
error[E0301]: use of future information
16 |     let signal = prod(1.0 + fill_nan(ret[month: m - 12 ..= m], 0.0), over: month) - 1.0
   |                                      ^^^^^^^^^^^^^^^^^^^^^^^^ known at end(m), but the portfolio of month m is decided at end(m - 1)
   = hint: `b3.evaluate` trades the weights of month m at the close of month m - 1: use data up to `m - 1` (momentum skips the last month: `m - 2`)
```

## Holdout de uso único

O laboratório já gastou o holdout de 2019 em diante numa busca de alpha. Por isso, um protocolo novo só pode reservar o futuro ([`kairos/examples/holdout.krs`](kairos/examples/holdout.krs)):

```krs
protocol "momentum-demo" {
    research: until 202610
    holdout: from 202611
    min_months: 12
    rule: "accept if the holdout's CAPM alpha > 0 and its PSR >= 90%"
}
let research = b3.evaluate(weights(), cost_bps: 25.0, lag: 0)    // só enxerga até 2026-10
let h = b3.holdout("momentum-demo")
let final = open_once(h, || weights(), cost_bps: 25.0, lag: 0)   // o holdout só existe aqui dentro
```

```text
research: 310 months, Sharpe 0.22142871787716945
trials in the registry: 14 -> DSR 0.0919859189145863
runtime error[R0503]: the holdout of protocol "momentum-demo" starts in 2026-11, but the data ends in 2026-10: there is nothing to open yet
   = hint: nothing was opened or recorded: run again when the data has more months
```

- **O momentum é só mais uma tentativa:** com as 14 tentativas que o registro do laboratório tinha na v0.5.0, a chance de o Sharpe dele ser real (DSR) é de 9%.
- **O holdout reservado** fica escondido de todos os programas até haver 12 meses de dados novos. Aí ele abre, uma vez, e o resultado fica no livro-razão (`kairos ledger`).
- **Abrir de novo** é erro na compilação (E0501) e entre execuções (R0502).
- **Um holdout no passado** é recusado (R0502): aqueles meses já foram vistos pela busca do laboratório.

## Primeiro modelo: carteira treinada no patrimônio depois de IR (M6)

[`kairos/research/tax_portfolio.krs`](kairos/research/tax_portfolio.krs), pré-registrado em [`m6-protocol.md`](kairos/research/m6-protocol.md) antes de tocar nos dados reais:

- **O que o modelo é:** uma carteira long-only cujos parâmetros são treinados por `grad` no patrimônio **depois de custos e do IR de pessoa física**. A isenção de R$ 20 mil de vendas por mês entra como degrau suave.
- **Treino:** walk-forward, refeito todo janeiro. A análise temporal prova na compilação que nenhuma decisão usa dados do próprio mês.
- **Avaliação:** pelo simulador realista, contra o equal-weight e o momentum, de 2006 a 2026, com R$ 100 mil.

| carteira | CAGR depois de custos e IR | giro por ano | DSR (16 tentativas) |
|---|---|---|---|
| modelo | 7,50% | 0,58 | 0,8% |
| equal-weight | 5,52% | 1,28 | 0,5% |
| momentum | 9,74% | 5,04 | 3,5% |
| CDI (bruto) | 10,36% | | |

**O que o resultado mostra:**
- **Comportamento aprendido:** o modelo passou a negociar devagar, com 9 vezes menos giro que o momentum, e assim bateu o equal-weight.
- **Sem alpha:** pela regra registrada antes, ele é **rejeitado** (DSR de 0,8%), e nenhuma carteira de ações bateu o CDI depois de impostos.
- **Holdout:** o futuro, a partir de 2026-11, fica reservado e abre uma vez com 24 meses de dados.

## Código nativo (M7)

Funções escalares puras (`i64`, `f64`, `bool`, laços, `if`, recursão, `sqrt`/`exp`/`ln`…) viram código de máquina com o Cranelift na primeira chamada. Elas dão os mesmos bits e os mesmos erros do interpretador. Medido com o `kairos-bench`:

| benchmark | o que mede | interpretador | nativo (JIT) | Rust | nativo ÷ Rust |
|---|---|---|---|---|---|
| `fib(30)` | chamadas recursivas | 315 ms | 3,8 ms | 3,1 ms | 1,25× |
| `sum_sqrt` (20 mi) | laço de f64 com `sqrt` | 1,95 s | 24,2 ms | 24,1 ms | 1,00× |
| `collatz` (100 mil) | inteiros verificados com desvios | 1,62 s | 21,5 ms | 14,4 ms | 1,50× |
| `black_scholes` (300 mil) | finanças: `exp`, `ln`, `sqrt`, CDF normal | 647 ms | 11,0 ms | 9,1 ms | 1,20× |
| `mandelbrot` (300×200) | laços aninhados de f64 com saída antecipada | 950 ms | 6,3 ms | 5,3 ms | 1,19× |
| `monte_carlo` (2 mi) | gerador Park-Miller + f64 | 747 ms | 16,9 ms | 15,0 ms | 1,13× |
| **média geométrica** | | | | | **1,20×** |

```text
cargo run --release -p kairos --bin kairos-bench
```

O JIT funciona em x86_64 (Linux, macOS, Windows, FreeBSD) e ARM64 (Linux, macOS). Nas outras plataformas tudo roda no interpretador, com os mesmos resultados.

## O laboratório em Kairós (M8)

A pesquisa do laboratório `quant-b3` é escrita em Kairós, em [`kairos/lab/`](kairos/lab). O universo, a estatística, as carteiras e as decisões ficam em `.krs`. O Rust guarda os dados, o simulador e o registro de tentativas. Cada estudo imprime o relatório do Rust linha por linha:

```text
kairos run kairos/lab/momentum.krs     # replicação do NEFIN e momentum (= quant-b3 momentum)
kairos run kairos/lab/risk.krs         # controle de risco (= momentum-risco)
kairos run kairos/lab/realistic.krs    # custo real, atraso e IR (= realista)
kairos run kairos/lab/smallcaps.krs    # small caps, hipóteses registradas (= small-caps)
kairos run kairos/lab/search.krs       # busca de alpha do protocolo registrado (= busca)
```

O universo líquido do NEFIN, em [`universe.krs`](kairos/lab/universe.krs), dá exatamente o do Rust: 41.106 pares ativo-mês, sem nenhuma diferença. O compilador também prova que ele só usa o passado. As estatísticas de [`stats.krs`](kairos/lab/stats.krs), MQO com Newey-West incluído, são bit a bit iguais às do Rust. Rodar os estudos não infla o registro: uma estratégia conta como uma tentativa, por mais vezes que seja executada.

## A base de dados em Kairós (M10)

Os dados do laboratório (cotações da B3 desde 1995, eventos corporativos, CDI, IPCA e fatores do NEFIN) são baixados e montados por programas Kairós, e cada etapa grava os mesmos bytes que o Rust gravava:

```text
kairos run kairos/lab/fetch_cotahist.krs   # ZIPs anuais da B3
kairos run kairos/lab/ingest.krs           # COTAHIST -> data/processed/bars.bin
kairos run kairos/lab/fetch_events.krs     # proventos, desdobramentos, cisões... (API da B3)
kairos run kairos/lab/fetch_macro.krs      # CDI, IPCA, NEFIN
kairos run kairos/lab/build.krs            # retorno total diário -> data/processed/panel.bin
kairos run kairos/lab/update.krs           # tudo isso, incremental, como a atualização diária
```

## Como usar

**Binários prontos** nas [Releases](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/releases), sem dependências:

| sistema | arquiteturas |
|---|---|
| Linux (qualquer distribuição) | x86_64, i686 (32 bits), ARM64, ARMv7 e ARMv6 (32 bits, Raspberry Pi), RISC-V 64, PowerPC64 LE, s390x |
| macOS | Intel e Apple Silicon |
| Windows | x86_64, i686 (32 bits), ARM64 |
| FreeBSD, NetBSD, OpenBSD | x86_64 |
| WebAssembly (WASI) | qualquer sistema com runtime WASI (wasmtime, wasmer…) |

**Em qualquer outro sistema** que o Rust suporta (illumos, DragonFly, outras arquiteturas…), compile a partir do código. A Kairós é Rust puro: nenhuma biblioteca do sistema é necessária.

```text
cargo build --release -p kairos          # target/release/kairos
kairos repl                              # sessão interativa
kairos run kairos/examples/finance.krs   # exemplo do núcleo
cargo test -p kairos                     # testes dourados + teste de aceitação
```

**Mesmo resultado em qualquer lugar:** `exp`, `ln` e `^` usam matemática em Rust puro, não a do sistema operacional. O mesmo programa dá os mesmos bits em todos os sistemas. A CI roda os testes a cada push em 11 combinações de sistema e arquitetura. Nativas: Linux x86_64 e ARM64, macOS, Windows x86_64 e i686, FreeBSD. Emuladas: Linux i686 e ARMv7, RISC-V 64, PowerPC64 LE e s390x (big-endian). E também WebAssembly.

**Arquivos `.krs`:**
- **VS Code:** instale a extensão `kairos-vscode-*.vsix` das Releases (`code --install-extension arquivo.vsix`). Ela traz reconhecimento, cores de sintaxe e blocos ` ```krs ` no Markdown.
- **Outros editores:** veja [`editors/`](editors/). O mesmo diretório explica por que o GitHub ainda mostra "Rust".
- **Scripts:** com `#!/usr/bin/env kairos` na primeira linha e `chmod +x`, o programa roda direto: `./programa.krs`.

O módulo `b3` precisa dos dados da B3, que o laboratório baixa e monta: são cerca de 1 GB vindos da B3, do BCB e do NEFIN-USP, e as pastas `data/` e `results/` não ficam no repositório.

```text
cargo run --release -p quant-b3 -- all   # baixa e monta data/processed/panel.bin
kairos run kairos/examples/momentum.krs  # da raiz do repositório (ou defina KAIROS_DATA)
```

## Licença e segurança

[MIT](LICENSE): use, modifique e distribua à vontade, mantendo o aviso de copyright. Para reportar uma falha de segurança, veja [SECURITY.md](SECURITY.md).
