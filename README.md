# KAIROS

**Kairós** é uma linguagem de programação criada do zero para pesquisa e IA no mercado financeiro. *Kairós* (καιρός) é grego para "o momento certo". A garantia central da linguagem é que cada decisão só enxerga o que já era conhecido quando foi tomada.

O repositório tem duas partes que trabalham juntas:

| pasta | o que é |
|---|---|
| [`kairos/`](kairos/) | **A linguagem**: léxico, parser, verificador de tipos com inferência, interpretador, REPL e biblioteca padrão escrita em Kairós. Ver o [README](kairos/README.md) e a [especificação](kairos/docs/spec.md). |
| [`src/`](src/) | **quant-b3**, laboratório em Rust que fornece os dados da B3 e julga as estratégias: simulador realista com custos e IR de pessoa física, registro de testes com Deflated Sharpe e paper trading. Ver a [documentação](docs/quant-b3.md). |

## O que torna a Kairós diferente

1. **Tipos temporais** (M3). Usar dado do futuro numa decisão é erro de compilação.
2. **Holdout de uso único** (M5). Os dados de validação só podem ser abertos uma vez, e cada avaliação entra sozinha no registro de tentativas.
3. **Diferenciação nativa** (M4), inclusive através de regras discretas como IR, isenção e custos, via relaxações declaradas.
4. **Tensores com eixos nomeados** (M2, pronto). Misturar eixos é erro de compilação.

Nenhuma linguagem cria alpha sozinha. A proposta é tornar impossíveis, ou pelo menos visíveis, os erros que fabricam alpha falso.

## Estado: v0.2.0 (M2)

| marco | entrega | estado |
|---|---|---|
| M1 | núcleo: tipos, inferência, `match`, closures, `inout`, interpretador, REPL | ✅ |
| M2 | tensores com eixos nomeados + módulo `b3` sobre os dados reais | ✅ |
| M3 | tipos temporais | próximo |
| M4 | `grad` nativo + motor de IA escrito em Kairós | planejado |
| M5 | holdout de uso único + registro de testes e DSR automáticos | planejado |
| M6 | primeiro modelo inédito (carteira treinada no patrimônio depois de IR) | planejado |
| M7 | back-end nativo (Cranelift) | planejado |

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

## Como usar

O binário para Windows está nas [Releases](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/releases). Para compilar do código (Rust 1.85+):

```text
cargo build --release -p kairos          # target/release/kairos
kairos repl                              # sessão interativa
kairos run kairos/examples/finance.krs   # exemplo do núcleo
cargo test -p kairos                     # testes dourados + teste de aceitação
```

O módulo `b3` precisa dos dados da B3, que o laboratório baixa e monta: são cerca de 1 GB vindos da B3, do BCB e do NEFIN-USP, e as pastas `data/` e `results/` não ficam no repositório.

```text
cargo run --release -p quant-b3 -- all   # baixa e monta data/processed/panel.bin
kairos run kairos/examples/momentum.krs  # da raiz do repositório (ou defina KAIROS_DATA)
```
