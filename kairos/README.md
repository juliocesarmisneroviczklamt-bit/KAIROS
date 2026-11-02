# Kairós

Linguagem de programação para pesquisa e IA no mercado financeiro. *Kairós*: "o momento certo". Cada decisão só enxerga o que já era conhecido quando foi tomada.

Especificação: [`docs/spec.md`](docs/spec.md). Versão atual: **0.3.0**, com o M1 (núcleo), o M2 (tensores com eixos nomeados + dados da B3) e o M3 (tipos temporais) prontos. Roda em 32 e 64 bits (x86, ARM, RISC-V, PowerPC, s390x) e em Linux, macOS, Windows, FreeBSD, NetBSD, OpenBSD e WebAssembly: é Rust puro, e a matemática dá os mesmos bits em todos eles.

```text
cargo build --release -p kairos              # gera target/release/kairos (kairos.exe no Windows)
kairos run kairos/examples/momentum.krs      # momentum na B3, escrito em Kairós (rode da raiz do repositório)
kairos run kairos/examples/finance.krs       # exemplo do núcleo
kairos check arquivo.krs                     # só verifica os tipos
kairos repl                                  # sessão interativa (ou só `kairos`)
cargo test -p kairos                         # testes dourados + teste de aceitação com os dados reais
```

## O que já funciona

**M1 (núcleo)**
- Tipos `i64`, `f64`, `bool`, `str`, `List[T]`, tuplas, `struct`, `enum` com dados e funções como valores.
- Inferência de tipos.
- `match` exaustivo.
- `inout` (mudança no lugar sem apelidamento) e chamada uniforme (`x.f(y)` = `f(x, y)`).
- Semântica de valor com cópia sob escrita.
- Lambdas que capturam por valor.
- Aritmética inteira verificada: estouro e divisão por zero são erros.
- `i64` e `f64` nunca se misturam em silêncio.
- Mensagens de erro com código, trecho do código-fonte e dica.
- Ponto e vírgula opcional.

**M2 (tensores + B3)**
- `Tensor[month, asset; f64]`, sempre indexado pelo nome: `ret[month: m]`, `ret[month: m - 12 ..= m - 2]`.
- Operações elemento a elemento alinhadas pelo nome do eixo.
- Reduções com `over: eixo`.
- NaN como dado faltante (`nan_count`, `fill_nan`).
- Ranking (`top_k`, `rank`) e contração (`contract`, a multiplicação de matrizes).
- Eixo errado é erro de **compilação** (E0200–E0202).
- Módulo `b3`, que lê os mesmos dados do laboratório e é julgado pelo mesmo simulador:
  - `b3.monthly_returns()`, `b3.eligible()`, `b3.cdi()`;
  - `b3.evaluate(pesos, cost_bps:, lag:, ...)`.

**Teste de aceitação do M2:** [`examples/momentum.krs`](examples/momentum.krs) tem 13 linhas de estratégia e roda em ~0,3 s sobre 31 anos de B3. Ele reproduz **bit a bit** os 9 números da mesma carteira calculada em Rust pelo `quant-b3`; quem confere é [`tests/b3_bridge.rs`](tests/b3_bridge.rs).

| | Kairós = Rust |
|---|---|
| CAGR | 14,63% a.a. |
| Sharpe sobre o CDI | 0,22 |
| maxDD | −48,6% |
| alpha CAPM | +2,3% a.a. (t = 1,14) |

**M3 (tipos temporais)**
- **Usar informação do futuro numa decisão é erro de compilação.** O compilador sabe quando cada valor fica conhecido:
  - o retorno do mês t, em `end(t)`;
  - o universo NEFIN do mês t, em `end(t − 1)`;
  - a média de todo o histórico, só no fim dos dados.
- **Onde ele confere:**
  - `b3.evaluate`, que negocia os pesos do mês t no fechamento de t − 1;
  - funções anotadas: `fn sinal(r: Tensor[month; f64], m: i64) -> f64 @ end(m - 1) { ... }`.
- **O que a análise acompanha:**
  - fluxo de controle (`if`, `continue`);
  - valores que passam de uma iteração do laço para a próxima;
  - dados próprios marcados com `observed(x, lag: k)`.
- **Até isto não compila:** escolher a estratégia pelo resultado do backtest e avaliar a escolha no mesmo histórico.
- **Testes:** 14 casos de vazamento em [`tests/temporal.rs`](tests/temporal.rs). A versão do momentum com `m - 12 ..= m` dá E0301.

A biblioteca padrão é escrita em Kairós: [`std/prelude.krs`](std/prelude.krs), com `variance`, `std_dev`, `compound`, `wealth`, `max_drawdown`, `sharpe` e a struct `Evaluation`.

## Próximos marcos

| marco | entrega |
|---|---|
| M4 | `grad` nativo + `smooth` + motor de IA em Kairós |
| M5 | holdout de uso único + registro de testes |
| M6 | primeiro modelo inédito |
| M7 | back-end nativo (Cranelift); candidato: pesos ternários `t3` |
