# Kairós

Linguagem de programação para pesquisa e IA no mercado financeiro. *Kairós*: "o momento certo". Cada decisão só enxerga o que já era conhecido quando foi tomada.

Especificação: [`docs/spec.md`](docs/spec.md). Versão atual: **0.5.0**, com o M1 (núcleo), o M2 (tensores + dados da B3), o M3 (tipos temporais), o M4 (diferenciação nativa + motor de IA em Kairós) e o M5 (holdout de uso único + registro de tentativas) prontos. Roda em 32 e 64 bits (x86, ARM, RISC-V, PowerPC, s390x) e em Linux, macOS, Windows, FreeBSD, NetBSD, OpenBSD e WebAssembly: é Rust puro, e a matemática dá os mesmos bits em todos eles.

```text
cargo build --release -p kairos              # gera target/release/kairos (kairos.exe no Windows)
kairos run kairos/examples/momentum.krs      # momentum na B3, escrito em Kairós (rode da raiz do repositório)
kairos run kairos/examples/finance.krs       # exemplo do núcleo
kairos run kairos/examples/holdout.krs       # protocolo, holdout de uso único e registro de tentativas
kairos check arquivo.krs                     # só verifica os tipos
kairos ledger                                # o livro-razão de holdouts
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

**M4 (diferenciação nativa + motor de IA)**
- `grad(f)`, `grad(f, wrt: k)` e `value_and_grad(f)`: diferenciação reversa sobre o caminho executado, para escalares, tensores, structs, tuplas e listas. O gradiente confere com diferenças finitas até 10⁻¹⁰.
- Espaço tangente: `p - 0.01 * g` funciona com structs, e funções genéricas como `fn adam_step[P](...)` valem para quaisquer parâmetros.
- `tanh`, `sigmoid`, `relu`, `softplus`, `randn(seed:, ...)` reprodutível, e regras suaves (`smooth_max`, `smooth_step`) que são exatas fora do `grad` e suaves dentro.
- O motor de IA é escrito em Kairós: `Mlp`, `mse`, `sgd_step`, `adam_step` ([`std/prelude.krs`](std/prelude.krs)). Em [`examples/mlp.krs`](examples/mlp.krs) uma rede aprende `y = x³ − x` (perda 0,566 → 0,00017) em 0,14 s.

**M5 (holdout de uso único + registro de tentativas)**
- **`protocol "nome" { research: until AAAAMM  holdout: from AAAAMM  min_months: N  rule: "..." }`** cerca os dados:
  - o programa só enxerga a pesquisa;
  - depois da primeira execução, nenhum programa enxerga o holdout até ele ser aberto.
- **`Holdout` é afim:** `open_once(h, || pesos(), ...)` o consome. Usar de novo, num laço ou numa lambda é E0501.
- **Livro-razão:** `results/holdout_ledger.csv` grava protocolos, meses vistos e aberturas. Abrir de novo é R0502, e abrir antes de `min_months` meses de dados é R0503, sem gravar nada.
- **Holdout novo:** só pode começar depois de tudo o que programas Kairós já viram. O livro-razão nasce com a busca do laboratório (2019-01..2026-10), então o único holdout limpo hoje é o futuro.
- **Registro de tentativas:** toda `b3.evaluate` entra em `results/registro_testes.csv`, e `Evaluation` traz `trials` e `dsr` (Deflated Sharpe Ratio). Comandos `kairos ledger` e `kairos ledger withdraw <nome>`.
- **Testes:** 5 de integração num mercado sintético ([`tests/holdout.rs`](tests/holdout.rs)) e 7 dourados.

A biblioteca padrão é escrita em Kairós: [`std/prelude.krs`](std/prelude.krs), com `variance`, `std_dev`, `compound`, `wealth`, `max_drawdown`, `sharpe`, a struct `Evaluation` e o motor de IA.

## Próximos marcos

| marco | entrega |
|---|---|
| M6 | primeiro modelo inédito |
| M7 | back-end nativo (Cranelift); candidato: pesos ternários `t3` |
