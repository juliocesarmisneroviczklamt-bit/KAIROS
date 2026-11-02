# Changelog

Versões da linguagem Kairós (pasta `kairos/`). Cada versão vira uma release no GitHub, gerada automaticamente por `.github/workflows/release.yml` a partir da seção correspondente deste arquivo.

## 0.4.0 — M4: diferenciação nativa e o motor de IA em Kairós

**`grad` faz parte da linguagem, e uma rede neural escrita em Kairós aprende.**

```krs
var p = mlp_init(7, 1, 16, 1)
var m = zeros_like(p)
var v = zeros_like(p)
for t in 1..=1500 {
    adam_step(inout p, grad(loss)(p, x, y), inout m, inout v, t, 0.02)
}
```

- **`grad(f)`, `grad(f, wrt: k)` e `value_and_grad(f)`:** diferenciação reversa sobre o caminho executado, com uma fita, para escalares, tensores, structs, tuplas e listas.
  - Cobre aritmética com broadcasting por eixo, `exp`, `ln`, `sqrt`, ativações, reduções, seleção e atribuição em fatias, `where` e `contract`.
  - Confere com diferenças finitas até 10⁻¹⁰.
- **Espaço tangente:** `p - 0.01 * g` funciona com `p` sendo uma struct; `sqrt`, `zeros_like` e as ativações também.
- **Funções genéricas:** `fn adam_step[P](...)` vale para quaisquer parâmetros diferenciáveis.
- **Funções novas:**
  - `tanh`, `sigmoid`, `relu`, `softplus`;
  - `randn(seed:, eixos...)`, reprodutível em qualquer sistema;
  - `smooth_max`, `smooth_min` e `smooth_step`: regras exatas fora do `grad`, suaves dentro.
- **Motor de IA escrito em Kairós** (`std/prelude.krs`):
  - `Mlp`, `mlp_init` e `mlp`;
  - `mse`;
  - `sgd_step` e `adam_step`.
- **Aceitação:**
  - `tests/programs/mlp_learns.krs`: o MLP aprende `y = x³ − x`, com perda de 0,566 para 0,00017;
  - o valor final da perda é conferido até o último bit, igual em todos os sistemas e arquiteturas da CI.
- **Novos erros:**

  | código | situação |
  |---|---|
  | E0401 | a função diferenciada não devolve `f64` |
  | E0402 | o tipo não é diferenciável |
  | R0401 | `grad` dentro de `grad` |

- **Correção:** em `f(inout p, g(p))`, os argumentos por valor agora são avaliados antes de `p` ser movido para a chamada.
- **Exemplo:** `examples/mlp.krs`.

## 0.3.0 — M3: tipos temporais

**Usar informação do futuro numa decisão é erro de compilação.** É a garantia nº 1 da linguagem: a causa mais comum de alpha falso deixa de compilar.

```text
error[E0301]: use of future information
16 |     let signal = prod(1.0 + fill_nan(ret[month: m - 12 ..= m], 0.0), over: month) - 1.0
   |                                      ^^^^^^^^^^^^^^^^^^^^^^^^ known at end(m), but the portfolio of month m is decided at end(m - 1)
   = hint: `b3.evaluate` trades the weights of month m at the close of month m - 1: use data up to `m - 1` (momentum skips the last month: `m - 2`)
```

- **Cada valor tem um tempo de conhecimento**, calculado pelo compilador (`src/temporal.rs`, análise sobre o HIR tipado):
  - os dados da B3 trazem o próprio tempo: retorno e CDI do mês t são conhecidos em `end(t)`, e o universo NEFIN do mês t em `end(t − 1)`;
  - operações levam o mais tardio dos operandos;
  - reduzir ou ranquear ao longo do tempo usa o histórico inteiro.
- **Fluxo de controle conta:** atribuições dentro de um `if`, e tudo o que vem depois de um `continue`/`break`/`return`, dependem da condição.
- **Laços até um ponto fixo:** o que uma iteração deixa para a próxima fica um mês mais velho.
- **Decisões checadas:**
  - `b3.evaluate`: os pesos do mês t só podem usar o que se sabe em `end(t − 1)`;
  - funções anotadas com `-> T @ end(m - 1)`.
- **`observed(x, lag: k)`:** declara quando cada linha de dados próprios é publicada.
- **Erros:**

  | código | situação |
  |---|---|
  | E0301 | uso do futuro |
  | E0302 | não dá pra provar |
  | E0303 | dado publicado depois do que a função assume |

- **Consequência:** escolher uma estratégia pelo backtest e avaliar a escolhida no mesmo histórico também não compila.
- **Aceitação:**
  - o momentum de `examples/momentum.krs` continua idêntico, bit a bit, ao laboratório em Rust;
  - a versão que inclui o mês corrente não compila;
  - 14 testes de vazamento em `tests/temporal.rs`: `if`, `continue`, valores carregados pelo laço, estatística do histórico inteiro, funções anotadas, sessão interativa (REPL) e escolha por backtest.
- **Extensão do VS Code:** cores para `@ end(...)` e `observed`.

## 0.2.1 — Kairós em qualquer sistema

- **Binários para 32 e 64 bits, x86, ARM e além**, gerados e publicados automaticamente a cada versão:

  | sistema | arquiteturas |
  |---|---|
  | Linux (estático, roda em qualquer distribuição) | x86_64, i686 (32 bits), ARM64, ARMv7 e ARMv6 (32 bits, Raspberry Pi), RISC-V 64, PowerPC64 LE, s390x |
  | macOS | Intel e Apple Silicon |
  | Windows | x86_64, i686 (32 bits), ARM64 |
  | FreeBSD | x86_64 |
  | NetBSD, OpenBSD | x86_64 (compilados nativamente) |
  | WebAssembly (WASI) | qualquer sistema com um runtime WASI: wasmtime, wasmer… |
  | experimentais (compilados, sem teste em execução) | FreeBSD i686, illumos x86_64 |

  Nos outros sistemas é só compilar com `cargo build --release -p kairos`.
- **Rust puro:** a Kairós não depende de nenhuma biblioteca do sistema. O acesso à rede do laboratório virou o recurso opcional `net`, que a linguagem não usa, e com isso o OpenSSL deixou de ser necessário.
- **Os mesmos bits em qualquer sistema:** `exp`, `ln` e `^` usam uma implementação matemática em Rust puro (`libm`), e não a do sistema operacional.
- **`.krs` reconhecido:**
  - extensão do VS Code (`kairos-vscode-0.2.1.vsix`): reconhece arquivos `.krs`, dá cores de sintaxe, comentários (`Ctrl+/`), fechamento de colchetes e cores nos blocos ` ```krs ` do Markdown;
  - gramática TextMate em `editors/vscode/syntaxes/kairos.tmLanguage.json`, que serve também no Sublime Text e nos editores JetBrains, e é a base para o reconhecimento pelo GitHub (ver `editors/README.md`);
  - scripts com `#!/usr/bin/env kairos` na primeira linha rodam direto no Linux, macOS e BSD (`chmod +x programa.krs` e depois `./programa.krs`).
- **CI:** testes a cada push:
  - nativos em Linux (x86_64 e ARM64), macOS, Windows (x86_64 e i686 de 32 bits) e FreeBSD;
  - emulados (QEMU) em Linux i686 e ARMv7 de 32 bits, RISC-V 64, PowerPC64 LE e s390x, que é big-endian;
  - WebAssembly, rodando sob o wasmtime.
- **Licença MIT** (`LICENSE`), incluída também nos pacotes e na extensão.
- **Política de segurança** (`SECURITY.md`): versões com suporte e reporte privado pela aba Security.
- **32 bits de verdade:**
  - índices e tamanhos são conferidos em 64 bits antes de virar `usize`, então nada é truncado em silêncio;
  - a pilha do interpretador recua para um tamanho menor quando falta espaço de endereçamento;
  - recursão profunda é sempre um erro limpo (R0005), com o limite proporcional à pilha. O teste novo `tests/stack.rs` cobre isso.
- **WebAssembly:** sem threads, o interpretador roda na thread principal. A profundidade de chamadas é limitada pelo runtime: no wasmtime, 512 KiB por padrão, cerca de 30 níveis de recursão. Para mais, rode com `wasmtime run -W max-wasm-stack=N --env KAIROS_STACK_BYTES=N`.

## 0.2.0 — M2: tensores com eixos nomeados + dados da B3

- **Tensores com eixos nomeados:** `Tensor[month, asset; f64]`, sempre indexados pelo nome (`ret[month: m]`, `ret[month: m - 12 ..= m - 2]`).
- **Operações sobre tensores:**
  - elemento a elemento, alinhadas pelo eixo;
  - reduções com `over:`;
  - NaN como dado faltante;
  - ranking (`top_k`, `rank`) e contração (`contract`).
- **Eixo errado é erro de compilação** (E0200, E0201, E0202).
- **Módulo `b3`:** os dados reais da B3 e o simulador realista do laboratório quant-b3.
- **Aceitação:** `examples/momentum.krs` reproduz bit a bit os 9 números da mesma carteira calculada em Rust (CAGR 14,63%, Sharpe 0,22, maxDD −48,6%).
- Inclui o M1, o núcleo da linguagem:
  - tipos, inferência, `match` exaustivo, closures e `inout`;
  - interpretador e REPL;
  - biblioteca padrão escrita em Kairós.
