# Changelog

Versões da linguagem Kairós (pasta `kairos/`). Cada versão vira uma release no GitHub, gerada automaticamente por `.github/workflows/release.yml` a partir da seção correspondente deste arquivo.

## 0.11.0 — M11: o simulador escrito em Kairós

**`b3.evaluate` e `b3.simulate` agora rodam o simulador realista do laboratório escrito em Kairós** ([`kairos/std/sim.krs`](kairos/std/sim.krs)), carregado junto com o prelúdio. O runtime ficou só com o que é garantia da linguagem: holdout aberto, cerca dos dados, pesos válidos e o registro da tentativa.

- **O simulador:**
  - custos de corretagem e emolumentos, metade do spread real (mediana de 21 pregões) e impacto k·σ·√(ordem/volume);
  - atraso de execução e limite de capacidade;
  - IR de pessoa física: isenção de R$ 20 mil, prejuízo compensado, custo médio e JCP retido;
  - o CDI no caixa.
- **A avaliação:**
  - CAGR, volatilidade, Sharpe sobre o CDI e maxDD;
  - alpha CAPM contra o mercado do NEFIN, com t de Newey-West.
- **Mesmos bits:** as somas seguem a ordem do simulador Rust.
  - um mercado sintético (dividendos e JCP, cotações faltando, spreads sem compra/venda, papéis ilíquidos, quedas) confere mês a mês custo fixo e real, atrasos de 0 a 3 pregões, limite de capacidade e meses isentos e tributados, nos 12 ambientes do CI;
  - os cinco estudos imprimem os mesmos relatórios;
  - o teste da ponte dá os mesmos 9 números, CAGR incluído.
- **Os dados para o simulador:** o runtime entrega o painel em forma compacta, só os pregões em que cada ativo negociou, já cortado pela cerca do holdout, e o calendário. Tudo é montado uma vez por sessão.
- **`std/stats.krs`:** as estatísticas do laboratório entram na biblioteca padrão (`stat_mean`, `stat_sd`, `ols_nw`, `nw_lags`…). [`lab/stats.krs`](kairos/lab/stats.krs) mantém os nomes do laboratório.
- **Desempenho:** fora do `grad`, funções nativas e chamadas ao módulo `b3` não copiam mais os argumentos.
- **Referência:** `KAIROS_RUST_SIM=1` usa o simulador Rust antigo, só para comparação.

## 0.10.0 — M10: a base de dados do laboratório montada em Kairós

**Do download ao painel, os dados do laboratório agora são montados por programas Kairós.** Cada etapa grava os mesmos bytes que o Rust gravava.

| etapa | programa ([`kairos/lab/`](kairos/lab)) | conferido contra o Rust |
|---|---|---|
| ZIPs anuais da COTAHIST | `fetch_cotahist.krs` | o mesmo ZIP baixado; ano já completo não é baixado de novo |
| COTAHIST → `bars.bin` | `ingest.krs` | 220 MB iguais, byte a byte: 2,3 milhões de pregões, 2.241 séries, 381 trocas de código encadeadas por ISIN, em 42 s |
| eventos da API da B3 | `fetch_events.krs` | os arquivos de cache iguais, byte a byte, e os mesmos nomes de arquivo (hash FNV-1a) nos 2.632 já baixados |
| CDI, IPCA, NEFIN | `fetch_macro.krs` | os 6 arquivos iguais; o CDI também se atualiza só no fim |
| eventos + `bars.bin` → `panel.bin` | `build.krs` | `panel.bin` (80 MB), os 3 CSVs de diagnóstico e as estatísticas iguais, em 16 s |
| a atualização diária | `update.krs` | as mesmas etapas que o paper trading roda antes de operar |

- **O painel em Kairós:** proventos e JCP, desdobramentos e grupamentos, cisões, subscrições, desdobramentos inferidos (marcador EB/EG ou razão enorme exata) e saltos sem explicação neutralizados. Reproduz até o jeito do `serde` de ignorar um arquivo de eventos inteiro quando um campo vem com o tipo errado.
- **Runtime:**
  - `read_lines(arquivo, filtros)` lê texto de largura fixa, inclusive de dentro de ZIPs, filtrando por coluna durante a leitura: os 86 MB da COTAHIST 2026 viram 296.697 linhas em 1 s;
  - o tipo `Bytes` (`put_*`/`get_*` de u8, u32, u64, i64, f32, f64 e texto; `write_bytes`/`read_bytes`) lê e grava os formatos binários do laboratório;
  - operações de bits, aritmética que dá a volta (`wrapping_add/mul`), `to_hex`, `file_size`, `file_date` e `http_get_all` (vários downloads em paralelo);
  - `substr` em texto ASCII corta os bytes direto.
- **Biblioteca padrão:**
  - `std/base64.krs`;
  - `std/json.krs` ganhou `json_try`, que devolve o erro em vez de parar, e lê com o rigor do `serde_json`.
- **Testes:**
  - um mercado sintético que exercita todas as regras do painel tem de dar o mesmo `panel.bin` e os mesmos CSVs do Rust, nos 12 ambientes do CI;
  - os mesmos testes rodam com os dados reais (`--ignored`).
- **Ainda em Rust:**
  - o simulador e o registro, chamados pelo módulo `b3` (o próximo marco);
  - o paper trading. Ele precisa ver os dados de nov/2026 em diante, que a Kairós esconde até o holdout do M6 abrir (~2028-11). A decisão sobre isso é do usuário.

## 0.9.0 — M9: arquivos, texto, mapas, internet e a biblioteca padrão em Kairós

**A Kairós passa a ler e escrever arquivos, baixar da internet e processar texto, e a biblioteca padrão é escrita nela mesma.** Com isso, mais duas partes do laboratório saíram do Rust: o registro de tentativas e a coleta dos dados macro.

- **Texto:**
  - `split`, `lines`, `join`, `trim`, `upper`, `lower`, `replace`, `starts_with`, `ends_with`, `find`, `contains`, `codes` e `from_codes`;
  - `parse_int` (texto que não é número dá R0010) e `parse_float` (dá NaN, como dado faltando);
  - escapes novos: `\r`, `\0` e `\u{e9}`.
- **Arquivos e ambiente:**
  - `read_text` (UTF-8, ou Latin-1 quando não é UTF-8, como os arquivos da B3), `write_text`, `append_text`, `file_exists`, `list_dir`, `make_dir`; erros de arquivo são R0011;
  - `args()` (`kairos run prog.krs a b`), `env(nome)` e `today()`.
- **Internet:** `http_get(url)` e `download(url, arquivo)` usam o `curl` do sistema (Windows 10+, macOS, Linux). O HTTPS fica com o TLS do sistema e o binário continua Rust puro nas 20 plataformas; falha é R0013.
- **Mapas `Map[K, V]`:**
  - chaves `i64` ou `str`, na ordem de inserção: `["PETR4": 38.5]`, `[:]`, `m[k] = v`, `get`, `keys`, `values`, `items`, `remove`;
  - semântica de valor, como as listas;
  - chave ausente é R0012.
- **Biblioteca padrão escrita em Kairós**, embutida no executável (`import "std/json.krs"`):
  - `std/json.krs`: leitura e escrita (RFC 8259);
  - `std/csv.krs` (RFC 4180);
  - `std/dates.krs`: datas AAAAMMDD.
- **Tempo:** o que vem de arquivo ou da rede não tem hora conhecida. Um valor lido assim não decide pesos (E0301), a menos que se declare com `observed` quando ele fica conhecido. `args()`, `env()` e `today()` são conhecidos antes de tudo.
- **Laboratório:**
  - [`lab/registry.krs`](kairos/lab/registry.krs) é o `registro`: o DSR, a normal e a inversa dela escritos em Kairós, com o relatório igual ao do Rust;
  - [`lab/fetch_macro.krs`](kairos/lab/fetch_macro.krs) é o `fetch-macro`: CDI e IPCA da API do Banco Central e os fatores do NEFIN, com os 6 arquivos iguais aos do Rust, byte a byte.
- **Desempenho:** fora do `grad`, as funções nativas não percorrem mais listas inteiras procurando valores rastreados. Um `len(xs)` dentro de um laço deixou de ser quadrático, e um JSON de 300 KB da B3, que levava 70 s, agora é lido em 0,2 s.

## 0.8.0 — M8: o laboratório escrito em Kairós

**Os cinco estudos do laboratório `quant-b3` agora são programas Kairós, e cada um imprime o relatório do Rust linha por linha.** As regras de pesquisa saíram do Rust: universo, estatística, carteiras, simulações e decisões. O runtime em Rust ficou com os dados, o simulador e o registro de tentativas, como o CPython fica com o C.

- **Os estudos** ([`kairos/lab/`](kairos/lab)), conferidos nos dados reais por `cargo test --release -p kairos --test lab -- --ignored`:

  | programa | estudo do laboratório | relatório |
  |---|---|---|
  | `momentum.krs` | `quant-b3 momentum`: replicação do WML do NEFIN, prêmio long-short, long-only depois de custos, regressões de fatores | 31 linhas, idênticas |
  | `risk.krs` | `momentum-risco`: WML escalado pela vol realizada, long-only com exposição controlada | 21 linhas, idênticas |
  | `realistic.krs` | `realista`: custo real, 1 pregão de atraso e IR de pessoa física em três capitais | 25 linhas, idênticas |
  | `smallcaps.krs` | `small-caps`: duas hipóteses registradas, pesquisa, holdout e capacidade | 34 linhas, idênticas |
  | `search.krs` | `busca`: a busca de alpha do protocolo registrado (6 sinais de ações + tendência) | 25 linhas, idênticas |

- **A biblioteca do laboratório em Kairós:**
  - `universe.krs`: o universo líquido do NEFIN e o segmento small. Dá exatamente o do Rust (41.106 pares ativo-mês, nenhuma diferença), e a análise temporal prova que ele é causal;
  - `stats.krs`: média, desvio, correlação, CAGR, maxDD, defasagens de Newey-West e MQO com estatística t de Newey-West, bit a bit iguais ao Rust;
  - `portfolios.krs`: formação do momentum, carteiras diárias equal-weight, vol realizada;
  - `fmt.krs`: a formatação dos relatórios.
- **Linguagem:**
  - `import "arquivo.krs"`: cada arquivo roda uma vez, e o caminho é relativo ao arquivo que importa ou a `KAIROS_PATH`;
  - dados diários como `Tensor[month, session, asset]`, com a linha m conhecida em `end(m)`: `b3.daily_returns()`, `daily_volume()`, `daily_price()` e `daily_dividends()`, mais `b3.sessions()` e `b3.session_dates()`;
  - `b3.issuers()`, `b3.classes()`, `b3.nefin(fator, vw:)`, `b3.nefin_daily(...)` e `b3.ipca()`;
  - `b3.simulate(w, ...)`: o simulador do laboratório mês a mês (retorno, exposição, giro, custo, IR, isenções), como a struct `Simulation`;
  - `b3.record_trial(...)`, `b3.deflate_trial(série)` e `b3.results_sha256(arquivo)`;
  - as opções `cap_adv:` (limite de posição em dias de volume) e `trial: false` (referência que não conta como tentativa, visível no código);
  - as funções nativas `format_fixed(x, d)`, `substr(s, início, n)`, `nan_quantile(x, q, over:)` e `merge_days(x)`.
- **Contagem honesta de tentativas:** uma estratégia é uma tentativa, por mais vezes que seja executada. Os mesmos pesos com outro custo, atraso, capital ou IR não contam de novo. Rodar os cinco estudos deixa o registro igual byte a byte, com as mesmas 16 tentativas.
- **Análise temporal mais precisa:**
  - uma chamada cujo argumento temporal é conhecido antes ou depois do que a função supõe é analisada no contexto da chamada, como se estivesse expandida no lugar. Assim, uma função que recebe o universo, conhecido um mês antes, pode usá-lo no próprio mês;
  - um símbolo de laço limitado por outro se junta a ele: estatísticas de duas passadas numa janela (a média, depois os desvios) são provadas causais;
  - simular sob uma condição que depende de um resultado continua rejeitado, porque é escolher a estratégia pelo backtest. A busca simula todos os candidatos antes de escolher.
- **Testes:** `tests/lab.rs` confere universo, estatística, formatação e segmento small num mercado sintético feito para os casos de borda, e tem os 5 testes de relatório com os dados reais. Há também 1 teste temporal novo.
- **Ainda em Rust:** a coleta e a montagem dos dados (B3, BCB, NEFIN, `panel.bin`), o simulador e o registro, chamados pelo módulo `b3`, e o paper trading. Os próximos marcos levam isso para Kairós (§13 da spec).

## 0.7.0 — M7: código nativo com Cranelift

**Funções escalares puras viram código de máquina na primeira chamada, com os mesmos bits e os mesmos erros do interpretador.** Na média geométrica de seis programas numéricos, o resultado fica em 1,20× do Rust.

- **JIT** ([`kairos/src/native.rs`](kairos/src/native.rs)):
  - **o que compila:** funções com parâmetros, variáveis e resultado `i64`, `f64` ou `bool`; aritmética, comparações, `and`/`or`/`not`, `if`, `while`, `for`, `break`/`continue`/`return`, chamadas e recursão, e `sqrt`, `exp`, `ln`, `abs`, `floor`, `ceil`, `round`, `float`, `int`, `min` e `max`;
  - **o resto** fica no interpretador, e as duas formas se chamam livremente.
- **Mesma semântica:**
  - estouro, divisão por zero, potência negativa e `int` fora do intervalo dão os mesmos erros, com a mesma mensagem e o mesmo trecho (R0006, R0002, R0007, R0009);
  - a profundidade de chamadas tem o mesmo limite (R0005);
  - `exp`, `ln`, `^` e `round` chamam as mesmas funções em Rust puro;
  - [`kairos/tests/native.rs`](kairos/tests/native.rs) roda tudo com e sem JIT e exige igualdade.
- **Otimizações:**
  - escalares e o orçamento de chamadas aninhadas vão em registradores, e o erro volta como segundo resultado;
  - condições com `and`/`or`/`not` viram desvios diretos;
  - divisão e resto por constante dispensam as verificações que não podem falhar, e por potência de 2 viram deslocamentos;
  - `x % 2^k == 0` vira um teste de bits.
- **`kairos-bench`** (`cargo run --release -p kairos --bin kairos-bench`): seis programas em Kairós e em Rust, com a mesma semântica e resultados idênticos.

| benchmark | o que mede | interpretador | nativo (JIT) | Rust | nativo ÷ Rust |
|---|---|---|---|---|---|
| `fib(30)` | chamadas recursivas | 315 ms | 3,8 ms | 3,1 ms | 1,25× |
| `sum_sqrt` (20 mi) | laço de f64 com `sqrt` | 1,95 s | 24,2 ms | 24,1 ms | 1,00× |
| `collatz` (100 mil) | inteiros verificados com desvios | 1,62 s | 21,5 ms | 14,4 ms | 1,50× |
| `black_scholes` (300 mil) | finanças: `exp`, `ln`, `sqrt`, CDF normal | 647 ms | 11,0 ms | 9,1 ms | 1,20× |
| `mandelbrot` (300×200) | laços aninhados de f64 com saída antecipada | 950 ms | 6,3 ms | 5,3 ms | 1,19× |
| `monte_carlo` (2 mi) | gerador Park-Miller + f64 | 747 ms | 16,9 ms | 15,0 ms | 1,13× |
| **média geométrica** | | | | | **1,20×** |

- **Plataformas:**
  - **com JIT:** x86_64 em Linux, macOS, Windows e FreeBSD, e ARM64 em Linux e macOS;
  - **sem JIT:** nas demais tudo roda no interpretador, com os mesmos resultados;
  - `KAIROS_JIT=0` desliga.
- **Fica para depois:** o elemento ternário `t3`, tensores no código nativo e compilação antecipada (AOT).

## 0.6.0 — M6: o primeiro modelo, pré-registrado e avaliado sem autoengano

**Uma carteira treinada por `grad` no patrimônio depois de IR, julgada pelas regras que ela mesma registrou antes.** O resultado é honesto: ela aprendeu a negociar devagar e bateu o equal-weight, mas não há evidência de alpha, então foi rejeitada.

- **Pré-registro:** [`kairos/research/m6-protocol.md`](kairos/research/m6-protocol.md) (SHA-256 `d44da677…`) e o programa [`kairos/research/tax_portfolio.krs`](kairos/research/tax_portfolio.krs) foram publicados (commit `52a186d`) antes de qualquer execução nos dados reais.
- **O modelo:**
  - softmax de sinais clássicos (momentum 12-2, reversão, volatilidade) sobre os ativos vivos do NEFIN;
  - fração investida guiada pela tendência do mercado;
  - velocidade de negociação aprendida.
- **O treino:**
  - log-crescimento do patrimônio depois de custo e IR de pessoa física, com isenção de R$ 20 mil como degrau suave, 15% sobre o ganho, prejuízo compensado e custo médio;
  - walk-forward a cada janeiro, de 2006 em diante.
- **O resultado** (2006–2026, custos reais, IR, R$ 100 mil, [relatório](kairos/research/m6-report.md)):

  | carteira | CAGR depois de custos e IR | giro por ano | DSR |
  |---|---|---|---|
  | modelo | 7,50% | 0,58 | 0,8% |
  | equal-weight | 5,52% | 1,28 | 0,5% |
  | momentum | 9,74% | 5,04 | 3,5% |

  O CDI rendeu 10,36% ao ano. A regra exigia DSR ≥ 90%: **rejeitado**. O holdout (2026-11 em diante, ≥ 24 meses) fica reservado no livro-razão.
- **Análise temporal mais precisa:**
  - cada função tem um horizonte de dados, trocado em cada chamada pelo que foi passado: treinar em `x[month: a ..= m - 1]` é provado causal, e treinar até `m` (direto ou dentro de `grad`) é E0301;
  - `grad(f)(x)` é analisado como `f(x)`;
  - laços `for` sobre o tamanho de um parâmetro temporal são acompanhados.
- **Testes:**
  - `tests/m6.rs` roda o programa registrado num mercado sintético com efeito momentum, e o modelo o encontra;
  - 2 testes temporais novos.

## 0.5.0 — M5: holdout de uso único, protocolos e registro de tentativas

**O holdout abre uma vez, e cada avaliação conta.** É a garantia nº 2 da linguagem: olhar a validação de novo e testar até dar certo deixam rastro, ou não compilam.

```krs
protocol "momentum-demo" {
    research: until 202610
    holdout: from 202611
    min_months: 12
    rule: "accept if the holdout's CAPM alpha > 0 and its PSR >= 90%"
}
let research = b3.evaluate(weights(), lag: 0)                // só enxerga até 2026-10
let h = b3.holdout("momentum-demo")                          // Holdout: usável uma vez
let final = open_once(h, || weights(), lag: 0)               // o holdout só existe aqui dentro
let again = open_once(h, || weights(), lag: 1)               // error[E0501]: holdout `h` was already used
```

- **`protocol "nome" { research: until AAAAMM  holdout: from AAAAMM  min_months: N  rule: "..." }`:**
  - quem declara um protocolo só enxerga a pesquisa: os tensores de `b3.*` terminam em `research: until`;
  - na primeira execução, o protocolo é registrado com hash no livro-razão `results/holdout_ledger.csv`;
  - a partir daí, os meses do holdout ficam escondidos de **todos** os programas até a abertura;
  - mudar o protocolo com o mesmo nome é E0112.
- **`Holdout` é um tipo afim:**
  - `b3.holdout("nome")` o cria, e `open_once(h, || pesos(), ...)` o consome;
  - usar de novo, num laço ou numa lambda é **E0501**;
  - os meses do holdout só existem enquanto a função de pesos roda, e a avaliação é uma carteira nova a partir do primeiro mês do holdout.
- **Livro-razão entre execuções:**
  - **Gravação:** a abertura é gravada antes de qualquer mês do holdout ser visto, e o resultado depois. Abrir de novo é **R0502**.
  - **Holdout sem dados suficientes** (`min_months`, padrão 12): **R0503**, sem gravar nada.
  - **Holdout novo:** precisa começar depois de todo mês que um programa Kairós já viu.
  - **Origem:** o livro-razão nasce com a busca do laboratório, que já abriu 2019-01..2026-10. Hoje o único holdout limpo é o futuro.
- **Registro de tentativas automático:**
  - toda `b3.evaluate` entra em `results/registro_testes.csv`, junto com as estratégias que o laboratório testou em Rust;
  - séries idênticas não contam duas vezes;
  - `Evaluation` ganhou `trials` (tentativas no registro) e `dsr` (Deflated Sharpe Ratio). Rodando o momentum com o registro atual do laboratório: 14 tentativas, DSR 9%.
- **`kairos ledger`** mostra o livro-razão, e **`kairos ledger withdraw <nome>`** desiste de um protocolo nunca aberto. A variável `KAIROS_RESULTS` escolhe outra pasta de resultados.
- **Novos erros:**

  | código | situação |
  |---|---|
  | E0501 | holdout usado duas vezes, num laço ou numa lambda |
  | R0500 | livro-razão ou registro ilegível ou impossível de gravar |
  | R0502 | holdout já aberto ou retirado, ou que começa em meses já vistos |
  | R0503 | holdout ainda sem `min_months` meses de dados |
  | R0504 | `b3.evaluate` ou `open_once` dentro de uma abertura |
  | R0505 | falha da função de pesos durante a abertura (o holdout fica gasto) |

- **Testes:**
  - 7 testes dourados novos;
  - 5 testes de integração num mercado sintético (`tests/holdout.rs`), que rodam em todas as plataformas da CI;
  - o laboratório ganhou `Data::build`, que monta os dados sem ler arquivos.
- **Exemplo:** `examples/holdout.krs`.
- **Limite honesto:** o livro-razão só sabe o que programas Kairós viram, e é um arquivo que o dono pode editar. Ele mantém honesto quem quer ser honesto.

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
