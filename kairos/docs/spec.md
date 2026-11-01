# Kairós — especificação v0.1

> *Kairós* (καιρός): em grego, "o momento certo". A garantia central da linguagem é exatamente essa: cada decisão só enxerga o que já era conhecido no momento em que foi tomada.

**Decisões fechadas com o usuário em 2026-11-01:**
- nome **Kairós** (executável `kairos`, arquivos `.krs`);
- palavras-chave em **inglês**, e por consequência mensagens de erro em inglês (tradução para português fica para depois);
- blocos com **chaves `{}`**;
- **`f64`** como precisão padrão.

**Estado:**
- **M1 (núcleo) pronto:** REPL, biblioteca padrão escrita em Kairós (`std/prelude.krs`) e testes dourados.
- **M2 (tensores com eixos nomeados + ponte `b3`) pronto:**
  - erros de eixo saem na compilação;
  - o momentum escrito em Kairós sobre os dados reais da B3 reproduz **bit a bit** o resultado do laboratório em Rust (`tests/b3_bridge.rs`).
- As garantias de tempo, holdout e `grad` chegam nos marcos M3–M5 (§13).

## 1. Por que mais uma linguagem

Nenhuma linguagem existente dá **garantias** para os erros que mais destroem pesquisa quantitativa:

1. **Usar informação do futuro** (look-ahead): em Python/C++/Rust é só um índice errado e nada avisa.
2. **Olhar o holdout de novo** e ajustar parâmetro depois de ver o resultado.
3. **Não contar quantos testes foram feitos** (data snooping).
4. **Regras financeiras discretas** (IR com isenção, compensação de prejuízo, custo) que não dá pra otimizar por gradiente.
5. **Misturar eixos** (tempo com ativo, feature com amostra) em contas de tensor.

Kairós transforma cada um desses em **erro de compilação** (ou, quando não dá pra provar estaticamente, em erro em tempo de execução, nunca silêncio). Isso vem junto com desempenho de linguagem de sistema e ergonomia de linguagem de script.

## 2. Objetivos, critérios e o que significa "superar"

Superar **todas** as linguagens em **todos** os critérios é impossível por construção, porque alguns critérios brigam entre si:
- compilação instantânea contra otimização máxima;
- liberdade total de ponteiro contra segurança garantida;
- o ecossistema de 30 anos do Python contra uma linguagem nova.

A estratégia é **ser nº 1 onde ninguém compete** e **empatar com o melhor de cada critério clássico**, sempre com meta **medida**:

| critério | melhor hoje | meta Kairós | como | como medir |
|---|---|---|---|---|
| Proteção contra dado do futuro | nenhuma linguagem | **nº 1** | tipos temporais (§6) | suíte com 20 bugs clássicos de look-ahead: 100% pegos |
| Holdout e contagem de testes | nenhuma | **nº 1** | tipos afins + registro (§7) | 2º acesso ao holdout = erro; DSR automático |
| Diferenciar regras financeiras | JAX/PyTorch (na mão) | **nº 1** | `smooth` nativo (§8) | gradiente de IR/custo confere com diferença finita |
| Eixos de tensor checados | Dex (pesquisa), xarray (em execução) | **nº 1 em uso prático** | eixos nomeados (§5) | erro de eixo na compilação |
| Reprodutibilidade bit a bit + procedência | nenhuma por padrão | **nº 1** | reduções determinísticas + recibo (§11) | mesma saída em 2 execuções e 2 máquinas |
| Desempenho numérico | C, C++, Rust, Fortran | **empatar** (≤ 1,2× Rust) | código nativo, fusão de laços, SIMD, agendamento (§9) | `kairos-bench` |
| Latência previsível | C, C++, Rust, Zig | **empatar** | sem coletor de lixo; laço sem alocação verificado | zero alocações em laço marcado |
| Ergonomia | Python, Julia | **empatar** | inferência de tipos, sintaxe enxuta, REPL | linhas e tempo pra escrever a mesma estratégia |
| Segurança de memória | Rust | **empatar, com regras mais simples** | semântica de valor mutável (§4.4) | nenhum comportamento indefinido fora de `unsafe` |
| Tempo de compilação | Go, Zig | **perto** | front-end simples, back-end Cranelift | verificar 10 mil linhas em < 1 s |
| Ecossistema | Python | **perder no começo** | FFI com Rust/C, leitura de CSV/Parquet | — |

## 3. Três níveis numa linguagem só

A mesma linguagem e o mesmo compilador. O nível alto é o padrão; desce-se só onde o desempenho pede.

**Alto nível:** estratégia em poucas linhas, com garantias.

```krs
fn momentum(m: month) -> Tensor[asset; f64] @ end(m - 1) {
    let signal = prod(1.0 + ret[m - 12 ..= m - 2], over: month) - 1.0
    equal_weights(top_tercile(signal))
}
```

**Nível de kernel:** separa **o que** calcular de **como** calcular (ideia do Halide/TVM). Os eixos nomeados deixam o agendamento legível e verificado.

```krs
kernel matmul(a: Tensor[i, k; f32], b: Tensor[k, j; f32]) -> Tensor[i, j; f32] {
    compute c[i, j] = sum(a[i, k] * b[k, j], over: k)
    schedule { tile(i: 64, j: 64, k: 256); vectorize(j: 8); parallel(i) }
}
```

**Baixo nível:** ponteiro, layout de memória e SIMD explícito, sem rede de proteção, só dentro de `unsafe`.

```krs
fn fast_sum(x: List[f64]) -> f64 {
    var acc = Simd[f64; 4].zero()
    unsafe {
        let p = x.ptr()
        for i in 0..len(x) / 4 { acc = acc + load_simd(p + 4 * i) }
    }
    acc.horizontal_sum() + sum(x[(len(x) / 4) * 4 ..])
}
```

## 4. Núcleo da linguagem (M1)

### 4.1 Léxico

- Fonte em UTF-8; identificadores aceitam acentos (`preço`).
- Comentários com `//` e `/* */`.
- Números: `20_000`, `1.5`, `1e-3`.
- Texto: `"..."`, com `\n \t \" \\`.
- **Ponto e vírgula opcional**: a quebra de linha termina a instrução, no estilo Go/Swift.
  - Dentro de `( )` e `[ ]` a quebra de linha é ignorada.
  - Uma linha que começa com `.` continua a anterior (encadeamento de métodos).
  - `else` pode ficar na linha de baixo.

### 4.2 Palavras-chave

**No M1:**
- `let` / `var`, `fn`, `if` / `else`, `match`, `for` … `in`, `while`, `break`, `continue`, `return`;
- `struct`, `enum`, `inout`;
- `and`, `or`, `not`, `true`, `false`.

**No M2:** `axis`.

**Reservadas para os próximos marcos:** usar qualquer uma como nome é erro E0002, com a dica de quando ela chega.
- `smooth` (M4), `protocol` e `sink` (M5);
- `kernel`, `compute`, `schedule`, `unsafe`, `parallel` (M7);
- `trait`, `import`, `pub`, `as`, `loop`, `type`, `const`, `use`, `self`.

### 4.3 Tipos

| grupo | tipos |
|---|---|
| escalares (M1) | `i64`, `f64`, `bool`, `str`, `()` |
| coleções (M1) | `List[T]`, tuplas `(A, B)` |
| definidos pelo usuário (M1) | `struct`, `enum` (com dados), funções `fn(A) -> B` |
| depois | `f32`/`i32`…, `date`, `Money` (ponto fixo), `Option[T]`/`Result[T, E]`, `Tensor[eixos; T]`, `Simd[T; N]`, `trait` |

- **Inferência:** só os parâmetros de função exigem tipo; o retorno é anotado com `->` (sem anotação, é `()`).
- **Literal inteiro se adapta:** `let x: f64 = 1` e `x * 2` funcionam (o `2` vira `f64`).
- **`i64` e `f64` nunca se misturam em variáveis:** `x * n`, com `x: f64` e `n: i64`, é erro E0107 com a dica `float(n)`.
- **Chamada uniforme:** `x.f(y)` é o mesmo que `f(x, y)`. Não há classes nem `impl`.
- **`match` precisa ser exaustivo:** faltou um caso do `enum`, é erro E0110.

### 4.4 Memória: semântica de valor mutável, sem coletor de lixo

- Toda variável **é dona** do seu valor. `let b = a` **copia semanticamente**, mas listas, textos e estruturas grandes usam **cópia sob escrita**: copiar custa nada até alguém escrever.
- Parâmetros são só leitura por padrão. Com **`inout`**, a função muda o valor do chamador no lugar; na chamada se escreve `f(inout x)`, ou `x.f()`, em que o receptor vira `inout` sozinho.
- Duas passagens `inout` da mesma variável na mesma chamada é erro E0111: não existe apelidamento.
- Não existe referência guardada dentro de estrutura em código seguro. Logo não há ponteiro pendurado, corrida de dados nem coletor de lixo (ideia da linguagem Hylo/Val, mais simples que o borrow checker do Rust).
- **Inteiros nunca estouram em silêncio:** estouro e divisão por zero são erro em tempo de execução (R0006, R0002).

### 4.5 Exemplo do núcleo (roda no M1)

```krs
struct Order { ticker: str, buy: bool, value: f64 }

enum Shape { Circle(f64), Rect(f64, f64), Empty }

fn area(s: Shape) -> f64 {
    match s {
        Circle(r) => 3.14159 * r ^ 2,
        Rect(w, h) => w * h,
        Empty => 0.0,
    }
}

fn add_fee(inout o: Order, rate: f64) {
    o.value = o.value * (1.0 + rate)
}

var o = Order { ticker: "PETR4", buy: true, value: 1000.0 }
o.add_fee(0.0003)                        // UFCS: o vira inout sozinho
print(o.value)                           // 1000.3

let returns = [0.02, -0.01, 0.03]
print(compound(returns))                 // função da biblioteca padrão, escrita em Kairós
print(returns.map(|r| r * 100.0))        // lambdas capturam por valor
```

## 5. Eixos nomeados (M2)

```krs
axis asset
axis feature
axis sample
axis month: temporal(monthly)
axis day: temporal(daily)
```

- **Operação elemento a elemento** casa eixos **pelo nome**. A ordem não importa: `x: Tensor[month, asset]` + `y: Tensor[asset]` repete `y` ao longo de `month`.
- **Redução e contração** são sempre pelo nome:
  - `sum(x, over: asset)` remove o eixo do tipo;
  - `contract(x, w, over: feature)` é multiplicação de matriz.
- **Somar eixos diferentes** é erro de compilação:

```text
error[E0201]: incompatible axes in `x + y`
  --> model.krs:12:5
   |
12 |     x + y
   |     ^^^^^ `x` has axes [sample, feature], `y` has [sample, hidden]
   = hint: maybe you meant `contract(x, w, over: feature)` first
```

### 5.1 O que o M2 implementa

**Tipos e eixos**
- `Tensor[month, asset; f64]`: os elementos são `f64` ou `bool`. A ordem dos eixos no tipo não importa (`Tensor[asset, month; f64]` é o mesmo tipo).
- `month` (mensal), `day` (diário) e `asset` já vêm declarados. Declarar de novo igual é permitido; outros eixos se declaram com `axis nome`.
- Um tensor sem eixos é um escalar: `sum(x)` dá `f64`.

**Indexação sempre pelo nome**
- `ret[month: m]` remove o eixo;
- `ret[month: m - 12 ..= m - 2]` mantém o eixo, com intervalo inclusivo (`..` é exclusivo);
- `x[month: i, asset: j]` dá um escalar.

**Atribuição**
- `w[month: m] = v` aceita um escalar ou um tensor com exatamente os eixos que sobram.
- `w[month: m] += v` aceita qualquer tensor cujos eixos estejam contidos no destino.
- Copiar é valor: o tensor só é duplicado se estiver compartilhado.

**Elemento a elemento**
- `+ - * / % ^` e comparações (`<`, `==`, …, que dão tensor de `bool`);
- `and`, `or`, `not` em tensores de `bool`;
- `-x`, `abs`, `sqrt`, `exp`, `ln`, `floor`, `ceil`, `round`;
- `max(x, y)` e `min(x, y)`: com NaN, o resultado é NaN;
- `where(c, a, b)`.

**Dado faltante é NaN**
- constantes `nan`, `inf` e `pi`;
- `is_nan(x)` e `fill_nan(x, v)`;
- `nan_sum`, `nan_mean`, `nan_prod` e `nan_count` ignoram NaN.

**Reduções:** `sum`, `mean`, `prod`, `max` e `min`, com `over: eixo` opcional. São somadas na ordem dos índices, então o resultado é reprodutível bit a bit. `count(mask)` conta os `true`.

**Ranking:** `rank(x, over:)` dá o percentil em [0, 1]. `top_k(x, k, over:)` e `bottom_k(x, k, over:)` dão uma máscara `bool`. NaN nunca entra, e o empate segue a ordem dos ativos (ordenação estável).

**Construção e forma**
- `zeros(month: 12, asset: 3)`, `ones(...)`, `full(v, ...)`, `zeros_like(x)`, `full_like(x, v)`;
- `tensor([..], asset)` e `to_list(x)`;
- `size(x, month)`.

**Contração:** `contract(x, w, over: feature)`. Os outros eixos compartilhados viram eixos de lote; os demais formam o produto externo.

**Erros**

| código | situação |
|---|---|
| E0200 | eixo desconhecido |
| E0201 | eixos incompatíveis, ou eixo que o tensor não tem |
| E0202 | índice por posição num tensor (ou por nome numa lista) |
| R0201 | mesmos eixos com tamanhos diferentes, em tempo de execução (tamanhos ainda não estão no tipo) |
| R0001 | índice fora do intervalo |

**Argumentos nomeados** (`over: asset`) existem por enquanto só nas funções embutidas e no módulo `b3`.

## 6. Tipos temporais (M3, garantia nº 1)

Todo valor tem um **tempo de conhecimento** κ:

- constantes têm κ = −∞;
- `ret[m]` tem κ = `end(m)`, e um intervalo `ret[a ..= b]` tem κ = `end(b)`;
- o κ de uma operação é o **máximo** dos operandos (mesmo princípio dos tipos de fluxo de informação);
- uma função anotada `-> T @ τ` **só compila se κ(corpo) ≤ τ**.

```krs
fn momentum(m: month) -> Tensor[asset; f64] @ end(m - 1) {
    let window = ret[m - 12 ..= m - 2]        // κ = end(m-2) <= end(m-1): ok
    equal_weights(top_tercile(prod(1.0 + window, over: month) - 1.0))
}

fn momentum_bad(m: month) -> Tensor[asset; f64] @ end(m - 1) {
    let window = ret[m - 12 ..= m]            // κ = end(m): olha o futuro
    equal_weights(top_tercile(prod(1.0 + window, over: month) - 1.0))
}
```

```text
error[E0301]: use of future information
  --> momentum.krs:7:18
   |
 7 |     let window = ret[m - 12 ..= m]
   |                  ^^^^^^^^^^^^^^^^^ known at end(m), but the decision is made at end(m - 1)
   = hint: momentum signals usually skip the last month: use `m - 2`
```

**Como o verificador decide:**
- Tempos são lineares (`variável + constante`) e o calendário é monotônico. Comparar tempos com a mesma base é aritmética de constantes.
- Laços usam análise de intervalo.
- Quando não dá pra provar, o compilador insere uma **checagem em tempo de execução** e avisa (W0300). Nunca é silencioso.

**Fontes de dados carregam o tempo certo:**
- provento: conhecido na **data de aprovação**, não na data-com;
- balanço: conhecido na **publicação na CVM**, não no fim do trimestre.

Quem define isso é a ponte de dados (§10), não cada script.

## 7. Holdout de uso único, protocolos e contagem de testes (M5, garantia nº 2)

```krs
protocol "portfolio-tax-v1" {
    research: until #2018-12-31
    holdout:  from #2019-01-01
    rule:     "accept if CAPM alpha > 0 on holdout and DSR >= 90%"
}

let (research, holdout) = b3.data().split("portfolio-tax-v1")   // holdout: Holdout[Data]
let θ = train(research)                                          // repeat as much as you want...
let final = holdout.open_once(|h| b3.evaluate(policy(θ), h))     // ...but the holdout opens once
let again = holdout.open_once(|h| b3.evaluate(other, h))         // error[E0501]: holdout already consumed
```

- **`Holdout[T]` é afim:** pode ser usado **no máximo uma vez**, porque `open_once` consome o valor. Checado na compilação.
- **Entre execuções**, um livro-razão grava a abertura. Rodar de novo dá E0502 com a data e o hash do protocolo.
- **O `protocol` é gravado com hash** na primeira execução; mudar o texto cria outro protocolo.
- **Toda `b3.evaluate` em dados de pesquisa vira tentativa no registro**, e `Evaluation.dsr` já vem descontado pelo número de tentativas.

## 8. Diferenciação nativa e regras suaves (M4, garantia nº 3)

- `grad(f)` devolve o gradiente em relação ao 1º argumento; `grad(f, wrt: 2)` escolhe outro. `value_and_grad(f)` devolve os dois.
- **Tipos diferenciáveis:** `f32`, `f64`, tensores deles, e estruturas e tuplas compostas por eles. O espaço tangente é derivado sozinho, então `p - 0.01 * g` funciona com `p` sendo uma estrutura.
- O controle de fluxo é diferenciado ao longo do caminho executado.

**`smooth`: regras discretas que ganham gradiente.**
- **Fora de `grad`** valem as regras **exatas**: a avaliação usa sempre o IR real.
- **Dentro de `grad`** valem as **relaxações declaradas**, iguais no cálculo e na derivada:

```krs
fn monthly_tax(sales: f64, result: f64, carried_loss: f64) -> f64 {
    if sales <= 20_000.0 smooth(scale: 2_000.0) {                 // degrau -> sigmoide
        0.0
    } else {
        0.15 * max(result - carried_loss, 0.0) smooth(scale: 100.0)  // max -> softplus
    }
}
```

**Relaxações nativas:**
- `if … smooth` → sigmoide;
- `max/min … smooth` → softplus;
- `sort smooth` / `top_k smooth` → ordenação diferenciável (Blondel et al., 2020);
- `round smooth`.

A escala pode cair durante o treino (recozimento). O compilador avisa quando uma regra discreta sem `smooth` está no caminho de um `grad`.

## 9. Eficiência

**Pipeline:**
1. léxico → sintaxe (Pratt) → AST;
2. nomes e tipos (inferência; depois eixos, tempo e afinidade);
3. **HIR tipada e resolvida** (slots de variável, chamadas resolvidas);
4. passes: monomorfização, inlining, análise de unicidade (mudança no lugar), diferenciação reversa, **fusão de laços guiada por eixos**, avaliação em tempo de compilação, checagem de `#[no_alloc]`;
5. back-ends:
   - (a) **interpretador de referência** (M1, corretude);
   - (b) **nativo via Cranelift** (M7), em Rust puro e compatível com o toolchain desta máquina, com JIT e AOT;
   - (c) LLVM opcional;
   - (d) GPU.

**Paralelismo:** `parallel for` com **reduções determinísticas**, concorrência estruturada, e semântica de valor sem corrida de dados.

**`kairos-bench`** compara Kairós com Rust `-O`, C `-O2`, Python + NumPy/PyTorch e Julia em:
- laço escalar (Monte Carlo);
- matmul 1024²;
- backtest completo de momentum;
- treino de MLP pequeno.

## 10. Ponte com o laboratório `quant-b3` (M2+)

**Implementado no M2.** Os dados são carregados na primeira chamada: de `KAIROS_DATA`, `./data`, `../data` ou da pasta `data` do checkout que contém o executável.

| Kairós | o que devolve | chama no `quant-b3` (biblioteca) |
|---|---|---|
| `b3.monthly_returns()` | `Tensor[month, asset; f64]`, retorno total do mês; NaN = não negociou | `common::Data::load` (`mret`, `traded`) |
| `b3.eligible()` | `Tensor[month, asset; bool]`, universo líquido NEFIN do ano do mês | `universe::eligible` |
| `b3.cdi()` | `Tensor[month; f64]` | `Data::cdi_m` |
| `b3.months()` / `b3.tickers()` | `List[i64]` (AAAAMM) / `List[str]` | `Data::cal`, `panel` |
| `b3.month_index(200101)` | índice do primeiro mês ≥ AAAAMM | `Cal::idx_from` |
| `b3.evaluate(w, cost_bps: 25.0, lag: 1, capital: 1e6, real_costs: false, tax: false)` | `Evaluation` (meses, CAGR, vol, Sharpe sobre o CDI, maxDD, beta, alpha CAPM e seu t, giro, custo médio) | `backtest::simulate` + `common::perf` + `search::evaluate` |

**Detalhes de `b3.evaluate`:**
- A carteira é long-only: pesos ≥ 0 e soma ≤ 1 por mês; o resto rende CDI.
- Mês com todos os pesos zerados não opera.
- `real_costs: true` usa o custo realista do laboratório: emolumentos + meio spread + impacto.

**Ainda por vir:**

| Kairós | chama no `quant-b3` (biblioteca) |
|---|---|
| tentativas e DSR | `registry::{Trial, record_and_describe, deflate}` (M5) |
| `b3.paper(...)` | `paper` |

## 11. Reprodutibilidade e procedência

Cada execução gera um **recibo** com:
- hash do código;
- versão do compilador;
- hash dos dados;
- protocolo;
- semente.

Reduções são determinísticas por padrão; o modo `fast` é explícito e fica marcado no recibo.

**Os mesmos bits em qualquer sistema (desde a 0.2.1).** `exp`, `ln` e `^` usam uma implementação matemática em Rust puro (`libm`), e não a biblioteca do sistema operacional, que muda de um sistema para outro no último bit. O resto já é exato pelo IEEE 754: `+ − × ÷`, `sqrt`, `floor`, `ceil`, `round`, `%`. A formatação de números é a mesma em todo lugar.

## 12. Esboço do primeiro modelo inédito (M6)

```krs
import b3
import ai.{Attention, Dense, Adam}

axis asset
axis feature
axis hidden
axis month: temporal(monthly)

struct Policy { att: Attention[asset, feature, hidden], out: Dense[hidden, 1] }

fn weights(θ: Policy, x: Tensor[asset, feature; f32] @ end(m - 1)) -> Tensor[asset; f32] @ end(m - 1) {
    softmax(θ.out.apply(θ.att.apply(x)), over: asset)
}

fn wealth(θ: Policy, data: b3.Data) -> f64 {
    b3.simulate(|m| weights(θ, data.features(m)), costs: real, tax: individual).final()
}

protocol "portfolio-tax-v1" { research: until #2018-12-31  holdout: from #2019-01-01 }

let (research, holdout) = b3.data().split("portfolio-tax-v1")
var θ = Policy.random(seed: 7)
var opt = Adam.new(lr: 1e-3)
for step in 0..2_000 {
    opt.step(inout θ, -grad(wealth)(θ, research))     // gradient flows through tax and costs (smooth)
}
print(holdout.open_once(|h| b3.evaluate(|m| weights(θ, h.features(m)), h)))
```

## 13. Roteiro

| marco | entrega | pronto quando |
|---|---|---|
| M0 | spec | ✅ revisada (decisões no topo) |
| M1 | léxico, sintaxe, tipos, interpretador, REPL, `kairos run/check/repl`, biblioteca padrão em Kairós | ✅ 18 testes dourados (funcionalidade, erros de compilação e de execução) |
| M2 | tensores com eixos nomeados + ponte com os dados da B3 | ✅ erros de eixo na compilação (E0200–E0202); o momentum em Kairós (`examples/momentum.krs`) reproduz **bit a bit** os 9 números do `quant-b3` (CAGR, vol, Sharpe, maxDD, beta, alpha, t, giro, custo) |
| M3 | tipos temporais | o momentum continua idêntico ao `quant-b3`, e a versão que olha `ret[month: m]` (o futuro) **não compila** |
| M4 | `grad` nativo + `smooth` + motor de IA em Kairós (Dense, Attention, Adam) | gradiente confere com diferença finita; MLP converge |
| M5 | holdout de uso único + protocolo + registro e DSR automáticos | 2º acesso = erro; tentativas no registro |
| M6 | primeiro modelo inédito (carteira com IR suave) | protocolo antes; simulador real; DSR; paper trading |
| M7 | back-end nativo Cranelift + `kairos-bench`; candidato: elemento `t3` (pesos ternários {−1, 0, +1}, 5 por byte, matmul só com soma e subtração, como no BitNet b1.58), com kernel CPU e CUDA | ≤ 1,2× Rust nos benchmarks numéricos |
