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
- **M3 (tipos temporais) pronto:** usar informação do futuro numa decisão é erro de compilação (§6.1). A versão do momentum que olha o mês corrente não compila.
- **M4 (diferenciação nativa + motor de IA) pronto:** `grad` reverso sobre o caminho executado, espaço tangente (aritmética de structs), funções genéricas, regras suaves e um MLP com Adam escritos em Kairós que aprende (§8.1).
- **M5 (holdout de uso único) pronto:** protocolos cercam os dados de pesquisa e reservam o holdout para todos os programas; o holdout abre uma vez (E0501 na compilação, R0502 entre execuções); toda avaliação vira tentativa no registro, com DSR (§7.1).

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

**No M2:** `axis`. **No M5:** `protocol`.

**Reservadas para os próximos marcos:** usar qualquer uma como nome é erro E0002, com a dica de quando ela chega.
- `smooth` (relaxações declaradas; hoje há `smooth_max`, `smooth_min` e `smooth_step`) e `sink`;
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

### 4.6 Texto, arquivos e ambiente (M9)

Para coletar dados e escrever relatórios em Kairós, sem passar por outra linguagem.

| função | o que faz |
|---|---|
| `split(s, sep)`, `lines(s)`, `join(xs, sep)` | corta por separador, por linha (`\n` ou `\r\n`) e junta |
| `trim(s)`, `upper(s)`, `lower(s)`, `replace(s, de, para)` | transformações |
| `starts_with(s, p)`, `ends_with(s, p)`, `contains(s, sub)`, `find(s, sub)` | buscas (`find` dá o índice do caractere, ou −1) |
| `substr(s, início, n)`, `len(s)`, `codes(s)`, `from_codes(xs)` | caracteres e pontos de código Unicode |
| `parse_int(s)`, `parse_float(s)` | número escrito no texto; `parse_int` inválido é R0010, `parse_float` inválido dá NaN (dado faltando) |
| `read_text(p)`, `write_text(p, s)`, `append_text(p, s)` | arquivo inteiro: UTF-8, ou Latin-1 quando não é UTF-8 (os arquivos da B3) |
| `file_exists(p)`, `list_dir(p)`, `make_dir(p)` | pastas (`list_dir` vem em ordem alfabética) |
| `args()`, `env(nome)`, `today()` | argumentos (`kairos run prog.krs a b`), variáveis de ambiente, a data de hoje (UTC, AAAAMMDD) |

- **Escapes de texto:** `\n`, `\t`, `\r`, `\0`, `\"`, `\\` e `\u{e9}`, este com 1 a 6 dígitos hexadecimais.
- **Erros de arquivo** (não existe, sem permissão…) param o programa com R0011 e a mensagem do sistema.
- **Tempo:** o que vem de um arquivo não tem hora conhecida. Para o compilador, configuração e dado de mercado são iguais, então um valor lido de arquivo não decide pesos (E0301), a menos que se declare quando cada linha fica conhecida, com `observed`. `args()`, `env()` e `today()` são conhecidos antes de tudo.
- **A cerca do holdout** (§7) vale para o módulo `b3`. Ler os arquivos de mercado diretamente fica fora dela, mas aparece no código.

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

### 6.1 O que o M3 implementa (v0.3.0)

A análise roda sobre o HIR tipado, depois da checagem de tipos, como interpretação abstrata (`src/temporal.rs`). O código continua idêntico: no exemplo do momentum, `m` é a variável do laço (`for m in ...`) e `ret` vem de `b3.monthly_returns()`.

**Tempo de cada valor**
- **O índice de mês** (ou de dia) é acompanhado como constante ou como **símbolo mais constante**. Os símbolos são a variável de um laço `for` ou um parâmetro inteiro: `m`, `m - 1`, `m - 12`.
- **Tensores sobre um eixo temporal** guardam que a linha `t` é conhecida em `end(t + defasagem)`:
  - `b3.monthly_returns()` e `b3.cdi()`: `end(t)`;
  - `b3.eligible()`: `end(t − 1)`, porque o universo NEFIN do ano é fixado com a liquidez do ano anterior;
  - `observed(x, lag: k)`: declara `end(t + k)` para qualquer tensor, como dados próprios.
- **Seleção de linhas:** `x[month: e]` é conhecido em `end(e + defasagem)`; um intervalo `x[month: a ..= b]`, em `end(b + defasagem)`.
- **Operações** levam o mais tardio dos operandos.
- **O histórico inteiro:** reduzir sobre o eixo temporal (`mean(ret, over: month)`), ranquear ao longo dele ou listar todas as linhas só fica conhecido **no fim dos dados**.
- **Fluxo de controle conta:**
  - uma atribuição dentro de um `if`/`match` depende da condição;
  - depois de `if x { continue }` (ou `break`, `return`), todo o resto depende de `x`.
- **Laços** são analisados até um ponto fixo: o que uma iteração deixa para a seguinte fica **um mês mais velho** (`end(m)` vira `end(m − 1)`).

**Onde as decisões são checadas**
- **`b3.evaluate(w)`:** os pesos do mês `t` são negociados no fechamento de `t − 1`, então a linha `t` de `w` só pode usar o que se sabe em `end(t − 1)`. Quando o código grava `w[month: m] = v`, o compilador compara o tempo de `v` com `end(m − 1)`.
- **`fn f(...) -> T @ end(e)`:** o resultado só pode usar o que se sabe em `end(e)`, com `e` sendo um parâmetro mais uma constante.
  - Dentro de `f`, a linha `t` de um tensor temporal recebido é tratada como conhecida em `end(t)`.
  - Na chamada, o resultado vale `end(e)` com os argumentos substituídos.

**Erros**

| código | situação |
|---|---|
| E0301 | uso de informação do futuro |
| E0302 | não dá pra provar: o mês não é a variável do laço ou um parâmetro mais uma constante. Ainda é erro; a checagem em tempo de execução com aviso W0300 fica para depois |
| E0303 | um tensor cujas linhas são conhecidas depois do que a função anotada assume |

**Consequência:** escolher uma estratégia pelo resultado de `b3.evaluate` sobre o histórico e avaliar a escolhida nesse mesmo histórico também é E0301, porque a escolha só é conhecida no fim dos dados. Esse é exatamente o autoengano do §7.

**Ainda não:**
- defasagem declarada por parâmetro de função;
- dados diários na ponte;
- datas de publicação de proventos e balanços (chegam com novas fontes de dados).

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

### 7.1 O que o M5 implementa (v0.5.0)

A sintaxe ficou mais simples que o esboço acima. Os dados continuam vindo de `b3.*`, e o protocolo é que decide o que cada programa enxerga:

```krs
protocol "momentum-demo" {
    research: until 202610          // último mês da pesquisa (AAAAMM)
    holdout: from 202611            // primeiro mês do holdout
    min_months: 12                  // opcional (padrão 12): o holdout só abre com 12 meses de dados
    rule: "accept if the holdout's CAPM alpha > 0 and its PSR >= 90%"
}

fn weights() -> Tensor[month, asset; f64] { ... }                // lê b3.monthly_returns() etc.

let research = b3.evaluate(weights(), cost_bps: 25.0, lag: 0)    // só a pesquisa existe aqui
let h = b3.holdout("momentum-demo")                              // h: Holdout (afim)
let final = open_once(h, || weights(), cost_bps: 25.0, lag: 0)   // o holdout só existe aqui dentro
```

**O protocolo cerca os dados.**
- **No próprio programa:** quem declara um protocolo só enxerga os meses até `research: until`. Isso vale para `b3.monthly_returns()`, `b3.eligible()`, `b3.cdi()`, `b3.months()` e `b3.month_index()`; os tensores simplesmente terminam ali.
- **Em todos os programas:** o protocolo é registrado com hash no livro-razão `results/holdout_ledger.csv` na primeira execução. Enquanto o holdout não for aberto, os meses dele ficam escondidos de **todos** os programas, e uma nota avisa onde os dados param.
- **Com defasagem:** com `lag: k > 0`, a carteira do último mês visível precisaria dos primeiros pregões do mês seguinte, então ela não é negociada.

**O holdout abre uma vez.**
- **Na compilação:** `Holdout` é afim. `open_once` o consome, e usá-lo de novo, dentro de um laço ou capturado por uma lambda é **E0501**.
- **Uma lambda só:** `open_once` recebe a *função* que calcula os pesos, não os pesos. Os meses do holdout passam a existir enquanto ela roda. Fora dela, a cerca continua.
- **A avaliação do holdout:**
  - é uma carteira nova a partir do primeiro mês do holdout;
  - não entra no registro de tentativas: vem com `trials = 1`;
  - o `dsr` dela é a probabilidade de o Sharpe ser positivo (PSR).
- **Entre execuções:** o livro-razão grava a abertura antes de qualquer mês do holdout ser visto, e depois o resultado. Rodar de novo dá **R0502** já em `b3.holdout(...)`, antes da pesquisa, com a data e o resultado da abertura.
- **Se a função de pesos falhar durante a abertura,** o holdout está gasto do mesmo jeito: a falha fica registrada.
- **R0503:** se o holdout ainda não tem `min_months` meses de dados, nada é aberto e nada é registrado.

**Um holdout novo começa depois de tudo o que já foi visto.**
- **A marca d'água:** o livro-razão guarda o último mês que qualquer programa Kairós (ou abertura) viu.
- **R0502 no registro:** um protocolo cujo holdout comece até essa marca é recusado na primeira execução.
- **O livro-razão nasce com a busca do laboratório:** `quant-b3 busca` já abriu 2019-01..2026-10 (protocolo `results/protocolo_busca.md`). Por isso hoje **o único holdout limpo é o futuro**, a partir de 2026-11.
- **Abrir um holdout que se sobrepõe a outro pendente** estraga o outro, que passa a dar R0502.
- **`kairos ledger`** mostra o livro-razão, e **`kairos ledger withdraw <nome>`** desiste de um protocolo nunca aberto (os meses dele deixam de ser cercados).

**Toda avaliação de pesquisa é uma tentativa.**
- **O registro:** cada `b3.evaluate` entra em `results/registro_testes.csv`, o mesmo registro das estratégias que o laboratório testou em Rust.
- **A chave:** nome do programa, hash dos pesos e opções.
- **Sem contar duas vezes:** uma série de retornos idêntica a uma já registrada (mesmo n, Sharpe, assimetria e curtose) não conta de novo.
- **Os campos novos de `Evaluation`:**
  - `trials` é o tamanho do registro;
  - `dsr` é o Deflated Sharpe Ratio (Bailey & López de Prado), a probabilidade de o Sharpe verdadeiro superar o melhor de `trials` tentativas feitas sem habilidade nenhuma.
- **Sem pasta de resultados** (testes, uso embutido), o livro-razão e o registro ficam em memória. `KAIROS_RESULTS` escolhe outra pasta.

**Erros novos:**

| código | situação |
|---|---|
| E0501 | holdout usado duas vezes, num laço ou numa lambda |
| R0500 | livro-razão ou registro ilegível ou impossível de gravar (nada roda sem eles) |
| R0502 | holdout já aberto, protocolo retirado, ou holdout que começa em meses já vistos |
| R0503 | o holdout ainda não tem `min_months` meses de dados (nada é gravado) |
| R0504 | `b3.evaluate` ou `open_once` dentro da função de pesos de um `open_once` |
| R0505 | registro da falha da função de pesos durante a abertura |

O E0502 do esboço virou R0502: quem detecta a segunda abertura entre execuções é o runtime, não o compilador.

**Limites:**
- **O livro-razão mantém honesto quem quer ser honesto.** É um arquivo comum, que o dono pode editar.
- **Ele só sabe o que programas Kairós viram.** O que alguém olhou em planilha, Python ou nas ferramentas em Rust do laboratório fica de fora; por isso o protocolo bom é o que reserva o futuro.

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

### 8.1 O que o M4 implementa (v0.4.0)

**`grad` reverso, sobre o caminho executado** (`src/autodiff.rs`)
- `grad(f)`, `grad(f, wrt: k)` e `value_and_grad(f)`: `f` devolve `f64` (uma perda), e o resultado é uma função com os mesmos parâmetros que devolve o gradiente (e o valor).
- Durante a chamada, os `f64` e tensores de `f64` do argumento viram valores rastreados numa fita. `if`, laços, chamadas e lambdas simplesmente rodam, e a fita é percorrida de trás pra frente a partir da perda.
- Derivadas implementadas:
  - `+ − × ÷ ^` e negação, com broadcasting por nome de eixo (o gradiente soma sobre os eixos repetidos);
  - `exp`, `ln`, `sqrt`, `abs`, `tanh`, `sigmoid`, `relu`, `softplus`, `fill_nan`;
  - reduções `sum`, `mean`, `prod`, `max`, `min`, `nan_sum`, `nan_mean`;
  - seleção `x[month: i]`, atribuição em fatia `x[month: i] = v`, `where`, `max`/`min` de dois valores;
  - `contract`, cujos gradientes também são contrações;
  - `tensor([...])` e `to_list`.
- O que não é diferenciável (contagens, ranks, arredondamentos, comparações) entra como constante.
- `grad` dentro de `grad` ainda não existe (R0401).

**Tipos diferenciáveis:** `f64`, tensores de `f64`, e structs, tuplas e listas deles. O gradiente tem a mesma forma do parâmetro. Fora disso é E0402; uma função que não devolve `f64` dá E0401.

**Espaço tangente:**
- `+ − × ÷` funcionam campo a campo entre dois valores do mesmo tipo, ou entre um deles e um `f64`: `p - 0.01 * g`.
- `sqrt`, `exp`, `ln`, `abs`, as ativações e `zeros_like` também valem para structs.

**Funções genéricas:** `fn adam_step[P](inout p: P, g: P, ...)`.
- Cada parâmetro de tipo representa um tipo diferenciável.
- Em cada chamada ele é instanciado, e o tipo concreto é conferido (E0402).

**Regras suaves:**
- `smooth_max(x, y, scale: s)` e `smooth_min` viram softplus dentro do `grad`;
- `smooth_step(x, scale: s)`, o degrau, vira sigmoide dentro do `grad`;
- fora do `grad` valem as regras exatas.

**Aleatoriedade reprodutível:** `randn(seed: k, feature: 4, hidden: 16)` usa SplitMix64, xoshiro256** e Box-Muller, todos em Rust puro, e dá os mesmos números em qualquer sistema.

**Motor de IA escrito em Kairós** (`std/prelude.krs`):
- eixos `sample`, `feature`, `hidden`, `output`;
- `Mlp`, `mlp_init`, `mlp`, `mse`;
- `sgd_step[P]` e `adam_step[P]`, genéricos.

**Aceitação**
- `tests/programs/grad.krs`: os gradientes conferem com diferenças finitas (diferença de até 10⁻¹⁰).
- `tests/programs/mlp_learns.krs`: um MLP de 16 unidades aprende `y = x³ − x`. A perda vai de 0,566 para 0,00017 em 1.500 passos de Adam, conferida até o último bit em todos os sistemas da CI.

**Ainda não:**
- `if … smooth` como sintaxe;
- ordenação e `top_k` diferenciáveis;
- segunda derivada;
- gradiente através do simulador `b3.evaluate`. Este último é o objetivo do M6.

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

### 9.1 O que o M7 implementa (v0.7.0)

**Código nativo para funções escalares puras**, com Cranelift ([`src/native.rs`](../src/native.rs)).

- **O que é compilado:** na primeira chamada, uma função cujos parâmetros, variáveis e resultado são `i64`, `f64` ou `bool` vira código de máquina. Isso vale para funções com:
  - aritmética, comparações, `and`/`or`/`not`;
  - `if`, `while`, `for`, `break`/`continue`/`return`;
  - chamadas entre elas, inclusive recursivas;
  - `sqrt`, `exp`, `ln`, `abs`, `floor`, `ceil`, `round`, `float`, `int`, `min` e `max`.
- **O que fica no interpretador:** o resto (tensores, listas, structs, `inout`, `grad`, `print`). As duas formas se chamam livremente.
- **Mesma semântica, mesmos bits:**
  - a aritmética de `i64` é verificada, com os mesmos erros e mensagens: R0006, R0002, R0007 e R0009;
  - `exp`, `ln`, `^` e `round` chamam as mesmas funções em Rust puro que o interpretador;
  - NaN segue as mesmas regras;
  - a profundidade de chamadas tem o mesmo limite (R0005);
  - [`tests/native.rs`](../tests/native.rs) roda cada programa com e sem JIT e exige saída e erros idênticos.
- **A convenção de chamada nativa:**
  - os escalares e o orçamento de chamadas aninhadas vão em registradores;
  - o retorno é (resultado, falhou?), e os detalhes do erro só são gravados no caminho de erro.
- **Plataformas:**
  - **com JIT:** x86_64 em Linux, macOS, Windows e FreeBSD, e ARM64 em Linux e macOS;
  - **sem JIT:** nas demais (32 bits, RISC-V, PowerPC, s390x, BSDs, WebAssembly) tudo roda no interpretador, com os mesmos resultados;
  - `KAIROS_JIT=0` desliga o JIT.

**`kairos-bench`** ([`src/bin/kairos_bench.rs`](../src/bin/kairos_bench.rs), programas em [`bench/`](../bench/)): os mesmos seis programas em Kairós e em Rust `--release`, com a mesma semântica (`checked_*` e as mesmas `exp`/`ln`). Os resultados batem até o último bit. Tempos medidos num Intel Core i7-12700KF, melhor de 7 execuções:

| benchmark | o que mede | interpretador | nativo (JIT) | Rust | nativo ÷ Rust |
|---|---|---|---|---|---|
| `fib(30)` | chamadas recursivas | 315 ms | 3,8 ms | 3,1 ms | 1,25× |
| `sum_sqrt` (20 mi) | laço de f64 com `sqrt` | 1,95 s | 24,2 ms | 24,1 ms | 1,00× |
| `collatz` (100 mil) | inteiros verificados com desvios | 1,62 s | 21,5 ms | 14,4 ms | 1,50× |
| `black_scholes` (300 mil) | finanças: `exp`, `ln`, `sqrt`, CDF normal | 647 ms | 11,0 ms | 9,1 ms | 1,20× |
| `mandelbrot` (300×200) | laços aninhados de f64 com saída antecipada | 950 ms | 6,3 ms | 5,3 ms | 1,19× |
| `monte_carlo` (2 mi) | gerador Park-Miller + f64 | 747 ms | 16,9 ms | 15,0 ms | 1,13× |
| **média geométrica** | | | | | **1,20×** |

**A meta de ≤ 1,2× Rust foi atingida na média geométrica.** Por programa, vai de 1,00× a 1,50×:
- **Onde o Cranelift perde para o LLVM:** a pior distância é nas contas inteiras verificadas, em que o Cranelift gera os testes de estouro menos enxutos.
- **Recursão:** a distância vem da contagem de profundidade, que o interpretador também faz.

**Ainda não:**
- compilação antecipada (AOT);
- tensores no código nativo (os kernels de tensor já são Rust);
- o elemento ternário `t3`, que fica para depois;
- comparação com C, Julia e NumPy.

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

**Acrescentado no M8**, para escrever o laboratório em Kairós:

| Kairós | o que devolve |
|---|---|
| `b3.daily_returns()`, `daily_volume()`, `daily_price()`, `daily_dividends()` | `Tensor[month, session, asset; f64]`: o pregão `session` (até 23 por mês), com a linha m conhecida em `end(m)` |
| `b3.sessions()` / `b3.session_dates()` | os pregões de cada mês (`Tensor[month, session; bool]`) e suas datas (AAAAMMDD) |
| `b3.issuers()` / `b3.classes()` | a empresa e a classe de cada ativo |
| `b3.nefin(fator, vw: false)` / `b3.nefin_daily(...)` / `b3.ipca()` | fatores do NEFIN (mensais e diários) e a inflação |
| `b3.simulate(w, ...)` | `Simulation`: o simulador mês a mês (retorno, exposição, giro, custo, IR, valor final, meses isentos); conta como tentativa, como `b3.evaluate` |
| `b3.record_trial(estudo, nome, parâmetros, período, excesso)` | grava uma série no registro e devolve a linha (Sharpe, tentativas, acaso, DSR) |
| `b3.deflate_trial(excesso)` | `[tentativas, Sharpe do acaso, DSR]` contra o registro, sem gravar |
| `b3.results_sha256(arquivo)` | o SHA-256 de um arquivo da pasta de resultados (protocolos registrados) |

Opções de `b3.evaluate` e `b3.simulate`:
- `cap_adv: d` limita cada posição a `d` dias do volume mediano;
- `trial: false` marca uma referência que não conta como tentativa, visível no código.

Uma estratégia é uma tentativa: os mesmos pesos executados de novo, com outro custo, atraso, capital ou IR, não contam outra vez.

**Ainda em Rust**, para os próximos marcos: o paper trading (`paper`), a coleta e a montagem dos dados e o simulador.

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
| M3 | tipos temporais | ✅ o momentum continua idêntico ao `quant-b3` e a versão que olha `ret[month: m]` (o futuro) **não compila** (E0301); 14 testes de vazamento: `if`, `continue`, laços, estatística do histórico inteiro, funções anotadas, escolha por backtest |
| M4 | `grad` nativo + `smooth` + motor de IA em Kairós (MLP, MSE, SGD, Adam) | ✅ gradiente confere com diferença finita (até 10⁻¹⁰); o MLP escrito em Kairós aprende `y = x³ − x` (perda 0,566 → 0,00017), com os mesmos bits em todos os sistemas |
| M5 | holdout de uso único + protocolo + registro e DSR automáticos | ✅ 2º acesso = erro (E0501 na compilação, R0502 entre execuções, pelo livro-razão); o protocolo cerca os dados de pesquisa e reserva o holdout para todos os programas; toda avaliação vira tentativa no `registro_testes.csv` com DSR; 7 testes dourados e 5 de integração num mercado sintético |
| M6 | primeiro modelo inédito (carteira com IR suave) | ✅ protocolo publicado antes (commit `52a186d`, `research/m6-protocol.md`); walk-forward provado causal; simulador real com custos e IR; 3 tentativas no registro; DSR 0,8%: **rejeitado** pela própria regra ([relatório](../research/m6-report.md)); holdout (o futuro, ≥ 24 meses) reservado no livro-razão |
| M7 | back-end nativo Cranelift + `kairos-bench`; candidato: elemento `t3` (pesos ternários {−1, 0, +1}, 5 por byte, matmul só com soma e subtração, como no BitNet b1.58), com kernel CPU e CUDA | ✅ JIT de funções escalares puras, com os mesmos bits e erros do interpretador; `kairos-bench`: 1,20× Rust na média geométrica de 6 programas (1,00× a 1,50×); `t3` fica para depois (§9.1) |
| M8 | o laboratório escrito em Kairós: `import`, dados diários, `b3.simulate`, registro (§10) | ✅ os 5 estudos do `quant-b3` (`lab/*.krs`) imprimem o relatório do Rust linha por linha; universo NEFIN igual (41.106 pares); estatística bit a bit; o registro não muda |
| M9 | arquivos e texto: leitura e escrita, CSV, JSON, datas | o paper trading e a coleta de dados escritos em Kairós |
| M10 | o simulador em Kairós (custos, impacto, IR) | os relatórios continuam idênticos com o simulador escrito em Kairós |
| M11 | auto-hospedagem | o compilador da Kairós, escrito em Kairós, compila a si mesmo |
