# Changelog

Versões da linguagem Kairós (pasta `kairos/`). Cada versão vira uma release no GitHub, gerada automaticamente por `.github/workflows/release.yml` a partir da seção correspondente deste arquivo.

## 0.22.0 — M22: o código nativo mais rápido

**Os executáveis que o compilador escrito em Kairós gera ficaram 2 a 3 vezes mais rápidos e com metade do tamanho, com os mesmos números.** O tradutor do M21 ([`native.krs`](kairos/compiler/native.krs)) gravava cada valor na memória a cada instrução. Agora ele adia isso:

- **A pilha de valores preguiçosa.** Um valor fica como constante, como o valor de um local, no topo em `rax` ou `xmm0`, ou como comparação nos flags, até uma instrução precisar dele:
  - um operando constante vira imediato;
  - o resultado de uma conta segue no registrador para a próxima;
  - uma comparação vai direto para o salto (`cmp` + `jl`).
  - Os valores só vão para a memória nos saltos, nas fronteiras de bloco e nas chamadas.
- **Os locais em registradores.** Os seis locais mais usados de cada função (contados no código, oito vezes mais dentro de cada laço) vivem em `rbx`, `rsi`, `rdi`, `r12`, `r13` e `r14`.
- **Medido:**
  - o `kn.exe` compila a si mesmo e o comando `kairos` em 11 s (eram 31 s) e tem 3,1 MB (eram 5,9);
  - o `momentum` com os dados da B3 caiu de 6,5 s para 2,2 s. O V8, que otimiza o mesmo módulo, faz 1,7 s; o interpretador Rust, com a ponte da B3 escrita em Rust, 1,5 s;
  - um laço de inteiros e floats leva 0,12 s, contra 0,47 s no interpretador Rust com o JIT do M7.
- **Um erro achado pelo ponto fixo:** uma chamada sem argumentos destruía o valor guardado em `rax` (`x + f()`), e os 62 programas de teste não tinham esse caso. Está corrigido e virou o teste `tests/wasm/native_calls.krs`.
- **Conferido:**
  - os 63 programas de `tests/wasm`, os 31 dourados que rodam e os 10 com os dados da B3 (com o mesmo registro e o mesmo livro-razão) saem iguais ao interpretador;
  - o comando `kairos` como executável é igual ao Rust;
  - o `kn.exe` compila a si mesmo nos mesmos bytes.
- **Próximo (M23):** executáveis ELF para Linux.

## 0.21.0 — M21: o back-end nativo escrito em Kairós

**O compilador escrito em Kairós passou a gerar executáveis do Windows:** código de máquina x86-64 que ele mesmo escreve, sem Rust e sem runtime de WebAssembly (wasmtime, node). O próprio compilador vira um executável que compila a si mesmo, nos mesmos bytes.

```
kairos run kairos/compiler/kn.krs kairos/std out programa.krs    # out/programa.exe
```

- **O caminho:** o front-end em Kairós; o `wasmgen.krs`, que faz o módulo de WebAssembly; e então três arquivos novos, todos em Kairós:
  - [`native.krs`](kairos/compiler/native.krs) traduz o módulo para x86-64, função por função: cada valor da pilha do WebAssembly tem um lugar no quadro, e a memória do módulo começa em `r15`;
  - [`x64.krs`](kairos/compiler/x64.krs) codifica as instruções;
  - [`pe.krs`](kairos/compiler/pe.krs) grava o executável (PE32+).
- **O sistema operacional também é Kairós:** [`rt_win.krs`](kairos/compiler/rt_win.krs) implementa sobre a `kernel32.dll` as 13 funções do WASI que o runtime usa. São os arquivos e as pastas, o console (em UTF-16, para mostrar os acentos), os relógios, os argumentos e o ambiente.
- **Conferido:**
  - os 62 programas de `tests/wasm`, os 31 dourados que rodam e os 10 com os dados reais da B3 dão a saída, os erros e o código de saída do interpretador;
  - o registro de tentativas e o livro-razão saem iguais byte a byte;
  - o comando `kairos` escrito em Kairós, como executável, é igual ao Rust nas 59 linhas e nas 3 sessões de REPL do teste do M16;
  - o `kn.exe` (5,9 MB) compila a si mesmo em 24 s nos mesmos bytes.
- **Testes:** `tests/native_x64.rs` (Windows x86-64, release) e o job `native-x64` do CI. A release traz `kairos-selfhost.exe`, o comando escrito em Kairós como executável.
- **Limites:**
  - o código ainda é o de base, com cada valor passando pela memória: o `momentum` leva 6,5 s, contra 1,5 s no interpretador Rust (M22: registradores);
  - por enquanto, só Windows x86-64 (M22: ELF para Linux).

## 0.20.0 — M20: texto formatado

**A Kairós ganhou o texto formatado:** `f"{name} {ym}: equity {equity / 1e9:.1} bi"`. Antes, o mesmo texto era uma soma de pedaços com `str(...)` e `format_fixed(..., 1)`.

- **A sintaxe:**
  - `{expr}` é o texto da expressão (o de `str`), e `{expr:.N}` o número com N casas (o de `format_fixed`);
  - `{{` e `}}` são chaves, e os escapes são os dos textos comuns;
  - a expressão não tem aspas nem quebras de linha.
- **Como funciona:** o léxico escreve o texto formatado como os tokens da concatenação. O parser, os tipos, a análise temporal e os back-ends não mudam. Os tokens das expressões guardam o lugar no arquivo, então um erro dentro de `{}` aponta para ele.
- **Erros novos, no léxico (E0001):** um `}` sozinho, um `{` sem fechar, um `{}` vazio, duas expressões num `{}` e casas demais.
- **As implementações:** os dois léxicos (Rust e Kairós) dão os mesmos tokens e erros em todos os arquivos do repositório. O Rust, o WebAssembly e o interpretador em Kairós dão a mesma saída.
- **O laboratório:** [`lab/check_cvm.krs`](kairos/lab/check_cvm.krs) foi reescrito com textos formatados e imprime o mesmo relatório.
- **O editor:** a extensão do VS Code (`editors/vscode`, 0.13.0) colore o `f`, as chaves, o formato e as expressões dentro delas. Também passou a conhecer o tipo `t3`, os escapes `\r`, `\0` e `\u{...}` e os builtins que faltavam (`ternary`, `format_fixed`, `read_line`, `zip_entries` e outros).
- **Testes:**
  - 10 casos do léxico (`tests/lexer/fstr_*.krs`);
  - o programa dourado `tests/programs/fstrings.krs` e dois programas de erro;
  - `tests/wasm/fstrings.krs`.

## 0.19.0 — M19: os demonstrativos da CVM, no ponto do tempo

**O laboratório ganhou informação nova: os demonstrativos financeiros das companhias abertas,** dos dados abertos da CVM (DFP desde 2010, ITR desde 2011), escritos em Kairós. Cada número só é conhecido a partir do dia em que a CVM recebeu a entrega que o traz. E a linguagem passou a distinguir o tempo que ela prova do tempo que não conhece.

- **Os dados:**
  - [`lab/fetch_cvm.krs`](kairos/lab/fetch_cvm.krs) baixa os 31 arquivos (~600 MB, 20 s);
  - [`lab/build_cvm.krs`](kairos/lab/build_cvm.krs) lê 42 mil entregas em 35 s: ativo, patrimônio, receita, lucro e ações, consolidados quando existem;
  - `cvm_fundamentals("data", n)` ([`lab/cvm.krs`](kairos/lab/cvm.krs)) dá, mês a mês, o que cada companhia tinha informado até então, com receita e lucro dos últimos doze meses. Liga 1.490 dos 2.241 ativos do painel às suas companhias.
- **O ponto do tempo:** os arquivos guardam só a última versão de cada entrega, então uma reapresentação vale do dia em que foi recebida. O BRB entregou o DFP de 2023 seis vezes, de abril a julho de 2024: seus números valem desde julho. A data da primeira versão vazaria a reapresentação 3,5 meses antes. Uma conferência independente achou 0 valores, de 142.648, vindos de entregas recebidas depois do fim do seu mês.
- **A linguagem:**
  - builtins `zip_entries(p)` e `read_zip_lines(p, entrada, keep)`, mais as variantes `try_`, para ler um arquivo de dentro de um ZIP pelo nome;
  - **o que vem de fora tem uma marca própria.** Um valor de arquivo, da rede ou da entrada não tem mais o tempo do "fim dos dados" (o de quem usa o histórico inteiro), e sim uma marca à parte, que nenhuma decisão aceita: E0301 "use of information whose time is not known";
  - `observed(x, lag: k)` tira a marca e declara quando cada linha é conhecida, mas não apaga um uso do futuro que o compilador prova (`observed(ternary(ret))` continua sendo erro). Antes, a declaração não liberava dado de arquivo nenhum.
- **As três implementações** (Rust, WebAssembly e o interpretador escrito em Kairós) dão a mesma saída nos builtins de ZIP. Os dois front-ends dão a mesma análise temporal em todos os arquivos.
- **Testes:**
  - `tests/wasm/zip_entries.krs`;
  - `observed_declares_the_time_of_what_comes_from_outside` (em `tests/temporal.rs`);
  - `tests/temporal/fundamentals_*.krs`;
  - três casos de erro do verificador;
  - [`lab/check_cvm.krs`](kairos/lab/check_cvm.krs) (cobertura, a Petrobras mês a mês e a regra do tempo).
- **Nada foi avaliado:** os dados estão prontos para pesquisa, mas uma hipótese nova precisa de um protocolo registrado antes. Como os meses de 2026-11 em diante estão reservados ao holdout do M6, a validação de uma estratégia com fundamentos fica para o futuro.

## 0.18.0 — M18: o elemento ternário `t3`

**A Kairós ganhou um tipo para valores de {−1, 0, +1}.** São os pesos do BitNet b1.58, matrizes ternárias que se multiplicam só com somas e subtrações. Também é a forma natural de uma carteira comprada, fora ou vendida. `t3` é um tipo escalar e um elemento de tensor (`Tensor[feature, hidden; t3]`).

- **Construir:**
  - `ternary(x)`, a quantização do BitNet b1.58: `round(x / γ)` limitado a [−1, 1], com γ a média de |x|. NaN vira 0;
  - `ternary(x, threshold: t)`, a carteira de um sinal: +1 acima de `t`, −1 abaixo de `−t`, 0 no resto.
- **No tempo:** a escala de `ternary(x)` usa o tensor inteiro. Sobre o histórico, numa decisão, a análise temporal recusa (E0301). Mês a mês, ou com `threshold:`, passa.
- **Usar:**
  - `contract` com um lado `t3` só soma e subtrai: o retorno de uma carteira sem nenhuma multiplicação;
  - `sum` e `mean` dão a exposição líquida;
  - `float(w)`, `-w`, `==`, índices, fatias, atribuição, `where`, `to_list` e `print`;
  - um `t3` sozinho entra em listas, tuplas, structs e funções.
- **Guardar:** no interpretador Rust, um tensor `t3` guarda cinco trits por byte (1/40 de um `f64`).
- **Treinar:** dentro de `grad`, `ternary` passa o gradiente como a identidade (o estimador *straight-through*). [`examples/ternary.krs`](kairos/examples/ternary.krs) treina uma camada ternária que acha os trits exatos de um alvo em 10 passos, e monta uma carteira comprada/fora/vendida.
- **As três implementações:** o interpretador Rust, o back-end WebAssembly e o interpretador escrito em Kairós dão a mesma saída, bit a bit. O verificador escrito em Kairós dá os mesmos tipos, HIR e erros.
- **Testes:**
  - `tests/programs/ternary.krs` e `tests/wasm/t3.krs` cobrem NaN, infinito, tudo zero, o meio do arredondamento, contrações com eixos de lote, `t3` em structs e listas, e `grad`;
  - 23 casos de erro em `tests/checker/t3_*.krs` e dois casos temporais;
  - um teste de unidade cobre o empacotamento de cinco trits por byte.
- **Limites:** no WebAssembly, um tensor `t3` ainda ocupa um `f64` por trit. Dentro de `grad`, um `t3` que vem de um tensor rastreado aparece como o `f64` correspondente.

## 0.17.0 — M17: um alocador que reaproveita a memória

**Os programas compilados para WebAssembly passaram a usar uma fração da memória, com os mesmos números.** O `momentum` sobre os dados reais da B3 terminava com 1.565 MB, dos quais 1.283 MB estavam livres mas presos: cada bloco liberado ficava na lista do seu tamanho (uma potência de 2) e nunca era juntado a outro, então uma lista que cresce dobrando deixava um rastro de blocos de todos os tamanhos. Agora o mesmo programa termina com 542 MB (−65%) e roda um pouco mais rápido. O modelo do M6 (treinado com `grad`) caiu de 2,75 GB para 1,1 GB.

O alocador do runtime ([`compiler/rt.krs`](kairos/compiler/rt.krs)) continua escrito em Kairós:

- **Objetos pequenos** (até 32 KiB) continuam em blocos de 2^k bytes, com uma lista livre por tamanho. Uma lista vazia corta uma placa de 64 KiB tirada dos blocos grandes.
- **Blocos grandes** (listas, `Bytes`, textos e mapas de mais de 32 KiB, e as placas) têm o tamanho pedido, arredondado para 8. Os livres ficam numa lista em ordem de endereço, e dois blocos livres que se tocam viram um. Um pedido leva o fim do primeiro bloco livre que caiba. Sem nenhum, a memória vem do fim do heap, e um bloco livre que termina ali cresce para dentro dela.
- **Uma lista ou um `Bytes` que cresce** cresce no lugar, sem cópia, quando a memória logo depois dele está livre ou é o fim do heap.
- **Um arquivo é lido num bloco do tamanho dele.** O `panel.bin` de 83 MB ocupava 128 MB.

Testes:
- `tests/wasm/heap.krs` cobre listas que crescem juntas, blocos liberados e tomados de novo, `Bytes`, textos e mapas grandes, com a mesma saída do interpretador;
- o teste `b3` do back-end falha se o `momentum` voltar a passar de 750 MB;
- o compilador continua compilando a si mesmo num ponto fixo.

## 0.16.0 — M16: o comando `kairos` escrito em Kairós

**O comando `kairos` inteiro agora também existe em Kairós.** [`compiler/kairos.krs`](kairos/compiler/kairos.krs) faz o que o executável Rust faz, com a mesma saída, os mesmos erros e o mesmo código de saída. Compilado para WebAssembly pelo compilador em Kairós, vira um `kairos.wasm` sem nenhuma linha de Rust, que roda em qualquer runtime WASI. Esta release traz esse arquivo pronto (`kairos-0.16.0-selfhost-wasm.tar.gz`).

- **Os subcomandos:**
  - `run`, `check` e o atalho `kairos programa.krs`, que também serve a scripts com `#!/usr/bin/env kairos`;
  - `repl` (e `kairos` sozinho), o REPL;
  - `exec`, que roda a imagem gravada pelo `kc.krs`. A leitura da imagem foi escrita em Kairós ([`compiler/image_read.krs`](kairos/compiler/image_read.krs)), com as mensagens de erro do leitor Rust;
  - `ledger` e `ledger withdraw`, o livro-razão dos holdouts;
  - `tokens`, `ast`, `types` e `hir`;
  - `--version`.
- **A biblioteca padrão vai dentro do programa**, como no executável Rust. [`compiler/gen_std.krs`](kairos/compiler/gen_std.krs) gera [`compiler/std_embed.krs`](kairos/compiler/std_embed.krs) a partir de `kairos/std` e da versão do `Cargo.toml`, e um teste avisa quando o arquivo fica velho.
- **O REPL** é igual ao do Rust:
  - o prompt `kai> ` e a continuação `...  ` enquanto houver `(`, `[` ou `{` aberto;
  - `:help` e `:q`;
  - cada entrada é verificada contra tudo o que veio antes e roda no mesmo quadro de variáveis;
  - o valor de uma expressão aparece como `= valor`;
  - uma entrada com erro de verificação deixa a sessão como estava;
  - os `import` de uma entrada são verificados e rodados antes dela.
- **Builtins novos na linguagem:** `read_line()` lê uma linha da entrada padrão, com a quebra de linha (`""` no fim da entrada). `print_raw(s)` escreve sem quebra de linha.
- **Testes:**
  - `cargo test --release -p kairos --test wasm the_command_written` compila o comando para WebAssembly e o compara com o executável Rust. Cobre os subcomandos sobre programas de teste, o `exec` de imagens (inclusive malformadas) e sessões de REPL dadas pela entrada padrão;
  - o hospedeiro WASI dos testes ganhou entrada padrão;
  - no CI, este teste e o do interpretador em Kairós rodam num job próprio em release (`selfhost`), que também roda o comando escrito em Kairós no wasmtime. Em debug eles demoravam demais.
- **Limites conhecidos:**
  - o hospedeiro precisa de uma pilha funda para programas que recursam fundo;
  - a memória do WebAssembly continua grande com os dados da B3. O alocador que reaproveita memória entre tamanhos passou para o M17.

## 0.15.0 — M15: o interpretador escrito em Kairós

**A Kairós agora tem um interpretador escrito nela mesma.** [`compiler/interp.krs`](kairos/compiler/interp.krs) roda o HIR do front-end em Kairós como o interpretador Rust roda o seu. A saída, os erros com as mesmas mensagens, os números e o código de saída são os mesmos. [`compiler/ki.krs`](kairos/compiler/ki.krs) é o equivalente de `kairos run`. Compilado para WebAssembly pelo compilador em Kairós, o `ki.wasm` roda programas Kairós sem nenhum Rust: junto com o `kw.wasm`, que compila (e compila a si mesmo), a linguagem se basta.

- **Tudo o que as suítes de teste cobrem roda igual:**
  - a linguagem inteira, tensores e `grad` (a fita de `rt_diff.krs`, gravada nó a nó como a do Rust, com os mesmos gradientes, bit a bit);
  - arquivos, rede e `Bytes`;
  - a ponte `b3`, os protocolos e os holdouts de uso único: o `momentum` roda sobre os dados reais da B3 e grava o mesmo registro de testes e o mesmo livro-razão;
  - 158 programas de teste, inclusive os 55 que usam `b3`. As únicas diferenças vêm do hospedeiro WebAssembly: o texto de um erro do sistema, e o `list_dir`, que o node no Windows não implementa.
- **Como funciona:**
  - os valores são uma struct, não um enum, para listas, mapas e structs mudarem no lugar, copiados só quando outro valor os compartilha (a semântica de valor da própria linguagem);
  - o controle de fluxo (`return`, `break`, `continue` e os erros) é um estado do interpretador;
  - tensores, `grad` e a ponte `b3` reaproveitam o runtime em Kairós do back-end WebAssembly.
- **A ponte `b3` escrita em Kairós** (`rt_b3.krs`) passou a ter o núcleo com o estado explícito. Os programas compilados e o interpretador usam o mesmo código. O `open_once` foi dividido em abertura, fechamento e falha. As estatísticas do simulador recebem o CDI e o NEFIN do estado (`sim_eval_on`).
- **Builtins novos na linguagem:**
  - `eprint(...)`, que escreve como `print` no stderr, e `exit(código)`;
  - nove variantes que devolvem o erro em vez de parar o programa: `try_read_text`, `try_read_bytes`, `try_write_text`, `try_write_bytes`, `try_list_dir`, `try_make_dir`, `try_read_lines`, `try_http_get` e `try_download`. Os programas Kairós passam a poder tratar falhas de arquivo e de rede.
- **O verificador em Kairós** agora escreve o rótulo depois dos `^^^^` (os 11 rótulos do verificador Rust e da análise temporal) e guarda o arquivo do erro. Assim os erros de compilação saem como os do `kairos run`.
- **Testes:** `cargo test --release -p kairos --test wasm the_interpreter_written` compila o interpretador para WebAssembly e roda cada programa de teste pelos dois interpretadores; com os dados da B3, quatro programas `b3` também.
- **Limites conhecidos:**
  - o REPL e os outros subcomandos de `kairos` (`check`, `ledger`, `tokens`…) ainda são só do Rust;
  - o hospedeiro WebAssembly precisa de uma pilha funda: um interpretador recursa tão fundo quanto o programa que roda (no node, uma worker com `stackSizeMb`; no wasmi, `set_max_stack_height`);
  - a memória do WebAssembly continua grande com os dados da B3 (o alocador não junta blocos livres).

## 0.14.0 — M14: o back-end WebAssembly completo

**Os programas da Kairós rodam como WebAssembly com os mesmos números do interpretador, inclusive os que usam o laboratório, os holdouts e o `grad`.** O back-end escrito em Kairós ([`compiler/wasmgen.krs`](kairos/compiler/wasmgen.krs)) agora compila closures, tensores, a ponte `b3`, os protocolos e a diferenciação automática. O runtime dos módulos continua escrito em Kairós.

- **Os programas do laboratório, os mesmos bits:**
  - `examples/momentum.krs` compilado imprime os mesmos 9 números e grava a mesma linha no registro de testes e no livro-razão;
  - o modelo do M6 (`research/tax_portfolio.krs`, treinado com `grad` no patrimônio depois de custos e IR) dá a mesma saída, o mesmo registro e o mesmo livro-razão;
  - `examples/mlp.krs` treina 1500 passos de Adam com as mesmas perdas;
  - todo programa da suíte dourada do interpretador que compila dá a mesma saída e os mesmos erros, menos os textos do sistema operacional.
- **Closures e tensores:**
  - lambdas com capturas e funções como valores;
  - tensores com eixos nomeados, com o runtime em Kairós ([`rt_tensor.krs`](kairos/compiler/rt_tensor.krs)), operação por operação como o do interpretador.
- **A ponte `b3` escrita em Kairós** ([`rt_b3.krs`](kairos/compiler/rt_b3.krs), sobre [`kairos/lab`](kairos/lab)):
  - os dados do laboratório (`panel.bin`, o CDI, o IPCA, os fatores NEFIN mensais e diários, os tensores diários);
  - a avaliação e a simulação pelo simulador em Kairós;
  - o registro de testes com o DSR, `record_trial`, `deflate_trial` e o SHA-256 dos resultados;
  - a cerca do livro-razão de holdouts.
- **Protocolos e holdouts de uso único**, com as garantias do M5:
  - o programa vê só a pesquisa, e `b3.holdout` recusa um holdout já aberto (R0502);
  - `open_once` grava a abertura antes de o holdout ser visto;
  - um erro dentro da função dos pesos gasta o holdout, e o livro-razão grava a falha.
- **`grad`, `value_and_grad` e as regras suaves:**
  - a fita é escrita em Kairós ([`rt_diff.krs`](kairos/compiler/rt_diff.krs)), nó por nó como a do interpretador;
  - o back-end faz uma cópia rastreada de cada função que o `grad` alcança. Os floats e tensores viram nós, e cada operação calcula o valor comum com o código de sempre;
  - funções com nome e lambdas, o espaço tangente, enums e as atribuições em fatias;
  - `grad` dentro de `grad` para com R0401, como no interpretador.
- **O espaço tangente e as funções genéricas:** `p - lr * g` em structs, campo a campo, e `fn f[P]` instanciada para cada tipo.
- **Builtins novos no runtime:**
  - `format_fixed` exato, `lines`, `today`;
  - `minimum`, `maximum`, `append_text`, `list_dir` e `file_date`;
  - `nan_quantile`, `merge_days`, `contract` e `randn`, este com o `sin` e o `cos` da `libm` portados;
  - `upper` e `lower` como os do Rust;
  - `sum`, `mean` e `prod` de listas;
  - `read_lines`, também de arquivos ZIP, com o inflate escrito em Kairós. Um COTAHIST da B3 sai igual em 4 s.
- **Os mesmos bits em todo sistema:** o DSR do registro em Rust passou a usar o `exp` e o `ln` da `libm`.
- **Limites conhecidos:**
  - a memória: o alocador separa os blocos livres por tamanho e não os junta, e os dados do laboratório ocupam muito mais que no interpretador (o `momentum` chega a 1,5 GB, perto dos 4 GB do wasm32);
  - a internet (`http_get`, `download`) não compila: o WASI não tem rede;
  - dentro de uma função derivada ainda não compilam lambdas, chamadas de funções-valor com floats, criar enums que guardam floats e `x op= v` numa parte de um valor.

## 0.13.0 — M13: o back-end WebAssembly escrito em Kairós

**O compilador da Kairós, escrito em Kairós, agora gera WebAssembly e compila a si mesmo.** [`compiler/wasmgen.krs`](kairos/compiler/wasmgen.krs) transforma o HIR de um programa num módulo WASI (`kairos run kairos/compiler/kw.krs kairos/std saida programa.krs` grava `saida/programa.wasm`), que roda em wasmtime, num navegador ou em qualquer runtime WASI, sem Rust.

- **A mesma semântica do interpretador:**
  - os 32 programas de `tests/wasm` dão a mesma saída, os mesmos erros de execução com as mesmas mensagens (inclusive as que mostram valores, como `index -1 out of bounds for a list of length 3`) e o mesmo código de saída;
  - o teste roda cada módulo no wasmi, com um WASI mínimo (argumentos, ambiente e arquivos).
- **O que compila:**
  - inteiros com aritmética verificada, floats, booleanos e textos;
  - tuplas, structs, listas, mapas (na ordem de inserção), enums com dados e recursivos, padrões;
  - parâmetros `inout`, pelos retornos múltiplos do WebAssembly;
  - `Bytes` e arquivos via WASI;
  - os builtins que o compilador usa.
  - Ainda não compila: closures, tensores, `grad`, protocolos e holdouts, a ponte `b3` e a internet.
- **Memória:**
  - os valores no heap contam suas referências: são copiados antes de mudar quando outros os compartilham (a semântica de valor da linguagem) e liberados na última referência;
  - blocos livres por tamanho: um milhão de structs e textos criados e descartados ocupam uma página de memória;
  - cada tipo ganha suas funções de liberar, copiar, escrever e comparar, geradas conforme o programa as usa.
- **O runtime dos módulos é escrito em Kairós** ([`compiler/rt.krs`](kairos/compiler/rt.krs)):
  - floats escritos como o `{:?}` do Rust, pelo algoritmo Ryu, com as tabelas calculadas em Kairós com inteiros grandes; conferido em 2 milhões de floats;
  - floats lidos como o `parse::<f64>` do Rust, com arredondamento correto e subnormais; conferido em 11 mil textos e 300 pontos médios exatos entre floats vizinhos;
  - a `libm` portada operação por operação (`exp`, `ln`, `^`, `tanh`, `sigmoid`, `softplus`): os mesmos bits do interpretador em 13 milhões de valores;
  - o `%` exato, o alocador, os mapas com hash, UTF-8, Latin-1 e as classes Unicode (`is_alphabetic`…, com tabelas geradas a partir do próprio Rust).
- **Auto-hospedagem:**
  - o compilador inteiro (léxico, parser, verificador, análise temporal, o gerador de imagens e o próprio back-end) vira WebAssembly;
  - rodando como WebAssembly, ele gera os mesmos módulos que o interpretador, byte a byte, e compila a si mesmo no mesmo `kw.wasm` (987 KB): um ponto fixo;
  - o front-end compilado imprime os mesmos tokens, árvores, tipos e HIR que o interpretador, e o compilador de imagens compilado grava as mesmas imagens;
  - `cargo test --release -p kairos --test wasm -- --ignored the_compiler` confere.
- **Os mesmos bits em todo sistema:**
  - o `round` do MinGW dava 1,0 para 0,49999999999999994: `round`, `floor` e `ceil` passam a usar a `libm`;
  - o `relu` deixa de depender do sinal de max(−0,0; 0,0), que muda de plataforma para plataforma.
- **Novos builtins:** `to_bits(x: f64) -> i64` e `from_bits(b: i64) -> f64`.

## 0.12.0 — M12: o compilador da Kairós escrito em Kairós

**O compilador da Kairós agora é escrito em Kairós e compila a si mesmo.** [`kairos/compiler/`](kairos/compiler) tem o front-end inteiro: léxico, parser, verificador de tipos, HIR e análise temporal. O compilador grava a imagem de um programa, e `kairos exec` a executa sem o front-end Rust.

- **O front-end em Kairós** (6.796 linhas):
  - `lexer.krs` e `parser.krs`: os mesmos tokens e a mesma árvore do Rust, com os mesmos erros e dicas;
  - `check.krs` e `body.krs`: inferência por unificação, eixos e tensores, `grad`, holdouts, a ponte `b3` e o HIR;
  - `temporal.krs`: a análise temporal (E0301, E0302, E0303), com os mesmos resumos de função e tempos de conhecimento;
  - `front.krs`, `image.krs` e `kc.krs`: o driver, a imagem do programa e o compilador de linha de comando.
- **Igual ao Rust, conferido nó a nó:**
  - `kairos tokens`, `kairos ast`, `kairos types` e `kairos hir` imprimem o que o front-end Rust vê;
  - o teste de auto-hospedagem exige o mesmo texto dos dois front-ends nos 529 arquivos `.krs` do repositório: 155 com os mesmos tipos, HIR e tempos de conhecimento (2.917 funções), 374 com o mesmo primeiro erro;
  - casos-limite: 23 do léxico, 48 do parser, 306 do verificador e 41 da análise temporal.
- **A inicialização (bootstrap):**
  - o compilador compila a si mesmo, e a imagem que sai (1,4 MB), executada sem o front-end Rust, compila o compilador de novo na mesma imagem, byte a byte;
  - os programas dourados compilados por ele rodam com a mesma saída, os mesmos erros e as mesmas mensagens.
- **`kairos exec programa.kim`** executa a imagem de um programa compilado com `kairos run kairos/compiler/kc.krs kairos/std <saída> programa.krs`.

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
