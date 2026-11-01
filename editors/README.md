# Suporte a editores e reconhecimento de `.krs`

| onde | como | estado |
|---|---|---|
| VS Code | extensão em [`vscode/`](vscode/), `.vsix` em cada release | ✅ |
| Sublime Text, editores JetBrains | importar a gramática TextMate [`vscode/syntaxes/kairos.tmLanguage.json`](vscode/syntaxes/kairos.tmLanguage.json) | ✅ (manual) |
| Linux, macOS, BSD | `#!/usr/bin/env kairos` na 1ª linha + `chmod +x` torna o `.krs` executável | ✅ |
| GitHub (barra de linguagens, cores) | entrar no [Linguist](https://github.com/github-linguist/linguist) | depende de adoção |

## Por que o GitHub ainda mostra "Rust"

A barra de linguagens do GitHub e as cores dos arquivos vêm do **Linguist**, que só conhece as linguagens da lista dele. Em novembro de 2026, nem a extensão `.krs` nem a linguagem Kairós estavam registradas lá.

Para entrar, a [regra do Linguist](https://github.com/github-linguist/linguist/blob/main/CONTRIBUTING.md) exige:

1. **Uso real:** pelo menos **2.000 arquivos `.krs`** indexados pela busca do GitHub no último ano, sem contar forks, espalhados por usuários e repositórios diferentes.
2. **Gramática com licença aberta** (MIT, Apache, BSD…). ✅ A gramática está neste repositório, sob licença [MIT](../LICENSE).
3. **Amostras de código:** os programas de [`kairos/examples/`](../kairos/examples/) e [`kairos/tests/programs/`](../kairos/tests/programs/) servem.

Quando essas condições estiverem cumpridas, o pull request para o Linguist leva a entrada abaixo (o `language_id` é gerado pelo `script/update-ids` deles):

```yaml
Kairós:
  type: programming
  color: "#C9A227"
  extensions:
  - ".krs"
  interpreters:
  - kairos
  tm_scope: source.kairos
  ace_mode: text
```

## Testar a gramática

Ela é testada com o mesmo motor do VS Code (`vscode-textmate` + Oniguruma). Os nomes de escopo seguem as convenções do TextMate:

| trecho | escopo |
|---|---|
| eixos (`Tensor[month, asset; f64]`, `over: asset`) | `entity.name.type.axis.kairos` |
| `b3.evaluate` | `support.function.b3.kairos` |
| argumentos nomeados e seletores de eixo | `variable.parameter.label.kairos` |
