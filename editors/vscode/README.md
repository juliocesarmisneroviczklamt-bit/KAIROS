# Kairós para VS Code

Suporte à linguagem [Kairós](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS):

- **Reconhecimento:** arquivos `.krs` e scripts com `#!/usr/bin/env kairos` são identificados como Kairós.
- **Cores de sintaxe:**
  - palavras-chave, tipos, `Tensor[month, asset; f64]` com os eixos destacados;
  - argumentos nomeados (`over: asset`), funções embutidas e o módulo `b3`;
  - números, strings e comentários.
- **Edição:**
  - `Ctrl+/` comenta a linha (`//`) e `Shift+Alt+A` comenta o bloco (`/* */`);
  - fechamento automático de `{ [ ( "`;
  - indentação.
- **Markdown:** blocos ` ```krs ` e ` ```kairos ` também ganham cores.

## Instalar

Baixe o `.vsix` da página de [Releases](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/releases) e rode:

```text
code --install-extension kairos-vscode-<versão>.vsix
```

A mesma gramática (`syntaxes/kairos.tmLanguage.json`, formato TextMate) funciona também no Sublime Text e nos editores JetBrains (via TextMate Bundles).
