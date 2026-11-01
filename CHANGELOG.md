# Changelog

Versões da linguagem Kairós (pasta `kairos/`). Cada versão vira uma release no GitHub, gerada automaticamente por `.github/workflows/release.yml` a partir da seção correspondente deste arquivo.

## 0.2.1 — Kairós em qualquer sistema

- **Binários para todos os sistemas**, gerados e publicados automaticamente a cada versão:

  | sistema | arquiteturas |
  |---|---|
  | Linux (estático, roda em qualquer distribuição) | x86_64 e ARM64 |
  | macOS | Intel e Apple Silicon |
  | Windows | x86_64 e ARM64 |
  | FreeBSD | x86_64 |
  | NetBSD | x86_64 |

  Nos outros sistemas, inclusive OpenBSD, é só compilar com `cargo build --release -p kairos`.
- **Rust puro:** a Kairós não depende de nenhuma biblioteca do sistema. O acesso à rede do laboratório virou o recurso opcional `net`, que a linguagem não usa, e com isso o OpenSSL deixou de ser necessário.
- **Os mesmos bits em qualquer sistema:** `exp`, `ln` e `^` usam uma implementação matemática em Rust puro (`libm`), e não a do sistema operacional.
- **`.krs` reconhecido:**
  - extensão do VS Code (`kairos-vscode-0.2.1.vsix`): reconhece arquivos `.krs`, dá cores de sintaxe, comentários (`Ctrl+/`), fechamento de colchetes e cores nos blocos ` ```krs ` do Markdown;
  - gramática TextMate em `editors/vscode/syntaxes/kairos.tmLanguage.json`, que serve também no Sublime Text e nos editores JetBrains, e é a base para o reconhecimento pelo GitHub (ver `editors/README.md`);
  - scripts com `#!/usr/bin/env kairos` na primeira linha rodam direto no Linux, macOS e BSD (`chmod +x programa.krs` e depois `./programa.krs`).
- **CI:** testes a cada push em Linux (x86_64 e ARM64), macOS, Windows e FreeBSD.
- **Sistemas de 32 bits:** a pilha do interpretador recua para um tamanho menor quando falta espaço de endereçamento.

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
