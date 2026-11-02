# Política de segurança

## Versões com suporte

Correções de segurança saem numa nova versão de correção da linha mais recente.

| versão | suporte |
| ------- | ------- |
| 0.5.x (a mais recente) | :white_check_mark: |
| < 0.5 | :x: |

## Como reportar uma vulnerabilidade

**Não abra uma issue pública.** Use o reporte privado do GitHub:

1. abra a aba [**Security**](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/security) deste repositório;
2. clique em **Report a vulnerability**;
3. descreva o problema e informe:
   - a versão (`kairos --version`);
   - o sistema operacional;
   - se possível, um programa `.krs` mínimo que reproduza a falha.

O que esperar:
- este é um projeto de pesquisa mantido por uma pessoa, então as respostas são dadas no melhor esforço;
- confirmada a falha, a correção sai numa nova versão, e o aviso de segurança (GitHub Security Advisory) é publicado junto, com crédito a quem reportou, se a pessoa quiser;
- se o reporte não for considerado uma vulnerabilidade, você recebe a explicação.

## O que conta como vulnerabilidade

A Kairós executa programas `.krs`. Hoje a linguagem **não** tem acesso a arquivos, rede nem processos. O único acesso externo é o módulo `b3`, que só lê os dados locais da B3 (`data/processed/panel.bin`).

**Contam como vulnerabilidade:**
- um programa `.krs` que consiga ler ou escrever arquivos, acessar a rede ou executar comandos;
- comportamento de memória indefinido no interpretador. O código da Kairós é Rust seguro, sem `unsafe`, então isso seria uma falha séria;
- um arquivo de dados manipulado que faça o módulo `b3` ou o laboratório executar código.

**Não contam** (são limitações conhecidas; abra uma issue comum se te incomodarem):
- um programa que consome CPU ou memória demais, como laços infinitos ou tensores enormes. A Kairós ainda não limita recursos, então **não execute `.krs` de quem você não confia**;
- erros nos dados públicos que o laboratório baixa (B3, BCB, NEFIN-USP).

## Binários

Os binários das [releases](https://github.com/juliocesarmisneroviczklamt-bit/KAIROS/releases) são compilados pela CI do GitHub a partir da tag correspondente. Confira o download com o `SHA256SUMS.txt` publicado junto:

```text
sha256sum -c SHA256SUMS.txt --ignore-missing
```
