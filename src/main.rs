use anyhow::Result;
use clap::{Parser, Subcommand};
use quant_b3::util::Paths;
use quant_b3::{
    b3events, cotahist, macrodata, momentum, panel, paper, realistic, registry, risk, search, smallcaps, store,
    util,
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "quant-b3",
    version,
    about = "Laboratório de pesquisa de alpha na B3: dados sem viés de sobrevivência, retorno total e testes honestos"
)]
struct Cli {
    /// Pasta de dados (raw/ e processed/)
    #[arg(long, global = true, default_value = "data")]
    data: PathBuf,
    /// Pasta de resultados
    #[arg(long, global = true, default_value = "results")]
    results: PathBuf,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Baixa os arquivos anuais COTAHIST da B3
    FetchCotahist {
        /// Primeiro ano
        #[arg(long, default_value_t = 1995)]
        from: u32,
        /// Último ano (padrão: ano corrente)
        #[arg(long)]
        to: Option<u32>,
        /// Downloads simultâneos
        #[arg(long, default_value_t = 4)]
        threads: usize,
    },
    /// Lê os COTAHIST e grava data/processed/bars.bin (ações e units, inclusive deslistadas)
    Ingest,
    /// Baixa da B3 os eventos societários (proventos, desdobramentos, subscrições) de todos os emissores
    FetchEvents {
        /// Requisições simultâneas
        #[arg(long, default_value_t = 4)]
        threads: usize,
        /// Baixa de novo mesmo o que já está em cache
        #[arg(long)]
        refresh: bool,
    },
    /// Baixa o CDI (BCB/SGS série 12) e os fatores de risco do NEFIN
    FetchMacro,
    /// Calcula retornos totais diários ajustados e grava data/processed/panel.bin
    Build,
    /// Replica o fator WML do NEFIN e testa momentum long-only com custos
    Momentum(momentum::Args),
    /// Estudo A: momentum com controle de risco (escala pela volatilidade realizada)
    MomentumRisco,
    /// Estudo B: execução realista (spread real, impacto, atraso) e IR de pessoa física
    Realista,
    /// Estudo C: small caps abaixo do filtro de liquidez, com custo real e capacidade
    SmallCaps,
    /// Lista todos os testes já feitos, descontados pelo número de tentativas (DSR)
    Registro,
    /// Busca de alpha seguindo results/protocolo_busca.md (registrado antes de rodar)
    Busca,
    /// Paper trading: operação simulada com dados reais, atualizada a cada pregão
    Paper {
        #[command(subcommand)]
        acao: PaperCmd,
    },
    /// Mostra pregões, eventos e retornos ajustados de um papel (pra auditar os dados)
    Inspect {
        /// Código de negociação (atual ou antigo)
        ticker: String,
        /// Data inicial AAAAMMDD
        #[arg(long, default_value_t = 0)]
        from: u32,
        /// Data final AAAAMMDD
        #[arg(long, default_value_t = 99_991_231)]
        to: u32,
    },
    /// Roda o pipeline inteiro (o que já está em cache não é baixado de novo)
    All {
        /// Primeiro ano do COTAHIST
        #[arg(long, default_value_t = 1995)]
        from: u32,
        #[command(flatten)]
        mom: momentum::Args,
    },
}

#[derive(Subcommand)]
enum PaperCmd {
    /// Congela as regras e mostra as ordens do primeiro mês
    Iniciar {
        /// Capital por estratégia (R$)
        #[arg(long, default_value_t = 100_000.0)]
        capital: f64,
        /// Primeiro mês AAAAMM (padrão: o próximo mês; meses passados só pra teste)
        #[arg(long)]
        inicio: Option<u32>,
    },
    /// Baixa os dados novos da B3/BCB, atualiza as carteiras e mostra a situação
    Rodar {
        /// Não baixa nada; usa os dados locais
        #[arg(long)]
        offline: bool,
    },
    /// Mostra a situação sem baixar nada
    Status,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let p = Paths::new(&cli.data, &cli.results);
    p.ensure()?;
    match cli.cmd {
        Cmd::FetchCotahist { from, to, threads } => cotahist::fetch(&p.raw_cot, from, to, threads),
        Cmd::Ingest => ingest(&p),
        Cmd::FetchEvents { threads, refresh } => fetch_events(&p, threads, refresh),
        Cmd::FetchMacro => fetch_macro(&p),
        Cmd::Build => build(&p),
        Cmd::Momentum(a) => momentum::run(&p, &a),
        Cmd::MomentumRisco => risk::run(&p),
        Cmd::Realista => realistic::run(&p),
        Cmd::SmallCaps => smallcaps::run(&p),
        Cmd::Registro => registry::print_all(&p.results.join("registro_testes.csv")),
        Cmd::Busca => search::run(&p),
        Cmd::Paper { acao } => match acao {
            PaperCmd::Iniciar { capital, inicio } => paper::iniciar(&p, capital, inicio),
            PaperCmd::Rodar { offline } => paper::rodar(&p, offline),
            PaperCmd::Status => paper::rodar(&p, true),
        },
        Cmd::Inspect { ticker, from, to } => inspect(&p, &ticker.to_uppercase(), from, to),
        Cmd::All { from, mom } => {
            cotahist::fetch(&p.raw_cot, from, None, 4)?;
            ingest(&p)?;
            fetch_events(&p, 4, false)?;
            fetch_macro(&p)?;
            build(&p)?;
            momentum::run(&p, &mom)?;
            risk::run(&p)?;
            realistic::run(&p)?;
            smallcaps::run(&p)?;
            registry::print_all(&p.results.join("registro_testes.csv"))
        }
    }
}

fn ingest(p: &Paths) -> Result<()> {
    let store = cotahist::ingest(&p.raw_cot)?;
    util::save_bin(&store, &p.bars)?;
    println!("Gravado {}", p.bars.display());
    Ok(())
}

fn fetch_events(p: &Paths, threads: usize, refresh: bool) -> Result<()> {
    let store: store::BarStore = util::load_bin(&p.bars)?;
    b3events::fetch(&store, &p.raw_b3, threads, refresh)
}

fn fetch_macro(p: &Paths) -> Result<()> {
    macrodata::fetch_cdi(&p.raw_bcb, 1995)?;
    macrodata::fetch_ipca(&p.raw_bcb)?;
    macrodata::fetch_nefin(&p.raw_nefin)
}

fn inspect(p: &Paths, ticker: &str, from: u32, to: u32) -> Result<()> {
    let store: store::BarStore = util::load_bin(&p.bars)?;
    let Some((k, s)) = store
        .series
        .iter()
        .enumerate()
        .find(|(_, s)| s.ticker == ticker || s.aliases.iter().any(|a| a == ticker))
    else {
        anyhow::bail!("{ticker} não encontrado na base");
    };
    println!("{}  classe {}  códigos antigos {:?}", s.ticker, s.class, s.aliases);
    println!("ISINs {:?}", s.isins);
    for (n, a, b) in &s.names {
        println!("nome de pregão '{n}' de {} a {}", util::fmt_date(*a), util::fmt_date(*b));
    }
    let has_events = std::fs::read_dir(&p.raw_b3).map(|mut d| d.next().is_some()).unwrap_or(false);
    if has_events {
        let (evs, _) = b3events::load(&store, &p.raw_b3)?;
        for e in evs[k].iter().filter(|e| e.datacom >= from && e.datacom <= to) {
            println!("  evento  data-com {}  {:<16} {:?}", util::fmt_date(e.datacom), e.label, e.ev);
        }
    }
    let rets: std::collections::HashMap<u32, f64> = util::load_bin::<panel::Panel>(&p.panel)
        .ok()
        .and_then(|pn| {
            let tk = pn.tks.iter().find(|t| t.ticker == s.ticker)?;
            Some(tk.days.iter().map(|d| (pn.dates[d.di as usize], d.r)).collect())
        })
        .unwrap_or_default();
    println!(
        "{:<10} {:<14} {:>5} {:>12} {:>14} {:>16} {:>9} {:>9}",
        "data", "especi", "fat", "fechamento", "qtd", "volume R$", "vol/qtd", "ret.aj."
    );
    for b in s.bars.iter().filter(|b| b.date >= from && b.date <= to) {
        println!(
            "{:<10} {:<14} {:>5} {:>12.4} {:>14.0} {:>16.2} {:>9.4} {:>9}",
            util::fmt_date(b.date),
            b.especi,
            b.fatcot,
            b.close,
            b.qty,
            b.vol_brl,
            b.vol_brl / b.qty.max(1.0),
            rets.get(&b.date).map(|r| util::pct(*r)).unwrap_or_default()
        );
    }
    Ok(())
}

fn build(p: &Paths) -> Result<()> {
    let store: store::BarStore = util::load_bin(&p.bars)?;
    let (evs, es) = b3events::load(&store, &p.raw_b3)?;
    es.print();
    let (pn, bs) = panel::build(&store, &evs, &p.diag)?;
    bs.print();
    util::save_bin(&pn, &p.panel)?;
    println!("Gravado {} ({} papéis, {} pregões)", p.panel.display(), pn.tks.len(), pn.dates.len());
    Ok(())
}
