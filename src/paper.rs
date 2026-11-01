//! Paper trading: operação simulada com dados reais da B3, atualizados a cada pregão.
//!
//! As regras ficam congeladas em results/paper/config.json e protocolo_paper.md (com hash).
//! A cada execução: baixa os dados novos, recalcula todas as carteiras desde o início com o
//! mesmo simulador do backtest (custo real, 1 pregão de atraso, IR de pessoa física) e só
//! ACRESCENTA ao registro os pregões novos. Se o recálculo de um pregão já registrado mudar
//! (dado revisado pela B3 ou código alterado), avisa em vez de reescrever.

use crate::backtest::{self, Plan, SimCfg, SimOut};
use crate::common::Data;
use crate::search::{self, Ctx, Strat};
use crate::util::{self, Paths, pct};
use crate::{b3events, cotahist, macrodata, panel};
use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
pub struct Config {
    /// AAAAMM do primeiro mês operado
    pub inicio_mes: u32,
    pub capital: f64,
    pub estrategias: Vec<String>,
    pub criado_em: String,
    /// último fechamento disponível quando as regras foram congeladas
    pub base_fechamento: u32,
    pub protocolo_sha256: String,
}

fn dir(p: &Paths) -> PathBuf {
    p.results.join("paper")
}

fn bova(d: &Data) -> Option<usize> {
    d.panel.tks.iter().position(|t| t.ticker == "BOVA11")
}

/// TENDÊNCIA no paper: BOVA11 se ele bateu o CDI nos 12 meses anteriores; senão, tudo no CDI.
fn trend_in(d: &Data, b: usize, mi: usize) -> bool {
    if mi < 12 {
        return false;
    }
    let mk: f64 = (mi - 12..mi).map(|m| 1.0 + d.mret[b][m]).product();
    let cd: f64 = (mi - 12..mi).map(|m| 1.0 + d.cdi_m[m]).product();
    mk > cd
}

fn plans_for(d: &Data, ctx: &mut Ctx, s: Strat, start: usize) -> Vec<Plan> {
    if s == Strat::Trend {
        let Some(b) = bova(d) else { return Vec::new() };
        return (start..d.cal.months.len())
            .map(|mi| Plan { mi, w: if trend_in(d, b, mi) { vec![(b, 1.0)] } else { Vec::new() } })
            .collect();
    }
    ctx.plans(s, start)
}

fn cfg_sim(c: &Config) -> SimCfg {
    SimCfg { capital0: c.capital, lag: 1, costs: search::REAL, tax: true, cap_adv: None }
}

pub fn iniciar(p: &Paths, capital: f64, inicio: Option<u32>) -> Result<()> {
    let dir = dir(p);
    std::fs::create_dir_all(&dir)?;
    let cfgp = dir.join("config.json");
    if cfgp.exists() {
        bail!("paper trading já iniciado ({}). Pra recomeçar, apague a pasta {}.", cfgp.display(), dir.display());
    }
    let d = Data::load(p)?;
    let last = *d.panel.dates.last().unwrap();
    let next = util::ym_add(util::ym(last), 1);
    let inicio = inicio.unwrap_or(next);
    let busca = std::fs::read_to_string(p.results.join("busca_resultado.json")).unwrap_or_default();
    let proto = format!(
        "# Protocolo do paper trading\n\n\
Congelado em {hoje}. Base: último fechamento disponível, {base}. Primeiro mês operado: {ini}.\n\n\
Contexto: a busca registrada (results/protocolo_busca.md) **não validou nenhum alpha**. Resultado da busca:\n\n\
```\n{busca}\n```\n\n\
Por isso o paper trading é o teste fora da amostra que sobrou. Ele acompanha **os 7 candidatos da busca ao mesmo tempo**, sem escolher nenhum.\n\n\
## Regras congeladas\n\n\
- Capital: R$ {cap:.0} por estratégia, nominal.\n\
- A carteira do mês é decidida com dados até o último pregão do mês anterior.\n\
- As ordens são executadas no fechamento do 1º pregão do mês.\n\
- Custo real: emolumentos de 0,03% + meio spread real + impacto 0,5·σ·√(ordem/volume).\n\
- IR de pessoa física: 15% sobre o ganho do mês; isenção se as vendas do mês somam até R$ 20 mil; prejuízo compensa ganho futuro; JCP com 15% retido; dividendo isento.\n\
- O caixa rende CDI.\n\
- MOM, RESMOM, MAX52, LOWVOL, DY e MOM+LOWVOL seguem exatamente o protocolo da busca.\n\
- TENDENCIA usa o **BOVA11** como mercado (no sinal e no ativo), porque o NEFIN publica com atraso: BOVA11 se ele bateu o CDI nos 12 meses anteriores, senão CDI.\n\
- Referências: CDI e BOVA11 desde o fechamento do pregão da primeira execução.\n\n\
## Avaliação\n\n\
- Nenhuma estratégia é escolhida antes de **24 meses** de paper.\n\
- Na escolha, as 7 contam como tentativas no registro de testes (DSR).\n\
- O que o paper confirma antes disso: que dados, execução, custos e IR funcionam com dado real, e que cada estratégia se comporta dentro do esperado pelo backtest.\n\
- O registro só recebe acréscimos. Se um pregão já registrado mudar no recálculo, isso gera aviso.\n",
        hoje = util::fmt_date(util::today()),
        base = util::fmt_date(last),
        ini = util::fmt_ym(inicio),
        busca = busca.trim(),
        cap = capital
    );
    std::fs::write(dir.join("protocolo_paper.md"), &proto)?;
    let cfg = Config {
        inicio_mes: inicio,
        capital,
        estrategias: Strat::ALL.iter().map(|s| s.id().to_string()).collect(),
        criado_em: util::fmt_date(util::today()),
        base_fechamento: last,
        protocolo_sha256: search::sha256_hex(proto.as_bytes()),
    };
    std::fs::write(&cfgp, serde_json::to_string_pretty(&cfg)?)?;
    println!(
        "Paper trading congelado: começa em {}, R$ {:.0} por estratégia, protocolo SHA-256 {}",
        util::fmt_ym(inicio),
        capital,
        &cfg.protocolo_sha256[..16]
    );
    // carteiras que serão montadas no fechamento do 1º pregão do mês de início
    if inicio == next {
        let mut ctx = Ctx::new(&d);
        let nm = d.cal.months.len();
        let mut txt = format!(
            "Ordens do mês {} (decididas com dados até {}; executadas no fechamento do 1º pregão de {}):\n",
            util::fmt_ym(inicio),
            util::fmt_date(last),
            util::fmt_ym(inicio)
        );
        for s in Strat::ALL {
            if s == Strat::Trend {
                let alvo = match bova(&d) {
                    Some(b) if trend_in(&d, b, nm) => "100% BOVA11",
                    Some(_) => "100% CDI (BOVA11 não bateu o CDI em 12 meses)",
                    None => "BOVA11 não encontrado nos dados",
                };
                let _ = writeln!(txt, "  {:<11} {alvo}", s.id());
                continue;
            }
            match ctx.form(s, nm) {
                Some(f) => {
                    let mut t: Vec<&str> = f.winners.iter().map(|&t| d.panel.tks[t].ticker.as_str()).collect();
                    t.sort_unstable();
                    let _ = writeln!(txt, "  {:<11} {} ações, {:.2}% cada: {}", s.id(), t.len(), 100.0 / t.len() as f64, t.join(" "));
                }
                None => {
                    let _ = writeln!(txt, "  {:<11} sem carteira (poucos candidatos)", s.id());
                }
            }
        }
        print!("{txt}");
        std::fs::write(dir.join(format!("ordens_{}.txt", inicio)), &txt)?;
    }
    Ok(())
}

fn atualizar_dados(p: &Paths) -> Result<()> {
    let today = util::today();
    let y = util::year(today);
    let from = if (today / 100) % 100 == 1 { y - 1 } else { y };
    cotahist::fetch(&p.raw_cot, from, None, 2)?;
    let store = cotahist::ingest(&p.raw_cot)?;
    util::save_bin(&store, &p.bars)?;
    b3events::refresh_active(&store, &p.raw_b3, 4)?;
    macrodata::fetch_cdi_recent(&p.raw_bcb)?;
    let marker = p.raw_bcb.join(format!(".mensal_{}", util::ym(today)));
    if !marker.exists() {
        macrodata::fetch_ipca(&p.raw_bcb)?;
        macrodata::fetch_nefin(&p.raw_nefin)?;
        std::fs::write(marker, "")?;
    }
    let (evs, _) = b3events::load(&store, &p.raw_b3)?;
    let (pn, _) = panel::build(&store, &evs, &p.diag)?;
    util::save_bin(&pn, &p.panel)?;
    Ok(())
}

/// Acrescenta ao registro os pregões e operações novos; devolve quantos pregões antigos divergiram.
fn registrar(dir: &std::path::Path, id: &str, d: &Data, o: &SimOut) -> Result<usize> {
    let slug = id.replace('+', "_");
    let dpath = dir.join(format!("{slug}_diario.csv"));
    let mut old: BTreeMap<u32, f64> = BTreeMap::new();
    let mut text = std::fs::read_to_string(&dpath).unwrap_or_default();
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if let (Some(dd), Some(v)) = (f.first().and_then(|x| x.replace('-', "").parse::<u32>().ok()), f.get(1).and_then(|x| x.parse::<f64>().ok())) {
            old.insert(dd, v);
        }
    }
    if text.is_empty() {
        text.push_str("data,patrimonio,caixa\n");
    }
    let mut diverg = 0;
    for &(di, nav, cash) in &o.daily {
        let date = d.panel.dates[di];
        match old.get(&date) {
            Some(&v) => {
                if ((nav - v) / v).abs() > 1e-6 {
                    diverg += 1;
                }
            }
            None => {
                let _ = writeln!(text, "{},{nav:.2},{cash:.2}", util::fmt_date(date));
            }
        }
    }
    std::fs::write(&dpath, text)?;

    let tpath = dir.join(format!("{slug}_operacoes.csv"));
    let mut ttext = std::fs::read_to_string(&tpath).unwrap_or_default();
    let logged: HashSet<String> = ttext.lines().skip(1).map(|l| l.split(',').take(3).collect::<Vec<_>>().join(",")).collect();
    if ttext.is_empty() {
        ttext.push_str("data,papel,lado,valor,preco,qtd_aprox,custo\n");
    }
    for t in &o.trades {
        let key = format!("{},{},{}", util::fmt_date(t.date), t.ticker, if t.buy { "C" } else { "V" });
        if logged.contains(&key) {
            continue;
        }
        let qtd = if t.price > 0.0 { (t.value / t.price).round() } else { 0.0 };
        let _ = writeln!(ttext, "{key},{:.2},{:.2},{qtd:.0},{:.2}", t.value, t.price, t.cost);
    }
    std::fs::write(&tpath, ttext)?;

    let mut ir = String::from("mes,vendas,resultado_realizado,ir_pago,isento\n");
    for x in &o.taxes {
        let _ = writeln!(ir, "{},{:.2},{:.2},{:.2},{}", util::fmt_ym(x.ym), x.sales, x.realized, x.tax, if x.exempt { "sim" } else { "nao" });
    }
    std::fs::write(dir.join(format!("{slug}_ir.csv")), ir)?;
    Ok(diverg)
}

pub fn rodar(p: &Paths, offline: bool) -> Result<()> {
    let dir = dir(p);
    let cfg: Config = serde_json::from_str(
        &std::fs::read_to_string(dir.join("config.json")).context("rode `quant-b3 paper iniciar` antes")?,
    )?;
    if !offline {
        atualizar_dados(p)?;
    }
    let d = Data::load(p)?;
    let last = *d.panel.dates.last().unwrap();
    let Some(start) = d.cal.months.iter().position(|&m| m == cfg.inicio_mes) else {
        println!(
            "Último pregão nos dados: {}. Aguardando o 1º pregão de {} pra executar as ordens (ver results\\paper\\ordens_{}.txt).",
            util::fmt_date(last),
            util::fmt_ym(cfg.inicio_mes),
            cfg.inicio_mes
        );
        return Ok(());
    };
    let exec_di = d.cal.mrange[start].0; // fechamento do 1º pregão = execução das primeiras ordens
    let mut ctx = Ctx::new(&d);
    let sim = cfg_sim(&cfg);
    let w_cdi: f64 = ((exec_di + 1)..d.panel.dates.len()).map(|i| 1.0 + d.cdi_d[i]).product();
    let w_bova = bova(&d).and_then(|b| {
        let days = &d.panel.tks[b].days;
        let p0 = days.iter().rev().find(|x| (x.di as usize) <= exec_di)?.px as f64;
        let p1 = days.last()?.px as f64;
        Some(p1 / p0)
    });

    let mut rep = String::new();
    let _ = writeln!(
        rep,
        "\n=== PAPER TRADING desde {} | R$ {:.0} por estratégia | último pregão {} | {} ===",
        util::fmt_ym(cfg.inicio_mes),
        cfg.capital,
        util::fmt_date(last),
        util::fmt_date(util::today())
    );
    let _ = writeln!(
        rep,
        "{:<11} {:>12} {:>9} {:>8} {:>9} {:>8} {:>9} {:>12}",
        "estratégia", "patrimônio", "retorno", "vs CDI", "vs BOVA11", "posições", "IR pago", "divergências"
    );
    for id in &cfg.estrategias {
        let Some(s) = Strat::from_id(id) else { continue };
        let plans = plans_for(&d, &mut ctx, s, start);
        let o = backtest::simulate(&d, &plans, &sim);
        let diverg = registrar(&dir, id, &d, &o)?;
        let r = o.final_value / cfg.capital - 1.0;
        let _ = writeln!(
            rep,
            "{:<11} {:>12.2} {:>9} {:>8} {:>9} {:>8} {:>9.2} {:>12}",
            id,
            o.final_value,
            pct(r),
            pct((1.0 + r) / w_cdi - 1.0),
            w_bova.map(|w| pct((1.0 + r) / w - 1.0)).unwrap_or_default(),
            o.holdings.len(),
            o.tax_paid,
            if diverg > 0 { format!("{diverg} AVISO") } else { "0".into() }
        );
    }
    let _ = writeln!(rep, "{:<11} {:>12.2} {:>9}", "CDI", cfg.capital * w_cdi, pct(w_cdi - 1.0));
    if let Some(w) = w_bova {
        let _ = writeln!(rep, "{:<11} {:>12.2} {:>9}", "BOVA11", cfg.capital * w, pct(w - 1.0));
    }
    let _ = writeln!(rep, "Detalhes em {}: *_diario.csv, *_operacoes.csv, *_ir.csv", dir.display());
    print!("{rep}");
    let mut log = std::fs::read_to_string(dir.join("status.txt")).unwrap_or_default();
    log.push_str(&rep);
    std::fs::write(dir.join("status.txt"), log)?;
    Ok(())
}
