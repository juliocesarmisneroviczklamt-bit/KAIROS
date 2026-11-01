//! Estudo A: momentum com controle de risco. Parâmetros fixados ANTES de rodar:
//! - Long-short: WML escalado por (alvo de vol / vol realizada dos 126 pregões anteriores),
//!   como em Barroso & Santa-Clara (2015). Alvo 12% a.a., alavancagem máxima 2x.
//! - Long-only: exposição = min(1, 15% / vol realizada dos vencedores), resto no CDI
//!   (carteiras geridas por volatilidade, Moreira & Muir 2017).

use crate::backtest::{self, Costs, Plan, SimCfg};
use crate::common::{self, Data, Form, Report};
use crate::macrodata::Factors;
use crate::registry::{self, Trial};
use crate::stats;
use crate::universe::{self, LiqRule};
use crate::util::{self, Paths, pct};
use anyhow::Result;
use std::fmt::Write as _;

pub const WIN: usize = 126;
pub const TARGET_LS: f64 = 0.12;
pub const MAX_LEV: f64 = 2.0;
pub const TARGET_LO: f64 = 0.15;

/// Vol anualizada (soma dos quadrados, sem tirar a média) dos `WIN` pregões antes de `s`.
pub fn realized_vol(series: &[f64], s: usize) -> Option<f64> {
    let lo = s.saturating_sub(WIN);
    let x: Vec<f64> = series[lo..s].iter().copied().filter(|v| v.is_finite()).collect();
    if x.len() < WIN / 2 {
        return None;
    }
    Some((x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 * 252.0).sqrt())
}

/// Carteiras de momentum no universo líquido (NEFIN).
pub fn liquid_forms(d: &Data, min_px: f32) -> Vec<Form> {
    common::momentum_forms(d, &mut |y| universe::eligible(&d.panel, y, LiqRule::Dias, 500_000.0), 15, min_px)
}

/// Séries diárias (por índice de data) de vencedores e perdedores, EW com rebalanceamento diário.
pub fn daily_series(d: &Data, forms: &[Form]) -> (Vec<f64>, Vec<f64>) {
    let nd = d.panel.dates.len();
    let (mut w, mut l) = (vec![f64::NAN; nd], vec![f64::NAN; nd]);
    for f in forms {
        let (s, e) = d.window(f.mi, 0);
        let wd = common::daily_ew(&d.panel, &f.winners, s, e);
        let ld = common::daily_ew(&d.panel, &f.losers, s, e);
        for i in 0..wd.len() {
            w[s + i] = wd[i];
            l[s + i] = ld[i];
        }
    }
    (w, l)
}

/// Long-only nos vencedores (pesos iguais); com `w_daily`, exposição controlada pela vol.
pub fn winner_plans(d: &Data, forms: &[Form], w_daily: Option<&[f64]>) -> Vec<Plan> {
    forms
        .iter()
        .map(|f| {
            let (s, _) = d.window(f.mi, 0);
            let expo = match w_daily {
                Some(series) => realized_vol(series, s).map(|v| (TARGET_LO / v).min(1.0)).unwrap_or(1.0),
                None => 1.0,
            };
            let n = f.winners.len() as f64;
            Plan { mi: f.mi, w: f.winners.iter().map(|&t| (t, expo / n)).collect() }
        })
        .collect()
}

fn excess_and_factors(d: &Data, mi: &[usize], ret: &[f64]) -> (Vec<f64>, Vec<Factors>) {
    mi.iter().zip(ret).filter_map(|(&m, &r)| d.nef[m].map(|f| (r - f.rf, f))).unzip()
}

fn sharpe(x: &[f64]) -> f64 {
    stats::mean(x) / stats::std(x) * 12f64.sqrt()
}

pub fn run(p: &Paths) -> Result<()> {
    let d = Data::load(p)?;
    let all = liquid_forms(&d, 0.0);
    // as séries diárias usam todo o histórico, pra já ter vol estimada em jan/2001
    let (w_d, l_d) = daily_series(&d, &all);
    let wml_d: Vec<f64> = w_d.iter().zip(&l_d).map(|(a, b)| a - b).collect();
    let start = d.cal.idx_from(200101);
    let forms: Vec<Form> = all.into_iter().filter(|f| f.mi >= start).collect();

    // long-short
    let (mut ls, mut ls_s, mut levs, mut months) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for f in &forms {
        let (s, e) = d.window(f.mi, 0);
        let wml = common::compound(&w_d[s..e]) - common::compound(&l_d[s..e]);
        let lev = realized_vol(&wml_d, s).map(|v| (TARGET_LS / v).min(MAX_LEV)).unwrap_or(1.0);
        ls.push(wml);
        ls_s.push(lev * wml);
        levs.push(lev);
        months.push(f.mi);
    }

    // long-only (custo fixo de 25 bps por lado, como na fase 1, pra comparar só o efeito do controle)
    let cfg = SimCfg { capital0: 1_000_000.0, lag: 0, costs: Costs::Fixed { bps: 25.0 }, tax: false, cap_adv: None };
    let lo0 = backtest::simulate(&d, &winner_plans(&d, &forms, None), &cfg);
    let lo1 = backtest::simulate(&d, &winner_plans(&d, &forms, Some(&w_d)), &cfg);
    let cdi: Vec<f64> = lo0.mi.iter().map(|&m| d.cdi_m[m]).collect();

    let mut rep = Report::default();
    rep.line("");
    rep.line("================ A. MOMENTUM COM CONTROLE DE RISCO ================");
    rep.line(format!(
        "Parâmetros fixados antes de rodar: vol dos últimos {WIN} pregões; long-short com alvo de {:.0}% a.a. e alavancagem máx {MAX_LEV:.0}x; long-only com alvo de {:.0}% (o resto fica no CDI).",
        TARGET_LS * 100.0,
        TARGET_LO * 100.0
    ));
    rep.line(format!(
        "Período: {} a {} ({} meses)",
        util::fmt_ym(d.ym(months[0])),
        util::fmt_ym(d.ym(*months.last().unwrap())),
        months.len()
    ));
    rep.line("");
    rep.line(format!(
        "{:<30} {:>8} {:>8} {:>7} {:>7} {:>8} {:>9} {:>7}",
        "long-short (WML)", "média", "vol", "Sharpe", "t(NW)", "maxDD", "pior mês", "assim."
    ));
    for (name, x) in [("original", &ls), ("escalado pela vol", &ls_s)] {
        let t = stats::ols_nw(x, &[], stats::nw_lags(x.len())).map(|o| o.t[0]).unwrap_or(f64::NAN);
        let (_, sk, _) = registry::moments(x);
        let worst = x.iter().copied().fold(f64::INFINITY, f64::min);
        rep.line(format!(
            "{name:<30} {:>8} {:>8} {:>7.2} {t:>7.2} {:>8} {:>9} {sk:>7.2}",
            pct(stats::mean(x) * 12.0),
            pct(stats::std(x) * 12f64.sqrt()),
            sharpe(x),
            pct(stats::max_dd(x)),
            pct(worst)
        ));
    }
    rep.line(format!(
        "alavancagem do escalado: média {:.2}x, mín {:.2}x, máx {:.2}x",
        stats::mean(&levs),
        levs.iter().copied().fold(f64::INFINITY, f64::min),
        levs.iter().copied().fold(0.0, f64::max)
    ));

    rep.line("");
    rep.line(format!(
        "{:<30} {:>8} {:>8} {:>8} {:>8} {:>9} {:>17} {:>17}",
        "long-only vencedores (25bps)", "CAGR", "vol", "Sharpe*", "maxDD", "exposição", "alpha 4F (t)", "alpha 5F (t)"
    ));
    for (name, o) in [("original", &lo0), ("com controle de vol", &lo1)] {
        let pf = common::perf(&o.ret, &cdi);
        let (y, f) = excess_and_factors(&d, &o.mi, &o.ret);
        let fa = |a: Option<(f64, f64)>| a.map(|(a, t)| format!("{} ({t:.2})", pct(a))).unwrap_or_default();
        rep.line(format!(
            "{name:<30} {:>8} {:>8} {:>8.2} {:>8} {:>8.0}% {:>17} {:>17}",
            pct(pf.cagr),
            pct(pf.vol),
            pf.sharpe,
            pct(pf.mdd),
            100.0 * stats::mean(&o.exposure),
            fa(common::alpha(&y, &f, false)),
            fa(common::alpha(&y, &f, true))
        ));
    }
    let cdi_cagr = stats::cagr(&cdi, 12.0);
    rep.line(format!("CDI no período: {}   (* Sharpe sobre o CDI)", pct(cdi_cagr)));

    // consistência entre subperíodos
    rep.line("");
    let split = d.cal.idx_from(201901);
    let half = |x: &[f64], mi: &[usize], early: bool| -> Vec<f64> {
        x.iter().zip(mi).filter(|(_, m)| (**m < split) == early).map(|(v, _)| *v).collect()
    };
    let ex = |o: &backtest::SimOut| -> Vec<f64> { o.ret.iter().zip(&cdi).map(|(a, b)| a - b).collect() };
    let (ex0, ex1) = (ex(&lo0), ex(&lo1));
    rep.line("Sharpe por subperíodo        2001-2018   2019-hoje");
    for (name, x, mi) in [
        ("WML original", &ls, &months),
        ("WML escalado", &ls_s, &months),
        ("long-only original*", &ex0, &lo0.mi),
        ("long-only com controle*", &ex1, &lo1.mi),
    ] {
        rep.line(format!(
            "{name:<28} {:>9.2} {:>11.2}",
            sharpe(&half(x, mi, true)),
            sharpe(&half(x, mi, false))
        ));
    }

    // registro de testes
    rep.line("");
    rep.line("Registro de testes (desconta o número de tentativas):");
    let reg = p.results.join("registro_testes.csv");
    for t in [
        Trial::new("momentum", "WML long-short", "NEFIN terciles 12-2", "2001-hoje", &ls),
        Trial::new("momentum", "long-only vencedores", "custo 25bps", "2001-hoje", &ex0),
        Trial::new("momentum-risco", "WML escalado pela vol", "alvo 12% jan 126 alav 2x", "2001-hoje", &ls_s),
        Trial::new("momentum-risco", "long-only com controle de vol", "alvo 15% jan 126 custo 25bps", "2001-hoje", &ex1),
    ] {
        rep.line(format!("  {}", registry::record_and_describe(&reg, &t)?));
    }

    let mut csv = String::from("mes,wml,wml_escalado,alavancagem,longonly,longonly_controle,exposicao,cdi\n");
    for i in 0..lo0.mi.len().min(months.len()) {
        let _ = writeln!(
            csv,
            "{},{:.6},{:.6},{:.3},{:.6},{:.6},{:.3},{:.6}",
            util::fmt_ym(d.ym(months[i])),
            ls[i],
            ls_s[i],
            levs[i],
            lo0.ret[i],
            lo1.ret[i],
            lo1.exposure[i],
            cdi[i]
        );
    }
    std::fs::write(p.results.join("risco_mensal.csv"), csv)?;
    std::fs::write(p.results.join("risco_relatorio.txt"), &rep.text)?;
    Ok(())
}
