//! Estudo C: small caps "onde fundo grande não entra".
//!
//! Hipóteses e regra de decisão registradas ANTES de rodar (não mexer depois de ver o holdout):
//!  H1: momentum long-only (tercil de vencedores) no segmento abaixo do filtro de liquidez do
//!      NEFIN rende mais, já com o custo real de spread, que o mesmo momentum no segmento líquido.
//!  H2: o segmento ilíquido inteiro (pesos iguais) rende mais que o líquido (prêmio de iliquidez).
//!  Passa se: diferença com t > 2 em 2001-2018 E mesmo sinal no holdout 2019-hoje.
//!
//! Segmento: classe mais negociada da empresa, fora do filtro NEFIN, negociada em >80% dos
//! pregões do ano anterior, volume mediano >= R$ 50 mil, preço >= R$ 1 na formação.
//! Execução: custo real, 1 pregão de atraso, posição máxima = 1 dia de volume mediano,
//! R$ 300 mil de hoje (sem IR, pra comparar só o efeito do segmento).

use crate::backtest::{self, Costs, Plan, SimCfg, SimOut};
use crate::common::{self, Data, Form, Report};
use crate::macrodata::Factors;
use crate::registry::{self, Trial};
use crate::stats;
use crate::universe;
use crate::util::{self, Paths, pct};
use anyhow::Result;
use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;

const MIN_MED_VOL: f64 = 50_000.0;
const MIN_PX: f32 = 1.0;
const SPLIT: u32 = 201901;
const CAP_ADV: f64 = 1.0;
const REAL: Costs = Costs::Real { fee: 0.0003, impact_k: 0.5 };

fn winners(f: &Form) -> &Vec<usize> {
    &f.winners
}
fn alive(f: &Form) -> &Vec<usize> {
    &f.alive
}

fn ew_plans(forms: &[Form], pick: fn(&Form) -> &Vec<usize>) -> Vec<Plan> {
    forms
        .iter()
        .map(|f| {
            let v = pick(f);
            Plan { mi: f.mi, w: v.iter().map(|&t| (t, 1.0 / v.len() as f64)).collect() }
        })
        .collect()
}

struct PeriodStats {
    cagr: f64,
    sharpe: f64,
    mdd: f64,
    alpha: Option<(f64, f64)>,
}

fn period_stats(d: &Data, o: &SimOut, early: bool, split: usize) -> PeriodStats {
    let idx: Vec<usize> = (0..o.mi.len()).filter(|&i| (o.mi[i] < split) == early).collect();
    let ret: Vec<f64> = idx.iter().map(|&i| o.ret[i]).collect();
    let cdi: Vec<f64> = idx.iter().map(|&i| d.cdi_m[o.mi[i]]).collect();
    let pf = common::perf(&ret, &cdi);
    let (y, f): (Vec<f64>, Vec<Factors>) =
        idx.iter().filter_map(|&i| d.nef[o.mi[i]].map(|f| (o.ret[i] - f.rf, f))).unzip();
    PeriodStats { cagr: pf.cagr, sharpe: pf.sharpe, mdd: pf.mdd, alpha: common::alpha(&y, &f, false) }
}

/// diferença mensal a - b: (média a.a., t de Newey-West)
fn diff_test(a: &SimOut, b: &SimOut, early: bool, split: usize) -> (f64, f64) {
    let x: Vec<f64> = (0..a.ret.len().min(b.ret.len()))
        .filter(|&i| (a.mi[i] < split) == early)
        .map(|i| a.ret[i] - b.ret[i])
        .collect();
    let t = stats::ols_nw(&x, &[], stats::nw_lags(x.len())).map(|o| o.t[0]).unwrap_or(f64::NAN);
    (stats::mean(&x) * 12.0, t)
}

pub fn run(p: &Paths) -> Result<()> {
    let d = Data::load(p)?;
    let small_all = common::momentum_forms(
        &d,
        &mut |y| universe::small_segment(&d.panel, y, 500_000.0, MIN_MED_VOL),
        15,
        MIN_PX,
    );
    let liq_all = crate::risk::liquid_forms(&d, MIN_PX);
    let start = d.cal.idx_from(200101);
    let split = d.cal.idx_from(SPLIT);
    // só meses em que os dois segmentos têm carteira
    let lm: HashSet<usize> = liq_all.iter().map(|f| f.mi).collect();
    let small: Vec<Form> = small_all.into_iter().filter(|f| f.mi >= start && lm.contains(&f.mi)).collect();
    let sm: HashSet<usize> = small.iter().map(|f| f.mi).collect();
    let liq: Vec<Form> = liq_all.into_iter().filter(|f| sm.contains(&f.mi)).collect();
    let ym0 = d.ym(small[0].mi);
    let defl = d.ipca_factor_from(ym0);
    let cfg = |cap_today: f64| SimCfg { capital0: cap_today / defl, lag: 1, costs: REAL, tax: false, cap_adv: Some(CAP_ADV) };

    let base = cfg(300_000.0);
    let names = ["H1 small: vencedores", "H2 small: todas (EW)", "líquido: vencedores", "líquido: todas (EW)"];
    let runs: Vec<SimOut> = [
        ew_plans(&small, winners),
        ew_plans(&small, alive),
        ew_plans(&liq, winners),
        ew_plans(&liq, alive),
    ]
    .iter()
    .map(|pl| backtest::simulate(&d, pl, &base))
    .collect();

    let mut rep = Report::default();
    rep.line("");
    rep.line("================ C. SMALL CAPS: ONDE FUNDO GRANDE NÃO ENTRA ================");
    rep.line("Hipóteses registradas antes de rodar. H1: momentum no segmento ilíquido > momentum no líquido. H2: segmento ilíquido > líquido.");
    rep.line("Passa se a diferença tiver t > 2 em 2001-2018 E o mesmo sinal no holdout 2019-hoje.");
    let mut by_year: BTreeMap<u32, (usize, usize, usize)> = BTreeMap::new();
    for (s, l) in small.iter().zip(&liq) {
        let e = by_year.entry(d.ym(s.mi) / 100).or_insert((0, 0, 0));
        e.0 += s.alive.len();
        e.1 += l.alive.len();
        e.2 += 1;
    }
    rep.line(format!(
        "Ações por mês (small | líquido): {}",
        by_year
            .iter()
            .filter(|(y, _)| *y % 5 == 0 || **y == 2001)
            .map(|(y, (a, b, c))| format!("{y}: {}|{}", a / c, b / c))
            .collect::<Vec<_>>()
            .join("  ")
    ));
    rep.line(format!(
        "Execução: custo real, 1 pregão de atraso, posição máx = {CAP_ADV:.0} dia de volume mediano, R$ 300 mil de hoje, sem IR."
    ));

    for (label, early) in [("PESQUISA 2001-2018", true), ("HOLDOUT 2019-hoje (olhado uma vez)", false)] {
        rep.line("");
        rep.line(format!("--- {label} ---"));
        rep.line(format!(
            "{:<26} {:>8} {:>8} {:>8} {:>18} {:>11} {:>7}",
            "carteira", "CAGR", "Sharpe*", "maxDD", "alpha 4F (t)", "custo/lado", "giro"
        ));
        for (name, o) in names.iter().zip(&runs) {
            let st = period_stats(&d, o, early, split);
            let idx: Vec<usize> = (0..o.mi.len()).filter(|&i| (o.mi[i] < split) == early).collect();
            let giro = stats::mean(&idx.iter().map(|&i| o.turnover[i]).collect::<Vec<_>>());
            rep.line(format!(
                "{name:<26} {:>8} {:>8.2} {:>8} {:>18} {:>7.0} bps {:>6.0}%",
                pct(st.cagr),
                st.sharpe,
                pct(st.mdd),
                st.alpha.map(|(a, t)| format!("{} ({t:.2})", pct(a))).unwrap_or_default(),
                o.avg_cost_bps(),
                100.0 * giro
            ));
        }
        let cdi: Vec<f64> = runs[0].mi.iter().filter(|&&m| (m < split) == early).map(|&m| d.cdi_m[m]).collect();
        rep.line(format!("{:<26} {:>8}", "CDI", pct(stats::cagr(&cdi, 12.0))));
        let (h1, t1) = diff_test(&runs[0], &runs[2], early, split);
        let (h2, t2) = diff_test(&runs[1], &runs[3], early, split);
        rep.line(format!("H1 (small vencedores - líquido vencedores): {} a.a. (t={t1:.2})", pct(h1)));
        rep.line(format!("H2 (small todas - líquido todas):           {} a.a. (t={t2:.2})", pct(h2)));
    }

    // decisão pela regra registrada
    rep.line("");
    for (h, a, b) in [("H1", 0usize, 2usize), ("H2", 1, 3)] {
        let (m_in, t_in) = diff_test(&runs[a], &runs[b], true, split);
        let (m_out, _) = diff_test(&runs[a], &runs[b], false, split);
        let ok = t_in > 2.0 && m_out.signum() == m_in.signum();
        rep.line(format!(
            "{h}: {} (pesquisa {} t={t_in:.2}; holdout {})",
            if ok { "PASSOU" } else { "NÃO PASSOU" },
            pct(m_in),
            pct(m_out)
        ));
    }

    // capacidade: quanto dinheiro cabe
    rep.line("");
    rep.line("Capacidade (período todo, custo real com impacto, posição máx = 1 dia de volume):");
    rep.line(format!(
        "{:<14} {:>22} {:>12} {:>10} {:>22} {:>12}",
        "capital hoje", "small vencedores CAGR", "custo/lado", "cortado", "líquido vencedores CAGR", "custo/lado"
    ));
    let (h1p, c1p) = (ew_plans(&small, winners), ew_plans(&liq, winners));
    for cap in [50_000.0, 300_000.0, 3_000_000.0, 30_000_000.0] {
        let (a, b) = (backtest::simulate(&d, &h1p, &cfg(cap)), backtest::simulate(&d, &c1p, &cfg(cap)));
        rep.line(format!(
            "R$ {:<11} {:>22} {:>8.0} bps {:>9.0}% {:>22} {:>8.0} bps",
            if cap >= 1e6 { format!("{:.0} mi", cap / 1e6) } else { format!("{:.0} mil", cap / 1e3) },
            pct(stats::cagr(&a.ret, 12.0)),
            a.avg_cost_bps(),
            100.0 * a.capped_frac,
            pct(stats::cagr(&b.ret, 12.0)),
            b.avg_cost_bps()
        ));
    }

    // registro de testes (só o período de pesquisa conta como tentativa)
    rep.line("");
    rep.line("Registro de testes:");
    let reg = p.results.join("registro_testes.csv");
    for (i, nome) in [(0usize, "H1 momentum small"), (1, "H2 small EW")] {
        let o = &runs[i];
        let ex: Vec<f64> = (0..o.mi.len()).filter(|&k| o.mi[k] < split).map(|k| o.ret[k] - d.cdi_m[o.mi[k]]).collect();
        let t = Trial::new("small-caps", nome, "vol mediano>=50k preco>=1 custo real lag1 cap 1 dia", "2001-2018", &ex);
        rep.line(format!("  {}", registry::record_and_describe(&reg, &t)?));
    }

    let mut csv = String::from("mes,h1_small_vencedores,h2_small_todas,liquido_vencedores,liquido_todas,cdi\n");
    for i in 0..runs.iter().map(|o| o.ret.len()).min().unwrap_or(0) {
        let m = runs[0].mi[i];
        let _ = writeln!(
            csv,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}",
            util::fmt_ym(d.ym(m)),
            runs[0].ret[i],
            runs[1].ret[i],
            runs[2].ret[i],
            runs[3].ret[i],
            d.cdi_m[m]
        );
    }
    std::fs::write(p.results.join("smallcaps_mensal.csv"), csv)?;
    std::fs::write(p.results.join("smallcaps_relatorio.txt"), &rep.text)?;
    Ok(())
}
