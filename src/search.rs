//! Busca de alpha seguindo results/protocolo_busca.md (registrado antes de rodar).
//! Os sinais usam só dados até o fim do mês anterior à carteira.

use crate::backtest::{self, Costs, Plan, SimCfg, SimOut};
use crate::common::{self, Data, Form, Report};
use crate::macrodata::Factors;
use crate::panel::Day;
use crate::registry::{self, Trial};
use crate::stats;
use crate::universe::{self, LiqRule};
use crate::util::{self, Paths, pct};
use anyhow::Result;
use std::collections::HashMap;
use std::fmt::Write as _;

pub const SPLIT: u32 = 201901;
pub const MIN_PX: f32 = 1.0;
pub const MIN_STOCKS: usize = 15;
pub const REAL: Costs = Costs::Real { fee: 0.0003, impact_k: 0.5 };
const PROTOCOL_SHA: &str = "e83eb42722ba9da5850c3793b483ec1acc9a928846267319877b060e4cd89771";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Strat {
    Mom,
    ResMom,
    Max52,
    LowVol,
    Dy,
    MomLowVol,
    Trend,
}

impl Strat {
    pub const ALL: [Strat; 7] =
        [Strat::Mom, Strat::ResMom, Strat::Max52, Strat::LowVol, Strat::Dy, Strat::MomLowVol, Strat::Trend];
    pub fn id(&self) -> &'static str {
        match self {
            Strat::Mom => "MOM",
            Strat::ResMom => "RESMOM",
            Strat::Max52 => "MAX52",
            Strat::LowVol => "LOWVOL",
            Strat::Dy => "DY",
            Strat::MomLowVol => "MOM+LOWVOL",
            Strat::Trend => "TENDENCIA",
        }
    }
    pub fn desc(&self) -> &'static str {
        match self {
            Strat::Mom => "momentum 12-2",
            Strat::ResMom => "momentum residual (CAPM, 36m)",
            Strat::Max52 => "perto da máxima de 52 semanas",
            Strat::LowVol => "baixa volatilidade (252d)",
            Strat::Dy => "dividend yield (252d)",
            Strat::MomLowVol => "momentum + baixa vol",
            Strat::Trend => "tendência: mercado ou CDI",
        }
    }
    pub fn from_id(s: &str) -> Option<Strat> {
        Strat::ALL.iter().copied().find(|x| x.id().eq_ignore_ascii_case(s))
    }
}

/// Últimos `n` pregões (do calendário) antes do mês `mi`.
fn trailing(d: &Data, t: usize, mi: usize, n: usize) -> &[Day] {
    let a0 = d.month_start(mi);
    common::days_in(&d.panel.tks[t].days, a0.saturating_sub(n), a0)
}

fn vol252(d: &Data, t: usize, mi: usize) -> Option<f64> {
    let w = trailing(d, t, mi, 252);
    if w.len() < 150 {
        return None;
    }
    let r: Vec<f64> = w.iter().map(|x| x.r).collect();
    Some(stats::std(&r))
}

fn max52(d: &Data, t: usize, mi: usize) -> Option<f64> {
    let w = trailing(d, t, mi, 252);
    if w.len() < 150 {
        return None;
    }
    let (mut lvl, mut mx) = (1.0f64, 1.0f64);
    for x in w {
        lvl *= 1.0 + x.r;
        mx = mx.max(lvl);
    }
    Some(lvl / mx)
}

fn dy252(d: &Data, t: usize, mi: usize) -> Option<f64> {
    let w = trailing(d, t, mi, 252);
    if w.len() < 150 {
        return None;
    }
    Some(w.iter().map(|x| (x.div + x.jcp) as f64).sum())
}

fn pct_ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let den = (v.len().max(2) - 1) as f64;
    for (k, &i) in idx.iter().enumerate() {
        r[i] = k as f64 / den;
    }
    r
}

/// Contexto com o mercado EW mensal (pro momentum residual) e cache de universo.
pub struct Ctx<'a> {
    pub d: &'a Data,
    pub mkt_ew: Vec<f64>,
    univ: HashMap<u32, Vec<usize>>,
}

impl<'a> Ctx<'a> {
    pub fn new(d: &'a Data) -> Self {
        let nm = d.cal.months.len();
        let mut c = Ctx { d, mkt_ew: vec![f64::NAN; nm], univ: HashMap::new() };
        for mi in 1..nm {
            let al: Vec<usize> = c.universe(d.cal.months[mi] / 100).into_iter().filter(|&t| d.traded[t][mi - 1]).collect();
            if al.len() >= 10 {
                c.mkt_ew[mi] = al.iter().map(|&t| d.mret[t][mi]).sum::<f64>() / al.len() as f64;
            }
        }
        c
    }

    pub fn universe(&mut self, y: u32) -> Vec<usize> {
        let d = self.d;
        self.univ.entry(y).or_insert_with(|| universe::eligible(&d.panel, y, LiqRule::Dias, 500_000.0)).clone()
    }

    /// candidatos do mês mi: universo do ano, negociou em mi-1, preço >= R$ 1
    pub fn alive(&mut self, mi: usize) -> Vec<usize> {
        let d = self.d;
        self.universe(d.ym_any(mi) / 100)
            .into_iter()
            .filter(|&t| d.traded[t][mi - 1] && d.last_px[t][mi - 1] >= MIN_PX)
            .collect()
    }

    fn resmom(&self, t: usize, mi: usize) -> Option<f64> {
        let d = self.d;
        if mi < 36 {
            return None;
        }
        let (mut xs, mut ys, mut ms) = (Vec::new(), Vec::new(), Vec::new());
        for m in mi - 36..mi {
            if d.traded[t][m] && self.mkt_ew[m].is_finite() {
                xs.push(self.mkt_ew[m] - d.cdi_m[m]);
                ys.push(d.mret[t][m] - d.cdi_m[m]);
                ms.push(m);
            }
        }
        if ys.len() < 24 {
            return None;
        }
        let o = stats::ols_nw(&ys, &[&xs], 0)?;
        let e: Vec<f64> = (0..ys.len()).map(|i| ys[i] - o.coef[0] - o.coef[1] * xs[i]).collect();
        let sd = stats::std(&e);
        if !(sd > 0.0) {
            return None;
        }
        let s: f64 = ms.iter().zip(&e).filter(|(m, _)| **m >= mi - 12 && **m <= mi - 2).map(|(_, v)| v).sum();
        Some(s / sd)
    }

    /// sinal (maior = melhor) do papel t para a carteira do mês mi
    pub fn signal(&self, s: Strat, t: usize, mi: usize) -> Option<f64> {
        let d = self.d;
        match s {
            Strat::Mom => Some(common::mom_signal(&d.mret, t, mi)),
            Strat::ResMom => self.resmom(t, mi),
            Strat::Max52 => max52(d, t, mi),
            Strat::LowVol => vol252(d, t, mi).map(|v| -v),
            Strat::Dy => dy252(d, t, mi),
            Strat::MomLowVol | Strat::Trend => None,
        }
    }

    /// carteira (tercil de cima do sinal) do mês mi
    pub fn form(&mut self, s: Strat, mi: usize) -> Option<Form> {
        if mi < 13 || s == Strat::Trend {
            return None;
        }
        let alive = self.alive(mi);
        let mut sig: Vec<(usize, f64)> = if s == Strat::MomLowVol {
            let both: Vec<(usize, f64, f64)> = alive
                .iter()
                .filter_map(|&t| Some((t, common::mom_signal(&self.d.mret, t, mi), -vol252(self.d, t, mi)?)))
                .collect();
            let (rm, rv) = (
                pct_ranks(&both.iter().map(|x| x.1).collect::<Vec<_>>()),
                pct_ranks(&both.iter().map(|x| x.2).collect::<Vec<_>>()),
            );
            both.iter().enumerate().map(|(i, x)| (x.0, 0.5 * (rm[i] + rv[i]))).collect()
        } else {
            alive.iter().filter_map(|&t| self.signal(s, t, mi).filter(|v| v.is_finite()).map(|v| (t, v))).collect()
        };
        if sig.len() < MIN_STOCKS {
            return None;
        }
        sig.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n = sig.len();
        let k3 = ((n as f64) / 3.0).round() as usize;
        Some(Form {
            mi,
            alive: sig.iter().map(|x| x.0).collect(),
            losers: sig[..k3].iter().map(|x| x.0).collect(),
            winners: sig[n - k3..].iter().map(|x| x.0).collect(),
        })
    }

    /// Planos (pesos iguais no tercil de cima) a partir do mês `start`.
    pub fn plans(&mut self, s: Strat, start: usize) -> Vec<Plan> {
        (start..self.d.cal.months.len())
            .filter_map(|mi| self.form(s, mi))
            .map(|f| {
                let n = f.winners.len() as f64;
                Plan { mi: f.mi, w: f.winners.iter().map(|&t| (t, 1.0 / n)).collect() }
            })
            .collect()
    }
}

/// Tendência: mercado NEFIN (VW) se o mercado bateu o CDI nos 12 meses anteriores, senão CDI.
pub fn trend_series(d: &Data, start: usize) -> (Vec<usize>, Vec<f64>) {
    let (mut mis, mut rs) = (Vec::new(), Vec::new());
    let mut prev: Option<bool> = None;
    for mi in start.max(12)..d.cal.months.len() {
        if (mi - 12..=mi).any(|m| d.nef_vw[m].is_none()) {
            continue;
        }
        let mk: f64 = (mi - 12..mi).map(|m| d.nef_vw[m].map(|f| 1.0 + f.rm_rf + f.rf).unwrap_or(1.0)).product();
        let cd: f64 = (mi - 12..mi).map(|m| 1.0 + d.cdi_m[m]).product();
        let inside = mk > cd;
        let f = d.nef_vw[mi].unwrap();
        let mut r = if inside { f.rm_rf + f.rf } else { d.cdi_m[mi] };
        if prev.is_some_and(|p| p != inside) {
            r -= 0.0010;
        }
        prev = Some(inside);
        mis.push(mi);
        rs.push(r);
    }
    (mis, rs)
}

pub struct Ev {
    pub cagr: f64,
    pub sharpe: f64,
    pub mdd: f64,
    pub beta: f64,
    pub alpha: f64,
    pub t_alpha: f64,
    pub ir: f64,
    pub a4: Option<(f64, f64)>,
}

/// Estatísticas num subperíodo (early = antes do SPLIT). Devolve também a série hedgeada.
pub fn evaluate(d: &Data, mis: &[usize], rs: &[f64], early: Option<bool>, split: usize) -> Option<(Ev, Vec<f64>)> {
    let idx: Vec<usize> = (0..mis.len())
        .filter(|&i| early.is_none_or(|e| (mis[i] < split) == e) && d.nef_vw[mis[i]].is_some() && d.nef[mis[i]].is_some())
        .collect();
    if idx.len() < 24 {
        return None;
    }
    let r: Vec<f64> = idx.iter().map(|&i| rs[i]).collect();
    let cdi: Vec<f64> = idx.iter().map(|&i| d.cdi_m[mis[i]]).collect();
    let fvw: Vec<Factors> = idx.iter().map(|&i| d.nef_vw[mis[i]].unwrap()).collect();
    let few: Vec<Factors> = idx.iter().map(|&i| d.nef[mis[i]].unwrap()).collect();
    let y: Vec<f64> = r.iter().zip(&fvw).map(|(a, f)| a - f.rf).collect();
    let x: Vec<f64> = fvw.iter().map(|f| f.rm_rf).collect();
    let o = stats::ols_nw(&y, &[&x], stats::nw_lags(y.len()))?;
    let (alpha, beta) = (o.coef[0], o.coef[1]);
    let h: Vec<f64> = y.iter().zip(&x).map(|(a, b)| a - beta * b).collect();
    let pf = common::perf(&r, &cdi);
    let y4: Vec<f64> = r.iter().zip(&few).map(|(a, f)| a - f.rf).collect();
    Some((
        Ev {
            cagr: pf.cagr,
            sharpe: pf.sharpe,
            mdd: pf.mdd,
            beta,
            alpha: alpha * 12.0,
            t_alpha: o.t[0],
            ir: stats::mean(&h) / stats::std(&h) * 12f64.sqrt(),
            a4: common::alpha(&y4, &few, false),
        },
        h,
    ))
}

struct Res {
    s: Strat,
    mis: Vec<usize>,
    rs: Vec<f64>,
    cost_bps: f64,
    research: Ev,
    holdout: Option<Ev>,
}

fn base_cfg(d: &Data, start: usize) -> SimCfg {
    let defl = d.ipca_factor_from(d.ym(start));
    SimCfg { capital0: 300_000.0 / defl, lag: 1, costs: REAL, tax: false, cap_adv: None }
}

pub fn run(p: &Paths) -> Result<()> {
    let sha = std::fs::read(p.results.join("protocolo_busca.md")).map(|b| sha256_hex(&b)).unwrap_or_default();
    let d = Data::load(p)?;
    let mut ctx = Ctx::new(&d);
    let start = d.cal.idx_from(200101);
    let split = d.cal.idx_from(SPLIT);
    let cfg = base_cfg(&d, start);

    let mut res: Vec<Res> = Vec::new();
    for s in Strat::ALL {
        let (mis, rs, cost_bps) = if s == Strat::Trend {
            let (m, r) = trend_series(&d, start);
            (m, r, 10.0)
        } else {
            let plans = ctx.plans(s, start);
            let o: SimOut = backtest::simulate(&d, &plans, &cfg);
            let c = o.avg_cost_bps();
            (o.mi, o.ret, c)
        };
        let Some((research, _)) = evaluate(&d, &mis, &rs, Some(true), split) else { continue };
        let holdout = evaluate(&d, &mis, &rs, Some(false), split).map(|x| x.0);
        res.push(Res { s, mis, rs, cost_bps, research, holdout });
    }

    let mut rep = Report::default();
    rep.line("");
    rep.line("================ BUSCA DE ALPHA (protocolo registrado) ================");
    rep.line(format!(
        "Protocolo: results\\protocolo_busca.md  SHA-256 {}  ({})",
        &sha[..16.min(sha.len())],
        if sha == PROTOCOL_SHA { "confere com o registrado" } else { "ATENÇÃO: NÃO CONFERE" }
    ));
    rep.line("Alpha = CAPM contra o mercado NEFIN (VW, proxy de ETF). IR = Sharpe da série hedgeada. Sharpe* = sobre o CDI.");
    for (label, early) in [("PESQUISA 2001-2018", true), ("HOLDOUT 2019-hoje (olhado uma vez)", false)] {
        rep.line("");
        rep.line(format!("--- {label} ---"));
        rep.line(format!(
            "{:<11} {:<31} {:>7} {:>7} {:>7} {:>5} {:>15} {:>6} {:>15} {:>9}",
            "id", "sinal", "CAGR", "Sharpe*", "maxDD", "beta", "alpha CAPM (t)", "IR", "alpha 4F (t)", "custo"
        ));
        for r in &res {
            let e = if early { Some(&r.research) } else { r.holdout.as_ref() };
            let Some(e) = e else { continue };
            rep.line(format!(
                "{:<11} {:<31} {:>7} {:>7.2} {:>7} {:>5.2} {:>15} {:>6.2} {:>15} {:>5.0} bps",
                r.s.id(),
                r.s.desc(),
                pct(e.cagr),
                e.sharpe,
                pct(e.mdd),
                e.beta,
                format!("{} ({:.2})", pct(e.alpha), e.t_alpha),
                e.ir,
                e.a4.map(|(a, t)| format!("{} ({t:.2})", pct(a))).unwrap_or_default(),
                r.cost_bps
            ));
        }
        let mis: Vec<usize> = (start..d.cal.months.len()).filter(|&m| (m < split) == early && d.nef_vw[m].is_some()).collect();
        let mk: Vec<f64> = mis.iter().map(|&m| d.nef_vw[m].map(|f| f.rm_rf + f.rf).unwrap_or(0.0)).collect();
        let cdi: Vec<f64> = mis.iter().map(|&m| d.cdi_m[m]).collect();
        let pm = common::perf(&mk, &cdi);
        rep.line(format!(
            "{:<11} {:<31} {:>7} {:>7.2} {:>7}   (CDI {})",
            "MERCADO",
            "NEFIN value-weighted",
            pct(pm.cagr),
            pm.sharpe,
            pct(pm.mdd),
            pct(stats::cagr(&cdi, 12.0))
        ));
    }

    // registro: todas as tentativas, série hedgeada do período de pesquisa
    rep.line("");
    let reg = p.results.join("registro_testes.csv");
    let mut all = Vec::new();
    for r in &res {
        if let Some((_, h)) = evaluate(&d, &r.mis, &r.rs, Some(true), split) {
            let t = Trial::new("busca", r.s.id(), "protocolo e83eb427 alpha CAPM hedgeado", "2001-2018", &h);
            all = registry::upsert(&reg, &t)?;
        }
    }

    // seleção e validação pela regra registrada
    let chosen = res.iter().filter(|r| r.research.t_alpha > 2.0).max_by(|a, b| a.research.ir.total_cmp(&b.research.ir));
    let (sel, validated, motivo) = match chosen {
        None => {
            let best = res.iter().max_by(|a, b| a.research.ir.total_cmp(&b.research.ir)).unwrap();
            (best, false, "nenhum candidato com alpha CAPM t > 2 na pesquisa: nenhum alpha encontrado".to_string())
        }
        Some(c) => {
            let (_, h) = evaluate(&d, &c.mis, &c.rs, Some(true), split).unwrap();
            let t = Trial::new("busca", c.s.id(), "protocolo e83eb427 alpha CAPM hedgeado", "2001-2018", &h);
            let df = registry::deflate(&t, &all);
            let hold_ok = c.holdout.as_ref().is_some_and(|e| e.alpha > 0.0);
            let dsr_ok = df.dsr >= 0.90;
            let msg = format!(
                "selecionado {} (IR pesquisa {:.2}, t={:.2}); holdout alpha {} -> {}; DSR {:.0}% com {} tentativas (acaso ~{:.2}) -> {}",
                c.s.id(),
                c.research.ir,
                c.research.t_alpha,
                c.holdout.as_ref().map(|e| pct(e.alpha)).unwrap_or_default(),
                if hold_ok { "ok" } else { "falhou" },
                100.0 * df.dsr,
                df.n_trials,
                df.sr0_ann,
                if dsr_ok { "ok" } else { "falhou" }
            );
            (c, hold_ok && dsr_ok, msg)
        }
    };
    rep.line(format!("Decisão pela regra registrada: {motivo}"));
    rep.line(format!(
        "RESULTADO: {}",
        if validated {
            format!("ALPHA VALIDADO: {} ({})", sel.s.id(), sel.s.desc())
        } else {
            format!("nenhum alpha validado; candidato em observação: {} ({})", sel.s.id(), sel.s.desc())
        }
    ));

    // relevância prática do selecionado: depois de IR, contra ETF e CDI
    if sel.s != Strat::Trend {
        rep.line("");
        rep.line("Na prática (período todo, custo real, 1 pregão de atraso, depois do IR de pessoa física):");
        let plans = ctx.plans(sel.s, start);
        let defl = d.ipca_factor_from(d.ym(start));
        for cap in [100_000.0, 300_000.0] {
            let o = backtest::simulate(&d, &plans, &SimCfg { capital0: cap / defl, lag: 1, costs: REAL, tax: true, cap_adv: None });
            let n = o.ret.len() as f64;
            let w_cdi: f64 = o.mi.iter().map(|&m| 1.0 + d.cdi_m[m]).product();
            let w_mkt: f64 = o.mi.iter().map(|&m| d.nef_vw[m].map(|f| 1.0 + f.rm_rf + f.rf).unwrap_or(1.0 + d.cdi_m[m])).product();
            let after = |w: f64| if w > 1.0 { 1.0 + 0.85 * (w - 1.0) } else { w };
            let g = |w: f64| w.powf(12.0 / n) - 1.0;
            rep.line(format!(
                "  R$ {:.0} mil de hoje: {} {} ao ano | ETF {} | CDI {} (meses isentos {}/{})",
                cap / 1000.0,
                sel.s.id(),
                pct(g(o.final_value / (cap / defl))),
                pct(g(after(w_mkt))),
                pct(g(after(w_cdi))),
                o.exempt_months,
                o.exempt_months + o.taxed_months
            ));
        }
    }

    let out = serde_json::json!({
        "estrategia": sel.s.id(),
        "validado": validated,
        "motivo": motivo,
        "protocolo_sha256": sha,
        "data": util::fmt_date(util::today()),
    });
    std::fs::write(p.results.join("busca_resultado.json"), serde_json::to_string_pretty(&out)?)?;
    let mut csv = String::from("mes");
    for r in &res {
        let _ = write!(csv, ",{}", r.s.id());
    }
    csv.push('\n');
    let mut by_m: HashMap<usize, Vec<String>> = HashMap::new();
    for (k, r) in res.iter().enumerate() {
        for (m, v) in r.mis.iter().zip(&r.rs) {
            by_m.entry(*m).or_insert_with(|| vec![String::new(); res.len()])[k] = format!("{v:.6}");
        }
    }
    let mut ms: Vec<usize> = by_m.keys().copied().collect();
    ms.sort_unstable();
    for m in ms {
        let _ = writeln!(csv, "{},{}", util::fmt_ym(d.ym(m)), by_m[&m].join(","));
    }
    std::fs::write(p.results.join("busca_mensal.csv"), csv)?;
    std::fs::write(p.results.join("busca_relatorio.txt"), &rep.text)?;
    Ok(())
}

/// SHA-256 simples (pra conferir o hash do protocolo sem dependência extra).
pub fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
        0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
        0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
        0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[4 * i], chunk[4 * i + 1], chunk[4 * i + 2], chunk[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|x| format!("{x:08x}")).collect()
}
