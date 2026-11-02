//! Peças comuns aos estudos: calendário mensal, retornos mensais, CDI, fatores, IPCA,
//! formação de carteiras de momentum e métricas.

use crate::macrodata::{self, Factors};
use crate::panel::{Day, Panel};
use crate::stats;
use crate::util::{self, Paths};
use anyhow::Result;
use std::collections::{BTreeMap, HashMap};
use std::ops::Bound::{Excluded, Included};

pub struct Cal {
    pub months: Vec<u32>,
    /// faixa [a, b) de índices de data de cada mês
    pub mrange: Vec<(usize, usize)>,
    pub date_month: Vec<usize>,
}

impl Cal {
    pub fn new(dates: &[u32]) -> Self {
        let mut months: Vec<u32> = Vec::new();
        let mut mrange: Vec<(usize, usize)> = Vec::new();
        let mut date_month = Vec::with_capacity(dates.len());
        for (i, &d) in dates.iter().enumerate() {
            let m = util::ym(d);
            if months.last() != Some(&m) {
                months.push(m);
                mrange.push((i, i + 1));
            } else if let Some(r) = mrange.last_mut() {
                r.1 = i + 1;
            }
            date_month.push(months.len() - 1);
        }
        Cal { months, mrange, date_month }
    }

    /// índice do primeiro mês >= ym
    pub fn idx_from(&self, ym: u32) -> usize {
        self.months.partition_point(|&m| m < ym)
    }
}

pub struct Data {
    pub panel: Panel,
    pub cal: Cal,
    /// retorno mensal [papel][mês] (0 nos dias sem negócio)
    pub mret: Vec<Vec<f64>>,
    pub traded: Vec<Vec<bool>>,
    /// último preço sem ajuste do mês (NaN se não negociou)
    pub last_px: Vec<Vec<f32>>,
    /// CDI acumulado entre o pregão anterior e este, por índice de data
    pub cdi_d: Vec<f64>,
    pub cdi_m: Vec<f64>,
    /// fatores NEFIN mensais (equal-weighted) por índice de mês
    pub nef: Vec<Option<Factors>>,
    pub nef_vw: Vec<Option<Factors>>,
    /// IPCA mensal (AAAAMM -> fração)
    pub ipca: BTreeMap<u32, f64>,
}

impl Data {
    pub fn load(p: &Paths) -> Result<Self> {
        let panel: Panel = util::load_bin(&p.panel)?;
        let cdi = macrodata::load_cdi(&p.raw_bcb)?;
        let ew = macrodata::load_factors(&p.raw_nefin.join("nefin_factors_equally_weighted_monthly.csv"), true)?;
        let vw = macrodata::load_factors(&p.raw_nefin.join("nefin_factors_monthly.csv"), true)?;
        let ipca = macrodata::load_ipca(&p.raw_bcb)?;
        Ok(Self::build(panel, &cdi, &ew, &vw, ipca))
    }

    /// Monta os dados a partir do painel e das séries macro já lidas: CDI diário (AAAAMMDD ->
    /// fração), fatores NEFIN mensais equal- e value-weighted (AAAAMM) e IPCA mensal (AAAAMM).
    /// Serve também para dados sintéticos em testes.
    pub fn build(
        panel: Panel,
        cdi: &BTreeMap<u32, f64>,
        ew: &BTreeMap<u32, Factors>,
        vw: &BTreeMap<u32, Factors>,
        ipca: BTreeMap<u32, f64>,
    ) -> Self {
        let cal = Cal::new(&panel.dates);
        let (nm, nt, nd) = (cal.months.len(), panel.tks.len(), panel.dates.len());
        let mut mret = vec![vec![0.0f64; nm]; nt];
        let mut traded = vec![vec![false; nm]; nt];
        let mut last_px = vec![vec![f32::NAN; nm]; nt];
        for (t, tk) in panel.tks.iter().enumerate() {
            for d in &tk.days {
                let m = cal.date_month[d.di as usize];
                mret[t][m] = (1.0 + mret[t][m]) * (1.0 + d.r) - 1.0;
                traded[t][m] = true;
                last_px[t][m] = d.px;
            }
        }
        let mut cdi_d = vec![0.0f64; nd];
        for i in 0..nd {
            let cur = panel.dates[i];
            let g: f64 = if i == 0 {
                cdi.get(&cur).map(|r| 1.0 + r).unwrap_or(1.0)
            } else {
                cdi.range((Excluded(panel.dates[i - 1]), Included(cur))).map(|(_, r)| 1.0 + r).product()
            };
            cdi_d[i] = g - 1.0;
        }
        let cdi_m = cal.mrange.iter().map(|&(a, b)| compound(&cdi_d[a..b])).collect();
        let nef = cal.months.iter().map(|m| ew.get(m).copied()).collect();
        let nef_vw = cal.months.iter().map(|m| vw.get(m).copied()).collect();
        Data { panel, cal, mret, traded, last_px, cdi_d, cdi_m, nef, nef_vw, ipca }
    }

    /// Janela [s, e) de índices de data em que a carteira do mês `mi` fica montada, com `lag`
    /// pregões de atraso (lag 0 = compra no fechamento do último pregão do mês anterior).
    pub fn window(&self, mi: usize, lag: usize) -> (usize, usize) {
        let n = self.panel.dates.len();
        let s = self.cal.mrange[mi].0 + lag;
        let e = if mi + 1 < self.cal.months.len() { self.cal.mrange[mi + 1].0 + lag } else { n };
        (s.min(n), e.min(n))
    }

    /// Inflação acumulada (IPCA) de ym0 até o último mês disponível.
    pub fn ipca_factor_from(&self, ym0: u32) -> f64 {
        self.ipca.range(ym0..).map(|(_, r)| 1.0 + r).product()
    }

    pub fn ym(&self, mi: usize) -> u32 {
        self.cal.months[mi]
    }

    /// Índice do primeiro pregão do mês mi. Para mi = nº de meses (o próximo mês, ainda sem
    /// pregões), devolve o fim dos dados: serve pra calcular a carteira do mês que vem.
    pub fn month_start(&self, mi: usize) -> usize {
        if mi < self.cal.months.len() { self.cal.mrange[mi].0 } else { self.panel.dates.len() }
    }

    /// AAAAMM do mês mi, aceitando o próximo mês ainda sem pregões.
    pub fn ym_any(&self, mi: usize) -> u32 {
        if mi < self.cal.months.len() {
            self.cal.months[mi]
        } else {
            util::ym_add(*self.cal.months.last().unwrap(), (mi - self.cal.months.len() + 1) as i32)
        }
    }
}

pub fn compound(x: &[f64]) -> f64 {
    x.iter().map(|r| 1.0 + r).product::<f64>() - 1.0
}

/// dias do papel com índice de data em [s, e)
pub fn days_in(days: &[Day], s: usize, e: usize) -> &[Day] {
    let lo = days.partition_point(|d| (d.di as usize) < s);
    let hi = days.partition_point(|d| (d.di as usize) < e);
    &days[lo..hi]
}

/// Retorno diário equal-weighted com rebalanceamento diário (r=0 sem negócio), como o NEFIN.
pub fn daily_ew(panel: &Panel, set: &[usize], s: usize, e: usize) -> Vec<f64> {
    let mut acc = vec![0.0f64; e.saturating_sub(s)];
    if set.is_empty() {
        return acc;
    }
    for &t in set {
        for d in days_in(&panel.tks[t].days, s, e) {
            acc[d.di as usize - s] += d.r;
        }
    }
    let inv = 1.0 / set.len() as f64;
    acc.iter_mut().for_each(|v| *v *= inv);
    acc
}

/// Momentum: retorno acumulado de t-12 a t-2 (pula t-1).
pub fn mom_signal(mret: &[Vec<f64>], t: usize, mi: usize) -> f64 {
    (mi - 12..=mi - 2).map(|m| 1.0 + mret[t][m]).product::<f64>() - 1.0
}

pub struct Form {
    pub mi: usize,
    /// candidatos vivos na formação (negociaram em t-1 e passaram no preço mínimo)
    pub alive: Vec<usize>,
    pub winners: Vec<usize>,
    pub losers: Vec<usize>,
}

/// Terciles de momentum mês a mês. `univ(ano)` devolve os candidatos do ano.
pub fn momentum_forms(d: &Data, univ: &mut dyn FnMut(u32) -> Vec<usize>, min_stocks: usize, min_px: f32) -> Vec<Form> {
    let mut cache: HashMap<u32, Vec<usize>> = HashMap::new();
    let mut out = Vec::new();
    for mi in 12..d.cal.months.len() {
        let y = d.cal.months[mi] / 100;
        let cands = cache.entry(y).or_insert_with(|| univ(y));
        let alive: Vec<usize> = cands
            .iter()
            .copied()
            .filter(|&t| d.traded[t][mi - 1] && d.last_px[t][mi - 1] >= min_px)
            .collect();
        if alive.len() < min_stocks {
            continue;
        }
        let mut sig: Vec<(usize, f64)> = alive.iter().map(|&t| (t, mom_signal(&d.mret, t, mi))).collect();
        sig.sort_by(|a, b| a.1.total_cmp(&b.1));
        let n = sig.len();
        let k3 = ((n as f64) / 3.0).round() as usize;
        out.push(Form {
            mi,
            losers: sig[..k3].iter().map(|x| x.0).collect(),
            winners: sig[n - k3..].iter().map(|x| x.0).collect(),
            alive,
        });
    }
    out
}

pub struct Perf {
    pub cagr: f64,
    pub vol: f64,
    /// Sharpe sobre o CDI
    pub sharpe: f64,
    pub mdd: f64,
}

pub fn perf(x: &[f64], cdi: &[f64]) -> Perf {
    let ex: Vec<f64> = x.iter().zip(cdi).map(|(a, b)| a - b).collect();
    Perf {
        cagr: stats::cagr(x, 12.0),
        vol: stats::std(x) * 12f64.sqrt(),
        sharpe: stats::mean(&ex) / stats::std(&ex) * 12f64.sqrt(),
        mdd: stats::max_dd(x),
    }
}

/// Alpha anualizado (e t de Newey-West) contra os fatores NEFIN; `with_wml` inclui o WML.
/// `y` são retornos já em excesso ao Rf do NEFIN.
pub fn alpha(y: &[f64], f: &[Factors], with_wml: bool) -> Option<(f64, f64)> {
    let col = |g: fn(&Factors) -> f64| -> Vec<f64> { f.iter().map(g).collect() };
    let (rm, smb, hml, iml, wml) = (
        col(|x| x.rm_rf),
        col(|x| x.smb),
        col(|x| x.hml),
        col(|x| x.iml),
        col(|x| x.wml),
    );
    let mut xs: Vec<&[f64]> = vec![&rm, &smb, &hml, &iml];
    if with_wml {
        xs.push(&wml);
    }
    let o = stats::ols_nw(y, &xs, stats::nw_lags(y.len()))?;
    Some((o.coef[0] * 12.0, o.t[0]))
}

/// Saída que vai pra tela e pro relatório em arquivo.
#[derive(Default)]
pub struct Report {
    pub text: String,
}

impl Report {
    pub fn line(&mut self, s: impl AsRef<str>) {
        println!("{}", s.as_ref());
        self.text.push_str(s.as_ref());
        self.text.push('\n');
    }
}
