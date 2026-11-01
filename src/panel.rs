//! Retorno total diário por papel: preço + proventos + desdobramentos/grupamentos + cisões +
//! direitos de subscrição. Saltos sem evento conhecido passam por um detector conservador de
//! desdobramento e, se continuarem estranhos, vão pra um CSV de diagnóstico.

use crate::b3events::{Ev, Event};
use crate::store::{Bar, BarStore};
use crate::util;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Day {
    /// índice no calendário `Panel::dates`
    pub di: u32,
    /// retorno total desde o pregão anterior em que o papel negociou
    pub r: f64,
    /// volume financeiro do dia (R$)
    pub vol: f64,
    /// parte de `r` que veio de dividendos/rendimentos (isentos de IR pra pessoa física)
    pub div: f32,
    /// parte de `r` que veio de JCP (bruto; 15% retidos na fonte)
    pub jcp: f32,
    /// spread relativo entre melhor compra e melhor venda no fechamento (NaN se não houver)
    pub spr: f32,
    /// preço de fechamento sem ajuste (R$ por ação)
    pub px: f32,
}

fn spread(b: &Bar) -> f32 {
    if b.bid > 0.0 && b.ask >= b.bid {
        ((b.ask - b.bid) / (0.5 * (b.ask + b.bid))) as f32
    } else {
        f32::NAN
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Tk {
    pub ticker: String,
    pub issuer: String,
    pub class: String,
    pub first: u32,
    pub last: u32,
    pub days: Vec<Day>,
}

#[derive(Serialize, Deserialize, Default)]
pub struct Panel {
    pub dates: Vec<u32>,
    pub tks: Vec<Tk>,
}

#[derive(Default)]
pub struct BuildStats {
    pub splits_b3: usize,
    pub splits_inferred: usize,
    pub cash_applied: usize,
    pub spin_applied: usize,
    pub spin_unresolved: usize,
    pub rights_applied: usize,
    pub b3_dropped: usize,
    pub b3_corrected: usize,
    pub neutralized: usize,
    pub event_residuals: usize,
    pub big_moves: usize,
    pub returns: usize,
}

impl BuildStats {
    pub fn print(&self) {
        println!("Ajuste de retornos ({} retornos diários):", self.returns);
        println!(
            "  desdobr./grupam. B3: {} (descartados por não aparecerem no preço: {}, fator corrigido: {}) | inferidos (marcador EB/EG ou razão enorme exata): {}",
            self.splits_b3, self.b3_dropped, self.b3_corrected, self.splits_inferred
        );
        println!(
            "  proventos: {} | cisões: {} (sem preço: {}) | subscrições: {}",
            self.cash_applied, self.spin_applied, self.spin_unresolved, self.rights_applied
        );
        println!(
            "  dias com evento de ações e |retorno| > 40% (conferir): {} | neutralizados (>+200%/<-90% sem explicação): {} | extremos mantidos (>+150%/<-60%): {}",
            self.event_residuals, self.neutralized, self.big_moves
        );
    }
}

/// Múltiplos "redondos" de desdobramento/grupamento (penny stocks dos anos 2000 tiveram
/// grupamentos de dezenas de milhares pra 1, ex.: TLCP4 em 2005).
const NICE: [f64; 41] = [
    2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0, 100.0,
    120.0, 150.0, 200.0, 250.0, 300.0, 400.0, 500.0, 1000.0, 2000.0, 2500.0, 4000.0, 5000.0, 10000.0, 20000.0,
    25000.0, 50000.0, 100000.0, 200000.0, 250000.0, 500000.0, 1000000.0,
];

/// Multiplicador "redondo" m tal que q*m ≈ 1 (erro relativo em log < tol), com o erro.
fn nearest_nice(q: f64, tol: f64) -> Option<(f64, f64)> {
    let target = 1.0 / q;
    let mut best: Option<(f64, f64)> = None;
    for &k in NICE.iter() {
        for m in [k, 1.0 / k] {
            let err = (target / m).ln().abs();
            if err < tol && best.is_none_or(|(_, e)| err < e) {
                best = Some((m, err));
            }
        }
    }
    best
}

/// Marcadores de evento de quantidade no ESPECI (ex.: "ON  EG  NM" = ex-grupamento).
pub fn stock_marker(especi: &str) -> bool {
    especi.split(|c: char| c.is_whitespace() || c == '*').filter(|t| !t.is_empty()).skip(1).any(|tok| {
        tok.len() >= 2
            && tok.starts_with('E')
            && !matches!(tok, "ED" | "EJ" | "EDJ" | "ES" | "ER" | "EJS" | "EDS")
            && (tok.contains('G') || tok.contains('X') || tok.contains('B'))
    })
}

fn median(mut v: Vec<f64>) -> f64 {
    v.retain(|x| x.is_finite() && *x > 0.0);
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

/// (razão de quantidade negociada, razão de volume financeiro): 10 pregões depois / 10 antes.
fn vol_ratios(bars: &[Bar], i: usize) -> (f64, f64) {
    let lo = i.saturating_sub(10);
    let hi = (i + 10).min(bars.len());
    let (before, after) = (&bars[lo..i], &bars[i..hi]);
    let q = median(after.iter().map(|b| b.qty).collect()) / median(before.iter().map(|b| b.qty).collect());
    let v = median(after.iter().map(|b| b.vol_brl).collect()) / median(before.iter().map(|b| b.vol_brl).collect());
    (q, v)
}

/// Salto de preço sem evento da B3. Volume NÃO serve de prova: depois de um crash o mercado
/// segue negociando os mesmos R$ a um preço menor, então a quantidade sobe igualzinho a um
/// desdobramento (caso AMER3, 12/01/2023). Só aceita com:
///  - marcador EB/EG novo no ESPECI naquele pregão + razão redonda (erro < 10%), ou
///  - razão enorme e exata (>= ~10x, erro < 3%), que não acontece organicamente num dia.
fn infer_split(bars: &[Bar], i: usize, q: f64) -> Option<(f64, &'static str)> {
    let new_marker = stock_marker(&bars[i].especi) && !stock_marker(&bars[i - 1].especi);
    if new_marker {
        if let Some((m, _)) = nearest_nice(q, 0.10) {
            return Some((m, "marcador+razao redonda"));
        }
        // marcador de evento de quantidade mas razão quebrada (ex.: GFSA3 13,48:1 em 2017,
        // ausente na B3): neutraliza o dia em vez de inventar um retorno de +1250%
        return Some((1.0 / q, "marcador+razao quebrada: neutralizado"));
    }
    match nearest_nice(q, 0.03) {
        Some((m, _)) if m.ln().abs() >= 9.5f64.ln() => Some((m, "razao enorme exata")),
        _ => None,
    }
}

pub fn build(store: &BarStore, evs: &[Vec<Event>], diag: &Path) -> Result<(Panel, BuildStats)> {
    let mut dates: Vec<u32> = store.series.iter().flat_map(|s| s.bars.iter().map(|b| b.date)).collect();
    dates.sort_unstable();
    dates.dedup();
    let di = |d: u32| dates.binary_search(&d).expect("data fora do calendário") as u32;
    let mut by_isin: HashMap<&str, usize> = HashMap::new();
    for (i, s) in store.series.iter().enumerate() {
        for isin in &s.isins {
            by_isin.insert(isin.as_str(), i);
        }
    }

    let mut st = BuildStats::default();
    let mut inferred = String::from("ticker,data,preco_ant,preco,mult_inferido,razao_qtd,razao_vol_rs,especi,motivo\n");
    let mut big = String::from("ticker,data,preco_ant,preco,mult,proventos,retorno,eventos,especi\n");
    let mut resid = String::from("ticker,data,preco_ant,preco,mult,retorno,eventos\n");
    let mut tks = Vec::with_capacity(store.series.len());

    for (k, s) in store.series.iter().enumerate() {
        let bars = &s.bars;
        if bars.is_empty() {
            continue;
        }
        let ev = &evs[k];
        let mut j = 0usize;
        let mut days = Vec::with_capacity(bars.len());
        days.push(Day {
            di: di(bars[0].date),
            r: 0.0,
            vol: bars[0].vol_brl,
            div: 0.0,
            jcp: 0.0,
            spr: spread(&bars[0]),
            px: bars[0].close as f32,
        });
        for i in 1..bars.len() {
            let (b0, b1) = (&bars[i - 1], &bars[i]);
            let (p0, p1) = (b0.close, b1.close);
            let (mut m, mut y, mut split_ev) = (1.0f64, 0.0f64, false);
            let (mut y_div, mut y_jcp) = (0.0f64, 0.0f64);
            let mut labels: Vec<&str> = Vec::new();
            // eventos com data-com em [d0, d1): o ajuste cai no pregão d1
            while j < ev.len() && ev[j].datacom < b1.date {
                let e = &ev[j];
                j += 1;
                if e.datacom < b0.date {
                    continue;
                }
                labels.push(e.label.as_str());
                match &e.ev {
                    Ev::Split(mult) => {
                        m *= *mult;
                        split_ev = true;
                        st.splits_b3 += 1;
                    }
                    Ev::Cash { y: yy, jcp } => {
                        y += *yy;
                        if *jcp {
                            y_jcp += *yy;
                        } else {
                            y_div += *yy;
                        }
                        st.cash_applied += 1;
                    }
                    Ev::SpinOff { isin, ratio } => {
                        let px = by_isin
                            .get(isin.as_str())
                            .and_then(|&t| {
                                let a = &store.series[t];
                                a.idx_after(e.datacom).map(|ix| &a.bars[ix])
                            })
                            .filter(|b| b.date <= util::add_days(e.datacom, 60))
                            .map(|b| b.close);
                        match px {
                            Some(c) => {
                                y += ratio * c / p0;
                                st.spin_applied += 1;
                            }
                            None => st.spin_unresolved += 1,
                        }
                    }
                    Ev::Rights { isin, s: sr, price } => {
                        let pi = if s.isins.iter().any(|x| x == isin) {
                            Some(p0)
                        } else {
                            by_isin.get(isin.as_str()).and_then(|&t| {
                                let a = &store.series[t];
                                a.idx_on_or_before(e.datacom).map(|ix| a.bars[ix].close)
                            })
                        };
                        if let Some(pi) = pi {
                            let v = sr * (pi - price).max(0.0) / (1.0 + sr);
                            if v > 0.0 {
                                y += v / p0;
                                st.rights_applied += 1;
                            }
                        }
                    }
                }
            }
            let q_raw = p1 / p0;
            if split_ev && (q_raw * m).ln().abs() > 2f64.ln() {
                let m_b3 = m;
                if q_raw.ln().abs() < 1.25f64.ln() {
                    // o preço não mexeu: o evento não aconteceu neste pregão (data ou fator errado na B3)
                    m = 1.0;
                    st.b3_dropped += 1;
                } else if (q_raw * m).ln().abs() > 3f64.ln() {
                    // fator da B3 incompatível com o salto de preço: usa a razão redonda observada
                    if let Some((mm, _)) = nearest_nice(q_raw, 0.15) {
                        if (q_raw * mm).ln().abs() < (q_raw * m).ln().abs() {
                            m = mm;
                            st.b3_corrected += 1;
                        }
                    }
                }
                if m != m_b3 {
                    let _ = writeln!(
                        resid,
                        "{},{},{:.6},{:.6},{:.6},{:.4},{} (fator B3 {:.6} -> {:.6})",
                        s.ticker,
                        util::fmt_date(b1.date),
                        p0,
                        p1,
                        m,
                        p1 * m / p0 - 1.0 + y,
                        labels.join("|"),
                        m_b3,
                        m
                    );
                }
            }
            if !split_ev && (q_raw > 1.8 || q_raw < 0.55) {
                if let Some((mm, why)) = infer_split(bars, i, q_raw) {
                    m = mm;
                    split_ev = true;
                    st.splits_inferred += 1;
                    let (vq, vb) = vol_ratios(bars, i);
                    let _ = writeln!(
                        inferred,
                        "{},{},{:.6},{:.6},{:.6},{:.3},{:.3},{},{}",
                        s.ticker,
                        util::fmt_date(b1.date),
                        p0,
                        p1,
                        mm,
                        vq,
                        vb,
                        b1.especi,
                        why
                    );
                }
            }
            let mut r = p1 * m / p0 - 1.0 + y;
            // filtro final: >+200% ou <-90% num dia, sem explicação, é erro de dado ou evento
            // não registrado. Neutraliza (fica só o provento) e registra.
            if 1.0 + r > 3.0 || 1.0 + r < 0.10 {
                st.neutralized += 1;
                let _ = writeln!(
                    big,
                    "{},{},{:.6},{:.6},{:.6},{:.4},{:.4},NEUTRALIZADO{}{},{}",
                    s.ticker,
                    util::fmt_date(b1.date),
                    p0,
                    p1,
                    m,
                    y,
                    r,
                    if labels.is_empty() { "" } else { "|" },
                    labels.join("|"),
                    b1.especi
                );
                r = y.min(0.6);
            }
            if split_ev && r.abs() > 0.4 {
                st.event_residuals += 1;
                let _ = writeln!(
                    resid,
                    "{},{},{:.6},{:.6},{:.6},{:.4},{}",
                    s.ticker,
                    util::fmt_date(b1.date),
                    p0,
                    p1,
                    m,
                    r,
                    labels.join("|")
                );
            }
            if r > 1.5 || r < -0.6 {
                st.big_moves += 1;
                let _ = writeln!(
                    big,
                    "{},{},{:.6},{:.6},{:.6},{:.4},{:.4},{},{}",
                    s.ticker,
                    util::fmt_date(b1.date),
                    p0,
                    p1,
                    m,
                    y,
                    r,
                    labels.join("|"),
                    b1.especi
                );
            }
            st.returns += 1;
            days.push(Day {
                di: di(b1.date),
                r,
                vol: b1.vol_brl,
                div: y_div as f32,
                jcp: y_jcp as f32,
                spr: spread(b1),
                px: b1.close as f32,
            });
        }
        tks.push(Tk {
            ticker: s.ticker.clone(),
            issuer: s.issuer(),
            class: s.class.clone(),
            first: s.first(),
            last: s.last(),
            days,
        });
    }
    std::fs::write(diag.join("splits_inferidos.csv"), inferred)?;
    std::fs::write(diag.join("movimentos_extremos.csv"), big)?;
    std::fs::write(diag.join("residuos_eventos.csv"), resid)?;
    Ok((Panel { dates, tks }, st))
}
