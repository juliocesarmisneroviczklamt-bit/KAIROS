//! Universos no estilo NEFIN: para o ano t, usando só dados de t-1.

use crate::panel::Panel;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum LiqRule {
    /// >80% dos pregões do ano anterior com volume acima do mínimo (leitura literal do NEFIN)
    Dias,
    /// >80% dos pregões negociados e volume médio acima do mínimo
    Media,
}

/// Liquidez de um papel no ano anterior.
pub struct YStat {
    pub vol: f64,
    pub n_traded: u32,
    /// pregões do ano (contando desde a listagem, se listou no meio do ano)
    pub n_window: u32,
    /// pregões com volume acima do mínimo
    pub n_liq: u32,
    pub med_vol: f64,
}

fn year_range(dates: &[u32], y: u32) -> (usize, usize) {
    let a = dates.partition_point(|&d| d < y * 10000 + 101);
    let b = dates.partition_point(|&d| d < (y + 1) * 10000 + 101);
    (a, b)
}

/// Estatísticas de year-1 dos papéis listados antes de dezembro de year-1.
pub fn year_stats(panel: &Panel, year: u32, min_vol: f64) -> HashMap<usize, YStat> {
    let py = year - 1;
    let (a, b) = year_range(&panel.dates, py);
    let mut out = HashMap::new();
    if a == b {
        return out;
    }
    let dec1 = py * 10000 + 1201;
    for (t, tk) in panel.tks.iter().enumerate() {
        // ETFs (classe CI) ficam fora dos universos de ações
        if tk.first >= dec1 || tk.class == "CI" {
            continue;
        }
        let lo = tk.days.partition_point(|d| (d.di as usize) < a);
        let hi = tk.days.partition_point(|d| (d.di as usize) < b);
        if lo == hi {
            continue;
        }
        let mut vols: Vec<f64> = tk.days[lo..hi].iter().map(|d| d.vol).collect();
        let vol: f64 = vols.iter().sum();
        let n_liq = vols.iter().filter(|&&v| v > min_vol).count() as u32;
        let start = a.max(panel.dates.partition_point(|&d| d < tk.first));
        vols.sort_by(|x, y| x.total_cmp(y));
        out.insert(
            t,
            YStat {
                vol,
                n_traded: (hi - lo) as u32,
                n_window: b.saturating_sub(start) as u32,
                n_liq,
                med_vol: vols[vols.len() / 2],
            },
        );
    }
    out
}

/// Classe mais negociada de cada empresa (prefixo de 4 letras do código).
pub fn leaders(panel: &Panel, st: &HashMap<usize, YStat>) -> Vec<usize> {
    let mut best: HashMap<&str, (usize, f64)> = HashMap::new();
    for (&t, s) in st {
        let e = best.entry(panel.tks[t].issuer.as_str()).or_insert((t, -1.0));
        if s.vol > e.1 || (s.vol == e.1 && t < e.0) {
            *e = (t, s.vol);
        }
    }
    let mut v: Vec<usize> = best.into_values().map(|(t, _)| t).collect();
    v.sort_unstable();
    v
}

pub fn passes_nefin(s: &YStat, rule: LiqRule, min_vol: f64) -> bool {
    if s.n_window == 0 {
        return false;
    }
    let nw = s.n_window as f64;
    match rule {
        LiqRule::Dias => s.n_liq as f64 / nw > 0.8,
        LiqRule::Media => s.n_traded as f64 / nw > 0.8 && s.vol / s.n_traded as f64 > min_vol,
    }
}

/// Elegíveis no ano `year` (NEFIN): classe mais negociada da empresa, líquida em year-1,
/// listada antes de dezembro de year-1.
pub fn eligible(panel: &Panel, year: u32, rule: LiqRule, min_vol: f64) -> Vec<usize> {
    let st = year_stats(panel, year, min_vol);
    leaders(panel, &st).into_iter().filter(|t| passes_nefin(&st[t], rule, min_vol)).collect()
}

/// Segmento "onde fundo grande não entra": classe mais negociada da empresa que NÃO passa no
/// filtro de liquidez do NEFIN, mas negociou em >80% dos pregões com volume mediano >= min_med_vol.
pub fn small_segment(panel: &Panel, year: u32, nefin_min_vol: f64, min_med_vol: f64) -> Vec<usize> {
    let st = year_stats(panel, year, nefin_min_vol);
    leaders(panel, &st)
        .into_iter()
        .filter(|t| {
            let s = &st[t];
            s.n_window > 0
                && !passes_nefin(s, LiqRule::Dias, nefin_min_vol)
                && s.n_traded as f64 / s.n_window as f64 > 0.8
                && s.med_vol >= min_med_vol
        })
        .collect()
}
