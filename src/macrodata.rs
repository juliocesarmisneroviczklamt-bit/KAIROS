//! CDI diário (BCB/SGS série 12) e fatores de risco do NEFIN-USP.

use crate::{http, util};
use anyhow::{Context, Result, anyhow};
use std::collections::BTreeMap;
use std::path::Path;

pub const NEFIN_FILES: [&str; 4] = [
    "nefin_factors.csv",
    "nefin_factors_equally_weighted.csv",
    "nefin_factors_monthly.csv",
    "nefin_factors_equally_weighted_monthly.csv",
];

pub fn fetch_nefin(dir: &Path) -> Result<()> {
    let agent = http::agent()?;
    for f in NEFIN_FILES {
        let url = format!("https://nefin.com.br/nefindata/risk-factors/{f}");
        let n = http::download(&agent, &url, &dir.join(f))?;
        println!("  NEFIN {f}: {:.0} KB", n as f64 / 1e3);
    }
    Ok(())
}

/// A API do SGS limita séries diárias a janelas de 10 anos: baixa em blocos de 5.
pub fn fetch_cdi(dir: &Path, from_year: u32) -> Result<()> {
    let agent = http::agent()?;
    let today = util::today();
    let last_year = util::year(today);
    let mut out = String::from("data,cdi_pct_dia\n");
    let (mut y, mut n) = (from_year, 0usize);
    while y <= last_year {
        let y2 = (y + 4).min(last_year);
        let fim = if y2 == last_year { util::fmt_date(today) } else { format!("{y2}-12-31") };
        let fim = format!("{}/{}/{}", &fim[8..10], &fim[5..7], &fim[0..4]);
        let url = format!(
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial=01/01/{y}&dataFinal={fim}"
        );
        let body = http::get_text(&agent, &url)?;
        let v: Vec<serde_json::Value> = serde_json::from_str(&body)
            .with_context(|| format!("CDI {y}-{y2}: resposta inesperada: {}", &body[..body.len().min(200)]))?;
        for it in v {
            let d = it.get("data").and_then(|x| x.as_str()).and_then(util::parse_ddmmyyyy);
            let val = it.get("valor").and_then(|x| x.as_str()).and_then(|s| s.trim().parse::<f64>().ok());
            if let (Some(d), Some(val)) = (d, val) {
                out.push_str(&format!("{d},{val}\n"));
                n += 1;
            }
        }
        y = y2 + 1;
    }
    std::fs::write(dir.join("cdi.csv"), out)?;
    println!("  CDI: {n} dias");
    Ok(())
}

/// Atualiza só o fim da série do CDI (rápido; usado no paper trading).
pub fn fetch_cdi_recent(dir: &Path) -> Result<()> {
    let path = dir.join("cdi.csv");
    let mut m: BTreeMap<u32, String> = BTreeMap::new();
    if let Ok(txt) = std::fs::read_to_string(&path) {
        for line in txt.lines().skip(1) {
            if let Some((d, v)) = line.split_once(',') {
                if let Ok(d) = d.parse::<u32>() {
                    m.insert(d, v.to_string());
                }
            }
        }
    }
    let Some(&last) = m.keys().last() else { return fetch_cdi(dir, 1995) };
    let today = util::today();
    let ini = util::add_days(last, -10);
    let f = |d: u32| format!("{:02}/{:02}/{}", d % 100, (d / 100) % 100, d / 10000);
    let url = format!(
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json&dataInicial={}&dataFinal={}",
        f(ini),
        f(today)
    );
    let agent = http::agent()?;
    let body = http::get_text(&agent, &url)?;
    let v: Vec<serde_json::Value> = serde_json::from_str(&body)
        .with_context(|| format!("CDI recente: resposta inesperada: {}", &body[..body.len().min(200)]))?;
    for it in v {
        let d = it.get("data").and_then(|x| x.as_str()).and_then(util::parse_ddmmyyyy);
        let val = it.get("valor").and_then(|x| x.as_str()).map(|s| s.trim().to_string());
        if let (Some(d), Some(val)) = (d, val) {
            m.insert(d, val);
        }
    }
    let mut out = String::from("data,cdi_pct_dia\n");
    for (d, v) in &m {
        out.push_str(&format!("{d},{v}\n"));
    }
    std::fs::write(&path, out)?;
    Ok(())
}

/// IPCA mensal (BCB/SGS série 433), pra expressar capital em reais de hoje.
pub fn fetch_ipca(dir: &Path) -> Result<()> {
    let agent = http::agent()?;
    let today = util::today();
    let fim = format!("{:02}/{:02}/{}", today % 100, (today / 100) % 100, today / 10000);
    let url = format!(
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados?formato=json&dataInicial=01/01/1995&dataFinal={fim}"
    );
    let body = http::get_text(&agent, &url)?;
    let v: Vec<serde_json::Value> = serde_json::from_str(&body)
        .with_context(|| format!("IPCA: resposta inesperada: {}", &body[..body.len().min(200)]))?;
    let mut out = String::from("mes,ipca_pct_mes\n");
    let mut n = 0;
    for it in v {
        let d = it.get("data").and_then(|x| x.as_str()).and_then(util::parse_ddmmyyyy);
        let val = it.get("valor").and_then(|x| x.as_str()).and_then(|s| s.trim().parse::<f64>().ok());
        if let (Some(d), Some(val)) = (d, val) {
            out.push_str(&format!("{},{val}\n", util::ym(d)));
            n += 1;
        }
    }
    std::fs::write(dir.join("ipca.csv"), out)?;
    println!("  IPCA: {n} meses");
    Ok(())
}

/// AAAAMM -> inflação do mês em fração
pub fn load_ipca(dir: &Path) -> Result<BTreeMap<u32, f64>> {
    let p = dir.join("ipca.csv");
    let txt = std::fs::read_to_string(&p).with_context(|| format!("abrindo {} (rode fetch-macro)", p.display()))?;
    let mut m = BTreeMap::new();
    for line in txt.lines().skip(1) {
        let mut it = line.split(',');
        if let (Some(d), Some(v)) = (it.next(), it.next()) {
            if let (Ok(d), Ok(v)) = (d.parse::<u32>(), v.parse::<f64>()) {
                m.insert(d, v / 100.0);
            }
        }
    }
    Ok(m)
}

/// data AAAAMMDD -> taxa diária em fração (0,0004 = 0,04% ao dia)
pub fn load_cdi(dir: &Path) -> Result<BTreeMap<u32, f64>> {
    let p = dir.join("cdi.csv");
    let txt = std::fs::read_to_string(&p).with_context(|| format!("abrindo {} (rode fetch-macro)", p.display()))?;
    let mut m = BTreeMap::new();
    for line in txt.lines().skip(1) {
        let mut it = line.split(',');
        if let (Some(d), Some(v)) = (it.next(), it.next()) {
            if let (Ok(d), Ok(v)) = (d.parse::<u32>(), v.parse::<f64>()) {
                m.insert(d, v / 100.0);
            }
        }
    }
    Ok(m)
}

#[derive(Clone, Copy, Debug)]
pub struct Factors {
    pub rm_rf: f64,
    pub smb: f64,
    pub hml: f64,
    pub wml: f64,
    pub iml: f64,
    pub rf: f64,
}

/// Lê um CSV do NEFIN. Chave: AAAAMMDD (diário) ou AAAAMM (mensal).
pub fn load_factors(path: &Path, monthly: bool) -> Result<BTreeMap<u32, Factors>> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path)
        .with_context(|| format!("abrindo {} (rode fetch-macro)", path.display()))?;
    let h = rdr.headers()?.clone();
    let col = |name: &str| {
        h.iter()
            .position(|x| x.trim_matches('"') == name)
            .ok_or_else(|| anyhow!("coluna {name} ausente em {}", path.display()))
    };
    let (c_rm, c_smb, c_hml, c_wml, c_iml, c_rf) =
        (col("Rm_minus_Rf")?, col("SMB")?, col("HML")?, col("WML")?, col("IML")?, col("Risk_Free")?);
    let (k1, k2) = if monthly { (col("year")?, col("month")?) } else { (col("Date")?, 0) };
    let mut out = BTreeMap::new();
    for rec in rdr.records() {
        let rec = rec?;
        let key = if monthly {
            match (rec[k1].trim().parse::<u32>(), rec[k2].trim().parse::<u32>()) {
                (Ok(y), Ok(m)) => y * 100 + m,
                _ => continue,
            }
        } else {
            match util::parse_iso(&rec[k1]) {
                Some(d) => d,
                None => continue,
            }
        };
        let g = |c: usize| rec.get(c).and_then(|x| x.trim().parse::<f64>().ok()).unwrap_or(f64::NAN);
        out.insert(
            key,
            Factors { rm_rf: g(c_rm), smb: g(c_smb), hml: g(c_hml), wml: g(c_wml), iml: g(c_iml), rf: g(c_rf) },
        );
    }
    Ok(out)
}
