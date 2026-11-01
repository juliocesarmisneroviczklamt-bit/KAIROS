//! Eventos societários da B3: proventos em dinheiro (dividendos, JCP, rendimentos), desdobramentos,
//! grupamentos, bonificações, cisões e subscrições. Inclui empresas que já saíram da bolsa.

use crate::http;
use crate::store::{BarStore, issuer_of};
use crate::util;
use anyhow::{Context, Result};
use base64::Engine as _;
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

const BASE: &str = "https://sistemaswebb3-listados.b3.com.br/listedCompaniesProxy/CompanyCall";

fn b64(s: &str) -> String {
    base64::engine::general_purpose::STANDARD.encode(s.as_bytes())
}

fn supp_path(dir: &Path, issuer: &str) -> PathBuf {
    dir.join(format!("supp_{}.json", util::slug(issuer)))
}

fn cash_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("cash_{}.json", util::slug(name)))
}

// ------------------------------------------------------------------ download

enum Job {
    Supp(String),
    Cash(String),
}

pub fn fetch(store: &BarStore, dir: &Path, threads: usize, refresh: bool) -> Result<()> {
    let mut issuers = BTreeSet::new();
    let mut names = BTreeSet::new();
    for s in &store.series {
        if s.class == "CI" {
            continue;
        }
        issuers.insert(issuer_of(&s.ticker));
        for a in &s.aliases {
            issuers.insert(issuer_of(a));
        }
        for (n, _, _) in &s.names {
            if !n.is_empty() {
                names.insert(n.clone());
            }
        }
    }
    let mut jobs: Vec<Job> = issuers.into_iter().map(Job::Supp).collect();
    jobs.extend(names.into_iter().map(Job::Cash));
    let jobs: Vec<Job> = jobs
        .into_iter()
        .filter(|j| {
            refresh
                || !match j {
                    Job::Supp(i) => supp_path(dir, i).exists(),
                    Job::Cash(n) => cash_path(dir, n).exists(),
                }
        })
        .collect();
    run_jobs(jobs, dir, threads)
}

/// Atualiza (baixa de novo) os eventos das ações negociadas e líquidas nas últimas semanas,
/// pra pegar proventos recém-anunciados. Uma vez por dia no máximo.
pub fn refresh_active(store: &BarStore, dir: &Path, threads: usize) -> Result<()> {
    let today = util::today();
    let marker = dir.join(format!(".atualizado_{today}"));
    if marker.exists() {
        return Ok(());
    }
    let cutoff = util::add_days(today, -45);
    let (mut issuers, mut names) = (BTreeSet::new(), BTreeSet::new());
    for s in &store.series {
        if s.class == "CI" || s.last() < cutoff {
            continue;
        }
        let w = &s.bars[s.bars.len().saturating_sub(21)..];
        let mut v: Vec<f64> = w.iter().map(|b| b.vol_brl).collect();
        v.sort_by(|a, b| a.total_cmp(b));
        if v.is_empty() || v[v.len() / 2] < 200_000.0 {
            continue;
        }
        issuers.insert(issuer_of(&s.ticker));
        if let Some((n, _, _)) = s.names.iter().max_by_key(|x| x.2) {
            names.insert(n.clone());
        }
    }
    let mut jobs: Vec<Job> = issuers.into_iter().map(Job::Supp).collect();
    jobs.extend(names.into_iter().map(Job::Cash));
    run_jobs(jobs, dir, threads)?;
    std::fs::write(marker, "")?;
    Ok(())
}

fn run_jobs(jobs: Vec<Job>, dir: &Path, threads: usize) -> Result<()> {
    println!("Eventos B3: {} requisições ({threads} em paralelo)", jobs.len());
    if jobs.is_empty() {
        return Ok(());
    }
    let agent = http::agent()?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build()?;
    let done = AtomicUsize::new(0);
    let fails: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let total = jobs.len();
    pool.install(|| {
        jobs.par_iter().for_each(|job| {
            let res = match job {
                Job::Supp(i) => fetch_supp(&agent, dir, i),
                Job::Cash(n) => fetch_cash(&agent, dir, n),
            };
            if let Err(e) = res {
                fails.lock().unwrap().push(format!("{e:#}"));
            }
            let k = done.fetch_add(1, Ordering::Relaxed) + 1;
            if k % 250 == 0 || k == total {
                println!("  {k}/{total}");
            }
        })
    });
    let fails = fails.into_inner().unwrap();
    if !fails.is_empty() {
        eprintln!(
            "{} falhas (rode fetch-events de novo: só o que falta é baixado). Exemplos:",
            fails.len()
        );
        for f in fails.iter().take(5) {
            eprintln!("  {f}");
        }
    }
    Ok(())
}

fn fetch_supp(agent: &http::Agent, dir: &Path, issuer: &str) -> Result<()> {
    let q = serde_json::json!({"issuingCompany": issuer, "language": "pt-br"}).to_string();
    let url = format!("{BASE}/GetListedSupplementCompany/{}", b64(&q));
    let body = http::get_text(agent, &url)?;
    let v: serde_json::Value = if body.trim().is_empty() {
        serde_json::Value::Array(vec![])
    } else {
        serde_json::from_str(&body).with_context(|| format!("supp {issuer}: resposta não-JSON"))?
    };
    std::fs::write(supp_path(dir, issuer), serde_json::json!({"issuer": issuer, "data": v}).to_string())?;
    Ok(())
}

fn fetch_cash(agent: &http::Agent, dir: &Path, name: &str) -> Result<()> {
    let mut all: Vec<serde_json::Value> = Vec::new();
    let (mut page, mut total_pages) = (1u64, 1u64);
    while page <= total_pages {
        let q = serde_json::json!({
            "language": "pt-br", "pageNumber": page, "pageSize": 120, "tradingName": name
        })
        .to_string();
        let url = format!("{BASE}/GetListedCashDividends/{}", b64(&q));
        let body = http::get_text(agent, &url)?;
        if body.trim().is_empty() {
            break;
        }
        let v: serde_json::Value =
            serde_json::from_str(&body).with_context(|| format!("cash {name}: resposta não-JSON"))?;
        total_pages = v.pointer("/page/totalPages").and_then(|x| x.as_u64()).unwrap_or(1).min(60);
        if let Some(arr) = v.get("results").and_then(|r| r.as_array()) {
            all.extend(arr.iter().cloned());
        }
        page += 1;
    }
    let out = serde_json::json!({"tradingName": name, "results": all});
    std::fs::write(cash_path(dir, name), out.to_string())?;
    Ok(())
}

// ------------------------------------------------------------------ parsing

#[derive(Debug, Clone)]
pub enum Ev {
    /// multiplicador da quantidade de ações (2 = desdobramento 1:2; 0,1 = grupamento 10:1)
    Split(f64),
    /// provento em dinheiro como fração do preço na data-com (jcp = juros sobre capital próprio)
    Cash { y: f64, jcp: bool },
    /// recebe `ratio` ações do ativo `isin` por ação (cisão, bonificação em outra classe)
    SpinOff { isin: String, ratio: f64 },
    /// direito de subscrever `s` ações de `isin` por ação, ao preço `price`
    Rights { isin: String, s: f64, price: f64 },
}

#[derive(Debug, Clone)]
pub struct Event {
    /// última data com direito; o ajuste entra no pregão seguinte (data ex)
    pub datacom: u32,
    pub ev: Ev,
    pub label: String,
}

#[derive(Default, Debug)]
pub struct EvStats {
    pub supp_files: usize,
    pub cash_files: usize,
    pub stock_total: usize,
    pub stock_mapped: usize,
    pub stock_unmapped: usize,
    pub labels: BTreeMap<String, usize>,
    pub ignored: BTreeMap<String, usize>,
    pub subs_total: usize,
    pub subs_used: usize,
    pub cash_total: usize,
    pub cash_mapped: usize,
    pub cash_no_candidate: usize,
    pub cash_ambiguous: usize,
    pub cash_dup_across_files: usize,
    pub cash_close_checked: usize,
    pub cash_close_ok: usize,
    pub cash_bad_yield: usize,
    pub cash_kinds: BTreeMap<String, usize>,
}

impl EvStats {
    pub fn print(&self) {
        println!("Eventos B3 ({} arquivos de empresa, {} de proventos):", self.supp_files, self.cash_files);
        println!(
            "  ações (desdobr./grup./bonif./cisão): {} lidos, {} mapeados por ISIN, {} sem ISIN na base",
            self.stock_total, self.stock_mapped, self.stock_unmapped
        );
        println!("  tipos: {}", fmt_map(&self.labels));
        if !self.ignored.is_empty() {
            println!("  ignorados: {}", fmt_map(&self.ignored));
        }
        println!("  subscrições: {} lidas, {} usadas", self.subs_total, self.subs_used);
        println!(
            "  dinheiro: {} lidos, {} aplicados, {} sem ticker correspondente, {} ambíguos resolvidos pelo preço, {} duplicados entre arquivos, {} yield absurdo",
            self.cash_total,
            self.cash_mapped,
            self.cash_no_candidate,
            self.cash_ambiguous,
            self.cash_dup_across_files,
            self.cash_bad_yield
        );
        if self.cash_close_checked > 0 {
            println!(
                "  conferência de preço na data-com (B3 x COTAHIST): {}/{} batem ({:.1}%)",
                self.cash_close_ok,
                self.cash_close_checked,
                100.0 * self.cash_close_ok as f64 / self.cash_close_checked as f64
            );
        }
        println!("  tipos de provento: {}", fmt_map(&self.cash_kinds));
    }
}

fn fmt_map(m: &BTreeMap<String, usize>) -> String {
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1));
    v.iter().take(14).map(|(k, n)| format!("{k}={n}")).collect::<Vec<_>>().join(" ")
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SuppFile {
    data: serde_json::Value,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct SuppCompany {
    #[serde(rename = "stockDividends")]
    stock_dividends: Option<Vec<StockDiv>>,
    subscriptions: Option<Vec<Subscr>>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct StockDiv {
    #[serde(rename = "assetIssued")]
    asset_issued: Option<String>,
    factor: Option<String>,
    #[serde(rename = "isinCode")]
    isin_code: Option<String>,
    label: Option<String>,
    #[serde(rename = "lastDatePrior")]
    last_date_prior: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct Subscr {
    #[serde(rename = "assetIssued")]
    asset_issued: Option<String>,
    percentage: Option<String>,
    #[serde(rename = "priceUnit")]
    price_unit: Option<String>,
    #[serde(rename = "isinCode")]
    isin_code: Option<String>,
    label: Option<String>,
    #[serde(rename = "lastDatePrior")]
    last_date_prior: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct CashDiv {
    #[serde(rename = "typeStock")]
    type_stock: Option<String>,
    #[serde(rename = "valueCash")]
    value_cash: Option<String>,
    #[serde(rename = "corporateAction")]
    corporate_action: Option<String>,
    #[serde(rename = "lastDatePriorEx")]
    last_date_prior_ex: Option<String>,
    #[serde(rename = "closingPricePriorExDate")]
    closing_price: Option<String>,
    #[serde(rename = "quotedPerShares")]
    quoted_per_shares: Option<String>,
}

#[derive(Deserialize, Default)]
#[serde(default)]
struct CashFile {
    #[serde(rename = "tradingName")]
    trading_name: String,
    results: Vec<CashDiv>,
}

fn ev_key(e: &Ev) -> (u8, i64) {
    match e {
        Ev::Split(m) => (0, (m * 1e9).round() as i64),
        Ev::Cash { y, .. } => (1, (y * 1e9).round() as i64),
        Ev::SpinOff { ratio, .. } => (2, (ratio * 1e9).round() as i64),
        Ev::Rights { s, price, .. } => (3, ((s + price) * 1e6).round() as i64),
    }
}

/// Lê o cache de eventos e mapeia cada um para as séries da base (por ISIN ou nome+classe).
pub fn load(store: &BarStore, dir: &Path) -> Result<(Vec<Vec<Event>>, EvStats)> {
    let n = store.series.len();
    let mut evs: Vec<Vec<Event>> = vec![Vec::new(); n];
    let mut st = EvStats::default();
    let mut by_isin: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut by_name: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, s) in store.series.iter().enumerate() {
        for isin in &s.isins {
            by_isin.entry(isin.as_str()).or_default().push(i);
        }
        for (nm, _, _) in &s.names {
            let v = by_name.entry(nm.as_str()).or_default();
            if !v.contains(&i) {
                v.push(i);
            }
        }
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
    files.sort();
    let mut seen_cash: HashMap<(usize, u32, String, i64), usize> = HashMap::new();
    for (fid, path) in files.iter().enumerate() {
        let fname = path.file_name().and_then(|x| x.to_str()).unwrap_or("");
        if fname.starts_with("supp_") {
            st.supp_files += 1;
            parse_supp(path, &by_isin, &mut evs, &mut st)?;
        } else if fname.starts_with("cash_") {
            st.cash_files += 1;
            parse_cash(path, fid, store, &by_name, &mut evs, &mut seen_cash, &mut st)?;
        }
    }
    for v in evs.iter_mut() {
        v.sort_by(|a, b| {
            a.datacom.cmp(&b.datacom).then_with(|| a.label.cmp(&b.label)).then_with(|| ev_key(&a.ev).cmp(&ev_key(&b.ev)))
        });
        // eventos de ações/subscrições repetidos (mesmo ISIN em mais de um emissor) contam uma vez;
        // proventos em dinheiro idênticos dentro do mesmo arquivo são parcelas legítimas.
        v.dedup_by(|a, b| {
            !matches!(a.ev, Ev::Cash { .. }) && a.datacom == b.datacom && a.label == b.label && ev_key(&a.ev) == ev_key(&b.ev)
        });
    }
    Ok((evs, st))
}

fn parse_supp(path: &Path, by_isin: &HashMap<&str, Vec<usize>>, evs: &mut [Vec<Event>], st: &mut EvStats) -> Result<()> {
    let txt = std::fs::read_to_string(path)?;
    let f: SuppFile = serde_json::from_str(&txt).unwrap_or_default();
    let arr: Vec<SuppCompany> = serde_json::from_value(f.data).unwrap_or_default();
    for c in arr {
        for sd in c.stock_dividends.unwrap_or_default() {
            st.stock_total += 1;
            let label = sd.label.unwrap_or_default().trim().to_uppercase();
            *st.labels.entry(label.clone()).or_default() += 1;
            let Some(dc) = sd.last_date_prior.as_deref().and_then(util::parse_ddmmyyyy) else { continue };
            let Some(f) = sd.factor.as_deref().and_then(util::parse_br_num) else { continue };
            let isin = sd.isin_code.unwrap_or_default().trim().to_string();
            let asset = sd.asset_issued.unwrap_or_default().trim().to_string();
            let other_class = !asset.is_empty() && asset != isin;
            let ev = match label.as_str() {
                "DESDOBRAMENTO" | "BONIFICACAO" if f > 0.0 => {
                    if other_class {
                        Ev::SpinOff { isin: asset, ratio: f / 100.0 }
                    } else {
                        Ev::Split(1.0 + f / 100.0)
                    }
                }
                "GRUPAMENTO" if f > 0.0 && f < 1.0 => Ev::Split(f),
                l if (l.starts_with("CIS") || l == "REST CAP ACOES") && other_class && f > 0.0 => {
                    Ev::SpinOff { isin: asset, ratio: f / 100.0 }
                }
                _ => {
                    *st.ignored.entry(label).or_default() += 1;
                    continue;
                }
            };
            match by_isin.get(isin.as_str()) {
                Some(tks) => {
                    st.stock_mapped += 1;
                    for &t in tks {
                        evs[t].push(Event { datacom: dc, ev: ev.clone(), label: label.clone() });
                    }
                }
                None => st.stock_unmapped += 1,
            }
        }
        for sb in c.subscriptions.unwrap_or_default() {
            st.subs_total += 1;
            let label = sb.label.unwrap_or_default().trim().to_uppercase();
            if label != "SUBSCRICAO" {
                continue;
            }
            let (Some(dc), Some(pct), Some(price)) = (
                sb.last_date_prior.as_deref().and_then(util::parse_ddmmyyyy),
                sb.percentage.as_deref().and_then(util::parse_br_num),
                sb.price_unit.as_deref().and_then(util::parse_br_num),
            ) else {
                continue;
            };
            let s = pct / 100.0;
            if !(s > 0.0 && s <= 10.0 && price > 0.0 && price < 1e5) {
                continue;
            }
            let isin = sb.isin_code.unwrap_or_default().trim().to_string();
            let asset = sb.asset_issued.unwrap_or_default().trim().to_string();
            let target = if asset.is_empty() { isin.clone() } else { asset };
            if let Some(tks) = by_isin.get(isin.as_str()) {
                st.subs_used += 1;
                for &t in tks {
                    evs[t].push(Event {
                        datacom: dc,
                        ev: Ev::Rights { isin: target.clone(), s, price },
                        label: label.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

fn close_err(store: &BarStore, t: usize, dc: u32, b3close: f64, qps: f64) -> f64 {
    let s = &store.series[t];
    match s.idx_on_or_before(dc) {
        Some(i) if b3close > 0.0 => {
            let c = s.bars[i].close;
            ((c * qps) / b3close - 1.0).abs().min((c / b3close - 1.0).abs())
        }
        _ => f64::INFINITY,
    }
}

#[allow(clippy::too_many_arguments)]
fn parse_cash(
    path: &Path,
    file_id: usize,
    store: &BarStore,
    by_name: &HashMap<&str, Vec<usize>>,
    evs: &mut [Vec<Event>],
    seen: &mut HashMap<(usize, u32, String, i64), usize>,
    st: &mut EvStats,
) -> Result<()> {
    let txt = std::fs::read_to_string(path)?;
    let Ok(cf) = serde_json::from_str::<CashFile>(&txt) else { return Ok(()) };
    let Some(cands_all) = by_name.get(cf.trading_name.as_str()) else { return Ok(()) };
    for r in cf.results {
        st.cash_total += 1;
        let class = r.type_stock.unwrap_or_default().trim().to_uppercase();
        let kind = r.corporate_action.unwrap_or_default().trim().to_uppercase();
        let Some(dc) = r.last_date_prior_ex.as_deref().and_then(util::parse_ddmmyyyy) else { continue };
        let Some(val) = r.value_cash.as_deref().and_then(util::parse_br_num) else { continue };
        if val <= 0.0 {
            continue;
        }
        let b3close = r.closing_price.as_deref().and_then(util::parse_br_num).unwrap_or(0.0);
        let qps = r
            .quoted_per_shares
            .as_deref()
            .and_then(util::parse_br_num)
            .filter(|&q| q > 0.0)
            .unwrap_or(1.0);
        // candidatos: mesmo nome de pregão, mesma classe e vivos na data-com
        let cands: Vec<usize> = cands_all
            .iter()
            .copied()
            .filter(|&t| {
                let s = &store.series[t];
                s.class == class && s.first() <= dc && dc <= s.last()
            })
            .collect();
        let pick = match cands.len() {
            0 => {
                st.cash_no_candidate += 1;
                continue;
            }
            1 => cands[0],
            _ => {
                st.cash_ambiguous += 1;
                *cands
                    .iter()
                    .min_by(|&&a, &&b| {
                        close_err(store, a, dc, b3close, qps).total_cmp(&close_err(store, b, dc, b3close, qps))
                    })
                    .unwrap()
            }
        };
        let s = &store.series[pick];
        let ours = s.idx_on_or_before(dc).map(|i| s.bars[i].close);
        if b3close > 0.0 && ours.is_some() {
            st.cash_close_checked += 1;
            if close_err(store, pick, dc, b3close, qps) < 0.03 {
                st.cash_close_ok += 1;
            }
        }
        let y = if b3close > 0.0 {
            val / b3close
        } else if let Some(c) = ours {
            val / (c * qps)
        } else {
            continue;
        };
        if !(y > 0.0 && y < 0.6) {
            st.cash_bad_yield += 1;
            continue;
        }
        let key = (pick, dc, kind.clone(), (val * 1e8).round() as i64);
        match seen.get(&key) {
            Some(&f) if f != file_id => {
                st.cash_dup_across_files += 1;
                continue;
            }
            Some(_) => {}
            None => {
                seen.insert(key, file_id);
            }
        }
        *st.cash_kinds.entry(kind.clone()).or_default() += 1;
        st.cash_mapped += 1;
        let jcp = kind.contains("JRS") || kind.contains("JUROS");
        evs[pick].push(Event { datacom: dc, ev: Ev::Cash { y, jcp }, label: kind });
    }
    Ok(())
}
