//! The `b3` module: bridge to the quant-b3 lab. Kairós programs read the same B3 data the lab
//! uses (adjusted monthly returns, the NEFIN liquid universe, the CDI) and are judged by the same
//! realistic simulator, so a strategy written in Kairós and the same strategy written in Rust can
//! be compared number for number.
//!
//! M5 adds the research discipline:
//! - **protocols fence the data.** A program that declares `protocol` sees only its research
//!   period, and a registered protocol's holdout is hidden from every program until it is opened;
//! - **a holdout opens once** (`open_once`), even across runs: the holdout ledger
//!   (`results/holdout_ledger.csv`, see [`crate::ledger`]) records each opening;
//! - **every research evaluation is a trial** in the lab's registry
//!   (`results/registro_testes.csv`), and `Evaluation.dsr` is deflated by the number of trials.

use crate::ledger::{self, Kind, Ledger, Row};
use crate::tensor::TensorVal;
use crate::types::{ProtocolDef, AX_ASSET, AX_MONTH, AX_SESSION};
use quant_b3::backtest::{self, Costs, Plan, SimCfg, SimOut};
use quant_b3::common::{self, Data};
use quant_b3::registry::{self, Trial};
use quant_b3::search;
use quant_b3::stats;
use quant_b3::universe::{self, LiqRule};
use quant_b3::util::{self, Paths};
use std::cell::{Cell, OnceCell, RefCell, RefMut};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::Rc;

/// A runtime error from the data module (code, message, optional hint).
#[derive(Debug)]
pub struct HostErr {
    pub code: &'static str,
    pub msg: String,
    pub hint: Option<String>,
}

fn herr(code: &'static str, msg: impl Into<String>) -> HostErr {
    HostErr { code, msg: msg.into(), hint: None }
}

impl HostErr {
    fn hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }
}

/// Options of `b3.evaluate`.
pub struct EvalCfg {
    pub cost_bps: f64,
    pub lag: i64,
    pub capital: f64,
    pub real_costs: bool,
    pub tax: bool,
}

/// Result of `b3.evaluate` (becomes the prelude's `Evaluation` struct).
pub struct EvalOut {
    pub months: i64,
    pub cagr: f64,
    pub vol: f64,
    pub sharpe: f64,
    pub max_dd: f64,
    pub beta: f64,
    pub alpha: f64,
    pub t_alpha: f64,
    pub turnover: f64,
    pub cost_bps: f64,
    /// trials the DSR accounts for: the registry's size (research), 1 (holdout)
    pub trials: i64,
    /// deflated Sharpe ratio: probability that the true Sharpe beats the best of `trials` tries
    /// made without skill (on a holdout, simply that it is positive)
    pub dsr: f64,
}

/// The lab's registry of trials, inside the results folder.
pub const REGISTRY: &str = "registro_testes.csv";

/// Lazily loaded lab data plus the tensors built from it (shared, copied on write), the ledger
/// and the registry.
#[derive(Default)]
pub struct Host {
    data: OnceCell<Rc<Data>>,
    ret: OnceCell<Rc<TensorVal>>,
    elig: OnceCell<Rc<TensorVal>>,
    cdi: OnceCell<Rc<TensorVal>>,
    /// daily returns, volumes and prices (M8)
    daily: [OnceCell<Rc<TensorVal>>; 3],
    sess: OnceCell<Rc<TensorVal>>,
    dates: OnceCell<Rc<TensorVal>>,
    /// daily NEFIN factors (equally weighted, value weighted), read on first use
    nefin_d: [OnceCell<Option<std::collections::BTreeMap<u32, quant_b3::macrodata::Factors>>>; 2],
    /// where the data was loaded from (None: given in memory)
    data_dir: OnceCell<PathBuf>,
    /// the same tensors cut at the last month the program may see: (which, months)
    views: RefCell<HashMap<(u8, usize), Rc<TensorVal>>>,
    /// folder of the ledger and of the registry; `None`: kept in memory (tests, embedding)
    results: OnceCell<Option<PathBuf>>,
    ledger: RefCell<Option<Ledger>>,
    /// the registry of trials when there is no results folder
    trials: RefCell<Vec<Trial>>,
    /// name of the program in the registry of trials
    program: RefCell<String>,
    /// how many of the program's protocols are already registered in the ledger
    synced: Cell<usize>,
    /// the protocol whose holdout is open (while the function given to `open_once` runs)
    opening: Cell<Option<usize>>,
    /// a fixed date (YYYYMMDD) instead of today's, for tests
    today: Cell<u32>,
    fence_noted: Cell<bool>,
}

/// The hash that identifies a protocol in the ledger: changing any term makes another protocol.
pub fn protocol_sha(p: &ProtocolDef) -> String {
    let text = format!("kairos protocol v1\n{}\n{}\n{}\n{}\n{}", p.name, p.research_until, p.holdout_from, p.min_months, p.rule);
    search::sha256_hex(text.as_bytes())
}

/// `"key": "value"` in a small JSON file.
fn json_str(txt: &str, key: &str) -> Option<String> {
    let k = format!("\"{key}\"");
    let i = txt.find(&k)? + k.len();
    let rest = txt[i..].trim_start().strip_prefix(':')?.trim_start().strip_prefix('"')?;
    Some(rest[..rest.find('"')?].to_string())
}

/// The ledger starts with what happened before it existed: the lab's alpha search
/// (`quant-b3 busca`, protocol `results/protocolo_busca.md`) opened the holdout from 2019-01.
fn lab_seed(dir: &Path) -> Vec<Row> {
    let Ok(txt) = std::fs::read_to_string(dir.join("busca_resultado.json")) else { return Vec::new() };
    let date = json_str(&txt, "data").and_then(|s| util::parse_iso(&s)).unwrap_or_else(util::today);
    let sha = json_str(&txt, "protocolo_sha256").unwrap_or_default();
    let why = json_str(&txt, "motivo").unwrap_or_default();
    // the last month the search saw
    let until = std::fs::read_to_string(dir.join("busca_mensal.csv"))
        .ok()
        .and_then(|c| c.lines().skip(1).filter_map(|l| ledger::parse_ym(l.split(',').next()?)).filter(|m| *m > 0).max())
        .unwrap_or_else(|| util::ym(date));
    let note = format!("results/protocolo_busca.md: {why}");
    vec![Row::new(Kind::Opened, "busca-de-alpha (quant-b3)", &sha, search::SPLIT, until, date, &note)]
}

/// The most trading sessions in a month (the width of the daily tensors).
fn sessions_per_month(d: &Data) -> usize {
    d.cal.mrange.iter().map(|(a, b)| b - a).max().unwrap_or(1).max(1)
}

/// The first `n` months of a tensor whose first axis is `month`.
fn first_months(t: &TensorVal, n: usize) -> TensorVal {
    let row: usize = t.shape[1..].iter().product();
    let mut shape = t.shape.clone();
    shape[0] = n;
    match &t.data {
        crate::tensor::TData::F(v) => TensorVal::new_f(t.axes.clone(), shape, v[..n * row].to_vec()),
        crate::tensor::TData::B(v) => TensorVal::new_b(t.axes.clone(), shape, v[..n * row].to_vec()),
    }
}

impl Host {
    pub fn new() -> Self {
        Host::default()
    }

    /// A host over data that is already loaded (tests, embedding). The ledger and the registry
    /// stay in memory unless [`Host::with_results`] gives them a folder.
    pub fn with_data(d: Rc<Data>) -> Self {
        let h = Host::default();
        let _ = h.data.set(d);
        let _ = h.results.set(None);
        h
    }

    /// Keeps the ledger and the registry of trials in `dir` (normally the lab's `results`).
    pub fn with_results(mut self, dir: impl Into<PathBuf>) -> Self {
        self.results.take();
        let _ = self.results.set(Some(dir.into()));
        self
    }

    /// The program's name in the registry of trials.
    pub fn set_program(&self, name: &str) {
        *self.program.borrow_mut() = name.to_string();
    }

    /// Uses a fixed date (YYYYMMDD) in the ledger instead of today's (tests).
    pub fn set_today(&self, date: u32) {
        self.today.set(date);
    }

    fn today(&self) -> u32 {
        match self.today.get() {
            0 => util::today(),
            d => d,
        }
    }

    fn program(&self) -> String {
        let p = self.program.borrow();
        if p.is_empty() { "kairos".to_string() } else { p.clone() }
    }

    /// Folder with `processed/panel.bin`: `KAIROS_DATA`, else `./data`, `../data`, or the `data`
    /// folder of the quant-b3 checkout whose `target/<profile>/` holds this executable.
    pub fn find_data_dir() -> Option<PathBuf> {
        let mut cands: Vec<PathBuf> = Vec::new();
        if let Ok(p) = std::env::var("KAIROS_DATA") {
            cands.push(PathBuf::from(p));
        }
        if let Ok(cwd) = std::env::current_dir() {
            cands.push(cwd.join("data"));
            cands.push(cwd.join("..").join("data"));
        }
        if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.to_path_buf())) {
            cands.push(dir.join("..").join("..").join("data"));
        }
        cands.into_iter().find(|d| d.join("processed").join("panel.bin").is_file())
    }

    /// The lab's `results` folder, next to its `data` folder.
    fn results_of(data_dir: &Path) -> PathBuf {
        data_dir.parent().map(|p| p.join("results")).unwrap_or_else(|| data_dir.join("results"))
    }

    /// `KAIROS_RESULTS`, else the `results` folder next to the data.
    fn results_dir(&self) -> Option<&Path> {
        self.results
            .get_or_init(|| match std::env::var("KAIROS_RESULTS") {
                Ok(p) if !p.is_empty() => Some(PathBuf::from(p)),
                _ => Self::find_data_dir().map(|d| Self::results_of(&d)),
            })
            .as_deref()
    }

    pub fn data(&self) -> Result<&Rc<Data>, HostErr> {
        if let Some(d) = self.data.get() {
            return Ok(d);
        }
        let Some(dir) = Self::find_data_dir() else {
            return Err(herr("R0100", "B3 data not found").hint(
                "set KAIROS_DATA to the quant-b3 `data` folder (the one with processed/panel.bin), or run from the quant-b3 folder",
            ));
        };
        let d = Data::load(&Paths::new(&dir, &Self::results_of(&dir)))
            .map_err(|e| herr("R0100", format!("cannot load the B3 data from {}: {e:#}", dir.display())))?;
        let _ = self.data_dir.set(dir.clone());
        let _ = self.data.set(Rc::new(d));
        Ok(self.data.get().expect("data was just loaded"))
    }

    // ------------------------------------------------------------------ the ledger (M5)

    fn ledger(&self) -> Result<RefMut<'_, Ledger>, HostErr> {
        {
            let mut slot = self.ledger.borrow_mut();
            if slot.is_none() {
                let l = match self.results_dir() {
                    Some(dir) => Ledger::open(&dir.join(ledger::FILE), || lab_seed(dir)).map_err(|e| {
                        herr("R0500", e).hint("the ledger is a CSV file: fix or restore it (Kairós will not run evaluations without it)")
                    })?,
                    None => Ledger::in_memory(Vec::new()),
                };
                *slot = Some(l);
            }
        }
        Ok(RefMut::map(self.ledger.borrow_mut(), |s| s.as_mut().expect("the ledger was just loaded")))
    }

    /// `kairos ledger`: the ledger in words.
    pub fn ledger_report(&self) -> Result<String, HostErr> {
        if self.results_dir().is_none() {
            return Err(herr("R0100", "no results folder: the ledger lives next to the B3 data").hint("set KAIROS_DATA to the quant-b3 `data` folder"));
        }
        Ok(self.ledger()?.describe())
    }

    /// `kairos ledger withdraw <protocol>`: gives up a pending protocol.
    pub fn withdraw(&self, which: &str) -> Result<String, HostErr> {
        if self.results_dir().is_none() {
            return Err(herr("R0100", "no results folder: the ledger lives next to the B3 data").hint("set KAIROS_DATA to the quant-b3 `data` folder"));
        }
        let today = self.today();
        self.ledger()?.withdraw(which, today).map_err(|e| herr("R0502", e))
    }

    /// Registers the program's protocols in the ledger, the first time each is seen. A protocol
    /// whose holdout overlaps months already seen is refused.
    fn sync(&self, protos: &[ProtocolDef]) -> Result<(), HostErr> {
        if self.synced.get() >= protos.len() {
            return Ok(());
        }
        let today = self.today();
        let mut l = self.ledger()?;
        for p in &protos[self.synced.get()..] {
            l.register(&p.name, &protocol_sha(p), p.holdout_from as u32, today, &p.rule).map_err(|e| {
                herr("R0502", e).hint("a holdout has to be data nobody has looked at: the future is always clean")
            })?;
        }
        self.synced.set(protos.len());
        Ok(())
    }

    /// How many months of data the program may see: up to the end of its protocols' research
    /// period, and never into a reserved holdout. Records what was seen in the ledger.
    fn visible(&self, protos: &[ProtocolDef]) -> Result<usize, HostErr> {
        let d = self.data()?;
        let nm = d.cal.months.len();
        if self.opening.get().is_some() {
            return Ok(nm);
        }
        self.sync(protos)?;
        let mut limit: Option<(u32, String)> = None;
        for p in protos {
            let u = p.research_until as u32;
            if limit.as_ref().is_none_or(|(x, _)| u < *x) {
                limit = Some((u, format!("the research period of protocol \"{}\" ends there", p.name)));
            }
        }
        let today = self.today();
        let mut l = self.ledger()?;
        if let Some((f, r)) = l.fence() {
            if limit.as_ref().is_none_or(|(x, _)| f < *x) {
                limit = Some((
                    f,
                    format!(
                        "the months from {} are the holdout of protocol \"{}\" (registered on {}), which opens only through `open_once`",
                        ledger::fmt_ym(r.holdout_from),
                        r.protocol,
                        ledger::fmt_date(r.date)
                    ),
                ));
            }
        }
        let n = match &limit {
            Some((u, _)) => d.cal.idx_from(ledger::next_month(*u)),
            None => nm,
        };
        if n > 0 {
            l.saw(d.cal.months[n - 1], today, &format!("program {}", self.program())).map_err(|e| herr("R0500", e))?;
        }
        if n < nm && !self.fence_noted.replace(true) {
            if let Some((_, why)) = &limit {
                let last = if n > 0 { ledger::fmt_ym(d.cal.months[n - 1]) } else { "before the first month".into() };
                eprintln!("note: this program sees the data up to {last}: {why}");
            }
        }
        Ok(n)
    }

    fn view(&self, which: u8, n: usize, full: Rc<TensorVal>) -> Rc<TensorVal> {
        if full.shape[0] == n {
            return full;
        }
        self.views.borrow_mut().entry((which, n)).or_insert_with(|| Rc::new(first_months(&full, n))).clone()
    }

    fn cached(cell: &OnceCell<Rc<TensorVal>>, build: impl FnOnce() -> TensorVal) -> Rc<TensorVal> {
        cell.get_or_init(|| Rc::new(build())).clone()
    }

    // ------------------------------------------------------------------ data

    /// `Tensor[month, asset; f64]`: total return of each asset in each month (dividends, splits
    /// and other corporate events included); NaN where the asset did not trade in the month.
    pub fn monthly_returns(&self, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let full = Self::cached(&self.ret, || {
            let (nm, nt) = (d.cal.months.len(), d.panel.tks.len());
            let mut v = vec![f64::NAN; nm * nt];
            for t in 0..nt {
                for m in 0..nm {
                    if d.traded[t][m] {
                        v[m * nt + t] = d.mret[t][m];
                    }
                }
            }
            TensorVal::new_f(vec![AX_MONTH, AX_ASSET], vec![nm, nt], v)
        });
        Ok(self.view(0, n, full))
    }

    /// `Tensor[month, asset; bool]`: the NEFIN liquid universe of the month's year (most traded
    /// class of each company, liquid on more than 80% of the sessions of the previous year).
    pub fn eligible(&self, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let full = Self::cached(&self.elig, || {
            let (nm, nt) = (d.cal.months.len(), d.panel.tks.len());
            let mut v = vec![false; nm * nt];
            let mut by_year: HashMap<u32, Vec<usize>> = HashMap::new();
            for m in 0..nm {
                let y = d.cal.months[m] / 100;
                let set = by_year.entry(y).or_insert_with(|| universe::eligible(&d.panel, y, LiqRule::Dias, 500_000.0));
                for &t in set.iter() {
                    v[m * nt + t] = true;
                }
            }
            TensorVal::new_b(vec![AX_MONTH, AX_ASSET], vec![nm, nt], v)
        });
        Ok(self.view(1, n, full))
    }

    /// `Tensor[month; f64]`: CDI compounded over each month.
    pub fn cdi(&self, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let full = Self::cached(&self.cdi, || TensorVal::new_f(vec![AX_MONTH], vec![d.cal.months.len()], d.cdi_m.clone()));
        Ok(self.view(2, n, full))
    }

    /// `Tensor[month, session, asset; f64]` (M8): one row per month, one column per trading session
    /// of the month (B3 months have up to 23); NaN where the asset did not trade, and in the
    /// sessions a month does not have. `field`: 0 total return since the previous trade (dividends
    /// and corporate events included), 1 traded volume in R$, 2 unadjusted close price.
    pub fn daily(&self, field: u8, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let k = field.min(2) as usize;
        let full = Self::cached(&self.daily[k], || {
            let (nm, nt, s) = (d.cal.months.len(), d.panel.tks.len(), sessions_per_month(d));
            let mut v = vec![f64::NAN; nm * s * nt];
            for (t, tk) in d.panel.tks.iter().enumerate() {
                for day in &tk.days {
                    let di = day.di as usize;
                    let m = d.cal.date_month[di];
                    let j = di - d.cal.mrange[m].0;
                    v[(m * s + j) * nt + t] = match k {
                        0 => day.r,
                        1 => day.vol,
                        _ => day.px as f64,
                    };
                }
            }
            TensorVal::new_f(vec![AX_MONTH, AX_SESSION, AX_ASSET], vec![nm, s, nt], v)
        });
        Ok(self.view(10 + k as u8, n, full))
    }

    /// `Tensor[month, session; bool]`: the trading sessions each month has.
    pub fn sessions(&self, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let full = Self::cached(&self.sess, || {
            let (nm, s) = (d.cal.months.len(), sessions_per_month(d));
            let mut v = vec![false; nm * s];
            for (m, &(a, b)) in d.cal.mrange.iter().enumerate() {
                for j in 0..(b - a) {
                    v[m * s + j] = true;
                }
            }
            TensorVal::new_b(vec![AX_MONTH, AX_SESSION], vec![nm, s], v)
        });
        Ok(self.view(13, n, full))
    }

    /// `Tensor[month, session; f64]`: the date of each session, as YYYYMMDD (NaN where the month
    /// has no such session).
    pub fn session_dates(&self, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let full = Self::cached(&self.dates, || {
            let (nm, s) = (d.cal.months.len(), sessions_per_month(d));
            let mut v = vec![f64::NAN; nm * s];
            for (m, &(a, b)) in d.cal.mrange.iter().enumerate() {
                for j in 0..(b - a) {
                    v[m * s + j] = d.panel.dates[a + j] as f64;
                }
            }
            TensorVal::new_f(vec![AX_MONTH, AX_SESSION], vec![nm, s], v)
        });
        Ok(self.view(14, n, full))
    }

    /// `Tensor[month, session; f64]`: a daily NEFIN factor, NaN on the days NEFIN does not have.
    /// Read from the data folder's raw/nefin.
    pub fn nefin_daily(&self, factor: u8, vw: bool, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let table = self.nefin_d[vw as usize].get_or_init(|| {
            let dir = self.data_dir.get()?;
            let file = if vw { "nefin_factors.csv" } else { "nefin_factors_equally_weighted.csv" };
            quant_b3::macrodata::load_factors(&dir.join("raw").join("nefin").join(file), false).ok()
        });
        let Some(table) = table else {
            return Err(herr("R0100", "the daily NEFIN factors are not available").hint("they are read from the data folder's raw/nefin (`quant-b3 fetch-macro`)"));
        };
        let s = sessions_per_month(d);
        let mut v = vec![f64::NAN; n * s];
        for m in 0..n {
            let (a, b) = d.cal.mrange[m];
            for j in 0..(b - a) {
                if let Some(f) = table.get(&d.panel.dates[a + j]) {
                    v[m * s + j] = match factor {
                        0 => f.rm_rf,
                        1 => f.smb,
                        2 => f.hml,
                        3 => f.wml,
                        4 => f.iml,
                        _ => f.rf,
                    };
                }
            }
        }
        Ok(Rc::new(TensorVal::new_f(vec![AX_MONTH, AX_SESSION], vec![n, s], v)))
    }

    /// The company (`issuers`) or the class (`classes`, e.g. ON, PN, UNT, CI for ETFs) of each asset.
    pub fn asset_names(&self, issuers: bool) -> Result<Vec<String>, HostErr> {
        Ok(self.data()?.panel.tks.iter().map(|t| if issuers { t.issuer.clone() } else { t.class.clone() }).collect())
    }

    /// `Tensor[month; f64]`: a NEFIN factor (equally weighted, or value weighted); NaN where NEFIN
    /// has no value.
    pub fn nefin(&self, factor: u8, vw: bool, protos: &[ProtocolDef]) -> Result<Rc<TensorVal>, HostErr> {
        let n = self.visible(protos)?;
        let d = self.data()?;
        let src = if vw { &d.nef_vw } else { &d.nef };
        let v = src[..n]
            .iter()
            .map(|f| match f {
                Some(f) => match factor {
                    0 => f.rm_rf,
                    1 => f.smb,
                    2 => f.hml,
                    3 => f.wml,
                    4 => f.iml,
                    _ => f.rf,
                },
                None => f64::NAN,
            })
            .collect();
        Ok(Rc::new(TensorVal::new_f(vec![AX_MONTH], vec![n], v)))
    }

    pub fn months(&self, protos: &[ProtocolDef]) -> Result<Vec<i64>, HostErr> {
        let n = self.visible(protos)?;
        Ok(self.data()?.cal.months[..n].iter().map(|&m| m as i64).collect())
    }

    pub fn tickers(&self) -> Result<Vec<String>, HostErr> {
        Ok(self.data()?.panel.tks.iter().map(|t| t.ticker.clone()).collect())
    }

    /// Index of the first month on or after `ym` (YYYYMM); the number of months the program sees
    /// if it is later than that.
    pub fn month_index(&self, ym: i64, protos: &[ProtocolDef]) -> Result<i64, HostErr> {
        if !(100_001..=999_912).contains(&ym) || !(1..=12).contains(&(ym % 100)) {
            return Err(herr("R0101", format!("`{ym}` is not a month in YYYYMM form")).hint("e.g. `b3.month_index(200101)`"));
        }
        let n = self.visible(protos)?;
        Ok(self.data()?.cal.idx_from(ym as u32).min(n) as i64)
    }

    // ------------------------------------------------------------------ evaluation

    /// The plans of the months in `from..n`: the weights must cover the `n` months the program
    /// sees, and a month is traded only if its holding window ends within them (with `lag` > 0,
    /// the last one would need the first sessions of the next month).
    fn plans(&self, d: &Data, w: &TensorVal, cfg: &EvalCfg, n: usize, from: usize, cut_note: &str) -> Result<Vec<Plan>, HostErr> {
        let (nm, nt) = (d.cal.months.len(), d.panel.tks.len());
        let (Some(pm), Some(pa)) = (w.pos(AX_MONTH), w.pos(AX_ASSET)) else {
            return Err(herr("R0102", "the weights must have the axes `month` and `asset`"));
        };
        if w.shape[pm] != n || w.shape[pa] != nt {
            let mut e = herr("R0102", format!("the weights have {} months x {} assets, but the B3 data has {n} x {nt}{cut_note}", w.shape[pm], w.shape[pa]));
            e = e.hint("build them from the data, e.g. `zeros_like(b3.monthly_returns())`");
            return Err(e);
        }
        let Ok(lag) = usize::try_from(cfg.lag) else {
            return Err(herr("R0102", format!("`lag` must be between 0 and the number of sessions, got {}", cfg.lag)));
        };
        if !(cfg.capital > 0.0 && cfg.capital.is_finite()) {
            return Err(herr("R0102", format!("`capital` must be positive, got {:?}", cfg.capital)));
        }
        if !(cfg.cost_bps >= 0.0 && cfg.cost_bps.is_finite()) {
            return Err(herr("R0102", format!("`cost_bps` must be >= 0, got {:?}", cfg.cost_bps)));
        }
        let end = if n < nm { d.month_start(n) } else { d.panel.dates.len() };
        let st = w.strides();
        let v = w.f();
        let mut plans = Vec::new();
        for m in 0..n {
            let mut ws = Vec::new();
            let mut tot = 0.0;
            for t in 0..nt {
                let x = v[m * st[pm] + t * st[pa]];
                if x == 0.0 {
                    continue;
                }
                if !x.is_finite() || x < 0.0 {
                    return Err(herr("R0102", format!("invalid weight {x:?} for {} in {}", d.panel.tks[t].ticker, util::fmt_ym(d.cal.months[m])))
                        .hint("weights are long-only fractions of wealth: finite and >= 0"));
                }
                ws.push((t, x));
                tot += x;
            }
            if tot > 1.0 + 1e-9 {
                return Err(herr("R0102", format!("the weights of {} add up to {tot:.6}, more than 1", util::fmt_ym(d.cal.months[m])))
                    .hint("Kairós evaluates long-only portfolios without leverage: what is not invested earns CDI"));
            }
            if !ws.is_empty() && m >= from && (n == nm || d.window(m, lag).1 <= end) {
                plans.push(Plan { mi: m, w: ws });
            }
        }
        if plans.is_empty() {
            return Err(herr("R0102", "all weights are zero: there is nothing to evaluate"));
        }
        Ok(plans)
    }

    fn simulate(&self, d: &Data, plans: &[Plan], cfg: &EvalCfg) -> Result<(SimOut, EvalOut, Vec<f64>), HostErr> {
        let costs = if cfg.real_costs { search::REAL } else { Costs::Fixed { bps: cfg.cost_bps } };
        let sim = SimCfg { capital0: cfg.capital, lag: cfg.lag as usize, costs, tax: cfg.tax, cap_adv: None };
        let out = backtest::simulate(d, plans, &sim);
        if out.ret.is_empty() {
            return Err(herr("R0102", "the simulation produced no months").hint("are the weights only in the first month or after the end of the data?"));
        }
        let cdi: Vec<f64> = out.mi.iter().map(|&m| d.cdi_m[m]).collect();
        let pf = common::perf(&out.ret, &cdi);
        let ev = search::evaluate(d, &out.mi, &out.ret, None, 0).map(|(e, _)| e);
        let pick = |f: fn(&search::Ev) -> f64| ev.as_ref().map_or(f64::NAN, f);
        let excess: Vec<f64> = out.ret.iter().zip(&cdi).map(|(a, b)| a - b).collect();
        let e = EvalOut {
            months: out.ret.len() as i64,
            cagr: pf.cagr,
            vol: pf.vol,
            sharpe: pf.sharpe,
            max_dd: pf.mdd,
            beta: pick(|e| e.beta),
            alpha: pick(|e| e.alpha),
            t_alpha: pick(|e| e.t_alpha),
            turnover: stats::mean(&out.turnover) * 12.0,
            cost_bps: out.avg_cost_bps(),
            trials: 0,
            dsr: f64::NAN,
        };
        Ok((out, e, excess))
    }

    /// Simulates a long-only portfolio with the lab's realistic simulator: one weight per month and
    /// asset (fraction of wealth, the rest earns CDI). Rows that are all zero are not traded. The
    /// evaluation is a trial in the registry, and its DSR counts every trial there.
    pub fn evaluate(&self, w: &TensorVal, cfg: &EvalCfg, protos: &[ProtocolDef]) -> Result<EvalOut, HostErr> {
        if let Some(k) = self.opening.get() {
            return Err(herr("R0504", format!("`b3.evaluate` cannot run while the holdout of protocol \"{}\" is open", protos[k].name))
                .hint("the function given to `open_once` only computes the weights; `open_once` evaluates them"));
        }
        let n = self.visible(protos)?;
        let d = self.data()?;
        let nm = d.cal.months.len();
        let cut = if n < nm { " visible to this program (see the note above)" } else { "" };
        let plans = self.plans(d, w, cfg, n, 0, cut)?;
        let (out, mut e, excess) = self.simulate(d, &plans, cfg)?;
        let (Some(&first), Some(&last)) = (out.mi.first(), out.mi.last()) else { unreachable!("the simulation produced months") };
        let periodo = format!("{}..{}", util::fmt_ym(d.cal.months[first]), util::fmt_ym(d.cal.months[last]));
        let costs = if cfg.real_costs { "custo real".to_string() } else { format!("custo {} bps", cfg.cost_bps) };
        let ir = if cfg.tax { "com IR" } else { "sem IR" };
        let params = format!("pesos {}; {costs}; lag {}; capital {}; {ir}", &weights_sha(w, n)[..12], cfg.lag, cfg.capital);
        let t = Trial::new("kairos", &self.program(), &params, &periodo, &excess);
        let (trials, dsr) = self.record(t)?;
        e.trials = trials;
        e.dsr = dsr;
        Ok(e)
    }

    /// Records a research trial (once per distinct return series: the same strategy evaluated
    /// again, here or by the lab in Rust, is not a new trial) and deflates its Sharpe ratio by
    /// the number of trials in the registry.
    fn record(&self, t: Trial) -> Result<(i64, f64), HostErr> {
        let same = |x: &Trial| {
            x.n == t.n && (x.sr_m - t.sr_m).abs() <= 1e-6 && (x.skew - t.skew).abs() <= 1e-4 && (x.kurt - t.kurt).abs() <= 1e-4
        };
        let all = match self.results_dir() {
            Some(dir) => {
                let path = dir.join(REGISTRY);
                let all = registry::load(&path);
                if all.iter().any(same) {
                    all
                } else {
                    std::fs::create_dir_all(dir).map_err(|e| herr("R0500", format!("cannot create {}: {e}", dir.display())))?;
                    registry::upsert(&path, &t).map_err(|e| herr("R0500", format!("cannot write the registry of trials {}: {e:#}", path.display())))?
                }
            }
            None => {
                let mut mem = self.trials.borrow_mut();
                if !mem.iter().any(same) {
                    let key = |x: &Trial| (x.estudo.clone(), x.nome.clone(), x.params.clone(), x.periodo.clone());
                    mem.retain(|x| key(x) != key(&t));
                    mem.push(t.clone());
                }
                mem.clone()
            }
        };
        let df = registry::deflate(&t, &all);
        Ok((df.n_trials as i64, df.dsr))
    }

    // ------------------------------------------------------------------ holdouts

    /// Why the holdout of `p` cannot be opened, if it cannot; else its hash.
    fn openable(&self, l: &Ledger, p: &ProtocolDef) -> Result<String, HostErr> {
        let sha = protocol_sha(p);
        let once = "a holdout opens once: to test a new idea, declare a new protocol whose holdout starts after the months already seen";
        if let Some(r) = l.find(Kind::Opened, &sha) {
            return Err(herr("R0502", format!("the holdout of protocol \"{}\" was already opened on {} ({})", p.name, ledger::fmt_date(r.date), r.note))
                .hint(once));
        }
        if let Some(r) = l.find(Kind::Withdrawn, &sha) {
            return Err(herr("R0502", format!("protocol \"{}\" was withdrawn on {}", p.name, ledger::fmt_date(r.date))).hint(once));
        }
        let h = p.holdout_from as u32;
        if h <= l.watermark() {
            let who = l.seen_by(h).map(Row::who).unwrap_or_default();
            return Err(herr("R0502", format!("the holdout of protocol \"{}\" (from {}) is no longer clean: those months were seen by {who}", p.name, ledger::fmt_ym(h)))
                .hint(once));
        }
        Ok(sha)
    }

    /// `b3.holdout("name")`: fails right away if that holdout can no longer be opened, so a long
    /// research run does not end in R0502.
    pub fn holdout(&self, protos: &[ProtocolDef], k: usize) -> Result<(), HostErr> {
        self.sync(protos)?;
        let l = self.ledger()?;
        self.openable(&l, &protos[k]).map(|_| ())
    }

    /// Starts `open_once` for protocol `k`: checks the ledger and the data, records the opening
    /// before any month of the holdout is seen, and lifts the fence until [`Host::finish_opening`].
    pub fn begin_opening(&self, protos: &[ProtocolDef], k: usize) -> Result<(), HostErr> {
        if let Some(j) = self.opening.get() {
            return Err(herr("R0504", format!("the holdout of protocol \"{}\" is already open: holdouts are not opened inside one another", protos[j].name)));
        }
        self.sync(protos)?;
        let p = &protos[k];
        let d = self.data()?;
        let today = self.today();
        let mut l = self.ledger()?;
        let sha = self.openable(&l, p)?;
        let h = p.holdout_from as u32;
        let nm = d.cal.months.len();
        let have = nm - d.cal.idx_from(h).min(nm);
        let last = d.cal.months.last().copied().unwrap_or(0);
        if have == 0 || (have as i64) < p.min_months {
            let msg = if have == 0 {
                format!("the holdout of protocol \"{}\" starts in {}, but the data ends in {}: there is nothing to open yet", p.name, ledger::fmt_ym(h), ledger::fmt_ym(last))
            } else {
                format!(
                    "the holdout of protocol \"{}\" has {have} month(s) of data ({}..{}), and the protocol asks for at least {} before opening",
                    p.name,
                    ledger::fmt_ym(h),
                    ledger::fmt_ym(last),
                    p.min_months
                )
            };
            return Err(herr("R0503", msg).hint("nothing was opened or recorded: run again when the data has more months"));
        }
        l.push(Row::new(Kind::Opened, &p.name, &sha, h, last, today, "opening: the weights function is running")).map_err(|e| herr("R0500", e))?;
        self.opening.set(Some(k));
        Ok(())
    }

    /// Ends `open_once`: evaluates the weights on the holdout's months (a fresh portfolio from
    /// its first month) and writes the result in the ledger. `Err` is the error of the weights
    /// function: the holdout was open while it ran, so it is spent all the same.
    pub fn finish_opening(&self, protos: &[ProtocolDef], weights: Result<&TensorVal, String>, cfg: &EvalCfg) -> Result<EvalOut, HostErr> {
        let Some(k) = self.opening.take() else {
            return Err(herr("R0199", "internal: no holdout is open"));
        };
        let p = &protos[k];
        let res = match weights {
            Ok(w) => self.eval_holdout(w, cfg, p),
            Err(msg) => Err(herr("R0505", msg)),
        };
        let note = match &res {
            Ok(e) => format!(
                "{} months; CAGR {:.2}%; Sharpe {:.2}; alpha {:.2}% a.a. (t {:.2}); PSR {:.0}%",
                e.months,
                100.0 * e.cagr,
                e.sharpe,
                100.0 * e.alpha,
                e.t_alpha,
                100.0 * e.dsr
            ),
            Err(e) => format!("failed: {} {}", e.code, e.msg),
        };
        self.ledger()?.set_opened_note(&protocol_sha(p), &note).map_err(|e| herr("R0500", e))?;
        res
    }

    fn eval_holdout(&self, w: &TensorVal, cfg: &EvalCfg, p: &ProtocolDef) -> Result<EvalOut, HostErr> {
        let d = self.data()?;
        let nm = d.cal.months.len();
        let from = d.cal.idx_from(p.holdout_from as u32);
        let plans = self.plans(d, w, cfg, nm, from, " (inside `open_once` the program sees all the data)")?;
        let (_, mut e, excess) = self.simulate(d, &plans, cfg)?;
        // a single trial, made once: its DSR is the probability that the Sharpe is positive (PSR)
        let t = Trial::new("kairos-holdout", &p.name, "", "", &excess);
        e.trials = 1;
        e.dsr = registry::deflate(&t, std::slice::from_ref(&t)).dsr;
        Ok(e)
    }
}

/// Identifies the weights in the registry of trials: their months up to `n`, bit for bit.
fn weights_sha(w: &TensorVal, n: usize) -> String {
    let (Some(pm), Some(pa)) = (w.pos(AX_MONTH), w.pos(AX_ASSET)) else { return String::from("000000000000") };
    let st = w.strides();
    let v = w.f();
    let mut bytes = Vec::with_capacity(16 + n * w.shape[pa] * 8);
    bytes.extend_from_slice(&(n as u64).to_le_bytes());
    bytes.extend_from_slice(&(w.shape[pa] as u64).to_le_bytes());
    for m in 0..n {
        for t in 0..w.shape[pa] {
            bytes.extend_from_slice(&v[m * st[pm] + t * st[pa]].to_bits().to_le_bytes());
        }
    }
    search::sha256_hex(&bytes)
}
