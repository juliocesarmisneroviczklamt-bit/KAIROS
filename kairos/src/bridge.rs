//! The `b3` module: bridge to the quant-b3 lab. Kairós programs read the same B3 data the lab
//! uses (adjusted monthly returns, the NEFIN liquid universe, the CDI) and are judged by the same
//! realistic simulator, so a strategy written in Kairós and the same strategy written in Rust can
//! be compared number for number.

use crate::tensor::TensorVal;
use crate::types::{AX_ASSET, AX_MONTH};
use quant_b3::backtest::{self, Costs, Plan, SimCfg};
use quant_b3::common::{self, Data};
use quant_b3::search;
use quant_b3::stats;
use quant_b3::universe::{self, LiqRule};
use quant_b3::util::{self, Paths};
use std::cell::OnceCell;
use std::collections::HashMap;
use std::path::PathBuf;
use std::rc::Rc;

/// A runtime error from the data module (code, message, optional hint).
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
}

/// Lazily loaded lab data plus the tensors built from it (shared, copied on write).
#[derive(Default)]
pub struct Host {
    data: OnceCell<Rc<Data>>,
    ret: OnceCell<Rc<TensorVal>>,
    elig: OnceCell<Rc<TensorVal>>,
    cdi: OnceCell<Rc<TensorVal>>,
}

impl Host {
    pub fn new() -> Self {
        Host::default()
    }

    /// A host over data that is already loaded (tests, embedding).
    pub fn with_data(d: Rc<Data>) -> Self {
        let h = Host::default();
        let _ = h.data.set(d);
        h
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

    pub fn data(&self) -> Result<&Rc<Data>, HostErr> {
        if let Some(d) = self.data.get() {
            return Ok(d);
        }
        let Some(dir) = Self::find_data_dir() else {
            return Err(herr("R0100", "B3 data not found").hint(
                "set KAIROS_DATA to the quant-b3 `data` folder (the one with processed/panel.bin), or run from the quant-b3 folder",
            ));
        };
        let results = dir.parent().map(|p| p.join("results")).unwrap_or_else(|| dir.join("results"));
        let d = Data::load(&Paths::new(&dir, &results))
            .map_err(|e| herr("R0100", format!("cannot load the B3 data from {}: {e:#}", dir.display())))?;
        let _ = self.data.set(Rc::new(d));
        Ok(self.data.get().expect("data was just loaded"))
    }

    fn cached(cell: &OnceCell<Rc<TensorVal>>, build: impl FnOnce() -> TensorVal) -> Rc<TensorVal> {
        cell.get_or_init(|| Rc::new(build())).clone()
    }

    /// `Tensor[month, asset; f64]`: total return of each asset in each month (dividends, splits
    /// and other corporate events included); NaN where the asset did not trade in the month.
    pub fn monthly_returns(&self) -> Result<Rc<TensorVal>, HostErr> {
        let d = self.data()?;
        Ok(Self::cached(&self.ret, || {
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
        }))
    }

    /// `Tensor[month, asset; bool]`: the NEFIN liquid universe of the month's year (most traded
    /// class of each company, liquid on more than 80% of the sessions of the previous year).
    pub fn eligible(&self) -> Result<Rc<TensorVal>, HostErr> {
        let d = self.data()?;
        Ok(Self::cached(&self.elig, || {
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
        }))
    }

    /// `Tensor[month; f64]`: CDI compounded over each month.
    pub fn cdi(&self) -> Result<Rc<TensorVal>, HostErr> {
        let d = self.data()?;
        Ok(Self::cached(&self.cdi, || TensorVal::new_f(vec![AX_MONTH], vec![d.cal.months.len()], d.cdi_m.clone())))
    }

    pub fn months(&self) -> Result<Vec<i64>, HostErr> {
        Ok(self.data()?.cal.months.iter().map(|&m| m as i64).collect())
    }

    pub fn tickers(&self) -> Result<Vec<String>, HostErr> {
        Ok(self.data()?.panel.tks.iter().map(|t| t.ticker.clone()).collect())
    }

    /// Index of the first month on or after `ym` (YYYYMM); the number of months if it is later
    /// than the data.
    pub fn month_index(&self, ym: i64) -> Result<i64, HostErr> {
        if !(100_001..=999_912).contains(&ym) || !(1..=12).contains(&(ym % 100)) {
            return Err(herr("R0101", format!("`{ym}` is not a month in YYYYMM form")).hint("e.g. `b3.month_index(200101)`"));
        }
        Ok(self.data()?.cal.idx_from(ym as u32) as i64)
    }

    /// Simulates a long-only portfolio with the lab's realistic simulator: one weight per month and
    /// asset (fraction of wealth, the rest earns CDI). Rows that are all zero are not traded.
    pub fn evaluate(&self, w: &TensorVal, cfg: &EvalCfg) -> Result<EvalOut, HostErr> {
        let d = self.data()?;
        let (nm, nt) = (d.cal.months.len(), d.panel.tks.len());
        let (Some(pm), Some(pa)) = (w.pos(AX_MONTH), w.pos(AX_ASSET)) else {
            return Err(herr("R0102", "the weights must have the axes `month` and `asset`"));
        };
        if w.shape[pm] != nm || w.shape[pa] != nt {
            return Err(herr("R0102", format!("the weights have {} months x {} assets, but the B3 data has {nm} x {nt}", w.shape[pm], w.shape[pa]))
                .hint("build them from the data, e.g. `zeros_like(b3.monthly_returns())`"));
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
        let st = w.strides();
        let v = w.f();
        let mut plans = Vec::new();
        for m in 0..nm {
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
            if !ws.is_empty() {
                plans.push(Plan { mi: m, w: ws });
            }
        }
        if plans.is_empty() {
            return Err(herr("R0102", "all weights are zero: there is nothing to evaluate"));
        }
        let costs = if cfg.real_costs { search::REAL } else { Costs::Fixed { bps: cfg.cost_bps } };
        let sim = SimCfg { capital0: cfg.capital, lag, costs, tax: cfg.tax, cap_adv: None };
        let out = backtest::simulate(d, &plans, &sim);
        if out.ret.is_empty() {
            return Err(herr("R0102", "the simulation produced no months").hint("are the weights only in the first month or after the end of the data?"));
        }
        let cdi: Vec<f64> = out.mi.iter().map(|&m| d.cdi_m[m]).collect();
        let pf = common::perf(&out.ret, &cdi);
        let ev = search::evaluate(d, &out.mi, &out.ret, None, 0).map(|(e, _)| e);
        let pick = |f: fn(&search::Ev) -> f64| ev.as_ref().map_or(f64::NAN, f);
        Ok(EvalOut {
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
        })
    }
}
