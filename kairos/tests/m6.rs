//! M6: the research program `research/tax_portfolio.krs`, exactly as registered, on a synthetic
//! market with a momentum effect (persistent drifts). It must type-check (the walk-forward is
//! proven causal by the temporal checker), train, be judged by the realistic simulator against
//! equal weights and momentum, count three trials, and stop at the holdout (R0503: the future
//! holdout has no data yet). No real data is touched.

use kairos::bridge::{Host, REGISTRY};
use kairos::ledger::{self, Kind, Ledger};
use kairos::session::Session;
use quant_b3::common::Data;
use quant_b3::macrodata::Factors;
use quant_b3::panel::{Day, Panel, Tk};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::rc::Rc;

struct Buf(Rc<RefCell<Vec<u8>>>);

impl Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        // sum of 12 uniforms: close enough to N(0, 1) for a test market
        (0..12).map(|_| self.unit()).sum::<f64>() - 6.0
    }
}

/// 2001-01..`last_year`-12, 20 sessions a month, `n` assets whose monthly drift is persistent
/// (an AR(1) with 0.9), so last year's winners tend to keep winning.
fn market(last_year: u32, n: usize) -> Rc<Data> {
    let mut dates = Vec::new();
    for y in 2001..=last_year {
        for m in 1..=12u32 {
            for d in 1..=20u32 {
                dates.push(y * 10000 + m * 100 + d);
            }
        }
    }
    let nm = dates.len() / 20;
    let mut rng = Rng(0x5DEE_CE66_D1CE_4E5B);
    let mut tks = Vec::new();
    for i in 0..n {
        let mut mu = 0.0f64;
        let mut days = Vec::with_capacity(dates.len());
        for m in 0..nm {
            mu = 0.9 * mu + 0.01 * rng.normal();
            for d in 0..20 {
                let di = m * 20 + d;
                days.push(Day { di: di as u32, r: mu / 20.0 + 0.015 * rng.normal(), vol: 3_000_000.0, div: 0.0, jcp: 0.0, spr: 0.002, px: 20.0 });
            }
        }
        tks.push(Tk { ticker: format!("SYN{i:03}3"), issuer: format!("COMPANY {i}"), class: "ON".into(), first: dates[0], last: *dates.last().unwrap(), days });
    }
    let cdi: BTreeMap<u32, f64> = dates.iter().map(|&d| (d, 0.0004)).collect();
    let mut ff = BTreeMap::new();
    let mut ipca = BTreeMap::new();
    for y in 2001..=last_year {
        for m in 1..=12u32 {
            ff.insert(y * 100 + m, Factors { rm_rf: 0.05 * (rng.unit() - 0.5), smb: 0.0, hml: 0.0, wml: 0.0, iml: 0.0, rf: 0.008 });
            ipca.insert(y * 100 + m, 0.004);
        }
    }
    Rc::new(Data::build(Panel { dates, tks }, &cdi, &ff, &ff.clone(), ipca))
}

fn run_program(d: &Rc<Data>, dir: &Path) -> (String, Result<(), String>) {
    let src = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("research").join("tax_portfolio.krs")).unwrap();
    let host = Host::with_data(d.clone()).with_results(dir);
    host.set_today(20261102);
    host.set_program("tax_portfolio");
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), host);
    let res = s.run("tax_portfolio.krs", &src).map(|_| ()).map_err(|e| s.render(&e));
    let out = String::from_utf8(buf.borrow().clone()).unwrap();
    (out, res)
}

/// Runs the program on `market(last_year, assets)` and checks the whole path.
fn check(name: &str, last_year: u32, assets: usize, fits: usize, months: i64) {
    let d = market(last_year, assets);
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kairos-m6-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let t0 = std::time::Instant::now();
    let (out, res) = run_program(&d, &dir);
    print!("{out}");
    println!("({assets} assets, 2001..{last_year}: {:.1} s)", t0.elapsed().as_secs_f64());
    if let Err(e) = &res {
        eprintln!("{e}");
    }
    // one fit per January of decisions (from 2006)
    assert_eq!(out.matches("refit ").count(), fits, "{out}");
    for p in ["model", "equal-weight", "momentum"] {
        let line = format!("{p} months {months} ");
        assert!(out.contains(&line), "missing `{line}` in:\n{out}");
    }
    assert!(out.contains("research verdict:"), "{out}");
    // the holdout is the future: not enough of it to open, and nothing recorded
    let e = res.unwrap_err();
    assert!(e.contains("error[R0503]") && e.contains("2026-11"), "{e}");
    let l = Ledger::open(&dir.join(ledger::FILE), Vec::new).unwrap();
    assert!(l.rows.iter().any(|r| r.kind == Kind::Protocol && r.protocol == "tax-portfolio-v1"));
    assert!(l.rows.iter().all(|r| r.kind != Kind::Opened));
    // three research trials in the registry
    let reg = std::fs::read_to_string(dir.join(REGISTRY)).unwrap();
    assert_eq!(reg.lines().skip(1).filter(|l| l.starts_with("kairos,tax_portfolio,")).count(), 3, "{reg}");
}

fn big_stack(f: fn()) {
    match std::thread::Builder::new().stack_size(256 << 20).spawn(f) {
        Ok(h) => {
            if let Err(p) = h.join() {
                std::panic::resume_unwind(p);
            }
        }
        Err(_) => f(),
    }
}

/// Small enough for every CI platform (emulated ones included): one January of decisions.
#[test]
fn the_registered_program_runs_on_a_synthetic_market() {
    big_stack(|| check("ci", 2006, 32, 1, 12))
}

/// Closer to the real size (run by hand, in release): `cargo test --release -p kairos --test m6
/// -- --ignored --nocapture`.
#[test]
#[ignore]
fn timing_closer_to_the_real_size() {
    // with lag 1 the last research month (2026-10) is not traded: it needs 2026-11
    big_stack(|| check("big", 2026, 300, 21, 249))
}
