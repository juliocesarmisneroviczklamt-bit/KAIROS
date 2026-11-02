//! M2 acceptance test: momentum written in Kairós (`examples/momentum.krs`), running on the real
//! B3 data, must reproduce bit for bit the same portfolio computed in Rust by the quant-b3 lab
//! (the long-only winners of `src/risk.rs`: 25 bps, lag 0, R$ 1 million, from 2001-01).
//! Skipped when the data folder is not there.

use kairos::bridge::Host;
use kairos::session::Session;
use quant_b3::backtest::{self, Costs, SimCfg};
use quant_b3::common::{self, Data};
use quant_b3::util::Paths;
use quant_b3::{risk, search, stats};
use std::cell::RefCell;
use std::collections::HashMap;
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

#[test]
fn momentum_in_kairos_matches_the_rust_lab() {
    match std::thread::Builder::new().stack_size(256 << 20).spawn(run) {
        Ok(h) => {
            if let Err(panic) = h.join() {
                std::panic::resume_unwind(panic);
            }
        }
        // no threads (WebAssembly/WASI)
        Err(_) => run(),
    }
}

fn run() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let data_dir = root.join("data");
    if !data_dir.join("processed").join("panel.bin").is_file() {
        eprintln!("skipped: no B3 data in {}", data_dir.display());
        return;
    }
    let d = Rc::new(Data::load(&Paths::new(&data_dir, &root.join("results"))).expect("B3 data"));

    // reference: quant-b3 in Rust (src/risk.rs, long-only winners)
    let start = d.cal.idx_from(200101);
    let forms: Vec<_> = risk::liquid_forms(&d, 0.0).into_iter().filter(|f| f.mi >= start).collect();
    let cfg = SimCfg { capital0: 1_000_000.0, lag: 0, costs: Costs::Fixed { bps: 25.0 }, tax: false, cap_adv: None };
    let out = backtest::simulate(&d, &risk::winner_plans(&d, &forms, None), &cfg);
    let cdi: Vec<f64> = out.mi.iter().map(|&m| d.cdi_m[m]).collect();
    let pf = common::perf(&out.ret, &cdi);
    let capm = search::evaluate(&d, &out.mi, &out.ret, None, 0).expect("enough months").0;

    // the same strategy written in Kairós
    let src = std::fs::read_to_string(root.join("kairos").join("examples").join("momentum.krs")).expect("example");
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), Host::with_data(d.clone()));
    if let Err(e) = s.run("momentum.krs", &src) {
        panic!("{}", s.render(&e));
    }
    let text = String::from_utf8(buf.borrow().clone()).expect("utf-8");
    let got: HashMap<&str, f64> = text
        .lines()
        .filter_map(|l| {
            let mut it = l.split_whitespace();
            Some((it.next()?, it.next()?.parse().ok()?))
        })
        .collect();
    let num = |k: &str| *got.get(k).unwrap_or_else(|| panic!("`{k}` missing from the output:\n{text}"));

    assert_eq!(num("months") as usize, out.ret.len(), "number of simulated months");
    let want = [
        ("cagr", pf.cagr),
        ("vol", pf.vol),
        ("sharpe", pf.sharpe),
        ("max_dd", pf.mdd),
        ("beta", capm.beta),
        ("alpha", capm.alpha),
        ("t_alpha", capm.t_alpha),
        ("turnover", stats::mean(&out.turnover) * 12.0),
        ("cost_bps", out.avg_cost_bps()),
    ];
    for (k, w) in want {
        let g = num(k);
        assert_eq!(g.to_bits(), w.to_bits(), "{k}: Kairós {g:?} vs Rust {w:?}");
        println!("{k:<9} Kairós {g:<22?} Rust {w:<22?} identical");
    }
}

/// The lab's data written in Kairós (lab/data.krs: the panel, its calendar, the monthly returns,
/// the NEFIN liquid universe, the CDI and NEFIN's factors) gives what the runtime's `b3.*` (Rust)
/// gives, value for value.
/// Skipped when the data folder is not there.
#[test]
fn the_lab_data_in_kairos_is_the_rust_one() {
    match std::thread::Builder::new().stack_size(256 << 20).spawn(run_data_check) {
        Ok(h) => {
            if let Err(panic) = h.join() {
                std::panic::resume_unwind(panic);
            }
        }
        Err(_) => run_data_check(),
    }
}

fn run_data_check() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let data_dir = root.join("data");
    if !data_dir.join("processed").join("panel.bin").is_file() {
        eprintln!("skipped: no B3 data in {}", data_dir.display());
        return;
    }
    let d = Rc::new(Data::load(&Paths::new(&data_dir, &root.join("results"))).expect("B3 data"));
    let path = root.join("kairos").join("lab").join("check_data.krs");
    let src = std::fs::read_to_string(&path).expect("lab/check_data.krs");
    let buf = Rc::new(RefCell::new(Vec::new()));
    let host = Host::with_data(d);
    host.set_args(vec![data_dir.to_string_lossy().into_owned()]);
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), host);
    if let Err(e) = s.run(&path.to_string_lossy(), &src) {
        panic!("{}", s.render(&e));
    }
    let text = String::from_utf8(buf.borrow().clone()).expect("utf-8");
    for what in ["monthly returns: the same", "eligible: the same", "cdi: the same", "nefin: the same"] {
        assert!(text.contains(what), "`{what}` missing from the output:\n{text}");
    }
}
