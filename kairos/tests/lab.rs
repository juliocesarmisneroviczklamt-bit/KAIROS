//! M8: the lab in Kairós. Each rule ported from quant-b3 (Rust) to `kairos/lab/*.krs` must give
//! exactly the same result as the Rust it replaces. These tests use a synthetic market built to
//! hit the edge cases (so they run everywhere); the same checks run on the real B3 data with
//! `kairos run kairos/lab/check_universe.krs`.

use kairos::bridge::Host;
use kairos::session::Session;
use quant_b3::common::Data;
use quant_b3::macrodata::Factors;
use quant_b3::panel::{Day, Panel, Tk};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
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
}

/// 2001..2004, months of 19 to 22 sessions. The assets:
/// - AAAA3/AAAA4: one company, the leading class changes from year to year;
/// - BBBB3: liquid on about 85% of the days in 2001, 70% in 2002 (in, then out);
/// - CCCC11: an ETF (class CI), very liquid, never eligible;
/// - DDDD3: lists in July 2002 (its 2002 window starts there);
/// - EEEE3: lists in December 2002 (not eligible in 2003);
/// - FFFF3: stops trading in March 2002 (not eligible in 2003);
/// - GGGG3/GGGG4: one company, the same volume every day (the first class wins);
/// - 20 plain liquid stocks.
fn market() -> Rc<Data> {
    let mut dates = Vec::new();
    for y in 2001..=2004u32 {
        for m in 1..=12u32 {
            let n = 19 + (m + y) % 4;
            for d in 1..=n {
                dates.push(y * 10000 + m * 100 + d);
            }
        }
    }
    let mut rng = Rng(0x0DDB_1A5E_5BAD_5EED);
    let mut tks = Vec::new();
    let mut add = |ticker: &str, class: &str, vol: &dyn Fn(u32, &mut Rng) -> Option<f64>, rng: &mut Rng| {
        let mut days = Vec::new();
        for (di, &date) in dates.iter().enumerate() {
            if let Some(v) = vol(date, rng) {
                days.push(Day { di: di as u32, r: 0.02 * (rng.unit() - 0.5), vol: v, div: 0.0, jcp: 0.0, spr: 0.002, px: 10.0 });
            }
        }
        if days.is_empty() {
            return;
        }
        let (first, last) = (dates[days[0].di as usize], dates[days[days.len() - 1].di as usize]);
        tks.push(Tk { ticker: ticker.into(), issuer: ticker[..4].into(), class: class.into(), first, last, days });
    };
    // the leading class flips: AAAA4 leads in odd years, AAAA3 in even ones
    add("AAAA3", "ON", &|d, _| Some(if (d / 10000) % 2 == 0 { 9e6 } else { 2e6 }), &mut rng);
    add("AAAA4", "PN", &|d, _| Some(if (d / 10000) % 2 == 1 { 9e6 } else { 2e6 }), &mut rng);
    add("BBBB3", "ON", &|d, r| {
        let p = if d < 20020101 { 0.86 } else { 0.70 };
        Some(if r.unit() < p { 800_000.0 } else { 200_000.0 })
    }, &mut rng);
    add("CCCC11", "CI", &|_, _| Some(5e7), &mut rng);
    add("DDDD3", "ON", &|d, _| (d >= 20020710).then_some(3e6), &mut rng);
    add("EEEE3", "ON", &|d, _| (d >= 20021205).then_some(3e6), &mut rng);
    add("FFFF3", "ON", &|d, _| (d < 20020301).then_some(3e6), &mut rng);
    add("GGGG3", "ON", &|_, _| Some(4e6), &mut rng);
    add("GGGG4", "PN", &|_, _| Some(4e6), &mut rng);
    for i in 0..20 {
        let t = format!("P{i:03}3");
        // a few days without trades
        add(&t, "ON", &|_, r| (r.unit() > 0.03).then_some(1e6 + 1e5 * r.unit()), &mut rng);
    }
    let cdi: BTreeMap<u32, f64> = dates.iter().map(|&d| (d, 0.0004)).collect();
    let mut ff = BTreeMap::new();
    let mut ipca = BTreeMap::new();
    for y in 2001..=2004u32 {
        for m in 1..=12u32 {
            ff.insert(y * 100 + m, Factors { rm_rf: 0.01, smb: 0.0, hml: 0.0, wml: 0.0, iml: 0.0, rf: 0.008 });
            ipca.insert(y * 100 + m, 0.004);
        }
    }
    Rc::new(Data::build(Panel { dates, tks }, &cdi, &ff, &ff.clone(), ipca))
}

fn run(d: &Rc<Data>, src: &str) -> String {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), Host::with_data(d.clone()));
    // a program in the lab folder: its imports resolve there
    let name = concat!(env!("CARGO_MANIFEST_DIR"), "/lab/test.krs");
    if let Err(e) = s.run(name, src) {
        panic!("{}", s.render(&e));
    }
    String::from_utf8(buf.borrow().clone()).unwrap()
}

fn on_big_stack(f: fn()) {
    use kairos::interp::STACK_BYTES;
    use std::sync::atomic::Ordering;
    let spawned = std::thread::Builder::new().stack_size(256 << 20).spawn(move || {
        STACK_BYTES.store(256 << 20, Ordering::Relaxed);
        f()
    });
    match spawned {
        Ok(h) => {
            if let Err(p) = h.join() {
                std::panic::resume_unwind(p);
            }
        }
        Err(_) => {
            STACK_BYTES.store(4 << 20, Ordering::Relaxed);
            f()
        }
    }
}

#[test]
fn the_nefin_universe_in_kairos_is_the_rust_one() {
    on_big_stack(|| {
        let d = market();
        let out = run(
            &d,
            r#"
import "universe.krs"
let mine = nefin_universe(500000.0)
let rust = b3.eligible()
print(sum(where(where(mine, 1.0, 0.0) != where(rust, 1.0, 0.0), 1.0, 0.0)))
let t = b3.tickers()
let jan = [b3.month_index(200201), b3.month_index(200301), b3.month_index(200401)]
for m in jan {
    var names = []
    for a in 0..len(t) {
        if mine[month: m, asset: a] and not (t[a] == "P0003" or t[a] == "P0013") {
            names.push(t[a])
        }
    }
    print(b3.months()[m], names)
}
"#,
        );
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "0.0", "the Kairós universe differs from the Rust one:\n{out}");
        // 2002 (from 2001): AAAA4 leads, BBBB3 liquid, GGGG3 wins the tie; no ETF
        assert!(lines[1].starts_with("200201 [\"AAAA4\", \"BBBB3\", \"FFFF3\", \"GGGG3\""), "{out}");
        // 2003 (from 2002): AAAA3 leads, BBBB3 out, DDDD3 in (from its listing), EEEE3 and FFFF3 out
        assert!(lines[2].starts_with("200301 [\"AAAA3\", \"DDDD3\", \"GGGG3\""), "{out}");
        assert!(lines[3].starts_with("200401 [\"AAAA4\", \"DDDD3\", \"EEEE3\", \"GGGG3\""), "{out}");
    })
}

fn list(xs: &[f64]) -> String {
    format!("[{}]", xs.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(", "))
}

#[test]
fn the_statistics_in_kairos_are_the_labs_bit_for_bit() {
    on_big_stack(|| {
        use quant_b3::stats;
        let mut rng = Rng(0x5EED_0F_57A7_5u64);
        let mut series = |n: usize, scale: f64| -> Vec<f64> { (0..n).map(|_| scale * (rng.unit() - 0.45)).collect() };
        let (x, y, z, w) = (series(240, 0.08), series(240, 0.05), series(240, 0.03), series(240, 0.02));
        let src = format!(
            "import \"stats.krs\"\nlet x = {}\nlet y = {}\nlet z = {}\nlet w = {}\nprint(avg(x))\nprint(sd(x))\nprint(corr(x, y))\nprint(max_dd(x))\nprint(nw_lags(len(x)))\nprint(cagr(x, 12.0))\nlet o = ols_nw(y, [x, z, w], nw_lags(len(y)))\nfor v in o.coef {{ print(v) }}\nfor v in o.t {{ print(v) }}\nprint(o.r2)\nlet a = annual(x)\nprint(a.0)\nprint(a.1)\nprint(a.2)\nprint(a.3)\n",
            list(&x),
            list(&y),
            list(&z),
            list(&w)
        );
        let out = run(&market(), &src);
        let got: Vec<f64> = out.lines().map(|l| l.trim().parse::<f64>().unwrap()).collect();
        let o = stats::ols_nw(&y, &[&x, &z, &w], stats::nw_lags(y.len())).unwrap();
        let ann = stats::ols_nw(&x, &[], stats::nw_lags(x.len())).unwrap();
        let (m, s) = (stats::mean(&x), stats::std(&x));
        let mut want = vec![m, s, stats::corr(&x, &y), stats::max_dd(&x), stats::nw_lags(x.len()) as f64];
        let cagr = stats::cagr(&x, 12.0);
        want.extend(o.coef.iter().chain(&o.t));
        want.push(o.r2);
        want.extend([m * 12.0, s * 12f64.sqrt(), ann.t[0], m / s * 12f64.sqrt()]);
        // everything but the CAGR (x^y: pure-Rust libm in Kairós, the system's pow in quant-b3)
        let mut got_exact = got.clone();
        let got_cagr = got_exact.remove(5);
        assert!((got_cagr - cagr).abs() <= 1e-15 * cagr.abs().max(1.0), "cagr {got_cagr} vs {cagr}");
        assert_eq!(got_exact.len(), want.len(), "{out}");
        for (i, (g, w)) in got_exact.iter().zip(&want).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "value {i}: Kairós {g:?}, Rust {w:?}");
        }
    })
}

#[test]
fn report_formatting_is_rusts() {
    on_big_stack(|| {
        let vals = [0.0, -0.0, 0.15, 0.25, 2.5, -2.5, 0.0005, -0.0004, 1.005, 123.456789, -98.7654, 1e-9, f64::NAN, f64::INFINITY, -f64::INFINITY];
        let mut src = String::from("import \"fmt.krs\"\n");
        for v in vals {
            let lit = if v.is_nan() {
                "nan".to_string()
            } else if v.is_infinite() {
                if v > 0.0 { "inf".into() } else { "-inf".into() }
            } else {
                format!("{v:?}")
            };
            src.push_str(&format!("print(fixed({lit}, 1) + \"|\" + fixed({lit}, 2) + \"|\" + signed({lit}, 1) + \"|\" + pct({lit}) + \"|\" + right(fixed({lit}, 2), 9) + \"|\" + left(fixed({lit}, 0), 6) + \"|\")\n"));
        }
        src.push_str("print(ym_text(202610), date_text(20261030), join([\"a\", \"b\", \"c\"], \" | \"))\n");
        let out = run(&market(), &src);
        let mut want = String::new();
        for v in vals {
            let pct = if v.is_finite() { format!("{:+.1}%", v * 100.0) } else { "n/d".into() };
            want.push_str(&format!("{v:.1}|{v:.2}|{v:+.1}|{pct}|{:>9}|{:<6}|\n", format!("{v:.2}"), format!("{v:.0}")));
        }
        want.push_str("2026-10 2026-10-30 a | b | c\n");
        assert_eq!(out, want);
    })
}

/// A study written in Kairós prints exactly the report of the Rust study it replaces. Needs the
/// real data (and the Rust report), so these run by hand:
/// `cargo test --release -p kairos --test lab -- --ignored`.
fn same_report(study: &'static str, report: &'static str) {
    on_big_stack_with(move || {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let report = root.join("results").join(report);
        let Ok(want) = std::fs::read_to_string(&report) else {
            eprintln!("skipped: no {}", report.display());
            return;
        };
        let path = root.join("kairos").join("lab").join(study);
        let src = std::fs::read_to_string(&path).unwrap();
        let buf = Rc::new(RefCell::new(Vec::new()));
        let mut s = Session::new(Box::new(Buf(buf.clone())));
        if let Err(e) = s.run(&path.to_string_lossy(), &src) {
            panic!("{}", s.render(&e));
        }
        let got = String::from_utf8(buf.borrow().clone()).unwrap();
        let lines = |t: &str| t.lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect::<Vec<_>>();
        assert_eq!(lines(&got), lines(&want));
    })
}

fn on_big_stack_with(f: impl FnOnce() + Send + 'static) {
    use kairos::interp::STACK_BYTES;
    use std::sync::atomic::Ordering;
    // WebAssembly has no threads: the main one, with the stack CI gives wasmtime (4 MiB)
    if cfg!(target_family = "wasm") {
        STACK_BYTES.store(4 << 20, Ordering::Relaxed);
        return f();
    }
    let h = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            STACK_BYTES.store(256 << 20, Ordering::Relaxed);
            f()
        })
        .unwrap();
    if let Err(p) = h.join() {
        std::panic::resume_unwind(p);
    }
}

/// `quant-b3 momentum` → results/momentum_relatorio.txt
#[test]
#[ignore]
fn the_momentum_study_in_kairos_prints_the_rust_report() {
    same_report("momentum.krs", "momentum_relatorio.txt");
}

/// `quant-b3 momentum-risco` → results/risco_relatorio.txt
#[test]
#[ignore]
fn the_risk_study_in_kairos_prints_the_rust_report() {
    same_report("risk.krs", "risco_relatorio.txt");
}

/// `quant-b3 realista` → results/realista_relatorio.txt
#[test]
#[ignore]
fn the_realistic_study_in_kairos_prints_the_rust_report() {
    same_report("realistic.krs", "realista_relatorio.txt");
}

/// `quant-b3 small-caps` → results/smallcaps_relatorio.txt
#[test]
#[ignore]
fn the_smallcaps_study_in_kairos_prints_the_rust_report() {
    same_report("smallcaps.krs", "smallcaps_relatorio.txt");
}

/// `quant-b3 busca` → results/busca_relatorio.txt (the registry is upserted with the same rows)
#[test]
#[ignore]
fn the_alpha_search_in_kairos_prints_the_rust_report() {
    same_report("search.krs", "busca_relatorio.txt");
}

/// The small segment (lab/universe.krs `small_universe`) is quant-b3's `small_segment`.
#[test]
fn the_small_segment_in_kairos_is_the_rust_one() {
    on_big_stack(|| {
        let d = market();
        let out = run(&d, "import \"universe.krs\"
let s = small_universe(500000.0, 50000.0)
let t = b3.tickers()
for m in [b3.month_index(200201), b3.month_index(200301), b3.month_index(200401)] {
    var names = []
    for a in 0..len(t) {
        if s[month: m, asset: a] { names.push(t[a]) }
    }
    print(names)
}
");
        let mut want = String::new();
        for y in [2002u32, 2003, 2004] {
            let mut names: Vec<String> = quant_b3::universe::small_segment(&d.panel, y, 500_000.0, 50_000.0).iter().map(|&t| format!("{:?}", d.panel.tks[t].ticker)).collect();
            names.sort();
            want.push_str(&format!("[{}]
", names.join(", ")));
        }
        assert_eq!(out, want);
    })
}

/// The registry report written in Kairós (lab/registry.krs: the CSV, the Deflated Sharpe Ratio,
/// the normal CDF and its inverse) prints quant-b3's `registro` exactly.
#[test]
fn the_registry_report_in_kairos_is_the_rust_one() {
    on_big_stack_with(|| {
        use quant_b3::registry::{self, Trial};
        // WASI has no temporary folder of its own (CI maps the host's root)
        let tmp = if cfg!(target_family = "wasm") { std::path::PathBuf::from("/tmp") } else { std::env::temp_dir() };
        // (and no process ids: one test process at a time there)
        let pid = if cfg!(target_family = "wasm") { 0 } else { std::process::id() };
        let dir = tmp.join(format!("kairos_registry_{pid}"));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("registro_testes.csv");
        let _ = std::fs::remove_file(&path);
        let run = |path: &std::path::Path| {
            let src = std::fs::read_to_string(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lab").join("registry.krs")).unwrap();
            let buf = Rc::new(RefCell::new(Vec::new()));
            let mut s = Session::new(Box::new(Buf(buf.clone())));
            s.host().set_args(vec![path.to_string_lossy().into_owned()]);
            let lab = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lab").join("registry.krs");
            if let Err(e) = s.run(&lab.to_string_lossy(), &src) {
                panic!("{}", s.render(&e));
            }
            String::from_utf8(buf.borrow().clone()).unwrap()
        };
        // the empty registry
        assert_eq!(run(&path), registry::report(&path));
        // trials of different lengths, drifts, skews and fat tails (one with a comma in its name)
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for (k, (study, name, n, drift)) in [
            ("momentum", "WML long-short", 310, 0.012),
            ("busca", "MOM, com custo", 216, 0.004),
            ("busca", "BAIXA_VOL", 216, 0.002),
            ("kairos", "tax_portfolio", 250, -0.001),
            ("small-caps", "H1", 120, 0.006),
            ("risco", "vol alvo", 48, 0.020),
        ]
        .into_iter()
        .enumerate()
        {
            let excess: Vec<f64> = (0..n)
                .map(|i| {
                    let u = rng.unit() - 0.5;
                    let jump = if (i + k) % 37 == 0 { -0.15 } else { 0.0 };
                    drift + 0.08 * u * u * u.signum() + jump
                })
                .collect();
            registry::upsert(&path, &Trial::new(study, name, "params", "2001-2018", &excess)).unwrap();
            assert_eq!(run(&path), registry::report(&path), "after {} trials", k + 1);
        }
        let _ = std::fs::remove_dir_all(&dir);
    })
}
