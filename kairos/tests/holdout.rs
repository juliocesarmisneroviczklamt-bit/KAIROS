//! M5 acceptance tests: protocols fence the data, a holdout opens once (also across runs, through
//! the ledger), and every research evaluation is a trial in the registry, with its DSR.
//!
//! They run on a small synthetic market (30 assets, 2015-01..2020-12), so they need no B3 data
//! and run everywhere the CI does.

use kairos::bridge::{Host, REGISTRY};
use kairos::ledger::{self, Kind, Ledger};
use kairos::session::Session;
use quant_b3::common::Data;
use quant_b3::macrodata::Factors;
use quant_b3::panel::{Day, Panel, Tk};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
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

/// xorshift64*: deterministic returns on every platform.
struct Rng(u64);

impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// 72 months of 20 sessions, 30 liquid assets (one company each), CDI and NEFIN factors.
fn market() -> Rc<Data> {
    let mut dates = Vec::new();
    for y in 2015..=2020u32 {
        for m in 1..=12u32 {
            for d in 1..=20u32 {
                dates.push(y * 10000 + m * 100 + d);
            }
        }
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut tks = Vec::new();
    for i in 0..30 {
        let drift = (i as f64 - 15.0) * 0.00004;
        let days = (0..dates.len())
            .map(|di| Day {
                di: di as u32,
                r: drift + 0.03 * (rng.unit() - 0.5),
                vol: 2_000_000.0,
                div: 0.0,
                jcp: 0.0,
                spr: 0.002,
                px: 10.0,
            })
            .collect();
        tks.push(Tk {
            ticker: format!("SYN{i:02}3"),
            issuer: format!("COMPANY {i}"),
            class: "ON".into(),
            first: dates[0],
            last: *dates.last().unwrap(),
            days,
        });
    }
    let cdi: BTreeMap<u32, f64> = dates.iter().map(|&d| (d, 0.0004)).collect();
    let mut ew = BTreeMap::new();
    let mut ipca = BTreeMap::new();
    for y in 2015..=2020u32 {
        for m in 1..=12u32 {
            let f = Factors { rm_rf: 0.04 * (rng.unit() - 0.5), smb: 0.0, hml: 0.0, wml: 0.0, iml: 0.0, rf: 0.008 };
            ew.insert(y * 100 + m, f);
            ipca.insert(y * 100 + m, 0.004);
        }
    }
    Rc::new(Data::build(Panel { dates, tks }, &cdi, &ew, &ew.clone(), ipca))
}

/// A fresh results folder for one test.
fn results(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("kairos-m5-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn run(d: &Rc<Data>, dir: &Path, today: u32, src: &str) -> Result<String, String> {
    let host = Host::with_data(d.clone()).with_results(dir);
    host.set_today(today);
    host.set_program("m5-test");
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), host);
    let out = s.run("t.krs", src).map_err(|e| s.render(&e));
    out.map(|_| String::from_utf8(buf.borrow().clone()).unwrap())
}

/// Runs a test body on a large stack, like the CLI (the interpreter recurses on the HIR).
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

/// The weights: the best third of last month's returns among the liquid assets, equal weights.
const WEIGHTS: &str = r#"
fn weights() -> Tensor[month, asset; f64] {
    let ret = b3.monthly_returns()
    let elig = b3.eligible()
    var w = zeros_like(ret)
    for m in 13..size(ret, month) {
        let alive = elig[month: m] and not is_nan(ret[month: m - 1])
        let s = where(alive, ret[month: m - 1], nan)
        let cnt = int(nan_count(s, over: asset))
        if cnt < 6 { continue }
        let k = cnt / 3
        w[month: m] = where(top_k(s, k, over: asset), 1.0 / float(k), 0.0)
    }
    w
}
"#;

fn ledger_of(dir: &Path) -> Ledger {
    Ledger::open(&dir.join(ledger::FILE), Vec::new).unwrap()
}

#[test]
fn research_is_fenced_and_the_holdout_opens_once() {
    big_stack(|| {
        let d = market();
        let dir = results("once");
        let prog = format!(
            r#"protocol "syn-v1" {{
    research: until 201812
    holdout: from 201901
    min_months: 12
    rule: "accept if the holdout's CAPM alpha > 0"
}}
{WEIGHTS}
print("research months", size(b3.monthly_returns(), month), b3.months()[size(b3.monthly_returns(), month) - 1])
let ev = b3.evaluate(weights(), cost_bps: 10.0, lag: 0)
print("research", ev.months, ev.trials)
let h = b3.holdout("syn-v1")
let ho = open_once(h, || weights(), cost_bps: 10.0, lag: 0)
print("holdout", ho.months, ho.trials)
"#
        );
        let out = run(&d, &dir, 20210110, &prog).unwrap();
        // the research sees 2015-01..2018-12; the evaluation trades 2016-02..2018-12
        assert!(out.contains("research months 48 201812"), "{out}");
        assert!(out.contains("research 35 1\n"), "{out}");
        // the holdout: 2019-01..2020-12, a fresh portfolio
        assert!(out.contains("holdout 24 1\n"), "{out}");

        // the ledger: the protocol, what the research saw, the opening and its result
        let l = ledger_of(&dir);
        let kinds: Vec<Kind> = l.rows.iter().map(|r| r.kind).collect();
        assert_eq!(kinds, vec![Kind::Protocol, Kind::Seen, Kind::Opened], "{:?}", l.rows);
        assert_eq!(l.rows[1].data_until, 201812);
        let opened = &l.rows[2];
        assert_eq!((opened.holdout_from, opened.data_until, opened.date), (201901, 202012, 20210110));
        assert!(opened.note.starts_with("24 months; CAGR"), "{}", opened.note);

        // the registry: one research trial, by this program, over the research period only
        let reg = std::fs::read_to_string(dir.join(REGISTRY)).unwrap();
        let rows: Vec<&str> = reg.lines().skip(1).collect();
        assert_eq!(rows.len(), 1, "{reg}");
        assert!(rows[0].starts_with("kairos,m5-test,pesos "), "{reg}");
        assert!(rows[0].contains(",2016-02..2018-12,35,"), "{reg}");

        // a second run: `b3.holdout` refuses right away (R0502), before any research
        let e = run(&d, &dir, 20210111, &prog).unwrap_err();
        assert!(e.contains("error[R0502]") && e.contains("already opened on 2021-01-10"), "{e}");

        // running the same research again is not a new trial
        let again = format!("protocol \"syn-v1\" {{ research: until 201812  holdout: from 201901  min_months: 12  rule: \"accept if the holdout's CAPM alpha > 0\" }}\n{WEIGHTS}\nprint(b3.evaluate(weights(), cost_bps: 10.0, lag: 0).trials)\n");
        assert_eq!(run(&d, &dir, 20210112, &again).unwrap(), "1\n");

        // a new protocol cannot take months already seen as its holdout
        let late = "protocol \"syn-v2\" { research: until 201906  holdout: from 201907  rule: \"r\" }\nprint(size(b3.cdi(), month))\n";
        let e = run(&d, &dir, 20210113, late).unwrap_err();
        assert!(e.contains("error[R0502]") && e.contains("2021-01 or later"), "{e}");
    })
}

#[test]
fn a_pending_holdout_is_hidden_from_every_program_until_withdrawn() {
    big_stack(|| {
        let d = market();
        let dir = results("fence");
        let p = format!("protocol \"p\" {{ research: until 201806  holdout: from 201901  rule: \"r\" }}\n{WEIGHTS}\nprint(b3.evaluate(weights(), cost_bps: 10.0, lag: 1).months)\n");
        // with lag 1, 2018-06 would need the first session of 2018-07: it is not traded
        assert_eq!(run(&d, &dir, 20210110, &p).unwrap(), "28\n");
        // another program, without a protocol, sees up to 2018-12: 2019+ is reserved
        let plain = "print(size(b3.monthly_returns(), month), size(b3.eligible(), month), size(b3.cdi(), month), b3.month_index(202001))\n";
        assert_eq!(run(&d, &dir, 20210111, plain).unwrap(), "48 48 48 48\n");
        // withdrawing the protocol frees the months, and it can no longer be opened
        let host = Host::with_data(d.clone()).with_results(&dir);
        host.set_today(20210112);
        assert!(host.withdraw("p").unwrap().contains("withdrawn"));
        assert_eq!(run(&d, &dir, 20210113, plain).unwrap(), "72 72 72 60\n");
        let open = format!("{p}let h = b3.holdout(\"p\")\n");
        let e = run(&d, &dir, 20210114, &open).unwrap_err();
        assert!(e.contains("error[R0502]"), "{e}");
    })
}

#[test]
fn an_unripe_holdout_stays_closed_and_a_failure_inside_spends_it() {
    big_stack(|| {
        let d = market();
        let dir = results("unripe");
        let proto = "protocol \"q\" { research: until 202006  holdout: from 202007  rule: \"r\" }\n";
        let prog = format!("{proto}{WEIGHTS}let h = b3.holdout(\"q\")\nprint(open_once(h, || weights(), lag: 0).months)\n");
        // 6 months of holdout, and the protocol asks for 12 (the default): nothing is recorded
        let e = run(&d, &dir, 20210110, &prog).unwrap_err();
        assert!(e.contains("error[R0503]") && e.contains("at least 12"), "{e}");
        assert!(ledger_of(&dir).rows.iter().all(|r| r.kind != Kind::Opened));

        // `b3.evaluate` inside the weights function: an error, and the holdout is spent
        let ripe = "protocol \"r\" { research: until 201912  holdout: from 202001  rule: \"r\" }\n";
        let prog = format!("{ripe}{WEIGHTS}let h = b3.holdout(\"r\")\nprint(open_once(h, || {{ print(b3.evaluate(weights()).months); weights() }}, lag: 0).months)\n");
        let e = run(&d, &dir, 20210111, &prog).unwrap_err();
        assert!(e.contains("error[R0504]") && e.contains("is spent"), "{e}");
        let l = ledger_of(&dir);
        let opened = l.rows.iter().find(|r| r.kind == Kind::Opened && r.protocol == "r").unwrap();
        assert!(opened.note.starts_with("failed:"), "{}", opened.note);
        let e = run(&d, &dir, 20210112, &prog).unwrap_err();
        assert!(e.contains("error[R0502]"), "{e}");
    })
}

#[test]
fn the_ledger_starts_with_the_labs_search() {
    big_stack(|| {
        let d = market();
        let dir = results("seed");
        std::fs::write(
            dir.join("busca_resultado.json"),
            "{\n  \"data\": \"2020-07-01\",\n  \"motivo\": \"nenhum alpha\",\n  \"protocolo_sha256\": \"e83eb427\"\n}",
        )
        .unwrap();
        std::fs::write(dir.join("busca_mensal.csv"), "mes,MOM\n2019-01,0.01\n2020-06,0.02\n").unwrap();
        // the search opened 2019-01..2020-06: a holdout from 2020-01 is refused...
        let e = run(&d, &dir, 20210110, "protocol \"a\" { research: until 201912  holdout: from 202001  rule: \"r\" }\nprint(b3.months()[0])\n")
            .unwrap_err();
        assert!(e.contains("error[R0502]") && e.contains("busca-de-alpha") && e.contains("2020-07 or later"), "{e}");
        // ...and one from 2020-07 is fine
        let ok = run(&d, &dir, 20210110, "protocol \"b\" { research: until 202006  holdout: from 202007  min_months: 6  rule: \"r\" }\nprint(size(b3.cdi(), month))\n");
        assert_eq!(ok.unwrap(), "66\n");
        let l = ledger_of(&dir);
        assert_eq!(l.rows[0].kind, Kind::Opened);
        assert_eq!((l.rows[0].holdout_from, l.rows[0].data_until, l.rows[0].date), (201901, 202006, 20200701));
    })
}

/// The same weights are one strategy, one trial, however they are executed (here: another
/// delay); other weights are another trial.
#[test]
fn without_a_results_folder_the_trials_are_counted_in_memory() {
    big_stack(|| {
        let d = market();
        let host = Host::with_data(d.clone());
        let buf = Rc::new(RefCell::new(Vec::new()));
        let mut s = Session::with_host(Box::new(Buf(buf.clone())), host);
        let src = format!(
            "{WEIGHTS}let a = b3.evaluate(weights(), lag: 0)\nlet b = b3.evaluate(weights(), lag: 0)\nlet c = b3.evaluate(weights(), lag: 1)\nlet d = b3.evaluate(weights() * 0.5, lag: 0)\nprint(a.trials, b.trials, c.trials, d.trials)\nprint(a.dsr >= 0.0 and a.dsr <= 1.0)\n"
        );
        s.run("t.krs", &src).map_err(|e| s.render(&e)).unwrap();
        assert_eq!(String::from_utf8(buf.borrow().clone()).unwrap(), "1 1 1 2\ntrue\n");
    })
}
