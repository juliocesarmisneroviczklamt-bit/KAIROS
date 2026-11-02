//! `kairos-bench`: the same numeric programs in Kairós (interpreter and native code) and in Rust,
//! with the same semantics: checked i64 arithmetic, IEEE f64, and the same pure-Rust `exp`/`ln`.
//! The results must agree to the last bit; the times are the best of several runs.
//!
//!     cargo run --release -p kairos --bin kairos-bench

use kairos::session::Session;
use std::cell::RefCell;
use std::hint::black_box;
use std::io::Write;
use std::rc::Rc;
use std::time::Instant;

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

// ------------------------------------------------------------------ Rust, same semantics

fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n.checked_sub(1).unwrap()).checked_add(fib(n.checked_sub(2).unwrap())).unwrap() }
}

fn sum_sqrt(n: i64) -> f64 {
    let mut s = 0.0;
    let mut i = 1i64;
    while i <= n {
        s += (i as f64).sqrt();
        i = i.checked_add(1).unwrap();
    }
    s
}

fn chain(n: i64) -> i64 {
    let mut x = n;
    let mut steps = 1i64;
    while x != 1 {
        if x.checked_rem(2).unwrap() == 0 {
            x = x.checked_div(2).unwrap();
        } else {
            x = x.checked_mul(3).unwrap().checked_add(1).unwrap();
        }
        steps = steps.checked_add(1).unwrap();
    }
    steps
}

fn longest(limit: i64) -> i64 {
    let (mut best, mut arg) = (0i64, 0i64);
    for n in 1..limit {
        let c = chain(n);
        if c > best {
            best = c;
            arg = n;
        }
    }
    arg.checked_mul(1000).unwrap().checked_add(best).unwrap()
}

fn ncdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.2316419 * x.abs());
    let poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    let tail = libm::exp(-0.5 * x * x) / (2.0 * 3.141592653589793f64).sqrt() * poly;
    if x >= 0.0 { 1.0 - tail } else { tail }
}

fn call_price(s: f64, k: f64, r: f64, vol: f64, t: f64) -> f64 {
    let d1 = (libm::log(s / k) + (r + 0.5 * vol * vol) * t) / (vol * t.sqrt());
    let d2 = d1 - vol * t.sqrt();
    s * ncdf(d1) - k * libm::exp(-r * t) * ncdf(d2)
}

fn book(n: i64) -> f64 {
    let mut total = 0.0;
    for i in 0..n {
        let k = 50.0 + (i.checked_rem(100).unwrap()) as f64;
        let t = 0.25 + (i.checked_rem(7).unwrap()) as f64 * 0.25;
        total += call_price(100.0, k, 0.1, 0.3, t);
    }
    total
}

fn inside(cx: f64, cy: f64, max_iter: i64) -> bool {
    let (mut x, mut y, mut i) = (0.0f64, 0.0f64, 0i64);
    while i < max_iter && x * x + y * y <= 4.0 {
        let xt = x * x - y * y + cx;
        y = 2.0 * x * y + cy;
        x = xt;
        i = i.checked_add(1).unwrap();
    }
    i == max_iter
}

fn count(w: i64, h: i64, max_iter: i64) -> i64 {
    let mut n = 0i64;
    for py in 0..h {
        for px in 0..w {
            let cx = -2.0 + 3.0 * px as f64 / w as f64;
            let cy = -1.2 + 2.4 * py as f64 / h as f64;
            if inside(cx, cy, max_iter) {
                n = n.checked_add(1).unwrap();
            }
        }
    }
    n
}

fn pi_estimate(n: i64) -> f64 {
    let (mut seed, mut hits) = (12345i64, 0i64);
    for _ in 0..n {
        seed = seed.checked_mul(16807).unwrap().checked_rem(2147483647).unwrap();
        let x = seed as f64 / 2147483647.0;
        seed = seed.checked_mul(16807).unwrap().checked_rem(2147483647).unwrap();
        let y = seed as f64 / 2147483647.0;
        if x * x + y * y <= 1.0 {
            hits = hits.checked_add(1).unwrap();
        }
    }
    4.0 * hits as f64 / n as f64
}

// ------------------------------------------------------------------ the runs

struct Bench {
    name: &'static str,
    src: &'static str,
    call: String,
    rust: Box<dyn Fn() -> String>,
}

fn best<T>(reps: usize, mut f: impl FnMut() -> T) -> (f64, T) {
    let mut out = None;
    let mut t = f64::INFINITY;
    for _ in 0..reps {
        let t0 = Instant::now();
        let r = f();
        t = t.min(t0.elapsed().as_secs_f64());
        out = Some(r);
    }
    (t, out.expect("at least one run"))
}

/// Time of the call alone (the definitions are loaded first), and what it printed.
fn kairos(b: &Bench, jit: bool, reps: usize) -> (f64, String) {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::new(Box::new(Buf(buf.clone())));
    s.set_jit(jit);
    if let Err(d) = s.run(b.name, b.src) {
        panic!("{}", s.render(&d));
    }
    let line = format!("print({})", b.call);
    best(reps, || {
        buf.borrow_mut().clear();
        if let Err(d) = s.run("bench", &line) {
            panic!("{}", s.render(&d));
        }
        String::from_utf8(buf.borrow().clone()).unwrap().trim().to_string()
    })
}

fn ms(t: f64) -> String {
    if t >= 1.0 { format!("{t:.2} s") } else { format!("{:.1} ms", t * 1e3) }
}

fn main() {
    // the interpreter recurses on the HIR: give it a large stack, as the CLI does
    let worker = std::thread::Builder::new().stack_size(512 << 20).spawn(|| {
        kairos::interp::STACK_BYTES.store(512 << 20, std::sync::atomic::Ordering::Relaxed);
        run()
    });
    std::process::exit(match worker.map(|h| h.join()) {
        Ok(Ok(code)) => code,
        _ => 1,
    });
}

fn run() -> i32 {
    let benches = vec![
        Bench {
            name: "fib",
            src: include_str!("../../bench/fib.krs"),
            call: "fib(30)".into(),
            rust: Box::new(|| format!("{}", fib(black_box(30)))),
        },
        Bench {
            name: "sum_sqrt",
            src: include_str!("../../bench/sum_sqrt.krs"),
            call: "sum_sqrt(20000000)".into(),
            rust: Box::new(|| format!("{:?}", sum_sqrt(black_box(20_000_000)))),
        },
        Bench {
            name: "collatz",
            src: include_str!("../../bench/collatz.krs"),
            call: "longest(100000)".into(),
            rust: Box::new(|| format!("{}", longest(black_box(100_000)))),
        },
        Bench {
            name: "black_scholes",
            src: include_str!("../../bench/black_scholes.krs"),
            call: "book(300000)".into(),
            rust: Box::new(|| format!("{:?}", book(black_box(300_000)))),
        },
        Bench {
            name: "mandelbrot",
            src: include_str!("../../bench/mandelbrot.krs"),
            call: "grid(300, 200, 200)".into(),
            rust: Box::new(|| format!("{}", count(black_box(300), black_box(200), black_box(200)))),
        },
        Bench {
            name: "monte_carlo",
            src: include_str!("../../bench/monte_carlo.krs"),
            call: "pi_estimate(2000000)".into(),
            rust: Box::new(|| format!("{:?}", pi_estimate(black_box(2_000_000)))),
        },
    ];
    let native = kairos::native::Jit::new().enabled();
    println!("kairos-bench (native code: {})", if native { "on" } else { "not available on this target" });
    println!();
    println!("| benchmark | interpreter | native (JIT) | Rust | JIT / Rust | same result |");
    println!("|---|---|---|---|---|---|");
    let mut ok = true;
    let mut ratios = Vec::new();
    for b in &benches {
        let (ti, ri) = kairos(b, false, 1);
        let (tj, rj) = kairos(b, true, 7);
        let (tr, rr) = best(7, || (b.rust)());
        let same = ri == rr && rj == rr;
        ok &= same;
        let ratio = tj / tr;
        ratios.push(ratio);
        println!(
            "| {} | {} | {} | {} | {:.2}x | {} |",
            b.name,
            ms(ti),
            ms(tj),
            ms(tr),
            ratio,
            if same { "yes".to_string() } else { format!("NO: interpreter {ri}, JIT {rj}, Rust {rr}") }
        );
    }
    let geo = (ratios.iter().map(|r| r.ln()).sum::<f64>() / ratios.len() as f64).exp();
    println!();
    println!("geometric mean of JIT / Rust: {geo:.2}x");
    if ok { 0 } else { 1 }
}
