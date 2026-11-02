//! M7: native code must behave exactly like the interpreter — the same output, the same bits,
//! the same runtime errors with the same messages and spans. Every program runs twice, with the
//! JIT on and off. On targets without the JIT both runs use the interpreter (and still pass).

use kairos::session::Session;
use std::cell::RefCell;
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

/// (output, rendered error, functions compiled to machine code)
fn run(src: &str, jit: bool) -> (String, Option<String>, usize) {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::new(Box::new(Buf(buf.clone())));
    s.set_jit(jit);
    let err = s.run("t.krs", src).err().map(|d| s.render(&d));
    let out = String::from_utf8(buf.borrow().clone()).unwrap();
    (out, err, s.native_functions())
}

/// Same results both ways; returns how many functions were compiled with the JIT on.
fn same(src: &str) -> usize {
    let (o1, e1, n) = run(src, true);
    let (o0, e0, _) = run(src, false);
    assert_eq!(o1, o0, "output differs with the JIT:\n{src}");
    assert_eq!(e1, e0, "error differs with the JIT:\n{src}");
    n
}

/// The interpreter recurses on the HIR: a large stack, and the depth limit that goes with it.
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
        // WebAssembly: CI runs wasmtime with -W max-wasm-stack=4194304
        Err(_) => {
            STACK_BYTES.store(4 << 20, Ordering::Relaxed);
            f()
        }
    }
}

const SCALAR: &str = r#"
fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}
fn even(n: i64) -> bool { if n == 0 { true } else { odd(n - 1) } }
fn odd(n: i64) -> bool { if n == 0 { false } else { even(n - 1) } }
fn sum_sqrt(n: i64) -> f64 {
    var s = 0.0
    for i in 1..=n {
        s += sqrt(float(i))
    }
    s
}
fn collatz(n: i64) -> i64 {
    var x = n
    var steps = 0
    while x != 1 {
        if x % 2 == 0 { x = x / 2 } else { x = 3 * x + 1 }
        steps += 1
    }
    steps
}
fn math(a: f64, b: f64) -> f64 {
    exp(ln(abs(a - b) + 1.0)) + round(a) + floor(b) + ceil(-a) + a ^ 2.0 + 7.5 % 2.0 - sqrt(b * b)
}
fn loops(limit: i64) -> i64 {
    var k = 0
    var acc = 0
    for i in 0..1000 {
        if i % 3 == 0 { continue }
        acc += i
        if i * i > limit { k = i; break }
    }
    k * 1000000 + acc
}
fn early(x: f64) -> f64 {
    if x < 0.0 { return -1.0 }
    if x > 100.0 { return 1.0 }
    x / 100.0
}
fn logic(a: i64, b: i64) -> bool {
    (a > b and a % 2 == 0) or (not (a == b) and b < 0)
}
fn ints(a: i64, b: i64) -> i64 {
    min(a, b) * 3 - max(a, b) + abs(a - b) + a / 7 - b % 5 + 2 ^ 10 + int(float(a) * 1.5)
}
fn nan_rules(x: f64) -> bool {
    let n = x - x * 1.0 / 0.0
    n < 1.0 or n >= 1.0 or n == n
}
fn signed_zero() -> f64 {
    1.0 / min(0.0, -0.0) + 1.0 / max(0.0, -0.0)
}
fn conds(a: i64, b: i64) -> i64 {
    var n = 0
    var i = a
    while i < b and not (i % 7 == 0 and i > a + 20) or i == a {
        if i % 2 == 0 or (i % 4 != 0 and not (i < 0)) { n += 1 }
        if not (i % 8 == 0) and (i % 3 == 0 or i % 5 == 0) { n += 10 }
        i += 1
    }
    n * 1000 + i
}
print(fib(24), even(1001), odd(1001))
print(sum_sqrt(100000))
print(collatz(837799), collatz(27))
print(math(2.5, -1.25), math(-7.75, 3.0))
print(loops(5000), loops(1000000000))
print(early(-3.0), early(250.0), early(42.0))
print(logic(8, 3), logic(3, 8), logic(3, -8), logic(4, 4))
print(ints(17, -4), ints(-100, 9))
print(min(0.0 / 0.0, 1.0), max(2.0, 0.0 / 0.0), nan_rules(1.0))
print(signed_zero())
print(int(3.99), int(-3.99), round(2.5), round(-2.5))
print(conds(-40, 60), conds(5, 6), conds(-3, -1), -7 % 2 == 0, -8 % 4 == 0, -9 % 4)
"#;

#[test]
fn scalar_code_gives_the_same_bits() {
    on_big_stack(|| {
        let n = same(SCALAR);
        if cfg!(any(
            all(target_arch = "x86_64", any(target_os = "linux", target_os = "macos", target_os = "windows", target_os = "freebsd")),
            all(target_arch = "aarch64", any(target_os = "linux", target_os = "macos"))
        )) {
            // every function above runs as machine code
            assert_eq!(n, 13, "functions compiled");
        }
    })
}

#[test]
fn runtime_errors_are_the_same() {
    on_big_stack(|| {
        for (src, code) in [
            ("fn f(x: i64) -> i64 { x + 1 }\nprint(f(9223372036854775807))", "R0006"),
            ("fn f(x: i64) -> i64 { x * x * x }\nprint(f(3000000))", "R0006"),
            ("fn f(x: i64) -> i64 { -x }\nprint(f(-9223372036854775807 - 1))", "R0006"),
            ("fn f(x: i64) -> i64 { abs(x) }\nprint(f(-9223372036854775807 - 1))", "R0006"),
            ("fn f(x: i64, y: i64) -> i64 { x / y }\nprint(f(7, 0))", "R0002"),
            ("fn f(x: i64, y: i64) -> i64 { x % y }\nprint(f(7, 0))", "R0002"),
            ("fn f(x: i64, y: i64) -> i64 { x / y }\nprint(f(-9223372036854775807 - 1, -1))", "R0006"),
            ("fn f(x: i64) -> i64 { 2 ^ x }\nprint(f(-1))", "R0007"),
            ("fn f(x: i64) -> i64 { 3 ^ x }\nprint(f(64))", "R0006"),
            ("fn f(x: f64) -> i64 { int(x) }\nprint(f(1e300))", "R0009"),
            ("fn f(x: f64) -> i64 { int(x * 0.0 / 0.0) }\nprint(f(1.0))", "R0009"),
            // the error comes from a function called by a function
            ("fn g(x: i64) -> i64 { x * 4611686018427387904 }\nfn f(x: i64) -> i64 { g(x) + 1 }\nprint(f(1))\nprint(f(2))", "R0006"),
            ("fn down(n: i64) -> i64 { if n == 0 { 0 } else { 1 + down(n - 1) } }\nprint(down(10))\nprint(down(100000000))", "R0005"),
        ] {
            let (_, e1, _) = run(src, true);
            let (_, e0, _) = run(src, false);
            let e1 = e1.unwrap_or_else(|| panic!("expected {code}:\n{src}"));
            assert!(e1.contains(&format!("error[{code}]")), "expected {code}, got:\n{e1}");
            assert_eq!(Some(e1), e0, "the JIT's error differs:\n{src}");
            same(src);
        }
    })
}

#[test]
fn everything_else_still_runs_in_the_interpreter() {
    on_big_stack(|| {
        // lists, tensors, structs, strings, `inout`, generic and `grad`ed functions are not compiled,
        // and calls between them and native code mix freely
        let src = r#"
axis feature
fn sq(x: f64) -> f64 { x * x }
fn total(xs: List[f64]) -> f64 {
    var s = 0.0
    for x in xs { s += sq(x) }
    s
}
fn bump(inout x: i64) { x += 1 }
struct P { a: f64 }
fn norm(p: P) -> f64 { sq(p.a) }
fn loss(w: f64) -> f64 { sq(w - 3.0) }
var k = 1
bump(inout k)
print(total([1.0, 2.0, 3.0]), norm(P { a: 4.0 }), k)
print(grad(loss)(1.0), grad(sq)(5.0))
print(sum(tensor([1.0, 2.0], feature) * sq(2.0)))
"#;
        same(src);
    })
}
