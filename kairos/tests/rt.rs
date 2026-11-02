//! The runtime of the WebAssembly modules (kairos/compiler/rt.krs) is written in Kairós: the
//! writing of floats (Ryu), their reading (parse_float), `%` on floats (fmod) and the mathematics
//! of libm (exp, ln, ^, tanh, sigmoid, softplus). Run in the interpreter, the programs of tests/rt
//! check each one against Rust and libm, bit for bit, on special values and on random ones, on
//! every platform of CI.

use kairos::session::Session;
use std::cell::RefCell;
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

/// Runs tests/rt/<program> with `count` random cases, on a thread with a big stack (the
/// interpreter of a debug build takes a lot of it for each call): it must find no difference.
fn check(program: &'static str, count: usize) {
    use kairos::interp::STACK_BYTES;
    use std::sync::atomic::Ordering;
    // WebAssembly (WASI) has no threads: there the checks run on the stack they have
    if cfg!(target_family = "wasm") {
        return check_here(program, count);
    }
    let h = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            STACK_BYTES.store(256 << 20, Ordering::Relaxed);
            check_here(program, count)
        })
        .unwrap();
    if let Err(p) = h.join() {
        std::panic::resume_unwind(p);
    }
}

fn check_here(program: &str, count: usize) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("rt").join(program);
    let src = std::fs::read_to_string(&path).unwrap();
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::new(Box::new(Buf(buf.clone())));
    s.host().set_args(vec![count.to_string()]);
    if let Err(e) = s.run(&path.to_string_lossy(), &src) {
        panic!("{}", s.render(&e));
    }
    let out = String::from_utf8(buf.borrow().clone()).unwrap();
    assert!(out.trim_end().ends_with(", 0 different"), "{program}:\n{out}");
    println!("{program}: {}", out.trim_end());
}

#[test]
fn floats_are_written_as_rust_writes_them() {
    check("ryu.krs", 2000);
}

#[test]
fn the_remainder_of_floats_is_exact() {
    check("fmod.krs", 2000);
}

#[test]
fn exp_ln_and_powers_are_the_bits_of_libm() {
    check("math.krs", 300);
}

#[test]
fn tanh_sigmoid_and_softplus_are_the_bits_of_libm() {
    check("act.krs", 2000);
}

#[test]
fn randn_draws_the_samples_of_the_interpreter() {
    check("randn.krs", 2001);
}

#[test]
fn the_tape_of_grad_gives_the_gradients_of_the_interpreter() {
    check("diff.krs", 30);
}

#[test]
fn floats_are_read_as_rust_reads_them() {
    // big integers in the interpreter of a debug build need more stack than WebAssembly's 4 MiB
    if cfg!(target_family = "wasm") {
        return;
    }
    check("decimal.krs", 300);
}
