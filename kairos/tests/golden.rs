//! Golden tests: every `tests/programs/*.krs` runs in a fresh session.
//! - `// out: text` lines are the expected output, in order;
//! - `// error: CODE` means the program must fail with that diagnostic code
//!   (compile `E...` or runtime `R...`); output printed before a runtime error is still checked.

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

/// The reference interpreter recurses on the HIR, so it runs on a large stack (as the CLI does).
/// Without threads (WebAssembly/WASI) it runs on the main thread, with the stack the runtime allows.
#[test]
fn golden_programs() {
    use kairos::interp::STACK_BYTES;
    use std::sync::atomic::Ordering;
    let spawned = std::thread::Builder::new().stack_size(256 << 20).spawn(|| {
        STACK_BYTES.store(256 << 20, Ordering::Relaxed);
        run_golden()
    });
    match spawned {
        Ok(h) => {
            if let Err(panic) = h.join() {
                std::panic::resume_unwind(panic);
            }
        }
        Err(_) => {
            // WebAssembly: CI runs wasmtime with -W max-wasm-stack=4194304
            STACK_BYTES.store(4 << 20, Ordering::Relaxed);
            run_golden()
        }
    }
}

fn run_golden() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("programs");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("tests/programs")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "krs"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no golden programs found");
    let mut failures = Vec::new();
    for path in &files {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let src = std::fs::read_to_string(path).unwrap();
        let expected_out: Vec<String> =
            src.lines().filter_map(|l| l.split_once("// out: ").map(|(_, r)| r.trim_end().to_string())).collect();
        let expected_err = src.lines().find_map(|l| l.split_once("// error: ").map(|(_, r)| r.trim().to_string()));
        let buf = Rc::new(RefCell::new(Vec::new()));
        let mut s = Session::new(Box::new(Buf(buf.clone())));
        let res = s.run(&name, &src);
        let out = String::from_utf8(buf.borrow().clone()).unwrap();
        let got: Vec<String> = out.lines().map(str::to_string).collect();
        match (&expected_err, &res) {
            (Some(code), Err(d)) if d.code != code => {
                failures.push(format!("{name}: expected error {code}, got:\n{}", s.render(d)))
            }
            (Some(code), Ok(_)) => failures.push(format!("{name}: expected error {code}, but the program ran")),
            (None, Err(d)) => failures.push(format!("{name}: unexpected error:\n{}", s.render(d))),
            _ => {}
        }
        if got != expected_out {
            failures.push(format!("{name}: output differs\n  expected: {expected_out:?}\n  got:      {got:?}"));
        }
    }
    assert!(failures.is_empty(), "{} of {} golden programs failed:\n\n{}", failures.len(), files.len(), failures.join("\n\n"));
    println!("{} golden programs ok", files.len());
}
