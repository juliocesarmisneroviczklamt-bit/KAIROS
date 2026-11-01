//! Deep recursion must end in a clean runtime error (R0005), never in a crash, whatever the
//! stack: 32-bit systems and WebAssembly run the interpreter on much smaller stacks than the
//! 512 MB of a 64-bit desktop.

use kairos::interp::STACK_BYTES;
use kairos::session::Session;
use std::sync::atomic::Ordering;

fn deep(stack: usize) {
    STACK_BYTES.store(stack, Ordering::Relaxed);
    let mut s = Session::new(Box::new(std::io::sink()));
    let run = |s: &mut Session, src: &str| s.run("deep.krs", src).map_err(|e| (e.code, s.render(&e)));
    run(&mut s, "fn depth(n: i64) -> i64 { if n == 0 { 0 } else { 1 + depth(n - 1) } }").expect("definition");
    run(&mut s, "assert(depth(25) == 25)").expect("shallow recursion works on any stack");
    let (code, text) = run(&mut s, "print(depth(1000000000))").expect_err("a billion nested calls cannot fit");
    assert_eq!(code, "R0005", "{text}");
}

#[test]
fn deep_recursion_is_a_clean_error_on_small_stacks() {
    for mb in [8usize, 32] {
        match std::thread::Builder::new().stack_size(mb << 20).spawn(move || deep(mb << 20)) {
            Ok(h) => {
                if let Err(panic) = h.join() {
                    std::panic::resume_unwind(panic);
                }
            }
            // no threads (WebAssembly/WASI): the main thread has the stack set at link time
            Err(_) => {
                deep(64 << 20);
                return;
            }
        }
    }
}
