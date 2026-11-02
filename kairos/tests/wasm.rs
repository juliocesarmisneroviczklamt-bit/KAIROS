//! M13: the WebAssembly back end written in Kairós (kairos/compiler/wasmgen.krs, driven by
//! kw.krs). The programs of tests/wasm compile to WebAssembly modules for WASI; run in a
//! WebAssembly engine (wasmi, an interpreter in pure Rust, so this runs on every target), each
//! must print exactly what the interpreter prints, and fail with the same runtime error, the same
//! message and the same exit code.

use kairos::session::Session;
use std::cell::RefCell;
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

fn on_big_stack(f: fn()) {
    use kairos::interp::STACK_BYTES;
    use std::sync::atomic::Ordering;
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

/// Runs a program of kairos/compiler with arguments: its output.
fn run_compiler(program: &str, args: Vec<String>) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("compiler").join(program);
    let src = std::fs::read_to_string(&path).unwrap();
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::new(Box::new(Buf(buf.clone())));
    s.host().set_args(args);
    if let Err(e) = s.run(&path.to_string_lossy(), &src) {
        panic!("{}", s.render(&e));
    }
    String::from_utf8(buf.borrow().clone()).unwrap()
}

/// What a WASI program wrote and how it ended.
#[derive(Default)]
struct Wasi {
    out: Vec<u8>,
    err: Vec<u8>,
}

/// Runs a module's `_start` with the two WASI calls it imports: (stdout, stderr, exit code).
fn run_wasm(bytes: &[u8]) -> (String, String, i32) {
    use wasmi::{Caller, Config, Engine, Linker, Module, Store};
    // deep recursion, as in wasmtime or a browser (wasmi stops at 1000 nested calls by default)
    let mut config = Config::default();
    config.set_max_recursion_depth(100_000);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, bytes).expect("a valid WebAssembly module");
    let mut store = Store::new(&engine, Wasi::default());
    let mut linker = <Linker<Wasi>>::new(&engine);
    linker
        .func_wrap("wasi_snapshot_preview1", "fd_write", |mut caller: Caller<'_, Wasi>, fd: i32, iovs: i32, n: i32, nw: i32| -> i32 {
            let mem = caller.get_export("memory").and_then(|e| e.into_memory()).expect("the module's memory");
            let mut total = 0usize;
            for k in 0..n as usize {
                let mut iov = [0u8; 8];
                mem.read(&caller, iovs as usize + 8 * k, &mut iov).expect("an iovec");
                let ptr = u32::from_le_bytes(iov[..4].try_into().unwrap()) as usize;
                let len = u32::from_le_bytes(iov[4..].try_into().unwrap()) as usize;
                let mut bytes = vec![0u8; len];
                mem.read(&caller, ptr, &mut bytes).expect("the bytes to write");
                if fd == 1 {
                    caller.data_mut().out.extend_from_slice(&bytes);
                } else {
                    caller.data_mut().err.extend_from_slice(&bytes);
                }
                total += len;
            }
            mem.write(&mut caller, nw as usize, &(total as u32).to_le_bytes()).expect("the count");
            0
        })
        .unwrap();
    linker
        .func_wrap("wasi_snapshot_preview1", "proc_exit", |_: Caller<'_, Wasi>, code: i32| -> Result<(), wasmi::Error> {
            Err(wasmi::Error::i32_exit(code))
        })
        .unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).expect("instantiation");
    let start = instance.get_typed_func::<(), ()>(&store, "_start").expect("`_start`");
    let code = match start.call(&mut store, ()) {
        Ok(()) => 0,
        Err(e) => e.i32_exit_status().unwrap_or_else(|| panic!("the module trapped: {e}")),
    };
    let w = store.data();
    (String::from_utf8(w.out.clone()).unwrap(), String::from_utf8(w.err.clone()).unwrap(), code)
}

/// What the interpreter does with a program: (stdout, the rendered error, exit code).
fn run_interp(path: &str, src: &str) -> (String, String, i32) {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let mut s = Session::new(Box::new(Buf(buf.clone())));
    s.mark_imported(path);
    let err = match s.run(path, src) {
        Ok(_) => String::new(),
        Err(d) => s.render(&d),
    };
    let code = if err.is_empty() { 0 } else { 1 };
    (String::from_utf8(buf.borrow().clone()).unwrap(), err, code)
}

#[test]
fn programs_compiled_to_webassembly_run_like_the_interpreter() {
    // the compiler written in Kairós needs more stack than WebAssembly's 4 MiB in CI
    if cfg!(target_family = "wasm") {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("tests").join("wasm");
        let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "krs"))
            .collect();
        files.sort();
        let out = std::env::temp_dir().join(format!("kairos-wasm-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let mut args = vec![root.join("std").to_string_lossy().into_owned(), out.to_string_lossy().into_owned()];
        args.extend(files.iter().map(|f| f.to_string_lossy().into_owned()));
        let report = run_compiler("kw.krs", args);
        assert_eq!(report, "", "every program of tests/wasm compiles to WebAssembly");
        let mut errors = 0;
        for f in &files {
            let path = f.to_string_lossy().into_owned();
            let src = std::fs::read_to_string(f).unwrap();
            let interp = run_interp(&path, &src);
            // the program does what its comments say
            let expected: Vec<&str> = src.lines().filter_map(|l| l.split_once("// out: ").map(|(_, r)| r.trim_end())).collect();
            let got: Vec<&str> = interp.0.lines().collect();
            assert_eq!(got, expected, "{path}: the interpreter's output");
            let code = src.lines().find_map(|l| l.split_once("// error: ").map(|(_, r)| r.trim().to_string()));
            if let Some(c) = &code {
                assert!(interp.1.contains(&format!("[{c}]")), "{path}: expected {c}, got {}", interp.1);
                errors += 1;
            }
            let name = f.file_stem().unwrap().to_string_lossy().into_owned();
            let wasm = std::fs::read(out.join(format!("{name}.wasm"))).unwrap();
            assert_eq!(run_wasm(&wasm), interp, "{path}: WebAssembly and the interpreter differ");
        }
        let _ = std::fs::remove_dir_all(&out);
        assert!(files.len() >= 10 && errors >= 5, "{} programs, {errors} with errors", files.len());
        println!("{} programs run the same in WebAssembly, {errors} of them stopping with the same runtime error", files.len());
    })
}
