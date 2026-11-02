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

/// What a WASI program wrote, the files it has open, and the folder its paths start from.
#[derive(Default)]
struct Wasi {
    out: Vec<u8>,
    err: Vec<u8>,
    root: PathBuf,
    files: std::collections::HashMap<i32, std::fs::File>,
    next_fd: i32,
    args: Vec<String>,
    env: Vec<String>,
}

/// WASI's code for an error of the file system.
fn errno(e: &std::io::Error) -> i32 {
    match e.kind() {
        std::io::ErrorKind::NotFound => 44,
        std::io::ErrorKind::PermissionDenied => 2,
        std::io::ErrorKind::AlreadyExists => 20,
        _ => 29,
    }
}

type Caller<'a> = wasmi::Caller<'a, Wasi>;

fn memory(c: &Caller<'_>) -> wasmi::Memory {
    c.get_export("memory").and_then(|e| e.into_memory()).expect("the module's memory")
}

fn mem_read(c: &Caller<'_>, at: i32, n: i32) -> Vec<u8> {
    let mut b = vec![0u8; n as u32 as usize];
    memory(c).read(c, at as u32 as usize, &mut b).expect("memory to read");
    b
}

fn mem_write(c: &mut Caller<'_>, at: i32, b: &[u8]) {
    let m = memory(c);
    m.write(c, at as u32 as usize, b).expect("memory to write");
}

fn u32_at(c: &Caller<'_>, at: i32) -> i32 {
    i32::from_le_bytes(mem_read(c, at, 4).try_into().unwrap())
}

fn wasi_path(c: &Caller<'_>, at: i32, n: i32) -> PathBuf {
    c.data().root.join(String::from_utf8(mem_read(c, at, n)).expect("a UTF-8 path"))
}

/// C strings for args_get and environ_get: (count, total bytes, the strings).
fn c_strings(list: &[String]) -> (i32, i32, Vec<Vec<u8>>) {
    let v: Vec<Vec<u8>> = list.iter().map(|s| [s.as_bytes(), &[0u8][..]].concat()).collect();
    (v.len() as i32, v.iter().map(|x| x.len() as i32).sum(), v)
}

fn put_c_strings(c: &mut Caller<'_>, list: &[String], ptrs: i32, buf: i32) {
    let (_, _, v) = c_strings(list);
    let mut at = buf;
    for (i, s) in v.iter().enumerate() {
        mem_write(c, ptrs + 4 * i as i32, &at.to_le_bytes());
        mem_write(c, at, s);
        at += s.len() as i32;
    }
}

/// The environment of the module: the process's, with the variables of `extra` set (or replaced).
fn wasi_environ(extra: &[(&str, &str)]) -> Vec<String> {
    let mut env: Vec<String> = std::env::vars().filter(|(k, _)| !extra.iter().any(|(x, _)| x == k)).map(|(k, v)| format!("{k}={v}")).collect();
    env.extend(extra.iter().map(|(k, v)| format!("{k}={v}")));
    env
}

/// Runs a module's `_start` with the WASI functions it imports: a minimal WASI, with the program's
/// name as its only argument, the environment, and the files under the crate's folder (as
/// `wasmtime --dir=.` gives them). The result: (stdout, stderr, exit code).
fn run_wasm(bytes: &[u8]) -> (String, String, i32) {
    run_wasm_with(bytes, vec!["program.wasm".to_string()])
}

/// The same with the program's arguments (the first one its name).
fn run_wasm_with(bytes: &[u8], args: Vec<String>) -> (String, String, i32) {
    run_wasm_env(bytes, args, &[])
}

/// The same with variables set in its environment.
fn run_wasm_env(bytes: &[u8], args: Vec<String>, extra: &[(&str, &str)]) -> (String, String, i32) {
    use std::io::{Read, Write};
    use wasmi::{Config, Engine, Linker, Module, Store};
    // deep recursion, as in wasmtime or a browser (wasmi stops at 1000 nested calls by default)
    let mut config = Config::default();
    config.set_max_recursion_depth(100_000);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, bytes).expect("a valid WebAssembly module");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let mut store = Store::new(&engine, Wasi { root, next_fd: 4, args, env: wasi_environ(extra), ..Wasi::default() });
    let mut linker = <Linker<Wasi>>::new(&engine);
    let w = "wasi_snapshot_preview1";
    linker
        .func_wrap(w, "fd_write", |mut c: Caller<'_>, fd: i32, iovs: i32, n: i32, nw: i32| -> i32 {
            let mut total = 0usize;
            for k in 0..n {
                let ptr = u32_at(&c, iovs + 8 * k);
                let len = u32_at(&c, iovs + 8 * k + 4);
                let bytes = mem_read(&c, ptr, len);
                total += bytes.len();
                match fd {
                    1 => c.data_mut().out.extend_from_slice(&bytes),
                    2 => c.data_mut().err.extend_from_slice(&bytes),
                    _ => match c.data_mut().files.get_mut(&fd) {
                        Some(f) => {
                            if let Err(e) = f.write_all(&bytes) {
                                return errno(&e);
                            }
                        }
                        None => return 8,
                    },
                }
            }
            mem_write(&mut c, nw, &(total as u32).to_le_bytes());
            0
        })
        .unwrap();
    linker.func_wrap(w, "proc_exit", |_: Caller<'_>, code: i32| -> Result<(), wasmi::Error> { Err(wasmi::Error::i32_exit(code)) }).unwrap();
    linker
        .func_wrap(w, "args_sizes_get", |mut c: Caller<'_>, count: i32, size: i32| -> i32 {
            let args = c.data().args.clone();
            let (n, total, _) = c_strings(&args);
            mem_write(&mut c, count, &n.to_le_bytes());
            mem_write(&mut c, size, &total.to_le_bytes());
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "args_get", |mut c: Caller<'_>, ptrs: i32, buf: i32| -> i32 {
            let args = c.data().args.clone();
            put_c_strings(&mut c, &args, ptrs, buf);
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "environ_sizes_get", |mut c: Caller<'_>, count: i32, size: i32| -> i32 {
            let env = c.data().env.clone();
            let (n, total, _) = c_strings(&env);
            mem_write(&mut c, count, &n.to_le_bytes());
            mem_write(&mut c, size, &total.to_le_bytes());
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "environ_get", |mut c: Caller<'_>, ptrs: i32, buf: i32| -> i32 {
            let env = c.data().env.clone();
            put_c_strings(&mut c, &env, ptrs, buf);
            0
        })
        .unwrap();
    linker
        .func_wrap(
            w,
            "path_open",
            |mut c: Caller<'_>, dir: i32, _flags: i32, at: i32, n: i32, oflags: i32, _rights: i64, _inherit: i64, _fdflags: i32, fd_out: i32| -> i32 {
                if dir != 3 {
                    return 8;
                }
                let p = wasi_path(&c, at, n);
                // 1: create, 8: truncate
                let r = if oflags & 1 != 0 {
                    std::fs::OpenOptions::new().write(true).create(true).truncate(oflags & 8 != 0).open(&p)
                } else {
                    std::fs::File::open(&p)
                };
                match r {
                    Ok(f) => {
                        let fd = c.data().next_fd;
                        c.data_mut().next_fd += 1;
                        c.data_mut().files.insert(fd, f);
                        mem_write(&mut c, fd_out, &fd.to_le_bytes());
                        0
                    }
                    Err(e) => errno(&e),
                }
            },
        )
        .unwrap();
    linker
        .func_wrap(w, "fd_read", |mut c: Caller<'_>, fd: i32, iovs: i32, n: i32, nr: i32| -> i32 {
            let mut total = 0usize;
            for k in 0..n {
                let ptr = u32_at(&c, iovs + 8 * k);
                let len = u32_at(&c, iovs + 8 * k + 4);
                let mut buf = vec![0u8; len as u32 as usize];
                let got = match c.data_mut().files.get_mut(&fd) {
                    Some(f) => match f.read(&mut buf) {
                        Ok(g) => g,
                        Err(e) => return errno(&e),
                    },
                    None => return 8,
                };
                mem_write(&mut c, ptr, &buf[..got]);
                total += got;
                if got < buf.len() {
                    break;
                }
            }
            mem_write(&mut c, nr, &(total as u32).to_le_bytes());
            0
        })
        .unwrap();
    linker.func_wrap(w, "fd_close", |mut c: Caller<'_>, fd: i32| -> i32 { if c.data_mut().files.remove(&fd).is_some() { 0 } else { 8 } }).unwrap();
    linker
        .func_wrap(w, "path_create_directory", |c: Caller<'_>, dir: i32, at: i32, n: i32| -> i32 {
            if dir != 3 {
                return 8;
            }
            match std::fs::create_dir(wasi_path(&c, at, n)) {
                Ok(()) => 0,
                Err(e) => errno(&e),
            }
        })
        .unwrap();
    linker
        .func_wrap(w, "path_filestat_get", |mut c: Caller<'_>, dir: i32, _flags: i32, at: i32, n: i32, buf: i32| -> i32 {
            if dir != 3 {
                return 8;
            }
            match std::fs::metadata(wasi_path(&c, at, n)) {
                Ok(m) => {
                    // filestat: the file type at 16 (3 a folder, 4 a file), the size at 32
                    let mut st = [0u8; 64];
                    st[16] = if m.is_dir() { 3 } else { 4 };
                    st[32..40].copy_from_slice(&m.len().to_le_bytes());
                    mem_write(&mut c, buf, &st);
                    0
                }
                Err(e) => errno(&e),
            }
        })
        .unwrap();
    linker
        .func_wrap(w, "clock_time_get", |mut c: Caller<'_>, _id: i32, _precision: i64, at: i32| -> i32 {
            let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
            mem_write(&mut c, at, &ns.to_le_bytes());
            0
        })
        .unwrap();
    let instance = linker.instantiate_and_start(&mut store, &module).expect("instantiation");
    let start = instance.get_typed_func::<(), ()>(&store, "_start").expect("`_start`");
    let code = match start.call(&mut store, ()) {
        Ok(()) => 0,
        Err(e) => e.i32_exit_status().unwrap_or_else(|| panic!("the module trapped: {e}")),
    };
    if std::env::var("KAIROS_WASM_PAGES").is_ok() {
        let mem = instance.get_memory(&store, "memory").expect("the module's memory");
        eprintln!("memory: {} pages", mem.size(&store));
    }
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

/// What the interpreter does with a program that uses `b3` (the bridge in Rust), its ledger and
/// its registry of trials in `results`: (stdout, the rendered error, exit code).
fn run_interp_b3(path: &str, src: &str, results: &Path, name: &str) -> (String, String, i32) {
    let buf = Rc::new(RefCell::new(Vec::new()));
    let host = kairos::bridge::Host::new().with_results(results);
    host.set_program(name);
    let mut s = Session::with_host(Box::new(Buf(buf.clone())), host);
    s.mark_imported(path);
    let err = match s.run(path, src) {
        Ok(_) => String::new(),
        Err(d) => s.render(&d),
    };
    let code = if err.is_empty() { 0 } else { 1 };
    (String::from_utf8(buf.borrow().clone()).unwrap(), err, code)
}

/// M14: `b3` in WebAssembly. The bridge written in Kairós (compiler/rt_b3.krs, over kairos/lab)
/// is compiled with each program that uses `b3`: examples/momentum.krs and the programs of
/// tests/wasm_b3, run in wasmi, print what the interpreter (with the bridge in Rust) prints, stop
/// with the same errors, and leave the same registry of trials and the same holdout ledger, each
/// in a folder of its own (data.krs under the fence of a pending protocol). Skipped without the
/// B3 data and in a debug build (slow): `cargo test --release -p kairos --test wasm b3`.
#[test]
fn b3_programs_compiled_to_webassembly_run_like_the_interpreter() {
    if cfg!(target_family = "wasm") || cfg!(debug_assertions) {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("..").join("data").join("processed").join("panel.bin").is_file() {
            eprintln!("skipped: no B3 data");
            return;
        }
        // relative to the crate's folder, where the module's paths start (as `wasmtime --dir=.`)
        let tmp = "tests/wasm/tmp/b3";
        let _ = std::fs::remove_dir_all(root.join(tmp));
        let progs = ["examples/momentum.krs", "tests/wasm_b3/data.krs", "tests/wasm_b3/err_month.krs", "tests/wasm_b3/err_weights.krs"];
        let paths: Vec<String> = progs.iter().map(|p| root.join(p).to_string_lossy().into_owned()).collect();
        let out = root.join(tmp).join("modules");
        let mut args = vec![root.join("std").to_string_lossy().into_owned(), out.to_string_lossy().into_owned()];
        args.extend(paths.iter().cloned());
        assert_eq!(run_compiler("kw.krs", args), "", "the programs that use b3 compile to WebAssembly");
        let ledger = "kind,protocol,sha256,holdout_from,data_until,date,note\nprotocol,teste,0123abcd,2020-01,,2026-01-01,a protocol of the tests\n";
        let note = "note: this program sees the data up to 2019-12: the months from 2020-01 are the holdout of protocol \"teste\" (registered on 2026-01-01), which opens only through `open_once`\n";
        for path in &paths {
            let name = Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
            let fenced = name == "data";
            let (wdir, idir) = (format!("{tmp}/{name}-wasm"), format!("{tmp}/{name}-interp"));
            for d in [&wdir, &idir] {
                std::fs::create_dir_all(root.join(d)).unwrap();
                if fenced {
                    std::fs::write(root.join(d).join("holdout_ledger.csv"), ledger).unwrap();
                }
            }
            let src = std::fs::read_to_string(path).unwrap();
            let mut interp = run_interp_b3(path, &src, &root.join(&idir), &name);
            // the interpreter writes the fence's note to the process's standard error
            if fenced {
                interp.1 = format!("{note}{}", interp.1);
            }
            let wasm = std::fs::read(out.join(format!("{name}.wasm"))).unwrap();
            let t = std::time::Instant::now();
            let got = run_wasm_env(&wasm, vec![format!("{name}.wasm")], &[("KAIROS_RESULTS", wdir.as_str())]);
            assert_eq!(got, interp, "{path}: WebAssembly and the interpreter differ");
            for f in ["registro_testes.csv", "holdout_ledger.csv"] {
                let a = std::fs::read(root.join(&wdir).join(f)).ok();
                let b = std::fs::read(root.join(&idir).join(f)).ok();
                assert!(a == b, "{path}: {f} differs between WebAssembly and the interpreter");
            }
            println!("{name}: the same in WebAssembly ({:.1} s in wasmi)", t.elapsed().as_secs_f64());
        }
        let _ = std::fs::remove_dir_all(root.join(tmp));
    })
}

/// Runs the module named by KAIROS_WASM (with the arguments in KAIROS_WASM_ARGS, separated by
/// spaces), for looking at a module by hand:
/// `KAIROS_WASM=m.wasm cargo test --release --test wasm run_a_module -- --ignored --nocapture`
#[test]
#[ignore]
fn run_a_module() {
    let path = std::env::var("KAIROS_WASM").expect("KAIROS_WASM: the module to run");
    let mut args = vec![path.clone()];
    args.extend(std::env::var("KAIROS_WASM_ARGS").unwrap_or_default().split_whitespace().map(String::from));
    let bytes = std::fs::read(&path).unwrap();
    let t = std::time::Instant::now();
    let (out, err, code) = run_wasm_with(&bytes, args);
    print!("{out}");
    eprint!("{err}");
    eprintln!("exit {code} in {:?}", t.elapsed());
}

/// The compiler written in Kairós, compiled to WebAssembly, compiles itself: kw.krs run by the
/// interpreter makes kw.wasm; kw.wasm, run in wasmi, makes the same module again, byte for byte
/// (and the same kc.wasm and types.wasm). Slow in a debug build:
/// `cargo test --release -p kairos --test wasm -- --ignored the_compiler`
#[test]
#[ignore]
fn the_compiler_in_webassembly_compiles_itself() {
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let progs = ["compiler/kw.krs", "compiler/kc.krs", "compiler/types.krs"];
        let first = std::env::temp_dir().join(format!("kairos-self-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&first);
        // the same relative paths both times: the paths are in the messages each module has
        assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), root.canonicalize().unwrap(), "run from the crate's folder");
        let mut args = vec!["std".to_string(), first.to_string_lossy().into_owned()];
        args.extend(progs.iter().map(|p| p.to_string()));
        assert_eq!(run_compiler("kw.krs", args), "", "the compiler compiles to WebAssembly");
        // the module of the compiler compiles them again (paths relative to the crate, as WASI has it)
        let kw = std::fs::read(first.join("kw.wasm")).unwrap();
        let mut wargs = vec!["kw.wasm".to_string(), "std".to_string(), "tests/wasm/tmp/self".to_string()];
        wargs.extend(progs.iter().map(|p| p.to_string()));
        let (out, err, code) = run_wasm_with(&kw, wargs);
        assert_eq!((out.as_str(), err.as_str(), code), ("", "", 0), "kw.wasm compiles the compiler");
        for p in progs {
            let name = Path::new(p).file_stem().unwrap().to_string_lossy().into_owned();
            let a = std::fs::read(first.join(format!("{name}.wasm"))).unwrap();
            let b = std::fs::read(root.join("tests/wasm/tmp/self").join(format!("{name}.wasm"))).unwrap();
            assert!(a == b, "{name}.wasm: the module made by kw.wasm differs from the one made by the interpreter");
            println!("{name}.wasm: {} bytes, the same from the interpreter and from kw.wasm", a.len());
        }
        let _ = std::fs::remove_dir_all(&first);
    })
}
