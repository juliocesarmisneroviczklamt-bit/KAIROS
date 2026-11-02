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

thread_local! {
    /// the pages of memory the last module run (in this thread) ended with
    static LAST_PAGES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

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
    dirs: std::collections::HashMap<i32, PathBuf>,
    next_fd: i32,
    args: Vec<String>,
    env: Vec<String>,
    /// the standard input, and how much of it was read
    input: Vec<u8>,
    input_at: usize,
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
    run_wasm_io(bytes, args, extra, b"")
}

/// The same with a standard input.
fn run_wasm_io(bytes: &[u8], args: Vec<String>, extra: &[(&str, &str)], input: &[u8]) -> (String, String, i32) {
    use std::io::{Read, Write};
    use wasmi::{Config, Engine, Linker, Module, Store};
    // deep recursion, as in wasmtime or a browser (wasmi stops at 1000 nested calls by default)
    let mut config = Config::default();
    config.set_max_recursion_depth(100_000);
    // and a deep value stack: an interpreter running as WebAssembly recurses as deep as its programs
    config.set_max_stack_height(1 << 30);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, bytes).expect("a valid WebAssembly module");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let mut store = Store::new(&engine, Wasi { root, next_fd: 4, args, env: wasi_environ(extra), input: input.to_vec(), ..Wasi::default() });
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
            |mut c: Caller<'_>, dir: i32, _flags: i32, at: i32, n: i32, oflags: i32, _rights: i64, _inherit: i64, fdflags: i32, fd_out: i32| -> i32 {
                if dir != 3 {
                    return 8;
                }
                let p = wasi_path(&c, at, n);
                // 2: a folder, to read with fd_readdir
                if oflags & 2 != 0 {
                    return match std::fs::metadata(&p) {
                        Ok(m) if m.is_dir() => {
                            let fd = c.data().next_fd;
                            c.data_mut().next_fd += 1;
                            c.data_mut().dirs.insert(fd, p);
                            mem_write(&mut c, fd_out, &fd.to_le_bytes());
                            0
                        }
                        Ok(_) => 54,
                        Err(e) => errno(&e),
                    };
                }
                // 1: create, 8: truncate; fdflags 1: append
                let r = if oflags & 1 != 0 {
                    std::fs::OpenOptions::new().write(true).create(true).truncate(oflags & 8 != 0).append(fdflags & 1 != 0).open(&p)
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
                let got = if fd == 0 {
                    let w = c.data_mut();
                    let g = (w.input.len() - w.input_at).min(buf.len());
                    buf[..g].copy_from_slice(&w.input[w.input_at..w.input_at + g]);
                    w.input_at += g;
                    g
                } else {
                    match c.data_mut().files.get_mut(&fd) {
                        Some(f) => match f.read(&mut buf) {
                            Ok(g) => g,
                            Err(e) => return errno(&e),
                        },
                        None => return 8,
                    }
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
    linker
        .func_wrap(w, "fd_close", |mut c: Caller<'_>, fd: i32| -> i32 {
            if c.data_mut().files.remove(&fd).is_some() || c.data_mut().dirs.remove(&fd).is_some() { 0 } else { 8 }
        })
        .unwrap();
    linker
        .func_wrap(w, "fd_readdir", |mut c: Caller<'_>, fd: i32, buf: i32, len: i32, cookie: i64, used: i32| -> i32 {
            let Some(dir) = c.data().dirs.get(&fd).cloned() else { return 8 };
            let mut names: Vec<String> = match std::fs::read_dir(&dir) {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect(),
                Err(e) => return errno(&e),
            };
            names.sort();
            // entries from the cookie on: next cookie, inode, the name's length, type, 3 more, the
            // name; the last one may be cut at the end of the buffer
            let mut out = Vec::new();
            for (k, name) in names.iter().enumerate().skip(cookie.max(0) as usize) {
                out.extend_from_slice(&(k as u64 + 1).to_le_bytes());
                out.extend_from_slice(&0u64.to_le_bytes());
                out.extend_from_slice(&(name.len() as u32).to_le_bytes());
                out.extend_from_slice(&[0, 0, 0, 0]);
                out.extend_from_slice(name.as_bytes());
                if out.len() >= len as usize {
                    break;
                }
            }
            out.truncate(len as usize);
            mem_write(&mut c, buf, &out);
            mem_write(&mut c, used, &(out.len() as u32).to_le_bytes());
            0
        })
        .unwrap();
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
                    // the time it last changed (ns)
                    let ns = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as u64);
                    st[48..56].copy_from_slice(&ns.to_le_bytes());
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
    let pages = instance.get_memory(&store, "memory").expect("the module's memory").size(&store);
    LAST_PAGES.with(|p| p.set(pages));
    if std::env::var("KAIROS_WASM_PAGES").is_ok() {
        eprintln!("memory: {pages} pages");
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
        // where the programs write their files (ignored by git, so not in a fresh checkout)
        std::fs::create_dir_all(dir.join("tmp")).unwrap();
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

/// What the interpreter (the `kairos` command, with the bridge in Rust) does with a program that
/// uses `b3`, its ledger and its registry of trials in `results` (relative to the crate's folder):
/// (stdout, stderr, exit code). The fence's note goes to stderr, as the module's does.
fn run_kairos_b3(path: &str, results: &str) -> (String, String, i32) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_kairos"))
        .args(["run", path])
        .env("KAIROS_RESULTS", results)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("the kairos command");
    (String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap(), out.status.code().unwrap_or(-1))
}

/// M14: `b3` in WebAssembly. The bridge written in Kairós (compiler/rt_b3.krs, over kairos/lab)
/// is compiled with each program that uses `b3`: examples/momentum.krs and the programs of
/// tests/wasm_b3, run in wasmi, print what the `kairos` command (with the bridge in Rust) prints,
/// with the same notes and errors on stderr and the same exit code, and leave the same registry of
/// trials and the same holdout ledger, each in a folder of its own: the data under the fence of a
/// pending protocol, protocols and their holdouts (opened once, an error inside spends them).
/// Skipped without the B3 data and in a debug build (slow): `cargo test --release -p kairos
/// --test wasm b3`.
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
        let progs = [
            "examples/momentum.krs",
            "tests/wasm_b3/data.krs",
            "tests/wasm_b3/more.krs",
            "tests/wasm_b3/daily.krs",
            "tests/wasm_b3/protocol.krs",
            "tests/wasm_b3/err_month.krs",
            "tests/wasm_b3/err_weights.krs",
            "tests/wasm_b3/err_open_fail.krs",
            "tests/wasm_b3/err_open_eval.krs",
            "tests/wasm_b3/err_unripe.krs",
            "tests/wasm_b3/err_twice.krs",
        ];
        let paths: Vec<String> = progs.iter().map(|p| root.join(p).to_string_lossy().into_owned()).collect();
        let out = root.join(tmp).join("modules");
        let mut args = vec![root.join("std").to_string_lossy().into_owned(), out.to_string_lossy().into_owned()];
        args.extend(paths.iter().cloned());
        assert_eq!(run_compiler("kw.krs", args), "", "the programs that use b3 compile to WebAssembly");
        let ledger = "kind,protocol,sha256,holdout_from,data_until,date,note
protocol,teste,0123abcd,2020-01,,2026-01-01,a protocol of the tests
";
        for path in &paths {
            let name = Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
            let (wdir, idir) = (format!("{tmp}/{name}-wasm"), format!("{tmp}/{name}-interp"));
            for d in [&wdir, &idir] {
                std::fs::create_dir_all(root.join(d)).unwrap();
                if name == "data" {
                    std::fs::write(root.join(d).join("holdout_ledger.csv"), ledger).unwrap();
                }
            }
            let interp = run_kairos_b3(path, &idir);
            let wasm = std::fs::read(out.join(format!("{name}.wasm"))).unwrap();
            let t = std::time::Instant::now();
            let got = run_wasm_env(&wasm, vec![format!("{name}.wasm")], &[("KAIROS_RESULTS", wdir.as_str())]);
            assert_eq!(got, interp, "{path}: WebAssembly and the interpreter differ");
            for f in ["registro_testes.csv", "holdout_ledger.csv"] {
                let a = std::fs::read(root.join(&wdir).join(f)).ok();
                let b = std::fs::read(root.join(&idir).join(f)).ok();
                assert!(a == b, "{path}: {f} differs between WebAssembly and the interpreter");
            }
            let pages = LAST_PAGES.with(|p| p.get());
            // M17: the allocator joins the free blocks that touch and reuses them for any size
            // (momentum took 25,043 pages before, about 8,700 since)
            if name == "momentum" {
                assert!(pages < 12_000, "momentum.wasm ended with {pages} pages of memory");
            }
            println!("{name}: the same in WebAssembly ({:.1} s in wasmi, {} MB)", t.elapsed().as_secs_f64(), pages / 16);
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

/// M15: the interpreter written in Kairós (compiler/ki.krs), compiled to WebAssembly by the
/// compiler written in Kairós, runs the test programs as the Rust interpreter does: the same
/// output, the same errors (of the checker too) and the same exit code. What it does not run yet
/// stops with K0001 and is counted apart. Only in a release build (compiling the whole interpreter
/// in a debug one takes too long; the CI runs it in its `selfhost` job).
#[test]
fn the_interpreter_written_in_kairos_runs_like_the_rust_one() {
    if cfg!(target_family = "wasm") || cfg!(debug_assertions) {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), root.canonicalize().unwrap(), "run from the crate's folder");
        let out = std::env::temp_dir().join(format!("kairos-ki-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let report = run_compiler("kw.krs", vec!["std".to_string(), out.to_string_lossy().into_owned(), "compiler/ki.krs".to_string()]);
        assert_eq!(report, "", "the interpreter compiles to WebAssembly");
        let ki = std::fs::read(out.join("ki.wasm")).unwrap();
        let _ = std::fs::remove_dir_all(&out);
        std::fs::create_dir_all(root.join("tests/wasm/tmp")).unwrap();
        let mut files: Vec<String> = Vec::new();
        for dir in ["tests/wasm", "tests/programs", "tests/temporal"] {
            let mut fs: Vec<String> = std::fs::read_dir(root.join(dir))
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.file_name().to_string_lossy().into_owned()))
                .filter(|n| n.ends_with(".krs"))
                .map(|n| format!("{dir}/{n}"))
                .collect();
            fs.sort();
            files.extend(fs);
        }
        let (mut same, mut later) = (0, 0);
        for path in &files {
            let src = std::fs::read_to_string(root.join(path)).unwrap();
            // what reads the lab's data or the network is left to the other tests
            if src.contains("b3.") || src.contains("http_get") || src.contains("download(") {
                continue;
            }
            let k = run_wasm_with(&ki, vec!["ki.wasm".to_string(), "std".to_string(), path.clone()]);
            if k.1.contains("[K0001]") {
                later += 1;
                continue;
            }
            let r = run_interp(path, &src);
            // the text of a system's error (`cannot read p: ...`) is the host's: WASI writes POSIX's,
            // Windows its own (in its own language)
            let os = |s: &str| -> String {
                s.lines()
                    .map(|l| match l.find("]: cannot ") {
                        Some(i) if l.starts_with("runtime error[R0011]") => match l[i..].find(": ").and_then(|j| l[i + j + 2..].find(": ").map(|q| i + j + 2 + q)) {
                            Some(cut) => format!("{}: <the system's error>", &l[..cut]),
                            None => l.to_string(),
                        },
                        _ => l.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            };
            assert_eq!((k.0, os(&k.1), k.2), (r.0, os(&r.1), r.2), "{path}: the interpreter written in Kairós and the Rust one differ");
            same += 1;
        }
        assert!(same >= 110, "{same} programs run the same, {later} not yet");
        println!("{same} programs run the same in the interpreter written in Kairós; {later} use what it does not run yet");
    })
}

/// M15: the interpreter written in Kairós, compiled to WebAssembly, runs programs that use `b3`
/// (the lab's data, protocols, holdouts) as the `kairos` command does: the same output, notes and
/// errors, the same registry of trials and ledger. Skipped without the B3 data and in a debug
/// build (slow): `cargo test --release -p kairos --test wasm the_interpreter_written_in_kairos_runs_b3`.
#[test]
fn the_interpreter_written_in_kairos_runs_b3_programs() {
    if cfg!(target_family = "wasm") || cfg!(debug_assertions) {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("..").join("data").join("processed").join("panel.bin").is_file() {
            eprintln!("skipped: no B3 data");
            return;
        }
        let tmp = "tests/wasm/tmp/kib3";
        let _ = std::fs::remove_dir_all(root.join(tmp));
        let out = root.join(tmp).join("modules");
        let report = run_compiler("kw.krs", vec!["std".to_string(), out.to_string_lossy().into_owned(), "compiler/ki.krs".to_string()]);
        assert_eq!(report, "", "the interpreter compiles to WebAssembly");
        let ki = std::fs::read(out.join("ki.wasm")).unwrap();
        let progs = ["examples/momentum.krs", "tests/wasm_b3/protocol.krs", "tests/wasm_b3/err_open_fail.krs", "tests/wasm_b3/err_weights.krs"];
        for path in progs {
            let name = Path::new(path).file_stem().unwrap().to_string_lossy().into_owned();
            let (wdir, idir) = (format!("{tmp}/{name}-ki"), format!("{tmp}/{name}-interp"));
            for d in [&wdir, &idir] {
                std::fs::create_dir_all(root.join(d)).unwrap();
            }
            let interp = run_kairos_b3(path, &idir);
            let t = std::time::Instant::now();
            let got = run_wasm_env(&ki, vec!["ki.wasm".to_string(), "std".to_string(), path.to_string()], &[("KAIROS_RESULTS", wdir.as_str())]);
            assert_eq!(got, interp, "{path}: the interpreter written in Kairós and the Rust one differ");
            for f in ["registro_testes.csv", "holdout_ledger.csv"] {
                let a = std::fs::read(root.join(&wdir).join(f)).ok();
                let b = std::fs::read(root.join(&idir).join(f)).ok();
                assert!(a == b, "{path}: {f} differs between the interpreter written in Kairós and the Rust one");
            }
            println!("{name}: the same in the interpreter written in Kairós ({:.1} s in wasmi)", t.elapsed().as_secs_f64());
        }
        let _ = std::fs::remove_dir_all(root.join(tmp));
    })
}

/// M16: compiler/std_embed.krs (the standard library and the version the command written in
/// Kairós carries) is what compiler/gen_std.krs makes from kairos/std and Cargo.toml now: run it
/// again after changing either.
#[test]
fn the_standard_library_the_command_in_kairos_carries_is_up_to_date() {
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let out = std::env::temp_dir().join(format!("kairos-std-embed-{}.krs", std::process::id()));
        let report = run_compiler("gen_std.krs", vec![root.to_string_lossy().into_owned(), out.to_string_lossy().into_owned()]);
        assert!(report.starts_with("std_embed.krs: "), "{report}");
        let made = std::fs::read_to_string(&out).unwrap();
        let _ = std::fs::remove_file(&out);
        let kept = std::fs::read_to_string(root.join("compiler").join("std_embed.krs")).unwrap();
        // (a checkout may keep the file with CR LF)
        assert!(made.replace("\r\n", "\n") == kept.replace("\r\n", "\n"), "compiler/std_embed.krs is stale: `kairos run kairos/compiler/gen_std.krs kairos`");
    })
}

/// M16: the `kairos` command written in Kairós (compiler/kairos.krs), compiled to WebAssembly by
/// the compiler written in Kairós, does what the Rust command does: each subcommand gives the same
/// output, the same errors and the same exit code, and a REPL session given on its standard input
/// goes the same way. Only in a release build, as the test above.
#[test]
fn the_command_written_in_kairos_runs_like_the_rust_one() {
    if cfg!(target_family = "wasm") || cfg!(debug_assertions) {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), root.canonicalize().unwrap(), "run from the crate's folder");
        let out = std::env::temp_dir().join(format!("kairos-cmd-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let report = run_compiler("kw.krs", vec!["std".to_string(), out.to_string_lossy().into_owned(), "compiler/kairos.krs".to_string()]);
        assert_eq!(report, "", "the command compiles to WebAssembly");
        let cmd = std::fs::read(out.join("kairos.wasm")).unwrap();
        let _ = std::fs::remove_dir_all(&out);
        // what the programs write goes to a folder of their own (no registry, no ledger of the lab)
        let tmp = root.join("tests/wasm/tmp/cmd");
        std::fs::create_dir_all(&tmp).unwrap();
        let results = tmp.to_string_lossy().into_owned();
        std::fs::write(tmp.join("show_args.krs"), "print(args())\nprint(len(args()))\n").unwrap();
        std::fs::write(tmp.join("lib.krs"), "fn twice(n: i64) -> i64 { 2 * n }\nprint(\"lib loaded\")\n").unwrap();
        // images for `exec`, written by the compiler in Kairós, and two that are not right
        let kim = tmp.join("kim");
        let progs = ["basics", "closures", "grad", "rt_div"];
        let mut kc = vec!["std".to_string(), kim.to_string_lossy().into_owned()];
        kc.extend(progs.iter().map(|p| format!("tests/programs/{p}.krs")));
        assert_eq!(run_compiler("kc.krs", kc), "", "the images");
        std::fs::write(kim.join("bad_item.kim"), "kairos-image 1\n(main 0 ())\n(bogus x41)\n").unwrap();
        std::fs::write(kim.join("bad_body.kim"), "kairos-image 1\n(fn x66 () i64)\n(body 0 0 () - -)\n").unwrap();
        let rust = |args: &[&str], input: &str| -> (String, String, i32) {
            let mut p = std::process::Command::new(env!("CARGO_BIN_EXE_kairos"))
                .args(args)
                .current_dir(root)
                .env("KAIROS_RESULTS", &results)
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .stderr(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            p.stdin.take().unwrap().write_all(input.as_bytes()).unwrap();
            let o = p.wait_with_output().unwrap();
            (String::from_utf8(o.stdout).unwrap(), String::from_utf8(o.stderr).unwrap(), o.status.code().unwrap_or(-1))
        };
        // the text of a system's error is the host's (WASI writes POSIX's, Windows its own)
        let os = |s: &str| -> String {
            s.lines()
                .map(|l| match l.find(": ") {
                    Some(i) if l.starts_with("cannot read ") => format!("{}: <the system's error>", &l[..i]),
                    _ => l.to_string(),
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        let words = |s: &str| -> Vec<String> { s.split(' ').map(|w| w.to_string()).collect() };
        let mut lines: Vec<Vec<String>> = ["--version", "-V", "run", "types", "bogus", "check tests/nope.krs", "tests/programs/shebang.krs"].iter().map(|l| words(l)).collect();
        lines.push(vec!["run".into(), "tests/wasm/tmp/cmd/show_args.krs".into(), "a".into(), "b c".into()]);
        lines.push(words("tests/wasm/tmp/cmd/show_args.krs x"));
        for p in progs.iter().chain(["bad_item", "bad_body"].iter()) {
            lines.push(words(&format!("exec tests/wasm/tmp/cmd/kim/{p}.kim a b")));
        }
        lines.push(words("exec tests/programs/basics.krs"));
        lines.push(words("exec"));
        for p in ["basics", "closures", "grad", "err_type", "err_future", "rt_div", "rt_index"] {
            for sub in ["check", "run", "tokens", "ast", "types", "hir"] {
                lines.push(vec![sub.to_string(), format!("tests/programs/{p}.krs")]);
            }
        }
        for line in &lines {
            let mut args = vec!["kairos.wasm".to_string()];
            args.extend(line.iter().cloned());
            let k = run_wasm_io(&cmd, args, &[("KAIROS_RESULTS", &results)], b"");
            let r = rust(&line.iter().map(|a| a.as_str()).collect::<Vec<_>>(), "");
            assert_eq!((k.0, os(&k.1), k.2), (r.0, os(&r.1), r.2), "`kairos {}`: the command written in Kairós and the Rust one differ", line.join(" "));
        }
        let session = "1 + 2\nlet x = 5\nx * 2\n\n  :help\n\"abc\"\n[1, 2, 3]\nprint(\"printed\")\n\
            fn sq(n: i64) -> i64 {\n    n * n\n}\nsq(7)\nlet y: i64 = \"a\"\ny\nvar c = 0\nc += 1\nc\n[1, 2][5]\n\
            struct P { a: i64, b: str }\nP { a: 1, b: \"q\" }\nlet f = |k: i64| k + x\nf(10)\nmean([1.0, 2.0, 4.0])\n\
            import \"tests/wasm/tmp/cmd/lib.krs\"\ntwice(21)\nimport \"tests/wasm/tmp/cmd/nope.krs\"\nimport \"std/json.krs\"\n\
            json_parse(\"[1, 2]\")\nlet = 5\n1 + )\nsq(3)\r\n[\n:q\n]\n\
            let a = [1,\n2,\n";
        for input in [session, "1 + 1\n:quit\n2 + 2\n", ""] {
            let k = run_wasm_io(&cmd, vec!["kairos.wasm".to_string(), "repl".to_string()], &[("KAIROS_RESULTS", &results)], input.as_bytes());
            let r = rust(&["repl"], input);
            assert_eq!(k, r, "a REPL session: the command written in Kairós and the Rust one differ:\n{input}");
        }
        println!("{} command lines and 3 REPL sessions: the same from the command written in Kairós", lines.len());
    })
}
