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

fn wasi_args() -> Vec<String> {
    vec!["program.wasm".to_string()]
}

fn wasi_environ() -> Vec<String> {
    std::env::vars().map(|(k, v)| format!("{k}={v}")).collect()
}

/// Runs a module's `_start` with the WASI functions it imports: a minimal WASI, with the program's
/// name as its only argument, the environment, and the files under the crate's folder (as
/// `wasmtime --dir=.` gives them). The result: (stdout, stderr, exit code).
fn run_wasm(bytes: &[u8]) -> (String, String, i32) {
    use std::io::{Read, Write};
    use wasmi::{Config, Engine, Linker, Module, Store};
    // deep recursion, as in wasmtime or a browser (wasmi stops at 1000 nested calls by default)
    let mut config = Config::default();
    config.set_max_recursion_depth(100_000);
    let engine = Engine::new(&config);
    let module = Module::new(&engine, bytes).expect("a valid WebAssembly module");
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    let mut store = Store::new(&engine, Wasi { root, next_fd: 4, ..Wasi::default() });
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
            let (n, total, _) = c_strings(&wasi_args());
            mem_write(&mut c, count, &n.to_le_bytes());
            mem_write(&mut c, size, &total.to_le_bytes());
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "args_get", |mut c: Caller<'_>, ptrs: i32, buf: i32| -> i32 {
            put_c_strings(&mut c, &wasi_args(), ptrs, buf);
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "environ_sizes_get", |mut c: Caller<'_>, count: i32, size: i32| -> i32 {
            let (n, total, _) = c_strings(&wasi_environ());
            mem_write(&mut c, count, &n.to_le_bytes());
            mem_write(&mut c, size, &total.to_le_bytes());
            0
        })
        .unwrap();
    linker
        .func_wrap(w, "environ_get", |mut c: Caller<'_>, ptrs: i32, buf: i32| -> i32 {
            put_c_strings(&mut c, &wasi_environ(), ptrs, buf);
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
