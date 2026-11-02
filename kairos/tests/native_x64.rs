//! M21 (Windows) and M23 (Linux): the native back end written in Kairós (kairos/compiler/native.krs,
//! driven by kn.krs). The WebAssembly module of each program of tests/wasm (made by wasmgen.krs,
//! with the operating system written in Kairós, rt_win.krs or rt_linux.krs) is translated to
//! x86-64 and written as an executable of the system the tests run on, which must print what the
//! interpreter prints, fail with the same runtime error and exit with the same code. And the
//! compiler itself, as an executable, compiles itself to the same bytes. The executables run on
//! Windows and Linux x86-64; the tests run in release (the compiler written in Kairós takes minutes
//! in a debug build).

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

fn runs_here() -> bool {
    (cfg!(windows) || cfg!(target_os = "linux")) && cfg!(target_arch = "x86_64") && !cfg!(debug_assertions)
}

/// An executable's file name: `.exe` on Windows.
fn exe_name(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

/// What kn writes is a plain file: on Linux it is made executable (as `chmod +x`).
fn executable(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut p = std::fs::metadata(path).unwrap().permissions();
        p.set_mode(0o755);
        std::fs::set_permissions(path, p).unwrap();
    }
    let _ = path;
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

/// Runs a program of kairos/compiler (interpreted) with arguments: its output.
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

/// What the interpreter does with a program: (stdout, stderr, exit code).
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

/// What an executable does, run from the crate's folder (where the programs' paths start).
fn run_exe(exe: &Path, args: &[&str], results: &Path) -> (String, String, i32) {
    let o = std::process::Command::new(exe)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("KAIROS_RESULTS", results)
        .output()
        .unwrap();
    (String::from_utf8(o.stdout).unwrap(), String::from_utf8(o.stderr).unwrap(), o.status.code().unwrap_or(-1))
}

fn kairos_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "krs"))
        .collect();
    files.sort();
    files
}

#[test]
fn programs_compiled_to_windows_executables_run_like_the_interpreter() {
    if !runs_here() {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let dir = root.join("tests").join("wasm");
        let files = kairos_files(&dir);
        std::fs::create_dir_all(dir.join("tmp")).unwrap();
        // what a program writes goes to a folder of its own (no registry, no ledger of the lab)
        let results = dir.join("tmp").join("native_results");
        std::fs::create_dir_all(&results).unwrap();
        let out = std::env::temp_dir().join(format!("kairos-native-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let mut args = vec![root.join("std").to_string_lossy().into_owned(), out.to_string_lossy().into_owned()];
        args.extend(files.iter().map(|f| f.to_string_lossy().into_owned()));
        let report = run_compiler("kn.krs", args);
        assert_eq!(report, "", "every program of tests/wasm compiles to an executable");
        let mut errors = 0;
        for f in &files {
            let path = f.to_string_lossy().into_owned();
            let src = std::fs::read_to_string(f).unwrap();
            let interp = run_interp(&path, &src);
            if interp.2 != 0 {
                errors += 1;
            }
            let name = f.file_stem().unwrap().to_string_lossy().into_owned();
            let exe = out.join(exe_name(&name));
            executable(&exe);
            let got = run_exe(&exe, &[], &results);
            assert_eq!(got, interp, "{path}: the executable and the interpreter differ");
        }
        let _ = std::fs::remove_dir_all(&out);
        assert!(files.len() >= 10 && errors >= 5, "{} programs, {errors} with errors", files.len());
        println!("{} programs run the same as native executables, {errors} of them stopping with the same runtime error", files.len());
    })
}

/// kn.krs, interpreted, compiles kn.krs to kn.exe; kn.exe compiles kn.krs to the same bytes. Then
/// the `kairos` command it compiles runs a program as the Rust one does.
#[test]
fn the_native_compiler_compiles_itself() {
    if !runs_here() {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        // paths from the crate's folder (where the tests run): the runtime is found from the std
        // folder, and the command imports parts of it again, which must be the same path
        assert_eq!(std::env::current_dir().unwrap().canonicalize().unwrap(), root.canonicalize().unwrap(), "run from the crate's folder");
        let std_dir = "std".to_string();
        let out = std::env::temp_dir().join(format!("kairos-native-self-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&out);
        let (one, two) = (out.join("one"), out.join("two"));
        let kn = "compiler/kn.krs".to_string();
        let report = run_compiler("kn.krs", vec![std_dir.clone(), one.to_string_lossy().into_owned(), kn.clone()]);
        assert_eq!(report, "", "kn.krs compiles to an executable");
        let results = root.join("tests").join("wasm").join("tmp").join("native_results");
        std::fs::create_dir_all(&results).unwrap();
        let t = std::time::Instant::now();
        let kn_exe = one.join(exe_name("kn"));
        executable(&kn_exe);
        let again = run_exe(&kn_exe, &[&std_dir, &two.to_string_lossy(), &kn, "compiler/kairos.krs"], &results);
        assert_eq!(again, (String::new(), String::new(), 0), "kn compiles kn.krs and kairos.krs");
        let a = std::fs::read(&kn_exe).unwrap();
        let b = std::fs::read(two.join(exe_name("kn"))).unwrap();
        assert!(a == b, "kn compiled by itself is another executable ({} and {} bytes)", a.len(), b.len());
        println!("kn ({} bytes) compiles itself to the same bytes, and the command, in {:.1} s", a.len(), t.elapsed().as_secs_f64());
        // the command written in Kairós, as an executable
        let cmd = two.join(exe_name("kairos"));
        executable(&cmd);
        let program = "tests/programs/basics.krs";
        let src = std::fs::read_to_string(root.join(program)).unwrap();
        assert_eq!(run_exe(&cmd, &["run", program], &results), run_interp(program, &src), "`kairos.exe run {program}`");
        let version = run_exe(&cmd, &["--version"], &results);
        assert_eq!(version.0, format!("kairos {}\n", env!("CARGO_PKG_VERSION")));
        let _ = std::fs::remove_dir_all(&out);
    })
}
