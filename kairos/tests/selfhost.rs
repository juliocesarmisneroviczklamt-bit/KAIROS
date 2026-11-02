//! M12: Kairós written in Kairós. Each piece of the compiler written in Kairós
//! (kairos/compiler/*.krs) must do exactly what the Rust one does, on every .krs file of the
//! repository and on the edge cases of tests/lexer.

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
    if cfg!(target_family = "wasm") {
        STACK_BYTES.store(4 << 20, Ordering::Relaxed);
        return f();
    }
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

/// Every .krs file under the crate (programs, the lab, the standard library, the tests).
fn krs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "target") {
                krs_files(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "krs") {
            out.push(p);
        }
    }
}

/// Runs a program of kairos/compiler with arguments; its output.
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

#[test]
fn the_lexer_in_kairos_is_the_rust_one() {
    on_big_stack(|| {
        let mut files = Vec::new();
        krs_files(Path::new(env!("CARGO_MANIFEST_DIR")), &mut files);
        assert!(files.len() > 100, "the repository's programs");
        let mut want = String::new();
        for f in &files {
            want.push_str(&format!("== {}\n", f.to_string_lossy()));
            let src = std::fs::read_to_string(f).unwrap();
            match kairos::lexer::lex(&src, 0) {
                Ok(toks) => {
                    for t in &toks {
                        want.push_str(&kairos::lexer::canonical(t));
                        want.push('\n');
                    }
                }
                Err(d) => {
                    want.push_str(&format!("error {} {} {} {}\n", d.code, d.span.lo, d.span.hi, d.msg));
                    if let Some(h) = &d.hint {
                        want.push_str(&format!("hint {h}\n"));
                    }
                }
            }
        }
        let got = run_compiler("tokens.krs", files.iter().map(|f| f.to_string_lossy().into_owned()).collect());
        if got != want {
            let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let k = g.iter().zip(&w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
            let file = w[..k].iter().rev().find(|l| l.starts_with("== ")).copied().unwrap_or("");
            panic!("the lexers differ in {file}, line {k}:\n  Kairós: {:?}\n  Rust:   {:?}", g.get(k), w.get(k));
        }
    })
}

/// The parser written in Kairós (kairos/compiler/parser.krs) builds the Rust parser's tree, node
/// for node with the same spans, and fails with the same errors and hints.
#[test]
fn the_parser_in_kairos_is_the_rust_one() {
    on_big_stack(|| {
        let mut files = Vec::new();
        krs_files(Path::new(env!("CARGO_MANIFEST_DIR")), &mut files);
        // WebAssembly runs these tests unoptimized on a 4 MiB stack, too little for a recursive
        // parser interpreted on deep files: there, the edge cases (the other targets parse all)
        if cfg!(target_family = "wasm") {
            files.retain(|f| f.parent().and_then(|d| d.file_name()).is_some_and(|d| d == "parser" || d == "lexer"));
        }
        let mut want = String::new();
        for f in &files {
            want.push_str(&format!("== {}\n", f.to_string_lossy()));
            let src = std::fs::read_to_string(f).unwrap();
            match kairos::lexer::lex(&src, 0).and_then(kairos::parser::parse) {
                Ok(items) => want.push_str(&kairos::ast::canonical(&items)),
                Err(d) => {
                    want.push_str(&format!("error {} {} {} {}\n", d.code, d.span.lo, d.span.hi, d.msg));
                    if let Some(h) = &d.hint {
                        want.push_str(&format!("hint {h}\n"));
                    }
                }
            }
        }
        let got = run_compiler("ast.krs", files.iter().map(|f| f.to_string_lossy().into_owned()).collect());
        if got != want {
            let (g, w): (Vec<&str>, Vec<&str>) = (got.lines().collect(), want.lines().collect());
            let k = g.iter().zip(&w).position(|(a, b)| a != b).unwrap_or(g.len().min(w.len()));
            let file = w[..k].iter().rev().find(|l| l.starts_with("== ")).copied().unwrap_or("");
            panic!("the parsers differ in {file}, line {k}:\n  Kairós: {:?}\n  Rust:   {:?}", g.get(k), w.get(k));
        }
    })
}

/// The front end written in Kairós (kairos/compiler: check.krs, body.krs, temporal.krs,
/// front.krs) registers what each file declares (structs, enums, axes, protocols, the functions'
/// signatures, through its imports), gives every variable of every function and of the top level
/// the type the Rust checker gives it (tensors, `grad`, holdouts and the `b3` bridge included),
/// builds the same HIR (every function, lambda and top-level statement, node for node), finds
/// the same knowledge times in the temporal analysis (each function's summary, each top-level
/// variable), and fails with the same first error: on every .krs file of the repository.
#[test]
fn the_front_end_in_kairos_is_the_rust_one() {
    // the checker reads the prelude first (hundreds of lines): too deep for the parser
    // interpreted unoptimized on WebAssembly's 4 MiB stack; the other targets run it
    if cfg!(target_family = "wasm") {
        return;
    }
    on_big_stack(|| {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        krs_files(root, &mut files);
        let mut args = vec![root.join("std").to_string_lossy().into_owned()];
        args.extend(files.iter().map(|f| f.to_string_lossy().into_owned()));
        let got = run_compiler("types.krs", args);
        let mut sections: Vec<(String, String)> = Vec::new();
        for l in got.lines() {
            match l.strip_prefix("== ") {
                Some(p) => sections.push((p.to_string(), String::new())),
                None => {
                    let s = &mut sections.last_mut().expect("a file").1;
                    s.push_str(l);
                    s.push('\n');
                }
            }
        }
        assert_eq!(sections.len(), files.len());
        let (mut compared, mut bodies, mut errors) = (0, 0, 0);
        for (path, krs) in &sections {
            // K0000: something the Rust checker knows and the one in Kairós does not (yet)
            assert!(!krs.starts_with("error K0000"), "the checker written in Kairós does not check {path}: {krs}");
            let src = std::fs::read_to_string(path).unwrap();
            let mut s = Session::new(Box::new(std::io::sink()));
            s.mark_imported(path);
            let rust = s.hir_dump(path, &src);
            assert_eq!(krs, &rust, "the front ends differ on {path}");
            if krs.starts_with("error") {
                errors += 1;
            } else {
                compared += 1;
                bodies += krs.lines().filter(|l| l.starts_with("body ")).count();
            }
        }
        assert!(compared > 100 && bodies > 1500, "the repository's programs ({compared} files, {bodies} functions)");
        println!("{compared} files with the same types, HIR and knowledge times ({bodies} functions), {errors} with the same first error");
    })
}
