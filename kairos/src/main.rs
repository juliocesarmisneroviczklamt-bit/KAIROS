//! `kairos` command line: run, check and an interactive REPL.

use kairos::interp::STACK_BYTES;
use kairos::session::Session;
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;

const USAGE: &str = "Kairós — a language for financial research and AI

usage:
  kairos run <file.krs> [args...]
                            check and run a program (`args()` gives the arguments)
  kairos check <file.krs>   type-check only
  kairos repl               interactive session (also: just `kairos`)
  kairos ledger             the holdout ledger: months seen, protocols, holdouts opened
  kairos ledger withdraw <protocol>
                            give up a protocol whose holdout was never opened
  kairos --version";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // The reference interpreter recurses on the HIR, so it runs on a large stack. Systems with a
    // small address space (32-bit) get a smaller one, and the main thread as a last resort.
    for mb in [512usize, 64, 16] {
        let a = args.clone();
        let spawned = std::thread::Builder::new().stack_size(mb << 20).spawn(move || {
            STACK_BYTES.store(mb << 20, Ordering::Relaxed);
            real_main(a)
        });
        if let Ok(worker) = spawned {
            std::process::exit(worker.join().unwrap_or(101));
        }
    }
    // No threads (WebAssembly/WASI). There the depth of calls is limited by the runtime, not by
    // the program: wasmtime allows 512 KiB unless run with `-W max-wasm-stack=N`; then pass the
    // same N in KAIROS_STACK_BYTES (`--env KAIROS_STACK_BYTES=N`) to allow deeper recursion.
    let default = if cfg!(target_family = "wasm") { 512 << 10 } else { 1 << 20 };
    let main_stack = std::env::var("KAIROS_STACK_BYTES").ok().and_then(|s| s.parse().ok()).unwrap_or(default);
    STACK_BYTES.store(main_stack, Ordering::Relaxed);
    std::process::exit(real_main(args));
}

fn real_main(args: Vec<String>) -> i32 {
    match args.get(1).map(String::as_str) {
        Some("run") => file_cmd(args.get(2), args.get(3..).unwrap_or_default(), true),
        Some("check") => file_cmd(args.get(2), &[], false),
        Some("repl") | None => repl(),
        Some("ledger") => ledger_cmd(&args[2..]),
        // the lexer's tokens, one per line (to compare with the lexer written in Kairós)
        Some("tokens") => tokens_cmd(args.get(2)),
        // the syntax tree, one item per line (to compare with the parser written in Kairós)
        Some("ast") => ast_cmd(args.get(2)),
        // what a file declares, as the checker sees it (to compare with the checker in Kairós)
        Some("types") => match args.get(2).map(|p| (p, std::fs::read_to_string(p))) {
            Some((p, Ok(src))) => {
                let mut s = Session::new(Box::new(std::io::stdout()));
                s.mark_imported(p);
                print!("{}", s.types_dump(p, &src));
                0
            }
            _ => {
                eprintln!("{USAGE}");
                2
            }
        },
        Some("--version") | Some("-V") => {
            println!("kairos {}", env!("CARGO_PKG_VERSION"));
            0
        }
        // `kairos programa.krs`, and scripts with `#!/usr/bin/env kairos` (any file name)
        Some(path) if path.ends_with(".krs") || std::path::Path::new(path).is_file() => {
            file_cmd(Some(&path.to_string()), &args[2..], true)
        }
        _ => {
            eprintln!("{USAGE}");
            2
        }
    }
}

fn tokens_cmd(path: Option<&String>) -> i32 {
    let Some(path) = path else {
        eprintln!("{USAGE}");
        return 2;
    };
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 2;
        }
    };
    match kairos::lexer::lex(&src, 0) {
        Ok(toks) => {
            for t in &toks {
                println!("{}", kairos::lexer::canonical(t));
            }
        }
        Err(d) => {
            println!("error {} {} {} {}", d.code, d.span.lo, d.span.hi, d.msg);
            if let Some(h) = &d.hint {
                println!("hint {h}");
            }
        }
    }
    0
}

fn ast_cmd(path: Option<&String>) -> i32 {
    let Some(path) = path else {
        eprintln!("{USAGE}");
        return 2;
    };
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 2;
        }
    };
    match kairos::lexer::lex(&src, 0).and_then(kairos::parser::parse) {
        Ok(items) => print!("{}", kairos::ast::canonical(&items)),
        Err(d) => {
            println!("error {} {} {} {}", d.code, d.span.lo, d.span.hi, d.msg);
            if let Some(h) = &d.hint {
                println!("hint {h}");
            }
        }
    }
    0
}

fn file_cmd(path: Option<&String>, prog_args: &[String], run: bool) -> i32 {
    let Some(path) = path else {
        eprintln!("{USAGE}");
        return 2;
    };
    let src = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return 2;
        }
    };
    let mut s = Session::new(Box::new(std::io::stdout()));
    // the program's name in the registry of trials
    let stem = std::path::Path::new(path).file_stem().map(|x| x.to_string_lossy().into_owned()).unwrap_or_default();
    s.host().set_program(&stem);
    s.host().set_args(prog_args.to_vec());
    s.mark_imported(path);
    let res = if run { s.run(path, &src).map(|_| ()) } else { s.check(path, &src) };
    match res {
        Ok(()) => {
            if !run {
                println!("{path}: ok");
            }
            0
        }
        Err(d) => {
            eprint!("{}", s.render(&d));
            1
        }
    }
}

/// `kairos ledger [withdraw <protocol>]`
fn ledger_cmd(rest: &[String]) -> i32 {
    let host = kairos::bridge::Host::new();
    let res = match rest {
        [] => host.ledger_report(),
        [w, which] if w == "withdraw" => host.withdraw(which),
        _ => {
            eprintln!("{USAGE}");
            return 2;
        }
    };
    match res {
        Ok(text) => {
            print!("{text}");
            if !text.ends_with('\n') {
                println!();
            }
            0
        }
        Err(e) => {
            eprintln!("error[{}]: {}", e.code, e.msg);
            if let Some(h) = e.hint {
                eprintln!("  = hint: {h}");
            }
            1
        }
    }
}

/// Counts unclosed delimiters so multi-line input can be typed naturally.
fn open_delims(src: &str) -> i32 {
    let mut depth = 0;
    let mut in_str = false;
    let mut prev = '\0';
    for line in src.lines() {
        let code = if in_str { line } else { line.split("//").next().unwrap_or("") };
        for c in code.chars() {
            if in_str {
                if c == '"' && prev != '\\' {
                    in_str = false;
                }
            } else {
                match c {
                    '"' => in_str = true,
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => depth -= 1,
                    _ => {}
                }
            }
            prev = c;
        }
    }
    depth
}

fn repl() -> i32 {
    println!("Kairós {} — type :help for help, :q to quit", env!("CARGO_PKG_VERSION"));
    let mut s = Session::new(Box::new(std::io::stdout()));
    s.host().set_program("repl");
    let stdin = std::io::stdin();
    let mut buf = String::new();
    let mut n = 0;
    loop {
        print!("{}", if buf.is_empty() { "kai> " } else { "...  " });
        let _ = std::io::stdout().flush();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            return 0;
        }
        let trimmed = line.trim();
        if buf.is_empty() {
            match trimmed {
                ":q" | ":quit" | ":exit" => return 0,
                ":help" | ":h" => {
                    println!("Write declarations (fn, struct, enum) or statements; the value of an expression is shown.");
                    println!("Multi-line input continues while ( [ {{ are open. Standard library: mean, variance,");
                    println!("std_dev, product, compound, wealth, max_drawdown, sharpe (std/prelude.krs).");
                    continue;
                }
                "" => continue,
                _ => {}
            }
        }
        buf.push_str(&line);
        if open_delims(&buf) > 0 {
            continue;
        }
        n += 1;
        match s.run(&format!("<repl:{n}>"), &buf) {
            Ok(Some(v)) => println!("= {v}"),
            Ok(None) => {}
            Err(d) => eprint!("{}", s.render(&d)),
        }
        buf.clear();
    }
}
