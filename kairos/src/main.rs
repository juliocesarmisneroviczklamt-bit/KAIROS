//! `kairos` command line: run, check and an interactive REPL.

use kairos::interp::STACK_BYTES;
use kairos::session::Session;
use std::io::{BufRead, Write};
use std::sync::atomic::Ordering;

const USAGE: &str = "Kairós — a language for financial research and AI

usage:
  kairos run <file.krs>     check and run a program
  kairos check <file.krs>   type-check only
  kairos repl               interactive session (also: just `kairos`)
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
    // no threads (WebAssembly/WASI): the main thread's stack is set at link time
    // (.cargo/config.toml); elsewhere assume a small one
    let main_stack = if cfg!(target_family = "wasm") { 64 << 20 } else { 1 << 20 };
    STACK_BYTES.store(main_stack, Ordering::Relaxed);
    std::process::exit(real_main(args));
}

fn real_main(args: Vec<String>) -> i32 {
    match args.get(1).map(String::as_str) {
        Some("run") => file_cmd(args.get(2), true),
        Some("check") => file_cmd(args.get(2), false),
        Some("repl") | None => repl(),
        Some("--version") | Some("-V") => {
            println!("kairos {}", env!("CARGO_PKG_VERSION"));
            0
        }
        // `kairos programa.krs`, and scripts with `#!/usr/bin/env kairos` (any file name)
        Some(path) if path.ends_with(".krs") || std::path::Path::new(path).is_file() => {
            file_cmd(Some(&path.to_string()), true)
        }
        _ => {
            eprintln!("{USAGE}");
            2
        }
    }
}

fn file_cmd(path: Option<&String>, run: bool) -> i32 {
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
