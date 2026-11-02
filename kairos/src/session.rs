//! A session holds the checker state, the top-level frame and the sources. A file run and a
//! REPL are both sessions: each input is checked against everything defined before it, and a
//! failed check leaves the session exactly as it was.

use crate::bridge::Host;
use crate::diag::{self, Diag};
use crate::interp::{Ctl, Interp};
use crate::lexer;
use crate::parser;
use crate::types::{Checker, TopStmt};
use crate::value::Value;
use std::io::Write;

/// Standard library written in Kairós itself, loaded into every session.
pub const PRELUDE: &str = include_str!("../std/prelude.krs");

pub struct Session {
    checker: Checker,
    frame: Vec<Value>,
    out: Box<dyn Write>,
    sources: Vec<(String, String)>,
    /// the `b3` data module (data loads on first use)
    host: Host,
}

impl Session {
    pub fn new(out: Box<dyn Write>) -> Self {
        Self::with_host(out, Host::new())
    }

    pub fn with_host(out: Box<dyn Write>, host: Host) -> Self {
        let mut s = Session { checker: Checker::new(), frame: Vec::new(), out, sources: Vec::new(), host };
        if let Err(d) = s.run("<prelude>", PRELUDE) {
            panic!("the Kairós prelude does not compile:\n{}", s.render(&d));
        }
        s
    }

    fn front(&mut self, name: &str, src: &str) -> Result<Vec<TopStmt>, Diag> {
        let file = self.sources.len() as u16;
        self.sources.push((name.to_string(), src.to_string()));
        let toks = lexer::lex(src, file)?;
        let items = parser::parse(toks)?;
        let backup = self.checker.clone();
        match self.checker.check_items(&items) {
            Ok(s) => Ok(s),
            Err(d) => {
                self.checker = backup;
                Err(d)
            }
        }
    }

    /// Type-checks without running.
    pub fn check(&mut self, name: &str, src: &str) -> Result<(), Diag> {
        self.front(name, src).map(|_| ())
    }

    /// Checks and runs; returns the text of the last expression's value (for the REPL).
    pub fn run(&mut self, name: &str, src: &str) -> Result<Option<String>, Diag> {
        let stmts = self.front(name, src)?;
        let need = self.checker.main_slots() as usize;
        if self.frame.len() < need {
            self.frame.resize(need, Value::Unit);
        }
        let mut echo = None;
        {
            let mut it = Interp::new(&self.checker, &mut *self.out, &self.host);
            for s in &stmts {
                match it.eval(&s.hir, &mut self.frame) {
                    Ok(v) => {
                        echo = if s.echo && !matches!(v, Value::Unit) { Some(it.display(&v, false)) } else { None };
                    }
                    Err(Ctl::Err(d)) => return Err(d),
                    Err(_) => unreachable!("control flow escaped to the top level"),
                }
            }
        }
        let _ = self.out.flush();
        Ok(echo)
    }

    /// The `b3` data module of this session.
    pub fn host(&self) -> &Host {
        &self.host
    }

    pub fn render(&self, d: &Diag) -> String {
        match self.sources.get(d.span.file as usize) {
            Some((name, src)) => diag::render(d, src, name),
            None => format!("error[{}]: {}\n", d.code, d.msg),
        }
    }
}
