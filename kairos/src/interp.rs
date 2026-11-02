//! Reference interpreter (M1): walks the typed HIR. Correctness first; the native backend is M7.
//!
//! `inout` arguments are moved out of the caller's place for the call and moved back afterwards
//! (no copy, no aliasing). Integer arithmetic is checked: overflow and division by zero are
//! runtime errors, never silent.

use crate::ast::{BinOp, UnOp};
use crate::bridge::Host;
use crate::diag::{Diag, Span};
use crate::hir::*;
use crate::types::Checker;
use crate::value::{FuncVal, MapVal, Value, total_cmp};
use std::cmp::Ordering;
use std::io::Write;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

mod diff;
mod tensors;

pub enum Ctl {
    Ret(Value),
    Brk,
    Cont,
    Err(Diag),
}

pub type R<T> = Result<T, Ctl>;

/// Stack size (bytes) of the thread that runs the interpreter, set by whoever creates it. The
/// recursion limit scales with it, so deep recursion is a clean R0005 on every system, including
/// 32-bit ones that get a smaller stack. 0 = unknown (assume a large stack).
pub static STACK_BYTES: AtomicUsize = AtomicUsize::new(0);

fn max_depth() -> u32 {
    // stack used per Kairós call, measured with a margin (debug builds use far more)
    let per_call: usize = if cfg!(debug_assertions) { 128 << 10 } else { 16 << 10 };
    match STACK_BYTES.load(AtomicOrdering::Relaxed) {
        0 => 20_000,
        bytes => u32::try_from(bytes / per_call).unwrap_or(u32::MAX).clamp(32, 20_000),
    }
}

fn err<T>(code: &'static str, sp: Span, msg: impl Into<String>) -> R<T> {
    Err(Ctl::Err(Diag::new(code, sp, msg)))
}

fn str_of(v: &Value) -> &str {
    match v {
        Value::Str(s) => s,
        other => unreachable!("expected a string, got {other:?}"),
    }
}

fn str_list<S: AsRef<str>>(xs: impl IntoIterator<Item = S>) -> Value {
    Value::List(Rc::new(xs.into_iter().map(|s| Value::Str(s.as_ref().into())).collect()))
}

/// A file's text: UTF-8 (without a byte-order mark), or Latin-1 when it is not UTF-8.
fn decode_text(bytes: Vec<u8>) -> String {
    let s = match String::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => e.into_bytes().iter().map(|&b| b as char).collect(),
    };
    match s.strip_prefix('\u{feff}') {
        Some(rest) => rest.to_string(),
        None => s,
    }
}

/// HTTP(S) through the system's `curl` (Windows 10+, macOS and Linux have it): the system's TLS,
/// without linking a C library into Kairós. Retries what fails for the network; 4xx fails at once.
fn curl(url: &str, out: Option<&str>) -> Result<Vec<u8>, String> {
    if cfg!(target_family = "wasm") {
        return Err("there is no network under WebAssembly".into());
    }
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-sS", "-f", "-L", "--retry", "4", "--retry-delay", "2", "--connect-timeout", "30", "--max-time", "900"]);
    cmd.args(["-A", concat!("Mozilla/5.0 (Windows NT 10.0; Win64; x64) kairos/", env!("CARGO_PKG_VERSION"))]);
    if let Some(p) = out {
        cmd.args(["-o", p]);
    }
    cmd.arg(url);
    match cmd.output() {
        Ok(o) if o.status.success() => Ok(o.stdout),
        Ok(o) => Err(format!("{url}: {}", String::from_utf8_lossy(&o.stderr).trim())),
        Err(e) => Err(format!("cannot run curl ({e}): downloads need curl installed")),
    }
}

/// The lines of a text file, or of the first file inside a .zip, that have each `(column, text)`
/// of `keep` (byte columns: B3's fixed-width files are Latin-1, one byte per character). A line
/// is UTF-8 when it is valid UTF-8, Latin-1 otherwise; `\n` and `\r\n` are removed.
fn read_lines_kept(path: &str, keep: &[(usize, Vec<u8>)]) -> std::io::Result<Vec<String>> {
    use std::io::{BufRead, BufReader};
    let mut out = Vec::new();
    let mut scan = |rd: &mut dyn BufRead| -> std::io::Result<()> {
        let mut buf = Vec::with_capacity(512);
        loop {
            buf.clear();
            if rd.read_until(b'\n', &mut buf)? == 0 {
                return Ok(());
            }
            while matches!(buf.last(), Some(b'\n' | b'\r')) {
                buf.pop();
            }
            if keep.iter().all(|(at, t)| buf.get(*at..at + t.len()) == Some(&t[..])) {
                out.push(match std::str::from_utf8(&buf) {
                    Ok(s) => s.to_string(),
                    Err(_) => buf.iter().map(|&c| c as char).collect(),
                });
            }
        }
    };
    let f = std::fs::File::open(path)?;
    if path.to_ascii_lowercase().ends_with(".zip") {
        let mut z = zip::ZipArchive::new(f).map_err(std::io::Error::other)?;
        let entry = z.by_index(0).map_err(std::io::Error::other)?;
        scan(&mut BufReader::with_capacity(1 << 20, entry))?;
    } else {
        scan(&mut BufReader::with_capacity(1 << 20, f))?;
    }
    Ok(out)
}

/// GETs `parallel` at a time (each one a `curl`), in the order given.
fn curl_all(urls: &[String], parallel: usize) -> Vec<Result<Vec<u8>, String>> {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let out: Mutex<Vec<Option<Result<Vec<u8>, String>>>> = Mutex::new(vec![None; urls.len()]);
    let next = AtomicUsize::new(0);
    let work = || loop {
        let i = next.fetch_add(1, Ordering::Relaxed);
        if i >= urls.len() {
            return;
        }
        let r = curl(&urls[i], None);
        out.lock().expect("results")[i] = Some(r);
    };
    if cfg!(target_family = "wasm") || parallel <= 1 {
        work();
    } else {
        std::thread::scope(|s| {
            for _ in 0..parallel.min(32) {
                s.spawn(work);
            }
        });
    }
    out.into_inner().expect("results").into_iter().map(|r| r.unwrap_or_else(|| Err("not run".into()))).collect()
}

/// `n` bytes of `b` from `at`, or R0001.
fn bytes_at(b: &[u8], at: i64, n: usize, sp: Span) -> R<&[u8]> {
    let ok = usize::try_from(at).ok().and_then(|a| b.get(a..a.checked_add(n)?));
    match ok {
        Some(s) => Ok(s),
        None => err("R0001", sp, format!("reading {n} byte(s) at {at}, out of bounds for {} bytes", b.len())),
    }
}

fn as_bool(v: Value) -> bool {
    matches!(v, Value::Bool(true))
}

fn as_int(v: Value) -> i64 {
    match v {
        Value::Int(i) => i,
        other => unreachable!("expected int, got {other:?}"),
    }
}

enum Step {
    Field(u32),
    /// a list index or a map key
    Index(Value, Span),
}

/// The text of a map key in messages.
fn show_key(k: &Value) -> String {
    match k {
        Value::Str(s) => format!("{:?}", &**s),
        other => format!("{other:?}").trim_start_matches("Int(").trim_end_matches(')').to_string(),
    }
}

fn place_mut<'v>(root: &'v mut Value, path: &[Step]) -> R<&'v mut Value> {
    place_at(root, path, false)
}

/// The value at `path`; with `insert`, a missing map key at the end is added (`m[k] = v`).
fn place_at<'v>(root: &'v mut Value, path: &[Step], insert: bool) -> R<&'v mut Value> {
    let mut cur = root;
    let last = path.len().saturating_sub(1);
    for (k, st) in path.iter().enumerate() {
        cur = match (st, cur) {
            (Step::Field(i), Value::Struct(_, fs) | Value::Tuple(fs)) => &mut Rc::make_mut(fs)[*i as usize],
            (Step::Field(_), _) => unreachable!("field of a non-struct"),
            (Step::Index(key, sp), Value::Map(m)) => {
                let m = Rc::make_mut(m);
                if insert && k == last {
                    m.slot(key)
                } else {
                    match m.get_mut(key) {
                        Some(v) => v,
                        None => return err("R0012", *sp, format!("no key {} in the map", show_key(key))),
                    }
                }
            }
            (Step::Index(iv, sp), Value::List(xs)) => {
                let i = as_int(iv.clone());
                let xs = Rc::make_mut(xs);
                let n = xs.len();
                if i < 0 || i as u64 >= n as u64 {
                    return err("R0001", *sp, format!("index {i} out of bounds for a list of length {n}"));
                }
                &mut xs[i as usize]
            }
            (Step::Index(..), _) => unreachable!("index of a non-list"),
        };
    }
    Ok(cur)
}

fn overflow(sp: Span) -> Ctl {
    Ctl::Err(Diag::new("R0006", sp, "integer overflow").hint("i64 arithmetic is checked; use f64 for very large values"))
}

fn ordered(op: BinOp, o: Option<Ordering>) -> bool {
    use Ordering::*;
    matches!(
        (op, o),
        (BinOp::Lt, Some(Less)) | (BinOp::Le, Some(Less | Equal)) | (BinOp::Gt, Some(Greater)) | (BinOp::Ge, Some(Greater | Equal))
    )
}

pub fn binop(op: BinOp, a: Value, b: Value, sp: Span) -> R<Value> {
    use Value::*;
    Ok(match (op, a, b) {
        (BinOp::Add, Int(x), Int(y)) => Int(x.checked_add(y).ok_or_else(|| overflow(sp))?),
        (BinOp::Sub, Int(x), Int(y)) => Int(x.checked_sub(y).ok_or_else(|| overflow(sp))?),
        (BinOp::Mul, Int(x), Int(y)) => Int(x.checked_mul(y).ok_or_else(|| overflow(sp))?),
        (BinOp::Div | BinOp::Rem, Int(_), Int(0)) => return err("R0002", sp, "division by zero"),
        (BinOp::Div, Int(x), Int(y)) => Int(x.checked_div(y).ok_or_else(|| overflow(sp))?),
        (BinOp::Rem, Int(x), Int(y)) => Int(x.checked_rem(y).ok_or_else(|| overflow(sp))?),
        (BinOp::Pow, Int(x), Int(y)) => {
            if y < 0 {
                return Err(Ctl::Err(
                    Diag::new("R0007", sp, "negative exponent in integer power").hint("use floats: `float(x) ^ float(y)`"),
                ));
            }
            let e = u32::try_from(y).map_err(|_| overflow(sp))?;
            Int(x.checked_pow(e).ok_or_else(|| overflow(sp))?)
        }
        (BinOp::Add, Float(x), Float(y)) => Float(x + y),
        (BinOp::Sub, Float(x), Float(y)) => Float(x - y),
        (BinOp::Mul, Float(x), Float(y)) => Float(x * y),
        (BinOp::Div, Float(x), Float(y)) => Float(x / y),
        (BinOp::Rem, Float(x), Float(y)) => Float(x % y),
        (BinOp::Pow, Float(x), Float(y)) => Float(libm::pow(x, y)),
        (BinOp::Add, Str(x), Str(y)) => Str(format!("{x}{y}").into()),
        (BinOp::Eq, a, b) => Bool(a == b),
        (BinOp::Ne, a, b) => Bool(a != b),
        (op, Int(x), Int(y)) => Bool(ordered(op, x.partial_cmp(&y))),
        (op, Float(x), Float(y)) => Bool(ordered(op, x.partial_cmp(&y))),
        (op, Str(x), Str(y)) => Bool(ordered(op, x.partial_cmp(&y))),
        (op, a, b) => unreachable!("type checker guarantees operands: {op:?} {a:?} {b:?}"),
    })
}

pub struct Interp<'a> {
    ck: &'a Checker,
    out: &'a mut dyn Write,
    host: &'a Host,
    /// M7: native code for pure scalar functions
    jit: &'a std::cell::RefCell<crate::native::Jit>,
    depth: u32,
    max_depth: u32,
    /// the tape of the `grad` being computed (M4)
    tape: std::cell::RefCell<Option<crate::autodiff::Tape>>,
}

impl<'a> Interp<'a> {
    pub fn new(ck: &'a Checker, out: &'a mut dyn Write, host: &'a Host, jit: &'a std::cell::RefCell<crate::native::Jit>) -> Self {
        Interp { ck, out, host, jit, depth: 0, max_depth: max_depth(), tape: std::cell::RefCell::new(None) }
    }

    fn eval_all(&mut self, xs: &[HExpr], f: &mut Vec<Value>) -> R<Vec<Value>> {
        let mut v = Vec::with_capacity(xs.len());
        for x in xs {
            v.push(self.eval(x, f)?);
        }
        Ok(v)
    }

    fn run_loop_body(&mut self, body: &HExpr, f: &mut Vec<Value>) -> R<bool> {
        match self.eval(body, f) {
            Ok(_) | Err(Ctl::Cont) => Ok(true),
            Err(Ctl::Brk) => Ok(false),
            Err(e) => Err(e),
        }
    }

    pub fn eval(&mut self, e: &HExpr, f: &mut Vec<Value>) -> R<Value> {
        match e {
            HExpr::Lit(v) => Ok(v.clone()),
            HExpr::IntLit(v, _) => Ok(Value::Int(*v)),
            HExpr::Local(s) => Ok(f[*s as usize].clone()),
            HExpr::FnRef(id) => Ok(Value::Func(Rc::new(FuncVal::User(*id)))),
            HExpr::List(xs) => Ok(Value::List(Rc::new(self.eval_all(xs, f)?))),
            HExpr::MapLit(xs) => {
                let mut m = MapVal::default();
                let mut it = self.eval_all(xs, f)?.into_iter();
                while let (Some(k), Some(v)) = (it.next(), it.next()) {
                    m.insert(k, v);
                }
                Ok(Value::Map(Rc::new(m)))
            }
            HExpr::Tuple(xs) => Ok(Value::Tuple(Rc::new(self.eval_all(xs, f)?))),
            HExpr::Struct(id, xs) => Ok(Value::Struct(*id, Rc::new(self.eval_all(xs, f)?))),
            HExpr::Variant(en, vi, xs) => Ok(Value::Enum(*en, *vi, Rc::new(self.eval_all(xs, f)?))),
            HExpr::Field(x, i) => match self.eval(x, f)? {
                Value::Struct(_, fs) | Value::Tuple(fs) => Ok(fs[*i as usize].clone()),
                other => unreachable!("field of {other:?}"),
            },
            HExpr::Index(x, i, sp) => {
                let xs = self.eval(x, f)?;
                let iv = self.eval(i, f)?;
                if let Value::Map(m) = &xs {
                    return match m.get(&iv) {
                        Some(v) => Ok(v.clone()),
                        None => err("R0012", *sp, format!("no key {} in the map", show_key(&iv))),
                    };
                }
                let i = as_int(iv);
                match xs {
                    Value::List(xs) => {
                        if i < 0 || i as u64 >= xs.len() as u64 {
                            return err("R0001", *sp, format!("index {i} out of bounds for a list of length {}", xs.len()));
                        }
                        Ok(xs[i as usize].clone())
                    }
                    other => unreachable!("index of {other:?}"),
                }
            }
            HExpr::TensorIndex(x, sels, sp) => self.tensor_index(x, sels, f, *sp),
            HExpr::TensorAssign(p, sels, op, v, sp) => self.tensor_assign(p, sels, *op, v, f, *sp),
            HExpr::Host(hf, args, sp) => {
                let v = self.eval_all(args, f)?;
                self.host_call(*hf, v, *sp)
            }
            HExpr::CallFn(id, args, sp) => self.call_user(*id, args, f, *sp),
            HExpr::CallValue(c, args, sp) => {
                let fv = self.eval(c, f)?;
                let vals = self.eval_all(args, f)?;
                self.call_value(&fv, vals, *sp)
            }
            HExpr::CallBuiltin(b, args, sp) => self.builtin(*b, args, f, *sp),
            HExpr::Unary(op, x, sp) => match (op, self.eval(x, f)?) {
                (UnOp::Neg, Value::Int(i)) => i.checked_neg().map(Value::Int).ok_or_else(|| overflow(*sp)),
                (UnOp::Neg, Value::Float(x)) => Ok(Value::Float(-x)),
                (UnOp::Not, Value::Bool(b)) => Ok(Value::Bool(!b)),
                (op, Value::Tensor(t)) => self.tensor_unary(*op, &t, *sp),
                // tracked values and structs/tuples of them (the tangent space)
                (UnOp::Neg, v @ (Value::Node(_) | Value::Struct(..) | Value::Tuple(_))) => self.neg(v, *sp),
                (op, v) => unreachable!("unary {op:?} on {v:?}"),
            },
            HExpr::Binary(op, a, b, sp) => {
                let a = self.eval(a, f)?;
                let b = self.eval(b, f)?;
                self.bin(*op, a, b, *sp)
            }
            HExpr::And(a, b) => {
                if as_bool(self.eval(a, f)?) {
                    self.eval(b, f)
                } else {
                    Ok(Value::Bool(false))
                }
            }
            HExpr::Or(a, b) => {
                if as_bool(self.eval(a, f)?) {
                    Ok(Value::Bool(true))
                } else {
                    self.eval(b, f)
                }
            }
            HExpr::Assign(p, v) => {
                let v = self.eval(v, f)?;
                let (root, path) = self.resolve_place(p, f)?;
                *place_at(&mut f[root], &path, true)? = v;
                Ok(Value::Unit)
            }
            HExpr::OpAssign(p, op, v, sp) => {
                let rhs = self.eval(v, f)?;
                let (root, path) = self.resolve_place(p, f)?;
                let slot = place_mut(&mut f[root], &path)?;
                let cur = slot.clone();
                *slot = self.bin(*op, cur, rhs, *sp)?;
                Ok(Value::Unit)
            }
            HExpr::If(c, t, els) => {
                if as_bool(self.eval(c, f)?) {
                    self.eval(t, f)
                } else if let Some(x) = els {
                    self.eval(x, f)
                } else {
                    Ok(Value::Unit)
                }
            }
            HExpr::While(c, body) => {
                while as_bool(self.eval(c, f)?) {
                    if !self.run_loop_body(body, f)? {
                        break;
                    }
                }
                Ok(Value::Unit)
            }
            HExpr::ForRange { slot, lo, hi, inclusive, body, .. } => {
                let lo = as_int(self.eval(lo, f)?);
                let hi = as_int(self.eval(hi, f)?);
                let mut i = lo;
                while if *inclusive { i <= hi } else { i < hi } {
                    f[*slot as usize] = Value::Int(i);
                    if !self.run_loop_body(body, f)? {
                        break;
                    }
                    match i.checked_add(1) {
                        Some(n) => i = n,
                        None => break,
                    }
                }
                Ok(Value::Unit)
            }
            HExpr::ForList { pat, list, body } => {
                let xs = match self.eval(list, f)? {
                    Value::List(xs) => xs,
                    other => unreachable!("for over {other:?}"),
                };
                for x in xs.iter() {
                    matches_pat(pat, x, f);
                    if !self.run_loop_body(body, f)? {
                        break;
                    }
                }
                Ok(Value::Unit)
            }
            HExpr::Match(s, arms, sp) => {
                let v = self.eval(s, f)?;
                for (p, body) in arms {
                    if matches_pat(p, &v, f) {
                        return self.eval(body, f);
                    }
                }
                err("R0008", *sp, "no match arm matched")
            }
            HExpr::Block(stmts, tail) => {
                for s in stmts {
                    self.eval(s, f)?;
                }
                match tail {
                    Some(t) => self.eval(t, f),
                    None => Ok(Value::Unit),
                }
            }
            HExpr::Let(p, init) => {
                let v = self.eval(init, f)?;
                matches_pat(p, &v, f);
                Ok(Value::Unit)
            }
            HExpr::Lambda(def, outer) => Ok(Value::Func(Rc::new(FuncVal::Closure {
                def: *def,
                captured: outer.iter().map(|s| f[*s as usize].clone()).collect(),
            }))),
            HExpr::Return(v) => {
                let v = match v {
                    Some(x) => self.eval(x, f)?,
                    None => Value::Unit,
                };
                Err(Ctl::Ret(v))
            }
            HExpr::Break => Err(Ctl::Brk),
            HExpr::Continue => Err(Ctl::Cont),
        }
    }

    fn resolve_place(&mut self, p: &HPlace, f: &mut Vec<Value>) -> R<(usize, Vec<Step>)> {
        match p {
            HPlace::Local(s) => Ok((*s as usize, Vec::new())),
            HPlace::Field(inner, i) => {
                let (r, mut path) = self.resolve_place(inner, f)?;
                path.push(Step::Field(*i));
                Ok((r, path))
            }
            HPlace::Index(inner, ie, sp) => {
                let (r, mut path) = self.resolve_place(inner, f)?;
                let i = self.eval(ie, f)?;
                path.push(Step::Index(i, *sp));
                Ok((r, path))
            }
        }
    }

    fn enter(&mut self, sp: Span, run: impl FnOnce(&mut Self) -> R<Value>) -> R<Value> {
        if self.depth >= self.max_depth {
            return err("R0005", sp, "stack overflow: too many nested calls");
        }
        self.depth += 1;
        let r = run(self);
        self.depth -= 1;
        match r {
            Ok(v) | Err(Ctl::Ret(v)) => Ok(v),
            Err(e) => Err(e),
        }
    }

    fn call_user(&mut self, id: u32, args: &[HArg], f: &mut Vec<Value>, sp: Span) -> R<Value> {
        let ck = self.ck;
        let body = ck.bodies[id as usize].as_ref().expect("checked body");
        let mut frame = vec![Value::Unit; body.n_slots as usize];
        let mut back: Vec<(usize, usize, Vec<Step>)> = Vec::new();
        // values first: they may read a variable that is also passed as `inout` (`f(inout p, g(p))`)
        for (i, a) in args.iter().enumerate() {
            if let HArg::Val(e) = a {
                frame[i] = self.eval(e, f)?;
            }
        }
        // M7: a pure scalar function runs as machine code, when it can be compiled
        if args.iter().all(|a| matches!(a, HArg::Val(_))) {
            if self.depth >= self.max_depth {
                return err("R0005", sp, "stack overflow: too many nested calls");
            }
            let n = ck.sigs[id as usize].params.len();
            let native = self.jit.borrow_mut().call(ck, id, &frame[..n], self.depth + 1, self.max_depth);
            match native {
                Some(Ok(v)) => return Ok(v),
                Some(Err(d)) => return Err(Ctl::Err(d)),
                None => {}
            }
        }
        // then move the `inout` places into the call (and back afterwards)
        for (i, a) in args.iter().enumerate() {
            if let HArg::InOut(p) = a {
                let moved = self.resolve_place(p, f).and_then(|(root, path)| {
                    let slot = place_mut(&mut f[root], &path)?;
                    Ok((std::mem::take(slot), root, path))
                });
                match moved {
                    Ok((v, root, path)) => {
                        frame[i] = v;
                        back.push((i, root, path));
                    }
                    Err(e) => {
                        for (j, root, path) in back {
                            if let Ok(slot) = place_mut(&mut f[root], &path) {
                                *slot = std::mem::take(&mut frame[j]);
                            }
                        }
                        return Err(e);
                    }
                }
            }
        }
        let r = self.enter(sp, |me| me.eval(&body.body, &mut frame));
        for (i, root, path) in back {
            if let Ok(slot) = place_mut(&mut f[root], &path) {
                *slot = std::mem::take(&mut frame[i]);
            }
        }
        r
    }

    pub fn call_value(&mut self, fv: &Value, args: Vec<Value>, sp: Span) -> R<Value> {
        let Value::Func(func) = fv else { unreachable!("call of a non-function") };
        let ck = self.ck;
        match &**func {
            FuncVal::User(id) => {
                let body = ck.bodies[*id as usize].as_ref().expect("checked body");
                let mut frame = vec![Value::Unit; body.n_slots as usize];
                for (i, a) in args.into_iter().enumerate() {
                    frame[i] = a;
                }
                self.enter(sp, |me| me.eval(&body.body, &mut frame))
            }
            FuncVal::Closure { def, captured } => {
                let lb = &ck.lambdas[*def as usize];
                let mut frame = vec![Value::Unit; lb.n_slots as usize];
                for (i, a) in args.into_iter().enumerate() {
                    frame[i] = a;
                }
                for (k, s) in lb.capture_slots.iter().enumerate() {
                    frame[*s as usize] = captured[k].clone();
                }
                self.enter(sp, |me| me.eval(&lb.body, &mut frame))
            }
            FuncVal::Grad { f, wrt, with_value } => self.run_grad(f, *wrt, *with_value, args, sp),
        }
    }

    fn val(&mut self, a: &HArg, f: &mut Vec<Value>) -> R<Value> {
        match a {
            HArg::Val(e) => self.eval(e, f),
            HArg::InOut(_) => unreachable!("inout argument to a pure builtin"),
        }
    }

    fn builtin(&mut self, b: Builtin, args: &[HArg], f: &mut Vec<Value>, sp: Span) -> R<Value> {
        if matches!(b, Builtin::PutU8 | Builtin::PutU32 | Builtin::PutU64 | Builtin::PutI64 | Builtin::PutF64 | Builtin::PutF32 | Builtin::PutStr) {
            let HArg::InOut(p) = &args[0] else { unreachable!("put without a place") };
            let x = self.val(&args[1], f)?;
            let (root, path) = self.resolve_place(p, f)?;
            let Value::Bytes(bs) = place_mut(&mut f[root], &path)? else { unreachable!("put into non-bytes") };
            let bs = Rc::make_mut(bs);
            let range = |n: i64, max: u64, what: &str| -> R<u64> {
                match u64::try_from(n) {
                    Ok(u) if u <= max => Ok(u),
                    _ => err("R0014", sp, format!("{n} does not fit in {what}")),
                }
            };
            match (b, x) {
                (Builtin::PutU8, Value::Int(n)) => bs.push(range(n, 255, "u8")? as u8),
                (Builtin::PutU32, Value::Int(n)) => bs.extend_from_slice(&(range(n, u64::from(u32::MAX), "u32")? as u32).to_le_bytes()),
                (Builtin::PutU64, Value::Int(n)) => bs.extend_from_slice(&range(n, u64::MAX, "u64")?.to_le_bytes()),
                (Builtin::PutI64, Value::Int(n)) => bs.extend_from_slice(&n.to_le_bytes()),
                (Builtin::PutF64, Value::Float(x)) => bs.extend_from_slice(&x.to_le_bytes()),
                (Builtin::PutF32, Value::Float(x)) => bs.extend_from_slice(&(x as f32).to_le_bytes()),
                (Builtin::PutStr, Value::Str(s)) => bs.extend_from_slice(s.as_bytes()),
                (b, x) => unreachable!("{b:?} of {x:?}"),
            }
            return Ok(Value::Unit);
        }
        if b == Builtin::Remove {
            let HArg::InOut(p) = &args[0] else { unreachable!("remove without a place") };
            let key = self.val(&args[1], f)?;
            let (root, path) = self.resolve_place(p, f)?;
            let Value::Map(m) = place_mut(&mut f[root], &path)? else { unreachable!("remove on a non-map") };
            Rc::make_mut(m).remove(&key);
            return Ok(Value::Unit);
        }
        if matches!(b, Builtin::Push | Builtin::Pop | Builtin::Sort | Builtin::Reverse) {
            let HArg::InOut(p) = &args[0] else { unreachable!("mutating builtin without place") };
            let pushed = if b == Builtin::Push { Some(self.val(&args[1], f)?) } else { None };
            let (root, path) = self.resolve_place(p, f)?;
            let slot = place_mut(&mut f[root], &path)?;
            let Value::List(xs) = slot else { unreachable!("list builtin on a non-list") };
            let xs = Rc::make_mut(xs);
            return match b {
                Builtin::Push => {
                    xs.push(pushed.expect("pushed value"));
                    Ok(Value::Unit)
                }
                Builtin::Pop => xs.pop().ok_or_else(|| Ctl::Err(Diag::new("R0004", sp, "`pop` from an empty list"))),
                Builtin::Sort => {
                    xs.sort_by(total_cmp);
                    Ok(Value::Unit)
                }
                _ => {
                    xs.reverse();
                    Ok(Value::Unit)
                }
            };
        }
        let mut v = Vec::with_capacity(args.len());
        for a in args {
            v.push(self.val(a, f)?);
        }
        // M4: grad, the tangent space and the AI builtins (also strips tracked values elsewhere)
        if let Some(r) = self.diff_builtin(b, &mut v, sp)? {
            return Ok(r);
        }
        if let Some(r) = self.tensor_builtin(b, &v, sp)? {
            return Ok(r);
        }
        let float = |x: &Value| match x {
            Value::Float(x) => *x,
            other => unreachable!("expected float, got {other:?}"),
        };
        match b {
            Builtin::Print => {
                let s = v.iter().map(|x| self.display(x, true)).collect::<Vec<_>>().join(" ");
                let _ = writeln!(self.out, "{s}");
                Ok(Value::Unit)
            }
            Builtin::Len => Ok(Value::Int(match &v[0] {
                Value::List(xs) => xs.len() as i64,
                Value::Bytes(b) => b.len() as i64,
                Value::Map(m) => m.len() as i64,
                Value::Str(s) => s.chars().count() as i64,
                other => unreachable!("len of {other:?}"),
            })),
            Builtin::Contains => match &v[0] {
                Value::Str(s) => Ok(Value::Bool(s.contains(str_of(&v[1])))),
                Value::List(xs) => Ok(Value::Bool(xs.contains(&v[1]))),
                Value::Map(m) => Ok(Value::Bool(m.contains(&v[1]))),
                other => unreachable!("contains on {other:?}"),
            },
            Builtin::Keys | Builtin::Values | Builtin::Items | Builtin::Get => {
                let Value::Map(m) = &v[0] else { unreachable!("a map builtin on a non-map") };
                Ok(match b {
                    Builtin::Keys => Value::List(Rc::new(m.keys().to_vec())),
                    Builtin::Values => Value::List(Rc::new(m.values().to_vec())),
                    Builtin::Items => Value::List(Rc::new(
                        m.keys().iter().zip(m.values()).map(|(k, x)| Value::Tuple(Rc::new(vec![k.clone(), x.clone()]))).collect(),
                    )),
                    _ => m.get(&v[1]).cloned().unwrap_or_else(|| v[2].clone()),
                })
            }
            Builtin::Map | Builtin::Filter => {
                let Value::List(xs) = &v[0] else { unreachable!() };
                let mut out = Vec::with_capacity(xs.len());
                for x in xs.iter() {
                    let r = self.call_value(&v[1], vec![x.clone()], sp)?;
                    if b == Builtin::Map {
                        out.push(r);
                    } else if as_bool(r) {
                        out.push(x.clone());
                    }
                }
                Ok(Value::List(Rc::new(out)))
            }
            Builtin::Sum => {
                let Value::List(xs) = &v[0] else { unreachable!() };
                let mut acc = v[1].clone();
                for x in xs.iter() {
                    acc = binop(BinOp::Add, acc, x.clone(), sp)?;
                }
                Ok(acc)
            }
            Builtin::Min | Builtin::Max => {
                if let (Value::Float(x), Value::Float(y)) = (&v[0], &v[1]) {
                    if x.is_nan() || y.is_nan() {
                        return Ok(Value::Float(f64::NAN));
                    }
                }
                let a_less = as_bool(binop(BinOp::Lt, v[0].clone(), v[1].clone(), sp)?);
                let pick_first = if b == Builtin::Min { a_less } else { !a_less };
                Ok(if pick_first { v[0].clone() } else { v[1].clone() })
            }
            Builtin::Minimum | Builtin::Maximum => {
                let Value::List(xs) = &v[0] else { unreachable!() };
                let Some(first) = xs.first() else {
                    return err("R0004", sp, format!("`{}` of an empty list", builtin_name(b)));
                };
                let mut best = first.clone();
                for x in xs.iter().skip(1) {
                    let better = if b == Builtin::Minimum {
                        binop(BinOp::Lt, x.clone(), best.clone(), sp)?
                    } else {
                        binop(BinOp::Lt, best.clone(), x.clone(), sp)?
                    };
                    if as_bool(better) {
                        best = x.clone();
                    }
                }
                Ok(best)
            }
            Builtin::Abs => match &v[0] {
                Value::Int(i) => i.checked_abs().map(Value::Int).ok_or_else(|| overflow(sp)),
                Value::Float(x) => Ok(Value::Float(x.abs())),
                other => unreachable!("abs of {other:?}"),
            },
            Builtin::Sqrt => Ok(Value::Float(float(&v[0]).sqrt())),
            Builtin::Exp => Ok(Value::Float(libm::exp(float(&v[0])))),
            Builtin::Ln => Ok(Value::Float(libm::log(float(&v[0])))),
            Builtin::Floor => Ok(Value::Float(float(&v[0]).floor())),
            Builtin::Ceil => Ok(Value::Float(float(&v[0]).ceil())),
            Builtin::Round => Ok(Value::Float(float(&v[0]).round())),
            Builtin::ToFloat => Ok(Value::Float(match &v[0] {
                Value::Int(i) => *i as f64,
                Value::Float(x) => *x,
                other => unreachable!("float of {other:?}"),
            })),
            Builtin::ToInt => match &v[0] {
                Value::Int(i) => Ok(Value::Int(*i)),
                Value::Float(x) => {
                    if !x.is_finite() || x.abs() >= 9.2e18 {
                        return err("R0009", sp, format!("cannot convert {x:?} to i64"));
                    }
                    Ok(Value::Int(x.trunc() as i64))
                }
                other => unreachable!("int of {other:?}"),
            },
            Builtin::ToStr => Ok(Value::Str(self.display(&v[0], true).into())),
            Builtin::Split => {
                let sep = str_of(&v[1]);
                if sep.is_empty() {
                    return err("R0010", sp, "`split` needs a separator that is not empty");
                }
                Ok(str_list(str_of(&v[0]).split(sep)))
            }
            Builtin::Lines => Ok(str_list(str_of(&v[0]).lines())),
            Builtin::Trim => Ok(Value::Str(str_of(&v[0]).trim().into())),
            Builtin::Upper => Ok(Value::Str(str_of(&v[0]).to_uppercase().into())),
            Builtin::Lower => Ok(Value::Str(str_of(&v[0]).to_lowercase().into())),
            Builtin::StartsWith => Ok(Value::Bool(str_of(&v[0]).starts_with(str_of(&v[1])))),
            Builtin::EndsWith => Ok(Value::Bool(str_of(&v[0]).ends_with(str_of(&v[1])))),
            Builtin::Find => {
                let s = str_of(&v[0]);
                Ok(Value::Int(s.find(str_of(&v[1])).map_or(-1, |b| s[..b].chars().count() as i64)))
            }
            Builtin::Replace => Ok(Value::Str(str_of(&v[0]).replace(str_of(&v[1]), str_of(&v[2])).into())),
            Builtin::ParseInt => {
                let s = str_of(&v[0]);
                match s.trim().parse::<i64>() {
                    Ok(n) => Ok(Value::Int(n)),
                    Err(_) => err("R0010", sp, format!("not an integer: {s:?}")),
                }
            }
            Builtin::ParseFloat => Ok(Value::Float(str_of(&v[0]).trim().parse::<f64>().unwrap_or(f64::NAN))),
            Builtin::Join => {
                let Value::List(xs) = &v[0] else { unreachable!("join of a non-list") };
                let sep = str_of(&v[1]);
                let mut out = String::new();
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        out.push_str(sep);
                    }
                    out.push_str(str_of(x));
                }
                Ok(Value::Str(out.into()))
            }
            Builtin::Codes => Ok(Value::List(Rc::new(str_of(&v[0]).chars().map(|c| Value::Int(i64::from(u32::from(c)))).collect()))),
            Builtin::FromCodes => {
                let Value::List(xs) = &v[0] else { unreachable!("from_codes of a non-list") };
                let mut out = String::with_capacity(xs.len());
                for x in xs.iter() {
                    let n = as_int(x.clone());
                    match u32::try_from(n).ok().and_then(char::from_u32) {
                        Some(c) => out.push(c),
                        None => return err("R0010", sp, format!("{n} is not a Unicode code point")),
                    }
                }
                Ok(Value::Str(out.into()))
            }
            Builtin::ReadText => {
                let p = str_of(&v[0]);
                match std::fs::read(p) {
                    Ok(bytes) => Ok(Value::Str(decode_text(bytes).into())),
                    Err(e) => err("R0011", sp, format!("cannot read {p}: {e}")),
                }
            }
            Builtin::WriteText | Builtin::AppendText => {
                let (p, s) = (str_of(&v[0]), str_of(&v[1]));
                let r = if b == Builtin::WriteText {
                    std::fs::write(p, s.as_bytes())
                } else {
                    std::fs::OpenOptions::new().create(true).append(true).open(p).and_then(|mut fh| fh.write_all(s.as_bytes()))
                };
                match r {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => err("R0011", sp, format!("cannot write {p}: {e}")),
                }
            }
            Builtin::FileExists => Ok(Value::Bool(std::path::Path::new(str_of(&v[0])).exists())),
            Builtin::ListDir => {
                let p = str_of(&v[0]);
                match std::fs::read_dir(p) {
                    Ok(rd) => {
                        let mut names: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
                        names.sort();
                        Ok(str_list(names))
                    }
                    Err(e) => err("R0011", sp, format!("cannot list {p}: {e}")),
                }
            }
            Builtin::MakeDir => {
                let p = str_of(&v[0]);
                match std::fs::create_dir_all(p) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => err("R0011", sp, format!("cannot create {p}: {e}")),
                }
            }
            Builtin::Env => Ok(Value::Str(std::env::var(str_of(&v[0])).unwrap_or_default().into())),
            Builtin::Args => Ok(str_list(self.host.args())),
            Builtin::Today => Ok(Value::Int(i64::from(self.host.today_date()))),
            Builtin::HttpGet => match curl(str_of(&v[0]), None) {
                Ok(bytes) => Ok(Value::Str(decode_text(bytes).into())),
                Err(e) => err("R0013", sp, e),
            },
            Builtin::ReadLines => {
                let p = str_of(&v[0]);
                let Value::List(ks) = &v[1] else { unreachable!("read_lines without a list") };
                let mut keep = Vec::with_capacity(ks.len());
                for k in ks.iter() {
                    let Value::Tuple(t) = k else { unreachable!("read_lines keeps (column, text)") };
                    let at = as_int(t[0].clone());
                    let Ok(at) = usize::try_from(at) else { return err("R0010", sp, format!("column {at} is negative")) };
                    keep.push((at, str_of(&t[1]).as_bytes().to_vec()));
                }
                match read_lines_kept(p, &keep) {
                    Ok(ls) => Ok(str_list(ls)),
                    Err(e) => err("R0011", sp, format!("cannot read {p}: {e}")),
                }
            }
            Builtin::BytesNew => Ok(Value::Bytes(Rc::new(Vec::new()))),
            Builtin::BitAnd | Builtin::BitOr | Builtin::BitXor | Builtin::WrappingAdd | Builtin::WrappingMul => {
                let (a, c) = (as_int(v[0].clone()), as_int(v[1].clone()));
                Ok(Value::Int(match b {
                    Builtin::BitAnd => a & c,
                    Builtin::BitOr => a | c,
                    Builtin::BitXor => a ^ c,
                    Builtin::WrappingAdd => a.wrapping_add(c),
                    _ => a.wrapping_mul(c),
                }))
            }
            Builtin::Shl | Builtin::Shr => {
                let (a, n) = (as_int(v[0].clone()), as_int(v[1].clone()));
                if !(0..64).contains(&n) {
                    return err("R0014", sp, format!("shift by {n}: it must be from 0 to 63"));
                }
                Ok(Value::Int(if b == Builtin::Shl { ((a as u64) << n) as i64 } else { ((a as u64) >> n) as i64 }))
            }
            Builtin::ToHex => {
                let (x, w) = (as_int(v[0].clone()), as_int(v[1].clone()).clamp(0, 64) as usize);
                Ok(Value::Str(format!("{:0w$x}", x as u64).into()))
            }
            Builtin::FileSize => Ok(Value::Int(std::fs::metadata(str_of(&v[0])).map_or(-1, |m| m.len() as i64))),
            Builtin::FileDate => {
                let days = std::fs::metadata(str_of(&v[0]))
                    .and_then(|m| m.modified())
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|d| (d.as_secs() / 86_400) as i64);
                Ok(Value::Int(days.map_or(0, |n| i64::from(quant_b3::util::from_days(n)))))
            }
            Builtin::HttpGetAll => {
                let Value::List(us) = &v[0] else { unreachable!("http_get_all of a non-list") };
                let urls: Vec<String> = us.iter().map(|u| str_of(u).to_string()).collect();
                let par = as_int(v[1].clone()).max(1) as usize;
                let res = curl_all(&urls, par);
                Ok(Value::List(Rc::new(
                    res.into_iter()
                        .map(|r| {
                            let (ok, text) = match r {
                                Ok(bytes) => (true, decode_text(bytes)),
                                Err(e) => (false, e),
                            };
                            Value::Tuple(Rc::new(vec![Value::Bool(ok), Value::Str(text.into())]))
                        })
                        .collect(),
                )))
            }
            Builtin::Utf8Len => Ok(Value::Int(str_of(&v[0]).len() as i64)),
            Builtin::GetU8 | Builtin::GetU32 | Builtin::GetU64 | Builtin::GetI64 | Builtin::GetF64 | Builtin::GetF32 | Builtin::GetStr => {
                let Value::Bytes(bs) = &v[0] else { unreachable!("get from non-bytes") };
                let at = as_int(v[1].clone());
                Ok(match b {
                    Builtin::GetU8 => Value::Int(i64::from(bytes_at(bs, at, 1, sp)?[0])),
                    Builtin::GetU32 => Value::Int(i64::from(u32::from_le_bytes(bytes_at(bs, at, 4, sp)?.try_into().expect("4 bytes")))),
                    Builtin::GetU64 => {
                        let u = u64::from_le_bytes(bytes_at(bs, at, 8, sp)?.try_into().expect("8 bytes"));
                        match i64::try_from(u) {
                            Ok(n) => Value::Int(n),
                            Err(_) => return err("R0014", sp, format!("{u} does not fit in i64")),
                        }
                    }
                    Builtin::GetI64 => Value::Int(i64::from_le_bytes(bytes_at(bs, at, 8, sp)?.try_into().expect("8 bytes"))),
                    Builtin::GetF64 => Value::Float(f64::from_le_bytes(bytes_at(bs, at, 8, sp)?.try_into().expect("8 bytes"))),
                    Builtin::GetF32 => Value::Float(f64::from(f32::from_le_bytes(bytes_at(bs, at, 4, sp)?.try_into().expect("4 bytes")))),
                    _ => {
                        let n = as_int(v[2].clone());
                        let Ok(n) = usize::try_from(n) else { return err("R0001", sp, format!("{n} bytes")) };
                        match std::str::from_utf8(bytes_at(bs, at, n, sp)?) {
                            Ok(s) => Value::Str(s.into()),
                            Err(_) => return err("R0010", sp, format!("the {n} bytes at {at} are not UTF-8 text")),
                        }
                    }
                })
            }
            Builtin::WriteBytes => {
                let p = str_of(&v[0]);
                let Value::Bytes(bs) = &v[1] else { unreachable!("write_bytes of non-bytes") };
                match std::fs::write(p, &bs[..]) {
                    Ok(()) => Ok(Value::Unit),
                    Err(e) => err("R0011", sp, format!("cannot write {p}: {e}")),
                }
            }
            Builtin::ReadBytes => {
                let p = str_of(&v[0]);
                match std::fs::read(p) {
                    Ok(bs) => Ok(Value::Bytes(Rc::new(bs))),
                    Err(e) => err("R0011", sp, format!("cannot read {p}: {e}")),
                }
            }
            Builtin::Download => {
                let (url, dest) = (str_of(&v[0]), str_of(&v[1]));
                // into `dest.part`, renamed when complete: an interrupted download leaves no file
                let part = format!("{dest}.part");
                let res = curl(url, Some(&part)).and_then(|_| std::fs::rename(&part, dest).map_err(|e| format!("{dest}: {e}")));
                match res.and_then(|_| std::fs::metadata(dest).map_err(|e| e.to_string())) {
                    Ok(m) => Ok(Value::Int(m.len() as i64)),
                    Err(e) => {
                        let _ = std::fs::remove_file(&part);
                        err("R0013", sp, e)
                    }
                }
            }
            Builtin::Substr => {
                let Value::Str(s) = &v[0] else { unreachable!("substr of a non-string") };
                let a = as_int(v[1].clone()).max(0) as usize;
                let c = as_int(v[2].clone()).max(0) as usize;
                // one byte per character: slice directly (fixed-width files are read this way)
                if s.is_ascii() {
                    let a = a.min(s.len());
                    return Ok(Value::Str(s[a..a.saturating_add(c).min(s.len())].into()));
                }
                Ok(Value::Str(s.chars().skip(a).take(c).collect::<String>().into()))
            }
            Builtin::FormatFixed => {
                let x = float(&v[0]);
                let d = as_int(v[1].clone()).clamp(0, 30) as usize;
                Ok(Value::Str(format!("{x:.d$}").into()))
            }
            Builtin::Assert => {
                if as_bool(v[0].clone()) {
                    Ok(Value::Unit)
                } else {
                    let msg = match v.get(1) {
                        Some(Value::Str(s)) => s.to_string(),
                        _ => "assertion failed".to_string(),
                    };
                    err("R0003", sp, msg)
                }
            }
            other => unreachable!("builtin {other:?} is handled elsewhere"),
        }
    }

    /// Text form of a value; strings are quoted when nested (`top = false`).
    pub fn display(&self, v: &Value, top: bool) -> String {
        let join = |xs: &[Value]| xs.iter().map(|x| self.display(x, false)).collect::<Vec<_>>().join(", ");
        match v {
            Value::Unit => "()".into(),
            Value::Int(i) => i.to_string(),
            Value::Float(x) => format!("{x:?}"),
            Value::Bool(b) => b.to_string(),
            Value::Str(s) => {
                if top {
                    s.to_string()
                } else {
                    format!("{:?}", &**s)
                }
            }
            Value::List(xs) => format!("[{}]", join(xs)),
            Value::Bytes(b) => format!("<{} bytes>", b.len()),
            Value::Map(m) => {
                if m.is_empty() {
                    "[:]".into()
                } else {
                    let body = m
                        .keys()
                        .iter()
                        .zip(m.values())
                        .map(|(k, x)| format!("{}: {}", self.display(k, false), self.display(x, false)))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("[{body}]")
                }
            }
            Value::Tuple(xs) => {
                if xs.len() == 1 {
                    format!("({},)", join(xs))
                } else {
                    format!("({})", join(xs))
                }
            }
            Value::Struct(id, fs) => {
                let def = &self.ck.structs[*id as usize];
                let body = def
                    .fields
                    .iter()
                    .zip(fs.iter())
                    .map(|((n, _), x)| format!("{n}: {}", self.display(x, false)))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{} {{ {body} }}", def.name)
            }
            Value::Enum(e, vi, xs) => {
                let name = &self.ck.enums[*e as usize].variants[*vi as usize].0;
                if xs.is_empty() { name.clone() } else { format!("{name}({})", join(xs)) }
            }
            Value::Func(fv) => match &**fv {
                FuncVal::User(id) => format!("<fn {}>", self.ck.sigs[*id as usize].name),
                FuncVal::Closure { .. } => "<closure>".into(),
                FuncVal::Grad { with_value, .. } => if *with_value { "<value_and_grad>" } else { "<grad>" }.into(),
            },
            Value::Tensor(t) => crate::tensor::describe(t, |a| self.ck.axes[a as usize].name.clone()),
            Value::Node(_) => self.display(&self.fwd(v), top),
            Value::Holdout(p) => format!("<holdout of protocol {:?}>", self.ck.protocols[*p as usize].name),
        }
    }
}

fn matches_pat(p: &HPat, v: &Value, f: &mut [Value]) -> bool {
    match p {
        HPat::Wild => true,
        HPat::Bind(s) => {
            f[*s as usize] = v.clone();
            true
        }
        HPat::Lit(l) => l == v,
        HPat::IntLit(i, _) => *v == Value::Int(*i),
        HPat::Tuple(ps) => match v {
            Value::Tuple(xs) => ps.iter().zip(xs.iter()).all(|(p, x)| matches_pat(p, x, f)),
            _ => false,
        },
        HPat::Variant(_, vi, ps) => match v {
            Value::Enum(_, vj, xs) if vj == vi => ps.iter().zip(xs.iter()).all(|(p, x)| matches_pat(p, x, f)),
            _ => false,
        },
    }
}
