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
use crate::value::{FuncVal, Value, total_cmp};
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
    Index(i64, Span),
}

fn place_mut<'v>(root: &'v mut Value, path: &[Step]) -> R<&'v mut Value> {
    let mut cur = root;
    for st in path {
        cur = match st {
            Step::Field(i) => match cur {
                Value::Struct(_, fs) | Value::Tuple(fs) => &mut Rc::make_mut(fs)[*i as usize],
                _ => unreachable!("field of a non-struct"),
            },
            Step::Index(i, sp) => match cur {
                Value::List(xs) => {
                    let xs = Rc::make_mut(xs);
                    let n = xs.len();
                    if *i < 0 || *i as u64 >= n as u64 {
                        return err("R0001", *sp, format!("index {i} out of bounds for a list of length {n}"));
                    }
                    &mut xs[*i as usize]
                }
                _ => unreachable!("index of a non-list"),
            },
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
            HExpr::Tuple(xs) => Ok(Value::Tuple(Rc::new(self.eval_all(xs, f)?))),
            HExpr::Struct(id, xs) => Ok(Value::Struct(*id, Rc::new(self.eval_all(xs, f)?))),
            HExpr::Variant(en, vi, xs) => Ok(Value::Enum(*en, *vi, Rc::new(self.eval_all(xs, f)?))),
            HExpr::Field(x, i) => match self.eval(x, f)? {
                Value::Struct(_, fs) | Value::Tuple(fs) => Ok(fs[*i as usize].clone()),
                other => unreachable!("field of {other:?}"),
            },
            HExpr::Index(x, i, sp) => {
                let xs = self.eval(x, f)?;
                let i = as_int(self.eval(i, f)?);
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
                *place_mut(&mut f[root], &path)? = v;
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
                let i = as_int(self.eval(ie, f)?);
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
                Value::Str(s) => s.chars().count() as i64,
                other => unreachable!("len of {other:?}"),
            })),
            Builtin::Contains => {
                let Value::List(xs) = &v[0] else { unreachable!() };
                Ok(Value::Bool(xs.contains(&v[1])))
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
