//! Temporal types (M3): every value has a knowledge time, and a decision that uses information
//! from its own future is a compile error.
//!
//! The pass runs on the typed HIR after the checker, as an abstract interpretation:
//! - each value gets the moment it becomes known, `end(i)` for a month (or day) index `i` that the
//!   compiler tracks as a constant or as a symbol plus a constant (the loop variable `m`, an
//!   integer parameter), and the latest of its inputs when values are combined;
//! - data carries its own time: row `t` of `b3.monthly_returns()` and of `b3.cdi()` is known at
//!   `end(t)`, row `t` of `b3.eligible()` at `end(t - 1)` (the NEFIN universe of a year is fixed
//!   with the previous year's liquidity), and `observed(x, lag: k)` declares it for any tensor;
//! - tensors over a temporal axis remember that row `t` is known at `end(t + lag)`; selecting rows
//!   turns that into a time, reducing over the axis (or ranking across it) uses the whole history;
//! - control flow counts: after `if x { continue }` everything depends on `x`, and so does every
//!   assignment inside a branch;
//! - loops are analysed to a fixpoint: what one iteration leaves for the next is one month older.
//!
//! Decisions are checked where they are made:
//! - `b3.evaluate(w)` trades the weights of month `t` at the close of month `t - 1`, so row `t` of
//!   `w` may only use what is known at `end(t - 1)`;
//! - a function declared `-> T @ end(e)` may only return what is known at `end(e)`.
//!
//! E0301: use of future information. E0302: it cannot be proven (the month index is not a loop
//! variable or a parameter plus a constant). E0303: a temporal tensor whose rows are known later
//! than an annotated function may assume.

use crate::ast::{BinOp, UnOp};
use crate::diag::{Diag, KResult, Span};
use crate::hir::*;
use crate::types::{AX_MONTH, Checker, Ty};
use crate::value::Value;
use std::collections::HashMap;

/// A month (or day) index as the compiler sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lin {
    Const(i64),
    /// symbol (loop variable or integer parameter) + constant
    Sym(u32, i64),
    Unknown,
}

impl Lin {
    fn add(self, k: i64) -> Lin {
        match self {
            Lin::Const(c) => c.checked_add(k).map_or(Lin::Unknown, Lin::Const),
            Lin::Sym(s, c) => c.checked_add(k).map_or(Lin::Unknown, |c| Lin::Sym(s, c)),
            Lin::Unknown => Lin::Unknown,
        }
    }
}

fn lin_add(a: Lin, b: Lin) -> Lin {
    match (a, b) {
        (Lin::Const(x), Lin::Const(y)) => Lin::Const(x).add(y),
        (Lin::Sym(s, c), Lin::Const(k)) | (Lin::Const(k), Lin::Sym(s, c)) => Lin::Sym(s, c).add(k),
        _ => Lin::Unknown,
    }
}

fn lin_sub(a: Lin, b: Lin) -> Lin {
    match (a, b) {
        (Lin::Const(x), Lin::Const(y)) => x.checked_sub(y).map_or(Lin::Unknown, Lin::Const),
        (Lin::Sym(s, c), Lin::Const(k)) => k.checked_neg().map_or(Lin::Unknown, |k| Lin::Sym(s, c).add(k)),
        (Lin::Sym(s, c), Lin::Sym(t, d)) if s == t => c.checked_sub(d).map_or(Lin::Unknown, Lin::Const),
        _ => Lin::Unknown,
    }
}

/// The moment a value becomes known.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum At {
    /// known in advance: constants, shapes, the calendar
    Always,
    /// at the end of month (or day) `index`
    End(Lin),
    /// only at the end of the data: it uses the whole history
    Late,
}

thread_local! {
    /// Loop symbols bounded by another symbol: `s ≤ t + c` (for `for s in a..b` whose last value is
    /// `t + c`), so that the time of a value that mixes both is still a month.
    static BOUNDS: std::cell::RefCell<HashMap<u32, (u32, i64)>> = std::cell::RefCell::new(HashMap::new());
}

/// `end(s + x) ⊔ end(t + y)` when `s ≤ t + c`: `end(t + max(y, c + x))`.
fn bounded(s: u32, x: i64, t: u32, y: i64) -> Option<Lin> {
    BOUNDS.with(|b| b.borrow().get(&s).filter(|(u, _)| *u == t).and_then(|(_, c)| c.checked_add(x)).map(|cx| Lin::Sym(t, y.max(cx))))
}

fn join_at(a: At, b: At) -> At {
    use At::*;
    match (a, b) {
        (Always, x) | (x, Always) => x,
        (Late, _) | (_, Late) => Late,
        (End(Lin::Const(x)), End(Lin::Const(y))) => End(Lin::Const(x.max(y))),
        (End(Lin::Sym(s, x)), End(Lin::Sym(t, y))) if s == t => End(Lin::Sym(s, x.max(y))),
        (End(Lin::Sym(s, x)), End(Lin::Sym(t, y))) => match bounded(s, x, t, y).or_else(|| bounded(t, y, s, x)) {
            Some(l) => End(l),
            None => End(Lin::Unknown),
        },
        _ => End(Lin::Unknown),
    }
}

/// Is `a` known no later than `b`? None: it cannot be decided.
fn known_by(a: At, b: At) -> Option<bool> {
    use At::*;
    match (a, b) {
        (Always, _) => Some(true),
        (_, Always) => Some(false),
        (Late, _) => Some(false),
        (_, Late) => Some(true),
        (End(Lin::Const(x)), End(Lin::Const(y))) => Some(x <= y),
        (End(Lin::Sym(s, x)), End(Lin::Sym(t, y))) if s == t => Some(x <= y),
        _ => None,
    }
}

/// A time and the code that made it that late (for messages).
#[derive(Clone, Copy, Debug)]
pub struct Time {
    pub at: At,
    pub span: Option<Span>,
}

impl Time {
    const ALWAYS: Time = Time { at: At::Always, span: None };

    fn new(at: At, span: Span) -> Time {
        Time { at, span: Some(span) }
    }

    fn join(self, o: Time) -> Time {
        let at = join_at(self.at, o.at);
        let span = if at == self.at && self.span.is_some() {
            self.span
        } else if at == o.at && o.span.is_some() {
            o.span
        } else {
            self.span.or(o.span)
        };
        Time { at, span }
    }

    fn map(self, f: impl Fn(Lin) -> Lin) -> Time {
        match self.at {
            At::End(l) => Time { at: At::End(f(l)), span: self.span },
            _ => self,
        }
    }

    /// Replaces a function's data horizon `sym` by the time it stands for at a call.
    fn horizon(self, sym: u32, to: At) -> Time {
        match self.at {
            At::End(Lin::Sym(s, c)) if s == sym => {
                let at = match to {
                    At::End(l) => At::End(l.add(c)),
                    other => other,
                };
                Time { at, span: self.span }
            }
            _ => self,
        }
    }
}

/// Rows of a tensor over a temporal axis: row `t` is known at `end(t + lag)`.
#[derive(Clone, Copy, Debug)]
pub struct Rows {
    pub axis: u32,
    /// None: it cannot be determined
    pub lag: Option<i64>,
    /// the index of the last row when it is a function's data horizon (a symbol replaced, at each
    /// call, by what the caller passed); None: the end of the data
    pub end: Option<Lin>,
    /// where the latest rows come from (for messages)
    pub span: Option<Span>,
    /// the store that set the lag: row index and the time of the stored value (for messages)
    pub store: Option<(Lin, At)>,
}

fn join_rows(a: Option<Rows>, b: Option<Rows>) -> Option<Rows> {
    match (a, b) {
        (None, x) | (x, None) => x,
        (Some(x), Some(y)) => {
            let end = if x.end == y.end { x.end } else { None };
            if x.axis != y.axis {
                return Some(Rows { lag: None, end: None, ..x });
            }
            let pick = match (x.lag, y.lag) {
                (Some(p), Some(q)) => {
                    if q > p {
                        y
                    } else {
                        x
                    }
                }
                (None, _) => x,
                (_, None) => y,
            };
            Some(Rows { end, ..pick })
        }
    }
}

fn same_rows(a: Option<Rows>, b: Option<Rows>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => x.axis == y.axis && x.lag == y.lag && x.end == y.end,
        _ => false,
    }
}

/// What the analysis knows about a value.
#[derive(Clone, Copy, Debug)]
pub struct TV {
    pub time: Time,
    pub rows: Option<Rows>,
    /// integer values: the index as a linear form
    pub int: Lin,
}

impl TV {
    pub const ALWAYS: TV = TV { time: Time::ALWAYS, rows: None, int: Lin::Unknown };

    fn int(l: Lin) -> TV {
        TV { int: l, ..TV::ALWAYS }
    }

    fn join(&self, o: &TV) -> TV {
        TV {
            time: self.time.join(o.time),
            rows: join_rows(self.rows, o.rows),
            int: if self.int == o.int { self.int } else { Lin::Unknown },
        }
    }

    fn with_time(mut self, t: Time) -> TV {
        self.time = self.time.join(t);
        self
    }

    fn no_int(mut self) -> TV {
        self.int = Lin::Unknown;
        self
    }

    fn same(&self, o: &TV) -> bool {
        self.time.at == o.time.at && self.int == o.int && same_rows(self.rows, o.rows)
    }

    /// The value one iteration later, seen from the next value of loop symbol `sym`.
    fn shift(self, sym: u32, d: i64) -> TV {
        let f = |l: Lin| match l {
            Lin::Sym(s, c) if s == sym => Lin::Sym(s, c).add(d),
            x => x,
        };
        TV { time: self.time.map(f), rows: self.rows, int: f(self.int) }
    }

    /// After the loop: symbol `sym` stands for an unknown month.
    fn forget(self, sym: u32) -> TV {
        let f = |l: Lin| match l {
            Lin::Sym(s, _) if s == sym => Lin::Unknown,
            x => x,
        };
        let rows = self.rows.map(|r| match r.end {
            Some(Lin::Sym(s, _)) if s == sym => Rows { end: None, ..r },
            _ => r,
        });
        TV { time: self.time.map(f), rows, int: f(self.int) }
    }

    /// After a `for` loop whose variable `sym` ends at `last`: what the iterations left is known
    /// by the time of the last one (times only grow with the loop variable).
    fn bound(self, sym: u32, last: Lin) -> TV {
        let f = |l: Lin| match l {
            Lin::Sym(s, c) if s == sym => last.add(c),
            x => x,
        };
        let int = match self.int {
            Lin::Sym(s, _) if s == sym => Lin::Unknown,
            x => x,
        };
        TV { time: self.time.map(f), rows: self.rows, int }
    }
}

/// The analysis state kept across REPL inputs: the top-level frame.
#[derive(Clone, Default, Debug)]
pub struct State {
    pub main: Vec<TV>,
}

/// Checks one batch: the new functions (annotations, decisions inside them) and the new
/// top-level statements, continuing from the state of the previous batches.
pub fn check(ck: &Checker, new_fns: &[u32], top: &[&HExpr], state: &mut State) -> KResult<()> {
    let mut an = An { ck, names: Vec::new(), summaries: HashMap::new(), pc: Time::ALWAYS, ret: None, in_ctx: Vec::new() };
    for &id in new_fns {
        an.summary(id)?;
    }
    state.main.resize(ck.main_slots() as usize, TV::ALWAYS);
    for e in top {
        an.pc = Time::ALWAYS;
        an.eval(e, &mut state.main)?;
    }
    Ok(())
}

#[derive(Clone)]
struct Summary {
    /// the result in terms of the parameters' symbols
    result: TV,
    /// symbol of each integer parameter
    param_syms: Vec<Option<u32>>,
    /// the last row of the temporal parameters (one symbol for all of them: at a call it stands
    /// for the latest of the temporal arguments)
    horizon: Option<u32>,
}

enum ParamKind {
    Int,
    Temporal(u32),
    Other,
}

struct An<'a> {
    ck: &'a Checker,
    /// symbol names (loop variables, parameters), for messages
    names: Vec<String>,
    summaries: HashMap<u32, Option<Summary>>,
    /// control dependence: when the decision to be here was made
    pc: Time,
    /// values returned so far by the function being analysed
    ret: Option<TV>,
    /// functions being analysed in the context of a call (and how deep that goes)
    in_ctx: Vec<u32>,
}

/// How deep calls are analysed in their context before falling back to summaries.
const MAX_CTX: usize = 6;

fn axis_lit(a: &HArg) -> Option<i64> {
    match a {
        HArg::Val(HExpr::Lit(Value::Int(x))) => Some(*x),
        _ => None,
    }
}

fn children<'e>(e: &'e HExpr, out: &mut Vec<&'e HExpr>) {
    fn place<'e>(p: &'e HPlace, out: &mut Vec<&'e HExpr>) {
        match p {
            HPlace::Local(_) => {}
            HPlace::Field(inner, _) => place(inner, out),
            HPlace::Index(inner, i, _) => {
                place(inner, out);
                out.push(i);
            }
        }
    }
    fn sels<'e>(ss: &'e [(u32, HSel)], out: &mut Vec<&'e HExpr>) {
        for (_, s) in ss {
            match s {
                HSel::At(i) => out.push(i),
                HSel::Range(a, b, _) => {
                    out.push(a);
                    out.push(b);
                }
            }
        }
    }
    match e {
        HExpr::Lit(_) | HExpr::IntLit(..) | HExpr::Local(_) | HExpr::FnRef(_) | HExpr::Break | HExpr::Continue | HExpr::Lambda(..) => {}
        HExpr::List(xs) | HExpr::MapLit(xs) | HExpr::Tuple(xs) | HExpr::Struct(_, xs) | HExpr::Variant(_, _, xs) | HExpr::Host(_, xs, _) => out.extend(xs.iter()),
        HExpr::Field(x, _) | HExpr::Unary(_, x, _) => out.push(x),
        HExpr::Index(a, b, _) | HExpr::Binary(_, a, b, _) | HExpr::And(a, b) | HExpr::Or(a, b) | HExpr::While(a, b) => {
            out.push(a);
            out.push(b);
        }
        HExpr::TensorIndex(x, ss, _) => {
            out.push(x);
            sels(ss, out);
        }
        HExpr::TensorAssign(p, ss, _, v, _) => {
            place(p, out);
            sels(ss, out);
            out.push(v);
        }
        HExpr::CallFn(_, args, _) | HExpr::CallBuiltin(_, args, _) => {
            for a in args {
                match a {
                    HArg::Val(x) => out.push(x),
                    HArg::InOut(p) => place(p, out),
                }
            }
        }
        HExpr::CallValue(c, args, _) => {
            out.push(c);
            out.extend(args.iter());
        }
        HExpr::Assign(p, v) | HExpr::OpAssign(p, _, v, _) => {
            place(p, out);
            out.push(v);
        }
        HExpr::If(c, t, e) => {
            out.push(c);
            out.push(t);
            if let Some(x) = e {
                out.push(x);
            }
        }
        HExpr::ForRange { lo, hi, body, .. } => {
            out.push(lo);
            out.push(hi);
            out.push(body);
        }
        HExpr::ForList { list, body, .. } => {
            out.push(list);
            out.push(body);
        }
        HExpr::Match(s, arms, _) => {
            out.push(s);
            for (_, b) in arms {
                out.push(b);
            }
        }
        HExpr::Block(st, tail) => {
            out.extend(st.iter());
            if let Some(t) = tail {
                out.push(t);
            }
        }
        HExpr::Let(_, init) => out.push(init),
        HExpr::Return(v) => {
            if let Some(x) = v {
                out.push(x);
            }
        }
    }
}

/// Can `e` leave the code that follows it (break/continue of the enclosing loop, or return)?
/// With `breaks_only`, `continue` does not count.
fn jumps(e: &HExpr, depth: u32, breaks_only: bool) -> bool {
    match e {
        HExpr::Return(_) => true,
        HExpr::Break => depth == 0,
        HExpr::Continue => depth == 0 && !breaks_only,
        HExpr::Lambda(..) => false,
        HExpr::While(..) | HExpr::ForRange { .. } | HExpr::ForList { .. } => {
            let mut cs = Vec::new();
            children(e, &mut cs);
            cs.iter().any(|c| jumps(c, depth + 1, breaks_only))
        }
        _ => {
            let mut cs = Vec::new();
            children(e, &mut cs);
            cs.iter().any(|c| jumps(c, depth, breaks_only))
        }
    }
}

/// The function a call goes to, when it is known: `f(...)` through a value, `grad(f)(...)`.
fn static_fn(c: &HExpr) -> Option<u32> {
    match c {
        HExpr::FnRef(id) => Some(*id),
        HExpr::CallBuiltin(Builtin::Grad | Builtin::ValueAndGrad, a, _) => match a.first() {
            Some(HArg::Val(HExpr::FnRef(id))) => Some(*id),
            _ => None,
        },
        _ => None,
    }
}

fn root(p: &HPlace) -> usize {
    match p {
        HPlace::Local(s) => *s as usize,
        HPlace::Field(i, _) | HPlace::Index(i, _, _) => root(i),
    }
}

fn join_frames(a: &[TV], b: &[TV]) -> Vec<TV> {
    a.iter().zip(b).map(|(x, y)| x.join(y)).collect()
}

fn same_frames(a: &[TV], b: &[TV]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.same(y))
}

const MAX_ROUNDS: usize = 12;

impl<'a> An<'a> {
    fn new_sym(&mut self, name: &str) -> u32 {
        self.names.push(name.to_string());
        (self.names.len() - 1) as u32
    }

    fn show_lin(&self, l: Lin) -> String {
        match l {
            Lin::Const(c) => format!("{c}"),
            Lin::Sym(s, c) => {
                let n = &self.names[s as usize];
                match c {
                    0 => n.clone(),
                    c if c > 0 => format!("{n} + {c}"),
                    c => format!("{n} - {}", c.unsigned_abs()),
                }
            }
            Lin::Unknown => "?".into(),
        }
    }

    fn show_at(&self, a: At) -> String {
        match a {
            At::Always => "known in advance".into(),
            At::End(Lin::Unknown) => "at the end of a month the compiler cannot determine".into(),
            At::End(l) => format!("at end({})", self.show_lin(l)),
            At::Late => "only at the end of the data (it uses the whole history, or a file)".into(),
        }
    }

    fn is_temporal(&self, axis: u32) -> bool {
        self.ck.axes.get(axis as usize).is_some_and(|a| a.temporal.is_some())
    }

    fn param_kind(&self, t: &Ty) -> ParamKind {
        match t {
            Ty::Int => ParamKind::Int,
            Ty::Tensor(axes, _) => match axes.iter().find(|a| self.is_temporal(**a)) {
                Some(a) => ParamKind::Temporal(*a),
                None => ParamKind::Other,
            },
            _ => ParamKind::Other,
        }
    }

    /// Stores happen under the current control dependence.
    fn under_pc(&self, v: TV) -> TV {
        v.with_time(self.pc)
    }

    // ---------------------------------------------------------------- functions

    fn summary(&mut self, id: u32) -> KResult<Summary> {
        match self.summaries.get(&id) {
            Some(Some(s)) => return Ok(s.clone()),
            // recursion: assume the worst
            Some(None) => {
                return Ok(Summary { result: TV { time: Time { at: At::Late, span: None }, ..TV::ALWAYS }, param_syms: Vec::new(), horizon: None });
            }
            None => {}
        }
        self.summaries.insert(id, None);
        let ck = self.ck;
        let sig = &ck.sigs[id as usize];
        let body = ck.bodies[id as usize].as_ref().expect("checked body");
        let mut frame = vec![TV::ALWAYS; body.n_slots as usize];
        let mut param_syms = Vec::new();
        let temporal = sig.params.iter().any(|(_, ty, _)| matches!(self.param_kind(ty), ParamKind::Temporal(_)));
        let horizon = if temporal { Some(self.new_sym(&format!("the last month given to `{}`", sig.name))) } else { None };
        for (i, (name, ty, _)) in sig.params.iter().enumerate() {
            match self.param_kind(ty) {
                ParamKind::Int => {
                    let s = self.new_sym(name);
                    frame[i] = TV::int(Lin::Sym(s, 0));
                    param_syms.push(Some(s));
                }
                ParamKind::Temporal(axis) => {
                    // data passed in: row t known at end(t) (checked at the call sites of
                    // annotated functions, accounted for at the others)
                    let end = horizon.map(|h| Lin::Sym(h, 0));
                    frame[i] = TV { rows: Some(Rows { axis, lag: Some(0), end, span: None, store: None }), ..TV::ALWAYS };
                    param_syms.push(None);
                }
                ParamKind::Other => param_syms.push(None),
            }
        }
        let saved_pc = std::mem::replace(&mut self.pc, Time::ALWAYS);
        let saved_ret = self.ret.take();
        let r = self.eval(&body.body, &mut frame);
        let pc = std::mem::replace(&mut self.pc, saved_pc);
        let ret = std::mem::replace(&mut self.ret, saved_ret);
        let mut result = r?.with_time(pc);
        if let Some(rv) = ret {
            result = result.join(&rv);
        }
        if let Some((ann, asp)) = &body.known_at {
            let bound = self.eval(ann, &mut frame)?.int;
            self.check_bound(&result, bound, &sig.name, *asp)?;
        }
        let s = Summary { result, param_syms, horizon };
        self.summaries.insert(id, Some(s.clone()));
        Ok(s)
    }

    /// `-> T @ end(bound)`: the result must be known by the end of `bound`.
    fn check_bound(&self, result: &TV, bound: Lin, fname: &str, asp: Span) -> KResult<()> {
        if matches!(bound, Lin::Unknown) {
            return Err(Diag::new("E0302", asp, "cannot use this time in a temporal annotation")
                .hint("write a parameter plus a constant, e.g. `@ end(m - 1)`"));
        }
        let at = if result.rows.is_some() { At::Late } else { result.time.at };
        let b = At::End(bound);
        match known_by(at, b) {
            Some(true) => Ok(()),
            Some(false) => {
                let span = result.time.span.or(result.rows.and_then(|r| r.span)).unwrap_or(asp);
                Err(Diag::new("E0301", span, "use of future information")
                    .label(format!("known {}, but `{fname}` decides at end({})", self.show_at(at), self.show_lin(bound)))
                    .hint(format!("`{fname}` promises (`@ end({})`) to use only what is known by then", self.show_lin(bound))))
            }
            None => {
                let span = result.time.span.unwrap_or(asp);
                Err(Diag::new("E0302", span, "cannot prove that this is known in time")
                    .label(format!("known {}; `{fname}` decides at end({})", self.show_at(at), self.show_lin(bound)))
                    .hint("index temporal axes with a parameter or the loop variable plus a constant, e.g. `r[month: m - 1]`"))
            }
        }
    }

    fn call_fn(&mut self, id: u32, args: &[HArg], f: &mut Vec<TV>) -> KResult<TV> {
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(match a {
                HArg::Val(e) => self.eval(e, f)?,
                HArg::InOut(p) => self.load_place(p, f)?,
            });
        }
        // a temporal argument whose rows are known earlier (or later) than the summary assumes
        // (row t at end(t)): analyse the body with what was passed, as if it were inlined
        let in_progress = matches!(self.summaries.get(&id), Some(None)) || self.in_ctx.contains(&id);
        if !in_progress && self.in_ctx.len() < MAX_CTX {
            let ck = self.ck;
            let sig = &ck.sigs[id as usize];
            let shifted = sig.params.iter().enumerate().any(|(i, (_, ty, _))| {
                matches!(self.param_kind(ty), ParamKind::Temporal(_)) && vals[i].rows.is_some_and(|r| r.lag != Some(0))
            });
            let annotated = ck.bodies[id as usize].as_ref().is_some_and(|b| b.known_at.is_some());
            if shifted && !annotated {
                return self.in_context(id, &vals, args, f);
            }
        }
        let s = self.summary(id)?;
        let ck = self.ck;
        let sig = &ck.sigs[id as usize];
        let annotated = ck.bodies[id as usize].as_ref().is_some_and(|b| b.known_at.is_some());
        // the body assumed row t of its temporal tensors known at end(t)
        let mut shift = 0i64;
        let mut lost = false;
        for (i, (pname, ty, _)) in sig.params.iter().enumerate() {
            if let ParamKind::Temporal(_) = self.param_kind(ty) {
                if let Some(r) = vals[i].rows {
                    match r.lag {
                        Some(l) => shift = shift.max(l),
                        None => lost = true,
                    }
                    if annotated && r.lag.is_none_or(|l| l > 0) {
                        let span = match &args[i] {
                            HArg::Val(e) => self.span_of(e).or(r.span),
                            HArg::InOut(_) => r.span,
                        };
                        let when = r.lag.map_or("at a time the compiler cannot determine".to_string(), |l| format!("at end(t + {l})"));
                        return Err(Diag::new("E0303", span.or(r.span).unwrap_or_else(|| self.any_span(args)), "this data is known later than the function assumes")
                            .label(format!("row t of this tensor is known {when}, but `{}` reads parameter `{pname}` as known at end(t)", sig.name))
                            .hint("pass data whose row t is known at the end of t, or shift the decision time"));
                    }
                }
            }
        }
        // the result in terms of the arguments
        let subst = |l: Lin| match l {
            Lin::Sym(sym, c) => match s.param_syms.iter().position(|p| *p == Some(sym)) {
                Some(i) => vals[i].int.add(c),
                None => l,
            },
            x => x,
        };
        let mut res = TV { time: s.result.time.map(subst), rows: s.result.rows, int: Lin::Unknown };
        if shift > 0 {
            // (times relative to the horizon already count the arguments' lag)
            res.time = res.time.map(|l| match l {
                Lin::Sym(x, _) if Some(x) == s.horizon => l,
                _ => l.add(shift),
            });
            if let Some(r) = res.rows.as_mut() {
                r.lag = r.lag.map(|l| l + shift);
            }
        }
        if let Some(h) = s.horizon {
            // the latest temporal argument: when its last row is known
            let mut to = At::Always;
            let mut end: Option<Option<Lin>> = None;
            for (i, (_, ty, _)) in sig.params.iter().enumerate() {
                if let ParamKind::Temporal(_) = self.param_kind(ty) {
                    let (at, e) = match vals[i].rows {
                        Some(r) => match (r.end, r.lag) {
                            (Some(e), Some(l)) => (At::End(e.add(l)), Some(e)),
                            _ => (At::Late, None),
                        },
                        // a slice or a snapshot: all of it is known at its own time
                        None => (vals[i].time.at, None),
                    };
                    to = join_at(to, at);
                    end = Some(match end {
                        None => e,
                        Some(prev) if prev == e => e,
                        Some(_) => None,
                    });
                }
            }
            res.time = res.time.horizon(h, to);
            if let Some(r) = res.rows.as_mut() {
                if matches!(r.end, Some(Lin::Sym(x, _)) if x == h) {
                    r.end = end.flatten();
                }
            }
        }
        if lost {
            res.time = res.time.join(Time { at: At::Late, span: None });
        }
        for v in &vals {
            res.time = res.time.join(v.time);
        }
        // `inout` arguments may have been changed with anything the function saw
        for (a, _) in args.iter().zip(&vals) {
            if let HArg::InOut(p) = a {
                let r = root(p);
                f[r] = f[r].join(&res).no_int();
            }
        }
        Ok(res)
    }

    /// The result of a call, from the body analysed with the arguments' own knowledge times.
    fn in_context(&mut self, id: u32, vals: &[TV], args: &[HArg], f: &mut Vec<TV>) -> KResult<TV> {
        let ck = self.ck;
        let body = ck.bodies[id as usize].as_ref().expect("checked body");
        let mut frame = vec![TV::ALWAYS; body.n_slots as usize];
        for (i, v) in vals.iter().enumerate() {
            frame[i] = *v;
        }
        self.in_ctx.push(id);
        let saved_pc = std::mem::replace(&mut self.pc, Time::ALWAYS);
        let saved_ret = self.ret.take();
        let r = self.eval(&body.body, &mut frame);
        let pc = std::mem::replace(&mut self.pc, saved_pc);
        let ret = std::mem::replace(&mut self.ret, saved_ret);
        self.in_ctx.pop();
        let mut res = r?.with_time(pc);
        if let Some(rv) = ret {
            res = res.join(&rv);
        }
        // `inout` arguments may have been changed with anything the function saw
        for a in args {
            if let HArg::InOut(p) = a {
                let r = root(p);
                f[r] = f[r].join(&res).no_int();
            }
        }
        Ok(res.no_int())
    }

    fn span_of(&self, e: &HExpr) -> Option<Span> {
        match e {
            HExpr::TensorIndex(_, _, sp) | HExpr::Host(_, _, sp) | HExpr::CallFn(_, _, sp) | HExpr::CallBuiltin(_, _, sp) => Some(*sp),
            _ => None,
        }
    }

    fn any_span(&self, args: &[HArg]) -> Span {
        args.iter()
            .find_map(|a| match a {
                HArg::Val(e) => self.span_of(e),
                HArg::InOut(_) => None,
            })
            .unwrap_or(Span::new(0, 0, 0))
    }

    // ---------------------------------------------------------------- places

    fn index_time(&mut self, p: &HPlace, f: &mut Vec<TV>) -> KResult<Time> {
        Ok(match p {
            HPlace::Local(_) => Time::ALWAYS,
            HPlace::Field(i, _) => self.index_time(i, f)?,
            HPlace::Index(i, idx, _) => {
                let a = self.index_time(i, f)?;
                a.join(self.eval(idx, f)?.time)
            }
        })
    }

    fn load_place(&mut self, p: &HPlace, f: &mut Vec<TV>) -> KResult<TV> {
        let t = self.index_time(p, f)?;
        let v = f[root(p)];
        Ok(match p {
            HPlace::Local(_) => v,
            _ => v.with_time(t).no_int(),
        })
    }

    fn store(&mut self, p: &HPlace, v: TV, f: &mut Vec<TV>) -> KResult<()> {
        let t = self.index_time(p, f)?;
        let v = self.under_pc(v).with_time(t);
        match p {
            HPlace::Local(s) => f[*s as usize] = v,
            _ => {
                let r = root(p);
                f[r] = f[r].join(&v).no_int();
            }
        }
        Ok(())
    }

    fn bind(&self, p: &HPat, v: TV, f: &mut [TV]) {
        match p {
            HPat::Bind(s) => f[*s as usize] = v,
            HPat::Tuple(ps) | HPat::Variant(_, _, ps) => {
                for q in ps {
                    self.bind(q, v.no_int(), f);
                }
            }
            HPat::Wild | HPat::Lit(_) | HPat::IntLit(..) => {}
        }
    }

    // ---------------------------------------------------------------- tensors

    /// Selected index: (time of the index value, row of a temporal axis as a linear form).
    fn sel(&mut self, s: &HSel, f: &mut Vec<TV>) -> KResult<(Time, Lin, Lin)> {
        Ok(match s {
            HSel::At(i) => {
                let v = self.eval(i, f)?;
                (v.time, v.int, v.int)
            }
            HSel::Range(lo, hi, inclusive) => {
                let a = self.eval(lo, f)?;
                let b = self.eval(hi, f)?;
                let last = if *inclusive { b.int } else { b.int.add(-1) };
                (a.time.join(b.time), a.int, last)
            }
        })
    }

    fn tensor_index(&mut self, x: &HExpr, sels: &[(u32, HSel)], sp: Span, f: &mut Vec<TV>) -> KResult<TV> {
        let base = self.eval(x, f)?;
        let mut out = base.no_int();
        for (axis, s) in sels {
            let (t, _first, last) = self.sel(s, f)?;
            out.time = out.time.join(t);
            if let Some(r) = base.rows.filter(|r| r.axis == *axis) {
                let at = match r.lag {
                    Some(l) => At::End(last.add(l)),
                    None => At::End(Lin::Unknown),
                };
                out.time = out.time.join(Time::new(at, sp));
                out.rows = None;
            }
        }
        Ok(out)
    }

    fn tensor_assign(&mut self, p: &HPlace, sels: &[(u32, HSel)], v: &HExpr, sp: Span, f: &mut Vec<TV>) -> KResult<()> {
        let stored = self.eval(v, f)?;
        let mut val = self.under_pc(stored);
        let r = root(p);
        let mut target = f[r];
        let idx_time = self.index_time(p, f)?;
        val = val.with_time(idx_time);
        let mut row: Option<(u32, Lin)> = None;
        for (axis, s) in sels {
            let (t, first, _) = self.sel(s, f)?;
            val = val.with_time(t);
            if self.is_temporal(*axis) {
                // the latest row of a range written at once is its first one, relatively
                row = Some((*axis, first));
            }
        }
        match row {
            Some((axis, idx)) => {
                // row `idx` gets a value known at `val.time`: lag = time - row
                let lag = match (val.time.at, idx) {
                    (At::Always, _) => None,
                    // known only at the end of the data: too late for any month but the last
                    (At::Late, _) => {
                        target.time = target.time.join(val.time);
                        None
                    }
                    (At::End(Lin::Sym(s, d)), Lin::Sym(t, c)) if s == t => Some(Some(d - c)),
                    (At::End(Lin::Const(d)), Lin::Const(c)) => Some(Some(d - c)),
                    _ => Some(None),
                };
                if let Some(lag) = lag {
                    let rows = Rows { axis, lag, end: None, span: val.time.span.or(Some(sp)), store: Some((idx, val.time.at)) };
                    target.rows = join_rows(target.rows, Some(rows));
                }
                // a stored tensor that still has a temporal axis brings its rows along
                target.rows = join_rows(target.rows, val.rows);
            }
            None => {
                // no temporal row selected: the value reaches every row
                target.time = target.time.join(val.time);
                target.rows = join_rows(target.rows, val.rows);
            }
        }
        f[r] = target.no_int();
        Ok(())
    }

    /// Reducing (or ranking, or contracting) over the temporal axis uses the whole history.
    fn over_time(&self, x: TV, axis: Option<i64>, sp: Span) -> TV {
        match x.rows {
            Some(r) if axis.is_none_or(|a| a < 0 || a == i64::from(r.axis)) => {
                // known when its last row is: a function's data horizon, or the end of the data
                let at = match (r.end, r.lag) {
                    (Some(e), Some(l)) => At::End(e.add(l)),
                    _ => At::Late,
                };
                TV { time: x.time.join(Time::new(at, sp)), rows: None, int: Lin::Unknown }
            }
            _ => x.no_int(),
        }
    }

    fn val(&mut self, a: &HArg, f: &mut Vec<TV>) -> KResult<TV> {
        match a {
            HArg::Val(e) => self.eval(e, f),
            HArg::InOut(p) => self.load_place(p, f),
        }
    }

    fn builtin(&mut self, b: Builtin, args: &[HArg], sp: Span, f: &mut Vec<TV>) -> KResult<TV> {
        use Builtin as B;
        match b {
            // removing a key: which keys are left depends on it
            B::Push | B::Remove | B::PutU8 | B::PutU32 | B::PutU64 | B::PutI64 | B::PutF64 | B::PutStr => {
                let x = self.val(&args[1], f)?;
                let HArg::InOut(p) = &args[0] else { unreachable!("push without a place") };
                self.store_weak(p, x, f)?;
                Ok(TV::ALWAYS)
            }
            B::Pop => {
                let HArg::InOut(p) = &args[0] else { unreachable!("pop without a place") };
                Ok(self.load_place(p, f)?.no_int())
            }
            B::Sort | B::Reverse => Ok(TV::ALWAYS),
            B::Print | B::Assert => {
                for a in args {
                    self.val(a, f)?;
                }
                Ok(TV::ALWAYS)
            }
            B::ToInt => {
                let v = self.val(&args[0], f)?;
                Ok(v)
            }
            B::Size => {
                // the number of rows of a temporal parameter is its horizon + 1
                let x = self.val(&args[0], f)?;
                let int = match (x.rows, args.get(1).and_then(axis_lit)) {
                    (Some(r), Some(a)) if i64::from(r.axis) == a => r.end.map_or(Lin::Unknown, |e| e.add(1)),
                    _ => Lin::Unknown,
                };
                Ok(TV { int, ..TV::ALWAYS })
            }
            B::Zeros | B::Ones | B::ZerosLike => {
                for a in args {
                    self.val(a, f)?;
                }
                Ok(TV::ALWAYS)
            }
            B::Full | B::FullLike => {
                let k = if b == B::Full { 0 } else { 1 };
                let v = self.val(&args[k], f)?;
                Ok(TV { time: v.time, rows: None, int: Lin::Unknown })
            }
            B::Tensor => {
                let v = self.val(&args[0], f)?;
                Ok(TV { time: v.time, rows: None, int: Lin::Unknown })
            }
            B::ToList => {
                let v = self.val(&args[0], f)?;
                Ok(self.over_time(v, None, sp))
            }
            B::Sum | B::Mean | B::Prod | B::Max | B::Min | B::NanSum | B::NanMean | B::NanProd | B::NanCount | B::Count => {
                let x = self.val(&args[0], f)?;
                match args.get(1).and_then(axis_lit) {
                    Some(ax) if args.len() == 2 => Ok(self.over_time(x, Some(ax), sp)),
                    _ => {
                        let mut t = x.no_int();
                        for a in &args[1..] {
                            t = t.join(&self.val(a, f)?).no_int();
                        }
                        Ok(t)
                    }
                }
            }
            B::Rank => {
                let x = self.val(&args[0], f)?;
                Ok(self.over_time(x, args.get(1).and_then(axis_lit), sp))
            }
            B::NanQuantile => {
                let x = self.val(&args[0], f)?;
                let q = self.val(&args[1], f)?;
                Ok(self.over_time(x, args.get(2).and_then(axis_lit), sp).with_time(q.time))
            }
            // the days lose their months: reaching the result reaches every row
            B::MergeDays => {
                let x = self.val(&args[0], f)?;
                Ok(self.over_time(x, Some(i64::from(AX_MONTH)), sp))
            }
            B::TopK | B::BottomK => {
                let x = self.val(&args[0], f)?;
                let k = self.val(&args[1], f)?;
                Ok(self.over_time(x, args.get(2).and_then(axis_lit), sp).with_time(k.time))
            }
            B::Contract => {
                let x = self.val(&args[0], f)?;
                let y = self.val(&args[1], f)?;
                let ax = args.get(2).and_then(axis_lit);
                let (x, y) = (self.over_time(x, ax, sp), self.over_time(y, ax, sp));
                Ok(x.join(&y).no_int())
            }
            // what is in a file has no known time: it cannot reach a decision unless declared
            // with `observed` (configuration and market data look the same to the compiler)
            B::ReadText | B::FileExists | B::ListDir | B::HttpGet | B::Download | B::ReadLines | B::ReadBytes => {
                for a in args {
                    self.val(a, f)?;
                }
                Ok(TV { time: Time::new(At::Late, sp), rows: None, int: Lin::Unknown })
            }
            B::Observed => {
                let x = self.val(&args[0], f)?;
                let axis = args.get(1).and_then(axis_lit).unwrap_or(i64::from(AX_MONTH)) as u32;
                let lag = args.get(2).and_then(axis_lit).unwrap_or(0);
                Ok(TV { time: x.time, rows: Some(Rows { axis, lag: Some(lag), end: None, span: Some(sp), store: None }), int: Lin::Unknown })
            }
            _ => {
                // element-wise and list functions: the latest of the inputs
                let mut t = TV::ALWAYS;
                for a in args {
                    t = t.join(&self.val(a, f)?);
                }
                Ok(t.no_int())
            }
        }
    }

    fn store_weak(&mut self, p: &HPlace, v: TV, f: &mut Vec<TV>) -> KResult<()> {
        let t = self.index_time(p, f)?;
        let v = self.under_pc(v).with_time(t);
        let r = root(p);
        f[r] = f[r].join(&v).no_int();
        Ok(())
    }

    // ---------------------------------------------------------------- the B3 module

    fn host(&mut self, h: HostFn, args: &[HExpr], sp: Span, f: &mut Vec<TV>) -> KResult<TV> {
        let mut vals = Vec::with_capacity(args.len());
        for a in args {
            vals.push(self.eval(a, f)?);
        }
        let data = |lag: i64| TV { time: Time::ALWAYS, rows: Some(Rows { axis: AX_MONTH, lag: Some(lag), end: None, span: Some(sp), store: None }), int: Lin::Unknown };
        Ok(match h {
            // row t: the total return of month t, known when the month closes
            HostFn::MonthlyReturns => data(0),
            // row t: the NEFIN universe of t's year, fixed with the previous year's liquidity
            HostFn::Eligible => data(-1),
            HostFn::Cdi | HostFn::Ipca => data(0),
            HostFn::Months | HostFn::Tickers => TV::ALWAYS,
            HostFn::MonthIndex => TV { time: vals[0].time, rows: None, int: Lin::Unknown },
            HostFn::RecordTrial | HostFn::DeflateTrial => TV { time: Time::new(At::Late, sp), rows: None, int: Lin::Unknown },
            HostFn::Evaluate { .. } | HostFn::Simulate { .. } => {
                self.check_weights(&vals[0], sp)?;
                // the evaluation of a whole history is known at its end
                TV { time: Time::new(At::Late, sp), rows: None, int: Lin::Unknown }
            }
            HostFn::Holdout { .. } => TV::ALWAYS,
            // the sessions of month t are known at end(t); the calendar and the companies always
            HostFn::Daily { .. } | HostFn::Nefin { .. } | HostFn::NefinDaily { .. } => data(0),
            HostFn::Sessions | HostFn::SessionDates | HostFn::Issuers | HostFn::Classes | HostFn::ResultsSha256 => TV::ALWAYS,
            HostFn::OpenOnce { .. } => {
                // the holdout trades the same way: no look-ahead there either
                self.check_weights(&vals[1], sp)?;
                TV { time: Time::new(At::Late, sp), rows: None, int: Lin::Unknown }
            }
        })
    }

    /// `b3.evaluate(w)`: the weights of month t are traded at the close of month t - 1.
    fn check_weights(&self, w: &TV, sp: Span) -> KResult<()> {
        let hint = "`b3.evaluate` trades the weights of month m at the close of month m - 1: use data up to `m - 1` (momentum skips the last month: `m - 2`)";
        match w.time.at {
            At::Always => {}
            At::End(Lin::Unknown) => {
                return Err(Diag::new("E0302", w.time.span.unwrap_or(sp), "cannot prove that the weights use only past information")
                    .label("this reaches every month of the weights, at a time the compiler cannot determine")
                    .hint("index months with the loop variable plus a constant: `ret[month: m - 1]`, `weights[month: m] = ...`"));
            }
            at => {
                return Err(Diag::new("E0301", w.time.span.unwrap_or(sp), "use of future information")
                    .label(format!("known {}, but the weights of each month are decided at the end of the previous one", self.show_at(at)))
                    .hint(hint));
            }
        }
        let Some(r) = w.rows else { return Ok(()) };
        let span = r.span.unwrap_or(sp);
        if r.axis != AX_MONTH {
            return Err(Diag::new("E0302", span, "the weights are timed on another axis").hint("`b3.evaluate` trades month by month: index the weights by `month`"));
        }
        match r.lag {
            None => Err(Diag::new("E0302", span, "cannot prove that the weights use only past information")
                .label("the month of this data cannot be related to the month of the weights")
                .hint("index months with the loop variable plus a constant: `ret[month: m - 1]`, `weights[month: m] = ...`")),
            Some(l) if l > -1 => {
                let label = match r.store {
                    Some((row, at)) => format!(
                        "known {}, but the portfolio of month {} is decided at end({})",
                        self.show_at(at),
                        self.show_lin(row),
                        self.show_lin(row.add(-1))
                    ),
                    None => format!("row t of this data is known at end(t{}), but the portfolio of month t is decided at end(t - 1)", if l == 0 { String::new() } else { format!(" + {l}") }),
                };
                Err(Diag::new("E0301", span, "use of future information").label(label).hint(hint))
            }
            _ => Ok(()),
        }
    }

    // ---------------------------------------------------------------- loops

    /// Runs `body` to a fixpoint. `sym`: the loop variable of a range loop, whose next value is
    /// one more (what an iteration leaves for the next is one month older).
    fn loop_fix(&mut self, f: &mut Vec<TV>, sym: Option<u32>, slot: Option<u32>, body: &HExpr, cond: Option<&HExpr>, last: Lin) -> KResult<()> {
        let entry_pc = self.pc;
        let mut head = f.clone();
        let mut head_pc = entry_pc;
        let mut last_frame = f.clone();
        for round in 0..MAX_ROUNDS {
            let mut st = head.clone();
            if let (Some(s), Some(slot)) = (sym, slot) {
                st[slot as usize] = TV::int(Lin::Sym(s, 0));
            }
            self.pc = head_pc;
            if let Some(c) = cond {
                // while: running the body depends on the condition
                let cv = self.eval(c, &mut st)?;
                self.pc = self.pc.join(cv.time);
            }
            self.eval(body, &mut st)?;
            let exit_pc = self.pc;
            let carried: Vec<TV> = match sym {
                Some(s) => st.iter().map(|v| v.shift(s, -1)).collect(),
                None => st.clone(),
            };
            let carried_pc = match sym {
                Some(s) => TV { time: exit_pc, ..TV::ALWAYS }.shift(s, -1).time,
                None => exit_pc,
            };
            let new_head = join_frames(&head, &carried);
            let new_pc = head_pc.join(carried_pc);
            last_frame = st;
            if same_frames(&new_head, &head) && new_pc.at == head_pc.at {
                break;
            }
            head = new_head;
            head_pc = new_pc;
            if round + 2 == MAX_ROUNDS {
                // widen: whatever still moves cannot be pinned to a month
                if let Some(s) = sym {
                    head = head.iter().map(|v| v.forget(s)).collect();
                }
            }
        }
        let mut out = join_frames(&head, &last_frame);
        if let Some(s) = sym {
            out = out.iter().map(|v| v.bound(s, last)).collect();
        }
        *f = out;
        // after the loop: a `break` (or `return`) decided by data makes what follows depend on it
        let broke = jumps(body, 0, true);
        self.pc = if broke {
            let mut p = head_pc;
            if let Some(s) = sym {
                p = TV { time: p, ..TV::ALWAYS }.forget(s).time;
            }
            entry_pc.join(p)
        } else {
            entry_pc
        };
        if let Some(c) = cond {
            // when a `while` stops depends on its condition
            let mut tmp = f.clone();
            let cv = self.eval(c, &mut tmp)?;
            self.pc = self.pc.join(match sym {
                Some(s) => TV { time: cv.time, ..TV::ALWAYS }.forget(s).time,
                None => cv.time,
            });
        }
        Ok(())
    }

    // ---------------------------------------------------------------- expressions

    fn eval(&mut self, e: &HExpr, f: &mut Vec<TV>) -> KResult<TV> {
        Ok(match e {
            HExpr::Lit(Value::Int(i)) | HExpr::IntLit(i, _) => TV::int(Lin::Const(*i)),
            HExpr::Lit(_) => TV::ALWAYS,
            HExpr::Local(s) => f[*s as usize],
            HExpr::FnRef(id) => {
                let s = self.summary(*id)?;
                let mut t = s.result.no_int();
                for p in s.param_syms.iter().flatten() {
                    t = t.forget(*p);
                }
                if let Some(h) = s.horizon {
                    // called with who knows what: as late as the data
                    t.time = t.time.horizon(h, At::Late);
                    t = t.forget(h);
                }
                t
            }
            HExpr::List(xs) | HExpr::MapLit(xs) | HExpr::Tuple(xs) | HExpr::Struct(_, xs) | HExpr::Variant(_, _, xs) => {
                let mut t = TV::ALWAYS;
                for x in xs {
                    t = t.join(&self.eval(x, f)?);
                }
                t.no_int()
            }
            HExpr::Field(x, _) => self.eval(x, f)?.no_int(),
            HExpr::Index(x, i, _) => {
                let a = self.eval(x, f)?;
                let b = self.eval(i, f)?;
                a.join(&b).no_int()
            }
            HExpr::TensorIndex(x, sels, sp) => self.tensor_index(x, sels, *sp, f)?,
            HExpr::TensorAssign(p, sels, _, v, sp) => {
                self.tensor_assign(p, sels, v, *sp, f)?;
                TV::ALWAYS
            }
            HExpr::Host(h, args, sp) => self.host(*h, args, *sp, f)?,
            HExpr::CallFn(id, args, _) => self.call_fn(*id, args, f)?,
            HExpr::CallValue(c, args, _) if static_fn(c).is_some() => {
                // grad(f)(x...) depends on the data exactly as f(x...) does
                let id = static_fn(c).expect("checked by the guard");
                let hargs: Vec<HArg> = args.iter().map(|a| HArg::Val(a.clone())).collect();
                self.call_fn(id, &hargs, f)?
            }
            HExpr::CallValue(c, args, sp) => {
                let mut t = self.eval(c, f)?;
                for a in args {
                    let v = self.eval(a, f)?;
                    // an unknown function may read a temporal argument at any row
                    let v = if v.rows.is_some() { self.over_time(v, None, *sp) } else { v };
                    t = t.join(&v);
                }
                t.no_int()
            }
            HExpr::CallBuiltin(b, args, sp) => self.builtin(*b, args, *sp, f)?,
            HExpr::Unary(op, x, _) => {
                let v = self.eval(x, f)?;
                let int = match (op, v.int) {
                    (UnOp::Neg, Lin::Const(c)) => c.checked_neg().map_or(Lin::Unknown, Lin::Const),
                    _ => Lin::Unknown,
                };
                TV { int, ..v }
            }
            HExpr::Binary(op, a, b, _) => {
                let x = self.eval(a, f)?;
                let y = self.eval(b, f)?;
                let mut t = x.join(&y);
                t.int = match op {
                    BinOp::Add => lin_add(x.int, y.int),
                    BinOp::Sub => lin_sub(x.int, y.int),
                    _ => Lin::Unknown,
                };
                t
            }
            HExpr::And(a, b) | HExpr::Or(a, b) => {
                let x = self.eval(a, f)?;
                let saved = self.pc;
                self.pc = self.pc.join(x.time);
                let y = self.eval(b, f);
                self.pc = saved;
                x.join(&y?).no_int()
            }
            HExpr::Assign(p, v) => {
                let tv = self.eval(v, f)?;
                self.store(p, tv, f)?;
                TV::ALWAYS
            }
            HExpr::OpAssign(p, op, v, _) => {
                let rhs = self.eval(v, f)?;
                let cur = self.load_place(p, f)?;
                let mut t = cur.join(&rhs);
                t.int = match op {
                    BinOp::Add => lin_add(cur.int, rhs.int),
                    BinOp::Sub => lin_sub(cur.int, rhs.int),
                    _ => Lin::Unknown,
                };
                self.store(p, t, f)?;
                TV::ALWAYS
            }
            HExpr::If(c, t, els) => {
                let cv = self.eval(c, f)?;
                let saved = self.pc;
                let inner = saved.join(cv.time);
                self.pc = inner;
                let mut ft = f.clone();
                let a = self.eval(t, &mut ft)?;
                let pc_a = self.pc;
                self.pc = inner;
                let (b, pc_b) = match els {
                    Some(x) => {
                        let v = self.eval(x, f)?;
                        (v, self.pc)
                    }
                    None => (TV::ALWAYS, inner),
                };
                *f = join_frames(&ft, f);
                let jumped = jumps(t, 0, false) || els.as_ref().is_some_and(|x| jumps(x, 0, false));
                // a branch that leaves (continue/break/return) makes what follows depend on `c`
                self.pc = if jumped { pc_a.join(pc_b) } else { saved };
                a.join(&b).with_time(cv.time)
            }
            HExpr::While(c, body) => {
                self.loop_fix(f, None, None, body, Some(c), Lin::Unknown)?;
                TV::ALWAYS
            }
            HExpr::ForRange { slot, name, lo, hi, inclusive, body } => {
                let a = self.eval(lo, f)?;
                let b = self.eval(hi, f)?;
                let sym = self.new_sym(name);
                let last = if *inclusive { b.int } else { b.int.add(-1) };
                // how many times the body runs depends on the bounds
                let saved = self.pc;
                self.pc = saved.join(a.time).join(b.time);
                if let Lin::Sym(t, c) = last {
                    BOUNDS.with(|bs| bs.borrow_mut().insert(sym, (t, c)));
                }
                let r = self.loop_fix(f, Some(sym), Some(*slot), body, None, last);
                BOUNDS.with(|bs| bs.borrow_mut().remove(&sym));
                r?;
                if !jumps(body, 0, true) {
                    self.pc = saved;
                }
                TV::ALWAYS
            }
            HExpr::ForList { pat, list, body } => {
                let l = self.eval(list, f)?;
                let saved = self.pc;
                // how many times the body runs depends on the list
                self.pc = saved.join(l.time);
                // every element is as late as the list
                self.loop_fix_list(f, pat, l.no_int(), body)?;
                if !jumps(body, 0, true) {
                    self.pc = saved;
                }
                TV::ALWAYS
            }
            HExpr::Match(s, arms, _) => {
                let sv = self.eval(s, f)?;
                let saved = self.pc;
                let inner = saved.join(sv.time);
                let mut out = TV::ALWAYS;
                let mut frames = Vec::with_capacity(arms.len());
                let mut pcs = inner;
                let mut jumped = false;
                for (p, b) in arms {
                    let mut st = f.clone();
                    self.pc = inner;
                    self.bind(p, sv.no_int(), &mut st);
                    let v = self.eval(b, &mut st)?;
                    pcs = pcs.join(self.pc);
                    jumped |= jumps(b, 0, false);
                    out = out.join(&v);
                    frames.push(st);
                }
                for st in &frames {
                    *f = join_frames(f, st);
                }
                self.pc = if jumped { pcs } else { saved };
                out.with_time(sv.time).no_int()
            }
            HExpr::Block(stmts, tail) => {
                for s in stmts {
                    self.eval(s, f)?;
                }
                match tail {
                    Some(t) => self.eval(t, f)?,
                    None => TV::ALWAYS,
                }
            }
            HExpr::Let(p, init) => {
                let v = self.eval(init, f)?;
                let v = self.under_pc(v);
                self.bind(p, v, f);
                TV::ALWAYS
            }
            HExpr::Lambda(def, outer) => {
                let lb = &self.ck.lambdas[*def as usize];
                let mut frame = vec![TV::ALWAYS; lb.n_slots as usize];
                let mut caps = TV::ALWAYS;
                for (k, s) in lb.capture_slots.iter().enumerate() {
                    let v = f[outer[k] as usize];
                    frame[*s as usize] = v;
                    caps = caps.join(&v);
                }
                let saved_pc = std::mem::replace(&mut self.pc, Time::ALWAYS);
                let saved_ret = self.ret.take();
                let r = self.eval(&lb.body, &mut frame);
                let pc = std::mem::replace(&mut self.pc, saved_pc);
                let ret = std::mem::replace(&mut self.ret, saved_ret);
                let mut body = r?.with_time(pc);
                if let Some(rv) = ret {
                    body = body.join(&rv);
                }
                caps.join(&body).no_int()
            }
            HExpr::Return(v) => {
                let tv = match v {
                    Some(x) => self.eval(x, f)?,
                    None => TV::ALWAYS,
                };
                let tv = self.under_pc(tv);
                self.ret = Some(match self.ret {
                    Some(r) => r.join(&tv),
                    None => tv,
                });
                TV::ALWAYS
            }
            HExpr::Break | HExpr::Continue => TV::ALWAYS,
        })
    }

    fn loop_fix_list(&mut self, f: &mut Vec<TV>, pat: &HPat, elem: TV, body: &HExpr) -> KResult<()> {
        // the element changes every iteration: bind it at the head of each round
        let entry_pc = self.pc;
        let mut head = f.clone();
        let mut head_pc = entry_pc;
        let mut last = f.clone();
        for _ in 0..MAX_ROUNDS {
            let mut st = head.clone();
            self.bind(pat, elem, &mut st);
            self.pc = head_pc;
            self.eval(body, &mut st)?;
            let new_head = join_frames(&head, &st);
            let new_pc = head_pc.join(self.pc);
            last = st;
            if same_frames(&new_head, &head) && new_pc.at == head_pc.at {
                break;
            }
            head = new_head;
            head_pc = new_pc;
        }
        *f = join_frames(&head, &last);
        self.pc = if jumps(body, 0, true) { entry_pc.join(head_pc) } else { entry_pc };
        Ok(())
    }
}
