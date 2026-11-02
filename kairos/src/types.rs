//! Type checker (M1): inference by unification, name resolution into frame slots, mutability and
//! `inout` rules, exhaustive `match`. Produces the HIR.
//!
//! Integer literals start as a "numeric" type variable that can become `i64` or `f64`; at `let`
//! boundaries unresolved numeric variables default to `i64` (so `let n = 2` is an `i64`, while
//! `x * 2` with `x: f64` makes the literal a `f64`). Variables of different numeric types never mix.

use crate::ast::*;
use crate::diag::{Diag, KResult, Span};
use crate::hir::*;
use crate::value::Value;
use std::collections::HashMap;

mod tensors;
pub use tensors::{AX_ASSET, AX_DAY, AX_MONTH};

#[derive(Clone, Debug, PartialEq)]
pub enum Ty {
    Int,
    Float,
    Bool,
    Str,
    Unit,
    /// type of diverging expressions (`return`, `break`); unifies with anything
    Never,
    List(Box<Ty>),
    Tuple(Vec<Ty>),
    Struct(u32),
    Enum(u32),
    Fn(Vec<Ty>, Box<Ty>),
    Var(u32),
    /// type parameter of a generic function (`fn adam[P](...)`): a differentiable type
    Param(u32),
    /// named-axis tensor: axis ids kept sorted (the order of axes never matters for typing),
    /// element `f64` or `bool`; never empty (a tensor with no axes is a scalar)
    Tensor(Vec<u32>, Box<Ty>),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Freq {
    Daily,
    Monthly,
}

#[derive(Clone, Debug)]
pub struct AxisDef {
    pub name: String,
    pub temporal: Option<Freq>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Any,
    Num,
}

#[derive(Clone, Debug)]
pub struct StructDef {
    pub name: String,
    pub fields: Vec<(String, Ty)>,
}

#[derive(Clone, Debug)]
pub struct EnumDef {
    pub name: String,
    pub variants: Vec<(String, Vec<Ty>)>,
}

#[derive(Clone, Debug)]
pub struct FnSig {
    pub name: String,
    /// (name, type, inout)
    pub params: Vec<(String, Ty, bool)>,
    pub ret: Ty,
    /// `Ty::Param` ids of `fn f[P](...)` (instantiated at each call)
    pub type_params: Vec<u32>,
}

#[derive(Clone, Debug)]
struct Local {
    slot: u32,
    ty: Ty,
    mutable: bool,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum CtxKind {
    Main,
    Fn,
    Lambda,
}

#[derive(Clone, Debug)]
struct Ctx {
    scopes: Vec<HashMap<String, Local>>,
    n_slots: u32,
    ret: Ty,
    kind: CtxKind,
    /// (outer slot, inner slot) for lambdas
    captures: Vec<(u32, u32)>,
    loops: u32,
}

impl Ctx {
    fn new(kind: CtxKind, ret: Ty) -> Self {
        Ctx { scopes: vec![HashMap::new()], n_slots: 0, ret, kind, captures: Vec::new(), loops: 0 }
    }
}

/// A checked top-level statement; `echo` = expression whose value the REPL shows.
pub struct TopStmt {
    pub hir: HExpr,
    pub echo: bool,
}

#[derive(Clone)]
pub struct Checker {
    pub structs: Vec<StructDef>,
    struct_ids: HashMap<String, u32>,
    pub enums: Vec<EnumDef>,
    enum_ids: HashMap<String, u32>,
    pub sigs: Vec<FnSig>,
    fn_ids: HashMap<String, u32>,
    pub bodies: Vec<Option<FnBody>>,
    pub lambdas: Vec<LambdaBody>,
    subst: Vec<Option<Ty>>,
    kinds: Vec<Kind>,
    ctxs: Vec<Ctx>,
    main: Ctx,
    /// names bound by top-level `let`s (only for better error hints inside functions)
    top_names: std::collections::HashSet<String>,
    /// declared axes (ids 0..3 are the built-in `month`, `day`, `asset`)
    pub axes: Vec<AxisDef>,
    axis_ids: HashMap<String, u32>,
    /// type parameters in scope while checking a generic function (name -> `Ty::Param` id)
    generic_scope: HashMap<String, u32>,
    /// names of the type parameters of generic functions (`Ty::Param` ids)
    type_param_names: Vec<String>,
    /// knowledge times of the top-level variables (kept across REPL inputs)
    temporal: crate::temporal::State,
}

fn pattern_names(p: &Pattern, out: &mut std::collections::HashSet<String>) {
    match p {
        Pattern::Bind(n, _) => {
            out.insert(n.clone());
        }
        Pattern::Tuple(ps, _) => {
            for q in ps {
                pattern_names(q, out);
            }
        }
        _ => {}
    }
}

const MIX_HINT: &str = "Kairós never mixes `i64` and `f64` silently: convert with `float(x)` or `int(x)`";

fn mismatch(got: &str, want: &str, span: Span) -> Diag {
    let mixed = (got == "i64" && want == "f64") || (got == "f64" && want == "i64");
    let code = if mixed { "E0107" } else { "E0101" };
    let d = Diag::new(code, span, format!("mismatched types: expected `{want}`, found `{got}`")).label(format!("expected `{want}`"));
    if mixed { d.hint(MIX_HINT) } else { d }
}

fn irrefutable(p: &HPat) -> bool {
    match p {
        HPat::Wild | HPat::Bind(_) => true,
        HPat::Tuple(ps) => ps.iter().all(irrefutable),
        _ => false,
    }
}

impl Default for Checker {
    fn default() -> Self {
        Self::new()
    }
}

impl Checker {
    pub fn new() -> Self {
        let (axes, axis_ids) = Self::builtin_axes();
        Checker {
            structs: Vec::new(),
            struct_ids: HashMap::new(),
            enums: Vec::new(),
            enum_ids: HashMap::new(),
            sigs: Vec::new(),
            fn_ids: HashMap::new(),
            bodies: Vec::new(),
            lambdas: Vec::new(),
            subst: Vec::new(),
            kinds: Vec::new(),
            ctxs: Vec::new(),
            main: Ctx::new(CtxKind::Main, Ty::Unit),
            top_names: std::collections::HashSet::new(),
            axes,
            axis_ids,
            temporal: crate::temporal::State::default(),
            type_param_names: Vec::new(),
            generic_scope: HashMap::new(),
        }
    }

    pub fn main_slots(&self) -> u32 {
        self.main.n_slots
    }

    // ------------------------------------------------------------------ entry point

    /// Checks one batch of items (a file or one REPL input) and returns the new top-level
    /// statements as HIR. Functions and types are hoisted within the batch.
    pub fn check_items(&mut self, items: &[Item]) -> KResult<Vec<TopStmt>> {
        let lambda_base = self.lambdas.len();
        let mut new_structs = Vec::new();
        let mut new_enums = Vec::new();
        for it in items {
            match it {
                Item::Struct(s) => {
                    self.ensure_new_name(&s.name, s.span, false)?;
                    let id = self.structs.len() as u32;
                    self.struct_ids.insert(s.name.clone(), id);
                    self.structs.push(StructDef { name: s.name.clone(), fields: Vec::new() });
                    new_structs.push((id, s));
                }
                Item::Enum(e) => {
                    self.ensure_new_name(&e.name, e.span, false)?;
                    let id = self.enums.len() as u32;
                    self.enum_ids.insert(e.name.clone(), id);
                    self.enums.push(EnumDef { name: e.name.clone(), variants: Vec::new() });
                    new_enums.push((id, e));
                }
                Item::Axis(a) => self.declare_axis(a)?,
                _ => {}
            }
        }
        for (id, s) in &new_structs {
            let mut fields: Vec<(String, Ty)> = Vec::new();
            for (n, te) in &s.fields {
                if fields.iter().any(|(m, _)| m == n) {
                    return Err(Diag::new("E0112", te.span(), format!("field `{n}` is declared twice")));
                }
                let t = self.resolve_type(te)?;
                fields.push((n.clone(), t));
            }
            self.structs[*id as usize].fields = fields;
        }
        for (id, e) in &new_enums {
            let mut variants: Vec<(String, Vec<Ty>)> = Vec::new();
            for (n, tes) in &e.variants {
                if variants.iter().any(|(m, _)| m == n) {
                    return Err(Diag::new("E0112", e.span, format!("variant `{n}` is declared twice")));
                }
                let mut ts = Vec::new();
                for te in tes {
                    ts.push(self.resolve_type(te)?);
                }
                variants.push((n.clone(), ts));
            }
            self.enums[*id as usize].variants = variants;
        }
        let mut new_fns = Vec::new();
        for it in items {
            if let Item::Fn(f) = it {
                self.ensure_new_name(&f.name, f.span, true)?;
                // `fn f[P](...)`: each type parameter is a differentiable type
                let mut type_params = Vec::new();
                self.generic_scope.clear();
                for (tp, tsp) in &f.type_params {
                    if self.generic_scope.contains_key(tp) {
                        return Err(Diag::new("E0112", *tsp, format!("type parameter `{tp}` is declared twice")));
                    }
                    let pid = self.type_param_names.len() as u32;
                    self.type_param_names.push(tp.clone());
                    self.generic_scope.insert(tp.clone(), pid);
                    type_params.push(pid);
                }
                let mut params: Vec<(String, Ty, bool)> = Vec::new();
                for p in &f.params {
                    if params.iter().any(|(n, _, _)| *n == p.name) {
                        self.generic_scope.clear();
                        return Err(Diag::new("E0112", p.span, format!("parameter `{}` is declared twice", p.name)));
                    }
                    let t = self.resolve_type(&p.ty);
                    let t = t.inspect_err(|_| self.generic_scope.clear())?;
                    params.push((p.name.clone(), t, p.inout));
                }
                let ret = match &f.ret {
                    Some(t) => self.resolve_type(t).inspect_err(|_| self.generic_scope.clear())?,
                    None => Ty::Unit,
                };
                self.generic_scope.clear();
                let id = self.sigs.len() as u32;
                self.fn_ids.insert(f.name.clone(), id);
                self.sigs.push(FnSig { name: f.name.clone(), params, ret, type_params });
                self.bodies.push(None);
                new_fns.push((id, f));
            }
        }
        for it in items {
            if let Item::Stmt(Stmt::Let { pat, .. }) = it {
                pattern_names(pat, &mut self.top_names);
            }
        }
        for (id, f) in &new_fns {
            self.check_fn(*id, f)?;
        }
        let mut out = Vec::new();
        let main = std::mem::replace(&mut self.main, Ctx::new(CtxKind::Main, Ty::Unit));
        self.ctxs.push(main);
        let mut res = Ok(());
        for it in items {
            if let Item::Stmt(s) = it {
                match self.stmt(s) {
                    Ok((h, _)) => out.push(TopStmt { hir: h, echo: matches!(s, Stmt::Expr(_)) }),
                    Err(e) => {
                        res = Err(e);
                        break;
                    }
                }
            }
        }
        self.main = self.ctxs.pop().expect("main context");
        res?;
        // integer literals become i64 or f64 now that inference is done
        for (id, _) in &new_fns {
            let mut b = self.bodies[*id as usize].take().expect("checked body");
            self.fin_expr(&mut b.body);
            if let Some((k, _)) = &mut b.known_at {
                self.fin_expr(k);
            }
            self.bodies[*id as usize] = Some(b);
        }
        for i in lambda_base..self.lambdas.len() {
            let mut body = std::mem::replace(&mut self.lambdas[i].body, HExpr::Break);
            self.fin_expr(&mut body);
            self.lambdas[i].body = body;
        }
        for t in out.iter_mut() {
            self.fin_expr(&mut t.hir);
        }
        // temporal types (M3): no decision may use information from its future
        let ids: Vec<u32> = new_fns.iter().map(|(id, _)| *id).collect();
        let tops: Vec<&HExpr> = out.iter().map(|t| &t.hir).collect();
        let mut st = std::mem::take(&mut self.temporal);
        let r = crate::temporal::check(self, &ids, &tops, &mut st);
        self.temporal = st;
        r?;
        Ok(out)
    }

    fn ensure_new_name(&self, name: &str, span: Span, is_fn: bool) -> KResult<()> {
        if self.struct_ids.contains_key(name) || self.enum_ids.contains_key(name) || self.fn_ids.contains_key(name) {
            return Err(Diag::new("E0112", span, format!("`{name}` is already defined")));
        }
        if is_fn && builtin_by_name(name).is_some() {
            return Err(Diag::new("E0112", span, format!("`{name}` is a builtin function and cannot be redefined")));
        }
        Ok(())
    }

    fn check_fn(&mut self, id: u32, f: &FnDecl) -> KResult<()> {
        let sig = self.sigs[id as usize].clone();
        let mut ctx = Ctx::new(CtxKind::Fn, sig.ret.clone());
        for (name, ty, inout) in &sig.params {
            let slot = ctx.n_slots;
            ctx.n_slots += 1;
            ctx.scopes[0].insert(name.clone(), Local { slot, ty: ty.clone(), mutable: *inout });
        }
        self.ctxs.push(ctx);
        // the type parameters are in scope in the body (`let z: P = ...`)
        self.generic_scope = sig.type_params.iter().map(|p| (self.type_param_names[*p as usize].clone(), *p)).collect();
        // `@ end(e)`: an integer expression over the parameters (checked by the temporal pass)
        let known_at = match &f.known_at {
            Some((e, sp)) => match self.expr(e, Some(&Ty::Int)) {
                Ok((h, t)) => match self.expect_ty(&t, &Ty::Int, e.span) {
                    Ok(()) => Ok(Some((h, *sp))),
                    Err(d) => Err(d),
                },
                Err(d) => Err(d),
            },
            None => Ok(None),
        };
        let r = known_at.and_then(|k| self.block(&f.body, Some(&sig.ret)).map(|b| (b, k)));
        let ctx = self.ctxs.pop().expect("fn context");
        self.generic_scope.clear();
        let ((body, t), known_at) = r?;
        if self.unify(&t, &sig.ret).is_err() {
            let span = f.body.tail.as_ref().map(|e| e.span).unwrap_or(f.body.span);
            let (g, w) = (self.show(&t), self.show(&sig.ret));
            let mut d = Diag::new("E0108", span, format!("function `{}` must return `{w}`, but its body has type `{g}`", sig.name));
            if w == "()" {
                d = d.hint(format!("add `-> {g}` to the signature"));
            } else if g == "()" {
                d = d.hint("the last line of the body is the return value: remove a trailing `;` or use `return`");
            }
            return Err(d);
        }
        self.bodies[id as usize] = Some(FnBody { n_slots: ctx.n_slots, body, known_at });
        Ok(())
    }

    // ------------------------------------------------------------------ type variables

    fn fresh(&mut self, k: Kind) -> (Ty, u32) {
        let id = self.subst.len() as u32;
        self.subst.push(None);
        self.kinds.push(k);
        (Ty::Var(id), id)
    }

    fn resolve(&self, t: &Ty) -> Ty {
        let mut t = t.clone();
        while let Ty::Var(v) = t {
            match &self.subst[v as usize] {
                Some(n) => t = n.clone(),
                None => break,
            }
        }
        t
    }

    fn unify(&mut self, a: &Ty, b: &Ty) -> Result<(), ()> {
        let (a, b) = (self.resolve(a), self.resolve(b));
        match (&a, &b) {
            (Ty::Never, _) | (_, Ty::Never) => Ok(()),
            (Ty::Var(x), Ty::Var(y)) if x == y => Ok(()),
            (Ty::Var(x), _) => self.bind(*x, b.clone()),
            (_, Ty::Var(y)) => self.bind(*y, a.clone()),
            (Ty::List(p), Ty::List(q)) => self.unify(p, q),
            (Ty::Tuple(p), Ty::Tuple(q)) if p.len() == q.len() => {
                for (x, y) in p.iter().zip(q.iter()) {
                    self.unify(x, y)?;
                }
                Ok(())
            }
            (Ty::Fn(pa, ra), Ty::Fn(pb, rb)) if pa.len() == pb.len() => {
                for (x, y) in pa.iter().zip(pb.iter()) {
                    self.unify(x, y)?;
                }
                self.unify(ra, rb)
            }
            (Ty::Tensor(xa, ea), Ty::Tensor(xb, eb)) if xa == xb => self.unify(ea, eb),
            _ => {
                if a == b {
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }

    fn bind(&mut self, v: u32, t: Ty) -> Result<(), ()> {
        if let Ty::Var(w) = t {
            if self.kinds[v as usize] == Kind::Num {
                self.kinds[w as usize] = Kind::Num;
            }
            self.subst[v as usize] = Some(Ty::Var(w));
            return Ok(());
        }
        if self.kinds[v as usize] == Kind::Num && !matches!(t, Ty::Int | Ty::Float) {
            return Err(());
        }
        if self.occurs(v, &t) {
            return Err(());
        }
        self.subst[v as usize] = Some(t);
        Ok(())
    }

    fn occurs(&self, v: u32, t: &Ty) -> bool {
        match self.resolve(t) {
            Ty::Var(w) => w == v,
            Ty::List(x) => self.occurs(v, &x),
            Ty::Tuple(xs) => xs.iter().any(|x| self.occurs(v, x)),
            Ty::Fn(ps, r) => ps.iter().any(|x| self.occurs(v, x)) || self.occurs(v, &r),
            Ty::Tensor(_, e) => self.occurs(v, &e),
            _ => false,
        }
    }

    /// Unresolved numeric variables inside `t` become `i64` (done at `let` boundaries).
    fn default_nums(&mut self, t: &Ty) {
        match self.resolve(t) {
            Ty::Var(v) => {
                if self.kinds[v as usize] == Kind::Num {
                    self.subst[v as usize] = Some(Ty::Int);
                }
            }
            Ty::List(x) => self.default_nums(&x),
            Ty::Tuple(xs) => {
                for x in &xs {
                    self.default_nums(x);
                }
            }
            Ty::Fn(ps, r) => {
                for x in &ps {
                    self.default_nums(x);
                }
                self.default_nums(&r);
            }
            _ => {}
        }
    }

    pub fn show(&self, t: &Ty) -> String {
        match self.resolve(t) {
            Ty::Int => "i64".into(),
            Ty::Float => "f64".into(),
            Ty::Bool => "bool".into(),
            Ty::Str => "str".into(),
            Ty::Unit => "()".into(),
            Ty::Never => "!".into(),
            Ty::List(x) => format!("List[{}]", self.show(&x)),
            Ty::Tuple(xs) => format!("({})", xs.iter().map(|x| self.show(x)).collect::<Vec<_>>().join(", ")),
            Ty::Struct(i) => self.structs[i as usize].name.clone(),
            Ty::Enum(i) => self.enums[i as usize].name.clone(),
            Ty::Fn(ps, r) => {
                format!("fn({}) -> {}", ps.iter().map(|x| self.show(x)).collect::<Vec<_>>().join(", "), self.show(&r))
            }
            Ty::Param(p) => self.type_param_names.get(p as usize).cloned().unwrap_or_else(|| format!("T{p}")),
            Ty::Tensor(axes, e) => {
                let names: Vec<&str> = axes.iter().map(|a| self.axes[*a as usize].name.as_str()).collect();
                format!("Tensor[{}; {}]", names.join(", "), self.show(&e))
            }
            Ty::Var(v) => {
                if self.kinds[v as usize] == Kind::Num {
                    "{integer}".into()
                } else {
                    "_".into()
                }
            }
        }
    }

    fn expect_ty(&mut self, got: &Ty, want: &Ty, span: Span) -> KResult<()> {
        if self.unify(got, want).is_ok() {
            return Ok(());
        }
        Err(mismatch(&self.show(got), &self.show(want), span))
    }

    fn ensure_numeric(&mut self, t: &Ty, span: Span, what: &str) -> KResult<()> {
        match self.resolve(t) {
            Ty::Int | Ty::Float | Ty::Never => Ok(()),
            Ty::Var(v) => {
                self.kinds[v as usize] = Kind::Num;
                Ok(())
            }
            other => Err(Diag::new("E0101", span, format!("{what} needs a number, found `{}`", self.show(&other)))),
        }
    }

    fn list_elem(&mut self, t: &Ty, span: Span) -> KResult<Ty> {
        match self.resolve(t) {
            Ty::List(e) => Ok(*e),
            Ty::Var(_) => {
                let (e, _) = self.fresh(Kind::Any);
                self.expect_ty(t, &Ty::List(Box::new(e.clone())), span)?;
                Ok(e)
            }
            other => Err(Diag::new("E0101", span, format!("expected a list, found `{}`", self.show(&other)))),
        }
    }

    fn resolve_type(&mut self, te: &TypeExpr) -> KResult<Ty> {
        match te {
            TypeExpr::Named { name, args, span } => {
                let arity = |n: usize| -> KResult<()> {
                    if args.len() != n {
                        Err(Diag::new("E0101", *span, format!("`{name}` takes {n} type argument(s)")))
                    } else {
                        Ok(())
                    }
                };
                Ok(match name.as_str() {
                    "i64" => {
                        arity(0)?;
                        Ty::Int
                    }
                    "f64" => {
                        arity(0)?;
                        Ty::Float
                    }
                    "bool" => {
                        arity(0)?;
                        Ty::Bool
                    }
                    "str" => {
                        arity(0)?;
                        Ty::Str
                    }
                    "List" => {
                        arity(1)?;
                        Ty::List(Box::new(self.resolve_type(&args[0])?))
                    }
                    "Tensor" => {
                        return Err(Diag::new("E0101", *span, "`Tensor` needs its axes and element type")
                            .hint("write it like `Tensor[month, asset; f64]`"));
                    }
                    "f32" | "i32" | "i16" | "i8" | "u64" | "u32" | "u16" | "u8" | "Money" | "date" | "Option"
                    | "Result" | "Simd" => {
                        return Err(Diag::new("E0002", *span, format!("type `{name}` is not available yet")).hint(
                            "Kairós supports i64, f64, bool, str, List[T], Tensor[axes; f64|bool], tuples, structs, enums and fn types",
                        ));
                    }
                    n if self.generic_scope.contains_key(n) => {
                        arity(0)?;
                        Ty::Param(self.generic_scope[n])
                    }
                    n => {
                        if let Some(&id) = self.struct_ids.get(n) {
                            arity(0)?;
                            Ty::Struct(id)
                        } else if let Some(&id) = self.enum_ids.get(n) {
                            arity(0)?;
                            Ty::Enum(id)
                        } else {
                            return Err(Diag::new("E0100", *span, format!("unknown type `{n}`")));
                        }
                    }
                })
            }
            TypeExpr::Tuple(ts, _) => {
                if ts.is_empty() {
                    return Ok(Ty::Unit);
                }
                let mut v = Vec::new();
                for t in ts {
                    v.push(self.resolve_type(t)?);
                }
                Ok(Ty::Tuple(v))
            }
            TypeExpr::Fn(ps, r, _) => {
                let mut v = Vec::new();
                for t in ps {
                    v.push(self.resolve_type(t)?);
                }
                Ok(Ty::Fn(v, Box::new(self.resolve_type(r)?)))
            }
            TypeExpr::Tensor { axes, elem, span } => self.resolve_tensor_type(axes, elem, *span),
        }
    }

    // ------------------------------------------------------------------ scopes

    fn ctx(&mut self) -> &mut Ctx {
        self.ctxs.last_mut().expect("context")
    }

    fn declare(&mut self, name: &str, ty: Ty, mutable: bool) -> u32 {
        let c = self.ctx();
        let slot = c.n_slots;
        c.n_slots += 1;
        c.scopes.last_mut().expect("scope").insert(name.to_string(), Local { slot, ty, mutable });
        slot
    }

    fn lookup(&mut self, name: &str) -> Option<Local> {
        let n = self.ctxs.len();
        if n == 0 {
            return None;
        }
        self.lookup_in(n - 1, name)
    }

    /// Lambdas capture outer locals by value: the first use creates a slot in the lambda frame.
    fn lookup_in(&mut self, ci: usize, name: &str) -> Option<Local> {
        for s in self.ctxs[ci].scopes.iter().rev() {
            if let Some(l) = s.get(name) {
                return Some(l.clone());
            }
        }
        if self.ctxs[ci].kind == CtxKind::Lambda && ci > 0 {
            let outer = self.lookup_in(ci - 1, name)?;
            let c = &mut self.ctxs[ci];
            let slot = c.n_slots;
            c.n_slots += 1;
            c.captures.push((outer.slot, slot));
            let l = Local { slot, ty: outer.ty, mutable: false };
            c.scopes[0].insert(name.to_string(), l.clone());
            return Some(l);
        }
        None
    }

    fn with_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> KResult<T>) -> KResult<T> {
        self.ctx().scopes.push(HashMap::new());
        let r = f(self);
        self.ctx().scopes.pop();
        r
    }

    // ------------------------------------------------------------------ statements and blocks

    fn stmt(&mut self, s: &Stmt) -> KResult<(HExpr, Ty)> {
        match s {
            Stmt::Let { mutable, pat, ty, init, .. } => {
                let declared = match ty {
                    Some(t) => Some(self.resolve_type(t)?),
                    None => None,
                };
                let (h, t) = self.expr(init, declared.as_ref())?;
                let t = match declared {
                    Some(d) => {
                        self.expect_ty(&t, &d, init.span)?;
                        d
                    }
                    None => {
                        self.default_nums(&t);
                        t
                    }
                };
                let hp = self.pattern(pat, &t, *mutable, false)?;
                Ok((HExpr::Let(hp, Box::new(h)), Ty::Unit))
            }
            Stmt::Expr(e) => self.expr(e, None),
        }
    }

    fn block(&mut self, b: &Block, exp: Option<&Ty>) -> KResult<(HExpr, Ty)> {
        self.with_scope(|me| {
            let mut hs = Vec::with_capacity(b.stmts.len());
            let mut diverges = false;
            for s in &b.stmts {
                let (h, t) = me.stmt(s)?;
                if matches!(me.resolve(&t), Ty::Never) {
                    diverges = true;
                }
                hs.push(h);
            }
            match &b.tail {
                Some(e) => {
                    let (h, t) = me.expr(e, exp)?;
                    Ok((HExpr::Block(hs, Some(Box::new(h))), t))
                }
                None => Ok((HExpr::Block(hs, None), if diverges { Ty::Never } else { Ty::Unit })),
            }
        })
    }

    // ------------------------------------------------------------------ patterns

    fn pattern(&mut self, p: &Pattern, t: &Ty, mutable: bool, refutable_ok: bool) -> KResult<HPat> {
        let refutable = |sp: Span| -> KResult<()> {
            if refutable_ok {
                Ok(())
            } else {
                Err(Diag::new("E0117", sp, "this pattern can fail to match here").hint("use `match` to test values; `let` and `for` only bind names and tuples"))
            }
        };
        match p {
            Pattern::Wild(_) => Ok(HPat::Wild),
            Pattern::Bind(n, _) => {
                let slot = self.declare(n, t.clone(), mutable);
                Ok(HPat::Bind(slot))
            }
            Pattern::Int(v, sp) => {
                refutable(*sp)?;
                let (vt, id) = self.fresh(Kind::Num);
                self.expect_ty(&vt, t, *sp)?;
                Ok(HPat::IntLit(*v, id))
            }
            Pattern::Str(s, sp) => {
                refutable(*sp)?;
                self.expect_ty(&Ty::Str, t, *sp)?;
                Ok(HPat::Lit(Value::Str(s.as_str().into())))
            }
            Pattern::Bool(b, sp) => {
                refutable(*sp)?;
                self.expect_ty(&Ty::Bool, t, *sp)?;
                Ok(HPat::Lit(Value::Bool(*b)))
            }
            Pattern::Tuple(ps, sp) => {
                let elems = match self.resolve(t) {
                    Ty::Tuple(ts) if ts.len() == ps.len() => ts,
                    _ => {
                        let ts: Vec<Ty> = ps.iter().map(|_| self.fresh(Kind::Any).0).collect();
                        self.expect_ty(&Ty::Tuple(ts.clone()), t, *sp)?;
                        ts
                    }
                };
                let mut hs = Vec::new();
                for (p, et) in ps.iter().zip(elems.iter()) {
                    hs.push(self.pattern(p, et, mutable, refutable_ok)?);
                }
                Ok(HPat::Tuple(hs))
            }
            Pattern::Variant { enum_name, variant, args, span } => {
                refutable(*span)?;
                let eid = match enum_name {
                    Some(n) => *self.enum_ids.get(n).ok_or_else(|| Diag::new("E0100", *span, format!("unknown enum `{n}`")))?,
                    None => match self.resolve(t) {
                        Ty::Enum(e) => e,
                        _ => {
                            let found: Vec<u32> = (0..self.enums.len() as u32)
                                .filter(|e| self.enums[*e as usize].variants.iter().any(|(n, _)| n == variant))
                                .collect();
                            match found.as_slice() {
                                [e] => *e,
                                [] => return Err(Diag::new("E0100", *span, format!("unknown variant `{variant}`"))),
                                _ => {
                                    return Err(Diag::new("E0100", *span, format!("ambiguous variant `{variant}`"))
                                        .hint("qualify it: `Enum.Variant`"));
                                }
                            }
                        }
                    },
                };
                self.expect_ty(&Ty::Enum(eid), t, *span)?;
                let def = self.enums[eid as usize].clone();
                let Some(vi) = def.variants.iter().position(|(n, _)| n == variant) else {
                    return Err(Diag::new("E0100", *span, format!("enum `{}` has no variant `{variant}`", def.name)));
                };
                let ptys = &def.variants[vi].1;
                if ptys.len() != args.len() {
                    return Err(Diag::new(
                        "E0103",
                        *span,
                        format!("variant `{variant}` has {} field(s), but the pattern has {}", ptys.len(), args.len()),
                    ));
                }
                let mut hs = Vec::new();
                for (p, pt) in args.iter().zip(ptys.iter()) {
                    hs.push(self.pattern(p, pt, mutable, refutable_ok)?);
                }
                Ok(HPat::Variant(eid, vi as u32, hs))
            }
        }
    }

    fn check_exhaustive(&self, t: &Ty, pats: &[HPat], sp: Span) -> KResult<()> {
        if pats.iter().any(irrefutable) {
            return Ok(());
        }
        let missing: Option<String> = match self.resolve(t) {
            Ty::Bool => {
                let has = |b: bool| pats.iter().any(|p| matches!(p, HPat::Lit(Value::Bool(x)) if *x == b));
                if !has(true) {
                    Some("true".into())
                } else if !has(false) {
                    Some("false".into())
                } else {
                    None
                }
            }
            Ty::Enum(eid) => {
                let def = &self.enums[eid as usize];
                def.variants
                    .iter()
                    .enumerate()
                    .find(|(vi, _)| {
                        !pats.iter().any(
                            |p| matches!(p, HPat::Variant(e, v, sub) if *e == eid && *v == *vi as u32 && sub.iter().all(irrefutable)),
                        )
                    })
                    .map(|(_, (n, _))| format!("{}.{n}", def.name))
            }
            _ => Some("_".into()),
        };
        match missing {
            None => Ok(()),
            Some(m) => Err(Diag::new("E0110", sp, format!("non-exhaustive match: `{m}` is not covered"))
                .hint("add an arm for it, or a catch-all `_ => ...` arm")),
        }
    }

    // ------------------------------------------------------------------ places (assignment targets)

    fn place(&mut self, e: &Expr) -> KResult<(HPlace, Ty)> {
        match &e.kind {
            ExprKind::Name(n) => {
                let Some(l) = self.lookup(n) else {
                    return Err(Diag::new("E0100", e.span, format!("cannot find `{n}` in this scope")));
                };
                if !l.mutable {
                    return Err(Diag::new("E0104", e.span, format!("cannot modify `{n}`: it is not mutable"))
                        .hint(format!("declare it with `var {n} = ...` (or as an `inout` parameter)")));
                }
                Ok((HPlace::Local(l.slot), l.ty))
            }
            ExprKind::Field(obj, f) => {
                let (p, t) = self.place(obj)?;
                let (i, ft) = self.field_of(&t, f, e.span)?;
                Ok((HPlace::Field(Box::new(p), i), ft))
            }
            ExprKind::TupleIndex(obj, k) => {
                let (p, t) = self.place(obj)?;
                let et = self.tuple_elem(&t, *k, e.span)?;
                Ok((HPlace::Field(Box::new(p), *k as u32), et))
            }
            ExprKind::Index(obj, items) => {
                let (p, t) = self.place(obj)?;
                if self.tensor_parts(&t).is_some() {
                    return Err(Diag::new("E0116", e.span, "a tensor slice cannot be passed as `inout` or changed in parts")
                        .hint("assign to the slice directly, e.g. `x[month: i] = v`"));
                }
                let idx = self.list_index(items, &t)?;
                let elem = self.list_elem(&t, obj.span)?;
                let (hi, ti) = self.expr(idx, Some(&Ty::Int))?;
                self.expect_ty(&ti, &Ty::Int, idx.span)?;
                Ok((HPlace::Index(Box::new(p), Box::new(hi), e.span), elem))
            }
            _ => Err(Diag::new("E0116", e.span, "this expression cannot be assigned to or passed as `inout`")),
        }
    }

    fn root_slot(p: &HPlace) -> u32 {
        match p {
            HPlace::Local(s) => *s,
            HPlace::Field(i, _) | HPlace::Index(i, _, _) => Self::root_slot(i),
        }
    }

    fn field_of(&mut self, t: &Ty, f: &str, sp: Span) -> KResult<(u32, Ty)> {
        match self.resolve(t) {
            Ty::Struct(sid) => {
                let def = &self.structs[sid as usize];
                match def.fields.iter().position(|(n, _)| n == f) {
                    Some(i) => Ok((i as u32, def.fields[i].1.clone())),
                    None => Err(Diag::new("E0105", sp, format!("struct `{}` has no field `{f}`", def.name))),
                }
            }
            other => {
                let hint = if matches!(other, Ty::Var(_)) {
                    "add a type annotation so Kairós knows the type here"
                } else {
                    "fields exist on structs; tuple elements are `.0`, `.1`, ..."
                };
                Err(Diag::new("E0105", sp, format!("type `{}` has no field `{f}`", self.show(&other))).hint(hint))
            }
        }
    }

    fn tuple_elem(&mut self, t: &Ty, k: usize, sp: Span) -> KResult<Ty> {
        match self.resolve(t) {
            Ty::Tuple(ts) if k < ts.len() => Ok(ts[k].clone()),
            Ty::Tuple(ts) => Err(Diag::new("E0105", sp, format!("tuple has {} element(s), there is no `.{k}`", ts.len()))),
            other => Err(Diag::new("E0105", sp, format!("type `{}` is not a tuple", self.show(&other)))
                .hint("add a type annotation so Kairós knows the type here")),
        }
    }

    // ------------------------------------------------------------------ expressions

    fn expr(&mut self, e: &Expr, exp: Option<&Ty>) -> KResult<(HExpr, Ty)> {
        let sp = e.span;
        match &e.kind {
            ExprKind::Int(v) => {
                let (t, id) = self.fresh(Kind::Num);
                if let Some(x) = exp {
                    let _ = self.unify(&t, x);
                }
                Ok((HExpr::IntLit(*v, id), t))
            }
            ExprKind::Float(v) => Ok((HExpr::Lit(Value::Float(*v)), Ty::Float)),
            ExprKind::Str(s) => Ok((HExpr::Lit(Value::Str(s.as_str().into())), Ty::Str)),
            ExprKind::Bool(b) => Ok((HExpr::Lit(Value::Bool(*b)), Ty::Bool)),
            ExprKind::Unit => Ok((HExpr::Lit(Value::Unit), Ty::Unit)),
            ExprKind::Name(n) => self.name_expr(n, sp),
            ExprKind::List(items) => {
                let elem = match exp.map(|t| self.resolve(t)) {
                    Some(Ty::List(t)) => *t,
                    _ => self.fresh(Kind::Any).0,
                };
                let mut hs = Vec::new();
                for it in items {
                    let (h, t) = self.expr(it, Some(&elem))?;
                    self.expect_ty(&t, &elem, it.span)?;
                    hs.push(h);
                }
                Ok((HExpr::List(hs), Ty::List(Box::new(elem))))
            }
            ExprKind::Tuple(items) => {
                let exp_elems = match exp.map(|t| self.resolve(t)) {
                    Some(Ty::Tuple(ts)) if ts.len() == items.len() => Some(ts),
                    _ => None,
                };
                let mut hs = Vec::new();
                let mut ts = Vec::new();
                for (i, it) in items.iter().enumerate() {
                    let (h, t) = self.expr(it, exp_elems.as_ref().map(|v| &v[i]))?;
                    hs.push(h);
                    ts.push(t);
                }
                Ok((HExpr::Tuple(hs), Ty::Tuple(ts)))
            }
            ExprKind::StructLit { name, fields } => self.struct_lit(name, fields, sp),
            ExprKind::Field(obj, f) => {
                if let ExprKind::Name(tn) = &obj.kind {
                    if self.lookup(tn).is_none() {
                        if let Some(&eid) = self.enum_ids.get(tn.as_str()) {
                            return self.variant_ctor(eid, f, &[], sp);
                        }
                    }
                }
                let (h, t) = self.expr(obj, None)?;
                let (i, ft) = self.field_of(&t, f, sp)?;
                Ok((HExpr::Field(Box::new(h), i), ft))
            }
            ExprKind::TupleIndex(obj, k) => {
                let (h, t) = self.expr(obj, None)?;
                let et = self.tuple_elem(&t, *k, sp)?;
                Ok((HExpr::Field(Box::new(h), *k as u32), et))
            }
            ExprKind::Index(obj, items) => {
                let (ho, to) = self.expr(obj, None)?;
                if let Some((axes, elem)) = self.tensor_parts(&to) {
                    let (sels, kept) = self.tensor_sels(&axes, &to, items)?;
                    return Ok((HExpr::TensorIndex(Box::new(ho), sels, sp), Self::mk_tensor(kept, elem)));
                }
                let idx = self.list_index(items, &to)?;
                let elem = self.list_elem(&to, obj.span)?;
                let (hi, ti) = self.expr(idx, Some(&Ty::Int))?;
                self.expect_ty(&ti, &Ty::Int, idx.span)?;
                Ok((HExpr::Index(Box::new(ho), Box::new(hi), sp), elem))
            }
            ExprKind::Call(callee, args) => self.call_expr(callee, args, sp),
            ExprKind::MethodCall { recv, name, args } => self.method_call(recv, name, args, sp),
            ExprKind::Unary(op, x) => {
                let (h, t) = self.expr(x, exp)?;
                if let Some((_, elem)) = self.tensor_parts(&t) {
                    let (want, word) = if *op == UnOp::Neg { (Ty::Float, "`-`") } else { (Ty::Bool, "`not`") };
                    if elem != want {
                        let s = self.show(&t);
                        return Err(Diag::new("E0101", sp, format!("{word} needs a `{}` tensor, found `{s}`", self.show(&want))));
                    }
                    return Ok((HExpr::Unary(*op, Box::new(h), sp), t));
                }
                match op {
                    UnOp::Neg if self.tangent(&t) => {}
                    UnOp::Neg => self.ensure_numeric(&t, sp, "`-`")?,
                    UnOp::Not => self.expect_ty(&t, &Ty::Bool, x.span)?,
                }
                Ok((HExpr::Unary(*op, Box::new(h), sp), t))
            }
            ExprKind::Binary(op, l, r) => self.binary(*op, l, r, sp),
            ExprKind::And(l, r) | ExprKind::Or(l, r) => {
                let is_and = matches!(e.kind, ExprKind::And(..));
                let (hl, tl) = self.expr(l, Some(&Ty::Bool))?;
                let lt = self.tensor_parts(&tl).is_some();
                if !lt {
                    self.expect_ty(&tl, &Ty::Bool, l.span)?;
                }
                let (hr, tr) = self.expr(r, Some(&Ty::Bool))?;
                if !lt && self.tensor_parts(&tr).is_none() {
                    self.expect_ty(&tr, &Ty::Bool, r.span)?;
                    let h = if is_and { HExpr::And(Box::new(hl), Box::new(hr)) } else { HExpr::Or(Box::new(hl), Box::new(hr)) };
                    return Ok((h, Ty::Bool));
                }
                // element-wise on bool tensors (no short-circuit)
                let word = if is_and { "`and`" } else { "`or`" };
                let (axes, elem) = self.combine(&[(&tl, &**l), (&tr, &**r)], word, sp)?;
                if elem != Ty::Bool {
                    return Err(Diag::new("E0101", sp, format!("{word} needs `bool` values, found `{}`", self.show(&elem))));
                }
                let b = if is_and { Builtin::TAnd } else { Builtin::TOr };
                Ok((HExpr::CallBuiltin(b, vec![HArg::Val(hl), HArg::Val(hr)], sp), Self::mk_tensor(axes, Ty::Bool)))
            }
            ExprKind::Assign { target, op, value } => {
                if let ExprKind::Index(obj, items) = &target.kind {
                    let (p, t) = self.place(obj)?;
                    if let Some((axes, elem)) = self.tensor_parts(&t) {
                        let (sels, kept) = self.tensor_sels(&axes, &t, items)?;
                        let (hv, tv) = self.expr(value, Some(&elem))?;
                        self.check_tensor_store(&kept, &elem, &tv, *op, true, value, sp)?;
                        return Ok((HExpr::TensorAssign(p, sels, *op, Box::new(hv), sp), Ty::Unit));
                    }
                }
                let (place, t) = self.place(target)?;
                if let Some((axes, elem)) = self.tensor_parts(&t) {
                    let (hv, tv) = self.expr(value, Some(&t))?;
                    self.check_tensor_store(&axes, &elem, &tv, *op, false, value, sp)?;
                    let h = match op {
                        None => HExpr::Assign(place, Box::new(hv)),
                        Some(bop) => HExpr::OpAssign(place, *bop, Box::new(hv), sp),
                    };
                    return Ok((h, Ty::Unit));
                }
                let (hv, tv) = self.expr(value, Some(&t))?;
                self.expect_ty(&tv, &t, value.span)?;
                match op {
                    None => Ok((HExpr::Assign(place, Box::new(hv)), Ty::Unit)),
                    Some(bop) => {
                        if !(matches!(self.resolve(&t), Ty::Str) && *bop == BinOp::Add) {
                            self.ensure_numeric(&t, sp, &format!("`{}=`", op_str(*bop)))?;
                        }
                        Ok((HExpr::OpAssign(place, *bop, Box::new(hv), sp), Ty::Unit))
                    }
                }
            }
            ExprKind::If { cond, then, els } => {
                let (hc, tc) = self.expr(cond, Some(&Ty::Bool))?;
                self.expect_ty(&tc, &Ty::Bool, cond.span)?;
                let (ht, tt) = self.block(then, exp)?;
                match els {
                    None => {
                        if self.unify(&tt, &Ty::Unit).is_err() {
                            let span = then.tail.as_ref().map(|t| t.span).unwrap_or(then.span);
                            return Err(mismatch(&self.show(&tt), "()", span)
                                .hint("an `if` without `else` produces no value: add an `else` branch"));
                        }
                        Ok((HExpr::If(Box::new(hc), Box::new(ht), None), Ty::Unit))
                    }
                    Some(eb) => {
                        let want = if matches!(self.resolve(&tt), Ty::Never) { exp.cloned() } else { Some(tt.clone()) };
                        let (he, te) = self.expr(eb, want.as_ref())?;
                        if self.unify(&tt, &te).is_err() {
                            let (a, b) = (self.show(&tt), self.show(&te));
                            return Err(Diag::new("E0101", eb.span, format!("`if` and `else` have incompatible types: `{a}` and `{b}`")));
                        }
                        let t = if matches!(self.resolve(&tt), Ty::Never) { te } else { tt };
                        Ok((HExpr::If(Box::new(hc), Box::new(ht), Some(Box::new(he))), t))
                    }
                }
            }
            ExprKind::While { cond, body } => {
                let (hc, tc) = self.expr(cond, Some(&Ty::Bool))?;
                self.expect_ty(&tc, &Ty::Bool, cond.span)?;
                let (hb, tb) = self.loop_body(body)?;
                self.expect_ty(&tb, &Ty::Unit, body.span)?;
                Ok((HExpr::While(Box::new(hc), Box::new(hb)), Ty::Unit))
            }
            ExprKind::For { pat, iter, body } => self.for_expr(pat, iter, body),
            ExprKind::Range { .. } => {
                Err(Diag::new("E0113", sp, "ranges can only be used in `for` loops and tensor indices (for now)"))
            }
            ExprKind::Match { scrut, arms } => self.match_expr(scrut, arms, sp, exp),
            ExprKind::Lambda { params, body } => self.lambda(params, body, exp),
            ExprKind::Block(b) => self.block(b, exp),
            ExprKind::Return(v) => {
                if self.ctx().kind == CtxKind::Main {
                    return Err(Diag::new("E0118", sp, "`return` outside of a function"));
                }
                let ret = self.ctx().ret.clone();
                let h = match v {
                    Some(x) => {
                        let (h, t) = self.expr(x, Some(&ret))?;
                        self.expect_ty(&t, &ret, x.span)?;
                        Some(Box::new(h))
                    }
                    None => {
                        self.expect_ty(&Ty::Unit, &ret, sp)?;
                        None
                    }
                };
                Ok((HExpr::Return(h), Ty::Never))
            }
            ExprKind::Break | ExprKind::Continue => {
                if self.ctx().loops == 0 {
                    return Err(Diag::new("E0119", sp, "`break` and `continue` must be inside a loop"));
                }
                let h = if matches!(e.kind, ExprKind::Break) { HExpr::Break } else { HExpr::Continue };
                Ok((h, Ty::Never))
            }
        }
    }

    fn loop_body(&mut self, body: &Block) -> KResult<(HExpr, Ty)> {
        self.ctx().loops += 1;
        let r = self.block(body, Some(&Ty::Unit));
        self.ctx().loops -= 1;
        r
    }

    fn for_expr(&mut self, pat: &Pattern, iter: &Expr, body: &Block) -> KResult<(HExpr, Ty)> {
        if let ExprKind::Range { lo, hi, inclusive } = &iter.kind {
            let (hlo, tlo) = self.expr(lo, Some(&Ty::Int))?;
            self.expect_ty(&tlo, &Ty::Int, lo.span)?;
            let (hhi, thi) = self.expr(hi, Some(&Ty::Int))?;
            self.expect_ty(&thi, &Ty::Int, hi.span)?;
            let inclusive = *inclusive;
            return self.with_scope(|me| {
                let slot = match pat {
                    Pattern::Bind(n, _) => me.declare(n, Ty::Int, false),
                    Pattern::Wild(_) => me.declare("_", Ty::Int, false),
                    other => return Err(Diag::new("E0117", other.span(), "a range loop binds a single name")),
                };
                let (hb, tb) = me.loop_body(body)?;
                me.expect_ty(&tb, &Ty::Unit, body.span)?;
                let name = match pat {
                    Pattern::Bind(n, _) => n.clone(),
                    _ => "_".to_string(),
                };
                Ok((HExpr::ForRange { slot, name, lo: Box::new(hlo), hi: Box::new(hhi), inclusive, body: Box::new(hb) }, Ty::Unit))
            });
        }
        let (hl, tl) = self.expr(iter, None)?;
        let elem = self.list_elem(&tl, iter.span)?;
        self.with_scope(|me| {
            let hp = me.pattern(pat, &elem, false, false)?;
            let (hb, tb) = me.loop_body(body)?;
            me.expect_ty(&tb, &Ty::Unit, body.span)?;
            Ok((HExpr::ForList { pat: hp, list: Box::new(hl), body: Box::new(hb) }, Ty::Unit))
        })
    }

    fn match_expr(&mut self, scrut: &Expr, arms: &[Arm], sp: Span, exp: Option<&Ty>) -> KResult<(HExpr, Ty)> {
        let (hs, ts) = self.expr(scrut, None)?;
        let mut result: Option<Ty> = exp.cloned();
        let mut harms = Vec::new();
        for arm in arms {
            let (hp, hb, tb) = self.with_scope(|me| {
                let hp = me.pattern(&arm.pat, &ts, false, true)?;
                let (hb, tb) = me.expr(&arm.body, result.as_ref())?;
                Ok((hp, hb, tb))
            })?;
            match result.clone() {
                Some(rt) => {
                    if self.unify(&tb, &rt).is_err() {
                        let (a, b) = (self.show(&rt), self.show(&tb));
                        return Err(Diag::new("E0101", arm.body.span, format!("match arms have incompatible types: `{a}` and `{b}`")));
                    }
                    if matches!(self.resolve(&rt), Ty::Never) {
                        result = Some(tb);
                    }
                }
                None => result = Some(tb),
            }
            harms.push((hp, hb));
        }
        let pats: Vec<HPat> = harms.iter().map(|(p, _)| p.clone()).collect();
        self.check_exhaustive(&ts, &pats, sp)?;
        Ok((HExpr::Match(Box::new(hs), harms, sp), result.unwrap_or(Ty::Unit)))
    }

    fn lambda(&mut self, params: &[(String, Option<TypeExpr>, Span)], body: &Expr, exp: Option<&Ty>) -> KResult<(HExpr, Ty)> {
        let expected = match exp.map(|t| self.resolve(t)) {
            Some(Ty::Fn(ps, r)) if ps.len() == params.len() => Some((ps, *r)),
            _ => None,
        };
        let mut ptys = Vec::new();
        for (i, (_, te, psp)) in params.iter().enumerate() {
            let t = match te {
                Some(te) => {
                    let t = self.resolve_type(te)?;
                    if let Some((ps, _)) = &expected {
                        let want = ps[i].clone();
                        self.expect_ty(&t, &want, *psp)?;
                    }
                    t
                }
                None => match &expected {
                    Some((ps, _)) => ps[i].clone(),
                    None => self.fresh(Kind::Any).0,
                },
            };
            ptys.push(t);
        }
        let ret = match &expected {
            Some((_, r)) => r.clone(),
            None => self.fresh(Kind::Any).0,
        };
        let mut ctx = Ctx::new(CtxKind::Lambda, ret.clone());
        for ((name, _, _), t) in params.iter().zip(ptys.iter()) {
            let slot = ctx.n_slots;
            ctx.n_slots += 1;
            ctx.scopes[0].insert(name.clone(), Local { slot, ty: t.clone(), mutable: false });
        }
        self.ctxs.push(ctx);
        let r = self.expr(body, Some(&ret));
        let ctx = self.ctxs.pop().expect("lambda context");
        let (hb, tb) = r?;
        self.expect_ty(&tb, &ret, body.span)?;
        let def = self.lambdas.len() as u32;
        let (outer, inner): (Vec<u32>, Vec<u32>) = ctx.captures.iter().copied().unzip();
        self.lambdas.push(LambdaBody { capture_slots: inner, n_params: params.len() as u32, n_slots: ctx.n_slots, body: hb });
        Ok((HExpr::Lambda(def, outer), Ty::Fn(ptys, Box::new(ret))))
    }

    fn name_expr(&mut self, n: &str, sp: Span) -> KResult<(HExpr, Ty)> {
        if let Some(l) = self.lookup(n) {
            return Ok((HExpr::Local(l.slot), l.ty));
        }
        if let Some(&fid) = self.fn_ids.get(n) {
            let sig = &self.sigs[fid as usize];
            if !sig.type_params.is_empty() {
                return Err(Diag::new("E0114", sp, format!("generic function `{n}` cannot be used as a value yet")).hint("call it directly, or wrap it in a lambda with concrete types"));
            }
            if sig.params.iter().any(|p| p.2) {
                return Err(Diag::new("E0114", sp, format!("`{n}` has `inout` parameters and cannot be used as a value")));
            }
            let t = Ty::Fn(sig.params.iter().map(|p| p.1.clone()).collect(), Box::new(sig.ret.clone()));
            return Ok((HExpr::FnRef(fid), t));
        }
        match n {
            "nan" => return Ok((HExpr::Lit(Value::Float(f64::NAN)), Ty::Float)),
            "inf" => return Ok((HExpr::Lit(Value::Float(f64::INFINITY)), Ty::Float)),
            "pi" => return Ok((HExpr::Lit(Value::Float(std::f64::consts::PI)), Ty::Float)),
            "b3" => {
                return Err(Diag::new("E0101", sp, "`b3` is the B3 data module, not a value")
                    .hint("call its functions, e.g. `b3.monthly_returns()`"));
            }
            _ => {}
        }
        if builtin_by_name(n).is_some() {
            return Err(Diag::new("E0114", sp, format!("builtin `{n}` cannot be used as a value yet"))
                .hint(format!("wrap it in a lambda, e.g. `|x| {n}(x)`")));
        }
        if self.axis_ids.contains_key(n) {
            return Err(Diag::new("E0101", sp, format!("`{n}` is an axis, not a value"))
                .hint(format!("axes appear in types, in indices like `x[{n}: i]` and in arguments like `over: {n}`")));
        }
        let mut d = Diag::new("E0100", sp, format!("cannot find `{n}` in this scope"));
        let in_fn = self.ctxs.first().is_some_and(|c| c.kind != CtxKind::Main);
        if in_fn && (self.top_names.contains(n) || self.main.scopes.iter().any(|s| s.contains_key(n))) {
            d = d.hint("functions cannot read top-level variables: pass it as a parameter");
        }
        Err(d)
    }

    fn struct_lit(&mut self, name: &str, fields: &[(String, Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        let Some(&sid) = self.struct_ids.get(name) else {
            return Err(Diag::new("E0100", sp, format!("unknown struct `{name}`")));
        };
        let def_fields = self.structs[sid as usize].fields.clone();
        let mut vals: Vec<Option<HExpr>> = (0..def_fields.len()).map(|_| None).collect();
        for (fname, fe) in fields {
            let Some(idx) = def_fields.iter().position(|(n, _)| n == fname) else {
                return Err(Diag::new("E0105", fe.span, format!("struct `{name}` has no field `{fname}`")));
            };
            if vals[idx].is_some() {
                return Err(Diag::new("E0112", fe.span, format!("field `{fname}` is given twice")));
            }
            let fty = def_fields[idx].1.clone();
            let (h, t) = self.expr(fe, Some(&fty))?;
            self.expect_ty(&t, &fty, fe.span)?;
            vals[idx] = Some(h);
        }
        if let Some(i) = vals.iter().position(|v| v.is_none()) {
            return Err(Diag::new("E0105", sp, format!("missing field `{}` in `{name}`", def_fields[i].0)));
        }
        Ok((HExpr::Struct(sid, vals.into_iter().map(|v| v.expect("field")).collect()), Ty::Struct(sid)))
    }

    fn variant_ctor(&mut self, eid: u32, vname: &str, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        let def = self.enums[eid as usize].clone();
        let Some(vi) = def.variants.iter().position(|(n, _)| n == vname) else {
            return Err(Diag::new("E0100", sp, format!("enum `{}` has no variant `{vname}`", def.name)));
        };
        let ptys = &def.variants[vi].1;
        if ptys.len() != args.len() {
            return Err(Diag::new(
                "E0103",
                sp,
                format!("`{}.{vname}` takes {} value(s), but {} were given", def.name, ptys.len(), args.len()),
            ));
        }
        Self::no_named_args(args, &format!("`{}.{vname}`", def.name))?;
        let mut hs = Vec::new();
        for (a, pt) in args.iter().zip(ptys.iter()) {
            if a.inout {
                return Err(Diag::new("E0115", a.expr.span, "enum values are not `inout`"));
            }
            let (h, t) = self.expr(&a.expr, Some(pt))?;
            self.expect_ty(&t, pt, a.expr.span)?;
            hs.push(h);
        }
        Ok((HExpr::Variant(eid, vi as u32, hs), Ty::Enum(eid)))
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr, sp: Span) -> KResult<(HExpr, Ty)> {
        let (hl, tl) = self.expr(l, None)?;
        let (hr, tr) = self.expr(r, Some(&tl))?;
        if self.tensor_parts(&tl).is_some() || self.tensor_parts(&tr).is_some() {
            let t = self.tensor_binary(op, &tl, &tr, l, r, sp)?;
            return Ok((HExpr::Binary(op, Box::new(hl), Box::new(hr), sp), t));
        }
        // the tangent space: structs/tuples/type parameters of differentiable values
        if let Some(t) = self.tangent_binary(op, &tl, &tr, sp)? {
            return Ok((HExpr::Binary(op, Box::new(hl), Box::new(hr), sp), t));
        }
        if self.unify(&tl, &tr).is_err() {
            let (a, b) = (self.show(&tl), self.show(&tr));
            let mixed = (a == "i64" && b == "f64") || (a == "f64" && b == "i64");
            let d = Diag::new(if mixed { "E0107" } else { "E0101" }, sp, format!("cannot apply `{}` to `{a}` and `{b}`", op_str(op)));
            return Err(if mixed { d.hint(MIX_HINT) } else { d });
        }
        let t = self.resolve(&tl);
        let res = match op {
            BinOp::Add => {
                if !matches!(t, Ty::Str) {
                    self.ensure_numeric(&t, sp, "`+`")?;
                }
                t
            }
            BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Rem | BinOp::Pow => {
                self.ensure_numeric(&t, sp, &format!("`{}`", op_str(op)))?;
                t
            }
            BinOp::Eq | BinOp::Ne => {
                if matches!(t, Ty::Fn(..)) {
                    return Err(Diag::new("E0101", sp, "functions cannot be compared"));
                }
                Ty::Bool
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                match t {
                    Ty::Int | Ty::Float | Ty::Str | Ty::Never => {}
                    Ty::Var(_) => self.ensure_numeric(&t, sp, "a comparison")?,
                    _ => return Err(Diag::new("E0101", sp, format!("`{}` values cannot be ordered", self.show(&t)))),
                }
                Ty::Bool
            }
        };
        Ok((HExpr::Binary(op, Box::new(hl), Box::new(hr), sp), res))
    }

    // ------------------------------------------------------------------ calls

    fn call_expr(&mut self, callee: &Expr, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        if let ExprKind::Name(n) = &callee.kind {
            if self.lookup(n).is_none() {
                if let Some(&fid) = self.fn_ids.get(n.as_str()) {
                    return self.call_user(fid, None, args, sp);
                }
                if let Some(b) = builtin_by_name(n) {
                    return self.call_builtin(b, None, args, sp);
                }
            }
        }
        let (hc, tc) = self.expr(callee, None)?;
        let (ps, r) = match self.resolve(&tc) {
            Ty::Fn(ps, r) => (ps, *r),
            Ty::Var(_) => {
                let ps: Vec<Ty> = args.iter().map(|_| self.fresh(Kind::Any).0).collect();
                let r = self.fresh(Kind::Any).0;
                self.expect_ty(&tc, &Ty::Fn(ps.clone(), Box::new(r.clone())), callee.span)?;
                (ps, r)
            }
            other => return Err(Diag::new("E0102", callee.span, format!("`{}` is not a function", self.show(&other)))),
        };
        Self::no_named_args(args, "a function value")?;
        if ps.len() != args.len() {
            return Err(Diag::new("E0103", sp, format!("this function takes {} argument(s), but {} were given", ps.len(), args.len())));
        }
        let mut hargs = Vec::new();
        for (a, p) in args.iter().zip(ps.iter()) {
            if a.inout {
                return Err(Diag::new("E0115", a.expr.span, "only named functions take `inout` arguments"));
            }
            let (h, t) = self.expr(&a.expr, Some(p))?;
            self.expect_ty(&t, p, a.expr.span)?;
            hargs.push(h);
        }
        Ok((HExpr::CallValue(Box::new(hc), hargs, sp), r))
    }

    /// `x.f(a)` is `f(x, a)` (uniform call syntax); `Enum.Variant(a)` builds an enum value.
    fn method_call(&mut self, recv: &Expr, name: &str, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        if let ExprKind::Name(tn) = &recv.kind {
            if self.lookup(tn).is_none() {
                if tn == "b3" {
                    return self.b3_call(name, args, sp);
                }
                if let Some(&eid) = self.enum_ids.get(tn.as_str()) {
                    return self.variant_ctor(eid, name, args, sp);
                }
            }
        }
        if let Some(&fid) = self.fn_ids.get(name) {
            return self.call_user(fid, Some(recv), args, sp);
        }
        if let Some(b) = builtin_by_name(name) {
            return self.call_builtin(b, Some(recv), args, sp);
        }
        Err(Diag::new("E0100", sp, format!("no function named `{name}`")).hint("`x.f(y)` calls the function `f(x, y)`"))
    }

    /// Replaces type parameters by their instances.
    fn subst_params(&self, t: &Ty, inst: &[(u32, Ty)]) -> Ty {
        match t {
            Ty::Param(p) => inst.iter().find(|(q, _)| q == p).map(|(_, t)| t.clone()).unwrap_or(Ty::Param(*p)),
            Ty::List(x) => Ty::List(Box::new(self.subst_params(x, inst))),
            Ty::Tuple(xs) => Ty::Tuple(xs.iter().map(|x| self.subst_params(x, inst)).collect()),
            Ty::Fn(ps, r) => Ty::Fn(ps.iter().map(|x| self.subst_params(x, inst)).collect(), Box::new(self.subst_params(r, inst))),
            Ty::Tensor(a, e) => Ty::Tensor(a.clone(), Box::new(self.subst_params(e, inst))),
            other => other.clone(),
        }
    }

    fn call_user(&mut self, fid: u32, recv: Option<&Expr>, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        let mut sig = self.sigs[fid as usize].clone();
        // a generic function: fresh instances of its type parameters at every call
        let inst: Vec<(u32, Ty)> = sig.type_params.iter().map(|p| (*p, self.fresh(Kind::Any).0)).collect();
        if !inst.is_empty() {
            for p in sig.params.iter_mut() {
                p.1 = self.subst_params(&p.1, &inst);
            }
            sig.ret = self.subst_params(&sig.ret, &inst);
        }
        let (h, t) = self.call_user_sig(fid, &sig, recv, args, sp)?;
        for (p, t) in &inst {
            if !self.differentiable(t) {
                let name = &self.type_param_names[*p as usize];
                return Err(Diag::new("E0402", sp, format!("`{}` needs `{name}` to be differentiable, found `{}`", sig.name, self.show(t)))
                    .hint("differentiable types: f64, f64 tensors, and structs, tuples and lists of them"));
            }
        }
        Ok((h, t))
    }

    fn call_user_sig(&mut self, fid: u32, sig: &FnSig, recv: Option<&Expr>, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        Self::no_named_args(args, &format!("`{}`", sig.name))?;
        let mut all: Vec<(&Expr, bool, bool)> = Vec::new();
        if let Some(r) = recv {
            all.push((r, false, true));
        }
        for a in args {
            all.push((&a.expr, a.inout, false));
        }
        if all.len() != sig.params.len() {
            return Err(Diag::new(
                "E0103",
                sp,
                format!("`{}` takes {} argument(s), but {} were given", sig.name, sig.params.len(), all.len()),
            ));
        }
        let mut hargs = Vec::new();
        let mut roots: Vec<u32> = Vec::new();
        for ((e, explicit, is_recv), (pname, pty, pin)) in all.iter().zip(sig.params.iter()) {
            if *pin {
                if !explicit && !is_recv {
                    return Err(Diag::new("E0115", e.span, format!("parameter `{pname}` of `{}` is `inout`", sig.name))
                        .hint("pass it as `inout x`; the function will modify `x`"));
                }
                let (place, t) = self.place(e)?;
                self.expect_ty(&t, pty, e.span)?;
                let root = Self::root_slot(&place);
                if roots.contains(&root) {
                    return Err(Diag::new("E0111", e.span, "the same variable is passed as `inout` twice"));
                }
                roots.push(root);
                hargs.push(HArg::InOut(place));
            } else {
                if *explicit {
                    return Err(Diag::new("E0115", e.span, format!("parameter `{pname}` of `{}` is not `inout`", sig.name)));
                }
                let (h, t) = self.expr(e, Some(pty))?;
                self.expect_ty(&t, pty, e.span)?;
                hargs.push(HArg::Val(h));
            }
        }
        Ok((HExpr::CallFn(fid, hargs, sp), sig.ret.clone()))
    }

    fn call_builtin(&mut self, b: Builtin, recv: Option<&Expr>, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        let name = builtin_name(b);
        let mut all: Vec<(&Expr, bool)> = Vec::new();
        let mut named: Vec<(&str, Span, &Expr)> = Vec::new();
        if let Some(r) = recv {
            all.push((r, true));
        }
        for a in args {
            match &a.name {
                Some((nm, nsp)) => {
                    if a.inout {
                        return Err(Diag::new("E0115", a.expr.span, "named arguments are never `inout`"));
                    }
                    if named.iter().any(|(m, _, _)| *m == nm.as_str()) {
                        return Err(Diag::new("E0112", *nsp, format!("argument `{nm}` is given twice")));
                    }
                    named.push((nm.as_str(), *nsp, &a.expr));
                }
                None => {
                    if !named.is_empty() {
                        return Err(Diag::new("E0120", a.expr.span, "positional arguments must come before named ones"));
                    }
                    all.push((&a.expr, a.inout));
                }
            }
        }
        let mutating = matches!(b, Builtin::Push | Builtin::Pop | Builtin::Sort | Builtin::Reverse);
        for (i, (e, flag)) in all.iter().enumerate() {
            let is_recv = recv.is_some() && i == 0;
            if mutating && i == 0 {
                if !flag {
                    return Err(Diag::new("E0115", e.span, format!("`{name}` modifies its first argument"))
                        .hint(format!("write `x.{name}(...)` or `{name}(inout x, ...)`")));
                }
            } else if *flag && !is_recv {
                return Err(Diag::new("E0115", e.span, format!("this argument of `{name}` is not `inout`")));
            }
        }
        if Self::tensor_aware(b) {
            return self.tensor_builtin(b, &all, &named, sp);
        }
        if let Some((nm, nsp, _)) = named.first() {
            return Err(Diag::new("E0120", *nsp, format!("`{name}` does not take named arguments"))
                .hint(format!("pass `{nm}` by position")));
        }
        let n = all.len();
        let arity = |want: usize| -> KResult<()> {
            if n != want {
                Err(Diag::new("E0103", sp, format!("`{name}` takes {want} argument(s), but {n} were given")))
            } else {
                Ok(())
            }
        };
        let (hargs, ty): (Vec<HArg>, Ty) = match b {
            Builtin::Print => {
                let mut hs = Vec::new();
                for (e, _) in &all {
                    let (h, t) = self.expr(e, None)?;
                    self.default_nums(&t);
                    hs.push(HArg::Val(h));
                }
                (hs, Ty::Unit)
            }
            Builtin::Len => {
                arity(1)?;
                let (h, t) = self.expr(all[0].0, None)?;
                match self.resolve(&t) {
                    Ty::Str | Ty::List(_) => {}
                    _ => {
                        self.list_elem(&t, all[0].0.span)?;
                    }
                }
                (vec![HArg::Val(h)], Ty::Int)
            }
            Builtin::Push | Builtin::Pop | Builtin::Sort | Builtin::Reverse => {
                arity(if b == Builtin::Push { 2 } else { 1 })?;
                let (place, t) = self.place(all[0].0)?;
                let elem = self.list_elem(&t, all[0].0.span)?;
                let mut hs = vec![HArg::InOut(place)];
                let ty = match b {
                    Builtin::Push => {
                        let (h, te) = self.expr(all[1].0, Some(&elem))?;
                        self.expect_ty(&te, &elem, all[1].0.span)?;
                        hs.push(HArg::Val(h));
                        Ty::Unit
                    }
                    Builtin::Pop => elem,
                    _ => Ty::Unit,
                };
                (hs, ty)
            }
            Builtin::Contains => {
                arity(2)?;
                let (hl, tl) = self.expr(all[0].0, None)?;
                let elem = self.list_elem(&tl, all[0].0.span)?;
                let (hx, tx) = self.expr(all[1].0, Some(&elem))?;
                self.expect_ty(&tx, &elem, all[1].0.span)?;
                (vec![HArg::Val(hl), HArg::Val(hx)], Ty::Bool)
            }
            Builtin::Map | Builtin::Filter => {
                arity(2)?;
                let (hl, tl) = self.expr(all[0].0, None)?;
                let elem = self.list_elem(&tl, all[0].0.span)?;
                let out = if b == Builtin::Map { self.fresh(Kind::Any).0 } else { Ty::Bool };
                let want = Ty::Fn(vec![elem.clone()], Box::new(out.clone()));
                let (hf, tf) = self.expr(all[1].0, Some(&want))?;
                self.expect_ty(&tf, &want, all[1].0.span)?;
                let ty = if b == Builtin::Map { Ty::List(Box::new(out)) } else { Ty::List(Box::new(elem)) };
                (vec![HArg::Val(hl), HArg::Val(hf)], ty)
            }
            Builtin::Minimum | Builtin::Maximum => {
                arity(1)?;
                let (hl, tl) = self.expr(all[0].0, None)?;
                let elem = self.list_elem(&tl, all[0].0.span)?;
                self.ensure_numeric(&elem, all[0].0.span, &format!("`{name}`"))?;
                (vec![HArg::Val(hl)], elem)
            }
            Builtin::ToFloat | Builtin::ToInt => {
                arity(1)?;
                let (h, t) = self.expr(all[0].0, None)?;
                self.ensure_numeric(&t, all[0].0.span, &format!("`{name}`"))?;
                self.default_nums(&t);
                (vec![HArg::Val(h)], if b == Builtin::ToFloat { Ty::Float } else { Ty::Int })
            }
            Builtin::ToStr => {
                arity(1)?;
                let (h, t) = self.expr(all[0].0, None)?;
                self.default_nums(&t);
                (vec![HArg::Val(h)], Ty::Str)
            }
            Builtin::Assert => {
                if n != 1 && n != 2 {
                    return Err(Diag::new("E0103", sp, "`assert` takes a condition and an optional message"));
                }
                let (hc, tc) = self.expr(all[0].0, Some(&Ty::Bool))?;
                self.expect_ty(&tc, &Ty::Bool, all[0].0.span)?;
                let mut hs = vec![HArg::Val(hc)];
                if n == 2 {
                    let (hm, tm) = self.expr(all[1].0, Some(&Ty::Str))?;
                    self.expect_ty(&tm, &Ty::Str, all[1].0.span)?;
                    hs.push(HArg::Val(hm));
                }
                (hs, Ty::Unit)
            }
            _ => unreachable!("tensor-aware builtins are checked in types/tensors.rs"),
        };
        Ok((HExpr::CallBuiltin(b, hargs, sp), ty))
    }

    // ------------------------------------------------------------------ finalization

    fn fin_lit(&self, v: i64, var: u32) -> Value {
        match self.resolve(&Ty::Var(var)) {
            Ty::Float => Value::Float(v as f64),
            _ => Value::Int(v),
        }
    }

    fn fin_pat(&self, p: &mut HPat) {
        match p {
            HPat::IntLit(v, var) => {
                let lit = self.fin_lit(*v, *var);
                *p = HPat::Lit(lit);
            }
            HPat::Tuple(ps) | HPat::Variant(_, _, ps) => {
                for q in ps {
                    self.fin_pat(q);
                }
            }
            _ => {}
        }
    }

    fn fin_place(&self, p: &mut HPlace) {
        match p {
            HPlace::Local(_) => {}
            HPlace::Field(inner, _) => self.fin_place(inner),
            HPlace::Index(inner, i, _) => {
                self.fin_place(inner);
                self.fin_expr(i);
            }
        }
    }

    fn fin_sels(&self, sels: &mut [(u32, HSel)]) {
        for (_, s) in sels {
            match s {
                HSel::At(i) => self.fin_expr(i),
                HSel::Range(a, b, _) => {
                    self.fin_expr(a);
                    self.fin_expr(b);
                }
            }
        }
    }

    fn fin_args(&self, args: &mut [HArg]) {
        for a in args {
            match a {
                HArg::Val(e) => self.fin_expr(e),
                HArg::InOut(p) => self.fin_place(p),
            }
        }
    }

    fn fin_expr(&self, e: &mut HExpr) {
        match e {
            HExpr::IntLit(v, var) => {
                let lit = self.fin_lit(*v, *var);
                *e = HExpr::Lit(lit);
            }
            HExpr::Lit(_) | HExpr::Local(_) | HExpr::FnRef(_) | HExpr::Lambda(..) | HExpr::Break | HExpr::Continue => {}
            HExpr::List(xs) | HExpr::Tuple(xs) | HExpr::Struct(_, xs) | HExpr::Variant(_, _, xs) => {
                for x in xs {
                    self.fin_expr(x);
                }
            }
            HExpr::Field(x, _) | HExpr::Unary(_, x, _) => self.fin_expr(x),
            HExpr::Index(a, b, _) | HExpr::Binary(_, a, b, _) | HExpr::And(a, b) | HExpr::Or(a, b) | HExpr::While(a, b) => {
                self.fin_expr(a);
                self.fin_expr(b);
            }
            HExpr::TensorIndex(x, sels, _) => {
                self.fin_expr(x);
                self.fin_sels(sels);
            }
            HExpr::TensorAssign(p, sels, _, v, _) => {
                self.fin_place(p);
                self.fin_sels(sels);
                self.fin_expr(v);
            }
            HExpr::Host(_, xs, _) => {
                for x in xs {
                    self.fin_expr(x);
                }
            }
            HExpr::CallFn(_, args, _) | HExpr::CallBuiltin(_, args, _) => self.fin_args(args),
            HExpr::CallValue(c, args, _) => {
                self.fin_expr(c);
                for a in args {
                    self.fin_expr(a);
                }
            }
            HExpr::Assign(p, v) | HExpr::OpAssign(p, _, v, _) => {
                self.fin_place(p);
                self.fin_expr(v);
            }
            HExpr::If(c, t, els) => {
                self.fin_expr(c);
                self.fin_expr(t);
                if let Some(x) = els {
                    self.fin_expr(x);
                }
            }
            HExpr::ForRange { lo, hi, body, .. } => {
                self.fin_expr(lo);
                self.fin_expr(hi);
                self.fin_expr(body);
            }
            HExpr::ForList { pat, list, body } => {
                self.fin_pat(pat);
                self.fin_expr(list);
                self.fin_expr(body);
            }
            HExpr::Match(s, arms, _) => {
                self.fin_expr(s);
                for (p, b) in arms {
                    self.fin_pat(p);
                    self.fin_expr(b);
                }
            }
            HExpr::Block(stmts, tail) => {
                for s in stmts {
                    self.fin_expr(s);
                }
                if let Some(t) = tail {
                    self.fin_expr(t);
                }
            }
            HExpr::Let(p, init) => {
                self.fin_pat(p);
                self.fin_expr(init);
            }
            HExpr::Return(v) => {
                if let Some(x) = v {
                    self.fin_expr(x);
                }
            }
        }
    }
}
