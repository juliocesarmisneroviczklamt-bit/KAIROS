//! Type rules for named-axis tensors (M2) and for the `b3` data module.
//!
//! A tensor type is `Tensor[axes; f64|bool]` with the axis ids kept sorted, so the order in which
//! axes are written never matters. Element-wise operations need one operand's axes to contain
//! every other operand's (smaller operands repeat along the missing axes); anything else is E0201.
//! Reductions, indices, ranks and contractions always name the axis. Unknown axes are E0200, and
//! a tensor indexed by position instead of by axis name is E0202.

use super::*;

pub const AX_MONTH: u32 = 0;
pub const AX_DAY: u32 = 1;
pub const AX_ASSET: u32 = 2;

fn expr_name(e: &Expr) -> Option<&str> {
    match &e.kind {
        ExprKind::Name(n) => Some(n),
        _ => None,
    }
}

impl Checker {
    pub(super) fn builtin_axes() -> (Vec<AxisDef>, HashMap<String, u32>) {
        let axes = vec![
            AxisDef { name: "month".into(), temporal: Some(Freq::Monthly) },
            AxisDef { name: "day".into(), temporal: Some(Freq::Daily) },
            AxisDef { name: "asset".into(), temporal: None },
        ];
        let ids = axes.iter().enumerate().map(|(i, a)| (a.name.clone(), i as u32)).collect();
        (axes, ids)
    }

    /// `axis name` or `axis name: temporal(freq)`. Declaring an existing axis again exactly as it
    /// is (e.g. the built-in `month`) is allowed, so a program can state the axes it relies on.
    pub(super) fn declare_axis(&mut self, a: &AxisDecl) -> KResult<()> {
        let temporal = match &a.temporal {
            None => None,
            Some((f, fsp)) => Some(match f.as_str() {
                "daily" => Freq::Daily,
                "monthly" => Freq::Monthly,
                other => {
                    return Err(Diag::new("E0001", *fsp, format!("unknown frequency `{other}`"))
                        .hint("use `temporal(daily)` or `temporal(monthly)`"));
                }
            }),
        };
        if let Some(&id) = self.axis_ids.get(&a.name) {
            let old = self.axes[id as usize].temporal;
            if old == temporal {
                return Ok(());
            }
            let show = |t: Option<Freq>| match t {
                None => "a plain axis",
                Some(Freq::Daily) => "`temporal(daily)`",
                Some(Freq::Monthly) => "`temporal(monthly)`",
            };
            return Err(Diag::new("E0112", a.span, format!("axis `{}` is already defined as {}", a.name, show(old))));
        }
        let id = self.axes.len() as u32;
        self.axes.push(AxisDef { name: a.name.clone(), temporal });
        self.axis_ids.insert(a.name.clone(), id);
        Ok(())
    }

    pub(super) fn resolve_tensor_type(&mut self, axes: &[(String, Span)], elem: &TypeExpr, span: Span) -> KResult<Ty> {
        if axes.is_empty() {
            return Err(Diag::new("E0200", span, "a tensor type needs at least one axis").hint("e.g. `Tensor[asset; f64]`"));
        }
        let mut ids = Vec::new();
        for (n, nsp) in axes {
            let id = self.axis_id(n, *nsp)?;
            if ids.contains(&id) {
                return Err(Diag::new("E0112", *nsp, format!("axis `{n}` appears twice")));
            }
            ids.push(id);
        }
        let e = self.resolve_type(elem)?;
        if !matches!(e, Ty::Float | Ty::Bool) {
            return Err(Diag::new("E0101", elem.span(), format!("tensor elements must be `f64` or `bool`, found `{}`", self.show(&e))));
        }
        Ok(Self::mk_tensor(ids, e))
    }

    pub(super) fn axis_id(&self, n: &str, sp: Span) -> KResult<u32> {
        self.axis_ids.get(n).copied().ok_or_else(|| {
            Diag::new("E0200", sp, format!("unknown axis `{n}`"))
                .hint(format!("declare it with `axis {n}`; the built-in axes are `month`, `day` and `asset`"))
        })
    }

    /// An axis written as an argument: `over: asset`, `size(x, month)`.
    pub(super) fn axis_of_expr(&self, e: &Expr) -> KResult<u32> {
        match &e.kind {
            ExprKind::Name(n) => self.axis_id(n, e.span),
            _ => Err(Diag::new("E0200", e.span, "expected an axis name").hint("e.g. `over: asset`")),
        }
    }

    pub(super) fn tensor_parts(&self, t: &Ty) -> Option<(Vec<u32>, Ty)> {
        match self.resolve(t) {
            Ty::Tensor(a, e) => Some((a, *e)),
            _ => None,
        }
    }

    /// Element type of a tensor, or the type itself for a scalar.
    pub(super) fn elem_of(&self, t: &Ty) -> Ty {
        match self.resolve(t) {
            Ty::Tensor(_, e) => *e,
            other => other,
        }
    }

    /// Tensor type over these axes; with no axes left it is a scalar.
    pub(super) fn mk_tensor(mut axes: Vec<u32>, elem: Ty) -> Ty {
        if axes.is_empty() {
            elem
        } else {
            axes.sort_unstable();
            Ty::Tensor(axes, Box::new(elem))
        }
    }

    fn axis_label(&self, id: u32) -> &str {
        &self.axes[id as usize].name
    }

    pub(super) fn axes_str(&self, axes: &[u32]) -> String {
        let names: Vec<&str> = axes.iter().map(|a| self.axis_label(*a)).collect();
        format!("[{}]", names.join(", "))
    }

    /// Element type for a new tensor: an integer literal or unknown becomes `f64`; only `f64` and
    /// `bool` are allowed.
    fn tensor_elem(&mut self, t: &Ty, sp: Span) -> KResult<Ty> {
        let mut e = self.resolve(t);
        if let Ty::Var(_) = e {
            let _ = self.unify(&e, &Ty::Float);
            e = Ty::Float;
        }
        if !matches!(e, Ty::Float | Ty::Bool) {
            return Err(Diag::new("E0101", sp, format!("tensor elements must be `f64` or `bool`, found `{}`", self.show(&e))));
        }
        Ok(e)
    }

    pub(super) fn no_named_args(args: &[Arg], what: &str) -> KResult<()> {
        match args.iter().find_map(|a| a.name.as_ref()) {
            Some((n, sp)) => Err(Diag::new("E0120", *sp, format!("{what} does not take named arguments"))
                .hint(format!("pass `{n}` by position; named arguments like `over: asset` are for builtin functions"))),
            None => Ok(()),
        }
    }

    /// The single positional index of a list (`xs[i]`).
    pub(super) fn list_index<'e>(&self, items: &'e [IndexArg], t: &Ty) -> KResult<&'e Expr> {
        if let Some((ax, sp)) = items.iter().find_map(|i| i.axis.as_ref()) {
            return Err(match self.resolve(t) {
                Ty::Var(_) => Diag::new("E0202", *sp, format!("cannot index by axis `{ax}` here: the type of this value is not known yet"))
                    .hint("add a type annotation, e.g. `x: Tensor[month, asset; f64]`"),
                _ => Diag::new("E0202", *sp, format!("`{}` has no named axes", self.show(t)))
                    .hint("only tensors are indexed by axis; lists take a position: `xs[i]`"),
            });
        }
        if items.len() > 1 {
            return Err(Diag::new("E0103", items[1].expr.span, "a list takes a single index").hint("for a list of lists write `xs[i][j]`"));
        }
        Ok(&items[0].expr)
    }

    fn int_expr(&mut self, e: &Expr) -> KResult<HExpr> {
        let (h, t) = self.expr(e, Some(&Ty::Int))?;
        self.expect_ty(&t, &Ty::Int, e.span)?;
        Ok(h)
    }

    /// Selectors of `x[month: i, asset: a .. b]` and the axes that remain.
    pub(super) fn tensor_sels(&mut self, axes: &[u32], t: &Ty, items: &[IndexArg]) -> KResult<(Vec<(u32, HSel)>, Vec<u32>)> {
        let mut sels: Vec<(u32, HSel)> = Vec::new();
        let mut kept = axes.to_vec();
        for it in items {
            let Some((an, asp)) = &it.axis else {
                let first = self.axis_label(axes[0]).to_string();
                return Err(Diag::new("E0202", it.expr.span, "tensor indices must name their axis")
                    .label(format!("`{}` has axes {}", self.show(t), self.axes_str(axes)))
                    .hint(format!("write `x[{first}: i]`: axes have names, not positions")));
            };
            let id = self.axis_id(an, *asp)?;
            if !axes.contains(&id) {
                return Err(Diag::new("E0201", *asp, format!("`{}` has no axis `{an}`", self.show(t))));
            }
            if sels.iter().any(|(a, _)| *a == id) {
                return Err(Diag::new("E0112", *asp, format!("axis `{an}` is indexed twice")));
            }
            let sel = match &it.expr.kind {
                ExprKind::Range { lo, hi, inclusive } => {
                    let lo_h = self.int_expr(lo)?;
                    let hi_h = self.int_expr(hi)?;
                    HSel::Range(lo_h, hi_h, *inclusive)
                }
                _ => {
                    kept.retain(|a| *a != id);
                    HSel::At(self.int_expr(&it.expr)?)
                }
            };
            sels.push((id, sel));
        }
        Ok((sels, kept))
    }

    fn axes_mismatch(&self, what: &str, a: (&[u32], &Expr), b: (&[u32], &Expr), sp: Span) -> Diag {
        let label = match (expr_name(a.1), expr_name(b.1)) {
            (Some(x), Some(y)) if x != y => format!("`{x}` has axes {}, `{y}` has {}", self.axes_str(a.0), self.axes_str(b.0)),
            _ => format!("one side has axes {}, the other has {}", self.axes_str(a.0), self.axes_str(b.0)),
        };
        Diag::new("E0201", sp, format!("incompatible axes in {what}")).label(label).hint(
            "element-wise operations need one side's axes to contain the other's: reduce with `sum(x, over: ...)`, select with `x[axis: i]` or use `contract(x, w, over: ...)`",
        )
    }

    /// Axes of an element-wise combination: the tensor operand with the most axes must contain
    /// every other tensor operand's axes. Scalars repeat everywhere.
    pub(super) fn frame_axes(&self, ops: &[(&Ty, &Expr)], what: &str, sp: Span) -> KResult<Vec<u32>> {
        let parts: Vec<(usize, Vec<u32>)> =
            ops.iter().enumerate().filter_map(|(i, (t, _))| self.tensor_parts(t).map(|(a, _)| (i, a))).collect();
        let Some(bi) = (0..parts.len()).max_by_key(|&k| parts[k].1.len()) else {
            return Ok(Vec::new());
        };
        for k in 0..parts.len() {
            if !parts[k].1.iter().all(|x| parts[bi].1.contains(x)) {
                let (p, q) = if parts[k].0 < parts[bi].0 { (k, bi) } else { (bi, k) };
                return Err(self.axes_mismatch(what, (&parts[p].1, ops[parts[p].0].1), (&parts[q].1, ops[parts[q].0].1), sp));
            }
        }
        Ok(parts[bi].1.clone())
    }

    /// Like `frame_axes`, plus agreement of the element types (for `and` / `or`).
    pub(super) fn combine(&mut self, ops: &[(&Ty, &Expr)], what: &str, sp: Span) -> KResult<(Vec<u32>, Ty)> {
        let axes = self.frame_axes(ops, what, sp)?;
        let first = self.elem_of(ops[0].0);
        for (t, e) in &ops[1..] {
            let et = self.elem_of(t);
            if self.unify(&first, &et).is_err() {
                let (a, b) = (self.show(&first), self.show(&et));
                return Err(Diag::new("E0101", e.span, format!("{what} mixes `{a}` and `{b}` values")));
            }
        }
        Ok((axes, self.resolve(&first)))
    }

    /// `x op y` where at least one side is a tensor.
    pub(super) fn tensor_binary(&mut self, op: BinOp, tl: &Ty, tr: &Ty, l: &Expr, r: &Expr, sp: Span) -> KResult<Ty> {
        let sym = op_str(op);
        let what = match (expr_name(l), expr_name(r)) {
            (Some(x), Some(y)) => format!("`{x} {sym} {y}`"),
            _ => format!("`{sym}`"),
        };
        let axes = self.frame_axes(&[(tl, l), (tr, r)], &what, sp)?;
        let (el, er) = (self.elem_of(tl), self.elem_of(tr));
        if self.unify(&el, &er).is_err() {
            let (a, b) = (self.show(tl), self.show(tr));
            let (x, y) = (self.show(&el), self.show(&er));
            let mixed = (x == "i64" && y == "f64") || (x == "f64" && y == "i64");
            let d = Diag::new(if mixed { "E0107" } else { "E0101" }, sp, format!("cannot apply `{sym}` to `{a}` and `{b}`"));
            return Err(if mixed { d.hint(MIX_HINT) } else { d });
        }
        let elem = self.resolve(&el);
        let res = match op {
            BinOp::Eq | BinOp::Ne => Ty::Bool,
            _ => {
                if elem != Ty::Float {
                    return Err(Diag::new("E0101", sp, format!("`{sym}` needs `f64` values, found `{}`", self.show(&elem))));
                }
                if matches!(op, BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge) { Ty::Bool } else { Ty::Float }
            }
        };
        Ok(Self::mk_tensor(axes, res))
    }

    /// Store into a tensor: `slice` is a selection like `x[month: i] = v`, otherwise the whole
    /// variable. A slice takes a scalar (filling it) or a tensor with exactly its axes; `op=` also
    /// accepts any tensor whose axes the destination contains. A whole variable keeps its type.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_tensor_store(
        &mut self,
        dst: &[u32],
        elem: &Ty,
        tv: &Ty,
        op: Option<BinOp>,
        slice: bool,
        value: &Expr,
        sp: Span,
    ) -> KResult<()> {
        let dst_ty = Self::mk_tensor(dst.to_vec(), elem.clone());
        if let Some(bop) = op {
            if *elem != Ty::Float {
                let s = self.show(&dst_ty);
                return Err(Diag::new("E0101", sp, format!("`{}=` needs `f64` values, found `{s}`", op_str(bop))));
            }
            return match self.tensor_parts(tv) {
                Some((va, ve)) if ve == Ty::Float && va.iter().all(|a| dst.contains(a)) => Ok(()),
                Some(_) => Err(Diag::new("E0201", value.span, format!("cannot apply `{}=` with `{}` to `{}`", op_str(bop), self.show(tv), self.show(&dst_ty)))
                    .hint("the right side's axes must be among the destination's")),
                None => self.expect_ty(tv, &Ty::Float, value.span),
            };
        }
        match self.tensor_parts(tv) {
            Some((va, ve)) => {
                if va.as_slice() != dst {
                    Err(Diag::new("E0201", value.span, format!("cannot store `{}` into `{}`", self.show(tv), self.show(&dst_ty)))
                        .hint("the stored tensor must have exactly the destination's axes"))
                } else if ve != *elem {
                    Err(mismatch(&self.show(tv), &self.show(&dst_ty), value.span))
                } else {
                    Ok(())
                }
            }
            None if slice || dst.is_empty() => self.expect_ty(tv, elem, value.span),
            None => Err(mismatch(&self.show(tv), &self.show(&dst_ty), value.span)
                .hint("to set every element write `x = full_like(x, v)`, or assign a slice: `x[month: i] = v`")),
        }
    }

    pub(super) fn tensor_aware(b: Builtin) -> bool {
        use Builtin as B;
        matches!(
            b,
            B::Sum
                | B::Mean
                | B::Prod
                | B::Max
                | B::Min
                | B::NanSum
                | B::NanMean
                | B::NanProd
                | B::NanCount
                | B::Count
                | B::Abs
                | B::Sqrt
                | B::Exp
                | B::Ln
                | B::Floor
                | B::Ceil
                | B::Round
                | B::IsNan
                | B::FillNan
                | B::Where
                | B::Size
                | B::Zeros
                | B::Ones
                | B::Full
                | B::ZerosLike
                | B::FullLike
                | B::Tensor
                | B::ToList
                | B::Rank
                | B::TopK
                | B::BottomK
                | B::Contract
                | B::Observed
                | B::Grad
                | B::ValueAndGrad
                | B::Tanh
                | B::Sigmoid
                | B::Relu
                | B::Softplus
                | B::Randn
                | B::SmoothMax
                | B::SmoothMin
                | B::SmoothStep
        )
    }

    /// f64, f64 tensors, and structs, tuples and lists of them (the types `grad` works on).
    pub(super) fn differentiable(&self, t: &Ty) -> bool {
        fn go(ck: &Checker, t: &Ty, seen: &mut Vec<u32>) -> bool {
            match ck.resolve(t) {
                Ty::Float => true,
                Ty::Tensor(_, e) => *e == Ty::Float,
                Ty::Struct(id) => {
                    if seen.contains(&id) {
                        return false;
                    }
                    seen.push(id);
                    let ok = ck.structs[id as usize].fields.iter().all(|(_, ft)| go(ck, ft, seen));
                    seen.pop();
                    ok
                }
                Ty::Tuple(ts) => ts.iter().all(|x| go(ck, x, seen)),
                Ty::List(e) => go(ck, &e, seen),
                Ty::Param(_) => true,
                _ => false,
            }
        }
        go(self, t, &mut Vec::new())
    }

    /// A struct, tuple or type parameter made of differentiable values: the tangent space, where
    /// `p - 0.01 * g` works field by field.
    pub(super) fn tangent(&self, t: &Ty) -> bool {
        matches!(self.resolve(t), Ty::Struct(_) | Ty::Tuple(_) | Ty::Param(_)) && self.differentiable(t)
    }

    /// `+ - * /` on the tangent space: two values of the same type, or one with an `f64`.
    pub(super) fn tangent_binary(&mut self, op: BinOp, tl: &Ty, tr: &Ty, sp: Span) -> KResult<Option<Ty>> {
        let (a, b) = (self.tangent(tl), self.tangent(tr));
        if !(a || b) || !matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div) {
            return Ok(None);
        }
        let sym = op_str(op);
        let bad = |me: &Self| {
            Diag::new("E0101", sp, format!("cannot apply `{sym}` to `{}` and `{}`", me.show(tl), me.show(tr)))
                .hint("field-by-field arithmetic takes two values of the same type, or one of them and an `f64`")
        };
        let (s, other) = if a { (tl, tr) } else { (tr, tl) };
        if a && b {
            if self.unify(tl, tr).is_err() {
                return Err(bad(self));
            }
        } else if self.unify(other, &Ty::Float).is_err() {
            return Err(bad(self));
        }
        Ok(Some(self.resolve(s)))
    }

    fn int_const(e: &Expr, what: &str) -> KResult<i64> {
        match &e.kind {
            ExprKind::Int(v) => Ok(*v),
            ExprKind::Unary(UnOp::Neg, x) => match x.kind {
                ExprKind::Int(v) => Ok(-v),
                _ => Err(Diag::new("E0101", e.span, format!("`{what}` must be an integer constant"))),
            },
            _ => Err(Diag::new("E0101", e.span, format!("`{what}` must be an integer constant"))),
        }
    }

    /// `grad(f)` / `grad(f, wrt: k)` / `value_and_grad(f)`: f returns f64 (a loss); the result is
    /// a function with the same parameters returning the gradient (and the value) with respect
    /// to parameter k (from 1).
    fn grad_ty(&mut self, b: Builtin, all: &[(&Expr, bool)], named: &[(&str, Span, &Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        let name = builtin_name(b);
        if all.len() != 1 {
            return Err(Diag::new("E0103", sp, format!("`{name}` takes the function to differentiate")).hint(format!("e.g. `{name}(loss)` or `{name}(loss, wrt: 2)`")));
        }
        let mut wrt = 1i64;
        for (nm, nsp, e) in named {
            if *nm != "wrt" {
                return Err(Diag::new("E0120", *nsp, format!("`{name}` has no argument `{nm}`")).hint("choose the parameter with `wrt: k` (from 1)"));
            }
            wrt = Self::int_const(e, "wrt:")?;
        }
        let (hf, tf) = self.expr(all[0].0, None)?;
        let (ps, r) = match self.resolve(&tf) {
            Ty::Fn(ps, r) => (ps, *r),
            other => {
                return Err(Diag::new("E0101", all[0].0.span, format!("`{name}` needs a function, found `{}`", self.show(&other)))
                    .hint("pass a named function, or a lambda with typed parameters: `|x: f64| x * x`"));
            }
        };
        if self.unify(&r, &Ty::Float).is_err() {
            return Err(Diag::new("E0401", all[0].0.span, format!("`{name}` needs a function that returns `f64` (a loss), found one returning `{}`", self.show(&r))));
        }
        let Some(k) = usize::try_from(wrt - 1).ok().filter(|k| *k < ps.len()) else {
            return Err(Diag::new("E0103", sp, format!("`wrt: {wrt}`, but the function has {} parameter(s)", ps.len())));
        };
        let p = ps[k].clone();
        if !self.differentiable(&p) {
            return Err(Diag::new("E0402", all[0].0.span, format!("cannot differentiate with respect to a `{}`", self.show(&p)))
                .hint("differentiable types: f64, f64 tensors, and structs, tuples and lists of them"));
        }
        let ret = if b == Builtin::ValueAndGrad { Ty::Tuple(vec![Ty::Float, p]) } else { p };
        let lit = |v: Value| HArg::Val(HExpr::Lit(v));
        let args = vec![HArg::Val(hf), lit(Value::Int(k as i64)), lit(Value::Bool(b == Builtin::ValueAndGrad))];
        Ok((HExpr::CallBuiltin(b, args, sp), Ty::Fn(ps, Box::new(ret))))
    }

    /// `randn(seed: k, month: 12, asset: 3)`: reproducible normal samples.
    fn randn_ty(&mut self, all: &[(&Expr, bool)], named: &[(&str, Span, &Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        if !all.is_empty() {
            return Err(Diag::new("E0103", sp, "`randn` takes only named arguments").hint("e.g. `randn(seed: 7, feature: 4, hidden: 16)`"));
        }
        let Some((_, _, seed_e)) = named.iter().find(|(n, _, _)| *n == "seed") else {
            return Err(Diag::new("E0103", sp, "`randn` needs `seed:` (results must be reproducible)").hint("e.g. `randn(seed: 7, feature: 4, hidden: 16)`"));
        };
        let mut hs = vec![HArg::Val(self.int_expr(seed_e)?)];
        let mut axes = Vec::new();
        for (nm, nsp, e) in named {
            if *nm == "seed" {
                continue;
            }
            let id = self.axis_id(nm, *nsp)?;
            if axes.contains(&id) {
                return Err(Diag::new("E0112", *nsp, format!("axis `{nm}` is given twice")));
            }
            hs.push(HArg::Val(HExpr::Lit(Value::Int(i64::from(id)))));
            hs.push(HArg::Val(self.int_expr(e)?));
            axes.push(id);
        }
        if axes.is_empty() {
            return Err(Diag::new("E0103", sp, "`randn` needs the size of each axis").hint("e.g. `randn(seed: 7, feature: 4, hidden: 16)`"));
        }
        Ok((HExpr::CallBuiltin(Builtin::Randn, hs, sp), Self::mk_tensor(axes, Ty::Float)))
    }

    /// `smooth_max(x, y, scale: s)`, `smooth_min`, `smooth_step(x, scale: s)`: the exact rule
    /// outside `grad`, a smooth relaxation (softplus, sigmoid) inside it.
    fn smooth_ty(&mut self, b: Builtin, all: &[(&Expr, bool)], named: &[(&str, Span, &Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        let name = builtin_name(b);
        let want = if b == Builtin::SmoothStep { 1 } else { 2 };
        if all.len() != want {
            return Err(Diag::new("E0103", sp, format!("`{name}` takes {want} value(s) and `scale:`")));
        }
        let Some((_, _, scale_e)) = named.iter().find(|(n, _, _)| *n == "scale") else {
            return Err(Diag::new("E0103", sp, format!("`{name}` needs `scale:`: how wide the smooth transition is")).hint(format!("e.g. `{name}(x{}, scale: 100.0)`", if want == 2 { ", 0.0" } else { "" })));
        };
        if let Some((nm, nsp, _)) = named.iter().find(|(n, _, _)| *n != "scale") {
            return Err(Diag::new("E0120", *nsp, format!("`{name}` has no argument `{nm}`")));
        }
        let float = Ty::Float;
        let (hs, t) = if b == Builtin::SmoothStep {
            let (h, t) = self.expr(all[0].0, Some(&float))?;
            match self.tensor_parts(&t) {
                Some((_, e)) => self.need_elem(name, &t, &e, &float, all[0].0)?,
                None => self.expect_ty(&t, &float, all[0].0.span)?,
            }
            (vec![HArg::Val(h)], t)
        } else {
            let (h, t) = self.max_min2(if b == Builtin::SmoothMax { Builtin::Max } else { Builtin::Min }, all, sp)?;
            let HExpr::CallBuiltin(_, hs, _) = h else { unreachable!("max/min call") };
            if self.resolve(&t) == Ty::Int {
                return Err(Diag::new("E0101", sp, format!("`{name}` needs `f64` values")));
            }
            (hs, t)
        };
        let (hsc, tsc) = self.expr(scale_e, Some(&float))?;
        self.expect_ty(&tsc, &float, scale_e.span)?;
        let mut args = hs;
        args.push(HArg::Val(hsc));
        Ok((HExpr::CallBuiltin(b, args, sp), t))
    }

    /// `observed(x)` / `observed(x, lag: k)`: declares that row `t` of `x` (over its temporal
    /// axis) becomes known at the end of `t + k`. Identity at run time; the temporal pass uses it.
    fn observed(&mut self, all: &[(&Expr, bool)], named: &[(&str, Span, &Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        if all.len() != 1 {
            return Err(Diag::new("E0103", sp, format!("`observed` takes 1 positional argument, but {} were given", all.len())));
        }
        let mut lag = 0i64;
        for (nm, nsp, e) in named {
            if *nm != "lag" {
                return Err(Diag::new("E0120", *nsp, format!("`observed` has no argument `{nm}`")).hint("the delay is given as `lag: k`"));
            }
            lag = match &e.kind {
                ExprKind::Int(v) => *v,
                ExprKind::Unary(UnOp::Neg, x) if matches!(x.kind, ExprKind::Int(_)) => {
                    let ExprKind::Int(v) = x.kind else { unreachable!() };
                    -v
                }
                _ => return Err(Diag::new("E0101", e.span, "`lag:` must be an integer constant").hint("e.g. `observed(x, lag: 1)`")),
            };
        }
        let (h, t) = self.expr(all[0].0, None)?;
        let (axes, _) = self.need_tensor("observed", &t, all[0].0)?;
        let temporal: Vec<u32> = axes.iter().copied().filter(|a| self.axes[*a as usize].temporal.is_some()).collect();
        let [axis] = temporal[..] else {
            return Err(Diag::new("E0201", all[0].0.span, format!("`observed` needs a tensor with exactly one temporal axis, found `{}`", self.show(&t)))
                .hint("temporal axes are `month`, `day` and those declared with `axis name: temporal(...)`"));
        };
        let lit = |v: i64| HArg::Val(HExpr::Lit(Value::Int(v)));
        Ok((HExpr::CallBuiltin(Builtin::Observed, vec![HArg::Val(h), lit(i64::from(axis)), lit(lag)], sp), t))
    }

    fn need_tensor(&self, name: &str, t: &Ty, e: &Expr) -> KResult<(Vec<u32>, Ty)> {
        self.tensor_parts(t)
            .ok_or_else(|| Diag::new("E0101", e.span, format!("`{name}` needs a tensor, found `{}`", self.show(t))))
    }

    fn need_elem(&self, name: &str, t: &Ty, elem: &Ty, want: &Ty, e: &Expr) -> KResult<()> {
        if elem == want {
            Ok(())
        } else {
            Err(Diag::new("E0101", e.span, format!("`{name}` needs a `{}` tensor, found `{}`", self.show(want), self.show(t))))
        }
    }

    fn has_axis(&self, axes: &[u32], a: u32, t: &Ty, asp: Span) -> KResult<()> {
        if axes.contains(&a) {
            Ok(())
        } else {
            Err(Diag::new("E0201", asp, format!("`{}` has no axis `{}`", self.show(t), self.axis_label(a))))
        }
    }

    /// The axis of a rank/top-k: `over:`, or the only axis of a one-axis tensor.
    fn over_axis(&self, over: Option<(u32, Span)>, axes: &[u32], t: &Ty, name: &str, sp: Span) -> KResult<u32> {
        match over {
            Some((a, asp)) => self.has_axis(axes, a, t, asp).map(|_| a),
            None if axes.len() == 1 => Ok(axes[0]),
            None => Err(Diag::new("E0201", sp, format!("`{name}` over which axis of `{}`?", self.show(t)))
                .hint(format!("name it with `over: {}`", self.axis_label(axes[0])))),
        }
    }

    /// `max(a, b)` / `min(a, b)`: numbers, or element-wise when a tensor is involved.
    fn max_min2(&mut self, b: Builtin, all: &[(&Expr, bool)], sp: Span) -> KResult<(HExpr, Ty)> {
        let name = builtin_name(b);
        let (ha, ta) = self.expr(all[0].0, None)?;
        let ea = self.elem_of(&ta);
        let (hb, tb) = self.expr(all[1].0, Some(&ea))?;
        let args = vec![HArg::Val(ha), HArg::Val(hb)];
        if self.tensor_parts(&ta).is_some() || self.tensor_parts(&tb).is_some() {
            let axes = self.frame_axes(&[(&ta, all[0].0), (&tb, all[1].0)], &format!("`{name}`"), sp)?;
            let eb = self.elem_of(&tb);
            if self.unify(&ea, &eb).is_err() || self.resolve(&ea) != Ty::Float {
                let (x, y) = (self.show(&ta), self.show(&tb));
                return Err(Diag::new("E0101", sp, format!("`{name}` needs `f64` values, found `{x}` and `{y}`")));
            }
            return Ok((HExpr::CallBuiltin(b, args, sp), Self::mk_tensor(axes, Ty::Float)));
        }
        self.expect_ty(&tb, &ta, all[1].0.span)?;
        self.ensure_numeric(&ta, sp, &format!("`{name}`"))?;
        Ok((HExpr::CallBuiltin(b, args, sp), ta))
    }

    /// Builtins that understand tensors. Axis arguments become `Lit(Int(axis id))`, with -1 for
    /// "all axes".
    pub(super) fn tensor_builtin(&mut self, b: Builtin, all: &[(&Expr, bool)], named: &[(&str, Span, &Expr)], sp: Span) -> KResult<(HExpr, Ty)> {
        use Builtin as B;
        match b {
            B::Observed => return self.observed(all, named, sp),
            B::Grad | B::ValueAndGrad => return self.grad_ty(b, all, named, sp),
            B::Randn => return self.randn_ty(all, named, sp),
            B::SmoothMax | B::SmoothMin | B::SmoothStep => return self.smooth_ty(b, all, named, sp),
            _ => {}
        }
        let name = builtin_name(b);
        let n = all.len();
        let arity = |want: usize| -> KResult<()> {
            if n != want {
                Err(Diag::new("E0103", sp, format!("`{name}` takes {want} positional argument(s), but {n} were given")))
            } else {
                Ok(())
            }
        };
        let takes_over = matches!(
            b,
            B::Sum | B::Mean | B::Prod | B::Max | B::Min | B::NanSum | B::NanMean | B::NanProd | B::NanCount | B::Count | B::Rank | B::TopK | B::BottomK | B::Contract
        );
        let takes_sizes = matches!(b, B::Zeros | B::Ones | B::Full);
        if !takes_sizes {
            for (nm, nsp, _) in named {
                if !takes_over {
                    return Err(Diag::new("E0120", *nsp, format!("`{name}` does not take named arguments")).hint(format!("pass `{nm}` by position")));
                }
                if *nm != "over" {
                    return Err(Diag::new("E0120", *nsp, format!("`{name}` has no argument `{nm}`")).hint("the axis is given as `over: axis`"));
                }
            }
        }
        let over: Option<(u32, Span)> = match named.first() {
            Some((_, _, e)) if takes_over => Some((self.axis_of_expr(e)?, e.span)),
            _ => None,
        };
        let ax_lit = |a: Option<u32>| HArg::Val(HExpr::Lit(Value::Int(a.map(i64::from).unwrap_or(-1))));
        let call = |args: Vec<HArg>, t: Ty| -> KResult<(HExpr, Ty)> { Ok((HExpr::CallBuiltin(b, args, sp), t)) };
        let float = Ty::Float;
        match b {
            B::Sum | B::Mean | B::Prod | B::Max | B::Min | B::NanSum | B::NanMean | B::NanProd | B::NanCount | B::Count => {
                if matches!(b, B::Max | B::Min) && n == 2 {
                    if let Some((_, osp)) = over {
                        return Err(Diag::new("E0103", osp, format!("`{name}` of two values takes no `over:`")));
                    }
                    return self.max_min2(b, all, sp);
                }
                arity(1)?;
                let e0 = all[0].0;
                let (h, t) = self.expr(e0, None)?;
                if let Some((axes, elem)) = self.tensor_parts(&t) {
                    match b {
                        B::Count => self.need_elem(name, &t, &elem, &Ty::Bool, e0)?,
                        B::Sum | B::Mean => {}
                        _ => self.need_elem(name, &t, &elem, &float, e0)?,
                    }
                    let out: Vec<u32> = match over {
                        Some((a, asp)) => {
                            self.has_axis(&axes, a, &t, asp)?;
                            axes.iter().copied().filter(|x| *x != a).collect()
                        }
                        None => Vec::new(),
                    };
                    return call(vec![HArg::Val(h), ax_lit(over.map(|o| o.0))], Self::mk_tensor(out, Ty::Float));
                }
                if let Some((_, osp)) = over {
                    return Err(Diag::new("E0201", osp, format!("`over:` needs a tensor, found `{}`", self.show(&t))));
                }
                match b {
                    B::Sum => {
                        let elem = self.list_elem(&t, e0.span)?;
                        self.ensure_numeric(&elem, e0.span, "`sum`")?;
                        let (zt, zid) = self.fresh(Kind::Num);
                        let _ = self.unify(&zt, &elem);
                        call(vec![HArg::Val(h), HArg::Val(HExpr::IntLit(0, zid))], elem)
                    }
                    B::Mean | B::Prod => {
                        let elem = self.list_elem(&t, e0.span)?;
                        self.expect_ty(&elem, &float, e0.span)?;
                        call(vec![HArg::Val(h)], Ty::Float)
                    }
                    B::Max | B::Min => {
                        let (big, list_fn) = if b == B::Max { ("largest", "maximum") } else { ("smallest", "minimum") };
                        Err(Diag::new("E0103", sp, format!("`{name}` takes two values or a tensor"))
                            .hint(format!("for the {big} element of a list use `{list_fn}(xs)`")))
                    }
                    _ => Err(Diag::new("E0101", e0.span, format!("`{name}` needs a tensor, found `{}`", self.show(&t)))),
                }
            }
            B::Abs | B::Sqrt | B::Exp | B::Ln | B::Floor | B::Ceil | B::Round | B::Tanh | B::Sigmoid | B::Relu | B::Softplus => {
                arity(1)?;
                let want = if b == B::Abs { None } else { Some(&float) };
                let (h, t) = self.expr(all[0].0, want)?;
                if let Some((_, elem)) = self.tensor_parts(&t) {
                    self.need_elem(name, &t, &elem, &float, all[0].0)?;
                    return call(vec![HArg::Val(h)], t);
                }
                // field by field on the tangent space (rounding is not differentiable)
                if self.tangent(&t) && !matches!(b, B::Floor | B::Ceil | B::Round) {
                    return call(vec![HArg::Val(h)], t);
                }
                if b == B::Abs {
                    self.ensure_numeric(&t, all[0].0.span, "`abs`")?;
                    return call(vec![HArg::Val(h)], t);
                }
                self.expect_ty(&t, &float, all[0].0.span)?;
                call(vec![HArg::Val(h)], Ty::Float)
            }
            B::IsNan => {
                arity(1)?;
                let (h, t) = self.expr(all[0].0, Some(&float))?;
                if let Some((axes, elem)) = self.tensor_parts(&t) {
                    self.need_elem(name, &t, &elem, &float, all[0].0)?;
                    return call(vec![HArg::Val(h)], Self::mk_tensor(axes, Ty::Bool));
                }
                self.expect_ty(&t, &float, all[0].0.span)?;
                call(vec![HArg::Val(h)], Ty::Bool)
            }
            B::FillNan => {
                arity(2)?;
                let (h, t) = self.expr(all[0].0, Some(&float))?;
                let (hv, tv) = self.expr(all[1].0, Some(&float))?;
                self.expect_ty(&tv, &float, all[1].0.span)?;
                if let Some((_, elem)) = self.tensor_parts(&t) {
                    self.need_elem(name, &t, &elem, &float, all[0].0)?;
                    return call(vec![HArg::Val(h), HArg::Val(hv)], t);
                }
                self.expect_ty(&t, &float, all[0].0.span)?;
                call(vec![HArg::Val(h), HArg::Val(hv)], Ty::Float)
            }
            B::Where => {
                arity(3)?;
                let (hc, tc) = self.expr(all[0].0, Some(&Ty::Bool))?;
                let (ha, ta) = self.expr(all[1].0, None)?;
                let ea = self.elem_of(&ta);
                let (hb, tb) = self.expr(all[2].0, Some(&ea))?;
                let ec = self.elem_of(&tc);
                self.expect_ty(&ec, &Ty::Bool, all[0].0.span)?;
                let eb = self.elem_of(&tb);
                if self.unify(&ea, &eb).is_err() {
                    let (x, y) = (self.show(&ta), self.show(&tb));
                    return Err(Diag::new("E0101", all[2].0.span, format!("`where` branches have different types: `{x}` and `{y}`")));
                }
                let ops = [(&tc, all[0].0), (&ta, all[1].0), (&tb, all[2].0)];
                let any_tensor = ops.iter().any(|(t, _)| self.tensor_parts(t).is_some());
                let args = vec![HArg::Val(hc), HArg::Val(ha), HArg::Val(hb)];
                if !any_tensor {
                    let t = self.resolve(&ea);
                    return call(args, t);
                }
                let axes = self.frame_axes(&ops, "`where`", sp)?;
                let elem = self.tensor_elem(&ea, all[1].0.span)?;
                call(args, Self::mk_tensor(axes, elem))
            }
            B::Size => {
                arity(2)?;
                let (h, t) = self.expr(all[0].0, None)?;
                let ax = self.axis_of_expr(all[1].0)?;
                let Some((axes, _)) = self.tensor_parts(&t) else {
                    return Err(Diag::new("E0101", all[0].0.span, format!("`size` needs a tensor, found `{}`", self.show(&t)))
                        .hint("the length of a list is `len(xs)`"));
                };
                self.has_axis(&axes, ax, &t, all[1].0.span)?;
                call(vec![HArg::Val(h), ax_lit(Some(ax))], Ty::Int)
            }
            B::Zeros | B::Ones | B::Full => {
                arity(if b == B::Full { 1 } else { 0 })?;
                if named.is_empty() {
                    let v = if b == B::Full { "0.0, " } else { "" };
                    return Err(Diag::new("E0103", sp, format!("`{name}` needs the size of each axis"))
                        .hint(format!("e.g. `{name}({v}month: 12, asset: 3)`")));
                }
                let mut hs = Vec::new();
                let mut elem = Ty::Float;
                if b == B::Full {
                    let (hv, tv) = self.expr(all[0].0, Some(&float))?;
                    elem = self.tensor_elem(&tv, all[0].0.span)?;
                    hs.push(HArg::Val(hv));
                }
                let mut axes = Vec::new();
                for (nm, nsp, e) in named {
                    let id = self.axis_id(nm, *nsp)?;
                    if axes.contains(&id) {
                        return Err(Diag::new("E0112", *nsp, format!("axis `{nm}` is given twice")));
                    }
                    let hn = self.int_expr(e)?;
                    hs.push(ax_lit(Some(id)));
                    hs.push(HArg::Val(hn));
                    axes.push(id);
                }
                call(hs, Self::mk_tensor(axes, elem))
            }
            B::ZerosLike | B::FullLike => {
                arity(if b == B::FullLike { 2 } else { 1 })?;
                let (h, t) = self.expr(all[0].0, None)?;
                // the zero of the tangent space (optimizer state, accumulated gradients)
                if b == B::ZerosLike && (self.tangent(&t) || self.resolve(&t) == Ty::Float) {
                    return call(vec![HArg::Val(h)], t);
                }
                let (axes, _) = self.need_tensor(name, &t, all[0].0)?;
                let mut hs = vec![HArg::Val(h)];
                let mut elem = Ty::Float;
                if b == B::FullLike {
                    let (hv, tv) = self.expr(all[1].0, Some(&float))?;
                    elem = self.tensor_elem(&tv, all[1].0.span)?;
                    hs.push(HArg::Val(hv));
                }
                call(hs, Self::mk_tensor(axes, elem))
            }
            B::Tensor => {
                arity(2)?;
                let (h, t) = self.expr(all[0].0, None)?;
                let ax = self.axis_of_expr(all[1].0)?;
                let e = self.list_elem(&t, all[0].0.span)?;
                let elem = self.tensor_elem(&e, all[0].0.span)?;
                let is_bool = HArg::Val(HExpr::Lit(Value::Bool(elem == Ty::Bool)));
                call(vec![HArg::Val(h), ax_lit(Some(ax)), is_bool], Ty::Tensor(vec![ax], Box::new(elem)))
            }
            B::ToList => {
                arity(1)?;
                let (h, t) = self.expr(all[0].0, None)?;
                let (axes, elem) = self.need_tensor(name, &t, all[0].0)?;
                if axes.len() != 1 {
                    return Err(Diag::new("E0201", all[0].0.span, format!("`to_list` needs a tensor with one axis, found `{}`", self.show(&t)))
                        .hint("select or reduce the other axes first"));
                }
                call(vec![HArg::Val(h)], Ty::List(Box::new(elem)))
            }
            B::Rank | B::TopK | B::BottomK => {
                arity(if b == B::Rank { 1 } else { 2 })?;
                let (h, t) = self.expr(all[0].0, None)?;
                let (axes, elem) = self.need_tensor(name, &t, all[0].0)?;
                self.need_elem(name, &t, &elem, &float, all[0].0)?;
                let ax = self.over_axis(over, &axes, &t, name, sp)?;
                let mut hs = vec![HArg::Val(h)];
                if b != B::Rank {
                    hs.push(HArg::Val(self.int_expr(all[1].0)?));
                }
                hs.push(ax_lit(Some(ax)));
                let out = if b == B::Rank { Ty::Float } else { Ty::Bool };
                call(hs, Self::mk_tensor(axes, out))
            }
            B::Contract => {
                arity(2)?;
                let (hx, t_x) = self.expr(all[0].0, None)?;
                let (hy, t_y) = self.expr(all[1].0, None)?;
                let (ax_x, ex) = self.need_tensor(name, &t_x, all[0].0)?;
                let (ax_y, ey) = self.need_tensor(name, &t_y, all[1].0)?;
                self.need_elem(name, &t_x, &ex, &float, all[0].0)?;
                self.need_elem(name, &t_y, &ey, &float, all[1].0)?;
                let Some((a, asp)) = over else {
                    return Err(Diag::new("E0103", sp, "`contract` needs the axis to sum over").hint("e.g. `contract(x, w, over: feature)`"));
                };
                self.has_axis(&ax_x, a, &t_x, asp)?;
                self.has_axis(&ax_y, a, &t_y, asp)?;
                let mut out: Vec<u32> = ax_x.iter().chain(ax_y.iter()).copied().filter(|x| *x != a).collect();
                out.sort_unstable();
                out.dedup();
                call(vec![HArg::Val(hx), HArg::Val(hy), ax_lit(Some(a))], Self::mk_tensor(out, Ty::Float))
            }
            _ => unreachable!("not a tensor-aware builtin: {name}"),
        }
    }

    /// `b3.f(...)`: the bridge to the quant-b3 lab (data and the realistic simulator).
    pub(super) fn b3_call(&mut self, method: &str, args: &[Arg], sp: Span) -> KResult<(HExpr, Ty)> {
        let what = format!("`b3.{method}`");
        if method != "evaluate" {
            Self::no_named_args(args, &what)?;
        }
        if let Some(a) = args.iter().find(|a| a.inout) {
            return Err(Diag::new("E0115", a.expr.span, format!("{what} takes no `inout` arguments")));
        }
        let positional = args.iter().filter(|a| a.name.is_none()).count();
        let nargs = |want: usize| -> KResult<()> {
            if positional != want {
                Err(Diag::new("E0103", sp, format!("{what} takes {want} argument(s), but {positional} were given")))
            } else {
                Ok(())
            }
        };
        let host = |f: HostFn, xs: Vec<HExpr>, t: Ty| -> KResult<(HExpr, Ty)> { Ok((HExpr::Host(f, xs, sp), t)) };
        let month_asset = vec![AX_MONTH, AX_ASSET];
        match method {
            "monthly_returns" => {
                nargs(0)?;
                host(HostFn::MonthlyReturns, vec![], Ty::Tensor(month_asset, Box::new(Ty::Float)))
            }
            "eligible" => {
                nargs(0)?;
                host(HostFn::Eligible, vec![], Ty::Tensor(month_asset, Box::new(Ty::Bool)))
            }
            "cdi" => {
                nargs(0)?;
                host(HostFn::Cdi, vec![], Ty::Tensor(vec![AX_MONTH], Box::new(Ty::Float)))
            }
            "months" => {
                nargs(0)?;
                host(HostFn::Months, vec![], Ty::List(Box::new(Ty::Int)))
            }
            "tickers" => {
                nargs(0)?;
                host(HostFn::Tickers, vec![], Ty::List(Box::new(Ty::Str)))
            }
            "month_index" => {
                nargs(1)?;
                let h = self.int_expr(&args[0].expr)?;
                host(HostFn::MonthIndex, vec![h], Ty::Int)
            }
            "evaluate" => {
                nargs(1)?;
                if args[0].name.is_some() {
                    return Err(Diag::new("E0120", args[0].expr.span, "positional arguments must come before named ones"));
                }
                let Some(&sid) = self.struct_ids.get("Evaluation") else {
                    return Err(Diag::new("E0100", sp, "the prelude's `Evaluation` struct is missing"));
                };
                let want = Ty::Tensor(month_asset, Box::new(Ty::Float));
                let (hw, tw) = self.expr(&args[0].expr, Some(&want))?;
                if self.unify(&tw, &want).is_err() {
                    let code = if self.tensor_parts(&tw).is_some() { "E0201" } else { "E0101" };
                    return Err(Diag::new(code, args[0].expr.span, format!("`b3.evaluate` needs weights of type `{}`, found `{}`", self.show(&want), self.show(&tw)))
                        .hint("one weight per month and asset; a month whose weights are all zero stays in cash (CDI)"));
                }
                let mut opts: Vec<(&str, HExpr, Ty)> = vec![
                    ("cost_bps", HExpr::Lit(Value::Float(25.0)), Ty::Float),
                    ("lag", HExpr::Lit(Value::Int(1)), Ty::Int),
                    ("capital", HExpr::Lit(Value::Float(1_000_000.0)), Ty::Float),
                    ("real_costs", HExpr::Lit(Value::Bool(false)), Ty::Bool),
                    ("tax", HExpr::Lit(Value::Bool(false)), Ty::Bool),
                ];
                let mut seen: Vec<&str> = Vec::new();
                for a in &args[1..] {
                    let Some((nm, nsp)) = &a.name else {
                        return Err(Diag::new("E0120", a.expr.span, "positional arguments must come before named ones")
                            .hint("e.g. `b3.evaluate(w, cost_bps: 25.0, lag: 1)`"));
                    };
                    let Some(k) = opts.iter().position(|(o, _, _)| *o == nm.as_str()) else {
                        return Err(Diag::new("E0120", *nsp, format!("`b3.evaluate` has no option `{nm}`"))
                            .hint("its options are cost_bps, lag, capital, real_costs and tax"));
                    };
                    if seen.contains(&nm.as_str()) {
                        return Err(Diag::new("E0112", *nsp, format!("option `{nm}` is given twice")));
                    }
                    seen.push(nm.as_str());
                    let t = opts[k].2.clone();
                    let (h, ht) = self.expr(&a.expr, Some(&t))?;
                    self.expect_ty(&ht, &t, a.expr.span)?;
                    opts[k].1 = h;
                }
                let mut xs = vec![hw];
                xs.extend(opts.into_iter().map(|(_, h, _)| h));
                host(HostFn::Evaluate { result: sid }, xs, Ty::Struct(sid))
            }
            _ => Err(Diag::new("E0100", sp, format!("the `b3` module has no function `{method}`"))
                .hint("available: monthly_returns, eligible, cdi, months, tickers, month_index, evaluate")),
        }
    }
}
