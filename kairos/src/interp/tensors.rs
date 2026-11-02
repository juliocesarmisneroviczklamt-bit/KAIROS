//! Runtime of tensor operators, tensor builtins and the `b3` module (M2).

use super::*;
use crate::bridge::{EvalCfg, EvalOut, HostErr};
use crate::tensor::{self, Elem, Op, Red, Sel, TData, TErr, TensorVal, map_n};

/// Largest tensor `zeros`/`ones`/`full` may build (2^28 elements = 2 GB of f64).
const MAX_ELEMS: usize = 1 << 28;

/// A tensor with no axes left is a scalar.
pub(super) fn wrap(t: TensorVal) -> Value {
    if t.axes.is_empty() {
        match t.data {
            TData::F(v) => Value::Float(v[0]),
            TData::B(v) => Value::Bool(v[0]),
        }
    } else {
        Value::Tensor(Rc::new(t))
    }
}

pub(super) fn as_op(v: &Value) -> Op<'_> {
    match v {
        Value::Tensor(t) => Op::T(t),
        Value::Float(x) => Op::F(*x),
        Value::Bool(b) => Op::B(*b),
        other => unreachable!("tensor operand {other:?}"),
    }
}

fn is_tensor(v: &Value) -> bool {
    matches!(v, Value::Tensor(_))
}

/// Axis argument encoded by the checker: an axis id, or -1 for "all axes".
fn axis_arg(v: &Value) -> Option<u32> {
    match v {
        Value::Int(a) if *a >= 0 => Some(*a as u32),
        _ => None,
    }
}

fn tensor_of(v: &Value) -> &TensorVal {
    match v {
        Value::Tensor(t) => t,
        other => unreachable!("expected a tensor, got {other:?}"),
    }
}

fn float_of(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        other => unreachable!("expected f64, got {other:?}"),
    }
}

fn reduction(b: Builtin) -> Red {
    match b {
        Builtin::Sum => Red::Sum,
        Builtin::Mean => Red::Mean,
        Builtin::Prod => Red::Prod,
        Builtin::Max => Red::Max,
        Builtin::Min => Red::Min,
        Builtin::NanSum => Red::NanSum,
        Builtin::NanMean => Red::NanMean,
        Builtin::NanProd => Red::NanProd,
        Builtin::NanCount => Red::NanCount,
        Builtin::Count => Red::Count,
        other => unreachable!("not a reduction: {other:?}"),
    }
}

impl Interp<'_> {
    fn axis_name(&self, a: u32) -> &str {
        &self.ck.axes[a as usize].name
    }

    pub(super) fn terr(&self, e: TErr, sp: Span) -> Ctl {
        Ctl::Err(match e {
            TErr::SizeMismatch { axis, a, b } => {
                Diag::new("R0201", sp, format!("size mismatch on axis `{}`: {a} vs {b}", self.axis_name(axis)))
            }
            TErr::OutOfBounds { axis, index, size } => Diag::new(
                "R0001",
                sp,
                format!("index {index} out of bounds for axis `{}` of size {size}", self.axis_name(axis)),
            ),
            TErr::Reversed { axis, lo, hi } => {
                Diag::new("R0001", sp, format!("reversed range {lo}..{hi} on axis `{}`", self.axis_name(axis)))
            }
            TErr::Shape(m) => Diag::new("R0202", sp, m),
        })
    }

    pub(super) fn eval_sels(&mut self, sels: &[(u32, HSel)], f: &mut Vec<Value>, sp: Span) -> R<Vec<(u32, Sel)>> {
        let mut out = Vec::with_capacity(sels.len());
        for (a, s) in sels {
            let sel = match s {
                HSel::At(i) => Sel::At(as_int(self.eval(i, f)?)),
                HSel::Range(lo, hi, inclusive) => {
                    let lo = as_int(self.eval(lo, f)?);
                    let hi = as_int(self.eval(hi, f)?);
                    let hi = if *inclusive { hi.checked_add(1).ok_or_else(|| overflow(sp))? } else { hi };
                    Sel::Range(lo, hi)
                }
            };
            out.push((*a, sel));
        }
        Ok(out)
    }

    /// `x[month: i, asset: a .. b]` (kept out of `eval` so its stack frame stays small).
    #[inline(never)]
    pub(super) fn tensor_index(&mut self, x: &HExpr, sels: &[(u32, HSel)], f: &mut Vec<Value>, sp: Span) -> R<Value> {
        let v = self.eval(x, f)?;
        let sels = self.eval_sels(sels, f, sp)?;
        if matches!(v, Value::Node(_)) {
            return self.select_diff(&v, &sels, sp);
        }
        let t = tensor_of(&v);
        let r = tensor::select(t, &sels).map_err(|e| self.terr(e, sp))?;
        Ok(wrap(r))
    }

    /// `x[month: i] = v` and `x[month: i] op= v`; the tensor is copied only if it is shared.
    #[inline(never)]
    pub(super) fn tensor_assign(
        &mut self,
        p: &HPlace,
        sels: &[(u32, HSel)],
        op: Option<BinOp>,
        v: &HExpr,
        f: &mut Vec<Value>,
        sp: Span,
    ) -> R<Value> {
        let rhs = self.eval(v, f)?;
        let sels = self.eval_sels(sels, f, sp)?;
        let (root, path) = self.resolve_place(p, f)?;
        let slot = place_mut(&mut f[root], &path)?;
        if matches!(slot, Value::Node(_)) || matches!(rhs, Value::Node(_)) {
            // inside `grad`: the assignment becomes a node of the tape
            let old = slot.clone();
            let rhs = match op {
                None => rhs,
                Some(bop) => {
                    let cur = if matches!(old, Value::Node(_)) {
                        self.select_diff(&old, &sels, sp)?
                    } else {
                        wrap(tensor::select(tensor_of(&old), &sels).map_err(|e| self.terr(e, sp))?)
                    };
                    self.bin(bop, cur, rhs, sp)?
                }
            };
            let new = self.assign_diff(&old, &sels, &rhs, sp)?;
            *place_mut(&mut f[root], &path)? = new;
            return Ok(Value::Unit);
        }
        let Value::Tensor(t) = slot else { unreachable!("tensor assignment to a non-tensor") };
        let rhs = match op {
            None => rhs,
            Some(bop) => {
                let cur = tensor::select(t, &sels).map_err(|e| self.terr(e, sp))?;
                self.bin(bop, wrap(cur), rhs, sp)?
            }
        };
        tensor::assign(Rc::make_mut(t), &sels, as_op(&rhs)).map_err(|e| self.terr(e, sp))?;
        Ok(Value::Unit)
    }

    /// Binary operator: scalars as in M1, element-wise (aligned by axis name) with tensors.
    pub(super) fn bin(&self, op: BinOp, a: Value, b: Value, sp: Span) -> R<Value> {
        let plain = |v: &Value| matches!(v, Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Str(_) | Value::Tensor(_) | Value::Unit);
        if !plain(&a) || !plain(&b) {
            // tracked values (grad) and structs/tuples of floats (the tangent space)
            if let Some(r) = self.bin_diff(op, a.clone(), b.clone(), sp)? {
                return Ok(r);
            }
        }
        if !is_tensor(&a) && !is_tensor(&b) {
            return binop(op, a, b, sp);
        }
        let cmp = matches!(op, BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge);
        let r = map_n(&[as_op(&a), as_op(&b)], cmp, |e| match (e[0], e[1]) {
            (Elem::F(x), Elem::F(y)) => match op {
                BinOp::Add => Elem::F(x + y),
                BinOp::Sub => Elem::F(x - y),
                BinOp::Mul => Elem::F(x * y),
                BinOp::Div => Elem::F(x / y),
                BinOp::Rem => Elem::F(x % y),
                BinOp::Pow => Elem::F(libm::pow(x, y)),
                BinOp::Eq => Elem::B(x == y),
                BinOp::Ne => Elem::B(x != y),
                BinOp::Lt => Elem::B(x < y),
                BinOp::Le => Elem::B(x <= y),
                BinOp::Gt => Elem::B(x > y),
                BinOp::Ge => Elem::B(x >= y),
            },
            (Elem::B(x), Elem::B(y)) => Elem::B(if op == BinOp::Eq { x == y } else { x != y }),
            _ => unreachable!("mixed tensor elements"),
        });
        r.map(wrap).map_err(|e| self.terr(e, sp))
    }

    pub(super) fn tensor_unary(&self, op: UnOp, t: &TensorVal, sp: Span) -> R<Value> {
        map_n(&[Op::T(t)], op == UnOp::Not, |e| match e[0] {
            Elem::F(x) => Elem::F(-x),
            Elem::B(b) => Elem::B(!b),
        })
        .map(wrap)
        .map_err(|e| self.terr(e, sp))
    }

    fn map_f(&self, t: &TensorVal, sp: Span, f: impl Fn(f64) -> f64) -> R<Value> {
        map_n(&[Op::T(t)], false, |e| match e[0] {
            Elem::F(x) => Elem::F(f(x)),
            Elem::B(_) => unreachable!("f64 map on a bool tensor"),
        })
        .map(wrap)
        .map_err(|e| self.terr(e, sp))
    }

    fn filled(&self, axes: Vec<u32>, shape: Vec<usize>, fill: &Value, sp: Span) -> R<Value> {
        let mut n: usize = 1;
        for (a, s) in axes.iter().zip(&shape) {
            n = n
                .checked_mul(*s)
                .filter(|n| *n <= MAX_ELEMS)
                .ok_or_else(|| Ctl::Err(Diag::new("R0203", sp, format!("tensor too large (axis `{}` of size {s})", self.axis_name(*a)))))?;
        }
        let t = match fill {
            Value::Bool(x) => TensorVal::new_b(axes, shape, vec![*x; n]),
            other => TensorVal::new_f(axes, shape, vec![float_of(other); n]),
        };
        Ok(Value::Tensor(Rc::new(t)))
    }

    /// Builtins that involve tensors (plus `mean`/`prod` of lists). `None`: an M1 case.
    pub(super) fn tensor_builtin(&mut self, b: Builtin, v: &[Value], sp: Span) -> R<Option<Value>> {
        use Builtin as B;
        let any_t = v.iter().any(is_tensor);
        let te = |me: &Self, e: TErr| me.terr(e, sp);
        let out = match b {
            // reductions: [tensor, axis or -1]
            B::Sum | B::Mean | B::Prod | B::Max | B::Min | B::NanSum | B::NanMean | B::NanProd | B::NanCount | B::Count
                if is_tensor(&v[0]) && matches!(v.get(1), Some(Value::Int(_))) =>
            {
                wrap(tensor::reduce(tensor_of(&v[0]), axis_arg(&v[1]), reduction(b)).map_err(|e| te(self, e))?)
            }
            // element-wise max / min of two operands (NaN propagates)
            B::Max | B::Min if any_t => {
                let is_max = b == B::Max;
                let r = map_n(&[as_op(&v[0]), as_op(&v[1])], false, |e| match (e[0], e[1]) {
                    (Elem::F(x), Elem::F(y)) => Elem::F(if x.is_nan() || y.is_nan() {
                        f64::NAN
                    } else if is_max {
                        x.max(y)
                    } else {
                        x.min(y)
                    }),
                    _ => unreachable!("max/min of bool tensors"),
                });
                wrap(r.map_err(|e| te(self, e))?)
            }
            B::Mean | B::Prod => {
                let Value::List(xs) = &v[0] else { unreachable!("mean/prod of {:?}", v[0]) };
                if b == B::Mean {
                    if xs.is_empty() {
                        return Err(Ctl::Err(Diag::new("R0004", sp, "`mean` of an empty list")));
                    }
                    let mut acc = 0.0;
                    for x in xs.iter() {
                        acc += float_of(x);
                    }
                    Value::Float(acc / xs.len() as f64)
                } else {
                    let mut acc = 1.0;
                    for x in xs.iter() {
                        acc *= float_of(x);
                    }
                    Value::Float(acc)
                }
            }
            B::Abs | B::Sqrt | B::Exp | B::Ln | B::Floor | B::Ceil | B::Round if any_t => {
                let f: fn(f64) -> f64 = match b {
                    B::Abs => f64::abs,
                    B::Sqrt => f64::sqrt,
                    B::Exp => libm::exp,
                    B::Ln => libm::log,
                    B::Floor => f64::floor,
                    B::Ceil => f64::ceil,
                    _ => f64::round,
                };
                self.map_f(tensor_of(&v[0]), sp, f)?
            }
            B::IsNan => match &v[0] {
                Value::Tensor(t) => {
                    let r = map_n(&[Op::T(t)], true, |e| match e[0] {
                        Elem::F(x) => Elem::B(x.is_nan()),
                        Elem::B(_) => unreachable!("is_nan of a bool tensor"),
                    });
                    wrap(r.map_err(|e| te(self, e))?)
                }
                x => Value::Bool(float_of(x).is_nan()),
            },
            B::FillNan => {
                let fill = float_of(&v[1]);
                match &v[0] {
                    Value::Tensor(t) => self.map_f(t, sp, |x| if x.is_nan() { fill } else { x })?,
                    x => {
                        let x = float_of(x);
                        Value::Float(if x.is_nan() { fill } else { x })
                    }
                }
            }
            B::Where => {
                if !any_t {
                    if matches!(v[0], Value::Bool(true)) { v[1].clone() } else { v[2].clone() }
                } else {
                    let out_bool = match &v[1] {
                        Value::Bool(_) => true,
                        Value::Tensor(t) => matches!(t.data, TData::B(_)),
                        _ => false,
                    };
                    let r = map_n(&[as_op(&v[0]), as_op(&v[1]), as_op(&v[2])], out_bool, |e| match e[0] {
                        Elem::B(c) => {
                            if c {
                                e[1]
                            } else {
                                e[2]
                            }
                        }
                        Elem::F(_) => unreachable!("`where` with a f64 condition"),
                    });
                    wrap(r.map_err(|e| te(self, e))?)
                }
            }
            B::Size => {
                let t = tensor_of(&v[0]);
                let a = axis_arg(&v[1]).expect("axis id");
                Value::Int(t.size(a).expect("axis checked by the type checker") as i64)
            }
            B::Zeros | B::Ones | B::Full => {
                let (fill, rest) = match b {
                    B::Zeros => (Value::Float(0.0), v),
                    B::Ones => (Value::Float(1.0), v),
                    _ => (v[0].clone(), &v[1..]),
                };
                let mut axes = Vec::new();
                let mut shape = Vec::new();
                for pair in rest.chunks(2) {
                    let a = axis_arg(&pair[0]).expect("axis id");
                    let n = as_int(pair[1].clone());
                    if n < 0 {
                        return Err(Ctl::Err(Diag::new("R0203", sp, format!("negative size {n} for axis `{}`", self.axis_name(a)))));
                    }
                    let Ok(n) = usize::try_from(n) else {
                        return Err(Ctl::Err(Diag::new("R0203", sp, format!("tensor too large (axis `{}` of size {n})", self.axis_name(a)))));
                    };
                    axes.push(a);
                    shape.push(n);
                }
                self.filled(axes, shape, &fill, sp)?
            }
            B::ZerosLike | B::FullLike => {
                let t = tensor_of(&v[0]);
                let fill = if b == B::ZerosLike { Value::Float(0.0) } else { v[1].clone() };
                self.filled(t.axes.clone(), t.shape.clone(), &fill, sp)?
            }
            B::Tensor => {
                let Value::List(xs) = &v[0] else { unreachable!("tensor of {:?}", v[0]) };
                let a = axis_arg(&v[1]).expect("axis id");
                let t = if matches!(v[2], Value::Bool(true)) {
                    TensorVal::new_b(vec![a], vec![xs.len()], xs.iter().map(|x| matches!(x, Value::Bool(true))).collect())
                } else {
                    TensorVal::new_f(vec![a], vec![xs.len()], xs.iter().map(float_of).collect())
                };
                Value::Tensor(Rc::new(t))
            }
            B::ToList => {
                let xs: Vec<Value> = match &tensor_of(&v[0]).data {
                    TData::F(d) => d.iter().map(|x| Value::Float(*x)).collect(),
                    TData::B(d) => d.iter().map(|x| Value::Bool(*x)).collect(),
                };
                Value::List(Rc::new(xs))
            }
            B::Rank => {
                let a = axis_arg(&v[1]).expect("axis id");
                wrap(tensor::rank(tensor_of(&v[0]), a).map_err(|e| te(self, e))?)
            }
            B::TopK | B::BottomK => {
                let k = as_int(v[1].clone());
                let a = axis_arg(&v[2]).expect("axis id");
                wrap(tensor::top_k(tensor_of(&v[0]), a, k, b == B::TopK).map_err(|e| te(self, e))?)
            }
            B::Contract => {
                let a = axis_arg(&v[2]).expect("axis id");
                wrap(tensor::contract(tensor_of(&v[0]), tensor_of(&v[1]), a).map_err(|e| te(self, e))?)
            }
            // only the temporal check cares: at run time the data is the data
            B::Observed => v[0].clone(),
            B::TAnd | B::TOr => {
                let and = b == B::TAnd;
                let r = map_n(&[as_op(&v[0]), as_op(&v[1])], true, |e| match (e[0], e[1]) {
                    (Elem::B(x), Elem::B(y)) => Elem::B(if and { x && y } else { x || y }),
                    _ => unreachable!("`and`/`or` of f64 values"),
                });
                wrap(r.map_err(|e| te(self, e))?)
            }
            _ => return Ok(None),
        };
        Ok(Some(out))
    }

    /// `b3.f(...)`: data and the realistic simulator of the quant-b3 lab.
    pub(super) fn host_call(&mut self, hf: HostFn, v: Vec<Value>, sp: Span) -> R<Value> {
        // the simulator has no gradient (yet): it sees plain values
        let v: Vec<Value> = v.iter().map(|x| self.deep_fwd(x)).collect();
        let host = self.host;
        let ck = self.ck;
        // the program's protocols fence the data it sees (M5)
        let protos = &ck.protocols;
        let herr = |e: HostErr| {
            let mut d = Diag::new(e.code, sp, e.msg);
            if let Some(h) = e.hint {
                d = d.hint(h);
            }
            Ctl::Err(d)
        };
        let cfg_of = |o: &[Value]| EvalCfg {
            cost_bps: float_of(&o[0]),
            lag: as_int(o[1].clone()),
            capital: float_of(&o[2]),
            real_costs: as_bool(o[3].clone()),
            tax: as_bool(o[4].clone()),
        };
        Ok(match hf {
            HostFn::MonthlyReturns => Value::Tensor(host.monthly_returns(protos).map_err(herr)?),
            HostFn::Eligible => Value::Tensor(host.eligible(protos).map_err(herr)?),
            HostFn::Cdi => Value::Tensor(host.cdi(protos).map_err(herr)?),
            HostFn::Months => Value::List(Rc::new(host.months(protos).map_err(herr)?.into_iter().map(Value::Int).collect())),
            HostFn::Tickers => {
                Value::List(Rc::new(host.tickers().map_err(herr)?.into_iter().map(|s| Value::Str(s.into())).collect()))
            }
            HostFn::MonthIndex => Value::Int(host.month_index(as_int(v[0].clone()), protos).map_err(herr)?),
            HostFn::Evaluate { result } => {
                let ev = host.evaluate(tensor_of(&v[0]), &cfg_of(&v[1..]), protos).map_err(herr)?;
                self.evaluation(result, &ev, sp)?
            }
            HostFn::Daily { field } => Value::Tensor(host.daily(field, protos).map_err(herr)?),
            HostFn::Sessions => Value::Tensor(host.sessions(protos).map_err(herr)?),
            HostFn::Issuers | HostFn::Classes => {
                let names = host.asset_names(matches!(hf, HostFn::Issuers)).map_err(herr)?;
                Value::List(Rc::new(names.into_iter().map(|s| Value::Str(s.into())).collect()))
            }
            HostFn::Nefin { factor } => Value::Tensor(host.nefin(factor, as_bool(v[0].clone()), protos).map_err(herr)?),
            HostFn::Holdout { proto } => {
                host.holdout(protos, proto as usize).map_err(herr)?;
                Value::Holdout(proto)
            }
            HostFn::OpenOnce { result } => {
                let Value::Holdout(k) = v[0] else { unreachable!("the checker gives `open_once` a holdout") };
                let cfg = cfg_of(&v[2..]);
                host.begin_opening(protos, k as usize).map_err(herr)?;
                // the holdout's months exist only while the weights function runs
                match self.call_value(&v[1], vec![], sp) {
                    Ok(w) => {
                        let w = self.deep_fwd(&w);
                        let ev = host.finish_opening(protos, Ok(tensor_of(&w)), &cfg).map_err(herr)?;
                        self.evaluation(result, &ev, sp)?
                    }
                    Err(Ctl::Err(d)) => {
                        let spent = format!("the weights function failed while the holdout was open: {} {}", d.code, d.msg);
                        // the opening is recorded as failed; only a ledger that cannot be written
                        // takes precedence over the program's own error
                        if let Err(e) = host.finish_opening(protos, Err(spent), &cfg) {
                            if e.code == "R0500" {
                                return Err(herr(e));
                            }
                        }
                        let name = &protos[k as usize].name;
                        let spent = format!("the holdout of protocol \"{name}\" was open, so it is spent: the failure is recorded in the ledger");
                        let hint = match &d.hint {
                            Some(h) => format!("{h}\n{spent}"),
                            None => spent,
                        };
                        return Err(Ctl::Err(d.hint(hint)));
                    }
                    Err(other) => return Err(other),
                }
            }
        })
    }

    /// The prelude's `Evaluation`, field by field.
    fn evaluation(&self, result: u32, ev: &EvalOut, sp: Span) -> R<Value> {
        let def = &self.ck.structs[result as usize];
        let mut fields = Vec::with_capacity(def.fields.len());
        for (name, _) in &def.fields {
            fields.push(match name.as_str() {
                "months" => Value::Int(ev.months),
                "cagr" => Value::Float(ev.cagr),
                "vol" => Value::Float(ev.vol),
                "sharpe" => Value::Float(ev.sharpe),
                "max_dd" => Value::Float(ev.max_dd),
                "beta" => Value::Float(ev.beta),
                "alpha" => Value::Float(ev.alpha),
                "t_alpha" => Value::Float(ev.t_alpha),
                "turnover" => Value::Float(ev.turnover),
                "cost_bps" => Value::Float(ev.cost_bps),
                "trials" => Value::Int(ev.trials),
                "dsr" => Value::Float(ev.dsr),
                other => {
                    return Err(Ctl::Err(Diag::new("R0199", sp, format!("internal: `Evaluation.{other}` is not produced by `b3.evaluate`"))));
                }
            });
        }
        Ok(Value::Struct(result, Rc::new(fields)))
    }
}
