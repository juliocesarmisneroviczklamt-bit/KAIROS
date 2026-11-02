//! Runtime of M4: `grad` (reverse mode, on a tape), the tangent space (arithmetic on structs and
//! tuples of floats and tensors, field by field) and the AI builtins (activations, `randn`, the
//! smooth relaxations that are exact outside `grad` and smooth inside it).

use super::*;
use crate::autodiff::{GOp, MapFn, Tape, sigmoid, softplus};
use crate::tensor::{self, Red, TData, TensorVal};

fn is_node(v: &Value) -> bool {
    matches!(v, Value::Node(_))
}

/// Does `v` contain a tracked value (also inside structs, tuples and lists)?
pub(super) fn has_node(v: &Value) -> bool {
    match v {
        Value::Node(_) => true,
        Value::Struct(_, xs) | Value::Tuple(xs) | Value::List(xs) | Value::Enum(_, _, xs) => xs.iter().any(has_node),
        _ => false,
    }
}

fn float(v: &Value) -> f64 {
    match v {
        Value::Float(x) => *x,
        other => unreachable!("expected f64, got {other:?}"),
    }
}

impl Interp<'_> {
    pub(super) fn tracking(&self) -> bool {
        self.tape.borrow().is_some()
    }

    /// The plain value behind a tracked one.
    pub(super) fn fwd(&self, v: &Value) -> Value {
        match v {
            Value::Node(i) => self.tape.borrow().as_ref().expect("an active tape").val(*i).clone(),
            other => other.clone(),
        }
    }

    /// `fwd` all the way down structs, tuples and lists.
    pub(super) fn deep_fwd(&self, v: &Value) -> Value {
        match v {
            Value::Node(_) => self.fwd(v),
            Value::Struct(id, xs) => Value::Struct(*id, Rc::new(xs.iter().map(|x| self.deep_fwd(x)).collect())),
            Value::Tuple(xs) => Value::Tuple(Rc::new(xs.iter().map(|x| self.deep_fwd(x)).collect())),
            Value::List(xs) => Value::List(Rc::new(xs.iter().map(|x| self.deep_fwd(x)).collect())),
            Value::Enum(e, k, xs) => Value::Enum(*e, *k, Rc::new(xs.iter().map(|x| self.deep_fwd(x)).collect())),
            other => other.clone(),
        }
    }

    fn push(&self, val: Value, op: GOp) -> Value {
        self.tape.borrow_mut().as_mut().expect("an active tape").push(val, op)
    }

    fn node_of(&self, v: &Value) -> u32 {
        match v {
            Value::Node(i) => *i,
            other => match self.push(other.clone(), GOp::Const) {
                Value::Node(i) => i,
                _ => unreachable!("push returns a node"),
            },
        }
    }

    // ---------------------------------------------------------------- arithmetic

    /// Arithmetic that `bin` hands over: tracked values, and structs/tuples field by field.
    pub(super) fn bin_diff(&self, op: BinOp, a: Value, b: Value, sp: Span) -> R<Option<Value>> {
        let arith = matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Pow | BinOp::Rem);
        let fields = |v: &Value| matches!(v, Value::Struct(..) | Value::Tuple(_));
        if arith && (fields(&a) || fields(&b)) {
            return self.bin_fields(op, a, b, sp).map(Some);
        }
        if !is_node(&a) && !is_node(&b) {
            return Ok(None);
        }
        let r = self.bin(op, self.fwd(&a), self.fwd(&b), sp)?;
        if !arith {
            return Ok(Some(r)); // comparisons: plain booleans, no gradient
        }
        let (ia, ib) = (self.node_of(&a), self.node_of(&b));
        Ok(Some(self.push(r, GOp::Bin(op, ia, ib))))
    }

    fn bin_fields(&self, op: BinOp, a: Value, b: Value, sp: Span) -> R<Value> {
        let zip = |xs: &[Value], ys: &[Value]| -> R<Vec<Value>> {
            xs.iter().zip(ys).map(|(x, y)| self.bin(op, x.clone(), y.clone(), sp)).collect()
        };
        let left = |xs: &[Value], y: &Value| -> R<Vec<Value>> { xs.iter().map(|x| self.bin(op, x.clone(), y.clone(), sp)).collect() };
        let right = |x: &Value, ys: &[Value]| -> R<Vec<Value>> { ys.iter().map(|y| self.bin(op, x.clone(), y.clone(), sp)).collect() };
        Ok(match (&a, &b) {
            (Value::Struct(id, xs), Value::Struct(_, ys)) => Value::Struct(*id, Rc::new(zip(xs, ys)?)),
            (Value::Tuple(xs), Value::Tuple(ys)) => Value::Tuple(Rc::new(zip(xs, ys)?)),
            (Value::Struct(id, xs), y) => Value::Struct(*id, Rc::new(left(xs, y)?)),
            (Value::Tuple(xs), y) => Value::Tuple(Rc::new(left(xs, y)?)),
            (x, Value::Struct(id, ys)) => Value::Struct(*id, Rc::new(right(x, ys)?)),
            (x, Value::Tuple(ys)) => Value::Tuple(Rc::new(right(x, ys)?)),
            _ => unreachable!("field arithmetic without fields"),
        })
    }

    pub(super) fn neg(&self, v: Value, sp: Span) -> R<Value> {
        match v {
            Value::Node(i) => {
                let r = self.map_plain(MapFn::Neg, &self.fwd(&Value::Node(i)), sp)?;
                Ok(self.push(r, GOp::Map(MapFn::Neg, i)))
            }
            Value::Struct(id, xs) => Ok(Value::Struct(id, Rc::new(xs.iter().map(|x| self.neg(x.clone(), sp)).collect::<R<_>>()?))),
            Value::Tuple(xs) => Ok(Value::Tuple(Rc::new(xs.iter().map(|x| self.neg(x.clone(), sp)).collect::<R<_>>()?))),
            Value::Float(x) => Ok(Value::Float(-x)),
            Value::Int(i) => i.checked_neg().map(Value::Int).ok_or_else(|| overflow(sp)),
            Value::Tensor(t) => self.tensor_unary(UnOp::Neg, &t, sp),
            other => unreachable!("negation of {other:?}"),
        }
    }

    /// f applied element-wise to a plain float or tensor.
    fn map_plain(&self, f: MapFn, v: &Value, sp: Span) -> R<Value> {
        let g = |x: f64| match f {
            MapFn::Neg => -x,
            MapFn::Exp => libm::exp(x),
            MapFn::Ln => libm::log(x),
            MapFn::Sqrt => x.sqrt(),
            MapFn::Abs => x.abs(),
            MapFn::Tanh => libm::tanh(x),
            MapFn::Sigmoid => sigmoid(x),
            MapFn::Relu => x.max(0.0),
            MapFn::Softplus => softplus(x),
            MapFn::FillNan => x,
        };
        match v {
            Value::Float(x) => Ok(Value::Float(g(*x))),
            Value::Tensor(t) => self.map_tensor(t, sp, g),
            other => unreachable!("math on {other:?}"),
        }
    }

    /// f applied to a value: tracked (on the tape), plain, or a struct/tuple field by field.
    fn map_any(&self, f: MapFn, v: &Value, sp: Span) -> R<Value> {
        match v {
            Value::Node(i) => {
                let r = self.map_plain(f, &self.fwd(v), sp)?;
                Ok(self.push(r, GOp::Map(f, *i)))
            }
            Value::Struct(id, xs) => Ok(Value::Struct(*id, Rc::new(xs.iter().map(|x| self.map_any(f, x, sp)).collect::<R<_>>()?))),
            Value::Tuple(xs) => Ok(Value::Tuple(Rc::new(xs.iter().map(|x| self.map_any(f, x, sp)).collect::<R<_>>()?))),
            other => self.map_plain(f, other, sp),
        }
    }

    fn map_tensor(&self, t: &TensorVal, sp: Span, g: impl Fn(f64) -> f64) -> R<Value> {
        let r = tensor::map_n(&[tensor::Op::T(t)], false, |e| match e[0] {
            tensor::Elem::F(x) => tensor::Elem::F(g(x)),
            tensor::Elem::B(_) => unreachable!("math on a bool tensor"),
        });
        r.map(tensors::wrap).map_err(|e| self.terr(e, sp))
    }

    /// The zero of the tangent space of `v` (same shape, all zeros).
    fn zero_of(&self, v: &Value) -> Value {
        match v {
            Value::Node(_) => crate::autodiff::zero_like(&self.fwd(v)),
            Value::Float(_) => Value::Float(0.0),
            Value::Tensor(t) => Value::Tensor(Rc::new(tensor::full(t.axes.clone(), t.shape.clone(), 0.0))),
            Value::Struct(id, xs) => Value::Struct(*id, Rc::new(xs.iter().map(|x| self.zero_of(x)).collect())),
            Value::Tuple(xs) => Value::Tuple(Rc::new(xs.iter().map(|x| self.zero_of(x)).collect())),
            Value::List(xs) => Value::List(Rc::new(xs.iter().map(|x| self.zero_of(x)).collect())),
            other => other.clone(),
        }
    }

    // ---------------------------------------------------------------- tensors

    pub(super) fn select_diff(&self, base: &Value, sels: &[(u32, crate::tensor::Sel)], sp: Span) -> R<Value> {
        let Value::Node(i) = base else { unreachable!("select_diff of a plain value") };
        let Value::Tensor(t) = self.fwd(base) else { unreachable!("tracked tensor") };
        let r = tensor::select(&t, sels).map_err(|e| self.terr(e, sp))?;
        Ok(self.push(tensors::wrap(r), GOp::Select(*i, sels.to_vec())))
    }

    /// `x[sels] = v` when x or v is tracked: a new node replaces x.
    pub(super) fn assign_diff(&self, old: &Value, sels: &[(u32, crate::tensor::Sel)], v: &Value, sp: Span) -> R<Value> {
        let Value::Tensor(t) = self.fwd(old) else { unreachable!("tracked tensor") };
        let mut nt = (*t).clone();
        let fv = self.fwd(v);
        tensor::assign(&mut nt, sels, tensors::as_op(&fv)).map_err(|e| self.terr(e, sp))?;
        let (io, iv) = (self.node_of(old), self.node_of(v));
        Ok(self.push(Value::Tensor(Rc::new(nt)), GOp::Assign(io, iv, sels.to_vec())))
    }

    // ---------------------------------------------------------------- builtins

    fn axis(v: &Value) -> Option<u32> {
        match v {
            Value::Int(a) if *a >= 0 => Some(*a as u32),
            _ => None,
        }
    }

    /// M4 builtins, and builtins applied to tracked values or structs. None: the M1/M2 code
    /// handles it (after any tracked value has been replaced by its plain value).
    pub(super) fn diff_builtin(&mut self, b: Builtin, v: &mut [Value], sp: Span) -> R<Option<Value>> {
        use Builtin as B;
        match b {
            B::Grad | B::ValueAndGrad => {
                let wrt = as_int(v[1].clone()) as usize;
                let with_value = matches!(v[2], Value::Bool(true));
                return Ok(Some(Value::Func(Rc::new(FuncVal::Grad { f: v[0].clone(), wrt, with_value }))));
            }
            B::Randn => {
                let seed = as_int(v[0].clone()) as u64;
                let mut axes = Vec::new();
                let mut shape = Vec::new();
                for pair in v[1..].chunks(2) {
                    let a = Self::axis(&pair[0]).expect("axis id");
                    let n = as_int(pair[1].clone());
                    let Ok(n) = usize::try_from(n) else {
                        return Err(Ctl::Err(Diag::new("R0203", sp, format!("invalid size {n} for axis `{}`", self.ck.axes[a as usize].name))));
                    };
                    axes.push(a);
                    shape.push(n);
                }
                if shape.iter().try_fold(1usize, |acc, n| acc.checked_mul(*n)).is_none_or(|n| n > 1 << 28) {
                    return Err(Ctl::Err(Diag::new("R0203", sp, "tensor too large")));
                }
                return Ok(Some(Value::Tensor(Rc::new(tensor::randn(seed, axes, shape)))));
            }
            B::Tanh | B::Sigmoid | B::Relu | B::Softplus => {
                let f = match b {
                    B::Tanh => MapFn::Tanh,
                    B::Sigmoid => MapFn::Sigmoid,
                    B::Relu => MapFn::Relu,
                    _ => MapFn::Softplus,
                };
                return self.map_any(f, &v[0], sp).map(Some);
            }
            B::SmoothMax | B::SmoothMin => {
                let s = float(&self.fwd(&v[2]));
                let (x, y) = (v[0].clone(), v[1].clone());
                if !self.tracking() {
                    // the exact rule outside `grad`
                    let base = if b == B::SmoothMax { B::Max } else { B::Min };
                    return self.tensor_builtin(base, &[x.clone(), y.clone()], sp).and_then(|r| match r {
                        Some(r) => Ok(Some(r)),
                        None => Ok(Some(self.max_plain(base, &x, &y))),
                    });
                }
                // inside `grad`: max(x, y) ≈ y + s·softplus((x − y)/s); min(x, y) = −max(−x, −y)
                let (x, y) = if b == B::SmoothMax { (x, y) } else { (self.neg(x, sp)?, self.neg(y, sp)?) };
                let d = self.bin(BinOp::Sub, x, y.clone(), sp)?;
                let d = self.bin(BinOp::Div, d, Value::Float(s), sp)?;
                let sp_ = self.map_any(MapFn::Softplus, &d, sp)?;
                let m = self.bin(BinOp::Mul, sp_, Value::Float(s), sp)?;
                let r = self.bin(BinOp::Add, y, m, sp)?;
                return Ok(Some(if b == B::SmoothMax { r } else { self.neg(r, sp)? }));
            }
            B::SmoothStep => {
                let s = float(&self.fwd(&v[1]));
                if !self.tracking() {
                    let step = |x: f64| if x > 0.0 { 1.0 } else { 0.0 };
                    return Ok(Some(match &v[0] {
                        Value::Float(x) => Value::Float(step(*x)),
                        Value::Tensor(t) => self.map_tensor(t, sp, step)?,
                        other => unreachable!("smooth_step of {other:?}"),
                    }));
                }
                let d = self.bin(BinOp::Div, v[0].clone(), Value::Float(s), sp)?;
                return self.map_any(MapFn::Sigmoid, &d, sp).map(Some);
            }
            B::Sqrt | B::Exp | B::Ln | B::Abs if v.iter().any(|x| has_node(x) || matches!(x, Value::Struct(..) | Value::Tuple(_))) => {
                let f = match b {
                    B::Sqrt => MapFn::Sqrt,
                    B::Exp => MapFn::Exp,
                    B::Ln => MapFn::Ln,
                    _ => MapFn::Abs,
                };
                return self.map_any(f, &v[0], sp).map(Some);
            }
            B::ZerosLike if matches!(v[0], Value::Struct(..) | Value::Tuple(_) | Value::List(_) | Value::Float(_) | Value::Node(_)) => {
                return Ok(Some(self.zero_of(&v[0])));
            }
            _ => {}
        }
        if !v.iter().any(has_node) {
            return Ok(None);
        }
        // tracked values reaching a builtin
        let out = match b {
            B::Sum | B::Mean | B::Prod | B::Max | B::Min | B::NanSum | B::NanMean
                if is_node(&v[0]) && matches!(v.get(1), Some(Value::Int(_))) && v.len() == 2 =>
            {
                let Value::Tensor(t) = self.fwd(&v[0]) else { unreachable!("tracked tensor") };
                let kind = match b {
                    B::Sum => Red::Sum,
                    B::Mean => Red::Mean,
                    B::Prod => Red::Prod,
                    B::Max => Red::Max,
                    B::Min => Red::Min,
                    B::NanSum => Red::NanSum,
                    _ => Red::NanMean,
                };
                let ax = Self::axis(&v[1]);
                let r = tensor::reduce(&t, ax, kind).map_err(|e| self.terr(e, sp))?;
                let Value::Node(i) = v[0] else { unreachable!() };
                Some(self.push(tensors::wrap(r), GOp::Reduce(i, ax, kind)))
            }
            B::Sum | B::Mean | B::Prod if matches!(v[0], Value::List(_)) => {
                // a list with tracked elements: fold with the tracked arithmetic
                let Value::List(xs) = &v[0] else { unreachable!() };
                if xs.is_empty() && b == B::Mean {
                    return Err(Ctl::Err(Diag::new("R0004", sp, "`mean` of an empty list")));
                }
                let (start, op) = if b == B::Prod { (Value::Float(1.0), BinOp::Mul) } else { (Value::Float(0.0), BinOp::Add) };
                let mut acc = start;
                for x in xs.iter() {
                    acc = self.bin(op, acc, x.clone(), sp)?;
                }
                if b == B::Mean {
                    acc = self.bin(BinOp::Div, acc, Value::Float(xs.len() as f64), sp)?;
                }
                Some(acc)
            }
            B::Minimum | B::Maximum => {
                // pick by the plain values, keep the tracked element
                let Value::List(xs) = &v[0] else { unreachable!() };
                let Some(first) = xs.first() else {
                    return Err(Ctl::Err(Diag::new("R0004", sp, format!("`{}` of an empty list", builtin_name(b)))));
                };
                let mut best = first.clone();
                for x in xs.iter().skip(1) {
                    let (fx, fb) = (float(&self.fwd(x)), float(&self.fwd(&best)));
                    if (b == B::Minimum && fx < fb) || (b == B::Maximum && fb < fx) {
                        best = x.clone();
                    }
                }
                Some(best)
            }
            B::Max | B::Min if v.len() == 2 => {
                // element-wise max/min of two values: a `where` on the plain values
                let (fa, fb) = (self.fwd(&v[0]), self.fwd(&v[1]));
                let cond = self.bin(if b == B::Max { BinOp::Ge } else { BinOp::Le }, fa.clone(), fb.clone(), sp)?;
                let plain = self.max_plain(b, &fa, &fb);
                let plain = match self.tensor_builtin(b, &[fa, fb], sp)? {
                    Some(r) => r,
                    None => plain,
                };
                let (ia, ib) = (self.node_of(&v[0]), self.node_of(&v[1]));
                Some(self.push(plain, GOp::Where(cond, ia, ib)))
            }
            B::Where => {
                let cond = self.fwd(&v[0]);
                let plain = self.tensor_builtin(B::Where, &[cond.clone(), self.fwd(&v[1]), self.fwd(&v[2])], sp)?.expect("where");
                let (ia, ib) = (self.node_of(&v[1]), self.node_of(&v[2]));
                Some(self.push(plain, GOp::Where(cond, ia, ib)))
            }
            B::FillNan if is_node(&v[0]) => {
                let plain = self.tensor_builtin(B::FillNan, &[self.fwd(&v[0]), self.fwd(&v[1])], sp)?.expect("fill_nan");
                let Value::Node(i) = v[0] else { unreachable!() };
                Some(self.push(plain, GOp::Map(MapFn::FillNan, i)))
            }
            B::Contract => {
                let (fx, fy) = (self.fwd(&v[0]), self.fwd(&v[1]));
                let plain = self.tensor_builtin(B::Contract, &[fx, fy, v[2].clone()], sp)?.expect("contract");
                let ax = Self::axis(&v[2]).expect("axis id");
                let (ix, iy) = (self.node_of(&v[0]), self.node_of(&v[1]));
                Some(self.push(plain, GOp::Contract(ix, iy, vec![ax])))
            }
            B::Tensor => {
                let Value::List(xs) = &v[0] else { unreachable!() };
                let plain_list = Value::List(Rc::new(xs.iter().map(|x| self.fwd(x)).collect()));
                let plain = self.tensor_builtin(B::Tensor, &[plain_list, v[1].clone(), v[2].clone()], sp)?.expect("tensor");
                let parts: Vec<u32> = xs.iter().map(|x| self.node_of(x)).collect();
                Some(self.push(plain, GOp::Stack(parts)))
            }
            B::ToList if is_node(&v[0]) => {
                let Value::Tensor(t) = self.fwd(&v[0]) else { unreachable!("tracked tensor") };
                let Value::Node(i) = v[0] else { unreachable!() };
                let TData::F(d) = &t.data else { unreachable!("f64 tensor") };
                Some(Value::List(Rc::new(d.iter().enumerate().map(|(k, x)| self.push(Value::Float(*x), GOp::Elem(i, k))).collect())))
            }
            B::ToFloat if is_node(&v[0]) => Some(v[0].clone()),
            B::Full | B::FullLike => {
                // a tracked fill value: zeros + value, so the gradient sums over the tensor
                let k = if b == B::Full { 0 } else { 1 };
                let fill = v[k].clone();
                let mut plain_args = v.to_vec();
                plain_args[k] = Value::Float(0.0);
                let zeros = self.tensor_builtin(b, &plain_args, sp)?.expect("full");
                Some(self.bin(BinOp::Add, zeros, fill, sp)?)
            }
            _ => None,
        };
        if out.is_some() {
            return Ok(out);
        }
        // not differentiable (counts, ranks, rounding, printing...): plain values from here on
        for x in v.iter_mut() {
            *x = self.deep_fwd(x);
        }
        Ok(None)
    }

    fn max_plain(&self, b: Builtin, x: &Value, y: &Value) -> Value {
        match (x, y) {
            (Value::Float(p), Value::Float(q)) => {
                if p.is_nan() || q.is_nan() {
                    Value::Float(f64::NAN)
                } else if b == Builtin::Max {
                    Value::Float(p.max(*q))
                } else {
                    Value::Float(p.min(*q))
                }
            }
            (Value::Int(p), Value::Int(q)) => Value::Int(if b == Builtin::Max { *p.max(q) } else { *p.min(q) }),
            _ => unreachable!("max/min of mixed values"),
        }
    }

    // ---------------------------------------------------------------- grad

    /// `grad(f)(args)`: runs f with argument `wrt` tracked, then walks the tape backwards.
    pub(super) fn run_grad(&mut self, f: &Value, wrt: usize, with_value: bool, mut args: Vec<Value>, sp: Span) -> R<Value> {
        if self.tracking() {
            return Err(Ctl::Err(
                Diag::new("R0401", sp, "`grad` inside `grad` is not supported yet").hint("second derivatives arrive in a later milestone"),
            ));
        }
        *self.tape.borrow_mut() = Some(Tape::default());
        let leaves = self.make_leaves(&args[wrt]);
        args[wrt] = leaves.clone();
        let r = self.call_value(f, args, sp);
        let tape = self.tape.borrow_mut().take().expect("the tape of this grad");
        let out = r?;
        let (value, grads) = match &out {
            Value::Node(i) => {
                let g = tape.backward(*i).map_err(|e| self.terr(e, sp))?;
                (tape.val(*i).clone(), g)
            }
            // the result does not depend on the argument: zero gradient
            Value::Float(_) => (out.clone(), Vec::new()),
            other => unreachable!("grad of a function returning {other:?}"),
        };
        let g = Self::collect(&leaves, &tape, &grads);
        Ok(if with_value { Value::Tuple(Rc::new(vec![value, g])) } else { g })
    }

    fn make_leaves(&self, v: &Value) -> Value {
        match v {
            Value::Float(_) | Value::Tensor(_) => self.push(v.clone(), GOp::Leaf),
            Value::Struct(id, xs) => Value::Struct(*id, Rc::new(xs.iter().map(|x| self.make_leaves(x)).collect())),
            Value::Tuple(xs) => Value::Tuple(Rc::new(xs.iter().map(|x| self.make_leaves(x)).collect())),
            Value::List(xs) => Value::List(Rc::new(xs.iter().map(|x| self.make_leaves(x)).collect())),
            other => other.clone(),
        }
    }

    fn collect(leaves: &Value, tape: &Tape, grads: &[Option<Value>]) -> Value {
        match leaves {
            Value::Node(i) => match grads.get(*i as usize).cloned().flatten() {
                Some(g) => g,
                None => crate::autodiff::zero_like(tape.val(*i)),
            },
            Value::Struct(id, xs) => Value::Struct(*id, Rc::new(xs.iter().map(|x| Self::collect(x, tape, grads)).collect())),
            Value::Tuple(xs) => Value::Tuple(Rc::new(xs.iter().map(|x| Self::collect(x, tape, grads)).collect())),
            Value::List(xs) => Value::List(Rc::new(xs.iter().map(|x| Self::collect(x, tape, grads)).collect())),
            other => other.clone(),
        }
    }
}
