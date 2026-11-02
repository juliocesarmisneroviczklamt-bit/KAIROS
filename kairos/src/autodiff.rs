//! Reverse-mode automatic differentiation (M4).
//!
//! `grad(f)(x)` runs `f` with the floats and float tensors of `x` replaced by tracked values
//! (`Value::Node`). Every operation on them is recorded on a tape, following the path that
//! actually runs (`if`, loops and calls just execute), and the tape is then walked backwards from
//! the scalar result. Broadcasting by axis name is undone by summing over the extra axes, and a
//! contraction's gradients are contractions too.

use crate::ast::BinOp;
use crate::tensor::{self, Elem, Op, Red, Sel, TData, TResult, TensorVal, map_n};
use crate::value::Value;
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MapFn {
    Neg,
    Exp,
    Ln,
    Sqrt,
    Abs,
    Tanh,
    Sigmoid,
    Relu,
    Softplus,
    FillNan,
    /// M18: `ternary`, by the straight-through estimator: the gradient passes as it is
    Ste,
}

/// How a node was computed from its parents (node ids).
#[derive(Clone, Debug)]
pub enum GOp {
    /// a value we differentiate with respect to
    Leaf,
    /// a value without gradient
    Const,
    Bin(BinOp, u32, u32),
    Map(MapFn, u32),
    /// reduction over one axis (None: all axes)
    Reduce(u32, Option<u32>, Red),
    Select(u32, Vec<(u32, Sel)>),
    /// `old[sels] = val` as a new tensor
    Assign(u32, u32, Vec<(u32, Sel)>),
    /// `where(cond, a, b)`; the condition has no gradient
    Where(Value, u32, u32),
    Contract(u32, u32, Vec<u32>),
    /// one-axis tensor built from scalars
    Stack(Vec<u32>),
    /// element `k` of a one-axis tensor
    Elem(u32, usize),
}

pub struct Node {
    pub val: Value,
    pub op: GOp,
}

#[derive(Default)]
pub struct Tape {
    pub nodes: Vec<Node>,
}

/// f64 operand of an element-wise computation.
fn op(v: &Value) -> Op<'_> {
    match v {
        Value::Tensor(t) => Op::T(t),
        Value::Float(x) => Op::F(*x),
        other => unreachable!("not a differentiable value: {other:?}"),
    }
}

fn wrap(t: TensorVal) -> Value {
    if t.axes.is_empty() {
        match t.data {
            TData::F(v) => Value::Float(v[0]),
            TData::B(v) => Value::Bool(v[0]),
            TData::T(v) => Value::T3(v.get(0)),
        }
    } else {
        Value::Tensor(Rc::new(t))
    }
}

/// Element-wise f(a, b) with alignment by axis name.
pub fn ew(a: &Value, b: &Value, f: impl Fn(f64, f64) -> f64) -> TResult<Value> {
    if let (Value::Float(x), Value::Float(y)) = (a, b) {
        return Ok(Value::Float(f(*x, *y)));
    }
    let t = map_n(&[op(a), op(b)], false, |e| match (e[0], e[1]) {
        (Elem::F(x), Elem::F(y)) => Elem::F(f(x, y)),
        _ => unreachable!("bool in a gradient"),
    })?;
    Ok(wrap(t))
}

/// Element-wise f(a, b, c).
fn ew3(a: &Value, b: &Value, c: &Value, f: impl Fn(f64, f64, f64) -> f64) -> TResult<Value> {
    if let (Value::Float(x), Value::Float(y), Value::Float(z)) = (a, b, c) {
        return Ok(Value::Float(f(*x, *y, *z)));
    }
    let t = map_n(&[op(a), op(b), op(c)], false, |e| match (e[0], e[1], e[2]) {
        (Elem::F(x), Elem::F(y), Elem::F(z)) => Elem::F(f(x, y, z)),
        _ => unreachable!("bool in a gradient"),
    })?;
    Ok(wrap(t))
}

fn as_tensor(v: &Value) -> TensorVal {
    match v {
        Value::Tensor(t) => (**t).clone(),
        Value::Float(x) => TensorVal::new_f(Vec::new(), Vec::new(), vec![*x]),
        other => unreachable!("not a differentiable value: {other:?}"),
    }
}

/// Zero with the shape of `v`.
pub fn zero_like(v: &Value) -> Value {
    match v {
        Value::Tensor(t) => Value::Tensor(Rc::new(tensor::full(t.axes.clone(), t.shape.clone(), 0.0))),
        _ => Value::Float(0.0),
    }
}

/// The gradient `g` of a broadcast result, summed back to the shape of `target`.
fn unbroadcast(g: &Value, target: &Value) -> TResult<Value> {
    match (g, target) {
        (Value::Float(_), _) => match target {
            // a scalar result can only come from scalar operands
            Value::Tensor(t) => ew(&Value::Tensor(Rc::new(tensor::full(t.axes.clone(), t.shape.clone(), 0.0))), g, |_, y| y),
            _ => Ok(g.clone()),
        },
        (Value::Tensor(gt), Value::Float(_)) => Ok(Value::Float(gt.f().iter().sum())),
        (Value::Tensor(gt), Value::Tensor(t)) => Ok(wrap(tensor::sum_to(gt, &t.axes)?)),
        _ => unreachable!("gradient of a non-differentiable value"),
    }
}

impl Tape {
    pub fn push(&mut self, val: Value, op: GOp) -> Value {
        self.nodes.push(Node { val, op });
        Value::Node((self.nodes.len() - 1) as u32)
    }

    pub fn val(&self, i: u32) -> &Value {
        &self.nodes[i as usize].val
    }

    /// Gradients of node `out` (a scalar) with respect to every node.
    pub fn backward(&self, out: u32) -> TResult<Vec<Option<Value>>> {
        let n = out as usize + 1;
        let mut grads: Vec<Option<Value>> = vec![None; n];
        grads[out as usize] = Some(Value::Float(1.0));
        for i in (0..n).rev() {
            let Some(g) = grads[i].clone() else { continue };
            let node = &self.nodes[i];
            let add = |grads: &mut Vec<Option<Value>>, j: u32, c: Value| -> TResult<()> {
                let j = j as usize;
                grads[j] = Some(match grads[j].take() {
                    None => c,
                    Some(prev) => ew(&prev, &c, |a, b| a + b)?,
                });
                Ok(())
            };
            match &node.op {
                GOp::Leaf | GOp::Const => {}
                GOp::Bin(bop, a, b) => {
                    let (av, bv) = (self.val(*a), self.val(*b));
                    let b_const = matches!(self.nodes[*b as usize].op, GOp::Const);
                    let a_const = matches!(self.nodes[*a as usize].op, GOp::Const);
                    let (ga, gb) = match bop {
                        BinOp::Add => (g.clone(), g.clone()),
                        BinOp::Sub => (g.clone(), ew(&g, &g, |x, _| -x)?),
                        BinOp::Mul => (ew(&g, bv, |x, y| x * y)?, ew(&g, av, |x, y| x * y)?),
                        BinOp::Div => (ew(&g, bv, |x, y| x / y)?, ew3(&g, av, bv, |x, p, q| -x * p / (q * q))?),
                        BinOp::Pow => {
                            let ga = ew3(&g, av, bv, |x, p, q| x * q * libm::pow(p, q - 1.0))?;
                            // d/db only when the exponent itself is tracked: ln(a) is NaN for a < 0
                            let gb = if b_const { zero_like(&g) } else { ew3(&g, &node.val, av, |x, o, p| x * o * libm::log(p))? };
                            (ga, gb)
                        }
                        BinOp::Rem => (g.clone(), ew3(&g, av, bv, |x, p, q| -x * (p / q).trunc())?),
                        other => unreachable!("comparison {other:?} on the tape"),
                    };
                    if !a_const {
                        add(&mut grads, *a, unbroadcast(&ga, av)?)?;
                    }
                    if !b_const {
                        add(&mut grads, *b, unbroadcast(&gb, bv)?)?;
                    }
                }
                GOp::Map(f, a) => {
                    let (av, o) = (self.val(*a), &node.val);
                    let c = match f {
                        MapFn::Neg => ew(&g, &g, |x, _| -x)?,
                        MapFn::Exp => ew(&g, o, |x, y| x * y)?,
                        MapFn::Ln => ew(&g, av, |x, y| x / y)?,
                        MapFn::Sqrt => ew(&g, o, |x, y| 0.5 * x / y)?,
                        MapFn::Abs => ew(&g, av, |x, y| if y > 0.0 { x } else if y < 0.0 { -x } else { 0.0 })?,
                        MapFn::Tanh => ew(&g, o, |x, y| x * (1.0 - y * y))?,
                        MapFn::Sigmoid => ew(&g, o, |x, y| x * y * (1.0 - y))?,
                        MapFn::Relu => ew(&g, av, |x, y| if y > 0.0 { x } else { 0.0 })?,
                        MapFn::Softplus => ew(&g, av, |x, y| x * sigmoid(y))?,
                        MapFn::FillNan => ew(&g, av, |x, y| if y.is_nan() { 0.0 } else { x })?,
                        MapFn::Ste => g.clone(),
                    };
                    add(&mut grads, *a, c)?;
                }
                GOp::Reduce(a, axis, kind) => {
                    let av = self.val(*a);
                    let c = reduce_grad(av, &node.val, &g, *axis, *kind)?;
                    add(&mut grads, *a, c)?;
                }
                GOp::Select(a, sels) => {
                    let av = self.val(*a);
                    let mut z = as_tensor(&zero_like(av));
                    tensor::assign(&mut z, sels, op(&g))?;
                    add(&mut grads, *a, wrap(z))?;
                }
                GOp::Assign(old, v, sels) => {
                    // the old tensor keeps the gradient outside the region; the value gets the region
                    let mut gt = as_tensor(&g);
                    let region = wrap(tensor::select(&gt, sels)?);
                    let zero = zero_like(&region);
                    tensor::assign(&mut gt, sels, op(&zero))?;
                    if !matches!(self.nodes[*old as usize].op, GOp::Const) {
                        add(&mut grads, *old, wrap(gt))?;
                    }
                    if !matches!(self.nodes[*v as usize].op, GOp::Const) {
                        add(&mut grads, *v, unbroadcast(&region, self.val(*v))?)?;
                    }
                }
                GOp::Where(cond, a, b) => {
                    let pick = |first: bool| -> TResult<Value> {
                        let t = map_n(&[cond_op(cond), op(&g)], false, |e| match (e[0], e[1]) {
                            (Elem::B(c), Elem::F(x)) => Elem::F(if c == first { x } else { 0.0 }),
                            _ => unreachable!("where condition"),
                        })?;
                        Ok(wrap(t))
                    };
                    add(&mut grads, *a, unbroadcast(&pick(true)?, self.val(*a))?)?;
                    add(&mut grads, *b, unbroadcast(&pick(false)?, self.val(*b))?)?;
                }
                GOp::Contract(a, b, over) => {
                    let (at, bt, gt) = (as_tensor(self.val(*a)), as_tensor(self.val(*b)), as_tensor(&g));
                    let only_b: Vec<u32> = bt.axes.iter().copied().filter(|x| !at.axes.contains(x)).collect();
                    let only_a: Vec<u32> = at.axes.iter().copied().filter(|x| !bt.axes.contains(x)).collect();
                    let _ = over;
                    add(&mut grads, *a, wrap(tensor::contract_over(&gt, &bt, &only_b)?))?;
                    add(&mut grads, *b, wrap(tensor::contract_over(&gt, &at, &only_a)?))?;
                }
                GOp::Stack(parts) => {
                    let gt = as_tensor(&g);
                    for (k, p) in parts.iter().enumerate() {
                        if !matches!(self.nodes[*p as usize].op, GOp::Const) {
                            add(&mut grads, *p, Value::Float(gt.f()[k]))?;
                        }
                    }
                }
                GOp::Elem(a, k) => {
                    let mut z = as_tensor(&zero_like(self.val(*a)));
                    if let TData::F(d) = &mut z.data {
                        d[*k] = match g {
                            Value::Float(x) => x,
                            _ => unreachable!("element gradient"),
                        };
                    }
                    add(&mut grads, *a, wrap(z))?;
                }
            }
        }
        Ok(grads)
    }
}

fn cond_op(v: &Value) -> Op<'_> {
    match v {
        Value::Tensor(t) => Op::T(t),
        Value::Bool(b) => Op::B(*b),
        other => unreachable!("where condition {other:?}"),
    }
}

pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + libm::exp(-x))
    } else {
        let e = libm::exp(x);
        e / (1.0 + e)
    }
}

/// log(1 + e^x) without overflow
pub fn softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { libm::log1p(libm::exp(x)) }
}

/// Gradient of a reduction: `g` (shape of the result) spread back over the reduced axis.
fn reduce_grad(a: &Value, out: &Value, g: &Value, axis: Option<u32>, kind: Red) -> TResult<Value> {
    let at = as_tensor(a);
    // g and the result, broadcast to the shape of `a`
    let zero = Value::Tensor(Rc::new(tensor::full(at.axes.clone(), at.shape.clone(), 0.0)));
    let gb = ew(&zero, g, |_, y| y)?;
    let ob = ew(&zero, out, |_, y| y)?;
    let n = match axis {
        Some(ax) => at.size(ax).unwrap_or(1),
        None => at.len(),
    } as f64;
    match kind {
        Red::Sum => Ok(gb),
        Red::Mean => ew(&gb, &gb, |x, _| x / n),
        Red::NanSum => ew(&gb, a, |x, y| if y.is_nan() { 0.0 } else { x }),
        Red::NanMean => {
            let cnt = wrap(tensor::reduce(&at, axis, Red::NanCount)?);
            let cb = ew(&zero, &cnt, |_, y| y)?;
            ew3(&gb, a, &cb, |x, y, c| if y.is_nan() { 0.0 } else { x / c })
        }
        Red::Max | Red::Min => {
            // the gradient goes to the extreme element(s), shared equally
            let hits = ew(a, &ob, |x, o| if x == o { 1.0 } else { 0.0 })?;
            let ht = as_tensor(&hits);
            let cnt = wrap(tensor::reduce(&ht, axis, Red::Sum)?);
            let cb = ew(&zero, &cnt, |_, y| y)?;
            ew3(&gb, &hits, &cb, |x, h, c| x * h / c)
        }
        Red::Prod => Ok(wrap(tensor::prod_grad(&at, axis, &as_tensor(&gb))?)),
        Red::NanProd | Red::NanCount | Red::Count => Ok(zero),
    }
}
