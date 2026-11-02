//! Named-axis tensors (M2): dense, row-major in the order of `axes`, shared copy-on-write (`Rc`).
//!
//! Element-wise operations align operands by axis NAME: one operand's axes must contain the
//! other's (the type checker guarantees it) and the smaller one is repeated along the missing
//! axes. Reductions, slices, ranks and contractions always name the axis. Kernels are plain
//! loops for now; the native backend (M7) fuses and vectorizes them.

use std::cmp::Ordering;

#[derive(Clone, Debug, PartialEq)]
pub enum TData {
    F(Vec<f64>),
    B(Vec<bool>),
    /// M18: trits (-1, 0, +1)
    T(Trits),
}

/// Trits (M18), five to a byte (3^5 = 243 <= 256): a byte holds Σ (t_k + 1)·3^k for its five, and
/// the unused places of the last byte hold 0, so two equal sequences have equal bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct Trits {
    len: usize,
    bytes: Vec<u8>,
}

const POW3: [u8; 5] = [1, 3, 9, 27, 81];

impl Trits {
    pub fn from_slice(v: &[i8]) -> Self {
        let mut bytes = vec![0u8; v.len().div_ceil(5)];
        for (i, t) in v.iter().enumerate() {
            bytes[i / 5] += (*t + 1) as u8 * POW3[i % 5];
        }
        Trits { len: v.len(), bytes }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn get(&self, i: usize) -> i8 {
        ((self.bytes[i / 5] / POW3[i % 5]) % 3) as i8 - 1
    }

    pub fn set(&mut self, i: usize, t: i8) {
        let (b, p) = (self.bytes[i / 5], POW3[i % 5]);
        self.bytes[i / 5] = b - ((b / p) % 3) * p + (t + 1) as u8 * p;
    }

    pub fn to_vec(&self) -> Vec<i8> {
        (0..self.len).map(|i| self.get(i)).collect()
    }

    /// the bytes it takes (for the tests)
    pub fn packed_len(&self) -> usize {
        self.bytes.len()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TensorVal {
    pub axes: Vec<u32>,
    pub shape: Vec<usize>,
    pub data: TData,
}

/// Errors carry axis ids; the interpreter turns them into messages with axis names.
#[derive(Debug)]
pub enum TErr {
    SizeMismatch { axis: u32, a: usize, b: usize },
    OutOfBounds { axis: u32, index: i64, size: usize },
    Reversed { axis: u32, lo: i64, hi: i64 },
    Shape(String),
}

pub type TResult<T> = Result<T, TErr>;

/// Operand of an element-wise operation: a tensor or a scalar repeated everywhere.
#[derive(Clone, Copy)]
pub enum Op<'a> {
    T(&'a TensorVal),
    F(f64),
    B(bool),
    /// a `t3` (M18)
    Trit(i8),
}

#[derive(Clone, Copy, Debug)]
pub enum Elem {
    F(f64),
    B(bool),
    /// a `t3` (M18)
    T(i8),
}

/// Selector along one axis: a single index (removes the axis) or a half-open range `[lo, hi)`.
#[derive(Clone, Copy, Debug)]
pub enum Sel {
    At(i64),
    Range(i64, i64),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Red {
    Sum,
    Mean,
    Prod,
    Max,
    Min,
    NanSum,
    NanMean,
    NanProd,
    NanCount,
    Count,
}

impl TensorVal {
    pub fn new_f(axes: Vec<u32>, shape: Vec<usize>, data: Vec<f64>) -> Self {
        TensorVal { axes, shape, data: TData::F(data) }
    }

    pub fn new_b(axes: Vec<u32>, shape: Vec<usize>, data: Vec<bool>) -> Self {
        TensorVal { axes, shape, data: TData::B(data) }
    }

    pub fn new_t(axes: Vec<u32>, shape: Vec<usize>, data: &[i8]) -> Self {
        TensorVal { axes, shape, data: TData::T(Trits::from_slice(data)) }
    }

    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn pos(&self, ax: u32) -> Option<usize> {
        self.axes.iter().position(|a| *a == ax)
    }

    pub fn size(&self, ax: u32) -> Option<usize> {
        self.pos(ax).map(|p| self.shape[p])
    }

    pub fn strides(&self) -> Vec<usize> {
        let n = self.shape.len();
        let mut s = vec![1usize; n];
        for k in (0..n.saturating_sub(1)).rev() {
            s[k] = s[k + 1] * self.shape[k + 1];
        }
        s
    }

    pub fn f(&self) -> &[f64] {
        match &self.data {
            TData::F(v) => v,
            TData::B(_) | TData::T(_) => panic!("expected a f64 tensor"),
        }
    }

    fn get(&self, off: usize) -> Elem {
        match &self.data {
            TData::F(v) => Elem::F(v[off]),
            TData::B(v) => Elem::B(v[off]),
            TData::T(v) => Elem::T(v.get(off)),
        }
    }

    /// value as f64 (bool -> 1/0)
    fn getf(&self, off: usize) -> f64 {
        match &self.data {
            TData::F(v) => v[off],
            TData::B(v) => {
                if v[off] {
                    1.0
                } else {
                    0.0
                }
            }
            TData::T(v) => f64::from(v.get(off)),
        }
    }
}

/// Strides of `t` over the axis list `axes` (0 for axes `t` does not have: repetition).
fn mapped(t: &TensorVal, axes: &[u32]) -> Vec<usize> {
    let st = t.strides();
    axes.iter().map(|a| t.pos(*a).map(|p| st[p]).unwrap_or(0)).collect()
}

/// Visits every element of `shape` in row-major order with one running offset per stride set.
fn walk(shape: &[usize], strides: &[Vec<usize>], mut f: impl FnMut(&[usize])) {
    let n: usize = shape.iter().product();
    let nd = shape.len();
    let mut idx = vec![0usize; nd];
    let mut offs = vec![0usize; strides.len()];
    for _ in 0..n {
        f(&offs);
        let mut k = nd;
        while k > 0 {
            k -= 1;
            idx[k] += 1;
            for (o, s) in offs.iter_mut().zip(strides) {
                *o += s[k];
            }
            if idx[k] < shape[k] {
                break;
            }
            for (o, s) in offs.iter_mut().zip(strides) {
                *o -= s[k] * idx[k];
            }
            idx[k] = 0;
        }
    }
}

/// Axes and sizes of the combination: the operand with the most axes is the frame; every other
/// tensor's axes must be a subset of it with equal sizes.
fn frame(ops: &[Op]) -> TResult<(Vec<u32>, Vec<usize>)> {
    let mut best: Option<&TensorVal> = None;
    for o in ops {
        if let Op::T(t) = o {
            if best.is_none_or(|b| t.axes.len() > b.axes.len()) {
                best = Some(t);
            }
        }
    }
    let Some(b) = best else { return Ok((Vec::new(), Vec::new())) };
    for o in ops {
        if let Op::T(t) = o {
            for (k, ax) in t.axes.iter().enumerate() {
                match b.pos(*ax) {
                    Some(p) if b.shape[p] != t.shape[k] => {
                        return Err(TErr::SizeMismatch { axis: *ax, a: b.shape[p], b: t.shape[k] });
                    }
                    Some(_) => {}
                    None => return Err(TErr::Shape("incompatible axes".into())),
                }
            }
        }
    }
    Ok((b.axes.clone(), b.shape.clone()))
}

/// Element-wise map over any number of operands, repeating smaller ones by axis name.
pub fn map_n(ops: &[Op], out_bool: bool, f: impl Fn(&[Elem]) -> Elem) -> TResult<TensorVal> {
    map_into(ops, if out_bool { Out::B } else { Out::F }, f)
}

/// The element type a map gives.
#[derive(Clone, Copy, PartialEq)]
pub enum Out {
    F,
    B,
    T,
}

/// The same, giving elements of the type `out`.
pub fn map_into(ops: &[Op], out: Out, f: impl Fn(&[Elem]) -> Elem) -> TResult<TensorVal> {
    let out_bool = out == Out::B;
    let (axes, shape) = frame(ops)?;
    let strides: Vec<Vec<usize>> = ops
        .iter()
        .map(|o| match o {
            Op::T(t) => mapped(t, &axes),
            _ => vec![0; axes.len()],
        })
        .collect();
    let n: usize = shape.iter().product();
    let (mut fo, mut bo, mut to) = (Vec::new(), Vec::new(), Vec::new());
    if out_bool {
        bo.reserve(n);
    } else if out == Out::F {
        fo.reserve(n);
    }
    let mut buf = vec![Elem::F(0.0); ops.len()];
    walk(&shape, &strides, |offs| {
        for (j, o) in ops.iter().enumerate() {
            buf[j] = match o {
                Op::T(t) => t.get(offs[j]),
                Op::F(x) => Elem::F(*x),
                Op::B(b) => Elem::B(*b),
                Op::Trit(t) => Elem::T(*t),
            };
        }
        match f(&buf) {
            Elem::F(x) => fo.push(x),
            Elem::B(b) => bo.push(b),
            Elem::T(t) => to.push(t),
        }
    });
    let data = match out {
        Out::F => TData::F(fo),
        Out::B => TData::B(bo),
        Out::T => TData::T(Trits::from_slice(&to)),
    };
    Ok(TensorVal { axes, shape, data })
}

fn fold(kind: Red, it: impl Iterator<Item = f64>) -> f64 {
    match kind {
        Red::Sum => it.sum(),
        Red::Prod => it.product(),
        Red::Mean => {
            let (s, c) = it.fold((0.0, 0usize), |(s, c), x| (s + x, c + 1));
            if c == 0 { f64::NAN } else { s / c as f64 }
        }
        Red::Max => it.fold(f64::NEG_INFINITY, |m, x| if m.is_nan() || x.is_nan() { f64::NAN } else { m.max(x) }),
        Red::Min => it.fold(f64::INFINITY, |m, x| if m.is_nan() || x.is_nan() { f64::NAN } else { m.min(x) }),
        Red::NanSum => it.filter(|x| !x.is_nan()).sum(),
        Red::NanProd => it.filter(|x| !x.is_nan()).product(),
        Red::NanMean => {
            let (s, c) = it.filter(|x| !x.is_nan()).fold((0.0, 0usize), |(s, c), x| (s + x, c + 1));
            if c == 0 { f64::NAN } else { s / c as f64 }
        }
        Red::NanCount => it.filter(|x| !x.is_nan()).count() as f64,
        Red::Count => it.filter(|x| *x != 0.0).count() as f64,
    }
}

/// Reduction over one axis (or all axes when `ax` is None). Values are folded in index order,
/// so results are reproducible bit for bit.
pub fn reduce(t: &TensorVal, ax: Option<u32>, kind: Red) -> TResult<TensorVal> {
    let Some(a) = ax else {
        let v = fold(kind, (0..t.len()).map(|i| t.getf(i)));
        return Ok(TensorVal::new_f(Vec::new(), Vec::new(), vec![v]));
    };
    let p = t.pos(a).ok_or_else(|| TErr::Shape("reduction over an axis the tensor does not have".into()))?;
    let outer: usize = t.shape[..p].iter().product();
    let n = t.shape[p];
    let inner: usize = t.shape[p + 1..].iter().product();
    let mut out = Vec::with_capacity(outer * inner);
    for o in 0..outer {
        for i in 0..inner {
            out.push(fold(kind, (0..n).map(|k| t.getf(o * n * inner + k * inner + i))));
        }
    }
    let mut axes = t.axes.clone();
    let mut shape = t.shape.clone();
    axes.remove(p);
    shape.remove(p);
    Ok(TensorVal::new_f(axes, shape, out))
}

/// (start, length, kept?) per axis for a set of selectors.
fn region(t: &TensorVal, sels: &[(u32, Sel)]) -> TResult<(Vec<usize>, Vec<usize>, Vec<bool>)> {
    let nd = t.axes.len();
    let mut start = vec![0usize; nd];
    let mut len = t.shape.clone();
    let mut keep = vec![true; nd];
    for (ax, s) in sels {
        let p = t.pos(*ax).ok_or_else(|| TErr::Shape("index over an axis the tensor does not have".into()))?;
        let n = t.shape[p];
        // bounds are checked in 64 bits before narrowing, so 32-bit systems behave the same
        match *s {
            Sel::At(i) => {
                if i < 0 || i as u64 >= n as u64 {
                    return Err(TErr::OutOfBounds { axis: *ax, index: i, size: n });
                }
                start[p] = i as usize;
                len[p] = 1;
                keep[p] = false;
            }
            Sel::Range(lo, hi) => {
                if hi < lo {
                    return Err(TErr::Reversed { axis: *ax, lo, hi });
                }
                if lo < 0 || hi as u64 > n as u64 {
                    let bad = if lo < 0 { lo } else { hi - 1 };
                    return Err(TErr::OutOfBounds { axis: *ax, index: bad, size: n });
                }
                start[p] = lo as usize;
                len[p] = (hi - lo) as usize;
            }
        }
    }
    Ok((start, len, keep))
}

/// `x[axis: i, other: a..b]`
pub fn select(t: &TensorVal, sels: &[(u32, Sel)]) -> TResult<TensorVal> {
    let (start, len, keep) = region(t, sels)?;
    let st = t.strides();
    let base: usize = start.iter().zip(&st).map(|(s, k)| s * k).sum();
    let (mut fo, mut bo, mut to) = (Vec::new(), Vec::new(), Vec::new());
    walk(&len, &[st.clone()], |offs| match &t.data {
        TData::F(v) => fo.push(v[base + offs[0]]),
        TData::B(v) => bo.push(v[base + offs[0]]),
        TData::T(v) => to.push(v.get(base + offs[0])),
    });
    let axes: Vec<u32> = t.axes.iter().zip(&keep).filter(|(_, k)| **k).map(|(a, _)| *a).collect();
    let shape: Vec<usize> = len.iter().zip(&keep).filter(|(_, k)| **k).map(|(n, _)| *n).collect();
    let data = match t.data {
        TData::F(_) => TData::F(fo),
        TData::B(_) => TData::B(bo),
        TData::T(_) => TData::T(Trits::from_slice(&to)),
    };
    Ok(TensorVal { axes, shape, data })
}

/// `x[axis: i] = v` where `v` is a scalar or a tensor over the remaining axes (any order).
pub fn assign(t: &mut TensorVal, sels: &[(u32, Sel)], v: Op) -> TResult<()> {
    let (start, len, keep) = region(t, sels)?;
    let st = t.strides();
    let base: usize = start.iter().zip(&st).map(|(s, k)| s * k).sum();
    let vs: Vec<usize> = match v {
        Op::T(vt) => {
            let vst = vt.strides();
            let mut out = Vec::with_capacity(t.axes.len());
            for (k, ax) in t.axes.iter().enumerate() {
                match (keep[k], vt.pos(*ax)) {
                    (true, Some(p)) => {
                        if vt.shape[p] != len[k] {
                            return Err(TErr::SizeMismatch { axis: *ax, a: len[k], b: vt.shape[p] });
                        }
                        out.push(vst[p]);
                    }
                    (true, None) => return Err(TErr::Shape("the assigned tensor is missing an axis".into())),
                    (false, _) => out.push(0),
                }
            }
            out
        }
        _ => vec![0; t.axes.len()],
    };
    let mut writes: Vec<(usize, usize)> = Vec::new();
    walk(&len, &[st, vs], |offs| writes.push((base + offs[0], offs[1])));
    match (&mut t.data, v) {
        (TData::F(d), Op::T(vt)) => {
            let src = vt.f();
            for (dst, s) in writes {
                d[dst] = src[s];
            }
        }
        (TData::F(d), Op::F(x)) => {
            for (dst, _) in writes {
                d[dst] = x;
            }
        }
        (TData::B(d), Op::T(vt)) => {
            let TData::B(src) = &vt.data else { return Err(TErr::Shape("element type mismatch".into())) };
            for (dst, s) in writes {
                d[dst] = src[s];
            }
        }
        (TData::B(d), Op::B(x)) => {
            for (dst, _) in writes {
                d[dst] = x;
            }
        }
        (TData::T(d), Op::T(vt)) => {
            let TData::T(src) = &vt.data else { return Err(TErr::Shape("element type mismatch".into())) };
            for (dst, s) in writes {
                d.set(dst, src.get(s));
            }
        }
        (TData::T(d), Op::Trit(x)) => {
            for (dst, _) in writes {
                d.set(dst, x);
            }
        }
        _ => return Err(TErr::Shape("element type mismatch".into())),
    }
    Ok(())
}

/// Visits every fiber along axis `p`: calls `f(offsets_of_the_fiber)`.
fn fibers(t: &TensorVal, p: usize, mut f: impl FnMut(&[usize])) {
    let outer: usize = t.shape[..p].iter().product();
    let n = t.shape[p];
    let inner: usize = t.shape[p + 1..].iter().product();
    let mut offs = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            offs.clear();
            offs.extend((0..n).map(|k| o * n * inner + k * inner + i));
            f(&offs);
        }
    }
}

/// Valid (non-NaN) entries of a fiber, stably sorted ascending: (offset, value).
fn sorted_valid(v: &[f64], offs: &[usize]) -> Vec<(usize, f64)> {
    let mut valid: Vec<(usize, f64)> = offs.iter().map(|&o| (o, v[o])).filter(|(_, x)| !x.is_nan()).collect();
    valid.sort_by(|a, b| a.1.total_cmp(&b.1));
    valid
}

/// Percentile rank in [0, 1] along `ax`, ignoring NaN (NaN stays NaN; ties keep index order).
pub fn rank(t: &TensorVal, ax: u32) -> TResult<TensorVal> {
    let p = t.pos(ax).ok_or_else(|| TErr::Shape("rank over an axis the tensor does not have".into()))?;
    let v = t.f();
    let mut out = vec![f64::NAN; v.len()];
    fibers(t, p, |offs| {
        let s = sorted_valid(v, offs);
        let m = s.len();
        for (r, (o, _)) in s.iter().enumerate() {
            out[*o] = if m == 1 { 0.5 } else { r as f64 / (m - 1) as f64 };
        }
    });
    Ok(TensorVal::new_f(t.axes.clone(), t.shape.clone(), out))
}

/// Along `ax`, the value at position floor(q·n) of the valid (non-NaN) values sorted ascending;
/// NaN when there are none. q = 0.5 gives the middle value (the upper middle for an even count).
pub fn nan_quantile(t: &TensorVal, q: f64, ax: u32) -> TResult<TensorVal> {
    let p = t.pos(ax).ok_or_else(|| TErr::Shape("nan_quantile over an axis the tensor does not have".into()))?;
    let v = t.f();
    let outer: usize = t.shape[..p].iter().product();
    let inner: usize = t.shape[p + 1..].iter().product();
    let mut out = vec![f64::NAN; outer * inner];
    let mut k = 0;
    fibers(t, p, |offs| {
        let s = sorted_valid(v, offs);
        if !s.is_empty() {
            let i = ((q.clamp(0.0, 1.0) * s.len() as f64).floor() as usize).min(s.len() - 1);
            out[k] = s[i].1;
        }
        k += 1;
    });
    let mut axes = t.axes.clone();
    let mut shape = t.shape.clone();
    axes.remove(p);
    shape.remove(p);
    Ok(TensorVal::new_f(axes, shape, out))
}

/// `[.., month, .., session, ..]` as `[day, ..]` (day = month · sessions + session), the other
/// axes in their order.
pub fn merge_days(t: &TensorVal, month: u32, session: u32, day: u32) -> TensorVal {
    let pm = t.pos(month).expect("month axis");
    let ps = t.pos(session).expect("session axis");
    let rest: Vec<usize> = (0..t.axes.len()).filter(|&k| k != pm && k != ps).collect();
    let mut order = vec![pm, ps];
    order.extend(&rest);
    let mut axes = vec![day];
    axes.extend(rest.iter().map(|&k| t.axes[k]));
    let mut shape = vec![t.shape[pm] * t.shape[ps]];
    shape.extend(rest.iter().map(|&k| t.shape[k]));
    // already in that layout: the same data
    if order.iter().enumerate().all(|(i, &k)| i == k) {
        return TensorVal { axes, shape, data: t.data.clone() };
    }
    let st = t.strides();
    let dims: Vec<usize> = order.iter().map(|&k| t.shape[k]).collect();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut offs = Vec::with_capacity(total);
    for _ in 0..total {
        offs.push(idx.iter().zip(&order).map(|(i, &k)| i * st[k]).sum::<usize>());
        for d in (0..dims.len()).rev() {
            idx[d] += 1;
            if idx[d] < dims[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    let data = match &t.data {
        TData::F(v) => TData::F(offs.iter().map(|&o| v[o]).collect()),
        TData::B(v) => TData::B(offs.iter().map(|&o| v[o]).collect()),
        TData::T(v) => TData::T(Trits::from_slice(&offs.iter().map(|&o| v.get(o)).collect::<Vec<_>>())),
    };
    TensorVal { axes, shape, data }
}

/// Mask of the `k` largest (or smallest) valid values along `ax`. Same tie rule as a stable
/// ascending sort taking the last (first) k, i.e. exactly what the quant-b3 portfolios do.
pub fn top_k(t: &TensorVal, ax: u32, k: i64, top: bool) -> TResult<TensorVal> {
    let p = t.pos(ax).ok_or_else(|| TErr::Shape("top_k over an axis the tensor does not have".into()))?;
    let v = t.f();
    let mut out = vec![false; v.len()];
    fibers(t, p, |offs| {
        let s = sorted_valid(v, offs);
        let k = usize::try_from(k.max(0)).unwrap_or(usize::MAX).min(s.len());
        let chosen = if top { &s[s.len() - k..] } else { &s[..k] };
        for (o, _) in chosen {
            out[*o] = true;
        }
    });
    Ok(TensorVal::new_b(t.axes.clone(), t.shape.clone(), out))
}

/// Sum over `ax` of x * y; other shared axes are batch axes, the rest form an outer product.
pub fn contract(x: &TensorVal, y: &TensorVal, ax: u32) -> TResult<TensorVal> {
    let px = x.pos(ax).ok_or_else(|| TErr::Shape("contraction axis missing on the left".into()))?;
    let py = y.pos(ax).ok_or_else(|| TErr::Shape("contraction axis missing on the right".into()))?;
    let n = x.shape[px];
    if y.shape[py] != n {
        return Err(TErr::SizeMismatch { axis: ax, a: n, b: y.shape[py] });
    }
    let mut axes = Vec::new();
    let mut shape = Vec::new();
    for (k, a) in x.axes.iter().enumerate() {
        if *a != ax {
            axes.push(*a);
            shape.push(x.shape[k]);
        }
    }
    for (k, a) in y.axes.iter().enumerate() {
        if *a == ax {
            continue;
        }
        match x.pos(*a) {
            Some(p) if x.shape[p] != y.shape[k] => return Err(TErr::SizeMismatch { axis: *a, a: x.shape[p], b: y.shape[k] }),
            Some(_) => {}
            None => {
                axes.push(*a);
                shape.push(y.shape[k]);
            }
        }
    }
    let (sx, sy) = (mapped(x, &axes), mapped(y, &axes));
    let (kx, ky) = (x.strides()[px], y.strides()[py]);
    let mut out = Vec::with_capacity(shape.iter().product());
    // M18: with a `t3` side there is no product: the x where w is +1 are added, those where it is
    // -1 subtracted, in the axis' order
    let ternary = |xf: &[f64], w: &Trits, ox: usize, ow: usize, kx: usize, kw: usize| {
        let mut acc = 0.0;
        for k in 0..n {
            match w.get(ow + k * kw) {
                1 => acc += xf[ox + k * kx],
                -1 => acc -= xf[ox + k * kx],
                _ => {}
            }
        }
        acc
    };
    match (&x.data, &y.data) {
        (TData::F(xf), TData::T(w)) => walk(&shape, &[sx, sy], |offs| out.push(ternary(xf, w, offs[0], offs[1], kx, ky))),
        (TData::T(w), TData::F(yf)) => walk(&shape, &[sx, sy], |offs| out.push(ternary(yf, w, offs[1], offs[0], ky, kx))),
        _ => {
            let (xf, yf) = (x.f(), y.f());
            walk(&shape, &[sx, sy], |offs| {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += xf[offs[0] + k * kx] * yf[offs[1] + k * ky];
                }
                out.push(acc);
            });
        }
    }
    Ok(TensorVal::new_f(axes, shape, out))
}

/// Sum over every axis in `over` of x * y: the general named-axis contraction (`contract` is the
/// one-axis case, and the gradients of a contraction are contractions too). Axes shared by x and
/// y that are not in `over` are batch axes; the others form an outer product.
pub fn contract_over(x: &TensorVal, y: &TensorVal, over: &[u32]) -> TResult<TensorVal> {
    let mut axes: Vec<u32> = Vec::new();
    let mut shape = Vec::new();
    for (k, a) in x.axes.iter().enumerate() {
        if !over.contains(a) {
            axes.push(*a);
            shape.push(x.shape[k]);
        }
    }
    for (k, a) in y.axes.iter().enumerate() {
        match x.pos(*a) {
            Some(p) if x.shape[p] != y.shape[k] => return Err(TErr::SizeMismatch { axis: *a, a: x.shape[p], b: y.shape[k] }),
            Some(_) => {}
            None if !over.contains(a) => {
                axes.push(*a);
                shape.push(y.shape[k]);
            }
            None => {}
        }
    }
    let mut full_axes = axes.clone();
    let mut full_shape = shape.clone();
    for a in over {
        let n = x.size(*a).or(y.size(*a)).ok_or_else(|| TErr::Shape("contraction over an axis neither tensor has".into()))?;
        full_axes.push(*a);
        full_shape.push(n);
    }
    let inner: usize = full_shape[axes.len()..].iter().product();
    let n_out: usize = shape.iter().product();
    if inner == 0 {
        return Ok(TensorVal::new_f(axes, shape, vec![0.0; n_out]));
    }
    let (sx, sy) = (mapped(x, &full_axes), mapped(y, &full_axes));
    let (xf, yf) = (x.f(), y.f());
    let mut out = Vec::with_capacity(n_out);
    let (mut acc, mut k) = (0.0, 0usize);
    walk(&full_shape, &[sx, sy], |offs| {
        acc += xf[offs[0]] * yf[offs[1]];
        k += 1;
        if k == inner {
            out.push(acc);
            acc = 0.0;
            k = 0;
        }
    });
    Ok(TensorVal::new_f(axes, shape, out))
}

/// Gradient of a product over `ax` (None: all elements): for each element, the product of the
/// others times the upstream gradient `g` (already in the shape and axis order of `a`). Prefix
/// and suffix products, so zeros are handled exactly.
pub fn prod_grad(a: &TensorVal, ax: Option<u32>, g: &TensorVal) -> TResult<TensorVal> {
    let (av, gv) = (a.f(), g.f());
    let mut out = vec![0.0; av.len()];
    let mut one = |offs: &[usize]| {
        let n = offs.len();
        let mut prefix = vec![1.0; n + 1];
        for k in 0..n {
            prefix[k + 1] = prefix[k] * av[offs[k]];
        }
        let mut suffix = 1.0;
        for k in (0..n).rev() {
            out[offs[k]] = prefix[k] * suffix * gv[offs[k]];
            suffix *= av[offs[k]];
        }
    };
    match ax {
        None => one(&(0..av.len()).collect::<Vec<_>>()),
        Some(axis) => {
            let p = a.pos(axis).ok_or_else(|| TErr::Shape("product over an axis the tensor does not have".into()))?;
            fibers(a, p, one);
        }
    }
    Ok(TensorVal::new_f(a.axes.clone(), a.shape.clone(), out))
}

/// Sums `t` over every axis not in `keep` (undoing a broadcast).
pub fn sum_to(t: &TensorVal, keep: &[u32]) -> TResult<TensorVal> {
    let mut cur = t.clone();
    let extra: Vec<u32> = t.axes.iter().copied().filter(|a| !keep.contains(a)).collect();
    for a in extra {
        cur = reduce(&cur, Some(a), Red::Sum)?;
    }
    Ok(cur)
}

/// Deterministic normal samples: SplitMix64 seeds a xoshiro256** stream, Box-Muller turns pairs
/// of uniforms into normals. Integer arithmetic plus `libm`, so every system draws the same bits.
pub fn randn(seed: u64, axes: Vec<u32>, shape: Vec<usize>) -> TensorVal {
    let mut sm = seed;
    let mut next_sm = || {
        sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = sm;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    let mut s = [next_sm(), next_sm(), next_sm(), next_sm()];
    let mut next = || {
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    };
    // uniform in (0, 1]: 53 random bits
    let mut unif = || ((next() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n + 1);
    while v.len() < n {
        let (u1, u2) = (unif(), unif());
        let r = (-2.0 * libm::log(u1)).sqrt();
        let th = 2.0 * std::f64::consts::PI * u2;
        v.push(r * libm::cos(th));
        v.push(r * libm::sin(th));
    }
    v.truncate(n);
    TensorVal::new_f(axes, shape, v)
}

/// `ternary(x)` (M18): BitNet b1.58's quantization. With g the mean of |x| over the elements that
/// are not NaN, each one becomes round(x / g) (the half away from zero) limited to [-1, 1]; NaN
/// gives 0, and so does everything when g is 0 or NaN. With a threshold th: +1 above th, -1 below
/// -th, 0 otherwise (NaN included).
pub fn ternary(t: &TensorVal, threshold: Option<f64>) -> TensorVal {
    let v = t.f();
    let out: Vec<i8> = match threshold {
        Some(th) => v
            .iter()
            .map(|&x| {
                if x > th {
                    1
                } else if x < -th {
                    -1
                } else {
                    0
                }
            })
            .collect(),
        None => {
            let (mut s, mut c) = (0.0, 0usize);
            for &x in v {
                if !x.is_nan() {
                    s += x.abs();
                    c += 1;
                }
            }
            let g = if c == 0 { f64::NAN } else { s / c as f64 };
            v.iter()
                .map(|&x| {
                    let r = libm::round(x / g);
                    if r >= 1.0 {
                        1
                    } else if r <= -1.0 {
                        -1
                    } else {
                        0
                    }
                })
                .collect()
        }
    };
    TensorVal::new_t(t.axes.clone(), t.shape.clone(), &out)
}

/// `-w` of a `t3` tensor
pub fn neg_t3(t: &TensorVal) -> TensorVal {
    let TData::T(v) = &t.data else { unreachable!("neg_t3 of a tensor that is not t3") };
    let out: Vec<i8> = v.to_vec().iter().map(|x| -x).collect();
    TensorVal::new_t(t.axes.clone(), t.shape.clone(), &out)
}

/// `float(w)` of a `t3` tensor: -1.0, 0.0, 1.0
pub fn t3_to_f(t: &TensorVal) -> TensorVal {
    let TData::T(v) = &t.data else { unreachable!("float of a tensor that is not t3") };
    TensorVal::new_f(t.axes.clone(), t.shape.clone(), v.to_vec().iter().map(|x| f64::from(*x)).collect())
}

pub fn full(axes: Vec<u32>, shape: Vec<usize>, v: f64) -> TensorVal {
    let n = shape.iter().product();
    TensorVal::new_f(axes, shape, vec![v; n])
}

/// Text form: small one-axis tensors show their values.
pub fn describe(t: &TensorVal, name: impl Fn(u32) -> String) -> String {
    let dims = t.axes.iter().zip(&t.shape).map(|(a, n)| format!("{}: {n}", name(*a))).collect::<Vec<_>>().join(", ");
    let elem = match t.data {
        TData::F(_) => "f64",
        TData::B(_) => "bool",
        TData::T(_) => "t3",
    };
    if t.axes.len() == 1 && t.len() <= 20 {
        let vals = match &t.data {
            TData::F(v) => v.iter().map(|x| format!("{x:?}")).collect::<Vec<_>>().join(", "),
            TData::B(v) => v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
            TData::T(v) => v.to_vec().iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
        };
        format!("Tensor[{dims}; {elem}] [{vals}]")
    } else {
        format!("Tensor[{dims}; {elem}]")
    }
}

/// Total order of f64 used for sorting.
pub fn cmp_f(a: f64, b: f64) -> Ordering {
    a.total_cmp(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M18: five trits to a byte, read and written in place; equal trits, equal bytes
    #[test]
    fn trits_take_a_fifth_of_a_byte() {
        let v: Vec<i8> = (0..23).map(|i| (i % 3) as i8 - 1).collect();
        let mut t = Trits::from_slice(&v);
        assert_eq!((t.len(), t.packed_len()), (23, 5));
        assert_eq!(t.to_vec(), v);
        t.set(7, 1);
        t.set(22, -1);
        let mut w = v.clone();
        w[7] = 1;
        w[22] = -1;
        assert_eq!(t.to_vec(), w);
        assert_eq!(t, Trits::from_slice(&w));
    }
}
