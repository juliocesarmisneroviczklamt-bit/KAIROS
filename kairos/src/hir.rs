//! HIR: the typed, name-resolved tree produced by the checker. Variables are frame slots, calls
//! are resolved to user functions, builtins or function values, and every assignment target is
//! an explicit place. Later passes (time checks, autodiff, native codegen) work on this.

use crate::ast::{BinOp, UnOp};
use crate::diag::Span;
use crate::value::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Builtin {
    Print,
    Len,
    Push,
    Pop,
    Sort,
    Reverse,
    Contains,
    Map,
    Filter,
    Sum,
    Min,
    Max,
    Minimum,
    Maximum,
    Abs,
    Sqrt,
    Exp,
    Ln,
    Floor,
    Ceil,
    Round,
    ToFloat,
    ToInt,
    ToStr,
    Assert,
    // ---- M2: reductions (lists and tensors)
    Mean,
    Prod,
    NanSum,
    NanMean,
    NanProd,
    NanCount,
    Count,
    // ---- M2: element-wise
    IsNan,
    FillNan,
    Where,
    // ---- M2: shape, construction, ranking, contraction
    Size,
    Zeros,
    Ones,
    Full,
    ZerosLike,
    FullLike,
    Tensor,
    ToList,
    Rank,
    TopK,
    BottomK,
    Contract,
    /// `observed(x, lag: k)`: row t of x (over its temporal axis) is known at end(t + k); identity at run time
    Observed,
    // ---- M4: differentiation and the AI engine
    Grad,
    ValueAndGrad,
    Tanh,
    Sigmoid,
    Relu,
    Softplus,
    Randn,
    SmoothMax,
    SmoothMin,
    SmoothStep,
    /// element-wise `and` / `or` of bool tensors (the scalar forms short-circuit and are HIR nodes)
    TAnd,
    TOr,
}

pub const BUILTINS: &[(&str, Builtin)] = &[
    ("print", Builtin::Print),
    ("len", Builtin::Len),
    ("push", Builtin::Push),
    ("pop", Builtin::Pop),
    ("sort", Builtin::Sort),
    ("reverse", Builtin::Reverse),
    ("contains", Builtin::Contains),
    ("map", Builtin::Map),
    ("filter", Builtin::Filter),
    ("sum", Builtin::Sum),
    ("min", Builtin::Min),
    ("max", Builtin::Max),
    ("minimum", Builtin::Minimum),
    ("maximum", Builtin::Maximum),
    ("abs", Builtin::Abs),
    ("sqrt", Builtin::Sqrt),
    ("exp", Builtin::Exp),
    ("ln", Builtin::Ln),
    ("floor", Builtin::Floor),
    ("ceil", Builtin::Ceil),
    ("round", Builtin::Round),
    ("float", Builtin::ToFloat),
    ("int", Builtin::ToInt),
    ("str", Builtin::ToStr),
    ("assert", Builtin::Assert),
    ("mean", Builtin::Mean),
    ("prod", Builtin::Prod),
    ("nan_sum", Builtin::NanSum),
    ("nan_mean", Builtin::NanMean),
    ("nan_prod", Builtin::NanProd),
    ("nan_count", Builtin::NanCount),
    ("count", Builtin::Count),
    ("is_nan", Builtin::IsNan),
    ("fill_nan", Builtin::FillNan),
    ("where", Builtin::Where),
    ("size", Builtin::Size),
    ("zeros", Builtin::Zeros),
    ("ones", Builtin::Ones),
    ("full", Builtin::Full),
    ("zeros_like", Builtin::ZerosLike),
    ("full_like", Builtin::FullLike),
    ("tensor", Builtin::Tensor),
    ("to_list", Builtin::ToList),
    ("rank", Builtin::Rank),
    ("top_k", Builtin::TopK),
    ("bottom_k", Builtin::BottomK),
    ("contract", Builtin::Contract),
    ("observed", Builtin::Observed),
    ("grad", Builtin::Grad),
    ("value_and_grad", Builtin::ValueAndGrad),
    ("tanh", Builtin::Tanh),
    ("sigmoid", Builtin::Sigmoid),
    ("relu", Builtin::Relu),
    ("softplus", Builtin::Softplus),
    ("randn", Builtin::Randn),
    ("smooth_max", Builtin::SmoothMax),
    ("smooth_min", Builtin::SmoothMin),
    ("smooth_step", Builtin::SmoothStep),
    // keywords, so never reachable by name: listed only for messages
    ("and", Builtin::TAnd),
    ("or", Builtin::TOr),
];

/// Functions of the `b3` data module (the bridge to the quant-b3 lab).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostFn {
    MonthlyReturns,
    Eligible,
    Cdi,
    Months,
    Tickers,
    MonthIndex,
    /// args: weights, cost_bps, lag, capital, real_costs, tax; returns the `Evaluation` struct
    Evaluate { result: u32 },
}

/// Selector of one tensor axis: `axis: i` (removes the axis) or `axis: a .. b` / `a ..= b`.
#[derive(Clone, Debug)]
pub enum HSel {
    At(HExpr),
    Range(HExpr, HExpr, bool),
}

pub fn builtin_by_name(n: &str) -> Option<Builtin> {
    BUILTINS.iter().find(|(s, _)| *s == n).map(|(_, b)| *b)
}

pub fn builtin_name(b: Builtin) -> &'static str {
    BUILTINS.iter().find(|(_, x)| *x == b).map(|(s, _)| *s).unwrap_or("?")
}

#[derive(Clone, Debug)]
pub enum HExpr {
    Lit(Value),
    /// integer literal whose final type (`i64` or `f64`) is known only after inference
    IntLit(i64, u32),
    Local(u32),
    FnRef(u32),
    List(Vec<HExpr>),
    Tuple(Vec<HExpr>),
    Struct(u32, Vec<HExpr>),
    Variant(u32, u32, Vec<HExpr>),
    /// struct field or tuple element
    Field(Box<HExpr>, u32),
    Index(Box<HExpr>, Box<HExpr>, Span),
    /// `x[month: i, asset: a .. b]` (axis ids)
    TensorIndex(Box<HExpr>, Vec<(u32, HSel)>, Span),
    /// `x[month: i] = v` / `x[month: i] += v`
    TensorAssign(HPlace, Vec<(u32, HSel)>, Option<BinOp>, Box<HExpr>, Span),
    /// call into the `b3` data module
    Host(HostFn, Vec<HExpr>, Span),
    CallFn(u32, Vec<HArg>, Span),
    CallValue(Box<HExpr>, Vec<HExpr>, Span),
    CallBuiltin(Builtin, Vec<HArg>, Span),
    Unary(UnOp, Box<HExpr>, Span),
    Binary(BinOp, Box<HExpr>, Box<HExpr>, Span),
    And(Box<HExpr>, Box<HExpr>),
    Or(Box<HExpr>, Box<HExpr>),
    Assign(HPlace, Box<HExpr>),
    OpAssign(HPlace, BinOp, Box<HExpr>, Span),
    If(Box<HExpr>, Box<HExpr>, Option<Box<HExpr>>),
    While(Box<HExpr>, Box<HExpr>),
    /// `name` is kept for the messages of the temporal check
    ForRange { slot: u32, name: String, lo: Box<HExpr>, hi: Box<HExpr>, inclusive: bool, body: Box<HExpr> },
    ForList { pat: HPat, list: Box<HExpr>, body: Box<HExpr> },
    Match(Box<HExpr>, Vec<(HPat, HExpr)>, Span),
    Block(Vec<HExpr>, Option<Box<HExpr>>),
    Let(HPat, Box<HExpr>),
    /// closure creation: lambda definition + outer slots copied into it (capture by value)
    Lambda(u32, Vec<u32>),
    Return(Option<Box<HExpr>>),
    Break,
    Continue,
}

#[derive(Clone, Debug)]
pub enum HPlace {
    Local(u32),
    Field(Box<HPlace>, u32),
    Index(Box<HPlace>, Box<HExpr>, Span),
}

#[derive(Clone, Debug)]
pub enum HArg {
    Val(HExpr),
    InOut(HPlace),
}

#[derive(Clone, Debug)]
pub enum HPat {
    Wild,
    Bind(u32),
    Lit(Value),
    IntLit(i64, u32),
    Tuple(Vec<HPat>),
    Variant(u32, u32, Vec<HPat>),
}

#[derive(Clone, Debug)]
pub struct FnBody {
    pub n_slots: u32,
    pub body: HExpr,
    /// `-> T @ end(e)`: the month index `e` (over the parameters) by which the result is known
    pub known_at: Option<(HExpr, Span)>,
}

#[derive(Clone, Debug)]
pub struct LambdaBody {
    pub capture_slots: Vec<u32>,
    pub n_params: u32,
    pub n_slots: u32,
    pub body: HExpr,
}
