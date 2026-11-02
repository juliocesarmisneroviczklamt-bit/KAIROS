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
    /// `nan_quantile(x, q, over: axis)`: the value at position floor(q·n) of the valid values
    /// sorted (q = 0.5: the middle one, the upper of the two for an even count)
    NanQuantile,
    /// `merge_days(x)`: `Tensor[month, session, ...]` as `Tensor[day, ...]`
    MergeDays,
    /// `format_fixed(x, d)`: `x` with `d` decimals, rounded exactly (as Rust's `{:.d}`)
    FormatFixed,
    /// `substr(s, start, count)`: characters `start .. start + count` of `s` (clamped)
    Substr,
    // ---- M9: text
    /// `split(s, sep)`: the pieces of `s` between the separators
    Split,
    /// `lines(s)`: the lines of `s` (without `\n` or `\r\n`)
    Lines,
    Trim,
    StartsWith,
    EndsWith,
    /// `find(s, sub)`: the character index of the first `sub` in `s`, or -1
    Find,
    Replace,
    Upper,
    Lower,
    /// `parse_int(s)`: the integer written in `s` (spaces around are allowed); R0010 if it is not one
    ParseInt,
    /// `parse_float(s)`: the number written in `s`; NaN if it is not one (missing data)
    ParseFloat,
    /// `join(xs, sep)`: the strings of `xs` with `sep` between them
    Join,
    /// `codes(s)`: the Unicode code points of `s`; `from_codes(xs)`: back to a string
    Codes,
    FromCodes,
    // ---- M9: files and the environment
    /// `read_text(path)`: the whole file (UTF-8; a file that is not UTF-8 is read as Latin-1,
    /// as B3's files are)
    ReadText,
    WriteText,
    AppendText,
    FileExists,
    /// `list_dir(path)`: the names in a folder, sorted
    ListDir,
    /// `make_dir(path)`: creates the folder and its parents
    MakeDir,
    /// `env(name)`: an environment variable ("" if unset)
    Env,
    /// `args()`: the program's arguments (`kairos run file.krs a b` gives ["a", "b"])
    Args,
    /// `today()`: today's date (UTC) as YYYYMMDD
    Today,
    /// `http_get(url)`: the body of a GET (text: UTF-8, or Latin-1 when it is not UTF-8)
    HttpGet,
    /// `download(url, path)`: saves the body of a GET in a file; the number of bytes
    Download,
    // ---- M10: data files
    /// `read_lines(path, keep)`: the lines of a text file (or of the first file inside a .zip)
    /// that have, for every `(column, text)` of `keep`, `text` at that column (in bytes)
    ReadLines,
    /// `bytes()`: empty binary data; `put_u8/u32/u64/i64/f64/str(inout b, x)` append to it
    /// (little-endian; `put_str` writes the UTF-8 bytes, `utf8_len(s)` counts them)
    BytesNew,
    PutU8,
    PutU32,
    PutU64,
    PutI64,
    PutF64,
    /// `put_f32(inout b, x)`: `x` rounded to the nearest f32 (as Rust's `x as f32`)
    PutF32,
    PutStr,
    Utf8Len,
    /// `get_u8/u32/u64/i64/f64(b, at)`, `get_str(b, at, n)`: read at a byte offset
    GetU8,
    GetU32,
    GetU64,
    GetI64,
    GetF64,
    GetF32,
    GetStr,
    WriteBytes,
    ReadBytes,
    /// bits of an i64 (two's complement): `bit_and/or/xor(a, b)`, `shl(a, n)`, `shr(a, n)`
    /// (logical), and arithmetic that wraps around instead of failing: `wrapping_add/mul`
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    WrappingAdd,
    WrappingMul,
    /// `to_hex(x, width)`: lowercase hex of the 64 bits, zero-padded to `width`
    ToHex,
    /// `file_size(p)` (-1 when missing), `file_date(p)`: the day it was last written (UTC,
    /// YYYYMMDD; 0 when missing)
    FileSize,
    FileDate,
    /// `http_get_all(urls, parallel)`: GETs, `parallel` at a time, in order:
    /// `(true, body)` or `(false, what went wrong)`
    HttpGetAll,
    /// Unicode classes of a code point (as Rust's `char`): `is_alphabetic(c)`,
    /// `is_alphanumeric(c)`, `is_whitespace(c)`
    IsAlphabetic,
    IsAlphanumeric,
    IsWhitespace,
    IsUppercase,
    // ---- M9: maps
    /// `keys(m)`, `values(m)`, `items(m)`: in insertion order
    Keys,
    Values,
    Items,
    /// `get(m, k, default)`
    Get,
    /// `remove(inout m, k)`: removes the key if it is there
    Remove,
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
    // ---- M5
    OpenOnce,
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
    ("format_fixed", Builtin::FormatFixed),
    ("substr", Builtin::Substr),
    ("split", Builtin::Split),
    ("lines", Builtin::Lines),
    ("trim", Builtin::Trim),
    ("starts_with", Builtin::StartsWith),
    ("ends_with", Builtin::EndsWith),
    ("find", Builtin::Find),
    ("replace", Builtin::Replace),
    ("upper", Builtin::Upper),
    ("lower", Builtin::Lower),
    ("parse_int", Builtin::ParseInt),
    ("parse_float", Builtin::ParseFloat),
    ("join", Builtin::Join),
    ("codes", Builtin::Codes),
    ("from_codes", Builtin::FromCodes),
    ("read_text", Builtin::ReadText),
    ("write_text", Builtin::WriteText),
    ("append_text", Builtin::AppendText),
    ("file_exists", Builtin::FileExists),
    ("list_dir", Builtin::ListDir),
    ("make_dir", Builtin::MakeDir),
    ("env", Builtin::Env),
    ("args", Builtin::Args),
    ("today", Builtin::Today),
    ("http_get", Builtin::HttpGet),
    ("download", Builtin::Download),
    ("read_lines", Builtin::ReadLines),
    ("bytes", Builtin::BytesNew),
    ("put_u8", Builtin::PutU8),
    ("put_u32", Builtin::PutU32),
    ("put_u64", Builtin::PutU64),
    ("put_i64", Builtin::PutI64),
    ("put_f64", Builtin::PutF64),
    ("put_f32", Builtin::PutF32),
    ("put_str", Builtin::PutStr),
    ("utf8_len", Builtin::Utf8Len),
    ("get_u8", Builtin::GetU8),
    ("get_u32", Builtin::GetU32),
    ("get_u64", Builtin::GetU64),
    ("get_i64", Builtin::GetI64),
    ("get_f64", Builtin::GetF64),
    ("get_f32", Builtin::GetF32),
    ("get_str", Builtin::GetStr),
    ("write_bytes", Builtin::WriteBytes),
    ("read_bytes", Builtin::ReadBytes),
    ("bit_and", Builtin::BitAnd),
    ("bit_or", Builtin::BitOr),
    ("bit_xor", Builtin::BitXor),
    ("shl", Builtin::Shl),
    ("shr", Builtin::Shr),
    ("wrapping_add", Builtin::WrappingAdd),
    ("wrapping_mul", Builtin::WrappingMul),
    ("to_hex", Builtin::ToHex),
    ("file_size", Builtin::FileSize),
    ("file_date", Builtin::FileDate),
    ("http_get_all", Builtin::HttpGetAll),
    ("is_alphabetic", Builtin::IsAlphabetic),
    ("is_alphanumeric", Builtin::IsAlphanumeric),
    ("is_whitespace", Builtin::IsWhitespace),
    ("is_uppercase", Builtin::IsUppercase),
    ("keys", Builtin::Keys),
    ("values", Builtin::Values),
    ("items", Builtin::Items),
    ("get", Builtin::Get),
    ("remove", Builtin::Remove),
    ("nan_quantile", Builtin::NanQuantile),
    ("merge_days", Builtin::MergeDays),
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
    ("open_once", Builtin::OpenOnce),
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
    /// monthly inflation (IPCA)
    Ipca,
    Months,
    Tickers,
    MonthIndex,
    /// args: weights, cost_bps, lag, capital, real_costs, tax, research_until (YYYYMM, 0: the
    /// whole history), protocol name; returns the `Evaluation` struct
    Evaluate { result: u32 },
    /// `b3.holdout("name")`: the holdout of a protocol (M5)
    Holdout { proto: u32 },
    /// `open_once(h, weights, ...)`: args as `Evaluate`, then the protocol: name, research_until,
    /// holdout_from, rule
    OpenOnce { result: u32 },
    /// M8, daily data as `Tensor[month, session, asset]`: 0 returns, 1 traded volume (R$),
    /// 2 unadjusted close price, 3 dividends + JCP (fraction of the price)
    Daily { field: u8 },
    /// `b3.results_sha256("file")`: SHA-256 of a file in the results folder ("" if missing)
    ResultsSha256,
    /// `b3.deflate_trial(excess)`: [trials in the registry, Sharpe chance gives (a year), DSR],
    /// without recording anything
    DeflateTrial,
    /// `Tensor[month, session; bool]`: the sessions that exist
    Sessions,
    /// company and class of each asset
    Issuers,
    Classes,
    /// a NEFIN factor (0 rm_rf, 1 smb, 2 hml, 3 wml, 4 iml, 5 rf); args: value-weighted?
    Nefin { factor: u8 },
    /// the same, daily: `Tensor[month, session; f64]`
    NefinDaily { factor: u8 },
    /// `b3.simulate(w, ...)`: args as `Evaluate`; returns the `Simulation` struct
    Simulate { result: u32 },
    /// `b3.record_trial(study, name, params, period, excess)`: the lab's registry; returns the
    /// line that describes the trial
    RecordTrial,
    /// `Tensor[month, session; f64]`: the date of each session (YYYYMMDD)
    SessionDates,
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
    /// map literal: keys and values interleaved (`k0, v0, k1, v1, ...`)
    MapLit(Vec<HExpr>),
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
    /// the type of every slot (parameters first), as far as the function determines it
    pub slot_types: Vec<crate::types::Ty>,
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
