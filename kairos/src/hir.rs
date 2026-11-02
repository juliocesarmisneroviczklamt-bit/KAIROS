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
    /// M13: the IEEE 754 bits of a float and back (`to_bits(x)`, `from_bits(b)`), as Rust's
    /// `f64::to_bits` and `f64::from_bits`
    ToBits,
    FromBits,
    /// M15: `eprint(...)` writes as `print` does, to the standard error; `exit(code)` ends the
    /// program with that exit code (the standard output written first)
    Eprint,
    Exit,
    /// M15: the file and network builtins that give their error instead of stopping the program:
    /// `try_read_text(p) -> (str, str)` (the text and the error, "" when there is none), and so on
    TryReadText,
    TryReadBytes,
    TryWriteText,
    TryWriteBytes,
    TryListDir,
    TryMakeDir,
    TryReadLines,
    /// M19: the names of the files in a ZIP archive, and the lines of one of them (as read_lines)
    ZipEntries,
    ReadZipLines,
    TryZipEntries,
    TryReadZipLines,
    TryHttpGet,
    TryDownload,
    /// M16: `read_line()`, a line of the standard input with its line break ("" at its end), and
    /// `print_raw(s)`, a text to the standard output without a line break (the REPL's prompt)
    ReadLine,
    PrintRaw,
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
    /// `ternary(x)`, `ternary(x, threshold: t)` (M18): a `f64` tensor to a `t3` one
    Ternary,
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
    ("eprint", Builtin::Eprint),
    ("exit", Builtin::Exit),
    ("try_read_text", Builtin::TryReadText),
    ("try_read_bytes", Builtin::TryReadBytes),
    ("try_write_text", Builtin::TryWriteText),
    ("try_write_bytes", Builtin::TryWriteBytes),
    ("try_list_dir", Builtin::TryListDir),
    ("try_make_dir", Builtin::TryMakeDir),
    ("try_read_lines", Builtin::TryReadLines),
    ("zip_entries", Builtin::ZipEntries),
    ("read_zip_lines", Builtin::ReadZipLines),
    ("try_zip_entries", Builtin::TryZipEntries),
    ("try_read_zip_lines", Builtin::TryReadZipLines),
    ("try_http_get", Builtin::TryHttpGet),
    ("try_download", Builtin::TryDownload),
    ("read_line", Builtin::ReadLine),
    ("print_raw", Builtin::PrintRaw),
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
    ("to_bits", Builtin::ToBits),
    ("from_bits", Builtin::FromBits),
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
    ("ternary", Builtin::Ternary),
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

// ---------------------------------------------------------------------- the canonical text (M12)

/// A HIR node as uniform data, `(kind text #n #m children... lo:hi)`: the text M12 compares with
/// the HIR the checker written in Kairós (kairos/compiler/body.krs) builds. `n` and `m` are
/// slots, ids and indices (-1: none); the kinds follow `HExpr`, `HPlace`, `HArg`, `HSel`, `HPat`.
pub struct HNode {
    kind: &'static str,
    text: String,
    n: i64,
    m: i64,
    kids: Vec<HNode>,
    span: Option<Span>,
}

fn hn(kind: &'static str, text: String, n: i64, m: i64, kids: Vec<HNode>, span: Option<Span>) -> HNode {
    HNode { kind, text, n, m, kids, span }
}

fn hk(kind: &'static str, kids: Vec<HNode>) -> HNode {
    hn(kind, String::new(), -1, -1, kids, None)
}

fn none_node() -> HNode {
    hk("-", Vec::new())
}

impl HNode {
    pub fn render(&self, out: &mut String) {
        if self.kind == "-" {
            out.push('-');
            return;
        }
        out.push('(');
        out.push_str(self.kind);
        if !self.text.is_empty() {
            out.push(' ');
            out.push_str(&self.text);
        }
        for x in [self.n, self.m] {
            if x >= 0 {
                out.push_str(&format!(" #{x}"));
            }
        }
        for k in &self.kids {
            out.push(' ');
            k.render(out);
        }
        if let Some(sp) = self.span {
            out.push_str(&format!(" {}:{}", sp.lo, sp.hi));
        }
        out.push(')');
    }
}

pub fn value_text(v: &Value) -> String {
    match v {
        Value::Unit => "unit".into(),
        Value::Int(i) => format!("i64 {i}"),
        Value::Float(x) => format!("f64 {x:?}"),
        Value::Bool(b) => format!("bool {b}"),
        Value::Str(s) => format!("str {}", crate::ast::hex_text(s)),
        _ => "?".into(),
    }
}

fn host_text(f: &HostFn) -> String {
    match f {
        HostFn::MonthlyReturns => "monthly_returns".into(),
        HostFn::Eligible => "eligible".into(),
        HostFn::Cdi => "cdi".into(),
        HostFn::Ipca => "ipca".into(),
        HostFn::Months => "months".into(),
        HostFn::Tickers => "tickers".into(),
        HostFn::MonthIndex => "month_index".into(),
        HostFn::Evaluate { result } => format!("evaluate {result}"),
        HostFn::Holdout { proto } => format!("holdout {proto}"),
        HostFn::OpenOnce { result } => format!("open_once {result}"),
        HostFn::Daily { field } => format!("daily {field}"),
        HostFn::ResultsSha256 => "results_sha256".into(),
        HostFn::DeflateTrial => "deflate_trial".into(),
        HostFn::Sessions => "sessions".into(),
        HostFn::Issuers => "issuers".into(),
        HostFn::Classes => "classes".into(),
        HostFn::Nefin { factor } => format!("nefin {factor}"),
        HostFn::NefinDaily { factor } => format!("nefin_daily {factor}"),
        HostFn::Simulate { result } => format!("simulate {result}"),
        HostFn::RecordTrial => "record_trial".into(),
        HostFn::SessionDates => "session_dates".into(),
    }
}

fn exprs(xs: &[HExpr]) -> Vec<HNode> {
    xs.iter().map(expr_node).collect()
}

fn place_node(p: &HPlace) -> HNode {
    match p {
        HPlace::Local(s) => hn("plocal", String::new(), *s as i64, -1, vec![], None),
        HPlace::Field(q, i) => hn("pfield", String::new(), *i as i64, -1, vec![place_node(q)], None),
        HPlace::Index(q, e, sp) => hn("pindex", String::new(), -1, -1, vec![place_node(q), expr_node(e)], Some(*sp)),
    }
}

fn arg_node(a: &HArg) -> HNode {
    match a {
        HArg::Val(e) => hk("val", vec![expr_node(e)]),
        HArg::InOut(p) => hk("inout", vec![place_node(p)]),
    }
}

fn sel_node(ax: u32, s: &HSel) -> HNode {
    match s {
        HSel::At(e) => hn("at", String::new(), ax as i64, -1, vec![expr_node(e)], None),
        HSel::Range(a, b, incl) => hn("range", String::new(), ax as i64, *incl as i64, vec![expr_node(a), expr_node(b)], None),
    }
}

pub fn pat_node(p: &HPat) -> HNode {
    match p {
        HPat::Wild => hk("pwild", vec![]),
        HPat::Bind(s) => hn("pbind", String::new(), *s as i64, -1, vec![], None),
        HPat::Lit(v) => hn("plit", value_text(v), -1, -1, vec![], None),
        HPat::IntLit(v, var) => hn("pintlit", v.to_string(), *var as i64, -1, vec![], None),
        HPat::Tuple(ps) => hk("ptuple", ps.iter().map(pat_node).collect()),
        HPat::Variant(e, v, ps) => hn("pvariant", String::new(), *e as i64, *v as i64, ps.iter().map(pat_node).collect(), None),
    }
}

fn opt_node(e: &Option<Box<HExpr>>) -> HNode {
    e.as_ref().map(|x| expr_node(x)).unwrap_or_else(none_node)
}

fn op_text(op: &Option<BinOp>) -> String {
    op.map_or("=".to_string(), |o| crate::ast::op_str(o).to_string())
}

pub fn expr_node(e: &HExpr) -> HNode {
    let none = String::new;
    match e {
        HExpr::Lit(v) => hn("lit", value_text(v), -1, -1, vec![], None),
        HExpr::IntLit(v, var) => hn("intlit", v.to_string(), *var as i64, -1, vec![], None),
        HExpr::Local(s) => hn("local", none(), *s as i64, -1, vec![], None),
        HExpr::FnRef(f) => hn("fnref", none(), *f as i64, -1, vec![], None),
        HExpr::List(xs) => hk("list", exprs(xs)),
        HExpr::MapLit(xs) => hk("map", exprs(xs)),
        HExpr::Tuple(xs) => hk("tuple", exprs(xs)),
        HExpr::Struct(sid, xs) => hn("struct", none(), *sid as i64, -1, exprs(xs), None),
        HExpr::Variant(eid, vi, xs) => hn("variant", none(), *eid as i64, *vi as i64, exprs(xs), None),
        HExpr::Field(x, i) => hn("field", none(), *i as i64, -1, vec![expr_node(x)], None),
        HExpr::Index(a, b, sp) => hn("index", none(), -1, -1, vec![expr_node(a), expr_node(b)], Some(*sp)),
        HExpr::TensorIndex(x, sels, sp) => {
            let mut kids = vec![expr_node(x)];
            kids.extend(sels.iter().map(|(a, s)| sel_node(*a, s)));
            hn("tindex", none(), -1, -1, kids, Some(*sp))
        }
        HExpr::TensorAssign(p, sels, op, v, sp) => {
            let mut kids = vec![place_node(p)];
            kids.extend(sels.iter().map(|(a, s)| sel_node(*a, s)));
            kids.push(expr_node(v));
            hn("tassign", op_text(op), -1, -1, kids, Some(*sp))
        }
        HExpr::Host(f, xs, sp) => hn("host", host_text(f), -1, -1, exprs(xs), Some(*sp)),
        HExpr::CallFn(f, args, sp) => hn("call", none(), *f as i64, -1, args.iter().map(arg_node).collect(), Some(*sp)),
        HExpr::CallValue(c, xs, sp) => {
            let mut kids = vec![expr_node(c)];
            kids.extend(exprs(xs));
            hn("callv", none(), -1, -1, kids, Some(*sp))
        }
        HExpr::CallBuiltin(b, args, sp) => hn("builtin", builtin_name(*b).to_string(), -1, -1, args.iter().map(arg_node).collect(), Some(*sp)),
        HExpr::Unary(op, x, sp) => {
            let t = if *op == UnOp::Neg { "-" } else { "not" };
            hn("unary", t.to_string(), -1, -1, vec![expr_node(x)], Some(*sp))
        }
        HExpr::Binary(op, a, b, sp) => hn("binary", crate::ast::op_str(*op).to_string(), -1, -1, vec![expr_node(a), expr_node(b)], Some(*sp)),
        HExpr::And(a, b) => hk("and", vec![expr_node(a), expr_node(b)]),
        HExpr::Or(a, b) => hk("or", vec![expr_node(a), expr_node(b)]),
        HExpr::Assign(p, v) => hk("assign", vec![place_node(p), expr_node(v)]),
        HExpr::OpAssign(p, op, v, sp) => hn("opassign", crate::ast::op_str(*op).to_string(), -1, -1, vec![place_node(p), expr_node(v)], Some(*sp)),
        HExpr::If(c, t, els) => hk("if", vec![expr_node(c), expr_node(t), opt_node(els)]),
        HExpr::While(c, b) => hk("while", vec![expr_node(c), expr_node(b)]),
        HExpr::ForRange { slot, name, lo, hi, inclusive, body } => {
            hn("forrange", name.clone(), *slot as i64, *inclusive as i64, vec![expr_node(lo), expr_node(hi), expr_node(body)], None)
        }
        HExpr::ForList { pat, list, body } => hk("forlist", vec![pat_node(pat), expr_node(list), expr_node(body)]),
        HExpr::Match(s, arms, sp) => {
            let mut kids = vec![expr_node(s)];
            kids.extend(arms.iter().map(|(p, b)| hk("arm", vec![pat_node(p), expr_node(b)])));
            hn("match", none(), -1, -1, kids, Some(*sp))
        }
        HExpr::Block(stmts, tail) => {
            let mut kids = exprs(stmts);
            kids.push(opt_node(tail));
            hk("block", kids)
        }
        HExpr::Let(p, init) => hk("let", vec![pat_node(p), expr_node(init)]),
        HExpr::Lambda(def, outer) => {
            let caps: Vec<String> = outer.iter().map(|s| s.to_string()).collect();
            hn("lambda", caps.join(" "), *def as i64, -1, vec![], None)
        }
        HExpr::Return(x) => hk("return", vec![opt_node(x)]),
        HExpr::Break => hk("break", vec![]),
        HExpr::Continue => hk("continue", vec![]),
    }
}

/// The canonical text of an expression.
pub fn render(e: &HExpr) -> String {
    let mut out = String::new();
    expr_node(e).render(&mut out);
    out
}
