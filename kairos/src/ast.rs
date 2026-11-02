//! Abstract syntax tree produced by the parser (untyped, names unresolved).

use crate::diag::Span;

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Axis(AxisDecl),
    Stmt(Stmt),
}

/// `axis asset` or `axis month: temporal(monthly)`
#[derive(Debug, Clone)]
pub struct AxisDecl {
    pub name: String,
    pub temporal: Option<(String, Span)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct FnDecl {
    pub name: String,
    /// `fn f[P](...)`: type parameters, each standing for a differentiable type
    pub type_params: Vec<(String, Span)>,
    pub params: Vec<Param>,
    pub ret: Option<TypeExpr>,
    /// `-> T @ end(e)`: the result may only use what is known at the end of month `e`
    pub known_at: Option<(Expr, Span)>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
    pub inout: bool,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, TypeExpr)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<(String, Vec<TypeExpr>)>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Named { name: String, args: Vec<TypeExpr>, span: Span },
    Tuple(Vec<TypeExpr>, Span),
    Fn(Vec<TypeExpr>, Box<TypeExpr>, Span),
    /// `Tensor[month, asset; f64]`
    Tensor { axes: Vec<(String, Span)>, elem: Box<TypeExpr>, span: Span },
}

impl TypeExpr {
    pub fn span(&self) -> Span {
        match self {
            TypeExpr::Named { span, .. } | TypeExpr::Tuple(_, span) | TypeExpr::Fn(_, _, span) => *span,
            TypeExpr::Tensor { span, .. } => *span,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail: Option<Box<Expr>>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let { mutable: bool, pat: Pattern, ty: Option<TypeExpr>, init: Expr, span: Span },
    Expr(Expr),
}

#[derive(Debug, Clone)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

pub fn op_str(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::Rem => "%",
        BinOp::Pow => "^",
        BinOp::Eq => "==",
        BinOp::Ne => "!=",
        BinOp::Lt => "<",
        BinOp::Le => "<=",
        BinOp::Gt => ">",
        BinOp::Ge => ">=",
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg,
    Not,
}

/// Call argument: `x`, `inout x` or named `over: asset`.
#[derive(Debug, Clone)]
pub struct Arg {
    pub name: Option<(String, Span)>,
    pub inout: bool,
    pub expr: Expr,
}

/// Index item: positional `xs[i]` (lists) or named `ret[month: m]`, `ret[month: a ..= b]` (tensors).
#[derive(Debug, Clone)]
pub struct IndexArg {
    pub axis: Option<(String, Span)>,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub enum ExprKind {
    Int(i64),
    Float(f64),
    Str(String),
    Bool(bool),
    Unit,
    Name(String),
    List(Vec<Expr>),
    Tuple(Vec<Expr>),
    StructLit { name: String, fields: Vec<(String, Expr)> },
    Field(Box<Expr>, String),
    TupleIndex(Box<Expr>, usize),
    Index(Box<Expr>, Vec<IndexArg>),
    Call(Box<Expr>, Vec<Arg>),
    MethodCall { recv: Box<Expr>, name: String, args: Vec<Arg> },
    Unary(UnOp, Box<Expr>),
    Binary(BinOp, Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    Assign { target: Box<Expr>, op: Option<BinOp>, value: Box<Expr> },
    If { cond: Box<Expr>, then: Block, els: Option<Box<Expr>> },
    While { cond: Box<Expr>, body: Block },
    For { pat: Pattern, iter: Box<Expr>, body: Block },
    Range { lo: Box<Expr>, hi: Box<Expr>, inclusive: bool },
    Match { scrut: Box<Expr>, arms: Vec<Arm> },
    Lambda { params: Vec<(String, Option<TypeExpr>, Span)>, body: Box<Expr> },
    Block(Block),
    Return(Option<Box<Expr>>),
    Break,
    Continue,
}

#[derive(Debug, Clone)]
pub struct Arm {
    pub pat: Pattern,
    pub body: Expr,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Wild(Span),
    Bind(String, Span),
    Int(i64, Span),
    Str(String, Span),
    Bool(bool, Span),
    Tuple(Vec<Pattern>, Span),
    /// `Shape.Circle(r)` or `Circle(r)`; a bare name starting with an uppercase letter is a variant.
    Variant { enum_name: Option<String>, variant: String, args: Vec<Pattern>, span: Span },
}

impl Pattern {
    pub fn span(&self) -> Span {
        match self {
            Pattern::Wild(s)
            | Pattern::Bind(_, s)
            | Pattern::Int(_, s)
            | Pattern::Str(_, s)
            | Pattern::Bool(_, s)
            | Pattern::Tuple(_, s) => *s,
            Pattern::Variant { span, .. } => *span,
        }
    }
}
