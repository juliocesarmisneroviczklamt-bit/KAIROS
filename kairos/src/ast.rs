//! Abstract syntax tree produced by the parser (untyped, names unresolved).

use crate::diag::Span;

#[derive(Debug, Clone)]
pub enum Item {
    Fn(FnDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Axis(AxisDecl),
    Protocol(ProtocolDecl),
    /// `import "path.krs"`: the file's declarations become available (each file once)
    Import(String, crate::diag::Span),
    Stmt(Stmt),
}

/// `protocol "momentum-v1" { research: until 201812  holdout: from 201901  rule: "..." }` (M5)
#[derive(Debug, Clone)]
pub struct ProtocolDecl {
    pub name: String,
    /// last month (YYYYMM) of the research period
    pub research_until: i64,
    /// first month (YYYYMM) of the holdout
    pub holdout_from: i64,
    /// the holdout opens only once it has at least this many months of data (default 12)
    pub min_months: i64,
    /// the acceptance rule, in words, fixed before any result is seen
    pub rule: String,
    pub span: Span,
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
    /// `["a": 1, "b": 2]`, `[:]` (M9)
    Map(Vec<(Expr, Expr)>),
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

// ------------------------------------------------------------------ canonical text (M12)

/// The syntax tree as uniform nodes, one line: `(kind text children... lo:hi)`; the
/// parser written in Kairós (kairos/compiler/parser.krs) prints the same (`kairos ast file.krs`).
/// `-` is an absent part; nodes without a span in the tree print none.
pub struct Node {
    kind: &'static str,
    text: String,
    kids: Vec<Node>,
    span: Option<Span>,
}

fn node(kind: &'static str, text: impl Into<String>, kids: Vec<Node>, span: Option<Span>) -> Node {
    Node { kind, text: text.into(), kids, span }
}

fn none() -> Node {
    node("-", "", Vec::new(), None)
}

/// A string as its code points in hex, as the lexer dump writes strings.
pub fn hex_text(s: &str) -> String {
    format!("s({})", s.chars().map(|c| format!("{:x}", c as u32)).collect::<Vec<_>>().join(" "))
}

impl Node {
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

/// The program's items, one per line.
pub fn canonical(items: &[Item]) -> String {
    let mut out = String::new();
    for it in items {
        item_node(it).render(&mut out);
        out.push('\n');
    }
    out
}

fn item_node(it: &Item) -> Node {
    match it {
        Item::Fn(f) => {
            let tps = node("tparams", "", f.type_params.iter().map(|(n, sp)| node("tp", n.clone(), vec![], Some(*sp))).collect(), None);
            let ps = node(
                "params",
                "",
                f.params
                    .iter()
                    .map(|p| node("param", if p.inout { format!("inout {}", p.name) } else { p.name.clone() }, vec![type_node(&p.ty)], Some(p.span)))
                    .collect(),
                None,
            );
            let ret = f.ret.as_ref().map(type_node).unwrap_or_else(none);
            let at = f.known_at.as_ref().map(|(e, sp)| node("at", "", vec![expr_node(e)], Some(*sp))).unwrap_or_else(none);
            node("fn", f.name.clone(), vec![tps, ps, ret, at, block_node(&f.body)], Some(f.span))
        }
        Item::Struct(s) => node(
            "struct",
            s.name.clone(),
            s.fields.iter().map(|(n, t)| node("fieldty", n.clone(), vec![type_node(t)], None)).collect(),
            Some(s.span),
        ),
        Item::Enum(e) => node(
            "enum",
            e.name.clone(),
            e.variants.iter().map(|(n, ts)| node("variant", n.clone(), ts.iter().map(type_node).collect(), None)).collect(),
            Some(e.span),
        ),
        Item::Axis(a) => {
            let t = a.temporal.as_ref().map(|(f, sp)| node("temporal", f.clone(), vec![], Some(*sp))).unwrap_or_else(none);
            node("axis", a.name.clone(), vec![t], Some(a.span))
        }
        Item::Protocol(p) => node(
            "protocol",
            format!("{} {} {} {} {}", hex_text(&p.name), p.research_until, p.holdout_from, p.min_months, hex_text(&p.rule)),
            vec![],
            Some(p.span),
        ),
        Item::Import(path, sp) => node("import", hex_text(path), vec![], Some(*sp)),
        Item::Stmt(s) => stmt_node(s),
    }
}

fn type_node(t: &TypeExpr) -> Node {
    match t {
        TypeExpr::Named { name, args, span } => node("ty", name.clone(), args.iter().map(type_node).collect(), Some(*span)),
        TypeExpr::Tuple(ts, span) => node("tytuple", "", ts.iter().map(type_node).collect(), Some(*span)),
        TypeExpr::Fn(ps, r, span) => node("tyfn", "", vec![node("tyargs", "", ps.iter().map(type_node).collect(), None), type_node(r)], Some(*span)),
        TypeExpr::Tensor { axes, elem, span } => node(
            "tytensor",
            "",
            vec![node("axes", "", axes.iter().map(|(n, sp)| node("ax", n.clone(), vec![], Some(*sp))).collect(), None), type_node(elem)],
            Some(*span),
        ),
    }
}

fn block_node(b: &Block) -> Node {
    let tail = b.tail.as_ref().map(|e| expr_node(e)).unwrap_or_else(none);
    node("block", "", vec![node("stmts", "", b.stmts.iter().map(stmt_node).collect(), None), tail], Some(b.span))
}

fn stmt_node(s: &Stmt) -> Node {
    match s {
        Stmt::Let { mutable, pat, ty, init, span } => node(
            if *mutable { "var" } else { "let" },
            "",
            vec![pat_node(pat), ty.as_ref().map(type_node).unwrap_or_else(none), expr_node(init)],
            Some(*span),
        ),
        Stmt::Expr(e) => expr_node(e),
    }
}

fn pat_node(p: &Pattern) -> Node {
    match p {
        Pattern::Wild(sp) => node("pwild", "", vec![], Some(*sp)),
        Pattern::Bind(n, sp) => node("pbind", n.clone(), vec![], Some(*sp)),
        Pattern::Int(v, sp) => node("pint", v.to_string(), vec![], Some(*sp)),
        Pattern::Str(s, sp) => node("pstr", hex_text(s), vec![], Some(*sp)),
        Pattern::Bool(b, sp) => node("pbool", b.to_string(), vec![], Some(*sp)),
        Pattern::Tuple(ps, sp) => node("ptuple", "", ps.iter().map(pat_node).collect(), Some(*sp)),
        Pattern::Variant { enum_name, variant, args, span } => {
            node("pvariant", format!("{}.{variant}", enum_name.clone().unwrap_or_default()), args.iter().map(pat_node).collect(), Some(*span))
        }
    }
}

fn args_node(args: &[Arg]) -> Node {
    node(
        "args",
        "",
        args.iter()
            .map(|a| {
                let name = a.name.as_ref().map(|(n, sp)| node("argname", n.clone(), vec![], Some(*sp))).unwrap_or_else(none);
                node("arg", if a.inout { "inout" } else { "" }, vec![name, expr_node(&a.expr)], None)
            })
            .collect(),
        None,
    )
}

fn expr_node(e: &Expr) -> Node {
    let sp = Some(e.span);
    let n = |kind: &'static str, text: String, kids: Vec<Node>| node(kind, text, kids, sp);
    match &e.kind {
        ExprKind::Int(v) => n("int", v.to_string(), vec![]),
        ExprKind::Float(v) => n("float", format!("{v:?}"), vec![]),
        ExprKind::Str(s) => n("str", hex_text(s), vec![]),
        ExprKind::Bool(b) => n("bool", b.to_string(), vec![]),
        ExprKind::Unit => n("unit", String::new(), vec![]),
        ExprKind::Name(x) => n("name", x.clone(), vec![]),
        ExprKind::List(xs) => n("list", String::new(), xs.iter().map(expr_node).collect()),
        ExprKind::Map(ps) => n("map", String::new(), ps.iter().map(|(k, v)| node("pair", "", vec![expr_node(k), expr_node(v)], None)).collect()),
        ExprKind::Tuple(xs) => n("tuple", String::new(), xs.iter().map(expr_node).collect()),
        ExprKind::StructLit { name, fields } => {
            n("structlit", name.clone(), fields.iter().map(|(f, x)| node("init", f.clone(), vec![expr_node(x)], None)).collect())
        }
        ExprKind::Field(x, f) => n("field", f.clone(), vec![expr_node(x)]),
        ExprKind::TupleIndex(x, k) => n("tindex", k.to_string(), vec![expr_node(x)]),
        ExprKind::Index(x, items) => {
            let args = items
                .iter()
                .map(|a| {
                    let axis = a.axis.as_ref().map(|(n, sp)| node("axis", n.clone(), vec![], Some(*sp))).unwrap_or_else(none);
                    node("iarg", "", vec![axis, expr_node(&a.expr)], None)
                })
                .collect();
            n("index", String::new(), vec![expr_node(x), node("iargs", "", args, None)])
        }
        ExprKind::Call(f, args) => n("call", String::new(), vec![expr_node(f), args_node(args)]),
        ExprKind::MethodCall { recv, name, args } => n("method", name.clone(), vec![expr_node(recv), args_node(args)]),
        ExprKind::Unary(UnOp::Neg, x) => n("neg", String::new(), vec![expr_node(x)]),
        ExprKind::Unary(UnOp::Not, x) => n("not", String::new(), vec![expr_node(x)]),
        ExprKind::Binary(op, a, b) => n("bin", op_str(*op).to_string(), vec![expr_node(a), expr_node(b)]),
        ExprKind::And(a, b) => n("and", String::new(), vec![expr_node(a), expr_node(b)]),
        ExprKind::Or(a, b) => n("or", String::new(), vec![expr_node(a), expr_node(b)]),
        ExprKind::Assign { target, op, value } => {
            n("assign", op.map_or("=".to_string(), |o| op_str(o).to_string()), vec![expr_node(target), expr_node(value)])
        }
        ExprKind::If { cond, then, els } => {
            n("if", String::new(), vec![expr_node(cond), block_node(then), els.as_ref().map(|x| expr_node(x)).unwrap_or_else(none)])
        }
        ExprKind::While { cond, body } => n("while", String::new(), vec![expr_node(cond), block_node(body)]),
        ExprKind::For { pat, iter, body } => n("for", String::new(), vec![pat_node(pat), expr_node(iter), block_node(body)]),
        ExprKind::Range { lo, hi, inclusive } => n("range", if *inclusive { "..=" } else { ".." }.to_string(), vec![expr_node(lo), expr_node(hi)]),
        ExprKind::Match { scrut, arms } => n(
            "match",
            String::new(),
            vec![
                expr_node(scrut),
                node("arms", "", arms.iter().map(|a| node("arm", "", vec![pat_node(&a.pat), expr_node(&a.body)], Some(a.span))).collect(), None),
            ],
        ),
        ExprKind::Lambda { params, body } => n(
            "lambda",
            String::new(),
            vec![
                node(
                    "lparams",
                    "",
                    params.iter().map(|(p, t, sp)| node("lp", p.clone(), vec![t.as_ref().map(type_node).unwrap_or_else(none)], Some(*sp))).collect(),
                    None,
                ),
                expr_node(body),
            ],
        ),
        ExprKind::Block(b) => block_node(b),
        ExprKind::Return(x) => n("return", String::new(), vec![x.as_ref().map(|x| expr_node(x)).unwrap_or_else(none)]),
        ExprKind::Break => n("break", String::new(), vec![]),
        ExprKind::Continue => n("continue", String::new(), vec![]),
    }
}
