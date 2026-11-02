//! Recursive-descent parser with Pratt precedence for expressions.
//!
//! Binding powers (left, right), low to high:
//! assignment (1,1) < range (2,3) < `or` (3,4) < `and` (5,6) < comparison (7,8)
//! < `+ -` (9,10) < `* / %` (11,12) < unary `-` (operand at 13) < `^` (14,14, right-assoc)
//! < postfix: call, index, field, method call.

use crate::ast::*;
use crate::diag::{Diag, KResult, Span};
use crate::lexer::{Tok, Token, describe};

pub fn parse(toks: Vec<Token>) -> KResult<Vec<Item>> {
    Parser { toks, pos: 0, no_struct: false }.program()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    /// inside `if`/`while`/`for`/`match` heads `Name {` is a block, not a struct literal
    no_struct: bool,
}

fn reserved_err(r: &str, sp: Span) -> Diag {
    let hint = match r {
        "smooth" => "`smooth` is reserved for declared relaxations; today use `smooth_max`, `smooth_min` and `smooth_step`",
        "sink" => "sinks arrive in a later milestone",
        "kernel" | "compute" | "schedule" | "unsafe" | "parallel" => {
            "the kernel and low-level layers arrive with the native backend (M7)"
        }
        _ => "reserved for a future version of Kairós",
    };
    Diag::new("E0002", sp, format!("`{r}` is a reserved keyword")).hint(hint)
}

enum Infix {
    Assign,
    Range,
    Or,
    And,
    Bin(BinOp),
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos].tok
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.pos + k).min(self.toks.len() - 1)].tok
    }

    fn span(&self) -> Span {
        self.toks[self.pos].span
    }

    fn prev_span(&self) -> Span {
        self.toks[self.pos.saturating_sub(1)].span
    }

    fn bump(&mut self) -> Token {
        let t = self.toks[self.pos].clone();
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
        t
    }

    fn eat(&mut self, t: &Tok) -> bool {
        if self.peek() == t {
            self.bump();
            true
        } else {
            false
        }
    }

    fn err_expected(&self, what: &str) -> Diag {
        match self.peek() {
            Tok::Reserved(r) => reserved_err(r, self.span()),
            Tok::At => Diag::new("E0001", self.span(), "unexpected `@`")
                .hint("temporal annotations go after a function's return type: `-> f64 @ end(m - 1)`"),
            t => Diag::new("E0001", self.span(), format!("expected {what}, found {}", describe(t))),
        }
    }

    fn expect(&mut self, t: Tok, what: &str) -> KResult<Span> {
        if *self.peek() == t { Ok(self.bump().span) } else { Err(self.err_expected(what)) }
    }

    fn skip_semis(&mut self) {
        while *self.peek() == Tok::Semi {
            self.bump();
        }
    }

    fn skip_separators(&mut self) {
        while matches!(self.peek(), Tok::Semi | Tok::Comma) {
            self.bump();
        }
    }

    fn ident(&mut self, what: &str) -> KResult<(String, Span)> {
        match self.peek().clone() {
            Tok::Ident(s) => {
                let sp = self.bump().span;
                Ok((s, sp))
            }
            _ => Err(self.err_expected(what)),
        }
    }

    fn with_structs<T>(&mut self, allowed: bool, f: impl FnOnce(&mut Self) -> KResult<T>) -> KResult<T> {
        let save = self.no_struct;
        self.no_struct = !allowed;
        let r = f(self);
        self.no_struct = save;
        r
    }

    // ------------------------------------------------------------------ items

    fn program(&mut self) -> KResult<Vec<Item>> {
        let mut items = Vec::new();
        loop {
            self.skip_semis();
            if *self.peek() == Tok::Eof {
                break;
            }
            items.push(self.item()?);
        }
        Ok(items)
    }

    fn item(&mut self) -> KResult<Item> {
        match self.peek() {
            Tok::Fn => Ok(Item::Fn(self.fn_decl()?)),
            Tok::Struct => Ok(Item::Struct(self.struct_decl()?)),
            Tok::Enum => Ok(Item::Enum(self.enum_decl()?)),
            Tok::Axis => Ok(Item::Axis(self.axis_decl()?)),
            Tok::Protocol => Ok(Item::Protocol(self.protocol_decl()?)),
            Tok::Import => {
                let start = self.span();
                self.bump();
                match self.peek().clone() {
                    Tok::Str(path) => {
                        self.bump();
                        Ok(Item::Import(path, start.to(self.prev_span())))
                    }
                    _ => Err(self.err_expected("the file to import in quotes, e.g. `import \"lab/universe.krs\"`")),
                }
            }
            _ => {
                let (s, block_like) = self.stmt()?;
                if !(self.eat(&Tok::Semi) || *self.peek() == Tok::Eof || block_like) {
                    return Err(self.err_expected("a new line or `;`"));
                }
                Ok(Item::Stmt(s))
            }
        }
    }

    fn fn_decl(&mut self) -> KResult<FnDecl> {
        let start = self.expect(Tok::Fn, "`fn`")?;
        let (name, _) = self.ident("a function name")?;
        // `fn adam[P](...)`: type parameters (differentiable types: the tangent space)
        let mut type_params = Vec::new();
        if self.eat(&Tok::LBracket) {
            while *self.peek() != Tok::RBracket {
                type_params.push(self.ident("a type parameter name")?);
                if !self.eat(&Tok::Comma) {
                    break;
                }
            }
            self.expect(Tok::RBracket, "`]`")?;
        }
        self.expect(Tok::LParen, "`(`")?;
        let mut params = Vec::new();
        while *self.peek() != Tok::RParen {
            let ps = self.span();
            let inout = self.eat(&Tok::Inout);
            let (pname, _) = self.ident("a parameter name")?;
            self.expect(Tok::Colon, "`:` and the parameter type")?;
            let ty = self.type_expr()?;
            params.push(Param { name: pname, ty, inout, span: ps.to(self.prev_span()) });
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::RParen, "`)`")?;
        let ret = if self.eat(&Tok::Arrow) { Some(self.type_expr()?) } else { None };
        // `-> T @ end(m - 1)`: when the result is known (temporal types, M3)
        let known_at = if self.eat(&Tok::At) {
            let at = self.prev_span();
            if ret.is_none() {
                return Err(Diag::new("E0001", at, "a temporal annotation needs a return type")
                    .hint("e.g. `fn signal(r: Tensor[month; f64], m: i64) -> f64 @ end(m - 1) { ... }`"));
            }
            let (kw, ksp) = self.ident("`end(...)`")?;
            if kw != "end" {
                return Err(Diag::new("E0001", ksp, "expected `end(...)` after `@`").hint("write `@ end(m - 1)`"));
            }
            self.expect(Tok::LParen, "`(`")?;
            let e = self.with_structs(true, |p| p.expr())?;
            let close = self.expect(Tok::RParen, "`)`")?;
            Some((e, at.to(close)))
        } else {
            None
        };
        let body = self.block()?;
        Ok(FnDecl { name, type_params, params, ret, known_at, span: start.to(body.span), body })
    }

    /// `protocol "name" { research: until 201812  holdout: from 201901  rule: "..." }`
    fn protocol_decl(&mut self) -> KResult<ProtocolDecl> {
        let start = self.expect(Tok::Protocol, "`protocol`")?;
        let name = match self.peek().clone() {
            Tok::Str(s) => {
                self.bump();
                s
            }
            _ => return Err(self.err_expected("the protocol's name in quotes, e.g. `protocol \"momentum-v1\" { ... }`")),
        };
        self.expect(Tok::LBrace, "`{`")?;
        let fields_hint = "the fields are `research: until YYYYMM`, `holdout: from YYYYMM`, `rule: \"...\"` and, optionally, `min_months: N` (default 12)";
        let (mut until, mut from, mut rule, mut min_months) = (None, None, None, None);
        loop {
            while self.eat(&Tok::Semi) || self.eat(&Tok::Comma) {}
            if self.eat(&Tok::RBrace) {
                break;
            }
            let (field, fsp) = self.ident("a protocol field")?;
            self.expect(Tok::Colon, "`:`")?;
            let month = |p: &mut Self, kw: &str| -> KResult<i64> {
                let (w, wsp) = p.ident(&format!("`{kw}`"))?;
                if w != kw {
                    return Err(Diag::new("E0001", wsp, format!("expected `{kw}`")).hint(fields_hint));
                }
                match p.peek().clone() {
                    Tok::Int(v) if (100_001..=999_912).contains(&v) && (1..=12).contains(&(v % 100)) => {
                        p.bump();
                        Ok(v)
                    }
                    _ => Err(p.err_expected("a month as YYYYMM, e.g. `201812`")),
                }
            };
            match field.as_str() {
                "research" => until = Some(month(self, "until")?),
                "holdout" => from = Some(month(self, "from")?),
                "rule" => match self.peek().clone() {
                    Tok::Str(s) => {
                        self.bump();
                        rule = Some(s);
                    }
                    _ => return Err(self.err_expected("the rule in quotes")),
                },
                // the holdout cannot be opened before it has this many months of data
                "min_months" => match self.peek().clone() {
                    Tok::Int(v) if (1..=1200).contains(&v) => {
                        self.bump();
                        min_months = Some(v);
                    }
                    _ => return Err(self.err_expected("a number of months between 1 and 1200")),
                },
                other => return Err(Diag::new("E0001", fsp, format!("unknown protocol field `{other}`")).hint(fields_hint)),
            }
        }
        let span = start.to(self.prev_span());
        let (Some(research_until), Some(holdout_from), Some(rule)) = (until, from, rule) else {
            return Err(Diag::new("E0001", span, "the protocol needs `research`, `holdout` and `rule`").hint(fields_hint));
        };
        if holdout_from <= research_until {
            return Err(Diag::new("E0001", span, "the holdout must start after the research period ends"));
        }
        Ok(ProtocolDecl { name, research_until, holdout_from, min_months: min_months.unwrap_or(12), rule, span })
    }

    fn axis_decl(&mut self) -> KResult<AxisDecl> {
        let start = self.expect(Tok::Axis, "`axis`")?;
        let (name, _) = self.ident("an axis name")?;
        let temporal = if self.eat(&Tok::Colon) {
            let (kw, ksp) = self.ident("`temporal(daily)` or `temporal(monthly)`")?;
            if kw != "temporal" {
                return Err(Diag::new("E0001", ksp, "expected `temporal(daily)` or `temporal(monthly)`"));
            }
            self.expect(Tok::LParen, "`(`")?;
            let freq = self.ident("`daily` or `monthly`")?;
            self.expect(Tok::RParen, "`)`")?;
            Some(freq)
        } else {
            None
        };
        Ok(AxisDecl { name, temporal, span: start.to(self.prev_span()) })
    }

    fn struct_decl(&mut self) -> KResult<StructDecl> {
        let start = self.expect(Tok::Struct, "`struct`")?;
        let (name, _) = self.ident("a struct name")?;
        self.expect(Tok::LBrace, "`{`")?;
        let mut fields = Vec::new();
        loop {
            self.skip_separators();
            if self.eat(&Tok::RBrace) {
                break;
            }
            let (f, _) = self.ident("a field name")?;
            self.expect(Tok::Colon, "`:` and the field type")?;
            fields.push((f, self.type_expr()?));
        }
        Ok(StructDecl { name, fields, span: start.to(self.prev_span()) })
    }

    fn enum_decl(&mut self) -> KResult<EnumDecl> {
        let start = self.expect(Tok::Enum, "`enum`")?;
        let (name, _) = self.ident("an enum name")?;
        self.expect(Tok::LBrace, "`{`")?;
        let mut variants = Vec::new();
        loop {
            self.skip_separators();
            if self.eat(&Tok::RBrace) {
                break;
            }
            let (v, _) = self.ident("a variant name")?;
            let mut tys = Vec::new();
            if self.eat(&Tok::LParen) {
                while *self.peek() != Tok::RParen {
                    tys.push(self.type_expr()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                self.expect(Tok::RParen, "`)`")?;
            }
            variants.push((v, tys));
        }
        Ok(EnumDecl { name, variants, span: start.to(self.prev_span()) })
    }

    fn type_list(&mut self, close: Tok) -> KResult<Vec<TypeExpr>> {
        let mut ts = Vec::new();
        while *self.peek() != close {
            ts.push(self.type_expr()?);
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        Ok(ts)
    }

    fn type_expr(&mut self) -> KResult<TypeExpr> {
        let sp = self.span();
        match self.peek().clone() {
            Tok::LParen => {
                self.bump();
                let ts = self.type_list(Tok::RParen)?;
                let end = self.expect(Tok::RParen, "`)`")?;
                Ok(TypeExpr::Tuple(ts, sp.to(end)))
            }
            Tok::Fn => {
                self.bump();
                self.expect(Tok::LParen, "`(`")?;
                let ps = self.type_list(Tok::RParen)?;
                self.expect(Tok::RParen, "`)`")?;
                let ret = if self.eat(&Tok::Arrow) { self.type_expr()? } else { TypeExpr::Tuple(vec![], self.prev_span()) };
                Ok(TypeExpr::Fn(ps, Box::new(ret), sp.to(self.prev_span())))
            }
            Tok::Ident(name) => {
                self.bump();
                if name == "Tensor" && *self.peek() == Tok::LBracket {
                    self.bump();
                    let mut axes = Vec::new();
                    while !matches!(self.peek(), Tok::Semi | Tok::RBracket) {
                        axes.push(self.ident("an axis name")?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(Tok::Semi, "`;` and the element type, e.g. `Tensor[asset; f64]`")?;
                    let elem = self.type_expr()?;
                    self.expect(Tok::RBracket, "`]`")?;
                    return Ok(TypeExpr::Tensor { axes, elem: Box::new(elem), span: sp.to(self.prev_span()) });
                }
                let mut args = Vec::new();
                if self.eat(&Tok::LBracket) {
                    args = self.type_list(Tok::RBracket)?;
                    self.expect(Tok::RBracket, "`]`")?;
                }
                Ok(TypeExpr::Named { name, args, span: sp.to(self.prev_span()) })
            }
            _ => Err(self.err_expected("a type")),
        }
    }

    // ------------------------------------------------------------------ statements

    fn block(&mut self) -> KResult<Block> {
        let start = self.expect(Tok::LBrace, "`{`")?;
        self.with_structs(true, |p| {
            let mut stmts = Vec::new();
            let mut tail = None;
            loop {
                p.skip_semis();
                if *p.peek() == Tok::RBrace {
                    break;
                }
                let (st, block_like) = p.stmt()?;
                if p.eat(&Tok::Semi) {
                    stmts.push(st);
                    continue;
                }
                if *p.peek() == Tok::RBrace {
                    match st {
                        Stmt::Expr(e) => tail = Some(Box::new(e)),
                        other => stmts.push(other),
                    }
                    break;
                }
                if block_like {
                    stmts.push(st);
                    continue;
                }
                return Err(p.err_expected("a new line or `;`"));
            }
            let end = p.expect(Tok::RBrace, "`}`")?;
            Ok(Block { stmts, tail, span: start.to(end) })
        })
    }

    fn stmt(&mut self) -> KResult<(Stmt, bool)> {
        match self.peek() {
            Tok::Let | Tok::Var => {
                let start = self.span();
                let mutable = *self.peek() == Tok::Var;
                self.bump();
                let pat = self.pattern()?;
                let ty = if self.eat(&Tok::Colon) { Some(self.type_expr()?) } else { None };
                self.expect(Tok::Assign, "`=`")?;
                let init = self.expr()?;
                let span = start.to(init.span);
                Ok((Stmt::Let { mutable, pat, ty, init, span }, false))
            }
            Tok::Fn | Tok::Struct | Tok::Enum | Tok::Axis | Tok::Protocol | Tok::Import => {
                Err(Diag::new("E0003", self.span(), "declarations are only allowed at the top level (for now)"))
            }
            _ => {
                let e = self.expr()?;
                let block_like = matches!(
                    e.kind,
                    ExprKind::If { .. } | ExprKind::While { .. } | ExprKind::For { .. } | ExprKind::Match { .. } | ExprKind::Block(_)
                );
                Ok((Stmt::Expr(e), block_like))
            }
        }
    }

    fn pattern(&mut self) -> KResult<Pattern> {
        let sp = self.span();
        match self.peek().clone() {
            Tok::Ident(name) => {
                self.bump();
                if name == "_" {
                    return Ok(Pattern::Wild(sp));
                }
                let (enum_name, variant) = if *self.peek() == Tok::Dot {
                    self.bump();
                    let (v, _) = self.ident("a variant name")?;
                    (Some(name), v)
                } else {
                    (None, name)
                };
                let is_variant = enum_name.is_some() || variant.chars().next().is_some_and(char::is_uppercase);
                if !is_variant {
                    return Ok(Pattern::Bind(variant, sp));
                }
                let mut args = Vec::new();
                if self.eat(&Tok::LParen) {
                    while *self.peek() != Tok::RParen {
                        args.push(self.pattern()?);
                        if !self.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    self.expect(Tok::RParen, "`)`")?;
                }
                Ok(Pattern::Variant { enum_name, variant, args, span: sp.to(self.prev_span()) })
            }
            Tok::Int(v) => {
                self.bump();
                Ok(Pattern::Int(v, sp))
            }
            Tok::Minus => {
                self.bump();
                match self.peek().clone() {
                    Tok::Int(v) => {
                        self.bump();
                        Ok(Pattern::Int(-v, sp.to(self.prev_span())))
                    }
                    _ => Err(self.err_expected("an integer")),
                }
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Pattern::Str(s, sp))
            }
            Tok::True | Tok::False => {
                let b = *self.peek() == Tok::True;
                self.bump();
                Ok(Pattern::Bool(b, sp))
            }
            Tok::LParen => {
                self.bump();
                let mut ps = Vec::new();
                while *self.peek() != Tok::RParen {
                    ps.push(self.pattern()?);
                    if !self.eat(&Tok::Comma) {
                        break;
                    }
                }
                let end = self.expect(Tok::RParen, "`)`")?;
                Ok(Pattern::Tuple(ps, sp.to(end)))
            }
            _ => Err(self.err_expected("a pattern")),
        }
    }

    // ------------------------------------------------------------------ expressions

    pub fn expr(&mut self) -> KResult<Expr> {
        self.expr_bp(0)
    }

    fn cond_expr(&mut self) -> KResult<Expr> {
        self.with_structs(false, |p| p.expr())
    }

    fn expr_bp(&mut self, min: u8) -> KResult<Expr> {
        let mut lhs = self.prefix()?;
        loop {
            match self.peek().clone() {
                Tok::LParen => {
                    let (args, end) = self.args()?;
                    lhs = Expr { span: lhs.span.to(end), kind: ExprKind::Call(Box::new(lhs), args) };
                    continue;
                }
                Tok::LBracket => {
                    self.bump();
                    let items = self.with_structs(true, |p| {
                        let mut items = Vec::new();
                        while *p.peek() != Tok::RBracket {
                            let axis = p.named_prefix("an axis name")?;
                            items.push(IndexArg { axis, expr: p.expr()? });
                            if !p.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        Ok(items)
                    })?;
                    let end = self.expect(Tok::RBracket, "`]`")?;
                    if items.is_empty() {
                        return Err(Diag::new("E0001", end, "empty index"));
                    }
                    lhs = Expr { span: lhs.span.to(end), kind: ExprKind::Index(Box::new(lhs), items) };
                    continue;
                }
                Tok::Dot => {
                    self.bump();
                    match self.peek().clone() {
                        Tok::Int(k) => {
                            let sp = self.bump().span;
                            lhs = Expr { span: lhs.span.to(sp), kind: ExprKind::TupleIndex(Box::new(lhs), k as usize) };
                        }
                        Tok::Ident(name) => {
                            let sp = self.bump().span;
                            if *self.peek() == Tok::LParen {
                                let (args, end) = self.args()?;
                                lhs = Expr {
                                    span: lhs.span.to(end),
                                    kind: ExprKind::MethodCall { recv: Box::new(lhs), name, args },
                                };
                            } else {
                                lhs = Expr { span: lhs.span.to(sp), kind: ExprKind::Field(Box::new(lhs), name) };
                            }
                        }
                        _ => return Err(self.err_expected("a field or method name after `.`")),
                    }
                    continue;
                }
                _ => {}
            }
            let (op, lbp, rbp) = match self.peek() {
                Tok::Assign | Tok::PlusAssign | Tok::MinusAssign | Tok::StarAssign | Tok::SlashAssign => (Infix::Assign, 1, 1),
                Tok::DotDot | Tok::DotDotEq => (Infix::Range, 2, 3),
                Tok::Or => (Infix::Or, 3, 4),
                Tok::And => (Infix::And, 5, 6),
                Tok::EqEq => (Infix::Bin(BinOp::Eq), 7, 8),
                Tok::NotEq => (Infix::Bin(BinOp::Ne), 7, 8),
                Tok::Lt => (Infix::Bin(BinOp::Lt), 7, 8),
                Tok::Le => (Infix::Bin(BinOp::Le), 7, 8),
                Tok::Gt => (Infix::Bin(BinOp::Gt), 7, 8),
                Tok::Ge => (Infix::Bin(BinOp::Ge), 7, 8),
                Tok::Plus => (Infix::Bin(BinOp::Add), 9, 10),
                Tok::Minus => (Infix::Bin(BinOp::Sub), 9, 10),
                Tok::Star => (Infix::Bin(BinOp::Mul), 11, 12),
                Tok::Slash => (Infix::Bin(BinOp::Div), 11, 12),
                Tok::Percent => (Infix::Bin(BinOp::Rem), 11, 12),
                Tok::Caret => (Infix::Bin(BinOp::Pow), 14, 14),
                _ => break,
            };
            if lbp < min {
                break;
            }
            let op_tok = self.bump().tok;
            let rhs = self.expr_bp(rbp)?;
            let span = lhs.span.to(rhs.span);
            lhs = match op {
                Infix::Assign => {
                    let bop = match op_tok {
                        Tok::PlusAssign => Some(BinOp::Add),
                        Tok::MinusAssign => Some(BinOp::Sub),
                        Tok::StarAssign => Some(BinOp::Mul),
                        Tok::SlashAssign => Some(BinOp::Div),
                        _ => None,
                    };
                    Expr { span, kind: ExprKind::Assign { target: Box::new(lhs), op: bop, value: Box::new(rhs) } }
                }
                Infix::Range => Expr {
                    span,
                    kind: ExprKind::Range { lo: Box::new(lhs), hi: Box::new(rhs), inclusive: op_tok == Tok::DotDotEq },
                },
                Infix::Or => Expr { span, kind: ExprKind::Or(Box::new(lhs), Box::new(rhs)) },
                Infix::And => Expr { span, kind: ExprKind::And(Box::new(lhs), Box::new(rhs)) },
                Infix::Bin(b) => Expr { span, kind: ExprKind::Binary(b, Box::new(lhs), Box::new(rhs)) },
            };
        }
        Ok(lhs)
    }

    /// `name:` prefix of a named argument or named index (`over: asset`, `month: m`).
    fn named_prefix(&mut self, what: &str) -> KResult<Option<(String, Span)>> {
        if matches!(self.peek(), Tok::Ident(_)) && *self.peek_at(1) == Tok::Colon {
            let n = self.ident(what)?;
            self.bump();
            Ok(Some(n))
        } else {
            Ok(None)
        }
    }

    fn args(&mut self) -> KResult<(Vec<Arg>, Span)> {
        self.expect(Tok::LParen, "`(`")?;
        self.with_structs(true, |p| {
            let mut args = Vec::new();
            while *p.peek() != Tok::RParen {
                let name = p.named_prefix("an argument name")?;
                let inout = p.eat(&Tok::Inout);
                args.push(Arg { name, inout, expr: p.expr()? });
                if !p.eat(&Tok::Comma) {
                    break;
                }
            }
            let end = p.expect(Tok::RParen, "`)` to close the argument list")?;
            Ok((args, end))
        })
    }

    fn prefix(&mut self) -> KResult<Expr> {
        let sp = self.span();
        match self.peek().clone() {
            Tok::Int(v) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Int(v), span: sp })
            }
            Tok::Float(v) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Float(v), span: sp })
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Expr { kind: ExprKind::Str(s), span: sp })
            }
            Tok::True | Tok::False => {
                let b = *self.peek() == Tok::True;
                self.bump();
                Ok(Expr { kind: ExprKind::Bool(b), span: sp })
            }
            Tok::Ident(name) => {
                self.bump();
                if *self.peek() == Tok::LBrace && !self.no_struct {
                    return self.struct_lit(name, sp);
                }
                Ok(Expr { kind: ExprKind::Name(name), span: sp })
            }
            Tok::LParen => {
                self.bump();
                self.with_structs(true, |p| {
                    if p.eat(&Tok::RParen) {
                        return Ok(Expr { kind: ExprKind::Unit, span: sp.to(p.prev_span()) });
                    }
                    let first = p.expr()?;
                    if p.eat(&Tok::Comma) {
                        let mut items = vec![first];
                        while *p.peek() != Tok::RParen {
                            items.push(p.expr()?);
                            if !p.eat(&Tok::Comma) {
                                break;
                            }
                        }
                        let end = p.expect(Tok::RParen, "`)`")?;
                        return Ok(Expr { kind: ExprKind::Tuple(items), span: sp.to(end) });
                    }
                    let end = p.expect(Tok::RParen, "`)`")?;
                    Ok(Expr { kind: first.kind, span: sp.to(end) })
                })
            }
            Tok::LBracket => {
                self.bump();
                self.with_structs(true, |p| {
                    // `[:]`: the empty map
                    if p.eat(&Tok::Colon) {
                        let end = p.expect(Tok::RBracket, "`]`")?;
                        return Ok(Expr { kind: ExprKind::Map(Vec::new()), span: sp.to(end) });
                    }
                    let mut items = Vec::new();
                    let mut pairs = Vec::new();
                    while *p.peek() != Tok::RBracket {
                        let e = p.expr()?;
                        if p.eat(&Tok::Colon) {
                            if !items.is_empty() {
                                return Err(Diag::new("E0003", e.span, "a list element followed by a map entry")
                                    .hint("a list is `[a, b]`, a map is `[key: value, ...]`"));
                            }
                            let v = p.expr()?;
                            pairs.push((e, v));
                        } else {
                            if !pairs.is_empty() {
                                return Err(Diag::new("E0003", e.span, "a map entry needs `key: value`"));
                            }
                            items.push(e);
                        }
                        if !p.eat(&Tok::Comma) {
                            break;
                        }
                    }
                    let end = p.expect(Tok::RBracket, "`]`")?;
                    let kind = if pairs.is_empty() { ExprKind::List(items) } else { ExprKind::Map(pairs) };
                    Ok(Expr { kind, span: sp.to(end) })
                })
            }
            Tok::Pipe => self.lambda(),
            Tok::Minus => {
                self.bump();
                let e = self.expr_bp(13)?;
                let span = sp.to(e.span);
                Ok(Expr { kind: ExprKind::Unary(UnOp::Neg, Box::new(e)), span })
            }
            Tok::Not => {
                self.bump();
                let e = self.expr_bp(7)?;
                let span = sp.to(e.span);
                Ok(Expr { kind: ExprKind::Unary(UnOp::Not, Box::new(e)), span })
            }
            Tok::If => self.if_expr(),
            Tok::While => {
                self.bump();
                let cond = self.cond_expr()?;
                let body = self.block()?;
                Ok(Expr { span: sp.to(body.span), kind: ExprKind::While { cond: Box::new(cond), body } })
            }
            Tok::For => {
                self.bump();
                let pat = self.pattern()?;
                self.expect(Tok::In, "`in`")?;
                let iter = self.cond_expr()?;
                let body = self.block()?;
                Ok(Expr { span: sp.to(body.span), kind: ExprKind::For { pat, iter: Box::new(iter), body } })
            }
            Tok::Match => self.match_expr(),
            Tok::LBrace => {
                let b = self.block()?;
                Ok(Expr { span: b.span, kind: ExprKind::Block(b) })
            }
            Tok::Return => {
                self.bump();
                if matches!(self.peek(), Tok::Semi | Tok::RBrace | Tok::Eof) {
                    return Ok(Expr { kind: ExprKind::Return(None), span: sp });
                }
                let e = self.expr()?;
                let span = sp.to(e.span);
                Ok(Expr { kind: ExprKind::Return(Some(Box::new(e))), span })
            }
            Tok::Break => {
                self.bump();
                Ok(Expr { kind: ExprKind::Break, span: sp })
            }
            Tok::Continue => {
                self.bump();
                Ok(Expr { kind: ExprKind::Continue, span: sp })
            }
            _ => Err(self.err_expected("an expression")),
        }
    }

    fn struct_lit(&mut self, name: String, start: Span) -> KResult<Expr> {
        self.expect(Tok::LBrace, "`{`")?;
        let mut fields = Vec::new();
        loop {
            self.skip_separators();
            if *self.peek() == Tok::RBrace {
                break;
            }
            let (f, _) = self.ident("a field name")?;
            self.expect(Tok::Colon, "`:`")?;
            let e = self.with_structs(true, |p| p.expr())?;
            fields.push((f, e));
        }
        let end = self.expect(Tok::RBrace, "`}`")?;
        Ok(Expr { span: start.to(end), kind: ExprKind::StructLit { name, fields } })
    }

    fn if_expr(&mut self) -> KResult<Expr> {
        let start = self.expect(Tok::If, "`if`")?;
        let cond = self.cond_expr()?;
        let then = self.block()?;
        // `else` may start the next line
        if *self.peek() == Tok::Semi && *self.peek_at(1) == Tok::Else {
            self.bump();
        }
        let els = if self.eat(&Tok::Else) {
            if *self.peek() == Tok::If {
                Some(Box::new(self.if_expr()?))
            } else {
                let b = self.block()?;
                Some(Box::new(Expr { span: b.span, kind: ExprKind::Block(b) }))
            }
        } else {
            None
        };
        let end = els.as_ref().map(|e| e.span).unwrap_or(then.span);
        Ok(Expr { span: start.to(end), kind: ExprKind::If { cond: Box::new(cond), then, els } })
    }

    fn match_expr(&mut self) -> KResult<Expr> {
        let start = self.expect(Tok::Match, "`match`")?;
        let scrut = self.cond_expr()?;
        self.expect(Tok::LBrace, "`{`")?;
        self.with_structs(true, |p| {
            let mut arms = Vec::new();
            loop {
                p.skip_separators();
                if *p.peek() == Tok::RBrace {
                    break;
                }
                let pat = p.pattern()?;
                p.expect(Tok::FatArrow, "`=>`")?;
                let body = p.expr()?;
                arms.push(Arm { span: pat.span().to(body.span), pat, body });
                if !matches!(p.peek(), Tok::Comma | Tok::Semi | Tok::RBrace) {
                    return Err(p.err_expected("`,` or a new line between match arms"));
                }
            }
            let end = p.expect(Tok::RBrace, "`}`")?;
            Ok(Expr { span: start.to(end), kind: ExprKind::Match { scrut: Box::new(scrut), arms } })
        })
    }

    fn lambda(&mut self) -> KResult<Expr> {
        let start = self.expect(Tok::Pipe, "`|`")?;
        let mut params = Vec::new();
        while *self.peek() != Tok::Pipe {
            let (name, sp) = self.ident("a parameter name")?;
            let ty = if self.eat(&Tok::Colon) { Some(self.type_expr()?) } else { None };
            params.push((name, ty, sp));
            if !self.eat(&Tok::Comma) {
                break;
            }
        }
        self.expect(Tok::Pipe, "`|`")?;
        let body = self.with_structs(true, |p| p.expr())?;
        Ok(Expr { span: start.to(body.span), kind: ExprKind::Lambda { params, body: Box::new(body) } })
    }
}
