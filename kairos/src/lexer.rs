//! Lexer with Go/Swift-style automatic semicolon insertion.
//!
//! A newline ends a statement when the previous token can end one (identifier, literal, `)`,
//! `]`, `}`, `return`, `break`, `continue`). Newlines inside `( )` and `[ ]` are ignored, a line
//! starting with `.` continues the previous one, and no `;` is inserted before `}` so the last
//! line of a block stays its value.

use crate::diag::{Diag, KResult, Span};

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Int(i64),
    Float(f64),
    Str(String),
    Ident(String),
    // keywords
    Let,
    Var,
    Fn,
    If,
    Else,
    Match,
    For,
    In,
    While,
    Break,
    Continue,
    Return,
    Struct,
    Enum,
    Axis,
    Protocol,
    Import,
    Inout,
    True,
    False,
    And,
    Or,
    Not,
    /// keyword reserved for a later milestone
    Reserved(&'static str),
    // punctuation
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Colon,
    Semi,
    Dot,
    DotDot,
    DotDotEq,
    Arrow,
    FatArrow,
    Pipe,
    At,
    Question,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Caret,
    Assign,
    PlusAssign,
    MinusAssign,
    StarAssign,
    SlashAssign,
    EqEq,
    NotEq,
    Lt,
    Le,
    Gt,
    Ge,
    Eof,
}

#[derive(Debug, Clone)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
    pub nl_before: bool,
}

/// Reserved for later milestones (axes, time, smooth, linear types, kernels...).
pub const RESERVED: [&str; 15] = [
    "trait", "kernel", "compute", "schedule", "unsafe", "parallel", "smooth", "sink",
    "pub", "as", "loop", "type", "const", "use", "self",
];

fn keyword(s: &str) -> Option<Tok> {
    Some(match s {
        "let" => Tok::Let,
        "var" => Tok::Var,
        "fn" => Tok::Fn,
        "if" => Tok::If,
        "else" => Tok::Else,
        "match" => Tok::Match,
        "for" => Tok::For,
        "in" => Tok::In,
        "while" => Tok::While,
        "break" => Tok::Break,
        "continue" => Tok::Continue,
        "return" => Tok::Return,
        "struct" => Tok::Struct,
        "enum" => Tok::Enum,
        "inout" => Tok::Inout,
        "axis" => Tok::Axis,
        "protocol" => Tok::Protocol,
        "import" => Tok::Import,
        "true" => Tok::True,
        "false" => Tok::False,
        "and" => Tok::And,
        "or" => Tok::Or,
        "not" => Tok::Not,
        _ => return RESERVED.iter().find(|r| **r == s).map(|r| Tok::Reserved(r)),
    })
}

/// A token as one line of text, for comparing lexers (`kairos tokens file.krs`; the lexer written
/// in Kairós prints the same): `Kind(payload) lo hi`, `nl` when a newline came before it. Strings
/// are their code points in hex.
pub fn canonical(t: &Token) -> String {
    let kind = match &t.tok {
        Tok::Str(s) => format!("Str({})", s.chars().map(|c| format!("{:x}", c as u32)).collect::<Vec<_>>().join(" ")),
        Tok::Ident(s) => format!("Ident({s})"),
        Tok::Reserved(r) => format!("Reserved({r})"),
        other => format!("{other:?}"),
    };
    format!("{kind} {} {}{}", t.span.lo, t.span.hi, if t.nl_before { " nl" } else { "" })
}

/// Human description of a token for error messages.
pub fn describe(t: &Tok) -> String {
    match t {
        Tok::Int(v) => format!("`{v}`"),
        Tok::Float(v) => format!("`{v:?}`"),
        Tok::Str(_) => "a string".into(),
        Tok::Ident(s) => format!("`{s}`"),
        Tok::Reserved(r) => format!("`{r}`"),
        Tok::Semi => "a new line or `;`".into(),
        Tok::Eof => "end of file".into(),
        other => {
            let s = match other {
                Tok::Let => "let",
                Tok::Var => "var",
                Tok::Fn => "fn",
                Tok::If => "if",
                Tok::Else => "else",
                Tok::Match => "match",
                Tok::For => "for",
                Tok::In => "in",
                Tok::While => "while",
                Tok::Break => "break",
                Tok::Continue => "continue",
                Tok::Return => "return",
                Tok::Struct => "struct",
                Tok::Enum => "enum",
                Tok::Inout => "inout",
                Tok::Axis => "axis",
                Tok::Protocol => "protocol",
                Tok::Import => "import",
                Tok::True => "true",
                Tok::False => "false",
                Tok::And => "and",
                Tok::Or => "or",
                Tok::Not => "not",
                Tok::LParen => "(",
                Tok::RParen => ")",
                Tok::LBrace => "{",
                Tok::RBrace => "}",
                Tok::LBracket => "[",
                Tok::RBracket => "]",
                Tok::Comma => ",",
                Tok::Colon => ":",
                Tok::Dot => ".",
                Tok::DotDot => "..",
                Tok::DotDotEq => "..=",
                Tok::Arrow => "->",
                Tok::FatArrow => "=>",
                Tok::Pipe => "|",
                Tok::At => "@",
                Tok::Question => "?",
                Tok::Plus => "+",
                Tok::Minus => "-",
                Tok::Star => "*",
                Tok::Slash => "/",
                Tok::Percent => "%",
                Tok::Caret => "^",
                Tok::Assign => "=",
                Tok::PlusAssign => "+=",
                Tok::MinusAssign => "-=",
                Tok::StarAssign => "*=",
                Tok::SlashAssign => "/=",
                Tok::EqEq => "==",
                Tok::NotEq => "!=",
                Tok::Lt => "<",
                Tok::Le => "<=",
                Tok::Gt => ">",
                Tok::Ge => ">=",
                _ => "?",
            };
            format!("`{s}`")
        }
    }
}

pub fn lex(src: &str, file: u16) -> KResult<Vec<Token>> {
    let chars: Vec<(usize, char)> = src.char_indices().collect();
    let n = chars.len();
    let pos = |k: usize| if k < n { chars[k].0 } else { src.len() };
    let mut toks: Vec<Token> = Vec::new();
    let mut i = 0;
    let mut nl = false;
    // `#!/usr/bin/env kairos` on the first line: a .krs file can be an executable script
    if src.starts_with("#!") {
        while i < n && chars[i].1 != '\n' {
            i += 1;
        }
    }
    while i < n {
        let (start, c) = chars[i];
        if c == '\n' {
            nl = true;
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c == '/' && i + 1 < n && chars[i + 1].1 == '/' {
            while i < n && chars[i].1 != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '/' && i + 1 < n && chars[i + 1].1 == '*' {
            i += 2;
            while i + 1 < n && !(chars[i].1 == '*' && chars[i + 1].1 == '/') {
                if chars[i].1 == '\n' {
                    nl = true;
                }
                i += 1;
            }
            if i + 1 >= n {
                return Err(Diag::new("E0001", Span::new(file, start, start + 2), "unterminated block comment"));
            }
            i += 2;
            continue;
        }
        // M20: f"...": a formatted string, written as the tokens of its concatenation
        if c == 'f' && i + 1 < n && chars[i + 1].1 == '"' {
            let (ts, next) = fstring(src, &chars, i, file)?;
            let mut first = true;
            for mut t in ts {
                t.nl_before = first && nl;
                first = false;
                toks.push(t);
            }
            i = next;
            nl = false;
            continue;
        }
        let tok = if c.is_alphabetic() || c == '_' {
            while i < n && (chars[i].1.is_alphanumeric() || chars[i].1 == '_') {
                i += 1;
            }
            let s = &src[start..pos(i)];
            keyword(s).unwrap_or_else(|| Tok::Ident(s.to_string()))
        } else if c.is_ascii_digit() {
            let mut is_float = false;
            while i < n && (chars[i].1.is_ascii_digit() || chars[i].1 == '_') {
                i += 1;
            }
            if i + 1 < n && chars[i].1 == '.' && chars[i + 1].1.is_ascii_digit() {
                is_float = true;
                i += 1;
                while i < n && (chars[i].1.is_ascii_digit() || chars[i].1 == '_') {
                    i += 1;
                }
            }
            if i < n && (chars[i].1 == 'e' || chars[i].1 == 'E') {
                let mut j = i + 1;
                if j < n && (chars[j].1 == '+' || chars[j].1 == '-') {
                    j += 1;
                }
                if j < n && chars[j].1.is_ascii_digit() {
                    is_float = true;
                    i = j;
                    while i < n && chars[i].1.is_ascii_digit() {
                        i += 1;
                    }
                }
            }
            let text: String = src[start..pos(i)].chars().filter(|&ch| ch != '_').collect();
            let sp = Span::new(file, start, pos(i));
            if is_float {
                Tok::Float(text.parse().map_err(|_| Diag::new("E0001", sp, "invalid number"))?)
            } else {
                Tok::Int(text.parse().map_err(|_| Diag::new("E0001", sp, "integer literal too large for `i64`"))?)
            }
        } else if c == '"' {
            i += 1;
            let mut s = String::new();
            loop {
                if i >= n {
                    return Err(Diag::new("E0001", Span::new(file, start, src.len()), "unterminated string"));
                }
                let ch = chars[i].1;
                if ch == '"' {
                    i += 1;
                    break;
                }
                if ch == '\\' && i + 1 < n {
                    let e = chars[i + 1].1;
                    // `\u{e9}`: a Unicode code point, 1 to 6 hex digits
                    if e == 'u' {
                        let bad = |j: usize| {
                            Diag::new("E0001", Span::new(file, chars[i].0, pos(j)), "invalid escape")
                                .hint("write a Unicode code point as `\\u{...}` with 1 to 6 hex digits, e.g. `\\u{e9}`")
                        };
                        if i + 2 >= n || chars[i + 2].1 != '{' {
                            return Err(bad(i + 2));
                        }
                        let mut j = i + 3;
                        let mut hex = String::new();
                        while j < n && chars[j].1 != '}' && hex.len() < 7 {
                            hex.push(chars[j].1);
                            j += 1;
                        }
                        let c = if j < n && chars[j].1 == '}' && (1..=6).contains(&hex.len()) {
                            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
                        } else {
                            None
                        };
                        let Some(c) = c else { return Err(bad((j + 1).min(n))) };
                        s.push(c);
                        i = j + 1;
                        continue;
                    }
                    s.push(match e {
                        'n' => '\n',
                        't' => '\t',
                        'r' => '\r',
                        '0' => '\0',
                        '"' => '"',
                        '\\' => '\\',
                        _ => {
                            return Err(Diag::new(
                                "E0001",
                                Span::new(file, chars[i].0, pos(i + 2)),
                                format!("unknown escape `\\{e}`"),
                            ));
                        }
                    });
                    i += 2;
                    continue;
                }
                s.push(ch);
                i += 1;
            }
            Tok::Str(s)
        } else {
            let next = if i + 1 < n { chars[i + 1].1 } else { '\0' };
            let next2 = if i + 2 < n { chars[i + 2].1 } else { '\0' };
            let (t, len) = match (c, next) {
                ('-', '>') => (Tok::Arrow, 2),
                ('=', '>') => (Tok::FatArrow, 2),
                ('=', '=') => (Tok::EqEq, 2),
                ('!', '=') => (Tok::NotEq, 2),
                ('<', '=') => (Tok::Le, 2),
                ('>', '=') => (Tok::Ge, 2),
                ('+', '=') => (Tok::PlusAssign, 2),
                ('-', '=') => (Tok::MinusAssign, 2),
                ('*', '=') => (Tok::StarAssign, 2),
                ('/', '=') => (Tok::SlashAssign, 2),
                ('.', '.') => {
                    if next2 == '=' {
                        (Tok::DotDotEq, 3)
                    } else {
                        (Tok::DotDot, 2)
                    }
                }
                ('(', _) => (Tok::LParen, 1),
                (')', _) => (Tok::RParen, 1),
                ('{', _) => (Tok::LBrace, 1),
                ('}', _) => (Tok::RBrace, 1),
                ('[', _) => (Tok::LBracket, 1),
                (']', _) => (Tok::RBracket, 1),
                (',', _) => (Tok::Comma, 1),
                (':', _) => (Tok::Colon, 1),
                (';', _) => (Tok::Semi, 1),
                ('.', _) => (Tok::Dot, 1),
                ('|', _) => (Tok::Pipe, 1),
                ('@', _) => (Tok::At, 1),
                ('?', _) => (Tok::Question, 1),
                ('+', _) => (Tok::Plus, 1),
                ('-', _) => (Tok::Minus, 1),
                ('*', _) => (Tok::Star, 1),
                ('/', _) => (Tok::Slash, 1),
                ('%', _) => (Tok::Percent, 1),
                ('^', _) => (Tok::Caret, 1),
                ('=', _) => (Tok::Assign, 1),
                ('<', _) => (Tok::Lt, 1),
                ('>', _) => (Tok::Gt, 1),
                ('!', _) => {
                    return Err(Diag::new("E0001", Span::new(file, start, start + 1), "unexpected `!`")
                        .hint("Kairós uses `not` for negation and `!=` for inequality"));
                }
                ('&', _) => {
                    return Err(Diag::new("E0001", Span::new(file, start, start + 1), "unexpected `&`")
                        .hint("Kairós uses `and` / `or` for boolean logic"));
                }
                _ => {
                    return Err(Diag::new("E0001", Span::new(file, start, pos(i + 1)), format!("unexpected character `{c}`")));
                }
            };
            i += len;
            t
        };
        toks.push(Token { tok, span: Span::new(file, start, pos(i)), nl_before: nl });
        nl = false;
    }
    toks.push(Token { tok: Tok::Eof, span: Span::new(file, src.len(), src.len()), nl_before: nl });
    Ok(insert_semicolons(toks))
}

/// The escape at chars[i] (a backslash): its character and where what follows starts.
fn escape(src: &str, chars: &[(usize, char)], i: usize, file: u16) -> KResult<(char, usize)> {
    let n = chars.len();
    let pos = |k: usize| if k < n { chars[k].0 } else { src.len() };
    let e = chars[i + 1].1;
    // `\u{e9}`: a Unicode code point, 1 to 6 hex digits
    if e == 'u' {
        let bad = |j: usize| {
            Diag::new("E0001", Span::new(file, chars[i].0, pos(j)), "invalid escape")
                .hint("write a Unicode code point as `\\u{...}` with 1 to 6 hex digits, e.g. `\\u{e9}`")
        };
        if i + 2 >= n || chars[i + 2].1 != '{' {
            return Err(bad(i + 2));
        }
        let mut j = i + 3;
        let mut hex = String::new();
        while j < n && chars[j].1 != '}' && hex.len() < 7 {
            hex.push(chars[j].1);
            j += 1;
        }
        let c = if j < n && chars[j].1 == '}' && (1..=6).contains(&hex.len()) {
            u32::from_str_radix(&hex, 16).ok().and_then(char::from_u32)
        } else {
            None
        };
        let Some(c) = c else { return Err(bad((j + 1).min(n))) };
        return Ok((c, j + 1));
    }
    let c = match e {
        'n' => '\n',
        't' => '\t',
        'r' => '\r',
        '0' => '\0',
        '"' => '"',
        '\\' => '\\',
        _ => return Err(Diag::new("E0001", Span::new(file, chars[i].0, pos(i + 2)), format!("unknown escape `\\{e}`"))),
    };
    Ok((c, i + 2))
}

/// M20: the formatted string at chars[i] (`f"`): its tokens and where what follows starts. The
/// text `a{x}b{y:.2}` becomes `( "a" + str ( x ) + "b" + format_fixed ( y , 2 ) )`: each `{expr}`
/// is the text of the expression, each `{expr:.N}` that of the number with N decimals; `{{` and
/// `}}` are braces. An expression has no quotes and no line breaks.
fn fstring(src: &str, chars: &[(usize, char)], i0: usize, file: u16) -> KResult<(Vec<Token>, usize)> {
    let n = chars.len();
    let pos = |k: usize| if k < n { chars[k].0 } else { src.len() };
    let start = chars[i0].0;
    let tk = |tok: Tok, lo: usize, hi: usize| Token { tok, span: Span::new(file, lo, hi), nl_before: false };
    // the pieces: a text (its characters and bytes), or an expression's tokens and its format
    enum Piece {
        Text(String, usize, usize),
        Expr(Vec<Token>, Option<(i64, usize, usize)>, usize, usize),
    }
    let mut pieces: Vec<Piece> = Vec::new();
    let mut text = String::new();
    let mut text_lo = pos(i0 + 2);
    let mut i = i0 + 2;
    loop {
        if i >= n {
            return Err(Diag::new("E0001", Span::new(file, start, src.len()), "unterminated string"));
        }
        let ch = chars[i].1;
        if ch == '"' {
            break;
        }
        if ch == '\\' && i + 1 < n {
            let (c, next) = escape(src, chars, i, file)?;
            text.push(c);
            i = next;
            continue;
        }
        if (ch == '{' || ch == '}') && i + 1 < n && chars[i + 1].1 == ch {
            text.push(ch);
            i += 2;
            continue;
        }
        if ch == '}' {
            return Err(Diag::new("E0001", Span::new(file, chars[i].0, pos(i + 1)), "single `}` in a formatted string").hint("write `}}` for a brace"));
        }
        if ch != '{' {
            text.push(ch);
            i += 1;
            continue;
        }
        // an expression: to the `}` that closes it
        if !text.is_empty() {
            pieces.push(Piece::Text(std::mem::take(&mut text), text_lo, chars[i].0));
        }
        let open = i;
        let mut j = i + 1;
        let mut depth = 0i32;
        let mut colon: Option<usize> = None;
        while j < n && !(chars[j].1 == '}' && depth == 0) {
            match chars[j].1 {
                '(' | '[' | '{' => depth += 1,
                ')' | ']' | '}' => depth -= 1,
                ':' if depth == 0 => colon = Some(j),
                '"' | '\n' => break,
                _ => {}
            }
            j += 1;
        }
        if j >= n || chars[j].1 != '}' {
            return Err(Diag::new("E0001", Span::new(file, chars[open].0, pos(open + 1)), "unclosed `{` in a formatted string")
                .hint("close it with `}` (an expression has no quotes or line breaks), or write `{{` for a brace"));
        }
        // a format: `:.N` at the end, N decimals
        let mut end = j;
        let mut spec = None;
        if let Some(k) = colon {
            let rest: String = chars[k + 1..j].iter().map(|c| c.1).collect();
            if rest.len() >= 2 && rest.starts_with('.') && rest[1..].chars().all(|c| c.is_ascii_digit()) {
                let Ok(d) = rest[1..].parse::<i64>() else {
                    return Err(Diag::new("E0001", Span::new(file, chars[k].0, chars[j].0), "too many decimals"));
                };
                spec = Some((d, chars[k + 2].0, chars[j].0));
                end = k;
            }
        }
        let (a, b) = (pos(open + 1), pos(end));
        if src[a..b].trim().is_empty() {
            return Err(Diag::new("E0001", Span::new(file, chars[open].0, pos(j + 1)), "empty `{}` in a formatted string").hint("put an expression inside, e.g. `{x}`"));
        }
        let mut inner = lex(&src[a..b], file).map_err(|mut d| {
            d.span.lo += a as u32;
            d.span.hi += a as u32;
            d
        })?;
        inner.pop();
        for t in inner.iter_mut() {
            t.span.lo += a as u32;
            t.span.hi += a as u32;
            t.nl_before = false;
        }
        if inner.iter().any(|t| t.tok == Tok::Semi) {
            return Err(Diag::new("E0001", Span::new(file, chars[open].0, pos(j + 1)), "a `{}` of a formatted string holds one expression"));
        }
        pieces.push(Piece::Expr(inner, spec, chars[open].0, pos(j + 1)));
        i = j + 1;
        text_lo = pos(i);
    }
    let close = chars[i].0;
    if !text.is_empty() {
        pieces.push(Piece::Text(std::mem::take(&mut text), text_lo, close));
    }
    let end = close + 1;
    // no expression: a plain string
    if !pieces.iter().any(|p| matches!(p, Piece::Expr(..))) {
        let s = pieces.into_iter().map(|p| if let Piece::Text(s, _, _) = p { s } else { String::new() }).collect::<String>();
        return Ok((vec![tk(Tok::Str(s), start, end)], i + 1));
    }
    let mut out = vec![tk(Tok::LParen, start, start + 2)];
    for (k, p) in pieces.into_iter().enumerate() {
        let lo = match &p {
            Piece::Text(_, lo, _) => *lo,
            Piece::Expr(_, _, lo, _) => *lo,
        };
        if k > 0 {
            out.push(tk(Tok::Plus, lo, lo));
        }
        match p {
            Piece::Text(s, lo, hi) => out.push(tk(Tok::Str(s), lo, hi)),
            Piece::Expr(ts, spec, lo, hi) => {
                out.push(tk(Tok::Ident(if spec.is_some() { "format_fixed" } else { "str" }.to_string()), lo, lo + 1));
                out.push(tk(Tok::LParen, lo, lo + 1));
                out.extend(ts);
                if let Some((d, dlo, dhi)) = spec {
                    out.push(tk(Tok::Comma, dlo - 1, dlo));
                    out.push(tk(Tok::Int(d), dlo, dhi));
                }
                out.push(tk(Tok::RParen, hi - 1, hi));
            }
        }
    }
    out.push(tk(Tok::RParen, close, end));
    Ok((out, i + 1))
}

fn ends_statement(t: &Tok) -> bool {
    matches!(
        t,
        Tok::Ident(_)
            | Tok::Int(_)
            | Tok::Float(_)
            | Tok::Str(_)
            | Tok::True
            | Tok::False
            | Tok::RParen
            | Tok::RBracket
            | Tok::RBrace
            | Tok::Return
            | Tok::Break
            | Tok::Continue
            | Tok::Question
    )
}

fn insert_semicolons(toks: Vec<Token>) -> Vec<Token> {
    let mut out: Vec<Token> = Vec::with_capacity(toks.len() + toks.len() / 4);
    let mut stack: Vec<char> = Vec::new();
    for t in toks {
        let active = matches!(stack.last(), None | Some('{'));
        if t.nl_before && active && !matches!(t.tok, Tok::RBrace | Tok::Dot) {
            if let Some(prev) = out.last() {
                if ends_statement(&prev.tok) {
                    let sp = Span { lo: prev.span.hi, hi: prev.span.hi, ..prev.span };
                    out.push(Token { tok: Tok::Semi, span: sp, nl_before: false });
                }
            }
        }
        match t.tok {
            Tok::LParen => stack.push('('),
            Tok::LBracket => stack.push('['),
            Tok::LBrace => stack.push('{'),
            Tok::RParen | Tok::RBracket | Tok::RBrace => {
                stack.pop();
            }
            _ => {}
        }
        out.push(t);
    }
    out
}
