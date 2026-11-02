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
