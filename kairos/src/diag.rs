//! Diagnostics: source spans, error codes and rendering with the offending source line.
//!
//! Codes: `E0xxx` compile errors, `R0xxx` runtime errors, `W0xxx` warnings.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub file: u16,
    pub lo: u32,
    pub hi: u32,
}

impl Span {
    pub fn new(file: u16, lo: usize, hi: usize) -> Self {
        Span { file, lo: lo as u32, hi: hi as u32 }
    }

    /// Smallest span covering both.
    pub fn to(self, other: Span) -> Span {
        Span { file: self.file, lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) }
    }
}

#[derive(Debug, Clone)]
pub struct Diag {
    pub code: &'static str,
    pub msg: String,
    pub span: Span,
    pub label: Option<String>,
    pub hint: Option<String>,
}

impl Diag {
    pub fn new(code: &'static str, span: Span, msg: impl Into<String>) -> Self {
        Diag { code, msg: msg.into(), span, label: None, hint: None }
    }

    pub fn label(mut self, l: impl Into<String>) -> Self {
        self.label = Some(l.into());
        self
    }

    pub fn hint(mut self, h: impl Into<String>) -> Self {
        self.hint = Some(h.into());
        self
    }
}

pub type KResult<T> = Result<T, Diag>;

/// Renders a diagnostic like:
/// ```text
/// error[E0101]: mismatched types: expected `f64`, found `str`
///  --> prog.krs:3:14
///   |
/// 3 | let x: f64 = "a"
///   |              ^^^ expected `f64`
/// ```
pub fn render(d: &Diag, src: &str, file: &str) -> String {
    let lo = (d.span.lo as usize).min(src.len());
    let hi = (d.span.hi as usize).clamp(lo, src.len());
    let line_start = src[..lo].rfind('\n').map(|i| i + 1).unwrap_or(0);
    let line_end = src[lo..].find('\n').map(|i| lo + i).unwrap_or(src.len());
    let line_no = src[..lo].matches('\n').count() + 1;
    let col = src[line_start..lo].chars().count() + 1;
    let text = src[line_start..line_end].trim_end_matches('\r');
    let width = src[lo..hi.min(line_end)].chars().count().max(1);
    let pad = " ".repeat(line_no.to_string().len());
    let kind = match d.code.as_bytes().first() {
        Some(b'R') => "runtime error",
        Some(b'W') => "warning",
        _ => "error",
    };
    let mut s = format!(
        "{kind}[{}]: {}\n{pad}--> {file}:{line_no}:{col}\n{pad} |\n{line_no} | {text}\n{pad} | {}{}",
        d.code,
        d.msg,
        " ".repeat(col - 1),
        "^".repeat(width)
    );
    if let Some(l) = &d.label {
        s.push(' ');
        s.push_str(l);
    }
    s.push('\n');
    if let Some(h) = &d.hint {
        s.push_str(&format!("{pad} = hint: {h}\n"));
    }
    s
}
