//! The image of a program written by the front end in Kairós (kairos/compiler/image.krs, M12):
//! its files, axes, structs, enums, protocols, functions, the type of every slot and the HIR of
//! every function, lambda and top-level statement. Loading it rebuilds the checker's tables and
//! the HIR, so the runtime runs the program without the Rust front end (`kairos exec`).

use super::*;

enum Sx {
    Atom(String),
    List(Vec<Sx>),
}

/// One line: an S-expression of atoms (all ASCII, no spaces or parentheses inside).
fn parse_sx(s: &str) -> Result<Sx, String> {
    let b = s.as_bytes();
    let mut stack: Vec<Vec<Sx>> = vec![Vec::new()];
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'(' => {
                stack.push(Vec::new());
                i += 1;
            }
            b')' => {
                let l = stack.pop().ok_or("unbalanced `)`")?;
                stack.last_mut().ok_or("unbalanced `)`")?.push(Sx::List(l));
                i += 1;
            }
            c if c.is_ascii_whitespace() => i += 1,
            _ => {
                let start = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'(' && b[i] != b')' {
                    i += 1;
                }
                stack.last_mut().ok_or("unbalanced `)`")?.push(Sx::Atom(s[start..i].to_string()));
            }
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(mut top), true) if top.len() == 1 => Ok(top.pop().expect("one item")),
        _ => Err("malformed line".into()),
    }
}

fn atom(s: &Sx) -> Result<&str, String> {
    match s {
        Sx::Atom(a) => Ok(a),
        Sx::List(_) => Err("expected an atom".into()),
    }
}

fn items(s: &Sx) -> Result<&[Sx], String> {
    match s {
        Sx::List(l) => Ok(l),
        Sx::Atom(a) => Err(format!("expected a list, found `{a}`")),
    }
}

fn num(s: &Sx) -> Result<i64, String> {
    let a = atom(s)?;
    a.parse().map_err(|_| format!("expected a number, found `{a}`"))
}

fn idx(s: &Sx) -> Result<u32, String> {
    let n = num(s)?;
    u32::try_from(n).map_err(|_| format!("expected an index, found {n}"))
}

/// `x61.62` -> "ab"
fn text(s: &Sx) -> Result<String, String> {
    let a = atom(s)?;
    let hex = a.strip_prefix('x').ok_or_else(|| format!("expected a text, found `{a}`"))?;
    if hex.is_empty() {
        return Ok(String::new());
    }
    hex.split('.')
        .map(|h| u32::from_str_radix(h, 16).ok().and_then(char::from_u32).ok_or_else(|| format!("invalid character `{h}`")))
        .collect()
}

fn at<'a>(l: &'a [Sx], k: usize) -> Result<&'a Sx, String> {
    l.get(k).ok_or_else(|| "missing part".to_string())
}

impl Checker {
    fn image_ty(&mut self, s: &Sx) -> Result<Ty, String> {
        let l = match s {
            Sx::Atom(a) => {
                return Ok(match a.as_str() {
                    "i64" => Ty::Int,
                    "f64" => Ty::Float,
                    "bool" => Ty::Bool,
                    "t3" => Ty::T3,
                    "str" => Ty::Str,
                    "unit" => Ty::Unit,
                    "never" => Ty::Never,
                    "bytes" => Ty::Bytes,
                    "holdout" => Ty::Holdout,
                    other => return Err(format!("unknown type `{other}`")),
                });
            }
            Sx::List(l) => l,
        };
        let head = atom(at(l, 0)?)?;
        Ok(match head {
            "list" => Ty::List(Box::new(self.image_ty(at(l, 1)?)?)),
            "map" => Ty::Map(Box::new(self.image_ty(at(l, 1)?)?), Box::new(self.image_ty(at(l, 2)?)?)),
            "tuple" => Ty::Tuple(l[1..].iter().map(|t| self.image_ty(t)).collect::<Result<_, _>>()?),
            "struct" => Ty::Struct(idx(at(l, 1)?)?),
            "enum" => Ty::Enum(idx(at(l, 1)?)?),
            "fn" => {
                let ps = items(at(l, 1)?)?.iter().map(|t| self.image_ty(t)).collect::<Result<_, _>>()?;
                Ty::Fn(ps, Box::new(self.image_ty(at(l, 2)?)?))
            }
            "param" => Ty::Param(idx(at(l, 1)?)?),
            "tensor" => {
                let axes = items(at(l, 1)?)?.iter().map(idx).collect::<Result<_, _>>()?;
                Ty::Tensor(axes, Box::new(self.image_ty(at(l, 2)?)?))
            }
            // a type inference left open: a variable of its kind
            "var" => {
                let k = match num(at(l, 1)?)? {
                    1 => Kind::Num,
                    2 => Kind::Key,
                    _ => Kind::Any,
                };
                self.fresh(k).0
            }
            other => return Err(format!("unknown type `{other}`")),
        })
    }

    /// Rebuilds the checker's tables and the HIR from an image: the checker, the top-level
    /// statements in the order they run, and the files (for messages).
    pub fn from_image(image: &str) -> Result<(Checker, Vec<TopStmt>, Vec<String>), String> {
        let mut ck = Checker::new();
        ck.axes.clear();
        ck.axis_ids.clear();
        let mut tops = Vec::new();
        let mut files = Vec::new();
        let mut lines = image.lines();
        if lines.next() != Some("kairos-image 1") {
            return Err("not an image of a Kairós program (version 1)".into());
        }
        for (k, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            ck.image_line(line, &mut tops, &mut files).map_err(|e| format!("line {}: {e}", k + 2))?;
        }
        Ok((ck, tops, files))
    }

    fn image_line(&mut self, line: &str, tops: &mut Vec<TopStmt>, files: &mut Vec<String>) -> Result<(), String> {
        let sx = parse_sx(line)?;
        let l = items(&sx)?;
        match atom(at(l, 0)?)? {
            "file" => files.push(text(at(l, 2)?)?),
            "axis" => {
                let name = text(at(l, 1)?)?;
                let temporal = match text(at(l, 2)?)?.as_str() {
                    "" => None,
                    "daily" => Some(Freq::Daily),
                    "monthly" => Some(Freq::Monthly),
                    other => return Err(format!("unknown frequency `{other}`")),
                };
                self.axis_ids.insert(name.clone(), self.axes.len() as u32);
                self.axes.push(AxisDef { name, temporal });
            }
            "tparams" => self.type_param_names = l[1..].iter().map(text).collect::<Result<_, _>>()?,
            "struct" => {
                let name = text(at(l, 1)?)?;
                let mut fields = Vec::new();
                for f in &l[2..] {
                    let fl = items(f)?;
                    fields.push((text(at(fl, 0)?)?, self.image_ty(at(fl, 1)?)?));
                }
                self.struct_ids.insert(name.clone(), self.structs.len() as u32);
                self.structs.push(StructDef { name, fields });
            }
            "enum" => {
                let name = text(at(l, 1)?)?;
                let mut variants = Vec::new();
                for v in &l[2..] {
                    let vl = items(v)?;
                    let ts = vl[1..].iter().map(|t| self.image_ty(t)).collect::<Result<_, _>>()?;
                    variants.push((text(at(vl, 0)?)?, ts));
                }
                self.enum_ids.insert(name.clone(), self.enums.len() as u32);
                self.enums.push(EnumDef { name, variants });
            }
            "protocol" => self.protocols.push(ProtocolDef {
                name: text(at(l, 1)?)?,
                research_until: num(at(l, 2)?)?,
                holdout_from: num(at(l, 3)?)?,
                min_months: num(at(l, 4)?)?,
                rule: text(at(l, 5)?)?,
            }),
            "fn" => {
                let name = text(at(l, 1)?)?;
                let type_params = items(at(l, 2)?)?.iter().map(idx).collect::<Result<_, _>>()?;
                let ret = self.image_ty(at(l, 3)?)?;
                let mut params = Vec::new();
                for p in &l[4..] {
                    let pl = items(p)?;
                    params.push((text(at(pl, 0)?)?, self.image_ty(at(pl, 2)?)?, num(at(pl, 1)?)? == 1));
                }
                self.fn_ids.insert(name.clone(), self.sigs.len() as u32);
                self.sigs.push(FnSig { name, params, ret, type_params });
                self.bodies.push(None);
            }
            "body" => {
                let id = idx(at(l, 1)?)? as usize;
                let n_slots = idx(at(l, 2)?)?;
                let slot_types = items(at(l, 3)?)?.iter().map(|t| self.image_ty(t)).collect::<Result<_, _>>()?;
                let body = expr(at(l, 4)?)?;
                let known_at = match node(at(l, 5)?)? {
                    None => None,
                    Some(n) => Some((expr(at(n.kids, 0)?)?, n.span())),
                };
                let slot = self.bodies.get_mut(id).ok_or_else(|| format!("no function {id}"))?;
                *slot = Some(FnBody { n_slots, slot_types, body, known_at });
            }
            "lambda" => {
                let capture_slots = items(at(l, 4)?)?.iter().map(idx).collect::<Result<_, _>>()?;
                self.lambdas.push(LambdaBody { capture_slots, n_params: idx(at(l, 2)?)?, n_slots: idx(at(l, 3)?)?, body: expr(at(l, 5)?)? });
            }
            "main" => {
                let tys: Vec<Ty> = items(at(l, 2)?)?.iter().map(|t| self.image_ty(t)).collect::<Result<_, _>>()?;
                let mut main = Ctx::new(CtxKind::Main, Ty::Unit);
                for t in &tys {
                    main.new_slot(t);
                }
                self.main = main;
            }
            "top" => tops.push(TopStmt { hir: expr(at(l, 1)?)?, echo: false }),
            other => return Err(format!("unknown item `{other}`")),
        }
        Ok(())
    }
}

/// A HIR node: (KIND TEXT N M LO HI FILE CHILD ...).
struct Node<'a> {
    kind: &'a str,
    text: String,
    n: i64,
    m: i64,
    lo: i64,
    hi: i64,
    file: i64,
    kids: &'a [Sx],
}

impl Node<'_> {
    fn span(&self) -> Span {
        Span::new(u16::try_from(self.file).unwrap_or(0), self.lo.max(0) as usize, self.hi.max(0) as usize)
    }

    fn kid(&self, k: usize) -> Result<&Sx, String> {
        at(self.kids, k)
    }

    fn n(&self) -> u32 {
        self.n.max(0) as u32
    }
}

/// `-` is an absent part.
fn node(s: &Sx) -> Result<Option<Node<'_>>, String> {
    let l = match s {
        Sx::Atom(a) if a == "-" => return Ok(None),
        Sx::Atom(a) => return Err(format!("expected a node, found `{a}`")),
        Sx::List(l) => l,
    };
    Ok(Some(Node {
        kind: atom(at(l, 0)?)?,
        text: text(at(l, 1)?)?,
        n: num(at(l, 2)?)?,
        m: num(at(l, 3)?)?,
        lo: num(at(l, 4)?)?,
        hi: num(at(l, 5)?)?,
        file: num(at(l, 6)?)?,
        kids: &l[7..],
    }))
}

fn value(t: &str) -> Result<Value, String> {
    if t == "unit" {
        return Ok(Value::Unit);
    }
    let (k, v) = t.split_once(' ').ok_or_else(|| format!("invalid literal `{t}`"))?;
    Ok(match k {
        "i64" => Value::Int(v.parse().map_err(|_| format!("invalid integer `{v}`"))?),
        "f64" => Value::Float(v.parse().map_err(|_| format!("invalid number `{v}`"))?),
        "bool" => Value::Bool(v == "true"),
        "str" => {
            let inner = v.strip_prefix("s(").and_then(|x| x.strip_suffix(')')).ok_or_else(|| format!("invalid string `{v}`"))?;
            let s: String = inner
                .split(' ')
                .filter(|h| !h.is_empty())
                .map(|h| u32::from_str_radix(h, 16).ok().and_then(char::from_u32).ok_or_else(|| format!("invalid character `{h}`")))
                .collect::<Result<_, _>>()?;
            Value::Str(s.as_str().into())
        }
        other => return Err(format!("invalid literal `{other}`")),
    })
}

fn binop(op: &str) -> Result<BinOp, String> {
    Ok(match op {
        "+" => BinOp::Add,
        "-" => BinOp::Sub,
        "*" => BinOp::Mul,
        "/" => BinOp::Div,
        "%" => BinOp::Rem,
        "^" => BinOp::Pow,
        "==" => BinOp::Eq,
        "!=" => BinOp::Ne,
        "<" => BinOp::Lt,
        "<=" => BinOp::Le,
        ">" => BinOp::Gt,
        ">=" => BinOp::Ge,
        other => return Err(format!("unknown operator `{other}`")),
    })
}

fn host(t: &str) -> Result<HostFn, String> {
    let mut w = t.split(' ');
    let name = w.next().unwrap_or("");
    let mut arg = || -> Result<u32, String> { w.next().and_then(|x| x.parse().ok()).ok_or_else(|| format!("`{t}` needs a number")) };
    Ok(match name {
        "monthly_returns" => HostFn::MonthlyReturns,
        "eligible" => HostFn::Eligible,
        "cdi" => HostFn::Cdi,
        "ipca" => HostFn::Ipca,
        "months" => HostFn::Months,
        "tickers" => HostFn::Tickers,
        "month_index" => HostFn::MonthIndex,
        "evaluate" => HostFn::Evaluate { result: arg()? },
        "holdout" => HostFn::Holdout { proto: arg()? },
        "open_once" => HostFn::OpenOnce { result: arg()? },
        "daily" => HostFn::Daily { field: arg()? as u8 },
        "results_sha256" => HostFn::ResultsSha256,
        "deflate_trial" => HostFn::DeflateTrial,
        "sessions" => HostFn::Sessions,
        "issuers" => HostFn::Issuers,
        "classes" => HostFn::Classes,
        "nefin" => HostFn::Nefin { factor: arg()? as u8 },
        "nefin_daily" => HostFn::NefinDaily { factor: arg()? as u8 },
        "simulate" => HostFn::Simulate { result: arg()? },
        "record_trial" => HostFn::RecordTrial,
        "session_dates" => HostFn::SessionDates,
        other => return Err(format!("unknown `b3` function `{other}`")),
    })
}

fn bx(e: HExpr) -> Box<HExpr> {
    Box::new(e)
}

fn exprs(xs: &[Sx]) -> Result<Vec<HExpr>, String> {
    xs.iter().map(expr).collect()
}

fn opt(s: &Sx) -> Result<Option<Box<HExpr>>, String> {
    Ok(match node(s)? {
        None => None,
        Some(_) => Some(bx(expr(s)?)),
    })
}

fn place(s: &Sx) -> Result<HPlace, String> {
    let n = node(s)?.ok_or("an absent place")?;
    Ok(match n.kind {
        "plocal" => HPlace::Local(n.n()),
        "pfield" => HPlace::Field(Box::new(place(n.kid(0)?)?), n.n()),
        "pindex" => HPlace::Index(Box::new(place(n.kid(0)?)?), bx(expr(n.kid(1)?)?), n.span()),
        other => return Err(format!("unknown place `{other}`")),
    })
}

fn args(xs: &[Sx]) -> Result<Vec<HArg>, String> {
    xs.iter()
        .map(|s| {
            let n = node(s)?.ok_or("an absent argument")?;
            Ok(match n.kind {
                "val" => HArg::Val(expr(n.kid(0)?)?),
                "inout" => HArg::InOut(place(n.kid(0)?)?),
                other => return Err(format!("unknown argument `{other}`")),
            })
        })
        .collect()
}

fn sels(xs: &[Sx]) -> Result<Vec<(u32, HSel)>, String> {
    xs.iter()
        .map(|s| {
            let n = node(s)?.ok_or("an absent selector")?;
            Ok(match n.kind {
                "at" => (n.n(), HSel::At(expr(n.kid(0)?)?)),
                "range" => (n.n(), HSel::Range(expr(n.kid(0)?)?, expr(n.kid(1)?)?, n.m == 1)),
                other => return Err(format!("unknown selector `{other}`")),
            })
        })
        .collect()
}

fn pat(s: &Sx) -> Result<HPat, String> {
    let n = node(s)?.ok_or("an absent pattern")?;
    Ok(match n.kind {
        "pwild" => HPat::Wild,
        "pbind" => HPat::Bind(n.n()),
        "plit" => HPat::Lit(value(&n.text)?),
        "ptuple" => HPat::Tuple(n.kids.iter().map(pat).collect::<Result<_, _>>()?),
        "pvariant" => HPat::Variant(n.n(), n.m.max(0) as u32, n.kids.iter().map(pat).collect::<Result<_, _>>()?),
        other => return Err(format!("unknown pattern `{other}`")),
    })
}

fn expr(s: &Sx) -> Result<HExpr, String> {
    let n = node(s)?.ok_or("an absent expression")?;
    let sp = n.span();
    let k = |i: usize| n.kid(i);
    Ok(match n.kind {
        "lit" => HExpr::Lit(value(&n.text)?),
        "local" => HExpr::Local(n.n()),
        "fnref" => HExpr::FnRef(n.n()),
        "list" => HExpr::List(exprs(n.kids)?),
        "map" => HExpr::MapLit(exprs(n.kids)?),
        "tuple" => HExpr::Tuple(exprs(n.kids)?),
        "struct" => HExpr::Struct(n.n(), exprs(n.kids)?),
        "variant" => HExpr::Variant(n.n(), n.m.max(0) as u32, exprs(n.kids)?),
        "field" => HExpr::Field(bx(expr(k(0)?)?), n.n()),
        "index" => HExpr::Index(bx(expr(k(0)?)?), bx(expr(k(1)?)?), sp),
        "tindex" => HExpr::TensorIndex(bx(expr(k(0)?)?), sels(n.kids.get(1..).unwrap_or(&[]))?, sp),
        "tassign" => {
            let last = n.kids.len().checked_sub(1).ok_or("an empty store")?;
            let op = if n.text == "=" { None } else { Some(binop(&n.text)?) };
            HExpr::TensorAssign(place(k(0)?)?, sels(n.kids.get(1..last).unwrap_or(&[]))?, op, bx(expr(k(last)?)?), sp)
        }
        "host" => HExpr::Host(host(&n.text)?, exprs(n.kids)?, sp),
        "call" => HExpr::CallFn(n.n(), args(n.kids)?, sp),
        "callv" => HExpr::CallValue(bx(expr(k(0)?)?), exprs(n.kids.get(1..).unwrap_or(&[]))?, sp),
        "builtin" => HExpr::CallBuiltin(builtin_by_name(&n.text).ok_or_else(|| format!("unknown builtin `{}`", n.text))?, args(n.kids)?, sp),
        "unary" => HExpr::Unary(if n.text == "-" { UnOp::Neg } else { UnOp::Not }, bx(expr(k(0)?)?), sp),
        "binary" => HExpr::Binary(binop(&n.text)?, bx(expr(k(0)?)?), bx(expr(k(1)?)?), sp),
        "and" => HExpr::And(bx(expr(k(0)?)?), bx(expr(k(1)?)?)),
        "or" => HExpr::Or(bx(expr(k(0)?)?), bx(expr(k(1)?)?)),
        "assign" => HExpr::Assign(place(k(0)?)?, bx(expr(k(1)?)?)),
        "opassign" => HExpr::OpAssign(place(k(0)?)?, binop(&n.text)?, bx(expr(k(1)?)?), sp),
        "if" => HExpr::If(bx(expr(k(0)?)?), bx(expr(k(1)?)?), opt(k(2)?)?),
        "while" => HExpr::While(bx(expr(k(0)?)?), bx(expr(k(1)?)?)),
        "forrange" => HExpr::ForRange {
            slot: n.n(),
            name: n.text.clone(),
            lo: bx(expr(k(0)?)?),
            hi: bx(expr(k(1)?)?),
            inclusive: n.m == 1,
            body: bx(expr(k(2)?)?),
        },
        "forlist" => HExpr::ForList { pat: pat(k(0)?)?, list: bx(expr(k(1)?)?), body: bx(expr(k(2)?)?) },
        "match" => {
            let mut arms = Vec::new();
            for a in n.kids.get(1..).unwrap_or(&[]) {
                let an = node(a)?.ok_or("an absent arm")?;
                arms.push((pat(an.kid(0)?)?, expr(an.kid(1)?)?));
            }
            HExpr::Match(bx(expr(k(0)?)?), arms, sp)
        }
        "block" => {
            let last = n.kids.len().checked_sub(1).ok_or("a block without its tail")?;
            HExpr::Block(exprs(&n.kids[..last])?, opt(k(last)?)?)
        }
        "let" => HExpr::Let(pat(k(0)?)?, bx(expr(k(1)?)?)),
        "lambda" => {
            let outer = n.text.split_whitespace().map(|x| x.parse().map_err(|_| format!("invalid slot `{x}`"))).collect::<Result<_, _>>()?;
            HExpr::Lambda(n.n(), outer)
        }
        "return" => HExpr::Return(opt(k(0)?)?),
        "break" => HExpr::Break,
        "continue" => HExpr::Continue,
        other => return Err(format!("unknown HIR node `{other}`")),
    })
}
