//! M7: the native back-end. Pure scalar functions are compiled to machine code with Cranelift
//! the first time they are called:
//! - `i64`, `f64` and `bool` parameters, locals and results;
//! - arithmetic, comparisons, `and`/`or`, `if`, `while`, `for`, `break`/`continue`/`return`;
//! - calls between such functions, recursion included;
//! - `sqrt`, `exp`, `ln`, `abs`, `floor`, `ceil`, `round`, `float`, `int`, `min`, `max`.
//!
//! Everything else runs in the interpreter, and so does everything on the targets Cranelift does
//! not support (see build.rs). `KAIROS_JIT=0` turns it off.
//!
//! Same semantics, same bits:
//! - i64 arithmetic is checked (R0006 overflow, R0002 division by zero, R0007 negative power);
//! - `exp`, `ln`, `^` and `round` call the same pure-Rust functions as the interpreter;
//! - comparisons with NaN are false, and `min`/`max` propagate NaN;
//! - nested calls have the interpreter's depth limit (R0005).

use crate::ast::{BinOp, UnOp};
use crate::diag::{Diag, Span};
use crate::hir::{Builtin, HArg, HExpr, HPat, HPlace};
use crate::types::{Checker, Ty};
use crate::value::Value;
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{AbiParam, Block, BlockArg, InstBuilder, MemFlagsData, Signature, Type, UserFuncName, Value as Cv, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Sc {
    I,
    F,
    B,
}

fn sc_of(t: &Ty) -> Option<Sc> {
    match t {
        Ty::Int => Some(Sc::I),
        Ty::Float => Some(Sc::F),
        Ty::Bool => Some(Sc::B),
        _ => None,
    }
}

fn cty(s: Sc) -> Type {
    match s {
        Sc::I => types::I64,
        Sc::F => types::F64,
        Sc::B => types::I8,
    }
}

/// The details of a runtime error, written by native code on its error paths. Native functions
/// also return a failure flag next to their result, and take the number of nested calls they may
/// still make (the interpreter's depth limit) as their last parameter.
#[repr(C)]
struct Ctx {
    /// 0, or the kind of runtime error (E_*)
    err: u64,
    /// index of the error's span in `Jit::spans`
    span: u64,
    /// the f64 of R0009 (its bits)
    aux: u64,
}

const OFF_ERR: i32 = 0;
const OFF_SPAN: i32 = 8;

const E_OVERFLOW: u64 = 1;
const E_DIV0: u64 = 2;
const E_NEGPOW: u64 = 3;
const E_TOINT: u64 = 4;
const E_DEPTH: u64 = 5;

// The pure-Rust math of the interpreter, called from native code.
extern "C" fn h_exp(x: f64) -> f64 {
    libm::exp(x)
}
extern "C" fn h_ln(x: f64) -> f64 {
    libm::log(x)
}
extern "C" fn h_pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}
extern "C" fn h_round(x: f64) -> f64 {
    x.round()
}
extern "C" fn h_fmod(x: f64, y: f64) -> f64 {
    x % y
}
/// The interpreter's integer power: R0007 for a negative exponent, R0006 on overflow.
extern "C" fn h_ipow(x: i64, y: i64, ctx: *mut Ctx, span: u64) -> i64 {
    let fail = |kind: u64| {
        // SAFETY: `ctx` is the live context of the native call in progress
        unsafe {
            (*ctx).err = kind;
            (*ctx).span = span;
        }
        0
    };
    if y < 0 {
        return fail(E_NEGPOW);
    }
    match u32::try_from(y).ok().and_then(|e| x.checked_pow(e)) {
        Some(v) => v,
        None => fail(E_OVERFLOW),
    }
}
/// The interpreter's `int(x)`: R0009 when it does not fit.
extern "C" fn h_toint(x: f64, ctx: *mut Ctx, span: u64) -> i64 {
    if !x.is_finite() || x.abs() >= 9.2e18 {
        // SAFETY: as in `h_ipow`
        unsafe {
            (*ctx).err = E_TOINT;
            (*ctx).span = span;
            (*ctx).aux = x.to_bits();
        }
        return 0;
    }
    x.trunc() as i64
}

/// (one word per argument, the error context, the calls still allowed) -> the result as a word
type Entry = unsafe extern "C" fn(*const u64, *mut Ctx, u64) -> u64;

struct Compiled {
    entry: Entry,
    params: Vec<Sc>,
    ret: Sc,
}

pub struct Jit {
    on: bool,
    module: Option<JITModule>,
    /// declared native functions (the fast signature)
    native: HashMap<u32, FuncId>,
    /// the result of trying each function: None runs in the interpreter
    done: HashMap<u32, Option<Compiled>>,
    helpers: HashMap<&'static str, FuncId>,
    spans: Vec<Span>,
}

impl Default for Jit {
    fn default() -> Self {
        Jit::new()
    }
}

impl Jit {
    /// On unless `KAIROS_JIT=0`.
    pub fn new() -> Jit {
        Jit::with(std::env::var("KAIROS_JIT").map_or(true, |v| v != "0"))
    }

    pub fn with(on: bool) -> Jit {
        Jit { on, module: None, native: HashMap::new(), done: HashMap::new(), helpers: HashMap::new(), spans: Vec::new() }
    }

    /// Whether native code is in use (it is not when turned off or on an unsupported CPU).
    pub fn enabled(&self) -> bool {
        self.on
    }

    /// How many functions run as machine code.
    pub fn compiled(&self) -> usize {
        self.done.values().filter(|c| c.is_some()).count()
    }

    /// Runs function `id` as machine code if it is (or can now be) compiled; None: run it in the
    /// interpreter. `depth` counts this call; nested native calls stop at `max_depth`.
    pub fn call(&mut self, ck: &Checker, id: u32, args: &[Value], depth: u32, max_depth: u32) -> Option<Result<Value, Diag>> {
        if !self.on {
            return None;
        }
        if !self.done.contains_key(&id) {
            self.prepare(ck, id);
        }
        let c = self.done.get(&id)?.as_ref()?;
        if args.len() != c.params.len() {
            return None;
        }
        let mut raw = Vec::with_capacity(args.len());
        for (a, p) in args.iter().zip(&c.params) {
            raw.push(match (a, p) {
                (Value::Int(i), Sc::I) => *i as u64,
                (Value::Float(x), Sc::F) => x.to_bits(),
                (Value::Bool(b), Sc::B) => *b as u64,
                // anything else (a value tracked by `grad`, say) runs in the interpreter
                _ => return None,
            });
        }
        let mut ctx = Ctx { err: 0, span: 0, aux: 0 };
        // this call is at `depth`: it may nest `max_depth - depth` more
        let budget = max_depth.saturating_sub(depth) as u64;
        // SAFETY: the entry was compiled for exactly these parameter kinds, `raw` holds one word
        // per parameter, and `ctx` outlives the call
        let out = unsafe { (c.entry)(raw.as_ptr(), &mut ctx, budget) };
        if ctx.err != 0 {
            let sp = self.spans[ctx.span as usize];
            return Some(Err(match ctx.err {
                E_OVERFLOW => Diag::new("R0006", sp, "integer overflow").hint("i64 arithmetic is checked; use f64 for very large values"),
                E_DIV0 => Diag::new("R0002", sp, "division by zero"),
                E_NEGPOW => Diag::new("R0007", sp, "negative exponent in integer power").hint("use floats: `float(x) ^ float(y)`"),
                E_TOINT => Diag::new("R0009", sp, format!("cannot convert {:?} to i64", f64::from_bits(ctx.aux))),
                _ => Diag::new("R0005", sp, "stack overflow: too many nested calls"),
            }));
        }
        Some(Ok(match c.ret {
            Sc::I => Value::Int(out as i64),
            Sc::F => Value::Float(f64::from_bits(out)),
            Sc::B => Value::Bool(out & 0xff != 0),
        }))
    }

    fn make_module() -> Option<JITModule> {
        let mut flags = settings::builder();
        flags.set("use_colocated_libcalls", "false").ok()?;
        flags.set("is_pic", "false").ok()?;
        flags.set("opt_level", "speed").ok()?;
        let isa = cranelift_native::builder().ok()?.finish(settings::Flags::new(flags)).ok()?;
        let mut b = JITBuilder::with_isa(isa, default_libcall_names());
        let helpers: [(&str, *const u8); 7] = [
            ("kairos_exp", h_exp as *const u8),
            ("kairos_ln", h_ln as *const u8),
            ("kairos_pow", h_pow as *const u8),
            ("kairos_round", h_round as *const u8),
            ("kairos_fmod", h_fmod as *const u8),
            ("kairos_ipow", h_ipow as *const u8),
            ("kairos_toint", h_toint as *const u8),
        ];
        for (name, ptr) in helpers {
            b.symbol(name, ptr);
        }
        Some(JITModule::new(b))
    }

    /// Compiles `id` and every function it calls, or records that it runs in the interpreter.
    fn prepare(&mut self, ck: &Checker, id: u32) {
        let mut group = Vec::new();
        if !self.collect(ck, id, &mut HashSet::new(), &mut group) {
            self.done.insert(id, None);
            return;
        }
        if self.module.is_none() {
            match Self::make_module() {
                Some(m) => self.module = Some(m),
                None => {
                    self.on = false;
                    return;
                }
            }
        }
        if self.compile(ck, &group).is_err() {
            for g in group {
                self.done.entry(g).or_insert(None);
            }
            self.done.insert(id, None);
        }
    }

    /// Whether `id` can be compiled: a scalar signature, scalar locals, supported code, and
    /// callees that can be compiled too. Collects the functions still to compile.
    fn collect(&self, ck: &Checker, id: u32, seen: &mut HashSet<u32>, group: &mut Vec<u32>) -> bool {
        if let Some(c) = self.done.get(&id) {
            return c.is_some();
        }
        if !seen.insert(id) {
            return true;
        }
        let Some(body) = ck.bodies.get(id as usize).and_then(|b| b.as_ref()) else { return false };
        if scalar_sig(ck, id).is_none() || body.slot_types.iter().any(|t| sc_of(t).is_none()) {
            return false;
        }
        let mut calls = Vec::new();
        if !supported(&body.body, &mut calls) {
            return false;
        }
        group.push(id);
        calls.into_iter().all(|c| self.collect(ck, c, seen, group))
    }

    fn compile(&mut self, ck: &Checker, group: &[u32]) -> Result<(), ()> {
        let module = self.module.as_mut().ok_or(())?;
        let tcfg = module.target_config();
        let ptr = tcfg.pointer_type();
        // the helpers, once
        if self.helpers.is_empty() {
            let sig = |ps: &[Type], r: Type| {
                let mut s = module.make_signature();
                s.params.extend(ps.iter().map(|t| AbiParam::new(*t)));
                s.returns.push(AbiParam::new(r));
                s
            };
            let f = types::F64;
            let i = types::I64;
            let decls: [(&'static str, Signature); 7] = [
                ("kairos_exp", sig(&[f], f)),
                ("kairos_ln", sig(&[f], f)),
                ("kairos_pow", sig(&[f, f], f)),
                ("kairos_round", sig(&[f], f)),
                ("kairos_fmod", sig(&[f, f], f)),
                ("kairos_ipow", sig(&[i, i, ptr, i], i)),
                ("kairos_toint", sig(&[f, ptr, i], i)),
            ];
            for (name, s) in decls {
                let fid = module.declare_function(name, Linkage::Import, &s).map_err(|_| ())?;
                self.helpers.insert(name, fid);
            }
        }
        // declare the native functions: (scalars..., context, calls left) -> (result, failed)
        let mut entries = Vec::new();
        for &id in group {
            let (ps, r) = scalar_sig(ck, id).ok_or(())?;
            let mut s = module.make_signature();
            s.params.extend(ps.iter().map(|p| AbiParam::new(cty(*p))));
            s.params.push(AbiParam::new(ptr));
            s.params.push(AbiParam::new(types::I64));
            s.returns.push(AbiParam::new(cty(r)));
            s.returns.push(AbiParam::new(types::I8));
            let fid = module.declare_function(&format!("kairos_fn_{id}"), Linkage::Local, &s).map_err(|_| ())?;
            self.native.insert(id, fid);
            let mut es = module.make_signature();
            es.params.push(AbiParam::new(ptr));
            es.params.push(AbiParam::new(ptr));
            es.params.push(AbiParam::new(types::I64));
            es.returns.push(AbiParam::new(types::I64));
            let eid = module.declare_function(&format!("kairos_entry_{id}"), Linkage::Local, &es).map_err(|_| ())?;
            entries.push((id, fid, eid, ps, r, s, es));
        }
        let mut fctx = FunctionBuilderContext::new();
        let mut cctx = module.make_context();
        for (id, fid, eid, ps, r, s, es) in &entries {
            // the function itself
            cctx.func.signature = s.clone();
            cctx.func.name = UserFuncName::user(0, fid.as_u32());
            {
                let body = ck.bodies[*id as usize].as_ref().ok_or(())?;
                let mut b = FunctionBuilder::new(&mut cctx.func, &mut fctx);
                let entry = b.create_block();
                b.append_block_params_for_function_params(entry);
                b.switch_to_block(entry);
                let params: Vec<Cv> = b.block_params(entry).to_vec();
                let mut vars = Vec::with_capacity(body.slot_types.len());
                for (slot, t) in body.slot_types.iter().enumerate() {
                    let sc = sc_of(t).ok_or(())?;
                    let v = b.declare_var(cty(sc));
                    let init = if slot < ps.len() { params[slot] } else { zero(&mut b, sc) };
                    b.def_var(v, init);
                    vars.push((v, sc));
                }
                let ret = b.create_block();
                b.append_block_param(ret, cty(*r));
                b.append_block_param(ret, types::I8);
                let mut g = Gen {
                    m: module,
                    b,
                    vars,
                    ctx: params[ps.len()],
                    budget: params[ps.len() + 1],
                    ret,
                    ret_sc: *r,
                    loops: Vec::new(),
                    native: &self.native,
                    helpers: &self.helpers,
                    spans: &mut self.spans,
                };
                let v = g.expr(&body.body)?;
                let out = match v {
                    V::Val(x, sc) if sc == *r => x,
                    V::Never => zero(&mut g.b, *r),
                    _ => return Err(()),
                };
                let ok = g.b.ins().iconst(types::I8, 0);
                g.b.ins().jump(ret, &[out.into(), ok.into()]);
                g.b.switch_to_block(ret);
                let (rv, failed) = (g.b.block_params(ret)[0], g.b.block_params(ret)[1]);
                g.b.ins().return_(&[rv, failed]);
                g.b.seal_all_blocks();
                g.b.finalize(tcfg);
            }
            module.define_function(*fid, &mut cctx).map_err(|_| ())?;
            module.clear_context(&mut cctx);
            // its entry from the interpreter: words in, a word out
            cctx.func.signature = es.clone();
            cctx.func.name = UserFuncName::user(0, eid.as_u32());
            {
                let mut b = FunctionBuilder::new(&mut cctx.func, &mut fctx);
                let blk = b.create_block();
                b.append_block_params_for_function_params(blk);
                b.switch_to_block(blk);
                let (argp, ctxp, budget) = (b.block_params(blk)[0], b.block_params(blk)[1], b.block_params(blk)[2]);
                let mut call_args = Vec::new();
                for (k, p) in ps.iter().enumerate() {
                    let w = b.ins().load(types::I64, MemFlagsData::trusted(), argp, (8 * k) as i32);
                    call_args.push(match p {
                        Sc::I => w,
                        Sc::F => b.ins().bitcast(types::F64, MemFlagsData::new(), w),
                        Sc::B => b.ins().ireduce(types::I8, w),
                    });
                }
                call_args.push(ctxp);
                call_args.push(budget);
                let callee = module.declare_func_in_func(*fid, b.func);
                let call = b.ins().call(callee, &call_args);
                let res = b.inst_results(call)[0];
                let word = match r {
                    Sc::I => res,
                    Sc::F => b.ins().bitcast(types::I64, MemFlagsData::new(), res),
                    Sc::B => b.ins().uextend(types::I64, res),
                };
                b.ins().return_(&[word]);
                b.seal_all_blocks();
                b.finalize(tcfg);
            }
            module.define_function(*eid, &mut cctx).map_err(|_| ())?;
            module.clear_context(&mut cctx);
        }
        module.finalize_definitions().map_err(|_| ())?;
        for (id, _, eid, ps, r, _, _) in entries {
            let code = module.get_finalized_function(eid);
            // SAFETY: `code` is the entry just compiled with the signature of `Entry`
            let entry = unsafe { std::mem::transmute::<*const u8, Entry>(code) };
            self.done.insert(id, Some(Compiled { entry, params: ps, ret: r }));
        }
        Ok(())
    }
}

fn scalar_sig(ck: &Checker, id: u32) -> Option<(Vec<Sc>, Sc)> {
    let sig = ck.sigs.get(id as usize)?;
    if !sig.type_params.is_empty() {
        return None;
    }
    let mut ps = Vec::new();
    for (_, t, inout) in &sig.params {
        if *inout {
            return None;
        }
        ps.push(sc_of(t)?);
    }
    Some((ps, sc_of(&sig.ret)?))
}

/// The code the native back-end handles (types are checked again while generating it).
fn supported(e: &HExpr, calls: &mut Vec<u32>) -> bool {
    use HExpr as E;
    let val = |a: &HArg, calls: &mut Vec<u32>| matches!(a, HArg::Val(x) if supported(x, calls));
    match e {
        E::Lit(v) => matches!(v, Value::Int(_) | Value::Float(_) | Value::Bool(_) | Value::Unit),
        E::Local(_) | E::Break | E::Continue => true,
        E::Unary(_, x, _) => supported(x, calls),
        E::Binary(_, a, b, _) | E::And(a, b) | E::Or(a, b) | E::While(a, b) => supported(a, calls) && supported(b, calls),
        E::Assign(HPlace::Local(_), x) | E::OpAssign(HPlace::Local(_), _, x, _) => supported(x, calls),
        E::If(c, t, e) => supported(c, calls) && supported(t, calls) && e.as_ref().is_none_or(|x| supported(x, calls)),
        E::ForRange { lo, hi, body, .. } => supported(lo, calls) && supported(hi, calls) && supported(body, calls),
        E::Block(st, tail) => st.iter().all(|s| supported(s, calls)) && tail.as_ref().is_none_or(|t| supported(t, calls)),
        E::Let(HPat::Bind(_) | HPat::Wild, x) => supported(x, calls),
        E::Return(x) => x.as_ref().is_none_or(|x| supported(x, calls)),
        E::CallFn(id, args, _) => {
            calls.push(*id);
            args.iter().all(|a| val(a, calls))
        }
        E::CallBuiltin(b, args, _) => {
            let n = match b {
                Builtin::Sqrt
                | Builtin::Exp
                | Builtin::Ln
                | Builtin::Abs
                | Builtin::Floor
                | Builtin::Ceil
                | Builtin::Round
                | Builtin::ToFloat
                | Builtin::ToInt => 1,
                Builtin::Min | Builtin::Max => 2,
                _ => return false,
            };
            args.len() == n && args.iter().all(|a| val(a, calls))
        }
        _ => false,
    }
}

fn zero(b: &mut FunctionBuilder, s: Sc) -> Cv {
    match s {
        Sc::I => b.ins().iconst(types::I64, 0),
        Sc::F => b.ins().f64const(0.0),
        Sc::B => b.ins().iconst(types::I8, 0),
    }
}

/// What an expression produced.
#[derive(Clone, Copy)]
enum V {
    Unit,
    Val(Cv, Sc),
    /// control left (return, break, continue): what follows is unreachable
    Never,
}

struct Gen<'a, 'b> {
    m: &'a mut JITModule,
    b: FunctionBuilder<'b>,
    vars: Vec<(Variable, Sc)>,
    ctx: Cv,
    /// nested calls this function may still make
    budget: Cv,
    /// (result, failed)
    ret: Block,
    ret_sc: Sc,
    /// (continue target, break target)
    loops: Vec<(Block, Block)>,
    native: &'a HashMap<u32, FuncId>,
    helpers: &'a HashMap<&'static str, FuncId>,
    spans: &'a mut Vec<Span>,
}

type G<T> = Result<T, ()>;

impl Gen<'_, '_> {
    /// Continues in a new block that nothing jumps to (after return/break/continue).
    fn dead(&mut self) {
        let blk = self.b.create_block();
        self.b.switch_to_block(blk);
    }

    fn span(&mut self, sp: Span) -> i64 {
        self.spans.push(sp);
        (self.spans.len() - 1) as i64
    }

    /// Leaves with the error if `bad` is true.
    fn fail_if(&mut self, bad: Cv, kind: u64, sp: Span) {
        let err = self.b.create_block();
        let ok = self.b.create_block();
        self.b.ins().brif(bad, err, &[], ok, &[]);
        self.b.switch_to_block(err);
        let k = self.b.ins().iconst(types::I64, kind as i64);
        self.b.ins().store(MemFlagsData::trusted(), k, self.ctx, OFF_ERR);
        let si = self.span(sp);
        let s = self.b.ins().iconst(types::I64, si);
        self.b.ins().store(MemFlagsData::trusted(), s, self.ctx, OFF_SPAN);
        self.leave_failed();
        self.b.switch_to_block(ok);
    }

    /// Returns with the failure flag set (the context already holds the error).
    fn leave_failed(&mut self) {
        let z = zero(&mut self.b, self.ret_sc);
        let one = self.b.ins().iconst(types::I8, 1);
        self.b.ins().jump(self.ret, &[z.into(), one.into()]);
    }

    /// Leaves if `failed` (a callee's flag) is set.
    fn propagate(&mut self, failed: Cv) {
        let leave = self.b.create_block();
        let ok = self.b.create_block();
        self.b.ins().brif(failed, leave, &[], ok, &[]);
        self.b.switch_to_block(leave);
        self.leave_failed();
        self.b.switch_to_block(ok);
    }

    /// After a helper that reports errors in the context.
    fn propagate_ctx(&mut self) {
        let e = self.b.ins().load(types::I64, MemFlagsData::trusted(), self.ctx, OFF_ERR);
        self.propagate(e);
    }

    /// The value of an integer constant, if `v` is one.
    fn konst(&self, v: Cv) -> Option<i64> {
        let cranelift_codegen::ir::ValueDef::Result(inst, _) = self.b.func.dfg.value_def(v) else { return None };
        match self.b.func.dfg.insts[inst] {
            cranelift_codegen::ir::InstructionData::UnaryImm { opcode: cranelift_codegen::ir::Opcode::Iconst, imm } => Some(imm.bits()),
            _ => None,
        }
    }

    fn helper(&mut self, name: &str, args: &[Cv]) -> G<Cv> {
        let fid = *self.helpers.get(name).ok_or(())?;
        let f = self.m.declare_func_in_func(fid, self.b.func);
        let call = self.b.ins().call(f, args);
        Ok(self.b.inst_results(call)[0])
    }

    /// Branches to `yes` or `no` on a condition. `and`, `or` and `not` become branches instead
    /// of a bool that is then tested again.
    fn cond(&mut self, e: &HExpr, yes: Block, no: Block) -> G<()> {
        match e {
            HExpr::And(a, b) => {
                let mid = self.b.create_block();
                self.cond(a, mid, no)?;
                self.b.switch_to_block(mid);
                self.cond(b, yes, no)
            }
            HExpr::Or(a, b) => {
                let mid = self.b.create_block();
                self.cond(a, yes, mid)?;
                self.b.switch_to_block(mid);
                self.cond(b, yes, no)
            }
            HExpr::Unary(UnOp::Not, x, _) => self.cond(x, no, yes),
            _ => match self.expr(e)? {
                V::Val(v, Sc::B) => {
                    self.b.ins().brif(v, yes, &[], no, &[]);
                    Ok(())
                }
                V::Never => {
                    self.b.ins().jump(no, &[]);
                    Ok(())
                }
                _ => Err(()),
            },
        }
    }

    /// Ends a branch: jumps to `merge` with its value (a placeholder if it never gets there).
    fn join(&mut self, merge: Block, ty: Option<Sc>, v: V) -> G<()> {
        match (ty, v) {
            (Some(s), V::Val(x, xs)) => {
                if xs != s {
                    return Err(());
                }
                self.b.ins().jump(merge, &[x.into()]);
            }
            (Some(s), V::Never) => {
                let z = zero(&mut self.b, s);
                self.b.ins().jump(merge, &[z.into()]);
            }
            (Some(_), V::Unit) => return Err(()),
            (None, _) => {
                self.b.ins().jump(merge, &[]);
            }
        }
        Ok(())
    }

    /// The scalar an expression produces, if any (without generating code).
    fn sty(&self, e: &HExpr) -> Option<Sc> {
        use HExpr as E;
        match e {
            E::Lit(Value::Int(_)) => Some(Sc::I),
            E::Lit(Value::Float(_)) => Some(Sc::F),
            E::Lit(Value::Bool(_)) => Some(Sc::B),
            E::Local(s) => self.vars.get(*s as usize).map(|v| v.1),
            E::Block(_, Some(t)) => self.sty(t),
            E::Unary(UnOp::Not, ..) | E::And(..) | E::Or(..) => Some(Sc::B),
            E::Unary(_, x, _) => self.sty(x),
            E::Binary(op, a, _, _) => match op {
                BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => Some(Sc::B),
                _ => self.sty(a),
            },
            E::If(_, t, Some(x)) => self.sty(t).or_else(|| self.sty(x)),
            E::CallFn(id, _, _) => {
                let fid = self.native.get(id)?;
                let t = self.m.declarations().get_function_decl(*fid).signature.returns.first()?.value_type;
                Some(if t == types::I64 {
                    Sc::I
                } else if t == types::F64 {
                    Sc::F
                } else {
                    Sc::B
                })
            }
            E::CallBuiltin(b, args, _) => match b {
                Builtin::Sqrt | Builtin::Exp | Builtin::Ln | Builtin::Floor | Builtin::Ceil | Builtin::Round | Builtin::ToFloat => Some(Sc::F),
                Builtin::ToInt => Some(Sc::I),
                _ => match args.first() {
                    Some(HArg::Val(x)) => self.sty(x),
                    _ => None,
                },
            },
            _ => None,
        }
    }

    fn val(&mut self, e: &HExpr) -> G<(Cv, Sc)> {
        match self.expr(e)? {
            V::Val(v, s) => Ok((v, s)),
            _ => Err(()),
        }
    }

    fn expr(&mut self, e: &HExpr) -> G<V> {
        use HExpr as E;
        Ok(match e {
            E::Lit(Value::Int(i)) => V::Val(self.b.ins().iconst(types::I64, *i), Sc::I),
            E::Lit(Value::Float(x)) => V::Val(self.b.ins().f64const(*x), Sc::F),
            E::Lit(Value::Bool(x)) => V::Val(self.b.ins().iconst(types::I8, *x as i64), Sc::B),
            E::Lit(Value::Unit) => V::Unit,
            E::Local(s) => {
                let (var, sc) = *self.vars.get(*s as usize).ok_or(())?;
                V::Val(self.b.use_var(var), sc)
            }
            E::Block(stmts, tail) => {
                let mut never = false;
                for s in stmts {
                    if let V::Never = self.expr(s)? {
                        never = true;
                    }
                }
                let t = match tail {
                    Some(t) => self.expr(t)?,
                    None => V::Unit,
                };
                if never { V::Never } else { t }
            }
            E::Let(HPat::Bind(s), x) | E::Assign(HPlace::Local(s), x) => match self.expr(x)? {
                V::Val(v, sc) => {
                    let (var, want) = *self.vars.get(*s as usize).ok_or(())?;
                    if sc != want {
                        return Err(());
                    }
                    self.b.def_var(var, v);
                    V::Unit
                }
                V::Never => V::Never,
                V::Unit => return Err(()),
            },
            E::Let(HPat::Wild, x) => match self.expr(x)? {
                V::Never => V::Never,
                _ => V::Unit,
            },
            E::OpAssign(HPlace::Local(s), op, x, sp) => {
                let (var, sc) = *self.vars.get(*s as usize).ok_or(())?;
                let cur = self.b.use_var(var);
                let (r, rs) = self.val(x)?;
                let (v, vs) = self.binary(*op, cur, sc, r, rs, *sp)?;
                if vs != sc {
                    return Err(());
                }
                self.b.def_var(var, v);
                V::Unit
            }
            E::Unary(op, x, sp) => {
                let (v, s) = self.val(x)?;
                match (op, s) {
                    (UnOp::Neg, Sc::I) => {
                        let z = self.b.ins().iconst(types::I64, 0);
                        let (r, of) = self.b.ins().ssub_overflow(z, v);
                        self.fail_if(of, E_OVERFLOW, *sp);
                        V::Val(r, Sc::I)
                    }
                    (UnOp::Neg, Sc::F) => V::Val(self.b.ins().fneg(v), Sc::F),
                    (UnOp::Not, Sc::B) => V::Val(self.b.ins().icmp_imm_s(IntCC::Equal, v, 0), Sc::B),
                    _ => return Err(()),
                }
            }
            // x % 2^k == 0: a test of the low bits (the same answer for negative x)
            E::Binary(op @ (BinOp::Eq | BinOp::Ne), a, b, _)
                if matches!(&**b, E::Lit(Value::Int(0)))
                    && matches!(&**a, E::Binary(BinOp::Rem, _, d, _) if matches!(&**d, E::Lit(Value::Int(c)) if *c > 0 && c & (c - 1) == 0)) =>
            {
                let E::Binary(_, x, d, _) = &**a else { return Err(()) };
                let E::Lit(Value::Int(c)) = &**d else { return Err(()) };
                let (v, vs) = self.val(x)?;
                if vs != Sc::I {
                    return Err(());
                }
                let mask = self.b.ins().iconst(types::I64, c - 1);
                let low = self.b.ins().band(v, mask);
                let cc = if *op == BinOp::Eq { IntCC::Equal } else { IntCC::NotEqual };
                V::Val(self.b.ins().icmp_imm_s(cc, low, 0), Sc::B)
            }
            E::Binary(op, a, b, sp) => {
                let (x, xs) = self.val(a)?;
                let (y, ys) = self.val(b)?;
                let (v, s) = self.binary(*op, x, xs, y, ys, *sp)?;
                V::Val(v, s)
            }
            E::And(a, b) | E::Or(a, b) => {
                let is_and = matches!(e, E::And(..));
                let (x, xs) = self.val(a)?;
                if xs != Sc::B {
                    return Err(());
                }
                let rhs = self.b.create_block();
                let merge = self.b.create_block();
                self.b.append_block_param(merge, types::I8);
                let short = self.b.ins().iconst(types::I8, if is_and { 0 } else { 1 });
                let sarg: [BlockArg; 1] = [short.into()];
                if is_and {
                    self.b.ins().brif(x, rhs, &[], merge, &sarg);
                } else {
                    self.b.ins().brif(x, merge, &sarg, rhs, &[]);
                }
                self.b.switch_to_block(rhs);
                let y = match self.expr(b)? {
                    V::Val(y, Sc::B) => y,
                    V::Never => zero(&mut self.b, Sc::B),
                    _ => return Err(()),
                };
                self.b.ins().jump(merge, &[y.into()]);
                self.b.switch_to_block(merge);
                V::Val(self.b.block_params(merge)[0], Sc::B)
            }
            E::If(c, t, els) => {
                // the value of the `if` (a block must be finished before the next one starts, so
                // the merge's parameter is decided first)
                let ty = match els {
                    Some(x) => self.sty(t).or_else(|| self.sty(x)),
                    None => None,
                };
                let then_b = self.b.create_block();
                let else_b = self.b.create_block();
                let merge = self.b.create_block();
                if let Some(s) = ty {
                    self.b.append_block_param(merge, cty(s));
                }
                self.cond(c, then_b, else_b)?;
                self.b.switch_to_block(then_b);
                let tv = self.expr(t)?;
                self.join(merge, ty, tv)?;
                self.b.switch_to_block(else_b);
                let ev = match els {
                    Some(x) => self.expr(x)?,
                    None => V::Unit,
                };
                self.join(merge, ty, ev)?;
                self.b.switch_to_block(merge);
                match (ty, tv, ev) {
                    (_, V::Never, V::Never) => V::Never,
                    (Some(s), _, _) => V::Val(self.b.block_params(merge)[0], s),
                    (None, _, _) => V::Unit,
                }
            }
            E::While(c, body) => {
                let head = self.b.create_block();
                let inner = self.b.create_block();
                let exit = self.b.create_block();
                self.b.ins().jump(head, &[]);
                self.b.switch_to_block(head);
                self.cond(c, inner, exit)?;
                self.b.switch_to_block(inner);
                self.loops.push((head, exit));
                self.expr(body)?;
                self.loops.pop();
                self.b.ins().jump(head, &[]);
                self.b.switch_to_block(exit);
                V::Unit
            }
            E::ForRange { slot, lo, hi, inclusive, body, .. } => {
                let (l, ls) = self.val(lo)?;
                let (h, hs) = self.val(hi)?;
                if ls != Sc::I || hs != Sc::I {
                    return Err(());
                }
                let (var, vs) = *self.vars.get(*slot as usize).ok_or(())?;
                if vs != Sc::I {
                    return Err(());
                }
                let i = self.b.declare_var(types::I64);
                self.b.def_var(i, l);
                let head = self.b.create_block();
                let inner = self.b.create_block();
                let step = self.b.create_block();
                let next = self.b.create_block();
                let exit = self.b.create_block();
                self.b.ins().jump(head, &[]);
                self.b.switch_to_block(head);
                let iv = self.b.use_var(i);
                let cc = if *inclusive { IntCC::SignedLessThanOrEqual } else { IntCC::SignedLessThan };
                let go = self.b.ins().icmp(cc, iv, h);
                self.b.ins().brif(go, inner, &[], exit, &[]);
                self.b.switch_to_block(inner);
                let iv = self.b.use_var(i);
                self.b.def_var(var, iv);
                self.loops.push((step, exit));
                self.expr(body)?;
                self.loops.pop();
                self.b.ins().jump(step, &[]);
                // i + 1, and stop instead of overflowing (as the interpreter does)
                self.b.switch_to_block(step);
                let iv = self.b.use_var(i);
                let one = self.b.ins().iconst(types::I64, 1);
                let (n, of) = self.b.ins().sadd_overflow(iv, one);
                self.b.ins().brif(of, exit, &[], next, &[]);
                self.b.switch_to_block(next);
                self.b.def_var(i, n);
                self.b.ins().jump(head, &[]);
                self.b.switch_to_block(exit);
                V::Unit
            }
            E::Break | E::Continue => {
                let &(cont, brk) = self.loops.last().ok_or(())?;
                let target = if matches!(e, E::Break) { brk } else { cont };
                self.b.ins().jump(target, &[]);
                self.dead();
                V::Never
            }
            E::Return(x) => {
                let v = match x {
                    Some(x) => match self.expr(x)? {
                        V::Val(v, s) if s == self.ret_sc => v,
                        V::Never => return Ok(V::Never),
                        _ => return Err(()),
                    },
                    None => return Err(()),
                };
                let ok = self.b.ins().iconst(types::I8, 0);
                self.b.ins().jump(self.ret, &[v.into(), ok.into()]);
                self.dead();
                V::Never
            }
            E::CallFn(id, args, sp) => {
                let mut vals = Vec::with_capacity(args.len() + 1);
                for a in args {
                    let HArg::Val(x) = a else { return Err(()) };
                    match self.expr(x)? {
                        V::Val(v, _) => vals.push(v),
                        V::Never => return Ok(V::Never),
                        V::Unit => return Err(()),
                    }
                }
                let fid = *self.native.get(id).ok_or(())?;
                // one more nested call, within the interpreter's limit
                let deep = self.b.ins().icmp_imm_s(IntCC::Equal, self.budget, 0);
                self.fail_if(deep, E_DEPTH, *sp);
                let left = self.b.ins().iadd_imm_s(self.budget, -1);
                vals.push(self.ctx);
                vals.push(left);
                let f = self.m.declare_func_in_func(fid, self.b.func);
                let call = self.b.ins().call(f, &vals);
                let (r, failed) = (self.b.inst_results(call)[0], self.b.inst_results(call)[1]);
                self.propagate(failed);
                let ty = self.b.func.dfg.value_type(r);
                let sc = if ty == types::I64 {
                    Sc::I
                } else if ty == types::F64 {
                    Sc::F
                } else {
                    Sc::B
                };
                V::Val(r, sc)
            }
            E::CallBuiltin(bi, args, sp) => {
                let mut xs = Vec::with_capacity(2);
                for a in args {
                    let HArg::Val(x) = a else { return Err(()) };
                    match self.expr(x)? {
                        V::Val(v, s) => xs.push((v, s)),
                        V::Never => return Ok(V::Never),
                        V::Unit => return Err(()),
                    }
                }
                let (v, s) = self.builtin(*bi, &xs, *sp)?;
                V::Val(v, s)
            }
            _ => return Err(()),
        })
    }

    fn builtin(&mut self, b: Builtin, xs: &[(Cv, Sc)], sp: Span) -> G<(Cv, Sc)> {
        let (x, s) = xs[0];
        Ok(match (b, s) {
            (Builtin::Sqrt, Sc::F) => (self.b.ins().sqrt(x), Sc::F),
            (Builtin::Exp, Sc::F) => (self.helper("kairos_exp", &[x])?, Sc::F),
            (Builtin::Ln, Sc::F) => (self.helper("kairos_ln", &[x])?, Sc::F),
            (Builtin::Floor, Sc::F) => (self.b.ins().floor(x), Sc::F),
            (Builtin::Ceil, Sc::F) => (self.b.ins().ceil(x), Sc::F),
            (Builtin::Round, Sc::F) => (self.helper("kairos_round", &[x])?, Sc::F),
            (Builtin::Abs, Sc::F) => (self.b.ins().fabs(x), Sc::F),
            (Builtin::Abs, Sc::I) => {
                let min = self.b.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                self.fail_if(min, E_OVERFLOW, sp);
                let neg = self.b.ins().ineg(x);
                let lt = self.b.ins().icmp_imm_s(IntCC::SignedLessThan, x, 0);
                (self.b.ins().select(lt, neg, x), Sc::I)
            }
            (Builtin::ToFloat, Sc::I) => (self.b.ins().fcvt_from_sint(types::F64, x), Sc::F),
            (Builtin::ToFloat, Sc::F) | (Builtin::ToInt, Sc::I) => (x, s),
            (Builtin::ToInt, Sc::F) => {
                let si = self.span(sp);
                let spv = self.b.ins().iconst(types::I64, si);
                let r = self.helper("kairos_toint", &[x, self.ctx, spv])?;
                self.propagate_ctx();
                (r, Sc::I)
            }
            (Builtin::Min | Builtin::Max, _) => {
                let (y, ys) = xs[1];
                if ys != s {
                    return Err(());
                }
                // the interpreter: NaN if either is NaN; else min = a < b ? a : b, max = a < b ? b : a
                let less = match s {
                    Sc::I => self.b.ins().icmp(IntCC::SignedLessThan, x, y),
                    Sc::F => self.b.ins().fcmp(FloatCC::LessThan, x, y),
                    Sc::B => return Err(()),
                };
                let pick = if b == Builtin::Min { self.b.ins().select(less, x, y) } else { self.b.ins().select(less, y, x) };
                if s == Sc::F {
                    let nan = self.b.ins().fcmp(FloatCC::Unordered, x, y);
                    let q = self.b.ins().f64const(f64::NAN);
                    (self.b.ins().select(nan, q, pick), Sc::F)
                } else {
                    (pick, Sc::I)
                }
            }
            _ => return Err(()),
        })
    }

    fn binary(&mut self, op: BinOp, x: Cv, xs: Sc, y: Cv, ys: Sc, sp: Span) -> G<(Cv, Sc)> {
        if xs != ys {
            return Err(());
        }
        let cmp_i = |op: BinOp| match op {
            BinOp::Eq => Some(IntCC::Equal),
            BinOp::Ne => Some(IntCC::NotEqual),
            BinOp::Lt => Some(IntCC::SignedLessThan),
            BinOp::Le => Some(IntCC::SignedLessThanOrEqual),
            BinOp::Gt => Some(IntCC::SignedGreaterThan),
            BinOp::Ge => Some(IntCC::SignedGreaterThanOrEqual),
            _ => None,
        };
        Ok(match xs {
            Sc::I => {
                if let Some(cc) = cmp_i(op) {
                    return Ok((self.b.ins().icmp(cc, x, y), Sc::B));
                }
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul => {
                        let (r, of) = match op {
                            BinOp::Add => self.b.ins().sadd_overflow(x, y),
                            BinOp::Sub => self.b.ins().ssub_overflow(x, y),
                            _ => self.b.ins().smul_overflow(x, y),
                        };
                        self.fail_if(of, E_OVERFLOW, sp);
                        (r, Sc::I)
                    }
                    BinOp::Div | BinOp::Rem if self.konst(y).is_some_and(|c| c > 0 && c & (c - 1) == 0) => {
                        // by 2^k: truncate toward zero with shifts, as the hardware division would
                        let k = self.konst(y).map_or(0, |c| c.trailing_zeros()) as i64;
                        let q = if k == 0 {
                            x
                        } else {
                            let sign = self.b.ins().sshr_imm_u(x, 63);
                            let bias = self.b.ins().ushr_imm_u(sign, 64 - k);
                            let t = self.b.ins().iadd(x, bias);
                            self.b.ins().sshr_imm_u(t, k)
                        };
                        if op == BinOp::Div {
                            (q, Sc::I)
                        } else {
                            let m = self.b.ins().ishl_imm_u(q, k);
                            (self.b.ins().isub(x, m), Sc::I)
                        }
                    }
                    BinOp::Div | BinOp::Rem if self.konst(y).is_some_and(|c| c != 0 && c != -1) => {
                        let r = if op == BinOp::Div { self.b.ins().sdiv(x, y) } else { self.b.ins().srem(x, y) };
                        (r, Sc::I)
                    }
                    BinOp::Div | BinOp::Rem => {
                        let z = self.b.ins().icmp_imm_s(IntCC::Equal, y, 0);
                        self.fail_if(z, E_DIV0, sp);
                        let a = self.b.ins().icmp_imm_s(IntCC::Equal, x, i64::MIN);
                        let c = self.b.ins().icmp_imm_s(IntCC::Equal, y, -1);
                        let both = self.b.ins().band(a, c);
                        self.fail_if(both, E_OVERFLOW, sp);
                        let r = if op == BinOp::Div { self.b.ins().sdiv(x, y) } else { self.b.ins().srem(x, y) };
                        (r, Sc::I)
                    }
                    BinOp::Pow => {
                        let si = self.span(sp);
                        let spv = self.b.ins().iconst(types::I64, si);
                        let r = self.helper("kairos_ipow", &[x, y, self.ctx, spv])?;
                        self.propagate_ctx();
                        (r, Sc::I)
                    }
                    _ => return Err(()),
                }
            }
            Sc::F => {
                let cc = match op {
                    BinOp::Eq => Some(FloatCC::Equal),
                    BinOp::Ne => Some(FloatCC::NotEqual),
                    BinOp::Lt => Some(FloatCC::LessThan),
                    BinOp::Le => Some(FloatCC::LessThanOrEqual),
                    BinOp::Gt => Some(FloatCC::GreaterThan),
                    BinOp::Ge => Some(FloatCC::GreaterThanOrEqual),
                    _ => None,
                };
                if let Some(cc) = cc {
                    return Ok((self.b.ins().fcmp(cc, x, y), Sc::B));
                }
                match op {
                    BinOp::Add => (self.b.ins().fadd(x, y), Sc::F),
                    BinOp::Sub => (self.b.ins().fsub(x, y), Sc::F),
                    BinOp::Mul => (self.b.ins().fmul(x, y), Sc::F),
                    BinOp::Div => (self.b.ins().fdiv(x, y), Sc::F),
                    BinOp::Rem => (self.helper("kairos_fmod", &[x, y])?, Sc::F),
                    BinOp::Pow => (self.helper("kairos_pow", &[x, y])?, Sc::F),
                    _ => return Err(()),
                }
            }
            Sc::B => match op {
                BinOp::Eq => (self.b.ins().icmp(IntCC::Equal, x, y), Sc::B),
                BinOp::Ne => (self.b.ins().icmp(IntCC::NotEqual, x, y), Sc::B),
                _ => return Err(()),
            },
        })
    }
}
