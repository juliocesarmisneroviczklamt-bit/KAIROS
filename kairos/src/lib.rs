//! Kairós: programming language for financial research and AI.
//!
//! - M1 (core): lexer, parser, type checker (inference by unification), typed HIR and the
//!   reference interpreter.
//! - M2: named-axis tensors (axis errors at compile time) and the `b3` module, the bridge to the
//!   quant-b3 lab (B3 data and its realistic simulator).
//!
//! Temporal types, `grad` and affine holdouts build on this in M3-M5; the native backend
//! (Cranelift) is M7. Spec: `docs/spec.md`.

pub mod ast;
pub mod autodiff;
pub mod bridge;
pub mod diag;
pub mod hir;
pub mod interp;
pub mod lexer;
pub mod parser;
pub mod session;
pub mod temporal;
pub mod tensor;
pub mod types;
pub mod value;
