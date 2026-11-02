//! Targets without Cranelift support (see build.rs): everything runs in the interpreter, with the
//! same results.

use crate::diag::Diag;
use crate::types::Checker;
use crate::value::Value;

#[derive(Default)]
pub struct Jit;

impl Jit {
    pub fn new() -> Jit {
        Jit
    }

    pub fn with(_on: bool) -> Jit {
        Jit
    }

    pub fn enabled(&self) -> bool {
        false
    }

    pub fn compiled(&self) -> usize {
        0
    }

    pub fn call(&mut self, _ck: &Checker, _id: u32, _args: &[Value], _depth: u32, _max_depth: u32) -> Option<Result<Value, Diag>> {
        None
    }
}
