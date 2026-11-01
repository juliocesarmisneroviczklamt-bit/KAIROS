//! Runtime values. Aggregates are `Rc`-shared and copied on write (`Rc::make_mut`): this is how
//! Kairós gives value semantics (`let b = a` is a copy) without paying for copies nobody mutates.

use crate::tensor::TensorVal;
use std::cmp::Ordering;
use std::rc::Rc;

#[derive(Clone, Debug, Default)]
pub enum Value {
    #[default]
    Unit,
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(Rc<str>),
    List(Rc<Vec<Value>>),
    Tuple(Rc<Vec<Value>>),
    Struct(u32, Rc<Vec<Value>>),
    Enum(u32, u32, Rc<Vec<Value>>),
    Func(Rc<FuncVal>),
    /// named-axis tensor (M2); a tensor with no axes left is always a scalar `Float`/`Bool`
    Tensor(Rc<TensorVal>),
}

#[derive(Debug)]
pub enum FuncVal {
    User(u32),
    Closure { def: u32, captured: Vec<Value> },
}

impl PartialEq for Value {
    fn eq(&self, o: &Self) -> bool {
        use Value::*;
        match (self, o) {
            (Unit, Unit) => true,
            (Int(a), Int(b)) => a == b,
            (Float(a), Float(b)) => a == b,
            (Bool(a), Bool(b)) => a == b,
            (Str(a), Str(b)) => a == b,
            (List(a), List(b)) | (Tuple(a), Tuple(b)) => a == b,
            (Struct(i, a), Struct(j, b)) => i == j && a == b,
            (Enum(e, v, a), Enum(f, w, b)) => e == f && v == w && a == b,
            (Func(a), Func(b)) => Rc::ptr_eq(a, b),
            (Tensor(a), Tensor(b)) => a == b,
            _ => false,
        }
    }
}

/// Total order used by `sort` (floats via `total_cmp`, lists and tuples lexicographic).
pub fn total_cmp(a: &Value, b: &Value) -> Ordering {
    use Value::*;
    match (a, b) {
        (Int(x), Int(y)) => x.cmp(y),
        (Float(x), Float(y)) => x.total_cmp(y),
        (Bool(x), Bool(y)) => x.cmp(y),
        (Str(x), Str(y)) => x.cmp(y),
        (List(x), List(y)) | (Tuple(x), Tuple(y)) => {
            for (p, q) in x.iter().zip(y.iter()) {
                let c = total_cmp(p, q);
                if c != Ordering::Equal {
                    return c;
                }
            }
            x.len().cmp(&y.len())
        }
        _ => Ordering::Equal,
    }
}
