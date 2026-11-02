//! Runtime values. Aggregates are `Rc`-shared and copied on write (`Rc::make_mut`): this is how
//! Kairós gives value semantics (`let b = a` is a copy) without paying for copies nobody mutates.

use crate::tensor::TensorVal;
use std::cmp::Ordering;
use std::collections::HashMap;
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
    /// the holdout of a protocol (M5): affine, opened at most once (index of the protocol)
    Holdout(u32),
    /// `Map[K, V]` (M9): keys `i64` or `str`, in insertion order
    Map(Rc<MapVal>),
    /// `Bytes` (M10): binary data, little-endian
    Bytes(Rc<Vec<u8>>),
    /// a float or float tensor tracked by `grad` (M4): index of its node on the active tape
    Node(u32),
    /// a `t3` (M18): -1, 0 or +1
    T3(i8),
}

#[derive(Debug)]
pub enum FuncVal {
    User(u32),
    Closure { def: u32, captured: Vec<Value> },
    /// `grad(f, wrt: k)` / `value_and_grad(f, wrt: k)` (argument index from 0)
    Grad { f: Value, wrt: usize, with_value: bool },
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
            (Node(a), Node(b)) => a == b,
            (Holdout(a), Holdout(b)) => a == b,
            (Map(a), Map(b)) => a == b,
            (Bytes(a), Bytes(b)) => a == b,
            (T3(a), T3(b)) => a == b,
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
        (T3(x), T3(y)) => x.cmp(y),
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

/// A map key: the checker allows only `i64` and `str`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Key {
    I(i64),
    S(Rc<str>),
}

fn key_of(v: &Value) -> Key {
    match v {
        Value::Int(i) => Key::I(*i),
        Value::Str(s) => Key::S(s.clone()),
        other => unreachable!("map key {other:?}"),
    }
}

/// The value of a `Map[K, V]`: entries in insertion order, found by hashing. Writing to an
/// existing key keeps its place; removing shifts the later ones.
#[derive(Clone, Debug, Default)]
pub struct MapVal {
    keys: Vec<Value>,
    vals: Vec<Value>,
    index: HashMap<Key, usize>,
}

impl MapVal {
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn keys(&self) -> &[Value] {
        &self.keys
    }

    pub fn values(&self) -> &[Value] {
        &self.vals
    }

    pub fn get(&self, k: &Value) -> Option<&Value> {
        self.index.get(&key_of(k)).map(|&i| &self.vals[i])
    }

    pub fn contains(&self, k: &Value) -> bool {
        self.index.contains_key(&key_of(k))
    }

    /// The value of `k`, if there is one, to change in place.
    pub fn get_mut(&mut self, k: &Value) -> Option<&mut Value> {
        let i = *self.index.get(&key_of(k))?;
        Some(&mut self.vals[i])
    }

    /// The value of `k`, added (as `()`, to be written) when it is missing.
    pub fn slot(&mut self, k: &Value) -> &mut Value {
        let key = key_of(k);
        let i = match self.index.get(&key) {
            Some(&i) => i,
            None => {
                self.index.insert(key, self.keys.len());
                self.keys.push(k.clone());
                self.vals.push(Value::Unit);
                self.keys.len() - 1
            }
        };
        &mut self.vals[i]
    }

    pub fn insert(&mut self, k: Value, v: Value) {
        *self.slot(&k) = v;
    }

    pub fn remove(&mut self, k: &Value) -> Option<Value> {
        let i = self.index.remove(&key_of(k))?;
        self.keys.remove(i);
        let v = self.vals.remove(i);
        for (j, key) in self.keys.iter().enumerate().skip(i) {
            self.index.insert(key_of(key), j);
        }
        Some(v)
    }
}

/// Maps are equal when they have the same keys with equal values (in any order).
impl PartialEq for MapVal {
    fn eq(&self, o: &Self) -> bool {
        self.len() == o.len() && self.keys.iter().zip(&self.vals).all(|(k, v)| o.get(k) == Some(v))
    }
}
