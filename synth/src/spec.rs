//! Predicate spec AST (CONTRACT.md) + canonical JSON.

use serde_json::{json, Value};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Spec {
    TokIn { off: i32, ids: Vec<u16> },
    Cls { off: i32, cls: String },
    WinIn { lo: i32, hi: i32, ids: Vec<u16> },
    WinCls { lo: i32, hi: i32, cls: String },
    And(Box<Spec>, Box<Spec>),
    Or(Box<Spec>, Box<Spec>),
}

impl Spec {
    pub fn to_json(&self) -> Value {
        match self {
            Spec::TokIn { off, ids } => json!({"op": "tok_in", "off": off, "ids": ids}),
            Spec::Cls { off, cls } => json!({"op": "cls", "off": off, "cls": cls}),
            Spec::WinIn { lo, hi, ids } => json!({"op": "win_in", "lo": lo, "hi": hi, "ids": ids}),
            Spec::WinCls { lo, hi, cls } => json!({"op": "win_cls", "lo": lo, "hi": hi, "cls": cls}),
            Spec::And(a, b) => json!({"op": "and", "args": [a.to_json(), b.to_json()]}),
            Spec::Or(a, b) => json!({"op": "or", "args": [a.to_json(), b.to_json()]}),
        }
    }

    /// Canonical string used for distinctness / deterministic ordering.  A hand-rolled
    /// writer: this runs once per generated candidate, where building a serde_json Value
    /// (BTreeMap + escaping) per spec is a measurable cost.
    pub fn key(&self) -> String {
        let mut out = String::with_capacity(64);
        self.write_key(&mut out);
        out
    }

    fn write_key(&self, out: &mut String) {
        match self {
            Spec::TokIn { off, ids } => {
                out.push_str("tok_in:");
                out.push_str(&off.to_string());
                out.push(':');
                for id in ids {
                    out.push_str(&id.to_string());
                    out.push(',');
                }
            }
            Spec::Cls { off, cls } => {
                out.push_str("cls:");
                out.push_str(&off.to_string());
                out.push(':');
                out.push_str(cls);
            }
            Spec::WinIn { lo, hi, ids } => {
                out.push_str("win_in:");
                out.push_str(&lo.to_string());
                out.push(':');
                out.push_str(&hi.to_string());
                out.push(':');
                for id in ids {
                    out.push_str(&id.to_string());
                    out.push(',');
                }
            }
            Spec::WinCls { lo, hi, cls } => {
                out.push_str("win_cls:");
                out.push_str(&lo.to_string());
                out.push(':');
                out.push_str(&hi.to_string());
                out.push(':');
                out.push_str(cls);
            }
            Spec::And(a, b) => {
                out.push_str("and(");
                a.write_key(out);
                out.push(',');
                b.write_key(out);
                out.push(')');
            }
            Spec::Or(a, b) => {
                out.push_str("or(");
                a.write_key(out);
                out.push(',');
                b.write_key(out);
                out.push(')');
            }
        }
    }
}
