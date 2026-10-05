//! Python emission for predicate specs: a byte-for-byte mirror of
//! engine/swa/predicate.py `spec_to_python` (readable class tests where exactly
//! equivalent over the 2048-token vocab, set literals otherwise) + short English desc.

use crate::spec::Spec;
use crate::vocab::Vocab;

const OPERATOR_CHARS: &str = "+-*/%=<>!&|^~";

fn is_ascii_alpha(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Python `str.isspace()`.
fn py_isspace(c: char) -> bool {
    c.is_ascii_whitespace()
        || ('\u{1c}'..='\u{1f}').contains(&c)
        || c == '\u{85}'
        || c == '\u{a0}'
        || c.is_whitespace()
}

fn py_isdigit(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

fn py_isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(is_ascii_alpha)
}

/// Python `str.islower()`: at least one cased char, all cased chars lowercase.
fn py_islower(s: &str) -> bool {
    let mut cased = false;
    for c in s.chars() {
        if is_ascii_alpha(c) {
            cased = true;
            if !c.is_ascii_lowercase() {
                return false;
            }
        }
    }
    cased
}

fn py_isalnum(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

fn class_test(cls: &str, s: &str) -> Option<bool> {
    Some(match cls {
        "digit_only" => py_isdigit(s),
        "has_digit" => s.chars().any(|c| c.is_ascii_digit()),
        "alpha_only" => py_isalpha(s),
        "upper_start" => s.chars().next().map(|c| c.is_ascii_uppercase()).unwrap_or(false),
        "lower_only" => py_islower(s),
        "space_lead" => s.starts_with(' '),
        "newline" => s.contains('\n'),
        "indent" => s.trim_matches(py_isspace).is_empty(),
        "punct_only" => !s.is_empty() && s.chars().all(|c| !py_isalnum(c) && !py_isspace(c)),
        "quote" => s.contains('\'') || s.contains('"'),
        "bracket_open" => s.chars().any(|c| "([{".contains(c)),
        "bracket_close" => s.chars().any(|c| ")]}".contains(c)),
        "operator" => !s.is_empty() && s.chars().all(|c| OPERATOR_CHARS.contains(c)),
        "dot" => s.contains('.'),
        "comma" => s.contains(','),
        "colon" => s.contains(':'),
        "underscore" => s.contains('_'),
        "eot" => s == "<|endoftext|>",
        _ => return None,
    })
}

/// Python-source template for a class test, `{t}` = token expression. `keyword` has none.
fn class_src(cls: &str) -> Option<fn(&str) -> String> {
    Some(match cls {
        "digit_only" => |t: &str| format!("({t}).isdigit()"),
        "has_digit" => |t: &str| format!("any(c.isdigit() for c in {t})"),
        "alpha_only" => |t: &str| format!("({t}).isalpha()"),
        "upper_start" => |t: &str| format!("({t})[:1].isupper()"),
        "lower_only" => |t: &str| format!("({t}).islower()"),
        "space_lead" => |t: &str| format!("({t}).startswith(' ')"),
        "newline" => |t: &str| format!("'\\n' in {t}"),
        "indent" => |t: &str| format!("({t}).strip() == ''"),
        "punct_only" => |t: &str| {
            format!("({t}) != '' and all((not c.isalnum()) and (not c.isspace()) for c in ({t}))")
        },
        "quote" => |t: &str| format!("(\"'\" in {t}) or ('\"' in {t})"),
        "bracket_open" => |t: &str| format!("any(c in '([{{' for c in {t})"),
        "bracket_close" => |t: &str| format!("any(c in ')]}}' for c in {t})"),
        "operator" => |t: &str| format!("({t}) != '' and all(c in '{OPERATOR_CHARS}' for c in ({t}))"),
        "dot" => |t: &str| format!("'.' in {t}"),
        "comma" => |t: &str| format!("',' in {t}"),
        "colon" => |t: &str| format!("':' in {t}"),
        "underscore" => |t: &str| format!("'_' in {t}"),
        "eot" => |t: &str| format!("({t}) == '<|endoftext|>'"),
        _ => return None,
    })
}

/// Python `repr()` of a string (single/double quote choice + escapes).
pub fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        if c == '\\' {
            out.push_str("\\\\");
        } else if c == quote {
            out.push('\\');
            out.push(quote);
        } else if c == '\n' {
            out.push_str("\\n");
        } else if c == '\r' {
            out.push_str("\\r");
        } else if c == '\t' {
            out.push_str("\\t");
        } else if (c as u32) < 0x20 || (c as u32) == 0x7f {
            out.push_str(&format!("\\x{:02x}", c as u32));
        } else {
            out.push(c);
        }
    }
    out.push(quote);
    out
}

/// Python `_set_lit`: `{...}` of sorted unique reprs, `set()` when empty.
pub fn set_lit(strings: &[String]) -> String {
    let mut uniq: Vec<&String> = strings.iter().collect();
    uniq.sort();
    uniq.dedup();
    if uniq.is_empty() {
        return "set()".to_string();
    }
    let parts: Vec<String> = uniq.iter().map(|s| py_repr(s)).collect();
    format!("{{{}}}", parts.join(", "))
}

pub fn class_py_expr(cls: &str, tok_expr: &str, vocab: &Vocab, ids: &[u16]) -> String {
    if let Some(src) = class_src(cls) {
        let member: Vec<bool> = (0..vocab.tokens.len())
            .map(|i| ids.binary_search(&(i as u16)).is_ok())
            .collect();
        let mut ok = true;
        for (i, s) in vocab.tokens.iter().enumerate() {
            let want = member[i];
            let got = class_test(cls, s).unwrap_or(false);
            if got != want {
                ok = false;
                break;
            }
        }
        if ok {
            return src(tok_expr);
        }
    }
    let strings: Vec<String> = ids
        .iter()
        .filter(|i| (**i as usize) < vocab.tokens.len())
        .map(|i| vocab.tokens[*i as usize].clone())
        .collect();
    format!("{tok_expr} in {}", set_lit(&strings))
}

fn tok_expr(off: i32) -> String {
    if off == 0 {
        "tokens[pos]".to_string()
    } else {
        format!("tokens[pos{off:+}]")
    }
}

fn guard(off: i32) -> String {
    if off >= 0 {
        String::new()
    } else {
        format!("pos >= {} and ", -off)
    }
}

fn emit(spec: &Spec, vocab: &Vocab) -> String {
    match spec {
        Spec::TokIn { off, ids } => {
            let in_range: Vec<u16> = ids
                .iter()
                .copied()
                .filter(|i| (*i as usize) < vocab.tokens.len())
                .collect();
            let strings: Vec<String> = in_range
                .iter()
                .map(|i| vocab.tokens[*i as usize].clone())
                .collect();
            format!("{}{} in {}", guard(*off), tok_expr(*off), set_lit(&strings))
        }
        Spec::Cls { off, cls } => {
            let ci = vocab
                .class_index(cls)
                .unwrap_or_else(|| panic!("unknown class {cls:?}"));
            let expr = class_py_expr(cls, &tok_expr(*off), vocab, &vocab.class_ids[ci]);
            format!("{}({})", guard(*off), expr)
        }
        Spec::WinIn { lo, hi, ids } => {
            let strings: Vec<String> = ids
                .iter()
                .filter(|i| (**i as usize) < vocab.tokens.len())
                .map(|i| vocab.tokens[*i as usize].clone())
                .collect();
            let inner = format!("tokens[j] in {}", set_lit(&strings));
            format!(
                "any({inner} for j in range(max(0, pos{lo:+}), pos{:+}))",
                hi + 1
            )
        }
        Spec::WinCls { lo, hi, cls } => {
            let ci = vocab
                .class_index(cls)
                .unwrap_or_else(|| panic!("unknown class {cls:?}"));
            let inner = class_py_expr(cls, "tokens[j]", vocab, &vocab.class_ids[ci]);
            format!(
                "any({inner} for j in range(max(0, pos{lo:+}), pos{:+}))",
                hi + 1
            )
        }
        Spec::And(a, b) => format!("({}) and ({})", emit(a, vocab), emit(b, vocab)),
        Spec::Or(a, b) => format!("({}) or ({})", emit(a, vocab), emit(b, vocab)),
    }
}

/// `def f(tokens, pos):\n    return <expr>\n`
pub fn spec_to_python(spec: &Spec, vocab: &Vocab) -> String {
    format!("def f(tokens, pos):\n    return {}\n", emit(spec, vocab))
}

fn id_label(id: u16, vocab: &Vocab) -> String {
    let s = &vocab.tokens[id as usize];
    let n = vocab.tokens.iter().filter(|t| *t == s).count();
    if n > 1 {
        format!("{} (x{})", py_repr(s), n)
    } else {
        py_repr(s)
    }
}

fn ids_label(ids: &[u16], vocab: &Vocab) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut seen: Vec<&str> = Vec::new();
    for id in ids {
        let s = vocab.tokens[*id as usize].as_str();
        if seen.contains(&s) {
            continue;
        }
        seen.push(s);
        parts.push(id_label(*id, vocab));
        if parts.len() == 6 {
            break;
        }
    }
    let mut out = parts.join(", ");
    if seen.len() < ids.len() {
        out.push_str(", ...");
    }
    out
}

/// Short template English one-liner.
pub fn describe(spec: &Spec, vocab: &Vocab) -> String {
    match spec {
        Spec::TokIn { off, ids } => format!("token at pos{off:+} in {{{}}}", ids_label(ids, vocab)),
        Spec::Cls { off, cls } => format!("token at pos{off:+} is {cls}"),
        Spec::WinIn { lo, hi, ids } => format!(
            "any of pos{lo:+}..pos{hi:+} in {{{}}}",
            ids_label(ids, vocab)
        ),
        Spec::WinCls { lo, hi, cls } => format!("any of pos{lo:+}..pos{hi:+} is {cls}"),
        Spec::And(a, b) => format!("({} and {})", describe(a, vocab), describe(b, vocab)),
        Spec::Or(a, b) => format!("({} or {})", describe(a, vocab), describe(b, vocab)),
    }
}
