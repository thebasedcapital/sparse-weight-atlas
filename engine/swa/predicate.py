"""Predicate specs for weight-sparse atlas: vectorized evaluation + Python emission.

Spec AST (CONTRACT.md, causal only):
    {"op":"tok_in","off":k,"ids":[...]}          tokens[pos+k] in ids,  k in [-8, 0]
    {"op":"cls","off":k,"cls":"digit_only"}      tokens[pos+k] in vocab.classes[cls]
    {"op":"win_in","lo":a,"hi":b,"ids":[...]}    exists j in [pos+a, pos+b]: tokens[j] in ids
    {"op":"win_cls","lo":a,"hi":b,"cls":"..."}   same with a class
    {"op":"and","args":[A,B]}  {"op":"or","args":[A,B]}  {"op":"not","arg":A}

Evaluation is per row of the slice (positions never cross rows) and causal (never reads
pos > 0 offset).  Out-of-range positions (pos + k < 0 within the scored window) are false.

Public API
    load_vocab(model, runs_dir="runs") -> dict with "tokens" and "classes"
    eval_spec(spec, tokens_inputs, classes) -> BoolTensor[n_seq, T]
    spec_to_python(spec, vocab_tokens) -> str (source of ``def f(tokens, pos):``)
    check_python(spec, py, tokens_inputs, classes, vocab_tokens, ...) -> bool
    check_python_detail(...) -> dict
    validate_spec(spec) -> None (raises ValueError on a malformed spec)
"""

from __future__ import annotations

import json
import keyword
import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

VOCAB_SIZE = 2048
OFF_MIN, OFF_MAX = -8, 0
WIN_MIN = -8

# --------------------------------------------------------------------------------------
# vocab
# --------------------------------------------------------------------------------------


def load_vocab(model: str, runs_dir: str = "runs") -> Dict[str, Any]:
    """Read runs/<model>/vocab.json -> {"tokens": [...2048], "classes": {name: [ids]}}."""
    path = os.path.join(runs_dir, model, "vocab.json")
    with open(path, "r", encoding="utf-8") as fh:
        v = json.load(fh)
    if not isinstance(v, dict) or "tokens" not in v:
        raise ValueError(f"{path}: missing 'tokens'")
    v.setdefault("classes", {})
    if len(v["tokens"]) != VOCAB_SIZE:
        raise ValueError(f"{path}: expected {VOCAB_SIZE} tokens, got {len(v['tokens'])}")
    return v


def _norm_classes(classes: Any) -> Dict[str, Sequence[int]]:
    """Accept either the classes mapping or a whole vocab dict."""
    if isinstance(classes, dict) and "classes" in classes and "tokens" in classes:
        return classes["classes"]
    if classes is None:
        return {}
    return classes


# --------------------------------------------------------------------------------------
# spec validation
# --------------------------------------------------------------------------------------

_OPS = {"tok_in", "cls", "win_in", "win_cls", "and", "or", "not"}


def validate_spec(spec: Any) -> None:
    """Raise ValueError unless *spec* is a well-formed causal predicate spec."""
    if not isinstance(spec, dict):
        raise ValueError(f"spec must be a dict, got {type(spec).__name__}")
    op = spec.get("op")
    if op not in _OPS:
        raise ValueError(f"unknown op {op!r}")
    if op in ("tok_in", "cls"):
        off = spec.get("off")
        if not isinstance(off, int) or not (OFF_MIN <= off <= OFF_MAX):
            raise ValueError(f"{op}: off must be an int in [{OFF_MIN}, {OFF_MAX}], got {off!r}")
    if op in ("win_in", "win_cls"):
        lo, hi = spec.get("lo"), spec.get("hi")
        if not isinstance(lo, int) or not isinstance(hi, int):
            raise ValueError(f"{op}: lo/hi must be ints, got {lo!r}/{hi!r}")
        if not (WIN_MIN <= lo <= hi <= -1):
            raise ValueError(f"{op}: require {WIN_MIN} <= lo <= hi <= -1, got lo={lo}, hi={hi}")
    if op in ("tok_in", "win_in"):
        ids = spec.get("ids")
        if not isinstance(ids, (list, tuple)) or any(not isinstance(i, int) for i in ids):
            raise ValueError(f"{op}: ids must be a list of ints")
    if op in ("cls", "win_cls"):
        if not isinstance(spec.get("cls"), str):
            raise ValueError(f"{op}: cls must be a string")
    if op in ("and", "or"):
        args = spec.get("args")
        if not isinstance(args, (list, tuple)) or len(args) < 1:
            raise ValueError(f"{op}: args must be a non-empty list")
        for a in args:
            validate_spec(a)
    if op == "not":
        validate_spec(spec.get("arg"))


# --------------------------------------------------------------------------------------
# vectorized evaluation
# --------------------------------------------------------------------------------------


def _id_mask(ids: Sequence[int], size: int, device: torch.device) -> torch.Tensor:
    m = torch.zeros(size, dtype=torch.bool, device=device)
    if ids:
        idx = torch.as_tensor(sorted({int(i) for i in ids}), dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < size)]
        if idx.numel():
            m[idx.to(device)] = True
    return m


def _cls_ids(spec: Dict[str, Any], classes: Dict[str, Sequence[int]]) -> Sequence[int]:
    name = spec["cls"]
    if name not in classes:
        raise KeyError(f"unknown class {name!r}; available: {sorted(classes)}")
    return classes[name]


def _gather(tok: torch.Tensor, off: int, T: int, size: int, mask: torch.Tensor) -> torch.Tensor:
    """tokens[:, pos+off] in mask (out-of-range -> False). tok [n, T] ints."""
    pos = torch.arange(T, device=tok.device)
    idx = pos + off
    valid = idx >= 0
    idx = idx.clamp(min=0)
    vals = tok[:, idx]
    in_range = vals < size
    safe = vals.clamp(max=size - 1)
    return mask[safe] & in_range & valid.unsqueeze(0)


def _eval(spec: Dict[str, Any], tok: torch.Tensor, classes: Dict[str, Sequence[int]],
          size: int) -> torch.Tensor:
    op = spec["op"]
    T = tok.shape[1]
    if op == "tok_in":
        m = _id_mask(spec["ids"], size, tok.device)
        return _gather(tok, spec["off"], T, size, m)
    if op == "cls":
        m = _id_mask(_cls_ids(spec, classes), size, tok.device)
        return _gather(tok, spec["off"], T, size, m)
    if op in ("win_in", "win_cls"):
        ids = spec["ids"] if op == "win_in" else _cls_ids(spec, classes)
        m = _id_mask(ids, size, tok.device)
        out = torch.zeros_like(tok, dtype=torch.bool)
        for off in range(spec["lo"], spec["hi"] + 1):
            out |= _gather(tok, off, T, size, m)
        return out
    if op == "and":
        out = _eval(spec["args"][0], tok, classes, size)
        for a in spec["args"][1:]:
            out = out & _eval(a, tok, classes, size)
        return out
    if op == "or":
        out = _eval(spec["args"][0], tok, classes, size)
        for a in spec["args"][1:]:
            out = out | _eval(a, tok, classes, size)
        return out
    if op == "not":
        return ~_eval(spec["arg"], tok, classes, size)
    raise ValueError(f"unknown op {op!r}")


def eval_spec(spec: Any, tokens_inputs: torch.Tensor, classes: Any) -> torch.Tensor:
    """Evaluate *spec* on tokens_inputs [n_seq, T] -> BoolTensor [n_seq, T].

    ``classes`` may be the classes mapping (name -> ids) or a whole vocab dict.
    """
    validate_spec(spec)
    tok = torch.as_tensor(tokens_inputs)
    if tok.dim() != 2:
        raise ValueError(f"tokens_inputs must be [n_seq, T], got shape {tuple(tok.shape)}")
    tok = tok.to(torch.long)
    cls = _norm_classes(classes)
    size = VOCAB_SIZE
    if tok.numel():
        size = max(size, int(tok.max()) + 1)
    for name in _collect_cls_names(spec):
        if name in cls and cls[name]:
            size = max(size, max(int(i) for i in cls[name]) + 1)
    return _eval(spec, tok, cls, size)


def _collect_cls_names(spec: Dict[str, Any]) -> List[str]:
    op = spec["op"]
    if op in ("cls", "win_cls"):
        return [spec["cls"]]
    if op in ("and", "or"):
        out: List[str] = []
        for a in spec["args"]:
            out += _collect_cls_names(a)
        return out
    if op == "not":
        return _collect_cls_names(spec["arg"])
    return []


# --------------------------------------------------------------------------------------
# Python emission
# --------------------------------------------------------------------------------------

_OPERATOR_CHARS = "+-*/%=<>!&|^~"
_KEYWORDS = frozenset(keyword.kwlist) | frozenset(getattr(keyword, "softkwlist", []))


def _is_digit_only(s: str) -> bool:
    return s.isdigit() and s != ""


def _is_alpha_only(s: str) -> bool:
    return s.isalpha() and s != ""


def _is_lower_only(s: str) -> bool:
    return s.islower() and s != ""


def _is_indent(s: str) -> bool:
    return s.strip() == ""


def _is_punct_only(s: str) -> bool:
    return s != "" and all((not c.isalnum()) and (not c.isspace()) for c in s)


def _is_operator(s: str) -> bool:
    return s != "" and all(c in _OPERATOR_CHARS for c in s)


def _is_bracket_open(s: str) -> bool:
    return any(c in "([{" for c in s)


def _is_bracket_close(s: str) -> bool:
    return any(c in ")]}" for c in s)


def _is_quote(s: str) -> bool:
    return ("'" in s) or ('"' in s)


def _is_upper_start(s: str) -> bool:
    return s[:1].isupper()


def _is_keyword(s: str) -> bool:
    return s.strip() in _KEYWORDS


# class name -> (python callable over decoded strings, source template using `{t}`)
_CLS_TESTS: Dict[str, Callable[[str], bool]] = {
    "digit_only": _is_digit_only,
    "has_digit": lambda s: any(c.isdigit() for c in s),
    "alpha_only": _is_alpha_only,
    "upper_start": _is_upper_start,
    "lower_only": _is_lower_only,
    "space_lead": lambda s: s.startswith(" "),
    "newline": lambda s: "\n" in s,
    "indent": _is_indent,
    "punct_only": _is_punct_only,
    "quote": _is_quote,
    "bracket_open": _is_bracket_open,
    "bracket_close": _is_bracket_close,
    "operator": _is_operator,
    "dot": lambda s: "." in s,
    "comma": lambda s: "," in s,
    "colon": lambda s: ":" in s,
    "underscore": lambda s: "_" in s,
    "keyword": _is_keyword,
    "eot": lambda s: s == "<|endoftext|>",
}

_CLS_SRC: Dict[str, Callable[[str], str]] = {
    "digit_only": lambda t: f"({t}).isdigit()",
    "has_digit": lambda t: f"any(c.isdigit() for c in {t})",
    "alpha_only": lambda t: f"({t}).isalpha()",
    "upper_start": lambda t: f"({t})[:1].isupper()",
    "lower_only": lambda t: f"({t}).islower()",
    "space_lead": lambda t: f"({t}).startswith(' ')",
    "newline": lambda t: f"'\\n' in {t}",
    "indent": lambda t: f"({t}).strip() == ''",
    "punct_only": lambda t: (
        f"({t}) != '' and all((not c.isalnum()) and (not c.isspace()) for c in ({t}))"
    ),
    "quote": lambda t: f"(\"'\" in {t}) or ('\"' in {t})",
    "bracket_open": lambda t: f"any(c in '([{{' for c in {t})",
    "bracket_close": lambda t: f"any(c in ')]}}' for c in {t})",
    "operator": lambda t: f"({t}) != '' and all(c in '{_OPERATOR_CHARS}' for c in ({t}))",
    "dot": lambda t: f"'.' in {t}",
    "comma": lambda t: f"',' in {t}",
    "colon": lambda t: f"':' in {t}",
    "underscore": lambda t: f"'_' in {t}",
    "eot": lambda t: f"({t}) == '<|endoftext|>'",
}
# keyword is emitted as a set literal of the keyword strings (still exactly equivalent).


def class_py_expr(cls: str, tok_expr: str, vocab_tokens: Sequence[str],
                  cls_ids: Sequence[int]) -> str:
    """Readable Python test for class *cls* on token expression *tok_expr*.

    Emits the readable form only when it is exactly equivalent to the class id set over the
    2048-token vocab; otherwise emits an explicit set literal of the decoded tokens.
    """
    test = _CLS_TESTS.get(cls)
    if test is not None:
        ok = True
        for i, s in enumerate(vocab_tokens):
            want = i in _idset(cls_ids)
            try:
                got = bool(test(s))
            except Exception:
                got = None
            if got != want:
                ok = False
                break
        if ok and cls in _CLS_SRC:
            return _CLS_SRC[cls](tok_expr)
    return f"{tok_expr} in {_set_lit([vocab_tokens[i] for i in cls_ids if 0 <= i < len(vocab_tokens)])}"


def _idset(ids: Sequence[int]) -> set:
    return {int(i) for i in ids}


def _set_lit(strings: Sequence[str]) -> str:
    uniq = sorted(set(strings))
    if not uniq:
        return "set()"
    return "{" + ", ".join(repr(s) for s in uniq) + "}"


def _tok_expr(off: int) -> str:
    return "tokens[pos]" if off == 0 else f"tokens[pos{off:+d}]"


def _guard(off: int) -> str:
    if off >= 0:
        return ""
    return f"pos >= {-off} and "


def _emit(spec: Dict[str, Any], vocab_tokens: Sequence[str],
          classes: Dict[str, Sequence[int]]) -> str:
    op = spec["op"]
    if op == "tok_in":
        ids = [int(i) for i in spec["ids"] if 0 <= int(i) < len(vocab_tokens)]
        lit = _set_lit([vocab_tokens[i] for i in ids])
        return f"{_guard(spec['off'])}{_tok_expr(spec['off'])} in {lit}"
    if op == "cls":
        name = spec["cls"]
        ids = classes.get(name)
        if ids is None:
            raise KeyError(f"unknown class {name!r}; available: {sorted(classes)}")
        expr = class_py_expr(name, _tok_expr(spec["off"]), vocab_tokens, ids)
        # parenthesise: readable class tests contain top-level and/or that the offset guard
        # must not bind inside (``pos >= 1 and A or B`` would leave B unguarded)
        return f"{_guard(spec['off'])}({expr})"
    if op in ("win_in", "win_cls"):
        lo, hi = spec["lo"], spec["hi"]
        start, end = f"max(0, pos{lo:+d})", f"pos{hi + 1:+d}"
        if op == "win_in":
            ids = [int(i) for i in spec["ids"] if 0 <= int(i) < len(vocab_tokens)]
            inner = f"tokens[j] in {_set_lit([vocab_tokens[i] for i in ids])}"
        else:
            name = spec["cls"]
            ids = classes.get(name)
            if ids is None:
                raise KeyError(f"unknown class {name!r}; available: {sorted(classes)}")
            inner = class_py_expr(name, "tokens[j]", vocab_tokens, ids)
        return f"any({inner} for j in range({start}, {end}))"
    if op in ("and", "or"):
        parts = [_emit(a, vocab_tokens, classes) for a in spec["args"]]
        sep = f") {op} ("
        return "(" + sep.join(parts) + ")"
    if op == "not":
        return f"not ({_emit(spec['arg'], vocab_tokens, classes)})"
    raise ValueError(f"unknown op {op!r}")


def spec_to_python(spec: Any, vocab_tokens: Sequence[str],
                   classes: Optional[Dict[str, Sequence[int]]] = None) -> str:
    """Emit readable ``def f(tokens, pos):`` source for *spec* (tokens = decoded strings)."""
    validate_spec(spec)
    if len(vocab_tokens) != VOCAB_SIZE:
        raise ValueError(f"vocab_tokens must have {VOCAB_SIZE} entries, got {len(vocab_tokens)}")
    cls = _norm_classes(classes)
    expr = _emit(spec, list(vocab_tokens), cls)
    return f"def f(tokens, pos):\n    return {expr}\n"


# --------------------------------------------------------------------------------------
# exec-based verification
# --------------------------------------------------------------------------------------


def _exec_f(py: str) -> Callable:
    ns: Dict[str, Any] = {}
    exec(compile(py, "<predicate>", "exec"), ns)
    f = ns.get("f")
    if not callable(f):
        raise ValueError("emitted source does not define a callable f(tokens, pos)")
    return f


def check_python_detail(spec: Any, py: str, tokens_inputs: torch.Tensor, classes: Any,
                        vocab_tokens: Sequence[str], positions: Optional[Sequence[Any]] = None,
                        seed: int = 0, max_positions: Optional[int] = None) -> Dict[str, Any]:
    """Exec *py* and compare against eval_spec on a sample of positions.

    ``positions``: optional iterable of (row, pos) pairs.  Otherwise all n_seq*T positions
    (or a seeded random sample of ``max_positions``).
    Returns {"ok", "checked", "mismatches", "first_mismatch"}.
    """
    tok = torch.as_tensor(tokens_inputs)
    n, T = tok.shape
    expected = eval_spec(spec, tok, classes)
    f = _exec_f(py)
    rows = [[str(vocab_tokens[int(i)]) for i in row] for row in tok]

    if positions is None:
        total = n * T
        if max_positions is None or total <= max_positions:
            pairs = ((r, p) for r in range(n) for p in range(T))
        else:
            g = torch.Generator().manual_seed(seed)
            flat = torch.randperm(total, generator=g)[:max_positions]
            pairs = ((int(i) // T, int(i) % T) for i in flat)
    else:
        pairs = positions

    checked = 0
    mismatches = 0
    first = None
    for r, p in pairs:
        want = bool(expected[r, p])
        try:
            got = bool(f(rows[r], int(p)))
        except Exception as exc:  # a raised error is a mismatch
            got = f"<error {type(exc).__name__}: {exc}>"
        checked += 1
        if got is not want:
            mismatches += 1
            if first is None:
                first = {"row": int(r), "pos": int(p), "expected": want, "got": got}
    return {"ok": mismatches == 0, "checked": checked, "mismatches": mismatches,
            "first_mismatch": first}


def check_python(spec: Any, py: str, tokens_inputs: torch.Tensor, classes: Any,
                 vocab_tokens: Sequence[str], **kw: Any) -> bool:
    """True iff the emitted source agrees with eval_spec on the sampled positions."""
    return bool(check_python_detail(spec, py, tokens_inputs, classes, vocab_tokens, **kw)["ok"])
