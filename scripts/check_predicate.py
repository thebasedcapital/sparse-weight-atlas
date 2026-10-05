#!/usr/bin/env python
"""Acceptance gate: swa.predicate vs a naive per-position Python loop over emitted source.

For every spec op (tok_in, cls, win_in, win_cls, and, or, not) on random token tensors,
evaluate the spec vectorized and compare against the exec'd ``def f(tokens, pos):`` source
on every scored position (>= 10k per op), including window/out-of-range edges.

Usage:
    PYTHONPATH=engine .venv/bin/python scripts/check_predicate.py [--seqs 48] [--seed 0]
Exit code 0 iff 100% agreement everywhere.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Sequence

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine"))

from swa.predicate import (  # noqa: E402
    VOCAB_SIZE,
    _exec_f,
    check_python,
    eval_spec,
    load_vocab,
    spec_to_python,
    validate_spec,
)

CLASS_NAMES = [
    "digit_only", "has_digit", "alpha_only", "upper_start", "lower_only", "space_lead",
    "newline", "indent", "punct_only", "quote", "bracket_open", "bracket_close",
    "operator", "dot", "comma", "colon", "underscore", "keyword", "eot",
]

# Independent class definitions (from CONTRACT.md prose) -- deliberately NOT imported
# from swa.predicate, so the id sets are an independent oracle for the emitted source.
import keyword as _keyword  # noqa: E402


def cls_member(name: str, s: str) -> bool:
    if name == "digit_only":
        return s != "" and s.isdigit()
    if name == "has_digit":
        return any(c.isdigit() for c in s)
    if name == "alpha_only":
        return s != "" and s.isalpha()
    if name == "upper_start":
        return s[:1].isupper()
    if name == "lower_only":
        return s != "" and s.islower()
    if name == "space_lead":
        return s.startswith(" ")
    if name == "newline":
        return "\n" in s
    if name == "indent":
        return s.strip() == ""
    if name == "punct_only":
        return s != "" and all((not c.isalnum()) and (not c.isspace()) for c in s)
    if name == "quote":
        return ("'" in s) or ('"' in s)
    if name == "bracket_open":
        return any(c in "([{" for c in s)
    if name == "bracket_close":
        return any(c in ")]}" for c in s)
    if name == "operator":
        return s != "" and all(c in "+-*/%=<>!&|^~" for c in s)
    if name == "dot":
        return "." in s
    if name == "comma":
        return "," in s
    if name == "colon":
        return ":" in s
    if name == "underscore":
        return "_" in s
    if name == "keyword":
        return s.strip() in _keyword.kwlist
    if name == "eot":
        return s == "<|endoftext|>"
    raise KeyError(name)


def synthetic_vocab(seed: int = 1234) -> List[str]:
    """A 2048-entry decoded-token vocab exercising every class."""
    rng = random.Random(seed)
    pool = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        "_ \t\n()[]{}'\",.:;=+-*/%<>!&|^~@#$\\"
    )
    seeds = [
        "", " ", "  ", "\n", "\n\n", "\t", "    ",
        "0", "1", "2", "3", "4", "5", "9", "42", "64", "50", "59", "60", "100", "200", "300",
        "007", "1234", "12", "13", "16", "17", "28", "08", "27", "35", "32", "56", "41", "46",
        "90", "89", "95", "6", "20", "8", "67", "7",
        "e", "f", "a", "A", "Z", "if", "for", "else", "def", "return", "None", "True", "in",
        "is", "not", "and", "or", "class", "import", "lambda", "yield",
        "(", ")", "[", "]", "{", "}", "'", '"', "'''", '"""', ".", ",", ":", ";", "_", "__",
        "==", "!=", ">=", "<=", "=", "+", "-", "*", "/", "%", "**", "//", "&&", "||", "^", "~",
        "abc", "def_", "hello", "world", "foo123", "x1", "CamelCase", "snake_case", "with space",
        "end.", "a,b", "x:y", "p_q", "True_", "\nif", " x", "123abc", "abc123", "#", "\\n",
    ]
    assert len(set(seeds)) == len(seeds), "seed tokens must be unique"
    # Real BPE vocabs are injective (id -> decoded string is one-to-one); string-based
    # emission can only be exactly equivalent when it holds, so the synthetic vocab is
    # deduplicated by construction.
    out: List[str] = []
    seen = set()
    for i in range(VOCAB_SIZE - 1):
        if i < len(seeds):
            s = seeds[i]
        else:
            n = rng.randint(1, 6)
            s = "".join(rng.choice(pool) for _ in range(n))
            while s in seen:
                s += rng.choice(pool)
        seen.add(s)
        out.append(s)
    out.append("<|endoftext|>")  # id 2047
    assert len(set(out)) == VOCAB_SIZE, "synthetic vocab must be injective"
    return out


def build_classes(tokens: Sequence[str]) -> Dict[str, List[int]]:
    return {name: [i for i, s in enumerate(tokens) if cls_member(name, s)] for name in CLASS_NAMES}


def random_ids(rng: random.Random, allowed: Sequence[int]) -> List[int]:
    k = min(rng.randint(1, 8), len(allowed))
    return sorted(rng.sample(list(allowed), k))


def random_spec(rng: random.Random, allowed: Sequence[int], depth: int = 0) -> dict:
    choices = ["tok_in", "cls", "win_in", "win_cls"] + ([] if depth >= 2 else ["and", "or", "not"])
    op = rng.choice(choices)
    if op == "tok_in":
        return {"op": "tok_in", "off": rng.choice([-8, -7, -5, -3, -2, -1, 0]), "ids": random_ids(rng, allowed)}
    if op == "cls":
        return {"op": "cls", "off": rng.choice([-8, -4, -1, 0]), "cls": rng.choice(CLASS_NAMES)}
    if op in ("win_in", "win_cls"):
        lo = rng.choice([-8, -6, -3, -1])
        hi = rng.choice([-1, -2, lo])
        lo, hi = min(lo, hi), max(lo, hi)
        if op == "win_in":
            return {"op": "win_in", "lo": lo, "hi": hi, "ids": random_ids(rng, allowed)}
        return {"op": "win_cls", "lo": lo, "hi": hi, "cls": rng.choice(CLASS_NAMES)}
    if op == "not":
        return {"op": "not", "arg": random_spec(rng, allowed, depth + 1)}
    return {"op": op, "args": [random_spec(rng, allowed, depth + 1),
                               random_spec(rng, allowed, depth + 1)]}


def edge_specs(allowed: Sequence[int]) -> List[dict]:
    """Specs that stress out-of-range edges (pos < 8) and window boundaries."""
    return [
        {"op": "tok_in", "off": -8, "ids": random_ids(random.Random(7), allowed)},
        {"op": "tok_in", "off": -1, "ids": [0]},
        {"op": "cls", "off": -8, "cls": "digit_only"},
        {"op": "cls", "off": -1, "cls": "eot"},
        {"op": "win_in", "lo": -8, "hi": -1, "ids": random_ids(random.Random(11), allowed)},
        {"op": "win_in", "lo": -1, "hi": -1, "ids": [2047]},
        {"op": "win_in", "lo": -8, "hi": -8, "ids": [0]},
        {"op": "win_cls", "lo": -8, "hi": -1, "cls": "digit_only"},
        {"op": "win_cls", "lo": -3, "hi": -1, "cls": "newline"},
        {"op": "not", "arg": {"op": "win_cls", "lo": -8, "hi": -1, "cls": "indent"}},
        {"op": "and", "args": [{"op": "cls", "off": 0, "cls": "digit_only"},
                               {"op": "win_in", "lo": -8, "hi": -1, "ids": [0, 1, 2]}]},
        {"op": "or", "args": [{"op": "tok_in", "off": -2, "ids": [5]},
                              {"op": "not", "arg": {"op": "cls", "off": -1, "cls": "space_lead"}}]},
    ]


def make_tokens(rng: random.Random, n_seq: int, T: int, vocab_tokens: Sequence[str],
                allowed: Sequence[int]) -> torch.Tensor:
    """Random token ids drawn from *allowed*, biased towards class-interesting ids.

    *allowed* excludes ids whose decoded string is shared with another id: no function of the
    decoded strings can distinguish those, so string-based emission cannot be exact for them.
    """
    interesting = [i for i in allowed if any(cls_member(c, vocab_tokens[i]) for c in CLASS_NAMES)]
    allow = torch.tensor(list(allowed), dtype=torch.long)
    tok = allow[torch.randint(0, allow.numel(), (n_seq, T),
                              generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))]
    pool = torch.tensor(interesting, dtype=torch.long)
    sel = pool[torch.randint(0, pool.numel(), (n_seq, T),
                             generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))]
    mix = torch.rand(n_seq, T, generator=torch.Generator().manual_seed(rng.randrange(1 << 30))) < 0.5
    tok = torch.where(mix, sel, tok)
    # force exact ids into the first columns so every offset has known hits
    tok[:, 0] = 2047
    for j in range(1, min(10, T)):
        tok[:, j] = allowed[j * 7 % len(allowed)]
    return tok


def unambiguous_ids(vocab_tokens: Sequence[str]) -> List[int]:
    seen = {}
    for i, s in enumerate(vocab_tokens):
        seen.setdefault(s, []).append(i)
    return [i for i, s in enumerate(vocab_tokens) if len(seen[s]) == 1]


def run_suite(vocab_tokens: Sequence[str], classes: Dict[str, List[int]],
              specs: List[dict], tok: torch.Tensor, label: str) -> List[dict]:
    n_seq, T = tok.shape
    rows = [[vocab_tokens[int(i)] for i in row] for row in tok]
    results = []
    for si, spec in enumerate(specs):
        validate_spec(spec)
        expected = eval_spec(spec, tok, classes)
        py = spec_to_python(spec, vocab_tokens, classes)
        f = _exec_f(py)
        checked = mism = 0
        first = None
        for r in range(n_seq):
            row = rows[r]
            for p in range(T):
                want = bool(expected[r, p])
                try:
                    got = bool(f(row, p))
                except Exception as exc:
                    got = f"<error {type(exc).__name__}: {exc}>"
                checked += 1
                if got is not want:
                    mism += 1
                    if first is None:
                        first = {"row": r, "pos": p, "expected": want, "got": got, "py": py.strip()}
        ok = check_python(spec, py, tok, classes, vocab_tokens)  # exercise the public checker
        results.append({"suite": label, "op": spec["op"], "idx": si, "checked": checked,
                        "mismatches": mism, "ok": mism == 0, "check_python": bool(ok),
                        "first": first, "spec": spec, "py": py.strip()})
    return results


def per_class_suite(classes: Dict[str, List[int]]) -> List[dict]:
    specs = []
    for cname in CLASS_NAMES:
        specs.append({"op": "cls", "off": 0, "cls": cname})
        specs.append({"op": "win_cls", "lo": -8, "hi": -1, "cls": cname})
    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--specs-per-op", type=int, default=24)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--runs", default="runs")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    results: List[dict] = []

    vocabs = [("synthetic", synthetic_vocab(), None)]
    vpath = os.path.join(args.runs, args.model, "vocab.json")
    if os.path.exists(vpath):
        v = load_vocab(args.model, args.runs)
        vocabs.append(("engine:" + args.model, v["tokens"], v["classes"]))
    else:
        print(f"[check_predicate] {vpath} absent -> synthetic vocab only")

    for label, vocab_tokens, engine_classes in vocabs:
        classes = engine_classes if engine_classes is not None else build_classes(vocab_tokens)
        allowed = unambiguous_ids(vocab_tokens)
        if len(allowed) != len(vocab_tokens):
            seen: Dict[str, List[int]] = {}
            for i, s in enumerate(vocab_tokens):
                seen.setdefault(s, []).append(i)
            s0, ids0 = next((s, v) for s, v in seen.items() if len(v) > 1)
            print(f"[check_predicate] {label}: {len(vocab_tokens) - len(allowed)} of "
                  f"{len(vocab_tokens)} ids decode to a string shared with another id "
                  f"(e.g. {s0!r} <- ids {ids0[:4]}{'...' if len(ids0) > 4 else ''}); those ids are "
                  f"excluded from the token draws because no function of decoded strings can "
                  f"distinguish them")
        tok = make_tokens(rng, args.seqs, 256, vocab_tokens, allowed)
        specs: List[dict] = []
        for _ in range(args.specs_per_op):
            for op in ("tok_in", "cls", "win_in", "win_cls", "and", "or", "not"):
                while True:
                    s = random_spec(rng, allowed)
                    if s["op"] == op:
                        break
                specs.append(s)
        specs += edge_specs(allowed)
        specs += per_class_suite(classes)
        results += run_suite(vocab_tokens, classes, specs, tok, label)

    total_pos = sum(r["checked"] for r in results)
    bad = [r for r in results if not r["ok"] or not r["check_python"]]
    per_op: Dict[str, Dict[str, int]] = {}
    for r in results:
        d = per_op.setdefault(r["op"], {"specs": 0, "checked": 0, "mismatches": 0})
        d["specs"] += 1
        d["checked"] += r["checked"]
        d["mismatches"] += r["mismatches"]
    print(f"specs checked: {len(results)}  positions checked: {total_pos}")
    for op, d in sorted(per_op.items()):
        print(f"  {op:8s} specs={d['specs']:3d} positions={d['checked']:7d} mismatches={d['mismatches']}")
    if bad:
        print(f"FAIL: {len(bad)} spec(s) disagreed")
        for r in bad[:5]:
            print(json.dumps(r["first"], ensure_ascii=False))
        return 1
    print("PASS: 100% agreement between eval_spec and exec'd emitted source on all ops")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
