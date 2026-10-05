#!/usr/bin/env python
"""Generate a realistic-size synthetic fixture model under site/data/fixture/.

Mimics the csp_yolo1 shape exactly (12 layers, 393,216 nonzero MLP weights, 32,768 per
layer, c_fc + c_proj) with random statuses, scores, fake predicate specs and decoded
contexts, so the atlas UI can be developed and perf-tested without pipeline outputs.

Usage:
    python site/fixtures/make_fixture.py [--out site/data] [--seed 42]
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

N_LAYERS = 12
PER_LAYER = 32768
TOTAL = N_LAYERS * PER_LAYER  # 393,216
D_MODEL, D_MLP = 1024, 4096
FC_PER_LAYER = 20480  # rest of the layer's 32,768 are c_proj
BUCKET = 1000
PAPER_ROBUST_RATE = 0.15

STATUS_POOL = [("scored", 0.55), ("noise_floor", 0.20),
               ("no_candidate", 0.15), ("unscored", 0.10)]

CLASSES = ["digit_only", "has_digit", "alpha_only", "upper_start", "lower_only",
           "space_lead", "newline", "indent", "punct_only", "quote", "bracket_open",
           "bracket_close", "operator", "dot", "comma", "colon", "underscore",
           "keyword", "eot"]

TOKEN_POOL = ["def", " ", "self", "(", ")", ":", "=", ".", "import", "return", "1",
              "0", "00", "100", "'", '"', "print", "range", "len", "for", "in", "if",
              "    ", "\n", "[", "]", ",", "+", "x", "y", "name", "str", "int", "#",
              "True", "None", "try", "except", "class", "os", "path", "join", "0.5",
              "append", "split", "encode", "True", "while", "break", "not", "and"]

DESC_TEMPLATES = [
    "fires on {q} tokens",
    "previous token is {q}",
    "inside {q} literals",
    "start of an indented {q} block",
    "after a {q} operator",
    "closes a {q} bracket group",
    "mid-word lowercase fragment {q}",
    "digit run of length {n}",
]


def make_spec(rng, simple=False):
    kinds = ["tok_in", "cls", "win_in", "win_cls"] if simple else \
        ["tok_in", "cls", "win_in", "win_cls", "and", "or"]
    kind = rng.choice(kinds)
    ids = sorted(rng.sample(range(2048), rng.randint(2, 11)))
    idset = "{" + ", ".join(map(str, ids)) + "}"
    if kind == "tok_in":
        off = rng.randint(-2, 0)
        spec = {"op": "tok_in", "off": off, "ids": ids}
        tgt = "tokens[pos]" if off == 0 else f"tokens[pos{off:+d}]"
        py = f"def f(tokens, pos):\n    return {tgt} in {idset}"
        desc = rng.choice(DESC_TEMPLATES).format(q="membership", n=3)
        return spec, py, desc
    if kind == "cls":
        cls = rng.choice(CLASSES)
        off = rng.randint(-2, 0)
        spec = {"op": "cls", "off": off, "cls": cls}
        tgt = "tokens[pos]" if off == 0 else f"tokens[pos{off:+d}]"
        py = f"def f(tokens, pos):\n    return {tgt}.isdigit() is {cls == 'digit_only'} or _cls({tgt}) == {cls!r}"
        desc = f"focus token is {cls}"
        return spec, py, desc
    if kind in ("win_in", "win_cls"):
        lo, hi = sorted([rng.randint(-8, -1), rng.randint(-8, -1)])
        if lo == hi:
            hi = lo + 1
        if kind == "win_in":
            spec = {"op": "win_in", "lo": lo, "hi": hi, "ids": ids}
            py = (f"def f(tokens, pos):\n    return any(tokens[j] in {idset} "
                  f"for j in range(pos{lo:+d}, pos{hi:+d}))")
            desc = f"a token in the previous {-lo} tokens is in a {len(ids)}-token set"
        else:
            cls = rng.choice(CLASSES)
            spec = {"op": "win_cls", "lo": lo, "hi": hi, "cls": cls}
            py = (f"def f(tokens, pos):\n    return any(_cls(tokens[j]) == {cls!r} "
                  f"for j in range(pos{lo:+d}, pos{hi:+d}))")
            desc = f"some token in the previous {-lo} tokens is {cls}"
        return spec, py, desc
    a_spec, a_py, a_desc = make_spec(rng, simple=True)
    # keep nesting shallow: combine with a simple cls test
    cls = rng.choice(CLASSES)
    b_spec = {"op": "cls", "off": 0, "cls": cls}
    b_py = f"def f(tokens, pos):\n    return _cls(tokens[pos]) == {cls!r}"
    op = "and" if kind == "and" else "or"
    spec = {"op": op, "args": [a_spec, b_spec]}
    py = f"def f(tokens, pos):\n    return ({a_py.splitlines()[1].strip()}) {op} ({b_py.splitlines()[1].strip()})"
    desc = f"{a_desc} ({op} focus is {cls})"
    return spec, py, desc


def make_scores(rng, robust):
    cov = float(rng.uniform(0.002, 0.5))
    if robust:
        rec, inv = float(rng.uniform(0.8, 0.98)), float(rng.uniform(0.02, 0.18))
    else:
        rec, inv = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.1, 0.9))
    score = min(rec, 1.0 - inv)

    def block():
        r = rec + float(rng.normal(0, 0.05))
        i = inv + float(rng.normal(0, 0.05))
        return {"rec": r, "inv": i, "cov": cov, "score": min(r, 1.0 - i)}

    s10 = [min(max(score + float(rng.normal(0, 0.04)), 0.0), 1.0) for _ in range(10)]
    mean = float(np.mean(s10))
    sem = float(np.std(s10, ddof=1) / np.sqrt(10)) if len(s10) > 1 else 0.0
    null_mean, null_sd = float(rng.uniform(0.05, 0.3)), float(rng.uniform(0.01, 0.08))
    rob = {"scores": s10, "mean": mean, "sem": sem,
           "null_mean": null_mean, "null_sd": null_sd,
           "pass": bool(mean >= 0.75 and mean - 2 * sem > null_mean + 2 * null_sd)}
    return block(), block(), rob


def make_contexts(rng):
    out = []
    for _ in range(3):
        n = int(rng.integers(9, 17))
        toks = [rng.choice(TOKEN_POOL) for _ in range(n)]
        out.append({"tokens": toks, "focus": n // 2,
                    "dce": float(10 ** rng.uniform(-6, -1)),
                    "kl": float(10 ** rng.uniform(-4, 1.5))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    pyrng = random.Random(args.seed)
    out_dir = Path(args.out) / "fixture"
    (out_dir / "w").mkdir(parents=True, exist_ok=True)

    statuses = rng.choice([s for s, _ in STATUS_POOL], p=[p for _, p in STATUS_POOL],
                          size=TOTAL)
    robust_flags = rng.random(TOTAL) < PAPER_ROBUST_RATE
    dce = 10.0 ** rng.uniform(-8, -2, size=TOTAL)
    values = rng.normal(0, 0.02, size=TOTAL)

    counts = {s: 0 for s in ("scored", "noise_floor", "no_candidate", "unscored")}
    robust_pass = 0
    per_layer = []
    layer_rows = []
    bucket_cards = {}
    a_recs, b_recs = [], []

    def flush_layer(L):
        (out_dir / f"layer_{L:02d}.json").write_text(
            json.dumps({"layer": L, "n": len(layer_rows), "w": layer_rows},
                       separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        per_layer.append({"layer": L, "total": len(layer_rows), **dict(counts_local),
                          "pass": pass_local})
        layer_rows.clear()

    def flush_bucket(b):
        (out_dir / "w" / f"{b}.json").write_text(
            json.dumps({"bucket": b, "w": bucket_cards.pop(b)},
                       separators=(",", ":"), ensure_ascii=False), encoding="utf-8")

    counts_local = {}
    pass_local = 0
    for wid in range(TOTAL):
        L = wid // PER_LAYER
        m = 0 if (wid % PER_LAYER) < FC_PER_LAYER else 1
        if m == 0:
            r = int(rng.integers(0, D_MLP)); c = int(rng.integers(0, D_MODEL))
        else:
            r = int(rng.integers(0, D_MODEL)); c = int(rng.integers(0, D_MLP))

        status = str(statuses[wid])
        counts[status] += 1
        counts_local[status] = counts_local.get(status, 0) + 1
        rob = None
        A = B = None
        spec = py = desc = None
        if status == "scored":
            robust = bool(robust_flags[wid])
            A, B, rob = make_scores(rng, robust)
            a_recs.append(A); b_recs.append(B)
            spec, py, desc = make_spec(pyrng)
            if robust:
                robust_pass += 1
                pass_local += 1
        dce_v = float(dce[wid]) if status != "unscored" else None

        layer_rows.append({
            "wid": wid, "matrix": m, "row": r, "col": c, "value": float(values[wid]),
            "dce": dce_v, "status": status,
            "score_robust_mean": rob["mean"] if rob else None,
            "pass": bool(rob and rob["pass"]), "desc": desc,
        })

        bucket_cards.setdefault(wid // BUCKET, {})[wid] = {
            "wid": wid, "layer": L,
            "matrix": m, "matrix_name": "c_fc" if m == 0 else "c_proj",
            "row": r, "col": c, "value": float(values[wid]),
            "status": status, "dce": dce_v, "spec": spec, "py": py, "desc": desc,
            "A": A, "B": B, "robust": rob,
            "contexts": make_contexts(rng) if status == "scored" else [],
        }

        if (wid + 1) % BUCKET == 0:
            flush_bucket(wid // BUCKET)
        if (wid + 1) % PER_LAYER == 0:
            flush_layer(L)
            counts_local = {}
            pass_local = 0

    def rate(recs):
        if not recs:
            return None
        n = sum(1 for r in recs if r["score"] >= 0.75 and 0 < r["cov"] <= 0.5)
        return n / len(recs)

    bin_edges = [1e-6, 1e-5, 1e-4]
    bin_labels = ["<1e-6", "1e-6..1e-5", "1e-5..1e-4", ">=1e-4"]
    bins = [{"bin": b, "total": 0, "pass": 0} for b in bin_labels]
    for wid in range(TOTAL):
        if statuses[wid] == "unscored":
            continue
        adce = abs(float(dce[wid]))
        bi = 0 if adce < bin_edges[0] else 1 if adce < bin_edges[1] \
            else 2 if adce < bin_edges[2] else 3
        bins[bi]["total"] += 1
        if robust_flags[wid] and statuses[wid] == "scored":
            bins[bi]["pass"] += 1

    verdicted = counts["scored"] + counts["noise_floor"] + counts["no_candidate"]
    summary = {
        "model": "fixture",
        "fixture": True,
        "counts": {"total": TOTAL, "scored": counts["scored"],
                   "noise_floor": counts["noise_floor"],
                   "no_candidate": counts["no_candidate"],
                   "unscored": counts["unscored"], "robust_pass": robust_pass},
        "rates": {"robust": (robust_pass / verdicted) if verdicted else None,
                  "robust_of_total": robust_pass / TOTAL,
                  "interpA": rate(a_recs), "interpB": rate(b_recs)},
        "paper": {"robust": PAPER_ROBUST_RATE, "interpA": 0.373, "interpB": 0.353,
                  "T": 0.75, "c": 0.5},
        "per_layer": per_layer,
        "per_bin": bins,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    manifest_path = Path(args.out) / "models.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text()).get("models", [])
        except (json.JSONDecodeError, OSError):
            manifest = []
    entry = {"id": "fixture", "summary": "data/fixture/summary.json", "fixture": True}
    manifest = [m for m in manifest if m.get("id") != "fixture"] + [entry]
    manifest_path.write_text(json.dumps({"models": manifest}, indent=1), encoding="utf-8")

    print(f"[make_fixture] {TOTAL} weights -> {out_dir} "
          f"(scored={counts['scored']}, robust_pass={robust_pass}, "
          f"rate={robust_pass / verdicted:.3f} of verdicted)")


if __name__ == "__main__":
    main()
