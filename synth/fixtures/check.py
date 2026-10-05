#!/usr/bin/env python3
"""Acceptance check for swa-synth on the synthetic fixture.

1. regenerate the fixture (unless --no-gen)
2. run `swa-synth --model fixture --root synth/fixtures/out --force`
3. planted recovery: each planted kind recovered with proxy score >= 0.95
4. emission parity: every emitted `py` equals engine/swa/predicate.spec_to_python
5. proxy exactness: eval_spec coverage == emitted cov, and check_python(spec, py) is True
6. throughput from the binary's log

Exit code 0 iff all checks pass.

    ~/sparse-weight-atlas/.venv/bin/python synth/fixtures/check.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "engine"))

from swa.predicate import (  # noqa: E402
    _set_lit,
    check_python_detail,
    class_py_expr,
    eval_spec,
    spec_to_python,
    validate_spec,
)

T = 256


def canon(spec):
    """Canonical form: recursively sort and/or args (order-agnostic spec equality)."""
    if not isinstance(spec, dict):
        return spec
    if spec.get("op") in ("and", "or"):
        return {"op": spec["op"], "args": sorted((canon(x) for x in spec["args"]), key=json.dumps)}
    return {k: canon(v) for k, v in spec.items()}


def load_candidates(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fired_positions(spec, tokens, classes):
    m = eval_spec(spec, tokens, classes).numpy()
    return np.flatnonzero(m.reshape(-1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--model", default="fixture")
    ap.add_argument("--shard", default="L00_fc_000")
    ap.add_argument("--bin", default=os.path.join(REPO, "synth", "target", "release", "swa-synth"))
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--no-gen", action="store_true")
    ap.add_argument("--check-python-sample", type=int, default=200)
    ap.add_argument("--check-python-positions", type=int, default=1500)
    a = ap.parse_args()

    if not a.no_gen:
        subprocess.run(
            [sys.executable, os.path.join(HERE, "make_fixture.py"), "--out", a.out, "--model", a.model,
             "--shard", a.shard],
            check=True,
        )

    cmd = [a.bin, "--model", a.model, "--root", a.out, "--force"]
    if a.threads:
        cmd += ["--threads", str(a.threads)]
    print("[check] $", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print("[check] FAIL: swa-synth exited", proc.returncode)
        return 1

    run_dir = os.path.join(a.out, "runs", a.model)
    truth = json.load(open(os.path.join(run_dir, "fixture_truth.json"), encoding="utf-8"))
    vocab = json.load(open(os.path.join(run_dir, "vocab.json"), encoding="utf-8"))
    tokens = vocab["tokens"]
    classes = vocab["classes"]
    slice_arr = np.load(os.path.join(a.out, "data", a.model, "slice_00.npy"))
    inputs = np.ascontiguousarray(slice_arr[:, :T])
    n_pos = int(inputs.size)
    assert n_pos == truth["n_pos"], (n_pos, truth["n_pos"])

    cand_path = os.path.join(run_dir, "candidates", f"{a.shard}.jsonl")
    entries = load_candidates(cand_path)
    by_wid = {int(e["wid"]): e.get("cands") or [] for e in entries}
    print(f"[check] {len(entries)} weight entries, {sum(len(v) for v in by_wid.values())} candidates")

    failures = []

    # ---- 3. planted recovery -----------------------------------------------------
    print("\n== planted recovery ==")
    for k in truth["kinds"]:
        wid = int(k["wid"])
        planted = k["spec"]
        want_pos = fired_positions(planted, inputs, classes)
        cands = by_wid.get(wid, [])
        best = None
        exact = None
        tight = None
        for c in cands:
            score = float(c["proxy"]["score"])
            got_pos = fired_positions(c["spec"], inputs, classes)
            recovers = np.isin(want_pos, got_pos).all()
            if recovers and (best is None or score > best[0]):
                best = (score, c, int(got_pos.size))
            if got_pos.size == want_pos.size and np.array_equal(got_pos, want_pos) and score >= 0.95:
                if tight is None or score > tight[0]:
                    tight = (score, c)
            if canon(c["spec"]) == canon(planted):
                exact = c
        ok = best is not None and best[0] >= 0.95 and tight is not None
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures.append(f"planted wid={wid} {k['name']}: not recovered (cands={len(cands)})")
        print(
            f"[{status}] wid={wid} {k['name']:16s} planted_cov={k['cov']:.4f} "
            + (
                f"recovered_score={best[0]:.6f} recovered_cov={best[2] / n_pos:.4f} "
                if best is not None
                else "recovered_score=-- recovered_cov=-- "
            )
            + f"exact_spec={'yes' if exact is not None else 'no'} "
            + f"same_fired_set={'yes' if tight is not None else 'no'} cands={len(cands)}"
        )
        if tight is not None:
            print(f"         tightest recovered spec: {json.dumps(tight[1]['spec'], sort_keys=True)}")
        if best is not None:
            print(f"         recovered spec: {json.dumps(best[1]['spec'], sort_keys=True)}")
            print(f"         desc: {best[1]['desc']}")
        if exact is not None:
            print(f"         exact planted spec present: score={float(exact['proxy']['score']):.6f} "
                  f"cov={float(exact['proxy']['cov']):.4f}")

    # ---- noise weights ----------------------------------------------------------
    print("\n== noise weights ==")
    for wid in truth["noise_wids"]:
        cands = by_wid.get(int(wid), [])
        ok = len(cands) == 0
        if not ok:
            failures.append(f"noise wid={wid}: expected cands: [], got {len(cands)}")
        print(f"[{'PASS' if ok else 'FAIL'}] wid={wid} cands={len(cands)} (expected 0)")

    # ---- extra weights (informational) ------------------------------------------
    extras = truth.get("extra_wids", [])
    if extras:
        scores = []
        for wid in extras:
            cands = by_wid.get(int(wid), [])
            scores.append(max((float(c["proxy"]["score"]) for c in cands), default=0.0))
        ge75 = sum(1 for s in scores if s >= 0.75)
        print(f"\n== extra weights (informational) ==")
        print(f"[INFO] {len(extras)} weights, best proxy score >= 0.75: {ge75} "
              f"({ge75 / len(extras):.1%}), median best score {np.median(scores):.3f}")

    # ---- emitter parity over the whole vocab (classes + repr) --------------------
    print("\n== emitter parity (all classes + repr over 2048 tokens) ==")
    st = subprocess.run(
        [a.bin, "--model", a.model, "--root", a.out, "--selftest-emit"],
        capture_output=True,
        text=True,
    )
    if st.returncode != 0:
        print("[FAIL] --selftest-emit failed:", st.stderr.strip())
        failures.append("selftest-emit failed")
    else:
        want_classes = {
            name: class_py_expr(name, "tokens[pos]", tokens, classes[name])
            for name in classes
        }
        got_classes = {}
        got_setlit = None
        for line in st.stdout.splitlines():
            parts = line.split("\t")
            if parts[0] == "CLASS":
                got_classes[parts[1]] = parts[2]
            elif parts[0] == "SETLIT":
                got_setlit = parts[1]
        bad = [n for n in want_classes if got_classes.get(n) != want_classes[n]]
        if bad:
            failures.append(f"class emission parity: {len(bad)} classes differ ({bad})")
        for n in bad[:3]:
            print(f"  class {n}:\n   got: {got_classes.get(n)!r}\n   want:{want_classes[n]!r}")
        print(f"[{'PASS' if not bad else 'FAIL'}] class expressions: {len(want_classes) - len(bad)}"
              f"/{len(want_classes)} identical to predicate.class_py_expr")
        want_sl = _set_lit(tokens)
        ok_sl = got_setlit == want_sl
        if not ok_sl:
            failures.append("set literal (repr) parity over all 2048 tokens differs")
        print(f"[{'PASS' if ok_sl else 'FAIL'}] token repr set literal over all 2048 tokens identical")

    # ---- 4./5. emission parity + proxy exactness --------------------------------
    all_cands = [(int(e["wid"]), c) for e in entries for c in (e.get("cands") or [])]
    print(f"\n== emission parity ({len(all_cands)} candidates) ==")
    py_mismatch = 0
    cov_mismatch = 0
    cp_fail = 0
    for wid, c in all_cands:
        validate_spec(c["spec"])
        want_py = spec_to_python(c["spec"], tokens, classes)
        if c["py"] != want_py:
            py_mismatch += 1
            if py_mismatch <= 3:
                print(f"  py mismatch wid={wid}:\n   got: {c['py']!r}\n   want:{want_py!r}")
    if py_mismatch:
        failures.append(f"py parity: {py_mismatch}/{len(all_cands)} mismatched spec_to_python")
    print(f"[{'PASS' if py_mismatch == 0 else 'FAIL'}] py parity: {py_mismatch} mismatches / {len(all_cands)}")

    # coverage exactness on every candidate (vectorized), check_python on a sample
    sample = all_cands[: a.check_python_sample] if a.check_python_sample else all_cands
    for wid, c in all_cands:
        fired = fired_positions(c["spec"], inputs, classes)
        cov = fired.size / n_pos
        if abs(cov - float(c["proxy"]["cov"])) > 1e-12:
            cov_mismatch += 1
            if cov_mismatch <= 3:
                print(f"  cov mismatch wid={wid}: spec={cov} emitted={c['proxy']['cov']}")
    for wid, c in sample:
        det = check_python_detail(
            c["spec"], c["py"], inputs, classes, tokens, max_positions=a.check_python_positions
        )
        if not det["ok"]:
            cp_fail += 1
            if cp_fail <= 3:
                print(f"  check_python FAIL wid={wid}: {det['first_mismatch']}")
    if cov_mismatch:
        failures.append(f"coverage: {cov_mismatch}/{len(all_cands)} mismatched eval_spec")
    if cp_fail:
        failures.append(f"check_python: {cp_fail}/{len(sample)} failed")
    print(f"[{'PASS' if cov_mismatch == 0 else 'FAIL'}] coverage == eval_spec: {cov_mismatch} mismatches / {len(all_cands)}")
    print(f"[{'PASS' if cp_fail == 0 else 'FAIL'}] check_python(spec, py): {cp_fail} failures / {len(sample)} sampled")

    # ---- 6. throughput ----------------------------------------------------------
    m = re.search(r"weights/s\)", proc.stderr)
    rate = re.search(r"in ([\d.]+)s \(([\d.]+) weights/s\)", proc.stderr)
    print("\n== throughput ==")
    if rate:
        print(f"[PASS] fixture: {rate.group(2)} weights/s ({rate.group(1)}s)")
    else:
        print("[FAIL] could not parse throughput from log")
        failures.append("throughput not logged")
    _ = m

    print("\n== summary ==")
    if failures:
        for f in failures:
            print("  FAIL:", f)
        print(f"RESULT: FAIL ({len(failures)} issues)")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
