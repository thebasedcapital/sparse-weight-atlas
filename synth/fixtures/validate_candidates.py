#!/usr/bin/env python3
"""Validate emitted candidates through the Python predicate engine (CONTRACT gate).

For a sample of candidates in runs/<model>/candidates/*.jsonl:
  * eval_spec coverage must equal the emitted proxy cov (exact)
  * check_python(spec, py) must be True
  * py must equal swa.predicate.spec_to_python(spec, tokens, classes)
  * the proxy (rec/inv/cov/score) must be reproduced from the shard CSR + dce_total

    ~/sparse-weight-atlas/.venv/bin/python synth/fixtures/validate_candidates.py --model csp_yolo1 -n 50
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "engine"))

from swa.predicate import check_python, eval_spec, spec_to_python, validate_spec  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--root", default=REPO)
    ap.add_argument("-n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-positions", type=int, default=2000)
    a = ap.parse_args()

    runs = os.path.join(a.root, "runs", a.model)
    vocab = json.load(open(os.path.join(runs, "vocab.json"), encoding="utf-8"))
    tokens, classes = vocab["tokens"], vocab["classes"]
    sl = np.load(os.path.join(a.root, "data", a.model, "slice_00.npy"))
    inputs = np.ascontiguousarray(sl[:, :256])
    n_pos = int(inputs.size)

    shards, wid_to_shard = {}, {}
    for d in sorted(glob.glob(os.path.join(runs, "profiles", "*"))):
        if not os.path.exists(os.path.join(d, "wid.npy")):
            continue  # shard being written by the profiler
        name = os.path.basename(d)
        shards[name] = (
            np.load(os.path.join(d, "wid.npy")),
            np.load(os.path.join(d, "indptr.npy")),
            np.load(os.path.join(d, "pos.npy")),
            np.load(os.path.join(d, "dce.npy")),
            np.load(os.path.join(d, "dce_total.npy")),
        )
        for w in shards[name][0]:
            wid_to_shard[int(w)] = name

    cands = [
        (int(e["wid"]), c)
        for f in sorted(glob.glob(os.path.join(runs, "candidates", "*.jsonl")))
        for e in (json.loads(line) for line in open(f, encoding="utf-8"))
        for c in (e.get("cands") or [])
    ]
    if not cands:
        print(f"[validate] no candidates under {runs}/candidates", file=sys.stderr)
        return 1
    n = min(a.n, len(cands))
    cp_idx = set(int(i) for i in np.random.default_rng(a.seed).choice(len(cands), size=n, replace=False))
    print(f"[validate] {a.model}: {len(cands)} emitted candidates, "
          f"cov/py/proxy on all, check_python on {n}")

    cov_ok = py_ok = proxy_ok = cp_ok = cp_n = 0
    bad = []
    for i, (wid, c) in enumerate(cands):
        spec = c["spec"]
        validate_spec(spec)
        fired = eval_spec(spec, inputs, classes).numpy().reshape(-1)
        cov = float(fired.sum()) / n_pos
        if abs(cov - float(c["proxy"]["cov"])) < 1e-12:
            cov_ok += 1
        else:
            bad.append(("cov", wid, cov, c["proxy"]["cov"]))
        if c["py"] == spec_to_python(spec, tokens, classes):
            py_ok += 1
        else:
            bad.append(("py", wid))
        wids, indptr, pos, dce, dct = shards[wid_to_shard[wid]]
        j = int(np.flatnonzero(wids == wid)[0])
        a0, b0 = int(indptr[j]), int(indptr[j + 1])
        kept = pos[a0:b0].astype(np.int64)
        dk = dce[a0:b0].astype(np.float64)
        in_f = float(dk[fired[kept]].sum())
        D = float(dct[j])
        rec = 1.0 - (D - in_f) / D
        inv = 1.0 - in_f / D
        sc = min(rec, 1.0 - inv)
        p = c["proxy"]
        if max(abs(rec - p["rec"]), abs(inv - p["inv"]), abs(sc - p["score"])) / max(1.0, abs(sc)) < 1e-5:
            proxy_ok += 1
        else:
            bad.append(("proxy", wid, (rec, inv, sc), (p["rec"], p["inv"], p["score"])))
        if i in cp_idx:
            cp_n += 1
            if check_python(spec, c["py"], inputs, classes, tokens, max_positions=a.max_positions):
                cp_ok += 1
            else:
                bad.append(("check_python", wid))

    m = len(cands)
    print(f"[{'PASS' if cov_ok == m else 'FAIL'}] cov == eval_spec coverage: {cov_ok}/{m}")
    print(f"[{'PASS' if py_ok == m else 'FAIL'}] py == spec_to_python: {py_ok}/{m}")
    print(f"[{'PASS' if proxy_ok == m else 'FAIL'}] proxy recomputed from CSR: {proxy_ok}/{m}")
    print(f"[{'PASS' if cp_ok == cp_n else 'FAIL'}] check_python(spec, py): {cp_ok}/{cp_n}")
    if bad:
        print("[validate] mismatches:", bad[:5])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
