#!/usr/bin/env python3
"""Generate a synthetic profile shard + slice + vocab with planted ground truth.

Writes a self-contained pipeline root under ``synth/fixtures/out``:

    data/fixture/slice_00.npy                    uint16 [n_seq, 257]
    runs/fixture/vocab.json                      copy of runs/csp_yolo1/vocab.json
    runs/fixture/profiles/L00_fc_000/            CSR shard, one weight per planted kind
    runs/fixture/fixture_truth.json              planted specs + position counts

Planted kinds (dce = 1.0 exactly at the kind's positions, 0 elsewhere):
    0  token in {A, B, C} at offset 0
    1  previous token == Q           (tok_in off -1)
    2  a quote token within the last 5 tokens (win_cls lo=-5 hi=-1 cls=quote)
    3  class digit_only at offset 0 AND prev token '('
    4,5 noise weights (dce_total == 0 -> synth must emit "cands": [])

Run with the project venv:
    ~/sparse-weight-atlas/.venv/bin/python synth/fixtures/make_fixture.py
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

T = 256


def shift(inp: np.ndarray, k: int, fill: int = -1) -> np.ndarray:
    """value at (r, t) = inp[r, t - k]; fill when t - k < 0."""
    out = np.full_like(inp, fill)
    if k > 0:
        out[:, k:] = inp[:, :-k]
    return out


def shift_at(inp: np.ndarray, off: int, fill: int = -1) -> np.ndarray:
    """value at (r, t) = inp[r, t + off] (off in [-8, 0]); fill out of range."""
    if off == 0:
        return inp.copy()
    out = np.full_like(inp, fill)
    k = -off
    out[:, : inp.shape[1] - k] = inp[:, k:]
    return out


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--model", default="fixture")
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--vocab", default=os.path.join(repo, "runs", "csp_yolo1", "vocab.json"))
    ap.add_argument("--shard", default="L00_fc_000")
    ap.add_argument(
        "--extra",
        type=int,
        default=256,
        help="number of extra non-planted weights (diffuse dce + a planted spot) for throughput",
    )
    a = ap.parse_args()

    vocab = json.load(open(a.vocab, encoding="utf-8"))
    tokens = vocab["tokens"]
    classes = vocab["classes"]
    tid = {s: i for i, s in enumerate(tokens)}

    A, B, C, Q = tid["A"], tid["B"], tid["C"], tid["Q"]
    LPAREN = tid["("]
    QUOTE = tid["'"]
    DIGITS = [tid[str(d)] for d in range(10)]
    FILLER = [tid[c] for c in ["x", "y", "z", "w", "v", "u", "t", "s", "r", "q"]]
    UFFFD = [i for i, s in enumerate(tokens) if s == "\ufffd"]
    assert len(UFFFD) == 128, len(UFFFD)
    for name, ids in [("A/B/C", [A, B, C]), ("Q", [Q]), ("(", [LPAREN]), ("'", [QUOTE])]:
        assert all(tokens[i] not in ("\ufffd",) for i in ids), name
    assert set(DIGITS) <= set(classes["digit_only"])
    assert QUOTE in classes["quote"] and LPAREN not in classes["quote"]

    rng = np.random.default_rng(a.seed)
    n_seq = a.n_seq
    cols = T + 1
    n_pos = n_seq * T

    # categorical token draw per position
    cats = [
        (0.15, [A, B, C]),
        (0.12, [Q]),
        (0.25, DIGITS),
        (0.06, [LPAREN]),
        (0.04, [QUOTE]),
        (0.10, UFFFD),
        (0.28, FILLER),
    ]
    tok = np.zeros((n_seq, cols), dtype=np.uint16)
    u = rng.random((n_seq, cols))
    acc = np.zeros((n_seq, cols))
    for p, ids in cats:
        sel = rng.integers(0, len(ids), size=(n_seq, cols))
        vals = np.asarray(ids, dtype=np.uint16)[sel]
        tok = np.where((u >= acc) & (u < acc + p), vals, tok).astype(np.uint16)
        acc += p
    # any position left unassigned by float rounding -> filler
    tok = np.where(u >= acc, np.uint16(FILLER[0]), tok).astype(np.uint16)

    inp = tok[:, :T].astype(np.int64)

    mask_k1 = np.isin(inp, [A, B, C])
    mask_k2 = shift(inp, 1) == Q
    quote_win = np.zeros_like(inp, dtype=bool)
    for k in range(1, 6):
        quote_win |= shift(inp, k) == QUOTE
    mask_k3 = quote_win
    mask_k4 = np.isin(inp, DIGITS) & (shift(inp, 1) == LPAREN)
    mask_k5 = np.isin(inp, UFFFD)

    kinds = [
        (
            "tok_set_ABC",
            {"op": "tok_in", "off": 0, "ids": [A, B, C]},
            mask_k1,
        ),
        (
            "prev_Q",
            {"op": "tok_in", "off": -1, "ids": [Q]},
            mask_k2,
        ),
        (
            "quote_within_5",
            {"op": "win_cls", "lo": -5, "hi": -1, "cls": "quote"},
            mask_k3,
        ),
        (
            "digit_and_lparen",
            {
                "op": "and",
                "args": [
                    {"op": "cls", "off": 0, "cls": "digit_only"},
                    {"op": "tok_in", "off": -1, "ids": [LPAREN]},
                ],
            },
            mask_k4,
        ),
        (
            "ufffd_group",
            {"op": "tok_in", "off": 0, "ids": sorted(UFFFD)},
            mask_k5,
        ),
    ]

    noise = [5, 6]
    n_planted = len(kinds)
    n_base = n_planted + len(noise)
    n_w = n_base + a.extra
    wid = np.arange(n_w, dtype=np.int32)

    pos_parts = []
    dce_parts = []
    kl_parts = []
    indptr = [0]
    dce_total = np.zeros(n_w, dtype=np.float32)
    dce_rest = np.zeros(n_w, dtype=np.float32)
    kl_total = np.zeros(n_w, dtype=np.float32)
    kl_max = np.zeros(n_w, dtype=np.float32)
    truth_kinds = []
    for i, (name, spec, mask) in enumerate(kinds):
        flat = np.flatnonzero(mask.reshape(-1)).astype(np.int32)
        assert flat.size <= 4096, f"{name}: {flat.size} kept positions > 4096 cap"
        assert 0 < flat.size, name
        d = np.ones(flat.size, dtype=np.float32)
        pos_parts.append(flat)
        dce_parts.append(d)
        kl_parts.append(d.copy())
        indptr.append(indptr[-1] + flat.size)
        dce_total[i] = float(d.sum())
        kl_total[i] = float(d.sum())
        kl_max[i] = 1.0
        truth_kinds.append(
            {
                "wid": i,
                "name": name,
                "spec": spec,
                "n_positions": int(flat.size),
                "cov": float(flat.size) / n_pos,
            }
        )
    for _ in noise:
        indptr.append(indptr[-1])

    # extra weights: diffuse dce + one random concentrated spot, kept = top 4096 by |dce|
    rng2 = np.random.default_rng(a.seed + 1)
    pool_ids = FILLER + [A, B, C, Q, LPAREN, QUOTE] + DIGITS
    extra_wids = []
    for e in range(a.extra):
        i = n_base + e
        k = int(rng2.integers(1, 5))
        ids = rng2.choice(pool_ids, size=k, replace=False)
        off = int(rng2.integers(-8, 1))
        spot = np.isin(shift_at(inp, off), ids)
        full = rng2.standard_normal((n_seq, T)) + 3.0 * spot
        flat_full = full.reshape(-1)
        order = np.argsort(-np.abs(flat_full), kind="stable")[:4096]
        order.sort()
        kept = order.astype(np.int32)
        dk = flat_full[kept].astype(np.float32)
        pos_parts.append(kept)
        dce_parts.append(dk)
        kl_parts.append(np.abs(dk))
        indptr.append(indptr[-1] + kept.size)
        dce_total[i] = float(flat_full.sum())
        dce_rest[i] = float(flat_full.sum() - dk.sum())
        kl_total[i] = float(np.abs(dk).sum())
        kl_max[i] = float(np.abs(dk).max())
        extra_wids.append(i)

    pos = np.concatenate(pos_parts) if pos_parts else np.zeros(0, dtype=np.int32)
    dce = np.concatenate(dce_parts) if dce_parts else np.zeros(0, dtype=np.float32)
    kl = np.concatenate(kl_parts) if kl_parts else np.zeros(0, dtype=np.float32)

    t0 = time.time()
    data_dir = os.path.join(a.out, "data", a.model)
    run_dir = os.path.join(a.out, "runs", a.model)
    shard_dir = os.path.join(run_dir, "profiles", a.shard)
    for d in (data_dir, shard_dir):
        os.makedirs(d, exist_ok=True)

    np.save(os.path.join(data_dir, "slice_00.npy"), tok)
    with open(os.path.join(run_dir, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False)

    np.save(os.path.join(shard_dir, "wid.npy"), wid)
    np.save(os.path.join(shard_dir, "dce_total.npy"), dce_total)
    np.save(os.path.join(shard_dir, "kl_total.npy"), kl_total)
    np.save(os.path.join(shard_dir, "kl_max.npy"), kl_max)
    np.save(os.path.join(shard_dir, "indptr.npy"), np.asarray(indptr, dtype=np.int64))
    np.save(os.path.join(shard_dir, "pos.npy"), pos)
    np.save(os.path.join(shard_dir, "dce.npy"), dce)
    np.save(os.path.join(shard_dir, "kl.npy"), kl)
    np.save(os.path.join(shard_dir, "dce_rest.npy"), dce_rest)
    with open(os.path.join(shard_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": a.model,
                "slice": 0,
                "dtype": "float32",
                "n_positions": n_pos,
                "n_weights": n_w,
                "timing": {"wall_s": time.time() - t0},
                "complete": True,
                "synthetic": True,
            },
            fh,
            indent=2,
        )

    with open(os.path.join(run_dir, "fixture_truth.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "model": a.model,
                "shard": a.shard,
                "n_seq": n_seq,
                "n_pos": n_pos,
                "seed": a.seed,
                "tokens": {"A": A, "B": B, "C": C, "Q": Q, "LPAREN": LPAREN, "QUOTE": QUOTE, "DIGITS": DIGITS},
                "kinds": truth_kinds,
                "noise_wids": noise,
                "extra_wids": extra_wids,
            },
            fh,
            indent=2,
        )

    print(f"[fixture] wrote {a.out}")
    for k in truth_kinds:
        print(f"[fixture] wid={k['wid']} {k['name']:16s} n={k['n_positions']:5d} cov={k['cov']:.4f} {k['spec']}")
    print(f"[fixture] noise wids={noise} (dce_total=0)")
    print(f"[fixture] extra weights: {len(extra_wids)} (wids {n_base}..{n_w - 1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
