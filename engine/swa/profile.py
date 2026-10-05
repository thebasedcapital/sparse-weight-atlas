"""Whole-model weight profiling: per-weight ΔCE and KL on the discovery slice.

CLI
---
``python -m swa.profile --model csp_yolo1 [--layers 0,1] [--limit N] [--wids-file f.npy]``
writes ``runs/<model>/profiles/<shard>/`` for slice 00, where
``shard = L<layer:02d>_<fc|proj>_<chunk:03d>`` and each shard holds up to 4096 wids.  Runs
are resumable: a shard is skipped when its ``meta.json`` exists with ``"complete": true``.

``--triage`` computes a first-order attribution for every MLP weight in one forward +
backward pass per slice and writes ``runs/<model>/triage.npy`` (float32, ``[W]``, wid order).
It is an **estimate**: ``|W * dL/dW|`` ignores higher-order terms, the data-dependent top-k
selection boundaries, and cross-position propagation, so it is only a cheap ordering /
skip heuristic -- never a substitute for the exact ablation.

``--bench`` measures exact-ablation throughput (slice 00, early/mid/late layer) and the
triage wall time, writing ``runs/<model>/bench.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from swa.model import DATA, ROOT, d_mlp, load_model, n_layers
from swa.replay import Replayer
from swa.weights import load_weights

T = 256
SHARD_WIDS = 4096
KEEP_CAP = 4096
ABLATE_BATCH = 256


# --------------------------------------------------------------------- shard helpers

def shard_name(layer: int, matrix: int, chunk: int) -> str:
    return f"L{layer:02d}_{'fc' if matrix == 0 else 'proj'}_{chunk:03d}"


def shard_map(w: dict) -> dict[str, list[int]]:
    """wid -> home shard, using the contract ordering (layer, matrix, nonzero order)."""
    layer, matrix = w["layer"], w["matrix"]
    W = len(layer)
    starts: dict[tuple[int, int], int] = {}
    for i in range(W):
        key = (int(layer[i]), int(matrix[i]))
        starts.setdefault(key, i)
    out: dict[str, list[int]] = {}
    for key, start in sorted(starts.items()):
        L, M = key
        sel = np.flatnonzero((layer == L) & (matrix == M))
        for c in range(0, len(sel), SHARD_WIDS):
            out[shard_name(L, M, c // SHARD_WIDS)] = sel[c:c + SHARD_WIDS].tolist()
    return out


def kept_positions(dce_row: np.ndarray, kl_row: np.ndarray, cap: int = KEEP_CAP) -> np.ndarray:
    """union(top-90%-cumulative-KL positions, 99%-of-Σ|dce| positions), capped at ``cap``."""
    n = dce_row.shape[0]
    keep = np.zeros(n, dtype=bool)
    tot_kl = float(kl_row.sum())
    if tot_kl > 0:
        order = np.argsort(-kl_row, kind="stable")
        cum = np.cumsum(kl_row[order])
        k = int(np.searchsorted(cum, 0.9 * tot_kl, side="left")) + 1
        keep[order[:k]] = True
    adce = np.abs(dce_row)
    tot_d = float(adce.sum())
    if tot_d > 0:
        order = np.argsort(-adce, kind="stable")
        cum = np.cumsum(adce[order])
        k = int(np.searchsorted(cum, 0.99 * tot_d, side="left")) + 1
        keep[order[:k]] = True
    idx = np.flatnonzero(keep)
    if idx.size > cap:
        score = adce[idx] + kl_row[idx]
        sel = np.argsort(-score, kind="stable")[:cap]
        idx = np.sort(idx[sel])
    return idx


# --------------------------------------------------------------------- shard run

def run_shard(r: Replayer, model_name: str, shard: str, wids: list[int],
              force: bool = False, partial: bool = False) -> dict:
    out = ROOT / "runs" / model_name / "profiles" / shard
    meta_path = out / "meta.json"
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text())
        if meta.get("complete"):
            return {"shard": shard, "skipped": True, **meta}
    if partial and meta_path.exists():
        # a partial (--wids-file) run must never replace a complete shard's profile, even
        # with --force: it would silently shrink the shard to a handful of wids
        meta = json.loads(meta_path.read_text())
        if meta.get("complete"):
            print(f"[profile] {shard}: refusing to overwrite a complete profile with a "
                  f"partial (--wids-file) run", file=sys.stderr, flush=True)
            return {"shard": shard, "skipped": True, **meta}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = len(wids)
    n_pos = r.n_seq * T
    dce_all = np.empty((n, n_pos), dtype=np.float32)
    kl_all = np.empty((n, n_pos), dtype=np.float32)
    ce0 = r.ce_base().cpu().numpy().reshape(-1).astype(np.float32)
    for s in range(0, n, ABLATE_BATCH):
        e = min(s + ABLATE_BATCH, n)
        ce, kl = r.ablate(torch.tensor(wids[s:e]), want_kl=True)
        dce_all[s:e] = (ce.cpu().numpy().reshape(e - s, n_pos) - ce0[None, :])
        kl_all[s:e] = kl.cpu().numpy().reshape(e - s, n_pos)
        del ce, kl
    dce_total = dce_all.sum(1).astype(np.float32)
    kl_total = kl_all.sum(1).astype(np.float32)
    kl_max = kl_all.max(1).astype(np.float32)
    indptr = np.zeros(n + 1, dtype=np.int64)
    pos_parts, dce_parts, kl_parts = [], [], []
    dce_rest = np.empty(n, dtype=np.float32)
    for i in range(n):
        idx = kept_positions(dce_all[i], kl_all[i])
        pos_parts.append(idx.astype(np.int32))
        dce_parts.append(dce_all[i][idx])
        kl_parts.append(kl_all[i][idx])
        indptr[i + 1] = indptr[i] + idx.size
        mask = np.ones(n_pos, dtype=bool)
        mask[idx] = False
        dce_rest[i] = dce_all[i][mask].sum()
    np.save(out / "wid.npy", np.asarray(wids, dtype=np.int32))
    np.save(out / "dce_total.npy", dce_total)
    np.save(out / "kl_total.npy", kl_total)
    np.save(out / "kl_max.npy", kl_max)
    np.save(out / "indptr.npy", indptr)
    np.save(out / "pos.npy", np.concatenate(pos_parts) if pos_parts else np.zeros(0, np.int32))
    np.save(out / "dce.npy", np.concatenate(dce_parts) if dce_parts else np.zeros(0, np.float32))
    np.save(out / "kl.npy", np.concatenate(kl_parts) if kl_parts else np.zeros(0, np.float32))
    np.save(out / "dce_rest.npy", dce_rest)
    elapsed = time.time() - t0
    meta = {
        "model": model_name,
        "slice": r.slice,
        "shard": shard,
        "n": n,
        "dtype": "float32",
        "n_positions": n_pos,
        "n_seq": r.n_seq,
        "T": T,
        "complete": not partial,
        "partial": partial,
        "elapsed_s": elapsed,
        "weights_per_sec": n / elapsed if elapsed > 0 else None,
        "timing": {"start": t0, "elapsed_s": elapsed},
    }
    meta_path.write_text(json.dumps(meta, indent=1))
    return meta


# --------------------------------------------------------------------- triage

def triage(model_name: str, device: str = "cuda", slice_k: int = 0) -> dict:
    """``|W * dL/dW|`` for every MLP weight, one backward pass over the slice."""
    t0 = time.time()
    model, _ = load_model(model_name, device=device, dtype=torch.float32)
    nl = n_layers(model)
    arr = np.load(DATA / model_name / f"slice_{slice_k:02d}.npy")
    tokens = torch.from_numpy(arr.astype(np.int64)).to(device)
    n_seq = tokens.shape[0]
    mlp_params = []
    for L in range(nl):
        for name in ("c_fc", "c_proj"):
            p = getattr(model.transformer.h[L].mlp, name).weight
            p.requires_grad_(True)
            mlp_params.append(p)
    grads = {}
    chunk = 8 if d_mlp(model) <= 4096 else 4
    with torch.enable_grad():
        for s in range(0, n_seq, chunk):
            idx = tokens[s:s + chunk, :256]
            logits, _, _ = model(idx)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tokens[s:s + chunk, 1:].reshape(-1), reduction="sum")
            loss.backward()
            for L in range(nl):
                for name in ("c_fc", "c_proj"):
                    p = getattr(model.transformer.h[L].mlp, name).weight
                    g = p.grad
                    grads[(L, name)] = (grads[(L, name)] + g) if (L, name) in grads else g.clone()
                    p.grad = None
            del logits, loss
    w = load_weights(model_name)
    W = len(w["layer"])
    out = np.empty(W, dtype=np.float32)
    for L in range(nl):
        for mi, name in enumerate(("c_fc", "c_proj")):
            p = getattr(model.transformer.h[L].mlp, name).weight.detach()
            attr = (p * grads[(L, name)]).abs()
            sel = np.flatnonzero((w["layer"] == L) & (w["matrix"] == mi))
            out[sel] = attr[w["row"][sel], w["col"][sel]].cpu().numpy()
    path = ROOT / "runs" / model_name / "triage.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, out)
    return {"path": str(path), "W": int(W), "elapsed_s": time.time() - t0,
            "finite": bool(np.isfinite(out).all())}


# --------------------------------------------------------------------- bench

def breakdown(model_name: str, device: str = "cuda", reps: int = 3) -> dict:
    """Wall-time breakdown of one layer-0 ablation pass (where the cost goes)."""
    r = Replayer(model_name, device=device)
    r.load_slice(0)
    m, tok = r.model, r.tokens
    blk = m.transformer.h[0]
    ch = r.seq_chunk

    def t(fn, n=reps):
        fn()
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t0) / n * 1000.0

    def _lk(s0: int, s1: int):
        l = lg_chunks[s0 // ch].float() + r._bigram[s0:s1]
        lp = torch.log_softmax(l, dim=-1)
        ce = -lp.gather(-1, tok[s0:s1, 1:].unsqueeze(-1)).squeeze(-1)
        lb = r._logp_base[s0:s1]
        return ce, (lb.exp() * (lb - lp)).sum(-1)

    def attn(b, x):
        return b.forward_attn_block(x, r._pos) if r.cat_pos_emb else b.forward_attn_block(x)

    def mlp(b, x):
        return b.forward_mlp_block(x, r._pos) if r.cat_pos_emb else b.forward_mlp_block(x)

    with torch.no_grad():
        x_in = r._resid_mid[0].to(r.device)
        x_out = mlp(blk, x_in)
        lg_chunks = [m.forward_tail(x_out[s:s + ch], 1, pos_emb_to_cat=r._pos)[0]
                     for s in range(0, r.n_seq, ch)]
        base = r._mlp_base(0, x_in)
        row0, col0 = int(r.weights["row"][0]), int(r.weights["col"][0])
        proj = int(np.flatnonzero(r.weights["matrix"] == 1)[0])
        prow, pcol = int(r.weights["row"][proj]), int(r.weights["col"][proj])
        res = {
            "n_seq": r.n_seq,
            "seq_chunk": ch,
            "full_model_forward_ms": t(
                lambda: [m(tok[s:s + ch, :256]) for s in range(0, r.n_seq, ch)]),
            "ablate_one_weight_L0_cfc_ms": t(lambda: r.ablate(torch.tensor([0]), want_kl=True)),
            "suffix_from_L1_ms": t(
                lambda: [m.forward_tail(x_out[s:s + ch], 1, pos_emb_to_cat=r._pos)
                         for s in range(0, r.n_seq, ch)]),
            "suffix_attention_only_ms": t(
                lambda: [[attn(b, x_out[s:s + ch]) for b in m.transformer.h[1:]]
                         for s in range(0, r.n_seq, ch)]),
            "suffix_mlp_only_ms": t(
                lambda: [[mlp(b, x_out[s:s + ch]) for b in m.transformer.h[1:]]
                         for s in range(0, r.n_seq, ch)]),
            "head_lnf_lmhead_ms": t(lambda: m.lm_head(m.transformer.ln_f(x_out))),
            "mlp_base_L0_ms": t(lambda: [r._mlp_base(0, x_in[s:s + ch])
                                         for s in range(0, r.n_seq, ch)]),
            "mlp_ablated_L0_cfc_ms": t(lambda: [r._mlp_ablated(0, base, 0, row0, col0)
                                                for s in range(0, r.n_seq, ch)]),
            "mlp_ablated_L0_cproj_ms": t(lambda: [r._mlp_ablated(0, base, 1, prow, pcol)
                                                  for s in range(0, r.n_seq, ch)]),
            "logits_softmax_ce_kl_ms": t(lambda: [_lk(s, min(s + ch, r.n_seq))
                                                  for s in range(0, r.n_seq, ch)]),
        }
        del x_in, x_out, lg_chunks, base
    del r
    torch.cuda.empty_cache()
    return res


def bench(model_name: str, device: str = "cuda", k_per_layer: int = 8) -> dict:
    r = Replayer(model_name, device=device)
    r.load_slice(0)
    nl = r.n_layer
    layers = sorted({0, nl // 2, nl - 1})
    rates: dict[int, float] = {}
    for L in layers:
        sel = np.flatnonzero(r.weights["layer"] == L)[:k_per_layer]
        wids = torch.tensor(sel.tolist())
        r.ablate(wids[:1], want_kl=True)          # warmup
        torch.cuda.synchronize()
        t0 = time.time()
        r.ablate(wids, want_kl=True)
        torch.cuda.synchronize()
        dt = time.time() - t0
        rates[L] = len(wids) / dt
        print(f"  bench {model_name} L{L}: {rates[L]:.2f} weights/s")
    # cost model: seconds per weight ~ a*(n_layer-L) + b  (fit through the 3 points)
    ls = np.asarray(layers, dtype=float)
    cps = np.asarray([1.0 / rates[L] for L in layers])
    A = np.stack([nl - ls, np.ones_like(ls)], axis=1)
    coef, *_ = np.linalg.lstsq(A, cps, rcond=None)
    n_w = np.bincount(r.weights["layer"], minlength=nl).astype(float)
    est = float(((nl - np.arange(nl)) * coef[0] + coef[1]) @ n_w)
    W = int(len(r.weights["layer"]))
    # fp16 deviation of the base forward, reported separately (fp16 is NOT used by default)
    fp16 = {}
    try:
        tok = r.tokens
        lg32 = None
        from swa.model import base_logits, per_position_ce
        lg32 = per_position_ce(base_logits(r.model, tok, chunk=r.seq_chunk), tok)
        m16, _ = load_model(model_name, device=device, dtype=torch.float16)
        ce16 = per_position_ce(base_logits(m16, tok, chunk=r.seq_chunk), tok)
        d = (lg32 - ce16).abs().cpu().numpy()
        fp16 = {"max_abs_ce_diff": float(d.max()), "mean_abs_ce_diff": float(d.mean())}
        del m16
    except Exception as exc:  # pragma: no cover - reported, never fatal
        fp16 = {"error": repr(exc)}
    torch.cuda.empty_cache()
    res = {
        "model": model_name,
        "device": str(r.device),
        "dtype": "float32",
        "slice": 0,
        "n_seq": r.n_seq,
        "n_positions": r.n_seq * T,
        "weights_per_sec": {f"L{L}": rates[L] for L in layers},
        "sec_per_weight_model": {"a": float(coef[0]), "b": float(coef[1])},
        "n_weights": W,
        "est_hours_all_weights": est / 3600.0,
        "fp16_base_ce_deviation": fp16,
    }
    del r
    torch.cuda.empty_cache()
    res["breakdown"] = breakdown(model_name, device)
    return res


# --------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--layers", default=None, help="comma separated layer ids")
    ap.add_argument("--matrices", default=None, help="comma separated: fc,proj")
    ap.add_argument("--limit", type=int, default=None, help="max wids per shard")
    ap.add_argument("--wids-file", type=Path, default=None, help=".npy of explicit wids")
    ap.add_argument("--force", action="store_true", help="redo even complete shards")
    ap.add_argument("--triage", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--all", action="store_true", help="run triage and bench too")
    args = ap.parse_args(argv)

    tri = None
    if args.triage or args.all:
        tri = triage(args.model, args.device)
        print("triage:", json.dumps(tri))
    if args.bench or args.all:
        b = bench(args.model, args.device)
        if tri is not None:
            b["triage_seconds"] = tri["elapsed_s"]
            b["triage_weights"] = tri["W"]
        p = ROOT / "runs" / args.model / "bench.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        prev = json.loads(p.read_text()) if p.exists() else {}
        prev.update(b)
        p.write_text(json.dumps(prev, indent=1))
        print("bench:", json.dumps(b))
        if not (args.layers or args.limit or args.wids_file):
            return

    w = load_weights(args.model)
    smap = shard_map(w)
    layers = None
    if args.layers:
        layers = {int(x) for x in args.layers.split(",") if x.strip() != ""}
    mats = None
    if args.matrices:
        mats = {0 if x.strip() == "fc" else 1 for x in args.matrices.split(",") if x.strip()}

    if args.wids_file is not None:
        wids = np.load(args.wids_file).astype(np.int64).reshape(-1).tolist()
        by_shard: dict[str, list[int]] = {}
        home = {}
        for shard, wl in smap.items():
            for x in wl:
                home[x] = shard
        for x in wids:
            by_shard.setdefault(home[x], []).append(x)
        todo = by_shard
    else:
        todo = {}
        for shard, wl in smap.items():
            L = int(shard[1:3])
            M = 0 if shard[4:6] == "fc" else 1
            if layers is not None and L not in layers:
                continue
            if mats is not None and M not in mats:
                continue
            if args.limit is not None:
                wl = wl[: args.limit]
            todo[shard] = wl

    r = Replayer(args.model, device=args.device)
    r.load_slice(0)
    total = sum(len(v) for v in todo.values())
    print(f"{args.model}: {len(todo)} shards, {total} weights")
    for i, (shard, wl) in enumerate(sorted(todo.items())):
        m = run_shard(r, args.model, shard, wl, force=args.force,
                      partial=args.wids_file is not None)
        tag = "skip" if m.get("skipped") else f"{m['elapsed_s']:.1f}s {m['weights_per_sec']:.2f} w/s"
        print(f"[{i + 1}/{len(todo)}] {shard} n={m.get('n', len(wl))} {tag}", flush=True)


if __name__ == "__main__":
    main()
