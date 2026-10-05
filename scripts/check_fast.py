#!/usr/bin/env python3
"""Measure and validate opt-in fp32 profiling acceleration without touching live outputs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from swa.model import ROOT
from swa.replay import Replayer
from swa.fastreplay import FastReplayer


def memory_limit():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(2.5 * 1024**3 / total)


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start, value


def measure(r, out):
    L = 0
    x = r._resid_mid[L][:r.seq_chunk].to(r.device)
    base = r._mlp_base(L, x)
    wid = 7509
    w = r.weights
    x = x + r._mlp_ablated(L, base, int(w['matrix'][wid]), int(w['row'][wid]), int(w['col'][wid]))
    del base
    originals = []
    for block in r.model.transformer.h[1:]:
        for name, label in [('forward_attn_block', 'suffix_attention'), ('forward_mlp_block', 'suffix_mlp')]:
            original = getattr(block, name)
            def wrapped(*args, original=original, label=label, **kwargs):
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)
            originals.append((block, name, original))
            setattr(block, name, wrapped)
    fn = lambda: r.model.forward_tail(x, n=1, pos_emb_to_cat=r._pos)[0]
    try:
        fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            elapsed, value = timed(fn)
            del value
        trace = out.with_name('fast_trace.json')
        prof.export_chrome_trace(str(trace))
        events = prof.key_averages()
        scopes = {e.key: e.device_time_total / 1000 for e in events if e.key in ('suffix_attention', 'suffix_mlp')}
        launch_us = sum(e.self_cpu_time_total for e in events if 'cudaLaunch' in e.key)
        kernels = [e for e in json.loads(trace.read_text())['traceEvents'] if e.get('cat') == 'kernel']
        result = {'suffix_chunk_wall_ms': elapsed * 1000, 'seq_chunk': r.seq_chunk,
                  'kernel_count': len(kernels), 'kernel_ms_sum': sum(e['dur'] for e in kernels) / 1000,
                  'launch_cpu_ms_sum': launch_us / 1000, 'launch_cpu_over_wall': launch_us / (elapsed * 1e6),
                  'scopes_cuda_ms': scopes, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'trace': str(trace), 'top_ops': [{'name': e.key, 'calls': e.count, 'cuda_ms': e.self_device_time_total / 1000} for e in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:15]]}
        print(json.dumps(result, indent=2), flush=True)
        out.write_text(json.dumps({'measurement': result}, indent=2))
    finally:
        for block, name, original in originals:
            setattr(block, name, original)


def select_weights(w):
    result = []
    for layer in (0, 5, 11):
        for matrix in (0, 1):
            sel = np.flatnonzero((w['layer'] == layer) & (w['matrix'] == matrix))
            result.extend(sel[np.linspace(0, len(sel) - 1, 5, dtype=int)].tolist())
    return sorted(set(result + [7509, 7511, 7514]))


def experiments(r, technique, compile_mode, batch=1, reps=3):
    fast = FastReplayer.from_reference(r, technique, compile_mode)
    rows = []
    for layer in (0, 5, 11):
        selected = []
        for matrix in (0, 1):
            pool = np.flatnonzero((r.weights['layer'] == layer) & (r.weights['matrix'] == matrix))
            selected.extend(pool[:max(1, batch // 2)].tolist())
        selected = selected[:batch]
        eager_fn = lambda: r.ablate(selected, want_kl=True)
        fast_fn = lambda: fast.ablate(selected, want_kl=True)
        eager_fn()
        warm_s, _ = timed(fast_fn)
        ea, fa = [], []
        max_ce = max_kl = 0.0
        for rep in range(reps):
            if rep % 2:
                ft, fv = timed(fast_fn)
                et, ev = timed(eager_fn)
            else:
                et, ev = timed(eager_fn)
                ft, fv = timed(fast_fn)
            ea.append(et)
            fa.append(ft)
            max_ce = max(max_ce, float((fv[0] - ev[0]).abs().max()))
            max_kl = max(max_kl, float((fv[1] - ev[1]).abs().max()))
            del fv, ev
        row = {'layer': layer, 'wids': selected, 'eager_seconds': ea, 'fast_seconds': fa,
               'eager_wps': len(selected) / np.median(ea), 'fast_wps': len(selected) / np.median(fa),
               'speedup': float(np.median(ea) / np.median(fa)), 'warm_seconds': warm_s,
               'max_ce_diff': max_ce, 'max_kl_diff': max_kl}
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {'technique': technique, 'compile_mode': compile_mode, 'rows': rows,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measure', action='store_true')
    ap.add_argument('--experiments', choices=['graph', 'compile', 'sparse', 'cache'])
    ap.add_argument('--compile-mode', default='max-autotune-no-cudagraphs')
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/csp_yolo1/fast_bench.json')
    args = ap.parse_args()
    memory_limit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        r = Replayer(seq_chunk=8)
        r.load_slice(0)
        if args.measure:
            measure(r, args.out)
        if args.experiments:
            result = experiments(r, args.experiments, args.compile_mode, args.batch, args.reps)
            previous = json.loads(args.out.read_text()) if args.out.exists() else {}
            previous.setdefault('experiments', []).append(result)
            args.out.write_text(json.dumps(previous, indent=2))


if __name__ == '__main__':
    main()
