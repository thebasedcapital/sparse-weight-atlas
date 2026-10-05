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
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        result = {'suffix_chunk_wall_ms': elapsed * 1000, 'seq_chunk': r.seq_chunk,
                  'kernel_count': len(kernels), 'kernel_ms_sum': sum(e.device_time_total for e in kernels) / 1000,
                  'launch_cpu_ms_sum': launch_us / 1000, 'launch_cpu_over_wall': launch_us / (elapsed * 1e6),
                  'scopes_cuda_ms': scopes, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                  'trace': str(trace), 'top_ops': [{'name': e.key, 'calls': e.count, 'cuda_ms': e.self_device_time_total / 1000} for e in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:15]]}
        print(json.dumps(result, indent=2), flush=True)
        out.write_text(json.dumps({'measurement': result}, indent=2))
    finally:
        for block, name, original in originals:
            setattr(block, name, original)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measure', action='store_true')
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/csp_yolo1/fast_bench.json')
    args = ap.parse_args()
    memory_limit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        r = Replayer()
        r.load_slice(0)
        if args.measure:
            measure(r, args.out)


if __name__ == '__main__':
    main()
