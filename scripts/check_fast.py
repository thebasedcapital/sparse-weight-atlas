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


class BoundedReference(Replayer):
    """Identical eager arithmetic, dropping only target tensors unused in full ablation."""
    def _mlp_ablated(self, L, base, mat, row, col):
        base.pop('out', None)
        base.pop('a' if mat == 0 else 'act', None)
        return super()._mlp_ablated(L, base, mat, row, col)


def bounded_tail(model):
    """Same blocks and arithmetic as forward_tail, without the unused hs tensor list."""
    def forward(x, n, targets=None, additional_logits_bias=None,
                include_resid_mid=False, pos_emb_to_cat=None):
        if targets is not None or include_resid_mid:
            raise ValueError('measurement tail supports inference without split blocks only')
        for block in model.transformer.h[n:]:
            x = block(x, pos_emb_to_cat) if model.config.cat_pos_emb else block(x)
        logits = (model.lm_head(model.transformer.ln_f(x)) + model.final_logits_bias
                  + (additional_logits_bias if additional_logits_bias is not None else 0))
        return logits, torch.zeros(1, device=x.device), []
    return forward


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
        previous = json.loads(out.read_text()) if out.exists() else {}
        previous['measurement'] = result
        out.write_text(json.dumps(previous, indent=2))
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


def benchmark(r, batch=4, reps=2):
    fast = FastReplayer.from_reference(r)
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
    return {'technique': 'sparse', 'seq_chunk': r.seq_chunk, 'rows': rows,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}


def extrapolate(r, rows):
    layers = np.array([row['layer'] for row in rows])
    result = {}
    for mode in ('eager', 'fast'):
        seconds = np.array([1 / row[f'{mode}_wps'] for row in rows])
        counts = np.bincount(r.weights['layer'], minlength=r.n_layer)
        result[f'{mode}_hours'] = float(np.dot(counts, np.interp(np.arange(r.n_layer), layers, seconds)) / 3600)
    result['speedup'] = result['eager_hours'] / result['fast_hours']
    result['weights'] = len(r.weights['layer'])
    result['method'] = 'piecewise-linear seconds/weight at layers 0/5/11, measured under live GPU contention'
    return result


def gate(r):
    wids = select_weights(r.weights)
    if len(wids) < 32 or not {7509, 7511, 7514}.issubset(wids):
        raise ValueError('gate requires at least 32 weights including all three digit weights')
    fast = FastReplayer.from_reference(r)
    alt = BoundedReference.__new__(BoundedReference)
    alt.__dict__.update(r.__dict__)
    alt.seq_chunk = 8 if r.seq_chunk != 8 else 4
    FastReplayer.load_slice(alt, 0)
    noise_chunk = alt.seq_chunk
    alternate = {}
    for wid in wids:
        a = alt.ablate([wid], want_kl=True)
        alternate[wid] = ((a[0][0] - alt.ce_base()).cpu().numpy(), a[1][0].cpu().numpy())
        del a
    del alt
    torch.cuda.empty_cache()
    floor = {'max_dce_position': 0.0, 'max_kl_position': 0.0, 'max_mean_dce': 0.0}
    rows = []
    for index, wid in enumerate(wids):
        # Alternate A/B ordering; synchronize each arm so live contention treats both alike.
        if index % 2:
            _, f = timed(lambda: fast.ablate([wid], want_kl=True))
            _, e = timed(lambda: r.ablate([wid], want_kl=True))
        else:
            _, e = timed(lambda: r.ablate([wid], want_kl=True))
            _, f = timed(lambda: fast.ablate([wid], want_kl=True))
        ad, ak = alternate[wid]
        ed = (e[0][0] - r.ce_base()).cpu().numpy()
        fd = (f[0][0] - r.ce_base()).cpu().numpy()
        ek, fk = (v[1][0].cpu().numpy() for v in (e, f))
        floor['max_dce_position'] = max(floor['max_dce_position'], float(np.abs(ed - ad).max()))
        floor['max_kl_position'] = max(floor['max_kl_position'], float(np.abs(ek - ak).max()))
        em, fm, am = (float(v.mean(dtype=np.float64)) for v in (ed, fd, ad))
        floor['max_mean_dce'] = max(floor['max_mean_dce'], abs(em - am))
        rows.append({'wid': wid, 'layer': int(r.weights['layer'][wid]),
                     'matrix': int(r.weights['matrix'][wid]), 'max_dce_error': float(np.abs(ed - fd).max()),
                     'max_kl_error': float(np.abs(ek - fk).max()), 'mean_dce_eager': em,
                     'mean_dce_fast': fm, 'mean_dce_error': abs(em - fm),
                     'mean_tolerance': max(1e-7, 0.01 * abs(em))})
        del e, f
    for row in rows:
        row['pass'] = (row['max_dce_error'] <= floor['max_dce_position']
                       and row['max_kl_error'] <= floor['max_kl_position']
                       and row['mean_dce_error'] <= row['mean_tolerance'])
    result = {'pass': all(row['pass'] for row in rows), 'technique': 'sparse',
              'n_weights': len(wids), 'seq_chunk': r.seq_chunk, 'noise_seq_chunk': noise_chunk,
              'floor': floor, 'rows': rows, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    print('measured eager numerical floor:', json.dumps(floor), flush=True)
    print('PASS' if result['pass'] else 'FAIL', f"{sum(row['pass'] for row in rows)}/{len(rows)} weights", flush=True)
    for row in rows:
        if not row['pass']:
            print('FAILED:', json.dumps(row), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--measure', action='store_true')
    ap.add_argument('--bench', action='store_true', help='interleaved A/B throughput benchmark')
    ap.add_argument('--gate', action='store_true', help='numerical-floor gate (default without other modes)')
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--seq-chunk', type=int, default=64)
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/csp_yolo1/fast_bench.json')
    args = ap.parse_args()
    if args.batch < 1 or args.reps < 1 or args.seq_chunk < 1:
        ap.error('batch, reps and seq-chunk must be positive')
    if not (args.measure or args.bench or args.gate):
        args.gate = True
    memory_limit()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        loader = FastReplayer(seq_chunk=args.seq_chunk)
        loader.load_slice(0)
        r = BoundedReference.__new__(BoundedReference)
        r.__dict__.update(loader.__dict__)
        r.model.forward_tail = bounded_tail(r.model)
        stored_base = np.load(ROOT / 'runs/csp_yolo1/base/ce0_slice00.npy')
        baseline_diff = float(np.abs(r.ce_base().cpu().numpy() - stored_base).max())
        print('cached baseline max difference from saved eager64:', baseline_diff, flush=True)
        if args.seq_chunk == 64 and baseline_diff != 0:
            raise RuntimeError('memory-bounded eager prefix differs from saved production baseline')
        if args.measure:
            measure(r, args.out)
        if args.bench:
            result = benchmark(r, args.batch, args.reps)
            result['extrapolation'] = extrapolate(r, result['rows'])
            previous = json.loads(args.out.read_text()) if args.out.exists() else {}
            previous.setdefault('experiments', []).append(result)
            args.out.write_text(json.dumps(previous, indent=2))
        if args.gate:
            result = gate(r)
            previous = json.loads(args.out.read_text()) if args.out.exists() else {}
            previous.setdefault('gates', []).append(result)
            args.out.write_text(json.dumps(previous, indent=2))
            if not result['pass']:
                raise SystemExit(1)


if __name__ == '__main__':
    main()
