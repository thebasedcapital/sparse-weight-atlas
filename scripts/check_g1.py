"""G1 engine exactness gate (CONTRACT.md "Validation gates").

For >=6 weights per model (both matrices, first/middle/last layer) check that the replay CE
equals a plain full forward with that weight set to 0.0 in a model copy (fp32, max abs
per-position diff < 1e-4), that an all-true restore mask reproduces the base CE, and that an
all-false restore mask reproduces the ablated CE.

Run: ``PYTHONPATH=engine .venv/bin/python scripts/check_g1.py [--model NAME]``
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from swa.replay import Replayer, reference_full_forward
from swa.weights import load_weights

TOL = 1e-4


def pick_wids(w: dict, per_group: int = 2) -> list[int]:
    nl = int(w["layer"].max()) + 1
    layers = sorted({0, nl // 2, nl - 1})
    out: list[int] = []
    for L in layers:
        for M in (0, 1):
            sel = np.flatnonzero((w["layer"] == L) & (w["matrix"] == M))
            out += sel[:per_group].tolist()
    return out


def check(model_name: str, device: str = "cuda") -> dict:
    w = load_weights(model_name)
    r = Replayer(model_name, device=device, dtype=torch.float32)
    r.load_slice(0)
    wids = pick_wids(w)

    # the npz value must be the model's parameter value (exactness of "set to 0.0")
    for wid in wids:
        L, M, row, col = (int(w["layer"][wid]), int(w["matrix"][wid]),
                          int(w["row"][wid]), int(w["col"][wid]))
        name = "c_fc" if M == 0 else "c_proj"
        pv = float(r.model.transformer.h[L].mlp._modules[name].weight[row, col])
        assert pv == float(w["value"][wid]), f"wid {wid}: npz value != model param"

    ref = np.stack([
        reference_full_forward(r.model, r.tokens, int(w["layer"][wid]), int(w["matrix"][wid]),
                               int(w["row"][wid]), int(w["col"][wid]),
                               chunk=r.seq_chunk).cpu().numpy()
        for wid in wids])

    # ablate() requires all wids in one layer -> group by layer, keep the original order
    order = list(range(len(wids)))
    order.sort(key=lambda i: int(w["layer"][wids[i]]))
    abl = np.empty((len(wids), r.n_seq, r.T), dtype=np.float32)
    res = np.empty_like(abl)
    non = np.empty_like(abl)
    for L in sorted({int(w["layer"][x]) for x in wids}):
        grp = [i for i in order if int(w["layer"][wids[i]]) == L]
        gi = torch.tensor([wids[i] for i in grp])
        ones = torch.ones((len(grp), r.n_seq, r.T), dtype=torch.bool)
        zeros = torch.zeros((len(grp), r.n_seq, r.T), dtype=torch.bool)
        ce_abl, _ = r.ablate(gi)
        ce_res, _ = r.ablate(gi, restore=ones)
        ce_none, _ = r.ablate(gi, restore=zeros)
        for k, i in enumerate(grp):
            abl[i] = ce_abl[k].cpu().numpy()
            res[i] = ce_res[k].cpu().numpy()
            non[i] = ce_none[k].cpu().numpy()
    base = r.ce_base().cpu().numpy()

    d_ref = np.abs(abl - ref)
    d_rest = np.abs(res - base[None])
    d_rest_ref = np.abs(res - ref)
    d_zero = np.abs(non - abl)

    rows = []
    for i, wid in enumerate(wids):
        rows.append({
            "wid": int(wid), "layer": int(w["layer"][wid]), "matrix": int(w["matrix"][wid]),
            "row": int(w["row"][wid]), "col": int(w["col"][wid]),
            "value": float(w["value"][wid]),
            "max_abs_diff_vs_zeroed_full_forward": float(d_ref[i].max()),
            "max_abs_diff_restore_true_vs_base": float(d_rest[i].max()),
            "max_abs_diff_restore_false_vs_ablated": float(d_zero[i].max()),
        })
    result = {
        "model": model_name,
        "n_weights": len(wids),
        "max_abs_diff_vs_zeroed_full_forward": float(d_ref.max()),
        "max_abs_diff_restore_true_vs_base": float(d_rest.max()),
        "max_abs_diff_restore_false_vs_ablated": float(d_zero.max()),
        "max_abs_diff_restore_true_vs_zeroed_full_forward": float(d_rest_ref.max()),
        "tolerance": TOL,
        "pass": bool(d_ref.max() < TOL and d_rest.max() < TOL and d_zero.max() == 0.0),
        "weights": rows,
    }
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None, help="default: both models")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    models = [args.model] if args.model else ["csp_yolo1", "csp_yolo2"]
    allres = {}
    for m in models:
        res = check(m, args.device)
        allres[m] = res
        print(f"== G1 {m}: n={res['n_weights']} "
              f"max|replay-zeroed_full_fwd|={res['max_abs_diff_vs_zeroed_full_forward']:.3e} "
              f"max|restore_true-base|={res['max_abs_diff_restore_true_vs_base']:.3e} "
              f"max|restore_false-ablated|={res['max_abs_diff_restore_false_vs_ablated']:.3e} "
              f"-> {'PASS' if res['pass'] else 'FAIL'}")
        for row in res["weights"]:
            print(f"   wid={row['wid']:6d} L{row['layer']:02d} "
                  f"{'fc' if row['matrix'] == 0 else 'proj'} "
                  f"[{row['row']},{row['col']}] diff={row['max_abs_diff_vs_zeroed_full_forward']:.3e}")
    print(json.dumps({m: {"pass": r["pass"],
                          "max_abs_diff_vs_zeroed_full_forward":
                              r["max_abs_diff_vs_zeroed_full_forward"]}
                      for m, r in allres.items()}))


if __name__ == "__main__":
    main()
