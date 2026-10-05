"""G2 oracle gate (CONTRACT.md "Validation gates").

* baseline per-position CE of csp_yolo1 on slice 00 (paper's Gao et al. number: 2.34);
* ``kl_max`` of L0 ``c_fc[1863, 464]`` next to Weightpedia's 26.766;
* Spearman correlation of our ``kl_max`` against Weightpedia ``max_kl`` for the weights of
  ranks 1..200 (``deep_profile_<rank>.json``; ``mlp.W_in [i,n] -> c_fc[n,i]``,
  ``mlp.W_out [n,o] -> c_proj[o,n]``).

Run: ``PYTHONPATH=engine .venv/bin/python scripts/check_g2.py``
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import torch

from swa.model import ROOT
from swa.replay import Replayer

WP_URL = ("https://weightpedia.org/individual-parameters-in-sparse-transformers/"
          "data/deep_profile_{rank}.json")
CACHE = ROOT / "runs" / "csp_yolo1" / "weightpedia"


def fetch(rank: int, cache: Path = CACHE) -> dict:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"deep_profile_{rank}.json"
    if not p.exists() or p.stat().st_size == 0:
        with urllib.request.urlopen(WP_URL.format(rank=rank), timeout=60) as fh:
            p.write_bytes(fh.read())
    return json.loads(p.read_text())


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))


def wp_to_rc(rec: dict) -> tuple[int, int, int] | None:
    """Weightpedia record -> (matrix, row, col) in our orientation, or None if not MLP.

    ``mlp.W_in  [i, n]`` -> ``c_fc[n, i]``  -> row = indices[1], col = indices[0]
    ``mlp.W_out [n, o]`` -> ``c_proj[o, n]`` -> row = indices[1], col = indices[0]
    """
    i0, i1 = int(rec["indices"][0]), int(rec["indices"][1])
    if rec["component"] == "mlp.W_in":
        return 0, i1, i0
    if rec["component"] == "mlp.W_out":
        return 1, i1, i0
    return None


def check(device: str = "cuda", max_rank: int = 200) -> dict:
    r = Replayer("csp_yolo1", device=device)
    r.load_slice(0)
    ce0 = r.ce_base()
    n_pos = r.n_seq * r.T
    base_mean_ce = float(ce0.mean())
    w = r.weights
    index = r.wid_index()

    recs = []
    skipped_nonmlp = []
    for rank in range(1, max_rank + 1):
        d = fetch(rank)
        rc = wp_to_rc(d)
        if rc is None:
            skipped_nonmlp.append({"rank": rank, "component": d["component"]})
            continue
        M, row, col = rc
        L = int(d["layer"])
        wid = index.get((L, M, row, col))
        recs.append({"rank": rank, "layer": L, "matrix": M, "row": row, "col": col,
                     "wp_max_kl": float(d["max_kl"]), "wp_mean_kl": float(d["mean_kl"]),
                     "wid": wid, "component": d["component"]})
    missing = [x for x in recs if x["wid"] is None]

    by_layer: dict[int, list[int]] = {}
    for x in recs:
        if x["wid"] is not None:
            by_layer.setdefault(x["layer"], []).append(x["wid"])
    ours: dict[int, float] = {}
    for L, wids in sorted(by_layer.items()):
        ce, kl = r.ablate(torch.tensor(wids), want_kl=True)
        km = kl.amax(dim=(1, 2)).cpu().numpy()
        for wid, v in zip(wids, km):
            ours[wid] = float(v)
        print(f"  layer {L}: {len(wids)} weightpedia weights, kl_max range "
              f"[{km.min():.4f}, {km.max():.4f}]", flush=True)

    for x in recs:
        x["our_kl_max"] = ours.get(x["wid"]) if x["wid"] is not None else None
    paired = [x for x in recs if x["our_kl_max"] is not None]
    rho = spearman(np.array([x["wp_max_kl"] for x in paired]),
                   np.array([x["our_kl_max"] for x in paired])) if len(paired) > 2 else None

    tgt = index.get((0, 0, 1863, 464))
    tgt_kl = None
    if tgt is not None:
        _, kl = r.ablate(torch.tensor([tgt]), want_kl=True)
        tgt_kl = float(kl.max())
    return {
        "model": "csp_yolo1",
        "slice": 0,
        "n_positions": n_pos,
        "baseline_mean_ce": base_mean_ce,
        "paper_baseline_ce": 2.34,
        "weightpedia_positions": 638206,
        "L0_c_fc_1863_464": {"wid": tgt, "our_kl_max": tgt_kl, "weightpedia_max_kl": 26.766170501708984},
        "n_weightpedia_ranks": max_rank,
        "n_mlp_ranks": len(recs),
        "n_skipped_non_mlp": len(skipped_nonmlp),
        "skipped_non_mlp": skipped_nonmlp,
        "n_matched": len(paired),
        "n_missing_from_our_model": len(missing),
        "missing": missing[:10],
        "spearman_kl_max": rho,
        "pairs": paired,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rank", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "csp_yolo1" / "g2.json")
    args = ap.parse_args(argv)
    res = check(args.device, args.max_rank)
    args.out.write_text(json.dumps(res, indent=1))
    print(f"baseline mean CE slice00 = {res['baseline_mean_ce']:.4f} (paper 2.34)")
    t = res["L0_c_fc_1863_464"]
    print(f"L0 c_fc[1863,464] kl_max = {t['our_kl_max']} (weightpedia 26.766)")
    print(f"spearman(kl_max, weightpedia max_kl) over {res['n_matched']} ranks = "
          f"{res['spearman_kl_max']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
