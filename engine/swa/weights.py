"""Build ``runs/<model>/weights.npz`` and ``runs/<model>/base/ce0_slice{kk}.npy``.

Weight ordering (CONTRACT.md): layer ascending, then matrix (0 = ``c_fc``, 1 = ``c_proj``),
then ``torch.nonzero`` row-major order.  Index into these arrays is the **wid**.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from swa.model import (DATA, ROOT, base_logits, d_mlp, default_seq_chunk, load_model, n_layers,
                       per_position_ce)

MATRICES = ("c_fc", "c_proj")
N_SLICES = 12
T = 256


def build_weights(model_name: str, device: str = "cuda") -> Path:
    model, _ = load_model(model_name, device=device, dtype=torch.float32)
    layers, matrices, rows, cols, values = [], [], [], [], []
    for L in range(n_layers(model)):
        for mi, name in enumerate(MATRICES):
            W = getattr(model.transformer.h[L].mlp, name).weight.detach()
            nz = torch.nonzero(W, as_tuple=False).cpu()
            layers.append(torch.full((nz.shape[0],), L, dtype=torch.int16))
            matrices.append(torch.full((nz.shape[0],), mi, dtype=torch.int8))
            rows.append(nz[:, 0].to(torch.int32))
            cols.append(nz[:, 1].to(torch.int32))
            values.append(W[nz[:, 0], nz[:, 1]].cpu().to(torch.float32))
    out = ROOT / "runs" / model_name / "weights.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        layer=torch.cat(layers).numpy(),
        matrix=torch.cat(matrices).numpy(),
        row=torch.cat(rows).numpy(),
        col=torch.cat(cols).numpy(),
        value=torch.cat(values).numpy(),
    )
    return out


def build_base(model_name: str, device: str = "cuda", n_slices: int = N_SLICES,
               chunk: int | None = None) -> Path:
    model, _ = load_model(model_name, device=device, dtype=torch.float32)
    if chunk is None:
        chunk = default_seq_chunk(model)
    out_dir = ROOT / "runs" / model_name / "base"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(n_slices):
        path = DATA / model_name / f"slice_{k:02d}.npy"
        arr = np.load(path)
        tokens = torch.from_numpy(arr.astype(np.int64)).to(device)
        logits = base_logits(model, tokens, chunk=chunk)
        ce = per_position_ce(logits, tokens)
        np.save(out_dir / f"ce0_slice{k:02d}.npy", ce.cpu().numpy().astype(np.float32))
        del logits, ce
        torch.cuda.empty_cache()
    return out_dir


def load_weights(model_name: str) -> dict[str, np.ndarray]:
    return dict(np.load(ROOT / "runs" / model_name / "weights.npz"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-base", action="store_true")
    ap.add_argument("--n-slices", type=int, default=N_SLICES)
    args = ap.parse_args(argv)
    p = build_weights(args.model, device=args.device)
    w = load_weights(args.model)
    print(f"wrote {p}: W={len(w['layer'])}")
    for L in sorted(set(w["layer"].tolist())):
        m = w["layer"] == L
        print(f"  L{L:02d}: fc={int((m & (w['matrix'] == 0)).sum())} "
              f"proj={int((m & (w['matrix'] == 1)).sum())}")
    if not args.skip_base:
        d = build_base(args.model, device=args.device, n_slices=args.n_slices)
        print(f"wrote {d}/ce0_slice{{00..{args.n_slices - 1:02d}}}.npy")


if __name__ == "__main__":
    main()
