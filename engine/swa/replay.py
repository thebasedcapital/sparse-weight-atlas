"""Exact weight-ablation replay.

``Replayer`` implements the API in CONTRACT.md / the build contract:

    r = Replayer(model_name, device, dtype)
    r.load_slice(k); r.n_seq; r.T; r.tokens
    r.ce_base() -> float32 [n_seq, 256]
    r.ablate(wids, restore=None, want_kl=False) -> (ce [B,n_seq,256], kl or None)

Exactness strategy (validated by ``scripts/check_g1.py``)
--------------------------------------------------------
Ablating ``(layer, matrix, row=o, col=j)`` is done by *actually zeroing that single entry in
a copy of the layer's MLP weight matrix* and recomputing that MLP, then running the remaining
blocks.  That makes the replay reproduce the reference computation kernel-for-kernel:

* ``nn.Linear.forward`` is ``F.linear(input, weight, bias)``; recomputing ``c_fc`` /
  ``c_proj`` with a one-entry-modified copy is bitwise the same arithmetic as a full forward
  of a model whose weight was set to 0.0.
* The residual entering layer ``L``'s MLP (``resid_mid``) is cached per slice from the *same*
  forward that produces the baseline logits, so the prefix is bitwise identical.
* Sequences are processed in the same chunk size as the baseline forward (``seq_chunk``), so
  every matmul/SDPA call sees the same batch shape.  This matters: on this model a change of
  batch shape alone perturbs per-position CE by up to ~3e-4, far above the 1e-4 gate, and the
  error of a *delta* implementation (~1e-7 in the MLP output) is amplified ~2000x by the
  remaining blocks.

``restore[b, s, t]`` selects, per scored position, between the ablated MLP output and the
original one (``torch.where``), i.e. the weight's exact contribution is restored exactly
where the mask is True.  ``restore=None`` is a full ablation; all-True reproduces the base
CE bitwise.

``want_kl`` returns ``KL(P_orig || P_abl)`` in nats per position.
"""

from __future__ import annotations

import numpy as np
import torch

try:  # reduce fragmentation on this 8 GiB card; harmless if unsupported
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
except Exception:  # pragma: no cover
    pass

from circuit_sparsity.hook_utils import hook_recorder
from swa.model import (DATA, ROOT, base_logits, d_mlp, d_model, default_seq_chunk, load_model,
                       n_layers, per_position_ce)
from swa.weights import load_weights

T = 256


class Replayer:
    def __init__(self, model_name: str = "csp_yolo1", device: str = "cuda",
                 dtype: torch.dtype = torch.float32, seq_chunk: int | None = None):
        self.model_name = model_name
        self.device = torch.device(device)
        self.dtype = dtype
        self.model, self.cfg = load_model(model_name, self.device, dtype)
        self.weights: dict[str, np.ndarray] = load_weights(model_name)
        self.n_layer = n_layers(self.model)
        self.d_mlp = d_mlp(self.model)
        self.d_model = d_model(self.model)
        self.cat_pos_emb = bool(self.model.config.cat_pos_emb)
        self.T = T
        self.n_seq = 0
        self.slice = None
        self.seq_chunk = int(seq_chunk) if seq_chunk else default_seq_chunk(self.model)
        self.tokens: torch.Tensor | None = None
        self._resid_mid: list[torch.Tensor] | None = None   # CPU tensors
        self._ce_base: torch.Tensor | None = None
        self._logp_base: torch.Tensor | None = None
        self._bigram: torch.Tensor | None = None
        self._pos: torch.Tensor | None = None

    # ---------------------------------------------------------------- slice load

    def load_slice(self, k: int) -> None:
        """Load ``data/<model>/slice_{k:02d}.npy`` and cache everything replay needs."""
        path = DATA / self.model_name / f"slice_{k:02d}.npy"
        arr = np.load(path)
        if arr.shape[1] != T + 1:
            raise ValueError(f"{path}: expected [n_seq,{T + 1}], got {arr.shape}")
        self.slice = k
        self.n_seq = int(arr.shape[0])
        self.tokens = torch.from_numpy(arr.astype(np.int64)).to(self.device)

        m = self.model
        resid: dict[int, list[torch.Tensor]] = {L: [] for L in range(self.n_layer)}
        logits_chunks = []
        with torch.no_grad():
            for s in range(0, self.n_seq, self.seq_chunk):
                idx = self.tokens[s:s + self.seq_chunk, :256]
                with hook_recorder(r"^\d+\.resid_mid$") as h:
                    logits, _, _ = m(idx)
                    for key, val in h.items():
                        resid[int(key.split(".")[0])].append(val.detach().to("cpu"))
                logits_chunks.append(logits.detach().float())
                del logits
        self._resid_mid = [torch.cat(resid[L], dim=0).contiguous() for L in range(self.n_layer)]
        logits_all = torch.cat(logits_chunks, dim=0)
        self._ce_base = per_position_ce(logits_all, self.tokens)
        self._logp_base = torch.log_softmax(logits_all, dim=-1)
        del logits_all, logits_chunks
        torch.cuda.empty_cache()

        if self.model.config.enable_bigram_table:
            with torch.no_grad():
                self._bigram = torch.nn.functional.embedding(
                    self.tokens[:, :256], self.model.bigram_table).to(self.dtype)
        else:
            self._bigram = None
        if self.cat_pos_emb:
            self._pos = self.model.transformer.wpe.weight[:T].unsqueeze(0).to(self.dtype)
        else:
            self._pos = None

    def _require_slice(self) -> None:
        if self._resid_mid is None:
            raise RuntimeError("call load_slice(k) first")

    # ---------------------------------------------------------------- public API

    def ce_base(self) -> torch.Tensor:
        self._require_slice()
        return self._ce_base

    # ---------------------------------------------------------------- internals

    def _mlp_base(self, L: int, x_in: torch.Tensor) -> dict:
        """Unmodified layer-L MLP tensors for one sequence chunk: c_fc input, c_proj input,
        and the MLP output."""
        blk = self.model.transformer.h[L]
        cfg = blk.config
        h = blk.ln_2(x_in)
        if self.cat_pos_emb:
            p = blk.ln_p2(self._pos).expand(h.shape[0], -1, -1)
            act = cfg.maybe_activation_sparsity(torch.cat([h, p], dim=-1), "mlp_in")
        else:
            act = h
        z = torch.nn.functional.linear(act, blk.mlp.c_fc.weight, blk.mlp.c_fc.bias)
        a = cfg.maybe_activation_sparsity(blk.mlp.act_fn(z), "mlp_neuron")
        del z
        out_raw = torch.nn.functional.linear(a, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias)
        out = cfg.maybe_activation_sparsity(out_raw, "mlp_out")
        return {"x_in": x_in, "act": act, "a": a, "out": out}

    def _mlp_ablated(self, L: int, base: dict, mat: int, row: int, col: int) -> torch.Tensor:
        """Layer-L MLP output with exactly ``W[row, col]`` set to 0.0 (same arithmetic as a
        full forward of a model copy whose entry was zeroed)."""
        blk = self.model.transformer.h[L]
        cfg = blk.config
        if mat == 0:
            W2 = blk.mlp.c_fc.weight.clone()
            W2[row, col] = 0.0
            z2 = torch.nn.functional.linear(base["act"], W2, blk.mlp.c_fc.bias)
            del W2
            a2 = cfg.maybe_activation_sparsity(blk.mlp.act_fn(z2), "mlp_neuron")
            del z2
            ob = torch.nn.functional.linear(a2, blk.mlp.c_proj.weight, blk.mlp.c_proj.bias)
            del a2
            return cfg.maybe_activation_sparsity(ob, "mlp_out")
        W2 = blk.mlp.c_proj.weight.clone()
        W2[row, col] = 0.0
        oraw = torch.nn.functional.linear(base["a"], W2, blk.mlp.c_proj.bias)
        del W2
        return cfg.maybe_activation_sparsity(oraw, "mlp_out")

    def ablate(self, wids: torch.Tensor, restore: torch.Tensor | None = None,
               want_kl: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._require_slice()
        wids = torch.as_tensor(wids, dtype=torch.long, device="cpu").reshape(-1)
        B = int(wids.numel())
        if B == 0:
            raise ValueError("empty wids")
        wi = wids.numpy()
        layers = self.weights["layer"][wi]
        L = int(layers[0])
        if not np.all(layers == L):
            raise ValueError(f"wids span multiple layers: {sorted(set(layers.tolist()))}")
        mats = self.weights["matrix"][wi]
        rows = self.weights["row"][wi]
        cols = self.weights["col"][wi]

        if restore is not None:
            restore = torch.as_tensor(restore, device="cpu")
            if tuple(restore.shape) != (B, self.n_seq, T):
                raise ValueError(
                    f"restore must be [B,n_seq,T]={(B, self.n_seq, T)}, got {tuple(restore.shape)}")
            rc = restore
            any_restore = bool(rc.any())
        else:
            rc = None
            any_restore = False

        n_seq = self.n_seq
        tgt_all = self.tokens[:, 1:]
        ce_out = torch.empty((B, n_seq, T), dtype=torch.float32, device=self.device)
        kl_out = (torch.empty((B, n_seq, T), dtype=torch.float32, device=self.device)
                  if want_kl else None)

        for b in range(B):
            mat, row, col = int(mats[b]), int(rows[b]), int(cols[b])
            for s in range(0, n_seq, self.seq_chunk):
                e = min(s + self.seq_chunk, n_seq)
                x_in = self._resid_mid[L][s:e].to(self.device)
                base = self._mlp_base(L, x_in)
                if any_restore and bool(rc[b][s:e].all()):
                    out_b = base["out"]
                else:
                    abl = self._mlp_ablated(L, base, mat, row, col)
                    if any_restore:
                        out_b = torch.where(rc[b][s:e].to(self.device).unsqueeze(-1),
                                            base["out"], abl)
                        del abl
                    else:
                        out_b = abl
                x_out = x_in + out_b
                del out_b, base
                logits, _, _ = self.model.forward_tail(
                    x_out, n=L + 1, targets=None, additional_logits_bias=None,
                    include_resid_mid=False, pos_emb_to_cat=self._pos)
                del x_out
                lg = logits.float()
                del logits
                if self._bigram is not None:
                    lg = lg + self._bigram[s:e]
                lp = torch.log_softmax(lg, dim=-1)
                del lg
                ce_out[b, s:e] = -lp.gather(-1, tgt_all[s:e].unsqueeze(-1)).squeeze(-1)
                if want_kl:
                    lb = self._logp_base[s:e]
                    kl_out[b, s:e] = (lb.exp() * (lb - lp)).sum(-1)
                    del lb
                del lp
        torch.cuda.empty_cache()
        return ce_out, kl_out

    # ---------------------------------------------------------------- convenience

    def ablate_one(self, wid: int, want_kl: bool = False):
        return self.ablate(torch.tensor([wid]), want_kl=want_kl)

    def wid_index(self) -> dict[tuple[int, int, int, int], int]:
        w = self.weights
        return {(int(w["layer"][i]), int(w["matrix"][i]), int(w["row"][i]), int(w["col"][i])): i
                for i in range(len(w["layer"]))}


def reference_full_forward(model, tokens: torch.Tensor, layer: int, matrix: int,
                           row: int, col: int, chunk: int | None = None) -> torch.Tensor:
    """Plain full forward with one MLP weight set to 0.0 (the G1 reference).

    ``chunk=None`` means the whole slice in one batch; pass the Replayer's ``seq_chunk`` to
    match its batching exactly.  Returns per-position CE ``[n_seq, 256]`` float32.
    """
    name = "c_fc" if matrix == 0 else "c_proj"
    W = getattr(model.transformer.h[layer].mlp, name).weight
    saved = W[row, col].clone()
    W[row, col] = 0.0
    try:
        logits = base_logits(model, tokens, chunk=chunk or tokens.shape[0])
        return per_position_ce(logits, tokens)
    finally:
        W[row, col] = saved
