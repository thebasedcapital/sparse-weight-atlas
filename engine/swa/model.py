"""Model loading for the sparse-weight-atlas engine.

Both checkpoints are OpenAI ``circuit_sparsity`` GPTs; see CONTRACT.md "Models".
``load_model`` returns ``(model, cfg_dict)`` with ``grad_checkpointing`` disabled and
the module in eval mode, matching the recon in ``docs/recon/ScoutForward.md``.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import torch

from circuit_sparsity.gpt import GPT, GPTConfig

ROOT = Path(__file__).resolve().parents[2]
# Corpus root; override with SWA_DATA to evaluate on an alternative slice set (e.g. larger slices).
DATA = Path(os.environ.get("SWA_DATA", ROOT / "data"))

MODEL_SPECS = {
    "csp_yolo1": {
        "config": ROOT / "model" / "csp_yolo1" / "beeg_config.json",
        "weights": ROOT / "model" / "csp_yolo1" / "final_model.pt",
        "kind": "pt",
    },
    "csp_yolo2": {
        "config": ROOT / "model" / "config.json",
        "weights": ROOT / "model" / "model.safetensors",
        "kind": "safetensors",
    },
}

TOKENIZER_PATH = ROOT / "model" / "tokenizer.json"


def cfg_dict(model_name: str) -> dict:
    """Raw config with only the keys ``GPTConfig`` actually declares."""
    raw = json.loads(MODEL_SPECS[model_name]["config"].read_text())
    fields = {f.name for f in dataclasses.fields(GPTConfig)}
    return {k: v for k, v in raw.items() if k in fields}


def _state_dict(spec: dict) -> dict:
    if spec["kind"] == "pt":
        obj = torch.load(spec["weights"], map_location="cpu", weights_only=True)
        if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
            obj = obj["model"]
        return obj
    from safetensors.torch import load_file

    sd = load_file(str(spec["weights"]))
    # HF export prefixes every parameter with ``circuit_model.``
    return {k.replace("circuit_model.", "", 1): v for k, v in sd.items()}


def load_model(model_name: str = "csp_yolo1", device: str = "cuda",
               dtype: torch.dtype = torch.float32):
    """Load a model for inference.

    Returns ``(model, cfg_dict)``.  ``grad_checkpointing`` is forced off (it is on by
    default and would wrap every block in ``torch.utils.checkpoint``); the model is in
    ``eval()`` mode.  Callers must use ``torch.no_grad()`` around forwards.
    """
    if model_name not in MODEL_SPECS:
        raise KeyError(f"unknown model {model_name!r}; expected one of {sorted(MODEL_SPECS)}")
    spec = MODEL_SPECS[model_name]
    cfg = cfg_dict(model_name)
    model = GPT(GPTConfig(**cfg))
    sd = _state_dict(spec)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state dict mismatch: missing={missing} unexpected={unexpected}")
    model.config.grad_checkpointing = False
    model.eval()
    model.to(device=device, dtype=dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


def base_logits(model, tokens: torch.Tensor, chunk: int = 8) -> torch.Tensor:
    """Full-model logits for a slice.

    ``tokens`` is ``[n_seq, 257]`` long; returns ``[n_seq, 256, vocab]`` float32 with the
    logits at each scored position.  Chunks over sequences (rows are independent: the only
    cross-position op is causal attention inside a row) to bound memory.
    """
    n_seq = tokens.shape[0]
    out = None
    with torch.no_grad():
        for s in range(0, n_seq, chunk):
            idx = tokens[s:s + chunk, :256]
            logits, _, _ = model(idx)
            logits = logits.float()
            if out is None:
                out = torch.empty(n_seq, 256, logits.shape[-1], dtype=torch.float32,
                                  device=logits.device)
            out[s:s + chunk] = logits
    return out


def per_position_ce(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Per-position CE of ``logits [n_seq,256,V]`` against targets ``tokens[:,1:]``.

    Returns float32 ``[n_seq, 256]``.
    """
    targets = tokens[:, 1:]
    logp = torch.log_softmax(logits.float(), dim=-1)
    ce = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return ce.float()


def layer_mlp(model, layer: int):
    """The ``MLP``/``MLPCatPosEmb`` module of ``layer``."""
    return model.transformer.h[layer].mlp


def default_seq_chunk(model) -> int:
    """Sequence batch size used for baseline forwards and replay suffixes.

    ``csp_yolo2``'s ``SDPAWithSink`` materialises a ``[B, H, T, T+1]`` float mask, so a
    64-sequence batch would need ~2 GiB for that mask alone.
    """
    if getattr(model.config, "sink", False) or d_mlp(model) > 4096:
        return 8
    return 64


def n_layers(model) -> int:
    return len(model.transformer.h)


def d_mlp(model) -> int:
    return model.transformer.h[0].mlp.c_fc.weight.shape[0]


def d_model(model) -> int:
    return model.transformer.h[0].mlp.c_proj.weight.shape[0]
