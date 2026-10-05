"""Opt-in fp32 CSR profiling replay, validated against the eager numerical floor."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from swa.replay import Replayer, T
from swa.model import DATA, per_position_ce




class FastReplayer(Replayer):
    """Reuse target MLP inputs across variants and accelerate fixed-shape suffixes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure()

    @classmethod
    def from_reference(cls, reference):
        """Share immutable model/slice caches for bounded-memory A/B measurements."""
        obj = cls.__new__(cls)
        obj.__dict__.update(reference.__dict__)
        obj._configure()
        return obj

    def _configure(self):
        if (self.model_name != 'csp_yolo1' or self.dtype != torch.float32
                or self.cat_pos_emb or self.model.config.afrac is not None):
            raise ValueError('fast profiling supports csp_yolo1 fp32 without activation top-k only')
        self._sparse = {}

    @torch.no_grad()
    def load_slice(self, k):
        """Eager fixed-shape prefix, releasing hook-only residuals immediately."""
        arr = np.load(DATA / self.model_name / f'slice_{k:02d}.npy')
        if arr.ndim != 2 or arr.shape[1] != T + 1:
            raise ValueError(f'expected [n_seq,{T + 1}], got {arr.shape}')
        self.slice, self.n_seq = k, int(arr.shape[0])
        self.tokens = torch.from_numpy(arr.astype(np.int64)).to(self.device)
        m = self.model
        resid = [[] for _ in range(self.n_layer)]
        logits_parts = []
        for s in range(0, self.n_seq, self.seq_chunk):
            idx = self.tokens[s:s + self.seq_chunk, :T]
            x = m.transformer.drop(m.transformer.wte(idx) + m.transformer.wpe.weight[:T].unsqueeze(0))
            for L, block in enumerate(m.transformer.h):
                x = block.forward_attn_block(x)
                resid[L].append(x.cpu())
                x = block.forward_mlp_block(x)
            bias = F.embedding(idx, m.bigram_table, padding_idx=-1).to(x.dtype) if m.config.enable_bigram_table else None
            logits_parts.append(m.forward_tail(x, n=self.n_layer, additional_logits_bias=bias)[0].float())
            del x, bias
        self._resid_mid = [torch.cat(parts).contiguous() for parts in resid]
        logits = torch.cat(logits_parts)
        self._ce_base = per_position_ce(logits, self.tokens)
        self._logp_base = torch.log_softmax(logits, dim=-1)
        del logits, logits_parts
        self._bigram = F.embedding(self.tokens[:, :T], m.bigram_table).to(self.dtype) if m.config.enable_bigram_table else None
        self._pos = None
        torch.cuda.empty_cache()


    def _suffix(self, x, L):
        for i in range(L + 1, self.n_layer):
            block = self.model.transformer.h[i]
            x = block.forward_attn_block(x)
            h = block.ln_2(x)
            for name in ('c_fc', 'c_proj'):
                linear = getattr(block.mlp, name)
                key = (i, name)
                if key not in self._sparse:
                    self._sparse[key] = linear.weight.to_sparse_csr()
                shape = h.shape
                h = torch.sparse.mm(self._sparse[key], h.reshape(-1, shape[-1]).T).T.contiguous()
                h = h.reshape(*shape[:-1], h.shape[-1])
                if linear.bias is not None:
                    h = h + linear.bias
                if name == 'c_fc':
                    h = block.mlp.act_fn(h)
            x = x + h
            if block.config.residual_activation_type == 'relu':
                x = torch.relu(x)
            x = block.config.maybe_activation_sparsity(x, 'resid_post_mlp')
        return self.model.lm_head(self.model.transformer.ln_f(x)) + self.model.final_logits_bias + 0

    def _metrics(self, x, L, bigram, target, lb, want_kl):
        logits = self._suffix(x, L).float()
        if bigram is not None:
            logits = logits + bigram
        lp = torch.log_softmax(logits, dim=-1)
        ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        kl = (lb.exp() * (lb - lp)).sum(-1) if want_kl else None
        return ce, kl


    @torch.no_grad()
    def ablate(self, wids, restore=None, want_kl=False):
        if restore is not None:
            return super().ablate(wids, restore=restore, want_kl=want_kl)
        self._require_slice()
        wi = torch.as_tensor(wids, dtype=torch.long, device='cpu').reshape(-1).numpy()
        if not len(wi):
            raise ValueError('empty wids')
        layers = self.weights['layer'][wi]
        L = int(layers[0])
        if not np.all(layers == L):
            raise ValueError('wids span multiple layers')
        B = len(wi)
        mats = self.weights['matrix'][wi]
        ce = torch.empty((B, self.n_seq, T), device=self.device, dtype=torch.float32)
        kl = torch.empty_like(ce) if want_kl else None
        for s in range(0, self.n_seq, self.seq_chunk):
            e = min(s + self.seq_chunk, self.n_seq)
            x = self._resid_mid[L][s:e].to(self.device)
            act = self.model.transformer.h[L].ln_2(x)
            for mat in (0, 1):
                variants = np.flatnonzero(mats == mat)
                if not len(variants):
                    continue
                base = {'act': act}
                if mat == 1:
                    block = self.model.transformer.h[L]
                    z = F.linear(act, block.mlp.c_fc.weight, block.mlp.c_fc.bias)
                    base['a'] = block.mlp.act_fn(z)
                    del z
                for b in variants:
                    wid = wi[b]
                    row, col = (int(self.weights[name][wid]) for name in ('row', 'col'))
                    out = self._mlp_ablated(L, base, mat, row, col)
                    cb, kb = self._metrics(
                        x + out, L, None if self._bigram is None else self._bigram[s:e],
                        self.tokens[s:e, 1:], self._logp_base[s:e], want_kl)
                    ce[b, s:e] = cb
                    if want_kl:
                        kl[b, s:e] = kb
                    del out, cb, kb
                del base
            del x, act
        return ce, kl
