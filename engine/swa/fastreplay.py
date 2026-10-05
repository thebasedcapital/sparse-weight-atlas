"""Opt-in fp32 replay experiments; the reference replayer is never modified."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from swa.replay import Replayer, T


class FastReplayer(Replayer):
    """Reuse target MLP inputs across variants and accelerate fixed-shape suffixes."""
    def __init__(self, *args, technique='sparse', compile_mode='max-autotune-no-cudagraphs', **kwargs):
        super().__init__(*args, **kwargs)
        self._configure(technique, compile_mode)

    @classmethod
    def from_reference(cls, reference, technique='sparse', compile_mode='max-autotune-no-cudagraphs'):
        """Share immutable model/slice caches for bounded-memory A/B measurements."""
        obj = cls.__new__(cls)
        obj.__dict__.update(reference.__dict__)
        obj._configure(technique, compile_mode)
        return obj

    def _configure(self, technique, compile_mode):
        if self.dtype != torch.float32:
            raise ValueError('fast replay requires fp32')
        self.technique = technique
        self.compile_mode = compile_mode
        self._sparse = {}
        self._compiled = {}
        self._graph = None
        self._graph_key = None

    def _suffix(self, x, L):
        if self.technique == 'sparse':
            if self.model.config.cat_pos_emb or self.model.config.afrac is not None:
                raise ValueError('sparse replay requires non-cat, no activation top-k model')
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
        elif self.technique == 'compile':
            torch.compiler.cudagraph_mark_step_begin()
            for i in range(L + 1, self.n_layer):
                if i not in self._compiled:
                    self._compiled[i] = torch.compile(self.model.transformer.h[i], mode=self.compile_mode)
                x = self._compiled[i](x)
        else:
            return self.model.forward_tail(x, n=L + 1, pos_emb_to_cat=self._pos)[0]
        return self.model.lm_head(self.model.transformer.ln_f(x)) + self.model.final_logits_bias + 0

    def _metrics(self, x, L, bigram, target, lb, want_kl):
        logits = self._suffix(x, L).float()
        if bigram is not None:
            logits = logits + bigram
        lp = torch.log_softmax(logits, dim=-1)
        ce = -lp.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        kl = (lb.exp() * (lb - lp)).sum(-1) if want_kl else None
        return ce, kl

    def _graph_metrics(self, x, L, bigram, target, lb, want_kl):
        key = (L, tuple(x.shape), want_kl)
        if self._graph_key != key:
            self._graph = None
            self._graph_key = None
            torch.cuda.empty_cache()
            inputs = [v.clone() if v is not None else None for v in (x, bigram, target, lb)]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self._metrics(inputs[0], L, *inputs[1:], want_kl)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = self._metrics(inputs[0], L, *inputs[1:], want_kl)
            self._graph = (graph, inputs, outputs)
            self._graph_key = key
        graph, inputs, outputs = self._graph
        for dst, src in zip(inputs, (x, bigram, target, lb)):
            if dst is not None:
                dst.copy_(src)
        graph.replay()
        return outputs

    @torch.no_grad()
    def ablate(self, wids, restore=None, want_kl=False):
        self._require_slice()
        wi = torch.as_tensor(wids, dtype=torch.long, device='cpu').reshape(-1).numpy()
        if not len(wi):
            raise ValueError('empty wids')
        layers = self.weights['layer'][wi]
        L = int(layers[0])
        if not np.all(layers == L):
            raise ValueError('wids span multiple layers')
        B = len(wi)
        rc = None if restore is None else torch.as_tensor(restore, device='cpu')
        if rc is not None and tuple(rc.shape) != (B, self.n_seq, T):
            raise ValueError('restore shape must be [B,n_seq,T]')
        ce = torch.empty((B, self.n_seq, T), device=self.device, dtype=torch.float32)
        kl = torch.empty_like(ce) if want_kl else None
        for s in range(0, self.n_seq, self.seq_chunk):
            e = min(s + self.seq_chunk, self.n_seq)
            x = self._resid_mid[L][s:e].to(self.device)
            base = self._mlp_base(L, x)
            for b, wid in enumerate(wi):
                mat, row, col = (int(self.weights[name][wid]) for name in ('matrix', 'row', 'col'))
                out = self._mlp_ablated(L, base, mat, row, col)
                if rc is not None:
                    out = torch.where(rc[b, s:e].to(self.device).unsqueeze(-1), base['out'], out)
                fn = self._graph_metrics if self.technique == 'graph' else self._metrics
                cb, kb = fn(x + out, L, None if self._bigram is None else self._bigram[s:e],
                            self.tokens[s:e, 1:], self._logp_base[s:e], want_kl)
                ce[b, s:e] = cb
                if want_kl:
                    kl[b, s:e] = kb
            del x, base, out, cb, kb
        return ce, kl
