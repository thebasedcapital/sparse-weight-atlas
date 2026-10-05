## 1. Exact forward path and line references

Relevant locations (line ranges are from the inspected files):
- `gpt.py:48-176`: base attention; `gpt.py:89-101` applies `attn_in` top-k before c_attn, `gpt.py:105-113` applies `attn_k/q/v` after qkv projection, `gpt.py:137-169` invokes flash SDPA or manual causal attention, and `gpt.py:174-176` applies `attn_out` after the output projection.
- `gpt.py:179-209`: base MLP order is `mlp_in` top-k → c_fc → GELU → `mlp_neuron` top-k → c_proj → dropout → `mlp_out` top-k.
- `gpt.py:239-281`: `SDPAWithSink`; it prepends a zero K/V slot, always permits that slot, adds per-head `sink_logit` to its mask, then calls `scaled_dot_product_attention(..., is_causal=False)` with an explicit shifted causal mask.
- `gpt.py:295-358`: residual block order and checkpoint branches.
- `gpt.py:394-486`: cat-pos classes and `BlockCatPosEmb`; `ln_p1`/`ln_p2` are RMSNorm for this config and are applied to the 32-wide positional stream.
- `gpt.py:511-553`: block selection, embedding modules, untied lm_head, final bias, and bigram table.
- `gpt.py:574-621`: top-level embedding/bigram setup and entry to `forward_tail`.
- `gpt.py:624-668`: block loop, final residual hook, final norm, lm_head, final bias, and additional bigram bias.

For this exact config (`model/config.json`):
1. `idx [B,T]` → `transformer.wte(idx)` gives `tok_emb [B,T,2048]` (`gpt.py:594-598`). `wpe.weight[:T].unsqueeze(0)` gives `pos_emb [1,T,32]` (`gpt.py:598-599`). Since `cat_pos_emb` is true, `x=drop(tok_emb)` and position is not added to x (`gpt.py:600-603`). Dropout probability is 0.
2. Bigram path (`gpt.py:606-610`): `F.embedding(idx, bigram_table)` yields `[B,T,2048]`, one entire logit-bias row indexed by the current token at each position. It is not a previous-token lookup; cross-position dependence comes from causal attention. It is cast to `x.dtype` and added only at the final logits.
3. `pos_emb_to_cat=pos_emb` because `dropout_cat_pos_emb=false` (`gpt.py:612-616`).
4. `GPT.__init__` chooses `BlockCatPosEmb` (`gpt.py:511-514`). Each of 8 blocks runs `forward_attn_block` then `forward_mlp_block` (`gpt.py:479-486`).
5. Attention block (`gpt.py:442-461`): save residual input; RMSNorm residual `[B,T,2048]` via `ln_1`; RMSNorm position `[1,T,32]` via `ln_p1`; `CausalSelfAttentionCatPosEmb.forward` broadcasts p to B and concatenates, making `[B,T,2080]` (`gpt.py:394-403`). `attn_in` keeps `int(0.25*2080)=520` entries per token (`gpt.py:89-101`, `gpt.py:462-465`). c_attn is Linear `[3*128*16,2080]=[6144,2080]`, split into q/k/v each `[B,T,2048]`; each of `attn_k`, `attn_q`, `attn_v` keeps 512 entries (`gpt.py:105-113`; config afrac list). Reshape is `[B,128,T,16]`. Flash is enabled and sink is enabled, so `SDPAWithSink` is selected (`gpt.py:75-81`, config `flash:true,sink:true`). Sink is a zero K/V slot with a learnable 128-vector logit; real keys are causal. Output is `[B,T,2048]`, c_proj `[2048,2048]`, then `attn_out` keeps 512 entries (`gpt.py:169-176`). Add to residual; residual activation is identity; `resid_post_attn` is not in the configured afrac list, so no additional top-k.
6. MLP block (`gpt.py:466-478`): RMSNorm residual with `ln_2`, RMSNorm p with `ln_p2`, concatenate to `[B,T,2080]` (`gpt.py:408-424`). `mlp_in` keeps 520 of 2080 (`gpt.py:198-201`). c_fc is `[8192,2080]`; GELU; `mlp_neuron` keeps `int(0.25*8192)=2048`; c_proj is `[2048,8192]`; dropout 0; `mlp_out` keeps 512 of 2048. Add MLP delta to residual; identity residual activation; `resid_post_mlp` is not selected.
7. After block 7, `final_resid` is saved, `transformer.ln_f` is RMSNorm over 2048 (`gpt.py:664-665`). Untied `lm_head` maps `[B,T,2048]` to `[B,T,2048]` (`gpt.py:526-529`, `gpt.py:666-668`); `final_logits_bias [2048]` is added; then the bigram bias `[B,T,2048]` is added.

## 2. MLP tensor shapes, orientation, and measured sparsity

`config.Linear` returns `nn.Linear` (`gpt.py:536-538`), whose weight convention is `[out_features,in_features]`; forward computes x @ W.T + bias.
- c_fc: `[8192,2080]`; first 2048 input coordinates are residual, last 32 are normalized positional coordinates. Every layer has 16,640 nonzeros out of 17,039,360 (density 0.09765625%; zeros 17,022,720).
- c_proj: `[2048,8192]`; 16,384 nonzeros out of 16,777,216 (density 0.09765625%; zeros 16,760,832).
- Across 8 layers, c_fc nonzeros = 133,120 and c_proj nonzeros = 131,072; total MLP nonzeros = 264,192, matching the stated measurement.

Per-layer `(c_fc nonzeros, max nonzeros/output row, max nonzeros/input column; c_proj nonzeros, max nonzeros/output row, max nonzeros/input column)`:
- L0: `(16640, 51, 516; 16384, 135, 55)`
- L1: `(16640, 151, 521; 16384, 68, 150)`
- L2: `(16640, 110, 256; 16384, 69, 101)`
- L3: `(16640, 70, 95; 16384, 131, 73)`
- L4: `(16640, 44, 251; 16384, 83, 46)`
- L5: `(16640, 48, 138; 16384, 86, 37)`
- L6: `(16640, 53, 106; 16384, 99, 52)`
- L7: `(16640, 34, 166; 16384, 176, 52)`

The safetensors header was parsed directly: 12,944-byte JSON header, metadata `{'format':'pt'}`. Example header entries: c_fc dtype `F32`, shape `[8192,2080]`, offsets `[101540608,169698048]`; c_proj dtype `F32`, shape `[2048,8192]`, offsets `[169706240,236815104]`; bigram dtype `F32`, shape `[2048,2048]`, offsets `[0,16777216]`. Value reads confirmed all listed matrix zeros compare exactly to numeric 0.0; all nonzero counts above are from `t != 0`, not a threshold.

## 3. Exact single-weight ablation dependency

For c_fc weight `W[o,j]` (row/output neuron o, column/input j), the multiplied activation is the j-th coordinate of the concatenated `[RMSNorm(residual), RMSNorm(position)]` vector after `mlp_in` top-k. If j is not selected by that token's mlp_in top-k, its activation is exactly zero. The weight contributes to c_fc preactivation neuron o; after it are GELU and `mlp_neuron` AbsTopK (2048/8192), so changing/ablating it can change both neuron values and the selected post-GELU top-k set. Then c_proj, `mlp_out` top-k (512/2048), residual addition, and later blocks follow.

For c_proj weight `W[o,j]`, the multiplied activation is hidden coordinate j after c_fc, GELU, and `mlp_neuron` top-k. It contributes to output coordinate o before dropout and before `mlp_out` AbsTopK. Thus changing it can change the `mlp_out` selected set and the residual stream. In both cases, use the actual stored parameter value and set only that entry to zero; do not merely subtract an effect after top-k, because downstream top-k sets are data-dependent.

Biases are not part of the requested weight ablation; both Linear layers have bias=True. If implementing algebraic batched ablation, c_fc delta is `-W[o,j]*A[...,j]` at preactivation coordinate o; c_proj delta is `-W[o,j]*H[...,j]` at output coordinate o, where A/H are the exact post-mlp_in/post-mlp_neuron activations respectively. Then rerun the relevant nonlinear/top-k operation.

## 4. Cross-position mixing, precision, checkpointing

There is no cross-position mixing in embeddings, RMSNorm, GELU, top-k, or either MLP; all act independently on the final dimension. The bigram table is also position-local: `F.embedding(idx, bigram_table)` uses the current token's row and contributes its 2048-vector to that same position's logits. The only cross-position operation is causal self-attention; sink adds a denominator-only slot and no value signal.

The safetensors checkpoint tensors inspected here are F32, while `config.json` says `torch_dtype: float32`. The source itself does not autocast or force fp16; runtime fp16 requires the caller to cast the model/input or use an autocast context. With dropout=0 and eval/no-grad, the mathematical path is deterministic, but GPU SDPA backend/reduction arithmetic can still produce small backend/precision-dependent differences; exact bitwise reproducibility requires fixed dtype/backend and relevant PyTorch deterministic settings.

`grad_checkpointing` defaults true and is tested directly in the block methods (`gpt.py:303-314`, `gpt.py:329-338`); it is not gated on `training`. The wrapper is in `hook_utils.py:120-158` and calls `torch.utils.checkpoint.checkpoint(..., use_reentrant=...)`. For inference, bypass it by setting `model.config.grad_checkpointing=False` before the forward, or invoke block methods with that flag false. This avoids recomputation/context overhead and is preferable for cached/replay passes.

## 5. Concrete cache/replay/batching recipe

A. Cache layer-input residuals:
```python
model.eval(); model.config.grad_checkpointing = False
with torch.no_grad(), hook_recorder(r"^\\d+\\.resid_in$") as h:
    logits, loss, _ = model(idx)
resid_in = {int(k.split('.')[0]): v.detach().clone() for k,v in h.items()}
```
This records x immediately before each attention block, after embedding and all preceding blocks. For MLP replay also record `r"^\\d+\\.resid_mid$"`; this is the post-attention residual immediately before the layer's MLP. The corresponding `pos_emb_to_cat` is `model.transformer.wpe.weight[:T].unsqueeze(0)` (expand for B), and additional logits bias is `F.embedding(idx, model.bigram_table).to(x.dtype)`.

B. Replay beginning at layer L's MLP: cache `resid_mid[L]`, calculate the exact modified MLP output for that batch (including `ln_2`, `ln_p2`, position concat, mlp_in, c_fc/GELU/mlp_neuron/c_proj/mlp_out, and residual add), then run later complete blocks with `model.forward_tail(x_after_mlp, n=L+1, additional_logits_bias=bigram_bias, pos_emb_to_cat=pos, include_resid_mid=False)`. It returns logits after final RMSNorm/lm_head/bias/bigram. An alternative that preserves the split-block indexing is `forward_tail(resid_mid[L], n=2*L+1, include_resid_mid=True, ...)`, but direct MLP-then-tail is clearer and avoids relying on the deprecated split mode. For an ablation at c_fc/c_proj in layer L, do not reuse the original post-MLP residual; rerun the layer MLP so downstream top-k and all later computation see the changed stream.

C. Batch B ablations along batch dimension: share/copy the cached `[1,T,2048]` or `[1,T,2080]` inputs by expand, materialize B only where each candidate differs, and apply a fused delta to the relevant preactivation/output coordinate before the following top-k. Use `A[B,T,2080]` and `H[B,T,8192]`; stack/expand the position stream to B. All B candidates can then pass through the same later blocks as one batch. Weight storage is shared; do not copy the 8k×2k matrices. Keep each candidate's exact `(layer, matrix, row=o, col=j, original_value)` metadata.

At T=256, fp16 bytes per candidate for representative live tensors are: residual `[256,2048]` = 1 MiB; cat input `[256,2080]` ≈1.02 MiB; MLP hidden `[256,8192]` = 4 MiB; q/k/v combined `[256,6144]` = 3 MiB; logits `[256,2048]` = 1 MiB. Flash SDPA avoids storing the full attention score tensor. A non-flash implementation would additionally need `[128,256,256]` scores ≈16 MiB fp16 per candidate (32 MiB fp32). Therefore a practical flash replay budget is roughly 10–20 MiB fp16 per candidate (20–40 MiB fp32), depending on kernel workspaces and how many intermediates autograd retains; use `no_grad()` and chunk B. This is an engineering estimate, not a measured allocator trace.

## 6. Dominant operations and sparse speedup

At B=1,T=256, per layer dense multiply counts (counting scalar multiplications, not 2-FLOP MAC convention) are approximately:
- MLP c_fc: `256*2080*8192 = 4.362B`; c_proj: `256*8192*2048 = 4.295B`; total ≈8.657B/layer, ≈69.3B over 8 layers.
- Attention projections: qkv `256*2080*6144 = 3.260B`; output `256*2048*2048 = 1.074B`; score/value contractions together ≈0.268B/layer for 128 heads and T=256. Thus attention is projection-heavy but still about half the MLP arithmetic; Flash reduces materialization/memory cost, not the projection arithmetic.
- Bigram lookup is only `256*2048 = 0.524M` fetched values per pass, negligible versus matmuls (though the 2048² table occupies 16 MiB F32).

The MLP matrices are exactly zero at all measured zero entries and only 0.09765625% dense, so CSR/gather can be mathematically exact if it preserves the same dtype/order and the activation top-k boundaries are handled identically. `[INFERENCE]` Arithmetic-only MLP work could be reduced by roughly 1/1024 (~1000×) with a perfectly efficient sparse kernel. End-to-end speedup will be far lower because attention, norms, GELU/top-k, residuals, and logits remain; a custom fused sparse MLP plus dense attention is plausibly several-fold to low-tens-fold faster, but the actual attainable number depends on kernel launch/indexing overhead and GPU implementation and was not benchmarked here.