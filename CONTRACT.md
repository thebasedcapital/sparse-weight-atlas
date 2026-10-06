# sparse-weight-atlas — build contract

Goal: every nonzero MLP weight of OpenAI's weight-sparse transformer, ablated, profiled, and explained by an
executable Python predicate found WITHOUT an LLM, scored with the exact metric of arXiv 2607.02964
("Individual Parameters in Weight-Sparse Transformers Appear Interpretable"), published as a static atlas.

## Models
| name | source | layers | d_model | d_mlp | MLP nonzeros | notes |
|---|---|---|---|---|---|---|
| `csp_yolo1` (PRIMARY) | `model/csp_yolo1/final_model.pt` + `beeg_config.json` (state dict, no prefix) | 12 | 1024 | 4096 | 393,216 (32,768/layer) | the paper's + Weightpedia's model. `afrac: null` → no activation top-k. |
| `csp_yolo2` | `model/model.safetensors` (keys prefixed `circuit_model.`) + `model/config.json` | 8 | 2048 | 8192 | 264,192 | HF `openai/circuit-sparsity`; activation AbsTopK afrac 0.25 everywhere. |

Both load with `engine/circuit_sparsity/gpt.py`: `GPT(GPTConfig(**{k:v for k,v in cfg.items() if k in dataclass fields}))`,
`load_state_dict(...)` → "All keys matched". ALWAYS set `model.config.grad_checkpointing = False` for inference
(it is on by default and wraps blocks in torch.utils.checkpoint). Tokenizer: `model/tokenizer.json` (ByteLevel BPE,
vocab 2048, `<|endoftext|>` = 2047; same tinypython_2k vocab for both models — the engine must assert decoded
vocab equality is plausible by checking CE, see G2).

Forward facts (csp_yolo2 verified by recon; csp_yolo1 same code path minus top-k): cat-pos-emb blocks
(`BlockCatPosEmb`), attention then MLP per block; MLP = RMSNorm(resid)‖RMSNorm(pos32) → [mlp_in topk] → c_fc
→ GELU → [mlp_neuron topk] → c_proj → [mlp_out topk] → +resid. Bigram logit bias is per-current-token, added at
the end. The ONLY cross-position op is causal attention (with a denominator-only sink). `nn.Linear` weights are
`[out, in]`. For csp_yolo1, check the actual block class / whether pos-emb is concatenated from `d_pos_emb` in its
config (it has none → likely additive wpe) — read gpt.py, do not assume.

Weightpedia orientation: `W_in [i, n]` = `c_fc.weight[n, i]`; `W_out [n, o]` = `c_proj.weight[o, n]`.

## Layout
```
engine/circuit_sparsity/   vendored model code (do not edit semantics)
engine/swa/                python package (PYTHONPATH=engine), run with .venv/bin/python
synth/                     Rust crate `swa-synth` (binary `swa-synth`)
site/                      static atlas (no build step required to view; plain HTML/JS/CSS)
data/<model>/              corpus slices
runs/<model>/              all pipeline outputs
```

## File formats (THE interface between slices — do not deviate)
All arrays are `.npy` (little-endian, C order). JSON is UTF-8. JSONL = one object per line.

### Corpus — `data/<model>/`
- `slice_{00..11}.npy`: `uint16 [n_seq, 257]`. Inputs `[:, :256]`, targets `[:, 1:]`. Every one of the 256
  positions per row is scored. Default `n_seq = 64` (16,384 positions/slice); configurable.
- Slice 00 = discovery, 01 = InterpB held-out, 02..11 = K=10 robust held-out.
- Source: HF datasets-server `codeparrot/codeparrot-clean-valid` (`/rows` or the parquet shards listed in
  `docs/recon/ScoutCorpus.md`). Documents tokenized, `2047` appended after each, packed into 257-token rows.
  Slices are repo-exclusive (group by `repo_name`, assign groups by sha256(repo_name)), content-hash deduped.
  Diversity caps: one file per repo (`--files-per-repo 1`) and at most 512 tokens per document incl. EOS
  (`--max-doc-tokens 512`). Without them a 16k-token slice was ~5 large files, and per-slice token statistics
  swung from 0 to 49 occurrences of 3-digit tokens across slices.
- `SWA_DATA` env var overrides the corpus root (default `data/`), e.g. `SWA_DATA=data_div256` for 256-seq slices.
- `manifest.json`: per slice: rows used (repo_name, path, hash, license, doc_truncated), token counts.

### Weights — `runs/<model>/weights.npz`
Arrays, all length W (= number of nonzero MLP weights), ordered: layer asc, matrix (0=c_fc, 1=c_proj), then
`torch.nonzero` row-major order. Index into these arrays = **wid**.
`layer int16, matrix int8, row int32, col int32, value float32`.

### Baseline — `runs/<model>/base/`
`ce0_slice{kk}.npy float32 [n_seq,256]` per-position CE of the unmodified model (fp32 accumulation).

### Profiles (discovery slice 00) — `runs/<model>/profiles/<shard>/` (shard = e.g. `L03_fc_000`)
- `wid.npy int32 [n]`
- `dce_total.npy float32 [n]` = Σ_positions (CE_abl − CE_0) over the whole slice
- `kl_total.npy float32 [n]`, `kl_max.npy float32 [n]` (KL(P_orig ‖ P_abl) per position, nats)
- CSR of kept positions: `indptr.npy int64 [n+1]`, `pos.npy int32` (flat = seq*256 + t), `dce.npy float32`,
  `kl.npy float32`. Kept set per weight = union(positions covering top 90% cumulative KL, positions covering 99% of
  Σ|dce|), capped at 4096 by |dce|+kl rank.
- `dce_rest.npy float32 [n]` = Σ dce over positions NOT kept.
- `meta.json`: model, slice, dtype, n_positions, timing.

### Vocab classes — `runs/<model>/vocab.json`
`{"tokens": [decoded string for id 0..2047], "classes": {"digit": [ids...], ...}}` produced by the engine
(see class list below). Rust and Python both treat a class purely as its id set.
Classes (a token id belongs if its decoded string satisfies): `digit_only`, `has_digit`, `alpha_only`, `upper_start`,
`lower_only`, `space_lead` (starts with space), `newline` (contains \n), `indent` (only whitespace),
`punct_only`, `quote` (contains ' or "), `bracket_open` (contains ( [ {), `bracket_close` (contains ) ] }),
`operator` (only chars of +-*/%=<>!&|^~), `dot`, `comma`, `colon`, `underscore`, `keyword` (strip() is a Python keyword),
`eot` (id 2047).

### Predicate spec (shared AST; causal only — never reads positions > pos)
```
{"op":"tok_in","off":k,"ids":[...]}            tokens[pos+k] ∈ ids,  k ∈ [-8, 0]
{"op":"cls","off":k,"cls":"digit_only"}       tokens[pos+k] ∈ vocab.classes[cls]
{"op":"win_in","lo":a,"hi":b,"ids":[...]}     ∃ j∈[pos+a, pos+b]: tokens[j] ∈ ids,  -8 ≤ a ≤ b ≤ -1
{"op":"win_cls","lo":a,"hi":b,"cls":"..."}    same with a class
{"op":"and","args":[A,B]}  {"op":"or","args":[A,B]}  {"op":"not","arg":A}
```
Out-of-range positions (pos+k < 0 within the 256-window) evaluate to false. Evaluation is per row of the slice
(positions never cross rows).
Emitted Python: `def f(tokens, pos):` where `tokens` is a list of decoded token STRINGS; must be semantically
identical to the spec (Python evaluator in engine checks this by exec on samples).

### Candidates — `runs/<model>/candidates/<shard>.jsonl`
`{"wid": int, "cands": [{"spec": <spec>, "py": "<source>", "desc": "<one line>",
  "proxy": {"rec": f, "inv": f, "cov": f, "score": f, "score_train": f, "score_val": f,
            "kl_recall_val": f}}]}` — top M (default 8) by validation objective, coverage
`0 < cov ≤ 0.5`.
Proxy (local-effect approximation, exact when the weight's effect does not propagate across positions):
`rec ≈ 1 − Σ_{p∉f} dce_p / dce_total`, `inv ≈ 1 − Σ_{p∈f} dce_p / dce_total`, `score = min(rec, 1−inv)`;
`dce_rest` counts as ∉f (conservative). `cov` = fraction of all slice positions where f fires.

Synthesis selection (swa-synth). Slice 00 is split by row parity into a TRAIN half (even rows)
and a VALIDATION half (odd rows): greedy token sets are seeded from the weight's top-KL
positions on the train half (ids occurring ≥ 2 times there) and grown on the TRAIN objective,
but the emitted ordering is by the VALIDATION objective, and a candidate is dropped when
another candidate with the same objective fires strictly inside it. The objective is
`min(clip(rec), 1 − clip(inv), kl_recall) − λ·complexity − μ·cov`, where `rec`/`inv` are the
signed ratios of the half, `kl_recall` is the fraction of the weight's KL mass (`kl.npy`)
inside f, and `complexity` counts the distinct decoded token strings the predicate names (a
class or a window counts 1); λ defaults to 0.01 (`swa-synth --lambda`) and μ to 0.5
(`swa-synth --cov-penalty`).
μ prices the fired positions themselves: the first three terms saturate at ~1 for any
predicate covering the weight's dce/KL mass and `kl_recall` is monotone in coverage, so
without μ the broadest candidate under the 0.5 cap wins (near-cap `punct_only` windows
outranked `digit_only` for the paper's neuron-1863 weights). At μ=0.5 a predicate firing on
40% of the slice must be ~0.2 better than a tight one to win, while a 2.9%-coverage predicate
pays only 0.015. `proxy` therefore reports, in addition to
the full-slice fields above, `score_train`, `score_val` (the objective on each half) and
`kl_recall_val`.

### Scores — `runs/<model>/scores/<shard>.jsonl`
```
{"wid", "status": "scored"|"noise_floor"|"no_candidate",
 "dce": mean ΔCE per position on slice 00, "spec", "py", "desc",
 "A": {"rec","inv","cov","score"},          exact conditional-zero on slice 00 (InterpA)
 "B": {...same on slice 01...},             InterpB
 "robust": {"scores":[10 floats], "mean","sem","null_mean","null_sd","pass": bool},
 "select_obj": A.score − μ·A.cov of the chosen f*, "cov_penalty": μ}
```
f* selection (OUR deviation from the paper, which takes the plain argmax of `A.score` over its
candidates): among the exactly-scored slice-00 candidates the scorer takes the argmax of
`A.score − μ·A.cov`, μ default 0.5 (`swa.score --cov-penalty`); `--cov-penalty 0` reproduces
the paper's plain argmax. The same μ is the synthesis coverage penalty, so candidate ordering
and f* choice price coverage identically. Rationale: a broad predicate that restores the
weight over most of the slice scores almost as well as a tight one on slice 00 while failing
the coverage-matched null on held-out slices; see `runs/csp_yolo1/exp_slices/`.
Exact conditional-zero: weight zeroed everywhere, its contribution restored exactly at positions where f holds
(c_fc[o,j]: add back `W·A[...,j]` to preactivation o at those positions BEFORE any following nonlinearity/top-k;
c_proj[o,j]: add back `W·H[...,j]` to output o before mlp_out top-k), full suffix of the network re-run.
Paper metric (T=0.75, c=0.5): `rec = 1 − (CE_f − CE_0)/ΔCE`, `inv = 1 − (CE_¬f − CE_0)/ΔCE`,
`score = min(rec, 1−inv)`, interpretable iff `score ≥ 0.75 and 0 < cov ≤ 0.5`.
Robust: per held-out slice k=02..11, score s_k of f*, and null score n_k of a coverage-matched random position
mask (same per-slice coverage, fixed seed). Three rules are reported (`swa.pilot` report):
- `robust` (headline, paper Fig. 5 reading — "above the bar and outside the ±2σ null cone" in the (SEM, mean)
  plane): `mean(s) ≥ 0.75 and mean(s) − 2·sem(s) > 0`.
- `robust_beats_null_mean`: additionally `mean(s) − 2·sem(s) > mean(n)`.
- `robust_strict` (= `robust.pass` stored in score records): `mean(s) − 2·sem(s) > mean(n) + 2·sd(n)`.

### Site data — `site/data/`
`summary.json` (counts, rates, per-layer/per-bin tables), `layer_{LL}.json` (compact per-weight rows for the grid:
wid, matrix, row, col, value, dce, status, score_robust_mean, pass, pass_null_mean, pass_strict, desc),
`w/{wid//1000}.json` (full cards for 1000 wids: spec, py, desc, A, B, robust, the same three top-level
rule booleans, top 3 contexts each as ±8 decoded tokens with the focus index and dce/kl at focus).
Exported `pass` is the headline cone rule; `pass_null_mean` also beats the null mean; `pass_strict` clears
the null mean + 2 SD. Every rule includes the mean ≥ 0.75 gate. These booleans are derived from numeric
score fields; full cards retain the original `robust` record, whose `pass` is strict, without mutation.
Summary counts use `robust_pass`, `robust_beats_null_mean`, `robust_strict`; rates use `robust`,
`robust_beats_null_mean`, `robust_strict`. These rates and InterpA/B divide by all verdicts
(`scored + noise_floor + no_candidate`), excluding only unscored weights. Per-layer/per-bin tables expose
the three rule counts, their rates, and explicit verdict denominators; their existing `pass` count is cone.
Provenance identifies `kind=pipeline_export`, all `scores_dirs` in merge order, and `exported_at` (UTC).
Both exporters accept either `--scores-dir` or `--scores-dirs DIR [DIR ...]`; later directories win by wid.
Merged exports need not be uniform samples: the pure uniform-pilot report remains the headline comparison.
Annotated `model.py` uses the same counts/rates/provenance and verdict denominator, with unchanged
`(wid, row, col, value, predicate_source_or_None, description_or_None)` tuples.

## Validation gates (owners must run their own gate once at the end of their slice)
- **G1 engine exactness:** for ≥6 weights (both matrices, first/middle/last layer, both models): replay CE equals a
  plain full forward with that weight set to 0.0 in a model copy (fp32, max abs per-position diff < 1e-4);
  conditional restore with all-true mask == original CE; all-false == ablated CE.
- **G2 oracle:** csp_yolo1 baseline CE on slice 00 is near the paper's 2.34 (report it); L0 `c_fc[1863,464]` ablation
  max KL is reported next to Weightpedia's 26.766 over 638,206 positions (different corpus — same order of magnitude
  expected); Spearman correlation of our kl_max vs Weightpedia max_kl for the Weightpedia weights whose
  deep_profile JSON is fetched (ranks 1..200 at
  `https://weightpedia.org/individual-parameters-in-sparse-transformers/data/deep_profile_<rank>.json`, fields
  `component` "mlp.W_in"/"mlp.W_out", `layer`, `indices`, `max_kl`).
- **G3 throughput:** weights/second for profiling and exact scoring on this GPU (Quadro RTX 4000, 8 GB, sm_75; other
  processes hold ~1.5 GB), written to `runs/<model>/bench.json`.
