# sparse-weight-atlas

Every nonzero MLP weight of OpenAI's weight-sparse transformer `csp_yolo1` (Gao et al. 2025), ablated one at a
time, profiled, and given an executable Python predicate for *when it matters* — found by a Rust program
synthesizer, **no LLM in the loop** — then scored with the exact metric of
[arXiv 2607.02964](https://arxiv.org/abs/2607.02964) ("Individual Parameters in Weight-Sparse Transformers Appear
Interpretable", Marin-Llobet & Heimersheim, NeurIPS 2026).

```
corpus (CodeParrot-valid, 12 repo-exclusive slices) ──► swa.profile   exact single-weight ablation, per-token ΔCE/KL
                                                     ──► swa-synth     Rust predicate search over a token/class/window grammar
                                                     ──► swa.score     exact conditional-zero CE: InterpA, InterpB, robust (K=10)
                                                     ──► swa.export_*  static atlas site + annotated model.py
```

Formats, metric definitions and validation gates: [`CONTRACT.md`](CONTRACT.md). Recon notes: `docs/recon/`.

## Setup

```bash
uv venv .venv && VIRTUAL_ENV=$PWD/.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu126
VIRTUAL_ENV=$PWD/.venv uv pip install safetensors tokenizers numpy pyarrow matplotlib
B=https://openaipublic.blob.core.windows.net/circuit-sparsity/models/csp_yolo1
mkdir -p model/csp_yolo1 && curl -L -o model/csp_yolo1/final_model.pt $B/final_model.pt \
  && curl -L -o model/csp_yolo1/beeg_config.json $B/beeg_config.json
(cd synth && cargo build --release)
export PYTHONPATH=engine
.venv/bin/python -m swa.corpus  --model csp_yolo1          # 12 slices x 64 x 256 tokens
.venv/bin/python -m swa.vocab   --model csp_yolo1
.venv/bin/python -m swa.weights --model csp_yolo1          # weights.npz + baseline CE
```

## Run

```bash
.venv/bin/python scripts/check_g1.py                       # replay == zeroed full forward, bitwise
.venv/bin/python -m swa.pilot --model csp_yolo1 --n 1024   # paper metric on a uniform sample -> runs/csp_yolo1/pilot/report.json
systemd-run --user --unit swa-full --working-directory="$PWD" scripts/run_full.sh   # pilot + all 393,216 weights
cd site && python -m http.server 8765                      # browse the atlas
```

Measured on a Quadro RTX 4000 (8 GB, fp32 — fp16 breaks bitwise exactness): exact profiling 0.9 (layer 0) to
8 (layer 11) weights/s, ~66 h for all 393,216 MLP weights; exact scoring ~43 s/weight; synthesis ~600 weights/s.

Opt-in accelerated profiling keeps the production sequence chunk (64) and fp32, uses CSR MLPs only in the
suffix, and reuses target-layer activations across weight variants. The eager default and shard format are
unchanged. `scripts/check_fast.py` passed 33 weights across layers 0/5/11 and both matrices, including digit
weights 7509/7511/7514: measured eager chunk-64 vs chunk-8 noise floors were `2.40564e-4` per-position ΔCE,
`6.10352e-5` per-position KL, and `1.57797e-7` mean ΔCE; fast mean ΔCE must additionally agree within
`max(1e-7, 1% relative)`. This is numerical-floor equivalence, **not** the default's bitwise G1 guarantee.
Interleaved four-weight benchmarks with the pilot sharing the GPU measured eager/fast throughput of
`0.423/0.548`, `0.715/0.942`, and `4.199/7.551` weights/s at layers 0/5/11. Piecewise-linear extrapolation
over 393,216 weights gives `142.17 -> 107.21 h` (1.326×) **under that contention**, not an uncontended ETA.
Measurements and rejected graph/compile/padded-attention/gather experiments: `runs/csp_yolo1/fast_bench.json`.

```bash
PYTHONPATH=engine .venv/bin/python scripts/check_fast.py --gate --bench
# For a new/resumed run only, with no other profiler writing these shards:
PYTHONPATH=engine .venv/bin/python -m swa.profile --model csp_yolo1 --layers 0,1,2,3,4,5,6,7,8,9,10,11 --fast
```

The existing `swa-full` unit is not switched, stopped, or restarted by this opt-in path.

## Deviations from the paper (documented, deliberate)

- Predicates come from enumerative search over a fixed causal grammar (token sets, character classes, ±8-token
  windows, and/or), not from an LLM; candidates are chosen on a train/validation split of slice 00.
- Exactly-scored candidates are selected by `A.score − 0.5·A.cov`, not the paper's plain `A.score` argmax.
- The robust null is ours (the paper leaves it unspecified): a coverage-matched random position mask per slice;
  pass = mean ≥ 0.75 and mean − 2·SEM > null mean + 2·null SD.
- Slices are 16,384 positions, one file per repository, ≤512 tokens per document.

## Launch assets

```bash
.venv/bin/python scripts/make_launch.py \
  --pilot-report runs/csp_yolo1/pilot/report.json \
  --scores-dirs runs/csp_yolo1/scores_casestudy runs/csp_yolo1/scores_pilot
```

Writes measured stills, a draft X thread, and a ≤30-second 1600×900 browser tour under `assets/launch/`.
Missing pilot results skip `headline.png` and `layers_bins.png` with an explicit message; the case-study
diagram and predicate gallery use existing score files and read-only profile contexts. `--top-n` selects
the gallery size and `--skip-tour` regenerates only images/text. Rendering requires Pillow, matplotlib,
and NumPy; the tour requires `playwright` with Chromium (`.venv/bin/playwright install chromium`) and ffmpeg.
It serves `site/` on an ephemeral localhost port and closes the server/browser afterwards; no GPU is used.
Real site exports carry score-directory and timestamp provenance. Unprovenanced model data is excluded;
if only a labeled fixture can supply a passing card, the recording is named `site_tour_fixture.mp4`.
The thread distinguishes a selected case study from the uniform pilot and never invents a missing rate.
