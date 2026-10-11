# sparse-weight-atlas

**Every nonzero MLP weight of OpenAI's weight-sparse transformer, explained by an executable Python rule — with
no LLM in the loop.**

This project is inspired by, and reproduces the metric of,
**[Individual Parameters in Weight-Sparse Transformers Appear Interpretable](https://arxiv.org/abs/2607.02964)**
by Arnau Marin-Llobet and Stefan Heimersheim (NeurIPS 2026; project site
[weightpedia.org](https://weightpedia.org/individual-parameters-in-sparse-transformers/)). The paper asks an LLM
to write 100 candidate Python rules `f(tokens, pos) -> bool` per weight and tests whether a rule explains *when*
that single weight matters. We ask whether a plain program search can do the same, and run it on **all 393,216**
nonzero MLP weights of the model the paper studies, `csp_yolo1` from
[Gao et al. 2025, *Weight-sparse transformers have interpretable circuits*](https://arxiv.org/abs/2511.13653)
([openai/circuit_sparsity](https://github.com/openai/circuit_sparsity)).

## Results

```
 csp_yolo1: 160,570,368 params → 807,936 nonzero transformer params → 393,216 MLP weights (all profiled)

 robustly interpretable (10 held-out slices)     paper's test   + beats random rule   + strict null
 uniform sample, 1,024 weights                   17.3% ± 1.2    10.5% ± 1.0           1.8% ± 0.4
 paper, Table 1 (LLM-written rules)              15.0% ± 1.7
 all 1,669 high-impact weights (|ΔCE| ≥ 1e-3)    40.5% ± 1.2    39.2% ± 1.2          33.7% ± 1.2
```

- **No LLM needed for the headline.** A Rust search over token/character-class/±8-token-window rules reaches
  17.3% robustly interpretable weights under the paper's test (paper: 15.0%).
- **But the test can be met by broad rules.** 65% of those passing rules fire on >30% of all tokens (e.g. "a
  recent token is alphabetic"). Requiring a rule to beat a random position mask of the same coverage leaves
  10.5%; a strict null leaves 1.8%.
- **Where it matters, it is real.** On every weight whose ablation moves the loss by ≥1e-3 nats/token, 40.5%
  pass, a third pass even the strict null, and the passing rules are narrow (median coverage 7%).
- **The paper's digit detector, rediscovered.** For 8 input/output weights of neuron 1863 (paper Fig. 7) the
  search independently finds `tokens[pos].isdigit()`, held-out mean 0.95–1.02 on 10 unseen slices.

All numbers: [`assets/results.json`](assets/results.json). The high-impact set is selected by impact and is not
comparable to the paper's uniform sample.

## How it works

```
CodeParrot-valid, 12 repo-exclusive slices of 16,384 tokens
  └─ swa.profile     zero one weight, replay the model; exact per-token ΔCE / KL     (bit-identical to zeroing it)
      └─ swa-synth   Rust search over a causal rule grammar, ~600 weights/s           (train/validation split)
          └─ swa.score   restore the weight only where the rule fires; exact CE on    (paper metric: rec, inv,
                         the discovery slice, slice 01, and 10 held-out slices         score, coverage ≤ 0.5)
              └─ swa.export_*   static atlas site + annotated model.py
```

Formats, metric definitions and validation gates: [`CONTRACT.md`](CONTRACT.md).

## Reproduce

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
.venv/bin/python scripts/check_g1.py                       # replay == zeroed full forward, bitwise

.venv/bin/python -m swa.pilot --model csp_yolo1 --n 1024   # paper metric on a uniform sample
scripts/run_rest.sh                                         # profile + synthesize all 393,216 weights, export
scripts/run_highimpact.sh                                   # exact scores for every |ΔCE| >= 1e-3 weight
cd site && python -m http.server 8765                       # browse the atlas
```

Every long stage checkpoints per shard and resumes. Measured on one Quadro RTX 4000 (8 GB, fp32): profiling all
weights 50.8 h with `--fast` (CSR suffix MLPs, gated by `scripts/check_fast.py` against eager's own numerical
noise floor), pilot scoring 11.8 h, high-impact scoring 19.4 h. fp16 is not usable: it moves CE by up to 0.86.

## Deviations from the paper

- Rules come from enumerative search over a fixed causal grammar, not an LLM; candidates are chosen on a
  train/validation split of the discovery slice, so our InterpA (76%) is inflated and not comparable.
- Among exactly scored candidates, f* maximizes `A.score − 0.5·A.cov` instead of plain `A.score`.
- The paper does not spell out its robust null. We read its Fig. 5 cone as `mean ≥ 0.75 and mean − 2·SEM > 0`
  (headline) and always report two stricter coverage-matched random-mask nulls next to it.
- Corpus: `codeparrot/codeparrot-clean-valid`, one file per repository, ≤512 tokens per document.
- Atlas cards for weights without exact scores show the synthesizer's best rule labelled **unverified**.

## Credits and licenses

- Method and metric: Marin-Llobet & Heimersheim, [arXiv:2607.02964](https://arxiv.org/abs/2607.02964) (CC BY 4.0).
  Thanks to the authors for the paper and Weightpedia.
- Model, tokenizer and inference code: Gao et al. 2025, [openai/circuit_sparsity](https://github.com/openai/circuit_sparsity).
  `engine/circuit_sparsity/` and `model/*.json` are vendored from it under Apache-2.0
  ([`engine/circuit_sparsity/LICENSE`](engine/circuit_sparsity/LICENSE)). Weights are downloaded, not redistributed.
- Corpus: [codeparrot/codeparrot-clean-valid](https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid),
  downloaded at build time, not redistributed.
- Everything else: MIT, see [`LICENSE`](LICENSE).

```bibtex
@inproceedings{marinllobet2026individual,
  title  = {Individual Parameters in Weight-Sparse Transformers Appear Interpretable},
  author = {Marin-Llobet, Arnau and Heimersheim, Stefan},
  booktitle = {NeurIPS},
  year   = {2026},
  eprint = {2607.02964},
  archivePrefix = {arXiv}
}
```
