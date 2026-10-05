# x thread draft

draft only; link placeholders must be replaced before posting.

## post 1 (232 chars)

the receipt: 393,216 nonzero mlp weights in csp_yolo1. ablate each weight, search for an executable predicate, then test it. no llm calls in the interpretation loop. the atlas target is the whole model; the measured subset is below.

## post 2 (229 chars)

method: ablate → synthesize → exact held-out test. the predicate controls where a single weight is restored. enumerative rust search proposes code; exact conditional-zero cross-entropy scores what that code preserves and removes.

## post 3 (183 chars)

the uniform pilot report is not available yet. no whole-model interpretation rate is being claimed. the digit-neuron case study is real; it is a selected example, not a random sample.

## post 4 (224 chars)

the paper's digit detector, neuron 1863: 23 scored input/output weights, 8 robust passes. the passing predicates test whether the current token is digit-only. executable source, coverage and held-out scores are in the image.

## post 5 (203 chars)

available score files: 37 weights, 8 robust passes. selected case-study files are not evidence for a whole-model rate. a high raw score alone is not a robust pass: it must beat the coverage-matched null.

## post 6 (199 chars)

corpus: codeparrot/codeparrot-clean-valid. slices are repo-exclusive, one file per repo, at most 512 tokens per document. 16,384 positions per slice. this is a different corpus budget from the paper.

## post 7 (224 chars)

honest limits: fixed causal grammar, not llm hypotheses; train/validation discovery split; coverage-penalized candidate selection, not plain argmax; our coverage-matched random robust null. the readme records the deviations.

## post 8 (95 chars)

browse the atlas and inspect the code, scores and contexts: {site_url}
reproduce it: {repo_url}

## quote tweet

source: https://x.com/Arnauya/status/2106229708438254025

@Arnauya @sheimersheim this pushed me to build the executable version: single-weight ablations → rust predicate search → exact held-out tests. no llm calls in the interpretation loop. real digit-neuron results; whole-model claims wait for the run. {repo_url}
