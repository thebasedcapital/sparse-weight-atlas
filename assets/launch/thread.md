# x thread draft

draft only; link placeholders must be replaced before posting.

## post 1 (189 chars)

zero-llm search matches the paper's headline under our reading of its test: 17.3% vs 15.0%. but against a coverage-matched random null: 14.1%; strict null: 1.8%. that tension is the result.

## post 2 (163 chars)

receipt: 393,216 nonzero mlp weights in csp_yolo1. uniform pilot: 1,024 weights. cone rate 17.3% ± 1.2% sem; paper 15.0% ± 1.7%. this is not a matched replication.

## post 3 (235 chars)

method: ablate → synthesize → exact held-out test. the predicate controls where a single weight is restored. rust enumerative search proposes code; exact conditional-zero cross-entropy tests what it preserves and removes. no llm calls.

## post 4 (174 chars)

interpa 76.0% ± 1.3% sem vs paper 37.3%; interpb 28.5% ± 1.4% sem vs 35.3%. interpa is inflated: the synthesizer directly searches the discovery slice. held-out tests matter.

## post 5 (216 chars)

116/177 cone-passing predicates (65.5%) have coverage > 30%. most passes are broad class/window gates, like a recent token being alphabetic. the narrow predicate gallery is a selected exception, not the distribution.

## post 6 (205 chars)

cone: mean ≥ 0.75 and mean − 2·sem > 0. random-null mean: the lower bound also beats a coverage-matched random mask's mean. strict null: it beats that mean + 2·sd. same scores, different robustness claims.

## post 7 (187 chars)

high-impact bin |δce| >=1e-4: 24/104 cone passes (23.1%). the paper calls this roughly a quarter. a local high-impact result still does not establish narrow explanations for most weights.

## post 8 (235 chars)

the paper's digit detector, neuron 1863: 23 scored input/output weights, 8 strict-null passes. those passing predicates test the current token for digits. the image keeps executable source, coverage and exact scores beside each weight.

## post 9 (185 chars)

corpus: codeparrot/codeparrot-clean-valid. repo-exclusive slices; one file per repo; at most 512 tokens per document; 16,384 positions per slice. different corpus budget from the paper.

## post 10 (249 chars)

honest limits: fixed causal grammar, not llm hypotheses; discovery train/validation split; coverage-penalized candidate selection, not plain argmax. null-cone reading and both coverage-matched alternatives are explicit. deviations are in the readme.

## post 11 (76 chars)

browse predicates, scores and contexts: {site_url}\nreproduce it: {repo_url}

## quote tweet

source: https://x.com/Arnauya/status/2106229708438254025

@Arnauya @sheimersheim zero-llm search reaches 17.3% vs the paper's 15.0% under its null-cone test. but most predicates are broad; random-null mean 14.1%, strict null 1.8%. executable code + exact tests: {repo_url}
