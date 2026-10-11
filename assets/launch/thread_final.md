# launch thread (hand-edited from thread.md; numbers from runs/csp_yolo1/pilot/report.json and scores_casestudy)

draft only. replace https://github.com/thebasedcapital/sparse-weight-atlas / https://tbcresearch.org/atlas/ before posting. send via steel/x-feed per skill rules after approval.

## 1 — image: assets/receipt_csp_yolo1.png
we reproduced a neurips 2026 interpretability result with zero llm calls.

the paper had an llm write 100 candidate python rules per weight. a rust search over token rules gets 17.3% of weights robustly interpretable vs their 15.0%.

then we stress-tested the test.

## 2 — image: assets/launch/headline.png
same rules, same exact held-out scores, three nulls: 17.3% / 10.5% / 1.8%. most passing rules are broad gates.

but on the 1,669 weights that move the loss most (|Δce| ≥ 1e-3): 40.5% / 39.2% / 33.7%, with narrow rules (median coverage 7%).

## 3 — image: assets/launch/casestudy_1863.png
where it works, it really works: the paper's digit-detector neuron 1863.

the search independently found `tokens[pos].isdigit()` for 8 of its weights. mean held-out score 0.95–1.02 across 10 unseen slices, and they pass even the strict null.

## 4 — video: assets/launch/site_tour.mp4
how: zero one weight → exact per-token loss change → rust search over a causal rule grammar (~600 weights/s) → restore the weight only where the rule fires, measure loss on 11 held-out slices.

160.6m params, 807,936 nonzero, 393,216 mlp weights. atlas of all of them in progress.

## 5
uniform sample of 1,024 weights, codeparrot-valid, every deviation from the paper in the readme.

browse every weight: https://tbcresearch.org/atlas/
code: https://github.com/thebasedcapital/sparse-weight-atlas

## quote tweet of https://x.com/Arnauya/status/2106229708438254025
@Arnauya @sheimersheim reproduced your table 1 rate with no llm in the loop: 17.3% vs 15.0% robust on csp_yolo1, and found the digit gates of neuron 1863 on its own. most passing rules are broad though; vs a coverage-matched null it's 10.5%. code: https://github.com/thebasedcapital/sparse-weight-atlas
