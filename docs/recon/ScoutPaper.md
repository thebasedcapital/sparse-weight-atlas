## 1. Algorithm 1, constants, and scope

Local extracted paper lines 599–635 reproduce Algorithm 1. Verbatim-equivalent operational sequence:

1. Input M, one nonzero weight w, corpus X, auto-interp LLM A, coverage cap c; output best predicate f* and (r*, v*, p*, s*).
2. Run M and M^{w=0} on X.
3. Compute ΔCE ← CE_∅ − CE_0.
4. Compute KL_p at every corpus position; retain positions accounting for the top 90% of cumulative KL.
5. Pass selected positions/contexts to A and sample K predicates f_1,…,f_K. The prose calls this candidate count N; the pseudocode uses K. Default is N=100 (local lines 238–246; Appendix A.4 lines 692–696). Do not conflate this K with held-out K=10.
6. For each candidate, p_k = (1/|X|) Σ_{x∈X}[f_k(x)]. Reject p_k=0, p_k=1, or p_k>c by assigning s_k=−∞. Thus the effective gate is 0<p≤c; c=0.5. The prose says “at least one position but no more than half.”
7. Build M^{f_k}: ablate w everywhere, restore w only where f_k fires. Build M^{¬f_k}: ablate w everywhere, restore w only where f_k does not fire.
8. r_k = 1 − (CE_{f_k}−CE_0)/ΔCE; v_k = 1 − (CE_{¬f_k}−CE_0)/ΔCE; s_k=min(r_k,1−v_k).
9. Return the candidate maximizing s_k and its r,v,p,s. Interpretability at threshold T requires r*≥T, v*≤1−T, p*≤c; default T=0.75, c=0.5.

Constants and corpus split:
- T=0.75 and c=0.5: local lines 243–246 and 626–632.
- N=100: local lines 238–242 and 692–696.
- top-90%-cumulative-KL selection: local lines 204–210 and Algorithm line 3.
- context window exactly ±8 tokens: local lines 219–224 and 711–720.
- 12 non-overlapping slices: local lines 280–289; slice 1 generates/selects f*, remaining 11 are held out. One held-out slice is used for InterpB; “the other remaining K=10 separate held-out slices” are used for robust resampling (local lines 220–235 and 331–342). Thus there are 12 total = 1 discovery + 1 ordinary held-out + 10 robust held-out, although the prose also describes all remaining slices collectively as held out.
- Figure filenames contain n800 (e.g. robust figure), but the paper does not state that each text slice has 800 sequences/tokens. The only explicit 5,000-sequence sample is the separate Drori neuron 2461 analysis (local lines 776–778), not the Gao whole-model pipeline.
- Sequence length, number of tokens per slice, exact CodeParrot stream size, and exact CE averaging convention are not stated in the paper text/HTML. [UNVERIFIED/NOT SPECIFIED] Do not silently substitute the local block_size=1024 or the local measured T=256 benchmark.

Matrices sampled: the method repeatedly says “single nonzero weight in an MLP layer” and “sample nonzero weights uniformly across MLP layers” (local lines 140–145 and 262–269), without restricting to c_fc. Weightpedia’s live records explicitly include both `mlp.W_in` and `mlp.W_out` (equivalent input/c_fc and output/c_proj directions): e.g. neuron 1863 has W_in and W_out sections; neuron 863 likewise. Therefore the evidence supports both MLP directions, not c_fc-only. The paper does not provide a separate c_fc-vs-c_proj sampling quota.

KL direction: the paper says “KL divergence between the original and ablated model” (local lines 143–147), but does not print the argument order in a formula. The natural reading is KL(P_original || P_ablated), [INFERENCE]. This direction should be confirmed from implementation before reproducing exact numbers.

## 2. Exact metric definitions and what is/ is not specified

Definitions in local lines 213–243 and 599–635:
- CE_0 = CE of unmodified M.
- CE_∅ = CE of globally ablated w.
- ΔCE = CE_∅−CE_0.
- CE_f = CE of global ablation with w restored only where f holds.
- CE_¬f = CE of global ablation with w restored only where f does not hold.
- recovery(f) = 1 − (CE_f−CE_0)/ΔCE.
- inverse(f) = 1 − (CE_¬f−CE_0)/ΔCE.
- score(f) = min(recovery(f), 1−inverse(f)).
- p(f) = fraction of corpus positions where f fires; Algorithm gives p=(1/|X|)Σ[f(x)].
- Candidate survives iff 0<p≤0.5; Algorithm explicitly rejects p=0, p=1, p>c.
- InterpA: best predicate scored on the same discovery text used to write it (Table 1 caption, local lines 244–248).
- InterpB: same selected predicate scored on separate held-out text never shown to the LLM (local lines 274–289 and 331–342).

Robust and reliability:
- K=10 disjoint held-out slices; each weight gets ten held-out scores, summarized by their mean and standard error (local lines 220–235; HTML §3.5).
- Figure 5 defines the plotted mean as s̄=min(rec,1−inv), horizontal variation as standard error, and says robust requires s̄≥0.75 and being outside a “not-interpretable” ±2σ null cone (local lines 297–300; HTML Figure 5 caption).
- The paper does NOT define the null distribution, its center, or the exact σ estimator in prose. It calls it a “not-interpretable baseline” and only labels the cone ±2σ. It does not state whether σ is across random predicates, across weights, across slices, or a pooled SEM-derived quantity. Therefore “null exactly” is under-specified by the release; implementation must not invent a formula. [NOT SPECIFIED]
- Reliability is described as the fraction of cross-weight score variance that survives redrawing the corpus: sample-to-sample wobble is subtracted, and the surviving share is reliability (local lines 331–342; Figure 13 caption lines 678–680 and 815–817). The exact variance-components estimator, bootstrap details, and whether variance is computed on s, rec, or 1−inv are not given. [NOT SPECIFIED]
- ±1 SEM in Table 1 is explicitly the uncertainty on pooled per-weight rates (local lines 244–248, 254–257), but the paper does not give the exact resampling unit/formula.

## 3. Table 1, impact bins, predicate classes

Table 1 (local lines 254–257):
- Gao et al. 2025: InterpA 37.3 ± 3.9%; InterpB 35.3 ± 3.9%; Robust 15.0 ± 1.7%.
- Drori sparse: 21.4 ± 1.5%; 17.2 ± 1.4%; 9.6 ± 1.1%.
- Drori dense: 13.2 ± 1.4%; 9.0 ± 1.2%; 1.5 ± 0.6%.
- Pythia-70m: 12.0 ± 0.9%; 5.0 ± 0.6%; 0.4 ± 0.3%.

Impact bins: Figure 1 is the only source for per-bin numeric bars. The text gives no machine-readable per-bin table and the extracted paper/HTML captions do not expose the bar values. Exact textual facts are: bins span above 10^-3 down below 10^-6 (local lines 312–318 / 575–576); Figure 1’s gray bars are total nonzero weights per |ΔCE| bin and colored bars are the fraction passing the interpretability test, with ±1 SEM; exact-zero parameters are a separate rightmost column excluded from rates (local lines 303–310). The high-impact region |ΔCE|≥10^-4 contains a real sparse-model population and “roughly a quarter” of sparse high-impact weights are robustly interpretable; dense models barely populate it (local lines 303–308). The <10^-6 bin is at the CE noise floor and can score accidentally (local lines 312–325). Exact Gao per-bin counts/rates are therefore unavailable from the released text and should be read from Figure 1 or reconstructed from the authors’ underlying data if obtained. [NOT MACHINE-READABLE IN SOURCES FETCHED]

Predicate classes (Appendix A.2 in this paper is actually titled “Distributions of recovery and inverse”; structural predicate statistics are Figure 12 and §4.3):
- Most common: single-token lookup/membership, examples “focus token is a digit,” “token in {th, ode, b},” “mid-word lowercase fragment” (local lines 382–384).
- Smaller fraction: immediate-neighbor tests, e.g. previous word ends in -ed, next token is an inflectional suffix (local lines 385–387).
- Longer predicates usually span only two or three adjacent tokens; no predicates maintaining prompt state such as inside an open string were observed (local lines 388–392).
- Figure 12 reports class distribution and code length, but exact percentages are only graphical. Text gives median predicate length ≤4 lines for all models, and 2 lines for Drori sparse and dense (local lines 649–652). The explicit example is `tokens[pos] in {"ed","ing","s","es","ly"}` (local lines 453–455).

## 4. Gao checkpoint identity vs local HF model

Evidence identifies the paper/Weightpedia deployment as `csp_yolo1`, not the local 8-layer checkpoint:
- Weightpedia flat page says active “Gao et al. 2025”, `csp_yolo1`, 118M params, 95% sparse.
- Weightpedia browse page says 191,701 weights with effect and **12 Layers**, then enumerates layers 0–11 with nonzero counts: 44,777; 44,893; 43,860; 44,884; 44,921; 43,951; 44,973; 45,008; 44,006; 45,019; 44,051; 45,065. It links neuron 1863 and neuron 863 as layer-0 studies.
- Original Gao paper arXiv HTML Figure 4 describes the quote circuit’s attention as layer 10, consistent with a 12-layer csp_yolo1-like model.
- Weightpedia’s case-study pages explicitly decode Table 3-style data with the `tinypython_2k` tokenizer (paper local lines 731–733; HTML Table 3 caption).
- The local `/home/dell/sparse-weight-atlas/model/config.json` is explicitly 8 layers, 2048 model width, 8192 MLP width, 128 heads ×16, vocab 2048, block_size 1024. This is not the 12-layer Weightpedia/csp_yolo1 model.
- GitHub README says `csp_yolo1` is the 118M model used for `single_double_quote` qualitative results; `csp_yolo2` is a 475M model used for `bracket_counting` and `set_or_string_fixedvarname`. README does not print layer counts, but the live Weightpedia data and the original paper’s layer-10 circuit establish csp_yolo1=12 layers; the local 8-layer checkpoint is consistent with csp_yolo2 being a different release, not the paper’s csp_yolo1 weightpedia dataset. Exact csp_yolo2 tokenizer/model config was not fetched from the blob checkpoint in this read-only investigation. [PARTLY VERIFIED]

## 5. Case studies / parity targets

### Neuron 1863, layer 0, MLP
Paper Table 3 (local lines 707–710, 731–745) gives nine nonzero input channels, highest-KL representative tokens, decoded with `tinypython_2k`:
- ch 464: max KL 26.8; `100, 200, 300, 111`.
- ch 71: 22.7; `64, 50, 59, 60`.
- ch 395: 17.1; `28, 08, 27, 35`.
- ch 799: 16.1; `32, 56, 41, 46`.
- ch 578: 13.7; `16, 12, 13, 17`.
- ch 131: 8.0; `4, 2, 3, e`.
- ch 678: 6.0; `7, 8`.
- ch 87: 1.9; `90, 89, 95, f`.
- ch 558: 1.4; `6, 20, 8, 67`.

Weightpedia adds exact W_in indices/ranks/max-KL/mean-KL for these plus a tenth low-impact input weight omitted from Table 3:
- [464,1863], rank #16, max 26.7662, mean 0.009274;
- [71,1863], #23, 22.7326, 0.004690;
- [395,1863], #30, 17.1137, 0.002023;
- [799,1863], #34, 16.0585, 0.002183;
- [578,1863], #40, 13.6505, 0.005442;
- [131,1863], #103, 8.0407, 0.030843;
- [678,1863], #163, 5.9731, 0.004917;
- [87,1863], #1032, 1.9184, 0.000055;
- [558,1863], #2101, 1.0570, 0.000906;
- [299,1863], #3751, 0.0121, 0.000036 (not in Table 3).

Weightpedia neuron 1863 says total connected weights=21: 10 W_in and 11 W_out. W_out indices/ranks/max-KL/mean-KL are: [1863,968] #718 2.4633/0.003398; [1863,773] #1166 1.7583/0.001808; [1863,495] #2617 0.8604/0.001381; [1863,360] #3180 0.6373/0.000885; [1863,915] #3694 0.4525/0.001033; [1863,126] #3714 0.4328/0.000428; [1863,903] #3745 0.2944/0.000787; [1863,756] #3747 0.2026/0.000323; [1863,396] #3748 0.1069/0.000297; [1863,267] #3749 0.0995/0.000183; [1863,696] #3750 0.0833/0.000105.

Interpretation stated by the paper: the neuron is predominantly numeric; its input predicates cluster into single-digit, two-digit, three-digit, generic, and other gates, collectively composing a “detects digits” activation label (local lines 382–396 and 707–710). The exact nine predicate source strings, recovery, inverse, score, and coverage values are printed only inside Figure 7 (image), not in paper text/HTML table. Weightpedia’s per-weight JSON exposes KL histograms and decoded contexts, but grep/inspection found no predicate/recovery/inverse/score fields. [NOT MACHINE-READABLE IN SOURCES FETCHED]

### String-closing, neurons 863 and 2790, layer 0, MLP
Paper Appendix A.7 (local lines 711–720) says the token-local-only best score is −0.98; permitting look-back `TOKENS[pos−k]` for k≤8 gives +0.83. This is the explicit parity target for context-window support. It does not print the exact winning predicate, recovery, inverse, or coverage.

Weightpedia neuron 863: double-quote detector, 8 total connected weights = 4 W_in + 4 W_out.
- W_in [207,863] rank #172 max KL 5.8564 mean 0.000400.
- W_in [294,863] #367 3.7430/0.000933.
- W_in [864,863] #808 2.2600/0.000640.
- W_in [463,863] #1122 1.8087/0.000133.
- W_out [863,460] #208 5.2128/0.002293.
- W_out [863,985] #716 2.4641/0.001015.
- W_out [863,363] #3661 0.4693/0.000053.
- W_out [863,606] #3752 0.0026/0.000003.

Weightpedia neuron 2790: single-quote detector, 6 total connected weights = 3 W_in + 3 W_out.
- W_in [316,2790] rank #240 max KL 4.7504 mean 0.000261.
- W_in [205,2790] #494 2.9701/0.000662.
- W_in [180,2790] #579 2.6834/0.001804.
- W_out [2790,460] #163 5.8814/0.002123.
- W_out [2790,985] #222 4.9557/0.000829.
- W_out [2790,363] #3690 0.2531/0.000053.

The original Gao paper’s mechanistic circuit uses layer 0 MLP quote detector/type-classifier channels and layer-10 attention head 82; Figure 17 of the target paper expands per-weight cards for the two quote detectors. Exact Figure-17 predicate strings and rec/inv/score values are image-only in the fetched sources; the live Weightpedia pages provide indices and KL but not those cards’ predicates/scores.

## 6. Weightpedia machine-readable status

Confirmed machine-readable URL pattern: `https://weightpedia.org/individual-parameters-in-sparse-transformers/data/deep_profile_<rank>.json`, loaded by the weight page’s inline JS (`fetch('../data/deep_profile_' + WEIGHT_RANK + '.json')`). Example rank 16 JSON includes param name, indices, rank, layer, component, max/mean KL, 50-bin log-KL histogram, percentiles, per-bin counts/examples, token IDs, decoded tokens, prompt KL, probability shifts, and CE deltas. Rank 172 works similarly.

The HTML has machine-rendered fields for deep KL profile and an inline client function, but inspection of the JSON and page found no `predicate`, `recovery`, `inverse`, `score`, or coverage fields. Thus per-weight KL/context data is machine-readable; per-weight interpretability predicates/scores are not exposed at the discovered JSON URL. Flat `weights.html` is an Alpine client table and does not itself embed all rows; layer pages are server-rendered index tables.

## 7. Other released resources and licenses

- `openai/circuit_sparsity` repository: inference GPT implementation, hook utilities, sparse kernels, tokenizer extension (`tinypython`), visualizer, training code, examples, tests, registries, and model/viz data layout. README says models are hosted under `https://openaipublic.blob.core.windows.net/circuit-sparsity` with `models/<model_name>/beeg_config.json` + `final_model.pt`; visualization payloads under `viz/.../viz_data.pkl`; train curves under `train_curves/.../progress.json`; raw pruning artifacts under experiment directories.
- Repository LICENSE is Apache License 2.0. Note pyproject classifiers incorrectly list MIT, but the checked-in LICENSE and GitHub repository metadata say Apache-2.0; use the actual LICENSE for source redistribution.
- The original Gao paper arXiv:2511.13653 is CC BY 4.0 (arXiv page and HTML license banner). The target paper arXiv:2607.02964 is also CC BY 4.0.
- Weightpedia pages/data were fetched successfully, but no explicit site/data license was found in the inspected pages. Treat Weightpedia data licensing as [UNVERIFIED] and do not assume it inherits either arXiv CC BY or the GitHub Apache license.
- The model blob/checkpoint-specific license was not surfaced by the README or fetched metadata; [UNVERIFIED].

## Source line map
- Local paper Algorithm/decision: 599–635.
- KL/CE and candidate generation: 140–160, 204–224.
- Conditional-zero formulas and coverage: 213–246.
- Slice split / InterpB / K=10: 274–289, 331–342.
- Table 1: 244–257.
- Impact-bin discussion: 286–325.
- Reliability discussion: 331–342, 678–680, 815–817.
- Predicate class prose: 382–396, 649–652.
- Table 3 and A.6/A.7: 707–720, 731–750.
- Weightpedia live facts: browse.html, weights.html, neuron/1863.html, neuron/863.html, neuron/2790.html, data/deep_profile_16.json, data/deep_profile_172.json.
- GitHub resources/license: README and LICENSE at github.com/openai/circuit_sparsity.