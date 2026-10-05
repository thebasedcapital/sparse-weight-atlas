"""Exact conditional-zero scoring of predicate candidates (CONTRACT.md "Scores").

CLI:
    # score Rust-synthesised candidates for every shard in runs/<model>/candidates/
    PYTHONPATH=engine .venv/bin/python -m swa.score --model csp_yolo1
    PYTHONPATH=engine .venv/bin/python -m swa.score --model csp_yolo1 --shards L03_fc_000
    # score an explicit (e.g. pilot-sampled) wid list -> runs/<model>/scores_<tag>/<shard>.jsonl
    PYTHONPATH=engine .venv/bin/python -m swa.score --model csp_yolo1 \
        --wids-file runs/csp_yolo1/pilot/wids.npy --tag pilot
    # score arbitrary hand-written specs
    PYTHONPATH=engine .venv/bin/python -m swa.score --model csp_yolo1 \
        --from-specs specs.jsonl --out runs/csp_yolo1/scores/from_specs.jsonl

Per weight w:
  * slice 00 (discovery): full ablation CE_∅, and for each candidate f: CE_f (restore where f
    fires), CE_¬f (restore where f does not fire); rec/inv/score/cov; pick f* by score.
  * slice 01 (InterpB): same three numbers for f*.
  * slices 02..11 (robust, K=10): score s_k of f* and null score n_k of a coverage-matched
    random position mask (same per-slice coverage, fixed seed per wid), plus ΔCE_k per slice.
    pass = mean(s) >= 0.75 and mean(s) - 2*sem(s) > mean(n) + 2*sd(n).

Statuses: "scored", "noise_floor" (|ΔCE| on slice 00 < 1e-6 nats/position), "no_candidate"
(no candidate passes 0 < cov <= 0.5).

Cost: ``score_weight`` is the reference implementation (one weight, slice-inner; used by
``--from-specs``) and reloads slices 01..11 for every weight.  ``Scorer.score_many`` is the
production path: it loops slice-outer (each slice loaded once per call, not once per weight)
and packs up to ``--batch`` variants across weights of one layer into each ``ablate`` call.
``--cov-penalty MU`` (default 0.5) selects f* among the exactly-scored slice-00 candidates by
``A.score - MU * A.cov`` instead of the paper's plain ``A.score`` argmax, and records the
winning ``select_obj`` (``--cov-penalty 0`` reproduces the plain argmax).  See CONTRACT.md.

``--exact-top N`` limits the slice-00 exact evaluation to the top-N candidates by proxy score;
the default ``0`` scores every gated candidate (paper Algorithm 1) and reproduces ``score_weight``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .predicate import check_python_detail, eval_spec, load_vocab, spec_to_python, validate_spec
from .replay import Replayer

COVERAGE_MAX = 0.5
# f* selection: A.score - COVERAGE_PENALTY * A.cov (see CONTRACT.md "Scores"); 0 reproduces
# the paper's plain argmax.  Same default as swa-synth --cov-penalty, so synthesis and
# scoring price coverage identically.
COVERAGE_PENALTY = 0.5
NOISE_FLOOR = 1e-6
ROBUST_T = 0.75
N_ROBUST = 10
SLICE_INTERPB = 1
ROBUST_SLICES = tuple(range(2, 2 + N_ROBUST))


# --------------------------------------------------------------------------------------
# small stats helpers
# --------------------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def _sd(xs: Sequence[float]) -> Optional[float]:
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else (0.0 if xs else None)


def _sem(xs: Sequence[float]) -> Optional[float]:
    if not xs:
        return None
    if len(xs) == 1:
        return 0.0
    return float(np.std(xs, ddof=1) / math.sqrt(len(xs)))


def _ratio(numer: float, denom: float) -> Optional[float]:
    """1 - numer/denom, None when denom is (near) zero."""
    if denom is None or abs(denom) < 1e-30:
        return None
    return 1.0 - numer / denom


# --------------------------------------------------------------------------------------
# scorer
# --------------------------------------------------------------------------------------


class Scorer:
    def __init__(self, model: str, device: str = "cuda", dtype: torch.dtype = torch.float32,
                 batch: int = 8, runs_dir: str = "runs", verbose: bool = True,
                 cov_penalty: float = COVERAGE_PENALTY):
        self.model = model
        self.runs = Path(runs_dir) / model
        self.verbose = verbose
        self.batch = max(1, int(batch))
        self.cov_penalty = float(cov_penalty)
        self.rep = Replayer(model, device=device, dtype=dtype)
        v = load_vocab(model, runs_dir)
        self.vocab_tokens: List[str] = v["tokens"]
        self.classes: Dict[str, List[int]] = v["classes"]
        self.weights = self.rep.weights  # same arrays the Replayer indexes (no second load)
        self.dev = self.rep.device
        self._k: Optional[int] = None
        self._inputs: Optional[torch.Tensor] = None
        self._ce0_mean: Optional[float] = None
        self._n_seq = int(self.rep.n_seq)

    # -- slice handling -----------------------------------------------------------------

    def load(self, k: int) -> None:
        if self._k == k:
            return
        self.rep.load_slice(k)
        self._k = k
        self._inputs = self.rep.tokens[:, : self.rep.T].contiguous()
        ce0 = self.rep.ce_base()
        self._ce0_mean = float(ce0.double().mean())
        self._n_seq = int(self.rep.n_seq)

    @property
    def inputs(self) -> torch.Tensor:
        assert self._inputs is not None
        return self._inputs

    @property
    def ce0_mean(self) -> float:
        assert self._ce0_mean is not None
        return self._ce0_mean

    @property
    def n_positions(self) -> int:
        return self._n_seq * int(self.rep.T)

    def layer_of(self, wid: int) -> int:
        return int(self.weights["layer"][wid])

    def mask(self, spec: Any) -> torch.Tensor:
        return eval_spec(spec, self.inputs, self.classes)

    def select_obj(self, a: Optional[Dict[str, Any]]) -> float:
        """``A.score - mu * A.cov``: the quantity f* is chosen by (-inf when unusable)."""
        if not a or a.get("score") is None:
            return -math.inf
        return float(a["score"]) - self.cov_penalty * float(a.get("cov") or 0.0)

    def _cand_obj(self, cand: Dict[str, Any]) -> float:
        """Selection quantity from a candidate's own proxy (used to order --exact-top)."""
        pr = cand.get("proxy") or {}
        base = pr.get("score_val", pr.get("score"))
        if base is None:
            return -math.inf
        return float(base) - self.cov_penalty * float(pr.get("cov") or 0.0)

    # -- CE variants --------------------------------------------------------------------

    def _ablate(self, wids: Sequence[int], restore: Optional[torch.Tensor]) -> torch.Tensor:
        w = torch.tensor(list(wids), dtype=torch.long, device=self.dev)
        ce, _ = self.rep.ablate(w, restore=restore)
        return ce

    def full_ce(self, wids: Sequence[int]) -> List[float]:
        """Mean CE over all scored positions for full ablation of each wid.

        Passes an explicit all-False restore mask (contract: True = contribution restored), so
        the result does not depend on how ``restore=None`` is interpreted.
        """
        out: List[float] = []
        for i in range(0, len(wids), self.batch):
            part = wids[i:i + self.batch]
            zeros = torch.zeros((len(part), self._n_seq, int(self.rep.T)), dtype=torch.bool,
                                device=self.dev)
            ce = self._ablate(part, zeros)
            out += ce.double().mean(dim=(1, 2)).tolist()
        return out

    def masked_ce(self, wids: Sequence[int], masks: Sequence[torch.Tensor]) -> List[float]:
        """Mean CE for `wid` ablated everywhere except where its mask is True."""
        out: List[float] = []
        for i in range(0, len(wids), self.batch):
            part = masks[i:i + self.batch]
            rest = torch.stack([m.to(torch.bool) for m in part], dim=0)
            ce = self._ablate(wids[i:i + self.batch], rest)
            out += ce.double().mean(dim=(1, 2)).tolist()
        return out

    def _ce_many(self, items: Sequence[Tuple[int, torch.Tensor]]) -> List[float]:
        """Mean CE for a batch of ``(wid, restore)`` pairs on the currently loaded slice.

        ``restore`` is a bool ``[n_seq, T]`` mask (True = the weight's contribution is
        restored, all-False = full ablation).  The Replayer requires every wid of one
        ``ablate`` call to share a layer, so items are grouped by layer and packed up to
        ``self.batch`` variants per call.  Each variant's arithmetic is independent of the
        other members of its call, so the numbers are bitwise identical to per-weight calls.
        """
        out = [0.0] * len(items)
        by_layer: Dict[int, List[int]] = {}
        for j, (w, _m) in enumerate(items):
            by_layer.setdefault(self.layer_of(w), []).append(j)
        for L in sorted(by_layer):
            idxs = by_layer[L]
            for s in range(0, len(idxs), self.batch):
                grp = idxs[s:s + self.batch]
                rest = torch.stack([items[j][1].to(torch.bool) for j in grp], dim=0)
                ce = self._ablate([int(items[j][0]) for j in grp], rest)
                for j, m in zip(grp, ce.double().mean(dim=(1, 2)).tolist()):
                    out[j] = m
                del ce, rest
        return out

    def _zeros(self) -> torch.Tensor:
        """All-False restore mask for the currently loaded slice (full ablation)."""
        return torch.zeros((self._n_seq, int(self.rep.T)), dtype=torch.bool, device=self.dev)

    def null_mask(self, wid: int, k: int, count: int) -> torch.Tensor:
        """Coverage-matched random position mask: `count` True of n_positions, seed (wid, k)."""
        total = self.n_positions
        count = max(0, min(total, int(count)))
        rng = np.random.default_rng([int(wid), int(k)])
        flat = torch.zeros(total, dtype=torch.bool)
        if count:
            idx = rng.choice(total, size=count, replace=False)
            flat[torch.as_tensor(np.sort(idx), dtype=torch.long)] = True
        return flat.view(self._n_seq, int(self.rep.T)).to(self.dev)

    # -- py validation ------------------------------------------------------------------

    def resolve_py(self, spec: Any, py: Optional[str], max_positions: int = 4096) -> Tuple[str, Optional[dict]]:
        """Prefer the provided py when it agrees with the spec; else re-emit from the spec."""
        d: Optional[dict] = None
        if py:
            try:
                d = check_python_detail(spec, py, self.inputs, self.classes, self.vocab_tokens,
                                        seed=0, max_positions=max_positions)
                if d["ok"]:
                    return py, None
            except Exception as exc:  # syntax error / non-callable f / etc.
                d = {"ok": False, "checked": 0, "mismatches": -1, "first_mismatch": repr(exc)}
        return spec_to_python(spec, self.vocab_tokens, self.classes), d

    # -- per-slice metrics --------------------------------------------------------------

    def metrics_here(self, wid: int, spec: Any) -> Tuple[Dict[str, Any], float, torch.Tensor]:
        """A/B-style metrics for one spec on the currently loaded slice."""
        m = self.mask(spec)
        cov = float(m.double().mean())
        ce_full = self.full_ce([wid])[0]
        dce = ce_full - self.ce0_mean
        ce_f, ce_nf = self.masked_ce([wid, wid], [m, ~m])
        rec = _ratio(ce_f - self.ce0_mean, dce)
        inv = _ratio(ce_nf - self.ce0_mean, dce)
        score = None if (rec is None or inv is None) else min(rec, 1.0 - inv)
        return {"rec": rec, "inv": inv, "cov": cov, "score": score}, dce, m

    # -- robust -------------------------------------------------------------------------

    def robust(self, wid: int, spec: Any) -> Dict[str, Any]:
        scores: List[float] = []
        nulls: List[float] = []
        for k in ROBUST_SLICES:
            self.load(k)
            dce = self.full_ce([wid])[0] - self.ce0_mean
            m = self.mask(spec)
            nm = self.null_mask(wid, k, int(m.sum()))
            ce_f, ce_nf, ce_nul, ce_nnul = self.masked_ce(
                [wid] * 4, [m, ~m, nm, ~nm])
            rec = _ratio(ce_f - self.ce0_mean, dce)
            inv = _ratio(ce_nf - self.ce0_mean, dce)
            nrec = _ratio(ce_nul - self.ce0_mean, dce)
            ninv = _ratio(ce_nnul - self.ce0_mean, dce)
            scores.append(min(rec, 1.0 - inv) if None not in (rec, inv) else float("nan"))
            nulls.append(min(nrec, 1.0 - ninv) if None not in (nrec, ninv) else float("nan"))
        ms, mn = _mean(scores), _mean(nulls)
        ss, sn = _sem(scores), _sd(nulls)
        ok = all(not math.isnan(v) for v in scores + nulls)
        passed = bool(ok and ms is not None and mn is not None and ss is not None and sn is not None
                      and ms >= ROBUST_T and (ms - 2 * ss) > (mn + 2 * sn))
        return {"scores": scores, "mean": ms, "sem": ss, "null_mean": mn, "null_sd": sn,
                "pass": passed}

    # -- one weight ---------------------------------------------------------------------

    def score_weight(self, wid: int, cands: List[Dict[str, Any]], gate: bool = True,
                     want_robust: bool = True) -> Dict[str, Any]:
        """cands: [{"spec","py","desc","proxy"?}]. Assumes slice 00 is loaded."""
        assert self._k == 0, "score_weight requires slice 00 loaded"
        rec: Dict[str, Any] = {"wid": int(wid), "status": "scored", "dce": None, "spec": None,
                               "py": None, "desc": None, "A": None, "B": None, "robust": None,
                               "select_obj": None, "cov_penalty": self.cov_penalty}
        dce = self.full_ce([wid])[0] - self.ce0_mean
        rec["dce"] = dce

        # A weight whose ΔCE sits at the CE noise floor has no meaningful rec/inv; that is a
        # property of the weight itself, so it takes precedence over candidate availability.
        if abs(dce) < NOISE_FLOOR:
            rec["status"] = "noise_floor"
            rec["robust"] = {"scores": [], "mean": None, "sem": None, "null_mean": None,
                             "null_sd": None, "pass": False}
            return rec

        cands = [c for c in cands if c.get("spec") is not None]
        if not cands:
            rec["status"] = "no_candidate"
            return rec

        masks: List[torch.Tensor] = []
        covs: List[float] = []
        pys: List[Optional[str]] = []
        cands2: List[Dict[str, Any]] = []
        for c in cands:
            try:
                validate_spec(c["spec"])
            except ValueError as exc:
                print(f"[score] wid={wid}: skipping invalid candidate spec ({exc})", flush=True)
                continue
            m = self.mask(c["spec"])
            cands2.append(c)
            masks.append(m)
            covs.append(float(m.double().mean()))
            pys.append(c.get("py"))
        cands = cands2
        if not cands:
            rec["status"] = "no_candidate"
            return rec

        keep = [i for i, cov in enumerate(covs) if (not gate) or (0.0 < cov <= COVERAGE_MAX)]
        if not keep:
            rec["status"] = "no_candidate"
            return rec

        variants: List[Tuple[int, bool, torch.Tensor]] = []
        for i in keep:
            variants.append((i, False, masks[i]))
            variants.append((i, True, ~masks[i]))
        ces = self.masked_ce([wid] * len(variants), [v[2] for v in variants])
        ce_f: Dict[int, float] = {}
        ce_nf: Dict[int, float] = {}
        for (i, neg, _), ce in zip(variants, ces):
            (ce_nf if neg else ce_f)[i] = ce

        best = None
        best_score = -math.inf
        for i in keep:
            a = {"rec": _ratio(ce_f[i] - self.ce0_mean, dce),
                 "inv": _ratio(ce_nf[i] - self.ce0_mean, dce),
                 "cov": covs[i]}
            a["score"] = None if (a["rec"] is None or a["inv"] is None) else min(a["rec"], 1.0 - a["inv"])
            obj = self.select_obj(a)
            if obj > best_score:
                best_score = obj
                best = (i, a, obj)
        if best is None:
            rec["status"] = "no_candidate"
            return rec
        i, A, select_obj = best
        rec["select_obj"] = select_obj
        spec = cands[i]["spec"]
        py, mismatch = self.resolve_py(spec, pys[i])
        rec["spec"] = spec
        rec["py"] = py
        rec["desc"] = cands[i].get("desc")
        rec["A"] = A
        if mismatch is not None and self.verbose:
            print(f"[score] wid={wid}: provided py disagreed with spec "
                  f"({mismatch['mismatches']}/{mismatch['checked']}); re-emitted from spec")

        # InterpB on slice 01
        self.load(SLICE_INTERPB)
        B, _, _ = self.metrics_here(wid, spec)
        rec["B"] = B

        if want_robust:
            rec["robust"] = self.robust(wid, spec)
        else:
            rec["robust"] = {"scores": [], "mean": None, "sem": None, "null_mean": None,
                             "null_sd": None, "pass": False}
        # leave slice 00 loaded: robust() walked slices 02..11 and callers loop per weight
        self.load(0)
        return rec

    # -- many weights, slice-outer ------------------------------------------------------

    def _log_progress(self, stage: str, done: int, total: int, t0: float,
                      stage_no: int = 0, n_stages: int = 0) -> None:
        if not self.verbose:
            return
        dt = time.time() - t0
        eta = (dt / done) * (total - done) if done > 0 else float("nan")
        tag = f" [{stage_no}/{n_stages}]" if n_stages else ""
        print(f"[score_many]{tag} {stage}: {done}/{total} weights in {dt:.1f}s "
              f"(eta {eta:.1f}s)", flush=True)

    def score_many(self, wids: Sequence[int],
                   cands_by_wid: Dict[int, List[Dict[str, Any]]],
                   gate: bool = True, want_robust: bool = True,
                   exact_top: int = 0) -> Dict[int, Dict[str, Any]]:
        """Score many weights with slice-outer loops; returns ``wid -> record``.

        Same per-weight schema as :meth:`score_weight`.  Slices are loaded once each
        (12 loads total instead of ~13 per weight): slice 00 for every weight (full
        ablation -> ``dce``, then CE_f / CE_¬f for the top ``exact_top`` candidates by proxy
        score -- ``exact_top <= 0`` means every gated candidate, which reproduces
        ``score_weight`` exactly), slice 01 for every weight with an f*, then slices 02..11
        once each for those weights (robust scores + coverage-matched nulls).

        Variants are grouped by layer and packed up to ``self.batch`` per ``ablate`` call.
        """
        wids = [int(w) for w in wids]
        recs: Dict[int, Dict[str, Any]] = {}
        for w in wids:
            recs[w] = {"wid": w, "status": "scored", "dce": None, "spec": None, "py": None,
                       "desc": None, "A": None, "B": None, "robust": None,
                       "select_obj": None, "cov_penalty": self.cov_penalty}
        t0 = time.time()
        n = len(wids)
        empty_robust = {"scores": [], "mean": None, "sem": None, "null_mean": None,
                        "null_sd": None, "pass": False}
        stages = 2 + (len(ROBUST_SLICES) if want_robust else 0)

        # ---- slice 00: full ablation for every weight -> dce --------------------------
        self.load(0)
        ces = self._ce_many([(w, self._zeros()) for w in wids])
        live: List[int] = []
        for w, ce in zip(wids, ces):
            recs[w]["dce"] = ce - self.ce0_mean
            if abs(recs[w]["dce"]) >= NOISE_FLOOR:
                live.append(w)
            else:
                # a CE-noise-floor weight has no meaningful rec/inv: property of the weight
                recs[w]["status"] = "noise_floor"
                recs[w]["robust"] = dict(empty_robust)
        self._log_progress("slice00 full ablation", n, n, t0, 1, stages)

        # ---- slice 00: candidate masks + exact CE_f / CE_¬f ---------------------------
        prep: Dict[int, Dict[str, Any]] = {}
        for w in live:
            cands: List[Dict[str, Any]] = []
            masks: List[torch.Tensor] = []
            covs: List[float] = []
            pys: List[Optional[str]] = []
            for c in (cands_by_wid.get(w) or []):
                if c.get("spec") is None:
                    continue
                try:
                    validate_spec(c["spec"])
                except ValueError as exc:
                    print(f"[score] wid={w}: skipping invalid candidate spec ({exc})",
                          flush=True)
                    continue
                m = self.mask(c["spec"])
                cands.append(c)
                masks.append(m)
                covs.append(float(m.double().mean()))
                pys.append(c.get("py"))
            keep = [i for i, cov in enumerate(covs) if (not gate) or (0.0 < cov <= COVERAGE_MAX)]
            if not cands or not keep:
                recs[w]["status"] = "no_candidate"
                continue

            def _proxy(i: int, _cands: List[Dict[str, Any]] = cands) -> float:
                return self._cand_obj(_cands[i])

            order = sorted(keep, key=lambda i: (-_proxy(i), i))
            chosen = order if (exact_top is None or int(exact_top) <= 0) else order[:int(exact_top)]
            prep[w] = {"cands": cands, "masks": masks, "covs": covs, "pys": pys,
                       "chosen": sorted(chosen)}

        items: List[Tuple[int, torch.Tensor]] = []
        owner: List[Tuple[int, int, bool]] = []
        for w in live:
            p = prep.get(w)
            if p is None:
                continue
            for i in p["chosen"]:
                items.append((w, p["masks"][i]))
                owner.append((w, i, False))
                items.append((w, ~p["masks"][i]))
                owner.append((w, i, True))
        ces = self._ce_many(items) if items else []
        ce_f: Dict[Tuple[int, int], float] = {}
        ce_nf: Dict[Tuple[int, int], float] = {}
        for (w, i, neg), ce in zip(owner, ces):
            (ce_nf if neg else ce_f)[(w, i)] = ce

        have_f: List[int] = []
        for w in live:
            p = prep.get(w)
            if p is None:
                continue
            dce = recs[w]["dce"]
            best: Optional[Tuple[int, Dict[str, Any], float]] = None
            best_score = -math.inf
            for i in p["chosen"]:
                a = {"rec": _ratio(ce_f[(w, i)] - self.ce0_mean, dce),
                     "inv": _ratio(ce_nf[(w, i)] - self.ce0_mean, dce),
                     "cov": p["covs"][i]}
                a["score"] = (None if (a["rec"] is None or a["inv"] is None)
                              else min(a["rec"], 1.0 - a["inv"]))
                obj = self.select_obj(a)
                if obj > best_score:
                    best_score = obj
                    best = (i, a, obj)
            if best is None:
                recs[w]["status"] = "no_candidate"
                continue
            i, A, select_obj = best
            recs[w]["select_obj"] = select_obj
            spec = p["cands"][i]["spec"]
            py, mismatch = self.resolve_py(spec, p["pys"][i])
            recs[w]["spec"] = spec
            recs[w]["py"] = py
            recs[w]["desc"] = p["cands"][i].get("desc")
            recs[w]["A"] = A
            if mismatch is not None and self.verbose:
                print(f"[score] wid={w}: provided py disagreed with spec "
                      f"({mismatch['mismatches']}/{mismatch['checked']}); re-emitted from spec")
            have_f.append(w)
        self._log_progress("slice00 candidate variants", n, n, t0, 1, stages)

        # ---- slice 01: InterpB for every weight with an f* ----------------------------
        if have_f:
            self.load(SLICE_INTERPB)
            masks01: Dict[int, Tuple[torch.Tensor, float]] = {}
            items = []
            owner = []
            z = self._zeros()
            for w in have_f:
                m = self.mask(recs[w]["spec"])
                masks01[w] = (m, float(m.double().mean()))
                items.append((w, z))
                owner.append((w, "full"))
                items.append((w, m))
                owner.append((w, "f"))
                items.append((w, ~m))
                owner.append((w, "nf"))
            acc: Dict[int, Dict[str, float]] = {w: {} for w in have_f}
            for (w, tag), ce in zip(owner, self._ce_many(items)):
                acc[w][tag] = ce
            for w in have_f:
                dce = acc[w]["full"] - self.ce0_mean
                rec = _ratio(acc[w]["f"] - self.ce0_mean, dce)
                inv = _ratio(acc[w]["nf"] - self.ce0_mean, dce)
                recs[w]["B"] = {"rec": rec, "inv": inv, "cov": masks01[w][1],
                                "score": None if (rec is None or inv is None)
                                else min(rec, 1.0 - inv)}
            self._log_progress("slice01 InterpB", len(have_f), n, t0, 2, stages)

        # ---- slices 02..11: robust scores + coverage-matched nulls ---------------------
        if want_robust and have_f:
            scores: Dict[int, List[float]] = {w: [] for w in have_f}
            nulls: Dict[int, List[float]] = {w: [] for w in have_f}
            for ki, k in enumerate(ROBUST_SLICES):
                self.load(k)
                items = []
                owner = []
                z = self._zeros()
                for w in have_f:
                    m = self.mask(recs[w]["spec"])
                    nm = self.null_mask(w, k, int(m.sum()))
                    for tag, mask in (("full", z), ("f", m), ("nf", ~m),
                                      ("nul", nm), ("nnul", ~nm)):
                        items.append((w, mask))
                        owner.append((w, tag))
                acc = {w: {} for w in have_f}
                for (w, tag), ce in zip(owner, self._ce_many(items)):
                    acc[w][tag] = ce
                for w in have_f:
                    dce = acc[w]["full"] - self.ce0_mean
                    rec = _ratio(acc[w]["f"] - self.ce0_mean, dce)
                    inv = _ratio(acc[w]["nf"] - self.ce0_mean, dce)
                    nrec = _ratio(acc[w]["nul"] - self.ce0_mean, dce)
                    ninv = _ratio(acc[w]["nnul"] - self.ce0_mean, dce)
                    scores[w].append(min(rec, 1.0 - inv) if None not in (rec, inv)
                                     else float("nan"))
                    nulls[w].append(min(nrec, 1.0 - ninv) if None not in (nrec, ninv)
                                    else float("nan"))
                self._log_progress(f"robust slice {k}", len(have_f), n, t0,
                                   3 + ki, stages)
            for w in have_f:
                ms, mn = _mean(scores[w]), _mean(nulls[w])
                ss, sn = _sem(scores[w]), _sd(nulls[w])
                ok = all(not math.isnan(v) for v in scores[w] + nulls[w])
                passed = bool(ok and ms is not None and mn is not None and ss is not None
                              and sn is not None and ms >= ROBUST_T
                              and (ms - 2 * ss) > (mn + 2 * sn))
                recs[w]["robust"] = {"scores": scores[w], "mean": ms, "sem": ss,
                                     "null_mean": mn, "null_sd": sn, "pass": passed}
        elif have_f:
            for w in have_f:
                recs[w]["robust"] = dict(empty_robust)

        # leave slice 00 loaded, like score_weight
        self.load(0)
        if self.verbose:
            print(f"[score_many] {n} weights in {time.time() - t0:.1f}s "
                  f"({n / max(time.time() - t0, 1e-9):.2f} weights/s): "
                  f"{sum(1 for r in recs.values() if r['status'] == 'scored')} scored, "
                  f"{sum(1 for r in recs.values() if r['status'] == 'noise_floor')} noise_floor, "
                  f"{sum(1 for r in recs.values() if r['status'] == 'no_candidate')} no_candidate",
                  flush=True)
        return recs


# --------------------------------------------------------------------------------------
# shard / file drivers
# --------------------------------------------------------------------------------------


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_shard(scorer: Scorer, shard: str, gate: bool = True, robust: bool = True,
              exact_top: int = 0, cand_dir: Optional[Path] = None,
              out_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Score one candidate shard with the slice-outer batch mode."""
    cand_path = (cand_dir or scorer.runs / "candidates") / f"{shard}.jsonl"
    out_path = (out_dir or scorer.runs / "scores") / f"{shard}.jsonl"
    entries = load_candidates(cand_path)
    t0 = time.time()
    wids = [int(e["wid"]) for e in entries]
    cands_by_wid = {int(e["wid"]): (e.get("cands") or []) for e in entries}
    recs = scorer.score_many(wids, cands_by_wid, gate=gate, want_robust=robust,
                             exact_top=exact_top)
    records = [recs[w] for w in wids]
    _write_jsonl(out_path, records)
    dt = time.time() - t0
    n_scored = sum(1 for r in records if r["status"] == "scored")
    print(f"[score] {shard}: {len(records)} weights ({n_scored} scored) in {dt:.1f}s "
          f"({len(records) / max(dt, 1e-9):.2f} weights/s) -> {out_path}", flush=True)
    return {"shard": shard, "n": len(records), "scored": n_scored, "wall_s": dt}


def home_shards(model: str, wids: Sequence[int]) -> Dict[int, str]:
    """wid -> home profile shard name (contract ordering: layer, matrix, nonzero order)."""
    from .profile import shard_map
    from .weights import load_weights
    smap = shard_map(load_weights(model))
    out: Dict[int, str] = {}
    for shard, wl in smap.items():
        for w in wl:
            out[w] = shard
    missing = [int(w) for w in wids if int(w) not in out]
    if missing:
        raise KeyError(f"wids outside the weight arrays: {missing[:8]}")
    return {int(w): out[int(w)] for w in wids}


def run_wids(scorer: Scorer, wids: Sequence[int], out_dir: Path, cand_dirs: Sequence[Path],
             shards: Optional[Sequence[str]] = None, gate: bool = True, robust: bool = True,
             exact_top: int = 0, force: bool = False) -> Dict[str, Any]:
    """Score an explicit wid list; write one ``<shard>.jsonl`` per home shard in ``out_dir``.

    ``cand_dirs`` is searched in order for ``<shard>.jsonl`` (first hit wins).
    """
    wids = [int(w) for w in wids]
    home = home_shards(scorer.model, wids)
    if shards:
        want = set(shards)
        wids = [w for w in wids if home[w] in want]
        if not wids:
            print(f"[score] no requested wids in shards {sorted(want)}", file=sys.stderr)
            return {"n": 0, "shards": [], "wall_s": 0.0}
    by_shard: Dict[str, List[int]] = {}
    for w in wids:
        by_shard.setdefault(home[w], []).append(w)
    todo = {s: wl for s, wl in sorted(by_shard.items())
            if force or not (out_dir / f"{s}.jsonl").exists()}
    if not todo:
        print(f"[score] nothing to do (all {len(by_shard)} shard outputs exist; --force to redo)",
              flush=True)
        return {"n": 0, "shards": [], "wall_s": 0.0}
    skipped = len(by_shard) - len(todo)
    if skipped:
        print(f"[score] skip {skipped} shard(s) with existing output in {out_dir} "
              f"(--force to redo)", flush=True)

    cands_by_wid: Dict[int, List[Dict[str, Any]]] = {}
    missing_shards: List[str] = []
    for s in todo:
        path = next((d / f"{s}.jsonl" for d in cand_dirs if (d / f"{s}.jsonl").exists()), None)
        if path is None:
            missing_shards.append(s)
            for w in todo[s]:
                cands_by_wid[w] = []
            continue
        for e in load_candidates(path):
            cands_by_wid[int(e["wid"])] = e.get("cands") or []
    if missing_shards:
        print(f"[score] no candidate file for {len(missing_shards)} shard(s) "
              f"({', '.join(missing_shards[:6])}{'...' if len(missing_shards) > 6 else ''}) "
              f"in {[str(d) for d in cand_dirs]}: those weights get status no_candidate",
              file=sys.stderr, flush=True)

    todo_wids = [w for s in sorted(todo) for w in todo[s]]
    t0 = time.time()
    recs = scorer.score_many(todo_wids, cands_by_wid, gate=gate, want_robust=robust,
                             exact_top=exact_top)
    for s, wl in sorted(todo.items()):
        _write_jsonl(out_dir / f"{s}.jsonl", [recs[w] for w in wl])
    dt = time.time() - t0
    n_scored = sum(1 for r in recs.values() if r["status"] == "scored")
    print(f"[score] {len(todo_wids)} weights over {len(todo)} shards ({n_scored} scored) "
          f"in {dt:.1f}s ({len(todo_wids) / max(dt, 1e-9):.2f} weights/s) -> {out_dir}",
          flush=True)
    return {"n": len(todo_wids), "shards": sorted(todo), "scored": n_scored, "wall_s": dt,
            "out_dir": str(out_dir)}


def run_from_specs(scorer: Scorer, specs_path: Path, out_path: Path, gate: bool = False,
                   robust: bool = True) -> Dict[str, Any]:
    entries = load_candidates(specs_path)
    scorer.load(0)
    records = []
    t0 = time.time()
    for n, e in enumerate(entries):
        wid = int(e["wid"])
        cand = {"spec": e["spec"], "py": e.get("py"), "desc": e.get("desc")}
        t1 = time.time()
        rec = scorer.score_weight(wid, [cand], gate=gate, want_robust=robust)
        rec["wall_s"] = time.time() - t1
        records.append(rec)
        a, b, rb = rec.get("A") or {}, rec.get("B") or {}, rec.get("robust") or {}
        print(f"[score] spec {n + 1}/{len(entries)} wid={wid} L{scorer.layer_of(wid)} "
              f"status={rec['status']} dce={rec['dce']:.4e} "
              f"A(rec={a.get('rec')},inv={a.get('inv')},cov={a.get('cov')},score={a.get('score')}) "
              f"B(score={b.get('score')}) robust(mean={rb.get('mean')},pass={rb.get('pass')}) "
              f"{rec['wall_s']:.1f}s", flush=True)
    _write_jsonl(out_path, records)
    print(f"[score] wrote {len(records)} records -> {out_path} ({time.time() - t0:.1f}s)", flush=True)
    return {"n": len(records), "wall_s": time.time() - t0, "out": str(out_path)}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="swa.score")
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--shards", default=None,
                    help="comma-separated shard names (default: every runs/<model>/candidates/*.jsonl)")
    ap.add_argument("--wids-file", default=None,
                    help=".npy of explicit wids; restricts scoring to those weights and writes to "
                         "runs/<model>/scores_<tag>/<shard>.jsonl (never a full shard output)")
    ap.add_argument("--tag", default="pilot",
                    help="output dir suffix with --wids-file: runs/<model>/scores_<tag>")
    ap.add_argument("--cov-penalty", type=float, default=COVERAGE_PENALTY,
                    help="select f* among the exactly-scored slice-00 candidates by "
                         "A.score - MU*A.cov (default 0.5; 0 reproduces the paper's plain "
                         "argmax); recorded per record as select_obj")
    ap.add_argument("--exact-top", type=int, default=0,
                    help="exactly score the top-N candidates by proxy score (0 = every gated "
                         "candidate, i.e. identical to score_weight)")
    ap.add_argument("--candidates-dir", default=None,
                    help="comma-separated dirs searched for <shard>.jsonl "
                         "(default: runs/<model>/candidates)")
    ap.add_argument("--from-specs", default=None, help="JSONL of {wid, spec[, py, desc]}")
    ap.add_argument("--out", default=None, help="output jsonl for --from-specs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=8, help="variants per Replayer.ablate call")
    ap.add_argument("--runs", default="runs",
                    help="root of pipeline outputs (candidates/, scores/, vocab.json, weights.npz); "
                         "model + corpus slice paths are resolved by swa.replay")
    ap.add_argument("--force", action="store_true", help="re-score shards whose output exists")
    ap.add_argument("--no-robust", action="store_true", help="skip slices 02..11 (slice 00/01 only)")
    ap.add_argument("--gate", action="store_true",
                    help="apply the 0<cov<=0.5 candidate gate in --from-specs mode")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    runs = Path(args.runs) / args.model
    cand_dirs = ([Path(x) for x in args.candidates_dir.split(",") if x.strip()]
                 if args.candidates_dir else [runs / "candidates"])
    if args.from_specs:
        scorer = Scorer(args.model, device=args.device, batch=args.batch, runs_dir=args.runs,
                        verbose=not args.quiet, cov_penalty=args.cov_penalty)
        out = Path(args.out) if args.out else runs / "scores" / "from_specs.jsonl"
        run_from_specs(scorer, Path(args.from_specs), out, gate=args.gate,
                       robust=not args.no_robust)
        return 0

    if args.wids_file:
        wids = np.load(args.wids_file).astype(np.int64).reshape(-1).tolist()
        if not wids:
            print(f"[score] no wids in {args.wids_file}", file=sys.stderr)
            return 1
        shards = ([s.strip() for s in args.shards.split(",") if s.strip()]
                  if args.shards else None)
        out_dir = runs / f"scores_{args.tag}"
        for d in cand_dirs:
            if not d.is_dir():
                print(f"[score] warning: candidates dir {d} does not exist", file=sys.stderr)
        scorer = Scorer(args.model, device=args.device, batch=args.batch, runs_dir=args.runs,
                        verbose=not args.quiet, cov_penalty=args.cov_penalty)
        run_wids(scorer, wids, out_dir, cand_dirs, shards=shards, gate=True,
                 robust=not args.no_robust, exact_top=args.exact_top, force=args.force)
        return 0

    cdir = cand_dirs[0]
    if not cdir.is_dir():
        print(f"[score] no candidates dir at {cdir}", file=sys.stderr)
        return 1
    if args.shards:
        shards = [s.strip() for s in args.shards.split(",") if s.strip()]
    else:
        shards = sorted(p.stem for p in cdir.glob("*.jsonl"))
    if not shards:
        print(f"[score] no candidate shards in {cdir}", file=sys.stderr)
        return 1

    scorer = None
    todo = []
    for shard in shards:
        out_path = runs / "scores" / f"{shard}.jsonl"
        if out_path.exists() and not args.force:
            print(f"[score] skip {shard} (output exists; --force to redo)", flush=True)
            continue
        todo.append(shard)
    if not todo:
        print("[score] nothing to do", flush=True)
        return 0
    scorer = Scorer(args.model, device=args.device, batch=args.batch, runs_dir=args.runs,
                    verbose=not args.quiet, cov_penalty=args.cov_penalty)
    for shard in todo:
        run_shard(scorer, shard, gate=True, robust=not args.no_robust,
                  exact_top=args.exact_top, cand_dir=cdir, out_dir=runs / "scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
