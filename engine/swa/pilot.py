"""Paper-comparison pilot: sample N nonzero MLP weights, then profile -> synth -> score ->
report (and export) them, to compare against the paper's Table-1 rates before a full run.

Usage
-----
    PYTHONPATH=engine .venv/bin/python -m swa.pilot --model csp_yolo1 --n 1024 --seed 0
    PYTHONPATH=engine .venv/bin/python -m swa.pilot --model csp_yolo1 --n 64 --seed 123 \
        --stage all --tag pilot_smoke

Stages (``--stage sample|profile|synth|score|export|all``; comma-separate to pick several).
Every artifact lands under ``runs/<model>/<tag>/`` and ``runs/<model>/scores_<tag>/``:

sample
    Sample ``--n`` wids uniformly at random over ALL nonzero MLP weights (the paper's
    "sample nonzero weights uniformly across MLP layers") -> ``runs/<model>/<tag>/wids.npy``
    (sorted int64; ``--seed`` makes it reproducible).

profile
    Profile the sampled wids.  Wids whose home shard already holds a COMPLETE profile
    (``meta.json`` ``complete: true``, e.g. ``L00_fc_000``) are read from it and never
    re-profiled.  The rest go through ``swa.profile --wids-file``, which writes a PARTIAL
    profile (``complete: false``) holding the union of the sampled wids and whatever wids
    that shard already had -- so profiles only ever grow and a partial run can never
    overwrite a complete shard (``profile.run_shard`` skips those).

synth
    ``swa-synth --shards <non-complete home shards> --force``: regenerates the candidate
    file for each partial shard so it matches the freshly written profile (only the profiled
    wids get candidates).  Those files are then MOVED to ``runs/<model>/<tag>/candidates/``
    so a partial candidate file never sits in ``runs/<model>/candidates/`` and can never be
    mistaken for a complete shard's candidates by a later full run.

score
    ``swa.score --wids-file runs/<model>/<tag>/wids.npy --tag <tag>
    --candidates-dir runs/<model>/<tag>/candidates,runs/<model>/candidates``
    -> ``runs/<model>/scores_<tag>/<shard>.jsonl``.  The pilot passes ``--exact-top 0`` by
    default (every gated candidate scored exactly, i.e. the ``score_weight`` protocol): the
    paper scores all of its candidates exactly, and on the 5-weight L00_fc_000 equality check
    the exact argmax sits at proxy rank 3, 1, 1, 1, 7 -- so the score CLI's default
    ``--exact-top 4`` would change f* for some weights (it only saves ~10% of the run).

export
    Writes ``runs/<model>/<tag>/report.json`` (counts by status, InterpA/InterpB/robust
    rates with binomial SEM, per-layer and per-|ΔCE| tables, paper Table-1 numbers), then
    runs ``swa.export_site`` and ``swa.export_modelpy`` against ``scores_<tag>`` into
    ``site/pilot/<tag>/`` (the canonical ``site/data`` is left untouched).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model import ROOT
from .score import COVERAGE_MAX, NOISE_FLOOR, ROBUST_T, home_shards
from .weights import load_weights

STAGES = ("sample", "profile", "synth", "score", "export")
BIN_EDGES = (1e-6, 1e-5, 1e-4)
BIN_LABELS = ("<1e-6", "1e-6..1e-5", "1e-5..1e-4", ">=1e-4")
PAPER = {"interpA": 0.373, "interpB": 0.353, "robust": 0.15, "T": 0.75, "c": 0.5,
         "source": "Gao et al. 2025, Table 1 (csp_yolo1; InterpA 37.3 +/- 3.9%, "
                   "InterpB 35.3 +/- 3.9%, Robust 15.0 +/- 1.7%)"}
VERDICTS = ("scored", "noise_floor", "no_candidate")


# --------------------------------------------------------------------- paths / plumbing


def runs_root() -> Path:
    return ROOT / "runs"


def pilot_dir(model: str, tag: str) -> Path:
    return runs_root() / model / tag


def scores_dir(model: str, tag: str) -> Path:
    return runs_root() / model / f"scores_{tag}"


def _py() -> str:
    return sys.executable or "python"


def _run(cmd: Sequence[Any], label: str) -> float:
    cmd = [str(c) for c in cmd]
    print(f"[pilot] $ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    pp = str(ROOT / "engine")
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    dt = time.time() - t0
    if rc != 0:
        raise SystemExit(f"[pilot] {label} failed with exit code {rc}: {' '.join(cmd)}")
    print(f"[pilot] {label} done in {dt:.1f}s", flush=True)
    return dt


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _record_stage(model: str, tag: str, stage: str, wall: float, **extra: Any) -> None:
    path = pilot_dir(model, tag) / "stages.json"
    obj = _read_json(path)
    obj.setdefault("stages", {})[stage] = {"wall_s": wall, **extra}
    obj["model"] = model
    obj["tag"] = tag
    obj["updated"] = time.time()
    _write_json(path, obj)


# --------------------------------------------------------------------- profile helpers


def _profile_state(shard_dir: Path) -> Tuple[Optional[List[int]], bool]:
    """(wids already in the profile or None, complete?) for one profile shard dir."""
    wid_path = shard_dir / "wid.npy"
    wids = np.load(wid_path).astype(np.int64).tolist() if wid_path.exists() else None
    meta = _read_json(shard_dir / "meta.json")
    return wids, bool(meta.get("complete"))


def _group_by_shard(model: str, wids: Sequence[int]) -> Dict[str, List[int]]:
    home = home_shards(model, [int(w) for w in wids])
    out: Dict[str, List[int]] = {}
    for w in wids:
        out.setdefault(home[int(w)], []).append(int(w))
    return out


# --------------------------------------------------------------------- stages


def stage_sample(model: str, tag: str, n: int, seed: int) -> Dict[str, Any]:
    w = load_weights(model)
    W = int(len(w["layer"]))
    k = int(min(int(n), W))
    rng = np.random.default_rng(int(seed))
    wids = np.sort(rng.choice(W, size=k, replace=False)).astype(np.int64)
    pdir = pilot_dir(model, tag)
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / "wids.npy", wids)
    print(f"[pilot] sampled {k}/{W} nonzero MLP weights (seed {seed}) -> {pdir / 'wids.npy'}",
          flush=True)
    return {"n": k, "W": W, "seed": int(seed), "wids_path": str(pdir / "wids.npy")}


def stage_profile(model: str, tag: str) -> Dict[str, Any]:
    pdir = pilot_dir(model, tag)
    wids = np.load(pdir / "wids.npy").astype(np.int64).tolist()
    by_shard = _group_by_shard(model, wids)
    prof_root = runs_root() / model / "profiles"
    todo: Dict[str, List[int]] = {}
    complete_shards: List[str] = []
    for shard in sorted(by_shard):
        existing, complete = _profile_state(prof_root / shard)
        if complete:
            have = set(existing or [])
            missing = [w for w in by_shard[shard] if w not in have]
            if missing:
                raise SystemExit(f"[pilot] complete profile {shard} is missing sampled wids "
                                 f"{missing[:8]} -- profile/weights out of sync")
            complete_shards.append(shard)
            continue
        todo[shard] = sorted(set(existing or []) | set(by_shard[shard]))
    print(f"[pilot] {len(by_shard)} home shards: {len(complete_shards)} already complete "
          f"(read from the existing profile), {len(todo)} to profile "
          f"({sum(len(v) for v in todo.values())} wids incl. previously partial ones)",
          flush=True)
    if not todo:
        return {"shards_profiled": [], "shards_complete": complete_shards, "n_wids": 0}
    all_wids = np.asarray(sorted({w for wl in todo.values() for w in wl}), dtype=np.int64)
    wf = pdir / "profile_wids.npy"
    np.save(wf, all_wids)
    wall = _run([_py(), "-m", "swa.profile", "--model", model, "--wids-file", wf],
                "swa.profile")
    return {"shards_profiled": sorted(todo), "shards_complete": complete_shards,
            "n_wids": int(all_wids.size), "wall_s": wall}


def _partial_shards(model: str, by_shard: Iterable[str]) -> List[str]:
    prof_root = runs_root() / model / "profiles"
    out = []
    for shard in sorted(by_shard):
        _, complete = _profile_state(prof_root / shard)
        if not complete:
            out.append(shard)
    return out


def stage_synth(model: str, tag: str, force: bool = True) -> Dict[str, Any]:
    pdir = pilot_dir(model, tag)
    wids = np.load(pdir / "wids.npy").astype(np.int64).tolist()
    by_shard = _group_by_shard(model, wids)
    partial = _partial_shards(model, by_shard)
    unprofiled = [s for s in sorted(by_shard)
                  if not (runs_root() / model / "profiles" / s / "wid.npy").exists()]
    if unprofiled:
        raise SystemExit(f"[pilot] {len(unprofiled)} home shard(s) have no profile at all "
                         f"({', '.join(unprofiled[:6])}); run --stage profile first")
    complete = [s for s in sorted(by_shard) if s not in set(partial)]
    # complete profiles whose canonical candidate file is missing still need a synth run
    missing_cand = [s for s in complete
                    if not (runs_root() / model / "candidates" / f"{s}.jsonl").exists()]
    binary = ROOT / "synth" / "target" / "release" / "swa-synth"
    if partial or missing_cand:
        if not binary.exists():
            raise SystemExit(f"[pilot] missing synth binary {binary} (cargo build --release)")
    wall = 0.0
    if partial:
        cmd: List[Any] = [binary, "--model", model, "--root", ROOT,
                          "--shards", ",".join(partial)]
        if force:
            cmd.append("--force")
        wall += _run(cmd, "swa-synth (partial shards)")
    if missing_cand:
        # no --force: a complete shard's candidates must be generated canonically and stay there
        wall += _run([binary, "--model", model, "--root", ROOT,
                      "--shards", ",".join(missing_cand)], "swa-synth (complete, no candidates)")
    cand_dir = pdir / "candidates"
    moved: List[str] = []
    for shard in partial:
        src = runs_root() / model / "candidates" / f"{shard}.jsonl"
        if src.exists():
            cand_dir.mkdir(parents=True, exist_ok=True)
            os.replace(src, cand_dir / f"{shard}.jsonl")
            moved.append(shard)
    if partial:
        print(f"[pilot] moved {len(moved)} partial candidate file(s) to {cand_dir} so they can "
              f"never masquerade as a complete shard's candidates", flush=True)
    if not partial and not missing_cand:
        print("[pilot] nothing to synthesise: every sampled wid has a complete profile with "
              "candidates", flush=True)
    return {"shards": partial, "complete_missing_candidates": missing_cand,
            "moved": moved, "wall_s": wall}


def stage_score(model: str, tag: str, exact_top: int = 4, robust: bool = True,
                force: bool = False) -> Dict[str, Any]:
    pdir = pilot_dir(model, tag)
    wids_file = pdir / "wids.npy"
    if not wids_file.exists():
        raise SystemExit(f"[pilot] no {wids_file} (run --stage sample first)")
    cand_dirs = f"{pdir / 'candidates'},{runs_root() / model / 'candidates'}"
    cmd: List[Any] = [_py(), "-m", "swa.score", "--model", model,
                      "--runs", runs_root(), "--wids-file", wids_file, "--tag", tag,
                      "--candidates-dir", cand_dirs, "--exact-top", int(exact_top)]
    if not robust:
        cmd.append("--no-robust")
    if force:
        cmd.append("--force")
    wall = _run(cmd, "swa.score")
    return {"wall_s": wall, "scores_dir": str(scores_dir(model, tag))}


def _load_score_records(model: str, tag: str) -> Dict[int, Dict[str, Any]]:
    sdir = scores_dir(model, tag)
    recs: Dict[int, Dict[str, Any]] = {}
    if not sdir.is_dir():
        return recs
    for path in sorted(sdir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    recs[int(r["wid"])] = r
    return recs


def _rate(sel: Sequence[Dict[str, Any]], pred) -> Dict[str, Any]:
    n = len(sel)
    k = sum(1 for r in sel if pred(r))
    p = (k / n) if n else None
    sem = math.sqrt(p * (1.0 - p) / n) if n and p is not None else None
    return {"k": k, "n": n, "rate": p, "sem": sem}


def _interp(rec) -> bool:
    if not rec:
        return False
    s, cov = rec.get("score"), rec.get("cov")
    return bool(s is not None and cov is not None and s >= ROBUST_T and 0.0 < cov <= COVERAGE_MAX)


def _robust_pass(r: Dict[str, Any]) -> bool:
    rob = r.get("robust") or {}
    return bool(rob.get("pass"))


def _bin_of(dce: Optional[float]) -> Optional[int]:
    if dce is None:
        return None
    a = abs(dce)
    for i, e in enumerate(BIN_EDGES):
        if a < e:
            return i
    return len(BIN_EDGES)


def build_report(model: str, tag: str, seed: Optional[int], n_requested: int,
                 exact_top: int) -> Dict[str, Any]:
    recs = _load_score_records(model, tag)
    layers = load_weights(model)["layer"]
    rows = [recs[w] for w in sorted(recs)]
    for r in rows:
        r["_layer"] = int(layers[int(r["wid"])])
    verdicts = [r for r in rows if r.get("status") in VERDICTS]
    nonzero = [r for r in verdicts
               if r.get("dce") is not None and abs(r["dce"]) >= NOISE_FLOOR]

    counts = {s: sum(1 for r in rows if r.get("status") == s) for s in
              ("scored", "noise_floor", "no_candidate")}
    counts["unscored"] = sum(1 for r in rows if r.get("status") not in VERDICTS)
    counts["total"] = len(rows)

    def rates(sel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "interpA": _rate(sel, lambda r: _interp(r.get("A"))),
            "interpB": _rate(sel, lambda r: _interp(r.get("B"))),
            "robust": _rate(sel, _robust_pass),
        }

    per_layer = []
    for L in sorted({r["_layer"] for r in rows}):
        sel = [r for r in verdicts if r["_layer"] == L]
        per_layer.append({"layer": L, "n": len(sel),
                          **{s: sum(1 for r in sel if r.get("status") == s) for s in VERDICTS},
                          **rates(sel)})
    per_bin = []
    for i, label in enumerate(BIN_LABELS):
        sel = [r for r in rows if _bin_of(r.get("dce")) == i]
        per_bin.append({"bin": label, "n": len(sel),
                        **rates(sel)})

    stages = _read_json(pilot_dir(model, tag) / "stages.json").get("stages", {})
    report = {
        "model": model,
        "tag": tag,
        "seed": seed,
        "n_requested": int(n_requested),
        "n_records": len(rows),
        "exact_top": int(exact_top),
        "scores_dir": str(scores_dir(model, tag)),
        "counts": counts,
        "rates": {
            "over_verdicts": rates(verdicts),
            "over_nonzero_dce": rates(nonzero),
            "over_scored_only": rates([r for r in rows if r.get("status") == "scored"]),
        },
        "paper": PAPER,
        "table1": {
            "ours": {
                "interpA": rates(verdicts)["interpA"]["rate"],
                "interpB": rates(verdicts)["interpB"]["rate"],
                "robust": rates(verdicts)["robust"]["rate"],
            },
            "paper": {"interpA": PAPER["interpA"], "interpB": PAPER["interpB"],
                      "robust": PAPER["robust"]},
            "note": "ours: rates over all sampled weights with a verdict "
                    "(scored+noise_floor+no_candidate); paper: Gao et al. 2025 Table 1",
        },
        "per_layer": per_layer,
        "per_bin": per_bin,
        "wall_s": {k: v.get("wall_s") for k, v in stages.items()},
        "generated": time.time(),
    }
    return report


def stage_export(model: str, tag: str, seed: Optional[int], n_requested: int,
                 exact_top: int, site_out: Optional[Path] = None) -> Dict[str, Any]:
    pdir = pilot_dir(model, tag)
    stages = _read_json(pdir / "stages.json").get("stages", {})
    if (pdir / "wids.npy").exists():
        n_requested = int(np.load(pdir / "wids.npy").size)
    if stages.get("sample", {}).get("seed") is not None:
        seed = stages["sample"]["seed"]
    t0 = time.time()
    report = build_report(model, tag, seed, n_requested, exact_top)
    _write_json(pdir / "report.json", report)
    if not report["n_records"]:
        print(f"[pilot] WARNING: no score records in {scores_dir(model, tag)} "
              f"-- run --stage score first", file=sys.stderr, flush=True)
    out = Path(site_out) if site_out else ROOT / "site" / "pilot" / tag
    wall_report = time.time() - t0
    wall_site = _run([_py(), "-m", "swa.export_site", "--model", model,
                      "--runs", runs_root(), "--scores-dir", scores_dir(model, tag),
                      "--out", out], "swa.export_site")
    wall_py = _run([_py(), "-m", "swa.export_modelpy", "--model", model,
                    "--runs", runs_root(), "--scores-dir", scores_dir(model, tag),
                    "--out", out], "swa.export_modelpy")
    print(f"[pilot] report.json: {report['counts']} | "
          f"InterpA {report['table1']['ours']['interpA']} vs paper {PAPER['interpA']} | "
          f"InterpB {report['table1']['ours']['interpB']} vs paper {PAPER['interpB']} | "
          f"robust {report['table1']['ours']['robust']} vs paper {PAPER['robust']}", flush=True)
    return {"report": str(pdir / "report.json"), "site_out": str(out),
            "wall_s": wall_report + wall_site + wall_py,
            "counts": report["counts"], "table1": report["table1"]}


# --------------------------------------------------------------------- main


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="swa.pilot", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--n", type=int, default=1024, help="weights to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    help="sample|profile|synth|score|export|all (comma-separated allowed)")
    ap.add_argument("--tag", default="pilot",
                    help="artifact dir runs/<model>/<tag> and runs/<model>/scores_<tag>")
    ap.add_argument("--exact-top", type=int, default=0,
                    help="exactly score the top-N candidates by proxy score on slice 00 "
                         "(0 = every gated candidate, default).  The proxy ranking is not the "
                         "exact ranking: on the 5-weight L00_fc_000 check the exact argmax sits "
                         "at proxy rank 3,1,1,1,7, so --exact-top 4 changes f* for some weights "
                         "and breaks equality with score_weight.  N>0 is a ~10%% speed knob.")
    ap.add_argument("--no-robust", action="store_true", help="skip slices 02..11")
    ap.add_argument("--force", action="store_true",
                    help="re-score shards whose scores_<tag>/<shard>.jsonl exists")
    ap.add_argument("--site-out", default=None,
                    help="export_site/export_modelpy out root (default site/pilot/<tag>)")
    args = ap.parse_args(argv)

    stages = [s.strip() for s in args.stage.split(",") if s.strip()]
    if "all" in stages:
        stages = list(STAGES)
    bad = [s for s in stages if s not in STAGES]
    if bad:
        print(f"[pilot] unknown stage(s) {bad}; expected {STAGES} or all", file=sys.stderr)
        return 2

    pdir = pilot_dir(args.model, args.tag)
    seed: Optional[int] = args.seed
    n_requested = int(args.n)
    t_all = time.time()
    for stage in stages:
        t0 = time.time()
        if stage == "sample":
            info = stage_sample(args.model, args.tag, args.n, args.seed)
            n_requested = info["n"]
            _record_stage(args.model, args.tag, stage, time.time() - t0, **info)
        elif stage == "profile":
            info = stage_profile(args.model, args.tag)
            _record_stage(args.model, args.tag, stage, time.time() - t0, **info)
        elif stage == "synth":
            info = stage_synth(args.model, args.tag)
            _record_stage(args.model, args.tag, stage, time.time() - t0,
                          shards=info.get("shards"),
                          complete_missing_candidates=info.get("complete_missing_candidates"),
                          moved=info.get("moved"))
        elif stage == "score":
            info = stage_score(args.model, args.tag, exact_top=args.exact_top,
                               robust=not args.no_robust, force=args.force)
            _record_stage(args.model, args.tag, stage, time.time() - t0, **info)
        elif stage == "export":
            info = stage_export(args.model, args.tag, seed, n_requested, args.exact_top,
                                Path(args.site_out) if args.site_out else None)
            _record_stage(args.model, args.tag, stage, time.time() - t0,
                          report=info["report"], site_out=info["site_out"])
    if len(stages) > 1:
        print(f"[pilot] {','.join(stages)} done in {time.time() - t_all:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
