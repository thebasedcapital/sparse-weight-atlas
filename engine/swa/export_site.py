"""Export pipeline outputs (runs/<model>) to static site data (site/data/<model>).

Usage:
    PYTHONPATH=engine python -m swa.export_site --model csp_yolo1
    # optional: --runs, --data, --out, --root for non-standard locations

Reads:
    runs/<model>/weights.npz            (layer, matrix, row, col, value arrays; index = wid)
    runs/<model>/vocab.json             {"tokens": [...], "classes": {...}}
    runs/<model>/scores/*.jsonl         per-wid score records (or --scores-dir)
    runs/<model>/profiles/<shard>/      wid.npy, indptr/pos/dce/kl CSR, dce_total, kl_total
    data/<model>/slice_00.npy           uint16 [n_seq, 257] discovery tokens

Writes (per CONTRACT "Site data"):
    site/data/<model>/summary.json
    site/data/<model>/layer_{LL}.json
    site/data/<model>/w/{wid//1000}.json
    site/data/models.json               (manifest of available models, upserted)
"""
import argparse
import json
from pathlib import Path

import numpy as np

MATRIX_NAMES = ("c_fc", "c_proj")
STATUSES = ("scored", "noise_floor", "no_candidate", "unscored")
PAPER_ROBUST_RATE = 0.15
PAPER_INTERPA_RATE = 0.373
PAPER_INTERPB_RATE = 0.353
N_CONTEXTS = 3
CONTEXT_RADIUS = 8


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_scores(runs_dir: Path, scores_dir: Path | None = None) -> dict:
    """Merge every <scores-dir>/<shard>.jsonl into wid -> record."""
    scores = {}
    sdir = scores_dir if scores_dir is not None else runs_dir / "scores"
    if not sdir.is_dir():
        return scores
    for path in sorted(sdir.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                scores[int(rec["wid"])] = rec
    return scores


class ProfileIndex:
    """wid -> (shard_dir, row) for every profiled weight; lazy per-shard CSR loads."""

    def __init__(self, runs_dir: Path):
        self.root = runs_dir / "profiles"
        self.loc = {}  # wid -> (shard_dir, index)
        if self.root.is_dir():
            for shard in sorted(self.root.iterdir()):
                wid_path = shard / "wid.npy"
                if not wid_path.exists():
                    continue
                wids = np.load(wid_path)
                for i, w in enumerate(wids):
                    self.loc[int(w)] = (shard, i)
        self._cache_dir = None
        self._cache = None

    def _load(self, shard: Path):
        if self._cache_dir == shard:
            return self._cache
        indptr = np.load(shard / "indptr.npy")
        pos = np.load(shard / "pos.npy")
        dce = np.load(shard / "dce.npy")
        kl = np.load(shard / "kl.npy")
        self._cache_dir = shard
        self._cache = (indptr, pos, dce, kl)
        return self._cache

    def top_contexts(self, wid: int, inputs: np.ndarray, tokens: list, k: int = N_CONTEXTS):
        """Top-k kept positions by KL, each as decoded +-8-token window with focus dce/kl."""
        loc = self.loc.get(wid)
        if loc is None or inputs is None or not tokens:
            return []
        shard, i = loc
        indptr, pos, dce, kl = self._load(shard)
        s, e = int(indptr[i]), int(indptr[i + 1])
        if e <= s:
            return []
        order = np.argsort(kl[s:e])[::-1][:k]
        out = []
        for j in order:
            flat = int(pos[s + j])
            seq, t = flat // 256, flat % 256
            lo, hi = max(0, t - CONTEXT_RADIUS), min(255, t + CONTEXT_RADIUS)
            out.append({
                "tokens": [tokens[x] for x in inputs[seq, lo:hi + 1].tolist()],
                "focus": t - lo,
                "dce": float(dce[s + j]),
                "kl": float(kl[s + j]),
            })
        return out


def interp_rate(recs):
    """Paper rate: score >= 0.75 and 0 < cov <= 0.5, over the given records."""
    if not recs:
        return None
    n = 0
    for r in recs:
        s = r.get("score")
        cov = r.get("cov")
        if s is not None and cov is not None and s >= 0.75 and 0 < cov <= 0.5:
            n += 1
    return n / len(recs)


def export_model(model: str, runs: Path, data: Path, out: Path,
                 scores_dir: Path | None = None) -> dict:
    runs_dir, data_dir = runs / model, data / model
    out_dir = out / model
    (out_dir / "w").mkdir(parents=True, exist_ok=True)

    npz = np.load(runs_dir / "weights.npz")
    layer, matrix, row, col, value = (npz["layer"], npz["matrix"], npz["row"],
                                      npz["col"], npz["value"])
    W = len(layer)

    scores = load_scores(runs_dir, scores_dir)
    profiles = ProfileIndex(runs_dir)

    tokens = None
    inputs = None
    vocab_path = runs_dir / "vocab.json"
    slice_path = data_dir / "slice_00.npy"
    if vocab_path.exists():
        tokens = json.loads(vocab_path.read_text(encoding="utf-8"))["tokens"]
    if slice_path.exists():
        inputs = np.load(slice_path)[:, :256]

    # ---- per-weight rows (layer files) and cards (bucket files) ----
    n_layers = int(layer.max()) + 1 if W else 0
    layer_rows = [[] for _ in range(n_layers)]
    buckets = {}  # bucket id -> {wid: card}
    counts = {s: 0 for s in STATUSES}
    robust_pass = 0
    a_recs, b_recs = [], []
    bin_edges = [1e-6, 1e-5, 1e-4]
    bin_labels = ["<1e-6", "1e-6..1e-5", "1e-5..1e-4", ">=1e-4"]
    bins = [{"bin": b, "total": 0, "pass": 0} for b in bin_labels]

    for wid in range(W):
        L = int(layer[wid]); m = int(matrix[wid])
        rec = scores.get(wid)
        if rec is None:
            status, dce, spec, py, desc, A, B, rob = "unscored", None, None, None, None, None, None, None
        else:
            status = rec.get("status", "unscored")
            dce = rec.get("dce")
            spec, py, desc = rec.get("spec"), rec.get("py"), rec.get("desc")
            A, B, rob = rec.get("A"), rec.get("B"), rec.get("robust")
        counts[status] += 1
        passed = bool(rob and rob.get("pass"))
        robust_pass += passed
        if A:
            a_recs.append(A)
        if B:
            b_recs.append(B)
        if dce is not None:
            adce = abs(dce)
            bi = 0 if adce < bin_edges[0] else 1 if adce < bin_edges[1] else 2 if adce < bin_edges[2] else 3
            bins[bi]["total"] += 1
            bins[bi]["pass"] += passed

        score_mean = rob.get("mean") if rob else None
        layer_rows[L].append({
            "wid": wid, "matrix": m, "row": int(row[wid]), "col": int(col[wid]),
            "value": float(value[wid]), "dce": dce, "status": status,
            "score_robust_mean": score_mean, "pass": passed, "desc": desc,
        })

        card = {
            "wid": wid, "layer": L, "matrix": m, "matrix_name": MATRIX_NAMES[m],
            "row": int(row[wid]), "col": int(col[wid]), "value": float(value[wid]),
            "status": status, "dce": dce, "spec": spec, "py": py, "desc": desc,
            "A": A, "B": B, "robust": rob,
        }
        if status == "unscored" or profiles.loc.get(wid) is None:
            card["contexts"] = []
        else:
            card["contexts"] = profiles.top_contexts(wid, inputs, tokens) if tokens is not None and inputs is not None else []
        buckets.setdefault(wid // 1000, {})[wid] = card

    # ---- write layer files ----
    per_layer = []
    for L in range(n_layers):
        rows = layer_rows[L]
        (out_dir / f"layer_{L:02d}.json").write_text(
            json.dumps({"layer": L, "n": len(rows), "w": rows},
                       separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        per_layer.append({
            "layer": L, "total": len(rows),
            **{s: sum(1 for r in rows if r["status"] == s) for s in STATUSES},
            "pass": sum(1 for r in rows if r["pass"]),
        })

    # ---- write bucket files ----
    for b, cards in buckets.items():
        (out_dir / "w" / f"{b}.json").write_text(
            json.dumps({"bucket": b, "w": cards}, separators=(",", ":"), ensure_ascii=False),
            encoding="utf-8")

    # ---- summary ----
    scored = counts["scored"]
    verdicted = scored + counts["noise_floor"] + counts["no_candidate"]
    interp_a = interp_rate(a_recs)
    interp_b = interp_rate(b_recs)
    summary = {
        "model": model,
        "counts": {
            "total": W, "scored": scored,
            "noise_floor": counts["noise_floor"],
            "no_candidate": counts["no_candidate"],
            "unscored": counts["unscored"],
            "robust_pass": robust_pass,
        },
        "rates": {
            # paper protocol: denominator = weights with a verdict (excludes unscored)
            "robust": (robust_pass / verdicted) if verdicted else None,
            "robust_of_total": (robust_pass / W) if W else None,
            "interpA": interp_a,
            "interpB": interp_b,
        },
        "paper": {"robust": PAPER_ROBUST_RATE, "interpA": PAPER_INTERPA_RATE,
                  "interpB": PAPER_INTERPB_RATE, "T": 0.75, "c": 0.5},
        "per_layer": per_layer,
        "per_bin": bins,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")

    # ---- models.json manifest upsert ----
    manifest_path = out / "models.json"
    manifest = []
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")).get("models", [])
        except (json.JSONDecodeError, OSError):
            manifest = []
    entry = {"id": model, "summary": f"data/{model}/summary.json"}
    manifest = [m for m in manifest if m.get("id") != model] + [entry]
    manifest_path.write_text(json.dumps({"models": manifest}, indent=1), encoding="utf-8")

    return summary


def main():
    root = repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--root", default=str(root), help="project root (default: repo root)")
    ap.add_argument("--runs", default=None, help="default: <root>/runs")
    ap.add_argument("--data", default=None, help="default: <root>/data")
    ap.add_argument("--out", default=None, help="default: <root>/site/data")
    ap.add_argument("--scores-dir", default=None,
                    help="directory of <shard>.jsonl score records "
                         "(default: <runs>/<model>/scores), e.g. runs/<model>/scores_pilot")
    args = ap.parse_args()

    runs = Path(args.runs) if args.runs else Path(args.root) / "runs"
    data = Path(args.data) if args.data else Path(args.root) / "data"
    out = Path(args.out) if args.out else Path(args.root) / "site" / "data"

    scores_dir = Path(args.scores_dir) if args.scores_dir else None
    summary = export_model(args.model, runs, data, out, scores_dir=scores_dir)
    c = summary["counts"]
    print(f"[export_site] {args.model}: total={c['total']} scored={c['scored']} "
          f"noise_floor={c['noise_floor']} no_candidate={c['no_candidate']} "
          f"unscored={c['unscored']} robust_pass={c['robust_pass']} "
          f"({(summary['rates']['robust'] or 0) * 100:.2f}% vs paper {PAPER_ROBUST_RATE * 100:.1f}%)")
    print(f"[export_site] wrote {out / args.model}")


if __name__ == "__main__":
    main()
