"""Build the corpus slices from ``codeparrot/codeparrot-clean-valid``.

Per CONTRACT.md: 12 repo-exclusive, content-hash-deduped slices; documents tokenized with
``model/tokenizer.json``, one EOS (2047) appended after each document, packed into
``uint16 [n_seq, 257]`` rows.  Inputs are ``[:, :256]`` and targets ``[:, 1:]``.

Packing uses non-overlapping 256-token input windows (``tokens[i:i+256]`` /
``tokens[i+1:i+257]``), i.e. row ``i`` stores ``tokens[256*i : 256*i+257]``; a slice
therefore needs ``n_seq*256 + 1`` tokens.  This is the literal reading of the corpus recon
(``docs/recon/ScoutCorpus.md`` §3) and keeps the 256 scored positions contiguous.

Slices are filled by consuming repository groups in ``sha256(repo_name)`` order, so no
repository is ever split across two slices; the final (partial) group of a slice is
dropped rather than reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from swa.model import DATA, ROOT, TOKENIZER_PATH

N_SLICES = 12
T = 256
EOS = 2047
PARQUET_URLS = [
    "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet",
    "https://huggingface.co/datasets/codeparrot/codeparrot-clean-valid/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0001.parquet",
]
COLUMNS = ["repo_name", "path", "hash", "license", "content"]


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dest)
    return dest


def fetch_rows(cache_dir: Path) -> list[dict]:
    """All rows of the validation split, in parquet order."""
    import pyarrow.parquet as pq

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for i, url in enumerate(PARQUET_URLS):
        path = _download(url, cache_dir / f"valid-{i:04d}.parquet")
        print(f"reading {path.name} ...", flush=True)
        table = pq.read_table(path, columns=COLUMNS)
        data = table.to_pydict()
        n = len(data["repo_name"])
        for j in range(n):
            rows.append({c: data[c][j] for c in COLUMNS})
        print(f"  {n} rows", flush=True)
    return rows


def _group_rows(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    """Dedup by content hash, group by repo, order groups by sha256(repo_name)."""
    seen: set[str] = set()
    groups: dict[str, list[dict]] = {}
    for r in rows:
        content = r["content"]
        repo = r["repo_name"]
        if not content or not repo or not r["path"]:
            continue
        ch = hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()
        if ch in seen:
            continue
        seen.add(ch)
        rec = {
            "repo_name": r["repo_name"],
            "path": r["path"],
            "hash": r["hash"],
            "license": r["license"],
            "content_hash": ch,
            "content": content,
        }
        groups.setdefault(r["repo_name"], []).append(rec)
    out = []
    for repo, files in groups.items():
        files.sort(key=lambda f: (f["path"], f["content_hash"]))
        out.append((repo, files))
    out.sort(key=lambda rf: hashlib.sha256(rf[0].encode("utf-8", "surrogatepass")).hexdigest())
    return out


def _tokenize(tok: Tokenizer, content: str) -> list[int]:
    return tok.encode(content, add_special_tokens=False).ids


def build(model_name: str = "csp_yolo1", n_seq: int = 64, n_slices: int = N_SLICES,
          cache_dir: Path | None = None, link_models: tuple[str, ...] = ("csp_yolo2",),
          seed: int = 0, max_doc_tokens: int = 512, files_per_repo: int = 1) -> dict:
    """Build ``data/<model_name>/`` and link the same corpus into ``link_models``."""
    del seed  # ordering is fully determined by sha256(repo_name); kept for CLI symmetry
    cache_dir = cache_dir or (ROOT / "data" / "_raw")
    tok = Tokenizer.from_file(str(TOKENIZER_PATH))
    need = n_seq * T + 1

    rows = fetch_rows(cache_dir)
    groups = _group_rows(rows)
    print(f"corpus: {len(rows)} rows -> {len(groups)} unique repos")

    out_dir = DATA / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    gi = 0
    manifest = {
        "model": model_name,
        "source": "codeparrot/codeparrot-clean-valid",
        "tokenizer": "model/tokenizer.json",
        "T": T,
        "eos": EOS,
        "n_seq": n_seq,
        "packing": "non-overlapping 256-token input windows; row i = tokens[256*i:256*i+257]",
        # Diversity: without caps a 16k-token slice is ~5 large files from a handful of repos,
        # so per-slice statistics (e.g. how often a digit token occurs) swing wildly.
        "max_doc_tokens": max_doc_tokens,
        "files_per_repo": files_per_repo,
        "slices": [],
    }
    for s in range(n_slices):
        toks: list[int] = []
        used: list[dict] = []
        while len(toks) < need and gi < len(groups):
            repo, files = groups[gi]
            gi += 1
            for f in files[:files_per_repo]:
                full = _tokenize(tok, f["content"])
                ids = full[:max_doc_tokens - 1]
                before = len(toks)
                toks.extend(ids)
                toks.append(EOS)
                used.append({
                    "repo_name": repo,
                    "path": f["path"],
                    "hash": f["hash"],
                    "license": f["license"],
                    "content_hash": f["content_hash"],
                    "tokens": len(ids) + 1,
                    "doc_truncated": len(full) > len(ids),
                })
                if len(toks) >= need:
                    # the slice ends inside this document: record what actually contributed
                    used[-1]["tokens"] = need - before
                    used[-1]["truncated"] = True
                    break
        if len(toks) < need:
            raise RuntimeError(f"ran out of corpus at slice {s}: {len(toks)} < {need} tokens")
        toks = toks[:need]
        arr = np.empty((n_seq, T + 1), dtype=np.uint16)
        for i in range(n_seq):
            arr[i] = np.asarray(toks[i * T: i * T + T + 1], dtype=np.uint16)
        np.save(out_dir / f"slice_{s:02d}.npy", arr)
        manifest["slices"].append({
            "slice": s,
            "n_seq": n_seq,
            "n_positions": n_seq * T,
            "n_tokens": len(toks),
            "repos": sorted({u["repo_name"] for u in used}),
            "files": used,
        })
        print(f"  slice {s:02d}: {len(toks)} tokens, {len(used)} files, "
              f"{len(manifest['slices'][-1]['repos'])} repos")

    # repo exclusivity / dedup invariant check
    seen_repos: set[str] = set()
    seen_hashes: set[str] = set()
    for sl in manifest["slices"]:
        for r in sl["repos"]:
            assert r not in seen_repos, f"repo {r} reused across slices"
            seen_repos.add(r)
        for f in sl["files"]:
            assert f["content_hash"] not in seen_hashes, "content hash reused"
            seen_hashes.add(f["content_hash"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))

    for other in link_models:
        other_dir = DATA / other
        other_dir.mkdir(parents=True, exist_ok=True)
        for s in range(n_slices):
            src = Path("..") / model_name / f"slice_{s:02d}.npy"
            dst = other_dir / f"slice_{s:02d}.npy"
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        m = dict(manifest)
        m["model"] = other
        m["corpus_link"] = f"data/{model_name}"
        (other_dir / "manifest.json").write_text(json.dumps(m, indent=1))
        print(f"linked corpus into data/{other}/")
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="csp_yolo1")
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--n-slices", type=int, default=N_SLICES)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--no-link", action="store_true")
    ap.add_argument("--max-doc-tokens", type=int, default=512,
                    help="keep at most this many tokens per document (incl. the appended EOS)")
    ap.add_argument("--files-per-repo", type=int, default=1)
    args = ap.parse_args(argv)
    build(args.model, n_seq=args.n_seq, n_slices=args.n_slices, cache_dir=args.cache_dir,
          link_models=() if args.no_link else ("csp_yolo2",),
          max_doc_tokens=args.max_doc_tokens, files_per_repo=args.files_per_repo)


if __name__ == "__main__":
    main()
