#!/usr/bin/env bash
# Exact paper-metric scoring of every high-impact weight (|ΔCE| >= 1e-3 on slice 00) not already scored.
# wids: runs/csp_yolo1/highimpact/wids.npy. One swa.score call per home shard, so each finished shard is a
# checkpoint (runs/csp_yolo1/scores_highimpact/<shard>.jsonl, skipped on rerun); a crash loses <= one shard.
# Launch: systemd-run --user --unit swa-highimpact --working-directory="$PWD" scripts/run_highimpact.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=engine
PY=.venv/bin/python
MODEL=${MODEL:-csp_yolo1}
WIDS=runs/$MODEL/highimpact/wids.npy

for shard in $(ls "runs/$MODEL/profiles"); do
  echo "[highimpact] $(date -Is) $shard"
  $PY -u -m swa.score --model "$MODEL" --wids-file "$WIDS" --tag highimpact --shards "$shard" --quiet
done

echo "[highimpact] $(date -Is) export"
DIRS=("runs/$MODEL/scores_casestudy" "runs/$MODEL/scores_pilot" "runs/$MODEL/scores_highimpact")
$PY -u -m swa.export_site --model "$MODEL" --scores-dirs "${DIRS[@]}"
$PY -u -m swa.export_modelpy --model "$MODEL" --scores-dirs "${DIRS[@]}"
echo "[highimpact] $(date -Is) done"
