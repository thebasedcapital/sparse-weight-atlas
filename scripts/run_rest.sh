#!/usr/bin/env bash
# Remaining full-run stages after the pilot, using the 1.32x fast profiler (scripts/check_fast.py gate).
# Launch: systemd-run --user --unit swa-rest --working-directory="$PWD" scripts/run_rest.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=engine
PY=.venv/bin/python
MODEL=${MODEL:-csp_yolo1}
echo "[run_rest] $(date -Is) profile all weights (--fast)"
$PY -u -m swa.profile --model "$MODEL" --layers 0,1,2,3,4,5,6,7,8,9,10,11 --fast
echo "[run_rest] $(date -Is) synth all shards"
synth/target/release/swa-synth --model "$MODEL" --root . --force
echo "[run_rest] $(date -Is) export"
$PY -u -m swa.export_site --model "$MODEL" --scores-dir "runs/$MODEL/scores_pilot"
$PY -u -m swa.export_modelpy --model "$MODEL" --scores-dir "runs/$MODEL/scores_pilot"
echo "[run_rest] $(date -Is) done"
