#!/usr/bin/env bash
# Full local run for csp_yolo1, resumable (every stage skips finished shards).
#   1. pilot: exact paper metric on N uniformly sampled MLP weights  (~12 h for 1024 on a Quadro RTX 4000)
#   2. profile every nonzero MLP weight on discovery slice 00         (~66 h)
#   3. synthesize predicates for every weight (Rust, minutes)
#   4. export the atlas site + annotated model.py (pilot scores merged in)
# Launch detached:  systemd-run --user --unit swa-full --working-directory="$PWD" scripts/run_full.sh
# Follow:           journalctl --user -u swa-full -f
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=engine
PY=.venv/bin/python
MODEL=${MODEL:-csp_yolo1}
N=${PILOT_N:-1024}

echo "[run_full] $(date -Is) pilot n=$N"
$PY -u -m swa.pilot --model "$MODEL" --n "$N" --seed 0 --tag pilot --stage all

echo "[run_full] $(date -Is) profile all weights"
$PY -u -m swa.profile --model "$MODEL" --layers 0,1,2,3,4,5,6,7,8,9,10,11

echo "[run_full] $(date -Is) synth all shards"
# --force: earlier partial runs (case study, pilot) left candidate files for incomplete shards.
synth/target/release/swa-synth --model "$MODEL" --root . --force

echo "[run_full] $(date -Is) export"
$PY -u -m swa.export_site --model "$MODEL" --scores-dir "runs/$MODEL/scores_pilot"
$PY -u -m swa.export_modelpy --model "$MODEL" --scores-dir "runs/$MODEL/scores_pilot"
echo "[run_full] $(date -Is) done"
