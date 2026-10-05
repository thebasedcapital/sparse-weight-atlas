#!/usr/bin/env bash
# Wait until swa-full finishes its pilot (report.json written and profiling started), then switch
# the remaining work to the faster profiler in unit swa-rest.
cd "$(dirname "$0")/.."
until [ -f runs/csp_yolo1/pilot/report.json ] && journalctl --user -u swa-full --no-pager | grep -q 'profile all weights'; do sleep 60; done
systemctl --user stop swa-full
sleep 5
systemd-run --user --unit swa-rest --working-directory="$PWD" scripts/run_rest.sh
echo "[handoff] $(date -Is) switched to swa-rest"
