#!/usr/bin/env bash
# Put the box in demo shape, or take it back out.
#
# Why this exists: the corpus worker and the live read share one GPU. A live
# read costs ~10 s with the GPU to itself and ~21 s while the worker is mid
# frame, because the worker can only yield between frames and a frame is 12 s.
# During a three-minute demo the corpus does not need to grow, so we stop the
# worker outright and hand the whole GPU to the thing the judges are watching.
#
#   scripts/demo_mode.sh on     before presenting
#   scripts/demo_mode.sh off    afterwards, corpus resumes where it left off
#   scripts/demo_mode.sh check  time a live read without changing anything

set -euo pipefail
cd "$(dirname "$0")/.."

WORKER_PAT="safewalk.caption_worker"
CAM="${SAFEWALK_DEMO_CAM:-CMR-0176}"

worker_pid() { pgrep -f "$WORKER_PAT" || true; }

time_live_read() {
  local t0 t1
  t0=$(python3 -c 'import time;print(time.time())')
  curl -s "localhost:8010/api/camera/${CAM}/read?live=true" -o /tmp/safewalk_demo_read.json
  t1=$(python3 -c 'import time;print(time.time())')
  python3 - "$t0" "$t1" <<'PY'
import json, sys
elapsed = float(sys.argv[2]) - float(sys.argv[1])
try:
    d = json.load(open('/tmp/safewalk_demo_read.json'))
    inner = d.get('_seconds', '?')
except Exception:
    inner = '?'
print(f"  live read: {elapsed:.1f}s wall, {inner}s in model")
PY
}

case "${1:-check}" in
  on)
    pid=$(worker_pid)
    if [ -n "$pid" ]; then
      kill -STOP $pid
      echo "corpus worker suspended (pid $pid) — GPU is now dedicated to live reads"
    else
      echo "corpus worker not running — GPU already dedicated"
    fi
    rm -f data/meta/vlm_pause.lock
    curl -s -o /dev/null -w "  api: http=%{http_code}\n" localhost:8010/api/stats
    time_live_read
    echo "DEMO MODE ON. Run 'scripts/demo_mode.sh off' afterwards."
    ;;
  off)
    pid=$(worker_pid)
    if [ -n "$pid" ]; then
      kill -CONT $pid
      echo "corpus worker resumed (pid $pid)"
    else
      echo "corpus worker not running — start it with:"
      echo "  nohup .venv/bin/python -m safewalk.caption_worker > logs/captions.log 2>&1 &"
    fi
    ;;
  check)
    pid=$(worker_pid)
    state=$(ps -o state= -p "${pid:-0}" 2>/dev/null || echo "-")
    echo "corpus worker: pid=${pid:-none} state=${state}  (T = suspended)"
    time_live_read
    ;;
  *)
    echo "usage: $0 [on|off|check]" >&2
    exit 1
    ;;
esac
