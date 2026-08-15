#!/bin/bash
# Resume: mount "Extreme SSD", run this. Skips finished repos (.DONE), resumes partial shards.
# Molmo2 -> external SSD, same convention as HACKATHON_MODELS/README.md (idempotent).
export PATH="$HOME/.local/bin:$PATH"
SSD="/Volumes/Extreme SSD/HACKATHON_MODELS"
LOG="$SSD/logs/hf.log"

get() {  # get <repo> <subdir> [extra args...]
  local repo="$1"; local sub="$2"; shift 2
  local dest="$SSD/$sub/$(basename "$repo")"
  local stamp="$dest/.DONE"
  if [ -f "$stamp" ]; then echo "[skip] $repo" >> "$LOG"; return 0; fi
  echo "=== [$(date +%H:%M:%S)] START $repo -> $dest" >> "$LOG"
  if hf download "$repo" --local-dir "$dest" "$@" >> "$LOG" 2>&1; then
    touch "$stamp"
    echo "=== [$(date +%H:%M:%S)] OK $repo ($(du -sh "$dest" | cut -f1))" >> "$LOG"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL $repo" >> "$LOG"
  fi
}

echo "########## RUN molmo2 $(date) ##########" >> "$LOG"
get mlx-community/Molmo2-8B-4bit  vlm     # Mac / MLX test copy (6.5 GB)
get allenai/Molmo2-8B             vlm     # Spark copy, fp32 shards (34.7 GB)
echo "########## DONE molmo2 $(date) ##########" >> "$LOG"
