#!/bin/bash
# Real-world 3D models -> external SSD kit (idempotent; same convention as dl_molmo2.sh).
# Needs: SSD mounted at /Volumes/Extreme SSD, network. Re-run to resume.
#
#   3d/Sharp        apple/Sharp        single photo -> 3D Gaussians (<1 s on GPU). Fits our
#                                      single-viewpoint traffic cams. Code: toolchain/repos/ml-sharp
#   3d/VGGT-1B      facebook/VGGT-1B   multi-view -> cameras+depth+points, one pass. For
#                                      "phone video of a block -> 3D". Code: toolchain/repos/vggt
export PATH="$HOME/.local/bin:$PATH"
SSD="/Volumes/Extreme SSD/HACKATHON_MODELS"
LOG="$SSD/logs/hf.log"
[ -d "$SSD" ] || { echo "SSD not mounted" >&2; exit 2; }

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

clone() {  # clone <github url> — code the box needs; shallow, no history
  local url="$1"; local dest="$SSD/toolchain/repos/$(basename "$url" .git)"
  if [ -f "$dest/.DONE" ]; then echo "[skip] $url" >> "$LOG"; return 0; fi
  echo "=== [$(date +%H:%M:%S)] CLONE $url -> $dest" >> "$LOG"
  rm -rf "$dest"
  if git clone --depth 1 "$url" "$dest" >> "$LOG" 2>&1; then
    rm -rf "$dest/.git"   # exFAT: 1 MiB per file; a .git tree is thousands of tiny files
    touch "$dest/.DONE"
    echo "=== [$(date +%H:%M:%S)] OK $url" >> "$LOG"
  else
    echo "=== [$(date +%H:%M:%S)] FAIL $url" >> "$LOG"
  fi
}

echo "########## RUN splat $(date) ##########" >> "$LOG"
get   apple/Sharp        3d
clone https://github.com/apple/ml-sharp.git
get   facebook/VGGT-1B   3d
clone https://github.com/facebookresearch/vggt.git
echo "########## DONE splat $(date) ##########" >> "$LOG"
