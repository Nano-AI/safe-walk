"""Continuously caption the newest frame from every camera, worst blocks first.

On the Mac a full pass over 646 cameras costs roughly two and a half hours, so
the order matters: cameras watching the segments with the worst pedestrian
collision history get read first, and the long tail fills in behind them. On the
Spark the same pass is a background tick and the priority ordering stops
mattering -- which is precisely the point we make on stage.

Writes two things:
  captions.jsonl  append-only history, one line per (camera, frame) read
  latest.json     newest observation per camera, what the API actually serves
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from . import config, graph as gmod, vision

CAPTIONS = config.META / "captions.jsonl"
LATEST = config.META / "latest.json"


def camera_priority() -> list[str]:
    """Cameras ranked by the worst pedestrian collision count they oversee."""
    G = gmod.load_or_build()
    score: dict[str, int] = {}
    for *_, d in G.edges(data=True):
        for cam in d["cameras"]:
            cid = cam["id"]
            score[cid] = max(score.get(cid, 0), d["ped_collisions"])
    ranked = sorted(score.items(), key=lambda kv: -kv[1])
    return [cid for cid, _ in ranked]


def newest_frame(cam_id: str) -> Path | None:
    d = config.FRAMES / cam_id
    if not d.is_dir():
        return None
    # Ignore the downscaled cache files vision.py writes beside the originals.
    frames = sorted(p for p in d.glob("*.jpg") if ".s" not in p.suffixes[0:1] and p.stem.endswith("Z"))
    return frames[-1] if frames else None


def _load_latest() -> dict:
    if LATEST.exists():
        try:
            return json.loads(LATEST.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def run_forever() -> None:
    order = camera_priority()
    print(f"[captions] {len(order)} cameras, priority-ordered by ped collisions", flush=True)
    vision._ensure_loaded()
    print(f"[captions] model ready: {vision.MODEL_ID}", flush=True)

    latest = _load_latest()
    done_pass = 0
    while True:
        started = time.time()
        read = 0
        for cam_id in order:
            # A judge is waiting on a live read; get off the GPU.
            waited = 0.0
            while config.VLM_PAUSE_LOCK.exists() and waited < 120:
                time.sleep(0.4)
                waited += 0.4

            frame = newest_frame(cam_id)
            if frame is None:
                continue
            # Skip if we already read this exact frame.
            if latest.get(cam_id, {}).get("frame_name") == frame.name:
                continue
            try:
                obs = vision.caption_frame(frame)
            except Exception as exc:  # noqa: BLE001 - one bad frame must not stop the pass
                print(f"[captions] {cam_id} FAILED {type(exc).__name__}: {exc}", flush=True)
                continue

            obs["cam_id"] = cam_id
            obs["frame_name"] = frame.name
            obs["read_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            with CAPTIONS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(obs) + "\n")
            latest[cam_id] = obs
            LATEST.write_text(json.dumps(latest))
            read += 1
            if read % 10 == 0:
                rate = (time.time() - started) / read
                print(f"[captions] pass {done_pass}: {read} read, "
                      f"{rate:.1f}s/frame, {len(latest)}/{len(order)} cameras covered",
                      flush=True)

        done_pass += 1
        print(f"[captions] pass {done_pass} complete: {read} new reads in "
              f"{(time.time() - started) / 60:.1f} min, "
              f"{len(latest)}/{len(order)} cameras covered", flush=True)
        if read == 0:
            time.sleep(60)  # nothing new yet; wait for the scraper


if __name__ == "__main__":
    run_forever()
