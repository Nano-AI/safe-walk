"""Sweep daemon: fetch every camera on an interval, forever.

Two things this has to get right that a naive fetch loop gets wrong:

1. A dead camera returns HTTP 200 with a "CAMERA UNDER MAINTENANCE" placeholder
   JPEG, not a 404. Caption those blind and a chunk of the city reads back as
   "a sign about maintenance". We detect them two ways: a persisted hash
   blocklist, and the observation that a hash appearing on several *different*
   cameras in one sweep cannot be a real street view.

2. The interval is the whole Spark story. Mac runs at SWEEP_INTERVAL=900 because
   a full VLM pass costs 20-30 min there. The Spark runs the same code at 60.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from PIL import Image

from . import cameras as cameras_mod
from . import config

# Resolution tiers. Low res is NOT a placeholder signal: the whole WSDOT network
# streams real freeway views at ~335x249. Tier only decides caption priority --
# the pedestrian product runs on SDOT hd/sd, WSDOT low is a freeway side layer.
TIER_HD = 1280
TIER_SD = 640

# A hash seen on at least this many distinct *image urls* in one sweep is
# boilerplate ("CAMERA UNDER MAINTENANCE"). Counting urls rather than camera ids
# matters because the feed lists some cameras under several ids sharing one url,
# and those legitimately produce identical bytes.
SHARED_HASH_THRESHOLD = 3


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_placeholder_hashes() -> set[str]:
    if config.PLACEHOLDER_HASHES.exists():
        return set(json.loads(config.PLACEHOLDER_HASHES.read_text()))
    return set()


def _save_placeholder_hashes(hashes: set[str]) -> None:
    config.PLACEHOLDER_HASHES.write_text(json.dumps(sorted(hashes), indent=2))


async def _fetch_one(
    client: httpx.AsyncClient,
    cam: cameras_mod.Camera,
    sem: asyncio.Semaphore,
    stamp: str,
) -> dict:
    """Fetch a single camera frame. Never raises -- failures become records."""
    record = {
        "ts": stamp,
        "cam_id": cam.id,
        "source": cam.source,
        "image_url": cam.image_url,
        "status": "error",
        "md5": None,
        "bytes": 0,
        "width": None,
        "height": None,
        "path": None,
        "error": None,
    }
    async with sem:
        try:
            # Cache-bust: these endpoints happily serve a stale CDN copy.
            resp = await client.get(cam.image_url, params={"_": int(time.time())})
            record["http"] = resp.status_code
            if resp.status_code != 200 or not resp.content:
                record["error"] = f"http {resp.status_code}"
                return record

            blob = resp.content
            record["bytes"] = len(blob)
            record["md5"] = hashlib.md5(blob).hexdigest()

            # Some cameras answer 200 with an HTML error page or a zero-length
            # body. Reject on magic bytes before handing anything to Pillow.
            if not blob.startswith(b"\xff\xd8\xff"):
                record["error"] = f"not-jpeg ct={resp.headers.get('content-type')!r}"
                record["status"] = "notjpeg"
                return record

            try:
                with Image.open(io.BytesIO(blob)) as im:
                    record["width"], record["height"] = im.size
            except Exception as exc:  # noqa: BLE001 - corrupt jpeg is just a bad frame
                record["error"] = f"decode: {exc}"
                return record

            out_dir = config.FRAMES / cam.id
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{stamp}.jpg"
            out_path.write_bytes(blob)
            record["path"] = str(out_path.relative_to(config.ROOT))
            record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - network flake is expected
            record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def _tier(width: int | None) -> str:
    if not width:
        return "unknown"
    if width >= TIER_HD:
        return "hd"
    if width >= TIER_SD:
        return "sd"
    return "low"


def _classify(records: list[dict], known_bad: set[str]) -> tuple[list[dict], set[str]]:
    """Mark placeholder frames and learn new placeholder hashes from this sweep."""
    hash_to_urls: dict[str, set[str]] = {}
    for r in records:
        if r["md5"]:
            hash_to_urls.setdefault(r["md5"], set()).add(r["image_url"])

    learned = {h for h, urls in hash_to_urls.items() if len(urls) >= SHARED_HASH_THRESHOLD}
    known_bad = known_bad | learned

    for r in records:
        r["tier"] = _tier(r["width"])
        if r["status"] == "ok" and r["md5"] in known_bad:
            r["status"] = "placeholder"

    # Placeholder frames are worthless on disk; drop them immediately.
    for r in records:
        if r["status"] == "placeholder" and r["path"]:
            Path(config.ROOT / r["path"]).unlink(missing_ok=True)
            r["path"] = None

    return records, known_bad


def _prune() -> None:
    """Keep recent frames per camera, thin the rest to one frame per hour."""
    for cam_dir in config.FRAMES.iterdir():
        if not cam_dir.is_dir():
            continue
        frames = sorted(cam_dir.glob("*.jpg"))
        recent = frames[-config.KEEP_RECENT_PER_CAM :]
        older = frames[: -config.KEEP_RECENT_PER_CAM]
        seen_hours: set[str] = set()
        # Walk newest-first so the frame we keep for an hour is its last one.
        for path in reversed(older):
            hour = path.stem[:11]  # YYYYmmddTHH
            if hour in seen_hours:
                path.unlink(missing_ok=True)
            else:
                seen_hours.add(hour)
        del recent


async def sweep(cams: list[cameras_mod.Camera]) -> dict:
    """One full pass over every camera."""
    stamp = _now_stamp()
    started = time.monotonic()
    sem = asyncio.Semaphore(config.FETCH_CONCURRENCY)
    headers = {"User-Agent": config.USER_AGENT}

    async with httpx.AsyncClient(
        timeout=config.FETCH_TIMEOUT, headers=headers, follow_redirects=True
    ) as client:
        records = await asyncio.gather(
            *(_fetch_one(client, c, sem, stamp) for c in cams)
        )

    records, known_bad = _classify(list(records), _load_placeholder_hashes())
    _save_placeholder_hashes(known_bad)

    with config.SWEEPS_JSONL.open("a", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    _prune()

    counts = Counter(r["status"] for r in records)
    summary = {
        "ts": stamp,
        "cameras": len(records),
        "ok": counts.get("ok", 0),
        "placeholder": counts.get("placeholder", 0),
        "error": counts.get("error", 0),
        "seconds": round(time.monotonic() - started, 1),
        "known_placeholder_hashes": len(known_bad),
    }
    return summary


async def run_forever() -> None:
    cams = cameras_mod.load_or_fetch()
    print(f"[scraper] {len(cams)} cameras, interval={config.SWEEP_INTERVAL}s", flush=True)
    while True:
        started = time.monotonic()
        try:
            summary = await sweep(cams)
            print(f"[sweep] {json.dumps(summary)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the daemon must not die
            print(f"[sweep] FAILED {type(exc).__name__}: {exc}", flush=True)
        elapsed = time.monotonic() - started
        await asyncio.sleep(max(5.0, config.SWEEP_INTERVAL - elapsed))


if __name__ == "__main__":
    import sys

    if "--once" in sys.argv:
        cams_now = cameras_mod.load_or_fetch()
        print(json.dumps(asyncio.run(sweep(cams_now)), indent=2))
    else:
        asyncio.run(run_forever())
