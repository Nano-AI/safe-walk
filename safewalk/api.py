"""HTTP surface for the map UI.

Split by honesty: /route and /camera/{id}/latest read the pre-computed corpus and
answer instantly. /camera/{id}/live goes to the city right now -- the frame comes
back immediately, the VLM read follows on a second call, because a correct read
costs about thirteen seconds on this laptop and we would rather show the fresh
pixels first than stall behind the model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import cameras as cmod
from . import config, graph as gmod, live as L, routing as R
from .caption_worker import LATEST, newest_frame

app = FastAPI(title="safe walk", docs_url="/api/docs")

_G = None
_CAMS: dict[str, cmod.Camera] = {}

# Named starting points so a judge can pick a route without hunting the map.
PLACES = [
    {"name": "Westlake Center", "lat": 47.6115, "lon": -122.3376},
    {"name": "Pike Place Market", "lat": 47.6097, "lon": -122.3421},
    {"name": "Pioneer Square", "lat": 47.6015, "lon": -122.3343},
    {"name": "Seattle Center", "lat": 47.6205, "lon": -122.3493},
    {"name": "Convention Center", "lat": 47.6115, "lon": -122.3320},
    {"name": "Capitol Hill (Broadway & Pike)", "lat": 47.6140, "lon": -122.3205},
    {"name": "King Street Station", "lat": 47.5985, "lon": -122.3300},
    {"name": "South Lake Union", "lat": 47.6255, "lon": -122.3370},
]


def graph():
    global _G
    if _G is None:
        _G = R.annotate(gmod.load_or_build())
    return _G


def cams() -> dict[str, cmod.Camera]:
    global _CAMS
    if not _CAMS:
        _CAMS = {c.id: c for c in cmod.load_or_fetch()}
    return _CAMS


def _latest_reads() -> dict:
    if LATEST.exists():
        try:
            return json.loads(LATEST.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _ranks(reads: dict) -> dict:
    """Where each camera sits against the rest of downtown at this moment.

    "At this moment" is enforced: a read whose frame is older than LIVE_MAX_AGE
    is left out, otherwise a 3 a.m. read still sitting in latest.json gets
    ranked against 1 p.m. reads and the percentile means nothing.
    """
    from . import baseline

    try:
        fresh, _ = L.fresh_reads(reads)
        return baseline.get(refresh=True).rank_now(fresh)
    except Exception:  # noqa: BLE001 - ranking is enrichment, never a hard failure
        return {}


def _age(obs: dict | None) -> int | None:
    t = L.frame_time(obs) if obs else None
    return int(time.time() - t) if t else None


def _decorate(route: dict | None, reads: dict, ranks: dict) -> dict | None:
    """Attach the newest VLM read to every camera named in a route."""
    if not route:
        return None
    seen: dict[str, dict] = {}
    for seg in route["segments"]:
        for cam in seg["cameras"]:
            cid = cam["id"]
            if cid in seen:
                continue
            meta = cams().get(cid)
            obs = reads.get(cid)
            seen[cid] = {
                "id": cid,
                "description": meta.description if meta else cid,
                "lat": meta.lat if meta else None,
                "lon": meta.lon if meta else None,
                "dist_m": cam["dist_m"],
                "observation": {k: v for k, v in (obs or {}).items() if not k.startswith("_")} or None,
                "read_seconds": (obs or {}).get("_seconds"),
                "frame_age_s": _age(obs),
                "rank": ranks.get(cid),
            }
    route["camera_details"] = sorted(seen.values(), key=lambda c: c["dist_m"])
    route["cameras_read"] = sum(1 for c in seen.values() if c["observation"])
    return route


@app.get("/api/places")
def places():
    return PLACES


@app.get("/api/stats")
def stats():
    G = graph()
    edges = [d for *_, d in G.edges(data=True)]
    reads = _latest_reads()
    watched = sum(1 for d in edges if d["cameras"])
    ped_total = sum(d["ped_collisions"] for d in edges)
    ped_watched = sum(d["ped_collisions"] for d in edges if d["cameras"])
    sweeps = 0
    if config.SWEEPS_JSONL.exists():
        with config.SWEEPS_JSONL.open() as fh:
            sweeps = sum(1 for _ in fh)
    return {
        "cameras_total": len(cams()),
        "cameras_read": len(reads),
        "segments": len(edges),
        "segments_watched": watched,
        "segments_watched_pct": round(100 * watched / len(edges)),
        "ped_collisions": ped_total,
        "ped_collisions_watched_pct": round(100 * ped_watched / ped_total),
        "frames_captured": sweeps,
        "sweep_interval_s": config.SWEEP_INTERVAL,
    }


@app.get("/api/network")
def network():
    """The whole downtown street graph, for drawing the basemap locally.

    Rendering our own wireframe instead of pulling map tiles is not a stylistic
    whim: it means the demo has zero external dependencies and looks identical
    with the venue wifi unplugged.
    """
    G = graph()
    segs = []
    for *_, d in G.edges(data=True):
        segs.append({
            "id": d["seg_id"],
            "g": [[round(lon, 5), round(lat, 5)] for lon, lat in d["geometry"]],
            "r": d["risk"],
            "p": d["ped_collisions"],
            "c": len(d["cameras"]),
            "n": d["name"],
        })
    pins = [
        {"id": c.id, "lat": round(c.lat, 5), "lon": round(c.lon, 5), "d": c.description}
        for c in cams().values()
        if c.downtown
    ]
    lat_min, lat_max, lon_min, lon_max = config.DOWNTOWN_BBOX
    return {"segments": segs, "cameras": pins,
            "bbox": {"lat_min": lat_min, "lat_max": lat_max,
                     "lon_min": lon_min, "lon_max": lon_max}}


@app.get("/api/route")
def get_route(from_lat: float, from_lon: float, to_lat: float, to_lon: float,
              safe_weight: float = 3.0, use_live: bool = True):
    """Direct vs evidence-led route.

    `use_live=false` routes on the static record alone -- the ablation a judge
    can ask for, and the answer to "how much did the cameras actually change".
    """
    G = graph()
    t0 = time.time()
    reads = _latest_reads()
    live, live_summary = L.penalties(G, reads) if use_live else ({}, None)
    res = R.compare(G, (from_lat, from_lon), (to_lat, to_lon),
                    safe_weight=safe_weight, live=live)
    if not res["direct"]:
        raise HTTPException(404, "no route between those points")
    ranks = _ranks(reads)
    res["direct"] = _decorate(res["direct"], reads, ranks)
    res["safer"] = _decorate(res["safer"], reads, ranks)
    res["live"] = live_summary
    res["safe_weight"] = safe_weight
    res["compute_ms"] = round((time.time() - t0) * 1000, 1)
    return res


@app.get("/api/busiest")
def busiest(limit: int = 8):
    """The corners with the most going on right now, city-relative.

    Ranking cameras against each other in the same sweep cancels out sun,
    weather and day-of-week, which is what makes this answerable from one day
    of archive when "busier than usual for a Saturday" is not.
    """
    reads = _latest_reads()
    ranks = _ranks(reads)
    rows = []
    for cam_id, r in sorted(ranks.items(), key=lambda kv: -kv[1]["percentile"])[:limit]:
        meta = cams().get(cam_id)
        obs = reads.get(cam_id) or {}
        rows.append({
            "id": cam_id,
            "description": meta.description if meta else cam_id,
            "percentile": r["percentile"],
            "of": r["of"],
            "traffic": obs.get("traffic"),
            "people_visible": obs.get("people_visible"),
            "crowding": obs.get("crowding"),
            "sidewalk_blocked": obs.get("sidewalk_blocked"),
            "construction": obs.get("construction"),
            "notable": obs.get("notable"),
        })
    return {"ranked_over": len(ranks), "cameras": rows}


@app.get("/api/camera/{cam_id}/latest")
def camera_latest(cam_id: str):
    meta = cams().get(cam_id)
    if not meta:
        raise HTTPException(404, "unknown camera")
    obs = _latest_reads().get(cam_id)
    frame = newest_frame(cam_id)
    return {
        "id": cam_id,
        "description": meta.description,
        "lat": meta.lat,
        "lon": meta.lon,
        "frame_name": frame.name if frame else None,
        "observation": obs,
    }


@app.get("/api/camera/{cam_id}/frame.jpg")
def camera_frame(cam_id: str, live: bool = False):
    """Serve the newest stored frame, or fetch a brand new one from the city."""
    meta = cams().get(cam_id)
    if not meta:
        raise HTTPException(404, "unknown camera")

    if live:
        out = config.FRAMES / cam_id / "live.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        # A failed fetch must not leave last time's live.jpg for /read to find.
        out.unlink(missing_ok=True)
        try:
            r = httpx.get(meta.image_url, params={"_": int(time.time())},
                          headers={"User-Agent": config.USER_AGENT}, timeout=12,
                          follow_redirects=True)
            if r.status_code == 200 and r.content.startswith(b"\xff\xd8\xff"):
                out.write_bytes(r.content)
                return FileResponse(out, media_type="image/jpeg",
                                    headers={"Cache-Control": "no-store",
                                             "X-Frame-Age": "0"})
        except httpx.HTTPError:
            pass  # fall through to the cached frame

    frame = newest_frame(cam_id)
    if not frame:
        raise HTTPException(404, "no frame captured yet")
    age = int(time.time() - frame.stat().st_mtime)
    return FileResponse(frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store", "X-Frame-Age": str(age)})


@app.get("/api/camera/{cam_id}/read")
def camera_read(cam_id: str, live: bool = False):
    """Run the VLM now. Slow by design -- this is the beat we do live on stage."""
    from . import vision

    meta = cams().get(cam_id)
    if not meta:
        raise HTTPException(404, "unknown camera")

    live_jpg = config.FRAMES / cam_id / "live.jpg"
    source = "stored"
    if live and live_jpg.exists() and time.time() - live_jpg.stat().st_mtime <= config.LIVE_FRAME_TTL:
        target = live_jpg
        source = "live"
    else:
        target = newest_frame(cam_id)
        if not target:
            raise HTTPException(404, "no frame captured yet")
    frame_age = int(time.time() - target.stat().st_mtime)

    # Hold the GPU for the duration; the corpus worker waits its turn.
    config.VLM_PAUSE_LOCK.write_text(str(time.time()))
    try:
        obs = vision.caption_frame(target)
    finally:
        config.VLM_PAUSE_LOCK.unlink(missing_ok=True)

    obs["cam_id"] = cam_id
    obs["description"] = meta.description
    obs["frame_source"] = source
    obs["frame_age_s"] = frame_age
    return JSONResponse(obs)


@app.on_event("startup")
def _prewarm():
    """Load the model at boot so the first live read on stage is not the slow one."""
    import threading

    def load():
        try:
            from . import vision

            vision._ensure_loaded()
            print("[api] vlm pre-warmed", flush=True)
        except Exception as exc:  # noqa: BLE001 - the rest of the API works without it
            print(f"[api] pre-warm failed: {exc}", flush=True)

    threading.Thread(target=load, daemon=True).start()
    config.VLM_PAUSE_LOCK.unlink(missing_ok=True)


UI = config.ROOT / "web"
if UI.is_dir():
    app.mount("/", StaticFiles(directory=str(UI), html=True), name="ui")
