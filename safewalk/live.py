"""Turn the newest camera reads into per-segment routing nudges.

The static risk term is history: collisions, sidewalks, grade. This module is
the present tense -- what a camera showed within the last hour -- and it only
ever *adds* to a segment's cost. Three rules keep it honest:

1. Only conditions the prompt is allowed to assert (a blocked walkway,
   construction, emergency vehicles, an unlit street) move a route. Crowding
   and people counts do not: a busy sidewalk is not a hazard, and a router that
   steers around crowds steers around neighbourhoods.
2. A read older than LIVE_MAX_AGE is ignored, and every nudge carries the frame
   age so the UI can show it. "Sidewalk blocked" from six hours ago is not news.
3. The nudge is capped well below the static term, so a camera can tilt a close
   call but cannot overrule the collision record.

The dict this returns is keyed by seg_id and consumed by routing._edge_cost.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import networkx as nx

from . import config

# Additive risk per condition, before the cap. Emergency activity is the
# strongest because it is the one thing in the schema that means "right now".
NUDGE = {
    "emergency_activity": (0.25, "emergency activity"),
    "sidewalk_blocked": (0.20, "sidewalk blocked"),
    "construction": (0.10, "construction"),
}
UNLIT_NUDGE = (0.15, "street unlit")
LIVE_CAP = 0.35


def frame_time(obs: dict) -> float | None:
    """Capture time of the frame a read describes, as epoch seconds.

    Frames are named by capture time (20260815T174216Z.jpg); that is what
    matters for freshness, not when the model got round to reading it.
    """
    name = obs.get("frame_name") or ""
    stem = name.rsplit(".", 1)[0]
    try:
        return datetime.strptime(stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        pass
    ra = obs.get("read_at")
    if ra:
        try:
            return datetime.strptime(ra, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def fresh_reads(reads: dict[str, dict], now: float | None = None,
                max_age: int | None = None) -> tuple[dict[str, dict], int]:
    """Split the corpus into reads young enough to describe the present.

    Returns (fresh, stale_count). Fresh reads gain an `_age_s` key.
    """
    now = time.time() if now is None else now
    max_age = config.LIVE_MAX_AGE if max_age is None else max_age
    fresh: dict[str, dict] = {}
    stale = 0
    for cam_id, obs in reads.items():
        if not obs or "parse_error" in obs:
            continue
        t = frame_time(obs)
        if t is None:
            stale += 1
            continue
        age = now - t
        if age > max_age:
            stale += 1
            continue
        fresh[cam_id] = dict(obs, _age_s=int(age))
    return fresh, stale


def nudge_for(obs: dict) -> tuple[float, list[str]]:
    """Additive risk and the human reasons for one read, uncapped."""
    total = 0.0
    reasons: list[str] = []
    for key, (w, label) in NUDGE.items():
        if obs.get(key) is True:
            total += w
            reasons.append(label)
    if obs.get("lighting") == "dark_unlit":
        total += UNLIT_NUDGE[0]
        reasons.append(UNLIT_NUDGE[1])
    return total, reasons


def penalties(G: nx.MultiGraph, reads: dict[str, dict],
              now: float | None = None) -> tuple[dict[str, dict], dict]:
    """Build ({seg_id: {"penalty", "reasons", "cameras"}}, summary).

    Only segments with a non-zero nudge appear, so the router's lookup is a
    miss for the vast majority of edges. The summary is what the API reports so
    the UI can say how much of the corpus was fresh enough to count.
    """
    fresh, stale = fresh_reads(reads, now=now)
    per_cam: dict[str, tuple[float, list[str], int]] = {}
    for cam_id, obs in fresh.items():
        p, reasons = nudge_for(obs)
        if p > 0:
            per_cam[cam_id] = (p, reasons, obs["_age_s"])

    out: dict[str, dict] = {}
    if per_cam:
        for _, _, d in G.edges(data=True):
            hits = [c["id"] for c in d.get("cameras", ()) if c["id"] in per_cam]
            if not hits:
                continue
            total = 0.0
            reasons: list[str] = []
            cams: list[dict] = []
            for cid in hits:
                p, rs, age = per_cam[cid]
                total += p
                for r in rs:
                    if r not in reasons:
                        reasons.append(r)
                cams.append({"id": cid, "age_s": age, "reasons": rs})
            out[d["seg_id"]] = {
                "penalty": round(min(LIVE_CAP, total), 3),
                "reasons": reasons,
                "cameras": cams,
            }

    summary = {
        "reads_total": len(reads),
        "reads_fresh": len(fresh),
        "reads_stale": stale,
        "max_age_s": config.LIVE_MAX_AGE,
        "cameras_flagging": len(per_cam),
        "segments_nudged": len(out),
    }
    return out, summary
