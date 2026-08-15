"""Per-camera baselines: is this corner busier than it normally is?

The honest scope of this feature is set by how much history exists. With one
day of archive we can say "busier than this corner has been all day". We
deliberately do NOT say "unusual for a Sunday afternoon" -- that needs weeks,
and claiming it from ten hours of data is the kind of thing a judge catches.

Every answer carries the sample size it was computed from, so the UI can show
its own confidence rather than asserting one.
"""
from __future__ import annotations

import json
from collections import defaultdict

from . import config

CAPTIONS = config.META / "captions.jsonl"

# Ordinal scales. "stopped" sits with "moderate": a stopped queue is congestion,
# not an empty street, and the VLM uses it for signal-held traffic.
TRAFFIC = {"none": 0, "light": 1, "moderate": 2, "stopped": 2, "heavy": 3}
CROWDING = {"none": 0, "light": 1, "moderate": 2, "heavy": 3}

MIN_SAMPLES = 6


def _load_history() -> dict[str, list[dict]]:
    hist: dict[str, list[dict]] = defaultdict(list)
    if not CAPTIONS.exists():
        return hist
    with CAPTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "parse_error" in r or "cam_id" not in r:
                continue
            hist[r["cam_id"]].append(r)
    return hist


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


class Baselines:
    """Snapshot of every camera's own typical state, rebuilt on demand."""

    def __init__(self) -> None:
        self.stats: dict[str, dict] = {}
        self.rebuild()

    def rebuild(self) -> None:
        stats: dict[str, dict] = {}
        for cam_id, reads in _load_history().items():
            traffic = [TRAFFIC[r["traffic"]] for r in reads if r.get("traffic") in TRAFFIC]
            crowd = [CROWDING[r["crowding"]] for r in reads if r.get("crowding") in CROWDING]
            people = [r["people_visible"] for r in reads
                      if isinstance(r.get("people_visible"), int)]
            if len(traffic) < MIN_SAMPLES:
                continue
            stats[cam_id] = {
                "n": len(reads),
                "traffic_median": _median(traffic),
                "traffic_max": max(traffic),
                "crowd_median": _median(crowd) if crowd else 0.0,
                "people_median": _median(people) if people else 0.0,
                "people_max": max(people) if people else 0,
                "blocked_rate": sum(1 for r in reads if r.get("sidewalk_blocked")) / len(reads),
            }
        self.stats = stats

    def rank_now(self, latest: dict[str, dict]) -> dict[str, dict]:
        """Rank every camera against every other camera *right now*.

        This is the claim the archive genuinely supports. Comparing a corner to
        its own past needs a baseline built from comparable hours, and ten hours
        of one day gives a baseline dominated by 3 a.m. -- against which almost
        anything at 10 a.m. reads as "busier than usual", which is a signal that
        sounds insightful and means nothing.

        Comparing every camera to every other camera in the same sweep has no
        such problem: the sun, the weather and the day of week are shared by the
        whole city, so they cancel out, and what is left is the difference
        between this corner and the rest of downtown at this moment.
        """
        scored: list[tuple[str, float]] = []
        for cam_id, obs in latest.items():
            if not obs or "parse_error" in obs:
                continue
            t = TRAFFIC.get(obs.get("traffic"))
            c = CROWDING.get(obs.get("crowding"))
            p = obs.get("people_visible") if isinstance(obs.get("people_visible"), int) else 0
            if t is None and c is None:
                continue
            # People carry the most pedestrian meaning, so they weigh heaviest.
            activity = (t or 0) + 1.5 * (c or 0) + min(p, 12) * 0.5
            scored.append((cam_id, activity))

        if len(scored) < 8:
            return {}
        scored.sort(key=lambda kv: kv[1])
        n = len(scored)
        out: dict[str, dict] = {}
        for i, (cam_id, activity) in enumerate(scored):
            pct = round(100 * i / (n - 1))
            out[cam_id] = {"activity": round(activity, 2), "percentile": pct, "of": n}
        return out

    def deviation(self, cam_id: str, obs: dict | None) -> dict | None:
        """Flag what this corner's own record says, without overclaiming."""
        base = self.stats.get(cam_id)
        if not base or not obs:
            return None

        notes: list[str] = []
        level = "typical"

        # One signal decides the level, so the label can never contradict its
        # own note. Traffic is the most reliably read of the three.
        t_now = TRAFFIC.get(obs.get("traffic"))
        if t_now is not None:
            delta = t_now - base["traffic_median"]
            if delta >= 1.5:
                level = "much_busier"
                notes.append("traffic much heavier than this corner's own record")
            elif delta >= 1:
                level = "busier"
                notes.append("traffic heavier than this corner's own record")
            elif delta <= -1:
                level = "quieter"
                notes.append("quieter than this corner's own record")

        p_now = obs.get("people_visible")
        if isinstance(p_now, int) and p_now >= max(3, base["people_median"] * 3):
            notes.append(f"{p_now} people in frame against a typical {base['people_median']:.0f}")

        # A blockage on a camera that is almost never blocked is worth saying;
        # one that is always blocked is a standing condition, not news.
        if obs.get("sidewalk_blocked") and base["blocked_rate"] < 0.25:
            notes.append("walking path blocked, which is not usual here")

        return {
            "level": level,
            "notes": notes,
            "samples": base["n"],
            "window_hours": 10,
            "traffic_median": base["traffic_median"],
        }


_SINGLETON: Baselines | None = None
_BUILT_FROM: float = -1.0  # captions.jsonl mtime the singleton reflects


def get(refresh: bool = False) -> Baselines:
    """Shared instance. `refresh` re-reads the corpus only if it has grown --
    a full parse costs ~15 ms and would otherwise land on every route request."""
    global _SINGLETON, _BUILT_FROM
    mtime = CAPTIONS.stat().st_mtime if CAPTIONS.exists() else 0.0
    if _SINGLETON is None:
        _SINGLETON = Baselines()
        _BUILT_FROM = mtime
    elif refresh and mtime != _BUILT_FROM:
        _SINGLETON.rebuild()
        _BUILT_FROM = mtime
    return _SINGLETON


if __name__ == "__main__":
    from .caption_worker import LATEST

    b = get()
    latest = json.loads(LATEST.read_text()) if LATEST.exists() else {}
    print(f"baselines built for {len(b.stats)} cameras "
          f"(min {MIN_SAMPLES} samples)\n")

    ranks = b.rank_now(latest)
    print(f"ranked {len(ranks)} cameras against each other, right now\n")
    top = sorted(ranks.items(), key=lambda kv: -kv[1]["percentile"])[:10]
    print("BUSIEST CORNERS IN DOWNTOWN RIGHT NOW")
    for cam_id, r in top:
        o = latest[cam_id]
        print(f"  p{r['percentile']:<3} {cam_id:11} traffic={o.get('traffic'):9} "
              f"people={o.get('people_visible'):<3} crowd={o.get('crowding'):8} "
              f"{(o.get('notable') or '')[:44]}")
    quiet = sorted(ranks.items(), key=lambda kv: kv[1]["percentile"])[:3]
    print("\nQUIETEST")
    for cam_id, r in quiet:
        o = latest[cam_id]
        print(f"  p{r['percentile']:<3} {cam_id:11} traffic={o.get('traffic'):9} "
              f"people={o.get('people_visible')}")
    ctr = [(c,d) for c,o in latest.items() if (d:=b.deviation(c,o)) and d['level']!='typical'
           and any('quieter' in n for n in d['notes']) and 'busier' in d['level']]
    print(f"\ncontradictory labels (must be 0): {len(ctr)}")
