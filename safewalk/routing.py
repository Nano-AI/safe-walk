"""Walking routes over the street graph, scored by what the data actually says.

Two routes come back for every request: the short one and the one that trades
distance for evidence. Evidence is the collision record plus whatever a camera
showed within the last hour (see live.py); the present tense can tilt a close
call but is capped so it never overrules history. The difference between them is the product -- a single
"safety score" is a number a judge cannot check, whereas "this block has 32
pedestrian collisions and 11 cameras watching it" is a fact they can.

Deliberately NOT modelled: crime. The claim we make is limited to what the
collision record, the sidewalk inventory, and a live camera can support.
"""
from __future__ import annotations

import math
from heapq import heappop, heappush

import networkx as nx

from . import graph as gmod

# Risk term weights. These sum to 1.0 so `risk` stays in [0, 1].
W_PED = 0.40        # pedestrian/cyclist collision density
W_SERIOUS = 0.15    # killed or seriously injured
W_SIDEWALK = 0.20   # missing or unusable sidewalk
W_ARTERIAL = 0.15   # traffic exposure: arterial class and posted speed
W_SLOPE = 0.10      # Seattle hills: a 12% grade is a real walking cost

# A block with this many pedestrian collisions per 100 m is treated as maxed out.
PED_PER_100M_CAP = 6.0
WALK_SPEED_MPS = 1.35  # ~3 mph


def static_risk(d: dict) -> tuple[float, dict]:
    """Score one segment in [0, 1] and return the component breakdown."""
    length = max(d["length_m"], 10.0)
    per100 = d["ped_collisions"] / (length / 100.0)
    ped = min(1.0, per100 / PED_PER_100M_CAP)

    serious = min(1.0, d["serious_collisions"] / 3.0)
    sidewalk = 1.0 - min(1.0, d["sidewalk_ratio"])
    if d["sidewalk_condition"] in ("POOR", "VERYPOOR"):
        sidewalk = min(1.0, sidewalk + 0.25)

    arterial = min(1.0, (d["artclass"] / 5.0) * 0.6 + (d["speed_limit"] / 40.0) * 0.4)
    slope = min(1.0, d["slope_pct"] / 12.0)

    risk = (
        W_PED * ped
        + W_SERIOUS * serious
        + W_SIDEWALK * sidewalk
        + W_ARTERIAL * arterial
        + W_SLOPE * slope
    )
    return round(risk, 4), {
        "pedestrian_collisions": round(ped, 3),
        "serious_injuries": round(serious, 3),
        "sidewalk_gap": round(sidewalk, 3),
        "traffic_exposure": round(arterial, 3),
        "slope": round(slope, 3),
    }


def annotate(G: nx.MultiGraph) -> nx.MultiGraph:
    """Precompute the static risk term on every edge, once."""
    for _, _, d in G.edges(data=True):
        d["risk"], d["risk_parts"] = static_risk(d)
    return G


def nearest_node(G: nx.MultiGraph, lat: float, lon: float):
    x, y = gmod.to_xy(lon, lat)
    best, best_d = None, float("inf")
    for n, nd in G.nodes(data=True):
        if "x" not in nd:
            continue
        dist = (nd["x"] - x) ** 2 + (nd["y"] - y) ** 2
        if dist < best_d:
            best, best_d = n, dist
    return best, math.sqrt(best_d)


def _live_penalty(d: dict, live: dict | None) -> float:
    if not live:
        return 0.0
    hit = live.get(d["seg_id"])
    return hit["penalty"] if hit else 0.0


def _edge_cost(d: dict, risk_weight: float, live: dict | None) -> float:
    """Metres of 'effective walking' -- real distance inflated by risk.

    `live` is the dict built by live.penalties(): fresh camera reads nudge a
    segment upward, capped so static history still dominates.
    """
    risk = min(1.0, d["risk"] + _live_penalty(d, live))
    return d["length_m"] * (1.0 + risk_weight * risk)


def _astar(G, source, target, risk_weight, live):
    """A* with a straight-line-distance heuristic, admissible because the
    heuristic ignores the risk multiplier that only ever increases cost."""
    tx, ty = G.nodes[target]["x"], G.nodes[target]["y"]

    def h(n):
        nd = G.nodes[n]
        return math.hypot(nd["x"] - tx, nd["y"] - ty)

    open_heap = [(h(source), 0.0, source, None)]
    came: dict = {}
    best_g: dict = {source: 0.0}
    seen: set = set()

    while open_heap:
        _, g, node, parent = heappop(open_heap)
        if node in seen:
            continue
        seen.add(node)
        came[node] = parent
        if node == target:
            break
        for nbr, edges in G.adj[node].items():
            if nbr in seen:
                continue
            cheapest = min(edges.values(), key=lambda d: _edge_cost(d, risk_weight, live))
            ng = g + _edge_cost(cheapest, risk_weight, live)
            if ng < best_g.get(nbr, float("inf")):
                best_g[nbr] = ng
                heappush(open_heap, (ng + h(nbr), ng, nbr, (node, cheapest)))

    if target not in came:
        return None

    path = []
    cur = target
    while came.get(cur) is not None:
        prev, edge = came[cur]
        path.append(edge)
        cur = prev
    path.reverse()
    return path


def route(
    G: nx.MultiGraph,
    start: tuple[float, float],
    end: tuple[float, float],
    risk_weight: float = 0.0,
    live: dict | None = None,
) -> dict | None:
    """Route between two (lat, lon) pairs. risk_weight=0 is shortest path."""
    s, _ = nearest_node(G, *start)
    t, _ = nearest_node(G, *end)
    if s is None or t is None or s == t:
        return None

    path = _astar(G, s, t, risk_weight, live)
    if not path:
        return None

    segments = []
    total_m = 0.0
    for d in path:
        total_m += d["length_m"]
        segments.append(
            {
                "seg_id": d["seg_id"],
                "name": d["name"],
                "from": d["xstr_lo"],
                "to": d["xstr_hi"],
                "length_m": d["length_m"],
                "risk": d["risk"],
                "risk_parts": d["risk_parts"],
                "ped_collisions": d["ped_collisions"],
                "serious_collisions": d["serious_collisions"],
                "sidewalk_ratio": d["sidewalk_ratio"],
                "sidewalk_condition": d["sidewalk_condition"],
                "slope_pct": d["slope_pct"],
                "speed_limit": d["speed_limit"],
                "cameras": d["cameras"],
                "geometry": d["geometry"],
                "live": (live or {}).get(d["seg_id"]),
            }
        )

    ped_total = sum(s["ped_collisions"] for s in segments)
    serious_total = sum(s["serious_collisions"] for s in segments)
    cam_ids = {c["id"] for s in segments for c in s["cameras"]}
    weighted_risk = (
        sum(s["risk"] * s["length_m"] for s in segments) / total_m if total_m else 0.0
    )
    no_sidewalk_m = sum(s["length_m"] * (1 - s["sidewalk_ratio"]) for s in segments)
    # What the cameras added, in walking order, so the UI can name it.
    live_notes = [
        {
            "seg_id": s["seg_id"],
            "name": s["name"],
            "from": s["from"],
            "to": s["to"],
            "penalty": s["live"]["penalty"],
            "reasons": s["live"]["reasons"],
            "cameras": s["live"]["cameras"],
        }
        for s in segments if s["live"]
    ]

    return {
        "risk_weight": risk_weight,
        "distance_m": round(total_m),
        "walk_minutes": round(total_m / WALK_SPEED_MPS / 60, 1),
        "mean_risk": round(weighted_risk, 4),
        "ped_collisions": ped_total,
        "serious_collisions": serious_total,
        "cameras": sorted(cam_ids),
        "camera_count": len(cam_ids),
        "no_sidewalk_m": round(no_sidewalk_m),
        "live_notes": live_notes,
        "segments": segments,
    }


def compare(G, start, end, safe_weight: float = 3.0, live: dict | None = None) -> dict:
    """The demo call: the direct route and the evidence-led alternative."""
    direct = route(G, start, end, risk_weight=0.0, live=live)
    safer = route(G, start, end, risk_weight=safe_weight, live=live)
    return {"direct": direct, "safer": safer}


if __name__ == "__main__":
    import json
    import time

    G = annotate(gmod.load_or_build())
    gmod.save(G)

    # Westlake Center -> Pioneer Square, straight through the retail core.
    START = (47.6115, -122.3376)
    END = (47.6015, -122.3343)

    t0 = time.time()
    res = compare(G, START, END)
    elapsed = (time.time() - t0) * 1000

    for label in ("direct", "safer"):
        r = res[label]
        if not r:
            print(f"{label}: NO ROUTE")
            continue
        print(
            f"\n{label.upper():7} {r['distance_m']:5} m  {r['walk_minutes']:4} min  "
            f"risk={r['mean_risk']:.3f}  ped={r['ped_collisions']:3}  "
            f"serious={r['serious_collisions']}  cams={r['camera_count']}  "
            f"segments={len(r['segments'])}"
        )
        for s in r["segments"][:6]:
            print(f"    {s['name'][:16]:16} {s['from'][:12]:12}->{s['to'][:12]:12} "
                  f"{s['length_m']:6.0f}m risk={s['risk']:.2f} ped={s['ped_collisions']:2} "
                  f"cams={len(s['cameras'])}")

    d, sf = res["direct"], res["safer"]
    if d and sf:
        print(f"\ndetour: +{sf['distance_m'] - d['distance_m']} m "
              f"({(sf['distance_m'] / d['distance_m'] - 1):+.0%}), "
              f"ped exposure {d['ped_collisions']} -> {sf['ped_collisions']}, "
              f"risk {d['mean_risk']:.3f} -> {sf['mean_risk']:.3f}")
    print(f"\nboth routes computed in {elapsed:.0f} ms")
