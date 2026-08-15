"""Build the walkable street graph and attach every static risk layer to it.

Downtown is small enough (1.7k segments) that we can project to a local metric
plane with a flat-earth approximation and skip pyproj entirely. Error over a 4 km
extent is well under a metre, which is far below the accuracy of the snapping
tolerances anyway.

Everything a route needs to explain itself is baked onto the edge: the street
name, the two cross streets, pedestrian collision history, sidewalk presence and
condition, slope, and which cameras can see it.
"""
from __future__ import annotations

import json
import math
import pickle
from collections import defaultdict

import networkx as nx
from shapely.geometry import LineString, Point, shape
from shapely.strtree import STRtree

from . import cameras as cameras_mod
from . import config

STATIC = config.DATA / "static"
GRAPH_PICKLE = config.META / "graph.pkl"

# Local projection origin: centre of the downtown bbox.
LAT0 = (config.DOWNTOWN_BBOX[0] + config.DOWNTOWN_BBOX[1]) / 2
LON0 = (config.DOWNTOWN_BBOX[2] + config.DOWNTOWN_BBOX[3]) / 2
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))

# Snapping tolerances, metres.
NODE_SNAP = 12.0        # endpoints closer than this are the same intersection
COLLISION_SNAP = 25.0
SIDEWALK_SNAP = 22.0
CAMERA_SNAP = 80.0   # a mast-mounted camera sees roughly this far down each approach


def to_xy(lon: float, lat: float) -> tuple[float, float]:
    return ((lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT)


def to_lonlat(x: float, y: float) -> tuple[float, float]:
    return (x / M_PER_DEG_LON + LON0, y / M_PER_DEG_LAT + LAT0)


def _project(geom) -> LineString | None:
    """Project a GeoJSON LineString/MultiLineString into local metres."""
    g = shape(geom)
    if g.geom_type == "MultiLineString":
        parts = list(g.geoms)
        if not parts:
            return None
        g = max(parts, key=lambda p: p.length)
    if g.geom_type != "LineString" or g.is_empty:
        return None
    return LineString([to_xy(x, y) for x, y in g.coords])


def _load(name: str) -> list[dict]:
    return json.loads((STATIC / f"{name}.geojson").read_text())["features"]


def _node_key(x: float, y: float) -> tuple[int, int]:
    """Quantise a coordinate so nearby endpoints collapse to one intersection."""
    return (round(x / NODE_SNAP), round(y / NODE_SNAP))


def build() -> nx.MultiGraph:
    print("loading streets...", flush=True)
    street_feats = _load("streets")

    geoms: list[LineString] = []
    props: list[dict] = []
    for f in street_feats:
        line = _project(f["geometry"])
        if line is None or line.length < 1.0:
            continue
        geoms.append(line)
        props.append(f["properties"])

    print(f"  {len(geoms)} usable segments", flush=True)
    tree = STRtree(geoms)

    # --- attach pedestrian collision history ------------------------------
    print("snapping collisions...", flush=True)
    ped = defaultdict(int)
    allc = defaultdict(int)
    serious = defaultdict(int)
    night = defaultdict(int)
    for f in _load("collisions"):
        g = f.get("geometry")
        if not g or g.get("type") != "Point":
            continue
        lon, lat = g["coordinates"][:2]
        pt = Point(*to_xy(lon, lat))
        idx = tree.query_nearest(pt, max_distance=COLLISION_SNAP, return_distance=False)
        if len(idx) == 0:
            continue
        i = int(idx[0])
        p = f["properties"]
        allc[i] += 1
        if (p.get("PEDCOUNT") or 0) > 0 or (p.get("PEDCYLCOUNT") or 0) > 0:
            ped[i] += 1
        if (p.get("SERIOUSINJURIES") or 0) > 0 or (p.get("FATALITIES") or 0) > 0:
            serious[i] += 1
        light = (p.get("LIGHTCOND") or "").lower()
        if "dark" in light:
            night[i] += 1

    # --- attach sidewalk coverage -----------------------------------------
    print("snapping sidewalks...", flush=True)
    sw_len = defaultdict(float)
    sw_cond: dict[int, set[str]] = defaultdict(set)
    sw_width: dict[int, list[float]] = defaultdict(list)
    for f in _load("sidewalks"):
        line = _project(f["geometry"])
        if line is None:
            continue
        p = f["properties"]
        if (p.get("CURRENT_STATUS") or "INSVC") != "INSVC":
            continue
        mid = line.interpolate(0.5, normalized=True)
        idx = tree.query_nearest(mid, max_distance=SIDEWALK_SNAP, return_distance=False)
        if len(idx) == 0:
            continue
        i = int(idx[0])
        sw_len[i] += line.length
        if p.get("CONDITION"):
            sw_cond[i].add(str(p["CONDITION"]).upper())
        if p.get("SW_WIDTH"):
            try:
                sw_width[i].append(float(p["SW_WIDTH"]) / 12.0)  # inches -> feet
            except (TypeError, ValueError):
                pass

    # --- attach cameras ----------------------------------------------------
    # A camera mounted at an intersection watches every approach to it, not just
    # the one segment its pin happens to land on. Assign it to everything inside
    # its sight radius and keep the distance so the UI can rank by closeness.
    print("snapping cameras...", flush=True)
    seg_cams: dict[int, list[dict]] = defaultdict(list)
    cam_segs: dict[str, list[int]] = defaultdict(list)
    for cam in cameras_mod.load_or_fetch():
        pt = Point(*to_xy(cam.lon, cam.lat))
        for raw in tree.query(pt.buffer(CAMERA_SNAP)):
            i = int(raw)
            dist = geoms[i].distance(pt)
            if dist > CAMERA_SNAP:
                continue
            seg_cams[i].append({"id": cam.id, "dist_m": round(dist, 1)})
            cam_segs[cam.id].append(i)
    for i in seg_cams:
        seg_cams[i].sort(key=lambda c: c["dist_m"])

    # --- assemble the graph -------------------------------------------------
    print("assembling graph...", flush=True)
    G = nx.MultiGraph()
    node_pos: dict[tuple[int, int], tuple[float, float]] = {}

    for i, (line, p) in enumerate(zip(geoms, props)):
        coords = list(line.coords)
        a, b = _node_key(*coords[0]), _node_key(*coords[-1])
        if a == b:
            continue  # degenerate loop segment
        node_pos.setdefault(a, coords[0])
        node_pos.setdefault(b, coords[-1])

        length = line.length
        sidewalk_ratio = min(1.0, sw_len[i] / max(length, 1.0)) if length else 0.0
        conds = sw_cond.get(i, set())
        worst = None
        for rank in ("VERYPOOR", "POOR", "FAIR", "GOOD", "EXCELLENT"):
            if rank in conds:
                worst = rank
                break

        G.add_edge(
            a,
            b,
            key=i,
            seg_id=i,
            compkey=p.get("COMPKEY"),
            name=(p.get("STNAME_ORD") or "").title() or "Unnamed",
            xstr_lo=(p.get("XSTRLO") or "").title(),
            xstr_hi=(p.get("XSTRHI") or "").title(),
            length_m=round(length, 1),
            artclass=p.get("ARTCLASS") or 0,
            speed_limit=p.get("SPEEDLIMIT") or 0,
            slope_pct=abs(p.get("SLOPE_PCT") or 0),
            ped_collisions=ped.get(i, 0),
            all_collisions=allc.get(i, 0),
            serious_collisions=serious.get(i, 0),
            dark_collisions=night.get(i, 0),
            sidewalk_ratio=round(sidewalk_ratio, 2),
            sidewalk_condition=worst,
            sidewalk_width_ft=round(min(sw_width[i]), 1) if sw_width.get(i) else None,
            cameras=seg_cams.get(i, []),
            geometry=[to_lonlat(x, y) for x, y in coords],
        )

    for key, (x, y) in node_pos.items():
        if key in G:
            lon, lat = to_lonlat(x, y)
            G.nodes[key]["x"] = x
            G.nodes[key]["y"] = y
            G.nodes[key]["lon"] = lon
            G.nodes[key]["lat"] = lat

    G.graph["camera_segments"] = dict(cam_segs)
    return G


def save(G: nx.MultiGraph) -> None:
    GRAPH_PICKLE.write_bytes(pickle.dumps(G))


def load() -> nx.MultiGraph:
    return pickle.loads(GRAPH_PICKLE.read_bytes())


def load_or_build() -> nx.MultiGraph:
    if GRAPH_PICKLE.exists():
        return load()
    G = build()
    save(G)
    return G


if __name__ == "__main__":
    G = build()
    save(G)
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    edges = list(G.edges(data=True))
    with_cam = sum(1 for *_, d in edges if d["cameras"])
    with_ped = sum(1 for *_, d in edges if d["ped_collisions"])
    no_walk = sum(1 for *_, d in edges if d["sidewalk_ratio"] < 0.25)
    print(f"\nnodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print(f"components={len(comps)} largest={len(comps[0])} "
          f"({len(comps[0]) / G.number_of_nodes():.1%} of nodes)")
    print(f"edges with a camera:       {with_cam}")
    print(f"edges with ped collisions: {with_ped}")
    print(f"edges with ~no sidewalk:   {no_walk}")
    worst = sorted(edges, key=lambda e: -e[2]["ped_collisions"])[:8]
    print("\nworst pedestrian-collision segments:")
    for _, _, d in worst:
        print(f"  {d['ped_collisions']:3} ped  {d['name']:22} "
              f"{d['xstr_lo']} -> {d['xstr_hi']}  cams={len(d['cameras'])}")
    print(f"\nsaved {GRAPH_PICKLE}")
