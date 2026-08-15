"""Pull the static SDOT layers once, to disk, so the demo never needs the network.

Everything here is downtown-bbox scoped. Citywide is 261k collisions and 46k
sidewalk segments; downtown is 52k and 3.1k, which fits in memory and keeps the
routing graph small enough to answer in milliseconds.

Source: SDOT's public ArcGIS FeatureServer. No key, no auth.
"""
from __future__ import annotations

import json
import time

import httpx

from . import config

ARCGIS_BASE = "https://services.arcgis.com/ZOyb2t4B0UYuYNYH/ArcGIS/rest/services"
STATIC = config.DATA / "static"
STATIC.mkdir(parents=True, exist_ok=True)

# Only the collision layer needs trimming; the others are small enough for "*".
COLLISION_FIELDS = [
    "OBJECTID", "INCKEY", "ADDRTYPE", "LOCATION", "PERSONCOUNT", "PEDCOUNT",
    "PEDCYLCOUNT", "VEHCOUNT", "INJURIES", "SERIOUSINJURIES", "FATALITIES",
    "INCDATE", "INCDTTM", "JUNCTIONTYPE", "WEATHER", "ROADCOND", "LIGHTCOND",
    "PEDROWNOTGRNT", "SPEEDING", "MAXSEVERITYCODE", "MAXSEVERITYDESC",
    "CROSSWALKKEY", "SEGLANEKEY", "ADDR_SEGINTKEY",
]

LAYERS: dict[str, dict] = {
    "streets": {"service": "Seattle_Streets_1", "fields": "*"},
    "sidewalks": {"service": "Sidewalks_(Active)", "fields": "*"},
    "collisions": {"service": "SDOT_Collisions_All_Years_1", "fields": ",".join(COLLISION_FIELDS)},
    "high_collision": {"service": "HighCollisionLocations2021", "fields": "*"},
}

PAGE = 2000


def _envelope() -> str:
    lat_min, lat_max, lon_min, lon_max = config.DOWNTOWN_BBOX
    return json.dumps(
        {
            "xmin": lon_min,
            "ymin": lat_min,
            "xmax": lon_max,
            "ymax": lat_max,
            "spatialReference": {"wkid": 4326},
        }
    )


def fetch_layer(key: str, spec: dict, client: httpx.Client) -> dict:
    """Page through one layer's downtown features and return a GeoJSON dict."""
    url = f"{ARCGIS_BASE}/{spec['service']}/FeatureServer/0/query"
    base_params = {
        "where": "1=1",
        "outFields": spec["fields"],
        "outSR": "4326",
        "f": "geojson",
        "geometry": _envelope(),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "resultRecordCount": str(PAGE),
    }

    features: list[dict] = []
    offset = 0
    while True:
        params = dict(base_params, resultOffset=str(offset))
        resp = client.get(url, params=params)
        resp.raise_for_status()
        payload = resp.json()
        if "error" in payload:
            raise RuntimeError(f"{key}: {payload['error']}")
        batch = payload.get("features", [])
        features.extend(batch)
        print(f"  {key}: {len(features)}", flush=True)
        if len(batch) < PAGE or not payload.get("properties", {}).get("exceededTransferLimit", len(batch) == PAGE):
            if len(batch) < PAGE:
                break
        offset += PAGE
        time.sleep(0.2)  # be polite to the city's server

    return {"type": "FeatureCollection", "features": features}


def pull_all() -> dict[str, int]:
    counts: dict[str, int] = {}
    with httpx.Client(timeout=90, headers={"User-Agent": config.USER_AGENT}) as client:
        for key, spec in LAYERS.items():
            out = STATIC / f"{key}.geojson"
            if out.exists():
                existing = json.loads(out.read_text())
                counts[key] = len(existing.get("features", []))
                print(f"  {key}: cached ({counts[key]})", flush=True)
                continue
            print(f"fetching {key} <- {spec['service']}", flush=True)
            fc = fetch_layer(key, spec, client)
            out.write_text(json.dumps(fc))
            counts[key] = len(fc["features"])
            print(f"  {key}: {counts[key]} -> {out.name} "
                  f"({out.stat().st_size / 1e6:.1f} MB)", flush=True)
    return counts


if __name__ == "__main__":
    print(json.dumps(pull_all(), indent=2))
