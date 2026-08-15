"""Camera inventory: pull the travelers-map feed and normalize it.

The City of Seattle travelers map is the only public index of these cameras --
data.seattle.gov has no camera dataset (searching it returns Calgary and Austin).
Feed shape:

    {"Features": [
        {"PointCoordinate": [47.52, -122.39],
         "Cameras": [{"Id": "CMR-0112",
                      "Description": "Fauntleroy Way SW & SW Cloverdale St",
                      "ImageUrl": "Fauntleroy_SW_Cloverdale_NS.jpg",
                      "Type": "sdot"}]}]}
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import httpx

from . import config


@dataclass(frozen=True)
class Camera:
    id: str
    description: str
    image_url: str
    source: str  # "sdot" | "wsdot"
    lat: float
    lon: float

    @property
    def downtown(self) -> bool:
        lat_min, lat_max, lon_min, lon_max = config.DOWNTOWN_BBOX
        return lat_min <= self.lat <= lat_max and lon_min <= self.lon <= lon_max


def _image_url(raw: str, source: str) -> str:
    if raw.startswith("http"):
        return raw
    base = config.WSDOT_IMAGE_BASE if source == "wsdot" else config.SDOT_IMAGE_BASE
    return base + raw


def fetch_inventory() -> list[Camera]:
    """Hit the live feed and return every camera it advertises."""
    headers = {"User-Agent": config.USER_AGENT}
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        payload = client.get(config.INVENTORY_URL).json()

    cams: dict[str, Camera] = {}
    for feature in payload.get("Features", []):
        coord = feature.get("PointCoordinate") or [None, None]
        lat, lon = coord[0], coord[1]
        if lat is None or lon is None:
            continue
        for entry in feature.get("Cameras", []):
            raw_url = entry.get("ImageUrl")
            cam_id = entry.get("Id")
            if not raw_url or not cam_id:
                continue
            source = (entry.get("Type") or "sdot").lower()
            # Duplicate ids exist in the feed; last writer wins, which is fine
            # since the duplicates carry identical image urls.
            cams[cam_id] = Camera(
                id=cam_id,
                description=(entry.get("Description") or "").strip(),
                image_url=_image_url(raw_url, source),
                source=source,
                lat=float(lat),
                lon=float(lon),
            )
    return sorted(cams.values(), key=lambda c: c.id)


def save(cams: list[Camera]) -> None:
    config.CAMERAS_JSON.write_text(
        json.dumps([asdict(c) for c in cams], indent=2), encoding="utf-8"
    )


def load() -> list[Camera]:
    raw = json.loads(config.CAMERAS_JSON.read_text(encoding="utf-8"))
    return [Camera(**r) for r in raw]


def load_or_fetch() -> list[Camera]:
    if config.CAMERAS_JSON.exists():
        return load()
    cams = fetch_inventory()
    save(cams)
    return cams


if __name__ == "__main__":
    cameras = fetch_inventory()
    save(cameras)
    downtown = [c for c in cameras if c.downtown]
    by_source: dict[str, int] = {}
    for c in cameras:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    print(f"cameras: {len(cameras)}  {by_source}")
    print(f"downtown bbox: {len(downtown)}")
    print(f"written: {config.CAMERAS_JSON}")
