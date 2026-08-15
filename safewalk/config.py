"""Central config. Every knob that changes between the Mac and the Spark lives here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FRAMES = DATA / "frames"
META = DATA / "meta"
LOGS = ROOT / "logs"

for _d in (DATA, FRAMES, META, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# --- feed endpoints -------------------------------------------------------
# zoomId matters more than it looks: the travelers map CLUSTERS cameras for
# display, and PointCoordinate is the cluster centroid, not the camera. At
# zoomId=13 all 22 cameras of a downtown cluster report identical coordinates,
# which silently destroys every spatial join downstream. 18 is the highest zoom
# that still returns data (20 returns nothing) and leaves only genuinely
# co-located pairs -- two cameras on one pole facing different ways.
INVENTORY_URL = "https://web.seattle.gov/travelers/api/map/data?zoomId=18&type=2"
SDOT_IMAGE_BASE = "https://www.seattle.gov/trafficcams/images/"
WSDOT_IMAGE_BASE = "https://images.wsdot.wa.gov/nw/"

# --- the env-var that becomes the Spark story -----------------------------
# Mac: a full 650-camera sweep costs ~20-30 min of VLM time, so refreshing
# faster than that is pointless. Spark: sub-minute sweeps, so drop this to 60.
SWEEP_INTERVAL = int(os.getenv("SWEEP_INTERVAL", "900"))  # seconds

# Politeness: concurrent image fetches against city servers.
FETCH_CONCURRENCY = int(os.getenv("FETCH_CONCURRENCY", "10"))
FETCH_TIMEOUT = float(os.getenv("FETCH_TIMEOUT", "15"))
USER_AGENT = os.getenv(
    "SAFEWALK_UA",
    "safe-walk/0.1 (Spark Hack Seattle 2026 research project; contact adi.bankoti@gmail.com)",
)

# --- retention (61 GB free on the dev Mac, so be disciplined) -------------
# Full-resolution frames kept per camera before pruning to hourly archive.
KEEP_RECENT_PER_CAM = int(os.getenv("KEEP_RECENT_PER_CAM", "24"))  # 6h @ 15min
KEEP_HOURLY_HOURS = int(os.getenv("KEEP_HOURLY_HOURS", "48"))

# --- downtown Seattle bounding box (MVP scope) ----------------------------
DOWNTOWN_BBOX = (47.588, 47.628, -122.355, -122.320)  # lat_min, lat_max, lon_min, lon_max

# --- derived paths --------------------------------------------------------
CAMERAS_JSON = META / "cameras.json"
SWEEPS_JSONL = META / "sweeps.jsonl"
PLACEHOLDER_HASHES = META / "placeholder_hashes.json"

# Live reads and the background corpus worker share one GPU. Measured: a live
# read costs ~10 s alone and ~30 s while the worker is mid-frame. The worker
# yields whenever this file exists.
VLM_PAUSE_LOCK = META / "vlm_pause.lock"

# --- live reads steering routes -------------------------------------------
# A camera read only nudges routing while its *frame* is younger than this.
# On the Mac a caption pass takes ~2.5 h, so most of the corpus is older than
# one sweep by the time it is read; on the Spark every read is minutes old and
# this gate passes everything. The gap between the two is a number we report.
LIVE_MAX_AGE = int(os.getenv("LIVE_MAX_AGE", "3600"))  # seconds

# live.jpg written by /frame.jpg?live=true counts as "the frame we just fetched"
# only this long; after that /read?live=true falls back to the stored frame and
# says so, rather than silently reading a photo from a previous live beat.
LIVE_FRAME_TTL = int(os.getenv("LIVE_FRAME_TTL", "90"))  # seconds
