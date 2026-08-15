# Safe Walk

Seattle runs 646 public traffic cameras that nobody watches. We watch all of them,
read every frame with a vision-language model, and route you around what they see.

Two routes for every trip: the shortest, and the one that trades a little distance
for evidence — fewer recorded pedestrian collisions, working sidewalks, and whatever
a camera showed within the last hour. Never a "safety score": every claim is a
number a judge can check.

Full brief (product, demo script, architecture, measured numbers):
[`docs/safe-walk-brief.html`](docs/safe-walk-brief.html).
Latest thinking on the government-facing angle:
[`docs/feedback-nvidia-2026-08-15.md`](docs/feedback-nvidia-2026-08-15.md).

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m safewalk.static_data     # SDOT layers -> data/static (45 MB, once)
python -m safewalk.cameras         # camera inventory -> data/meta/cameras.json
python -m safewalk.routing         # builds + caches the street graph, prints a sample route

# three long-running processes (separate terminals, or nohup ... &):
python -m safewalk.scraper         # every camera every 15 min -> data/frames
python -m safewalk.caption_worker  # VLM reads newest frame per camera -> data/meta/captions.jsonl
python -m uvicorn safewalk.api:app --host 127.0.0.1 --port 8010
# open http://127.0.0.1:8010
```

Env knobs live in `safewalk/config.py` — `SWEEP_INTERVAL` (900 on a Mac, 60 on the
Spark), `SAFEWALK_VLM` (default `mlx-community/Qwen2.5-VL-7B-Instruct-4bit`),
`LIVE_MAX_AGE` (how old a camera read may be and still steer a route, 3600 s).

Before presenting: `scripts/demo_mode.sh on` suspends the corpus worker so a live
read gets the whole GPU (~10 s vs ~21 s contended). `off` resumes it.

## Layout

| Path | What |
|---|---|
| `safewalk/cameras.py` | camera inventory from the city's travelers-map feed |
| `safewalk/scraper.py` | sweep daemon; detects dead-camera placeholder JPEGs |
| `safewalk/vision.py` | VLM prompt + strict JSON schema; forbids danger/crime verdicts |
| `safewalk/caption_worker.py` | reads newest frame per camera, worst-collision blocks first |
| `safewalk/static_data.py` | one-shot pull of SDOT collisions / sidewalks / streets |
| `safewalk/graph.py` | street graph; joins collisions, sidewalks, cameras onto segments |
| `safewalk/routing.py` | static risk per segment; A* over risk-weighted length |
| `safewalk/live.py` | fresh camera reads -> capped per-segment nudges |
| `safewalk/baseline.py` | "busier than N% of downtown right now" |
| `safewalk/api.py` | FastAPI: `/api/route`, `/api/camera/{id}/{latest,frame.jpg,read}` |
| `web/index.html` | hand-rolled SVG map + contact sheet, zero dependencies |
| `scripts/` | `demo_mode.sh`, `dl_molmo2.sh` (Molmo 2 weights -> SSD kit) |
| `docs/` | brief, feedback notes |

Data on disk (`data/`) is gitignored: frames, the read corpus and the SDOT layers
are all rebuilt by the commands above.
