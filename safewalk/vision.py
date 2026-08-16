"""VLM read of a single camera frame.

Design rule that the whole product rests on: the model reports what is visible,
never a verdict. It may say "the near sidewalk is blocked by a parked truck".
It may not say "this street is unsafe". Safety judgements are the user's, made
from evidence we show them, because a model's opinion about danger is exactly
the claim we cannot defend and the collision record already answers better.

The Spark swap lives behind `caption_frame` / `caption_frames`: same call, same
prompt, same schema, different backend. `SAFEWALK_VLM_BACKEND=mlx` (Mac, batch 1)
or `ollama` (the GB10 box, N concurrent requests against a local ollama server).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from PIL import Image

from . import config

BACKEND = os.getenv("SAFEWALK_VLM_BACKEND", "mlx")  # "mlx" | "ollama"
_DEFAULT_MODEL = {"mlx": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit", "ollama": "qwen2.5vl:7b"}
MODEL_ID = os.getenv("SAFEWALK_VLM", _DEFAULT_MODEL.get(BACKEND, _DEFAULT_MODEL["mlx"]))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
# How many frames to keep in flight. MLX is one model, one stream; ollama on the
# box serves several requests at once (OLLAMA_NUM_PARALLEL on the server side).
CONCURRENCY = int(os.getenv("SAFEWALK_VLM_CONCURRENCY", "1" if BACKEND == "mlx" else "4"))
# 1920x1080 is far more pixels than this task needs; the long edge dominates
# prefill cost. 1024 keeps sidewalks and crowds legible for a fraction of it.
MAX_EDGE = int(os.getenv("SAFEWALK_VLM_EDGE", "1024"))
MAX_TOKENS = int(os.getenv("SAFEWALK_VLM_TOKENS", "256"))  # below ~128 the JSON truncates

PROMPT = """You are reading a fixed traffic camera in Seattle. Report only what is visibly in frame.

Answer with JSON and nothing else:
{
  "lighting": "daylight" | "dusk" | "dark_lit" | "dark_unlit",
  "weather_surface": "dry" | "wet" | "raining" | "snow" | "unclear",
  "people_visible": integer,
  "crowding": "none" | "light" | "moderate" | "heavy",
  "traffic": "none" | "light" | "moderate" | "heavy" | "stopped",
  "sidewalk_blocked": true | false,
  "construction": true | false,
  "emergency_activity": true | false,
  "notable": "one short clause naming anything a person walking here would want to know, or empty string"
}

Rules: count only people you can actually see. "sidewalk_blocked" means a
scaffold, vehicle, barrier or debris is on the walking path. "emergency_activity"
means visible emergency vehicles or flashing lights. Never guess at danger,
crime, or anything outside the frame."""

_MODEL = None
_PROCESSOR = None
_CONFIG = None


def _ensure_loaded():
    """Load (mlx) or warm (ollama) the model. Idempotent; called at API boot."""
    global _MODEL, _PROCESSOR, _CONFIG
    if BACKEND == "ollama":
        if _MODEL is None:
            import httpx

            # An empty generate loads the weights and keeps them resident, so the
            # first real read on stage is not the slow one.
            r = httpx.post(f"{OLLAMA_URL}/api/generate",
                           json={"model": MODEL_ID, "prompt": "", "keep_alive": "24h"},
                           timeout=300)
            r.raise_for_status()
            _MODEL = MODEL_ID
        return _MODEL, None, None
    if _MODEL is None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        _MODEL, _PROCESSOR = load(MODEL_ID)
        _CONFIG = load_config(MODEL_ID)
    return _MODEL, _PROCESSOR, _CONFIG


def _downscale(path: Path) -> Path:
    """Shrink to MAX_EDGE and cache next to the original."""
    out = path.with_suffix(f".s{MAX_EDGE}.jpg")
    if out.exists():
        return out
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) > MAX_EDGE:
            scale = MAX_EDGE / max(im.size)
            im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
        im.save(out, quality=88)
    return out


_JSON_RE = re.compile(r"\{.*\}", re.S)


def _parse(text: str) -> dict:
    match = _JSON_RE.search(text)
    if not match:
        return {"parse_error": text[:200]}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # Models occasionally trail a comma; one cheap repair beats a retry.
        repaired = re.sub(r",\s*([}\]])", r"\1", match.group(0))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            return {"parse_error": text[:200]}


def _generate_mlx(small: Path) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, cfg = _ensure_loaded()
    formatted = apply_chat_template(processor, cfg, PROMPT, num_images=1)
    result = generate(model, processor, formatted, [str(small)],
                      max_tokens=MAX_TOKENS, temperature=0.0, verbose=False)
    return result.text if hasattr(result, "text") else str(result)


def _generate_ollama(small: Path) -> str:
    import base64

    import httpx

    _ensure_loaded()
    payload = {
        "model": MODEL_ID,
        "prompt": PROMPT,
        "images": [base64.b64encode(small.read_bytes()).decode("ascii")],
        "stream": False,
        "format": "json",  # constrained decoding: the schema comes back as JSON or not at all
        "keep_alive": "24h",
        "options": {"temperature": 0.0, "num_predict": MAX_TOKENS},
    }
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    return r.json().get("response", "")


def caption_frame(path: str | Path) -> dict:
    """Read one frame. Returns the parsed observation plus timing."""
    small = _downscale(Path(path))
    t0 = time.time()
    text = _generate_ollama(small) if BACKEND == "ollama" else _generate_mlx(small)
    elapsed = time.time() - t0

    obs = _parse(text)
    obs["_seconds"] = round(elapsed, 2)
    obs["_frame"] = str(Path(path).relative_to(config.ROOT))
    return obs


def caption_frames(paths: list[str | Path]) -> list[dict]:
    """Batch entry point. Sequential on the Mac; CONCURRENCY-wide on the box."""
    if CONCURRENCY <= 1 or len(paths) <= 1:
        return [caption_frame(p) for p in paths]
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        return list(ex.map(caption_frame, paths))


if __name__ == "__main__":
    import sys

    frames = [l for l in Path("/tmp/vlm_frames.txt").read_text().splitlines() if l]
    if len(sys.argv) > 1:
        frames = sys.argv[1:]

    _ensure_loaded()
    print(f"backend={BACKEND} model={MODEL_ID} edge={MAX_EDGE} concurrency={CONCURRENCY}\n")
    wall0 = time.time()
    results = caption_frames(frames)
    wall = time.time() - wall0
    total = 0.0
    for f, obs in zip(frames, results):
        total += obs["_seconds"]
        cam = Path(f).parent.name
        print(f"--- {cam} ({obs['_seconds']}s)")
        print("   ", json.dumps({k: v for k, v in obs.items() if not k.startswith("_")}))
    n = len(frames)
    print(f"\n{n} frames: {total / n:.2f}s/frame in-model, {wall / n:.2f}s/frame wall "
          f"at concurrency {CONCURRENCY} ({wall:.1f}s total)")
    print(f"projected full 646-camera sweep: {wall / n * 646 / 60:.1f} min")
