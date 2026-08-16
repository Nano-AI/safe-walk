"""VLM read of a single camera frame.

Design rule that the whole product rests on: the model reports what is visible,
never a verdict. It may say "the near sidewalk is blocked by a parked truck".
It may not say "this street is unsafe". Safety judgements are the user's, made
from evidence we show them, because a model's opinion about danger is exactly
the claim we cannot defend and the collision record already answers better.

The Spark swap lives behind `caption_frames`: same call, same schema, different
backend and batch size. Mac runs MLX at batch 1; the box runs many at once.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from PIL import Image

from . import config

MODEL_ID = os.getenv("SAFEWALK_VLM", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
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
    global _MODEL, _PROCESSOR, _CONFIG
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


def caption_frame(path: str | Path) -> dict:
    """Read one frame. Returns the parsed observation plus timing."""
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    model, processor, cfg = _ensure_loaded()
    small = _downscale(Path(path))

    formatted = apply_chat_template(processor, cfg, PROMPT, num_images=1)
    t0 = time.time()
    result = generate(
        model,
        processor,
        formatted,
        [str(small)],
        max_tokens=MAX_TOKENS,
        temperature=0.0,
        verbose=False,
    )
    elapsed = time.time() - t0

    text = result.text if hasattr(result, "text") else str(result)
    obs = _parse(text)
    obs["_seconds"] = round(elapsed, 2)
    obs["_frame"] = str(Path(path).relative_to(config.ROOT))
    return obs


def caption_frames(paths: list[str | Path]) -> list[dict]:
    """Batch entry point. Sequential on the Mac; the Spark parallelises here."""
    return [caption_frame(p) for p in paths]


if __name__ == "__main__":
    import sys

    frames = [l for l in Path("/tmp/vlm_frames.txt").read_text().splitlines() if l]
    if len(sys.argv) > 1:
        frames = sys.argv[1:]

    _ensure_loaded()
    print(f"model={MODEL_ID} edge={MAX_EDGE}\n")
    total = 0.0
    for f in frames:
        obs = caption_frame(f)
        total += obs["_seconds"]
        cam = Path(f).parent.name
        print(f"--- {cam} ({obs['_seconds']}s)")
        print("   ", json.dumps({k: v for k, v in obs.items() if not k.startswith("_")}))
    print(f"\n{len(frames)} frames, {total:.1f}s total, {total / len(frames):.2f}s/frame")
    print(f"projected full 646-camera sweep: {total / len(frames) * 646 / 60:.1f} min")
