"""VLM lab: try models and prompts on sample frames, look at what comes back.

This is the workbench for the "VLM step" -- deciding *how* we read a frame:
which model, which prompt, what output shape, and whether the model can point
at the people it counts. It never touches the corpus; it reads a folder of
sample frames and writes a run folder you can eyeball.

    python -m safewalk.lab caption --frames data/lab/samples --model qwen2.5vl:7b
    python -m safewalk.lab people  --frames data/lab/samples --model qwen3-vl:8b
    python -m safewalk.lab both    --frames data/lab/samples --model qwen2.5vl:7b

Tasks
  caption  the production schema from vision.PROMPT (parse rate, latency, fields)
  people   grounding: one box per person -> overlay with boxes + centre dots
  both     caption then people on every frame; reports count agreement

Output: data/lab/runs/<name>/{results.jsonl, summary.md, overlays/*.jpg}

Backend is the same ollama HTTP API vision.py uses (OLLAMA_URL), so what works
here is what ships. Coordinate convention differs by family: Qwen2.5-VL returns
absolute pixels of the image it saw; Qwen3-VL returns 0-1000 normalised. We feed
the model an image whose long edge is a multiple of 28 (Qwen patch size) so
"absolute" matches our file, and detect normalised output by range.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from PIL import Image, ImageDraw

from . import config
from .vision import PROMPT as CAPTION_PROMPT, _parse

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
LAB = config.DATA / "lab"

PEOPLE_PROMPT = """You are reading a fixed traffic camera in Seattle. Find every person visible in the
image: pedestrians, people on bikes or scooters, people in wheelchairs, workers.
Do not include people inside vehicles. Do not guess at people you cannot see.

Answer with JSON and nothing else:
{"people": [{"bbox_2d": [x1, y1, x2, y2], "kind": "pedestrian" | "cyclist" | "worker" | "other"}, ...]}
If there are no people, answer {"people": []}."""


# --- image prep -------------------------------------------------------------
def prep(path: Path, edge: int = 1008) -> tuple[Path, tuple[int, int]]:
    """Long edge = `edge` (multiple of 28) so absolute coords line up. Cached."""
    out = LAB / "prepped" / f"{path.stem}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as im:
        im = im.convert("RGB")
        if max(im.size) != edge:
            s = edge / max(im.size)
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        if not out.exists():
            im.save(out, quality=90)
        return out, im.size


# --- model call ---------------------------------------------------------------
def ask(model: str, prompt: str, image: Path, max_tokens: int = 512, json_mode: bool = True) -> tuple[str, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
        "stream": False,
        "keep_alive": "24h",
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"
    t0 = time.time()
    r = httpx.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json().get("response", ""), round(time.time() - t0, 2)


# --- people boxes -------------------------------------------------------------
def _boxes(obs: dict, size: tuple[int, int], model: str = "") -> list[dict]:
    w, h = size
    normalised_1000 = "qwen3" in model.lower()  # Qwen3-VL grounds in 0..1000; Qwen2.5-VL in pixels
    out = []
    for p in obs.get("people") or []:
        bb = p.get("bbox_2d") or p.get("bbox") or p.get("box")
        if not (isinstance(bb, list) and len(bb) == 4):
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bb]
        except (TypeError, ValueError):
            continue
        if max(x1, y1, x2, y2) <= 1.0:            # fraction of width/height
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
        elif normalised_1000:                      # 0..1000 grid
            x1, x2 = x1 / 1000 * w, x2 / 1000 * w
            y1, y2 = y1 / 1000 * h, y2 / 1000 * h
        # else: absolute pixels of the image we sent (long edge 1008)
        x1, x2 = sorted((max(0, x1), min(w, x2)))
        y1, y2 = sorted((max(0, y1), min(h, y2)))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        out.append({"box": [round(x1), round(y1), round(x2), round(y2)],
                    "dot": [round((x1 + x2) / 2), round(y2 - (y2 - y1) * 0.5)],
                    "kind": p.get("kind", "person")})
    return out


def overlay(image: Path, boxes: list[dict], out: Path, note: str = "") -> None:
    with Image.open(image) as im:
        im = im.convert("RGB")
        d = ImageDraw.Draw(im)
        for b in boxes:
            x1, y1, x2, y2 = b["box"]
            d.rectangle([x1, y1, x2, y2], outline=(242, 169, 59), width=2)
            cx, cy = b["dot"]
            r = 6
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(232, 68, 58), outline=(255, 255, 255), width=2)
        if note:
            d.rectangle([0, 0, min(im.width, 8 + 7 * len(note)), 18], fill=(0, 0, 0))
            d.text((4, 3), note, fill=(255, 255, 255))
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=88)


# --- run ----------------------------------------------------------------------
def run_frame(task: str, model: str, path: Path, run_dir: Path) -> dict:
    img, size = prep(path)
    # samples are flat files named tag__CMR-xxxx__<frame>.jpg; corpus frames live in CMR-xxxx/
    parts = path.stem.split("__")
    cam = parts[1] if len(parts) >= 3 else path.parent.name
    rec: dict = {"frame": str(path), "cam": cam, "tag": parts[0] if len(parts) >= 3 else "", "model": model, "size": size}
    if task in ("caption", "both"):
        text, secs = ask(model, CAPTION_PROMPT, img, max_tokens=256)
        obs = _parse(text)
        rec["caption"] = obs
        rec["caption_seconds"] = secs
        rec["caption_ok"] = "parse_error" not in obs
    if task in ("people", "both"):
        text, secs = ask(model, PEOPLE_PROMPT, img, max_tokens=768)
        obs = _parse(text)
        boxes = _boxes(obs, size, model) if "parse_error" not in obs else []
        rec["people"] = boxes
        rec["people_raw"] = obs if "parse_error" in obs else None
        rec["people_seconds"] = secs
        rec["people_ok"] = "parse_error" not in obs
        note = f"{model} · {len(boxes)} people · {secs}s"
        if task == "both" and rec.get("caption_ok"):
            note += f" · caption says {rec['caption'].get('people_visible')}"
        overlay(img, boxes, run_dir / "overlays" / f"{path.stem}.jpg", note)
    return rec


def summarize(records: list[dict], task: str, model: str, run_dir: Path, wall: float) -> str:
    n = len(records)
    lines = [f"# lab run · task={task} · model={model} · {n} frames · {wall:.1f}s wall\n"]
    if task in ("caption", "both"):
        ok = sum(r["caption_ok"] for r in records)
        secs = [r["caption_seconds"] for r in records]
        lines.append(f"**caption**: parse {ok}/{n} · {sum(secs)/n:.2f}s/frame mean · p90 {sorted(secs)[int(0.9*(n-1))]:.2f}s")
        fields = ["lighting", "people_visible", "crowding", "traffic", "sidewalk_blocked", "construction", "emergency_activity"]
        lines.append("\n| tag | cam | s | " + " | ".join(fields) + " | notable |\n|---|---|---|" + "---|" * len(fields) + "---|")
        for r in records:
            c = r["caption"]
            lines.append(f"| {r.get('tag','')} | {r['cam']} | {r['caption_seconds']} | " + " | ".join(str(c.get(f, "—")) for f in fields)
                         + f" | {(c.get('notable') or '')[:60]} |")
    if task in ("people", "both"):
        ok = sum(r["people_ok"] for r in records)
        secs = [r["people_seconds"] for r in records]
        lines.append(f"\n**people**: parse {ok}/{n} · {sum(secs)/n:.2f}s/frame mean · total people boxed {sum(len(r['people']) for r in records)}")
        if task == "both":
            agree = sum(1 for r in records if r.get("caption_ok") and r["caption"].get("people_visible") == len(r["people"]))
            lines.append(f"caption `people_visible` == boxes on {agree}/{n} frames")
        lines.append("\n| tag | cam | boxes | caption count | s | overlay |\n|---|---|---|---|---|---|")
        for r in records:
            cc = r.get("caption", {}).get("people_visible", "—") if task == "both" else "—"
            ov = f"overlays/{Path(r['frame']).stem}.jpg"
            lines.append(f"| {r.get('tag','')} | {r['cam']} | {len(r['people'])} | {cc} | {r['people_seconds']} | {ov} |")
    lines.append(f"\nprojected 646-camera sweep at this rate: "
                 f"{wall / n * 646 / 60:.1f} min (serial wall time as measured)")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task", choices=["caption", "people", "both"])
    ap.add_argument("--frames", default=str(LAB / "samples"), help="folder of jpgs (recursively) or a single file")
    ap.add_argument("--model", default=os.getenv("SAFEWALK_VLM", "qwen2.5vl:7b"))
    ap.add_argument("--name", default=None, help="run name (default task-model)")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("SAFEWALK_VLM_CONCURRENCY", "2")))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    src = Path(a.frames)
    frames = [src] if src.is_file() else sorted(p for p in src.rglob("*.jpg") if ".s" not in p.suffixes[:1])
    if a.limit:
        frames = frames[: a.limit]
    if not frames:
        raise SystemExit(f"no frames under {src}")
    run_dir = LAB / "runs" / (a.name or f"{a.task}-{re.sub(r'[^A-Za-z0-9.]+', '_', a.model)}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # warm
    httpx.post(f"{OLLAMA_URL}/api/generate", json={"model": a.model, "prompt": "", "keep_alive": "24h"}, timeout=600).raise_for_status()
    print(f"[lab] {a.task} · {a.model} · {len(frames)} frames · concurrency {a.concurrency} -> {run_dir}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        records = list(ex.map(lambda p: run_frame(a.task, a.model, p, run_dir), frames))
    wall = time.time() - t0

    with (run_dir / "results.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    md = summarize(records, a.task, a.model, run_dir, wall)
    (run_dir / "summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
