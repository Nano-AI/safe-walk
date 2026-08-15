# Model inventory — Mac + "Extreme SSD" (as of Sat 15 Aug 2026, ~15:50)

Every model weight we have on hand, where it lives, and what it is for. Three tiers:

- **A — Safe Walk stack**: in the demo path or a candidate for it.
- **B — hackathon kit**: pulled Thu/Fri for the agentic 3D + circuits idea
  (`HACKATHON_MODELS/README.md` on the SSD). Portable to the Spark; not used by Safe Walk.
- **C — other**: earlier projects / experiments on this Mac. Listed so nobody wonders.

Locations:
- **Mac HF cache** `~/.cache/huggingface/hub/` — only Qwen2.5-VL is real weights; the
  4 KB entries are stubs left by `hf download --local-dir` (weights went to the SSD).
- **SSD** `/Volumes/Extreme SSD/HACKATHON_MODELS/{vlm,cad,3d,embed,imagegen}` — HF
  safetensors, `.DONE` stamp = complete. exFAT, mounts on Mac and Spark.
- **SSD ollama store** `/Volumes/Extreme SSD/OLLAMA MODELS/` (253 GB) — the Mac's
  `OLLAMA_MODELS` points here, so `ollama list` shows *this* store while the SSD is
  mounted. With the SSD unplugged, local `~/.ollama/models` has only `ornith:9b`.
- **SSD extras**: `ollama-portable/` (ollama 0.32.4 linux-arm64 tarballs +
  `install-offline.sh` for the Spark), `toolchain/` (KiCad libs, aarch64 wheels —
  kit B, not models).

---

## A — Safe Walk stack

| Model | Where | Size / format | Use case |
|---|---|---|---|
| **`mlx-community/Qwen2.5-VL-7B-Instruct-4bit`** | Mac HF cache | 5.3 GB, MLX 4-bit | **The production VLM.** Reads every camera frame into the strict JSON schema in `safewalk/vision.py` (lighting, people, crowding, traffic, sidewalk_blocked, construction, emergency, notable). 11.9 s/frame p50 on M4; 100 % JSON parse over 2,700 reads. Set by `SAFEWALK_VLM`. |
| `Qwen/Qwen2.5-VL-7B-Instruct` | SSD `vlm/` | 15 GB, bf16 safetensors | Same model, full precision, for the **Spark** (vLLM / TensorRT-LLM behind `vision.caption_frame()`). Zero prompt change. |
| `Qwen/Qwen3-VL-8B-Instruct` | SSD `vlm/` | 16 GB, safetensors | Newer Qwen VLM: better OCR + spatial reasoning. Drop-in candidate on the Spark if the stack supports it. Untested on our schema. |
| `nvidia/Cosmos-Reason1-7B` | SSD `vlm/` | 15 GB, safetensors | NVIDIA physical-world reasoning VLM. Brief names it as the on-box swap ("scores points on the See track"). Untested; prompt compliance unknown. |
| **`mlx-community/Molmo2-8B-4bit`** | SSD `vlm/` | 6.1 GB, MLX 4-bit ✅ | Ai2 Molmo 2. **Pointing + counting** ("point at every person" → dots on the frame = verifiable `people_visible`), multi-image/video (feed 3–4 consecutive frames → persistence). Runs on the Mac today via existing `mlx-vlm` (`SAFEWALK_VLM=/Volumes/Extreme SSD/HACKATHON_MODELS/vlm/Molmo2-8B-4bit`). Untested on JSON schema and speed. Candidate for a live "count by pointing" beat, not the corpus model. |
| `allenai/Molmo2-8B` | SSD `vlm/` | 34.7 GB, fp32 shards ✅ | Full-precision Molmo 2 for the Spark (vLLM loads bf16). |
| `qwen2.5vl:7b` (ollama) | SSD ollama | 6.0 GB, GGUF | Same VLM served over the ollama API. Fallback path if MLX/vLLM misbehaves on the box; also the easiest way for a teammate to poke the prompt without the Python stack. |
| `Qwen/Qwen3-Embedding-0.6B` | SSD `embed/` | 1.2 GB | Embeddings. Only relevant if we do the stretch "natural-language search over the read corpus" (`notable` strings). Cut first per the brief. |
| `nomic-embed-text` (ollama) | SSD ollama | 274 MB | Same job, ollama flavour. |

## B — hackathon kit (agentic 3D + circuits idea; not Safe Walk)

Kept because the SSD goes to the Spark anyway. Details and Spark install notes in
`HACKATHON_MODELS/README.md`.

| Model | Where | Size | Use case |
|---|---|---|---|
| `filapro/cad-recode-v1.5` | SSD `cad/` | 2.9 GB | Point cloud → CadQuery Python (mesh → editable CAD). Qwen2-1.5B base. |
| `tencent/Hunyuan3D-2.1` | SSD `3d/` | 14 GB | Best open image → textured mesh, PBR materials. Needs CUDA kernels built on the Spark. |
| `tencent/Hunyuan3D-2mini` | SSD `3d/` | 24 GB | Turbo variants; seconds per mesh; agent inner loop. |
| `microsoft/TRELLIS-image-large` | SSD `3d/` | 3.1 GB | Structured latents → mesh / 3D Gaussians / radiance field. *Object*-level; not scene reconstruction — this is why "Gaussian splatting the streets" is not a weekend job (see `docs/feedback-nvidia-2026-08-15.md`). |
| `VAST-AI/TripoSG` | SSD `3d/` | 7.4 GB | Rectified-flow single-image → mesh; clean topology. |
| `flux1-schnell` (fp8) + `flux_text_encoders` (T5/CLIP) | SSD `imagegen/` | 16 + 4.8 GB | 4-step text → image, Apache-2.0. Feeds image-to-3D. Could make a **concept render for a slide** if we want one. |
| `qwen3-coder:30b` (ollama) | SSD ollama | 18 GB | 30B-A3B MoE coding agent; the kit's primary brain. Fast on the Spark's bandwidth. |
| `devstral:24b` (ollama) | SSD ollama | 14 GB | Agentic-coding specialist; second opinion. |
| `qwen3:32b` (ollama) | SSD ollama | 20 GB | Dense reasoning. |
| `gpt-oss:120b` (ollama) | SSD ollama | 65 GB | MXFP4 MoE, ~5B active; fits Spark's 128 GB. Best local reasoner we own. |
| `glm-4.7-flash` (ollama) | SSD ollama | 19 GB | Fast general agent. |
| `deepseek-ocr:3b` (ollama) | SSD ollama | 6.7 GB | Document / schematic OCR. |
| `glm-ocr` (ollama) | SSD ollama | 2.2 GB | OCR, smaller. |
| `gemma4:12b` (ollama) | SSD ollama | 7.6 GB | General chat/agent. |
| `qwen3.5:{0.8b,4b,4b-mlx,9b}` (ollama) | SSD ollama | 1.0 / 3.4 / 4.0 / 6.6 GB | Small general models; quick local tests. |
| `qwen3:8b`, `qwen2.5:{3b,7b}-instruct`, `qwen2.5-coder:7b`, `phi4-mini:3.8b`, `deepseek-r1:{1.5b,8b}` (ollama) | SSD ollama | 1–5 GB each | Older small LLMs; nothing in the demo path uses them. |

## C — other (earlier work on this Mac; ignore for the hackathon)

| Model | Where | Size | Note |
|---|---|---|---|
| `medgemma:4b`, `MedAIBase/MedGemma1.5:4b` (ollama) | SSD ollama | 3.3 / 7.8 GB | Medical LLMs, prior project. |
| `ornith:9b` (ollama) | **Mac local** `~/.ollama` | 5.6 GB | Custom model; only thing in the local ollama store. |
| `test1`, `reefer/minimonica`, `draganis/vanessa`, `lobotomy` (ollama) | SSD ollama | 2–5 GB | Custom/experimental tags; provenance unknown here. |
| `bucketresearch/politicalBiasBERT`, `matous-volf/political-leaning-politics`, `launch/POLITICS`, `ProsusAI/finbert` | Mac HF cache | 0.4–0.8 GB | Text classifiers, prior project. |
| `timm/vit_base_patch32_clip_224.openai` | Mac HF cache | 577 MB | CLIP ViT, prior project. |

---

## What runs where (Safe Walk)

| Machine | Corpus VLM | Live read | Notes |
|---|---|---|---|
| Mac M4 24 GB (today) | Qwen2.5-VL-7B 4-bit, MLX | same model, second copy in the API process | Two copies + worker grinding = the machine "dying". Worker is `kill -STOP`ped; `scripts/demo_mode.sh off` resumes. |
| DGX Spark GB10 128 GB | Qwen2.5-VL-7B (full) or Qwen3-VL-8B or Cosmos-Reason1-7B via vLLM | Molmo2-8B for pointing beat (optional) | One environment variable (`SWEEP_INTERVAL=60`); the inference backend swap is the only real porting task, isolated to `vision.caption_frame()`. Nothing GB10-specific has been tested yet. |

## Housekeeping

- Total on SSD ≈ 253 GB (ollama) + ~230 GB (`HACKATHON_MODELS`). 1.3 TB free.
- `OLLAMA MODELS.zip` (2.1 GB) at SSD root is an old partial archive of the store; the
  live store is the folder. Safe to delete once someone confirms nothing else uses it.
- To add a model to the SSD kit, follow the `get <repo> <subdir>` pattern in
  `scripts/dl_molmo2.sh` (idempotent, logs to `HACKATHON_MODELS/logs/hf.log`).
