# Hackathon facts — NVIDIA Spark Hack: Seattle (14–16 Aug 2026)

Source: event page (Luma), read Sat 15 Aug 15:53. Things the brief got wrong or
left out are marked **⚠**.

## Logistics
- Venue: thinkspace SEATTLE, 1700 Westlake Ave N #200. Fri 5 PM → **Sun 4 PM**.
- Hosts: NVIDIA Developer Community (Kiana Steele, Jen Haller). Co-host: Ascend (pre-seed VC).
- Max 35 teams, 3–5 people. Judging expected Sunday afternoon; our brief plans the
  demo at 3 PM, freeze at 1 PM.

## Hardware **⚠**
- Every team gets an **Acer Veriton GN100** — Acer's GB10 Grace Blackwell box.
  Same silicon as the DGX Spark (128 GB unified, aarch64 Linux, ~273 GB/s), so every
  "Spark" note in `HACKATHON_MODELS/README.md`, `docs/models.md` and the brief
  applies unchanged. Our docs said "DGX Spark"; the box in the room is the Acer.
- Prizes include a Veriton GN100, an RTX 5080, Brev credits, investor meetings.

## Tracks — the part that matters **⚠**
> **See** — "AI that understands the physical world … Teams in this track **will work
> with NVIDIA's VSS (Video Search and Summarization) skills** to build
> perception-first applications."

We are a See-track entry and **nothing in the repo or the brief mentions VSS.**
Judges on this track will expect to hear how VSS fits. Options, cheapest first:

1. **Frame it (do this regardless).** VSS = ingest → VLM dense captions per chunk →
   index (CA-RAG: vector + graph DB) → summarize / Q&A / alerts. Safe Walk is that
   shape, built lean for a workload VSS wasn't designed for: 646 *still-image*
   feeds at one frame per sweep, not video streams. Same VLM family (Cosmos-Reason1
   is VSS's default VLM and is already on our SSD), same "read every frame into a
   typed observation" idea, our observation index is the CA-RAG-lite, our route
   nudges are the alerts. Say this out loud in the pitch and on the grill card.
2. **Use VSS's VLM on the box.** The brief already plans to swap
   `vision.caption_frame()` to Cosmos-Reason1-7B on the box. Do it via vLLM (or the
   NIM if it's on the box already). That is literally "working with VSS skills":
   same model, same prompt discipline. Low risk; it's the planned porting task.
3. **Deploy the actual VSS blueprint on the Acer** (Spark playbook, docker compose,
   NGC key, large container pulls) and feed it stitched per-camera clips
   (JPEG sequence → short mp4). Gets us real VSS summaries/alerts as an *extra*
   observation source. **Only if** the venue network holds and someone has 3–4
   hours Saturday night; otherwise it is a demo-day risk, not a feature. Do not
   put it in the demo path.

Recommendation: 1 + 2 tonight; 3 is stretch. Decide as a team before rehearsal.

Links from the event page (fetch when wifi allows):
- Start Building on DGX Spark (GB10 instructions/examples)
- NVIDIA Build model endpoints (hosted NIMs — note: cloud; our pitch is on-prem)
- **NVIDIA VSS Spark Playbook** — `github.com/NVIDIA/dgx-spark-playbooks/tree/main/nvidia/vss`;
  VSS docs `docs.nvidia.com/vss/`; blueprint repo `NVIDIA-AI-Blueprints/video-search-and-summarization`.
  Two modes on a single Spark: *Event Reviewer* (fully local VLM pipeline) and
  *Standard* (hybrid, remote LLM/embedding endpoints).

## Other tracks, for the record
- **Do**: agentic / tool-use / long-running workflows.
- **Spark**: anything, incl. generative / digital twin. (If the 3D idea ever
  becomes real, it lives here, not in See.)

## Data rule
Any public open data allowed as long as the problem is real; City of Seattle open
data portal explicitly suggested. We use SDOT + the travelers-map camera feed. ✔
