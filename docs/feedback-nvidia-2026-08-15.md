# Feedback from NVIDIA (Sat 15 Aug, on site) — and what to do with it

Someone from NVIDIA gave us four ideas. This is what they said, what we think, and
what (if anything) we build before Sunday's freeze. Short version: **take idea 1 as
pitch framing (cheap, strong); put 2–4 on a roadmap slide, do not build them.**

## What they said

1. **Government-facing.** Instead of (or as well as) a consumer app, face the city:
   the city offers the routing to residents, *and* gets a view of which spots are
   under-covered by cameras and which spots would most benefit users.
2. **Real-world 3D / Gaussian splatting.** Model the streets in 3D; understand
   where 3D could add value.
3. **Google Maps Street View integration.**
4. (Implied by 1) Give the city actionable data, not just an app.

## Our read

### 1. Government-facing — yes. It's framing, not a pivot.

Everything needed already exists in the data; the city is just a second consumer of
the same observation index the pedestrian route uses.

What the city gets, from queries we can already run over the graph:

| City question | Where it comes from |
|---|---|
| "Where should the next camera go?" | Segments ranked by pedestrian collisions **with no camera watching**. Today 52% of blocks / 39% of ped collisions are unwatched (`/api/stats`). |
| "Which cameras are wasted?" | Cameras whose segments have near-zero collision history — re-aim candidates. |
| "Where would infrastructure help walkers most?" | Segments the "safer" route detours around most, with `risk_parts` saying *why* (missing sidewalk vs. arterial speed vs. collision density). |
| "Is coverage equitable?" | Same query cut by neighbourhood — ties into the redlining audit we discussed (see below). |

Why this matters for the pitch:
- **B2G is how this ships.** A city can't push an app to citizens; a city *can* adopt an
  internal dashboard and expose an API. SDOT already owns the cameras and the data —
  nobody has joined them.
- **It sharpens the Spark story.** One on-prem box; no citizen imagery leaves the
  building; no cloud vendor sees the frames. For public-sector procurement, that
  privacy/sovereignty argument is stronger than "cheaper than API calls." Use it.
- The redlining/equity metrics we listed earlier (diversion ratio by tract, detour
  tax, coverage per km by tract) become a *feature* for the city, not just a defence.

**Decision: pitch it, don't build it before Sunday.** The pedestrian contact sheet
stays the money moment. Add one line to the deck and one line to the grill card.
If someone has a spare 90 minutes after rehearsal, a `/api/coverage` endpoint plus a
"For the City" panel is the smallest possible build; it is not load-bearing.

### 2. Gaussian splatting / 3D — a trap this weekend. Say it, don't build it.

- Fixed traffic cameras are single-viewpoint, low-res, and uncalibrated (no pose).
  You cannot splat from them. 3D needs multi-view capture (Street View, drive-by),
  which is different data.
- The `3d/` models on the SSD (Hunyuan3D, TRELLIS, TripoSG) are *object* image→mesh
  generators, not scene reconstruction. Not applicable.
- What 3D would actually buy: eye-level sightlines, lighting simulation, "what this
  corner looks like at night from a person's height." Real, but a different product.

Sentence for the roadmap slide: *"3D is where this goes once the city gives us
camera calibration and we fuse Street View. Today's product is the observation
index that 3D would sit on top of."*

### 3. Street View — partial. Not in the demo path.

- Street View Static API gives an eye-level still per lat/lon + heading. Cheap way to
  show *unwatched* blocks. But it is months–years stale, which contradicts the whole
  claim ("what does that street look like **right now**"). It also needs an API key,
  has ToS limits on storing imagery, and puts a cloud call in the demo path we
  promised to keep offline.
- Good use, if any: **one side-by-side** — "Street View (2023) vs. our camera
  (10 minutes ago)." Dramatises freshness. One image, not a feature.

## What actually changes before Sunday

- Deck: one slide "Two customers, one index" — pedestrian route + city coverage view.
  One roadmap slide: 3D / Street View / second city.
- Grill card: "Why not just Street View?" → staleness. "Why not 3D?" → data doesn't
  support it yet; observation index first. "Who pays?" → the city; on-prem box;
  imagery never leaves.
- Code: nothing new for this. Reserve.

## Related: equity / redlining audit (from earlier today)

Metrics we should be able to answer if asked (none built yet; all derivable):
diversion ratio per tract (safer-route metres / direct-route metres), detour tax for
trips starting/ending in each tract, camera density per km by tract, mean risk and
`risk_parts` by tract vs. ACS / Seattle RSEI, HOLC-grade overlay. Live nudges
deliberately exclude crowding and people counts so the router never steers around
"busy" — see `safewalk/live.py` docstring.

## Appendix — code changes landed today (so nobody is surprised by the API)

- `safewalk/live.py` (new): fresh camera reads (frame age ≤ `LIVE_MAX_AGE`, default
  3600 s) become per-segment nudges — emergency +0.25, sidewalk blocked +0.20,
  construction +0.10, unlit +0.15, capped at 0.35. Only adds cost; static record
  still dominates.
- `/api/route` now takes `safe_weight` and `use_live` (default true). Response adds
  `live` (fresh/stale/flagging counts) and per-route `live_notes`; segments carry
  `live`. `use_live=false` is the ablation a judge can ask for.
- `/api/camera/{id}/read` returns `frame_source` (`live`/`stored`) and
  `frame_age_s`. Fixed: a failed live fetch no longer leaves a stale `live.jpg` for
  the read to silently caption.
- Rankings ("busier than N% of downtown") now only include fresh reads.
- `baseline.get(refresh=True)` no longer re-parses the corpus on every request
  (mtime-gated). Route requests ~5 ms.
- UI: detour-appetite slider (0–8), "let fresh camera reads steer the route"
  toggle, live box naming what the cameras added/avoided, frame-age badges on
  tiles, live/stored badge on the live-read dialog.
- `scripts/dl_molmo2.sh`: pulls Molmo2-8B (MLX 4-bit for Mac; full weights for
  the Spark) to the SSD kit. Molmo 2 = Ai2's VLM with pointing/counting and
  multi-image; candidate for a "count people by pointing at them" live beat.
  Untested on our JSON schema; not swapping the corpus model before Sunday.
- Ops: caption worker is **suspended** (`kill -STOP`), not dead — it and the API
  each hold a copy of the VLM and the Mac was struggling. `scripts/demo_mode.sh
  off` resumes it.
