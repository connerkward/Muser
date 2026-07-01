# Feature parity across Muser instances

A maintainer's reference: the three Muser "platforms" are **the same codebase**
(`/Users/conner/dev/Muser`) run as three separate processes, each pointed at a
different **`MUSER_HOME`** data root. There is no per-platform fork — one FastAPI
service (`muser/service.py`) and one single-file frontend (`muser/web/app.html`)
serve all three. Divergence funnels through exactly three gating points:

1. **`MUSER_HOME`** (`muser/paths.py`) — the data root. Every DB / sidecar / thumbnail
   path routes through it, so setting the env var relocates the *entire* store
   (own LanceDB, `scores.json`, facets, captions, clusters). Isolation, not config flags.
2. **`/api/status` instance flags** — the service self-identifies from the root's
   basename: `personal` (`is_personal()` ⇒ `MUSER_HOME.name == ".muser-personal"`) and
   `outpaintings` (`_is_outpaintings_instance()` ⇒ basename `.muser-outpaintings`).
   `is_personal()` was tightened to match `.muser-personal` *specifically* so the
   outpaintings root doesn't falsely reveal the personal Triage tab.
3. **Nav gating (`app.html`)** — `syncPersonalNav()` / `syncOutpaintingsNav()` read
   those flags on boot and show/hide tabs + controls accordingly (with retries while
   the model warms).

- **Repo:** `/Users/conner/dev/Muser` · branch `main`
- **Generated:** 2026-06-30. Reflects the code on `main`; the primary source is
  `CLAUDE.md` + the two nav-gating functions in `muser/web/app.html`.

---

## 0. The three instances

| Instance | `MUSER_HOME` | Port | Hostnames | LaunchAgent | Bind |
|---|---|---|---|---|---|
| **Base** (aesthetic library) | `~/.muser` | 7777 | `muser.localhost` · `muser` · `muser.local` | `com.muser.serve` | `127.0.0.1` |
| **Personal** (Google-Photos triage) | `~/.muser-personal` | 7780 | `personal.muser.local` · `personal-muser.localhost` | `com.muser-personal.serve` | `127.0.0.1` |
| **Outpaintings** (curated read-only) | `~/.muser-outpaintings` | 7781 | `outpaintings.muser.local` · `outpaintings-muser.localhost` | `com.muser-outpaintings.serve` | `0.0.0.0` |

All three sit behind the same **Caddy** reverse proxy (`/opt/homebrew/etc/Caddyfile`)
and are warm-GPU LaunchAgents (`ProcessType=Interactive`, `LowPriorityIO=false`,
`SoftResourceLimits NumberOfFiles=8192`). Base and Personal launch `muser serve` /
`muser personal serve`; Outpaintings launches a plain `muser serve` with the env var —
its "outpaintings" identity comes purely from the root's basename, not a subcommand.

- **Base** — the full aesthetic-image-search app: every tab, the fal.ai generate
  pipeline (cart → nano_banana / gpt-image / 3D / LoRA), all facets computed.
- **Personal** — the hidden Google-Photos triage sub-tool (`muser/personal/`), running
  as an isolated process. `/api/status` → `personal:true` reveals Triage / Evaluate /
  Cleanup / Trash and hides AI / Jobs / Color / Results.
- **Outpaintings** — a curated **read-only** library of ~5,535 determined outpaintings,
  **indexed in place** (absolute paths into `~/ideas-syncthing/w/…` — no copy), scored
  for aesthetic + NSFW only. `/api/status` → `outpaintings:true` strips tabs down to
  Search / Interesting / Review and applies a full-width denser grid (`body.op`).

---

## 1. FEATURE × INSTANCE PARITY MATRIX

Legend: ✅ full · 🟡 partial / constrained · ❌ gap (hidden or not computed) · — n/a

| Feature | Base | Personal | Outpaintings | Gate |
|---|:--:|:--:|:--:|---|
| **Semantic search** (SigLIP) | ✅ | ✅ | ✅ | shared `/api/search` |
| Multi-concept comma query (blend / match-all) | ✅ | ✅ | ✅ | shared |
| Folder-scoped search (`?folder=`) | ✅ | ✅ | 🟡 search works; **Index-folder hidden** (read-only, in-place) | `#indexPick` hidden on op |
| Reverse-image / Similar | ✅ | ✅ | ✅ | shared |
| Image-upload search | ✅ | ✅ | ✅ | shared |
| **Color search** (LAB palette) | ✅ | ❌ tab hidden | ❌ hidden + never computed | `#navColor` hidden |
| **Aesthetic scoring** (dedup + `scores.json`) | ✅ | ✅ | ✅ | shared facet |
| **Interesting** tab | ✅ | ✅ (mixed view) | ✅ (explicitly surfaced) | shown |
| **NSFW scoring** / **Review** tab | ✅ normal mode | ✅ (mixed view) | ✅ demo-mode off ⇒ shown | demo-gated on Base |
| **AI-origin / C2PA** ("AI" tab) | ✅ | ❌ tab hidden | ❌ hidden + never computed | `#navAI` hidden |
| **AI-likelihood** (aidet `ai_pct` filter/sort) | ✅ | 🟡 filter present, AI tab hidden | ❌ not computed | facet |
| **Explore** 3D projection (PCA cloud) | ✅ | ✅ (not hidden) | ❌ tab hidden (no clusters) | `#navExplore` hidden on op |
| DINOv2 space (Explore + Similar) | ✅ | 🟡 sidecars are Base-library | ❌ | `space=dinov2` |
| **Generate pipeline** (cart → fal) | ✅ | ❌ (Results/Jobs hidden) | ❌ cart hidden | `#cartOpen`/Results hidden |
| image→3D (Meshy / Hunyuan multiview) | ✅ | ❌ | ❌ | part of generate |
| **Jobs** tab (live fal status) | ✅ | ❌ hidden | ❌ hidden | `#navJobs` |
| **Results** tab (generate outputs) | ✅ | ❌ hidden | ❌ hidden | `#navResults` |
| **Masonry** wall + infinite scroll | ✅ | ✅ | ✅ | debug menu |
| Sort-blend re-rank slider | ✅ | ✅ | ✅ (drives the landing) | shared |
| **Faces** tab (InsightFace clusters) | 🟡 if `faces` ran | ✅ | ❌ hidden | `s.faces` (main OR personal) |
| **Triage** tab (bucketing) | ❌ | ✅ | ❌ | `s.personal` |
| **Evaluate** tab (label + retrain) | ❌ | ✅ | ❌ | `s.personal` |
| **Cleanup** tab (delete-candidates) | ❌ | ✅ | ❌ | `s.personal` |
| **Trash** tab (pre-delete review) | ❌ | ✅ | ❌ | `s.personal` |
| Personalness classifier (P(personal)) | ❌ | ✅ | ❌ | `/api/personal/*` |
| People curation (name / merge) | ❌ | ✅ | ❌ | `/api/person` |
| VLM triage (gpt-4o-mini band) | ❌ | ✅ | ❌ | `/api/personal/vlm-*` |
| Export-to-main library | ❌ | ✅ | ❌ | `/api/personal/export-to-main` |
| Quick-mark bar (P/In·b/R/🗑 per thumb) | ❌ | ✅ | ❌ | `enableQuickMark()` |
| Per-image classification frame | ❌ | ✅ (mixed views) | ❌ | `enableFrames()` |
| Header 🗑 hide-flagged toggle | ❌ | ✅ | ❌ | `syncHideFlagged()` |
| **Full-width denser grid** (`body.op`) | ❌ | ❌ | ✅ | `document.body.classList.add("op")` |
| **Full-window** max-density mode (⛶) | ✅ present | ✅ present | ✅ (its home use) | `#fullwinBtn` (ungated) |
| **Aesthetic-blend landing** (ranked, not shuffled) | ❌ shuffled mix | ❌ shuffled mix | ✅ `_showcaseRank` | `window._OUTPAINT` |
| Quality metric in blend bar | ❌ | ❌ | 🟡 being added | `SORT_METRICS` |
| Reverse-image clipboard (`/api/clipboard`) | ✅ (secure ctx) | ✅ | ✅ (`*.localhost` secure ctx) | shared |

---

## 2. Base Muser — the aesthetic library

- **Root / port / host:** `~/.muser` · 7777 · `muser.localhost` (+ `muser`, `muser.local`)
- **LaunchAgent:** `com.muser.serve` (warms `siglip2-b` into RAM ~14 s, holds resident)
- **Purpose:** the full local-first semantic-image-search app over the aesthetic corpus.
- **Distinctive / everything-on:** all tabs visible (Search, Explore, Interesting,
  Review, Color, AI, Results, Jobs, + Faces when clustered); the whole **fal.ai generate
  pipeline** (cart → nano_banana / gpt-image / both / LoRA, BiRefNet cutout, prompt
  expansion, image→3D); Color + C2PA + AI-likelihood facets all computed and surfaced.
- **Hidden here:** the personal-triage surfaces (Triage/Evaluate/Cleanup/Trash and the
  quick-mark/frame/hide-flagged machinery) — they only reveal on `personal:true`.
- **Landing:** homepage showcase is a **shuffled** curated mix of the most aesthetic
  images (paged 60/scroll).

---

## 3. Personal Muser — hidden Google-Photos triage

- **Root / port / host:** `~/.muser-personal` · 7780 · `personal.muser.local`
- **LaunchAgent:** `com.muser-personal.serve` (`EnvironmentVariables MUSER_HOME=…`)
- **Purpose:** an isolated triage tool that sorts a Google Takeout export into
  **personal / in-between / reference**, with cleanup + people curation. Its DB, scores,
  facets, and captions never touch the aesthetic `~/.muser`. `personal/__init__.py`
  re-execs with the env var so import-time root resolution is correct.
- **Revealed (via `syncPersonalNav`, `s.personal`):**
  - **Triage** — bucket galleries + VLM uncertainty-band slider (cost shown first).
  - **Evaluate** — label ground truth, live accuracy/confusion, **Retrain on labels**
    (logreg P(personal) on SigLIP embeddings, ~86% CV), export → main library.
  - **Cleanup** — delete-candidates worst-first (opencv junk score + learned P(delete)).
  - **Trash** — 🗑-flagged review before moving to system Trash (`/usr/bin/trash`).
  - **Faces** — people clusters (InsightFace ArcFace + HDBSCAN); name / merge; a named
    person force-buckets their photos personal.
  - Cross-cutting: quick-mark P/In·b/R/🗑 bar on every thumb, per-image classification
    frame on mixed views, header 🗑 hide-flagged toggle.
- **Hidden here (noise for a camera roll):** **AI, Jobs, Color, Results** tabs
  (`syncPersonalNav` sets them `display:none`, and boots off a hidden tab if the hash
  points at one). Explore is *not* hidden.
- **Endpoints:** the `/api/personal/*` family (summary, bucket, vlm-*, eval-*, set-bucket,
  train, cleanup-*, delete-candidates, trash, export-*, hide-flagged) + `/api/faces`,
  `/api/person`. All guarded by `_is_personal_instance()`.

---

## 4. Outpaintings Muser — curated read-only library

- **Root / port / host:** `~/.muser-outpaintings` · 7781 · `outpaintings.muser.local`
  (binds `0.0.0.0`)
- **LaunchAgent:** `com.muser-outpaintings.serve` (plain `muser serve` + env var)
- **Purpose:** a **read-only** presentation surface over ~5,535 determined outpaintings,
  **indexed in place** — the LanceDB stores absolute paths into
  `~/ideas-syncthing/w/w-mb-outpaint-portfolio2026/…` (no copy of the images). Scored for
  aesthetic (aesthetic_v2 / pickscore / aesthetic_v25 / hps_v21) + NSFW
  (falconsai / adamcodd / marqo / consensus) only.
- **Distinctive (via `syncOutpaintingsNav`, `s.outpaintings`):**
  - `body.op` → **full-width, denser search grid**.
  - **Full-window** max-density mode (⛶, Esc to exit) — strips all chrome to just
    thumbnails + score + the blend console; state persisted in `localStorage`.
  - **Aesthetic-blend-ranked landing** — the empty-query showcase is ranked by the
    sort-blend (`_showcaseRank`), not shuffled, turning the landing into a
    model-comparison surface; each card carries a `_blendPct` relative to the top.
  - **Quality + Taste metrics in the blend bar** (`SORT_METRICS`/`ALLOC_METRICS`).
  - **Album curation** (`muser/albums.py` + `/api/album`) — region-dedup of a cover's
    re-rolls (fixed album-art box `x∈[0.52,0.64] y∈[0.13,0.74]`, region cosine ≥0.92):
    - **Unique spread** — one rep per cover on the landing (`_collapseAlbums`), "N variants" badge.
    - **Click-to-cluster** — click a cover → its variants re-ranked, ← back to spread.
    - **Grey out (negative bar)** — toggle NSFW classifiers (Falconsai/AdamCodd/Marqo) + a
      threshold; flagged covers grey out in place (red ✕), ranking untouched. A filter, not a weight.
    - Curated defaults: grey-out Marqo@0.91, blend AesV2 23%/PickScore 33%/Quality 31%/Taste 13%.
    - Full writeup: `docs/outpaintings-curation.md`. All gated to `s.albums` / `body.op`.
  - Demo-mode flipped **off** so the NSFW **Review** tab is shown.
- **Tabs kept:** Search (SigLIP) + Interesting (aesthetic) + Review (NSFW). Everything
  else is hidden: Explore, Faces, Color, AI, Results, Jobs, plus `#cartOpen` (generate)
  and `#indexPick` (Index-folder — it's read-only). Boots off a hidden tab to Search.

---

## 5. How instances are differentiated (the one flow)

```
MUSER_HOME (env, per-process)                        muser/paths.py
   ├─ ~/.muser              → basename ".muser"            → (neither flag)
   ├─ ~/.muser-personal     → is_personal()==True          → /api/status personal:true
   └─ ~/.muser-outpaintings → basename ".muser-outpaintings"→ /api/status outpaintings:true
                                        │
                                        ▼  app.html on boot
   syncPersonalNav()  → reveal Triage/Evaluate/Cleanup/Trash/Faces + quick-mark/frames;
                         hide AI/Jobs/Color/Results
   syncOutpaintingsNav() → body.op (full-width); hide Explore/Faces/Color/AI/Results/
                           Jobs/cart/Index; keep Search/Interesting/Review; ranked landing
   (Base) → no flag set → every tab shown, nothing gated off
```

- **`MUSER_HOME`** relocates the whole data root (isolation is a directory, not a config).
- **`/api/status`** derives `personal` / `outpaintings` from the root's *basename* —
  server-authoritative, no client guessing.
- **`syncPersonalNav` / `syncOutpaintingsNav`** are the only two client gates; both retry
  (up to 8× / 2.5 s) while the model warms so the tabs reliably reveal.
- Server endpoints double-guard: `/api/personal/*` all check `_is_personal_instance()`,
  so a stray request to a non-personal instance is refused regardless of the UI.

---

*File: `FEATURE-PARITY.md` — regenerate when `paths.py`, the `/api/status` instance
flags, or the `syncPersonalNav` / `syncOutpaintingsNav` gating change.*
