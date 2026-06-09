# Muser — agent notes

Local-first semantic image search with a built-in retrieval-eval harness, plus a
fal.ai **generate** pipeline (text/image→image, 3D, LoRA) layered on the cart.
**Scope of "local/offline/$0":** the *core* — embedding, index, search, color, dedup,
on-disk captions — is genuinely local, offline, and free (no keys, no network). Two
things are NOT: the **generate pipeline** (Nano Banana / ChatGPT-image / 3D / LoRA) is a
fal.ai/OpenAI cloud add-on that needs network + `FAL_KEY`/`OPENAI_API_KEY` and costs
money; and the **3D viewer (`<model-viewer>`) + Explore point-cloud viz** load three.js /
model-viewer from CDNs, so they need internet to *render* (both degrade gracefully offline
— model-viewer → download link, viz → hidden). Captioning also calls OpenAI when run.
**Python 3.12 + uv.** A TypeScript MCP ext-app lives in `mcp-ts/` — a thin HTTP
client of the embedded service (`muser serve`) that renders results in an
interactive gallery UI inside Claude Desktop. It loads no model and never touches
LanceDB; it proxies the JSON API. Search tools (`search_images` w/ folder scope,
`search_by_color` multi-hex, `search_similar` reverse-image) return a numbered
**contact sheet** image (top 36) + top-3 detail thumbs so the *model* sees the
results, plus a hidden LLM-facing metadata block; the *user* gets the full gallery.

See `REQUIREMENTS.md` for scope/decisions.

## Architecture

- `muser/embedders.py` — model-agnostic image/text embedders in a shared space.
  Backends: `SentenceTransformerEmbedder` (CLIP/SigLIP baselines) and
  `JinaV4Embedder` (2026 frontier default). Heavy deps lazy-imported.
- `muser/registry.py` — `name -> (tier, factory)` model registry. Add a model
  once → it appears in CLI, index, and benchmark. `DEFAULT_MODEL = "siglip2-b"`.
- `muser/index.py` — `MuserIndex`: one embedded LanceDB at `~/.muser/db`, one
  table per model (`img__<model>`), cosine over L2-normalized vectors.
  Incremental by mtime; skips corrupt files.
- `muser/cli.py` — `muser` entrypoint (typer): `index`, `reindex-metadata`, `search`,
  `bench`, `models`, `cluster`, `score`, `caption`, `detect`, `aiscore`, `color`, `serve`,
  `web`, `uid`. `detect` runs the C2PA library scan headless (writes `~/.muser/c2pa.json`; same
  data the web "AI" tab uses). The generate pipeline is web/MCP-only (no CLI command).
- `muser/service.py` — **embedded service** (`muser serve`): FastAPI app that warms
  the model once and owns the index; serves JSON API + the web search UI
  (`muser/web/app.html`). Warm search ≈ 40 ms (precomputed dedup, see below).
  Endpoints: /api/search, /api/search-upload (multipart image→image), /api/search-color
  (multi-hex CSV + `mode=all|any`), /api/search-image, /api/search-compose, /api/filter,
  /api/index, /api/thumb (PIL; `fit=cover` square smart-crop default | `fit=contain`
  aspect-preserving for masonry), /api/image, /api/upscale (4× → cached + /tmp mirror, for
  the photo before/after slider), /api/projection (3D PCA cloud; `space=dinov2` + `method=
  hdbscan|kmeans`), /api/clusters & /api/cluster (`space=dinov2`), /api/contact-sheet
  (numbered montage for the MCP→LLM), /api/reveal (open -R), /api/model, /api/status,
  /api/jobs (foreground task + background jobs + recent pipeline runs, for the Jobs page),
  /api/folders, /api/c2pa, /api/ai, /api/ai/scan, /api/color[/scan], GET|POST
  /api/demo-mode. **Generate pipeline:** POST /api/checkout (`mode=zip|generate`),
  /api/checkout/status, /api/checkout/zip, /api/pipelines, /api/pipeline/{id},
  /api/pipeline/{id}/file/{path} (path-traversal-guarded static serve incl. .glb),
  POST /api/pipeline/{id}/threed (`multi_angle`, post-hoc 3D on a generated output).
  **AI-origin detection (C2PA):**
  `c2patool` (optional, `brew install c2patool`) reads a file's signed Content
  Credentials and reports whether they *declare* it AI-generated/-edited (IPTC
  `trainedAlgorithmicMedia` / `compositeWithTrainedAlgorithmicMedia`). Two surfaces,
  both in `muser/c2pa.py`: (1) **per-result badge** — `/api/search` inlines the verdict
  on every result (`r.c2pa = {ai, kind, tool}`) from a RAM cache primed at service
  startup from `~/.muser/c2pa.json` (the persisted scan), pinning an amber **"AI?"**
  badge synchronously; `/api/c2pa?path=` remains a fallback for paths the sidecar
  doesn't know about (newly indexed); (2) **"AI"
  tab** — `/api/ai` lists every flagged image from a persisted library scan
  (`~/.muser/c2pa.json`, incremental by mtime+size, parallel; same sidecar pattern as
  `scores.json`/`clusters.json`), `POST /api/ai/scan` runs that scan in the background
  with progress, and `muser detect [--in <folder>]` does it **standalone** — no running
  `muser serve` and no embedding model (just a LanceDB read + c2patool), and safe to run
  concurrently with the service (LanceDB allows concurrent readers). **Auto-trigger:** indexing a folder
  (web "Index folder", `muser index`, or `--local`) automatically runs the detector over the
  just-indexed subtree when it finishes — incremental, so only new/changed files spawn
  c2patool, and it's scoped to that folder (not a full re-scan). Positive-only and a **lower bound** —
  only cloud generators (OpenAI/Firefly/Google) ship credentials; local SD/Flux/ComfyUI
  output carries none, so `ai=False`/absence ≠ "confirmed real". Degrades to
  `available:false` (no badge, no tab results) when the binary is absent.
  **Folder-scoped search:** `/api/search?folder=<dir>` restricts
  results to images under that directory (any depth) — pushed into LanceDB as a
  `prefilter` half-open range on `path` (`>= dir/ AND < dir⁺`, so wildcard chars
  like `_` can't false-match). The web UI has a **scope** box (datalist of indexed
  dirs + counts from `/api/folders`); free-text so any path prefix works. Same
  scoping on the CLI (`muser search "…" --in <folder>`) and MCP
  (`search_images(query, k, folder=…)`) — both thin clients of `/api/search`.
- `muser/caption.py` — per-image natural-language captions via **OpenAI GPT-4o-mini**
  (chat-completions, stdlib `urllib` only — no `openai` SDK dep). System prompt
  baked for SDXL/Flux LoRA captions (one sentence, concrete subjects, no style
  descriptors). Images sent as JPEG-base64 data URLs with `detail: "low"`
  (~85 image tokens). Cost ~$0.001-0.005/image. `OPENAI_API_KEY` auto-loaded from
  `/Users/conner/dev/central/.env`. `muser caption [folder] [--paths …]` writes
  one row per image to `~/.muser/captions.jsonl` (append-only:
  `{path, caption, model, mtime, ts, prompt}` — `model="gpt-4o-mini"`; `prompt`
  records the system prompt used, so a custom caption prompt is reproducible).
  `caption_image`/`caption_paths` take an optional `prompt=` override (defaults to
  `DEFAULT_CAPTION_PROMPT`). Cart UI's
  "Caption missing (N)" button POSTs to `/api/caption-bulk`, which drives the
  existing busy overlay via `state.task = {"kind": "captioning", done, total}`.
  `/api/caption?path=…` returns the latest caption for a single file.
- `muser/jobs.py` — tiny thread-safe **in-process job registry** (`REGISTRY` singleton)
  backing the non-blocking cart **checkout** (both zip-mode and generate-mode). A job
  carries the legacy `caption`/`upscale`/`zip` slots **and** a generic ordered `stages`
  list (`set_stage(name,status,done,total)`) used by the multi-stage generate pipeline.
  Jobs are ephemeral RAM (evicted after 1h or once 32 newer exist; zip bytes never
  serialized). `POST /api/checkout {items:[{path,upscale}], caption_prompt?,
  force_caption?, mode?}` returns a `job_id`; **zip mode** (default) runs captioning
  (network, OpenAI) **concurrently** with upscaling (local 4×, flagged items only) in a
  daemon thread — deliberately NOT on the single `state.task` slot, so search/index stay
  usable. `GET /api/checkout/status?id=` reports `{status, caption, upscale, stages,
  errors, captioned, upscaled, captions_missing, zip_ready, error}` (+ `outputs` for
  pipeline jobs, read from run.json which survives eviction); `GET /api/checkout/zip?id=`
  serves the prebuilt zip. The job re-loads captions.jsonl before building the zip, so
  freshly-captioned items land in the bundle (fixes the prior all-null-caption export).
  Zip-building is `_build_cart_zip()`, shared by `/api/zip` and the checkout job.
  Upscaled bytes are NOT durably stored (only the `~/.muser/upscale_cache` warm cache).
  Web UI: a dismissible bottom-right progress panel (+ minimizable pill) polls status;
  the caption-prompt textarea persists to `localStorage["muser-caption-prompt"]`.
- `muser/generators.py` — **fal.ai generation primitives** (stdlib `urllib` only, no
  `fal` SDK): plain JSON POSTs to `https://fal.run/<id>` with `Authorization: Key`,
  3-attempt 429/5xx backoff, fal-CDN upload (local bytes/file → public URL). `FAL_KEY`
  auto-loaded from `central/.env`, never logged. Endpoint IDs are the single source of
  truth: `nano_banana` (`gemini-3.1-flash-image-preview/edit`, ≤14 refs — `NANO_MAX_REFS`,
  422s on 15+), `gpt_image` (OpenAI `gpt-image-1` **edits** endpoint, multipart, ≤8 refs —
  `GPT_IMAGE_MAX_REFS`), `cutout` (BiRefNet v2 → transparent PNG), `expand_prompts`
  (any-llm/vision = Gemini 2.5 Flash, brief→N distinct prompts), `image_to_3d`
  (**Meshy 6** — won a bake-off vs Hunyuan3D-v3 / Tripo v2.5 on clean ¾ inputs),
  `image_to_3d_multiview` (Hunyuan3D v3, accepts L/B/R extra views — `EP_MULTIVIEW_3D`),
  `generate_views` (3 concurrent nano_banana edits → L/B/R product shots feeding the
  multiview path), `train_lora`/`flux_lora_generate` (FLUX LoRA, expensive route).
  **fal queue API (opt-in):** a thread-local **status sink** (`set_status_sink`) makes
  `fal_run` route through the QUEUE endpoint (`fal_run_queued`: submit → poll status →
  fetch `response_url`, same payload as sync) and emit `{request_id, status,
  queue_position}` per poll. No sink set → plain sync POST (unchanged for every non-pipeline
  caller). The pipeline uses this to surface real fal-side progress on the Jobs page.
- `muser/pipeline.py` — generate-mode **checkout pipeline orchestrator**
  (`run_pipeline`), spawned in a daemon thread by `POST /api/checkout {mode:"generate"}`.
  Staged + resilient (a single failed prompt/image records a stage error, never aborts):
  caption → refs(upscale+upload) → cutout → prompt-set(cap 14) → prompts (explicit
  `gen_prompts` | LLM `expand_prompts` | replicate brief) → generate (`generator` ∈
  {`nano_banana`, `chatgpt`, `both` (fan out per prompt), `lora` (zip→train→gen)}) →
  optional 3d (per generated image; `multi_angle` → synth L/B/R views + multiview recon).
  Persists to `~/.muser/pipelines/<run_id>/run.json` (atomic write under a per-run lock)
  + `outputs/<files>`, with a newest-first `index.json` of run summaries. Final status
  `done` iff ≥1 image materialized, else `error`. Each cloud stage runs inside a
  `_fal_stage(name)` context manager that installs the fal status sink, so the stage's
  `note`/`request` fields carry live fal queue status (`fal: queued · pos N` / `running on
  GPU` / `completed`) + the fal request id — visible on the Jobs page.
- `muser/dinov2.py` — **opt-in DINOv2 space** for Explore + Similar (additive; SigLIP stays
  default). Pure-numpy, no model load — reuses precomputed DINOv2-large vectors
  (`~/.muser/dinov3_canon.npz`, ~17.6k canonical; file named `dinov3_*` for historical
  reasons but the vectors are DINOv2). `similar(path,k)` = cosine kNN; `clusters()` reads
  `dinov3_clusters.json` (217 clusters vs SigLIP's 89 — finer visual grouping). Surfaced via
  `space=dinov2` on `/api/search-image`, `/api/clusters`, `/api/cluster`, `/api/projection`;
  falls back to SigLIP when a sidecar/path is missing. **Cluster names** are baked into
  `dinov3_clusters.json` as a `label` per cluster (a one-off gpt-4o-mini pass over each
  cluster's rep images — "bmw logo", "car interior", "modern architecture").
- `muser/projection.py` — `/api/projection` 3D point cloud for the **Explore** tab:
  pulls a stride-sampled (≤`MAX_N=5000`) set of (path, vector) rows from the model's
  LanceDB table, PCA-projects to 3D via **numpy SVD** (no sklearn), colors each point by
  its HDBSCAN cluster (golden-angle palette, pure fn of cluster id), and skips hidden
  paths via the service's `_hidden` predicate. Service caches the payload keyed by
  (n, folder, db-mtime, clusters-mtime). **Render is not offline:** the Explore tab loads
  three.js from a CDN (`ajax.googleapis.com`/`cdn.jsdelivr.net`), so the viz is hidden when
  there's no internet (the projection JSON itself is computed locally).
- `muser/facets.py` — shared sidecar scaffolding for **per-image precomputed facets**
  (the c2pa.py cache pattern factored out): a `~/.muser/<name>.json` keyed by path with
  `m`(mtime_ns)+`s`(size) for incremental skip, a thread-pool `scan(paths, compute)`, and a
  RAM cache primed once for O(1) enrichment/ranking. c2pa.py keeps its own copy (shipped,
  untouched); new facets build on this. The RAM cache **auto-re-primes when the sidecar
  file's mtime changes** (`_maybe_reprime` on lookup/entries), so the running service
  reflects writes from ANY process — a standalone `muser aiscore`, the post-index hook,
  another window — with no restart. `scan(...)` checkpoints atomically every
  `checkpoint_every` computed files (re-priming in-process too) so a long scan is
  crash-safe, resumable, and goes live progressively.
- `muser/aidet.py` — **AI-likelihood facet**: a soft pixel-level "how likely is this
  AI-generated?" score (0–100%), complementing the metadata-only C2PA verdict. Backend:
  **Community Forensics** (Park & Owens, CVPR'25 — `OwensLab/commfor-model-384`, MIT, ViT-S/16
  trained on 4,803 generators; ~88 MB weight auto-downloaded from HF + cached). **It replaced
  GRIP `Grag2021_latent`**, which over-flagged catastrophically on this diverse library —
  GRIP is really an "is this a clean *camera photo*?" detector, so it flagged digital art /
  3D renders / screenshots / vintage-scanned / recompressed-JPEG images as AI (~100% false
  positives on the corpus). An A/B on the same images cut that to ~5% (real photos ~0%) while
  still catching SDXL/ComfyUI output ~100%; it's *weaker* on Flux/ChatGPT/Gemini (~12–46%),
  so it stays a **soft hint paired with C2PA**, never an authoritative claim (the GRIP era —
  benchmark, Platt calibration, `_grip_resnet.py`, 269 MB weight — is removed; history in
  `~/Desktop/2026-06-08-ai-detector-benchmark/` + the swap A/B). Preprocess matches the
  authors' test transform (Resize 440 → CenterCrop 384 → ImageNet norm); the model's
  **sigmoid output IS the calibrated P(AI)** so `pct=round(100·σ(logit))` — no Platt step.
  Built on `facets.Sidecar` (`~/.muser/aidet.json`, incremental + content-addressed);
  `available()` only needs torch+timm. Soft, always-present score (every image), surfaced as
  a **percentage**, never a binary claim. CLI `muser aiscore` (standalone scan, or `-q <path>`
  for one image); auto-triggers post-index alongside c2pa/color; primes at startup. Surfaced
  per result as `ai_pct` (via `_attach_aidet`); the web filter is a 3-way `[All|Hide AI|Only AI]`
  control (`ai_min`/`ai_max` band) + `sort=ai|ai_asc` on `/api/search` (+ `/api/filter`), all
  reachable from web, CLI (`--ai-min`/`--ai-max`/`--sort`), and MCP (`search_images`).
- `muser/color.py` — **color search** (a *separate LAB-palette index, not the embedder*).
  Per image: median-cut dominant-color palette → CIE-LAB swatches + fractions, persisted to
  `~/.muser/color.json`. `search(rgb)` ranks by `Σ frac·sim(palette,query)` (LAB ΔE, decays
  at `COLOR_TAU`). `available()` always True. CLI `muser color [--query "#rrggbb"] [--in dir]`
  (no query → build), API `/api/search-color?hex=`, `/api/color[/scan]`, web **Color** tab
  (picker + preset swatches). Fully local, $0, no model load. New core dep
  `opencv-python-headless` (LAB conversion; no GUI/Qt libs → clean on server + CI).
  The color scan **auto-triggers** post-index (incremental, folder-scoped) alongside c2pa,
  and primes at service startup.
- *(removed 2026-06-05)* **skin-tone search** (Monk-scale) was built then removed at the
  user's request. Full implementation preserved at git tag `skintone-v4-archived`; design,
  benchmark results, and revival steps in `reports/skintone-archive/`.
- `eval/datasets.py` — standard benchmarks reduced to {image_paths, queries,
  qrels}. Flickr30k via HF's `refs/convert/parquet` branch (scripts unsupported).
- `eval/harness.py` — embeds corpus → LanceDB → queries → **ranx** metrics
  (hits@1, recall@5/10, mrr, ndcg@10, map) + latency. `format_table` for a comparison.
- `eval/web.py` — Gradio UI: Benchmark tab (run + compare) and Inspect tab
  (query the corpus, gallery flags the ground-truth image).

## Web UI surfaces (`muser/web/app.html`)

Single-file frontend. Beyond Search, the tabs/controls added this session:

- **Generate (checkout, `mode=generate`).** Cart → fal pipeline (cloud, needs network +
  `FAL_KEY`/`OPENAI_API_KEY`, costs money); a Results tab renders outputs incl.
  `<model-viewer>` 3D (wireframe toggle) for `.glb`, with a post-hoc "make 3D" button per
  image (POST /api/pipeline/{id}/threed, `multi_angle`). `<model-viewer>` loads from a CDN,
  so 3D needs internet to render and falls back to a download link offline.
- **Explore** — the /api/projection 3D point cloud, cluster-colored; tracks the
  space (SigLIP/DINOv2) + method (HDBSCAN/K-means) pickers (re-projects + re-colors on
  toggle, single canvas).
- **Jobs** tab — polls /api/jobs (1.2s while open): foreground task progress, background
  checkout/generate jobs with per-stage progress + **live fal status** (`fal: queued · pos
  N` / `running on GPU`, fal request id on hover) + errors, and clickable recent
  generate-run cards (→ Results). RAM-backed jobs evict after 1h/32 newer; persisted runs
  survive restarts.
- **Masonry** (debug-menu toggle, press `d`) — Pinterest/Cosmos image wall, **#results
  only**. JS append-only layout (`layoutMasonry`): tiles absolute-positioned into the
  shortest column, heights from each result's `data-w/h` so **no pop-in** and appending a
  page never reshuffles existing tiles. Labels hidden, even 14px gaps. Uses `fit=contain`
  thumbs (square smart-crops can't vary height). Per-page sort-blend re-rank is suppressed
  in masonry (it would reshuffle the wall on scroll). The analytic grids
  (Interesting/Review/AI/Color) are NOT masonry'd — they render through their own functions
  and stay a normal square grid.
- **Infinite scroll** now also covers the **homepage showcase** (pool 600/metric, shuffled,
  paged 60/scroll) and **reverse-image/Similar** (deeper top-N, k += 120) — previously
  fixed-size.
- **Sort blend** — a segmented allocation-bar slider that re-ranks search hits client-side
  by a weighted sum of per-result aesthetic metrics (`_SORT_BLEND_METRICS`: aesthetic_v2,
  pickscore, aesthetic_v25, hps_v21, aesthetic) blended with relevance; metrics are inlined
  per hit so no re-fetch is needed.
- **Image-upload search** (drag/drop or picker → /api/search-upload) and **multi-color
  search** (multiple swatches + all/any mode → /api/search-color).
- **Random-aesthetic homepage**, **debug menu** (press `d`: thumb size / columns +
  masonry toggle + demo-mode toggle), **photo before/after upscale slider** (/api/upscale)
  with a grain overlay. The NSFW **Review** tab is shown in normal mode and hidden only in
  demo mode (`syncNsfwNav` on boot + `applyNsfwNav` on demo toggle).

## Reverse-image search (web UI)

Every result, score card, and dupes-modal file has two client-side, **$0** buttons,
both driven by one helper `engineSearch(path, btn, url, hint)` in `app.html`:

- **⌖ Source** (green) → Google Lens — "where does this image appear" / visually similar.
- **⏱ Origin** (amber) → TinEye — provenance. The toast tells the user to paste then
  **sort by "Oldest"**, surfacing the earliest *crawled* copy ≈ original source.

`engineSearch` fetches `/api/image`, converts to PNG (`createImageBitmap` → canvas,
since clipboard image writes must be PNG), `navigator.clipboard.write`s it (works on
`127.0.0.1` — a secure context), flashes a "✓ Copied" pop, then opens the engine; the
user presses ⌘V. Nothing leaves the machine until they paste.

Two honest limits: (1) the ⌘V is irreducible for a *local* file — a plain link can't
auto-upload it, and `…/uploadbyurl?url=` needs a *public* URL (rejected: would require
tunnelling the service). (2) "first time posted on the internet" is **not knowable** —
TinEye's oldest = earliest *it* crawled, not the true first post (origin may be
deleted/uncrawled/offline). Best-effort proxy, not a guarantee. A fully automated
earliest-date answer would need the **paid TinEye Search API** (`sort=crawl_date asc`,
~$0.04/search) wired server-side — not built.

The paid alternative (programmatic JSON results, batchable) is **Cloud Vision Web
Detection** — a restricted key lives in `central/.env` (`GCP_VISION_API_KEY`,
project `muser-2605300220`); see the `gcloud` skill in central. ~$59 for a one-time
pass over the 17.9k uniques, so reserved for selective lookups, not wired in.

## Conventions / gotchas

- **Run via uv**: `uv run muser ...` / `uv run python ...`. Editable-install with
  `uv pip install -e .` so the `muser` script and `eval/` imports resolve.
- **Don't reinvent**: metrics = ranx; models = transformers/sentence-transformers;
  index = LanceDB; benchmarks = HF datasets; web = Gradio.
- **Adding a model**: one line in `registry.py`. Baselines are a quality *floor*;
  the shipped default must be a 2026 frontier model.
- Big/corrupt images: `_load_rgb` disables PIL's bomb guard (trusted local files)
  and downscales to ≤1024px before encoding.
- MPS is used automatically on Apple Silicon (CUDA on NVIDIA, CPU otherwise).
- **Cross-platform:** core (index/search/serve/web/CLI/MCP) runs on macOS/Windows/Linux.
  OS-specific bits branch on `platform.system()` and degrade to a clear 501 when the tool
  is absent: `_reveal` (open -R / explorer /select / FileManager1 D-Bus + xdg-open),
  `_copy_image_to_clipboard` (osascript / PowerShell WinForms / wl-copy|xclip),
  `/api/pick-folder` (osascript / FolderBrowserDialog / zenity|kdialog). The MLX backend
  (`jina-v4-mlx`) is Apple-Silicon-only — gated out of the registry via an import-free
  `find_spec` probe, and the `mac` extra carries `sys_platform`/`platform_machine` markers
  so `uv` installs cleanly off-Apple. UI hint copy: "⌘V (macOS) / Ctrl+V (Win/Linux)".
  Verified green on ubuntu/windows/macos via `.github/workflows/ci.yml` (`tests/`).
- **Headless-Linux subprocess gotchas (do NOT reintroduce):** (1) `xclip`/`wl-copy` fork a
  daemon that keeps owning the selection and *inherits stdout* — so `capture_output=True`
  deadlocks `subprocess.run` on pipe EOF forever; use `DEVNULL` + a `timeout`. (2)
  `dbus-send --print-reply` blocks when no FileManager1 service answers; always pass a
  `timeout`. Both surfaced as multi-hour CI hangs before the fix. CI runs pytest with
  `--timeout=180` so any new hang fails fast with a stack trace instead of stalling.
- **Global hide-filter (`_hidden`/`_filter_results`).** Every result-producing endpoint
  funnels through `_filter_results`, which drops hidden paths (and prunes them from each
  result's `dupes[]`) and purges dead paths from the index in the background. `_hidden(path)`
  is O(1): dead-file (always), then — only when the **demo-mode** flag `_DEMO["hide"]` is on
  (default; toggled via GET/POST /api/demo-mode from the web debug menu) — NSFW above
  `NSFW_HIDE_THRESHOLD=0.9` (a rank-normalized *percentile*, so 0.9 = top ~10% most-NSFW,
  NOT a raw probability) and membership in any `HIDDEN_CLUSTER_MATCHERS` HDBSCAN cluster
  (substring-matched against cluster captions; seeded with a people/selfie set). Dead-file
  hiding is unconditional; NSFW + cluster hiding are demo-mode-gated. All three constants are
  module-level for one-line tuning.
- **Query normalization.** Text queries are lowercased before embedding (`q.lower()`) in
  /api/search, /api/search-compose, /api/score, etc. — the SigLIP tokenizer is
  case-sensitive, so "Red Car" and "red car" otherwise embed differently.
- **Multi-concept queries (comma syntax).** `/api/search` splits the query on commas into
  signed concepts (`_parse_concepts`): each is embedded separately and a leading `-`
  subtracts it (vector arithmetic, generalizing the older `neg` param, which is folded in as
  one more negative). `match=blend` (default) sums the positive concept vectors and subtracts
  negatives into ONE query vector (`_blend_vector` — prompt-ensembling, so each concept gets
  equal weight instead of one long string letting the bag-of-words encoder fixate); `match=all`
  is intersection (`_search_match_all`): one kNN per concept, a result scored by the MIN of its
  per-concept similarities (high only when EVERY concept is strongly present), missing concepts
  floored at that concept's weakest pooled hit, negatives subtracted. Both paths share
  `_postprocess` (live-file/dead-group handling + hide policy + enrichment + ai_min/sort),
  factored out of `_run_search`. Single concept + no neg → the unchanged single-vector path.
  Surfaced in web (a Blend/Match-all segmented toggle that appears when the query has ≥2 comma
  concepts), CLI (`muser search "a, b" --match all`), and MCP (`search_images match=`).
- **Known limitation — case-sensitive folder scoping:** the `/api/search?folder=` path
  prefilter compares case-sensitively, but NTFS/APFS are case-insensitive, so scoping with
  altered casing returns 0 results. Proper fix = a normalized `pathkey` column (schema
  change + re-index); not yet done.
- **Search perf — precomputed dedup + path-only projection.** `/api/search` warm latency
  is ~40 ms because of two pieces that look like they could be "simplified" but should not:
  (1) `_search_dedup_precomputed` reuses the dupes-groups already computed in `scores.json`
  to collapse near-duplicates via an O(1) `rep_of` inverse hashmap — instead of LanceDB's
  full-vector over-fetch (288 rows × 1024-dim) + Python pairwise-cosine walk, which costs
  ~200 ms. Falls back to the live cosine-walk in `index.search_dedup` only when
  `method=='phash'/'both'` or before scoring has been run. (2) `MuserIndex.search()` does
  `.select(["path"])` so the LanceDB result projection skips the vector column entirely —
  at limit=288 that drops the scan from ~150 ms to ~10 ms. All current callers only read
  `path` + `_distance`, so this is pure overhead reduction. Don't undo either without
  a measurement showing it helps. C2PA badges are also inlined per result (see
  Architecture above), eliminating ~120 follow-on `/api/c2pa` round-trips per search.
- **Dead-file filter in /api/search.** LanceDB's index is append-by-mtime so deleted
  files linger in the table until a full re-index. `_run_search` filters out results
  whose canonical path is gone, and **promotes a live group member** (pulled from
  `scores.json`'s `dupes` group) when the leading path was deleted — so a search hit
  is only dropped when the entire dedup group is dead. Same logic applies to `/api/score`
  (Interesting/Review tabs), which walks past `offset+limit` to fill the page with live
  entries. ~1 ms overhead per search.

## Status

Default model: **siglip2-b** (Apache, best quality/speed/license — see reports/).
Embedded service + web search UI working (`muser serve` → http://127.0.0.1:7777).
Core (embed/index/search), harness (Flickr30k/COCO/domain + ranx), CLI, and web UI are
working and verified. Color facet search shipped (`color.py`). Skin-tone search was
built then removed (archived at tag `skintone-v4-archived`, see `reports/skintone-archive/`).
**Generate pipeline shipped** (`generators.py`/`pipeline.py`): cart → fal.ai image
gen (nano_banana / gpt-image / both / LoRA), BiRefNet cutout, prompt expansion, and
image→3D (Meshy 6, or Hunyuan3D multiview with synthesized L/B/R views). The generate
pipeline drives fal via the **queue API** with live per-stage fal status on the Jobs page.
**Explore 3D projection** (`projection.py`, space+method aware) and the restyled MCP gallery
(contact-sheet to the LLM) shipped. **Opt-in DINOv2 space** (`dinov2.py`) for Explore +
Similar (217 named clusters). **Jobs page** (`/api/jobs`) for in-flight work. **Masonry**
(Pinterest/Cosmos, `#results`, JS append-only) + infinite scroll on homepage/Similar.
Global demo-mode hide-filter (dead/NSFW/people clusters) on by default; NSFW Review tab
shown in normal mode. Always-on daemon shipped as a macOS LaunchAgent
(`~/Library/LaunchAgents/com.muser.serve.plist`, documented in central per-machine
doc). Benchmarks in `reports/` (latest: siglip2-b vs Cohere Embed v4 on abstract queries —
siglip marginally ahead, Cohere not worth adopting for a photo library). Next: wire
`jina-v4` run (7.5GB download), add Qwen3-VL, VLM-generated ground truth for the user's own
folders; optional per-request fal tracking for concurrent "both"-mode generate.
