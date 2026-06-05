# Muser — agent notes

Local-first semantic image search with a built-in retrieval-eval harness.
**Python 3.12 + uv.** A TypeScript MCP ext-app lives in `mcp-ts/` — a thin HTTP
client of the embedded service (`muser serve`) that renders results in an
interactive gallery UI inside Claude Desktop. It loads no model and never touches
LanceDB; it proxies `/api/search`, `/api/thumb`, `/api/index`, `/api/status`.

See `REQUIREMENTS.md` for scope/decisions.

## Architecture

- `muser/embedders.py` — model-agnostic image/text embedders in a shared space.
  Backends: `SentenceTransformerEmbedder` (CLIP/SigLIP baselines) and
  `JinaV4Embedder` (2026 frontier default). Heavy deps lazy-imported.
- `muser/registry.py` — `name -> (tier, factory)` model registry. Add a model
  once → it appears in CLI, index, and benchmark. `DEFAULT_MODEL = "jina-v4"`.
- `muser/index.py` — `MuserIndex`: one embedded LanceDB at `~/.muser/db`, one
  table per model (`img__<model>`), cosine over L2-normalized vectors.
  Incremental by mtime; skips corrupt files.
- `muser/cli.py` — `muser` entrypoint (typer): `index`, `search`, `bench`,
  `models`, `cluster`, `score`, `detect`, `serve`, `web`. `detect` runs the C2PA
  library scan headless (writes `~/.muser/c2pa.json`; same data the web "AI" tab uses).
- `muser/service.py` — **embedded service** (`muser serve`): FastAPI app that warms
  the model once and owns the index; serves JSON API + the web search UI
  (`muser/web/app.html`). Warm search ≈ 40 ms (precomputed dedup, see below).
  Endpoints: /api/search, /api/index,
  /api/thumb (PIL), /api/image, /api/reveal (open -R), /api/model, /api/status,
  /api/folders, /api/c2pa, /api/ai, /api/ai/scan. **AI-origin detection (C2PA):**
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
- `muser/jobs.py` — tiny thread-safe **in-process job registry** (`REGISTRY`) backing
  the non-blocking cart **checkout**. `POST /api/checkout {items:[{path,upscale}],
  caption_prompt?, force_caption?}` returns a `job_id` and runs captioning (network,
  OpenAI) **concurrently** with upscaling (local 4×, flagged items only) in a daemon
  thread — deliberately NOT on the single `state.task` slot, so search/index stay
  usable while it runs. `GET /api/checkout/status?id=` reports `{status, caption:{done,
  total}, upscale:{done,total}, errors, captioned, upscaled, captions_missing,
  zip_ready}`; `GET /api/checkout/zip?id=` serves the prebuilt zip. The job re-loads
  captions.jsonl before building the zip, so freshly-captioned items land in the
  bundle (fixes the prior all-null-caption export, where "Download zip" only bundled
  pre-cached captions). Zip-building is factored into `_build_cart_zip()` shared by
  `/api/zip` and the checkout job. Upscaled bytes are NOT durably stored (cheap to
  recompute — only the existing `~/.muser/upscale_cache` warm cache). Web UI: a
  dismissible bottom-right progress panel (+ minimizable pill) polls status; the
  optional caption-prompt textarea persists to `localStorage["muser-caption-prompt"]`.
- `muser/facets.py` — shared sidecar scaffolding for **per-image precomputed facets**
  (the c2pa.py cache pattern factored out): a `~/.muser/<name>.json` keyed by path with
  `m`(mtime_ns)+`s`(size) for incremental skip, a thread-pool `scan(paths, compute)`, and a
  RAM cache primed once for O(1) enrichment/ranking. c2pa.py keeps its own copy (shipped,
  untouched); new facets build on this.
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
Always-on daemon shipped as a macOS LaunchAgent
(`~/Library/LaunchAgents/com.muser.serve.plist`, documented in central per-machine
doc). Next: wire `jina-v4` run (7.5GB download), add Qwen3-VL, VLM-generated ground
truth for the user's own folders.
