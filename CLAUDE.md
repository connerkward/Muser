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
  `models`, `serve`, `web`.
- `muser/service.py` — **embedded service** (`muser serve`): FastAPI app that warms
  the model once and owns the index; serves JSON API + the web search UI
  (`muser/web/app.html`). Warm search ≈ 30 ms. Endpoints: /api/search, /api/index,
  /api/thumb (PIL), /api/image, /api/reveal (open -R), /api/model, /api/status.
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
- MPS is used automatically on Apple Silicon.

## Status

Default model: **siglip2-b** (Apache, best quality/speed/license — see reports/).
Embedded service + web search UI working (`muser serve` → http://127.0.0.1:7777).
Core (embed/index/search), harness (Flickr30k/COCO/domain + ranx), CLI, and web UI are
working and verified. Next: wire `jina-v4` run (7.5GB download), add Qwen3-VL,
VLM-generated ground truth for the user's own folders, embedded-service daemon.
