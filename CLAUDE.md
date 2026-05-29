# Muser — agent notes

Local-first semantic image search with a built-in retrieval-eval harness.
**Python 3.12 + uv.** (A retired Bun/TS prototype lives in `legacy-ts/`.)

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
  `models`, `web`.
- `eval/datasets.py` — standard benchmarks reduced to {image_paths, queries,
  qrels}. Flickr30k via HF's `refs/convert/parquet` branch (scripts unsupported).
- `eval/harness.py` — embeds corpus → LanceDB → queries → **ranx** metrics
  (hits@1, recall@5/10, mrr, ndcg@10, map) + latency. `format_table` for a comparison.
- `eval/web.py` — Gradio UI: Benchmark tab (run + compare) and Inspect tab
  (query the corpus, gallery flags the ground-truth image).

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

Core (embed/index/search), harness (Flickr30k + ranx), CLI, and web UI are
working and verified. Next: wire `jina-v4` run (7.5GB download), add Qwen3-VL,
VLM-generated ground truth for the user's own folders, embedded-service daemon.
