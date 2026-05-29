# Muser — Requirements (v0)

Local-first semantic image search with a built-in retrieval-evaluation harness.
Concise by design; expands as we lock decisions.

## Goal
Point Muser at a folder of images, embed them, and retrieve by **any** query
modality. Ship a model that is **2026-cutting-edge** *and* **reliable**, proven
by an **automated benchmark** so quality is never eyeballed.

## Scope — now
- Index a folder of images (incremental: add/update/remove by mtime).
- Semantic search: **text → image** and **image → image** (sketch, pose-image,
  reference photo are all image queries into the shared space).
- **Auto eval harness** + **web UI** to run/inspect benchmarks. (Primary near-term deliverable.)
- Surfaces: **CLI** + **web interface**. (MCP server later.)

## Scope — later (not now, but must not be precluded)
- Color / skin-tone search (separate LAB color-histogram index — not the embedder).
- Composed queries ("this but red") via composed-retrieval or vector arithmetic.
- Dedup (pHash → PDQ), captioning (BLIP/LLaVA), aesthetics (PickScore).
- Provenance/lineage graph; ComfyUI nodes; Electron product UI; embedding-map viz.

## Non-negotiables (from notes + this session)
- **Cross-platform**: mac / Windows / Linux servers. Python 3.12, ONNX/torch.
- **Local-first**: runs on Apple Silicon / consumer GPU; no mandatory cloud.
- **Don't reinvent**: use `open_clip`/`transformers`/`sentence-transformers`,
  LanceDB, `ranx`/`mteb`, HF `datasets`, Gradio. Wire, don't rebuild.
- **Embedded-service** architecture: one long-lived process owns models + index +
  watcher; CLI/web/MCP are thin clients (no OS service-manager dependency).
- **Offload to OS/filesystem**: don't rebuild file browser/thumbnails/folders.

## Architecture
```
core: embedders (model-agnostic) → LanceDB index (~/.muser/db, table per model)
        ▲                              ▲
        └──────── eval harness ────────┘   (ranx + standard benchmarks + VLM-gen GT)
surfaces: CLI · web (Gradio) · [MCP later]  → thin clients of the core/service
```

## Models (scaffold all behind one interface; harness picks)
- **frontier / default**: `jina-embeddings-v4` (Qwen2.5-VL-3B; single + multi-vector). 2026.
- **challengers**: Qwen3-VL-2B, Nemotron-ColEmbed-v2 (docs only).
- **baselines (floor)**: CLIP ViT-B/32, ViT-L/14, SigLIP 2-base.

## Evaluation (the harness)
- **Reuse standard benchmarks**: MIEB/MTEB, Flickr30k, MS-COCO-5k, CIRR/Fashion-IQ
  (composed), ViDoRe (docs, optional). Metrics via `ranx`: Recall@1/5/10, MRR, nDCG, mAP.
- **Domain eval on your folder**: VLM-generated captions → queries whose known
  answer is the source image (auto ground truth, no hand-labeling) + hard negatives.
- **Output**: comparison table across {model × method}, latency + index size,
  regression gate (fail if Recall@5 drops > threshold) so it doubles as CI.

## Success criteria
- One command indexes a folder and answers text + image queries.
- One command runs the benchmark and prints a model-comparison table.
- Adding a model = one registry entry; it appears in CLI, index, and benchmark.
