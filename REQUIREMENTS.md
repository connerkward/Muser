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

### Measured (Flickr30k, this session) + hardware guidance
| model | hits@1 | ndcg@10 | ms/img (M-series MPS) |
|---|---|---|---|
| jina-v4 | 0.967 | 0.986 | ~4000 (fp32) |
| clip-l14 | 0.893 | 0.952 | ~90 |
| clip-b32 | 0.853 | 0.930 | ~110 |

- jina-v4 is best **but ~45× slower** on MPS (every 2026 frontier model is VLM-scale
  → slow off CUDA). fp16 **crashes on MPS** (autocast); bf16 is ~3× slower than fp32.
  → **fp32 off CUDA** (`MUSER_JINA_DTYPE` to override).
- **Policy**: all models usable on Mac (`--model`); for bulk-indexing a large corpus
  on Mac, prefer `clip-l14`/`siglip2-b`; run `jina-v4` on a CUDA server (fp16, fast)
  via the embedded-service. Optional: re-rank top-K with jina-v4.

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
