# siglip2-b vs Cohere Embed v4 — abstract/stylistic query benchmark

**Date:** 2026-06-07
**Question:** Does a frontier *multimodal VLM-grade* embedder (Cohere Embed v4) beat
the shipped local default (siglip2-b) on the abstract/mood queries that are SigLIP's
known weak spot?

> Originally planned as Cohere vs **GME-Qwen2-VL-2B**. GME was dropped: it can't run
> on this Mac (transformers conflict + a Qwen2-VL/MPS forward-pass deadlock — model
> loads 6 GB then hangs at 0 % CPU). Cohere Embed v4 is the practical frontier
> multimodal embedder, so it replaced GME as the comparison arm.

## Setup

- **Corpus:** identical 4,000-image stride sample of the user's library
  (`~/.muser/bench_sample.json`), embedded by *both* models.
  - siglip2-b vectors pulled from the live LanceDB table (`img__siglip2_b`).
  - Cohere vectors via `embed-v4.0` REST (`input_type=search_document`, image content
    blocks, 512 px JPEG q80), float embeddings, dim 1536.
- **Queries:** 28 abstract/stylistic prompts (`~/.muser/bench_queries.json`) —
  "moody cinematic lighting", "minimalist composition", "nostalgic analog
  photography", "elegant luxury feel", etc. siglip queries lowercased (matches the
  service); Cohere `input_type=search_query`.
- **Ranking:** in-memory cosine over the same 4,000 corpus per model, top-10.
- **Judge:** gpt-4o, low-detail, per (query,image) → {0 irrelevant, 1 somewhat,
  2 highly}. Cached per pair (siglip/cohere top-10 overlap dedupes → 488 unique
  judgments). P@10 counts rel ≥ 1; nDCG@10 uses gains 2^rel−1.

## Result

| Model | P@10 | nDCG@10 |
|---|---:|---:|
| **siglip2-b** (local, Apache-2.0, $0) | **0.961** | **0.960** |
| Cohere Embed v4 (cloud, paid) | 0.943 | 0.937 |

**siglip2-b wins, marginally.** The two are effectively tied; the stronger/larger
multimodal model did **not** close any abstract-query gap — it was slightly behind.

## Cost (actual)

- **Cohere:** 784,756 image tokens + 131 text tokens → **$0.094** at $0.12/1M.
  ≈ 196 image-tokens/image on average (a simple smoke image billed only 56; real
  library images tile higher). Full 27.6k-image library would be ≈ **$0.66**.
- **gpt-4o judge:** 488 calls ≈ **$0.29**.
- Total this run ≈ **$0.39**. Cohere $20 spend cap untouched.

## Caveats

- **Ceiling effect.** Both models score ~0.95 because a 4,000-image corpus + a
  lenient bar (rel ≥ 1 counts) leaves the top-10 easy to fill with "somewhat
  relevant" hits. This is why the absolute numbers are far above the earlier
  full-library abstract bench (P@10 0.31, `bench-abstract.md`): that ran over the
  full 27.6k with stricter conditions. The **relative** comparison here is still
  valid — same corpus, same judge, same queries for both models — but it doesn't
  finely discriminate at the top. A discriminating rerun would use the full 27.6k
  corpus and/or a strict rel == 2 bar.

## Conclusion

For Muser's abstract/stylistic retrieval, **Cohere Embed v4 is not worth adopting**:
it ties (slightly loses to) the local siglip2-b while adding a cloud dependency,
per-query latency, and ongoing cost. The abstract-query weakness is **not** an
embedder-capacity problem a bigger model fixes — it's the contrastive-embedding
paradigm. The earlier recommendation stands: attack abstract queries with
**caption-as-text / BM25 hybrid + relevance feedback**, not a heavier embedder.
**siglip2-b remains the right default.**
