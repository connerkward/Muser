# Muser — Embedding Throughput, Time & Cost

Indexing cost/time for the candidate models, **this Mac vs. a rented H100**, with
extrapolations across corpus sizes. Goal: decide *what to run where*.

> **How to read this:** "ms/img" is per-image embed time. Mac numbers are
> **measured on this hardware** (Apple Silicon, 64 GB, MPS, single stream, small
> batch). H100 numbers are **researched estimates** (bf16, batch 32–64) — image
> encoders aren't benchmarked in isolation publicly, so these are FLOPs/anchor
> estimates (see Sources); treat as ±40%. Both columns are *pure embed* — real
> end-to-end indexing adds image-decode + LanceDB write (~1.5–2× at scale) plus a
> one-time model load (seconds). The relative picture is what matters.

## Throughput

| Model | Params | Mac ms/img (measured) | Mac img/s | H100 img/s (est.) | H100 ms/img |
|---|---|---|---|---|---|
| clip-b32 (ViT-B/32) | 150M | **24** | 41.7 | ~9,000 | 0.11 |
| clip-l14 (ViT-L/14) | 430M | **46** | 21.7 | ~3,000 | 0.33 |
| **siglip2-b** | 375M | **71** | 14.1 | ~6,000 | 0.17 |
| siglip2-so400m | 1.1B | **269** | 3.7 | ~2,500 | 0.40 |
| **jina-v4 (3.75B VLM)** | 3.75B | ~2,100 (MLX)¹ / ~4,000 (torch)² | 0.47 | **~30** | ~33 |

¹ jina-v4-mlx runs stably on Mac but its cross-modal retrieval is currently broken
(see REQUIREMENTS). ² jina-v4 torch crashes on MPS with real images. **On Mac,
jina-v4 is effectively unusable today**; numbers shown for the H100 comparison.

The headline: a 300–400M CLIP/SigLIP encoder is **~50–300× faster per image** than
the 3.75B VLM. The VLM runs its full 3B decoder over ~1,000 image tokens; CLIP runs
one ViT over ~256 tokens with no decoder.

## Time to index a corpus

**On this Mac** (measured ms/img, single machine):

| Model | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| clip-b32 | 24 s | 4.0 min | 40 min | 6.7 hr |
| **clip-l14** | 46 s | 7.7 min | 1.3 hr | 12.8 hr |
| siglip2-b | 71 s | 11.8 min | 2.0 hr | 19.7 hr |
| siglip2-so400m | 4.5 min | 45 min | 7.5 hr | 3.1 days |
| jina-v4 (MLX) | 35 min | 5.8 hr | 2.4 days | 24 days |

**On one H100** (batched, est.):

| Model | 1k | 10k | 100k | 1M |
|---|---|---|---|---|
| clip-l14 | 0.3 s | 3.3 s | 33 s | 5.6 min |
| siglip2-b | 0.2 s | 1.7 s | 17 s | 2.8 min |
| jina-v4 | 33 s | 5.6 min | 56 min | 9.3 hr |

## H100 cost

Rented H100: **~$2.50/hr on-demand, ~$1.50/hr spot/marketplace** (RunPod/Vast/Lambda;
per-second billing on Modal/RunPod-serverless avoids paying for idle). For CLIP/SigLIP
the job is seconds–minutes, so cost is rounding error (≪$0.10, min-billing dominates).
Cost only matters for the **jina-v4 VLM**:

| Corpus | jina-v4 H100 time | On-demand ($2.50) | Spot ($1.50) | + fp8 (~2× faster) |
|---|---|---|---|---|
| 10k | 5.6 min | $0.23 | $0.14 | ~$0.07–0.12 |
| 100k | 56 min | $2.32 | $1.39 | ~$0.70–1.16 |
| 1M | 9.3 hr | $23.1 | $13.9 | ~$7–12 |

(CLIP/SigLIP at 1M on H100: ~3–6 min → a few cents.)

## Mac vs H100 — the crossover

| | CLIP / SigLIP | jina-v4 (3.75B VLM) |
|---|---|---|
| H100 speedup vs Mac | ~65–140× | ~60–130× |
| Mac practical ceiling | **~1M images (overnight)** | **~a few thousand** |
| When to rent H100 | rarely — Mac is already fast & free | **always, for any real corpus** |
| 1M-image cost | $0 (Mac, overnight) | $0 on Mac is 24 days+broken → **rent: ~$14 spot, 9 hr** |

## Recommendations

1. **CLIP/SigLIP → run on Mac.** clip-l14 indexes 100k images in 1.3 hr and 1M
   overnight (12.8 hr), for free. clip-b32 is ~2× faster if you want speed over
   quality. No GPU rental is justified for these.
2. **jina-v4 (frontier quality) → rent an H100.** On Mac it's 24 days for 1M and
   currently broken; on a spot H100 it's **~9 hr for ~$14**. fp8 (TensorRT-LLM/vLLM)
   roughly halves both. This is the embedded-service / server path.
3. **Batch jobs are per-second-billable** (Modal/RunPod serverless) — spin up an
   H100, embed, tear down; you pay only for the ~minutes–hours used, not an idle hour.
4. **Practical default:** clip-l14 on Mac for everyday indexing; reserve an H100
   burst for when you want to re-embed the whole corpus with frontier jina-v4 quality.

## Caveats / honesty

- Mac ms/img is **pure embed**; real end-to-end indexing measured ~1.5–2× slower
  (image decode + LanceDB write), e.g. clip-l14 ~87 ms/img end-to-end on 600 images.
  Apply the same factor to the time tables for real wall-clock.
- H100 throughput is **estimated** (no public isolated image-encoder benchmarks);
  the VLM number is anchored on ColPali's measured 0.39 s/page (L4) scaled to H100.
  Biggest swing for the VLM is the configured image-token budget (resolution).
- Mac numbers are single-stream; MPS batches poorly vs CUDA, which is *why* the
  H100 advantage is large for the small encoders despite their tiny size.

## Sources
- H100/L4 specs & ColPali anchor: [ColPali (arXiv 2407.01449)](https://arxiv.org/html/2407.01449v5),
  [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/), [L4](https://www.nvidia.com/en-us/data-center/l4/).
- Model FLOPs/params: [timm ViT-L/14](https://huggingface.co/timm/vit_large_patch14_clip_224.openai),
  [SigLIP2](https://huggingface.co/blog/siglip2), [jina-v4 (arXiv 2506.18902)](https://arxiv.org/abs/2506.18902).
- H100 pricing (May 2026): [Thunder Compute](https://www.thundercompute.com/blog/nvidia-h100-pricing),
  [RunPod](https://www.runpod.io/pricing), [Vast.ai](https://vast.ai/pricing), [Modal](https://modal.com/pricing).
- Mac throughput: measured this session (`/tmp/throughput.py`, 100 z-to-sort images, MPS).
