"""Registry of embedding models the harness can compare.

Add a model here once; it becomes available to the CLI, the index, and the eval
harness. Keys are short stable names used everywhere (and in result tables).

Tiers:
  - baseline : fast, small, 2025-or-earlier. A speed/quality *floor*, not a ship target.
  - frontier : 2026 cutting-edge default.
"""

from __future__ import annotations

from typing import Callable

from .embedders import (
    Embedder,
    JinaCLIPv2Embedder,
    JinaV4Embedder,
    JinaV4MLXEmbedder,
    SentenceTransformerEmbedder,
    SigLIP2Embedder,
)

# name -> (tier, factory)
_REGISTRY: dict[str, tuple[str, Callable[[], Embedder]]] = {
    # --- baselines (floor references) ---
    "clip-b32": ("baseline", lambda: SentenceTransformerEmbedder("clip-b32", "sentence-transformers/clip-ViT-B-32")),
    "clip-l14": ("baseline", lambda: SentenceTransformerEmbedder("clip-l14", "sentence-transformers/clip-ViT-L-14")),
    # SigLIP 2 (Apache-2.0): the quality leaders on Mac — siglip2-b 0.945, so400m 0.955
    # hits@1 on Flickr30k, both beating CLIP and commercial-safe.
    "siglip2-b": ("frontier", lambda: SigLIP2Embedder("siglip2-b", "google/siglip2-base-patch16-512")),
    # Mac-friendly CLIP-architecture models in jina-clip-v2's quality tier (no VLM blowup):
    # jina-clip-v2: 865M, multilingual, strong — but CC-BY-NC-4.0 (non-commercial).
    "jina-clip-v2": ("frontier", lambda: JinaCLIPv2Embedder()),
    # SigLIP2-So400m: ~400M vision, Apache-2.0 — the commercial-safe top option.
    "siglip2-so400m": ("frontier", lambda: SigLIP2Embedder("siglip2-so400m", "google/siglip2-so400m-patch16-512")),
    # --- 2026 frontier ---
    # jina-v4 (transformers): best quality (0.967 Flickr) but leaks MPS memory and
    # hard-crashes on real images — CUDA server only.
    "jina-v4": ("frontier", lambda: JinaV4Embedder()),
    # jina-v4-mlx: runs stably on Apple Silicon, but cross-modal retrieval is broken
    # in this integration (~random Flickr hits@1). EXPERIMENTAL — needs debugging.
    "jina-v4-mlx": ("experimental", lambda: JinaV4MLXEmbedder()),
}

# siglip2-b: best quality/speed/license balance on Mac — 0.945 hits@1 (beats CLIP),
# Apache-2.0, 71ms/img. siglip2-so400m for max quality (0.955) at 269ms/img.
# Frontier jina-v4 is the CUDA-server default (wired with the embedded-service).
DEFAULT_MODEL = "siglip2-b"


def model_names() -> list[str]:
    return list(_REGISTRY)


def model_tier(name: str) -> str:
    return _REGISTRY[name][0]


def load_model(name: str) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(_REGISTRY)}")
    return _REGISTRY[name][1]()
