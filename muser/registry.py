"""Registry of embedding models the harness can compare.

Add a model here once; it becomes available to the CLI, the index, and the eval
harness. Keys are short stable names used everywhere (and in result tables).

Tiers:
  - baseline : fast, small, 2025-or-earlier. A speed/quality *floor*, not a ship target.
  - frontier : 2026 cutting-edge default.
"""

from __future__ import annotations

from typing import Callable

from .embedders import Embedder, JinaV4Embedder, SentenceTransformerEmbedder

# name -> (tier, factory)
_REGISTRY: dict[str, tuple[str, Callable[[], Embedder]]] = {
    # --- baselines (floor references) ---
    "clip-b32": ("baseline", lambda: SentenceTransformerEmbedder("clip-b32", "sentence-transformers/clip-ViT-B-32")),
    "clip-l14": ("baseline", lambda: SentenceTransformerEmbedder("clip-l14", "sentence-transformers/clip-ViT-L-14")),
    "siglip2-b": ("baseline", lambda: SentenceTransformerEmbedder("siglip2-b", "google/siglip2-base-patch16-512")),
    # --- 2026 frontier (default) ---
    "jina-v4": ("frontier", lambda: JinaV4Embedder()),
}

DEFAULT_MODEL = "jina-v4"


def model_names() -> list[str]:
    return list(_REGISTRY)


def model_tier(name: str) -> str:
    return _REGISTRY[name][0]


def load_model(name: str) -> Embedder:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(_REGISTRY)}")
    return _REGISTRY[name][1]()
