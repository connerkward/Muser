"""Model-agnostic embedding interface.

Every embedder maps images and text queries into one shared vector space so that
a natural-language query retrieves matching images by cosine similarity. The
point of the abstraction is the eval harness: register N models behind the same
interface and let the benchmark pick the winner on *your* data.

Heavy deps (torch/transformers/sentence-transformers) are imported lazily inside
methods so `import muser` stays cheap (the CLI lists models without loading any).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np


def _load_rgb(path: str, max_side: int = 1024):
    """Open an image as RGB, tolerating huge local files and downscaling big ones.

    These are the user's own trusted files, so PIL's decompression-bomb guard is
    disabled. Large images are downscaled (embedding models resize to <=512px
    anyway), which also bounds memory and speeds up encoding.
    """
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path)
    img.draft("RGB", (max_side, max_side))  # cheap pre-decode downscale for JPEG
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return img


def _device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _l2(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize so dot product == cosine similarity."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


@runtime_checkable
class Embedder(Protocol):
    """A single-vector image/text embedder in a shared space."""

    name: str
    dim: int

    def embed_images(self, paths: Sequence[str], batch_size: int = 16) -> np.ndarray: ...

    def embed_queries(self, queries: Sequence[str], batch_size: int = 64) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# sentence-transformers backend — covers CLIP-family baselines (ViT-B/32,
# SigLIP, etc.) with one uniform .encode() for both modalities. No custom code.
# ---------------------------------------------------------------------------
@dataclass
class SentenceTransformerEmbedder:
    name: str
    model_id: str
    dim: int = 0  # filled in after load
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=_device(), trust_remote_code=True)
            get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
            self.dim = int(get_dim())
        return self._model

    def embed_images(self, paths: Sequence[str], batch_size: int = 16) -> np.ndarray:
        model = self._load()
        imgs = [_load_rgb(p) for p in paths]
        vecs = model.encode(imgs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return _l2(np.asarray(vecs, dtype=np.float32))

    def embed_queries(self, queries: Sequence[str], batch_size: int = 64) -> np.ndarray:
        model = self._load()
        vecs = model.encode(list(queries), batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        return _l2(np.asarray(vecs, dtype=np.float32))


# ---------------------------------------------------------------------------
# Jina Embeddings v4 — 2026 frontier multimodal embedder (Qwen2.5-VL backbone).
# Uses the official task-aware API via transformers AutoModel (trust_remote_code).
# Single-vector mode here; the model also supports multi-vector late interaction.
# ---------------------------------------------------------------------------
@dataclass
class JinaV4Embedder:
    name: str = "jina-v4"
    model_id: str = "jinaai/jina-embeddings-v4"
    task: str = "retrieval"
    dim: int = 2048
    _model: object = field(default=None, repr=False)

    def _load(self):
        if self._model is None:
            import torch
            from PIL import Image
            from transformers import AutoModel

            Image.MAX_IMAGE_PIXELS = None  # trusted local files
            dev = _device()
            # fp16 autocast is broken on MPS ("Unexpected floating ScalarType");
            # use fp32 off CUDA by default — reliable. Override via MUSER_JINA_DTYPE
            # (e.g. "bfloat16" on MPS for ~half the memory / faster).
            import os

            override = os.environ.get("MUSER_JINA_DTYPE")
            dtype = getattr(torch, override) if override else (torch.float16 if dev == "cuda" else torch.float32)
            self._model = (
                AutoModel.from_pretrained(self.model_id, trust_remote_code=True, dtype=dtype)
                .to(dev)
                .eval()
            )
        return self._model

    @staticmethod
    def _to_np(out) -> np.ndarray:
        """encode_* returns a list of (float16) tensors; stack to float32 ndarray."""
        return np.stack([t.float().cpu().numpy() for t in out]).astype(np.float32)

    def embed_images(self, paths: Sequence[str], batch_size: int = 8) -> np.ndarray:
        model = self._load()
        out = []
        for i in range(0, len(paths), batch_size):
            chunk = list(paths[i : i + batch_size])
            out.append(self._to_np(model.encode_image(images=chunk, task=self.task, batch_size=len(chunk))))
        v = np.concatenate(out, axis=0)
        self.dim = v.shape[-1]
        return _l2(v)

    def embed_queries(self, queries: Sequence[str], batch_size: int = 32) -> np.ndarray:
        model = self._load()
        out = []
        for i in range(0, len(queries), batch_size):
            chunk = list(queries[i : i + batch_size])
            out.append(
                self._to_np(
                    model.encode_text(texts=chunk, task=self.task, prompt_name="query", batch_size=len(chunk))
                )
            )
        v = np.concatenate(out, axis=0)
        self.dim = v.shape[-1]
        return _l2(v)
