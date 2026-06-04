"""Per-image dominant-color (LAB palette) index + color search.

A *separate* color index — not the CLIP/SigLIP embedder (REQUIREMENTS.md). For
each image we extract a small dominant-color **palette** (median-cut quantize,
counts → fractions), convert each swatch to CIE-LAB (perceptually uniform, so
distance ≈ perceived difference), and persist it to ``~/.muser/color.json``.

Search by a target RGB color: score every image by how much of it sits near the
query color — ``sum_i frac_i · sim(palette_i, query)`` where ``sim`` falls off
with LAB ΔE. An image that is *mostly* the query color scores ~1; a thin accent
of it scores low. Fully local, $0, no model load.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .embedders import _load_rgb
from .facets import Sidecar

_SIDE = Sidecar("color")

N_COLORS = 6      # palette size (median-cut)
SAMPLE = 200      # downscale longest side before quantizing — palette is robust to it
# LAB here is OpenCV's 8-bit encoding (L,a,b ∈ 0..255). ΔE ~ Euclidean distance.
# TAU sets how fast similarity decays with distance: at ΔE == TAU, sim == 0.
COLOR_TAU = 70.0


def available() -> bool:
    return True


def sidecar() -> Sidecar:
    return _SIDE


def _rgb_to_lab(rgb_rows: np.ndarray) -> np.ndarray:
    """(K,3) uint8 RGB -> (K,3) float32 OpenCV-LAB."""
    import cv2

    return cv2.cvtColor(rgb_rows.reshape(1, -1, 3).astype(np.uint8),
                        cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def _palette(path: str) -> dict:
    """Dominant-color palette for one image: [[L, a, b, fraction], ...] desc by fraction."""
    img = _load_rgb(path, max_side=SAMPLE).convert("RGB")
    q = img.quantize(colors=N_COLORS, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()                       # flat [r,g,b, r,g,b, ...]
    counts = q.getcolors() or []               # [(count, palette_index), ...]
    if not counts:
        return {"palette": []}
    total = sum(c for c, _ in counts) or 1
    rgb = np.array([pal[i * 3:i * 3 + 3] for _, i in counts], np.uint8)
    lab = _rgb_to_lab(rgb)
    palette = [
        [int(l), int(a), int(b), round(cnt / total, 4)]
        for (cnt, _), (l, a, b) in zip(counts, lab)
    ]
    palette.sort(key=lambda x: -x[3])
    return {"palette": palette}


def scan(paths, progress=None, workers: int = 8) -> dict:
    """Build/refresh the color palette index over ``paths`` (incremental)."""
    return _SIDE.scan(paths, _palette, progress=progress, workers=workers)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


def search(rgb, k: int = 24, folder: str | None = None) -> list[tuple[str, float]]:
    """Rank indexed images by presence of the query RGB color. Returns [(path, score)]."""
    qlab = _rgb_to_lab(np.array([rgb], np.uint8))[0]
    pre = os.path.join(folder, "") if folder else None
    scored: list[tuple[str, float]] = []
    for (p, _m, _s), e in _SIDE.entries().items():
        if pre and not (p == folder or p.startswith(pre)):
            continue
        pal = e.get("palette") or []
        if not pal:
            continue
        score = 0.0
        for l, a, b, frac in pal:
            de = float(np.sqrt((l - qlab[0]) ** 2 + (a - qlab[1]) ** 2 + (b - qlab[2]) ** 2))
            sim = 1.0 - de / COLOR_TAU
            if sim > 0:
                score += frac * sim
        if score > 0:
            scored.append((p, min(1.0, score)))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
