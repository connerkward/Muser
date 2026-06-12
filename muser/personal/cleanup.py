"""Delete-candidate (junk) detector — a local, $0 quality facet.

Scores how likely an image is *junk you'd delete* from intrinsic, content-free
signals (no model, just pixels):

- **blur**  — variance of the Laplacian (focus measure). Low ⇒ out-of-focus or
  motion-blurred. The single strongest junk signal for a camera roll.
- **exposure** — fraction of near-black / near-white pixels + mean luma. A frame
  that's mostly black (lens-cap, pocket shot) or blown out is junk.
- **tiny** — original megapixels. Thumbnails / tiny saved assets are rarely
  worth keeping in a *photo* library.

`delete_score()` fuses these as the **worst single defect** (a photo that's very
blurry OR very dark OR tiny is a candidate), returning 0–100 + human reasons.
Near-duplicate membership is layered on at the service endpoint from
`scores.json` (it already groups dupes), so this module stays self-contained and
dependency-free beyond opencv (already a core dep via `color.py`).

Built on `facets.Sidecar` (`cleanup.json`, incremental by mtime+size, parallel
scan) like `color.py`/`aidet.py`. Nothing here deletes anything — the Cleanup
tab only *flags* candidates into the Evaluate delete set for the user to confirm.
"""
from __future__ import annotations

import numpy as np

from ..embedders import _load_rgb
from ..facets import Sidecar

_SIDE = Sidecar("cleanup")
SAMPLE = 512          # downscale for the blur/exposure measure (focus survives this)


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def sidecar() -> Sidecar:
    return _SIDE


def _compute(path: str) -> dict:
    import cv2
    from PIL import Image
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass
    # original dimensions (cheap header read) for the tiny-resolution signal
    try:
        with Image.open(path) as im0:
            w0, h0 = im0.size
    except Exception:
        w0 = h0 = 0
    img = _load_rgb(path, max_side=SAMPLE)
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    dark = float((g < 16).mean())
    bright = float((g > 240).mean())
    return {"blur": round(blur, 1), "dark": round(dark, 4), "bright": round(bright, 4),
            "w": int(w0), "h": int(h0), "mp": round(w0 * h0 / 1e6, 3)}


def scan(paths, progress=None, workers: int = 8) -> dict:
    """Build/refresh the junk-signal facet over ``paths`` (incremental)."""
    return _SIDE.scan(paths, _compute, progress=progress, workers=workers)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


# --- scoring thresholds (module-level for one-line tuning) ---
BLUR_HARD = 60.0      # var below this ⇒ clearly blurry
BLUR_SOFT = 140.0     # var below this ⇒ soft (mild signal)
DARK_HARD = 0.70      # ≥70% near-black pixels ⇒ mostly black
DARK_SOFT = 0.50
BRIGHT_HARD = 0.60    # ≥60% near-white ⇒ blown out
MP_TINY = 0.08        # < 0.08 MP (~300×260) ⇒ tiny
MP_SMALL = 0.20


def delete_score(e: dict) -> tuple[int, list[str]]:
    """Fuse one image's junk signals → (score 0–100, reasons). The score is the
    WORST single defect — one bad-enough axis makes it a delete candidate."""
    blur = e.get("blur", 9e9)
    dark = e.get("dark", 0.0)
    bright = e.get("bright", 0.0)
    mp = e.get("mp", 9e9)
    score = 0.0
    reasons: list[str] = []
    if blur < BLUR_SOFT:
        frac = (BLUR_SOFT - blur) / BLUR_SOFT          # 0..1
        score = max(score, 0.40 + 0.55 * min(1.0, frac))
        reasons.append("blurry" if blur < BLUR_HARD else "soft focus")
    if dark >= DARK_HARD:
        score = max(score, 0.88); reasons.append("mostly black")
    elif dark >= DARK_SOFT:
        score = max(score, 0.60); reasons.append("dark")
    if bright >= BRIGHT_HARD:
        score = max(score, 0.82); reasons.append("blown out")
    if mp < MP_TINY:
        score = max(score, 0.78); reasons.append("tiny")
    elif mp < MP_SMALL:
        score = max(score, 0.50); reasons.append("small")
    return int(round(score * 100)), reasons


def candidates(min_score: int = 55, extra: dict | None = None) -> list[dict]:
    """All scored images at/above ``min_score``, worst first. ``extra`` maps
    path -> additional (score_boost, reason) for near-dup membership injected by
    the caller (keeps this module free of scores.json coupling)."""
    extra = extra or {}
    out: list[dict] = []
    for key, e in _SIDE.entries().items():
        path = key[0] if isinstance(key, tuple) else key
        score, reasons = delete_score(e)
        if path in extra:
            boost, reason = extra[path]
            score = max(score, boost)
            if reason and reason not in reasons:
                reasons.append(reason)
        if score >= min_score:
            out.append({"path": path, "score": score, "reasons": reasons,
                        "blur": e.get("blur"), "mp": e.get("mp")})
    out.sort(key=lambda d: -d["score"])
    return out
