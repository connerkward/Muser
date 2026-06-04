"""Per-image skin-tone index + search on the Monk Skin Tone (MST) scale.

Accurate-by-construction: skin is sampled from *detected faces* (OpenCV's YuNet
detector, a tiny bundled ONNX model — local, no download), not from a global
skin-color guess that sand/wood/lighting would trip. Within each face box a
YCrCb skin mask isolates skin pixels; their median color → CIE-LAB → the nearest
of the 10 official **Monk Skin Tone** swatches (Google's MST scale).

Persisted to ``~/.muser/skintone.json``: per image, one entry per detected face
(``mst`` 1–10, sampled ``lab``, ``frac`` of frame, detector ``conf``) plus a
``mst`` summary (the most prominent face's tone). Search by tone 1–10 ranks
images whose faces are closest to that tone, weighted by face prominence.

Honest limits: detection is the ceiling — undetected faces (heavy occlusion,
extreme angle, tiny) contribute nothing; strong color casts shift sampled tone.
Positive signal, not a demographic classifier.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .embedders import _load_rgb
from .facets import Sidecar

_SIDE = Sidecar("skintone")
_MODEL_PATH = Path(__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"

# Official Monk Skin Tone 10-point scale (Google), sRGB hex, light → dark.
MST_HEX = [
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420",
]

DET_SIDE = 640           # downscale longest side before detection (speed)
DET_CONF = 0.6           # YuNet score threshold
MIN_SKIN_PX = 30         # need at least this many masked skin pixels to sample a tone

_mst_lab = None
_tls = threading.local()  # one detector per worker thread (YuNet isn't thread-safe)


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return _MODEL_PATH.exists()


def sidecar() -> Sidecar:
    return _SIDE


def _mst_lab_arr():
    global _mst_lab
    if _mst_lab is None:
        import cv2

        rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in MST_HEX], np.uint8)
        _mst_lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return _mst_lab


def _detector():
    d = getattr(_tls, "det", None)
    if d is None:
        import cv2

        d = cv2.FaceDetectorYN.create(str(_MODEL_PATH), "", (320, 320), DET_CONF)
        _tls.det = d
    return d


def _detect_and_tone(path: str) -> dict:
    import cv2

    img = np.asarray(_load_rgb(path, max_side=DET_SIDE).convert("RGB"))
    h, w = img.shape[:2]
    det = _detector()
    det.setInputSize((w, h))
    _, faces = det.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if faces is None:
        return {"faces": [], "mst": None}
    mst_lab = _mst_lab_arr()
    frame_area = float(w * h) or 1.0
    out = []
    for f in faces:
        x, y, fw, fh = (int(v) for v in f[:4])
        conf = float(f[-1])
        x, y = max(0, x), max(0, y)
        crop = img[y:y + fh, x:x + fw]
        if crop.size == 0:
            continue
        ycc = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
        cr, cb = ycc[:, :, 1], ycc[:, :, 2]
        mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
        pix = crop[mask]
        if len(pix) < MIN_SKIN_PX:
            continue
        med = np.median(pix, axis=0).astype(np.uint8)
        lab = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        mst = int(np.argmin(np.linalg.norm(mst_lab - lab, axis=1))) + 1
        out.append({
            "mst": mst,
            "lab": [int(lab[0]), int(lab[1]), int(lab[2])],
            "frac": round((fw * fh) / frame_area, 4),
            "conf": round(conf, 3),
        })
    out.sort(key=lambda d: -d["frac"])
    return {"faces": out, "mst": (out[0]["mst"] if out else None)}


def scan(paths, progress=None, workers: int = 6) -> dict:
    """Build/refresh the skin-tone index over ``paths`` (incremental)."""
    return _SIDE.scan(paths, _detect_and_tone, progress=progress, workers=workers)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def cache_exists() -> bool:
    return _SIDE.exists()


def tone_histogram() -> list[int]:
    """Count of images whose most-prominent face falls in each MST bin (1..10)."""
    hist = [0] * 10
    for _key, e in _SIDE.entries().items():
        m = e.get("mst")
        if m and 1 <= m <= 10:
            hist[m - 1] += 1
    return hist


def search(tone: int, k: int = 24, folder: str | None = None) -> list[tuple[str, float]]:
    """Rank indexed images by closeness to MST ``tone`` (1..10), face-prominence weighted."""
    pre = os.path.join(folder, "") if folder else None
    scored: list[tuple[str, float]] = []
    for (p, _m, _s), e in _SIDE.entries().items():
        if pre and not (p == folder or p.startswith(pre)):
            continue
        faces = e.get("faces") or []
        best = 0.0
        for fc in faces:
            close = 1.0 - abs(fc["mst"] - tone) / 9.0          # 1.0 at exact tone
            best = max(best, close * (0.5 + 0.5 * fc["frac"]))  # weight by face prominence
        if best > 0:
            scored.append((p, best))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
