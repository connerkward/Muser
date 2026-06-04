"""Per-image skin-tone index + search on the Monk Skin Tone (MST) scale.

Two detectors, so a *person* — not just a face — anchors the skin sample:

1. **Face** (OpenCV YuNet, tiny bundled ONNX) — the high-confidence source.
   Face skin is the most consistent tone reference (it's what the Monk-scale
   work samples), so faces are always preferred.
2. **Person** (OpenCV MobileNet-SSD, bundled Caffe model) — the fallback for
   people whose face isn't visible (turned away, profile, distant, occluded).
   We sample skin from *any* visible skin inside the person box (arms, hands,
   legs, neck), but only for person boxes that do **not** already contain a
   detected face (so a face-bearing person is sampled from the face, never
   double-counted).

Within each box a YCrCb skin mask isolates skin pixels; their median → CIE-LAB →
nearest of the 10 official Monk Skin Tone swatches. Persisted to
``~/.muser/skintone.json``: per image a list of detections (``src`` "face"/"body",
``mst`` 1–10, ``lab``, ``frac`` of frame, ``conf``) + a ``mst`` summary (the
most-prominent face, else the most-prominent body). Search by tone 1–10 ranks by
closeness × prominence, with body detections weighted lower than faces.

Honest limits: detection is the ceiling, AND a detected person with **no visible
skin** (fully clothed, back turned in a coat) yields no tone — "person found" ≠
"tone available". Strong color casts shift the sampled tone. A positive signal,
not a demographic classifier.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .embedders import _load_rgb
from .facets import Sidecar

_SIDE = Sidecar("skintone")
_ASSETS = Path(__file__).parent / "assets"
_FACE_MODEL = _ASSETS / "face_detection_yunet_2023mar.onnx"
_PERSON_PROTO = _ASSETS / "MobileNetSSD_deploy.prototxt"
_PERSON_MODEL = _ASSETS / "MobileNetSSD_deploy.caffemodel"

# Bump when the detection/sampling algorithm changes so the incremental scan
# recomputes already-indexed images instead of skipping them by mtime.
SCAN_VERSION = 2  # v1 = face-only; v2 = face + person-fallback

# Official Monk Skin Tone 10-point scale (Google), sRGB hex, light → dark.
MST_HEX = [
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420",
]

DET_SIDE = 640           # downscale longest side before detection (speed)
DET_CONF = 0.6           # YuNet face score threshold
PERSON_CONF = 0.45       # MobileNet-SSD person score threshold
PERSON_CLASS = 15        # "person" in the VOC label set MobileNet-SSD was trained on
MIN_SKIN_PX = 30         # need at least this many masked skin pixels to sample a tone
BODY_WEIGHT = 0.6        # body-sampled tones rank below face-sampled ones

_mst_lab = None
_tls = threading.local()  # detectors are per-thread (not thread-safe to share)


def available() -> bool:
    # Face detection is the floor; person detection is an optional enhancement.
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return _FACE_MODEL.exists()


def person_available() -> bool:
    return _PERSON_PROTO.exists() and _PERSON_MODEL.exists()


def sidecar() -> Sidecar:
    return _SIDE


def _mst_lab_arr():
    global _mst_lab
    if _mst_lab is None:
        import cv2

        rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in MST_HEX], np.uint8)
        _mst_lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return _mst_lab


def _face_detector():
    d = getattr(_tls, "face", None)
    if d is None:
        import cv2

        d = cv2.FaceDetectorYN.create(str(_FACE_MODEL), "", (320, 320), DET_CONF)
        _tls.face = d
    return d


def _person_net():
    if not person_available():
        return None
    n = getattr(_tls, "person", None)
    if n is None:
        import cv2

        n = cv2.dnn.readNetFromCaffe(str(_PERSON_PROTO), str(_PERSON_MODEL))
        _tls.person = n
    return n


def _tone_from_crop(crop) -> tuple[int, list[int]] | None:
    """Median skin tone within a crop → (mst, lab) or None if too little skin."""
    import cv2

    if crop.size == 0:
        return None
    ycc = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    mask = (cr >= 135) & (cr <= 180) & (cb >= 85) & (cb <= 135)
    pix = crop[mask]
    if len(pix) < MIN_SKIN_PX:
        return None
    med = np.median(pix, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    mst = int(np.argmin(np.linalg.norm(_mst_lab_arr() - lab, axis=1))) + 1
    return mst, [int(lab[0]), int(lab[1]), int(lab[2])]


def _detect_and_tone(path: str) -> dict:
    import cv2

    img = np.asarray(_load_rgb(path, max_side=DET_SIDE).convert("RGB"))
    h, w = img.shape[:2]
    frame_area = float(w * h) or 1.0
    dets = []
    face_centers = []  # to suppress person boxes that already contain a face

    # --- 1. faces (preferred, high-confidence skin source) ---
    fd = _face_detector()
    fd.setInputSize((w, h))
    _, faces = fd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for f in (faces if faces is not None else []):
        x, y, fw, fh = (int(v) for v in f[:4])
        x, y = max(0, x), max(0, y)
        face_centers.append((x + fw / 2, y + fh / 2))
        tone = _tone_from_crop(img[y:y + fh, x:x + fw])
        if tone is None:
            continue
        mst, lab = tone
        dets.append({"src": "face", "mst": mst, "lab": lab,
                     "frac": round((fw * fh) / frame_area, 4), "conf": round(float(f[-1]), 3)})

    # --- 2. persons whose face wasn't found (fallback, lower-confidence) ---
    net = _person_net()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (300, 300)),
            0.007843, (300, 300), 127.5)
        net.setInput(blob)
        out = net.forward()
        for i in range(out.shape[2]):
            if int(out[0, 0, i, 1]) != PERSON_CLASS or float(out[0, 0, i, 2]) < PERSON_CONF:
                continue
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * [w, h, w, h]).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            # Skip if a detected face sits inside this person box — the face
            # already represents that person and is the better sample.
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in face_centers):
                continue
            tone = _tone_from_crop(img[y1:y2, x1:x2])
            if tone is None:
                continue
            mst, lab = tone
            dets.append({"src": "body", "mst": mst, "lab": lab,
                         "frac": round(((x2 - x1) * (y2 - y1)) / frame_area, 4),
                         "conf": round(float(out[0, 0, i, 2]), 3)})

    # Summary tone: most-prominent face if any, else most-prominent body.
    faces_only = [d for d in dets if d["src"] == "face"]
    pool = faces_only or dets
    dominant = max(pool, key=lambda d: d["frac"])["mst"] if pool else None
    dets.sort(key=lambda d: (d["src"] != "face", -d["frac"]))  # faces first, then by size
    return {"dets": dets, "mst": dominant}


def scan(paths, progress=None, workers: int = 6) -> dict:
    """Build/refresh the skin-tone index over ``paths`` (incremental; version-gated)."""
    return _SIDE.scan(paths, _detect_and_tone, progress=progress, workers=workers, version=SCAN_VERSION)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def cache_exists() -> bool:
    return _SIDE.exists()


def tone_histogram() -> list[int]:
    """Count of images whose summary tone falls in each MST bin (1..10)."""
    hist = [0] * 10
    for _key, e in _SIDE.entries().items():
        m = e.get("mst")
        if m and 1 <= m <= 10:
            hist[m - 1] += 1
    return hist


def search(tone: int, k: int = 24, folder: str | None = None) -> list[tuple[str, float]]:
    """Rank indexed images by closeness to MST ``tone`` (1..10).

    Score per detection = closeness × prominence × source-weight (faces full,
    bodies ``BODY_WEIGHT``); an image takes its best detection.
    """
    pre = os.path.join(folder, "") if folder else None
    scored: list[tuple[str, float]] = []
    for (p, _m, _s), e in _SIDE.entries().items():
        if pre and not (p == folder or p.startswith(pre)):
            continue
        best = 0.0
        for d in (e.get("dets") or []):
            close = 1.0 - abs(d["mst"] - tone) / 9.0
            w = 1.0 if d.get("src") == "face" else BODY_WEIGHT
            best = max(best, close * (0.5 + 0.5 * d["frac"]) * w)
        if best > 0:
            scored.append((p, best))
    scored.sort(key=lambda x: -x[1])
    return scored[:k]
