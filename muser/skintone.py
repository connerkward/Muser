"""Per-image skin-tone index + search on the Monk Skin Tone (MST) scale.

Pipeline, best-source first:

1. **Face** (OpenCV YuNet) localizes faces; within each face we get **exact skin
   pixels** from a **face-parsing segmentation model** (SegFormer, `transformers`)
   that labels skin/nose/ears apart from eyes, brows, lips, hair and background —
   far cleaner than a box + colour heuristic. Falls back to the central-region
   YCrCb heuristic if the parser is unavailable or finds no skin.
2. **Person** (OpenCV MobileNet-SSD) is the fallback for people whose face isn't
   visible (turned away / profile / distant). Skin is sampled from any visible
   skin in the person box (arms/hands/legs), but only for person boxes that don't
   already contain a detected face (no double-counting). Bodies use the colour
   heuristic (no face to parse) and are weighted lower in ranking.

**Illuminant normalization.** Skin is sampled as *captured* colour, which a warm/
cool light tints — pushing e.g. a light-skinned face under tungsten into a darker
bucket. Before mapping we estimate the scene illuminant with **Shades-of-Gray**
(Minkowski p=6) over the whole image and divide it out (chroma only — mean
brightness preserved), so the tone reflects reflectance, not the lamp. (Plain
gray-world on the face crop was tried and *hurt* — a face crop isn't gray; this
estimates from the scene instead.)

Skin pixels → 25–75th luminance trim (drop specular highlights + cast shadows) →
median → CIE-LAB → nearest of the 10 official Monk Skin Tone swatches. Persisted
to ``~/.muser/skintone.json``: per image a list of detections (``src`` "face"/
"body", ``mst`` 1–10, ``lab``, ``frac``, ``conf``) + an ``mst`` summary.

Honest limits: detection is the ceiling; a person with **no visible skin** (fully
clothed / back turned) yields no tone; and illuminant estimation can't undo
exposure (a genuinely under-exposed face stays dark). A positive signal, not a
demographic classifier.
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
_PARSE_MODEL_ID = "jonathandinu/face-parsing"  # SegFormer, 19-class CelebAMask labels
# Face-parsing label ids that are skin surface (exclude eyes/brows/lips/hair/cloth).
_SKIN_LABELS = (1, 2, 8, 9)  # skin, nose, l_ear, r_ear

# Bump when the detection/sampling algorithm changes so the incremental scan
# recomputes already-indexed images instead of skipping them by mtime.
SCAN_VERSION = 4  # v1 face-only; v2 +person; v3 central+trim; v4 face-parse + illuminant-norm

USE_WB = True            # apply Shades-of-Gray illuminant normalization (validated by benchmark)

# Official Monk Skin Tone 10-point scale (Google), sRGB hex, light → dark.
MST_HEX = [
    "#f6ede4", "#f3e7db", "#f7ead0", "#eadaba", "#d7bd96",
    "#a07e56", "#825c43", "#604134", "#3a312a", "#292420",
]

DET_SIDE = 640           # downscale longest side before detection (speed)
DET_CONF = 0.6           # YuNet face score threshold
PERSON_CONF = 0.45       # MobileNet-SSD person score threshold
PERSON_CLASS = 15        # "person" in the VOC label set MobileNet-SSD was trained on
MIN_SKIN_PX = 30         # need at least this many skin pixels to sample a tone
BODY_WEIGHT = 0.6        # body-sampled tones rank below face-sampled ones

_mst_lab = None
_tls = threading.local()           # YuNet / MobileNet detectors are per-thread
_parser = None                     # (model, proc, dev) | False (load failed)
_parser_load_lock = threading.Lock()
_parser_infer_lock = threading.Lock()  # serialize MPS/GPU inference across scan threads


def available() -> bool:
    # Face detection is the floor; parsing + person detection are enhancements.
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return _FACE_MODEL.exists()


def person_available() -> bool:
    return _PERSON_PROTO.exists() and _PERSON_MODEL.exists()


def parse_available() -> bool:
    try:
        import transformers  # noqa: F401
        import torch  # noqa: F401
    except Exception:
        return False
    return _parser is not False


def sidecar() -> Sidecar:
    return _SIDE


# --- Monk-scale helpers ----------------------------------------------------
def _mst_lab_arr():
    global _mst_lab
    if _mst_lab is None:
        import cv2

        rgb = np.array([[int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)] for h in MST_HEX], np.uint8)
        _mst_lab = cv2.cvtColor(rgb.reshape(1, -1, 3), cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    return _mst_lab


def pixels_to_mst(pixels) -> tuple[int, list[int]] | None:
    """Skin pixels (N×3 RGB) → (mst, lab). 25–75th luminance trim drops specular
    highlights + cast shadows before the median."""
    import cv2

    if pixels is None or len(pixels) < MIN_SKIN_PX:
        return None
    px = pixels.astype(np.float32)
    y = px @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(y, [25, 75])
    keep = (y >= lo) & (y <= hi)
    if keep.sum() >= MIN_SKIN_PX:
        px = px[keep]
    med = np.median(px, axis=0).astype(np.uint8)
    lab = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    mst = int(np.argmin(np.linalg.norm(_mst_lab_arr() - lab, axis=1))) + 1
    return mst, [int(lab[0]), int(lab[1]), int(lab[2])]


# --- illuminant normalization (Shades-of-Gray, p=6) ------------------------
def illuminant_gain(img_rgb, p: int = 6) -> np.ndarray:
    """Per-channel gain that divides out the estimated scene illuminant, scaled to
    preserve mean brightness (corrects colour cast, not exposure)."""
    f = img_rgb.reshape(-1, 3).astype(np.float32)
    illum = np.power(np.mean(np.power(f, p), axis=0), 1.0 / p)
    illum = np.maximum(illum, 1e-3)
    gain = illum.mean() / illum
    return gain.astype(np.float32)


def apply_gain(pixels, gain) -> np.ndarray:
    return np.clip(pixels.astype(np.float32) * gain, 0, 255).astype(np.uint8)


# --- detectors -------------------------------------------------------------
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


def _load_parser():
    """Lazy singleton (model, proc, dev) for the face-parsing SegFormer, or None."""
    global _parser
    if _parser is None:
        with _parser_load_lock:
            if _parser is None:
                try:
                    import torch
                    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

                    proc = SegformerImageProcessor.from_pretrained(_PARSE_MODEL_ID)
                    model = AutoModelForSemanticSegmentation.from_pretrained(_PARSE_MODEL_ID)
                    dev = ("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available() else "cpu")
                    model.to(dev).eval()
                    _parser = (model, proc, dev)
                except Exception:
                    _parser = False
    return _parser if _parser else None


# --- skin-pixel extractors -------------------------------------------------
def skin_pixels_parse(crop_rgb):
    """Exact skin pixels (N×3 RGB) from the face-parsing model, or None."""
    p = _load_parser()
    if p is None or crop_rgb.size == 0:
        return None
    import torch
    from PIL import Image as _Image

    model, proc, dev = p
    pil = _Image.fromarray(crop_rgb)
    with _parser_infer_lock:
        inp = proc(images=pil, return_tensors="pt").to(dev)
        with torch.inference_mode():
            logits = model(**inp).logits
        up = torch.nn.functional.interpolate(
            logits, size=crop_rgb.shape[:2], mode="bilinear", align_corners=False)
        seg = up.argmax(1)[0].cpu().numpy()
    mask = np.isin(seg, _SKIN_LABELS)
    pix = crop_rgb[mask]
    return pix if len(pix) >= MIN_SKIN_PX else None


def skin_pixels_heuristic(crop_rgb, central: bool = False):
    """Skin pixels via a widened YCrCb colour mask. ``central`` first restricts to
    the cheek/jaw sub-rectangle (for faces); else samples the whole crop (bodies)."""
    import cv2

    if crop_rgb.size == 0:
        return None
    if central:
        h, w = crop_rgb.shape[:2]
        sub = crop_rgb[int(0.35 * h):int(0.92 * h), int(0.18 * w):int(0.82 * w)]
        if sub.size:
            crop_rgb = sub
    ycc = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    mask = (cr >= 133) & (cr <= 183) & (cb >= 77) & (cb <= 133)
    pix = crop_rgb[mask]
    return pix if len(pix) >= MIN_SKIN_PX else None


def _face_pixels(crop_rgb):
    """Best skin pixels for a face crop: parse if available, else central heuristic."""
    px = skin_pixels_parse(crop_rgb)
    if px is None:
        px = skin_pixels_heuristic(crop_rgb, central=True)
    return px


def _tone(pixels, gain):
    if pixels is None:
        return None
    if gain is not None:
        pixels = apply_gain(pixels, gain)
    return pixels_to_mst(pixels)


def _detect_and_tone(path: str) -> dict:
    import cv2

    img = np.asarray(_load_rgb(path, max_side=DET_SIDE).convert("RGB"))
    h, w = img.shape[:2]
    frame_area = float(w * h) or 1.0
    gain = illuminant_gain(img) if USE_WB else None
    dets = []
    face_centers = []

    # --- 1. faces → parsed skin (preferred) ---
    fd = _face_detector()
    fd.setInputSize((w, h))
    _, faces = fd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    for f in (faces if faces is not None else []):
        x, y, fw, fh = (int(v) for v in f[:4])
        x, y = max(0, x), max(0, y)
        face_centers.append((x + fw / 2, y + fh / 2))
        # expand box ~20% so the parser has a little context
        mx, my = int(fw * 0.2), int(fh * 0.2)
        cx0, cy0 = max(0, x - mx), max(0, y - my)
        cx1, cy1 = min(w, x + fw + mx), min(h, y + fh + my)
        tone = _tone(_face_pixels(img[cy0:cy1, cx0:cx1]), gain)
        if tone is None:
            continue
        dets.append({"src": "face", "mst": tone[0], "lab": tone[1],
                     "frac": round((fw * fh) / frame_area, 4), "conf": round(float(f[-1]), 3)})

    # --- 2. face-less persons → colour-heuristic skin (fallback) ---
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
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for cx, cy in face_centers):
                continue
            tone = _tone(skin_pixels_heuristic(img[y1:y2, x1:x2], central=False), gain)
            if tone is None:
                continue
            dets.append({"src": "body", "mst": tone[0], "lab": tone[1],
                         "frac": round(((x2 - x1) * (y2 - y1)) / frame_area, 4),
                         "conf": round(float(out[0, 0, i, 2]), 3)})

    faces_only = [d for d in dets if d["src"] == "face"]
    pool = faces_only or dets
    dominant = max(pool, key=lambda d: d["frac"])["mst"] if pool else None
    dets.sort(key=lambda d: (d["src"] != "face", -d["frac"]))
    return {"dets": dets, "mst": dominant}


def scan(paths, progress=None, workers: int = 6) -> dict:
    """Build/refresh the skin-tone index (incremental; version-gated). Parsing
    serializes on the GPU via a lock, so threads mainly overlap image I/O."""
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
    """Rank indexed images by closeness to MST ``tone`` (1..10), prominence- and
    source-weighted (faces full, bodies ``BODY_WEIGHT``)."""
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
