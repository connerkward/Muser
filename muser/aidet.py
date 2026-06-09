"""AI-likelihood facet — a forensic pixel-level "how likely is this image
AI-generated?" score, complementing the metadata-only C2PA verdict.

Backend: **GRIP `Grag2021_latent`** (Corvi, Cozzolino, Verdoliva — GRIP-UNINA,
ICASSP 2023, Apache-2.0; ResNet-50 stride-1 fully-convolutional, vendored in
`_grip_resnet.py`). It won an internal bake-off vs UniversalFakeDetect (CLIP
linear probe) and two HF ViT classifiers, with mean AUC 0.990 across SDXL /
Midjourney / Flux / ChatGPT-image / nano-banana and ~97% detection at 5% FPR
once the operating point is calibrated (see reports/ai-detector-benchmark).

Per image: full-res forward (ImageNet norm, **no resize** beyond a 1024px
long-side cap — downscaling destroys the high-frequency forensic traces) → a
mean-pooled logit → a **Platt-calibrated** AI-likelihood percentage in [0,100]
(`pct = 100·σ(A·logit + B)`, A/B fit on the labeled benchmark set). Unlike
C2PA this is a *soft, always-present* score (every image gets one), not a hard
signed badge — so it's surfaced as a percentage, never as a binary "AI" claim.

Built on `facets.Sidecar` (`~/.muser/aidet.json`, incremental by mtime+size).
The 269 MB weight lives at `~/.muser/models/grip_latent.pth` (not git-tracked).
Degrades to `available() == False` when torch or the weight is absent.
"""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .facets import Sidecar

_SIDE = Sidecar("aidet")
MODEL_PATH = Path.home() / ".muser" / "models" / "grip_latent.pth"
VERSION = 1  # bump to force a full re-scan after an algorithm/calibration change

# Platt scaling logit -> percentage, fit on the SDXL/MJ/Flux/ChatGPT/nano vs
# Flickr-real benchmark (scripts in reports/ai-detector-benchmark). Boundary
# (50%) sits at logit = -B/A; reals fall well below it, generators above.
PLATT_A = 0.561700
PLATT_B = 5.307426

_model = None
_load_lock = threading.Lock()
_fwd_lock = threading.Lock()  # serialize the GPU/MPS forward (not thread-parallel)


def available() -> bool:
    """True when the GRIP weight and torch are both present."""
    if not MODEL_PATH.exists():
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _load_model():
    global _model
    if _model is not None:
        return _model
    with _load_lock:
        if _model is not None:
            return _model
        import importlib.util as u
        import torch
        import torchvision.transforms as T

        here = Path(__file__).parent / "_grip_resnet.py"
        spec = u.spec_from_file_location("_grip_resnet", here)
        gm = u.module_from_spec(spec)
        spec.loader.exec_module(gm)
        net = gm.resnet50(num_classes=1, gap_size=1, stride0=1)
        dat = torch.load(MODEL_PATH, map_location="cpu")
        sd = dat["model"] if "model" in dat else dat
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(dev).eval()
        norm = T.Compose([T.ToTensor(),
                          T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        _model = (net, norm, dev, torch)
    return _model


def score_logit(path: str) -> float | None:
    """Forensic logit for one image (higher → more synthetic). None on failure."""
    try:
        net, norm, dev, torch = _load_model()
        from PIL import Image
        im = Image.open(path).convert("RGB")
        if max(im.size) > 1024:  # bound memory; still no square/aspect resize
            s = 1024 / max(im.size)
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.BICUBIC)
        x = norm(im).unsqueeze(0).to(dev)
        with _fwd_lock, torch.no_grad():
            out = net(x).cpu().numpy()[:, 0]
        return float(np.mean(out))
    except Exception:
        return None


def pct_from_logit(logit: float) -> int:
    """Platt-calibrated AI-likelihood percentage in [0,100]."""
    p = 100.0 / (1.0 + np.exp(-(PLATT_A * logit + PLATT_B)))
    return int(round(min(100.0, max(0.0, p))))


def _compute(path: str) -> dict:
    lg = score_logit(path)
    if lg is None:
        return {}
    return {"logit": round(lg, 4), "pct": pct_from_logit(lg)}


def scan(paths, progress=None, workers: int = 4) -> dict:
    """Incrementally score `paths`. workers>1 overlaps image decode; the GPU
    forward itself is serialized by `_fwd_lock` (MPS isn't thread-parallel)."""
    return _SIDE.scan(paths, _compute, progress=progress, workers=workers, version=VERSION,
                      checkpoint_every=400,  # ~every 1–2 min: scored images go live mid-scan
                      content_addressed=True)  # reuse by blake2b: moved/re-imported/dup files skip the forward


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    """RAM-only {logit, pct} for a path if the sidecar knows it (mtime+size match)."""
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


def sidecar() -> Sidecar:
    return _SIDE
