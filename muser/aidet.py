"""AI-likelihood facet — a soft "how likely is this image AI-generated?" score.

Backend: **Community Forensics** (Park & Owens, CVPR 2025 — `OwensLab/commfor-model-384`,
MIT, a ViT-S/16-384 trained on 4,803 generators). It REPLACED the earlier GRIP
`Grag2021_latent` forensic detector, which over-flagged badly on a diverse library: GRIP
is really an "is this a clean *camera photograph*?" detector, so it called digital art,
3D renders, screenshots, vintage/scanned photos, and recompressed JPEGs "AI" (~100% false
positives on this corpus). On the same images Community Forensics drops that to ~5%, keeps
real photos at ~0%, and still catches SDXL/ComfyUI output ~100% — weaker on Flux/ChatGPT/
Gemini (~12–46%), which is why it stays a *soft hint* paired with the signed C2PA badge,
never an authoritative "AI" claim. See reports/ai-detector-benchmark + the A/B that drove
the swap.

The model's sigmoid output IS the calibrated P(AI), so the percentage is just
`round(100·σ(logit))` — no Platt step. Preprocess (matches the authors' test transform):
Resize shorter side→440, center-crop 384, ImageNet norm. ViT-S = 22M params, runs on
MPS/CPU; weights (~88 MB) download once from HF and cache. Degrades to `available()==False`
when torch/timm is absent.

Built on `facets.Sidecar` (`~/.muser/aidet.json`, incremental + content-addressed).
"""
from __future__ import annotations

import threading

import numpy as np

from .facets import Sidecar

_SIDE = Sidecar("aidet")
HF_REPO = "OwensLab/commfor-model-384"
VERSION = 2  # bumped when GRIP → Community Forensics; forces a clean re-scan

_model = None
_load_lock = threading.Lock()
_fwd_lock = threading.Lock()  # serialize the GPU/MPS forward (not thread-parallel)


def available() -> bool:
    """True when torch + timm are importable (the weights download lazily)."""
    try:
        import timm  # noqa: F401
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
        import timm
        import torch
        import torchvision.transforms as T
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        net = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
        net.head = torch.nn.Linear(384, 1)  # binary head, matches the published ViTClassifier
        sd = load_file(hf_hub_download(HF_REPO, "model.safetensors"))
        sd = {k[4:] if k.startswith("vit.") else k: v for k, v in sd.items()}  # strip 'vit.'
        net.load_state_dict(sd)
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(dev).eval()
        # Authors' test transform: Resize(440) → CenterCrop(384) → ToTensor → ImageNet norm.
        prep = T.Compose([
            T.Resize(440), T.CenterCrop(384), T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _model = (net, prep, dev, torch)
    return _model


def score_logit(path: str) -> float | None:
    """Raw logit for one image (higher → more likely AI). None on failure."""
    try:
        net, prep, dev, torch = _load_model()
        from PIL import Image
        x = prep(Image.open(path).convert("RGB")).unsqueeze(0).to(dev)
        with _fwd_lock, torch.no_grad():
            return float(net(x).flatten()[0].cpu())
    except Exception:
        return None


def pct_from_logit(logit: float) -> int:
    """Calibrated AI-likelihood percentage — the model's own σ(logit), ×100."""
    p = 100.0 / (1.0 + np.exp(-logit))
    return int(round(min(100.0, max(0.0, p))))


def _compute(path: str) -> dict:
    lg = score_logit(path)
    if lg is None:
        return {}
    return {"logit": round(lg, 4), "pct": pct_from_logit(lg)}


def scan(paths, progress=None, workers: int = 4) -> dict:
    """Incrementally score `paths`. workers>1 overlaps image decode; the GPU forward
    is serialized by `_fwd_lock`. Content-addressed so moved/re-imported/duplicate
    files reuse a prior score instead of re-running the model."""
    return _SIDE.scan(paths, _compute, progress=progress, workers=workers, version=VERSION,
                      checkpoint_every=400, content_addressed=True)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    """RAM-only {logit, pct} for a path if the sidecar knows it (mtime+size match)."""
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


def sidecar() -> Sidecar:
    return _SIDE
