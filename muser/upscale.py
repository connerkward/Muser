"""4× UltraSharp upscaler for cart items.

Used by the cart-zip endpoint: items flagged ``upscale=true`` are upscaled with
the **4x-UltraSharp** ESRGAN variant (uwg/upscaler HF mirror) and bundled into
the zip as JPEG quality-92 under the original filename. Original files stay
untouched on disk — the upscale only materializes at zip time.

Why UltraSharp only:
    Bake-off on 2026-06-04 over four representative library images (photo,
    AI/ComfyUI, screenshot, stylized) showed UltraSharp wins on 3/4. The user's
    library is mostly AI / stylized content where Real-ESRGAN's denoising
    flattens painterly texture; UltraSharp preserves it. See
    ~/Desktop/2026-06-04-upscale-bake/compare.txt for the eval.

Loading: `spandrel` 0.4.2 (`ModelLoader().load_from_file(...)` returns an
`ImageModelDescriptor` that wraps the RRDBNet architecture without basicsr's
Apple-Silicon-hostile install). Inference: fp32, MPS on Apple Silicon, CUDA on
NVIDIA, CPU fallback. Tiled at 256² with a 16-px overlap pad — same settings as
the eval that picked this backbone.

Caching: every successful upscale lands at
``~/.muser/upscale_cache/<uid>.jpg``, keyed on the file's blake2b uid and the
on-disk mtime. A re-zip of the same cart item returns the cached bytes; an edit
to the source bumps the mtime and recomputes.

Weight download: lazy, on first call. Pulled from the `uwg/upscaler` HF mirror
to ``~/.muser/upscaler/4x-UltraSharp.pth`` (~67 MB). The HF mirror's the same
file the eval used; spandrel auto-detects the architecture.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .index import uid_for

# ----- on-disk locations -----------------------------------------------------
MUSER_DIR = Path.home() / ".muser"
UPSCALE_CACHE = MUSER_DIR / "upscale_cache"
WEIGHTS_DIR = MUSER_DIR / "upscaler"
WEIGHTS_PATH = WEIGHTS_DIR / "4x-UltraSharp.pth"

# Hugging Face mirror for community ESRGAN weights. UltraSharp is the
# Kim2091 community fine-tune of RRDBNet (https://huggingface.co/uwg/upscaler).
_WEIGHT_URL = (
    "https://huggingface.co/uwg/upscaler/resolve/main/"
    "ESRGAN/4x-UltraSharp.pth"
)

# Tiling: same as the bake-off. 256² input → 1024² output per tile on a 4× model.
_TILE = 256
_TILE_PAD = 16
_JPEG_QUALITY = 92

# Module-singleton model — loaded once on first upscale() call, kept resident.
_lock = threading.Lock()
_model = None  # spandrel.ImageModelDescriptor | None
_device = None
_dtype = None


def _ensure_weights() -> Path:
    """Download the UltraSharp weights to ~/.muser/upscaler on first use."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 1_000_000:
        return WEIGHTS_PATH
    # Atomic download: write to .partial then rename. Avoids serving a truncated
    # weights file if the process is killed mid-download.
    import urllib.request

    tmp = WEIGHTS_PATH.with_suffix(".pth.partial")
    print(f"[upscale] downloading 4x-UltraSharp.pth (~67 MB) → {WEIGHTS_PATH}", flush=True)
    t0 = time.time()
    urllib.request.urlretrieve(_WEIGHT_URL, tmp)
    tmp.replace(WEIGHTS_PATH)
    print(f"[upscale] download complete in {time.time()-t0:.1f}s", flush=True)
    return WEIGHTS_PATH


def _pick_device():
    """MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise. fp32 — MPS half-precision
    is buggy with several ESRGAN ops (the bake-off saw stripe artifacts), so we
    pin float32 across all backends for consistent output."""
    import torch

    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float32
    return torch.device("cpu"), torch.float32


def _load_model():
    """Load the model once, holding `_lock` so two concurrent first-callers
    don't both attempt the download/load (would waste GPU memory and corrupt
    the weights file via overlapping writes)."""
    global _model, _device, _dtype
    with _lock:
        if _model is not None:
            return _model
        import torch
        from spandrel import ModelLoader

        _ensure_weights()
        _device, _dtype = _pick_device()
        m = ModelLoader().load_from_file(str(WEIGHTS_PATH))
        m.model.eval().to(_device, _dtype)
        # Tiny self-check so a corrupted weight file fails here, not deep in
        # the inference path.
        with torch.inference_mode():
            probe = torch.zeros((1, 3, 16, 16), device=_device, dtype=_dtype)
            _ = m(probe)
        _model = m
        print(f"[upscale] loaded UltraSharp on {_device} ({_dtype})", flush=True)
        return _model


def _to_tensor(img):
    import numpy as np
    import torch

    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()
    return t.to(_device, _dtype)


def _to_pil(t):
    from PIL import Image

    arr = t.squeeze(0).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def _tiled(model, img_t, tile: int = _TILE, pad: int = _TILE_PAD):
    """Tile the input so peak VRAM is bounded.

    Copied straight from the bake-off (run_compare.py) — same numbers (256 / 16)
    that produced the comparison the user picked from. Pad creates an overlap
    that's cropped out of the per-tile output so seams don't show.
    """
    import torch

    scale = model.scale
    _, _, H, W = img_t.shape
    if H <= tile and W <= tile:
        with torch.inference_mode():
            return model(img_t)
    out = torch.zeros((1, 3, H * scale, W * scale), device=img_t.device, dtype=img_t.dtype)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y0, x0 = y, x
            y1, x1 = min(y + tile, H), min(x + tile, W)
            py0 = max(0, y0 - pad); px0 = max(0, x0 - pad)
            py1 = min(H, y1 + pad); px1 = min(W, x1 + pad)
            patch = img_t[:, :, py0:py1, px0:px1]
            with torch.inference_mode():
                up = model(patch)
            top_pad = (y0 - py0) * scale
            left_pad = (x0 - px0) * scale
            bot_pad = (py1 - y1) * scale
            right_pad = (px1 - x1) * scale
            up_h, up_w = up.shape[2], up.shape[3]
            up_crop = up[
                :, :,
                top_pad : up_h - bot_pad if bot_pad else up_h,
                left_pad : up_w - right_pad if right_pad else up_w,
            ]
            out[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = up_crop
    return out


def upscale_4x(path: str) -> bytes:
    """Upscale `path` 4× with UltraSharp; return JPEG quality-92 bytes.

    Caches per ``(path, mtime)`` at ``~/.muser/upscale_cache/<uid>.jpg``. The
    mtime suffix in the on-disk metadata lets a re-zip of the same item return
    the cached bytes in ~1 ms even with a cold model. Editing the source bumps
    mtime, so the cache is recomputed; deleting ``~/.muser/upscale_cache/``
    invalidates everything.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    UPSCALE_CACHE.mkdir(parents=True, exist_ok=True)
    uid = uid_for(path)
    mt = int(os.path.getmtime(path))
    cached = UPSCALE_CACHE / f"{uid}_{mt}.jpg"
    if cached.exists():
        return cached.read_bytes()

    # PIL bomb-guard is fine here — these are user files we already accepted into
    # the index. Cap input dim at 1536 short-side so a 6k photo doesn't take a
    # minute on MPS; output of a capped input is still substantially upscaled vs
    # the original card thumb.
    from PIL import Image, ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    # No pre-downscale here — the user picked this knowing speed (~10–20s per image)
    # is the trade for honest 4× output. Tiling caps peak memory anyway.

    model = _load_model()
    import torch

    img_t = _to_tensor(img)
    t0 = time.perf_counter()
    out_t = _tiled(model, img_t)
    if _device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    out_pil = _to_pil(out_t)
    print(f"[upscale] {os.path.basename(path)}: {img.size[0]}×{img.size[1]} → "
          f"{out_pil.size[0]}×{out_pil.size[1]} in {elapsed:.1f}s", flush=True)

    # Save to cache atomically, then re-read bytes.
    tmp = cached.with_suffix(".jpg.tmp")
    out_pil.save(tmp, "JPEG", quality=_JPEG_QUALITY, optimize=False)
    tmp.replace(cached)

    # Free GPU memory between calls — a batch of cart items would otherwise keep
    # MPS holding the last tile's allocation forever (no autograd, no obvious release).
    del img_t, out_t, out_pil
    if _device.type == "mps":
        torch.mps.empty_cache()
    elif _device.type == "cuda":
        torch.cuda.empty_cache()

    return cached.read_bytes()
