"""Per-image scores — "Interesting" and "Review" (sensitivity triage).

Batch artifact (`muser score`): reuse the indexed siglip2-b embeddings (no new model
pass) to compute, for every image:

  - novelty      — how isolated the image is in embedding space (mean similarity to
                   its nearest neighbors, inverted). High = unusual / one-of-a-kind.
  - aesthetic    — a zero-shot "striking photo" vs "dull snapshot" signal — weak,
                   free proxy; kept for comparison against the real models below.
  - aesthetic_v2 — LAION-Aesthetics V2 (Schuhmann et al.): a tiny MLP
                   (768→1024→128→64→16→1) trained on SAC+LOGOS+AVA1 ratings, on top
                   of OpenAI CLIP-ViT-L/14 image features. Raw output ≈ 1 (ugly) to
                   ~7.5 (beautiful); we percentile-normalize. Weights:
                   `camenduru/improved-aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth`
                   (~3.7 MB). Runs CLIP-L/14 if the active model isn't already it.
  - pickscore    — PickScore v1 (Kirstain et al., NeurIPS 2023): a CLIP-H/14 model
                   fine-tuned on human pairwise preferences from Pick-a-Pic. Per
                   image we compute the calibrated cosine `logit_scale·cos(img, txt)`
                   against the neutral prompt "a high-quality image" — an
                   "unconditional aesthetic" reading. Weights: `yuvalkirstain/PickScore_v1`
                   (~700 MB). Slow first pass: ~50 ms/image on MPS (CLIP-H is large).
  - aesthetic_v25 — Aesthetic Predictor V2.5 (Discus0434, 2024): direct upgrade to
                   LAION-V2 — same MLP-head idea but with a 5-layer scoring head on
                   `google/siglip-so400m-patch14-384` features (much stronger backbone
                   than CLIP-L/14). Head checkpoint:
                   `discus0434/aesthetic-predictor-v2-5/aesthetic_predictor_v2_5.pth`
                   (~2.5 MB); backbone is ~3.5 GB and shared with no one else here.
                   Raw output ≈ 1.0 (ugly) to ~9.0 (beautiful) per the upstream readme;
                   we percentile-normalize.
  - hps_v21      — HPS-v2.1 (Wu et al., updated 2024): direct upgrade to PickScore —
                   open_clip ViT-H-14 fine-tuned on the HPDv2 dataset (~798k human
                   preference pairs, larger and more diverse than Pick-a-Pic). We
                   score each image against the empty/unconditional prompt and take
                   `logit_scale·cos(img, txt)` — the same calibrated cosine the
                   official `hpsv2.score` API returns. Weights:
                   `xswu/HPSv2/HPS_v2.1_compressed.pt` (~3.7 GB) + open_clip's
                   `laion2b_s32b_b79k` ViT-H-14 base (~3.9 GB). Loaded via open_clip
                   directly — the `hpsv2` PyPI package was rejected because its
                   install_requires pin `protobuf<4` conflicts with our
                   `transformers>=4.52`, and it pulls in pytest/webdataset/clint as
                   runtime deps.
  - interesting  — blend of novelty + aesthetic.
  - nsfw / private / political — zero-shot similarity to sensitivity concept sets.
                   These are HEURISTIC FLAGS FOR HUMAN REVIEW of your own library
                   (find candidates before sharing), not classifiers / verdicts.

Each metric is percentile-normalized to [0,1] so they're comparable and sortable.
Writes ~/.muser/scores.json; the Explore "Interesting" and "Review" tabs read it.

The heavy new metrics (`aesthetic_v2`, `pickscore`, `aesthetic_v25`, `hps_v21`) are
computed only on the canonical set (deduped representatives), then broadcast to each
dup group's members — near-identical copies share a score by definition. Per-path
results are cached to `~/.muser/{aesthetic_v2,pickscore,aesthetic_v25,hps_v21}_cache.json`
keyed by `path|mtime|size` so incremental re-runs only score new/changed files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .index import MuserIndex
from .registry import load_model

SCORES_JSON = Path.home() / ".muser" / "scores.json"

AESTHETIC_POS = [
    "a beautiful, striking, high-quality photograph", "stunning professional photography",
    "an aesthetically pleasing, well-composed image", "award-winning photography, dramatic lighting",
]
AESTHETIC_NEG = [
    "a boring, low-quality snapshot", "a blurry, badly-lit photo",
    "a mundane screenshot", "an ugly, cluttered image",
]
# Sensitivity concept sets (zero-shot triage — flag for human review).
RISK = {
    "nsfw": [
        "explicit nude content", "a naked person", "sexual or pornographic content",
        "a person in underwear or lingerie", "explicit adult content",
    ],
    "private": [
        "a screenshot of private text messages", "a login screen with a password",
        "a credit card or bank statement", "a passport or ID document",
        "personal financial information on a screen", "a private email inbox",
    ],
}


# Benchmark-tuned weight (eval/nsfw_bench.py): nsfw = NSFW_W·Falconsai + (1-NSFW_W)·zero-shot.
# 0.9 maximized AUC (0.913) vs Falconsai alone (0.873) and zero-shot alone (0.797).
NSFW_W = 0.9

_FALCON = {"pipe": None}


def _falconsai_nsfw(paths, on_progress, batch_size: int = 24) -> np.ndarray:
    """Real ViT NSFW probability per image (Falconsai/nsfw_image_detection)."""
    from transformers import pipeline

    from .embedders import _device, _load_rgb

    if _FALCON["pipe"] is None:
        _FALCON["pipe"] = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=_device())
    pipe = _FALCON["pipe"]
    out = np.zeros(len(paths), dtype=np.float32)
    for i in range(0, len(paths), batch_size):
        chunk = paths[i : i + batch_size]
        try:
            res = pipe([_load_rgb(p) for p in chunk], batch_size=batch_size)
            for j, r in enumerate(res):
                out[i + j] = next((x["score"] for x in r if x["label"].lower() == "nsfw"), 0.0)
        except Exception:
            pass  # leave unreadable images at 0
        if i % (batch_size * 20) == 0:
            on_progress(f"  Falconsai {i}/{len(paths)}")
    return out


def _pct(x: np.ndarray) -> np.ndarray:
    """Rank → percentile in [0,1] (robust, comparable across metrics)."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x))
    return ranks / max(len(x) - 1, 1)


def _cache_key(p: str) -> str | None:
    """Stable per-file cache key: `<path>|<mtime>|<size>`. None if file missing."""
    try:
        st = Path(p).stat()
    except OSError:
        return None
    return f"{p}|{int(st.st_mtime)}|{st.st_size}"


def _read_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


# ---- LAION-Aesthetics V2 ----------------------------------------------------
# Tiny MLP (768→1024→128→64→16→1) trained on SAC+LOGOS+AVA1 ratings, on top of
# OpenAI CLIP-ViT-L/14 L2-normalized image features. ~3.7 MB checkpoint.
AES_V2_REPO = "camenduru/improved-aesthetic-predictor"
AES_V2_FILE = "sac+logos+ava1-l14-linearMSE.pth"
AES_V2_CACHE = Path.home() / ".muser" / "aesthetic_v2_cache.json"


def _laion_aes_mlp(state_path: str):
    """Build the LAION-Aes V2 MLP and load the checkpoint."""
    import torch
    from torch import nn

    mlp = nn.Sequential(
        nn.Linear(768, 1024), nn.Dropout(0.2),
        nn.Linear(1024, 128), nn.Dropout(0.2),
        nn.Linear(128, 64),   nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )
    # The published checkpoint nests the layers under "layers." so reshape it:
    sd = torch.load(state_path, map_location="cpu", weights_only=True)
    # Remap "layers.N.{weight,bias}" → "N.{weight,bias}" (nn.Sequential names by index).
    remap = {k.replace("layers.", "", 1): v for k, v in sd.items()}
    mlp.load_state_dict(remap)
    mlp.eval()
    return mlp


def _aesthetic_v2(paths: list[str], active_model: str, X: np.ndarray, on_progress) -> np.ndarray:
    """LAION-Aesthetics V2 score per path.

    If the active model is `clip-l14`, reuse its 768-d L2-normalized vectors (X is
    aligned to paths). Otherwise, run CLIP-L/14 over `paths` (cache by mtime+size).
    Returns float32 array aligned to `paths`.
    """
    from huggingface_hub import hf_hub_download

    on_progress("aesthetic_v2: downloading LAION-Aes V2 MLP (~3.7 MB)…")
    weights = hf_hub_download(repo_id=AES_V2_REPO, filename=AES_V2_FILE)
    mlp = _laion_aes_mlp(weights)

    cache = _read_cache(AES_V2_CACHE)
    out = np.zeros(len(paths), dtype=np.float32)

    if active_model == "clip-l14" and X.shape[1] == 768:
        on_progress("aesthetic_v2: scoring from indexed CLIP-L/14 vectors…")
        import torch
        with torch.no_grad():
            scores = mlp(torch.from_numpy(X)).squeeze(-1).numpy()
        for i, p in enumerate(paths):
            out[i] = float(scores[i])
            key = _cache_key(p)
            if key:
                cache[key] = float(scores[i])
        _write_cache(AES_V2_CACHE, cache)
        return out

    # Otherwise embed via CLIP-L/14 ourselves, batched, with mtime+size cache.
    needed_idx = []
    needed_paths = []
    for i, p in enumerate(paths):
        key = _cache_key(p)
        if key and key in cache:
            out[i] = cache[key]
        else:
            needed_idx.append(i)
            needed_paths.append(p)
    if not needed_paths:
        on_progress(f"aesthetic_v2: all {len(paths)} cached, skipping CLIP-L/14 pass")
        return out

    on_progress(f"aesthetic_v2: loading CLIP-L/14 for {len(needed_paths)} new/changed images…")
    clip_l14 = load_model("clip-l14")
    import torch
    bs = 32
    t0 = _now()
    for k in range(0, len(needed_paths), bs):
        chunk = needed_paths[k : k + bs]
        try:
            vecs = clip_l14.embed_images(chunk, batch_size=len(chunk))  # already L2-normalized
            with torch.no_grad():
                s = mlp(torch.from_numpy(np.asarray(vecs, dtype=np.float32))).squeeze(-1).numpy()
            for j, p in enumerate(chunk):
                v = float(s[j])
                out[needed_idx[k + j]] = v
                key = _cache_key(p)
                if key:
                    cache[key] = v
        except Exception as e:
            on_progress(f"  skip batch {k} ({e})")
        if (k // bs) % 10 == 0:
            done = min(k + bs, len(needed_paths))
            eta = (_now() - t0) / max(done, 1) * (len(needed_paths) - done)
            on_progress(f"  aesthetic_v2 {done}/{len(needed_paths)} (~{int(eta)}s left)")
    _write_cache(AES_V2_CACHE, cache)
    return out


# ---- PickScore (Kirstain et al., NeurIPS 2023) -------------------------------
# CLIP-ViT-H/14 fine-tuned on Pick-a-Pic human pairwise preferences. We score
# each image against a neutral prompt for an "unconditional aesthetic" reading.
PICKSCORE_MODEL = "yuvalkirstain/PickScore_v1"
PICKSCORE_PROCESSOR = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
PICKSCORE_PROMPT = "a high-quality image"
PICKSCORE_CACHE = Path.home() / ".muser" / "pickscore_cache.json"


def _now() -> float:
    import time
    return time.time()


def _pickscore(paths: list[str], on_progress, batch_size: int = 16) -> np.ndarray:
    """PickScore per path. Calibrated cosine against a fixed neutral prompt.

    Score = logit_scale · cos(image_emb, text_emb), with both L2-normalized — the
    same per-image scalar PickScore returns before its softmax. Higher = more
    aligned with what humans called "high quality." Cached by path+mtime+size.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    from .embedders import _device, _load_rgb

    cache = _read_cache(PICKSCORE_CACHE)
    out = np.zeros(len(paths), dtype=np.float32)

    needed_idx, needed_paths = [], []
    for i, p in enumerate(paths):
        key = _cache_key(p)
        if key and key in cache:
            out[i] = cache[key]
        else:
            needed_idx.append(i)
            needed_paths.append(p)
    if not needed_paths:
        on_progress(f"pickscore: all {len(paths)} cached, skipping model load")
        return out

    on_progress(f"pickscore: loading PickScore v1 (~700 MB, one-time download)…")
    dev = _device()
    processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR, use_fast=True)
    model = AutoModel.from_pretrained(PICKSCORE_MODEL).eval().to(dev)
    on_progress(f"pickscore: scoring {len(needed_paths)} images on {dev}…")

    # Compute text embedding once.
    with torch.no_grad():
        text_inputs = processor(text=[PICKSCORE_PROMPT], padding=True, truncation=True,
                                max_length=77, return_tensors="pt").to(dev)
        text_emb = model.get_text_features(**text_inputs)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = float(model.logit_scale.exp())

    t0 = _now()
    n = len(needed_paths)
    for k in range(0, n, batch_size):
        chunk = needed_paths[k : k + batch_size]
        try:
            imgs = [_load_rgb(p) for p in chunk]
            with torch.no_grad():
                img_inputs = processor(images=imgs, return_tensors="pt").to(dev)
                img_emb = model.get_image_features(**img_inputs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                scores = (logit_scale * (img_emb @ text_emb.T)).squeeze(-1).float().cpu().numpy()
            for j, p in enumerate(chunk):
                v = float(scores[j])
                out[needed_idx[k + j]] = v
                key = _cache_key(p)
                if key:
                    cache[key] = v
        except Exception as e:
            on_progress(f"  skip batch {k} ({type(e).__name__}: {e})")
        if (k // batch_size) % 5 == 0:
            done = min(k + batch_size, n)
            elapsed = _now() - t0
            eta = elapsed / max(done, 1) * (n - done)
            bar_w = 20
            filled = int(bar_w * done / max(n, 1))
            bar = "#" * filled + "-" * (bar_w - filled)
            on_progress(f"  pickscore [{bar}] {done}/{n} ({int(elapsed)}s elapsed, ~{int(eta)}s left)")
    _write_cache(PICKSCORE_CACHE, cache)
    return out


# ---- Aesthetic Predictor V2.5 (Discus0434, 2024) ----------------------------
# Direct upgrade to LAION-V2: 5-layer MLP scoring head on top of SigLIP-SO400M
# image features. The upstream `aesthetic-predictor-v2-5` PyPI package is a thin
# ~5 KB wrapper that does exactly this; we inline the equivalent to avoid adding
# a third-party dep that only saves a few lines of code.
AES_V25_HEAD_URL = (
    "https://github.com/discus0434/aesthetic-predictor-v2-5/raw/main/models/"
    "aesthetic_predictor_v2_5.pth"
)
AES_V25_BACKBONE = "google/siglip-so400m-patch14-384"
AES_V25_CACHE = Path.home() / ".muser" / "aesthetic_v25_cache.json"


def _aes_v25_head(hidden_size: int):
    """Build the V2.5 scoring head (matches upstream `AestheticPredictorV2_5Head`)."""
    from torch import nn

    return nn.Sequential(
        nn.Linear(hidden_size, 1024), nn.Dropout(0.5),
        nn.Linear(1024, 128),         nn.Dropout(0.5),
        nn.Linear(128, 64),           nn.Dropout(0.5),
        nn.Linear(64, 16),            nn.Dropout(0.2),
        nn.Linear(16, 1),
    )


def _aesthetic_v25(paths: list[str], on_progress, batch_size: int = 24) -> np.ndarray:
    """Aesthetic Predictor V2.5 per path. Backbone = SigLIP-SO400M-384.

    For each image: pool the SigLIP vision tower output, L2-normalize, run the
    5-layer MLP head, take the scalar logit. Cached by path+mtime+size.
    """
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel

    from .embedders import _device, _load_rgb

    cache = _read_cache(AES_V25_CACHE)
    out = np.zeros(len(paths), dtype=np.float32)

    needed_idx, needed_paths = [], []
    for i, p in enumerate(paths):
        key = _cache_key(p)
        if key and key in cache:
            out[i] = cache[key]
        else:
            needed_idx.append(i)
            needed_paths.append(p)
    if not needed_paths:
        on_progress(f"aesthetic_v25: all {len(paths)} cached, skipping model load")
        return out

    on_progress(f"aesthetic_v25: loading SigLIP-SO400M-384 backbone (~3.5 GB, one-time)…")
    dev = _device()
    backbone = SiglipVisionModel.from_pretrained(AES_V25_BACKBONE).eval().to(dev)
    processor = SiglipImageProcessor.from_pretrained(AES_V25_BACKBONE)
    on_progress("aesthetic_v25: downloading V2.5 MLP head (~2.5 MB)…")
    head_sd = torch.hub.load_state_dict_from_url(AES_V25_HEAD_URL, map_location="cpu")
    # Upstream checkpoint nests under "scoring_head."; strip so it loads into a
    # bare nn.Sequential (whose params are named by index, e.g. "0.weight").
    head_sd = {k.replace("scoring_head.", "", 1): v for k, v in head_sd.items()}
    head = _aes_v25_head(backbone.config.hidden_size)
    head.load_state_dict(head_sd)
    head = head.eval().to(dev)
    on_progress(f"aesthetic_v25: scoring {len(needed_paths)} images on {dev}…")

    t0 = _now()
    n = len(needed_paths)
    for k in range(0, n, batch_size):
        chunk = needed_paths[k : k + batch_size]
        try:
            imgs = [_load_rgb(p) for p in chunk]
            with torch.no_grad():
                inputs = processor(images=imgs, return_tensors="pt").to(dev)
                feats = backbone(**inputs).pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
                scores = head(feats).squeeze(-1).float().cpu().numpy()
            for j, p in enumerate(chunk):
                v = float(scores[j])
                out[needed_idx[k + j]] = v
                key = _cache_key(p)
                if key:
                    cache[key] = v
        except Exception as e:
            on_progress(f"  skip batch {k} ({type(e).__name__}: {e})")
        if (k // batch_size) % 5 == 0:
            done = min(k + batch_size, n)
            elapsed = _now() - t0
            eta = elapsed / max(done, 1) * (n - done)
            bar_w = 20
            filled = int(bar_w * done / max(n, 1))
            bar = "#" * filled + "-" * (bar_w - filled)
            on_progress(f"  aesthetic_v25 [{bar}] {done}/{n} ({int(elapsed)}s elapsed, ~{int(eta)}s left)")
    _write_cache(AES_V25_CACHE, cache)
    return out


# ---- HPS-v2.1 (Wu et al., updated 2024) -------------------------------------
# Direct upgrade to PickScore: open_clip ViT-H-14 fine-tuned on the HPDv2
# dataset (~798k human preference pairs). We score each image against the
# empty/unconditional prompt — the same `logit_scale·cos(img, txt)` the official
# `hpsv2.score` API returns before its softmax.
HPS_V21_REPO = "xswu/HPSv2"
HPS_V21_FILE = "HPS_v2.1_compressed.pt"
HPS_V21_BASE = ("ViT-H-14", "laion2b_s32b_b79k")
HPS_V21_PROMPT = ""  # unconditional aesthetic reading
HPS_V21_CACHE = Path.home() / ".muser" / "hps_v21_cache.json"


def _hps_v21(paths: list[str], on_progress, batch_size: int = 8) -> np.ndarray:
    """HPS-v2.1 per path. open_clip ViT-H-14 + fine-tuned weights, unconditional prompt.

    Score = logit_scale · cos(image_emb, text_emb), with both L2-normalized — the
    same per-image scalar the official `hpsv2.score(img, prompt='', hps_version='v2.1')`
    returns. Cached by path+mtime+size.
    """
    import open_clip
    import torch
    from huggingface_hub import hf_hub_download

    from .embedders import _device, _load_rgb

    cache = _read_cache(HPS_V21_CACHE)
    out = np.zeros(len(paths), dtype=np.float32)

    needed_idx, needed_paths = [], []
    for i, p in enumerate(paths):
        key = _cache_key(p)
        if key and key in cache:
            out[i] = cache[key]
        else:
            needed_idx.append(i)
            needed_paths.append(p)
    if not needed_paths:
        on_progress(f"hps_v21: all {len(paths)} cached, skipping model load")
        return out

    on_progress("hps_v21: loading open_clip ViT-H-14 base (~3.9 GB, one-time)…")
    dev = _device()
    arch, pretrained = HPS_V21_BASE
    model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(arch)
    on_progress("hps_v21: downloading HPS-v2.1 weights (~3.7 GB)…")
    cp = hf_hub_download(HPS_V21_REPO, HPS_V21_FILE)
    sd = torch.load(cp, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        on_progress(f"  hps_v21 state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
    model = model.eval().to(dev)

    # Compute text embedding once (empty prompt = unconditional).
    with torch.no_grad():
        text = tokenizer([HPS_V21_PROMPT]).to(dev)
        text_emb = model.encode_text(text)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = float(model.logit_scale.exp())

    on_progress(f"hps_v21: scoring {len(needed_paths)} images on {dev}…")
    t0 = _now()
    n = len(needed_paths)
    for k in range(0, n, batch_size):
        chunk = needed_paths[k : k + batch_size]
        try:
            with torch.no_grad():
                imgs = torch.stack([preprocess(_load_rgb(p)) for p in chunk]).to(dev)
                img_emb = model.encode_image(imgs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                scores = (logit_scale * (img_emb @ text_emb.T)).squeeze(-1).float().cpu().numpy()
            for j, p in enumerate(chunk):
                v = float(scores[j])
                out[needed_idx[k + j]] = v
                key = _cache_key(p)
                if key:
                    cache[key] = v
        except Exception as e:
            on_progress(f"  skip batch {k} ({type(e).__name__}: {e})")
        if (k // batch_size) % 5 == 0:
            done = min(k + batch_size, n)
            elapsed = _now() - t0
            eta = elapsed / max(done, 1) * (n - done)
            bar_w = 20
            filled = int(bar_w * done / max(n, 1))
            bar = "#" * filled + "-" * (bar_w - filled)
            on_progress(f"  hps_v21 [{bar}] {done}/{n} ({int(elapsed)}s elapsed, ~{int(eta)}s left)")
    _write_cache(HPS_V21_CACHE, cache)
    return out


def score_all(model: str = "siglip2-b", on_progress=print) -> dict:
    """Score every image and write ~/.muser/scores.json.

    Canonical paths are processed in lexicographic order so partial-coverage runs
    are deterministic and cross-metric comparable — a subset run scoring the
    "first 200" always picks the same 200, and per-metric caches accumulate
    against the same path prefix across runs.
    """
    idx = MuserIndex()
    t = idx._open(model)
    if t is None:
        raise RuntimeError(f"nothing indexed for {model}")
    rows = t.search().select(["path", "vector"]).limit(100_000_000).to_list()
    # Deterministic order: LanceDB row order isn't guaranteed across runs/iterations.
    # Sorting by path here makes every downstream slice (including the canonical
    # subset and any partial-coverage scoring pass) reproducible.
    rows.sort(key=lambda r: r["path"])
    paths = [r["path"] for r in rows]
    X = np.asarray([r["vector"] for r in rows], dtype=np.float32)  # already L2-normalized
    n = len(paths)
    on_progress(f"loaded {n} vectors (dim {X.shape[1]}, sorted by path)")

    emb = load_model(model)

    def concept(prompts):
        return emb.embed_queries(prompts)  # (P, D) normalized

    # ---- novelty: 1 - mean cosine to nearest neighbors (approx via pynndescent) ----
    on_progress("novelty (approx kNN)…")
    from pynndescent import NNDescent

    ann = NNDescent(X, metric="cosine", n_neighbors=12, random_state=42)
    nbr, dist = ann.neighbor_graph  # neighbor indices + cosine distances incl. self at col 0
    novelty = dist[:, 1:].mean(axis=1)  # higher distance = more isolated = more novel

    # ---- aesthetic: striking vs dull ----
    on_progress("aesthetic (zero-shot)…")
    aesthetic = (X @ concept(AESTHETIC_POS).T).mean(1) - (X @ concept(AESTHETIC_NEG).T).mean(1)

    # ---- risk categories: max cosine to the concept set ----
    risk = {}
    for cat, prompts in RISK.items():
        on_progress(f"risk:{cat} (zero-shot)…")
        risk[cat] = (X @ concept(prompts).T).max(1)

    # ---- nsfw: blend the real Falconsai ViT classifier with the zero-shot signal
    # at the benchmark-optimal weight (see eval/nsfw_bench.py / NSFW_W). ----
    on_progress("nsfw (Falconsai ViT pass)…")
    falcon = _falconsai_nsfw(paths, on_progress)
    nsfw_blend = NSFW_W * falcon + (1 - NSFW_W) * _pct(risk["nsfw"])

    nov_p, aes_p = _pct(novelty), _pct(aesthetic)
    interesting = 0.6 * nov_p + 0.4 * aes_p  # initial blend (zero-shot aesthetic only)
    # ---- duplicate groups (reuse the kNN graph): union near-identical images so the
    # ranked tabs show one canonical per group with its copies, like search dedup. ----
    on_progress("duplicate groups…")
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(1, nbr.shape[1]):
            if dist[i, j] <= 0.015:  # cosine distance ≤ 0.015  ⇔  similarity ≥ 0.985
                ri, rj = find(i), find(int(nbr[i, j]))
                if ri != rj:
                    parent[ri] = rj
    from collections import defaultdict

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    canon_idx_for: dict[int, int] = {}  # global idx → its canonical idx
    canonical_idx: list[int] = []
    canonical: list[str] = []
    dupes: dict[str, list[str]] = {}
    for members in groups.values():
        members.sort(key=lambda i: -interesting[i])  # rep = highest-interesting copy
        rep_i = members[0]
        canonical_idx.append(rep_i)
        canonical.append(paths[rep_i])
        for m in members:
            canon_idx_for[m] = rep_i
        if len(members) > 1:
            dupes[paths[rep_i]] = [paths[i] for i in members]
    on_progress(f"  {len(canonical)} canonical of {n}")

    # ---- heavy real aesthetic models — compute on canonical only, broadcast to dupes ----
    # Sort canonical_idx by path so canon_paths is strictly lexicographic — partial
    # runs (the heavy passes that hours and get killed) accumulate cache entries
    # against the same alphabetical prefix every time, making subset comparisons
    # across metrics apples-to-apples.
    canonical_idx.sort(key=lambda i: paths[i])
    canon_paths = [paths[i] for i in canonical_idx]
    canon_X = X[canonical_idx]  # for the clip-l14 fast path

    on_progress(f"aesthetic_v2: LAION-Aes V2 (MLP on CLIP-L/14) over {len(canon_paths)} canonical…")
    try:
        aes_v2_canon = _aesthetic_v2(canon_paths, model, canon_X, on_progress)
    except Exception as e:
        on_progress(f"  aesthetic_v2 FAILED: {type(e).__name__}: {e}")
        aes_v2_canon = np.full(len(canon_paths), np.nan, dtype=np.float32)

    on_progress(f"pickscore: PickScore v1 over {len(canon_paths)} canonical…")
    try:
        pick_canon = _pickscore(canon_paths, on_progress)
    except Exception as e:
        on_progress(f"  pickscore FAILED: {type(e).__name__}: {e}")
        pick_canon = np.full(len(canon_paths), np.nan, dtype=np.float32)

    on_progress(f"aesthetic_v25: Aesthetic Predictor V2.5 over {len(canon_paths)} canonical…")
    try:
        aes_v25_canon = _aesthetic_v25(canon_paths, on_progress)
    except Exception as e:
        on_progress(f"  aesthetic_v25 FAILED: {type(e).__name__}: {e}")
        aes_v25_canon = np.full(len(canon_paths), np.nan, dtype=np.float32)

    on_progress(f"hps_v21: HPS-v2.1 over {len(canon_paths)} canonical…")
    try:
        hps_canon = _hps_v21(canon_paths, on_progress)
    except Exception as e:
        on_progress(f"  hps_v21 FAILED: {type(e).__name__}: {e}")
        hps_canon = np.full(len(canon_paths), np.nan, dtype=np.float32)

    # Broadcast canonical scores to every member of the group.
    aes_v2 = np.zeros(n, dtype=np.float32)
    pick = np.zeros(n, dtype=np.float32)
    aes_v25 = np.zeros(n, dtype=np.float32)
    hps = np.zeros(n, dtype=np.float32)
    canon_pos = {ci: k for k, ci in enumerate(canonical_idx)}
    for i in range(n):
        k = canon_pos[canon_idx_for[i]]
        aes_v2[i] = aes_v2_canon[k]
        pick[i] = pick_canon[k]
        aes_v25[i] = aes_v25_canon[k]
        hps[i] = hps_canon[k]

    aes_v2_p = _pct(aes_v2) if not np.all(np.isnan(aes_v2)) else np.zeros(n, dtype=np.float64)
    pick_p = _pct(pick) if not np.all(np.isnan(pick)) else np.zeros(n, dtype=np.float64)
    aes_v25_p = _pct(aes_v25) if not np.all(np.isnan(aes_v25)) else np.zeros(n, dtype=np.float64)
    hps_p = _pct(hps) if not np.all(np.isnan(hps)) else np.zeros(n, dtype=np.float64)

    metrics = {
        "interesting": _pct(interesting),  # unchanged blend (novelty + zero-shot aesthetic)
        "novelty": nov_p, "aesthetic": aes_p,
        "aesthetic_v2": aes_v2_p, "pickscore": pick_p,
        "aesthetic_v25": aes_v25_p, "hps_v21": hps_p,
        "nsfw": _pct(nsfw_blend), "private": _pct(risk["private"]),
    }

    scores = {paths[i]: {m: round(float(metrics[m][i]), 4) for m in metrics} for i in range(n)}

    out = {"model": model, "n": n, "metrics": list(metrics.keys()), "scores": scores,
           "canonical": canonical, "dupes": dupes}
    SCORES_JSON.write_text(json.dumps(out))
    on_progress(f"wrote {SCORES_JSON}  ({len(canonical)} unique of {n})")
    return out
