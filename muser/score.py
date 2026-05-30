"""Per-image scores — "Interesting" and "Review" (sensitivity triage).

Batch artifact (`muser score`): reuse the indexed siglip2-b embeddings (no new model
pass) to compute, for every image:

  - novelty      — how isolated the image is in embedding space (mean similarity to
                   its nearest neighbors, inverted). High = unusual / one-of-a-kind.
  - aesthetic    — a zero-shot "striking photo" vs "dull snapshot" signal (a weak,
                   free proxy; a real aesthetic model like PickScore is the upgrade).
  - interesting  — blend of novelty + aesthetic.
  - nsfw / private / political — zero-shot similarity to sensitivity concept sets.
                   These are HEURISTIC FLAGS FOR HUMAN REVIEW of your own library
                   (find candidates before sharing), not classifiers / verdicts.

Each metric is percentile-normalized to [0,1] so they're comparable and sortable.
Writes ~/.muser/scores.json; the Explore "Interesting" and "Review" tabs read it.
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


def score_all(model: str = "siglip2-b", on_progress=print) -> dict:
    idx = MuserIndex()
    t = idx._open(model)
    if t is None:
        raise RuntimeError(f"nothing indexed for {model}")
    rows = t.search().select(["path", "vector"]).limit(100_000_000).to_list()
    paths = [r["path"] for r in rows]
    X = np.asarray([r["vector"] for r in rows], dtype=np.float32)  # already L2-normalized
    n = len(paths)
    on_progress(f"loaded {n} vectors (dim {X.shape[1]})")

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
    interesting = 0.6 * nov_p + 0.4 * aes_p
    metrics = {
        "interesting": _pct(interesting), "novelty": nov_p, "aesthetic": aes_p,
        "nsfw": _pct(nsfw_blend), "private": _pct(risk["private"]),
    }

    scores = {paths[i]: {m: round(float(metrics[m][i]), 4) for m in metrics} for i in range(n)}

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
    inter = metrics["interesting"]
    canonical, dupes = [], {}
    for members in groups.values():
        members.sort(key=lambda i: -inter[i])  # representative = highest-interesting copy
        canon = paths[members[0]]
        canonical.append(canon)
        if len(members) > 1:
            dupes[canon] = [paths[i] for i in members]

    out = {"model": model, "n": n, "metrics": list(metrics.keys()), "scores": scores,
           "canonical": canonical, "dupes": dupes}
    SCORES_JSON.write_text(json.dumps(out))
    on_progress(f"wrote {SCORES_JSON}  ({len(canonical)} unique of {n})")
    return out
