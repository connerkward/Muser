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
    "political": [
        "a political protest or rally", "a political campaign poster",
        "a politician giving a speech", "political propaganda", "a controversial political image",
    ],
}


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
    _, dist = ann.neighbor_graph  # cosine distances incl. self at col 0
    novelty = dist[:, 1:].mean(axis=1)  # higher distance = more isolated = more novel

    # ---- aesthetic: striking vs dull ----
    on_progress("aesthetic (zero-shot)…")
    aesthetic = (X @ concept(AESTHETIC_POS).T).mean(1) - (X @ concept(AESTHETIC_NEG).T).mean(1)

    # ---- risk categories: max cosine to the concept set ----
    risk = {}
    for cat, prompts in RISK.items():
        on_progress(f"risk:{cat} (zero-shot)…")
        risk[cat] = (X @ concept(prompts).T).max(1)

    nov_p, aes_p = _pct(novelty), _pct(aesthetic)
    interesting = 0.6 * nov_p + 0.4 * aes_p
    metrics = {
        "interesting": _pct(interesting), "novelty": nov_p, "aesthetic": aes_p,
        **{c: _pct(risk[c]) for c in risk},
    }

    scores = {paths[i]: {m: round(float(metrics[m][i]), 4) for m in metrics} for i in range(n)}
    out = {"model": model, "n": n, "metrics": list(metrics.keys()), "scores": scores}
    SCORES_JSON.write_text(json.dumps(out))
    on_progress(f"wrote {SCORES_JSON}")
    return out
