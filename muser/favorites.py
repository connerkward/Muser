"""Taste model — a per-image P(favorite) trained on the user's favorites folder.

The user keeps a folder of *verified favorite* outpaintings. Those are stored as
watermarked "Verify" renders that are (correctly) excluded from the aesthetic index,
so we can't train on them directly — and even if we could, a watermark-vs-clean split
would just teach the model to detect the watermark. Instead we **recover each
favorite's CLEAN twin** by nearest-neighbor in the index (the watermark is a small
perturbation, so the clean version is the top cosine match), then fit a logistic
regression on the SigLIP embeddings: positives = recovered clean twins, negatives =
the rest of the library.

It is a SOFT signal, not a strong classifier: favorites occupy the same semantic
region as the rest of the outpaintings, so separability is weak (~3× lift over base
rate). Surfaced as an opt-in "Taste" weight in the Sort blend — a gentle nudge toward
the user's picks, not an authoritative ranker. Per-image P(favorite) is persisted to
``favorites_pred.json`` and looked up like the other facets.

Mirrors the personal instance's ``cleanup.train_model`` / ``personalness.train`` recipe
(logreg on embeddings), but sourced from a folder of positives instead of flags.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .paths import data_file

PRED = data_file("favorites_pred.json")
_CACHE = {"mtime": -1.0, "preds": {}}


def available() -> bool:
    return PRED.exists()


def preds() -> dict:
    """{path: P(favorite) in 0..1}, cached by favorites_pred.json mtime."""
    try:
        mt = PRED.stat().st_mtime
    except OSError:
        return {}
    if mt != _CACHE["mtime"]:
        try:
            _CACHE["preds"] = json.loads(PRED.read_text())
            _CACHE["mtime"] = mt
        except Exception:
            return {}
    return _CACHE["preds"]


def lookup(path: str):
    return preds().get(path)


def _library_vectors(model: str):
    from .index import MuserIndex
    t = MuserIndex()._open(model)
    if t is None:
        return [], None
    rows = t.search().select(["path", "vector"]).limit(10_000_000).to_list()
    paths = [r["path"] for r in rows]
    X = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return paths, X


def train(fav_paths, model: str = "siglip2-b", knn_thresh: float = 0.80, on_progress=print) -> dict:
    """Recover clean twins of the favorites, fit a logreg P(favorite), persist per-image
    predictions. Returns a report dict (recovery, CV precision/recall, counts)."""
    from .registry import load_model

    fav_paths = [str(p) for p in fav_paths]
    paths, Xn = _library_vectors(model)
    if not paths:
        return {"trained": False, "message": "empty index"}

    emb = load_model(model)
    fv = np.asarray(emb.embed_images(fav_paths), np.float32)
    fv /= (np.linalg.norm(fv, axis=1, keepdims=True) + 1e-9)
    sims = fv @ Xn.T
    twin_i = sims.argmax(1)
    twin_s = sims.max(1)
    pos_idx = sorted({int(twin_i[i]) for i in range(len(fav_paths)) if twin_s[i] >= knn_thresh})
    matched = int((twin_s >= knn_thresh).sum())
    on_progress(f"recovered {len(pos_idx)} clean positives from {matched}/{len(fav_paths)} "
                f"matched favorites (median sim {float(np.median(twin_s)):.3f})")
    if len(pos_idx) < 10:
        return {"trained": False, "message": f"only {len(pos_idx)} positives recovered — need ≥10"}

    y = np.zeros(len(paths), np.float32)
    y[pos_idx] = 1.0
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, Xn, y, cv=cv, method="predict_proba")[:, 1]
    pred = proba >= 0.7
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, float(y.sum()))
    on_progress(f"CV @P≥0.7: precision {precision:.1%}, recall {recall:.1%}")

    clf.fit(Xn, y)
    w = clf.coef_.reshape(-1).astype(np.float32)
    b = float(clf.intercept_[0])
    P = 1.0 / (1.0 + np.exp(-(Xn @ w + b)))
    preds_map = {paths[i]: round(float(P[i]), 4) for i in range(len(paths))}
    tmp = PRED.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(preds_map))
    os.replace(tmp, PRED)
    _CACHE["mtime"] = -1.0

    is_pos = set(pos_idx)
    strong = int(sum(1 for i in range(len(P)) if P[i] >= 0.8 and i not in is_pos))
    return {"trained": True, "positives": len(pos_idx), "matched": matched,
            "precision_at_70": round(precision, 3), "recall_at_70": round(recall, 3),
            "strong_new": strong,
            "message": (f"trained on {len(pos_idx)} recovered favorites — {precision:.0%} precision / "
                        f"{recall:.0%} recall @P≥0.7 (soft signal); {strong} strong new candidates (P≥0.8)")}
