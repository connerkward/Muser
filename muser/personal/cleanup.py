"""Delete-candidate (junk) detector — a local, $0 quality facet.

Scores how likely an image is *junk you'd delete* from intrinsic, content-free
signals (no model, just pixels):

- **blur**  — variance of the Laplacian (focus measure). Low ⇒ out-of-focus or
  motion-blurred. The single strongest junk signal for a camera roll.
- **exposure** — fraction of near-black / near-white pixels + mean luma. A frame
  that's mostly black (lens-cap, pocket shot) or blown out is junk.
- **tiny** — original megapixels. Thumbnails / tiny saved assets are rarely
  worth keeping in a *photo* library.

`delete_score()` fuses these as the **worst single defect** (a photo that's very
blurry OR very dark OR tiny is a candidate), returning 0–100 + human reasons.
Near-duplicate membership is layered on at the service endpoint from
`scores.json` (it already groups dupes), so this module stays self-contained and
dependency-free beyond opencv (already a core dep via `color.py`).

Built on `facets.Sidecar` (`cleanup.json`, incremental by mtime+size, parallel
scan) like `color.py`/`aidet.py`. Nothing here deletes anything — the Cleanup
tab only *flags* candidates into the Evaluate delete set for the user to confirm.
"""
from __future__ import annotations

import numpy as np

from ..embedders import _load_rgb
from ..facets import Sidecar

_SIDE = Sidecar("cleanup")
SAMPLE = 512          # downscale for the blur/exposure measure (focus survives this)


def available() -> bool:
    try:
        import cv2  # noqa: F401
    except Exception:
        return False
    return True


def sidecar() -> Sidecar:
    return _SIDE


def _compute(path: str) -> dict:
    import cv2
    from PIL import Image
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass
    # original dimensions (cheap header read) for the tiny-resolution signal
    try:
        with Image.open(path) as im0:
            w0, h0 = im0.size
    except Exception:
        w0 = h0 = 0
    img = _load_rgb(path, max_side=SAMPLE)
    g = np.asarray(img.convert("L"), dtype=np.uint8)
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    dark = float((g < 16).mean())
    bright = float((g > 240).mean())
    return {"blur": round(blur, 1), "dark": round(dark, 4), "bright": round(bright, 4),
            "w": int(w0), "h": int(h0), "mp": round(w0 * h0 / 1e6, 3)}


def scan(paths, progress=None, workers: int = 8) -> dict:
    """Build/refresh the junk-signal facet over ``paths`` (incremental)."""
    return _SIDE.scan(paths, _compute, progress=progress, workers=workers)


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


# --- scoring thresholds (module-level for one-line tuning) ---
BLUR_HARD = 60.0      # var below this ⇒ clearly blurry
BLUR_SOFT = 140.0     # var below this ⇒ soft (mild signal)
DARK_HARD = 0.70      # ≥70% near-black pixels ⇒ mostly black
DARK_SOFT = 0.50
BRIGHT_HARD = 0.60    # ≥60% near-white ⇒ blown out
MP_TINY = 0.08        # < 0.08 MP (~300×260) ⇒ tiny
MP_SMALL = 0.20


def delete_score(e: dict) -> tuple[int, list[str]]:
    """Fuse one image's junk signals → (score 0–100, reasons). The score is the
    WORST single defect — one bad-enough axis makes it a delete candidate."""
    blur = e.get("blur", 9e9)
    dark = e.get("dark", 0.0)
    bright = e.get("bright", 0.0)
    mp = e.get("mp", 9e9)
    score = 0.0
    reasons: list[str] = []
    if blur < BLUR_SOFT:
        frac = (BLUR_SOFT - blur) / BLUR_SOFT          # 0..1
        score = max(score, 0.40 + 0.55 * min(1.0, frac))
        reasons.append("blurry" if blur < BLUR_HARD else "soft focus")
    if dark >= DARK_HARD:
        score = max(score, 0.88); reasons.append("mostly black")
    elif dark >= DARK_SOFT:
        score = max(score, 0.60); reasons.append("dark")
    if bright >= BRIGHT_HARD:
        score = max(score, 0.82); reasons.append("blown out")
    if mp < MP_TINY:
        score = max(score, 0.78); reasons.append("tiny")
    elif mp < MP_SMALL:
        score = max(score, 0.50); reasons.append("small")
    return int(round(score * 100)), reasons


# ---- learned deletion model (trained on the user's 🗑 / saved flags) ----------
def _model_files():
    from ..paths import data_file
    return data_file("delete_model.npz"), data_file("delete_pred.json")


def _deleted_npz():
    from ..paths import data_file
    return data_file("deleted_vectors.npz")


def snapshot_deleted(paths, model: str = "siglip2-b") -> int:
    """Persist the SigLIP embeddings of about-to-be-trashed images so they remain
    POSITIVE training examples for the deletion model after the file (and its
    index row) is gone — symmetric with the reference-export snapshot. Merge by
    path; called from the Trash 'move to system Trash' path BEFORE the move."""
    if not paths:
        return 0
    import numpy as np

    from ..index import MuserIndex
    try:
        import lancedb
        db = lancedb.connect(MuserIndex().db_path)
        t = db.open_table("img__siglip2_b").to_arrow()
        tp = t.column("path").to_pylist()
        tv = t.column("vector")
        want = set(paths)
        new = {tp[i]: np.asarray(tv[i].as_py(), np.float16) for i in range(len(tp)) if tp[i] in want}
    except Exception:
        return 0
    if not new:
        return 0
    f = _deleted_npz()
    merged: dict = {}
    if f.exists():
        try:
            z = np.load(f, allow_pickle=False)
            merged = {p: z["X"][i] for i, p in enumerate(z["ids"])}
        except Exception:
            merged = {}
    merged.update(new)
    ids = list(merged.keys())
    X = np.stack([merged[p] for p in ids]).astype(np.float16)
    tmp = f.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, ids=np.array(ids, dtype=object).astype("U"), X=X)
    import os as _os
    _os.replace(tmp, f)
    return len(new)


def deleted_examples():
    """Snapshotted embeddings of already-trashed images (permanent delete
    positives). Returns (paths, X) or ([], None)."""
    f = _deleted_npz()
    if not f.exists():
        return [], None
    try:
        import numpy as np
        z = np.load(f, allow_pickle=False)
        key = "ids" if "ids" in z else "paths"
        return [str(p) for p in z[key]], z["X"].astype(np.float32)
    except Exception:
        return [], None


def train_model(model: str = "siglip2-b", progress=None) -> dict:
    """Fit a P(delete) head on the SigLIP embeddings from the user's flags:
    positives = 🗑 delete-flagged; negatives = saved ('keep') + bucket-labeled
    images the user chose NOT to flag. Reports 5-fold CV precision/recall, then
    scores the WHOLE corpus and persists delete_pred.json (path → pct) so the
    Cleanup candidate list can surface model-suspected junk the heuristics miss.
    Same recipe as personalness.train — measured 96.7% precision at P≥0.7 on
    1.7k flags. No model load; vectors come from LanceDB."""
    def log(m):
        if progress:
            progress(m)
    import json as _json
    import os as _os

    from .personalness import MuserIndex, _table_vectors
    from . import evaluate as _ev
    labels = _ev._load_labels()
    pos = {p for p, e in labels.items() if e.get("flag") == "delete"}
    neg = {p for p, e in labels.items()
           if e.get("flag") == "keep"
           or (e.get("label") in ("personal", "in_between", "reference")
               and e.get("flag") != "delete")}
    neg -= pos
    if len(pos) < 20 or len(neg) < 20:
        return {"trained": False,
                "message": f"need ≥20 each; have {len(pos)} delete / {len(neg)} not"}

    paths, Xp, _wh = _table_vectors(MuserIndex(), model)
    pidx = {p: i for i, p in enumerate(paths)}
    X, y = [], []
    seen = set()
    for p in pos:
        if p in pidx:
            X.append(Xp[pidx[p]]); y.append(1.0); seen.add(p)
    for p in neg:
        if p in pidx:
            X.append(Xp[pidx[p]]); y.append(0.0); seen.add(p)
    # already-trashed images: their files/index rows are gone, but the snapshot
    # keeps them as permanent delete positives so the model doesn't forget them.
    dp, dX = deleted_examples()
    added_trashed = 0
    if dX is not None:
        for k, p in enumerate(dp):
            if p not in seen and p not in neg:
                X.append(dX[k]); y.append(1.0); added_trashed += 1
    X = np.stack(X); y = np.asarray(y, np.float32)
    log(f"{int(y.sum())} delete / {int((1 - y).sum())} not-delete"
        f"{f' (incl. {added_trashed} already-trashed)' if added_trashed else ''}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    pred = proba >= 0.7
    tp = float(((pred) & (y == 1)).sum()); fp = float(((pred) & (y == 0)).sum())
    precision = tp / max(1.0, tp + fp)
    recall = tp / max(1.0, float(y.sum()))
    log(f"CV @P≥0.7: precision {precision:.1%}, recall {recall:.1%}")

    clf.fit(X, y)
    w = clf.coef_.reshape(-1).astype(np.float32); b = float(clf.intercept_[0])
    npz_f, pred_f = _model_files()
    np.savez(npz_f, coef=w, intercept=np.float32(b), feat="emb", n=len(y))
    # score the whole corpus → persisted predictions for the candidates endpoint
    P = 1.0 / (1.0 + np.exp(-(Xp @ w + b)))
    preds = {p: int(round(float(P[i]) * 100)) for i, p in enumerate(paths) if P[i] >= 0.5}
    tmp = pred_f.with_suffix(".json.tmp")
    tmp.write_text(_json.dumps(preds))
    _os.replace(tmp, pred_f)
    _MODEL_PRED_CACHE["mtime"] = -1.0   # force re-prime
    log(f"{len(preds)} images scored P(delete)≥50%")
    return {"trained": True, "n_labeled": len(y),
            "precision_at_70": round(precision, 3), "recall_at_70": round(recall, 3),
            "suspected": len(preds),
            "message": (f"trained on {len(y)} flags — {precision:.0%} precision / "
                        f"{recall:.0%} recall @P≥0.7; {len(preds)} suspected")}


_MODEL_PRED_CACHE = {"mtime": -1.0, "preds": {}}


def model_preds() -> dict:
    """{path: pct} P(delete) predictions ≥50%, cached by delete_pred.json mtime."""
    import json as _json
    _npz, pred_f = _model_files()
    try:
        mt = pred_f.stat().st_mtime
    except OSError:
        return {}
    if mt != _MODEL_PRED_CACHE["mtime"]:
        try:
            _MODEL_PRED_CACHE["preds"] = _json.loads(pred_f.read_text())
            _MODEL_PRED_CACHE["mtime"] = mt
        except Exception:
            return {}
    return _MODEL_PRED_CACHE["preds"]


def candidates(min_score: int = 55, extra: dict | None = None) -> list[dict]:
    """All scored images at/above ``min_score``, worst first. ``extra`` maps
    path -> additional (score_boost, reason) for near-dup membership injected by
    the caller (keeps this module free of scores.json coupling)."""
    extra = extra or {}
    out: list[dict] = []
    for key, e in _SIDE.entries().items():
        path = key[0] if isinstance(key, tuple) else key
        score, reasons = delete_score(e)
        if path in extra:
            boost, reason = extra[path]
            score = max(score, boost)
            if reason and reason not in reasons:
                reasons.append(reason)
        if score >= min_score:
            out.append({"path": path, "score": score, "reasons": reasons,
                        "blur": e.get("blur"), "mp": e.get("mp")})
    out.sort(key=lambda d: -d["score"])
    return out
