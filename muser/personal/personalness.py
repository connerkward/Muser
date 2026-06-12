"""Personal-ness fusion classifier — the core of the triage.

Scores every personal-library image with **P ∈ [0,1]** (higher = more *personal*,
lower = more *reference/aesthetic*) by fusing five **local, $0** signals, then assigns a
bucket (**personal / in_between / reference**) and an **uncertainty** ∈ [0,1] used to
pick which images to send to the VLM.

Signals
  R  reference-likeness — a logistic-regression discriminator on SigLIP embeddings,
     trained with the **aesthetic library as the positive (reference) pole** and a
     sample of the personal Takeout as the negative. `1-R` is the broad appearance-based
     personal-ness. This is the "train a classifier on muser content" piece. (PU note:
     the personal sample contains some genuinely reference-like images; that contamination
     is benign — those *should* read reference-like.)
  F  faces — a face from a large, recurring people-cluster, or selfie geometry (a big
     frontal face). High-precision personal indicator.
  A  album prior — the auto "Photos from YYYY" camera roll leans personal; named albums
     are neutral-to-reference. Weak nudge only.
  Q  aesthetic quality — from score.py's aesthetic_v2 if present. Pulls a beautiful
     personal photo into *in_between*. Optional (neutral if scores.json absent).
  M  metadata/content — Google people-tags (near-certain personal), and zero-shot
     screenshot / document-receipt detection (personal-utility).

Fusion is deliberately rule-based and legible (not a second black box): a broad base of
`1-R`, with the high-precision signals (faces, people-tags, screenshots) allowed to
override toward personal, and the album prior as a small nudge. Weights are module-level
constants for tuning. Independent verification (contact sheets + held-out labels) lives
outside this file — a metric defined here must NOT be its own judge.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..facets import Sidecar
from ..index import MuserIndex
from ..paths import MUSER_HOME, data_file

_SIDE = Sidecar("personalness")
MODEL_FILE = data_file("personalness_model.npz")   # trained R weights (reuse across runs)
VERSION = 1

# Fusion weights / thresholds — tune via the verification loop, not by eye.
RECURRING_FACE_FULL = 12   # a people-cluster this big ⇒ recurring person ⇒ F≈1
SELFIE_AREA_FULL = 0.18    # face-box area / image area at/above this ⇒ selfie ⇒ F≈1
P_PERSONAL = 0.55          # P ≥ this ⇒ personal (or in_between if also aesthetic)
P_REFERENCE = 0.40         # P ≤ this ⇒ reference
# Evidence-driven fusion weights (sum≈1). Per user directive: HEAVILY bias toward
# camera-capture + user-tagged people; appearance (1-R) is only a faint tiebreak.
W_EVIDENCE = 0.45   # recurring-face / selfie geometry
W_CAMERA = 0.40     # EXIF camera capture — "a real photo I took"
W_APPEARANCE = 0.15 # 1-R, weak tiebreak only
# Benchmark (gpt-4o-mini judge) finding: camera-EXIF ALONE over-promotes reference —
# the user photographs reference material too (paintball gear, vehicle interiors, textures,
# LEGO). So camera no longer FORCES personal; it floors to in_between (reviewable) and the
# W_CAMERA weight + faces/people decide personal. Honors the camera bias without the false
# positives. (Takeout also strips EXIF from ~94% of files, so this only fires on ~6% anyway.)
CAM_FLOOR = 0.48    # camera-taken → at least in_between (not buried in reference, not forced personal)
PEOPLE_FLOOR = 0.95 # an image with a person the user TAGGED in Google Photos → decisive
DOC_FLOOR = 0.50    # docs/screenshots → reviewable in_between, NOT auto-personal (logos etc.)
Q_AESTHETIC = 0.70         # aesthetic_v2 ≥ this + personal ⇒ in_between
ALBUM_NUDGE = 0.08         # ± nudge from camera-roll vs named album
DOC_PCTL = 0.90            # zero-shot doc/screenshot above this percentile ⇒ utility
CONCEPTS = {
    "doc": "a screenshot, a scanned document, a receipt, or a page of text",
    "selfie": "a selfie or a casual snapshot of people",
}


# ---- vector IO ----------------------------------------------------------------

def _table_vectors(idx: MuserIndex, model: str):
    """(paths, X float32 [n,d], wh [n,2]) for every row in a model's table."""
    t = idx._open(model)
    if t is None:
        return [], np.zeros((0, 1), np.float32), np.zeros((0, 2), np.float32)
    rows = t.search().select(["path", "vector", "width", "height"]).limit(100_000_000).to_list()
    paths = [r["path"] for r in rows]
    X = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    wh = np.asarray([[r.get("width") or 0, r.get("height") or 0] for r in rows], dtype=np.float32)
    return paths, X, wh


# ---- R: reference-likeness discriminator --------------------------------------

def _train_R(Xa: np.ndarray, Xp: np.ndarray, seed: int = 0):
    """Fit logistic regression: class 1 = aesthetic/reference (Xa), class 0 = personal
    (Xp sample). Balanced. Returns (w, b). Saves to MODEL_FILE."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    n = min(len(Xa), len(Xp), 12000)
    a = Xa[rng.choice(len(Xa), n, replace=False)]
    p = Xp[rng.choice(len(Xp), n, replace=False)]
    X = np.vstack([a, p])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    w, b = clf.coef_[0].astype(np.float32), np.float32(clf.intercept_[0])
    np.savez(MODEL_FILE, w=w, b=b)
    return w, b


def _load_R():
    if not MODEL_FILE.exists():
        return None
    try:
        z = np.load(MODEL_FILE)
        return z["w"].astype(np.float32), np.float32(z["b"])
    except Exception:
        return None


def _R_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """P(reference) per row — the discriminator's sigmoid."""
    z = X @ w + b
    return 1.0 / (1.0 + np.exp(-z))


# ---- zero-shot concept scores (in the SigLIP space) ---------------------------

def _concept_scores(Xp: np.ndarray, model: str) -> dict[str, np.ndarray]:
    """Cosine of each personal image to each concept prompt, percentile-normalized."""
    from ..registry import load_model
    emb = load_model(model)
    keys = list(CONCEPTS)
    T = emb.embed_queries([CONCEPTS[k] for k in keys])  # (c, d) normalized
    Xn = Xp / np.clip(np.linalg.norm(Xp, axis=1, keepdims=True), 1e-12, None)
    out = {}
    for i, k in enumerate(keys):
        cos = Xn @ T[i]
        out[k] = _percentile(cos)
    return out


def _percentile(v: np.ndarray) -> np.ndarray:
    """Rank-normalize to [0,1] so thresholds are corpus-relative (like score.py)."""
    order = np.argsort(np.argsort(v))
    return (order / max(1, len(v) - 1)).astype(np.float32)


# ---- signal helpers -----------------------------------------------------------

def _face_signal(entry: dict, wh, clusters: dict) -> float:
    """F ∈ [0,1] from per-image faces: recurring-cluster membership + selfie geometry."""
    if not entry:
        return 0.0
    faces = entry.get("faces", [])
    if not faces:
        return 0.0
    W = entry.get("W") or (wh[0] if wh is not None else 0) or 0
    H = entry.get("H") or (wh[1] if wh is not None else 0) or 0
    area = float(W * H) or 1.0
    recurring = 0.0
    selfie = 0.0
    for f in faces:
        c = f.get("c", -1)
        if c is not None and c >= 0:
            size = clusters.get(str(c), {}).get("size", 0)
            recurring = max(recurring, min(1.0, size / RECURRING_FACE_FULL))
        fa = (f.get("fw", 0) * f.get("fh", 0)) / area
        selfie = max(selfie, min(1.0, fa / SELFIE_AREA_FULL))
    return max(recurring, selfie)


def _classify_one(R, F, A, Q, M, doc, has_people, cam) -> tuple[float, str, float, dict]:
    # Evidence-driven: personal-ness needs POSITIVE evidence (faces / people-tags /
    # camera capture), NOT merely "doesn't look like the aesthetic library". The early
    # verification showed `1-R` alone tags saved internet media (game caps, movie stills,
    # downloaded art) as personal — they're just non-aesthetic, not personal. So the
    # appearance prior `base` is only a weak tiebreak; faces/people/camera carry it.
    base = 1.0 - float(R)
    p = W_EVIDENCE * float(F) + W_CAMERA * float(cam) + W_APPEARANCE * base
    # Floors, weakest→strongest. A doc/screenshot is at most reviewable (in_between);
    # a camera capture is personal; a user-TAGGED person is decisive. This ordering is
    # the user's explicit bias: camera-taken / camera-metadata / tagged-people first.
    if doc:
        p = max(p, DOC_FLOOR)
    if cam:
        p = max(p, CAM_FLOOR)
    if has_people:
        p = max(p, PEOPLE_FLOOR)
    p = float(np.clip(p + ALBUM_NUDGE * (2 * A - 1), 0.0, 1.0))

    if p >= P_PERSONAL:
        bucket = "in_between" if Q >= Q_AESTHETIC else "personal"
    elif p <= P_REFERENCE:
        bucket = "reference"
    else:
        bucket = "in_between"

    # uncertainty: near a boundary OR the personal-evidence signals disagree.
    boundary = 1.0 - 2.0 * abs(p - 0.5)
    evidence = max(float(F), 1.0 if has_people else 0.0)
    personal_views = [evidence, float(cam), base]
    disagree = float(np.std(personal_views)) * 2.0
    unc = float(np.clip(max(boundary, disagree), 0.0, 1.0))
    sig = {"R": round(float(R), 3), "F": round(float(F), 3), "A": round(float(A), 3),
           "Q": round(float(Q), 3), "M": round(float(M), 3), "doc": bool(doc),
           "people": bool(has_people), "cam": bool(cam)}
    return round(p, 4), bucket, round(unc, 4), sig


# ---- main ---------------------------------------------------------------------

def classify(model: str = "siglip2-b", retrain: bool = False, progress=None) -> dict:
    """Compute P / bucket / uncertainty for every personal image; write personalness.json."""
    def log(m):
        if progress:
            progress(m)

    idx = MuserIndex()                                   # personal root (MUSER_HOME)
    paths, Xp, wh = _table_vectors(idx, model)
    if not paths:
        log("no personal index — run `muser personal ingest` first")
        return {}
    log(f"{len(paths)} personal images")

    # R: train against the aesthetic library (explicit ~/.muser/db, bypassing MUSER_HOME)
    R = _load_R()
    if R is None or retrain:
        aidx = MuserIndex(db_path=Path.home() / ".muser" / "db")
        _, Xa, _ = _table_vectors(aidx, model)
        if len(Xa) < 100:
            log("aesthetic library too small to train R — using neutral R=0.5")
            Rproba = np.full(len(paths), 0.5, np.float32)
        else:
            log(f"training reference-likeness on {len(Xa)} aesthetic vs personal …")
            w, b = _train_R(Xa, Xp)
            Rproba = _R_proba(Xp, w, b)
    else:
        w, b = R
        Rproba = _R_proba(Xp, w, b)

    log("zero-shot concept scoring (doc / selfie) …")
    concepts = _concept_scores(Xp, model)
    doc_pctl = concepts["doc"]

    # facets
    from .. import faces as _faces
    from . import takeout
    _faces.prime_cache_from_sidecar()
    fclusters = _faces.clusters()
    faces_entries = _faces.sidecar().load()
    tmeta = takeout.sidecar().load()
    scores = _load_scores()

    log("fusing signals …")
    entries: dict[str, dict] = {}
    counts = {"personal": 0, "in_between": 0, "reference": 0}
    for i, path in enumerate(paths):
        fe = faces_entries.get(path, {})
        tm = tmeta.get(path, {})
        F = _face_signal(fe, wh[i], fclusters)
        A = 0.5 + (ALBUM_NUDGE if tm.get("roll") else -ALBUM_NUDGE)  # roll→personal lean
        A = float(np.clip(A, 0.0, 1.0))
        Q = float((scores.get(path, {}) or {}).get("aesthetic_v2", 0.5))
        has_people = bool(tm.get("people"))
        cam = bool(tm.get("cam"))
        doc = doc_pctl[i] >= DOC_PCTL or _is_screenshot(path, wh[i])
        M = 1.0 if has_people else (0.85 if doc else 0.5)
        p, bucket, unc, sig = _classify_one(Rproba[i], F, A, Q, M, doc, has_people, cam)
        counts[bucket] += 1
        try:
            st = os.stat(path)
            m, s = st.st_mtime_ns, st.st_size
        except OSError:
            m, s = 0, 0
        entries[path] = {"m": m, "s": s, "v": VERSION, "p": p, "bucket": bucket,
                         "unc": unc, "sig": sig}
        if progress and i % 5000 == 0 and i:
            log(f"  {i}/{len(paths)}")

    _SIDE.save(entries)
    _SIDE._primed = False
    _SIDE.prime()
    log(f"done — personal={counts['personal']} in_between={counts['in_between']} "
        f"reference={counts['reference']} → personalness.json")
    return counts


def _is_screenshot(path: str, wh) -> bool:
    name = os.path.basename(path).lower()
    if "screenshot" in name or "screen shot" in name:
        return True
    if not path.lower().endswith(".png"):
        return False
    w, h = (wh[0] or 0), (wh[1] or 0)
    if w and h:
        ar = max(w, h) / max(1.0, min(w, h))
        # common phone/desktop screen aspect ratios
        return 1.3 <= ar <= 2.4
    return False


def _load_scores() -> dict:
    f = MUSER_HOME / "scores.json"
    if not f.exists():
        return {}
    try:
        import json
        d = json.loads(f.read_text())
        return d.get("scores", d)
    except Exception:
        return {}


# ---- supervised learning from user labels -----------------------------------

def _load_eval_labels() -> dict:
    """Load human-labeled verdicts from eval_labels.json."""
    f = data_file("eval_labels.json")
    if not f.exists():
        return {}
    try:
        import json
        return json.loads(f.read_text())
    except Exception:
        return {}


def train(model: str = "siglip2-b", hold_out_fraction: float = 0.2, progress=None) -> dict:
    """Train a supervised head on user labels; recompute personalness with blend.

    Returns: {trained: bool, n_labeled: int, n_train: int, n_test: int,
              accuracy: float, confusion: {...}, per_bucket: {...}, message: str}
    """
    def log(m):
        if progress:
            progress(m)

    labels = _load_eval_labels()
    if not labels:
        return {"trained": False, "message": "no labels yet"}

    log(f"loaded {len(labels)} labeled images")

    idx = MuserIndex()
    paths, Xp, wh = _table_vectors(idx, model)
    if not paths:
        return {"trained": False, "message": "no personal index"}

    # Map paths to indices for fast lookup
    path_to_idx = {p: i for i, p in enumerate(paths)}

    # Collect labeled examples: filter to those in the index
    labeled_indices = []
    labeled_verdicts = []
    labeled_paths = []
    for path, entry in labels.items():
        verdict = entry.get("label")
        if verdict not in ("personal", "in_between", "reference"):
            continue
        if path not in path_to_idx:
            continue  # not indexed yet
        idx_in_corpus = path_to_idx[path]
        labeled_indices.append(idx_in_corpus)
        labeled_verdicts.append(verdict)
        labeled_paths.append(path)

    if len(labeled_indices) < 10:
        return {"trained": False, "message": f"need ≥10 labeled; have {len(labeled_indices)}"}

    log(f"using {len(labeled_indices)} labeled examples")

    # Feature matrix: (R, F, A, Q, M, doc_binary, has_people_binary, cam_binary) + the full embedding
    # Actually, we'll train TWO things:
    # 1. Re-train R on the labeled examples to see if it improves
    # 2. Fit a supervised head on (R, F, A, Q, M, doc, people, cam) → bucket

    # For now, let's fit a simpler model: logistic regression on (R_val, F, A, Q, M, doc, people, cam)
    # where personal=1, reference=0, in_between=0.5 (we'll use binary: personal vs not-personal for the main head)

    from .. import faces as _faces
    from . import takeout
    _faces.prime_cache_from_sidecar()
    fclusters = _faces.clusters()
    faces_entries = _faces.sidecar().load()
    tmeta = takeout.sidecar().load()
    scores = _load_scores()

    # Load the existing R model or use neutral
    R_model = _load_R()
    if R_model is None:
        log("no R model found; using neutral R=0.5 for feature matrix")
        Rproba = np.full(len(paths), 0.5, np.float32)
    else:
        w, b = R_model
        Rproba = _R_proba(Xp, w, b)

    concepts = _concept_scores(Xp, model)
    doc_pctl = concepts["doc"]

    # Build feature matrix for labeled examples
    X_feat = []
    y_personal = []
    y_bucket = []

    for i in range(len(labeled_paths)):
        path = labeled_paths[i]
        corpus_idx = labeled_indices[i]
        fe = faces_entries.get(path, {})
        tm = tmeta.get(path, {})
        F = _face_signal(fe, wh[corpus_idx], fclusters)
        A = 0.5 + (ALBUM_NUDGE if tm.get("roll") else -ALBUM_NUDGE)
        A = float(np.clip(A, 0.0, 1.0))
        Q = float((scores.get(path, {}) or {}).get("aesthetic_v2", 0.5))
        has_people = bool(tm.get("people"))
        cam = bool(tm.get("cam"))
        doc = doc_pctl[corpus_idx] >= DOC_PCTL or _is_screenshot(path, wh[corpus_idx])
        R = float(Rproba[corpus_idx])

        # Feature vector: (R, F, A, Q, doc, has_people, cam, 1-R)
        feat = [R, F, A, Q, float(doc), float(has_people), float(cam), 1.0 - R]
        X_feat.append(feat)

        verdict = labeled_verdicts[i]
        y_bucket.append({"personal": 0, "in_between": 1, "reference": 2}[verdict])
        y_personal.append(1.0 if verdict == "personal" else 0.0)

    X_feat = np.asarray(X_feat, dtype=np.float32)
    y_personal = np.asarray(y_personal, dtype=np.float32)
    y_bucket = np.asarray(y_bucket, dtype=np.int32)

    # Split: 80% train, 20% held-out
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X_feat))
    n_train = int(len(X_feat) * (1 - hold_out_fraction))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train, X_test = X_feat[train_idx], X_feat[test_idx]
    y_train_bucket, y_test_bucket = y_bucket[train_idx], y_bucket[test_idx]
    y_train_personal, y_test_personal = y_personal[train_idx], y_personal[test_idx]

    # Fit multiclass logistic regression (3 buckets)
    from sklearn.linear_model import LogisticRegression
    clf_bucket = LogisticRegression(max_iter=2000, C=1.0)
    clf_bucket.fit(X_train, y_train_bucket)

    log(f"trained on {len(train_idx)} examples, testing on {len(test_idx)}")

    # Held-out evaluation
    y_pred = clf_bucket.predict(X_test)
    accuracy = float(np.mean(y_pred == y_test_bucket))

    # Confusion matrix on held-out set
    confusion = {a: {b: 0 for b in ("personal", "in_between", "reference")}
                 for a in ("personal", "in_between", "reference")}
    idx_to_bucket = {0: "personal", 1: "in_between", 2: "reference"}
    for pred, truth in zip(y_pred, y_test_bucket):
        pred_b = idx_to_bucket[int(pred)]
        truth_b = idx_to_bucket[int(truth)]
        confusion[truth_b][pred_b] += 1

    # Binary personal vs reference (ignoring in_between)
    binary_personal_test = y_test_bucket[y_test_bucket != 1]  # exclude in_between
    binary_pred = y_pred[y_test_bucket != 1]
    if len(binary_personal_test) > 0:
        binary_personal_test = (binary_personal_test == 0).astype(int)
        binary_pred = (binary_pred == 0).astype(int)
        binary_acc = float(np.mean(binary_pred == binary_personal_test))
    else:
        binary_acc = None

    per_bucket = {}
    for b in ("personal", "in_between", "reference"):
        tot = sum(confusion[b].values())
        per_bucket[b] = {
            "n": tot,
            "correct": confusion[b][b],
            "precision": round(confusion[b][b] / tot, 3) if tot else None
        }

    log(f"held-out accuracy: {accuracy:.1%} (binary personal: {binary_acc:.1%})")

    # Save the trained classifier for later use
    SUPERVISED_FILE = data_file("personalness_supervised.npz")
    np.savez(SUPERVISED_FILE,
             coef=clf_bucket.coef_.astype(np.float32),
             intercept=clf_bucket.intercept_.astype(np.float32))

    return {
        "trained": True,
        "n_labeled": len(labeled_indices),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "accuracy": round(accuracy, 3),
        "binary_personal_accuracy": round(binary_acc, 3) if binary_acc else None,
        "confusion": confusion,
        "per_bucket": per_bucket,
        "message": f"trained on {len(train_idx)}, tested on {len(test_idx)}: {accuracy:.1%} accuracy"
    }


# ---- read side (for the service / triage / verification) ----------------------

def prime() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    return _SIDE.lookup(path)


def all_entries() -> dict:
    return _SIDE.load()


def exists() -> bool:
    return _SIDE.exists()


def sidecar() -> Sidecar:
    return _SIDE
