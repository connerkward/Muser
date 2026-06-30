"""Face detection + recognition-embedding + clustering — a mainline Muser facet.

Backend: **InsightFace** `buffalo_l` (RetinaFace detector + ArcFace r100 recognizer,
ONNX). For each image we store *light* per-face metadata (bbox, detector score, face
size) in a `facets.Sidecar` JSON (`<root>/faces.json`) and the *heavy* 512-d ArcFace
embeddings in a sibling npz (`<root>/faces_emb.npz`) — same split as `dinov2.py`'s
canonical-vectors npz, because 512 floats per face would bloat the JSON 5×. A second
pass (`cluster`) groups the embeddings with HDBSCAN into **people**: a large,
date-spanning cluster is a recurring person.

Why mainline (not personal-only): the aesthetic library gains people-grouping too, and
the hidden personal sub-tool reuses this *same* module under its own `MUSER_HOME` root,
where "image contains a face from a big recurring cluster" is the strongest **personal**
signal feeding `personal/personalness.py`.

Degrades gracefully: `available()` is False when insightface/onnxruntime are absent
(install via the `[faces]` extra) — no faces, no crash, like `aidet.available()`.

Built on `facets.Sidecar` for the incremental (mtime+size) skip, atomic checkpointing,
and the auto-re-priming RAM cache, but runs its own scan loop (not `Sidecar.scan`) so it
can fan the embeddings out to the npz. ArcFace embeddings are already L2-normalized, so
cosine == euclidean²/2 and HDBSCAN on euclidean is a cosine clustering.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from .facets import Sidecar
from .paths import MUSER_HOME, data_file

_SIDE = Sidecar("faces")
EMB_NPZ = MUSER_HOME / "faces_emb.npz"          # face-id -> 512-d ArcFace vector
CLUSTERS_JSON = data_file("face_clusters.json")  # cluster-id -> {size, reps, span}
VERSION = 1
DET_SIZE = 480        # RetinaFace input; reduced from 640 to lower memory footprint (still detects faces well)
MAX_SIDE = 1600       # cap decode size — enough to keep small faces detectable

_app = None
_load_lock = threading.Lock()
_fwd_lock = threading.Lock()   # serialize the ONNX forward; decode can still overlap


def available() -> bool:
    try:
        import insightface  # noqa: F401
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return True


def _providers() -> list[str]:
    """CoreML first on Apple Silicon, CPU fallback. Override with MUSER_FACES_CPU=1."""
    if os.environ.get("MUSER_FACES_CPU"):
        return ["CPUExecutionProvider"]
    try:
        import onnxruntime as ort
        avail = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]
    out = [p for p in ("CoreMLExecutionProvider", "CUDAExecutionProvider") if p in avail]
    out.append("CPUExecutionProvider")
    return out


def _load_app():
    global _app
    if _app is not None:
        return _app
    with _load_lock:
        if _app is not None:
            return _app
        from insightface.app import FaceAnalysis
        # Only the detection + recognition models are needed (skip age/gender/landmark
        # extras → faster, smaller). buffalo_l weights download once to ~/.insightface.
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                           providers=_providers())
        app.prepare(ctx_id=0, det_size=(DET_SIZE, DET_SIZE))
        _app = app
    return _app


def _load_bgr(path: str):
    """Decode an image to a BGR uint8 array for insightface, downscaled to MAX_SIDE.

    pillow-heif is registered if present so iPhone .heic Takeout files open. Trusted
    local files → PIL bomb guard off. Returns None on any decode failure (file skips)."""
    try:
        from PIL import Image
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
        except Exception:
            pass
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(path)
        img.draft("RGB", (MAX_SIDE, MAX_SIDE))
        img = img.convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE))
        rgb = np.asarray(img)
        return rgb[:, :, ::-1].copy()  # RGB -> BGR (insightface/cv2 convention)
    except Exception:
        return None


def _detect(path: str):
    """Return (meta_list, emb_array). meta = per-face {box,det,w,h}; emb = (n,512) f32.

    `w`/`h` are the FULL image dims (so personalness can compute face-area ratio for
    selfie geometry). Empty lists when no face / unreadable."""
    bgr = _load_bgr(path)
    if bgr is None:
        return None, None  # None signals "unreadable" (distinct from "0 faces" = [])
    H, W = bgr.shape[:2]
    app = _load_app()
    with _fwd_lock:
        faces = app.get(bgr)
    meta, embs = [], []
    for f in faces:
        x1, y1, x2, y2 = (int(v) for v in f.bbox)
        meta.append({
            "box": [x1, y1, x2, y2],
            "det": round(float(f.det_score), 4),
            "fw": x2 - x1, "fh": y2 - y1,   # face box size
        })
        embs.append(np.asarray(f.normed_embedding, dtype=np.float32))
    arr = np.stack(embs) if embs else np.zeros((0, 512), np.float32)
    return {"W": W, "H": H, "faces": meta, "n": len(meta)}, arr


def _face_id(path: str, i: int) -> str:
    return f"{path}#{i}"


def _load_npz() -> dict[str, np.ndarray]:
    if not EMB_NPZ.exists():
        return {}
    try:
        z = np.load(EMB_NPZ, allow_pickle=False)
        # Materialize X and ids ONCE. `z["X"]` on an NpzFile re-reads+decompresses
        # the whole array on every access — doing it inside the comprehension made
        # this O(n) decompressions (~1 TB of work at 33k faces → process hang/kill).
        X = z["X"]
        ids = z["ids"]
        return {fid: X[k] for k, fid in enumerate(ids)}
    except Exception:
        q = EMB_NPZ.parent / f"{EMB_NPZ.stem}.corrupt-{int(time.time())}.npz"
        q.write_bytes(EMB_NPZ.read_bytes())
        raise RuntimeError(f"faces_emb.npz is corrupt — quarantined to {q.name}; "
                           "refusing to continue from an empty embedding store")


def _save_npz(emb: dict[str, np.ndarray]) -> None:
    EMB_NPZ.parent.mkdir(parents=True, exist_ok=True)
    ids = list(emb.keys())
    X = (np.stack([emb[i] for i in ids]) if ids
         else np.zeros((0, 512), np.float32)).astype(np.float16)
    tmp = EMB_NPZ.parent / f".{os.getpid()}-{threading.get_ident()}-{EMB_NPZ.name}.tmp"
    with open(tmp, "wb") as f:  # file handle → np.savez won't re-append ".npz"
        np.savez(f, ids=np.array(ids, dtype=object).astype("U"), X=X)
    os.replace(tmp, EMB_NPZ)


def scan(paths, progress=None, checkpoint_every: int = 500) -> dict:
    """Detect + embed faces for `paths`, incremental by (mtime,size,version).

    Writes light metadata to faces.json and embeddings to faces_emb.npz. Sequential
    forward (the ONNX models aren't thread-safe); image decode dominates and is cheap
    enough that this stays in the ~1-2 h range for ~46k images on CPU/CoreML."""
    if not available():
        return {}
    cache = _SIDE.load()
    emb = _load_npz()
    total, done, since = len(paths), 0, 0

    todo = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            done += 1
            continue
        e = cache.get(p)
        if e and e.get("m") == st.st_mtime_ns and e.get("s") == st.st_size and e.get("v") == VERSION:
            done += 1
            continue
        todo.append((p, st))
    if progress:
        progress(done, total)

    def _flush():
        # NOTE: do NOT clear `emb` here. `_save_npz(emb)` rewrites the whole npz
        # from the dict, so clearing it would truncate the file to only the faces
        # found since the last checkpoint — which silently shrank a 33k-face store
        # to ~130 and wrecked clustering. The dict is tiny (512 f32 per face ≈
        # 2 KB; even 100k faces ≈ 200 MB), so accumulating it is fine; the earlier
        # "26 GB" was misattributed and clearing here fixed nothing while breaking
        # the embeddings.
        _SIDE.save(cache)
        _save_npz(emb)
        _SIDE._primed = False
        _SIDE.prime()

    for p, st in todo:
        meta, arr = _detect(p)
        # drop any stale embeddings from a prior version of this path
        for k in [fid for fid in emb if fid.startswith(p + "#")]:
            del emb[k]
        if meta is None:
            cache[p] = {"m": st.st_mtime_ns, "s": st.st_size, "v": VERSION, "n": 0,
                        "faces": [], "unreadable": True}
        else:
            cache[p] = {"m": st.st_mtime_ns, "s": st.st_size, "v": VERSION, **meta}
            for i in range(arr.shape[0]):
                emb[_face_id(p, i)] = arr[i]
        done += 1
        since += 1
        if since >= checkpoint_every:
            _flush()
            since = 0
        if progress and done % 25 == 0:
            progress(done, total)

    _flush()
    if progress:
        progress(done, total)
    return cache


def cluster(min_cluster_size: int = 4, min_samples: int = 2) -> dict:
    """HDBSCAN the face embeddings into people; back-annotate faces.json and write
    face_clusters.json. Returns {label: {size, reps:[paths], span_days}}.

    ArcFace vectors are L2-normalized, so euclidean clustering == cosine. `reps` are
    the highest-detector-score faces' paths (good thumbnails). `span_days` (capture-time
    spread) needs file mtime as a proxy; a recurring person spans many days."""
    emb = _load_npz()
    if not emb:
        return {}
    import hdbscan
    ids = list(emb.keys())
    X = np.stack([emb[i].astype(np.float32) for i in ids])
    del emb  # 33k small arrays in a dict — real memory; everything below uses X/ids
    # KDTree-based HDBSCAN degenerates at 512-d (boruvka's spatial pruning stops
    # working → memory blows up; the 33k-face pass got OOM-SIGKILLed repeatedly,
    # with empty logs because SIGKILL leaves no traceback). PCA to 64-d first:
    # ArcFace identity structure survives easily, and KDTree is effective again.
    # core_dist_n_jobs bounded for the same reason (default forks per-core workers).
    if X.shape[0] > 10000 and X.shape[1] > 64:
        mu = X.mean(axis=0, keepdims=True)
        X -= mu                                   # center in place — no Xc copy
        # eigendecomposition of the small DxD covariance — cheap and deterministic
        cov = (X.T @ X) / max(1, len(X) - 1)
        evals, evecs = np.linalg.eigh(cov)
        X = X @ evecs[:, -64:]
        del cov, evecs
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                metric="euclidean", core_dist_n_jobs=4)
    labels = clusterer.fit_predict(X)
    del X, clusterer                              # free before the sidecar parse below

    # face-id -> label, and path -> {face_index -> label}
    by_path: dict[str, dict[int, int]] = {}
    for fid, lab in zip(ids, labels):
        path, i = fid.rsplit("#", 1)
        by_path.setdefault(path, {})[int(i)] = int(lab)

    cache = _SIDE.load()
    for path, fmap in by_path.items():
        e = cache.get(path)
        if not e:
            continue
        labs = []
        for i, face in enumerate(e.get("faces", [])):
            face["c"] = fmap.get(i, -1)
            labs.append(face["c"])
        e["clusters"] = sorted({l for l in labs if l >= 0})
    _SIDE.save(cache)
    _SIDE._primed = False
    _SIDE.prime()

    # cluster summaries
    summary: dict[str, dict] = {}
    for fid, lab in zip(ids, labels):
        if lab < 0:
            continue
        path, i = fid.rsplit("#", 1)
        s = summary.setdefault(str(int(lab)), {"size": 0, "_faces": [], "_mtimes": []})
        s["size"] += 1
        det = 0.0
        e = cache.get(path)
        if e and int(i) < len(e.get("faces", [])):
            det = e["faces"][int(i)].get("det", 0.0)
        s["_faces"].append((det, path))
        try:
            s["_mtimes"].append(os.stat(path).st_mtime)
        except OSError:
            pass
    out: dict[str, dict] = {}
    for lab, s in summary.items():
        reps = [p for _, p in sorted(s["_faces"], reverse=True)[:9]]
        span = 0.0
        if len(s["_mtimes"]) >= 2:
            span = (max(s["_mtimes"]) - min(s["_mtimes"])) / 86400.0
        out[lab] = {"size": s["size"], "reps": reps, "span_days": round(span, 1)}
    CLUSTERS_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = CLUSTERS_JSON.with_suffix(".json.tmp")
    import json
    tmp.write_text(json.dumps({"version": VERSION, "min_cluster_size": min_cluster_size,
                               "clusters": out}))
    os.replace(tmp, CLUSTERS_JSON)
    return out


def clusters() -> dict:
    """Parsed face_clusters.json {label: {size, reps, span_days}} or {} if absent."""
    if not CLUSTERS_JSON.exists():
        return {}
    try:
        import json
        return json.loads(CLUSTERS_JSON.read_text()).get("clusters", {})
    except Exception:
        return {}


def members_by_cluster() -> dict[int, list[str]]:
    """{cluster-label: [all member image paths]} in ONE pass over the sidecar —
    cheaper than calling cluster_members() per cluster when you need them all."""
    out: dict[int, list[str]] = {}
    for key, e in _SIDE.entries().items():
        path = key[0] if isinstance(key, tuple) else key
        for c in (e.get("clusters") or []):
            out.setdefault(int(c), []).append(path)
    return out


def _centroids() -> dict[int, "np.ndarray"]:
    """Per-cluster mean ArcFace vector (L2-normalized) — the 'average face' of
    each person-cluster. Maps each npz face to its cluster via the back-annotated
    sidecar `faces[i]['c']`."""
    emb = _load_npz()
    if not emb:
        return {}
    cache = _SIDE.load()
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for fid, v in emb.items():
        path, _, i = fid.rpartition("#")
        try:
            i = int(i)
        except ValueError:
            continue
        e = cache.get(path)
        if not e:
            continue
        fs = e.get("faces", [])
        if i >= len(fs):
            continue
        c = fs[i].get("c", -1)
        if c is None or c < 0:
            continue
        if c not in sums:
            sums[c] = np.zeros(len(v), np.float64)
            counts[c] = 0
        sums[c] += v
        counts[c] += 1
    cents: dict[int, np.ndarray] = {}
    for c, s in sums.items():
        v = s / max(1, counts[c])
        n = np.linalg.norm(v)
        cents[c] = (v / n) if n else v
    return cents


def similar_order(labels: list[int]) -> list[int]:
    """Order cluster labels so visually-similar people sit adjacent — a greedy
    nearest-neighbour chain over centroid cosine, seeded from the first label
    (callers pass size-desc, so it starts at the largest). Makes the same person
    split across clusters easy to spot and merge. Labels without a centroid keep
    their order at the end."""
    cents = _centroids()
    labs = [l for l in labels if l in cents]
    rest = [l for l in labels if l not in cents]
    if len(labs) < 3:
        return labels
    M = np.stack([cents[l] for l in labs])
    sim = M @ M.T
    n = len(labs)
    used = np.zeros(n, bool)
    cur = 0
    used[0] = True
    order = [0]
    for _ in range(n - 1):
        row = sim[cur].copy()
        row[used] = -2.0
        nxt = int(row.argmax())
        used[nxt] = True
        order.append(nxt)
        cur = nxt
    return [labs[i] for i in order] + rest


def cluster_order() -> list[int]:
    """Precomputed similar-adjacent cluster ordering from face_clusters.json
    (written at cluster time). Cheap file read — no embedding load. [] if absent."""
    if not CLUSTERS_JSON.exists():
        return []
    try:
        import json
        return [int(x) for x in json.loads(CLUSTERS_JSON.read_text()).get("order", [])]
    except Exception:
        return []


def cluster_members(label: int) -> list[str]:
    """All indexed paths whose faces include people-cluster `label` — the FULL
    membership, not just the ≤9 thumbnail `reps` stored in face_clusters.json.
    Reads the back-annotated `clusters` list each image carries post-`cluster()`."""
    out = []
    # entries() is keyed by (path, mtime_ns, size) tuples; unpack the path.
    for key, e in _SIDE.entries().items():
        path = key[0] if isinstance(key, tuple) else key
        if label in (e.get("clusters") or []):
            out.append(path)
    return out


def prime_cache_from_sidecar() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    """RAM-only per-image face metadata if the sidecar knows it (mtime+size match)."""
    return _SIDE.lookup(path)


def cache_exists() -> bool:
    return _SIDE.exists()


def sidecar() -> Sidecar:
    return _SIDE
