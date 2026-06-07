"""DINOv2 — an OPTIONAL, additive embedding space for Explore + Similar.

Reuses *already-computed* DINOv2-large vectors (no model load, no new
embeddings). Two sidecars, both written offline:

  ~/.muser/dinov3_canon.npz      paths (U…, N) + X (float16, N×1024)
  ~/.muser/dinov3_clusters.json  {model, reducer, n_clusters, noise,
                                  clusters: {id: {size, sample:[basenames]}}}

(The file is named ``dinov3_*`` for historical reasons but the vectors are
DINOv2-large — the UI labels it "DINOv2".)

Everything here is pure-numpy and lazy: the npz is mmap-loaded + L2-normalized
once on first use and cached for the process. When either sidecar is missing the
public helpers degrade to ``None`` / empty so callers transparently fall back to
the default (SigLIP) space.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

NPZ = Path.home() / ".muser" / "dinov3_canon.npz"
CLUSTERS = Path.home() / ".muser" / "dinov3_clusters.json"

_lock = threading.Lock()
_store: dict | None = None  # {"paths": np.ndarray[str], "X": float32 N×D (L2-normed), "idx": {path: row}}


def available() -> bool:
    """True when the npz sidecar is present (DINOv2 Similar is usable)."""
    return NPZ.exists()


def clusters_available() -> bool:
    return CLUSTERS.exists()


def _load():
    """Lazy-load + L2-normalize the npz once; cache for the process.

    Returns the cache dict or None when the npz is absent / unreadable.
    """
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is not None:
            return _store
        if not NPZ.exists():
            return None
        try:
            import numpy as np

            d = np.load(NPZ, allow_pickle=True)
            paths = d["paths"].astype(str)
            X = d["X"].astype(np.float32)  # float16 → float32 for stable math
            n = np.linalg.norm(X, axis=1, keepdims=True)
            n[n == 0] = 1.0
            X = X / n
            idx = {p: i for i, p in enumerate(paths)}
            _store = {"paths": paths, "X": X, "idx": idx}
        except Exception:
            _store = None
        return _store


def similar(path: str, k: int = 24):
    """Cosine kNN of an indexed `path` in DINOv2 space.

    Returns a list of ``(path, score)`` (score = cosine similarity, same
    convention as ``MuserIndex.search``), excluding `path` itself. Returns
    ``None`` when DINOv2 is unavailable OR `path` has no stored vector — the
    caller then falls back to the default (SigLIP) space.
    """
    store = _load()
    if store is None:
        return None
    i = store["idx"].get(path)
    if i is None:
        return None
    import numpy as np

    X = store["X"]
    sims = X @ X[i]  # cosine (rows are unit vectors)
    sims[i] = -2.0   # exclude self
    k = min(int(k), sims.shape[0] - 1)
    if k <= 0:
        return []
    top = np.argpartition(-sims, k)[:k]
    top = top[np.argsort(-sims[top])]
    paths = store["paths"]
    return [(str(paths[j]), float(sims[j])) for j in top]


_clusters_cache: dict = {"mtime": -1.0, "data": None}


def clusters() -> dict | None:
    """Parsed ~/.muser/dinov3_clusters.json (mtime-cached) or None if absent."""
    try:
        mt = CLUSTERS.stat().st_mtime
    except OSError:
        return None
    if mt == _clusters_cache["mtime"]:
        return _clusters_cache["data"]
    try:
        data = json.loads(CLUSTERS.read_text())
    except Exception:
        data = None
    _clusters_cache["mtime"] = mt
    _clusters_cache["data"] = data
    return data


def _basename_index() -> dict[str, str]:
    """Map basename → full path from the npz, so the clusters sidecar's
    ``sample`` basenames can be resolved to real on-disk paths for thumbnails."""
    store = _load()
    if store is None:
        return {}
    cache = store.get("_byname")
    if cache is None:
        cache = {}
        for p in store["paths"]:
            p = str(p)
            cache.setdefault(os.path.basename(p), p)  # first wins on collision
        store["_byname"] = cache
    return cache


def cluster_members(cluster_id: int) -> list[str]:
    """Full paths of a cluster's sample images (basenames resolved via npz).

    The clusters sidecar only stores a handful of representative *basenames*
    per cluster (not full member lists), so this returns just those samples
    mapped back to paths. Unresolved basenames are dropped.
    """
    data = clusters()
    if not data:
        return []
    cl = data.get("clusters", {}).get(str(cluster_id))
    if not cl:
        return []
    byname = _basename_index()
    out = []
    for bn in cl.get("sample", []):
        p = byname.get(bn)
        if p:
            out.append(p)
    return out
