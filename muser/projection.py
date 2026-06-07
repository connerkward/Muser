"""3D PCA projection of indexed embeddings for the Explore tab's point-cloud viz.

Pulls a sample of (path, vector) pairs from the model's LanceDB table, projects
them to 3D via a numpy SVD PCA (no sklearn — keeps the warm path dep-light),
colors each point by its HDBSCAN cluster (from ~/.muser/clusters.json), and
returns a compact JSON payload. The result is expensive to compute (a few
hundred ms for a 2000×1024 SVD plus the table scan), so callers cache it keyed
by (n, folder, index-mtime, clusters-mtime).

Pure functions here; the service owns the cache and the hide-filter. The only
state is the deterministic palette, which is a pure function of cluster id.
"""

from __future__ import annotations

import colorsys
import os
from pathlib import Path

import numpy as np

from .index import MuserIndex, uid_for

# Sampling cap — a 5000×1024 SVD is ~0.5 s and the payload (~5000 points,
# 4 floats each) stays well under a megabyte. Larger samples don't add legibility
# to a point cloud, so we hard-cap rather than let a caller request the full 27k.
MAX_N = 5000


def _palette_color(cid: int) -> str:
    """Deterministic, well-separated #rrggbb for a cluster id. Pure function of
    the id (no global palette table to keep in sync), so the same cluster always
    gets the same color across requests and across page reloads.

    The hue is spread by the golden-angle (0.618…) so consecutive ids land far
    apart on the color wheel instead of in adjacent buckets — the classic trick
    for generating an unbounded set of visually distinct colors. id == -1
    (unclustered) is rendered a neutral mid-gray so it recedes behind the
    real clusters.
    """
    if cid < 0:
        return "#8a8a8a"
    h = (cid * 0.6180339887498949) % 1.0
    # Vary saturation/lightness slightly by id too so two ids that happen to land
    # on near-identical hues still differ. Kept in a high-chroma, mid-light band
    # so points stay vivid on the dark Explore canvas.
    s = 0.62 + 0.18 * ((cid * 7) % 5) / 4.0
    l = 0.55 + 0.10 * ((cid * 3) % 3) / 2.0
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"


def _path_to_cluster(
    clusters: dict | None, method: str | None = None
) -> tuple[dict[str, int], dict[int, dict]]:
    """From a loaded clusters.json, build:
      - path -> cluster_id  (from the chosen method's ``members`` map)
      - cluster_id -> {id, label, color}  (from ``clusters``)
    Honors ``method`` ("hdbscan" | "kmeans") so the point-cloud coloring tracks
    the Explore method picker. Falls back to HDBSCAN, then the first available
    method. Empty maps when clusters.json is missing/malformed (every point then
    becomes id -1).
    """
    p2c: dict[str, int] = {}
    meta: dict[int, dict] = {}
    if not clusters:
        return p2c, meta
    methods = clusters.get("methods", {})
    if method and method in methods:
        m_name = method
    else:
        m_name = "hdbscan" if "hdbscan" in methods else (next(iter(methods)) if methods else None)
    if not m_name:
        return p2c, meta
    m = methods[m_name]
    for cl in m.get("clusters", []):
        cid = int(cl["id"])
        meta[cid] = {
            "id": cid,
            "label": cl.get("label") or ("unclustered" if cid < 0 else f"cluster {cid}"),
            "color": _palette_color(cid),
        }
    for cid_str, paths in m.get("members", {}).items():
        cid = int(cid_str)
        for p in paths:
            p2c[p] = cid
    return p2c, meta


def _sample_rows(
    index: MuserIndex,
    model: str,
    n: int,
    folder: str | None,
    is_hidden,
) -> list[tuple[str, np.ndarray]]:
    """Read up to ``n`` (path, vector) pairs from the model's table, skipping
    hidden paths. If the (post-hide) population is larger than ``n``, take an
    *even* stride sample rather than the first ``n`` — a contiguous prefix would
    bias toward whatever insertion order LanceDB happens to have (often one
    folder). Vectors are L2-normalized at index time, so no renorm here.
    """
    t = index._open(model)
    if t is None:
        return []
    q = t.search().select(["path", "vector"])
    if folder:
        where = MuserIndex._folder_where(folder)
        if where:
            q = q.where(where, prefilter=True)
    rows = q.limit(100_000_000).to_list()
    # Drop hidden (dead / nsfw / hidden-cluster) before sampling so the returned
    # count is honest and a hidden point never reaches the viz.
    live = [r for r in rows if not is_hidden(r["path"])]
    if len(live) > n:
        # Evenly-spaced stride sample (deterministic — same sample for a given
        # population, so the cache key stays meaningful).
        idxs = np.linspace(0, len(live) - 1, num=n).round().astype(int)
        live = [live[i] for i in idxs]
    return [(r["path"], np.asarray(r["vector"], dtype=np.float32)) for r in live]


def _pca3(X: np.ndarray) -> np.ndarray:
    """3D PCA via numpy SVD on the mean-centered matrix, normalized so each axis
    spans roughly [-1, 1]. Returns an (N, 3) float array. Handles the degenerate
    cases (fewer than 3 rows, or a corpus that collapses to <3 dims) by
    zero-padding the missing components.
    """
    if X.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    Xc = X - X.mean(axis=0, keepdims=True)
    # full_matrices=False → Vt is (min(N,D), D); first 3 rows are the top PCs.
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:3]
    coords = Xc @ comps.T  # (N, k<=3)
    if coords.shape[1] < 3:  # corpus spans <3 dims — pad
        coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
    # Per-axis normalize to [-1, 1] by the max abs magnitude (preserves the
    # origin-centering; a min-max would shift the cloud off-center).
    scale = np.abs(coords).max(axis=0)
    scale[scale == 0] = 1.0
    return (coords / scale).astype(np.float32)


def compute_projection(
    index: MuserIndex,
    model: str,
    clusters: dict | None,
    n: int,
    folder: str | None,
    is_hidden,
    total_indexed: int,
    method: str | None = None,
) -> dict:
    """Build the projection payload. ``is_hidden(path) -> bool`` is the service's
    unified hide predicate; ``total_indexed`` is the model's full row count.
    ``method`` ("hdbscan" | "kmeans") selects which clustering colors the cloud.

    Response shape::

        {"points":   [{"x","y","z","c","uid"}, ...],   # c = cluster id
         "clusters": [{"id","label","color","count"}, ...],
         "total_indexed": int, "sampled": int}
    """
    n = max(1, min(int(n), MAX_N))
    p2c, meta = _path_to_cluster(clusters, method)

    sample = _sample_rows(index, model, n, folder, is_hidden)
    if not sample:
        return {"points": [], "clusters": [], "total_indexed": total_indexed, "sampled": 0}

    paths = [p for p, _ in sample]
    X = np.stack([v for _, v in sample])
    coords = _pca3(X)

    points = []
    counts: dict[int, int] = {}
    for (path, _), (x, y, z) in zip(sample, coords):
        cid = p2c.get(path, -1)
        counts[cid] = counts.get(cid, 0) + 1
        points.append(
            {
                "x": round(float(x), 3),
                "y": round(float(y), 3),
                "z": round(float(z), 3),
                "c": cid,
                "uid": uid_for(path),
            }
        )

    # Only emit cluster legend entries that actually appear in the sample, each
    # with its sampled count. Unknown ids (a member of a cluster missing from the
    # `clusters` list, or the -1 fallback) get a synthesized entry.
    cluster_legend = []
    for cid, cnt in sorted(counts.items(), key=lambda kv: (kv[0] < 0, -kv[1], kv[0])):
        info = meta.get(cid) or {
            "id": cid,
            "label": "unclustered" if cid < 0 else f"cluster {cid}",
            "color": _palette_color(cid),
        }
        cluster_legend.append({**info, "count": cnt})

    return {
        "points": points,
        "clusters": cluster_legend,
        "total_indexed": total_indexed,
        "sampled": len(points),
    }


def cache_key(
    n: int,
    folder: str | None,
    db_path: Path,
    clusters_path: Path,
    space: str = "",
    method: str = "",
) -> tuple:
    """Cache key = (clamped n, normalized folder, index-table mtime, clusters
    mtime, space, method). Any file changing (re-index or re-cluster) or a
    space/method toggle invalidates the entry. Index mtime uses the LanceDB
    dir's mtime as a cheap proxy for "table changed" (a new fragment/version
    file bumps the directory mtime).
    """
    def _mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:
            return -1.0

    n = max(1, min(int(n), MAX_N))
    folder_norm = str(Path(folder).expanduser().resolve()) if folder else ""
    return (n, folder_norm, _mtime(db_path), _mtime(clusters_path), space or "", method or "")


def compute_projection_dinov2(n: int, is_hidden) -> dict:
    """DINOv2-space point cloud: PCA-3 of a sample of the precomputed DINOv2
    vectors (``~/.muser/dinov3_canon.npz``), colored by the DINOv2 clustering
    (``dinov3_clusters.json``).

    That sidecar only stores a handful of *representative* basenames per cluster
    (not full membership), so points are colored by nearest-representative in
    DINOv2 space (1-NN to the union of cluster reps) — an approximation that
    nonetheless makes the cloud track the DINOv2 clusters + their labels. Folder
    scoping doesn't apply (the npz is the whole canonical set). Degrades to an
    empty cloud when the npz is absent.
    """
    from . import dinov2 as _dino

    store = _dino._load()
    if store is None:
        return {"points": [], "clusters": [], "total_indexed": 0, "sampled": 0}
    paths_all = store["paths"]
    X_all = store["X"]  # already L2-normalized
    total = len(paths_all)

    # Live (non-hidden) rows, then an even stride sample to n.
    live_idx = [i for i in range(total) if not is_hidden(str(paths_all[i]))]
    n = max(1, min(int(n), MAX_N))
    if len(live_idx) > n:
        sel = np.linspace(0, len(live_idx) - 1, num=n).round().astype(int)
        live_idx = [live_idx[i] for i in sel]
    if not live_idx:
        return {"points": [], "clusters": [], "total_indexed": total, "sampled": 0}
    sample_paths = [str(paths_all[i]) for i in live_idx]
    Xs = X_all[live_idx]
    coords = _pca3(Xs)

    # Build cluster meta (id -> label/color) + a reference matrix of cluster reps
    # for 1-NN coloring.
    data = _dino.clusters() or {}
    byname = _dino._basename_index()
    meta: dict[int, dict] = {}
    ref_vecs: list[np.ndarray] = []
    ref_cids: list[int] = []
    row_of = store["idx"]
    for cid_str, cl in (data.get("clusters", {}) or {}).items():
        try:
            cid = int(cid_str)
        except (TypeError, ValueError):
            continue
        if cid < 0:
            continue
        label = (cl.get("label") or "").strip() or f"cluster {cid}"
        meta[cid] = {"id": cid, "label": label, "color": _palette_color(cid)}
        for bn in cl.get("sample", []):
            p = byname.get(bn)
            r = row_of.get(p) if p else None
            if r is not None:
                ref_vecs.append(X_all[r])
                ref_cids.append(cid)

    if ref_vecs:
        R = np.stack(ref_vecs)              # (M, D) unit vectors
        nn = np.argmax(Xs @ R.T, axis=1)    # nearest rep per sampled point (cosine)
        point_cids = [ref_cids[j] for j in nn]
    else:
        point_cids = [-1] * len(sample_paths)

    points = []
    counts: dict[int, int] = {}
    for path, (x, y, z), cid in zip(sample_paths, coords, point_cids):
        counts[cid] = counts.get(cid, 0) + 1
        points.append({
            "x": round(float(x), 3), "y": round(float(y), 3), "z": round(float(z), 3),
            "c": int(cid), "uid": uid_for(path),
        })

    cluster_legend = []
    for cid, cnt in sorted(counts.items(), key=lambda kv: (kv[0] < 0, -kv[1], kv[0])):
        info = meta.get(cid) or {
            "id": cid, "label": "unclustered" if cid < 0 else f"cluster {cid}",
            "color": _palette_color(cid),
        }
        cluster_legend.append({**info, "count": cnt})

    return {
        "points": points, "clusters": cluster_legend,
        "total_indexed": total, "sampled": len(points),
    }
