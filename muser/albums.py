"""Album facet — group outpaintings by their album-art region, flag blurred ones.

Outpaintings are square album covers extended to ~6:1 panoramas. The original cover
sits at a FIXED offset-right box — empirically derived from cross-variant pixel
variance and confirmed by the ComfyUI outpaint pad (``left=3000, right=2000`` → the
cover is pushed right of center). Everything outside the box is AI-hallucinated wings.

Cropping to that box and embedding it (SigLIP, region only) lets us:
  - **group** variants of the same cover by region cosine (dedup / stack — catches
    re-rolls that whole-image dedup misses, since the wings differ), and
  - measure **blur inside the cover region** to drop blurred re-rolls.

Feeds the outpaintings instance's unique-spread (one representative per album) and
click-to-cluster (a cover's variants, re-ranked). Region masking is what the earlier
filename-number grouping lacked — it fixes the false positives.

Persistence (under ``MUSER_HOME``):
  ``album_vecs.npz``    — path-aligned region embeddings + region-blur (incremental)
  ``album_groups.json`` — grouping result: ``{by_path:{group,blur,removed}, groups:[…]}``

Derivation of the mask + threshold + blur cutoff is recorded in
``docs/ephemeral-design-interfaces.md`` / ``docs/variance-maps.md``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .paths import MUSER_HOME

# Fixed album-art region (fraction of full W×H) — offset-right square, derived + pad-confirmed.
MASK = (0.52, 0.64, 0.13, 0.74)          # (x0, x1, y0, y1)
THRESHOLD = 0.92                          # region-cosine merge threshold (same-album p10≈0.87, diff p90≈0.71)
BLUR_CUTOFF = 5.0                         # region Laplacian variance below this = blurred cover → removed
DEFAULT_MODEL = "siglip2-b"

VECS = MUSER_HOME / "album_vecs.npz"
GROUPS = MUSER_HOME / "album_groups.json"
_CACHE = {"mtime": -1.0, "data": None}


def available() -> bool:
    return GROUPS.exists()


def crop_region(im):
    W, H = im.size
    x0, x1, y0, y1 = MASK
    return im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))


# ---- data access -----------------------------------------------------------------

def _groups() -> dict:
    """Cached album_groups.json ({by_path, groups, ...}), reloaded on mtime change."""
    try:
        mt = GROUPS.stat().st_mtime
    except OSError:
        return {}
    if mt != _CACHE["mtime"]:
        try:
            _CACHE["data"] = json.loads(GROUPS.read_text())
            _CACHE["mtime"] = mt
        except Exception:
            return {}
    return _CACHE["data"] or {}


def lookup(path: str):
    """Per-path album entry ``{group, blur, removed, count}`` or None."""
    g = _groups()
    e = g.get("by_path", {}).get(path)
    if e is None:
        return None
    return {**e, "count": g.get("group_size", {}).get(str(e["group"]), 1)}


def members(group_id) -> list:
    """All paths in an album group (the rep first)."""
    g = _groups()
    gid = int(group_id)
    ms = [p for p, e in g.get("by_path", {}).items() if e["group"] == gid]
    reps = {gr["rep"]: gr for gr in g.get("groups", [])}
    rep = next((gr["rep"] for gr in g.get("groups", []) if gr["id"] == gid), None)
    if rep in ms:
        ms.remove(rep); ms.insert(0, rep)
    return ms


def group_of(path: str):
    e = _groups().get("by_path", {}).get(path)
    return e["group"] if e else None


def prime() -> int:
    """Warm the RAM cache at service startup. Returns entry count."""
    return len(_groups().get("by_path", {}))


# ---- build -----------------------------------------------------------------------

def _load_vecs() -> dict:
    """{path: (vec, blur)} from album_vecs.npz, if present."""
    if not VECS.exists():
        return {}
    z = np.load(VECS, allow_pickle=True)
    paths = [str(p) for p in z["paths"]]
    vecs = z["vecs"].astype(np.float32); blur = z["blur"].astype(np.float32)
    return {paths[i]: (vecs[i], float(blur[i])) for i in range(len(paths))}


def _embed_new(paths, model, on_progress=print):
    """Crop→embed the album region + measure region blur for `paths`. Returns
    {path: (l2-normed vec, blur)}. Embedder-agnostic: crops to temp jpegs, uses
    the standard embed_images (SigLIP has no PIL-encode entrypoint)."""
    import tempfile
    from PIL import Image
    import cv2
    from .registry import load_model
    from .embedders import _load_rgb

    emb = load_model(model)
    tmp = tempfile.mkdtemp(prefix="album_crops_")
    idx, blur, crop_paths = [], {}, []
    for i, p in enumerate(paths):
        try:
            r = crop_region(_load_rgb(p))
            g = cv2.cvtColor(np.asarray(r.resize((256, 256))), cv2.COLOR_RGB2GRAY)
            blur[p] = float(cv2.Laplacian(g, cv2.CV_64F).var())
            cp = os.path.join(tmp, f"{i:06d}.jpg"); r.convert("RGB").save(cp, "JPEG", quality=92)
            crop_paths.append(cp); idx.append(p)
        except Exception:
            pass
        if i % 500 == 0:
            on_progress(f"  crop {i}/{len(paths)}")
    out = {}
    if crop_paths:
        V = emb.embed_images(crop_paths)                 # L2-normalized
        for k, p in enumerate(idx):
            out[p] = (V[k].astype(np.float32), blur[p])
    return out


def _aesthetic(path: str, scores: dict) -> float:
    s = scores.get(path)
    if not s:
        return -1.0
    return float(s.get("aesthetic_v2", s.get("aesthetic", 0.0)))


def build(model: str = DEFAULT_MODEL, on_progress=print) -> dict:
    """Embed any new region crops (incremental), cluster by region cosine excluding
    blurred, pick the best-aesthetic rep per group, persist album_groups.json."""
    from .index import MuserIndex
    t = MuserIndex()._open(model)
    if t is None:
        return {"built": False, "message": "empty index"}
    paths = [r["path"] for r in t.search().select(["path"]).limit(10**7).to_list()]

    have = _load_vecs()
    new = [p for p in paths if p not in have]
    if new:
        on_progress(f"embedding {len(new)} new album regions…")
        have.update(_embed_new(new, model, on_progress))
    # keep only live paths; persist the npz
    have = {p: have[p] for p in paths if p in have}
    allp = list(have)
    V = np.stack([have[p][0] for p in allp]).astype(np.float32)
    blur = np.array([have[p][1] for p in allp], np.float32)
    np.savez(VECS, paths=np.array(allp), vecs=V, blur=blur)

    # blurred = region-blur below cutoff OR in a hand-sorted /blurred/ folder
    removed = np.array([blur[i] < BLUR_CUTOFF or "/blurred/" in allp[i].lower()
                        for i in range(len(allp))])
    keep = np.where(~removed)[0]

    # cluster the kept regions: union-find on cosine ≥ THRESHOLD
    Vk = V[keep]; S = Vk @ Vk.T; np.fill_diagonal(S, 0.0)
    parent = list(range(len(keep)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a in range(len(keep)):
        for b in np.where(S[a] >= THRESHOLD)[0]:
            ra, rb = find(a), find(int(b))
            if ra != rb:
                parent[ra] = rb
    import collections
    raw = collections.defaultdict(list)
    for a in range(len(keep)):
        raw[find(a)].append(int(keep[a]))

    # rep = highest-aesthetic member
    try:
        scores = json.loads((MUSER_HOME / "scores.json").read_text()).get("scores", {})
    except Exception:
        scores = {}
    groups, by_path, group_size = [], {}, {}
    for gid, (_, idxs) in enumerate(sorted(raw.items(), key=lambda kv: -len(kv[1]))):
        mem = [allp[i] for i in idxs]
        rep = max(mem, key=lambda p: (_aesthetic(p, scores), blur[allp.index(p)]))
        groups.append({"id": gid, "rep": rep, "count": len(mem),
                       "aesthetic": round(_aesthetic(rep, scores), 4)})
        group_size[str(gid)] = len(mem)
        for i in idxs:
            by_path[allp[i]] = {"group": gid, "blur": round(float(blur[i]), 1), "removed": False}
    # removed paths get no group (group -1), still recorded so the UI can drop them
    for i in np.where(removed)[0]:
        by_path[allp[i]] = {"group": -1, "blur": round(float(blur[i]), 1), "removed": True}

    out = {"version": 1, "mask": list(MASK), "threshold": THRESHOLD, "blur_cutoff": BLUR_CUTOFF,
           "n_groups": len(groups), "n_removed": int(removed.sum()),
           "groups": groups, "group_size": group_size, "by_path": by_path}
    tmp = GROUPS.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out)); os.replace(tmp, GROUPS)
    _CACHE["mtime"] = -1.0
    multi = sum(1 for g in groups if g["count"] > 1)
    msg = (f"{len(groups)} album groups ({multi} multi-variant), "
           f"{int(removed.sum())} blurred removed, from {len(allp)} images")
    on_progress(msg)
    return {"built": True, "n_groups": len(groups), "n_removed": int(removed.sum()),
            "multi": multi, "message": msg}
