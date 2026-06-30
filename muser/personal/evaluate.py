"""Human evaluation harness — the user labels a diverse sample; we score the model.

This is the gold-standard check (verify-outputs-rule): ground truth from the *user*,
independent of every signal the model uses and of the VLM judge. A diverse sample
(balanced across the three buckets, spread across albums so it isn't one folder) is
served to an interactive page; the user's verdicts persist to eval_labels.json and we
report model-vs-human accuracy + a confusion matrix + per-bucket precision.
"""
from __future__ import annotations

import json
import os
import random
import threading

from ..paths import data_file
from . import personalness

BUCKETS = ("personal", "in_between", "reference")
LABELS_FILE = data_file("eval_labels.json")


def _album(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def sample(n: int = 60, seed: int = 0) -> list[dict]:
    """A diverse sample: balanced across predicted buckets, capped per album.
    Already-labeled / already-flagged images are excluded — the point of a fresh
    batch is NEW ground truth, and the client-side dedupe only lived per-session
    (reloads kept re-serving tagged images)."""
    labels = _load_labels()
    done = {p for p, e in labels.items() if e.get("label") or e.get("flag")}
    entries = personalness.all_entries()
    items = [(p, e) for p, e in entries.items() if p not in done]
    rng = random.Random(seed)
    rng.shuffle(items)
    target = max(1, n // 3)
    cap = max(2, n // 10)             # at most ~10% of the sample from any one album
    chosen: list[tuple[str, dict]] = []
    per_bucket = {b: 0 for b in BUCKETS}
    album_n: dict[str, int] = {}
    # pass 1 — balanced + album-diverse
    for path, e in items:
        b = e.get("bucket")
        if b not in BUCKETS or per_bucket[b] >= target:
            continue
        a = _album(path)
        if album_n.get(a, 0) >= cap:
            continue
        chosen.append((path, e)); per_bucket[b] += 1; album_n[a] = album_n.get(a, 0) + 1
        if len(chosen) >= n:
            break
    # pass 2 — top up to n if album caps left us short
    if len(chosen) < n:
        have = {p for p, _ in chosen}
        for path, e in items:
            if path in have or e.get("bucket") not in BUCKETS:
                continue
            chosen.append((path, e)); have.add(path)
            if len(chosen) >= n:
                break
    rng.shuffle(chosen)
    return [{"path": p, "bucket": e.get("bucket"), "p": e.get("p"), "unc": e.get("unc"),
             "sig": e.get("sig"), "album": _album(p),
             "label": labels.get(p, {}).get("label"),
             "flag": labels.get(p, {}).get("flag")}
            for p, e in chosen]


# Disposition flags — orthogonal to the personal/reference bucket. A "delete" image is
# a delete-candidate (junk / blurry / unwanted); "depri" = keep but rank it down;
# "keep" = explicitly SAVED from deletion on the Cleanup page (reviewed, not junk) —
# permanently excluded from the delete-candidate list so it never resurfaces there.
FLAGS = ("depri", "delete", "keep")


# Serializes every read-modify-write of eval_labels.json within the service
# process — concurrent label/flag requests otherwise race on the shared tmp file
# (both write .json.tmp; the loser's os.replace hits FileNotFoundError).
_IO_LOCK = threading.Lock()


def _load_labels() -> dict:
    """Load labels. A MISSING file is a normal first run → {}. A CORRUPT file is
    NOT — silently returning {} here is what destroyed ~1,800 labels on
    2026-06-12 (a tmp-file write race promoted corrupt JSON; the next click then
    'saved' a file containing only itself). Quarantine the corrupt file and fail
    loudly instead, so no later save can clobber from an empty base."""
    if not LABELS_FILE.exists():
        return {}
    raw = LABELS_FILE.read_text()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        import time as _t
        q = LABELS_FILE.with_suffix(f".corrupt-{int(_t.time())}.json")
        q.write_text(raw)
        raise RuntimeError(f"eval_labels.json is corrupt — quarantined to {q.name}; "
                           "refusing to continue from an empty label set")


def _save(labels: dict) -> None:
    # Anti-clobber: a save that would shrink the on-disk set by >50 entries AND
    # >50% is almost certainly a bug, not the user un-labeling — keep a copy.
    try:
        old = json.loads(LABELS_FILE.read_text())
        if len(old) - len(labels) > 50 and len(labels) < len(old) / 2:
            import time as _t
            _snap = LABELS_FILE.with_suffix(f".pre-shrink-{int(_t.time())}.json")
            _stmp = _snap.parent / (_snap.name + ".tmp")
            _stmp.write_text(json.dumps(old))
            os.replace(_stmp, _snap)
    except Exception:
        pass
    tmp = LABELS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(labels))
    os.replace(tmp, LABELS_FILE)
    _maybe_backup()


_BAK_STATE = {"last": 0.0}


def _maybe_backup() -> None:
    """Rolling backup of the labels file, at most every 5 minutes — cheap
    insurance: label work is hours of human time in a single JSON."""
    import time as _t
    now = _t.time()
    if now - _BAK_STATE["last"] < 300:
        return
    _BAK_STATE["last"] = now
    try:
        bak = LABELS_FILE.with_suffix(".bak.json")
        bak_tmp = bak.parent / (bak.name + ".tmp")
        bak_tmp.write_text(LABELS_FILE.read_text())
        os.replace(bak_tmp, bak)
    except Exception:
        pass


def label(path: str, verdict: str | None, model_bucket: str | None = None) -> None:
    """Persist (or clear, if verdict is None) one human bucket label; keeps any flag."""
    with _IO_LOCK:
        labels = _load_labels()
        e = labels.get(path, {})
        if verdict is None:
            e.pop("label", None); e.pop("model", None)
        else:
            e["label"] = verdict; e["model"] = model_bucket
        if e:
            labels[path] = e
        else:
            labels.pop(path, None)
        _save(labels)


def label_bulk(paths, verdict: str, model_bucket: str | None = None) -> int:
    """Persist one bucket label on many paths in ONE load-modify-save (lock-held).
    Used by Triage's mark-all-shown — per-path label() would rewrite the file N
    times (see human-labeled-data rule)."""
    with _IO_LOCK:
        labels = _load_labels()
        for path in paths:
            e = labels.get(path, {})
            e["label"] = verdict
            if model_bucket is not None:
                e["model"] = model_bucket
            labels[path] = e
        _save(labels)
    return len(paths)


def set_flag(path: str, flag: str | None) -> None:
    """Set/clear the disposition flag ('depri' | 'delete' | None); keeps any bucket label."""
    set_flags_bulk([path], flag)


def set_flags_bulk(paths, flag: str | None) -> int:
    """Set/clear one disposition flag on many paths in ONE load-modify-save,
    under the write lock. The per-path version did a full read→tmp→os.replace
    cycle each call; two concurrent requests raced on the same tmp file and one
    500'd with FileNotFoundError (and N paths cost N rewrites)."""
    with _IO_LOCK:
        labels = _load_labels()
        for path in paths:
            e = labels.get(path, {})
            if flag in FLAGS:
                e["flag"] = flag
            else:
                e.pop("flag", None)
            if e:
                labels[path] = e
            else:
                labels.pop(path, None)
        _save(labels)
    return len(paths)


def flagged() -> dict:
    """Lists of paths the user flagged, by disposition — actionable (export/trash later)."""
    labels = _load_labels()
    out = {f: [] for f in FLAGS}
    for p, e in labels.items():
        if e.get("flag") in FLAGS:
            out[e["flag"]].append(p)
    return {**out, "counts": {f: len(out[f]) for f in FLAGS}}


# ---- export human-tagged images to the MAIN (aesthetic) library ----------------
# Human "reference" = saved inspiration, not my life → it belongs in the main
# corpus and NOT in personal: the file is MOVED (personal index entry goes dead
# and is hidden/purged automatically). Human "in_between" = a personal photo
# that's also aesthetically strong → it lives in BOTH: the file is COPIED.
# A manifest records every src → {dst, mode} so re-runs are idempotent and a
# move is reversible (move the file back, delete the manifest entry).
EXPORT_ROOT = os.path.expanduser("~/ideas-syncthing/from-personal")
EXPORT_MANIFEST = data_file("export_manifest.json")
_EXPORT_MODE = {"reference": "move", "in_between": "copy"}


def _load_manifest() -> dict:
    try:
        return json.loads(EXPORT_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def export_status() -> dict:
    """Counts of human reference/in_between labels not yet exported."""
    labels = _load_labels()
    manifest = _load_manifest()
    pending = {b: [] for b in _EXPORT_MODE}
    for p, e in labels.items():
        b = e.get("label")
        if b in _EXPORT_MODE and p not in manifest and os.path.exists(p):
            pending[b].append(p)
    return {"pending": {b: len(v) for b, v in pending.items()},
            "pending_paths": pending, "exported": len(manifest), "dest": EXPORT_ROOT}


REF_VECTORS_NPZ = data_file("exported_ref_vectors.npz")


def _snapshot_ref_vectors(paths) -> int:
    """Persist the SigLIP embeddings of about-to-be-MOVED reference images.
    Moving a file eventually purges its row from the personal index — which
    would silently drop the example from the personal-vs-reference training set,
    draining the reference class as the user curates. The npz keeps every
    exported reference example trainable forever (path → 1152-d fp16)."""
    if not paths:
        return 0
    import numpy as np

    from ..index import MuserIndex
    try:
        import lancedb
        db = lancedb.connect(MuserIndex().db_path)
        t = db.open_table("img__siglip2_b").to_arrow()
        tp = t.column("path").to_pylist()
        want = set(paths)
        idx = [i for i, p in enumerate(tp) if p in want]
        if not idx:
            return 0
        tv = t.column("vector")
        new = {tp[i]: np.asarray(tv[i].as_py(), np.float16) for i in idx}
    except Exception:
        return 0
    # merge with any existing snapshot (union by path, new wins)
    merged = {}
    if REF_VECTORS_NPZ.exists():
        try:
            z = np.load(REF_VECTORS_NPZ, allow_pickle=False)
            merged = {p: z["X"][i] for i, p in enumerate(z["ids"])}
        except Exception:
            merged = {}
    merged.update(new)
    ids = list(merged.keys())
    X = np.stack([merged[p] for p in ids]).astype(np.float16)
    tmp = REF_VECTORS_NPZ.with_suffix(".npz.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, ids=np.array(ids, dtype=object).astype("U"), X=X)
    os.replace(tmp, REF_VECTORS_NPZ)
    return len(new)


def export_to_main() -> dict:
    """MOVE human-'reference' / COPY human-'in_between' images into the main
    library tree (EXPORT_ROOT/<bucket>/). Collision-safe, manifest-tracked.
    The caller asks the main instance to index EXPORT_ROOT afterwards."""
    import hashlib
    import shutil
    st = export_status()
    # Snapshot embeddings of reference images BEFORE moving them — keeps the
    # exported examples in the training set after the index purges their rows.
    _snapshot_ref_vectors(st["pending_paths"].get("reference", []))
    manifest = _load_manifest()
    done = {"reference": 0, "in_between": 0}
    for bucket, paths in st["pending_paths"].items():
        mode = _EXPORT_MODE[bucket]
        dest_dir = os.path.join(EXPORT_ROOT, bucket.replace("_", "-"))
        os.makedirs(dest_dir, exist_ok=True)
        for src in paths:
            base = os.path.basename(src)
            dst = os.path.join(dest_dir, base)
            if os.path.exists(dst):                   # name collision → hash suffix
                stem, ext = os.path.splitext(base)
                h = hashlib.blake2b(src.encode(), digest_size=4).hexdigest()
                dst = os.path.join(dest_dir, f"{stem}-{h}{ext}")
            try:
                if mode == "move":
                    shutil.move(src, dst)
                else:
                    shutil.copy2(src, dst)
            except OSError:
                continue
            manifest[src] = {"dst": dst, "mode": mode}
            done[bucket] += 1
    tmp = EXPORT_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, EXPORT_MANIFEST)
    return {"moved_reference": done["reference"], "copied_in_between": done["in_between"],
            "total_exported": len(manifest), "dest": EXPORT_ROOT}


_DEL_CACHE = {"mtime": -1.0, "set": frozenset()}


def delete_set() -> frozenset:
    """Paths flagged 'delete', cached by eval_labels.json mtime — O(1) per-path
    lookups for the service's global hide filter (re-primes when any process
    writes a new flag)."""
    try:
        mt = LABELS_FILE.stat().st_mtime
    except OSError:
        return frozenset()
    if mt != _DEL_CACHE["mtime"]:
        labels = _load_labels()
        _DEL_CACHE["set"] = frozenset(p for p, e in labels.items()
                                      if e.get("flag") == "delete")
        _DEL_CACHE["mtime"] = mt
    return _DEL_CACHE["set"]


def results() -> dict:
    """Model-vs-human accuracy + confusion + per-bucket precision over labeled images."""
    labels = _load_labels()
    entries = personalness.all_entries()
    confusion = {a: {b: 0 for b in BUCKETS} for a in BUCKETS}  # [model][human]
    n = correct = 0
    for path, lab in labels.items():
        human = lab.get("label")
        model = (entries.get(path) or {}).get("bucket") or lab.get("model")
        if human not in BUCKETS or model not in BUCKETS:
            continue
        confusion[model][human] += 1
        n += 1
        if human == model:
            correct += 1
    per_bucket = {}
    for b in BUCKETS:
        tot = sum(confusion[b].values())
        per_bucket[b] = {"n": tot, "correct": confusion[b][b],
                         "precision": round(confusion[b][b] / tot, 3) if tot else None}
    # binary personal-vs-reference (ignoring in_between), the meaningful axis
    pp = confusion["personal"]; rr = confusion["reference"]
    bin_p = pp["personal"] / ((pp["personal"] + pp["reference"]) or 1)
    bin_r = rr["reference"] / ((rr["reference"] + rr["personal"]) or 1)
    return {"labeled": n, "correct": correct,
            "accuracy": round(correct / n, 3) if n else None,
            "binary_personal": round(bin_p, 3), "binary_reference": round(bin_r, 3),
            "confusion": confusion, "per_bucket": per_bucket}
