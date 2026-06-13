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
    """A diverse sample: balanced across predicted buckets, capped per album."""
    entries = personalness.all_entries()
    items = list(entries.items())
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
    labels = _load_labels()
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
    try:
        return json.loads(LABELS_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _save(labels: dict) -> None:
    tmp = LABELS_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(labels))
    os.replace(tmp, LABELS_FILE)


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
