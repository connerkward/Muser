"""User-curated **people** on top of the raw face clusters — name, merge, and
auto-classify-as-personal.

The HDBSCAN people-clusters in `faces.py` are unstable (cluster ids renumber on
every re-cluster) and imperfect (one real person can split across clusters; a
cluster can mix people). So this layer lets the user *curate*: give a cluster a
name, merge several clusters/people into one, and have **every photo of a named
person automatically bucketed `personal`**.

Anchoring: a person is stored as a **set of image paths** (seeded from a
cluster's members at naming time, unioned on merge), NOT cluster ids — so a name
survives a re-cluster. Re-clustering just offers fresh unnamed clusters the user
can merge in.

Persistence: `people.json` (one per `MUSER_HOME`, so the personal instance has
its own). The auto-personal effect is a **write-through** into
`personalness.json`: a named person's images get `bucket="personal"` with the
original bucket remembered in `obucket` so unnaming is reversible. The write-
through is a no-op when `personalness.json` is absent (e.g. the main instance),
so naming/merging still works there purely as organization.
"""
from __future__ import annotations

import json
import os
import threading
import time

from ..paths import data_file

PEOPLE_FILE = data_file("people.json")
VERSION = 1
CLAIM_FRAC = 0.5   # a raw cluster is "claimed" (hidden from the unnamed list) when
                   # ≥ this fraction of its members belong to a named person.


def load() -> dict:
    if not PEOPLE_FILE.exists():
        return {"version": VERSION, "people": []}
    raw = PEOPLE_FILE.read_text()
    try:
        d = json.loads(raw)
        d.setdefault("people", [])
        return d
    except json.JSONDecodeError:
        q = PEOPLE_FILE.with_suffix(f".corrupt-{int(time.time())}.json")
        q.write_text(raw)
        raise RuntimeError(f"people.json is corrupt — quarantined to {q.name}; "
                           "refusing to continue from an empty people store")


def _save(d: dict) -> None:
    PEOPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = PEOPLE_FILE.parent / f".{os.getpid()}-{threading.get_ident()}-{PEOPLE_FILE.name}.tmp"
    tmp.write_text(json.dumps(d))
    os.replace(tmp, PEOPLE_FILE)


def _next_id(d: dict) -> str:
    n = 1
    existing = {p["id"] for p in d["people"]}
    while f"p{n}" in existing:
        n += 1
    return f"p{n}"


def _person_by_name(d: dict, name: str) -> dict | None:
    nl = name.strip().lower()
    for p in d["people"]:
        if p["name"].strip().lower() == nl:
            return p
    return None


def list_people() -> list[dict]:
    """Curated people, largest first: {id, name, size, paths}."""
    d = load()
    out = [{"id": p["id"], "name": p["name"], "size": len(p.get("paths", [])),
            "paths": p.get("paths", [])} for p in d["people"]]
    out.sort(key=lambda p: -p["size"])
    return out


def claimed_clusters() -> dict[int, str]:
    """Map raw cluster-id -> owning person name, for clusters whose members are
    mostly inside a named person (so /api/faces can hide them from the unnamed
    list and surface them under the person instead)."""
    from .. import faces as _faces
    d = load()
    if not d["people"]:
        return {}
    owned = set()
    name_of_path: dict[str, str] = {}
    for p in d["people"]:
        for path in p.get("paths", []):
            owned.add(path)
            name_of_path[path] = p["name"]
    claimed: dict[int, str] = {}
    for label in _faces.clusters():
        members = _faces.cluster_members(int(label))
        if not members:
            continue
        inside = [m for m in members if m in owned]
        if len(inside) >= CLAIM_FRAC * len(members):
            # attribute to the person owning the plurality of claimed members
            from collections import Counter
            top = Counter(name_of_path[m] for m in inside).most_common(1)
            claimed[int(label)] = top[0][0] if top else ""
    return claimed


def _person_paths_for_clusters(cluster_ids: list[int]) -> list[str]:
    from .. import faces as _faces
    seen: set[str] = set()
    out: list[str] = []
    for cid in cluster_ids:
        for p in _faces.cluster_members(int(cid)):
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def name_cluster(cluster_id: int, name: str) -> dict:
    """Name a single raw cluster. Extends an existing same-name person (so naming
    two clusters 'Mom' merges them) or creates a new one. Triggers write-through."""
    name = name.strip()
    if not name:
        raise ValueError("empty name")
    d = load()
    paths = _person_paths_for_clusters([cluster_id])
    person = _person_by_name(d, name)
    if person is None:
        person = {"id": _next_id(d), "name": name, "paths": []}
        d["people"].append(person)
    person["name"] = name
    person["paths"] = _union(person.get("paths", []), paths)
    _save(d)
    apply_personal()
    return {"id": person["id"], "name": person["name"], "size": len(person["paths"])}


def merge(refs: list[str], name: str) -> dict:
    """Merge a mix of raw clusters (`c:<id>`) and existing people (`p:<id>`) into
    one named person (union of all their image paths). Merged people are removed."""
    name = name.strip()
    if not name:
        raise ValueError("empty name")
    d = load()
    paths: list[str] = []
    cluster_ids: list[int] = []
    drop_person_ids: set[str] = set()
    for ref in refs:
        kind, _, val = ref.partition(":")
        if kind == "c":
            cluster_ids.append(int(val))
        elif kind == "p":
            drop_person_ids.add(val)
    if cluster_ids:
        paths = _union(paths, _person_paths_for_clusters(cluster_ids))
    target = _person_by_name(d, name)
    for p in d["people"]:
        if p["id"] in drop_person_ids:
            paths = _union(paths, p.get("paths", []))
    if target is None:
        target = {"id": _next_id(d), "name": name, "paths": []}
        d["people"].append(target)
    target["name"] = name
    target["paths"] = _union(target.get("paths", []), paths)
    # remove the merged-away people (but never the target itself)
    d["people"] = [p for p in d["people"]
                   if p["id"] == target["id"] or p["id"] not in drop_person_ids]
    _save(d)
    apply_personal()
    return {"id": target["id"], "name": target["name"], "size": len(target["paths"])}


def rename(person_id: str, name: str) -> bool:
    name = name.strip()
    if not name:
        raise ValueError("empty name")
    d = load()
    for p in d["people"]:
        if p["id"] == person_id:
            p["name"] = name
            _save(d)
            apply_personal()
            return True
    return False


def delete(person_id: str) -> bool:
    """Un-name a person: drop the registry entry and revert its images' buckets."""
    d = load()
    before = len(d["people"])
    d["people"] = [p for p in d["people"] if p["id"] != person_id]
    if len(d["people"]) == before:
        return False
    _save(d)
    apply_personal()
    return True


def _union(a: list[str], b: list[str]) -> list[str]:
    seen = set(a)
    out = list(a)
    for x in b:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def apply_personal() -> int:
    """Write-through: force every named person's images to bucket `personal`
    (remembering the original in `obucket`), and revert any image that is no
    longer owned. Returns the number of changed entries. No-op without a
    personalness sidecar (e.g. the main instance)."""
    from . import personalness as _pn
    if not _pn.exists():
        return 0
    entries = _pn.all_entries()
    if not entries:
        return 0
    owned: dict[str, str] = {}
    for person in load()["people"]:
        for path in person.get("paths", []):
            owned[path] = person["name"]
    changed = 0
    for path, e in entries.items():
        if path in owned:
            if "obucket" not in e:
                e["obucket"] = e.get("bucket")
            if e.get("bucket") != "personal" or e.get("forced") != owned[path]:
                e["bucket"] = "personal"
                e["forced"] = owned[path]
                changed += 1
        elif "forced" in e:                       # released → revert to model bucket
            e["bucket"] = e.get("obucket", e.get("bucket"))
            e.pop("forced", None)
            e.pop("obucket", None)
            changed += 1
    if changed:
        _pn.sidecar().save(entries)
        _pn.sidecar()._primed = False
        _pn.sidecar().prime()
    return changed
