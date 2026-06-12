"""Parse Google Takeout JSON sidecars → a per-image metadata facet.

Each image ``NAME.EXT`` has a sibling ``NAME.EXT.supplemental-metadata.json`` (older
exports: ``NAME.EXT.json``) carrying capture time, GPS, and — when the user tagged
them in Google Photos — **people names**. Album folders carry a folder-level
``metadata.json``. The JSON→image name match is fuzzy (Google truncates long names and
appends ``(1)`` for dup-named files), so we try a few candidate names then fall back to
a per-directory glob.

We extract only what the classifier uses:
  - ``taken``    capture epoch (photoTakenTime; falls back to file mtime) — recency / face date-span
  - ``lat,lon``  GPS (0/0 ⇒ absent) — home-location clustering (optional signal)
  - ``people``   Google's own people tags — a STRONG personal signal when present
  - ``album``    parent folder name
  - ``roll``     True when the folder is an auto "Photos from YYYY" camera-roll bucket

Stored via ``facets.Sidecar("takeout_meta")`` so it lives under the personal root and
is incremental by (mtime,size).
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

from ..facets import Sidecar
from ..index import IMAGE_EXTS, walk_images

_SIDE = Sidecar("takeout_meta")
VERSION = 1
_ROLL_RE = re.compile(r"^Photos from \d{4}$")
_dir_cache: dict[str, list[str]] = {}   # dir -> list of .json basenames (globbed once)


def album_of(path: str) -> str:
    return Path(path).parent.name


def is_camera_roll(path: str) -> bool:
    """True for the auto 'Photos from YYYY' backup buckets (raw camera roll)."""
    return bool(_ROLL_RE.match(album_of(path)))


def _candidate_jsons(img: Path) -> list[Path]:
    d, name = img.parent, img.name
    cands = [
        d / f"{name}.supplemental-metadata.json",
        d / f"{name}.json",
    ]
    # "-edited" variants share the original's sidecar
    stem = img.stem
    if stem.endswith("-edited"):
        base = name.replace("-edited", "")
        cands += [d / f"{base}.supplemental-metadata.json", d / f"{base}.json"]
    return cands


def _find_sidecar(img: Path) -> Path | None:
    for c in _candidate_jsons(img):
        if c.exists():
            return c
    # Fallback: Google truncates long json names (51-char cap) and mangles
    # "supplemental-metadata" → "supplemental-met…". Glob NAME*.json in the dir.
    d = str(img.parent)
    if d not in _dir_cache:
        try:
            _dir_cache[d] = [os.path.basename(p) for p in glob.glob(os.path.join(glob.escape(d), "*.json"))]
        except OSError:
            _dir_cache[d] = []
    pre = img.name[:46]  # leave room for the truncated suffix
    for b in _dir_cache[d]:
        if b.startswith(pre) or b.startswith(img.stem[:40]):
            return img.parent / b
    return None


def _parse(js: Path) -> dict:
    try:
        d = json.loads(js.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict = {}
    pt = (d.get("photoTakenTime") or {}).get("timestamp")
    if pt:
        try:
            out["taken"] = int(pt)
        except (TypeError, ValueError):
            pass
    geo = d.get("geoData") or {}
    lat, lon = geo.get("latitude") or 0.0, geo.get("longitude") or 0.0
    if lat or lon:
        out["lat"], out["lon"] = round(float(lat), 6), round(float(lon), 6)
    people = [p.get("name") for p in (d.get("people") or []) if p.get("name")]
    if people:
        out["people"] = people
    desc = (d.get("description") or "").strip()
    if desc:
        out["desc"] = desc[:200]
    return out


def _compute(path: str) -> dict:
    img = Path(path)
    meta = {"album": album_of(path), "roll": is_camera_roll(path)}
    js = _find_sidecar(img)
    if js is not None:
        meta.update(_parse(js))
    if "taken" not in meta:
        try:
            meta["taken"] = int(os.stat(path).st_mtime)  # fallback: file mtime
        except OSError:
            pass
    return meta


def scan(paths, progress=None) -> dict:
    """Build the takeout-metadata facet for `paths` (JSON parse, no model)."""
    return _SIDE.scan(paths, _compute, progress=progress, workers=8, version=VERSION,
                      checkpoint_every=2000)


def walk(src: str | Path) -> list[str]:
    """All indexable images under the Takeout tree (heic included; video/raw skipped)."""
    return walk_images(src, recursive=True)


def skipped_summary(src: str | Path) -> dict[str, int]:
    """Count files dropped from v1 scope (video/raw/other) — logged, never silent."""
    counts: dict[str, int] = {}
    for p in Path(src).rglob("*"):
        if p.is_file() and not p.name.startswith("."):
            ext = p.suffix.lower()
            if ext and ext not in IMAGE_EXTS and ext != ".json":
                counts[ext] = counts.get(ext, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def prime() -> int:
    return _SIDE.prime()


def lookup(path: str) -> dict | None:
    return _SIDE.lookup(path)


def sidecar() -> Sidecar:
    return _SIDE
