"""Shared sidecar scaffolding for per-image precomputed *facets* (color, skin-tone).

Mirrors the proven `c2pa.py` cache pattern, factored out so the color and
skin-tone indexes don't each re-implement it:

- one ``~/.muser/<name>.json`` keyed by absolute path, each entry carrying
  ``m`` (mtime_ns) + ``s`` (size) so re-scans skip unchanged files (incremental),
- a thread-pool scan (`compute(path) -> payload`) — image decode + numpy/opencv
  release the GIL, so threads parallelize the per-file work,
- a RAM cache primed once (at service startup) for O(1) result enrichment and
  in-memory ranking, exactly like the C2PA verdict cache.

`c2pa.py` predates this and keeps its own copy — left untouched so a working,
shipped path isn't disturbed. New facets build on this.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from .paths import MUSER_HOME


def blake2b_file(path: str, _buf: int = 1 << 20) -> str | None:
    """Exact content hash of a file's bytes (blake2b, 16-byte digest → 32 hex
    chars). Used as a content-address so a previously-computed facet can be
    reused when the same bytes reappear at a new path / after deletion. MUST be
    an exact (not perceptual) hash: facet computes like GRIP depend on the exact
    pixels, so a perceptual collision would reuse a wrong score. None on error."""
    try:
        h = hashlib.blake2b(digest_size=16)
        with open(path, "rb") as f:
            while chunk := f.read(_buf):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


class Sidecar:
    def __init__(self, name: str):
        self.name = name
        self.path = MUSER_HOME / f"{name}.json"
        self._cache: dict[tuple, dict] = {}  # (path, mtime_ns, size) -> payload
        self._primed = False
        self._sidecar_mtime = -1.0  # mtime of self.path at last prime, for auto-reprime

    def exists(self) -> bool:
        return self.path.exists()

    def _load_raw(self) -> dict:
        if not self.path.exists():
            return {}
        raw = self.path.read_text()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            q = self.path.with_suffix(f".corrupt-{int(time.time())}.json")
            q.write_text(raw)
            raise RuntimeError(f"{self.path.name} is corrupt — quarantined to {q.name}; "
                               "refusing to load from an empty base")

    def load(self) -> dict:
        return self._load_raw().get("entries", {})

    def load_by_hash(self) -> dict:
        """Content-addressed store: blake2b(bytes) -> payload (no m/s; keyed by
        bytes, so it survives a file being moved, renamed, deleted, or re-imported).
        Empty for facets that don't opt into content addressing."""
        return self._load_raw().get("by_hash", {})

    def save(self, entries: dict, by_hash: dict | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve any existing content store when a non-content-addressed save
        # (color/c2pa) passes by_hash=None — never clobber it to {}.
        if by_hash is None:
            by_hash = self._load_raw().get("by_hash", {})
        tmp = self.path.parent / f".{os.getpid()}-{threading.get_ident()}-{self.path.name}.tmp"
        tmp.write_text(json.dumps({"version": 1, "entries": entries, "by_hash": by_hash}))
        tmp.replace(self.path)  # atomic — never leave a half-written sidecar

    def prime(self) -> int:
        """Populate the RAM cache from the persisted sidecar. Idempotent."""
        if self._primed:
            return len(self._cache)
        cache: dict[tuple, dict] = {}
        for p, e in self.load().items():
            m, s = e.get("m"), e.get("s")
            if m is None or s is None:
                continue
            cache[(p, m, s)] = e
        self._cache = cache
        self._primed = True
        try:
            self._sidecar_mtime = self.path.stat().st_mtime
        except OSError:
            self._sidecar_mtime = -1.0
        return len(self._cache)

    def _maybe_reprime(self) -> None:
        """Re-prime if the sidecar file changed on disk since we last primed, so a
        reader (the service) reflects writes from ANY process — a standalone
        `muser aiscore`, the post-index hook, another window — without a restart.
        A cheap stat per call; the full reload happens only when mtime actually
        moves (e.g. once per scan checkpoint)."""
        try:
            mt = self.path.stat().st_mtime
        except OSError:
            return
        if not self._primed or mt != self._sidecar_mtime:
            self._primed = False
            self.prime()

    def entries(self) -> dict:
        """RAM cache as {(path, m, s): payload}; primes (or re-primes on disk change)."""
        self._maybe_reprime()
        return self._cache

    def lookup(self, path: str) -> dict | None:
        """RAM-only payload lookup, valid only if (mtime, size) still match."""
        self._maybe_reprime()
        try:
            st = os.stat(path)
        except OSError:
            return None
        return self._cache.get((path, st.st_mtime_ns, st.st_size))

    def scan(self, paths, compute, progress=None, workers: int = 8, version: int = 1,
             checkpoint_every: int = 1000, content_addressed: bool = False) -> dict:
        """Incrementally compute the facet for each path. Returns the full cache.

        ``compute(path) -> dict`` produces the per-image payload (``m``/``s``/``v``
        are added here). Files whose (mtime, size) match AND whose stored ``v``
        equals ``version`` are skipped — so bumping ``version`` after an algorithm
        change forces a full recompute even on unchanged files.
        ``progress(done, total)`` is called as it goes.

        ``content_addressed`` (opt-in, for *expensive* computes like GRIP): also
        key payloads by ``blake2b(bytes)`` in a persistent ``by_hash`` store, so a
        file that was already scored under another path — or was deleted and later
        re-imported — is reused (one cheap hash read) instead of re-running the
        model. Existing stat-hit entries are backfilled into the store (hash only,
        never recompute). Skip it for cheap facets (color/c2pa): the hashing isn't
        worth it when the compute is already milliseconds.
        """
        if content_addressed:
            return self._scan_content_addressed(
                paths, compute, progress, workers, version, checkpoint_every)
        cache = self.load()
        total, done, changed = len(paths), 0, False

        todo = []
        for p in paths:
            try:
                st = os.stat(p)
            except OSError:
                done += 1
                continue
            e = cache.get(p)
            if (e and e.get("m") == st.st_mtime_ns and e.get("s") == st.st_size
                    and e.get("v", 1) == version):
                done += 1
                continue
            todo.append((p, st))
        if progress:
            progress(done, total)

        def work(item):
            p, st = item
            try:
                payload = compute(p) or {}
            except Exception:
                payload = {}
            return p, st, payload

        since_ckpt = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, st, payload in ex.map(work, todo):
                payload = dict(payload)
                payload["m"], payload["s"], payload["v"] = st.st_mtime_ns, st.st_size, version
                cache[p] = payload
                changed = True
                done += 1
                since_ckpt += 1
                if progress and done % 50 == 0:
                    progress(done, total)
                # Periodic atomic checkpoint so a long scan (e.g. the full-corpus
                # GRIP pass, ~80 min) is crash-safe and resumable — the next run
                # skips already-saved (mtime,size,version)-matching entries. A
                # partial sidecar is never observable (write-to-tmp + rename).
                # Re-prime the RAM cache too, so a long scan's results go live
                # progressively (badges/filters work mid-scan, not just at the end).
                if since_ckpt >= checkpoint_every:
                    self.save(cache)
                    self._primed = False
                    self.prime()
                    since_ckpt = 0
        if changed:
            self.save(cache)
        if progress:
            progress(done, total)
        # Refresh the RAM cache so search/enrichment see the new entries.
        self._primed = False
        self.prime()
        return cache

    def _scan_content_addressed(self, paths, compute, progress, workers, version,
                                checkpoint_every) -> dict:
        cache = self.load()
        by_hash = self.load_by_hash()
        total, done, changed = len(paths), 0, False

        def _facet_only(entry: dict) -> dict:
            # The reusable payload: facet fields + version, minus per-file keys.
            return {k: v for k, v in entry.items() if k not in ("m", "s", "h")}

        # Triage: fully-skippable (already scored AND already in the content
        # store), backfill (scored but no hash yet → hash to populate the store,
        # NEVER recompute), and todo (needs a value — reuse-by-hash or compute).
        backfill, todo = [], []
        for p in paths:
            try:
                st = os.stat(p)
            except OSError:
                done += 1
                continue
            e = cache.get(p)
            if (e and e.get("m") == st.st_mtime_ns and e.get("s") == st.st_size
                    and e.get("v", 1) == version):
                if "h" in e:
                    done += 1            # fully cached + indexed → skip
                else:
                    backfill.append((p, st, e))
            else:
                todo.append((p, st))
        if progress:
            progress(done, total)

        def _ckpt():
            self.save(cache, by_hash)
            self._primed = False
            self.prime()

        # Single streaming pass: each worker hashes its file, then either reuses a
        # stored payload (by content hash) or computes once — so progress, the
        # sidecar, and the content store all advance continuously (no front-loaded
        # hashing phase that looks frozen). A lock guards by_hash so duplicates
        # within the same scan reuse rather than both computing; a rare race (two
        # identical files resolved in the same instant) just costs one extra
        # compute, never a wrong value. Backfill items only hash (never compute).
        lock = threading.Lock()

        def _safe_compute(p):
            try:
                return dict(compute(p) or {})
            except Exception:
                return {}

        def work(item):
            kind, p, st, e = item
            h = blake2b_file(p)
            if kind == "backfill":
                return ("backfill", p, st, e, h, None)
            if h is None:  # unreadable → compute directly, don't cache by content
                return ("nohash", p, st, None, None, _safe_compute(p))
            with lock:
                hit = by_hash.get(h)
                if hit and hit.get("v", 1) == version:
                    return ("reuse", p, st, None, h, dict(hit))
            payload = _safe_compute(p)
            payload["v"] = version
            with lock:
                by_hash.setdefault(h, payload)   # canonicalize (handles the race)
                payload = dict(by_hash[h])
            return ("computed", p, st, None, h, payload)

        items = ([("backfill", p, st, e) for p, st, e in backfill]
                 + [("todo", p, st, None) for p, st in todo])
        since_ckpt = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for kind, p, st, e, h, payload in ex.map(work, items):
                if kind == "backfill":
                    if h:
                        e["h"] = h
                        with lock:
                            by_hash.setdefault(h, _facet_only(e))
                else:
                    entry = dict(payload or {})
                    entry["m"], entry["s"], entry["v"] = st.st_mtime_ns, st.st_size, version
                    if h:
                        entry["h"] = h
                    cache[p] = entry
                done += 1
                changed = True
                since_ckpt += 1
                if since_ckpt >= checkpoint_every:
                    _ckpt(); since_ckpt = 0
                if progress and done % 50 == 0:
                    progress(done, total)

        if changed:
            self.save(cache, by_hash)
        if progress:
            progress(done, total)
        self._primed = False
        self.prime()
        return cache
