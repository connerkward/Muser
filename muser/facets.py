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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


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
        self.path = Path.home() / ".muser" / f"{name}.json"
        self._cache: dict[tuple, dict] = {}  # (path, mtime_ns, size) -> payload
        self._primed = False
        self._sidecar_mtime = -1.0  # mtime of self.path at last prime, for auto-reprime

    def exists(self) -> bool:
        return self.path.exists()

    def _load_raw(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, json.JSONDecodeError):
            return {}

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
        tmp = self.path.with_suffix(".json.tmp")
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

        # Pass 1 — hash every file we need to touch (parallel I/O).
        hashed = {}  # path -> hash
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, h in ex.map(lambda it: (it[0], blake2b_file(it[0])),
                               backfill + [(p, st) for p, st in todo]):
                hashed[p] = h

        # Backfill the content store from existing scores — hash only, no compute.
        since_ckpt = 0
        for p, st, e in backfill:
            h = hashed.get(p)
            if h:
                e["h"] = h
                by_hash.setdefault(h, _facet_only(e))
            done += 1
            changed = True
            since_ckpt += 1
            if since_ckpt >= checkpoint_every:
                _ckpt(); since_ckpt = 0
            if progress and done % 50 == 0:
                progress(done, total)

        # Group todo by content hash; only hashes absent from the store (or at an
        # older version) need the model — and each unique hash runs exactly once,
        # so in-scan duplicates collapse too.
        by_hash_paths: dict[str, list] = {}
        no_hash = []  # unreadable → compute directly, no caching
        for p, st in todo:
            h = hashed.get(p)
            if h:
                by_hash_paths.setdefault(h, []).append((p, st))
            else:
                no_hash.append((p, st))
        need = [(h, grp[0][0]) for h, grp in by_hash_paths.items()
                if not (by_hash.get(h) and by_hash[h].get("v", 1) == version)]

        def _compute_one(h_path):
            h, p = h_path
            try:
                payload = compute(p) or {}
            except Exception:
                payload = {}
            payload = dict(payload)
            payload["v"] = version
            return h, payload

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for h, payload in ex.map(_compute_one, need):
                by_hash[h] = payload
                changed = True
                since_ckpt += 1
                if since_ckpt >= checkpoint_every:
                    _ckpt(); since_ckpt = 0

        # Assign resolved payloads (reused or freshly computed) to every todo path.
        for h, grp in by_hash_paths.items():
            base = by_hash.get(h, {})
            for p, st in grp:
                entry = dict(base)
                entry["m"], entry["s"], entry["v"], entry["h"] = (
                    st.st_mtime_ns, st.st_size, version, h)
                cache[p] = entry
                done += 1
                changed = True
                since_ckpt += 1
                if since_ckpt >= checkpoint_every:
                    _ckpt(); since_ckpt = 0
                if progress and done % 50 == 0:
                    progress(done, total)
        for p, st in no_hash:  # unhashable: compute, don't cache by content
            try:
                payload = dict(compute(p) or {})
            except Exception:
                payload = {}
            payload["m"], payload["s"], payload["v"] = st.st_mtime_ns, st.st_size, version
            cache[p] = payload
            done += 1
            changed = True

        if changed:
            self.save(cache, by_hash)
        if progress:
            progress(done, total)
        self._primed = False
        self.prime()
        return cache
