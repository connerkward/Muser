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

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


class Sidecar:
    def __init__(self, name: str):
        self.name = name
        self.path = Path.home() / ".muser" / f"{name}.json"
        self._cache: dict[tuple, dict] = {}  # (path, mtime_ns, size) -> payload
        self._primed = False

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text()).get("entries", {})
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, entries: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"version": 1, "entries": entries}))
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
        return len(self._cache)

    def entries(self) -> dict:
        """RAM cache as {(path, m, s): payload}; primes on first call."""
        if not self._primed:
            self.prime()
        return self._cache

    def lookup(self, path: str) -> dict | None:
        """RAM-only payload lookup, valid only if (mtime, size) still match."""
        if not self._primed:
            self.prime()
        try:
            st = os.stat(path)
        except OSError:
            return None
        return self._cache.get((path, st.st_mtime_ns, st.st_size))

    def scan(self, paths, compute, progress=None, workers: int = 8, version: int = 1) -> dict:
        """Incrementally compute the facet for each path. Returns the full cache.

        ``compute(path) -> dict`` produces the per-image payload (``m``/``s``/``v``
        are added here). Files whose (mtime, size) match AND whose stored ``v``
        equals ``version`` are skipped — so bumping ``version`` after an algorithm
        change forces a full recompute even on unchanged files.
        ``progress(done, total)`` is called as it goes.
        """
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

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for p, st, payload in ex.map(work, todo):
                payload = dict(payload)
                payload["m"], payload["s"], payload["v"] = st.st_mtime_ns, st.st_size, version
                cache[p] = payload
                changed = True
                done += 1
                if progress and done % 50 == 0:
                    progress(done, total)
        if changed:
            self.save(cache)
        if progress:
            progress(done, total)
        # Refresh the RAM cache so search/enrichment see the new entries.
        self._primed = False
        self.prime()
        return cache
