"""C2PA provenance check — read Content Credentials and report AI-generated origin.

A thin wrapper over the `c2patool` CLI (https://github.com/contentauth/c2patool).
Given a file path, it answers one question: does the embedded, cryptographically
signed C2PA manifest *declare* this content as AI-generated?

This is a **positive-only** signal — the honest framing the UI shows is "AI?".

- A signed manifest with an IPTC ``digitalSourceType`` of ``trainedAlgorithmicMedia``
  (fully synthetic) or ``compositeWithTrainedAlgorithmicMedia`` (AI-edited) → ``ai=True``.
  This is how OpenAI (DALL·E / GPT-image / ChatGPT), Adobe Firefly, and Google tag
  their cloud output.
- No manifest, or a manifest with no AI source-type → ``ai=False``. The overwhelming
  majority of files — including everything from local SD/Flux/ComfyUI — land here, so
  ``ai=False`` means "no provenance says AI", NOT "confirmed real".
- ``c2patool`` not installed, or the file unreadable → ``ai=None`` (unknown); the UI
  simply shows no badge.

Degrades gracefully when the binary is absent, mirroring the service's other
optional OS integrations (_reveal, clipboard).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# IPTC digitalSourceType codes that mean "AI was involved". Matched as a
# case-insensitive substring against the whole report, so the check survives
# c2patool/spec version churn (actions vs actions.v2, full URI vs bare code).
_GENERATED = "trainedalgorithmicmedia"            # fully synthetic
_COMPOSITE = "compositewithtrainedalgorithmicmedia"  # AI-edited / partly synthetic

# Probe the binary once; cache verdicts by (path, mtime, size) so repeated
# searches over the same results don't re-spawn a subprocess per thumbnail.
_tool: str | None | bool = False  # False = not yet probed; None = absent; str = path
_cache: dict[tuple, dict] = {}
_primed: bool = False


def _tool_path() -> str | None:
    global _tool
    if _tool is False:
        _tool = shutil.which("c2patool")
    return _tool  # type: ignore[return-value]


def available() -> bool:
    return _tool_path() is not None


def _tool_name(manifests: dict, active: str | None) -> str | None:
    """Best-effort human label for the signer/generator, e.g. 'DALL·E' / 'Adobe Firefly'."""
    m = manifests.get(active) if active else None
    if not m and manifests:
        m = next(iter(manifests.values()))
    if not isinstance(m, dict):
        return None
    # Newer manifests carry a structured list; older ones a flat string.
    info = m.get("claim_generator_info")
    if isinstance(info, list) and info and isinstance(info[0], dict):
        return info[0].get("name") or m.get("claim_generator")
    return m.get("claim_generator")


def verdict(path: str) -> dict:
    """Return {available, ai, kind, tool} for one file. Never raises."""
    if not _tool_path():
        return {"available": False, "ai": None, "kind": None, "tool": None}
    try:
        st = os.stat(path)
    except OSError:
        return {"available": True, "ai": None, "kind": None, "tool": None}

    key = (path, st.st_mtime_ns, st.st_size)
    if key in _cache:
        return _cache[key]

    res = {"available": True, "ai": False, "kind": None, "tool": None}
    try:
        proc = subprocess.run(
            [_tool_path(), path],  # type: ignore[list-item]
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        res["ai"] = None
        _cache[key] = res
        return res

    out = proc.stdout or ""
    if proc.returncode != 0:
        # No claim found is the common, expected case for ordinary files → not AI.
        # Anything else (unsupported format, read error) is genuinely unknown.
        err = (proc.stderr or "").lower()
        if "no claim" not in err and "no manifest" not in err and "jumbf" not in err:
            res["ai"] = None
        _cache[key] = res
        return res

    low = out.lower()
    if _COMPOSITE in low:
        res["ai"], res["kind"] = True, "composite"
    elif _GENERATED in low:
        res["ai"], res["kind"] = True, "generated"
    if res["ai"]:
        try:
            report = json.loads(out)
            res["tool"] = _tool_name(report.get("manifests", {}), report.get("active_manifest"))
        except (json.JSONDecodeError, AttributeError):
            pass
    _cache[key] = res
    return res


# ---------------------------------------------------------------------------
# Library scan — the "AI images" tab. Probing 27k+ files on every tab open is a
# non-starter (one subprocess each), so a scan runs once over the whole index and
# persists verdicts to ~/.muser/c2pa.json, mirroring scores.json / clusters.json.
# Incremental: a file unchanged by (mtime, size) is skipped on re-scan. The scan
# is subprocess-bound, so a thread pool parallelizes it (GIL released in execve).
# ---------------------------------------------------------------------------

CACHE_JSON = Path.home() / ".muser" / "c2pa.json"


def cache_exists() -> bool:
    return CACHE_JSON.exists()


def _load_cache() -> dict:
    try:
        return json.loads(CACHE_JSON.read_text()).get("entries", {})
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(entries: dict) -> None:
    CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    CACHE_JSON.write_text(json.dumps({"version": 1, "entries": entries}))


def prime_cache_from_sidecar() -> int:
    """Populate the in-memory verdict cache from the persisted scan.

    Called once at service startup so /api/c2pa lookups and search-result
    enrichment serve from RAM instead of spawning c2patool per request.
    Returns the number of entries primed. Idempotent.
    """
    global _primed
    if _primed:
        return len(_cache)
    if not available():
        _primed = True
        return 0
    primed = 0
    for p, e in _load_cache().items():
        m, s = e.get("m"), e.get("s")
        if m is None or s is None:
            continue
        _cache[(p, m, s)] = {"available": True, "ai": bool(e.get("ai")),
                             "kind": e.get("kind"), "tool": e.get("tool")}
        primed += 1
    _primed = True
    return primed


def lookup(path: str) -> dict | None:
    """Best-effort RAM-only verdict lookup. Returns the cached entry if the
    file's (mtime, size) match the primed sidecar key; otherwise None.

    Used to enrich search results without ever spawning c2patool — falls back
    to absence (no badge) when the cache misses, which is the right policy:
    the next direct /api/c2pa hit will do the real check and populate _cache.
    """
    if not _primed or not available():
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return _cache.get((path, st.st_mtime_ns, st.st_size))


def ai_images() -> list[dict]:
    """AI-positive entries from the persisted scan, newest-tagged first, existing files only."""
    out = []
    for p, e in _load_cache().items():
        if e.get("ai") and os.path.exists(p):
            out.append({"path": p, "name": os.path.basename(p),
                        "kind": e.get("kind"), "tool": e.get("tool")})
    return out


def scan(paths, progress=None, workers: int = 12) -> dict:
    """Probe every path's Content Credentials, persisting verdicts. Returns the cache.

    ``progress(done, total, found)`` is called as it goes. Unchanged files are
    skipped via the persisted (mtime, size); only new/changed files spawn c2patool.
    """
    if not available():
        return {}
    cache = _load_cache()
    total, done, found, changed = len(paths), 0, 0, False

    todo = []
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            done += 1
            continue
        e = cache.get(p)
        if e and e.get("m") == st.st_mtime_ns and e.get("s") == st.st_size:
            done += 1
            found += bool(e.get("ai"))
            continue
        todo.append((p, st))
    if progress:
        progress(done, total, found)

    def work(item):
        p, st = item
        return p, st, verdict(p)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for p, st, v in ex.map(work, todo):
            cache[p] = {"m": st.st_mtime_ns, "s": st.st_size,
                        "ai": bool(v.get("ai")), "kind": v.get("kind"), "tool": v.get("tool")}
            changed = True
            done += 1
            found += bool(v.get("ai"))
            if progress and done % 50 == 0:
                progress(done, total, found)
    if changed:
        _save_cache(cache)
    if progress:
        progress(done, total, found)
    return cache
