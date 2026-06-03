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

# IPTC digitalSourceType codes that mean "AI was involved". Matched as a
# case-insensitive substring against the whole report, so the check survives
# c2patool/spec version churn (actions vs actions.v2, full URI vs bare code).
_GENERATED = "trainedalgorithmicmedia"            # fully synthetic
_COMPOSITE = "compositewithtrainedalgorithmicmedia"  # AI-edited / partly synthetic

# Probe the binary once; cache verdicts by (path, mtime, size) so repeated
# searches over the same results don't re-spawn a subprocess per thumbnail.
_tool: str | None | bool = False  # False = not yet probed; None = absent; str = path
_cache: dict[tuple, dict] = {}


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
