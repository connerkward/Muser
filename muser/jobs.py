"""In-process job registry for non-blocking background work (e.g. cart checkout).

The embedded service (``muser serve``) has a single ``state.task`` slot used by
blocking, UI-foregrounding tasks (index, backfill, captioning) — only one runs at
a time and it gates the busy overlay. A *checkout* must NOT take that slot: the
user keeps searching/indexing while captioning (network) and upscaling (local 4×)
run concurrently in the background. This module is that out-of-band registry —
thread-safe, pollable, and holding the finished zip in RAM until downloaded.

Jobs are ephemeral: evicted after 1 hour or once 32 newer jobs exist. Nothing is
persisted to disk (the upscale_cache and captions.jsonl are the durable artifacts;
the assembled zip is throwaway).
"""

from __future__ import annotations

import secrets
import threading
import time

_MAX_JOBS = 32
_MAX_AGE_S = 3600.0


class JobRegistry:
    """Thread-safe dict of background jobs keyed by short id.

    A job dict has the shape::

        {
          id, kind, status: "running"|"done"|"error", created: <epoch>,
          caption: {done, total}, upscale: {done, total},
          stages: [{name, status, done, total}, ...],
          errors: [{path, stage, error}, ...],
          captioned: int, upscaled: int, captions_missing: int,
          zip: <bytes|None>, zip_ready: bool, error: <str|None>,
        }

    ``stages`` is a generic, ordered progress list used by multi-stage jobs (the
    generate-mode checkout pipeline). ``caption``/``upscale``/``zip`` stay for the
    legacy zip-mode checkout; new jobs drive ``stages`` via :meth:`set_stage`.

    The raw ``zip`` bytes are never serialized — :func:`public` strips them.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict] = {}

    def _evict_locked(self) -> None:
        # Drop anything older than _MAX_AGE_S, then cap the survivors at the
        # newest _MAX_JOBS. Called under self._lock on create().
        now = time.time()
        stale = [jid for jid, j in self._jobs.items() if now - j["created"] > _MAX_AGE_S]
        for jid in stale:
            del self._jobs[jid]
        if len(self._jobs) > _MAX_JOBS:
            # Keep newest by created; drop the oldest overflow.
            ordered = sorted(self._jobs.items(), key=lambda kv: kv[1]["created"], reverse=True)
            for jid, _ in ordered[_MAX_JOBS:]:
                del self._jobs[jid]

    def create(self, kind: str, *, caption_total: int, upscale_total: int) -> str:
        jid = secrets.token_hex(8)
        job = {
            "id": jid,
            "kind": kind,
            "status": "running",
            "created": time.time(),
            "caption": {"done": 0, "total": int(caption_total)},
            "upscale": {"done": 0, "total": int(upscale_total)},
            "stages": [],
            "errors": [],
            "captioned": 0,
            "upscaled": 0,
            "captions_missing": 0,
            "zip": None,
            "zip_ready": False,
            "error": None,
        }
        with self._lock:
            self._evict_locked()
            self._jobs[jid] = job
        return jid

    def get(self, jid: str) -> dict | None:
        with self._lock:
            return self._jobs.get(jid)

    def update(self, jid: str, **fields) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None:
                j.update(fields)

    def set_progress(self, jid: str, stage: str, done: int) -> None:
        """Set the ``done`` counter for ``stage`` ("caption" | "upscale")."""
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None and stage in ("caption", "upscale"):
                j[stage]["done"] = int(done)

    def set_stage(
        self,
        jid: str,
        name: str,
        status: str | None = None,
        done: int | None = None,
        total: int | None = None,
        note: str | None = None,
        request: str | None = None,
    ) -> None:
        """Upsert a generic ``stages`` entry by ``name``.

        Creates the stage on first reference; subsequent calls patch only the
        supplied fields (``status`` / ``done`` / ``total`` / ``note`` /
        ``request``). ``note`` is a short human string (e.g. fal queue status)
        and ``request`` the fal request id, surfaced on the Jobs page. Used by
        the generate-mode pipeline to publish ordered, named progress without
        touching the legacy ``caption``/``upscale`` slots.
        """
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                return
            stages = j.setdefault("stages", [])
            entry = next((s for s in stages if s.get("name") == name), None)
            if entry is None:
                entry = {"name": name, "status": "running", "done": 0, "total": 0}
                stages.append(entry)
            if status is not None:
                entry["status"] = status
            if done is not None:
                entry["done"] = int(done)
            if total is not None:
                entry["total"] = int(total)
            if note is not None:
                entry["note"] = note
            if request is not None:
                entry["request"] = request

    def add_error(self, jid: str, path: str, stage: str, msg: str) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None:
                j["errors"].append({"path": path, "stage": stage, "error": msg})

    def attach_zip(self, jid: str, data: bytes) -> None:
        with self._lock:
            j = self._jobs.get(jid)
            if j is not None:
                j["zip"] = data
                j["zip_ready"] = True

    def pop_zip(self, jid: str) -> bytes | None:
        """Return the zip bytes (does not delete the job; lets re-download)."""
        with self._lock:
            j = self._jobs.get(jid)
            if j is None:
                return None
            return j.get("zip")


def public(job: dict) -> dict:
    """Status payload for ``/api/checkout/status`` — never includes raw zip bytes.

    Snapshots under the registry lock so a worker thread mutating
    ``stages``/``errors`` (set_stage/add_error) can't change a list mid-copy."""
    with REGISTRY._lock:
        return {
            "id": job["id"],
            "status": job["status"],
            "caption": dict(job["caption"]),
            "upscale": dict(job["upscale"]),
            "stages": [dict(s) for s in job.get("stages", [])],
            "errors": list(job["errors"]),
            "captioned": job["captioned"],
            "upscaled": job["upscaled"],
            "captions_missing": job["captions_missing"],
            "zip_ready": job["zip_ready"],
            "error": job["error"],
        }


# Module singleton — the service imports REGISTRY directly.
REGISTRY = JobRegistry()
