"""Single source of truth for the Muser data root.

Every module used to hard-code ``Path.home() / ".muser"`` for its DB and sidecar
files. They now all reference ``MUSER_HOME`` here, so the entire data root can be
relocated with one env var — which is how the **hidden "personal" sub-tool** stays
*completely isolated* from the aesthetic library: it runs as a separate process with
``MUSER_HOME=~/.muser-personal`` set, and its index, scores, captions, facets,
clusters, thumbnails, pipelines — everything — land under that root instead of
``~/.muser``. No table sharing, no per-call plumbing.

Backward-compatible: with the env var unset, ``MUSER_HOME`` is exactly
``~/.muser`` as before.

Resolved once at import. Because the personal instance sets the env *before* the
Python process starts, import-time resolution is correct (the module-level
constants in index.py / score.py / facets.py / … all pick it up).
"""
from __future__ import annotations

import os
from pathlib import Path

#: The Muser data root. Override with ``MUSER_HOME=/some/dir`` (used by the
#: isolated personal instance); defaults to ``~/.muser``.
MUSER_HOME: Path = Path(os.environ.get("MUSER_HOME") or (Path.home() / ".muser")).expanduser()


def data_file(name: str) -> Path:
    """A file directly under the data root, e.g. ``data_file("scores.json")``."""
    return MUSER_HOME / name


def db_dir() -> Path:
    """The embedded-LanceDB directory under the data root."""
    return MUSER_HOME / "db"


def is_personal() -> bool:
    """True when running against a non-default root (the personal sub-tool).

    Used to gate the hidden Triage UI: the main aesthetic instance never shows it.
    """
    return MUSER_HOME.resolve() != (Path.home() / ".muser").resolve()
