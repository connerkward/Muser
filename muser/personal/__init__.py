"""Hidden Google-Photos *personal* sub-tool — completely isolated from the aesthetic library.

This package triages a Google Takeout export into **personal / in-between /
reference** buckets. It reuses Muser's core (embedder, LanceDB index, score facets,
captions, the mainline `faces` facet) but runs under its **own data root**
``~/.muser-personal`` so nothing it indexes ever touches the aesthetic library.

Because ``paths.MUSER_HOME`` is resolved at import time from the ``MUSER_HOME`` env
var, the personal commands must run in a process that has that env set *before* any
heavy module imports. `ensure_personal_root()` guarantees that by **re-exec**-ing the
process with the env set — so import order can never leak the wrong root.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: The isolated data root for everything personal. Disjoint from ~/.muser.
PERSONAL_HOME = Path.home() / ".muser-personal"

#: Default Takeout location (the user's export). Override with --src.
DEFAULT_TAKEOUT = Path.home() / "dev" / "g-takeout" / "google-takeout" / "Takeout" / "Google Photos"


def register_heic() -> None:
    """Make PIL open iPhone .heic/.heif Takeout files (no-op if pillow-heif absent)."""
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except Exception:
        pass


def ensure_personal_root() -> None:
    """Re-exec this process with MUSER_HOME=~/.muser-personal if not already set.

    Idempotent: once we're running under the personal root we just return. The
    re-exec replaces the current process (os.execv), so after it returns to the
    command body, every ``from .paths import MUSER_HOME`` resolves to the personal
    root. This is what keeps the sub-tool's index/scores/facets/captions isolated.
    """
    want = str(PERSONAL_HOME)
    if os.environ.get("MUSER_HOME") == want:
        register_heic()
        return
    env = dict(os.environ)
    env["MUSER_HOME"] = want
    PERSONAL_HOME.mkdir(parents=True, exist_ok=True)
    os.execvpe(sys.argv[0], sys.argv, env)  # replaces the process; never returns
