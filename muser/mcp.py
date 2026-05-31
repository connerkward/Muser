"""Muser MCP server — search your indexed images from inside Claude / agents.

This is a THIN HTTP CLIENT of the embedded Muser service (`muser serve`, see
``muser/service.py``). The service holds the embedding model WARM and owns the
LanceDB index; this process never loads a model. If the service isn't already
listening on http://127.0.0.1:7777 we auto-spawn it (detached) and wait until
``GET /api/status`` responds — mirroring the pattern the CLI uses.

Tools
-----
- ``search_images(query, k=24, folder=None)`` — semantic image search, optionally
  scoped to images under ``folder``. Returns a text ranking (path, name, score)
  plus one base64 JPEG thumbnail per hit as image content, so any host that
  renders ImageContent (Claude Desktop, etc.) shows a gallery.
- ``index_folder(folder, recursive=True)`` — index a folder into the live index.
- ``index_info()`` — model, indexed count, db path, available models.

Run
---
- ``muser-mcp``            → stdio transport (default; for Claude Desktop / agents)
- ``muser-mcp --http``     → streamable-http transport on http://127.0.0.1:8000/mcp

UI / "MCP app" path (researched 2026-05, NOT shipped here, intentionally)
------------------------------------------------------------------------
The original prototype (``legacy-ts/src/server.ts``) rendered an interactive
gallery via ``@modelcontextprotocol/ext-apps`` (a ``ui://`` HTML resource the
host loads in a sandboxed iframe). That ext-apps / Apps-SDK UI story is still
TS-first. The Python port (``mcp-ui`` on PyPI) is v0.1.2 and self-classified
"Development Status :: 3 - Alpha" — too immature to depend on. Rather than force
it, we return base64 ImageContent thumbnails, which mature hosts render directly
as a result gallery with zero extra deps. When the Python apps-UI SDK stabilizes,
add a ``ui://muser/search.html`` resource (reuse ``muser/web/app.html``) and tag
``search_images`` with ``_meta={"ui": {"resourceUri": ...}}`` — the structured
results this tool already returns are the exact shape that UI would consume.
"""

from __future__ import annotations

import argparse
import base64
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request

from mcp.server.fastmcp import FastMCP, Image

SERVICE_URL = os.environ.get("MUSER_SERVICE_URL", "http://127.0.0.1:7777")
_SPAWN_TIMEOUT = 120.0  # first start loads the model; can take a while

mcp = FastMCP("muser")


# --------------------------------------------------------------------------- #
# Thin HTTP client of the embedded service (muser/service.py)
# --------------------------------------------------------------------------- #
def _get(path: str, timeout: float = 30.0) -> bytes:
    with urllib.request.urlopen(f"{SERVICE_URL}{path}", timeout=timeout) as r:
        return r.read()


def _get_json(path: str, timeout: float = 30.0):
    import json

    return json.loads(_get(path, timeout))


def _post_json(path: str, payload: dict, timeout: float = 600.0):
    import json

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{SERVICE_URL}{path}", data=data, headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _service_up() -> bool:
    try:
        _get_json("/api/status", timeout=2.0)
        return True
    except Exception:
        return False


def ensure_service() -> None:
    """Auto-spawn `muser serve` (detached) if the embedded service is down."""
    if _service_up():
        return
    # Spawn the service in its own session so it outlives this MCP process.
    subprocess.Popen(
        [sys.executable, "-c", "from muser.service import serve; serve()"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + _SPAWN_TIMEOUT
    while time.time() < deadline:
        if _service_up():
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"muser service did not become ready at {SERVICE_URL} within "
        f"{_SPAWN_TIMEOUT:.0f}s (model load may have failed)."
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def search_images(query: str, k: int = 24, folder: str | None = None) -> list:
    """Semantic image search over the indexed library.

    Finds images matching a natural-language description (e.g. "a diagram",
    "person on a snowy mountain") using the warm embedding model in the Muser
    service. Returns a ranked text list (path, name, score 0-1) followed by a
    base64 JPEG thumbnail of each hit, so the host can render a result gallery.

    Args:
        query: Natural-language description of the image to find.
        k: Number of results to return (default 24).
        folder: Optional absolute path — limit results to images under this
            directory (any depth). Omit to search the whole library.
    """
    ensure_service()
    params = {"q": query, "k": k}
    if folder:
        params["folder"] = folder
    data = _get_json(f"/api/search?{urllib.parse.urlencode(params)}")
    results = data.get("results", [])

    if not results:
        where = f" under {folder}" if folder else ""
        return [f'No matches for "{query}"{where}. Is the library indexed for the current model?']

    scope = f" in {folder}" if folder else ""
    lines = [f'Top {len(results)} matches for "{query}"{scope} (model: {data.get("model")}):']
    content: list = []
    for i, h in enumerate(results, 1):
        lines.append(f'{i}. {h["name"]}  —  {h["score"]:.3f}\n   {h["path"]}')
    content.append("\n".join(lines))

    # Append one thumbnail per hit as image content (rendered by capable hosts).
    for h in results:
        try:
            tp = urllib.parse.urlencode({"path": h["path"], "size": 260})
            jpeg = _get(f"/api/thumb?{tp}", timeout=15.0)
            content.append(Image(data=jpeg, format="jpeg"))
        except Exception:
            # Skip thumbnails that fail to render; text result still stands.
            continue
    return content


@mcp.tool()
def index_folder(folder: str, recursive: bool = True) -> str:
    """Index a folder of images into the live Muser index.

    Adds/updates the folder's images (incremental by mtime) so they become
    searchable with `search_images`. Heavy work runs in the warm service.

    Args:
        folder: Absolute path to the image folder to index.
        recursive: Descend into subfolders (default True).
    """
    ensure_service()
    r = _post_json("/api/index", {"folder": folder, "recursive": recursive})
    return (
        f'Indexed {folder}: +{r["added"]} added, {r["updated"]} updated, '
        f'{r["removed"]} removed. {r["total"]} images now searchable.'
    )


@mcp.tool()
def index_info() -> str:
    """Report the current model, indexed image count, db path, and models available."""
    ensure_service()
    s = _get_json("/api/status")
    models = ", ".join(s.get("models", []))
    return (
        f'Model: {s.get("model")}\n'
        f'Indexed images (this model): {s.get("indexed")}\n'
        f'DB: {s.get("db")}\n'
        f'Available models: {models}'
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Muser MCP server (thin client of `muser serve`).")
    parser.add_argument(
        "--http", action="store_true", help="Serve over streamable-http instead of stdio."
    )
    args = parser.parse_args()
    mcp.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
