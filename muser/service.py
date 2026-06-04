"""Muser embedded service — one warm process that owns the model + index.

`muser serve` loads the embedding model once and keeps it resident, owns the
LanceDB index, and serves both a JSON API and a browser search UI. CLI/MCP become
thin HTTP clients of this. Local-only by default (binds 127.0.0.1).
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

from pydantic import BaseModel

from .index import MetaFilter, MuserIndex, uid_for
from .registry import DEFAULT_MODEL, load_model, model_names

WEB = Path(__file__).resolve().parent / "web" / "app.html"


class IndexReq(BaseModel):
    folder: str
    recursive: bool = True


class PathReq(BaseModel):
    path: str


class ModelReq(BaseModel):
    name: str


class CartItem(BaseModel):
    # Per-cart-item flag pair used by /api/zip. `upscale=True` runs the file
    # through muser/upscale.py (4x UltraSharp) and bundles the upscaled bytes
    # under the original filename (with a .jpg extension); `upscale=False`
    # bundles the file verbatim.
    path: str
    upscale: bool = False


class CartReq(BaseModel):
    # Two accepted shapes:
    #   - new:    {items: [{path, upscale}, ...]}     -- preferred, supports flags
    #   - legacy: {paths: [...]}                       -- pre-flag clients (CLI, MCP)
    # When both are present, `items` wins; when only `paths` is sent, every
    # item is treated as `upscale=false`.
    paths: list[str] | None = None
    items: list[CartItem] | None = None


class CaptionWriteReq(BaseModel):
    path: str
    caption: str


class BulkCaptionReq(BaseModel):
    paths: list[str]
    force: bool = False


class State:
    def __init__(self, model: str):
        self.model_name = model
        self.embedder = None
        self.index = MuserIndex()
        # label_index: lazy path → cluster-label cache for the refinement
        # chips on /api/search. Built once from ~/.muser/clusters.json.
        self.label_index = None
        # Background-task plumbing. Model warm runs in a thread so uvicorn
        # accepts connections immediately and the page can render its
        # "loading model" overlay instead of hanging at connect. Indexing
        # runs the same way so the user gets streamed progress.
        # `task` is the single in-flight long-running activity reported via
        # /api/status: {kind:"loading_model", model} | {kind:"indexing",
        # folder, done, total} | {kind:"indexing_done", added, updated,
        # total} | {kind:"indexing_error", error} | None.
        self._ready = threading.Event()
        self._lock = threading.Lock()
        self.task: dict | None = None
        # C2PA library-scan progress for the "AI" tab. Separate from `task` so the
        # long scan doesn't trip the model/index busy overlay. Polled by /api/ai.
        self.c2pa = {"scanning": False, "done": 0, "total": 0, "found": 0}

    def warm(self):
        # Idempotent. Holds a lock so two concurrent first-callers don't both
        # call load_model.
        with self._lock:
            if self.embedder is None:
                self.embedder = load_model(self.model_name)
                self.embedder.embed_queries(["warm up"])  # trigger weight load
            self._ready.set()
            return self.embedder

    def wait_ready(self, timeout: float | None = None):
        # Called by request handlers before doing model-dependent work.
        # Blocks until warm() has populated self.embedder.
        if not self._ready.is_set():
            self._ready.wait(timeout=timeout)
        return self.embedder

    def start_warm(self):
        # Kick off background load. The caller returns immediately; the UI
        # polls /api/status to know when it's done.
        self._ready.clear()
        self.task = {"kind": "loading_model", "model": self.model_name}
        def _bg():
            try:
                self.warm()
            finally:
                # Clear only if we're still the loader task — an indexing
                # task that started later shouldn't be wiped out here.
                if self.task and self.task.get("kind") == "loading_model":
                    self.task = None
        threading.Thread(target=_bg, daemon=True, name="muser-warm").start()

    def set_model(self, name: str):
        with self._lock:
            self.model_name = name
            self.embedder = None
            self.label_index = None
        self.start_warm()


def _smart_crop_square(img, target: int):
    """Crop a square out of `img` framed around its most edge-dense window,
    then resize to target × target.

    For near-square images this collapses to ImageOps.fit (a centered crop).
    For ultra-wide / tall images we'd otherwise lose resolution: `_load_rgb`
    only caps the long side, so a 4000×1000 image at max_side=540 becomes
    540×135 — and the card's `object-fit: cover` crop then has only 135 source
    lines of vertical resolution to work with. By cropping to a square *first*
    (at full source resolution) and *then* resizing, the rendered card always
    gets the full target × target = ~540² source pixels.

    Auto-framing uses a cheap edge-energy proxy: |dx| + |dy| on a 400-px-max
    working copy, summed across the cropped axis, cumulative-sum trick to find
    the best window in O(n). No model, no extra deps. Works well on the kinds
    of imagery in this corpus (creative/AI-gen/photo); doesn't try to be
    face-aware. Cost: ~few ms per image; amortized to zero by the disk cache.
    """
    import numpy as np
    from PIL import Image, ImageOps

    W, H = img.size
    if abs(W - H) <= 4:  # already-square: just centered fit
        return ImageOps.fit(img, (target, target), Image.LANCZOS, centering=(0.5, 0.5))

    # Low-res working copy for the saliency pass — full-res is wasteful.
    work = img.copy()
    work.thumbnail((400, 400), Image.BILINEAR)
    wW, wH = work.size
    gray = np.asarray(work.convert("L"), dtype=np.int16)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    energy = (gx + gy).astype(np.float32)

    if wW > wH:  # landscape — slide horizontally
        col = energy.sum(axis=0)
        cum = np.concatenate(([0.0], col.cumsum()))
        win = wH
        scores = cum[win:] - cum[:-win]
        best_w = int(scores.argmax()) if scores.size else 0
        side = H  # crop a full-height square out of source
        x0 = int(round(best_w * (W / wW)))
        x0 = max(0, min(x0, W - side))
        y0 = 0
        crop = img.crop((x0, y0, x0 + side, y0 + side))
    else:  # portrait — slide vertically
        row = energy.sum(axis=1)
        cum = np.concatenate(([0.0], row.cumsum()))
        win = wW
        scores = cum[win:] - cum[:-win]
        best_h = int(scores.argmax()) if scores.size else 0
        side = W
        y0 = int(round(best_h * (H / wH)))
        y0 = max(0, min(y0, H - side))
        x0 = 0
        crop = img.crop((x0, y0, x0 + side, y0 + side))

    return crop.resize((target, target), Image.LANCZOS)


def _reveal(path: str):
    s = platform.system()
    if s == "Darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif s == "Windows":
        # Explorer's /select, needs backslashes; the web UI hands us forward slashes.
        subprocess.run(["explorer", f"/select,{os.path.normpath(path)}"], check=False)
    else:
        # GNOME/KDE file managers can highlight the file via the FileManager1 D-Bus API;
        # fall back to just opening the containing folder (xdg-open can't select).
        try:
            subprocess.run(
                ["dbus-send", "--session", "--print-reply",
                 "--dest=org.freedesktop.FileManager1", "/org/freedesktop/FileManager1",
                 "org.freedesktop.FileManager1.ShowItems",
                 f"array:string:file://{path}", "string:"],
                check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            try:
                subprocess.run(["xdg-open", os.path.dirname(path)], check=False, timeout=10)
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass


def _copy_image_to_clipboard(path: str) -> bool:
    """Put an image on the OS clipboard server-side (no browser needed).

    The web UI runs on the user's own machine, so the *service* can write straight to
    the system clipboard — sidestepping the browser's async Clipboard API, which is
    gated behind a secure context and therefore unavailable on http://*.local. This is
    what lets the reverse-image button (Lens) copy-then-paste work regardless of which
    hostname the page is served from. Returns False when no OS clipboard tool is found.
    """
    import shutil
    import tempfile

    from .embedders import _load_rgb

    img = _load_rgb(path, max_side=2048)  # PNG for a lossless paste; cap pathological sizes
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        img.save(tmp.name, "PNG")
        tmp.close()
        s = platform.system()
        try:
            if s == "Darwin":
                r = subprocess.run(
                    ["osascript", "-e", f'set the clipboard to (read (POSIX file "{tmp.name}") as «class PNGf»)'],
                    capture_output=True, text=True,
                )
                return r.returncode == 0
            if s == "Windows":
                # PowerShell + WinForms (no extra deps). -STA is required for the clipboard apartment.
                ps = (
                    "Add-Type -AssemblyName System.Windows.Forms,System.Drawing; "
                    f"$img=[System.Drawing.Image]::FromFile('{tmp.name}'); "
                    "[System.Windows.Forms.Clipboard]::SetImage($img); $img.Dispose()"
                )
                r = subprocess.run(["powershell", "-NoProfile", "-STA", "-Command", ps], capture_output=True, text=True)
                return r.returncode == 0
            # Linux: Wayland (wl-copy) or X11 (xclip); both read image/png from stdin.
            data = open(tmp.name, "rb").read()
            if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
                cmd = ["wl-copy", "--type", "image/png"]
            elif shutil.which("xclip"):
                cmd = ["xclip", "-selection", "clipboard", "-t", "image/png"]
            else:
                return False
            # Both tools FORK a daemon to keep owning the selection. Must NOT capture
            # stdout/stderr — the daemon inherits the pipe and never closes it, so a
            # PIPE-based run() would block on EOF forever. DEVNULL + timeout avoids that.
            return subprocess.run(
                cmd, input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15
            ).returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    finally:
        os.unlink(tmp.name)


def create_app(model: str = DEFAULT_MODEL):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, Response

    state = State(model)
    app = FastAPI(title="Muser")

    @app.on_event("startup")
    def _startup():
        # Pull OPENAI_API_KEY (and any other shared secrets) from the central .env
        # before anything that might need them — caption.py does the same at import,
        # but calling here too keeps the service self-contained if started standalone.
        from .caption import _load_env_file
        _load_env_file()
        # Preload the C2PA verdict cache from ~/.muser/c2pa.json so per-result
        # AI-badge enrichment and /api/c2pa lookups serve from RAM instead of
        # shelling out to c2patool per request. ~27k entries → few-hundred-ms
        # startup cost, ~6 seconds removed from every search render.
        from . import c2pa as _c2pa
        primed = _c2pa.prime_cache_from_sidecar()
        if primed:
            print(f"  primed c2pa cache: {primed} entries", flush=True)
        # Non-blocking: fire the warm in a thread so uvicorn starts accepting
        # connections in milliseconds. The page can load, see task=loading_model
        # in /api/status, and render its overlay while weights load.
        print(f"  loading model {state.model_name} (background) ...", flush=True)
        state.start_warm()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return WEB.read_text()

    # SVG favicon: solid-fill knockout mark — italic lowercase 'm' carved
    # out of a charcoal square. Single-character favicons need the solid
    # field to carry weight at 16px; an outlined glyph alone reads as a
    # squiggle at tab size. The wordmark connection still holds because
    # the glyph is the same italic lowercase 'm' used in the header.
    # prefers-color-scheme inverts fg/bg so the mark stays legible against
    # both light and dark browser chrome (Safari + Chromium honor the
    # media query inside SVG favicons).
    _FAVICON_SVG = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
        '<style>'
        '.bg{fill:#1a1815}.fg{fill:#f5f1e8}'
        '@media(prefers-color-scheme:dark){.bg{fill:#ece6d6}.fg{fill:#16140f}}'
        '</style>'
        '<rect width="32" height="32" rx="4" class="bg"/>'
        '<text x="16" y="16" font-family="Georgia,serif" font-style="italic" '
        'font-size="24" text-anchor="middle" dominant-baseline="central" '
        'class="fg">m</text>'
        '</svg>'
    ).encode("utf-8")

    @app.get("/favicon.ico")
    def favicon():
        return Response(_FAVICON_SVG, media_type="image/svg+xml",
                        headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/api/status")
    def status():
        # `ready` flips true once the model is warm. `task` is the single
        # in-flight long-running activity (loading_model / indexing) for the
        # busy overlay; null when idle. Frontend polls this — 4s idle,
        # ~400ms when task != null.
        # `metadata` is the (scored, total) coverage for the Filter panel —
        # cheap (a path-only scan), so it can ride the same poll the busy
        # overlay already uses without a second round-trip.
        return {
            "model": state.model_name,
            "models": model_names(),
            "indexed": state.index.count(state.model_name),
            "db": str(state.index.db_path),
            "ready": state._ready.is_set(),
            "task": state.task,
            "metadata": state.index.metadata_coverage(state.model_name),
        }

    @app.post("/api/index")
    def do_index(req: IndexReq):
        # Returns immediately; the actual indexing runs in a thread and
        # publishes progress through state.task so the busy overlay can
        # show done/total. UI polls /api/status to follow along.
        folder = os.path.expanduser(req.folder)
        if not os.path.isdir(folder):
            raise HTTPException(400, f"not a folder: {folder}")
        if state.task is not None:
            raise HTTPException(409, f"busy: {state.task.get('kind')}")
        # Need the model warm to embed during indexing.
        state.wait_ready()

        state.task = {"kind": "indexing", "folder": folder, "done": 0, "total": 0}

        def on_progress(done: int, total: int):
            # Keep the dict identity stable so the polling JSON updates in place.
            t = state.task
            if t and t.get("kind") == "indexing":
                t["done"] = done
                t["total"] = total

        def _bg():
            ok = False
            try:
                res = state.index.index_folder(
                    folder, state.model_name, state.embedder,
                    recursive=req.recursive, on_progress=on_progress,
                )
                state.task = {
                    "kind": "indexing_done", "folder": folder,
                    "added": res.added, "updated": res.updated,
                    "removed": res.removed, "total": res.total,
                }
                ok = True
            except Exception as e:
                state.task = {"kind": "indexing_error", "folder": folder, "error": str(e)}
            # Auto-clear the terminal state after a short window so the next
            # index attempt isn't blocked by the 409 "busy" check.
            threading.Timer(4.0, lambda: setattr(state, "task", None)).start()
            # Then auto-run the C2PA AI detector over the just-indexed files. Runs
            # inline (this thread is already a daemon) AFTER the task auto-clears, so
            # the busy overlay isn't held by the scan; progress shows in the AI tab.
            # No-op if c2patool is absent or a scan is already in flight; incremental,
            # so only genuinely new/changed files spawn a subprocess.
            if ok:
                _scan_c2pa(folder)

        threading.Thread(target=_bg, daemon=True, name="muser-index").start()
        return {"started": True, "folder": folder}

    def _scan_c2pa(folder: str | None = None):
        # Shared by the post-index hook and POST /api/ai/scan. `folder=None` scans the
        # whole index; a folder restricts to that subtree (post-index path).
        from . import c2pa as c2
        if not c2.available() or state.c2pa.get("scanning"):
            return
        paths = state.index.paths(state.model_name, under=folder)
        if not paths:
            return
        state.c2pa = {"scanning": True, "done": 0, "total": len(paths), "found": 0}

        def _prog(done, total, found):
            state.c2pa.update(done=done, total=total, found=found)
        try:
            c2.scan(paths, progress=_prog)
        finally:
            state.c2pa["scanning"] = False

    # ---- path → cluster-label map, used for the post-search refinement chips ----
    # Reads ~/.muser/clusters.json once and caches a path → cluster-label dict.
    # No embedding pass — the only consumer is _refinements, which does pure
    # bucket-counting on result paths.
    def _path_to_label():
        if state.label_index is not None:
            return state.label_index
        clusters_file = Path.home() / ".muser" / "clusters.json"
        if not clusters_file.exists():
            state.label_index = {}
            return state.label_index
        c = json.loads(clusters_file.read_text())
        m_name = "hdbscan" if "hdbscan" in c["methods"] else next(iter(c["methods"]))
        m = c["methods"][m_name]
        path_to_label = {}
        for cl in m["clusters"]:
            if cl["id"] == -1:  # skip "misc / unclustered"
                continue
            for p in m["members"].get(str(cl["id"]), []):
                path_to_label[p] = cl["label"]
        state.label_index = path_to_label
        return state.label_index

    def _refinements(result_paths, query):
        # Dense-vector pseudo-relevance feedback: which clusters do the top
        # results belong to? Most-common labels (minus any that effectively
        # ARE the query) become the "you might also try" chips.
        from collections import Counter
        ptl = _path_to_label()
        if not ptl:
            return []
        cnts = Counter()
        for p in result_paths[:20]:
            lbl = ptl.get(p)
            if lbl:
                cnts[lbl] += 1
        q_lower = query.lower().strip()
        out = []
        for lbl, cnt in cnts.most_common():
            ll = lbl.lower()
            if q_lower and (q_lower in ll or ll in q_lower):
                continue
            out.append({"label": lbl, "count": cnt})
            if len(out) >= 5:
                break
        return out

    def _meta_from_params(
        min_width=None, max_width=None, min_height=None, max_height=None,
        min_short_side=None, max_short_side=None, min_long_side=None, max_long_side=None,
        min_aspect=None, max_aspect=None, min_filesize=None, max_filesize=None,
    ) -> MetaFilter | None:
        """Build a MetaFilter from the search/filter endpoint's optional bounds,
        or None if every bound is unset. Returning None lets the index layer
        skip building (and SQL-evaluating) an empty where-clause."""
        f = MetaFilter(
            min_width=min_width, max_width=max_width,
            min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        return None if f.is_empty() else f

    def _attach_dims(results, paths):
        """Enrich result dicts with width/height/filesize (when known) so the
        UI can show a "1920×1080 · 540 KB" line without a per-card round-trip.
        Pulled in one LanceDB scan keyed on path. Missing metadata → keys
        simply absent on that result."""
        if not results:
            return
        t = state.index._open(state.model_name)
        if t is None or "width" not in {f.name for f in t.schema}:
            return
        # `path IN ('a','b',...)` is the cleanest predicate; SQL-quote each.
        quoted = ",".join("'" + p.replace("'", "''") + "'" for p in paths)
        try:
            rows = (
                t.search().select(["path", "width", "height", "filesize"])
                .where(f"path IN ({quoted})")
                .limit(len(paths)).to_list()
            )
        except Exception:
            return
        by_path = {r["path"]: r for r in rows}
        for r in results:
            row = by_path.get(r["path"])
            if not row:
                continue
            if row.get("width") is not None:
                r["width"] = int(row["width"])
            if row.get("height") is not None:
                r["height"] = int(row["height"])
            if row.get("filesize") is not None:
                r["filesize"] = int(row["filesize"])

    def _run_search(qv, k, dedup, method, folder, meta=None):
        if dedup:
            results = state.index.search_dedup(
                state.model_name, qv, k=k, method=method, folder=folder, meta=meta,
            )
        else:
            results = [
                {"path": p, "name": os.path.basename(p), "score": round(s, 4), "dupes": [p], "dupe_count": 1}
                for p, s in state.index.search(state.model_name, qv, k=k, folder=folder, meta=meta)
            ]
        # Drop results whose canonical path no longer exists on disk, and prune
        # dead entries from each result's `dupes[]`. The LanceDB index is
        # append-by-mtime so deleted files linger in the table until the next
        # full re-index; filtering at query time keeps stale hits out of the
        # UI without forcing a rebuild. os.path.exists() is ~10 µs/file on SSD
        # — 24 results × ~5 dupes ≈ 120 stats = ~1 ms, negligible.
        kept = []
        for r in results:
            if not os.path.exists(r["path"]):
                continue
            d = r.get("dupes") or [r["path"]]
            live = [p for p in d if os.path.exists(p)]
            if not live:
                continue
            r["dupes"] = live
            r["dupe_count"] = len(live)
            kept.append(r)
        results = kept
        for r in results:
            r["uid"] = uid_for(r["path"])
        _add_prob(results)
        _attach_dims(results, [r["path"] for r in results])
        _attach_scores(results)
        _attach_c2pa(results)
        return results

    # Cache the parsed scores.json by mtime so the Search-tab sort-blend join
    # is one disk read per scores.json update, not one per request.
    _scores_cache: dict = {"mtime": -1.0, "map": {}}
    _scores_lock = threading.Lock()

    def _attach_scores(results):
        # Surface per-image aesthetic / pickscore from ~/.muser/scores.json on each
        # search hit so the Search-tab "Sort blend" control can re-rank client-side
        # (vec + aesthetic_v2 + pickscore) without re-fetching. Missing entries
        # → key simply absent.
        if not results:
            return
        scores_path = Path.home() / ".muser" / "scores.json"
        try:
            mt = scores_path.stat().st_mtime
        except OSError:
            return
        with _scores_lock:
            if mt != _scores_cache["mtime"]:
                try:
                    s = json.loads(scores_path.read_text())
                    _scores_cache["map"] = s.get("scores", {})
                    _scores_cache["mtime"] = mt
                except Exception:
                    _scores_cache["map"] = {}
                    _scores_cache["mtime"] = mt
            m = _scores_cache["map"]
        for r in results:
            row = m.get(r["path"])
            if not row:
                continue
            if "aesthetic_v2" in row:
                r["aesthetic_v2"] = round(float(row["aesthetic_v2"]), 4)
            if "pickscore" in row:
                r["pickscore"] = round(float(row["pickscore"]), 4)

    def _attach_c2pa(results):
        # Inline the C2PA verdict per result so the Search-tab UI doesn't fire
        # one /api/c2pa fetch per card (24 results × ~5 dupes ≈ 120 round-trips
        # of ~50ms each = 6 seconds of trailing network on every search).
        #
        # The verdict cache is primed from ~/.muser/c2pa.json at service
        # startup; this is a RAM-only hashmap lookup. Set `c2pa` on every
        # result the sidecar knew about (positive AND negative) so the client
        # can render synchronously without a fetch. Misses (file changed since
        # scan, or no sidecar entry) → key simply absent and the client
        # falls back to the direct endpoint for those few.
        from . import c2pa as _c2pa
        for r in results:
            v = _c2pa.lookup(r["path"])
            if v is None:
                continue
            r["c2pa"] = {"ai": bool(v.get("ai")), "kind": v.get("kind"), "tool": v.get("tool")}

    def _add_prob(results):
        # Attach a calibrated match probability (SigLIP sigmoid) per result so the UI's
        # "%" is a real confidence, not a flat raw cosine. No-op for models without a
        # per-pair probability — the UI falls back to cosine then.
        cal = getattr(state.embedder, "calibrate", None)
        if not results or cal is None:
            return
        probs = cal([r["score"] for r in results])
        if probs is None:
            return
        for r, p in zip(results, probs):
            r["prob"] = round(float(p), 4)

    @app.get("/api/search")
    def search(q: str, k: int = 24, dedup: bool = True, method: str = "embed", folder: str | None = None,
               neg: str | None = None, neg_strength: float = 0.5,
               min_width: int | None = None, max_width: int | None = None,
               min_height: int | None = None, max_height: int | None = None,
               min_short_side: int | None = None, max_short_side: int | None = None,
               min_long_side: int | None = None, max_long_side: int | None = None,
               min_aspect: float | None = None, max_aspect: float | None = None,
               min_filesize: int | None = None, max_filesize: int | None = None):
        # method: "embed" (cosine, default), "phash" (perceptual-hash Hamming,
        # robust to recompression/resize/crop), or "both" (collapse on either).
        # folder: restrict results to images under this directory (any depth).
        # neg: optional text describing concepts to push away from. CLIP/SigLIP
        # encoders ignore in-prompt negation (bag-of-words effect, see
        # Yuksekgonul et al. ICLR 2023), so suppression has to happen as
        # vector arithmetic in the embedding space, not as natural-language "not".
        # min_/max_ {width,height,short_side,long_side,aspect,filesize}: optional
        # metadata constraints — pushed into LanceDB as a single prefilter so
        # the kNN limit applies AFTER the cut (otherwise the top-k of out-of-
        # filter neighbors could return zero in-filter hits).
        emb = state.warm()
        qv = emb.embed_queries([q])[0]
        if neg and neg.strip():
            import numpy as np
            nv = emb.embed_queries([neg])[0]
            qv = qv - neg_strength * nv
            n = float(np.linalg.norm(qv))
            if n > 0:
                qv = qv / n
        meta = _meta_from_params(
            min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        results = _run_search(qv, k, dedup, method, folder, meta=meta)
        refinements = _refinements([r["path"] for r in results], q)
        return {"query": q, "model": state.model_name, "results": results, "refinements": refinements}

    @app.get("/api/search-image")
    def search_image(path: str, k: int = 24, dedup: bool = True, method: str = "embed", folder: str | None = None,
                     min_width: int | None = None, max_width: int | None = None,
                     min_height: int | None = None, max_height: int | None = None,
                     min_short_side: int | None = None, max_short_side: int | None = None,
                     min_long_side: int | None = None, max_long_side: int | None = None,
                     min_aspect: float | None = None, max_aspect: float | None = None,
                     min_filesize: int | None = None, max_filesize: int | None = None):
        # Image-to-image search: embed the file at `path`, find nearest neighbors
        # in the index. Works for any file PIL can open — doesn't need to be indexed.
        if not os.path.exists(path):
            raise HTTPException(404, f"no such image: {path}")
        emb = state.warm()
        qv = emb.embed_images([path])[0]
        meta = _meta_from_params(
            min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        results = _run_search(qv, k, dedup, method, folder, meta=meta)
        # Filter the reference image itself out of its own results.
        results = [r for r in results if r["path"] != path]
        refinements = _refinements([r["path"] for r in results], os.path.basename(path))
        return {"query": f"image: {os.path.basename(path)}", "ref_path": path,
                "model": state.model_name, "results": results, "refinements": refinements}

    @app.get("/api/search-compose")
    def search_compose(img: str, text: str, alpha: float = 0.5, beta: float = 0.5,
                       k: int = 30, dedup: bool = True, method: str = "embed",
                       folder: str | None = None,
                       min_width: int | None = None, max_width: int | None = None,
                       min_height: int | None = None, max_height: int | None = None,
                       min_short_side: int | None = None, max_short_side: int | None = None,
                       min_long_side: int | None = None, max_long_side: int | None = None,
                       min_aspect: float | None = None, max_aspect: float | None = None,
                       min_filesize: int | None = None, max_filesize: int | None = None):
        # Composed Image Retrieval, vector-arithmetic baseline (CIRR/Liu et al. ICCV 2021):
        #   q = α·embed_image(img) + β·embed_text(text)  (then L2-normalize)
        # Works well for attribute swaps ("red dress" + "in blue"); weaker on structural
        # mods. α=1,β=0 ≡ /api/search-image; α=0,β=1 ≡ /api/search. No new model load —
        # reuses the warm shared-space embedder.
        import numpy as np
        if not os.path.isfile(img):
            raise HTTPException(404, f"no such image: {img}")
        if not text or not text.strip():
            raise HTTPException(400, "text modification is required")
        emb = state.wait_ready()
        iv = emb.embed_images([img])[0]
        tv = emb.embed_queries([text])[0]
        qv = alpha * iv + beta * tv
        n = float(np.linalg.norm(qv))
        if not np.isfinite(n) or n == 0:
            raise HTTPException(400, "composed vector degenerate (α=β=0?)")
        qv = qv / n
        meta = _meta_from_params(
            min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        results = _run_search(qv, k, dedup, method, folder, meta=meta)
        results = [r for r in results if r["path"] != img]
        refinements = _refinements([r["path"] for r in results], text)
        return {"query": f"compose: {os.path.basename(img)} + {text!r}",
                "compose": {"img": img, "text": text, "alpha": alpha, "beta": beta},
                "model": state.model_name, "results": results, "refinements": refinements}

    @app.get("/api/filter")
    def filter_(limit: int = 200, offset: int = 0, folder: str | None = None,
                dedup: bool = True,
                min_width: int | None = None, max_width: int | None = None,
                min_height: int | None = None, max_height: int | None = None,
                min_short_side: int | None = None, max_short_side: int | None = None,
                min_long_side: int | None = None, max_long_side: int | None = None,
                min_aspect: float | None = None, max_aspect: float | None = None,
                min_filesize: int | None = None, max_filesize: int | None = None):
        """List images matching the metadata constraints — no text/image query.
        Drives the Filter panel's "Apply with no query" path: e.g. "find every
        image whose short side is under 768" for an upscale-candidates triage.
        Same bound semantics as the metadata params on /api/search; results
        carry (path, uid, name, width, height, filesize). When ``dedup`` is on
        (default), near-duplicate paths are collapsed via the persisted
        canonical map (~/.muser/scores.json), so the page mirrors the Search
        tab's "one card per unique image" idiom; set ``dedup=false`` for the
        raw row dump."""
        meta = _meta_from_params(
            min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        # Fetch a wide-enough rows window for the canonical-only collapse to
        # converge in one pass. The index layer returns all matches with the
        # given bounds (sorted by path), so we slice on this side after the
        # canonical filter.
        all_rows, raw_total = state.index.filter(
            state.model_name, meta=meta, folder=folder, limit=10_000_000, offset=0,
        )

        if dedup:
            # Map every path back to its canonical (deduped) representative;
            # the filter ships one row per representative when multiple of its
            # dupes land in the filtered set. Falls back to "no dedup" when
            # scores.json hasn't been built — better than failing silently.
            scores_path = Path.home() / ".muser" / "scores.json"
            try:
                canonical_meta = json.loads(scores_path.read_text()) if scores_path.exists() else None
            except Exception:
                canonical_meta = None
            canon_set: set[str] | None = None
            dup_to_canon: dict[str, str] = {}
            if canonical_meta:
                canon_list = canonical_meta.get("canonical") or []
                dupes_map: dict[str, list[str]] = canonical_meta.get("dupes") or {}
                canon_set = set(canon_list)
                for c, ds in dupes_map.items():
                    for d in ds:
                        dup_to_canon[d] = c
            if canon_set:
                # Each canonical that has at least one of its members in
                # all_rows survives; we prefer the canonical row itself when
                # present, else the first dupe (path-sorted upstream).
                by_canon: dict[str, dict] = {}
                for r in all_rows:
                    c = r["path"] if r["path"] in canon_set else dup_to_canon.get(r["path"])
                    if c is None:
                        # Unscored (no canonical recorded) → treat as its own.
                        c = r["path"]
                    if c not in by_canon or r["path"] == c:
                        by_canon[c] = r
                rows = sorted(by_canon.values(), key=lambda r: r["path"])
            else:
                rows = all_rows
        else:
            rows = all_rows

        total = len(rows)
        page = rows[offset : offset + limit]
        results = []
        for r in page:
            results.append({
                "path": r["path"],
                "uid": uid_for(r["path"]),
                "name": os.path.basename(r["path"]),
                "width": r.get("width"),
                "height": r.get("height"),
                "filesize": r.get("filesize"),
                "dupe_count": 1, "dupes": [r["path"]],
            })
        return {
            "model": state.model_name,
            "results": results,
            "total": total,
            "raw_total": raw_total,
            "indexed": state.index.count(state.model_name),
            "offset": offset, "limit": limit,
        }

    @app.post("/api/backfill-metadata")
    def backfill_metadata():
        """Compute width/height/filesize for every indexed row missing them.
        Returns immediately; the scan runs in a thread and publishes progress
        via ``state.task`` (kind = ``backfill_metadata``) so the UI's busy
        overlay can show N/M. ~10 s wall-clock for a 27 k-image corpus."""
        if state.task is not None:
            raise HTTPException(409, f"busy: {state.task.get('kind')}")
        state.task = {"kind": "backfill_metadata", "done": 0, "total": 0}

        def on_progress(done: int, total: int):
            t = state.task
            if t and t.get("kind") == "backfill_metadata":
                t["done"] = done
                t["total"] = total

        def _bg():
            try:
                res = state.index.backfill_metadata(state.model_name, on_progress=on_progress)
                state.task = {"kind": "backfill_metadata_done", **res}
            except Exception as e:
                state.task = {"kind": "backfill_metadata_error", "error": str(e)}
            threading.Timer(4.0, lambda: setattr(state, "task", None)).start()

        threading.Thread(target=_bg, daemon=True, name="muser-backfill-meta").start()
        return {"started": True}

    @app.get("/api/folders")
    def folders():
        # uid the folder *path* itself so the picker has stable React-style keys
        # (same hash function — a folder is just another path string).
        items = state.index.folders(state.model_name)
        for it in items:
            it["uid"] = uid_for(it["folder"])
        return {"folders": items}

    @app.post("/api/zip")
    def cart_zip(req: CartReq):
        # Bundles the requested files into a ZIP. Filenames collide on basename;
        # we disambiguate with a numeric suffix ('_2', '_3', ...). Files are
        # stored without compression — most are already-compressed image
        # formats, so deflate would burn CPU for ~0% gain. In-memory build
        # (not streaming) because ZipFile's central directory writes back into
        # the buffer at the end, which streaming-by-truncation breaks.
        #
        # For LoRA training (fal flux-lora-fast-training / kohya), each image
        # also gets a sibling `<stem>.txt` containing its caption — same stem
        # as the (possibly disambiguated) image file. Images without a cached
        # caption ship plain (no .txt); we surface the missing count in the
        # `X-Captions-Missing` response header so the UI can warn.
        #
        # Items flagged `upscale=true` are routed through muser/upscale.py
        # (4x UltraSharp) — original bytes are NOT bundled, only the upscaled
        # JPEG. The arcname keeps the original stem but forces ``.jpg``
        # extension (lossless source extensions like PNG/AVIF would be
        # misleading otherwise). The upscaler caches per (path, mtime), so a
        # second zip of the same cart returns in milliseconds.
        import io, zipfile
        from fastapi.responses import Response

        # Normalize input shapes. The web UI sends `items`; older CLI/MCP
        # clients still send `paths`. Treat absent flags as upscale=False.
        if req.items:
            in_items = [(it.path, bool(it.upscale)) for it in req.items]
        elif req.paths:
            in_items = [(p, False) for p in req.paths]
        else:
            raise HTTPException(400, "request must include `items` or `paths`")

        captions = _load_captions()
        seen, members = {}, []
        for p, upscale in in_items:
            if not p or not os.path.isfile(p):
                continue
            base = os.path.basename(p)
            stem, ext = os.path.splitext(base)
            # Upscaled bytes are JPEG regardless of source format. PNG/AVIF
            # input keeps its stem but the extension flips to .jpg so the
            # downstream consumer doesn't get a PNG with JPEG bytes.
            out_ext = ".jpg" if upscale else ext
            n = seen.get(base, 0) + 1
            seen[base] = n
            arc_stem = stem if n == 1 else f"{stem}_{n}"
            arcname = f"{arc_stem}{out_ext}"
            members.append((p, arcname, arc_stem, upscale))
        if not members:
            raise HTTPException(400, "no valid files in cart")
        # Lazy-import so the spandrel + torch model load only happens when an
        # actual upscale is requested (don't pay the latency on plain zips).
        upscale_fn = None
        if any(u for _, _, _, u in members):
            from .upscale import upscale_4x as upscale_fn  # noqa: F811

        missing = 0
        upscaled_n = 0
        manifest: dict[str, dict] = {}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for src, arc, arc_stem, upscale in members:
                if upscale:
                    try:
                        data = upscale_fn(src)
                        zf.writestr(arc, data)
                        upscaled_n += 1
                    except Exception as e:
                        # Don't kill the whole zip if one item fails — bundle
                        # the original instead and tell the user via the
                        # X-Upscale-Errors header.
                        print(f"[upscale] failed for {src}: {e}", flush=True)
                        zf.write(src, arc)
                else:
                    zf.write(src, arc)
                cap = captions.get(src)
                if cap:
                    # UTF-8, no trailing newline — fal/kohya read the file verbatim
                    # as the training caption.
                    zf.writestr(f"{arc_stem}.txt", cap.encode("utf-8"))
                else:
                    missing += 1
                # Manifest: uid → {path, arcname, caption?, upscaled}. Lets a
                # downstream LoRA training run correlate every shipped
                # basename back to its original source on disk (and know
                # which items were upscaled, in case it cares).
                manifest[uid_for(src)] = {
                    "path": src,
                    "arcname": arc,
                    "caption": cap or None,
                    "upscaled": bool(upscale),
                }
            zf.writestr(
                "manifest.json",
                json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
            )
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="muser-cart.zip"',
                "X-Captions-Missing": str(missing),
                "X-Captions-Total": str(len(members)),
                "X-Upscaled-Count": str(upscaled_n),
            },
        )

    @app.post("/api/pick-folder")
    def pick_folder(kind: str = "scope"):
        # Pops a native folder picker on the server (== this machine, since
        # muser serve is local-only). Returns {path} or {cancelled: true}.
        # `kind` selects one of two fixed prompts — no string injection into AppleScript.
        prompts = {
            "scope": "Limit search to which folder?",
            "index": "Pick a folder to index",
        }
        prompt = prompts.get(kind, "Choose a folder")
        import shutil
        s = platform.system()
        try:
            if s == "Darwin":
                out = subprocess.run(
                    ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")'],
                    capture_output=True, text=True, timeout=300, check=True,
                ).stdout.strip().rstrip("/")
                return {"path": out}
            if s == "Windows":
                ps = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    f"$d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='{prompt}'; "
                    "if($d.ShowDialog() -eq 'OK'){[Console]::Out.Write($d.SelectedPath)}"
                )
                out = subprocess.run(
                    ["powershell", "-NoProfile", "-STA", "-Command", ps],
                    capture_output=True, text=True, timeout=300, check=True,
                ).stdout.strip()
                return {"path": out} if out else {"cancelled": True}
            # Linux: zenity (GTK) or kdialog (KDE).
            if shutil.which("zenity"):
                cmd = ["zenity", "--file-selection", "--directory", f"--title={prompt}"]
            elif shutil.which("kdialog"):
                cmd = ["kdialog", "--getexistingdirectory", os.path.expanduser("~")]
            else:
                raise HTTPException(501, "no folder picker found — install zenity or kdialog, "
                                         "or type a path into the scope box")
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=True).stdout.strip()
            return {"path": out} if out else {"cancelled": True}
        except subprocess.CalledProcessError:
            return {"cancelled": True}   # user cancelled the dialog
        except FileNotFoundError:
            raise HTTPException(501, "native folder picker unavailable on this platform — "
                                     "type a path into the scope box instead")

    # On-disk thumb cache. Keyed on (path, mtime, size) so editing a file
    # invalidates its cached thumbs but the same (path, size) for an
    # unchanged file always hits. Lives at ~/.muser/thumb_cache; delete
    # the dir to invalidate everything. Versioned key prefix so we can
    # bust the cache when the crop algorithm changes.
    THUMB_CACHE = Path.home() / ".muser" / "thumb_cache"
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)
    THUMB_KEY_VERSION = "v2-smartcrop"

    @app.get("/api/thumb")
    def thumb(path: str, size: int = 540):
        # Frontend asks for size=Math.round(280*devicePixelRatio); 540 is the
        # safe default for DPR=2. The thumb is a smart-cropped *square* (focal
        # window picked by edge energy), so the card's `object-fit: cover` is
        # already a no-op and ultra-wide / tall sources don't get squashed.
        if not os.path.isfile(path):
            raise HTTPException(404, "not found")
        try:
            mtime = int(os.path.getmtime(path))
        except OSError:
            raise HTTPException(404, "stat failed")
        import hashlib
        key = hashlib.blake2b(
            f"{THUMB_KEY_VERSION}|{path}|{mtime}|{size}".encode(),
            digest_size=12,
        ).hexdigest()
        cached = THUMB_CACHE / f"{key}.jpg"
        # The URL key changes when mtime or size changes, so we can be
        # aggressive with the browser cache. A week is generous.
        headers = {"Cache-Control": "public, max-age=604800, immutable"}
        if cached.exists():
            return FileResponse(cached, media_type="image/jpeg", headers=headers)
        from .embedders import _load_rgb
        try:
            # Decode at a working res that's big enough to crop a square from
            # the short side then resize to `size`. max_side = 2 * size covers
            # most aspect ratios; ultra-wide (e.g. 16:1) would still benefit
            # but the bandwidth cost of decoding the original is prohibitive.
            img = _load_rgb(path, max_side=max(size * 2, 1024))
            sq = _smart_crop_square(img, size)
            tmp = cached.with_suffix(".jpg.tmp")
            sq.save(tmp, "JPEG", quality=82, optimize=False)
            tmp.replace(cached)  # atomic swap so partial writes never serve
            return FileResponse(cached, media_type="image/jpeg", headers=headers)
        except Exception:
            raise HTTPException(404, "thumb failed")

    @app.get("/api/image")
    def image(path: str):
        if not os.path.isfile(path):
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.get("/api/ai")
    def ai_list():
        # The "AI" tab. Returns AI-flagged images from the persisted C2PA scan plus
        # live scan progress. Positive-only and a lower bound — only cloud generators
        # (OpenAI/Firefly/Google) ship Content Credentials; local SD/Flux carry none.
        from . import c2pa as c2
        avail = c2.available()
        # uid injected so each item can deep-link to the detail page (#/image/<uid>)
        # without the UI needing its own blake2b implementation.
        items = c2.ai_images() if avail else []
        for it in items:
            it["uid"] = uid_for(it["path"])
        return {
            "available": avail,
            "built": c2.cache_exists(),
            "items": items,
            **state.c2pa,
        }

    @app.post("/api/ai/scan")
    def ai_scan():
        # Kick off a background C2PA scan over every indexed path. Idempotent while
        # one is running. Progress lands in state.c2pa, surfaced via /api/ai.
        from . import c2pa as c2
        if not c2.available():
            raise HTTPException(501, "c2patool not installed — `brew install c2patool`")
        if state.c2pa["scanning"]:
            return {"started": False, "scanning": True}
        total = state.index.count(state.model_name)
        threading.Thread(target=_scan_c2pa, daemon=True, name="muser-c2pa-scan").start()
        return {"started": True, "total": total}

    @app.get("/api/c2pa")
    def c2pa(path: str):
        # Content Credentials provenance check. Returns {available, ai, kind, tool};
        # the UI shows an "AI?" badge only when ai is truthy. Deterministic, $0,
        # local — but positive-only: ai=False means "no provenance says AI", not
        # "confirmed real". No-op (available=False) when c2patool isn't installed.
        from .c2pa import verdict
        return verdict(path)

    @app.post("/api/reveal")
    def reveal(req: PathReq):
        _reveal(req.path)
        return {"ok": True}

    @app.post("/api/clipboard")
    def clipboard(req: PathReq):
        # Server-side image copy so Lens works on any origin (incl. http://muser.local),
        # where the browser's secure-context Clipboard API is unavailable. Cross-platform:
        # osascript (mac) / PowerShell (Windows) / wl-copy|xclip (Linux).
        if not os.path.isfile(req.path):
            raise HTTPException(404, "not found")
        if not _copy_image_to_clipboard(req.path):
            raise HTTPException(501, "server clipboard unavailable — on Linux install "
                                     "xclip (X11) or wl-clipboard (Wayland)")
        return {"ok": True}

    @app.post("/api/model")
    def set_model(req: ModelReq):
        # Switches the active model and kicks off the new warm in the
        # background. Returns immediately; UI polls /api/status and shows
        # the busy overlay until task=null.
        if req.name not in model_names():
            raise HTTPException(400, f"unknown model {req.name}")
        if state.task and state.task.get("kind") != "loading_model":
            raise HTTPException(409, f"busy: {state.task.get('kind')}")
        state.set_model(req.name)
        return {"started": True, "model": state.model_name}

    # ---- Explore: clusters (read ~/.muser/clusters.json, written by `muser cluster`) ----
    CLUSTERS = Path.home() / ".muser" / "clusters.json"

    def _clusters():
        return json.loads(CLUSTERS.read_text()) if CLUSTERS.exists() else None

    @app.get("/api/clusters")
    def clusters(method: str = "hdbscan"):
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"built": False, "clusters": []}
        m = c["methods"][method]
        return {
            "built": True, "method": method, "methods": list(c["methods"].keys()), "n": c["n"],
            "clusters": [
                {
                    "id": cl["id"], "label": cl["label"], "sublabel": cl["sublabel"],
                    "size": cl["size"],
                    # reps is a list of path strings; promote to {path, uid} objects
                    # so the UI can use uid as a stable React-style key.
                    "reps": [{"path": p, "uid": uid_for(p)} for p in cl["reps"]],
                }
                for cl in m["clusters"]
            ],
        }

    @app.get("/api/cluster")
    def cluster_members(method: str, id: int, offset: int = 0, limit: int = 80):
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"total": 0, "members": []}
        members = c["methods"][method]["members"].get(str(id), [])
        page = members[offset : offset + limit]
        return {
            "total": len(members),
            "members": [{"path": p, "name": os.path.basename(p), "uid": uid_for(p)} for p in page],
        }

    # ---- per-image captions: GPT-4o-mini + user edits (~/.muser/captions.jsonl) ----
    # JSONL is append-only. Multiple rows per path are kept on disk for history;
    # in memory we collapse to latest-wins by `ts` (so a user-edited row written
    # after the model's row beats it, even if the model row appears later in the
    # file). Cached by file mtime; the captioning pass may append while we read,
    # but `stat().st_mtime` jumps on each append, so the next request re-parses.
    # Reads-of-partial-final-line are safe: we skip lines that fail to JSON-parse.
    CAPTIONS = Path.home() / ".muser" / "captions.jsonl"
    _captions: dict = {"mtime": -1.0, "map": {}}
    _captions_lock = threading.Lock()

    def _load_captions() -> dict[str, str]:
        try:
            mt = CAPTIONS.stat().st_mtime
        except OSError:
            with _captions_lock:
                _captions["map"] = {}
                _captions["mtime"] = -1.0
                return _captions["map"]
        with _captions_lock:
            if mt == _captions["mtime"]:
                return _captions["map"]
        # Re-parse outside the lock to avoid holding it during file IO.
        latest: dict[str, tuple[int, str]] = {}  # path -> (ts, caption)
        try:
            with CAPTIONS.open("r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue  # partial-final-line during a concurrent append
                    p = row.get("path")
                    cap = row.get("caption")
                    if not p or not cap:
                        continue
                    ts = int(row.get("ts") or 0)
                    cur = latest.get(p)
                    if cur is None or ts >= cur[0]:
                        latest[p] = (ts, cap)
        except OSError:
            pass
        m = {p: cap for p, (_, cap) in latest.items()}
        with _captions_lock:
            _captions["map"] = m
            _captions["mtime"] = mt
        return m

    def _save_caption(path: str, caption: str) -> None:
        # Append-only. Latest-wins on load, so a freshly-appended user-edit row
        # immediately overrides any prior model row for this path. The cache is
        # invalidated naturally by the file's mtime bumping on append.
        try:
            mtime = int(os.path.getmtime(path))
        except OSError:
            mtime = 0
        row = {
            "path": path,
            "caption": caption,
            "model": "user-edited",
            "mtime": mtime,
            "ts": int(time.time()),
        }
        CAPTIONS.parent.mkdir(parents=True, exist_ok=True)
        with _captions_lock:
            with CAPTIONS.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            # Eagerly merge into the cache so the next /api/caption GET reflects
            # the edit even before the OS settles the mtime tick.
            _captions["map"][path] = caption

    @app.get("/api/caption")
    def caption(path: str):
        cap = _load_captions().get(path)
        if not cap:
            raise HTTPException(404, "no caption for this path")
        return {"path": path, "uid": uid_for(path), "caption": cap}

    @app.get("/api/captions")
    def captions_bulk():
        # Bulk read of every known caption ({path: caption}). Used by the
        # Interesting / Review / AI tabs' inline filter so a keystroke can
        # match on caption text without one /api/caption round-trip per card.
        return {"captions": _load_captions()}

    @app.post("/api/caption")
    def caption_write(req: CaptionWriteReq):
        if not os.path.isfile(req.path):
            raise HTTPException(404, f"no such image: {req.path}")
        _save_caption(req.path, req.caption)
        return {"ok": True}

    @app.post("/api/caption-bulk")
    def caption_bulk(req: BulkCaptionReq):
        """Caption many paths via gpt-4o-mini. Publishes progress to state.task
        so the frontend's busy overlay shows N/M while it runs.

        Skips paths already in captions.jsonl (latest-wins). Cached-only requests
        return immediately. The newly-written rows land in JSONL on disk and the
        in-memory cache is invalidated by the file's mtime tick.
        """
        # Concurrency guard: refuse if another long-running task is in flight
        # (model loading / indexing). Caption tasks themselves we let through —
        # the second call just re-uses the same task slot's progress feed.
        if state.task is not None and state.task.get("kind") not in (None, "captioning"):
            raise HTTPException(409, f"busy: {state.task.get('kind')}")
        if state.task is not None and state.task.get("kind") == "captioning":
            raise HTTPException(409, "busy: captioning")

        existing = _load_captions()
        paths = [p for p in (req.paths or []) if isinstance(p, str)]
        if req.force:
            todo = [p for p in paths if os.path.isfile(p)]
        else:
            todo = [p for p in paths if os.path.isfile(p) and p not in existing]
        skipped = len(paths) - len(todo)
        if not todo:
            return {"captioned": 0, "skipped": skipped, "errors": [], "total": len(paths)}

        from .caption import caption_paths

        state.task = {"kind": "captioning", "done": 0, "total": len(todo)}
        errors: list[dict] = []
        written_n = 0

        def on_progress(*args):
            # caption_paths calls on_progress(done, total) per image AND on_progress(str) for status.
            if len(args) == 2 and isinstance(args[0], int):
                t = state.task
                if t and t.get("kind") == "captioning":
                    t["done"] = args[0]
                    t["total"] = args[1]

        try:
            written_rows = caption_paths(
                todo, on_progress=on_progress, force=req.force,
            )
            written_n = len(written_rows)
            # Any path that was in `todo` but not in `written_rows` failed.
            written_paths = {r["path"] for r in written_rows}
            for p in todo:
                if p not in written_paths:
                    errors.append({"path": p, "error": "caption failed (see service logs)"})
            # Bump our in-memory cache so subsequent /api/caption GETs see the new rows
            # without waiting for the file-mtime check to round-trip.
            _load_captions()
        except RuntimeError as e:
            # Most common: OPENAI_API_KEY missing. Surface a structured error the UI
            # can map to a "set OPENAI_API_KEY" toast — never echo the key.
            msg = str(e)
            state.task = None
            if "OPENAI_API_KEY" in msg:
                raise HTTPException(400, "OPENAI_API_KEY not set")
            raise HTTPException(500, msg)
        finally:
            # Auto-clear after a short window (mirrors indexing) so the next call isn't blocked.
            def _clear():
                if state.task and state.task.get("kind") == "captioning":
                    state.task = None
            threading.Timer(1.5, _clear).start()

        return {
            "captioned": written_n,
            "skipped": skipped,
            "errors": errors,
            "total": len(paths),
        }

    # ---- per-image scores: Interesting / Review (read ~/.muser/scores.json) ----
    SCORES = Path.home() / ".muser" / "scores.json"
    MUSER_DIR = Path.home() / ".muser"
    # Metrics backed by a per-image model pass (slow, often subset-scored). Coverage
    # = entries in the cache file / total canonical. Everything else is derived
    # in-script from the SigLIP embeddings and is implicitly 100% covered.
    CACHE_BACKED = {
        "aesthetic_v2": "aesthetic_v2_cache.json",
        "pickscore": "pickscore_cache.json",
        "aesthetic_v25": "aesthetic_v25_cache.json",
        "hps_v21": "hps_v21_cache.json",
        # NSFW: three real ViT classifiers, each cached separately so re-runs
        # are incremental. Cache populates over the FULL path set (not just
        # canonical), so once a pass completes its scored>=total clamps to 100%.
        "nsfw_falconsai": "nsfw_falconsai_cache.json",
        "nsfw_adamcodd": "nsfw_adamcodd_cache.json",
        "nsfw_marqo": "nsfw_marqo_cache.json",
    }

    def _metric_coverage(metric: str, total: int) -> dict:
        """{scored, total} for a metric. Derived metrics are 100%; cache-backed
        metrics report cache-file length capped at total."""
        if metric not in CACHE_BACKED:
            return {"scored": total, "total": total}
        cache_path = MUSER_DIR / CACHE_BACKED[metric]
        if not cache_path.exists():
            return {"scored": 0, "total": total}
        try:
            scored = len(json.loads(cache_path.read_text()))
        except Exception:
            scored = 0
        return {"scored": min(scored, total), "total": total}

    @app.get("/api/score")
    def score_rank(metric: str = "interesting", order: str = "desc", offset: int = 0, limit: int = 80,
                   q: str | None = None):
        # `q` semantic-narrows the canonical set before metric sort: typing "car" on the
        # Interesting tab should return cars-ranked-by-aesthetic, not "cards whose basename
        # contains the substring 'car'" (which used to false-match Carlos, Carver, etc).
        if not SCORES.exists():
            return {"built": False, "items": []}
        s = json.loads(SCORES.read_text())
        if metric not in s["metrics"]:
            return {"built": False, "items": []}
        canon = s.get("canonical") or list(s["scores"].keys())  # deduped reps (fallback for old files)
        if q and q.strip():
            # Embed the query, fetch top-500 deduped semantic matches, intersect
            # with canonical so we only sort over paths that actually have scores.
            state.wait_ready()
            emb = state.warm()
            qv = emb.embed_queries([q.strip()])[0]
            hits = state.index.search_dedup(state.model_name, qv, k=500, method="embed")
            sem_paths = {h["path"] for h in hits}
            canon = [p for p in canon if p in sem_paths]
        dupes = s.get("dupes", {})
        ranked = sorted(canon, key=lambda p: s["scores"][p].get(metric, 0), reverse=(order == "desc"))
        # Walk the ranked list pulling live entries until we've filled the
        # requested page. scores.json is written once per scoring pass and
        # never re-validates the path on disk; filtering here keeps deleted
        # files out of the UI without forcing a full re-score. We over-walk
        # past `offset+limit` if needed, but stop once we have `limit` live
        # items, so the cost stays bounded by the number of dead files seen.
        items = []
        i = 0
        skipped = 0
        for p in ranked[offset:]:
            i += 1
            if not os.path.exists(p):
                skipped += 1
                continue
            d = [dp for dp in dupes.get(p, [p]) if os.path.exists(dp)] or [p]
            items.append({
                "path": p, "uid": uid_for(p), "name": os.path.basename(p),
                "score": s["scores"][p].get(metric, 0),
                "scores": s["scores"][p], "dupes": d, "dupe_count": len(d),
            })
            if len(items) >= limit:
                break
        return {"built": True, "metric": metric, "metrics": s["metrics"],
                "total": len(ranked) - skipped if skipped else len(ranked),
                "items": items, "coverage": _metric_coverage(metric, len(canon))}

    # ---- per-image detail page (uid → everything we know about that file) ----
    # The web UI's #/image/<uid> route bundles every per-image fact into a single
    # request: scores, caption, cluster membership, dupes, C2PA, plus
    # dimensions/filesize/mtime from the LanceDB metadata columns.
    #
    # uid→path: 12-char blake2b hashes are non-reversible, so we keep a cached
    # reverse map keyed on the active model + row-count. Rebuilt lazily when the
    # index size changes (incremental indexing) or the model switches. Walking
    # 27k paths is ~20 ms — cheap, but we still cache to keep clicks instant.
    _uid_map: dict = {"key": None, "map": {}}
    _uid_lock = threading.Lock()

    def _uid_to_path(uid: str) -> str | None:
        n = state.index.count(state.model_name)
        key = (state.model_name, n)
        with _uid_lock:
            if _uid_map["key"] != key:
                _uid_map["map"] = {uid_for(p): p for p in state.index.paths(state.model_name)}
                _uid_map["key"] = key
            return _uid_map["map"].get(uid)

    def _cluster_for(path: str) -> dict | None:
        # Reuse `muser cluster`'s output to tell the user which cluster this file lives in.
        # None if `muser cluster` has not been run, or the file landed in -1 (unclustered).
        c = _clusters()
        if not c:
            return None
        m_name = "hdbscan" if "hdbscan" in c["methods"] else next(iter(c["methods"]))
        m = c["methods"][m_name]
        for cl in m["clusters"]:
            if cl["id"] == -1:
                continue
            if path in m["members"].get(str(cl["id"]), []):
                return {"method": m_name, "id": cl["id"], "label": cl["label"],
                        "sublabel": cl.get("sublabel"), "size": cl["size"]}
        return None

    def _scores_for(path: str) -> tuple[dict, list[str]]:
        # Returns (scores, dupes) for `path`. Looks up the path directly first, then
        # walks the dupes table — the user may have clicked into a non-canonical sibling
        # whose scores live under a different canonical rep.
        if not SCORES.exists():
            return {}, [path]
        try:
            s = json.loads(SCORES.read_text())
        except Exception:
            return {}, [path]
        if path in s.get("scores", {}):
            return s["scores"][path], s.get("dupes", {}).get(path, [path])
        for canon, members in s.get("dupes", {}).items():
            if path in members:
                return s.get("scores", {}).get(canon, {}), members
        return {}, [path]

    @app.get("/api/image-detail")
    def image_detail(uid: str):
        # All-in-one read for #/image/<uid>. Anything missing on disk degrades to a
        # null field rather than a 404 — except an unknown uid, which IS a 404 because
        # the rest of the page has nothing to render.
        path = _uid_to_path(uid)
        if path is None:
            raise HTTPException(404, f"unknown uid: {uid}")

        # filesize/mtime from stat (truth on disk); width/height from the LanceDB row.
        width = height = filesize = mtime = None
        try:
            st = os.stat(path)
            filesize = int(st.st_size)
            mtime = float(st.st_mtime)
        except OSError:
            pass
        t = state.index._open(state.model_name)
        if t is not None and "width" in {f.name for f in t.schema}:
            quoted = path.replace("'", "''")
            try:
                rows = (
                    t.search().select(["path", "width", "height", "filesize"])
                    .where(f"path = '{quoted}'").limit(1).to_list()
                )
                if rows:
                    r0 = rows[0]
                    if r0.get("width") is not None: width = int(r0["width"])
                    if r0.get("height") is not None: height = int(r0["height"])
                    if r0.get("filesize") is not None and filesize is None: filesize = int(r0["filesize"])
            except Exception:
                pass

        scores, dupes = _scores_for(path)
        caption = _load_captions().get(path)
        cluster = _cluster_for(path)

        # C2PA provenance (positive-only, may be unavailable).
        from .c2pa import verdict as c2_verdict
        c2pa_info = c2_verdict(path)

        return {
            "path": path,
            "uid": uid,
            "name": os.path.basename(path),
            "exists": os.path.isfile(path),
            "width": width, "height": height,
            "filesize": filesize, "mtime": mtime,
            "scores": scores,
            "caption": caption,
            "cluster": cluster,
            "dupes": dupes,
            "dupe_count": len(dupes),
            "c2pa": c2pa_info,
        }

    return app


def serve(host: str = "127.0.0.1", port: int = 7777, model: str = DEFAULT_MODEL):
    import uvicorn

    print(f"Muser serving on http://{host}:{port}  (model: {model})", flush=True)
    uvicorn.run(create_app(model), host=host, port=port, log_level="warning")
