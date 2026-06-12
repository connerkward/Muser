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
from .paths import MUSER_HOME

# Module-level so `from __future__ import annotations` can resolve the UploadFile
# annotation on /api/search-upload (a function-scope import leaves it an
# unresolved ForwardRef → pydantic "not fully defined").
from fastapi import File, Form, UploadFile
from pydantic import BaseModel

from .index import MetaFilter, MuserIndex, uid_for
from .registry import DEFAULT_MODEL, load_model, model_names

WEB = Path(__file__).resolve().parent / "web" / "app.html"

# ---- Result-hiding policy (applied to EVERY result-producing endpoint) -------
# Three orthogonal filters funnel through one `_hidden(path)` predicate so a
# path that is dead / NSFW / in a hidden cluster never reaches the UI from any
# tab (search, color, upload, score/Interesting, filter, ai, cluster members).
#
# NSFW: `nsfw_consensus` (fallback `nsfw`) from ~/.muser/scores.json above this
# threshold is hidden by default in every tab. NOTE: in the current scores.json
# these metrics are RANK-NORMALIZED (uniform 0..1 percentiles — median is exactly
# 0.5), so 0.5 hides the top ~50% of the library. The spec pinned 0.5 as the
# default; tune this constant up (e.g. 0.9 = hide top decile) if that's too
# aggressive. Exposed as a module constant precisely so it's a one-line change.
# nsfw_consensus is a rank-normalized percentile (median 0.5), NOT a raw probability,
# so 0.5 would hide the top HALF of the library. 0.9 = top ~10% most-NSFW-scored.
NSFW_HIDE_THRESHOLD = 0.9

# HDBSCAN clusters to hide everywhere, matched case-insensitively as substrings
# against each cluster's representative caption / label in ~/.muser/clusters.json.
# Seeded with the "conceptual demo" people-hiding set; append to extend, empty
# the list to disable cluster-hiding entirely.
HIDDEN_CLUSTER_MATCHERS = [
    "a man that is sitting down with a remote in his hand",
    "smiling woman holding a cell phone up to her ear",
    "arafed photo of a woman taking a selfie with her phone",
]


# "Demo mode" hides NSFW + the people/selfie clusters everywhere. **Off by
# default** (NSFW/clusters shown), and the user's choice **persists across
# restarts** via ~/.muser/demo_mode.json — so once you turn it on it stays on
# until you turn it off. Toggled from the web debug menu via /api/demo-mode.
# Dead-file hiding is unconditional (not part of demo mode).
_DEMO_FILE = MUSER_HOME / "demo_mode.json"


def _load_demo_hide() -> bool:
    try:
        return bool(json.loads(_DEMO_FILE.read_text()).get("hide", False))
    except Exception:
        return False


_DEMO = {"hide": _load_demo_hide()}


class DemoModeReq(BaseModel):
    on: bool


class ContactSheetReq(BaseModel):
    paths: list[str]
    cols: int = 6
    cell: int = 256
    max: int = 36


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


class ThreeDReq(BaseModel):
    # Module-level (not nested in create_app) so `from __future__ import
    # annotations` can resolve it as a request body — a function-scope model
    # degrades to a query param (FastAPI can't resolve the ForwardRef) → 422.
    output_index: int
    # When true, synthesize left/back/right views and use the multi-view 3D path.
    multi_angle: bool = False


class CheckoutReq(BaseModel):
    # Non-blocking cart checkout: caption (network) + upscale (local) run
    # concurrently in the background, then a zip is assembled. `force_caption`
    # re-captions even files that already have a cached caption; `caption_prompt`
    # overrides the baked LoRA system prompt (threaded into caption.py, which
    # persists it on each row).
    items: list[CartItem] = []
    caption_prompt: str | None = None
    force_caption: bool = False
    # ---- generate mode -------------------------------------------------------
    # mode="zip" (default) is the existing LoRA-zip checkout, unchanged.
    # mode="generate" runs the cart through fal.ai (muser/pipeline.py) to produce
    # new images / 3D. These fields are ignored in zip mode.
    mode: str = "zip"  # "zip" | "generate"
    generator: str = "nano_banana"  # "nano_banana" | "chatgpt" | "both" | "lora"
    lora_type: str = "concept"
    caption: bool = False
    cutout: bool = False
    upscale_refs: bool = False
    expand_prompts: bool = True
    brief: str = ""
    gen_prompts: list[str] | None = None
    n_outputs: int = 4
    make_3d: bool = False
    # Synthesize left/back/right views and reconstruct 3D from all of them.
    multi_angle: bool = False


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
        # Color-palette facet scan progress (Color tab), polled by /api/color.
        self.color = {"scanning": False, "done": 0, "total": 0}
        # AI-likelihood (GRIP) facet scan progress, polled by /api/ai-likelihood.
        self.aidet = {"scanning": False, "done": 0, "total": 0}
        # Face detect/embed/cluster facet progress (People), polled by /api/faces.
        self.faces = {"scanning": False, "done": 0, "total": 0}

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
    from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
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
        # Same priming for the color facet cache so /api/search-color ranks
        # in-RAM with no disk read per query.
        from . import color as _color
        cprimed = _color.prime_cache_from_sidecar()
        if cprimed:
            print(f"  primed color cache: {cprimed} entries", flush=True)
        # And the AI-likelihood (GRIP) facet cache so /api/search inlines ai_pct
        # per result from RAM and the ai_min filter ranks without a disk read.
        from . import aidet as _aidet
        aprimed = _aidet.prime_cache_from_sidecar()
        if aprimed:
            print(f"  primed aidet cache: {aprimed} entries", flush=True)
        # Face metadata cache (per-image faces + cluster ids) for People enrichment
        # and the personal-tool's recurring-person signal.
        from . import faces as _faces
        fprimed = _faces.prime_cache_from_sidecar()
        if fprimed:
            print(f"  primed faces cache: {fprimed} entries", flush=True)
        # Prime the result-hiding sets (NSFW from scores.json, hidden-cluster
        # paths from clusters.json) so the first request's _hidden() lookups are
        # O(1) RAM hits, not a cold disk parse. Both are mtime-cached and refresh
        # lazily on the next request after either file changes.
        try:
            nset = _refresh_scores_cache()["nsfw"]
            hset = _hidden_cluster_paths()
            print(f"  primed hide sets: nsfw={len(nset)} hidden_cluster={len(hset)} "
                  f"(nsfw_threshold={NSFW_HIDE_THRESHOLD})", flush=True)
        except Exception as e:
            print(f"  hide-set prime skipped: {e}", flush=True)
        # Non-blocking: fire the warm in a thread so uvicorn starts accepting
        # connections in milliseconds. The page can load, see task=loading_model
        # in /api/status, and render its overlay while weights load.
        print(f"  loading model {state.model_name} (background) ...", flush=True)
        state.start_warm()

    @app.get("/", response_class=HTMLResponse)
    def home():
        return WEB.read_text()

    @app.get("/proto/weights", response_class=HTMLResponse)
    def proto_weights():
        # Interactive prototype studio for per-concept weighting UX (same-origin so
        # it can hit the live /api/search). Read per request so iterating the file
        # needs no restart. Harmless if the file is absent.
        f = Path(__file__).resolve().parent / "web" / "weights-proto.html"
        return f.read_text() if f.exists() else "<h1>prototype not found</h1>"

    @app.get("/proto/weights-spread", response_class=HTMLResponse)
    def proto_weights_spread():
        # Lookdev component spread: many divergent per-concept weighting controls,
        # all bound to one shared concept set + live results.
        f = Path(__file__).resolve().parent / "web" / "weights-spread.html"
        return f.read_text() if f.exists() else "<h1>spread not found</h1>"

    @app.get("/proto/weights-inline", response_class=HTMLResponse)
    def proto_weights_inline():
        # Inline weighting: the search bar IS the control — concepts are tokens with
        # a weight-bar under each word, polarity toggled per token.
        f = Path(__file__).resolve().parent / "web" / "weights-inline.html"
        return f.read_text() if f.exists() else "<h1>inline proto not found</h1>"

    @app.get("/proto/weights-cloud", response_class=HTMLResponse)
    def proto_weights_cloud():
        # Inline weighted word-cloud search bar: word size = weight, drag up/down.
        f = Path(__file__).resolve().parent / "web" / "weights-cloud.html"
        return f.read_text() if f.exists() else "<h1>cloud proto not found</h1>"

    @app.get("/proto/weights-cloud-split", response_class=HTMLResponse)
    def proto_weights_cloud_split():
        # Word-cloud bar with separate positive (find) and negative (exclude) zones.
        f = Path(__file__).resolve().parent / "web" / "weights-cloud-split.html"
        return f.read_text() if f.exists() else "<h1>split cloud proto not found</h1>"

    @app.get("/proto/weights-recommended", response_class=HTMLResponse)
    def proto_weights_recommended():
        # The recommended design: split find/exclude token chips with a length-based
        # weight slider (not font-size), with the design rationale on the page.
        f = Path(__file__).resolve().parent / "web" / "weights-recommended.html"
        return f.read_text() if f.exists() else "<h1>recommended proto not found</h1>"

    @app.get("/proto/weights-1line", response_class=HTMLResponse)
    def proto_weights_1line():
        # Split find/exclude word-cloud on ONE line (like muser's search bar).
        f = Path(__file__).resolve().parent / "web" / "weights-1line.html"
        return f.read_text() if f.exists() else "<h1>1-line proto not found</h1>"

    @app.get("/proto/weights-budget", response_class=HTMLResponse)
    def proto_weights_budget():
        # Budget bars with selectable per-family hue ranges (pull vs push).
        f = Path(__file__).resolve().parent / "web" / "weights-budget.html"
        return f.read_text() if f.exists() else "<h1>budget proto not found</h1>"

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
        from . import c2pa as _c2pa
        from . import aidet as _aidet_mod
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
            "c2pa": _c2pa.available(),
            "aidet": _aidet_mod.available(),
            # True only on the hidden personal instance (MUSER_HOME=~/.muser-personal).
            # The frontend uses this to reveal the Triage tab; never set on the main one.
            "personal": _is_personal_instance(),
        }

    # ---- Hidden personal-triage endpoints (only meaningful on the personal root) ----
    def _is_personal_instance() -> bool:
        from .paths import is_personal
        return is_personal()

    @app.get("/api/personal/summary")
    def personal_summary():
        """Bucket counts + uncertainty histogram for the Triage tab. classified=False
        until `muser personal classify` has run."""
        from .personal import personalness as _pn
        entries = _pn.all_entries()
        if not entries:
            return {"classified": False, "total": 0,
                    "buckets": {"personal": 0, "in_between": 0, "reference": 0},
                    "vlm": 0, "hist": [0] * 20}
        buckets = {"personal": 0, "in_between": 0, "reference": 0}
        hist = [0] * 20
        vlm = 0
        for e in entries.values():
            b = e.get("bucket")
            if b in buckets:
                buckets[b] += 1
            if "vlm" in e:
                vlm += 1
            u = e.get("unc")
            if u is not None:
                hist[min(19, max(0, int(u * 20)))] += 1
        return {"classified": True, "total": len(entries), "buckets": buckets,
                "vlm": vlm, "hist": hist}

    @app.get("/api/personal/bucket")
    def personal_bucket(name: str, offset: int = 0, limit: int = 120, sort: str = "p"):
        """Paged image paths in a bucket. sort: 'p' (most→least personal), 'unc'
        (most uncertain first). Live-file filtered."""
        from .personal import personalness as _pn
        rows = [(p, e) for p, e in _pn.all_entries().items() if e.get("bucket") == name]
        if sort == "unc":
            rows.sort(key=lambda pe: -(pe[1].get("unc") or 0))
        else:
            personal = name != "reference"
            rows.sort(key=lambda pe: -(pe[1].get("p") or 0) if personal else (pe[1].get("p") or 0))
        total = len(rows)
        page = rows[offset:offset + limit]
        items = [{"path": p, "p": e.get("p"), "unc": e.get("unc"),
                  "sig": e.get("sig"), "vlm": e.get("vlm")} for p, e in page]
        return {"name": name, "total": total, "offset": offset, "items": items}

    @app.get("/api/personal/vlm-estimate")
    def personal_vlm_estimate(umin: float = 0.0, umax: float = 1.0):
        from .personal import vlm_triage as _vt
        n, cost = _vt.estimate(umin, umax)
        return {"count": n, "cost": cost}

    @app.post("/api/personal/vlm-run")
    def personal_vlm_run(req: dict):
        """Run the VLM pass over an uncertainty band in a background thread; poll
        /api/personal/vlm-status. Idempotent: no-op while a run is in flight."""
        from .personal import vlm_triage as _vt
        if getattr(state, "_vlm", None) and state._vlm.get("running"):
            return {"started": False, "reason": "already running"}
        umin = float(req.get("umin", 0.0))
        umax = float(req.get("umax", 1.0))
        n, cost = _vt.estimate(umin, umax)
        state._vlm = {"running": True, "done": 0, "total": n, "umin": umin, "umax": umax,
                      "cost": cost, "result": None}

        def _bg():
            try:
                res = _vt.run(umin, umax,
                              on_status=lambda d, t: state._vlm.update(done=d, total=t))
                state._vlm["result"] = res
            finally:
                state._vlm["running"] = False
        threading.Thread(target=_bg, daemon=True, name="muser-vlm-triage").start()
        return {"started": True, "count": n, "cost": cost}

    @app.get("/api/personal/vlm-status")
    def personal_vlm_status():
        return getattr(state, "_vlm", None) or {"running": False}

    @app.get("/api/personal/report")
    def personal_report():
        """Aggregate for the interactive results dashboard (/report): bucket counts,
        P + uncertainty histograms, what signal drove each bucket, and the VLM-judge
        benchmark (confusion matrix + disagreement list) if `muser personal benchmark`
        has run. Reads live from personalness.json + benchmark.json."""
        import json as _json
        from .personal import personalness as _pn
        from .paths import data_file as _df
        entries = _pn.all_entries()
        buckets = {"personal": 0, "in_between": 0, "reference": 0}
        p_hist = [0] * 20
        unc_hist = [0] * 20
        # what carried each bucket (priority order matches the fusion floors)
        drivers = {b: {"people": 0, "camera": 0, "face": 0, "doc": 0, "appearance": 0}
                   for b in buckets}
        for e in entries.values():
            b = e.get("bucket")
            if b not in buckets:
                continue
            buckets[b] += 1
            pv, uv = e.get("p"), e.get("unc")
            if pv is not None:
                p_hist[min(19, max(0, int(pv * 20)))] += 1
            if uv is not None:
                unc_hist[min(19, max(0, int(uv * 20)))] += 1
            s = e.get("sig", {})
            key = ("people" if s.get("people") else "camera" if s.get("cam")
                   else "face" if (s.get("F", 0) or 0) > 0 else "doc" if s.get("doc")
                   else "appearance")
            drivers[b][key] += 1
        bench = None
        bf = _df("benchmark.json")
        if bf.exists():
            try:
                bench = _json.loads(bf.read_text())
            except Exception:
                bench = None
        return {"classified": bool(entries), "total": len(entries), "buckets": buckets,
                "vlm": sum(1 for e in entries.values() if "vlm" in e),
                "p_hist": p_hist, "unc_hist": unc_hist, "drivers": drivers,
                "weights": {"evidence": 0.45, "camera": 0.40, "appearance": 0.15},
                "benchmark": bench}

    @app.get("/report", response_class=HTMLResponse)
    def personal_report_page():
        f = Path(__file__).resolve().parent / "web" / "personal_report.html"
        return f.read_text() if f.exists() else "<h1>report page not found</h1>"

    @app.get("/api/personal/eval-sample")
    def personal_eval_sample(n: int = 60, seed: int = 0):
        from .personal import evaluate as _ev
        return {"items": _ev.sample(n=n, seed=seed)}

    @app.post("/api/personal/eval-label")
    def personal_eval_label(req: dict):
        from .personal import evaluate as _ev
        _ev.label(req["path"], req.get("label"), req.get("model"))
        return {"ok": True}

    @app.get("/api/personal/eval-results")
    def personal_eval_results():
        from .personal import evaluate as _ev
        return _ev.results()

    @app.post("/api/personal/eval-flag")
    def personal_eval_flag(req: dict):
        from .personal import evaluate as _ev
        _ev.set_flag(req["path"], req.get("flag"))
        return {"ok": True}

    @app.get("/api/personal/eval-flagged")
    def personal_eval_flagged():
        from .personal import evaluate as _ev
        return _ev.flagged()

    @app.get("/evaluate", response_class=HTMLResponse)
    def personal_evaluate_page():
        f = Path(__file__).resolve().parent / "web" / "personal_evaluate.html"
        return f.read_text() if f.exists() else "<h1>evaluate page not found</h1>"

    @app.get("/api/jobs")
    def jobs_list():
        """Everything in flight, for the Jobs page:
          - task:      the single foreground activity (index / caption / backfill)
          - jobs:      background checkout/generate jobs from the in-process registry
          - pipelines: recent generate-pipeline runs (newest-first, persisted)
        """
        from . import jobs as _jobs
        from . import pipeline as _pl
        # Snapshot job refs under the lock, then build public dicts outside it
        # (public() re-acquires the same non-reentrant lock).
        with _jobs.REGISTRY._lock:
            raw = list(_jobs.REGISTRY._jobs.values())
        out = []
        for j in raw:
            pub = _jobs.public(j)
            pub["kind"] = j.get("kind")
            pub["created"] = j.get("created")
            out.append(pub)
        out.sort(key=lambda d: d.get("created") or 0, reverse=True)
        try:
            pipes = _pl.list_runs()[:20]
        except Exception:
            pipes = []
        return {"task": state.task, "jobs": out, "pipelines": pipes}

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
                _scan_color(folder)
                _scan_aidet(folder)
                _scan_faces(folder)

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

    def _scan_color(folder: str | None = None):
        # Shared by the post-index hook and POST /api/color/scan.
        from . import color as _color
        if state.color.get("scanning"):
            return
        paths = state.index.paths(state.model_name, under=folder)
        if not paths:
            return
        state.color = {"scanning": True, "done": 0, "total": len(paths)}
        try:
            _color.scan(paths, progress=lambda d, t: state.color.update(done=d, total=t))
        finally:
            state.color["scanning"] = False

    def _scan_aidet(folder: str | None = None):
        # Shared by the post-index hook and POST /api/ai-likelihood/scan. Scores the
        # GRIP AI-likelihood facet over the (sub)tree. No-op if the weight/torch is
        # absent or a scan is already running; incremental, so only new/changed files
        # run the forward. Slow (full-res ResNet forward per image) — runs in a daemon
        # thread off the busy overlay, progress in state.aidet.
        from . import aidet as _aidet
        if not _aidet.available() or state.aidet.get("scanning"):
            return
        paths = state.index.paths(state.model_name, under=folder)
        if not paths:
            return
        state.aidet = {"scanning": True, "done": 0, "total": len(paths), "started": time.time()}
        try:
            _aidet.scan(paths, progress=lambda d, t: state.aidet.update(done=d, total=t))
        finally:
            state.aidet["scanning"] = False

    def _scan_faces(folder: str | None = None):
        # Shared by the post-index hook and POST /api/faces/scan. Detects + embeds
        # faces over the (sub)tree, then re-clusters ALL faces into people (HDBSCAN is
        # global, so new faces can join/extend existing clusters). No-op if insightface
        # is absent or a scan is running; incremental, so only new/changed files run the
        # forward. Slow — daemon thread off the busy overlay, progress in state.faces.
        from . import faces as _faces
        if not _faces.available() or state.faces.get("scanning"):
            return
        paths = state.index.paths(state.model_name, under=folder)
        if not paths:
            return
        state.faces = {"scanning": True, "done": 0, "total": len(paths), "started": time.time()}
        try:
            _faces.scan(paths, progress=lambda d, t: state.faces.update(done=d, total=t))
            _faces.cluster()  # re-cluster the full face set; cheap vs the detection pass
        finally:
            state.faces["scanning"] = False

    # ---- path → cluster-label map, used for the post-search refinement chips ----
    # Reads ~/.muser/clusters.json once and caches a path → cluster-label dict.
    # No embedding pass — the only consumer is _refinements, which does pure
    # bucket-counting on result paths.
    def _path_to_label():
        if state.label_index is not None:
            return state.label_index
        clusters_file = MUSER_HOME / "clusters.json"
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

    def _search_dedup_precomputed(qv, k, method, folder, meta):
        """Fast dedup path: use the precomputed dupes map from ~/.muser/scores.json
        instead of LanceDB's full-vector over-fetch + pairwise cosine walk.

        Trades ~55 ms (288-row full-vector fetch + Python pairwise loop) for
        ~10 ms (path-only LanceDB query + O(N) hashmap lookups). Same shape
        of result list as `index.search_dedup`.

        Returns None when the precomputed map isn't usable — caller falls
        back to the cosine walk. We bail when:
        - scores.json doesn't exist yet
        - the dupes map is empty
        - method is "phash" or "both" (those want fresh image hashing)
        """
        if method != "embed":
            return None
        rep_of = _refresh_scores_cache()["rep_of"]
        if not rep_of:
            return None
        # Over-fetch enough raw rows to fill k unique reps even when early
        # results all collapse into one group. Path-only LanceDB query, so a
        # large n is cheap (~10 ms at limit=288 vs ~55 ms with full vectors).
        # Matches the original cosine-walk's n = max(k*12, 200) for parity.
        n = max(k * 12, 200)
        raw = state.index.search(state.model_name, qv, k=n, folder=folder, meta=meta)
        seen: dict[str, dict] = {}
        order: list[str] = []
        for path, score in raw:
            rep = rep_of.get(path, path)
            entry = seen.get(rep)
            if entry is None:
                entry = {"path": path, "score": float(score), "dupes": [path]}
                seen[rep] = entry
                order.append(rep)
            elif path not in entry["dupes"]:
                entry["dupes"].append(path)
        out = []
        for rep in order[:k]:
            e = seen[rep]
            out.append({
                "path": e["path"],
                "name": os.path.basename(e["path"]),
                "score": round(e["score"], 4),
                "dupes": e["dupes"],
                "dupe_count": len(e["dupes"]),
            })
        return out

    def _run_search(qv, k, dedup, method, folder, meta=None, ai_min=0, ai_max=100, sort=None):
        # ai_min/ai_max ∈ [0,100]: keep results with GRIP AI-likelihood % in the band
        # (ai_max<100 = remove AI; ai_min>0 = only AI). The cut is post-kNN (the facet
        # isn't a LanceDB column), so over-fetch a wider candidate pool first and trim
        # to k after filtering. sort: "ai"/"ai_asc" reorders by AI-likelihood.
        _ai_min = int(ai_min or 0)
        _ai_max = 100 if ai_max is None else int(ai_max)
        widen = _ai_min > 0 or _ai_max < 100 or sort in ("ai", "ai_asc")
        fetch_k = min(k * 8, 1500) if widen else k
        if dedup:
            results = _search_dedup_precomputed(qv, fetch_k, method, folder, meta)
            if results is None:
                results = state.index.search_dedup(
                    state.model_name, qv, k=fetch_k, method=method, folder=folder, meta=meta,
                )
        else:
            results = [
                {"path": p, "name": os.path.basename(p), "score": round(s, 4), "dupes": [p], "dupe_count": 1}
                for p, s in state.index.search(state.model_name, qv, k=fetch_k, folder=folder, meta=meta)
            ]
        return _postprocess(results, k, ai_min=_ai_min, ai_max=ai_max, sort=sort)

    def _postprocess(results, k, ai_min=0, ai_max=100, sort=None):
        """Shared tail for every result-producing path (kNN blend AND multi-concept
        match-all): live-file filtering + dead-group promotion, the centralized
        hide policy, per-result enrichment (uid / prob / dims / scores / c2pa /
        AI-likelihood), then the optional ai_min filter + AI sort, trimmed to k.
        Input result dicts must carry at least path/name/score/dupes/dupe_count."""
        # Drop results whose canonical path no longer exists on disk, and prune
        # dead entries from each result's `dupes[]`. Only when the leading path is
        # dead do we walk the precomputed dedup group (scores.json) to promote a
        # live member — the common case (everything alive) stays a cheap stat.
        kept = []
        dead_leads = []
        for r in results:
            d = r.get("dupes") or [r["path"]]
            lead_alive = os.path.exists(r["path"])
            if lead_alive:
                live = [p for p in d if p == r["path"] or os.path.exists(p)]
            else:
                groups = _refresh_scores_cache()["groups"]
                rep_of_map = _refresh_scores_cache()["rep_of"]
                canon = rep_of_map.get(r["path"]) or r["path"]
                candidates = list(d)
                for p in groups.get(canon, []):
                    if p not in candidates:
                        candidates.append(p)
                live = [p for p in candidates if os.path.exists(p)]
                if live:
                    r["path"] = live[0]
                    r["name"] = os.path.basename(live[0])
            if not live:
                dead_leads.append(r["path"])
                continue
            r["dupes"] = live
            r["dupe_count"] = len(live)
            kept.append(r)
        results = kept
        if dead_leads:
            _purge_dead(dead_leads)
        results = _filter_results(results)
        for r in results:
            r["uid"] = uid_for(r["path"])
        _add_prob(results)
        _attach_dims(results, [r["path"] for r in results])
        _attach_scores(results)
        _attach_c2pa(results)
        _attach_aidet(results)
        if int(ai_min or 0) > 0 or int(ai_max if ai_max is not None else 100) < 100:
            results = _filter_by_ai_band(results, ai_min, ai_max)
        if sort in ("ai", "ai_asc"):
            results.sort(key=lambda r: (_ai_pct(r) if _ai_pct(r) is not None else -1),
                         reverse=(sort == "ai"))
        return results[:k]

    def _parse_concepts(q: str):
        """Split a query into signed concepts. Commas separate concepts; WITHIN a
        comma-part a whitespace token that starts with '-'/'+' opens a new signed
        sub-concept (unsigned tokens extend the current one), so a pasted prompt like
        "chair, metal -retro" parses as chair(+), metal(+), retro(−) — not the literal
        string "metal -retro". A '-' inside a word ("self-portrait") is NOT a boundary
        (only a leading sign on a space-delimited token is). Optional trailing
        ":<number>" sets the concept weight ("car:2", "snow:0.5"). Returns
        [(text, signed_weight), ...] lowercased. A single unsigned phrase → one
        concept (the plain single-vector path, unchanged)."""
        out = []

        def emit(words, sign):
            text = " ".join(words).strip()
            if not text:
                return
            weight = 1.0
            if ":" in text:
                head, _, tail = text.rpartition(":")
                try:
                    weight = abs(float(tail.strip()))
                    text = head.strip()
                except ValueError:
                    pass
            if text:
                out.append((text.lower(), sign * weight))

        for part in q.split(","):
            sign, words = 1, []
            for tok in part.split():
                if len(tok) > 1 and tok[0] in "+-":   # leading-sign token → concept boundary
                    emit(words, sign)
                    sign = -1 if tok[0] == "-" else 1
                    words = [tok[1:]] if tok[1:] else []
                else:
                    words.append(tok)
            emit(words, sign)
        return out

    def _blend_vector(emb, pos, neg, neg_strength):
        """Weighted sum of positive concept embeddings minus weighted negatives,
        renormalized → one query vector. pos/neg are (text, weight) lists. This is
        prompt ensembling with per-concept weights: each concept pulls the blend in
        proportion to its weight, instead of one long string letting the bag-of-words
        encoder fixate on whichever token dominates."""
        import numpy as np
        texts = [t for t, _ in pos] + [t for t, _ in neg]
        vecs = emb.embed_queries(texts) if texts else []
        qv = None
        for i, (_t, w) in enumerate(pos):
            c = w * vecs[i]
            qv = c.copy() if qv is None else qv + c
        for j, (_t, w) in enumerate(neg):
            c = neg_strength * w * vecs[len(pos) + j]
            qv = -c if qv is None else qv - c
        n = float(np.linalg.norm(qv)) if qv is not None else 0.0
        return qv / n if n > 0 else qv

    def _search_match_all(pos, neg, k, folder, meta, neg_strength, ai_min, ai_max, sort):
        """Match-all (intersection) over multiple concepts: a result's score is the
        MIN of its per-concept similarities — high only when EVERY concept is
        strongly present (true AND), unlike a blended centroid which rewards
        'a bit of each'. One kNN per concept (cheap), merged by min; a concept a
        result misses entirely is floored at that concept's weakest pooled hit so
        it ranks below results present in all. Negatives subtract their similarity.
        Weights (the `:N` suffix) don't apply here — min-of-similarities has no clean
        weighting; they shape the blend mode instead. pos/neg are (text, weight)."""
        emb = state.warm()
        pos_vecs = emb.embed_queries([t for t, _ in pos])
        neg_vecs = emb.embed_queries([t for t, _ in neg]) if neg else []
        pool = min(max(k * 12, 240), 2400)
        per = [dict(state.index.search(state.model_name, v, k=pool, folder=folder, meta=meta))
               for v in pos_vecs]
        neg_per = [dict(state.index.search(state.model_name, v, k=pool, folder=folder, meta=meta))
                   for v in neg_vecs]
        floors = [min(d.values()) if d else 0.0 for d in per]
        cand = set().union(*[set(d) for d in per]) if per else set()
        # Collapse near-dupes to canonical (scores.json), keeping the best score.
        rep_of_map = _refresh_scores_cache()["rep_of"]
        best: dict[str, float] = {}
        for p in cand:
            s = min(per[i].get(p, floors[i]) for i in range(len(per)))
            for nh in neg_per:
                s -= neg_strength * nh.get(p, 0.0)
            canon = rep_of_map.get(p, p)
            if canon not in best or s > best[canon]:
                best[canon] = s
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[: max(k * 4, 200)]
        results = [
            {"path": p, "name": os.path.basename(p), "score": round(float(s), 4),
             "dupes": [p], "dupe_count": 1}
            for p, s in ranked
        ]
        return _postprocess(results, k, ai_min=ai_min, ai_max=ai_max, sort=sort)

    # Cache the parsed scores.json by mtime so the Search-tab sort-blend join
    # is one disk read per scores.json update, not one per request. Also
    # carries an inverse `rep_of` map (path → canonical representative) built
    # from scores.json's `dupes` so search can collapse near-duplicates via
    # an O(1) hashmap lookup instead of a 288-row LanceDB+cosine walk.
    # `nsfw` carries the NSFW hidden-path set (built in the same parse pass —
    # one disk read, not a second sidecar). A path is in `nsfw` when its
    # nsfw_consensus (fallback nsfw) exceeds NSFW_HIDE_THRESHOLD.
    # `full` carries the entire parsed scores.json dict (metrics/canonical/scores/
    # dupes) so endpoints that need more than `map`/`rep_of`/`nsfw` (e.g. /api/score,
    # image-detail, _scores_for) can read it from the same mtime-cached parse instead
    # of re-`json.loads`-ing the 15 MB file per request.
    _scores_cache: dict = {"mtime": -1.0, "map": {}, "rep_of": {}, "groups": {}, "nsfw": set(), "full": {}}
    _scores_lock = threading.Lock()

    def _refresh_scores_cache():
        # Reload _scores_cache from disk if scores.json has changed. Returns
        # the cache dict (caller already holds _scores_lock or doesn't care).
        scores_path = MUSER_HOME / "scores.json"
        try:
            mt = scores_path.stat().st_mtime
        except OSError:
            return _scores_cache
        with _scores_lock:
            if mt == _scores_cache["mtime"]:
                return _scores_cache
            try:
                s = json.loads(scores_path.read_text())
                groups = s.get("dupes", {}) or {}
                rep_of: dict[str, str] = {}
                for canon, members in groups.items():
                    rep_of[canon] = canon
                    for m in members:
                        rep_of[m] = canon
                smap = s.get("scores", {})
                nsfw: set[str] = {
                    p for p, row in smap.items()
                    if float(row.get("nsfw_consensus", row.get("nsfw", 0.0)) or 0.0) > NSFW_HIDE_THRESHOLD
                }
                _scores_cache["map"] = smap
                _scores_cache["rep_of"] = rep_of
                _scores_cache["groups"] = groups
                _scores_cache["nsfw"] = nsfw
                _scores_cache["full"] = s
                _scores_cache["mtime"] = mt
            except Exception:
                _scores_cache["map"] = {}
                _scores_cache["rep_of"] = {}
                _scores_cache["groups"] = {}
                _scores_cache["nsfw"] = set()
                _scores_cache["full"] = {}
                _scores_cache["mtime"] = mt
        return _scores_cache

    # ---- Hidden-cluster set (people-hiding "conceptual demo") ----------------
    # Resolve HIDDEN_CLUSTER_MATCHERS to a union of member paths of every
    # HDBSCAN cluster whose representative caption/label matches (case-insensitive
    # substring). Cached by clusters.json mtime so it's a single parse per
    # clusters.json update, then O(1) set membership per result.
    _hidden_clusters: dict = {"mtime": -1.0, "paths": set()}
    _hidden_clusters_lock = threading.Lock()

    def _hidden_cluster_paths() -> set[str]:
        clusters_path = MUSER_HOME / "clusters.json"
        if not HIDDEN_CLUSTER_MATCHERS:
            return set()
        try:
            mt = clusters_path.stat().st_mtime
        except OSError:
            return _hidden_clusters["paths"]
        with _hidden_clusters_lock:
            if mt == _hidden_clusters["mtime"]:
                return _hidden_clusters["paths"]
            paths: set[str] = set()
            try:
                c = json.loads(clusters_path.read_text())
                methods = c.get("methods", {})
                m_name = "hdbscan" if "hdbscan" in methods else (next(iter(methods)) if methods else None)
                if m_name:
                    m = methods[m_name]
                    matchers = [s.lower() for s in HIDDEN_CLUSTER_MATCHERS]
                    for cl in m.get("clusters", []):
                        if cl.get("id") == -1:
                            continue
                        # Match against representative caption(s) AND the label.
                        hay = " ".join(
                            str(x).lower() for x in (
                                cl.get("label") or "", cl.get("sublabel") or "",
                                *(cl.get("reps") or []),
                            )
                        )
                        if any(mt_s in hay for mt_s in matchers):
                            for p in m.get("members", {}).get(str(cl["id"]), []):
                                paths.add(p)
            except Exception:
                paths = set()
            _hidden_clusters["paths"] = paths
            _hidden_clusters["mtime"] = mt
            return paths

    # ---- Unified hide predicate + dead-file purge ----------------------------
    def _purge_dead(paths: list[str]) -> set[str]:
        """Stat each path; collect the ones gone from disk, fire a best-effort
        background DELETE of those rows from the LanceDB table (so they never
        resurface), and return the dead set. Never raises — a failed delete must
        not 500 a search. Empty/all-alive → no thread spawned."""
        dead = {p for p in paths if not os.path.exists(p)}
        if dead:
            def _bg(victims: list[str]):
                try:
                    state.index.delete_paths(state.model_name, victims)
                except Exception:
                    pass
            threading.Thread(
                target=_bg, args=(list(dead),), daemon=True, name="muser-purge-dead",
            ).start()
        return dead

    def _hidden(path: str) -> bool:
        """True iff `path` should be hidden from ALL result endpoints: file gone
        from disk, NSFW above threshold, or a member of a hidden cluster. Pure
        set/dict lookups (plus one stat for the dead check) — O(1) per result."""
        if not os.path.exists(path):
            return True
        if not _DEMO["hide"]:
            return False  # demo mode off → only dead files are hidden
        if path in _refresh_scores_cache()["nsfw"]:
            return True
        if path in _hidden_cluster_paths():
            return True
        return False

    def _filter_results(results: list[dict]) -> list[dict]:
        """Drop every result whose `path` is hidden (dead / NSFW / hidden-cluster),
        and purge any dead paths from the index in the background. Routed through
        by every result-producing endpoint so the policy is centralized. Result
        shapes are untouched; only the list shrinks."""
        if not results:
            return results
        _purge_dead([r["path"] for r in results])
        out = []
        for r in results:
            if _hidden(r["path"]):
                continue
            # Prune hidden entries from each result's dupes[] list too, so a
            # "show all copies" flow can't surface a hidden sibling.
            d = [p for p in (r.get("dupes") or [r["path"]]) if not _hidden(p)]
            if d:
                r["dupes"] = d
                r["dupe_count"] = len(d)
            out.append(r)
        return out

    # Aesthetic metrics surfaced per search hit so the Search-tab "Sort blend"
    # can re-rank client-side without re-fetching. Mirrors the non-combined
    # options in the Interesting tab's dropdown (combined "interesting"/"novelty"
    # are derived rankings, not raw aesthetic scores, so they're excluded).
    _SORT_BLEND_METRICS = ("aesthetic_v2", "pickscore", "aesthetic_v25", "hps_v21", "aesthetic")

    def _attach_scores(results):
        # Surface per-image aesthetic scores from ~/.muser/scores.json on each
        # search hit. Missing entries → key simply absent.
        if not results:
            return
        m = _refresh_scores_cache()["map"]
        for r in results:
            row = m.get(r["path"])
            if not row:
                continue
            for key in _SORT_BLEND_METRICS:
                if key in row:
                    r[key] = round(float(row[key]), 4)

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

    def _attach_aidet(results):
        # Inline the GRIP AI-likelihood percentage per result (RAM-only lookup from
        # the primed ~/.muser/aidet.json). `ai_pct` ∈ [0,100] when the facet knows
        # the path; absent on a miss (not yet scanned / file changed). The card UI
        # renders the badge synchronously; the ai_min filter and sort=ai read it too.
        from . import aidet as _aidet
        for r in results:
            v = _aidet.lookup(r["path"])
            if v is not None and v.get("pct") is not None:
                r["ai_pct"] = int(v["pct"])

    def _ai_pct(r):
        # AI-likelihood percentage for one result, from the inlined value or a
        # direct facet lookup. None when the facet hasn't scored this path yet.
        if "ai_pct" in r:
            return r["ai_pct"]
        from . import aidet as _aidet
        v = _aidet.lookup(r["path"])
        return int(v["pct"]) if v is not None and v.get("pct") is not None else None

    def _filter_by_ai_band(results, ai_min, ai_max):
        # Keep results whose GRIP AI-likelihood % is in [ai_min, ai_max].
        #   ai_max < 100  → "remove AI": drop likely-AI images above the ceiling.
        #   ai_min > 0    → "only AI":   keep just images above the floor.
        # Asymmetric on UNSCORED paths (no facet entry), and deliberately so:
        #   - a floor (only-AI) DROPS unscored — we can't assert they're AI;
        #   - a ceiling (remove-AI) KEEPS unscored — we can't assert they're AI,
        #     so we don't remove what we can't confirm.
        ai_min = int(ai_min or 0)
        ai_max = 100 if ai_max is None else int(ai_max)
        if ai_min <= 0 and ai_max >= 100:
            return results
        out = []
        for r in results:
            p = _ai_pct(r)
            if ai_min > 0 and (p is None or p < ai_min):
                continue
            if ai_max < 100 and p is not None and p > ai_max:
                continue
            out.append(r)
        return out

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
               ai_min: int = 0, ai_max: int = 100, sort: str | None = None, match: str = "blend",
               min_width: int | None = None, max_width: int | None = None,
               min_height: int | None = None, max_height: int | None = None,
               min_short_side: int | None = None, max_short_side: int | None = None,
               min_long_side: int | None = None, max_long_side: int | None = None,
               min_aspect: float | None = None, max_aspect: float | None = None,
               min_filesize: int | None = None, max_filesize: int | None = None):
        # ai_min: keep only results with GRIP AI-likelihood % >= ai_min (0 = off).
        # sort: "ai"/"ai_asc" reorders by AI-likelihood (default = relevance).
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
        meta = _meta_from_params(
            min_width=min_width, max_width=max_width, min_height=min_height, max_height=max_height,
            min_short_side=min_short_side, max_short_side=max_short_side,
            min_long_side=min_long_side, max_long_side=max_long_side,
            min_aspect=min_aspect, max_aspect=max_aspect,
            min_filesize=min_filesize, max_filesize=max_filesize,
        )
        # Comma syntax → multiple concepts ("depth map, person, -blurry"). Each is
        # embedded separately; a leading '-' subtracts. The shared `neg` param is
        # folded in as one more negative. With one concept and no neg this is the
        # plain single-vector path (lowercased — SigLIP/CLIP tokenizers are
        # case-sensitive and trained on lowercased alt-text).
        concepts = _parse_concepts(q)               # [(text, signed_weight), ...]
        pos = [(t, w) for t, w in concepts if w > 0] or [(q.lower(), 1.0)]
        negs = [(t, -w) for t, w in concepts if w < 0]   # positive magnitudes
        if neg and neg.strip():
            negs.append((neg.lower(), 1.0))
        if match == "all" and len(pos) >= 2:
            # Match-all (intersection): each concept must be strongly present.
            results = _search_match_all(pos, negs, k, folder, meta, neg_strength, ai_min, ai_max, sort)
        else:
            # Blend (default): weighted sum of positive concept vectors − negatives.
            if len(pos) == 1 and not negs and pos[0][1] == 1.0:
                qv = emb.embed_queries([pos[0][0]])[0]   # plain single-vector fast path
            else:
                qv = _blend_vector(emb, pos, negs, neg_strength)
            results = _run_search(qv, k, dedup, method, folder, meta=meta, ai_min=ai_min, ai_max=ai_max, sort=sort)
        refinements = _refinements([r["path"] for r in results], q)
        return {"query": q, "model": state.model_name, "results": results,
                "refinements": refinements, "concepts": concepts, "match": match}

    def _dinov2_results(path: str, k: int):
        """Reverse-image (Similar) in the OPTIONAL DINOv2 space, reusing stored
        vectors (no model load). Returns result dicts shaped exactly like
        /api/search-image's SigLIP path — run through the same _filter_results
        hide policy + per-result enrichment — or None when DINOv2 can't serve
        this path (sidecar missing or path not in the npz) so the caller falls
        back to SigLIP."""
        from . import dinov2 as _dino
        neighbors = _dino.similar(path, k=k)
        if neighbors is None:
            return None
        results = [
            {"path": p, "name": os.path.basename(p), "score": round(s, 4), "dupes": [p], "dupe_count": 1}
            for p, s in neighbors
        ]
        results = [r for r in results if r["path"] != path]
        results = _filter_results(results)
        for r in results:
            r["uid"] = uid_for(r["path"])
        _attach_dims(results, [r["path"] for r in results])
        _attach_scores(results)
        _attach_c2pa(results)
        _attach_aidet(results)
        return results

    @app.get("/api/search-image")
    def search_image(path: str, k: int = 24, dedup: bool = True, method: str = "embed", folder: str | None = None,
                     space: str | None = None,
                     min_width: int | None = None, max_width: int | None = None,
                     min_height: int | None = None, max_height: int | None = None,
                     min_short_side: int | None = None, max_short_side: int | None = None,
                     min_long_side: int | None = None, max_long_side: int | None = None,
                     min_aspect: float | None = None, max_aspect: float | None = None,
                     min_filesize: int | None = None, max_filesize: int | None = None):
        # Image-to-image search: embed the file at `path`, find nearest neighbors
        # in the index. Works for any file PIL can open — doesn't need to be indexed.
        # `space=dinov2` (optional) ranks in the stored DINOv2 space instead of
        # SigLIP, reusing precomputed vectors (no model load); absent/any other
        # value = SigLIP (default), unchanged. Falls back to SigLIP if the path
        # isn't in the DINOv2 npz.
        if not os.path.exists(path):
            raise HTTPException(404, f"no such image: {path}")
        if space == "dinov2":
            results = _dinov2_results(path, k)
            if results is not None:
                refinements = _refinements([r["path"] for r in results], os.path.basename(path))
                return {"query": f"image: {os.path.basename(path)}", "ref_path": path,
                        "model": "dinov2", "space": "dinov2",
                        "results": results, "refinements": refinements}
            # else: fall through to SigLIP (path not in npz / sidecar absent)
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
        tv = emb.embed_queries([text.lower()])[0]  # case-insensitive (see /api/search)
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

    @app.post("/api/search-upload")
    async def search_upload(
        file: UploadFile = File(...),
        k: int = Form(default=60),
        folder: str | None = Form(default=None),
    ):
        # Image-to-image search from an UPLOADED image (file picker in the search
        # bar) rather than an indexed path. Multipart form-data only (`file`,
        # optional `k`/`folder`) — FastAPI can't mix a JSON Body with File/Form in
        # one route. Bytes go to a temp file, embedded with the same image path
        # /api/search-image uses, then run through the identical _run_search
        # post-processing so result dicts are byte-for-byte shaped like /api/search
        # (the frontend card renderer works unchanged).
        import tempfile as _tempfile

        raw = await file.read()
        if not raw:
            raise HTTPException(400, "no image supplied — send multipart `file`")

        tmp = _tempfile.NamedTemporaryFile(suffix=".img", delete=False)
        try:
            tmp.write(raw)
            tmp.close()
            state.wait_ready()
            emb = state.warm()
            try:
                qv = emb.embed_images([tmp.name])[0]
            except Exception:
                raise HTTPException(400, "could not decode uploaded image")
            results = _run_search(qv, k, dedup=True, method="embed", folder=folder)
            return {"query": "image: (uploaded)", "model": state.model_name,
                    "results": results, "refinements": []}
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass

    @app.get("/api/filter")
    def filter_(limit: int = 200, offset: int = 0, folder: str | None = None,
                dedup: bool = True, ai_min: int = 0, ai_max: int = 100,
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
            # mtime-cached parse (shared with /api/search, /api/score) — avoids a
            # per-filter-request 15 MB json.loads of scores.json.
            canonical_meta = _refresh_scores_cache()["full"] or None
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

        # Centralized hide policy: drop NSFW / hidden-cluster / dead rows BEFORE
        # pagination so the page count is honest and hidden items don't leave
        # gaps. Dead rows get background-purged from the index.
        dead = _purge_dead([r["path"] for r in rows])
        rows = [r for r in rows if r["path"] not in dead and not _hidden(r["path"])]

        # AI-likelihood band: keep rows with GRIP AI % in [ai_min, ai_max]
        # (ai_max<100 removes AI; ai_min>0 keeps only AI). Rows here are raw index
        # dicts (no `ai_pct`) so _filter_by_ai_band falls back to the facet lookup.
        rows = _filter_by_ai_band(rows, ai_min, ai_max)

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
        from fastapi.responses import Response

        # Normalize input shapes. The web UI sends `items`; older CLI/MCP
        # clients still send `paths`. Treat absent flags as upscale=False.
        if req.items:
            in_items = [(it.path, bool(it.upscale)) for it in req.items]
        elif req.paths:
            in_items = [(p, False) for p in req.paths]
        else:
            raise HTTPException(400, "request must include `items` or `paths`")

        try:
            data, stats = _build_cart_zip(in_items, _load_captions())
        except ValueError:
            raise HTTPException(400, "no valid files in cart")
        return Response(
            content=data,
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="muser-cart.zip"',
                "X-Captions-Missing": str(stats["missing"]),
                "X-Captions-Total": str(stats["total"]),
                "X-Upscaled-Count": str(stats["upscaled"]),
            },
        )

    def _build_cart_zip(
        in_items: list[tuple[str, bool]], captions: dict
    ) -> tuple[bytes, dict]:
        """Assemble the cart ZIP in memory. Single source of truth for both the
        synchronous ``/api/zip`` endpoint and the async checkout job.

        ``in_items`` is a list of ``(path, upscale)``. ``captions`` is a
        ``{path: caption}`` map (caller passes ``_load_captions()``, re-loaded so
        captions just written by a checkout job are included). Returns
        ``(zip_bytes, {"missing", "total", "upscaled"})``.

        Filenames collide on basename; disambiguated with a numeric suffix.
        Files stored without compression (already-compressed image formats).
        Items flagged ``upscale=True`` are routed through muser/upscale.py (4×
        UltraSharp) — only the upscaled JPEG is bundled, under the original stem
        with a forced ``.jpg`` extension. Each image gets a sibling
        ``<stem>.txt`` caption when one exists (for fal/kohya LoRA training);
        items without a caption ship plain and bump the ``missing`` count. A
        ``manifest.json`` maps every uid → {path, arcname, caption?, upscaled}.

        Raises ``ValueError`` when no input path is a real file.
        """
        import io, zipfile

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
            raise ValueError("no valid files in cart")
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
        return buf.getvalue(), {
            "missing": missing,
            "total": len(members),
            "upscaled": upscaled_n,
        }

    @app.post("/api/checkout")
    def checkout(req: CheckoutReq):
        # Non-blocking cart checkout. Captioning (OpenAI, network-bound) and
        # upscaling (local 4×, MPS/CPU-bound) run CONCURRENTLY in a background
        # thread; the zip is built once both finish. Returns a job id immediately
        # so the UI can poll /api/checkout/status while the user keeps using
        # search/index. Deliberately does NOT touch state.task — multiple
        # checkouts coexist as separate jobs.
        from .jobs import REGISTRY
        import concurrent.futures as cf

        in_items = [(it.path, bool(it.upscale)) for it in (req.items or [])]
        in_items = [(p, u) for p, u in in_items if p and os.path.isfile(p)]
        if not in_items:
            raise HTTPException(400, "no valid files in cart")

        # ---- generate mode: spawn the fal.ai pipeline, return a run/job id ---
        if req.mode == "generate":
            from . import pipeline as _pipeline
            from .caption import caption_paths as _caption_paths
            from .upscale import upscale_4x as _upscale_4x

            run_id = REGISTRY.create("pipeline", caption_total=0, upscale_total=0)
            config = {
                "generator": req.generator,
                "lora_type": req.lora_type,
                "caption": bool(req.caption),
                "caption_prompt": req.caption_prompt,
                "cutout": bool(req.cutout),
                "upscale_refs": bool(req.upscale_refs),
                "expand_prompts": bool(req.expand_prompts),
                "brief": req.brief or "",
                "gen_prompts": req.gen_prompts,
                "n_outputs": int(req.n_outputs),
                "make_3d": bool(req.make_3d),
                "multi_angle": bool(req.multi_angle),
                "items": [{"path": p, "upscale": u} for p, u in in_items],
            }
            services = {
                "caption_paths": _caption_paths,
                "upscale_4x": _upscale_4x,
                "load_captions": _load_captions,
                "build_cart_zip": _build_cart_zip,
                "uid_for": uid_for,
                "state": state,
            }
            threading.Thread(
                target=_pipeline.run_pipeline,
                args=(run_id, config, REGISTRY, services),
                daemon=True,
                name=f"muser-pipeline-{run_id}",
            ).start()
            return {"job_id": run_id}

        have = _load_captions()
        caption_todo = [
            p for p, _ in in_items
            if req.force_caption or p not in have
        ]
        upscale_todo = [p for p, u in in_items if u]

        job_id = REGISTRY.create(
            "checkout",
            caption_total=len(caption_todo),
            upscale_total=len(upscale_todo),
        )

        def _caption_worker():
            if not caption_todo:
                return
            from .caption import caption_paths

            def on_prog(*a):
                # caption_paths calls (done, total) per image and (message) for
                # status lines; we only care about the numeric pair.
                if len(a) == 2 and isinstance(a[0], int):
                    REGISTRY.set_progress(job_id, "caption", a[0])

            written = caption_paths(
                caption_todo,
                on_progress=on_prog,
                force=req.force_caption,
                prompt=req.caption_prompt,
            )
            REGISTRY.update(job_id, captioned=len(written))
            done_paths = {r["path"] for r in written}
            for p in caption_todo:
                if p not in done_paths:
                    REGISTRY.add_error(job_id, p, "caption", "captioning failed or skipped")

        def _upscale_worker():
            if not upscale_todo:
                return
            from .upscale import upscale_4x

            done = 0
            lock = threading.Lock()

            def one(p):
                nonlocal done
                try:
                    upscale_4x(p)  # warms ~/.muser/upscale_cache; zip reuses it
                except Exception as e:
                    REGISTRY.add_error(job_id, p, "upscale", f"{type(e).__name__}: {e}")
                with lock:
                    done += 1
                    REGISTRY.set_progress(job_id, "upscale", done)

            # Modest pool — upscaling is MPS/CPU-bound, oversubscription hurts.
            with cf.ThreadPoolExecutor(max_workers=2) as ex:
                list(ex.map(one, upscale_todo))
            REGISTRY.update(job_id, upscaled=done)

        def _run():
            try:
                with cf.ThreadPoolExecutor(max_workers=2) as ex:
                    futs = [ex.submit(_caption_worker), ex.submit(_upscale_worker)]
                    for f in futs:
                        f.result()  # re-raise a fatal worker error here
                # Re-load captions so rows the caption worker just wrote are
                # bundled — this is the fix for the all-null-captions export.
                data, stats = _build_cart_zip(in_items, _load_captions())
                REGISTRY.attach_zip(job_id, data)
                REGISTRY.update(
                    job_id,
                    captions_missing=stats["missing"],
                    status="done",
                )
            except Exception as e:
                REGISTRY.update(job_id, status="error", error=str(e))

        threading.Thread(target=_run, daemon=True, name=f"muser-checkout-{job_id}").start()
        return {"job_id": job_id}

    @app.get("/api/checkout/status")
    def checkout_status(id: str):
        from .jobs import REGISTRY, public
        from . import pipeline as _pipeline

        job = REGISTRY.get(id)
        if job is not None:
            view = public(job)
            # For a pipeline job, prefer run.json's authoritative status/stages —
            # the in-RAM job mirrors them but run.json survives eviction.
            if job.get("kind") == "pipeline":
                run = _pipeline.read_run(id)
                if run is not None:
                    view["status"] = run.get("status", view["status"])
                    view["stages"] = run.get("stages", view.get("stages"))
                    view["error"] = run.get("error")
                    view["outputs"] = run.get("outputs", [])
            return view
        # Evicted from the registry but persisted on disk (pipeline runs only).
        run = _pipeline.read_run(id)
        if run is not None:
            return {
                "id": id,
                "status": run.get("status"),
                "stages": run.get("stages", []),
                "error": run.get("error"),
                "outputs": run.get("outputs", []),
            }
        raise HTTPException(404, "no such checkout job")

    @app.get("/api/checkout/zip")
    def checkout_zip(id: str):
        from .jobs import REGISTRY
        from fastapi.responses import Response

        job = REGISTRY.get(id)
        if job is None or job["status"] != "done":
            raise HTTPException(404, "checkout not ready")
        data = REGISTRY.pop_zip(id)
        if data is None:
            raise HTTPException(404, "checkout zip unavailable")
        return Response(
            content=data,
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="muser-cart.zip"'},
        )

    # ---- generate-mode pipeline browsing ------------------------------------
    def _pipeline_file_url(run_id: str, relpath: str) -> str:
        return f"/api/pipeline/{run_id}/file/{relpath}"

    def _public_run(run: dict) -> dict:
        """Return a run.json view with each output's ``url`` rewritten to a
        servable ``/api/pipeline/<id>/file/<relpath>`` link."""
        run_id = run["id"]
        outputs = []
        for o in run.get("outputs", []):
            o2 = dict(o)
            if o2.get("file"):
                o2["url"] = _pipeline_file_url(run_id, o2["file"])
            if o2.get("glb_file"):
                o2["glb_url"] = _pipeline_file_url(run_id, o2["glb_file"])
            if o2.get("thumb_file"):
                o2["thumb_url"] = _pipeline_file_url(run_id, o2["thumb_file"])
            outputs.append(o2)
        view = dict(run)
        view["outputs"] = outputs
        return view

    @app.get("/api/pipelines")
    def pipelines_list():
        from . import pipeline as _pipeline

        return {"runs": _pipeline.list_runs()}

    @app.get("/api/pipeline/{run_id}")
    def pipeline_get(run_id: str):
        from . import pipeline as _pipeline

        run = _pipeline.read_run(run_id)
        if run is None:
            raise HTTPException(404, "no such pipeline run")
        return _public_run(run)

    @app.get("/api/pipeline/{run_id}/file/{path:path}")
    def pipeline_file(run_id: str, path: str):
        from fastapi.responses import FileResponse
        from . import pipeline as _pipeline

        try:
            base = _pipeline.run_dir(run_id).resolve()
        except ValueError:
            raise HTTPException(404, "not found")  # run_id like ".." rejected
        target = (base / path).resolve()
        # Path-traversal guard: target must stay under the run dir.
        if base != target and base not in target.parents:
            raise HTTPException(403, "path escapes run dir")
        if not target.is_file():
            raise HTTPException(404, "not found")
        ext = target.suffix.lower()
        media = {
            ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".glb": "model/gltf-binary",
            ".json": "application/json",
        }.get(ext, "application/octet-stream")
        return FileResponse(str(target), media_type=media)

    @app.post("/api/pipeline/{run_id}/threed")
    def pipeline_threed(run_id: str, req: ThreeDReq):
        from . import pipeline as _pipeline
        from . import generators as _gen

        run = _pipeline.read_run(run_id)
        if run is None:
            raise HTTPException(404, "no such pipeline run")
        outputs = run.get("outputs", [])
        idx = req.output_index
        if idx < 0 or idx >= len(outputs) or outputs[idx].get("kind") != "image":
            raise HTTPException(400, "output_index is not an image output")

        d = _pipeline.run_dir(run_id)
        out_dir = d / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = d / outputs[idx]["file"]
        if not img_path.is_file():
            raise HTTPException(404, "source image missing")

        n3d = sum(1 for o in outputs if o.get("kind") == "3d")
        view_outputs: list[dict] = []
        try:
            img_url = _gen.fal_upload_path(str(img_path))
            if req.multi_angle:
                # Synthesize L/B/R views, persist whatever succeeded, then feed
                # them into the multi-view reconstruction.
                brief = (run.get("config", {}) or {}).get("brief") or ""
                views = _gen.generate_views(img_url, prompt_hint=brief)
                for label in ("left", "back", "right"):
                    vurl = (views or {}).get(label)
                    if not vurl:
                        continue
                    try:
                        vpng = _gen.download(vurl)
                        v_rel = f"view_post_{n3d:03d}_{label}.png"
                        (out_dir / v_rel).write_bytes(vpng)
                        view_outputs.append({
                            "kind": "image",
                            "file": f"outputs/{v_rel}",
                            "view": label,
                            "generator": "nano_banana",
                            "src_image_index": idx,
                        })
                    except Exception:
                        pass
                res = _gen.image_to_3d_multiview(
                    img_url,
                    (views or {}).get("left"),
                    (views or {}).get("back"),
                    (views or {}).get("right"),
                )
            else:
                res = _gen.image_to_3d(img_url)
            glb = _gen.download(res["glb"])
        except Exception as e:
            raise HTTPException(502, f"image_to_3d failed: {type(e).__name__}: {e}")

        glb_rel = f"model_post_{n3d:03d}.glb"
        (out_dir / glb_rel).write_bytes(glb)
        o3d = {
            "kind": "3d",
            "file": f"outputs/{glb_rel}",
            "glb_file": f"outputs/{glb_rel}",
            "src_image_index": idx,
        }
        thumb_url = None
        if res.get("thumbnail"):
            try:
                th = _gen.download(res["thumbnail"])
                th_rel = f"model_post_{n3d:03d}_thumb.png"
                (out_dir / th_rel).write_bytes(th)
                o3d["thumb_file"] = f"outputs/{th_rel}"
                thumb_url = _pipeline_file_url(run_id, o3d["thumb_file"])
            except Exception:
                pass

        # Append to run.json under the per-run lock so a concurrent threed call
        # (e.g. a double-click) can't read-modify-write over our output and lose
        # it. _lock_for is an RLock, so _write_run re-acquiring it is fine.
        with _pipeline._lock_for(run_id):
            fresh = _pipeline.read_run(run_id) or run
            fresh.setdefault("outputs", []).extend(view_outputs)
            fresh["outputs"].append(o3d)
            _pipeline._write_run(run_id, fresh)

        return {
            "glb_url": _pipeline_file_url(run_id, o3d["glb_file"]),
            "thumb_url": thumb_url,
        }

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
    THUMB_CACHE = MUSER_HOME / "thumb_cache"
    THUMB_CACHE.mkdir(parents=True, exist_ok=True)
    THUMB_KEY_VERSION = "v2-smartcrop"

    # ---- File-serving jail (path traversal / secret-file leak guard) ---------
    # /api/image|thumb|upscale serve files named by a `?path=` query param. Without
    # a guard, `?path=/Users/.../central/.env` or `~/.ssh/id_rsa` leaks any local
    # file. Restrict the served path to live under one of the INDEXED folder roots:
    # the parent directory of every indexed image is an allowed root (a path is
    # allowed iff it equals or is under one). That's exact for indexed images (their
    # own dir is always a root) and tolerant for thumbnails / upscales of those same
    # files. Cached by (model, row-count) so we rebuild only when the index changes,
    # not per request (the thumb path is hot).
    _allowed_roots: dict = {"key": None, "roots": []}
    _allowed_roots_lock = threading.Lock()

    def _serve_roots() -> list[str]:
        key = (state.model_name, state.index.count(state.model_name))
        with _allowed_roots_lock:
            if _allowed_roots["key"] != key:
                roots: set[str] = set()
                for p in state.index.paths(state.model_name):
                    d = os.path.dirname(os.path.realpath(p))
                    if d:
                        roots.add(d)
                # Drop any root that is a descendant of another (keep the shallowest)
                # so membership checks stay against a minimal set.
                srt = sorted(roots, key=len)
                minimal: list[str] = []
                for r in srt:
                    rsep = r + os.sep
                    if not any(rsep.startswith(m + os.sep) for m in minimal):
                        minimal.append(r)
                _allowed_roots["roots"] = minimal
                _allowed_roots["key"] = key
            return _allowed_roots["roots"]

    def _serve_allowed(path: str) -> bool:
        """True iff `path` resolves to a file under an indexed folder root."""
        try:
            rp = os.path.realpath(path)
        except OSError:
            return False
        if not os.path.isfile(rp):
            return False
        for root in _serve_roots():
            if rp == root or rp.startswith(root + os.sep):
                return True
        return False

    @app.get("/api/thumb")
    def thumb(path: str, size: int = 540, fit: str = "cover"):
        # Frontend asks for size=Math.round(280*devicePixelRatio); 540 is the
        # safe default for DPR=2. Default `fit=cover` is a smart-cropped *square*
        # (focal window picked by edge energy), so the card's `object-fit: cover`
        # is a no-op and ultra-wide / tall sources don't get squashed.
        # `fit=contain` preserves aspect (longest side = size) — used by the
        # masonry layout so tiles pack by their true proportions.
        if not _serve_allowed(path):
            raise HTTPException(404, "not found")
        fit = "contain" if fit == "contain" else "cover"
        try:
            mtime = int(os.path.getmtime(path))
        except OSError:
            raise HTTPException(404, "stat failed")
        import hashlib
        key = hashlib.blake2b(
            f"{THUMB_KEY_VERSION}|{path}|{mtime}|{size}|{fit}".encode(),
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
            if fit == "contain":
                from PIL import Image  # not in this scope otherwise (imported per-fn)
                sq = img.copy()
                sq.thumbnail((size, size), Image.LANCZOS)  # aspect-preserving
            else:
                sq = _smart_crop_square(img, size)
            tmp = cached.with_suffix(".jpg.tmp")
            sq.save(tmp, "JPEG", quality=82, optimize=False)
            tmp.replace(cached)  # atomic swap so partial writes never serve
            return FileResponse(cached, media_type="image/jpeg", headers=headers)
        except Exception:
            raise HTTPException(404, "thumb failed")

    # Formats browsers render natively — served raw (fast FileResponse path).
    # Anything else (TIFF/BMP/…) is transcoded to a browser-safe image on the
    # fly, because <img>/window.open can't display e.g. image/tiff.
    _WEB_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif", ".svg"}

    @app.get("/api/image")
    def image(path: str):
        if not _serve_allowed(path):
            raise HTTPException(404, "not found")
        ext = os.path.splitext(path)[1].lower()
        if ext in _WEB_IMG_EXT:
            return FileResponse(path)
        # Non-web formats (tiff/tif/bmp/…): decode with PIL and re-encode to a
        # format the browser can show. PNG when the source has alpha (preserve
        # transparency), JPEG otherwise (smaller for opaque photos). No downscale
        # for the full-detail view — cap only pathological dimensions like the
        # thumb path does, so huge multi-hundred-MP TIFFs don't OOM the encoder.
        from PIL import Image
        import io

        Image.MAX_IMAGE_PIXELS = None  # trusted local files
        try:
            img = Image.open(path)
            has_alpha = img.mode in ("RGBA", "LA", "PA") or (
                img.mode == "P" and "transparency" in img.info
            )
            if max(img.size) > 8192:
                img.thumbnail((8192, 8192), Image.LANCZOS)
            buf = io.BytesIO()
            if has_alpha:
                img.convert("RGBA").save(buf, "PNG", optimize=False)
                media = "image/png"
            else:
                img.convert("RGB").save(buf, "JPEG", quality=90, optimize=False)
                media = "image/jpeg"
        except Exception:
            raise HTTPException(404, "decode failed")
        return Response(content=buf.getvalue(), media_type=media)

    @app.get("/api/demo-mode")
    def demo_mode_get():
        return {"hide": _DEMO["hide"]}

    @app.post("/api/demo-mode")
    def demo_mode_set(req: DemoModeReq):
        # Flip the global hide flag (NSFW + people/selfie clusters) AND persist it
        # to ~/.muser/demo_mode.json so it survives restarts. Off reveals
        # everything except dead files. Affects all result endpoints immediately.
        _DEMO["hide"] = bool(req.on)
        try:
            _DEMO_FILE.parent.mkdir(parents=True, exist_ok=True)
            _DEMO_FILE.write_text(json.dumps({"hide": _DEMO["hide"]}))
        except Exception:
            pass
        return {"hide": _DEMO["hide"]}

    @app.post("/api/contact-sheet")
    def contact_sheet(req: ContactSheetReq):
        # Montage up to `max` images into one numbered grid JPEG — used by the MCP
        # to show the model the whole result set in a single image (token-cheap).
        # Cells are square smart-crops; a 1-based number is drawn in each corner so
        # the model can map a cell back to the Nth result.
        from PIL import Image, ImageDraw, ImageFont
        from .embedders import _load_rgb
        paths = [p for p in (req.paths or []) if os.path.isfile(p)][: max(1, req.max)]
        if not paths:
            raise HTTPException(400, "no valid paths")
        cols = max(1, req.cols)
        cell = max(64, req.cell)
        rows = (len(paths) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), (18, 18, 18))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        for i, p in enumerate(paths):
            try:
                im = _smart_crop_square(_load_rgb(p, max_side=cell * 2), cell)
            except Exception:
                continue
            x, y = (i % cols) * cell, (i // cols) * cell
            sheet.paste(im, (x, y))
            lbl = str(i + 1)
            draw.rectangle([x + 3, y + 3, x + 3 + 14 * len(lbl) + 10, y + 33], fill=(0, 0, 0))
            draw.text((x + 9, y + 6), lbl, fill=(255, 255, 255), font=font)
        buf = io.BytesIO()
        sheet.save(buf, "JPEG", quality=85)
        return Response(content=buf.getvalue(), media_type="image/jpeg",
                        headers={"X-Sheet-Count": str(len(paths))})

    @app.get("/api/upscale")
    def upscale(path: str):
        # On-demand 4× UltraSharp upscale for the photo-detail before/after view.
        # Slow (~10-20s cold, model load + tiled inference on MPS) — the frontend
        # shows a spinner. upscale_4x() caches to ~/.muser/upscale_cache; we ALSO
        # mirror the JPEG to a transient /tmp path so repeat calls reuse it and the
        # OS can reclaim it. Lazy import keeps spandrel/torch out of the warm path.
        if not _serve_allowed(path):
            raise HTTPException(404, "not found")
        import time as _time

        tmp_dir = "/tmp/muser-upscale"
        os.makedirs(tmp_dir, exist_ok=True)
        # Best-effort sweep: drop /tmp upscales older than ~1h. Never fail the request.
        try:
            cutoff = _time.time() - 3600
            for name in os.listdir(tmp_dir):
                fp = os.path.join(tmp_dir, name)
                try:
                    if os.path.isfile(fp) and os.path.getmtime(fp) < cutoff:
                        os.remove(fp)
                except OSError:
                    pass
        except OSError:
            pass

        tmp_path = os.path.join(tmp_dir, f"{uid_for(path)}.jpg")
        try:
            if os.path.isfile(tmp_path):
                data = open(tmp_path, "rb").read()
            else:
                from .upscale import upscale_4x

                data = upscale_4x(path)
                tmp = tmp_path + ".tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, tmp_path)  # atomic so partial writes never serve
        except Exception as e:  # noqa: BLE001 — log internally, never leak to client
            print(f"[upscale] failed for {path}: {e}")
            raise HTTPException(500, "upscale failed")

        headers = {}
        try:
            from PIL import Image
            import io as _io

            with Image.open(_io.BytesIO(data)) as im:
                headers["X-Upscaled-Size"] = f"{im.width}x{im.height}"
        except Exception:
            pass
        return Response(content=data, media_type="image/jpeg", headers=headers)

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
        # Centralized hide policy: drop dead / NSFW / hidden-cluster items.
        items = [it for it in items if not _hidden(it["path"])]
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

    # ---- Color facet search (separate LAB index, not the embedder) ----
    def _facet_results(pairs):
        """Turn [(path, score)] from a facet search into enriched result dicts
        matching /api/search's shape (so the same card renderer works). Drops
        results whose file is gone (the facet caches are append-by-mtime like the
        vector index)."""
        results = []
        for p, sc in pairs:
            if not os.path.exists(p):
                continue
            results.append({
                "path": p, "name": os.path.basename(p), "score": round(float(sc), 4),
                "dupes": [p], "dupe_count": 1, "uid": uid_for(p),
            })
        # Centralized hide policy (NSFW / hidden-cluster / dead-file purge).
        results = _filter_results(results)
        _attach_dims(results, [r["path"] for r in results])
        _attach_scores(results)
        _attach_c2pa(results)
        _attach_aidet(results)
        return results

    @app.get("/api/search-color")
    def search_color(hex: str, k: int = 24, folder: str | None = None, mode: str = "all"):
        # Rank indexed images by how much of `hex` they contain — LAB palette
        # distance, no model. `hex` accepts a COMMA-SEPARATED list of #rrggbb
        # colors; with >1, `mode` aggregates them: "all" (default) requires every
        # color be present (score = mean of per-color sims), "any" takes the best
        # single match (max). A single color is identical to the old behavior.
        # `#` may be URL-encoded or bare. Over-fetch then drop dead files to fill k.
        from . import color as _color
        rgbs = []
        hs = []
        for tok in hex.split(","):
            h = tok.strip().lstrip("#")
            if len(h) != 6:
                raise HTTPException(400, "hex must be one or more comma-separated #rrggbb")
            try:
                rgbs.append((int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)))
            except ValueError:
                raise HTTPException(400, "hex must be one or more comma-separated #rrggbb")
            hs.append(h)
        if not rgbs:
            raise HTTPException(400, "hex must be one or more comma-separated #rrggbb")
        if mode not in ("all", "any"):
            raise HTTPException(400, "mode must be 'all' or 'any'")
        pairs = _color.search_multi(rgbs, k=k * 2, folder=folder, mode=mode)
        results = _facet_results(pairs)[:k]
        query = "+".join(f"#{h}" for h in hs)
        return {"query": query, "model": state.model_name, "results": results, "refinements": []}

    @app.get("/api/color")
    def color_status():
        from . import color as _color
        return {"available": _color.available(), "built": _color.cache_exists(), **state.color}

    @app.post("/api/color/scan")
    def color_scan():
        if state.color["scanning"]:
            return {"started": False, "scanning": True}
        total = state.index.count(state.model_name)
        threading.Thread(target=_scan_color, daemon=True, name="muser-color-scan").start()
        return {"started": True, "total": total}

    @app.get("/api/ai-likelihood")
    def aidet_status():
        from . import aidet as _aidet
        # Coverage works for ANY scanner (in-service OR a standalone `muser aiscore`):
        # how many of the indexed images already carry an AI-likelihood score.
        scored = len(_aidet.sidecar().entries())  # mtime-aware; reflects on-disk sidecar
        indexed = state.index.count(state.model_name)
        out = {"available": _aidet.available(), "built": _aidet.cache_exists(),
               "scored": scored, "indexed": indexed, **state.aidet}
        # ETA for an in-service scan, from its own start time + throughput.
        st = state.aidet
        if st.get("scanning") and st.get("started") and st.get("total"):
            elapsed = max(1e-6, time.time() - st["started"])
            done = st.get("done", 0)
            rate = done / elapsed
            out["elapsed"] = round(elapsed, 1)
            out["rate"] = round(rate, 2)
            out["eta_seconds"] = round((st["total"] - done) / rate) if rate > 0 else None
        return out

    @app.post("/api/ai-likelihood/scan")
    def aidet_scan():
        from . import aidet as _aidet
        if not _aidet.available():
            raise HTTPException(501, "AI-likelihood detector unavailable — needs torch + timm")
        if state.aidet["scanning"]:
            return {"started": False, "scanning": True}
        total = state.index.count(state.model_name)
        threading.Thread(target=_scan_aidet, daemon=True, name="muser-aidet-scan").start()
        return {"started": True, "total": total}

    @app.get("/api/ai-likelihood/score")
    def aidet_score(path: str):
        """On-demand AI-likelihood % for one path (computes + caches on a miss)."""
        from . import aidet as _aidet
        if not _aidet.available():
            return {"available": False}
        v = _aidet.lookup(path)
        if v is None:
            v = _aidet._compute(path)  # score + (the facet caches lazily on next scan)
        return {"available": True, "pct": v.get("pct"), "logit": v.get("logit")}

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
    CLUSTERS = MUSER_HOME / "clusters.json"

    def _clusters():
        return json.loads(CLUSTERS.read_text()) if CLUSTERS.exists() else None

    def _cluster_is_hidden(cl: dict) -> bool:
        # A whole cluster is hidden when its representative caption / label
        # matches a HIDDEN_CLUSTER_MATCHERS substring (same rule that builds the
        # hidden-path set). Keeps these clusters out of the Explore grid too, not
        # just their members out of search. Only active when demo-mode "hide" is on
        # (off → show everything, matching the search/_hidden behavior).
        if not _DEMO["hide"]:
            return False
        if not HIDDEN_CLUSTER_MATCHERS:
            return False
        hay = " ".join(
            str(x).lower() for x in (
                cl.get("label") or "", cl.get("sublabel") or "", *(cl.get("reps") or []),
            )
        )
        return any(mt.lower() in hay for mt in HIDDEN_CLUSTER_MATCHERS)

    def _dinov2_clusters_payload():
        """Map ~/.muser/dinov3_clusters.json into the SAME response shape
        /api/clusters returns, so the Explore grid renders unchanged. Labels
        reuse the representative image's caption (captions.jsonl) when available,
        else "cluster N". Thumbnails use the cluster's sample paths (basenames
        resolved to full paths via the npz). Degrades to built:False when the
        DINOv2 cluster sidecar is absent."""
        from . import dinov2 as _dino
        data = _dino.clusters()
        if not data:
            return {"built": False, "clusters": []}
        caps = _load_captions()
        cl_map = data.get("clusters", {})
        out_clusters = []
        total = 0
        for cid, cl in cl_map.items():
            try:
                cid_int = int(cid)
            except (TypeError, ValueError):
                continue
            if cid_int == -1:  # noise / unclustered
                continue
            members = [p for p in _dino.cluster_members(cid_int) if not _hidden(p)]
            size = cl.get("size", len(members))
            total += size
            # Label precedence: precomputed VLM label baked into the sidecar
            # (muser/scripts label pass) → representative caption → "cluster N".
            label = (cl.get("label") or "").strip() or None
            if not label:
                for p in members:
                    cap = caps.get(p)
                    if cap:
                        label = cap
                        break
            if not label:
                label = f"cluster {cid_int}"
            out_clusters.append({
                "id": cid_int, "label": label, "sublabel": "",
                "size": size,
                "reps": [{"path": p, "uid": uid_for(p)} for p in members[:4]],
            })
        out_clusters.sort(key=lambda c: c["size"], reverse=True)
        return {
            "built": True, "method": "hdbscan", "methods": ["hdbscan"],
            "n": total, "clusters": out_clusters,
        }

    @app.get("/api/clusters")
    def clusters(method: str = "hdbscan", space: str | None = None):
        # `space=dinov2` (optional) serves the DINOv2 cluster sidecar instead of
        # the default SigLIP clusters.json; absent/any other value = SigLIP.
        if space == "dinov2":
            return _dinov2_clusters_payload()
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"built": False, "clusters": []}
        m = c["methods"][method]
        members_by_id = m.get("members", {})

        def _live_size(cl: dict) -> int:
            # Demo-mode off → show the raw cluster size (only dead-file drift,
            # out of scope). Demo-mode on → exclude hidden (NSFW) members so the
            # count matches what the grid actually shows. The full-member walk is
            # only paid in demo mode, which the user opted into.
            if not _DEMO["hide"]:
                return cl["size"]
            mem = members_by_id.get(str(cl["id"]), [])
            return sum(1 for p in mem if not _hidden(p))

        return {
            "built": True, "method": method, "methods": list(c["methods"].keys()), "n": c["n"],
            "clusters": [
                {
                    "id": cl["id"], "label": cl["label"], "sublabel": cl["sublabel"],
                    "size": _live_size(cl),
                    # reps is a list of path strings; promote to {path, uid} objects
                    # so the UI can use uid as a stable React-style key. Hidden
                    # (dead / NSFW) reps are dropped so a hidden image can't
                    # surface as a cluster thumbnail.
                    "reps": [{"path": p, "uid": uid_for(p)} for p in cl["reps"] if not _hidden(p)],
                }
                # Centralized hide policy: omit whole hidden clusters from Explore.
                for cl in m["clusters"]
                if not _cluster_is_hidden(cl)
            ],
        }

    @app.get("/api/cluster")
    def cluster_members(method: str, id: int, offset: int = 0, limit: int = 80, space: str | None = None):
        # `space=dinov2` (optional) serves members from the DINOv2 cluster
        # sidecar (its `sample` paths) instead of SigLIP; absent = SigLIP.
        if space == "dinov2":
            from . import dinov2 as _dino
            members = [p for p in _dino.cluster_members(int(id)) if not _hidden(p)]
            page = members[offset : offset + limit]
            return {
                "total": len(members),
                "members": [{"path": p, "name": os.path.basename(p), "uid": uid_for(p)} for p in page],
            }
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"total": 0, "members": []}
        members = c["methods"][method]["members"].get(str(id), [])
        # Centralized hide policy: drop dead / NSFW / hidden-cluster members
        # before pagination (no gaps, honest total).
        members = [p for p in members if not _hidden(p)]
        page = members[offset : offset + limit]
        return {
            "total": len(members),
            "members": [{"path": p, "name": os.path.basename(p), "uid": uid_for(p)} for p in page],
        }

    # ---- Explore: 3D embedding projection (point-cloud viz) ----
    # A 3D PCA of a sample of the indexed embeddings, colored by HDBSCAN cluster.
    # Expensive (table scan + SVD), so cached by (n, folder, index-mtime,
    # clusters-mtime) — repeat calls are an O(1) dict hit until a re-index or
    # re-cluster bumps a file mtime. Degrades to an empty cloud when nothing is
    # indexed or clusters.json is missing.
    _projection_cache: dict = {"key": None, "payload": None}
    _projection_lock = threading.Lock()

    @app.get("/api/projection")
    def projection(
        n: int = 2000,
        folder: str | None = None,
        space: str | None = None,
        method: str = "hdbscan",
    ):
        from . import dinov2 as _dino
        from . import projection as _proj

        is_dino = space == "dinov2"
        # Cache key tracks space + method so toggling either re-projects/re-colors.
        # DINOv2 keys on its own clusters sidecar mtime (folder scoping N/A there).
        clusters_path = _dino.CLUSTERS if is_dino else CLUSTERS
        key = _proj.cache_key(
            n, (None if is_dino else folder), state.index.db_path, clusters_path,
            space=("dinov2" if is_dino else ""), method=("" if is_dino else method),
        )
        with _projection_lock:
            if _projection_cache["key"] == key and _projection_cache["payload"] is not None:
                return _projection_cache["payload"]
        # Compute outside the lock (SVD can take a few hundred ms) so a slow
        # build doesn't serialize unrelated requests; last writer wins, which is
        # fine since the key fully determines the payload.
        if is_dino:
            payload = _proj.compute_projection_dinov2(n=n, is_hidden=_hidden)
        else:
            payload = _proj.compute_projection(
                index=state.index,
                model=state.model_name,
                clusters=_clusters(),
                n=n,
                folder=folder,
                is_hidden=_hidden,
                total_indexed=state.index.count(state.model_name),
                method=method,
            )
        with _projection_lock:
            _projection_cache["key"] = key
            _projection_cache["payload"] = payload
        return payload

    # ---- per-image captions: GPT-4o-mini + user edits (~/.muser/captions.jsonl) ----
    # JSONL is append-only. Multiple rows per path are kept on disk for history;
    # in memory we collapse to latest-wins by `ts` (so a user-edited row written
    # after the model's row beats it, even if the model row appears later in the
    # file). Cached by file mtime; the captioning pass may append while we read,
    # but `stat().st_mtime` jumps on each append, so the next request re-parses.
    # Reads-of-partial-final-line are safe: we skip lines that fail to JSON-parse.
    CAPTIONS = MUSER_HOME / "captions.jsonl"
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
        # Absent caption returns 200 with caption:null (not 404) — the cart fires
        # one GET per item, so a 404 for every uncaptioned image floods the console
        # with errors. null is the honest "no caption yet" signal.
        cap = _load_captions().get(path)
        return {"path": path, "uid": uid_for(path), "caption": cap or None}

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
    SCORES = MUSER_HOME / "scores.json"
    MUSER_DIR = MUSER_HOME
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
        s = _refresh_scores_cache()["full"]
        if not s or metric not in s.get("metrics", {}):
            return {"built": False, "items": []}
        canon = s.get("canonical") or list(s["scores"].keys())  # deduped reps (fallback for old files)
        if q and q.strip():
            # Embed the query, fetch top-500 deduped semantic matches, intersect
            # with canonical so we only sort over paths that actually have scores.
            state.wait_ready()
            emb = state.warm()
            qv = emb.embed_queries([q.strip().lower()])[0]  # case-insensitive (see /api/search)
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
        for p in ranked[offset:]:
            # Centralized hide policy: skip dead / NSFW / hidden-cluster reps.
            # (Dead-file purge runs lazily via the search/filter endpoints; here
            # we just skip so the Interesting page stays clean.)
            if _hidden(p):
                continue
            d = [dp for dp in dupes.get(p, [p]) if not _hidden(dp)] or [p]
            items.append({
                "path": p, "uid": uid_for(p), "name": os.path.basename(p),
                "score": s["scores"][p].get(metric, 0),
                "scores": s["scores"][p], "dupes": d, "dupe_count": len(d),
            })
            if len(items) >= limit:
                break
        # `total` = count of all live (non-hidden, on-disk) ranked entries, so the
        # paginator's last page lines up. Walk the whole ranked list once (O(n) over
        # ~deduped reps; _hidden is O(1) set/dict lookups + one stat). Cheap relative
        # to the per-request scores parse this endpoint used to do.
        total = sum(1 for p in ranked if not _hidden(p))
        return {"built": True, "metric": metric, "metrics": s["metrics"],
                "total": total,
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
        s = _refresh_scores_cache()["full"]
        if not s:
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
