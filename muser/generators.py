"""fal.ai generation primitives for the checkout "generate" pipeline.

Thin, stdlib-only (``urllib``) wrappers around a handful of fal REST endpoints.
No ``fal`` SDK dep — every call is a plain JSON POST to ``https://fal.run/<id>``
(synchronous) with an ``Authorization: Key <FAL_KEY>`` header, mirroring the way
``caption.py`` talks to OpenAI.

The ``FAL_KEY`` is auto-loaded from ``/Users/conner/dev/central/.env`` (a single
``FAL_KEY=...`` line). It is **never logged** — :func:`fal_key` raises a generic
RuntimeError if it's missing and otherwise returns the value only to the caller.

Endpoints used (all fal-ai):
    - ``fal-ai/gemini-3.1-flash-image-preview/edit``  — Nano Banana image gen/edit
    - ``fal-ai/birefnet/v2``                          — BiRefNet cutout (transparent PNG)
    - ``fal-ai/hunyuan3d-v3/image-to-3d``             — image → GLB
    - ``fal-ai/any-llm/vision``                       — prompt expansion (Gemini 2.5 Flash)
    - ``fal-ai/flux-lora-fast-training``              — LoRA training (expensive route)
    - ``fal-ai/flux-lora``                            — LoRA generation

Plus the fal CDN storage REST (``rest.alpha.fal.ai/storage/upload/initiate`` +
PUT) for turning local bytes/files into public URLs that downstream calls accept.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

# ---- endpoint ids (single source of truth) ----------------------------------
EP_NANO_BANANA = "fal-ai/gemini-3.1-flash-image-preview/edit"
EP_CUTOUT = "fal-ai/birefnet/v2"
# Meshy 6 won a bake-off vs Hunyuan3D-v3 and Tripo v2.5 on clean 3/4 inputs
# (cleaner topology, recognizable geometry). Multiview still uses Hunyuan since it
# accepts the left/back/right extra views.
EP_IMAGE_TO_3D = "fal-ai/meshy/v6/image-to-3d"
EP_MULTIVIEW_3D = "fal-ai/meshy/v6/multi-image-to-3d"
EP_EXPAND = "fal-ai/any-llm/vision"
EP_LORA_TRAIN = "fal-ai/flux-lora-fast-training"
EP_LORA_GEN = "fal-ai/flux-lora"

_FAL_RUN = "https://fal.run/"
_FAL_QUEUE = "https://queue.fal.run/"
_FAL_STORAGE_INITIATE = "https://rest.alpha.fal.ai/storage/upload/initiate"

# OpenAI image-edits endpoint (gpt-image-1) — the ChatGPT image generator route.
_OPENAI_IMAGE_EDITS = "https://api.openai.com/v1/images/edits"
# Cap on reference images fed to the edits endpoint (keeps payload / cost sane).
GPT_IMAGE_MAX_REFS = 8

# Nano Banana hard-caps at 14 reference image_urls (15+ → HTTP 422).
NANO_MAX_REFS = 14


# ---- auth -------------------------------------------------------------------
def _load_env_file(path: str = "/Users/conner/dev/central/.env") -> None:
    """Hand-parse ``KEY=value`` lines into os.environ (no python-dotenv dep).

    ``FAL_KEY`` from central/.env always wins over a shell export (central is the
    shared secret source of truth); other keys use ``setdefault``. Malformed lines
    / missing file: silent no-op. Same shape as ``caption._load_env_file``.
    """
    OVERRIDE = {"FAL_KEY", "OPENAI_API_KEY", "GCP_VISION_API_KEY", "ANTHROPIC_API_KEY"}
    if not os.path.isfile(path):
        return
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if not k:
                    continue
                if k in OVERRIDE:
                    os.environ[k] = v
                else:
                    os.environ.setdefault(k, v)
    except OSError:
        pass


_load_env_file()


def fal_key() -> str:
    """Return ``FAL_KEY`` from env. Raise (never echo the value) if missing."""
    key = os.environ.get("FAL_KEY")
    if not key:
        raise RuntimeError(
            "FAL_KEY not set. Add it to /Users/conner/dev/central/.env and retry."
        )
    return key


def openai_key() -> str:
    """Return ``OPENAI_API_KEY`` from env (loaded from central/.env at import).

    Raises a generic RuntimeError if missing — never echoes the value.
    """
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to /Users/conner/dev/central/.env and retry."
        )
    return key


# ---- low-level HTTP ---------------------------------------------------------
def _request_json(req: urllib.request.Request, timeout: float) -> dict:
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body else {}


def _retry(call, *, label: str):
    """3 attempts, 1s/2s/4s backoff on 429 or 5xx. Other HTTP errors raise now."""
    last: Exception | None = None
    for attempt, sleep_s in enumerate((1, 2, 4)):
        try:
            return call()
        except urllib.error.HTTPError as e:
            last = e
            if e.code != 429 and e.code < 500:
                # 4xx (bad request / unauthorized) — surface the body so the
                # caller sees the real reason. Never includes the key.
                detail = ""
                try:
                    detail = e.read().decode("utf-8", "replace")[:500]
                except Exception:
                    pass
                raise RuntimeError(f"{label}: HTTP {e.code}: {detail}") from None
            if attempt < 2:
                time.sleep(sleep_s)
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            if attempt < 2:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label}: retries exhausted: {last!r}")


# ---- fal-side progress reporting (queue API) --------------------------------
# A per-thread "status sink": when set (via set_status_sink), fal_run routes
# through the QUEUE API (submit → poll status → fetch result) instead of the sync
# endpoint, and reports each {request_id, status, queue_position} to the sink so
# callers (the generate pipeline) can surface real fal-side progress. When no
# sink is set the call stays synchronous — unchanged behavior for every other
# caller. Thread-local so concurrent generate workers each report independently.
_status_sink = threading.local()


def set_status_sink(fn) -> None:
    """Install a per-thread status callback ``fn(event: dict)`` (or None to clear)."""
    _status_sink.fn = fn


def clear_status_sink() -> None:
    _status_sink.fn = None


def _emit_status(ev: dict) -> None:
    fn = getattr(_status_sink, "fn", None)
    if fn:
        try:
            fn(ev)
        except Exception:
            pass  # progress reporting must never break a generation


def fal_run_queued(endpoint_id: str, body: dict, timeout: float = 600, poll: float = 1.5) -> dict:
    """Queue-API variant of :func:`fal_run`: submit the job, poll its status
    (emitting each transition to the active status sink), then fetch the result.
    Same return shape as the sync endpoint — fal's ``response_url`` returns the
    identical payload. ``timeout`` bounds the total queue+run wait.
    """
    key = fal_key()
    hdr = {"Authorization": f"Key {key}", "Content-Type": "application/json"}

    def _submit():
        req = urllib.request.Request(
            _FAL_QUEUE + endpoint_id, data=json.dumps(body).encode("utf-8"), headers=hdr
        )
        return _request_json(req, 60)

    sub = _retry(_submit, label=f"fal-queue:{endpoint_id}")
    req_id = sub.get("request_id") or ""
    status_url = sub.get("status_url")
    response_url = sub.get("response_url")
    if not status_url or not response_url:
        raise RuntimeError(f"fal-queue:{endpoint_id}: no status/response url in {sub!r}")
    _emit_status({"request_id": req_id, "status": "IN_QUEUE", "queue_position": sub.get("queue_position")})

    deadline = time.monotonic() + max(30.0, float(timeout))
    last = None
    while True:
        def _stat():
            req = urllib.request.Request(status_url, headers={"Authorization": f"Key {key}"})
            return _request_json(req, 30)

        st = _retry(_stat, label=f"fal-queue-status:{endpoint_id}")
        status = st.get("status") or ""
        # Emit on every IN_QUEUE poll (queue_position moves) and on transitions.
        if status != last or status == "IN_QUEUE":
            _emit_status({"request_id": req_id, "status": status, "queue_position": st.get("queue_position")})
            last = status
        if status == "COMPLETED":
            break
        if time.monotonic() > deadline:
            raise RuntimeError(f"fal-queue:{endpoint_id}: timeout after {timeout}s (req {req_id})")
        time.sleep(poll)

    def _result():
        req = urllib.request.Request(response_url, headers={"Authorization": f"Key {key}"})
        return _request_json(req, timeout=300)

    return _retry(_result, label=f"fal-queue-result:{endpoint_id}")


def fal_run(endpoint_id: str, body: dict, timeout: float = 300) -> dict:
    """POST ``body`` to fal and return JSON.

    When a status sink is active in this thread, routes through the QUEUE API so
    fal-side request id + queue position can be surfaced; otherwise a plain sync
    POST to ``https://fal.run/<endpoint_id>`` (unchanged for all other callers).
    """
    if getattr(_status_sink, "fn", None) is not None:
        return fal_run_queued(endpoint_id, body, timeout=max(float(timeout), 600.0))

    key = fal_key()

    def _call():
        req = urllib.request.Request(
            _FAL_RUN + endpoint_id,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Key {key}",
                "Content-Type": "application/json",
            },
        )
        return _request_json(req, timeout)

    return _retry(_call, label=f"fal:{endpoint_id}")


# ---- CDN upload -------------------------------------------------------------
def fal_upload_bytes(data: bytes, file_name: str, content_type: str) -> str:
    """Upload ``data`` to the fal CDN and return its public ``file_url``.

    Two-step: POST initiate → ``{upload_url, file_url}``, then PUT bytes to
    ``upload_url``. Downstream calls (cutout/3d/nano-banana refs) need URLs, not
    raw bytes — this is how a local file becomes one.
    """
    key = fal_key()

    def _init():
        req = urllib.request.Request(
            _FAL_STORAGE_INITIATE,
            data=json.dumps({"file_name": file_name, "content_type": content_type}).encode("utf-8"),
            headers={"Authorization": f"Key {key}", "Content-Type": "application/json"},
        )
        return _request_json(req, timeout=60)

    info = _retry(_init, label="fal:storage/initiate")
    upload_url = info["upload_url"]
    file_url = info["file_url"]

    def _put():
        req = urllib.request.Request(
            upload_url, data=data, method="PUT",
            headers={"Content-Type": content_type},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            r.read()
        return True

    _retry(_put, label="fal:storage/put")
    return file_url


def fal_upload_path(path: str) -> str:
    """Read a local file and upload it to the fal CDN; return its ``file_url``."""
    import mimetypes

    with open(path, "rb") as f:
        data = f.read()
    ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
    return fal_upload_bytes(data, os.path.basename(path), ctype)


def _fetch_url(url: str, timeout: float = 300) -> bytes:
    """GET ``url`` and return the raw bytes (used to download fal outputs)."""
    req = urllib.request.Request(url, headers={"User-Agent": "muser/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# convenience alias for callers (pipeline downloads outputs by URL)
download = _fetch_url


# ---- high-level generators --------------------------------------------------
def expand_prompts(brief: str, n: int, ref_urls: list[str] | None = None) -> list[str]:
    """Expand a one-line ``brief`` into ``n`` distinct generation prompts.

    Uses ``fal-ai/any-llm/vision`` (Gemini 2.5 Flash) with the reference images
    (if any) as visual context. The model is asked to return one prompt per line;
    parsing is defensive — JSON array, numbered list, or bare newlines all work.
    Falls back to replicating ``brief`` ``n`` times if the response is unusable.
    """
    n = max(1, int(n))
    system_prompt = (
        "You write image-generation prompts for a diffusion / image-edit model. "
        f"Given a brief and optional reference images, produce exactly {n} DISTINCT, "
        "concrete prompts that each realize the brief from a different angle "
        "(varying composition, pose, framing, lighting, or setting while keeping "
        "the core subject). Output ONLY the prompts, one per line, no numbering, "
        "no commentary, no markdown."
    )
    body: dict = {
        "model": "google/gemini-2.5-flash",
        "system_prompt": system_prompt,
        "prompt": f"Brief: {brief}\n\nReturn exactly {n} prompts, one per line.",
        "max_tokens": 600,
    }
    if ref_urls:
        body["image_urls"] = list(ref_urls)[:NANO_MAX_REFS]

    try:
        resp = fal_run(EP_EXPAND, body, timeout=120)
        text = (resp.get("output") or "").strip()
    except Exception:
        text = ""

    prompts = _parse_prompt_list(text)
    if not prompts:
        return [brief] * n
    # Normalize count: pad by cycling, or truncate, to exactly n.
    if len(prompts) < n:
        prompts = (prompts * ((n // len(prompts)) + 1))[:n]
    return prompts[:n]


def _parse_prompt_list(text: str) -> list[str]:
    """Pull a list of prompts out of an LLM response (JSON array or line list)."""
    if not text:
        return []
    # Try a JSON array first (the model sometimes obliges with one).
    s = text.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            out = [str(x).strip() for x in arr if str(x).strip()]
            if out:
                return out
        except Exception:
            pass
    out: list[str] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip common list markers: "1. ", "1) ", "- ", "* ".
        for i, ch in enumerate(line):
            if ch.isdigit() or ch in ".) -*\t":
                continue
            line = line[i:].strip()
            break
        # Drop wrapping quotes.
        line = line.strip().strip('"').strip("'").strip()
        if line:
            out.append(line)
    return out


def cutout(image_url: str) -> bytes:
    """BiRefNet background removal → transparent PNG bytes for ``image_url``."""
    resp = fal_run(
        EP_CUTOUT,
        {
            "image_url": image_url,
            "model": "General Use (Heavy)",
            "operating_resolution": "2048x2048",
            "output_format": "png",
            "refine_foreground": True,
        },
    )
    out_url = resp["image"]["url"]
    return _fetch_url(out_url)


def nano_banana(
    ref_urls: list[str], prompt: str, n: int = 1, resolution: str = "2K"
) -> list[str]:
    """Nano Banana (Gemini Flash Image edit). Returns generated image URLs.

    ``ref_urls`` is capped at 14 (the endpoint 422s on 15+). ``n`` maps to
    ``num_images``; the pipeline calls this with ``n=1`` per prompt so each prompt
    yields one output it can attribute.
    """
    refs = list(ref_urls or [])[:NANO_MAX_REFS]
    resp = fal_run(
        EP_NANO_BANANA,
        {
            "prompt": prompt,
            "image_urls": refs,
            "num_images": int(n),
            "resolution": resolution,
            "aspect_ratio": "1:1",
            "safety_tolerance": "6",
            "output_format": "png",
        },
    )
    return [im["url"] for im in resp.get("images", []) if im.get("url")]


def gpt_image(
    ref_paths: list[str], prompt: str, n: int = 1, size: str = "1024x1024"
) -> list[bytes]:
    """ChatGPT image generator (gpt-image-1) via OpenAI's **image edits** endpoint.

    Sends the cart reference images as multipart ``image[]`` files (capped at
    :data:`GPT_IMAGE_MAX_REFS`) so the model edits/composes from them, alongside
    the text ``prompt``. Returns one raw PNG ``bytes`` blob per generated image
    (decoded from each ``data[i].b64_json``). Stdlib ``urllib`` only — no
    ``openai`` SDK dep — mirroring ``caption.py``. The key is never logged.

    Retries on 429/5xx (3 attempts, 1s/2s/4s backoff) via :func:`_retry`.
    """
    import base64
    import mimetypes

    key = openai_key()
    refs = [p for p in (ref_paths or []) if p and os.path.isfile(p)][:GPT_IMAGE_MAX_REFS]

    # Build a multipart/form-data body by hand (stdlib has no helper).
    boundary = f"----muser{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []

    def _field(name: str, value: str) -> None:
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(
            f'Content-Disposition: form-data; name="{name}"'.encode() + crlf + crlf
        )
        parts.append(value.encode("utf-8") + crlf)

    _field("model", "gpt-image-1")
    _field("prompt", prompt)
    _field("n", str(int(n)))
    _field("size", size)
    for p in refs:
        ctype = mimetypes.guess_type(p)[0] or "image/png"
        with open(p, "rb") as f:
            data = f.read()
        parts.append(b"--" + boundary.encode() + crlf)
        parts.append(
            (
                f'Content-Disposition: form-data; name="image[]"; '
                f'filename="{os.path.basename(p)}"'
            ).encode()
            + crlf
        )
        parts.append(f"Content-Type: {ctype}".encode() + crlf + crlf)
        parts.append(data + crlf)
    parts.append(b"--" + boundary.encode() + b"--" + crlf)
    body = b"".join(parts)

    def _call():
        req = urllib.request.Request(
            _OPENAI_IMAGE_EDITS,
            data=body,
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        return _request_json(req, timeout=300)

    resp = _retry(_call, label="openai:images/edits")
    out: list[bytes] = []
    for item in resp.get("data", []) or []:
        b64 = item.get("b64_json")
        if b64:
            out.append(base64.b64decode(b64))
    if not out:
        raise RuntimeError(f"gpt_image: no image data in response: {resp!r}")
    return out


def image_to_3d(image_url: str) -> dict:
    """Meshy 6 image→3D. Returns ``{"glb": <url>, "thumbnail": <url|None>}``."""
    resp = fal_run(
        EP_IMAGE_TO_3D,
        {"image_url": image_url, "target_polycount": 50000},
    )
    glb = (resp.get("model_glb") or {}).get("url")
    thumb = (resp.get("thumbnail") or {}).get("url")
    if not glb:
        raise RuntimeError(f"image_to_3d: no model_glb in response: {resp!r}")
    return {"glb": glb, "thumbnail": thumb}


def image_to_3d_multiview(
    front_url: str,
    left_url: str | None = None,
    back_url: str | None = None,
    right_url: str | None = None,
) -> dict:
    """Meshy 6 **multi-image** image→3D.

    Meshy's ``multi-image-to-3d`` takes a flat ``image_urls`` array (1–4 images of
    the *same object from different angles*) with no per-view semantics, so the
    available views are passed in front, left, back, right order; missing views are
    simply omitted (front is required). ``target_polycount`` matches the single-image
    path. Returns ``{"glb": <url>, "thumbnail": <url|None>}`` (unchanged contract).
    """
    image_urls = [
        u for u in (front_url, left_url, back_url, right_url) if u
    ]
    body: dict = {
        "image_urls": image_urls,
        "target_polycount": 50000,
    }
    resp = fal_run(EP_MULTIVIEW_3D, body)
    glb = (resp.get("model_glb") or {}).get("url")
    thumb = (resp.get("thumbnail") or {}).get("url")
    if not glb:
        raise RuntimeError(f"image_to_3d_multiview: no model_glb in response: {resp!r}")
    return {"glb": glb, "thumbnail": thumb}


def generate_views(image_url: str, prompt_hint: str = "") -> dict:
    """Synthesize LEFT / BACK / RIGHT views of the subject in ``image_url``.

    Three Nano Banana edit calls (run concurrently), each re-rendering the same
    object/subject rotated to a new side with identical style/materials on a plain
    white background. ``prompt_hint`` is an optional short description of the
    subject prepended to the rotation instruction. Returns
    ``{"left": url, "back": url, "right": url}`` — each value is an uploaded fal
    URL, or ``None`` if that view failed (callers degrade gracefully).
    """
    subject = (prompt_hint.strip() + " ") if prompt_hint and prompt_hint.strip() else "The same object/subject "
    sides = {
        "left": "LEFT",
        "back": "BACK",
        "right": "RIGHT",
    }

    def _one(side_word: str) -> str | None:
        prompt = (
            f"{subject}rotated to show its {side_word} side, identical style and "
            "materials, plain white background, centered, product shot"
        )
        try:
            urls = nano_banana([image_url], prompt, n=1)
            if not urls:
                return None
            # Re-upload to fal CDN so the URL is a stable input for the 3D call.
            png = _fetch_url(urls[0])
            return fal_upload_bytes(png, f"view_{side_word.lower()}.png", "image/png")
        except Exception:
            return None

    out: dict[str, str | None] = {"left": None, "back": None, "right": None}
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(_one, word): key for key, word in sides.items()}
        for fut in cf.as_completed(futs):
            out[futs[fut]] = fut.result()
    return out


def train_lora(zip_url: str, trigger: str, is_style: bool) -> str:
    """Train a FLUX LoRA from a captioned image zip. Returns the LoRA file URL.

    ``zip_url`` is a fal-CDN URL of a zip built by ``_build_cart_zip`` (images +
    sibling ``<stem>.txt`` captions). Expensive route (~minutes, real $).
    """
    resp = fal_run(
        EP_LORA_TRAIN,
        {
            "images_data_url": zip_url,
            "trigger_word": trigger,
            "is_style": bool(is_style),
            "create_masks": False,
            "steps": 1500,
        },
        timeout=600,
    )
    url = (resp.get("diffusers_lora_file") or {}).get("url")
    if not url:
        raise RuntimeError(f"train_lora: no diffusers_lora_file in response: {resp!r}")
    return url


def flux_lora_generate(lora_url: str, prompt: str, n: int = 1) -> list[str]:
    """Generate images from a trained FLUX LoRA. Returns image URLs."""
    resp = fal_run(
        EP_LORA_GEN,
        {
            "prompt": prompt,
            "loras": [{"path": lora_url, "scale": 1}],
            "image_size": "square_hd",
            "num_images": int(n),
            "output_format": "png",
        },
    )
    return [im["url"] for im in resp.get("images", []) if im.get("url")]
