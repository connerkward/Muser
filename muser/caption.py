"""Image captions — natural-language descriptions per image, tuned for LoRA training.

Backend-selectable: pick one per call.

  - ``gpt-4o-mini`` (default) — OpenAI vision-chat-completions over stdlib
    ``urllib`` (no ``openai`` SDK dep). Image goes as a JPEG-base64 data URL with
    ``detail="low"`` (~85 image tokens). Cost ~$0.001-0.005/image. Needs
    ``OPENAI_API_KEY`` (auto-loaded from ``/Users/conner/dev/central/.env``).

  - ``joycaption-beta-one`` — local VLM, ``fancyfeast/llama-joycaption-beta-one-hf-llava``
    (LLaVA-style: Llama-3.1-8B + SigLIP2-so400m vision tower). ~8 GB one-time
    download; ~5-15 s/image on M1 Max MPS in bfloat16. $0 / no network /
    uncensored — bake-off control for the cloud captioner above. Beta One is the
    current public release; the older Alpha-Two is **not** wired here.

Both backends share the same SDXL/Flux-LoRA system prompt (one sentence,
concrete nouns, no style descriptors, no editorializing — the LoRA learns style
implicitly as a trigger binding).

Persistence (unchanged shape): one JSONL row per image at
``~/.muser/captions.jsonl``::

    {"path": str, "caption": str, "model": str, "mtime": int, "ts": int}

The ``model`` field discriminates the backend (``"gpt-4o-mini"`` /
``"joycaption-beta-one"`` / ``"user-edited"``); latest-wins on read.

Resume: existing (path, mtime) rows are skipped unless ``force=True``. JSONL is
append-only — a crash mid-pass loses only the in-flight image.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

CAPTIONS_JSONL = Path.home() / ".muser" / "captions.jsonl"
DEFAULT_BACKEND = "gpt-4o-mini"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Stable string written to the JSONL `model` field per backend. Keep in sync with
# the `--backend` CLI choices and the cart-modal <select> in web/app.html.
BACKENDS = ("gpt-4o-mini", "joycaption-beta-one")

# OpenAI public pricing for gpt-4o-mini (per 1M tokens) — used only to print the
# post-batch cost estimate. Update if the public price changes; doesn't affect calls.
_PRICE_IN_PER_M = 0.15
_PRICE_OUT_PER_M = 0.60

SYSTEM_PROMPT = (
    "You write training captions for SDXL/Flux LoRA models. Output a single sentence that:\n"
    "- Describes the subject, action, composition, and key visual elements with concrete nouns\n"
    "- Includes camera framing if obvious (close-up / wide shot / overhead)\n"
    '- Skips style descriptors (no "watercolor", "oil painting", "pixel art", "anime style", "cinematic") '
    "— the LoRA learns style as an implicit trigger binding\n"
    '- Skips editorializing words like "beautiful", "stunning", "interesting"\n'
    "- 15-35 words, no trailing punctuation, no quotes, no markdown"
)

# JoyCaption user prompt — same shape, restated as a direct instruction (its
# chat template wants the substantive ask on the user side; the system slot
# stays generic per the model card to keep its baked-in chat behavior healthy).
_JOYCAPTION_USER_PROMPT = (
    "Write a descriptive caption for this image targeting SDXL/Flux LoRA "
    "training. Output a single sentence with concrete nouns describing "
    "subject, action, composition, and camera framing. Do NOT use style "
    "descriptors (no 'watercolor', 'oil painting', 'pixel art', 'anime "
    "style', 'cinematic'). Do NOT use editorializing words ('beautiful', "
    "'stunning'). 15-35 words, no trailing punctuation, no quotes."
)


def _load_env_file(path: str = "/Users/conner/dev/central/.env") -> None:
    """Hand-parse ``KEY=value`` lines into os.environ (no python-dotenv dep).

    central/.env is the source of truth for shared secrets, so this **overwrites**
    pre-existing shell exports for select keys (notably ``OPENAI_API_KEY`` — a
    common local LM Studio stub in the user's shell would otherwise win and shadow
    the real production key here). Other unrelated env vars use ``setdefault`` to
    preserve shell intent. Malformed lines / missing file: silent no-op.
    """
    # Keys whose central/.env value should always win over a shell export.
    OVERRIDE = {"OPENAI_API_KEY", "GCP_VISION_API_KEY", "ANTHROPIC_API_KEY"}
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


# ---------------------------------------------------------------------------
# Backend 1: OpenAI gpt-4o-mini (cloud, default)
# ---------------------------------------------------------------------------
def _encode_image_b64(path: str, max_side: int = 1024) -> str:
    """Read ``path``, downscale (long side ≤ ``max_side``), JPEG-encode, base64.

    Reuses the project's ``_load_rgb`` so corrupt files raise consistently.
    ``detail="low"`` on the OpenAI side keeps image tokens to ~85 regardless,
    but we still resize locally to bound the upload size.
    """
    from .embedders import _load_rgb  # lazy: pulls PIL

    img = _load_rgb(path)
    # _load_rgb already caps to ≤1024px, but be defensive in case its cap moves.
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize((max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _post_openai(payload: dict, api_key: str, timeout: float = 90.0) -> dict:
    """POST chat-completions; raise urllib.error.HTTPError on non-2xx."""
    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _retry(call, *, label: str = "openai"):
    """3 attempts, 1s/2s/4s sleep on 429 or 5xx. Other errors raise immediately."""
    last_exc: Exception | None = None
    for attempt, sleep_s in enumerate((1, 2, 4)):
        try:
            return call()
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code != 429 and e.code < 500:
                # Not a transient class — surface immediately (401 invalid key, 400 bad
                # payload, etc.). Caller wants the specific error.
                raise
            if attempt < 2:
                time.sleep(sleep_s)
        except (urllib.error.URLError, TimeoutError) as e:
            last_exc = e
            if attempt < 2:
                time.sleep(sleep_s)
    raise RuntimeError(f"{label}: retries exhausted: {last_exc!r}")


def _api_key() -> str:
    """Pull OPENAI_API_KEY from env. Raise RuntimeError if missing — never echo the value."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Add it to /Users/conner/dev/central/.env and retry."
        )
    return key


def _caption_via_openai(path: str) -> tuple[str, dict]:
    """gpt-4o-mini caption. Returns ``(caption, usage_dict)``.

    ``usage_dict`` is the OpenAI ``usage`` field (``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``) for cost accounting; ``{}`` if the
    server omitted it.
    """
    key = _api_key()
    b64 = _encode_image_b64(path)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64}",
                            "detail": "low",
                        },
                    }
                ],
            },
        ],
        "max_tokens": 200,
    }
    resp = _retry(lambda: _post_openai(payload, key), label="gpt-4o-mini")
    try:
        text = resp["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError) as e:
        raise RuntimeError(f"unexpected OpenAI response shape: {e}: {resp!r}")
    # Defensive cleanup: strip wrapping quotes / trailing punctuation in case the
    # model ignores the system prompt's "no trailing punctuation" rule.
    text = _clean_caption(text)
    return text, resp.get("usage", {}) or {}


# ---------------------------------------------------------------------------
# Backend 2: JoyCaption Beta One (local LLaVA, free, ~10s/image on MPS)
# ---------------------------------------------------------------------------
# Singleton stash, mirroring the `_FALCON` pattern in score.py — the model is
# ~8 GB; loading once per batch is unavoidable, loading once per *image* is not.
# Loaded lazily so `import muser.caption` stays cheap and the gpt-4o-mini path
# never touches transformers / the 8 GB weights cache.
_JOYCAPTION = {"model": None, "processor": None, "device": None, "dtype": None}
_JOYCAPTION_REPO = "fancyfeast/llama-joycaption-beta-one-hf-llava"


def _joycaption_load():
    """Load + cache the model on first call. Returns (model, processor, device, dtype).

    bfloat16 on MPS works on macOS Sequoia / PyTorch ≥ 2.4 (tested 2026-06-03 on
    M1 Max). CUDA also uses bf16. CPU falls back to float32 — the model is too
    large to be useful there anyway, but it won't crash.
    """
    if _JOYCAPTION["model"] is not None:
        return _JOYCAPTION["model"], _JOYCAPTION["processor"], _JOYCAPTION["device"], _JOYCAPTION["dtype"]

    import torch
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    from .embedders import _device

    device = _device()
    # bf16 native for Llama-3.1 + SigLIP2; fp32 on CPU because bf16 matmul on
    # CPU is patchy and torture-slow for an 8B model.
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

    # use_fast=True picks the Rust-backed image processor — measurably quicker on
    # the per-image CPU pre-encode path; the slight numerical drift is irrelevant
    # for a 384x384 SigLIP image tower.
    processor = AutoProcessor.from_pretrained(_JOYCAPTION_REPO, use_fast=True)
    # transformers ≥ 4.52 renamed the kwarg `torch_dtype` → `dtype`; the older
    # name still works but warns. Use the new spelling.
    model = LlavaForConditionalGeneration.from_pretrained(
        _JOYCAPTION_REPO,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model = model.to(device)
    model.eval()

    _JOYCAPTION.update(model=model, processor=processor, device=device, dtype=dtype)
    return model, processor, device, dtype


def _caption_via_joycaption(path: str) -> tuple[str, dict]:
    """JoyCaption Beta One caption. Returns ``(caption, {})`` — no usage/cost.

    Greedy-ish sampling (the model card's recommended ``temperature=0.6, top_p=0.9``)
    plus ``max_new_tokens=120`` — caps a 15-35-word caption with headroom for
    occasional verbosity. We don't enforce length; the system prompt does the
    asking and a final cleanup pass strips quotes / trailing punctuation.
    """
    import torch

    from .embedders import _load_rgb

    model, processor, device, dtype = _joycaption_load()
    img = _load_rgb(path)  # PIL.RGB, already capped to ≤1024px

    # The model card stresses that HF's LLaVA chat handling is fragile — use the
    # exact apply_chat_template + processor combo it ships with. Anything else
    # tends to double-<bos> or omit the image token.
    convo = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _JOYCAPTION_USER_PROMPT},
    ]
    convo_string = processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
    assert isinstance(convo_string, str)

    inputs = processor(text=[convo_string], images=[img], return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        generate_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            top_k=None,
            suppress_tokens=None,
            use_cache=True,
        )[0]

    # Trim off the prompt; tokenizer.decode skips special tokens so the chat
    # template's <eot_id> / image-token placeholders don't bleed into output.
    generate_ids = generate_ids[inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return _clean_caption(text), {}


def _clean_caption(text: str) -> str:
    """Strip wrapping quotes, trailing punctuation, leading 'Caption:' chatter."""
    t = (text or "").strip()
    # Some VLMs prepend "Caption:" or "Here is a caption:" — drop common prefixes.
    for prefix in ("Caption:", "Here is a caption:", "Here's a caption:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].lstrip()
    t = t.strip().strip('"').strip("'").rstrip(".").strip()
    return t


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------
def caption_image(path: str, backend: str = DEFAULT_BACKEND) -> tuple[str, dict]:
    """Caption one image with the named backend. Returns ``(caption, usage_dict)``.

    ``usage_dict`` is non-empty only for backends that meter (gpt-4o-mini). Local
    backends return ``{}``.
    """
    if backend == "gpt-4o-mini":
        return _caption_via_openai(path)
    if backend == "joycaption-beta-one":
        return _caption_via_joycaption(path)
    raise ValueError(f"unknown caption backend: {backend!r} (choose from {BACKENDS})")


def _read_existing(path: Path) -> dict[str, int]:
    """Load already-captioned ``{path: mtime}`` from JSONL. Missing/corrupt → empty.

    Latest-wins is handled at the in-service ``_load_captions`` layer; for the
    skip-cache check here we only need to know whether *any* row covers
    (path, mtime). Backend-agnostic by design: if gpt-4o-mini already wrote a
    row at mtime M, a JoyCaption pass over the same image with no ``--force``
    skips it (the user picked one backend per pass; the JSONL keeps both
    historical rows if they did force a re-caption).
    """
    if not path.exists():
        return {}
    have: dict[str, int] = {}
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                p = row.get("path")
                mt = int(row.get("mtime", 0))
                if p and isinstance(p, str):
                    have[p] = mt
            except Exception:
                pass
    return have


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens / 1_000_000.0) * _PRICE_IN_PER_M + (
        completion_tokens / 1_000_000.0
    ) * _PRICE_OUT_PER_M


def caption_paths(
    paths: list[str],
    on_progress=print,
    force: bool = False,
    backend: str = DEFAULT_BACKEND,
) -> list[dict]:
    """Caption many images with ``backend``. Skips already-cached rows unless ``force``.

    Calls ``on_progress(done, total)`` per image (or ``on_progress(message: str)``
    for a textual status). Returns the list of newly-written rows
    (``{path, caption, model, mtime, ts}``) and appends each to the JSONL on disk.
    The ``model`` field is the backend name verbatim — so a downstream reader can
    tell which captioner wrote each row.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown caption backend: {backend!r} (choose from {BACKENDS})")

    # Fail fast on auth / model-load problems before we open the JSONL.
    if backend == "gpt-4o-mini":
        _api_key()
    elif backend == "joycaption-beta-one":
        # Cheap pre-check: import torch & transformers so a missing dep raises
        # here, not deep inside the loop. The model itself loads lazily on the
        # first caption call (or earlier if the caller already warmed it).
        import torch  # noqa: F401
        import transformers  # noqa: F401

    have = {} if force else _read_existing(CAPTIONS_JSONL)
    todo: list[str] = []
    for p in paths:
        try:
            mt = int(os.path.getmtime(p))
        except OSError:
            continue
        if not force and have.get(p) == mt:
            continue
        todo.append(p)

    n = len(todo)
    skipped = len(paths) - n
    on_progress(
        f"caption[{backend}]: {len(paths)} requested, {skipped} already cached, {n} to caption"
    )
    if n == 0:
        return []

    CAPTIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    failed: list[tuple[str, str]] = []
    total_in = 0
    total_out = 0
    t0 = time.time()
    with CAPTIONS_JSONL.open("a", buffering=1) as out:
        for i, p in enumerate(todo, start=1):
            try:
                cap, usage = caption_image(p, backend=backend)
                mt = int(os.path.getmtime(p))
                row = {
                    "path": p,
                    "caption": cap,
                    "model": backend,
                    "mtime": mt,
                    "ts": int(time.time()),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                written.append(row)
                total_in += int(usage.get("prompt_tokens") or 0)
                total_out += int(usage.get("completion_tokens") or 0)
            except Exception as e:
                failed.append((p, f"{type(e).__name__}: {e}"))
            # Numeric progress hook (done, total) — service.py uses this; CLI's
            # `print` callback just stringifies the call. Wrap so a (done,total) call
            # works for both.
            try:
                on_progress(i, n)
            except TypeError:
                pass

    elapsed = time.time() - t0
    per_img = elapsed / max(len(written), 1) if written else 0.0
    if backend == "gpt-4o-mini":
        cost = _estimate_cost(total_in, total_out)
        per = (cost / max(len(written), 1)) if written else 0.0
        on_progress(
            f"caption[{backend}]: done — wrote {len(written)}, failed {len(failed)} in "
            f"{elapsed:.1f}s ({per_img:.2f}s/img) · usage in={total_in} out={total_out} "
            f"· ~${cost:.3f} total (~${per:.4f}/img)"
        )
    else:
        # Local backends: no $ — surface wall-clock per image instead.
        on_progress(
            f"caption[{backend}]: done — wrote {len(written)}, failed {len(failed)} in "
            f"{elapsed:.1f}s ({per_img:.2f}s/img) · local · $0"
        )
    if failed:
        on_progress(f"caption[{backend}]: {len(failed)} failed — first: {failed[0][0]}: {failed[0][1]}")
    return written


def lookup(path: str) -> str | None:
    """Most recent caption for ``path`` from ``captions.jsonl``, or ``None``."""
    if not CAPTIONS_JSONL.exists():
        return None
    found: str | None = None
    with CAPTIONS_JSONL.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if row.get("path") == path:
                    found = row.get("caption") or found
            except Exception:
                continue
    return found
