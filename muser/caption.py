"""Image captions — natural-language descriptions per image, tuned for LoRA training.

Single backend: **OpenAI ``gpt-4o-mini``** over stdlib ``urllib`` (no ``openai``
SDK dep). Image goes as a JPEG-base64 data URL with ``detail="low"`` (~85 image
tokens). Cost ~$0.001-0.005/image. Needs ``OPENAI_API_KEY`` (auto-loaded from
``/Users/conner/dev/central/.env``).

System prompt is tuned for SDXL/Flux LoRA captions (one sentence, concrete
nouns, no style descriptors, no editorializing — the LoRA learns style
implicitly as a trigger binding).

Persistence: one JSONL row per image at ``~/.muser/captions.jsonl``::

    {"path": str, "caption": str, "model": str, "mtime": int, "ts": int, "prompt": str}

The ``model`` field is ``"gpt-4o-mini"`` for fresh rows (or ``"user-edited"``
when the cart UI writes a manual override); legacy rows from earlier backends
(e.g. ``"joycaption-beta-one"``) are preserved on read. Latest-wins per path.
The ``prompt`` field records the system prompt actually used for the caption
(``DEFAULT_CAPTION_PROMPT`` unless the caller supplied a custom one at checkout);
legacy rows written before this field existed simply omit it.

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
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# OpenAI public pricing for gpt-4o-mini (per 1M tokens) — used only to print the
# post-batch cost estimate. Update if the public price changes; doesn't affect calls.
_PRICE_IN_PER_M = 0.15
_PRICE_OUT_PER_M = 0.60

# Public name for the baked LoRA-caption system prompt. ``SYSTEM_PROMPT`` is kept
# as an alias so existing references (and the request body below) keep working.
DEFAULT_CAPTION_PROMPT = (
    "You write training captions for SDXL/Flux LoRA models. Output a single sentence that:\n"
    "- Describes the subject, action, composition, and key visual elements with concrete nouns\n"
    "- Includes camera framing if obvious (close-up / wide shot / overhead)\n"
    '- Skips style descriptors (no "watercolor", "oil painting", "pixel art", "anime style", "cinematic") '
    "— the LoRA learns style as an implicit trigger binding\n"
    '- Skips editorializing words like "beautiful", "stunning", "interesting"\n'
    "- 15-35 words, no trailing punctuation, no quotes, no markdown"
)
SYSTEM_PROMPT = DEFAULT_CAPTION_PROMPT


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


def caption_image(path: str, *, prompt: str | None = None) -> tuple[str, dict]:
    """gpt-4o-mini caption for one image. Returns ``(caption, meta_dict)``.

    ``prompt`` overrides the baked ``DEFAULT_CAPTION_PROMPT`` as the system-prompt
    content when it's a non-empty string; ``None``/empty falls back to the default.

    ``meta_dict`` is the OpenAI ``usage`` field (``prompt_tokens``,
    ``completion_tokens``, ``total_tokens``) for cost accounting — ``{}`` if the
    server omitted it — plus a ``"prompt"`` key holding the system prompt actually
    used (the default, or the caller's override).
    """
    key = _api_key()
    system_prompt = prompt if (prompt and prompt.strip()) else DEFAULT_CAPTION_PROMPT
    b64 = _encode_image_b64(path)
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt},
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
    meta = dict(resp.get("usage", {}) or {})
    meta["prompt"] = system_prompt
    return text, meta


def _clean_caption(text: str) -> str:
    """Strip wrapping quotes, trailing punctuation, leading 'Caption:' chatter."""
    t = (text or "").strip()
    # Some VLMs prepend "Caption:" or "Here is a caption:" — drop common prefixes.
    for prefix in ("Caption:", "Here is a caption:", "Here's a caption:"):
        if t.lower().startswith(prefix.lower()):
            t = t[len(prefix):].lstrip()
    t = t.strip().strip('"').strip("'").rstrip(".").strip()
    return t


def _read_existing(path: Path) -> dict[str, int]:
    """Load already-captioned ``{path: mtime}`` from JSONL. Missing/corrupt → empty.

    Latest-wins is handled at the in-service ``_load_captions`` layer; for the
    skip-cache check here we only need to know whether *any* row covers
    (path, mtime). Includes legacy rows written by retired backends.
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
    prompt: str | None = None,
) -> list[dict]:
    """Caption many images via gpt-4o-mini. Skips already-cached rows unless ``force``.

    ``prompt`` (a custom system prompt supplied at checkout) is threaded through to
    every ``caption_image`` call; ``None``/empty uses ``DEFAULT_CAPTION_PROMPT``.

    Calls ``on_progress(done, total)`` per image (or ``on_progress(message: str)``
    for a textual status). Returns the list of newly-written rows
    (``{path, caption, model, mtime, ts, prompt}``) and appends each to the JSONL on
    disk. The ``prompt`` field records the resolved system prompt actually used.
    """
    # Fail fast on auth before we open the JSONL.
    _api_key()
    resolved_prompt = prompt if (prompt and prompt.strip()) else DEFAULT_CAPTION_PROMPT

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
        f"caption: {len(paths)} requested, {skipped} already cached, {n} to caption"
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
                cap, usage = caption_image(p, prompt=resolved_prompt)
                mt = int(os.path.getmtime(p))
                row = {
                    "path": p,
                    "caption": cap,
                    "model": "gpt-4o-mini",
                    "mtime": mt,
                    "ts": int(time.time()),
                    "prompt": resolved_prompt,
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
    cost = _estimate_cost(total_in, total_out)
    per = (cost / max(len(written), 1)) if written else 0.0
    on_progress(
        f"caption: done — wrote {len(written)}, failed {len(failed)} in "
        f"{elapsed:.1f}s ({per_img:.2f}s/img) · usage in={total_in} out={total_out} "
        f"· ~${cost:.3f} total (~${per:.4f}/img)"
    )
    if failed:
        on_progress(f"caption: {len(failed)} failed — first: {failed[0][0]}: {failed[0][1]}")
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
