"""Image captions — natural-language descriptions per image.

Batch artifact (`muser caption`): run a small vision-language model over every
indexed image, write one caption per image to ``~/.muser/captions.jsonl``.

Default model: Microsoft Florence-2-base-ft (~230 M params; ~270 MB on disk),
greedy decoding with the ``<MORE_DETAILED_CAPTION>`` task. 1-3 sentence
descriptions suitable for SDXL / Flux LoRA training prompts. JSONL (one row per
image) so re-runs can append and a crash mid-pass loses nothing.

Schema (per line):
    {"path": str, "caption": str, "model": "florence-2-base", "mtime": int, "ts": int}

Resume: existing rows are loaded on startup, indexed by path+mtime, and skipped
unless ``--force``. Failures are logged once and the image is skipped (no retry).

Florence-2 + transformers >= 4.55: the model's bundled
``prepare_inputs_for_generation`` was written against the old tuple-form
``past_key_values`` interface and crashes on the new ``DynamicCache``. We
monkey-patch it on the language model to translate the cache before delegating —
cheap, reliable, and isolated to this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

CAPTIONS_JSONL = Path.home() / ".muser" / "captions.jsonl"
DEFAULT_CAPTION_MODEL = "florence-2-base"
FLORENCE_REPO = "microsoft/Florence-2-base-ft"
CAPTION_TASK = "<MORE_DETAILED_CAPTION>"


def _patch_florence_cache(model) -> None:
    """Bridge Florence-2's tuple-form past_key_values to transformers' DynamicCache.

    Florence-2's ``prepare_inputs_for_generation`` does ``past_key_values[0][0].shape[2]``
    — only valid for the legacy tuple-of-tuples cache. Newer transformers passes a
    ``DynamicCache`` object. We wrap ``prepare_inputs_for_generation`` on the inner
    language model to translate the cache (or drop it if empty) before delegating.
    Idempotent — safe to call repeatedly.
    """
    lm = model.language_model
    if getattr(lm, "_muser_cache_patched", False):
        return
    orig = lm.prepare_inputs_for_generation

    def _patched(input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kw):
        if past_key_values is not None and not isinstance(past_key_values, tuple):
            try:
                legacy = past_key_values.to_legacy_cache()
                # An empty cache round-trips to a tuple-of-Nones; treat as "no cache".
                if (
                    not legacy
                    or legacy[0] is None
                    or (isinstance(legacy[0], tuple) and (not legacy[0] or legacy[0][0] is None))
                ):
                    past_key_values = None
                else:
                    past_key_values = legacy
            except Exception:
                past_key_values = None
        return orig(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            use_cache=use_cache,
            **kw,
        )

    lm.prepare_inputs_for_generation = _patched
    lm._muser_cache_patched = True


_MODEL: dict = {"proc": None, "model": None, "dev": None, "dtype": None}


def _load_florence():
    """Lazy-load Florence-2 + processor, cached at module level. Returns (proc, model, dev, dtype)."""
    if _MODEL["model"] is not None:
        return _MODEL["proc"], _MODEL["model"], _MODEL["dev"], _MODEL["dtype"]
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from .embedders import _device

    dev = _device()
    # fp16 on MPS/CUDA (well-tested for Florence-2), fp32 on CPU.
    dtype = torch.float16 if dev in ("mps", "cuda") else torch.float32
    proc = AutoProcessor.from_pretrained(FLORENCE_REPO, trust_remote_code=True)
    # attn_implementation="eager" sidesteps a transformers>=4.55 check that breaks
    # on Florence-2's custom modeling code (missing _supports_sdpa attribute).
    model = (
        AutoModelForCausalLM.from_pretrained(
            FLORENCE_REPO, trust_remote_code=True, dtype=dtype, attn_implementation="eager",
        )
        .to(dev)
        .eval()
    )
    _patch_florence_cache(model)
    _MODEL.update(proc=proc, model=model, dev=dev, dtype=dtype)
    return proc, model, dev, dtype


def _caption_batch(paths: list[str], proc, model, dev, dtype, max_new_tokens: int = 96) -> list[str | None]:
    """Caption a batch of images. Returns one caption per path (or None on failure).

    Failures fall back to per-image retries so one bad file doesn't sink the batch.
    """
    import torch

    from .embedders import _load_rgb

    imgs: list = []
    keep: list[int] = []
    for i, p in enumerate(paths):
        try:
            imgs.append(_load_rgb(p))
            keep.append(i)
        except Exception:
            pass  # caller fills None for missing slots
    out: list[str | None] = [None] * len(paths)
    if not imgs:
        return out

    def _run(batch_imgs):
        inputs = proc(text=[CAPTION_TASK] * len(batch_imgs), images=batch_imgs, return_tensors="pt",
                      padding=True).to(dev, dtype)
        # input_ids must stay int64 (the .to(dtype) cast above flips them to float16 otherwise).
        inputs["input_ids"] = inputs["input_ids"].long()
        with torch.inference_mode():
            gen = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        texts = proc.batch_decode(gen, skip_special_tokens=False)
        captions: list[str] = []
        for img, txt in zip(batch_imgs, texts):
            parsed = proc.post_process_generation(txt, task=CAPTION_TASK, image_size=(img.width, img.height))
            # post_process_generation leaves the <pad> padding tokens in — strip them
            # along with any stray whitespace so the caption is clean for downstream use.
            cap = str(parsed.get(CAPTION_TASK, "")).replace("<pad>", "").strip()
            captions.append(cap)
        return captions

    try:
        captions = _run(imgs)
        for ki, cap in zip(keep, captions):
            out[ki] = cap
        return out
    except Exception:
        pass

    # Retry one-by-one so one bad/oversized image doesn't kill the whole batch.
    for ki, img in zip(keep, imgs):
        try:
            cap = _run([img])[0]
            out[ki] = cap
        except Exception:
            out[ki] = None
    return out


def _read_existing(path: Path) -> dict[str, int]:
    """Load already-captioned {path: mtime} from JSONL. Missing/corrupt → empty."""
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


def _humantime(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def caption_all(
    model_name: str = DEFAULT_CAPTION_MODEL,
    folder: str | None = None,
    force: bool = False,
    limit: int | None = None,
    batch_size: int = 4,
    on_progress=print,
) -> dict:
    """Caption every indexed image (or those under ``folder``).

    Appends one JSON row per image to ``~/.muser/captions.jsonl``. Resumes by
    skipping paths whose (path, mtime) is already on disk unless ``force=True``.
    """
    import os

    from .index import MuserIndex
    from .registry import DEFAULT_MODEL

    if model_name != DEFAULT_CAPTION_MODEL:
        raise ValueError(f"only {DEFAULT_CAPTION_MODEL} is wired up today, got {model_name!r}")

    idx = MuserIndex()
    # The captioner is index-agnostic — we just need the list of indexed paths.
    # Use the active default embedding model's table as the source of truth.
    embed_model = DEFAULT_MODEL
    paths = idx.paths(embed_model, under=folder)
    if not paths:
        raise RuntimeError(
            f"no indexed images for embedding model {embed_model}"
            + (f" under {folder}" if folder else "")
            + " — run `muser index <folder>` first"
        )

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

    if limit is not None:
        todo = todo[:limit]

    n = len(todo)
    on_progress(f"caption: {len(paths)} indexed, {len(paths) - n} already cached, {n} to caption")
    if n == 0:
        return {"written": 0, "skipped": 0, "failed": 0, "total": len(paths)}

    on_progress(f"caption: loading {FLORENCE_REPO} (~270 MB, one-time download)…")
    proc, model, dev, dtype = _load_florence()
    on_progress(f"caption: running on {dev} dtype={dtype} batch={batch_size}")

    CAPTIONS_JSONL.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    failed = 0
    t0 = time.time()
    last_print = t0
    with CAPTIONS_JSONL.open("a", buffering=1) as out:  # line-buffered: durable mid-run
        for i in range(0, n, batch_size):
            chunk = todo[i : i + batch_size]
            try:
                caps = _caption_batch(chunk, proc, model, dev, dtype)
            except Exception as e:
                on_progress(f"  batch {i} failed entirely ({type(e).__name__}: {e})")
                caps = [None] * len(chunk)
            ts = int(time.time())
            for p, cap in zip(chunk, caps):
                if not cap:
                    failed += 1
                    continue
                try:
                    mt = int(os.path.getmtime(p))
                except OSError:
                    failed += 1
                    continue
                out.write(json.dumps({
                    "path": p, "caption": cap, "model": DEFAULT_CAPTION_MODEL,
                    "mtime": mt, "ts": ts,
                }) + "\n")
                written += 1

            done = min(i + batch_size, n)
            # Progress ping every ~50 images OR every 30s, whichever comes first.
            now = time.time()
            if (done % 50 < batch_size) or (now - last_print > 30):
                last_print = now
                elapsed = now - t0
                rate = done / max(elapsed, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                bar_w = 24
                filled = int(bar_w * done / max(n, 1))
                bar = "#" * filled + "-" * (bar_w - filled)
                on_progress(
                    f"  caption [{bar}] {done}/{n} "
                    f"({rate:.2f} img/s, elapsed {_humantime(elapsed)}, eta {_humantime(eta)}, "
                    f"failed {failed})"
                )

    elapsed = time.time() - t0
    on_progress(
        f"caption: done — wrote {written}, failed {failed} in {_humantime(elapsed)} "
        f"({written / max(elapsed, 1e-6):.2f} img/s) → {CAPTIONS_JSONL}"
    )
    return {"written": written, "skipped": len(paths) - n, "failed": failed, "total": len(paths)}


def lookup(path: str) -> str | None:
    """Most recent caption for ``path`` from ``captions.jsonl``, or None.

    JSONL append-only; later rows win on duplicate path (re-caption / --force).
    Scans the file linearly — fine for single lookups and small files; the
    service caches the read.
    """
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
