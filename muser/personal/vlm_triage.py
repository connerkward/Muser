"""Deferred VLM pass — resolve the uncertain band with gpt-4o-mini.

The local classifier (`personalness.py`) tags every image with an **uncertainty**
∈ [0,1]. Rather than VLM-ing everything, the user picks an uncertainty **range**
``[umin, umax]`` and the system shows the **count + estimated cost first**, then runs
gpt-4o-mini only on that band — "be generous", so the default band is wide.

Each image is classified into **personal / in_between / reference** with a short reason;
the verdict is merged back into personalness.json (`entry["vlm"]`), and the image's
`bucket` is updated to the VLM's call. Reuses caption.py's OpenAI plumbing (stdlib
urllib, `detail:"low"`, key from central/.env) — no new dependency, no SDK.
"""
from __future__ import annotations

import json

from ..caption import _api_key, _encode_image_b64, _post_openai, _retry
from . import personalness, register_heic

register_heic()  # so _encode_image_b64 (PIL) can open iPhone .heic Takeout files

# gpt-4o-mini, detail:"low" (~85 image tokens) + ~150 prompt + ~60 completion, at
# $0.15/1M input · $0.60/1M output ≈ $0.0002/image with overhead. The cost shown to the
# user before a run; deliberately a small round over-estimate so the bill never surprises.
VLM_COST_PER_IMG = 0.0002

SYSTEM = (
    "You sort a person's photo library into exactly one of three buckets. "
    "PERSONAL: snapshots of the user's own life — selfies, family/friends, pets, "
    "everyday moments, screenshots, documents, receipts. "
    "REFERENCE: images saved deliberately as visual inspiration or reference — "
    "design, art, architecture, objects, scenes, products, with no personal subject. "
    "IN_BETWEEN: a personal photo that is also aesthetically strong (a beautiful shot of "
    "family/friends/a place the user was at). "
    'Reply ONLY as compact JSON: {"bucket":"personal|in_between|reference",'
    '"confidence":0.0-1.0,"reason":"<=12 words"}.'
)


def _band(umin: float, umax: float):
    """(path, entry) pairs whose uncertainty is in [umin, umax], skipping done ones."""
    out = []
    for path, e in personalness.all_entries().items():
        u = e.get("unc")
        if u is None or "vlm" in e:
            continue
        if umin <= u <= umax:
            out.append((path, e))
    return out


def estimate(umin: float = 0.0, umax: float = 1.0) -> tuple[int, float]:
    """(count, est_cost_usd) for the uncertainty band — shown BEFORE any run."""
    n = len(_band(umin, umax))
    return n, round(n * VLM_COST_PER_IMG, 2)


def _classify_one(path: str) -> dict | None:
    try:
        b64 = _encode_image_b64(path)
    except Exception:
        return None   # unreadable image → skip, never abort the batch
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
            ]},
        ],
        "max_tokens": 80,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    key = _api_key()
    resp = _retry(lambda: _post_openai(payload, key), label="vlm-triage")
    try:
        txt = resp["choices"][0]["message"]["content"]
        d = json.loads(txt)
        b = d.get("bucket")
        if b not in ("personal", "in_between", "reference"):
            return None
        return {"bucket": b, "confidence": float(d.get("confidence", 0.0)),
                "reason": str(d.get("reason", ""))[:120]}
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def run(umin: float = 0.0, umax: float = 1.0, progress=None, on_status=None) -> dict:
    """VLM-classify the band; merge verdicts into personalness.json. Resilient: a failed
    image is skipped, never aborts. `on_status(done,total)` for the UI job poller."""
    band = _band(umin, umax)
    total = len(band)
    if progress:
        progress(f"VLM band [{umin},{umax}]: {total} images (~${total*VLM_COST_PER_IMG:.2f})")
    entries = personalness.all_entries()
    done = ok = 0
    counts: dict[str, int] = {}
    for path, _ in band:
        verdict = _classify_one(path)
        done += 1
        if verdict:
            e = entries.get(path, {})
            e["vlm"] = verdict
            e["bucket"] = verdict["bucket"]   # VLM call wins
            entries[path] = e
            counts[verdict["bucket"]] = counts.get(verdict["bucket"], 0) + 1
            ok += 1
        if on_status:
            on_status(done, total)
        if progress and done % 100 == 0:
            progress(f"  {done}/{total}  ok={ok}")
        if done % 200 == 0:                   # periodic durable checkpoint
            personalness.sidecar().save(entries)
    personalness.sidecar().save(entries)
    personalness.sidecar()._primed = False
    personalness.sidecar().prime()
    if progress:
        progress(f"done — resolved {ok}/{total}: " +
                 " ".join(f"{k}={v}" for k, v in counts.items()))
    return {"resolved": ok, "total": total, "counts": counts}
