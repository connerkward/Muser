"""Independent quality benchmark for the triage — gpt-4o-mini as an outside judge.

Per verify-outputs-rule, the classifier must be graded by something INDEPENDENT of the
signals it was built from. The local fusion uses R (SigLIP-vs-aesthetic) + faces + EXIF +
people-tags; this benchmark asks a *different model with a different modality* (gpt-4o-mini
reading the pixels) to bucket a random sample, then compares. The VLM is a second opinion,
not gold truth — but agreement is evidence the local call is right, and each disagreement
localizes a likely error (and the saved disagreement list is exactly what to eyeball).

Stratified by the local bucket so every bucket gets coverage even when one dominates.
Reports per-bucket agreement (≈ precision treating the VLM as reference), an overall
agreement rate, and a confusion matrix; writes the disagreements to a JSON for inspection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import personalness, vlm_triage
from ..paths import data_file

BUCKETS = ("personal", "in_between", "reference")


def run(per_bucket: int = 70, seed: int = 0, progress=None) -> dict:
    entries = personalness.all_entries()
    if not entries:
        return {"error": "not classified — run `muser personal classify` first"}
    rng = random.Random(seed)
    sample: list[tuple[str, dict]] = []
    for b in BUCKETS:
        pool = [(p, e) for p, e in entries.items() if e.get("bucket") == b]
        rng.shuffle(pool)
        sample.extend(pool[:per_bucket])
    rng.shuffle(sample)

    # confusion[local][vlm]
    confusion = {a: {b: 0 for b in BUCKETS} for a in BUCKETS}
    disagreements = []
    judged = 0
    for i, (path, e) in enumerate(sample):
        v = vlm_triage._classify_one(path)
        if not v:
            continue
        local, vlm = e.get("bucket"), v["bucket"]
        if local in confusion and vlm in confusion[local]:
            confusion[local][vlm] += 1
            judged += 1
            if local != vlm:
                disagreements.append({"path": path, "local": local, "vlm": vlm,
                                      "p": e.get("p"), "unc": e.get("unc"),
                                      "sig": e.get("sig"), "vlm_reason": v.get("reason")})
        if progress and (i + 1) % 20 == 0:
            progress(f"  judged {i+1}/{len(sample)}")

    agree = sum(confusion[b][b] for b in BUCKETS)
    overall = agree / judged if judged else 0.0
    per_bucket = {}
    for b in BUCKETS:
        tot = sum(confusion[b].values())
        per_bucket[b] = {"n": tot, "agree": confusion[b][b],
                         "agreement": round(confusion[b][b] / tot, 3) if tot else None}
    result = {"judged": judged, "overall_agreement": round(overall, 3),
              "per_bucket": per_bucket, "confusion": confusion,
              "n_disagreements": len(disagreements)}
    out = data_file("benchmark.json")
    out.write_text(json.dumps({**result, "disagreements": disagreements}, indent=2))
    result["_saved"] = str(out)
    return result
