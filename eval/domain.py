"""Domain eval — rank models on the user's OWN folder, not just Flickr.

Auto ground truth without hand-labeling: caption each image with a VLM (BLIP); the
caption becomes a query whose correct answer is its source image. Retrieval is
scored on whether the source image comes back. Captions are cached per folder so
re-runs are instant.

Caveat: in a caption-heavy/near-duplicate corpus two images can get near-identical
captions, which depresses absolute scores — but the difficulty is identical across
models, so the *relative* ranking stays valid.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from muser.index import walk_images

from .datasets import Benchmark

CAPTION_CACHE = Path.home() / ".muser" / "captions"


def _clean_caption(c: str) -> str:
    """Strip BLIP-large's hallucinated leading tokens (arafed/araffe/araflane/...)."""
    c = c.strip()
    first = c.split(" ", 1)
    if first and first[0].lower().startswith("araf"):
        c = first[1] if len(first) > 1 else ""
    return c.strip()


def caption_folder(
    folder: str,
    n: int = 500,
    seed: int = 0,
    model_id: str = "Salesforce/blip-image-captioning-large",
    batch_size: int = 8,
) -> Benchmark:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from muser.embedders import _device, _load_rgb

    random.seed(seed)
    imgs = walk_images(folder)
    sample = sorted(random.sample(imgs, min(n, len(imgs))))

    cache_path = CAPTION_CACHE / (Path(folder).name + ".json")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [p for p in sample if p not in cache]
    if todo:
        dev = _device()
        proc = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(dev).eval()
        with torch.inference_mode():
            for i in range(0, len(todo), batch_size):
                chunk = todo[i : i + batch_size]
                inp = proc(images=[_load_rgb(p) for p in chunk], return_tensors="pt").to(dev)
                caps = proc.batch_decode(model.generate(**inp, max_new_tokens=40), skip_special_tokens=True)
                for p, c in zip(chunk, caps):
                    cache[p] = _clean_caption(c)
                cache_path.write_text(json.dumps(cache))
                print(f"  captioned {min(i + batch_size, len(todo))}/{len(todo)}", end="\r", flush=True)
        print()

    queries, qrels = [], {}
    for j, p in enumerate(sample):
        qid = f"q{j}"
        queries.append((qid, cache[p]))
        qrels[qid] = {p: 1}
    return Benchmark(f"{Path(folder).name}-domain", sample, queries, qrels)
