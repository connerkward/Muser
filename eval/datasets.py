"""Standard retrieval benchmarks for the harness — we reuse, not reinvent.

A benchmark is reduced to a common shape the harness can run against any model:
  - image_paths: the corpus (files on disk; we materialize HF images to a cache)
  - queries:     list of (query_id, text)
  - qrels:       {query_id: {image_path: relevance}}  (ground truth)

Flickr30k / COCO give real text->image ground truth for free (each image has
captions; the caption's correct answer is its source image).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CACHE = Path.home() / ".muser" / "bench-cache"


@dataclass
class Benchmark:
    name: str
    image_paths: list[str]
    queries: list[tuple[str, str]]  # (query_id, text)
    qrels: dict[str, dict[str, int]]


def load_flickr30k(n_images: int = 200, captions_per_image: int = 1, seed: int = 0) -> Benchmark:
    """Flickr30k test split: text->image retrieval ground truth, no labeling."""
    from datasets import load_dataset

    # Use HF's auto-converted parquet branch (dataset scripts are unsupported in
    # datasets>=4). The parquet export merges splits, so filter to the canonical test set.
    ds = load_dataset("nlphuji/flickr30k", revision="refs/convert/parquet", split="test")
    ds = ds.filter(lambda r: r.get("split") == "test")
    # Deterministic subsample.
    idx = list(range(len(ds)))
    rng = _rng(seed)
    rng.shuffle(idx)
    idx = sorted(idx[:n_images])

    out_dir = CACHE / "flickr30k"
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    queries: list[tuple[str, str]] = []
    qrels: dict[str, dict[str, int]] = {}

    for row_i in idx:
        row = ds[row_i]
        img_id = str(row.get("img_id", row.get("filename", row_i)))
        path = out_dir / f"{img_id}.jpg"
        if not path.exists():
            row["image"].convert("RGB").save(path, "JPEG", quality=90)
        path_s = str(path.resolve())
        image_paths.append(path_s)

        caps = row["caption"]
        if isinstance(caps, str):
            caps = [caps]
        for j, cap in enumerate(caps[:captions_per_image]):
            qid = f"{img_id}__{j}"
            queries.append((qid, cap))
            qrels[qid] = {path_s: 1}

    return Benchmark("flickr30k", image_paths, queries, qrels)


def _rng(seed: int):
    import random

    r = random.Random(seed)
    return r


LOADERS = {"flickr30k": load_flickr30k}
