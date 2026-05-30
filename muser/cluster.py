"""Cluster the indexed embeddings — surface "what's in the library".

Batch artifact (`muser cluster`): reduce all embeddings with UMAP, cluster with BOTH
HDBSCAN (density, auto count, noise bucket) and K-means (fixed k) so they can be
compared, label each cluster two ways — a zero-shot headline from siglip2-b's text
encoder vs a concept vocabulary, plus a BLIP caption subtitle — pick representatives,
and write ~/.muser/clusters.json. The Explore UI reads that file. Re-run after large
index changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .index import MuserIndex
from .registry import load_model

CLUSTERS_JSON = Path.home() / ".muser" / "clusters.json"

# Headline-label vocabulary — broad coverage of a creative/design/photo library.
VOCAB = [
    "a photograph", "a screenshot", "a screenshot of a website", "a screenshot of an app UI",
    "a user interface design", "an illustration", "a digital painting", "a sketch", "a drawing",
    "a logo", "a wordmark", "a poster", "a graphic design", "typography", "a font specimen",
    "a diagram", "a chart or graph", "an infographic", "a document or text", "a meme",
    "a 3D render", "product photography", "a car", "a sports car", "a vintage car",
    "a person", "a portrait of a person", "a group of people", "a face", "fashion", "clothing",
    "shoes", "a model wearing clothes", "an animal", "a dog", "a cat", "a bird", "wildlife",
    "food", "a meal", "a plant", "flowers", "nature", "a landscape", "mountains", "a beach",
    "the ocean", "a forest", "a city", "a building", "architecture", "an interior", "a room",
    "furniture", "a chair", "a sofa", "a lamp", "a kitchen", "an apartment", "a house",
    "abstract art", "a texture", "a pattern", "a gradient", "color swatches", "a moodboard",
    "technology", "a phone", "a computer", "a circuit", "a map", "a painting in a museum",
    "street photography", "black and white photography", "an icon", "an emoji", "pixel art",
    "a comic", "anime", "a video game", "a movie still", "an advertisement", "packaging design",
    "a book cover", "an album cover", "calligraphy", "handwriting", "a tattoo design",
    "jewelry", "a watch", "a motorcycle", "a bicycle", "a boat", "an airplane", "a train",
    "a sunset", "the night sky", "fireworks", "snow", "rain", "a desert", "a waterfall",
    "a sculpture", "ceramics", "a chair design", "industrial design", "a render of a product",
    "a website landing page", "a dashboard", "a mobile app", "a presentation slide",
    "a spreadsheet", "code on a screen", "a terminal", "a wireframe", "a UI component",
    "a brand identity", "a business card", "a flyer", "a magazine spread", "editorial design",
    "a chinese painting", "an antique", "wooden furniture", "a vase", "a mirror", "a rug",
]


_BLIP = {"proc": None, "model": None}


def _blip_caption(paths: list[str]) -> list[str]:
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    from .embedders import _device, _load_rgb

    if _BLIP["model"] is None:
        mid = "Salesforce/blip-image-captioning-large"
        _BLIP["proc"] = BlipProcessor.from_pretrained(mid)
        _BLIP["model"] = BlipForConditionalGeneration.from_pretrained(mid).to(_device()).eval()
    proc, model = _BLIP["proc"], _BLIP["model"]
    out: list[str] = []
    with torch.inference_mode():
        for i in range(0, len(paths), 8):
            chunk = paths[i : i + 8]
            inp = proc(images=[_load_rgb(p) for p in chunk], return_tensors="pt").to(_device())
            out += proc.batch_decode(model.generate(**inp, max_new_tokens=30), skip_special_tokens=True)
    return [c.strip() for c in out]


def _load(model: str):
    idx = MuserIndex()
    t = idx._open(model)
    if t is None:
        raise RuntimeError(f"nothing indexed for {model}")
    rows = t.search().select(["path", "vector"]).limit(100_000_000).to_list()
    paths = [r["path"] for r in rows]
    X = np.asarray([r["vector"] for r in rows], dtype=np.float32)
    return paths, X


def _build(name: str, labels, paths, X, vocab_vecs, on_progress) -> dict:
    ids = sorted({int(l) for l in labels})
    members = {str(cid): [paths[i] for i, l in enumerate(labels) if int(l) == cid] for cid in ids}
    cid_order, centroids, reps = [], [], {}
    for cid in ids:
        idxs = [i for i, l in enumerate(labels) if int(l) == cid]
        c = X[idxs].mean(0)
        c = c / (np.linalg.norm(c) + 1e-9)
        sims = X[idxs] @ c
        top = [idxs[j] for j in np.argsort(sims)[::-1][:4]]
        cid_order.append(cid)
        centroids.append(c)
        reps[cid] = [paths[i] for i in top]
    centroids = np.asarray(centroids)
    head = centroids @ vocab_vecs.T  # (C, V) cosine to each concept
    on_progress(f"  BLIP subtitles for {len(cid_order)} {name} clusters…")
    caps = _blip_caption([reps[cid][0] for cid in cid_order])
    clusters = []
    for k, cid in enumerate(cid_order):
        if cid == -1:
            label = "misc / unclustered"
        else:
            top2 = np.argsort(head[k])[::-1][:2]
            label = " / ".join(VOCAB[i].replace("a ", "").replace("an ", "") for i in top2)
        clusters.append(
            {"id": cid, "label": label, "sublabel": caps[k], "size": len(members[str(cid)]), "reps": reps[cid]}
        )
    clusters.sort(key=lambda c: -c["size"])
    return {"clusters": clusters, "members": members}


def cluster_all(model: str = "siglip2-b", kmeans_k: int = 40, min_cluster_size: int | None = None,
                on_progress=print) -> dict:
    import hdbscan as hdbscan_lib
    import umap
    from sklearn.cluster import KMeans

    paths, X = _load(model)
    n = len(paths)
    on_progress(f"loaded {n} vectors (dim {X.shape[1]})")

    on_progress("UMAP → 10d (cosine)…")
    red = umap.UMAP(n_components=10, metric="cosine", n_neighbors=30, min_dist=0.0, random_state=42).fit_transform(X)

    emb = load_model(model)
    vocab_vecs = emb.embed_queries(VOCAB)

    mcs = min_cluster_size or max(30, n // 400)
    runs = {
        "hdbscan": hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=5).fit_predict(red),
        "kmeans": KMeans(n_clusters=kmeans_k, n_init=4, random_state=42).fit_predict(red),
    }
    methods = {}
    for name, labels in runs.items():
        nc = len({int(l) for l in labels if int(l) != -1})
        on_progress(f"{name}: {nc} clusters")
        methods[name] = _build(name, labels, paths, X, vocab_vecs, on_progress)

    out = {"model": model, "n": n, "methods": methods}
    CLUSTERS_JSON.write_text(json.dumps(out))
    on_progress(f"wrote {CLUSTERS_JSON}")
    return out
