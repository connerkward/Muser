"""
UMAP dimensionality reduction and HDBSCAN clustering with semantic labels.
Uses CLIP zero-shot for images, keyword extraction for text.
"""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import umap
import hdbscan

DATA_DIR = Path(__file__).parent.parent / "data"
FRONTEND_PUBLIC_DATA_DIR = Path(__file__).parent.parent / "frontend" / "public" / "data"

# Candidate labels for zero-shot image classification
IMAGE_LABELS = [
    "landscape painting", "portrait", "architecture", "vehicle", "car",
    "military", "nature photography", "abstract art", "digital art",
    "vintage photograph", "movie still", "product photo", "fashion",
    "interior design", "furniture", "technology", "diagram", "map",
    "meme", "screenshot", "UI design", "illustration", "sculpture",
    "historical photo", "aerial view", "food", "animal", "person",
    "building", "artwork", "document", "book cover", "poster",
    "romantic painting", "impressionist art", "classical art"
]


def load_embeddings(mode: str = None) -> dict:
    """Load raw embeddings from JSON."""
    # Try mode-specific file first, fall back to generic
    if mode:
        path = DATA_DIR / f"embeddings_raw_{mode}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
    with open(DATA_DIR / "embeddings_raw.json") as f:
        return json.load(f)


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    """Project embeddings to 2D using UMAP."""
    print("Running UMAP projection...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42
    )
    coords = reducer.fit_transform(embeddings)
    print(f"UMAP complete: {coords.shape}")
    return coords


def _fallback_image_label(image_paths: list[str]) -> str:
    """Cheap, local-only labeler from filenames/paths (no ML deps, no network)."""
    stop = {
        'image', 'images', 'img', 'jpeg', 'jpg', 'png', 'webp', 'gif',
        'final', 'copy', 'screen', 'screenshot', 'photo', 'pics', 'picture',
        'download', 'downloads', 'edited', 'edit', 'export', 'output',
    }
    counts = Counter()
    for p in (image_paths or [])[:40]:
        s = str(p).lower()
        # tokenization from path + basename
        tok = []
        cur = []
        for ch in s:
            if 'a' <= ch <= 'z':
                cur.append(ch)
            else:
                if cur:
                    tok.append(''.join(cur))
                    cur = []
        if cur:
            tok.append(''.join(cur))
        for t in tok:
            if len(t) < 4:
                continue
            if t in stop:
                continue
            counts[t] += 1
    if not counts:
        return "Images"
    return counts.most_common(1)[0][0].title()


def get_image_semantic_label(model, processor, device, image_paths: list[str], top_k: int = 1) -> str:
    """Use CLIP zero-shot to classify cluster content (fallbacks locally if unavailable)."""
    sample_paths = image_paths[:5]
    
    images = []
    for path in sample_paths:
        try:
            from PIL import Image  # lazy import (optional dep)
            img = Image.open(path).convert("RGB")
            images.append(img)
        except:
            continue
    
    if not images:
        return _fallback_image_label(image_paths)

    try:
        import torch  # lazy import (optional dep)
    except Exception:
        return _fallback_image_label(image_paths)
    
    text_inputs = processor(
        text=[f"a photo of {label}" for label in IMAGE_LABELS],
        return_tensors="pt",
        padding=True
    ).to(device)
    
    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    label_scores = torch.zeros(len(IMAGE_LABELS), device=device)
    
    for img in images:
        image_inputs = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (image_features @ text_features.T).squeeze(0)
            label_scores += similarity
    
    label_scores = label_scores / len(images)
    top_indices = label_scores.topk(top_k).indices.cpu().numpy()
    
    labels = [IMAGE_LABELS[i].title() for i in top_indices]
    # Single-label clusters only (no "X/Y" compound titles).
    return labels[0] if labels else _fallback_image_label(image_paths)


def get_text_cluster_label(items: list[dict]) -> str:
    """Extract meaningful label from text content using common words."""
    # Common stopwords to filter out
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this',
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
        'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
        'not', 'only', 'same', 'so', 'than', 'too', 'very', 'just', 'also',
        'now', 'here', 'there', 'then', 'if', 'as', 'because', 'until', 'while',
        'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below'
    }
    
    # Get first 500 chars from each item's preview or read from file
    word_counts = Counter()
    
    for item in items[:10]:  # Sample up to 10 items
        text = item.get("preview", "")
        if not text:
            try:
                text = Path(item["content"]).read_text(encoding="utf-8", errors="ignore")[:500]
            except:
                continue
        
        # Extract words
        words = text.lower().split()
        words = [w.strip('.,!?()[]{}":;-_#*') for w in words]
        words = [w for w in words if len(w) > 3 and w not in stopwords and w.isalpha()]
        word_counts.update(words)
    
    # Get top words
    top_words = [word for word, _ in word_counts.most_common(3)]
    if top_words:
        # Single-label clusters only (no "X/Y" compound titles).
        return top_words[0].title()
    return "misc"


def run_clustering(coords: np.ndarray, items: list[dict], mode: str, model=None, processor=None, device=None) -> tuple[np.ndarray, list[dict]]:
    """Cluster points using HDBSCAN and generate semantic labels."""
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=3,
        metric='euclidean'
    )
    labels = clusterer.fit_predict(coords)
    
    clusters = []
    unique_labels = set(labels) - {-1}
    
    print("Generating cluster labels...")
    for cluster_id in sorted(unique_labels):
        mask = labels == cluster_id
        centroid = coords[mask].mean(axis=0)
        
        cluster_items = [items[i] for i in range(len(items)) if mask[i]]
        
        if mode == "image":
            image_paths = [item["content"] for item in cluster_items]
            label = get_image_semantic_label(model, processor, device, image_paths)
        else:
            label = get_text_cluster_label(cluster_items)
        
        clusters.append({
            "id": int(cluster_id),
            "label": label,
            "centroid": centroid.tolist(),
            "size": int(mask.sum())
        })
        print(f"  Cluster {cluster_id}: {label} ({mask.sum()} items)")
    
    print(f"Found {len(clusters)} clusters, {(labels == -1).sum()} noise points")
    
    return labels, clusters


def main(mode: str = None):
    data = load_embeddings(mode)
    items = data["items"]
    # Prefer explicit mode passed to main() since raw files may not include "mode".
    mode = mode or data.get("mode", "image")
    
    print(f"Mode: {mode}")
    
    model, processor, device = None, None, None
    if mode == "image":
        try:
            import torch
            from transformers import CLIPProcessor, CLIPModel
            print("Loading CLIP for semantic labeling...")
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            model.eval()
        except Exception as e:
            print(f"CLIP unavailable ({e}); using filename-based fallback labels.")
    
    embeddings = np.array([item["embedding"] for item in items])
    coords = run_umap(embeddings)
    labels, clusters = run_clustering(coords, items, mode, model, processor, device)
    
    for i, item in enumerate(items):
        item["umap"] = coords[i].tolist()
        item["cluster"] = int(labels[i])
    
    output = {
        "items": items,
        "clusters": clusters,
        "mode": mode
    }
    
    # Keep legacy filenames used by the frontend/data loader.
    # - image mode: embeddings_clustered.json
    # - text mode:  embeddings_clustered_text.json
    suffix = "_text" if mode == "text" else ""
    output_path = DATA_DIR / f"embeddings_clustered{suffix}.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    # Also write directly to the Vite public data directory so the app picks it up immediately.
    # main.js loads:
    # - /data/embeddings_image.json
    # - /data/embeddings_text.json
    FRONTEND_PUBLIC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    public_name = "embeddings_text.json" if mode == "text" else "embeddings_image.json"
    public_path = FRONTEND_PUBLIC_DATA_DIR / public_name
    with open(public_path, "w") as f:
        json.dump(output, f)
    
    print(f"Saved clustered data to {output_path}")
    print(f"Saved clustered data to {public_path}")
    return output


if __name__ == "__main__":
    main()
