"""
Embedding generation for images (CLIP) or text (sentence-transformers).
"""
import argparse
import os
import json
from pathlib import Path
from PIL import Image
import torch

# Config
OUTPUT_DIR = Path(__file__).parent.parent / "data"
BATCH_SIZE = 32
MAX_ITEMS = 500
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
TEXT_EXTENSIONS = {'.md', '.txt', '.markdown'}


def get_image_files(source_dir: Path, max_count: int = MAX_ITEMS) -> list[Path]:
    """Get all image files from source directory."""
    files = []
    for f in source_dir.rglob("*"):
        if f.suffix.lower() in IMAGE_EXTENSIONS and f.is_file():
            files.append(f)
            if len(files) >= max_count:
                break
    return sorted(files, key=lambda p: p.stat().st_mtime)


def get_text_files(source_dir: Path, max_count: int = MAX_ITEMS) -> list[Path]:
    """Get all text files from source directory recursively."""
    files = []
    for f in source_dir.rglob("*"):
        if f.suffix.lower() in TEXT_EXTENSIONS and f.is_file():
            files.append(f)
            if len(files) >= max_count:
                break
    return sorted(files, key=lambda p: p.stat().st_mtime)


def load_clip_model():
    """Load CLIP model for images."""
    from transformers import CLIPProcessor, CLIPModel
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, processor, device


def load_text_model():
    """Load sentence-transformers model for text."""
    from sentence_transformers import SentenceTransformer
    print("Loading text embedding model...")
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    print(f"Model loaded on {device}")
    return model, device


def embed_images(model, processor, device, image_paths: list[Path]) -> list[dict]:
    """Generate embeddings for all images."""
    items = []
    
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i:i + BATCH_SIZE]
        images = []
        valid_paths = []
        
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path.name}: {e}")
        
        if not images:
            continue
        
        inputs = processor(images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            embeddings = outputs.cpu().numpy()
        
        # Normalize
        embeddings = embeddings / (embeddings ** 2).sum(axis=1, keepdims=True) ** 0.5
        
        for path, emb in zip(valid_paths, embeddings):
            stat = path.stat()
            items.append({
                "id": path.stem[:50],
                "type": "image",
                "content": str(path),
                "timestamp": int(stat.st_mtime),
                "embedding": emb.tolist()
            })
        
        print(f"Processed {min(i + BATCH_SIZE, len(image_paths))}/{len(image_paths)} images")
    
    return items


def embed_texts(model, device, text_paths: list[Path], max_chars: int = 8000) -> list[dict]:
    """Generate embeddings for all text files."""
    items = []
    
    for i, path in enumerate(text_paths):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception as e:
            print(f"Skipping {path.name}: {e}")
            continue
        
        if not content.strip():
            continue
        
        # Get embedding
        emb = model.encode(content, normalize_embeddings=True)
        
        stat = path.stat()
        items.append({
            "id": path.stem[:50],
            "type": "text",
            "content": str(path),
            "preview": content[:200],
            "full_text": content[:4000],  # Limit to avoid stack overflow in browser
            "timestamp": int(stat.st_mtime),
            "embedding": emb.tolist()
        })
        
        if (i + 1) % 50 == 0 or i == len(text_paths) - 1:
            print(f"Processed {i + 1}/{len(text_paths)} text files")
    
    return items


def main(mode: str = "image", source_dir: str = None, max_items: int = MAX_ITEMS):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if source_dir is None:
        if mode == "image":
            source_dir = "/Users/CONWARD/ideas-syncthing"
        else:
            source_dir = "/Users/CONWARD/Library/Mobile Documents/iCloud~md~obsidian/Documents"
    
    source_path = Path(source_dir)
    print(f"Mode: {mode}")
    print(f"Scanning {source_path}...")
    
    if mode == "image":
        files = get_image_files(source_path, max_items)
        print(f"Found {len(files)} images")
        model, processor, device = load_clip_model()
        items = embed_images(model, processor, device, files)
    else:
        files = get_text_files(source_path, max_items)
        print(f"Found {len(files)} text files")
        model, device = load_text_model()
        items = embed_texts(model, device, files)
    
    output_path = OUTPUT_DIR / f"embeddings_raw_{mode}.json"
    with open(output_path, "w") as f:
        json.dump({"items": items, "mode": mode}, f)
    
    print(f"Saved {len(items)} embeddings to {output_path}")
    return items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for images or text")
    parser.add_argument("--mode", choices=["image", "text"], default="image",
                        help="Embedding mode: image or text")
    parser.add_argument("--input", type=str, help="Source directory path")
    parser.add_argument("--max", type=int, default=MAX_ITEMS, help="Max items to process")
    
    args = parser.parse_args()
    main(mode=args.mode, source_dir=args.input, max_items=args.max)
