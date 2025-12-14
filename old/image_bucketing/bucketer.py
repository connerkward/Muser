"""Core image bucketing functionality."""

import os
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def list_images(root: str) -> List[str]:
    """Recursively list all image files in a directory."""
    paths = []
    for base, _, files in os.walk(root):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IMAGE_EXTS:
                paths.append(os.path.join(base, f))
    return sorted(paths)


class ImageBucketer:
    """Image bucketing and search using CLIP-style embeddings."""

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-L-14"):
        """
        Initialize the ImageBucketer.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default: clip-ViT-L-14 (best accuracy, large model).
                       Other options: clip-ViT-B-16, clip-ViT-B-32
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the model."""
        if self._model is None:
            print(f"Loading model {self.model_name}...", file=sys.stderr)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_index(
        self,
        root: str,
        categories: List[str],
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Build an in-memory index of images and their category assignments.

        Args:
            root: Root directory containing images (scanned recursively)
            categories: List of category labels
            progress_callback: Optional callback function(current, total) for progress updates

        Returns:
            Dictionary containing:
                - categories: List of category names
                - cat_emb: Tensor of category embeddings (C, D)
                - items: List of dicts with path/category/score for each image
                - img_embs: Tensor of image embeddings (N, D)
        """
        # Encode categories
        print("Encoding categories...", file=sys.stderr)
        cat_emb = self.model.encode(
            categories,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )  # shape: (C, D)

        # List images
        print(f"Scanning images under: {root}", file=sys.stderr)
        image_paths = list_images(root)
        if not image_paths:
            print("No images found.", file=sys.stderr)
            return {
                "categories": categories,
                "cat_emb": cat_emb,
                "items": [],
                "img_embs": torch.tensor([]),
            }

        total = len(image_paths)
        print(f"Found {total} images. Encoding...", file=sys.stderr)
        img_embs = []
        items: List[Dict[str, Any]] = []

        for i, path in enumerate(image_paths, 1):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {path}: {e}", file=sys.stderr)
                continue

            emb = self.model.encode(
                img, convert_to_tensor=True, normalize_embeddings=True
            )  # (D,)
            img_embs.append(emb)

            # Bucket into categories
            sims = (emb @ cat_emb.T).detach().cpu().numpy()  # (C,)
            best_idx = int(np.argmax(sims))
            best_cat = categories[best_idx]
            best_score = float(sims[best_idx])

            items.append(
                {
                    "path": path,
                    "category": best_cat,
                    "score": best_score,
                }
            )

            if progress_callback:
                progress_callback(i, total)

            # Progress updates every 10 images or at milestones
            if i % 10 == 0 or i == total or (i <= 100 and i % 25 == 0):
                pct = (i / total) * 100
                print(f"      Encoding: {i}/{total} ({pct:.1f}%)", file=sys.stderr)

        if not img_embs:
            print("No valid images encoded.", file=sys.stderr)
            return {
                "categories": categories,
                "cat_emb": cat_emb,
                "items": [],
                "img_embs": torch.tensor([]),
            }

        img_embs_tensor = torch.stack(img_embs)  # (N, D)
        print("Index built.", file=sys.stderr)

        return {
            "categories": categories,
            "cat_emb": cat_emb,  # (C, D) tensor
            "items": items,  # list of dicts with path/category/score
            "img_embs": img_embs_tensor,  # (N, D) tensor
        }

    def search_by_text(
        self,
        index: Dict[str, Any],
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search images by text query.

        Args:
            index: Index dictionary from build_index
            query: Text query string
            top_k: Number of results to return

        Returns:
            List of result dictionaries with path, category, score, and rank
        """
        q_emb = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )  # (D,)

        img_embs = index["img_embs"]  # (N, D)
        if img_embs.numel() == 0:
            return []

        sims = (q_emb @ img_embs.T).detach().cpu().numpy()  # (N,)
        top_k = min(top_k, sims.shape[0])
        idxs = np.argsort(-sims)[:top_k]

        items = index["items"]
        results = []
        for rank, idx in enumerate(idxs, 1):
            item = items[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "path": item["path"],
                    "category": item["category"],
                    "score": float(sims[idx]),
                }
            )

        return results

    def search_by_image(
        self,
        index: Dict[str, Any],
        image_path: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search images by example image.

        Args:
            index: Index dictionary from build_index
            image_path: Path to query image
            top_k: Number of results to return

        Returns:
            List of result dictionaries with path, category, score, and rank
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert("RGB")
        q_emb = self.model.encode(
            img,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )  # (D,)

        img_embs = index["img_embs"]  # (N, D)
        if img_embs.numel() == 0:
            return []

        sims = (q_emb @ img_embs.T).detach().cpu().numpy()  # (N,)
        top_k = min(top_k, sims.shape[0])
        idxs = np.argsort(-sims)[:top_k]

        items = index["items"]
        results = []
        for rank, idx in enumerate(idxs, 1):
            item = items[int(idx)]
            results.append(
                {
                    "rank": rank,
                    "path": item["path"],
                    "category": item["category"],
                    "score": float(sims[idx]),
                }
            )

        return results

    def get_bucket_summary(self, index: Dict[str, Any]) -> Dict[str, int]:
        """
        Get summary of images per bucket.

        Args:
            index: Index dictionary from build_index

        Returns:
            Dictionary mapping category names to counts
        """
        cats = [item["category"] for item in index["items"]]
        return dict(Counter(cats))

    def save_index(self, index: Dict[str, Any], filepath: str) -> None:
        """
        Save index to disk.

        Args:
            index: Index dictionary from build_index
            filepath: Path to save index file
        """
        # Convert tensors to numpy for serialization
        save_data = {
            "categories": index["categories"],
            "cat_emb": index["cat_emb"].detach().cpu().numpy(),
            "items": index["items"],
            "img_embs": index["img_embs"].detach().cpu().numpy(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)

    def load_index(self, filepath: str) -> Dict[str, Any]:
        """
        Load index from disk.

        Args:
            filepath: Path to saved index file

        Returns:
            Index dictionary compatible with build_index output
        """
        with open(filepath, "rb") as f:
            save_data = pickle.load(f)

        return {
            "categories": save_data["categories"],
            "cat_emb": torch.tensor(save_data["cat_emb"]),
            "items": save_data["items"],
            "img_embs": torch.tensor(save_data["img_embs"]),
        }


