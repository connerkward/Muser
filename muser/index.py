"""LanceDB-backed vector index — the real retrieval engine.

One embedded LanceDB at ~/.muser/db. One table per model (vector dims differ),
named ``img__<model>``. Rows: {path, mtimeMs, size, vector}. Search is cosine
over L2-normalized vectors. The eval harness uses this same engine so benchmarks
measure the real path, not a separate in-memory shortcut.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .embedders import Embedder

DEFAULT_DB = Path.home() / ".muser" / "db"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".tiff"}


def walk_images(folder: str | Path, recursive: bool = True) -> list[str]:
    folder = Path(folder)
    out: list[str] = []
    it = folder.rglob("*") if recursive else folder.glob("*")
    for p in it:
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS:
            out.append(str(p.resolve()))
    return sorted(out)


@dataclass
class IndexResult:
    added: int
    updated: int
    removed: int
    total: int


class MuserIndex:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        import lancedb

        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))

    @staticmethod
    def _table(model: str) -> str:
        return "img__" + model.replace("/", "_").replace("-", "_")

    def _open(self, model: str):
        name = self._table(model)
        return self.db.open_table(name) if name in self.db.table_names() else None

    def count(self, model: str) -> int:
        t = self._open(model)
        return t.count_rows() if t else 0

    def add_images(
        self,
        model: str,
        embedder: Embedder,
        paths: Sequence[str],
        batch_size: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> int:
        """Embed and upsert the given image paths into the model's table."""
        if not paths:
            return 0
        good_paths: list[str] = []
        vectors: list[np.ndarray] = []
        for i in range(0, len(paths), batch_size):
            chunk = list(paths[i : i + batch_size])
            try:
                vectors.append(embedder.embed_images(chunk, batch_size=batch_size))
                good_paths.extend(chunk)
            except Exception:
                # A bad file in the batch — retry one-by-one, skipping failures.
                for p in chunk:
                    try:
                        vectors.append(embedder.embed_images([p], batch_size=1))
                        good_paths.append(p)
                    except Exception as e:
                        print(f"  skipped {p}: {e}")
            if on_progress:
                on_progress(min(i + batch_size, len(paths)), len(paths))
        if not good_paths:
            return 0
        vecs = np.concatenate(vectors, axis=0)
        rows = []
        for p, v in zip(good_paths, vecs):
            st = os.stat(p)
            rows.append(
                {"path": p, "mtimeMs": st.st_mtime * 1000.0, "size": st.st_size, "vector": v.tolist()}
            )
        name = self._table(model)
        t = self._open(model)
        if t is None:
            self.db.create_table(name, rows)
        else:
            t.add(rows)
        return len(rows)

    def index_folder(
        self,
        folder: str | Path,
        model: str,
        embedder: Embedder,
        recursive: bool = True,
        batch_size: int = 16,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> IndexResult:
        on_disk = walk_images(folder, recursive)
        on_disk_set = set(on_disk)

        existing: dict[str, float] = {}
        t = self._open(model)
        if t is not None:
            for r in t.search().select(["path", "mtimeMs"]).limit(10_000_000).to_list():
                existing[r["path"]] = float(r["mtimeMs"])

        to_embed = [p for p in on_disk if abs(existing.get(p, -1) - os.stat(p).st_mtime * 1000.0) > 1.0]
        to_remove = [p for p in existing if p not in on_disk_set]

        # Drop stale/replaced rows before re-adding.
        stale = to_remove + [p for p in to_embed if p in existing]
        if t is not None and stale:
            t.delete(" OR ".join("path = '" + p.replace("'", "''") + "'" for p in stale))

        added_paths = [p for p in to_embed if p not in existing]
        self.add_images(model, embedder, to_embed, batch_size=batch_size, on_progress=on_progress)

        return IndexResult(
            added=len(added_paths),
            updated=len(to_embed) - len(added_paths),
            removed=len(to_remove),
            total=self.count(model),
        )

    def search(self, model: str, query_vec: np.ndarray, k: int = 12) -> list[tuple[str, float]]:
        t = self._open(model)
        if t is None:
            return []
        rows = t.search(query_vec.tolist()).distance_type("cosine").limit(k).to_list()
        return [(r["path"], 1.0 - float(r["_distance"])) for r in rows]

    def search_batch(
        self, model: str, query_vecs: np.ndarray, k: int = 12
    ) -> list[list[tuple[str, float]]]:
        return [self.search(model, q, k) for q in query_vecs]
