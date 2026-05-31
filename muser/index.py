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

    def folders(self, model: str, min_count: int = 2, limit: int = 400) -> list[dict]:
        """Indexed directories (each ancestor that contains images, any depth) with
        image counts, for the UI's folder-scope picker. Drops the shallow shared
        roots (``/``, ``/Users``, ``/Users/<me>``) and any folder holding *every*
        image (scoping to it == no scope). Most-populated first, capped at ``limit``."""
        from collections import Counter

        t = self._open(model)
        if t is None:
            return []
        c: Counter[str] = Counter()
        for r in t.search().select(["path"]).limit(10_000_000).to_list():
            p = Path(r["path"]).parent
            while True:
                c[str(p)] += 1
                if p == p.parent:
                    break
                p = p.parent
        total = self.count(model)
        items = [
            {"folder": d, "count": n}
            for d, n in c.items()
            if n >= min_count and n < total and len(Path(d).parts) >= 4
        ]
        items.sort(key=lambda x: (-x["count"], x["folder"]))
        return items[:limit]

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
        # Only prune rows UNDER this folder — the table accumulates many folders, so a
        # path indexed from elsewhere must not be removed when re-indexing this one.
        abs_path = str(Path(folder).resolve())
        prefix = abs_path + os.sep
        to_remove = [
            p for p in existing if (p == abs_path or p.startswith(prefix)) and p not in on_disk_set
        ]

        # Drop stale/replaced rows before re-adding.
        stale = to_remove + [p for p in to_embed if p in existing]
        if t is not None and stale:
            t.delete(" OR ".join("path = '" + p.replace("'", "''") + "'" for p in stale))

        added_paths = [p for p in to_embed if p not in existing]
        # Commit in chunks so a long index is durable, searchable mid-run, and resumable
        # by mtime if interrupted (a single add-at-the-end would lose everything on crash).
        CHUNK = 512
        done = 0
        for i in range(0, len(to_embed), CHUNK):
            chunk = to_embed[i : i + CHUNK]
            self.add_images(
                model, embedder, chunk, batch_size=batch_size,
                on_progress=(lambda d, _t: on_progress(done + d, len(to_embed))) if on_progress else None,
            )
            done += len(chunk)

        return IndexResult(
            added=len(added_paths),
            updated=len(to_embed) - len(added_paths),
            removed=len(to_remove),
            total=self.count(model),
        )

    @staticmethod
    def _folder_where(folder: str | None) -> str | None:
        """SQL predicate restricting ``path`` to files under ``folder`` (any depth),
        or None for no scoping. Uses a half-open string range ``[prefix, prefix⁺)``
        rather than ``LIKE 'prefix%'`` so wildcard chars common in paths (``_``)
        can't cause false matches — only single quotes need escaping. Applied as a
        ``prefilter`` so the vector ``limit`` is taken *after* the folder cut, not
        before (otherwise an out-of-folder top-k could return nothing in-folder)."""
        if not folder:
            return None
        prefix = str(Path(folder).expanduser().resolve()) + os.sep
        lo = prefix.replace("'", "''")
        hi = (prefix[:-1] + chr(ord(os.sep) + 1)).replace("'", "''")
        return f"path >= '{lo}' AND path < '{hi}'"

    def search(
        self, model: str, query_vec: np.ndarray, k: int = 12, folder: str | None = None
    ) -> list[tuple[str, float]]:
        t = self._open(model)
        if t is None:
            return []
        q = t.search(query_vec.tolist()).distance_type("cosine")
        where = self._folder_where(folder)
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(k).to_list()
        return [(r["path"], 1.0 - float(r["_distance"])) for r in rows]

    def search_batch(
        self, model: str, query_vecs: np.ndarray, k: int = 12
    ) -> list[list[tuple[str, float]]]:
        return [self.search(model, q, k) for q in query_vecs]

    def search_dedup(
        self,
        model: str,
        query_vec: np.ndarray,
        k: int = 24,
        threshold: float = 0.985,
        method: str = "embed",
        phash_hamming: int = 8,
        folder: str | None = None,
    ) -> list[dict]:
        """Search, collapsing near-identical images (same picture saved in many
        folders, or recompressed/resized/cropped copies) into one result.

        Over-fetch, then greedily group. Three grouping methods:

        - ``"embed"`` (default): cosine of the (L2-normalized) embedding >=
          ``threshold``. Original behavior — fast, no image loading.
        - ``"phash"``: perceptual-hash Hamming distance <= ``phash_hamming``.
          Loads each over-fetched image and computes a 64-bit pHash (DCT-based,
          via the ``imagehash`` library), which is robust to recompression,
          rescaling, and mild crops where the embedding can drift below 0.985.
        - ``"both"``: collapse when the embedding cosine is high OR the pHash
          Hamming is low. Most aggressive / most robust to crops+recompression.

        For phash/both we also compute a dHash (gradient hash) and collapse on
        whichever of pHash/dHash is within range — dHash catches some
        recompression artifacts pHash misses, at negligible extra cost.

        pHash is 64-bit; a Hamming distance <= ~8 (≈12% of bits) is the standard
        "same image" threshold (Krawetz / pHash.org). Lower (4–6) = stricter,
        only near-exact; higher (10–12) starts to risk false merges. We default
        to 8 as a balance: catches recompressed/resized copies without merging
        genuinely distinct images.

        Each result carries its duplicate file paths so the UI can offer a
        'show all copies' flow.
        """
        t = self._open(model)
        if t is None:
            return []
        n = max(k * 12, 200)
        q = t.search(query_vec.tolist()).distance_type("cosine")
        where = self._folder_where(folder)
        if where:
            q = q.where(where, prefilter=True)
        rows = q.limit(n).to_list()

        use_phash = method in ("phash", "both")
        use_embed = method in ("embed", "both")
        if not (use_phash or use_embed):
            raise ValueError(f"unknown dedup method: {method!r} (expected embed|phash|both)")

        kept: list[dict] = []
        for r in rows:
            vec = np.asarray(r["vector"], dtype=np.float32)
            score = 1.0 - float(r["_distance"])
            ph, dh = (self._phash_pair(r["path"]) if use_phash else (None, None))

            def is_dup(kp) -> bool:
                if use_embed and float(np.dot(vec, kp["_vec"])) >= threshold:
                    return True
                if use_phash and ph is not None:
                    if kp["_ph"] is not None and _hamming64(ph, kp["_ph"]) <= phash_hamming:
                        return True
                    if kp["_dh"] is not None and _hamming64(dh, kp["_dh"]) <= phash_hamming:
                        return True
                return False

            dup_of = next((kp for kp in kept if is_dup(kp)), None)
            if dup_of is not None:
                dup_of["dupes"].append(r["path"])
            elif len(kept) < k:
                kept.append(
                    {"path": r["path"], "score": score, "_vec": vec,
                     "_ph": ph, "_dh": dh, "dupes": [r["path"]]}
                )
        return [
            {
                "path": kp["path"],
                "name": os.path.basename(kp["path"]),
                "score": round(kp["score"], 4),
                "dupes": kp["dupes"],
                "dupe_count": len(kp["dupes"]),
            }
            for kp in kept
        ]

    @staticmethod
    def _phash_pair(path: str) -> tuple[int | None, int | None]:
        """(pHash, dHash) as 64-bit ints for an image, or (None, None) on failure.

        Hashes are computed on the fly from the result-set images only — nothing
        is written to the LanceDB table (a concurrent indexer owns it).
        """
        try:
            import imagehash

            from .embedders import _load_rgb

            img = _load_rgb(path, max_side=256)  # pHash downscales to 32x32 anyway
            ph = int(str(imagehash.phash(img)), 16)
            dh = int(str(imagehash.dhash(img)), 16)
            return ph, dh
        except Exception:
            return None, None


def _hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()
