"""LanceDB-backed vector index — the real retrieval engine.

One embedded LanceDB at ~/.muser/db. One table per model (vector dims differ),
named ``img__<model>``. Rows: {path, mtimeMs, size, vector}. Search is cosine
over L2-normalized vectors. The eval harness uses this same engine so benchmarks
measure the real path, not a separate in-memory shortcut.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from .embedders import Embedder

DEFAULT_DB = Path.home() / ".muser" / "db"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".tiff"}

# Metadata columns added to LanceDB tables in 2026-06 for the Filter panel
# (resolution / aspect / file-size search). Nullable on legacy rows; populated
# for new rows at add-time and backfilled in place via `muser reindex-metadata`.
# Stored on the table itself (not a sidecar JSON) so filters compose with the
# kNN search as a single LanceDB prefilter — no client-side join, no extra IO.
METADATA_COLS = ("width", "height", "filesize")


def uid_for(path: str) -> str:
    """Stable, short, comparable-across-runs id for a path. 12-char hex (48 bits
    of blake2b — collision prob negligible at 27k images). Pure function: same
    path string in, same uid out, forever. Moving the file changes the path,
    which changes the uid — that's the right behavior for an indexer keyed by
    canonical absolute path.

    No DB lookup needed; this is hash-of-the-path. Computed on the fly at read
    time everywhere (cheap — ~µs per call), so we don't need to migrate the
    LanceDB schema or force a re-embed of the existing 27k rows.
    """
    return hashlib.blake2b(path.encode("utf-8"), digest_size=6).hexdigest()


# Alias kept private for callers that want the underscored convention.
_uid_for = uid_for


def _read_dims(path: str) -> tuple[int | None, int | None, int | None]:
    """(width, height, filesize) for an image, or Nones on a failed open. PIL
    reads only the image header for `.size` (no full decode), so this is ~0.2 ms
    per JPEG/PNG — ~10 s wall-clock for a 27 k-image corpus. Trusted local files,
    so the decompression-bomb guard is disabled."""
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as im:
            w, h = im.size
        return int(w), int(h), int(os.path.getsize(path))
    except Exception:
        return None, None, None


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


@dataclass
class MetaFilter:
    """Bounds on image metadata for the search / filter endpoints. All bounds
    inclusive; None = unbounded. ``short_side`` / ``long_side`` are
    ``LEAST(w, h)`` / ``GREATEST(w, h)`` — handy for "low-res = short side
    under 768" without caring about orientation. ``aspect`` = width / height.
    """
    min_width: int | None = None
    max_width: int | None = None
    min_height: int | None = None
    max_height: int | None = None
    min_short_side: int | None = None
    max_short_side: int | None = None
    min_long_side: int | None = None
    max_long_side: int | None = None
    min_aspect: float | None = None
    max_aspect: float | None = None
    min_filesize: int | None = None
    max_filesize: int | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)


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

    def paths(self, model: str, under: str | None = None) -> list[str]:
        """All indexed image paths for a model, optionally restricted to a folder subtree."""
        t = self._open(model)
        if t is None:
            return []
        out = [r["path"] for r in t.search().select(["path"]).limit(100_000_000).to_list()]
        if under:
            pre = os.path.join(under, "")
            out = [p for p in out if p == under or p.startswith(pre)]
        return out

    def folders(self, model: str, min_count: int = 2, limit: int = 400) -> list[dict]:
        """Indexed directories (each ancestor that contains images, any depth) with
        image counts, for the UI's folder-scope picker. Drops any folder holding *every*
        image (scoping to it == no scope) — which already excludes the all-encompassing
        shared roots, OS-agnostically. Most-populated first, capped at ``limit``."""
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
            if n >= min_count and n < total
        ]
        items.sort(key=lambda x: (-x["count"], x["folder"]))
        return items[:limit]

    def delete_paths(self, model: str, paths: Sequence[str]) -> int:
        """Hard-delete the given rows from the model's table by exact path match.

        Used to purge dead files (no longer on disk) discovered at query time —
        the index is append-by-mtime, so deleted files otherwise linger until a
        full re-index and keep resurfacing as filtered-out hits. Best-effort:
        a no-op on an empty list or a missing table, and quotes are SQL-escaped
        for the ``path IN (...)`` predicate. Returns the number of paths
        requested (LanceDB's ``delete`` doesn't report a deleted-row count)."""
        if not paths:
            return 0
        t = self._open(model)
        if t is None:
            return 0
        quoted = ",".join("'" + p.replace("'", "''") + "'" for p in paths)
        t.delete(f"path IN ({quoted})")
        return len(paths)

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
            # Image dims are read here once per add so fresh index runs carry
            # width/height/filesize from day one — no separate backfill pass for
            # newly-added images. PIL header read fails silently → Nones; the
            # row still indexes, only metadata-constrained queries skip it.
            w, h, fz = _read_dims(p)
            rows.append({
                "path": p,
                "mtimeMs": st.st_mtime * 1000.0,
                "size": st.st_size,
                "vector": v.tolist(),
                "width": w,
                "height": h,
                "filesize": fz if fz is not None else st.st_size,
            })
        name = self._table(model)
        t = self._open(model)
        if t is None:
            self.db.create_table(name, rows)
        else:
            self._ensure_metadata_cols(t)
            t.add(rows)
        return len(rows)

    @staticmethod
    def _ensure_metadata_cols(t) -> bool:
        """Add the (width, height, filesize) columns to an existing table if
        they're missing. Idempotent — called lazily on first add / backfill /
        filtered search so older tables transparently grow the new columns
        without a manual migration step. Returns True iff columns were added."""
        import pyarrow as pa

        have = {f.name for f in t.schema}
        missing = [c for c in METADATA_COLS if c not in have]
        if not missing:
            return False
        type_map = {"width": pa.int32(), "height": pa.int32(), "filesize": pa.int64()}
        t.add_columns([pa.field(c, type_map[c]) for c in missing])
        return True

    def backfill_metadata(
        self,
        model: str,
        on_progress: Callable[[int, int], None] | None = None,
        batch_size: int = 2000,
    ) -> dict:
        """Compute width/height/filesize for every indexed row that's missing
        them and merge_insert the values back in. No re-embedding — vectors
        stay untouched. ~10 s wall-clock on a 27 k-image corpus (≈0.2 ms PIL
        header read per image + a fast LanceDB upsert per batch).

        Returns ``{updated, missing, total}``. ``missing`` counts files PIL
        couldn't read a header from (corrupt / unknown format) — they still
        receive a `filesize` value when ``os.stat`` works, so the file-size
        filter doesn't silently lose them.
        """
        import pyarrow as pa

        t = self._open(model)
        if t is None:
            return {"updated": 0, "missing": 0, "total": 0}
        self._ensure_metadata_cols(t)

        # Select only the candidate rows — those with no width recorded yet.
        # An explicit NULL filter avoids re-reading dims for already-backfilled
        # files, so a partial-run restart costs roughly zero on the done rows.
        rows = (
            t.search().select(["path"])
            .where("width IS NULL", prefilter=True)
            .limit(10_000_000).to_list()
        )
        total = len(rows)
        if total == 0:
            return {"updated": 0, "missing": 0, "total": 0}

        updated = 0
        missing = 0
        buf: list[dict] = []

        def _flush():
            nonlocal updated
            if not buf:
                return
            src = pa.Table.from_pylist(buf)
            t.merge_insert("path").when_matched_update_all().execute(src)
            updated += len(buf)
            buf.clear()

        for i, r in enumerate(rows, 1):
            p = r["path"]
            w, h, fz = _read_dims(p)
            if w is None:
                missing += 1
                # Still write filesize when stat() works, so a header-only PIL
                # failure doesn't poison the file-size filter for this row too.
                try:
                    fz = os.path.getsize(p)
                except OSError:
                    fz = None
                if fz is None:
                    continue
                buf.append({"path": p, "width": None, "height": None, "filesize": int(fz)})
            else:
                buf.append({"path": p, "width": w, "height": h, "filesize": fz})
            if len(buf) >= batch_size:
                _flush()
            if on_progress and i % 200 == 0:
                on_progress(i, total)
        _flush()
        if on_progress:
            on_progress(total, total)
        return {"updated": updated, "missing": missing, "total": total}

    def metadata_coverage(self, model: str) -> dict:
        """{scored, total} for the metadata columns. ``scored`` is the count
        of rows whose ``width`` is populated — proxies "ready for the Filter
        panel". Cheap (single COUNT-WHERE), called by /api/status so the UI
        coverage line refreshes on every poll without a re-scan."""
        t = self._open(model)
        if t is None:
            return {"scored": 0, "total": 0}
        if "width" not in {f.name for f in t.schema}:
            return {"scored": 0, "total": t.count_rows()}
        total = t.count_rows()
        # count_rows with a filter isn't surfaced through the high-level API on
        # 0.33; the cheap workaround is `select(["path"]).where(...).limit(big)`
        # which only ships path strings, not vectors.
        scored = len(
            t.search().select(["path"])
            .where("width IS NOT NULL", prefilter=True)
            .limit(10_000_000).to_list()
        )
        return {"scored": scored, "total": total}

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
    def _metadata_where(f: "MetaFilter | None") -> str | None:
        """SQL predicate from a MetaFilter, or None if no constraints are set.
        Width/height comparisons implicitly drop NULL-metadata rows (NULL < x
        is NULL → false in SQL), which is the desired filter semantics."""
        if f is None:
            return None
        cs: list[str] = []
        if f.min_width is not None:        cs.append(f"width >= {int(f.min_width)}")
        if f.max_width is not None:        cs.append(f"width <= {int(f.max_width)}")
        if f.min_height is not None:       cs.append(f"height >= {int(f.min_height)}")
        if f.max_height is not None:       cs.append(f"height <= {int(f.max_height)}")
        if f.min_short_side is not None:   cs.append(f"LEAST(width, height) >= {int(f.min_short_side)}")
        if f.max_short_side is not None:   cs.append(f"LEAST(width, height) <= {int(f.max_short_side)}")
        if f.min_long_side is not None:    cs.append(f"GREATEST(width, height) >= {int(f.min_long_side)}")
        if f.max_long_side is not None:    cs.append(f"GREATEST(width, height) <= {int(f.max_long_side)}")
        if f.min_aspect is not None:       cs.append(f"(CAST(width AS DOUBLE) / CAST(height AS DOUBLE)) >= {float(f.min_aspect)}")
        if f.max_aspect is not None:       cs.append(f"(CAST(width AS DOUBLE) / CAST(height AS DOUBLE)) <= {float(f.max_aspect)}")
        if f.min_filesize is not None:     cs.append(f"filesize >= {int(f.min_filesize)}")
        if f.max_filesize is not None:     cs.append(f"filesize <= {int(f.max_filesize)}")
        return " AND ".join(cs) if cs else None

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
        self, model: str, query_vec: np.ndarray, k: int = 12,
        folder: str | None = None, meta: MetaFilter | None = None,
    ) -> list[tuple[str, float]]:
        t = self._open(model)
        if t is None:
            return []
        # `select(["path"])` keeps the result projection to just the path column
        # plus the implicit _distance — avoids materializing the 1024-dim vector
        # for every row in the top-k, which at k=288 cuts the LanceDB scan from
        # ~150 ms to ~10 ms with no semantic change. The dedup walk that used to
        # need the vector lives on `search_dedup`'s own path which still fetches
        # vectors explicitly.
        q = t.search(query_vec.tolist()).distance_type("cosine").select(["path"])
        # Compose folder + metadata constraints into one AND-chain; both run as
        # a single prefilter so the kNN limit applies AFTER the cut (otherwise
        # a top-k of out-of-filter neighbors could return zero matches).
        clauses: list[str] = []
        if (fw := self._folder_where(folder)):
            clauses.append(fw)
        if (mw := self._metadata_where(meta)):
            clauses.append(mw)
        if clauses:
            q = q.where(" AND ".join(clauses), prefilter=True)
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
        meta: MetaFilter | None = None,
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
        clauses: list[str] = []
        if (fw := self._folder_where(folder)):
            clauses.append(fw)
        if (mw := self._metadata_where(meta)):
            clauses.append(mw)
        if clauses:
            q = q.where(" AND ".join(clauses), prefilter=True)
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

    def filter(
        self,
        model: str,
        *,
        meta: MetaFilter | None = None,
        folder: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """List rows matching the given metadata constraints, no vector query.
        Returns ``(page, total_matched)``. All bounds inclusive; None unbounded.
        ``short_side`` / ``long_side`` are derived via SQL ``LEAST`` / ``GREATEST``
        so the filter still runs as a single LanceDB where-clause (no Python
        post-filter), and folder + metadata predicates compose with AND.

        Filters that need a non-null width/height implicitly drop rows with
        missing metadata — that's the intended semantic, and matches what the
        Filter panel surfaces (it explicitly opts into "rows we know about").
        """
        t = self._open(model)
        if t is None:
            return [], 0
        self._ensure_metadata_cols(t)

        clauses: list[str] = []
        if (mw := self._metadata_where(meta)):
            clauses.append(mw)
        if (fw := self._folder_where(folder)):
            clauses.append(fw)
        where = " AND ".join(clauses) if clauses else None

        # No vector query → use the plain row-filter path. Fetch only the
        # metadata columns (no vectors) so 27k matches doesn't ship megabytes.
        q = t.search().select(["path", "width", "height", "filesize"])
        if where:
            q = q.where(where)
        rows = q.limit(10_000_000).to_list()
        # Path-ascending order so pagination is deterministic across requests
        # — LanceDB's natural insertion order isn't.
        rows.sort(key=lambda r: r["path"])
        total = len(rows)
        return rows[offset : offset + limit], total

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
