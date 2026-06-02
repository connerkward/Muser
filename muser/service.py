"""Muser embedded service — one warm process that owns the model + index.

`muser serve` loads the embedding model once and keeps it resident, owns the
LanceDB index, and serves both a JSON API and a browser search UI. CLI/MCP become
thin HTTP clients of this. Local-only by default (binds 127.0.0.1).
"""

from __future__ import annotations

import io
import json
import os
import platform
import subprocess
from pathlib import Path

from pydantic import BaseModel

from .index import MuserIndex
from .registry import DEFAULT_MODEL, load_model, model_names

WEB = Path(__file__).resolve().parent / "web" / "app.html"


class IndexReq(BaseModel):
    folder: str
    recursive: bool = True


class PathReq(BaseModel):
    path: str


class ModelReq(BaseModel):
    name: str


class CartReq(BaseModel):
    paths: list[str]


class State:
    def __init__(self, model: str):
        self.model_name = model
        self.embedder = None
        self.index = MuserIndex()
        # label_index is populated lazily — embeds cluster labels for
        # /api/suggest (autocomplete) and the refinement chips on /api/search.
        # Invalidated whenever the active model changes (different embedding
        # space → label vectors no longer comparable to image vectors).
        self.label_index = None

    def warm(self):
        if self.embedder is None:
            self.embedder = load_model(self.model_name)
            self.embedder.embed_queries(["warm up"])  # trigger weight load
        return self.embedder

    def set_model(self, name: str):
        self.model_name = name
        self.embedder = None
        self.label_index = None
        self.warm()


def _reveal(path: str):
    s = platform.system()
    if s == "Darwin":
        subprocess.run(["open", "-R", path], check=False)
    elif s == "Windows":
        subprocess.run(["explorer", f"/select,{path}"], check=False)
    else:
        subprocess.run(["xdg-open", os.path.dirname(path)], check=False)


def create_app(model: str = DEFAULT_MODEL):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse, Response

    state = State(model)
    app = FastAPI(title="Muser")

    @app.on_event("startup")
    def _startup():
        print(f"  loading model {state.model_name} ...", flush=True)
        state.warm()
        print(f"  ready — {state.index.count(state.model_name)} images indexed for {state.model_name}", flush=True)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return WEB.read_text()

    @app.get("/api/status")
    def status():
        return {
            "model": state.model_name,
            "models": model_names(),
            "indexed": state.index.count(state.model_name),
            "db": str(state.index.db_path),
        }

    @app.post("/api/index")
    def do_index(req: IndexReq):
        folder = os.path.expanduser(req.folder)
        if not os.path.isdir(folder):
            raise HTTPException(400, f"not a folder: {folder}")
        emb = state.warm()
        res = state.index.index_folder(folder, state.model_name, emb, recursive=req.recursive)
        return {"added": res.added, "updated": res.updated, "removed": res.removed, "total": res.total}

    # ---- cluster-label index: powers /api/suggest + the refinement chips ----
    # Embeds each cluster label (from ~/.muser/clusters.json) into the current
    # model's text space ONCE. Suggest = cosine(typed-text, label-matrix).
    # Refine = look up which clusters the top-K result paths belong to.
    def _label_index():
        # Builds a vocabulary of (text → cluster) entries. Each cluster contributes
        # BOTH its short label ("car / sports car") AND its sublabel sentence
        # ("there is a car with a seat and a window") — same display label, but
        # the sentence-form gives suggest a finer-grained matching surface. At
        # query time, results are deduplicated by display label, best score wins.
        import numpy as np
        if state.label_index is not None:
            return state.label_index
        clusters_file = Path.home() / ".muser" / "clusters.json"
        if not clusters_file.exists():
            state.label_index = {"entries": [], "vectors": None, "path_to_label": {}}
            return state.label_index
        c = json.loads(clusters_file.read_text())
        m_name = "hdbscan" if "hdbscan" in c["methods"] else next(iter(c["methods"]))
        m = c["methods"][m_name]
        entries, seen = [], set()
        path_to_label = {}
        for cl in m["clusters"]:
            if cl["id"] == -1 or cl["label"] in seen:  # skip "misc / unclustered"
                continue
            seen.add(cl["label"])
            size = cl["size"]
            entries.append({"text": cl["label"], "display": cl["label"], "size": size})
            sub = (cl.get("sublabel") or "").strip()
            if sub and sub != cl["label"]:
                entries.append({"text": sub, "display": cl["label"], "size": size})
            for p in m["members"].get(str(cl["id"]), []):
                path_to_label[p] = cl["label"]
        if not entries:
            state.label_index = {"entries": [], "vectors": None, "path_to_label": path_to_label}
            return state.label_index
        emb = state.warm()
        vecs = emb.embed_queries([e["text"] for e in entries])
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(norms > 0, norms, 1.0)
        state.label_index = {"entries": entries, "vectors": vecs, "path_to_label": path_to_label}
        return state.label_index

    def _refinements(result_paths, query):
        # Dense-vector pseudo-relevance feedback: which clusters do the top
        # results belong to? Most-common labels (minus any that effectively
        # ARE the query) become the "you might also try" chips.
        from collections import Counter
        li = _label_index()
        if not li["path_to_label"]:
            return []
        cnts = Counter()
        for p in result_paths[:20]:
            lbl = li["path_to_label"].get(p)
            if lbl:
                cnts[lbl] += 1
        q_lower = query.lower().strip()
        out = []
        for lbl, cnt in cnts.most_common():
            ll = lbl.lower()
            if q_lower and (q_lower in ll or ll in q_lower):
                continue
            out.append({"label": lbl, "count": cnt})
            if len(out) >= 5:
                break
        return out

    def _run_search(qv, k, dedup, method, folder):
        if dedup:
            return state.index.search_dedup(state.model_name, qv, k=k, method=method, folder=folder)
        return [
            {"path": p, "name": os.path.basename(p), "score": round(s, 4), "dupes": [p], "dupe_count": 1}
            for p, s in state.index.search(state.model_name, qv, k=k, folder=folder)
        ]

    @app.get("/api/search")
    def search(q: str, k: int = 24, dedup: bool = True, method: str = "embed", folder: str | None = None,
               neg: str | None = None, neg_strength: float = 0.5):
        # method: "embed" (cosine, default), "phash" (perceptual-hash Hamming,
        # robust to recompression/resize/crop), or "both" (collapse on either).
        # folder: restrict results to images under this directory (any depth).
        # neg: optional text describing concepts to push away from. CLIP/SigLIP
        # encoders ignore in-prompt negation (bag-of-words effect, see
        # Yuksekgonul et al. ICLR 2023), so suppression has to happen as
        # vector arithmetic in the embedding space, not as natural-language "not".
        emb = state.warm()
        qv = emb.embed_queries([q])[0]
        if neg and neg.strip():
            import numpy as np
            nv = emb.embed_queries([neg])[0]
            qv = qv - neg_strength * nv
            n = float(np.linalg.norm(qv))
            if n > 0:
                qv = qv / n
        results = _run_search(qv, k, dedup, method, folder)
        refinements = _refinements([r["path"] for r in results], q)
        return {"query": q, "model": state.model_name, "results": results, "refinements": refinements}

    @app.get("/api/search-image")
    def search_image(path: str, k: int = 24, dedup: bool = True, method: str = "embed", folder: str | None = None):
        # Image-to-image search: embed the file at `path`, find nearest neighbors
        # in the index. Works for any file PIL can open — doesn't need to be indexed.
        if not os.path.exists(path):
            raise HTTPException(404, f"no such image: {path}")
        emb = state.warm()
        qv = emb.embed_images([path])[0]
        results = _run_search(qv, k, dedup, method, folder)
        # Filter the reference image itself out of its own results.
        results = [r for r in results if r["path"] != path]
        refinements = _refinements([r["path"] for r in results], os.path.basename(path))
        return {"query": f"image: {os.path.basename(path)}", "ref_path": path,
                "model": state.model_name, "results": results, "refinements": refinements}

    @app.get("/api/suggest")
    def suggest(q: str, k: int = 8):
        # Autocomplete grounded in the index: typed text → nearest cluster
        # vocabulary entries (labels + sublabels) by cosine similarity.
        # Dedup by display label so each cluster is at most one suggestion.
        import numpy as np
        li = _label_index()
        q = (q or "").strip()
        if not li["entries"] or not q:
            return {"suggestions": []}
        emb = state.warm()
        qv = emb.embed_queries([q])[0]
        n = float(np.linalg.norm(qv))
        if n > 0:
            qv = qv / n
        scores = li["vectors"] @ qv
        best = {}  # display -> (score, size)
        for i in np.argsort(scores)[::-1]:
            e = li["entries"][i]
            if e["display"] in best:
                continue
            best[e["display"]] = (float(scores[i]), e["size"])
            if len(best) >= k:
                break
        return {"suggestions": [
            {"label": lbl, "score": s, "size": sz} for lbl, (s, sz) in best.items()
        ]}

    @app.get("/api/folders")
    def folders():
        return {"folders": state.index.folders(state.model_name)}

    @app.post("/api/zip")
    def cart_zip(req: CartReq):
        # Bundles the requested files into a ZIP. Filenames collide on basename;
        # we disambiguate with a numeric suffix ('_2', '_3', ...). Files are
        # stored without compression — most are already-compressed image
        # formats, so deflate would burn CPU for ~0% gain. In-memory build
        # (not streaming) because ZipFile's central directory writes back into
        # the buffer at the end, which streaming-by-truncation breaks.
        import io, zipfile
        from fastapi.responses import Response
        seen, members = {}, []
        for p in req.paths:
            if not p or not os.path.isfile(p):
                continue
            base = os.path.basename(p)
            stem, ext = os.path.splitext(base)
            n = seen.get(base, 0) + 1
            seen[base] = n
            arcname = base if n == 1 else f"{stem}_{n}{ext}"
            members.append((p, arcname))
        if not members:
            raise HTTPException(400, "no valid files in cart")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            for src, arc in members:
                zf.write(src, arc)
        return Response(
            content=buf.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": 'attachment; filename="muser-cart.zip"'},
        )

    @app.post("/api/pick-folder")
    def pick_folder(kind: str = "scope"):
        # Pops a native folder picker on the server (== this machine, since
        # muser serve is local-only). Returns {path} or {cancelled: true}.
        # `kind` selects one of two fixed prompts — no string injection into AppleScript.
        prompts = {
            "scope": "Limit search to which folder?",
            "index": "Pick a folder to index",
        }
        prompt = prompts.get(kind, "Choose a folder")
        if platform.system() != "Darwin":
            raise HTTPException(501, "native folder picker is macOS-only for now")
        try:
            out = subprocess.run(
                ["osascript", "-e", f'POSIX path of (choose folder with prompt "{prompt}")'],
                capture_output=True, text=True, timeout=300, check=True,
            ).stdout.strip().rstrip("/")
            return {"path": out}
        except subprocess.CalledProcessError:
            return {"cancelled": True}

    @app.get("/api/thumb")
    def thumb(path: str, size: int = 260):
        from .embedders import _load_rgb

        try:
            img = _load_rgb(path, max_side=size)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=80)
            return Response(buf.getvalue(), media_type="image/jpeg")
        except Exception:
            raise HTTPException(404, "thumb failed")

    @app.get("/api/image")
    def image(path: str):
        if not os.path.isfile(path):
            raise HTTPException(404, "not found")
        return FileResponse(path)

    @app.post("/api/reveal")
    def reveal(req: PathReq):
        _reveal(req.path)
        return {"ok": True}

    @app.post("/api/model")
    def set_model(req: ModelReq):
        if req.name not in model_names():
            raise HTTPException(400, f"unknown model {req.name}")
        state.set_model(req.name)
        return {"model": state.model_name, "indexed": state.index.count(state.model_name)}

    # ---- Explore: clusters (read ~/.muser/clusters.json, written by `muser cluster`) ----
    CLUSTERS = Path.home() / ".muser" / "clusters.json"

    def _clusters():
        return json.loads(CLUSTERS.read_text()) if CLUSTERS.exists() else None

    @app.get("/api/clusters")
    def clusters(method: str = "hdbscan"):
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"built": False, "clusters": []}
        m = c["methods"][method]
        return {
            "built": True, "method": method, "methods": list(c["methods"].keys()), "n": c["n"],
            "clusters": [
                {"id": cl["id"], "label": cl["label"], "sublabel": cl["sublabel"], "size": cl["size"], "reps": cl["reps"]}
                for cl in m["clusters"]
            ],
        }

    @app.get("/api/cluster")
    def cluster_members(method: str, id: int, offset: int = 0, limit: int = 80):
        c = _clusters()
        if not c or method not in c["methods"]:
            return {"total": 0, "members": []}
        members = c["methods"][method]["members"].get(str(id), [])
        page = members[offset : offset + limit]
        return {"total": len(members), "members": [{"path": p, "name": os.path.basename(p)} for p in page]}

    # ---- per-image scores: Interesting / Review (read ~/.muser/scores.json) ----
    SCORES = Path.home() / ".muser" / "scores.json"

    @app.get("/api/score")
    def score_rank(metric: str = "interesting", order: str = "desc", offset: int = 0, limit: int = 80):
        if not SCORES.exists():
            return {"built": False, "items": []}
        s = json.loads(SCORES.read_text())
        if metric not in s["metrics"]:
            return {"built": False, "items": []}
        canon = s.get("canonical") or list(s["scores"].keys())  # deduped reps (fallback for old files)
        dupes = s.get("dupes", {})
        ranked = sorted(canon, key=lambda p: s["scores"][p].get(metric, 0), reverse=(order == "desc"))
        page = ranked[offset : offset + limit]
        items = []
        for p in page:
            d = dupes.get(p, [p])
            items.append({
                "path": p, "name": os.path.basename(p), "score": s["scores"][p].get(metric, 0),
                "scores": s["scores"][p], "dupes": d, "dupe_count": len(d),
            })
        return {"built": True, "metric": metric, "metrics": s["metrics"], "total": len(ranked), "items": items}

    return app


def serve(host: str = "127.0.0.1", port: int = 7777, model: str = DEFAULT_MODEL):
    import uvicorn

    print(f"Muser serving on http://{host}:{port}  (model: {model})", flush=True)
    uvicorn.run(create_app(model), host=host, port=port, log_level="warning")
