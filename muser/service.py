"""Muser embedded service — one warm process that owns the model + index.

`muser serve` loads the embedding model once and keeps it resident, owns the
LanceDB index, and serves both a JSON API and a browser search UI. CLI/MCP become
thin HTTP clients of this. Local-only by default (binds 127.0.0.1).
"""

from __future__ import annotations

import io
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


class State:
    def __init__(self, model: str):
        self.model_name = model
        self.embedder = None
        self.index = MuserIndex()

    def warm(self):
        if self.embedder is None:
            self.embedder = load_model(self.model_name)
            self.embedder.embed_queries(["warm up"])  # trigger weight load
        return self.embedder

    def set_model(self, name: str):
        self.model_name = name
        self.embedder = None
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

    @app.get("/api/search")
    def search(q: str, k: int = 24, dedup: bool = True, method: str = "embed"):
        # method: "embed" (cosine, default), "phash" (perceptual-hash Hamming,
        # robust to recompression/resize/crop), or "both" (collapse on either).
        emb = state.warm()
        qv = emb.embed_queries([q])[0]
        if dedup:
            results = state.index.search_dedup(state.model_name, qv, k=k, method=method)
        else:
            results = [
                {"path": p, "name": os.path.basename(p), "score": round(s, 4), "dupes": [p], "dupe_count": 1}
                for p, s in state.index.search(state.model_name, qv, k=k)
            ]
        return {"query": q, "model": state.model_name, "results": results}

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

    return app


def serve(host: str = "127.0.0.1", port: int = 7777, model: str = DEFAULT_MODEL):
    import uvicorn

    print(f"Muser serving on http://{host}:{port}  (model: {model})", flush=True)
    uvicorn.run(create_app(model), host=host, port=port, log_level="warning")
