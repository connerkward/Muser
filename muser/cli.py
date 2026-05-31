"""Muser CLI — index, search, benchmark, and launch the web UI.

    muser index <folder> [--model jina-v4]
    muser search "a dog on a beach" --in <folder> [--model ...] [-k 12]
    muser models
    muser bench [--models clip-b32,clip-l14] [--dataset flickr30k] [-n 200]
    muser web
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import typer
from rich.console import Console

from .registry import DEFAULT_MODEL, load_model, model_names

app = typer.Typer(add_completion=False, help="Local-first semantic image search + eval harness.")
con = Console()

# ---------------------------------------------------------------------------
# Thin-client: index/search talk to the warm `muser serve` process over HTTP
# (auto-spawned if down) so the terminal path is instant. `--local` bypasses it.
# ---------------------------------------------------------------------------
SERVICE = "http://127.0.0.1:7777"


def _get(path: str, **params):
    url = SERVICE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    with urllib.request.urlopen(url, timeout=600) as r:
        return json.load(r)


def _post(path: str, payload: dict):
    req = urllib.request.Request(
        SERVICE + path, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=86400) as r:
        return json.load(r)


def _service_up() -> bool:
    try:
        urllib.request.urlopen(SERVICE + "/api/status", timeout=0.6)
        return True
    except Exception:
        return False


def _ensure_service():
    if _service_up():
        return
    con.print("[dim]starting muser service (warming model, ~20s first time)…[/]")
    kwargs: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    subprocess.Popen([sys.executable, "-c", "from muser.service import serve; serve()"], **kwargs)
    for _ in range(120):
        time.sleep(0.5)
        if _service_up():
            return
    con.print("[red]service failed to start[/]")
    raise typer.Exit(1)


@app.command()
def models():
    """List registered embedding models."""
    from .registry import model_tier

    for n in model_names():
        con.print(f"  {n:14} [dim]{model_tier(n)}[/]" + ("  [green](default)[/]" if n == DEFAULT_MODEL else ""))


@app.command()
def index(
    folder: str = typer.Argument(..., help="Folder of images to index"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model (only with --local)"),
    recursive: bool = typer.Option(True, help="Descend into subfolders"),
    local: bool = typer.Option(False, "--local", help="Index in-process instead of via the service"),
):
    """Index (or incrementally re-index) a folder of images."""
    folder = str(Path(folder).expanduser())
    if not local:
        _ensure_service()
        con.print(f"[dim]indexing {folder} via service…[/]")
        r = _post("/api/index", {"folder": folder, "recursive": recursive})
        con.print(f"  +{r['added']} added · {r['updated']} updated · {r['removed']} removed · {r['total']} total")
        return

    from .index import MuserIndex

    emb = load_model(model)
    idx = MuserIndex()
    t0 = time.time()
    res = idx.index_folder(
        folder, model, emb, recursive=recursive,
        on_progress=lambda d, t: con.print(f"  embedding {d}/{t}", end="\r"),
    )
    con.print(
        f"\nIndexed {folder} [{model}]\n"
        f"  +{res.added} added · {res.updated} updated · {res.removed} removed · "
        f"{res.total} total  ({time.time()-t0:.1f}s)"
    )


@app.command()
def search(
    query: list[str] = typer.Argument(..., help='Text query, e.g. "a dog on a beach"'),
    k: int = typer.Option(12, "-k", "--limit", help="Number of results"),
    in_: str = typer.Option(None, "--in", help="Limit to images under this folder (any depth)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model (only with --local)"),
    local: bool = typer.Option(False, "--local", help="Search in-process instead of via the service"),
):
    """Search the index by natural-language description (optionally scoped to --in <folder>)."""
    q = " ".join(query)
    folder = str(Path(in_).expanduser()) if in_ else None
    if not local:
        _ensure_service()
        params = {"q": q, "k": k}
        if folder:
            params["folder"] = folder
        r = _get("/api/search", **params)
        results, used = r["results"], r["model"]
    else:
        from .index import MuserIndex

        emb = load_model(model)
        qv = emb.embed_queries([q])[0]
        results = [{"path": p, "name": os.path.basename(p), "score": s}
                   for p, s in MuserIndex().search(model, qv, k=k, folder=folder)]
        used = model

    if not results:
        if folder:
            con.print(f"No results under {folder}. Drop --in to search the whole index.")
        else:
            con.print("No results. Index a folder first:  muser index <folder>")
        raise typer.Exit(1)
    scope = f" [dim]in {folder}[/]" if folder else ""
    con.print(f'Top {len(results)} for "{q}" [dim]\\[{used}][/]{scope}:')
    for i, h in enumerate(results, 1):
        parent = "file://" + urllib.parse.quote(os.path.dirname(h["path"]))
        con.print(f"  {i:2}. {h['score']*100:5.1f}%  [link={parent}]{h['name']}[/link]  [dim]{h['path']}[/dim]")


@app.command()
def bench(
    models_: str = typer.Option("clip-b32,clip-l14", "--models", help="Comma-separated model names"),
    dataset: str = typer.Option("flickr30k", help="Benchmark dataset"),
    n: int = typer.Option(200, "-n", help="Corpus size (images)"),
    k: int = typer.Option(10, "-k", help="Retrieval depth"),
):
    """Run the retrieval benchmark and print a model-comparison table."""
    import tempfile

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.datasets import LOADERS
    from eval.harness import format_table, run_benchmark

    model_list = [m.strip() for m in models_.split(",") if m.strip()]
    con.print(f"Loading {dataset} (n={n})...")
    bench_data = LOADERS[dataset](n_images=n)
    con.print(f"  corpus={len(bench_data.image_paths)} queries={len(bench_data.queries)}")
    results = run_benchmark(model_list, bench_data, tempfile.mkdtemp(), k=k)
    con.print(format_table(results, bench_data.name))


@app.command()
def cluster(
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to cluster"),
    k: int = typer.Option(40, "-k", help="K-means cluster count"),
):
    """Cluster the indexed embeddings into labeled groups (writes ~/.muser/clusters.json)."""
    from .cluster import cluster_all

    out = cluster_all(model, kmeans_k=k, on_progress=lambda m: con.print(f"[dim]{m}[/]"))
    for name, mth in out["methods"].items():
        con.print(f"\n[bold]{name}[/] — {len(mth['clusters'])} clusters:")
        for c in mth["clusters"][:25]:
            con.print(f"  {c['size']:6}  {c['label']:28} [dim]{c['sublabel'][:50]}[/]")


@app.command()
def score(model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to score")):
    """Score every image (interesting / novelty / aesthetic / nsfw / private / political)."""
    from .score import score_all

    out = score_all(model, on_progress=lambda m: con.print(f"[dim]{m}[/]"))
    con.print(f"scored {out['n']} images · metrics: {', '.join(out['metrics'])}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(7777, help="Port"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model to serve"),
):
    """Run the embedded service: warm model + index + web search UI."""
    from .service import serve as _serve

    _serve(host=host, port=port, model=model)


@app.command()
def web():
    """Launch the benchmark/inspection web UI (Gradio)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval.web import launch

    launch()


if __name__ == "__main__":
    app()
