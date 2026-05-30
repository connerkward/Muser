"""Muser CLI — index, search, benchmark, and launch the web UI.

    muser index <folder> [--model jina-v4]
    muser search "a dog on a beach" --in <folder> [--model ...] [-k 12]
    muser models
    muser bench [--models clip-b32,clip-l14] [--dataset flickr30k] [-n 200]
    muser web
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer
from rich.console import Console

from .registry import DEFAULT_MODEL, load_model, model_names

app = typer.Typer(add_completion=False, help="Local-first semantic image search + eval harness.")
con = Console()


@app.command()
def models():
    """List registered embedding models."""
    from .registry import model_tier

    for n in model_names():
        con.print(f"  {n:14} [dim]{model_tier(n)}[/]" + ("  [green](default)[/]" if n == DEFAULT_MODEL else ""))


@app.command()
def index(
    folder: str = typer.Argument(..., help="Folder of images to index"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model"),
    recursive: bool = typer.Option(True, help="Descend into subfolders"),
):
    """Index (or incrementally re-index) a folder of images."""
    from .index import MuserIndex

    emb = load_model(model)
    idx = MuserIndex()
    t0 = time.time()

    def prog(done, total):
        con.print(f"  embedding {done}/{total}", end="\r")

    res = idx.index_folder(folder, model, emb, recursive=recursive, on_progress=prog)
    con.print(
        f"\nIndexed {Path(folder).resolve()} [{model}]\n"
        f"  +{res.added} added · {res.updated} updated · {res.removed} removed · "
        f"{res.total} total  ({time.time()-t0:.1f}s)"
    )


@app.command()
def search(
    query: list[str] = typer.Argument(..., help='Text query, e.g. "a dog on a beach"'),
    in_: str = typer.Option(..., "--in", help="Indexed folder (sets the search scope is global; folder informs nothing yet)"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model"),
    k: int = typer.Option(12, "-k", "--limit", help="Number of results"),
):
    """Search the index by natural-language description."""
    from .index import MuserIndex

    q = " ".join(query)
    emb = load_model(model)
    idx = MuserIndex()
    qv = emb.embed_queries([q])[0]
    hits = idx.search(model, qv, k=k)
    if not hits:
        con.print(f"No results. Index a folder first: muser index <folder> --model {model}")
        raise typer.Exit(1)
    con.print(f'Top {len(hits)} for "{q}" [{model}]:')
    for i, (p, s) in enumerate(hits, 1):
        con.print(f"  {i:2}. {s*100:5.1f}%  {p}")


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
