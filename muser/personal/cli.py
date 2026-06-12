"""`muser personal …` — the hidden Google-Photos triage sub-tool.

Every command first calls ``ensure_personal_root()``, which re-execs under
``MUSER_HOME=~/.muser-personal`` so the whole pipeline is isolated from the
aesthetic library. Registered into the main CLI as a typer sub-app.
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from . import DEFAULT_TAKEOUT, ensure_personal_root

app = typer.Typer(help="Hidden Google-Photos personal/reference triage (isolated root).")
con = Console()


def _progress(label: str, total: int):
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               TaskProgressColumn, TextColumn, TimeElapsedColumn,
                               TimeRemainingColumn)
    p = Progress(
        TextColumn(f"[bold]{label}"), BarColumn(), TaskProgressColumn(),
        MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
        TextColumn("• ETA"), TimeRemainingColumn(), console=con,
    )
    return p


@app.command()
def ingest(
    src: str = typer.Option(str(DEFAULT_TAKEOUT), "--src", help="Takeout 'Google Photos' folder"),
    model: str = typer.Option("siglip2-b", help="Embedding model"),
):
    """Embed the Takeout into the isolated index + build the takeout-metadata facet."""
    ensure_personal_root()
    from ..index import MuserIndex
    from ..registry import load_model
    from . import takeout

    src = str(Path(src).expanduser())
    if not Path(src).is_dir():
        con.print(f"[red]not a folder[/]: {src}")
        raise typer.Exit(1)

    paths = takeout.walk(src)
    con.print(f"[bold]{len(paths)}[/] indexable images under {src}")
    skipped = takeout.skipped_summary(src)
    if skipped:
        con.print("[dim]skipped (v1 scope — video/raw/other): "
                  + ", ".join(f"{n}×{e}" for e, n in list(skipped.items())[:8]) + "[/]")

    emb = load_model(model)
    idx = MuserIndex()
    with _progress("embed", len(paths)) as prog:
        t = prog.add_task("embed", total=len(paths))
        res = idx.index_folder(src, model, emb, recursive=True,
                               on_progress=lambda d, n: prog.update(t, completed=d, total=n))
    con.print(f"[bold]indexed[/] added={res.added} updated={res.updated} total={res.total}")

    con.print("building takeout-metadata facet …")
    with _progress("meta", len(paths)) as prog:
        t = prog.add_task("meta", total=len(paths))
        takeout.scan(paths, progress=lambda d, n: prog.update(t, completed=d, total=n))
    con.print("[bold]ingest done[/] — next: [cyan]muser personal faces[/], then [cyan]muser personal classify[/]")


@app.command()
def faces(min_cluster_size: int = typer.Option(4, help="HDBSCAN min cluster size")):
    """Detect/embed/cluster faces over the personal index (recurring-person signal)."""
    ensure_personal_root()
    from .. import faces as _faces
    from ..index import MuserIndex

    if not _faces.available():
        con.print("[red]insightface missing[/] — uv pip install -e \".[faces]\"")
        raise typer.Exit(1)
    paths = MuserIndex().paths("siglip2-b")
    if not paths:
        con.print("[red]no personal index[/] — run `muser personal ingest` first")
        raise typer.Exit(1)
    with _progress("faces", len(paths)) as prog:
        t = prog.add_task("faces", total=len(paths))
        _faces.scan(paths, progress=lambda d, n: prog.update(t, completed=d, total=n))
    con.print("clustering into people …")
    out = _faces.cluster(min_cluster_size=min_cluster_size)
    con.print(f"[bold]{len(out)}[/] people-clusters")


@app.command()
def classify(
    model: str = typer.Option("siglip2-b", help="Embedding model"),
    retrain: bool = typer.Option(False, "--retrain", help="Refit the reference-likeness model"),
):
    """Fuse the local signals → personal-ness P, bucket, and uncertainty per image."""
    ensure_personal_root()
    from . import personalness
    personalness.classify(model=model, retrain=retrain, progress=lambda m: con.print(f"  {m}"))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address (0.0.0.0 → reachable via Caddy)"),
    port: int = typer.Option(7778, help="Port (Caddy maps personal.muser.local → here)"),
    model: str = typer.Option("siglip2-b", help="Embedding model"),
):
    """Serve the personal instance (same web UI, isolated data, Triage tab gated on)."""
    ensure_personal_root()
    import uvicorn
    from ..service import create_app
    con.print(f"[bold]personal[/] muser → http://{host}:{port}  (root: ~/.muser-personal)")
    uvicorn.run(create_app(model), host=host, port=port)


@app.command()
def triage(
    umin: float = typer.Option(0.0, help="Lower uncertainty bound"),
    umax: float = typer.Option(1.0, help="Upper uncertainty bound"),
    estimate: bool = typer.Option(False, "--estimate", help="Only print count + $ estimate"),
):
    """Resolve the uncertain band with a gpt-4o-mini pass (cost shown before running)."""
    ensure_personal_root()
    from . import vlm_triage
    if estimate:
        n, cost = vlm_triage.estimate(umin, umax)
        con.print(f"[bold]{n}[/] images in uncertainty [{umin},{umax}] → est. [bold]${cost:.2f}[/]")
        return
    vlm_triage.run(umin, umax, progress=lambda m: con.print(f"  {m}"))
