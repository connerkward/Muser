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
def cleanup(min_score: int = typer.Option(55, help="Min junk score to list as a candidate")):
    """Scan the junk facet (blur/exposure/tiny) → delete candidates for the Cleanup tab."""
    ensure_personal_root()
    from . import cleanup as _cu
    from ..index import MuserIndex

    if not _cu.available():
        con.print("[red]opencv missing[/] — uv pip install opencv-python-headless")
        raise typer.Exit(1)
    paths = MuserIndex().paths("siglip2-b")
    if not paths:
        con.print("[red]no personal index[/] — run `muser personal ingest` first")
        raise typer.Exit(1)
    with _progress("cleanup", len(paths)) as prog:
        t = prog.add_task("cleanup", total=len(paths))
        _cu.scan(paths, progress=lambda d, n: prog.update(t, completed=d, total=n))
    cands = _cu.candidates(min_score)
    con.print(f"[bold]{len(cands)}[/] delete candidates ≥ {min_score} junk score "
              f"(of {len(paths)}). Review in the Cleanup tab.")


@app.command()
def train(
    model: str = typer.Option("siglip2-b", help="Embedding model"),
    hold_out: float = typer.Option(0.2, "--hold-out", help="Fraction for held-out test"),
):
    """Train a supervised head on user labels (from /evaluate); report held-out accuracy."""
    ensure_personal_root()
    from . import personalness
    r = personalness.train(model=model, hold_out_fraction=hold_out,
                           progress=lambda m: con.print(f"  {m}"))
    if not r.get("trained"):
        con.print(f"[red]{r.get('message', 'failed')}[/]")
        raise typer.Exit(1)
    con.print(f"\n[bold]supervised training results[/]")
    con.print(f"  labeled: {r['n_labeled']}  train: {r['n_train']}  test: {r['n_test']}")
    con.print(f"  accuracy: {r['accuracy']*100:.1f}%  (binary personal: {r['binary_personal_accuracy']*100:.1f}%)")
    con.print(f"\n  confusion matrix (rows=true, cols=pred):")
    cols = list(r["confusion"]["personal"].keys())
    con.print("            " + "  ".join(f"{c[:6]:>8s}" for c in cols))
    for a in cols:
        con.print(f"  {a:11s} " + "  ".join(f"{r['confusion'][a][c]:8d}" for c in cols))
    con.print(f"\n  per-bucket precision on held-out:")
    for b in cols:
        s = r["per_bucket"][b]
        prec = s.get("precision")
        if prec:
            con.print(f"    {b:11s} {s['n']:3d} examples, {s['correct']} correct, {prec:.1%} precision")


@app.command()
def sheets(out: str = typer.Option(None, "--out", help="Output dir (default ~/Desktop/personal-triage-sheets)")):
    """Render labeled contact sheets per bucket to inspect the classification (verify-outputs)."""
    ensure_personal_root()
    from . import sheets as _sheets
    ps = _sheets.generate(out)
    if not ps:
        con.print("[red]not classified[/] — run `muser personal classify` first")
        raise typer.Exit(1)
    for p in ps:
        con.print(f"  {p}")
    con.print(f"[bold]{len(ps)} sheets[/] → open and eyeball the buckets")


@app.command()
def benchmark(
    per_bucket: int = typer.Option(70, help="Images sampled per local bucket"),
    seed: int = typer.Option(0, help="Sampling seed"),
):
    """Independent quality benchmark — gpt-4o-mini judges a random sample vs the local buckets."""
    ensure_personal_root()
    from . import benchmark as _bm
    n = per_bucket * 3
    con.print(f"judging ~{n} images with gpt-4o-mini (~${n*0.0002:.2f}) …")
    r = _bm.run(per_bucket=per_bucket, seed=seed, progress=lambda m: con.print(m))
    if "error" in r:
        con.print(f"[red]{r['error']}[/]"); raise typer.Exit(1)
    con.print(f"\n[bold]overall agreement with VLM: {r['overall_agreement']*100:.0f}%[/]  ({r['judged']} judged)")
    for b, s in r["per_bucket"].items():
        a = s["agreement"]
        con.print(f"  {b:11s} n={s['n']:3d}  agreement={a*100:.0f}%" if a is not None else f"  {b}: n=0")
    con.print("  confusion (rows=local, cols=VLM):")
    cols = list(r["confusion"]["personal"].keys())
    con.print("              " + "  ".join(f"{c[:6]:>6s}" for c in cols))
    for a in cols:
        con.print(f"    {a:11s} " + "  ".join(f"{r['confusion'][a][c]:6d}" for c in cols))
    con.print(f"  {r['n_disagreements']} disagreements → {r['_saved']}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind address (Caddy proxies to here)"),
    port: int = typer.Option(7780, help="Port (Caddy maps personal.muser.local → here; 7777 muser/7778 feedsieve/7779 local)"),
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
