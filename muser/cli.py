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

# Hidden Google-Photos triage sub-tool (isolated ~/.muser-personal root). See muser/personal/.
from .personal.cli import app as _personal_app  # noqa: E402
app.add_typer(_personal_app, name="personal")

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

    # Auto-run the C2PA AI detector over the just-indexed files (incremental — only
    # new/changed files spawn c2patool). No-op if c2patool isn't installed.
    from .c2pa import ai_images, available, scan

    if available():
        folder_paths = idx.paths(model, under=folder)
        scan(folder_paths, progress=lambda d, t, f: con.print(f"  C2PA scan {d}/{t} — {f} AI-flagged", end="\r"))
        con.print(f"\n  C2PA: {len(ai_images())} AI-generated image(s) flagged in your library")

    # Auto-build the color palette index over the just-indexed files (incremental,
    # so only new/changed files are recomputed).
    from . import color as _color

    folder_paths = idx.paths(model, under=folder)
    _color.scan(folder_paths, progress=lambda d, t: con.print(f"  color index {d}/{t}", end="\r"))
    con.print(f"\n  Color: LAB palette built for {len(folder_paths)} image(s)")


@app.command("reindex-metadata")
def reindex_metadata(
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to backfill"),
    local: bool = typer.Option(False, "--local", help="Run in-process instead of via the service"),
):
    """Compute width / height / filesize for every indexed row missing them.

    One-time, in-place, idempotent — no re-embedding. ~10 s wall-clock for a 27k-image
    corpus on Apple Silicon. New index runs already store these fields automatically;
    this command exists so older tables (indexed before the Filter panel landed) catch
    up without a full rescan. Safe to re-run; only rows still lacking dims are touched.
    """
    if not local:
        _ensure_service()
        con.print("[dim]backfilling metadata via service…[/]")
        _post("/api/backfill-metadata", {})
        # Poll /api/status until task clears.
        last_done = -1
        while True:
            time.sleep(0.5)
            s = _get("/api/status")
            t = s.get("task")
            if t is None:
                con.print("  done.")
                break
            kind = t.get("kind", "")
            if kind.startswith("backfill_metadata"):
                done, total = t.get("done", 0), t.get("total", 0)
                if done != last_done:
                    last_done = done
                    if kind == "backfill_metadata_done":
                        con.print(f"\n  updated {t.get('updated', 0)} · missing dims {t.get('missing', 0)} · total {t.get('total', 0)}")
                    elif kind == "backfill_metadata_error":
                        con.print(f"[red]error:[/] {t.get('error')}")
                    else:
                        con.print(f"  {done}/{total}", end="\r")
        return

    from .index import MuserIndex
    idx = MuserIndex()
    t0 = time.time()
    res = idx.backfill_metadata(
        model,
        on_progress=lambda d, t: con.print(f"  {d}/{t}", end="\r"),
    )
    con.print(
        f"\n  +{res['updated']} updated · {res['missing']} missing dims · "
        f"{res['total']} scanned  ({time.time() - t0:.1f}s)"
    )


@app.command()
def search(
    query: list[str] = typer.Argument(..., help='Text query, e.g. "a dog on a beach"'),
    k: int = typer.Option(12, "-k", "--limit", help="Number of results"),
    in_: str = typer.Option(None, "--in", help="Limit to images under this folder (any depth)"),
    ai_min: int = typer.Option(0, "--ai-min", help="Keep only results with AI-likelihood %% >= this (show only AI)"),
    ai_max: int = typer.Option(100, "--ai-max", help="Drop results with AI-likelihood %% > this (remove AI)"),
    sort: str = typer.Option(None, "--sort", help='Reorder results: "ai" (most-AI first) | "ai_asc" (least-AI first)'),
    match: str = typer.Option("blend", "--match", help='Multi-concept (comma) query mode: "blend" (sum vectors) | "all" (each concept must match)'),
    min_short: int = typer.Option(None, "--min-short-side", help="Min short side in px"),
    max_long: int = typer.Option(None, "--max-long-side", help="Max long side in px"),
    model: str = typer.Option(DEFAULT_MODEL, help="Embedding model (only with --local)"),
    local: bool = typer.Option(False, "--local", help="Search in-process instead of via the service"),
):
    """Search the index by natural-language description.

    Multi-concept: separate concepts with commas — "car, snow" — to combine them as
    separate vectors; prefix one with "-" to subtract it ("car, -person"). --match blend
    (default) sums the concept vectors; --match all keeps only images where EACH concept
    is strongly present (intersection). Scope with --in <folder>; filter by AI-likelihood
    with --ai-min N; reorder with --sort ai|ai_asc; constrain resolution with
    --min-short-side / --max-long-side. All ride the same /api/search the web UI and MCP use.
    """
    q = " ".join(query)
    folder = str(Path(in_).expanduser()) if in_ else None
    if not local:
        _ensure_service()
        params = {"q": q, "k": k}
        if folder:
            params["folder"] = folder
        if ai_min:
            params["ai_min"] = ai_min
        if ai_max < 100:
            params["ai_max"] = ai_max
        if sort:
            params["sort"] = sort
        if match and match != "blend":
            params["match"] = match
        if min_short:
            params["min_short_side"] = min_short
        if max_long:
            params["max_long_side"] = max_long
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
        ai = f"  [magenta]AI {h['ai_pct']}%[/]" if h.get("ai_pct") is not None else ""
        con.print(f"  {i:2}. {h['score']*100:5.1f}%  [link={parent}]{h['name']}[/link]{ai}  [dim]{h['path']}[/dim]")


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
    """Score every image — writes ~/.muser/scores.json.

    Metrics: interesting / novelty / aesthetic (zero-shot, weak) /
    aesthetic_v2 (LAION-Aesthetics V2 MLP on CLIP-L/14) /
    pickscore (PickScore v1, CLIP-H/14 fine-tuned on human prefs) /
    nsfw (Falconsai ViT) / private (zero-shot).

    First run downloads:
      ~3.7 MB LAION-Aes V2 MLP  (camenduru/improved-aesthetic-predictor)
      ~700 MB PickScore v1      (yuvalkirstain/PickScore_v1)
      plus CLIP-L/14 weights if the active index isn't already clip-l14.
    PickScore is slow: ~50 ms/image on MPS. Cached by path|mtime|size so re-runs
    only score new/changed files.
    """
    from .score import score_all

    out = score_all(model, on_progress=lambda m: con.print(f"[dim]{m}[/]"))
    con.print(f"scored {out['n']} images · metrics: {', '.join(out['metrics'])}")


@app.command()
def albums(model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to group")):
    """Group outpaintings by album-art region + flag blurred covers.

    Crops each image to the fixed album-art box, embeds that region, clusters same-cover
    variants by cosine, and picks the highest-aesthetic rep per group — writing
    ~/.muser/album_groups.json (+ album_vecs.npz). Powers the outpaintings unique-spread
    (one rep/cover) and click-to-cluster. Incremental: only new/changed images re-embed.
    """
    from . import albums as _albums

    out = _albums.build(model, on_progress=lambda m: con.print(f"[dim]{m}[/]"))
    con.print(out["message"] if out.get("built") else f"[red]{out.get('message')}[/]")


@app.command()
def caption(
    folder: str = typer.Argument(None, help="Optional folder prefix to restrict (else: every indexed image)"),
    paths: list[str] = typer.Option(None, "--paths", help="Explicit paths to caption (bypasses the index)"),
    force: bool = typer.Option(False, "--force", help="Re-caption images already in the cache"),
    limit: int = typer.Option(None, "--limit", help="Cap N images (for testing)"),
):
    """Caption images for SDXL/Flux LoRA training — writes ~/.muser/captions.jsonl.

    Backend: **gpt-4o-mini** (OpenAI vision chat-completions). ~$0.001-0.005/image,
    ~3-5s/image; needs ``OPENAI_API_KEY`` in central/.env.

    Baked prompt: single sentence, concrete nouns, no style descriptors. JSONL
    append-only, one row per image: ``{path, caption, model, mtime, ts}``. Resume
    support — already-captioned (path, mtime) rows are skipped unless --force.
    """
    from .caption import caption_paths

    if not os.environ.get("OPENAI_API_KEY"):
        # Ensure the central .env loader has had a chance to fire.
        from .caption import _load_env_file
        _load_env_file()
    if not os.environ.get("OPENAI_API_KEY"):
        con.print(
            "[red]OPENAI_API_KEY not set[/] — add it to "
            "/Users/conner/dev/central/.env, then retry."
        )
        raise typer.Exit(1)

    if paths:
        targets = [str(Path(p).expanduser()) for p in paths]
    else:
        from .index import MuserIndex
        from .registry import DEFAULT_MODEL as EMBED_DEFAULT

        idx = MuserIndex()
        folder_path = str(Path(folder).expanduser()) if folder else None
        targets = idx.paths(EMBED_DEFAULT, under=folder_path)
        if not targets:
            where = f" under {folder_path}" if folder_path else ""
            con.print(f"[red]no indexed images[/]{where} — run `muser index <folder>` first")
            raise typer.Exit(1)

    if limit is not None:
        targets = targets[:limit]

    written = caption_paths(
        targets,
        on_progress=lambda *a: con.print(f"[dim]{a[0]}[/]") if len(a) == 1 and isinstance(a[0], str) else None,
        force=force,
    )
    con.print(f"captioned {len(written)} image(s) via gpt-4o-mini → ~/.muser/captions.jsonl")


@app.command()
def detect(
    folder: str = typer.Option(None, "--in", help="Restrict the scan to images under this folder"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to scan"),
):
    """Scan the index for AI-generated images via C2PA Content Credentials (writes ~/.muser/c2pa.json).

    Runs fully standalone — no running `muser serve` required, and no embedding model is
    loaded; it only reads indexed paths from LanceDB and shells out to c2patool. Safe to run
    while the service is up too (LanceDB allows concurrent readers). The web UI's "AI" tab
    reads the same ~/.muser/c2pa.json this writes.

    Positive-only: flags images whose signed provenance declares AI origin (OpenAI,
    Firefly, Google…). Local SD/Flux/ComfyUI output carries no credentials, so this
    is a lower bound, not every AI image.
    """
    from .c2pa import ai_images, available, scan

    if not available():
        con.print("[red]c2patool not installed[/] — `brew install c2patool` (or `cargo install c2patool`)")
        raise typer.Exit(1)
    from .index import MuserIndex

    under = str(Path(folder).expanduser()) if folder else None
    paths = MuserIndex().paths(model, under=under)
    if not paths:
        where = f" under {under}" if under else ""
        con.print(f"[red]no indexed images[/]{where} for model {model} — run `muser index <folder>` first")
        raise typer.Exit(1)

    def prog(done, total, found):
        con.print(f"  scanning {done}/{total} — {found} AI-flagged", end="\r")

    scan(paths, progress=prog)
    ai = ai_images()
    con.print(f"\n[bold]{len(ai)}[/] AI-generated image(s) found via C2PA over {len(paths)} indexed.")
    for h in ai[:25]:
        con.print(f"  [dim]{h['kind'] or 'ai':9}[/] {h['name']}  [dim]{h['tool'] or ''}[/]")


@app.command()
def aiscore(
    folder: str = typer.Option(None, "--in", help="Restrict the scan to images under this folder"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to scan"),
    query: str = typer.Option(None, "--query", "-q", help="Print the AI-likelihood % for ONE image path (no scan)"),
):
    """AI-likelihood facet — score how likely each indexed image is AI-generated, via the
    Community Forensics ViT detector (writes ~/.muser/aidet.json).

    Runs standalone (no `muser serve` needed). Unlike `detect` (C2PA metadata, positive-only),
    this is a soft pixel-level 0–100% score on EVERY image — it catches local SD/ComfyUI output
    that carries no credentials, while (unlike the old GRIP backend) NOT over-flagging digital
    art / renders / old photos. Weaker on Flux/ChatGPT/Gemini, so pair it with C2PA. Incremental
    by mtime+size, content-addressed; weights (~88 MB) download once from HF.
    """
    from . import aidet as _aidet

    if not _aidet.available():
        con.print("[red]GRIP weight missing[/] — put it at ~/.muser/models/grip_latent.pth")
        raise typer.Exit(1)
    if query:
        d = _aidet._compute(str(Path(query).expanduser()))
        if not d:
            con.print("[red]could not score[/]"); raise typer.Exit(1)
        con.print(f"  AI-likelihood [bold]{d['pct']}%[/]  [dim]logit={d['logit']}[/]")
        return
    from .index import MuserIndex

    under = str(Path(folder).expanduser()) if folder else None
    paths = MuserIndex().paths(model, under=under)
    if not paths:
        con.print(f"[red]no indexed images[/] — run `muser index <folder>` first")
        raise typer.Exit(1)
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               TaskProgressColumn, TextColumn, TimeElapsedColumn,
                               TimeRemainingColumn)
    with Progress(
        TextColumn("[bold]AI-likelihood"), BarColumn(), TaskProgressColumn(),
        MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
        TextColumn("• ETA"), TimeRemainingColumn(), console=con,
    ) as prog:
        task = prog.add_task("scoring", total=len(paths))
        _aidet.scan(paths, progress=lambda d, t: prog.update(task, completed=d, total=t))
    # quick coverage summary
    cache = _aidet.sidecar().load()
    flagged = sum(1 for e in cache.values() if (e.get("pct") or 0) >= 50)
    con.print(f"[bold]scored[/] {len(paths)} images → ~/.muser/aidet.json  "
              f"([bold]{flagged}[/] at ≥50% AI-likelihood)")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) != 6:
        raise typer.BadParameter(f"expected a #rrggbb hex color, got {h!r}")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@app.command()
def color(
    query: str = typer.Option(None, "--query", "-q", help="Search by color: a #rrggbb hex. Omit to (re)build the index."),
    folder: str = typer.Option(None, "--in", help="Restrict to images under this folder"),
    k: int = typer.Option(24, help="How many results to print when searching"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to scan"),
):
    """Dominant-color (LAB palette) index + color search — no embedding model, fully local.

    No ``--query`` → builds/refreshes ~/.muser/color.json over the indexed images.
    With ``--query "#rrggbb"`` → ranks images by how much of that color they contain.
    """
    from . import color as _color
    from .index import MuserIndex

    under = str(Path(folder).expanduser()) if folder else None
    if query:
        _color.prime_cache_from_sidecar()
        hits = _color.search(_hex_to_rgb(query), k=k, folder=under)
        con.print(f"[bold]{len(hits)}[/] image(s) matching {query}:")
        for p, sc in hits:
            con.print(f"  [dim]{sc:5.2f}[/] {Path(p).name}")
        return

    paths = MuserIndex().paths(model, under=under)
    if not paths:
        con.print(f"[red]no indexed images[/] — run `muser index <folder>` first")
        raise typer.Exit(1)
    _color.scan(paths, progress=lambda d, t: con.print(f"  scanning {d}/{t}", end="\r"))
    con.print(f"\n[bold]built color index[/] over {len(paths)} images → ~/.muser/color.json")


@app.command()
def faces(
    folder: str = typer.Option(None, "--in", help="Restrict to images under this folder"),
    model: str = typer.Option(DEFAULT_MODEL, help="Model whose index to scan"),
    min_cluster_size: int = typer.Option(4, help="HDBSCAN min cluster size (a 'person')"),
    no_cluster: bool = typer.Option(False, "--no-cluster", help="Detect/embed only, skip clustering"),
):
    """Face facet — detect + embed faces (InsightFace buffalo_l), then cluster into people.

    Writes per-image face metadata to ~/.muser/faces.json, 512-d ArcFace embeddings to
    ~/.muser/faces_emb.npz, and HDBSCAN people-clusters to ~/.muser/face_clusters.json.
    Standalone (no `muser serve` needed). Incremental by mtime+size; buffalo_l weights
    (~280 MB) download once. Install with the `[faces]` extra. The hidden `personal`
    sub-tool reuses this under MUSER_HOME=~/.muser-personal for its recurring-person signal.
    """
    from . import faces as _faces

    if not _faces.available():
        con.print("[red]insightface/onnxruntime missing[/] — install with: uv pip install -e \".[faces]\"")
        raise typer.Exit(1)
    from .index import MuserIndex

    under = str(Path(folder).expanduser()) if folder else None
    paths = MuserIndex().paths(model, under=under)
    if not paths:
        con.print("[red]no indexed images[/] — run `muser index <folder>` first")
        raise typer.Exit(1)
    from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                               TaskProgressColumn, TextColumn, TimeElapsedColumn,
                               TimeRemainingColumn)
    with Progress(
        TextColumn("[bold]faces"), BarColumn(), TaskProgressColumn(),
        MofNCompleteColumn(), TextColumn("•"), TimeElapsedColumn(),
        TextColumn("• ETA"), TimeRemainingColumn(), console=con,
    ) as prog:
        task = prog.add_task("detecting", total=len(paths))
        cache = _faces.scan(paths, progress=lambda d, t: prog.update(task, completed=d, total=t))
    nfaces = sum(e.get("n", 0) for e in cache.values())
    nimgs = sum(1 for e in cache.values() if e.get("n", 0) > 0)
    con.print(f"[bold]detected[/] {nfaces} faces across {nimgs}/{len(paths)} images")
    if not no_cluster:
        con.print("clustering into people …")
        out = _faces.cluster(min_cluster_size=min_cluster_size)
        big = sorted(out.values(), key=lambda c: -c["size"])[:5]
        con.print(f"[bold]{len(out)}[/] people-clusters → ~/.muser/face_clusters.json  "
                  f"(largest: {', '.join(str(c['size']) for c in big)})")


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


@app.command()
def uid(path: str = typer.Argument(..., help="Image path to hash")):
    """Print the stable 12-char uid for a given path.

    Uid = blake2b(path, digest_size=6).hexdigest(). Same string in, same uid
    out, forever. Useful for joining cache files (~/.muser/scores.json,
    aesthetic_v2_cache.json, captions.jsonl) keyed by path back to uids
    surfaced by the API and web UI. No DB lookup — just a hash of the string,
    so the path doesn't need to be indexed.
    """
    from .index import uid_for

    # Match the canonical-path resolution the indexer uses (Path.resolve()),
    # so `muser uid ./foo.jpg` and `muser uid /abs/.../foo.jpg` agree.
    resolved = str(Path(path).expanduser().resolve())
    con.print(uid_for(resolved))


if __name__ == "__main__":
    app()
