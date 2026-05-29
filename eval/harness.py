"""Retrieval evaluation harness.

For each candidate model: embed the benchmark corpus into LanceDB (the real
engine), run every query, score with ranx (Recall@k, MRR, nDCG, MAP), and record
build/query latency. Output a model-comparison table. Metrics are ranx's, not
hand-rolled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from muser.index import MuserIndex
from muser.registry import load_model, model_tier

from .datasets import Benchmark

METRICS = ["hits@1", "recall@5", "recall@10", "mrr", "ndcg@10", "map"]


@dataclass
class ModelResult:
    model: str
    tier: str
    scores: dict[str, float]
    index_s: float
    query_s: float
    corpus: int
    queries: int


def evaluate_model(
    model: str,
    bench: Benchmark,
    db_path: str | Path,
    k: int = 10,
    metrics: list[str] = METRICS,
) -> ModelResult:
    from ranx import Qrels, Run, evaluate

    emb = load_model(model)
    idx = MuserIndex(db_path=db_path)

    t0 = time.time()
    # Embed corpus directly (paths already materialized by the benchmark loader).
    idx.add_images(model, emb, bench.image_paths, batch_size=16)
    index_s = time.time() - t0

    # Run all queries.
    t1 = time.time()
    qids = [q for q, _ in bench.queries]
    texts = [t for _, t in bench.queries]
    qvecs = emb.embed_queries(texts)
    run_dict: dict[str, dict[str, float]] = {}
    for qid, qv in zip(qids, qvecs):
        hits = idx.search(model, qv, k=k)
        run_dict[qid] = {path: float(score) for path, score in hits}
    query_s = time.time() - t1

    qrels = Qrels(bench.qrels)
    run = Run(run_dict)
    scores = evaluate(qrels, run, metrics)
    if isinstance(scores, float):  # single-metric guard
        scores = {metrics[0]: scores}

    return ModelResult(
        model=model,
        tier=model_tier(model),
        scores={m: float(scores[m]) for m in metrics},
        index_s=round(index_s, 2),
        query_s=round(query_s, 2),
        corpus=len(bench.image_paths),
        queries=len(bench.queries),
    )


def run_benchmark(
    models: list[str], bench: Benchmark, db_root: str | Path, k: int = 10
) -> list[ModelResult]:
    import tempfile

    results = []
    for m in models:
        db = Path(db_root) / f"db_{m.replace('/', '_')}"
        results.append(evaluate_model(m, bench, db, k=k))
    return results


def format_table(results: list[ModelResult], bench_name: str) -> str:
    from rich.console import Console
    from rich.table import Table

    table = Table(title=f"Muser retrieval benchmark — {bench_name}")
    table.add_column("model")
    table.add_column("tier")
    for m in METRICS:
        table.add_column(m, justify="right")
    table.add_column("index(s)", justify="right")
    table.add_column("query(s)", justify="right")

    best = {m: max(r.scores[m] for r in results) for m in METRICS} if results else {}
    for r in sorted(results, key=lambda r: r.scores.get("ndcg@10", 0), reverse=True):
        cells = []
        for m in METRICS:
            v = r.scores[m]
            s = f"{v:.3f}"
            cells.append(f"[bold green]{s}[/]" if v == best.get(m) else s)
        table.add_row(r.model, r.tier, *cells, str(r.index_s), str(r.query_s))

    import io

    con = Console(record=True, width=120, file=io.StringIO())
    con.print(table)
    return con.export_text()
