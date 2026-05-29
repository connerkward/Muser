"""Gradio web UI to run and *verify* the retrieval harness.

Two tabs:
  - Benchmark: pick models/dataset/size, run, see the metric comparison table.
  - Inspect:   query the benchmark corpus, see the top-k images returned with
               scores; ground-truth images are flagged so you can eyeball that
               high recall actually means the right image came back.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache

import gradio as gr

from muser.registry import DEFAULT_MODEL, load_model, model_names

from .datasets import LOADERS
from .harness import METRICS, run_benchmark

_INDEX_CACHE: dict = {}


def _run_benchmark(models, dataset, n, k):
    if not models:
        return None
    bench = LOADERS[dataset](n_images=int(n))
    results = run_benchmark(list(models), bench, tempfile.mkdtemp(), k=int(k))
    results.sort(key=lambda r: r.scores.get("ndcg@10", 0), reverse=True)
    header = ["model", "tier", *METRICS, "index(s)", "query(s)"]
    rows = [
        [r.model, r.tier, *[round(r.scores[m], 3) for m in METRICS], r.index_s, r.query_s]
        for r in results
    ]
    return gr.Dataframe(value=rows, headers=header, label=f"{dataset} (corpus={results[0].corpus})")


def _get_index(model, dataset, n):
    key = (model, dataset, int(n))
    if key not in _INDEX_CACHE:
        from muser.index import MuserIndex

        bench = LOADERS[dataset](n_images=int(n))
        emb = load_model(model)
        idx = MuserIndex(db_path=tempfile.mkdtemp())
        idx.add_images(model, emb, bench.image_paths, batch_size=16)
        # map: path -> set of ground-truth queries (for flagging)
        gt = {}
        for qid, rels in bench.qrels.items():
            for path in rels:
                gt.setdefault(path, set()).add(qid)
        _INDEX_CACHE[key] = (idx, emb, bench, gt)
    return _INDEX_CACHE[key]


def _inspect(query, model, dataset, n, k, gt_path):
    idx, emb, bench, gt = _get_index(model, dataset, n)
    qv = emb.embed_queries([query])[0]
    hits = idx.search(model, qv, k=int(k))
    items = []
    for rank, (path, score) in enumerate(hits, 1):
        correct = " ✅" if gt_path and path == gt_path else ""
        items.append((path, f"#{rank}  {score*100:.1f}%{correct}"))
    return items


def _random_gt_query(model, dataset, n):
    import random

    _, _, bench, _ = _get_index(model, dataset, n)
    qid, text = random.choice(bench.queries)
    gt_path = next(iter(bench.qrels[qid]))
    return text, gt_path


def launch(share: bool = False):
    names = model_names()
    datasets = list(LOADERS)

    with gr.Blocks(title="Muser — retrieval harness") as demo:
        gr.Markdown("# Muser — retrieval harness\nBenchmark image-search models and inspect results.")

        with gr.Tab("Benchmark"):
            with gr.Row():
                m_in = gr.CheckboxGroup(names, value=["clip-b32", "clip-l14"], label="Models")
                d_in = gr.Dropdown(datasets, value=datasets[0], label="Dataset")
                n_in = gr.Slider(50, 1000, value=200, step=50, label="Corpus size")
                k_in = gr.Slider(5, 50, value=10, step=5, label="k")
            run_btn = gr.Button("Run benchmark", variant="primary")
            table = gr.Dataframe(label="results")
            run_btn.click(_run_benchmark, [m_in, d_in, n_in, k_in], table)

        with gr.Tab("Inspect"):
            with gr.Row():
                im_model = gr.Dropdown(names, value=DEFAULT_MODEL if DEFAULT_MODEL in names else names[0], label="Model")
                im_ds = gr.Dropdown(datasets, value=datasets[0], label="Dataset")
                im_n = gr.Slider(50, 1000, value=200, step=50, label="Corpus size")
                im_k = gr.Slider(4, 24, value=8, step=4, label="k")
            q_in = gr.Textbox(label="Query", value="a dog running on the beach")
            gt_state = gr.State(value=None)
            with gr.Row():
                search_btn = gr.Button("Search", variant="primary")
                rand_btn = gr.Button("🎲 Random ground-truth query")
            gallery = gr.Gallery(label="Top-k (✅ = the ground-truth image)", columns=4, height=520)

            search_btn.click(_inspect, [q_in, im_model, im_ds, im_n, im_k, gt_state], gallery)
            rand_btn.click(_random_gt_query, [im_model, im_ds, im_n], [q_in, gt_state]).then(
                _inspect, [q_in, im_model, im_ds, im_n, im_k, gt_state], gallery
            )

    demo.launch(share=share, inbrowser=False)


if __name__ == "__main__":
    launch()
