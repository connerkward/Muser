"""Generate-mode checkout pipeline — the orchestrator behind ``mode="generate"``.

Where zip-mode checkout assembles a LoRA-training zip, generate-mode runs the cart
through fal.ai to *produce new images* (and optionally 3D meshes). It's a staged,
resilient pipeline driven by :func:`run_pipeline`, spawned in a daemon thread by
``POST /api/checkout`` and polled via ``/api/checkout/status`` + ``/api/pipeline``.

Stages (each optional except generate):
    1. caption   — caption cart items (so the LoRA route / prompt-expansion has text).
    2. upscale   — 4× flagged items; warm bytes → fal CDN refs.
    3. cutout    — BiRefNet background removal on the reference images.
    4. refs      — choose the reference set (post cutout/upscale), cap to 14.
    5. prompts   — explicit ``gen_prompts`` | LLM ``expand_prompts`` | replicate brief.
    6. generate  — nano_banana (per-prompt, threaded) OR lora (train then generate).
    7. 3d        — optional image→GLB per generated image.
    8. persist   — download all outputs, write ``run.json`` + update ``index.json``.

Resilience: a single failed prompt/image records a stage error but never aborts the
run. Final status is ``"done"`` if ≥1 output materialized, else ``"error"``.

Persistence layout::

    ~/.muser/pipelines/<run_id>/run.json
    ~/.muser/pipelines/<run_id>/outputs/<files...>
    ~/.muser/pipelines/index.json            # newest-first run summaries

``run.json`` is written atomically (tmp + os.replace) under a per-run lock, so a
poller never reads a half-written file.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import threading
import time
from pathlib import Path

from . import generators as gen

PIPELINES_DIR = Path.home() / ".muser" / "pipelines"
INDEX_JSON = PIPELINES_DIR / "index.json"

# One lock per run_id guards its run.json + index.json writes.
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_index_lock = threading.Lock()


def _lock_for(run_id: str) -> threading.Lock:
    with _locks_guard:
        lk = _locks.get(run_id)
        if lk is None:
            lk = threading.Lock()
            _locks[run_id] = lk
        return lk


def run_dir(run_id: str) -> Path:
    return PIPELINES_DIR / run_id


def _atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def _write_run(run_id: str, run: dict) -> None:
    with _lock_for(run_id):
        _atomic_write_json(run_dir(run_id) / "run.json", run)


def read_run(run_id: str) -> dict | None:
    p = run_dir(run_id) / "run.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _update_index(entry: dict) -> None:
    """Upsert a run summary into index.json (newest-first by ``created``)."""
    with _index_lock:
        items = []
        if INDEX_JSON.is_file():
            try:
                items = json.loads(INDEX_JSON.read_text()) or []
            except Exception:
                items = []
        items = [it for it in items if it.get("id") != entry["id"]]
        items.append(entry)
        items.sort(key=lambda it: it.get("created", 0), reverse=True)
        _atomic_write_json(INDEX_JSON, items)


def list_runs() -> list[dict]:
    if not INDEX_JSON.is_file():
        return []
    try:
        items = json.loads(INDEX_JSON.read_text()) or []
    except Exception:
        return []
    items.sort(key=lambda it: it.get("created", 0), reverse=True)
    return items


def _index_entry(run: dict) -> dict:
    """The compact summary stored in index.json for one run."""
    thumb = None
    for o in run.get("outputs", []):
        if o.get("kind") == "image":
            thumb = o.get("file")
            break
    cfg = run.get("config", {})
    return {
        "id": run["id"],
        "created": run.get("created"),
        "status": run.get("status"),
        "generator": cfg.get("generator"),
        "brief": cfg.get("brief"),
        "n_outputs": cfg.get("n_outputs"),
        "thumb": thumb,
    }


# ----- the pipeline ----------------------------------------------------------
def run_pipeline(run_id: str, config: dict, registry, services) -> dict:
    """Execute the generate pipeline for ``run_id``. Returns the final run dict.

    Args:
        run_id: the job/run id (also the job id in ``registry``).
        config: the parsed checkout request (mode=="generate") as a plain dict —
            keys: generator, lora_type, caption (bool), caption_prompt, cutout,
            upscale_refs, expand_prompts (bool), brief, gen_prompts, n_outputs,
            make_3d, items: [{path, upscale}].
        registry: the ``jobs.JobRegistry`` singleton (``set_stage``/``update``/
            ``add_error``).
        services: a mapping exposing the in-service helpers ::

            {
              "caption_paths": fn(paths, on_progress=None, force=False, prompt=None),
              "upscale_4x":    fn(path)->bytes,
              "load_captions": fn()->{path: caption},
              "build_cart_zip":fn(in_items, captions)->(bytes, stats),
              "uid_for":       fn(path)->str,
              "state":         <service State>,
            }
    """
    d = run_dir(run_id)
    out_dir = d / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    items = [
        (it.get("path"), bool(it.get("upscale")))
        for it in config.get("items", [])
        if it.get("path")
    ]
    paths = [p for p, _ in items]

    run: dict = {
        "id": run_id,
        "created": time.time(),
        "status": "running",
        "config": config,
        "refs": list(paths),
        "stages": [],
        "outputs": [],
        "error": None,
    }
    _write_run(run_id, run)
    _update_index(_index_entry(run))

    def stage(name, status=None, done=None, total=None):
        registry.set_stage(run_id, name, status=status, done=done, total=total)
        # Mirror the registry's stage list into run.json so a /api/pipeline read
        # (which doesn't see the in-RAM job) still shows progress.
        job = registry.get(run_id)
        if job is not None:
            run["stages"] = [dict(s) for s in job.get("stages", [])]
        _write_run(run_id, run)

    def stage_error(name, msg):
        registry.add_error(run_id, "", name, msg)
        stage(name, status="error")

    def add_output(o: dict):
        run["outputs"].append(o)
        _write_run(run_id, run)
        _update_index(_index_entry(run))

    caption_paths = services["caption_paths"]
    upscale_4x = services["upscale_4x"]
    load_captions = services["load_captions"]
    build_cart_zip = services["build_cart_zip"]

    try:
        # ---- 1. caption -----------------------------------------------------
        if config.get("caption") and paths:
            stage("caption", status="running", done=0, total=len(paths))

            def on_prog(*a):
                if len(a) == 2 and isinstance(a[0], int):
                    registry.set_stage(run_id, "caption", done=a[0])

            try:
                caption_paths(paths, on_progress=on_prog, prompt=config.get("caption_prompt"))
                stage("caption", status="done", done=len(paths))
            except Exception as e:
                stage_error("caption", f"{type(e).__name__}: {e}")

        # ---- 2. upscale refs + upload (→ ref urls) --------------------------
        # Build a per-path reference URL. Upscaled items use their warm 4× bytes;
        # everything else uploads the original.
        stage("refs", status="running", done=0, total=len(paths))
        ref_urls: list[str] = []
        upscale_set = {p for p, u in items if u} if config.get("upscale_refs") else set()
        for i, p in enumerate(paths):
            try:
                if p in upscale_set:
                    data = upscale_4x(p)
                    url = gen.fal_upload_bytes(data, os.path.basename(p) + ".jpg", "image/jpeg")
                else:
                    url = gen.fal_upload_path(p)
                ref_urls.append(url)
            except Exception as e:
                registry.add_error(run_id, p, "refs", f"{type(e).__name__}: {e}")
            stage("refs", done=i + 1)

        # ---- 3. cutout ------------------------------------------------------
        if config.get("cutout") and ref_urls:
            stage("cutout", status="running", done=0, total=len(ref_urls))
            cut_urls: list[str] = []
            for i, url in enumerate(ref_urls):
                try:
                    png = gen.cutout(url)
                    rel = f"cutout_{i:03d}.png"
                    (out_dir / rel).write_bytes(png)
                    new_url = gen.fal_upload_bytes(png, rel, "image/png")
                    cut_urls.append(new_url)
                    add_output({"kind": "cutout", "file": f"outputs/{rel}", "src_image_index": i})
                except Exception as e:
                    registry.add_error(run_id, "", "cutout", f"{type(e).__name__}: {e}")
                    cut_urls.append(url)  # fall back to the uncut ref
                stage("cutout", done=i + 1)
            ref_urls = cut_urls
            stage("cutout", status="done")

        # ---- 4. reference set (cap to 14 for nano_banana) -------------------
        ref_urls = ref_urls[: gen.NANO_MAX_REFS]
        stage("refs", status="done")

        # ---- 5. prompts -----------------------------------------------------
        n_outputs = max(1, int(config.get("n_outputs") or 1))
        brief = config.get("brief") or ""
        stage("prompts", status="running", done=0, total=n_outputs)
        gen_prompts = config.get("gen_prompts")
        if gen_prompts:
            prompts = list(gen_prompts)
        elif config.get("expand_prompts", True):
            try:
                prompts = gen.expand_prompts(brief, n_outputs, ref_urls)
            except Exception as e:
                registry.add_error(run_id, "", "prompts", f"{type(e).__name__}: {e}")
                prompts = [brief] * n_outputs
        else:
            prompts = [brief] * n_outputs
        stage("prompts", status="done", done=len(prompts), total=len(prompts))

        # ---- 6. generate ----------------------------------------------------
        generator = config.get("generator") or "nano_banana"
        stage("generate", status="running", done=0, total=len(prompts))
        gen_done = threading.Lock()
        done_n = [0]

        def _record_image(rel_writer_url: str, prompt: str, idx: int):
            """Download a generated image URL into outputs/ and record it."""
            data = gen.download(rel_writer_url)
            rel = f"gen_{idx:03d}.png"
            (out_dir / rel).write_bytes(data)
            add_output({"kind": "image", "file": f"outputs/{rel}", "prompt": prompt})

        if generator == "lora":
            # Expensive route: build a captioned zip, upload, train, then generate.
            try:
                captions = load_captions()
                zip_bytes, _ = build_cart_zip([(p, False) for p in paths], captions)
                zip_url = gen.fal_upload_bytes(zip_bytes, "lora_train.zip", "application/zip")
                stage("train", status="running", total=1)
                trigger = (config.get("lora_type") or "concept").replace(" ", "_")
                is_style = (config.get("lora_type") or "").lower() == "style"
                lora_url = gen.train_lora(zip_url, trigger, is_style)
                stage("train", status="done", done=1)
            except Exception as e:
                stage_error("train", f"{type(e).__name__}: {e}")
                lora_url = None

            if lora_url:
                for idx, prompt in enumerate(prompts):
                    try:
                        urls = gen.flux_lora_generate(lora_url, prompt, n=1)
                        if urls:
                            _record_image(urls[0], prompt, idx)
                    except Exception as e:
                        registry.add_error(run_id, "", "generate", f"{type(e).__name__}: {e}")
                    done_n[0] += 1
                    stage("generate", done=done_n[0])
        else:
            # nano_banana — one image per prompt, parallelized with a small pool.
            def _one(args):
                idx, prompt = args
                try:
                    urls = gen.nano_banana(ref_urls, prompt, n=1)
                    if urls:
                        _record_image(urls[0], prompt, idx)
                    else:
                        registry.add_error(run_id, "", "generate", f"no image for prompt #{idx}")
                except Exception as e:
                    registry.add_error(run_id, "", "generate", f"{type(e).__name__}: {e}")
                with gen_done:
                    done_n[0] += 1
                    registry.set_stage(run_id, "generate", done=done_n[0])

            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_one, list(enumerate(prompts))))

        stage("generate", status="done", done=done_n[0], total=len(prompts))

        # ---- 7. optional 3d -------------------------------------------------
        if config.get("make_3d"):
            image_outputs = [o for o in run["outputs"] if o.get("kind") == "image"]
            stage("3d", status="running", done=0, total=len(image_outputs))
            for i, o in enumerate(image_outputs):
                try:
                    img_path = d / o["file"]
                    img_url = gen.fal_upload_path(str(img_path))
                    res = gen.image_to_3d(img_url)
                    glb = gen.download(res["glb"])
                    glb_rel = f"model_{i:03d}.glb"
                    (out_dir / glb_rel).write_bytes(glb)
                    o3d = {"kind": "3d", "file": f"outputs/{glb_rel}", "glb_file": f"outputs/{glb_rel}"}
                    if res.get("thumbnail"):
                        try:
                            th = gen.download(res["thumbnail"])
                            th_rel = f"model_{i:03d}_thumb.png"
                            (out_dir / th_rel).write_bytes(th)
                            o3d["thumb_file"] = f"outputs/{th_rel}"
                        except Exception:
                            pass
                    add_output(o3d)
                except Exception as e:
                    registry.add_error(run_id, "", "3d", f"{type(e).__name__}: {e}")
                stage("3d", done=i + 1)
            stage("3d", status="done")

        # ---- 8. finalize ----------------------------------------------------
        has_image = any(o.get("kind") == "image" for o in run["outputs"])
        run["status"] = "done" if has_image else "error"
        if not has_image and run.get("error") is None:
            run["error"] = "no images were generated"
        _write_run(run_id, run)
        registry.update(run_id, status=run["status"], error=run.get("error"))
        _update_index(_index_entry(run))
        return run

    except Exception as e:
        run["status"] = "error"
        run["error"] = f"{type(e).__name__}: {e}"
        _write_run(run_id, run)
        registry.update(run_id, status="error", error=run["error"])
        _update_index(_index_entry(run))
        return run
