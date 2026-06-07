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
    6. generate  — nano_banana / chatgpt / both (per-prompt, threaded) OR lora.
    7. 3d        — optional image→GLB per generated image (multi-view when multi_angle).
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
import re
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
            lk = threading.RLock()  # reentrant: _write_run re-acquires under add_output/stage
            _locks[run_id] = lk
        return lk


_RUNID_RE = re.compile(r"^[0-9a-f]{16}$")  # secrets.token_hex(8)


def run_dir(run_id: str) -> Path:
    # Validate before joining: an attacker-controlled run_id like ".." would let
    # the /api/pipeline/{run_id}/file/{path} endpoint escape the pipelines jail
    # and read ~/.muser/* (scores.json, captions.jsonl, …). Run ids are always
    # 16 hex chars; reject anything else.
    if not _RUNID_RE.match(run_id or ""):
        raise ValueError("invalid run_id")
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
    try:
        p = run_dir(run_id) / "run.json"
    except ValueError:
        return None
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
            keys: generator ("nano_banana"|"chatgpt"|"both"|"lora"), lora_type,
            caption (bool), caption_prompt, cutout, upscale_refs,
            expand_prompts (bool), brief, gen_prompts, n_outputs, make_3d,
            multi_angle (bool), items: [{path, upscale}].
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
        # (which doesn't see the in-RAM job) still shows progress. The run-lock
        # guards the shared run dict against concurrent worker mutations (else
        # json.dumps can hit "list changed size during iteration").
        with _lock_for(run_id):
            job = registry.get(run_id)
            if job is not None:
                run["stages"] = [dict(s) for s in job.get("stages", [])]
            _write_run(run_id, run)

    def stage_error(name, msg):
        registry.add_error(run_id, "", name, msg)
        stage(name, status="error")

    def add_output(o: dict):
        # Worker threads (nano/chatgpt/both pool) call this concurrently; guard
        # the shared run dict so append + json.dumps don't race.
        with _lock_for(run_id):
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
        # Monotonic counter for output filenames so concurrent generators
        # (nano_banana + chatgpt in "both" mode) never collide on a name.
        out_seq = [0]

        def _next_seq() -> int:
            with gen_done:
                out_seq[0] += 1
                return out_seq[0]

        def _save_image_bytes(data: bytes, prompt: str, generator_tag: str):
            """Write generated PNG bytes into outputs/ and record (tagged)."""
            seq = _next_seq()
            rel = f"gen_{seq:03d}.png"
            (out_dir / rel).write_bytes(data)
            add_output({
                "kind": "image",
                "file": f"outputs/{rel}",
                "prompt": prompt,
                "generator": generator_tag,
            })

        def _record_image(rel_writer_url: str, prompt: str, generator_tag: str):
            """Download a generated image URL into outputs/ and record (tagged)."""
            _save_image_bytes(gen.download(rel_writer_url), prompt, generator_tag)

        ref_paths = list(paths)  # cart image paths for the gpt-image (edits) route

        def _gen_nano(prompt: str):
            urls = gen.nano_banana(ref_urls, prompt, n=1)
            if urls:
                _record_image(urls[0], prompt, "nano_banana")
            else:
                registry.add_error(run_id, "", "generate", f"nano_banana: no image for prompt {prompt!r}")

        def _gen_chatgpt(prompt: str):
            blobs = gen.gpt_image(ref_paths, prompt, n=1)
            if blobs:
                _save_image_bytes(blobs[0], prompt, "chatgpt")
            else:
                registry.add_error(run_id, "", "generate", f"chatgpt: no image for prompt {prompt!r}")

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
                for prompt in prompts:
                    try:
                        urls = gen.flux_lora_generate(lora_url, prompt, n=1)
                        if urls:
                            _record_image(urls[0], prompt, "lora")
                    except Exception as e:
                        registry.add_error(run_id, "", "generate", f"{type(e).__name__}: {e}")
                    done_n[0] += 1
                    stage("generate", done=done_n[0])
        else:
            # nano_banana / chatgpt / both — one (or two) images per prompt.
            # In "both" mode each prompt fans out to BOTH generators concurrently,
            # producing two tagged image outputs the UI can compare side by side.
            want_nano = generator in ("nano_banana", "both")
            want_chatgpt = generator in ("chatgpt", "both")

            def _one(prompt):
                tasks = []
                if want_nano:
                    tasks.append(_gen_nano)
                if want_chatgpt:
                    tasks.append(_gen_chatgpt)
                if len(tasks) > 1:
                    # Run the two generators concurrently for this prompt.
                    with cf.ThreadPoolExecutor(max_workers=len(tasks)) as inner:
                        futs = [inner.submit(_run_gen, fn, prompt) for fn in tasks]
                        for f in cf.as_completed(futs):
                            f.result()
                else:
                    for fn in tasks:
                        _run_gen(fn, prompt)
                with gen_done:
                    done_n[0] += 1
                    registry.set_stage(run_id, "generate", done=done_n[0])

            def _run_gen(fn, prompt):
                try:
                    fn(prompt)
                except Exception as e:
                    registry.add_error(run_id, "", "generate", f"{type(e).__name__}: {e}")

            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                list(ex.map(_one, list(prompts)))

        stage("generate", status="done", done=done_n[0], total=len(prompts))

        # ---- 7. optional 3d -------------------------------------------------
        if config.get("make_3d"):
            multi_angle = bool(config.get("multi_angle"))
            brief = config.get("brief") or ""
            # Snapshot only the *generated* images (not view/cutout outputs) so the
            # multi-angle pass we add below doesn't recurse into its own view images.
            image_outputs = [o for o in run["outputs"] if o.get("kind") == "image"]
            stage("3d", status="running", done=0, total=len(image_outputs))
            for i, o in enumerate(image_outputs):
                try:
                    img_path = d / o["file"]
                    front_url = gen.fal_upload_path(str(img_path))

                    if multi_angle:
                        # Synthesize L/B/R views, save them as visible outputs, then
                        # feed whatever succeeded into the multi-view reconstruction.
                        views = {}
                        try:
                            views = gen.generate_views(front_url, prompt_hint=brief)
                        except Exception as e:
                            registry.add_error(run_id, "", "3d", f"generate_views: {type(e).__name__}: {e}")
                        for label in ("left", "back", "right"):
                            vurl = (views or {}).get(label)
                            if not vurl:
                                continue
                            try:
                                vpng = gen.download(vurl)
                                v_rel = f"view_{i:03d}_{label}.png"
                                (out_dir / v_rel).write_bytes(vpng)
                                add_output({
                                    "kind": "image",
                                    "file": f"outputs/{v_rel}",
                                    "view": label,
                                    "generator": "nano_banana",
                                    "src_image_index": i,
                                })
                            except Exception as e:
                                registry.add_error(run_id, "", "3d", f"view {label}: {type(e).__name__}: {e}")
                        res = gen.image_to_3d_multiview(
                            front_url,
                            (views or {}).get("left"),
                            (views or {}).get("back"),
                            (views or {}).get("right"),
                        )
                    else:
                        res = gen.image_to_3d(front_url)

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
