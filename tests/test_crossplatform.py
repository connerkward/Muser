"""Cross-platform behavior tests (macOS / Windows / Linux).

Run by .github/workflows/ci.yml on all three runners. The clipboard test needs an OS
clipboard tool + (on Linux) a display — CI installs xclip and runs under xvfb. The e2e
test downloads a small model (clip-b32, ~150 MB) and runs a real index+search on CPU into
a hermetic temp DB (never ~/.muser).
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil

import pytest
from PIL import Image

from muser import service
from muser.registry import load_model, model_names


def _png(path, color=(200, 30, 30)):
    Image.new("RGB", (64, 64), color).save(path)
    return str(path)


def test_default_model_present():
    assert "siglip2-b" in model_names()


def test_mlx_gating_matches_platform():
    # jina-v4-mlx must be registered iff MLX can actually load here (Apple Silicon + mlx).
    expect = (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx") is not None
    )
    assert ("jina-v4-mlx" in model_names()) == expect


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError):
        load_model("definitely-not-a-real-model")


def _clipboard_tool_expected() -> bool:
    s = platform.system()
    if s in ("Darwin", "Windows"):
        return True  # osascript / powershell always present
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    has_tool = bool(shutil.which("xclip") or shutil.which("wl-copy"))
    return has_display and has_tool


def test_clipboard_copy(tmp_path):
    ok = service._copy_image_to_clipboard(_png(tmp_path / "x.png"))
    if _clipboard_tool_expected():
        assert ok is True, "clipboard tool present but copy failed"
    else:
        assert ok is False  # must degrade to False, never raise


def test_reveal_never_raises(tmp_path):
    # _reveal shells out per-OS with check=False / try-except; must not raise anywhere.
    service._reveal(_png(tmp_path / "r.png"))


def test_e2e_index_and_search(tmp_path):
    from muser.index import MuserIndex

    colors = {"red": (220, 30, 30), "green": (30, 170, 50), "blue": (40, 60, 210)}
    for name, c in colors.items():
        _png(tmp_path / f"{name}.png", c)

    emb = load_model("clip-b32")  # small, fast — proves the torch/transformers path cross-OS
    idx = MuserIndex(db_path=tmp_path / "db")  # hermetic: never touches ~/.muser
    res = idx.index_folder(str(tmp_path), "clip-b32", emb, recursive=True)
    assert res.total >= 3

    qv = emb.embed_queries(["a solid red square"])[0]
    hits = idx.search("clip-b32", qv, k=3)
    assert hits, "search returned no results"
    # the red image should rank first for a 'red' query
    assert os.path.basename(hits[0][0]) == "red.png"
