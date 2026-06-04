"""Color + skin-tone facet indexes: sidecar mechanics, palette extraction, ranking.

No real faces needed — color works on synthetic swatches, and the skin-tone path
is exercised for availability + graceful empties (a faceless image → no tone).
"""
import numpy as np
import pytest
from PIL import Image

from muser import color, skintone
from muser.facets import Sidecar


def _solid(path, rgb, size=(64, 64)):
    Image.new("RGB", size, rgb).save(path)


def test_sidecar_roundtrip_and_incremental(tmp_path, monkeypatch):
    sc = Sidecar("unit_test_facet")
    monkeypatch.setattr(sc, "path", tmp_path / "facet.json")
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    _solid(a, (255, 0, 0)); _solid(b, (0, 255, 0))

    calls = []
    def compute(p):
        calls.append(p)
        return {"tag": "x"}

    sc.scan([str(a), str(b)], compute)
    assert len(calls) == 2                     # both computed first time
    assert sc.exists()
    assert sc.lookup(str(a)) is not None

    calls.clear()
    sc.scan([str(a), str(b)], compute)         # unchanged → skipped
    assert calls == []

    b.write_bytes(b.read_bytes())              # touch → mtime changes
    _solid(b, (0, 0, 255))
    calls.clear()
    sc.scan([str(a), str(b)], compute)
    assert calls == [str(b)]                   # only the changed file recomputed


def test_color_palette_and_search(tmp_path):
    assert color.available()
    red = tmp_path / "red.png"
    blue = tmp_path / "blue.png"
    _solid(red, (220, 20, 20)); _solid(blue, (20, 20, 220))

    pal = color._palette(str(red))["palette"]
    assert pal and pal[0][3] > 0.9            # one dominant color, ~whole image

    sc = color.sidecar()
    sc.path = tmp_path / "color.json"
    sc._primed = False
    color.scan([str(red), str(blue)])
    hits = color.search((230, 30, 30), k=2)
    assert hits[0][0] == str(red)             # red query → red image ranks first
    # blue is so far from red it scores 0 and is dropped, or ranks strictly below.
    ranks = dict(hits)
    assert ranks.get(str(blue), 0.0) < ranks[str(red)]


def test_skintone_available_and_empty(tmp_path):
    # opencv + bundled YuNet model ship as core deps, so detection is available
    # on every platform. A flat swatch has no face → no tone, no crash.
    assert skintone.available()
    flat = tmp_path / "flat.png"
    _solid(flat, (128, 128, 128), size=(128, 128))
    res = skintone._detect_and_tone(str(flat))
    assert res["dets"] == [] and res["mst"] is None
    assert len(skintone.MST_HEX) == 10
