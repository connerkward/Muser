"""Skin-tone method benchmark — illuminant robustness (label-free).

No MST ground truth exists for an arbitrary photo library, so we measure the
property that actually failed in practice: **stability of the predicted tone
under lighting changes.** For each face we synthesize colour casts (warm/cool/
green) and exposure shifts (dim/bright), run every sampling method, and measure
how far the predicted MST drifts. A lighting-robust method drifts little.

- chroma drift  → mean per-face stdev of MST across {orig, warm, cool, green}.
                  This is what illuminant normalization should shrink.
- exposure drift→ mean per-face stdev across {orig, dim, bright}. Honest control:
                  white balance can't undo exposure, so expect little improvement.
- coverage      → how many sample faces the method could read skin from at all.

Caveat: this scores *consistency*, not absolute correctness — pair it with the
visual contact sheet (saved alongside) to judge whether the tones look right.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json

from muser import skintone as S

CASTS = {
    "orig":   (1.00, 1.00, 1.00),
    "warm":   (1.18, 1.00, 0.82),
    "cool":   (0.85, 1.00, 1.22),
    "green":  (0.95, 1.12, 0.95),
    "dim":    (0.70, 0.70, 0.70),
    "bright": (1.28, 1.28, 1.28),
}
CHROMA = ["orig", "warm", "cool", "green"]
EXPOSURE = ["orig", "dim", "bright"]


def cast(img, g):
    return np.clip(img.astype(np.float32) * np.array(g, np.float32), 0, 255).astype(np.uint8)


def _mst(px):
    r = S.pixels_to_mst(px)
    return r[0] if r else None


METHODS = {
    "whole-box (v1)":   lambda box, full: _mst(S.skin_pixels_heuristic(box, central=False)),
    "central+trim (v3)": lambda box, full: _mst(S.skin_pixels_heuristic(box, central=True)),
    "central+trim+WB":   lambda box, full: _mst(_wb(S.skin_pixels_heuristic(box, central=True), full)),
    "face-parse (v4)":   lambda box, full: _mst(S.skin_pixels_parse(box)),
    "face-parse+WB":     lambda box, full: _mst(_wb(S.skin_pixels_parse(box), full)),
}


def _wb(px, full):
    return None if px is None else S.apply_gain(px, S.illuminant_gain(full))


def sample_faces(n=36):
    """Diverse face crops spread across the current MST histogram."""
    side = S.sidecar()
    side.prime()
    by_tone = {}
    for (p, _m, _s), e in side.entries().items():
        for d in (e.get("dets") or []):
            if d["src"] == "face" and os.path.exists(p):
                by_tone.setdefault(d["mst"], []).append(p)
    picks = []
    per = max(2, n // max(1, len(by_tone)))
    for t in sorted(by_tone):
        picks += by_tone[t][:per]
    return picks[:n]


def run():
    paths = sample_faces()
    fd = S._face_detector()
    # collect (full_img, face_box) per usable image (detect once on the original)
    items = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        fd.setInputSize((w, h))
        _, faces = fd.detect(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if faces is None or not len(faces):
            continue
        f = faces[0]
        x, y, fw, fh = (int(v) for v in f[:4])
        x, y = max(0, x), max(0, y)
        items.append((p, img, (x, y, fw, fh)))
    print(f"benchmarking {len(items)} faces × {len(CASTS)} casts × {len(METHODS)} methods\n")

    # preds[method][i][cast] = mst or None
    preds = {m: [] for m in METHODS}
    for p, img, (x, y, fw, fh) in items:
        casts_img = {c: cast(img, g) for c, g in CASTS.items()}
        for m, fn in METHODS.items():
            row = {}
            for c, ci in casts_img.items():
                box = ci[y:y + fh, x:x + fw]
                row[c] = fn(box, ci)
            preds[m].append(row)

    def drift(method, cast_keys):
        vals = []
        for row in preds[method]:
            xs = [row[c] for c in cast_keys if row[c] is not None]
            if len(xs) >= 2:
                vals.append(float(np.std(xs)))
        return float(np.mean(vals)) if vals else float("nan")

    def coverage(method):
        return sum(1 for row in preds[method] if row["orig"] is not None)

    print(f"{'method':<20}{'chroma drift':>14}{'exposure drift':>16}{'coverage':>11}")
    print("-" * 61)
    rows = []
    for m in METHODS:
        cd, ed, cov = drift(m, CHROMA), drift(m, EXPOSURE), coverage(m)
        rows.append((m, cd, ed, cov))
        print(f"{m:<20}{cd:>14.3f}{ed:>16.3f}{cov:>9}/{len(items)}")
    print("\nlower drift = more lighting-robust (MST steps of wobble per face)")

    # visual contact sheet: a few faces, each row a method, swatches per cast
    step = max(1, len(items) // 5)
    vis = [(idx, items[idx]) for idx in range(0, len(items), step)][:5]
    TH, SW, pad = 110, 52, 6
    cols = len(CASTS)
    label_w = 150
    cell_w = label_w + (SW + pad) * cols + pad
    cell_h = (len(METHODS)) * (28) + TH + 30
    sheet = Image.new("RGB", (cell_w, cell_h * len(vis) + pad), (245, 241, 232))
    dr = ImageDraw.Draw(sheet)
    Y = pad
    for i, (p, img, (x, y, fw, fh)) in vis:
        # thumbnails of each cast across the top
        dr.text((pad, Y + 4), os.path.basename(p)[:22], fill=(40, 40, 40))
        xo = label_w
        for c in CASTS:
            th = Image.fromarray(cast(img, CASTS[c])[y:y + fh, x:x + fw]).resize((SW, TH))
            sheet.paste(th, (xo, Y + 18))
            dr.text((xo + 2, Y + 4), c, fill=(90, 90, 90))
            xo += SW + pad
        yy = Y + 18 + TH + 4
        for m in METHODS:
            dr.text((pad, yy + 6), m, fill=(40, 40, 40))
            xo = label_w
            for c in CASTS:
                mst = preds[m][i][c]
                col = tuple(int(S.MST_HEX[mst - 1][k:k + 2], 16) for k in (1, 3, 5)) if mst else (200, 200, 200)
                dr.rectangle([xo, yy, xo + SW, yy + 24], fill=col)
                dr.text((xo + 4, yy + 6), str(mst or "-"), fill=(0, 0, 0) if (mst or 6) < 6 else (255, 255, 255))
                xo += SW + pad
            yy += 28
        Y += cell_h

    out_dir = os.path.expanduser("~/Desktop/2026-06-04-skintone-diag")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "benchmark.png")
    sheet.save(out)
    print(f"\nvisual contact sheet → {out}")
    # machine-readable summary
    summary = {"n_faces": len(items),
               "methods": [{"method": m, "chroma_drift": cd, "exposure_drift": ed, "coverage": cov}
                           for (m, cd, ed, cov) in rows]}
    print(json.dumps(summary))


if __name__ == "__main__":
    run()
