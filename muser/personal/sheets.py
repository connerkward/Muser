"""Labeled contact sheets per bucket — the independent verification artifact.

Per verify-outputs-rule, the classifier must not be judged by its own confidence. This
renders the actual images of each bucket (and the most-uncertain set) into labeled
montages you open and eyeball against the goal. Saved to ~/Desktop so they're easy to
scroll. Reused by `muser personal sheets`.
"""
from __future__ import annotations

import random
from pathlib import Path

from . import personalness

COLS, ROWS, CELL, PAD, LBL = 8, 6, 200, 4, 16


def _font(sz):
    from PIL import ImageFont
    for f in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(f, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def _sheet(items, title, out: Path):
    from PIL import Image, ImageDraw
    n = min(len(items), COLS * ROWS)
    W = COLS * (CELL + PAD) + PAD
    H = ROWS * (CELL + LBL + PAD) + PAD + 24
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    dr = ImageDraw.Draw(canvas)
    dr.text((PAD, 6), f"{title}  (n={len(items)}, showing {n})", fill=(230, 210, 150), font=_font(14))
    for i, (path, e) in enumerate(items[:n]):
        r, c = divmod(i, COLS)
        x = PAD + c * (CELL + PAD)
        y = 24 + PAD + r * (CELL + LBL + PAD)
        try:
            im = Image.open(path)
            im.draft("RGB", (CELL, CELL))
            im = im.convert("RGB")
            im.thumbnail((CELL, CELL))
            canvas.paste(im, (x + (CELL - im.width) // 2, y + (CELL - im.height) // 2))
        except Exception:
            dr.rectangle([x, y, x + CELL, y + CELL], fill=(60, 30, 30))
        sig = e.get("sig", {})
        lab = f"P{int(e.get('p',0)*100)} u{int(e.get('unc',0)*100)} R{int(sig.get('R',0)*100)}"
        if sig.get("people"): lab += " ppl"
        if sig.get("cam"): lab += " cam"
        if sig.get("doc"): lab += " doc"
        if e.get("vlm"): lab += " ✓vlm"
        dr.text((x, y + CELL + 1), lab, fill=(180, 180, 180), font=_font(11))
        dr.text((x, y + CELL + 13), Path(path).parent.name[:24], fill=(110, 130, 160), font=_font(9))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out


def generate(out_dir: str | Path | None = None) -> list[Path]:
    """Write bucket + uncertain + random-personal sheets; return their paths."""
    out = Path(out_dir).expanduser() if out_dir else (Path.home() / "Desktop" / "personal-triage-sheets")
    rows = list(personalness.all_entries().items())
    if not rows:
        return []
    buckets = {
        "personal":  sorted([r for r in rows if r[1].get("bucket") == "personal"], key=lambda r: -r[1].get("p", 0)),
        "reference": sorted([r for r in rows if r[1].get("bucket") == "reference"], key=lambda r: r[1].get("p", 0)),
        "in_between": sorted([r for r in rows if r[1].get("bucket") == "in_between"], key=lambda r: -r[1].get("p", 0)),
    }
    paths = [_sheet(items, f"BUCKET: {b}", out / f"bucket_{b}.png") for b, items in buckets.items()]
    paths.append(_sheet(sorted(rows, key=lambda r: -r[1].get("unc", 0)), "MOST UNCERTAIN", out / "uncertain.png"))
    rp = buckets["personal"][:]
    random.Random(1).shuffle(rp)
    paths.append(_sheet(rp, "PERSONAL (random sample)", out / "personal_random.png"))
    return paths
