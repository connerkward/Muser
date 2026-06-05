# Skin-tone search — archived (removed 2026-06-05)

A skin-tone facet (search the library by **Monk Skin Tone**, 1–10) was built and
then **removed** at the user's request. This doc preserves the design, the
benchmark results, and the cost numbers so it can be revived if wanted.

## How to revive

The full working implementation is at git tag **`skintone-v4-archived`**
(commit `9fbd9a4`, "Docs: skin-tone v4 (face-parsing + illuminant norm)").

```bash
git checkout skintone-v4-archived -- muser/skintone.py eval/skintone_bench.py
git show skintone-v4-archived:muser/assets/face_detection_yunet_2023mar.onnx > muser/assets/face_detection_yunet_2023mar.onnx
# (+ MobileNetSSD_deploy.{prototxt,caffemodel} the same way)
```
Then re-wire the service endpoints / CLI command / web tab / pyproject artifacts
(all visible in that commit's diff). The face-parsing model streams from HF
(`jonathandinu/face-parsing`); nothing else to download.

## What it did

Search images by the skin tone of people in them, mapped to the 10-point Monk
Skin Tone scale. Pipeline (best-accuracy variant, "face-parse + WB"):

1. **Detect** faces (OpenCV YuNet) and, for face-less people, persons
   (OpenCV MobileNet-SSD) — a *human* detector, not just a face detector, so
   turned-away / profile / distant people are covered too.
2. **Isolate skin** — on faces, a face-parsing SegFormer (`jonathandinu/face-parsing`)
   labels exact skin pixels (excludes eyes/brows/lips/hair); bodies fall back to a
   YCrCb colour mask.
3. **Normalize illuminant** — Shades-of-Gray (p=6) over the whole scene divides
   out warm/cool colour casts (chroma only; brightness preserved).
4. **Map** — 25–75th luminance trim (drop specular/shadow) → median → CIE-LAB →
   nearest Monk swatch.

## Benchmark — illuminant robustness (label-free)

No MST ground truth exists for an arbitrary photo library, so the benchmark
(`eval/skintone_bench.py` at the archived tag) measured the property that
actually failed in practice: **how much the predicted tone drifts when lighting
changes.** Each of 27 sample faces was re-lit with synthetic casts
(warm / cool / green / dim / bright) and every method's MST prediction recorded.
Metric = mean per-face standard deviation of MST across casts (lower = more
lighting-robust).

| method | chroma drift ↓ | exposure drift | coverage |
|---|---|---|---|
| whole-box (v1, original) | 0.643 | 0.852 | 27/27 |
| central+trim (v3) | 0.494 | 0.734 | 26/27 |
| central+trim + WB | 0.377 | 0.725 | 26/27 |
| face-parse (v4) | 0.352 | 1.084 | 21/27 |
| **face-parse + WB (shipped)** | **0.239** | 1.127 | 21/27 |

**Conclusion:** face-parsing **and** illuminant normalization each helped
independently; together they cut chroma drift **2.7×** (0.643 → 0.239) vs the
original whole-box median. Both levers earned their place.

**Honest limits found:**
- **Exposure** drift did *not* improve (WB corrects colour cast, not exposure — a
  genuinely under-exposed face stays dark). Parse's exposure drift was even
  higher than the heuristic's.
- Pure face-parse coverage was lower (21/27 — it whiffs on small/blurry faces),
  which is why production parse **fell back** to the heuristic.
- It remained a *positive signal, not a demographic classifier*; a person with no
  visible skin (fully clothed / back turned) yields no tone.

Visuals (in this folder):
- [`illuminant-robustness-benchmark.png`](illuminant-robustness-benchmark.png) —
  each face under 6 lighting casts × each method's predicted swatch (a steady row
  = robust).
- [`methods-compared.png`](methods-compared.png) — each real face (normal light)
  and the tone every method assigns. Note: on a non-photo the heuristic guesses a
  wild MST 10 while face-parse correctly declines.

## Coverage progression (full 27,794-image library)

| version | method | images with a tone |
|---|---|---|
| v1 | face-only | 4,950 |
| v2 | + person (face-less people) | 6,053 |
| v3 | central region + trim | 6,160 |
| v4 | face-parse + illuminant norm | **6,482** |

## Cost (the reason it was a heavy feature)

| item | cost |
|---|---|
| Search (query) | ~40 ms, RAM-only, **no model at query time** |
| Full precompute (v4) | **~31 min** for 27,794 images (face-parse serializes on GPU) |
| Per image w/ face | ~0.15 s (SegFormer parse) |
| Per image, no face | ~30 ms (YuNet + MobileNet) |

| model | size | location |
|---|---|---|
| SegFormer face-parsing | **323 MB** | HF cache (streamed) |
| MobileNet-SSD (person) | 22 MB | bundled in repo |
| YuNet (face) | 227 KB | bundled in repo |
| opencv-python-headless | 119 MB installed | pip (also used by **color search**, so it stays) |

The 323 MB face-parser (vs an originally-quoted ~50 MB) and the 31-min re-scan
on every algorithm change were the main weight. A ~50 MB parser (BiSeNet /
SegFormer-b0) was identified as a likely near-equivalent if revived at smaller cost.
