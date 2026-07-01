# Variance maps — generative art from outpainting variants

*A generative-art technique found while building the **outpaintings** instance of Muser
(~5,535 album covers extended to ~6:1 AI panoramas). Each album cover has many
**variants** — the same cover, different outpainted wings. Stack a cover's variants and
reduce per-pixel across the stack and you get a "variance field": striking art for free,
straight out of the pipeline's own byproduct.*

## What it is / why it works

- Outpainting **keeps the original fixed** and **hallucinates the rest**. So across all
  variants of one cover, the kept album-art region is (near-)identical pixel-for-pixel,
  while the outpainted wings differ wildly.
- Reduce the stack per-pixel (standard deviation is the main reduction): the **kept region
  goes dark/low** (no variation), the **wings go bright/high** (lots of variation).
- Colormap that scalar field and it reads as a glowing silhouette of the original cover
  floating in a field of hallucination — one image per album, no generation, no model.

## Method

Exact pipeline (the real numbers):

1. **Group variants** by the leading filename number — `^(\d{1,3})[_-]` on the basename.
   Keep groups with **≥6 variants**.
2. **Resize every variant to a common canvas** at the **true median source aspect
   6.0952:1** → **2072×340** (`H=340`, `W=round(340·6.0952)=2072`), LANCZOS.
   - *Why not 6.0:1?* The sources aren't exactly 6:1. Snapping to a round 6.0:1 would
     **squish every frame ~1.5% horizontally**, smearing the kept region and blurring the
     boundary you're trying to reveal. Use the measured median.
3. **Per-pixel reduction across the stack** — `stack.std(axis=0)` gives per-channel std;
   `.mean(axis=-1)` collapses to a scalar `H×W` field.
4. **Normalize** — clip to the 2nd/98th percentile, then a gentle `**0.85` gamma.
5. **Apply a colormap LUT** — `matplotlib.cm.get_cmap(name)(field)` → RGB. LUT only.
6. **Save chrome-free via PIL** — `Image.fromarray(...).save(...)`. **No matplotlib
   figure/axes/ticks** — just the raw colormapped array as pixels.

Core (~10 lines):

```python
stack = np.stack([np.asarray(Image.open(p).convert("RGB")
                  .resize((2072, 340), Image.LANCZOS), np.float32) / 255
                  for p in variants])          # (N, H, W, 3)
V = stack.std(axis=0).mean(axis=-1)            # per-pixel std -> H×W field
lo, hi = np.percentile(V, 2), np.percentile(V, 98)
V = np.clip((V - lo) / (hi - lo), 0, 1) ** 0.85  # normalize + gamma
rgb = cm.get_cmap("magma")(V)[..., :3]          # LUT only — no figure
Image.fromarray((rgb * 255).astype("uint8")).save("art.png")
```

## Encodings

Same stack, different reduction — each reveals something different:

- **std** (main) — `stack.std(0).mean(-1)`. The variance field: dark kept region, bright wings.
- **range** — `max(0) − min(0)`. Extremes only; harsher, high-contrast edges of the hallucination.
- **per-channel RGB variance** — `stack.std(0)` mapped straight to R/G/B (no colormap).
  Colorful: hue shows *which* channels vary where the wings disagree.
- **mean** ("ghost average") — `stack.mean(0)` in true color. A composite of every variant:
  the kept region stays sharp, the wings blur into a soft averaged ghost.

## Favorite colormaps (canonical set)

The chosen five for this art (perceptually-uniform; dark→bright reads as low→high variance):

**`viridis` · `twilight_shifted` · `magma` · `mako` · `inferno`**

(`magma` is the default hero; earlier runs also tried `cividis` — **dropped** from the set.
`mako` comes from seaborn: `import seaborn` registers it into matplotlib.)

## Global vs per-album

- **Per-album** — variance over one cover's variants. One glowing field per album; the dark
  region is *that* cover's silhouette.
- **Global** — variance over a big random sample (~1200) drawn across **all** albums at once.
  The logic **inverts**: the fixed mask position now burns **bright** — every album's cover
  is *different*, so that region is the single **highest-variance** spot — while the
  outpainted wings, all drifting toward similar neutral gradients, cool down. Result: the
  **universal fixed cover-mask box**, revealed as a bright rectangle.

## Gotchas

- **Aspect fix is load-bearing** — resample to the measured median **6.0952:1**, not a round
  6.0:1 (which squishes ~1.5% and smears the boundary).
- **Coherent-scene sources read soft** — when a cover's outpaints form a *coherent* scene
  (e.g. desert dunes) rather than random noise, the kept region isn't a crisp black box; it
  reads as a **ghostly scene with a darker region**. The **crisp black/inverted box** shows
  best in the **global** map, where the wings average out.
- **Baked-in watermark** — some sources carry a `P: ✓` **Playground.com** watermark baked
  into the pixels; it survives the reduction and can show up faintly in the field.

## Showcase

Rendered gallery (68 PNGs) + a how-it's-made page:
[~/Desktop/cc-muser/variance-art/](file:///Users/conner/Desktop/cc-muser/variance-art/)
(`index.html`). That folder is the deliverable — the images are **not** duplicated into this repo.
