# Outpaintings curation — unique spread, click-to-cluster, grey-out

The outpaintings instance (`MUSER_HOME=~/.muser-outpaintings`) is a **portfolio-curation
surface** over ~5,535 AI-outpainted album covers (square covers extended to ~6:1). The
goal: judge which aesthetic model/blend picks the good ones, and drill into any cover to
pick its best re-roll — on a **clean, one-per-cover spread**, not a wall of near-dupes.

## The album facet (`muser/albums.py`)

Album covers sit at a **fixed, offset-right box** in every outpainting —
`x∈[0.52,0.64] y∈[0.13,0.74]` (fraction of W×H). Derived empirically from cross-variant
pixel variance and confirmed by the ComfyUI outpaint pad (`ImagePadForOutpaint left=3000,
right=2000` → the cover is pushed right of center). Derivation + the verification overlays:
`docs/ephemeral-design-interfaces.md`.

Pipeline: crop each image to that box → embed the region (SigLIP) → cluster same-cover
variants by region cosine `≥ 0.92` → drop covers whose region-blur `< 5` or live in a
`/blurred/` folder → rep per group = highest-`aesthetic_v2` member. Persists:
- `~/.muser-outpaintings/album_vecs.npz` — path-aligned region embeddings + region-blur (incremental)
- `~/.muser-outpaintings/album_groups.json` — `{by_path:{group,blur,removed}, groups:[{id,rep,count,aesthetic}]}`

Built via `muser albums`. **391 groups (378 multi-variant), 255 blurred removed** at build.
Region grouping catches nameless `ComfyUI-*` variants that filename-number grouping missed.

Backend wiring (`muser/service.py`): `/api/status` `albums` flag; startup `prime()`;
`_attach_albums()` inlines `album`/`album_count`/`album_removed` on every result;
`/api/score` inlines the same so the landing pool carries grouping;
**`GET /api/album?path=`** returns a cover's full variant group as aesthetic-ranked rows.

## The two UI behaviors (outpaintings only)

- **Unique spread** — `_showcaseRank` collapses the ranked pool to one rep per album group
  (`_collapseAlbums`, best-ranked variant kept) and drops blurred covers. Each tile shows a
  gold **"N variants"** badge. The landing is a spread of *distinct* covers to compare
  aesthetic blends on.
- **Click-to-cluster** — clicking a cover (`showAlbumCluster` → `/api/album`) swaps the page
  for that cover's variants, re-ranked by the active blend, with a **← back to spread** bar.
  Cluster mode skips the unique-spread collapse.

## The negative "Grey out" bar

A second bar below the blend bar — a **visual filter, not a ranking term**. Toggle any NSFW
classifier (**Falconsai / AdamCodd / Marqo**, fields `nsfw_falconsai/adamcodd/marqo`) + a
threshold; any enabled model scoring **at/above** it greys the cover **in place**
(desaturate + dim + red outline + a red **✕** replacing the variant badge), leaving the
aesthetic ranking untouched. Live "N greyed" count; `_applyGrey()` runs in the render
finalize so it applies to every page. State: `_neg = {enabled, th}` → `localStorage
muser-neg-v1`.

Why a filter and not a weight: the blend bar is positive-only allocation, so a positive
NSFW weight *surfaces* NSFW (backwards for curation). Greying demotes visually without
hiding — see `docs/DECISIONS.md`.

## Curated defaults (outpaintings)

- **Grey out** — `NEG_DEFAULT = { enabled:{nmar:true}, th:0.91 }` (Marqo on, 0.91).
- **Aesthetic blend** (fresh load, no saved state) — Aesthetic V2 23% · PickScore 33% ·
  Quality 31% · Taste 13% (Relevance off), seeded in `syncOutpaintingsNav`.
- A saved blend / grey-out in `localStorage` always wins; defaults only seed first-run or
  cleared browsers.

## Known gaps

- The album facet does **not** auto-rebuild after re-indexing (other facets do). Rebuild
  with `muser albums`. Corpus is static, so not pressing.
- NSFW classifiers over-flag album art (~43% score ≥0.5 on `nsfw_consensus`), which is why
  the grey-out default threshold is high (0.91) — tune per taste.
