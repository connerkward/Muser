# Personal Google-Photos triage — session 2026-06-12/13

Built out the hidden `muser personal` sub-tool from "ingested + classified" to a
full interactive triage app, plus two label-trained models. All on `main`.

## Shipped

**Tabs (personal instance only; AI/Jobs/Color/Results hidden there):**
- **Faces** — 1,389 people-clusters (InsightFace ArcFace → HDBSCAN). Name / merge
  (path-anchored, survives re-cluster), bulk **👤 Personal** / **🗑 mark-for-deletion**,
  click-to-exclude → "label remaining personal" inside a person. Sorts: **look-alike**
  (greedy-NN over centroids, precomputed), **most photos**, **most likely deletable**
  (mean P(delete)). Hide-fully-tagged. Select-all ⇄ deselect-all toggles.
- **Evaluate** — labeled sample, live accuracy + confusion matrix, **Retrain on labels**,
  **→ main library** export, label-milestone learning-curve timeline, full-frame thumbs.
- **Cleanup** — delete candidates worst-first; click=save (persists `keep`), save-all /
  flag-rest / undo-last; metrics row + timeline; full-frame thumbs.
- **Trash** — all 🗑-flagged; restore; "Move all to system Trash" (`/usr/bin/trash`).
- **Cross-cutting:** header 🗑 hide-flagged toggle (persisted), per-thumbnail
  classification frame (personal/in-between/reference/trash) on mixed views, quick-mark
  P/In·b/R/🗑 bar on every thumbnail, ⓘ algorithm explainer with typeset annotated equations.

**Two label-trained models (logreg on the 768-d SigLIP embedding):**
- Personal-vs-reference head — **~86–90% CV** (beat the 8-signal fusion's 59% and the
  heuristic's 71%); re-buckets the corpus on retrain. in-between is derived (predicted-
  personal AND aesthetic), not learned. People-tag floors P≥0.95.
- Deletion model P(delete) — **~98% precision / 74% recall @P≥0.7**; suspects fold into
  Cleanup as `model NN%`.

**Training-data invariants (verified):** moving reference→main or trashing a delete never
drops an example from training — reference moves keep their embedding via a manifest+npz
link to the main corpus; deletes snapshot the embedding **at tag time** (`deleted_vectors.npz`)
and keep the flag. Train ONLY on the user's own Takeout labels (incl. exported refs),
never the general main corpus (distribution-bias trap).

## Bugs found + fixed (the expensive ones)

- **Label wipe (~1,800 → 80):** an unlocked per-path bulk write raced on a shared tmp
  file → corrupt JSON → loader's silent `except: return {}` → next save clobbered from
  empty. Restored 1,099 delete flags from the surviving model. Hardened: fail-loud on
  corrupt read (quarantine + raise), write-lock, anti-shrink guard, rolling `.bak`.
  Authored `central/rules/human-labeled-data-rule.md`.
- **`_load_npz` O(n) decompress** — `z["X"][k]` inside a 33k-iteration comprehension
  re-decompressed the whole npz each access (~1 TB of work → SIGKILL, looked like OOM).
  This was the real cause of every faces-clustering "crash" all session; masked by the
  corrupted 130-face npz. Fix: materialize `X`/`ids` once. (HDBSCAN also PCA-reduces 512→64.)
- **pillow-heif missing** — 2,374 iPhone HEICs silently skipped in faces + scores.
  Installed + registered in the shared loader; faces re-detected; scores topped up.
- **Horizontal page scroll** — header `flex-wrap:nowrap` overflowed 309px across
  720–1200px once the personal tabs were added. Fixed: `flex-wrap:wrap` + `body{overflow-x:clip}`.
  Boosted the design-skill overflow gate to mandatory/blocking.
- **"Mostly black" debias** — measured against the user's flags: dark≥0.70 was 56%
  delete-precision, BELOW the 76% base rate (anti-predictive). Now only near-total black
  (≥0.97/≥0.995) scores; mid-dark deferred to the learned model.

## Status / open

- Manual triage underway (~7% of 46,978 human-touched: 750 bucketed, ~3.4k delete flags).
  Bucket model plateaued ~88% → naming people on Faces is the highest-leverage action;
  deletion model still climbing.
- `scores.json` is **aesthetic-only** (NSFW consensus + aesthetic_v2 + pickscore +
  blended "interesting"). novelty / aesthetic_v25 / hps_v21 NOT computed — the slow
  scorer was stopped deliberately; caches are incremental if ever resumed.
