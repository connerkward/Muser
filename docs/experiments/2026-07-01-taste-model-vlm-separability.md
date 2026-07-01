# Taste-model VLM separability — is the user's taste in the pixels?

Date: 2026-07-01

## Question

The outpaintings "Taste" blend metric is a logreg trained on the 163 verified favorite
outpaintings vs the rest of the library, over SigLIP embeddings. It is weak: CV precision
~11% (≈3× base-rate lift). Two readings of that weakness are possible:

1. **The taste isn't in the pixels** — favorites are chosen for reasons an image encoder
   can't see (the song, the moment, sentiment), so *no* pixel model can separate them.
2. **SigLIP just can't see it** — the taste is a real visual style, but the SigLIP feature
   space + a linear head is too weak to capture it; a stronger judge could.

If (2), then distilling a stronger judge into the taste model has headroom: rate a few
thousand library images with a VLM taste-match, retrain the logreg on those soft scores.
If (1), distillation buys nothing and Taste should stay a soft signal.

Test: does an **independent** judge (gpt-4o-mini, a different model from SigLIP) separate
favorites from random library images by looking at the images? Its separability is an
upper-ish bound on what any pixel model — SigLIP included — can recover, and cross-checks
whether the logreg's weakness is real or an artifact of that one model.

## Method

The 163 favorites are watermarked; to judge on clean pixels we recover **twins**:

1. For each favorite, SigLIP kNN into the library; accept the nearest neighbor if
   cosine ≥ `KNN_THRESH = 0.80`. Yields **146 clean twins** (median twin cosine **0.949**).
2. Sample **146 matched random** library images (`random.seed(SEED=0)`), disjoint from the
   twins.
3. gpt-4o-mini blind-rates each image 0–100 in an **independent** call (no image sees any
   other's score). Two framings:
   - **Framing 1 — zero-shot generic aesthetic.** Fixed aesthetic-craft rubric, one image
     per call. Script `/tmp/vlm_taste_eval.py`.
   - **Framing 2 — few-shot taste-match.** The first `N_EXEMPLARS = 8` twins are shown as
     exemplars of the target's taste; the remaining **138** twins + **138** random are
     rated 0–100 on *how well they match that taste*. Held-out (exemplars excluded from
     scoring). Script `/tmp/vlm_taste_eval2.py`.

Shared params (both scripts, for reproducibility): model `gpt-4o-mini`, image
`detail: "low"`, `temperature = 0`, `SEED = 0`, `KNN_THRESH = 0.80`. `MUSER_HOME=
/Users/conner/.muser-outpaintings`. Run:
`MUSER_HOME=/Users/conner/.muser-outpaintings uv run python /tmp/vlm_taste_eval2.py`.

Separability metrics: **AUC** (rank-based, threshold-free P[fav > random]), **Cohen's d**
(standardized mean gap), and the raw mean gap.

## Results

| Framing | fav mean | rand mean | gap | Cohen's d | AUC | n (each) |
|---|---|---|---|---|---|---|
| 1 — zero-shot generic aesthetic | 76.9 | 75.5 | +1.4 | 0.099 | **0.647** | 146 |
| 2 — few-shot taste-match (8 exemplars) | 70.8 | 58.6 | +12.2 | 0.531 | **0.640** | 138 |

## Verdict — don't distill

**AUC is essentially unchanged between framings (0.647 → 0.640).** Showing the VLM 8
concrete exemplars of the target's taste produces a much larger *mean gap* and Cohen's d
(the model rates favorites higher on an absolute scale when primed), but it does **not
improve its ability to rank a favorite above a random image** — the ordering quality, which
is what a distilled ranker inherits, is flat at ~0.64.

The mean-gap inflation in framing 2 is a scale/anchoring effect (priming shifts both
distributions and widens their spread), not new discriminative signal. AUC and Cohen's d
disagree precisely because AUC is invariant to that monotonic rescaling; AUC is the metric
that matters for distillation.

Interpretation: **the user's taste ≈ generic prettiness as far as an independent pixel
judge can tell.** A VLM given the taste explicitly still can't rank favorites above random
library images better than a generic aesthetic pass. That is reading (1): the separating
signal isn't reliably in the pixels. Distilling gpt-4o-mini into the taste logreg has
little headroom — you'd be teaching the logreg to reproduce a ~0.64-AUC judge, no better
than what SigLIP already gives. **Keep Taste as a soft blend signal; do not spend the
few-thousand-image gpt-4o-mini distillation pass.**

Threshold for the opposite call (not met): had framing 2 reached AUC > ~0.75, taste would
be a learnable style the VLM sees and SigLIP misses, and distillation would be worth doing.

## Corroboration

Framing 1's AUC **0.647** independently reproduces the taste logreg's weak separability
(CV precision ~11%, ~3× base-rate lift) — from a **different model family** (gpt-4o-mini vs
SigLIP+logreg). Two unrelated pixel judges landing on the same weak separation is evidence
the weakness is **real** (property of the favorites), not an artifact of the logreg or of
one embedding space.

## Artifacts

- `/tmp/vlm_taste_eval_result.json` (framing 1, per-image scores)
- `/tmp/vlm_taste_eval2_result.json` (framing 2, summary)
- Scripts: `/tmp/vlm_taste_eval.py`, `/tmp/vlm_taste_eval2.py`
