# User-Controlled Search & Ranking for Muser

*Design proposal — 2026-06-07. Grounded in the current codebase; no code written beyond this file.*

## Thesis

Mainstream discovery products (TikTok For You, Instagram Explore, YouTube Home,
Spotify) hide their ranking function and tune it for **engagement** — watch-time,
session length, ad surface — which is an objective *adverse* to the user
(documented by Tristan Harris / Center for Humane Technology, and by Stray, Adler
& Hadfield-Menell on "aligned recommenders," 2021). The function is opaque by
design: exposing it would let users defeat the engagement optimization.

Muser has **no engagement incentive**. It is local-first, single-user, $0, with
no ads and no retention target. That removes the only structural reason to hide
ranking. So the differentiator is not "better relevance" — it is **transparency
and steerability**: the user *declares* what they want, *sees* exactly how every
result was ordered, and *controls* the ordering directly. Ranking becomes a
visible instrument panel, not a black box.

This is achievable cheaply because Muser already computes, per image, a vector of
orthogonal signals — embedding similarity, five aesthetic/preference scores,
novelty, color palette, dimensions, AI-provenance — all percentile-normalized to
`[0,1]` and cached. The raw materials for a glass-box ranker are already on disk.

---

## What already exists (the foundation — don't rebuild)

Read before proposing; every proposal below extends rather than replaces these.

| Capability | Where | Notes |
|---|---|---|
| Cosine kNN over L2-normalized SigLIP/Jina vectors | `index.py::MuserIndex.search` | path-only projection, ~10 ms at k=288 |
| Near-dup collapse (embed / pHash / both) | `index.py::search_dedup`, `service.py::_search_dedup_precomputed` | precomputed `rep_of` map from `scores.json` |
| **Draggable blend console** | `web/app.html` ~L2619–2917 | segmented 100%-budget bar; drag dividers to reallocate among `vec + aesthetic_v2 + pickscore + aesthetic_v25 + hps_v21 + aesthetic`; click legend to toggle a metric. Writes 6 weights, persists to localStorage. |
| Client-side linear re-rank | `web/app.html::applySortBlend`, `_sortBlendKey` | `final = Σ wᵢ·scoreᵢ` over per-hit fields |
| Negative / steering vector | `service.py::/api/search` `neg`, `neg_strength` | `qv = normalize(qv − s·nv)`; single concept |
| Composed retrieval (CIR L1) | `service.py::/api/search-compose` | `q = α·img + β·text`, normalized (Liu et al. CIRR ICCV'21 baseline) |
| Color-palette search | `color.py`, `/api/search-color` | LAB ΔE over median-cut palettes; separate index |
| Per-image scores (percentile-normalized) | `score.py::score_all` → `~/.muser/scores.json` | `interesting, novelty, aesthetic, aesthetic_v2, pickscore, aesthetic_v25, hps_v21, nsfw_*`; **all `_pct()`-normalized to `[0,1]`** (27,794 images scored) |
| Cluster labels (HDBSCAN) | `cluster.py` → `clusters.json` | already feeds `service.py::_refinements` (cluster-vote pseudo-relevance-feedback chips) |
| Captions | `caption.py` → `captions.jsonl` | GPT-4o-mini, one sentence/image; not yet a search index |
| Metadata filters | `index.py::MetaFilter`, `/api/filter` | width/height/aspect/filesize as a LanceDB prefilter |
| Eval harness | `eval/harness.py` | ranx: hits@1, recall@5/10, mrr, ndcg@10, map + latency |

Crucial property: because every score in `scores.json` is **percentile-normalized
to the same `[0,1]` scale**, a linear blend `Σ wᵢ·sᵢ` is already
unit-commensurable — no per-query min-max needed for the aesthetic terms. The one
exception is the live vector cosine (`vec`), which is per-query and *not*
percentile-normalized; see Proposal 1's note.

---

## Proposals

Each: idea · user control (and how it's shown transparently) · method/math ·
signals used / to build · evaluation · effort.

### 1. Declarative Weighted Blend, formalized + explainable ★ (build first)

**Idea.** The blend console exists but is (a) incomplete as a *declarative* ranker
— it omits recency, color-match, and caption-text as budgetable terms — and (b)
opaque per-result: the user sees the global budget but not *why this specific
image landed at rank 4*. Formalize the linear scoring function as a first-class,
fully-disclosed object and attach a per-result breakdown.

**User control.**
- The existing draggable 100%-budget bar, extended with three new toggleable
  segments: **Recency**, **Color-match** (active only when a color/hex is set),
  **Caption-BM25** (active only when query text exists). Budget still sums to 100%
  so the function is always a convex combination — interpretable by construction.
- **"Why this ranked here"** disclosure per card: a hover/expand stacked bar
  showing each term's *contribution* `wᵢ·sᵢ` to that image's final score, with the
  numeric value. This is the glass-box payoff — the user can read the ranking
  function off any single result.
- A copy-paste textual form of the function, e.g.
  `final = 0.55·relevance + 0.30·aesthetic_v2 + 0.15·pickscore`, shown live (the
  caption row at `web/app.html` L1075 already reserves this spot).

**Method.** `final(img) = Σᵢ wᵢ · sᵢ(img)`, `Σwᵢ = 1`, `wᵢ ≥ 0`.
- Existing terms: `vec` (cosine), `aesthetic_v2`, `pickscore`, `aesthetic_v25`,
  `hps_v21`, `aesthetic` — already shipped per-hit.
- **`vec` normalization fix:** the five aesthetic terms are percentile-normalized
  but the live cosine is not, so at high aesthetic weight the blend can be
  dominated by the un-normalized term's wider range. Rank-normalize `vec` *within
  the candidate set* (cheap: `argsort` over ≤288 hits client-side, or server-side
  in `_run_search`) so all terms share the `[0,1]` percentile scale and the
  declared weights mean what they say. This is a real correctness bug in the
  current blend, not just polish.
- **Recency** term: `s_recency = pct(mtime)` over the candidate set (mtime already
  in the LanceDB row as `mtimeMs`). Add it to the per-hit payload in
  `_attach_dims`/`_attach_scores`.
- **Color-match** term: reuse `color.py::search(rgb)`'s per-image `Σ frac·sim`
  score as `s_color`; surface it on hits when a hex is set (one extra cache
  lookup, already in RAM).
- **Caption-BM25** term: see Proposal 7 (caption text index); contributes
  `s_caption = pct(bm25(query, caption))`.

**Signals used:** all of `scores.json` + cosine + color cache + mtime. **To
build:** caption BM25 index (Prop 7); `vec` rank-norm; 3 new per-hit fields;
per-result contribution breakdown UI.

**Evaluation.** This is the one proposal with a clean automated metric. Sweep the
weight simplex on the domain eval set (`eval/harness.py`, VLM-generated GT) and
report ndcg@10 / recall@5 as a function of `(w_vec, w_aesthetic, …)` — gives the
user a *default* that is empirically justified, and proves the blend never hurts
relevance at `w_vec=1`. A/B "relevance-only" vs "user's saved blend" on a held-out
query set; report whether the user's chosen blend changes recall@5 (expected:
small cost, large subjective gain).

**Effort.** Medium. ~1 day: `vec` rank-norm + recency/color fields (~½ day), the
per-result contribution breakdown UI (~½ day). Caption-BM25 term gated on Prop 7.

---

### 2. Diversity / Serendipity control (MMR) ★ (build first)

**Idea.** Pure relevance ranking returns 24 near-identical hits (the dedup
collapse helps but only removes *near-duplicates*, not *semantic clumps* — e.g. 20
variations of the same car at the same angle). Give the user an explicit
anti-filter-bubble knob: trade relevance for *coverage of the result space*. This
is the direct inversion of engagement-maximizing recommenders, which narrow toward
a predicted-preferred cluster; here the user can deliberately widen.

**User control.** A single **Diversity** slider `λ ∈ [0,1]`, shown next to the
blend console with a live caption: *"0 = most relevant, 1 = widest variety."*
Transparent because the mechanism is nameable — "Maximal Marginal Relevance" — and
its effect is visible (results visibly spread across subjects as you drag).
Optional second mode: **cluster round-robin** ("one best per cluster"), driven by
the existing `clusters.json` labels.

**Method.** MMR (Carbonell & Goldstein, SIGIR 1998):
`MMR = argmax_{i∉S} [ (1−λ)·rel(i) − λ·max_{j∈S} sim(i,j) ]`,
greedily selecting from the over-fetched candidate pool. `rel(i)` = the blended
score from Prop 1; `sim(i,j)` = cosine of the cached embeddings. Over-fetch is
already done (`n = max(k*12, 200)` in `search_dedup`); MMR re-orders that pool.
Cluster round-robin is a degenerate cheap variant: bucket candidates by
`_path_to_label`, emit best-per-bucket in a rotation.

**Signals used:** embeddings (already over-fetched), blended relevance (Prop 1),
`clusters.json`. **To build:** an MMR re-rank step in `_run_search` (the vectors
are *not* currently fetched on the fast path — `select(["path"])` drops them; MMR
needs them, so either fetch vectors on the diversity path or precompute a
candidate-pool similarity from the cached vectors). ~60 LOC.

**Evaluation.** Diversity isn't a relevance metric, so measure both: (a) ndcg@10
must not collapse at low λ (sanity), and (b) **intra-list diversity** = mean
pairwise embedding distance of the top-k (Ziegler et al., WWW 2005), reported as a
function of λ. Optionally **α-nDCG** (Clarke et al., SIGIR 2008) if subtopic
labels (clusters) are treated as facets. The deliverable is the λ↔diversity curve,
shown to the user as "what this slider does."

**Effort.** Medium-low. ~½ day, plus the vector-availability plumbing on the
diversity path.

---

### 3. Relevance Feedback / iterative steering (live query-vector move) ★ (build first)

**Idea.** Turn search into a conversation with the index. After a result set, the
user marks hits 👍 / 👎; the query vector moves toward the liked and away from the
disliked, and re-queries — *and the user watches the vector move* (rendered as the
delta in the top results, plus an optional 2D projection dot). This is CIR's
"more like this but X" generalized to multi-example relevance feedback, and it is
the single most "agency-forward" interaction: the user is literally steering the
embedding.

**User control.** 👍/👎 buttons already have a natural home on each card. A
**"Refine"** button applies the feedback. Transparency: show *which* marked images
pulled the query and by how much (a small "your picks moved the search toward
[thumbnails]" strip), and optionally plot the old vs new query point on the
existing 2D projection (`projection.py` already produces a UMAP/PCA map).

**Method.** Rocchio relevance feedback (Rocchio 1971, the classical IR algorithm,
trivially adapted to dense vectors):
`q' = α·q + β·mean(liked_vecs) − γ·mean(disliked_vecs)`, then L2-normalize.
Defaults `α=1, β=0.75, γ=0.25`. This is *exactly* the same vector-arithmetic
machinery `/api/search` already uses for `neg` (which is the γ-term with one
example) and `/api/search-compose` (α·img+β·text) — it's a strict generalization,
so it reuses `_run_search` unchanged; only the query-vector construction is new.
The liked/disliked image vectors are already in LanceDB (one `path IN (...)`
fetch). Pseudo-relevance-feedback variant (auto, no clicks): take the top-m hits
as implicit positives — Muser already does a *label*-level version of this in
`_refinements`; this is the vector-level version.

**Signals used:** embeddings of marked images, current query vector. **To build:**
a `/api/search-feedback` endpoint (or extend `/api/search` with `pos=[uid]`,
`neg_uids=[uid]`); the 👍/👎 state + Refine UI; optional projection overlay. ~80
LOC server + UI.

**Evaluation.** Standard IR RF protocol: simulate feedback on the domain eval set
(mark the known GT image as 👍 when it appears in top-k, measure recall@5 *before
vs after* one feedback round). Report the recall lift per feedback round — RF
should monotonically improve recall on under-specified queries. ranx supports the
before/after comparison directly.

**Effort.** Medium. ~1 day. Highest agency-per-LOC of any proposal because the
arithmetic backend already exists.

---

### 4. Personal Aesthetic Model (Bradley-Terry, user-owned)

**Idea.** The five shipped aesthetic models encode *other people's* averaged taste
(LAION raters, Pick-a-Pic crowd, HPS prompt-preference). Let the user train a
6th score that is *theirs* — a preference model fit to their own pairwise choices,
on top of frozen embeddings. The political point: on a platform, the preference
model is the platform's asset and is hidden; here it is the user's, inspectable,
exportable, deletable.

**User control.** A lightweight **"A or B?"** compare mode (two images, pick the
one you prefer) — active-learning-ordered so each comparison is maximally
informative. The trained score then appears as a new toggleable segment
**"My taste"** in the blend console (Prop 1), identical in kind to the other
aesthetic terms. Transparency: show the model's current confidence, the number of
labels, and a "show me my highest/lowest taste images" sanity panel; the weights
live in a plain file the user owns (`~/.muser/taste.json`).

**Method.** Bradley-Terry (1952): `P(i ≻ j) = σ(f(xᵢ) − f(xⱼ))`, where `f` is a
small linear/MLP head on the frozen embedding `x` (no backbone fine-tuning — the
embeddings are already computed and stored). Fit by logistic loss over the user's
pairwise labels. **Active learning:** select the next pair near the current
decision boundary / max disagreement (uncertainty sampling) so ~100 pairs suffice
(consistent with the LAION/PickScore literature on small-head preference fitting).
At inference, `taste_score(img) = pct(f(xᵢ))` over the library, written into
`scores.json` like any other facet via the `facets.py` scaffolding.

**Signals used:** frozen embeddings (in LanceDB), user pairwise labels (new). **To
build:** the compare UI, the BT head fit (~scikit-learn `LogisticRegression` on
pairwise-difference features — *don't hand-roll*, per the "use packages" rule), a
`muser taste` CLI + facet writer. ~120 LOC + UI.

**Evaluation.** Held-out pairwise accuracy (fraction of held-out comparisons the
model predicts correctly) — report the learning curve vs #labels. This is the
honest metric; ranx doesn't apply (no relevance GT for "taste"). Cross-check:
correlation (Kendall-τ) between the user's taste score and each shipped aesthetic
model — tells the user *how their taste differs from the crowd*, which is itself a
transparency feature.

**Effort.** Medium-high. ~1.5 days (UI + active-learning loop is the bulk).

---

### 5. Ranking Inspector ("show me why")

**Idea.** A dedicated panel that fully explains the *current ordering* — not per
result (that's Prop 1's breakdown) but the whole list: the active scoring
function, each term's realized weight, the candidate pool size, what dedup/MMR
removed, what the hide-policy filtered (NSFW/cluster/dead), and a sortable table of
every candidate with all its raw signal values. The "view source" of the ranking.

**User control.** A toggle that flips the result grid into an audit table:
columns = `vec, aesthetic_v2, pickscore, …, recency, color, final, rank`,
sortable, with the filtered-out rows shown struck-through and labeled with *why*
they were dropped. Nothing is hidden — including Muser's own demo-mode/NSFW hiding
(`service.py::_filter_results`), which is itself disclosed rather than silent.

**Method.** No new ranking math — pure disclosure. Surface the data
`_run_search`/`applySortBlend` already compute: attach a `_debug` block per result
(its term contributions) plus a list-level summary (pool size, removed counts by
reason). This directly counters the engagement-platform anti-pattern where even
the *existence* of filtering is concealed.

**Signals used:** everything already on each hit. **To build:** a `debug=true`
param on `/api/search` that includes filtered rows + reasons; the audit-table UI.
~½ day.

**Evaluation.** No retrieval metric (it's an explainability surface). Validate by
correctness: every number in the table reproduces the displayed order when sorted
by `final`. This is a unit-testable invariant, not an A/B.

**Effort.** Low-medium. ~½–1 day. High trust-payoff for low cost; natural
companion to Prop 1.

---

### 6. Soft faceted preferences (filters as weights, not gates)

**Idea.** Today's metadata filters (`MetaFilter`) are *hard gates* — an image is
in or out. Offer a *soft* mode: declare a **preference** ("prefer landscape,"
"prefer high-res," "prefer recent") that biases ranking without excluding, so a
slightly-off but otherwise-perfect hit still surfaces. Same idea applies to color
("lean blue") and AI-provenance ("prefer non-AI").

**User control.** Each existing filter row gets a **hard / soft** switch and, in
soft mode, a weight slider that joins the blend budget. Shown transparently as
just more segments in the Prop 1 bar ("Aspect-pref 10%", "Recency 15%"). The user
decides per-facet whether it's a constraint or a nudge.

**Method.** Convert each facet to a `[0,1]` desirability score and add it as a
blend term: aspect-preference `s = exp(−|aspect − target|/τ)`; resolution-pref
`s = pct(long_side)`; recency `s = pct(mtime)`; provenance `s ∈ {0,1}` from the
c2pa cache. All percentile/bounded so they compose with the existing convex blend.
Hard filters remain available (LanceDB prefilter) for true constraints.

**Signals used:** `MetaFilter` columns, `mtimeMs`, c2pa cache, color cache. **To
build:** soft-score functions + per-facet hard/soft toggle UI; these become Prop 1
budget terms. ~½ day on top of Prop 1.

**Evaluation.** Sanity ndcg@10 (soft prefs shouldn't tank relevance at low
weight); primarily a UX/agency feature. Could measure "satisfied-constraint rate
in top-k" as a function of soft weight.

**Effort.** Low, *given Prop 1* (it's just more terms in the same machine).

---

### 7. Multi-concept steering vectors (push toward / push away, with strengths)

**Idea.** Generalize the single `neg` term into a **steering panel**: any number of
text or image concepts, each with a signed strength, summed into the query vector.
"More cinematic (+0.6), less cartoon (−0.8), toward [this reference image]
(+0.4)." CLIP/SigLIP ignore in-prompt negation (bag-of-words effect, Yuksekgonul
et al. ICLR 2023 — already cited in `service.py`), so this *must* be vector
arithmetic, which is exactly why it belongs as an explicit, visible control rather
than buried in prompt text.

**User control.** A chip list under the search bar: each chip = concept + a
−/+ strength slider; text or dropped-image concepts both allowed. Transparent
because each chip's contribution is nameable and the net effect is the same
arithmetic shown in the inspector (Prop 5).

**Method.** `qv = normalize(qv + Σₖ sₖ·cₖ)`, where `cₖ` is the embedding of
concept k (text or image) and `sₖ` its signed strength. This is the existing `neg`
math (one negative term) extended to N signed terms — reuses `_run_search`. Also
subsumes CIR-compose (`/api/search-compose`) as the special case of one positive
image + one positive text concept.

**Signals used:** the shared-space embedder (text + image), already warm. **To
build:** the multi-chip UI + an endpoint accepting `concepts=[{text|path,
strength}]`. ~½ day.

**Evaluation.** Hard to auto-eval (no GT for "more cinematic"). Use CIR benchmarks
(CIRR / Fashion-IQ from `eval/datasets.py`) for the special case of one
image + one modification text, reporting recall@10 against the composed-retrieval
GT — validates the arithmetic is sound. The multi-concept general case stays
qualitative.

**Effort.** Low-medium. ~½–1 day; the backend is a tiny generalization of `neg`.

---

## Cross-cutting design principles

- **Every term shares the `[0,1]` percentile scale** so a convex blend is honest.
  The one current violation (live `vec` cosine un-normalized) is a real bug to fix
  in Prop 1.
- **All controls persist locally** (the console already uses localStorage) — the
  user's declared preferences are *their* state, never server-side profiling.
- **The function is always disclosed** — a copy-pasteable formula + per-result
  contribution. No hidden re-ranking, no silent "engagement" boosts. Even Muser's
  own NSFW/cluster hiding is surfaced in the inspector (Prop 5).
- **Reuse the warm embedder + cached scores** — none of these require a new model
  load except Prop 4's tiny BT head (which trains on frozen embeddings in
  seconds).

## Recommended build order (top 3)

1. **Proposal 1 — Declarative Weighted Blend, formalized + explainable.** The
   console is 80% built; the highest-leverage work is the missing pieces that make
   it *honest and legible*: rank-normalize `vec` (correctness bug), add
   recency/color/caption terms, and the per-result "why this ranked here"
   breakdown. This is the spine every other proposal plugs into (Props 4, 6, 7 add
   terms; Prop 5 inspects it). It also has the cleanest automated eval
   (weight-simplex sweep on the domain GT). Ship this first.

2. **Proposal 3 — Relevance Feedback (live query-vector move).** Highest agency
   per line of code: the vector-arithmetic backend already exists (`neg`,
   `compose`), so this is mostly a `pos/neg uids` query-construction + 👍/👎 UI. It
   delivers the most visceral "I am steering this" experience and is the clearest
   embodiment of the thesis (the user moves the search, watches it move). Cleanly
   evaluable via simulated-feedback recall lift on the eval harness.

3. **Proposal 2 — Diversity / Serendipity (MMR).** The explicit anti-filter-bubble
   control — the conceptual opposite of an engagement recommender, and a strong
   demo of "Muser optimizes for *you*, not for retention." Cheap (~60 LOC, classic
   MMR), with a real measurable axis (intra-list diversity vs λ). Build third
   because it depends on Prop 1's blended `rel(i)` as its relevance term.

Props 5 (inspector) and 7 (multi-concept steering) are strong fast-follows — both
are mostly disclosure/UI over machinery the first three establish. Prop 4
(personal taste model) is the most ambitious and most distinctive long-term, but
needs the labeling-UX investment, so it lands after the blend spine is solid.

## Honest counterpoints

- **Most users won't touch a 9-segment budget bar.** Power-user features have a
  long tail of disuse. Mitigation: ship empirically-justified *defaults* (from
  Prop 1's simplex sweep) so the controls are opt-in polish, not required labor —
  the product must be excellent with every knob untouched.
- **Diversity/RF can *hurt* a well-specified query.** If the user knows exactly
  what they want, MMR and serendipity add noise. Both must default *off* and be
  clearly framed as "widen / refine," not "improve."
- **"Transparency" has a credibility ceiling here.** The embedding itself is an
  opaque 1024-d SigLIP vector — Muser can explain the *blend over signals* fully,
  but cannot explain *why SigLIP thinks two images are similar*. The honest claim
  is "transparent ranking *policy*," not "transparent *similarity*." Don't
  overclaim the latter.
- **Personal taste model = a filter bubble of one.** Prop 4 optimizes toward the
  user's past choices, which is the same narrowing mechanism the thesis critiques —
  just user-owned. The Kendall-τ-vs-crowd readout partially mitigates by making
  the narrowing *visible*, but the tension is real and worth stating.
