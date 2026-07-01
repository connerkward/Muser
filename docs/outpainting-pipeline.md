# The outpainting pipeline — end to end

> **Status / intent.** This documents the *full* album-art → in-car ultra-wide outpainting
> pipeline, end to end, for the **portfolio-2026** page. It lives in this repo for now and
> will move into `portfolio-2026` when the page is built. The Muser outpaintings instance
> (see [`outpaintings-curation.md`](./outpaintings-curation.md)) is the **filter-and-sort
> curation reconstruction** of this work — this doc is the pipeline that *produced* the
> corpus Muser curates. Context: this was work Conner was doing before being laid off;
> both this write-up and the Muser tool reconstruct it "as if we'd built it before then."

---

## Source braindump (verbatim — do not edit; expand below)

> gather top n played songs (Zipfian distrobution) -> inlcude marginal and total price
> evaluation estimate, given customer base / estimated fleet size based on per model
> ownership, average drive time + average song length to estiamte price of intial
> outpainting corpus cost / coverage and also running cost.
>
> take album art, put into outpainting pipeline (sub bullets of how it actually works,
> maybe go find the comfyui workflows candidates, render the node previews, and ill tell
> you which one it is and you can generate thse sub bullets) but i think it invovled taking
> album art, using SAM+Grounding Dino to remove text, logos, and labels, upscaled it with
> ultrashapr4x, then used a custom outpainting method for this ultrawide use case involving
> creating what ill call a 'latent smudge' which was essentiall a small model called biglama
> which tool the structure of the input album art and smudged it in a semi repeating pattern
> across the ouptainted region, which we would slightly blur to reduce complexity in final
> image tripping the heuristc driver distraction filter. then with somethig to go off of,
> the model outpaints, lots of prompt tuning and paramater tuning via cross checking with
> the heuristic driver distraction verifier. also explore using controlnets to limit
> geenrated detail.
>
> with goal of producing highest quality output while also getting legal sign off. we
> setteld on stable diffusion and legal said we should just continue until cease and desist.
>
> prepass with nsfw models, run all finished outpaintigs through semantic / nsfw pass vlm,
> heuristic based driver distraction verifier. next wanted to train small model on image
> arena / in house human labelers / aws mechnical turk workers (does this make sense?) the
> goal was eventually
>
> cross referenced with simulator eye tracking studies, built a unity + eye tracking system
> with quick vibe coded timeline interface for ux researcher analysis. driver distraction
> also tested for nighttime and daytime scenarios, using NHTSA guidlines.
>
> regenerate newly released songs weekly from top 1000 charts.
>
> deploy to car over wire using optimally packed webp, store in browser as archival
> multilayer tiff with all components, so can be regeneretd from any step. worked with
> extremely slow 3 country team ( india implmetnation, germany pm, remote but local on
> vehicle backend service devs, local aws dev). explain price saved from optimized webp vs
> png. optimized webp is based on actual screen res and differential compression (does this
> make sense?). on car was cached 100 ish images with tiem decay function. thought about
> reaching foward but spotify api i blevei had tehcnical lmitations.
>
> also did some one off experiements with using models which generate normals to do time of
> day / darkmode based relighting of images, but never got a chacne to test driver
> distraction for time of day variants.
>
> what other features that ive been working on would make sense as part of this in a dream
> scenario?

---

## Expanded pipeline

### 0. Song selection + cost model — *what to generate, and what it costs*

- **Corpus by play frequency (Zipfian).** Music listening follows a Zipf/power-law: a small
  head of songs accounts for most plays. So you don't generate art for *every* song — you
  cover the **top-N** by play count and get most of the real-world coverage cheaply. The
  Zipfian tail is exactly why a fixed corpus works: coverage rises steeply then flattens.
- **Coverage/price model.** Estimate:
  - **Fleet size** = customer base × per-model ownership mix (how many cars of each model
    have the display) → number of screens in the field.
  - **Exposure** = average drive time ÷ average song length → songs-per-drive → which slice
    of the Zipfian head actually gets seen per session.
  - **Initial corpus cost** = N × (per-image generation cost through the full pipeline —
    SAM/GroundingDINO + upscale + LaMa + diffusion + verification GPU-seconds).
  - **Marginal cost** = cost of covering the next song (falls as you go down the tail, since
    each additional song is played by fewer people → diminishing coverage per dollar).
  - **Running cost** = weekly regeneration of newly-charting songs (see §6) + storage +
    over-the-wire delivery.
  - The decision this model drives: **where on the Zipfian curve to cut N** — the point
    where marginal coverage-per-dollar drops below worth-it.

### 1. Album art → outpainting

The core: a square album cover → a clean, ultra-wide (~6:1) in-car image. **Confirmed from
the real production workflow** (embedded in every `ComfyUI-BigLama-*.png`'s metadata — full
node-map in [`outpainting-workflow-candidates.md`](./outpainting-workflow-candidates.md)). It
splits into an **upstream Python prep stage** and the **ComfyUI outpaint graph** — the
SAM/GroundingDINO + UltraSharp steps are *not* in the ComfyUI graph, they run before it.

**1a — Upstream prep (separate batch Python, not ComfyUI):**
- **Text / logo / label removal — SAM2 + GroundingDINO.** GroundingDINO does open-vocabulary
  detection ("text", "logo", "wordmark") → boxes; SAM2 turns boxes into precise masks; a
  LaMa remover (`AILab_LamaRemover`, same `big-lama.pt`) erases them. Outputs
  `*_outpaint_ready.jpg`. (Removes copy/branding so the outpaint isn't anchored to it, and
  kills the legible text a distraction heuristic penalizes.)
- **Upscale — 4× UltraSharp.** ESRGAN-family `4x-UltraSharp`, also a separate upstream stage,
  so the clean cover has enough resolution to seed the wide canvas without center softness.

**1b — ComfyUI outpaint graph** (one large ~580–630-node multi-branch dev graph):
1. **"Latent smudge" — BigLaMa.** The custom ultra-wide trick, and the novel part.
   `INPAINT_LoadInpaintModel → INPAINT_InpaintWithModel` with model **`big-lama.pt`** (LaMa,
   Suvorov et al. 2021 — Fourier convolutions, strong at large-mask structure) takes the
   album-art structure and **smears it in a semi-repeating pattern across the fill region**
   (noise seeded by an `Image Power Noise` blue-noise source), giving the diffusion model
   *something coherent to extend from* instead of blank latent. See the smudge output at
   `~/Desktop/cc-muser/outpaint-biglama-smudge.png` — sharp cover center, structure smeared
   + softened outward.
2. **Slight blur — reduce complexity.** `INPAINT_MaskedBlur [100,0]` (+ `Blur`×9,
   `ImageBlend`, `Image Blend by Mask`) softens the smudge so the final image's high-freq /
   busy content stays **under the driver-distraction heuristic's threshold**.
3. **Asymmetric outpaint pad.** `ImagePadForOutpaint [left=1680, top=104, right=1152,
   bottom=160, feather=100]` — **left ≈ 1.46× right**, so the cover sits **offset-right** of
   center. *(This is exactly the offset the Muser curation mask was independently re-derived
   to — `x∈[0.52,0.64]`, center ≈0.58 — from cross-variant variance. Full circle.)*
4. **Outpaint — SDXL inpaint.** Primary checkpoint **`realvisxlV50_v30InpaintBakedvae`**
   (RealVisXL 5.0, SDXL-inpaint). Experimental alt branches in the same graph: a dedicated
   `sdxl-inpaint` path and a **Flux-fill** (`flux1-fill-dev`) path (`FluxGuidance` 0.5/3) —
   the seed the crashcourse Flux template grew into. Heavy **prompt + param tuning**, each
   candidate cross-checked against the distraction verifier (below) — the verifier is *in the
   tuning loop*, not just a final gate.
5. **ControlNet — cap invented detail.** `sdxl-controlnet-tile` limits how much detail SDXL
   invents in the wings, keeping the extension calm (another distraction lever) — this is the
   "controlnet to limit generated detail" from the braindump, confirmed present.
6. **In-graph driver-distraction verifier — IMIC.** `IMIC TrafficLight` + `Distractive Area
   Percentage` / Entropy / Illumination / Edge Ratio nodes score each candidate's distraction
   right inside the graph, closing the tuning loop.

### 2. Model choice + legal

- **Stable Diffusion**, chosen for quality + control (local, tunable, ControlNet ecosystem)
  over closed APIs.
- **Legal posture:** the team got sign-off to **proceed until cease-and-desist** — i.e.,
  album art is copyrighted, the derived outpaintings are a legal gray area, and legal's
  call was to ship and stop only if a rights-holder objects, rather than pre-clear every
  cover. (Worth stating plainly in the portfolio as the real-world constraint it was.)

### 3. Quality / safety passes (every finished outpainting)

- **NSFW pre-pass** with NSFW classifiers (the same family Muser exposes: Falconsai /
  AdamCodd / Marqo) to drop covers that outpaint into unsafe content.
- **Semantic / NSFW VLM pass** — a vision-language model as a second, context-aware check
  (catches what pixel classifiers miss).
- **Heuristic driver-distraction verifier** — the safety gate: scores complexity / contrast
  / motion-suggestion / text against the distraction heuristic.
- **Planned: distill a small model** from **Image-Arena-style pairwise prefs + in-house
  labelers + MTurk** to make the aesthetic/distraction judgment fast and automatic. *(See
  "Does this make sense?" below — yes for aesthetics, with a caveat for the safety-critical
  distraction label.)*

### 4. Validation — simulator + eye-tracking

- **Unity + eye-tracking rig** with a **vibe-coded timeline interface** so a UX researcher
  could scrub eye-tracking sessions and analyze fixation/dwell on the generated art.
- Cross-referenced generated art against **simulator eye-tracking studies** — grounding the
  heuristic verifier in real human gaze data.
- **Day + night scenarios**, tested against **NHTSA driver-distraction guidelines** (the
  established regulatory basis for visual-manual distraction limits).

### 5. Refresh cadence

- **Weekly regeneration** from the **top-1000 charts** — newly-released / newly-charting
  songs get art each week, so the head of the Zipfian stays covered as it shifts.

### 6. Deployment + storage

- **Over-the-wire to the car: optimized WebP.** Encoded to the display's **actual screen
  resolution** (no wasted pixels) with **differential compression**. WebP over PNG is a large
  saving on photographic content (see cost note below).
- **Archival: multi-layer TIFF** stored in the browser/pipeline holding **all components**
  (source cover, mask, smudge, outpaint, final) so any image can be **regenerated from any
  step** — the same layered-TIFF-with-all-passes idea used elsewhere in the fleet. Deploy
  the compact WebP; archive the fat regenerable TIFF.
- **On-car cache** of ~**100 images** with a **time-decay function** (recently/likely-played
  art stays warm; stale art evicts).
- **Look-ahead (considered, blocked):** pre-generating the *next* song's art was limited by
  the **Spotify API** (queue/next-track access restrictions).

### 7. Team + delivery reality

- A **slow, 3-country team**: **India** (implementation), **Germany** (PM), on-vehicle
  **backend service devs** (remote but local-on-vehicle), and a local **AWS** dev. Worth a
  candid line in the portfolio — cross-timezone, cross-org coordination was a real constraint
  on velocity.

### 8. Cost saving — WebP vs PNG

- **The saving:** PNG is lossless and huge for photographic/gradient content; **lossy WebP at
  high quality is typically ~25–35% of the PNG size** for this kind of imagery, and encoding
  to the exact screen resolution removes further waste. Over a fleet × cache × weekly-refresh,
  that's the difference between viable and not for over-the-wire delivery.
- **Differential compression:** across the ~100-image cache (or across a song's variants),
  send only deltas / shared structure rather than each image whole. *(See assessment below.)*

### 9. One-off experiments

- **Normals-based relighting** — models that generate surface normals from an image, used to
  **relight** covers for **time-of-day / dark-mode** variants (a night version of each cover).
  Built but **never got to test driver-distraction for the relit variants**.

---

## "Does this make sense?" — honest assessments

- **Train a small model on Image Arena / in-house labelers / MTurk — yes, with a caveat.**
  Distilling human judgment into a fast preference model is standard (this is the reward-model
  half of RLHF). Image-Arena-style **pairwise** comparisons are the right elicitation format
  (easier + more reliable than absolute scores). **Caveat:** for the **safety-critical
  driver-distraction** label, MTurk crowd quality is risky — you'd want expert/in-house labels
  + gold-standard control questions, and ideally ground it in your **simulator eye-tracking**
  data rather than crowd opinion. So: MTurk/arena for *aesthetic* preference, expert + gaze
  data for *distraction*. Split the two labels; don't let a crowd vote on safety.
- **Optimized WebP = actual-screen-res + differential compression — yes.** Both levers are
  real: (1) never encode above the panel's native resolution (a common waste); (2)
  differential/delta compression across a set that shares structure (variants of one cover, or
  a cache with visual overlap) genuinely cuts bytes. One nuance: "differential" across
  *unrelated* covers buys little — it pays off within a song's variants or across the
  day/night pair of the same cover, where structure is shared. Frame it as delta-within-a-group.
- **Spotify look-ahead blocked by API — plausible/correct.** Spotify's API has historically
  restricted reliable access to the play **queue / next track** (and rate limits), so
  pre-generating the upcoming song's art wasn't dependable. Reasonable reason to shelve it.

---

## Dream-scenario features (what else would fit)

Building on features already in flight elsewhere in the stack:

- **Muser as the curation front-end for this pipeline** (already built) — the filter/sort/
  grey-out/unique-spread surface *is* the QA + selection layer this pipeline needed.
- **Per-driver / per-mood personalization** — bias generation toward a driver's aesthetic
  (the Taste model idea from Muser: a small preference model per user) so the same song
  renders to *their* look.
- **Time-of-day relighting, closed-loop** — finish the normals-relight experiment and put the
  day/night variants through the *same* distraction verifier + eye-tracking gate (the missing
  test), so dark-mode ships validated.
- **Variance-field "signatures" as a design artifact** — the per-pixel variance art
  ([`variance-maps.md`](./variance-maps.md)) is a natural visual for the portfolio *and* a QA
  lens (shows which region the model actually invented vs. kept).
- **Motion / ambient extension** — subtle, sub-distraction-threshold parallax or breathing on
  the outpainted wings (validated against NHTSA motion limits), instead of a static frame.
- **Live cost/coverage dashboard** — the §0 model as a running tool: dial N on the Zipfian
  curve, see fleet coverage, corpus cost, and weekly running cost update live (an ephemeral
  parameter-tuning interface, per the pattern in `ephemeral-design-interfaces.md`).
- **On-device tiny distraction verifier** — ship a distilled verifier *to the car* so
  relit/cached/variant images can be re-checked at the edge before display.
- **Provenance/regeneration UI** — expose the layered-TIFF "regenerate from any step" as a
  tool: pick a cover, re-run from the smudge or the outpaint with new params.

---

## Open items

- [x] **Real ComfyUI workflow identified** — embedded in `ComfyUI-BigLama-*.png` metadata
      (RealVisXL SDXL-inpaint + big-lama smudge + asymmetric pad + tile-ControlNet + IMIC
      distraction verifier; SAM/GroundingDINO + UltraSharp were separate upstream stages). §1
      above is written from it. Node-map: [`outpainting-workflow-candidates.md`](./outpainting-workflow-candidates.md).
      **Confirm with Conner** it's the right graph, then finalize.
- [ ] **Node-graph visual render** — not obtainable without launching ComfyUI (drag any
      `ComfyUI-BigLama-*.png` in to see the live graph). For the portfolio, the smudge-stage
      image itself (`~/Desktop/cc-muser/outpaint-biglama-smudge.png`) is the better visual.
- [ ] Fill exact numbers into §0 and §8 (fleet size, per-image GPU cost, WebP/PNG ratio on
      real samples) once available.
- [ ] Move this into `portfolio-2026` when the page is built; keep this as the source of truth
      until then.
