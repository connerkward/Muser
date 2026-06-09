# Muser — open items

Live ideas to revisit, not yet scheduled. Promote to issues / commits when picked up.

## Search quality

- **Hybrid search algorithm** — empirical comparison of:
  - SigLIP image embeddings only (current baseline)
  - + Florence-2 / JoyCaption text caption text-search (BM25 or vector-on-text)
  - + aesthetic-blended re-ranking (`final = α·cosine + (1−α)·aesthetic`, α slider)
  - + all three fused
  Need a small ground-truth set (the existing `eval/` framework can score retrieval quality across candidates). The point isn't to ship one algorithm — it's to know which combination wins on this corpus.
- **Aesthetic-blended search re-ranking** (option C from session 2026-06-03) — blend the aesthetic score into `/api/search` ranking so the Search tab itself surfaces beautiful matches first, not just highest-cosine. Currently aesthetic ranking only lives in the Interesting tab.
- **CIR (Composed Image Retrieval)** — level 1 (vector arithmetic `q = α·image_vec + β·text_vec`) is ~30 LOC, ~1 hr. Worth shipping as an experimental knob before committing to level 3 (MagicLens, Zhang et al. ICML 2024) which is a separate encoder + index table.
- **Caption-as-text search index** — index the captions (when present) into a separate text-search column; queries match either the image vector OR the caption tokens. Helps long-tail queries that SigLIP gets wrong but the caption nails.
- **`-word` parser** — sugar that auto-routes `-cute` in the main query box to the `neg=` parameter so users get expected Google-style negation. The dedicated Exclude field still exists; the parser is a convenience layer.

## Workflow

- **Caption-on-demand for cart items** — ✅ shipped 2026-06-03 (GPT-4o-mini via `/api/caption-bulk`; "Caption missing (N) →" button in the cart modal; busy overlay shows progress).
- **Checkout cost / time estimate in cart modal** — show estimated $ and wall-clock per captioning backend (GPT-4o-mini / JoyCaption Beta One) + per fal LoRA training endpoint (Flux fast / Flux general / SDXL fast), with running total that updates as cart size changes. ~30 LOC, ship after the JoyCaption agent finishes its cart-modal edits.
- **Caption quality bake-off** — GPT-4o-mini vs JoyCaption Beta One × LoRA quality. Reduced scope (~$8, ~3 hrs): one curated 30-image cluster, caption both ways, train 2 Flux LoRAs on fal, 10 test prompts each, manual eval. Full grid (~$30–60): {GPT, Joy} × {Flux fast, SDXL fast} × {subject, style} = 8 LoRAs. Defer until both captioners are stable.
- **Daemonize `muser serve`** — LaunchDaemon mirroring the Caddy pattern documented in `central/skills/machines/personal-machines/references/per_lappy_heavy.md`. Currently service requires manual `uv run muser serve` after reboot.
- **Cart count in mini bar** — the mini search sliver doesn't surface cart status; should mirror the header's `Cart (N)` link.

## Scoring / models

- **Newer aesthetic models** beyond the four already installed (`aesthetic_v2`, `pickscore`, `aesthetic_v25`, `hps_v21`): Q-Align (LLM-based multi-aspect), ImageReward (alignment + aesthetic + harmless), MPS (multi-dimensional preference). Add if the existing four don't span enough of the user's taste.
- **Personal aesthetic model** — pairwise Bradley-Terry with active learning on top of frozen SigLIP embeddings. ~100 labeled pairs from a small in-cart compare UI. Cost: ~10 min of labeling, ~80 LOC.
- **VLM-generated ground truth on z-to-sort** — REQUIREMENTS' "primary near-term deliverable." Replace BLIP-large captions in the domain eval with Qwen2.5-VL or Florence-2, regenerate clusters. Closes the eval loop on the user's actual corpus.

## LoRA pipeline

- **Captioner upgrade path** — current captioner is OpenAI GPT-4o-mini (single-sentence LoRA-shaped output, ~$0.001-0.005/img). Consider JoyCaption-Alpha-Two's "Training Prompt" mode (local, purpose-built for LoRA captions) or GPT-4o-full when going from "pipeline works" to "LoRA quality matters."
- **Caption post-process** — for style LoRAs, strip style-describing phrases from generated captions so the LoRA binds style to the trigger, not to explicit text. For subject LoRAs, leave style words in.
- **End-to-end Muser → fal automation** — once cart-with-captions zip exports cleanly, wire `muser train-lora <zip> --base flux-fast` to call the fal MCP and surface the resulting `.safetensors` URL + a few sample generations for human verification.

## UI / navigation

- **Improve the cluster visualization** — the Explore 3D point cloud (`projection.py` + the Explore tab) reads as a faint scatter; make the clustering legible: better point sizing/opacity/depth cues, stronger cluster color separation, hover/label affordances, and a tighter default camera so the structure is obvious at a glance instead of a haze of dots.
- **Sane routing — logical URLs + working back button** — give each surface a real, shareable location (`/#/search?q=…&folder=…`, `/#/explore`, `/#/jobs`, a result/detail route) so state lives in the URL, and make the browser **Back/Forward** button do what the user expects (return to the previous view/query, close an expanded panel, etc.) rather than dumping out of the app. Audit the current hash-router for dead-ends.
- **Update the Jobs page** (`http://muser.local/#/jobs`) — revisit the Jobs view: what it shows, layout, and live-status polling. (Scope TBD — define the concrete changes before picking up.)

## MCP ext-app (gallery)

- **Tool result exceeds 1 MB** — `search_images` returns "Tool result is too large. Maximum size is 1MB." The contact-sheet image + metadata block blows the MCP result cap. Fix: shrink the payload — fewer/smaller inline images, drop the full contact sheet, or return references instead of bytes.
- **Drop the blend slider in the ext-app** — remove the per-result aesthetic blend slider from the MCP gallery UI; use a single sensible default search-algorithm mix instead. (Slider stays in the full web UI.)
- **"Open in Muser" button** — add a button in the ext-app that opens `http://muser.local` (the full web UI) in the browser.
- **Sidebar instead of fullscreen** — change the expand/fullscreen mode to a sidebar display mode if the host supports it (`pip`/side-panel), rather than taking over the whole frame.
- **Return only 3–4 results** — cap the MCP gallery to the top 3–4 matches (keeps the payload small and the inline card compact; pairs with the 1 MB fix above).

## Polish

- **Negative-prompt α slider** — ✅ shipped 2026-06-03.
- **Cart persistence** — ✅ shipped 2026-06-03 (localStorage instead of sessionStorage).
- **Unique IDs per image** — ✅ in flight (uid = blake2b(path) first 12 hex chars).
