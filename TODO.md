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

- **Caption-on-demand for cart items** — small "Caption missing" button in the cart modal that subprocess-calls `muser caption --paths …` for missing items. Avoids the 12-hour full caption pass (Florence-2 on MPS is slow). ~80 LOC.
- **Daemonize `muser serve`** — LaunchDaemon mirroring the Caddy pattern documented in `central/skills/machines/personal-machines/references/per_lappy_heavy.md`. Currently service requires manual `uv run muser serve` after reboot.
- **Cart count in mini bar** — the mini search sliver doesn't surface cart status; should mirror the header's `Cart (N)` link.

## Scoring / models

- **Newer aesthetic models** beyond the four already installed (`aesthetic_v2`, `pickscore`, `aesthetic_v25`, `hps_v21`): Q-Align (LLM-based multi-aspect), ImageReward (alignment + aesthetic + harmless), MPS (multi-dimensional preference). Add if the existing four don't span enough of the user's taste.
- **Personal aesthetic model** — pairwise Bradley-Terry with active learning on top of frozen SigLIP embeddings. ~100 labeled pairs from a small in-cart compare UI. Cost: ~10 min of labeling, ~80 LOC.
- **VLM-generated ground truth on z-to-sort** — REQUIREMENTS' "primary near-term deliverable." Replace BLIP-large captions in the domain eval with Qwen2.5-VL or Florence-2, regenerate clusters. Closes the eval loop on the user's actual corpus.

## LoRA pipeline

- **Captioner upgrade path** — Florence-2 is fast-but-coarse for LoRA prompts. Consider JoyCaption-Alpha-Two's "Training Prompt" mode (slower, purpose-built for LoRA captions) when going from "pipeline works" to "LoRA quality matters."
- **Caption post-process** — for style LoRAs, strip style-describing phrases from generated captions so the LoRA binds style to the trigger, not to explicit text. For subject LoRAs, leave style words in.
- **End-to-end Muser → fal automation** — once cart-with-captions zip exports cleanly, wire `muser train-lora <zip> --base flux-fast` to call the fal MCP and surface the resulting `.safetensors` URL + a few sample generations for human verification.

## Polish

- **Negative-prompt α slider** — ✅ shipped 2026-06-03.
- **Cart persistence** — ✅ shipped 2026-06-03 (localStorage instead of sessionStorage).
- **Unique IDs per image** — ✅ in flight (uid = blake2b(path) first 12 hex chars).
