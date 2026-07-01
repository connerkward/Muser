# Muser — Decisions (ADR log)

Newest first. Real forks only — why one path over another. Format: Michael Nygard ADR.

## 2026-07-01 — NSFW as a grey-out FILTER, not a blend weight or a hide
Context: the outpaintings blend bar is a positive-only allocation, so weighting an NSFW
model *up* surfaces NSFW — backwards for curation; a binary hide is too blunt and makes
covers vanish silently.
Decision: a second "Grey out" bar — toggle NSFW classifiers + a threshold; flagged covers
grey out **in place** (dim + red ✕), ranking untouched.
Consequence: soft, reversible, per-model visual filter; nothing disappears. Supersedes the
brief per-model positive-weight chips (2026-07-01, same day) which were removed.

## 2026-07-01 — Album region grouping over filename numbers
Context: outpaintings need de-duping by cover. Filenames carry an album number for ~51%
of files; the other ~45% (`ComfyUI-*`) have none, and a fixed geometric mask fails (the
cover is offset-right and varies in size/style).
Decision: derive the fixed album-art box from cross-variant pixel variance (confirmed by
the outpaint pad), crop to it, embed the region, cluster by region cosine ≥ 0.92.
Consequence: groups covers filenames can't (nameless variants included); one facet drives
the unique spread + click-to-cluster. Filename numbers dropped as the grouping key.
