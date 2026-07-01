# Ultra-wide album-art outpainting — ComfyUI workflow candidates

Read-only investigation (2026-07-01). Goal: find the real production ComfyUI workflow behind
the ultra-wide album-cover outpainting pipeline (SAM+GroundingDINO text removal → 4× UltraSharp
→ "BigLaMa latent smudge" → slight blur → Stable-Diffusion ultra-wide outpaint → driver-distraction
verifier → ControlNet). No workflow was run or modified.

## TL;DR — the answer

**THE production workflow exists only as PNG-embedded metadata, not as a standalone `.json` on disk.**
Every `ComfyUI-BigLama-*.png` intermediate (543 of them under `ideas-comfy-shared/output/`) carries the
full ComfyUI graph in its `workflow` PNG chunk (~360 KB) plus the API `prompt` (~94 KB). The final-stage
output PNGs in that same tree also still carry metadata; only the *exported / portfolio* copies were
stripped. A representative extraction was saved to the session scratchpad
(`scratchpad/biglama_workflow.json`, `scratchpad/biglama_prompt.json`) — re-extractable any time from any
BigLama PNG.

There is **no** hand-saved `.json` of this graph anywhere on the system (searched all of `~/ideas-syncthing`,
`~/Desktop`, `~/Documents`, `~/Downloads`, and Google Drive `My Drive` + `Other computers`). The unique node
signatures `INPAINT_InpaintWithModel` / `IMIC TrafficLight` / `IMIC Distractive` returned zero `.json` hits.

## Ranked candidates

### 1. THE ONE — embedded workflow in `ComfyUI-BigLama-*.png` (PNG metadata)
- **Representative source:** [`ComfyUI-BigLama-_00001_.png`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/My%20Drive/ideas-comfy-shared/output/old/top10/folder/ComfyUI-BigLama-_00001_.png) (3072×504 ≈ 6.1:1 ultra-wide)
- **All BigLama PNGs:** [`ideas-comfy-shared/output/`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/My%20Drive/ideas-comfy-shared/output/) (543 files; e.g. `identified/6_6_tame_impala_the_less_i_know_the_better/old/`, `identified/Ordinary_-_Wedding_Version/old/`)
- **Extracted copy (this session):** [`scratchpad/biglama_workflow.json`](file:///private/tmp/claude-501/-Users-conner-dev-Muser/80cc5b5b-ac24-4490-b189-17d32ba3ddd5/scratchpad/biglama_workflow.json)
- **Size:** ~580–630 nodes (varies per render — it's a live, evolving dev graph re-saved with each run: 582 / 561 / 629 across three sampled PNGs).

This is a single large **multi-branch** production graph, not a clean linear pipeline. Key nodes (widget values):

| Stage | Node type | Key values |
|---|---|---|
| **BigLaMa (latent smudge)** | `INPAINT_LoadInpaintModel` → `INPAINT_InpaintWithModel` | model = **`big-lama.pt`** |
| **Slight blur** | `INPAINT_MaskedBlur` | `[100, 0]`; plus `Blur` (×9), `ImageBlend`, `Image Blend by Mask` |
| **Smudge noise source** | `Image Power Noise` | `512×512`, blue noise, 0.5 |
| **Asymmetric outpaint pad** | `ImagePadForOutpaint` | **`[left=1680, top=104, right=1152, bottom=160, feather=100]`** — left ≈ 1.46× right (per-source: also seen `[1800,112,1232,168,100]`) |
| **SD checkpoint (primary)** | `CheckpointLoaderSimple` | **`realvisxlV50_v30InpaintBakedvae.safetensors`** (RealVisXL 5.0, SDXL inpaint) |
| **SDXL-inpaint alt path** | `UNETLoader` + `DualCLIPLoader` + `VAELoader` | `sdxl-inpaint.safetensors` / `clip-sdxl-inpaint*` / `sdxl-inpaint-vae.safetensors` |
| **Flux-fill alt path** | `UNETLoader` + `DualCLIPLoader` | `flux1-fill-dev.safetensors` + `t5xxl_fp16` / `clip_l`; `FluxGuidance` 0.5 & 3 |
| **LoRA** | `LoraLoader` | `lora-JuggerCineXL2.safetensors` |
| **ControlNet (limit detail)** | `ControlNetLoader` ×2 → `ControlNetApplyAdvanced` ×2 | **`sdxl-controlnet-tile.safetensors`** (tile) |
| **Diffusion** | `KSampler` ×4 | SDXL: 25 steps, cfg 4, `euler_ancestral`/`karras`; Flux: 20 steps, cfg 0.5 |
| **Diff-diffusion** | `DifferentialDiffusion`, `InpaintModelConditioning`, `INPAINT_VAEEncodeInpaintConditioning` | — |
| **Driver-distraction verifier** | `IMIC TrafficLight`, `IMIC Distractive Area Percentage`, `IMIC Show Percentage`, `IMIC Entropy`, `IMIC Illumination`, `IMIC Edge Ratio` | go/no-go heuristic gate |
| **Prompt gen** | `OpenAIAPI` ×2 | positive/negative prompt authoring |
| **Output branches** | `SaveImage` ×12 | `ComfyUI-SDXL-Tile-Simple-`, `-Tile-Mid-`, `-SDXL-`, `-Final-BakedIC-`, `-Final-Blur`, `-Final-DDPass-`, `-Final-DDPass-Alt-`, `-Final-UIOverlay`, `-BigLama-`, `-Flux-` |

**Match flags:** LaMa/big-lama ✅ · `ImagePadForOutpaint` asymmetric ✅ · SD/SDXL checkpoint ✅ · ControlNet (tile) ✅ ·
driver-distraction verifier ✅ (the IMIC nodes) · blur ✅ · Flux-fill alt branch ✅ ·
SAM/GroundingDINO ❌ (done upstream — see #4) · UltraSharp upscaler ❌ (done upstream — see note).

Positive prompt (excerpt): *"A surreal, opulent dreamscape unfolds from a warm, golden-orange core… butterflies…
subtly mirrored in the outpainting as ethereal silhouettes…"* Negative prompt heavily loaded against text/logos/
people/limbs/interiors/bright-neon — consistent with the driver-distraction constraint.

> **Why this is THE one:** it is the only graph that simultaneously contains `big-lama.pt` inpainting, the
> asymmetric `ImagePadForOutpaint`, an SDXL (RealVisXL) inpaint checkpoint, tile-ControlNet, the blur nodes, AND
> the IMIC driver-distraction metric gate — every element of the user's description that belongs to the *outpaint*
> half of the pipeline. Its outputs are literally named `ComfyUI-BigLama-*`. The `.jpg`/PNG dimensions (3072×504,
> 3288×536) are the ~6:1 ultra-wide target.

### 2. `2-outpainting-flux-fill*.json` — crashcourse Flux-fill (NOT production)
- **Path:** [`comfyui-crashcourse-mar5/backups/workflows/2-outpainting-flux-fill.json`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/Other%20computers/My%20Mac/comfyui-crashcourse-mar5/backups/workflows/2-outpainting-flux-fill.json) (+ `_mod3/4/5` variants across `pack1-cloud`, `pack2-8-12gb`, `pack3-12gb+`)
- **14 nodes:** `LoadImage` → `ImagePadForOutpaint` → `InpaintModelConditioning` → `DifferentialDiffusion` → `UnetLoaderGGUF` (Flux fill GGUF) + `DualCLIPLoader` + `VAELoader` → `KSampler` → `VAEDecode` → `SaveImage`.
- **Match flags:** `ImagePadForOutpaint` ✅ · Flux-fill ✅ · LaMa ❌ · SAM/GroundingDINO ❌ · SDXL ❌ · ControlNet ❌ · IMIC verifier ❌.
- **Verdict:** the *seed/reference* the pipeline grew from (the Flux-fill alt branch in #1 descends from this), but it is a stock course template — not the production workflow.

### 3. `1-object-removal-sam2.json` — crashcourse SAM2 + LaMa removal template
- **Path:** [`comfyui-crashcourse-mar5/backups/workflows/1-object-removal-sam2.json`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/Other%20computers/My%20Mac/comfyui-crashcourse-mar5/backups/workflows/1-object-removal-sam2.json)
- **5 nodes:** `LoadImage` → `SAM2Segment` → `AILab_LamaRemover` → `PreviewImage` / `SaveImage`.
- **Match flags:** SAM (SAM2) ✅ · LaMa (`AILab_LamaRemover`, ComfyUI-RMBG) ✅ · GroundingDINO ❌ · outpaint pad ❌ · SDXL ❌.
- **Verdict:** the *removal-step* reference template; shows where the LaMa/SAM idea came from. Not the outpaint graph. (Note: the LaMa **model** here, `big-lama.pt`, is the same one #1 uses.)

### 4. `aix_text_removed*` — the real text-removal pass (Python, not ComfyUI)
- **Path:** [`ideas-comfy-shared/output/aix_text_removed_full/`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/My%20Drive/ideas-comfy-shared/output/aix_text_removed_full/) — `workflow.json` (name/description stub only) + `processing_results.json` (per-image input→`*_outpaint_ready.jpg`, `detected_regions`, status).
- GroundingDINO configs present at [`ideas-comfy-shared/models/grounding-dino/`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/My%20Drive/ideas-comfy-shared/models/grounding-dino/) (`GroundingDINO_SwinB.cfg.py`, `GroundingDINO_SwinT_OGC.cfg.py`).
- **Verdict:** the **SAM + GroundingDINO text/logo removal** stage the user described was a **standalone batch Python pipeline** (outputs `*_outpaint_ready.jpg`), run *before* the ComfyUI graph — not a node inside it. That's why the ComfyUI graph (#1) has no SAM/GroundingDINO nodes. The driving `.py` script was not located on this machine (only its outputs + the DINO model configs remain).

> **UltraSharp 4× upscale** likewise is not a node in graph #1 — no `UpscaleModelLoader`/`ImageUpscaleWithModel`/
> UltraSharp reference. It was a separate upstream stage (same pattern as text removal).

## Visual previews

A true node-graph *render* requires the ComfyUI GUI (load `scratchpad/biglama_workflow.json` — or drag any
`ComfyUI-BigLama-*.png` — into ComfyUI); that was **not** driven here (read-only, no execution). What IS directly
viewable without ComfyUI:
- The **BigLaMa smudge stage output itself** is the PNG the workflow is embedded in — e.g.
  [`ComfyUI-BigLama-_00001_.png`](file:///Users/conner/Library/CloudStorage/GoogleDrive-conner.k.ward@gmail.com/My%20Drive/ideas-comfy-shared/output/old/top10/folder/ComfyUI-BigLama-_00001_.png) — so the smudge/blur look is inspectable directly.
- Final-branch outputs (still with metadata) live alongside, e.g. `ComfyUI-SDXL-Tile-Mid-*.png`, `ComfyUI-Final-Mid-Blur*.png`.

No standalone thumbnail/preview image is referenced *inside* the workflow JSON (previews are live `PreviewImage`
nodes, not on-disk files).

## How to recover the workflow into ComfyUI
Drag any `ComfyUI-BigLama-*.png` onto the ComfyUI canvas (it reads the embedded `workflow` chunk), or load the
extracted `biglama_workflow.json`. To re-extract from a PNG headlessly:
`PIL.Image.open(png).info["workflow"]` (graph) / `["prompt"]` (API format).
