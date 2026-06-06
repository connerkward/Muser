// Shared Muser gallery: renders search hits into the card grid and wires the
// three search modes (text / color / reverse-image), the folder-scope box,
// and the duplicates modal. Knows nothing about MCP — callers provide handlers.
// Used by both `search-app.ts` (live MCP host) and `preview.ts` (mock preview).

import "./gallery.css";

export interface Hit {
  path: string;
  score: number;
  /** SigLIP-calibrated match probability (0–1) when the model provides one; the
   *  raw cosine `score` is flat/uncalibrated, so prefer this for the displayed %. */
  prob?: number;
  thumb: string | null;
  /** Large whole-image preview (base64) for the fullscreen lightbox, inlined by
   *  the server because the sandboxed iframe can't fetch /api/image. Falls back
   *  to `thumb` when absent (e.g. the mock preview). */
  full?: string | null;
  /** All paths that are this same image (includes `path`). Length 1 when unique. */
  dupes?: string[];
  dupe_count?: number;
  /** base64 thumb per dupe path, supplied inline by the server (the sandboxed
   *  iframe can't fetch /api/thumb). Falls back to the card thumb if absent. */
  dupeThumbs?: Record<string, string | null>;
  // ---- Per-result aesthetic scores (0–1), inlined by /api/search ----------
  // Used by the blend control to re-rank: final = Σ wᵢ·(hit[field] ?? 0).
  // Present on text/embedding hits; absent (→ treated as 0) on color/reverse.
  aesthetic_v2?: number;
  pickscore?: number;
  aesthetic_v25?: number;
  hps_v21?: number;
  aesthetic?: number;
}
export interface Payload {
  /** Which search produced these results — selects the active tab on render. */
  mode?: "text" | "color" | "image";
  folder: string;
  query: string;
  results: Hit[];
}

export interface ColorSearch {
  /** Comma-separated #rrggbb colors. */
  hex: string;
  mode: "all" | "any";
  folder?: string;
}
export interface ImageSearch {
  /** base64 image bytes (no data: prefix). */
  base64: string;
  filename: string;
  folder?: string;
}

export interface GalleryHandlers {
  onSearchText: (query: string, folder?: string) => void | Promise<void>;
  onSearchColor?: (req: ColorSearch) => void | Promise<void>;
  onSearchImage?: (req: ImageSearch) => void | Promise<void>;
  /** Fetch indexed folders to populate the scope datalist (live app only). */
  onLoadFolders?: () => Promise<string[]>;
  onSelect?: (hit: Hit) => void;
  /** Reveal a duplicate file in Finder. The sandboxed iframe can't call
   *  /api/reveal, so the live app routes this to the host as a chat message. */
  onReveal?: (path: string) => void;
  /** Toggle the *whole app* into the host's fullscreen display mode (MCP
   *  `requestDisplayMode`). Returns true if the host honoured it; when absent or
   *  false, the gallery falls back to a CSS maximized overlay. */
  onToggleFullscreen?: () => boolean | Promise<boolean>;
}

export interface Gallery {
  render: (payload: Payload) => void;
  setStatus: (text: string) => void;
  setBusy: (busy: boolean) => void;
  /** Re-sync the fullscreen button with the host's current display mode. */
  syncFullscreen: () => void;
  readonly folder: string;
}

type Mode = "text" | "color" | "image";

// Palette presets mirror the main web UI's color tab swatches.
const COLOR_PRESETS = [
  "#c83c3c", "#e08a2a", "#e8d44a", "#4a9e4a", "#3a7bd5",
  "#7a4ad0", "#d04a9e", "#1a1a1a", "#ffffff",
];

// ---- Aesthetic-blend re-rank model -----------------------------------------
// The blend re-orders the CURRENT result set client-side by a linear sum:
//   final = Σ wᵢ · (hit[field] ?? 0)
// where the weights are *shares* that sum to 1 over the enabled metrics. The
// per-result aesthetic scores are inlined by /api/search (already 0–1 normalized
// & comparable), so a plain weighted sum is exactly right — same math the main
// web UI uses. "Relevance" is the embedding cosine (`score`). Default is
// Relevance 100% → pure embedding order, identical to today.
interface BlendMetric {
  id: string;
  /** Key on a Hit (and the localStorage share map). */
  field: keyof Hit;
  label: string;
  color: string;
  hint: string;
}
const BLEND_METRICS: readonly BlendMetric[] = [
  { id: "vec", field: "score", label: "Relevance", color: "#5b6b86", hint: "Vector / embedding match — the primary signal." },
  { id: "aes", field: "aesthetic_v2", label: "Aesthetic V2", color: "#a9823f", hint: "LAION Aesthetics V2 — how 'pretty' the image is." },
  { id: "ps", field: "pickscore", label: "PickScore", color: "#9c5f74", hint: "PickScore — human-preference proxy (Pick-a-Pic)." },
  { id: "aes25", field: "aesthetic_v25", label: "Aesthetic V2.5", color: "#5f7d5e", hint: "LAION Aesthetics V2.5 — refined successor to V2." },
  { id: "hps", field: "hps_v21", label: "HPS v2.1", color: "#736a8c", hint: "Human Preference Score — prompt-aware preference." },
  { id: "aes0", field: "aesthetic", label: "Zero-shot", color: "#5a8487", hint: "Zero-shot CLIP aesthetic prompt — fast but weak." },
];
const BLEND_KEY = "muser.mcp.blend";
interface BlendState {
  shares: Record<string, number>;
  enabled: Record<string, boolean>;
}
// Normalize a {id:weight} map to shares summing to 1 over the *enabled* ids.
// Disabled ids → 0. If every enabled weight is 0, spread evenly so the bar is
// never empty.
function normShares(weights: Record<string, number>, enabled: Record<string, boolean>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const m of BLEND_METRICS) out[m.id] = 0;
  const on = BLEND_METRICS.filter((m) => enabled[m.id]);
  if (!on.length) return out;
  const tot = on.reduce((s, m) => s + Math.max(0, weights[m.id] ?? 0), 0);
  if (tot <= 1e-9) {
    for (const m of on) out[m.id] = 1 / on.length;
    return out;
  }
  for (const m of on) out[m.id] = Math.max(0, weights[m.id] ?? 0) / tot;
  return out;
}
function defaultBlend(): BlendState {
  // Relevance 100%; aesthetic metrics enabled at 0 so toggling them on is one
  // click. Pure-embedding order out of the box (unchanged from today).
  const enabled: Record<string, boolean> = {};
  const shares: Record<string, number> = {};
  for (const m of BLEND_METRICS) {
    enabled[m.id] = m.id === "vec";
    shares[m.id] = m.id === "vec" ? 1 : 0;
  }
  return { shares, enabled };
}
function loadBlend(): BlendState {
  let s: Partial<BlendState> | null = null;
  try {
    s = JSON.parse(localStorage.getItem(BLEND_KEY) ?? "null");
  } catch {
    s = null;
  }
  if (!s || !s.shares || !s.enabled) return defaultBlend();
  const def = defaultBlend();
  const enabled: Record<string, boolean> = {};
  const shares: Record<string, number> = {};
  for (const m of BLEND_METRICS) {
    enabled[m.id] = s.enabled[m.id] != null ? !!s.enabled[m.id] : def.enabled[m.id]!;
    shares[m.id] = s.shares[m.id] != null ? +s.shares[m.id]! : 0;
  }
  return { shares: normShares(shares, enabled), enabled };
}

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

export function createGallery(handlers: GalleryHandlers): Gallery {
  const grid = $<HTMLDivElement>("grid");
  const status = $<HTMLDivElement>("status");
  const scope = $<HTMLInputElement>("scope");

  // ---- Mode tabs -----------------------------------------------------------
  const tabButtons = Array.from(document.querySelectorAll<HTMLButtonElement>(".nav[data-mode]"));
  const panels = Array.from(document.querySelectorAll<HTMLElement>(".panel[data-panel]"));
  function setMode(mode: Mode): void {
    for (const b of tabButtons) b.classList.toggle("on", b.dataset.mode === mode);
    for (const p of panels) p.hidden = p.dataset.panel !== mode;
    colorSetRow.hidden = mode !== "color" || colorSet.length === 0;
  }
  for (const b of tabButtons) {
    b.addEventListener("click", () => setMode(b.dataset.mode as Mode));
  }

  const folderOrUndef = () => {
    const v = scope?.value.trim();
    return v ? v : undefined;
  };

  // ---- Text search ---------------------------------------------------------
  const form = $<HTMLFormElement>("form");
  const input = $<HTMLInputElement>("q");
  const goButton = $<HTMLButtonElement>("go");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (query) void handlers.onSearchText(query, folderOrUndef());
  });

  // ---- Color search --------------------------------------------------------
  const colorPicker = $<HTMLInputElement>("colorPicker");
  const colorPresets = $<HTMLSpanElement>("colorPresets");
  const colorSetRow = $<HTMLDivElement>("colorSet");
  let colorSet: string[] = [];

  const normHex = (h: string) => "#" + h.replace(/^#/, "").toLowerCase();

  function renderColorSet(): void {
    colorSetRow.replaceChildren();
    for (const hex of colorSet) {
      const chip = document.createElement("button");
      chip.className = "cschip";
      chip.title = `Remove ${hex}`;
      chip.type = "button";
      const sw = Object.assign(document.createElement("span"), { className: "sw" });
      sw.style.background = hex;
      chip.append(
        sw,
        Object.assign(document.createElement("span"), { textContent: hex }),
        Object.assign(document.createElement("span"), { className: "rm", textContent: "×" }),
      );
      chip.addEventListener("click", () => {
        colorSet = colorSet.filter((c) => c !== hex);
        renderColorSet();
        if (colorSet.length) runColor();
      });
      colorSetRow.append(chip);
    }
    colorSetRow.hidden = colorSet.length === 0;
  }
  function addColor(hex: string): void {
    const h = normHex(hex);
    if (!colorSet.includes(h)) colorSet.push(h);
    renderColorSet();
    runColor();
  }
  function runColor(): void {
    if (!handlers.onSearchColor) return;
    const hexes = colorSet.length ? colorSet.slice() : [normHex(colorPicker.value)];
    void handlers.onSearchColor({ hex: hexes.join(","), mode: "all", folder: folderOrUndef() });
  }
  for (const hex of COLOR_PRESETS) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "preset";
    b.title = "Add " + hex;
    b.style.background = hex;
    b.addEventListener("click", () => addColor(hex));
    colorPresets.append(b);
  }
  colorPicker.addEventListener("change", () => addColor(colorPicker.value));
  $<HTMLButtonElement>("colorAdd").addEventListener("click", () => addColor(colorPicker.value));
  $<HTMLButtonElement>("colorGo").addEventListener("click", () => {
    if (!colorSet.length) addColor(colorPicker.value);
    else runColor();
  });
  $<HTMLButtonElement>("colorClear").addEventListener("click", () => {
    colorSet = [];
    renderColorSet();
    grid.replaceChildren();
    setStatus("Pick a color or a swatch to search.");
  });

  // ---- Reverse-image search (drop or pick an image → similar library hits) --
  // Underlying call is POST /api/search-upload (embedding kNN over the index):
  // "find images in MY library that look like this one." Two entry points share
  // one helper — the file picker in the panel, and drag-and-drop onto the gallery.
  const imgFile = $<HTMLInputElement>("imgFile");
  const imgName = $<HTMLSpanElement>("imgName");

  /** Read an image File, base64-encode it, and run the reverse-image search. */
  function reverseSearchFile(file: File): void {
    if (!handlers.onSearchImage) return;
    // A drop can land while another tab is active — make the result legible by
    // switching to the reverse-image tab first.
    setMode("image");
    imgName.textContent = `Finding images similar to ${file.name}…`;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.includes(",") ? result.split(",", 2)[1]! : result;
      void handlers.onSearchImage!({ base64, filename: file.name, folder: folderOrUndef() });
    };
    reader.readAsDataURL(file);
  }

  imgFile.addEventListener("change", () => {
    const file = imgFile.files?.[0];
    if (file) reverseSearchFile(file);
  });

  // ---- Drag-and-drop reverse-image search ----------------------------------
  // Dropping an image file anywhere on the gallery triggers a reverse-image
  // search — no need to be on the tab first. A full-window overlay highlights
  // the drop target while a file is dragged over.
  const dropzone = $<HTMLDivElement>("dropzone");
  let dragDepth = 0; // dragenter/leave fire per child; count to avoid flicker.

  const hasImage = (dt: DataTransfer | null): boolean =>
    !!dt && (Array.from(dt.items).some((i) => i.kind === "file" && i.type.startsWith("image/"))
      || Array.from(dt.types).includes("Files"));

  if (handlers.onSearchImage) {
    window.addEventListener("dragenter", (e) => {
      if (!hasImage(e.dataTransfer)) return;
      e.preventDefault();
      dragDepth++;
      dropzone?.classList.add("over");
    });
    window.addEventListener("dragover", (e) => {
      if (!hasImage(e.dataTransfer)) return;
      e.preventDefault();
      if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
    });
    window.addEventListener("dragleave", (e) => {
      if (!hasImage(e.dataTransfer)) return;
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) dropzone?.classList.remove("over");
    });
    window.addEventListener("drop", (e) => {
      dragDepth = 0;
      dropzone?.classList.remove("over");
      const file = Array.from(e.dataTransfer?.files ?? []).find((f) => f.type.startsWith("image/"));
      if (!file) return;
      e.preventDefault();
      reverseSearchFile(file);
    });
  }

  // ---- Folder scope datalist (live app) ------------------------------------
  if (handlers.onLoadFolders) {
    void handlers.onLoadFolders().then((dirs) => {
      const dl = $<HTMLDataListElement>("folders");
      dl.replaceChildren();
      for (const d of dirs) {
        dl.append(Object.assign(document.createElement("option"), { value: d }));
      }
    }).catch(() => {});
  }

  // ---- Duplicates modal ----------------------------------------------------
  const modal = $<HTMLDivElement>("dupes");
  const modalTitle = $<HTMLHeadingElement>("dupesTitle");
  const modalSub = $<HTMLDivElement>("dupesSub");
  const modalFiles = $<HTMLDivElement>("dupesFiles");
  const modalClose = $<HTMLSpanElement>("dupesClose");

  const closeDupes = () => modal?.classList.remove("show");
  modalClose?.addEventListener("click", closeDupes);
  modal?.addEventListener("click", (e) => {
    if (e.target === modal) closeDupes();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDupes();
  });

  function openDupes(hit: Hit, name: string): void {
    if (!modal) return;
    const dupes = hit.dupes ?? [hit.path];
    modalTitle.textContent = name;
    modalSub.textContent = `${dupes.length} files are this same image`;
    modalFiles.replaceChildren();
    for (const p of dupes) {
      const file = document.createElement("div");
      file.className = "file";
      const src = hit.dupeThumbs?.[p] ?? hit.thumb;
      const img = src
        ? Object.assign(document.createElement("img"), { src, alt: p })
        : Object.assign(document.createElement("div"), { className: "noimg", textContent: "no preview" });
      const fn = Object.assign(document.createElement("div"), { className: "fn", textContent: p });
      fn.title = p;
      file.append(img, fn);
      if (handlers.onReveal) {
        const reveal = Object.assign(document.createElement("button"), { textContent: "Reveal", type: "button" });
        reveal.addEventListener("click", () => handlers.onReveal!(p));
        file.append(reveal);
      }
      modalFiles.append(file);
    }
    modal.classList.add("show");
  }

  // ---- App fullscreen toggle (whole gallery) -------------------------------
  // Prefer the host's real fullscreen display mode; if the host declines or
  // isn't connected (standalone preview), fall back to a CSS maximized layout
  // by toggling `data-app-fs` on <html>.
  const fsToggle = $<HTMLButtonElement>("fsToggle");
  function appIsFullscreen(): boolean {
    return document.documentElement.dataset.appFs === "1"
      || document.documentElement.dataset.mcpDisplay === "fullscreen";
  }
  function setCssFullscreen(on: boolean): void {
    document.documentElement.dataset.appFs = on ? "1" : "0";
    syncFsButton();
  }
  function syncFsButton(): void {
    if (!fsToggle) return;
    const on = appIsFullscreen();
    fsToggle.classList.toggle("on", on);
    fsToggle.title = on ? "Exit fullscreen (Esc)" : "Fullscreen";
    fsToggle.setAttribute("aria-pressed", String(on));
  }
  async function toggleAppFullscreen(): Promise<void> {
    const want = !appIsFullscreen();
    let hostHandled = false;
    if (handlers.onToggleFullscreen) {
      try {
        hostHandled = await handlers.onToggleFullscreen();
      } catch {
        hostHandled = false;
      }
    }
    // Host didn't take it (or no host) → CSS maximized fallback overlay.
    if (!hostHandled) setCssFullscreen(want);
    else syncFsButton();
  }
  fsToggle?.addEventListener("click", () => void toggleAppFullscreen());

  // ---- Image lightbox (whole image, not the cover-cropped thumb) -----------
  const lightbox = $<HTMLDivElement>("lightbox");
  const lightboxImg = $<HTMLImageElement>("lightboxImg");
  const lightboxCap = $<HTMLDivElement>("lightboxCap");
  const lightboxClose = $<HTMLSpanElement>("lightboxClose");

  const closeLightbox = () => lightbox?.classList.remove("show");
  function openLightbox(hit: Hit): void {
    if (!lightbox) return;
    const src = hit.full ?? hit.thumb;
    if (!src) return;
    lightboxImg.src = src;
    lightboxImg.alt = hit.path;
    if (lightboxCap) lightboxCap.textContent = hit.path.split("/").pop() ?? hit.path;
    lightbox.classList.add("show");
  }
  lightboxClose?.addEventListener("click", closeLightbox);
  lightbox?.addEventListener("click", (e) => {
    if (e.target === lightbox || e.target === lightboxImg) closeLightbox();
  });

  // One Esc handler closes whatever overlay is open, then exits CSS fullscreen.
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (lightbox?.classList.contains("show")) {
      closeLightbox();
    } else if (document.documentElement.dataset.appFs === "1") {
      setCssFullscreen(false);
    }
  });

  // ---- Status / busy / render ----------------------------------------------
  const setStatus = (text: string) => {
    status.textContent = text;
  };
  const setBusy = (busy: boolean) => {
    goButton.disabled = busy;
    $<HTMLButtonElement>("colorGo").disabled = busy;
  };

  let folder = "";

  // ---- Aesthetic-blend re-rank control -------------------------------------
  // A compact allocation bar (Relevance + 5 aesthetic models) that re-orders the
  // current results client-side. State persists to localStorage; the blend
  // re-applies whenever results change OR weights change. The grid is rebuilt
  // from `lastResults` (the raw service order) re-sorted by the blend key.
  // `#blend` may be absent in a minimal host HTML — degrade gracefully (no
  // control, results render in plain service order).
  const blendBox = document.getElementById("blend") as HTMLDivElement | null;
  let blend = loadBlend();
  let lastResults: Hit[] = [];

  const blendKey = (h: Hit): number =>
    BLEND_METRICS.reduce((sum, m) => {
      const w = blend.enabled[m.id] ? blend.shares[m.id] ?? 0 : 0;
      if (w <= 0) return sum;
      const v = h[m.field];
      return sum + w * (typeof v === "number" ? v : 0);
    }, 0);

  /** A stable, blend-ordered copy of the current results. Stable because we tag
   *  each hit with its original index and break ties on it — so equal-key hits
   *  (e.g. color results with no aesthetic fields) keep the service order. */
  function blendOrdered(): Hit[] {
    return lastResults
      .map((h, i) => ({ h, i, k: blendKey(h) }))
      .sort((a, b) => b.k - a.k || a.i - b.i)
      .map((x) => x.h);
  }

  function saveBlend(): void {
    try {
      localStorage.setItem(BLEND_KEY, JSON.stringify(blend));
    } catch {
      /* private mode / quota — non-fatal, just don't persist */
    }
  }

  /** True when the blend would change the pure-embedding order (anything other
   *  than Relevance-only carries weight). Drives a subtle "active" hint. */
  function blendActive(): boolean {
    return BLEND_METRICS.some(
      (m) => m.id !== "vec" && blend.enabled[m.id] && (blend.shares[m.id] ?? 0) > 1e-9,
    );
  }

  // Repaint segment widths + legend percentages from `blend`.
  function paintBlend(): void {
    const on = BLEND_METRICS.filter((m) => blend.enabled[m.id]);
    for (const m of BLEND_METRICS) {
      const seg = segEls.get(m.id);
      const pc = pcEls.get(m.id);
      const leg = legEls.get(m.id);
      const lv = legValEls.get(m.id);
      const share = blend.enabled[m.id] ? blend.shares[m.id] ?? 0 : 0;
      const pct = Math.round(share * 100);
      if (seg) {
        seg.style.flexBasis = share * 100 + "%";
        seg.style.display = blend.enabled[m.id] ? "" : "none";
        seg.classList.toggle("tiny", pct < 8);
        seg.title = `${m.label} · ${pct}% — ${m.hint}`;
      }
      if (pc) pc.textContent = pct + "%";
      if (leg) leg.classList.toggle("off", !blend.enabled[m.id]);
      if (lv) lv.textContent = blend.enabled[m.id] ? pct + "%" : "off";
    }
    blendBox?.classList.toggle("active", blendActive());
    if (resetBtn) resetBtn.disabled = !blendActive();
  }

  /** Set one metric's share (0–1), re-normalizing the OTHER enabled metrics to
   *  fill the remainder — so the bar always sums to 100%. Mirrors the web UI's
   *  allocation-drag semantics in a slider form. */
  function setShare(id: string, value: number): void {
    value = Math.max(0, Math.min(1, value));
    const others = BLEND_METRICS.map((m) => m.id).filter(
      (x) => x !== id && blend.enabled[x],
    );
    blend.shares[id] = value;
    const rest = 1 - value;
    const otherTot = others.reduce((s, x) => s + (blend.shares[x] ?? 0), 0);
    if (others.length) {
      if (otherTot <= 1e-9) {
        for (const x of others) blend.shares[x] = rest / others.length;
      } else {
        for (const x of others) blend.shares[x] = ((blend.shares[x] ?? 0) / otherTot) * rest;
      }
    }
    paintBlend();
    saveBlend();
    rerank();
  }

  function toggleMetric(id: string, enable: boolean): void {
    blend.enabled[id] = enable;
    blend.shares = normShares(blend.shares, blend.enabled);
    paintBlend();
    syncSliders();
    saveBlend();
    rerank();
  }

  function resetBlend(): void {
    blend = defaultBlend();
    paintBlend();
    syncSliders();
    saveBlend();
    rerank();
  }

  // Build the control once: a segmented bar + a legend row of toggle+slider per
  // metric. Element maps let paintBlend() mutate in place (no teardown).
  const segEls = new Map<string, HTMLElement>();
  const pcEls = new Map<string, HTMLElement>();
  const legEls = new Map<string, HTMLElement>();
  const legValEls = new Map<string, HTMLElement>();
  const sliderEls = new Map<string, HTMLInputElement>();
  let resetBtn: HTMLButtonElement | null = null;

  function syncSliders(): void {
    for (const m of BLEND_METRICS) {
      const sl = sliderEls.get(m.id);
      if (sl) sl.value = String(Math.round((blend.shares[m.id] ?? 0) * 100));
    }
  }

  function buildBlendControl(): void {
    if (!blendBox) return;
    blendBox.replaceChildren();

    const head = document.createElement("div");
    head.className = "blend-head";
    const title = Object.assign(document.createElement("span"), {
      className: "blend-title",
      textContent: "Blend",
      title:
        "Re-rank results by a weighted mix of relevance + aesthetic models. " +
        "Re-orders the current results only — it never changes which images matched.",
    });
    resetBtn = Object.assign(document.createElement("button"), {
      type: "button",
      className: "blend-reset ghost",
      textContent: "Reset",
      title: "Back to Relevance 100% (pure embedding order)",
    });
    resetBtn.addEventListener("click", resetBlend);
    head.append(title, resetBtn);

    const bar = document.createElement("div");
    bar.className = "blend-bar";
    for (const m of BLEND_METRICS) {
      const seg = document.createElement("div");
      seg.className = "blend-seg";
      seg.style.background = m.color;
      const pc = Object.assign(document.createElement("span"), { className: "blend-pc" });
      seg.append(pc);
      bar.append(seg);
      segEls.set(m.id, seg);
      pcEls.set(m.id, pc);
    }

    const legend = document.createElement("div");
    legend.className = "blend-legend";
    for (const m of BLEND_METRICS) {
      const row = document.createElement("label");
      row.className = "blend-leg";
      const cb = Object.assign(document.createElement("input"), { type: "checkbox" });
      cb.checked = !!blend.enabled[m.id];
      cb.addEventListener("change", () => toggleMetric(m.id, cb.checked));
      const dot = document.createElement("span");
      dot.className = "blend-dot";
      dot.style.background = m.color;
      const name = Object.assign(document.createElement("span"), {
        className: "blend-name",
        textContent: m.label,
      });
      const slider = Object.assign(document.createElement("input"), {
        type: "range",
        min: "0",
        max: "100",
        step: "1",
        className: "blend-slider",
      });
      slider.value = String(Math.round((blend.shares[m.id] ?? 0) * 100));
      slider.title = m.hint;
      slider.addEventListener("input", () => {
        // Dragging a disabled metric's slider implicitly enables it.
        if (!blend.enabled[m.id]) {
          cb.checked = true;
          blend.enabled[m.id] = true;
        }
        setShare(m.id, (parseFloat(slider.value) || 0) / 100);
      });
      const val = Object.assign(document.createElement("span"), { className: "blend-val" });
      row.append(cb, dot, name, slider, val);
      legend.append(row);
      legEls.set(m.id, row);
      legValEls.set(m.id, val);
      sliderEls.set(m.id, slider);
    }

    blendBox.append(head, bar, legend);
    paintBlend();
  }

  /** Re-render the grid in blend order from the current results, without a new
   *  search. Cheap: it just reorders existing DOM-equivalent data. */
  function rerank(): void {
    if (!lastResults.length) return;
    renderCards(blendOrdered());
  }

  buildBlendControl();

  function render(p: Payload): void {
    if (p.mode) setMode(p.mode);
    if (typeof p.folder === "string") {
      folder = p.folder;
      if (scope && p.folder) scope.value = p.folder;
    }
    if (typeof p.query === "string" && (p.mode ?? "text") === "text") input.value = p.query;
    lastResults = p.results ?? [];
    if (!lastResults.length) {
      grid.replaceChildren();
      if (blendBox) blendBox.hidden = true;
      setStatus(`No matches${folder ? " in " + folder : ""}.`);
      return;
    }
    const scopeLbl = folder || "library";
    setStatus(`${lastResults.length} result${lastResults.length === 1 ? "" : "s"} · ${scopeLbl}`);
    // Show the blend control whenever there are results, and render in the
    // user's chosen blend order. Default blend = Relevance 100%, so this is the
    // service's exact embedding-similarity (cosine kNN) order out of the box;
    // any aesthetic weight the user dials in re-ranks this same result set
    // client-side (missing aesthetic fields, e.g. color/reverse hits → 0).
    if (blendBox) blendBox.hidden = false;
    syncSliders();
    paintBlend();
    renderCards(blendOrdered());
  }

  /** Build the result grid from an already-ordered hit list. Pure DOM work —
   *  called by `render` (new results) and `rerank` (weights changed). */
  function renderCards(hits: Hit[]): void {
    grid.replaceChildren();
    for (const hit of hits) {
      const name = hit.path.split("/").pop() ?? hit.path;
      const card = document.createElement("div");
      card.className = "card";
      card.title = hit.path;

      // Thumb wrapper with overlaid score (top-left) + dupe badge (top-right),
      // matching the main web UI's `.scoreb` / `.dupe` treatment.
      const thumbWrap = document.createElement("div");
      thumbWrap.className = "thumb";
      const media = hit.thumb
        ? Object.assign(document.createElement("img"), { src: hit.thumb, alt: name })
        : Object.assign(document.createElement("div"), { className: "noimg", textContent: "no preview" });
      const scoreb = Object.assign(document.createElement("span"), {
        className: "scoreb",
        textContent: `${Math.round((hit.prob ?? hit.score) * 100)}`,
      });
      thumbWrap.append(media, scoreb);

      const dupeCount = hit.dupe_count ?? 1;
      if (dupeCount > 1) {
        const badge = Object.assign(document.createElement("span"), {
          className: "dupe",
          textContent: `${dupeCount} copies`,
          title: `${dupeCount} copies — click to see all`,
        });
        badge.addEventListener("click", (e) => {
          e.stopPropagation();
          openDupes(hit, name);
        });
        thumbWrap.append(badge);
      }
      card.append(thumbWrap);

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.append(Object.assign(document.createElement("div"), { className: "name", textContent: name }));
      card.append(meta);
      // One click does BOTH: (a) the existing add-to-prompt action, and
      // (b) opens the full image fullscreen so the user sees the whole thing,
      // not the cover-cropped thumb. The dupe badge stops-propagation, so its
      // own click still opens the duplicates modal instead.
      card.addEventListener("click", () => {
        handlers.onSelect?.(hit);
        openLightbox(hit);
      });
      grid.append(card);
    }
  }

  renderColorSet();
  syncFsButton();

  return {
    render,
    setStatus,
    setBusy,
    /** Re-read the host display mode (call when the host context changes) so the
     *  fullscreen button reflects host-driven mode switches. */
    syncFullscreen: syncFsButton,
    get folder() {
      return folder;
    },
  };
}
