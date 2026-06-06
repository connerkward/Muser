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

  function render(p: Payload): void {
    if (p.mode) setMode(p.mode);
    if (typeof p.folder === "string") {
      folder = p.folder;
      if (scope && p.folder) scope.value = p.folder;
    }
    if (typeof p.query === "string" && (p.mode ?? "text") === "text") input.value = p.query;
    grid.replaceChildren();
    if (!p.results?.length) {
      setStatus(`No matches${folder ? " in " + folder : ""}.`);
      return;
    }
    const scopeLbl = folder || "library";
    setStatus(`${p.results.length} result${p.results.length === 1 ? "" : "s"} · ${scopeLbl}`);
    // Render in the exact order the service returned — i.e. embedding-similarity
    // (cosine kNN) order. Do NOT sort or filter by aesthetic/sort-blend scores
    // here; selection and ordering must stay pure embedding. The per-card score
    // badge below shows the embedding score (calibrated prob when present),
    // which is fine — it's display only and doesn't affect ordering.
    for (const hit of p.results) {
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
