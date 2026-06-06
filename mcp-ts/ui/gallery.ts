// Shared Muser gallery: renders search hits into the card grid and wires the
// three search modes (text / color / image-similarity), the folder-scope box,
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
}

export interface Gallery {
  render: (payload: Payload) => void;
  setStatus: (text: string) => void;
  setBusy: (busy: boolean) => void;
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

  // ---- Image-similarity search ---------------------------------------------
  const imgFile = $<HTMLInputElement>("imgFile");
  const imgName = $<HTMLSpanElement>("imgName");
  imgFile.addEventListener("change", () => {
    const file = imgFile.files?.[0];
    if (!file || !handlers.onSearchImage) return;
    imgName.textContent = `Searching for images similar to ${file.name}…`;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const base64 = result.includes(",") ? result.split(",", 2)[1]! : result;
      void handlers.onSearchImage!({ base64, filename: file.name, folder: folderOrUndef() });
    };
    reader.readAsDataURL(file);
  });

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
      card.addEventListener("click", () => handlers.onSelect?.(hit));
      grid.append(card);
    }
  }

  renderColorSet();

  return {
    render,
    setStatus,
    setBusy,
    get folder() {
      return folder;
    },
  };
}
