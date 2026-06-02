// Shared Muser gallery: renders search hits into the card grid and wires the
// search box. Knows nothing about MCP — callers provide an `onSearch` handler.
// Used by both `search-app.ts` (live MCP host) and `preview.ts` (mock preview).

import "./gallery.css";

export interface Hit {
  path: string;
  score: number;
  thumb: string | null;
  /** All paths that are this same image (includes `path`). Length 1 when unique. */
  dupes?: string[];
  dupe_count?: number;
  /** base64 thumb per dupe path, supplied inline by the server (the sandboxed
   *  iframe can't fetch /api/thumb). Falls back to the card thumb if absent. */
  dupeThumbs?: Record<string, string | null>;
}
export interface Payload {
  folder: string;
  query: string;
  results: Hit[];
}

export interface GalleryHandlers {
  onSearch: (query: string) => void | Promise<void>;
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

export function createGallery(handlers: GalleryHandlers): Gallery {
  const grid = document.getElementById("grid") as HTMLDivElement;
  const input = document.getElementById("q") as HTMLInputElement;
  const form = document.getElementById("form") as HTMLFormElement;
  const status = document.getElementById("status") as HTMLDivElement;
  const goButton = document.getElementById("go") as HTMLButtonElement;

  // Click-into-duplicates modal.
  const modal = document.getElementById("dupes") as HTMLDivElement;
  const modalTitle = document.getElementById("dupesTitle") as HTMLHeadingElement;
  const modalSub = document.getElementById("dupesSub") as HTMLDivElement;
  const modalFiles = document.getElementById("dupesFiles") as HTMLDivElement;
  const modalClose = document.getElementById("dupesClose") as HTMLSpanElement;

  let folder = "";

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
      // Reveal is only wired in the live app (host-routed); skip if no handler.
      if (handlers.onReveal) {
        const reveal = Object.assign(document.createElement("button"), { textContent: "Reveal" });
        reveal.addEventListener("click", () => handlers.onReveal!(p));
        file.append(reveal);
      }
      modalFiles.append(file);
    }
    modal.classList.add("show");
  }

  const setStatus = (text: string) => {
    status.textContent = text;
  };
  const setBusy = (busy: boolean) => {
    goButton.disabled = busy;
  };

  function render(p: Payload): void {
    if (p.folder) folder = p.folder;
    if (typeof p.query === "string") input.value = p.query;
    grid.replaceChildren();
    if (!p.results?.length) {
      setStatus(`No matches in ${folder || "this folder"}.`);
      return;
    }
    setStatus(`${p.results.length} result${p.results.length === 1 ? "" : "s"} · ${folder}`);
    for (const hit of p.results) {
      const name = hit.path.split("/").pop() ?? hit.path;
      const card = document.createElement("div");
      card.className = "card";
      card.title = hit.path;

      const media = hit.thumb
        ? Object.assign(document.createElement("img"), { src: hit.thumb, alt: name })
        : Object.assign(document.createElement("div"), { className: "noimg", textContent: "no preview" });

      // Click-into-duplicates: a copy-count badge over the thumbnail.
      const dupeCount = hit.dupe_count ?? 1;
      if (dupeCount > 1) {
        const thumbWrap = document.createElement("div");
        thumbWrap.className = "thumb";
        const badge = Object.assign(document.createElement("span"), {
          className: "dupe",
          textContent: `${dupeCount} copies`,
          title: `${dupeCount} copies — click to see all`,
        });
        badge.addEventListener("click", (e) => {
          e.stopPropagation();
          openDupes(hit, name);
        });
        thumbWrap.append(media, badge);
        card.append(thumbWrap);
      } else {
        card.append(media);
      }

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.append(
        Object.assign(document.createElement("span"), { className: "name", textContent: name }),
        Object.assign(document.createElement("span"), {
          className: "score",
          textContent: `${(hit.score * 100).toFixed(0)}%`,
        }),
      );
      card.append(meta);
      card.addEventListener("click", () => handlers.onSelect?.(hit));
      grid.append(card);
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (query) void handlers.onSearch(query);
  });

  return {
    render,
    setStatus,
    setBusy,
    get folder() {
      return folder;
    },
  };
}
