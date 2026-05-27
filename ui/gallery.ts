// Shared Muser gallery: renders search hits into the card grid and wires the
// search box. Knows nothing about MCP — callers provide an `onSearch` handler.
// Used by both `search-app.ts` (live MCP host) and `preview.ts` (mock preview).

import "./gallery.css";

export interface Hit {
  path: string;
  score: number;
  thumb: string | null;
}
export interface Payload {
  folder: string;
  query: string;
  results: Hit[];
}

export interface GalleryHandlers {
  onSearch: (query: string) => void | Promise<void>;
  onSelect?: (hit: Hit) => void;
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

  let folder = "";

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

      const meta = document.createElement("div");
      meta.className = "meta";
      meta.append(
        Object.assign(document.createElement("span"), { className: "name", textContent: name }),
        Object.assign(document.createElement("span"), {
          className: "score",
          textContent: `${(hit.score * 100).toFixed(0)}%`,
        }),
      );
      card.append(media, meta);
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
