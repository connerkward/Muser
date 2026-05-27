// Muser search gallery — an ext-apps UI rendered by the MCP host.
//
// The host opens this app when the `search_images` tool is called and delivers
// the tool's result via `ontoolresult`. The search box re-runs the tool in place
// via `callServerTool`, keeping the same indexed folder.

import { App, applyHostStyleVariables, applyDocumentTheme } from "@modelcontextprotocol/ext-apps";

interface Hit {
  path: string;
  score: number;
  thumb: string | null;
}
interface Payload {
  folder: string;
  query: string;
  results: Hit[];
}

const grid = document.getElementById("grid") as HTMLDivElement;
const input = document.getElementById("q") as HTMLInputElement;
const form = document.getElementById("form") as HTMLFormElement;
const status = document.getElementById("status") as HTMLDivElement;
const goButton = document.getElementById("go") as HTMLButtonElement;

let folder = "";

const app = new App({ name: "Muser", version: "0.1.0" });

function payloadFromResult(result: unknown): Payload | null {
  const r = result as { structuredContent?: Payload } | undefined;
  if (r?.structuredContent?.results) return r.structuredContent;
  return null;
}

function render(p: Payload): void {
  if (p.folder) folder = p.folder;
  if (typeof p.query === "string") input.value = p.query;
  grid.replaceChildren();
  if (!p.results?.length) {
    status.textContent = `No matches in ${folder || "this folder"}.`;
    return;
  }
  status.textContent = `${p.results.length} result${p.results.length === 1 ? "" : "s"} · ${folder}`;
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
    const nameEl = Object.assign(document.createElement("span"), { className: "name", textContent: name });
    const scoreEl = Object.assign(document.createElement("span"), {
      className: "score",
      textContent: `${(hit.score * 100).toFixed(0)}%`,
    });
    meta.append(nameEl, scoreEl);
    card.append(media, meta);

    // Clicking a result mentions its path back in the conversation.
    card.addEventListener("click", () => {
      void app
        .sendMessage({ content: [{ type: "text", text: `Selected image: ${hit.path}` }] })
        .catch(() => {});
    });

    grid.append(card);
  }
}

async function runSearch(query: string): Promise<void> {
  status.textContent = "Searching…";
  goButton.disabled = true;
  try {
    const result = await app.callServerTool({
      name: "search_images",
      arguments: { folder, query },
    });
    const payload = payloadFromResult(result);
    if (payload) render(payload);
    else status.textContent = "No results.";
  } catch (err) {
    status.textContent = `Error: ${err instanceof Error ? err.message : String(err)}`;
  } finally {
    goButton.disabled = false;
  }
}

// Initial render comes from the tool call that opened this app.
app.ontoolresult = (result: unknown) => {
  const payload = payloadFromResult(result);
  if (payload) render(payload);
};

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (query) void runSearch(query);
});

app
  .connect()
  .then(() => {
    try {
      applyHostStyleVariables();
      applyDocumentTheme();
    } catch {
      /* host styling is best-effort */
    }
    if (status.textContent === "Connecting…") status.textContent = "Ready. Enter a description above.";
  })
  .catch((err: unknown) => {
    status.textContent = `Connection error: ${err instanceof Error ? err.message : String(err)}`;
  });
