// Live Muser gallery — the ext-apps UI rendered by an MCP host.
//
// The host opens this app when `search_images` is called and delivers the
// tool's result via `ontoolresult`. The search box re-runs the tool in place
// via `callServerTool`, keeping the same indexed folder.

import { App, applyHostStyleVariables, applyDocumentTheme } from "@modelcontextprotocol/ext-apps";
import { createGallery, type Payload } from "./gallery";

const app = new App({ name: "Muser", version: "0.1.0" });

function payloadFrom(result: unknown): Payload | null {
  const r = result as { structuredContent?: Payload } | undefined;
  return r?.structuredContent?.results ? r.structuredContent : null;
}

const gallery = createGallery({
  onSearch: async (query) => {
    gallery.setStatus("Searching…");
    gallery.setBusy(true);
    try {
      const result = await app.callServerTool({
        name: "search_images",
        arguments: { folder: gallery.folder, query },
      });
      const payload = payloadFrom(result);
      if (payload) gallery.render(payload);
      else gallery.setStatus("No results.");
    } catch (err) {
      gallery.setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      gallery.setBusy(false);
    }
  },
  onSelect: (hit) => {
    void app
      .sendMessage({ content: [{ type: "text", text: `Selected image: ${hit.path}` }] })
      .catch(() => {});
  },
});

// Initial render comes from the tool call that opened this app.
app.ontoolresult = (result: unknown) => {
  const payload = payloadFrom(result);
  if (payload) gallery.render(payload);
};

app
  .connect()
  .then(() => {
    try {
      applyHostStyleVariables();
      applyDocumentTheme();
    } catch {
      /* host styling is best-effort */
    }
    gallery.setStatus("Ready. Enter a description above.");
  })
  .catch((err: unknown) => {
    gallery.setStatus(`Connection error: ${err instanceof Error ? err.message : String(err)}`);
  });
