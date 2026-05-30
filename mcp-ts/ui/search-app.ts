// Live Muser gallery — the ext-apps UI rendered by an MCP host.
//
// The host opens this app when `search_images` is called and delivers the
// tool's result via `onToolResult`. The search box re-runs the tool in place
// via `callTool`. Host bridge + frame-fitting live in `mcp-frame.ts`.

import { connectMcpFrame, resultJson, type McpFrame } from "./mcp-frame";
import { createGallery, type Payload } from "./gallery";

let frame: McpFrame | null = null;

const gallery = createGallery({
  onSearch: async (query) => {
    if (!frame) {
      gallery.setStatus("Not connected to a host.");
      return;
    }
    gallery.setStatus("Searching…");
    gallery.setBusy(true);
    try {
      const result = await frame.callTool("search_images", { query });
      const payload = resultJson<Payload>(result);
      if (payload?.results) gallery.render(payload);
      else gallery.setStatus("No results.");
    } catch (err) {
      gallery.setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      gallery.setBusy(false);
    }
  },
  onSelect: (hit) => {
    frame?.app
      .sendMessage({ role: "user", content: [{ type: "text", text: `Selected image: ${hit.path}` }] })
      .catch(() => {});
  },
  // The sandboxed iframe can't call /api/reveal directly, so ask the host/agent
  // to reveal the file in Finder via a chat message.
  onReveal: (path) => {
    frame?.app
      .sendMessage({ role: "user", content: [{ type: "text", text: `Reveal this file in Finder: ${path}` }] })
      .catch(() => {});
  },
});

connectMcpFrame({
  name: "Muser",
  version: "0.2.0",
  // Initial render comes from the tool call that opened this app.
  onToolResult: (result) => {
    const payload = resultJson<Payload>(result);
    if (payload?.results) gallery.render(payload);
  },
})
  .then((f) => {
    frame = f;
    gallery.setStatus(
      f.connected ? "Ready. Enter a description above." : "Standalone preview — no host connected.",
    );
  })
  .catch((err: unknown) => {
    gallery.setStatus(`Connection error: ${err instanceof Error ? err.message : String(err)}`);
  });
