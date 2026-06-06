// Live Muser gallery — the ext-apps UI rendered by an MCP host.
//
// The host opens this app when one of the search tools is called and delivers
// the tool's result via `onToolResult`. The gallery's tabs (text / color /
// image-similarity) and folder-scope box re-run the corresponding MCP tool in
// place via `callTool`. Host bridge + frame-fitting live in `mcp-frame.ts`.

import { connectMcpFrame, resultJson, type McpFrame } from "./mcp-frame";
import { createGallery, type Payload } from "./gallery";

let frame: McpFrame | null = null;

/** Run an MCP tool and render its gallery payload, with shared busy/status. */
async function runTool(name: string, args: Record<string, unknown>, label: string): Promise<void> {
  if (!frame) {
    gallery.setStatus("Not connected to a host.");
    return;
  }
  gallery.setStatus(label);
  gallery.setBusy(true);
  try {
    const result = await frame.callTool(name, args);
    const payload = resultJson<Payload>(result);
    if (payload?.results) gallery.render(payload);
    else gallery.setStatus("No results.");
  } catch (err) {
    gallery.setStatus(`Error: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    gallery.setBusy(false);
  }
}

const gallery = createGallery({
  onSearchText: (query, folder) =>
    runTool("search_images", folder ? { query, folder } : { query }, "Searching…"),
  onSearchColor: ({ hex, mode, folder }) =>
    runTool("search_by_color", folder ? { hex, mode, folder } : { hex, mode }, "Searching by color…"),
  onSearchImage: ({ base64, filename, folder }) =>
    runTool(
      "search_similar",
      folder ? { image_base64: base64, filename, folder } : { image_base64: base64, filename },
      "Finding similar images…",
    ),
  onLoadFolders: async () => {
    if (!frame) return [];
    try {
      const result = await frame.callTool("list_folders", {});
      const payload = resultJson<{ folders?: { folder: string }[] }>(result);
      return (payload?.folders ?? []).map((f) => f.folder);
    } catch {
      return [];
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
  // Prefer the host's real fullscreen display mode. Return true only when the
  // host actually offers + switches it; otherwise the gallery uses its CSS
  // maximized fallback.
  onToggleFullscreen: async () => {
    if (!frame || !frame.connected || !frame.canFullscreen()) return false;
    await frame.toggleFullscreen();
    gallery.syncFullscreen();
    return true;
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
  // The host can switch display mode (e.g. user hits its own fullscreen
  // control); keep our toggle button in sync with it.
  onContextChanged: () => gallery.syncFullscreen(),
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
