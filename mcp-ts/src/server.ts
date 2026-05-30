// Muser MCP server: an MCP "app" that semantically searches an image library
// and renders the matches in an interactive gallery (ext-apps UI).
//
// THIN CLIENT ONLY. This server never loads a model or touches LanceDB. It
// proxies the warm Python embedded service (`muser serve`, http://127.0.0.1:7777),
// which owns the CLIP model + the LanceDB index. On a tool call we ensure the
// service is up (spawning it detached if needed), call /api/search, fetch a
// JPEG thumbnail per hit from /api/thumb (base64), and hand the results to the
// ext-app gallery UI to render.

import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import {
  ensureService,
  getStatus,
  indexFolder,
  search,
  thumbDataUri,
  type SearchHit,
} from "./service";

const RESOURCE_URI = "ui://muser/search.html";
const UI_HTML = path.join(import.meta.dirname, "..", "dist", "ui", "search.html");

/** A gallery card: service hit + a base64 thumbnail the sandboxed UI can show. */
interface GalleryHit extends SearchHit {
  thumb: string | null;
  /** base64 thumb per duplicate path, so the sandboxed iframe (which can't
   *  reach /api/thumb) can render the click-into-duplicates modal inline.
   *  Dupes are byte-identical images, so every entry reuses the rep thumb. */
  dupeThumbs?: Record<string, string | null>;
}

/** Run a search against the embedded service and attach thumbnails.
 *
 *  The ext-app runs in a sandboxed iframe with no direct network access, so it
 *  cannot call /api/thumb itself. We therefore inline every thumbnail as a
 *  base64 data URI in structuredContent: the rep thumb for the card, plus a
 *  per-dupe-path thumb map for the duplicates modal. Because duplicates are
 *  byte-identical files, all dupe entries share the rep image (no extra fetches). */
async function searchWithThumbs(query: string, k: number): Promise<GalleryHit[]> {
  const { results } = await search(query, k);
  return Promise.all(
    results.map(async (h) => {
      const thumb = await thumbDataUri(h.path);
      const hit: GalleryHit = { ...h, thumb };
      if (h.dupes && h.dupes.length > 1) {
        // Same image on disk → reuse the rep thumb for every duplicate path.
        hit.dupeThumbs = Object.fromEntries(h.dupes.map((p) => [p, thumb]));
      }
      return hit;
    }),
  );
}

export function createServer(): McpServer {
  const server = new McpServer({ name: "muser", version: "0.2.0" });

  // ---- App tool: opens the gallery UI ---------------------------------------
  registerAppTool(
    server,
    "search_images",
    {
      title: "Search images",
      description:
        "Semantically search your image library by natural-language description and open a visual " +
        "gallery of the best matches. Searches the whole index by default; pass a folder only to " +
        "scope/ensure indexing of a specific folder first.",
      inputSchema: {
        folder: z
          .string()
          .optional()
          .describe("Optional absolute folder path to index before searching. Omit to search the whole library."),
        query: z.string().describe("Natural-language description of the image to find"),
        k: z.number().optional().describe("How many results to return (default 24)"),
      },
      _meta: { ui: { resourceUri: RESOURCE_URI } },
    },
    async ({ folder, query, k }) => {
      await ensureService();
      if (folder) {
        // Ensure the requested folder is in the index before searching it.
        await indexFolder(folder, true).catch(() => undefined);
      }
      const count = k ?? 24;
      const results = await searchWithThumbs(query, count);
      const text = results.length
        ? results.map((h, i) => `${i + 1}. ${h.path} (${(h.score * 100).toFixed(1)}%)`).join("\n")
        : "No matches found. Index some images first with index_folder.";
      return {
        structuredContent: { folder: folder ?? "", query, results },
        content: [{ type: "text", text }],
      };
    },
  );

  // ---- Callback / utility tools (proxy the service) -------------------------
  server.tool(
    "index_folder",
    "Index (or refresh) a folder of images in the Muser library so it becomes searchable.",
    {
      folder: z.string().describe("Absolute path to the image folder to index"),
      recursive: z.boolean().optional().describe("Descend into subfolders (default true)"),
    },
    async ({ folder, recursive }) => {
      await ensureService();
      const res = await indexFolder(folder, recursive ?? true);
      return {
        content: [
          {
            type: "text",
            text:
              `Indexed ${folder}: +${res.added} added, ${res.updated} updated, ` +
              `${res.removed} removed. ${res.total} images now searchable.`,
          },
        ],
      };
    },
  );

  server.tool(
    "index_info",
    "Report Muser library status: active model, available models, indexed image count, and db path.",
    {},
    async () => {
      const status = (await getStatus()) ?? (await ensureService());
      return { content: [{ type: "text", text: JSON.stringify(status, null, 2) }] };
    },
  );

  // ---- The UI resource: the built single-file gallery HTML ------------------
  registerAppResource(
    server,
    RESOURCE_URI,
    RESOURCE_URI,
    { mimeType: RESOURCE_MIME_TYPE },
    async () => {
      let html: string;
      try {
        html = await fs.readFile(UI_HTML, "utf-8");
      } catch {
        html =
          `<html><body><pre style="font:13px ui-monospace;padding:16px">` +
          `Muser UI not built. Run: bun run build:ui</pre></body></html>`;
      }
      return { contents: [{ uri: RESOURCE_URI, mimeType: RESOURCE_MIME_TYPE, text: html }] };
    },
  );

  return server;
}
