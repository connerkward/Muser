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
  folders,
  getStatus,
  indexFolder,
  search,
  searchColor,
  searchUpload,
  thumbDataUri,
  type SearchHit,
  type SearchResponse,
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

/** Attach base64 thumbnails to a service response's hits.
 *
 *  The ext-app runs in a sandboxed iframe with no direct network access, so it
 *  cannot call /api/thumb itself. We therefore inline every thumbnail as a
 *  base64 data URI in structuredContent: the rep thumb for the card, plus a
 *  per-dupe-path thumb map for the duplicates modal. Because duplicates are
 *  byte-identical files, all dupe entries share the rep image (no extra fetches). */
async function attachThumbs(results: SearchHit[]): Promise<GalleryHit[]> {
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

/** Build the gallery tool result shared by every search mode (text/color/image). */
function galleryResult(
  mode: string,
  query: string,
  folder: string,
  results: GalleryHit[],
) {
  const text = results.length
    ? results.map((h, i) => `${i + 1}. ${h.path} (${((h.prob ?? h.score) * 100).toFixed(1)}%)`).join("\n")
    : "No matches found. Index some images first with index_folder.";
  return {
    structuredContent: { mode, folder, query, results },
    content: [{ type: "text" as const, text }],
  };
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
      const { results } = await search(query, count, folder);
      return galleryResult("text", query, folder ?? "", await attachThumbs(results));
    },
  );

  // ---- App tool: color-palette search (no model) ----------------------------
  registerAppTool(
    server,
    "search_by_color",
    {
      title: "Search by color",
      description:
        "Rank images by how much of one or more colors they contain (LAB color-palette " +
        "index — separate from the semantic model, fully local). Opens the same gallery.",
      inputSchema: {
        hex: z
          .string()
          .describe("One or more comma-separated #rrggbb colors, e.g. '#c83c3c' or '#3a7bd5,#e8d44a'"),
        mode: z
          .enum(["all", "any"])
          .optional()
          .describe("With multiple colors: 'all' (default) requires every color present; 'any' best single match."),
        folder: z.string().optional().describe("Optional folder path to scope results to."),
        k: z.number().optional().describe("How many results to return (default 24)"),
      },
      _meta: { ui: { resourceUri: RESOURCE_URI } },
    },
    async ({ hex, mode, folder, k }) => {
      await ensureService();
      const { query, results } = await searchColor(hex, k ?? 24, folder, mode ?? "all");
      return galleryResult("color", query, folder ?? "", await attachThumbs(results));
    },
  );

  // ---- App tool: image-to-image "find similar" ------------------------------
  registerAppTool(
    server,
    "search_similar",
    {
      title: "Find similar images",
      description:
        "Find images visually similar to an uploaded image (image-to-image search). " +
        "The gallery's image picker supplies the bytes; opens the same gallery of matches.",
      inputSchema: {
        image_base64: z.string().describe("Base64-encoded image bytes (no data: prefix)."),
        filename: z.string().optional().describe("Original filename (for logging only)."),
        folder: z.string().optional().describe("Optional folder path to scope results to."),
        k: z.number().optional().describe("How many results to return (default 24)"),
      },
      _meta: { ui: { resourceUri: RESOURCE_URI } },
    },
    async ({ image_base64, filename, folder, k }) => {
      await ensureService();
      const { results }: SearchResponse = await searchUpload(
        image_base64,
        filename ?? "upload.img",
        k ?? 24,
        folder,
      );
      return galleryResult("image", `image: ${filename ?? "(uploaded)"}`, folder ?? "", await attachThumbs(results));
    },
  );

  // ---- Utility: list indexed folders (drives the gallery's scope picker) -----
  server.tool(
    "list_folders",
    "List indexed folders (path + image count) — used to populate the gallery's folder-scope picker.",
    {},
    async () => {
      await ensureService();
      const items = await folders();
      return { content: [{ type: "text", text: JSON.stringify({ folders: items }, null, 2) }] };
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
