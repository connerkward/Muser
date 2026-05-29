// Muser MCP server: an MCP "app" that searches a folder of images and renders
// the matches in an interactive gallery (ext-apps UI).

import {
  registerAppResource,
  registerAppTool,
  RESOURCE_MIME_TYPE,
} from "@modelcontextprotocol/ext-apps/server";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import fs from "node:fs/promises";
import path from "node:path";
import { z } from "zod";
import { indexFolder, search, stats } from "./core";

const RESOURCE_URI = "ui://muser/search.html";
const UI_HTML = path.join(import.meta.dirname, "..", "dist", "ui", "search.html");

/** Render a small base64 JPEG thumbnail so the sandboxed UI can display local files. */
async function thumbnail(file: string): Promise<string | null> {
  try {
    const sharp = (await import("sharp")).default;
    const buf = await sharp(file)
      .rotate() // honor EXIF orientation
      .resize(240, 240, { fit: "cover" })
      .jpeg({ quality: 72 })
      .toBuffer();
    return `data:image/jpeg;base64,${buf.toString("base64")}`;
  } catch {
    return null;
  }
}

export function createServer(): McpServer {
  const server = new McpServer({ name: "muser", version: "0.1.0" });

  server.tool(
    "index_folder",
    {
      folder: z.string().describe("Absolute path to the image folder to index"),
      recursive: z.boolean().optional().describe("Descend into subfolders (default true)"),
    },
    async ({ folder, recursive }) => {
      const res = await indexFolder(folder, { recursive: recursive ?? true });
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
    { folder: z.string().describe("Absolute path to the image folder") },
    async ({ folder }) => {
      const s = await stats(folder);
      return { content: [{ type: "text", text: JSON.stringify(s, null, 2) }] };
    },
  );

  registerAppTool(
    server,
    "search_images",
    {
      title: "Search images",
      description:
        "Search a folder of images by natural-language description using CLIP, and open a visual gallery of the best matches. The folder must be indexed first with index_folder.",
      inputSchema: {
        folder: z.string().describe("Absolute path to the indexed image folder"),
        query: z.string().describe("Natural-language description of the image to find"),
        k: z.number().optional().describe("How many results to return (default 12)"),
      },
      _meta: { ui: { resourceUri: RESOURCE_URI } },
    },
    async ({ folder, query, k }) => {
      const hits = await search(folder, query, k ?? 12);
      const results = await Promise.all(
        hits.map(async (h) => ({ ...h, thumb: await thumbnail(h.path) })),
      );
      const text = hits.length
        ? hits.map((h, i) => `${i + 1}. ${h.path} (${(h.score * 100).toFixed(1)}%)`).join("\n")
        : "No matches found. Make sure the folder was indexed with index_folder first.";
      return { structuredContent: { folder, query, results }, content: [{ type: "text", text }] };
    },
  );

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
