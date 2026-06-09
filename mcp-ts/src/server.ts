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
  contactSheet,
  ensureService,
  folders,
  fullDataUri,
  getStatus,
  indexFolder,
  search,
  searchColor,
  searchUpload,
  thumbDataUri,
  type Refinement,
  type SearchHit,
  type SearchResponse,
} from "./service";

const RESOURCE_URI = "ui://muser/search.html";
const UI_HTML = path.join(import.meta.dirname, "..", "dist", "ui", "search.html");

/** A gallery card: service hit + a base64 thumbnail the sandboxed UI can show. */
interface GalleryHit extends SearchHit {
  thumb: string | null;
  /** Large base64 preview (whole image, not cover-cropped) for the fullscreen
   *  lightbox. The sandboxed iframe can't fetch /api/image, so it's inlined. */
  full?: string | null;
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
      // The card thumb (cover-cropped) and a large whole-image preview for the
      // fullscreen lightbox are both inlined as base64 — the sandboxed iframe
      // has no network access of its own.
      const [thumb, full] = await Promise.all([thumbDataUri(h.path), fullDataUri(h.path)]);
      const hit: GalleryHit = { ...h, thumb, full };
      if (h.dupes && h.dupes.length > 1) {
        // Same image on disk → reuse the rep thumb for every duplicate path.
        hit.dupeThumbs = Object.fromEntries(h.dupes.map((p) => [p, thumb]));
      }
      return hit;
    }),
  );
}

/** LLM-facing search context.
 *
 *  This block is appended to the tool result's `content` (the text the *model*
 *  reads) but deliberately NOT to `structuredContent` (what the gallery UI
 *  renders). So it gives the model better grounding for follow-up queries —
 *  index coverage, the active scope, the score spread, refinement chips, and a
 *  plain-language hint — without ever appearing in the user-facing gallery.
 *
 *  Uses only existing service endpoints (/api/status, /api/folders) plus the
 *  scores already returned, so it adds no new backend surface. */
async function buildLlmMeta(
  mode: string,
  query: string,
  folder: string,
  results: GalleryHit[],
  refinements: Refinement[] | undefined,
): Promise<string> {
  // Score the model actually saw — prefer the calibrated probability.
  const scores = results.map((h) => h.prob ?? h.score).filter((s) => typeof s === "number");
  let dist: { min: number; max: number; mean: number; spread: number } | null = null;
  if (scores.length) {
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
    dist = { min, max, mean, spread: max - min };
  }

  // Index coverage + scope, from endpoints this client already proxies.
  const [status, folderItems] = await Promise.all([
    getStatus().catch(() => null),
    folders().catch(() => [] as { folder: string; count: number }[]),
  ]);

  // A short, actionable hint string the model can lean on.
  const hints: string[] = [];
  hints.push(
    mode === "color"
      ? "scores reflect how much of the queried color(s) the image contains (LAB ΔE, not semantics)"
      : "scores are cosine similarity (calibrated probability when shown); higher = closer match",
  );
  if (dist) {
    if (results.length && dist.max < 0.2) {
      hints.push("top score is low — likely nothing in the index matches well; try rephrasing or indexing more folders");
    } else if (dist.spread < 0.03 && results.length > 3) {
      hints.push("scores are tightly clustered (low spread) — the query may be vague; suggest narrowing it");
    } else if (dist.spread > 0.15) {
      hints.push("wide score spread — the top few hits are clearly stronger than the tail");
    }
  }
  if (!results.length) {
    hints.push("no matches — index a folder with index_folder, or broaden the query/scope");
  }
  if (refinements?.length) {
    hints.push(
      "refinements are cluster labels common to the top hits — good 'you might also try' follow-up queries",
    );
  }

  const meta = {
    note: "LLM-only context — not shown to the user. Use it to craft better follow-up searches.",
    mode,
    query,
    scope: folder || "(whole library)",
    returned: results.length,
    indexed_total: status?.indexed ?? null,
    model: status?.model ?? null,
    indexed_folders: folderItems.slice(0, 12).map((f) => ({ folder: f.folder, count: f.count })),
    score_distribution: dist,
    refinements: refinements ?? [],
    hint: hints.join("; "),
  };
  return "muser_search_meta (LLM-only, hidden from user):\n" + JSON.stringify(meta, null, 2);
}

/** How many top results to pack into the LLM-facing contact sheet. A deliberate
 *  buffer below what's legibly resolvable in one 256px-cell montage, so cells
 *  stay readable and the token cost is a single image. The gallery
 *  (structuredContent) still shows ALL results — this only bounds what the model
 *  sees. */
const LLM_SHEET_MAX = 36;
/** How many top results to ALSO send as individual full-detail image blocks
 *  (after the sheet) for close reasoning. */
const LLM_DETAIL_COUNT = 3;
/** Fallback when the contact-sheet call fails: top-K individual thumbs. */
const LLM_FALLBACK_COUNT = 6;

/** An MCP image content block the host renders into the model's context. */
type ImageBlock = { type: "image"; data: string; mimeType: string };

/** Turn a base64 image data URI into an MCP image block, deriving the mimeType.
 *  Returns null if the URI is missing/unparseable. */
function dataUriToImageBlock(uri: string | null | undefined): ImageBlock | null {
  if (!uri) return null;
  const m = /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/.exec(uri);
  if (!m || !m[1] || !m[2]) return null;
  return { type: "image", data: m[2], mimeType: m[1] };
}

/** Turn the top-K hits' small thumbnails into MCP image content blocks (used as
 *  the fallback path when the contact-sheet endpoint is unavailable). */
function llmPreviewImages(results: GalleryHit[], count: number): ImageBlock[] {
  const blocks: ImageBlock[] = [];
  for (const h of results.slice(0, count)) {
    const block = dataUriToImageBlock(h.thumb);
    if (block) blocks.push(block);
  }
  return blocks;
}

/** Build the gallery tool result shared by every search mode (text/color/image). */
async function galleryResult(
  mode: string,
  query: string,
  folder: string,
  results: GalleryHit[],
  refinements?: Refinement[],
) {
  const list = results.length
    ? results
        .map((h, i) => {
          const ai = typeof h.ai_pct === "number" ? `, AI ${h.ai_pct}%` : "";
          return `${i + 1}. ${h.path} (${((h.prob ?? h.score) * 100).toFixed(1)}%${ai})`;
        })
        .join("\n")
    : "No matches found. Index some images first with index_folder.";
  const meta = await buildLlmMeta(mode, query, folder, results, refinements);

  const content: Array<
    { type: "text"; text: string } | ImageBlock
  > = [{ type: "text", text: list }];

  // Give the *model* a visual of the matches so it can reason (pick the best,
  // notice query drift, suggest refinements). The user still sees every result
  // in the gallery via structuredContent.
  //
  // Primary: ONE numbered contact sheet of the top results — cells numbered
  // 1..N matching the result-list order — plus the top few as full-detail
  // thumbs for close reasoning. Costs one image + a handful, not one-per-result.
  if (results.length) {
    const sheetPaths = results.slice(0, LLM_SHEET_MAX).map((h) => h.path);
    const sheet = await contactSheet(sheetPaths, { cols: 6, cell: 256, max: LLM_SHEET_MAX });

    if (sheet) {
      content.push({
        type: "text",
        text:
          `Contact sheet of the top ${sheet.count} results below — cells are numbered ` +
          `1..${sheet.count} matching the result list order (use the numbers to refer to ` +
          `specific results).`,
      });
      content.push({ type: "image", data: sheet.base64, mimeType: "image/jpeg" });

      const detail = llmPreviewImages(results, LLM_DETAIL_COUNT);
      if (detail.length) {
        content.push({ type: "text", text: `Top ${detail.length} in detail:` });
        content.push(...detail);
      }
    } else {
      // Contact-sheet endpoint unavailable → previous behavior: top-K thumbs.
      const images = llmPreviewImages(results, LLM_FALLBACK_COUNT);
      if (images.length) {
        content.push({
          type: "text",
          text:
            `Top ${images.length} result${images.length === 1 ? "" : "s"} shown below as ` +
            `image${images.length === 1 ? "" : "s"} (for your visual reference; the user sees the full gallery).`,
        });
        content.push(...images);
      }
    }
  }

  // Hidden-from-user, LLM-facing context block — kept last.
  content.push({ type: "text", text: meta });

  return {
    // structuredContent drives the gallery UI — no _meta here, so it never renders.
    structuredContent: { mode, folder, query, results },
    content,
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
        "scope/ensure indexing of a specific folder first. Optional filters: ai_min (keep only " +
        "results whose AI-generated likelihood %% is at least this), sort ('ai' = most-AI-likely " +
        "first, 'ai_asc' = least), and min_short_side / max_long_side resolution bounds in px. " +
        "Multi-concept: separate concepts with commas (e.g. 'car, snow') to combine them as " +
        "separate vectors, and prefix one with '-' to subtract it ('car, -person'); the `match` " +
        "param picks how comma concepts combine.",
      inputSchema: {
        folder: z
          .string()
          .optional()
          .describe("Optional absolute folder path to index before searching. Omit to search the whole library."),
        query: z.string().describe("Natural-language description; use commas for multiple concepts ('car, snow'), '-' to subtract ('car, -person')"),
        match: z
          .enum(["blend", "all"])
          .optional()
          .describe("How comma-separated concepts combine: 'blend' (sum vectors, looser; default) or 'all' (each concept must be strongly present — intersection)."),
        k: z.number().optional().describe("How many results to return (default 24)"),
        ai_min: z
          .number()
          .optional()
          .describe("Keep only results with AI-generated likelihood % >= this (0–100). Omit for no AI filter."),
        sort: z
          .enum(["ai", "ai_asc"])
          .optional()
          .describe("Reorder by AI-likelihood: 'ai' = most-AI first, 'ai_asc' = least-AI first. Omit for relevance order."),
        min_short_side: z.number().optional().describe("Only images whose shorter side is at least this many px"),
        max_long_side: z.number().optional().describe("Only images whose longer side is at most this many px"),
      },
      _meta: { ui: { resourceUri: RESOURCE_URI } },
    },
    async ({ folder, query, k, ai_min, sort, match, min_short_side, max_long_side }) => {
      await ensureService();
      if (folder) {
        // Ensure the requested folder is in the index before searching it.
        await indexFolder(folder, true).catch(() => undefined);
      }
      const count = k ?? 24;
      const { results, refinements } = await search(query, count, folder, {
        ai_min,
        sort,
        match,
        min_short_side,
        max_long_side,
      });
      return galleryResult("text", query, folder ?? "", await attachThumbs(results), refinements);
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
      const { query, results, refinements } = await searchColor(hex, k ?? 24, folder, mode ?? "all");
      return galleryResult("color", query, folder ?? "", await attachThumbs(results), refinements);
    },
  );

  // ---- App tool: reverse image search (image-to-image "find similar") -------
  registerAppTool(
    server,
    "search_similar",
    {
      title: "Reverse image search",
      description:
        "Reverse image search: given a query image, find the visually similar images in the " +
        "user's indexed library (image-to-image embedding kNN — NOT a web search; it looks " +
        "only inside the local Muser index). Pass the image bytes as base64; results open in " +
        "the gallery ranked by visual similarity. In the gallery UI this is the 'Reverse image' " +
        "tab, where the user can also drag-and-drop or pick an image directly.",
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
      const { results, refinements }: SearchResponse = await searchUpload(
        image_base64,
        filename ?? "upload.img",
        k ?? 24,
        folder,
      );
      return galleryResult(
        "image",
        `image: ${filename ?? "(uploaded)"}`,
        folder ?? "",
        await attachThumbs(results),
        refinements,
      );
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
