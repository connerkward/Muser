// Thin HTTP client of the Muser embedded Python service (`muser serve`).
//
// This MCP server NEVER loads a model or touches LanceDB itself — it proxies
// the warm Python service at http://127.0.0.1:7777, which owns the CLIP model
// and the LanceDB index. If the service is down, we spawn it detached and poll
// /api/status until it's ready.

import { spawn } from "node:child_process";

const BASE = process.env.MUSER_SERVICE_URL ?? "http://127.0.0.1:7777";
const PY = process.env.MUSER_PYTHON ?? "/Users/conner/dev/Muser/.venv/bin/python";

export interface SearchHit {
  path: string;
  name: string;
  score: number;
  /** SigLIP-calibrated match probability (0–1), present for SigLIP models. */
  prob?: number;
  /** All file paths that are byte-identical to this image (includes `path`). */
  dupes?: string[];
  /** Number of duplicate files (length of `dupes`). 1 when unique. */
  dupe_count?: number;
  // ---- Per-result aesthetic scores, inlined by /api/search (0–1 normalized) --
  // Present on text/embedding hits; absent on color/reverse-image hits. The
  // gallery's blend control re-ranks by a weighted sum of these (missing → 0).
  // `score` (embedding cosine, above) is the "Relevance" term of that blend.
  /** LAION Aesthetics V2 predictor. */
  aesthetic_v2?: number;
  /** PickScore — human-preference proxy (Pick-a-Pic). */
  pickscore?: number;
  /** LAION Aesthetics V2.5 — refined successor to V2. */
  aesthetic_v25?: number;
  /** Human Preference Score v2.1 — prompt-aware preference. */
  hps_v21?: number;
  /** Zero-shot CLIP aesthetic prompt. */
  aesthetic?: number;
  /** GRIP AI-generated likelihood, 0–100 (present once the aidet facet is scored). */
  ai_pct?: number;
}
/** Pseudo-relevance-feedback chip the service returns alongside results:
 *  a cluster label common to the top hits ("you might also try"). */
export interface Refinement {
  label: string;
  count: number;
}
export interface SearchResponse {
  query: string;
  model: string;
  results: SearchHit[];
  /** Cluster-label refinements derived from the top hits (may be absent/empty). */
  refinements?: Refinement[];
}
export interface StatusResponse {
  model: string;
  models: string[];
  indexed: number;
  db: string;
  ready?: boolean;
  task?: unknown;
  metadata?: unknown;
}
export interface IndexResponse {
  added: number;
  updated: number;
  removed: number;
  total: number;
}
export interface FolderItem {
  folder: string;
  count: number;
}

/** GET /api/status — short timeout; used both as a health check and for info. */
export async function getStatus(timeoutMs = 3000): Promise<StatusResponse | null> {
  try {
    const ctl = AbortSignal.timeout(timeoutMs);
    const r = await fetch(`${BASE}/api/status`, { signal: ctl });
    if (!r.ok) return null;
    return (await r.json()) as StatusResponse;
  } catch {
    return null;
  }
}

/** Spawn `muser serve` detached so it outlives this process, then poll status. */
async function spawnService(): Promise<void> {
  const child = spawn(
    PY,
    ["-c", "from muser.service import serve; serve()"],
    { detached: true, stdio: "ignore" },
  );
  child.unref();
}

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

/**
 * Ensure the Python service is up. Returns its status once ready.
 * If down, spawns it and polls /api/status for up to `maxWaitMs` (default 120s).
 */
export async function ensureService(maxWaitMs = 120_000): Promise<StatusResponse> {
  let status = await getStatus();
  if (status) return status;

  await spawnService();

  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    await sleep(1000);
    status = await getStatus(2000);
    if (status) return status;
  }
  throw new Error(
    `Muser service did not become ready at ${BASE} within ${Math.round(maxWaitMs / 1000)}s. ` +
      `Try starting it manually: ${PY} -c "from muser.service import serve; serve()"`,
  );
}

/** Optional filter/sort knobs that ride /api/search, exposed so MCP callers get
 *  the same filtering/sorting the web UI and CLI have. Default = clean kNN. */
export interface SearchOpts {
  ai_min?: number;        // keep only results with AI-likelihood % >= this (show only AI)
  ai_max?: number;        // drop results with AI-likelihood % > this (remove AI)
  sort?: string;          // "ai" (most-AI first) | "ai_asc" (least-AI first)
  match?: string;         // multi-concept (comma query) mode: "blend" | "all"
  min_short_side?: number;
  max_long_side?: number;
}

/** GET /api/search?q=&k=&folder=… — text query, optional folder scope + filters.
 *
 *  Base behaviour is a PURE EMBEDDING kNN (method="embed", cosine order). The
 *  optional `opts` add the same AI-likelihood / metadata filtering + sort the
 *  web UI exposes — passed through only when set, so an opts-less call stays a
 *  clean similarity ranking. (Still no sort-blend / `neg` — those stay web-only.) */
export async function search(
  query: string,
  k: number,
  folder?: string,
  opts: SearchOpts = {},
): Promise<SearchResponse> {
  let url = `${BASE}/api/search?q=${encodeURIComponent(query)}&k=${k}`;
  if (folder) url += `&folder=${encodeURIComponent(folder)}`;
  if (opts.ai_min) url += `&ai_min=${opts.ai_min}`;
  if (opts.ai_max != null && opts.ai_max < 100) url += `&ai_max=${opts.ai_max}`;
  if (opts.sort) url += `&sort=${encodeURIComponent(opts.sort)}`;
  if (opts.match && opts.match !== "blend") url += `&match=${encodeURIComponent(opts.match)}`;
  if (opts.min_short_side) url += `&min_short_side=${opts.min_short_side}`;
  if (opts.max_long_side) url += `&max_long_side=${opts.max_long_side}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new Error(`/api/search ${r.status}: ${await r.text()}`);
  return (await r.json()) as SearchResponse;
}

/** GET /api/search-color?hex=&k=&folder=&mode= — color-palette search (no model).
 *  `hex` is one or more comma-separated bare/`#`-prefixed RRGGBB colors; with >1,
 *  mode="all" requires every color present, mode="any" takes the best single match. */
export async function searchColor(
  hex: string,
  k: number,
  folder?: string,
  mode: "all" | "any" = "all",
): Promise<SearchResponse> {
  let url = `${BASE}/api/search-color?hex=${encodeURIComponent(hex)}&k=${k}&mode=${mode}`;
  if (folder) url += `&folder=${encodeURIComponent(folder)}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new Error(`/api/search-color ${r.status}: ${await r.text()}`);
  return (await r.json()) as SearchResponse;
}

/** POST /api/search-upload (multipart `file`) — image-to-image "find similar".
 *  `data` is a base64 image payload (no data: prefix); the sandboxed gallery
 *  reads a chosen file, base64-encodes it, and hands the bytes to the MCP tool. */
export async function searchUpload(
  base64: string,
  filename: string,
  k: number,
  folder?: string,
): Promise<SearchResponse> {
  const bytes = Buffer.from(base64, "base64");
  const form = new FormData();
  form.append("file", new Blob([bytes]), filename || "upload.img");
  form.append("k", String(k));
  if (folder) form.append("folder", folder);
  const r = await fetch(`${BASE}/api/search-upload`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(120_000),
  });
  if (!r.ok) throw new Error(`/api/search-upload ${r.status}: ${await r.text()}`);
  return (await r.json()) as SearchResponse;
}

/** GET /api/folders — indexed folders (path + image count) for the scope picker. */
export async function folders(): Promise<FolderItem[]> {
  try {
    const r = await fetch(`${BASE}/api/folders`, { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return [];
    const j = (await r.json()) as { folders?: FolderItem[] };
    return j.folders ?? [];
  } catch {
    return [];
  }
}

/** GET /api/thumb?path=&size= → JPEG bytes, returned as a data: URI for the sandboxed UI. */
export async function thumbDataUri(path: string, size = 260): Promise<string | null> {
  try {
    const url = `${BASE}/api/thumb?path=${encodeURIComponent(path)}&size=${size}`;
    const r = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!r.ok) return null;
    const buf = Buffer.from(await r.arrayBuffer());
    return `data:image/jpeg;base64,${buf.toString("base64")}`;
  } catch {
    return null;
  }
}

/** A large preview for the fullscreen lightbox. The sandboxed iframe can't reach
 *  /api/image itself, so we inline a big thumbnail (whole image, not cover-cropped)
 *  as a base64 data URI. Reuses /api/thumb at a large size — one extra fetch per hit. */
export function fullDataUri(path: string, size = 1400): Promise<string | null> {
  return thumbDataUri(path, size);
}

/** POST /api/contact-sheet {paths, cols, cell, max} → one numbered JPEG montage.
 *  Used to hand the *model* a single compact image of the top results (cells
 *  numbered 1..N matching result order) instead of many individual thumbnails —
 *  far fewer image tokens. Returns the raw base64 JPEG (no data: prefix), or
 *  null on any failure so the caller can fall back to per-result thumbs. */
export async function contactSheet(
  paths: string[],
  opts: { cols?: number; cell?: number; max?: number } = {},
): Promise<{ base64: string; count: number } | null> {
  if (!paths.length) return null;
  try {
    const r = await fetch(`${BASE}/api/contact-sheet`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        paths,
        cols: opts.cols ?? 6,
        cell: opts.cell ?? 256,
        max: opts.max ?? 36,
      }),
      signal: AbortSignal.timeout(60_000),
    });
    if (!r.ok) return null;
    const buf = Buffer.from(await r.arrayBuffer());
    if (!buf.length) return null;
    const header = r.headers.get("X-Sheet-Count");
    const count = header ? parseInt(header, 10) : NaN;
    return {
      base64: buf.toString("base64"),
      count: Number.isFinite(count) ? count : Math.min(paths.length, opts.max ?? 36),
    };
  } catch {
    return null;
  }
}

/** POST /api/index {folder, recursive} */
export async function indexFolder(folder: string, recursive: boolean): Promise<IndexResponse> {
  const r = await fetch(`${BASE}/api/index`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ folder, recursive }),
    signal: AbortSignal.timeout(600_000),
  });
  if (!r.ok) throw new Error(`/api/index ${r.status}: ${await r.text()}`);
  return (await r.json()) as IndexResponse;
}
