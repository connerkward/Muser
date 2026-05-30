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
  /** All file paths that are byte-identical to this image (includes `path`). */
  dupes?: string[];
  /** Number of duplicate files (length of `dupes`). 1 when unique. */
  dupe_count?: number;
}
export interface SearchResponse {
  query: string;
  model: string;
  results: SearchHit[];
}
export interface StatusResponse {
  model: string;
  models: string[];
  indexed: number;
  db: string;
}
export interface IndexResponse {
  added: number;
  updated: number;
  removed: number;
  total: number;
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

/** GET /api/search?q=&k= */
export async function search(query: string, k: number): Promise<SearchResponse> {
  const url = `${BASE}/api/search?q=${encodeURIComponent(query)}&k=${k}`;
  const r = await fetch(url, { signal: AbortSignal.timeout(60_000) });
  if (!r.ok) throw new Error(`/api/search ${r.status}: ${await r.text()}`);
  return (await r.json()) as SearchResponse;
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
