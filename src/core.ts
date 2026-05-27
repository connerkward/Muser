// Core indexing + search for Muser.
//
// Images are embedded with the CLIP *vision* encoder when indexed; text queries
// are embedded with the CLIP *text* encoder when searching. Both land in the same
// shared embedding space, so a natural-language query retrieves matching images.
// Vectors live in a per-folder LanceDB table at <folder>/.muser.

import path from "node:path";
import fs from "node:fs/promises";

export const MODEL = "Xenova/clip-vit-base-patch32";
export const DIM = 512;
const TABLE = "images";
const IMAGE_EXTS = new Set([
  ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".avif", ".tiff",
]);

// ---------------------------------------------------------------------------
// Lazy model loading. The MCP host lists tools at startup; we must not pull in
// the heavy ONNX runtime or download model weights until something is indexed
// or searched. Each loader is memoised.
// ---------------------------------------------------------------------------
let _tx: Promise<typeof import("@huggingface/transformers")> | null = null;
const tx = () => (_tx ??= import("@huggingface/transformers"));

let _vision: Promise<any> | null = null;
let _processor: Promise<any> | null = null;
let _text: Promise<any> | null = null;
let _tokenizer: Promise<any> | null = null;

const visionModel = () =>
  (_vision ??= tx().then((m) => m.CLIPVisionModelWithProjection.from_pretrained(MODEL)));
const imageProcessor = () =>
  (_processor ??= tx().then((m) => m.AutoProcessor.from_pretrained(MODEL)));
const textModel = () =>
  (_text ??= tx().then((m) => m.CLIPTextModelWithProjection.from_pretrained(MODEL)));
const textTokenizer = () =>
  (_tokenizer ??= tx().then((m) => m.AutoTokenizer.from_pretrained(MODEL)));

function l2normalize(v: number[]): number[] {
  let sum = 0;
  for (const x of v) sum += x * x;
  const norm = Math.sqrt(sum) || 1;
  return v.map((x) => x / norm);
}

/** Embed a single image file into the shared CLIP space (L2-normalized). */
export async function embedImage(file: string): Promise<number[]> {
  const { RawImage } = await tx();
  const [processor, model] = await Promise.all([imageProcessor(), visionModel()]);
  const image = await RawImage.read(file);
  const inputs = await processor(image);
  const { image_embeds } = await model(inputs);
  return l2normalize(Array.from(image_embeds.data as Float32Array));
}

/** Embed a natural-language query into the shared CLIP space (L2-normalized). */
export async function embedText(query: string): Promise<number[]> {
  const [tokenizer, model] = await Promise.all([textTokenizer(), textModel()]);
  const inputs = tokenizer([query], { padding: true, truncation: true });
  const { text_embeds } = await model(inputs);
  return l2normalize(Array.from(text_embeds.data as Float32Array));
}

/** Pre-load the embedding models (e.g. to warm the cache before first use). */
export async function warm(): Promise<void> {
  await Promise.all([imageProcessor(), visionModel(), textTokenizer(), textModel()]);
}

// ---------------------------------------------------------------------------
// LanceDB helpers
// ---------------------------------------------------------------------------
export function dbDir(folder: string): string {
  return path.join(path.resolve(folder), ".muser");
}

const escapeSql = (s: string) => s.replace(/\\/g, "\\\\").replace(/'/g, "\\'");

async function connect(folder: string) {
  const lancedb = await import("@lancedb/lancedb");
  return lancedb.connect(dbDir(folder));
}

async function* walk(dir: string, recursive: boolean): AsyncGenerator<string> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name.startsWith(".")) continue; // skip dotfiles incl. our own .muser dir
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (recursive) yield* walk(full, recursive);
    } else if (IMAGE_EXTS.has(path.extname(entry.name).toLowerCase())) {
      yield full;
    }
  }
}

export interface IndexResult {
  added: number;
  updated: number;
  removed: number;
  total: number;
}

export interface IndexOptions {
  recursive?: boolean;
  onProgress?: (done: number, total: number, file: string) => void;
}

/**
 * Incrementally index a folder of images. Unchanged files (matched by mtime)
 * are skipped; changed files are re-embedded; deleted files are pruned.
 */
export async function indexFolder(folder: string, opts: IndexOptions = {}): Promise<IndexResult> {
  const recursive = opts.recursive ?? true;
  const abs = path.resolve(folder);
  await fs.access(abs); // throws a clear error if the folder is missing

  const db = await connect(abs);
  const names = await db.tableNames();
  let table = names.includes(TABLE) ? await db.openTable(TABLE) : null;

  // What's already indexed?
  const existing = new Map<string, number>();
  if (table) {
    for (const row of await table.query().select(["path", "mtimeMs"]).toArray()) {
      existing.set(row.path as string, Number(row.mtimeMs));
    }
  }

  // What's on disk now?
  const onDisk: { path: string; mtimeMs: number; size: number }[] = [];
  for await (const file of walk(abs, recursive)) {
    const st = await fs.stat(file);
    onDisk.push({ path: file, mtimeMs: st.mtimeMs, size: st.size });
  }
  const onDiskSet = new Set(onDisk.map((f) => f.path));

  const toEmbed = onDisk.filter((f) => existing.get(f.path) !== f.mtimeMs);
  const toRemove = [...existing.keys()].filter((p) => !onDiskSet.has(p));

  // Drop rows that are being removed or replaced before re-adding.
  if (table) {
    const stalePaths = [
      ...toRemove,
      ...toEmbed.filter((f) => existing.has(f.path)).map((f) => f.path),
    ];
    if (stalePaths.length) {
      await table.delete(stalePaths.map((p) => `path = '${escapeSql(p)}'`).join(" OR "));
    }
  }

  // Embed (the slow part — report progress).
  const rows: { path: string; mtimeMs: number; size: number; vector: number[] }[] = [];
  let done = 0;
  for (const f of toEmbed) {
    try {
      const vector = await embedImage(f.path);
      rows.push({ path: f.path, mtimeMs: f.mtimeMs, size: f.size, vector });
    } catch (err) {
      console.error(`  skipped ${f.path}: ${(err as Error).message}`);
    }
    opts.onProgress?.(++done, toEmbed.length, f.path);
  }

  if (rows.length) {
    if (!table) table = await db.createTable(TABLE, rows);
    else await table.add(rows);
  }

  const total = table ? await table.countRows() : 0;
  const added = toEmbed.filter((f) => !existing.has(f.path)).length;
  return { added, updated: toEmbed.length - added, removed: toRemove.length, total };
}

export interface SearchHit {
  path: string;
  score: number; // cosine similarity in [-1, 1]; higher is better
}

/** Search an indexed folder for images matching a natural-language query. */
export async function search(folder: string, query: string, k = 12): Promise<SearchHit[]> {
  const abs = path.resolve(folder);
  const db = await connect(abs);
  if (!(await db.tableNames()).includes(TABLE)) return [];
  const table = await db.openTable(TABLE);
  const queryVector = await embedText(query);
  const rows = await table.search(queryVector).distanceType("cosine").limit(k).toArray();
  return rows.map((r: any) => ({ path: r.path as string, score: 1 - Number(r._distance) }));
}

export interface Stats {
  folder: string;
  total: number;
  model: string;
  dim: number;
  db: string;
}

export async function stats(folder: string): Promise<Stats> {
  const abs = path.resolve(folder);
  const db = await connect(abs);
  const total = (await db.tableNames()).includes(TABLE)
    ? await (await db.openTable(TABLE)).countRows()
    : 0;
  return { folder: abs, total, model: MODEL, dim: DIM, db: dbDir(abs) };
}
