# Muser — agent notes

Natural-language image search. CLI + MCP App. Bun + TypeScript.

## Architecture

- `src/core.ts` — the substance. CLIP embeddings (vision encoder for images, text encoder
  for queries; same shared space) via `@huggingface/transformers`, stored/searched in a
  per-folder LanceDB table at `<folder>/.muser`. Models are lazy-loaded and memoised.
- `src/cli.ts` — `muser` bin (commander): `index`, `search`, `info`.
- `src/server.ts` — `createServer()`: MCP tools `index_folder`, `index_info`, and the
  `search_images` **ext-apps** tool that renders the gallery UI.
- `src/mcp.ts` — `muser-mcp` bin: stdio (default) or `--http`.
- `ui/search.html` + `ui/search-app.ts` — the ext-apps gallery, built by Vite
  (`vite-plugin-singlefile`) into `dist/ui/search.html`, which `server.ts` serves.

## Conventions / gotchas

- **Run with Bun**, not node/tsc — bins have a `#!/usr/bin/env bun` shebang and TS runs
  directly. `bun run build:ui` must run before the MCP UI works (also runs on postinstall).
- Keep heavy deps (`@huggingface/transformers`, `@lancedb/lancedb`, `sharp`) **lazy-loaded**
  — MCP hosts list tools at startup and must not block on model downloads.
- LanceDB schema is **inferred from the first batch** of rows (no apache-arrow dep). The
  vector column is named `vector`; search is cosine distance, score = `1 - _distance`.
- The sandboxed UI can't read local files, so `search_images` returns base64 JPEG
  thumbnails (via `sharp`) in `structuredContent.results[].thumb`.

## Reference implementations

Patterns mirror two working MCP apps by the same author:
`~/dev/mcp-apple-notes` (transformers.js + LanceDB + ext-apps) and the `gritt` repo
(clean `server.ts`/`main.ts` + Vite singlefile ext-apps skeleton).
