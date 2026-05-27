# Muser

Index, vectorize, and search a folder of images **by natural language**. Muser embeds
every image with [CLIP](https://github.com/openai/CLIP) (running locally, on-device, via
[`@huggingface/transformers`](https://github.com/huggingface/transformers.js)) and stores
the vectors in an embedded [LanceDB](https://lancedb.com) table. Then you can search with a
text query like _"a dog on a beach at sunset"_ and get back the closest-matching photos —
no API keys, fully offline.

Muser ships two surfaces over the same core:

- **`muser`** — a command-line tool to index and search.
- **`muser-mcp`** — an [MCP](https://modelcontextprotocol.io) server (an _MCP App_) that
  searches images and renders the matches in an interactive gallery inside the host.

## How it works

Images are embedded with the CLIP **vision** encoder at index time; text queries are
embedded with the CLIP **text** encoder at search time. Both land in the same shared
embedding space, so a description retrieves visually matching images. Each indexed folder
gets its own LanceDB index at `<folder>/.muser`.

## Requirements

- [Bun](https://bun.sh) ≥ 1.3

## Install

```bash
git clone https://github.com/connerkward/Muser
cd Muser
bun install   # also builds the MCP UI (postinstall)
```

## CLI usage

```bash
# Index a folder (recursive by default). First run downloads the CLIP model (~150 MB).
bun src/cli.ts index ~/Pictures/screenshots

# Search it
bun src/cli.ts search "a login screen with a blue button" --in ~/Pictures/screenshots

# Limit results / get JSON
bun src/cli.ts search "sunset over water" --in ~/Pictures -k 5 --json

# Index stats
bun src/cli.ts info --in ~/Pictures/screenshots
```

Re-running `index` is incremental: unchanged files are skipped, changed files re-embedded,
deleted files pruned.

After `bun link` you can call `muser` / `muser-mcp` directly instead of `bun src/...`.

## MCP usage

Add to your MCP host (e.g. Claude Desktop) config:

```json
{
  "mcpServers": {
    "muser": { "command": "bun", "args": ["/absolute/path/to/Muser/src/mcp.ts"] }
  }
}
```

Tools exposed:

| Tool            | Purpose                                                              |
| --------------- | ------------------------------------------------------------------- |
| `index_folder`  | Index/re-index a folder of images.                                  |
| `index_info`    | Report index stats for a folder.                                    |
| `search_images` | Search by description and open the gallery UI (an _MCP App_).       |

Run over HTTP instead of stdio with `bun src/mcp.ts --http` (default port `3939`).

## Releasing

Versioning and changelogs are managed with [Changesets](https://github.com/changesets/changesets):

```bash
bun run changeset      # describe a change
bun run version        # apply pending changesets -> bump version + CHANGELOG
bun run release        # build + publish
```

## License

MIT
