# Muser

[![CI](https://github.com/connerkward/Muser/actions/workflows/ci.yml/badge.svg)](https://github.com/connerkward/Muser/actions/workflows/ci.yml)

**Accessible, local-first image search — with features pointed at asset generation
and taste / curation for model training.**

Muser indexes a folder of images, lets you search it by natural language, and treats
that index as the substrate for two adjacent jobs: (1) **finding source assets** for
generative work — image-to-image search, the selection cart, reverse-image lookup,
NSFW / C2PA-provenance filters — and (2) **curating training sets** for personal
LoRAs and aesthetic models — cluster exploration, aesthetic ranking via LAION-V2
and PickScore, selection/export. Everything runs on-device: no API keys, no cloud
round-trip, no telemetry. The same SigLIP embeddings power semantic search and
feed downstream scoring/curation so there's no second model pass for any of it.

Search with a text query like _"a dog on a beach at sunset"_ and get back the
closest-matching photos.

Three surfaces over one core (a warm local service that owns the model + index):

- **`muser`** — a CLI to index, search, benchmark.
- **`muser serve`** — the embedded service + a browser **search UI** at `http://127.0.0.1:7777`.
- **`muser-mcp`** — an [MCP](https://modelcontextprotocol.io) server (an _MCP App_) that
  searches and renders matches in an interactive gallery inside the host (e.g. Claude Desktop).

## How it works

Images are embedded with the model's **vision** encoder at index time; text queries with
its **text** encoder at search time. Both land in one shared space, so a description
retrieves visually matching images (cosine over L2-normalized vectors). State lives in a
single LanceDB at `~/.muser/db`, one table per model. Default model: **`siglip2-b`**
(Apache-2.0, strong quality/speed). See [`CLAUDE.md`](CLAUDE.md) for architecture and
[`REQUIREMENTS.md`](REQUIREMENTS.md) for scope.

## Requirements

- **Python 3.12** and [uv](https://docs.astral.sh/uv/)

## Install

```bash
git clone https://github.com/connerkward/Muser
cd Muser
uv pip install -e .
# Apple Silicon only, optional MLX backend: uv pip install -e '.[mac]'
```

First index/search downloads the model weights once (siglip2-b ≈ 1 GB).

## CLI usage

```bash
# Index a folder (recursive). Re-running is incremental (mtime-based; prunes deleted files).
uv run muser index ~/Pictures/screenshots

# Search the whole index, or scope to a folder
uv run muser search "a login screen with a blue button"
uv run muser search "sunset over water" --in ~/Pictures -k 5

# List models / run the retrieval-eval benchmark
uv run muser models
uv run muser bench

# Launch the embedded service + web search UI (http://127.0.0.1:7777)
uv run muser serve
```

Paths accept `~` (expands per-OS) and forward slashes on every platform.

## MCP usage

Add to your MCP host (e.g. Claude Desktop) config — the server is a console script, so
no path or runtime prefix is needed:

```json
{ "mcpServers": { "muser": { "command": "muser-mcp" } } }
```

| Tool            | Purpose                                                        |
| --------------- | -------------------------------------------------------------- |
| `index_folder`  | Index / re-index a folder of images.                           |
| `index_info`    | Report index stats.                                            |
| `search_images` | Search by description and open the gallery UI (an _MCP App_).  |

The MCP server is a thin HTTP client of `muser serve` and auto-spawns it if it's not
already running. Run over HTTP instead of stdio with `muser-mcp --http`
(`http://127.0.0.1:8000/mcp`).

## Platform support

Core (index, search, `serve`, `web`, CLI, MCP) is **cross-platform** (macOS, Windows,
Linux). The default install is portable; the `mac` extra (MLX backend) is Apple-Silicon
-only and auto-skips elsewhere via environment markers. Hardware acceleration: CUDA on
NVIDIA, Apple MPS on Apple Silicon, CPU otherwise (chosen automatically).

OS-integration niceties degrade gracefully when the platform tool is missing:

| Feature | macOS | Windows | Linux |
| --- | --- | --- | --- |
| Reveal in file manager | `open -R` | `explorer /select,` | FileManager1 D-Bus → `xdg-open` |
| Copy image to clipboard (Lens button) | `osascript` | PowerShell (WinForms) | `wl-copy` / `xclip` |
| Native folder picker | `osascript` | `FolderBrowserDialog` | `zenity` / `kdialog` |

On Linux, install `xclip` (X11) or `wl-clipboard` (Wayland) for the clipboard button, and
`zenity` or `kdialog` for the native folder picker; otherwise type a path into the scope box.

## Testing

A GitHub Actions matrix ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs the
test suite on **ubuntu / windows / macos** on every push: package install (incl. the
Apple-only extra resolving as a no-op off Apple Silicon), per-platform MLX gating, the
server-side clipboard branch (Linux runs under `xvfb` + `xclip`, so the real X11 path
executes), `_reveal`, the CLI entrypoints, and a CPU end-to-end index+search. Run locally:

```bash
uv run --with pytest --with pytest-timeout python -m pytest tests/ -v
```

### Known limitation

Folder-scoped search compares paths case-sensitively, but NTFS and APFS are
case-insensitive — so scoping with different casing than what was indexed (e.g.
`C:\Users\Me\Photos` vs a stored `...\me\photos`) can return no results. The fix needs a
normalized path-key column (a schema change + re-index); exact-case scoping works today.

## License

MIT
