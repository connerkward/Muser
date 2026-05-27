---
"muser": minor
---

Initial release. Index, vectorize, and search a folder of images by natural language using on-device CLIP embeddings (`@huggingface/transformers`) stored in LanceDB. Ships a `muser` CLI (`index` / `search` / `info`) and a `muser-mcp` MCP App server exposing `index_folder`, `index_info`, and a `search_images` tool that renders matches in an interactive ext-apps gallery.
