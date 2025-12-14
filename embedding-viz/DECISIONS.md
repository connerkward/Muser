# Technical Decisions

## Architecture Overview

Web-based visualization layer for embeddings with two views:
1. **Topographical Map** - density contours showing semantic clusters
2. **Phylogeny Tree** - radial tree showing image similarity over time

## Stack Choices

### Embeddings: CLIP (openai/clip-vit-base-patch32)

**Why**: 
- Pre-trained on 400M image-text pairs - excellent general understanding
- 512-dim embeddings balance quality vs. size
- Runs locally (no API costs), fast on MPS/CUDA
- Easy to swap for larger models (ViT-L/14) if quality matters later

**Alternatives considered**:
- OpenAI API embeddings: Higher quality but costs money, network dependency
- ResNet/EfficientNet: Image-only, less semantic understanding
- DINOv2: Better for fine-grained visual similarity, but CLIP is more semantic

### Dimensionality Reduction: UMAP

**Why**:
- Preserves local structure better than t-SNE - clusters stay coherent
- Faster than t-SNE for 500+ points
- Deterministic with `random_state` - reproducible visualizations
- `metric='cosine'` matches our normalized embeddings

**Alternatives considered**:
- t-SNE: Classic but slow, harder to tune, non-deterministic
- PCA: Fast but loses non-linear relationships
- TriMap: Similar to UMAP but less mature ecosystem

### Clustering: HDBSCAN

**Why**:
- No need to specify k (number of clusters)
- Handles noise gracefully (labels outliers as -1)
- Works well with UMAP's density-preserving output
- `min_cluster_size=5` prevents tiny spurious clusters

**Alternatives considered**:
- K-Means: Requires choosing k, assumes spherical clusters
- DBSCAN: HDBSCAN is strictly better (hierarchical)
- Spectral: Better for non-convex shapes but slower

### Frontend: Vanilla JS + Vite

**Why**:
- Zero framework overhead - fast iteration
- D3 integrates cleanly without React reconciliation issues
- Vite's HMR is instant for rapid UI tweaks
- No build complexity, easy to understand

**Alternatives considered**:
- React: Adds complexity, D3 fights React's DOM control
- Svelte: Good option but less familiar ecosystem
- Observable: Great for notebooks but not standalone apps

### Visualization: D3.js

**Why**:
- Industry standard for custom data viz
- `d3-contour` does exactly what we need for topography
- `d3.stratify` + tree layouts handle phylogeny perfectly
- Massive ecosystem, easy to find examples

**Alternatives considered**:
- Three.js/WebGL: Overkill for 500 points, harder to debug
- Vega-Lite: Declarative but less control for custom viz
- Plotly: Higher-level but less customizable

## Data Flow

```
Images → CLIP → 512-dim embeddings → UMAP → 2D coords → HDBSCAN → clusters
                                          ↓
                              Similarity matrix → MST → Phylogeny tree
```

All processing happens offline in Python. Frontend loads static JSON - no backend server needed.

## Phylogeny Algorithm

The tree represents "meme evolution" - how similar images relate over time.

1. **Cosine similarity matrix**: Normalized CLIP embeddings, so dot product = similarity
2. **Constrained edges**: Only connect if:
   - Temporally close (< 30 days apart), OR
   - Highly similar (> 0.85 cosine similarity)
3. **MST**: Minimum spanning tree finds optimal connections
4. **Rooting**: Earliest timestamp becomes root - tree flows past → present

This prevents weird long-range connections while allowing genuine viral spread.

## Topographical Map Algorithm

Treats embedding density as "elevation" to create terrain-like visualization.

1. UMAP gives (x, y) coordinates
2. KDE estimates density at each point
3. `d3-contour` generates elevation lines
4. Cluster labels placed at density peaks

## Future Improvements

### Easy upgrades (drop-in):
- Switch CLIP to `ViT-L/14` for better embeddings
- Add image thumbnails to tooltips
- Use LLM to generate cluster labels from representative images

### Medium effort:
- Interactive cluster exploration (click to zoom)
- Time slider to filter phylogeny by date range
- Search/filter by visual similarity

### Architectural changes:
- WebGL rendering for 10k+ points
- Backend API for real-time embedding generation
- Persistent database instead of JSON

