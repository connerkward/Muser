# Embedding Visualization

Interactive visualization of image and text embeddings using CLIP and sentence-transformers, rendered as a topographical map and network graph.

<video src="media/Screen Recording 2025-12-20 at 02.49.05.mov" controls width="100%"></video>

## Features

### Dual Mode Visualization

Toggle between **Image (IMG)** and **Text (TXT)** modes to visualize different embedding types:

![Main Image View](media/Screenshot%202025-12-13%20at%2019.17.32.png)

**Image Mode**: Visualizes image embeddings from CLIP model, showing semantic clusters of similar images with labeled categories like:
- FURNITURE / PRODUCT PHOTO
- SCREENSHOT / ILLUSTRATION
- UI DESIGN / POSTER
- PRODUCT PHOTO / TECHNOLOGY
- PRODUCT PHOTO / FASHION
- MILITARY / VEHICLE
- VEHICLE / CAR

![Main Text View](media/Screenshot%202025-12-13%20at%2023.45.21.png)

**Text Mode**: Visualizes text embeddings from sentence-transformers, clustering related concepts such as:
- PHILOSOPHY
- CHORDS
- APPLE
- WALL
- WOOD
- SOIL

### Topographical Map View

The main view renders embeddings as a topographic map with:
- **Colored clusters**: Semi-transparent nodes grouped by similarity
- **Contour lines**: Light blue elevation lines showing embedding density
- **Cluster labels**: White text labels identifying semantic categories
- **Color coding**: Different colors represent distinct clusters or categories

### Detailed Cluster Views

Clicking on clusters reveals detailed views:

![Image Detail View](media/Screenshot%202025-12-13%20at%2019.30.06.png)

**Image Detail**: Shows actual product images connected by curved lines, indicating similarity relationships. Features:
- Thumbnail grid of related images
- Connection lines showing proximity in embedding space
- Category focus (e.g., "PRODUCT PHOTO / FASHION")

![Text Detail View](media/Screenshot%202025-12-13%20at%2023.48.59.png)

**Text Detail**: Displays interconnected text nodes in a network graph:
- Rectangular nodes containing document titles and metadata
- Teal connection lines showing semantic relationships
- Creation/update timestamps
- Text previews and excerpts
- Pink-highlighted nodes indicating selected items

## Architecture

See [DECISIONS.md](DECISIONS.md) for technical details.

## Setup

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:3000

### Pipeline

```bash
cd pipeline
uv run python main.py --input /path/to/images --mode image
```

## Usage

1. Generate embeddings using the pipeline
2. Place JSON output in `frontend/public/data/`
3. Open the app in browser
4. Toggle between IMG/TXT modes
5. Click clusters to explore detailed views
