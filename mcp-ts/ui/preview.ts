// Standalone UI preview — renders the real gallery with mock data so the look
// and interactions can be iterated on in a browser (Vite dev server, HMR)
// without an MCP host or the embedding backend. Run with `bun run dev:ui`.

import { createGallery, type Hit } from "./gallery";
import redCircle from "./sample/red-circle.png";
import blueSquare from "./sample/blue-square.png";
import greenTriangle from "./sample/green-triangle.png";
import yellowStar from "./sample/yellow-star.png";
import purpleHeart from "./sample/purple-heart.png";

const FOLDER = "/demo/sample-images";

const SAMPLE: Hit[] = [
  {
    path: `${FOLDER}/red-circle.png`,
    score: 0.29,
    thumb: redCircle,
    dupe_count: 3,
    dupes: [
      `${FOLDER}/red-circle.png`,
      `${FOLDER}/backup/red-circle.png`,
      `${FOLDER}/archive/2024/red-circle-copy.png`,
    ],
    dupeThumbs: {
      [`${FOLDER}/red-circle.png`]: redCircle,
      [`${FOLDER}/backup/red-circle.png`]: redCircle,
      [`${FOLDER}/archive/2024/red-circle-copy.png`]: redCircle,
    },
  },
  { path: `${FOLDER}/blue-square.png`, score: 0.27, thumb: blueSquare },
  { path: `${FOLDER}/green-triangle.png`, score: 0.26, thumb: greenTriangle },
  { path: `${FOLDER}/yellow-star.png`, score: 0.25, thumb: yellowStar },
  { path: `${FOLDER}/purple-heart.png`, score: 0.24, thumb: purpleHeart },
];

// Naive keyword ranking against the filename so the search box feels live.
function mockSearch(query: string): Hit[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  return SAMPLE.map((h) => {
    const name = h.path.toLowerCase();
    const matches = terms.filter((t) => name.includes(t)).length;
    const score = Math.min(0.45, 0.18 + matches * 0.12 + Math.random() * 0.04);
    return { ...h, score };
  }).sort((a, b) => b.score - a.score);
}

const gallery = createGallery({
  onSearch: (query) => {
    gallery.setStatus("Searching… (mock)");
    setTimeout(() => gallery.render({ folder: FOLDER, query, results: mockSearch(query) }), 150);
  },
  onSelect: (hit) => gallery.setStatus(`(preview) selected ${hit.path}`),
  onReveal: (path) => gallery.setStatus(`(preview) reveal ${path}`),
});

gallery.render({ folder: FOLDER, query: "a red circle", results: SAMPLE });
