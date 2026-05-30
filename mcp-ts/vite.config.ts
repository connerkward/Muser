import { defineConfig } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

// Bundles ui/search.html + its TS/CSS into a single self-contained HTML file
// at dist/ui/search.html, which the MCP server serves as an ext-apps resource.
export default defineConfig({
  plugins: [viteSingleFile()],
  build: {
    outDir: "dist",
    emptyOutDir: false,
    rollupOptions: {
      input: "ui/search.html",
    },
  },
});
