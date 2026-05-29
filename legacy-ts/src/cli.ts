#!/usr/bin/env bun
// Muser CLI — index and search a folder of images from the terminal.

import path from "node:path";
import { Command } from "commander";
import { indexFolder, search, stats } from "./core";

const program = new Command();
program
  .name("muser")
  .description("Index, vectorize, and search a folder of images by natural language (CLIP).")
  .version("0.1.0");

program
  .command("index")
  .description("Index (or re-index) a folder of images")
  .argument("<folder>", "folder of images to index")
  .option("--no-recursive", "do not descend into subfolders")
  .action(async (folder: string, options: { recursive: boolean }) => {
    const started = Date.now();
    const res = await indexFolder(folder, {
      recursive: options.recursive,
      onProgress: (done, total, file) => {
        const line = `  embedding ${done}/${total}  ${path.basename(file)}`;
        process.stderr.write("\r" + line.padEnd(72).slice(0, 72));
      },
    });
    process.stderr.write("\r" + " ".repeat(72) + "\r");
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    console.log(
      `Indexed ${path.resolve(folder)}\n` +
        `  +${res.added} added · ${res.updated} updated · ${res.removed} removed · ${res.total} total  (${secs}s)`,
    );
  });

program
  .command("search")
  .description("Search an indexed folder by natural-language description")
  .argument("<query...>", 'what to look for, e.g. "a dog on a beach"')
  .requiredOption("--in <folder>", "folder that was previously indexed")
  .option("-k, --limit <n>", "number of results", "12")
  .option("--json", "output results as JSON")
  .action(async (queryParts: string[], options: { in: string; limit: string; json?: boolean }) => {
    const query = queryParts.join(" ");
    const hits = await search(options.in, query, parseInt(options.limit, 10));
    if (options.json) {
      console.log(JSON.stringify(hits, null, 2));
      return;
    }
    if (!hits.length) {
      console.log('No results. Have you run `muser index <folder>` on this folder yet?');
      return;
    }
    console.log(`Top ${hits.length} matches for "${query}":`);
    hits.forEach((h, i) => {
      console.log(`  ${String(i + 1).padStart(2)}.  ${(h.score * 100).toFixed(1)}%  ${h.path}`);
    });
  });

program
  .command("info")
  .description("Show index stats for a folder")
  .requiredOption("--in <folder>", "folder to inspect")
  .action(async (options: { in: string }) => {
    console.log(JSON.stringify(await stats(options.in), null, 2));
  });

program.parseAsync().catch((err) => {
  console.error(err instanceof Error ? err.message : String(err));
  process.exit(1);
});
