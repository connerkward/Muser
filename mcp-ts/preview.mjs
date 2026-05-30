#!/usr/bin/env node
/**
 * One-command local web preview / debug.
 *
 *   bun run preview      (or: node preview.mjs)
 *
 * Renders Muser's MCP App inside the OFFICIAL ext-apps "basic-host" — a real MCP
 * host in your browser with a tool picker, light/dark toggle, and inline/
 * fullscreen controls — so you can drive the search gallery exactly as a host
 * would. Steps:
 *   1. build the UI            (vite → dist/ui/search.html)
 *   2. start the MCP server    (bun src/mcp.ts --http → http://localhost:3939/mcp)
 *   3. on first run, clone + install the host under ext-apps/ (gitignored)
 *   4. build + serve the host on :8080, pointed at our server
 *
 * Then open http://localhost:8080, pick "search_images", fill in:
 *     folder = /absolute/path/to/your/indexed/folder
 *     query  = e.g. "a red circle"
 * and click Call. (Index a folder first with `bun src/cli.ts index <folder>`.)
 *
 * Install note: ext-apps is an npm-workspaces monorepo whose root `prepare` can
 * fail locally (husky), and its `serve.ts` needs the `bun` npm package's
 * postinstall. We install with Bun (tolerating the prepare error), run that one
 * postinstall, then build+serve the host directly with Bun.
 */
import { spawn, execSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const ROOT = import.meta.dirname;
const EXT_APPS = resolve(ROOT, "ext-apps");
const BASIC_HOST = resolve(EXT_APPS, "examples/basic-host");
const MCP_PORT = process.env.MCP_PORT ?? "3939";
const MCP_URL = `http://localhost:${MCP_PORT}/mcp`;
const HOST_PORT = process.env.HOST_PORT ?? "8080";
const SANDBOX_PORT = process.env.SANDBOX_PORT ?? "8081";
const children = [];

function run(cmd, args, opts = {}) {
  const c = spawn(cmd, args, { stdio: "inherit", ...opts });
  children.push(c);
  return c;
}
function shutdown() {
  for (const c of children) {
    try {
      c.kill("SIGINT");
    } catch {}
  }
  process.exit(0);
}
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

// 1) Build our UI.
console.log("→ Building UI…");
execSync("bun run build:ui", { cwd: ROOT, stdio: "inherit" });

// 2) Start our MCP server (HTTP mode).
console.log(`→ Starting MCP server on ${MCP_URL}`);
run("bun", ["src/mcp.ts", "--http"], { cwd: ROOT, env: { ...process.env, MCP_PORT } });

// 3) Ensure the official basic-host is present and installed.
if (!existsSync(EXT_APPS)) {
  console.log("→ First run: cloning official ext-apps basic-host…");
  execSync("git clone --depth 1 https://github.com/modelcontextprotocol/ext-apps.git", {
    cwd: ROOT,
    stdio: "inherit",
  });
}
if (!existsSync(resolve(EXT_APPS, "node_modules"))) {
  console.log("→ Installing ext-apps deps with Bun (one time, ~1 min)…");
  try {
    execSync("bun install", { cwd: EXT_APPS, stdio: "inherit" });
  } catch {
    console.log("  (root prepare script failed — expected; deps are installed)");
  }
  const bunPkg = resolve(EXT_APPS, "node_modules/bun");
  if (existsSync(resolve(bunPkg, "install.js"))) {
    console.log("→ Running the bun package's postinstall (needed by serve.ts)…");
    execSync("node install.js", { cwd: bunPkg, stdio: "inherit" });
  }
}

// 4) Build the host UI, then serve it directly with Bun.
console.log("→ Building basic-host…");
execSync("npm run build", {
  cwd: BASIC_HOST,
  stdio: "inherit",
  env: { ...process.env, NODE_ENV: "development" },
});

console.log("\n──────────────────────────────────────────────");
console.log(`  Open http://localhost:${HOST_PORT}`);
console.log('  Pick "search_images", set folder + query, click Call.');
console.log("  Ctrl+C to stop.");
console.log("──────────────────────────────────────────────\n");

run("bun", ["serve.ts"], {
  cwd: BASIC_HOST,
  env: { ...process.env, SERVERS: JSON.stringify([MCP_URL]), HOST_PORT, SANDBOX_PORT },
});
