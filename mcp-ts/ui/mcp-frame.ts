/**
 * mcp-frame — the reusable host bridge for an MCP App UI.
 *
 * This file is the heart of the harness and you usually DON'T edit it when
 * building a new app. It owns the one thing that is easy to get wrong: making
 * the UI sit correctly inside the host's frame in every display mode.
 *
 * What it does for you:
 *   1. Connects to the MCP host over the iframe postMessage bridge.
 *   2. Mirrors the host's display mode onto `<html data-mcp-display="...">`
 *      ("inline" | "fullscreen" | "pip") so your CSS can lay out per mode.
 *   3. Exposes the host's container size + mobile safe-area insets as CSS vars
 *      (`--mcp-max-w`, `--mcp-max-h`, `--mcp-safe-*`).
 *   4. Applies the host's theme + style variables (and re-applies on change).
 *   5. Ships a tiny base stylesheet that makes frame-fitting "just work":
 *      inline = content-sized (host auto-grows the iframe), fullscreen = fills
 *      the host viewport with internal scrolling.
 *   6. Gives you `callTool`, `toggleFullscreen`, and JSON/text result helpers.
 *
 * THE ONE RULE that keeps inline mode from ballooning: in inline mode the host
 * grows the iframe to fit your natural content height (the SDK measures
 * `max-content`). So in inline mode never pin the root to `100vh`/`100%` — let
 * it flow. Fullscreen is the opposite: the host fixes the viewport, so you fill
 * it. The base stylesheet below already encodes this; your app CSS only needs
 * to switch *layout* (e.g. stacked vs. two-column) by display mode.
 */
import {
  App,
  applyDocumentTheme,
  applyHostStyleVariables,
  applyHostFonts,
  type McpUiHostContext,
  type McpUiDisplayMode,
} from "@modelcontextprotocol/ext-apps";

export type { McpUiHostContext, McpUiDisplayMode };

type ToolContentBlock = { type: string; text?: string; [k: string]: unknown };
type ToolResult = {
  content?: ToolContentBlock[];
  structuredContent?: Record<string, unknown>;
  isError?: boolean;
};

export interface ConnectOptions {
  /** App identity reported to the host. */
  name: string;
  version?: string;
  /** Called with the host's tool-call arguments when the app opens. */
  onToolInput?: (args: Record<string, unknown> | undefined) => void;
  /** Called with the result of the tool that opened the app. */
  onToolResult?: (result: ToolResult) => void;
  /** Called if the opening tool call is cancelled by the host. */
  onToolCancelled?: (reason?: string) => void;
  /** Called whenever the host context changes (theme, size, display mode…). */
  onContextChanged?: (ctx: McpUiHostContext) => void;
  /** Give up waiting for a host after this many ms (so the UI can render
   *  standalone for local inspection). Default 1500. */
  connectTimeoutMs?: number;
}

export interface McpFrame {
  /** The underlying SDK App (escape hatch for advanced calls). */
  app: App;
  /** True if a host actually answered the handshake. */
  connected: boolean;
  /** Latest host context, or undefined when running standalone. */
  ctx(): McpUiHostContext | undefined;
  /** Current display mode (defaults to "inline" when standalone). */
  displayMode(): McpUiDisplayMode;
  /** Whether the host offers a fullscreen mode to toggle into. */
  canFullscreen(): boolean;
  /** Ask the host to switch display mode. No-op if unsupported. */
  toggleFullscreen(): Promise<void>;
  setDisplayMode(mode: McpUiDisplayMode): Promise<void>;
  /** Call a tool on this app's MCP server and get the parsed result back. */
  callTool(name: string, args?: Record<string, unknown>): Promise<ToolResult>;
}

/** First text block of a tool result. */
export function resultText(r: ToolResult): string | undefined {
  return r.content?.find((c) => c.type === "text")?.text;
}

/** Parse a tool result as JSON — from structuredContent first, else the text block. */
export function resultJson<T = unknown>(r: ToolResult): T | undefined {
  if (r.structuredContent) return r.structuredContent as T;
  const txt = resultText(r);
  if (!txt) return undefined;
  try {
    return JSON.parse(txt) as T;
  } catch {
    return undefined;
  }
}

const BASE_CSS = `
:root { color-scheme: light dark; }
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  /* Honour mobile safe areas the host reports. */
  padding-top: var(--mcp-safe-top, 0px);
  padding-right: var(--mcp-safe-right, 0px);
  padding-bottom: var(--mcp-safe-bottom, 0px);
  padding-left: var(--mcp-safe-left, 0px);
}

/* INLINE (minimized): the host grows the iframe to fit content height.
   Keep content natural-height; just stop it from stretching absurdly wide. */
html[data-mcp-display="inline"] body {
  max-width: var(--mcp-max-w, 720px);
  margin-inline: auto;
}

/* FULLSCREEN / PIP: the host fixes the viewport. Fill it and scroll inside. */
html[data-mcp-display="fullscreen"],
html[data-mcp-display="pip"] { height: 100%; }
html[data-mcp-display="fullscreen"] body,
html[data-mcp-display="pip"] body {
  min-height: 100dvh;
  max-width: none;
}
`;

function injectBaseCss() {
  if (document.getElementById("__mcp_frame_base")) return;
  const style = document.createElement("style");
  style.id = "__mcp_frame_base";
  style.textContent = BASE_CSS;
  // Prepend so the app's own stylesheet always wins on specificity ties.
  document.head.prepend(style);
}

function px(n: number | undefined): string | undefined {
  return typeof n === "number" ? `${Math.round(n)}px` : undefined;
}

function applyContext(ctx: McpUiHostContext | undefined) {
  if (!ctx) return;
  const root = document.documentElement;

  if (ctx.theme) applyDocumentTheme(ctx.theme);
  if (ctx.styles?.variables) applyHostStyleVariables(ctx.styles.variables);
  if (ctx.styles?.css?.fonts) applyHostFonts(ctx.styles.css.fonts);

  if (ctx.displayMode) root.dataset.mcpDisplay = ctx.displayMode;
  if (ctx.platform) root.dataset.mcpPlatform = ctx.platform;
  root.dataset.mcpCanFullscreen = String(
    (ctx.availableDisplayModes ?? []).includes("fullscreen"),
  );

  const dim = ctx.containerDimensions as
    | { width?: number; maxWidth?: number; height?: number; maxHeight?: number }
    | undefined;
  if (dim) {
    const maxW = px(dim.width ?? dim.maxWidth);
    const maxH = px(dim.height ?? dim.maxHeight);
    if (maxW) root.style.setProperty("--mcp-max-w", maxW);
    if (maxH) root.style.setProperty("--mcp-max-h", maxH);
  }

  const safe = ctx.safeAreaInsets;
  if (safe) {
    root.style.setProperty("--mcp-safe-top", px(safe.top)!);
    root.style.setProperty("--mcp-safe-right", px(safe.right)!);
    root.style.setProperty("--mcp-safe-bottom", px(safe.bottom)!);
    root.style.setProperty("--mcp-safe-left", px(safe.left)!);
  }
}

/**
 * Connect to the host and return a ready-to-use frame.
 *
 * Always resolves: if no host answers within `connectTimeoutMs`, you get a
 * frame with `connected: false` so the UI still renders for local inspection.
 */
export async function connectMcpFrame(opts: ConnectOptions): Promise<McpFrame> {
  injectBaseCss();
  // Sensible default so layout is correct before the first context arrives.
  document.documentElement.dataset.mcpDisplay = "inline";

  const app = new App({ name: opts.name, version: opts.version ?? "1.0.0" });

  // Handlers must be registered before connect().
  if (opts.onToolInput) app.ontoolinput = (p) => opts.onToolInput!(p.arguments);
  if (opts.onToolResult)
    app.ontoolresult = (r) => opts.onToolResult!(r as ToolResult);
  if (opts.onToolCancelled)
    app.ontoolcancelled = (p) => opts.onToolCancelled!(p.reason);
  app.onhostcontextchanged = (ctx) => {
    applyContext(ctx as McpUiHostContext);
    opts.onContextChanged?.(ctx as McpUiHostContext);
  };

  let connected = false;
  const timeoutMs = opts.connectTimeoutMs ?? 1500;
  try {
    await Promise.race([
      app.connect(),
      new Promise((_, rej) =>
        setTimeout(() => rej(new Error("no host")), timeoutMs),
      ),
    ]);
    connected = true;
    applyContext(app.getHostContext());
  } catch {
    connected = false; // standalone: render with defaults
  }
  // Expose host-connection state to CSS so an app can frame itself differently
  // when embedded (host provides the frame) vs. opened standalone.
  document.documentElement.dataset.mcpHost = String(connected);

  const frame: McpFrame = {
    app,
    connected,
    ctx: () => app.getHostContext(),
    displayMode: () => app.getHostContext()?.displayMode ?? "inline",
    canFullscreen: () =>
      (app.getHostContext()?.availableDisplayModes ?? []).includes(
        "fullscreen",
      ),
    async setDisplayMode(mode) {
      if (!connected) return;
      try {
        await app.requestDisplayMode({ mode });
        applyContext(app.getHostContext());
      } catch {
        /* host declined */
      }
    },
    async toggleFullscreen() {
      const next: McpUiDisplayMode =
        frame.displayMode() === "fullscreen" ? "inline" : "fullscreen";
      await frame.setDisplayMode(next);
    },
    async callTool(name, args = {}) {
      const r = await app.callServerTool({ name, arguments: args });
      return r as ToolResult;
    },
  };

  return frame;
}
