#!/usr/bin/env bun
// Muser MCP entry point. Defaults to stdio (how most MCP hosts launch it);
// pass --http to serve a Streamable HTTP endpoint instead.

import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import http from "node:http";
import { createServer } from "./server";

const PORT = parseInt(process.env.MCP_PORT ?? "3939", 10);

if (process.argv.includes("--http")) {
  const httpServer = http.createServer((req, res) => {
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization, Mcp-Session-Id, Mcp-Protocol-Version",
    );
    res.setHeader("Access-Control-Expose-Headers", "Mcp-Session-Id");
    if (req.method === "OPTIONS") {
      res.writeHead(204);
      res.end();
      return;
    }
    if (req.url !== "/mcp") {
      res.writeHead(404);
      res.end();
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c as Buffer));
    req.on("end", async () => {
      const body = JSON.parse(Buffer.concat(chunks).toString() || "{}");
      const server = createServer();
      const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
      res.on("close", () => {
        transport.close();
        server.close();
      });
      await server.connect(transport);
      await transport.handleRequest(req, res, body);
    });
  });
  httpServer.listen(PORT, () => console.error(`Muser MCP on http://localhost:${PORT}/mcp`));
} else {
  const server = createServer();
  await server.connect(new StdioServerTransport());
  console.error("Muser MCP running on stdio");
}
