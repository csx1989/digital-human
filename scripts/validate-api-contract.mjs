#!/usr/bin/env node
// scripts/validate-api-contract.mjs
// Lightweight, dependency-free contract drift guard for the v1 API surface.
//
// Usage:
//   node scripts/validate-api-contract.mjs
//
// Limitations:
// - This script parses TypeScript and YAML with conservative regexes, not a
//   full parser. It is designed to catch accidental drift, not to prove
//   type-correctness.
// - Error-code comparison uses the ApiErrorCode union in src/types.ts. Runtime
//   code may throw string literals that are not declared there.
// - WebSocket message cross-check verifies that each source type literal is
//   *mentioned* in docs/WEBSOCKET_PROTOCOL.md, not that it is fully documented.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, "..");

const DOCS_DIR = path.join(ROOT, "docs");
const EXAMPLES_DIR = path.join(ROOT, "examples");
const BRIDGE_SRC_DIR = path.join(
  ROOT,
  "avatar-saas-prototype",
  "cloudflare-session-bridge",
  "src",
);

const OPENAPI_PATH = path.join(DOCS_DIR, "openapi-v1.yaml");
const TYPES_PATH = path.join(BRIDGE_SRC_DIR, "types.ts");

const REQUIRED_DOCS = [
  "docs/API_V1.md",
  "docs/API_QUICKSTART.md",
  "docs/WEBSOCKET_PROTOCOL.md",
  "docs/API_SECURITY_VALIDATION.md",
];

const REQUIRED_OPENAPI_PATHS = [
  "/v1/health",
  "/v1/me",
  "/v1/providers",
  "/v1/avatars",
  "/v1/avatars/{avatar_id}",
  "/v1/backgrounds",
  "/v1/backgrounds/{bg_id}",
  "/v1/sessions",
  "/v1/sessions/{session_id}/start",
  "/v1/sessions/{session_id}/status",
  "/v1/sessions/{session_id}",
  "/v1/sessions/{session_id}/end",
  "/v1/sessions/{session_id}/stream",
  "/v1/sessions/{session_id}/ws-token",
  "/v1/sessions/{session_id}/ws",
  // SFU subscriber endpoints (browser-facing, sfu_token auth).
  // These are dormant when SFU is not configured but must exist in the
  // contract so the runtime endpoints are documented.
  "/v1/sessions/{session_id}/sfu/subscribe",
  "/v1/sessions/{session_id}/sfu/answer",
  "/v1/sessions/{session_id}/sfu/close",
];

const results = [];
let failed = false;

function log(severity, message) {
  results.push({ severity, message });
}

function assertTrue(condition, okMessage, failMessage) {
  if (condition) {
    log("ok", okMessage);
  } else {
    failed = true;
    log("fail", failMessage);
  }
}

function readText(filePath) {
  return fs.readFileSync(filePath, "utf8");
}

function fileExists(relativePath) {
  return fs.existsSync(path.join(ROOT, relativePath));
}

function extractBlock(source, startMarker, endMarkers) {
  const startIdx = source.indexOf(startMarker);
  if (startIdx === -1) return null;
  const afterStart = startIdx + startMarker.length;
  let endIdx = source.length;
  for (const marker of endMarkers) {
    const idx = source.indexOf(marker, afterStart);
    if (idx !== -1 && idx < endIdx) endIdx = idx;
  }
  return source.slice(afterStart, endIdx);
}

function extractStringLiterals(block) {
  const out = [];
  const re = /"([^"]+)"/g;
  let m;
  while ((m = re.exec(block)) !== null) {
    out.push(m[1]);
  }
  return out;
}

function extractTypeLiterals(block) {
  const out = [];
  const re = /type\s*:\s*"([^"]+)"/g;
  let m;
  while ((m = re.exec(block)) !== null) {
    out.push(m[1]);
  }
  return [...new Set(out)];
}

function extractOpenApiPaths(yaml) {
  const out = [];
  const re = /^  (\/v1\/[^\s:]+):$/gm;
  let m;
  while ((m = re.exec(yaml)) !== null) {
    out.push(m[1]);
  }
  return out;
}

function extractOpenApiErrorCodes(yaml) {
  const apiErrorBodyMatch = yaml.match(
    /ApiErrorBody:[\s\S]*?properties:[\s\S]*?code:[\s\S]*?enum:\s*([\s\S]*?)(?=\n\S|$)/,
  );
  if (!apiErrorBodyMatch) return null;
  const enumBlock = apiErrorBodyMatch[1];
  const out = [];
  const re = /^\s*-\s*(\S+)\s*$/gm;
  let m;
  while ((m = re.exec(enumBlock)) !== null) {
    out.push(m[1]);
  }
  return out;
}

function extractMarkdownLinks(docText, docRelativePath) {
  const links = [];
  const re = /\[([^\]]*)\]\(([^)]+)\)/g;
  let m;
  while ((m = re.exec(docText)) !== null) {
    const target = m[2].trim();
    if (
      target.startsWith("http://") ||
      target.startsWith("https://") ||
      target.startsWith("mailto:") ||
      target.startsWith("#")
    ) {
      continue;
    }
    const [beforeHash] = target.split("#");
    const docDir = path.dirname(path.join(ROOT, docRelativePath));
    const resolved = path.resolve(docDir, beforeHash);
    links.push({ raw: target, resolved });
  }
  return links;
}

function scanForRealKeys(dir, label) {
  const issues = [];
  if (!fs.existsSync(dir)) return issues;
  const files = [];
  function walk(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && /\.(md|mjs|js|ts|json|txt|yaml|yml)$/.test(entry.name)) {
        files.push(full);
      }
    }
  }
  walk(dir);

  const re = /avtr_(?:test|live)_[a-f0-9]{10,}/gi;
  for (const file of files) {
    const text = readText(file);
    let m;
    while ((m = re.exec(text)) !== null) {
      issues.push(`${path.relative(ROOT, file)}:${lineNumber(text, m.index)} ${m[0]}`);
    }
  }
  return issues;
}

function scanForFullTokens(dir, label) {
  const issues = [];
  if (!fs.existsSync(dir)) return issues;
  const files = [];
  function walk(current) {
    for (const entry of fs.readdirSync(current, { withFileTypes: true })) {
      const full = path.join(current, entry.name);
      if (entry.isDirectory()) walk(full);
      else if (entry.isFile() && /\.(md|mjs|js|ts|json|txt|yaml|yml)$/.test(entry.name)) {
        files.push(full);
      }
    }
  }
  walk(dir);

  const re = /token=[A-Za-z0-9_-]{20,}/g;
  for (const file of files) {
    const text = readText(file);
    let m;
    while ((m = re.exec(text)) !== null) {
      const after = text.slice(m.index + m[0].length, m.index + m[0].length + 3);
      if (after === "...") continue; // clearly truncated
      issues.push(`${path.relative(ROOT, file)}:${lineNumber(text, m.index)} ${m[0].slice(0, 40)}...`);
    }
  }
  return issues;
}

function lineNumber(text, index) {
  return text.slice(0, index).split("\n").length;
}

function main() {
  log("info", "API contract drift guard starting...");

  // 1. Required files exist
  for (const rel of REQUIRED_DOCS) {
    assertTrue(
      fileExists(rel),
      `Required doc exists: ${rel}`,
      `Missing required doc: ${rel}`,
    );
  }
  assertTrue(
    fs.existsSync(OPENAPI_PATH),
    `OpenAPI spec exists: docs/openapi-v1.yaml`,
    `Missing OpenAPI spec: docs/openapi-v1.yaml`,
  );
  assertTrue(
    fs.existsSync(TYPES_PATH),
    `Type definitions exist: src/types.ts`,
    `Missing type definitions: src/types.ts`,
  );

  const openapi = readText(OPENAPI_PATH);
  const types = readText(TYPES_PATH);

  // 2. Required OpenAPI paths
  const openapiPaths = extractOpenApiPaths(openapi);
  const missingPaths = REQUIRED_OPENAPI_PATHS.filter((p) => !openapiPaths.includes(p));
  assertTrue(
    missingPaths.length === 0,
    `All required OpenAPI paths present (${REQUIRED_OPENAPI_PATHS.length})`,
    `Missing OpenAPI paths: ${missingPaths.join(", ")}`,
  );

  // 3. Error-code cross-check
  const apiErrorBlock = extractBlock(
    types,
    "export type ApiErrorCode =",
    ["export interface", "export type"],
  );
  const tsErrorCodes = apiErrorBlock ? extractStringLiterals(apiErrorBlock) : null;
  const yamlErrorCodes = extractOpenApiErrorCodes(openapi);

  if (tsErrorCodes && yamlErrorCodes) {
    const tsSet = new Set(tsErrorCodes);
    const yamlSet = new Set(yamlErrorCodes);
    const onlyInTs = [...tsSet].filter((c) => !yamlSet.has(c));
    const onlyInYaml = [...yamlSet].filter((c) => !tsSet.has(c));
    assertTrue(
      onlyInTs.length === 0 && onlyInYaml.length === 0,
      `ApiErrorCode matches OpenAPI enum (${tsSet.size} codes)`,
      `ApiErrorCode drift: only in TS [${onlyInTs.join(", ")}], only in OpenAPI [${onlyInYaml.join(", ")}]`,
    );
  } else {
    failed = true;
    if (!tsErrorCodes) log("fail", "Could not extract ApiErrorCode from src/types.ts");
    if (!yamlErrorCodes) log("fail", "Could not extract error enum from docs/openapi-v1.yaml");
  }

  // 3b. SFU transport enum guard
  // If the ApiErrorCode union includes SFU codes (sfu_unavailable, etc.),
  // then V1Stream.transport in OpenAPI must include "webrtc_sfu" as a valid
  // transport value. This catches the case where SFU error codes are added
  // but the transport enum is not updated.
  const sfuCodePattern = /^sfu_/;
  const hasSfuErrorCodes = tsErrorCodes
    ? tsErrorCodes.some((c) => sfuCodePattern.test(c))
    : false;
  if (hasSfuErrorCodes) {
    const v1StreamBlock = extractBlock(openapi, "V1Stream:", ["V1StreamSfu:", "WsTokenResponse:"]);
    const v1StreamHasSfu = v1StreamBlock
      ? /transport:[\s\S]*?enum:\s*\[[^\]]*webrtc_sfu/.test(v1StreamBlock)
      : false;
    assertTrue(
      v1StreamHasSfu,
      "V1Stream.transport includes webrtc_sfu (SFU error codes are present)",
      "V1Stream.transport enum is missing webrtc_sfu but ApiErrorCode contains sfu_* codes",
    );
  }

  // 4. WebSocket message cross-check
  const clientBlock = extractBlock(types, "export type ClientMessage =", [
    "export type ServerMessage =",
  ]);
  const serverBlock = extractBlock(types, "export type ServerMessage =", [
    "export interface SessionState",
    "export type",
    "export interface",
  ]);
  const clientTypes = clientBlock ? extractTypeLiterals(clientBlock) : [];
  const serverTypes = serverBlock ? extractTypeLiterals(serverBlock) : [];
  const wsProtocolPath = path.join(DOCS_DIR, "WEBSOCKET_PROTOCOL.md");
  const wsProtocol = fs.existsSync(wsProtocolPath) ? readText(wsProtocolPath) : "";

  const wsUndocumented = [];
  for (const t of [...clientTypes, ...serverTypes]) {
    const mentioned =
      wsProtocol.includes(`"${t}"`) || wsProtocol.includes(`\`${t}\``);
    if (!mentioned) wsUndocumented.push(t);
  }
  assertTrue(
    wsUndocumented.length === 0,
    `All ClientMessage/ServerMessage types documented (${clientTypes.length + serverTypes.length})`,
    `WS message types missing from docs/WEBSOCKET_PROTOCOL.md: ${wsUndocumented.join(", ")}`,
  );

  // 4b. WebSocket error codes cross-check
  // Extract error codes sent by #sendError in avatar-session-do.ts and verify
  // they appear in WEBSOCKET_PROTOCOL.md error tables.
  const doSourcePath = path.join(BRIDGE_SRC_DIR, "avatar-session-do.ts");
  const doSource = fs.existsSync(doSourcePath) ? readText(doSourcePath) : "";
  const sendErrorMatches = [...doSource.matchAll(/#sendError\(\s*\w+\s*,\s*"([^"]+)"/g)];
  const wsErrorCodes = [...new Set(sendErrorMatches.map((m) => m[1]))];
  const wsUndocumentedErrors = [];
  for (const code of wsErrorCodes) {
    const mentioned = wsProtocol.includes(`| \`${code}\``) || wsProtocol.includes(`| "${code}"`);
    if (!mentioned) wsUndocumentedErrors.push(code);
  }
  assertTrue(
    wsUndocumentedErrors.length === 0,
    `All WS error codes from avatar-session-do.ts documented (${wsErrorCodes.length})`,
    `WS error codes missing from docs/WEBSOCKET_PROTOCOL.md: ${wsUndocumentedErrors.join(", ")}`,
  );

  // 5. Docs local links resolve
  const linkFailures = [];
  for (const rel of REQUIRED_DOCS) {
    const text = readText(path.join(ROOT, rel));
    for (const link of extractMarkdownLinks(text, rel)) {
      if (!fs.existsSync(link.resolved)) {
        linkFailures.push(`${rel} -> ${link.raw} (resolved ${path.relative(ROOT, link.resolved)})`);
      }
    }
  }
  assertTrue(
    linkFailures.length === 0,
    "All local markdown links in API docs resolve",
    `Broken local links: ${linkFailures.join("; ")}`,
  );

  // 6. OpenAPI must not leak modal_session_id
  assertTrue(
    !/modal_session_id/i.test(openapi),
    "OpenAPI spec does not contain modal_session_id",
    "OpenAPI spec leaks modal_session_id",
  );

  // 7. Key/token leak checks
  const keyIssues = [
    ...scanForRealKeys(DOCS_DIR, "docs"),
    ...scanForRealKeys(EXAMPLES_DIR, "examples"),
  ];
  assertTrue(
    keyIssues.length === 0,
    "No real-looking API keys in docs/examples",
    `Potential API key leaks:\n  ${keyIssues.join("\n  ")}`,
  );

  const tokenIssues = [
    ...scanForFullTokens(DOCS_DIR, "docs"),
    ...scanForFullTokens(EXAMPLES_DIR, "examples"),
  ];
  assertTrue(
    tokenIssues.length === 0,
    "No full ws_url tokens in docs/examples",
    `Potential full ws_url token leaks:\n  ${tokenIssues.join("\n  ")}`,
  );

  // 8. WebSocket protocol doc content checks
  assertTrue(
    wsProtocol.includes("## Stability"),
    "WEBSOCKET_PROTOCOL.md contains a Stability section",
    "WEBSOCKET_PROTOCOL.md missing Stability section",
  );
  // Check that `fatal` is described as optional in the error table/semantics.
  const fatalSection = wsProtocol.toLowerCase();
  const fatalOptionalWording =
    fatalSection.includes("fatal") &&
    (fatalSection.includes("optional") || fatalSection.includes("if omitted"));
  assertTrue(
    fatalOptionalWording,
    "WEBSOCKET_PROTOCOL.md documents fatal as optional",
    "WEBSOCKET_PROTOCOL.md does not clearly document fatal as optional",
  );

  // 9. Quickstart token refresh wording
  const quickstartPath = path.join(DOCS_DIR, "API_QUICKSTART.md");
  const quickstart = fs.existsSync(quickstartPath) ? readText(quickstartPath) : "";
  const badRefreshClaim =
    /revoke\s+old\s+tokens?/i.test(quickstart) ||
    /refresh(?:ing)?[^.\n]{0,60}\brevoke\b[^.\n]{0,60}\bold\b/i.test(quickstart);
  assertTrue(
    !badRefreshClaim,
    "API_QUICKSTART.md does not claim refresh revokes old tokens",
    "API_QUICKSTART.md contains a refresh-revokes-old-tokens claim",
  );

  // Print summary
  console.log("\n--- API Contract Validation Results ---\n");
  for (const r of results) {
    const prefix = r.severity === "ok" ? "✓" : r.severity === "fail" ? "✗" : "ℹ";
    console.log(`${prefix} ${r.message}`);
  }
  console.log("\n---------------------------------------");
  if (failed) {
    console.log("RESULT: FAIL");
    process.exit(1);
  } else {
    console.log("RESULT: PASS");
    process.exit(0);
  }
}

main();
