#!/usr/bin/env node

/**
 * Example Avatar API v1 client — dependency-free demo CLI.
 *
 * Basic flow (Node 18+): create session -> read status -> delete session.
 * With --ws (Node 22+): also opens the WebSocket and sends session.start.
 *
 * Usage:
 *   AVTR_API_KEY=... node client.mjs
 *   AVTR_API_KEY=... node client.mjs --ws
 *   AVTR_API_KEY=... AVTR_API_BASE=https://staging... node client.mjs
 *
 * All output is REDACTED — no full API keys or full ws_url values are printed.
 */

import { AvtrClient, AvtrApiError, redactApiKey, redactWsUrl } from "./avtr-client.mjs";

// ── Config ─────────────────────────────────────────────────────────────────

const API_KEY = process.env.AVTR_API_KEY;
const BASE_URL = process.env.AVTR_API_BASE;
const OPEN_WS = process.argv.includes("--ws");

if (!API_KEY) {
  console.error("Error: AVTR_API_KEY environment variable is required.");
  console.error("Example: AVTR_API_KEY=avtr_test_... node client.mjs");
  process.exit(1);
}

const client = new AvtrClient({ apiKey: API_KEY, baseUrl: BASE_URL });

// ── Runner ─────────────────────────────────────────────────────────────────

async function main() {
  console.log("=== Avatar API v1 Example Client ===");
  console.log(`Base URL : ${client.baseUrl}`);
  console.log(`API Key  : ${redactApiKey(API_KEY)}`);
  console.log(`WS Mode  : ${OPEN_WS ? "enabled (--ws)" : "disabled"}`);
  console.log();

  // Feature-check WebSocket if --ws is requested
  if (OPEN_WS && typeof globalThis.WebSocket === "undefined") {
    console.error("Error: --ws requires Node 22+ (global WebSocket).");
    console.error("Current Node version:", process.version);
    console.error("Run without --ws for the basic create/status/delete flow.");
    process.exit(1);
  }

  let sessionId = null;

  try {
    // Step 1: Verify auth
    console.log("[1/4] Verifying API key...");
    const me = await client._request("GET", "/v1/me");
    console.log(`  tenant_id : ${me.data.tenant_id}`);
    console.log(`  auth_mode : ${me.data.auth_mode}`);
    console.log(`  key_name  : ${me.data.key_name}`);
    console.log();

    // Step 2: Create session
    console.log("[2/4] Creating session...");
    const session = await client.createSession();
    sessionId = session.session_id;
    console.log(`  session_id : ${sessionId}`);
    console.log(`  status     : ${session.status}`);
    console.log(`  ws_url     : ${redactWsUrl(session.stream?.ws_url)}`);
    console.log(`  token_exp  : ${session.stream?.token_expires_at ?? "n/a"}`);
    console.log();

    // Step 3: Check status
    console.log("[3/4] Reading session status...");
    const status = await client.getSessionStatus(sessionId);
    console.log(`  session_id : ${status.session_id}`);
    console.log(`  status     : ${status.status}`);
    console.log(`  created_at : ${status.created_at ?? "n/a"}`);
    console.log();

    // Optional: Open WebSocket and send session.start
    if (OPEN_WS && session.stream?.ws_url) {
      console.log("[WS] Connecting to WebSocket...");
      await connectWs(session.stream.ws_url, session.session_id);
      console.log();
    }

    // Step 4: Cleanup
    console.log("[4/4] Ending session...");
    const ended = await client.endSession(sessionId);
    console.log(`  session_id : ${ended.session_id}`);
    console.log(`  status     : ${ended.status}`);
    console.log(`  end_ok     : ${ended.end_ok ?? "n/a"}`);
    sessionId = null; // mark as cleaned up
    console.log();

    console.log("=== Done ===");

  } catch (err) {
    console.error();
    if (err instanceof AvtrApiError) {
      console.error(`API Error: [${err.status}] ${err.code}`);
      console.error(`  Message   : ${err.message}`);
      if (err.requestId) console.error(`  Request ID: ${err.requestId}`);
    } else {
      console.error(`Unexpected error: ${err.message}`);
    }

    // Best-effort cleanup
    if (sessionId) {
      console.error("\nAttempting cleanup...");
      try {
        await client.endSession(sessionId);
        console.error("  Session ended successfully.");
      } catch (cleanupErr) {
        console.error(`  Cleanup failed: ${cleanupErr.message}`);
      }
    }

    process.exit(1);
  }
}

/**
 * Open a WebSocket connection, send session.start, and log incoming messages.
 * Closes after receiving session.ready or after a 10-second timeout.
 *
 * @param {string} wsUrl - Full ws_url with token (used as-is, never logged).
 * @param {string} sessionId - For display only.
 */
async function connectWs(wsUrl, sessionId) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    let resolved = false;
    const timeout = setTimeout(() => {
      if (!resolved) {
        resolved = true;
        console.log("  [WS] Timeout (10s) — closing.");
        try { ws.close(1000, "timeout"); } catch {}
        resolve();
      }
    }, 10000);

    ws.onopen = () => {
      console.log("  [WS] Connected. Sending session.start...");
      ws.send(JSON.stringify({ type: "session.start" }));
    };

    ws.onmessage = (event) => {
      if (typeof event.data !== "string") {
        console.log("  [WS] Binary frame received");
        return;
      }

      let msg;
      try { msg = JSON.parse(event.data); } catch { return; }

      switch (msg.type) {
        case "session.ready":
          console.log(`  [WS] session.ready — avatar: ${msg.avatarId}`);
          // Close cleanly after ready
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            console.log("  [WS] Closing connection.");
            try { ws.close(1000, "done"); } catch {}
            resolve();
          }
          break;
        case "avatar.fmp4.init":
          console.log(`  [WS] fMP4 init segment (${msg.byteLength} bytes)`);
          break;
        case "avatar.fmp4.segment":
          console.log(`  [WS] fMP4 segment #${msg.sequence} (${msg.byteLength} bytes)`);
          break;
        case "session.ended":
          console.log(`  [WS] session.ended — wasActive: ${msg.wasActive}`);
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            resolve();
          }
          break;
        case "error":
          console.error(`  [WS] error: ${msg.code} — ${msg.message}`);
          if (msg.fatal && !resolved) {
            resolved = true;
            clearTimeout(timeout);
            reject(new Error(`WebSocket fatal error: ${msg.code}`));
          }
          break;
        default:
          console.log(`  [WS] ${msg.type}`);
      }
    };

    ws.onerror = (err) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        // Don't reject on error if we already got useful output;
        // WebSocket errors after close are harmless.
        console.log("  [WS] Connection error (may be expected after close).");
        resolve();
      }
    };

    ws.onclose = (event) => {
      if (!resolved) {
        resolved = true;
        clearTimeout(timeout);
        console.log(`  [WS] Closed (code ${event.code}).`);
        resolve();
      }
    };
  });
}

// Run
main();
