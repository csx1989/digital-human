# Avatar API v1 — Node.js Example Client

A dependency-free example client demonstrating the Avatar API v1 session
lifecycle: create, status check, optional WebSocket connection, and cleanup.

## Prerequisites

| Flow | Node Version | Why |
|------|-------------|-----|
| Basic (create/status/delete) | Node 18+ | Built-in `fetch` |
| WebSocket (`--ws` flag) | Node 22+ | Global `WebSocket` |

The client feature-detects `globalThis.WebSocket` at runtime. If you request
`--ws` on a Node version without it, you get a clear error message.

## Setup

No `npm install` needed — zero dependencies.

```bash
# Set your API key
export AVTR_API_KEY=avtr_test_your_key_here

# Optional: override base URL (defaults to production)
export AVTR_API_BASE=https://avatar-session-bridge.vallangirakesh.workers.dev
```

## Usage

```bash
# Basic flow: create -> status -> delete
node client.mjs

# With WebSocket: create -> connect WS -> session.start -> delete
node client.mjs --ws
```

> **Note:** The `--ws` flag is a **protocol smoke test**, not a full media
> renderer. It logs message types but does not decode fMP4 segments or play
> audio. A real browser client must append fMP4 init/segments to a MediaSource
> SourceBuffer. See [`docs/WEBSOCKET_PROTOCOL.md`](../../docs/WEBSOCKET_PROTOCOL.md)
> for the full protocol reference and MSE implementation notes.

## What It Does

1. **Verifies auth** — calls `GET /v1/me` and prints tenant identity.
2. **Creates a session** — calls `POST /v1/sessions`, prints redacted session info.
3. **Reads status** — calls `GET /v1/sessions/:id/status`.
4. **(Optional) Connects WebSocket** — opens `ws_url`, sends `session.start`,
   logs incoming message types, closes after `session.ready` or 10s timeout.
   This is a protocol demo only — no media is decoded or rendered.
5. **Cleans up** — calls `DELETE /v1/sessions/:id` (always runs, even on error).

## Redaction

This client **never** prints:
- Full API key values (shows first 12 chars + `...`)
- Full `ws_url` with token (strips `?token=` param, truncates path)

This is intentional — `ws_url` contains a bearer credential. See
[`docs/API_SECURITY_VALIDATION.md`](../../docs/API_SECURITY_VALIDATION.md) for
the WS token security model.

## Files

| File | Purpose |
|------|---------|
| `avtr-client.mjs` | Reusable client module (`AvtrClient` class + redaction utilities). Import this into your own code. |
| `client.mjs` | Demo CLI script. Run this to test the API. |
| `.env.example` | Template for environment variables. |

## Using the Client Module

```javascript
import { AvtrClient, redactWsUrl, redactApiKey } from "./avtr-client.mjs";

const client = new AvtrClient({
  apiKey: process.env.AVTR_API_KEY,
  // baseUrl: "https://staging..." // optional
});

// Create a session
const session = await client.createSession();
console.log("session:", session.session_id);
console.log("ws:", redactWsUrl(session.stream.ws_url));

// Check status
const status = await client.getSessionStatus(session.session_id);
console.log("status:", status.status);

// End session
await client.endSession(session.session_id);
```

The `avtr-client.mjs` module exports:

- `AvtrClient` — class with `createSession()`, `getSessionStatus()`,
  `startSession()`, `refreshWsToken()`, `endSession()`
- `AvtrApiError` — thrown on non-success responses (has `.status`, `.code`,
  `.message`, `.requestId`)
- `redactApiKey(key)` — returns a safe display string
- `redactWsUrl(wsUrl)` — strips token param, returns safe display string

## API Reference

For full endpoint documentation, see:
- [`docs/API_V1.md`](../../docs/API_V1.md)
- [`docs/API_QUICKSTART.md`](../../docs/API_QUICKSTART.md)
- [`docs/openapi-v1.yaml`](../../docs/openapi-v1.yaml) (OpenAPI 3.0 spec)
