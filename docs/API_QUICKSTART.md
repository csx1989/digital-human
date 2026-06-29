# API Quickstart

Get from zero to a live avatar session in 5 minutes.

## Prerequisites

- An API key (request one from the platform team).
- [Node.js](https://nodejs.org/) 18+ (for the example client), or any HTTP client.
- `curl` for the manual walkthrough.

## Quick Links

| Resource | Link |
|----------|------|
| Full API reference | [`API_V1.md`](./API_V1.md) |
| WebSocket protocol | [`WEBSOCKET_PROTOCOL.md`](./WEBSOCKET_PROTOCOL.md) |
| OpenAPI 3.0 spec | [`openapi-v1.yaml`](./openapi-v1.yaml) |
| Security validation runbook | [`API_SECURITY_VALIDATION.md`](./API_SECURITY_VALIDATION.md) |
| Example Node.js client | [`examples/node-api-client/`](../examples/node-api-client/) |

## Base URL

```
https://avatar-session-bridge.vallangirakesh.workers.dev
```

---

## Step 1 — Verify Your Key

Every `/v1/*` route requires an API key via the `Authorization` header:

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/me \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

You should see:

```json
{
  "success": true,
  "data": {
    "tenant_id": "your_tenant",
    "key_name": "Your Key Name",
    "plan": "dev",
    "auth_mode": "metadata"
  },
  "timings": { "latency_ms": 1 }
}
```

If you get `401 missing_api_key`, your `AVTR_API_KEY` env var is unset. If you
get `403 invalid_api_key`, the key value is wrong or revoked.

## Step 2 — Create a Session

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c1a2b-....",
    "status": "created",
    "stream": {
      "transport": "websocket_fmp4",
      "ws_url": "wss://...workers.dev/v1/sessions/session:8f3c1a2b-.../ws?token=eyJ...",
      "token_expires_at": 1718352300000
    }
  },
  "timings": { "latency_ms": 9 }
}
```

Save `session_id` — you will need it for cleanup. Save the full `ws_url` — it
contains a short-lived bearer token you need to connect.

> **Security warning:** `ws_url` contains a signed bearer token in the
> `?token=` query parameter. Anyone with this URL can connect to the session's
> WebSocket. **Use it as-is to open the connection, but never log it,
> screenshot it, or store it in analytics.** The token expires in 5 minutes.

## Step 3 — Connect the WebSocket

This is the **canonical start path** for interactive/live avatar sessions.

Using the `ws_url` from Step 2, open a WebSocket connection. The token is
already embedded in the URL — do not add additional auth headers.

```javascript
const ws = new WebSocket(wsUrl);
ws.binaryType = "arraybuffer";

ws.onopen = () => {
  // Send session.start to begin rendering
  ws.send(JSON.stringify({ type: "session.start" }));
};

ws.onmessage = (event) => {
  if (typeof event.data !== "string") return; // skip binary frames
  const msg = JSON.parse(event.data);

  switch (msg.type) {
    case "session.ready":
      console.log("Avatar ready:", msg.avatarId);
      break;
    case "avatar.fmp4.init":
      console.log("fMP4 init segment received");
      break;
    case "avatar.fmp4.segment":
      console.log("Media segment", msg.sequence);
      break;
    case "session.ended":
      console.log("Session ended");
      break;
    case "error":
      console.error("Server error:", msg.code, msg.message);
      break;
  }
};
```

After `session.start`, the server sends `session.ready` followed by fMP4 init
and media segments. See the [message protocol reference](#message-protocol)
below for the full list.

## Step 4 — End the Session

Always clean up to release GPU resources:

```bash
curl -X DELETE https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SESSION_ID \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c1a2b-...",
    "status": "ended",
    "end_ok": true
  },
  "timings": { "latency_ms": 1820 }
}
```

DELETE is idempotent — calling it on an already-ended session returns `200`
with `wasActive: false`. It is also exempt from rate limits, so you can always
clean up.

---

## Refreshing the WebSocket Token

The token in `ws_url` expires after 5 minutes. If you need to reconnect after
the token has expired (e.g., after a network drop or browser sleep), mint a
fresh one:

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SESSION_ID/ws-token \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "ws_url": "wss://...workers.dev/v1/sessions/session:8f3c1a2b-.../ws?token=eyJ...",
    "token_expires_at": 1718352600000
  },
  "timings": { "latency_ms": 3 }
}
```

Use the newest `ws_url` for reconnects. Previously issued tokens may remain
valid until their own expiry — refreshing does not revoke them. To invalidate
all outstanding tokens, rotate `AVTR_SESSION_TOKEN_SECRET`.

---

## Advanced: REST Start (Control Plane)

In addition to the WebSocket `session.start` message (Step 3), there is a REST
endpoint that starts rendering without a WebSocket connection:

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SESSION_ID/start \
  -H "Authorization: Bearer $AVTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"avatar_id":"arnold","bg_id":"aloe_window"}'
```

This is an **advanced, control-plane** endpoint. It is **not** part of the
quickstart's canonical interactive path.

### Do not mix REST start with the WS live path

The WebSocket media stream (fMP4 init + segments) is initialized inside the
WS `session.start` handler. If a session is already active (e.g., started via
REST), connecting a WebSocket and sending `session.start` returns
`session_already_active` and does **not** attach the media stream to that
socket.

This means:

- **For interactive/live clients:** Always use the canonical WS path (Step 3):
  create session → connect `ws_url` → send WS `session.start` → receive
  `session.ready` + fMP4 media.
- **REST `/start` is for control-plane or SSE-oriented use only** (e.g.,
  pre-warming, headless orchestration via the SSE `/stream` endpoint).
  Do not assume a REST-started session can receive fMP4 media over a
  subsequent WebSocket connection without explicit attach behavior, which
  is not yet implemented.

---

## Common Errors

| HTTP | `error.code` | Cause | Fix |
|------|-------------|-------|-----|
| 401 | `missing_api_key` | No `Authorization` header | Set `AVTR_API_KEY` and pass it as `Bearer` |
| 403 | `invalid_api_key` | Key not recognized or revoked | Check the key value; contact platform team |
| 404 | `not_found` | Session doesn't exist or belongs to another tenant | Verify `session_id` and key ownership |
| 409 | `session_starting` | A start is already in progress | Wait for it to complete or call DELETE to cancel |
| 409 | `session_conflict` | REST `/start` called with a different avatar on an active session | End the session first, or use the same avatar |
| 409 | `session_cancelled` | Session was ended (DELETE) while a start was in progress | Create a new session |
| 429 | `rate_limited` | Too many requests in a 60-second window | Wait for `Retry-After` seconds |
| 500 | `modal_unreachable` | Rendering backend is down | Retry after a brief wait; check status page |
| 502 | `modal_start_failed` | Rendering backend rejected the start | Check avatar/background IDs; retry |
| 503 | `api_auth_not_configured` | Server has no keys configured | Platform team must configure secrets |

For the full error code reference, see [API_V1.md > Error Codes](./API_V1.md#error-codes).

---

## Rate Limits

All `/v1/*` routes are subject to per-tenant rate limits. These are abuse-control
safety nets, not billing quotas. Key categories:

| Category | Limit | Applies to |
|----------|-------|------------|
| Session create | 20 / 60s | `POST /v1/sessions` |
| GPU start | 5 / 60s | `POST /v1/sessions/:id/start` |
| WS connect | 5 / 60s | `GET /v1/sessions/:id/ws` (connection attempts) |
| Read | 120 / 60s | All GET routes |
| Cleanup | Exempt | DELETE and POST /end |

Limits are per-tenant and per-Cloudflare-location. If you hit `429`, wait for
the `Retry-After` header duration (60 seconds).

---

## Message Protocol Reference

Key WebSocket messages (JSON, keyed by `type`). For complete field-level
documentation, error code semantics, render modes, and the full turn lifecycle
diagram, see [`WEBSOCKET_PROTOCOL.md`](./WEBSOCKET_PROTOCOL.md).

### Client -> Server

| `type` | Purpose |
|--------|---------|
| `session.start` | Begin rendering (primary start path) |
| `mic.start` | Start microphone audio input |
| `mic.stop` | Stop microphone audio input |
| `session.end` | End session over WebSocket |
| `avatar.speech.playback.ended` | Acknowledge speech turn finished |

Binary frames (raw `ArrayBuffer` PCM16 audio) are sent for microphone input.

### Server -> Client

| `type` | Purpose |
|--------|---------|
| `session.ready` | Avatar loaded, rendering active |
| `avatar.fmp4.init` | fMP4 initialization segment (feed to MSE SourceBuffer) |
| `avatar.fmp4.segment` | fMP4 media segment (feed to MSE SourceBuffer) |
| `avatar.idle.start` | Idle animation sequence beginning |
| `avatar.idle.chunk` | Idle animation frame(s) |
| `avatar.speech.start` | Speech rendering begins for a turn |
| `avatar.speech.chunk` | Speech animation frame(s) for a turn |
| `speech.started` | User speech turn detected |
| `speech.finalized` | User speech turn complete |
| `transcript.final` | Final transcription result |
| `assistant.text` | LLM response text |
| `assistant.audio` | TTS audio data |
| `session.ended` | Session ended by server |
| `error` | Error message (check `code`, `message`, `fatal?`) |
| `status` | Pipeline status update |

---

## Using the Example Client

A dependency-free Node.js client is included:

```bash
cd examples/node-api-client

# Basic flow (create → status → delete) — requires Node 18+
AVTR_API_KEY=your_key_here node client.mjs

# With WebSocket connection — requires Node 22+ (global WebSocket)
AVTR_API_KEY=your_key_here node client.mjs --ws
```

The client prints **redacted** output only — no full API keys, no full WebSocket
URLs with tokens. See [`examples/node-api-client/README.md`](../examples/node-api-client/README.md)
for details.
