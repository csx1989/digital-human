# Cloudflare Session Bridge

Cloudflare Worker that bridges browser WebSocket clients to the Modal avatar
runtime lifecycle via a hibernatable Durable Object.

## Architecture

```
Browser/Test Client
       │
       │ WebSocket
       ▼
  Cloudflare Worker (index.ts)
       │
       │ GET /sessions/:id/ws → DO routing
       ▼
  AvatarSessionDO (Durable Object, hibernatable)
       │
       │ HTTP POST { action: "start"|"render_frame"|"end", ... }
       ▼
  Modal AvatarLifecycleL4 (GPU L4, snapshotted pipeline)
       │
       ▼
  Rendered JPEG frame → base64 → back to client
```

### Files

| File | Purpose |
|------|---------|
| `src/index.ts` | Worker entrypoint: health, session creation, WebSocket routing to DO |
| `src/avatar-session-do.ts` | `AvatarSessionDO` — hibernatable WebSocket DO with session state |
| `src/modal-client.ts` | HTTP fetch wrapper for the Modal lifecycle endpoint |
| `src/types.ts` | Shared TypeScript interfaces |
| `wrangler.jsonc` | Worker config: DO binding, env vars |

### Hibernation State Persistence

Session-critical state (`modalSessionId`, `active`, `avatarId`, `bgId`) is
persisted in Durable Object storage via `ctx.storage.put()`. On each
`webSocketMessage` handler, state is loaded from storage first (`#loadState()`),
and after mutations, saved back (`#saveState()`). This ensures the DO survives
memory eviction during hibernation.

Per-connection metadata uses `ws.serializeAttachment()`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/sessions` | Create a new session ID (`session:{uuid}`) |
| GET | `/sessions/:id/ws` | WebSocket upgrade to AvatarSessionDO |

## WebSocket Protocol (Phase 5B)

Client sends JSON messages:

```json
{ "type": "session.start", "avatarId": "...", "bgId": "..." }
{ "type": "render_debug_frame" }
{ "type": "session.end", "reason": "..." }
```

Server replies:

```json
{ "type": "session.ready", "sessionId": "...", "avatarId": "...", "width": 1280, "height": 720, "fps": 25, "framesPerChunk": 5, "warmupMs": 1234.5 }
{ "type": "avatar.frame", "dataBase64": "...", "mimeType": "image/jpeg", "width": 1280, "height": 720 }
{ "type": "session.ended", "sessionId": "...", "wasActive": true }
{ "type": "error", "code": "...", "message": "..." }
```

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Deploy the Modal lifecycle app and note the web endpoint URL:
   ```bash
   modal deploy ../modal-avatar-runtime/modal_lifecycle_app.py
   ```

3. Set the Modal URL in `wrangler.jsonc` under `vars.MODAL_LIFECYCLE_URL`, or
   pass it at dev time:
   ```bash
   npx wrangler dev --var MODAL_LIFECYCLE_URL=https://your-modal-endpoint.modal.run
   ```

4. Run locally:
   ```bash
   npx wrangler dev
   ```

5. Type check:
   ```bash
   npm run typecheck
   ```

## Testing with wscat

```bash
# Create a session
curl -X POST http://localhost:8787/sessions
# → { "sessionId": "session:abc-123", "status": "created", "wsUrl": "ws://localhost:8787/sessions/session:abc-123/ws" }

# Connect WebSocket (use wsUrl from response, or construct manually)
wscat -c ws://localhost:8787/sessions/session:abc-123/ws

# Inside wscat:
> {"type":"session.start"}
< {"type":"session.ready","sessionId":"...","avatarId":"arnold",...}

> {"type":"render_debug_frame"}
< {"type":"avatar.frame","dataBase64":"/9j/4AAQ...","mimeType":"image/jpeg",...}

> {"type":"session.end"}
< {"type":"session.ended","sessionId":"...","wasActive":true}
```

## Not Implemented (Phase 6+)

- VAD / Smart Turn service bindings
- STT / LLM / TTS providers
- Audio streaming from browser microphone
- Avatar rendering from audio (debug frame only)
- Authentication / API keys
- PostHog observability
