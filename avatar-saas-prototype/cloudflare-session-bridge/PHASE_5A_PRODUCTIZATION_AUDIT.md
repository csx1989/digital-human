# Phase 5A Productization Audit

**Date:** 2026-06-08
**Baseline Commit:** `e7cc4ac`
**Verdict: PHASE 5A AUDIT GO**

---

## 1. Current Architecture

```
Browser ←──WS──→ Bridge Worker (AvatarSessionDO) ←──SSE──→ Modal (PyTorch CPU-snapshot)
                      ↑                                       ↑
                      └── VAD DO, Smart Turn DO               └── push_audio (TTS float32 PCM)
```

**Session lifecycle:**

1. `POST /sessions` in `index.ts` creates a session ID and returns `{sessionId, wsUrl}`.
2. Browser opens WebSocket to `/sessions/:id/ws` → routed to `AvatarSessionDO.fetch()`.
3. Browser sends `session.start` → DO calls `startModalSession` (4C path) to Modal's `start-session` endpoint.
4. DO opens SSE stream to Modal's `stream-fmp4-av` endpoint via `startModalSessionFmp4Stream`.
5. DO relays fMP4 init/media events to browser as WS JSON messages (`avatar.fmp4.init`, `avatar.fmp4.segment`).
6. When user speaks, browser sends mic audio → DO runs VAD → STT → LLM → TTS pipeline.
7. TTS audio is pushed to Modal via `pushModalAudio` in 200ms float32 16kHz PCM chunks for lip-sync and FFmpeg muxing into fMP4 AV stream.
8. `assistant.audio` is sent to browser as metadata-only (no `dataBase64`) — audio is muxed in fMP4.
9. Browser plays muxed fMP4 AV passively via MSE SourceBuffer. No separate browser audio playback in AV mode.
10. Browser sends `session.end` → DO calls `endModalSession` → Modal cleans up.

---

## 2. Existing Public Routes

Source: `avatar-saas-prototype/cloudflare-session-bridge/src/index.ts`

| Route | Method | Purpose | Handler | External-API Ready? |
|-------|--------|---------|---------|---------------------|
| `/health` | GET | Health check | `index.ts` | Yes |
| `/sessions` | POST | Create session, return `{sessionId, wsUrl}` | `index.ts` | Yes |
| `/sessions/:id/ws` | GET | WebSocket upgrade to DO | `index.ts` → `DO.fetch()` | Yes (WS transport) |
| `/debug/posthog-test` | POST | PostHog connectivity test | `index.ts` | No (internal, requires debug token) |
| `*` | OPTIONS | CORS preflight | `index.ts` | N/A |

No other HTTP routes exist. All session lifecycle operations (start, end, status) are exclusively through WebSocket messages.

---

## 3. Existing WebSocket Messages

Source: `avatar-saas-prototype/cloudflare-session-bridge/src/types.ts:85-236`

### Client → Server

| Message | Purpose | External-API Ready? | Notes |
|---------|---------|---------------------|-------|
| `session.start` | Start Modal session (avatarId, bgId) | Yes (needs REST equivalent) | DO line 284 |
| `session.end` | End session | Yes (needs REST equivalent) | DO line 290 |
| `mic.start` | Start browser mic audio stream | No | Browser-only |
| `mic.stop` | Stop browser mic audio stream | No | Browser-only |
| `render_debug_frame` | Single frame debug render | No | Dev debug |
| `avatar.speech.playback.ended` | Browser playback acknowledgment | No | Browser-only |
| `client.telemetry` | Browser telemetry relay to PostHog | No | Browser-only |
| Binary ArrayBuffer | PCM16 audio chunks from browser mic | No | Browser mic input |

### Server → Client

| Message | Purpose | External-API Ready? | Notes |
|---------|---------|---------------------|-------|
| `session.ready` | Session started, renderMode, avatar info | Yes | Contains `renderMode: "fmp4_av"` |
| `status` | Progress updates (starting_modal, ending, etc.) | Yes | |
| `error` | Error notification | Partially | String-based codes |
| `avatar.fmp4.init` | fMP4 init segment (base64) | Yes (but base64 wasteful) | DO line 4246 |
| `avatar.fmp4.segment` | fMP4 media segment (base64) | Yes (but base64 wasteful) | DO line 4281 |
| `avatar.speech.start` | Speech rendering started | Yes | |
| `avatar.timeline.started` | Timeline lifecycle | Yes | |
| `avatar.timeline.stopped` | Timeline lifecycle | Yes | |
| `speech.started` | VAD speech detected | Partially | |
| `speech.segment.started` | Speech segment | Partially | |
| `speech.finalized` | Turn end (smart turn decision) | Partially | |
| `speech.continues` | Turn continues | Partially | |
| `transcript.final` | STT result text | Yes | |
| `assistant.text` | LLM response text | Yes | |
| `assistant.audio` | TTS audio metadata | Yes (metadata-only in AV mode) | No `dataBase64` in AV mode |
| `audio.stats` | Audio chunk statistics | No | Internal telemetry |
| `avatar.frame` | Legacy JPEG frame | No | Legacy only |
| `avatar.idle.start` / `avatar.idle.chunk` | Legacy idle frames | No | Legacy only |
| `avatar.speech.chunk` | Legacy speech frames | No | Legacy only |

### What is Browser/Demo-Only

- `mic.start` / `mic.stop` — browser microphone control
- Binary audio chunks — browser mic PCM16 input
- `render_debug_frame` — dev debug tool
- `avatar.speech.playback.ended` — browser playback acknowledgment
- `client.telemetry` — browser telemetry relay
- `avatar.frame`, `avatar.idle.*`, `avatar.speech.chunk` — legacy JPEG frame delivery

### What Can Become External API Contract

- Session lifecycle: create, start, end, status
- Stream delivery: fMP4 init + media segments
- Speech/text: `transcript.final`, `assistant.text`, `assistant.audio` (metadata)
- Status/error: `status`, `error`

---

## 4. Current fMP4 AV Delivery Path

### 4C Path (Active)

The 4C path is activated by the **presence of Modal session endpoint URLs** in `wrangler.jsonc`, not by any `ENABLE_*` flag.

**Step-by-step with file/line references:**

1. **Session created** — `index.ts:118-128`: `POST /sessions` generates UUID, returns `{sessionId, wsUrl}`.

2. **WS connects** — `index.ts:130-148`: GET `/sessions/:id/ws` routes to `AvatarSessionDO.fetch()` which accepts WebSocket (`avatar-session-do.ts:213-247`).

3. **session.start received** — `avatar-session-do.ts:284`: dispatches to `#handleSessionStart`.

4. **4C path selected** — `avatar-session-do.ts:3794-3795`:
   ```typescript
   const use4CPath = !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL
     && !!this.env.MODAL_SESSION_START_URL;
   ```
   Both URLs are set in `wrangler.jsonc:36,40`, so `use4CPath = true`.

5. **Modal session started** — `avatar-session-do.ts:3822-3826`:
   Calls `startModalSession` → HTTP POST to `MODAL_SESSION_START_URL` (`--start-session.modal.run`).

6. **session.ready sent** — `avatar-session-do.ts:3859-3871`:
   Sends `session.ready` with `renderMode: "fmp4_av"` to browser via WS.

7. **SSE stream opened** — `avatar-session-do.ts:3873`:
   Calls `#runModalSessionFmp4Stream(ws)` which calls `startModalSessionFmp4Stream` (`modal-client.ts:892-1004`).
   Opens SSE stream to `MODAL_SESSION_STREAM_FMP4_AV_URL` (`--stream-fmp4-av.modal.run`).

8. **fMP4 init relayed** — `avatar-session-do.ts:4246-4260`:
   `onInit` callback → sends `avatar.fmp4.init` (base64) to browser via WS.

9. **fMP4 segments relayed** — `avatar-session-do.ts:4281-4333`:
   `onMedia` callback → sends `avatar.fmp4.segment` (base64) to browser via WS.

10. **TTS audio pushed** — `avatar-session-do.ts:1526-1531`:
    When `use4CPath` is true, calls `#pushTtsChunksToModal` which:
    - Decodes WAV → PCM16 → Float32 (line 1682-1692)
    - Resamples to 16kHz if needed (line 1713-1721)
    - Chunks into 200ms (3200 samples) Float32 arrays (line 1727-1733)
    - Calls `pushModalAudio` for each chunk → HTTP POST to `MODAL_SESSION_PUSH_AUDIO_URL` (line 1750-1754)

11. **assistant.audio metadata** — `avatar-session-do.ts:1487-1498`:
    In AV mode, sends `assistant.audio` **without `dataBase64`** — metadata only.
    Audio is not played separately by browser; it is muxed into fMP4 by Modal FFmpeg.

12. **session.end** — `avatar-session-do.ts:290` → `#handleSessionEnd`:
    Calls `endModalSession` → HTTP POST to `MODAL_SESSION_END_URL`.
    Waits for fMP4 stream end (5s timeout). Sends `session.ended` to browser.

### Key Insight: No Separate Browser Audio in AV Mode

In AV mode, the audio path is:
- TTS generates WAV audio
- Bridge pushes float32 PCM to Modal via `pushModalAudio`
- Modal feeds PCM to pipeline for lip-sync (HuBERT features) and FFmpeg muxes it into fMP4 AAC track
- Browser receives muxed fMP4 AV and plays both video+audio via MSE
- `assistant.audio` message is metadata-only (no `dataBase64` field)

The browser does NOT play audio separately. This is confirmed at `avatar-session-do.ts:1487-1498` where `dataBase64` is omitted in AV mode.

---

## 5. Environment Flag Audit

Source: `avatar-saas-prototype/cloudflare-session-bridge/wrangler.jsonc`

### Active Flags (4C Path)

| Flag | Value | Status | Notes |
|------|-------|--------|-------|
| `MODAL_SESSION_START_URL` | `https://vallangirakesh--start-session.modal.run` | **Active** | 4C start endpoint |
| `MODAL_SESSION_END_URL` | `https://vallangirakesh--end-session.modal.run` | **Active** | 4C end endpoint |
| `MODAL_SESSION_STATUS_URL` | `https://vallangirakesh--session-status.modal.run` | **Active** | 4C status endpoint |
| `MODAL_SESSION_PUSH_AUDIO_URL` | `https://vallangirakesh--push-audio.modal.run` | **Active** | TTS audio push |
| `MODAL_SESSION_STREAM_FMP4_AV_URL` | `https://vallangirakesh--stream-fmp4-av.modal.run` | **Active** | 4C path trigger (URL presence activates 4C) |
| `ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY` | `"true"` | Active | Telemetry every 10 segments |

### Legacy/Confusing Flags

| Flag | Value | Status | Problem | Recommended Action |
|------|-------|--------|---------|-------------------|
| `ENABLE_MODAL_FMP4_AV_STREAM` | `"false"` | **CONFUSING** | 4C path is fully active despite this being "false". The flag only affects legacy path renderMode and some `isAvMode` checks. The 4C path is controlled by URL presence, not this flag. | Rename to `ENABLE_LEGACY_FMP4_AV_TIMELINE` or remove after legacy code is cleaned |
| `ENABLE_LEGACY_MODAL_PROTOCOL` | `"false"` | Disabled (correct) | Legacy per-frame protocol | Keep false, remove with legacy code in Phase 5F |
| `MODAL_LIFECYCLE_URL` | Set | Legacy only | Used by old `callModal` path | Keep until legacy removed |
| `MODAL_TIMELINE_CONTROL_URL` | Set | Legacy only | Used by legacy fmp4_av mode | Keep until legacy removed |
| `ENABLE_MODAL_PERSISTENT_TIMELINE` | `"true"` | Legacy only | Controls legacy timeline | Remove with legacy code |
| `ENABLE_AVATAR_TIMELINE` | `"true"` | Legacy only | Controls legacy avatar timeline | Remove with legacy code |
| `ENABLE_LIVE_IDLE` | `"false"` | Legacy only | Controls legacy idle loop | Remove with legacy code |
| `ENABLE_MODAL_FMP4_VIDEO_STREAM` | Not set | Legacy only | Video-only fMP4 mode | Remove with legacy code |
| `ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS` | `"true"` | Active | Controls which Modal URL to use for parallel TTS | Keep, confusing name |
| `OMNIVOICE_SAME_GPU_PARALLEL_TARGET` | `"l40s"` | Active | Parallel TTS GPU target | Keep |
| `MODAL_PARALLEL_L40S_LIFECYCLE_URL` | Set | Active | Parallel lifecycle URL | Keep |

### VAD/STT/LLM/TTS Configuration

| Flag | Value | Purpose |
|------|-------|---------|
| `ENABLE_SPEECH_STREAMING` | `"true"` | STT streaming enabled |
| `LLM_BASE_URL` | `https://cerebras-router-worker...` | LLM endpoint |
| `LLM_MODEL` | `gpt-oss-120b` | LLM model name |
| `OMNIVOICE_TTS_URL` | `https://vallangirakesh--avtr1-omnivoice-tts...` | TTS endpoint |
| `POSTHOG_HOST` | `https://eu.i.posthog.com` | PostHog ingestion |
| Service binding `VAD_SERVICE` | `any-toy-vad` | VAD DO |
| Service binding `SMART_TURN_SERVICE` | `any-toy-smart-turn` | Smart Turn DO |

---

## 6. API Product Gaps

| Gap | Priority | Notes |
|-----|----------|-------|
| **Authentication / API key** | Critical | No auth on any endpoint. CORS is wildcard (`*`). Anyone with session ID can connect. |
| **REST lifecycle endpoints** | High | No `POST /sessions/:id/start`, `POST /sessions/:id/end`, `GET /sessions/:id/status`. All lifecycle is WS-only. |
| **Stream endpoint** | High | No REST/SSE endpoint for fMP4 stream delivery. Only available via WS base64. |
| **Rate limiting** | Medium | No per-customer limits. |
| **Session TTL / cleanup** | Medium | No auto-expiry for stale sessions. |
| **Structured error codes** | Medium | String-based error codes, no HTTP status code mapping. |
| **Per-customer avatar ownership** | Low | All avatars available to all sessions. |
| **Billing hooks** | Low | No usage tracking for billing. |
| **API versioning** | Low | No `/v1/` prefix. |
| **Documentation** | Low | No OpenAPI spec. |

---

## 7. Recommended Production API Shape — Option C: Hybrid

### Why Option C

- **REST for lifecycle** — Standard CRUD operations (create, start, end, status) fit HTTP perfectly.
- **WebSocket or SSE for streaming** — fMP4 segments need real-time delivery. WS already works; SSE is simpler for HTTP clients.
- **Preserves existing architecture** — DO owns all session state. REST routes delegate to DO, not directly to Modal.

### Important: REST Routes Must Delegate to the Durable Object

All REST lifecycle endpoints (`/v1/sessions/:id/start`, `/v1/sessions/:id/end`, `/v1/sessions/:id/status`) must route through the existing `AvatarSessionDO`. The top-level Worker (`index.ts`) resolves the DO by session ID and delegates HTTP requests to it via `stub.fetch()`. The DO owns session state and is the sole caller of Modal endpoints. The Worker never calls Modal directly for session operations.

### Important: SSE is text/event-stream, Not Binary

SSE (`text/event-stream`) is a text protocol. fMP4 segments are binary data. For SSE delivery, segments must be base64-encoded in the `data:` field. This is the same encoding currently used for WS delivery. If binary efficiency becomes critical in the future, we should evaluate alternative transports (chunked HTTP, WebSocket binary frames, or multipart/x-mixed-replace), but SSE with base64 is the simplest starting point.

### Proposed API

```
REST Lifecycle (index.ts → DO):
  POST /v1/sessions             → {session_id, status}
  POST /v1/sessions/:id/start   → {session_id, render_mode, stream_url}
  POST /v1/sessions/:id/end     → {session_id, status, chunks, segments}
  GET  /v1/sessions/:id/status  → {session_id, state, chunks, segments}

Stream Delivery (DO):
  GET /v1/sessions/:id/stream   → text/event-stream (base64 fMP4 init + media)

Legacy (unchanged):
  WS /sessions/:id/ws           → existing browser WS path

Auth (Phase 5D):
  x-api-key header on all /v1/* endpoints
```

### Why Not Other Options

| Option | Why Not |
|--------|---------|
| **A. Keep WS only** | WS is stateful, hard for HTTP clients, requires connection management. API customers expect REST. |
| **B. REST + SSE only** | Loses real-time bidirectional communication. Mic audio, VAD, speech events need WS. |
| **D. R2/streaming** | Too far ahead. Requires infrastructure we don't have yet. |

---

## 8. Proposed Implementation Phases

| Phase | Scope | Risk | Dependencies |
|-------|-------|------|-------------|
| **5B** REST lifecycle routes | Add `POST /v1/sessions/:id/start`, `end`, `status` to `index.ts` routing to DO. DO gets HTTP handler for non-WS requests. | Low | Additive, existing WS path untouched |
| **5C** Stream endpoint / transport decision | Add `GET /v1/sessions/:id/stream` SSE endpoint on DO. Base64-encoded fMP4 segments in SSE events. Evaluate binary transport later. | Medium | New transport path, must not break WS |
| **5D** API key auth | `x-api-key` validation middleware. Key storage in KV or DO. | Medium | Need key storage mechanism |
| **5E** Rate limits / TTL / cleanup | Per-customer request limits. Session auto-expiry. Stale session cleanup alarm. | Low | Standard DO patterns |
| **5F** Flag cleanup / legacy removal | Remove legacy code paths, confusing flags. Rename `ENABLE_MODAL_FMP4_AV_STREAM`. | Low | Dead code removal after 5B/5C stable |

---

## 9. Risks

| Risk | Impact | Mitigation |
|------|--------|-----------|
| DO hibernation + long-lived SSE | SSE stream may break during DO hibernation | Use `ctx.waitUntil`, or consider short-lived polling |
| Base64 overhead in SSE | ~33% size increase over binary | Acceptable for prototype; binary transport in future |
| WS + REST dual transport accessing same DO | State consistency between WS and REST | DO is single-threaded per hibernation wake; serialize state access |
| API key storage | Need KV or dedicated DO for keys | Start with single static key in env var; scale to KV later |
| Legacy code complexity | 4500-line DO with many legacy paths | Phase 5F removes legacy after new paths are stable |

---

## 10. Non-Goals

- No browser web client changes
- No Modal optimization or GPU snapshots
- No real-time performance tuning
- No multi-avatar support
- No R2/streaming delivery
- No deployment during audit phase
- No changes to existing WS contract

---

## 11. Next Phase

**Phase 5B:** REST lifecycle endpoints.

The recommended Phase 5B prompt is documented separately and should be executed only after explicit instruction.
