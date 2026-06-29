# WebSocket Protocol Reference

Observed protocol reference for the avatar WebSocket message protocol. This
document is derived from the Durable Object source (`avatar-session-do.ts`) and
type definitions (`types.ts`).

**Related docs:**
- [API Quickstart](./API_QUICKSTART.md) — 5-minute walkthrough
- [API v1 Reference](./API_V1.md) — REST endpoints, auth, rate limits
- [Security Validation](./API_SECURITY_VALIDATION.md) — tenant isolation

---

## Stability

This is an **observed protocol reference**, not a frozen public contract.

- **Stable public messages:** Core session lifecycle (`session.ready`,
  `session.ended`, `status`, `error`), fMP4 media (`avatar.fmp4.init`,
  `avatar.fmp4.segment`), and the speech pipeline (`speech.started`,
  `speech.finalized`, `transcript.final`, `assistant.text`, `assistant.audio`,
  `audio.stats`).
- **Legacy messages:** JPEG timeline/debug-frame messages (`avatar.frame`,
  `avatar.speech.start`, `avatar.speech.chunk`, `avatar.idle.start`,
  `avatar.idle.chunk`, `render_debug_frame`). These are still sent for older
  render modes but are not the primary transport.
- **Internal/demo messages:** `client.telemetry` and diagnostic fields (e.g.,
  `avatar.fmp4.segment` timing fields, `containerInstanceId`) may change
  without notice.
- **Forward compatibility:** Clients should ignore unknown message `type`
  values and unknown object fields. The server may add new messages or fields
  at any time.

---

## Connection

### URL

```
wss://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/:session_id/ws?token=...
```

The full `ws_url` (including `?token=`) is returned by `POST /v1/sessions` in
`stream.ws_url`, and refreshed by `POST /v1/sessions/:id/ws-token`.

### Bearer credential warning

The `?token=` query parameter is a **short-lived bearer credential** (5-minute
TTL, HMAC-SHA256 signed). Anyone holding this URL can connect to the session's
WebSocket.

- **Use the `ws_url` as-is** to open the connection. Do not add auth headers.
- **Never log, screenshot, or store** the full `ws_url` in analytics, error
  reports, or browser history.
- Previously issued tokens remain valid until their own expiry. Refreshing does
  not revoke them. To invalidate all outstanding tokens, rotate
  `AVTR_SESSION_TOKEN_SECRET`.

### Upgrade flow

The Worker validates the upgrade request in this order before forwarding to the
Durable Object:

| Step | Check | On failure (HTTP) |
|------|-------|-------------------|
| 1 | Session ID format (`session:` prefix) | 400 `invalid_session_id` |
| 2 | `AVTR_SESSION_TOKEN_SECRET` configured | 503 `ws_token_secret_not_configured` |
| 3 | `?token=` query param present | 401 `ws_token_missing` |
| 4 | Token signature, expiry, session binding | 401 `ws_token_invalid` |
| 5 | Caller owns this session | 404 `not_found` |
| 6 | Rate limit (`ws_connect`: 5/min) | 429 `rate_limited` |
| 7 | Forward to DO → WebSocket 101 | — |

**Token verification reasons** (all surface as `ws_token_invalid`):

| Internal reason | Cause |
|-----------------|-------|
| `malformed` | Token not two dot-separated parts, payload JSON parse failed, or missing required fields |
| `invalid_signature` | HMAC-SHA256 mismatch |
| `expired` | `exp` field is in the past |
| `wrong_purpose` | `purpose` field is not `"ws"` |
| `session_mismatch` | Token's `sid` does not match the URL path session ID |

---

## Message Format

All JSON messages use a `type` field for dispatch. Binary frames are raw audio.

| Frame type | Content | Direction |
|------------|---------|-----------|
| Text (`string`) | JSON object with `type` field | Both directions |
| Binary (`ArrayBuffer`) | Raw PCM16 audio | Client → Server only |

Set `ws.binaryType = "arraybuffer"` before connecting to receive binary frames
correctly.

---

## Client → Server Messages

Source: `ClientMessage` type in `types.ts:99-111`.

### Core interactive messages

#### `session.start`

Begins avatar rendering and media streaming on this WebSocket connection. This
is the **canonical start path** for interactive/live sessions.

```json
{ "type": "session.start" }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `avatarId` | `string` | No | Avatar to render. Defaults to server default (`"arnold"`). |
| `bgId` | `string` | No | Background to use. Defaults to first available. |
| `transport` | `string` | No | Preferred transport: `"auto"` (default), `"webrtc_sfu"`, or `"websocket_fmp4"`. See [Transport Selection](#transport-selection) below. |

**Behavior:**
- Server replies with `session.ready` followed by media stream messages.
- If the session is already active: returns error `session_already_active`.
- If a start is in progress: returns error `session_starting`.
- If the session was ended (DELETE) while starting: returns error `session_cancelled`.

**Do not combine with REST `POST /v1/sessions/:id/start`.** The WS media stream
(fMP4 init + segments) is initialized inside the WS `session.start` handler. A
REST-started session connecting via WS gets `session_already_active` without
media stream attachment.

##### Transport Selection

The optional `transport` field controls which media transport the session uses:

| Value | Behavior |
|-------|----------|
| `websocket_fmp4` | Forces fMP4 over WebSocket. Never falls back. |
| `webrtc_sfu` | Requests Cloudflare Realtime SFU (WebRTC). Requires the session to be SFU-beta-authorized and SFU to be fully configured; otherwise the server returns `sfu_beta_not_enabled` or `sfu_not_configured`. If the SFU start fails, the server returns `sfu_unavailable`. Does NOT fall back to fMP4. |
| `auto` (default) | Server chooses. Currently always resolves to `websocket_fmp4` unless the server is explicitly configured to prefer SFU. If SFU is attempted and fails, falls back to fMP4 silently. |

When `transport: "webrtc_sfu"` is used, the `session.ready` reply includes the
`sfu` block (see below) and the browser subscribes to the SFU track via the REST
endpoints documented in [API_V1.md](API_V1.md#sfu-transport). The fMP4 media
stream pump is NOT started for SFU sessions.

> **Milestone 1 scope:** When SFU is used, video is delivered over WebRTC.
> Avatar speech audio continues to arrive over the existing WebSocket messages
> (`avatar.speech.*`, `assistant.audio`). There is no Opus audio track in the
> WebRTC stream.

#### `session.end`

Tears down the session and releases backend resources.

```json
{ "type": "session.end" }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `reason` | `string` | No | Optional reason for telemetry. |

Server replies with `session.ended`, then the client should close the socket.

#### `mic.start`

Signals that the client is ready to send microphone audio.

```json
{ "type": "mic.start" }
```

No fields. Server replies with `status: "mic_ready"`.

#### `mic.stop`

Stops accepting microphone audio. If a speech turn is open, it is force-finalized.

```json
{ "type": "mic.stop" }
```

No fields. Server replies with `status: "mic_stopped"`.

#### Binary audio (PCM16)

Raw 16 kHz mono PCM16 audio chunks sent as `ArrayBuffer` binary frames.

| Property | Value |
|----------|-------|
| Format | PCM16 (signed 16-bit little-endian) |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Frame type | `ArrayBuffer` (binary) |

Chunks must be non-empty with an even byte count. Odd-length or empty chunks
return error `invalid_audio_chunk`.

#### `avatar.speech.playback.ended`

Acknowledges that avatar speech video/audio for a turn has finished playing.
The server uses this to resume idle animation streaming.

```json
{ "type": "avatar.speech.playback.ended", "turnId": 3 }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `turnId` | `number` | Yes | The speech turn ID that finished playback. |

### Internal/demo messages

#### `client.telemetry` *(internal — demo client only)*

Relays browser-side telemetry to PostHog. Not required for API clients.

```json
{ "type": "client.telemetry", "event": "mic_started", "elapsedMs": 1234 }
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `event` | `string` | Yes | Event name for PostHog. |
| `elapsedMs` | `number` | No | Milliseconds since session start. |
| `properties` | `object` | No | Additional properties. |

> API clients should not send `client.telemetry`. It exists for the demo web
> client's browser-side observability. Sending it from an API client has no
> harmful effect but pollutes the demo tenant's PostHog data.

#### `render_debug_frame` *(legacy — JPEG timeline only)*

Requests a single static avatar frame. Only used in non-fMP4 render modes.

```json
{ "type": "render_debug_frame" }
```

> Not applicable to the `websocket_fmp4` transport. In fMP4 mode, media is
> delivered via `avatar.fmp4.init` + `avatar.fmp4.segment` messages.

---

## Server → Client Messages

Source: `ServerMessage` type in `types.ts:113-250`.

### Session lifecycle

#### `session.ready`

Session successfully started. Render dimensions and mode are included.

**fMP4 transport (default):**

```json
{
  "type": "session.ready",
  "sessionId": "session:8f3c...",
  "avatarId": "arnold",
  "bgId": "aloe_window",
  "width": 1280,
  "height": 720,
  "fps": 25,
  "framesPerChunk": 0,
  "warmupMs": 1200,
  "renderMode": "fmp4_av"
}
```

**WebRTC SFU transport** (when `transport: "webrtc_sfu"` was requested and SFU
is configured):

```json
{
  "type": "session.ready",
  "sessionId": "session:8f3c...",
  "avatarId": "arnold",
  "bgId": "aloe_window",
  "width": 1280,
  "height": 720,
  "fps": 25,
  "framesPerChunk": 0,
  "warmupMs": 1200,
  "transport": "webrtc_sfu",
  "sfu": { "publisher_track_name": "avatar-video" }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sessionId` | `string` | Session ID. |
| `avatarId` | `string` | Resolved avatar ID. |
| `bgId` | `string` | Resolved background ID. |
| `width` | `number` | Video width in pixels. |
| `height` | `number` | Video height in pixels. |
| `fps` | `number` | Frames per second. |
| `framesPerChunk` | `number` | Frames per JPEG chunk (legacy modes). |
| `warmupMs` | `number` | Modal warm-up time in milliseconds. |
| `renderMode` | `string?` | Render mode: `"fmp4_av"`, `"fmp4_video"`, or `"jpeg_timeline"`. Present only on fMP4 transports. Omitted when `transport` is `"webrtc_sfu"`. |
| `transport` | `string?` | Active transport: `"websocket_fmp4"` or `"webrtc_sfu"`. Present when the session uses an explicit transport. |
| `sfu` | `object?` | SFU details. Present only when `transport` is `"webrtc_sfu"`. Contains `publisher_track_name` — the track name the browser uses to subscribe via `POST /v1/sessions/:id/sfu/subscribe`. |

When `transport` is `"webrtc_sfu"`, the fMP4 media stream pump is NOT started.
The browser must subscribe to the SFU track via the REST endpoints described in
[API_V1.md](API_V1.md#sfu-transport). Avatar speech audio still arrives over
the existing WebSocket messages (`avatar.speech.*`, `assistant.audio`).

#### `session.ended`

Session fully torn down. The client should close the WebSocket after receiving
this.

```json
{
  "type": "session.ended",
  "sessionId": "session:8f3c...",
  "wasActive": true,
  "modalEndOk": true,
  "modalEndDurationMs": 1820
}
```

| Field | Type | Description |
|-------|------|-------------|
| `sessionId` | `string\|null` | Session ID. `null` if session was cancelled mid-start. |
| `wasActive` | `boolean` | Whether the session had an active Modal backend. |
| `modalEndOk` | `boolean` | Whether the Modal end call succeeded. |
| `modalEndDurationMs` | `number` | Modal end call latency. |
| `modalEndError` | `string` | Error message if `modalEndOk` is false. |

#### `status`

Generic lifecycle/pipeline state notification. The `status` string carries
the semantic.

```json
{ "type": "status", "status": "thinking" }
```

Known `status` values:

| Value | Meaning |
|-------|---------|
| `starting_modal` | Modal session start HTTP call in flight |
| `rendering` | Debug frame render requested (legacy mode) |
| `rendering_avatar_audio` | Avatar lip-sync render for speech in progress |
| `transcribing` | STT request sent |
| `thinking` | LLM request sent |
| `synthesizing` | TTS request sent |
| `mic_ready` | Microphone open; server accepting audio |
| `mic_stopped` | Microphone closed |
| `listening_same_turn` | Smart Turn predicts user still speaking; same turn continues |
| `ending` | Modal end call in flight |

#### `error`

Error notification. See [Error Semantics](#error-semantics) for observed codes.

```json
{ "type": "error", "code": "modal_unreachable", "message": "...", "fatal": false }
```

| Field | Type | Description |
|-------|------|-------------|
| `code` | `string` | Machine-readable error code. |
| `message` | `string` | Human-readable error message. |
| `fatal` | `boolean` | Optional. If omitted, treat as `false`. If `true`, the server may close (or has already closed) the socket. |

### Media — fMP4 transport (primary)

These messages carry the fragmented MP4 stream for MSE (Media Source
Extensions) playback. This is the primary transport (`websocket_fmp4`).

#### `avatar.fmp4.init`

fMP4 initialization segment. Sent once at stream start. Feed to
`MediaSource.SourceBuffer.appendBuffer()`.

```json
{
  "type": "avatar.fmp4.init",
  "dataBase64": "AAAAGGZ0...",
  "byteLength": 1024,
  "timingMs": 45
}
```

| Field | Type | Description |
|-------|------|-------------|
| `dataBase64` | `string` | Base64-encoded fMP4 init segment. |
| `byteLength` | `number` | Decoded byte length. |
| `timingMs` | `number` | Time to generate this segment. |

#### `avatar.fmp4.segment`

fMP4 media segment for MSE playback. Feed to
`MediaSource.SourceBuffer.appendBuffer()`.

```json
{
  "type": "avatar.fmp4.segment",
  "dataBase64": "AAAAM4Z0...",
  "byteLength": 4096,
  "sequence": 1,
  "mode": "idle",
  "isSpeechStart": false,
  "isSpeechEnd": false,
  "turnId": null,
  "segmentId": 1,
  "chunkIndex": 0,
  "chunkRenderMs": 200,
  "feedMs": 12,
  "framesFed": 5
}
```

**Core fields** (needed for playback):

| Field | Type | Description |
|-------|------|-------------|
| `dataBase64` | `string` | Base64-encoded fMP4 media segment. |
| `byteLength` | `number` | Decoded byte length. |
| `sequence` | `number` | Monotonically increasing segment number. |
| `mode` | `string` | `"idle"` or `"speech"`. Indicates what the segment contains. |
| `isSpeechStart` | `boolean` | `true` on the first segment of a speech turn. |
| `isSpeechEnd` | `boolean` | `true` on the last segment of a speech turn. |
| `turnId` | `number\|null` | Speech turn ID. `null` during idle mode. |

**Diagnostic fields** (for latency monitoring; not needed for playback):

| Field | Type | Description |
|-------|------|-------------|
| `segmentId` | `number` | Internal segment counter. |
| `chunkIndex` | `number` | Chunk index within the current render batch. |
| `chunkRenderMs` | `number` | Render time for this chunk. |
| `feedMs` | `number` | Time to feed audio to the renderer. |
| `framesFed` | `number` | Number of video frames fed. |
| `audioFeedMs` | `number` | *(AV mode only)* Audio feed time. |
| `audioSamplesPerChunk` | `number` | *(AV mode only)* Audio samples per chunk. |
| `containerInstanceId` | `string` | *(Internal)* Modal container ID. |
| `overlapWithTts` | `boolean` | *(Internal)* Whether this chunk overlaps TTS. |

> **Client implementation note:** Only `dataBase64`, `sequence`, `mode`,
> `isSpeechStart`, `isSpeechEnd`, and `turnId` are needed for basic playback.
> Diagnostic fields are useful for latency dashboards but can be ignored.

#### `avatar.timeline.started`

A continuous media timeline (idle + speech) has started streaming.

```json
{ "type": "avatar.timeline.started", "fps": 25, "mode": "fmp4_av" }
```

| Field | Type | Description |
|-------|------|-------------|
| `fps` | `number` | Target FPS. |
| `mode` | `string` | `"fmp4_av"`, `"fmp4_video"`, or `"jpeg_timeline"`. |

#### `avatar.timeline.stopped`

The media timeline has ended.

```json
{ "type": "avatar.timeline.stopped", "reason": "fmp4_stream_ended" }
```

| Field | Type | Description |
|-------|------|-------------|
| `reason` | `string` | Why the timeline stopped: `stream_ended`, `fmp4_stream_ended`, `fmp4_av_stream_ended`. |

### Media — JPEG timeline (legacy)

These messages are used only in `jpeg_timeline` render mode. In the current
`websocket_fmp4` transport, media arrives via fMP4 messages instead.

#### `avatar.frame` *(legacy)*

Single debug/static frame. Response to `render_debug_frame`.

#### `avatar.speech.start` *(legacy)*

Start of avatar lip-sync video for a speech turn (JPEG frame mode).

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `totalFrames` | `number` | Total frames. May be `0` when unknown. |
| `fps` | `number` | Frames per second. |
| `renderedAudioDurationMs` | `number` | Rendered audio duration. |
| `originalAudioDurationMs` | `number` | Original TTS audio duration. |

#### `avatar.speech.chunk` *(legacy)*

Batch of JPEG frames for avatar speech rendering.

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `chunkIndex` | `number` | Chunk index. |
| `frames` | `array` | Array of `{ dataBase64, mimeType, width, height }`. |
| `isLast` | `boolean` | `true` on the final chunk for this turn. |

#### `avatar.sfu.speech_queued`

SFU mode only. Notifies the browser that a speech turn's audio has been
successfully enqueued for the Modal SFU pipeline.

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `chunks` | `number` | Number of audio chunks queued. |
| `samples` | `number` | Total PCM samples queued. |

#### `avatar.idle.start` *(legacy)*

Idle (listening) animation loop begins.

| Field | Type | Description |
|-------|------|-------------|
| `totalFrames` | `number` | Total frames in the idle loop. |
| `fps` | `number` | Frames per second. |
| `durationMs` | `number` | Total duration. |

#### `avatar.idle.chunk` *(legacy)*

Batch of JPEG frames for idle animation.

| Field | Type | Description |
|-------|------|-------------|
| `chunkIndex` | `number` | Chunk index. |
| `frames` | `array` | Array of `{ dataBase64, mimeType, width, height }`. |
| `isLast` | `boolean` | `true` on the final chunk. |
| `live` | `boolean` | `true` for live-rendered (non-cached) stream. |

### Speech pipeline

These messages track the conversational turn lifecycle: user speech →
transcription → LLM response → TTS synthesis → avatar rendering.

#### `speech.started`

VAD detected speech start; a new turn has opened.

```json
{ "type": "speech.started", "turnId": 1 }
```

#### `speech.segment.started`

A new speech segment within an already-open turn (VAD re-triggered after a
pause within the same turn).

```json
{ "type": "speech.segment.started", "turnId": 1, "segmentIndex": 1 }
```

#### `speech.continues`

Smart Turn evaluated but predicts the user is still speaking. The turn stays
open.

```json
{
  "type": "speech.continues",
  "turnId": 1,
  "smartTurn": { "prediction": 0, "probability": 0.32, "rawLogit": -0.75 }
}
```

Always followed by `status: "listening_same_turn"`.

#### `speech.finalized`

Turn closed. The STT → LLM → TTS pipeline will run for this turn.

```json
{
  "type": "speech.finalized",
  "turnId": 1,
  "smartTurn": { "prediction": 1, "probability": 0.91, "rawLogit": 2.3 },
  "smartTurnFallback": false
}
```

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `smartTurn` | `object` | Smart Turn decision (see below). |
| `smartTurnFallback` | `boolean` | `true` if the result was synthetic (not from real inference — e.g., mic.stop force-finalize). |

**`SmartTurnDecision` object:**

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `0\|1` | 1 = turn complete, 0 = still speaking. |
| `probability` | `number` | Confidence (0.0–1.0). |
| `rawLogit` | `number` | Raw model logit. |
| `bufferSamples` | `number` | Audio buffer size in samples. *(diagnostic)* |
| `bufferApproxSeconds` | `number` | Approximate buffer duration. *(diagnostic)* |
| `processedSamples` | `number` | Processed samples. *(diagnostic)* |
| `processedApproxSeconds` | `number` | Processed duration. *(diagnostic)* |
| `trimmedToMaxSeconds` | `boolean` | Whether audio was trimmed. *(diagnostic)* |

#### `transcript.final`

STT (speech-to-text) result for a finalized turn.

```json
{
  "type": "transcript.final",
  "turnId": 1,
  "text": "Hello, how are you?",
  "languageCode": "en",
  "provider": "sarvam",
  "latencyMs": 340
}
```

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `text` | `string` | Transcribed text. |
| `languageCode` | `string` | Detected language code. |
| `provider` | `string` | STT provider name. |
| `latencyMs` | `number` | STT latency. |

#### `assistant.text`

LLM response text.

```json
{
  "type": "assistant.text",
  "turnId": 1,
  "text": "I'm doing well, thank you!",
  "provider": "openai-compatible",
  "latencyMs": 890
}
```

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `text` | `string` | LLM response. |
| `provider` | `string` | LLM provider name. |
| `latencyMs` | `number` | LLM latency. |

#### `assistant.audio`

TTS (text-to-speech) audio result.

```json
{
  "type": "assistant.audio",
  "turnId": 1,
  "mimeType": "audio/wav",
  "provider": "sarvam",
  "latencyMs": 210,
  "sampleRate": 16000,
  "audioBytes": 51200,
  "dataBase64": "UklGRi..."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `turnId` | `number` | Speech turn ID. |
| `dataBase64` | `string` | Base64-encoded WAV audio. **Omitted in `fmp4_av` mode** (audio is muxed into the fMP4 stream). |
| `mimeType` | `string` | Always `"audio/wav"`. |
| `provider` | `string` | TTS provider name. |
| `latencyMs` | `number` | TTS latency. |
| `sampleRate` | `number` | Always `16000`. |
| `audioBytes` | `number` | Audio byte count. `0` when `directEnqueued` is true. |
| `directEnqueued` | `boolean` | Audio was pushed directly to Modal renderer (no base64 payload). |
| `audioDurationMs` | `number` | Duration of the synthesized audio. |

> **fMP4 AV mode:** The `dataBase64` field is omitted because audio is muxed
> into the fMP4 stream. The client should NOT play it separately. In
> `fmp4_video` mode, `dataBase64` is present and should be played via Web Audio.

#### `audio.stats`

Periodic audio statistics (sent every 50 received chunks).

```json
{
  "type": "audio.stats",
  "chunksReceived": 50,
  "bytesReceived": 32000
}
```

---

## Error Semantics

There are three distinct error categories. They have different formats and
delivery mechanisms:

### 1. HTTP upgrade errors (REST JSON envelope)

Returned as HTTP responses when the WebSocket upgrade is rejected. These use
the canonical `{ success, error, timings }` JSON envelope.

| HTTP | `error.code` | Cause |
|------|-------------|-------|
| 400 | `invalid_session_id` | Session ID doesn't start with `session:` |
| 401 | `ws_token_missing` | No `?token=` query parameter |
| 401 | `ws_token_invalid` | Token malformed, bad signature, expired, wrong session, or wrong purpose |
| 404 | `not_found` | Session doesn't exist or belongs to another tenant |
| 429 | `rate_limited` | `ws_connect` rate limit exceeded (5/min) |
| 503 | `ws_token_secret_not_configured` | `AVTR_SESSION_TOKEN_SECRET` not set |

### 2. WebSocket error messages (JSON on the socket)

Sent as `{ type: "error", code, message, fatal? }` on an established WebSocket
connection. These are **not** HTTP responses.

#### Fatal errors (`fatal: true`)

| Code | Cause | Action |
|------|-------|--------|
| `invalid_session` | Session ID not found after hibernation restore | Server closes socket (1011). Create a new session. |

#### Session lifecycle errors (`fatal: false`)

| Code | Cause | Client action |
|------|-------|---------------|
| `session_already_active` | `session.start` sent on an already-active session | Do not resend. The session is live — media should already be streaming. |
| `session_starting` | `session.start` sent while a start is in progress (<180s) | Wait for `session.ready` or `error`. |
| `session_cancelled` | Session was ended (DELETE/session.end) during the start | Session is gone. Create a new session. |
| `no_modal_protocol_enabled` | Server has no Modal backend configured | Server misconfiguration. Contact platform team. |
| `modal_start_failed` | Modal backend rejected the start request | Check avatar/background IDs. Retry with a new session. |
| `avatar_not_found` | Requested avatar is unknown or unavailable | Choose an avatar from `GET /v1/avatars`. |
| `background_not_found` | Requested background is unknown or unavailable | Choose a background from `GET /v1/backgrounds` or omit `bgId`. |
| `modal_unreachable` | Modal backend network error / timeout | Retry after a brief wait. |
| `sfu_not_configured` | Client explicitly requested `webrtc_sfu` transport but SFU prerequisites are missing (`AVTR_REALTIME_SFU_ENABLED`, Cloudflare Calls credentials) | Server config issue. Retry with default transport (`websocket_fmp4`). |
| `sfu_beta_not_enabled` | Client explicitly requested `webrtc_sfu` transport but the session is not SFU-beta-authorized | Request SFU beta access via a beta-authorized tenant or a valid SFU beta token, then create a new session. |
| `sfu_unavailable` | SFU Modal start or subscriber/answer call failed | Retry after a brief wait or fall back to fMP4. |
| `sfu_turn_credentials_unavailable` | TURN relay credentials could not be generated (missing TURN Key, API failure, or no usable TURN URLs). SFU requires TURN for production-grade ICE reliability. | Server config issue. Ensure `AVTR_SFU_TURN_KEY_ID` and `AVTR_SFU_TURN_KEY_API_TOKEN` are set, or set `AVTR_SFU_REQUIRE_TURN=false` for dev diagnostics. |
| `sfu_capacity_exceeded` | Another SFU publisher is already active on the single-GPU Modal runtime. Only one SFU session at a time is supported. | Wait for the existing SFU session to end (`session.ended`), then retry. |
| `sfu_capacity_unavailable` | Capacity manager (singleton DO) is unreachable. Fail-closed to protect the single GPU from overlap. | Transient infrastructure failure. Retry shortly. |
| `sfu_direct_enqueued` | TTS audio was direct-enqueued in SFU mode without browser audio bytes | Server pipeline error. Retry or fall back to fMP4. |
| `no_active_session` | `render_debug_frame` sent without an active session | Send `session.start` first. |
| `render_frame_blocked` | Debug frame requested while timeline is running | Wait for timeline to stop. |
| `render_unavailable` | Render returned no frame | Retry or check avatar availability. |
| `render_failed` | Exception during frame render | Retry or report the error. |
| `no_modal_session` | Internal: no Modal session ID for fMP4 stream start | Server error. Create a new session. |

#### Pipeline errors (`fatal: false`)

| Code | Cause | Client action |
|------|-------|---------------|
| `invalid_json` | Message failed JSON.parse | Check your message serialization. |
| `unknown_type` | Unrecognized message `type` | Check the message type spelling. |
| `invalid_audio_chunk` | Binary frame empty or odd byte count | Ensure PCM16 chunks are non-empty with even byte count. |
| `stt_skipped` | No STT API key configured | Server misconfiguration. |
| `stt_failed` | STT transcription threw | Transcription failed for this turn; avatar will not respond. |
| `llm_skipped` | No LLM base URL configured | Server misconfiguration. |
| `llm_failed` | LLM call threw | LLM failed for this turn; avatar will not respond. |
| `tts_failed` | TTS synthesis returned null or threw | TTS failed for this turn; avatar may not speak. |
| `avatar_render_failed` | Avatar render/stream failed | Avatar video failed for this turn. Audio may still play. |

> Pipeline errors are per-turn. A failed STT/LLM/TTS/render does not kill the
> session — the user can speak again and the next turn may succeed. Only
> `invalid_session` is fatal (closes the socket).

### 3. REST API errors (HTTP JSON envelope)

Returned by `/v1/*` REST endpoints (not the WebSocket). See
[API_V1.md > Error Codes](./API_V1.md#error-codes) for the full list.

---

## Render Modes

The `renderMode` in `session.ready` determines how media is delivered:

| Mode | Video | Audio | Messages used |
|------|-------|-------|---------------|
| `fmp4_av` | fMP4 segments | Muxed in fMP4 segments | `avatar.fmp4.init`, `avatar.fmp4.segment` |
| `fmp4_video` | fMP4 segments | Separate WAV via `assistant.audio` | `avatar.fmp4.init`, `avatar.fmp4.segment`, `assistant.audio` |
| `jpeg_timeline` | JPEG frame chunks | Separate WAV via `assistant.audio` | `avatar.speech.start/chunk`, `avatar.idle.start/chunk`, `avatar.frame` |

The current production transport is `websocket_fmp4` with `fmp4_av` or
`fmp4_video` mode. The `jpeg_timeline` mode is legacy.

When `transport: "webrtc_sfu"` is used, `renderMode` is omitted and media is
delivered over WebRTC via the Cloudflare Realtime SFU — see
[Transport Selection](#transport-selection) and
[API_V1.md > SFU Transport](API_V1.md#sfu-transport). In Milestone 1, only video
is delivered over WebRTC; speech audio continues via the existing WS messages.

### fMP4 client implementation

A browser client uses Media Source Extensions (MSE):

```javascript
const mediaSource = new MediaSource();
videoEl.src = URL.createObjectURL(mediaSource);

mediaSource.addEventListener("sourceopen", () => {
  const sb = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E"');
  // fmp4_av also needs audio codec: 'video/mp4; codecs="avc1.42E01E,mp4a.40.2"'

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === "avatar.fmp4.init") {
      sb.appendBuffer(base64ToBuffer(msg.dataBase64));
    } else if (msg.type === "avatar.fmp4.segment") {
      sb.appendBuffer(base64ToBuffer(msg.dataBase64));
    }
  };
});
```

Append the init segment first, then all segments in order. MSE handles
playback buffering automatically.

---

## Turn Lifecycle

A speech "turn" is one conversational exchange:

```
User speaks                    Server                      Client
    |                            |                            |
    |-- binary PCM16 audio ---->|                            |
    |                            |                            |
    |                            |<- VAD detects speech ------|
    |<-- speech.started ---------|                            |
    |                            |                            |
    |                            |<- Smart Turn evaluates ----|
    |<-- speech.continues -------|   (still speaking)         |
    |<-- status: listening_same_turn                         |
    |                            |                            |
    |                            |<- Smart Turn: complete ----|
    |<-- speech.finalized -------|                            |
    |                            |                            |
    |                            |--- STT ------------------> |
    |<-- transcript.final -------|                            |
    |                            |--- LLM -------------------> |
    |<-- assistant.text ---------|                            |
    |                            |--- TTS -------------------> |
    |<-- assistant.audio --------|                            |
    |                            |                            |
    |<-- avatar.fmp4.segment     |   (isSpeechStart: true)    |
    |    (speech mode)           |                            |
    |                            |                            |
    |--- avatar.speech.playback.ended (turnId) -------------> |
    |                            |                            |
    |<-- avatar.fmp4.segment     |   (isSpeechEnd: true)      |
    |<-- avatar.fmp4.segment     |   (idle mode resumes)      |
```

The `turnId` correlates all messages belonging to one conversational exchange.
It increments monotonically per session.
