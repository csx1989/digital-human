# Phase 5A: Cloudflare Bridge Architecture Audit + Adaptation Plan

**Status: Completed (planning phase)**
**Date: 2026-05-29**

This document records the audit of the old toy-agent Cloudflare implementation,
decisions on what to reuse vs redesign, and the proposed architecture for the
new avatar-specific Cloudflare bridge.

No runtime code was changed. This is a planning document only.

---

## A. Old Toy-Agent Architecture Summary

### Files Inspected

All files under `/Users/ayya/developer/my-toy/cloudflare_toy_agent/`:

| File | Role | Lines |
|------|------|-------|
| `any-toy-agent/src/index.ts` | Worker entrypoint, routing, device auth, memory admin | 497 |
| `any-toy-agent/src/toy-agent.ts` | ToyAgentDO — main voice agent (VAD, Smart Turn, STT, LLM, TTS) | 1341 |
| `any-toy-agent/src/vad-session.ts` | VadSessionDO — simple RMS-based VAD (legacy, unused in prod) | 112 |
| `any-toy-agent/src/posthog-client.ts` | PostHog deferred observability via `ctx.waitUntil` | 114 |
| `any-toy-agent/src/device-auth.ts` | HMAC session token auth for ESP32 devices | 215 |
| `any-toy-agent/src/device-registry.ts` | DeviceRegistryDO — provisioning, pairing, secret rotation | 382 |
| `any-toy-agent/src/pcm-tts.ts` | Multi-provider TTS (Sarvam/Fish/OmniVoice/MeloTTS) | 1240 |
| `any-toy-agent/src/characters.ts` | Hardcoded character configs (system prompts, greetings, TTS speakers) | 40 |
| `any-toy-agent/src/openai-compatible-llm.ts` | OpenAI-compatible streaming LLM provider | ~300 |
| `any-toy-agent/src/sarvam-stt.ts` | Sarvam batch STT with language detection | ~200 |
| `any-toy-agent/src/sarvam-streaming-stt.ts` | Sarvam streaming STT | ~200 |
| `any-toy-agent/src/kid-memory.ts` | KidMemoryDO — Vectorize-backed memory storage/recall | ~300 |
| `any-toy-agent/src/fish-tts-presets.ts` | Fish TTS voice preset configs | ~50 |
| `any-toy-agent/wrangler.jsonc` | Bindings: AI, VAD_SERVICE, SMART_TURN_SERVICE, 4 DOs | 100 |
| `any-toy-agent/package.json` | Dependencies: @cloudflare/agents, @cloudflare/voice, posthog-node | 30 |
| `vad-worker/src/index.ts` | ONNX Silero VAD in WorkerEntrypoint + VADModelDO | 266 |
| `vad-worker/wrangler.jsonc` | Standalone VAD worker config with ONNX model bundling | 39 |
| `smart-turn-worker/src/index.ts` | ONNX Smart Turn in WorkerEntrypoint + SmartTurnDO | 301 |
| `smart-turn-worker/wrangler.jsonc` | Standalone Smart Turn worker config | 39 |

### Files in This Repo Also Inspected

| File | Role |
|------|------|
| `docs/roadmap.md` | Phase roadmap — Phases 1-4A complete, Phase 5 next |
| `docs/architecture.md` | Target architecture (Cloudflare ↔ Modal ↔ Browser) |
| `docs/protocol.md` | Draft event protocol (session.start, audio.input, avatar.frame, etc.) |
| `cloudflare-session-bridge/README.md` | Skeleton bridge README (placeholder only) |
| `cloudflare-session-bridge/src/index.ts` | Skeleton Worker (health, sessions, WebSocket echo) |
| `local-lifecycle-app/README.md` | Phase 4A local server docs |
| `modal-avatar-runtime/modal_lifecycle_app.py` | Phase 4A Modal L4 snapshot runtime |

### Request/WebSocket Flow (Connect → Speech End)

1. **ESP32** calls `GET /device/session-token` with `x-device-id` + `x-device-secret`
   (`index.ts:175-203`)
2. `issueBootstrapSession()` verifies device via `DeviceRegistryDO`, issues HMAC
   session token, returns `{ wsUrl }` with `?auth=...&kidId=...&characterId=...`
   (`device-auth.ts:156-214`)
3. **ESP32** connects WebSocket to `wss://.../agents/toy/{kid_id}?auth=...`
4. `index.ts:455-488` — `verifyDeviceSessionToken()` validates HMAC, then
   `routeAgentRequest()` from `@cloudflare/agents` routes to `ToyAgentDO` named
   `{kid_id}`
5. `toy-agent.ts:401-467` — `onConnect()` reads URL params (kidId, characterId,
   ttsProvider), resets VAD/SmartTurn sessions, primes session memory from
   `KidMemoryDO`
6. `toy-agent.ts:635-648` — `onCallStart()` triggers greeting via `speak()`
7. **ESP32** sends binary PCM audio frames (16kHz stereo interleaved)
8. `toy-agent.ts:210-394` — Custom `onMessage()` interceptor:
   - Converts stereo → mono (`toy-agent.ts:1166-1186`)
   - Runs VAD via `#processVadChunk()` (`toy-agent.ts:868-875`) — calls
     `VAD_SERVICE.processAudio(sessionName, chunk)` via service binding RPC
   - On `shouldStart`: injects `{ type: "start_of_speech" }` to voice handler
     (`toy-agent.ts:274-292`)
   - During speech: accumulates Smart Turn audio via `#accumulateSmartTurnAudio()`
     (`toy-agent.ts:887-905`) — calls `SMART_TURN_SERVICE.accumulateAudio()`
   - On `shouldEnd`: runs `#evaluateSmartTurn()` (`toy-agent.ts:907-951`) — calls
     `SMART_TURN_SERVICE.analyzeTurn()`. If prediction=1, injects
     `{ type: "end_of_speech" }`. If prediction=0, suppresses end-of-speech
     (`toy-agent.ts:344-364`).
   - Passes mono audio to `voiceOnMessage()` for STT processing
   - On Smart Turn error: falls back to VAD-only decision (`toy-agent.ts:366-393`)
9. `@cloudflare/voice` SDK handles: STT → `onTurn()` → LLM streaming → TTS →
   `speak()`
10. `toy-agent.ts:650-777` — `speak()`:
    - Sends `{ type: "status", status: "speaking" }`
    - Synthesizes full audio via `this.tts.synthesize(text)`
    - Streams 1KB PCM chunks with `setTimeout` pacing (64ms intervals)
    - Sends `{ type: "assistant_audio_end" }`, transcript, `{ status: "listening" }`
11. `toy-agent.ts:779-788` — `onCallEnd()`: resets VAD + SmartTurn sessions,
    emits `agent_disconnected` PostHog event
12. `toy-agent.ts:790-799` — `onInterrupt()`: sets `#speakAborted`, resets SmartTurn

### Durable Object Layout

| DO Class | Binding | Instance ID | Storage | Purpose |
|----------|---------|-------------|---------|---------|
| `ToyAgentDO` | `toy` | `{kid_id}` | SQLite | Main voice agent per kid |
| `VadSessionDO` | `vad` | `{kid_id}:vad` | In-memory | Legacy simple VAD (not used in prod) |
| `DeviceRegistryDO` | `device_registry` | `{device_id}` | SQLite | Device provisioning, auth |
| `KidMemoryDO` | `kid_memory` | `{kid_id}` | SQLite + Vectorize | Conversation memory |

Two standalone workers with their own DOs:
- `any-toy-vad` → `VADModelDO` (ONNX Silero VAD, per-session state, `vad-worker/src/index.ts:61-216`)
- `any-toy-smart-turn` → `SmartTurnDO` (ONNX Smart Turn, buffered audio + inference, `smart-turn-worker/src/index.ts:230-301`)

### Identity Model

- **Device → Kid → Character**: ESP32 has a `deviceId` provisioned by admin, mapped
  to a `kidId` and optional `characterId`
- Auth: HMAC session token with `sub=deviceId`, `kidId`, `path`, `exp`
  (`device-auth.ts:81-100`)
- DO routing: `ToyAgentDO` named by `kid_id` — one DO per child
- `routeAgentRequest()` from `@cloudflare/agents` maps `/agents/toy/{kid_id}` to
  the DO instance

### VAD Service Binding Pattern

```typescript
// wrangler.jsonc:46-49
"services": [{ "binding": "VAD_SERVICE", "service": "any-toy-vad" }]

// toy-agent.ts:868-875 — RPC call
async #processVadChunk(chunk: ArrayBuffer): Promise<VadDecision> {
    const vadService = this.env.VAD_SERVICE as VadServiceBinding | undefined;
    if (vadService?.processAudio) {
        return vadService.processAudio(this.#vadSessionName, chunk);
    }
    // fallback to local VadSessionDO...
}
```

VAD worker (`vad-worker/src/index.ts:45-58`): `WorkerEntrypoint` exposing
`processAudio(sessionId, chunk)` and `reset(sessionId)` as RPC methods.
Internally creates per-session `VADModelDO` instances running ONNX Silero VAD
inference.

### Smart Turn Service Binding Pattern

```typescript
// wrangler.jsonc:50-53
"services": [{ "binding": "SMART_TURN_SERVICE", "service": "any-toy-smart-turn" }]

// toy-agent.ts:887-905 — accumulate during speech
await service.accumulateAudio(this.#smartTurnSessionName, monoChunk);

// toy-agent.ts:907-951 — analyze on VAD end
const result = await service.analyzeTurn(this.#smartTurnSessionName);

// toy-agent.ts:953-982 — reset on cleanup
await service.reset(this.#smartTurnSessionName);
```

Smart Turn worker (`smart-turn-worker/src/index.ts:142-213`): `WorkerEntrypoint`
exposing `accumulateAudio()`, `analyzeTurn()`, `reset()` as RPC methods.
`SmartTurnDO` accumulates PCM chunks, computes Whisper mel spectrogram, runs ONNX
inference. Returns `{ prediction: 0|1, probability, rawLogit }`.

### STT/LLM/TTS Provider Flow

- **STT**: `@cloudflare/voice` SDK handles transcription via `beforeTranscribe` /
  `afterTranscribe` hooks. Sarvam streaming STT as primary provider. Language
  detection attached to turn context (`toy-agent.ts:839-866`).
- **LLM**: `OpenAICompatibleLLMProvider` — streams text via `streamText()`, yields
  chunks for streaming TTS. Cerebras router at `LLM_BASE_URL` with Qwen model
  (`openai-compatible-llm.ts`).
- **TTS**: `PCMTTSProvider` implements `TTSProvider` interface. Multi-provider
  chain: OmniVoice → Fish → Sarvam → MeloTTS. Outputs raw 16kHz PCM
  (`pcm-tts.ts:60-185`). `synthesizeStream()` handles buffering of tiny leading
  fragments (`pcm-tts.ts:805-1018`).

### Cleanup/Reset Flow

```
onCallEnd() (toy-agent.ts:779-788):
  - serverSpeechActive = false
  - reset VAD session (VAD_SERVICE.reset)
  - reset Smart Turn session (SMART_TURN_SERVICE.reset)
  - captureEvent("agent_disconnected")

onInterrupt() (toy-agent.ts:790-799):
  - speakAborted = true (stops in-progress speak() streaming)
  - serverSpeechActive = false
  - reset Smart Turn session
  - captureEvent("agent_interrupted")
```

### PostHog Observability Pattern

```typescript
// posthog-client.ts:79-114
// Deferred pattern: fires observability in ctx.waitUntil, never blocks the
// critical path.

function schedule(waitUntil: WaitUntilFn | undefined, promise: Promise<void>): void {
    if (waitUntil) {
        waitUntil(promise.catch((err) => console.error(`[PostHog] error:`, err)));
    } else {
        void promise.catch((err) => console.error(`[PostHog] error:`, err)));
    }
}

export function captureEventDeferred(env, waitUntil, event, properties?): void {
    schedule(waitUntil, captureEvent(env, event, properties));
}

export function captureErrorDeferred(env, waitUntil, error, context?): void {
    schedule(waitUntil, captureError(env, error, context));
}
```

Every lifecycle event, provider call, error, and timing metric is captured with
`kidId`, `turnId`, and relevant context properties.

---

## B. What We Should Reuse

### 1. VAD service binding RPC pattern
**Source**: `toy-agent.ts:868-875`, `wrangler.jsonc:46-49`, `vad-worker/src/index.ts:45-58`

The `WorkerEntrypoint` + service binding + RPC method pattern is clean and
proven. In Phase 6, the new bridge will call the same deployed `any-toy-vad`
worker via `VAD_SERVICE.processAudio(sessionId, chunk)` and
`VAD_SERVICE.reset(sessionId)`.

### 2. Smart Turn service binding RPC pattern
**Source**: `toy-agent.ts:887-982`, `wrangler.jsonc:50-53`, `smart-turn-worker/src/index.ts:142-213`

Same pattern. `SMART_TURN_SERVICE.accumulateAudio()`, `.analyzeTurn()`, `.reset()`.

### 3. PostHog deferred observability
**Source**: `posthog-client.ts:79-114`

The `schedule()` + `waitUntil()` pattern is excellent. Copy `posthog-client.ts`
into the new project with minimal changes (update `distinctId`, remove
toy-specific defaults).

### 4. Error fallback discipline
**Source**: `toy-agent.ts:366-393` (Smart Turn error → VAD-only fallback),
`toy-agent.ts:590-632` (LLM error → static fallback),
`pcm-tts.ts:120-185` (TTS provider chain fallback)

Every external call has a fallback. Smart Turn errors degrade to VAD-only. LLM
errors return a static message. TTS cascades through providers. This defensive
pattern is essential for real-time voice.

### 5. Audio PCM16 utilities
**Source**: `toy-agent.ts:1324-1341`

`computeAudioRms()`, `computeAudioPeak()`, `convertInterleavedStereoToMono()`.
Reusable directly in the new bridge.

### 6. WebSocket hibernation
**Source**: Cloudflare Workers documentation — Durable Objects WebSocket
Hibernation API

Use `this.ctx.acceptWebSocket(server)` + `webSocketMessage()` / `webSocketClose()`
handlers. Survives memory eviction. Better than the old `ws.addEventListener`
pattern. See Cloudflare docs reference below.

### 7. Session naming convention
**Source**: `toy-agent.ts:181-182`

`{sessionId}:vad` and `{sessionId}:smart-turn` for VAD/ST session isolation.
Clean and proven. The new bridge uses `{avatarSessionId}:vad` and
`{avatarSessionId}:smart-turn`.

### 8. Service binding config style
**Source**: `wrangler.jsonc:46-55`

Same `"services"` array pattern for VAD and Smart Turn. New wrangler.jsonc will
point at existing deployed workers (in Phase 6).

### 9. Multi-provider TTS fallback chain pattern
**Source**: `pcm-tts.ts:120-185`

The pattern of trying provider A, falling back to B, then C is sound. Do NOT
copy the TTS code directly (we intercept TTS output for avatar rendering), but
adopt the fallback chain pattern.

---

## C. What We Must Not Reuse

### 1. `@cloudflare/voice` / `withVoice(Agent)` / `routeAgentRequest`
**Decision**: Raw Durable Objects instead.

The `@cloudflare/voice` SDK's hooks (`onTurn`, `speak`, `beforeSynthesize`,
`afterSynthesize`) are too restrictive for our avatar rendering interception
point. We need full control over the audio pipeline to route TTS output through
the Modal avatar renderer before sending frames to the browser.

The `routeAgentRequest()` from `@cloudflare/agents` is also unnecessary — we use
standard `env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sessionId))`
directly.

### 2. `@cloudflare/agents` package
Not needed. Standard DO instantiation is sufficient.

### 3. Device/kid identity model
**Source**: `device-auth.ts`, `device-registry.ts`, `kid-memory.ts`, `characters.ts`

No ESP32, no `DeviceRegistryDO`, no `KidMemoryDO`, no `device-auth.ts`, no
`characters.ts`. Browser sessions use a different identity model:
`avatarSessionId` (UUID) with optional future `userId`/`projectId`.

### 4. KidMemoryDO / Vectorize memory
Not in scope for the avatar prototype. May be added later.

### 5. Stereo audio assumption
**Source**: `toy-agent.ts:1166-1186` (`convertInterleavedStereoToMono`)

The old code assumes ESP32 sends interleaved stereo. Browser `AudioWorklet` or
`MediaRecorder` sends mono by default. The stereo-to-mono conversion may still
be useful as a defensive measure but must not be the default assumption.

### 6. `VadSessionDO` in main worker
**Source**: `vad-session.ts` (112 lines)

This is the legacy simple RMS-based VAD. Production uses the ONNX VAD worker.
Do not copy.

### 7. `speakAborted` / `#serverSpeechActive` state tracking
**Source**: `toy-agent.ts:183-184`, `toy-agent.ts:679-687`

The old interrupt pattern is tightly coupled to `@cloudflare/voice`'s `speak()`
method. Our custom pipeline needs its own interrupt mechanism.

### 8. TTS direct-to-device streaming
**Source**: `pcm-tts.ts:698-737`, `toy-agent.ts:650-777`

The old code streams TTS PCM chunks directly to the ESP32. For avatar, TTS audio
must be routed to the Modal renderer first. The chunking and pacing are
different.

### 9. Modal Python SDK assumption
Cloudflare Workers cannot call `modal.Cls.from_name()` or `inst.method.remote()`.
The bridge calls Modal exclusively via HTTP `fetch()`. This must be explicit in
all documentation and code.

### 10. `/agents/toy/{kid_id}` URL routing
Different URL scheme for browser avatar sessions. The new bridge uses
`/sessions` and `/sessions/:id/ws`.

---

## D. Proposed New Avatar Cloudflare Architecture

### D1. Worker/Bridge Shape

```
avatar-saas-prototype/cloudflare-session-bridge/
├── src/
│   ├── index.ts              # Worker entrypoint, routing
│   ├── avatar-session-do.ts  # AvatarSessionDO — main session DO
│   ├── posthog-client.ts     # Copied from toy-agent (adapted)
│   ├── audio-utils.ts        # PCM utilities (reused)
│   ├── modal-client.ts       # HTTP client for Modal avatar runtime
│   └── types.ts              # Shared types/interfaces
├── wrangler.jsonc             # Bindings: AVATAR_SESSION DO
├── package.json
├── tsconfig.json
└── README.md
```

Phase 6 additions (not in 5B):
```
│   ├── stt-provider.ts       # STT provider
│   ├── llm-provider.ts       # LLM provider
│   ├── tts-provider.ts       # TTS provider
```

### D2. Durable Object Decision: YES

Avatar sessions need per-session state (Modal session ID, turn counter,
VAD/ST session names, conversation history). Durable Objects provide:

- Persistent state across WebSocket messages (even across hibernation)
- Single-threaded per-session guarantee (no race conditions)
- `ctx.waitUntil()` for deferred observability
- SQLite storage for conversation history

**DO Layout**:

| DO Class | Binding | Instance ID | Storage | Purpose |
|----------|---------|-------------|---------|---------|
| `AvatarSessionDO` | `AVATAR_SESSION` | `session:{uuid}` | SQLite | One per browser session |

No `DeviceRegistryDO`, no `KidMemoryDO`.

### D3. WebSocket Hibernation and State Persistence

The DO uses Cloudflare's Hibernatable WebSocket API:

```typescript
// In DO fetch():
this.ctx.acceptWebSocket(server);  // enables hibernation

// Handlers (called even after hibernation wake-up):
async webSocketMessage(ws: WebSocket, message: string | ArrayBuffer) { ... }
async webSocketClose(ws: WebSocket, code: number, reason: string, wasClean: boolean) { ... }
```

**Critical**: In-memory private fields on the DO may reset after hibernation.
The runtime can evict the DO from memory while the WebSocket is idle, then
re-instantiate it (running the constructor) when a new message arrives. This
means:

- Session-critical state (Modal session ID, avatar ID, session active flag)
  must be persisted in Durable Object storage (`this.ctx.storage.put()`) or
  WebSocket attachment metadata (`ws.serializeAttachment()` /
  `ws.deserializeAttachment()`).
- Private fields (`#sessionActive`, `#modalSessionId`, etc.) are only valid
  for the current in-memory lifetime. On wake from hibernation, they must be
  restored from storage or attachment.
- Heavy objects (VAD state, audio buffers) can remain in-memory-only since they
  are rebuilt per-turn anyway, but the DO must handle the case where they are
  `undefined` after wake-up.

Reference: Cloudflare docs on "Persist WebSocket Connection State with
serializeAttachment/deserializeAttachment" — see External References below.

### D4. Session ID Model

```
Session lifecycle:
1. Browser → POST /sessions → { sessionId: "session:a1b2c3d4" }
2. Browser → WebSocket /sessions/session:a1b2c3d4/ws
3. DO instance: env.AVATAR_SESSION.idFromName("session:a1b2c3d4")
4. VAD session (Phase 6): "session:a1b2c3d4:vad"
5. Smart Turn session (Phase 6): "session:a1b2c3d4:smart-turn"
6. Modal session: started via HTTP call to Modal lifecycle endpoint, ID
   returned in response body. Stored in DO storage.
```

The primary identity is `avatarSessionId` (the `session:{uuid}` string). Future
`userId` and `projectId` can be added as optional qualifiers when auth is
implemented.

### D5. Browser WebSocket Protocol

```
Client → Server:
  { type: "session.start", avatarId?: string, bgId?: string }
  { type: "render_debug_frame" }              (Phase 5B)
  [binary audio chunks — PCM16 16kHz mono]    (Phase 6)
  { type: "session.end", reason?: string }

Server → Client:
  { type: "session.ready", sessionId, avatarId, width, height, fps }
  { type: "avatar.frame", data: "<base64 JPEG>", frameIndex }
  { type: "transcript.partial", text, turnId }        (Phase 6)
  { type: "transcript.final", text, turnId }          (Phase 6)
  { type: "assistant.text", text, turnId }            (Phase 6)
  { type: "assistant.audio", data: "<base64 PCM>", turnId, chunkIndex }  (Phase 6)
  { type: "status", status: "listening"|"thinking"|"speaking"|"rendering" }
  { type: "error", code, message, fatal }
```

This diverges from the draft `protocol.md` but is pragmatic: JSON metadata +
binary audio on the same WebSocket, avatar frames as base64 JSON initially (not
binary, to keep Phase 5B simple). The protocol can be refined in later phases.

### D6. Modal Lifecycle Connection Strategy

#### Constraint: Cloudflare cannot call Modal Python SDK

Cloudflare Workers run JavaScript. They cannot use `modal.Cls.from_name()` or
`inst.method.remote()`. All Modal communication is via HTTP `fetch()`.

#### Approach: `@modal.fastapi_endpoint()` on the existing class

The preferred implementation adds `@modal.fastapi_endpoint()` methods directly
to the existing `AvatarLifecycleL4` class in `modal_lifecycle_app.py`. This
preserves the existing snapshot/lifecycle behavior because `@modal.fastapi_endpoint()`
shares the same container and the same loaded pipeline as `@modal.method()` and
`@modal.enter(snap=True)`.

Modal documentation confirms:

> "Combine container lifecycle hooks with web endpoint decorators like
> `@modal.fastapi_endpoint` within a Modal class."
> — Modal docs, "Apply lifecycle hooks to web endpoints"
> (https://modal.com/docs/guide/lifecycle-functions)

The pattern from Modal examples:

```python
@app.cls(gpu="H100", enable_memory_snapshot=True, ...)
class Inference:
    @modal.enter(snap=True)
    def load(self):
        self.pipeline = ...  # captured in snapshot

    @modal.method()
    def run(self, ...):
        ...  # programmatic access (Python SDK)

    @modal.fastapi_endpoint(method="POST")
    def web(self, request: Request):
        ...  # HTTP access — same container, same self.pipeline
```

**What this means for our code**: Add ONE `@modal.fastapi_endpoint(method="POST")`
method to `AvatarLifecycleL4` that accepts a JSON body with an `action` field and
dispatches to shared internal helper methods. Both the existing `@modal.method()`
functions and the new web endpoint call these same helpers — no `.local()` calls
from within the class.

```python
# Shared helpers (private):
def _do_health(self) -> dict: ...
def _do_start_session(self, avatar_id, bg_id) -> dict: ...
def _do_render_frame(self, session_id) -> dict: ...
def _do_end_session(self, session_id) -> dict: ...

# @modal.method() functions call helpers:
@modal.method()
def health(self) -> dict:
    return self._do_health()

# Web endpoint calls same helpers:
@modal.fastapi_endpoint(method="POST")
def handle(self, body: dict) -> dict:
    action = body.get("action")
    if action == "health": return self._do_health()
    ...
```

**What NOT to do**: Do not add a separate `@modal.asgi_app()` or standalone
`@modal.fastapi_endpoint()` function that creates its own pipeline instance.
Any HTTP endpoint that independently loads the pipeline would duplicate GPU
memory and bypass the snapshot. Do NOT use `.local()` calls from inside the
class — extract shared helpers instead.

#### URL is configuration, not source-code knowledge

The Modal web endpoint URL is assigned by Modal on deploy. It is visible in:
- `modal deploy` CLI output
- Modal dashboard
- `Cls.from_name("avtr1-lifecycle", "AvatarLifecycleL4").get_web_url()`
  (from Python, for automated retrieval)

The Cloudflare bridge receives this URL as an environment variable:

```
MODAL_LIFECYCLE_URL = "https://<assigned-by-modal>/handle"
```

This is the **full URL** of the single web endpoint. Cloudflare POSTs directly
to it with `{ "action": "...", ...payload }`. Do not assume a base URL + path
pattern — the URL is whatever Modal assigns.

Set in `wrangler.jsonc` `vars` or as a Cloudflare secret.

#### Single web endpoint on `AvatarLifecycleL4`

One `@modal.fastapi_endpoint(method="POST")` method accepts JSON:

```json
{ "action": "health" | "start" | "render_frame" | "end", ... }
```

| Action | Extra Fields | Returns |
|--------|-------------|---------|
| `health` | (none) | GPU status, pipeline state |
| `start` | `avatar_id?`, `bg_id?` | `{ sessionId, status: "avatar_ready", ... }` |
| `render_frame` | `session_id?` | `{ available, dataBase64, mimeType, width, height }` |
| `end` | `session_id?` | `{ ok, sessionId, wasActive }` |

Future (Phase 6+):

| `render_audio` | `audio_base64` | `{ frames: [...] }` |

### D7. Voice Pipeline Flow (Phase 6, documented for completeness)

```
Browser mic audio (PCM16 16kHz mono)
    │
    ▼ [WebSocket binary frame]
AvatarSessionDO.webSocketMessage()
    │
    ├──► VAD_SERVICE.processAudio(sessionId+":vad", chunk)
    │       │
    │       ▼ shouldStart
    │       Start speech tracking
    │
    ├──► SMART_TURN_SERVICE.accumulateAudio(sessionId+":st", monoChunk)
    │       (during speech)
    │
    │   ▼ shouldEnd + Smart Turn prediction=1
    │       │
    │       ▼
    │   STT: Send accumulated audio to STT provider
    │       │
    │       ▼ transcript
    │   LLM: Stream response from LLM provider
    │       │
    │       ▼ text chunks
    │   TTS: Synthesize text to PCM audio
    │       │
    │       ▼ PCM audio
    │   Modal: POST /session/render-audio with PCM
    │       │
    │       ▼ rendered frames (JPEG/YUV)
    │   WebSocket: Send frames + audio to browser
    │       │
    │       ▼
    │   status: "listening"
```

### D8. Stop/Disconnect Cleanup

```
webSocketClose() / session.end:
  1. Call Modal POST /session/end (release GPU state)
  2. (Phase 6) Call VAD_SERVICE.reset(sessionId+":vad")
  3. (Phase 6) Call SMART_TURN_SERVICE.reset(sessionId+":st")
  4. Clear session state from DO storage
  5. captureEventDeferred("session_ended", { sessionId, duration })
```

---

## E. Phase 5B Implementation Plan

**One broad task**: Get a browser or test client → Cloudflare DO → Modal
lifecycle round-trip working over WebSocket. No voice pipeline. Session start,
debug frame, session end only.

### E1. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `cloudflare-session-bridge/src/index.ts` | **Modify** | Add DO routing, `/sessions` POST, `/sessions/:id/ws` WebSocket upgrade to DO |
| `cloudflare-session-bridge/src/avatar-session-do.ts` | **Create** | AvatarSessionDO with hibernatable WebSocket, session state, Modal HTTP client calls |
| `cloudflare-session-bridge/src/modal-client.ts` | **Create** | Thin `fetch()` wrapper for Modal HTTP endpoints with error handling and timing |
| `cloudflare-session-bridge/src/types.ts` | **Create** | Shared interfaces (session state, Modal response types, WebSocket messages) |
| `cloudflare-session-bridge/wrangler.jsonc` | **Modify** | Add `AVATAR_SESSION` DO binding, `MODAL_LIFECYCLE_URL` var |
| `cloudflare-session-bridge/README.md` | **Modify** | Update with actual architecture, run commands |
| `modal-avatar-runtime/modal_lifecycle_app.py` | **Modify** | Add `@modal.fastapi_endpoint()` methods to `AvatarLifecycleL4` |

### E2. What to Implement First (Phase 5B Scope)

1. **AvatarSessionDO** with hibernatable WebSocket:
   - `fetch()` — WebSocket upgrade handler, persist session state in storage
   - `webSocketMessage()` — parse JSON messages (`session.start`,
     `render_debug_frame`, `session.end`), route to handlers
   - `webSocketClose()` — cleanup: call Modal end, clear storage

2. **Modal client** — `startSession()`, `renderDebugFrame()`, `endSession()`
   via HTTP `fetch()` to `MODAL_LIFECYCLE_URL`

3. **Worker routing**:
   - `POST /sessions` — generate session ID, return it
   - `GET /sessions/:id/ws` — WebSocket upgrade to AvatarSessionDO
   - `GET /health` — health check

4. **Browser/test client session flow**:
   - Client sends `{ type: "session.start" }`
   - DO calls Modal `POST /session/start`, receives `avatar_ready`
   - DO sends `{ type: "session.ready", sessionId, ... }` to client
   - Client sends `{ type: "render_debug_frame" }`
   - DO calls Modal `POST /session/render-frame`, receives JPEG
   - DO sends `{ type: "avatar.frame", data: "..." }` to client
   - Client sends `{ type: "session.end" }`
   - DO calls Modal `POST /session/end`, cleans up

### E3. Future Binding Shape (Phase 6 only)

These bindings are NOT added in Phase 5B. Documented here for planning:

```jsonc
// Future wrangler.jsonc additions (Phase 6):
"services": [
    { "binding": "VAD_SERVICE", "service": "any-toy-vad" },
    { "binding": "SMART_TURN_SERVICE", "service": "any-toy-smart-turn" }
]
```

### E4. What Stays Mocked or Documented (Not Implemented in 5B)

- **VAD**: No binding, no calls. Documented as Phase 6 dependency.
- **Smart Turn**: No binding, no calls. Documented as Phase 6 dependency.
- **STT**: Not implemented. Interface documented in `types.ts`.
- **LLM**: Not implemented. Interface documented in `types.ts`.
- **TTS**: Not implemented. Interface documented in `types.ts`.
- **Auth**: No auth for Phase 5B. Browser connects directly.
- **PostHog**: `posthog-client.ts` copied, minimal events in 5B.
- **Audio streaming from browser mic**: Not in 5B.
- **Avatar rendering from audio**: Not in 5B (debug frame only).

### E5. Explicit Phase 5B Non-Goals

- No voice pipeline (VAD, STT, LLM, TTS)
- No audio streaming from browser microphone
- No avatar rendering from audio (only debug frame from silent chunk)
- No auth / API keys
- No PostHog events beyond basic session lifecycle
- No VAD_SERVICE or SMART_TURN_SERVICE bindings in wrangler.jsonc
- No multi-session stress testing
- No frame encoding optimization (JPEG base64 is sufficient for 5B)
- No changes to the old toy-agent project

### E6. Verification Commands

```bash
# 1. Modal: deploy lifecycle app with new HTTP endpoints
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py

# 2. Retrieve Modal web URL (from CLI output or dashboard)
# Set as MODAL_LIFECYCLE_URL in wrangler.jsonc or as a secret

# 3. Type check
cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit

# 4. Local dev
cd avatar-saas-prototype/cloudflare-session-bridge && npx wrangler dev

# 5. Verify no changes outside avatar-saas-prototype/
git diff --stat

# 6. Verify no Python runtime changes (modal_lifecycle_app.py changes are
#    additive HTTP decorators only, no logic changes)
git diff avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py
```

---

## External Documentation References

1. **Cloudflare Workers — Durable Objects WebSocket Hibernation API**
   - https://developers.cloudflare.com/durable-objects/best-practices/websockets
   - `ctx.acceptWebSocket(server)`, `webSocketMessage()`, `webSocketClose()`,
     `serializeAttachment()`, `deserializeAttachment()`

2. **Cloudflare Workers — Service Bindings RPC**
   - https://developers.cloudflare.com/workers/runtime-apis/bindings/service-bindings/rpc
   - `WorkerEntrypoint`, RPC method calls between Workers

3. **Modal — Lifecycle Functions**
   - https://modal.com/docs/guide/lifecycle-functions
   - `@modal.enter(snap=True)` for snapshot-included initialization
   - Combining lifecycle hooks with `@modal.fastapi_endpoint()` on the same class

4. **Modal — GPU Memory Snapshots**
   - https://modal.com/docs/guide/memory-snapshots
   - `enable_memory_snapshot=True`, `experimental_options={"enable_gpu_snapshot": True}`
   - Alpha feature, redeploys invalidate snapshots

5. **Modal — Web Endpoint URLs**
   - https://modal.com/docs/guide/webhook-urls
   - `get_web_url()` for programmatic URL retrieval
   - URL is assigned by Modal on deploy, visible in CLI output and dashboard

6. **Modal — `@modal.fastapi_endpoint()` on Classes**
   - https://modal.com/docs/examples/fine_tune_asr
   - https://modal.com/docs/examples/text_to_image
   - Pattern: `@modal.enter()` + `@modal.method()` + `@modal.fastapi_endpoint()`
     on the same class, sharing the same container and loaded model
