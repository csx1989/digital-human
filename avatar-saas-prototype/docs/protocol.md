# Protocol

First-draft event protocol for the avatar SaaS prototype.

This is a **draft**. All event names, payloads, and transport details are
subject to change once we validate the runtime path and benchmark the renderer.
Nothing here is implemented.

Message formats are described as JSON for readability. The actual wire format
may differ (binary, protobuf, flatbuffers) pending performance testing.

Audio and video payload formats (PCM, YUV, VP9, etc.) are not final decisions.
They are placeholder candidates based on current AVTR-1 renderer observations
and will be revised during implementation.

## Channels

```
Browser <---> Cloudflare Session Bridge <---> Modal Avatar Runtime
  (WebSocket)     |                          (HTTP/WebSocket TBD)
                  |
                  v
          Existing Voice Agent Pipeline
          (VAD, Smart Turn, STT, LLM, TTS)
```

## 1. Client to Bridge

Events sent from the browser client to the Cloudflare session bridge.

### `session.start`

Initiate a new avatar session.

```json
{
  "type": "session.start",
  "kidId": "string",
  "characterId": "string",
  "avatarId": "string",
  "sessionId": "string (optional, for reconnection)"
}
```

### `audio.input`

Stream user audio to the bridge.

```json
{
  "type": "audio.input",
  "data": "<base64-encoded audio bytes>",
  "format": "pcm16 | opus | webm (TBD)",
  "sampleRate": 16000,
  "timestamp": 1234567890
}
```

The format field is a placeholder. PCM16 at 16 kHz is a likely starting point
because it matches the existing voice agent pipeline's internal format, but
web-friendly formats (Opus) may be preferred for the public API.

### `session.end`

End the current session.

```json
{
  "type": "session.end",
  "reason": "user_initiated | timeout | error"
}
```

## 2. Bridge to Client

Events sent from the Cloudflare session bridge to the browser client.

### `session.ready`

Session is active and ready for audio input.

```json
{
  "type": "session.ready",
  "sessionId": "string",
  "audioFormat": "pcm16",
  "sampleRate": 16000
}
```

### `transcript.partial`

Partial STT transcript (user is still speaking).

```json
{
  "type": "transcript.partial",
  "text": "string",
  "turnId": 1
}
```

### `transcript.final`

Final STT transcript for a completed turn.

```json
{
  "type": "transcript.final",
  "text": "string",
  "turnId": 1
}
```

### `assistant.text`

LLM response text.

```json
{
  "type": "assistant.text",
  "text": "string",
  "turnId": 1
}
```

### `assistant.audio`

TTS audio chunk for playback.

```json
{
  "type": "assistant.audio",
  "data": "<base64-encoded audio bytes>",
  "format": "pcm16 (placeholder)",
  "sampleRate": 24000,
  "turnId": 1,
  "chunkIndex": 0,
  "isFinal": false
}
```

### `avatar.stream.ready`

Avatar renderer is warmed up and streaming. The client should start rendering
incoming frames.

```json
{
  "type": "avatar.stream.ready",
  "frameFormat": "yuv_i420 | vp9 | av1 (placeholder, not final)",
  "width": 720,
  "height": 1280,
  "fps": 25
}
```

The frame format is a placeholder. The actual format will depend on browser
feasibility testing (raw YUV bandwidth vs. VP9/AV1 encoding latency).

### `error`

```json
{
  "type": "error",
  "code": "string",
  "message": "string",
  "fatal": false
}
```

## 3. Bridge to Modal Avatar Runtime

Internal events from the Cloudflare session bridge to the Modal-hosted renderer.

These are internal implementation details, not a public API.

### `avatar.session.start`

Start a new rendering session on the GPU.

```json
{
  "type": "avatar.session.start",
  "sessionId": "string",
  "avatarId": "string",
  "backgroundId": "string",
  "config": {
    "cfgSelfAudio": 2.0,
    "cfgOtherAudio": 2.0,
    "cfgKp": 4.0,
    "pixelFormat": "yuv_i420 (placeholder)"
  }
}
```

Config parameters are observed from the current AVTR-1 renderer defaults.
They are not final API decisions.

### `avatar.speak`

Send TTS audio to the renderer for a speech segment.

```json
{
  "type": "avatar.speak",
  "sessionId": "string",
  "audio": "<base64-encoded PCM bytes>",
  "format": "pcm16 at 16kHz (placeholder)",
  "segmentId": "string",
  "isFinal": false
}
```

### `avatar.session.end`

End the rendering session and release GPU resources.

```json
{
  "type": "avatar.session.end",
  "sessionId": "string"
}
```

## 4. Modal Avatar Runtime to Bridge

Responses from the Modal renderer back to the session bridge.

### `avatar.ready`

Rendering session is initialized and ready to accept audio.

```json
{
  "type": "avatar.ready",
  "sessionId": "string"
}
```

### `avatar.frame` (placeholder)

A rendered video frame. The exact payload structure is TBD.

Candidate options:

- Raw YUV I420 bytes (observed from current renderer, ~1.4 MB per frame at
  720x1280, likely too bandwidth-heavy for browser delivery)
- VP9 or AV1 encoded chunk (lower bandwidth, adds encoding latency)
- A sequence of frames per render chunk (current renderer produces 5 frames
  per call at 25 FPS = 200ms of video)

This will be defined after Phase 3 benchmarking.

### `avatar.audio.chunk` (placeholder)

Rendered audio synchronized with the video frames. May be passthrough TTS
audio or may include renderer-generated audio (e.g., for the listen track).

Exact format TBD.

### `avatar.error`

```json
{
  "type": "avatar.error",
  "sessionId": "string",
  "code": "string",
  "message": "string"
}
```
