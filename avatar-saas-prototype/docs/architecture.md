# Architecture

Current implementation of the avatar SaaS prototype.

## Product Constraint: Browser is Passive

The browser/web client is a **passive debug player only**. It does not own or
authoritate any stream timing, speech state, pacing, muxing, seeking, or
recovery decisions. All authority lives server-side in Modal and the Bridge.

This is a hard constraint for the future API/SaaS product and Cloudflare
real-time transport. The same server-paced muxed A/V source that currently feeds
the debug browser must feed API consumers, WebRTC tracks, and Cloudflare
Workers streams without modification.

## Current Data Flow (Implemented)

```
Browser Client (debug player, passive)
    |
    | WebSocket
    |   - up: mic PCM16 audio chunks, session/mic control messages
    |   - down: fMP4 init + media segments (SSE relayed through Bridge)
    v
Cloudflare Session Bridge (Worker + Durable Object)
    |
    |--- Voice pipeline:
    |       VAD DO (any-toy-vad) -> Smart Turn DO (any-toy-smart-turn)
    |       -> STT (Sarvam Saarika) -> LLM (Cerebras GPT-OSS-120B)
    |       -> TTS routing decision:
    |           a) Same-GPU OmniVoice (L40S in-process handoff)
    |           b) External OmniVoice (separate Modal app)
    |           c) Sarvam fallback
    |
    |--- Modal SSE stream: POST to Modal -> SSE fMP4 A/V stream
    |--- Bridge relays each SSE event -> browser WebSocket ws.send()
    |
    v
Modal Avatar Runtime (L40S GPU)
    |
    | AVTR-1 autoregressive renderer + OmniVoice TTS (same GPU)
    | FFmpeg NVENC H.264 + AAC -> fragmented MP4
    | _SegmentSender: server-side source playout pacing
    | _AudioHandoff: in-process TTS-to-AVTR audio queue
    |
    | SSE stream of fMP4 init + media segments
    v
Cloudflare Session Bridge (relay, no buffering)
    |
    | ws.send() per segment (no batching, no queueing)
    v
Browser Client (MSE SourceBuffer append, passive)
```

## Planes

### Control + Transport Plane (Cloudflare)

The Session Bridge Worker + Durable Object owns:

- Session lifecycle (create, WebSocket upgrade, session state, teardown)
- Mic audio routing through voice agent pipeline (VAD, Smart Turn, STT, LLM, TTS)
- TTS routing: same-GPU OmniVoice, external OmniVoice, or Sarvam fallback
- SSE stream relay from Modal to browser WebSocket (no buffering, no batching)
- Speech state management (speech.started, speech.finalized, transcript, assistant.text/audio)
- PostHog observability (all events: pipeline, relay, browser relay, stream summary)
- Per-segment send gap tracking (bridgeMaxSendGapMs)
- fMP4 segment forwarding counters (received, sent, failed)

**The Bridge does NOT:** authoritate pacing, buffer, time, or make stream
cadence decisions. It is a transparent SSE-to-WebSocket relay.

### Compute Plane (Modal L40S)

The Modal L40S app (`avtr1-lifecycle-parallel-l40s`) owns:

- AVTR-1 autoregressive avatar renderer (5 frames per chunk at 25fps = 200ms media)
- OmniVoice TTS inference (same GPU, `@modal.concurrent(max_inputs=2)`)
- Same-GPU audio handoff: `_AudioHandoff` in-process queue (not modal.Queue, not base64/WAV)
- FFmpeg NVENC H.264 + AAC encoding -> fragmented MP4 mux
- **Server-side source playout pacing** via `_SegmentSender`:
  - Burst-fills during startup until `TARGET_LEAD_S` (4.0s) of media lead
  - Then yields at steady 200ms intervals (real-time cadence)
  - Pacing sleep prevents GPU-speed burst delivery to browser
- Speech marker management: `isSpeechStart`/`isSpeechEnd` per turn
- GPU memory + utilization snapshots (periodic, not per-segment)
- Cadence diagnostics: max/avg inter-yield gap, sender queue depth, dry spell count

**The Modal app does NOT:** depend on the browser for timing, seeking, or
state recovery. It produces a self-paced fMP4 A/V stream that any consumer can
play.

#### TTS Architecture: Full-Generation-First (NOT Streaming)

Current OmniVoice path is **synchronous full-utterance**:

```
LLM text -> OmniVoice.generate() blocks until full waveform
          -> Resample to 16kHz
          -> Split into step-sized chunks (3200 samples / 200ms)
          -> Push all chunks to _AudioHandoff queue
          -> AVTR consumes chunks at real-time cadence
```

OmniVoice uses a Diffusion LLM (MaskGIT-style iterative unmasking, Qwen3-0.6B
backbone + HiggsAudio V2 codec). `generate()` is synchronous and returns the
complete waveform after all N unmasking steps. There is no streaming API, no
callback, no partial output.

**Impact:** The full TTS generation time (2-12s depending on text length) is
the dominant latency before first avatar speech video appears. The `_SegmentSender`
absorbs the burst audio and paces emission at real-time cadence.

**Proposed next optimization (not yet implemented):** Sentence-level TTS.
Split LLM output into sentences, call `generate()` per sentence, push each
sentence's audio to `_AudioHandoff` as soon as ready. This would reduce
first-byte latency from "full TTS time" to "first sentence TTS time" while
preserving natural pauses between sentences.

### Browser Client (Debug Player Only)

The web client at `avatar-web-client.pages.dev` is a **passive debug player**:

- Captures microphone audio, sends PCM16 chunks via WebSocket
- Receives fMP4 init + media segments via WebSocket
- Appends to MSE SourceBuffer (no seeking, no live-edge authority)
- Reports segment counts, buffer stats, and video events to PostHog
- Sends `avatar.speech.playback.ended` when video playback passes speech end
  (acknowledgment only, not authoritative)

**The browser does NOT:**
- Decide when to play or pause (MSE/HTMLVideoElement handles this automatically)
- Manage speech state or turn transitions
- Seek to live edge or control playback position
- Time or pace segment delivery
- Make any stream cadence decisions

## Deployed Services

| Service | URL | Runtime |
|---------|-----|---------|
| Web Client | `https://avatar-web-client.pages.dev` | Cloudflare Pages |
| Session Bridge | `https://avatar-session-bridge.vallangirakesh.workers.dev` | Worker + DO |
| VAD | `https://any-toy-vad.vallangirakesh.workers.dev` | Worker + DO |
| Smart Turn | `https://any-toy-smart-turn.vallangirakesh.workers.dev` | Worker + DO |
| Modal L40S | `https://vallangirakesh--avtr1-lifecycle-parallel-l40s-avatarlife-9c93ac.modal.run` | Modal GPU |

## Key Parameters

| Parameter | Value | Configurable |
|-----------|-------|-------------|
| GPU | NVIDIA L40S (46GB VRAM) | Modal app config |
| Resolution | 1280x720 | Avatar config |
| FPS | 25 | AVTR renderer |
| Frames per chunk | 5 | AVTR renderer |
| Chunk media duration | 200ms | 5 / 25fps |
| Audio sample rate | 16000 Hz | AVTR / OmniVoice |
| AVTR window | 6480 samples (405ms) | AVTR renderer |
| AVTR step | 3200 samples (200ms) | AVTR renderer |
| TARGET_LEAD_S | 4.0s | `AVTR1_TARGET_LEAD_S` env var |
| HANDOFF_WAIT_S | 0.05s (50ms) | Hardcoded |
| FFmpeg codec | H.264 NVENC p4 ull | Hardcoded |
| FFmpeg fragment | 200000us +frag_keyframe | Hardcoded |
| FFmpeg GOP | 25 frames (1s) | Hardcoded |
| Concurrent inputs | 2 (render + TTS) | `@modal.concurrent` |

## Current Flags (Bridge wrangler.jsonc)

| Flag | Value | Purpose |
|------|-------|---------|
| ENABLE_MODAL_FMP4_AV_STREAM | true | fMP4 A/V mode |
| ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS | true | Same-GPU TTS on L40S |
| OMNIVOICE_SAME_GPU_PARALLEL_TARGET | l40s | Target GPU |
| ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY | true | Per-segment diagnostics (temporary) |

## Observability

All events are captured via PostHog (project 146226, EU region). Key events:

| Event | Source | Purpose |
|-------|--------|---------|
| `avatar_fmp4_av_speech_started` | Modal | First speech segment per turn |
| `avatar_fmp4_av_speech_ended` | Modal | Last speech segment per turn |
| `avatar_fmp4_av_segment_forwarded` | Bridge | Per-segment relay counter |
| `avatar_fmp4_browser_segment_appended` | Browser | Per-segment append counter |
| `avatar_fmp4_av_stream_summary` | Bridge (from Modal end) | Session summary with cadence + GPU stats |
| `avatar_fmp4_av_stream_stopped` | Bridge | Stream teardown |
| `avatar_fmp4_video_event` | Browser | Video waiting/canplay/playing events |

### Cadence Stats (in stream_summary.modalCadenceStats)

| Field | Description |
|-------|-------------|
| `maxModalInterYieldGapMs` | Max wall-clock gap between paced segment yields |
| `avgModalInterYieldGapMs` | Average inter-yield gap |
| `totalPacingSleepMs` | Total time spent in pacing sleep |
| `senderMinQueueDepth` | Minimum sender buffer depth (0 = dry spell) |
| `senderMaxQueueDepth` | Peak sender buffer depth |
| `senderDrySpellCount` | How many times yield_events() found no segments |

### GPU Stats (in stream_summary)

| Field | Description |
|-------|-------------|
| `gpuSmiUsedMinMb/MaxMb` | nvidia-smi memory used range |
| `gpuSmiTotalMb` | Total VRAM |
| `gpuSmiGpuUtilMinPct/MaxPct/AvgPct` | GPU compute utilization |
| `gpuSmiMemoryUtilMinPct/MaxPct/AvgPct` | GPU memory bandwidth utilization |
| `gpuTorchAllocMinMb/MaxMb` | PyTorch allocated memory |
| `gpuTorchPeakMb` | PyTorch peak memory |

## What This Is Not

- **Not Pipecat.** We are not using Pipecat or any third-party conversation
  framework. The Cloudflare voice agent pipeline handles conversation flow.
- **Not the existing live streamer.** We replaced WebRTC/aiortc with
  Cloudflare-based orchestration + Modal GPU rendering.
- **Not a monolith.** Session bridge, avatar runtime, and voice agent pipeline
  are separate services.
- **Not browser-authoritative.** The browser is a passive consumer.

## Known Issues (Current)

1. **TTS is full-generation, not streaming.** First speech segment latency is
   dominated by OmniVoice generate() time (2-12s). See "Proposed next
   optimization" above.
2. **Max inter-yield gap ~1174ms.** The `_SegmentSender` maintains cadence
   after startup, but dry spells (sender queue drains to 0) can cause gaps up
   to ~1.2s during GPU contention. TARGET_LEAD_S=4.0s absorbs most of these.
3. **Bridge max send gap ~1585ms.** Mirrors Modal gaps + SSE framing overhead.
   Bridge adds no buffering.
4. **Initial startup waiting events.** Browser has 4-5 "waiting" events during
   the first ~20s while the playout buffer fills. After TARGET_LEAD_S is
   reached, waiting events stop for the rest of the session.
5. **Verbose telemetry enabled.** `ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY=true`
   should be removed once stable.

## Next Phase: API Product + Cloudflare Real-Time Transport

The current architecture produces a server-paced fMP4 A/V stream from Modal.
The next phase will:

1. **API/SaaS output path:** Expose the same fMP4 stream as an API endpoint.
   Consumers (mobile apps, third-party integrations) receive the same paced
   stream without browser involvement.

2. **Cloudflare real-time transport:** Replace SSE relay with Cloudflare
   Workers Streams, WebRTC (via Cloudflare Calls), or HTTP/2 push. The Modal
   source remains unchanged; only the transport layer changes.

3. **Sentence-level TTS:** Split LLM output into sentences, generate each
   sentence's audio independently, push to AVTR as ready. Reduces first-byte
   latency from full-utterance time to first-sentence time.

4. **GPU compute optimization:** Investigate render time reduction during TTS
   overlap. Current render is ~100-300ms per chunk depending on GPU contention.

5. **Private repo refactor:** Extract production paths from the prototype
   codebase into a clean, deployable service. Preserve the passive browser
   debug player as a separate tool.
