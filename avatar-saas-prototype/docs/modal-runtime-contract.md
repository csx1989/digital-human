# Modal Runtime Contract

Proposed API and implementation strategy for the AVTR-1 Modal wrapper (Phase 3).

This document defines what the Modal GPU service must expose, how it maps to the
existing renderer, and the recommended first implementation path.

---

## Recommended Phase 3 Approach

### Wrap the existing renderer, do not redesign it

The existing `avtr1_renderer.api.app` is a clean FastAPI HTTP service with:

- Well-defined multipart request format
- Streaming response with structured headers
- Safetensors state serialization
- Health and metadata endpoints

**Phase 3 should run this existing app inside a Modal container**, not rewrite
it. The only additions are:

1. A **session layer** that keeps `AVTR1State` on GPU between calls (eliminating
   the per-call D2H/H2D serialization overhead)
2. A **container lifecycle** that preloads `Pipeline.from_artifacts()` at startup
3. A **health check** that confirms GPU readiness

Everything else (audio parsing, motion generation, rendering, frame packing)
delegates to the existing pipeline unchanged.

### Why not create a new API from scratch

- The existing API has been tested and benchmarked by the streamer
- Rewriting introduces unnecessary risk with no benefit
- The streamer's `RendererClient` already proves the HTTP contract works
- Phase 3 can reuse `app.py`'s `_build_chunk()` and `run_chunk()` directly

### What changes from the current architecture

| Aspect | Current (local) | Phase 3 (Modal) |
|--------|----------------|-----------------|
| State ownership | Client-carried safetensors blob | Server-side per-session on GPU (recommended) |
| Transport | HTTP localhost | HTTP over network (Cloudflare bridge) |
| Container lifecycle | Manual subprocess | Modal auto-scaling |
| Load balancing | External LB or single-instance | Modal handles (or single instance for prototype) |
| Avatar loading | All at startup from local disk | All at startup from Modal Volume or NetworkFS |

---

## Proposed Phase 3 Modal Endpoints

### `GET /health`

Health check. Returns GPU status.

**Request**: None.

**Response**:
```json
{
  "status": "ok",
  "gpu": "available",
  "avatars_loaded": 1,
  "pipeline_ready": true
}
```

**Maps to**: Reuses `CudaHealthChecker` from `app.py:275-284`. Adds
`avatars_loaded` and `pipeline_ready` for Modal readiness probes.

**TBD**: None.

---

### `POST /session/start`

Initialize a rendering session. Preloads avatar data and creates initial state.

**Request**:
```json
{
  "avatar_id": "anya_03_studio",
  "bg_id": "bg_01",
  "pixel_format": "yuv_i420",
  "cfg_self_audio": 2.0,
  "cfg_other_audio": 2.0,
  "cfg_kp": 3.0,
  "noise_alpha": 2.0,
  "noise_trunc_z": 1.2
}
```

All config fields except `avatar_id` and `bg_id` are optional with defaults
matching the current `RenderOptions`.

**Response**:
```json
{
  "session_id": "uuid-string",
  "frame_height": 720,
  "frame_width": 1280,
  "fps": 25,
  "frames_per_chunk": 5,
  "current_samples": 3200,
  "future_samples": 3280
}
```

**Maps to**: 
- `avatar_id` → lookup in `Pipeline`'s avatar registry (same as current `app.py:180-186`)
- Initial state → `Pipeline.initial_state(avatar)` (currently done inside `process_chunk` when `state=None`)
- Sample counts → `motion_generator.chunk_size * motion_generator.frame_len` and `motion_generator.future_size * motion_generator.frame_len + motion_generator.audio_shift`

**What is TBD**:
- Whether to allow `pixel_format` and CFG overrides per-chunk or lock them per session
- Whether session IDs are UUIDs or Modal-assigned identifiers
- Session timeout and cleanup behavior

---

### `POST /session/chunk`

Process one audio chunk within an active session. Returns rendered frames and
updates server-side state.

**Request**: `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `session_id` | `str` (form) | Yes | Session from `/session/start` |
| `current_chunk` | `UploadFile` | Yes | Present speech audio, int16 PCM 16kHz, 3200 samples = 6400 bytes |
| `future_chunk` | `UploadFile` | Yes | Future speech audio, int16 PCM 16kHz, 3280 samples = 6560 bytes |
| `current_chunk_listen` | `UploadFile` | Yes | Present listen audio, int16 PCM 16kHz, 3200 samples = 6400 bytes |
| `future_chunk_listen` | `UploadFile` | Yes | Future listen audio, int16 PCM 16kHz, 3280 samples = 6560 bytes |

**Response**: Same as `/process-audio-v3` — streaming body with headers:

| Header | Example | Description |
|--------|---------|-------------|
| `X-Num-Frames` | `"5"` | Frames in response |
| `X-Frame-Height` | `"720"` | Frame height |
| `X-Frame-Width` | `"1280"` | Frame width |
| `X-Frame-Length-Bytes` | `"1382400"` | Bytes per frame |
| `X-State-Length-Bytes` | `"0"` | State now server-side, may be 0 |
| `X-Has-State` | `"no"` | State no longer in body |
| `X-Session-Id` | `"uuid"` | Echo session ID |

Body: `[frame_0] [frame_1] ... [frame_4]` (no state blob if server-side state).

**Maps to**:
- Audio parsing → identical to `_build_chunk()` in `app.py:66-80`
- Pipeline call → `Pipeline.process_chunk(avatar, chunk, state, options)` in `pipeline.py:258-319`
- Response streaming → same `StreamingResponse` pattern from `app.py:236-256`

**What is TBD**:
- Whether `X-State-Length-Bytes` should still be non-zero for debugging/fallback
- Whether to allow per-chunk `bg_id` or `pixel_format` overrides
- Error response format on invalid session_id or audio size mismatch

**Key difference from current `/process-audio-v3`**:
- No `state` upload field — state lives server-side on GPU
- No `avatar_id`, `bg_id`, or config params per chunk — locked at session start
- State blob may be omitted from response body (saves ~627 KiB D2H per call)

---

### `POST /session/end`

Terminate a session and release GPU state.

**Request**:
```json
{
  "session_id": "uuid-string"
}
```

**Response**:
```json
{
  "session_id": "uuid-string",
  "chunks_processed": 42,
  "session_duration_ms": 16800
}
```

**Maps to**: No current equivalent. The local renderer has no session concept.

**What is TBD**:
- Whether to return session metrics (latency histogram, total frames)
- Cleanup timeout if client disconnects without calling `/session/end`

---

## State Management Recommendation

### Option A: Server-side state (recommended for Phase 3)

Keep `AVTR1State` on GPU between calls. The session layer stores state in a
`dict[session_id, AVTR1State]` in process memory.

**Tradeoffs**:

| Aspect | Server-side | Client-carried (current) |
|--------|-------------|-------------------------|
| D2H/H2D overhead | Zero | ~627 KiB x 2 per call |
| Latency per call | Lower | Higher (serialization overhead) |
| Horizontal scaling | Sticky sessions required | Stateless, any worker can serve |
| Crash recovery | State lost on crash | Client can retry with last state |
| Implementation complexity | Session dict + cleanup | None (already works) |

**Why server-side for Phase 3**: We are running a single Modal instance for
prototype. No horizontal scaling needed yet. The latency savings (~1-2 ms
per call based on the D2H/H2D cost of 627 KiB) are worth the simplicity.

**Crash recovery**: If the Modal container restarts, the client must create a
new session. The first chunk will use cold-start state (avatar's neutral pose),
which is visually acceptable — the motion model converges within a few chunks.

### Option B: Client-carried state (simpler fallback)

Keep the current `/process-audio-v3` behavior exactly as-is. The bridge
passes the state blob between calls.

**When to use**: If Modal container recycling is frequent, or if we need
stateless horizontal scaling before investing in sticky sessions.

**Implementation**: Zero changes to `app.py`. Just deploy it as-is.

### Recommendation

Start with **Option A** (server-side state) for Phase 3. The implementation
is straightforward — replace the safetensors serialize/deserialize in
`run_chunk()` with a dict lookup. If it causes problems, fall back to Option B
with no changes to the renderer itself.

---

## First Render Test

The smallest possible test to validate Modal T4 can run the AVTR-1 pipeline.

### Prerequisites

1. Modal account with GPU access
2. AVTR-1 artifacts downloaded and accessible (Modal Volume or build-time copy)
3. TRT engines built for the target CUDA/TensorRT version (TBD — see open questions)

### Test 1: Pipeline startup

```python
import modal

app = modal.App("avtr1-smoke-test")

@app.function(
    image=modal.Image.from_registry(...),  # TBD: CUDA + deps
    gpu="T4",
    timeout=300,
)
def test_startup():
    from avtr1_renderer.pipeline import Pipeline

    pipeline, registry = Pipeline.from_artifacts(
        avatar_ids=["anya_03_studio"],
    )
    assert "anya_03_studio" in registry
    print(f"Loaded {len(registry)} avatars")
    print(f"Backgrounds: {sorted(pipeline._backgrounds.keys())}")
    return "OK"
```

**Success criteria**: No errors. Logs avatar count and backgrounds. Completes
within 120 seconds.

### Test 2: Single chunk render

```python
import numpy as np

@app.function(gpu="T4", timeout=60)
def test_single_chunk():
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, RenderOptions

    pipeline, registry = Pipeline.from_artifacts(avatar_ids=["anya_03_studio"])
    avatar = registry["anya_03_studio"]

    # Generate silence audio (405ms at 16kHz = 6480 samples)
    n_samples = 6480
    silence = np.zeros(n_samples, dtype=np.float32)
    chunk = Chunk(audio_speech=silence, audio_listen=silence)

    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id=sorted(pipeline._backgrounds.keys())[0],
    )

    import time
    t0 = time.perf_counter()
    state, frames = pipeline.process_chunk(avatar, chunk, state=None, options=options)
    elapsed = time.perf_counter() - t0

    frame_list = list(frames)
    print(f"Rendered {len(frame_list)} frames in {elapsed*1000:.0f}ms")
    print(f"State type: {type(state).__name__}")
    for i, f in enumerate(frame_list):
        print(f"  Frame {i}: shape={f.data.shape}, format={f.format}, "
              f"height={f.height}, width={f.width}")

    assert len(frame_list) == 5
    assert frame_list[0].height == 720
    assert frame_list[0].width == 1280
    assert frame_list[0].format == "yuv_i420"
    assert elapsed < 0.5  # 500ms budget (generous)

    return "OK"
```

**Success criteria**:
- 5 frames rendered
- Each frame is `(1080, 1280)` uint8 (yuv_i420 at 720p)
- Total render time < 500ms (the existing benchmark shows ~94ms for streaming
  mode, so 500ms is generous)
- No CUDA errors or OOM

### Test 3: Two-chunk state continuity

```python
@app.function(gpu="T4", timeout=60)
def test_state_continuity():
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, RenderOptions
    from avtr1_renderer.avtr1_motion_generator import state_to_safetensors, state_from_safetensors
    import numpy as np, time

    pipeline, registry = Pipeline.from_artifacts(avatar_ids=["anya_03_studio"])
    avatar = registry["anya_03_studio"]
    silence = np.zeros(6480, dtype=np.float32)
    chunk = Chunk(audio_speech=silence, audio_listen=silence)
    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id=sorted(pipeline._backgrounds.keys())[0],
    )

    # First chunk (cold start)
    t0 = time.perf_counter()
    state1, frames1 = pipeline.process_chunk(avatar, chunk, state=None, options=options)
    elapsed1 = time.perf_counter() - t0
    list(frames1)  # consume iterator

    # Second chunk (warm state)
    t0 = time.perf_counter()
    state2, frames2 = pipeline.process_chunk(avatar, chunk, state=state1, options=options)
    elapsed2 = time.perf_counter() - t0
    list(frames2)

    print(f"Chunk 1: {elapsed1*1000:.0f}ms")
    print(f"Chunk 2: {elapsed2*1000:.0f}ms")

    # Verify state serialization round-trip
    blob = state_to_safetensors(state2)
    state2_restored = state_from_safetensors(blob, device="cuda")
    assert state2_restored.past_cond.shape == state2.past_cond.shape

    return "OK"
```

**Success criteria**:
- Both chunks succeed
- Second chunk is no slower than first (state reuse should not add overhead)
- Safetensors round-trip preserves tensor shapes

### Test ordering

Run tests in order: Test 1 → Test 2 → Test 3. Each test depends on the
previous one passing. If Test 1 fails (pipeline won't start), the issue is
artifact/engine compatibility — do not proceed to Test 2.

---

## Major Risks and Go/No-Go Checks for Phase 3

### Go/No-Go criteria

| Check | Go | No-Go |
|-------|-----|-------|
| `Pipeline.from_artifacts()` completes on Modal T4 | Proceed | Investigate CUDA/TRT version mismatch |
| Single chunk renders in < 500ms | Proceed | Profile to find bottleneck |
| 5 consecutive chunks succeed with state passing | Proceed | Debug state continuity |
| VRAM usage measured < 12 GB (leaving 4 GB headroom on T4) | Proceed | Consider L4/A10G upgrade |

### Risk mitigation

| Risk | Mitigation |
|------|-----------|
| TRT engines don't load on Modal | Build engines inside Modal container using `scripts/build_avtr1_engines.py`. Pre-build as a Modal Image layer. |
| Cold start too slow | Use Modal warm pools to keep container alive. Accept ~60s first-request latency. |
| T4 VRAM insufficient | Fall back to ONNX for non-AVTR1 engines (decoder, warp, stitch, modnet). Only AVTR1 encode/decode require TRT. |
| Latency too high | Enable streaming mode (already default). Measure and optimize per-component. |

### Hard blockers (cannot proceed without resolving)

1. **TRT engine compatibility**: If AVTR1 encode/decode engines cannot load on
   Modal's CUDA version, we cannot render at all. There is no ONNX fallback.
2. **Artifact access**: If the Modal container cannot access HuggingFace
   artifacts (network or auth issues), pipeline initialization fails.

### Soft blockers (can work around)

1. **Cold start time**: Can be hidden behind warm pools.
2. **VRAM pressure**: Can reduce avatar count, use smaller backgrounds, or
   fall back to ONNX for non-critical engines.
3. **Latency**: Can adjust chunk size or frame rate if needed (but this
   changes the model's expected input contract).
