# AVTR-1 Avatar Runtime Analysis

Phase 2 analysis of the existing AVTR-1 renderer. Documents the exact API
contract, stateful behavior, and runtime requirements that a Modal wrapper
must implement.

All details are sourced from reading the codebase. Anything not directly
observed in source is marked **unverified** or **TBD**.

---

## Source Files Inspected

| File | Key findings |
|------|-------------|
| `scripts/run_local_stream.py` | Orchestrator: spawns renderer + streamer subprocesses |
| `src/avtr1_renderer/api/app.py` | FastAPI server with `/process-audio-v3` |
| `src/avtr1_renderer/pipeline.py` | `Pipeline.from_artifacts()` and `process_chunk()` |
| `src/avtr1_renderer/renderer.py` | `render_chunk_streaming()` per-frame GPU pipeline |
| `src/avtr1_renderer/types.py` | `Chunk`, `Frame`, `RenderOptions`, `KPInfo` dataclasses |
| `src/avtr1_renderer/avtr1_motion_generator.py` | `AVTR1MotionGenerator`, `AVTR1State`, state serialization |
| `src/avtr1_renderer/motion_generator.py` | `MotionGenerator` protocol definition |
| `src/avtr1_renderer/avatar_loader.py` | `Avatar`, `AvatarLoader` registration cascade |
| `src/avtr1_renderer/backgrounds.py` | Background image loading to CUDA tensor |
| `src/avtr1_renderer/frame_sink.py` | `pack_frames()` GPU-to-host conversion |
| `src/avtr1_renderer/components/pixel_format.py` | `PixelFormat`, bytes-per-frame calculation |
| `src/avtr1_renderer/constants.py` | `LIPSYNC_COORDS` (39 indices into 63-dim expression) |
| `src/avtr1_renderer/api/load_balancing.py` | Load balancer keep-alive (disabled in local mode) |
| `src/avtr1_renderer/avtr1_artifact_manager.py` | Artifact storage root and download logic |
| `src/avtr1_renderer/artifact_configs/v1.py` | HF repo ID, artifact layout, TRT engine paths |
| `src/avtr1_renderer/artifact_manage_basic.py` | `ArtifactManager` base class |
| `src/avtr1_renderer/runtime/loader.py` | Engine loader (TRT vs ONNX by file suffix) |
| `src/avtr1_renderer/intro_motion.py` | Intro animation library (optional, not used in SaaS path) |
| `src/avaturn_live_streamer/renderer/client.py` | HTTP client that calls `/process-audio-v3` |
| `src/avaturn_live_streamer/renderer/interface.py` | `RendererRequest`, `RenderConfig`, `RenderResponse` types |
| `src/avaturn_live_streamer/renderer/models.py` | Model durations, default CFG params |
| `src/avaturn_live_streamer/worklets/rendering.py` | `RenderingWorklet` main loop |
| `src/avaturn_live_streamer/speech/speech_scheduler.py` | Present/future audio chunking |
| `src/avaturn_live_streamer/speech/speech_buffer.py` | `SpeechBuffer` audio format handling |
| `src/avaturn_live_streamer/constant.py` | Sample rates, video resolution, FPS |
| `src/avaturn_live_streamer/types.py` | `PixelFormat` enum |
| `pixi.toml` | Dependencies, environments, CUDA version |

---

## Local Demo Entry Point

`scripts/run_local_stream.py` is the top-level orchestrator (observed in source).

### Startup sequence

1. Spawn the **renderer** subprocess:
   ```
   pixi run -e default python -m uvicorn avtr1_renderer.api.app:app --host 0.0.0.0 --port 8000
   ```
   - Sets `LOAD_BALANCER_URL=disabled` in env (no LB in local mode).
   - Renderer reads `AVTR1_LOCAL_STORAGE` for artifact paths.

2. Poll `GET http://localhost:8000/health` every 1s until HTTP 200 (timeout: 300s).

3. Spawn the **streamer** subprocess:
   ```
   python -m avaturn_live_streamer.local_stream_cli --host 127.0.0.1 --port 7860
   ```
   - Sets env `RENDERERS__RENDERERS__AVTRN_1__MODE=single` and
    `RENDERERS__RENDERERS__AVTRN_1__LB_OR_INSTANCE_URL=http://localhost:8000`
    so the streamer connects to the renderer directly (no load balancer).

4. Both subprocesses run until SIGINT/SIGTERM or one exits.

### What this tells us about Modal

The renderer is already a standalone HTTP service. The streamer connects to it
over plain HTTP. Modal can run the same FastAPI app with zero code changes,
exposing `/process-audio-v3` to the Cloudflare bridge.

---

## Renderer FastAPI Endpoints

All three endpoints are defined in `src/avtr1_renderer/api/app.py`.

### `GET /health`

Returns `{"status": "ok"}` with HTTP 200 after the CUDA health check passes.
Returns 503 with `{"detail": "starting"}` during startup or if CUDA is unhealthy.

Source: `app.py:274-284`.

### `GET /avatars`

Returns loaded avatar IDs and background IDs:
```json
{
  "avatars": ["anya_03_studio", ...],
  "backgrounds": ["bg_01", ...]
}
```
Backgrounds exclude the reserved `"transparent"` sentinel.

Source: `app.py:259-271`.

### `POST /process-audio-v3`

The primary rendering endpoint. Accepts audio + state, returns state + frames.

Source: `app.py:160-256`.

---

## `/process-audio-v3` Request Contract

From `src/avtr1_renderer/api/app.py:160-176`.

### Content type

`multipart/form-data` (FastAPI `UploadFile` fields + query/form params).

### Audio fields (all required, `UploadFile`)

| Field | Description | Byte count |
|-------|-------------|------------|
| `current_chunk` | Present speech audio | `3200 * 2 = 6400` bytes |
| `future_chunk` | Future speech audio | `3280 * 2 = 6560` bytes |
| `current_chunk_listen` | Present listen audio | `3200 * 2 = 6400` bytes |
| `future_chunk_listen` | Future listen audio | `3280 * 2 = 6560` bytes |

Audio format: **raw int16 little-endian PCM, 16 kHz, mono** (observed in
`app.py:13-15`, `app.py:50-63`).

The server converts int16 to float32 `[-1, 1]` via `np.frombuffer(..., dtype=np.int16).astype(np.float32) / 32768.0`.

### Byte-size validation

Exact. The server validates each audio field's byte count and rejects with
HTTP 400 if wrong (observed in `app.py:53-63`):

```python
expected_bytes = expected_samples * 2
if len(buf) != expected_bytes:
    raise HTTPException(status_code=400, detail=f"{field}: expected exactly {expected_bytes} bytes ...")
```

Where sample counts come from `app.py:150-151`:

```python
app.state.current_samples = mg.chunk_size * mg.frame_len       # 5 * 640 = 3200
app.state.future_samples = mg.future_size * mg.frame_len + mg.audio_shift  # 5 * 640 + 80 = 3280
```

### State field (optional, `UploadFile`)

| Field | Required | Description |
|-------|----------|-------------|
| `state` | No | Safetensors blob from previous response. Omit or send empty for cold start. |

If the state blob is empty bytes, it is treated as `None` (cold start).
Source: `app.py:167`, `app.py:197-199`.

### Config fields (query/form parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `avatar_id` | `str` | `"anya_03_studio"` | Must match a loaded avatar ID |
| `bg_id` | `str` | **(required)** | Background ID, must match loaded bg or `"transparent"` |
| `pixel_format` | `str` | `"yuv_i420"` | `"yuv_i420"` or `"yuv_i420_stacked_alpha"` |
| `cfg_self_audio` | `float` | `2.0` | CFG weight for self-audio condition |
| `cfg_other_audio` | `float` | `2.0` | CFG weight for other-audio condition |
| `cfg_kp` | `float` | `4.0` | CFG weight for keypoint condition |
| `noise_alpha` | `float` | `2.0` | AR(1) progressive noise alpha |
| `noise_trunc_z` | `float` | `1.2` | Truncated normal z-value for noise |

**Note on `cfg_kp` default**: `app.py:176` defaults `cfg_kp` to `4.0`. The
streamer's `models.py:43` uses `3.0` as the default. The streamer overrides
it via `extra_params`, so the app.py default is only used when calling the
renderer directly without the streamer. This is an observation, not a
protocol issue.

### Observation: extra multipart fields from streamer

The streamer client (`renderer/client.py:452-459`) sends two additional empty
multipart fields that the FastAPI endpoint does not declare:

```python
"past_chunk": b"",
"past_chunk_listen": b"",
```

These are **not required inputs**. The FastAPI endpoint signature does not
declare them. They appear to be a backward-compatibility hotfix in the
streamer client. FastAPI silently ignores extra multipart fields. This is
documented as an observation only.

---

## Audio Timing and Chunk Contract

All values observed in source.

### Constants (from `AVTR1MotionGenerator.__init__` defaults, `constant.py`)

| Constant | Value | Source |
|----------|-------|--------|
| `chunk_size` | 5 frames | `avtr1_motion_generator.py:294` |
| `past_size` | 75 frames | `avtr1_motion_generator.py:295` |
| `future_size` | 5 frames | `avtr1_motion_generator.py:296` |
| `frame_len` | 640 samples | `avtr1_motion_generator.py:299` |
| `audio_shift` | 80 samples | `avtr1_motion_generator.py:300` |
| `n_ode_steps` | 5 | `avtr1_motion_generator.py:301` |
| `VIDEO_FPS` | 25 fps | `constant.py:12` |
| `RENDERER_SPEECH_SAMPLE_RATE` | 16000 Hz | `constant.py:6` |

### Sample counts per audio field

```
current_samples = chunk_size * frame_len = 5 * 640 = 3200  (200 ms)
future_samples  = future_size * frame_len + audio_shift = 5 * 640 + 80 = 3280  (205 ms)
```

Each `current_*` field: **3200 samples = 6400 bytes** (int16).
Each `future_*` field: **3280 samples = 6560 bytes** (int16).

The pipeline validates total audio length per track (speech and listen combined
as `np.concatenate([current, future])`) against:

```
(chunk_size + future_size) * frame_len + audio_shift = 6480 samples (405 ms)
```

Source: `pipeline.py:275-283`.

### Speech Scheduler timing (from streamer)

The streamer's `SpeechScheduler` uses durations derived from `constant.py`:

```
present_duration = FRAME_DURATION * 5 = (1/25) * 5 = 200 ms
future_duration  = FRAME_DURATION * 5 + 80/16000 = 200 ms + 5 ms = 205 ms
```

Source: `renderer/models.py:28-33`, `constant.py:12-14`.

---

## `/process-audio-v3` Response Contract

From `app.py:236-256`.

### Headers

| Header | Value | Description |
|--------|-------|-------------|
| `X-Num-Frames` | `"5"` | Frames in this response |
| `X-Frame-Height` | e.g. `"720"` | Frame height in pixels |
| `X-Frame-Width` | e.g. `"1280"` | Frame width in pixels |
| `X-Frame-Length-Bytes` | e.g. `"1382400"` | Bytes per single frame |
| `X-Has-State` | `"yes"` | State blob is present |
| `X-State-Format` | `"safetensors"` | State serialization format |
| `X-State-Length-Bytes` | e.g. `"627000"` (variable) | State blob size in bytes |

### Body layout

```
[state_blob: X-State-Length-Bytes] [frame_0] [frame_1] ... [frame_4]
```

Streaming response (`StreamingResponse`). The state blob arrives first, then
frames are streamed one at a time. The client can start processing frames
before all frames have arrived.

Source: `app.py:236-244`.

### Frame dimensions and sizes

Default output resolution: **720 x 1280** (H x W), observed in
`pipeline.py:100` and `constant.py:11`.

Frame byte sizes depend on pixel format (`pixel_format.py:49-55`):

| Pixel format | Layout | Bytes per frame |
|-------------|--------|----------------|
| `yuv_i420` | `(1.5*H, W)` = `(1080, 1280)` uint8 | `H*W*3//2 = 1,382,400` (~1.32 MiB) |
| `yuv_i420_stacked_alpha` | `(3*H, W)` = `(2160, 1280)` uint8 | `H*W*3 = 2,764,800` (~2.64 MiB) |

5 frames per chunk. Total response body (yuv_i420):

```
~627 KiB state + 5 * 1.32 MiB frames = ~7.2 MiB
```

### Pixel format details

- **`yuv_i420`** (default): BT.601 limited-range YUV with 4:2:0 chroma
  subsampling. The renderer composites the rendered head over the background
  using MODNet alpha matte. No separate alpha plane is shipped.

- **`yuv_i420_stacked_alpha`**: I420 plus a second I420 carrying alpha as
  luma (constant 128 chroma). Layout: `(Y, AY, U, AU, V, AV)` packed as
  `(3*H, W)`. The alpha channel survives through H.264/VP9/AV1 pipelines.

Source: `pixel_format.py:10-30`, `pixel_format.py:49-55`.

---

## Stateful Behavior

### Current state model: client-carried

The renderer is **stateless** between HTTP calls. State is serialized to a
safetensors blob in the response, and the client must return it in the next
request.

Source: `app.py:88-121` (run_chunk), `app.py:197-199` (state reading),
`app.py:229` (state_blob = await state_fut), `app.py:238` (yield state_blob).

### State structure: `AVTR1State`

From `avtr1_motion_generator.py:127-151`.

```python
@dataclass(slots=True, frozen=True)
class AVTR1State:
    audio_prev_speech: Tensor   # (3*640,)  = (1920,) float32 CUDA
    audio_prev_listen: Tensor   # (1920,) float32 CUDA
    audio_features: Tensor      # (1, 75, 2048) float32 CUDA
    past_cond: Tensor           # (1, 75, 42) float32 CUDA
    noise_shared: Tensor | None # (1, 1, 42) float32 CUDA or None
```

### State lifecycle

1. **Cold start**: `state=None` or empty blob in request. The pipeline calls
   `motion_generator.initial_state(avatar)` which creates:
   - Zero-initialized audio buffers
   - Zero-initialized audio_features
   - `past_cond` tiled from the avatar's neutral pose (source-specific)
   - `noise_shared = None`
   Source: `avtr1_motion_generator.py:340-366`.

2. **Per call**: The server deserializes the safetensors blob to CUDA tensors
   via `state_from_safetensors(blob, device="cuda")`. After processing, the
   new state is serialized back via `state_to_safetensors(next_state)`.
   Source: `avtr1_motion_generator.py:88-124`, `app.py:98-107`, `app.py:110`.

3. **Approximate state size**: ~627 KiB (computed from tensor shapes).
   This is a D2H serialization + H2D deserialization overhead on every call.

### Per-chunk state transitions

Each call to `generate_chunk()` (observed in `avtr1_motion_generator.py:371-520`):

1. HuBERT processes audio (3-frame left context + 10 new frames)
2. `audio_features` shifts: drop oldest 5 frames, append 5 new current features
3. `past_cond` shifts: drop oldest 5, append 5 new normalized motion predictions
4. `audio_prev_*` rolls forward: extracts the 3-frame tail for next call
5. `noise_shared` carries the last frame's noise for AR(1) correlation

### Modal implication

The current client-carried state design forces a D2H/H2D round-trip (~627 KiB)
on every call. A Modal wrapper that keeps state on GPU between calls eliminates
this overhead. This is the single biggest optimization opportunity.

---

## Rendering Pipeline Internals

### Startup: `Pipeline.from_artifacts()`

`pipeline.py:93-253`. Steps:

1. Download missing artifacts from HuggingFace (`ArtifactManager.ensure_all_artifacts()`)
2. Load backgrounds from PNG files to CUDA tensors
3. Check for AVTR1 TRT engines (encode, decode) — **required, no ONNX fallback**
4. Load HuBERT engine (TRT if built, else ONNX)
5. Load renderer engines: decoder, warp, stitch, modnet (TRT if built, else ONNX)
6. Build `AvatarLoader` with 5 ONNX engines (face detection, landmarks, etc.)
7. Register all avatars from `reference_frames/*.png` (registration cascade per portrait)
8. Build `AVTR1MotionGenerator` with encode/decode/hubert engines + normalizer

### Per-call: `Pipeline.process_chunk()`

`pipeline.py:258-319`. Steps:

1. Validate chunk audio length
2. If `state=None`, create initial state from avatar's neutral pose
3. `motion_generator.generate_chunk()`: HuBERT + AVTR1 encode + ODE decode → `MotionFrame`
4. `render_chunk_streaming()`: stitch → warp → per-frame (decode → putback → matting → pack)
5. Return `(next_state, frame_iterator)`

### Motion generation: `AVTR1MotionGenerator.generate_chunk()`

`avtr1_motion_generator.py:371-520`. Per-chunk:

1. Concat 3-frame audio tail with new chunk audio, stack speech+listen as batch
2. Run HuBERT once (batch=2: speech + listen) → `(2, 10, 1024)` features
3. Concat features → `(1, 10, 2048)`, prepend 75-frame history → `(1, 85, 2048)`
4. Build `kp_cond` from avatar's normalized pose
5. AVTR1 encode once → 5 condition tensors
6. Sample initial noise (progressive AR(1) from previous chunk's `noise_shared`)
7. Euler ODE integration: 4 decode steps (n_ode_steps - 1) with CFG
8. De-normalize predictions into `MotionFrame(R, exp)`
9. Update state tensors

### Frame rendering: `render_chunk_streaming()`

`renderer.py:41-72`. Per-chunk:

1. `motion_stitch()`: convert motion to source/driving keypoints
2. `warp()` once on full chunk (batch=5) — shares work across frames
3. Per frame: `decoder()` → `putback()` → `matting()` → `pack_frames()`
4. Each frame is materialized on CPU before yielding (streaming)

### Frame packing: `pack_frames()`

`frame_sink.py:29-68`. Converts GPU float tensors to host uint8:

1. `assemble_frame_as_uint8()`: RGB(A) → YUV conversion on CUDA
2. Single batched `.cpu().numpy()` H2D copy
3. Return `list[Frame]` with numpy data

---

## Runtime Requirements

### GPU

- **CUDA 12.8** required (`pixi.toml:8`: `cuda = "12.8"`)
- **TensorRT** required for AVTR1 encode + decode engines (no ONNX fallback)
  - Source: `pipeline.py:146-155` — raises `RuntimeError` if engines missing
- All other engines (decoder, warp, stitch, modnet, hubert) fall back to ONNX
- **T4 (16 GB VRAM)**: feasible for prototype, requires measurement

### Python dependencies (from `pixi.toml:36-58`)

| Dependency | Version | Purpose |
|-----------|---------|---------|
| `torch` | `>=2.5.1,<2.8` (cu128) | Core tensor ops |
| `tensorrt` | `<=10.12` | AVTR1 TRT engines |
| `onnxruntime-gpu` | `>=1.20` | ONNX engine fallback |
| `kornia` | `>=0.8.1,<0.9` | Image warping, color conversion |
| `roma` | `>=1.5.4,<2` | Rotation matrix math |
| `safetensors` | `>=0.5,<1` | State serialization |
| `opencv-python-headless` | `>=4.11` | Image loading |
| `numpy` | `==2.0` | Array ops |
| `fastapi` | `>=0.116` | HTTP server |
| `uvicorn` | `>=0.35` | ASGI server |
| `huggingface_hub` | `>=0.30` | Artifact download |

### Artifact storage

- **HuggingFace repos**:
  - `avaturn-live/avtr-1` (main) — primary artifacts
  - `digital-avatar/ditto-talkinghead` (main) — shared LP/renderer ONNX models
- **Storage root**: `$AVTR1_LOCAL_STORAGE/{HF_REVISION}/` or `<project_root>/artifacts/{HF_REVISION}/`
- **TRT engines**: built locally by `scripts/build_*.py`, stored under `*_cc/` subdirectories
- **Key TRT engines** (from `artifact_configs/v1.py:114-122`):

| Engine name | Path |
|-------------|------|
| `avtr1_encode` | `speech2motion_runtime_artifacts_cc/avtr1_encode_fp16.engine` |
| `avtr1_decode` | `speech2motion_runtime_artifacts_cc/avtr1_decode_fp16.engine` |
| `hubert_lbs` | `speech2motion_runtime_artifacts_cc/hubert_lbs_fp16.engine` |
| `decoder` | `renderer_runtime_artifacts_cc/decoder_b5_fp16.engine` |
| `warp_network` | `renderer_runtime_artifacts_cc/warp_network_b5_fp16.engine` |
| `stitch_network` | `renderer_runtime_artifacts_cc/stitch_network_b5_fp16.engine` |
| `modnet` | `renderer_runtime_artifacts_cc/modnet_b5_fp16.engine` |

### Cold start

- `Pipeline.from_artifacts()` must complete before serving requests
- Steps: artifact download (if missing) → TRT engine deserialization → avatar registration
- Avatar registration runs face detection + landmarks + crop + appearance extraction + motion extraction per portrait
- **Cold start time**: unknown, needs measurement. Estimate: 30-60+ seconds.
- The `/health` endpoint only returns 200 after CUDA is initialized, so the load
  balancer (or orchestrator) can wait.

### VRAM footprint

**Unknown. Needs measurement.** Components loaded on GPU:

- 7 inference engines (TRT or ONNX) — each has its own weights + allocation buffers
- Avatar data: `f_s` `(1,32,16,64,64)` per avatar + KPInfo tensors + masks
- Background images: `(1,3,720,1280)` float32 each
- Per-session state: ~1.8 MiB of CUDA tensors (see state structure above)
- Working buffers: decoder outputs `(1,3,512,512)`, warped features, etc.

---

## How the Streamer Drives the Renderer

From `src/avaturn_live_streamer/worklets/rendering.py`.

### Rendering loop

1. `RenderingWorklet.run()` creates two `SpeechScheduler` instances:
   - One for avatar speech (TTS output) at 24 kHz
   - One for user speech (microphone) at 16 kHz

2. The main loop calls `speech_scheduler.do_step()` each iteration, which
   produces a `_StepResult` with `present` and `future` `SpeechBuffer`s.

3. Audio is resampled from 24 kHz to 16 kHz via `SpeechBuffer.resample(16000)`.

4. A `RendererRequest` is built with:
   - `current_chunk` = present avatar speech at 16 kHz (int16 bytes)
   - `future_chunk` = future avatar speech at 16 kHz (int16 bytes)
   - `current_chunk_listen` = present user speech at 16 kHz (int16 bytes)
   - `future_chunk_listen` = future user speech at 16 kHz (int16 bytes)
   - `state` = safetensors blob from previous response (or None)
   - `config` = avatar_id, bg_id, pixel_format, etc.

5. The renderer client POSTs to `/process-audio-v3`, streams the response,
   parses state + frames, and publishes `VideoFrameGenerated` events.

### Streamer client details

From `renderer/client.py`:

- Uses **sync httpx** on a worker thread for receiving (to avoid asyncio
  scheduling overhead — documented as a deliberate optimization)
- Reads response headers, pre-allocates a `bytearray(total)` buffer
- Streams raw bytes, splitting at state boundary and frame boundaries
- State is published only on full success (never from a failed attempt)
- Supports retry with `AsyncRetrying` on HTTP errors
- Supports load balancer mode (`acquire_worker`/`release_worker`) or direct
  instance mode (`mode=single`)

---

## Risks

| Risk | Severity | Notes |
|------|----------|-------|
| TRT engine compatibility | **High** | AVTR1 encode/decode require TRT. Engines built on one CUDA/TRT version may not load on another. Must verify Modal's CUDA version matches build environment. |
| Cold start time | **High** | Unknown. Modal container cold start + engine deserialization + avatar registration. Could be 30-60+ seconds. |
| VRAM on T4 | **Medium** | 7 engines + avatar data + backgrounds + state. 16 GB T4 may be tight. Needs measurement. |
| State serialization overhead | **Low** | ~627 KiB D2H + H2D per call. Server-side state ownership on Modal eliminates this. |
| Bandwidth | **Low** | ~7.2 MiB response per call at ~5 calls/sec = ~36 MiB/sec. Manageable. |
| Latency budget | **Low** | Existing measurement: ~94 ms for 5 frames. Well within the 200 ms present-duration budget. Source: `client.py:129-137` benchmark table. |

---

## Open Questions

1. **CUDA/TensorRT version on Modal T4**. Must the TRT engines be rebuilt on
   Modal's container, or can we ship pre-built engines? The engines depend on
   the TRT version used at build time.

2. **Exact cold start time** on Modal T4 with all engines and avatar registration.
   Must be measured.

3. **VRAM footprint** with all 7 engines loaded + avatar data + backgrounds.
   Must be measured. If T4 is too tight, L4 (24 GB) or A10G (24 GB) are
   candidate upgrades (noted in ROADMAP.md Phase 0).

4. **Whether pre-built TRT engines work on Modal T4** or must be rebuilt inside
   the Modal container. This depends on CUDA compute capability and TRT version
   compatibility.

5. **Best browser frame transport**. YUV I420 raw frames at ~1.32 MiB each are
   not browser-friendly. Options include VP9/AV1 encoding, WebCodecs, or
   WebRTC. This is a TBD protocol decision for Phase 5.

6. **Multiple concurrent sessions on one GPU**. Can a single Modal container
   serve multiple avatar sessions simultaneously? Depends on VRAM headroom
   and engine concurrency limits. TBD.
