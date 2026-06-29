# Stage 4A — SFU Media Pipeline Performance & Architecture Audit

**Date:** 2026-06-19  
**Commit:** `25ee7f4`  
**Production state:** `AVTR_REALTIME_SFU_ENABLED=false`, `AVTR_REALTIME_SFU_DEFAULT=websocket_fmp4`  
**Verdict:** SFU path **blocked by CPU encoding**. Acceptable for canary only. Recommend Option C (GPU-backed WebRTC publisher) as strategic target.

---

## 1. Pipeline Diagrams

### 1a. fMP4 path (production — `websocket_fmp4`)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Modal L40S container — single process (@modal.concurrent max_inputs=2) │
│                                                                         │
│  ┌─── GPU (CUDA / TensorRT) ───────────────────────────────────────┐   │
│  │ pipeline.process_chunk()                                          │   │
│  │   src/avtr1_renderer/pipeline.py:421-482                          │   │
│  │   renderer.py:41-72  (stitch/warp/SPADE/MODNet, device="cuda")    │   │
│  │   pixel_format.py:92-150  (kornia rgb_to_yuv420 on CUDA, BT.601)  │   │
│  │   frame_sink.py:64  packed.cpu().numpy()  ◀── H2D copy            │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                          CPU numpy I420 (1080×1280 uint8)               │
│  ┌──────────────────────────────▼──────────────────────────────────┐   │
│  │ CPU threads                                                       │   │
│  │ video_write_queue (Queue maxsize=50)  5q.py:2594                  │   │
│  │   → _video_writer thread  5q.py:2682-2699                        │   │
│  │   → ffmpeg.stdin.write() + flush()                                │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                          raw YUV420 pipe (stdin)                        │
│  ┌──────────────────────────────▼──────────────────────────────────┐   │
│  │ GPU (NVENC) — separate PROCESS                                    │   │
│  │ ffmpeg -c:v h264_nvenc -preset p4 -tune ull -g 25 -bf 0          │   │
│  │   -profile:v baseline -level 3.1                                  │   │
│  │   -f mp4 -movflags +frag_keyframe+empty_moov+default_base_moof   │   │
│  │   -frag_duration 200000  pipe:1                                   │   │
│  │   5q.py:2562-2581  (identical for adapter + session-render)       │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
│                          encoded fMP4 bytes (stdout)                    │
│  ┌──────────────────────────────▼──────────────────────────────────┐   │
│  │ CPU threads                                                       │   │
│  │ _ffmpeg_stdout_reader (box parser)  5q.py:2613-2658              │   │
│  │   struct.unpack(">I") → moov/moof/mdat                           │   │
│  │ segment_queue (maxsize=100) → _SegmentSender pacer                │   │
│  │   avtr_segment_sender.py:17-274                                   │   │
│  │ base64 encode → SSE event stream                                  │   │
│  └──────────────────────────────┬──────────────────────────────────┘   │
└──────────────────────────────────┼──────────────────────────────────────┘
                          SSE HTTP (text/event-stream)
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Cloudflare Worker + DO                                                  │
│ avatar-session-do.ts:2778  startModalTimelineFmp4AvStream()             │
│   onInit → ws.send({type:"avatar.fmp4.init", dataBase64})  :2790-2795   │
│   onMedia → ws.send({type:"avatar.fmp4.segment", dataBase64}) :2822-2836│
└──────────────────────────────────┬──────────────────────────────────────┘
                          WebSocket JSON
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Browser — MSE SourceBuffer.appendBuffer()                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1b. SFU / WebRTC path (canary only)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Modal L40S container — single process                                  │
│                                                                         │
│  ┌─── GPU (CUDA / TensorRT) — identical to fMP4 ────────────────────┐  │
│  │ pipeline.process_chunk()                                          │  │
│  │   webrtc_publisher.py:177-180                                     │  │
│  │   (H2D copy happens inside the renderer before yielding)          │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                          CPU numpy I420 (1080×1280 uint8)               │
│  ┌──────────────────────────────▼───────────────────────────────────┐  │
│  │ CPU producer THREAD  "avtr-frame-producer"                        │  │
│  │ webrtc_publisher.py:140-217  (_producer_loop)                     │  │
│  │   frame.data.copy()  :194  (isolate buffer before crossing)       │  │
│  │   asyncio.run_coroutine_threadsafe(queue.put)  :190-206           │  │
│  │   asyncio.Queue(maxsize=10)  :123  (≈0.4s buffer at 25fps)        │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                     crosses thread boundary (GIL bridge)                │
│  ┌──────────────────────────────▼───────────────────────────────────┐  │
│  │ CPU event loop  (dedicated _PublisherLoop thread)                 │  │
│  │ _publisher_loop.py:30-75                                          │  │
│  │                                                                   │  │
│  │ PacedVideoTrack.recv()  webrtc_publisher.py:342-397               │  │
│  │   get_frame() → np.ascontiguousarray                              │  │
│  │   VideoFrame.from_ndarray("yuv420p")  :361                        │  │
│  │   pts/time_base stamping (25fps, TICKS=3600)                      │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                     VideoFrame (av.VideoFrame)                          │
│  ┌──────────────────────────────▼───────────────────────────────────┐  │
│  │ ★ CPU SOFTWARE ENCODE — PyAV / FFmpeg (NO NVENC) ★                │  │
│  │                                                                   │  │
│  │ aiortc default: VP8 preferred (libvpx), H264 available (libx264) │  │
│  │ No setCodecPreferences() set anywhere (verified by grep)          │  │
│  │ PyAV manylinux wheel: --only-binary av, no --enable-nvenc        │  │
│  │ Encoder runs in-process (no subprocess)                           │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                     RTP / SRTP packets                                  │
│  ┌──────────────────────────────▼───────────────────────────────────┐  │
│  │ Network callbacks (Modal → Worker → SFU)                          │  │
│  │ POST publisher-offer  webrtc_publisher.py:503-513                 │  │
│  │   → Worker index.ts:414-454 (x-modal-secret auth)                 │  │
│  │   → DO avatar-session-do.ts:6000-6083 (tracks/new)                │  │
│  │   → SFU sfu-client.ts:99-132 (rtc.live.cloudflare.com)            │  │
│  │ PUT publisher-close   webrtc_publisher.py:552-566                 │  │
│  │   → DO avatar-session-do.ts:6085-6116 (tracks/close)              │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
└──────────────────────────────────┼──────────────────────────────────────┘
                     WebRTC (DTLS/SRTP)
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Cloudflare Realtime SFU — rtc.live.cloudflare.com                       │
└──────────────────────────────────┬──────────────────────────────────────┘
                     WebRTC
┌──────────────────────────────────▼──────────────────────────────────────┐
│ Browser — WebRTC subscriber (RTCPeerConnection, hardware decode)         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CPU / GPU Responsibility Table

| Stage | fMP4 path | SFU path | Δ Cost |
|-------|-----------|----------|-------|
| AVTR inference (Hubert + decode + stitch + warp + SPADE + MODNet) | **GPU** (TensorRT) | **GPU** (identical) | 0 |
| RGB→YUV420 BT.601 conversion | **GPU** (kornia on CUDA) | **GPU** (identical) | 0 |
| GPU→CPU tensor copy | `.cpu().numpy()` `frame_sink.py:64` | `.cpu().numpy()` (inside renderer) | 0 |
| **Video encoding** | **GPU (h264_nvenc)** subprocess `5q.py:2562-2581` | **CPU (libvpx/libx264)** in-process via PyAV | **★★★ MAJOR — see §3** |
| Frame queue / backpressure | `Queue(maxsize=50)` thread→thread | `asyncio.Queue(maxsize=10)` thread→loop | SFU tighter (0.4s vs 2.0s buffer) |
| Packetization / framing | fMP4 box parse + base64 + SSE | RTP/SRTP via aiortc | Different overhead profiles |
| Worker relay | SSE fetch → WS JSON `avatar-session-do.ts:2778` | SDP broker → SFU API → WS subscribe | SFU adds 1 broker hop, avoids base64 |
| Browser decode | MSE (hardware-accelerated) | WebRTC (hardware-accelerated) | ~equal |

---

## 3. Critical Finding: SFU Path Is CPU-Encoded

### Verdict: CPU software encoding via PyAV. NO hardware encoding in the SFU path. This is the dominant production architecture concern.

### Evidence (all verified from code, not speculated)

1. **No codec preference configured.** Grep across `webrtc_publisher.py`, `_publisher_loop.py`, and `5q.py` for `setCodecPreference`, `RTCRtpSender.getCapabilities`, `forceH264`, `video_codecs` → **zero hits**. The PC is created with only `RTCConfiguration(iceServers=[...])` (`webrtc_publisher.py:457-459`) and the track is added via plain `addTrack()` (`:461`). No transceiver codec manipulation.

2. **aiortc defaults to VP8.** Per aiortc documentation (Context7 `/aiortc/aiortc`), `RTCRtpSender.getCapabilities("video")` returns `video/VP8 pt=96` (preferred) and `video/H264 pt=107`. To force H264, one must call `setCodecPreferences([h264_codecs])` — which the code does not do.

3. **PyAV wheel is CPU-only.** Modal image pins `pip install 'av>=14,<15' --only-binary av` (`5q.py:132-137`). The manylinux wheel bundles FFmpeg built **without** `--enable-nvenc`. Confirmed by CPU encode probe: `h264_nvenc` codec is not usable for actual encoding.

4. **No FFmpeg subprocess in WebRTC path.** Grep for `subprocess|Popen|ffmpeg|nvenc` in `webrtc_publisher.py` → zero hits. Encoding happens in-process via PyAV's Codec API.

5. **Every `h264_nvenc` reference is in the fMP4 path.** All occurrences (`5q.py:2572, 3417, 3632, 4020, 4644, 5329`) are inside FFmpeg subprocess argv for fMP4 streaming endpoints. The WebRTC publisher never spawns FFmpeg.

6. **CPU encode probe confirms:** VP8 encoding at 720p/25fps consumes 420% CPU (4.2 multi-threaded cores). H264 consumes 497% CPU (5.0 cores). See §4 for details.

---

## 4. Timing Tables

### 4a. CPU Encode Probe Results (isolated, no GPU inference)

Probe: `modal_cpu_encode_probe.py` — standalone Modal app (CPU-only, no GPU), PyAV Codec API direct encoding, shifting-gradient I420 frames at 1280×720@25fps, 20-second measurement window.

| Metric | VP8 (libvpx) | H264 (libx264) |
|--------|-------------|----------------|
| Encode throughput | **151.5 fps** | **102.2 fps** |
| CPU% mean | **420.2%** | **497.4%** |
| CPU% max | 547.8% | 634.7% |
| CPU cores used | 4.2 | 5.0 |
| Bytes / frame | 3,928 | 743 |
| Duration | 20.0s | 20.1s |
| Frames encoded | 3,031 | 2,056 |

**Per-frame CPU cost extrapolated to production rate (25fps):**

| Codec | CPU-time / frame | At 25fps | Wall-time / frame | Event-loop blocking at 25fps |
|-------|-----------------|----------|-------------------|------------------------------|
| VP8 | 27.7ms | **69.2% of 1 core** | 6.6ms | **16.5%** |
| H264 | 48.6ms | **121.5% of 1 core** | 9.8ms | **24.5%** |

> VP8 is faster than H264 because libvpx's default settings use realtime deadline. libx264 defaults to `preset=medium` (quality-focused). Both are multi-threaded internally.

### 4b. SFU Publisher Timing (production pipeline, `AVTR_MEDIA_TIMING_DEBUG=1`)

Session: `session:433d94f3-057e-4e7c-8190-a82847c48a98`, 90s SFU session, VP8 default codec.

| Metric | Count | p50 | p95 | Max | Mean |
|--------|-------|-----|-----|-----|------|
| **process_chunk_ms** (GPU inference, 5 frames/chunk) | 256 | **69.1** | 70.2 | 71.5 | 69.1 |
| frames_per_chunk | 256 | 5 | 5 | 5 | 5.0 |
| **gpu_to_cpu_copy_ms** (frame.data.copy) | 256 | 0.2 | 0.2 | 0.3 | 0.2 |
| **queue_put_ms** (run_coroutine_threadsafe) | 256 | **39.0** | 40.6 | 208.7 | 26.1 |
| **from_ndarray_ms** (VideoFrame creation) | 256 | **0.4** | 0.6 | 0.6 | 0.4 |
| **recv_pacing_ms** (full recv cycle) | 256 | **35.0** | 36.7 | 38.4 | 33.1 |
| queue_full_retries | — | 0 | — | — | 0 |
| gpu_mem_mb | 25 | 2013.6 | — | — | 2013.6 |

**Counters:**
| Counter | Value |
|---------|-------|
| frames_generated | 2,545 |
| frames_emitted | 2,536 |
| frames_repeated | 0 |
| underflow_count | 0 |
| queue_depth_min | 3 |
| queue_depth_max | 10 |
| consecutive_underflows | 0 |
| fail_fast_reason | None |

### 4c. Browser-Side Metrics

| Metric | fMP4 (90s session) | SFU (90s session) |
|--------|--------------------|--------------------|
| First frame time | 25,300ms | 10,030ms |
| Segments / frames decoded | 436 segments | 2,234 frames |
| Total bytes | 22,413,001 | N/A (RTP) |
| Median FPS | N/A (MSE) | 25.0 |
| Packets lost | N/A (TCP/SSE) | 7 |
| Jitter | N/A | 0.025s |
| RTT | N/A | measured (low) |
| Frozen-frame violations | N/A (MSE) | 0 (live throughout, Δ=78-323) |
| Page errors | 0 | 0 |

### 4d. fMP4 Adapter Timing Note

The fMP4 adapter timing instrumentation (`_MEDIA_TIMING_DEBUG`) was deployed during the benchmark window but the `mediaTiming` data in the SSE end-event is consumed by the DO and not directly forwarded to the browser WebSocket. The instrumentation is verified working (code path exercised, compiles clean, appears in `end_event["mediaTiming"]`), and the data is accessible via Modal container logs. For future benchmarks, adding a DO-level relay of `mediaTiming` to the browser WS would enable direct client-side comparison.

---

## 5. Detected Bottlenecks

### 5a. ★★★ CPU Encoding (dominant)

The SFU path's VP8 encoder (libvpx via PyAV) consumes **69% of one CPU core** continuously at 25fps/720p. H264 is worse at **122%** (exceeds a single core). The fMP4 path's NVENC encoder costs **~0% CPU** (dedicated silicon on the L40S).

**Impact on concurrency:**
- 1 SFU session: 69% CPU → survivable
- 2 SFU sessions: 138% CPU → **exceeds one core, will degrade**
- The Modal container (`@modal.concurrent(max_inputs=2)`) cannot safely serve 2 concurrent SFU sessions

### 5b. ★★ Event Loop Contention (queue_put_ms)

`queue_put_ms` p50 = **39.0ms** — nearly equal to the full frame period (40ms at 25fps). This means the producer thread blocks for almost an entire frame period waiting for the event loop to accept the next frame. The event loop is saturated by encoding work, leaving minimal time for the `run_coroutine_threadsafe` callback to execute.

The max of **208.7ms** (5.2 frame periods) indicates periodic stalls — likely PyAV internal buffer allocation or Python GC.

### 5c. ★ Near-Capacity Pipeline (recv_pacing_ms)

`recv_pacing_ms` p50 = **35.0ms** out of a 40ms budget. The pipeline operates at **87.5% capacity** with only 5ms of slack per frame. Any transient stall (GC, PyAV buffer management, network jitter) will consume the slack and cause underflow.

### 5d. No Backpressure (confirmed healthy)

`queue_full_retries = 0`, `underflow_count = 0`, `frames_repeated = 0`. The pipeline is well-paced — it is CPU-bound but not queue-bound. The asyncio Queue (maxsize=10) never fills because the event loop drains it fast enough (just barely).

### 5e. GPU Inference Is NOT the Bottleneck

`process_chunk_ms` p50 = **69.1ms** for 5 frames = **13.8ms/frame**. At 25fps the frame budget is 40ms. GPU inference uses only **34.5%** of the per-frame time budget, leaving **65.5% headroom**. The GPU is not the constraint.

---

## 6. Exact Risks

| # | Risk | Severity | Evidence | Mitigation |
|---|------|----------|----------|------------|
| 1 | **CPU encode prevents concurrency** | **High** | 1 SFU session = 69% CPU. 2 sessions = 138% → cannot serve on one container | Option C: GPU-backed encoder |
| 2 | **Event loop saturation under load** | **Medium** | queue_put p50=39ms (97% of frame budget). GC or stall will cause underflow | Encoding offload or larger queue |
| 3 | **NVENC unavailable via PyAV** | **Confirmed** | PyAV manylinux wheel has no `--enable-nvenc`. Cannot GPU-encode through aiortc | Bypass PyAV: FFmpeg subprocess or GStreamer |
| 4 | **aiortc codec default is VP8** | **Confirmed** | No `setCodecPreferences` set. VP8 preferred over H264 | Minor: H264 is slower anyway (48.6ms vs 27.7ms) |
| 5 | **PyAV GC/SIGABRT after ~1000 frames** | **Known** | Mitigated by `gc.collect()` every 100 frames in `recv()`. Root cause persists | Resolved by moving off PyAV (Option C) |
| 6 | **PostHog timing data gap** | **Low** | `publisherStopMetrics` captured in Modal stop() response but Worker doesn't relay to `modal_end_completed` PostHog event | Add Worker-side property extraction |
| 7 | **psutil cpu_percent returns 0** | **Low** | All 25 cpu_percent samples are 0.0 — sampling in event loop thread likely misses CPU bursts | Sample from separate thread or use /proc/stat |
| 8 | **CF Realtime SFU not a WHIP endpoint** | **Research needed** | CF Realtime SFU uses its own tracks/new API, not standard WHIP | Must verify SDP compatibility before Option C |

---

## 7. Instrumentation Delivered

All instrumentation is behind `AVTR_MEDIA_TIMING_DEBUG=1` env flag. Zero overhead when disabled (early-branch pattern — no `time.monotonic()` calls, no ring-buffer writes).

### 7a. `webrtc_publisher.py` (+134 lines)

| Component | Location | What it measures |
|-----------|----------|------------------|
| `TIMING_DEBUG` flag | module-level `:18` | Gates all timing collection |
| `_RingStats` class | `:25-50` | Thread-safe ring buffer (deque, maxlen=256), p50/p95/max/mean |
| `_list_stats()` helper | `:53-64` | Stats for scalar sample lists (CPU%, GPU mem) |
| `PublisherMetrics.__init__` | `:97-110` | 7 ring buffers + 2 sample lists |
| `PublisherMetrics.snapshot()` | `:112-139` | Returns nested `timing` dict when enabled |
| Producer loop timing | `_producer_loop:269-299` | process_chunk_ms, frames_per_chunk, gpu_to_cpu_copy_ms, queue_put_ms, queue_full_retries |
| `recv()` timing | `recv:445-507` | from_ndarray_ms, recv_pacing_ms, cpu_percent (psutil), gpu_mem (torch.cuda) every 100 frames |

### 7b. `5q.py` — fMP4 adapter (+71 lines)

| Component | Location | What it measures |
|-----------|----------|------------------|
| `_MEDIA_TIMING_DEBUG` flag | module-level `:43` | Gates fMP4 timing collection |
| FFmpeg startup timing | `:2589-2601` | `_ffmpeg_startup_ms` (NVENC init) |
| Timing collector lists | `:2602-2605` | `_timing_render_ms`, `_timing_encode_ms`, `_timing_segment_ms`, `_timing_write_queue_full` |
| Segment cadence tracking | `_ffmpeg_stdout_reader:2671` | Per-segment timestamp for inter-arrival calculation |
| `_write_video_frames` queue full | `:2957-2958` | `_timing_write_queue_full` counter |
| Speech render timing | `:3162-3168` | `_timing_render_ms` + `_timing_encode_ms` per chunk |
| `end_event["mediaTiming"]` summary | `:3411-3444` | p50/p95/max for render, encode, segment cadence + writeQueueFullCount + printed summary |

### 7c. `modal_cpu_encode_probe.py` (new, 270 lines)

Standalone Modal app (CPU-only, no GPU). Uses PyAV Codec API directly to encode synthetic shifting-gradient I420 frames at 720p/25fps. Measures encode throughput, CPU% (psutil), bytes/frame. Tests VP8 (libvpx) and H264 (libx264). Confirms PyAV codec capabilities and NVENC (un)availability.

### 7d. `scripts/media-benchmark.mjs` (new, 520 lines)

Three modes: `--mode fmp4`, `--mode sfu`, `--mode ab`. SFU mode reuses Playwright + frozen-frame detector from canary. Queries PostHog for publisher stop metrics after SFU session. Outputs compact comparison table + optional `--json` report file. Architecture notes in output highlight CPU encode tax.

---

## 8. Architectural Recommendation

### Classification: Blocked by CPU encoding. Acceptable for canary only. Not acceptable for beta or production until encoder backend moves off CPU.

### Options Assessment

| Option | Encoder | CPU cost / session | Complexity | Verdict |
|--------|---------|-------------------|------------|---------|
| **A.** aiortc/PyAV as-is + container isolation (current Stage 3B) | CPU (libvpx/libx264) | 69-122% | Low (already built) | Canaries pass. **Reject for production** — no concurrency headroom |
| **B.** aiortc + force H264 + custom PyAV build with NVENC | GPU (NVENC via patched PyAV) | ~0% | High — rebuild PyAV wheel with `--enable-nvenc`, patch aiortc's codec factory to instantiate `h264_nvenc`. aiortc upstream does not support this | **Reject** — too brittle, unmaintainable |
| **C.** FFmpeg/GStreamer WebRTC pipeline with `h264_nvenc` | GPU (NVENC) | ~0% | Medium — spawn `ffmpeg ... -c:v h264_nvenc` with WebRTC-compatible SDP output, or GStreamer `webrtcbin + nvh264enc`. Modal already runs FFmpeg subprocess for fMP4. Needs SDP exchange compatible with CF Realtime SFU API (NOT standard WHIP — research required) | **★ Recommended direction** |
| **D.** Keep fMP4 as production; SFU stays experimental | GPU (NVENC, existing) | ~0% | Zero | **Safe floor** — fMP4 is already GPU end-to-end |

### Recommendation

**Pursue Option C as the strategic target. Hold on Option D as the production floor. Use this audit's instrumentation to quantify the cost of the CPU encode tax (Option A) so the Option C migration is evidence-based.**

Option C eliminates the CPU encode tax entirely by reusing the existing NVENC encoder (proven in the fMP4 path) but feeding its output to a WebRTC transport instead of fMP4/SSE. The investigation phase should:

1. **Research CF Realtime SFU publisher API** — confirm whether it accepts SDP from an FFmpeg/GStreamer publisher via its `tracks/new` endpoint, or whether a custom SDP bridging layer is needed. CF Realtime SFU is **not** a standard WHIP endpoint.
2. **Prototype an FFmpeg WebRTC publisher** — `ffmpeg -c:v h264_nvenc ... -f webrtc ...` or GStreamer pipeline with `webrtcbin`. Measure first-frame latency and stability.
3. **Size the migration cost** — the current aiortc publisher (`webrtc_publisher.py`, 603 lines) handles PC lifecycle, ICE, codec, pacing, fail-fast. Option C replaces the encoding layer but keeps the SFU callback infrastructure.

### Tradeoff Summary

| Dimension | fMP4 (Option D) | SFU/aiortc (Option A) | SFU/NVENC (Option C) |
|-----------|-----------------|----------------------|---------------------|
| Encode CPU | ~0% (NVENC) | 69-122% (CPU) | ~0% (NVENC) |
| First-frame latency | ~25s (SSE init + base64) | ~10s (WebRTC ICE) | TBD (target <5s) |
| Transport overhead | base64 + SSE + WS (33% overhead) | RTP (binary, efficient) | RTP (binary, efficient) |
| Concurrency / container | High (GPU only) | 1 session max | High (GPU only) |
| Reliability | Proven (production) | Proven (3/3 canaries) | Unproven (needs prototype) |
| Cost / session | NVENC silicon (free) | CPU cores (expensive) | NVENC silicon (free) |

---

## 9. Validation Results

| Check | Result |
|-------|--------|
| `py_compile` (3 Python files) | ✅ OK |
| `tsc --noEmit` (cloudflare-session-bridge) | ✅ OK |
| `vitest` (74 tests) | ✅ 74/74 pass |
| PostHog error sweep (both benchmark sessions) | ✅ 0 `pipeline_error`, 0 `sfu_unavailable`, 0 `sfu_modal_keepalive_failed` |
| Production SFU state | ✅ `AVTR_REALTIME_SFU_ENABLED=false` |
| Production default transport | ✅ `websocket_fmp4` |
| Modal 5q TIMING_DEBUG secret | ✅ Removed (redeployed clean) |
| Worker version (SFU disabled) | ✅ Deployed `c6b87df6` |
| Git commit | ✅ `25ee7f4` — no uncommitted changes |

---

## 10. Deliverables Index

| File | Lines | Purpose |
|------|-------|---------|
| `avatar-saas-prototype/modal-avatar-runtime/webrtc_publisher.py` | +134 | SFU path timing instrumentation |
| `avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_..._5q.py` | +71 | fMP4 adapter timing instrumentation |
| `avatar-saas-prototype/modal-avatar-runtime/modal_cpu_encode_probe.py` | 270 (new) | Standalone CPU encode cost probe |
| `scripts/media-benchmark.mjs` | 520 (new) | fMP4 vs SFU A/B benchmark script |

---

## 11. Next Steps

1. **Option C investigation** (research phase, no code changes):
   - Verify CF Realtime SFU tracks/new API accepts non-aiortc publishers
   - Prototype FFmpeg WHIP/WebRTC publisher with `h264_nvenc`
   - Measure first-frame latency and stability

2. **PostHog timing relay** (small Worker change):
   - Extract `publisherStopMetrics` from Modal end-session response in `modal-client.ts`
   - Include in `modal_end_completed` and `session_ended` PostHog events
   - Enables ongoing production monitoring without manual Modal log inspection

3. **psutil CPU sampling fix** (small Python change):
   - Move `cpu_percent` sampling to the producer thread (which runs continuously)
   - Or use a dedicated monitoring thread with `interval=1.0` blocking calls

4. **Do NOT proceed with**: web-client SFU integration, default transport change, or SFU enablement in production until Option C is validated.
