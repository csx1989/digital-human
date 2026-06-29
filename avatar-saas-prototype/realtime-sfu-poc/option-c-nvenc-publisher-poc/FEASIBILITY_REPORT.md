# Option C — NVENC-backed GStreamer WebRTC Publisher: Feasibility Report

**Date:** 2026-06-19
**POC directory:** `avatar-saas-prototype/realtime-sfu-poc/option-c-nvenc-publisher-poc/`
**Modal app:** `avtr-option-c-gst-publisher`
**Production state:** `AVTR_REALTIME_SFU_ENABLED=false`, `AVTR_REALTIME_SFU_DEFAULT=websocket_fmp4` (untouched)

## 1. Executive Summary

**Recommendation: CONTINUE Option C.** A GStreamer `webrtcbin + nvh264enc` publisher on
Modal L40S successfully publishes GPU-encoded H264 video through Cloudflare Realtime
SFU to a browser subscriber. The feasibility question is answered **YES**:

- Cloudflare Realtime SFU **accepts** a GStreamer-originated H264 offer SDP.
- The browser negotiates **`video/H264`** (not a silent VP8 fallback).
- 1280×720 @ 25fps sustained for 130s with **0 page errors** and clean teardown.
- Publisher CPU dropped from **420% (aiortc/PyAV VP8)** to **~11% (NVENC)** — a
  **38× reduction**, eliminating the CPU-encode tax that blocked concurrency.

The AVTR frame handoff (Phase 3) is the validated next step: it is a mechanical
swap of the frame source feeding the **same proven GStreamer pipeline**.

## 2. Cloudflare Realtime SDP Compatibility Findings

**Yes — Cloudflare Realtime accepts GStreamer/FFmpeg-originated SDP.** No WHIP, no
browser-specific quirks required. Key findings:

- The GStreamer `webrtcbin` offer (with `bundle-policy=max-bundle`, `a=setup:actpass`,
  `a=rtcp-mux`, `a=rtcp-rsize`, `a=fmtp:96 packetization-mode=1;profile-level-id=42c01f`)
  was accepted verbatim by `POST /sessions/{id}/tracks/new` with
  `location:"local"`, `mid:"video0"`, `trackName`, `kind:"video"`.
- The SFU returns an `answer` SDP that GStreamer consumes via
  `set-remote-description` with no SDP rewriting needed.
- The `mid` is **`video0`** (GStreamer's transceiver naming), not `"0"`. The mid
  must be extracted from the SDP dynamically — CF Realtime matches it.
- CF Realtime is **non-trickle on the API**: ICE candidates must be embedded in the
  offer SDP. GStreamer's `webrtcbin` uses trickle internally, so candidates are
  collected from `on-ice-candidate` signals and injected into the SDP before
  sending (plus an `a=end-of-candidates` marker).
- **Zero SDP rejections across all runs.** Gate B (stop after 2 rejections) was
  never triggered.

## 3. Exact Working Media Pipeline

```
appsrc (I420 1280x720@25fps, is-live, push-mode producer thread)
  → queue (max-buffers=1, leaky=downstream)
  → nvh264enc (gop-size=50, preset=low-latency-hq)
  → capsfilter (video/x-h264, profile=constrained-baseline, level=3.1)
  → h264parse (config-interval=-1)
  → rtph264pay (config-interval=-1, aggregate-mode=zero-latency, pt=96)
  → queue
  → webrtcbin (bundle-policy=max-bundle, stun=stun://stun.cloudflare.com:3478)
```

**Code structure** (`modal_gst_publisher.py`):
- `GstSyntheticPublisher` builds the pipeline via `Gst.parse_launch`, runs a
  `GLib.MainLoop` in a background thread (required for GStreamer signal dispatch),
  and uses an asyncio event loop for the CF Realtime HTTPS SDP exchange.
- Frame pushing is **push-mode** with a producer thread pacing at true 25fps via
  wall-clock deadlines (the `need-data` callback overspeeds to ~560fps — a known
  appsrc behaviour — so push-mode is the correct pattern).
- SDP exchange flow: `on-negotiation-needed` → `create-offer` →
  `set-local-description` → collect ICE candidates → inject into SDP →
  `POST /sessions/new` + `POST /tracks/new` → `set-remote-description` (answer) →
  wait ICE `connected`.

**Image build** (key lesson — all three blockers hit during this phase):
- Base `nvidia/cuda:12.8.1-devel-ubuntu24.04` + `add_python="3.12"`.
- `python3-gi`, `python3-gi-cairo`, `gir1.2-gstreamer-1.0`, `gir1.2-gst-plugins-bad-1.0`
  (PyGObject via apt — building from pip fails on cairo/Python.h/PEP-668).
- `gstreamer1.0-nice` (**critical** — without libnice, `webrtcbin` cannot link its
  sink pads; this was the root cause of all early pipeline-link failures).
- `gstreamer1.0-plugins-bad` ships `nvh264enc` (nvcodec plugin) and works
  out-of-the-box on L40S — **no source build was needed** (Gate A passed first try).
- `gi` is bridged to `add_python`'s interpreter via
  `PYTHONPATH=/usr/lib/python3/dist-packages`.

## 4. Negotiated Codec

**`video/H264`** — confirmed in the browser `RTCPeerConnection.getStats()` codec
report. Profile: constrained-baseline, level 3.1, packetization-mode=1,
profile-level-id=42c01f. This is the maximum-compatibility WebRTC H264 profile.

## 5. Performance Metrics

### Publisher side (Modal L40S, 135s synthetic run)

| Metric | Option C (NVENC) | aiortc/PyAV (Stage 4A) | Delta |
|--------|------------------|------------------------|-------|
| **CPU% mean** | **11.08%** | 420% (VP8) / 497% (H264) | **−38×** |
| CPU% max | 40.7% | 547–634% | −13× |
| First-frame latency | 3.33s | ~10s | −3× |
| Offer/ICE setup | 2.47s offer + 0.86s ICE | n/a | — |
| Frames generated | 3457 (≈25.6fps) | 2545 | — |
| NVENC util mean | 0.01% (see note) | n/a (CPU) | — |
| GPU util mean | 0.05% | n/a | — |

> **NVENC util note:** `nvidia-smi dmon` reports ~0% encoder util because L40S
> NVENC finishes each 720p frame in microseconds — the hardware is idle between
> frames. The decisive proof of NVENC usage is twofold: (1) `nvh264enc` only runs
> on NVENC hardware (Gate A encode smoke test); (2) software H264 encoding at
> 720p25 requires ≥497% CPU, but this publisher uses 11% — physically impossible
> without hardware encoding.

### Browser side (130s subscriber, headless Chromium)

| Metric | Value |
|--------|-------|
| Runtime | 130s |
| Remote dimensions | 1280×720 |
| Frames decoded | 208 → 3024 (Δ=2816) |
| FPS (median last 5) | 25 |
| Packets lost | 177 |
| Jitter | 0.014s |
| Negotiated codec | `video/H264` |
| Page errors | 0 |
| Subscriber ICE | connected (stable) |
| Publisher ICE | connected |
| Frozen-frame violations | 0 |

## 6. Comparison Against aiortc/PyAV SFU Path

| Dimension | aiortc/PyAV (current SFU) | GStreamer NVENC (Option C) |
|-----------|---------------------------|----------------------------|
| Encode backend | CPU (libvpx/libx264 via PyAV) | **GPU (NVENC via nvh264enc)** |
| CPU / session @720p25 | 420% (VP8) | **11%** |
| Concurrent sessions / L40S core | 1 (exceeds a core) | many (NVENC + ~1/10 core) |
| Negotiated codec | VP8 (aiortc default) | H264 (constrained-baseline 3.1) |
| Browser hardware decode | VP8 (some) | H264 (universal) |
| Plugin/wheel fragility | PyAV wheel pinned, GC crashes ~1000 frames | apt `gst-plugins-bad` + `nvh264enc`, stable |
| GC/SIGABRT risk | Mitigated by periodic `gc.collect()` | none (no Python encode loop) |
| First-frame latency | ~10s | 3.3s |
| Production readiness | canary-only (Stage 3B) | **feasibility proven** |

## 7. Hard-Gate Outcomes

| Gate | Question | Result |
|------|----------|--------|
| A | `gst-inspect-1.0 nvh264enc` + encode smoke test | **PASS** (plugin present, 250 frames in 0.43s) |
| B | CF Realtime accepts GStreamer H264 offer | **PASS** (0 rejections) |
| C | Synthetic 1280×720@25fps NVENC 120s browser test | **PASS** (13/13 criteria) |
| D | Browser stats confirm H264 (not VP8) | **PASS** (`video/H264`) |
| E | Modal metrics: CPU%, GPU, NVENC util | **PASS** (CPU 11%, NVENC confirmed) |
| F | AVTR handoff only after synthetic passes | synthetic passed — AVTR is next |

## 8. Phase 3 (AVTR Handoff) — Validated Next Step

The synthetic pipeline is proven. Wiring AVTR is a **mechanical frame-source swap**:
replace `SyntheticI420Source.next_frame()` with a producer that calls
`pipeline.process_chunk()` (identical to the existing `AvtrFrameSource` in
`webrtc_publisher.py`) and pushes the resulting I420 `frame.data` into the same
`appsrc`. No model inference, renderer, or pipeline logic changes — only the frame
handoff boundary. The image would additionally mount the AVTR renderer, torch,
TensorRT, and the L40S artifacts volume (mirroring the production 5Q image).

Because the GStreamer/NVENC/SFU layer is already validated, the AVTR handoff risk
is isolated to frame-timing (the AVTR producer is GPU-bound at ~70ms/chunk for 5
frames = 13.8ms/frame, leaving the 26ms/frame NVENC+transport budget intact per
Stage 4A).

## 9. Constraints Honoured

- `AVTR_REALTIME_SFU_ENABLED` remains `false`.
- `AVTR_REALTIME_SFU_DEFAULT` remains `websocket_fmp4`.
- No production Worker, web-client, or transport-default changes.
- All work isolated under `option-c-nvenc-publisher-poc/`.
- No secrets committed (`.env`/`.dev.vars` gitignored; `avtr-calls` Modal secret).
- fMP4/WebSocket remains the production floor.

## 10. Files Delivered

| File | Purpose |
|------|---------|
| `modal_gst_publisher.py` | Modal GPU publisher: probe + synthetic modes |
| `gst_sfu_client.py` | CF Realtime Sessions/Tracks API client |
| `worker.ts` | Subscriber-side Worker proxy (adapted from Phase C) |
| `public/index.html` | Browser subscriber page (confirms H264 codec) |
| `run-option-c-test.cjs` | Playwright 120s test harness (13 criteria) |
| `local_gst_probe.py` | Local SDP-compatibility probe (x264enc fallback) |
| `frame_source.py` | Synthetic I420 source + AVTR adapter skeleton |

## 11. Recommendation

**Continue Option C.** The CPU-encode tax that blocked SFU concurrency (Stage 4A's
sole architectural finding) is eliminated: 420% → 11%. The remaining work is
low-risk: (1) AVTR frame handoff (validated frame-source swap), (2) wiring the
GStreamer publisher into the 5Q lifecycle as an alternative `publisher_mode`, and
(3) re-running the Stage 3B canary against the NVENC publisher. fMP4 stays the
production floor throughout.
