# Stage 4C Closeout — Where We Are

**Date:** 2026-06-20
**POC verdict:** **GO** for the profile `1280x720 H264 NVENC @ 2 Mbps` (3/3 canaries pass all 14 hardened criteria). See `STAGE_4C_TRANSPORT_ISOLATION_REPORT.md`.
**Production transport:** **fMP4 over WebSocket → browser MSE.** SFU is **not** enabled.
**Production runtime changes in this closeout:** **None.** This is a docs/report/roadmap-only closeout. No Worker, Modal runtime, web-client, or repo-root `src/` code changed.

---

## 1. Pipeline evolution — three paths

### Production floor (LIVE) — fMP4
```
AVTR GPU frames → FFmpeg NVENC (h264_nvenc) → fMP4 → Worker/DO WebSocket → browser MSE
```
This is the deployed path (`avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s-5q` → `avatar-session-bridge` → `avatar-web-client.pages.dev`). fMP4 is produced via `ffmpeg ... -f mp4 -movflags +frag_keyframe+empty_moov+default_base_moof+frag_duration 200000`, paced by the `_SegmentSender`, and relayed to the browser as `avatar.fmp4.init` / `avatar.fmp4.segment` messages appended to a `SourceBuffer`. **This remains the production transport.**

### Rejected interim SFU path — aiortc / PyAV (NOT adopted)
```
AVTR frames → aiortc / PyAV CPU encode → SFU
```
Blocked at the Stage 4A audit: the CPU encoder (VP8 via aiortc/PyAV) saturated the CPU (~420%) and could not sustain 720p25 alongside the renderer. Recorded as a rejection, not a fallback. See `avatar-saas-prototype/modal-avatar-runtime/STAGE_4A_SFU_PERFORMANCE_AUDIT_REPORT.md`.

### Option C (POC — validated, NOT enabled) — GStreamer GPU NVENC → SFU
```
AVTR GPU frames → GStreamer appsrc → nvh264enc (GPU H264) → webrtcbin → Cloudflare Realtime SFU → browser WebRTC
```
Feasibility is proven (`FEASIBILITY_REPORT.md`) and the Stage 4C canaries pass at `bitrate=2000` (= **2 Mbps**; `nvh264enc` `bitrate` is in **kbit/sec**). The producer side (AVTR → NVENC → GStreamer) is healthy across all runs: 0 underflow, 0 errors, 66–71ms chunk latency. The remaining work is **production integration design**, not feasibility.

> **SFU remains disabled in production.** `AVTR_REALTIME_SFU_ENABLED` is unset (defaults to `false`); the default transport is `websocket_fmp4`. Enabling SFU is a separate, gated decision (see Next phase).

---

## 2. Corrected verdict summary

- **Profile:** `1280x720 H264 NVENC @ 2 Mbps`, `gop-size=50`, `preset=low-latency-hq`, constrained-baseline, level 3.1.
- **Why 2 Mbps:** The uncapped auto/VBR bitrate (~10–11 Mbps, ≈1,100 RTP packets/s) created excessive RTP packet pressure and caused visible freezes (1.0–1.2% observed loss, 8.4–25.7s freeze duration). Capping to 2 Mbps reduced packet pressure ≈4–5×; **observed loss fell to 0.0%** (0–1 packets lost over 120s) and all 3 canaries passed with 0.0–2.3s freeze duration.
- **What the evidence does not claim:** that the path's underlying loss rate is constant and merely scales with packet count, or that SFU routing is definitively ruled out. Candidate-pair instability was **not indicated** by the observed stats (consistent srflx→host, 13–22ms RTT, no ICE restarts), but receiver-side `packetsLost` cannot localize loss to Modal→SFU vs SFU→browser vs local ISP.
- **`freezeCount` is diagnostic, not a blocker.** Active blockers: active decoded FPS, total freeze duration, manual stalls, producer errors, underflow, codec, teardown, and source/publisher metrics. See `STAGE_4C_VALIDATION_HARDENING_REPORT.md` §1 for the canonical 14-criterion table.

---

## 3. Production safety (this closeout)

```
git diff HEAD -- src/ avatar-saas-prototype/cloudflare-session-bridge/ \
                  avatar-saas-prototype/modal-avatar-runtime/ avatar-saas-prototype/web-client/
→ empty (no tracked production runtime file modified)
```

The POC code files under `option-c-nvenc-publisher-poc/` (`modal_gst_publisher.py`, `frame_source.py`, `public/index.html`, `run-option-c-test.cjs`) are modified in the working tree from earlier POC work but are **POC-only** (not imported by any production runtime) and were **not** staged as part of this docs closeout.

---

## 4. Next recommended phase

Two independent workstreams. **Do not mix them.**

1. **Internal Modal endpoint authentication (pre-production blocker).** HMAC-sign Worker→Modal requests (shared secret + timestamp + request-id/nonce + body signature); Modal rejects missing/stale/invalid signatures. Covers `push-audio`, `push-audio-batch`, `start`, `status`, `stream`, `end`, and any SFU-callback-sensitive paths. No raw keys, WS tokens, audio payloads, or signatures logged. Tracked as a roadmap row in `docs/AVATAR_PLATFORM_ROADMAP.md`; **required before broad public API launch**, not necessarily blocking a controlled private beta.

2. **Gated SFU production-integration design (separate decision).** If/when SFU is to be enabled, design the transport switch (`AVTR_REALTIME_SFU_ENABLED`), the Modal publisher lifecycle in production, and the rollback path back to fMP4 — as its own phase, after the security hardening above.

Recommended order: **(1) first**, then (2) only when there is a product reason to move off fMP4.
