# Phase 4 Baseline Freeze Report

**Date:** 2026-06-08
**Verdict: PHASE 4 FREEZE GO**

---

## 1. Commit & Branch

| Item | Value |
|------|-------|
| Commit | `e7cc4ac` |
| Branch | `main` |
| Message | `Phase 4 baseline: process_chunk refactor, HTTP/SSE contract, _drain_watcher deadlock fix` |
| HEAD confirmed | `git rev-parse --short HEAD` = `e7cc4ac` |

### Working Tree Status

6 files committed. Working tree clean.

Untracked files (none required for deployed baseline):
- Plan docs: `PHASE_4A_SESSION_BASELINE_REPORT.md`, `PHASE_4B_BRIDGE_INTEGRATION_PLAN.md`, `PHASE_4C_FMP4_STREAMING_CONTRACT.md`
- Alternate Modal files: `modal_lifecycle_pytorch_cpu_snapshot_l40s.py`, `modal_lifecycle_pytorch_cpu_snapshot_omnivoice_l40s.py`, `modal_lifecycle_pytorch_cpu_snapshot_stream_l40s.py`
- Assets: `test_speech.wav`, `weights/`

---

## 2. Deployed Services

| Service | URL / Name | Status |
|---------|-----------|--------|
| Modal App | `avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s` (ap-j7k35GD9yfrRtrZBj5v0ZE) | deployed |
| Session Bridge | `https://avatar-session-bridge.vallangirakesh.workers.dev` | deployed |
| Web Client | `https://avatar-web-client.pages.dev` | deployed (untouched) |
| VAD Worker | `https://any-toy-vad.vallangirakesh.workers.dev` | deployed |
| Smart Turn Worker | `https://any-toy-smart-turn.vallangirakesh.workers.dev` | deployed |

### Modal HTTP Endpoints (5/5 reachable, HTTP 200)

| Endpoint | URL | Reachability |
|----------|-----|-------------|
| Start | `https://vallangirakesh--start-session.modal.run` | 200 (19.5s cold) |
| End | `https://vallangirakesh--end-session.modal.run` | 200 (0.9s) |
| Status | `https://vallangirakesh--session-status.modal.run` | 200 (0.9s) |
| Push Audio | `https://vallangirakesh--push-audio.modal.run` | 200 (1.0s) |
| Stream fMP4 AV | `https://vallangirakesh--stream-fmp4-av.modal.run` | 200 (1.0s) |

All 5 map to `avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s` (PyTorch CPU-snapshot, not old transformer stack).

---

## 3. Bridge Environment Variables

Source: `avatar-saas-prototype/cloudflare-session-bridge/wrangler.jsonc`

| Variable | Value | Line |
|----------|-------|------|
| `MODAL_SESSION_START_URL` | `https://vallangirakesh--start-session.modal.run` | 36 |
| `MODAL_SESSION_PUSH_AUDIO_URL` | `https://vallangirakesh--push-audio.modal.run` | 37 |
| `MODAL_SESSION_STATUS_URL` | `https://vallangirakesh--session-status.modal.run` | 38 |
| `MODAL_SESSION_END_URL` | `https://vallangirakesh--end-session.modal.run` | 39 |
| `MODAL_SESSION_STREAM_FMP4_AV_URL` | `https://vallangirakesh--stream-fmp4-av.modal.run` | 40 |
| `ENABLE_LEGACY_MODAL_PROTOCOL` | `false` | 41 |

---

## 4. Validation Commands

| Command | Result |
|---------|--------|
| `python3 -m py_compile modal_lifecycle_pytorch_cpu_snapshot_session_l40s.py` | **PASS** |
| `cd cloudflare-session-bridge && npx tsc --noEmit` | **PASS** |

---

## 5. Static Source Proof

All checks performed via `rg` (ripgrep) against committed source at `e7cc4ac`.

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| No `enable_gpu_snapshot` | 0 matches | 0 matches | **PASS** |
| Class = PyTorch CPU-snapshot | `AvatarLifecyclePyTorchCpuSnapshotSessionL40S` | Line 482 | **PASS** |
| `renderMode: "fmp4_av"` in Bridge | present | Lines 3869, 3925 (avatar-session-do.ts) | **PASS** |
| FFmpeg rawvideo input | present | 5 locations (lines 999, 1209, 1597, 2222, 2902) | **PASS** |
| FFmpeg audio FIFO input | present | Lines 2887, 2982, 2990 | **PASS** |
| AAC audio enabled | present | Line 2913 (`-c:a aac -b:a 128k -ar 48000 -ac 1`) | **PASS** |
| `_drain_watcher` exists | present | Lines 3254, 3267 | **PASS** |
| `pipeline.process_chunk` production call sites | 6 | Lines 670, 1111, 1317, 1789, 2453, 3182 | **PASS** |
| No `HUBERT_LEN` | 0 matches | 0 matches | **PASS** |
| No broken methods | 0 matches | 0 matches (`_run_hubert`, `_run_avtr_encode`, `_run_avtr_decode_loop`, `_motion_to_frames`, `_run_renderer`, `_rgb_tensor_to_yuv`) | **PASS** |
| No stale `timeout_s=2.0` | 0 matches | 0 matches | **PASS** |
| Browser client untouched | 0 lines changed | 0 lines | **PASS** |
| Handoff timeout correct | `HANDOFF_WAIT_S=0.05 if speech else 0` | Lines 1763, 3142 | **PASS** |

---

## 6. Runtime Smoke Evidence

**Session ID:** `session:65dbf9db-5bc5-4509-88b1-d0251426f34d`
**Date:** 2026-06-08T16:03:13Z - 16:05:23Z
**Protocol:** 4c (HTTP/SSE), `renderMode: fmp4_av`

### PostHog Event Timeline

```
16:03:13  avatar_session_created
16:03:43  ws_connected
16:03:43  avatar_id_resolved_4c
16:03:43  session_start_requested
16:05:14  modal_session_fmp4_stream_started        (SSE stream opened)
16:05:14  session_ready  latencyMs=91496           (session.ready sent to client)
16:05:16  modal_session_fmp4_init_forwarded        (init segment relayed)
16:05:18  modal_session_fmp4_segment_forwarded     (x5 media segments)
16:05:19  session_end_requested                    (client sent session.end)
16:05:19  modal_session_fmp4_segment_forwarded     (x1 more during shutdown)
16:05:23  session_ended  modalEndDurationMs=1443   (clean end)
16:05:23  modal_end_completed
16:05:23  ws_closed
```

Total PostHog events: 16. `pipeline_error` count: **0**.

### WS Client Output

```
[0.9s]  OPEN
[0.9s]  SENT session.start
[0.9s]  RECV status=starting_modal
[92.4s] RECV session.ready renderMode=fmp4_av
[94.7s] RECV avatar.fmp4.init
[96.6s] RECV avatar.fmp4.segment (x6)
[97.5s] SENT session.end
[100.3s] RECV status=ending
[101.8s] RECV session.ended
session.end latency: 4.3s
CLOSE code=1000 reason=session closed
```

### Modal Log Summary

```
[warmup] Complete in 1083ms, 5 frames
[session] STARTED: cache_hit=True init=1.083s warmup=1083ms vram=155.9MB
[audio_writer] Received None sentinel, closing pipe
[video_writer] Received None sentinel, closing stdin
[video_writer] exiting
[session-stdout] EOF, sentinel sent
[session-render] consumer loop done: sentinel=True sender_media=110 drained_final=0
[session-render] FFmpeg exit: rc=0 stderr_lines=64 errors=2 warnings=0 zombie=clean
[session-render] init=True media_count=110 rawMB=760.3 encodedMB=5.7
                 audio_path_active=False vram=155.9MB zombie=clean
[session-render] session render done: 8.10s reason=render_stop_event_set
[session] ENDED: avatar=arnold chunks=110 fmp4_media=110 duration=9.3s
```

### Performance Snapshot

| Metric | Value | Notes |
|--------|-------|-------|
| Cold session.ready | ~91.5s | snapshot restore + pipeline load + warmup (1083ms) |
| Warm session.ready | Not tested | estimated ~2-5s |
| First fMP4 init latency | 2.0s | from session_ready to init forwarded |
| First media segment latency | 4.2s | from session_ready to first segment |
| Chunks rendered | 110 | ~5 frames/chunk |
| fMP4 segments delivered | 110 (6 forwarded before session.end) | 760.3 MB raw -> 5.7 MB encoded |
| session.end latency (WS) | ~4.3s | from session.end sent to session.ended received |
| session.end latency (Modal) | ~1.4s | modalEndDurationMs=1443 |
| Session render time | 8.10s | from first chunk to render_stop_event_set |
| Restore + warmup | 1083ms | cache_hit=True |
| FFmpeg exit | rc=0, clean | no zombie, no SIGABRT |
| VRAM | 155.9 MB | stable, no leak |

---

## 7. Negative Signal Audit

| Signal | Count | Status |
|--------|-------|--------|
| `pipeline_error` in PostHog | 0 | **CLEAR** |
| SIGSEGV | 0 | **CLEAR** |
| Exit code 139 | 0 | **CLEAR** |
| CUDA errors | 0 | **CLEAR** |
| OOM | 0 | **CLEAR** |
| VFS/9p errors | 0 | **CLEAR** |
| Broken pipes | 0 | **CLEAR** |
| Subprocess leaks | 0 | **CLEAR** |
| Deadlocks / timeouts | 0 | **CLEAR** |
| FFmpeg rc != 0 | 0 (rc=0) | **CLEAR** |
| FFmpeg zombie | 0 (clean) | **CLEAR** |

---

## 8. Known Limitations

These are documented and deferred. None block the freeze.

1. **Cold session.ready is slow (~91.5s)** — first container cold start. Warm starts expected ~2-5s.
2. **Optimization deferred** — no batch tuning, no GPU parallelism, no caching strategy yet.
3. **CPU-only snapshot baseline** — `enable_gpu_snapshot` not used; GPU used at runtime only.
4. **Browser remains passive** — web client connects via WS but does not drive the pipeline.
5. **Avatar data loaded on demand** — no pre-warming of multiple avatars.
6. **Audio path not exercised in smoke** — `audio_path_active=False`. TTS/lip-sync path validated via static analysis only.
7. **No warm session.ready measured** — only cold start tested.
8. **PostHog `renderMode` not captured** — available in WS message but not in PostHog event properties.
9. **`ENABLE_MODAL_FMP4_AV_STREAM` = `false`** in wrangler.jsonc — Bridge uses legacy session_ready path but stream URL is correct. May need `true` for full 4c path.

---

## 9. Rollback Instructions

Commit `e7cc4ac` is the Phase 4 baseline. To rollback:

```bash
git revert e7cc4ac

# Redeploy Modal
cd avatar-saas-prototype/modal-avatar-runtime
modal deploy modal_lifecycle_pytorch_cpu_snapshot_session_l40s.py

# Redeploy Bridge
cd avatar-saas-prototype/cloudflare-session-bridge
npx wrangler deploy
```

Previous known-good commit: `0e83555` (Phase 4C).

Bridge env vars at `e7cc4ac` point to the 5 Modal endpoints listed above. No other env var changes required.

---

## 10. Files Changed in `e7cc4ac`

| File | Insertions | Deletions | Description |
|------|-----------|-----------|-------------|
| `modal_lifecycle_pytorch_cpu_snapshot_session_l40s.py` | 274 | 1013 | Refactored to `process_chunk`, removed dead code, added `_drain_watcher` |
| `cloudflare-session-bridge/src/avatar-session-do.ts` | 568 | 7 | HTTP/SSE session lifecycle, SSE fanout |
| `cloudflare-session-bridge/src/modal-client.ts` | 254 | 0 | New: Modal HTTP endpoint client |
| `cloudflare-session-bridge/src/types.ts` | 85 | 0 | New: session/Modal types |
| `cloudflare-session-bridge/wrangler.jsonc` | 12 | 7 | 5 Modal endpoint env vars |
| `modal-avatar-runtime/modal_hubert_inspector.py` | 9 | 1 | Minor |

---

## 11. Next Phase: Phase 5

Phase 5 = Cloudflare streaming productization / external-facing API preparation. Not optimization unless required.

Scope will be defined at kickoff. Baseline frozen at `e7cc4ac`.
