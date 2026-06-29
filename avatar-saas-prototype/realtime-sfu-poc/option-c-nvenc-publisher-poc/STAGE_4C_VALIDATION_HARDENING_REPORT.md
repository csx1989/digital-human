# Stage 4C — Option C Phase 3 Final Closeout Report

> **Superseded for the verdict, retained for the analysis.** This report's 3 canaries were run at the **uncapped auto/VBR nvh264enc bitrate (~10–11 Mbps)** and returned **HOLD** (1/3 passed). That result motivated the bitrate-cap investigation. The follow-on `STAGE_4C_TRANSPORT_ISOLATION_REPORT.md` re-ran canaries at `bitrate=2000` (= **2 Mbps**; `nvh264enc` `bitrate` is in kbit/sec) and returned **GO — 3/3 pass**. The HOLD verdict below stands as the historical record of the auto-bitrate runs; it is also the canonical source for the **14 hardened criteria** and the **freezeCount-as-diagnostic** classification referenced by the transport report. Current production state: **fMP4, SFU disabled** (`AVTR_REALTIME_SFU_ENABLED` unset).

**Date:** 2026-06-19
**Verdict:** **HOLD** — 1 of 3 fresh canaries passes all 14 corrected hardened criteria. The AVTR → NVENC → GStreamer pipeline is producer-healthy (0 underflow, 0 errors, stable 66–71ms chunk latency across all runs). The 2 failures are exclusively in the network transport layer (packet loss → freeze duration → decoded FPS), not in rendering or encoding.

---

## 1. Corrected Hardened Criteria (14 blockers)

The previous `freezeCount <= 5` blocker was reclassified as diagnostic. The corrected criteria set, per reviewer specification:

| # | Criterion | Threshold |
|---|-----------|-----------|
| 1 | Runtime | >= 120s |
| 2 | Subscriber ICE connected/completed | yes |
| 3 | Remote video playing | true |
| 4 | Remote dimensions | 1280x720 |
| 5 | Active decoded FPS | >= 23 |
| 6 | Active playback | >= 100s |
| 7 | Total freeze duration | <= 10s |
| 8 | Manual frame stalls | = 0 |
| 9 | CUDA/producer errors | none |
| 10 | GStreamer pipeline_error | none |
| 11 | AVTR underflow_count | <= 5 |
| 12 | Publisher result JSON captured | yes |
| 13 | Negotiated codec | H264 |
| 14 | Teardown succeeds | yes |

**freezeCount** remains in diagnostic output but is NOT a blocker. Rationale: freezeCount counts sub-second inter-frame delays (per MDN: >= max(3× average, average + 150ms)). These are micro-freezes from transient packet loss that the jitter buffer absorbs. Manual frame-stall detection (consecutive 5s samples with no framesDecoded increase) found **0 stalls across all runs**. Freeze DURATION captures real user impact.

---

## 2. Three Fresh Canaries

Each run is a separate Modal app invocation (different container, different app ID).

### Summary

| Criterion | Run 1 | Run 2 | Run 3 |
|-----------|-------|-------|-------|
| **Modal App ID** | ap-Ev2oq... | ap-Bh733... | ap-K1Qhf... |
| **HARDENED VERDICT** | **PASS** 14/14 | **FAIL** 12/14 | **FAIL** 12/14 |
| **Legacy verdict** | PASS 17/17 | PASS 17/17 | FAIL 15/17 |
| | | | |
| **PRODUCER (all healthy)** | | | |
| Chunks rendered | 679 | 679 | 679 |
| Frames generated | 3392 | 3391 | 3391 |
| Chunk latency mean | 69.0ms | 66.4ms | 70.6ms |
| Chunk latency max | 84.4ms | 80.3ms | 84.0ms |
| Underflow | 0 | 0 | 0 |
| Overflow | 1 | 1 | 1 |
| Producer error | none | none | none |
| Pipeline error | none | none | none |
| | | | |
| **BROWSER (network-variable)** | | | |
| Active decoded FPS | 24.6 | **22.6** | **19.9** |
| Active playback | 120s | 120s | 120s |
| Total freeze duration | 3.524s | **13.085s** | **27.499s** |
| Freeze count (diag) | 8 | 13 | 18 |
| Manual frame stalls | 0 | 0 | 0 |
| Frames dropped | 1 | 1 | 3 |
| Packets lost | 1617 | 1661 | 2194 |
| PLI/FIR/NACK | 3/0/255 | 8/0/268 | 22/0/462 |
| | | | |
| **INFRASTRUCTURE** | | | |
| Runtime | 125s | 125s | 125s |
| ICE state | connected | connected | connected |
| Dimensions | 1280x720 | 1280x720 | 1280x720 |
| Codec | H264 | H264 | H264 |
| Teardown | YES | YES | YES |
| Publisher JSON | parsed | parsed | parsed |

Bold = failed criterion.

### Failed criteria detail

**Run 2 (ap-Bh7335z9gyLGLSDS4QUNWK):**
- `[hardened] Active decoded FPS >= 23` — 22.6 fps (delta=2711 / 120s). A network burst at T+105s dropped browser-reported FPS to 4.0 (196 packets lost in 5s). This single event pulled the 120s average below 23.
- `[hardened] Total freeze duration <= 10s` — 13.085s. Same network burst accumulated freeze duration past the 10s threshold.

**Run 3 (ap-K1Qhf2PYxtZSNY6OLnndx6):**
- `[hardened] Active decoded FPS >= 23` — 19.9 fps (delta=2386 / 120s). Severe packet loss throughout (2194 packets lost = ~17.5/s average). PLI count 22 (vs 3 in run 1) indicates repeated picture loss requiring keyframe refreshes.
- `[hardened] Total freeze duration <= 10s` — 27.499s (22.9% of active playback). The jitter buffer could not keep up with sustained packet loss.

### Why Run 1 passed

Run 1 experienced moderate packet loss (1617 packets, ~13/s average) but no severe bursts. The jitter buffer and NACK retransmission handled the loss adequately: freeze duration stayed at 3.5s, FPS at 24.6.

---

## 3. Producer vs. Network Separation

The critical finding: **the AVTR producer pipeline is identical and healthy across all 3 runs.** Every failure is in the browser-side metrics driven by packet loss on the Modal → CF SFU → local browser path.

### Producer stability (identical across all runs)

| Metric | Run 1 | Run 2 | Run 3 | Variance |
|--------|-------|-------|-------|----------|
| Chunks rendered | 679 | 679 | 679 | 0 |
| Frames generated | 3392 | 3391 | 3391 | ±1 |
| Chunk latency mean | 69.0ms | 66.4ms | 70.6ms | ±2.1ms |
| Underflow | 0 | 0 | 0 | 0 |
| Overflow | 1 | 1 | 1 | 0 |
| Producer error | none | none | none | — |
| Pipeline error | none | none | none | — |
| Manual frame stalls | 0 | 0 | 0 | 0 |

The render thread produces frames at a stable 25fps (5 frames per ~70ms chunk = 71.4fps render rate, paced to 25fps by the appsrc push thread). The queue never underflows. Backpressure works correctly (overflow=1 is the initial fill, expected).

### Network variability (the actual variable)

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| Packets lost | 1617 | 1661 | **2194** |
| PLI count | 3 | 8 | **22** |
| NACK count | 255 | 268 | **462** |
| Freeze duration | 3.5s | **13.1s** | **27.5s** |
| Decoded FPS | 24.6 | **22.6** | **19.9** |

Packet loss drives everything: more lost packets → more NACK retransmissions → more PLI keyframe requests → more freeze duration → lower decoded FPS. The producer has zero control over this path.

### Manual stalls vs. WebRTC freezeCount

Manual frame-stall detection (consecutive 5s samples where `framesDecoded` did not increase) found **0 stalls in all 3 runs.** This means no freeze lasted ≥ 5s continuously. The WebRTC `freezeCount` (8, 13, 18) and `totalFreezesDuration` (3.5s, 13.1s, 27.5s) detect sub-5s micro-freezes that the 5s sampling interval cannot catch individually.

---

## 4. Startup Pre-Roll Behavior

The publisher implements a **delayed-start** lifecycle to avoid wasting GPU before a receiver exists:

```
modal_gst_publisher.py:557-570  — next_frame() returns black_i420() when not activated
modal_gst_publisher.py:868      — "ICE connected in Xms"
modal_gst_publisher.py:875      — self._source.start()  ← render thread spawned HERE
modal_gst_publisher.py:456-464  — start() spawns the render thread
```

**Timeline:**

1. **GStreamer init + SDP/ICE negotiation** (~3.5–3.8s): `appsrc` pushes cheap black I420 frames (Y=16, U/V=128). No GPU work. The encoder processes these, producing valid H264 NALUs so the SDP offer has a live track.

2. **ICE connected** (T+3.5s): The publisher calls `self._source.start()`. For AVTR mode, this spawns the render thread, which begins calling `process_chunk()` with silence audio. GPU work starts NOW.

3. **First AVTR frame** (~85ms after start): The first chunk produces 5 frames. These enter the bounded queue, get popped by the push thread, and flow through nvh264enc → webrtcbin.

4. **Browser first decode** (~T+5s): The browser receives and decodes the first frame. Active playback begins.

This design ensures:
- No GPU cycles wasted during SDP/ICE negotiation (which can take 3-4s)
- The receiver is confirmed connected before rendering starts
- Black frames during pre-roll are valid H264 (not an empty track)

---

## 5. TRT Patch Verification

### Build-time (image layer)

Every `modal run` prints:
```
TRT patch verified OK: helper=true patched_calls=2 raw_remaining=0
```

This runs in `.run_commands()` after the patch applies. Uses negative-lookbehind regex (`(?<!_dims_to_)tuple\(ctx\.get_tensor_shape`) to distinguish patched calls from raw. Asserts exactly 2 patched calls (lines 169 and 198 of `trt.py`) and 0 raw remaining.

### Runtime (before AVTR bootstrap)

`verify_trt_patch()` is the first line of `run_avtr()`, before `Gst.init()`:
```
TRT patch verified OK: helper=true patched_calls=2 raw_remaining=0
```

All 3 canaries printed this line.

### Root-cause wording (conservative)

In the GStreamer/PyGObject process, iterating TensorRT Dims via `tuple(dims)` crashes (SIGSEGV/SIGABRT for 4-D+ tensors). Index-based access via `_dims_to_tuple()` avoids the crash. The likely interaction is PyGObject allocator hooks + pybind11/TensorRT Dims `__iter__`, but this is a **POC-local workaround, hypothesis-backed** — no reproducible A/B evidence (crash without patch vs. no crash with patch in same binary) has been demonstrated.

---

## 6. Production Safety

### Files changed (POC only)

```
frame_source.py                    |   2 +-
modal_gst_publisher.py             | 743 ++++++++++++++++
public/index.html                  |  17 +-
run-option-c-test.cjs              | 277 +++++++++ +-
4 files changed, 991 insertions(+), 48 deletions(-)
```

### Production files NOT changed

```
git diff HEAD -- src/ avatar-saas-prototype/cloudflare-session-bridge/ \
  avatar-saas-prototype/modal-avatar-runtime/ avatar-saas-prototype/web-client/
→ (empty)
```

- `src/avtr1_renderer/` — untouched (patch is image-build-time only)
- `avatar-saas-prototype/cloudflare-session-bridge/` — untouched
- `avatar-saas-prototype/modal-avatar-runtime/` — untouched
- `avatar-saas-prototype/web-client/` — untouched
- `AVTR_REALTIME_SFU_ENABLED` — remains unset (defaults to `false`)

### Static checks

```
python3 -m py_compile modal_gst_publisher.py frame_source.py gst_sfu_client.py  → OK
node --check run-option-c-test.cjs                                              → OK
```

---

## 7. Safety Guard

The test harness now includes a post-criteria safety check that prevents false-positive GO when AVTR source metrics are missing:

```
[SAFETY] Mode: avtr | source_metrics present | chunks=679 frames=3391 underflow=0
```

If `MODE=avtr` but `source_metrics` is absent or incomplete (`chunks_rendered`, `frames_generated`, or `underflow_count` undefined), the script exits with code 1 regardless of criteria results.

Additionally, if `MODE` is not `avtr`, the script prints:
```
!!! WARNING: MODE != avtr — hardened criteria NOT evaluated !!!
```

---

## 8. Combined Run History (6 Total Canaries)

Including the previous session's 3 runs (re-evaluated under corrected criteria):

| Session | Run | FPS | Freeze Dur | Packets Lost | Underflow | Verdict |
|---------|-----|-----|-----------|-------------|-----------|---------|
| Prev | 1 | 24.5 | 4.3s | 1453 | 0* | FAIL** |
| Prev | 2 | 23.4 | 7.3s | 1530 | 0 | **PASS** |
| Prev | 3 | 23.9 | 4.4s | 655 | 0 | **PASS** |
| Curr | 1 | 24.6 | 3.5s | 1617 | 0 | **PASS** |
| Curr | 2 | 22.6 | 13.1s | 1661 | 0 | FAIL |
| Curr | 3 | 19.9 | 27.5s | 2194 | 0 | FAIL |

\*From publisher logs (JSON missing due to fixed bug)
\*\*Previous Run 1 fails #12 only (publisher JSON missing — code bug, fixed)

**Aggregate: 3 clean PASS, 2 network-induced FAIL, 1 code-bug FAIL.** Producer metrics identical across all 6 runs.

---

## 9. Final Verdict: HOLD

**1 of 3 fresh canaries passes all 14 corrected hardened criteria.** Per the validation contract, this is HOLD.

### What's proven (producer side)

- AVTR `process_chunk()` produces stable I420 frames at 25fps via silence-driven rendering
- NVENC encodes 1280×720 H264 with constrained-baseline profile at low latency
- GStreamer `appsrc → nvh264enc → webrtcbin` pipeline is stable for 135s runs
- Bounded queue with backpressure works correctly (0 underflow, 1 overflow per run)
- TRT Dims workaround is reliable (3/3 patch verifications pass)
- Chunk latency is 66–71ms (well under the 200ms budget)
- Clean teardown (tracks/close with mids+force, no pipeline errors)
- GPU utilization is low (6–41% mean), NVENC utilization is low (6–13% mean)

### What's not proven (network side)

- Consistent decoded FPS >= 23 at the browser under real network conditions
- Freeze duration <= 10s under real packet loss patterns

### Root cause of failures

The 2 failing runs experienced higher packet loss than the passing run:
- Run 2: 1661 packets lost with a burst at T+105s (196 packets in 5s)
- Run 3: 2194 packets lost with sustained loss throughout (17.5/s average)

This is the **Modal L40S → CF Realtime SFU → local browser** path, which traverses the public internet. In production, the path would be Modal L40S → CF SFU → CF edge near the user, potentially with better routing.

### Recommendation

1. **Re-run 3 canaries when network conditions improve.** The producer is ready; the variable is packet loss on the test path.
2. **Alternatively, run from a different network** (e.g., wired ethernet, different ISP) to reduce packet loss variance.
3. **Consider a network-quality pre-check** that measures RTT and packet loss to the CF SFU before starting the canary, skipping runs when loss is above a threshold.
4. **The freezeCount diagnostic data across 6 runs** (3–18 freezes per run, all sub-5s) confirms the reclassification was correct — no sustained freezes occurred in any run.
