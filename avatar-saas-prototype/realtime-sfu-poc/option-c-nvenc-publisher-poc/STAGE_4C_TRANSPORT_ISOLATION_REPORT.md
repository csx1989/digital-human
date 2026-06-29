# Stage 4C — Transport Isolation Debugging Report

**Date:** 2026-06-19
**Verdict:** **GO with profile: 1280x720 H264 NVENC, 2 Mbps, gop-size=50, preset=low-latency-hq**

3/3 canaries pass all 14 hardened criteria at 2 Mbps. The uncapped auto/VBR nvh264enc bitrate (~10–11 Mbps, ≈1,100 RTP packets/s) created excessive RTP packet pressure and caused visible freezes (1.0–1.2% observed loss, 25–41 freezes, 8.4–25.7s freeze duration). Capping to `bitrate=2000` — note `nvh264enc`'s `bitrate` property is in **kbit/sec**, so this is **2 Mbps** — reduced packet pressure ≈4–5×; observed loss fell to 0.0% (0–1 packets lost over 120s) and produced 3/3 passing canaries.

> **Note on loss framing (corrected).** An earlier draft of this report reasoned that "the same 1% loss rate at 2 Mbps yields only ~3 lost packets/s (~1 per 5s)." That arithmetic does not hold (250 pkt/s × 5s = 1,250 packets; 1% would be ~12.5 lost/5s, not 1), and more importantly it contradicts the observed data: at 2 Mbps the measured loss was **0.0%**, not 1%. The defensible statement is that capping bitrate reduced packet pressure and the **observed** loss dropped to ~0 — not that the loss rate was held constant while only the absolute count fell.

---

## 1. Root Cause Diagnosis

### What was observed

| Metric | Baseline (auto ~11 Mbps) | 2 Mbps Profile |
|--------|--------------------------|----------------|
| Avg received bitrate | 9,572–11,222 kbps | 2,021–2,026 kbps |
| Total RTP packets (120s) | 115K–135K | 30K–31K |
| Packets lost | 1,415–1,679 | 0–1 |
| Loss percentage | 1.0–1.2% | 0.0% |
| Freeze count | 25–41 | 0–1 |
| Freeze duration | 8.4–25.7s | 0.0–2.3s |
| Decoded FPS | 20.7–25.0 | 24.5–25.0 |
| NACK count | 284–393 | 0–18 |
| Manual frame stalls | 0 | 0 |

### Mechanism

At ~10–11 Mbps (auto/VBR) the publisher emits ≈1,100 RTP packets/s. Across the 120s canary window the browser observed **1,415–1,679 packets lost (1.0–1.2%)** with 25–41 freezes accumulating to 8.4–25.7s. This packet pressure produced visible micro-freezes that the WebRTC jitter buffer and NACK retransmission could not fully absorb within the playback window.

At `bitrate=2000` (2 Mbps; `nvh264enc` `bitrate` is in **kbit/sec**) the publisher emits ≈250 packets/s. **Observed loss dropped to 0.0%** — only 0–1 packets were lost over the full 120s window — and freeze duration fell to 0.0–2.3s.

> **What this does and does not prove.** The data shows a strong correlation between higher packet pressure (more packets on the wire) and higher *observed* loss + freezes at the browser. It does **not** prove the underlying loss rate of the path is constant and merely scales with packet count; the observed 0.0% loss at 2 Mbps is an empirical result, not a derivation from a fixed 1% rate. The safe, evidence-backed claim is: **capping bitrate reduced packet pressure, and the observed loss fell to ~0 — yielding 3/3 passing canaries.**

### What this is NOT

- **Not AVTR rendering:** Producer metrics are identical across all runs (679–684 chunks, 66–71ms chunk latency, 0 underflow, 0 errors). The rendering pipeline appears healthy regardless of bitrate.
- **Network bandwidth:** Available incoming bandwidth was 4.7–19.5 Mbps. The 2 Mbps profile fits comfortably inside that observed envelope; the uncapped ~10–11 Mbps profile sometimes approached or exceeded the lower observed available-bandwidth range, which supports using an explicit bitrate cap.
- **SFU routing not implicated (but not definitively ruled out):** The observed candidate pair was consistently local=srflx remote=host (STUN reflexive → CF SFU host), RTT 13–22ms, with no candidate-pair changes, ICE restarts, or TURN relay across runs. This means candidate-pair instability / SFU routing was **not indicated by the observed stats**. It is not proof that SFU routing is impossible as a loss contributor — `packetsLost` is measured at the browser and cannot localize whether loss occurs on Modal→SFU, SFU→browser, or the local ISP/Wi-Fi. The transport stats were consistent across runs, which points to packet count as the variable rather than path-quality changes.

---

## 2. Controlled Canary Results

### Baseline: 1280x720, auto bitrate (~11 Mbps) — 2/3 PASS

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| **App ID** | ap-QT6T... | ap-xM6s... | ap-nDf1... |
| **Verdict** | **PASS** 14/14 | **FAIL** 12/14 | **PASS** 14/14 |
| Active decoded FPS | 25.0 | **20.7** | 24.3 |
| Total freeze duration | 8.4s | **25.7s** | 8.4s |
| Freeze count | 41 | 29 | 25 |
| Packets lost | 1,415 | 1,479 | 1,646 |
| Loss % | 1.0% | 1.1% | 1.2% |
| Avg bitrate rcvd | 11,222 kbps | 9,572 kbps | 11,213 kbps |
| RTT | 14ms | 21ms | 14ms |
| Avail in BW | 19,302 kbps | 19,391 kbps | 19,513 kbps |
| Underflow | 0 | 0 | 0 |
| Manual stalls | 0 | 0 | 0 |

Run 2 failed on decoded FPS (20.7 < 23) and freeze duration (25.7s > 10s). Runs 1 and 3 passed but with thin margins (freeze duration 8.4s, close to the 10s threshold).

### Lower Bitrate: 1280x720, 2 Mbps — 3/3 PASS

| Metric | Run 1 | Run 2 | Run 3 |
|--------|-------|-------|-------|
| **App ID** | ap-WPKx... | ap-Xmew... | ap-n86Y... |
| **Verdict** | **PASS** 14/14 | **PASS** 14/14 | **PASS** 14/14 |
| Active decoded FPS | 24.5 | 25.0 | 24.9 |
| Total freeze duration | 2.3s | **0.0s** | 0.4s |
| Freeze count | 1 | **0** | 1 |
| Packets lost | **1** | **0** | **1** |
| Loss % | 0.0% | 0.0% | 0.0% |
| Avg bitrate rcvd | 2,026 kbps | 2,021 kbps | 2,022 kbps |
| RTT | 17ms | 14ms | 14ms |
| Avail in BW | 4,886 kbps | 4,826 kbps | 4,741 kbps |
| NACK count | 18 | **0** | 4 |
| Underflow | 0 | 0 | 0 |
| Manual stalls | 0 | 0 | 0 |

Run 2 is a perfect run: zero packet loss, zero freezes, zero NACKs, zero frames dropped. All three runs pass with enormous margins.

### Lower Resolution: 960x540 — Not Run

The lower bitrate profile (2 Mbps at 1280x720) already passes 3/3 with wide margins. The root cause is identified as bitrate/packet-count, not resolution. Resolution tests can be run as additional diagnostic if needed, but are not required for the GO verdict.

---

## 3. Transport Diagnosis

### Per-interval analysis methodology

The test harness computes per-5s-interval metrics from browser `getStats()`:
- Packet loss rate per interval (lost / (received + lost) × 100)
- Received bitrate per interval (bytesReceived delta × 8 / 5s)
- Freeze count delta per interval
- NACK count delta per interval
- Freeze-loss correlation (intervals with both freeze increment AND loss > 10%)

### Baseline loss pattern (11 Mbps)

- **Pattern:** Sustained low-rate loss (1.0–1.2% per interval, no bursts)
- **Loss timeline example (Run 3):** `5s:1%* 10s:1%* 15s:1%* 20s:1%* 25s:1% 30s:1%* 35s:1% ...`
- Every interval has 1–2% loss (60–130 packets). The `*` marks freeze increments.
- Freeze-loss correlation at 10% threshold: 0 — because loss never exceeds 10% in any single interval. However, freezes accumulate from the steady ~1% loss.
- **Key insight:** At ~11 Mbps the browser observed sustained 1.0–1.2% loss (1,415–1,679 packets lost over 120s). That absolute packet-loss volume, spread across the run, was enough to cause micro-freezes despite the *percentage* appearing "low."

### Lower bitrate loss pattern (2 Mbps)

- **Pattern:** Near-zero loss (0.0% across all intervals)
- **Loss timeline example (Run 2):** `5s:0% 10s:0% 15s:0% 20s:0% ... 120s:0%` — zero loss in every single interval
- NACK count: 0–18 (vs 284–393 at baseline)
- **Key insight:** At 2 Mbps the browser observed **0.0% loss (0–1 packets lost total over 120s)**, not the 1.0–1.2% seen at ~11 Mbps. Packet pressure dropped ≈4–5× and observed loss fell to ~0, which the jitter buffer handled trivially.

### ICE / Candidate Pair

Consistent across all 7 runs:
- **Local candidate:** srflx (server-reflexive, via STUN) — occasionally prflx (peer-reflexive)
- **Remote candidate:** host (direct CF SFU host)
- **Protocol:** UDP
- **RTT:** 13–22ms (median ~14ms)
- **Available incoming bitrate:** 4,700–19,500 kbps (plenty of headroom for 2 Mbps; adequate for 11 Mbps)

The candidate pair data confirms the browser connects directly to the CF SFU via STUN. No TURN relay, no candidate-pair failures, no ICE restarts.

### Freeze-loss correlation

At the 10% per-interval loss threshold: **0 intervals with correlated freeze+loss** in all 7 runs. This means no single 5s interval had both a freeze increment AND > 10% loss. The freezes are caused by accumulated low-rate loss (1%), not by burst loss events. This is consistent with the packet-count hypothesis: the jitter buffer handles individual losses but degrades over time as losses accumulate.

---

## 4. Encoder Settings Comparison

| Setting | Baseline | GO Profile |
|---------|----------|------------|
| Resolution | 1280x720 | 1280x720 |
| FPS | 25 | 25 |
| Encoder | nvh264enc | nvh264enc |
| Bitrate | auto (VBR, ~11 Mbps) | **2,000 kbps (2 Mbps)** |
| GOP size | 50 | 50 |
| Preset | low-latency-hq | low-latency-hq |
| Profile | constrained-baseline | constrained-baseline |
| Level | 3.1 | 3.1 |
| Keyframe interval | config-interval=-1 (every frame for SPS/PPS) | config-interval=-1 |

The only change is `bitrate=2000` added to the nvh264enc properties. Everything else remains identical. Note: `nvh264enc`'s `bitrate` property is expressed in **kbit/sec**, so `bitrate=2000` = **2 Mbps**.

### Why 2 Mbps is production-plausible for 720p25

- Google Meet 720p: 1–2 Mbps
- Zoom 720p: 1.5–3 Mbps  
- Twitch 720p30: 3–4 Mbps (but for motion-heavy gaming content)
- AVTR silence-driven avatar rendering has low motion entropy (static background, subtle facial movement), so 2 Mbps is more than sufficient for visual quality.

---

## 5. Startup Pre-Roll Behavior

```
modal_gst_publisher.py:557-570  — next_frame() returns black_i420() when not activated
modal_gst_publisher.py:868      — "ICE connected in Xms"
modal_gst_publisher.py:875      — self._source.start()  ← render thread spawned HERE
modal_gst_publisher.py:456-464  — start() spawns the render thread
```

1. **GStreamer init + SDP/ICE** (~3.5–4.1s): appsrc pushes black I420 frames. No GPU work.
2. **ICE connected** (T+3.5s): `self._source.start()` spawns the render thread. GPU work begins.
3. **First AVTR frame** (~70–85ms after start): First chunk produces 5 frames.
4. **Browser first decode** (T+5s): Active playback begins.

---

## 6. TRT Patch Verification

Every run printed both build-time and runtime verification:
```
TRT patch verified OK: helper=true patched_calls=2 raw_remaining=0
```

Root-cause wording: In the GStreamer/PyGObject process, iterating TensorRT Dims via `tuple(dims)` crashes; index-based access via `_dims_to_tuple()` avoids the crash. **Hypothesis-backed workaround** — no A/B evidence in the same binary.

---

## 7. Producer Health (identical across all 7 runs)

| Metric | Range |
|--------|-------|
| Chunks rendered | 679–684 |
| Frames generated | 3,392–3,418 |
| Chunk latency mean | 66–73ms |
| Chunk latency max | 81–209ms |
| Underflow | 0 (all runs) |
| Overflow | 0–1 (all runs) |
| Producer error | none (all runs) |
| Pipeline error | none (all runs) |
| Manual frame stalls | 0 (all runs) |

The AVTR producer pipeline appears healthy and stable regardless of encoder bitrate. The render thread, bounded queue, and backpressure mechanism work correctly in all configurations.

---

## 8. Production Safety

### Files changed (POC only)

```
frame_source.py                    |   2 +-
modal_gst_publisher.py             | 788 +++++++++-
public/index.html                  |  50 +-
run-option-c-test.cjs              | 372 +++++-
4 files changed, 1157 insertions(+), 55 deletions(-)
```

### Production files NOT changed

```
git diff HEAD -- src/ avatar-saas-prototype/cloudflare-session-bridge/ \
  avatar-saas-prototype/modal-avatar-runtime/ avatar-saas-prototype/web-client/
→ (empty)
```

- `AVTR_REALTIME_SFU_ENABLED` — remains unset (defaults to `false`)

---

## 9. Acceptance Criteria Met

The GO profile (1280x720 H264 NVENC, 2 Mbps) passes 3/3 canaries on all hardened criteria:

| Criterion | Run 1 | Run 2 | Run 3 | Threshold |
|-----------|-------|-------|-------|-----------|
| Runtime | 125s | 125s | 125s | >= 120s |
| ICE connected | connected | connected | connected | yes |
| Remote video playing | true | true | true | true |
| Dimensions | 1280x720 | 1280x720 | 1280x720 | 1280x720 |
| Active decoded FPS | 24.5 | 25.0 | 24.9 | >= 23 |
| Active playback | 120s | 120s | 120s | >= 100s |
| Total freeze duration | 2.3s | 0.0s | 0.4s | <= 10s |
| Manual frame stalls | 0 | 0 | 0 | = 0 |
| CUDA/producer errors | none | none | none | none |
| Pipeline error | none | none | none | none |
| Underflow | 0 | 0 | 0 | <= 5 |
| Publisher JSON | parsed | parsed | parsed | captured |
| Codec | H264 | H264 | H264 | H264 |
| Teardown | YES | YES | YES | succeeds |

### `freezeCount` is diagnostic, not a blocker

`freezeCount` is reported as a diagnostic signal but is **not** a pass/fail blocker. Per the corrected hardened-criteria set (see `STAGE_4C_VALIDATION_HARDENING_REPORT.md` §1), `freezeCount` counts sub-second inter-frame delays (MDN: `>= max(3× average, average + 150ms)`) — micro-freezes the jitter buffer absorbs — whereas **manual frame stalls** (consecutive 5s samples with no `framesDecoded` increase) capture real user-visible stalls. Manual stalls were **0 across all runs**.

The **active blockers** are: runtime, ICE connected, remote video playing, dimensions, **active decoded FPS**, active playback, **total freeze duration**, **manual frame stalls**, **producer/CUDA errors**, **pipeline error**, **underflow**, publisher JSON captured, **negotiated codec**, and **teardown**. `freezeCount` remains in diagnostic output only.

---

## 10. Final Verdict

**GO with profile: 1280x720 H264 NVENC, 2 Mbps, gop-size=50, preset=low-latency-hq, constrained-baseline, level 3.1.**

### Summary of findings

1. **Root cause of baseline failures:** The uncapped auto/VBR nvh264enc bitrate (~10–11 Mbps, ≈1,100 RTP packets/s) created excessive packet pressure. The browser observed 1.0–1.2% loss (1,415–1,679 packets lost over 120s) and freezes accumulated past the 10s threshold.

2. **Root cause of 2 Mbps success:** Capping to `bitrate=2000` (= **2 Mbps**; `nvh264enc` `bitrate` is in kbit/sec) cut packet pressure ≈4–5×. **Observed loss fell to 0.0%** (0–1 packets lost over 120s); freeze duration dropped to 0.0–2.3s and all 3 canaries passed. (Correction: an earlier draft claimed "the same 1% loss rate yields only ~3 lost packets/5s at 2 Mbps" — that arithmetic does not hold and contradicts the observed 0.0% loss. The defensible statement is that packet pressure dropped and **observed** loss fell to ~0.)

3. **Producer health:** The AVTR → NVENC pipeline appears healthy in all configurations. The TRT Dims workaround is stable. The render thread, bounded queue, and backpressure mechanism work correctly.

4. **Transport path:** Consistent srflx→host UDP candidate pair, 13–22ms RTT, 4.7–19.5 Mbps available bandwidth; candidate-pair instability was **not indicated** by the observed stats, so SFU routing is not implicated by this data (though not definitively ruled out — receiver-side `packetsLost` cannot localize Modal→SFU vs SFU→browser vs local ISP loss).

5. **Production action:** Set `bitrate=2000` on nvh264enc in the production pipeline string when implementing the SFU integration. This is a single-property change with no structural impact.
