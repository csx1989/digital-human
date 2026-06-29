# Option C — NVENC-backed GStreamer WebRTC Publisher POC

**Objective:** Determine whether a GStreamer `webrtcbin + nvh264enc` publisher on Modal L40S can feed Cloudflare Realtime SFU with GPU-encoded H264, replacing the current aiortc/PyAV CPU encoder.

**Scope:** Research + isolated POC only. No production Worker changes. No web-client integration. fMP4 remains the production floor.

**Modal app:** `avtr-option-c-gst-publisher`

## Architecture

```
Modal L40S (GStreamer + nvh264enc + webrtcbin)
       │
       │ HTTPS SDP exchange
       ▼
Cloudflare Realtime SFU
       │
       │ WebRTC
       ▼
Browser subscriber (via Option-C Worker proxy + Playwright)
```

## Directory Contents

| File | Purpose |
|------|---------|
| `modal_gst_publisher.py` | Modal GPU publisher: synthetic I420 mode and AVTR frame handoff mode |
| `local_gst_probe.py` | Local/dev-machine SDP compatibility probe (macOS uses `x264enc`) |
| `gst_sfu_client.py` | CF Realtime Sessions/Tracks API client |
| `frame_source.py` | Synthetic I420 frame source + AVTR adapter |
| `worker.ts` | Subscriber-side Worker proxy (copied/adapted from Phase C) |
| `public/index.html` | Browser subscriber page |
| `run-option-c-test.cjs` | Playwright automation harness |
| `requirements.txt` | Python dependencies |
| `.env.example` / `.dev.vars.example` | Secret templates |

## Hard Gates

1. **Gate A:** `gst-inspect-1.0 nvh264enc` passes **and** an actual encode smoke test succeeds on Modal.
2. **Gate B:** Cloudflare Realtime accepts the GStreamer H264 offer SDP. If rejected twice, stop and document exact SDP + response.
3. **Gate C:** Synthetic 1280×720 @ 25fps NVENC stream passes a 120s browser subscriber test.
4. **Gate D:** Browser stats confirm the negotiated codec is H264, not a silent VP8 fallback.
5. **Gate E:** Modal metrics capture CPU%, GPU utilization, and **NVENC encoder utilization**.
6. **Gate F:** Only after Gate C passes, attempt AVTR frame handoff.

## Local Development (Phase 1)

On macOS `nvh264enc` is unavailable; the local probe uses `x264enc` only to verify SDP flow. The real gate is Modal L40S.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# macOS GStreamer from Homebrew:
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav pygobject3

source .env
python3 local_gst_probe.py --duration 30
```

## Modal Synthetic Test (Phase 2)

```bash
modal run modal_gst_publisher.py --mode synthetic --duration 120
```

Capture `PUBLISHER_READY <sessionId> <trackName>`, then run the Playwright harness.

## Playwright Test Harness

```bash
npm install
npx playwright install chromium

# Terminal 1
npx wrangler dev --port 8790

# Terminal 2
POC_ACCESS_TOKEN=<token> node run-option-c-test.cjs
```

## AVTR Handoff (Phase 3)

Only after synthetic 120s passes:

```bash
modal run modal_gst_publisher.py --mode avtr --duration 120 --avatar-id arnold
```

## Notes

- This POC intentionally does **not** touch `cloudflare-session-bridge/`, `web-client/`, or production transport defaults.
- `AVTR_REALTIME_SFU_ENABLED` must remain `false` at the end.
- All secrets live in `.env` / `.dev.vars` and are gitignored.
