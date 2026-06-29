# AVTR-1 Phase 4A: Local Lifecycle Server

Browser Connect → Modal Snapshot Runtime → Avatar Ready → Stop.

## Prerequisites

1. The Modal lifecycle app must be deployed first:

```bash
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py
```

2. Python 3.11+ with the `modal` package installed.

## Running

```bash
python3 avatar-saas-prototype/local-lifecycle-app/server.py
```

Then open: http://localhost:8765

## Usage

1. Click **Connect** — triggers Modal snapshot restore + session start.
   - First call may take ~5s (snapshot restore, best case) to 15s+ (restore
     after scale-down, including Modal scheduling overhead).
   - First deploy may trigger a cold start (~50-100s) while fresh GPU snapshots
     are created.
   - Browser should show "Avatar ready".
   - A debug frame will be fetched automatically.
2. Click **Debug Frame** — renders a silent chunk and shows a JPEG preview.
3. Click **Stop** — ends the Modal session and resets the UI.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /api/health` | GET | Check Modal connectivity and pipeline status |
| `POST /api/connect` | POST | Start avatar session (calls `start_session`) |
| `POST /api/debug-frame` | POST | Get debug frame JPEG (calls `render_debug_frame`) |
| `POST /api/stop` | POST | End session (calls `end_session`) |

## Architecture

```
Browser
  ↓ HTTP JSON
Local server (stdlib http.server, port 8765)
  ↓ Modal Python SDK
  ↓ modal.Cls.from_name("avtr1-lifecycle", "AvatarLifecycleL4")
Modal L4 GPU snapshot runtime
  ↓ Pipeline.from_artifacts() + process_chunk()
AVTR-1 renderer (TensorRT engines on L4)
```

## Environment Variables

- `AVTR1_LIFECYCLE_PORT`: Server port (default: 8765)

## Known Limitations

- Single session only (not multi-tenant).
- No audio streaming — silent chunks only.
- No real-time video streaming.
- Debug frame is a single JPEG from a silent render, not live video.
- Browser must not be left idle too long (Modal container may scale down).
- GPU memory snapshots are an Alpha Modal feature. Snapshot restore may fail
  (e.g., after redeploy or platform changes), triggering a Modal-managed fallback
  to full cold start (~50-100s). Fresh snapshots are needed after each deploy.

## Stopping & Cleanup

- The browser **Stop** button ends the logical avatar session only — it does **not** stop the deployed Modal app.
- After Stop, Modal may keep the container warm briefly (default idle timeout). With `scaledown_window=10` on the deployed app, the container will shut down after 10 seconds of inactivity.
- To manually stop the deployed Modal app:

```bash
modal app stop avtr1-lifecycle
```

- To deploy again later:

```bash
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py
```
