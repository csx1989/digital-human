# Phase A — Echo Test

Browser publishes a canvas-generated video track to Cloudflare Realtime SFU, then subscribes to that same track in the same browser. This proves the native WebRTC video path through the SFU without camera permission flakiness.

## Prerequisites

- Node.js 18+
- A Cloudflare account with Realtime enabled
- `wrangler` (installed via `npm install` below)

## Step 1 — Create a Cloudflare Realtime App

1. Go to the [Cloudflare dashboard → Calls/Realtime](https://dash.cloudflare.com/?to=/:account/calls)
2. Click **Create Application** → choose **Realtime**
3. Give it a name (e.g., `avtr-poc`)
4. Copy the **App ID** and **App Secret** — you need both below

If you already have a Realtime App, reuse it. Sessions are scoped to a single App ID.

## Step 2 — Configure Credentials

### Local development

```bash
cd avatar-saas-prototype/realtime-sfu-poc/phase-a-echo-test
cp .dev.vars.example .dev.vars
```

Edit `.dev.vars`:
```
CALLS_APP_SECRET=<your App Secret from dashboard>
POC_ACCESS_TOKEN=<any random string, e.g. openssl rand -hex 24>
```

Edit `wrangler.toml` → set `CALLS_APP_ID`:
```toml
[vars]
CALLS_APP_ID = "<your App ID from dashboard>"
```

### Deployed (later, if needed)

```bash
npx wrangler secret put CALLS_APP_SECRET    # paste App Secret
npx wrangler secret put POC_ACCESS_TOKEN    # paste your chosen token
# CALLS_APP_ID stays in wrangler.toml [vars] — it's not secret
```

## Step 3 — Install Dependencies

```bash
npm install
```

This creates a **separate** `node_modules/` in this directory. It does NOT touch the `cloudflare-session-bridge/node_modules/` dirty state.

## Step 4 — Type-check

```bash
npm run typecheck
```

## Step 5 — Run Locally

```bash
npm run dev
```

Wrangler serves the Worker at `http://localhost:8787`.

1. Open `http://localhost:8787` in your browser
2. Paste your `POC_ACCESS_TOKEN` in the input field
3. Click **Check Config** — should show "SFU configured"
4. Click **Start Echo Test**
5. The left panel shows the animated canvas pattern (local)
6. The right panel shows the same pattern received from the SFU (remote)
7. Watch the diagnostics panel for ICE states, frame stats, uptime
8. Let it run for **2+ minutes** to confirm stability
9. Click **Disconnect** for clean teardown

### Camera mode (optional)

Append `?camera=1` to the URL or check "Camera mode" before starting. Uses `getUserMedia` instead of canvas.

## Automated Echo Test (Playwright)

The `run-echo-test.cjs` script automates the full 2-minute echo test in headless Chromium, including diagnostics capture and teardown verification.

### Prerequisites

```bash
npm install            # installs playwright + wrangler + typescript
npx playwright install chromium
```

### Running

Start the dev server in one terminal:

```bash
npm run dev
```

Run the automated test in another terminal (token must match `.dev.vars`):

```bash
POC_ACCESS_TOKEN=<your-token> node run-echo-test.cjs
```

Optional — specify a different URL:

```bash
POC_URL=http://localhost:8787 POC_ACCESS_TOKEN=<your-token> node run-echo-test.cjs
```

The script prints per-criterion PASS/FAIL and exits 0 on success, 1 on failure.

## Success Criteria

| Criterion | How to verify |
|-----------|---------------|
| Publisher + subscriber sessions created | Diagnostics shows `session.publisher` and `session.subscriber` |
| Canvas track published to SFU | Publisher badge shows "connected" |
| Remote track received | Remote `<video>` shows the animated pattern |
| Video plays 2+ minutes | `remote.uptime` in diagnostics exceeds 120s |
| Clean teardown | Click Disconnect — no errors in console |
| No production files modified | `git status` shows only `realtime-sfu-poc/` changes |

## API Endpoints (Worker)

All `/api/*` routes require `x-poc-token` header matching `POC_ACCESS_TOKEN`.

| Method | Path | SFU API Proxied |
|--------|------|-----------------|
| GET | `/api/health` | — (returns `{configured: boolean}`) |
| POST | `/api/sessions/new` | `POST /apps/{id}/sessions/new` |
| POST | `/api/sessions/:id/tracks/new` | `POST /apps/{id}/sessions/:id/tracks/new` |
| PUT | `/api/sessions/:id/renegotiate` | `PUT /apps/{id}/sessions/:id/renegotiate` |
| PUT | `/api/sessions/:id/tracks/close` | `PUT /apps/{id}/sessions/:id/tracks/close` |

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "unauthorized" on all API calls | Wrong `POC_ACCESS_TOKEN` | Check token matches `.dev.vars` |
| "sfu_not_configured" | `CALLS_APP_ID` or `CALLS_APP_SECRET` empty | Fill in `wrangler.toml` + `.dev.vars` |
| Publisher ICE timeout | Firewall blocking UDP | Try from a different network |
| Remote video black | SFU has no track to forward | Check publisher connected first |
| `framesDecoded` stays 0 | Codec mismatch / no media flowing | Check browser console for errors |
| Canvas not animating | JS error | Check console — must be same-origin |

## Security Notes

- `CALLS_APP_SECRET` is **never** sent to the browser
- `CALLS_APP_ID` is in `wrangler.toml [vars]` — it's not secret (session/track IDs are the sensitive part)
- `POC_ACCESS_TOKEN` gates all API access — required before any deploy
- The Worker uses a **strict allowlist** of SFU endpoints (no generic path forwarding)
- `.dev.vars` is gitignored
