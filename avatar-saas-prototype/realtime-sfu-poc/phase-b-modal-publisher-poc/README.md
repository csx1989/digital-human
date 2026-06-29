# Phase B — Modal WebRTC Publisher POC

## Overview

Proves that a server-side WebRTC publisher (Python aiortc) can publish a synthetic
25fps video track into Cloudflare Realtime SFU, and a browser subscriber can receive it.

**B1**: Local Python publisher (this directory)
**B2**: Modal-hosted publisher (added after B1 passes)

## Architecture

```
Python Publisher (aiortc)              Browser Subscriber (Playwright)
       |                                         |
       | Direct HTTPS → SFU API                  | Via Worker proxy
       | (CALLS_APP_SECRET from env)             | (Worker holds CALLS_APP_SECRET)
       v                                         v
┌──────────────────────────────────────────────────────┐
│            Cloudflare Realtime SFU                     │
│   Publisher Session ←──track──→ Subscriber Session     │
└──────────────────────────────────────────────────────┘
```

## Prerequisites

### Python 3.11+

```bash
python3.11 --version  # verify available
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Node.js

```bash
npm install
npx playwright install chromium
```

## Configuration

### .dev.vars (Worker secrets — gitignored)

```
CALLS_APP_ID=your-app-id
CALLS_APP_SECRET=your-app-secret
POC_ACCESS_TOKEN=your-poc-token
```

### .env (Python publisher secrets — gitignored)

```
CALLS_APP_ID=your-app-id
CALLS_APP_SECRET=your-app-secret
```

Copy `.dev.vars.example` and `.env.example` as templates.

## Running B1

### Automated test (recommended)

```bash
npm run dev   # start Worker on port 8788 (in one terminal)

# In another terminal:
POC_ACCESS_TOKEN=<token> node run-phase-b-test.cjs
```

The test runner will:
1. Verify Worker is running
2. Start the Python publisher
3. Wait for PUBLISHER_READY signal
4. Open subscriber page in headless Chromium
5. Run for 2+ minutes
6. Tear down both sides
7. Report pass/fail criteria

### Manual test

Terminal 1 — Worker:
```bash
npm run dev
```

Terminal 2 — Publisher:
```bash
source .env
source .venv/bin/activate
python3 publisher.py --duration 300
```

Note the `PUBLISHER_READY <sessionId> <trackName>` output.

Browser — Subscriber:
```
http://localhost:8788?token=<POC_TOKEN>&pubSession=<sessionId>&trackName=<trackName>
```

## Pass Criteria (12 checks)

1. Runtime >= 120s
2. Subscriber ICE connected/completed
3. Remote video playing
4. Remote dimensions valid (640x480)
5. framesDecoded delta >= 500
6. Final framesDecoded >= 1000
7. Final FPS >= 20
8. No page errors
9. Publisher ICE connected (from logs)
10. Subscriber teardown complete
11. Subscriber close used mids + force:true
12. Publisher close used mids + force:true
