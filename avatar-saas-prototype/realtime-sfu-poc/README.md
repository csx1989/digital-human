# Realtime SFU POC

Isolated proof-of-concept for publishing and subscribing native WebRTC video tracks through **Cloudflare Realtime SFU**. This is the target media transport for the AVTR avatar pipeline (replacing the current SSE fMP4 path in a future phase).

> **Isolation:** All files here are new. No production Worker, DO, Modal, or web-client code is modified.

## Architecture

```
Publisher (browser canvas / Modal headless)
  │ WebRTC PeerConnection (SDP offer)
  ▼
Worker (stateless auth proxy, holds CALLS_APP_SECRET)
  │ HTTPS → Cloudflare Realtime API
  ▼
Cloudflare Realtime SFU (global anycast)
  │ WebRTC PeerConnection (SDP offer)
  ▼
Subscriber (browser <video>)
```

The Worker never exposes `CALLS_APP_SECRET` to the browser. All `/api/*` calls are gated by `POC_ACCESS_TOKEN`.

## Phases

| Phase | Goal | Status |
|-------|------|--------|
| **A — Echo Test** | Browser publishes canvas video track to SFU, subscribes to it in same browser. Proves native WebRTC video path. | **Active** |
| B — Local aiortc Publisher | Python test-pattern publisher via aiortc → SFU → browser subscriber. | Planned |
| C — Modal Publisher | Modal container publishes raw I420 frames as WebRTC track. | Planned |
| D — Audio + Sync | Add audio track, A/V sync. | Planned |
| E — Integration | Bridge behind `AVTR_REALTIME_SFU_ENABLED` feature flag. | Planned |
| F — SDK Abstraction | Transport-agnostic SDK with fallback. | Planned |

## Directory

```
realtime-sfu-poc/
├── README.md            ← you are here
├── .gitignore
└── phase-a-echo-test/   ← Phase A: browser echo test
    ├── README.md           Setup + run instructions
    ├── worker.ts           Stateless SFU auth proxy
    ├── wrangler.toml       Worker config (no secrets)
    ├── package.json
    ├── tsconfig.json
    ├── .dev.vars.example   Template for local secrets
    └── public/
        └── index.html      Browser test page
```

See `phase-a-echo-test/README.md` for setup and run instructions.
