# Avatar SaaS Prototype

Isolated prototype workspace for bridging the Cloudflare voice agent stack to the
AVTR-1 avatar runtime.

## Status

**Prototype planning + Modal feasibility completed.** The workspace now contains
the initial scaffold, runtime analysis, Modal GPU smoke/benchmark tooling, and a
documented roadmap for the next implementation phases.

The chosen runtime baseline for the first SaaS prototype is:

- Modal **L4** GPU
- default TRT engines
- GPU memory snapshotting for cold-start reduction
- local lifecycle validation before Cloudflare/voice integration

## Purpose

The long-term goal is a production-grade avatar SaaS platform where:

- A browser client captures microphone audio and renders an animated avatar.
- A Cloudflare session bridge orchestrates the real-time session, routing audio
  through the existing voice agent pipeline (VAD, Smart Turn, STT, LLM, TTS).
- A Modal-hosted GPU runtime runs the AVTR-1 renderer and streams avatar frames
  back to the browser.

This workspace will evolve incrementally. Each component starts as a skeleton and
gets fleshed out phase by phase.

## Isolation Guarantee

The following production code is **not modified** by anything in this workspace:

- `src/avaturn_live_streamer/` — existing live streaming orchestration
- `src/avtr1_renderer/` — existing GPU renderer
- `scripts/` — build and run scripts
- Any Cloudflare workers (`any-toy-agent`, `any-toy-vad`, `smart-turn-worker`)

All prototype code lives exclusively under `avatar-saas-prototype/`.

## Directory Layout

```
avatar-saas-prototype/
  README.md                     This file
  docs/
    architecture.md             Target architecture and data flow
    roadmap.md                  Implementation phases
    protocol.md                 First-draft event protocol
  modal-avatar-runtime/
    README.md                   Modal app description
    modal_app.py                Skeleton: session lifecycle placeholders
  cloudflare-session-bridge/
    README.md                   Bridge description
    wrangler.jsonc              Minimal Worker config (no secrets)
    package.json                Minimal package definition
    tsconfig.json               TypeScript config
    src/
      index.ts                  Skeleton: /health, /sessions, WS upgrade
  web-client/
    README.md                   Client responsibilities
    package.json                Minimal package definition
    src/
      main.ts                   Placeholder for browser responsibilities
```

## Next Steps

See `docs/roadmap.md` for the phased implementation plan.

The immediate next phase is **SaaS lifecycle proof**:

1. Click Connect in a local web app.
2. Wake the snapshotted Modal avatar runtime.
3. Receive an `avatar_ready` message.
4. Show an avatar frame or intentionally documented placeholder.
5. Click Stop and cleanly release the Modal session.

Cloudflare bridge adaptation is intentionally a separate later phase. The old
toy-agent project is treated as a reference implementation only; this repository
will contain the new avatar-specific bridge/agent implementation.
