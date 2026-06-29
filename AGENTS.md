# AGENTS.md — Avtr-1 Project Reference

All agents working on this project must read this file first.

## PostHog Observability

### Credentials

All PostHog credentials are in `.env.posthog` (gitignored, project root). Source it before querying:

```bash
source .env.posthog
```

| Variable | Value | Purpose |
|----------|-------|---------|
| `POSTHOG_PROJECT_TOKEN` | `phc_...` (in .env.posthog) | Capture events (used by Workers) |
| `POSTHOG_PERSONAL_API_KEY` | `phx_...` (in .env.posthog) | Query events (used by scripts/agents) |
| `POSTHOG_HOST` | `https://eu.i.posthog.com` | Ingestion host (Workers send events here) |
| `POSTHOG_API_HOST` | `https://eu.posthog.com` | API host (scripts/agents query here) |
| `POSTHOG_PROJECT_ID` | `146226` | Project "Default project" |

### Query Helper Script

```bash
# List events for a session
./scripts/posthog-query.sh session <session-uuid>

# List events for any distinct_id
./scripts/posthog-query.sh events debug-posthog-test 10

# Search by event name
./scripts/posthog-query.sh search speech_continues 20

# Recent events across all sessions
./scripts/posthog-query.sh recent 50

# Count events by type for a session
./scripts/posthog-query.sh count session:<uuid>

# Live tail (polls every 5s)
./scripts/posthog-query.sh tail
```

### Direct HogQL Queries

For custom queries, use the PostHog API directly:

```bash
source .env.posthog

curl -s "${POSTHOG_API_HOST:-https://eu.posthog.com}/api/projects/${POSTHOG_PROJECT_ID}/query/" \
  -H "Authorization: Bearer ${POSTHOG_PERSONAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "kind": "HogQLQuery",
      "query": "SELECT timestamp, event, distinct_id FROM events WHERE distinct_id = '\''session:YOUR-UUID'\'' ORDER BY timestamp ASC LIMIT 100"
    }
  }' | python3 -m json.tool
```

### Distinct ID Format

- Sessions: `session:<uuid>` (e.g. `session:02732348-1350-413a-9f0c-d6f47210b417`)
- VAD DO: `session:<uuid>:vad`
- Smart Turn DO: `session:<uuid>:st`
- Debug: `debug-posthog-test`

### Event Names (Worker → PostHog)

| Event | Source | When |
|-------|--------|------|
| `avatar_session_created` | Worker `fetch()` | `POST /sessions` |
| `ws_connected` | DO | WebSocket upgrade |
| `session_start_requested` | DO | `session.start` message |
| `session_ready` | DO | Modal session started |
| `mic_start_requested` | DO | `mic.start` message |
| `first_audio_chunk_received` | DO | First PCM16 chunk |
| `audio_stats` | DO | Every 50 chunks |
| `vad_boundary` | DO | VAD speech start/end detected |
| `speech_started` | DO | Turn begins |
| `vad_start_suppressed_while_turn_active` | DO | VAD start during active turn (key diagnostic) |
| `speech_continues` | DO | Smart Turn says prediction=0 |
| `speech_finalized` | DO | Turn ends (reason: smart_turn_complete, smart_turn_error_fallback, mic_stop, session_end) |
| `smart_turn_analyze_started` | DO | Before ST inference |
| `smart_turn_analyze_completed` | DO | After ST inference (includes latencyMs) |
| `mic_stop_summary` | DO | `mic.stop` message |
| `mic_stop_requested` | DO | `mic.stop` message |
| `session_end_requested` | DO | `session.end` message |
| `session_ended` | DO | After Modal end attempt |
| `ws_closed` | DO | WebSocket close |
| `pipeline_error` | DO | Any pipeline error |
| Browser events | Browser via DO relay | `client.telemetry` messages (source: "browser") |

### Debug Endpoint

```bash
# Test PostHog connectivity (returns key prefix, SDK status, raw fetch status)
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/debug/posthog-test \
  -H "x-debug-token: avtr-debug-posthog-2026"
```

## Deployed Services

| Service | URL | Cloudflare Worker |
|---------|-----|-------------------|
| Web Client | `https://avatar-web-client.pages.dev` | Pages |
| Session Bridge | `https://avatar-session-bridge.vallangirakesh.workers.dev` | Worker + DO |
| VAD Worker | `https://any-toy-vad.vallangirakesh.workers.dev` | Worker + DO |
| Smart Turn Worker | `https://any-toy-smart-turn.vallangirakesh.workers.dev` | Worker + DO |

## Worker Tail Commands

```bash
npx wrangler tail avatar-session-bridge --format json    # Bridge + DO
npx wrangler tail any-toy-vad --format json               # VAD
npx wrangler tail any-toy-smart-turn --format json         # Smart Turn
```

## Key Project Directories

| Path | Purpose |
|------|---------|
| `avatar-saas-prototype/cloudflare-session-bridge/` | Bridge Worker (WebSocket, DO, VAD/ST orchestration, PostHog) |
| `avatar-saas-prototype/web-client/` | Browser client (AudioWorklet, WebSocket, UI) |
| `/Users/ayya/developer/my-toy/cloudflare_toy_agent/vad-worker/` | VAD Worker (shared, deployed) |
| `/Users/ayya/developer/my-toy/cloudflare_toy_agent/smart-turn-worker/` | Smart Turn Worker (shared, deployed) |

## Secrets

| Secret | Worker | Purpose |
|--------|--------|---------|
| `POSTHOG_API_KEY` | avatar-session-bridge | PostHog project token (phc_...) |
| `DEBUG_POSTHOG_TOKEN` | avatar-session-bridge | Debug endpoint auth |
| `POSTHOG_TOKEN` | any-toy-agent | (legacy, same project token) |

## Build & Deploy

```bash
# Bridge Worker
cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit && npx wrangler deploy

# Web Client
cd avatar-saas-prototype/web-client && npm run deploy

# VAD Worker
cd /Users/ayya/developer/my-toy/cloudflare_toy_agent/vad-worker && node build.mjs && npx wrangler deploy .worker/index.js --no-bundle

# Smart Turn Worker
cd /Users/ayya/developer/my-toy/cloudflare_toy_agent/smart-turn-worker && node build.mjs && npx wrangler deploy .worker/index.js --no-bundle
```
