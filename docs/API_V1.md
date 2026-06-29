# Avatar API v1

This documents the first stable public API surface for the avatar platform. It
wraps the existing rendering and media pipeline behind consistent `/v1/*`
contracts so the backend can evolve without breaking clients.

See `docs/API_ARCHITECTURE.md` for the runtime roles and design principles.
**Modal is an internal rendering provider.** No Modal endpoint names, session
ids, or provider concepts appear in public responses.

A machine-readable OpenAPI 3.0 specification is the formal contract:
[`docs/openapi-v1.yaml`](./openapi-v1.yaml).

## Quick Links

| Resource | Description |
|----------|-------------|
| [API Quickstart](./API_QUICKSTART.md) | 5-minute copy-pasteable walkthrough |
| [Security Validation Runbook](./API_SECURITY_VALIDATION.md) | Multi-tenant auth config + cross-tenant testing |
| [OpenAPI 3.0 Spec](./openapi-v1.yaml) | Formal machine-readable contract |
| [WebSocket Protocol](./WEBSOCKET_PROTOCOL.md) | Full WS message reference, error semantics, render modes |
| [Example Node.js Client](../examples/node-api-client/) | Dependency-free SDK-style helper + demo CLI |

## Contract validation

A lightweight drift guard keeps the OpenAPI spec, TypeScript types, and docs in
sync. It checks required endpoints, error codes, WebSocket message types, local
markdown links, and accidental secret leaks in `docs/` and `examples/`.

```bash
node scripts/validate-api-contract.mjs
```

Run this before any release commit that changes the public API surface.

## Base URL

```
https://avatar-session-bridge.vallangirakesh.workers.dev
```

## Authentication

All `/v1/*` routes require an API key, supplied via either header:

```http
Authorization: Bearer <api_key>
```

or:

```http
x-api-key: <api_key>
```

Keys are validated against `AVTR_API_KEY_METADATA_JSON` first (tenant
metadata), then `AVTR_API_KEYS` (legacy comma-separated list). Both are
Cloudflare Worker secrets. See [`GET /v1/me`](#get-me) for the resolution
order and how to configure metadata mode. Do not hardcode keys in source
or checked-in config.

### Auth errors

| Status | `error.code`            | When |
|--------|-------------------------|------|
| 401    | `missing_api_key`       | No key provided |
| 403    | `invalid_api_key`       | Key not in the configured list |
| 503    | `api_auth_not_configured` | Neither `AVTR_API_KEYS` nor valid `AVTR_API_KEY_METADATA_JSON` entries configured |

A lightweight unauthenticated health probe is available at the legacy
`GET /health` (outside `/v1`). `GET /v1/health` is authed for consistency.

## Request Correlation

Every `/v1/*` response includes an `X-Request-Id` header for tracing. If the
client provides `X-Request-Id` in the request, it is echoed back (if valid).
Otherwise, a UUID is generated.

```
Request:  X-Request-Id: my-trace-abc123     (optional, 1-128 chars: [a-zA-Z0-9_-])
Response: X-Request-Id: my-trace-abc123
```

For JSON responses, `request_id` is also included in the `timings` object:

```json
{
  "success": true,
  "data": { ... },
  "timings": { "latency_ms": 42, "request_id": "my-trace-abc123" }
}
```

The request_id is included in all telemetry events for end-to-end tracing.
Invalid characters in a client-provided `X-Request-Id` cause a UUID to be
generated instead (the request is not rejected).

## Session Lifecycle Hardening

The session start lifecycle is protected against concurrent request races:

### Duplicate start protection

When `POST /v1/sessions/:id/start` is called:

| Scenario | Response |
|----------|----------|
| Session not started | 200 — starts the session (synchronous) |
| Already active, same avatar | 200 — `state_note: "already_started"` |
| Already active, different avatar | 409 — `session_conflict` |
| Start in progress (<180s) | 409 — `session_starting` |
| Start in progress (stale >180s) | Clears stale state, proceeds |

The stale timeout is 180 seconds (1.5× the Modal start timeout of 120s).

### DELETE during starting

If `DELETE /v1/sessions/:id` arrives while a start is in progress:

1. DELETE returns **200 `ended`** immediately (idempotent cleanup)
2. The in-flight start detects cancellation via state reload + attempt token
3. The start returns **409 `session_cancelled`**
4. No session resurrection — the session stays ended
5. If Modal created a backend session, best-effort cleanup is attempted

### DELETE idempotency

`DELETE /v1/sessions/:id` is always safe to retry:

| Scenario | Status | Body |
|----------|--------|------|
| Active session | 200 | `status: "ended"`, summary included |
| During starting | 200 | `status: "ended"`, `start_cancelled: true` |
| Already ended | 200 | `status: "ended"`, `wasActive: false` |
| Never started | 200 | `status: "ended"`, `wasActive: false` |
| Non-existent/unowned | 404 | `not_found` |

DELETE and `POST /end` are **exempt from rate limits** (cleanup category).

## Rate Limits

All `/v1/*` routes are subject to **per-tenant abuse-control rate limits**.
These are lightweight safety nets to prevent accidental or malicious
overuse of expensive backend resources. They are **not billing quotas** —
no usage is counted, billed, or enforced globally.

### How it works

Rate limits are enforced using [Cloudflare Rate Limiting bindings](https://developers.cloudflare.com/workers/runtime-apis/bindings/rate-limit/).
Limits are tracked **per Cloudflare location** (not globally strict). This means
a tenant hitting from 2 CF locations gets 2× the limit. This is acceptable for
abuse control — legitimate clients typically hit from 1–2 locations.

### Categories

Routes are classified into categories with independent limits:

| Category | Limit | Period | Routes |
|----------|-------|--------|--------|
| `gpu_start` | 5 | 60s | `POST /v1/sessions/:id/start` |
| `stream_connect` | 5 | 60s | `GET /v1/sessions/:id/stream` |
| `ws_connect` | 5 | 60s | `GET /v1/sessions/:id/ws` (connection attempts, not session duration) |
| `session_create` | 20 | 60s | `POST /v1/sessions` |
| `ws_token` | 20 | 60s | `POST /v1/sessions/:id/ws-token` |
| `read` | 120 | 60s | All other GET routes (health, me, avatars, backgrounds, providers, status) |
| `cleanup` | Exempt | — | `DELETE /v1/sessions/:id`, `POST /v1/sessions/:id/end` |

**Cleanup is exempt** so that tenants can always end sessions even if they have
exhausted other rate limit buckets.

The `ws_connect` category limits **connection attempts** to the WebSocket upgrade
endpoint, not the duration of an active WebSocket session. Once a WebSocket
connection is established, it is not subject to rate limiting.

### 429 Response

When a limit is exceeded:

```json
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
Retry-After: 60

{
  "success": false,
  "error": {
    "code": "rate_limited",
    "message": "Rate limit exceeded for category \"session_create\". Retry after 60 seconds."
  },
  "timings": { "latency_ms": 2 }
}
```

The `Retry-After` header indicates the window duration in seconds.

### Telemetry

Each rejected request emits an `api_rate_limited` event to PostHog containing:
`tenantId`, `authMode`, `category`, `route`, `limit`, `period`. No other
tenant's data is included.

### Default values are beta

These limits are conservative defaults. They may be adjusted in future releases.
If you need higher limits for your use case, contact the platform team.

### Binding-missing policy (fail open)

If a Cloudflare Rate Limit binding is not configured (e.g., missing from
`wrangler.jsonc` or running locally without bindings), the request is **allowed**
(fail open) to avoid turning a misconfigured limiter into a production outage.
In this case, an `api_rate_limit_binding_missing` event is emitted to PostHog
with `category`, `bindingName`, `tenantId`, `authMode`, and `route`. This ensures
missing bindings are visible in telemetry rather than silently ignored.

## Response Shape

Every response uses one canonical envelope.

**Success:**

```json
{
  "success": true,
  "data": {},
  "timings": { "latency_ms": 123 }
}
```

**Error:**

```json
{
  "success": false,
  "error": { "code": "avatar_not_found", "message": "..." },
  "timings": { "latency_ms": 123 }
}
```

HTTP status codes are preserved (e.g. a 409 conflict returns HTTP 409 with the
error envelope).

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET`    | `/v1/health`                       | Authed API health check |
| `POST`   | `/v1/sessions`                     | Create a session record and connection URLs |
| `POST`   | `/v1/sessions/:session_id/start`   | Start Modal-backed rendering for an existing session |
| `GET`    | `/v1/sessions/:session_id/status`  | Return current session status and timing info |
| `DELETE` | `/v1/sessions/:session_id`         | End session and release backend resources |
| `POST`   | `/v1/sessions/:session_id/ws-token`| Mint a fresh short-lived WebSocket session token |
| `GET`    | `/v1/sessions/:session_id/ws`      | Authenticated WebSocket upgrade (token required) |
| `GET`    | `/v1/avatars`                      | List renderable avatars |
| `GET`    | `/v1/avatars/:avatar_id`           | Get avatar details |
| `GET`    | `/v1/backgrounds`                  | List selectable backgrounds |
| `GET`    | `/v1/backgrounds/:bg_id`           | Get background details |

### Compatibility aliases

- `POST /v1/sessions/:session_id/end` — same as `DELETE /v1/sessions/:session_id` (kept for backward compatibility).
- `GET /v1/sessions/:session_id/stream` — Server-Sent Events (SSE) fMP4 media stream, an alternate transport. Returned raw (not wrapped in the JSON envelope).

---

### GET /v1/health

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/health \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": { "status": "ok", "version": "v1", "transport": "websocket_fmp4" },
  "timings": { "latency_ms": 12 }
}
```

### POST /v1/sessions

Allocates a session id and returns connection URLs. The avatar is selected at
`/start`. Request body is optional.

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c...-...-...",
    "avatar_id": null,
    "bg_id": null,
    "status": "created",
    "created_at": 1718352000000,
    "stream": {
      "transport": "websocket_fmp4",
      "available_transports": ["websocket_fmp4"],
      "ws_url": "wss://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/session:8f3c.../ws?token=eyJ...",
      "status_url": "https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/session:8f3c.../status",
      "token_expires_at": 1718352300000
    }
  },
  "timings": { "latency_ms": 9 }
}
```

The `ws_url` includes a short-lived HMAC-signed token (`?token=...`) valid for 5
minutes. Open the WebSocket using this URL as-is. If the token expires before
you connect, call [`POST /v1/sessions/:id/ws-token`](#post-v1sessionsid-ws-token)
to mint a fresh one. See [WebSocket Authentication](#websocket-authentication).

#### SFU Private Beta Authorization

The Cloudflare Realtime SFU transport (`webrtc_sfu`) is gated. A session can be
SFU-beta-authorized via:

1. Per-API-key metadata: `"sfu_beta": true` in `AVTR_API_KEY_METADATA_JSON`.
2. Tenant allowlist: the tenant id appears in `AVTR_SFU_BETA_TENANTS`.
3. Short-lived HMAC-signed token: pass `X-SFU-Beta-Token: <token>` header or
   `sfu_beta_token` in the JSON body.

When authorized and SFU is fully configured, `available_transports` includes
`webrtc_sfu` and `stream.sfu` contains the subscriber/answer/close URLs and
`sfu_token`. The raw beta token is verified server-side and discarded; only an
authorization flag is persisted on the session.

### POST /v1/sessions/:session_id/start

Starts Modal-backed rendering. Optionally pre-select the avatar and background.

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SID/start \
  -H "Authorization: Bearer $AVTR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"avatar_id":"arnold","bg_id":"aloe_window"}'
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c...",
    "avatar_id": "arnold",
    "bg_id": null,
    "status": "ready",
    "created_at": null,
    "started_at": 1718352001000,
    "stream": { "transport": "websocket_fmp4", "ws_url": "...", "status_url": "..." }
  },
  "timings": { "latency_ms": 2143 }
}
```

Re-calling `/start` with the same avatar is idempotent (`status: "ready"`).
Calling it with a different avatar while active returns `409 session_conflict`.

### GET /v1/sessions/:session_id/status

Returns the authoritative session state and timings. Use this for the full
timing picture (`created_at`, `started_at`, `ended_at`).

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SID/status \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c...",
    "avatar_id": "arnold",
    "bg_id": null,
    "status": "active",
    "created_at": 1718352000000,
    "started_at": 1718352001000,
    "ended_at": null,
    "stream": { "transport": "websocket_fmp4", "ws_url": "...", "status_url": "..." }
  },
  "timings": { "latency_ms": 45 }
}
```

### DELETE /v1/sessions/:session_id

Ends the session and releases backend resources.

```bash
curl -X DELETE https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SID \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "session_id": "session:8f3c...",
    "avatar_id": "arnold",
    "bg_id": null,
    "status": "ended",
    "created_at": 1718352000000,
    "ended_at": 1718352060000,
    "stream": { "transport": "websocket_fmp4", "ws_url": "...", "status_url": "..." },
    "end_ok": true,
    "end_summary": { "chunks_rendered": 128, "duration_s": 60.0 }
  },
  "timings": { "latency_ms": 1820 }
}
```

### GET /v1/avatars

Lists renderable avatars. Responses come from a `CatalogProvider` abstraction
(see [Catalog Architecture](#catalog-architecture)).

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/avatars \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "avatars": [
      { "avatar_id": "arnold", "display_name": "Arnold", "available": true, "default_bg_id": "aloe_window" }
    ]
  },
  "timings": { "latency_ms": 7 }
}
```

### GET /v1/avatars/:avatar_id

Returns a single avatar. Unknown or unavailable avatars return `400`
`avatar_not_found`.

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/avatars/arnold \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "avatar": { "avatar_id": "arnold", "display_name": "Arnold", "available": true, "default_bg_id": "aloe_window" }
  },
  "timings": { "latency_ms": 5 }
}
```

### GET /v1/backgrounds

Lists selectable backgrounds. Same provider abstraction as avatars.

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/backgrounds \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "backgrounds": [
      { "bg_id": "aloe_window", "display_name": "Aloe Window", "available": true }
    ]
  },
  "timings": { "latency_ms": 6 }
}
```

### GET /v1/backgrounds/:bg_id

Returns a single background. Unknown or unavailable backgrounds return `400`
`background_not_found`.

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/backgrounds/aloe_window \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "background": { "bg_id": "aloe_window", "display_name": "Aloe Window", "available": true }
  },
  "timings": { "latency_ms": 4 }
}
```

### GET /v1/providers

Returns the capability configuration of the platform — which STT, LLM,
TTS, avatar-rendering, VAD, and Smart Turn providers are effectively
available. All fields are safe public identifiers; no API keys, internal
URLs, provider hostnames, or session/container IDs are exposed.

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/providers \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "stt": {
      "provider": "sarvam",
      "model": "saarika:v2.5",
      "language": "unknown"
    },
    "llm": {
      "provider": "openai-compatible",
      "model": "gpt-oss-120b"
    },
    "tts": {
      "provider_chain": "omnivoice_same_gpu_parallel_sarvam_melotts",
      "omnivoice_enabled": false,
      "same_gpu_parallel_enabled": true,
      "gpu_target": "l40s",
      "fallback_enabled": true
    },
    "avatar": {
      "transport": "websocket_fmp4",
      "render_backend": "managed_gpu_renderer"
    },
    "vad": { "enabled": true },
    "smart_turn": { "enabled": true }
  },
  "timings": { "latency_ms": 1 }
}
```

**Field semantics:**

| Field | Description |
|-------|-------------|
| `stt.provider` | Speech-to-text vendor identifier (`sarvam`) |
| `stt.model` | STT model name from server config |
| `stt.language` | Default language code (`unknown` if unset) |
| `llm.provider` | LLM compatibility class (`openai-compatible`) |
| `llm.model` | LLM model name from server config, `null` if unset |
| `tts.provider_chain` | Ordered fallback chain string |
| `tts.omnivoice_enabled` | True only when OmniVoice L4 TTS is effectively wired (flag + URL + parallel not active) |
| `tts.same_gpu_parallel_enabled` | True only when same-GPU parallel TTS is effectively wired (flag + lifecycle URL present) |
| `tts.gpu_target` | GPU class for parallel TTS (`l4`, `l40s`, `a10g`), `null` when disabled |
| `tts.fallback_enabled` | Always `true` — Sarvam → MeloTTS fallback is intrinsic |
| `avatar.transport` | Media transport type (`websocket_fmp4`) |
| `avatar.render_backend` | Provider-neutral rendering backend (`managed_gpu_renderer`) |
| `vad.enabled` | Voice Activity Detection service bound |
| `smart_turn.enabled` | Smart Turn detection service bound |

### GET /v1/me

Returns the authenticated caller's identity metadata. Useful for
client-side logging, multi-tenant routing, and plan-gating.

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/me \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

**Metadata mode** (key present in `AVTR_API_KEY_METADATA_JSON`):

```json
{
  "success": true,
  "data": {
    "tenant_id": "tenant_demo",
    "key_name": "Demo Tenant",
    "plan": "dev",
    "auth_mode": "metadata"
  },
  "timings": { "latency_ms": 1 }
}
```

**Legacy mode** (key in `AVTR_API_KEYS` only, no metadata):

```json
{
  "success": true,
  "data": {
    "tenant_id": "legacy",
    "key_name": null,
    "plan": null,
    "auth_mode": "legacy"
  },
  "timings": { "latency_ms": 1 }
}
```

**Field semantics:**

| Field | Description |
|-------|-------------|
| `tenant_id` | Tenant identifier from metadata, or `"legacy"` when no metadata is configured |
| `key_name` | Human-readable key name from metadata, `null` in legacy mode |
| `plan` | Plan identifier from metadata (`"dev"`, `"pro"`, etc.), `null` in legacy mode |
| `auth_mode` | `"metadata"` when key resolved via `AVTR_API_KEY_METADATA_JSON`, `"legacy"` for `AVTR_API_KEYS`-only keys |

**Auth resolution order:**

1. Key timing-safely matched in `AVTR_API_KEY_METADATA_JSON` → `auth_mode: "metadata"`
2. Key timing-safely matched in `AVTR_API_KEYS` → `auth_mode: "legacy"`
3. If same key exists in both, metadata wins (checked first)

**Configuring metadata (Worker secret):**

```bash
wrangler secret put AVTR_API_KEY_METADATA_JSON
# Paste JSON:
# {
#   "avtr_test_your_key_here": {
#     "tenant_id": "tenant_demo",
#     "name": "Demo Tenant",
#     "plan": "dev"
#   }
# }
```

Only entries with a non-empty `tenant_id` string are accepted. `name`
and `plan` may be string or null/omitted. Invalid entries are dropped
with a `console.warn` (never logging raw keys). Invalid JSON falls back
to legacy mode — never 5xx.

## Error Codes

| Code | Status | Meaning |
|---|---|---|
| `missing_api_key` | 401 | No API key provided |
| `invalid_api_key` | 403 | Key not recognized |
| `api_auth_not_configured` | 503 | Server has no `AVTR_API_KEYS` and no valid `AVTR_API_KEY_METADATA_JSON` entries |
| `invalid_session_id` | 400 | Session id does not start with `session:` |
| `method_not_allowed` | 405 | Wrong HTTP method for the route |
| `invalid_body` | 400 | Request body is not valid JSON |
| `not_found` | 404 | No route matched |
| `internal_error` | 500 | Unexpected server error |
| `session_not_started` | 409 | Operation requires `/start` first |
| `session_conflict` | 409 | Session already active with a different avatar |
| `session_starting` | 409 | A start is already in progress for this session |
| `session_cancelled` | 409 | Start was cancelled (session ended during start) |
| `modal_start_failed` | 502 | Rendering backend rejected start |
| `modal_unreachable` | 500 | Rendering backend unreachable |
| `no_modal_protocol_enabled` | 503 | No rendering protocol configured |
| `avatar_not_found` | 400 | Requested avatar is unknown or unavailable |
| `background_not_found` | 400 | Requested background is unknown or unavailable |
| `sfu_unavailable` | 502 | SFU Modal start or SFU REST call failed (SFU transport only) |
| `sfu_not_configured` | 503 | Client explicitly requested `webrtc_sfu` but SFU prerequisites are missing |
| `sfu_beta_not_enabled` | 403 | Client explicitly requested `webrtc_sfu` but the session is not SFU-beta-authorized |
| `sfu_session_not_found` | 404 | SFU subscriber/publisher session or track not found (SFU transport only) |
| `sfu_wrong_transport` | 409 | SFU REST endpoint called on a non-SFU session |
| `sfu_turn_credentials_unavailable` | 503 | TURN relay credentials could not be generated (missing TURN Key, API failure, or no usable TURN URLs) |
| `sfu_capacity_exceeded` | 503 | Another SFU session is already active on the single-GPU runtime. Wait for it to end before starting a new one. |
| `sfu_capacity_unavailable` | 503 | Capacity manager is unreachable (DO outage). SFU start is rejected fail-closed to protect the single GPU. Retry shortly. |
| `ws_token_missing` | 401 | WebSocket upgrade attempted without `?token=` |
| `ws_token_invalid` | 401 | Token malformed, bad signature, expired, or wrong session |
| `ws_token_secret_not_configured` | 503 | `AVTR_SESSION_TOKEN_SECRET` not set on server |

## WebSocket Authentication

The public API WebSocket route is `GET /v1/sessions/:session_id/ws?token=...`.
A short-lived HMAC-SHA256 signed token is **mandatory** from day one — there is
no unauthenticated mode on the `/v1/` WS path.

### Token lifecycle

| Property | Value |
|----------|-------|
| Signing | HMAC-SHA256 via Web Crypto |
| Secret | `AVTR_SESSION_TOKEN_SECRET` (Worker secret, 32+ chars) |
| TTL | 5 minutes (300 seconds) |
| Payload | `{sid, tid, purpose, exp}` — never the raw API key |
| Transport | `?token=` query param on `ws_url` |

Tokens are minted on:

1. **`POST /v1/sessions`** — the response `stream.ws_url` includes a fresh token.
2. **`POST /v1/sessions/:id/ws-token`** — returns a new token for an existing session.

Other responses (`/status`, `/start`, `/end`) include `stream.ws_url` pointing to
`/v1/sessions/:id/ws` **without** a token — use the token from session creation
or call the refresh endpoint.

### Security model

- Token binds to `sid` (session ID) — a token for session A is rejected on
  session B's WS path.
- Token binds to `tid` (tenant ID from API auth context) — included in payload
  for audit/telemetry but not exposed in error responses.
- `purpose` is always `"ws"` — prevents token reuse in other contexts.
- Signature verified with constant-time comparison.
- No token revocation mechanism (stateless). The 5-minute TTL is the revocation
  window. If a longer-lived credential is compromised, rotate
  `AVTR_SESSION_TOKEN_SECRET` (invalidates all outstanding tokens).

## Session Ownership

Every session created via `POST /v1/sessions` is **bound to the tenant** that
created it. All subsequent `/v1/sessions/:id/*` operations verify that the
caller's `tenant_id` matches the stored session owner.

### Ownership model

| Property | Value |
|----------|-------|
| Storage | Durable Object SQLite (`session:owner` key) |
| Written on | `POST /v1/sessions` (before response) |
| Fields | `tenant_id`, `auth_mode`, `key_name`, `plan`, `created_at` |
| Immutable | Yes — set once at creation, never overwritten |
| Survives hibernation | Yes — persisted in DO storage |

### Enforcement points

Ownership is verified before **all** session-scoped routes:

- `POST /v1/sessions/:id/start`
- `GET /v1/sessions/:id/status`
- `DELETE /v1/sessions/:id` / `POST /v1/sessions/:id/end`
- `GET /v1/sessions/:id/stream`
- `POST /v1/sessions/:id/ws-token`
- `GET /v1/sessions/:id/ws` (after HMAC token validation)

### Authorization rule

- Caller `tenant_id` must match stored owner `tenant_id`.
- Missing owner or mismatch → **`404 not_found`** with message `"Session not found."`
- This response is identical whether the session doesn't exist or belongs to
  another tenant — no information leakage.
- The WS route validates the HMAC token **first** (401 on failure), then checks
  ownership (404 on failure).

### Legacy auth caveat

When using `AVTR_API_KEYS` (legacy comma-separated keys), **all keys share
`tenant_id = "legacy"`**. This means any legacy key can access sessions created
by any other legacy key. Legacy mode is intended for **dev/single-tenant
compatibility only**.

Production multi-tenant deployments must use `AVTR_API_KEY_METADATA_JSON` with
distinct `tenant_id` values per key:

```json
{
  "avtr_live_xxx": { "tenant_id": "acme_corp", "name": "Acme Production", "plan": "enterprise" },
  "avtr_live_yyy": { "tenant_id": "beta_inc",  "name": "Beta Staging",   "plan": "starter" }
}
```

### Demo path (not part of public API)

The legacy `GET /sessions/:id/ws` route (no `/v1/` prefix) is **demo-only** and
requires no token and no ownership check. Sessions created via the demo
`POST /sessions` endpoint have no ownership record and will return `404` if
accessed via any `/v1/` route.

> **Client warning:** Do not log, screenshot, or store `ws_url` in browser
> history or analytics tools. The token in the query string is a bearer
> credential. Use it to open the WebSocket and then discard the URL.

### POST /v1/sessions/:session_id/ws-token

Mints a fresh 5-minute WebSocket session token for an existing session. Use this
for reconnects, delayed connections, or browser sleep recovery.

```bash
curl -X POST https://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/$SID/ws-token \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

```json
{
  "success": true,
  "data": {
    "ws_url": "wss://avatar-session-bridge.vallangirakesh.workers.dev/v1/sessions/session:8f3c.../ws?token=eyJ...",
    "token_expires_at": 1718352300000
  },
  "timings": { "latency_ms": 3 }
}
```

Requires normal `/v1/` API key authentication. The new token is bound to the
same session and tenant as the original.

## Media Transport

The current client-facing media transport is **`websocket_fmp4`**: the browser
connects to the `ws_url` from the `stream` object and receives fragmented MP4
init + media segments over WebSocket for playback via Media Source Extensions
(MSE). This transport is replaceable without changing session creation
semantics — only `stream.transport` and the connection fields will change.

The SSE `/v1/sessions/:id/stream` route is an alternate transport that emits
`fmp4_init` / `fmp4_media` / `end` events. It is returned raw (not JSON-wrapped).

## SFU Transport

The platform supports an alternative **`webrtc_sfu`** transport using Cloudflare
Realtime SFU. This transport is **gated and disabled by default** — it is only
available when the server is fully configured (`AVTR_REALTIME_SFU_ENABLED`,
Cloudflare Calls credentials, Modal worker secret, and Worker base URL).

> **Milestone 1 scope:** The SFU transport delivers **video only** over WebRTC
> (H264 via NVENC). Avatar speech audio continues to arrive over the existing
> WebSocket messages (`avatar.speech.*`, `assistant.audio`). There is no Opus
> audio track in the WebRTC stream. A/V sync is handled client-side.

### Availability

When SFU is configured, `POST /v1/sessions` returns additional fields in the
`stream` object:

| Field | Description |
|-------|-------------|
| `stream.available_transports` | `["websocket_fmp4", "webrtc_sfu"]` (includes `webrtc_sfu` only when configured). |
| `stream.sfu.subscribe_url` | `POST` — initiate subscriber track. |
| `stream.sfu.answer_url` | `POST` — send browser SDP answer. |
| `stream.sfu.close_url` | `POST` — tear down subscriber track. |
| `stream.sfu.token` | SFU session token (purpose `"sfu"`). Distinct from the WS token. |
| `stream.sfu.track_names` | Publisher track names (e.g. `["avatar-video"]`). |

When SFU is not configured, `stream.available_transports` is `["websocket_fmp4"]`
and the `sfu` block is absent. All sessions use fMP4 regardless of the
`transport` request field.

### Requesting SFU

Pass `transport: "webrtc_sfu"` at `POST /start` or WS `session.start`:

| Value | Behavior |
|-------|----------|
| `websocket_fmp4` | Forces fMP4. Never falls back. |
| `webrtc_sfu` | Requests SFU. Returns `sfu_not_configured` (503) if not configured, or `sfu_unavailable` (502) if start fails. Does **not** fall back to fMP4. |
| `auto` (default) | Server chooses. Currently always fMP4 unless `AVTR_REALTIME_SFU_DEFAULT` is explicitly set server-side. Falls back to fMP4 silently if SFU fails. |

### SFU subscribe flow

The browser uses the `sfu_token` (NOT the API key) to authenticate the three
SFU REST endpoints:

```
1. POST /v1/sessions/:id/sfu/subscribe  (?token=<sfu_token>)
   → { offer_sdp, offer_type, requires_renegotiation, mids }

2. Browser: pc.setRemoteDescription(offer)
             pc.setLocalDescription(await pc.createAnswer())
             await waitICEGathering(pc)

3. POST /v1/sessions/:id/sfu/answer     (?token=<sfu_token>)
   Body: { answer_sdp: pc.localDescription.sdp, answer_type: "answer" }
   → { status: "ok" }

4. [Media flows: SFU → browser via WebRTC RTP — H264 video]

5. POST /v1/sessions/:id/sfu/close      (?token=<sfu_token>)
   → { status: "closed" | "closed_best_effort" | "no_active_subscriber" }
```

All three endpoints accept the SFU token via `?token=` query parameter or
`Authorization: Bearer` header. Rate limits: subscribe 5/60s, answer 120/60s,
close exempt.

### Fallback to fMP4

- If `transport: "auto"` and SFU fails, the server silently falls back to fMP4.
- If `transport: "webrtc_sfu"` (explicit) and SFU fails, the server returns an
  error — it does NOT fall back. The client must reconnect with
  `transport: "websocket_fmp4"` or `transport: "auto"`.
- SFU can be globally disabled by unsetting `AVTR_REALTIME_SFU_ENABLED` on the
  server. All new sessions use fMP4.

## Telemetry

Every `/v1/*` request emits these PostHog events (distinct id = session id when
in the path, otherwise `api-public`):

- `api_request_started` — `{ route, method }`
- `api_request_completed` — `{ route, method, status, latencyMs, sessionId, avatarId }`
- `api_request_failed` — `{ route, method, status, latencyMs, sessionId, avatarId, errorCode }`

WebSocket authentication events:

- `ws_token_issued` — `{ source, tenantId, authMode, expiresAt }` — on session create and token refresh
- `ws_token_rejected` — `{ reason }` — on failed `/v1/sessions/:id/ws` validation (`missing`, `malformed`, `invalid_signature`, `expired`, `session_mismatch`, `wrong_purpose`, `secret_not_configured`)
- `ws_auth_passed` — `{ tenantId, authMode }` — on successful `/v1/sessions/:id/ws` validation

Session ownership events:

- `ownership_created` — `{ callerTenantId, authMode, route }` — on `POST /v1/sessions` after ownership persisted
- `ownership_check_passed` — `{ callerTenantId, authMode, route }` — when ownership verified before a session-scoped operation
- `ownership_check_failed` — `{ reason, callerTenantId, authMode, route }` — when ownership check fails. `reason` is `owner_not_found` or `tenant_mismatch`. **Stored owner tenant_id is never included.**

Session/media events emitted by the Durable Object are unchanged.

## Smoke Testing

Run the automated smoke test against any deployed instance:

```bash
AVTR_API_KEY=... node scripts/smoke-api-v1.mjs
# Optional: AVTR_API_BASE=https://staging.example.com
```

The script verifies auth failures, canonical response shapes, session
lifecycle (create / status / delete), and catalog endpoints. It uses two
separate sessions: one for the clean lifecycle test and one for `/start`. If
the Modal backend is unavailable, `/start` is reported as **SKIP** (known
outage), not **FAIL**. Exit code is `0` if all non-skipped tests pass.

### WS Token + Ownership Smoke Test

```bash
# Single-tenant (always runs):
AVTR_API_KEY=... node scripts/smoke-ws-token.mjs

# Cross-tenant (requires second key in AVTR_API_KEY_METADATA_JSON):
AVTR_API_KEY=... AVTR_API_KEY_B=... node scripts/smoke-ws-token.mjs
```

Tests token lifecycle (create, validate, refresh, tamper/cross-session rejection)
and session ownership (owner access, demo session isolation, nonexistent session
404, cross-tenant denial when `AVTR_API_KEY_B` is configured).

### Ownership Logic Unit Test (offline)

```bash
node scripts/ownership-unit-test.mjs
```

Pure logic tests for the ownership comparison function — no network needed.
Covers matching tenants, mismatched tenants, missing owners, legacy mode, and
edge cases.

### Rate Limit Categorization Unit Test (offline)

```bash
node scripts/rate-limit-unit-test.mjs
```

Pure logic tests for `categorizeRoute()` — verifies all 12 routes map to the
correct rate-limit category.

### Rate Limit Smoke Test (deployed)

```bash
# Wait at least 70s between runs (60s rate-limit window)
AVTR_API_KEY=... node scripts/smoke-rate-limit.mjs
```

Verifies the deployed rate-limit bindings enforce per-tenant limits: hammers
READ (121× `/v1/health`) and WRITE (21× `POST /v1/sessions`) categories to
trigger 429s, then cleans up created sessions (cleanup is exempt).

## Limitations

- Avatar and background catalogs are currently **static** — hardcoded `arnold`
  and `aloe_window` with an optional `AVTR_CATALOG_JSON` override. They may lag
  the rendering backend's actual cached set until dynamic discovery is wired in.
  See [Catalog Architecture](#catalog-architecture) below.
- `/start` responses do not include `created_at`; call `/status` for the full
  timing picture.
- `bg_id` is currently returned as `null` in normalized responses even after
  start (the backend does not yet echo the resolved background reliably).

## Catalog Architecture

`GET /v1/avatars` and `GET /v1/backgrounds` delegate to a `CatalogProvider`
interface (`src/catalog-provider.ts`), allowing the catalog source to change
without touching the public API shape or the Durable Object.

### Layered merge model

Catalog results are assembled from up to four layers, with higher layers
shadowing lower layers by `avatar_id` / `bg_id`:

| Priority | Layer | Source |
|----------|-------|--------|
| 4 (lowest) | Hardcoded defaults | `arnold`, `aloe_window` |
| 3 | `AVTR_CATALOG_JSON` overlay | Worker secret (JSON string) |
| 2 | D1 system rows | `owner_tenant_id = '__system__'` |
| 1 (highest) | D1 tenant rows | `owner_tenant_id = <tenant_id>` |

**Guarantees:**

- `arnold` and `aloe_window` are always available unless a higher layer
  explicitly shadows them with `available: false`.
- Missing rows in any layer never remove lower-layer assets.
- List results contain one row per `avatar_id` / `bg_id` (no duplicates).
- If `CATALOG_DB` is absent, only layers 4 + 3 are used.

### `available: false` behavior

| Context | Behavior |
|---------|----------|
| `listAvatars` / `listBackgrounds` | Exclude unavailable rows |
| `getAvatar` / `getBackground` | Return `null` for unavailable |
| `validateAvatarBgForStart` | Fail with `avatar_not_found` / `background_not_found` |

### Fail-open on D1 errors

If a D1 query throws (network, schema, or binding error), the provider logs
the error and returns static-only results (layers 4 + 3). Catalog endpoints
never return 5xx due to D1 issues.

### Catalog validation at session start

`POST /v1/sessions/:id/start` and the WebSocket `session.start` message
validate the resolved `avatar_id` and optional `bg_id` against the catalog
before invoking the rendering backend.  Unknown or unavailable assets fail
fast with public error codes and never leak Modal paths or cache internals.

| Scenario | REST | WebSocket |
|----------|------|-----------|
| Unknown avatar | `400 avatar_not_found` | `{ type: "error", code: "avatar_not_found" }` |
| Unknown background | `400 background_not_found` | `{ type: "error", code: "background_not_found" }` |
| Unavailable asset | same as unknown | same as unknown |

The canonical live path uses WebSocket `session.start`, so validation is
enforced there as well as on the REST alias.

### Config override: `AVTR_CATALOG_JSON`

Set the `AVTR_CATALOG_JSON` Cloudflare Worker secret (or environment variable)
to a stringified JSON value to **extend or shadow** the hardcoded defaults:

```json
{
  "avatars": [
    { "avatar_id": "sarah", "display_name": "Sarah", "available": true, "default_bg_id": "office" }
  ],
  "backgrounds": [
    { "bg_id": "office", "display_name": "Office", "available": true }
  ]
}
```

Entries overlay the hardcoded defaults by ID. To hide `arnold`, add an entry
with `"available": false`. Missing entries in the JSON do not remove defaults.

**Defensive parsing:** if `AVTR_CATALOG_JSON` is missing, invalid JSON, or
produces zero valid entries, the provider uses hardcoded defaults only and
logs a `console.warn`.

### D1 schema (Phase 6a.2)

When the optional `CATALOG_DB` D1 binding is configured, catalog metadata is
stored in two tables. Migrations live under
`avatar-saas-prototype/cloudflare-session-bridge/migrations/`.

Primary keys use a non-null `owner_tenant_id` column. System assets use the
reserved value `'__system__'`. Tenant IDs must never equal this value.

| Table | Primary key |
|-------|-------------|
| `catalog_avatars` | `(owner_tenant_id, avatar_id)` |
| `catalog_backgrounds` | `(owner_tenant_id, bg_id)` |

Production D1 activation requires creating the database, adding the binding to
`wrangler.jsonc`, and running migrations — all deferred to a separate commit.

### Future: dynamic discovery

Cloudflare Workers cannot directly inspect Modal volumes. Dynamic discovery
requires an HTTP round-trip to a Modal endpoint. Modal currently exposes
`list-cached-avatars-5q` (avatars only — no background-listing endpoint exists
yet). A future `ModalCatalogProvider` would:

1. Call the Modal avatar-listing endpoint.
2. Enrich discovered IDs with static metadata (display names, default bg).
3. Fall back to `StaticCatalogProvider` on timeout or error.

Until a Modal background-listing endpoint exists, backgrounds will remain
static-only even after avatar discovery is added.
