# Avatar API Architecture

## Purpose

This document defines the first production-shaped API boundary for the avatar platform. The goal is to give developers one stable API while we keep the rendering, media delivery, and provider internals swappable.

## Core Decision

Cloudflare is the public API/control plane. Modal is an internal rendering provider.

External customers should never need to know Modal endpoint names, Modal app versions, fMP4 adapter internals, or which rendering canary is currently deployed. They should interact with sessions, avatars, backgrounds, and streams through `/v1/*` APIs.

## Runtime Roles

| Layer | Responsibility | Notes |
|---|---|---|
| Cloudflare Worker | Public HTTP API gateway | Auth, routing, request validation, response normalization, telemetry. |
| Durable Object | Per-session coordinator | Owns session state, WebSocket lifecycle, Modal calls, media relay, VAD/ST orchestration. |
| Modal 5Q | Rendering backend | Starts avatar session, accepts audio, produces fMP4 media, handles avatar/background render state. |
| Web Client | Demo/reference client | Uses the same API concepts, but can keep demo-specific conveniences. |
| PostHog | Observability | Ground truth for session timelines, errors, and latency. |

## Public API Principles

1. The public API exposes product concepts, not infrastructure concepts.
2. `/v1/*` routes require API-key auth unless explicitly marked as demo/dev-only.
3. Response shapes must be consistent across endpoints.
4. Modal failures must be translated into stable API error codes.
5. Media transport must be replaceable without changing session creation semantics.
6. The current WebSocket to MSE path remains the active media path until a later canary proves a better transport.

## Canonical Objects

### Session

A live avatar interaction instance.

Minimum fields:

- `session_id`
- `avatar_id`
- `bg_id`
- `status`
- `created_at`
- `stream`
- `timings`

### Avatar

A renderable avatar backed by cached precomputed assets.

Minimum fields:

- `avatar_id`
- `display_name`
- `available`
- `default_bg_id`
- `preview_url` when available

### Background

A selectable scene/background for rendering.

Minimum fields:

- `bg_id`
- `display_name`
- `available`
- `preview_url` when available

### Stream

The client-facing media connection information.

Minimum fields:

- `transport`: initially `websocket_fmp4`
- `ws_url`
- `status_url`

## Initial API Surface

The first implementation should stay small and stable.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/sessions` | Create a session record and return connection URLs. |
| `POST` | `/v1/sessions/:session_id/start` | Start Modal-backed rendering for an existing session. |
| `GET` | `/v1/sessions/:session_id/status` | Return current session status and timing info. |
| `DELETE` | `/v1/sessions/:session_id` | End session and release backend resources. |
| `GET` | `/v1/avatars` | List cached/renderable avatars. |
| `GET` | `/v1/backgrounds` | List selectable backgrounds if discoverable. |
| `GET` | `/v1/health` | Lightweight API health check. |

Existing demo routes may continue to exist, but new product-facing work should use `/v1/*`.

## Request And Response Shape

Successful responses:

```json
{
  "success": true,
  "data": {},
  "timings": {
    "latency_ms": 123
  }
}
```

Error responses:

```json
{
  "success": false,
  "error": {
    "code": "avatar_not_found",
    "message": "Avatar is not available."
  },
  "timings": {
    "latency_ms": 123
  }
}
```

## Auth Boundary

External `/v1/*` API routes should require:

```http
Authorization: Bearer <api_key>
```

or:

```http
x-api-key: <api_key>
```

API keys must come from Cloudflare Worker secrets or environment bindings. Do not hardcode customer keys in source or checked-in config.

The existing browser demo may keep its current flow during the transition, but the API implementation should make the external auth boundary explicit.

## Media Transport Boundary

Session creation should return a stream object. The caller should not care whether the backend transport is:

- current WebSocket plus fMP4 segments,
- Cloudflare Stream,
- WebRTC,
- another future transport.

Only the `stream.transport` and connection fields should change.

## Telemetry Requirements

Every public API request should emit:

- `api_request_started`
- `api_request_completed`
- `api_request_failed`

Useful properties:

- `route`
- `method`
- `status`
- `latencyMs`
- `sessionId` when available
- `avatarId` when available
- `errorCode` on failure

Session/media events already emitted by the Durable Object should remain unchanged.

## Out Of Scope For This Phase

- Billing.
- Tenant database.
- Avatar upload pipeline.
- Background upload pipeline.
- Cloudflare Stream or WebRTC rewrite.
- GPU snapshot, Dynamo, or cold-start research.
- Multi-region routing.

## Implementation Guidance

Start by wrapping the current working Bridge and Modal 5Q behavior behind stable `/v1/*` contracts. Prefer additive changes. Do not disturb the validated 5Q media path unless a route needs to call it through a cleaner API boundary.

The immediate success condition is not a complete SaaS backend. It is a stable, documented API shell that can run the existing avatar session flow and survive future backend swaps.
