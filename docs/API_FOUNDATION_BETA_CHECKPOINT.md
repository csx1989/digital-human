# API Foundation Beta Checkpoint

**Status:** Beta-complete  
**Last validated:** 2026-06-15  
**Checkpoint commit:** `67dee27`  
**Contract drift guard commit:** `28d8339`

This document marks the API foundation/security/DX phase as complete enough for a
beta checkpoint. It captures what is in scope, what is intentionally deferred,
and the explicit decision to stop expanding this phase unless a security or
correctness blocker is found.

---

## What is now complete

- **`/v1` public API shell** — consistent REST routes under `/v1/*` with canonical
  `{ success, data, timings }` / `{ success, error, timings }` envelopes.
- **OpenAPI 3.0 spec** — `docs/openapi-v1.yaml` is the formal machine-readable
  contract.
- **API contract drift guard** — `scripts/validate-api-contract.mjs` keeps the
  spec, TypeScript types, docs, and examples in sync.
- **Metadata auth and `/v1/me`** — keys resolved against
  `AVTR_API_KEY_METADATA_JSON` (tenant metadata), with legacy `AVTR_API_KEYS`
  fallback.
- **Tenant/session ownership** — every session is bound to a tenant; all
  subsequent operations enforce ownership.
- **WebSocket session tokens** — short-lived HMAC tokens for secure WS upgrade.
- **Per-tenant rate limiting** — Cloudflare Rate Limiting bindings for
  `gpu_start`, `stream_connect`, `ws_connect`, `session_create`, `ws_token`,
  `read`, and `cleanup`.
- **Lifecycle reliability hardening** — start-attempt token, `starting` flag,
  DELETE-during-start cancellation, stale-timeout cleanup, and proper REST/WS
  end-session cleanup.
- **API security validation runbook** — `docs/API_SECURITY_VALIDATION.md`.
- **Developer experience** — `docs/API_QUICKSTART.md`, `docs/API_V1.md`,
  `docs/WEBSOCKET_PROTOCOL.md`, and a dependency-free Node.js example client in
  `examples/node-api-client/`.
- **Smoke tests** — `scripts/smoke-api-v1.mjs`, `scripts/smoke-ws-token.mjs`,
  `scripts/smoke-rate-limit.mjs`, `scripts/smoke-lifecycle.mjs`.

---

## What is intentionally NOT included in this phase

- Billing, plans, or production quota accounting.
- Customer dashboard or self-serve key management.
- Avatar/background CRUD upload pipeline.
- Cloudflare Stream live integration.
- Cold-start / GPU snapshot research.

These remain on the long-term roadmap and will be picked up in their own phases.

---

## Known external blocker

- **Modal backend/workspace outage** may still block full `/start` happy-path
  validation. The API layer is correct and deployed; end-to-end media tests will
  resume once Modal infrastructure is restored.

---

## Recommended next roadmap phase

**Avatar/background platform APIs** — CRUD for avatars, backgrounds, and session
config, plus storage/retrieval plumbing.

(Alternative: multi-avatar asset/cache lifecycle, depending on roadmap
prioritization.)

---

## Explicit decision

> Do **not** keep expanding the API foundation phase. Only reopen it for a
> confirmed security issue, ownership bug, rate-limit bypass, or spec/runtime
> mismatch found by the drift guard.

---

## Validation

Run before any future release commit that touches the public API:

```bash
node scripts/validate-api-contract.mjs
```

At this checkpoint the script reports **PASS** and `npx tsc --noEmit` in
`avatar-saas-prototype/cloudflare-session-bridge` also passes.
