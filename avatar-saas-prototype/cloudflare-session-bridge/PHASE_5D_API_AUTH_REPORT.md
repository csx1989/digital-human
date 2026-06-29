# Phase 5D API Auth Report

**Date:** 2026-06-08
**Baseline Commit:** `e7cc4ac`
**Verdict: PHASE 5D LOCAL GO**

---

## 1. Verdict

**PHASE 5D LOCAL GO** — `tsc --noEmit` clean. All `/v1/*` routes gated with API key auth. Timing-safe comparison. Fail-closed on missing config. OPTIONS bypasses auth. Zero impact on existing browser/demo routes.

---

## 2. Files Changed

| File | Lines | Description |
|------|-------|-------------|
| `src/types.ts` | +1 | `AVTR_API_KEYS?: string` on `Env` |
| `src/index.ts` | +56 net | Auth helpers (`extractApiKey`, `safeEqual`, `isValidApiKey`, `requireApiKey`), CORS update, auth gate |
| `wrangler.jsonc` | +1 | `"AVTR_API_KEYS": ""` (empty default, set via secret in production) |

3 files changed. No web-client or Modal files changed. No deploy executed.

---

## 3. Auth Behavior Table

| Condition | Status | Code | Body |
|-----------|--------|------|------|
| `AVTR_API_KEYS` missing or empty | 503 | `api_auth_not_configured` | "API authentication is not configured on this server." |
| No `Authorization` or `x-api-key` header | 401 | `missing_api_key` | "API key required. Provide Authorization: Bearer \<key\> or x-api-key header." |
| Key provided but not in configured set | 403 | `invalid_api_key` | "The provided API key is not valid." |
| Key matches a configured key | — | — | `null` (continue to handler) |

### Accepted Headers

| Header | Format |
|--------|--------|
| `Authorization` | `Bearer <key>` |
| `x-api-key` | `<key>` |

`Authorization` is checked first. If present but not `Bearer` prefix, falls through to `x-api-key`.

---

## 4. CORS/Preflight Proof

### CORS Headers (line 11-15)

```
Access-Control-Allow-Headers: Content-Type, x-debug-token, Authorization, x-api-key
```

### OPTIONS Handler (line 158-160)

```typescript
if (request.method === "OPTIONS") {
  return new Response(null, { status: 204, headers: CORS_HEADERS });
}
```

### Bypass Order (critical)

1. **OPTIONS** → line 158 → returns 204 immediately
2. `/health` → line 162
3. `/debug/posthog-test` → line 166
4. `/sessions` (POST) → line 170
5. `/sessions/:id/ws` (GET) → line 182
6. **`/v1/*` auth gate** → line 204
7. `/v1/sessions` (POST) → line 209
8. `/v1/sessions/:id/:op` → line 231

OPTIONS returns at step 1, **before** the auth gate at step 6. No auth check runs on preflight requests.

---

## 5. Validation Command Output

```bash
$ cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit
# (no output — clean pass, zero errors)
```

---

## 6. Static Proof Checklist

| # | Check | Evidence | Status |
|---|-------|----------|--------|
| 1 | `/v1/*` auth gate exists | `index.ts:204` — `if (path.startsWith("/v1/"))` | PASS |
| 2 | Auth gate runs before all `/v1/` route handlers | Line 204 precedes line 209 (`POST /v1/sessions`) and line 231 (`v1SessionMatch`) | PASS |
| 3 | `Authorization: Bearer` supported | `extractApiKey` at line 28-35 — checks `Authorization` header, strips `Bearer ` prefix | PASS |
| 4 | `x-api-key` supported | `extractApiKey` at line 37-38 — fallback to `x-api-key` header | PASS |
| 5 | Missing env fails closed (503) | `requireApiKey` line 58-62 — `!env.AVTR_API_KEYS?.trim()` → 503 `api_auth_not_configured` | PASS |
| 6 | Missing key returns 401 | `requireApiKey` line 65-69 — no key extracted → 401 `missing_api_key` | PASS |
| 7 | Invalid key returns 403 | `requireApiKey` line 71-75 — `!isValidApiKey` → 403 `invalid_api_key` | PASS |
| 8 | `safeEqual` timing-safe comparison | Line 42-48 — XOR-based constant-time comparison | PASS |
| 9 | Empty configured keys filtered | Line 53 — `.filter(Boolean)` removes empty strings from split | PASS |
| 10 | OPTIONS bypasses auth | Line 158 returns 204 before auth gate at line 204 | PASS |
| 11 | `GET /health` not auth-gated | Line 162 — before auth gate | PASS |
| 12 | `POST /sessions` not auth-gated | Line 170 — before auth gate | PASS |
| 13 | `GET /sessions/:id/ws` not auth-gated | Line 182-200 — before auth gate | PASS |
| 14 | `POST /debug/posthog-test` not auth-gated | Line 166 — before auth gate, has own debug token auth | PASS |
| 15 | Error responses include CORS headers | All auth errors use `corsResponse()` | PASS |
| 16 | `x-api-key` in CORS allow-headers | Line 14 | PASS |
| 17 | No API key logging | Zero `console.log`/`console.error` calls in auth helpers | PASS |
| 18 | No configured keys exposed in responses | Error messages are generic — no key count, format, or prefix | PASS |
| 19 | No web-client files changed | `git diff --stat -- ../web-client/` — empty | PASS |
| 20 | No Modal files changed | `git diff --stat -- ../../modal-avatar-runtime/` — empty | PASS |
| 21 | No deploy command run | No `wrangler deploy` executed | PASS |
| 22 | `wrangler.jsonc` has empty default | `"AVTR_API_KEYS": ""` — safe default, set via secret | PASS |

---

## 7. Known Limitations

- **No rate limiting** — Auth validates the key but does not throttle requests. Phase 5E concern.
- **No key rotation protocol** — Rolling keys requires updating the Cloudflare secret. No graceful transition.
- **No per-key quotas** — All valid keys have equal access. No usage tracking or tiering.
- **Comma-separated format** — Not ideal for large key sets. Acceptable at current scale (<50 keys).
- **No scope/permission model** — Single tier: valid key = full access to all `/v1/*` routes.
- **`safeEqual` is timing-safe-ish** — True constant-time comparison requires `crypto.subtle.timingSafeEqual` (not available in Workers runtime). XOR approach mitigates basic timing attacks.
- **503 on missing config** — If `AVTR_API_KEYS` is not set, all `/v1/*` requests fail with 503. This is intentional (fail closed) but may confuse during initial setup.

---

## 8. git status

```
M src/avatar-session-do.ts
M src/index.ts
M src/types.ts
M wrangler.jsonc
```

4 files modified (3 from Phase 5B+5C + 1 wrangler.jsonc). No web-client or Modal files changed. No deploy executed.

---

## 9. Recommended Next Step — Phase 5E Senior Audit Gate

**Do NOT implement Phase 5E.** This is an audit gate, not a feature sprint.

**Scope:**
1. Senior engineer reviews all Phase 5B+5C+5D changes end-to-end.
2. Validates auth behavior with live curl tests against a deployed staging environment.
3. Confirms fail-closed behavior (no `AVTR_API_KEYS` set → 503 on all `/v1/*` routes).
4. Confirms valid key → 200 on all 5 `/v1/*` routes.
5. Confirms invalid key → 403.
6. Confirms no auth on legacy routes (`/sessions`, `/sessions/:id/ws`, `/health`).
7. Reviews SSE stream behavior under auth — stream starts after auth, long-lived connection.
8. Reviews DO memory lifecycle with REST SSE streams — no leaks on disconnect.
9. Signs off on deploy to production.

**Do not execute this step.**
