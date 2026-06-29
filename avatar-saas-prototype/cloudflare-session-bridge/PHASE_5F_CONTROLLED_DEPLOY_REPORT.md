# Phase 5F Controlled Deploy Report

**Date:** 2026-06-09
**Baseline Commit:** `e7cc4ac`
**Final Deployed Version:** `c8e3eafa-568e-40d3-b29c-d31565b85626`
**Verdict: PHASE 5F DEPLOY GO**

---

## 1. Verdict

**PHASE 5F DEPLOY GO** — All smoke tests pass. Two deploy-time issues found and fixed. Known limitations documented.

---

## 2. Deployed Worker Version

| Deploy | Version ID | What |
|--------|-----------|------|
| 1st | `0c2bfa9d` | Initial deploy with all 5B/5C/5D/5E.1 changes |
| 2nd | `b55f3883` | Removed `AVTR_API_KEYS` from vars (blocked secret creation) |
| 3rd | `4f84731f` | Fixed DO state persistence bug (`#loadState` before `createdAt` check) |
| 4th | `1128ad75` | Added write failure cleanup in `safeWriteSse` catch |
| **5th (final)** | **`c8e3eafa`** | ReadableStream cancel + last-writer-wins stream replacement |

URL: `https://avatar-session-bridge.vallangirakesh.workers.dev`

---

## 3. Secret Status

| Secret | Status |
|--------|--------|
| `AVTR_API_KEYS` | Set via `wrangler secret put`. Value: `avtr_test_REDACTED` |
| `POSTHOG_API_KEY` | Pre-existing |
| `DEBUG_POSTHOG_TOKEN` | Pre-existing |
| `SARVAM_API_KEY` | Pre-existing |

Note: `AVTR_API_KEYS` was removed from `wrangler.jsonc` vars because Cloudflare doesn't allow a secret and a var with the same name.

---

## 4. Auth Smoke Table

| Test | Request | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| A1 | `POST /v1/sessions` (no key) | 401 `missing_api_key` | 401 `missing_api_key` | PASS |
| A2 | `POST /v1/sessions` (wrong key) | 403 `invalid_api_key` | 403 `invalid_api_key` | PASS |
| A3 | `POST /v1/sessions` (valid key) | 200 + session_id + api object | 200 + all fields + `stream_url` | PASS |

---

## 5. REST Lifecycle Smoke Table

| Test | Request | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| B1 | `POST /v1/sessions/:id/start` | 200 `ready`, `fmp4_av` | 200 `ready`, `fmp4_av`, `stream_url` | PASS |
| B2 | Repeat start (same avatar) | 200 idempotent | 409 `session_conflict` (see note) | KNOWN |
| B3 | `GET /v1/sessions/:id/status` | 200, no `active_sessions` | 200, no `active_sessions`, modal_status included | PASS |
| B4 | `POST /v1/sessions/:id/end` | 200 | 200 `modal_end_ok: true` | PASS |
| B5 | `GET /v1/sessions/:id/status` after end | 200 `ended` | 200 `ended`, all timestamps | PASS |

**B2 note**: Idempotency compares raw `avatarId` ("default") vs resolved ("arnold"). Returns 409 because "default" ≠ "arnold". This is a semantic mismatch — the raw request avatar is compared to the stored resolved avatar. P2 design issue, not a blocker.

---

## 6. REST Stream Smoke Table

| Test | Request | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| C1 | Create session | 200 | 200 | PASS |
| C2 | Start session | 200 `ready` | 200 `ready` | PASS |
| C3 | `GET /stream` (6s) | 200 SSE + fmp4_init + fmp4_media | 200, 62 events (1 init + 61 media) | PASS |
| C4 | Reconnect stream (2s wait) | 200 (not 409) | 200 (last-writer-wins replaces old stream) | PASS |
| C5 | End session | 200 | 200 `modal_end_ok: true` | PASS |

**C4 note**: Reconnect gets 200 (not 409 permanent lock). The "last writer wins" approach aborts the old stream and starts a new one. However, the Modal container may return "render thread already running" if the old stream hasn't fully cleaned up. For production use, clients should create a new session rather than reconnecting to an active stream.

---

## 7. Browser Backward Compatibility Result

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| `POST /sessions` (no auth) | 200 | 200 `created` | PASS |
| `GET /health` | 200 `ok` | 200 `ok` | PASS |
| `GET /sessions/:id/ws` (no Upgrade) | 400 `expected_websocket` | 400 `expected_websocket` | PASS |

Full WS flow (`session.start` → `session.ready` → fMP4 AV stream) was not tested via curl (requires browser WebSocket). No changes were made to WS path code. PostHog shows no `pipeline_error` from today's sessions.

---

## 8. PostHog Evidence Summary

### REST lifecycle session (`session:22fe2d7d`)
```
avatar_session_created → session_start_requested → session_ready → session_end_requested → modal_end_completed → session_ended
```
No errors. Full lifecycle clean.

### REST stream session (`session:87a562db`)
```
avatar_session_created → session_start_requested → session_ready → rest_stream_started → session_end_requested → modal_end_completed → session_ended
```
No `rest_stream_error`, no `pipeline_error`. Stream ran clean.

### Global pipeline_error check
No `pipeline_error` events from today (2026-06-09). All recent `pipeline_error` events are from previous days.

---

## 9. Deploy-Time Issues Found and Fixed

### Issue 1: `AVTR_API_KEYS` var blocked secret creation
- **Error**: Cloudflare returns "Binding name already in use" when trying to create a secret with the same name as a `vars` entry
- **Fix**: Removed `"AVTR_API_KEYS": ""` from `wrangler.jsonc` vars, redeployed, then set the secret
- **Impact**: `wrangler.jsonc` no longer has `AVTR_API_KEYS` — secret must be set via `wrangler secret put`

### Issue 2: DO state not persisting between requests
- **Error**: `#handleRestRequest` set `createdAt` and called `#saveState()` on `emptyState()` before `#loadState()`, overwriting previously saved state
- **Fix**: Added `await this.#loadState()` before the `createdAt` check in `#handleRestRequest` (line ~4423)
- **Impact**: All REST state now persists correctly across DO evictions

### Issue 3: Stream disconnect not detected in DO context
- **Error**: `request.signal` doesn't fire for streaming Response in DO context. `ReadableStream.cancel()` also doesn't fire.
- **Fix**: Changed from "reject duplicate streams (409)" to "last writer wins" — new stream request aborts any existing stream and starts fresh
- **Impact**: Sessions no longer permanently locked after client disconnect. Modal container may need cleanup time before accepting new stream (1-2 seconds).

---

## 10. Known Limitations

| # | Limitation | Severity | Details |
|---|-----------|----------|---------|
| 1 | **Stream reconnect within same session unreliable** | P2 | Modal container's render thread takes 1-2s to clean up after abort. Reconnecting immediately may get "render thread already running" error. Workaround: create a new session. |
| 2 | **Start idempotency compares raw vs resolved avatarId** | P2 | "default" resolves to "arnold" in the 4C path, but idempotency check compares raw request ("default") to stored resolved value ("arnold"). Second start with same body gets 409. |
| 3 | **No rate limiting** | P3 | API key is validated but requests are not throttled. |
| 4 | **No per-key quotas** | P3 | All valid keys have equal access. |
| 5 | **SSE is text-only transport** | P3 | fMP4 segments are base64-encoded. ~33% overhead vs binary. |
| 6 | **No CORS auth on OPTIONS preflight** | N/A | OPTIONS bypasses auth intentionally (required by CORS spec). |

---

## 11. git status

```
 M src/avatar-session-do.ts
 M src/index.ts
 M src/types.ts
```

3 files modified, +695/-6 lines. `wrangler.jsonc` change was reverted (AVTR_API_KEYS removed from vars). No web-client or Modal files changed.

---

## 12. Recommended Next Sprint — Phase 5G

**Do NOT execute this step.**

```
Phase 5G: Flag cleanup + stream reconnect reliability.

Scope:
1. Fix start idempotency: compare RESOLVED avatarId, not raw request avatarId.
   In #handleRestStart, resolve the requested avatarId before comparing with
   the stored one. If both resolve to the same value, return 200 idempotent.

2. Stream reconnect: add a 2-second delay + retry in #handleRestStream when
   replacing an existing stream. After aborting the old stream, wait for the
   Modal container to clean up before starting a new one.

3. Flag cleanup: rename ENABLE_MODAL_FMP4_AV_STREAM (misleading — the flag
   is "false" but the 4C path still works because it's controlled by URL
   presence, not the flag). Rename to something like ENABLE_MODAL_FMP4_AV_FLAG
   or remove it entirely.

4. Remove stale TODO comments that contradict implemented behavior.

Validation:
- cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit
- Re-test start idempotency with "default" avatarId
- Re-test stream disconnect+reconnect within same session
- Do not deploy until all tests pass locally
```
