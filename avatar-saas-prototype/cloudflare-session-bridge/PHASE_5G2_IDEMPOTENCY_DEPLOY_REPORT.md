# Phase 5G.2 Idempotency Deploy Report

**Date:** 2026-06-09
**Baseline Commit:** `68a048e`
**Deployed Version:** `fc0c1428-daea-4633-bfc2-5c4ffa87c049`
**Verdict: PHASE 5G.2 DEPLOY GO**

---

## 1. Verdict

**PHASE 5G.2 DEPLOY GO** — Idempotency fix verified. All smokes pass. No regression.

---

## 2. Deployed Version

| Attribute | Value |
|-----------|-------|
| Worker URL | `https://avatar-session-bridge.vallangirakesh.workers.dev` |
| Version ID | `fc0c1428-daea-4633-bfc2-5c4ffa87c049` |
| Upload Size | 493.27 KiB / gzip: 135.63 KiB |

---

## 3. Files Changed (Phase 5G.1)

| File | Delta | Description |
|------|-------|-------------|
| `src/avatar-session-do.ts` | +4/-3 | Idempotency resolves avatarId before comparison |
| `src/types.ts` | +3/-1 | Clarifying comment on `ENABLE_MODAL_FMP4_AV_STREAM` |
| `wrangler.jsonc` | +1/-0 | Clarifying comment on flag |

---

## 4. Smoke Tests

### A. Auth

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| No key | 401 `missing_api_key` | 401 `missing_api_key` | PASS |
| Wrong key | 403 `invalid_api_key` | 403 `invalid_api_key` | PASS |
| Valid key | 200 + session_id | 200 + session_id + api object | PASS |

### B. Idempotency (`session:3240fbf8`)

| Test | Request | Expected | Actual | Status |
|------|---------|----------|--------|--------|
| B2 | Start `"default"` | 200 `ready`, `avatar_id: "arnold"` | 200 `ready`, `avatar_id: "arnold"` | PASS |
| B3 | Repeat `"default"` | **200 idempotent** | **200 `idempotent_start`** | **PASS (was 409 before fix)** |
| B4 | Start `"arnold"` | 200 idempotent | 200 `idempotent_start` | PASS |
| B5 | Start `"sarah"` | 409 conflict | 409 `session_conflict` (resolved: sarah, raw: sarah) | PASS |
| B6 | Status | 200, no `active_sessions` | 200, no `active_sessions`, `modal_status` included | PASS |
| B7 | End | 200 | 200 `modal_end_ok: true` | PASS |

### C. Stream Regression (`session:9d1b3923`)

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| Content-Type | `text/event-stream` | `text/event-stream` | PASS |
| fmp4_init | 1 event | 1 event | PASS |
| fmp4_media | Multiple | 63 events | PASS |
| Total events (6s) | — | 64 | PASS |
| Reconnect | 200 (not 409) | 200 | PASS |
| End | 200 | 200 `modal_end_ok: true` | PASS |

### D. Legacy Routes

| Test | Expected | Actual | Status |
|------|----------|--------|--------|
| `GET /health` | 200 `ok` | 200 `ok` | PASS |
| `POST /sessions` (no key) | 200 | 200 `created`, has `wsUrl` | PASS |

---

## 5. No Modal/Web-Client Deploy

Only Bridge Worker deployed. Modal and web-client untouched.

---

## 6. Commit

If all smokes pass, commit with: `Phase 5 API idempotency cleanup`

Files to stage:
- `src/avatar-session-do.ts`
- `src/types.ts`
- `wrangler.jsonc`
- `PHASE_5G1_API_CORRECTNESS_REPORT.md`
- `PHASE_5G2_IDEMPOTENCY_DEPLOY_REPORT.md`
