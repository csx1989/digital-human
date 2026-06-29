# Phase 5G.1 API Correctness Report

**Date:** 2026-06-09
**Baseline Commit:** `68a048e`
**Verdict: PHASE 5G.1 LOCAL GO**

---

## 1. Verdict

**PHASE 5G.1 LOCAL GO** — Start idempotency fixed. Flag audit complete (flag not renamed — runtime risk deferred). No deploy.

---

## 2. Files Changed

| File | Delta | Description |
|------|-------|-------------|
| `src/avatar-session-do.ts` | +4/-3 | Idempotency now resolves avatarId before comparison |
| `src/types.ts` | +3/-1 | Clarifying comment on `ENABLE_MODAL_FMP4_AV_STREAM` |
| `wrangler.jsonc` | +1/-0 | Clarifying comment on flag |

**3 files, +8/-4 lines.** No Modal, web-client, or index.ts changes.

---

## 3. Idempotency Fix

### Problem
`POST /v1/sessions/:id/start` compared raw `requestedAvatarId` against stored `avatarId`:
- Request: `avatarId = "default"`
- Stored (resolved by Modal): `avatarId = "arnold"`
- Comparison: `"default" !== "arnold"` → 409 `session_conflict`

This was a false conflict because "default" resolves to "arnold" via the same resolution logic used in the actual start path.

### Fix
`avatar-session-do.ts:4474-4477` — Resolve requested avatarId before comparison:

```typescript
const rawRequested = body.avatarId || "default";
const resolvedRequested = rawRequested === "default" ? "arnold" : rawRequested;
const currentAvatarId = this.#state.avatarId || "default";
if (resolvedRequested === currentAvatarId) {
```

This applies the same resolution used in the 4C start path (`avatar-session-do.ts:4523-4525`).

### Behavior Matrix

| Request avatarId | Stored avatarId | Resolved request | Result |
|------------------|-----------------|------------------|--------|
| (missing) | "arnold" | "arnold" | 200 idempotent |
| "default" | "arnold" | "arnold" | 200 idempotent |
| "arnold" | "arnold" | "arnold" | 200 idempotent |
| "sarah" | "arnold" | "sarah" | 409 conflict |
| "default" | "sarah" | "arnold" | 409 conflict |

### WS Path Impact
None. The idempotency check is only in `#handleRestStart`. WS start uses `#handleSessionStart` which has separate logic.

---

## 4. Flag/Comment Audit

### `ENABLE_MODAL_FMP4_AV_STREAM`

**Value**: `"false"` in `wrangler.jsonc`
**Active runtime usages**: 9 locations in `avatar-session-do.ts`

| Line | Usage | What it controls |
|------|-------|-------------------|
| 352 | `webSocketClose` | Which stop endpoint to call for stream cleanup |
| 1403 | `#handleAvatarTimeline` | `fmp4AvMode` flag for telemetry |
| 1494 | Speech pipeline | `isAvMode` for audio routing |
| 1542 | Speech pipeline | `isFmp4Mode` detection |
| 1544 | Speech pipeline | `isAvMode` for control endpoint |
| 3918 | WS start stream | `renderMode` selection |
| 3938 | WS start stream | Audio mode check |
| 4051 | WS stream stop | Control endpoint selection |
| 4653 | REST end | Control endpoint selection |

**Why not renamed/removed**: The flag is actively used in 9 runtime locations controlling WS-path behavior (stop endpoint routing, render mode, speech pipeline). Renaming requires updating all 9 locations and retesting both WS and REST paths. This is a runtime-risky change that should be done in a dedicated legacy-cleanup phase.

**Why it says "false" but 4C works**: The 4C REST path gates on URL presence (`MODAL_SESSION_STREAM_FMP4_AV_URL`), not this flag. The flag only controls WS-path-specific behavior.

**Action taken**: Added clarifying comments in `types.ts:22-24` and `wrangler.jsonc:25` explaining that the flag is WS-path only and REST/4C gates on URL presence instead.

### Stale TODOs/Comments
- **No TODO/FIXME/HACK** found in our source files (all hits were in `node_modules/`).
- Phase 4/5 section markers (`// Phase 4C:`, `// Phase 5B:`, etc.) are accurate and consistent with implemented behavior.
- No comments contradict implemented behavior.

---

## 5. What Was NOT Done (Deferred)

| Item | Reason |
|------|--------|
| Rename `ENABLE_MODAL_FMP4_AV_STREAM` | Runtime risk — 9 active usages. Defer to legacy-cleanup phase. |
| Add reconnect sleep/delay | Not approved. Fixed sleeps are a smell. Need evidence first. |
| Remove legacy code | Out of scope for this phase. |
| Deploy | Not approved. Local changes only. |

---

## 6. Validation

```
cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit → clean (zero errors)
```

---

## 7. Static Proof Checklist

| Check | Status |
|-------|--------|
| Idempotency compares resolved avatarId | PASS — `resolvedRequested` at line 4475 |
| Same "default"/"arnold" returns 200 | PASS — both resolve to "arnold" |
| Different avatar returns 409 | PASS — "sarah" vs "arnold" → conflict |
| WS path unchanged | PASS — only `#handleRestStart` modified |
| No reconnect sleep/delay added | PASS |
| No Modal files changed | PASS |
| No web-client files changed | PASS |
| No deploy command run | PASS |

---

## 8. Recommended Next Step (Do Not Execute)

```
Phase 5G.2: Controlled deploy + idempotency smoke test

Scope:
1. Deploy Bridge Worker.
2. Smoke test: create session, start with "default", start again with
   "default" → expect 200 idempotent (was 409 before fix).
3. Smoke test: create session, start with "default", start with "sarah"
   → expect 409 conflict.
4. Re-run full lifecycle + stream smoke.
5. Confirm no regression.

Validation:
- cd avatar-saas-prototype/cloudflare-session-bridge && npx tsc --noEmit
- Deploy + smoke
- PostHog event check
```
