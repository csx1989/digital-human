# Internal Modal Endpoint Authentication — Phase Closeout

## Verdict

**Production-complete for the current fMP4 Worker->Modal path.**

This document closes out roadmap row #11 (`docs/AVATAR_PLATFORM_ROADMAP.md`).
No runtime code changes were made as part of this closeout — only status
documentation.

---

## Final State

| Item | Value |
|------|-------|
| Enforce flag | `AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true` (active on Modal 5Q) |
| Shared secret | `avtr-modal-request-signing-secret` (Modal secret) / `AVTR_MODAL_REQUEST_SIGNING_SECRET` (Worker env) |
| Signing algorithm | HMAC-SHA256 |
| Signature base | `METHOD\nURL\nTIMESTAMP\nREQUEST_ID\nBODY_SHA256` |
| Allowed clock skew | 300 seconds (`ALLOWED_SKEW_SECONDS`) |
| Modal app | `avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s-5q` |
| Commit chain | `c1915bc` -> `36910e2` -> `b3c71d9` -> `d123168` -> `dc05c33` |

### What is enforced

- The Cloudflare Worker signs every active Worker->Modal HTTP request with
  four headers: `X-AVTR-Timestamp`, `X-AVTR-Request-Id`,
  `X-AVTR-Body-SHA256`, `X-AVTR-Signature`.
- The Modal 5Q app calls `verify_modal_request()` on every public endpoint.
- Enforce mode is **on**: requests with missing headers, stale timestamps
  (>300s skew), or signature mismatch receive HTTP 401.
- No secrets, signatures, request bodies, or WS tokens are logged. Only the
  sanitized endpoint label and request-id appear in auth log lines.

### What is explicitly out of scope

- **OmniVoice `/web`** — signing code exists in `modal_omnivoice_tts_app.py`
  and `tts-provider.ts`, but the endpoint is config-gated off
  (`ENABLE_OMNIVOICE_TTS_ON_L4=false`) and **not deployed**. It will be
  revisited only if `ENABLE_OMNIVOICE_TTS_ON_L4` is enabled.
- **SFU transport** — SFU remains disabled
  (`AVTR_REALTIME_SFU_ENABLED=false`). Production transport is fMP4 over
  WebSocket. SFU-related Modal paths are not part of this phase.
- **Replay cache** — deferred. The current threat model (server-to-server
  over TLS with a 300-second timestamp window) makes a replay cache
  acceptable. See Future Work below.

---

## Endpoint Coverage

| # | Endpoint | Status |
|---|----------|--------|
| 1 | `start-session-5q` | Active, signed + verified |
| 2 | `session-status-5q` | Active, signed + verified |
| 3 | `push-audio-batch-5q` | Active, signed + verified |
| 4 | `end-session-adapter-5q` | Active, signed + verified |
| 5 | `stream-fmp4-av-adapter-5q` | Active, signed + verified |
| 6 | `tts-same-gpu-parallel-5q` | Active, signed + verified |
| 7 | `push-audio-5q` | Dead code (signed + verified if called, never called) |
| 8 | `list-cached-avatars-5q` | **Removed** — was the only unsigned public endpoint; no Worker caller existed |
| 9 | OmniVoice `/web` | Config-gated off, not deployed (out of scope) |

---

## Validation Summary

All validation was run post-enforcement with `dc05c33` deployed.

| Test | Result |
|------|--------|
| pytest (11 Python signing tests) | 11 passed |
| TS signing tests (14 tests) | 14 passed |
| smoke-api-v1 | 20 PASS, 0 FAIL |
| smoke-ws-token | 13 PASS, 0 FAIL, 1 SKIP |
| smoke-lifecycle | 15 PASS, 0 FAIL |
| Speech/fMP4 pipeline | 294 segments, `modalEndOk=true`, 0 pipeline_error |
| Negative: unsigned direct Modal POST | 401 `missing_headers` |
| Negative: tampered signature | 401 `signature_mismatch` |
| Negative: removed `list-cached-avatars-5q` | 404 |
| Modal `[modal-auth] invalid` count | 0 |
| PostHog `pipeline_error` | 0 |
| PostHog SFU events | 0 |

---

## Rollback Procedure

If enforcement causes issues (e.g., clock drift, signing header breakage):

1. Set `AVTR_ENFORCE_MODAL_REQUEST_SIGNING=false` in the Modal secret
   `avtr-modal-request-signing-secret`.
2. Redeploy Modal 5Q (`modal deploy ...5q.py`). Verification runs in
   observe mode (logs only, no rejects).
3. Worker rollback is **not needed** unless the signing headers themselves
   are broken — the Worker always sends the headers; enforcement is purely
   a Modal-side gate.

To fully disable signing (emergency only): remove the four `X-AVTR-*`
header calls from `buildSignedHeaders()` in
`avatar-saas-prototype/cloudflare-session-bridge/src/modal-request-signing.ts`
and redeploy the Worker. This is not recommended.

---

## Future Work

1. **Replay cache** — Currently deferred. A short-lived nonce/request-id
   cache would prevent replay within the 300-second skew window. Acceptable
   to skip for the current server-to-server TLS threat model. Reconsider if
   the threat model expands (e.g., if Modal endpoints become reachable from
   untrusted networks).

2. **Remove dead `push-audio-5q`** — The endpoint and its Worker-side
   caller (`pushModalAudio` in `modal-client.ts`) are defined but never
   invoked. Safe to remove once confirmed unused long-term.

3. **Revisit OmniVoice `/web` signing** — Only if
   `ENABLE_OMNIVOICE_TTS_ON_L4` is enabled for production. Signing code
   already exists and is tested; would need deployment + enforcement
   validation.

4. **SFU transport auth** — Out of scope while SFU is disabled. If SFU is
   enabled, any new Modal publisher/callback endpoints must be added to
   the signed+verified set before going live.

---

## Implementation Files

| File | Role |
|------|------|
| `avatar-saas-prototype/modal-avatar-runtime/request_signing.py` | Modal-side verification helper (`verify_modal_request`, `sign_request`, sanitize helpers) |
| `avatar-saas-prototype/modal-avatar-runtime/test_request_signing.py` | 11 Python unit tests |
| `avatar-saas-prototype/cloudflare-session-bridge/src/modal-request-signing.ts` | Worker-side signing helper (`buildSignedHeaders`) |
| `avatar-saas-prototype/cloudflare-session-bridge/test/modal-request-signing.test.ts` | 14 TypeScript unit tests |
| `avatar-saas-prototype/cloudflare-session-bridge/src/modal-client.ts` | All Worker->Modal fetches signed |
| `avatar-saas-prototype/cloudflare-session-bridge/src/tts-provider.ts` | OmniVoice fetches signed (config-gated off) |
