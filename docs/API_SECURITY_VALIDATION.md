# API Security Validation Runbook

Operational guide for configuring and validating multi-tenant API authentication
and cross-tenant session isolation on the Avatar Bridge Worker.

## Metadata Auth Configuration

### Overview

The Bridge Worker supports two API key modes:

| Mode | Secret | Tenant Isolation | Use Case |
|------|--------|------------------|----------|
| **Metadata** | `AVTR_API_KEY_METADATA_JSON` | Per-key `tenant_id` | Production / multi-tenant |
| **Legacy** | `AVTR_API_KEYS` | All keys share `tenant_id="legacy"` | Dev / single-tenant only |

Metadata mode is checked **first**. If a key matches in metadata, it resolves
with its configured `tenant_id`. If not, legacy keys are checked as fallback.

### Configuring Metadata Mode

Set `AVTR_API_KEY_METADATA_JSON` as a Cloudflare Worker secret (not in
`wrangler.jsonc` — secrets are write-only and never appear in source):

```bash
cd avatar-saas-prototype/cloudflare-session-bridge
npx wrangler secret put AVTR_API_KEY_METADATA_JSON
```

Paste a JSON object mapping each API key to its tenant metadata:

```json
{
  "avtr_live_REPLACE_ME_KEY_A": {
    "tenant_id": "tenant_a",
    "name": "Tenant A Production",
    "plan": "enterprise"
  },
  "avtr_live_REPLACE_ME_KEY_B": {
    "tenant_id": "tenant_b",
    "name": "Tenant B Staging",
    "plan": "starter"
  }
}
```

**Rules:**

- Each top-level key is the raw API key string.
- `tenant_id` must be a non-empty string (required).
- `name` and `plan` are optional (string or omitted).
- Entries with missing/empty `tenant_id` are silently dropped with a
  `console.warn` (never logging the raw key).
- Invalid JSON falls back to legacy mode — never 5xx.

Setting a secret creates a new Worker version and deploys immediately. Capture
the version ID via `npx wrangler deployments list`.

### Generating Strong API Keys

```bash
# Minimum: 24 bytes hex (192-bit)
echo "avtr_test_$(openssl rand -hex 24)"

# Stronger: 32 bytes hex (256-bit)
echo "avtr_live_$(openssl rand -hex 32)"
```

### Key Management Safety

- **Never** commit API keys to source, docs, or config files.
- **Never** paste full key values in terminal summaries, PostHog events, or
  issue trackers.
- **Never** write the metadata JSON to a file that could be committed. If a
  temp file is needed, use a gitignored directory and delete it immediately
  after `wrangler secret put` completes.
- Prefer interactive `wrangler secret put` (pasting via stdin) over shell
  piping to avoid shell history exposure.
- If using a temp file for the JSON payload, verify deletion with `ls`.

### Verifying Metadata Mode

```bash
curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/me \
  -H "Authorization: Bearer $AVTR_API_KEY"
```

**Expected (metadata mode):**

```json
{
  "success": true,
  "data": {
    "tenant_id": "tenant_a",
    "key_name": "Tenant A Production",
    "plan": "enterprise",
    "auth_mode": "metadata"
  }
}
```

If `auth_mode` is `"legacy"`, the key is not in the metadata JSON (it resolved
via `AVTR_API_KEYS` fallback).

## Cross-Tenant Smoke Tests

### Prerequisites

- Two API keys configured in `AVTR_API_KEY_METADATA_JSON` with **different**
  `tenant_id` values.
- Rate limit window clear (wait 70 seconds since last smoke run).

### Running

```bash
# Single-tenant regression (always runs):
AVTR_API_KEY=<key_a> node scripts/smoke-api-v1.mjs

# Cross-tenant isolation (requires AVTR_API_KEY_B):
AVTR_API_KEY=<key_a> AVTR_API_KEY_B=<key_b> node scripts/smoke-ws-token.mjs

# Rate limit enforcement (wait 70s after):
AVTR_API_KEY=<key_a> node scripts/smoke-rate-limit.mjs

# Lifecycle hardening (wait 70s after rate-limit test):
AVTR_API_KEY=<key_a> node scripts/smoke-lifecycle.mjs
```

### Pass Criteria

All of the following must be true before promoting to production:

| Check | Expected |
|-------|----------|
| `/v1/me` key A | `auth_mode: "metadata"`, `tenant_id: "tenant_a"` |
| `/v1/me` key B | `auth_mode: "metadata"`, `tenant_id: "tenant_b"` |
| `/v1/me` response | No raw API key value anywhere |
| Cross-tenant SKIP count | **0** (all cross-tenant tests must run) |
| B → A session status | 404 `not_found` |
| B → A ws-token refresh | 404 `not_found` |
| B → A session DELETE | 404 `not_found` |
| B creates + accesses own session | 200 |
| A → B session status | 404 `not_found` |
| Demo session via v1 API | 404 `not_found` |
| Nonexistent session | 404 `not_found` |
| Cleanup (DELETE) exempt from rate limits | No 429 |
| `ownership_check_failed` in PostHog | `reason: "tenant_mismatch"` events exist |
| PostHog failure telemetry | `callerTenantId` present, stored owner `tenant_id` absent |
| PostHog events | No raw API keys in any property |
| No new `pipeline_error` | Zero events caused by validation |

### Rate Limit Timing

Rate limits are per-tenant and per-Cloudflare-location with a 60-second window.
To avoid false failures:

1. Wait 70 seconds before the first smoke run.
2. Run `smoke-rate-limit.mjs` **last** (it intentionally exhausts limits).
3. Wait 70 seconds after `smoke-rate-limit.mjs` before any other test.

## Legacy Mode Warning

**Legacy mode (`AVTR_API_KEYS` only) is NOT production tenant isolation.**

When using legacy mode, all API keys share `tenant_id = "legacy"`. Any legacy
key can access sessions created by any other legacy key. This is acceptable
for:

- Local development
- Single-tenant staging
- Backward compatibility with existing integrations

Production multi-tenant deployments **must** use `AVTR_API_KEY_METADATA_JSON`
with distinct `tenant_id` values per key.

## WS Token Security

The WebSocket session token (`?token=...` in the `ws_url`) is a **short-lived
bearer credential** (5-minute TTL, HMAC-SHA256 signed).

### Security properties

- Token binds to a specific session ID (`sid`) — cannot be reused on another
  session.
- Token includes tenant ID (`tid`) for audit/telemetry.
- Token is minted by the Worker only (callers cannot self-mint).
- Token is verified before ownership check on the WS upgrade path.

### Bearer credential warning

Anyone holding a valid WS token can connect to the corresponding WebSocket as
the token's tenant. This is by design — the token **is** the authorization.

**Do NOT:**
- Log `ws_url` (contains token) in browser console, analytics, or error
  reports.
- Store `ws_url` in browser history or localStorage beyond the TTL.
- Share `ws_url` between users or tenants.

**Do:**
- Use the `ws_url` immediately to open the WebSocket connection.
- Call `POST /v1/sessions/:id/ws-token` for reconnects after TTL expiry.
- Treat the token with the same care as the API key itself.

If a token is compromised, the 5-minute TTL is the revocation window. For
longer-lived credential compromise, rotate `AVTR_SESSION_TOKEN_SECRET`
(invalidates all outstanding tokens).

## PostHog Verification

After running cross-tenant smoke tests, verify in PostHog:

```bash
source .env.posthog

# Check ownership_check_failed events
./scripts/posthog-query.sh search ownership_check_failed 20

# Or direct HogQL query for cross-tenant failures:
curl -s "${POSTHOG_API_HOST}/api/projects/${POSTHOG_PROJECT_ID}/query/" \
  -H "Authorization: Bearer ${POSTHOG_PERSONAL_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": {
      "kind": "HogQLQuery",
      "query": "SELECT timestamp, distinct_id, properties FROM events WHERE event = '\''ownership_check_failed'\'' AND timestamp > now() - interval 30 minute ORDER BY timestamp DESC LIMIT 20"
    }
  }' | python3 -m json.tool
```

### What to check

1. **Cross-tenant denial events exist** — `reason: "tenant_mismatch"` with
   `callerTenantId` of the denied tenant.
2. **No stored owner leak** — `ownership_check_failed` properties must NOT
   contain the stored session owner's `tenant_id`. Only the caller's
   `callerTenantId` is included.
3. **No raw API keys** — scan all event properties for `avtr_test_` or
   `avtr_live_` prefixes. None should appear.
4. **No new pipeline errors** — zero `pipeline_error` or
   `modal_session_fmp4_stream_failed` events caused by the validation.

## Adding a New Tenant

1. Generate a strong API key:
   ```bash
   echo "avtr_live_$(openssl rand -hex 32)"
   ```

2. Add the key to the metadata JSON (read current value is not possible —
   rebuild from known keys):
   ```bash
   npx wrangler secret put AVTR_API_KEY_METADATA_JSON
   # Paste updated JSON with all existing + new keys
   ```

3. Verify with `/v1/me`:
   ```bash
   curl https://avatar-session-bridge.vallangirakesh.workers.dev/v1/me \
     -H "Authorization: Bearer <new_key>"
   ```

4. Confirm `auth_mode: "metadata"` and correct `tenant_id`.

5. Run smoke tests with the new key to verify isolation.

## Secret Rotation

### Rotating an API key

1. Generate a new key.
2. Add it to `AVTR_API_KEY_METADATA_JSON` (alongside the old key temporarily).
3. Verify the new key works via `/v1/me`.
4. Remove the old key from the metadata JSON and push again.
5. The old key immediately stops working.

### Rotating the WS token secret

```bash
openssl rand -hex 32 | npx wrangler secret put AVTR_SESSION_TOKEN_SECRET
```

This invalidates **all** outstanding WS tokens. Active WebSocket connections
are not affected (already upgraded). New connections require a fresh token via
`POST /v1/sessions/:id/ws-token`.
