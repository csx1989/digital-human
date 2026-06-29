#!/usr/bin/env node
// Deployed rate-limit smoke test.
//
// Verifies that the CF Rate Limiting bindings enforce per-tenant limits:
//   1. READ category: hammer /v1/health until 429 triggers
//   2. CREATE category: hammer POST /v1/sessions until 429 triggers
//   3. WS_CONNECT category: hammer GET /v1/sessions/:id/ws until 429 triggers
//   4. CLEANUP is exempt: DELETE all created sessions without hitting 429
//
// Cloudflare Rate Limiting uses approximate counting, so we do NOT fail if
// enforcement happens a few requests earlier or later than the configured
// limit. We only assert that a 429 eventually fires with code=rate_limited.
//
// IMPORTANT: Rate limits are per Cloudflare location per 60-second window.
// Wait at least 70 seconds between runs to avoid stale counters.
//
// Usage:
//   AVTR_API_KEY=... node scripts/smoke-rate-limit.mjs

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;

if (!API_KEY) {
  console.error("Set AVTR_API_KEY");
  process.exit(1);
}

const headers = { "Authorization": `Bearer ${API_KEY}` };

let pass = 0;
let fail = 0;
const createdSessions = [];

function record(label, ok, detail) {
  const tag = ok ? "PASS" : "FAIL";
  console.log(`  ${tag}  ${label}  ${detail ?? ""}`);
  if (ok) pass++; else fail++;
}

async function api(method, path, opts = {}) {
  const url = path.startsWith("http") ? path : `${BASE}${path}`;
  const res = await fetch(url, {
    method,
    headers: { ...headers, ...opts.headers },
    body: opts.body,
  });
  let body = null;
  try { body = await res.json(); } catch {}
  return { status: res.status, body, headers: res.headers };
}

// ── Test 1: READ category (RL_LIGHT, limit=120/60s) ─────────────────
async function testReadLimit() {
  console.log("\n--- Test 1: READ category (limit=120/60s) ---");
  const LIMIT = 120;
  const MAX_ATTEMPTS = 140;
  let ok200 = 0;
  let got429 = false;
  let errorCode = null;

  for (let i = 1; i <= MAX_ATTEMPTS; i++) {
    const { status, body } = await api("GET", "/v1/health");
    if (status === 200) {
      ok200++;
    } else if (status === 429) {
      got429 = true;
      errorCode = body?.error?.code ?? null;
      break;
    } else {
      record(`READ request #${i}`, false, `(unexpected HTTP ${status})`);
      return;
    }
  }

  record("READ: 429 eventually triggered", got429, `(${ok200} OK before 429)`);
  record("READ: error code = rate_limited", errorCode === "rate_limited", `(got=${errorCode})`);
}

// ── Test 2: session_create category (RL_WRITE, limit=20/60s) ────────
async function testCreateLimit() {
  console.log("\n--- Test 2: session_create category (limit=20/60s) ---");
  const LIMIT = 20;
  const MAX_ATTEMPTS = 40;
  let ok200 = 0;
  let got429 = false;
  let errorCode = null;
  let retryAfter = null;

  for (let i = 1; i <= MAX_ATTEMPTS; i++) {
    const { status, body, headers: resHeaders } = await api("POST", "/v1/sessions");
    if (status === 200) {
      ok200++;
      const sid = body?.data?.session_id;
      const wsUrl = body?.data?.stream?.ws_url;
      if (sid) createdSessions.push({ sid, wsUrl });
    } else if (status === 429) {
      got429 = true;
      errorCode = body?.error?.code ?? null;
      retryAfter = resHeaders.get("retry-after");
      break;
    } else {
      record(`CREATE request #${i}`, false, `(unexpected HTTP ${status})`);
      return;
    }
  }

  record("CREATE: 429 eventually triggered", got429, `(${ok200} OK before 429)`);
  record("CREATE: error code = rate_limited", errorCode === "rate_limited", `(got=${errorCode})`);
  record("CREATE: Retry-After header present", retryAfter !== null, `(Retry-After=${retryAfter})`);
}

// ── Test 3: ws_connect category (RL_HEAVY, limit=5/60s) ─────────────
async function testWsConnectLimit() {
  console.log("\n--- Test 3: ws_connect category (limit=5/60s) ---");
  const LIMIT = 5;
  const MAX_ATTEMPTS = 25;
  const entry = createdSessions[0];
  if (!entry?.wsUrl) {
    record("WS_CONNECT: no session with token available", false, "(setup failure)");
    return;
  }

  // Extract the WS upgrade path with token from the ws_url.
  // ws_url is like wss://host/v1/sessions/session:xxx/ws?token=yyy
  const wsUrlObj = new URL(entry.wsUrl);
  const wsPath = wsUrlObj.pathname + wsUrlObj.search;

  let passed = 0;
  let got429 = false;
  let errorCode = null;
  let retryAfter = null;

  for (let i = 1; i <= MAX_ATTEMPTS; i++) {
    // Do NOT send Upgrade: websocket — we want to test the rate limit gate,
    // not complete a WS handshake. Any non-429 response means rate limit passed.
    const res = await fetch(`${BASE}${wsPath}`, { headers });
    if (res.status !== 429) {
      passed++;
    } else {
      got429 = true;
      try { const b = await res.json(); errorCode = b?.error?.code ?? null; } catch {}
      retryAfter = res.headers.get("retry-after");
      break;
    }
  }

  record("WS_CONNECT: 429 eventually triggered", got429, `(${passed} passed before 429)`);
  record("WS_CONNECT: error code = rate_limited", errorCode === "rate_limited", `(got=${errorCode})`);
  record("WS_CONNECT: Retry-After header present", retryAfter !== null, `(Retry-After=${retryAfter})`);
}

// ── Test 4: CLEANUP is exempt (no 429 even after many DELETEs) ──────
async function testCleanupExempt() {
  console.log(`\n--- Test 4: CLEANUP exempt (${createdSessions.length} sessions) ---`);
  if (createdSessions.length === 0) {
    record("CLEANUP: no sessions to delete", false, "(setup failure)");
    return;
  }

  let deleted = 0;
  let got429 = false;

  for (const { sid } of createdSessions) {
    const { status } = await api("DELETE", `/v1/sessions/${encodeURIComponent(sid)}`);
    if (status === 200 || status === 404) {
      deleted++;
    } else if (status === 429) {
      got429 = true;
      break;
    }
  }

  record(`CLEANUP: deleted ${deleted}/${createdSessions.length} sessions`, deleted === createdSessions.length, "");
  record("CLEANUP: no 429 on any DELETE", !got429, "(cleanup is exempt)");
}

// ── Main ────────────────────────────────────────────────────────────
async function main() {
  console.log("+===============================================================+");
  console.log("|                Rate Limit Smoke Test (deployed)               |");
  console.log(`|  ${BASE}`);
  console.log("+---------------------------------------------------------------+");
  console.log("|  NOTE: This test sends 150+ requests. Wait 70s between runs.  |");
  console.log("|  Cloudflare Rate Limiting is approximate; we only assert that |");
  console.log("|  a 429 eventually fires with code=rate_limited.               |");

  await testReadLimit();
  await testCreateLimit();
  await testWsConnectLimit();
  await testCleanupExempt();

  console.log("+---------------------------------------------------------------+");
  console.log(`|  ${pass} PASS  -  ${fail} FAIL`);
  console.log("+===============================================================+");

  process.exit(fail > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
