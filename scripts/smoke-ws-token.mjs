// WebSocket session token smoke test — zero dependencies, uses built-in fetch.
//
// Usage:
//   AVTR_API_KEY=... node scripts/smoke-ws-token.mjs
//   AVTR_API_KEY=... AVTR_API_BASE=https://staging.example.com node scripts/smoke-ws-token.mjs
//
// Exit code: 0 if all tests pass, 1 if any fail.

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;
const API_KEY_B = process.env.AVTR_API_KEY_B || null;

if (!API_KEY) {
  console.error("AVTR_API_KEY is required.");
  process.exit(1);
}

const authHeaders = { Authorization: `Bearer ${API_KEY}` };
const authHeadersB = API_KEY_B ? { Authorization: `Bearer ${API_KEY_B}` } : null;

// ── Helpers ───────────────────────────────────────────────────────────────

async function getJson(path, headers = {}) {
  const res = await fetch(`${BASE}${path}`, { headers });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body };
}

async function postJson(path, payload = null, headers = {}) {
  const opts = { method: "POST", headers: { ...headers } };
  if (payload) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(payload);
  }
  const res = await fetch(`${BASE}${path}`, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body };
}

function assert(cond, detail) {
  if (!cond) throw new Error(detail);
}

// Extract token from ws_url query string
function extractToken(wsUrl) {
  const url = new URL(wsUrl);
  return url.searchParams.get("token");
}

// ── Test runner ───────────────────────────────────────────────────────────

const results = [];

async function test(name, fn) {
  try {
    const { status: verb, detail } = await fn();
    results.push({ name, status: verb, detail });
  } catch (err) {
    results.push({ name, status: "FAIL", detail: err.message });
  }
}

// ── Tests ─────────────────────────────────────────────────────────────────

// Create two sessions for cross-session test
let sessionA = null;
let tokenA = null;
let sessionB = null;
let tokenB = null;

// a. POST /v1/sessions returns /v1/ ws_url with token and token_expires_at
await test("session create returns v1 ws_url with token", async () => {
  const { status, body } = await postJson("/v1/sessions", null, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");

  const wsUrl = body?.data?.stream?.ws_url;
  assert(typeof wsUrl === "string", "expected ws_url string");
  assert(wsUrl.includes("/v1/sessions/"), `expected /v1/sessions/ in ws_url got ${wsUrl}`);
  assert(wsUrl.includes("token="), "expected token= in ws_url");

  const token = extractToken(wsUrl);
  assert(token && token.length > 20, `expected non-trivial token length, got ${token?.length}`);

  assert(typeof body?.data?.stream?.token_expires_at === "number", "expected token_expires_at number");
  assert(body.data.stream.token_expires_at > Date.now(), "token_expires_at should be in the future");

  sessionA = body.data.session_id;
  tokenA = token;
  return { status: "PASS", detail: `session=${sessionA.slice(0, 18)}... token_len=${token.length}` };
});

// b. Valid token passes Worker gate (no Upgrade header → DO returns 400)
await test("valid token passes Worker gate", async () => {
  assert(sessionA && tokenA, "session A not created");
  const { status, body } = await getJson(`/v1/sessions/${sessionA}/ws?token=${tokenA}`);
  // Not 401 means the Worker accepted the token.
  // Without Upgrade header, DO returns 400 "expected_websocket".
  assert(status !== 401, `expected non-401 (token should pass) got ${status}`);
  return { status: "PASS", detail: `HTTP ${status} (not 401 → token accepted)` };
});

// c. Missing token rejected
await test("missing token rejected", async () => {
  assert(sessionA, "session A not created");
  const { status, body } = await getJson(`/v1/sessions/${sessionA}/ws`);
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "ws_token_missing", `expected ws_token_missing got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// d. Garbage token rejected
await test("garbage token rejected", async () => {
  assert(sessionA, "session A not created");
  const { status, body } = await getJson(`/v1/sessions/${sessionA}/ws?token=garbage`);
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "ws_token_invalid", `expected ws_token_invalid got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// e. Tampered signature rejected (flip last char of token)
await test("tampered token rejected", async () => {
  assert(sessionA && tokenA, "session A not created");
  const tampered = tokenA.slice(0, -1) + (tokenA.slice(-1) === "A" ? "B" : "A");
  const { status, body } = await getJson(`/v1/sessions/${sessionA}/ws?token=${tampered}`);
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.error?.code === "ws_token_invalid", `expected ws_token_invalid got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// f. Cross-session token rejected (create session B, use A's token on B)
await test("cross-session token rejected", async () => {
  assert(sessionA && tokenA, "session A not created");

  // Create session B
  const { status: createStatus, body: createBody } = await postJson("/v1/sessions", null, authHeaders);
  assert(createStatus === 200, `expected 200 creating session B got ${createStatus}`);
  sessionB = createBody.data.session_id;
  tokenB = extractToken(createBody.data.stream.ws_url);

  // Use A's token on B's WS path
  const { status, body } = await getJson(`/v1/sessions/${sessionB}/ws?token=${tokenA}`);
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.error?.code === "ws_token_invalid", `expected ws_token_invalid got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · token A rejected on session B` };
});

// g. Token refresh returns fresh valid token
await test("token refresh returns fresh token", async () => {
  assert(sessionA, "session A not created");
  const { status, body } = await postJson(`/v1/sessions/${sessionA}/ws-token`, null, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");

  const wsUrl = body?.data?.ws_url;
  assert(typeof wsUrl === "string", "expected ws_url string");
  assert(wsUrl.includes("/v1/sessions/"), "expected /v1/sessions/ in ws_url");
  assert(wsUrl.includes("token="), "expected token= in ws_url");

  const freshToken = extractToken(wsUrl);
  assert(freshToken && freshToken !== tokenA, "expected different token from refresh");
  assert(typeof body?.data?.token_expires_at === "number", "expected token_expires_at number");
  assert(body.data.token_expires_at > Date.now(), "token_expires_at should be in the future");

  // Verify the fresh token actually works
  const { status: wsStatus } = await getJson(`/v1/sessions/${sessionA}/ws?token=${freshToken}`);
  assert(wsStatus !== 401, `expected fresh token to pass (non-401) got ${wsStatus}`);

  return { status: "PASS", detail: `HTTP ${status} · fresh token validated` };
});

// h. Demo WS path still tokenless (no /v1/ prefix)
await test("demo WS path unchanged (no token required)", async () => {
  // Create a demo session via POST /sessions (no auth)
  const demoRes = await fetch(`${BASE}/sessions`, { method: "POST" });
  const demoBody = await demoRes.json();
  assert(demoRes.status === 200, `expected 200 creating demo session got ${demoRes.status}`);
  assert(demoBody.wsUrl, "expected wsUrl in demo response");
  assert(!demoBody.wsUrl.includes("/v1/"), "demo wsUrl should NOT have /v1/ prefix");
  assert(!demoBody.wsUrl.includes("token="), "demo wsUrl should NOT have token");

  // Demo WS path should not return 401 (tokenless, passes through)
  const demoSessionId = demoBody.sessionId;
  const { status } = await getJson(`/sessions/${demoSessionId}/ws`);
  assert(status !== 401, `expected non-401 on demo WS path got ${status}`);

  return { status: "PASS", detail: `demo wsUrl=${demoBody.wsUrl.slice(0, 50)}...` };
});

// i. Token refresh requires API key auth
await test("token refresh requires auth", async () => {
  assert(sessionA, "session A not created");
  const { status, body } = await postJson(`/v1/sessions/${sessionA}/ws-token`, null, {});
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.error?.code === "missing_api_key", `expected missing_api_key got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// j. WS session.start with unknown avatar returns avatar_not_found before Modal
await test("WS session.start unknown avatar", async () => {
  assert(sessionA && tokenA, "session A not created");

  const wsUrl = `${BASE.replace(/^http/, "ws")}/v1/sessions/${sessionA}/ws?token=${tokenA}`;
  const ws = new WebSocket(wsUrl);

  const openPromise = new Promise((resolve, reject) => {
    ws.onopen = resolve;
    ws.onerror = (e) => reject(new Error(`WebSocket error: ${e.message || "unknown"}`));
  });
  const messagePromise = new Promise((resolve) => {
    ws.onmessage = (e) => resolve(e.data);
  });
  const closePromise = new Promise((resolve) => {
    ws.onclose = resolve;
  });

  await openPromise;
  ws.send(JSON.stringify({ type: "session.start", avatarId: "does-not-exist" }));

  const data = await messagePromise;
  const msg = JSON.parse(data);

  ws.close();
  await closePromise;

  assert(msg.type === "error", `expected error message, got ${msg.type}`);
  assert(msg.code === "avatar_not_found", `expected avatar_not_found got ${msg.code}`);
  return { status: "PASS", detail: `WS error · ${msg.code}` };
});

// ── Ownership tests ─────────────────────────────────────────────────────────

let ownSession = null;

// j. Owner can get status (ownership check passes)
await test("owner can get session status", async () => {
  const { status, body } = await postJson("/v1/sessions", null, authHeaders);
  assert(status === 200, `expected 200 creating session got ${status}`);
  ownSession = body.data.session_id;

  const { status: st, body: stBody } = await getJson(`/v1/sessions/${ownSession}/status`, authHeaders);
  assert(st === 200, `expected 200 on status got ${st}`);
  assert(stBody?.success === true, "expected success=true on status");
  return { status: "PASS", detail: `status=${stBody?.data?.status}` };
});

// k. Demo-created session is inaccessible via v1 API (no ownership record)
await test("demo session inaccessible via v1 API (404)", async () => {
  // Create a demo session via POST /sessions (no auth, no ownership)
  const demoRes = await fetch(`${BASE}/sessions`, { method: "POST" });
  const demoBody = await demoRes.json();
  assert(demoRes.status === 200, `expected 200 creating demo session got ${demoRes.status}`);
  const demoSid = demoBody.sessionId;

  // Try to access via v1 API → should get 404 (no ownership record)
  const { status, body } = await getJson(`/v1/sessions/${demoSid}/status`, authHeaders);
  assert(status === 404, `expected 404 for demo session via v1 API got ${status}`);
  assert(body?.error?.code === "not_found", `expected not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `demo session ${demoSid.slice(0, 18)}... → 404` };
});

// l. Random session ID is rejected (no ownership record)
await test("nonexistent session returns 404", async () => {
  const fakeSid = "session:00000000-0000-0000-0000-000000000000";
  const { status, body } = await getJson(`/v1/sessions/${fakeSid}/status`, authHeaders);
  assert(status === 404, `expected 404 got ${status}`);
  assert(body?.error?.code === "not_found", `expected not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `fake session → 404 not_found` };
});

// m. Cross-tenant denial (only if AVTR_API_KEY_B is configured)
if (authHeadersB) {
  let tenantBSession = null;

  await test("cross-tenant: B cannot access A's session status", async () => {
    assert(ownSession, "own session not created");
    const { status, body } = await getJson(`/v1/sessions/${ownSession}/status`, authHeadersB);
    assert(status === 404, `expected 404 for cross-tenant access got ${status}`);
    assert(body?.error?.code === "not_found", `expected not_found got ${body?.error?.code}`);
    return { status: "PASS", detail: `tenant B → A's session → 404` };
  });

  await test("cross-tenant: B cannot refresh A's ws-token", async () => {
    assert(ownSession, "own session not created");
    const { status, body } = await postJson(`/v1/sessions/${ownSession}/ws-token`, null, authHeadersB);
    assert(status === 404, `expected 404 for cross-tenant token refresh got ${status}`);
    assert(body?.error?.code === "not_found", `expected not_found got ${body?.error?.code}`);
    return { status: "PASS", detail: `tenant B → A's ws-token → 404` };
  });

  await test("cross-tenant: B cannot delete A's session", async () => {
    assert(ownSession, "own session not created");
    const { status, body } = await fetch(`${BASE}/v1/sessions/${ownSession}`, {
      method: "DELETE",
      headers: authHeadersB,
    }).then(async (r) => ({ status: r.status, body: await r.json().catch(() => null) }));
    assert(status === 404, `expected 404 for cross-tenant delete got ${status}`);
    assert(body?.error?.code === "not_found", `expected not_found got ${body?.error?.code}`);
    return { status: "PASS", detail: `tenant B → A's session DELETE → 404` };
  });

  await test("cross-tenant: B can create and access own session", async () => {
    const { status: cs, body: cb } = await postJson("/v1/sessions", null, authHeadersB);
    assert(cs === 200, `expected 200 creating B session got ${cs}`);
    tenantBSession = cb.data.session_id;

    const { status: ss } = await getJson(`/v1/sessions/${tenantBSession}/status`, authHeadersB);
    assert(ss === 200, `expected 200 on B's own status got ${ss}`);

    // Cleanup
    try { await fetch(`${BASE}/v1/sessions/${tenantBSession}`, { method: "DELETE", headers: authHeadersB }); } catch {}

    return { status: "PASS", detail: `tenant B creates + accesses own session` };
  });

  // Also verify A cannot access B's session
  await test("cross-tenant: A cannot access B's session", async () => {
    const { status: cs, body: cb } = await postJson("/v1/sessions", null, authHeadersB);
    assert(cs === 200, `expected 200 creating B session got ${cs}`);
    const bSid = cb.data.session_id;

    const { status, body } = await getJson(`/v1/sessions/${bSid}/status`, authHeaders);
    assert(status === 404, `expected 404 for A accessing B's session got ${status}`);

    // Cleanup
    try { await fetch(`${BASE}/v1/sessions/${bSid}`, { method: "DELETE", headers: authHeadersB }); } catch {}

    return { status: "PASS", detail: `tenant A → B's session → 404` };
  });
} else {
  // SKIP cross-tenant tests
  results.push({
    name: "cross-tenant tests (requires AVTR_API_KEY_B)",
    status: "SKIP",
    detail: "Set AVTR_API_KEY_B to run cross-tenant ownership denial tests",
  });
}

// ── Cleanup ───────────────────────────────────────────────────────────────

for (const sid of [sessionA, sessionB, ownSession]) {
  if (sid) {
    try { await fetch(`${BASE}/v1/sessions/${sid}`, { method: "DELETE", headers: authHeaders }); } catch {}
  }
}

// ── Report ────────────────────────────────────────────────────────────────

const passes = results.filter((r) => r.status === "PASS").length;
const fails = results.filter((r) => r.status === "FAIL").length;
const total = results.length;

const minW = 60;
const contentRows = results.map((r) => {
  const det = r.detail ? `  (${r.detail})` : "";
  return `  ${r.status}  ${r.name}${det}`;
});
const W = Math.max(minW, ...contentRows.map((r) => r.length));

const pad = (s, w = W) => s + " ".repeat(Math.max(0, w - s.length));
const center = (s, w = W) => {
  const left = Math.floor((w - s.length) / 2);
  const right = w - s.length - left;
  return " ".repeat(Math.max(0, left)) + s + " ".repeat(Math.max(0, right));
};

const lines = [];
lines.push("+" + "=".repeat(W) + "+");
lines.push("|" + center("WS Token Smoke Test", W) + "|");
lines.push("|" + center(BASE, W) + "|");
lines.push("+" + "-".repeat(W) + "+");

for (const r of results) {
  const row = `  ${r.status}  ${r.name}`;
  const det = r.detail ? `  (${r.detail})` : "";
  lines.push("|" + pad(row + det, W) + "|");
}

lines.push("+" + "-".repeat(W) + "+");
const summary = `  ${passes} PASS  ·  ${fails} FAIL    ${total} total`;
lines.push("|" + pad(summary, W) + "|");
lines.push("+" + "=".repeat(W) + "+");

console.log(lines.join("\n"));

process.exit(fails > 0 ? 1 : 0);
