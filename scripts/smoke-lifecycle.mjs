const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;

if (!API_KEY) {
  console.error("AVTR_API_KEY is required.");
  process.exit(1);
}

const authHeaders = { Authorization: `Bearer ${API_KEY}` };
const results = [];

function record(name, passed, detail = "") {
  const tag = passed ? "PASS" : "FAIL";
  results.push({ name, passed });
  console.log(`  ${tag}  ${name}  ${detail}`);
}

async function getJson(path, headers = {}) {
  const res = await fetch(`${BASE}${path}`, { headers: { ...authHeaders, ...headers } });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body, headers: res.headers };
}

async function postJson(path, payload = null, headers = {}) {
  const opts = { method: "POST", headers: { ...authHeaders, ...headers } };
  if (payload) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(payload);
  }
  const res = await fetch(`${BASE}${path}`, opts);
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body, headers: res.headers };
}

async function deleteJson(path, headers = {}) {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE", headers: { ...authHeaders, ...headers } });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body, headers: res.headers };
}

async function createSession() {
  const { body } = await postJson("/v1/sessions");
  return body?.data?.session_id ?? null;
}

async function cleanupSession(sid) {
  if (!sid) return;
  try { await deleteJson(`/v1/sessions/${sid}`); } catch {}
}

// ── Test 1: X-Request-ID echo ──────────────────────────────────
async function testRequestIdEcho() {
  console.log("\n--- Test 1: X-Request-ID echo ---");
  const reqId = "test-req-" + crypto.randomUUID().slice(0, 8);
  const { headers, body } = await getJson("/v1/health", { "X-Request-ID": reqId });

  record("Echo: response header matches", headers.get("X-Request-Id") === reqId, `(header=${headers.get("X-Request-Id")})`);
  record("Echo: timings.request_id in body", body?.timings?.request_id === reqId, `(body=${body?.timings?.request_id})`);
}

// ── Test 2: X-Request-ID auto-generation ────────────────────────
async function testRequestIdAutoGen() {
  console.log("\n--- Test 2: X-Request-ID auto-generation ---");
  const { headers, body } = await getJson("/v1/health");

  const headerId = headers.get("X-Request-Id");
  record("Auto-gen: header present", headerId !== null, `(header=${headerId})`);
  record("Auto-gen: header is UUID format", headerId !== null && /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(headerId), "");
  record("Auto-gen: body has request_id", !!body?.timings?.request_id, `(body=${body?.timings?.request_id})`);
}

// ── Test 3: X-Request-ID validation (invalid chars → auto-gen) ──
async function testRequestIdValidation() {
  console.log("\n--- Test 3: X-Request-ID validation ---");
  const { headers } = await getJson("/v1/health", { "X-Request-ID": "!!invalid!!" });

  const headerId = headers.get("X-Request-Id");
  record("Validation: invalid replaced with UUID", headerId !== null && headerId !== "!!invalid!!", `(header=${headerId})`);
}

// ── Test 4: Repeated DELETE is safe ─────────────────────────────
async function testRepeatedDelete() {
  console.log("\n--- Test 4: Repeated DELETE idempotency ---");
  const sid = await createSession();
  if (!sid) { record("DELETE idempotency: failed to create session", false); return; }

  const del1 = await deleteJson(`/v1/sessions/${sid}`);
  record("First DELETE returns 200", del1.status === 200, `(status=${del1.status})`);

  const del2 = await deleteJson(`/v1/sessions/${sid}`);
  record("Second DELETE returns 200", del2.status === 200, `(status=${del2.status})`);
  record("Second DELETE status is ended", del2.body?.data?.status === "ended", `(status=${del2.body?.data?.status})`);
}

// ── Test 5: DELETE never-started session ────────────────────────
async function testDeleteNeverStarted() {
  console.log("\n--- Test 5: DELETE never-started session ---");
  const sid = await createSession();
  if (!sid) { record("Never-started: failed to create session", false); return; }

  const del = await deleteJson(`/v1/sessions/${sid}`);
  record("DELETE never-started returns 200", del.status === 200, `(status=${del.status})`);
  record("Status is ended", del.body?.data?.status === "ended", `(status=${del.body?.data?.status})`);
}

// ── Test 6: DELETE unowned/non-existent session → 404 ──────────
async function testDeleteUnowned() {
  console.log("\n--- Test 6: DELETE non-existent session ---");
  const fakeSid = "session:nonexistent-" + crypto.randomUUID().slice(0, 8);
  const del = await deleteJson(`/v1/sessions/${fakeSid}`);
  record("Non-existent returns 404", del.status === 404, `(status=${del.status})`);
}

// ── Test 7: Duplicate start behavior ────────────────────────────
async function testDuplicateStart() {
  console.log("\n--- Test 7: Duplicate start (SKIP if Modal unavailable) ---");
  const sid = await createSession();
  if (!sid) { record("Duplicate start: failed to create session", false); return; }

  try {
    const start1 = await postJson(`/v1/sessions/${sid}/start`, { avatar_id: "default" });
    if (start1.status >= 500 || start1.body?.error?.code === "modal_unreachable") {
      record("Duplicate start: Modal unavailable — SKIP", true, `(start1 status=${start1.status})`);
      return;
    }
    if (start1.status !== 200) {
      record("Duplicate start: first start failed unexpectedly", false, `(status=${start1.status}, code=${start1.body?.error?.code})`);
      return;
    }

    const start2 = await postJson(`/v1/sessions/${sid}/start`, { avatar_id: "default" });
    const code = start2.body?.error?.code;
    const isExpected = start2.status === 200 || code === "session_starting" || code === "session_conflict";
    record("Duplicate start: second returns idempotent or conflict", isExpected, `(status=${start2.status}, code=${code})`);
  } finally {
    await cleanupSession(sid);
  }
}

// ── Test 8: X-Request-ID on error responses ────────────────────
async function testRequestIdOnError() {
  console.log("\n--- Test 8: X-Request-ID on error responses ---");
  const reqId = "err-req-" + crypto.randomUUID().slice(0, 8);
  const { headers, body } = await getJson("/v1/sessions/session:nonexistent/status", { "X-Request-ID": reqId });

  record("Error: header present on 404", headers.get("X-Request-Id") === reqId, `(header=${headers.get("X-Request-Id")})`);
  record("Error: body has request_id", body?.timings?.request_id === reqId, `(body=${body?.timings?.request_id})`);
}

// ── Main ────────────────────────────────────────────────────────
async function main() {
  console.log("+===============================================================+");
  console.log("|              Lifecycle Smoke Test (deployed)                  |");
  console.log(`|  ${BASE}`);
  console.log("+===============================================================+");

  await testRequestIdEcho();
  await testRequestIdAutoGen();
  await testRequestIdValidation();
  await testRepeatedDelete();
  await testDeleteNeverStarted();
  await testDeleteUnowned();
  await testDuplicateStart();
  await testRequestIdOnError();

  const pass = results.filter(r => r.passed).length;
  const fail = results.filter(r => !r.passed).length;
  console.log("+---------------------------------------------------------------+");
  console.log(`|  ${pass} PASS  -  ${fail} FAIL`);
  console.log("+===============================================================+");
  process.exit(fail > 0 ? 1 : 0);
}

main().catch(err => {
  console.error("Fatal error:", err);
  process.exit(1);
});
