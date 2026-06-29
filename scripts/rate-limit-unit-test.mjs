#!/usr/bin/env node
// Offline rate-limit categorization unit test — pure logic, no network.
// Mirrors categorizeRoute() from src/rate-limit.ts.
// Usage: node scripts/rate-limit-unit-test.mjs

function categorizeRoute(method, path) {
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/start$/.test(path)) return "gpu_start";
  if (method === "POST" && path === "/v1/sessions") return "session_create";
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/ws-token$/.test(path)) return "ws_token";
  if (method === "GET" && /^\/v1\/sessions\/[^/]+\/stream$/.test(path)) return "stream_connect";
  if (method === "GET" && /^\/v1\/sessions\/[^/]+\/ws$/.test(path)) return "ws_connect";
  if (method === "DELETE" && /^\/v1\/sessions\/[^/]+$/.test(path)) return "cleanup";
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/end$/.test(path)) return "cleanup";
  return "read";
}

let pass = 0;
let fail = 0;

function test(label, method, path, expected) {
  const got = categorizeRoute(method, path);
  if (got === expected) {
    console.log(`  PASS  ${label}  ->  ${got}`);
    pass++;
  } else {
    console.log(`  FAIL  ${label}  ->  got=${got}  expected=${expected}`);
    fail++;
  }
}

console.log("+===============================================================+");
console.log("|              Rate Limit Categorization Unit Test              |");
console.log("+---------------------------------------------------------------+");

// ── gpu_start ──────────────────────────────────────────────────────
test("POST /v1/sessions/:id/start", "POST", "/v1/sessions/session:abc/start", "gpu_start");
test("POST start with UUID", "POST", "/v1/sessions/session:550e8400-e29b-41d4-a716-446655440000/start", "gpu_start");

// ── session_create ────────────────────────────────────────────────
test("POST /v1/sessions", "POST", "/v1/sessions", "session_create");

// ── ws_token ───────────────────────────────────────────────────────
test("POST /v1/sessions/:id/ws-token", "POST", "/v1/sessions/session:abc/ws-token", "ws_token");

// ── stream_connect ─────────────────────────────────────────────────
test("GET /v1/sessions/:id/stream", "GET", "/v1/sessions/session:abc/stream", "stream_connect");

// ── ws_connect ──────────────────────────────────────────────────────
test("GET /v1/sessions/:id/ws", "GET", "/v1/sessions/session:abc/ws", "ws_connect");
test("GET ws with UUID", "GET", "/v1/sessions/session:550e8400-e29b-41d4-a716-446655440000/ws", "ws_connect");
test("GET /v1/sessions/:id/status is NOT ws_connect", "GET", "/v1/sessions/session:abc/status", "read");
test("POST /v1/sessions/:id/ws -> read (not ws_connect)", "POST", "/v1/sessions/session:abc/ws", "read");

// ── cleanup (DELETE) ───────────────────────────────────────────────
test("DELETE /v1/sessions/:id", "DELETE", "/v1/sessions/session:abc", "cleanup");
test("DELETE with UUID", "DELETE", "/v1/sessions/session:550e8400-e29b-41d4-a716-446655440000", "cleanup");

// ── cleanup (POST end alias) ───────────────────────────────────────
test("POST /v1/sessions/:id/end", "POST", "/v1/sessions/session:abc/end", "cleanup");

// ── read (catch-all) ───────────────────────────────────────────────
test("GET /v1/health", "GET", "/v1/health", "read");
test("GET /v1/me", "GET", "/v1/me", "read");
test("GET /v1/avatars", "GET", "/v1/avatars", "read");
test("GET /v1/backgrounds", "GET", "/v1/backgrounds", "read");
test("GET /v1/providers", "GET", "/v1/providers", "read");
test("GET /v1/sessions/:id/status", "GET", "/v1/sessions/session:abc/status", "read");

// ── edge cases ─────────────────────────────────────────────────────
test("GET unknown /v1/foo -> read", "GET", "/v1/foo", "read");
test("POST unknown sub-path -> read", "POST", "/v1/sessions/session:abc/unknown", "read");
test("PUT /v1/sessions/:id -> read", "PUT", "/v1/sessions/session:abc", "read");
test("GET /v1/sessions (no :id) -> read", "GET", "/v1/sessions", "read");

// ── wrong method does not match specific categories ────────────────
test("GET /v1/sessions/:id/start -> read (not gpu_start)", "GET", "/v1/sessions/session:abc/start", "read");
test("DELETE /v1/sessions -> read (not session_create)", "DELETE", "/v1/sessions", "read");

console.log("+---------------------------------------------------------------+");
console.log(`|  ${pass} PASS  -  ${fail} FAIL    ${pass + fail} total`);
console.log("+===============================================================+");

process.exit(fail > 0 ? 1 : 0);
