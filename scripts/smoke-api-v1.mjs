// Avatar API v1 smoke test — zero dependencies, uses built-in fetch.
//
// Usage:
//   AVTR_API_KEY=... node scripts/smoke-api-v1.mjs
//   AVTR_API_KEY=... AVTR_API_BASE=https://staging.example.com node scripts/smoke-api-v1.mjs
//
// Exit code: 0 if all non-skipped tests pass, 1 if any fail.

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;

if (!API_KEY) {
  console.error("AVTR_API_KEY is required.");
  process.exit(1);
}

const authHeaders = { Authorization: `Bearer ${API_KEY}` };

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

async function deleteJson(path, headers = {}) {
  const res = await fetch(`${BASE}${path}`, { method: "DELETE", headers });
  const text = await res.text();
  let body = null;
  try { body = text ? JSON.parse(text) : null; } catch {}
  return { status: res.status, body };
}

function assert(cond, detail) {
  if (!cond) throw new Error(detail);
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

function skip(name, reason) {
  results.push({ name, status: "SKIP", detail: reason });
}

// ── Tests ─────────────────────────────────────────────────────────────────

// 1. Missing API key → 401 missing_api_key
await test("401 missing key", async () => {
  const { status, body } = await getJson("/v1/health");
  assert(status === 401, `expected 401 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "missing_api_key", `expected missing_api_key got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// 2. Wrong API key → 403 invalid_api_key
await test("403 wrong key", async () => {
  const { status, body } = await getJson("/v1/health", { Authorization: "Bearer wrong" });
  assert(status === 403, `expected 403 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "invalid_api_key", `expected invalid_api_key got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// 3. Health endpoint
await test("200 health", async () => {
  const { status, body } = await getJson("/v1/health", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.status === "ok", `expected data.status=ok got ${body?.data?.status}`);
  assert(typeof body?.timings?.latency_ms === "number", "expected timings.latency_ms number");
  return { status: "PASS", detail: `HTTP ${status} · ${body.data.status}` };
});

// 4. Avatars catalog
await test("200 avatars", async () => {
  const { status, body } = await getJson("/v1/avatars", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(Array.isArray(body?.data?.avatars), "expected data.avatars array");
  assert(body.data.avatars.length > 0, "expected at least 1 avatar");
  assert(typeof body.data.avatars[0].avatar_id === "string", "expected avatar_id string");
  return { status: "PASS", detail: `HTTP ${status} · ${body.data.avatars.length} avatar(s)` };
});

// 4a. Avatar detail — known
await test("200 avatar detail arnold", async () => {
  const { status, body } = await getJson("/v1/avatars/arnold", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.avatar?.avatar_id === "arnold", "expected avatar_id=arnold");
  assert(typeof body?.data?.avatar?.display_name === "string", "expected display_name string");
  assert(body?.data?.avatar?.available === true, "expected available=true");
  return { status: "PASS", detail: `HTTP ${status} · ${body.data.avatar.avatar_id}` };
});

// 4b. Avatar detail — unknown
await test("404 avatar detail unknown", async () => {
  const { status, body } = await getJson("/v1/avatars/does-not-exist", authHeaders);
  assert(status === 404, `expected 404 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "avatar_not_found", `expected avatar_not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// 5. Backgrounds catalog
await test("200 backgrounds", async () => {
  const { status, body } = await getJson("/v1/backgrounds", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(Array.isArray(body?.data?.backgrounds), "expected data.backgrounds array");
  assert(body.data.backgrounds.length > 0, "expected at least 1 background");
  assert(typeof body.data.backgrounds[0].bg_id === "string", "expected bg_id string");
  return { status: "PASS", detail: `HTTP ${status} · ${body.data.backgrounds.length} bg(s)` };
});

// 5a. Background detail — known
await test("200 background detail aloe_window", async () => {
  const { status, body } = await getJson("/v1/backgrounds/aloe_window", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.background?.bg_id === "aloe_window", "expected bg_id=aloe_window");
  assert(typeof body?.data?.background?.display_name === "string", "expected display_name string");
  assert(body?.data?.background?.available === true, "expected available=true");
  return { status: "PASS", detail: `HTTP ${status} · ${body.data.background.bg_id}` };
});

// 5b. Background detail — unknown
await test("404 background detail unknown", async () => {
  const { status, body } = await getJson("/v1/backgrounds/does-not-exist", authHeaders);
  assert(status === 404, `expected 404 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "background_not_found", `expected background_not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

// 6. Providers capability + secret-leak sweep
await test("200 providers", async () => {
  const { status, body } = await getJson("/v1/providers", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");

  const d = body?.data;
  assert(typeof d?.stt?.provider === "string", "expected stt.provider string");
  assert(d?.stt?.model !== undefined, "expected stt.model present");
  assert(d?.stt?.language !== undefined, "expected stt.language present");
  assert(typeof d?.llm?.provider === "string", "expected llm.provider string");
  assert(d?.llm?.model !== undefined, "expected llm.model present");
  assert(typeof d?.tts?.provider_chain === "string", "expected tts.provider_chain string");
  assert(typeof d?.tts?.omnivoice_enabled === "boolean", "expected tts.omnivoice_enabled boolean");
  assert(typeof d?.tts?.same_gpu_parallel_enabled === "boolean", "expected tts.same_gpu_parallel_enabled boolean");
  assert(typeof d?.tts?.fallback_enabled === "boolean", "expected tts.fallback_enabled boolean");
  assert(d?.avatar?.transport === "websocket_fmp4", `expected websocket_fmp4 got ${d?.avatar?.transport}`);
  assert(typeof d?.avatar?.render_backend === "string", "expected avatar.render_backend string");
  assert(typeof d?.vad?.enabled === "boolean", "expected vad.enabled boolean");
  assert(typeof d?.smart_turn?.enabled === "boolean", "expected smart_turn.enabled boolean");

  // Provider-neutral render backend — must not expose "modal"
  assert(d.avatar.render_backend !== "modal", "render_backend must not leak 'modal'");

  // Secret-leak sweep: scan keys and string values for dangerous patterns
  const dangerousPatterns = [
    /api[_-]?key/i,
    /secret/i,
    /token/i,
    /modal[._-]?(run|session|endpoint)/i,
    /vallangirakesh--/,
    /https?:\/\/[^/]*(workers\.dev|modal\.run)/i,
    /MODAL_/i,
  ];

  function deepScan(obj, path) {
    for (const [key, val] of Object.entries(obj ?? {})) {
      const keyStr = `${path}.${key}`;
      for (const re of dangerousPatterns) {
        if (re.test(key)) {
          throw new Error(`unsafe key: ${keyStr}`);
        }
      }
      if (typeof val === "string") {
        for (const re of dangerousPatterns) {
          if (re.test(val)) {
            throw new Error(`unsafe value at ${keyStr}: ${val.slice(0, 60)}`);
          }
        }
      } else if (typeof val === "object" && val !== null) {
        deepScan(val, keyStr);
      }
    }
  }
  deepScan(d, "data");

  return {
    status: "PASS",
    detail: `HTTP ${status} · ${d.tts.provider_chain.slice(0, 24)}`,
  };
});

// ── Session A: create / status / delete ───────────────────────────────────

let sessionA = null;

// 7. /v1/me — identity + leak sweep
await test("200 me", async () => {
  const { status, body } = await getJson("/v1/me", authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");

  const d = body?.data;
  assert(typeof d?.tenant_id === "string", "expected tenant_id string");
  assert(d?.auth_mode === "metadata" || d?.auth_mode === "legacy", `invalid auth_mode ${d?.auth_mode}`);

  if (d.auth_mode === "legacy") {
    assert(d.tenant_id === "legacy", `legacy mode expects tenant_id=legacy got ${d.tenant_id}`);
    assert(d.key_name === null, `legacy mode expects key_name=null got ${d.key_name}`);
    assert(d.plan === null, `legacy mode expects plan=null got ${d.plan}`);
  } else {
    assert(d.tenant_id !== "legacy", "metadata mode must not return legacy tenant_id");
    assert(typeof d.tenant_id === "string" && d.tenant_id.length > 0, "metadata mode requires non-empty tenant_id");
  }

  // Secret-leak sweep
  const dangerousPatterns = [
    /api[_-]?key/i,
    /secret/i,
    /token/i,
    /https?:\/\/[^/]*(workers\.dev|modal\.run)/i,
  ];
  function deepScan(obj, path) {
    for (const [key, val] of Object.entries(obj ?? {})) {
      for (const re of dangerousPatterns) {
        if (re.test(key)) throw new Error(`unsafe key: ${path}.${key}`);
      }
      if (typeof val === "string") {
        for (const re of dangerousPatterns) {
          if (re.test(val)) throw new Error(`unsafe value at ${path}.${key}: ${val.slice(0, 60)}`);
        }
      } else if (typeof val === "object" && val !== null) {
        deepScan(val, `${path}.${key}`);
      }
    }
  }
  deepScan(d, "data");

  return { status: "PASS", detail: `HTTP ${status} · auth_mode=${d.auth_mode} · tenant=${d.tenant_id}` };
});

await test("200 create session A", async () => {
  const { status, body } = await postJson("/v1/sessions", null, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.session_id?.startsWith("session:"), `expected session: prefix got ${body?.data?.session_id}`);
  assert(typeof body?.data?.stream?.ws_url === "string", "expected stream.ws_url string");
  assert(body?.data?.stream?.ws_url.includes("/v1/sessions/"), "expected /v1/sessions/ in ws_url");
  assert(body?.data?.stream?.ws_url.includes("token="), "expected token= in ws_url");
  assert(typeof body?.data?.stream?.token_expires_at === "number", "expected token_expires_at number");
  assert(body?.data?.status === "created", `expected status=created got ${body?.data?.status}`);
  sessionA = body.data.session_id;
  return { status: "PASS", detail: `HTTP ${status} · ${sessionA.slice(0, 20)}...` };
});

await test("200 session A status", async () => {
  const { status, body } = await getJson(`/v1/sessions/${sessionA}/status`, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.session_id === sessionA, "session_id mismatch");
  assert(["created", "ready", "active", "ended"].includes(body?.data?.status), `invalid status ${body?.data?.status}`);
  return { status: "PASS", detail: `HTTP ${status} · status=${body.data.status}` };
});

await test("200 session A delete", async () => {
  const { status, body } = await deleteJson(`/v1/sessions/${sessionA}`, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.success === true, "expected success=true");
  assert(body?.data?.status === "ended", `expected status=ended got ${body?.data?.status}`);
  assert(body?.data?.end_ok !== undefined, "expected end_ok present");
  return { status: "PASS", detail: `HTTP ${status} · status=${body.data.status}` };
});

// ── Session B: create / start / cleanup ───────────────────────────────────

let sessionB = null;

await test("200 create session B", async () => {
  const { status, body } = await postJson("/v1/sessions", null, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.data?.session_id?.startsWith("session:"), `expected session: prefix`);
  assert(body?.data?.stream?.ws_url.includes("/v1/sessions/"), "expected /v1/sessions/ in ws_url");
  assert(body?.data?.stream?.ws_url.includes("token="), "expected token= in ws_url");
  assert(typeof body?.data?.stream?.token_expires_at === "number", "expected token_expires_at number");
  sessionB = body.data.session_id;
  return { status: "PASS", detail: `HTTP ${status} · ${sessionB.slice(0, 20)}...` };
});

await test("session B /start unknown avatar → 400", async () => {
  const { status, body } = await postJson(
    `/v1/sessions/${sessionB}/start`,
    { avatar_id: "does-not-exist", bg_id: "aloe_window" },
    authHeaders,
  );

  assert(status === 400, `expected 400 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "avatar_not_found", `expected avatar_not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

await test("session B /start unknown background → 400", async () => {
  const { status, body } = await postJson(
    `/v1/sessions/${sessionB}/start`,
    { avatar_id: "arnold", bg_id: "does-not-exist" },
    authHeaders,
  );

  assert(status === 400, `expected 400 got ${status}`);
  assert(body?.success === false, "expected success=false");
  assert(body?.error?.code === "background_not_found", `expected background_not_found got ${body?.error?.code}`);
  return { status: "PASS", detail: `HTTP ${status} · ${body.error.code}` };
});

await test("session B /start valid", async () => {
  const { status, body } = await postJson(
    `/v1/sessions/${sessionB}/start`,
    { avatar_id: "arnold", bg_id: "aloe_window" },
    authHeaders,
  );

  if (status === 200) {
    assert(body?.success === true, "expected success=true");
    assert(body?.data?.status === "ready", `expected status=ready got ${body?.data?.status}`);
    assert(body?.data?.modal_session_id === undefined, "modal_session_id must not leak");
    return { status: "PASS", detail: `HTTP ${status} · status=${body.data.status}` };
  }

  // Modal outage — known, not an API shell failure
  if (status >= 500 && body?.error?.code === "modal_unreachable") {
    return { status: "SKIP", detail: `HTTP ${status} · modal_unreachable (known backend outage)` };
  }
  if (status >= 500 && body?.error?.code === "modal_start_failed") {
    return { status: "SKIP", detail: `HTTP ${status} · modal_start_failed (known backend outage)` };
  }

  throw new Error(`unexpected: HTTP ${status} · ${body?.error?.code ?? JSON.stringify(body).slice(0, 100)}`);
});

// ── Session C: REST /start with default avatar_id ─────────────────────────

let sessionC = null;

await test("200 create session C", async () => {
  const { status, body } = await postJson("/v1/sessions", null, authHeaders);
  assert(status === 200, `expected 200 got ${status}`);
  assert(body?.data?.session_id?.startsWith("session:"), `expected session: prefix`);
  assert(body?.data?.stream?.ws_url.includes("/v1/sessions/"), "expected /v1/sessions/ in ws_url");
  assert(body?.data?.stream?.ws_url.includes("token="), "expected token= in ws_url");
  assert(typeof body?.data?.stream?.token_expires_at === "number", "expected token_expires_at number");
  sessionC = body.data.session_id;
  return { status: "PASS", detail: `HTTP ${status} · ${sessionC.slice(0, 20)}...` };
});

await test("session C /start with avatar_id default", async () => {
  const { status, body } = await postJson(
    `/v1/sessions/${sessionC}/start`,
    { avatar_id: "default" },
    authHeaders,
  );

  if (status === 200) {
    assert(body?.success === true, "expected success=true");
    assert(body?.data?.status === "ready", `expected status=ready got ${body?.data?.status}`);
    assert(body?.data?.avatar_id === "arnold", `expected avatar_id=arnold got ${body?.data?.avatar_id}`);
    assert(body?.data?.modal_session_id === undefined, "modal_session_id must not leak");
    return { status: "PASS", detail: `HTTP ${status} · default resolved to ${body.data.avatar_id}` };
  }

  // Modal outage — known, not an API shell failure
  if (status >= 500 && body?.error?.code === "modal_unreachable") {
    return { status: "SKIP", detail: `HTTP ${status} · modal_unreachable (known backend outage)` };
  }
  if (status >= 500 && body?.error?.code === "modal_start_failed") {
    return { status: "SKIP", detail: `HTTP ${status} · modal_start_failed (known backend outage)` };
  }

  throw new Error(`unexpected: HTTP ${status} · ${body?.error?.code ?? JSON.stringify(body).slice(0, 100)}`);
});

// Cleanup sessions B and C regardless of /start outcome
if (sessionB) {
  try {
    await deleteJson(`/v1/sessions/${sessionB}`, authHeaders);
  } catch {}
}
if (sessionC) {
  try {
    await deleteJson(`/v1/sessions/${sessionC}`, authHeaders);
  } catch {}
}

// ── Report ────────────────────────────────────────────────────────────────

const passes = results.filter((r) => r.status === "PASS").length;
const fails = results.filter((r) => r.status === "FAIL").length;
const skips = results.filter((r) => r.status === "SKIP").length;
const total = results.length;

// Dynamic width: base columns + longest row content
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
lines.push("|" + center("Avatar API v1 Smoke Test", W) + "|");
lines.push("|" + center(BASE, W) + "|");
lines.push("+" + "-".repeat(W) + "+");

for (const r of results) {
  const icon = r.status === "PASS" ? "PASS" : r.status === "SKIP" ? "SKIP" : "FAIL";
  const row = `  ${icon}  ${r.name}`;
  const det = r.detail ? `  (${r.detail})` : "";
  lines.push("|" + pad(row + det, W) + "|");
}

lines.push("+" + "-".repeat(W) + "+");
const summary = `  ${passes} PASS  ·  ${fails} FAIL  ·  ${skips} SKIP    ${total} total`;
lines.push("|" + pad(summary, W) + "|");
lines.push("+" + "=".repeat(W) + "+");

console.log(lines.join("\n"));

process.exit(fails > 0 ? 1 : 0);
