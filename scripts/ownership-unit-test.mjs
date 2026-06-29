// Offline ownership logic unit test — zero dependencies, pure logic.
//
// Tests the pure comparison function extracted from checkSessionOwnership.
// No network, no DO, no Worker runtime needed.
//
// Usage:  npx tsx scripts/ownership-unit-test.mjs
// (also runs with: node scripts/ownership-unit-test.mjs)

// ── Pure ownership logic (mirrors checkSessionOwnership decision tree) ──────

function evaluateOwnership(callerTenantId, storedOwner) {
  if (!storedOwner) {
    return { allowed: false, reason: "owner_not_found" };
  }
  if (storedOwner.tenant_id !== callerTenantId) {
    return { allowed: false, reason: "tenant_mismatch" };
  }
  return { allowed: true };
}

// ── Test runner ────────────────────────────────────────────────────────────

const results = [];

function test(name, fn) {
  try {
    const { status, detail } = fn();
    results.push({ name, status, detail });
  } catch (err) {
    results.push({ name, status: "FAIL", detail: err.message });
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

// ── Tests ──────────────────────────────────────────────────────────────────

test("matching tenant is allowed", () => {
  const result = evaluateOwnership("tenant_a", { tenant_id: "tenant_a" });
  assert(result.allowed === true, `expected allowed=true got ${result.allowed}`);
  assert(!result.reason, `expected no reason got ${result.reason}`);
  return { status: "PASS", detail: "tenant_a === tenant_a" };
});

test("different tenant is denied with tenant_mismatch", () => {
  const result = evaluateOwnership("tenant_b", { tenant_id: "tenant_a" });
  assert(result.allowed === false, `expected allowed=false got ${result.allowed}`);
  assert(result.reason === "tenant_mismatch", `expected tenant_mismatch got ${result.reason}`);
  return { status: "PASS", detail: "tenant_b != tenant_a" };
});

test("missing owner is denied with owner_not_found", () => {
  const result = evaluateOwnership("tenant_a", null);
  assert(result.allowed === false, `expected allowed=false got ${result.allowed}`);
  assert(result.reason === "owner_not_found", `expected owner_not_found got ${result.reason}`);
  return { status: "PASS", detail: "null owner -> owner_not_found" };
});

test("undefined owner is denied with owner_not_found", () => {
  const result = evaluateOwnership("tenant_a", undefined);
  assert(result.allowed === false, `expected allowed=false got ${result.allowed}`);
  assert(result.reason === "owner_not_found", `expected owner_not_found got ${result.reason}`);
  return { status: "PASS", detail: "undefined owner -> owner_not_found" };
});

test("legacy tenant matches legacy owner", () => {
  const result = evaluateOwnership("legacy", { tenant_id: "legacy" });
  assert(result.allowed === true, `expected allowed=true got ${result.allowed}`);
  return { status: "PASS", detail: "legacy === legacy" };
});

test("metadata tenant does not match legacy", () => {
  const result = evaluateOwnership("tenant_prod", { tenant_id: "legacy" });
  assert(result.allowed === false, `expected allowed=false got ${result.allowed}`);
  assert(result.reason === "tenant_mismatch", `expected tenant_mismatch got ${result.reason}`);
  return { status: "PASS", detail: "tenant_prod != legacy" };
});

test("legacy does not match metadata tenant", () => {
  const result = evaluateOwnership("legacy", { tenant_id: "tenant_prod" });
  assert(result.allowed === false, `expected allowed=false got ${result.allowed}`);
  assert(result.reason === "tenant_mismatch", `expected tenant_mismatch got ${result.reason}`);
  return { status: "PASS", detail: "legacy != tenant_prod" };
});

test("empty string tenant_id is handled (edge case)", () => {
  const result = evaluateOwnership("", { tenant_id: "" });
  assert(result.allowed === true, `expected allowed=true for matching empty strings`);
  return { status: "PASS", detail: "empty === empty (edge)" };
});

test("owner with extra fields still works", () => {
  const result = evaluateOwnership("tenant_a", {
    tenant_id: "tenant_a",
    auth_mode: "metadata",
    key_name: "prod-key",
    plan: "enterprise",
    created_at: 1700000000000,
  });
  assert(result.allowed === true, `expected allowed=true`);
  return { status: "PASS", detail: "full SessionOwner object" };
});

// ── init_owner validation logic (mirrors DO #handleRestInitOwner) ──────────

function validateInitOwnerBody(body) {
  if (typeof body.tenant_id !== "string" || body.tenant_id.trim() === "") {
    return { valid: false, error: "tenant_id must be a non-empty string" };
  }
  if (typeof body.auth_mode !== "string" || body.auth_mode.trim() === "") {
    return { valid: false, error: "auth_mode must be a non-empty string" };
  }
  return { valid: true };
}

test("init_owner rejects missing tenant_id", () => {
  const result = validateInitOwnerBody({ auth_mode: "metadata" });
  assert(result.valid === false, `expected invalid got ${result.valid}`);
  assert(result.error.includes("tenant_id"), `expected tenant_id error got ${result.error}`);
  return { status: "PASS", detail: "missing tenant_id → 400" };
});

test("init_owner rejects empty tenant_id", () => {
  const result = validateInitOwnerBody({ tenant_id: "  ", auth_mode: "metadata" });
  assert(result.valid === false, `expected invalid for whitespace-only tenant_id`);
  return { status: "PASS", detail: "blank tenant_id → 400" };
});

test("init_owner rejects missing auth_mode", () => {
  const result = validateInitOwnerBody({ tenant_id: "tenant_a" });
  assert(result.valid === false, `expected invalid got ${result.valid}`);
  assert(result.error.includes("auth_mode"), `expected auth_mode error got ${result.error}`);
  return { status: "PASS", detail: "missing auth_mode → 400" };
});

test("init_owner accepts valid body with null key_name/plan", () => {
  const result = validateInitOwnerBody({ tenant_id: "tenant_a", auth_mode: "legacy", key_name: null, plan: null });
  assert(result.valid === true, `expected valid got ${result.valid}`);
  return { status: "PASS", detail: "valid body accepted" };
});

test("init_owner accepts valid metadata body", () => {
  const result = validateInitOwnerBody({
    tenant_id: "acme_corp",
    auth_mode: "metadata",
    key_name: "prod-key",
    plan: "enterprise",
  });
  assert(result.valid === true, `expected valid got ${result.valid}`);
  return { status: "PASS", detail: "full metadata body accepted" };
});

// ── Report ─────────────────────────────────────────────────────────────────

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
lines.push("|" + center("Ownership Logic Unit Test", W) + "|");
lines.push("|" + center("(offline, no network)", W) + "|");
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
