import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env, SELF } from "cloudflare:test";
import { createWsToken } from "../src/ws-token";

const SESSION_PREFIX = "session:";
const TOKEN_SECRET = "test-token-secret";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}sub-${label}-${crypto.randomUUID().slice(0, 8)}`;
}

function internalRest(
  url: string,
  init: { method: string; headers?: Record<string, string>; body?: string },
): Request {
  const headers = new Headers(init.headers);
  headers.set("X-Internal-REST", "true");
  return new Request(url, { method: init.method, headers, body: init.body });
}

// Clear SFU capacity before each test to prevent cross-test interference
beforeEach(async () => {
  const mgr = env.SFU_CAPACITY_MANAGER.get(env.SFU_CAPACITY_MANAGER.idFromName("global"));
  await mgr.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
});

async function initOwner(sid: string, tenantId: string) {
  const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
  await stub.fetch(internalRest(
    `https://test/v1/sessions/${sid}/init_owner`,
    {
      method: "POST",
      headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
      body: JSON.stringify({ tenant_id: tenantId, auth_mode: "legacy", sfu_beta_authorized: true }),
    },
  ));
  return stub;
}

// ── Regression tests: SFU subscribe route auth/ownership ───────────────────
//
// These tests exercise the Worker's handleSfuBrowserRoute layer:
//   - Token verification (valid/invalid/missing)
//   - Ownership check (tenant matching)
//
// They do NOT start a full SFU session (that requires Modal mocks).
// For the valid-token case, the DO may return sfu_wrong_transport (409)
// since the session hasn't been started — but that proves the request
// passed through auth + ownership and reached the DO.

describe("SFU subscribe route — auth and ownership regression", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("returns 401 when sfu_token is missing", async () => {
    const sid = uniqueSid("notoken");
    await initOwner(sid, "legacy-beta-test");

    const subResp = await SELF.fetch(
      `https://test/v1/sessions/${sid}/sfu/subscribe`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: [] }),
      },
    );

    expect(subResp.status).toBe(401);
    const body = await subResp.json() as Record<string, unknown>;
    expect((body.error as Record<string, unknown>)?.code).toBe("ws_token_missing");
  });

  it("returns 401 when sfu_token is invalid", async () => {
    const sid = uniqueSid("badtoken");
    await initOwner(sid, "legacy-beta-test");

    const subResp = await SELF.fetch(
      `https://test/v1/sessions/${sid}/sfu/subscribe?token=invalid.token.here`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: [] }),
      },
    );

    expect(subResp.status).toBe(401);
    const body = await subResp.json() as Record<string, unknown>;
    expect((body.error as Record<string, unknown>)?.code).toBe("ws_token_invalid");
  });

  it("returns 404 when sfu_token tenant does not match owner tenant", async () => {
    const sid = uniqueSid("mismatch");
    await initOwner(sid, "legacy-beta-test");

    // Mint token with WRONG tenant
    const wrongToken = await createWsToken(TOKEN_SECRET, sid, "wrong-tenant", 300, "sfu");

    const subResp = await SELF.fetch(
      `https://test/v1/sessions/${sid}/sfu/subscribe?token=${wrongToken.token}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: [] }),
      },
    );

    expect(subResp.status).toBe(404);
    const body = await subResp.json() as Record<string, unknown>;
    expect((body.error as Record<string, unknown>)?.code).toBe("not_found");
  });

  it("passes ownership check when sfu_token tenant matches owner (reaches DO)", async () => {
    const sid = uniqueSid("ok");
    const tenantId = "legacy-beta-test";
    await initOwner(sid, tenantId);

    // Mint token with MATCHING tenant
    const tokenResult = await createWsToken(TOKEN_SECRET, sid, tenantId, 300, "sfu");

    const subResp = await SELF.fetch(
      `https://test/v1/sessions/${sid}/sfu/subscribe?token=${tokenResult.token}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tracks: [] }),
      },
    );

    // Should NOT be 404 (ownership passed) or 401 (token valid).
    // Will likely be 409 sfu_wrong_transport since no SFU session is started.
    expect(subResp.status).not.toBe(404);
    expect(subResp.status).not.toBe(401);

    const body = await subResp.json() as Record<string, unknown>;
    const errCode = (body.error as Record<string, unknown>)?.code;
    expect(errCode).not.toBe("not_found");
  });
});
