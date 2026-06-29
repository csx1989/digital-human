import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env } from "cloudflare:workers";

const SESSION_PREFIX = "session:";
const TEST_HOOK = "test-hook-secret";

function internalRest(
  url: string,
  init: { method: string; headers?: Record<string, string>; body?: string },
): Request {
  const headers = new Headers(init.headers);
  headers.set("X-Internal-REST", "true");
  return new Request(url, { method: init.method, headers, body: init.body });
}

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}test-${label}-${crypto.randomUUID().slice(0, 8)}`;
}

// Clear SFU capacity before each test to prevent cross-test interference
beforeEach(async () => {
  const mgr = env.SFU_CAPACITY_MANAGER.get(env.SFU_CAPACITY_MANAGER.idFromName("global"));
  await mgr.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
});

async function doGetState(stub: DurableObjectStub): Promise<{ status: string }> {
  const res = await stub.fetch(internalRest(
    "https://test/v1/sessions/x/status",
    { method: "GET", headers: { "X-REST-Operation": "status" } },
  ));
  return await res.json() as { status: string };
}

function mockFetchEnd(counter?: { count: number }) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);
    if (init?.method === "POST" && url.pathname === "/end") {
      if (counter) counter.count++;
      return new Response(JSON.stringify({ status: "ended", session_id: "mock" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response(JSON.stringify({ error: "unexpected fetch in test" }), { status: 500 });
  });
}

describe("DELETE during starting — resurrection race", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("should NOT resurrect session and SHOULD call Modal cleanup when DELETE arrives during start", async () => {
    const endCalls = { count: 0 };
    mockFetchEnd(endCalls);

    const sid = uniqueSid("resurrect");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    const startReq = internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: {
          "X-REST-Operation": "start",
          "Content-Type": "application/json",
          "X-Test-Hook": TEST_HOOK,
          "X-Test-Hook-Delay": "2000",
        },
        body: JSON.stringify({ avatarId: "default" }),
      },
    );

    const endReq = internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    );

    const startPromise = stub.fetch(startReq);
    const endPromise = new Promise<Response>((resolve) => {
      setTimeout(() => { resolve(stub.fetch(endReq)); }, 500);
    });

    const [startRes, endRes] = await Promise.all([startPromise, endPromise]);

    // DELETE returns 200 with start_cancelled
    expect(endRes.status).toBe(200);
    const endBody = await endRes.json() as Record<string, unknown>;
    expect(endBody.status).toBe("ended");
    expect(endBody.start_cancelled).toBe(true);

    // Start returns 409 session_cancelled (no resurrection)
    expect(startRes.status).toBe(409);
    const startBody = await startRes.json() as { error: { code: string } };
    expect(startBody.error.code).toBe("session_cancelled");

    // Final state is ended, not active/ready
    const state = await doGetState(stub);
    expect(state.status).not.toBe("active");
    expect(state.status).not.toBe("ready");
    expect(state.status).toBe("ended");

    // Modal cleanup endpoint was actually called at least once
    expect(endCalls.count).toBeGreaterThanOrEqual(1);
  });

  it("should return 200 ended when DELETE arrives after normal start completes", async () => {
    mockFetchEnd();

    const sid = uniqueSid("normal-end");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: {
          "X-REST-Operation": "start",
          "Content-Type": "application/json",
          "X-Test-Hook": TEST_HOOK,
          "X-Test-Hook-Delay": "100",
        },
        body: JSON.stringify({ avatarId: "default" }),
      },
    ));
    expect(startRes.status).toBe(200);

    const endRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));
    expect(endRes.status).toBe(200);
    const endBody = await endRes.json() as Record<string, unknown>;
    expect(endBody.status).toBe("ended");
    expect(endBody.start_cancelled).toBeFalsy();

    const state = await doGetState(stub);
    expect(state.status).toBe("ended");
  });

  it("should return 409 session_starting when second start arrives during starting", async () => {
    const sid = uniqueSid("dup-start");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    const startReq1 = internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: {
          "X-REST-Operation": "start",
          "Content-Type": "application/json",
          "X-Test-Hook": TEST_HOOK,
          "X-Test-Hook-Delay": "2000",
        },
        body: JSON.stringify({ avatarId: "default" }),
      },
    );

    const startReq2 = internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: {
          "X-REST-Operation": "start",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ avatarId: "default" }),
      },
    );

    const start1Promise = stub.fetch(startReq1);
    const start2Promise = new Promise<Response>((resolve) => {
      setTimeout(() => { resolve(stub.fetch(startReq2)); }, 500);
    });

    const [start1Res, start2Res] = await Promise.all([start1Promise, start2Promise]);

    expect(start1Res.status).toBe(200);

    expect(start2Res.status).toBe(409);
    const start2Body = await start2Res.json() as { error: { code: string } };
    expect(start2Body.error.code).toBe("session_starting");
  });

  it("should be idempotent: repeated DELETE on already-ended session returns 200", async () => {
    const sid = uniqueSid("del-idempotent");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    const endReq = internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    );

    const res1 = await stub.fetch(endReq);
    const res1Clone = res1.clone();
    expect(res1.status).toBe(200);
    const body1 = await res1Clone.json() as Record<string, unknown>;
    expect(body1.status).toBe("ended");

    const res2 = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));
    expect(res2.status).toBe(200);
    const body2 = await res2.json() as Record<string, unknown>;
    expect(body2.status).toBe("ended");
  });
});
