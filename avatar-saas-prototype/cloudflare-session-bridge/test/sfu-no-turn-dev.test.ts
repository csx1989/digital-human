import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env } from "cloudflare:test";

const SESSION_PREFIX = "session:";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}no-turn-${label}-${crypto.randomUUID().slice(0, 8)}`;
}

// Clear SFU capacity before each test to prevent cross-test interference
beforeEach(async () => {
  const mgr = env.SFU_CAPACITY_MANAGER.get(env.SFU_CAPACITY_MANAGER.idFromName("global"));
  await mgr.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
});

function internalRest(
  url: string,
  init: { method: string; headers?: Record<string, string>; body?: string },
): Request {
  const headers = new Headers(init.headers);
  headers.set("X-Internal-REST", "true");
  return new Request(url, { method: init.method, headers, body: init.body });
}

// ── AVTR_SFU_REQUIRE_TURN=false dev/diagnostics path ──────────────────────
//
// When TURN secrets are absent and AVTR_SFU_REQUIRE_TURN=false, the Worker
// must NOT call the Cloudflare TURN API and must NOT include ice_servers in
// the Modal start body. The session should start successfully using
// direct-ICE-only candidates.
//
// This test runs against wrangler.test.no-turn.jsonc which has:
//   AVTR_SFU_REQUIRE_TURN: "false"
//   (no AVTR_SFU_TURN_KEY_ID or AVTR_SFU_TURN_KEY_API_TOKEN)

describe("SFU no-TURN dev path (AVTR_SFU_REQUIRE_TURN=false)", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("starts SFU session without TURN credentials and omits ice_servers from Modal body", async () => {
    const fetchLog: { url: string; method: string }[] = [];
    let modalStartBody: Record<string, unknown> | null = null;

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);
      const method = init?.method ?? "GET";
      fetchLog.push({ url: url.toString(), method });

      // SFU session creation
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
        return new Response(JSON.stringify({ sessionId: "mock-sfu-no-turn" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/new
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/new")) {
        return new Response(JSON.stringify({
          sessionDescription: { sdp: "v=0\r\no=- mock\r\ns=-\r\n", type: "answer" },
          tracks: [{ mid: "0" }],
          requiresImmediateRenegotiation: false,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // SFU tracks/close
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/close")) {
        return new Response(JSON.stringify({ requiresImmediateRenegotiation: false }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal start
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/start") {
        try {
          modalStartBody = JSON.parse(init.body as string) as Record<string, unknown>;
        } catch { /* best-effort */ }
        return new Response(JSON.stringify({
          session_id: `modal-no-turn`,
          avatar_id: "arnold",
          status: "ready",
          cache_hit: false,
          init_s: 0,
          active_sessions: [],
          state_note: "mock",
          vram_mb: 0,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // Modal status
      if (url.hostname.includes("modal.run") && url.pathname === "/status") {
        return new Response(JSON.stringify({
          status: "ready",
          frames_buffered: 1,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // Modal end
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
        return new Response(JSON.stringify({ status: "ended" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ error: `unexpected: ${url}` }), { status: 500 });
    });

    const sid = uniqueSid("dev");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    // Authorize session for SFU
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant_id: "legacy-beta-test",
          auth_mode: "legacy",
          sfu_beta_authorized: true,
        }),
      },
    ));

    // Start SFU session
    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));

    // Session must start successfully (200, not 503)
    expect(startRes.status).toBe(200);

    // Assert: ZERO calls to TURN credentials API
    const turnApiCalls = fetchLog.filter(
      (f) => f.url.includes("/turn/keys/") && f.url.includes("/credentials/generate-ice-servers"),
    );
    expect(turnApiCalls).toHaveLength(0);

    // Assert: Modal start body does NOT contain ice_servers
    expect(modalStartBody).not.toBeNull();
    expect(modalStartBody).not.toHaveProperty("ice_servers");

    // Cleanup
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));
  });
});
