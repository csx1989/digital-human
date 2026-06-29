import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";

// Clear SFU capacity before each test to prevent cross-test interference
beforeEach(async () => {
  const { env } = await import("cloudflare:test");
  const mgr = env.SFU_CAPACITY_MANAGER.get(env.SFU_CAPACITY_MANAGER.idFromName("global"));
  await mgr.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
});
import { env } from "cloudflare:workers";
import { createWsToken, verifyWsToken } from "../src/ws-token";
import { checkRateLimit } from "../src/rate-limit";

const SESSION_PREFIX = "session:";
const TEST_HOOK = "test-hook-secret";
const TOKEN_SECRET = "test-token-secret";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}test-${label}-${crypto.randomUUID().slice(0, 8)}`;
}

function internalRest(
  url: string,
  init: { method: string; headers?: Record<string, string>; body?: string },
): Request {
  const headers = new Headers(init.headers);
  headers.set("X-Internal-REST", "true");
  return new Request(url, { method: init.method, headers, body: init.body });
}

// ── Token purpose tests ────────────────────────────────────────────────────

describe("Token purpose enforcement", () => {
  it("ws_token is rejected for SFU routes (purpose:ws not in [sfu])", async () => {
    const sid = uniqueSid("tok-ws");
    const { token } = await createWsToken(TOKEN_SECRET, sid, "tenant-1", 300, "ws");
    const result = await verifyWsToken(TOKEN_SECRET, token, sid, ["sfu"]);
    expect(result.valid).toBe(false);
    expect(result.reason).toBe("purpose_not_allowed");
  });

  it("sfu_token is accepted for SFU routes (purpose:sf in [sfu])", async () => {
    const sid = uniqueSid("tok-sfu");
    const { token } = await createWsToken(TOKEN_SECRET, sid, "tenant-1", 300, "sfu");
    const result = await verifyWsToken(TOKEN_SECRET, token, sid, ["sfu"]);
    expect(result.valid).toBe(true);
    expect(result.tenantId).toBe("tenant-1");
  });

  it("sfu_token is rejected for WS routes (purpose:sf not in [ws])", async () => {
    const sid = uniqueSid("tok-sfu-ws");
    const { token } = await createWsToken(TOKEN_SECRET, sid, "tenant-1", 300, "sfu");
    const result = await verifyWsToken(TOKEN_SECRET, token, sid, ["ws"]);
    expect(result.valid).toBe(false);
    expect(result.reason).toBe("purpose_not_allowed");
  });

  it("ws_token is accepted for WS routes (default behavior)", async () => {
    const sid = uniqueSid("tok-ws-default");
    const { token } = await createWsToken(TOKEN_SECRET, sid, "tenant-1", 300, "ws");
    const result = await verifyWsToken(TOKEN_SECRET, token, sid);
    expect(result.valid).toBe(true);
  });

  it("sfu_token bound to different session is rejected", async () => {
    const sid1 = uniqueSid("tok-bind-1");
    const sid2 = uniqueSid("tok-bind-2");
    const { token } = await createWsToken(TOKEN_SECRET, sid1, "tenant-1", 300, "sfu");
    const result = await verifyWsToken(TOKEN_SECRET, token, sid2, ["sfu"]);
    expect(result.valid).toBe(false);
    expect(result.reason).toBe("session_mismatch");
  });
});

// ── Rate limit category tests ───────────────────────────────────────────────

describe("SFU route rate limit categories", () => {
  it("cleanup category is exempt (always allowed)", async () => {
    // cleanup maps to null config → checkRateLimit returns allowed: true
    const result = await checkRateLimit(env, "test-tenant", "cleanup");
    expect(result.allowed).toBe(true);
  });

  it("stream_connect category is NOT cleanup", async () => {
    // Verify these are different categories
    const cleanupResult = await checkRateLimit(env, "test-tenant", "cleanup");
    expect(cleanupResult.allowed).toBe(true);
    expect(cleanupResult.limit).toBeUndefined();
    // stream_connect should have a limit (even if binding is missing in test)
    const streamResult = await checkRateLimit(env, "test-tenant", "stream_connect");
    // In test env, binding may be missing → allowed:true with bindingMissing
    if (streamResult.bindingMissing) {
      expect(streamResult.limit).toBe(5);
    }
  });
});

// ── DO-level test: subscribe response has no sfu_session_id ─────────────────

describe("SFU subscribe response — no raw session ID leak", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does not include sfu_session_id in the response body", async () => {
    const sid = uniqueSid("sfu-leak");

    // Create stub BEFORE the mock so the mock can simulate Modal's
    // publisher-offer callback (which runs concurrently during start).
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    // Mock fetch to intercept both SFU API and Modal API
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);

      // Cloudflare TURN credentials API
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/turn/keys/") && url.pathname.includes("/credentials/generate-ice-servers")) {
        return new Response(JSON.stringify({
          iceServers: [
            { urls: ["stun:stun.cloudflare.com:3478"] },
            {
              urls: [
                "turn:turn.cloudflare.com:3478?transport=udp",
                "turn:turn.cloudflare.com:3478?transport=tcp",
                "turns:turn.cloudflare.com:5349?transport=tcp",
              ],
              username: "test-turn-user-1234567890abcdef",
              credential: "test-turn-cred-0987654321fedcba",
            },
          ],
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // SFU session creation
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
        return new Response(JSON.stringify({ sessionId: "mock-sfu-session-123" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/new (publisher + subscriber)
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/new")) {
        return new Response(JSON.stringify({
          sessionDescription: { sdp: "v=0\r\no=- mock\r\ns=-\r\n", type: "answer" },
          tracks: [{ mid: "0" }],
          requiresImmediateRenegotiation: false,
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/close
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/close")) {
        return new Response(JSON.stringify({ requiresImmediateRenegotiation: false }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal start-session
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/start") {
        return new Response(JSON.stringify({
          session_id: `modal-${sid}`,
          avatar_id: "arnold",
          status: "ready",
          cache_hit: false,
          init_s: 0,
          active_sessions: [],
          state_note: "mock",
          vram_mb: 0,
          publisher_track_name: "avatar-video-mock",
          publisher_mode: "webrtc_sfu",
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal end-session
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
        return new Response(JSON.stringify({
          session_id: `modal-${sid}`,
          avatar_id: "arnold",
          status: "ended",
          chunks_rendered: 0,
          fmp4_media_count: 0,
          fmp4_media_total_bytes: 0,
          duration_s: 0,
          ffmpeg_rc: 0,
          zombie_check: "clean",
          handoff_closed: true,
          threads_joined: true,
          active_sessions: [],
          vram_mb: 0,
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
    });

    // Phase 3F: authorize session for SFU before starting.
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

    // Start session — SFU mode (shouldTrySfu=true since env is fully configured)
    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: {
          "X-REST-Operation": "start",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    const startBody = await startRes.json() as Record<string, unknown>;
    expect(startBody.transport).toBe("webrtc_sfu");

    // Simulate Modal's publisher-offer callback (lazy SFU session creation).
    // In production this runs during startModalSession(); here it runs after
    // start since the mock Modal can't call back concurrently.
    const offerRes = await stub.fetch(internalRest(
      `https://test/internal/sessions/${sid}/publisher-offer`,
      {
        method: "POST",
        headers: { "X-SFU-Operation": "publisher_offer", "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: "v=0\r\no=- mock-pub\r\ns=-\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\n",
          type: "offer", mid: "0", track_name: "avatar-video-mock", kind: "video",
        }),
      },
    ));
    expect(offerRes.status).toBe(200);

    // Now call subscribe
    const sfuTokenResult = await createWsToken(TOKEN_SECRET, sid, "legacy", 300, "sfu");
    const subRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/sfu/subscribe`,
      {
        method: "POST",
        headers: {
          "X-SFU-Operation": "sfu_subscribe",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ token: sfuTokenResult.token }),
      },
    ));
    const subBody = await subRes.json() as Record<string, unknown>;

    // Verify sfu_session_id is NOT in the response
    expect(subBody).not.toHaveProperty("sfu_session_id");
    // Verify expected fields ARE present
    expect(subBody).toHaveProperty("offer_sdp");
    expect(subBody).toHaveProperty("offer_type");
    expect(subBody).toHaveProperty("requires_renegotiation");
    expect(subBody).toHaveProperty("mids");

    // Cleanup
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));
  });
});

// ── DO-level test: resurrection race with SFU ───────────────────────────────

describe("DELETE during SFU start — resurrection race preserved", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("preserves ended/cancelled state when DELETE arrives during SFU start", async () => {
    const endCalls = { count: 0 };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);

      // Cloudflare TURN credentials API
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/turn/keys/") && url.pathname.includes("/credentials/generate-ice-servers")) {
        return new Response(JSON.stringify({
          iceServers: [
            { urls: ["stun:stun.cloudflare.com:3478"] },
            {
              urls: [
                "turn:turn.cloudflare.com:3478?transport=udp",
                "turn:turn.cloudflare.com:3478?transport=tcp",
                "turns:turn.cloudflare.com:5349?transport=tcp",
              ],
              username: "test-turn-user-1234567890abcdef",
              credential: "test-turn-cred-0987654321fedcba",
            },
          ],
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // SFU session creation — succeed
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
        return new Response(JSON.stringify({ sessionId: "mock-sfu-race" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/close
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/close")) {
        return new Response(JSON.stringify({ requiresImmediateRenegotiation: false }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal start-session — delay to simulate slow start
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/start") {
        await new Promise(r => setTimeout(r, 2000)); // 2s delay
        return new Response(JSON.stringify({
          session_id: "modal-sfu-race",
          avatar_id: "arnold",
          status: "ready",
          cache_hit: false,
          init_s: 0,
          active_sessions: [],
          state_note: "mock",
          vram_mb: 0,
          publisher_track_name: "avatar-video-race",
          publisher_mode: "webrtc_sfu",
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // Modal end-session
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
        endCalls.count++;
        return new Response(JSON.stringify({
          session_id: "modal-sfu-race",
          avatar_id: "arnold",
          status: "ended",
          chunks_rendered: 0,
          fmp4_media_count: 0,
          fmp4_media_total_bytes: 0,
          duration_s: 0,
          ffmpeg_rc: 0,
          zombie_check: "clean",
          handoff_closed: true,
          threads_joined: true,
          active_sessions: [],
          vram_mb: 0,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      return new Response(JSON.stringify({ error: `unexpected: ${url}` }), { status: 500 });
    });

    const sid = uniqueSid("sfu-resurrect");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    // Phase 3F: authorize session for SFU before starting.
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

    const startReq = internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "default", transport: "webrtc_sfu" }),
      },
    );

    const endReq = internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    );

    // Start + send DELETE after 500ms
    const startPromise = stub.fetch(startReq);
    const endPromise = new Promise<Response>((resolve) => {
      setTimeout(() => { resolve(stub.fetch(endReq)); }, 500);
    });

    const [startRes, endRes] = await Promise.all([startPromise, endPromise]);

    // Start returns 409 session_cancelled (no resurrection)
    expect(startRes.status).toBe(409);
    const startBody = await startRes.json() as { error: { code: string } };
    expect(startBody.error.code).toBe("session_cancelled");

    // DELETE returns 200
    expect(endRes.status).toBe(200);

    // Final state is ended
    const stateRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/status`,
      { method: "GET", headers: { "X-REST-Operation": "status" } },
    ));
    const state = await stateRes.json() as Record<string, unknown>;
    expect(state.status).not.toBe("active");
    expect(state.status).not.toBe("ready");

    // ── Blocker 2 proof: #cleanupSfu() persisted sfuSessionId = null ──
    // After cancellation, #cleanupSfu() ran in the start handler's cancellation
    // path and must have persisted cleared SFU state. We verify by calling
    // publisher-close, which calls #loadState() then checks sfuSessionId.
    //   - If cleanup persisted: sfuSessionId === null → "no_sfu_session"
    //   - If cleanup did NOT persist (bug): stale sfuSessionId loaded from
    //     storage → handler continues → "no_publisher_tracks" (mids were null
    //     from pre-save, callback never ran because Modal start was cancelled)
    const postCancelClose = await stub.fetch(internalRest(
      `https://test/internal/sessions/${sid}/publisher-close`,
      { method: "POST" },
    ));
    const postCancelCloseBody = await postCancelClose.json() as Record<string, unknown>;
    expect(postCancelCloseBody.status).toBe("no_sfu_session");
  });
});

// ── Blocker 3: Real callback-during-start test ──────────────────────────────
//
// In production, Modal calls /internal/sessions/:id/publisher-offer DURING
// startup (while the DO is still awaiting the Modal /start response). This
// test simulates that by:
//   1. Giving the mocked Modal /start a delay (1500ms).
//   2. Sending the publisher-offer request concurrently from the TEST CODE
//      at 300ms — while the DO is still awaiting the Modal fetch.
//
// We CANNOT call stub.fetch() from inside the fetch mock because Cloudflare
// Workers prohibits I/O on behalf of a different Durable Object within the
// same isolate. Sending the request from the test code (top-level) works
// because the DO input gate opens during the await, exactly as it does for
// the resurrection race test.
//
// This proves:
//   1. The internal publisher-offer callback succeeds while Modal start is
//      still in flight (the DO input gate allows the concurrent request).
//   2. The publisher mid written by the callback survives the success-block
//      state assignment (Blocker 1 fix).
//   3. publisher-close uses the preserved mid and sends a real Cloudflare
//      tracks/close request with the exact body shape
//      {"tracks":[{"mid":"0"}],"force":true}.

describe("Publisher-offer callback during Modal start", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("preserves publisher mid written by callback; cleanup uses real mid", async () => {
    const sid = uniqueSid("callback");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    // Capture callback response and tracks/close request bodies
    let callbackResponse: Record<string, unknown> | null = null;
    const tracksCloseBodies: Array<Record<string, unknown>> = [];

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);

      // Cloudflare TURN credentials API
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/turn/keys/") && url.pathname.includes("/credentials/generate-ice-servers")) {
        return new Response(JSON.stringify({
          iceServers: [
            { urls: ["stun:stun.cloudflare.com:3478"] },
            {
              urls: [
                "turn:turn.cloudflare.com:3478?transport=udp",
                "turn:turn.cloudflare.com:3478?transport=tcp",
                "turns:turn.cloudflare.com:5349?transport=tcp",
              ],
              username: "test-turn-user-1234567890abcdef",
              credential: "test-turn-cred-0987654321fedcba",
            },
          ],
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      // SFU session creation
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
        return new Response(JSON.stringify({ sessionId: "mock-sfu-callback" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/new (publisher offer + subscriber subscribe)
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/new")) {
        return new Response(JSON.stringify({
          sessionDescription: { sdp: "v=0\r\no=- mock-answer\r\ns=-\r\n", type: "answer" },
          tracks: [{ mid: "0" }],
          requiresImmediateRenegotiation: false,
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/close — capture the raw request body for shape assertion
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/close")) {
        let parsedBody: Record<string, unknown> = {};
        if (init?.body) {
          try { parsedBody = JSON.parse(init.body as string); } catch { /* ignore */ }
        }
        tracksCloseBodies.push(parsedBody);
        return new Response(JSON.stringify({ requiresImmediateRenegotiation: false }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal start — delayed to simulate slow startup.
      // The publisher-offer callback is sent concurrently from the test
      // code (NOT from this mock) while this delay is in progress.
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/start") {
        await new Promise(r => setTimeout(r, 1500));
        return new Response(JSON.stringify({
          session_id: `modal-${sid}`,
          avatar_id: "arnold",
          status: "ready",
          cache_hit: false,
          init_s: 0,
          active_sessions: [],
          state_note: "mock",
          vram_mb: 0,
          publisher_track_name: "avatar-video-callback",
          publisher_mode: "webrtc_sfu",
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // Modal end-session
      if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
        return new Response(JSON.stringify({
          session_id: `modal-${sid}`,
          avatar_id: "arnold",
          status: "ended",
          chunks_rendered: 0,
          fmp4_media_count: 0,
          fmp4_media_total_bytes: 0,
          duration_s: 0,
          ffmpeg_rc: 0,
          zombie_check: "clean",
          handoff_closed: true,
          threads_joined: true,
          active_sessions: [],
          vram_mb: 0,
        }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
    });

    // Phase 3F: authorize session for SFU before starting.
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

    // ── Phase 1: Start session in SFU mode (Modal start takes 1500ms) ──
    const startReq = internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    );
    const startPromise = stub.fetch(startReq);

    // ── Phase 2: Send publisher-offer concurrently at 300ms ──
    // This simulates Modal calling /internal/sessions/:id/publisher-offer
    // while the DO is still awaiting the Modal /start response. The DO
    // input gate is open during the await, so this request enters the DO.
    const publisherOfferReq = internalRest(
      `https://test.worker.dev/internal/sessions/${sid}/publisher-offer`,
      {
        method: "POST",
        body: JSON.stringify({
          sdp: "v=0\r\no=- modal-publisher\r\ns=-\r\n",
          type: "offer",
          mid: "0",
          track_name: "avatar-video-callback",
          kind: "video",
        }),
      },
    );

    const publisherOfferPromise = new Promise<Response>((resolve) => {
      setTimeout(() => { resolve(stub.fetch(publisherOfferReq)); }, 300);
    });

    // Wait for both start and publisher-offer to complete
    const [startRes, cbRes] = await Promise.all([startPromise, publisherOfferPromise]);

    callbackResponse = await cbRes.json() as Record<string, unknown>;
    const startBody = await startRes.json() as Record<string, unknown>;

    // ── Proof: start succeeded with SFU transport ──
    expect(startRes.status).toBe(200);
    expect(startBody.transport).toBe("webrtc_sfu");

    // ── Proof: internal publisher-offer callback ran during Modal start ──
    expect(callbackResponse).not.toBeNull();
    expect(callbackResponse).toHaveProperty("answer_sdp");

    // ── Proof: publisher mid is preserved in session state after start ──
    // Call publisher-close. If Blocker 1 bug exists (mid overwritten to null
    // by the success block), publisher-close returns "no_publisher_tracks".
    // If fixed, it returns "closed" because sfuPublisherMids.video === "0".
    const closeRes = await stub.fetch(internalRest(
      `https://test/internal/sessions/${sid}/publisher-close`,
      { method: "POST" },
    ));
    const closeBody = await closeRes.json() as Record<string, unknown>;
    expect(closeBody.status).toBe("closed");

    // ── Proof: publisher-close sent real Cloudflare tracks/close with the
    //    exact top-level body shape {"tracks":[{"mid":"0"}],"force":true} ──
    expect(tracksCloseBodies.length).toBeGreaterThanOrEqual(1);
    const closeReqBody = tracksCloseBodies[0];
    expect(closeReqBody).toEqual({
      tracks: [{ mid: "0" }],
      force: true,
    });
  });
});
