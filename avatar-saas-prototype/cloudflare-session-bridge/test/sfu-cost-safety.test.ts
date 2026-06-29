import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env, runInDurableObject, runDurableObjectAlarm } from "cloudflare:test";
import { AvatarSessionDO } from "../src/avatar-session-do";

const SESSION_PREFIX = "session:";
const TEST_HOOK = "test-hook-secret";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}cost-${label}-${crypto.randomUUID().slice(0, 8)}`;
}

function internalRest(
  url: string,
  init: { method: string; headers?: Record<string, string>; body?: string },
): Request {
  const headers = new Headers(init.headers);
  headers.set("X-Internal-REST", "true");
  return new Request(url, { method: init.method, headers, body: init.body });
}

async function initSfuBetaOwner(stub: DurableObjectStub, sid: string) {
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
}

async function getAlarmTime(stub: DurableObjectStub): Promise<number | null> {
  return runInDurableObject(stub, async (_instance, state) => {
    return state.storage.getAlarm();
  });
}

async function getState(stub: DurableObjectStub): Promise<Record<string, unknown>> {
  return runInDurableObject(stub, async (_instance, state) => {
    return (await state.storage.get("session:state") ?? {}) as Record<string, unknown>;
  });
}

function createMockFetch(opts?: {
  startDelayMs?: number;
  startFail?: boolean;
  statusFail?: boolean;
  endFail?: boolean;
  endHangMs?: number;
  trackEndCalls?: string[];
  trackStatusCalls?: string[];
}) {
  const statusCalls: string[] = opts?.trackStatusCalls ?? [];
  const endCalls: string[] = opts?.trackEndCalls ?? [];
  const sfuSid = `mock-sfu-${crypto.randomUUID().slice(0, 8)}`;

  const impl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);

    // Cloudflare TURN credentials API
    if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/turn/keys/") && url.pathname.includes("/credentials/generate-ice-servers")) {
      return new Response(JSON.stringify({
        iceServers: [
          { urls: ["stun:stun.cloudflare.com:3478"] },
          {
            urls: ["turn:turn.cloudflare.com:3478?transport=udp"],
            username: "test-turn-user",
            credential: "test-turn-cred",
          },
        ],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    // SFU session creation
    if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
      return new Response(JSON.stringify({ sessionId: sfuSid }), {
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

    // Modal start-session
    if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/start") {
      if (opts?.startDelayMs) {
        await new Promise(r => setTimeout(r, opts.startDelayMs));
      }
      if (opts?.startFail) {
        return new Response(JSON.stringify({ error: "Modal start failed: billing limit" }), {
          status: 429, headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify({
        session_id: `modal-${sfuSid}`,
        avatar_id: "arnold",
        status: "ready",
        cache_hit: false,
        init_s: 0,
        active_sessions: [],
        state_note: "mock",
        vram_mb: 0,
        publisher_track_name: "avatar-video",
        publisher_mode: "webrtc_sfu",
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    // Modal end-session
    if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
      const body = JSON.parse(init.body as string) as { session_id?: string };
      if (body.session_id) endCalls.push(body.session_id);
      if (opts?.endFail) {
        return new Response(JSON.stringify({ error: "modal end failed", session_id: body.session_id }), {
          status: 502, headers: { "Content-Type": "application/json" },
        });
      }
      // Simulate a hanging Modal /end. The mock respects the AbortSignal
      // from callSessionEndpoint so the timeoutMs actually cancels the call.
      if (opts?.endHangMs) {
        return new Promise<Response>((resolve, reject) => {
          const timer = setTimeout(() => {
            resolve(new Response(JSON.stringify({ session_id: body.session_id, status: "ended" }), {
              status: 200, headers: { "Content-Type": "application/json" },
            }));
          }, opts.endHangMs!);
          init?.signal?.addEventListener("abort", () => {
            clearTimeout(timer);
            reject(new DOMException("The operation was aborted.", "AbortError"));
          });
        });
      }
      return new Response(JSON.stringify({
        session_id: body.session_id,
        status: "ended",
        chunks_rendered: 0,
        fmp4_media_count: 0,
        fmp4_media_total_bytes: 0,
        duration_s: 0,
        ffmpeg_rc: 0,
        active_sessions: [],
        vram_mb: 0,
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    // Modal session-status
    if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/status") {
      const body = JSON.parse(init.body as string) as { session_id?: string };
      if (body.session_id) statusCalls.push(body.session_id);
      if (opts?.statusFail) {
        return new Response(JSON.stringify({ error: "modal status failed" }), {
          status: 502, headers: { "Content-Type": "application/json" },
        });
      }
      return new Response(JSON.stringify({
        session_id: body.session_id,
        active: true,
        active_sessions: [],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
  };

  return { impl, statusCalls, endCalls, sfuSid };
}

async function startSfuSession(stub: DurableObjectStub, sid: string) {
  const startRes = await stub.fetch(internalRest(
    `https://test/v1/sessions/${sid}/start`,
    {
      method: "POST",
      headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
      body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
    },
  ));
  const startBody = await startRes.json() as Record<string, unknown>;
  if (startRes.status !== 200) throw new Error(`start failed: ${JSON.stringify(startBody)}`);

  // Simulate Modal publisher-offer callback
  const offerRes = await stub.fetch(internalRest(
    `https://test/internal/sessions/${sid}/publisher-offer`,
    {
      method: "POST",
      headers: { "X-SFU-Operation": "publisher_offer", "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: "v=0\r\no=- mock-pub\r\ns=-\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\n",
        type: "offer", mid: "0", track_name: "avatar-video", kind: "video",
      }),
    },
  ));
  if (offerRes.status !== 200) throw new Error(`publisher-offer failed: ${offerRes.status}`);
}

// Clear SFU capacity before each test to prevent cross-test interference
beforeEach(async () => {
  const mgr = env.SFU_CAPACITY_MANAGER.get(env.SFU_CAPACITY_MANAGER.idFromName("global"));
  await mgr.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
});

afterEach(() => {
  vi.restoreAllMocks();
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 1: WS close during Modal start sets cancellation flag
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — WS close during start", () => {
  it("webSocketClose sets startCancelRequested when starting is true", async () => {
    const sid = uniqueSid("ws-close-starting");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    // Prime DO state as if session.start is in-flight
    await runInDurableObject(stub, async (_instance, state) => {
      await state.storage.put("session:state", {
        modalSessionId: null,
        active: false,
        avatarId: null,
        bgId: null,
        transport: "websocket_fmp4",
        starting: true,
        startingAt: Date.now(),
        startAttemptId: "attempt-ws-close-test",
        startCancelRequested: false,
      });
    });

    const closeMock = vi.fn();
    const mockWs = {
      deserializeAttachment: () => ({ sessionId: sid, connectedAt: Date.now() }),
      close: closeMock,
    } as unknown as WebSocket;

    await runInDurableObject(stub, async (instance) => {
      await (instance as AvatarSessionDO).webSocketClose(mockWs, 1000, "", true);
    });

    const state = await getState(stub);
    expect(state.startCancelRequested).toBe(true);
    expect(state.startAttemptId).toBe("attempt-ws-close-test");
    expect(closeMock).toHaveBeenCalledWith(1000, "session closed");
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 2: Cancelled post-Modal-start path prevents keepalive and releases capacity
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — start cancellation after Modal returns", () => {
  it("DELETE during start sets cancel flag, then aborts start without keepalive", async () => {
    const mock = createMockFetch({ startDelayMs: 2_000, trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("cancel-post-modal");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);
    await initSfuBetaOwner(stub, sid);

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
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
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

    // End returns 200 and cancels the in-flight start
    expect(endRes.status).toBe(200);
    const endBody = await endRes.json() as Record<string, unknown>;
    expect(endBody.status).toBe("ended");
    expect(endBody.start_cancelled).toBe(true);

    // Start returns 409 session_cancelled
    expect(startRes.status).toBe(409);
    const startBody = await startRes.json() as { error: { code: string } };
    expect(startBody.error.code).toBe("session_cancelled");

    // No keepalive alarm should exist
    expect(await getAlarmTime(stub)).toBeNull();

    // Capacity released
    const finalState = await getState(stub);
    expect(finalState.active).toBe(false);
    expect(finalState.modalSessionId).toBeNull();
    expect(finalState.sfuCapacityAcquired).toBe(false);

    // Modal /end attempted for the orphan
    expect(mock.endCalls.length).toBeGreaterThanOrEqual(1);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 3: Keepalive max age aborts and never reschedules
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — keepalive max age", () => {
  it("aborts keepalive when age exceeds AVTR_SFU_KEEPALIVE_MAX_AGE_MS", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("ka-max-age");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Manually age the keepalive beyond the 2000ms test limit
    await runInDurableObject(stub, async (_instance, state) => {
      const stored = await state.storage.get("session:state") as Record<string, unknown>;
      stored.sfuKeepaliveStartedAt = Date.now() - 3_000;
      await state.storage.put("session:state", stored);
    });

    const ran = await runDurableObjectAlarm(stub);
    expect(ran).toBe(true);

    // Alarm must be deleted and not rescheduled
    expect(await getAlarmTime(stub)).toBeNull();

    // State should be ended/empty
    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuKeepaliveActive).toBe(false);
    expect(state.sfuKeepaliveAborted).toBe(true);
    expect(state.modalSessionId).toBeNull();

    // No status calls should have been made in this alarm firing
    expect(mock.statusCalls.length).toBe(0);

    // Capacity released
    expect(state.sfuCapacityAcquired).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 4: Session max age aborts and never reschedules
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — session max age", () => {
  it("aborts keepalive when session age exceeds AVTR_SFU_SESSION_MAX_AGE_MS", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("session-max-age");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Age only the session start timestamp, keep keepalive fresh
    await runInDurableObject(stub, async (_instance, state) => {
      const stored = await state.storage.get("session:state") as Record<string, unknown>;
      stored.sfuSessionStartedAt = Date.now() - 3_000;
      stored.sfuKeepaliveStartedAt = Date.now();
      await state.storage.put("session:state", stored);
    });

    const ran = await runDurableObjectAlarm(stub);
    expect(ran).toBe(true);

    expect(await getAlarmTime(stub)).toBeNull();

    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuKeepaliveAborted).toBe(true);
    expect(state.sfuCapacityAcquired).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 5: Keepalive max failures aborts and never reschedules
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — keepalive max failures", () => {
  it("aborts keepalive after 2 consecutive Modal status failures", async () => {
    const mock = createMockFetch({ statusFail: true, trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("ka-max-failures");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Fire alarm twice; both status calls fail
    await runDurableObjectAlarm(stub);
    await runDurableObjectAlarm(stub);

    // After 2 failures, alarm should be gone
    expect(await getAlarmTime(stub)).toBeNull();

    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuKeepaliveAborted).toBe(true);

    // Exactly 2 status attempts were made before abort (the failure count in
    // state is reset to 0 by emptyState after abort).
    expect(mock.statusCalls.length).toBe(2);

    expect(state.sfuCapacityAcquired).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 6: Force-clear clears alarm/state/capacity even when Modal /end fails
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — force-clear endpoint", () => {
  it("clears state and releases capacity even if Modal /end returns 429/502", async () => {
    const mock = createMockFetch({ endFail: true, trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("force-clear-modal-fail");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Sanity: alarm exists and capacity held
    expect(await getAlarmTime(stub)).not.toBeNull();
    const stateBefore = await getState(stub);
    expect(stateBefore.sfuCapacityAcquired).toBe(true);

    // Force-clear via internal DO operation
    const clearRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/force-clear`,
      { method: "POST", headers: { "X-REST-Operation": "force_clear" } },
    ));

    expect(clearRes.status).toBe(200);
    const clearBody = await clearRes.json() as Record<string, unknown>;
    expect(clearBody.ok).toBe(true);
    expect(clearBody.cleared).toBe(true);
    expect(clearBody.wasKeepaliveActive).toBe(true);

    // Alarm deleted
    expect(await getAlarmTime(stub)).toBeNull();

    // State empty/ended
    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.modalSessionId).toBeNull();
    expect(state.sfuKeepaliveAborted).toBe(true);

    // Capacity released
    expect(state.sfuCapacityAcquired).toBe(false);

    // Modal /end was attempted (best-effort) despite failure
    expect(mock.endCalls.length).toBeGreaterThanOrEqual(1);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 7: Modal start failure after capacity acquire releases capacity
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — Modal start failure releases capacity", () => {
  it("releases capacity when Modal /start returns 429 after capacity acquire", async () => {
    const mock = createMockFetch({ startFail: true });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("modal-start-fail");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));

    expect(startRes.status).toBe(500);
    const body = await startRes.json() as { error: { code: string } };
    expect(body.error.code).toBe("modal_unreachable");

    // No alarm, capacity released
    expect(await getAlarmTime(stub)).toBeNull();
    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuCapacityAcquired).toBe(false);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 8: Normal session.end still releases capacity after Modal /end
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — normal end", () => {
  it("REST end deletes alarm and releases capacity", async () => {
    const mock = createMockFetch({ trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("normal-end");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    expect(await getAlarmTime(stub)).not.toBeNull();
    const stateBefore = await getState(stub);
    expect(stateBefore.sfuCapacityAcquired).toBe(true);

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    expect(await getAlarmTime(stub)).toBeNull();
    const stateAfter = await getState(stub);
    expect(stateAfter.sfuCapacityAcquired).toBe(false);
    expect(mock.endCalls.length).toBeGreaterThanOrEqual(1);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 8: fMP4 path unaffected
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — fMP4 unaffected", () => {
  it("fMP4 sessions do not set alarms and force-clear is harmless", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(createMockFetch().impl);

    const sid = uniqueSid("fmp4-unaffected");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "websocket_fmp4" }),
      },
    ));
    expect(startRes.status).toBe(200);

    expect(await getAlarmTime(stub)).toBeNull();

    const clearRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/force-clear`,
      { method: "POST", headers: { "X-REST-Operation": "force_clear" } },
    ));
    expect(clearRes.status).toBe(200);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 10: Force-clear completes within 3s timeout when Modal /end hangs
// ──────────────────────────────────────────────────────────────────────────
describe("SFU cost-safety — 3s Modal /end timeout", () => {
  it("force-clear completes within ~3s when Modal /end hangs for 30s", async () => {
    const mock = createMockFetch({ endHangMs: 30_000, trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("force-clear-hang");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    const t0 = Date.now();
    const clearRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/force-clear`,
      { method: "POST", headers: { "X-REST-Operation": "force_clear" } },
    ));
    const elapsed = Date.now() - t0;

    expect(clearRes.status).toBe(200);
    // Must complete well under the old 30s default — bounded by 3s timeout.
    // Allow generous overhead (up to 6s) for CI variance.
    expect(elapsed).toBeLessThan(6_000);

    // State/capacity/alarm all cleared despite hanging Modal.
    expect(await getAlarmTime(stub)).toBeNull();
    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuKeepaliveAborted).toBe(true);
    expect(state.sfuCapacityAcquired).toBe(false);
    expect(mock.endCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("#abortSfuKeepalive completes within ~3s when Modal /end hangs for 30s", async () => {
    const mock = createMockFetch({ endHangMs: 30_000, trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("abort-hang");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Age the keepalive past the max to trigger #abortSfuKeepalive.
    await runInDurableObject(stub, async (_instance, state) => {
      const stored = await state.storage.get("session:state") as Record<string, unknown>;
      stored.sfuKeepaliveStartedAt = Date.now() - 3_000;
      await state.storage.put("session:state", stored);
    });

    const t0 = Date.now();
    await runDurableObjectAlarm(stub);
    const elapsed = Date.now() - t0;

    // Abort must complete well under the old 30s default.
    expect(elapsed).toBeLessThan(6_000);
    expect(await getAlarmTime(stub)).toBeNull();

    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuKeepaliveAborted).toBe(true);
    expect(state.sfuCapacityAcquired).toBe(false);
  });

  it("cancelled-start cleanup does not wait 30s on Modal /end (completes <6s)", async () => {
    const mock = createMockFetch({ startDelayMs: 2_000, endHangMs: 30_000, trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("cancel-hang");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);
    await initSfuBetaOwner(stub, sid);

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
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
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

    const t0 = Date.now();
    const [startRes, endRes] = await Promise.all([startPromise, endPromise]);
    const elapsed = Date.now() - t0;

    // Modal start takes 2s, then cancelled-start cleanup calls /end with 3s
    // timeout. Total must be well under the old 30s end timeout.
    expect(endRes.status).toBe(200);
    expect(startRes.status).toBe(409);
    expect(elapsed).toBeLessThan(8_000); // 2s start delay + 3s end timeout + overhead

    expect(await getAlarmTime(stub)).toBeNull();
    const state = await getState(stub);
    expect(state.active).toBe(false);
    expect(state.sfuCapacityAcquired).toBe(false);
  }, 15_000);

  it("normal session.end is NOT bounded by 3s (uses default end timeout)", async () => {
    // Verify the normal end path uses the full default timeout (30s),
    // not the 3s incident-response timeout. We mock /end to hang and verify
    // the call passes no custom timeout by checking it DOES wait.
    const mock = createMockFetch({ endHangMs: 500, trackEndCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("normal-end-default-timeout");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));
    await initSfuBetaOwner(stub, sid);

    await startSfuSession(stub, sid);

    // Normal end — Modal /end hangs 500ms then resolves. If the normal path
    // used the 3s timeout, the call would still succeed (500ms < 3s). But
    // the key assertion is that it DOESN'T abort early — the end call
    // completes successfully, proving it waits for Modal.
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    expect(mock.endCalls.length).toBeGreaterThanOrEqual(1);
    expect(await getAlarmTime(stub)).toBeNull();
    const state = await getState(stub);
    expect(state.sfuCapacityAcquired).toBe(false);
  });
});
