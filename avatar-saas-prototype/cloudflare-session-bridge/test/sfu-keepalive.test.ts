import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env, runInDurableObject, runDurableObjectAlarm } from "cloudflare:test";
import { createWsToken } from "../src/ws-token";

const SESSION_PREFIX = "session:";
const TOKEN_SECRET = "test-token-secret";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}ka-${label}-${crypto.randomUUID().slice(0, 8)}`;
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

// ── Shared mock factory ─────────────────────────────────────────────────
// All tests share the same mock pattern as sfu-d1-repair.test.ts:
// mock fetch for Cloudflare SFU API, Modal start/end/status endpoints.

function createMockFetch(opts?: {
  startDelayMs?: number;
  statusDelayMs?: number;
  trackStatusCalls?: string[];
  sfuSessionId?: string;
  turnApiResponse?: "valid" | "empty" | "stun-only" | "error" | "no-credentials";
  captureStartBody?: { iceServersReceived?: unknown[] | null };
}) {
  const statusCalls: string[] = opts?.trackStatusCalls ?? [];
  const sfuSid = opts?.sfuSessionId ?? "mock-sfu-keepalive";
  const turnMode = opts?.turnApiResponse ?? "valid";

  const impl = async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);

    // Cloudflare TURN credentials API
    if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/turn/keys/") && url.pathname.includes("/credentials/generate-ice-servers")) {
      if (turnMode === "error") {
        return new Response(JSON.stringify({ error: "internal error" }), { status: 500 });
      }
      if (turnMode === "empty") {
        return new Response(JSON.stringify({ iceServers: [] }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }
      if (turnMode === "stun-only") {
        return new Response(JSON.stringify({
          iceServers: [{ urls: ["stun:stun.cloudflare.com:3478"] }],
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      if (turnMode === "no-credentials") {
        return new Response(JSON.stringify({
          iceServers: [{ urls: ["turn:turn.cloudflare.com:3478?transport=udp"] }],
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      // Default: valid TURN credentials
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
      // Capture ice_servers from request body for verification.
      if (opts?.captureStartBody) {
        try {
          const body = JSON.parse(init.body as string) as Record<string, unknown>;
          opts.captureStartBody.iceServersReceived = body.ice_servers ?? null;
        } catch { /* best-effort */ }
      }
      return new Response(JSON.stringify({
        session_id: `modal-ka-${sfuSid}`,
        avatar_id: "arnold",
        status: "ready",
        cache_hit: false,
        init_s: 0,
        active_sessions: [],
        state_note: "mock",
        vram_mb: 0,
        publisher_track_name: "avatar-video-ka",
        publisher_mode: "webrtc_sfu",
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    // Modal end-session
    if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/end") {
      return new Response(JSON.stringify({
        session_id: `modal-ka-${sfuSid}`,
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

    // Modal session-status — THIS is what the keepalive heartbeat calls
    if (init?.method === "POST" && url.hostname.includes("modal.run") && url.pathname === "/status") {
      if (opts?.statusDelayMs) {
        await new Promise(r => setTimeout(r, opts.statusDelayMs));
      }
      statusCalls.push(`modal-ka-${sfuSid}`);
      return new Response(JSON.stringify({
        session_id: `modal-ka-${sfuSid}`,
        active: true,
        active_sessions: [],
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }

    return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
  };

  return { impl, statusCalls };
}

// Helper: start an SFU session via REST (including publisher-offer callback)
async function startSfuSession(stub: DurableObjectStub, sid: string) {
  // Phase 3F: authorize the session for SFU before starting.
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

  // Simulate Modal's publisher-offer callback (lazy SFU session creation)
  const offerRes = await stub.fetch(internalRest(
    `https://test/internal/sessions/${sid}/publisher-offer`,
    {
      method: "POST",
      headers: { "X-SFU-Operation": "publisher_offer", "Content-Type": "application/json" },
      body: JSON.stringify({
        sdp: "v=0\r\no=- mock-pub\r\ns=-\r\nm=video 9 UDP/TLS/RTP/SAVPF 96\r\na=mid:0\r\n",
        type: "offer", mid: "0", track_name: "avatar-video-ka", kind: "video",
      }),
    },
  ));
  if (offerRes.status !== 200) throw new Error(`publisher-offer failed: ${offerRes.status}`);
}

// Helper: start an fMP4 session via REST (no SFU)
async function startFmp4Session(stub: DurableObjectStub, sid: string) {
  const startRes = await stub.fetch(internalRest(
    `https://test/v1/sessions/${sid}/start`,
    {
      method: "POST",
      headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
      body: JSON.stringify({ avatarId: "arnold", transport: "websocket_fmp4" }),
    },
  ));
  if (startRes.status !== 200) {
    const body = await startRes.json() as Record<string, unknown>;
    throw new Error(`fMP4 start failed: ${JSON.stringify(body)}`);
  }
}

// Helper: get alarm timestamp from DO storage
async function getAlarmTime(stub: DurableObjectStub): Promise<number | null> {
  return runInDurableObject(stub, async (_instance, state) => {
    return state.storage.getAlarm();
  });
}

// ──────────────────────────────────────────────────────────────────────────
// TEST 1: fMP4 session never sets an alarm
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — fMP4 sessions", () => {
  afterEach(() => vi.restoreAllMocks());

  it("fMP4 session does NOT set a keepalive alarm", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(createMockFetch().impl);
    const sid = uniqueSid("fmp4-no-ka");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startFmp4Session(stub, sid);

    const alarm = await getAlarmTime(stub);
    expect(alarm).toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 2: SFU session sets an alarm after ready
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — alarm scheduling", () => {
  afterEach(() => vi.restoreAllMocks());

  it("SFU session sets a keepalive alarm after session.ready", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(createMockFetch().impl);
    const sid = uniqueSid("sfu-alarm-set");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    const alarm = await getAlarmTime(stub);
    expect(alarm).not.toBeNull();
    expect(typeof alarm).toBe("number");
    // Alarm should be set for ~10s in the future
    const delta = alarm! - Date.now();
    expect(delta).toBeGreaterThan(5_000);
    expect(delta).toBeLessThan(15_000);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 3 & 4: Alarm handler calls Modal status and reschedules
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — alarm handler behavior", () => {
  afterEach(() => vi.restoreAllMocks());

  it("alarm handler calls Modal session-status and reschedules", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-alarm-fire");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Verify alarm is set before firing
    const alarmBefore = await getAlarmTime(stub);
    expect(alarmBefore).not.toBeNull();

    // Fire the alarm immediately
    const ran = await runDurableObjectAlarm(stub);
    expect(ran).toBe(true);

    // Modal status endpoint was called
    expect(mock.statusCalls.length).toBeGreaterThanOrEqual(1);

    // Alarm was rescheduled for the future
    const alarmAfter = await getAlarmTime(stub);
    expect(alarmAfter).not.toBeNull();
    expect(alarmAfter!).toBeGreaterThan(Date.now());
  });

  it("alarm handler keeps rescheduling while session is active", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-alarm-reschedule");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Fire alarm 3 times
    for (let i = 0; i < 3; i++) {
      await runDurableObjectAlarm(stub);
    }

    expect(mock.statusCalls.length).toBe(3);

    // Alarm is still set
    const alarm = await getAlarmTime(stub);
    expect(alarm).not.toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 5: Alarm handler self-terminates when state is no longer SFU/active
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — self-correction", () => {
  afterEach(() => vi.restoreAllMocks());

  it("alarm handler deletes itself when session is no longer active", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-self-stop");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // End the session (which should delete the alarm)
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    // Alarm should be deleted after end
    const alarmAfterEnd = await getAlarmTime(stub);
    expect(alarmAfterEnd).toBeNull();

    // Even if an alarm somehow fires (stale), it should self-terminate
    // Manually set an alarm, then fire it — it should delete itself
    await runInDurableObject(stub, async (_instance, state) => {
      await state.storage.setAlarm(Date.now() + 1000);
    });

    const ran = await runDurableObjectAlarm(stub);
    expect(ran).toBe(true);

    // No new alarm should be scheduled (session is ended)
    const alarmFinal = await getAlarmTime(stub);
    expect(alarmFinal).toBeNull();

    // No status calls should have been made after the session ended
    expect(mock.statusCalls.length).toBe(0);
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 6: REST end deletes the alarm
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — REST end", () => {
  afterEach(() => vi.restoreAllMocks());

  it("REST end deletes the keepalive alarm", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(createMockFetch().impl);
    const sid = uniqueSid("sfu-rest-end");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    const alarmBefore = await getAlarmTime(stub);
    expect(alarmBefore).not.toBeNull();

    // End session via REST
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    const alarmAfter = await getAlarmTime(stub);
    expect(alarmAfter).toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 7: End via REST deletes the alarm (WS session.end uses same #stopSfuKeepalive)
// ──────────────────────────────────────────────────────────────────────────
// NOTE: A full WebSocket test would require WS upgrade + Hibernation API
// setup which is not supported in the current test harness. Both paths
// (WS session.end → #handleSessionEnd → #stopSfuKeepalive("session_end")
//  REST end → #cleanupSfu → #stopSfuKeepalive("sfu_cleanup"))
// call the same #stopSfuKeepalive method. This test exercises that method
// via the REST path. The WS-specific integration is validated by the
// deployed canary (PostHog telemetry shows stop with reason=session_end).
describe("SFU keepalive — end session deletes alarm (REST path; WS path uses same stop)", () => {
  afterEach(() => vi.restoreAllMocks());

  it("REST end deletes the keepalive alarm (WS session.end calls same #stopSfuKeepalive)", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(createMockFetch().impl);
    const sid = uniqueSid("sfu-end");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    const alarmBefore = await getAlarmTime(stub);
    expect(alarmBefore).not.toBeNull();

    // End session via REST — exercises #cleanupSfu → #stopSfuKeepalive
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    const alarmAfter = await getAlarmTime(stub);
    expect(alarmAfter).toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 8: #cleanupSfu() early-return path still stops keepalive
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — cleanupSfu early-return", () => {
  afterEach(() => vi.restoreAllMocks());

  it("#cleanupSfu() stops keepalive even when sfuSessionId is already null", async () => {
    const mock = createMockFetch({ trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-cleanup-early");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Verify alarm is set
    const alarmBefore = await getAlarmTime(stub);
    expect(alarmBefore).not.toBeNull();

    // Manually null out sfuSessionId to simulate the early-return condition
    await runInDurableObject(stub, async (_instance, state) => {
      const stored = await state.storage.get("session:state") as Record<string, unknown>;
      stored.sfuSessionId = null;
      await state.storage.put("session:state", stored);
    });

    // Call REST end — #cleanupSfu() should still delete the alarm
    // even though sfuSessionId is null (early return path)
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    // Alarm must be deleted despite the early return in #cleanupSfu
    const alarmAfter = await getAlarmTime(stub);
    expect(alarmAfter).toBeNull();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TEST 9: In-flight teardown race — alarm does NOT reschedule when session
//         ends during an in-flight heartbeat fetch.
//
// This is the critical race-condition test:
//   1. Alarm fires, calls getModalSessionStatus() (2000ms delay)
//   2. During the fetch await, REST end arrives → #stopSfuKeepalive
//      deletes alarm, sets sfuKeepaliveActive=false
//   3. Alarm handler resumes, reloads state, sees session stopped
//   4. Alarm handler must NOT reschedule and must NOT overwrite stopped state
// ──────────────────────────────────────────────────────────────────────────
describe("SFU keepalive — in-flight teardown race", () => {
  afterEach(() => vi.restoreAllMocks());

  it("alarm does NOT reschedule when session ends during in-flight heartbeat", async () => {
    const mock = createMockFetch({ statusDelayMs: 2000, trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-race");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Verify alarm is set
    const alarmBefore = await getAlarmTime(stub);
    expect(alarmBefore).not.toBeNull();

    // Fire the alarm WITHOUT awaiting — the status fetch will delay 2000ms
    const alarmPromise = runDurableObjectAlarm(stub);

    // Wait 500ms for the alarm handler to reach the status fetch
    await new Promise(r => setTimeout(r, 500));

    // End the session WHILE the alarm's status fetch is in-flight.
    // The DO input gate opens during the await, allowing this request in.
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    // Wait for the alarm handler to complete (status fetch finishes,
    // state is reloaded, conditions checked)
    await alarmPromise;

    // PROOF: No alarm should be scheduled (the hardened alarm handler
    // detected the session was stopped and did NOT reschedule)
    const alarmAfter = await getAlarmTime(stub);
    expect(alarmAfter).toBeNull();

    // PROOF: Heartbeat was made (proving the race was real)
    expect(mock.statusCalls.length).toBeGreaterThanOrEqual(1);
  });

  it("alarm does NOT overwrite stopped state when session ends during heartbeat", async () => {
    const mock = createMockFetch({ statusDelayMs: 2000, trackStatusCalls: [] });
    vi.spyOn(globalThis, "fetch").mockImplementation(mock.impl);

    const sid = uniqueSid("sfu-race-state");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Fire alarm (status delays 2000ms)
    const alarmPromise = runDurableObjectAlarm(stub);

    // Wait 500ms, then end session during in-flight heartbeat
    await new Promise(r => setTimeout(r, 500));
    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/end`,
      { method: "POST", headers: { "X-REST-Operation": "end" } },
    ));

    await alarmPromise;

    // Verify state was NOT overwritten by stale alarm handler
    const state = await runInDurableObject(stub, async (_instance, s) => {
      return s.storage.get("session:state") as Record<string, unknown> | undefined;
    });
    expect(state).toBeDefined();
    // After end, state should reflect ended session
    expect(state!.active).toBe(false);
    expect(state!.sfuKeepaliveActive).toBeFalsy();
  });
});

// ──────────────────────────────────────────────────────────────────────────
// TURN credential verification tests
// ──────────────────────────────────────────────────────────────────────────

describe("SFU TURN credentials", () => {
  afterEach(() => vi.restoreAllMocks());

  it("passes ice_servers with TURN URLs to Modal start body", async () => {
    const capture: { iceServersReceived?: unknown[] | null } = {};
    vi.spyOn(globalThis, "fetch").mockImplementation(
      createMockFetch({ captureStartBody: capture }).impl,
    );
    const sid = uniqueSid("turn-pass");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await startSfuSession(stub, sid);

    // Verify ice_servers was forwarded to Modal with usable TURN URLs.
    expect(capture.iceServersReceived).toBeDefined();
    expect(Array.isArray(capture.iceServersReceived)).toBe(true);
    expect(capture.iceServersReceived!.length).toBeGreaterThan(0);

    // At least one entry must have username + credential + turn:/turns: URL.
    const servers = capture.iceServersReceived as Array<Record<string, unknown>>;
    const hasTurn = servers.some(
      (s) =>
        s.username &&
        s.credential &&
        Array.isArray(s.urls) &&
        (s.urls as string[]).some(
          (u) => typeof u === "string" && (u.startsWith("turn:") || u.startsWith("turns:")),
        ),
    );
    expect(hasTurn).toBe(true);
  });

  it("rejects with sfu_turn_credentials_unavailable when TURN API returns empty iceServers", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      createMockFetch({ turnApiResponse: "empty" }).impl,
    );
    const sid = uniqueSid("turn-empty");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "legacy-beta-test", auth_mode: "legacy", sfu_beta_authorized: true }),
      },
    ));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    expect(startRes.status).toBe(503);
    const body = await startRes.json() as Record<string, unknown>;
    expect((body.error as Record<string, string>)?.code).toBe("sfu_turn_credentials_unavailable");
  });

  it("rejects with sfu_turn_credentials_unavailable when TURN API returns STUN-only", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      createMockFetch({ turnApiResponse: "stun-only" }).impl,
    );
    const sid = uniqueSid("turn-stun-only");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "legacy-beta-test", auth_mode: "legacy", sfu_beta_authorized: true }),
      },
    ));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    expect(startRes.status).toBe(503);
    const body = await startRes.json() as Record<string, unknown>;
    expect((body.error as Record<string, string>)?.code).toBe("sfu_turn_credentials_unavailable");
  });

  it("rejects with sfu_turn_credentials_unavailable when TURN API returns no credentials", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      createMockFetch({ turnApiResponse: "no-credentials" }).impl,
    );
    const sid = uniqueSid("turn-no-cred");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "legacy-beta-test", auth_mode: "legacy", sfu_beta_authorized: true }),
      },
    ));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    expect(startRes.status).toBe(503);
    const body = await startRes.json() as Record<string, unknown>;
    expect((body.error as Record<string, string>)?.code).toBe("sfu_turn_credentials_unavailable");
  });

  it("rejects with sfu_turn_credentials_unavailable when TURN API returns HTTP error", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      createMockFetch({ turnApiResponse: "error" }).impl,
    );
    const sid = uniqueSid("turn-api-err");
    const stub = env.AVATAR_SESSION.get(env.AVATAR_SESSION.idFromName(sid));

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "legacy-beta-test", auth_mode: "legacy", sfu_beta_authorized: true }),
      },
    ));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    expect(startRes.status).toBe(503);
    const body = await startRes.json() as Record<string, unknown>;
    expect((body.error as Record<string, string>)?.code).toBe("sfu_turn_credentials_unavailable");
  });
});
