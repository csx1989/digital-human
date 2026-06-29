import { describe, it, expect, vi, afterEach, beforeEach } from "vitest";
import { env, runInDurableObject } from "cloudflare:test";
import {
  createSfuBetaToken,
  verifySfuBetaToken,
  parseSfuBetaTenants,
  resolveSfuBetaAuthorization,
  SFU_BETA_TOKEN_AUDIENCE,
} from "../src/sfu-beta-auth";
import { parseMetadataString, resolveApiKey, parseLegacyKeyString } from "../src/auth-core";
import { AvatarSessionDO } from "../src/avatar-session-do";

const SESSION_PREFIX = "session:";
const TOKEN_SECRET = "test-sfu-beta-secret-min-32-chars-long";

function uniqueSid(label: string): string {
  return `${SESSION_PREFIX}test-${label}-${crypto.randomUUID().slice(0, 8)}`;
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

// ── Unit: SFU beta token ───────────────────────────────────────────────────

describe("SfuBetaToken", () => {
  it("verifies a freshly created token", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, 60);
    expect(await verifySfuBetaToken(TOKEN_SECRET, token)).toBe(true);
  });

  it("rejects a tampered signature", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, 60);
    const tampered = token.slice(0, -4) + "XXXX";
    expect(await verifySfuBetaToken(TOKEN_SECRET, tampered)).toBe(false);
  });

  it("rejects an expired token", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, -1);
    expect(await verifySfuBetaToken(TOKEN_SECRET, token)).toBe(false);
  });

  it("rejects token signed with wrong secret", async () => {
    const token = await createSfuBetaToken("wrong-secret-min-32-chars-long", 60);
    expect(await verifySfuBetaToken(TOKEN_SECRET, token)).toBe(false);
  });

  it("rejects a token with wrong audience", async () => {
    const now = Math.floor(Date.now() / 1000);
    const payload = JSON.stringify({ aud: "wrong-aud", iat: now, exp: now + 60 });
    const encoder = new TextEncoder();
    const payloadB64 = btoa(String.fromCharCode(...new Uint8Array(encoder.encode(payload))))
      .replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    const key = await crypto.subtle.importKey(
      "raw",
      encoder.encode(TOKEN_SECRET),
      { name: "HMAC", hash: "SHA-256" },
      false,
      ["sign"],
    );
    const sig = new Uint8Array(await crypto.subtle.sign("HMAC", key, encoder.encode(payloadB64)));
    const sigB64 = btoa(String.fromCharCode(...sig)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
    expect(await verifySfuBetaToken(TOKEN_SECRET, `${payloadB64}.${sigB64}`)).toBe(false);
  });

  it("returns false when secret is missing", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, 60);
    expect(await verifySfuBetaToken(undefined, token)).toBe(false);
  });
});

// ── Unit: tenant allowlist ─────────────────────────────────────────────────

describe("parseSfuBetaTenants", () => {
  it("parses comma-separated tenant ids", () => {
    const set = parseSfuBetaTenants(" tenant-a , tenant-b ,tenant-c ");
    expect([...set]).toEqual(["tenant-a", "tenant-b", "tenant-c"]);
  });

  it("returns empty set for empty/missing input", () => {
    expect([...parseSfuBetaTenants("")]).toEqual([]);
    expect([...parseSfuBetaTenants(undefined)]).toEqual([]);
  });
});

// ── Unit: authorization resolver ───────────────────────────────────────────

describe("resolveSfuBetaAuthorization", () => {
  const envBase = {
    AVTR_SFU_BETA_TOKEN_SECRET: TOKEN_SECRET,
    AVTR_SFU_BETA_TENANTS: "beta-tenant",
  };

  it("authorizes via metadata flag", async () => {
    const result = await resolveSfuBetaAuthorization(envBase, {
      tenantId: "any-tenant",
      metadataSfuBeta: true,
    });
    expect(result).toEqual({ authorized: true, source: "metadata" });
  });

  it("authorizes via tenant allowlist", async () => {
    const result = await resolveSfuBetaAuthorization(envBase, {
      tenantId: "beta-tenant",
    });
    expect(result).toEqual({ authorized: true, source: "tenant" });
  });

  it("authorizes via valid token", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, 60);
    const result = await resolveSfuBetaAuthorization(envBase, { betaToken: token });
    expect(result).toEqual({ authorized: true, source: "token" });
  });

  it("rejects invalid token", async () => {
    const result = await resolveSfuBetaAuthorization(envBase, {
      betaToken: "invalid.token.here",
    });
    expect(result).toEqual({ authorized: false, source: "none" });
  });

  it("rejects when no source matches", async () => {
    const result = await resolveSfuBetaAuthorization(envBase, {
      tenantId: "unknown-tenant",
    });
    expect(result).toEqual({ authorized: false, source: "none" });
  });

  it("metadata flag takes precedence over tenant/token", async () => {
    const token = await createSfuBetaToken(TOKEN_SECRET, 60);
    const result = await resolveSfuBetaAuthorization(envBase, {
      tenantId: "beta-tenant",
      metadataSfuBeta: true,
      betaToken: token,
    });
    expect(result.source).toBe("metadata");
  });
});

// ── Unit: metadata parsing ─────────────────────────────────────────────────

describe("MetadataEntry sfu_beta parsing", () => {
  it("accepts sfu_beta true", () => {
    const raw = JSON.stringify({ key1: { tenant_id: "t1", sfu_beta: true } });
    const result = parseMetadataString(raw);
    expect(result).not.toBeNull();
    expect(result!.entries[0].sfu_beta).toBe(true);
  });

  it("defaults sfu_beta to false", () => {
    const raw = JSON.stringify({ key1: { tenant_id: "t1" } });
    const result = parseMetadataString(raw);
    expect(result!.entries[0].sfu_beta).toBe(false);
  });

  it("drops entries with non-boolean sfu_beta", () => {
    const raw = JSON.stringify({ key1: { tenant_id: "t1", sfu_beta: "yes" } });
    const result = parseMetadataString(raw);
    expect(result!.entries[0].sfu_beta).toBe(false);
  });

  it("resolveApiKey returns sfu_beta from metadata", () => {
    const raw = JSON.stringify({ key1: { tenant_id: "t1", sfu_beta: true } });
    const { entries } = parseMetadataString(raw)!;
    const result = resolveApiKey("key1", [], entries);
    expect(result?.sfu_beta).toBe(true);
  });
});

// ── DO integration: beta gate ──────────────────────────────────────────────

describe("AvatarSessionDO SFU beta gate", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects webrtc_sfu for unauthorized session", async () => {
    const sid = uniqueSid("sfu-deny");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    // Init owner without beta authorization.
    const initRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "unauthorized-tenant", auth_mode: "metadata" }),
      },
    ));
    expect(initRes.ok).toBe(true);

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold", transport: "webrtc_sfu" }),
      },
    ));
    expect(startRes.status).toBe(403);
    const body = await startRes.json() as Record<string, unknown>;
    expect((body.error as Record<string, string>)?.code).toBe("sfu_beta_not_enabled");
  });

  it("allows webrtc_sfu for beta-authorized session when SFU is ready", async () => {
    const sid = uniqueSid("sfu-allow");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

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

      // Cloudflare Realtime SFU session creation
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.endsWith("/sessions/new")) {
        return new Response(JSON.stringify({ sessionId: "mock-sfu-session" }), {
          status: 200, headers: { "Content-Type": "application/json" },
        });
      }

      // SFU tracks/new (publisher + subscriber)
      if (url.hostname === "rtc.live.cloudflare.com" && url.pathname.includes("/tracks/new")) {
        return new Response(JSON.stringify({
          sessionDescription: { sdp: "v=0\r\no=- mock\r\ns=-\r\n", type: "answer" },
          tracks: [{ mid: "0" }],
          requiresImmediateRenegotiation: false,
        }), { status: 200, headers: { "Content-Type": "application/json" } });
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
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }

      return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
    });

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant_id: "beta-tenant",
          auth_mode: "metadata",
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
    expect(startRes.status).toBe(200);
    const body = await startRes.json() as Record<string, unknown>;
    expect(body.transport).toBe("webrtc_sfu");
  });

  it("keeps default fMP4 behavior for unauthorized sessions", async () => {
    const sid = uniqueSid("fmp4-default");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? new URL(input) : input instanceof URL ? input : new URL(input.url);
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
          publisher_mode: "websocket_fmp4",
        }), { status: 200, headers: { "Content-Type": "application/json" } });
      }
      return new Response(JSON.stringify({ error: `unexpected fetch: ${url}` }), { status: 500 });
    });

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({ tenant_id: "unauthorized-tenant", auth_mode: "metadata" }),
      },
    ));

    const startRes = await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/start`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "start", "Content-Type": "application/json" },
        body: JSON.stringify({ avatarId: "arnold" }),
      },
    ));
    expect(startRes.status).toBe(200);
    const body = await startRes.json() as Record<string, unknown>;
    expect(body.transport).toBe("websocket_fmp4");
  });

  it("does not persist raw beta token in DO state or owner", async () => {
    const sid = uniqueSid("no-token-leak");
    const id = env.AVATAR_SESSION.idFromName(sid);
    const stub = env.AVATAR_SESSION.get(id);

    const token = await createSfuBetaToken(TOKEN_SECRET, 60);

    await stub.fetch(internalRest(
      `https://test/v1/sessions/${sid}/init_owner`,
      {
        method: "POST",
        headers: { "X-REST-Operation": "init_owner", "Content-Type": "application/json" },
        body: JSON.stringify({
          tenant_id: "token-tenant",
          auth_mode: "legacy",
          sfu_beta_authorized: true,
        }),
      },
    ));

    await runInDurableObject(stub, async (instance: AvatarSessionDO, state) => {
      const entries = await state.storage.list();
      const rawStorage = JSON.stringify([...entries.entries()]);
      expect(rawStorage).not.toContain(token);
    });
  });
});
