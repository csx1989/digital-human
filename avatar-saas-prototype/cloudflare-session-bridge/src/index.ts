import { PostHog } from "posthog-node";
import type { Env } from "./types";
import { AvatarSessionDO } from "./avatar-session-do";
import { SfuCapacityManager } from "./sfu-capacity-manager";
import { captureEvent } from "./posthog-client";
import { handleV1Api, CORS_HEADERS, corsJson, checkSessionOwnership, initSessionOwner } from "./api-v1";
import { verifyWsToken } from "./ws-token";
import { checkRateLimit, rateLimitedResponse, type RateCategory } from "./rate-limit";
import { safeEqual } from "./auth-core";
import { verifySfuBetaToken, createSfuBetaToken } from "./sfu-beta-auth";
import type { ApiAuthContext } from "./types";

export { AvatarSessionDO };
export { SfuCapacityManager };

const SESSION_PREFIX = "session:";
const DEFAULT_POSTHOG_HOST = "https://eu.i.posthog.com";

function extractOrGenerateRequestId(request: Request): string {
  const provided = request.headers.get("X-Request-ID") ?? request.headers.get("X-Request-Id");
  if (provided && /^[a-zA-Z0-9_-]{1,128}$/.test(provided)) {
    return provided;
  }
  return crypto.randomUUID();
}

function withRequestId(response: Response, requestId: string): Response {
  const headers = new Headers(response.headers);
  headers.set("X-Request-Id", requestId);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

function corsResponse(body: unknown, init?: ResponseInit): Response {
  const headers = new Headers(init?.headers);
  for (const [k, v] of Object.entries(CORS_HEADERS)) {
    headers.set(k, v);
  }
  return new Response(JSON.stringify(body), {
    ...init,
    headers,
  });
}

async function handlePosthogDebug(request: Request, env: Env): Promise<Response> {
  const debugToken = request.headers.get("x-debug-token");
  if (!env.DEBUG_POSTHOG_TOKEN || debugToken !== env.DEBUG_POSTHOG_TOKEN) {
    return corsResponse({ error: "not found" }, { status: 404 });
  }

  const apiKey = env.POSTHOG_API_KEY;
  const host = env.POSTHOG_HOST || DEFAULT_POSTHOG_HOST;

  const result: Record<string, unknown> = {
    keyPresent: !!apiKey,
    keyPrefix: apiKey ? apiKey.substring(0, 4) : null,
    host,
    sdkCapture: null as Record<string, unknown> | null,
    rawFetch: null as Record<string, unknown> | null,
  };

  if (apiKey) {
    // SDK test
    try {
      const client = new PostHog(apiKey, {
        host,
        flushAt: 1,
        flushInterval: 0,
      });
      await client.captureImmediate({
        distinctId: "debug-posthog-test",
        event: "debug_posthog_sdk_test",
        properties: { source: "worker-debug-sdk", timestamp: new Date().toISOString() },
      });
      await client.shutdown();
      result.sdkCapture = { status: "ok" };
    } catch (err) {
      result.sdkCapture = {
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        errorType: err instanceof Error ? err.constructor.name : typeof err,
      };
    }

    // Raw fetch test
    try {
      const rawResponse = await fetch(`${host}/i/v0/e/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          api_key: apiKey,
          event: "debug_posthog_raw_test",
          distinct_id: "debug-posthog-test",
          properties: { source: "worker-debug-raw", timestamp: new Date().toISOString() },
        }),
      });
      const rawBody = await rawResponse.text();
      result.rawFetch = {
        status: rawResponse.status,
        statusText: rawResponse.statusText,
        bodyPreview: rawBody.substring(0, 200),
      };
    } catch (err) {
      result.rawFetch = {
        status: "fetch_error",
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  return corsResponse(result);
}

export default {
  async fetch(
    request: Request,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method === "GET" && path === "/health") {
      return corsResponse({ status: "ok" });
    }

    if (request.method === "POST" && path === "/debug/posthog-test") {
      return handlePosthogDebug(request, env);
    }

    if (request.method === "POST" && path === "/debug/gen-beta-token") {
      const debugToken = request.headers.get("x-debug-token");
      if (!env.DEBUG_POSTHOG_TOKEN || debugToken !== env.DEBUG_POSTHOG_TOKEN) {
        return corsResponse({ error: "not found" }, { status: 404 });
      }
      if (!env.AVTR_SFU_BETA_TOKEN_SECRET) {
        return corsResponse({ error: "AVTR_SFU_BETA_TOKEN_SECRET not configured" }, { status: 503 });
      }
      const ttl = 7200;
      const token = await createSfuBetaToken(env.AVTR_SFU_BETA_TOKEN_SECRET, ttl);
      return corsResponse({ beta_token: token, ttl_seconds: ttl });
    }

    if (request.method === "POST" && path === "/debug/sfu-capacity-clear") {
      const debugToken = request.headers.get("x-debug-token");
      if (!env.DEBUG_POSTHOG_TOKEN || debugToken !== env.DEBUG_POSTHOG_TOKEN) {
        return corsResponse({ error: "not found" }, { status: 404 });
      }
      const mgrId = env.SFU_CAPACITY_MANAGER.idFromName("global");
      const stub = env.SFU_CAPACITY_MANAGER.get(mgrId);
      const resp = await stub.fetch(new Request("https://internal/sfu-capacity?op=clear", { method: "POST" }));
      const data = await resp.json();
      return corsResponse(data);
    }

    // ── Stage 3B: admin force-clear of a runaway session DO ─────────────
    // Protected by a dedicated secret (NOT DEBUG_POSTHOG_TOKEN). Deletes the
    // DO alarm, stops keepalive, clears state, and releases capacity even if
    // Modal is unreachable or billing-limited. Used for incident response.
    const forceClearMatch = path.match(/^\/debug\/sessions\/([^/]+)\/force-clear$/);
    if (request.method === "POST" && forceClearMatch) {
      const adminToken = request.headers.get("x-admin-token");
      if (!env.AVTR_ADMIN_FORCE_CLEAR_TOKEN || adminToken !== env.AVTR_ADMIN_FORCE_CLEAR_TOKEN) {
        return corsResponse({ error: "not found" }, { status: 404 });
      }
      const sessionId = decodeURIComponent(forceClearMatch[1]);
      if (!sessionId.startsWith(SESSION_PREFIX)) {
        return corsResponse({ error: "invalid session id" }, { status: 400 });
      }
      const doId = env.AVATAR_SESSION.idFromName(sessionId);
      const stub = env.AVATAR_SESSION.get(doId);
      const internalReq = new Request(
        `https://internal/v1/sessions/${encodeURIComponent(sessionId)}/force-clear`,
        {
          method: "POST",
          headers: {
            "X-Internal-REST": "true",
            "X-REST-Operation": "force_clear",
          },
        },
      );
      const resp = await stub.fetch(internalReq);
      const data = await resp.json();
      return corsResponse(data, { status: resp.status });
    }

    if (request.method === "POST" && path === "/sessions") {
      const sessionId = `${SESSION_PREFIX}${crypto.randomUUID()}`;

      // Phase 3F: legacy /sessions SFU beta gate. A short-lived HMAC-signed
      // beta token may be passed via query param for manual/private testing.
      // The raw token is verified server-side and never stored or logged.
      const betaToken = url.searchParams.get("sfu_beta_token")?.trim() ?? undefined;
      const betaAuthorized = betaToken
        ? await verifySfuBetaToken(env.AVTR_SFU_BETA_TOKEN_SECRET, betaToken)
        : false;

      if (betaToken && !betaAuthorized) {
        return corsResponse(
          { error: { code: "sfu_beta_not_enabled", message: "Invalid or expired SFU beta token." } },
          { status: 403 },
        );
      }

      // Eagerly initialize the DO owner with the beta authorization flag so
      // the browser does not need to re-send the token on the WebSocket URL.
      if (betaAuthorized) {
        const legacyAuthCtx: ApiAuthContext = {
          tenant_id: "legacy-browser-beta",
          key_name: null,
          plan: null,
          auth_mode: "legacy",
        };
        try {
          await initSessionOwner(env, sessionId, legacyAuthCtx, true);
        } catch {
          return corsResponse(
            { error: { code: "internal_error", message: "Failed to initialize session." } },
            { status: 500 },
          );
        }
      }

      const wsProtocol = url.protocol === "https:" ? "wss:" : "ws:";
      const wsUrl = `${wsProtocol}//${url.host}/sessions/${sessionId}/ws`;

      captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "avatar_session_created", {
        avatarSessionId: sessionId,
        sfuBetaAuthorized: betaAuthorized,
        sfuBetaTokenPresent: !!betaToken,
        sfuBetaSource: betaAuthorized ? "token" : "none",
        source: "legacy",
      });

      return corsResponse({ sessionId, status: "created", wsUrl });
    }

    if (
      request.method === "GET" &&
      path.startsWith("/sessions/") &&
      path.endsWith("/ws")
    ) {
      const segments = path.split("/");
      const sessionId = decodeURIComponent(segments[2]);

      if (!sessionId || !sessionId.startsWith(SESSION_PREFIX)) {
        return corsResponse(
          { error: "invalid session ID format" },
          { status: 400 },
        );
      }

      const doId = env.AVATAR_SESSION.idFromName(sessionId);
      const stub = env.AVATAR_SESSION.get(doId);
      return stub.fetch(request);
    }

    // ── v1 authenticated WebSocket upgrade ──────────────────────────────
    // GET /v1/sessions/:id/ws?token=...
    // Token is mandatory. Validated at the Worker before forwarding to DO.
    if (request.method === "GET") {
      const v1WsMatch = path.match(/^\/v1\/sessions\/([^/]+)\/ws$/);
      if (v1WsMatch) {
        const t0 = Date.now();
        const sessionId = decodeURIComponent(v1WsMatch[1]);
        const wsRequestId = extractOrGenerateRequestId(request);

        if (!sessionId.startsWith(SESSION_PREFIX)) {
          return withRequestId(corsJson(
            { success: false, error: { code: "invalid_session_id", message: "Session ID must start with 'session:'" }, timings: { latency_ms: Date.now() - t0, request_id: wsRequestId } },
            { status: 400 },
          ), wsRequestId);
        }

        if (!env.AVTR_SESSION_TOKEN_SECRET) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_token_rejected", {
            reason: "secret_not_configured",
            requestId: wsRequestId,
          });
          return withRequestId(corsJson(
            { success: false, error: { code: "ws_token_secret_not_configured", message: "WebSocket token secret not configured." }, timings: { latency_ms: Date.now() - t0, request_id: wsRequestId } },
            { status: 503 },
          ), wsRequestId);
        }

        const token = url.searchParams.get("token");
        if (!token) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_token_rejected", {
            reason: "missing",
            requestId: wsRequestId,
          });
          return withRequestId(corsJson(
            { success: false, error: { code: "ws_token_missing", message: "WebSocket session token required." }, timings: { latency_ms: Date.now() - t0, request_id: wsRequestId } },
            { status: 401 },
          ), wsRequestId);
        }

        const result = await verifyWsToken(env.AVTR_SESSION_TOKEN_SECRET, token, sessionId);
        if (!result.valid) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_token_rejected", {
            reason: result.reason,
            requestId: wsRequestId,
          });
          return withRequestId(corsJson(
            { success: false, error: { code: "ws_token_invalid", message: "Invalid or expired token." }, timings: { latency_ms: Date.now() - t0, request_id: wsRequestId } },
            { status: 401 },
          ), wsRequestId);
        }

        captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_auth_passed", {
          tenantId: result.tenantId,
          authMode: "token",
          requestId: wsRequestId,
        });

        const ownership = await checkSessionOwnership(env, sessionId, result.tenantId!);
        if (!ownership.allowed) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ownership_check_failed", {
            reason: ownership.reason,
            callerTenantId: result.tenantId,
            route: "/v1/sessions/:id/ws",
            requestId: wsRequestId,
          });
          return withRequestId(corsJson(
            { success: false, error: { code: "not_found", message: "Session not found." }, timings: { latency_ms: Date.now() - t0, request_id: wsRequestId } },
            { status: 404 },
          ), wsRequestId);
        }

        captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ownership_check_passed", {
          callerTenantId: result.tenantId,
          authMode: "token",
          route: "/v1/sessions/:id/ws",
          requestId: wsRequestId,
        });

        const rlResult = await checkRateLimit(env, result.tenantId!, "ws_connect");
        if (!rlResult.allowed) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "api_rate_limited", {
            source: "v1_api",
            tenantId: result.tenantId,
            authMode: "token",
            category: "ws_connect",
            route: "/v1/sessions/:id/ws",
            limit: rlResult.limit ?? null,
            period: rlResult.period ?? null,
            requestId: wsRequestId,
          });
          return withRequestId(
            rateLimitedResponse("ws_connect", rlResult.limit!, rlResult.period!, t0, wsRequestId),
            wsRequestId,
          );
        }
        if (rlResult.bindingMissing) {
          captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "api_rate_limit_binding_missing", {
            source: "v1_api",
            tenantId: result.tenantId,
            authMode: "token",
            category: "ws_connect",
            bindingName: rlResult.bindingName ?? null,
            route: "/v1/sessions/:id/ws",
            requestId: wsRequestId,
          });
        }

        const doId = env.AVATAR_SESSION.idFromName(sessionId);
        const stub = env.AVATAR_SESSION.get(doId);
        return stub.fetch(request);
      }
    }

    // ── Phase D1: Internal Modal callback routes ────────────────────────
    // These routes use x-modal-secret auth, NOT API key.
    // Must be before /v1/* to bypass public API auth.
    if (path.startsWith("/internal/")) {
      return await handleInternalRoute(request, env, ctx, url, path, method);
    }

    // ── Phase D1: SFU browser routes (sfu_token auth, NOT API key) ──────
    // Same validation pattern as the WS upgrade above.
    // Must be before /v1/* to bypass public API auth.
    if (path.includes("/sfu/") && method !== "OPTIONS") {
      return await handleSfuBrowserRoute(request, env, ctx, url, path, method);
    }

    // ── Phase 5B+/v1: stable public API ────────────────────────────────
    // All /v1/* routing, auth, telemetry, and response normalization lives
    // in src/api-v1.ts. The DO stays an internal coordinator; its raw REST
    // shapes are normalized into the canonical {success,data,timings}
    // envelope at this boundary.
    if (path.startsWith("/v1/")) {
      return handleV1Api(request, env, ctx);
    }

    return corsResponse({ error: "not found" }, { status: 404 });
  },
} satisfies ExportedHandler<Env>;

// ── Phase D1: SFU browser route handler ────────────────────────────────────
//
// Authenticates with sfu_token (purpose:"sfu"), checks session ownership,
// then forwards to the DO for SFU API proxy calls.
// Browser never sees CALLS_APP_SECRET or raw SFU session IDs.

async function handleSfuBrowserRoute(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  url: URL,
  path: string,
  method: string,
): Promise<Response> {
  const t0 = Date.now();
  const requestId = extractOrGenerateRequestId(request);

  // Match: /v1/sessions/:sessionId/sfu/(subscribe|answer|close)
  const match = path.match(/^\/v1\/sessions\/([^/]+)\/sfu\/(subscribe|answer|close)$/);
  if (!match) {
    return corsJson({ success: false, error: { code: "not_found", message: "Unknown SFU route" }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 404 });
  }

  const sessionId = decodeURIComponent(match[1]);
  const operation = match[2] as "subscribe" | "answer" | "close";

  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "sfu_subscribe_route_entered", {
    operation,
    requestId,
  });

  if (!sessionId.startsWith(SESSION_PREFIX)) {
    return corsJson({ success: false, error: { code: "invalid_session_id", message: "Session ID must start with 'session:'" }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 400 });
  }

  // Validate sfu_token
  if (!env.AVTR_SESSION_TOKEN_SECRET) {
    return corsJson({ success: false, error: { code: "ws_token_secret_not_configured", message: "Token secret not configured." }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 503 });
  }

  // Token can be in ?token= query param or Authorization: Bearer header
  const token = url.searchParams.get("token") ?? (() => {
    const auth = request.headers.get("Authorization");
    if (auth?.startsWith("Bearer ")) return auth.slice(7).trim();
    return null;
  })();

  if (!token) {
    return corsJson({ success: false, error: { code: "ws_token_missing", message: "SFU session token required." }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 401 });
  }

  const result = await verifyWsToken(env.AVTR_SESSION_TOKEN_SECRET, token, sessionId, ["sfu"]);
  if (!result.valid) {
    captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "sfu_token_rejected", {
      reason: result.reason,
      requestId,
    });
    return corsJson({ success: false, error: { code: "ws_token_invalid", message: "Invalid or expired SFU token." }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 401 });
  }

  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "sfu_subscribe_auth_ok", {
    operation,
    tenantId: result.tenantId,
    requestId,
  });

  // Ownership check
  const ownership = await checkSessionOwnership(env, sessionId, result.tenantId!);
  if (!ownership.allowed) {
    captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "sfu_subscribe_ownership_failed", {
      operation,
      reason: ownership.reason,
      tokenTenantId: result.tenantId,
      requestId,
    });
    return corsJson({ success: false, error: { code: "not_found", message: "Session not found." }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 404 });
  }

  // Rate limit — per-operation category
  // subscribe: strict stream_connect (5/60s)
  // answer: light read (120/60s)
  // close: cleanup (exempt — never blocked)
  const sfuRateCategory: Record<typeof operation, RateCategory> = {
    subscribe: "stream_connect",
    answer: "read",
    close: "cleanup",
  };
  const rlResult = await checkRateLimit(env, result.tenantId!, sfuRateCategory[operation]);
  if (!rlResult.allowed) {
    return corsJson({ success: false, error: { code: "rate_limited", message: "Rate limit exceeded." }, timings: { latency_ms: Date.now() - t0, request_id: requestId } }, { status: 429 });
  }

  // Forward to DO with internal REST header
  const doId = env.AVATAR_SESSION.idFromName(sessionId);
  const stub = env.AVATAR_SESSION.get(doId);

  const internalRequest = new Request(request.url, {
    method: request.method,
    headers: new Headers(request.headers),
    body: request.body,
  });
  internalRequest.headers.set("X-Internal-REST", "true");
  internalRequest.headers.set("X-SFU-Operation", `sfu_${operation}`);
  internalRequest.headers.set("X-SFU-Tenant-Id", result.tenantId!);

  return stub.fetch(internalRequest);
}

// ── Phase D1: Internal Modal callback route handler ────────────────────────
//
// Authenticates with x-modal-secret header. No API key needed.
// Modal calls these endpoints to exchange SDP offers/answers and close tracks.

async function handleInternalRoute(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  url: URL,
  path: string,
  method: string,
): Promise<Response> {
  const requestId = extractOrGenerateRequestId(request);

  // Match: /internal/sessions/:sessionId/(publisher-offer|publisher-close)
  const match = path.match(/^\/internal\/sessions\/([^/]+)\/(publisher-offer|publisher-close)$/);
  if (!match) {
    return corsJson({ success: false, error: { code: "not_found", message: "Unknown internal route" }, timings: { latency_ms: 0, request_id: requestId } }, { status: 404 });
  }

  const sessionId = decodeURIComponent(match[1]);
  const operation = match[2] as "publisher-offer" | "publisher-close";

  if (!sessionId.startsWith(SESSION_PREFIX)) {
    return corsJson({ success: false, error: { code: "invalid_session_id", message: "Session ID must start with 'session:'" }, timings: { latency_ms: 0, request_id: requestId } }, { status: 400 });
  }

  // Validate x-modal-secret
  if (!env.MODAL_WORKER_SECRET) {
    return corsJson({ success: false, error: { code: "internal_secret_invalid", message: "MODAL_WORKER_SECRET not configured" }, timings: { latency_ms: 0, request_id: requestId } }, { status: 503 });
  }

  const modalSecret = request.headers.get("x-modal-secret");
  if (!modalSecret || !env.MODAL_WORKER_SECRET || !safeEqual(modalSecret, env.MODAL_WORKER_SECRET)) {
    captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "internal_auth_rejected", {
      route: path,
      reason: modalSecret ? "wrong_secret" : "missing_secret",
      requestId,
    });
    return corsJson({ success: false, error: { code: "internal_secret_invalid", message: "Invalid internal secret" }, timings: { latency_ms: 0, request_id: requestId } }, { status: 401 });
  }

  // Forward to DO with internal REST header
  const doId = env.AVATAR_SESSION.idFromName(sessionId);
  const stub = env.AVATAR_SESSION.get(doId);

  const internalRequest = new Request(request.url, {
    method: request.method,
    headers: new Headers(request.headers),
    body: request.body,
  });
  internalRequest.headers.set("X-Internal-REST", "true");
  internalRequest.headers.set("X-SFU-Operation", operation === "publisher-offer" ? "publisher_offer" : "publisher_close");

  return stub.fetch(internalRequest);
}
