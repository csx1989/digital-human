import type {
  Env,
  ApiAuthContext,
  ApiSuccess,
  ApiFailure,
  V1Session,
  V1SessionStatus,
  V1Health,
  V1Stream,
  V1SessionEndSummary,
  V1Providers,
  V1Me,
} from "./types";
import { captureEvent } from "./posthog-client";
import { getCatalogProvider, validateAvatarBgForStart } from "./catalog-provider";
import {
  parseLegacyKeyString,
  parseMetadataString,
  resolveApiKey,
  type MetadataEntry,
} from "./auth-core";
import { createWsToken } from "./ws-token";
import type { V1StreamSfu } from "./types";
import { isSfuFullyConfigured } from "./transport-decision";
import type { TransportPreference } from "./transport-decision";
import { resolveSfuBetaAuthorization } from "./sfu-beta-auth";
import {
  categorizeRoute,
  checkRateLimit,
  rateLimitedResponse,
  type RateCategory,
} from "./rate-limit";

const SESSION_PREFIX = "session:";
const API_VERSION = "v1";
const API_TRANSPORT = "websocket_fmp4";

const API_DISTINCT_ID_DEFAULT = "api-public";

// ── CORS (single source; imported by index.ts for OPTIONS preflight) ───────
export const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, x-debug-token, Authorization, x-api-key, X-Request-Id",
  "Access-Control-Expose-Headers": "X-Request-Id",
};

// ── Static catalogs moved to catalog-provider.ts (CatalogProvider seam) ───

// ── Response builders ──────────────────────────────────────────────────────

export function corsJson(body: unknown, init?: ResponseInit): Response {
  const headers = new Headers(init?.headers);
  for (const [k, v] of Object.entries(CORS_HEADERS)) headers.set(k, v);
  headers.set("Content-Type", "application/json");
  return new Response(JSON.stringify(body), { ...init, headers });
}

function apiOk<T>(data: T, t0: number, requestId?: string): Response {
  const body: ApiSuccess<T> = {
    success: true,
    data,
    timings: { latency_ms: Date.now() - t0, ...(requestId ? { request_id: requestId } : {}) },
  };
  return corsJson(body, { status: 200 });
}

function apiFail(code: string, message: string, t0: number, status: number, requestId?: string): Response {
  const body: ApiFailure = {
    success: false,
    error: { code, message },
    timings: { latency_ms: Date.now() - t0, ...(requestId ? { request_id: requestId } : {}) },
  };
  return corsJson(body, { status });
}

// ── Auth ───────────────────────────────────────────────────────────────────

function extractApiKey(request: Request): string | null {
  const authHeader = request.headers.get("Authorization");
  if (authHeader) {
    const prefix = "Bearer ";
    if (authHeader.startsWith(prefix)) {
      const key = authHeader.slice(prefix.length).trim();
      if (key) return key;
    }
  }
  const apiKeyHeader = request.headers.get("x-api-key");
  if (apiKeyHeader?.trim()) return apiKeyHeader.trim();
  return null;
}

// ── Legacy key cache (keyed by raw source string) ──────────────────────────

let _legacySource: string | null = null;
let _legacyKeys: string[] = [];

function parseLegacyKeys(env: Env): string[] {
  const raw = env.AVTR_API_KEYS?.trim() ?? "";
  if (raw === _legacySource) return _legacyKeys;
  _legacySource = raw;
  _legacyKeys = parseLegacyKeyString(raw);
  return _legacyKeys;
}

// ── Metadata cache (keyed by raw source string) ────────────────────────────

let _metaSource: string | null = null;
let _metaParsed: MetadataEntry[] | null = null;

function parseKeyMetadata(env: Env): MetadataEntry[] | null {
  const raw = env.AVTR_API_KEY_METADATA_JSON?.trim() ?? "";
  if (!raw) return null;
  if (raw === _metaSource) return _metaParsed;
  _metaSource = raw;

  const result = parseMetadataString(raw);
  if (result === null) {
    console.warn("[auth] AVTR_API_KEY_METADATA_JSON is invalid JSON — ignoring");
    _metaParsed = null;
    return null;
  }

  if (result.dropped > 0) {
    console.warn(`[auth] AVTR_API_KEY_METADATA_JSON: dropped ${result.dropped} invalid entr(y|ies)`);
  }

  _metaParsed = result.entries.length > 0 ? result.entries : null;
  return _metaParsed;
}

// ── authenticate() — returns Response on rejection, ApiAuthContext on success.
// Delegates to resolveApiKey() in auth-core.ts for key matching.
// Caching wrappers (parseLegacyKeys / parseKeyMetadata) stay here.

function authenticate(request: Request, env: Env, t0: number, requestId?: string): Response | ApiAuthContext {
  const legacyKeys = parseLegacyKeys(env);
  const metadataEntries = parseKeyMetadata(env);

  if (legacyKeys.length === 0 && !metadataEntries) {
    return apiFail(
      "api_auth_not_configured",
      "API authentication is not configured on this server.",
      t0,
      503,
      requestId,
    );
  }

  const requestKey = extractApiKey(request);
  if (!requestKey) {
    return apiFail(
      "missing_api_key",
      "API key required. Provide Authorization: Bearer <key> or x-api-key header.",
      t0,
      401,
      requestId,
    );
  }

  const result = resolveApiKey(requestKey, legacyKeys, metadataEntries);
  if (result) {
    return {
      tenant_id: result.tenant_id,
      key_name: result.key_name,
      plan: result.plan,
      auth_mode: result.auth_mode,
      sfu_beta: result.sfu_beta === true,
    };
  }

  return apiFail(
    "invalid_api_key",
    "The provided API key is not valid.",
    t0,
    403,
    requestId,
  );
}

// ── Telemetry helpers ──────────────────────────────────────────────────────

function routeForTelemetry(path: string): string {
  const opMatch = path.match(/^\/v1\/sessions\/[^/]+\/(start|end|status|stream)$/);
  if (opMatch) return `/v1/sessions/:id/${opMatch[1]}`;
  if (/^\/v1\/sessions\/[^/]+\/ws-token$/.test(path)) return "/v1/sessions/:id/ws-token";
  if (/^\/v1\/sessions\/[^/]+$/.test(path)) return "/v1/sessions/:id";
  return path;
}

function sessionIdFromPath(path: string): string | null {
  const m = path.match(/^\/v1\/sessions\/([^/]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

// Reads error.code from a failure Response body without consuming it.
// Used so every failure (inline + DO-routed) reports errorCode in telemetry.
async function extractErrorCode(response: Response): Promise<string | null> {
  if (response.status < 400) return null;
  try {
    const clone = response.clone();
    const text = await clone.text();
    if (!text) return null;
    const parsed = JSON.parse(text) as { error?: { code?: string } };
    return parsed?.error?.code ?? null;
  } catch {
    return null;
  }
}

// Translates the public REST /start body (snake_case) into the internal
// Durable Object body (camelCase).  Undefined fields are omitted so the DO
// can apply its own defaulting semantics (e.g. default -> arnold).
export function buildInternalStartBody(body: {
  avatar_id?: string;
  bg_id?: string;
  transport?: TransportPreference;
}): { avatarId?: string; bgId?: string; transport?: TransportPreference } {
  const internal: { avatarId?: string; bgId?: string; transport?: TransportPreference } = {};
  if (body.avatar_id !== undefined) internal.avatarId = body.avatar_id;
  if (body.bg_id !== undefined) internal.bgId = body.bg_id;
  if (body.transport !== undefined) internal.transport = body.transport;
  return internal;
}

type ApiPhase = "started" | "completed" | "failed";

function captureApi(
  env: Env,
  ctx: ExecutionContext,
  distinctId: string,
  phase: ApiPhase,
  route: string,
  method: string,
  status: number,
  t0: number,
  extra?: {
    avatarId?: string | null;
    errorCode?: string;
    authCtx?: ApiAuthContext | null;
    requestId?: string;
  },
): void {
  const event =
    phase === "started"
      ? "api_request_started"
      : phase === "completed"
        ? "api_request_completed"
        : "api_request_failed";
  captureEvent(env, ctx.waitUntil.bind(ctx), distinctId, event, {
    source: "v1_api",
    route,
    method,
    status,
    latencyMs: Date.now() - t0,
    sessionId: distinctId === API_DISTINCT_ID_DEFAULT ? null : distinctId,
    avatarId: extra?.avatarId ?? null,
    errorCode: extra?.errorCode ?? null,
    tenantId: extra?.authCtx?.tenant_id ?? null,
    keyName: extra?.authCtx?.key_name ?? null,
    plan: extra?.authCtx?.plan ?? null,
    authMode: extra?.authCtx?.auth_mode ?? null,
    requestId: extra?.requestId ?? null,
  });
}

// NOTE: errorCode extraction happens inline where the failure Response is
// constructed (the code is known at that point). For DO-routed failures the
// code is parsed during normalization. This avoids a second body read.

// ── Stream URL builder ─────────────────────────────────────────────────────

function buildStream(url: URL, sessionId: string): V1Stream {
  const wsProtocol = url.protocol === "https:" ? "wss:" : "ws:";
  return {
    transport: "websocket_fmp4",
    ws_url: `${wsProtocol}//${url.host}/v1/sessions/${sessionId}/ws`,
    status_url: `${url.origin}/v1/sessions/${sessionId}/status`,
  };
}

// ── Per-request context (mutable; carries telemetry metadata) ──────────────

interface RequestContext {
  sessionId: string | null;
  avatarId: string | null;
  requestId: string;
}

function extractOrGenerateRequestId(request: Request): string {
  const provided = request.headers.get("X-Request-ID") ?? request.headers.get("X-Request-Id");
  if (provided && /^[a-zA-Z0-9_-]{1,128}$/.test(provided)) {
    return provided;
  }
  return crypto.randomUUID();
}

// ── DO response normalization ──────────────────────────────────────────────

async function normalizeDoResponse(
  doResponse: Response,
  sessionId: string,
  url: URL,
  t0: number,
  rctx: RequestContext,
): Promise<Response> {
  const status = doResponse.status;
  const requestId = rctx.requestId;

  // SSE / event-stream must never be parsed or wrapped — pass through.
  const contentType = doResponse.headers.get("content-type") || "";
  if (contentType.includes("text/event-stream")) {
    return doResponse;
  }

  const text = await doResponse.text();
  let body: Record<string, unknown> | null = null;
  try {
    body = text ? (JSON.parse(text) as Record<string, unknown>) : null;
  } catch {
    body = null;
  }

  // ── Error path: preserve status, wrap error ──
  if (status >= 400) {
    const errObj =
      body && typeof body === "object" && "error" in body && body.error && typeof body.error === "object"
        ? (body.error as { code?: string; message?: string })
        : null;
    const code = errObj?.code ?? "internal_error";
    const message = errObj?.message ?? "Upstream error";
    return apiFail(code, message, t0, status, requestId);
  }

  // ── Success path: map DO body → canonical V1Session ──
  const stream = buildStream(url, sessionId);

  // Respect the transport reported by the DO (resolved at start time)
  const doTransport = body?.transport;
  if (doTransport === "webrtc_sfu" || doTransport === "websocket_fmp4") {
    stream.transport = doTransport;
  }

  const doStatusRaw = (body?.status as string) ?? "active";

  const avatarId = typeof body?.avatar_id === "string" ? (body.avatar_id as string) : null;
  if (avatarId) rctx.avatarId = avatarId;

  const created_at =
    typeof body?.created_at === "number" ? (body.created_at as number) : null;
  const started_at =
    typeof body?.started_at === "number" ? (body.started_at as number) : undefined;
  const ended_at =
    typeof body?.ended_at === "number" ? (body.ended_at as number) : undefined;

  let normalizedStatus: V1SessionStatus;
  if (doStatusRaw === "ended") normalizedStatus = "ended";
  else if (doStatusRaw === "ready") normalizedStatus = "ready";
  else if (doStatusRaw === "created") normalizedStatus = "created";
  else normalizedStatus = "active";

  const session: V1Session = {
    session_id: sessionId,
    avatar_id: avatarId,
    bg_id: null,
    status: normalizedStatus,
    created_at,
    stream,
  };

  if (started_at !== undefined) session.started_at = started_at;
  if (ended_at !== undefined) session.ended_at = ended_at;

  // start responses carry no created_at; record started_at as now.
  if (normalizedStatus === "ready" && started_at === undefined && ended_at === undefined) {
    session.started_at = Date.now();
  }

  if (normalizedStatus === "ended") {
    session.ended_at = ended_at ?? Date.now();
    session.end_ok = body?.modal_end_ok === true;
    const summaryRaw = body?.summary;
    if (summaryRaw && typeof summaryRaw === "object") {
      const s = summaryRaw as V1SessionEndSummary;
      const summary: V1SessionEndSummary = {};
      if (typeof s.chunks_rendered === "number") summary.chunks_rendered = s.chunks_rendered;
      if (typeof s.fmp4_media_count === "number") summary.fmp4_media_count = s.fmp4_media_count;
      if (typeof s.fmp4_media_total_bytes === "number")
        summary.fmp4_media_total_bytes = s.fmp4_media_total_bytes;
      if (typeof s.duration_s === "number") summary.duration_s = s.duration_s;
      if (s.ffmpeg_rc !== undefined) summary.ffmpeg_rc = s.ffmpeg_rc ?? null;
      session.end_summary = summary;
    }
  }

  return apiOk(session, t0, requestId);
}

// ── DO routing ─────────────────────────────────────────────────────────────

async function routeToDo(
  request: Request,
  env: Env,
  sessionId: string,
  operation: "start" | "end" | "status" | "stream",
  httpMethod: string,
  url: URL,
  t0: number,
  rctx: RequestContext,
): Promise<Response> {
  const doId = env.AVATAR_SESSION.idFromName(sessionId);
  const stub = env.AVATAR_SESSION.get(doId);

  const internalUrl = new URL(url);
  internalUrl.pathname = `/v1/sessions/${encodeURIComponent(sessionId)}/${operation}`;

  const headers = new Headers();
  headers.set("X-Internal-REST", "true");
  headers.set("X-REST-Operation", operation);
  const reqContentType = request.headers.get("content-type");
  if (reqContentType) headers.set("content-type", reqContentType);

  const forwardBody = httpMethod === "POST" && operation === "start" ? request.body : undefined;

  const internalRequest = new Request(internalUrl, {
    method: httpMethod,
    headers,
    body: forwardBody,
    signal: request.signal,
  });

  const doResponse = await stub.fetch(internalRequest);

  // SSE pass-through (also re-checked inside normalizeDoResponse for safety).
  const ct = doResponse.headers.get("content-type") || "";
  if (ct.includes("text/event-stream")) {
    return doResponse;
  }

  return normalizeDoResponse(doResponse, sessionId, url, t0, rctx);
}

// ── Ownership helpers ──────────────────────────────────────────────────────

export async function initSessionOwner(
  env: Env,
  sessionId: string,
  authCtx: ApiAuthContext,
  sfuBetaAuthorized: boolean,
): Promise<void> {
  const doId = env.AVATAR_SESSION.idFromName(sessionId);
  const stub = env.AVATAR_SESSION.get(doId);

  const initReq = new Request(
    `https://internal/v1/sessions/${encodeURIComponent(sessionId)}/init_owner`,
    {
      method: "POST",
      headers: {
        "X-Internal-REST": "true",
        "X-REST-Operation": "init_owner",
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tenant_id: authCtx.tenant_id,
        auth_mode: authCtx.auth_mode,
        key_name: authCtx.key_name,
        plan: authCtx.plan,
        sfu_beta_authorized: sfuBetaAuthorized,
      }),
    },
  );

  const res = await stub.fetch(initReq);
  if (!res.ok) {
    throw new Error(`init_owner failed with status ${res.status}`);
  }
}

export type OwnershipCheckResult = {
  allowed: boolean;
  reason?: "owner_not_found" | "tenant_mismatch";
};

export async function checkSessionOwnership(
  env: Env,
  sessionId: string,
  expectedTenantId: string,
): Promise<OwnershipCheckResult> {
  const doId = env.AVATAR_SESSION.idFromName(sessionId);
  const stub = env.AVATAR_SESSION.get(doId);

  const ownerReq = new Request(
    `https://internal/v1/sessions/${encodeURIComponent(sessionId)}/get_owner`,
    {
      headers: {
        "X-Internal-REST": "true",
        "X-REST-Operation": "get_owner",
      },
    },
  );

  try {
    const ownerRes = await stub.fetch(ownerReq);
    if (!ownerRes.ok) {
      return { allowed: false, reason: "owner_not_found" };
    }
    const data = (await ownerRes.json()) as { owner?: { tenant_id?: string } };
    if (data.owner?.tenant_id !== expectedTenantId) {
      return { allowed: false, reason: "tenant_mismatch" };
    }
    return { allowed: true };
  } catch {
    return { allowed: false, reason: "owner_not_found" };
  }
}

// ── Inline handlers ────────────────────────────────────────────────────────

function handleHealth(t0: number, requestId?: string): Response {
  const data: V1Health = { status: "ok", version: API_VERSION, transport: API_TRANSPORT };
  return apiOk(data, t0, requestId);
}

async function handleAvatars(
  env: Env,
  authCtx: ApiAuthContext,
  t0: number,
  requestId?: string,
): Promise<Response> {
  const provider = getCatalogProvider(env);
  const avatars = await provider.listAvatars(authCtx.tenant_id);
  return apiOk({ avatars }, t0, requestId);
}

async function handleAvatarDetail(
  env: Env,
  authCtx: ApiAuthContext,
  avatarId: string,
  t0: number,
  requestId?: string,
): Promise<Response> {
  const provider = getCatalogProvider(env);
  const avatar = await provider.getAvatar(authCtx.tenant_id, avatarId);
  if (!avatar) {
    return apiFail("avatar_not_found", `Avatar "${avatarId}" not found.`, t0, 404, requestId);
  }
  return apiOk({ avatar }, t0, requestId);
}

async function handleBackgrounds(
  env: Env,
  authCtx: ApiAuthContext,
  t0: number,
  requestId?: string,
): Promise<Response> {
  const provider = getCatalogProvider(env);
  const backgrounds = await provider.listBackgrounds(authCtx.tenant_id);
  return apiOk({ backgrounds }, t0, requestId);
}

async function handleBackgroundDetail(
  env: Env,
  authCtx: ApiAuthContext,
  bgId: string,
  t0: number,
  requestId?: string,
): Promise<Response> {
  const provider = getCatalogProvider(env);
  const background = await provider.getBackground(authCtx.tenant_id, bgId);
  if (!background) {
    return apiFail("background_not_found", `Background "${bgId}" not found.`, t0, 404, requestId);
  }
  return apiOk({ background }, t0, requestId);
}

// ── Providers capability (reads env only; no external calls) ───────────────

const RENDER_BACKEND = "managed_gpu_renderer";

function computeTtsChain(env: Env): {
  provider_chain: string;
  omnivoice_enabled: boolean;
  same_gpu_parallel_enabled: boolean;
  gpu_target: string | null;
  fallback_enabled: boolean;
} {
  const enableParallelFlag = env.ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS === "true";
  const gpuTarget = enableParallelFlag
    ? (env.OMNIVOICE_SAME_GPU_PARALLEL_TARGET ?? "l4")
    : "none";
  const parallelUrl = enableParallelFlag
    ? (gpuTarget === "l40s"
        ? env.MODAL_PARALLEL_L40S_LIFECYCLE_URL
        : gpuTarget === "a10g"
          ? env.MODAL_PARALLEL_A10G_LIFECYCLE_URL
          : env.MODAL_PARALLEL_LIFECYCLE_URL)
    : undefined;

  const sameGpuParallelEnabled = enableParallelFlag && !!parallelUrl;
  const omnivoiceEnabled =
    !sameGpuParallelEnabled &&
    env.ENABLE_OMNIVOICE_TTS_ON_L4 === "true" &&
    !!env.OMNIVOICE_TTS_URL;

  let providerChain: string;
  if (sameGpuParallelEnabled) {
    providerChain = "omnivoice_same_gpu_parallel_sarvam_melotts";
  } else if (omnivoiceEnabled) {
    providerChain = "omnivoice_sarvam_melotts";
  } else {
    providerChain = "sarvam_melotts";
  }

  return {
    provider_chain: providerChain,
    omnivoice_enabled: omnivoiceEnabled,
    same_gpu_parallel_enabled: sameGpuParallelEnabled,
    gpu_target: sameGpuParallelEnabled ? gpuTarget : null,
    fallback_enabled: true,
  };
}

function handleProviders(env: Env, t0: number, requestId?: string): Response {
  const data: V1Providers = {
    stt: {
      provider: "sarvam",
      model: env.STT_MODEL ?? "saarika:v2.5",
      language: env.STT_LANGUAGE_CODE ?? "unknown",
    },
    llm: {
      provider: "openai-compatible",
      model: env.LLM_MODEL ?? null,
    },
    tts: computeTtsChain(env),
    avatar: {
      transport: API_TRANSPORT,
      render_backend: RENDER_BACKEND,
    },
    vad: { enabled: !!env.VAD_SERVICE },
    smart_turn: { enabled: !!env.SMART_TURN_SERVICE },
  };
  return apiOk(data, t0, requestId);
}

function handleMe(authCtx: ApiAuthContext, t0: number, requestId?: string): Response {
  const data: V1Me = {
    tenant_id: authCtx.tenant_id,
    key_name: authCtx.key_name,
    plan: authCtx.plan,
    auth_mode: authCtx.auth_mode,
  };
  return apiOk(data, t0, requestId);
}

async function handleCreateSession(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  url: URL,
  t0: number,
  authCtx: ApiAuthContext,
  requestId?: string,
): Promise<Response> {
  if (!env.AVTR_SESSION_TOKEN_SECRET) {
    return apiFail(
      "ws_token_secret_not_configured",
      "WebSocket token secret not configured.",
      t0,
      503,
      requestId,
    );
  }

  const sessionId = `${SESSION_PREFIX}${crypto.randomUUID()}`;

  // Phase 3F: resolve SFU private beta authorization before creating owner.
  // Token may be passed via header or JSON body; the raw value is never stored.
  let headerToken = request.headers.get("X-SFU-Beta-Token")?.trim() ?? undefined;
  let bodyToken: string | undefined;
  const contentType = request.headers.get("content-type") ?? "";
  if (contentType.includes("application/json")) {
    try {
      const text = await request.text();
      const body = text ? (JSON.parse(text) as { sfu_beta_token?: unknown }) : {};
      if (typeof body.sfu_beta_token === "string" && body.sfu_beta_token.trim() !== "") {
        bodyToken = body.sfu_beta_token.trim();
      }
    } catch {
      // ignore malformed body; beta auth will fall through to unauthorized
    }
  }
  const betaToken = bodyToken ?? headerToken;
  const betaAuth = await resolveSfuBetaAuthorization(env, {
    tenantId: authCtx.tenant_id,
    metadataSfuBeta: authCtx.sfu_beta,
    betaToken,
  });

  try {
    await initSessionOwner(env, sessionId, authCtx, betaAuth.authorized);
  } catch {
    return apiFail("internal_error", "Failed to initialize session.", t0, 500, requestId);
  }

  // Phase 6a.1: session creation does not pre-select avatar/background.
  // Selection happens at WS session.start or REST /start.

  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ownership_created", {
    callerTenantId: authCtx.tenant_id,
    authMode: authCtx.auth_mode,
    sfuBetaSource: betaAuth.source,
    sfuBetaAuthorized: betaAuth.authorized,
    route: "/v1/sessions",
    requestId,
  });

  const stream = buildStream(url, sessionId);

  const { token, expiresAt } = await createWsToken(
    env.AVTR_SESSION_TOKEN_SECRET,
    sessionId,
    authCtx.tenant_id,
  );
  stream.ws_url = `${stream.ws_url}?token=${token}`;
  stream.token_expires_at = expiresAt;

  // Phase D1: Mint sfu_token and add SFU endpoints only when fully configured.
  // Requires ALL of: AVTR_REALTIME_SFU_ENABLED, CALLS_APP_ID,
  // CALLS_APP_SECRET, MODAL_WORKER_SECRET, WORKER_BASE_URL.
  // Phase 3F: SFU transport is also gated by private beta authorization.
  // The sfu_token uses purpose:"sfu" — separate from the ws_token (purpose:"ws").
  // stream.transport stays "websocket_fmp4" until start resolves it.
  const sfuReady = isSfuFullyConfigured(env);
  const sfuAdvertised = sfuReady && betaAuth.authorized;
  stream.available_transports = sfuAdvertised
    ? ["websocket_fmp4", "webrtc_sfu"]
    : ["websocket_fmp4"];

  if (sfuAdvertised) {
    const sfuTokenResult = await createWsToken(
      env.AVTR_SESSION_TOKEN_SECRET,
      sessionId,
      authCtx.tenant_id,
      undefined,
      "sfu",
    );
    const sfuBlock: V1StreamSfu = {
      subscribe_url: `${url.origin}/v1/sessions/${sessionId}/sfu/subscribe`,
      answer_url: `${url.origin}/v1/sessions/${sessionId}/sfu/answer`,
      close_url: `${url.origin}/v1/sessions/${sessionId}/sfu/close`,
      token: sfuTokenResult.token,
      token_expires_at: sfuTokenResult.expiresAt,
      track_names: ["avatar-video"],
    };
    stream.sfu = sfuBlock;

    captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "sfu_token_issued", {
      source: "v1_session_create",
      tenantId: authCtx.tenant_id,
      expiresAt: sfuTokenResult.expiresAt,
      requestId,
    });
  }

  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "avatar_session_created", {
    avatarSessionId: sessionId,
    source: "v1_rest",
    requestId,
  });
  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_token_issued", {
    source: "v1_session_create",
    tenantId: authCtx.tenant_id,
    authMode: authCtx.auth_mode,
    expiresAt,
    requestId,
  });

  const session: V1Session = {
    session_id: sessionId,
    avatar_id: null,
    bg_id: null,
    status: "created",
    created_at: Date.now(),
    stream,
  };

  return apiOk(session, t0, requestId);
}

// ── WS token refresh ─────────────────────────────────────────────────────

async function handleWsTokenRefresh(
  env: Env,
  ctx: ExecutionContext,
  url: URL,
  sessionId: string,
  authCtx: ApiAuthContext,
  t0: number,
  requestId?: string,
): Promise<Response> {
  if (!env.AVTR_SESSION_TOKEN_SECRET) {
    return apiFail(
      "ws_token_secret_not_configured",
      "WebSocket token secret not configured.",
      t0,
      503,
      requestId,
    );
  }

  const wsProtocol = url.protocol === "https:" ? "wss:" : "ws:";
  const { token, expiresAt } = await createWsToken(
    env.AVTR_SESSION_TOKEN_SECRET,
    sessionId,
    authCtx.tenant_id,
  );

  captureEvent(env, ctx.waitUntil.bind(ctx), sessionId, "ws_token_issued", {
    source: "v1_ws_token_refresh",
    tenantId: authCtx.tenant_id,
    authMode: authCtx.auth_mode,
    expiresAt,
    requestId,
  });

  // Also refresh sfu_token if SFU is fully configured
  if (isSfuFullyConfigured(env)) {
    const sfuTokenResult = await createWsToken(
      env.AVTR_SESSION_TOKEN_SECRET,
      sessionId,
      authCtx.tenant_id,
      undefined,
      "sfu",
    );
    return apiOk({
      ws_url: `${wsProtocol}//${url.host}/v1/sessions/${sessionId}/ws?token=${token}`,
      token_expires_at: expiresAt,
      sfu_token: sfuTokenResult.token,
      sfu_token_expires_at: sfuTokenResult.expiresAt,
    }, t0, requestId);
  }

  return apiOk({
    ws_url: `${wsProtocol}//${url.host}/v1/sessions/${sessionId}/ws?token=${token}`,
    token_expires_at: expiresAt,
  }, t0, requestId);
}

// ── Dispatch ───────────────────────────────────────────────────────────────

async function dispatchV1(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
  url: URL,
  path: string,
  method: string,
  t0: number,
  rctx: RequestContext,
  authCtx: ApiAuthContext,
): Promise<Response> {
  const rid = rctx.requestId;

  // GET /v1/health
  if (method === "GET" && path === "/v1/health") {
    return handleHealth(t0, rid);
  }

  // GET /v1/me
  if (method === "GET" && path === "/v1/me") {
    return handleMe(authCtx, t0, rid);
  }

  // GET /v1/avatars
  if (method === "GET" && path === "/v1/avatars") {
    return handleAvatars(env, authCtx, t0, rid);
  }

  // GET /v1/avatars/:avatar_id
  const avatarDetailMatch = method === "GET" && path.match(/^\/v1\/avatars\/([^/]+)$/);
  if (avatarDetailMatch) {
    return handleAvatarDetail(env, authCtx, decodeURIComponent(avatarDetailMatch[1]), t0, rid);
  }

  // GET /v1/backgrounds
  if (method === "GET" && path === "/v1/backgrounds") {
    return handleBackgrounds(env, authCtx, t0, rid);
  }

  // GET /v1/backgrounds/:bg_id
  const backgroundDetailMatch = method === "GET" && path.match(/^\/v1\/backgrounds\/([^/]+)$/);
  if (backgroundDetailMatch) {
    return handleBackgroundDetail(env, authCtx, decodeURIComponent(backgroundDetailMatch[1]), t0, rid);
  }

  // GET /v1/providers
  if (method === "GET" && path === "/v1/providers") {
    return handleProviders(env, t0, rid);
  }

  // POST /v1/sessions
  if (method === "POST" && path === "/v1/sessions") {
    return await handleCreateSession(request, env, ctx, url, t0, authCtx, rid);
  }

  // ── Ownership gate: applies to ALL /v1/sessions/:id/* routes ──────────
  const scopedMatch = path.match(/^\/v1\/sessions\/([^/]+)/);
  if (scopedMatch) {
    const gateSessionId = decodeURIComponent(scopedMatch[1]);
    if (!gateSessionId.startsWith(SESSION_PREFIX)) {
      return apiFail("invalid_session_id", "Session ID must start with 'session:'", t0, 400, rid);
    }

    const ownership = await checkSessionOwnership(env, gateSessionId, authCtx.tenant_id);
    if (!ownership.allowed) {
      captureEvent(env, ctx.waitUntil.bind(ctx), gateSessionId, "ownership_check_failed", {
        reason: ownership.reason,
        callerTenantId: authCtx.tenant_id,
        authMode: authCtx.auth_mode,
        route: routeForTelemetry(path),
        requestId: rid,
      });
      return apiFail("not_found", "Session not found.", t0, 404, rid);
    }
    captureEvent(env, ctx.waitUntil.bind(ctx), gateSessionId, "ownership_check_passed", {
      callerTenantId: authCtx.tenant_id,
      authMode: authCtx.auth_mode,
      route: routeForTelemetry(path),
      requestId: rid,
    });
  }

  // POST /v1/sessions/:id/ws-token
  const wsTokenMatch = method === "POST" && path.match(/^\/v1\/sessions\/([^/]+)\/ws-token$/);
  if (wsTokenMatch) {
    const sessionId = decodeURIComponent(wsTokenMatch[1]);
    if (!sessionId.startsWith(SESSION_PREFIX)) {
      return apiFail("invalid_session_id", "Session ID must start with 'session:'", t0, 400, rid);
    }
    rctx.sessionId = sessionId;
    return handleWsTokenRefresh(env, ctx, url, sessionId, authCtx, t0, rid);
  }

  // DELETE /v1/sessions/:id  (canonical end; routes to DO end op)
  const deleteMatch = method === "DELETE" && path.match(/^\/v1\/sessions\/([^/]+)$/);
  if (deleteMatch) {
    const sessionId = decodeURIComponent(deleteMatch[1]);
    if (!sessionId.startsWith(SESSION_PREFIX)) {
      return apiFail("invalid_session_id", "Session ID must start with 'session:'", t0, 400, rid);
    }
    rctx.sessionId = sessionId;
    return routeToDo(request, env, sessionId, "end", "POST", url, t0, rctx);
  }

  // POST /v1/sessions/:id/start — validate avatar/background before routing to DO.
  const startMatch = method === "POST" && path.match(/^\/v1\/sessions\/([^/]+)\/start$/);
  if (startMatch) {
    const sessionId = decodeURIComponent(startMatch[1]);
    if (!sessionId.startsWith(SESSION_PREFIX)) {
      return apiFail("invalid_session_id", "Session ID must start with 'session:'", t0, 400, rid);
    }

    let body: { avatar_id?: string; bg_id?: string; transport?: TransportPreference } = {};
    const contentType = request.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      try {
        const text = await request.text();
        if (text) body = JSON.parse(text) as typeof body;
      } catch {
        return apiFail("invalid_body", "Request body must be valid JSON.", t0, 400, rid);
      }
    }

    const resolvedAvatarId = (body.avatar_id || "default") === "default" ? "arnold" : (body.avatar_id as string);
    const validation = await validateAvatarBgForStart(
      env,
      authCtx.tenant_id,
      resolvedAvatarId,
      body.bg_id,
    );
    if (!validation.ok) {
      return apiFail(validation.code, validation.message, t0, 400, rid);
    }

    // Re-create the request so the DO can still read the body if needed.
    // Public REST uses snake_case; the DO expects camelCase.
    const internalHeaders = new Headers(request.headers);
    internalHeaders.set("content-type", "application/json");
    const forwardRequest = new Request(request.url, {
      method: request.method,
      headers: internalHeaders,
      body: JSON.stringify(buildInternalStartBody(body)),
    });

    rctx.sessionId = sessionId;
    rctx.avatarId = resolvedAvatarId;
    return routeToDo(forwardRequest, env, sessionId, "start", "POST", url, t0, rctx);
  }

  // /v1/sessions/:id/{end,status,stream}
  const opMatch = path.match(/^\/v1\/sessions\/([^/]+)\/(end|status|stream)$/);
  if (opMatch) {
    const sessionId = decodeURIComponent(opMatch[1]);
    const operation = opMatch[2] as "end" | "status" | "stream";

    if (!sessionId.startsWith(SESSION_PREFIX)) {
      return apiFail("invalid_session_id", "Session ID must start with 'session:'", t0, 400, rid);
    }

    const expectedMethod: Record<typeof operation, string> = {
      end: "POST",
      status: "GET",
      stream: "GET",
    };
    if (method !== expectedMethod[operation]) {
      return apiFail(
        "method_not_allowed",
        `${method} not allowed for /v1/sessions/:id/${operation}`,
        t0,
        405,
        rid,
      );
    }

    rctx.sessionId = sessionId;
    return routeToDo(request, env, sessionId, operation, method, url, t0, rctx);
  }

  return apiFail("not_found", `No route for ${method} ${path}`, t0, 404, rid);
}

// ── Entry point ────────────────────────────────────────────────────────────

export async function handleV1Api(
  request: Request,
  env: Env,
  ctx: ExecutionContext,
): Promise<Response> {
  const t0 = Date.now();
  const url = new URL(request.url);
  const path = url.pathname;
  const method = request.method;
  const route = routeForTelemetry(path);
  const requestId = extractOrGenerateRequestId(request);

  const rctx: RequestContext = {
    sessionId: sessionIdFromPath(path),
    avatarId: null,
    requestId,
  };
  const distinctId = rctx.sessionId ?? API_DISTINCT_ID_DEFAULT;

  const authResult = authenticate(request, env, t0, requestId);
  if (authResult instanceof Response) {
    const errorCode = await extractErrorCode(authResult) ?? undefined;
    captureApi(env, ctx, distinctId, "failed", route, method, authResult.status, t0, {
      errorCode,
      requestId,
    });
    return withRequestId(authResult, requestId);
  }

  const authCtx = authResult;

  const category = categorizeRoute(method, path);
  if (category !== "cleanup") {
    const rlResult = await checkRateLimit(env, authCtx.tenant_id, category);
    if (!rlResult.allowed) {
      captureEvent(env, ctx.waitUntil.bind(ctx), distinctId, "api_rate_limited", {
        source: "v1_api",
        tenantId: authCtx.tenant_id,
        authMode: authCtx.auth_mode,
        category,
        route,
        limit: rlResult.limit ?? null,
        period: rlResult.period ?? null,
        requestId,
      });
      return withRequestId(
        rateLimitedResponse(category, rlResult.limit!, rlResult.period!, t0, requestId),
        requestId,
      );
    }
    if (rlResult.bindingMissing) {
      captureEvent(env, ctx.waitUntil.bind(ctx), distinctId, "api_rate_limit_binding_missing", {
        source: "v1_api",
        tenantId: authCtx.tenant_id,
        authMode: authCtx.auth_mode,
        category,
        bindingName: rlResult.bindingName ?? null,
        route,
        requestId,
      });
    }
  }

  captureApi(env, ctx, distinctId, "started", route, method, 0, t0, {
    authCtx,
    requestId,
  });

  let response: Response;
  try {
    response = await dispatchV1(request, env, ctx, url, path, method, t0, rctx, authCtx);
  } catch (err) {
    console.error("[v1 api] dispatch error", {
      route,
      method,
      requestId,
      error: err instanceof Error ? err.message : String(err),
    });
    response = apiFail("internal_error", "Internal server error", t0, 500, requestId);
  }

  const phase: ApiPhase = response.status < 400 ? "completed" : "failed";
  const errorCode =
    phase === "failed" ? (await extractErrorCode(response)) ?? undefined : undefined;
  captureApi(env, ctx, distinctId, phase, route, method, response.status, t0, {
    avatarId: rctx.avatarId,
    errorCode,
    authCtx,
    requestId,
  });

  return withRequestId(response, requestId);
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
