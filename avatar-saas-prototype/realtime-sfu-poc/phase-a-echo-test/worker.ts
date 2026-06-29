/**
 * Phase A — Realtime SFU Echo Test Worker
 *
 * Stateless auth proxy for Cloudflare Realtime SFU.
 * - Holds CALLS_APP_SECRET server-side; browser never sees it.
 * - Protected by POC_ACCESS_TOKEN on ALL /api/* calls via x-poc-token header.
 * - Strict allowlist of proxied SFU endpoints (no generic path forwarding).
 * - Non-API requests fall through to Workers Assets (index.html).
 */

interface Env {
  CALLS_APP_ID: string;
  CALLS_APP_SECRET: string;
  POC_ACCESS_TOKEN: string;
  ASSETS: Fetcher;
}

const SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps";

const CORS_HEADERS: Record<string, string> = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET, POST, PUT, OPTIONS",
  "access-control-allow-headers": "content-type, x-poc-token",
};

// SFU session IDs are hex strings — validate to prevent path injection.
const SESSION_ID_RE = /^[a-f0-9-]{8,}$/i;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...CORS_HEADERS },
  });
}

function checkPocToken(request: Request, env: Env): Response | null {
  if (!env.POC_ACCESS_TOKEN) {
    return jsonResponse({ error: "server_not_configured", detail: "POC_ACCESS_TOKEN not set" }, 503);
  }
  const token = request.headers.get("x-poc-token");
  if (!token || token !== env.POC_ACCESS_TOKEN) {
    return jsonResponse({ error: "unauthorized" }, 401);
  }
  return null;
}

function requireSfuCreds(env: Env): Response | null {
  if (!env.CALLS_APP_ID || !env.CALLS_APP_SECRET) {
    return jsonResponse({ error: "sfu_not_configured" }, 503);
  }
  return null;
}

async function proxySfu(env: Env, sfuPath: string, method: string, requestBody?: string): Promise<Response> {
  const url = `${SFU_API_BASE}/${env.CALLS_APP_ID}${sfuPath}`;
  const resp = await fetch(url, {
    method,
    headers: {
      authorization: `Bearer ${env.CALLS_APP_SECRET}`,
      "content-type": "application/json",
    },
    body: requestBody || undefined,
  });
  const text = await resp.text();
  return new Response(text, {
    status: resp.status,
    headers: {
      "content-type": resp.headers.get("content-type") ?? "application/json",
      "x-sfu-status": String(resp.status),
      ...CORS_HEADERS,
    },
  });
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Non-API requests → serve static assets (index.html etc.)
    if (!url.pathname.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    // ---- GET /api/health ----
    // Returns only { configured: boolean } — no detailed secret diagnostics.
    if (url.pathname === "/api/health" && request.method === "GET") {
      const gate = checkPocToken(request, env);
      if (gate) return gate;
      return jsonResponse({ configured: !!(env.CALLS_APP_ID && env.CALLS_APP_SECRET) });
    }

    // All remaining routes require BOTH POC token and SFU credentials
    const tokenError = checkPocToken(request, env);
    if (tokenError) return tokenError;
    const credsError = requireSfuCreds(env);
    if (credsError) return credsError;

    // ---- POST /api/sessions/new ----
    if (url.pathname === "/api/sessions/new" && request.method === "POST") {
      return proxySfu(env, "/sessions/new", "POST");
    }

    // ---- POST /api/sessions/:sessionId/tracks/new ----
    {
      const m = url.pathname.match(/^\/api\/sessions\/([^/]+)\/tracks\/new$/);
      if (m && request.method === "POST") {
        const sessionId = m[1];
        if (!SESSION_ID_RE.test(sessionId)) return jsonResponse({ error: "invalid_session_id" }, 400);
        const body = await request.text();
        return proxySfu(env, `/sessions/${sessionId}/tracks/new`, "POST", body);
      }
    }

    // ---- PUT /api/sessions/:sessionId/renegotiate ----
    {
      const m = url.pathname.match(/^\/api\/sessions\/([^/]+)\/renegotiate$/);
      if (m && request.method === "PUT") {
        const sessionId = m[1];
        if (!SESSION_ID_RE.test(sessionId)) return jsonResponse({ error: "invalid_session_id" }, 400);
        const body = await request.text();
        return proxySfu(env, `/sessions/${sessionId}/renegotiate`, "PUT", body);
      }
    }

    // ---- PUT /api/sessions/:sessionId/tracks/close ----
    // Best-effort: swallow errors so teardown never blocks.
    {
      const m = url.pathname.match(/^\/api\/sessions\/([^/]+)\/tracks\/close$/);
      if (m && request.method === "PUT") {
        const sessionId = m[1];
        if (!SESSION_ID_RE.test(sessionId)) return jsonResponse({ error: "invalid_session_id" }, 400);
        const body = await request.text();
        try {
          return await proxySfu(env, `/sessions/${sessionId}/tracks/close`, "PUT", body);
        } catch {
          return jsonResponse({ closed: false, bestEffort: true });
        }
      }
    }

    // ---- Fallback ----
    return jsonResponse({ error: "not_found" }, 404);
  },
};
