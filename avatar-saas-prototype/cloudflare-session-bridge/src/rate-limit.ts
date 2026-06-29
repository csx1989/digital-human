import type { Env } from "./types";

export type RateCategory =
  | "gpu_start"
  | "session_create"
  | "ws_token"
  | "stream_connect"
  | "ws_connect"
  | "read"
  | "cleanup";

interface CategoryConfig {
  binding: keyof Pick<Env, "RL_HEAVY" | "RL_WRITE" | "RL_LIGHT">;
  keySuffix: string;
  limit: number;
  period: number;
}

const CATEGORY_CONFIG: Record<RateCategory, CategoryConfig | null> = {
  gpu_start:      { binding: "RL_HEAVY", keySuffix: "gpu_start",      limit: 5,   period: 60 },
  stream_connect: { binding: "RL_HEAVY", keySuffix: "stream_connect", limit: 5,   period: 60 },
  ws_connect:     { binding: "RL_HEAVY", keySuffix: "ws_connect",     limit: 5,   period: 60 },
  session_create: { binding: "RL_WRITE", keySuffix: "session_create", limit: 20,  period: 60 },
  ws_token:       { binding: "RL_WRITE", keySuffix: "ws_token",       limit: 20,  period: 60 },
  read:           { binding: "RL_LIGHT", keySuffix: "read",           limit: 120, period: 60 },
  cleanup:        null,
};

const CORS_HEADERS: Record<string, string> = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, x-debug-token, Authorization, x-api-key",
};

export function categorizeRoute(method: string, path: string): RateCategory {
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/start$/.test(path)) return "gpu_start";
  if (method === "POST" && path === "/v1/sessions") return "session_create";
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/ws-token$/.test(path)) return "ws_token";
  if (method === "GET" && /^\/v1\/sessions\/[^/]+\/stream$/.test(path)) return "stream_connect";
  if (method === "GET" && /^\/v1\/sessions\/[^/]+\/ws$/.test(path)) return "ws_connect";
  if (method === "DELETE" && /^\/v1\/sessions\/[^/]+$/.test(path)) return "cleanup";
  if (method === "POST" && /^\/v1\/sessions\/[^/]+\/end$/.test(path)) return "cleanup";
  return "read";
}

export interface RateLimitResult {
  allowed: boolean;
  category: RateCategory;
  limit?: number;
  period?: number;
  bindingMissing?: boolean;
  bindingName?: string;
}

export async function checkRateLimit(
  env: Env,
  tenantId: string,
  category: RateCategory,
): Promise<RateLimitResult> {
  const config = CATEGORY_CONFIG[category];
  if (!config) {
    return { allowed: true, category };
  }

  const binding = env[config.binding];
  if (!binding) {
    return {
      allowed: true,
      category,
      limit: config.limit,
      period: config.period,
      bindingMissing: true,
      bindingName: config.binding,
    };
  }

  const key = `t:${tenantId}:${config.keySuffix}`;
  const { success } = await binding.limit({ key });

  return {
    allowed: success,
    category,
    limit: config.limit,
    period: config.period,
  };
}

export function rateLimitedResponse(
  category: RateCategory,
  limit: number,
  period: number,
  t0: number,
  requestId?: string,
): Response {
  const body = {
    success: false as const,
    error: {
      code: "rate_limited" as const,
      message: `Rate limit exceeded for category "${category}". Retry after ${period} seconds.`,
    },
    timings: { latency_ms: Date.now() - t0, ...(requestId ? { request_id: requestId } : {}) },
  };

  const headers = new Headers();
  for (const [k, v] of Object.entries(CORS_HEADERS)) headers.set(k, v);
  headers.set("Content-Type", "application/json");
  headers.set("Retry-After", String(period));

  return new Response(JSON.stringify(body), { status: 429, headers });
}
