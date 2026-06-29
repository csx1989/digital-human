// ── SFU Private Beta Authorization ───────────────────────────────────────────
//
// Pure helpers for authorizing access to the Cloudflare Realtime SFU transport.
// Supports three sources (checked in order):
//   1. Per-API-key metadata flag ("sfu_beta": true).
//   2. Tenant allowlist (AVTR_SFU_BETA_TENANTS env var).
//   3. Short-lived HMAC-signed beta token (AVTR_SFU_BETA_TOKEN_SECRET).
//
// The raw beta token is verified server-side and then discarded. Only a
// boolean authorization flag is persisted on the session.

import { safeEqual } from "./auth-core";

export const SFU_BETA_TOKEN_AUDIENCE = "sfu-beta";

export interface SfuBetaTokenPayload {
  aud: typeof SFU_BETA_TOKEN_AUDIENCE;
  iat: number;
  exp: number;
}

export interface SfuBetaAuthorization {
  authorized: boolean;
  source: "metadata" | "tenant" | "token" | "none";
}

const DEFAULT_TOKEN_TTL_SECONDS = 3600;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// ── Module-scope HMAC key cache (per isolate) ────────────────────────────────

let _cachedKey: CryptoKey | null = null;
let _cachedSecret = "";

async function getHmacKey(secret: string): Promise<CryptoKey> {
  if (_cachedKey && _cachedSecret === secret) return _cachedKey;
  _cachedKey = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
  _cachedSecret = secret;
  return _cachedKey;
}

// ── base64url helpers ───────────────────────────────────────────────────────

function base64urlEncode(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecodeToStr(b64url: string): string {
  const b64 = b64url.replace(/-/g, "+").replace(/_/g, "/");
  const padded = b64 + "=".repeat((4 - (b64.length % 4)) % 4);
  const binary = atob(padded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i);
  }
  return decoder.decode(bytes);
}

async function hmacSign(key: CryptoKey, data: string): Promise<string> {
  const sig = await crypto.subtle.sign("HMAC", key, encoder.encode(data));
  return base64urlEncode(sig);
}

// ── Token creation ───────────────────────────────────────────────────────────

export async function createSfuBetaToken(
  secret: string,
  ttlSeconds: number = DEFAULT_TOKEN_TTL_SECONDS,
): Promise<string> {
  const now = Math.floor(Date.now() / 1000);
  const payload: SfuBetaTokenPayload = {
    aud: SFU_BETA_TOKEN_AUDIENCE,
    iat: now,
    exp: now + ttlSeconds,
  };

  const payloadJson = JSON.stringify(payload);
  const payloadB64 = base64urlEncode(encoder.encode(payloadJson).buffer as ArrayBuffer);

  const key = await getHmacKey(secret);
  const sigB64 = await hmacSign(key, payloadB64);

  return `${payloadB64}.${sigB64}`;
}

// ── Token verification ───────────────────────────────────────────────────────

export async function verifySfuBetaToken(
  secret: string | undefined,
  token: string,
): Promise<boolean> {
  if (!secret) return false;

  // 1. Structural check: must be exactly two dot-separated parts.
  const parts = token.split(".");
  if (parts.length !== 2) return false;

  const [payloadB64, sigB64] = parts;

  // 2. Signature check: recompute HMAC and compare in constant time.
  const key = await getHmacKey(secret);
  const expectedSig = await hmacSign(key, payloadB64);
  if (!safeEqual(expectedSig, sigB64)) return false;

  // 3. Decode payload.
  let payload: unknown;
  try {
    payload = JSON.parse(base64urlDecodeToStr(payloadB64));
  } catch {
    return false;
  }

  // 4. Strict payload shape validation.
  if (
    typeof payload !== "object" ||
    payload === null ||
    (payload as Record<string, unknown>).aud !== SFU_BETA_TOKEN_AUDIENCE ||
    typeof (payload as Record<string, unknown>).iat !== "number" ||
    typeof (payload as Record<string, unknown>).exp !== "number" ||
    !Number.isFinite((payload as Record<string, unknown>).iat as number) ||
    !Number.isFinite((payload as Record<string, unknown>).exp as number)
  ) {
    return false;
  }

  const p = payload as SfuBetaTokenPayload;

  // 5. Expiry check.
  const now = Math.floor(Date.now() / 1000);
  if (p.exp < now) return false;

  return true;
}

// ── Tenant allowlist ─────────────────────────────────────────────────────────

export function parseSfuBetaTenants(raw: string | undefined): Set<string> {
  const s = raw?.trim() ?? "";
  if (!s) return new Set();
  return new Set(
    s
      .split(",")
      .map((t) => t.trim())
      .filter(Boolean),
  );
}

// ── Authorization resolver ───────────────────────────────────────────────────

export interface ResolveSfuBetaAuthOptions {
  tenantId?: string;
  metadataSfuBeta?: boolean;
  betaToken?: string;
}

export async function resolveSfuBetaAuthorization(
  env: { AVTR_SFU_BETA_TOKEN_SECRET?: string; AVTR_SFU_BETA_TENANTS?: string },
  opts: ResolveSfuBetaAuthOptions,
): Promise<SfuBetaAuthorization> {
  if (opts.metadataSfuBeta === true) {
    return { authorized: true, source: "metadata" };
  }

  if (opts.tenantId) {
    const tenants = parseSfuBetaTenants(env.AVTR_SFU_BETA_TENANTS);
    if (tenants.has(opts.tenantId)) {
      return { authorized: true, source: "tenant" };
    }
  }

  if (opts.betaToken) {
    const valid = await verifySfuBetaToken(env.AVTR_SFU_BETA_TOKEN_SECRET, opts.betaToken);
    if (valid) {
      return { authorized: true, source: "token" };
    }
  }

  return { authorized: false, source: "none" };
}
