// ── Session Token (WebSocket + SFU) ─────────────────────────────────────────
//
// Stateless HMAC-SHA256 signed tokens for authenticating browser-facing
// endpoints (WebSocket upgrades and SFU REST routes).  No DB/KV lookup
// needed — the signature proves the Worker minted the token.
//
// Token format:  base64url(payload).base64url(hmac_sha256(payload_b64, secret))
//
// Payload fields:
//   sid     — session ID (must match URL path)
//   tid     — tenant ID (from API auth context)
//   purpose — "ws" for WebSocket, "sfu" for SFU REST routes (scope limiter)
//   exp     — expiry in unix seconds
//
// Zero non-platform dependencies.  Uses Web Crypto (crypto.subtle), which is
// available in both the Workers runtime and Node.js 18+ (for offline tests).

import { safeEqual } from "./auth-core";

export type SessionTokenPurpose = "ws" | "sfu";

export interface WsTokenPayload {
  sid: string;
  tid: string;
  purpose: SessionTokenPurpose;
  exp: number;
}

export interface WsTokenCreateResult {
  token: string;
  expiresAt: number;
}

export type WsTokenRejectReason =
  | "malformed"
  | "invalid_signature"
  | "expired"
  | "session_mismatch"
  | "wrong_purpose"
  | "purpose_not_allowed";

export interface WsTokenVerifyResult {
  valid: boolean;
  reason?: WsTokenRejectReason;
  tenantId?: string;
}

const DEFAULT_TTL_SECONDS = 300;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// ── Module-scope HMAC key cache (per isolate) ──────────────────────────────

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

// ── Token creation ──────────────────────────────────────────────────────────

export async function createWsToken(
  secret: string,
  sessionId: string,
  tenantId: string,
  ttlSeconds: number = DEFAULT_TTL_SECONDS,
  purpose: SessionTokenPurpose = "ws",
): Promise<WsTokenCreateResult> {
  const now = Math.floor(Date.now() / 1000);
  const exp = now + ttlSeconds;

  const payload: WsTokenPayload = {
    sid: sessionId,
    tid: tenantId,
    purpose,
    exp,
  };

  const payloadJson = JSON.stringify(payload);
  const payloadB64 = base64urlEncode(encoder.encode(payloadJson).buffer as ArrayBuffer);

  const key = await getHmacKey(secret);
  const sigB64 = await hmacSign(key, payloadB64);

  return {
    token: `${payloadB64}.${sigB64}`,
    expiresAt: exp * 1000,
  };
}

// ── Token verification ──────────────────────────────────────────────────────

export async function verifyWsToken(
  secret: string,
  token: string,
  expectedSessionId: string,
  allowedPurposes: SessionTokenPurpose[] = ["ws"],
): Promise<WsTokenVerifyResult> {
  // 1. Structural check: must be exactly two dot-separated parts.
  const parts = token.split(".");
  if (parts.length !== 2) {
    return { valid: false, reason: "malformed" };
  }

  const [payloadB64, sigB64] = parts;

  // 2. Signature check: recompute HMAC and compare in constant time.
  const key = await getHmacKey(secret);
  const expectedSig = await hmacSign(key, payloadB64);
  if (!safeEqual(expectedSig, sigB64)) {
    return { valid: false, reason: "invalid_signature" };
  }

  // 3. Decode payload.
  let payload: unknown;
  try {
    payload = JSON.parse(base64urlDecodeToStr(payloadB64));
  } catch {
    return { valid: false, reason: "malformed" };
  }

  // 4. Strict payload shape validation — guardrail #2.
  if (
    typeof payload !== "object" ||
    payload === null ||
    typeof (payload as Record<string, unknown>).sid !== "string" ||
    typeof (payload as Record<string, unknown>).tid !== "string" ||
    typeof (payload as Record<string, unknown>).purpose !== "string" ||
    typeof (payload as Record<string, unknown>).exp !== "number" ||
    !Number.isFinite((payload as Record<string, unknown>).exp as number)
  ) {
    return { valid: false, reason: "malformed" };
  }

  const p = payload as WsTokenPayload;

  // 5. Expiry check.
  const now = Math.floor(Date.now() / 1000);
  if (p.exp < now) {
    return { valid: false, reason: "expired" };
  }

  // 6. Purpose check — must be in the allowed set.
  if (!allowedPurposes.includes(p.purpose)) {
    return { valid: false, reason: "purpose_not_allowed" };
  }

  // 7. Session binding check.
  if (p.sid !== expectedSessionId) {
    return { valid: false, reason: "session_mismatch" };
  }

  return { valid: true, tenantId: p.tid };
}
