// ── Modal Request Signing ────────────────────────────────────────────────────
//
// Stateless HMAC-SHA256 request signing for Cloudflare Worker → Modal internal
// endpoints. This is NOT browser/API auth; it authenticates outbound internal
// requests from the Worker/DO to Modal GPU services.
//
// Signature base string:
//   METHOD + "\n" + CANONICAL_URL + "\n" + TIMESTAMP + "\n" + REQUEST_ID + "\n" + BODY_SHA256
//
// The URL is canonicalized via new URL().href before signing. This ensures the
// Worker (which reads the raw env-var URL, e.g. "https://host.modal.run") and
// Modal (which sees str(request.url), e.g. "https://host.modal.run/") agree on
// the exact string. Both add the trailing slash for the root path.
//
// Headers:
//   X-AVTR-Timestamp   - Unix seconds
//   X-AVTR-Request-Id  - UUID v4
//   X-AVTR-Body-SHA256 - Hex SHA-256 of the exact request body bytes
//   X-AVTR-Signature   - Hex HMAC-SHA256 of the base string
//
// If AVTR_MODAL_REQUEST_SIGNING_SECRET is unset, no signing headers are sent,
// preserving backward compatibility during rollout. Never log the secret or
// signature values.

const encoder = new TextEncoder();

function hex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

let _cachedKey: CryptoKey | null = null;
let _cachedSecret = "";

async function importHmacKey(secret: string): Promise<CryptoKey> {
  if (_cachedKey && _cachedSecret === secret) return _cachedKey;
  _cachedKey = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  _cachedSecret = secret;
  return _cachedKey;
}

export interface ModalSigningHeaders {
  "X-AVTR-Timestamp": string;
  "X-AVTR-Request-Id": string;
  "X-AVTR-Body-SHA256": string;
  "X-AVTR-Signature": string;
}

export interface SignModalRequestResult {
  headers: Record<string, string>;
  requestId: string;
}

function normalizeBody(body: string | Uint8Array | null): Uint8Array {
  if (body === null || body === undefined) return new Uint8Array(0);
  if (typeof body === "string") return encoder.encode(body);
  return body;
}

function canonicalizeUrl(url: string): string {
  // Normalize so the Worker (env-var URL without trailing slash) and Modal
  // (str(request.url) with trailing slash) produce the same canonical string.
  return new URL(url).href;
}

export async function signModalRequest(
  secret: string | undefined,
  method: string,
  url: string,
  body: string | Uint8Array | null,
  requestId?: string,
  timestamp?: number,
): Promise<SignModalRequestResult> {
  const reqId = requestId ?? crypto.randomUUID();
  if (!secret) {
    return { headers: {}, requestId: reqId };
  }

  const ts = timestamp ?? Math.floor(Date.now() / 1000);
  const bodyBytes = normalizeBody(body);
  const bodyHash = await crypto.subtle.digest("SHA-256", bodyBytes).then(hex);

  const canonicalUrl = canonicalizeUrl(url);
  const base = `${method.toUpperCase()}\n${canonicalUrl}\n${ts}\n${reqId}\n${bodyHash}`;
  const key = await importHmacKey(secret);
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(base)).then(hex);

  return {
    headers: {
      "X-AVTR-Timestamp": String(ts),
      "X-AVTR-Request-Id": reqId,
      "X-AVTR-Body-SHA256": bodyHash,
      "X-AVTR-Signature": signature,
    },
    requestId: reqId,
  };
}

export interface VerifyModalSignatureResult {
  valid: boolean;
  reason?: string;
}

export async function verifyModalSignature(
  secret: string,
  method: string,
  url: string,
  body: string | Uint8Array | null,
  headers: Record<string, string>,
): Promise<VerifyModalSignatureResult> {
  const timestampHdr = headers["x-avtr-timestamp"] ?? headers["X-AVTR-Timestamp"];
  const requestIdHdr = headers["x-avtr-request-id"] ?? headers["X-AVTR-Request-Id"];
  const bodyHashHdr = headers["x-avtr-body-sha256"] ?? headers["X-AVTR-Body-SHA256"];
  const signatureHdr = headers["x-avtr-signature"] ?? headers["X-AVTR-Signature"];

  if (!timestampHdr || !requestIdHdr || !bodyHashHdr || !signatureHdr) {
    return { valid: false, reason: "missing_headers" };
  }

  let ts: number;
  try {
    ts = parseInt(timestampHdr, 10);
    if (!Number.isFinite(ts)) throw new Error("invalid");
  } catch {
    return { valid: false, reason: "invalid_timestamp" };
  }

  const now = Math.floor(Date.now() / 1000);
  if (Math.abs(now - ts) > 300) {
    return { valid: false, reason: "stale_timestamp" };
  }

  const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
  if (!uuidRegex.test(requestIdHdr)) {
    return { valid: false, reason: "invalid_request_id" };
  }

  const bodyBytes = normalizeBody(body);
  const expectedBodyHash = await crypto.subtle.digest("SHA-256", bodyBytes).then(hex);
  if (!constantTimeEqual(expectedBodyHash, bodyHashHdr)) {
    return { valid: false, reason: "body_hash_mismatch" };
  }

  const canonicalUrl = canonicalizeUrl(url);
  const base = `${method.toUpperCase()}\n${canonicalUrl}\n${ts}\n${requestIdHdr}\n${bodyHashHdr}`;
  const key = await importHmacKey(secret);
  const expectedSignature = await crypto.subtle.sign("HMAC", key, encoder.encode(base)).then(hex);
  if (!constantTimeEqual(expectedSignature, signatureHdr)) {
    return { valid: false, reason: "signature_mismatch" };
  }

  return { valid: true };
}

function constantTimeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}
