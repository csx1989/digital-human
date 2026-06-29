import { describe, it, expect } from "vitest";
import {
  signModalRequest,
  verifyModalSignature,
} from "../src/modal-request-signing";

const TEST_SECRET = "test-secret-32-bytes-long-for-hmac-sha256";

function lowerCaseKeys(headers: Record<string, string>): Record<string, string> {
  return Object.fromEntries(
    Object.entries(headers).map(([k, v]) => [k.toLowerCase(), v]),
  );
}

describe("modal-request-signing", () => {
  it("signs and verifies a valid request", async () => {
    const body = JSON.stringify({ session_id: "sess-123", avatar_id: "arnold" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
    );

    expect(signed.headers["X-AVTR-Timestamp"]).toMatch(/^\d+$/);
    expect(signed.headers["X-AVTR-Request-Id"]).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
    expect(signed.headers["X-AVTR-Body-SHA256"]).toHaveLength(64);
    expect(signed.headers["X-AVTR-Signature"]).toHaveLength(64);

    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
      signed.headers,
    );
    expect(verified.valid).toBe(true);
  });

  it("verifies with lowercase header names", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
    );
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
      lowerCaseKeys(signed.headers),
    );
    expect(verified.valid).toBe(true);
  });

  it("rejects missing headers", async () => {
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      "{}",
      {},
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("missing_headers");
  });

  it("rejects bad signature", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
    );
    const sig = signed.headers["X-AVTR-Signature"];
    const badHeaders = {
      ...signed.headers,
      "X-AVTR-Signature": sig.slice(0, -1) + (sig.endsWith("0") ? "1" : "0"),
    };
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
      badHeaders,
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("signature_mismatch");
  });

  it("rejects stale timestamp", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
      undefined,
      1000,
    );
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
      signed.headers,
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("stale_timestamp");
  });

  it("rejects tampered body", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
    );
    const tamperedBody = JSON.stringify({ session_id: "sess-456" });
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      tamperedBody,
      signed.headers,
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("body_hash_mismatch");
  });

  it("handles empty body", async () => {
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      null,
    );
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      null,
      signed.headers,
    );
    expect(verified.valid).toBe(true);
  });

  it("returns empty headers when secret is unset", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      undefined,
      "POST",
      "https://modal.run/start",
      body,
    );
    expect(Object.keys(signed.headers).length).toBe(0);
  });

  it("does not leak secret or signature in verify result", async () => {
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      "{}",
      {},
    );
    expect(verified.reason).not.toContain(TEST_SECRET);
  });

  it("binds signature to URL", async () => {
    const body = JSON.stringify({ session_id: "sess-123" });
    const signed = await signModalRequest(
      TEST_SECRET,
      "POST",
      "https://modal.run/start",
      body,
    );
    const verified = await verifyModalSignature(
      TEST_SECRET,
      "POST",
      "https://modal.run/other",
      body,
      signed.headers,
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("signature_mismatch");
  });
});

// ── Cross-language compatibility vector ─────────────────────────────────────
// Fixed values prove TS signModalRequest() and Python sign_request() produce
// identical signatures. Vector generated with Python hmac/hashlib, then
// verified to match TS output. If either implementation changes, these tests
// break, catching canonicalization drift.

const XL_SECRET = "avtr-cross-lang-test-secret";
const XL_URL = "https://vallangirakesh--start-session-5q.modal.run";
const XL_URL_SLASH = "https://vallangirakesh--start-session-5q.modal.run/";
const XL_BODY = '{"session_id":"session:cross-lang-001","avatar_id":"arnold"}';
const XL_TS = 1719811200;
const XL_REQ_ID = "12345678-1234-4234-8345-1234567890ab";
const XL_BODY_HASH = "f1f76708e2aef0e1894947069209ce948d8aecf8616aedfb5f10b9d5e01858bd";
const XL_SIGNATURE = "098cab3fb724707633124369239c2bb6e6eb53d8ac00cf5540936291efc644fc";

describe("cross-language compatibility (TS sign → Python verify)", () => {
  it("produces the exact expected body hash and signature", async () => {
    const signed = await signModalRequest(
      XL_SECRET, "POST", XL_URL, XL_BODY, XL_REQ_ID, XL_TS,
    );
    expect(signed.headers["X-AVTR-Body-SHA256"]).toBe(XL_BODY_HASH);
    expect(signed.headers["X-AVTR-Signature"]).toBe(XL_SIGNATURE);
  });

  it("canonicalizes URL (trailing slash) to match Python str(request.url)", async () => {
    const noSlash = await signModalRequest(
      XL_SECRET, "POST", XL_URL, XL_BODY, XL_REQ_ID, XL_TS,
    );
    const withSlash = await signModalRequest(
      XL_SECRET, "POST", XL_URL_SLASH, XL_BODY, XL_REQ_ID, XL_TS,
    );
    expect(noSlash.headers["X-AVTR-Signature"]).toBe(withSlash.headers["X-AVTR-Signature"]);
  });

  it("verify accepts a validly signed request (dynamic timestamp)", async () => {
    const signed = await signModalRequest(XL_SECRET, "POST", XL_URL, XL_BODY);
    const verified = await verifyModalSignature(
      XL_SECRET, "POST", XL_URL, XL_BODY, signed.headers,
    );
    expect(verified.valid).toBe(true);
  });

  it("rejects tampered body (dynamic timestamp)", async () => {
    const signed = await signModalRequest(XL_SECRET, "POST", XL_URL, XL_BODY);
    const verified = await verifyModalSignature(
      XL_SECRET, "POST", XL_URL,
      '{"session_id":"session:HACKED","avatar_id":"arnold"}',
      signed.headers,
    );
    expect(verified.valid).toBe(false);
    expect(verified.reason).toBe("body_hash_mismatch");
  });
});
