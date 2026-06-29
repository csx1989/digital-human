/**
 * AvtrClient — dependency-free API client for Avatar API v1.
 *
 * Uses built-in `fetch` (Node 18+). No npm dependencies.
 *
 * @example
 *   import { AvtrClient, redactApiKey, redactWsUrl } from "./avtr-client.mjs";
 *   const client = new AvtrClient({ apiKey: process.env.AVTR_API_KEY });
 *   const session = await client.createSession();
 *   console.log("session:", session.session_id);
 *   console.log("ws (redacted):", redactWsUrl(session.stream.ws_url));
 */

// ── Redaction utilities ───────────────────────────────────────────────────

/**
 * Redact an API key for safe display. Shows first 12 chars + "...".
 * @param {string} key
 * @returns {string}
 */
export function redactApiKey(key) {
  if (!key || typeof key !== "string") return "[REDACTED]";
  if (key.length <= 12) return key.slice(0, 4) + "...";
  return key.slice(0, 12) + "...";
}

/**
 * Redact a ws_url by stripping the `?token=` query parameter.
 * Returns a safe display string with only protocol + host + truncated path.
 * @param {string} wsUrl
 * @returns {string}
 */
export function redactWsUrl(wsUrl) {
  if (!wsUrl || typeof wsUrl !== "string") return "[none]";
  try {
    const url = new URL(wsUrl);
    url.searchParams.delete("token");
    const path = url.pathname;
    const shortPath = path.length > 50 ? path.slice(0, 47) + "..." : path;
    return `${url.protocol}//${url.host}${shortPath}`;
  } catch {
    return "[invalid-url]";
  }
}

// ── Error type ────────────────────────────────────────────────────────────

/**
 * Thrown when the API returns a non-success response.
 */
export class AvtrApiError extends Error {
  /**
   * @param {Object} opts
   * @param {number} opts.status - HTTP status code.
   * @param {string} opts.code - Machine-readable error code.
   * @param {string} opts.message - Human-readable message.
   * @param {string} [opts.requestId] - Request correlation ID.
   */
  constructor({ status, code, message, requestId }) {
    super(`[${status}] ${code}: ${message}`);
    this.name = "AvtrApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

// ── Client ─────────────────────────────────────────────────────────────────

/**
 * Avatar API v1 client.
 */
export class AvtrClient {
  /**
   * @param {Object} opts
   * @param {string} opts.apiKey - API key (required).
   * @param {string} [opts.baseUrl] - Base URL. Defaults to production.
   */
  constructor({ apiKey, baseUrl }) {
    if (!apiKey) throw new Error("AvtrClient: apiKey is required");
    this.apiKey = apiKey;
    this.baseUrl = (baseUrl || "https://avatar-session-bridge.vallangirakesh.workers.dev").replace(/\/+$/, "");
  }

  /**
   * Internal request helper. Parses the canonical envelope.
   * @param {string} method
   * @param {string} path
   * @param {Object} [body]
   * @returns {Promise<{data: Object, timings: Object, status: number}>}
   * @throws {AvtrApiError}
   */
  async _request(method, path, body) {
    const url = `${this.baseUrl}${path}`;
    const opts = {
      method,
      headers: { Authorization: `Bearer ${this.apiKey}` },
    };
    if (body !== undefined) {
      opts.headers["Content-Type"] = "application/json";
      opts.body = JSON.stringify(body);
    }

    const res = await fetch(url, opts);
    const text = await res.text();
    let parsed = null;
    try { parsed = text ? JSON.parse(text) : null; } catch {}

    if (!res.ok || !parsed?.success) {
      const errCode = parsed?.error?.code || "unknown_error";
      const errMsg = parsed?.error?.message || `HTTP ${res.status}`;
      throw new AvtrApiError({
        status: res.status,
        code: errCode,
        message: errMsg,
        requestId: parsed?.timings?.request_id,
      });
    }

    return {
      data: parsed.data,
      timings: parsed.timings || {},
      status: res.status,
    };
  }

  /**
   * Create a new avatar session.
   * @param {Object} [opts]
   * @param {string} [opts.avatar_id] - Optional pre-selected avatar.
   * @param {string} [opts.bg_id] - Optional pre-selected background.
   * @returns {Promise<Object>} Session data (session_id, status, stream).
   */
  async createSession(opts = {}) {
    const { data } = await this._request("POST", "/v1/sessions", opts && Object.keys(opts).length > 0 ? opts : null);
    return data;
  }

  /**
   * Get session status.
   * @param {string} sessionId - Session ID (starts with `session:`).
   * @returns {Promise<Object>} Session data (session_id, status, timings).
   */
  async getSessionStatus(sessionId) {
    const { data } = await this._request("GET", `/v1/sessions/${sessionId}/status`);
    return data;
  }

  /**
   * Start rendering via REST. Advanced/control-plane path — see API_QUICKSTART.md.
   *
   * Do NOT call this AND send `session.start` over WebSocket on the same session.
   * For interactive clients, prefer the WebSocket `session.start` message.
   *
   * @param {string} sessionId
   * @param {Object} [opts]
   * @param {string} [opts.avatar_id] - Avatar to render (default: "arnold").
   * @param {string} [opts.bg_id] - Background to use.
   * @returns {Promise<Object>} Session data with status "ready".
   */
  async startSession(sessionId, opts = {}) {
    const { data } = await this._request("POST", `/v1/sessions/${sessionId}/start`, opts && Object.keys(opts).length > 0 ? opts : null);
    return data;
  }

  /**
   * Mint a fresh WebSocket token for an existing session.
   * @param {string} sessionId
   * @returns {Promise<Object>} Token data (ws_url, token_expires_at).
   */
  async refreshWsToken(sessionId) {
    const { data } = await this._request("POST", `/v1/sessions/${sessionId}/ws-token`);
    return data;
  }

  /**
   * End a session and release backend resources. Idempotent.
   * @param {string} sessionId
   * @returns {Promise<Object>} Session data with status "ended".
   */
  async endSession(sessionId) {
    const { data } = await this._request("DELETE", `/v1/sessions/${sessionId}`);
    return data;
  }
}
