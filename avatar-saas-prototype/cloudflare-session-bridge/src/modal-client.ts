import type {
  Env,
  ModalActionRequest,
  ModalStartResponse,
  ModalRenderResponse,
  ModalRenderAudioResponse,
  ModalRenderIdleResponse,
  ModalRenderIdleChunkResponse,
  ModalEndResponse,
  ModalErrorResponse,
  ModalStreamCallbacks,
  ModalStreamEvent,
  ModalTimelineCallbacks,
  ModalTimelineFrameEvent,
  ModalTimelineStartEvent,
  ModalTimelineEndEvent,
  ModalTimelineEnqueueResponse,
  ModalTimelineStopResponse,
  ModalTimelineFmp4Callbacks,
  ModalTimelineFmp4InitEvent,
  ModalTimelineFmp4MediaEvent,
  ModalTimelineFmp4AvCallbacks,
  ModalTimelineFmp4AvMediaEvent,
  ModalSessionStartRequest,
  ModalSessionStartResponse,
  ModalSessionEndResponse,
  ModalSessionStatusResponse,
  ModalSessionNotActiveResponse,
  ModalPushAudioRequest,
  ModalPushAudioResponse,
} from "./types";
import { signModalRequest } from "./modal-request-signing";

const TIMEOUT: Record<string, number> = {
  health: 30_000,
  start: 120_000,
  render_frame: 60_000,
  render_audio: 300_000,
  render_idle: 300_000,
  render_idle_chunk: 30_000,
  render_audio_stream: 300_000,
  end: 5_000,
  timeline_enqueue: 30_000,
  timeline_stop: 10_000,
};

function getLifecycleUrl(env: Env): string {
  if (env.ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS === "true") {
    const target = env.OMNIVOICE_SAME_GPU_PARALLEL_TARGET ?? "l4";
    if (target === "l40s" && env.MODAL_PARALLEL_L40S_LIFECYCLE_URL) {
      return env.MODAL_PARALLEL_L40S_LIFECYCLE_URL;
    }
    if (target === "a10g" && env.MODAL_PARALLEL_A10G_LIFECYCLE_URL) {
      return env.MODAL_PARALLEL_A10G_LIFECYCLE_URL;
    }
    if (env.MODAL_PARALLEL_LIFECYCLE_URL) {
      return env.MODAL_PARALLEL_LIFECYCLE_URL;
    }
  }
  return env.MODAL_LIFECYCLE_URL;
}

export function getParallelTarget(env: Env): string {
  return env.ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS === "true"
    ? (env.OMNIVOICE_SAME_GPU_PARALLEL_TARGET ?? "l4")
    : "none";
}

export class ModalClientError extends Error {
  constructor(
    message: string,
    public readonly cause: unknown,
    public readonly statusCode?: number,
  ) {
    super(message);
    this.name = "ModalClientError";
  }
}

export interface ModalCallMeta {
  responseJsonParseMs: number;
}

export interface ModalCallResult<T> {
  data: T;
  meta: ModalCallMeta;
}

async function buildSignedHeaders(
  env: Env,
  method: string,
  url: string,
  body: string | null,
): Promise<Record<string, string>> {
  const result = await signModalRequest(
    env.AVTR_MODAL_REQUEST_SIGNING_SECRET,
    method,
    url,
    body,
  );
  return result.headers;
}

export async function callModal<T = unknown>(
  env: Env,
  action: ModalActionRequest["action"],
  payload?: Record<string, unknown>,
  timeoutMs?: number,
): Promise<ModalCallResult<T>> {
  const url = getLifecycleUrl(env);
  if (!url) {
    throw new ModalClientError("MODAL_LIFECYCLE_URL not configured", undefined);
  }

  const deadline = timeoutMs ?? TIMEOUT[action];
  const body: ModalActionRequest = { action, ...payload };
  const bodyJson = JSON.stringify(body);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), deadline);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    const jsonParseStart = Date.now();
    const data = (await response.json()) as T;
    const responseJsonParseMs = Date.now() - jsonParseStart;

    return { data, meta: { responseJsonParseMs } };
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ModalClientError(
        `Modal request timed out after ${deadline}ms`,
        err,
      );
    }
    throw new ModalClientError(
      `Modal request failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function health(env: Env) {
  const r = await callModal<Record<string, unknown>>(env, "health", undefined, TIMEOUT.health);
  return { data: r.data, meta: r.meta };
}

export async function startSession(env: Env, avatarId?: string, bgId?: string) {
  const r = await callModal<ModalStartResponse>(env, "start", {
    avatar_id: avatarId,
    bg_id: bgId,
  }, TIMEOUT.start);
  return { data: r.data, meta: r.meta };
}

export async function renderFrame(env: Env, sessionId?: string) {
  const r = await callModal<ModalRenderResponse>(env, "render_frame", {
    session_id: sessionId,
  }, TIMEOUT.render_frame);
  return { data: r.data, meta: r.meta };
}

export async function renderAudio(
  env: Env,
  sessionId: string,
  audioBase64: string,
  maxDurationS = 60.0,
  jpegQuality = 80,
) {
  const r = await callModal<ModalRenderAudioResponse>(env, "render_audio", {
    session_id: sessionId,
    audio_base64: audioBase64,
    max_duration_s: maxDurationS,
    jpeg_quality: jpegQuality,
  }, TIMEOUT.render_audio);
  return { data: r.data, meta: r.meta };
}

export async function renderIdle(
  env: Env,
  sessionId: string,
  durationS = 3.0,
  jpegQuality = 75,
) {
  const r = await callModal<ModalRenderIdleResponse>(env, "render_idle", {
    session_id: sessionId,
    duration_s: durationS,
    jpeg_quality: jpegQuality,
  }, TIMEOUT.render_idle);
  return { data: r.data, meta: r.meta };
}

export async function renderIdleChunk(
  env: Env,
  sessionId: string,
  durationS = 0.5,
  jpegQuality = 75,
) {
  const r = await callModal<ModalRenderIdleChunkResponse>(env, "render_idle_chunk", {
    session_id: sessionId,
    duration_s: durationS,
    jpeg_quality: jpegQuality,
  }, TIMEOUT.render_idle_chunk);
  return { data: r.data, meta: r.meta };
}

export async function endSession(env: Env, sessionId?: string) {
  const r = await callModal<ModalEndResponse>(env, "end", {
    session_id: sessionId,
  }, TIMEOUT.end);
  return { data: r.data, meta: r.meta };
}

export async function renderAudioStream(
  env: Env,
  sessionId: string,
  audioBase64: string | null,
  maxDurationS: number,
  jpegQuality: number,
  callbacks: ModalStreamCallbacks,
): Promise<void> {
  const url = getLifecycleUrl(env);
  if (!url) {
    throw new ModalClientError("MODAL_LIFECYCLE_URL not configured", undefined);
  }

  const deadline = TIMEOUT.render_audio_stream;
  const body: ModalActionRequest = {
    action: "render_audio_stream",
    session_id: sessionId,
    ...(audioBase64 ? { audio_base64: audioBase64 } : {}),
    max_duration_s: maxDurationS,
    jpeg_quality: jpegQuality,
  };
  const bodyJson = JSON.stringify(body);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), deadline);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal stream returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    if (!response.body) {
      throw new ModalClientError("Modal stream response has no body", undefined);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        for (const line of trimmed.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          let raw: Record<string, unknown>;
          try {
            raw = JSON.parse(jsonStr) as Record<string, unknown>;
          } catch {
            console.warn("[modal-client] Failed to parse SSE event:", jsonStr.slice(0, 200));
            continue;
          }
          if (raw.type === "error") {
            throw new ModalClientError(
              `Modal stream error: ${raw.error ?? "unknown"}`,
              undefined,
            );
          }
          try {
            const event = raw as unknown as ModalStreamEvent;
            switch (event.type) {
              case "start":
                callbacks.onStart(event);
                break;
              case "chunk":
                callbacks.onChunk(event);
                break;
              case "end":
                callbacks.onEnd(event);
                break;
            }
          } catch (cbErr) {
            if (cbErr instanceof ModalClientError) throw cbErr;
            throw new ModalClientError(
              `Modal stream callback failed [${String(raw.type)}]: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`,
              cbErr,
            );
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ModalClientError(
        `Modal stream timed out after ${deadline}ms`,
        err,
      );
    }
    throw new ModalClientError(
      `Modal stream failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function startModalTimelineStream(
  env: Env,
  sessionId: string,
  callbacks: ModalTimelineCallbacks,
  abortSignal?: AbortSignal,
): Promise<void> {
  const url = getLifecycleUrl(env);
  if (!url) {
    throw new ModalClientError("MODAL_LIFECYCLE_URL not configured", undefined);
  }

  const body: ModalActionRequest = {
    action: "timeline_stream",
    session_id: sessionId,
  };
  const bodyJson = JSON.stringify(body);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const linkedAbort = () => controller.abort();
  abortSignal?.addEventListener("abort", linkedAbort);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal timeline stream returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    if (!response.body) {
      throw new ModalClientError("Modal timeline stream response has no body", undefined);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        for (const line of trimmed.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          let raw: Record<string, unknown>;
          try {
            raw = JSON.parse(jsonStr) as Record<string, unknown>;
          } catch {
            console.warn("[modal-client] Failed to parse timeline SSE event:", jsonStr.slice(0, 200));
            continue;
          }
          if (raw.type === "error") {
            throw new ModalClientError(
              `Modal timeline stream error: ${raw.error ?? "unknown"}`,
              undefined,
            );
          }
          try {
            switch (raw.type) {
              case "start":
                callbacks.onStart(raw as unknown as ModalTimelineStartEvent);
                break;
              case "frame_chunk":
                callbacks.onFrame(raw as unknown as ModalTimelineFrameEvent);
                break;
              case "end":
                callbacks.onEnd(raw as unknown as ModalTimelineEndEvent);
                break;
            }
          } catch (cbErr) {
            if (cbErr instanceof ModalClientError) throw cbErr;
            throw new ModalClientError(
              `Modal timeline callback failed [${String(raw.type)}]: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`,
              cbErr,
            );
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    throw new ModalClientError(
      `Modal timeline stream failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    abortSignal?.removeEventListener("abort", linkedAbort);
  }
}

export async function startModalTimelineFmp4Stream(
  env: Env,
  sessionId: string,
  callbacks: ModalTimelineFmp4Callbacks,
  abortSignal?: AbortSignal,
): Promise<void> {
  const url = getLifecycleUrl(env);
  if (!url) {
    throw new ModalClientError("MODAL_LIFECYCLE_URL not configured", undefined);
  }

  const body: ModalActionRequest = {
    action: "timeline_stream_fmp4_video",
    session_id: sessionId,
  };
  const bodyJson = JSON.stringify(body);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const linkedAbort = () => controller.abort();
  abortSignal?.addEventListener("abort", linkedAbort);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal fMP4 stream returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    if (!response.body) {
      throw new ModalClientError("Modal fMP4 stream response has no body", undefined);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        for (const line of trimmed.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          let raw: Record<string, unknown>;
          try {
            raw = JSON.parse(jsonStr) as Record<string, unknown>;
          } catch {
            console.warn("[modal-client] Failed to parse fMP4 SSE event:", jsonStr.slice(0, 200));
            continue;
          }
          if (raw.type === "error") {
            throw new ModalClientError(
              `Modal fMP4 stream error: ${raw.error ?? "unknown"}`,
              undefined,
            );
          }
          try {
            switch (raw.type) {
              case "fmp4_init":
                callbacks.onInit(raw as unknown as ModalTimelineFmp4InitEvent);
                break;
              case "fmp4_media":
                callbacks.onMedia(raw as unknown as ModalTimelineFmp4MediaEvent);
                break;
              case "end":
                callbacks.onEnd(raw as unknown as ModalTimelineEndEvent);
                break;
            }
          } catch (cbErr) {
            if (cbErr instanceof ModalClientError) throw cbErr;
            throw new ModalClientError(
              `fMP4 callback failed [${String(raw.type)}]: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`,
              cbErr,
            );
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    throw new ModalClientError(
      `Modal fMP4 stream failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    abortSignal?.removeEventListener("abort", linkedAbort);
  }
}

export async function startModalTimelineFmp4AvStream(
  env: Env,
  sessionId: string,
  callbacks: ModalTimelineFmp4AvCallbacks,
  abortSignal?: AbortSignal,
): Promise<void> {
  const url = getLifecycleUrl(env);
  if (!url) {
    throw new ModalClientError("MODAL_LIFECYCLE_URL not configured", undefined);
  }

  const body: ModalActionRequest = {
    action: "timeline_stream_fmp4_av",
    session_id: sessionId,
  };
  const bodyJson = JSON.stringify(body);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const linkedAbort = () => controller.abort();
  abortSignal?.addEventListener("abort", linkedAbort);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal fMP4 AV stream returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    if (!response.body) {
      throw new ModalClientError("Modal fMP4 AV stream response has no body", undefined);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        for (const line of trimmed.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          const jsonStr = line.slice(6);
          let raw: Record<string, unknown>;
          try {
            raw = JSON.parse(jsonStr) as Record<string, unknown>;
          } catch {
            console.warn("[modal-client] Failed to parse fMP4 AV SSE event:", jsonStr.slice(0, 200));
            continue;
          }
          if (raw.type === "error") {
            throw new ModalClientError(
              `Modal fMP4 AV stream error: ${raw.error ?? "unknown"}`,
              undefined,
            );
          }
          try {
            switch (raw.type) {
              case "fmp4_init":
                callbacks.onInit(raw as unknown as ModalTimelineFmp4InitEvent);
                break;
              case "fmp4_media":
                callbacks.onMedia(raw as unknown as ModalTimelineFmp4AvMediaEvent);
                break;
              case "end":
                callbacks.onEnd(raw as unknown as ModalTimelineEndEvent);
                break;
            }
          } catch (cbErr) {
            if (cbErr instanceof ModalClientError) throw cbErr;
            throw new ModalClientError(
              `fMP4 AV callback failed [${String(raw.type)} seq=${String(raw.sequence ?? "?")} byteLen=${String(raw.byteLength ?? "?")}]: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`,
              cbErr,
            );
          }
        }
      }
    }
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      return;
    }
    throw new ModalClientError(
      `Modal fMP4 AV stream failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    abortSignal?.removeEventListener("abort", linkedAbort);
  }
}

async function callControlEndpoint<T = unknown>(
  env: Env,
  action: "enqueue" | "stop",
  payload: Record<string, unknown>,
  timeoutMs: number,
): Promise<ModalCallResult<T>> {
  const url = env.MODAL_TIMELINE_CONTROL_URL;
  if (!url) {
    throw new ModalClientError(
      "MODAL_TIMELINE_CONTROL_URL not configured — cannot reach CPU control endpoint",
      undefined,
    );
  }

  const bodyJson = JSON.stringify({ action, ...payload });
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Control endpoint returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    const jsonParseStart = Date.now();
    const data = (await response.json()) as T;
    const responseJsonParseMs = Date.now() - jsonParseStart;

    return { data, meta: { responseJsonParseMs } };
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ModalClientError(
        `Control endpoint timed out after ${timeoutMs}ms`,
        err,
      );
    }
    throw new ModalClientError(
      `Control endpoint failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function enqueueTimelineSpeech(
  env: Env,
  sessionId: string,
  turnId: number,
  audioBase64: string,
  audioDurationMs: number,
  options?: { useControlEndpoint?: boolean },
): Promise<ModalCallResult<ModalTimelineEnqueueResponse>> {
  if (options?.useControlEndpoint) {
    return callControlEndpoint<ModalTimelineEnqueueResponse>(env, "enqueue", {
      session_id: sessionId,
      turnId,
      audio_base64: audioBase64,
      audio_duration_ms: audioDurationMs,
    }, 30_000);
  }
  return callModal<ModalTimelineEnqueueResponse>(env, "timeline_enqueue", {
    session_id: sessionId,
    turnId,
    audio_base64: audioBase64,
    audio_duration_ms: audioDurationMs,
  }, 30_000);
}

export async function stopModalTimeline(
  env: Env,
  sessionId: string,
  options?: { useControlEndpoint?: boolean },
): Promise<ModalCallResult<ModalTimelineStopResponse>> {
  if (options?.useControlEndpoint) {
    return callControlEndpoint<ModalTimelineStopResponse>(env, "stop", {
      session_id: sessionId,
    }, 10_000);
  }
  return callModal<ModalTimelineStopResponse>(env, "timeline_stop", {
    session_id: sessionId,
  }, 10_000);
}

// ── Phase 4C: New Modal session-lifecycle contract (5 web endpoints) ──
// These functions call the per-method @modal.fastapi_endpoint() URLs
// exposed by the Modal class. Each method has its own auto-generated URL
// (label-based). See PHASE_4C_FMP4_STREAMING_CONTRACT.md for the contract.

const SESSION_TIMEOUTS: Record<string, number> = {
  start: 120_000,
  end: 30_000,
  status: 30_000,
  push_audio: 10_000,
  push_audio_batch: 30_000,
};

export function isLegacyModalEnabled(env: Env): boolean {
  return env.ENABLE_LEGACY_MODAL_PROTOCOL === "true";
}

export function getActiveModalBaseUrl(env: Env): string {
  // Priority: new 4C session stream URL > legacy URL (only if flag is on).
  // Bridge considers the 4C path active only when the SSE stream URL is
  // configured. If it is not, the legacy path may be used if enabled.
  const newUrl = env.MODAL_SESSION_STREAM_FMP4_AV_URL ?? "";
  if (newUrl) return newUrl;
  if (isLegacyModalEnabled(env)) return env.MODAL_LIFECYCLE_URL ?? "";
  return "";
}

async function callSessionEndpoint<T = unknown>(
  env: Env,
  url: string,
  methodLabel: string,
  payload: Record<string, unknown>,
  timeoutMs?: number,
): Promise<ModalCallResult<T>> {
  if (!url) {
    throw new ModalClientError(
      `Modal session endpoint URL not configured for ${methodLabel}`,
      undefined,
    );
  }
  const deadline = timeoutMs ?? SESSION_TIMEOUTS[methodLabel] ?? 60_000;
  const bodyJson = JSON.stringify(payload);
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), deadline);
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal session endpoint ${methodLabel} returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }
    const jsonParseStart = Date.now();
    const data = (await response.json()) as T;
    const responseJsonParseMs = Date.now() - jsonParseStart;
    return { data, meta: { responseJsonParseMs } };
  } catch (err) {
    if (err instanceof ModalClientError) throw err;
    if (err instanceof DOMException && err.name === "AbortError") {
      throw new ModalClientError(
        `Modal session request ${methodLabel} timed out after ${deadline}ms`,
        err,
      );
    }
    throw new ModalClientError(
      `Modal session request ${methodLabel} failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    clearTimeout(timer);
  }
}

export async function startModalSession(
  env: Env,
  request: ModalSessionStartRequest,
): Promise<ModalCallResult<ModalSessionStartResponse | { error: string }>> {
  return callSessionEndpoint<ModalSessionStartResponse | { error: string }>(
    env,
    env.MODAL_SESSION_START_URL ?? "",
    "start",
    request as unknown as Record<string, unknown>,
  );
}

export async function endModalSession(
  env: Env,
  sessionId: string,
  opts?: { timeoutMs?: number },
): Promise<ModalCallResult<ModalSessionEndResponse | ModalSessionNotActiveResponse>> {
  return callSessionEndpoint<ModalSessionEndResponse | ModalSessionNotActiveResponse>(
    env,
    env.MODAL_SESSION_END_URL ?? "",
    "end",
    { session_id: sessionId },
    opts?.timeoutMs,
  );
}

/**
 * Best-effort Modal /end for incident-response / cost-safety cleanup paths.
 *
 * Wraps endModalSession with a SHORT timeout (default 3000ms) and swallows
 * all errors. Use this in force-clear, keepalive-abort, cancelled-start
 * cleanup, and orphan-cleanup paths where Worker-side containment MUST
 * complete quickly even if Modal hangs, 429s, or is unreachable.
 *
 * Do NOT use this for normal user-initiated session.end — that path
 * intentionally waits for cleaner Modal teardown before releasing capacity.
 */
export async function endModalSessionBestEffort(
  env: Env,
  sessionId: string,
  timeoutMs = 3_000,
): Promise<void> {
  try {
    await endModalSession(env, sessionId, { timeoutMs });
  } catch {
    // best-effort — Worker-side containment proceeds regardless
  }
}

export async function getModalSessionStatus(
  env: Env,
  sessionId: string,
  opts?: { timeoutMs?: number },
): Promise<ModalCallResult<ModalSessionStatusResponse | ModalSessionNotActiveResponse>> {
  return callSessionEndpoint<ModalSessionStatusResponse | ModalSessionNotActiveResponse>(
    env,
    env.MODAL_SESSION_STATUS_URL ?? "",
    "status",
    { session_id: sessionId },
    opts?.timeoutMs,
  );
}

export async function pushModalAudio(
  env: Env,
  request: ModalPushAudioRequest,
): Promise<ModalCallResult<ModalPushAudioResponse | { error: string }>> {
  return callSessionEndpoint<ModalPushAudioResponse | { error: string }>(
    env,
    env.MODAL_SESSION_PUSH_AUDIO_URL ?? "",
    "push_audio",
    request as unknown as Record<string, unknown>,
  );
}

export async function pushModalAudioBatch(
  env: Env,
  request: Record<string, unknown>,
): Promise<ModalCallResult<Record<string, unknown>>> {
  return callSessionEndpoint<Record<string, unknown>>(
    env,
    env.MODAL_SESSION_PUSH_AUDIO_BATCH_URL ?? "",
    "push_audio_batch",
    request,
  );
}

// ── Phase 4C: SSE stream consumer (fMP4 init + media events) ──

export interface Fmp4RelayStats {
  maxQueueDepth: number;
  receivedMediaCount: number;
  forwardedMediaCount: number;
  overflowCount: number;
  queueCapacity: number;
  adaptiveFastForwardCount: number;
  minForwardDelayMs: number;
  maxForwardDelayMs: number;
  backpressureWaitCount: number;
  backpressureWaitMs: number;
  highWater: number;
  lowWater: number;
}

export interface ModalSessionFmp4Callbacks {
  onInit: (event: ModalTimelineFmp4InitEvent) => void;
  onMedia: (event: ModalTimelineFmp4AvMediaEvent) => void;
  onEnd: (event: ModalTimelineEndEvent) => void;
  onStats?: (stats: Fmp4RelayStats) => void;
  onOverflow?: (info: {
    droppedCount: number;
    queueLen: number;
    firstDroppedSeq?: number;
    lastDroppedSeq?: number;
    idleDropped: number;
    mode: "idle_preferred" | "oldest_batch";
  }) => void;
}

function getAdaptiveDelay(queueLen: number, baseMs: number): number {
  if (queueLen > 300) return 10;
  if (queueLen > 160) return 25;
  if (queueLen > 80) return 50;
  if (queueLen > 20) return 100;
  return baseMs;
}

export async function startModalSessionFmp4Stream(
  env: Env,
  sessionId: string,
  callbacks: ModalSessionFmp4Callbacks,
  abortSignal?: AbortSignal,
  paceMediaMs?: number,
): Promise<void> {
  const url = env.MODAL_SESSION_STREAM_FMP4_AV_URL;
  if (!url) {
    throw new ModalClientError(
      "MODAL_SESSION_STREAM_FMP4_AV_URL not configured",
      undefined,
    );
  }

  const fetchController = new AbortController();
  const forwarderController = new AbortController();
  const linkedAbort = () => {
    fetchController.abort();
    forwarderController.abort();
  };
  abortSignal?.addEventListener("abort", linkedAbort);

  const MAX_QUEUE = env.FMP4_RELAY_MAX_QUEUE
    ? parseInt(env.FMP4_RELAY_MAX_QUEUE, 10)
    : 500;

  const mediaQueue: ModalTimelineFmp4AvMediaEvent[] = [];
  let readerDone = false;
  let cancelled = false;
  let pendingEnd: ModalTimelineEndEvent | null = null;

  const basePaceMs = paceMediaMs && paceMediaMs > 0 ? paceMediaMs : 200;

  const stats: Fmp4RelayStats = {
    maxQueueDepth: 0,
    receivedMediaCount: 0,
    forwardedMediaCount: 0,
    overflowCount: 0,
    queueCapacity: MAX_QUEUE,
    adaptiveFastForwardCount: 0,
    minForwardDelayMs: basePaceMs,
    maxForwardDelayMs: basePaceMs,
    backpressureWaitCount: 0,
    backpressureWaitMs: 0,
    highWater: 0,
    lowWater: 0,
  };

  const pacedForward = paceMediaMs && paceMediaMs > 0;

  const forwarderPromise = pacedForward
    ? (async () => {
        let lastSend = 0;
        while (!cancelled && (!readerDone || mediaQueue.length > 0)) {
          if (mediaQueue.length === 0) {
            try {
              await scheduler.wait(5, { signal: forwarderController.signal });
            } catch (e) {
              if (e instanceof DOMException && e.name === "AbortError") break;
              throw e;
            }
            continue;
          }
          if (cancelled) break;

          const delay = getAdaptiveDelay(mediaQueue.length, basePaceMs);
          if (delay < basePaceMs) stats.adaptiveFastForwardCount++;
          if (delay < stats.minForwardDelayMs) stats.minForwardDelayMs = delay;
          if (delay > stats.maxForwardDelayMs) stats.maxForwardDelayMs = delay;

          const now = Date.now();
          if (lastSend > 0) {
            const elapsed = now - lastSend;
            if (elapsed < delay) {
              try {
                await scheduler.wait(delay - elapsed, { signal: forwarderController.signal });
              } catch (e) {
                if (e instanceof DOMException && e.name === "AbortError") break;
                throw e;
              }
            }
          }
          const event = mediaQueue.shift()!;
          callbacks.onMedia(event);
          stats.forwardedMediaCount++;
          lastSend = Date.now();
        }
      })()
    : Promise.resolve();

  const cancelForwarder = (): void => {
    cancelled = true;
    mediaQueue.length = 0;
    readerDone = true;
    forwarderController.abort();
  };

  const bodyJson = JSON.stringify({ session_id: sessionId });
  const signed = await buildSignedHeaders(env, "POST", url, bodyJson);

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...signed },
      body: bodyJson,
      signal: fetchController.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new ModalClientError(
        `Modal session fMP4 stream returned ${response.status}: ${text}`,
        undefined,
        response.status,
      );
    }

    if (!response.body) {
      throw new ModalClientError("Modal session fMP4 stream response has no body", undefined);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    readLoop: while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop()!;

      for (const part of parts) {
        const trimmed = part.trim();
        if (!trimmed) continue;

        if (trimmed.startsWith(":")) continue;

        let eventName: string | null = null;
        let dataStr = "";
        for (const line of trimmed.split("\n")) {
          if (line.startsWith("event: ")) {
            eventName = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            dataStr += line.slice(6);
          }
        }
        if (!eventName) continue;

        let raw: Record<string, unknown>;
        try {
          raw = JSON.parse(dataStr) as Record<string, unknown>;
        } catch {
          console.warn("[modal-client] Failed to parse session fMP4 SSE event:", dataStr.slice(0, 200));
          continue;
        }

        try {
          if (eventName === "fmp4_init") {
            callbacks.onInit({ type: "fmp4_init", ...(raw as object) } as ModalTimelineFmp4InitEvent);
          } else if (eventName === "fmp4_media") {
            const evt = { type: "fmp4_media", ...(raw as object) } as ModalTimelineFmp4AvMediaEvent;
            stats.receivedMediaCount++;
            if (mediaQueue.length > stats.maxQueueDepth) {
              stats.maxQueueDepth = mediaQueue.length;
            }

            if (pacedForward) {
              if (mediaQueue.length >= MAX_QUEUE) {
                stats.overflowCount++;
                const dropTarget = Math.min(50, Math.max(1, Math.ceil(MAX_QUEUE * 0.1)));
                let droppedCount = 0;
                let idleDropped = 0;
                let firstDroppedSeq: number | undefined;
                let lastDroppedSeq: number | undefined;
                let mode: "idle_preferred" | "oldest_batch" = "idle_preferred";

                // First pass: drop oldest idle segments
                while (mediaQueue.length > 0 && droppedCount < dropTarget && mediaQueue[0].mode === "idle") {
                  const dropped = mediaQueue.shift()!;
                  droppedCount++;
                  idleDropped++;
                  if (firstDroppedSeq === undefined) firstDroppedSeq = dropped.sequence;
                  lastDroppedSeq = dropped.sequence;
                }

                // If no idle segments were dropped, drop the oldest batch regardless of mode
                if (droppedCount === 0) {
                  mode = "oldest_batch";
                  for (let i = 0; i < dropTarget && mediaQueue.length > 0; i++) {
                    const dropped = mediaQueue.shift()!;
                    droppedCount++;
                    if (firstDroppedSeq === undefined) firstDroppedSeq = dropped.sequence;
                    lastDroppedSeq = dropped.sequence;
                  }
                }

                if (callbacks.onOverflow && droppedCount > 0) {
                  try {
                    callbacks.onOverflow({
                      droppedCount,
                      queueLen: mediaQueue.length,
                      firstDroppedSeq,
                      lastDroppedSeq,
                      idleDropped,
                      mode,
                    });
                  } catch {
                    // best-effort telemetry
                  }
                }
              }
              mediaQueue.push(evt);

              if (mediaQueue.length % 10 === 0) {
                await scheduler.wait(0);
              }
            } else {
              callbacks.onMedia(evt);
              stats.forwardedMediaCount++;
            }
          } else if (eventName === "end") {
            pendingEnd = { type: "end", ...(raw as object) } as ModalTimelineEndEvent;
            readerDone = true;
            break readLoop;
          } else if (eventName === "error") {
            throw new ModalClientError(
              `Modal session fMP4 stream error: ${raw.error ?? "unknown"}`,
              undefined,
            );
          }
        } catch (cbErr) {
          if (cbErr instanceof ModalClientError) throw cbErr;
          throw new ModalClientError(
            `session fMP4 callback failed [${eventName}]: ${cbErr instanceof Error ? cbErr.message : String(cbErr)}`,
            cbErr,
          );
        }
      }
    }
  } catch (err) {
    if (err instanceof ModalClientError) {
      cancelForwarder();
      await forwarderPromise;
      throw err;
    }
    if (err instanceof DOMException && err.name === "AbortError") {
      cancelForwarder();
      await forwarderPromise;
      return;
    }
    cancelForwarder();
    await forwarderPromise;
    throw new ModalClientError(
      `Modal session fMP4 stream failed: ${err instanceof Error ? err.message : String(err)}`,
      err,
    );
  } finally {
    readerDone = true;
    if (pacedForward) {
      await forwarderPromise;
    }
    if (callbacks.onStats) {
      try {
        callbacks.onStats(stats);
      } catch {
        // best-effort
      }
    }
    abortSignal?.removeEventListener("abort", linkedAbort);
  }

  if (pendingEnd && !cancelled) {
    callbacks.onEnd(pendingEnd);
  }
}
