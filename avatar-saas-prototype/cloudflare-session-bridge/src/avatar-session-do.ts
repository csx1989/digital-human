import { DurableObject } from "cloudflare:workers";
import type {
  Env,
  ClientMessage,
  ServerMessage,
  SessionState,
  SessionOwner,
  WsAttachment,
  VadDecision,
  VadServiceRpc,
  SmartTurnDecision,
  SmartTurnServiceRpc,
} from "./types";
import {
  startSession,
  renderFrame,
  renderAudio,
  renderAudioStream,
  renderIdle,
  renderIdleChunk,
  endSession,
  enqueueTimelineSpeech,
  stopModalTimeline,
  startModalTimelineStream,
  startModalTimelineFmp4Stream,
  startModalTimelineFmp4AvStream,
  ModalClientError,
  getParallelTarget,
  startModalSession,
  endModalSession,
  endModalSessionBestEffort,
  getModalSessionStatus,
  pushModalAudio,
  pushModalAudioBatch,
  startModalSessionFmp4Stream,
  getActiveModalBaseUrl,
  isLegacyModalEnabled,
  type ModalCallMeta,
} from "./modal-client";
import { captureEvent, captureError } from "./posthog-client";
import { SarvamSttProvider } from "./stt-provider";
import { OpenAiCompatibleProvider, type LlmMessage } from "./llm-provider";
import { BridgeTtsProvider } from "./tts-provider";
import { validateAvatarBgForStart } from "./catalog-provider";
import {
  createSfuSession,
  publisherTracksNew,
  subscriberTracksNew,
  sfuRenegotiate,
  sfuTracksClose,
  cleanupSfuTracks,
  SfuClientError,
} from "./sfu-client";
import {
  resolveTransportDecision,
  isSfuFullyConfigured,
  type TransportPreference,
} from "./transport-decision";
import { createWsToken } from "./ws-token";

const STORAGE_KEY = "session:state";
const STORAGE_OWNER_KEY = "session:owner";
const AUDIO_STATS_INTERVAL = 50;
const MAX_TURN_AUDIO_SECONDS = 25;
const MAX_TURN_AUDIO_BYTES = MAX_TURN_AUDIO_SECONDS * 16000 * 2;
const MAX_HISTORY_MESSAGES = 10;
const STARTING_STALE_MS = 180_000;
const SFU_KEEPALIVE_INTERVAL_MS = 10_000;

const LLM_SYSTEM_PROMPT =
  "You are a friendly, concise AI assistant powering a real-time avatar conversation. " +
  "Keep responses to 1-3 short sentences. Be conversational and direct. Do not use markdown formatting.";

function emptyState(): SessionState {
  return {
    modalSessionId: null,
    active: false,
    avatarId: null,
    bgId: null,
    transport: "websocket_fmp4",
    sfuSessionId: null,
    sfuPublisherMids: { video: null, audio: null },
    sfuSubscriberMids: [],
    sfuPublisherTrackName: null,
    sfuSubscriberSessionId: null,
    sfuKeepaliveActive: false,
    sfuKeepaliveStartedAt: null,
    sfuKeepaliveLastHeartbeatAt: null,
    sfuKeepaliveFailures: 0,
    sfuKeepaliveStopReason: null,
    sfuKeepaliveAborted: false,
    sfuSessionStartedAt: null,
    sfuCapacityAcquired: false,
  };
}

export class AvatarSessionDO extends DurableObject<Env> {
  #state: SessionState = emptyState();
  #sessionId = "unknown";
  #turnOpen = false;
  #speechSegmentActive = false;
  #turnCounter = 0;
  #segmentIndex = 0;
  #audioChunkCount = 0;
  #audioBytesReceived = 0;
  #smartTurnChunkCount = 0;
  #smartTurnAccumulatedBytes = 0;
  #firstAudioChunkCaptured = false;
  #currentTurnStartedAt = 0;
  #suppressedEndCount = 0;
  #suppressedStartCount = 0;
  #smartTurnAnalyzeCount = 0;
  #smartTurnAccumulatedBytesAtLastAnalyze = 0;
  #analysisId = 0;
  #analysisInFlight = false;
  #turnAudioChunks: ArrayBuffer[] = [];
  #turnAudioBytes = 0;
  #trimTelemetryChunkCounter = 0;
  #trimTelemetryTimestamp = 0;
  #trimStarted = false;
  #conversationHistory: LlmMessage[] = [];
  #sessionReadyAt = 0;
  #modalRenderBusy = false;
  #modalRenderBusyResolve: (() => void) | null = null;
  #idleLiveEnabled = false;
  #idleRenderInFlight = false;
  #idleChunkIndex = 0;
  #idleConsecutiveFailures = 0;
  #assistantSpeechActive = false;
  #sessionClosing = false;
  #pendingPlaybackAckTurnId: number | null = null;
  #playbackAckTimeoutMs = 0;
  #speechFirstFrameSentAt = 0;
  #idleFirstFrameSentAt = 0;
  #activeSpeechRenderTurnId: number | null = null;
  #avatarTimelineRunning = false;
  #avatarTimelineMode: "idle" | "speech" | "stopped" = "stopped";
  #avatarSpeechQueue: Array<{
    turnId: number;
    audioBase64: string;
    ttsProvider: string;
    audioBytes: number;
    durationMs: number;
  }> = [];
  #avatarSegmentIndex = 0;
  #avatarTimelineAbort = false;
  #lastAvatarFrameAt = 0;
  #idleConsecutiveFailuresTimeline = 0;
  #idleBackoffMs = 0;
  #avatarTimelineIdleStarted = false;
  #modalTimelineStreamActive = false;
  #modalTimelineAbort = false;
  #modalTimelineAbortController: AbortController | null = null;
  #modalTimelineIdleStarted = false;
  #modalTimelineFrameCount = 0;
  #modalFmp4StreamActive = false;
  #modalFmp4SegmentCount = 0;
  #fmp4EnqueueRequestedAt: Map<number, number> = new Map();
  #fmp4FirstSpeechSegmentSentAt: Map<number, number> = new Map();
  #fmp4StreamSummaryCaptured = false;
  #fmp4StreamEndResolve: (() => void) | null = null;
  #fmp4LastStreamMode: string | null = null;
  #fmp4LastStreamSequence: number | null = null;
  #avStreamMediaReceived = 0;
  #avStreamMediaSendAttempted = 0;
  #avStreamMediaSendSucceeded = 0;
  #avStreamMediaSendFailed = 0;
  #avStreamLastSequenceReceived: number | null = null;
  #avStreamLastSequenceSent: number | null = null;
  #avStreamLastMediaSentAt = 0;
  #avStreamMaxSendGapMs = 0;
  #avStreamLastSendGapMs = 0;

  constructor(ctx: DurableObjectState, env: Env) {
    super(ctx, env);
  }

  // ── Stage 3B: SFU Modal keepalive via DO alarm ───────────────────
  //
  // Modal's scaledown_window=30 recycles containers after 30s of no HTTP
  // requests, killing the background WebRTC publisher thread. This alarm
  // calls getModalSessionStatus() every 10s to keep the container alive
  // for the duration of an SFU session.
  //
  // The alarm is self-correcting: if state shows the session is no longer
  // SFU-active (ended, cancelled, transport changed), the alarm deletes
  // itself. This handles DO eviction/hibernation gracefully — the alarm
  // persists in storage and resumes on wake.

  async alarm(): Promise<void> {
    await this.#loadState();

    const isSfu = this.#state.transport === "webrtc_sfu";
    const isActive = this.#state.active === true;
    const hasModal = !!this.#state.modalSessionId;

    if (!isSfu || !isActive || !hasModal) {
      const hadAlarm = await this.ctx.storage.getAlarm();
      if (hadAlarm !== null) {
        this.#captureRest("sfu_modal_keepalive_stopped", {
          source: "do_alarm",
          reason: "session_no_longer_sfu_active",
          modalSessionId: this.#state.modalSessionId ?? null,
          transport: this.#state.transport ?? null,
          active: this.#state.active,
        });
      }
      await this.ctx.storage.deleteAlarm();
      return;
    }

    // Capture the modal session ID BEFORE the heartbeat fetch.
    // A concurrent session.end may change state during the await below.
    const heartbeatModalSessionId = this.#state.modalSessionId!;
    const heartbeatStart = Date.now();

    // Cost-safety hard caps: abort before even talking to Modal if the
    // keepalive or session has exceeded its maximum lifetime. This protects
    // against billing-limit runaways when Modal status calls succeed but the
    // browser is gone.
    const keepaliveAgeMs = this.#state.sfuKeepaliveStartedAt
      ? heartbeatStart - this.#state.sfuKeepaliveStartedAt
      : 0;
    if (keepaliveAgeMs > this.#getSfuKeepaliveMaxAgeMs()) {
      await this.#abortSfuKeepalive("max_age");
      return;
    }

    const sessionAgeMs = this.#state.sfuSessionStartedAt
      ? heartbeatStart - this.#state.sfuSessionStartedAt
      : 0;
    if (sessionAgeMs > this.#getSfuSessionMaxAgeMs()) {
      await this.#abortSfuKeepalive("max_session_age");
      return;
    }

    let heartbeatOk = false;
    let heartbeatError: string | undefined;

    try {
      // Short timeout: a stuck status call must not extend the keepalive
      // interval or keep the DO alarm running. Modal is on the same cloud
      // region; 3-5s is plenty.
      await getModalSessionStatus(this.env, heartbeatModalSessionId, { timeoutMs: 5_000 });
      heartbeatOk = true;
    } catch (err) {
      heartbeatError = err instanceof Error ? err.message : String(err);
    }

    // CRITICAL: Reload state from storage AFTER the heartbeat fetch.
    // A concurrent session.end / WS close / #cleanupSfu may have
    // stopped the keepalive and modified state during the await.
    // We must not overwrite that with stale in-memory state.
    await this.#loadState();

    // Verify the session is still SFU-active with the SAME modal session.
    // If the session was stopped during the in-flight heartbeat, do NOT
    // reschedule and do NOT overwrite the stopped state.
    const stillActive =
      this.#state.sfuKeepaliveActive === true &&
      this.#state.transport === "webrtc_sfu" &&
      this.#state.active === true &&
      this.#state.modalSessionId === heartbeatModalSessionId;

    if (!stillActive) {
      // Session was stopped during heartbeat — no reschedule, no state overwrite.
      // The stop handler already deleted the alarm and captured the stopped event.
      if (heartbeatOk) {
        this.#captureRest("sfu_modal_keepalive_ok", {
          source: "do_alarm",
          modalSessionId: heartbeatModalSessionId,
          durationMs: Date.now() - heartbeatStart,
          note: "heartbeat_completed_after_stop_no_reschedule",
        });
      }
      return;
    }

    // Session is still active — update heartbeat metadata and reschedule
    if (heartbeatOk) {
      this.#state.sfuKeepaliveLastHeartbeatAt = heartbeatStart;
      this.#state.sfuKeepaliveFailures = 0;
      await this.#saveState();

      this.#captureRest("sfu_modal_keepalive_ok", {
        source: "do_alarm",
        modalSessionId: heartbeatModalSessionId,
        durationMs: Date.now() - heartbeatStart,
        intervalMs: SFU_KEEPALIVE_INTERVAL_MS,
      });
    } else {
      const failures = (this.#state.sfuKeepaliveFailures ?? 0) + 1;
      this.#state.sfuKeepaliveFailures = failures;
      this.#state.sfuKeepaliveLastHeartbeatAt = heartbeatStart;
      await this.#saveState();

      this.#captureRest("sfu_modal_keepalive_failed", {
        source: "do_alarm",
        modalSessionId: heartbeatModalSessionId,
        durationMs: Date.now() - heartbeatStart,
        error: heartbeatError,
        failureCount: failures,
        maxFailures: this.#getSfuKeepaliveMaxFailures(),
      });

      // Cost-safety: abort after exceeding consecutive failure budget.
      if (failures >= this.#getSfuKeepaliveMaxFailures()) {
        await this.#abortSfuKeepalive("max_failures");
        return;
      }
    }

    await this.ctx.storage.setAlarm(Date.now() + SFU_KEEPALIVE_INTERVAL_MS);
  }

  async #getOwnerTenantId(): Promise<string> {
    const owner = await this.ctx.storage.get<SessionOwner>(STORAGE_OWNER_KEY);
    return owner?.tenant_id ?? "legacy";
  }

  // Phase 3F: true only when the session was explicitly SFU-beta-authorized
  // at creation time. The raw beta token is never stored.
  async #isSfuBetaAuthorized(): Promise<boolean> {
    if (this.#state.sfuBetaAuthorized === true) return true;
    const owner = await this.ctx.storage.get<SessionOwner>(STORAGE_OWNER_KEY);
    return owner?.sfu_beta_authorized === true;
  }

  #restoreSessionId(ws: WebSocket): string | null {
    const attachment = ws.deserializeAttachment() as WsAttachment | undefined;
    const sid = attachment?.sessionId;
    if (sid && sid.startsWith("session:")) {
      this.#sessionId = sid;
      return sid;
    }
    return null;
  }

  // ── Stage 3B: SFU cost-safety guard configuration ────────────────
  #getSfuKeepaliveMaxAgeMs(): number {
    const parsed = parseInt(this.env.AVTR_SFU_KEEPALIVE_MAX_AGE_MS ?? "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 120_000;
  }

  #getSfuSessionMaxAgeMs(): number {
    const parsed = parseInt(this.env.AVTR_SFU_SESSION_MAX_AGE_MS ?? "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 120_000;
  }

  #getSfuKeepaliveMaxFailures(): number {
    const parsed = parseInt(this.env.AVTR_SFU_KEEPALIVE_MAX_FAILURES ?? "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 2;
  }

  #capture(
    ws: WebSocket,
    event: string,
    properties?: Record<string, unknown>,
  ): void {
    const attachment = ws.deserializeAttachment() as WsAttachment | undefined;
    const distinctId = attachment?.sessionId ?? this.#sessionId;
    if (distinctId === "unknown") {
      console.error("[PostHog] capture called with unknown sessionId", {
        event,
      });
    }
    captureEvent(
      this.env,
      this.ctx.waitUntil.bind(this.ctx),
      distinctId,
      event,
      {
        avatarSessionId: distinctId,
        ...properties,
      },
    );
  }

  #captureError(
    ws: WebSocket,
    error: unknown,
    context?: Record<string, unknown>,
  ): void {
    const attachment = ws.deserializeAttachment() as WsAttachment | undefined;
    const distinctId = attachment?.sessionId ?? this.#sessionId;
    captureError(
      this.env,
      this.ctx.waitUntil.bind(this.ctx),
      distinctId,
      error,
      {
        avatarSessionId: distinctId,
        ...context,
      },
    );
  }

  /**
   * Generate short-lived TURN credentials from Cloudflare Realtime API.
   *
   * This is on the CRITICAL PATH (not fire-and-forget) — Modal needs TURN
   * credentials before starting the publisher's ICE gathering. The fetch
   * has a strict 3s timeout.
   *
   * TURN credentials are NEVER persisted in DO state or PostHog. They are
   * passed to Modal in the session start request body only.
   *
   * Behavior:
   *   - TURN secrets configured + API call succeeds → returns iceServers array.
   *   - TURN secrets configured + API call fails → throws SfuClientError.
   *   - TURN secrets NOT configured + AVTR_SFU_REQUIRE_TURN=true → throws SfuClientError.
   *   - TURN secrets NOT configured + AVTR_SFU_REQUIRE_TURN=false → returns null (dev mode).
   */
  async #generateTurnCredentials(): Promise<unknown[] | null> {
    const keyId = this.env.AVTR_SFU_TURN_KEY_ID;
    const apiToken = this.env.AVTR_SFU_TURN_KEY_API_TOKEN;
    const requireTurn = (this.env.AVTR_SFU_REQUIRE_TURN ?? "true") !== "false";

    if (!keyId || !apiToken) {
      if (requireTurn) {
        throw new SfuClientError(
          "sfu_turn_not_configured: TURN secrets missing and AVTR_SFU_REQUIRE_TURN is true",
          503,
          "",
        );
      }
      // Dev/diagnostics mode — no TURN, direct ICE only.
      return null;
    }

    // Critical-path fetch with strict 3s timeout.
    const resp = await fetch(
      `https://rtc.live.cloudflare.com/v1/turn/keys/${keyId}/credentials/generate-ice-servers`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ ttl: 7200 }), // 2 hours — covers any session
        signal: AbortSignal.timeout(3000),
      },
    );

    if (!resp.ok) {
      const status = resp.status;
      let bodyText = "";
      try { bodyText = await resp.text(); } catch { /* ignore */ }
      throw new SfuClientError(
        `sfu_turn_credentials_unavailable: TURN API returned ${status}`,
        503,
        bodyText.slice(0, 200),
      );
    }

    const data = await resp.json() as { iceServers?: unknown[] };
    const iceServers = data.iceServers;
    if (!Array.isArray(iceServers) || iceServers.length === 0) {
      throw new SfuClientError(
        "sfu_turn_credentials_unavailable: TURN API returned empty iceServers",
        503,
        "",
      );
    }

    // Validate that at least one entry has TURN URLs (not just STUN).
    // STUN-only iceServers cannot provide relay candidates — if none of the
    // entries contain turn:/turns: URLs with credentials, fail explicitly.
    const hasUsableTurn = iceServers.some((entry): boolean => {
      if (typeof entry !== "object" || entry === null) return false;
      const s = entry as Record<string, unknown>;
      if (!s.username || !s.credential) return false;
      const urls = s.urls;
      const urlList = typeof urls === "string" ? [urls] : Array.isArray(urls) ? urls : [];
      return urlList.some(
        (u) => typeof u === "string" && (u.startsWith("turn:") || u.startsWith("turns:")),
      );
    });
    if (!hasUsableTurn) {
      throw new SfuClientError(
        "sfu_turn_credentials_unavailable: TURN API returned iceServers but none contain usable TURN URLs with credentials",
        503,
        "",
      );
    }

    // DO NOT persist in DO state. DO NOT log credentials.
    // Pass through to Modal request body only.
    return iceServers;
  }

  /**
   * Extract an explicit allowlist of safe fields from the raw ice_debug dict.
   *
   * Sanitize at boundary: even though the Python side sanitizes, the Worker
   * must only forward explicitly allowlisted fields. No raw candidates, no
   * full SDP, no arbitrary nested object spread. All fields are truncated
   * where applicable to prevent unbounded PostHog payloads.
   */
  #extractIceDebugAllowlist(raw: Record<string, unknown>): Record<string, unknown> {
    const str = (v: unknown, max = 500): string =>
      typeof v === "string" ? v.slice(0, max) : v == null ? "" : String(v).slice(0, max);
    const numOrNull = (v: unknown): number | null =>
      typeof v === "number" && Number.isFinite(v) ? v : null;
    const arr = (v: unknown): unknown[] => (Array.isArray(v) ? v.slice(0, 50) : []);
    const boundedArr = (v: unknown): unknown[] => (Array.isArray(v) ? v.slice(0, 20) : []);

    const checkpoints = raw.checkpoints as Record<string, unknown> | undefined;
    const result: Record<string, unknown> = {
      attempt_id: str(raw.attempt_id, 100),
      track_name: str(raw.track_name, 100),
      publisher_type: str(raw.publisher_type, 50),
      answer_received: raw.answer_received === true,
      answer_sdp_length: numOrNull(raw.answer_sdp_length),
      local_candidate_count: numOrNull(raw.local_candidate_count),
      local_candidate_types: arr(raw.local_candidate_types),
      local_candidate_protocols: arr(raw.local_candidate_protocols),
      local_candidate_foundation_count: numOrNull(raw.local_candidate_foundation_count),
      final_ice_state: str(raw.final_ice_state, 50),
      final_gathering_state: str(raw.final_gathering_state, 50),
      final_signaling_state: str(raw.final_signaling_state, 50),
      final_connection_state: str(raw.final_connection_state, 50),
      stats_collection_timed_out: raw.stats_collection_timed_out === true,
      webrtcbin_error_messages: boundedArr(raw.webrtcbin_error_messages),
      gstreamer_bus_errors: boundedArr(raw.gstreamer_bus_errors),
      gstreamer_bus_warnings: boundedArr(raw.gstreamer_bus_warnings),
      gstreamer_bus_element_msgs: boundedArr(raw.gstreamer_bus_element_msgs),
      timeout_phase: str(raw.timeout_phase, 50),
      cleanup_ok: raw.cleanup_ok == null ? null : raw.cleanup_ok === true,
      turn_configured: raw.turn_configured === true,
      turn_url_count: numOrNull(raw.turn_url_count),
      turn_transports: arr(raw.turn_transports),
      selected_local_candidate_type: str(raw.selected_local_candidate_type, 50),
      selected_remote_candidate_type: str(raw.selected_remote_candidate_type, 50),
    };

    // Phase timings (allowlisted checkpoint keys only)
    if (checkpoints && typeof checkpoints === "object") {
      const timingKeys = [
        "pipeline_created",
        "pipeline_playing",
        "offer_created",
        "local_description_set",
        "ice_gathering_started",
        "ice_gathering_completed",
        "first_candidate",
        "offer_callback_started",
        "offer_callback_completed",
        "offer_callback_failed",
        "remote_description_set",
        "ice_connect_started",
        "ice_connected",
        "ice_timeout",
        "offer_ready",
      ];
      const timings: Record<string, number | null> = {};
      for (const k of timingKeys) {
        timings[k] = numOrNull((checkpoints as Record<string, unknown>)[k]);
      }
      result.phase_timings_ms = timings;
    }

    // State transition counts (summary only — full histories are in Modal logs)
    const countArr = (v: unknown): number =>
      Array.isArray(v) ? Math.min(v.length, 50) : 0;
    result.ice_connection_transition_count = countArr(raw.ice_connection_transitions);
    result.ice_gathering_transition_count = countArr(raw.ice_gathering_transitions);
    result.signaling_state_transition_count = countArr(raw.signaling_state_transitions);
    result.connection_state_transition_count = countArr(raw.connection_state_transitions);

    return result;
  }

  async #acquireModalLock(): Promise<void> {
    while (this.#modalRenderBusy) {
      await new Promise<void>((resolve) => {
        this.#modalRenderBusyResolve = resolve;
      });
    }
    this.#modalRenderBusy = true;
  }

  #releaseModalLock(): void {
    this.#modalRenderBusy = false;
    if (this.#modalRenderBusyResolve) {
      const resolve = this.#modalRenderBusyResolve;
      this.#modalRenderBusyResolve = null;
      resolve();
    }
  }

  async fetch(request: Request): Promise<Response> {
    const upgradeHeader = request.headers.get("Upgrade");
    if (upgradeHeader === "websocket") {
      return this.#handleWebSocketUpgrade(request);
    }

    const isInternalRest = request.headers.get("X-Internal-REST") === "true";
    if (isInternalRest) {
      return this.#handleRestRequest(request);
    }

    return Response.json(
      { error: { code: "expected_websocket", message: "Expected Upgrade: websocket header" } },
      { status: 400 },
    );
  }

  async #handleWebSocketUpgrade(request: Request): Promise<Response> {
    try {
      const url = new URL(request.url);
      const segments = url.pathname.split("/").filter(Boolean);
      if (segments.length >= 2 && segments[0] === "sessions") {
        this.#sessionId = decodeURIComponent(segments[1]);
      } else if (segments.length >= 3 && segments[0] === "v1" && segments[1] === "sessions") {
        this.#sessionId = decodeURIComponent(segments[2]);
      }
    } catch {
      // URL parsing failed; keep default sessionId
    }

    const pair = new WebSocketPair();
    const [client, server] = Object.values(pair);

    this.ctx.acceptWebSocket(server);
    server.serializeAttachment({
      connectedAt: Date.now(),
      sessionId: this.#sessionId,
    } satisfies WsAttachment);

    this.#capture(server, "ws_connected");

    return new Response(null, {
      status: 101,
      webSocket: client,
    });
  }

  async webSocketMessage(
    ws: WebSocket,
    message: string | ArrayBuffer,
  ): Promise<void> {
    if (!this.#restoreSessionId(ws)) {
      this.#sendError(ws, "invalid_session", "Session ID not found after hibernation", true);
      ws.close(1011, "invalid session");
      return;
    }

    if (typeof message !== "string") {
      await this.#processAudioChunk(ws, message);
      return;
    }

    let parsed: ClientMessage;
    try {
      parsed = JSON.parse(message);
    } catch {
      this.#sendError(ws, "invalid_json", "Message must be valid JSON");
      return;
    }

    if (parsed.type === "client.telemetry") {
      this.#capture(ws, parsed.event, {
        source: "browser",
        clientElapsedMs: parsed.elapsedMs,
        ...parsed.properties,
      });
      return;
    }

    await this.#loadState();

    switch (parsed.type) {
      case "session.start":
        await this.#handleSessionStart(ws, parsed);
        break;
      case "render_debug_frame":
        await this.#handleRenderFrame(ws);
        break;
      case "session.end":
        await this.#handleSessionEnd(ws);
        break;
      case "mic.start":
        await this.#handleMicStart(ws);
        break;
      case "mic.stop":
        await this.#handleMicStop(ws);
        break;
      case "avatar.speech.playback.ended":
        this.#handlePlaybackAck(ws, parsed);
        break;
      default:
        this.#sendError(
          ws,
          "unknown_type",
          `Unknown message type: ${(parsed as { type: string }).type}`,
        );
    }
  }

  async webSocketClose(
    ws: WebSocket,
    _code: number,
    _reason: string,
    _wasClean: boolean,
  ): Promise<void> {
    this.#restoreSessionId(ws);

    await this.#loadState();

    this.#sessionClosing = true;
    this.#idleLiveEnabled = false;
    this.#assistantSpeechActive = false;
    this.#pendingPlaybackAckTurnId = null;
    this.#activeSpeechRenderTurnId = null;
    this.#avatarTimelineAbort = true;
    this.#avatarSpeechQueue = [];

    if (this.#modalTimelineStreamActive && this.#state.modalSessionId) {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
      try {
        await stopModalTimeline(this.env, this.#state.modalSessionId);
      } catch {
        // best-effort
      }
    }

    if (this.#modalFmp4StreamActive && this.#state.modalSessionId) {
      const isAvMode = this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true";
      this.#modalTimelineAbort = true;

      try {
        await stopModalTimeline(this.env, this.#state.modalSessionId, { useControlEndpoint: isAvMode });
      } catch {
        // best-effort
      }

      if (!this.#fmp4StreamSummaryCaptured && this.#modalFmp4StreamActive) {
        const endPromise = new Promise<void>((resolve) => {
          this.#fmp4StreamEndResolve = resolve;
        });
        const timeoutPromise = new Promise<void>((resolve) => {
          setTimeout(resolve, 5_000);
        });
        await Promise.race([endPromise, timeoutPromise]);
        this.#fmp4StreamEndResolve = null;
      }

      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    } else {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    }

    this.#capture(ws, "ws_closed", {
      audioChunkCount: this.#audioChunkCount,
      audioBytesReceived: this.#audioBytesReceived,
      wasActive: this.#state.active,
      wasStarting: this.#state.starting,
      startAttemptId: this.#state.startAttemptId ?? null,
      modalSessionId: this.#state.modalSessionId,
    });

    // Stage 3B cost-safety: if the browser closes while Modal start is
    // in-flight, set a cancellation flag so the start path can abort BEFORE
    // emitting session_ready and BEFORE starting the keepalive alarm. Without
    // this, a closed tab can orphan a Modal GPU session that keeps pinging
    // for hours (Modal billing-limit incident).
    if (this.#state.starting && this.#state.startAttemptId) {
      this.#state.startCancelRequested = true;
      await this.#saveState();
      this.#capture(ws, "ws_start_cancel_requested", {
        startAttemptId: this.#state.startAttemptId,
      });
    }

    if (this.#state.active) {
      // Capacity lifecycle: tracks cleanup → Modal /end → capacity release.
      // The slot is held until Modal GPU teardown finishes so the next
      // session cannot start a second publisher on the same L40S.
      const wasCapacityAcquired = this.#state.sfuCapacityAcquired === true;
      if (this.#state.transport === "webrtc_sfu") {
        await this.#cleanupSfuTracksAndState();
      }
      try {
        await endSession(this.env, this.#state.modalSessionId ?? undefined).then(r => r.data);
      } catch {
        // best-effort cleanup on disconnect
      }
      if (wasCapacityAcquired) {
        await this.#releaseSfuCapacity(true);
      }
      this.#state = emptyState();
      this.#conversationHistory = [];
      await this.#saveState();
    }
    await this.#resetSpeechState();
    ws.close(1000, "session closed");
  }

  async #loadState(): Promise<void> {
    const stored = await this.ctx.storage.get<SessionState>(STORAGE_KEY);
    if (stored) this.#state = stored;
  }

  async #saveState(): Promise<void> {
    await this.ctx.storage.put(STORAGE_KEY, this.#state);
  }

  // ── Audio & Speech Pipeline ──────────────────────────────────────

  async #handleMicStart(ws: WebSocket): Promise<void> {
    this.#capture(ws, "mic_start_requested");

    await this.#resetSpeechState();
    this.#firstAudioChunkCaptured = false;

    ws.send(
      JSON.stringify({
        type: "status",
        status: "mic_ready",
      } satisfies ServerMessage),
    );
    console.log("[AvatarSessionDO] mic.start", {
      sessionId: this.#sessionId,
    });
  }

  async #handleMicStop(ws: WebSocket): Promise<void> {
    const activeTurnBeforeStop = this.#turnOpen;
    const currentTurnAgeMs = this.#turnOpen ? Date.now() - this.#currentTurnStartedAt : 0;

    this.#capture(ws, "mic_stop_summary", {
      activeTurnBeforeStop,
      currentTurnAgeMs,
      suppressedEndCount: this.#suppressedEndCount,
      suppressedStartCount: this.#suppressedStartCount,
      smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
      audioChunkCount: this.#audioChunkCount,
      audioBytesReceived: this.#audioBytesReceived,
      smartTurnAccumulatedBytes: this.#smartTurnAccumulatedBytes,
      smartTurnAccumulatedApproxSeconds: this.#smartTurnAccumulatedBytes / 2 / 16000,
    });

    this.#capture(ws, "mic_stop_requested", {
      audioChunkCount: this.#audioChunkCount,
      audioBytesReceived: this.#audioBytesReceived,
    });

    if (this.#turnOpen) {
      const forcedTurnId = this.#turnCounter;
      if (this.#turnAudioChunks.length > 0) {
        this.#capture(ws, "speech_finalized", {
          turnId: forcedTurnId,
          finalizationReason: "mic_stop",
          forcedFinalization: true,
          smartTurnFallback: true,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
          audioChunkCount: this.#audioChunkCount,
        });

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "speech.finalized",
              turnId: forcedTurnId,
              smartTurn: { prediction: 1, probability: 1, rawLogit: 0 },
              smartTurnFallback: true,
            } satisfies ServerMessage),
          );
        }

        console.log("[AvatarSessionDO] speech.finalized (mic_stop forced)", {
          sessionId: this.#sessionId,
          turnId: forcedTurnId,
          finalizationReason: "mic_stop",
        });

        this.#handleTurnFinalized(ws, forcedTurnId);
      } else {
        this.#capture(ws, "stt_discarded_empty_turn", {
          turnId: forcedTurnId,
          finalizationReason: "mic_stop",
        });
        console.log("[AvatarSessionDO] stt_discarded_empty_turn (mic_stop)", {
          sessionId: this.#sessionId,
          turnId: forcedTurnId,
        });
      }
    }

    await this.#resetSpeechState();
    ws.send(
      JSON.stringify({
        type: "status",
        status: "mic_stopped",
      } satisfies ServerMessage),
    );
    console.log("[AvatarSessionDO] mic.stop", {
      sessionId: this.#sessionId,
      audioChunkCount: this.#audioChunkCount,
      audioBytesReceived: this.#audioBytesReceived,
    });
    this.#audioChunkCount = 0;
    this.#audioBytesReceived = 0;
    this.#firstAudioChunkCaptured = false;
  }

  async #processAudioChunk(
    ws: WebSocket,
    chunk: ArrayBuffer,
  ): Promise<void> {
    if (chunk.byteLength === 0 || chunk.byteLength % 2 !== 0) {
      this.#sendError(
        ws,
        "invalid_audio_chunk",
        `Expected non-empty PCM16 (even byte count), got ${chunk.byteLength} bytes`,
      );
      return;
    }

    this.#audioChunkCount++;
    this.#audioBytesReceived += chunk.byteLength;

    if (!this.#firstAudioChunkCaptured) {
      this.#firstAudioChunkCaptured = true;
      this.#capture(ws, "first_audio_chunk_received", {
        byteLength: chunk.byteLength,
        audioChunkCount: this.#audioChunkCount,
      });
    }

    if (this.#audioChunkCount % AUDIO_STATS_INTERVAL === 0) {
      this.#capture(ws, "audio_stats", {
        audioChunkCount: this.#audioChunkCount,
        audioBytesReceived: this.#audioBytesReceived,
      });

      ws.send(
        JSON.stringify({
          type: "audio.stats",
          chunksReceived: this.#audioChunkCount,
          bytesReceived: this.#audioBytesReceived,
        } satisfies ServerMessage),
      );
    }

    let decision: VadDecision;
    try {
      decision = await this.#processVad(chunk);
    } catch (err) {
      console.error("[AvatarSessionDO] vad_error", {
        sessionId: this.#sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
      this.#captureError(ws, err, { stage: "vad_process" });
      return;
    }

    if (decision.shouldStart || decision.shouldEnd) {
      console.log("[AvatarSessionDO] vad_decision", {
        sessionId: this.#sessionId,
        shouldStart: decision.shouldStart,
        shouldEnd: decision.shouldEnd,
        inSpeech: decision.inSpeech,
        probability: decision.probability,
        audioChunkCount: this.#audioChunkCount,
        turnOpen: this.#turnOpen,
        speechSegmentActive: this.#speechSegmentActive,
      });
    }

    // ── VAD shouldStart ──────────────────────────────────────────────

    if (decision.shouldStart) {
      if (!this.#turnOpen) {
        this.#turnOpen = true;
        this.#speechSegmentActive = true;
        this.#turnCounter++;
        this.#segmentIndex = 0;
        this.#currentTurnStartedAt = Date.now();
        this.#suppressedEndCount = 0;
        this.#suppressedStartCount = 0;
        this.#smartTurnAnalyzeCount = 0;
        this.#smartTurnAccumulatedBytesAtLastAnalyze = 0;

        const turnId = this.#turnCounter;

        this.#capture(ws, "vad_boundary", {
          boundary: "start",
          turnId,
          segmentIndex: this.#segmentIndex,
          vadProbability: decision.probability,
          inSpeech: decision.inSpeech,
          audioChunkCount: this.#audioChunkCount,
          audioBytesReceived: this.#audioBytesReceived,
        });

        this.#capture(ws, "speech_started", {
          turnId,
          segmentIndex: this.#segmentIndex,
          audioChunkCount: this.#audioChunkCount,
          vadProbability: decision.probability,
        });

        ws.send(
          JSON.stringify({
            type: "speech.started",
            turnId,
          } satisfies ServerMessage),
        );

        console.log("[AvatarSessionDO] speech.started", {
          sessionId: this.#sessionId,
          turnId,
          segmentIndex: this.#segmentIndex,
          probability: decision.probability,
          audioChunkCount: this.#audioChunkCount,
        });
      } else if (!this.#speechSegmentActive) {
        if (this.#analysisInFlight) {
          this.#analysisId++;
          this.#capture(ws, "smart_turn_analysis_invalidated_by_new_segment", {
            analysisId: this.#analysisId,
            turnId: this.#turnCounter,
            newSegmentIndex: this.#segmentIndex + 1,
            audioChunkCount: this.#audioChunkCount,
          });
          console.log("[AvatarSessionDO] smart_turn_analysis_invalidated_by_new_segment", {
            sessionId: this.#sessionId,
            analysisId: this.#analysisId,
            turnId: this.#turnCounter,
            newSegmentIndex: this.#segmentIndex + 1,
            audioChunkCount: this.#audioChunkCount,
          });
        }

        this.#speechSegmentActive = true;
        this.#segmentIndex++;

        this.#capture(ws, "speech_segment_started", {
          turnId: this.#turnCounter,
          segmentIndex: this.#segmentIndex,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
          vadProbability: decision.probability,
          audioChunkCount: this.#audioChunkCount,
        });

        ws.send(
          JSON.stringify({
            type: "speech.segment.started",
            turnId: this.#turnCounter,
            segmentIndex: this.#segmentIndex,
          } satisfies ServerMessage),
        );

        console.log("[AvatarSessionDO] speech.segment.started", {
          sessionId: this.#sessionId,
          turnId: this.#turnCounter,
          segmentIndex: this.#segmentIndex,
          probability: decision.probability,
          audioChunkCount: this.#audioChunkCount,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
        });
      } else {
        this.#suppressedStartCount++;
        this.#capture(ws, "vad_start_suppressed_while_turn_active", {
          turnId: this.#turnCounter,
          audioChunkCount: this.#audioChunkCount,
          vadProbability: decision.probability,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
          suppressedStartCount: this.#suppressedStartCount,
        });
        console.warn("[AvatarSessionDO] vad_start_suppressed", {
          sessionId: this.#sessionId,
          turnId: this.#turnCounter,
          audioChunkCount: this.#audioChunkCount,
          vadProbability: decision.probability,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
          suppressedStartCount: this.#suppressedStartCount,
        });
      }
    }

    // ── Accumulate turn audio for STT (all chunks while turn open) ──

    if (this.#turnOpen) {
      this.#pushTurnAudio(chunk, ws);
    }

    // ── Accumulate to Smart Turn (only while actively speaking) ──────

    if (this.#turnOpen && this.#speechSegmentActive) {
      try {
        await this.#accumulateSmartTurn(chunk);
      } catch (err) {
        console.error("[AvatarSessionDO] smart_turn_accumulate_error", {
          sessionId: this.#sessionId,
          turnId: this.#turnCounter,
          error: err instanceof Error ? err.message : String(err),
        });
        this.#captureError(ws, err, {
          stage: "smart_turn_accumulate",
          turnId: this.#turnCounter,
        });
      }
    }

    // ── VAD shouldEnd ────────────────────────────────────────────────

    if (decision.shouldEnd && this.#turnOpen && this.#speechSegmentActive) {
      if (this.#analysisInFlight) {
        this.#capture(ws, "smart_turn_analyze_skipped_in_flight", {
          turnId: this.#turnCounter,
          segmentIndex: this.#segmentIndex,
          audioChunkCount: this.#audioChunkCount,
        });
        console.log("[AvatarSessionDO] smart_turn_analyze_skipped_in_flight", {
          sessionId: this.#sessionId,
          turnId: this.#turnCounter,
          segmentIndex: this.#segmentIndex,
          audioChunkCount: this.#audioChunkCount,
        });
      } else {
        const localAnalysisId = ++this.#analysisId;
        const localTurnId = this.#turnCounter;
        const localSegmentIndex = this.#segmentIndex;
        this.#analysisInFlight = true;
        this.#speechSegmentActive = false;
        this.#suppressedEndCount++;
        const currentTurnAgeMs = Date.now() - this.#currentTurnStartedAt;

        this.#capture(ws, "vad_boundary", {
          boundary: "end",
          turnId: localTurnId,
          segmentIndex: localSegmentIndex,
          vadProbability: decision.probability,
          inSpeech: decision.inSpeech,
          audioChunkCount: this.#audioChunkCount,
          audioBytesReceived: this.#audioBytesReceived,
          currentTurnAgeMs,
          suppressedEndCount: this.#suppressedEndCount,
        });

        try {
          const analyzeStartedAt = Date.now();
          this.#smartTurnAnalyzeCount++;
          this.#smartTurnAccumulatedBytesAtLastAnalyze = this.#smartTurnAccumulatedBytes;

          this.#capture(ws, "smart_turn_analyze_started", {
            analysisId: localAnalysisId,
            turnId: localTurnId,
            segmentIndex: localSegmentIndex,
            accumulatedChunks: this.#smartTurnChunkCount,
            accumulatedBytes: this.#smartTurnAccumulatedBytes,
            audioChunkCount: this.#audioChunkCount,
          });

          const { result: stResult } = await this.#evaluateSmartTurn();
          const latencyMs = Date.now() - analyzeStartedAt;

          const processedApproxSeconds = stResult.processedApproxSeconds ?? (this.#smartTurnAccumulatedBytes / 2 / 16000);
          const bufferApproxSeconds = stResult.bufferApproxSeconds ?? (this.#smartTurnAccumulatedBytes / 2 / 16000);

          if (
            this.#turnCounter !== localTurnId ||
            this.#segmentIndex !== localSegmentIndex ||
            !this.#turnOpen ||
            this.#speechSegmentActive ||
            localAnalysisId !== this.#analysisId
          ) {
            this.#capture(ws, "smart_turn_result_ignored_stale", {
              analysisId: localAnalysisId,
              turnId: localTurnId,
              segmentIndex: localSegmentIndex,
              currentTurnId: this.#turnCounter,
              currentSegmentIndex: this.#segmentIndex,
              currentTurnOpen: this.#turnOpen,
              currentSpeechSegmentActive: this.#speechSegmentActive,
              currentAnalysisId: this.#analysisId,
              smartTurnPrediction: stResult.prediction,
              smartTurnProbability: stResult.probability,
              smartTurnRawLogit: stResult.rawLogit,
              latencyMs,
              audioChunkCount: this.#audioChunkCount,
            });
            console.warn("[AvatarSessionDO] smart_turn_result_ignored_stale", {
              sessionId: this.#sessionId,
              analysisId: localAnalysisId,
              turnId: localTurnId,
              segmentIndex: localSegmentIndex,
              currentTurnId: this.#turnCounter,
              currentSegmentIndex: this.#segmentIndex,
              prediction: stResult.prediction,
              probability: stResult.probability,
              latencyMs,
              audioChunkCount: this.#audioChunkCount,
            });
          } else {
            this.#capture(ws, "smart_turn_analyze_completed", {
              analysisId: localAnalysisId,
              turnId: localTurnId,
              segmentIndex: localSegmentIndex,
              smartTurnPrediction: stResult.prediction,
              smartTurnProbability: stResult.probability,
              smartTurnRawLogit: stResult.rawLogit,
              latencyMs,
              accumulatedChunks: this.#smartTurnChunkCount,
              accumulatedBytes: this.#smartTurnAccumulatedBytes,
              bufferApproxSeconds,
              processedApproxSeconds,
              currentTurnAgeMs,
              suppressedEndCount: this.#suppressedEndCount,
              smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
              audioChunkCount: this.#audioChunkCount,
            });

            if (
              stResult.prediction === 1 &&
              processedApproxSeconds < 1.0 &&
              stResult.probability < 0.95
            ) {
              this.#capture(ws, "smart_turn_short_completion_suppressed", {
                analysisId: localAnalysisId,
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                smartTurnPrediction: stResult.prediction,
                smartTurnProbability: stResult.probability,
                smartTurnRawLogit: stResult.rawLogit,
                processedApproxSeconds,
                bufferApproxSeconds,
                latencyMs,
                currentTurnAgeMs,
                audioChunkCount: this.#audioChunkCount,
              });
              console.log("[AvatarSessionDO] smart_turn_short_completion_suppressed", {
                sessionId: this.#sessionId,
                analysisId: localAnalysisId,
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                probability: stResult.probability,
                processedApproxSeconds,
                latencyMs,
                audioChunkCount: this.#audioChunkCount,
              });

              this.#capture(ws, "speech_continues", {
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                reason: "short_completion_suppressed",
                smartTurnPrediction: stResult.prediction,
                smartTurnProbability: stResult.probability,
                smartTurnRawLogit: stResult.rawLogit,
                smartTurnLatencyMs: latencyMs,
                processedApproxSeconds,
                bufferApproxSeconds,
                currentTurnAgeMs,
                suppressedEndCount: this.#suppressedEndCount,
                smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
                audioChunkCount: this.#audioChunkCount,
              });

              ws.send(
                JSON.stringify({
                  type: "speech.continues",
                  turnId: localTurnId,
                  smartTurn: stResult,
                } satisfies ServerMessage),
              );

              ws.send(
                JSON.stringify({
                  type: "status",
                  status: "listening_same_turn",
                } satisfies ServerMessage),
              );

              console.log("[AvatarSessionDO] speech.continues (short suppressed)", {
                sessionId: this.#sessionId,
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                probability: stResult.probability,
                processedApproxSeconds,
                latencyMs,
                audioChunkCount: this.#audioChunkCount,
              });
            } else if (stResult.prediction === 1) {
              this.#turnOpen = false;
              this.#smartTurnChunkCount = 0;
              this.#smartTurnAccumulatedBytes = 0;

              this.#capture(ws, "speech_finalized", {
                turnId: localTurnId,
                finalizationReason: "smart_turn_complete",
                smartTurnPrediction: stResult.prediction,
                smartTurnProbability: stResult.probability,
                smartTurnRawLogit: stResult.rawLogit,
                latencyMs,
                currentTurnAgeMs,
                suppressedEndCount: this.#suppressedEndCount,
                smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
                processedApproxSeconds,
                bufferApproxSeconds,
                audioChunkCount: this.#audioChunkCount,
              });

              ws.send(
                JSON.stringify({
                  type: "speech.finalized",
                  turnId: localTurnId,
                  smartTurn: stResult,
                } satisfies ServerMessage),
              );

              console.log("[AvatarSessionDO] speech.finalized", {
                sessionId: this.#sessionId,
                turnId: localTurnId,
                prediction: stResult.prediction,
                probability: stResult.probability,
                rawLogit: stResult.rawLogit,
                currentTurnAgeMs,
                processedApproxSeconds,
                suppressedEndCount: this.#suppressedEndCount,
                smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
                audioChunkCount: this.#audioChunkCount,
              });

              this.#handleTurnFinalized(ws, localTurnId);
            } else {
              this.#capture(ws, "speech_continues", {
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                reason: "smart_turn_incomplete",
                smartTurnPrediction: stResult.prediction,
                smartTurnProbability: stResult.probability,
                smartTurnRawLogit: stResult.rawLogit,
                smartTurnLatencyMs: latencyMs,
                processedApproxSeconds,
                bufferApproxSeconds,
                currentTurnAgeMs,
                suppressedEndCount: this.#suppressedEndCount,
                smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
                audioChunkCount: this.#audioChunkCount,
              });

              ws.send(
                JSON.stringify({
                  type: "speech.continues",
                  turnId: localTurnId,
                  smartTurn: stResult,
                } satisfies ServerMessage),
              );

              ws.send(
                JSON.stringify({
                  type: "status",
                  status: "listening_same_turn",
                } satisfies ServerMessage),
              );

              console.log("[AvatarSessionDO] speech.continues", {
                sessionId: this.#sessionId,
                turnId: localTurnId,
                segmentIndex: localSegmentIndex,
                prediction: stResult.prediction,
                probability: stResult.probability,
                rawLogit: stResult.rawLogit,
                currentTurnAgeMs,
                processedApproxSeconds,
                suppressedEndCount: this.#suppressedEndCount,
                smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
                audioChunkCount: this.#audioChunkCount,
              });
            }
          }
        } catch (err) {
          this.#turnOpen = false;
          this.#speechSegmentActive = false;
          this.#smartTurnChunkCount = 0;
          this.#smartTurnAccumulatedBytes = 0;

          const errTurnAgeMs = Date.now() - this.#currentTurnStartedAt;

          this.#captureError(ws, err, {
            stage: "smart_turn_evaluate",
            turnId: localTurnId,
            fallback: true,
            analysisId: localAnalysisId,
          });

          this.#capture(ws, "speech_finalized", {
            turnId: localTurnId,
            finalizationReason: "smart_turn_error_fallback",
            smartTurnFallback: true,
            smartTurnPrediction: 1,
            smartTurnProbability: 1,
            smartTurnRawLogit: 0,
            currentTurnAgeMs: errTurnAgeMs,
            suppressedEndCount: this.#suppressedEndCount,
            smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
            accumulatedBytes: this.#smartTurnAccumulatedBytesAtLastAnalyze,
            audioChunkCount: this.#audioChunkCount,
          });

          ws.send(
            JSON.stringify({
              type: "speech.finalized",
              turnId: localTurnId,
              smartTurn: { prediction: 1, probability: 1, rawLogit: 0 },
              smartTurnFallback: true,
            } satisfies ServerMessage),
          );

          console.warn("[AvatarSessionDO] smart_turn_error_fallback", {
            sessionId: this.#sessionId,
            turnId: localTurnId,
            error: err instanceof Error ? err.message : String(err),
            currentTurnAgeMs: errTurnAgeMs,
            audioChunkCount: this.#audioChunkCount,
          });

          this.#handleTurnFinalized(ws, localTurnId);
        } finally {
          this.#analysisInFlight = false;
        }
      }
    }
  }

  async #processVad(chunk: ArrayBuffer): Promise<VadDecision> {
    const service = this.env.VAD_SERVICE as VadServiceRpc | undefined;
    if (!service?.processAudio) {
      throw new Error("VAD_SERVICE binding not available");
    }
    return service.processAudio(`${this.#sessionId}:vad`, chunk);
  }

  async #accumulateSmartTurn(chunk: ArrayBuffer): Promise<void> {
    const service = this.env
      .SMART_TURN_SERVICE as SmartTurnServiceRpc | undefined;
    if (!service?.accumulateAudio) return;

    this.#smartTurnChunkCount++;
    this.#smartTurnAccumulatedBytes += chunk.byteLength;

    await service.accumulateAudio(`${this.#sessionId}:st`, chunk, {
      sampleRate: 16000,
      encoding: "pcm16",
    });
  }

  async #evaluateSmartTurn(): Promise<{ result: SmartTurnDecision }> {
    const service = this.env
      .SMART_TURN_SERVICE as SmartTurnServiceRpc | undefined;
    if (!service?.analyzeTurn) {
      throw new Error("SMART_TURN_SERVICE binding not available");
    }

    const result = await service.analyzeTurn(`${this.#sessionId}:st`);

    console.log("[AvatarSessionDO] smart_turn_analyze", {
      sessionId: this.#sessionId,
      prediction: result.prediction,
      probability: result.probability,
      rawLogit: result.rawLogit,
      accumulatedChunks: this.#smartTurnChunkCount,
      accumulatedBytes: this.#smartTurnAccumulatedBytes,
    });

    return { result };
  }

  #pushTurnAudio(chunk: ArrayBuffer, ws: WebSocket): void {
    this.#turnAudioChunks.push(chunk.slice(0));
    this.#turnAudioBytes += chunk.byteLength;

    if (this.#turnAudioBytes > MAX_TURN_AUDIO_BYTES) {
      while (this.#turnAudioBytes > MAX_TURN_AUDIO_BYTES && this.#turnAudioChunks.length > 0) {
        const removed = this.#turnAudioChunks.shift()!;
        this.#turnAudioBytes -= removed.byteLength;
      }
      this.#trimTelemetryChunkCounter++;
      const now = Date.now();
      const shouldEmit = !this.#trimStarted
        || this.#trimTelemetryChunkCounter >= 50
        || (now - this.#trimTelemetryTimestamp) >= 5000;
      if (shouldEmit) {
        this.#trimStarted = true;
        this.#trimTelemetryChunkCounter = 0;
        this.#trimTelemetryTimestamp = now;
        this.#capture(ws, "stt_turn_audio_trimmed", {
          turnId: this.#turnCounter,
          audioBytes: this.#turnAudioBytes,
          maxBytes: MAX_TURN_AUDIO_BYTES,
          maxSeconds: MAX_TURN_AUDIO_SECONDS,
        });
        console.warn("[AvatarSessionDO] stt_turn_audio_trimmed", {
          sessionId: this.#sessionId,
          turnId: this.#turnCounter,
          audioBytes: this.#turnAudioBytes,
          maxBytes: MAX_TURN_AUDIO_BYTES,
        });
      }
    }
  }

  #snapshotTurnAudio(): { audio: ArrayBuffer; bytes: number } | null {
    if (this.#turnAudioChunks.length === 0) return null;
    const chunks = this.#turnAudioChunks;
    const bytes = this.#turnAudioBytes;
    this.#turnAudioChunks = [];
    this.#turnAudioBytes = 0;

    const total = chunks.reduce((sum, b) => sum + b.byteLength, 0);
    const result = new Uint8Array(total);
    let offset = 0;
    for (const b of chunks) {
      result.set(new Uint8Array(b), offset);
      offset += b.byteLength;
    }
    return { audio: result.buffer, bytes };
  }

  #runStt(ws: WebSocket, turnId: number, audio: ArrayBuffer, audioBytes: number): void {
    const apiKey = this.env.SARVAM_API_KEY;
    if (!apiKey) {
      console.warn("[AvatarSessionDO] stt_skipped_no_api_key", {
        sessionId: this.#sessionId,
        turnId,
      });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "error",
            code: "stt_skipped",
            message: "STT not configured (no API key)",
            fatal: false,
          } satisfies ServerMessage),
        );
      }
      return;
    }

    const approxSeconds = audioBytes / 2 / 16000;

    this.#capture(ws, "stt_requested", {
      turnId,
      audioBytes,
      approxAudioSeconds: approxSeconds,
      provider: "sarvam",
    });

    const provider = new SarvamSttProvider({
      apiKey,
      model: this.env.STT_MODEL,
      language: this.env.STT_LANGUAGE_CODE,
    });

    this.ctx.waitUntil(
      provider
        .transcribePcm16(audio, { sampleRate: 16000 })
        .then((result) => {
          this.#capture(ws, "stt_completed", {
            turnId,
            provider: result.provider,
            latencyMs: result.durationMs,
            languageCode: result.languageCode,
            textLength: result.text.length,
            audioBytes,
            approxAudioSeconds: approxSeconds,
          });

          console.log("[AvatarSessionDO] stt_completed", {
            sessionId: this.#sessionId,
            turnId,
            textLength: result.text.length,
            languageCode: result.languageCode,
            latencyMs: result.durationMs,
            provider: result.provider,
          });

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "transcript.final",
                turnId,
                text: result.text,
                languageCode: result.languageCode,
                provider: result.provider,
                latencyMs: result.durationMs,
              } satisfies ServerMessage),
            );
          }

          if (result.text.trim().length > 0) {
            this.#runLlm(ws, turnId, result.text);
          }
        })
        .catch((err) => {
          const errMsg = err instanceof Error ? err.message : String(err);
          this.#capture(ws, "stt_failed", {
            turnId,
            provider: "sarvam",
            error: errMsg,
            audioBytes,
            approxAudioSeconds: approxSeconds,
          });

          console.error("[AvatarSessionDO] stt_failed", {
            sessionId: this.#sessionId,
            turnId,
            error: errMsg,
          });

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "error",
                code: "stt_failed",
                message: `STT failed: ${errMsg}`,
                fatal: false,
              } satisfies ServerMessage),
            );
          }
        }),
    );
  }

  #runLlm(ws: WebSocket, turnId: number, transcriptText: string): void {
    const baseUrl = this.env.LLM_BASE_URL;
    if (!baseUrl) {
      console.warn("[AvatarSessionDO] llm_skipped_no_base_url", {
        sessionId: this.#sessionId,
        turnId,
      });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "error",
            code: "llm_skipped",
            message: "LLM not configured (no base URL)",
            fatal: false,
          } satisfies ServerMessage),
        );
      }
      return;
    }

    const provider = new OpenAiCompatibleProvider({
      apiKey: this.env.LLM_API_KEY,
      baseUrl,
      model: this.env.LLM_MODEL,
      providerName: "cerebras-router",
    });

    const userMessage: LlmMessage = { role: "user", content: transcriptText };
    const messages: LlmMessage[] = [
      { role: "system", content: LLM_SYSTEM_PROMPT },
      ...this.#conversationHistory,
      userMessage,
    ];

    this.#capture(ws, "llm_requested", {
      turnId,
      provider: provider.providerName,
      model: provider.model,
      transcriptTextLength: transcriptText.length,
      historyMessageCount: this.#conversationHistory.length,
    });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "status",
          status: "thinking",
        } satisfies ServerMessage),
      );
    }

    this.ctx.waitUntil(
      provider
        .chat(messages)
        .then((result) => {
          this.#conversationHistory.push(userMessage);
          this.#conversationHistory.push({
            role: "assistant",
            content: result.text,
          });
          while (this.#conversationHistory.length > MAX_HISTORY_MESSAGES) {
            this.#conversationHistory.shift();
          }

          this.#capture(ws, "llm_completed", {
            turnId,
            provider: result.provider,
            model: result.model,
            latencyMs: result.latencyMs,
            responseTextLength: result.text.length,
          });

          console.log("[AvatarSessionDO] llm_completed", {
            sessionId: this.#sessionId,
            turnId,
            provider: result.provider,
            model: result.model,
            latencyMs: result.latencyMs,
            responseTextLength: result.text.length,
          });

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "assistant.text",
                turnId,
                text: result.text,
                provider: result.provider,
                latencyMs: result.latencyMs,
              } satisfies ServerMessage),
            );
          }

          this.#runTts(ws, turnId, result.text);
        })
        .catch((err) => {
          const errMsg = err instanceof Error ? err.message : String(err);
          this.#capture(ws, "llm_failed", {
            turnId,
            provider: provider.providerName,
            model: provider.model,
            error: errMsg,
          });

          console.error("[AvatarSessionDO] llm_failed", {
            sessionId: this.#sessionId,
            turnId,
            error: errMsg,
          });

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "error",
                code: "llm_failed",
                message: `LLM failed: ${errMsg}`,
                fatal: false,
              } satisfies ServerMessage),
            );
          }
        }),
    );
  }

  #runTts(ws: WebSocket, turnId: number, text: string): void {
    const enableOmniVoice = this.env.ENABLE_OMNIVOICE_TTS_ON_L4 === "true";
    const enableParallel = this.env.ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS === "true";
    const parallelTarget = enableParallel ? (this.env.OMNIVOICE_SAME_GPU_PARALLEL_TARGET ?? "l4") : "none";
    const parallelUrl = enableParallel
      ? (parallelTarget === "l40s"
          ? this.env.MODAL_PARALLEL_L40S_LIFECYCLE_URL
          : parallelTarget === "a10g"
            ? this.env.MODAL_PARALLEL_A10G_LIFECYCLE_URL
            : this.env.MODAL_PARALLEL_LIFECYCLE_URL)
      : undefined;
    const parallelTimeoutMs = enableParallel
      ? parseInt(this.env.OMNIVOICE_SAME_GPU_PARALLEL_TIMEOUT_MS ?? "12000", 10)
      : undefined;
    const provider = new BridgeTtsProvider({
      ai: this.env.AI,
      sarvamApiKey: this.env.SARVAM_API_KEY,
      languageCode: this.env.STT_LANGUAGE_CODE,
      omnivoiceTtsUrl: !enableParallel && enableOmniVoice ? this.env.OMNIVOICE_TTS_URL : undefined,
      enableOmniVoiceTts: !enableParallel && enableOmniVoice,
      enableOmniVoiceSameGpuParallel: enableParallel,
      parallelLifecycleUrl: parallelUrl,
      sameGpuParallelTimeoutMs: parallelTimeoutMs,
      gpuTarget: parallelTarget,
      sessionId: this.#state?.modalSessionId ?? undefined,
      modalRequestSigningSecret: this.env.AVTR_MODAL_REQUEST_SIGNING_SECRET,
    });

    this.#capture(ws, "tts_requested", {
      turnId,
      providerChain: provider.providerChain,
      textLength: text.length,
    });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "status",
          status: "synthesizing",
        } satisfies ServerMessage),
      );
    }

    this.ctx.waitUntil(
      provider
        .synthesize(text, {
          turnId,
          fmp4AvMode: this.#state.transport !== "webrtc_sfu" && (
            (this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true" && this.#modalFmp4StreamActive)
            || (this.#modalTimelineStreamActive && !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL)
          ),
        })
        .then((result) => {
          if (!result) {
            for (const evt of provider.omnivoiceEvents) {
              this.#capture(ws, evt.event, { turnId, ...evt.properties });
            }
            this.#capture(ws, "tts_failed", {
              turnId,
              providerChain: provider.providerChain,
              error: "all_providers_failed",
            });

            console.error("[AvatarSessionDO] tts_failed", {
              sessionId: this.#sessionId,
              turnId,
            });

            if (ws.readyState === WebSocket.OPEN) {
              ws.send(
                JSON.stringify({
                  type: "error",
                  code: "tts_failed",
                  message: "TTS failed: all providers returned null",
                  fatal: false,
                } satisfies ServerMessage),
              );
            }
            return;
          }

          for (const evt of provider.omnivoiceEvents) {
            this.#capture(ws, evt.event, { turnId, ...evt.properties });
          }

          if (result.directEnqueued) {
            // SFU Milestone 1 requires browser-playable audio bytes because audio is delivered
            // over WebSocket. Direct-enqueued TTS returns no bytes, so it is incompatible with
            // SFU mode. The root-cause fix passes fmp4AvMode=false in SFU mode; this guard
            // fails visibly if direct-enqueued still somehow occurs.
            if (this.#state.transport === "webrtc_sfu") {
              this.#capture(ws, "sfu_direct_enqueued_without_browser_audio", {
                turnId,
                provider: result.provider,
                latencyMs: result.latencyMs,
                audioDurationMs: result.directEnqueueDurationMs,
              });
              console.error("[AvatarSessionDO] sfu_direct_enqueued_without_browser_audio", {
                sessionId: this.#sessionId,
                turnId,
                provider: result.provider,
                latencyMs: result.latencyMs,
                audioDurationMs: result.directEnqueueDurationMs,
              });
              if (ws.readyState === WebSocket.OPEN) {
                this.#sendError(
                  ws,
                  "sfu_direct_enqueued",
                  "Direct-enqueued TTS audio is not supported in SFU mode; no browser audio bytes received",
                );
              }
              return;
            }

            this.#capture(ws, "tts_completed", {
              turnId,
              provider: result.provider,
              latencyMs: result.latencyMs,
              directEnqueued: true,
              directEnqueueDurationMs: result.directEnqueueDurationMs,
            });

            console.log("[AvatarSessionDO] tts_completed (direct enqueued)", {
              sessionId: this.#sessionId,
              turnId,
              provider: result.provider,
              latencyMs: result.latencyMs,
              durationMs: result.directEnqueueDurationMs,
            });

            if (ws.readyState === WebSocket.OPEN) {
              ws.send(
                JSON.stringify({
                  type: "assistant.audio",
                  turnId,
                  mimeType: result.mimeType,
                  provider: result.provider,
                  latencyMs: result.latencyMs,
                  sampleRate: result.sampleRate,
                  audioBytes: 0,
                  directEnqueued: true,
                  audioDurationMs: result.directEnqueueDurationMs,
                } satisfies ServerMessage),
              );
            }
            return;
          }

          const base64 = arrayBufferToBase64(result.audio);

          this.#capture(ws, "tts_completed", {
            turnId,
            provider: result.provider,
            latencyMs: result.latencyMs,
            audioBytes: result.audio.byteLength,
            mimeType: result.mimeType,
            sampleRate: result.sampleRate,
          });

          console.log("[AvatarSessionDO] tts_completed", {
            sessionId: this.#sessionId,
            turnId,
            provider: result.provider,
            latencyMs: result.latencyMs,
            audioBytes: result.audio.byteLength,
          });

          if (ws.readyState === WebSocket.OPEN) {
            const isAvMode = (this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true" && this.#modalFmp4StreamActive)
              || (this.#modalTimelineStreamActive && !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL);
            if (isAvMode) {
              ws.send(
                JSON.stringify({
                  type: "assistant.audio",
                  turnId,
                  mimeType: result.mimeType,
                  provider: result.provider,
                  latencyMs: result.latencyMs,
                  sampleRate: result.sampleRate,
                  audioBytes: result.audio.byteLength,
                } satisfies ServerMessage),
              );
            } else {
              ws.send(
                JSON.stringify({
                  type: "assistant.audio",
                  turnId,
                  dataBase64: base64,
                  mimeType: result.mimeType,
                  provider: result.provider,
                  latencyMs: result.latencyMs,
                  sampleRate: result.sampleRate,
                  audioBytes: result.audio.byteLength,
                } satisfies ServerMessage),
              );
            }
          }

          if (this.#state.active) {
            const audioDurationMs = result.audio.byteLength > 44
              ? ((result.audio.byteLength - 44) / (result.sampleRate * 2)) * 1000
              : (result.audio.byteLength / (result.sampleRate * 2)) * 1000;

            // Phase 4D: 4C path takes priority over the 3E fmp4_av
            // timeline. When MODAL_SESSION_STREAM_FMP4_AV_URL is set,
            // the SSE producer on Modal consumes the session's
            // _AudioHandoff queue populated by pushModalAudio (200ms
            // float32 PCM chunks). The 3E enqueueTimelineSpeech path
            // is skipped entirely.
            const use4CPath = !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL
              && !!this.env.MODAL_SESSION_START_URL;
            if (use4CPath) {
              const generatedAt = Date.now();
              this.#pushTtsChunksToModal(ws, turnId, result.audio, generatedAt);
              return;
            }

            const isFmp4Mode = (this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true" || this.env.ENABLE_MODAL_FMP4_VIDEO_STREAM === "true") && this.#modalFmp4StreamActive;
            if (isFmp4Mode) {
              const isAvMode = this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true";
              this.#capture(ws, "fmp4_speech_enqueue_requested", {
                turnId,
                audioDurationMs,
              });
              const enqueueStartAt = Date.now();
              this.#fmp4EnqueueRequestedAt.set(turnId, enqueueStartAt);
              enqueueTimelineSpeech(
                this.env,
                this.#state.modalSessionId!,
                turnId,
                base64,
                audioDurationMs,
                { useControlEndpoint: isAvMode },
              ).then((r) => {
                const enqueueCallDurationMs = Date.now() - enqueueStartAt;
                this.#capture(ws, "fmp4_enqueue_result", {
                  turnId,
                  audioDurationMs: r.data.audioDurationMs,
                  ok: r.data.ok,
                  targetUrlKind: isAvMode ? "control" : "gpu",
                  enqueueCallDurationMs,
                });
              }).catch((enqueueErr) => {
                const errMsg = enqueueErr instanceof Error ? enqueueErr.message : String(enqueueErr);
                this.#capture(ws, "fmp4_enqueue_failed", {
                  turnId,
                  error: errMsg,
                  targetUrlKind: isAvMode ? "control" : "gpu",
                });
              });
            } else if (this.env.ENABLE_MODAL_PERSISTENT_TIMELINE !== "false" && this.#modalTimelineStreamActive) {
              this.#capture(ws, "modal_timeline_speech_enqueued", {
                turnId,
                audioDurationMs,
              });
              enqueueTimelineSpeech(
                this.env,
                this.#state.modalSessionId!,
                turnId,
                base64,
                audioDurationMs,
              ).then((r) => {
                this.#capture(ws, "modal_timeline_enqueue_result", {
                  turnId,
                  queueDepth: r.data.queueDepth,
                  audioDurationMs: r.data.audioDurationMs,
                  ok: r.data.ok,
                });
              }).catch((enqueueErr) => {
                const errMsg = enqueueErr instanceof Error ? enqueueErr.message : String(enqueueErr);
                this.#capture(ws, "modal_timeline_enqueue_failed", {
                  turnId,
                  error: errMsg,
                });
                console.error("[AvatarSessionDO] modal_timeline_enqueue_failed", {
                  sessionId: this.#sessionId,
                  turnId,
                  error: errMsg,
                });
              });
            } else if (this.env.ENABLE_AVATAR_TIMELINE !== "false" && this.#avatarTimelineRunning) {
              this.#avatarSpeechQueue.push({
                turnId,
                audioBase64: base64,
                ttsProvider: result.provider,
                audioBytes: result.audio.byteLength,
                durationMs: audioDurationMs,
              });
              this.#capture(ws, "avatar_timeline_speech_enqueued", {
                turnId,
                queueLength: this.#avatarSpeechQueue.length,
                audioDurationMs,
              });
            } else if (this.env.ENABLE_SPEECH_STREAMING !== "false") {
              this.#runAvatarAudioStream(ws, turnId, base64, result.provider, result.audio.byteLength);
            } else {
              this.#runAvatarAudioRender(ws, turnId, base64, result.provider, result.audio.byteLength);
            }
          }
        })
        .catch((err) => {
          const errMsg = err instanceof Error ? err.message : String(err);
          this.#capture(ws, "tts_failed", {
            turnId,
            providerChain: "sarvam_melotts",
            error: errMsg,
          });

          console.error("[AvatarSessionDO] tts_failed", {
            sessionId: this.#sessionId,
            turnId,
            error: errMsg,
          });

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "error",
                code: "tts_failed",
                message: `TTS failed: ${errMsg}`,
                fatal: false,
              } satisfies ServerMessage),
            );
          }
        }),
    );
  }

  // Phase 5M: 4C path — push TTS audio to Modal as ONE batch request.
  // The Bridge decodes WAV, downmixes/resamples to mono 16kHz float32,
  // base64-encodes the full utterance, and sends it in a single HTTP POST
  // to the Modal push-audio-batch endpoint. Modal splits into 200ms chunks
  // and pushes them into the session's _AudioHandoff in-process.
  #pushTtsChunksToModal(
    ws: WebSocket,
    turnId: number,
    audioBytes: ArrayBuffer,
    generatedAt: number,
  ): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    this.ctx.waitUntil((async () => {
      try {
        const u8 = new Uint8Array(audioBytes);
        const decoded = decodeWavPcm16Inline(u8);
        if (decoded.pcm.length === 0) {
          this.#capture(ws, "fmp4_push_audio_skipped_empty", { turnId });
          return;
        }

        let f32 = new Float32Array(decoded.pcm.length);
        for (let i = 0; i < decoded.pcm.length; i++) {
          f32[i] = decoded.pcm[i] / 32768;
        }

        if (decoded.channels > 1) {
          const mono = new Float32Array(Math.floor(f32.length / decoded.channels));
          for (let i = 0; i < mono.length; i++) {
            let sum = 0;
            for (let c = 0; c < decoded.channels; c++) {
              sum += f32[i * decoded.channels + c];
            }
            mono[i] = sum / decoded.channels;
          }
          f32 = mono;
        }

        const actualRate = decoded.sampleRate;
        if (actualRate !== 16000) {
          this.#capture(ws, "fmp4_push_audio_resampling", {
            turnId,
            fromRate: actualRate,
            toRate: 16000,
            qualityNote: "linear_prototype",
          });
          f32 = Float32Array.from(resampleLinear(f32, actualRate, 16000));
        }

        const nChunks = Math.max(1, Math.ceil(f32.length / 3200));
        this.#capture(ws, "fmp4_push_audio_batch_started", {
          turnId,
          totalSamples: f32.length,
          expectedChunks: nChunks,
          sampleRate: 16000,
        });

        const b64 = float32ToBase64(f32);
        const enqueueAt = Date.now();

        const r = await pushModalAudioBatch(this.env, {
          session_id: modalSessionId,
          audio_data_b64: b64,
          audio_mode: "pcm_f32_16k",
          turn_id: turnId,
          generated_at: generatedAt,
          enqueue_at: enqueueAt,
        });

        const data = r.data as Record<string, unknown>;
        if (data && data.ok) {
          this.#capture(ws, "fmp4_push_audio_batch_completed", {
            turnId,
            samples: data.samples,
            chunks: data.chunks,
            pushed: data.pushed,
            enqueueMs: data.enqueue_ms,
          });
          if (this.#state.transport === "webrtc_sfu") {
            try {
              ws.send(JSON.stringify({
                type: "avatar.sfu.speech_queued",
                turnId,
                chunks: (data.chunks as number) ?? 0,
                samples: (data.samples as number) ?? 0,
              } satisfies ServerMessage));
              this.#capture(ws, "sfu_speech_queued_sent", {
                turnId,
                chunks: data.chunks,
              });
            } catch {
              // ws may be closed; non-fatal
            }
          }
        } else {
          const error = (data?.error as string) ?? "unknown";
          this.#capture(ws, "fmp4_push_audio_batch_failed", {
            turnId,
            error,
          });
          console.error("[AvatarSessionDO] fmp4_push_audio_batch_failed", {
            sessionId: this.#sessionId,
            turnId,
            error,
          });
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        this.#capture(ws, "fmp4_push_audio_batch_failed", { turnId, error: msg });
        console.error("[AvatarSessionDO] fmp4_push_audio_batch_failed", {
          sessionId: this.#sessionId,
          turnId,
          error: msg,
        });
      }
    })());
  }

  // Future: once Modal exposes a streaming endpoint (StreamingResponse/NDJSON),
  // forward frames as they arrive instead of receiving full batch then chunking.
  // Modal FastAPI endpoints support StreamingResponse — current batch approach is by design.
  // Speech render acquires #modalRenderBusy lock — idle loop pauses while speech renders.
  // After sending chunks, DO waits for browser playback ack to release speech active state.
  #runAvatarAudioRender(
    ws: WebSocket,
    turnId: number,
    audioWavBase64: string,
    ttsProvider: string,
    audioBytes: number,
  ): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    const FRAMES_PER_CHUNK = 25;
    const PLAYBACK_ACK_FALLBACK_MS = 30_000;
    const timeSinceSessionReadyMs = this.#sessionReadyAt ? Date.now() - this.#sessionReadyAt : undefined;
    const speechRenderRequestedAt = Date.now();

    this.#assistantSpeechActive = true;
    this.#capture(ws, "avatar_audio_render_requested", {
      turnId,
      ttsProvider,
      audioBytes,
      modalSessionId,
      timeSinceSessionReadyMs,
    });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "status",
          status: "rendering_avatar_audio",
        } satisfies ServerMessage),
      );
    }

    console.log("[AvatarSessionDO] avatar_audio_render_requested", {
      sessionId: this.#sessionId,
      turnId,
      ttsProvider,
      audioBytes,
      modalSessionId,
    });

    const doRender = async () => {
      await this.#acquireModalLock();

      try {
        const modalCallStart = Date.now();
        const { data: result, meta: callMeta } = await renderAudio(this.env, modalSessionId, audioWavBase64, 60.0);
        const bridgeModalCallDurationMs = Date.now() - modalCallStart;

        if (!result.available || !result.frames || result.frames.length === 0) {
          const errMsg = result.error || "No frames returned";
          const modalSessionLost = errMsg.includes("No active session");

          this.#capture(ws, "avatar_audio_render_failed", {
            turnId,
            error: errMsg,
            ttsProvider,
            modalSessionId,
            modalSessionLost,
            timeSinceSessionReadyMs,
            bridgeModalCallDurationMs,
            responseJsonParseMs: callMeta.responseJsonParseMs,
          });

          console.error("[AvatarSessionDO] avatar_audio_render_failed", {
            sessionId: this.#sessionId,
            turnId,
            error: errMsg,
            modalSessionLost,
            bridgeModalCallDurationMs,
          });

          if (modalSessionLost) {
            this.#state.active = false;
            await this.#saveState();
            console.warn("[AvatarSessionDO] Modal session lost — marking bridge state inactive", {
              sessionId: this.#sessionId,
              modalSessionId,
            });
          }

          if (ws.readyState === WebSocket.OPEN) {
            ws.send(
              JSON.stringify({
                type: "error",
                code: "avatar_render_failed",
                message: `Avatar render failed: ${errMsg}`,
                fatal: false,
              } satisfies ServerMessage),
            );
          }
          return;
        }

        const totalFrames = result.frames.length;
        const fps = result.fps ?? 25;
        const totalChunks = Math.ceil(totalFrames / FRAMES_PER_CHUNK);
        let payloadBytes = 0;
        let firstChunkSent = false;
        let firstChunkStringifyMs = 0;
        let firstChunkSendMs = 0;
        let speechStartStringifyMs = 0;
        let speechStartSendMs = 0;

        if (ws.readyState === WebSocket.OPEN) {
          const ssT0 = Date.now();
          const speechStartMsg = JSON.stringify({
            type: "avatar.speech.start",
            turnId,
            totalFrames,
            fps,
            renderedAudioDurationMs: result.audioDurationMs,
            originalAudioDurationMs: result.originalDurationMs,
          } satisfies ServerMessage);
          speechStartStringifyMs = Date.now() - ssT0;

          const ssSend0 = Date.now();
          ws.send(speechStartMsg);
          speechStartSendMs = Date.now() - ssSend0;

          for (let i = 0; i < totalChunks; i++) {
            const start = i * FRAMES_PER_CHUNK;
            const end = Math.min(start + FRAMES_PER_CHUNK, totalFrames);
            const chunkFrames = result.frames.slice(start, end);
            const isLast = i === totalChunks - 1;

            const stT0 = Date.now();
            const msg = JSON.stringify({
              type: "avatar.speech.chunk",
              turnId,
              chunkIndex: i,
              frames: chunkFrames,
              isLast,
            } satisfies ServerMessage);
            const stringifyMs = Date.now() - stT0;

            const sendT0 = Date.now();
            ws.send(msg);
            const sendCallMs = Date.now() - sendT0;
            payloadBytes += msg.length;

            if (!firstChunkSent) {
              firstChunkSent = true;
              firstChunkStringifyMs = stringifyMs;
              firstChunkSendMs = sendCallMs;
              this.#speechFirstFrameSentAt = Date.now();
              this.#capture(ws, "avatar_speech_first_frame_latency", {
                modalSessionId,
                turnId,
                latencyMs: Date.now() - speechRenderRequestedAt,
                timeSinceSessionReadyMs,
                bridgeModalCallDurationMs,
                responseJsonParseMs: callMeta.responseJsonParseMs,
                modalReportedModelRenderDurationMs: result.modelRenderDurationMs,
                modalReportedJpegEncodeDurationMs: result.jpegEncodeDurationMs,
                modalReportedBase64EncodeDurationMs: result.base64EncodeDurationMs,
                modalReportedFrameConvertDurationMs: result.frameConvertDurationMs,
                modalReportedTotalHandlerDurationMs: result.totalHandlerDurationMs,
                modalReportedOverheadMs: result.overheadMs,
                firstChunkStringifyMs,
                firstChunkSendCallMs: firstChunkSendMs,
                speechStartStringifyMs,
                speechStartSendCallMs: speechStartSendMs,
                firstChunkPayloadChars: msg.length,
              });
            }
          }
        }

        this.#capture(ws, "avatar_audio_render_completed", {
          turnId,
          ttsProvider,
          audioDurationMs: result.audioDurationMs,
          framesRendered: totalFrames,
          fps,
          renderDurationMs: result.renderDurationMs,
          modelRenderDurationMs: result.modelRenderDurationMs,
          jpegEncodeDurationMs: result.jpegEncodeDurationMs,
          base64EncodeDurationMs: result.base64EncodeDurationMs,
          frameConvertDurationMs: result.frameConvertDurationMs,
          totalHandlerDurationMs: result.totalHandlerDurationMs,
          overheadMs: result.overheadMs,
          avgJpegEncodeMsPerFrame: result.avgJpegEncodeMsPerFrame,
          modalEncodedBytes: result.encodedBytes,
          modalBase64Chars: result.base64Chars,
          frameWidth: result.frameWidth,
          frameHeight: result.frameHeight,
          truncated: result.truncated,
          maxDurationS: result.maxDurationS,
          originalDurationMs: result.originalDurationMs,
          chunkCount: totalChunks,
          payloadBytes,
          modalSessionId,
          timeSinceSessionReadyMs,
          bridgeModalCallDurationMs,
          responseJsonParseMs: callMeta.responseJsonParseMs,
          speechStartStringifyMs,
          speechStartSendCallMs: speechStartSendMs,
          firstChunkStringifyMs,
          firstChunkSendCallMs: firstChunkSendMs,
        });

        console.log("[AvatarSessionDO] avatar_audio_render_completed", {
          sessionId: this.#sessionId,
          turnId,
          framesRendered: totalFrames,
          fps,
          renderDurationMs: result.renderDurationMs,
          modelRenderDurationMs: result.modelRenderDurationMs,
          jpegEncodeDurationMs: result.jpegEncodeDurationMs,
          base64EncodeDurationMs: result.base64EncodeDurationMs,
          frameConvertDurationMs: result.frameConvertDurationMs,
          totalHandlerDurationMs: result.totalHandlerDurationMs,
          overheadMs: result.overheadMs,
          audioDurationMs: result.audioDurationMs,
          originalDurationMs: result.originalDurationMs,
          truncated: result.truncated,
          chunkCount: totalChunks,
          payloadBytes,
          bridgeModalCallDurationMs,
          responseJsonParseMs: callMeta.responseJsonParseMs,
        });

        this.#pendingPlaybackAckTurnId = turnId;
        this.#playbackAckTimeoutMs = PLAYBACK_ACK_FALLBACK_MS;
        this.ctx.waitUntil(
          (async () => {
            const deadline = Date.now() + PLAYBACK_ACK_FALLBACK_MS;
            while (this.#pendingPlaybackAckTurnId === turnId && Date.now() < deadline) {
              await scheduler.wait(200);
            }
            if (this.#pendingPlaybackAckTurnId === turnId) {
              this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                turnId,
                fallbackMs: PLAYBACK_ACK_FALLBACK_MS,
              });
              this.#pendingPlaybackAckTurnId = null;
              this.#assistantSpeechActive = false;
            }
          })(),
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        const modalSessionLost = errMsg.includes("No active session");

        this.#capture(ws, "avatar_audio_render_failed", {
          turnId,
          error: errMsg,
          ttsProvider,
          modalSessionId,
          modalSessionLost,
          timeSinceSessionReadyMs,
        });

        console.error("[AvatarSessionDO] avatar_audio_render_failed", {
          sessionId: this.#sessionId,
          turnId,
          error: errMsg,
          modalSessionLost,
        });

        if (modalSessionLost) {
          this.#state.active = false;
          await this.#saveState();
          console.warn("[AvatarSessionDO] Modal session lost — marking bridge state inactive", {
            sessionId: this.#sessionId,
            modalSessionId,
          });
        }

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "error",
              code: "avatar_render_failed",
              message: `Avatar render failed: ${errMsg}`,
              fatal: false,
            } satisfies ServerMessage),
          );
        }
      } finally {
        this.#releaseModalLock();
      }
    };

    this.ctx.waitUntil(doRender());
  }

  #runAvatarAudioStream(
    ws: WebSocket,
    turnId: number,
    audioWavBase64: string,
    ttsProvider: string,
    audioBytes: number,
  ): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    const PLAYBACK_ACK_FALLBACK_MS = 30_000;
    const timeSinceSessionReadyMs = this.#sessionReadyAt ? Date.now() - this.#sessionReadyAt : undefined;
    const speechRenderRequestedAt = Date.now();

    this.#assistantSpeechActive = true;
    this.#activeSpeechRenderTurnId = turnId;

    this.#capture(ws, "avatar_audio_stream_requested", {
      turnId,
      ttsProvider,
      audioBytes,
      modalSessionId,
      timeSinceSessionReadyMs,
    });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "status",
          status: "rendering_avatar_audio",
        } satisfies ServerMessage),
      );
    }

    console.log("[AvatarSessionDO] avatar_audio_stream_requested", {
      sessionId: this.#sessionId,
      turnId,
      ttsProvider,
      audioBytes,
      modalSessionId,
    });

    let streamStarted = false;
    let chunksForwarded = 0;
    let firstChunkSentAt = 0;

    const doStream = async () => {
      await this.#acquireModalLock();

      try {
        const streamStartAt = Date.now();

        await renderAudioStream(
          this.env,
          modalSessionId,
          audioWavBase64,
          60.0,
          80,
          {
            onStart: (event) => {
              if (this.#activeSpeechRenderTurnId !== turnId) return;

              streamStarted = true;
              this.#capture(ws, "avatar_audio_stream_started", {
                turnId,
                modalSessionId,
                timeToStreamStartMs: Date.now() - speechRenderRequestedAt,
                fps: event.fps,
                totalSteps: event.totalSteps,
                originalDurationMs: event.originalAudioDurationMs,
                renderedDurationMs: event.renderedAudioDurationMs,
              });

              if (ws.readyState === WebSocket.OPEN) {
                const msg = JSON.stringify({
                  type: "avatar.speech.start",
                  turnId,
                  totalFrames: event.totalSteps * (event.fps > 0 ? Math.round(event.renderedAudioDurationMs / (1000 / event.fps) / event.totalSteps) : 5),
                  fps: event.fps,
                  renderedAudioDurationMs: event.renderedAudioDurationMs,
                  originalAudioDurationMs: event.originalAudioDurationMs,
                } satisfies ServerMessage);
                ws.send(msg);
              }
            },

            onChunk: (event) => {
              if (this.#activeSpeechRenderTurnId !== turnId) return;
              if (ws.readyState !== WebSocket.OPEN) return;

              chunksForwarded++;

              const stT0 = Date.now();
              const msg = JSON.stringify({
                type: "avatar.speech.chunk",
                turnId,
                chunkIndex: event.chunkIndex,
                frames: event.frames,
                isLast: event.isLast,
              } satisfies ServerMessage);
              const stringifyMs = Date.now() - stT0;

              const sendT0 = Date.now();
              ws.send(msg);
              const wsSendCallMs = Date.now() - sendT0;

              if (firstChunkSentAt === 0) {
                firstChunkSentAt = Date.now();
                this.#speechFirstFrameSentAt = Date.now();
                this.#capture(ws, "avatar_audio_stream_first_chunk", {
                  turnId,
                  modalSessionId,
                  timeToFirstChunkMs: Date.now() - speechRenderRequestedAt,
                  chunkIndex: event.chunkIndex,
                  framesInChunk: event.frames.length,
                  payloadBytes: msg.length,
                  stringifyMs,
                  wsSendCallMs,
                });
              }

              this.#capture(ws, "avatar_audio_stream_chunk_forwarded", {
                turnId,
                chunkIndex: event.chunkIndex,
                framesInChunk: event.frames.length,
                framesRendered: event.framesRendered,
                isLast: event.isLast,
                payloadBytes: msg.length,
                stringifyMs,
                wsSendCallMs,
                chunkRenderMs: event.timing.chunkRenderMs,
                chunkEncodeMs: event.timing.chunkEncodeMs,
                chunkBase64Chars: event.timing.chunkBase64Chars,
              });
            },

            onEnd: (event) => {
              if (this.#activeSpeechRenderTurnId !== turnId) return;

              const totalStreamMs = Date.now() - speechRenderRequestedAt;

              this.#capture(ws, "avatar_audio_stream_completed", {
                turnId,
                ttsProvider,
                modalSessionId,
                totalStreamMs,
                timeToFirstChunkMs: firstChunkSentAt ? firstChunkSentAt - speechRenderRequestedAt : 0,
                framesRendered: event.framesRendered,
                modelRenderDurationMs: event.modelRenderDurationMs,
                totalEncodeMs: event.totalEncodeMs,
                totalHandlerDurationMs: event.totalHandlerDurationMs,
                firstChunkReadyMs: event.firstChunkReadyMs,
                chunksForwarded,
                truncated: event.truncated,
                originalDurationMs: event.originalDurationMs,
                renderedAudioDurationMs: event.renderedAudioDurationMs,
              });

              console.log("[AvatarSessionDO] avatar_audio_stream_completed", {
                sessionId: this.#sessionId,
                turnId,
                framesRendered: event.framesRendered,
                totalStreamMs,
                modelRenderDurationMs: event.modelRenderDurationMs,
                totalEncodeMs: event.totalEncodeMs,
                totalHandlerDurationMs: event.totalHandlerDurationMs,
                firstChunkReadyMs: event.firstChunkReadyMs,
                chunksForwarded,
                truncated: event.truncated,
              });
            },
          },
        );

        if (chunksForwarded === 0) {
          console.warn("[AvatarSessionDO] Stream produced no chunks, falling back to batch render", {
            sessionId: this.#sessionId,
            turnId,
          });
          this.#releaseModalLock();
          await this.#fallbackBatchRender(ws, turnId, audioWavBase64, ttsProvider, audioBytes, speechRenderRequestedAt, "stream_produced_no_chunks");
          return;
        }

        this.#pendingPlaybackAckTurnId = turnId;
        this.#playbackAckTimeoutMs = PLAYBACK_ACK_FALLBACK_MS;
        this.ctx.waitUntil(
          (async () => {
            const deadline = Date.now() + PLAYBACK_ACK_FALLBACK_MS;
            while (this.#pendingPlaybackAckTurnId === turnId && Date.now() < deadline) {
              await scheduler.wait(200);
            }
            if (this.#pendingPlaybackAckTurnId === turnId) {
              this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                turnId,
                fallbackMs: PLAYBACK_ACK_FALLBACK_MS,
                source: "stream",
              });
              this.#pendingPlaybackAckTurnId = null;
              this.#assistantSpeechActive = false;
              this.#activeSpeechRenderTurnId = null;
            }
          })(),
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error("[AvatarSessionDO] avatar_audio_stream_failed", {
          sessionId: this.#sessionId,
          turnId,
          error: errMsg,
          streamStarted,
          chunksForwarded,
        });

        this.#capture(ws, "avatar_audio_stream_failed", {
          turnId,
          error: errMsg,
          streamStarted,
          chunksForwarded,
          modalSessionId,
        });

        if (!streamStarted) {
          this.#releaseModalLock();
          await this.#fallbackBatchRender(ws, turnId, audioWavBase64, ttsProvider, audioBytes, speechRenderRequestedAt, `stream_error: ${errMsg}`);
          return;
        }

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "error",
              code: "avatar_render_failed",
              message: `Avatar stream failed after partial chunks: ${errMsg}`,
              fatal: false,
            } satisfies ServerMessage),
          );
        }
      } finally {
        this.#releaseModalLock();
      }
    };

    this.ctx.waitUntil(doStream());
  }

  async #fallbackBatchRender(
    ws: WebSocket,
    turnId: number,
    audioWavBase64: string,
    ttsProvider: string,
    audioBytes: number,
    speechRenderRequestedAt: number,
    reason: string,
  ): Promise<void> {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    this.#capture(ws, "avatar_audio_stream_fallback_to_batch", {
      turnId,
      reason,
      modalSessionId,
    });

    console.log("[AvatarSessionDO] Falling back to batch render", {
      sessionId: this.#sessionId,
      turnId,
      reason,
    });

    await this.#acquireModalLock();
    try {
      const modalCallStart = Date.now();
      const { data: result, meta: callMeta } = await renderAudio(this.env, modalSessionId, audioWavBase64, 60.0);
      const bridgeModalCallDurationMs = Date.now() - modalCallStart;

      if (!result.available || !result.frames || result.frames.length === 0) {
        const errMsg = result.error || "No frames returned";
        this.#capture(ws, "avatar_audio_render_failed", {
          turnId,
          error: errMsg,
          ttsProvider,
          modalSessionId,
          fallbackReason: reason,
          bridgeModalCallDurationMs,
          responseJsonParseMs: callMeta.responseJsonParseMs,
        });

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "error",
              code: "avatar_render_failed",
              message: `Avatar render failed: ${errMsg}`,
              fatal: false,
            } satisfies ServerMessage),
          );
        }
        return;
      }

      const totalFrames = result.frames.length;
      const fps = result.fps ?? 25;
      const FRAMES_PER_CHUNK = 25;
      const totalChunks = Math.ceil(totalFrames / FRAMES_PER_CHUNK);

      if (ws.readyState === WebSocket.OPEN) {
        const speechStartMsg = JSON.stringify({
          type: "avatar.speech.start",
          turnId,
          totalFrames,
          fps,
          renderedAudioDurationMs: result.audioDurationMs,
          originalAudioDurationMs: result.originalDurationMs,
        } satisfies ServerMessage);
        ws.send(speechStartMsg);

        for (let i = 0; i < totalChunks; i++) {
          const start = i * FRAMES_PER_CHUNK;
          const end = Math.min(start + FRAMES_PER_CHUNK, totalFrames);
          const chunkFrames = result.frames.slice(start, end);
          const isLast = i === totalChunks - 1;
          const msg = JSON.stringify({
            type: "avatar.speech.chunk",
            turnId,
            chunkIndex: i,
            frames: chunkFrames,
            isLast,
          } satisfies ServerMessage);
          ws.send(msg);
        }
      }

      this.#capture(ws, "avatar_audio_render_completed", {
        turnId,
        ttsProvider,
        audioDurationMs: result.audioDurationMs,
        framesRendered: totalFrames,
        fps,
        renderDurationMs: result.renderDurationMs,
        modelRenderDurationMs: result.modelRenderDurationMs,
        jpegEncodeDurationMs: result.jpegEncodeDurationMs,
        base64EncodeDurationMs: result.base64EncodeDurationMs,
        frameConvertDurationMs: result.frameConvertDurationMs,
        totalHandlerDurationMs: result.totalHandlerDurationMs,
        overheadMs: result.overheadMs,
        bridgeModalCallDurationMs,
        responseJsonParseMs: callMeta.responseJsonParseMs,
        fallbackReason: reason,
        chunksForwarded: 0,
        totalStreamMs: Date.now() - speechRenderRequestedAt,
      });
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      this.#capture(ws, "avatar_audio_render_failed", {
        turnId,
        error: errMsg,
        ttsProvider,
        modalSessionId,
        fallbackReason: reason,
      });
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: "error",
            code: "avatar_render_failed",
            message: `Avatar render failed: ${errMsg}`,
            fatal: false,
          } satisfies ServerMessage),
        );
      }
    } finally {
      this.#releaseModalLock();
    }
  }

  #runModalTimelineStream(ws: WebSocket): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId || this.#modalTimelineStreamActive) return;

    this.#modalTimelineStreamActive = true;
    this.#modalTimelineAbort = false;
    this.#modalTimelineAbortController = new AbortController();
    this.#modalTimelineIdleStarted = false;
    this.#modalTimelineFrameCount = 0;

    this.#capture(ws, "modal_timeline_stream_started", { modalSessionId });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "avatar.timeline.started", fps: 25, mode: "jpeg_timeline" } satisfies ServerMessage));
    }

    const PLAYBACK_ACK_TIMEOUT_MS = 30_000;

    this.ctx.waitUntil((async () => {
      try {
        await startModalTimelineStream(
          this.env,
          modalSessionId,
          {
            onStart: (event) => {
              console.log("[AvatarSessionDO] modal_timeline_start", {
                sessionId: this.#sessionId,
                fps: event.fps,
              });
            },

            onFrame: (event) => {
              if (this.#modalTimelineAbort || ws.readyState !== WebSocket.OPEN) return;

              this.#modalTimelineFrameCount++;

              if (event.mode === "speech") {
                if (!this.#assistantSpeechActive && event.isSpeechStart) {
                  this.#assistantSpeechActive = true;
                  this.#activeSpeechRenderTurnId = event.turnId;
                  this.#speechFirstFrameSentAt = Date.now();

                  ws.send(JSON.stringify({
                    type: "avatar.speech.start",
                    turnId: event.turnId!,
                    totalFrames: 0,
                    fps: event.fps,
                  } satisfies ServerMessage));

                  this.#capture(ws, "modal_timeline_first_speech_frame", {
                    turnId: event.turnId,
                    segmentId: event.segmentId,
                    chunkIndex: event.chunkIndex,
                    enqueueToFirstFrameMs: event.enqueueToFirstFrameMs ?? null,
                  });
                }

                ws.send(JSON.stringify({
                  type: "avatar.speech.chunk",
                  turnId: event.turnId!,
                  chunkIndex: event.chunkIndex,
                  frames: event.frames,
                  isLast: event.isSpeechEnd,
                } satisfies ServerMessage));

                if (event.isSpeechEnd) {
                  this.#capture(ws, "modal_timeline_speech_completed", {
                    turnId: event.turnId,
                    segmentId: event.segmentId,
                    chunkIndex: event.chunkIndex,
                    speechRemainingMs: event.timing.speechRemainingMs,
                  });

                  this.#pendingPlaybackAckTurnId = event.turnId;
                  this.#playbackAckTimeoutMs = PLAYBACK_ACK_TIMEOUT_MS;
                  const ackDeadline = Date.now() + PLAYBACK_ACK_TIMEOUT_MS;
                  const waitForAck = async () => {
                    while (this.#pendingPlaybackAckTurnId === event.turnId && Date.now() < ackDeadline) {
                      await scheduler.wait(200);
                    }
                    if (this.#pendingPlaybackAckTurnId === event.turnId) {
                      this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                        turnId: event.turnId,
                        source: "modal_timeline",
                      });
                      this.#pendingPlaybackAckTurnId = null;
                    }
                    this.#assistantSpeechActive = false;
                    this.#activeSpeechRenderTurnId = null;
                    this.#modalTimelineIdleStarted = false;
                  };
                  this.ctx.waitUntil(waitForAck());
                }

                this.#capture(ws, "modal_timeline_frame_forwarded", {
                  mode: "speech",
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                  chunkIndex: event.chunkIndex,
                  framesInChunk: event.frames.length,
                  isSpeechStart: event.isSpeechStart,
                  isSpeechEnd: event.isSpeechEnd,
                  chunkRenderMs: event.timing.chunkRenderMs,
                  chunkEncodeMs: event.timing.chunkEncodeMs,
                  queueDepth: event.timing.queueDepth,
                  speechRemainingMs: event.timing.speechRemainingMs,
                  enqueueToFirstFrameMs: event.enqueueToFirstFrameMs ?? null,
                });

              } else {
                if (this.#assistantSpeechActive) return;

                if (!this.#modalTimelineIdleStarted) {
                  this.#modalTimelineIdleStarted = true;
                  ws.send(JSON.stringify({
                    type: "avatar.idle.start",
                    totalFrames: 0,
                    fps: event.fps,
                  } satisfies ServerMessage));
                }

                ws.send(JSON.stringify({
                  type: "avatar.idle.chunk",
                  chunkIndex: event.chunkIndex,
                  frames: event.frames,
                  isLast: false,
                  live: true,
                } satisfies ServerMessage));

                if (this.#modalTimelineFrameCount % 25 === 0) {
                  this.#capture(ws, "modal_timeline_frame_forwarded", {
                    mode: "idle",
                    chunkIndex: event.chunkIndex,
                    framesInChunk: event.frames.length,
                    totalFramesForwarded: this.#modalTimelineFrameCount,
                    chunkRenderMs: event.timing.chunkRenderMs,
                    chunkEncodeMs: event.timing.chunkEncodeMs,
                  });
                }
              }

              this.#lastAvatarFrameAt = Date.now();
            },

            onEnd: (event) => {
              console.log("[AvatarSessionDO] modal_timeline_end", {
                sessionId: this.#sessionId,
                chunksRendered: event.chunksRendered,
                stopReason: event.stopReason,
              });
            },
          },
          this.#modalTimelineAbortController?.signal,
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error("[AvatarSessionDO] modal_timeline_stream_error", {
          sessionId: this.#sessionId,
          error: errMsg,
        });
        this.#capture(ws, "modal_timeline_stream_failed", {
          error: errMsg,
          framesForwarded: this.#modalTimelineFrameCount,
        });
      } finally {
        this.#modalTimelineStreamActive = false;
        this.#modalTimelineAbortController = null;

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "avatar.timeline.stopped",
            reason: "stream_ended",
          } satisfies ServerMessage));
        }

        this.#capture(ws, "modal_timeline_stream_stopped", {
          framesForwarded: this.#modalTimelineFrameCount,
        });
      }
    })());
  }

  #runModalTimelineFmp4Stream(ws: WebSocket): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId || this.#modalFmp4StreamActive) return;

    this.#modalFmp4StreamActive = true;
    this.#modalTimelineAbort = false;
    this.#modalTimelineAbortController = new AbortController();
    this.#modalFmp4SegmentCount = 0;

    this.#capture(ws, "avatar_fmp4_stream_started", { modalSessionId });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "avatar.timeline.started", fps: 25, mode: "fmp4_video" } satisfies ServerMessage));
    }

    const PLAYBACK_ACK_TIMEOUT_MS = 30_000;

    this.ctx.waitUntil((async () => {
      try {
        await startModalTimelineFmp4Stream(
          this.env,
          modalSessionId,
          {
            onInit: (event) => {
              if (this.#modalTimelineAbort || ws.readyState !== WebSocket.OPEN) return;

              this.#capture(ws, "avatar_fmp4_init_forwarded", {
                byteLength: event.byteLength,
                timeToInitMs: event.timing.timeToInitMs,
              });

              ws.send(JSON.stringify({
                type: "avatar.fmp4.init",
                dataBase64: event.dataBase64,
                byteLength: event.byteLength,
                timingMs: event.timing.timeToInitMs,
              } satisfies ServerMessage));
            },

            onMedia: (event) => {
              if (this.#modalTimelineAbort || ws.readyState !== WebSocket.OPEN) return;

              this.#modalFmp4SegmentCount++;

              if (event.mode === "speech" && event.isSpeechStart) {
                this.#assistantSpeechActive = true;
                this.#activeSpeechRenderTurnId = event.turnId ?? null;
                this.#speechFirstFrameSentAt = Date.now();

                ws.send(JSON.stringify({
                  type: "avatar.speech.start",
                  turnId: event.turnId!,
                  totalFrames: 0,
                  fps: 25,
                } satisfies ServerMessage));

                this.#capture(ws, "avatar_fmp4_speech_start", {
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                  chunkIndex: event.chunkIndex,
                });
              }

              ws.send(JSON.stringify({
                type: "avatar.fmp4.segment",
                dataBase64: event.dataBase64,
                byteLength: event.byteLength,
                sequence: event.sequence,
                mode: event.mode,
                isSpeechStart: event.isSpeechStart ?? false,
                isSpeechEnd: event.isSpeechEnd ?? false,
                turnId: event.turnId ?? null,
                segmentId: event.segmentId ?? 0,
                chunkIndex: event.chunkIndex ?? 0,
                chunkRenderMs: event.timing?.chunkRenderMs ?? 0,
                feedMs: event.timing?.feedMs ?? 0,
                framesFed: event.timing?.framesFed ?? 0,
              } satisfies ServerMessage));

              if (event.mode === "speech" && event.isSpeechEnd) {
                this.#capture(ws, "avatar_fmp4_speech_end", {
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                });

                this.#pendingPlaybackAckTurnId = event.turnId ?? null;
                this.#playbackAckTimeoutMs = PLAYBACK_ACK_TIMEOUT_MS;
                const ackDeadline = Date.now() + PLAYBACK_ACK_TIMEOUT_MS;
                const waitForAck = async () => {
                  while (this.#pendingPlaybackAckTurnId === event.turnId && Date.now() < ackDeadline) {
                    await scheduler.wait(200);
                  }
                  if (this.#pendingPlaybackAckTurnId === event.turnId) {
                    this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                      turnId: event.turnId,
                      source: "modal_fmp4",
                    });
                    this.#pendingPlaybackAckTurnId = null;
                  }
                  this.#assistantSpeechActive = false;
                  this.#activeSpeechRenderTurnId = null;
                  this.#modalTimelineIdleStarted = false;
                };
                this.ctx.waitUntil(waitForAck());
              }

              if (this.#modalFmp4SegmentCount % 10 === 0) {
                this.#capture(ws, "avatar_fmp4_segment_forwarded", {
                  sequence: event.sequence,
                  mode: event.mode,
                  byteLength: event.byteLength,
                  totalSegments: this.#modalFmp4SegmentCount,
                });
              }

              this.#lastAvatarFrameAt = Date.now();
            },

            onEnd: (event) => {
              console.log("[AvatarSessionDO] fmp4_stream_end", {
                sessionId: this.#sessionId,
                chunksRendered: event.chunksRendered,
                stopReason: event.stopReason,
              });
            },
          },
          this.#modalTimelineAbortController?.signal,
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error("[AvatarSessionDO] fmp4_stream_error", {
          sessionId: this.#sessionId,
          error: errMsg,
        });
        this.#capture(ws, "avatar_fmp4_stream_failed", {
          error: errMsg,
          segmentsForwarded: this.#modalFmp4SegmentCount,
        });
      } finally {
        this.#modalFmp4StreamActive = false;
        this.#modalTimelineAbortController = null;

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "avatar.timeline.stopped",
            reason: "fmp4_stream_ended",
          } satisfies ServerMessage));
        }

        this.#capture(ws, "avatar_fmp4_stream_stopped", {
          segmentsForwarded: this.#modalFmp4SegmentCount,
        });
      }
    })());
  }

  #runModalTimelineFmp4AvStream(ws: WebSocket): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId || this.#modalFmp4StreamActive) return;

    this.#modalFmp4StreamActive = true;
    this.#modalTimelineAbort = false;
    this.#modalTimelineAbortController = new AbortController();
    this.#modalFmp4SegmentCount = 0;
    this.#fmp4StreamSummaryCaptured = false;
    this.#fmp4StreamEndResolve = null;
    this.#fmp4LastStreamMode = null;
    this.#fmp4LastStreamSequence = null;
    this.#avStreamMediaReceived = 0;
    this.#avStreamMediaSendAttempted = 0;
    this.#avStreamMediaSendSucceeded = 0;
    this.#avStreamMediaSendFailed = 0;
    this.#avStreamLastSequenceReceived = null;
    this.#avStreamLastSequenceSent = null;
    this.#avStreamLastMediaSentAt = 0;

    const verboseDelivery = this.env.ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY === "true";

    this.#capture(ws, "avatar_fmp4_av_stream_started", { modalSessionId, gpuTarget: getParallelTarget(this.env), verboseDelivery });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "avatar.timeline.started", fps: 25, mode: "fmp4_av" } satisfies ServerMessage));
    }

    const PLAYBACK_ACK_TIMEOUT_MS = 30_000;

    this.ctx.waitUntil((async () => {
      try {
        await startModalTimelineFmp4AvStream(
          this.env,
          modalSessionId,
          {
            onInit: (event) => {
              if (this.#modalTimelineAbort || ws.readyState !== WebSocket.OPEN) return;

              this.#capture(ws, "avatar_fmp4_av_init_forwarded", {
                byteLength: event.byteLength,
                timeToInitMs: event.timing.timeToInitMs,
                wsReadyState: ws.readyState,
              });

              try {
                ws.send(JSON.stringify({
                  type: "avatar.fmp4.init",
                  dataBase64: event.dataBase64,
                  byteLength: event.byteLength,
                  timingMs: event.timing.timeToInitMs,
                } satisfies ServerMessage));
              } catch (sendErr) {
                this.#capture(ws, "avatar_fmp4_av_init_send_failed", {
                  error: sendErr instanceof Error ? sendErr.message : String(sendErr),
                  wsReadyState: ws.readyState,
                });
                this.#modalTimelineAbort = true;
              }
            },

            onMedia: (event) => {
              this.#avStreamMediaReceived++;
              this.#modalFmp4SegmentCount++;
              this.#fmp4LastStreamMode = event.mode ?? null;
              this.#fmp4LastStreamSequence = event.sequence ?? null;
              if (event.sequence != null) {
                this.#avStreamLastSequenceReceived = event.sequence;
              }

              if (this.#modalTimelineAbort) return;
              if (ws.readyState !== WebSocket.OPEN) {
                this.#capture(ws, "avatar_fmp4_av_forward_skipped_ws_not_open", {
                  sequence: event.sequence,
                  mode: event.mode,
                  byteLength: event.byteLength,
                  wsReadyState: ws.readyState,
                  receivedCount: this.#avStreamMediaReceived,
                  sentCount: this.#avStreamMediaSendSucceeded,
                  failedCount: this.#avStreamMediaSendFailed,
                });
                return;
              }

              if (event.mode === "speech" && event.isSpeechStart) {
                this.#assistantSpeechActive = true;
                this.#activeSpeechRenderTurnId = event.turnId ?? null;
                this.#speechFirstFrameSentAt = Date.now();

                try {
                  ws.send(JSON.stringify({
                    type: "avatar.speech.start",
                    turnId: event.turnId!,
                    totalFrames: 0,
                    fps: 25,
                  } satisfies ServerMessage));
                } catch (speechStartErr) {
                  this.#capture(ws, "avatar_fmp4_av_speech_start_send_failed", {
                    error: speechStartErr instanceof Error ? speechStartErr.message : String(speechStartErr),
                    turnId: event.turnId,
                    wsReadyState: ws.readyState,
                  });
                }

                this.#capture(ws, "avatar_fmp4_av_speech_started", {
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                  chunkIndex: event.chunkIndex,
                  chunkRenderMs: event.timing?.chunkRenderMs,
                  enqueueToFirstFrameMs: event.timing?.enqueueToFirstFrameMs,
                  gpuMemorySmiUsedMb: event.gpu?.smiMemoryUsedMb,
                  gpuMemorySmiTotalMb: event.gpu?.smiMemoryTotalMb,
                  gpuMemoryTorchAllocMb: event.gpu?.torchAllocatedMb,
                  gpuMemoryTorchPeakMb: event.gpu?.torchPeakMb,
                  containerInstanceId: event.containerInstanceId,
                  overlapWithTts: event.timing?.overlapWithTts,
                  gpuTarget: getParallelTarget(this.env),
                  pacingSleepMs: event.timing?.pacingSleepMs,
                  mediaLeadMs: event.timing?.mediaLeadMs,
                  producedMediaMs: event.timing?.producedMediaMs,
                  wallElapsedMs: event.timing?.wallElapsedMs,
                  targetLeadMs: event.timing?.targetLeadMs,
                  pacingMode: event.timing?.pacingMode,
                  receivedCount: this.#avStreamMediaReceived,
                  sentCount: this.#avStreamMediaSendSucceeded,
                });

                const enqueueRequestedAt = event.turnId != null ? this.#fmp4EnqueueRequestedAt.get(event.turnId) : undefined;
                if (enqueueRequestedAt) {
                  this.#fmp4FirstSpeechSegmentSentAt.set(event.turnId!, Date.now());
                  this.#capture(ws, "avatar_fmp4_av_speech_first_segment_latency", {
                    turnId: event.turnId,
                    enqueueToSpeechStartMs: Date.now() - enqueueRequestedAt,
                  });
                  this.#fmp4EnqueueRequestedAt.delete(event.turnId!);
                }
              }

              this.#avStreamMediaSendAttempted++;
              try {
                ws.send(JSON.stringify({
                  type: "avatar.fmp4.segment",
                  dataBase64: event.dataBase64,
                  byteLength: event.byteLength,
                  sequence: event.sequence,
                  mode: event.mode,
                  isSpeechStart: event.isSpeechStart ?? false,
                  isSpeechEnd: event.isSpeechEnd ?? false,
                  turnId: event.turnId ?? null,
                  segmentId: event.segmentId ?? 0,
                  chunkIndex: event.chunkIndex ?? 0,
                  chunkRenderMs: event.timing?.chunkRenderMs ?? 0,
                  feedMs: event.timing?.feedMs ?? 0,
                  framesFed: event.timing?.framesFed ?? 0,
                  audioFeedMs: event.timing?.audioFeedMs ?? 0,
                  audioSamplesPerChunk: event.timing?.audioSamplesPerChunk ?? 0,
                  containerInstanceId: event.containerInstanceId,
                  overlapWithTts: event.timing?.overlapWithTts,
                } satisfies ServerMessage));
                this.#avStreamMediaSendSucceeded++;
                if (event.sequence != null) {
                  this.#avStreamLastSequenceSent = event.sequence;
                }
                const now = Date.now();
                if (this.#avStreamLastMediaSentAt > 0) {
                  const gap = now - this.#avStreamLastMediaSentAt;
                  this.#avStreamLastSendGapMs = gap;
                  if (gap > this.#avStreamMaxSendGapMs) {
                    this.#avStreamMaxSendGapMs = gap;
                  }
                }
                this.#avStreamLastMediaSentAt = now;
              } catch (sendErr) {
                this.#avStreamMediaSendFailed++;
                this.#capture(ws, "avatar_fmp4_av_forward_failed", {
                  sequence: event.sequence,
                  byteLength: event.byteLength,
                  mode: event.mode,
                  error: sendErr instanceof Error ? sendErr.message : String(sendErr),
                  wsReadyState: ws.readyState,
                  receivedCount: this.#avStreamMediaReceived,
                  sentCount: this.#avStreamMediaSendSucceeded,
                  failedCount: this.#avStreamMediaSendFailed,
                  lastSequenceSent: this.#avStreamLastSequenceSent,
                  containerInstanceId: event.containerInstanceId,
                });
                console.error("[AvatarSessionDO] fmp4_av_forward_failed", {
                  sessionId: this.#sessionId,
                  sequence: event.sequence,
                  error: sendErr instanceof Error ? sendErr.message : String(sendErr),
                  wsReadyState: ws.readyState,
                  receivedCount: this.#avStreamMediaReceived,
                  sentCount: this.#avStreamMediaSendSucceeded,
                  failedCount: this.#avStreamMediaSendFailed,
                });
                return;
              }

              if (event.mode === "speech" && event.isSpeechEnd) {
                this.#capture(ws, "avatar_fmp4_av_speech_ended", {
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                  audioFeedMs: event.timing?.audioFeedMs,
                  containerInstanceId: event.containerInstanceId,
                  gpuTarget: getParallelTarget(this.env),
                });

                this.#pendingPlaybackAckTurnId = event.turnId ?? null;
                this.#playbackAckTimeoutMs = PLAYBACK_ACK_TIMEOUT_MS;
                const ackDeadline = Date.now() + PLAYBACK_ACK_TIMEOUT_MS;
                const waitForAck = async () => {
                  while (this.#pendingPlaybackAckTurnId === event.turnId && Date.now() < ackDeadline) {
                    await scheduler.wait(200);
                  }
                  if (this.#pendingPlaybackAckTurnId === event.turnId) {
                    this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                      turnId: event.turnId,
                      source: "modal_fmp4_av",
                    });
                    this.#pendingPlaybackAckTurnId = null;
                  }
                  this.#assistantSpeechActive = false;
                  this.#activeSpeechRenderTurnId = null;
                  this.#modalTimelineIdleStarted = false;
                };
                this.ctx.waitUntil(waitForAck());
              }

              const verboseSample = verboseDelivery || this.#modalFmp4SegmentCount % 10 === 0;
              if (verboseSample) {
                this.#capture(ws, "avatar_fmp4_av_segment_forwarded", {
                  sequence: event.sequence,
                  mode: event.mode,
                  byteLength: event.byteLength,
                  totalSegments: this.#modalFmp4SegmentCount,
                  receivedCount: this.#avStreamMediaReceived,
                  sentCount: this.#avStreamMediaSendSucceeded,
                  failedCount: this.#avStreamMediaSendFailed,
                  wsReadyState: ws.readyState,
                  audioFeedMs: event.timing?.audioFeedMs,
                  audioSamplesPerChunk: event.timing?.audioSamplesPerChunk,
                  chunkRenderMs: event.timing?.chunkRenderMs,
                  speechEndToIdleResumeMs: event.timing?.speechEndToIdleResumeMs,
                  containerInstanceId: event.containerInstanceId,
                  overlapWithTts: event.timing?.overlapWithTts,
                  gpuTarget: getParallelTarget(this.env),
                  pacingSleepMs: event.timing?.pacingSleepMs,
                  mediaLeadMs: event.timing?.mediaLeadMs,
                  producedMediaMs: event.timing?.producedMediaMs,
                  wallElapsedMs: event.timing?.wallElapsedMs,
                  targetLeadMs: event.timing?.targetLeadMs,
                  pacingMode: event.timing?.pacingMode,
                });
              }

              if (event.timing?.overlapWithTts) {
                this.#capture(ws, "avatar_fmp4_av_overlap_segment_forwarded", {
                  sequence: event.sequence,
                  mode: event.mode,
                  byteLength: event.byteLength,
                  totalSegments: this.#modalFmp4SegmentCount,
                  chunkRenderMs: event.timing?.chunkRenderMs,
                  feedMs: event.timing?.feedMs,
                  turnId: event.turnId,
                  segmentId: event.segmentId,
                  chunkIndex: event.chunkIndex,
                  containerInstanceId: event.containerInstanceId,
                  gpuTarget: getParallelTarget(this.env),
                });
              }

              this.#lastAvatarFrameAt = Date.now();
            },

            onEnd: (event) => {
              console.log("[AvatarSessionDO] fmp4_av_stream_end", {
                sessionId: this.#sessionId,
                chunksRendered: event.chunksRendered,
                stopReason: event.stopReason,
              });
              this.#fmp4StreamSummaryCaptured = true;
              this.#capture(ws, "avatar_fmp4_av_stream_summary", {
                cleanEnd: true,
                summarySource: "modal_sse_end",
                chunksRendered: event.chunksRendered,
                mediaSegments: event.mediaSegments,
                audioBytes: event.audioBytes,
                videoBytes: event.videoBytes,
                encodedBytes: event.encodedBytes,
                ffmpegReturnCode: event.ffmpegReturnCode,
                ffmpegStderrErrorCount: event.ffmpegStderrErrorCount,
                gpuSmiUsedMinMb: event.gpu?.smiUsedMinMb,
                gpuSmiUsedMaxMb: event.gpu?.smiUsedMaxMb,
                gpuSmiTotalMb: event.gpu?.smiTotalMb,
                gpuTorchAllocMinMb: event.gpu?.torchAllocMinMb,
                gpuTorchAllocMaxMb: event.gpu?.torchAllocMaxMb,
                gpuTorchPeakMb: event.gpu?.torchPeakMb,
                gpuSnapshotCount: event.gpu?.snapshotCount,
                gpuSmiGpuUtilMinPct: event.gpu?.smiGpuUtilMinPct,
                gpuSmiGpuUtilMaxPct: event.gpu?.smiGpuUtilMaxPct,
                gpuSmiGpuUtilAvgPct: event.gpu?.smiGpuUtilAvgPct,
                gpuSmiMemoryUtilMinPct: event.gpu?.smiMemoryUtilMinPct,
                gpuSmiMemoryUtilMaxPct: event.gpu?.smiMemoryUtilMaxPct,
                gpuSmiMemoryUtilAvgPct: event.gpu?.smiMemoryUtilAvgPct,
                bridgeReceivedCount: this.#avStreamMediaReceived,
                bridgeSentCount: this.#avStreamMediaSendSucceeded,
                bridgeFailedCount: this.#avStreamMediaSendFailed,
                bridgeLastSequenceReceived: this.#avStreamLastSequenceReceived,
                bridgeLastSequenceSent: this.#avStreamLastSequenceSent,
                bridgeMaxSendGapMs: this.#avStreamMaxSendGapMs,
                modalCadenceStats: event.cadenceStats,
              });
              if (this.#fmp4StreamEndResolve) {
                this.#fmp4StreamEndResolve();
                this.#fmp4StreamEndResolve = null;
              }
            },
          },
          this.#modalTimelineAbortController?.signal,
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        console.error("[AvatarSessionDO] fmp4_av_stream_error", {
          sessionId: this.#sessionId,
          error: errMsg,
          receivedCount: this.#avStreamMediaReceived,
          sentCount: this.#avStreamMediaSendSucceeded,
          failedCount: this.#avStreamMediaSendFailed,
          lastSequenceReceived: this.#avStreamLastSequenceReceived,
          lastSequenceSent: this.#avStreamLastSequenceSent,
        });
        this.#capture(ws, "avatar_fmp4_av_stream_failed", {
          error: errMsg,
          segmentsForwarded: this.#modalFmp4SegmentCount,
          receivedCount: this.#avStreamMediaReceived,
          sentCount: this.#avStreamMediaSendSucceeded,
          failedCount: this.#avStreamMediaSendFailed,
          lastSequenceReceived: this.#avStreamLastSequenceReceived,
          lastSequenceSent: this.#avStreamLastSequenceSent,
        });
      } finally {
        this.#modalFmp4StreamActive = false;
        this.#modalTimelineAbortController = null;

        if (!this.#fmp4StreamSummaryCaptured) {
          this.#capture(ws, "avatar_fmp4_av_stream_summary", {
            cleanEnd: false,
            summarySource: "bridge_fallback",
            segmentsForwarded: this.#modalFmp4SegmentCount,
            lastKnownMode: this.#fmp4LastStreamMode,
            lastKnownSequence: this.#fmp4LastStreamSequence,
            sessionEndReason: this.#sessionClosing ? "session_end" : "stream_error",
            hadModalEndEvent: false,
            bridgeReceivedCount: this.#avStreamMediaReceived,
            bridgeSentCount: this.#avStreamMediaSendSucceeded,
            bridgeFailedCount: this.#avStreamMediaSendFailed,
            bridgeLastSequenceReceived: this.#avStreamLastSequenceReceived,
            bridgeLastSequenceSent: this.#avStreamLastSequenceSent,
            bridgeMaxSendGapMs: this.#avStreamMaxSendGapMs,
          });
        }

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({
            type: "avatar.timeline.stopped",
            reason: "fmp4_av_stream_ended",
          } satisfies ServerMessage));
        }

        this.#capture(ws, "avatar_fmp4_av_stream_stopped", {
          segmentsForwarded: this.#modalFmp4SegmentCount,
          bridgeReceivedCount: this.#avStreamMediaReceived,
          bridgeSentCount: this.#avStreamMediaSendSucceeded,
          bridgeFailedCount: this.#avStreamMediaSendFailed,
          bridgeLastSequenceReceived: this.#avStreamLastSequenceReceived,
          bridgeLastSequenceSent: this.#avStreamLastSequenceSent,
          bridgeMaxSendGapMs: this.#avStreamMaxSendGapMs,
          fmp4EnqueueRequestedAtSize: this.#fmp4EnqueueRequestedAt.size,
          fmp4FirstSpeechSegmentSentAtSize: this.#fmp4FirstSpeechSegmentSentAt.size,
          summaryWasCaptured: this.#fmp4StreamSummaryCaptured,
        });
        this.#fmp4EnqueueRequestedAt.clear();
        this.#fmp4FirstSpeechSegmentSentAt.clear();
        this.#fmp4StreamEndResolve = null;
      }
    })());
  }

  #runAvatarTimeline(ws: WebSocket): void {
    if (this.#avatarTimelineRunning) return;
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    this.#avatarTimelineRunning = true;
    this.#avatarTimelineMode = "idle";
    this.#avatarSegmentIndex = 0;
    this.#avatarTimelineAbort = false;
    this.#idleConsecutiveFailuresTimeline = 0;
    this.#idleBackoffMs = 0;

    this.#capture(ws, "avatar_timeline_started", { modalSessionId });

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "avatar.timeline.started", fps: 25, mode: "jpeg_timeline" } satisfies ServerMessage));
    }

    const IDLE_SEGMENT_DURATION_S = 0.5;
    const IDLE_JPEG_QUALITY = 75;
    const INTER_SEGMENT_DELAY_MS = 100;
    const MAX_IDLE_BACKOFF_MS = 5000;
    const MAX_IDLE_CONSECUTIVE_FAILURES = 10;
    const PLAYBACK_ACK_TIMEOUT_MS = 30_000;

    this.ctx.waitUntil((async () => {
      try {
        while (
          !this.#avatarTimelineAbort &&
          !this.#sessionClosing &&
          ws.readyState === WebSocket.OPEN
        ) {
          const speech = this.#avatarSpeechQueue.shift();

          if (speech) {
            this.#avatarTimelineMode = "speech";
            this.#assistantSpeechActive = true;
            this.#activeSpeechRenderTurnId = speech.turnId;
            this.#idleConsecutiveFailuresTimeline = 0;
            this.#idleBackoffMs = 0;

            const maxDurationS = Math.min(
              Math.ceil(speech.durationMs / 1000) + 1,
              120,
            );

            this.#capture(ws, "avatar_timeline_speech_dequeued", {
              turnId: speech.turnId,
              queueLength: this.#avatarSpeechQueue.length,
              maxDurationS,
            });

            if (ws.readyState === WebSocket.OPEN) {
              ws.send(JSON.stringify({ type: "status", status: "rendering_avatar_audio" } satisfies ServerMessage));
            }

            const speechResult = await this.#renderTimelineSegment(ws, {
              mode: "speech",
              audioBase64: speech.audioBase64,
              maxDurationS,
              jpegQuality: 80,
              turnId: speech.turnId,
            });

            if (!speechResult.ok && speech.audioBase64) {
              try {
                await this.#fallbackBatchRender(
                  ws,
                  speech.turnId,
                  speech.audioBase64,
                  speech.ttsProvider,
                  speech.audioBytes,
                  Date.now(),
                  speechResult.failReason ?? "timeline_stream_failed",
                );
              } catch (fallbackErr) {
                this.#captureError(ws, fallbackErr, { stage: "timeline_speech_fallback" });
              }
            }

            this.#pendingPlaybackAckTurnId = speech.turnId;
            this.#playbackAckTimeoutMs = PLAYBACK_ACK_TIMEOUT_MS;
            const ackDeadline = Date.now() + PLAYBACK_ACK_TIMEOUT_MS;
            while (this.#pendingPlaybackAckTurnId === speech.turnId && Date.now() < ackDeadline) {
              await scheduler.wait(200);
            }
            if (this.#pendingPlaybackAckTurnId === speech.turnId) {
              this.#capture(ws, "avatar_speech_playback_ack_timeout", {
                turnId: speech.turnId,
                source: "timeline",
              });
              this.#pendingPlaybackAckTurnId = null;
              this.#assistantSpeechActive = false;
              this.#activeSpeechRenderTurnId = null;
            }

            this.#avatarTimelineMode = "idle";
            this.#avatarTimelineIdleStarted = false;
          } else {
            this.#avatarTimelineMode = "idle";

            if (this.#idleBackoffMs > 0) {
              await scheduler.wait(this.#idleBackoffMs);
            }

            this.#capture(ws, "avatar_timeline_idle_segment_requested", {
              segmentIndex: this.#avatarSegmentIndex,
              consecutiveFailures: this.#idleConsecutiveFailuresTimeline,
              backoffMs: this.#idleBackoffMs,
            });

            const idleResult = await this.#renderTimelineSegment(ws, {
              mode: "idle",
              audioBase64: null,
              maxDurationS: IDLE_SEGMENT_DURATION_S,
              jpegQuality: IDLE_JPEG_QUALITY,
            });

            if (!idleResult.ok) {
              this.#idleConsecutiveFailuresTimeline++;
              this.#idleBackoffMs = Math.min(
                this.#idleBackoffMs > 0 ? this.#idleBackoffMs * 2 : 500,
                MAX_IDLE_BACKOFF_MS,
              );
              this.#capture(ws, "avatar_timeline_idle_segment_failed", {
                segmentIndex: this.#avatarSegmentIndex,
                consecutiveFailures: this.#idleConsecutiveFailuresTimeline,
                backoffMs: this.#idleBackoffMs,
                error: idleResult.failReason,
              });
              if (this.#idleConsecutiveFailuresTimeline >= MAX_IDLE_CONSECUTIVE_FAILURES) {
                this.#capture(ws, "avatar_timeline_stopped", {
                  reason: "idle_max_failures",
                  consecutiveFailures: this.#idleConsecutiveFailuresTimeline,
                });
                break;
              }
            } else {
              this.#idleConsecutiveFailuresTimeline = 0;
              this.#idleBackoffMs = 0;
            }

            await scheduler.wait(INTER_SEGMENT_DELAY_MS);
          }

          this.#avatarSegmentIndex++;
        }
      } catch (err) {
        this.#capture(ws, "avatar_timeline_stopped", {
          reason: "error",
          error: err instanceof Error ? err.message : String(err),
        });
      } finally {
        this.#avatarTimelineRunning = false;
        this.#avatarTimelineMode = "stopped";
        this.#capture(ws, "avatar_timeline_stopped", {
          reason: "loop_exited",
          segmentIndex: this.#avatarSegmentIndex,
        });
      }
    })());
  }

  async #renderTimelineSegment(
    ws: WebSocket,
    params: {
      mode: "idle" | "speech";
      audioBase64: string | null;
      maxDurationS: number;
      jpegQuality: number;
      turnId?: number;
    },
  ): Promise<{ ok: boolean; chunksForwarded: number; failReason?: string }> {
    const segmentStartAt = Date.now();
    let chunksForwarded = 0;
    let streamFailed = false;
    let failReason: string | undefined;

    this.#capture(ws, "avatar_timeline_segment_started", {
      mode: params.mode,
      segmentIndex: this.#avatarSegmentIndex,
      turnId: params.turnId ?? null,
      audioDurationMs: params.maxDurationS * 1000,
    });

    await this.#acquireModalLock();
    try {
      await renderAudioStream(
        this.env,
        this.#state.modalSessionId!,
        params.audioBase64,
        params.maxDurationS,
        params.jpegQuality,
        {
          onStart: (event) => {
            if (params.mode === "speech" && this.#activeSpeechRenderTurnId !== params.turnId) return;
            if (ws.readyState !== WebSocket.OPEN) return;

            if (params.mode === "speech") {
              ws.send(JSON.stringify({
                type: "avatar.speech.start",
                turnId: params.turnId!,
                totalFrames: event.totalSteps * 5,
                fps: event.fps,
                renderedAudioDurationMs: event.renderedAudioDurationMs,
                originalAudioDurationMs: event.originalAudioDurationMs,
              } satisfies ServerMessage));
            } else {
              if (!this.#avatarTimelineIdleStarted) {
                this.#avatarTimelineIdleStarted = true;
                ws.send(JSON.stringify({
                  type: "avatar.idle.start",
                  totalFrames: event.totalSteps * 5,
                  fps: event.fps,
                  durationMs: event.renderedAudioDurationMs,
                } satisfies ServerMessage));
              }
            }
          },

          onChunk: (event) => {
            if (params.mode === "speech" && this.#activeSpeechRenderTurnId !== params.turnId) return;
            if (ws.readyState !== WebSocket.OPEN) return;

            chunksForwarded++;

            if (params.mode === "speech") {
              ws.send(JSON.stringify({
                type: "avatar.speech.chunk",
                turnId: params.turnId!,
                chunkIndex: event.chunkIndex,
                frames: event.frames,
                isLast: event.isLast,
              } satisfies ServerMessage));
            } else {
              ws.send(JSON.stringify({
                type: "avatar.idle.chunk",
                chunkIndex: event.chunkIndex,
                frames: event.frames,
                isLast: event.isLast,
                live: true,
                durationMs: params.maxDurationS * 1000,
              } satisfies ServerMessage));
            }

            this.#lastAvatarFrameAt = Date.now();

            this.#capture(ws, "avatar_timeline_frame_chunk_forwarded", {
              mode: params.mode,
              segmentIndex: this.#avatarSegmentIndex,
              turnId: params.turnId ?? null,
              chunkIndex: event.chunkIndex,
              framesInChunk: event.frames.length,
              isLast: event.isLast,
            });
          },

          onEnd: (event) => {
            this.#capture(ws, "avatar_timeline_segment_completed", {
              mode: params.mode,
              segmentIndex: this.#avatarSegmentIndex,
              turnId: params.turnId ?? null,
              framesRendered: event.framesRendered,
              chunksForwarded,
              timeToFirstFrameMs: event.firstChunkReadyMs,
              segmentRenderMs: Date.now() - segmentStartAt,
              truncated: event.truncated,
              originalDurationMs: event.originalDurationMs,
              renderedAudioDurationMs: event.renderedAudioDurationMs,
              maxDurationS: params.maxDurationS,
            });

            console.log("[AvatarSessionDO] timeline_segment_completed", {
              sessionId: this.#sessionId,
              mode: params.mode,
              turnId: params.turnId ?? null,
              framesRendered: event.framesRendered,
              truncated: event.truncated,
              segmentRenderMs: Date.now() - segmentStartAt,
            });
          },
        },
      );

      if (chunksForwarded === 0 && params.mode === "speech") {
        streamFailed = true;
        failReason = "stream_produced_no_chunks";
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      streamFailed = true;
      failReason = errMsg;

      this.#capture(ws, "avatar_timeline_segment_failed", {
        mode: params.mode,
        segmentIndex: this.#avatarSegmentIndex,
        turnId: params.turnId ?? null,
        error: errMsg,
      });

      console.error("[AvatarSessionDO] timeline_segment_failed", {
        sessionId: this.#sessionId,
        mode: params.mode,
        turnId: params.turnId ?? null,
        error: errMsg,
      });
    } finally {
      this.#releaseModalLock();
    }

    return { ok: !streamFailed, chunksForwarded, failReason };
  }

  #runLiveIdleLoop(ws: WebSocket): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) return;

    this.#idleLiveEnabled = true;
    this.#idleChunkIndex = 0;
    this.#idleConsecutiveFailures = 0;
    this.#idleFirstFrameSentAt = 0;

    const sessionReadyAt = this.#sessionReadyAt;
    const IDLE_CHUNK_DURATION_S = 0.5;
    const MAX_CONSECUTIVE_FAILURES = 5;
    const LOOP_DELAY_MS = 400;

    this.#capture(ws, "avatar_idle_live_started", {
      modalSessionId,
      timeSinceSessionReadyMs: sessionReadyAt ? Date.now() - sessionReadyAt : undefined,
    });

    const loop = async () => {
      while (
        this.#idleLiveEnabled
        && !this.#sessionClosing
        && ws.readyState === WebSocket.OPEN
        && this.#idleConsecutiveFailures < MAX_CONSECUTIVE_FAILURES
      ) {
        if (this.#assistantSpeechActive || this.#modalRenderBusy) {
          await scheduler.wait(100);
          continue;
        }

        this.#idleChunkIndex++;
        const chunkIndex = this.#idleChunkIndex;

        await this.#acquireModalLock();
        if (this.#assistantSpeechActive || !this.#idleLiveEnabled || this.#sessionClosing) {
          this.#releaseModalLock();
          break;
        }

        this.#capture(ws, "avatar_idle_live_chunk_requested", {
          modalSessionId,
          chunkIndex,
          durationS: IDLE_CHUNK_DURATION_S,
          timeSinceSessionReadyMs: sessionReadyAt ? Date.now() - sessionReadyAt : undefined,
        });

        try {
          const modalCallStart = Date.now();
          const { data: result, meta: callMeta } = await renderIdleChunk(
            this.env, modalSessionId, IDLE_CHUNK_DURATION_S, 75,
          );
          const bridgeModalCallDurationMs = Date.now() - modalCallStart;

          if (!result.available || !result.frames || result.frames.length === 0) {
            const errMsg = result.error || "No frames returned";
            this.#idleConsecutiveFailures++;

            this.#capture(ws, "avatar_idle_live_chunk_failed", {
              modalSessionId,
              chunkIndex,
              error: errMsg,
              consecutiveFailures: this.#idleConsecutiveFailures,
              timeSinceSessionReadyMs: sessionReadyAt ? Date.now() - sessionReadyAt : undefined,
              bridgeModalCallDurationMs,
              responseJsonParseMs: callMeta.responseJsonParseMs,
            });

            console.warn("[AvatarSessionDO] avatar_idle_live_chunk_failed", {
              sessionId: this.#sessionId,
              chunkIndex,
              error: errMsg,
              consecutiveFailures: this.#idleConsecutiveFailures,
              bridgeModalCallDurationMs,
            });

            if (errMsg.includes("No active session")) {
              this.#state.active = false;
              await this.#saveState();
              this.#idleLiveEnabled = false;
              this.#releaseModalLock();
              break;
            }
          } else if (!this.#assistantSpeechActive && ws.readyState === WebSocket.OPEN) {
            this.#idleConsecutiveFailures = 0;

            const stT0 = Date.now();
            const msg = JSON.stringify({
              type: "avatar.idle.chunk",
              chunkIndex,
              frames: result.frames,
              isLast: false,
              live: true,
              durationMs: result.durationMs,
              renderDurationMs: result.renderDurationMs,
            } satisfies ServerMessage);
            const stringifyMs = Date.now() - stT0;

            if (msg.length > 1_000_000) {
              console.warn("[AvatarSessionDO] idle chunk payload exceeds 1MB", {
                sessionId: this.#sessionId,
                chunkIndex,
                payloadChars: msg.length,
              });
            }

            const sendT0 = Date.now();
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(msg);
            }
            const wsSendCallMs = Date.now() - sendT0;

            const isFirstChunk = this.#idleFirstFrameSentAt === 0;
            if (isFirstChunk) {
              this.#idleFirstFrameSentAt = Date.now();
              this.#capture(ws, "avatar_first_idle_frame_latency", {
                modalSessionId,
                latencyMs: Date.now() - sessionReadyAt,
              });
            }

            this.#capture(ws, "avatar_idle_live_chunk_completed", {
              modalSessionId,
              chunkIndex,
              durationS: IDLE_CHUNK_DURATION_S,
              framesRendered: result.framesRendered,
              renderDurationMs: result.renderDurationMs,
              modelRenderDurationMs: result.modelRenderDurationMs,
              jpegEncodeDurationMs: result.jpegEncodeDurationMs,
              base64EncodeDurationMs: result.base64EncodeDurationMs,
              frameConvertDurationMs: result.frameConvertDurationMs,
              totalHandlerDurationMs: result.totalHandlerDurationMs,
              overheadMs: result.overheadMs,
              avgJpegEncodeMsPerFrame: result.avgJpegEncodeMsPerFrame,
              modalEncodedBytes: result.encodedBytes,
              modalBase64Chars: result.base64Chars,
              payloadChars: msg.length,
              payloadBytes: msg.length,
              bridgeModalCallDurationMs,
              responseJsonParseMs: callMeta.responseJsonParseMs,
              stringifyMs,
              wsSendCallMs,
              timeSinceSessionReadyMs: sessionReadyAt ? Date.now() - sessionReadyAt : undefined,
            });
          }

          this.#releaseModalLock();
        } catch (err) {
          this.#releaseModalLock();
          const errMsg = err instanceof Error ? err.message : String(err);
          this.#idleConsecutiveFailures++;

          this.#capture(ws, "avatar_idle_live_chunk_failed", {
            modalSessionId,
            chunkIndex,
            error: errMsg,
            consecutiveFailures: this.#idleConsecutiveFailures,
          });

          console.warn("[AvatarSessionDO] avatar_idle_live_chunk_failed", {
            sessionId: this.#sessionId,
            chunkIndex,
            error: errMsg,
          });
        }

        if (this.#idleConsecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
          this.#capture(ws, "avatar_idle_live_stopped", {
            modalSessionId,
            reason: "max_consecutive_failures",
            chunkIndex,
            consecutiveFailures: this.#idleConsecutiveFailures,
          });
          this.#idleLiveEnabled = false;
          break;
        }

        await scheduler.wait(LOOP_DELAY_MS);
      }

      if (this.#idleLiveEnabled) {
        this.#idleLiveEnabled = false;
        this.#capture(ws, "avatar_idle_live_stopped", {
          modalSessionId,
          reason: "loop_exited",
          chunkIndex: this.#idleChunkIndex,
          consecutiveFailures: this.#idleConsecutiveFailures,
        });
      }
    };

    this.ctx.waitUntil(loop());
  }

  #handlePlaybackAck(
    ws: WebSocket,
    msg: ClientMessage & { type: "avatar.speech.playback.ended" },
  ): void {
    const turnId = msg.turnId;
    this.#capture(ws, "avatar_speech_playback_ack", {
      turnId,
      pendingTurnId: this.#pendingPlaybackAckTurnId,
    });

    if (this.#pendingPlaybackAckTurnId === turnId) {
      this.#pendingPlaybackAckTurnId = null;
      this.#playbackAckTimeoutMs = 0;
      this.#assistantSpeechActive = false;
    }
  }

  #handleTurnFinalized(ws: WebSocket, turnId: number): void {
    const snapshot = this.#snapshotTurnAudio();
    if (!snapshot) return;

    if (ws.readyState === WebSocket.OPEN) {
      ws.send(
        JSON.stringify({
          type: "status",
          status: "transcribing",
        } satisfies ServerMessage),
      );
    }

    this.#runStt(ws, turnId, snapshot.audio, snapshot.bytes);
  }

  async #resetSpeechState(): Promise<void> {
    const vadService = this.env.VAD_SERVICE as VadServiceRpc | undefined;
    const stService = this.env
      .SMART_TURN_SERVICE as SmartTurnServiceRpc | undefined;

    try {
      if (vadService?.reset) {
        await vadService.reset(`${this.#sessionId}:vad`);
      }
    } catch (err) {
      console.warn("[AvatarSessionDO] vad_reset_error", {
        sessionId: this.#sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
    }

    try {
      if (stService?.reset) {
        await stService.reset(`${this.#sessionId}:st`);
      }
    } catch (err) {
      console.warn("[AvatarSessionDO] smart_turn_reset_error", {
        sessionId: this.#sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
    }

    this.#turnOpen = false;
    this.#speechSegmentActive = false;
    this.#segmentIndex = 0;
    this.#smartTurnChunkCount = 0;
    this.#smartTurnAccumulatedBytes = 0;
    this.#currentTurnStartedAt = 0;
    this.#suppressedEndCount = 0;
    this.#suppressedStartCount = 0;
    this.#smartTurnAnalyzeCount = 0;
    this.#smartTurnAccumulatedBytesAtLastAnalyze = 0;
    this.#analysisId = 0;
    this.#analysisInFlight = false;
    this.#turnAudioChunks = [];
    this.#turnAudioBytes = 0;
    this.#trimTelemetryChunkCounter = 0;
    this.#trimTelemetryTimestamp = 0;
    this.#trimStarted = false;
  }

  // ── Modal Session Handlers ─────────────────────────────────────

  async #handleSessionStart(
    ws: WebSocket,
    msg: ClientMessage & { type: "session.start" },
  ): Promise<void> {
    if (this.#state.active) {
      this.#sendError(ws, "session_already_active", `Session ${this.#state.modalSessionId} is already active`);
      return;
    }

    if (this.#state.starting) {
      const age = Date.now() - (this.#state.startingAt ?? 0);
      if (age < STARTING_STALE_MS) {
        this.#sendError(ws, "session_starting", "Session start already in progress");
        return;
      }
      this.#capture(ws, "session_starting_stale_cleared", { ageMs: age });
      this.#state.starting = false;
      this.#state.startingAt = undefined;
      this.#state.startAttemptId = null;
    }

    const use4CPath = !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL
      && !!this.env.MODAL_SESSION_START_URL;
    const useLegacyPath = !use4CPath && isLegacyModalEnabled(this.env);

    if (!use4CPath && !useLegacyPath) {
      this.#sendError(
        ws,
        "no_modal_protocol_enabled",
        "Neither MODAL_SESSION_STREAM_FMP4_AV_URL (4C) nor ENABLE_LEGACY_MODAL_PROTOCOL (legacy) is set",
      );
      return;
    }

    // ── Phase D1: SFU transport decision (pure helper) ──
    const transportPref = msg.transport;
    const transportDecision = resolveTransportDecision(this.env, transportPref, use4CPath);

    // Explicit webrtc_sfu but config incomplete → hard error (no fallback)
    if (transportPref === "webrtc_sfu" && !transportDecision.shouldTrySfu) {
      this.#sendError(ws, "sfu_not_configured", "SFU explicitly requested but prerequisites missing (AVTR_REALTIME_SFU_ENABLED, CALLS_APP_ID, CALLS_APP_SECRET, MODAL_WORKER_SECRET, WORKER_BASE_URL)");
      return;
    }

    // Phase 3F: explicit webrtc_sfu also requires private beta authorization.
    if (transportPref === "webrtc_sfu" && !(await this.#isSfuBetaAuthorized())) {
      this.#sendError(ws, "sfu_beta_not_enabled", "SFU transport is not enabled for this session. Request access via a beta-authorized tenant or a valid SFU beta token.");
      return;
    }

    const shouldTrySfu = transportDecision.shouldTrySfu;

    const requestedAvatarId = msg.avatarId || "default";
    const resolvedAvatarId = requestedAvatarId === "default"
      ? "arnold"
      : requestedAvatarId;

    // Phase 6a.1: validate resolved avatar/background IDs against the catalog
    // before the expensive Modal start call.
    const validation = await validateAvatarBgForStart(
      this.env,
      await this.#getOwnerTenantId(),
      resolvedAvatarId,
      msg.bgId,
    );
    if (!validation.ok) {
      this.#capture(ws, "session_start_catalog_validation_failed", {
        requestedAvatarId,
        resolvedAvatarId,
        bgId: msg.bgId,
        code: validation.code,
      });
      this.#sendError(ws, validation.code, validation.message);
      return;
    }

    const attemptId = crypto.randomUUID();
    this.#state.starting = true;
    this.#state.startingAt = Date.now();
    this.#state.startAttemptId = attemptId;
    this.#state.startCancelRequested = false;
    this.#state.endedAt = undefined;
    await this.#saveState();

    this.#capture(ws, "session_start_requested", {
      requestedAvatarId,
      resolvedAvatarId,
      bgId: msg.bgId,
      startAttemptId: attemptId,
    });

    try {
      const startedAt = Date.now();
      ws.send(JSON.stringify({ type: "status", status: "starting_modal" } satisfies ServerMessage));

      if (use4CPath) {
        this.#capture(ws, "avatar_id_resolved_4c", {
          requestedAvatarId,
          resolvedAvatarId,
        });

        let result: Record<string, unknown> = {};
        let usingSfu = shouldTrySfu;
        let sfuSessionId: string | null = null;

        // ── SFU path: LAZY session creation in publisher-offer callback ──
        // The SFU session is NOT created here. It is created lazily inside
        // #handlePublisherOffer when Modal calls back with the offer SDP.
        // This avoids the 30s SFU session TTL expiring during Modal cold
        // start (CF Realtime sessions GC after 30s of inactivity).
        if (usingSfu) {
          // PRE-SAVE SFU state BEFORE Modal call — publisher-offer callback
          // during Modal startup needs transport=webrtc_sfu to be set.
          // sfuSessionId stays null; #handlePublisherOffer creates it lazily.
          this.#state.transport = "webrtc_sfu";
          this.#state.sfuSessionId = null;
          this.#state.sfuPublisherMids = { video: null, audio: null };
          this.#state.sfuSubscriberMids = [];
          this.#state.sfuPublisherTrackName = null;
          this.#state.sfuSubscriberSessionId = null;
          await this.#saveState();

          // Generate TURN credentials (critical path — Modal needs them
          // before ICE gathering). On failure, emit error + cleanup.
          let turnIceServers: unknown[] | null = null;
          try {
            turnIceServers = await this.#generateTurnCredentials();
          } catch (turnErr) {
            const turnMsg = turnErr instanceof SfuClientError ? turnErr.message : String(turnErr);
            this.#capture(ws, "sfu_turn_credentials_error", { error: turnMsg });
            // No capacity acquired yet — tracks-only cleanup, no release owed.
            await this.#cleanupSfuTracksAndState();
            this.#state.transport = "websocket_fmp4";
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            this.#sendError(ws, "sfu_turn_credentials_unavailable", turnMsg);
            return;
          }

          // Acquire SFU capacity slot — prevents GPU contention from
          // overlapping SFU publishers on the same Modal container.
          // FAIL-CLOSED: capacity manager outage → sfu_capacity_unavailable.
          const capacityResult = await this.#acquireSfuCapacity();
          if (!capacityResult.ok) {
            // Differentiate: capacity slot full (409) vs manager down (503).
            if (capacityResult.unavailable) {
              this.#capture(ws, "sfu_capacity_unavailable", { reason: "manager_unreachable" });
              await this.#cleanupSfuTracksAndState();
              this.#state.transport = "websocket_fmp4";
              this.#state.starting = false;
              this.#state.startAttemptId = null;
              await this.#saveState();
              this.#sendError(ws, "sfu_capacity_unavailable", "SFU capacity manager is unreachable. Retry shortly.");
              return;
            }
            const conflictMsg = capacityResult.conflict
              ? `Another SFU session is still active (${capacityResult.conflict.slice(-12)}). Wait for it to end.`
              : "SFU capacity exceeded.";
            this.#capture(ws, "sfu_capacity_exceeded", { conflict: capacityResult.conflict ?? null });
            // Acquire never succeeded — tracks-only cleanup, no release owed.
            await this.#cleanupSfuTracksAndState();
            this.#state.transport = "websocket_fmp4";
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            this.#sendError(ws, "sfu_capacity_exceeded", conflictMsg);
            return;
          }

          result = await startModalSession(this.env, {
            session_id: this.#sessionId,
            avatar_id: resolvedAvatarId,
            bg_id: msg.bgId || undefined,
            config: null,
            publisher_mode: "webrtc_sfu",
            publisher_callback_url: `${this.env.WORKER_BASE_URL}/internal/sessions/${this.#sessionId}/publisher-offer`,
            close_callback_url: `${this.env.WORKER_BASE_URL}/internal/sessions/${this.#sessionId}/publisher-close`,
            ...(turnIceServers ? { ice_servers: turnIceServers } : {}),
          }).then(r => r.data) as Record<string, unknown>;

          // Check cancellation during Modal call
          await this.#loadState();
          if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
            // Capacity was acquired above. Modal orphan /end is attempted
            // first, THEN we release the slot so the next session inherits
            // a clean GPU.
            if (result && !("error" in result) && result.session_id) {
              await endModalSessionBestEffort(this.env, result.session_id as string, 3_000);
            }
            await this.#cleanupSfuTracksAndState();
            await this.#releaseSfuCapacity();
            this.#capture(ws, "ws_session_start_cancelled", { usingSfu: true });
            this.#sendError(ws, "session_cancelled", "Session was ended while starting");
            return;
          }

          if ("error" in result) {
            // SFU-mode Modal start failed — clean up orphan + SFU, then maybe retry
            const orphanId = result.session_id as string | undefined;
            if (orphanId) {
              await endModalSessionBestEffort(this.env, orphanId, 3_000);
              this.#capture(ws, "sfu_fallback_modal_orphan_cleaned", { modalOrphanId: orphanId });
            }
            // Orphan Modal /end done above (best-effort). Now safe to
            // release capacity — GPU no longer has a publisher from us.
            await this.#cleanupSfuTracksAndState();
            await this.#releaseSfuCapacity();

            // Forensic: emit allowlisted sfu_ice_debug event (fire-and-forget).
            // Explicit allowlist only — no raw object spread. Safe summary fields.
            const rawIceDebug = (result.ice_debug ?? null) as Record<string, unknown> | null;
            if (rawIceDebug) {
              this.#capture(ws, "sfu_ice_debug", this.#extractIceDebugAllowlist(rawIceDebug));
            }

            if (!transportDecision.fallbackAllowed) {
              this.#state.transport = "websocket_fmp4";
              this.#state.starting = false;
              this.#state.startAttemptId = null;
              await this.#saveState();
              // Stage 3B reliability repair: include health_check_detail,
              // publisher_stop_metrics, and container_poisoned from Modal so
              // operators can see the exact reason (CUDA error, dead producer
              // thread, insufficient generation rate, etc.) in PostHog.
              this.#captureError(ws, new Error(result.error as string || "Modal start_session returned error"), {
                stage: "modal_start_4c_sfu",
                healthCheckDetail: result.health_check_detail ?? null,
                publisherStopMetrics: result.publisher_stop_metrics ?? null,
                containerPoisoned: result.container_poisoned === true,
                poisoningReason: result.poisoning_reason ?? null,
              });
              this.#sendError(ws, "sfu_unavailable", (result.error as string) || "SFU Modal start failed");
              return;
            }

            // Auto fallback — retry without publisher_mode
            this.#capture(ws, "sfu_fallback_to_fmp4", { reason: result.error as string, modalOrphanCleaned: !!orphanId });
            this.#state.transport = "websocket_fmp4";
            this.#state.sfuSessionId = null;
            await this.#saveState();
            usingSfu = false;
          }
        }

        if (!usingSfu) {
          // fMP4 path (normal start or auto-fallback from SFU failure)
          result = await startModalSession(this.env, {
            session_id: this.#sessionId,
            avatar_id: resolvedAvatarId,
            bg_id: msg.bgId || undefined,
            config: null,
          }).then(r => r.data) as Record<string, unknown>;

          await this.#loadState();
          if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
            if (result && !("error" in result) && result.session_id) {
              let cleanupOk = true;
              let cleanupError: string | undefined;
              try {
                await endModalSession(this.env, result.session_id as string, { timeoutMs: 3_000 }).then(r => r.data);
              } catch (e) {
                cleanupOk = false;
                cleanupError = e instanceof Error ? e.message : String(e);
              }
              this.#capture(ws, "modal_start_cancelled_cleanup", {
                modalSessionId: result.session_id,
                cancelledBy: "delete_during_start",
                cleanupOk,
                ...(cleanupError ? { cleanupError } : {}),
              });
            } else {
              this.#capture(ws, "ws_session_start_cancelled", { modalSessionId: null });
            }
            this.#sendError(ws, "session_cancelled", "Session was ended while starting");
            return;
          }

          if ("error" in result) {
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            this.#captureError(ws, new Error(result.error as string || "Modal start_session returned error"), { stage: "modal_start_4c" });
            this.#sendError(ws, "modal_start_failed", (result.error as string) || "Modal start_session returned error");
            return;
          }
        }

        // ── Success — save final state, preserving SFU callback fields ──
        // The publisher-offer callback may have run DURING startModalSession()
        // (via /internal/sessions/:id/publisher-offer) and written
        // sfuPublisherMids.video + sfuPublisherTrackName to storage. State was
        // reloaded at the post-Modal loadState() above, so this.#state already
        // holds those callback-written values. Capture and clone them before
        // building the final state object so they survive the assignment.
        const sfuSuccess = usingSfu;
        const savedPublisherMids = {
          video: this.#state.sfuPublisherMids?.video ?? null,
          audio: this.#state.sfuPublisherMids?.audio ?? null,
        };
        const savedSubscriberMids = [...(this.#state.sfuSubscriberMids ?? [])];
        const savedPublisherTrackName = this.#state.sfuPublisherTrackName ?? null;
        const savedSubscriberSessionId = this.#state.sfuSubscriberSessionId ?? null;

        this.#state = {
          modalSessionId: result.session_id as string,
          active: true,
          avatarId: result.avatar_id as string,
          bgId: null,
          starting: false,
          startAttemptId: null,
          startCancelRequested: false,
          ...(sfuSuccess ? {
            transport: "webrtc_sfu" as const,
            sfuSessionId: this.#state.sfuSessionId,
            // Preserve publisher mid + track name written by the callback.
            sfuPublisherMids: { ...savedPublisherMids },
            sfuSubscriberMids: [...savedSubscriberMids],
            sfuPublisherTrackName: savedPublisherTrackName ?? (result.publisher_track_name as string) ?? null,
            sfuSubscriberSessionId: savedSubscriberSessionId,
            // Preserve capacity ownership flag set by #acquireSfuCapacity().
            sfuCapacityAcquired: this.#state.sfuCapacityAcquired ?? false,
          } : {
            transport: "websocket_fmp4" as const,
          }),
        };
        this.#sessionReadyAt = Date.now();
        await this.#saveState();

        // Cost-safety final guard: a WS close / REST DELETE that happened
        // while Modal start was finishing must prevent session_ready and the
        // keepalive alarm from starting. This closes the runaway-Modal window
        // that caused the billing-limit incident.
        if (sfuSuccess) {
          const wasCancelled = await this.#cleanupCancelledSfuStart(
            result.session_id as string,
            attemptId,
            ws,
            "post_start_guard_ws",
          );
          if (wasCancelled) {
            this.#sendError(ws, "session_cancelled", "Session was ended while starting");
            return;
          }
        }

        const modalResult = result;
        const modalTimings = (modalResult.timings ?? {}) as Record<string, unknown>;

        this.#capture(ws, "session_ready", {
          modalSessionId: result.session_id,
          requestedAvatarId,
          resolvedAvatarId,
          avatarId: result.avatar_id,
          modalFetchMs: Date.now() - startedAt,
          protocol: "4c",
          transport: this.#state.transport,
          ...(this.#state.sfuSessionId ? { sfuSessionId: this.#state.sfuSessionId } : {}),
          modalTimings,
          modalEngineBackends: modalResult.engine_backends ?? {},
          modalEngineBytesEnabled: modalResult.engine_bytes_enabled ?? false,
          modalEngineBytesHit: modalResult.engine_bytes_hit ?? 0,
          modalEngineBytesTotalMb: modalResult.engine_bytes_total_mb ?? 0,
          modalCachedAvatarHit: modalResult.cached_avatar_hit ?? false,
          modalWarmupMs: modalResult.warmup_ms ?? 0,
          modalTotalPostRestoreS: modalTimings.total_post_restore_s,
          modalPipelineFromArtifactsS: modalTimings.pipeline_from_artifacts_s,
          modalCachedAvatarLoadS: modalTimings.cached_avatar_load_s,
          modalScaledownWindow: modalResult.scaledown_window,
          modalMinContainers: modalResult.min_containers,
          modalOptPhase: modalResult.opt_phase,
          bgId: result.bg_id,
          requestedBgId: result.requested_bg_id,
        });

        // Forensic: emit candidate summary on success for pass-vs-fail comparison.
        // Uses the same allowlist extraction as the failure path.
        const successIceDebug = (result.ice_debug_success ?? null) as Record<string, unknown> | null;
        if (successIceDebug && sfuSuccess) {
          this.#capture(ws, "sfu_ice_debug_success", this.#extractIceDebugAllowlist(successIceDebug));
        }

        // For SFU transport: mint sfu_token and construct subscriber URLs
        // so the browser can call /sfu/subscribe, /sfu/answer, /sfu/close
        // without needing an API key. Token is purpose:"sfu".
        // The tenant MUST match the stored owner's tenant_id so that
        // checkSessionOwnership in /sfu/subscribe passes. For the legacy
        // /sessions beta path, initSessionOwner sets tenant_id="legacy-browser-beta".
        // For plain legacy /sessions (no beta), owner is created below with "legacy".
        let sfuTokenVal: string | undefined;
        let sfuTokenExpiresAt: number | undefined;
        if (this.#state.transport === "webrtc_sfu" && this.env.AVTR_SESSION_TOKEN_SECRET) {
          // Ensure session owner exists for legacy sessions (POST /sessions doesn't
          // call initSessionOwner like POST /v1/sessions does). The /sfu/subscribe
          // endpoint runs checkSessionOwnership which requires a stored owner.
          const existingOwner = await this.ctx.storage.get<SessionOwner>(STORAGE_OWNER_KEY);
          if (!existingOwner) {
            const owner: SessionOwner = {
              tenant_id: "legacy",
              auth_mode: "legacy",
              key_name: null,
              plan: null,
              created_at: Date.now(),
            };
            await this.ctx.storage.put(STORAGE_OWNER_KEY, owner);
          }

          // Use the ACTUAL owner's tenant_id — not a hardcoded value.
          // This ensures the sfu_token tenant matches the stored owner,
          // so checkSessionOwnership passes when the browser calls /sfu/subscribe.
          const sfuTenantId = await this.#getOwnerTenantId();
          const sfuTokenResult = await createWsToken(
            this.env.AVTR_SESSION_TOKEN_SECRET,
            this.#sessionId,
            sfuTenantId,
            300,
            "sfu",
          );
          sfuTokenVal = sfuTokenResult.token;
          sfuTokenExpiresAt = sfuTokenResult.expiresAt;

          this.#capture(ws, "sfu_token_issued", {
            source: "session_ready_ws",
            tenantId: "legacy",
            expiresAt: sfuTokenExpiresAt,
          });
        }

        const reply: ServerMessage = {
          type: "session.ready",
          sessionId: result.session_id as string,
          avatarId: result.avatar_id as string,
          bgId: (result.bg_id as string) || "",
          width: 1280,
          height: 720,
          fps: 25,
          framesPerChunk: 0,
          warmupMs: Number(modalResult.warmup_ms ?? 0),
          ...(this.#state.transport === "webrtc_sfu" && sfuTokenVal ? {
            transport: "webrtc_sfu" as const,
            renderMode: undefined,
            sfu: {
              publisher_track_name: this.#state.sfuPublisherTrackName ?? "",
              subscribe_url: `${this.env.WORKER_BASE_URL}/v1/sessions/${this.#sessionId}/sfu/subscribe`,
              answer_url: `${this.env.WORKER_BASE_URL}/v1/sessions/${this.#sessionId}/sfu/answer`,
              close_url: `${this.env.WORKER_BASE_URL}/v1/sessions/${this.#sessionId}/sfu/close`,
              token: sfuTokenVal,
              token_expires_at: sfuTokenExpiresAt!,
            },
          } : {
            renderMode: "fmp4_av" as const,
          }),
        };
        ws.send(JSON.stringify(reply));

        // SFU transport: no fMP4 stream pump — browser subscribes via SFU REST
        if (this.#state.transport !== "webrtc_sfu") {
          this.#runModalSessionFmp4Stream(ws);
        } else {
          // Stage 3B: Start DO-alarm keepalive to prevent Modal container recycle
          await this.#startSfuKeepalive();
        }
        return;
      }

      const result = await startSession(
        this.env,
        msg.avatarId,
        msg.bgId,
      ).then(r => r.data);

      await this.#loadState();
      if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
        if (result && !("error" in result) && result.sessionId) {
          let cleanupOk = true;
          let cleanupError: string | undefined;
          try {
            await endSession(this.env, result.sessionId).then(r => r.data);
          } catch (e) {
            cleanupOk = false;
            cleanupError = e instanceof Error ? e.message : String(e);
          }
          this.#capture(ws, "modal_start_cancelled_cleanup", {
            modalSessionId: result.sessionId,
            cancelledBy: "delete_during_start",
            cleanupOk,
            ...(cleanupError ? { cleanupError } : {}),
          });
        } else {
          this.#capture(ws, "ws_session_start_cancelled", {
            modalSessionId: null,
          });
        }
        this.#sendError(ws, "session_cancelled", "Session was ended while starting");
        return;
      }

      if ("error" in result) {
        this.#state.starting = false;
        this.#state.startAttemptId = null;
        await this.#saveState();
        this.#captureError(ws, new Error((result as { error: string }).error || "Modal start_session returned error"), {
          stage: "modal_start",
        });
        this.#sendError(ws, "modal_start_failed", (result as { error: string }).error || "Modal start_session returned error");
        return;
      }

      this.#state = {
        modalSessionId: result.sessionId,
        active: true,
        avatarId: result.avatarId,
        bgId: result.bgId,
        starting: false,
        startAttemptId: null,
        startCancelRequested: false,
      };
      this.#sessionReadyAt = Date.now();
      await this.#saveState();

      this.#capture(ws, "session_ready", {
        modalSessionId: result.sessionId,
        avatarId: result.avatarId,
        bgId: result.bgId,
        latencyMs: Date.now() - startedAt,
        protocol: "legacy",
      });

      const renderMode = this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true"
        ? "fmp4_av" as const
        : this.env.ENABLE_MODAL_FMP4_VIDEO_STREAM === "true"
          ? "fmp4_video" as const
          : "jpeg_timeline" as const;

      const reply: ServerMessage = {
        type: "session.ready",
        sessionId: result.sessionId,
        avatarId: result.avatarId,
        bgId: result.bgId,
        width: result.width,
        height: result.height,
        fps: result.fps,
        framesPerChunk: result.framesPerChunk,
        warmupMs: result.warmupMs,
        renderMode,
      };
      ws.send(JSON.stringify(reply));

      if (this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true") {
        this.#runModalTimelineFmp4AvStream(ws);
      } else if (this.env.ENABLE_MODAL_FMP4_VIDEO_STREAM === "true") {
        this.#runModalTimelineFmp4Stream(ws);
      } else if (this.env.ENABLE_MODAL_PERSISTENT_TIMELINE !== "false") {
        this.#runModalTimelineStream(ws);
      } else if (this.env.ENABLE_AVATAR_TIMELINE !== "false") {
        this.#runAvatarTimeline(ws);
      } else if (this.env.ENABLE_LIVE_IDLE === "true") {
        this.#runLiveIdleLoop(ws);
      }
    } catch (err) {
      await this.#loadState();
      if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
        this.#sendError(ws, "session_cancelled", "Session was ended while starting");
        return;
      }
      // Cost-safety: if Modal start threw (e.g., 429 billing limit, network
      // timeout) after we acquired the SFU capacity slot, release it before
      // returning the error. Otherwise a billing-limit storm can wedge the
      // global slot indefinitely.
      await this.#cleanupSfuTracksAndState();
      await this.#releaseSfuCapacity();
      this.#state = emptyState();
      this.#state.endedAt = Date.now();
      await this.#saveState();
      const errMsg = err instanceof ModalClientError
        ? err.message
        : `Modal start failed: ${err instanceof Error ? err.message : String(err)}`;
      this.#captureError(ws, err, { stage: "modal_start" });
      this.#sendError(ws, "modal_unreachable", errMsg);
    }
  }

  async #handleRenderFrame(ws: WebSocket): Promise<void> {
    if (!this.#state.active) {
      this.#sendError(ws, "no_active_session", "No active session. Send session.start first.");
      return;
    }

    if (this.#avatarTimelineRunning || this.#modalTimelineStreamActive) {
      this.#sendError(ws, "render_frame_blocked", "Debug frame disabled while avatar timeline is running.");
      return;
    }

    this.#capture(ws, "render_debug_frame_requested", {
      modalSessionId: this.#state.modalSessionId,
    });

    try {
      const startedAt = Date.now();
      ws.send(JSON.stringify({ type: "status", status: "rendering" } satisfies ServerMessage));
      const result = await renderFrame(
        this.env,
        this.#state.modalSessionId ?? undefined,
      ).then(r => r.data);

      if (!result.available) {
        this.#sendError(ws, "render_unavailable", result.note || "Render frame not available");
        return;
      }

      const sizeKb = Math.round((result.dataBase64?.length ?? 0) * 0.75 / 1024);

      this.#capture(ws, "avatar_frame_sent", {
        modalSessionId: this.#state.modalSessionId,
        width: result.width,
        height: result.height,
        frameSizeKb: sizeKb,
        latencyMs: Date.now() - startedAt,
      });

      const reply: ServerMessage = {
        type: "avatar.frame",
        dataBase64: result.dataBase64!,
        mimeType: result.mimeType!,
        width: result.width!,
        height: result.height!,
      };
      ws.send(JSON.stringify(reply));
    } catch (err) {
      const msg = err instanceof ModalClientError
        ? err.message
        : `Render frame failed: ${err instanceof Error ? err.message : String(err)}`;
      this.#captureError(ws, err, { stage: "render_frame" });
      this.#sendError(ws, "render_failed", msg);
    }
  }

  async #handleSessionEnd(ws: WebSocket): Promise<void> {
    const wasActive = this.#state.active;
    const wasStarting = this.#state.starting;
    const modalSessionId = this.#state.modalSessionId;

    if (!wasActive && wasStarting) {
      this.#capture(ws, "session_start_cancelled_by_end", {
        modalSessionId,
        startAttemptId: this.#state.startAttemptId,
      });
      this.#state.starting = false;
      this.#state.startingAt = undefined;
      this.#state.startAttemptId = null;
      this.#state.startCancelRequested = true;
      this.#state.endedAt = Date.now();
      await this.#saveState();

      const reply: ServerMessage = {
        type: "session.ended",
        sessionId: null,
        wasActive: false,
        modalEndOk: false,
        modalEndDurationMs: 0,
      };
      ws.send(JSON.stringify(reply));
      return;
    }

    this.#capture(ws, "session_end_requested", {
      modalSessionId,
      wasActive,
      activeTurnBeforeEnd: this.#turnOpen,
      currentTurnAgeMs: this.#turnOpen ? Date.now() - this.#currentTurnStartedAt : 0,
      suppressedEndCount: this.#suppressedEndCount,
      suppressedStartCount: this.#suppressedStartCount,
      smartTurnAnalyzeCount: this.#smartTurnAnalyzeCount,
      audioChunkCount: this.#audioChunkCount,
      audioBytesReceived: this.#audioBytesReceived,
    });

    this.#sessionClosing = true;
    this.#idleLiveEnabled = false;
    this.#assistantSpeechActive = false;
    this.#pendingPlaybackAckTurnId = null;
    this.#activeSpeechRenderTurnId = null;
    this.#avatarTimelineAbort = true;
    this.#avatarSpeechQueue = [];

    if (this.#modalTimelineStreamActive && modalSessionId) {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
      try {
        await stopModalTimeline(this.env, modalSessionId);
      } catch {
        // best-effort
      }
    }

    if (this.#modalFmp4StreamActive && modalSessionId) {
      const isAvMode = this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true";
      this.#modalTimelineAbort = true;

      try {
        await stopModalTimeline(this.env, modalSessionId, { useControlEndpoint: isAvMode });
      } catch {
        // best-effort
      }

      if (!this.#fmp4StreamSummaryCaptured && this.#modalFmp4StreamActive) {
        const endPromise = new Promise<void>((resolve) => {
          this.#fmp4StreamEndResolve = resolve;
        });
        const timeoutPromise = new Promise<void>((resolve) => {
          setTimeout(resolve, 5_000);
        });
        await Promise.race([endPromise, timeoutPromise]);
        this.#fmp4StreamEndResolve = null;
      }

      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    } else {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    }

    if (this.#turnOpen) {
      const forcedTurnId = this.#turnCounter;
      if (this.#turnAudioChunks.length > 0) {
        this.#capture(ws, "speech_finalized", {
          turnId: forcedTurnId,
          finalizationReason: "session_end",
          forcedFinalization: true,
          smartTurnFallback: true,
          currentTurnAgeMs: Date.now() - this.#currentTurnStartedAt,
          audioChunkCount: this.#audioChunkCount,
        });

        if (ws.readyState === WebSocket.OPEN) {
          ws.send(
            JSON.stringify({
              type: "speech.finalized",
              turnId: forcedTurnId,
              smartTurn: { prediction: 1, probability: 1, rawLogit: 0 },
              smartTurnFallback: true,
            } satisfies ServerMessage),
          );
        }

        console.log("[AvatarSessionDO] speech.finalized (session_end forced)", {
          sessionId: this.#sessionId,
          turnId: forcedTurnId,
          finalizationReason: "session_end",
        });

        const snapshot = this.#snapshotTurnAudio();
        if (snapshot) {
          this.#runStt(ws, forcedTurnId, snapshot.audio, snapshot.bytes);
        }
      } else {
        this.#capture(ws, "stt_discarded_empty_turn", {
          turnId: forcedTurnId,
          finalizationReason: "session_end",
        });
        console.log("[AvatarSessionDO] stt_discarded_empty_turn (session_end)", {
          sessionId: this.#sessionId,
          turnId: forcedTurnId,
        });
      }
    }

    await this.#resetSpeechState();

    this.#audioChunkCount = 0;
    this.#audioBytesReceived = 0;
    this.#firstAudioChunkCaptured = false;

    // Stage 3B: SFU teardown — tracks cleanup runs first, Modal /end runs
    // next, capacity is released LAST (after Modal returns or fails
    // best-effort). This ordering prevents a new SFU session from
    // acquiring the slot while the GPU publisher is still alive.
    const wasCapacityAcquired = this.#state.sfuCapacityAcquired === true;
    if (this.#state.transport === "webrtc_sfu") {
      // Closes SFU publisher/subscriber tracks + clears local SFU state.
      // Does NOT release capacity — that happens after Modal /end below.
      await this.#cleanupSfuTracksAndState();
    }

    this.#state = emptyState();
    this.#conversationHistory = [];
    await this.#saveState();

    let modalEndOk = false;
    let modalEndError: string | undefined;
    let modalEndDurationMs = 0;

    if (wasActive && modalSessionId) {
      const modalEndStart = Date.now();
      try {
        ws.send(JSON.stringify({ type: "status", status: "ending" } satisfies ServerMessage));
        // Phase 4C: pick end_session endpoint based on the same
        // protocol chosen at start. If we used the 4C start path,
        // use the 4C end. Otherwise fall back to legacy endSession.
        const use4CEnd = !!this.env.MODAL_SESSION_END_URL && !!this.env.MODAL_SESSION_START_URL;
        if (use4CEnd) {
          await endModalSession(this.env, modalSessionId).then(r => r.data);
        } else {
          await endSession(this.env, modalSessionId).then(r => r.data);
        }
        modalEndOk = true;
        modalEndDurationMs = Date.now() - modalEndStart;

        this.#capture(ws, "modal_end_completed", {
          modalSessionId,
          modalEndDurationMs,
          protocol: use4CEnd ? "4c" : "legacy",
        });
      } catch (err) {
        modalEndOk = false;
        modalEndDurationMs = Date.now() - modalEndStart;
        modalEndError = err instanceof Error ? err.message : String(err);

        console.error("[AvatarSessionDO] modal_end_error", {
          sessionId: this.#sessionId,
          modalSessionId,
          modalEndDurationMs,
          error: modalEndError,
        });

        this.#captureError(ws, err, {
          stage: "modal_end",
          modalSessionId,
          modalEndDurationMs,
        });
      } finally {
        // GPU publisher is gone (or best-effort cleanup failed). NOW
        // safe to release the capacity slot — next session inherits a
        // clean L40S. force=true because emptyState() above already
        // cleared sfuCapacityAcquired.
        if (wasCapacityAcquired) {
          await this.#releaseSfuCapacity(true);
        }
      }
    } else if (wasCapacityAcquired) {
      // Defensive: no Modal session to end, but capacity was somehow
      // acquired (shouldn't happen). Release it.
      await this.#releaseSfuCapacity(true);
    }

    this.#capture(ws, "session_ended", {
      modalSessionId,
      wasActive,
      modalEndOk,
      modalEndDurationMs,
      modalEndError,
    });

    const reply: ServerMessage = {
      type: "session.ended",
      sessionId: modalSessionId,
      wasActive,
      modalEndOk,
      modalEndDurationMs,
      modalEndError,
    };
    ws.send(JSON.stringify(reply));
  }

  // ── Phase 4C: 4C fMP4+AV stream consumer ────────────────────────
  // Called from #handleSessionStart when MODAL_SESSION_STREAM_FMP4_AV_URL
  // is configured. Opens a long-lived SSE connection to Modal's
  // http_stream_fmp4_av endpoint, drains fmp4_init / fmp4_media / end
  // events, and forwards them to the browser as avatar.fmp4.init and
  // avatar.fmp4.segment messages. Speech-turn detection (Phase 4E) and
  // per-segment GPU telemetry (Phase 4E) are not wired here; only the
  // core init/media/end forwarding required to keep the browser MSE
  // pipeline alive.

  #runModalSessionFmp4Stream(ws: WebSocket): void {
    const modalSessionId = this.#state.modalSessionId;
    if (!modalSessionId) {
      console.error("[AvatarSessionDO] run_modal_session_fmp4_stream_no_session", {
        sessionId: this.#sessionId,
      });
      this.#sendError(ws, "no_modal_session", "No Modal session ID to stream from");
      return;
    }

    if (this.#modalTimelineStreamActive) {
      console.warn("[AvatarSessionDO] run_modal_session_fmp4_stream_already_active", {
        sessionId: this.#sessionId,
        modalSessionId,
      });
      return;
    }

    this.#capture(ws, "modal_session_fmp4_stream_started", {
      modalSessionId,
      url: this.env.MODAL_SESSION_STREAM_FMP4_AV_URL,
      timeSinceSessionReadyMs: this.#sessionReadyAt ? Date.now() - this.#sessionReadyAt : 0,
    });

    this.#modalTimelineStreamActive = true;
    this.#modalTimelineAbort = false;
    this.#modalTimelineAbortController = new AbortController();

    const verboseDelivery = this.env.ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY === "true";
    const abortSignal = this.#modalTimelineAbortController.signal;

    startModalSessionFmp4Stream(
      this.env,
      modalSessionId,
      {
        onInit: (event) => {
          if (this.#modalTimelineAbort) return;
          if (ws.readyState !== WebSocket.OPEN) {
            this.#capture(ws, "modal_session_fmp4_init_skipped_ws_not_open", {
              byteLength: event.byteLength,
            });
            return;
          }
          try {
            ws.send(JSON.stringify({
              type: "avatar.fmp4.init",
              dataBase64: event.dataBase64,
              byteLength: event.byteLength,
              timingMs: event.timing?.timeToInitMs ?? 0,
            } satisfies ServerMessage));
            this.#capture(ws, "modal_session_fmp4_init_forwarded", {
              byteLength: event.byteLength,
              timingMs: event.timing?.timeToInitMs,
            });
          } catch (sendErr) {
            this.#capture(ws, "modal_session_fmp4_init_send_failed", {
              error: sendErr instanceof Error ? sendErr.message : String(sendErr),
            });
          }
        },

        onMedia: (event) => {
          this.#modalFmp4SegmentCount++;
          this.#avStreamMediaReceived++;
          this.#fmp4LastStreamMode = event.mode ?? null;
          this.#fmp4LastStreamSequence = event.sequence ?? null;
          if (event.sequence != null) {
            this.#avStreamLastSequenceReceived = event.sequence;
          }
          if (this.#modalTimelineAbort) return;
          if (ws.readyState !== WebSocket.OPEN) {
            this.#capture(ws, "modal_session_fmp4_segment_skipped_ws_not_open", {
              sequence: event.sequence,
              byteLength: event.byteLength,
            });
            return;
          }
          this.#avStreamMediaSendAttempted++;
          try {
            ws.send(JSON.stringify({
              type: "avatar.fmp4.segment",
              dataBase64: event.dataBase64,
              byteLength: event.byteLength,
              sequence: event.sequence,
              mode: event.mode ?? "idle",
              isSpeechStart: event.isSpeechStart ?? false,
              isSpeechEnd: event.isSpeechEnd ?? false,
              turnId: event.turnId ?? null,
              segmentId: event.segmentId ?? 0,
              chunkIndex: event.chunkIndex ?? 0,
              chunkRenderMs: event.timing?.chunkRenderMs ?? 0,
              feedMs: event.timing?.feedMs ?? 0,
              framesFed: event.timing?.framesFed ?? 0,
              audioFeedMs: event.timing?.audioFeedMs,
              audioSamplesPerChunk: event.timing?.audioSamplesPerChunk,
              containerInstanceId: event.containerInstanceId,
              overlapWithTts: event.timing?.overlapWithTts,
            } satisfies ServerMessage));
            this.#avStreamMediaSendSucceeded++;
            if (event.sequence != null) {
              this.#avStreamLastSequenceSent = event.sequence;
            }
            const now = Date.now();
            if (this.#avStreamLastMediaSentAt > 0) {
              const gap = now - this.#avStreamLastMediaSentAt;
              this.#avStreamLastSendGapMs = gap;
              if (gap > this.#avStreamMaxSendGapMs) this.#avStreamMaxSendGapMs = gap;
            }
            this.#avStreamLastMediaSentAt = now;
            this.#lastAvatarFrameAt = now;
            const bufferedAmount = (ws as unknown as { bufferedAmount?: number }).bufferedAmount ?? 0;
            if (verboseDelivery || this.#modalFmp4SegmentCount % 10 === 0) {
              this.#capture(ws, "modal_session_fmp4_segment_forwarded", {
                sequence: event.sequence,
                mode: event.mode,
                byteLength: event.byteLength,
                totalSegments: this.#modalFmp4SegmentCount,
                receivedCount: this.#avStreamMediaReceived,
                sentCount: this.#avStreamMediaSendSucceeded,
                bufferedAmount,
              });
            }
            // Telemetry-only high-backpressure warning. Skipping/dropping segments at this threshold
            // requires runtime verification first; do not add skip logic without explicit approval.
            const WS_BUFFERED_AMOUNT_HIGH_THRESHOLD = 5 * 1024 * 1024; // 5 MiB
            if (bufferedAmount > WS_BUFFERED_AMOUNT_HIGH_THRESHOLD) {
              this.#capture(ws, "media_ws_backpressure_high", {
                bufferedAmount,
                threshold: WS_BUFFERED_AMOUNT_HIGH_THRESHOLD,
                sequence: event.sequence,
              });
            }
          } catch (sendErr) {
            this.#avStreamMediaSendFailed++;
            this.#capture(ws, "modal_session_fmp4_segment_send_failed", {
              sequence: event.sequence,
              error: sendErr instanceof Error ? sendErr.message : String(sendErr),
            });
            console.error("[AvatarSessionDO] modal_session_fmp4_segment_send_failed", {
              sessionId: this.#sessionId,
              sequence: event.sequence,
              error: sendErr instanceof Error ? sendErr.message : String(sendErr),
            });
          }
        },

        onEnd: (event) => {
          console.log("[AvatarSessionDO] modal_session_fmp4_stream_end", {
            sessionId: this.#sessionId,
            stopReason: event.stopReason,
            totalSegments: this.#modalFmp4SegmentCount,
          });
          this.#capture(ws, "modal_session_fmp4_stream_summary", {
            cleanEnd: true,
            totalSegments: this.#modalFmp4SegmentCount,
            receivedCount: this.#avStreamMediaReceived,
            sentCount: this.#avStreamMediaSendSucceeded,
            failedCount: this.#avStreamMediaSendFailed,
            stopReason: event.stopReason,
          });
          this.#modalTimelineStreamActive = false;
          if (this.#fmp4StreamEndResolve) {
            this.#fmp4StreamEndResolve();
            this.#fmp4StreamEndResolve = null;
          }
        },

        onOverflow: (info) => {
          this.#capture(ws, "media_queue_overflow_graceful", {
            droppedCount: info.droppedCount,
            queueLen: info.queueLen,
            firstDroppedSeq: info.firstDroppedSeq,
            lastDroppedSeq: info.lastDroppedSeq,
            idleDropped: info.idleDropped,
            mode: info.mode,
          });
        },

        onStats: (stats) => {
          this.#capture(ws, "modal_session_fmp4_stream_stats", {
            maxQueueDepth: stats.maxQueueDepth,
            receivedMediaCount: stats.receivedMediaCount,
            forwardedMediaCount: stats.forwardedMediaCount,
            overflowCount: stats.overflowCount,
            queueCapacity: stats.queueCapacity,
            adaptiveFastForwardCount: stats.adaptiveFastForwardCount,
            minForwardDelayMs: stats.minForwardDelayMs,
            maxForwardDelayMs: stats.maxForwardDelayMs,
            backpressureWaitCount: stats.backpressureWaitCount,
            backpressureWaitMs: stats.backpressureWaitMs,
            highWater: stats.highWater,
            lowWater: stats.lowWater,
          });
        },
      },
      abortSignal,
      this.env.FMP4_RELAY_PACE_MS ? parseInt(this.env.FMP4_RELAY_PACE_MS, 10) : 200,
    ).catch((err) => {
      this.#modalTimelineStreamActive = false;
      const msg = err instanceof ModalClientError
        ? err.message
        : err instanceof Error ? err.message : String(err);
      console.error("[AvatarSessionDO] modal_session_fmp4_stream_error", {
        sessionId: this.#sessionId,
        modalSessionId,
        error: msg,
      });
      this.#captureError(ws, err, {
        stage: "modal_session_fmp4_stream",
        modalSessionId,
      });
      this.#capture(ws, "modal_session_fmp4_stream_failed", {
        error: msg,
        receivedCount: this.#avStreamMediaReceived,
        sentCount: this.#avStreamMediaSendSucceeded,
      });
      if (this.#fmp4StreamEndResolve) {
        this.#fmp4StreamEndResolve();
        this.#fmp4StreamEndResolve = null;
      }
    });
  }

  #sendError(
    ws: WebSocket,
    code: string,
    message: string,
    fatal = false,
  ): void {
    const msg: ServerMessage = { type: "error", code, message, fatal };
    ws.send(JSON.stringify(msg));
  }

  // ── Phase 5B: REST lifecycle handlers ──────────────────────────

  #captureRest(event: string, properties?: Record<string, unknown>): void {
    captureEvent(
      this.env,
      this.ctx.waitUntil.bind(this.ctx),
      this.#sessionId,
      event,
      { avatarSessionId: this.#sessionId, source: "v1_rest", ...properties },
    );
  }

  async #handleRestRequest(request: Request): Promise<Response> {
    try {
      const url = new URL(request.url);
      const segments = url.pathname.split("/").filter(Boolean);

      // ── Phase D1: Internal Modal callbacks (/internal/sessions/:id/...) ──
      if (segments.length >= 4 && segments[0] === "internal" && segments[1] === "sessions") {
        this.#sessionId = decodeURIComponent(segments[2]);
        const operation = segments[3];

        await this.#loadState();

        if (!this.#state.createdAt) {
          this.#state.createdAt = Date.now();
          await this.#saveState();
        }

        switch (operation) {
          case "publisher-offer":
            return await this.#handlePublisherOffer(request);
          case "publisher-close":
            return await this.#handlePublisherClose(request);
        }
      }

      // ── Existing v1 REST routes ──
      if (segments.length >= 4 && segments[0] === "v1" && segments[1] === "sessions") {
        this.#sessionId = decodeURIComponent(segments[2]);
        const operation = segments[3];

        await this.#loadState();

        if (!this.#state.createdAt) {
          this.#state.createdAt = Date.now();
          await this.#saveState();
        }

        // ── Phase D1: SFU browser routes (/v1/sessions/:id/sfu/{subscribe|answer|close}) ──
        if (operation === "sfu" && segments.length >= 5) {
          const sfuOp = segments[4];
          switch (sfuOp) {
            case "subscribe":
              return await this.#handleSfuSubscribe(request);
            case "answer":
              return await this.#handleSfuAnswer(request);
            case "close":
              return await this.#handleSfuClose(request);
          }
        }

        switch (operation) {
          case "start":
            return await this.#handleRestStart(request);
          case "end":
            return await this.#handleRestEnd();
          case "status":
            return await this.#handleRestStatus();
          case "stream":
            return await this.#handleRestStream(request);
          case "init_owner":
            return await this.#handleRestInitOwner(request);
          case "get_owner":
            return await this.#handleRestGetOwner();
          case "force-clear":
            return await this.#handleForceClear();
        }
      }

      return Response.json(
        { error: { code: "unknown_operation", message: "Unknown REST operation" } },
        { status: 404 },
      );
    } catch (err) {
      console.error("[AvatarSessionDO] REST handler error", {
        sessionId: this.#sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
      return Response.json(
        { error: { code: "internal_error", message: "Internal server error" } },
        { status: 500 },
      );
    }
  }

  // ── Ownership operations (separate storage key, never touches SessionState) ──

  async #handleRestInitOwner(request: Request): Promise<Response> {
    let body: {
      tenant_id?: string;
      auth_mode?: string;
      key_name?: string;
      plan?: string;
      sfu_beta_authorized?: boolean;
    } = {};
    try {
      const text = await request.text();
      if (text) body = JSON.parse(text);
    } catch {
      return Response.json(
        { error: { code: "invalid_body", message: "Request body must be valid JSON" } },
        { status: 400 },
      );
    }

    // Idempotent: if owner already exists, return existing without overwriting.
    const existing = await this.ctx.storage.get<SessionOwner>(STORAGE_OWNER_KEY);
    if (existing) {
      return Response.json({ ok: true, owner: existing });
    }

    // Fail closed: require non-empty tenant_id and auth_mode.
    if (typeof body.tenant_id !== "string" || body.tenant_id.trim() === "") {
      return Response.json(
        { error: { code: "invalid_body", message: "tenant_id must be a non-empty string" } },
        { status: 400 },
      );
    }
    if (typeof body.auth_mode !== "string" || body.auth_mode.trim() === "") {
      return Response.json(
        { error: { code: "invalid_body", message: "auth_mode must be a non-empty string" } },
        { status: 400 },
      );
    }

    const owner: SessionOwner = {
      tenant_id: body.tenant_id,
      auth_mode: body.auth_mode,
      key_name: typeof body.key_name === "string" ? body.key_name : null,
      plan: typeof body.plan === "string" ? body.plan : null,
      created_at: Date.now(),
      sfu_beta_authorized: body.sfu_beta_authorized === true,
    };
    await this.ctx.storage.put(STORAGE_OWNER_KEY, owner);

    // Phase 3F: mirror the authorization flag onto SessionState so it is
    // available immediately when session.start runs (state is already loaded).
    this.#state.sfuBetaAuthorized = body.sfu_beta_authorized === true;
    await this.#saveState();

    return Response.json({ ok: true });
  }

  async #handleRestGetOwner(): Promise<Response> {
    const owner = await this.ctx.storage.get<SessionOwner>(STORAGE_OWNER_KEY);
    if (!owner) {
      return Response.json(
        { error: { code: "owner_not_found", message: "Session owner not found" } },
        { status: 404 },
      );
    }
    return Response.json({ owner });
  }

  /**
   * Admin force-clear: unconditional DO cleanup. Must succeed even if Modal
   * is unreachable, billing-limited, or 429ing. Deletes the alarm, clears
   * state, and releases capacity. Used for incident response.
   */
  async #handleForceClear(): Promise<Response> {
    await this.#loadState();

    const modalSessionId = this.#state.modalSessionId ?? null;
    const wasActive = this.#state.active === true;
    const wasKeepaliveActive = this.#state.sfuKeepaliveActive === true;
    const wasCapacityAcquired = this.#state.sfuCapacityAcquired === true;

    this.#captureRest("sfu_modal_force_clear", {
      modalSessionId,
      wasActive,
      wasKeepaliveActive,
      wasCapacityAcquired,
      transport: this.#state.transport ?? null,
    });

    // Delete alarm first — stop the loop immediately regardless of Modal.
    await this.ctx.storage.deleteAlarm();
    this.#state.sfuKeepaliveActive = false;
    await this.#saveState();

    // Best-effort Modal /end with a SHORT 3s timeout. If Modal is 429ing,
    // unreachable, or hanging, Worker-side cleanup still completes immediately.
    if (modalSessionId) {
      await endModalSessionBestEffort(this.env, modalSessionId, 3_000);
    }

    await this.#cleanupSfuTracksAndState();

    if (wasCapacityAcquired) {
      await this.#releaseSfuCapacity(true);
    }

    this.#state = emptyState();
    this.#state.endedAt = Date.now();
    this.#state.sfuKeepaliveAborted = true;
    await this.#saveState();

    return Response.json({
      ok: true,
      cleared: true,
      modalSessionId,
      wasActive,
      wasKeepaliveActive,
      wasCapacityAcquired,
    });
  }

  async #handleRestStart(request: Request): Promise<Response> {
    await this.#loadState();

    let body: { avatarId?: string; bgId?: string; transport?: TransportPreference } = {};
    try {
      const text = await request.text();
      if (text) body = JSON.parse(text);
    } catch {
      return Response.json(
        { error: { code: "invalid_body", message: "Request body must be valid JSON" } },
        { status: 400 },
      );
    }

    if (this.#state.active) {
      const rawRequested = body.avatarId || "default";
      const resolvedRequested = rawRequested === "default" ? "arnold" : rawRequested;
      const currentAvatarId = this.#state.avatarId || "default";
      if (resolvedRequested === currentAvatarId) {
        return Response.json({
          session_id: this.#sessionId,
          status: "ready",
          render_mode: "fmp4_av",
          stream_url: `/v1/sessions/${this.#sessionId}/stream`,
          state_note: "already_started",
          modal_session_id: this.#state.modalSessionId,
        });
      }
      return Response.json(
        {
          error: {
            code: "session_conflict",
            message: `Session already active with avatarId=${currentAvatarId}, requested ${resolvedRequested} (raw: ${rawRequested})`,
          },
        },
        { status: 409 },
      );
    }

    if (this.#state.starting) {
      const age = Date.now() - (this.#state.startingAt ?? 0);
      if (age < STARTING_STALE_MS) {
        return Response.json(
          { error: { code: "session_starting", message: "Session start already in progress" } },
          { status: 409 },
        );
      }
      this.#captureRest("session_starting_stale_cleared", { ageMs: age });
      this.#state.starting = false;
      this.#state.startingAt = undefined;
      this.#state.startAttemptId = null;
    }

    const resolvedAvatarId = (body.avatarId || "default") === "default" ? "arnold" : (body.avatarId || "default");

    // Phase 6a.1: validate resolved avatar/background IDs against the catalog
    // before the expensive Modal start call.  REST /start already validated at
    // the Worker edge, but the DO path may also be reached internally, so keep
    // this guard idempotent and safe.
    const validation = await validateAvatarBgForStart(
      this.env,
      await this.#getOwnerTenantId(),
      resolvedAvatarId,
      body.bgId,
    );
    if (!validation.ok) {
      this.#captureRest("session_start_catalog_validation_failed", {
        requestedAvatarId: body.avatarId ?? null,
        resolvedAvatarId,
        bgId: body.bgId,
        code: validation.code,
      });
      return Response.json(
        { error: { code: validation.code, message: validation.message } },
        { status: 400 },
      );
    }

    const use4CPath = !!this.env.MODAL_SESSION_STREAM_FMP4_AV_URL
      && !!this.env.MODAL_SESSION_START_URL;
    const useLegacyPath = !use4CPath && isLegacyModalEnabled(this.env);

    if (!use4CPath && !useLegacyPath) {
      return Response.json(
        {
          error: {
            code: "no_modal_protocol_enabled",
            message: "Neither 4C nor legacy Modal protocol is configured",
          },
        },
        { status: 503 },
      );
    }

    // ── Phase D1: SFU transport decision (pure helper) ──
    const transportPref = body.transport as TransportPreference | undefined;
    const transportDecision = resolveTransportDecision(this.env, transportPref, use4CPath);

    // Explicit webrtc_sfu but config incomplete → hard error (no fallback)
    if (transportPref === "webrtc_sfu" && !transportDecision.shouldTrySfu) {
      return Response.json(
        { error: { code: "sfu_not_configured", message: "SFU explicitly requested but prerequisites missing (AVTR_REALTIME_SFU_ENABLED, CALLS_APP_ID, CALLS_APP_SECRET, MODAL_WORKER_SECRET, WORKER_BASE_URL)" } },
        { status: 503 },
      );
    }

    // Phase 3F: explicit webrtc_sfu also requires private beta authorization.
    if (transportPref === "webrtc_sfu" && !(await this.#isSfuBetaAuthorized())) {
      return Response.json(
        { error: { code: "sfu_beta_not_enabled", message: "SFU transport is not enabled for this session. Request access via a beta-authorized tenant or a valid SFU beta token." } },
        { status: 403 },
      );
    }

    const shouldTrySfu = transportDecision.shouldTrySfu;

    const attemptId = crypto.randomUUID();
    this.#state.starting = true;
    this.#state.startingAt = Date.now();
    this.#state.startAttemptId = attemptId;
    this.#state.startCancelRequested = false;
    this.#state.endedAt = undefined;
    await this.#saveState();

    this.#captureRest("session_start_requested", {
      requestedAvatarId: body.avatarId ?? null,
      bgId: body.bgId,
      startAttemptId: attemptId,
    });

    const startedAt = Date.now();
    const useTestHook = this.env.AVTR_TEST_HOOK_SECRET
      && request.headers.get("X-Test-Hook") === this.env.AVTR_TEST_HOOK_SECRET;

    try {
      let result: Record<string, unknown> = {};

      if (use4CPath) {
        let usingSfu = shouldTrySfu;
        let sfuSessionId: string | null = null;

        // ── SFU path: LAZY session creation in publisher-offer callback ──
        // The SFU session is NOT created here. It is created lazily inside
        // #handlePublisherOffer when Modal calls back with the offer SDP.
        // This avoids the 30s SFU session TTL expiring during Modal cold
        // start (CF Realtime sessions GC after 30s of inactivity).
        if (usingSfu && !useTestHook) {
          // PRE-SAVE SFU state BEFORE Modal call
          this.#state.transport = "webrtc_sfu";
          this.#state.sfuSessionId = null;
          this.#state.sfuPublisherMids = { video: null, audio: null };
          this.#state.sfuSubscriberMids = [];
          this.#state.sfuPublisherTrackName = null;
          this.#state.sfuSubscriberSessionId = null;
          await this.#saveState();

          // Generate TURN credentials (critical path — Modal needs them
          // before ICE gathering). On failure, emit error + cleanup.
          let restTurnIceServers: unknown[] | null = null;
          try {
            restTurnIceServers = await this.#generateTurnCredentials();
          } catch (turnErr) {
            const turnMsg = turnErr instanceof SfuClientError ? turnErr.message : String(turnErr);
            this.#captureRest("sfu_turn_credentials_error", { error: turnMsg });
            // No capacity acquired — tracks-only cleanup, no release owed.
            await this.#cleanupSfuTracksAndState();
            this.#state.transport = "websocket_fmp4";
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            return Response.json(
              { error: { code: "sfu_turn_credentials_unavailable", message: turnMsg } },
              { status: 503 },
            );
          }

          // Acquire SFU capacity slot — prevents GPU contention from
          // overlapping SFU publishers on the same Modal container.
          // FAIL-CLOSED: capacity manager outage → sfu_capacity_unavailable.
          const restCapacityResult = await this.#acquireSfuCapacity();
          if (!restCapacityResult.ok) {
            if (restCapacityResult.unavailable) {
              this.#captureRest("sfu_capacity_unavailable", { reason: "manager_unreachable" });
              await this.#cleanupSfuTracksAndState();
              this.#state.transport = "websocket_fmp4";
              this.#state.starting = false;
              this.#state.startAttemptId = null;
              await this.#saveState();
              return Response.json(
                { error: { code: "sfu_capacity_unavailable", message: "SFU capacity manager is unreachable. Retry shortly." } },
                { status: 503 },
              );
            }
            const conflictMsg = restCapacityResult.conflict
              ? `Another SFU session is still active (${restCapacityResult.conflict.slice(-12)}). Wait for it to end.`
              : "SFU capacity exceeded.";
            this.#captureRest("sfu_capacity_exceeded", { conflict: restCapacityResult.conflict ?? null });
            // Acquire never succeeded — tracks-only cleanup, no release owed.
            await this.#cleanupSfuTracksAndState();
            this.#state.transport = "websocket_fmp4";
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            return Response.json(
              { error: { code: "sfu_capacity_exceeded", message: conflictMsg } },
              { status: 503 },
            );
          }

          result = await startModalSession(this.env, {
            session_id: this.#sessionId,
            avatar_id: resolvedAvatarId,
            bg_id: body.bgId || undefined,
            config: null,
            publisher_mode: "webrtc_sfu",
            publisher_callback_url: `${this.env.WORKER_BASE_URL}/internal/sessions/${this.#sessionId}/publisher-offer`,
            close_callback_url: `${this.env.WORKER_BASE_URL}/internal/sessions/${this.#sessionId}/publisher-close`,
            ...(restTurnIceServers ? { ice_servers: restTurnIceServers } : {}),
          }).then(r => r.data) as Record<string, unknown>;

          await this.#loadState();
          if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
            // Capacity was acquired. Modal orphan /end first, then release.
            if (result && !("error" in result) && result.session_id) {
              await endModalSessionBestEffort(this.env, result.session_id as string, 3_000);
            }
            await this.#cleanupSfuTracksAndState();
            await this.#releaseSfuCapacity();
            return Response.json(
              { error: { code: "session_cancelled", message: "Session was ended while starting" } },
              { status: 409 },
            );
          }

          if ("error" in result) {
            const orphanId = result.session_id as string | undefined;
            if (orphanId) {
              await endModalSessionBestEffort(this.env, orphanId, 3_000);
              this.#captureRest("sfu_fallback_modal_orphan_cleaned", { modalOrphanId: orphanId });
            }
            // Orphan Modal /end done above (best-effort). Now safe to release.
            await this.#cleanupSfuTracksAndState();
            await this.#releaseSfuCapacity();

            // Forensic: emit allowlisted sfu_ice_debug event (fire-and-forget).
            const restRawIceDebug = (result.ice_debug ?? null) as Record<string, unknown> | null;
            if (restRawIceDebug) {
              this.#captureRest("sfu_ice_debug", this.#extractIceDebugAllowlist(restRawIceDebug));
            }

            if (!transportDecision.fallbackAllowed) {
              this.#state.transport = "websocket_fmp4";
              this.#state.starting = false;
              this.#state.startAttemptId = null;
              await this.#saveState();
              this.#captureRest("sfu_start_failed_explicit", {
                error: result.error,
                // Stage 3B reliability repair: surface health_check_detail and
                // container_poisoned from Modal so PostHog shows the exact
                // failure reason + poisoning outcome.
                healthCheckDetail: result.health_check_detail ?? null,
                publisherStopMetrics: result.publisher_stop_metrics ?? null,
                containerPoisoned: result.container_poisoned === true,
                poisoningReason: result.poisoning_reason ?? null,
              });
              return Response.json(
                {
                  error: {
                    code: "sfu_unavailable",
                    message: (result.error as string) || "SFU Modal start failed",
                    // Include diagnostic detail (no secrets) for operators.
                    ...(result.health_check_detail ? { health_check_detail: result.health_check_detail } : {}),
                    ...(result.container_poisoned === true ? { container_poisoned: true } : {}),
                  },
                },
                { status: 502 },
              );
            }

            this.#captureRest("sfu_fallback_to_fmp4", { reason: result.error as string, modalOrphanCleaned: !!orphanId });
            this.#state.transport = "websocket_fmp4";
            this.#state.sfuSessionId = null;
            await this.#saveState();
            usingSfu = false;
          }
        } else if (usingSfu && useTestHook) {
          // Test hook path — mock the result, no real SFU
          usingSfu = false;
        }

        if (!usingSfu) {
          if (useTestHook) {
            const delayMs = parseInt(request.headers.get("X-Test-Hook-Delay") ?? "2000", 10);
            await scheduler.wait(delayMs);
            result = {
              session_id: `test-${this.#sessionId}`,
              avatar_id: resolvedAvatarId,
              status: "ready",
              cache_hit: false,
              init_s: 0,
              active_sessions: [],
              state_note: "test_mock",
              vram_mb: 0,
            };
          } else {
            result = await startModalSession(this.env, {
              session_id: this.#sessionId,
              avatar_id: resolvedAvatarId,
              bg_id: body.bgId || undefined,
              config: null,
            }).then(r => r.data) as Record<string, unknown>;
          }

          await this.#loadState();
          if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
            if (result && !("error" in result) && result.session_id) {
              const orphanId = result.session_id as string;
              let cleanupOk = true;
              let cleanupError: string | undefined;
              try {
                await endModalSession(this.env, orphanId, { timeoutMs: 3_000 }).then(r => r.data);
              } catch (e) {
                cleanupOk = false;
                cleanupError = e instanceof Error ? e.message : String(e);
              }
              this.#captureRest("modal_start_cancelled_cleanup", {
                modalSessionId: orphanId,
                cancelledBy: "delete_during_start",
                cleanupOk,
                ...(cleanupError ? { cleanupError } : {}),
              });
            }
            return Response.json(
              { error: { code: "session_cancelled", message: "Session was ended while starting" } },
              { status: 409 },
            );
          }

          if ("error" in result) {
            this.#state.starting = false;
            this.#state.startAttemptId = null;
            await this.#saveState();
            this.#captureRest("modal_start_failed", { error: result.error });
            return Response.json(
              { error: { code: "modal_start_failed", message: (result.error as string) || "Modal start_session returned error" } },
              { status: 502 },
            );
          }
        }

        // ── Success — save final state, preserving SFU callback fields ──
        // The publisher-offer callback may have run DURING startModalSession()
        // (via /internal/sessions/:id/publisher-offer) and written
        // sfuPublisherMids.video + sfuPublisherTrackName to storage. State was
        // reloaded at the post-Modal loadState() above, so this.#state already
        // holds those callback-written values. Capture and clone them before
        // building the final state object so they survive the assignment.
        const sfuSuccess = usingSfu;
        const savedPublisherMids = {
          video: this.#state.sfuPublisherMids?.video ?? null,
          audio: this.#state.sfuPublisherMids?.audio ?? null,
        };
        const savedSubscriberMids = [...(this.#state.sfuSubscriberMids ?? [])];
        const savedPublisherTrackName = this.#state.sfuPublisherTrackName ?? null;
        const savedSubscriberSessionId = this.#state.sfuSubscriberSessionId ?? null;

        this.#state = {
          modalSessionId: result.session_id as string,
          active: true,
          avatarId: result.avatar_id as string,
          bgId: null,
          createdAt: this.#state.createdAt ?? Date.now(),
          startedAt: Date.now(),
          starting: false,
          startAttemptId: null,
          startCancelRequested: false,
          ...(sfuSuccess ? {
            transport: "webrtc_sfu" as const,
            sfuSessionId: this.#state.sfuSessionId,
            // Preserve publisher mid + track name written by the callback.
            sfuPublisherMids: { ...savedPublisherMids },
            sfuSubscriberMids: [...savedSubscriberMids],
            sfuPublisherTrackName: savedPublisherTrackName ?? (result.publisher_track_name as string) ?? null,
            sfuSubscriberSessionId: savedSubscriberSessionId,
            // Preserve capacity ownership flag set by #acquireSfuCapacity().
            sfuCapacityAcquired: this.#state.sfuCapacityAcquired ?? false,
          } : {
            transport: "websocket_fmp4" as const,
          }),
        };
        await this.#saveState();

        // Cost-safety final guard for REST path (same as WS path).
        if (sfuSuccess) {
          const wasCancelled = await this.#cleanupCancelledSfuStart(
            result.session_id as string,
            attemptId,
            null,
            "post_start_guard_rest",
          );
          if (wasCancelled) {
            return Response.json(
              { error: { code: "session_cancelled", message: "Session was ended while starting" } },
              { status: 409 },
            );
          }
        }

        this.#captureRest("session_ready", {
          modalSessionId: result.session_id,
          resolvedAvatarId,
          latencyMs: Date.now() - startedAt,
          protocol: useTestHook ? "4c_rest_test_hook" : "4c_rest",
          transport: this.#state.transport,
          ...(this.#state.sfuSessionId ? { sfuSessionId: this.#state.sfuSessionId } : {}),
        });

        // Forensic: emit candidate summary on success for pass-vs-fail comparison.
        const restSuccessIceDebug = (result.ice_debug_success ?? null) as Record<string, unknown> | null;
        if (restSuccessIceDebug && this.#state.transport === "webrtc_sfu") {
          this.#captureRest("sfu_ice_debug_success", this.#extractIceDebugAllowlist(restSuccessIceDebug));
        }

        // Stage 3B: Start DO-alarm keepalive for SFU sessions
        if (this.#state.transport === "webrtc_sfu") {
          await this.#startSfuKeepalive();
        }

        return Response.json({
          session_id: this.#sessionId,
          status: "ready",
          transport: this.#state.transport,
          render_mode: this.#state.transport === "webrtc_sfu" ? null : "fmp4_av",
          stream_url: this.#state.transport === "webrtc_sfu" ? null : `/v1/sessions/${this.#sessionId}/stream`,
          state_note: "durable-object-owned",
          modal_session_id: result.session_id,
          avatar_id: result.avatar_id,
          ...(this.#state.transport === "webrtc_sfu" ? {
            sfu: {
              subscribe_url: `/v1/sessions/${this.#sessionId}/sfu/subscribe`,
              answer_url: `/v1/sessions/${this.#sessionId}/sfu/answer`,
              close_url: `/v1/sessions/${this.#sessionId}/sfu/close`,
              publisher_track_name: this.#state.sfuPublisherTrackName ?? "",
            },
          } : {}),
        });
      }

      if (useTestHook) {
        const delayMs = parseInt(request.headers.get("X-Test-Hook-Delay") ?? "2000", 10);
        await scheduler.wait(delayMs);
        result = { sessionId: `test-${this.#sessionId}`, avatarId: resolvedAvatarId, bgId: body.bgId ?? null };
      } else {
        result = await startSession(
          this.env,
          body.avatarId,
          body.bgId,
        ).then(r => r.data) as Record<string, unknown>;
      }

      await this.#loadState();
      if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
        if (result && !("error" in result) && result.sessionId) {
          const orphanId = result.sessionId as string;
          let cleanupOk = true;
          let cleanupError: string | undefined;
          try {
            await endSession(this.env, orphanId).then(r => r.data);
          } catch (e) {
            cleanupOk = false;
            cleanupError = e instanceof Error ? e.message : String(e);
          }
          this.#captureRest("modal_start_cancelled_cleanup", {
            modalSessionId: orphanId,
            cancelledBy: "delete_during_start",
            cleanupOk,
            ...(cleanupError ? { cleanupError } : {}),
          });
        }
        return Response.json(
          { error: { code: "session_cancelled", message: "Session was ended while starting" } },
          { status: 409 },
        );
      }

      if ("error" in result) {
        this.#state.starting = false;
        this.#state.startAttemptId = null;
        await this.#saveState();
        this.#captureRest("modal_start_failed", { error: (result as { error: string }).error });
        return Response.json(
          { error: { code: "modal_start_failed", message: (result as { error: string }).error || "Modal start returned error" } },
          { status: 502 },
        );
      }

      this.#state = {
        modalSessionId: result.sessionId as string,
        active: true,
        avatarId: result.avatarId as string,
        bgId: result.bgId as string,
        createdAt: this.#state.createdAt ?? Date.now(),
        startedAt: Date.now(),
        starting: false,
        startAttemptId: null,
        startCancelRequested: false,
      };
      await this.#saveState();

      this.#captureRest("session_ready", {
        modalSessionId: result.sessionId,
        latencyMs: Date.now() - startedAt,
        protocol: useTestHook ? "legacy_rest_test_hook" : "legacy_rest",
      });

      return Response.json({
        session_id: this.#sessionId,
        status: "ready",
        render_mode: "jpeg_timeline",
        state_note: "durable-object-owned",
        modal_session_id: result.sessionId,
        avatar_id: result.avatarId,
      });
    } catch (err) {
      await this.#loadState();
      if (this.#state.startAttemptId !== attemptId || this.#state.startCancelRequested) {
        return Response.json(
          { error: { code: "session_cancelled", message: "Session was ended while starting" } },
          { status: 409 },
        );
      }
      // Cost-safety: release SFU capacity if Modal start threw after acquire.
      await this.#cleanupSfuTracksAndState();
      await this.#releaseSfuCapacity();
      this.#state = emptyState();
      this.#state.endedAt = Date.now();
      await this.#saveState();
      const msg = err instanceof ModalClientError
        ? err.message
        : `Modal start failed: ${err instanceof Error ? err.message : String(err)}`;
      this.#captureRest("modal_start_error", { error: msg });
      return Response.json(
        { error: { code: "modal_unreachable", message: msg } },
        { status: 500 },
      );
    }
  }

  async #handleRestEnd(): Promise<Response> {
    await this.#loadState();

    const wasActive = this.#state.active;
    const wasStarting = this.#state.starting;
    const modalSessionId = this.#state.modalSessionId;

    if (!wasActive && wasStarting) {
      this.#captureRest("session_start_cancelled_by_end", {
        modalSessionId,
        startAttemptId: this.#state.startAttemptId,
      });
      this.#state.starting = false;
      this.#state.startingAt = undefined;
      this.#state.startAttemptId = null;
      this.#state.startCancelRequested = true;
      this.#state.endedAt = Date.now();
      await this.#saveState();
      return Response.json({
        session_id: this.#sessionId,
        status: "ended",
        modal_end_ok: false,
        modal_end_error: undefined,
        modal_end_duration_ms: 0,
        summary: {},
        start_cancelled: true,
      });
    }

    this.#captureRest("session_end_requested", {
      modalSessionId,
      wasActive,
    });

    this.#sessionClosing = true;
    this.#idleLiveEnabled = false;
    this.#assistantSpeechActive = false;
    this.#pendingPlaybackAckTurnId = null;
    this.#activeSpeechRenderTurnId = null;
    this.#avatarTimelineAbort = true;
    this.#avatarSpeechQueue = [];

    if (this.#modalTimelineStreamActive && modalSessionId) {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
      try {
        await stopModalTimeline(this.env, modalSessionId);
      } catch {
        // best-effort
      }
    }

    if (this.#modalFmp4StreamActive && modalSessionId) {
      const isAvMode = this.env.ENABLE_MODAL_FMP4_AV_STREAM === "true";
      this.#modalTimelineAbort = true;

      try {
        await stopModalTimeline(this.env, modalSessionId, { useControlEndpoint: isAvMode });
      } catch {
        // best-effort
      }

      if (!this.#fmp4StreamSummaryCaptured && this.#modalFmp4StreamActive) {
        const endPromise = new Promise<void>((resolve) => {
          this.#fmp4StreamEndResolve = resolve;
        });
        const timeoutPromise = new Promise<void>((resolve) => {
          setTimeout(resolve, 5_000);
        });
        await Promise.race([endPromise, timeoutPromise]);
        this.#fmp4StreamEndResolve = null;
      }

      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    } else {
      this.#modalTimelineAbort = true;
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
        this.#modalTimelineAbortController = null;
      }
    }

    let modalEndOk = false;
    let modalEndError: string | undefined;
    let modalEndDurationMs = 0;
    let endSummary: Record<string, unknown> = {};

    // Phase D1: SFU teardown — tracks cleanup runs first, Modal /end runs
    // next, capacity is released LAST (in a finally below) so the global
    // slot stays reserved while the GPU publisher is tearing down.
    const wasCapacityAcquired = this.#state.sfuCapacityAcquired === true;
    if (wasActive && this.#state.transport === "webrtc_sfu") {
      await this.#cleanupSfuTracksAndState();
    }

    if (wasActive && modalSessionId) {
      const modalEndStart = Date.now();
      try {
        const use4CEnd = !!this.env.MODAL_SESSION_END_URL && !!this.env.MODAL_SESSION_START_URL;
        const result = use4CEnd
          ? await endModalSession(this.env, modalSessionId).then(r => r.data)
          : await endSession(this.env, modalSessionId).then(r => r.data);
        modalEndOk = true;
        modalEndDurationMs = Date.now() - modalEndStart;

        if ("chunks_rendered" in result) {
          endSummary = {
            chunks_rendered: result.chunks_rendered,
            fmp4_media_count: result.fmp4_media_count,
            fmp4_media_total_bytes: result.fmp4_media_total_bytes,
            duration_s: result.duration_s,
            ffmpeg_rc: result.ffmpeg_rc,
          };
        }

        // Stage 3B reliability repair: surface publisher stop metrics and
        // container poisoning flag from Modal so we can detect CUDA-class
        // failures via PostHog (publisher_stop_metrics, container_poisoned,
        // poisoning_reason).  These fields are populated by the 5q app's
        // hardened _end_session_impl when the WebRTC publisher's producer
        // thread hit a fatal error.
        const endResultRaw = result as unknown as Record<string, unknown>;
        const publisherStopMetrics = endResultRaw.publisher_stop_metrics;
        const containerPoisoned = endResultRaw.container_poisoned;
        const poisoningReason = endResultRaw.poisoning_reason;
        if (publisherStopMetrics || containerPoisoned === true) {
          endSummary = {
            ...endSummary,
            publisher_stop_metrics: publisherStopMetrics ?? null,
            container_poisoned: containerPoisoned === true,
            poisoning_reason: poisoningReason ?? null,
          };
        }

        this.#captureRest("modal_end_completed", {
          modalSessionId,
          modalEndDurationMs,
          protocol: use4CEnd ? "4c_rest" : "legacy_rest",
          ...(containerPoisoned === true ? { containerPoisoned: true } : {}),
          ...(publisherStopMetrics
            ? { publisherStopMetrics: JSON.stringify(publisherStopMetrics).slice(0, 1200) }
            : {}),
        });
      } catch (err) {
        modalEndOk = false;
        modalEndDurationMs = Date.now() - modalEndStart;
        modalEndError = err instanceof Error ? err.message : String(err);
        this.#captureRest("modal_end_error", {
          modalSessionId,
          modalEndDurationMs,
          error: modalEndError,
        });
      } finally {
        // GPU publisher gone (or best-effort cleanup failed). NOW release
        // the capacity slot so the next session inherits a clean L40S.
        if (wasCapacityAcquired) {
          await this.#releaseSfuCapacity(true);
        }
      }
    } else if (wasCapacityAcquired) {
      // Defensive: no Modal session to end, but capacity was somehow acquired.
      await this.#releaseSfuCapacity(true);
    }

    this.#audioChunkCount = 0;
    this.#audioBytesReceived = 0;
    this.#firstAudioChunkCaptured = false;

    this.#state = {
      ...emptyState(),
      createdAt: this.#state.createdAt,
      startedAt: this.#state.startedAt,
      endedAt: Date.now(),
    };
    this.#conversationHistory = [];
    await this.#saveState();

    this.#captureRest("session_ended", {
      modalSessionId,
      wasActive,
      modalEndOk,
      modalEndDurationMs,
      ...(endSummary.publisher_stop_metrics
        ? { publisherStopMetrics: JSON.stringify(endSummary.publisher_stop_metrics).slice(0, 1200) }
        : {}),
      ...(endSummary.container_poisoned === true ? { containerPoisoned: true } : {}),
      ...(endSummary.poisoning_reason ? { poisoningReason: String(endSummary.poisoning_reason).slice(0, 400) } : {}),
    });

    return Response.json({
      session_id: this.#sessionId,
      status: "ended",
      modal_end_ok: modalEndOk,
      modal_end_error: modalEndError,
      modal_end_duration_ms: modalEndDurationMs,
      summary: endSummary,
    });
  }

  async #handleRestStatus(): Promise<Response> {
    await this.#loadState();

    const response: Record<string, unknown> = {
      session_id: this.#sessionId,
      status: this.#state.active ? "active" : (this.#state.endedAt ? "ended" : "created"),
      render_mode: this.#state.active ? "fmp4_av" : null,
      avatar_id: this.#state.avatarId,
      modal_session_id: this.#state.modalSessionId,
      created_at: this.#state.createdAt ?? null,
      started_at: this.#state.startedAt ?? null,
      ended_at: this.#state.endedAt ?? null,
    };

    if (this.#state.active && this.#state.modalSessionId && this.env.MODAL_SESSION_STATUS_URL) {
      try {
        const result = await getModalSessionStatus(this.env, this.#state.modalSessionId).then(r => r.data);
        const { active_sessions: _, ...sanitized } = result as unknown as Record<string, unknown>;
        response.modal_status = sanitized;
      } catch {
        response.modal_status = { error: "unreachable" };
      }
    }

    return Response.json(response);
  }

  // ── Phase 5C: REST SSE fMP4 AV stream ───────────────────────────

  async #handleRestStream(request: Request): Promise<Response> {
    await this.#loadState();

    if (!this.#state.active || !this.#state.modalSessionId) {
      return Response.json(
        { error: { code: "session_not_started", message: "Session must be started before streaming. Call POST /v1/sessions/:id/start first." } },
        { status: 409 },
      );
    }

    if (this.#modalTimelineStreamActive) {
      if (this.#modalTimelineAbortController) {
        this.#modalTimelineAbortController.abort();
      }
      this.#modalTimelineStreamActive = false;
      this.#modalTimelineAbortController = null;
      this.#captureRest("rest_stream_replaced", { modalSessionId: this.#state.modalSessionId });
    }

    const modalSessionId = this.#state.modalSessionId;
    this.#modalTimelineStreamActive = true;
    this.#modalTimelineAbort = false;
    const streamAbortController = new AbortController();
    this.#modalTimelineAbortController = streamAbortController;

    this.#captureRest("rest_stream_started", { modalSessionId });

    const clientDisconnect = () => {
      safeClose();
      streamAbortController.abort();
    };
    request.signal.addEventListener("abort", clientDisconnect);

    const { readable: rawReadable, writable } = new TransformStream();
    const writer = writable.getWriter();
    const encoder = new TextEncoder();

    let closed = false;
    let writeChain: Promise<void> = Promise.resolve();

    const doClose = (): void => {
      closed = true;
      this.#modalTimelineStreamActive = false;
      this.#modalTimelineAbortController = null;
      request.signal.removeEventListener("abort", clientDisconnect);
    };

    const safeWriteSse = (eventName: string, payload: Record<string, unknown>): void => {
      if (closed) return;
      const raw = `event: ${eventName}\ndata: ${JSON.stringify(payload)}\n\n`;
      const encoded = encoder.encode(raw);
      writeChain = writeChain.then(() => writer.write(encoded)).catch(() => {
        if (!closed) doClose();
        streamAbortController.abort();
      });
    };

    const safeClose = (): void => {
      if (closed) return;
      writeChain = writeChain.then(() => writer.close()).catch(() => {});
      doClose();
    };

    const safeAbort = (error: string): void => {
      if (closed) return;
      safeWriteSse("error", { code: "modal_stream_error", message: error });
      safeClose();
      this.#captureRest("rest_stream_error", { error });
    };

    startModalSessionFmp4Stream(
      this.env,
      modalSessionId,
      {
        onInit: (event) => {
          safeWriteSse("fmp4_init", {
            dataBase64: event.dataBase64,
            byteLength: event.byteLength,
            timing: event.timing,
          });
        },

        onMedia: (event) => {
          this.#modalFmp4SegmentCount++;
          this.#avStreamMediaReceived++;
          this.#fmp4LastStreamMode = event.mode ?? null;
          this.#fmp4LastStreamSequence = event.sequence ?? null;
          if (event.sequence != null) {
            this.#avStreamLastSequenceReceived = event.sequence;
          }
          safeWriteSse("fmp4_media", {
            dataBase64: event.dataBase64,
            byteLength: event.byteLength,
            sequence: event.sequence,
            mode: event.mode ?? "idle",
            isSpeechStart: event.isSpeechStart ?? false,
            isSpeechEnd: event.isSpeechEnd ?? false,
            turnId: event.turnId ?? null,
            segmentId: event.segmentId ?? 0,
            chunkIndex: event.chunkIndex ?? 0,
            chunkRenderMs: event.timing?.chunkRenderMs ?? 0,
            feedMs: event.timing?.feedMs ?? 0,
            framesFed: event.timing?.framesFed ?? 0,
          });
        },

        onEnd: (event) => {
          this.#captureRest("rest_stream_ended", {
            totalSegments: this.#modalFmp4SegmentCount,
            stopReason: event.stopReason,
          });
          safeWriteSse("end", {
            stopReason: event.stopReason,
            totalSegments: this.#modalFmp4SegmentCount,
            receivedCount: this.#avStreamMediaReceived,
          });
          safeClose();
        },

        onOverflow: (info) => {
          this.#captureRest("media_queue_overflow_graceful", {
            droppedCount: info.droppedCount,
            queueLen: info.queueLen,
            firstDroppedSeq: info.firstDroppedSeq,
            lastDroppedSeq: info.lastDroppedSeq,
            idleDropped: info.idleDropped,
            mode: info.mode,
          });
        },

        onStats: (stats) => {
          this.#captureRest("rest_stream_stats", {
            maxQueueDepth: stats.maxQueueDepth,
            receivedMediaCount: stats.receivedMediaCount,
            forwardedMediaCount: stats.forwardedMediaCount,
            overflowCount: stats.overflowCount,
            queueCapacity: stats.queueCapacity,
            adaptiveFastForwardCount: stats.adaptiveFastForwardCount,
            minForwardDelayMs: stats.minForwardDelayMs,
            maxForwardDelayMs: stats.maxForwardDelayMs,
            backpressureWaitCount: stats.backpressureWaitCount,
            backpressureWaitMs: stats.backpressureWaitMs,
            highWater: stats.highWater,
            lowWater: stats.lowWater,
          });
        },
      },
      streamAbortController.signal,
      this.env.FMP4_RELAY_PACE_MS ? parseInt(this.env.FMP4_RELAY_PACE_MS, 10) : 200,
    ).catch((err) => {
      const msg = err instanceof ModalClientError
        ? err.message
        : err instanceof Error ? err.message : String(err);
      console.error("[AvatarSessionDO] rest_stream_error", {
        sessionId: this.#sessionId,
        modalSessionId,
        error: msg,
      });
      safeAbort(msg);
    });

    const doRef = this;
    const readable = new ReadableStream({
      start(controller) {
        const reader = rawReadable.getReader();
        const pump = (): void => {
          reader.read().then(({ done, value }) => {
            if (done) {
              controller.close();
              return;
            }
            controller.enqueue(value);
            pump();
          }).catch(() => {
            controller.close();
          });
        };
        pump();
      },
      cancel() {
        if (!closed) {
          doClose();
        }
        streamAbortController.abort();
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, x-api-key",
      },
    });
  }

  // ── Phase D1: SFU REST handlers ───────────────────────────────────────
  //
  // All SFU API calls go through the Worker/DO. Neither the browser nor
  // Modal ever hold CALLS_APP_SECRET or call the SFU directly.

  #sfuRestResponse(data: Record<string, unknown>, status = 200): Response {
    return Response.json(data, {
      status,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, PUT, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization",
      },
    });
  }

  #sfuRestError(code: string, message: string, status = 400): Response {
    return this.#sfuRestResponse({ error: { code, message } }, status);
  }

  async #handleSfuSubscribe(request: Request): Promise<Response> {
    await this.#loadState();

    this.#captureRest("sfu_subscribe_do_entered", {});

    // Verify session is SFU transport
    if (this.#state.transport !== "webrtc_sfu" || !this.#state.sfuSessionId) {
      this.#captureRest("sfu_subscribe_state_missing", {
        transport: this.#state.transport,
        hasSfuSessionId: !!this.#state.sfuSessionId,
        hasPublisherTrack: !!this.#state.sfuPublisherTrackName,
      });
      return this.#sfuRestError("sfu_wrong_transport", "Session is not using SFU transport", 409);
    }
    if (!this.#state.sfuPublisherTrackName) {
      return this.#sfuRestError("sfu_session_not_found", "Publisher track not ready yet", 409);
    }

    // We need the publisher's SFU session ID (same session for D1 single-publisher)
    const publisherSfuSessionId = this.#state.sfuSessionId;
    const trackName = this.#state.sfuPublisherTrackName;

    try {
      // Create a new SFU session for the subscriber
      const subscriberSfuSessionId = await createSfuSession(this.env);

      // Subscribe to the publisher's track (no SDP — SFU will send an offer)
      const result = await subscriberTracksNew(this.env, {
        sfuSessionId: subscriberSfuSessionId,
        publisherSfuSessionId,
        trackName,
      });

      // Store subscriber session info
      if (!this.#state.sfuSubscriberMids) this.#state.sfuSubscriberMids = [];
      this.#state.sfuSubscriberMids.push(...result.mids);
      // Store subscriber SFU session ID (needed for answer/close)
      this.#state.sfuSubscriberSessionId = subscriberSfuSessionId;
      await this.#saveState();

      this.#captureRest("sfu_subscribed", {
        subscriberSfuSessionId,
        publisherTrackName: trackName,
        mids: result.mids,
      });

      return this.#sfuRestResponse({
        offer_sdp: result.offerSdp,
        offer_type: result.offerType,
        requires_renegotiation: result.requiresRenegotiation,
        mids: result.mids,
      });
    } catch (err) {
      const msg = err instanceof SfuClientError ? err.message : String(err);
      this.#captureRest("sfu_subscribe_error", { error: msg });
      return this.#sfuRestError("sfu_unavailable", msg, 502);
    }
  }

  async #handleSfuAnswer(request: Request): Promise<Response> {
    await this.#loadState();

    if (this.#state.transport !== "webrtc_sfu") {
      return this.#sfuRestError("sfu_wrong_transport", "Session is not using SFU transport", 409);
    }

    const subscriberSfuSessionId = this.#state.sfuSubscriberSessionId as string | undefined;
    if (!subscriberSfuSessionId) {
      return this.#sfuRestError("sfu_session_not_found", "No active subscriber session", 404);
    }

    let body: { answer_sdp?: string; answer_type?: string } = {};
    try {
      const text = await request.text();
      if (text) body = JSON.parse(text);
    } catch {
      return this.#sfuRestError("invalid_body", "Request body must be valid JSON", 400);
    }

    if (!body.answer_sdp) {
      return this.#sfuRestError("invalid_body", "answer_sdp is required", 400);
    }

    try {
      await sfuRenegotiate(
        this.env,
        subscriberSfuSessionId,
        body.answer_sdp,
        body.answer_type || "answer",
      );

      this.#captureRest("sfu_answer_completed", { subscriberSfuSessionId });
      return this.#sfuRestResponse({ status: "ok" });
    } catch (err) {
      const msg = err instanceof SfuClientError ? err.message : String(err);
      this.#captureRest("sfu_answer_error", { error: msg });
      return this.#sfuRestError("sfu_unavailable", msg, 502);
    }
  }

  async #handleSfuClose(request: Request): Promise<Response> {
    await this.#loadState();

    const subscriberSfuSessionId = this.#state.sfuSubscriberSessionId as string | undefined;
    const subscriberMids = this.#state.sfuSubscriberMids ?? [];

    if (!subscriberSfuSessionId || subscriberMids.length === 0) {
      return this.#sfuRestResponse({ status: "no_active_subscriber" });
    }

    try {
      await sfuTracksClose(this.env, subscriberSfuSessionId, subscriberMids, true);

      this.#state.sfuSubscriberMids = [];
      this.#state.sfuSubscriberSessionId = null;
      await this.#saveState();

      this.#captureRest("sfu_subscriber_closed", { subscriberSfuSessionId });
      return this.#sfuRestResponse({ status: "closed" });
    } catch (err) {
      const msg = err instanceof SfuClientError ? err.message : String(err);
      this.#captureRest("sfu_close_error", { error: msg, bestEffort: true });
      // Best-effort: clear state even on error
      this.#state.sfuSubscriberMids = [];
      this.#state.sfuSubscriberSessionId = null;
      await this.#saveState();
      return this.#sfuRestResponse({ status: "closed_best_effort", error: msg });
    }
  }

  // ── Phase D1: Publisher callback handlers (internal, x-modal-secret auth) ──

  async #handlePublisherOffer(request: Request): Promise<Response> {
    await this.#loadState();

    // Verify callback session ID matches this DO session
    const urlPath = new URL(request.url).pathname;
    const pathSegs = urlPath.split("/").filter(Boolean);
    const callbackSessionId = pathSegs.length >= 3 ? decodeURIComponent(pathSegs[2]) : "";
    if (callbackSessionId !== this.#sessionId) {
      return this.#sfuRestError("sfu_session_not_found", "Session ID mismatch", 403);
    }

    // Verify session is SFU transport.
    // sfuSessionId may be null here — it is created lazily below.
    if (this.#state.transport !== "webrtc_sfu") {
      return this.#sfuRestError("sfu_wrong_transport", "Session is not using SFU transport", 409);
    }

    let body: {
      sdp?: string;
      type?: string;
      mid?: string;
      track_name?: string;
      kind?: string;
    } = {};
    try {
      const text = await request.text();
      if (text) body = JSON.parse(text);
    } catch {
      return this.#sfuRestError("invalid_body", "Request body must be valid JSON", 400);
    }

    if (!body.sdp || !body.mid || !body.track_name) {
      return this.#sfuRestError("invalid_body", "sdp, mid, and track_name are required", 400);
    }

    // LAZY SFU session creation — created here (not at session.start) to avoid
    // the 30s CF Realtime session TTL expiring during Modal cold start.
    // Modal never holds CALLS_APP_SECRET; the DO is the sole SFU API caller.
    if (!this.#state.sfuSessionId) {
      try {
        const lazySessionId = await createSfuSession(this.env);
        this.#state.sfuSessionId = lazySessionId;
        await this.#saveState();
        this.#captureRest("sfu_session_created", { sfuSessionId: lazySessionId, source: "publisher_offer_lazy" });
      } catch (e) {
        const msg = e instanceof SfuClientError ? e.message : String(e);
        this.#captureRest("sfu_session_create_error", { error: msg, stage: "publisher_offer_lazy" });
        return this.#sfuRestError("sfu_unavailable", `SFU session creation failed: ${msg}`, 502);
      }
    }

    try {
      const result = await publisherTracksNew(this.env, {
        sfuSessionId: this.#state.sfuSessionId,
        sdp: body.sdp,
        sdpType: body.type || "offer",
        mid: body.mid,
        trackName: body.track_name,
        kind: body.kind,
      });

      // Store publisher mid and track name
      if (!this.#state.sfuPublisherMids) this.#state.sfuPublisherMids = { video: null, audio: null };
      this.#state.sfuPublisherMids.video = body.mid;
      this.#state.sfuPublisherTrackName = body.track_name;
      await this.#saveState();

      this.#captureRest("sfu_publisher_offer_completed", {
        sfuSessionId: this.#state.sfuSessionId,
        mid: body.mid,
        trackName: body.track_name,
      });

      return this.#sfuRestResponse({
        answer_sdp: result.answerSdp,
        answer_type: result.answerType,
        requires_renegotiation: result.requiresRenegotiation,
      });
    } catch (err) {
      const msg = err instanceof SfuClientError ? err.message : String(err);
      this.#captureRest("sfu_publisher_offer_error", { error: msg });
      return this.#sfuRestError("sfu_unavailable", msg, 502);
    }
  }

  async #handlePublisherClose(request: Request): Promise<Response> {
    await this.#loadState();

    if (this.#state.transport !== "webrtc_sfu" || !this.#state.sfuSessionId) {
      return this.#sfuRestResponse({ status: "no_sfu_session" });
    }

    const publisherMids = this.#state.sfuPublisherMids ?? { video: null, audio: null };
    const midsToClose = Object.values(publisherMids).filter((m): m is string => !!m);

    if (midsToClose.length === 0) {
      return this.#sfuRestResponse({ status: "no_publisher_tracks" });
    }

    try {
      await sfuTracksClose(this.env, this.#state.sfuSessionId, midsToClose, true);

      this.#state.sfuPublisherMids = { video: null, audio: null };
      this.#state.sfuPublisherTrackName = null;
      await this.#saveState();

      this.#captureRest("sfu_publisher_closed", { mids: midsToClose });
      return this.#sfuRestResponse({ status: "closed" });
    } catch (err) {
      const msg = err instanceof SfuClientError ? err.message : String(err);
      this.#captureRest("sfu_publisher_close_error", { error: msg, bestEffort: true });
      this.#state.sfuPublisherMids = { video: null, audio: null };
      this.#state.sfuPublisherTrackName = null;
      await this.#saveState();
      return this.#sfuRestResponse({ status: "closed_best_effort", error: msg });
    }
  }

  // ── Stage 3B: SFU cost-safety helpers ─────────────────────────────

  /**
   * Best-effort cleanup of an SFU start that was cancelled while Modal start
   * was in-flight. Used both when cancellation is detected right after Modal
   * returns and as a final guard before session_ready / keepalive start.
   */
  async #cleanupCancelledSfuStart(
    modalSessionId: string | null | undefined,
    attemptId: string,
    ws: WebSocket | null,
    source: string,
  ): Promise<boolean> {
    await this.#loadState();

    // After the success state assignment, startAttemptId is reset to null.
    // The only reliable cancellation signal is startCancelRequested. We still
    // pass attemptId for forensic logging.
    if (this.#state.startCancelRequested !== true) {
      return false;
    }

    // Cancelled. Modal may have already started a GPU session; end it
    // best-effort with a SHORT 3s timeout before releasing capacity so the
    // next session inherits a clean GPU. Worker cleanup completes regardless.
    if (modalSessionId) {
      await endModalSessionBestEffort(this.env, modalSessionId, 3_000);
    }

    await this.#cleanupSfuTracksAndState();
    await this.#releaseSfuCapacity();

    this.#state = emptyState();
    this.#state.endedAt = Date.now();
    await this.#saveState();

    const props = {
      source,
      startAttemptId: attemptId,
      cancelledAttemptId: this.#state.startAttemptId,
      modalSessionId: modalSessionId ?? null,
    };
    if (ws) {
      this.#capture(ws, "sfu_start_cancelled_after_modal_return", props);
    } else {
      this.#captureRest("sfu_start_cancelled_after_modal_return", props);
    }

    return true;
  }

  /**
   * Unconditional SFU keepalive/session abort. Must complete Worker cleanup
   * even if Modal is unreachable, billing-limited, or 429ing. Never
   * reschedules the alarm.
   */
  async #abortSfuKeepalive(reason: string): Promise<void> {
    const startedAt = this.#state.sfuKeepaliveStartedAt ?? null;
    const modalSessionId = this.#state.modalSessionId ?? null;
    const failureCount = this.#state.sfuKeepaliveFailures ?? 0;
    const ageMs = startedAt ? Date.now() - startedAt : 0;

    this.#captureRest("sfu_modal_keepalive_aborted", {
      reason,
      modalSessionId,
      startedAt,
      ageMs,
      failureCount,
      keepaliveMaxAgeMs: this.#getSfuKeepaliveMaxAgeMs(),
      sessionMaxAgeMs: this.#getSfuSessionMaxAgeMs(),
    });

    await this.#stopSfuKeepalive(reason);

    if (modalSessionId) {
      // SHORT 3s timeout: do not let a stuck Modal /end block the abort
      // and allow the keepalive to somehow resurrect. We would rather
      // leave Modal to its own scaledown_window than block here.
      await endModalSessionBestEffort(this.env, modalSessionId, 3_000);
    }

    await this.#cleanupSfuTracksAndState();

    const wasCapacityAcquired = this.#state.sfuCapacityAcquired === true;
    if (wasCapacityAcquired) {
      await this.#releaseSfuCapacity(true);
    }

    this.#state = emptyState();
    this.#state.endedAt = Date.now();
    this.#state.sfuKeepaliveAborted = true;
    await this.#saveState();
  }

  // ── Stage 3B: SFU keepalive helpers ───────────────────────────────

  async #startSfuKeepalive(): Promise<void> {
    if (this.#state.transport !== "webrtc_sfu") return;
    if (!this.#state.active) return;
    if (!this.#state.modalSessionId) return;

    // Cost-safety: never start keepalive for a cancelled or aborted start.
    if (this.#state.startCancelRequested || this.#state.sfuKeepaliveAborted) {
      this.#captureRest("sfu_modal_keepalive_start_suppressed", {
        reason: this.#state.startCancelRequested ? "start_cancel_requested" : "keepalive_aborted",
        modalSessionId: this.#state.modalSessionId,
      });
      return;
    }

    this.#state.sfuKeepaliveActive = true;
    this.#state.sfuKeepaliveStartedAt = Date.now();
    this.#state.sfuKeepaliveLastHeartbeatAt = null;
    this.#state.sfuKeepaliveFailures = 0;
    this.#state.sfuKeepaliveStopReason = null;
    this.#state.sfuKeepaliveAborted = false;
    if (!this.#state.sfuSessionStartedAt) {
      this.#state.sfuSessionStartedAt = Date.now();
    }
    await this.#saveState();

    await this.ctx.storage.setAlarm(Date.now() + SFU_KEEPALIVE_INTERVAL_MS);

    this.#captureRest("sfu_modal_keepalive_started", {
      source: "do_alarm",
      modalSessionId: this.#state.modalSessionId,
      intervalMs: SFU_KEEPALIVE_INTERVAL_MS,
      keepaliveMaxAgeMs: this.#getSfuKeepaliveMaxAgeMs(),
      sessionMaxAgeMs: this.#getSfuSessionMaxAgeMs(),
      maxFailures: this.#getSfuKeepaliveMaxFailures(),
    });
  }

  async #stopSfuKeepalive(reason: string): Promise<void> {
    const hadAlarm = await this.ctx.storage.getAlarm();
    const wasActive = this.#state.sfuKeepaliveActive === true;
    if (!hadAlarm && !wasActive) return;

    await this.ctx.storage.deleteAlarm();
    this.#state.sfuKeepaliveActive = false;
    this.#state.sfuKeepaliveStopReason = reason;
    await this.#saveState();

    this.#captureRest("sfu_modal_keepalive_stopped", {
      source: "do_alarm",
      reason,
      modalSessionId: this.#state.modalSessionId ?? null,
      startedAt: this.#state.sfuKeepaliveStartedAt ?? null,
      lastHeartbeatAt: this.#state.sfuKeepaliveLastHeartbeatAt ?? null,
      failureCount: this.#state.sfuKeepaliveFailures ?? 0,
    });
  }

  // ── Phase D1: SFU cleanup helper ──────────────────────────────────────

  // ── SFU capacity management ───────────────────────────────────────────
  // Single-GPU capacity guard: only one webrtc_sfu publisher at a time.
  // Uses the SfuCapacityManager singleton DO for cross-session coordination.
  //
  // FAIL-CLOSED policy: if the capacity manager DO is unreachable (network
  // partition, DO outage, binding misconfiguration), we reject the SFU
  // start with `sfu_capacity_unavailable` (503). Allowing overlap to
  // work around an infrastructure failure would reintroduce the CUDA
  // illegal-memory-access class of failures we built this guard to
  // prevent. The stale-lease TTL inside SfuCapacityManager handles the
  // case where a Worker crashes after acquire and cannot release.

  async #acquireSfuCapacity(): Promise<{ ok: boolean; conflict?: string; unavailable?: boolean }> {
    try {
      const mgrId = this.env.SFU_CAPACITY_MANAGER.idFromName("global");
      const stub = this.env.SFU_CAPACITY_MANAGER.get(mgrId);
      const resp = await stub.fetch(new Request("https://internal/sfu-capacity?op=acquire", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: this.#sessionId }),
      }));
      const data = await resp.json() as { ok: boolean; conflict?: string; error?: string };
      if (data.ok) {
        // Record ownership so cleanup paths know they owe a release.
        // Persist immediately — REST end calls #loadState() which would
        // otherwise lose this flag if the start path hasn't saved yet.
        this.#state.sfuCapacityAcquired = true;
        await this.#saveState();
      }
      return data;
    } catch (err) {
      // FAIL CLOSED — never silently allow overlap.
      console.error("[AvatarSessionDO] sfu_capacity_manager_unavailable", {
        sessionId: this.#sessionId,
        error: err instanceof Error ? err.message : String(err),
      });
      return { ok: false, unavailable: true };
    }
  }

  async #releaseSfuCapacity(force = false): Promise<void> {
    // Idempotent: only release if we previously acquired. TURN-fail and
    // capacity-fail paths never set sfuCapacityAcquired, so they can call
    // this safely without owing a release.
    //
    // `force=true` is used by end paths that have already called
    // emptyState() (clearing sfuCapacityAcquired). Callers capture the
    // flag locally first, then pass force=true to bypass the check.
    if (!force) {
      if (!this.#state.sfuCapacityAcquired) return;
      this.#state.sfuCapacityAcquired = false;
    }
    try {
      const mgrId = this.env.SFU_CAPACITY_MANAGER.idFromName("global");
      const stub = this.env.SFU_CAPACITY_MANAGER.get(mgrId);
      await stub.fetch(new Request("https://internal/sfu-capacity?op=release", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId: this.#sessionId }),
      }));
    } catch {
      // best-effort — stale-lease TTL inside the manager will reap this
      // slot if the release never lands.
    }
  }

  // Closes publisher/subscriber tracks on the SFU and clears local SFU
  // state. Does NOT touch the capacity slot. Capacity release must be
  // called separately AFTER Modal /end has completed (or best-effort
  // failed), so the GPU slot stays reserved while the publisher process
  // is still tearing down.
  //
  // Captures all needed values into locals up front because callers may
  // invoke emptyState() between this and the capacity release.
  async #cleanupSfuTracksAndState(): Promise<void> {
    // Stop keepalive FIRST, before any early return — even if sfuSessionId
    // is already null, the alarm may still be armed from a prior session.
    await this.#stopSfuKeepalive("sfu_cleanup");

    const sfuSessionId = this.#state.sfuSessionId;
    if (!sfuSessionId) return;

    const publisherMids = this.#state.sfuPublisherMids ?? { video: null, audio: null };
    const subscriberMids = this.#state.sfuSubscriberMids ?? [];

    // Close publisher tracks
    try {
      await cleanupSfuTracks(this.env, sfuSessionId, publisherMids, []);
    } catch {
      // best-effort
    }

    // Close subscriber tracks (separate SFU session)
    const subscriberSfuSessionId = this.#state.sfuSubscriberSessionId as string | undefined;
    if (subscriberSfuSessionId && subscriberMids.length > 0) {
      try {
        await cleanupSfuTracks(this.env, subscriberSfuSessionId, { video: null, audio: null }, subscriberMids);
      } catch {
        // best-effort
      }
    }

    this.#state.sfuSessionId = null;
    this.#state.sfuPublisherMids = { video: null, audio: null };
    this.#state.sfuSubscriberMids = [];
    this.#state.sfuPublisherTrackName = null;
    this.#state.sfuSubscriberSessionId = null;
    this.#state.sfuKeepaliveActive = false;
    this.#state.sfuKeepaliveStopReason = "sfu_cleanup";
    // Persist cleared state so cancellation paths (which return immediately
    // after calling this helper) do not leave stale sfuSessionId / mids in
    // DO storage. Without this save, a subsequent request that calls
    // #loadState() would reload the stale pre-save SFU fields.
    await this.#saveState();
  }
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// ── Phase 4D: WAV decode + linear resample helpers for pushModalAudio ──
//
// These helpers are inlined rather than reused from tts-provider.ts
// because:
//   1. decodeWavToPcm in tts-provider.ts is private and does not
//      expose sampleRate/channels — refactoring it would expand the
//      surface area of tts-provider.ts unnecessarily.
//   2. The existing resampleTo16000 returns ArrayBuffer (not Float32Array)
//      and is hard-coded to target 16000Hz. We need a more general
//      Float32Array-in/Float32Array-out resampler for the linear
//      prototype path.
//   3. Inline keeps tts-provider.ts untouched (lower risk for this
//      local-only fix).
//
// If reuse is preferred in a future phase, the cleanest path is to
// export decodeWavToPcm + extractWavToFloat32Mono + resampleTo16000
// from tts-provider.ts and remove these inline versions.

interface WavDecoded {
  pcm: Int16Array;
  sampleRate: number;
  channels: number;
}

function decodeWavPcm16Inline(wavBytes: Uint8Array): WavDecoded {
  // Minimal WAV decoder — returns {pcm, sampleRate, channels} by
  // reading the fmt + data chunks. Source of truth for the actual
  // sample rate of the encoded audio (TTS providers may report a
  // different sampleRate in result.sampleRate than what the WAV
  // header contains).
  const empty: WavDecoded = { pcm: new Int16Array(0), sampleRate: 0, channels: 0 };
  if (wavBytes.length < 44 ||
      String.fromCharCode(wavBytes[0], wavBytes[1], wavBytes[2], wavBytes[3]) !== "RIFF") {
    return empty;
  }
  const view = new DataView(wavBytes.buffer, wavBytes.byteOffset, wavBytes.byteLength);
  let offset = 12;
  let sampleRate = 0;
  let channels = 0;
  let bitsPerSample = 0;
  let dataOffset = -1;
  let dataSize = 0;
  while (offset + 8 <= wavBytes.length) {
    const chunkId = String.fromCharCode(
      wavBytes[offset], wavBytes[offset + 1], wavBytes[offset + 2], wavBytes[offset + 3],
    );
    const chunkSize = view.getUint32(offset + 4, true);
    const chunkDataOffset = offset + 8;
    if (chunkId === "fmt " && chunkDataOffset + 16 <= wavBytes.length) {
      channels = view.getUint16(chunkDataOffset + 2, true);
      sampleRate = view.getUint32(chunkDataOffset + 4, true);
      bitsPerSample = view.getUint16(chunkDataOffset + 14, true);
    } else if (chunkId === "data") {
      dataOffset = chunkDataOffset;
      dataSize = Math.min(chunkSize, wavBytes.length - chunkDataOffset);
      break;
    }
    offset = chunkDataOffset + chunkSize + (chunkSize % 2);
  }
  if (channels === 0 || bitsPerSample !== 16 || dataOffset < 0 || dataSize <= 0) {
    return empty;
  }
  return {
    pcm: new Int16Array(wavBytes.buffer, wavBytes.byteOffset + dataOffset, dataSize / 2),
    sampleRate,
    channels,
  };
}

function resampleLinear(
  samples: Float32Array,
  fromRate: number,
  toRate: number,
): Float32Array {
  // PROTOTYPE-QUALITY linear interpolation resampler. No anti-aliasing
  // filter. For TTS audio (8-24kHz voice bandwidth -> 16kHz) the
  // artifacts are minor and acceptable for an avatar. If production
  // quality is needed, swap in OfflineAudioContext.decodeAudioData()
  // (not suitable for Cloudflare Workers due to the Web Audio API
  // surface that Workers don't expose reliably).
  if (fromRate === toRate || samples.length === 0) return samples;
  const ratio = fromRate / toRate;
  const outLen = Math.floor(samples.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const srcIdx = i * ratio;
    const i0 = Math.floor(srcIdx);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = srcIdx - i0;
    out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
  }
  return out;
}

function float32ToBase64(f32: Float32Array): string {
  const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
  let binary = "";
  for (let i = 0; i < u8.length; i++) {
    binary += String.fromCharCode(u8[i]);
  }
  return btoa(binary);
}
