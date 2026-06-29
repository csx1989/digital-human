export interface Env {
  AVATAR_SESSION: DurableObjectNamespace;
  SFU_CAPACITY_MANAGER: DurableObjectNamespace;
  CATALOG_DB?: D1Database;
  MODAL_LIFECYCLE_URL: string;
  MODAL_TIMELINE_CONTROL_URL?: string;
  VAD_SERVICE: unknown;
  SMART_TURN_SERVICE: unknown;
  AI: Ai;
  POSTHOG_API_KEY?: string;
  POSTHOG_HOST: string;
  DEBUG_POSTHOG_TOKEN?: string;
  SARVAM_API_KEY?: string;
  STT_LANGUAGE_CODE?: string;
  STT_MODEL?: string;
  LLM_API_KEY?: string;
  LLM_BASE_URL?: string;
  LLM_MODEL?: string;
  ENABLE_LIVE_IDLE?: string;
  ENABLE_SPEECH_STREAMING?: string;
  ENABLE_AVATAR_TIMELINE?: string;
  ENABLE_MODAL_PERSISTENT_TIMELINE?: string;
  ENABLE_MODAL_FMP4_VIDEO_STREAM?: string;
  // Controls WS-path behavior (stop endpoint routing, render mode, speech pipeline).
  // The 4C REST path ignores this flag — it gates on URL presence instead
  // (MODAL_SESSION_STREAM_FMP4_AV_URL). Safe to leave "false" for REST-only usage.
  ENABLE_MODAL_FMP4_AV_STREAM?: string;
  ENABLE_OMNIVOICE_TTS_ON_L4?: string;
  OMNIVOICE_TTS_URL?: string;
  ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS?: string;
  MODAL_PARALLEL_LIFECYCLE_URL?: string;
  MODAL_PARALLEL_A10G_LIFECYCLE_URL?: string;
  MODAL_PARALLEL_L40S_LIFECYCLE_URL?: string;
  OMNIVOICE_SAME_GPU_PARALLEL_TARGET?: string;
  OMNIVOICE_SAME_GPU_PARALLEL_TIMEOUT_MS?: string;
  ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY?: string;
  AVTR_API_KEYS?: string;
  AVTR_API_KEY_METADATA_JSON?: string;
  AVTR_CATALOG_JSON?: string;
  AVTR_SESSION_TOKEN_SECRET?: string;
  AVTR_TEST_HOOK_SECRET?: string;
  RL_HEAVY?: RateLimit;
  RL_WRITE?: RateLimit;
  RL_LIGHT?: RateLimit;
  // Phase 4C: One URL per Modal @modal.fastapi_endpoint() method on the
  // session class. Each method is auto-routed to its own Modal URL by
  // Modal (label-based: <workspace>--<label>.modal.run). When the SSE
  // stream URL is set, the Bridge uses the 4C path; otherwise it falls
  // back to legacy per-frame protocol (gated by ENABLE_LEGACY_MODAL_PROTOCOL).
  MODAL_SESSION_START_URL?: string;
  MODAL_SESSION_PUSH_AUDIO_URL?: string;
  MODAL_SESSION_PUSH_AUDIO_BATCH_URL?: string;
  MODAL_SESSION_STATUS_URL?: string;
  MODAL_SESSION_END_URL?: string;
  MODAL_SESSION_STREAM_FMP4_AV_URL?: string;
  // Phase 4B: Default "false". Set "true" to keep old per-frame protocol
  // (render_frame, render_audio, timeline_stream, etc.). The new 4A
  // contract does not support per-frame or SSE-streaming render methods,
  // so legacy mode is fundamentally degraded for production.
  ENABLE_LEGACY_MODAL_PROTOCOL?: string;
  FMP4_RELAY_PACE_MS?: string;
  FMP4_RELAY_MAX_QUEUE?: string;
  // ── Phase D1: Cloudflare Realtime SFU ──────────────────────────────
  // Default "false". When "true", session start attempts SFU transport
  // (gated by transport preference). fMP4 remains as fallback.
  AVTR_REALTIME_SFU_ENABLED?: string;
  // Safety guard: when "websocket_fmp4", auto/omitted transport never attempts
  // SFU even if fully configured. Only explicit transport:"webrtc_sfu" does.
  AVTR_REALTIME_SFU_DEFAULT?: string;
  // Phase 3F: private SFU beta gate. Short-lived HMAC-signed tokens are
  // verified server-side; only a boolean authorization flag is persisted.
  AVTR_SFU_BETA_TOKEN_SECRET?: string;
  // Comma-separated tenant ids that are permanently SFU-beta-authorized.
  AVTR_SFU_BETA_TENANTS?: string;
  // Cloudflare Realtime SFU credentials (Worker/DO only — never exposed).
  CALLS_APP_ID?: string;
  CALLS_APP_SECRET?: string;
  // ── TURN relay (production-grade ICE reliability) ───────────────────
  // Cloudflare TURN Key ID + API Token for generating short-lived TURN
  // credentials. Worker generates per-session; Modal receives only
  // short-lived credentials (never the API token).
  AVTR_SFU_TURN_KEY_ID?: string;
  AVTR_SFU_TURN_KEY_API_TOKEN?: string;
  // When "true" (production default), TURN is required for SFU mode.
  // Missing TURN secrets → explicit sfu_turn_not_configured error.
  // Set "false" for local/dev diagnostics (allows direct-ICE-only).
  AVTR_SFU_REQUIRE_TURN?: string;
  // Internal Modal→Worker callback auth. Set as a Modal secret with the
  // same value. Never sent in request bodies.
  MODAL_WORKER_SECRET?: string;
  // Internal Worker→Modal request signing secret. Set as a Cloudflare secret
  // and a Modal secret with the same value. Never sent in request bodies.
  AVTR_MODAL_REQUEST_SIGNING_SECRET?: string;
  // Base URL of this Worker, used to build callback URLs for Modal.
  // e.g. "https://avatar-session-bridge.vallangirakesh.workers.dev"
  WORKER_BASE_URL?: string;
  // ── Stage 3B: SFU cost-safety hard limits ──────────────────────────
  // Hard maximum lifetime of the DO-alarm Modal keepalive loop. After this
  // age the alarm aborts unconditionally, even if Modal status calls succeed.
  // Default 120000 ms (2 minutes) for beta/incident response.
  AVTR_SFU_KEEPALIVE_MAX_AGE_MS?: string;
  // Hard maximum lifetime of an SFU session independent of keepalive state.
  // Default 120000 ms (2 minutes) for beta/incident response.
  AVTR_SFU_SESSION_MAX_AGE_MS?: string;
  // Consecutive Modal status failures before the keepalive aborts.
  // Default 2 for beta/incident response.
  AVTR_SFU_KEEPALIVE_MAX_FAILURES?: string;
  // Admin force-clear endpoint secret. NOT DEBUG_POSTHOG_TOKEN.
  // Used by POST /debug/sessions/:id/force-clear to clear a runaway session.
  AVTR_ADMIN_FORCE_CLEAR_TOKEN?: string;
}

export interface VadServiceRpc {
  processAudio(sessionName: string, chunk: ArrayBuffer): Promise<VadDecision>;
  reset(sessionName: string): Promise<void>;
}

export interface SmartTurnServiceRpc {
  accumulateAudio(
    sessionId: string,
    chunk: ArrayBuffer,
    options?: { sampleRate?: number; encoding?: "pcm16" },
  ): Promise<void>;
  analyzeTurn(
    sessionId: string,
    options?: { threshold?: number },
  ): Promise<SmartTurnDecision>;
  reset(sessionId: string): Promise<void>;
}

export interface VadDecision {
  shouldStart: boolean;
  shouldEnd: boolean;
  inSpeech: boolean;
  probability: number;
}

export interface SmartTurnDecision {
  prediction: 0 | 1;
  probability: number;
  rawLogit: number;
  bufferSamples?: number;
  bufferApproxSeconds?: number;
  processedSamples?: number;
  processedApproxSeconds?: number;
  trimmedToMaxSeconds?: boolean;
}

export type ClientMessage =
  | { type: "session.start"; avatarId?: string; bgId?: string; transport?: "auto" | "webrtc_sfu" | "websocket_fmp4" }
  | { type: "render_debug_frame" }
  | { type: "session.end"; reason?: string }
  | { type: "mic.start" }
  | { type: "mic.stop" }
  | { type: "avatar.speech.playback.ended"; turnId: number }
  | {
      type: "client.telemetry";
      event: string;
      elapsedMs?: number;
      properties?: Record<string, unknown>;
    };

export type ServerMessage =
  | {
      type: "session.ready";
      sessionId: string;
      avatarId: string;
      bgId: string;
      width: number;
      height: number;
      fps: number;
      framesPerChunk: number;
      warmupMs: number;
      renderMode?: "fmp4_av" | "fmp4_video" | "jpeg_timeline";
      transport?: "websocket_fmp4" | "webrtc_sfu";
      sfu?: {
        publisher_track_name: string;
        subscribe_url: string;
        answer_url: string;
        close_url: string;
        token: string;
        token_expires_at: number;
      };
    }
  | {
      type: "avatar.frame";
      dataBase64: string;
      mimeType: string;
      width: number;
      height: number;
    }
  | {
      type: "avatar.speech.start";
      turnId: number;
      totalFrames: number;
      fps: number;
      renderedAudioDurationMs?: number;
      originalAudioDurationMs?: number;
    }
  | {
      type: "avatar.sfu.speech_queued";
      turnId: number;
      chunks: number;
      samples: number;
    }
  | {
      type: "avatar.speech.chunk";
      turnId: number;
      chunkIndex: number;
      frames: Array<{
        dataBase64: string;
        mimeType: string;
        width: number;
        height: number;
      }>;
      isLast: boolean;
    }
  | {
      type: "avatar.idle.start";
      totalFrames: number;
      fps: number;
      durationMs?: number;
    }
  | {
      type: "avatar.idle.chunk";
      chunkIndex: number;
      frames: Array<{
        dataBase64: string;
        mimeType: string;
        width: number;
        height: number;
      }>;
      isLast: boolean;
      live?: boolean;
      durationMs?: number;
      renderDurationMs?: number;
    }
  | {
      type: "session.ended";
      sessionId: string | null;
      wasActive: boolean;
      modalEndOk?: boolean;
      modalEndDurationMs?: number;
      modalEndError?: string;
    }
  | { type: "status"; status: string }
  | { type: "error"; code: string; message: string; fatal?: boolean }
  | { type: "speech.started"; turnId: number }
  | {
      type: "speech.segment.started";
      turnId: number;
      segmentIndex: number;
    }
  | {
      type: "speech.finalized";
      turnId: number;
      smartTurn: SmartTurnDecision;
      smartTurnFallback?: boolean;
    }
  | { type: "speech.continues"; turnId: number; smartTurn: SmartTurnDecision }
  | {
      type: "transcript.final";
      turnId: number;
      text: string;
      languageCode?: string;
      provider: string;
      latencyMs: number;
    }
  | {
      type: "assistant.text";
      turnId: number;
      text: string;
      provider: string;
      latencyMs: number;
    }
  | {
      type: "assistant.audio";
      turnId: number;
      dataBase64?: string;
      mimeType: "audio/wav";
      provider: string;
      latencyMs: number;
      sampleRate: 16000;
      audioBytes: number;
      directEnqueued?: boolean;
      audioDurationMs?: number;
    }
  | { type: "audio.stats"; chunksReceived: number; bytesReceived: number }
  | { type: "avatar.timeline.started"; fps: number; mode: "fmp4_av" | "fmp4_video" | "jpeg_timeline" }
  | { type: "avatar.timeline.stopped"; reason: string }
  | {
      type: "avatar.fmp4.init";
      dataBase64: string;
      byteLength: number;
      timingMs: number;
    }
  | {
      type: "avatar.fmp4.segment";
      dataBase64: string;
      byteLength: number;
      sequence: number;
      mode: "idle" | "speech";
      isSpeechStart: boolean;
      isSpeechEnd: boolean;
      turnId: number | null;
      segmentId: number;
      chunkIndex: number;
      chunkRenderMs: number;
      feedMs: number;
      framesFed: number;
      audioFeedMs?: number;
      audioSamplesPerChunk?: number;
      containerInstanceId?: string;
      overlapWithTts?: boolean;
    };

export interface SessionState {
  modalSessionId: string | null;
  active: boolean;
  avatarId: string | null;
  bgId: string | null;
  createdAt?: number;
  startedAt?: number;
  endedAt?: number;
  starting?: boolean;
  startingAt?: number;
  startAttemptId?: string | null;
  startCancelRequested?: boolean;
  // ── Phase 3F: SFU private beta authorization ───────────────────
  // Set at session creation; never includes the raw beta token.
  sfuBetaAuthorized?: boolean;
  // ── Phase D1: SFU transport state ──────────────────────────────
  // All fields are optional so existing persisted sessions load safely.
  transport?: "websocket_fmp4" | "webrtc_sfu";
  sfuSessionId?: string | null;
  sfuPublisherMids?: { video: string | null; audio: string | null };
  sfuSubscriberMids?: string[];
  sfuPublisherTrackName?: string | null;
  sfuSubscriberSessionId?: string | null;
  // ── Stage 3B: SFU Modal keepalive (DO alarm) ───────────────────
  sfuKeepaliveActive?: boolean;
  sfuKeepaliveStartedAt?: number | null;
  sfuKeepaliveLastHeartbeatAt?: number | null;
  sfuKeepaliveFailures?: number;
  sfuKeepaliveStopReason?: string | null;
  sfuKeepaliveAborted?: boolean;
  // Timestamp when the SFU transport became active (session_ready). Used for
  // the independent session max-age cost guard.
  sfuSessionStartedAt?: number | null;
  // ── Single-GPU capacity ownership flag ────────────────────────
  // True iff this session successfully acquired the global SFU capacity
  // slot from SfuCapacityManager. Cleared after release. Lets cleanup
  // paths know whether they owe a release call (TURN/capacity-fail paths
  // never acquired and so must NOT release).
  sfuCapacityAcquired?: boolean;
}

export interface SessionOwner {
  tenant_id: string;
  auth_mode: string;
  key_name: string | null;
  plan: string | null;
  created_at: number;
  // Phase 3F: SFU private beta authorization flag (boolean only).
  sfu_beta_authorized?: boolean;
}

export interface WsAttachment {
  connectedAt: number;
  sessionId: string;
}

export interface ModalActionRequest {
  action: "health" | "start" | "render_frame" | "render_audio" | "render_idle" | "render_idle_chunk" | "render_audio_stream" | "end" | "timeline_stream" | "timeline_stream_fmp4_video" | "timeline_stream_fmp4_av" | "timeline_enqueue" | "timeline_stop";
  [key: string]: unknown;
}

export interface ModalStartResponse {
  sessionId: string;
  status: string;
  avatarId: string;
  bgId: string;
  width: number;
  height: number;
  fps: number;
  framesPerChunk: number;
  warmupMs: number;
}

export interface ModalRenderResponse {
  available: boolean;
  dataBase64?: string;
  mimeType?: string;
  width?: number;
  height?: number;
  note?: string;
}

export interface ModalEndResponse {
  ok: boolean;
  sessionId: string | null;
  wasActive: boolean;
}

export interface ModalErrorResponse {
  status?: string;
  error?: string;
  note?: string;
}

export interface ModalRenderAudioFrame {
  dataBase64: string;
  mimeType: string;
  width: number;
  height: number;
}

export interface ModalRenderAudioResponse {
  available: boolean;
  frames?: ModalRenderAudioFrame[];
  audioDurationMs?: number;
  framesRendered?: number;
  fps?: number;
  renderDurationMs?: number;
  modelRenderDurationMs?: number;
  jpegEncodeDurationMs?: number;
  base64EncodeDurationMs?: number;
  frameConvertDurationMs?: number;
  totalHandlerDurationMs?: number;
  overheadMs?: number;
  avgJpegEncodeMsPerFrame?: number;
  encodedBytes?: number;
  base64Chars?: number;
  frameWidth?: number;
  frameHeight?: number;
  truncated?: boolean;
  maxDurationS?: number;
  originalDurationMs?: number;
  error?: string;
}

export interface ModalRenderIdleResponse {
  available: boolean;
  frames?: ModalRenderAudioFrame[];
  durationMs?: number;
  framesRendered?: number;
  fps?: number;
  renderDurationMs?: number;
  modelRenderDurationMs?: number;
  jpegEncodeDurationMs?: number;
  base64EncodeDurationMs?: number;
  frameConvertDurationMs?: number;
  totalHandlerDurationMs?: number;
  overheadMs?: number;
  avgJpegEncodeMsPerFrame?: number;
  encodedBytes?: number;
  base64Chars?: number;
  frameWidth?: number;
  frameHeight?: number;
  error?: string;
}

export type ModalRenderIdleChunkResponse = ModalRenderIdleResponse;

export interface ModalStreamStartEvent {
  type: "start";
  mode?: "idle" | "speech";
  sessionId: string;
  fps: number;
  originalAudioDurationMs: number;
  renderedAudioDurationMs: number;
  totalSteps: number;
}

export interface ModalStreamChunkEvent {
  type: "chunk";
  mode?: "idle" | "speech";
  chunkIndex: number;
  frames: ModalRenderAudioFrame[];
  framesRendered: number;
  isLast: boolean;
  timing: {
    chunkRenderMs: number;
    chunkEncodeMs: number;
    chunkBase64Chars: number;
  };
}

export interface ModalStreamEndEvent {
  type: "end";
  mode?: "idle" | "speech";
  framesRendered: number;
  totalHandlerDurationMs: number;
  modelRenderDurationMs: number;
  totalEncodeMs: number;
  truncated: boolean;
  originalDurationMs: number;
  renderedAudioDurationMs: number;
  firstChunkReadyMs: number;
}

export type ModalStreamEvent = ModalStreamStartEvent | ModalStreamChunkEvent | ModalStreamEndEvent;

export interface ModalStreamCallbacks {
  onStart: (event: ModalStreamStartEvent) => void;
  onChunk: (event: ModalStreamChunkEvent) => void;
  onEnd: (event: ModalStreamEndEvent) => void;
}

export interface ModalTimelineFrameEvent {
  type: "frame_chunk";
  mode: "idle" | "speech";
  turnId: number | null;
  segmentId: number;
  chunkIndex: number;
  frames: ModalRenderAudioFrame[];
  fps: number;
  isSpeechStart: boolean;
  isSpeechEnd: boolean;
  enqueueToFirstFrameMs?: number;
  timing: {
    chunkRenderMs: number;
    chunkEncodeMs: number;
    base64Chars: number;
    queueDepth: number;
    speechRemainingMs: number;
  };
}

export interface ModalTimelineStartEvent {
  type: "start";
  mode: "idle";
  sessionId: string;
  fps: number;
}

export interface ModalTimelineEndEvent {
  type: "end";
  mode: "idle";
  chunksRendered: number;
  mediaSegments?: number;
  audioBytes?: number;
  videoBytes?: number;
  encodedBytes?: number;
  stopReason: string;
  ffmpegReturnCode?: number | null;
  ffmpegStderrErrorCount?: number;
  gpu?: {
    smiUsedMinMb?: number | null;
    smiUsedMaxMb?: number | null;
    smiTotalMb?: number | null;
    torchAllocMinMb?: number | null;
    torchAllocMaxMb?: number | null;
    torchPeakMb?: number | null;
    smiGpuUtilMinPct?: number | null;
    smiGpuUtilMaxPct?: number | null;
    smiGpuUtilAvgPct?: number | null;
    smiMemoryUtilMinPct?: number | null;
    smiMemoryUtilMaxPct?: number | null;
    smiMemoryUtilAvgPct?: number | null;
    snapshotCount?: number;
    snapshotLabels?: string[];
  };
  cadenceStats?: {
    targetLeadS?: number;
    chunkMediaS?: number;
    maxModalInterYieldGapMs?: number;
    avgModalInterYieldGapMs?: number;
    totalPacingSleepMs?: number;
    modalSegmentsYielded?: number;
    selectedFfmpegVariant?: string;
    senderMinQueueDepth?: number;
    senderMaxQueueDepth?: number;
    senderDrySpellCount?: number;
  };
}

export type ModalTimelineEvent =
  | ModalTimelineStartEvent
  | ModalTimelineFrameEvent
  | ModalTimelineEndEvent
  | { type: "error"; error: string };

export interface ModalTimelineCallbacks {
  onStart: (event: ModalTimelineStartEvent) => void;
  onFrame: (event: ModalTimelineFrameEvent) => void;
  onEnd: (event: ModalTimelineEndEvent) => void;
}

export interface ModalTimelineEnqueueResponse {
  ok: boolean;
  turnId?: number;
  queueDepth?: number;
  audioDurationMs?: number;
  error?: string;
}

export interface ModalTimelineStopResponse {
  ok: boolean;
  wasRunning: boolean;
}

export interface ModalTimelineFmp4InitEvent {
  type: "fmp4_init";
  dataBase64: string;
  byteLength: number;
  timing: { timeToInitMs: number };
}

export interface ModalTimelineFmp4MediaEvent {
  type: "fmp4_media";
  dataBase64: string;
  byteLength: number;
  sequence: number;
  mode: "idle" | "speech";
  isSpeechStart?: boolean;
  isSpeechEnd?: boolean;
  turnId?: number | null;
  segmentId?: number;
  chunkIndex?: number;
  timing?: {
    chunkRenderMs: number;
    feedMs: number;
    framesFed: number;
    enqueueToFirstFrameMs?: number;
  };
}

export interface ModalTimelineFmp4Callbacks {
  onInit: (event: ModalTimelineFmp4InitEvent) => void;
  onMedia: (event: ModalTimelineFmp4MediaEvent) => void;
  onEnd: (event: ModalTimelineEndEvent) => void;
}

export interface ModalTimelineFmp4AvMediaEvent {
  type: "fmp4_media";
  dataBase64: string;
  byteLength: number;
  sequence: number;
  mode: "idle" | "speech";
  isSpeechStart?: boolean;
  isSpeechEnd?: boolean;
  turnId?: number | null;
  segmentId?: number;
  chunkIndex?: number;
  timing?: {
    chunkRenderMs: number;
    feedMs: number;
    framesFed: number;
    enqueueToFirstFrameMs?: number;
    audioFeedMs?: number;
    audioSamplesPerChunk?: number;
    speechEndToIdleResumeMs?: number;
    overlapWithTts?: boolean;
    pacingSleepMs?: number;
    mediaLeadMs?: number;
    producedMediaMs?: number;
    wallElapsedMs?: number;
    targetLeadMs?: number;
    pacingMode?: string;
  };
  gpu?: {
    smiMemoryUsedMb?: number;
    smiMemoryTotalMb?: number;
    torchAllocatedMb?: number;
    torchPeakMb?: number;
  };
  containerInstanceId?: string;
}

export interface ModalTimelineFmp4AvCallbacks {
  onInit: (event: ModalTimelineFmp4InitEvent) => void;
  onMedia: (event: ModalTimelineFmp4AvMediaEvent) => void;
  onEnd: (event: ModalTimelineEndEvent) => void;
}

// ── Phase 4B: New Modal session-lifecycle contract (Phase 4A baseline) ──
// See PHASE_4A_SESSION_BASELINE_REPORT.md for full contract.

export interface ModalSessionStartRequest {
  session_id: string;
  avatar_id: string;
  bg_id?: string;
  config?: Record<string, unknown> | null;
  // ── Phase D1: SFU publisher mode (optional) ──
  publisher_mode?: "webrtc_sfu";
  sfu_session_id?: string;
  publisher_callback_url?: string;
  close_callback_url?: string;
  // ── TURN relay: ICE servers with short-lived credentials ──
  // Generated by the Worker per session; never persisted in DO state.
  // Format: [{urls: ["turn:...", "turns:..."], username: "...", credential: "..."}]
  ice_servers?: unknown[];
}

export interface ModalSessionStartResponse {
  session_id: string;
  avatar_id: string;
  status: "ready";
  cache_hit: boolean;
  init_s: number;
  active_sessions: string[];
  state_note: string;
  vram_mb: number;
  bg_id?: string;
  requested_bg_id?: string | null;
  loaded_avatar_ids?: string[];
  active_avatar_ids?: string[];
  // ── Phase D1: SFU publisher mode response ──
  publisher_track_name?: string;
  publisher_mode?: string;
}

export interface ModalSessionEndResponse {
  session_id: string;
  avatar_id: string;
  status: "ended";
  chunks_rendered: number;
  fmp4_media_count: number;
  fmp4_media_total_bytes: number;
  duration_s: number;
  ffmpeg_rc: number | null;
  zombie_check: "clean" | "zombie_still_running" | "unknown" | "n/a";
  handoff_closed: boolean;
  threads_joined: boolean;
  active_sessions: string[];
  vram_mb: number;
}

export interface ModalSessionStatusResponse {
  session_id: string;
  avatar_id: string;
  status: "ready" | "running" | "completed" | "ended";
  chunk_count: number;
  fmp4_media_count: number;
  fmp4_media_total_bytes: number;
  fmp4_init_segment: boolean;
  duration_s: number;
  render_errors: string[];
  active_sessions: string[];
  vram_mb: number;
}

export interface ModalSessionNotActiveResponse {
  session_id: string;
  status: "not_active";
  active_sessions: string[];
}

export interface ModalPushAudioRequest {
  session_id: string;
  audio_data_b64?: string | null;
  audio_mode?: "sine" | "silence" | "pcm";
  // Speech metadata propagated from Bridge to Modal for SSE stamping
  turn_id?: number | null;
  chunk_index?: number | null;
  is_first?: boolean | null;
  is_last?: boolean | null;
  total_chunks?: number | null;
  generated_at?: number | null;
  enqueue_at?: number | null;
}

export interface ModalPushAudioResponse {
  session_id: string;
  pushed: boolean;
  samples: number;
  queue_depth: number;
  session_status: string;
}

// ── Public v1 API canonical types (docs/API_ARCHITECTURE.md) ──────────────
// These describe the external `/v1/*` contract. Internal DO/Modal shapes are
// normalized into these at the Worker boundary (src/api-v1.ts). External
// callers must never see Modal session ids or other provider internals.

export interface ApiTimings {
  latency_ms: number;
  request_id?: string;
}

export interface ApiSuccess<T> {
  success: true;
  data: T;
  timings: ApiTimings;
}

export interface ApiErrorBody {
  code: string;
  message: string;
}

export interface ApiFailure {
  success: false;
  error: ApiErrorBody;
  timings: ApiTimings;
}

export type ApiResponse<T> = ApiSuccess<T> | ApiFailure;

export type ApiTransport = "websocket_fmp4" | "webrtc_sfu";

export interface V1StreamSfu {
  subscribe_url: string;
  answer_url: string;
  close_url: string;
  token: string;
  token_expires_at: number;
  track_names: string[];
}

export interface V1Stream {
  transport: ApiTransport;
  ws_url: string;
  status_url: string;
  token_expires_at?: number;
  sfu?: V1StreamSfu;
  available_transports?: string[];
}

export type V1SessionStatus = "created" | "ready" | "active" | "ended";

export interface V1SessionEndSummary {
  chunks_rendered?: number;
  fmp4_media_count?: number;
  fmp4_media_total_bytes?: number;
  duration_s?: number;
  ffmpeg_rc?: number | null;
}

export interface V1Session {
  session_id: string;
  avatar_id: string | null;
  bg_id: string | null;
  status: V1SessionStatus;
  created_at: number | null;
  started_at?: number | null;
  ended_at?: number | null;
  stream: V1Stream;
  end_ok?: boolean;
  end_summary?: V1SessionEndSummary;
}

export interface V1Avatar {
  avatar_id: string;
  display_name: string;
  available: boolean;
  default_bg_id: string;
  preview_url?: string;
}

export interface V1Background {
  bg_id: string;
  display_name: string;
  available: boolean;
  preview_url?: string;
}

export interface V1Health {
  status: "ok";
  version: string;
  transport: ApiTransport;
}

export interface CreateSessionRequest {
  avatar_id?: string;
  bg_id?: string;
  transport?: "auto" | "webrtc_sfu" | "websocket_fmp4";
}

export interface StartSessionRequest {
  avatar_id?: string;
  bg_id?: string;
  transport?: "auto" | "webrtc_sfu" | "websocket_fmp4";
}

export interface V1Providers {
  stt: {
    provider: string;
    model: string | null;
    language: string | null;
  };
  llm: {
    provider: string;
    model: string | null;
  };
  tts: {
    provider_chain: string;
    omnivoice_enabled: boolean;
    same_gpu_parallel_enabled: boolean;
    gpu_target: string | null;
    fallback_enabled: boolean;
  };
  avatar: {
    transport: ApiTransport;
    render_backend: string;
  };
  vad: { enabled: boolean };
  smart_turn: { enabled: boolean };
}

export interface ApiAuthContext {
  tenant_id: string;
  key_name: string | null;
  plan: string | null;
  auth_mode: "metadata" | "legacy";
  // Phase 3F: per-key SFU private beta flag.
  sfu_beta?: boolean;
}

export interface V1Me {
  tenant_id: string;
  key_name: string | null;
  plan: string | null;
  auth_mode: "metadata" | "legacy";
}

export type ApiErrorCode =
  | "missing_api_key"
  | "invalid_api_key"
  | "api_auth_not_configured"
  | "invalid_session_id"
  | "method_not_allowed"
  | "invalid_body"
  | "not_found"
  | "internal_error"
  | "session_not_started"
  | "session_conflict"
  | "modal_start_failed"
  | "modal_unreachable"
  | "no_modal_protocol_enabled"
  | "ws_token_missing"
  | "ws_token_invalid"
  | "ws_token_secret_not_configured"
  | "rate_limited"
  | "session_starting"
  | "session_cancelled"
  | "avatar_not_found"
  | "background_not_found"
  | "sfu_unavailable"
  | "sfu_not_configured"
  | "sfu_beta_not_enabled"
  | "sfu_session_not_found"
  | "sfu_wrong_transport"
  | "sfu_turn_credentials_unavailable"
  | "sfu_capacity_exceeded"
  | "sfu_capacity_unavailable";

/**
 * Error codes used only by internal (non-public) callback routes.
 * These are NOT part of the public v1 API contract and must not appear
 * in docs/openapi-v1.yaml.  They authenticate via x-modal-secret, not
 * the public API key, and are only reachable from Modal→Worker callbacks.
 */
export type InternalErrorCode =
  | "internal_secret_invalid";
