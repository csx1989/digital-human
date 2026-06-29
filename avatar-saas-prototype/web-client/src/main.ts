type Status =
  | "disconnected"
  | "creating"
  | "connecting"
  | "ready"
  | "rendering"
  | "stopped"
  | "error"
  | "mic_active";

type VideoBufferInfo = {
  bufferedStart: number;
  bufferedEnd: number;
  bufferAheadMs: number;
};

const emptyVideoBufferInfo: VideoBufferInfo = {
  bufferedStart: 0,
  bufferedEnd: 0,
  bufferAheadMs: 0,
};

type LoadingStage = "connecting" | "starting" | "preparing" | "live" | "error";

const LOADING_LABELS: Record<LoadingStage, string> = {
  connecting: "Connecting\u2026",
  starting: "Starting avatar\u2026",
  preparing: "Preparing live video\u2026",
  live: "Live",
  error: "Session failed",
};

const LOADING_STATUS_MAP: Record<LoadingStage, Status> = {
  connecting: "connecting",
  starting: "creating",
  preparing: "rendering",
  live: "ready",
  error: "error",
};

let placeholderShownAt = 0;
let currentLoadingStage: LoadingStage | null = null;

const $ = <T extends HTMLElement>(id: string) =>
  document.getElementById(id) as T;

const workerUrlInput = $<HTMLInputElement>("workerUrl");
const connectBtn = $<HTMLButtonElement>("connectBtn");
const stopBtn = $<HTMLButtonElement>("stopBtn");
const startMicBtn = $<HTMLButtonElement>("startMicBtn");
const stopMicBtn = $<HTMLButtonElement>("stopMicBtn");
const statusBar = $<HTMLElement>("statusBar");
const statusText = $<HTMLElement>("statusText");
const avatarFrame = $<HTMLElement>("avatarFrame");
const frameMeta = $<HTMLElement>("frameMeta");
const logOutput = $<HTMLElement>("logOutput");
const audioStats = $<HTMLElement>("audioStats");
const speechStatus = $<HTMLElement>("speechStatus");
const transcriptList = $<HTMLElement>("transcriptList");
const rawAudioCheckbox = $<HTMLInputElement>("rawAudio");
const copyLogsBtn = $<HTMLButtonElement>("copyLogsBtn");

let ws: WebSocket | null = null;
let sessionStartTime = 0;
let audioContext: AudioContext | null = null;
let micStream: MediaStream | null = null;
let workletNode: AudioWorkletNode | null = null;
let silentGain: GainNode | null = null;
let micChunksSent = 0;
let micBytesSent = 0;
let currentSessionId = "";
let audioRmsSum = 0;
let audioRmsCount = 0;
let audioPeakMax = 0;
let audioNearZeroSum = 0;
let audioClippedSum = 0;
let playbackAudioContext: AudioContext | null = null;

let sfuPc: RTCPeerConnection | null = null;
let sfuToken: string | null = null;
let sfuSubscribeUrl = "";
let sfuAnswerUrl = "";
let sfuCloseUrl = "";
let sfuStatsInterval: ReturnType<typeof setInterval> | null = null;
let sfuActive = false;
let sfuIceState = "new";
let sfuVideoAttachedAt = 0;
let sfuFirstFrameAt = 0;
let sfuLastFramesDecoded = 0;
let sfuRenderAttachFailed = false;
let sfuVideoListenersAttached = false;
let sfuStatsLogLastAt = 0;
let sfuStatsLogLastCurrentTime = 0;
let sfuStatsLogLastFramesDecoded = 0;
let sfuStatsLogLastPacketsLost = 0;
let sfuStatsTelemetryLastAt = 0;
let sfuStatsAnomalyLastAt = 0;

// ── Non-invasive SFU diagnostics (only active when sfu_debug=1) ──
// Observes WebRTC state passively. NEVER sets video.srcObject, NEVER
// overrides pc.ontrack or any app handler. Uses addEventListener only.
interface SfuDiagSnapshot {
  t: number;
  label: string;
  [k: string]: unknown;
}
interface SfuDiagState {
  events: SfuDiagSnapshot[];
  stats: SfuDiagSnapshot[];
  pcState: Record<string, unknown>;
}
const sfuDiagState: SfuDiagState = { events: [], stats: [], pcState: {} };
let sfuDiagInterval: ReturnType<typeof setInterval> | null = null;

// ── rVFC cadence tracker ──
interface RvfcEntry {
  wallClock: number;
  mediaTime: number;
  presentedFrames: number;
  processingDuration: number | null;
  wallDeltaMs: number | null;
  mediaDeltaMs: number | null;
  presentedDelta: number | null;
  stall: boolean;
  jump: boolean;
}
const rvfcBuffer: RvfcEntry[] = [];
const RVFC_MAX = 50;
let rvfcRafHandle: number | null = null;
const RVFC_EXPECTED_MS = 1000 / 25; // 40ms at 25fps
const RVFC_STALL_THRESHOLD = 60; // 1.5x expected
const RVFC_MEDIA_MIN = 0.032; // 0.8 * frame_duration
const RVFC_MEDIA_MAX = 0.048; // 1.2 * frame_duration

// ── Luma sampler ──
interface LumaEntry {
  t: number;
  yMean: number;
  mad: number;
}
const lumaBuffer: LumaEntry[] = [];
const LUMA_MAX = 100;
let lumaCanvas: HTMLCanvasElement | null = null;
let lumaCtx: CanvasRenderingContext2D | null = null;
let lumaPrevData: Uint8ClampedArray | null = null;

// ── Flicker marker state ──
let flickerMarkerCount = 0;
let sfuDebugStatsLastEmit = 0;

function sfuDiagLog(label: string, data: Record<string, unknown>): void {
  if (!SFU_DEBUG) return;
  const entry: SfuDiagSnapshot = { t: Date.now(), label, ...data };
  sfuDiagState.events.push(entry);
  console.log("[SFU_DIAG_EVENT]", JSON.stringify(entry));
}

/** Register PASSIVE listeners on the real PC. Does NOT override on* handlers. */
function attachSfuDiagListeners(pc: RTCPeerConnection): void {
  if (!SFU_DEBUG) return;

  pc.addEventListener("track", (e: RTCTrackEvent) => {
    sfuDiagLog("track", {
      kind: e.track?.kind,
      trackId: e.track?.id,
      trackEnabled: e.track?.enabled,
      trackReadyState: e.track?.readyState,
      trackMuted: e.track?.muted,
      streams: e.streams?.length,
      streamId: e.streams?.[0]?.id,
    });
  });

  pc.addEventListener("iceconnectionstatechange", () => {
    sfuDiagLog("iceconnectionstatechange", { state: pc.iceConnectionState });
  });

  pc.addEventListener("connectionstatechange", () => {
    sfuDiagLog("connectionstatechange", { state: pc.connectionState });
  });

  pc.addEventListener("signalingstatechange", () => {
    sfuDiagLog("signalingstatechange", { state: pc.signalingState });
  });

  pc.addEventListener("icecandidateerror", (e: RTCPeerConnectionIceErrorEvent) => {
    sfuDiagLog("icecandidateerror", {
      url: e.url,
      errorCode: e.errorCode,
      errorText: e.errorText,
    });
  });

  pc.addEventListener("icegatheringstatechange", () => {
    sfuDiagLog("icegatheringstatechange", { state: pc.iceGatheringState });
  });
}

// ── rVFC callback handler ──
function onRvfcCallback(
  now: number,
  metadata: { mediaTime: number; presentedFrames: number; processingDuration?: number; width: number; height: number },
): void {
  const prev = rvfcBuffer.length > 0 ? rvfcBuffer[rvfcBuffer.length - 1] : null;
  const wallDeltaMs = prev ? now - prev.wallClock : null;
  const mediaDeltaMs = prev ? (metadata.mediaTime - prev.mediaTime) * 1000 : null;
  const presentedDelta = prev ? metadata.presentedFrames - prev.presentedFrames : null;
  const stall = wallDeltaMs !== null && wallDeltaMs > RVFC_STALL_THRESHOLD;
  const jump = mediaDeltaMs !== null && (mediaDeltaMs < RVFC_MEDIA_MIN * 1000 || mediaDeltaMs > RVFC_MEDIA_MAX * 1000);

  const entry: RvfcEntry = {
    wallClock: now,
    mediaTime: metadata.mediaTime,
    presentedFrames: metadata.presentedFrames,
    processingDuration: metadata.processingDuration ?? null,
    wallDeltaMs,
    mediaDeltaMs,
    presentedDelta,
    stall,
    jump,
  };
  rvfcBuffer.push(entry);
  if (rvfcBuffer.length > RVFC_MAX) rvfcBuffer.shift();

  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (video && "requestVideoFrameCallback" in video) {
    rvfcRafHandle = (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: (now: number, metadata: Record<string, unknown>) => void) => number }).requestVideoFrameCallback(onRvfcCallback as Parameters<typeof video.requestVideoFrameCallback>[0]);
  }
}

function startRvfcTracking(): void {
  if (!SFU_DEBUG) return;
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (!video) return;
  if (!("requestVideoFrameCallback" in video)) {
    sfuDiagLog("rvfc_unsupported", {});
    return;
  }
  rvfcRafHandle = (video as HTMLVideoElement & { requestVideoFrameCallback: (cb: (now: number, metadata: Record<string, unknown>) => void) => number }).requestVideoFrameCallback(onRvfcCallback as Parameters<typeof video.requestVideoFrameCallback>[0]);
  sfuDiagLog("rvfc_started", {});
}

function stopRvfcTracking(): void {
  rvfcRafHandle = null;
}

// ── Luma sampler ──
function startLumaSampler(): void {
  if (!SFU_DEBUG) return;
  lumaCanvas = document.createElement("canvas");
  lumaCanvas.width = 160;
  lumaCanvas.height = 90;
  lumaCtx = lumaCanvas.getContext("2d", { willReadFrequently: true });
  if (!lumaCtx) return;
  sfuDiagLog("luma_sampler_started", {});
}

function sampleLuma(): void {
  if (!SFU_DEBUG || !lumaCtx) return;
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (!video || video.videoWidth === 0) return;
  try {
    lumaCtx.drawImage(video, 0, 0, 160, 90);
    const img = lumaCtx.getImageData(0, 0, 160, 90);
    const data = img.data;
    let sum = 0;
    const gray = new Uint8ClampedArray(160 * 90);
    for (let i = 0, j = 0; i < data.length; i += 4, j++) {
      // ITU-R BT.601 luma: 0.299R + 0.587G + 0.114B
      const y = (data[i] * 77 + data[i + 1] * 150 + data[i + 2] * 29) >> 8;
      gray[j] = y;
      sum += y;
    }
    const yMean = sum / (160 * 90);
    let mad = 0;
    if (lumaPrevData) {
      let dSum = 0;
      for (let j = 0; j < gray.length; j++) {
        dSum += Math.abs(gray[j] - lumaPrevData[j]);
      }
      mad = dSum / gray.length;
    }
    lumaPrevData = gray;
    lumaBuffer.push({ t: Date.now(), yMean, mad });
    if (lumaBuffer.length > LUMA_MAX) lumaBuffer.shift();
  } catch (_e) {
    // video not ready or cross-origin
  }
}

function getRvfcAggregates(): Record<string, unknown> {
  if (rvfcBuffer.length === 0) return { count: 0 };
  const stalls = rvfcBuffer.filter((e) => e.stall);
  const jumps = rvfcBuffer.filter((e) => e.jump);
  const wallDeltas = rvfcBuffer.filter((e) => e.wallDeltaMs !== null).map((e) => e.wallDeltaMs!);
  const mediaDeltas = rvfcBuffer.filter((e) => e.mediaDeltaMs !== null).map((e) => e.mediaDeltaMs!);
  const mean = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const max = (arr: number[]) => arr.length ? Math.max(...arr) : 0;
  return {
    count: rvfcBuffer.length,
    meanWallDeltaMs: Number(mean(wallDeltas).toFixed(2)),
    maxWallDeltaMs: Number(max(wallDeltas).toFixed(2)),
    meanMediaDeltaMs: Number(mean(mediaDeltas).toFixed(2)),
    stallCount: stalls.length,
    jumpCount: jumps.length,
    stallDetails: stalls.slice(-5).map((e) => ({ wallDelta: e.wallDeltaMs, mediaTime: e.mediaTime })),
    jumpDetails: jumps.slice(-5).map((e) => ({ mediaDelta: e.mediaDeltaMs, mediaTime: e.mediaTime })),
  };
}

function getLumaAggregates(): Record<string, unknown> {
  if (lumaBuffer.length === 0) return { count: 0 };
  const yMeans = lumaBuffer.map((e) => e.yMean);
  const mads = lumaBuffer.filter((e) => e.mad > 0).map((e) => e.mad);
  const mean = (arr: number[]) => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
  const std = (arr: number[]) => {
    if (!arr.length) return 0;
    const m = mean(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  };
  return {
    count: lumaBuffer.length,
    yMeanAvg: Number(mean(yMeans).toFixed(4)),
    yMeanStd: Number(std(yMeans).toFixed(4)),
    yMeanMin: Number(Math.min(...yMeans).toFixed(4)),
    yMeanMax: Number(Math.max(...yMeans).toFixed(4)),
    madAvg: mads.length ? Number(mean(mads).toFixed(6)) : 0,
    madMax: mads.length ? Number(Math.max(...mads).toFixed(6)) : 0,
  };
}

// ── Flicker marker ──
async function triggerFlickerMarker(): Promise<void> {
  if (!SFU_DEBUG) return;
  flickerMarkerCount++;
  const markerIndex = flickerMarkerCount;
  const markerTs = Date.now();
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;

  // Collect full stats snapshot
  let fullStats: Record<string, unknown> = {};
  if (sfuPc) {
    try {
      const report = await sfuPc.getStats();
      const snap = buildStatsSnapshot(report, sfuPc);
      fullStats = snap;
    } catch (_e) {
      fullStats = { error: "getStats failed" };
    }
  }

  const videoState = video ? {
    videoWidth: video.videoWidth,
    videoHeight: video.videoHeight,
    readyState: video.readyState,
    paused: video.paused,
    currentTime: video.currentTime,
    playbackRate: video.playbackRate,
  } : null;

  const rvfcAgg = getRvfcAggregates();
  const rvfcRecent = rvfcBuffer.slice(-10);
  const lumaAgg = getLumaAggregates();
  const lumaRecent = lumaBuffer.slice(-10);

  const payload = {
    markerIndex,
    wallClock: markerTs,
    elapsedSinceSessionStart: sessionStartTime ? markerTs - sessionStartTime : null,
    elapsedSinceFirstFrame: sfuFirstFrameAt ? markerTs - sfuFirstFrameAt : null,
    iceState: sfuIceState,
    stats: fullStats,
    rvfc: rvfcAgg,
    rvfcRecent,
    luma: lumaAgg,
    lumaRecent,
    videoState,
  };

  sendTelemetry("sfu_user_flicker_marker", payload);

  const ib = (fullStats.inboundVideo ?? {}) as Record<string, unknown>;
  console.log(
    `[FLICKER_MARKER #${markerIndex}] lost=${ib.packetsLost ?? "?"} nack=${ib.nackCount ?? "?"} pli=${ib.pliCount ?? "?"} ` +
    `dropped=${ib.framesDropped ?? "?"} corrupted=${ib.framesCorrupted ?? "?"} fps=${ib.framesPerSecond ?? "?"} ` +
    `rvfc_stalls=${(rvfcAgg as Record<string, unknown>).stallCount ?? "?"} luma_std=${(lumaAgg as Record<string, unknown>).yMeanStd ?? "?"}`,
  );

  // Visual feedback on the marker button
  const btn = document.getElementById("flickerMarkerBtn");
  if (btn) {
    btn.textContent = `MARKED #${markerIndex}`;
    setTimeout(() => { btn.textContent = "MARK FLICKER"; }, 1000);
  }
}

// ── Parse sdpFmtpLine into key-value ──
function parseSdpFmtp(fmtp: string | null | undefined): Record<string, string> {
  if (!fmtp) return {};
  const out: Record<string, string> = {};
  for (const part of fmtp.split(";")) {
    const [k, ...v] = part.trim().split("=");
    if (k) out[k] = v.join("=");
  }
  return out;
}

// ── Robustly find the selected ICE candidate pair in a getStats() report ──
// Browsers vary: some mark the active pair with `selected: true`, some only
// expose it via transport.selectedCandidatePairId, and some require looking
// at `nominated: true && state === "succeeded"`. Prefer transport id first.
function findSelectedCandidatePair(
  report: RTCStatsReport,
): Record<string, unknown> | undefined {
  let transportSelectedId: string | undefined;

  for (const [, s] of report) {
    const r = s as Record<string, unknown>;
    if (r.type === "transport" && typeof r.selectedCandidatePairId === "string") {
      transportSelectedId = r.selectedCandidatePairId;
    }
  }

  for (const [, s] of report) {
    const r = s as Record<string, unknown>;
    if (r.type !== "candidate-pair") continue;

    const pairId = (r.id as string) ?? "";
    const isSelectedByTransport =
      transportSelectedId !== undefined && pairId === transportSelectedId;
    const isSelected = r.selected === true;
    const isNominatedSucceeded =
      r.nominated === true && r.state === "succeeded";

    if (isSelectedByTransport || isSelected || isNominatedSucceeded) {
      return r;
    }
  }

  return undefined;
}

// ── Build a comprehensive stats snapshot from a getStats() report ──
function buildStatsSnapshot(report: RTCStatsReport, pc: RTCPeerConnection): Record<string, unknown> {
  const now = Date.now();
  const snap: Record<string, unknown> = {
    t: now,
    connectionState: pc.connectionState,
    iceConnectionState: pc.iceConnectionState,
    signalingState: pc.signalingState,
    iceGatheringState: pc.iceGatheringState,
  };

  let videoCodecId: string | null = null;

  for (const [, s] of report) {
    const r = s as Record<string, unknown>;

    if (r.type === "inbound-rtp" && r.kind === "video") {
      videoCodecId = (r.codecId as string) ?? null;
      snap.inboundVideo = {
        framesDecoded: r.framesDecoded ?? null,
        keyFramesDecoded: r.keyFramesDecoded ?? null,
        framesDropped: r.framesDropped ?? null,
        framesPerSecond: r.framesPerSecond ?? null,
        frameWidth: r.frameWidth ?? null,
        frameHeight: r.frameHeight ?? null,
        packetsReceived: r.packetsReceived ?? null,
        packetsLost: r.packetsLost ?? null,
        pliCount: r.pliCount ?? null,
        nackCount: r.nackCount ?? null,
        firCount: r.firCount ?? null,
        jitter: r.jitter ?? null,
        jitterBufferDelay: r.jitterBufferDelay ?? null,
        jitterBufferEmittedCount: r.jitterBufferEmittedCount ?? null,
        totalDecodeTime: r.totalDecodeTime ?? null,
        qpSum: r.qpSum ?? null,
        totalVideoFrames: r.totalVideoFrames ?? null,
        codecId: r.codecId ?? null,
      };
    }

    if (r.type === "remote-inbound-rtp" && r.kind === "video") {
      snap.remoteInboundVideo = {
        roundTripTime: r.roundTripTime ?? null,
        packetsLost: r.packetsLost ?? null,
        fractionLost: r.fractionLost ?? null,
      };
    }

    // Candidate-pair selection is handled after the loop via
    // findSelectedCandidatePair() for robustness across browsers.

    if (r.type === "track") {
      snap.track = {
        kind: r.kind ?? null,
        remoteSource: r.remoteSource ?? null,
        detached: r.detached ?? null,
        enabled: r.enabled ?? null,
        readyState: r.readyState ?? null,
        framesReceived: r.framesReceived ?? null,
        framesDropped: r.framesDropped ?? null,
        framesCorrupted: r.framesCorrupted ?? null,
        totalFramesDelay: r.totalFramesDelay ?? null,
      };
    }
  }

  // Resolve the selected ICE candidate pair using the most robust method.
  const pair = findSelectedCandidatePair(report);
  if (pair) {
    const localCand = report.get(pair.localCandidateId as string);
    const remoteCand = report.get(pair.remoteCandidateId as string);
    snap.selectedCandidatePair = {
      id: pair.id ?? null,
      state: pair.state ?? null,
      nominated: pair.nominated ?? null,
      bytesReceived: pair.bytesReceived ?? null,
      bytesSent: pair.bytesSent ?? null,
      currentRoundTripTime: pair.currentRoundTripTime ?? null,
      availableOutgoingBitrate: pair.availableOutgoingBitrate ?? null,
      localCandidateType: (localCand as Record<string, unknown>)?.candidateType ?? null,
      localCandidateProtocol: (localCand as Record<string, unknown>)?.protocol ?? null,
      remoteCandidateType: (remoteCand as Record<string, unknown>)?.candidateType ?? null,
      remoteCandidateProtocol: (remoteCand as Record<string, unknown>)?.protocol ?? null,
    };
  }

  // Codec details with parsed fmtp
  if (videoCodecId) {
    const codec = report.get(videoCodecId);
    if (codec) {
      const c = codec as Record<string, unknown>;
      const fmtp = parseSdpFmtp(c.sdpFmtpLine as string | undefined);
      snap.codec = {
        mimeType: c.mimeType ?? null,
        clockRate: c.clockRate ?? null,
        sdpFmtpLine: c.sdpFmtpLine ?? null,
        profileLevelId: fmtp["profile-level-id"] ?? null,
        packetizationMode: fmtp["packetization-mode"] ?? null,
        levelAsymmetryAllowed: fmtp["level-asymmetry-allowed"] ?? null,
      };
    }
  }

  // Video element state
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (video) {
    snap.video = {
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      readyState: video.readyState,
      paused: video.paused,
      currentTime: video.currentTime,
      hasSrcObject: video.srcObject !== null,
      srcObjectType: video.srcObject ? video.srcObject.constructor.name : null,
    };
  }

  return snap;
}

/** Collect a full stats snapshot every 2s. Emits telemetry every 10s only. */
async function collectSfuDiagStats(): Promise<void> {
  if (!SFU_DEBUG) return;
  const pc = sfuPc;
  if (!pc) return;

  const report = await pc.getStats();
  const snap = buildStatsSnapshot(report, pc);

  // Sample luma on the 2s cadence
  sampleLuma();

  sfuDiagState.stats.push({ t: snap.t as number, label: "stats", ...snap });
  sfuDiagState.pcState = snap;

  // Update visible panel
  updateSfuDiagPanel(snap);

  // Emit compact telemetry every 10s only
  const now = Date.now();
  if (now - sfuDebugStatsLastEmit >= 10_000) {
    sfuDebugStatsLastEmit = now;
    const ib = (snap.inboundVideo ?? {}) as Record<string, unknown>;
    const cp = (snap.selectedCandidatePair ?? {}) as Record<string, unknown>;
    sendTelemetry("sfu_debug_stats", {
      connectionState: snap.connectionState,
      iceConnectionState: snap.iceConnectionState,
      framesDecoded: ib.framesDecoded ?? null,
      keyFramesDecoded: ib.keyFramesDecoded ?? null,
      framesDropped: ib.framesDropped ?? null,
      framesPerSecond: ib.framesPerSecond ?? null,
      packetsReceived: ib.packetsReceived ?? null,
      packetsLost: ib.packetsLost ?? null,
      nackCount: ib.nackCount ?? null,
      pliCount: ib.pliCount ?? null,
      firCount: ib.firCount ?? null,
      jitter: ib.jitter ?? null,
      jitterBufferDelay: ib.jitterBufferDelay ?? null,
      jitterBufferEmittedCount: ib.jitterBufferEmittedCount ?? null,
      totalDecodeTime: ib.totalDecodeTime ?? null,
      qpSum: ib.qpSum ?? null,
      totalVideoFrames: ib.totalVideoFrames ?? null,
      framesReceived: ((snap.track ?? {}) as Record<string, unknown>).framesReceived ?? null,
      framesCorrupted: ((snap.track ?? {}) as Record<string, unknown>).framesCorrupted ?? null,
      localCandidateType: cp.localCandidateType ?? null,
      localCandidateProtocol: cp.localCandidateProtocol ?? null,
      remoteCandidateType: cp.remoteCandidateType ?? null,
      remoteCandidateProtocol: cp.remoteCandidateProtocol ?? null,
      candidatePairState: cp.state ?? null,
      candidatePairNominated: cp.nominated ?? null,
      candidatePairBytesSent: cp.bytesSent ?? null,
      candidatePairBytesReceived: cp.bytesReceived ?? null,
      candidatePairRtt: cp.currentRoundTripTime ?? null,
      candidatePairAvailableBitrate: cp.availableOutgoingBitrate ?? null,
      rvfc: getRvfcAggregates(),
      luma: getLumaAggregates(),
      markerCount: flickerMarkerCount,
    });
  }
}

function updateSfuDiagPanel(snap: Record<string, unknown>): void {
  const panel = document.getElementById("sfuDiagPanel");
  if (!panel) return;

  const v = snap.video as Record<string, unknown> | undefined;
  const ib = snap.inboundVideo as Record<string, unknown> | undefined;
  const cp = snap.selectedCandidatePair as Record<string, unknown> | undefined;
  const codec = snap.codec as Record<string, unknown> | undefined;

  const dims = v ? `${v.videoWidth ?? 0}x${v.videoHeight ?? 0}` : "—";
  const decoded = ib?.framesDecoded ?? "—";
  const keyframes = ib?.keyFramesDecoded ?? "—";
  const dropped = ib?.framesDropped ?? "—";
  const fps = ib?.framesPerSecond != null ? Number(ib.framesPerSecond).toFixed(1) : "—";
  const lost = ib?.packetsLost ?? "—";
  const nack = ib?.nackCount ?? "—";
  const pli = ib?.pliCount ?? "—";
  const fir = ib?.firCount ?? "—";
  const jitter = ib?.jitter != null ? Number(ib.jitter).toFixed(3) : "—";
  const qpSum = ib?.qpSum ?? "—";
  const ice = snap.iceConnectionState ?? "—";
  const conn = snap.connectionState ?? "—";
  const localType = cp?.localCandidateType ?? "—";
  const remoteType = cp?.remoteCandidateType ?? "—";
  const mimeType = codec?.mimeType ?? "—";

  const rvfcStalls = rvfcBuffer.filter((e) => e.stall).length;
  const rvfcJumps = rvfcBuffer.filter((e) => e.jump).length;
  const lumaStd = lumaBuffer.length > 1
    ? (() => {
        const ms = lumaBuffer.map((e) => e.yMean);
        const m = ms.reduce((a, b) => a + b, 0) / ms.length;
        return Math.sqrt(ms.reduce((s, v) => s + (v - m) ** 2, 0) / ms.length);
      })().toFixed(4)
    : "—";

  panel.innerHTML =
    `<div style="font-family:monospace;font-size:11px;line-height:1.5;padding:8px;background:#111;color:#0f0;border:1px solid #333;border-radius:4px;max-width:580px">` +
    `<b>SFU DIAG</b> conn=${conn} ice=${ice} markers=${flickerMarkerCount}<br>` +
    `video=${dims} decoded=${decoded} key=${keyframes} dropped=${dropped} fps=${fps}<br>` +
    `lost=${lost} nack=${nack} pli=${pli} fir=${fir} jitter=${jitter} qpSum=${qpSum}<br>` +
    `rvfc: stalls=${rvfcStalls} jumps=${rvfcJumps} | lumaStd=${lumaStd}<br>` +
    `cand: ${localType}→${remoteType} codec=${mimeType}` +
    `</div>`;
}

function startSfuDiagCollection(): void {
  if (!SFU_DEBUG) return;
  if (sfuDiagInterval) return;
  sfuDiagInterval = setInterval(() => { void collectSfuDiagStats(); }, 2_000);
  startLumaSampler();
  // Expose dump helper
  (window as unknown as Record<string, unknown>).__sfuDiag = sfuDiagState;
  (window as unknown as Record<string, unknown>).__dumpSfuDiag = (): string =>
    JSON.stringify(sfuDiagState, null, 2);
}

function createSfuDiagPanel(): void {
  if (!SFU_DEBUG) return;
  if (document.getElementById("sfuDiagPanel")) return;
  const panel = document.createElement("div");
  panel.id = "sfuDiagPanel";
  panel.style.cssText = "position:fixed;bottom:8px;right:8px;z-index:99999;pointer-events:none;";
  document.body.appendChild(panel);

  // Flicker marker button
  const btn = document.createElement("button");
  btn.id = "flickerMarkerBtn";
  btn.textContent = "MARK FLICKER";
  btn.style.cssText = "position:fixed;bottom:8px;left:50%;transform:translateX(-50%);z-index:99999;padding:12px 24px;font-size:16px;font-weight:bold;background:#d32f2f;color:white;border:none;border-radius:8px;cursor:pointer;box-shadow:0 4px 12px rgba(0,0,0,0.3);";
  btn.addEventListener("click", () => { void triggerFlickerMarker(); });
  document.body.appendChild(btn);

  // Keyboard shortcut
  document.addEventListener("keydown", (e: KeyboardEvent) => {
    if (!SFU_DEBUG) return;
    if (e.key === "f" || e.key === "F") {
      e.preventDefault();
      void triggerFlickerMarker();
    }
  });
}

interface PendingAudio {
  base64: string;
  sampleRate: number;
  turnId: number;
  timeoutId: number;
  arrivedAt: number;
}

let pendingAudio: PendingAudio | null = null;
interface SpeechAudioPlayback {
  audioContext: AudioContext;
  audioStartTime: number;
  audioDurationMs: number;
  stop: () => void;
}

let speechFrameImages: HTMLImageElement[] = [];
let speechAnimationRafId: number | null = null;
let speechAnimationFrameIndex = 0;
let speechFps = 25;
let speechRenderedDurationMs = 0;
let speechOriginalDurationMs = 0;
let speechPlaybackStarted = false;
let speechAudioPlayback: SpeechAudioPlayback | null = null;
let speechClockDriftLastReportAt = 0;
let idleFrameImages: HTMLImageElement[] = [];
let idleAnimationTimer: number | null = null;
let idleFrameIndex = 0;
let idleFps = 25;
let idleReady = false;
let avatarSpeechActive = false;
let idleIsLive = false;
let idleBufferMaxFrames = 50;
let currentSpeechTurnId = 0;
let activeSpeechTurnId: number | null = null;
let fallbackPlayedTurnIds: Set<number> = new Set();
const AVATAR_RENDER_TIMEOUT_MS = 20_000;
const AVATAR_FIRST_CHUNK_TIMEOUT_MS = 15_000;
const SFU_SPEECH_QUEUED_TIMEOUT_MS = 10_000;
const FRAMES_PER_SPEECH_CHUNK = 25;

function getSpeechStartBufferThreshold(expectedFrames: number): number {
  if (!Number.isFinite(expectedFrames) || expectedFrames <= 0) return 10;
  return Math.min(50, Math.max(10, Math.ceil(expectedFrames * 0.10)));
}

let idleChunkDiag = {
  count: 0,
  totalImageCreateMs: 0,
  totalImageDecodeMs: 0,
  totalBase64Chars: 0,
  totalPayloadChars: 0,
  lastReportAt: 0,
};
let idleUnderrunCountSinceReport = 0;
let idleUnderrunLastReportAt = 0;
let idleDiagIntervalId: number | null = null;

let speechTimingAssistantAudioAt = 0;
let speechTimingRenderStartAt = 0;
let speechTimingSpeechStartAt = 0;
let speechTimingFirstChunkAt = 0;
let speechTimingPlaybackStartAt = 0;
let speechHadChunksBeforeFallback = false;
let avatarFirstChunkTimeoutId: number | null = null;

let speechStreamEnded = false;
let speechExpectedTotalFrames = 0;
let speechDisplayedFrames = 0;
let speechUnderrunCount = 0;
let speechLastUnderrunReportAt = 0;
let speechAudioEnded = false;
let speechVideoEnded = false;
let speechAckSentForTurnId: number | null = null;
let speechPlaybackTurnId: number | null = null;

let fmp4InitSeen = false;
let fmp4MseReady = false;
let fmp4FatalError = false;
let fmp4ModeExpected = false;
let fmp4MediaSource: MediaSource | null = null;
let fmp4SourceBuffer: SourceBuffer | null = null;
interface Fmp4SegmentEntry {
  b64: string;
  byteLength: number;
  sequence: number;
  mode: string;
}

let fmp4SegmentQueue: Fmp4SegmentEntry[] = [];
let fmp4IsAppending = false;
let fmp4IsTrimming = false;
let fmp4SegmentCount = 0;
let fmp4InitReceivedAt = 0;
let fmp4LiveEdgeCheckInterval: number | null = null;
let fmp4BufferHealthInterval: number | null = null;
let fmp4DroppedFatalCount = 0;
let fmp4DroppedFatalLastTelemetryAt = 0;
let fmp4ListenersAttached = false;
let fmp4CurrentAppendByteLen: number | null = null;
let fmp4LastTimeupdateTelemetryAt = 0;
let fmp4AvMode = false;
let fmp4SpeechPlaybackAckTimerId: ReturnType<typeof setTimeout> | null = null;
let fmp4SpeechActive = false;
let fmp4LastSpeechEndedAt = 0;
let fmp4SeekBlockedUntil = 0;
let fmp4AvPlayOverlay: HTMLDivElement | null = null;
let fmp4SummaryQueuedCount = 0;
let fmp4SummaryAppendedCount = 0;
let fmp4SummaryTotalBytes = 0;
let fmp4SummaryLastSentAt = 0;
const FMP4_SUMMARY_INTERVAL_MS = 30000;
const FMP4_SUMMARY_MIN_SEGMENTS = 100;
const VERBOSE_AV_TELEMETRY = new URLSearchParams(window.location.search).has("verbose_av_telemetry");
let fmp4SessionReadyAt = 0;
let fmp4FirstPlayingAt = 0;
let fmp4WaitingCount = 0;
let fmp4StalledCount = 0;
let fmp4SegmentsReceived = 0;
let fmp4SegmentsQueued = 0;
let fmp4SegmentsAppended = 0;
let fmp4AppendErrors = 0;
let fmp4LastReceivedSequence: number | null = null;
let fmp4LastAppendedSequence: number | null = null;
let fmp4LastAppendedAt = 0;

type Fmp4LifecycleState = "idle" | "starting" | "playing" | "ending" | "destroyed" | "error";
let fmp4Lifecycle: Fmp4LifecycleState = "idle";
let fmp4Generation = 0;

function fmp4IsActive(): boolean {
  return fmp4Lifecycle === "starting" || fmp4Lifecycle === "playing";
}

const VERBOSE_DELIVERY_TELEMETRY = new URLSearchParams(window.location.search).has("verbose_delivery");
const SFU_MODE = new URLSearchParams(window.location.search).get("transport") === "webrtc_sfu";
const DIRECT_WEBRTC_DEBUG_MODE = new URLSearchParams(window.location.search).get("transport") === "direct_webrtc_debug";
const WHEP_MODE = new URLSearchParams(window.location.search).get("transport") === "whep";
const SFU_DEBUG = (SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE || WHEP_MODE) && new URLSearchParams(window.location.search).get("sfu_debug") === "1";

// Phase 3F/3G: read and immediately scrub the SFU beta token from the URL.
// The token is stored in sessionStorage (not localStorage) as a fallback so
// that a page reload after URL scrubbing still has access. sessionStorage is
// cleared when the tab closes. The token is short-lived (2h TTL) and is never
// logged or sent anywhere except POST /sessions.
const SFU_BETA_SESSION_KEY = "sfu_beta_token";

function extractAndScrubSfuBetaToken(): string | null {
  const params = new URLSearchParams(window.location.search);
  const token = params.get("sfu_beta_token");

  if (token !== null) {
    // Store in sessionStorage for reload-survival, then scrub from URL.
    try {
      sessionStorage.setItem(SFU_BETA_SESSION_KEY, token);
    } catch { /* sessionStorage may be unavailable in private mode */ }
    params.delete("sfu_beta_token");
    const newUrl = `${window.location.pathname}${params.toString() ? "?" + params.toString() : ""}${window.location.hash}`;
    window.history.replaceState({}, "", newUrl);
    return token;
  }

  // Token not in URL — try sessionStorage (reload after scrub).
  try {
    return sessionStorage.getItem(SFU_BETA_SESSION_KEY);
  } catch {
    return null;
  }
}

const SFU_BETA_TOKEN = extractAndScrubSfuBetaToken();

// Safe diagnostics: boolean only, never the token value.
const SFU_BETA_TOKEN_DETECTED = SFU_BETA_TOKEN !== null;
const SFU_BETA_TOKEN_SCRUBBED = (() => {
  const params = new URLSearchParams(window.location.search);
  return !params.has("sfu_beta_token");
})();

interface Fmp4PendingInit {
  dataBase64: string;
  byteLength: number;
  timingMs: number;
  receivedAt: number;
}

let fmp4PendingInitData: Fmp4PendingInit | null = null;
let fmp4MseEagerCreatedAt = 0;
let fmp4SourceopenAt = 0;
let fmp4InitAppendedAt = 0;

function showAvPlayOverlay(): void {
  if (fmp4AvPlayOverlay) return;
  const overlay = document.createElement("div");
  overlay.className = "av-play-overlay";
  overlay.innerHTML = '<button class="av-play-btn">Tap to start audio/video</button>';
  const handleClick = () => {
    sendTelemetry("avatar_fmp4_av_user_play_clicked");
    const video = ensureFmp4VideoElement();
    video.play().then(() => {
      sendTelemetry("avatar_fmp4_av_user_play_resolved");
      hideAvPlayOverlay();
    }).catch((err: unknown) => {
      sendTelemetry("avatar_fmp4_av_user_play_rejected", { error: String(err) });
      log("err", `AV user-triggered play rejected: ${err}`);
    });
  };
  overlay.addEventListener("click", handleClick);
  avatarFrame.appendChild(overlay);
  fmp4AvPlayOverlay = overlay;
}

function hideAvPlayOverlay(): void {
  if (!fmp4AvPlayOverlay) return;
  fmp4AvPlayOverlay.remove();
  fmp4AvPlayOverlay = null;
}

function initFullscreenBtn(): void {
  const btn = document.getElementById("fmp4FullscreenBtn");
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (avatarFrame.requestFullscreen) {
      avatarFrame.requestFullscreen();
    } else if ((avatarFrame as unknown as { webkitRequestFullscreen: () => void }).webkitRequestFullscreen) {
      (avatarFrame as unknown as { webkitRequestFullscreen: () => void }).webkitRequestFullscreen();
    }
  });
}

let pendingAudioSupersedeState: {
  turnId: number;
  arrivedAt: number;
  hadSpeechStart: boolean;
  hadSpeechChunks: number;
} | null = null;

const ROLLING_WINDOW = 50;
const rollingBuffer: { rms: number; peak: number; nearZeroRatio: number; clippedRatio: number }[] = [];

function flushIdleChunkDiag() {
  if (idleChunkDiag.count === 0) return;
  const now = performance.now();
  sendTelemetry("avatar_idle_chunk_browser_processed", {
    chunkCount: idleChunkDiag.count,
    avgImageCreateMs: idleChunkDiag.totalImageCreateMs / idleChunkDiag.count,
    avgImageDecodeMs: idleChunkDiag.totalImageDecodeMs / idleChunkDiag.count,
    totalBase64Chars: idleChunkDiag.totalBase64Chars,
    totalPayloadChars: idleChunkDiag.totalPayloadChars,
    bufferedFrames: idleFrameImages.length,
    idleIsLive,
  });
  idleChunkDiag = { count: 0, totalImageCreateMs: 0, totalImageDecodeMs: 0, totalBase64Chars: 0, totalPayloadChars: 0, lastReportAt: now };
}

function startIdleDiagInterval() {
  stopIdleDiagInterval();
  idleDiagIntervalId = window.setInterval(() => {
    flushIdleChunkDiag();
    if (idleUnderrunCountSinceReport > 0) {
      const now = performance.now();
      sendTelemetry("avatar_idle_buffer_underrun", {
        underrunCountSinceLastReport: idleUnderrunCountSinceReport,
        bufferedFrames: idleFrameImages.length,
        displayedIndex: idleFrameIndex,
        idleIsLive,
        reportIntervalMs: now - idleUnderrunLastReportAt,
      });
      idleUnderrunCountSinceReport = 0;
      idleUnderrunLastReportAt = now;
    }
    const chunks = idleChunkDiag.count;
    const avgModal = "see_posthog";
    const avgPayloadMB = idleChunkDiag.totalPayloadChars > 0 ? (idleChunkDiag.totalPayloadChars / idleChunkDiag.count * 0.75 / 1024 / 1024).toFixed(2) : "0";
    const avgDecode = idleChunkDiag.count > 0 ? (idleChunkDiag.totalImageDecodeMs / idleChunkDiag.count).toFixed(1) : "0";
    log("info", `[idle diagnostics] chunks=${chunks} avgPayloadMB=${avgPayloadMB} avgDecodeMs=${avgDecode} underruns=${idleUnderrunCountSinceReport} bufferLen=${idleFrameImages.length}`);
  }, 10_000);
}

function stopIdleDiagInterval() {
  if (idleDiagIntervalId !== null) {
    clearInterval(idleDiagIntervalId);
    idleDiagIntervalId = null;
  }
}

const telemetryQueue: Array<{
  event: string;
  properties?: Record<string, unknown>;
}> = [];

copyLogsBtn.addEventListener("click", () => {
  const entries = logOutput.querySelectorAll(".log-entry");
  const text = Array.from(entries)
    .map((el) => el.textContent?.trim() ?? "")
    .join("\n");
  navigator.clipboard.writeText(text).then(() => {
    copyLogsBtn.classList.add("copied");
    setTimeout(() => copyLogsBtn.classList.remove("copied"), 1500);
  });
});

initFullscreenBtn();

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) {
    bytes[i] = bin.charCodeAt(i);
  }
  return bytes.buffer;
}

function fmp4DrainQueue(): void {
  if (!fmp4IsActive()) return;
  if (!fmp4MseReady || !fmp4SourceBuffer || fmp4IsAppending || fmp4IsTrimming || fmp4SegmentQueue.length === 0) return;
  if (fmp4SourceBuffer.updating) return;

  fmp4IsAppending = true;
  const entry = fmp4SegmentQueue.shift()!;
  const segment = base64ToArrayBuffer(entry.b64);
  const byteLen = segment.byteLength;

  fmp4SummaryAppendedCount++;
  fmp4SummaryTotalBytes += byteLen;
  fmp4CurrentAppendByteLen = byteLen;

  try {
    fmp4SourceBuffer.appendBuffer(segment);
  } catch (appendErr) {
    log("err", `fMP4 appendBuffer failed: ${appendErr}`);
    fmp4AppendErrors++;
    sendTelemetry("avatar_fmp4_browser_append_failed", {
      error: String(appendErr),
      stage: "media_append",
      byteLength: byteLen,
      receivedCount: fmp4SegmentsReceived,
      queuedCount: fmp4SegmentsQueued,
      appendedCount: fmp4SegmentsAppended,
      appendErrors: fmp4AppendErrors,
      queueLength: fmp4SegmentQueue.length,
      segmentCount: fmp4SegmentCount,
    });
    fmp4IsAppending = false;
    fmp4CurrentAppendByteLen = null;
  }
}

function fmp4MaybeTrimBuffer(): void {
  if (!fmp4SourceBuffer || fmp4IsAppending || fmp4IsTrimming || fmp4SourceBuffer.updating) return;
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (!video) return;

  const sb = fmp4SourceBuffer;
  if (sb.buffered.length === 0) return;

  const currentTime = video.currentTime;
  const bufferedStart = sb.buffered.start(0);
  const bufferedEnd = sb.buffered.end(sb.buffered.length - 1);
  const TRIM_THRESHOLD_S = 10;
  const TRIM_KEEP_AHEAD_S = 5;

  if (currentTime - bufferedStart < TRIM_THRESHOLD_S) return;

  const removeEnd = Math.max(bufferedStart, currentTime - TRIM_KEEP_AHEAD_S);
  if (removeEnd <= bufferedStart) return;

  try {
    fmp4IsTrimming = true;
    log("info", `fMP4 SourceBuffer trim: [${bufferedStart.toFixed(2)}, ${removeEnd.toFixed(2)}] currentTime=${currentTime.toFixed(2)} bufferedEnd=${bufferedEnd.toFixed(2)}`);
    sb.remove(bufferedStart, removeEnd);
  } catch (err) {
    fmp4IsTrimming = false;
    log("err", `fMP4 SourceBuffer trim failed: ${err}`);
    sendTelemetry("media_buffer_trim_failed", {
      error: String(err),
      bufferedStart,
      removeEnd,
      currentTime,
    });
  }
}

function sfuFail(stage: string, detail: string): void {
  sfuActive = false;
  log("err", `SFU FAILED at ${stage}: ${detail}`);
  sendTelemetry("sfu_failed", { stage, detail });
  setStatus("error", `SFU failed (${stage}). Reload without ?transport=webrtc_sfu to use fMP4.`);
  sfuCleanupLocal();
  const diag = document.getElementById("sfuDiag");
  if (diag) diag.textContent = `SFU FAILED at ${stage}: ${detail}`;
}

function sfuCleanupLocal(): void {
  if (sfuStatsInterval !== null) {
    clearInterval(sfuStatsInterval);
    sfuStatsInterval = null;
  }
  if (sfuPc) {
    try {
      sfuPc.ontrack = null;
      sfuPc.oniceconnectionstatechange = null;
      sfuPc.getReceivers().forEach((r) => { try { r.track?.stop(); } catch { /* best-effort */ } });
      sfuPc.close();
    } catch { /* best-effort */ }
    sfuPc = null;
  }
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (video) {
    video.srcObject = null;
    video.style.display = "none";
  }
  sfuActive = false;
  sfuIceState = "new";
  sfuVideoAttachedAt = 0;
  sfuFirstFrameAt = 0;
  sfuLastFramesDecoded = 0;
  sfuRenderAttachFailed = false;
  sfuVideoListenersAttached = false;
  sfuStatsLogLastAt = 0;
  sfuStatsLogLastCurrentTime = 0;
  sfuStatsLogLastFramesDecoded = 0;
  sfuStatsLogLastPacketsLost = 0;
  sfuStatsTelemetryLastAt = 0;
  sfuStatsAnomalyLastAt = 0;
  sfuDebugStatsLastEmit = 0;
  stopRvfcTracking();
}

function markSfuVideoLive(video: HTMLVideoElement, source: string): void {
  if (sfuFirstFrameAt !== 0) return;
  if (video.videoWidth <= 0 || video.videoHeight <= 0) return;

  sfuFirstFrameAt = Date.now();
  sfuActive = true;
  setLoadingStage("live");
  hideAvatarLoadingOverlay();
  const modeLabel = DIRECT_WEBRTC_DEBUG_MODE ? "Direct WebRTC" : "SFU";
  setStatus("rendering", `${modeLabel} video streaming`);
  avatarFrame.classList.add("fmp4-active");
  frameMeta.textContent = DIRECT_WEBRTC_DEBUG_MODE ? "Direct WebRTC video mode" : "SFU WebRTC video mode";
  log("info", `${modeLabel} video live via ${source}: ${video.videoWidth}x${video.videoHeight} readyState=${video.readyState}`);
  sendTelemetry("sfu_video_first_frame", {
    source,
    width: video.videoWidth,
    height: video.videoHeight,
    timeSinceAttachMs: sfuVideoAttachedAt ? Date.now() - sfuVideoAttachedAt : undefined,
    timeSinceSessionReadyMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
  });
  if (placeholderShownAt) {
    sendTelemetry("first_live_frame_after_placeholder_ms", {
      durationMs: Date.now() - placeholderShownAt,
      sessionId: currentSessionId,
      transport: "webrtc_sfu",
    });
  }
}

function attachSfuVideoListeners(video: HTMLVideoElement): void {
  if (sfuVideoListenersAttached) return;
  sfuVideoListenersAttached = true;

  const events = ["loadedmetadata", "resize", "canplay", "playing", "waiting", "stalled", "pause"];
  for (const ev of events) {
    video.addEventListener(ev, () => {
      if ((!SFU_MODE && !DIRECT_WEBRTC_DEBUG_MODE) || !sfuPc) return;
      const modeLabel = DIRECT_WEBRTC_DEBUG_MODE ? "Direct WebRTC" : "SFU";
      log("info", `${modeLabel} video event: ${ev} currentTime=${video.currentTime.toFixed(2)} readyState=${video.readyState} paused=${video.paused} w=${video.videoWidth} h=${video.videoHeight}`);
      if (ev === "waiting" || ev === "stalled") {
        sendTelemetry("sfu_video_playback_issue", {
          eventName: ev,
          currentTime: video.currentTime,
          readyState: video.readyState,
          videoWidth: video.videoWidth,
          videoHeight: video.videoHeight,
          iceState: sfuIceState,
        });
      }
      markSfuVideoLive(video, ev);
    });
  }

  video.addEventListener("error", () => {
    if ((!SFU_MODE && !DIRECT_WEBRTC_DEBUG_MODE) || !sfuPc) return;
    const err = video.error;
    log("err", `SFU video error: code=${err?.code} message=${err?.message}`);
    sendTelemetry("sfu_video_error", {
      errorCode: err?.code,
      errorMessage: err?.message,
      readyState: video.readyState,
      networkState: video.networkState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    });
  });

  if (SFU_DEBUG) {
    const cs = getComputedStyle(video);
    sendTelemetry("sfu_debug_video_css", {
      display: cs.display,
      visibility: cs.visibility,
      opacity: cs.opacity,
      width: cs.width,
      height: cs.height,
      zIndex: cs.zIndex,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
    });
    startRvfcTracking();
  }
}

function sfuCleanup(): void {
  sfuCleanupLocal();
  if (sfuCloseUrl && sfuToken) {
    const url = `${sfuCloseUrl}?token=${encodeURIComponent(sfuToken)}`;
    try {
      void fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}", keepalive: true });
    } catch { /* best-effort */ }
  }
  sfuToken = null;
  sfuSubscribeUrl = "";
  sfuAnswerUrl = "";
  sfuCloseUrl = "";
}

async function startSfuSubscriber(): Promise<void> {
  if (!sfuToken || !sfuSubscribeUrl || !sfuAnswerUrl) {
    sfuFail("startSfuSubscriber", "Missing sfu_token or URLs");
    return;
  }
  try {
    log("info", "SFU subscriber: POST /sfu/subscribe ...");
    const subResp = await fetch(`${sfuSubscribeUrl}?token=${encodeURIComponent(sfuToken)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    if (!subResp.ok) {
      const text = await subResp.text();
      sfuFail("subscribe", `HTTP ${subResp.status}: ${text}`);
      return;
    }
    const sub = await subResp.json();
    if (sub.error) {
      sfuFail("subscribe", JSON.stringify(sub.error));
      return;
    }
    log("info", `SFU subscriber: offer received, mids=${JSON.stringify(sub.mids)}`);

    sfuPc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });

    // Non-invasive diagnostics: passive listeners only (sfu_debug=1).
    // Does NOT override ontrack/oniceconnectionstatechange or set srcObject.
    attachSfuDiagListeners(sfuPc);
    startSfuDiagCollection();

    sfuPc.oniceconnectionstatechange = () => {
      if (sfuPc) {
        sfuIceState = sfuPc.iceConnectionState;
        log("info", `SFU ICE state: ${sfuIceState}`);
        if (sfuIceState === "failed") {
          sfuFail("ice", "ICE connection failed");
        }
      }
    };

    const video = ensureFmp4VideoElement();
    sfuPc.ontrack = (e: RTCTrackEvent) => {
      log("info", `SFU ontrack: kind=${e.track.kind} id=${e.track.id} streams=${e.streams.length}`);
      if (e.track.kind === "video") {
        attachSfuVideoListeners(video);
        const eventStream = e.streams.find((stream) => stream.getVideoTracks().includes(e.track));
        const remoteStream = eventStream ?? new MediaStream([e.track]);
        video.srcObject = remoteStream;
        video.style.display = "block";
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;
        sfuVideoAttachedAt = Date.now();
        sfuRenderAttachFailed = false;
        log("info", `SFU video attached: streamTracks=${remoteStream.getVideoTracks().length}`);
        video.play()
          .then(() => markSfuVideoLive(video, "play-resolved"))
          .catch((err) => log("err", `SFU video play() failed: ${err}`));
        sfuActive = true;
        avatarFrame.classList.add("fmp4-active");
      }
    };

    await sfuPc.setRemoteDescription({ type: sub.offer_type || "offer", sdp: sub.offer_sdp });
    log("info", "SFU subscriber: remote description set");

    const answer = await sfuPc.createAnswer();
    await sfuPc.setLocalDescription(answer);
    log("info", "SFU subscriber: waiting for ICE gathering...");

    if (sfuPc.iceGatheringState !== "complete") {
      await new Promise<void>((resolve) => {
        const check = () => {
          if (sfuPc && sfuPc.iceGatheringState === "complete") {
            sfuPc.removeEventListener("icegatheringstatechange", check);
            resolve();
          }
        };
        sfuPc!.addEventListener("icegatheringstatechange", check);
        setTimeout(resolve, 8000);
      });
    }
    log("info", "SFU subscriber: ICE gathering done");

    const localSdp = sfuPc.localDescription!;
    const ansResp = await fetch(`${sfuAnswerUrl}?token=${encodeURIComponent(sfuToken)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_sdp: localSdp.sdp, answer_type: localSdp.type }),
    });
    if (!ansResp.ok) {
      const text = await ansResp.text();
      sfuFail("answer", `HTTP ${ansResp.status}: ${text}`);
      return;
    }
    const ansBody = await ansResp.json();
    if (ansBody.error) {
      sfuFail("answer", JSON.stringify(ansBody.error));
      return;
    }
    log("info", "SFU subscriber: answer accepted — waiting for media");
    if (sfuFirstFrameAt === 0) {
      setStatus("ready", "SFU answer accepted — waiting for video...");
    }

    sfuStatsInterval = setInterval(updateSfuDiag, 2000);
  } catch (err) {
    sfuFail("startSfuSubscriber", String(err));
  }
}

async function startDirectWebrtcSubscriber(offerUrl: string, answerUrl: string, iceServers: RTCIceServer[]): Promise<void> {
  try {
    log("info", `Direct WebRTC: polling offer from ${offerUrl.split("?")[0]} ...`);
    setStatus("ready", "Direct WebRTC — waiting for publisher offer...");

    const offer = await pollForDirectWebrtcOffer(offerUrl);
    if (!offer) {
      sfuFail("direct_webrtc_offer", "Timed out waiting for publisher offer");
      return;
    }
    log("info", `Direct WebRTC: offer received (type=${offer.type}, mid=${offer.mid})`);

    sfuPc = new RTCPeerConnection({ iceServers });

    attachSfuDiagListeners(sfuPc);
    startSfuDiagCollection();

    sfuPc.oniceconnectionstatechange = () => {
      if (sfuPc) {
        sfuIceState = sfuPc.iceConnectionState;
        log("info", `Direct WebRTC ICE state: ${sfuIceState}`);
        if (sfuIceState === "failed") {
          sfuFail("direct_webrtc_ice", "ICE connection failed");
        }
      }
    };

    const video = ensureFmp4VideoElement();
    sfuPc.ontrack = (e: RTCTrackEvent) => {
      log("info", `Direct WebRTC ontrack: kind=${e.track.kind} id=${e.track.id}`);
      if (e.track.kind === "video") {
        attachSfuVideoListeners(video);
        const eventStream = e.streams.find((stream) => stream.getVideoTracks().includes(e.track));
        const remoteStream = eventStream ?? new MediaStream([e.track]);
        video.srcObject = remoteStream;
        video.style.display = "block";
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;
        sfuVideoAttachedAt = Date.now();
        sfuRenderAttachFailed = false;
        log("info", `Direct WebRTC video attached`);
        video.play()
          .then(() => markSfuVideoLive(video, "direct-play-resolved"))
          .catch((err) => log("err", `Direct WebRTC video play() failed: ${err}`));
        sfuActive = true;
        avatarFrame.classList.add("fmp4-active");
      }
    };

    await sfuPc.setRemoteDescription({ type: (offer.type || "offer") as RTCSdpType, sdp: offer.sdp });
    log("info", "Direct WebRTC: remote description set");

    const answer = await sfuPc.createAnswer();
    await sfuPc.setLocalDescription(answer);
    log("info", "Direct WebRTC: waiting for ICE gathering...");

    if (sfuPc.iceGatheringState !== "complete") {
      await new Promise<void>((resolve) => {
        const check = () => {
          if (sfuPc && sfuPc.iceGatheringState === "complete") {
            sfuPc.removeEventListener("icegatheringstatechange", check);
            resolve();
          }
        };
        sfuPc!.addEventListener("icegatheringstatechange", check);
        setTimeout(resolve, 8000);
      });
    }
    log("info", "Direct WebRTC: ICE gathering done");

    const localSdp = sfuPc.localDescription!;
    const ansResp = await fetch(answerUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_sdp: localSdp.sdp, answer_type: localSdp.type }),
    });
    if (!ansResp.ok) {
      const text = await ansResp.text();
      sfuFail("direct_webrtc_answer", `HTTP ${ansResp.status}: ${text}`);
      return;
    }
    const ansBody = await ansResp.json();
    if (ansBody.error) {
      sfuFail("direct_webrtc_answer", JSON.stringify(ansBody.error));
      return;
    }
    log("info", "Direct WebRTC: answer accepted — waiting for media");
    if (sfuFirstFrameAt === 0) {
      setStatus("ready", "Direct WebRTC answer accepted — waiting for video...");
    }

    sfuStatsInterval = setInterval(updateSfuDiag, 2000);
  } catch (err) {
    sfuFail("startDirectWebrtcSubscriber", String(err));
  }
}

async function startWhepSubscriber(whepUrl: string): Promise<void> {
  try {
    log("info", `WHEP: connecting to CF Stream...`);
    setStatus("ready", "WHEP — negotiating with CF Stream...");

    const pc = new RTCPeerConnection({
      iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
    });
    sfuPc = pc;

    attachSfuDiagListeners(pc);
    startSfuDiagCollection();

    pc.oniceconnectionstatechange = () => {
      if (sfuPc) {
        sfuIceState = sfuPc.iceConnectionState;
        log("info", `WHEP ICE state: ${sfuIceState}`);
        if (sfuIceState === "failed") {
          sfuFail("whep_ice", "WHEP ICE connection failed");
        }
      }
    };

    const video = ensureFmp4VideoElement();
    pc.ontrack = (e: RTCTrackEvent) => {
      log("info", `WHEP ontrack: kind=${e.track.kind} id=${e.track.id}`);
      if (e.track.kind === "video") {
        attachSfuVideoListeners(video);
        const eventStream = e.streams.find((stream) => stream.getVideoTracks().includes(e.track));
        const remoteStream = eventStream ?? new MediaStream([e.track]);
        video.srcObject = remoteStream;
        video.style.display = "block";
        video.muted = true;
        video.autoplay = true;
        video.playsInline = true;
        sfuVideoAttachedAt = Date.now();
        sfuRenderAttachFailed = false;
        log("info", `WHEP video attached`);
        video.play()
          .then(() => markSfuVideoLive(video, "whep-play-resolved"))
          .catch((err) => log("err", `WHEP video play() failed: ${err}`));
        sfuActive = true;
        avatarFrame.classList.add("fmp4-active");
      }
    };

    pc.addTransceiver("video", { direction: "recvonly" });

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    log("info", "WHEP: waiting for ICE gathering...");

    if (pc.iceGatheringState !== "complete") {
      await new Promise<void>((resolve) => {
        const check = () => {
          if (sfuPc && sfuPc.iceGatheringState === "complete") {
            sfuPc.removeEventListener("icegatheringstatechange", check);
            resolve();
          }
        };
        sfuPc!.addEventListener("icegatheringstatechange", check);
        setTimeout(resolve, 8000);
      });
    }
    log("info", "WHEP: ICE gathering done, posting offer to WHEP endpoint");

    const localSdp = pc.localDescription!;
    const whepResp = await fetch(whepUrl, {
      method: "POST",
      headers: { "Content-Type": "application/sdp" },
      body: localSdp.sdp,
    });

    if (!whepResp.ok) {
      const text = await whepResp.text();
      sfuFail("whep_offer", `WHEP POST failed: HTTP ${whepResp.status}: ${text.substring(0, 200)}`);
      return;
    }

    const answerSdp = await whepResp.text();
    if (!answerSdp || !answerSdp.includes("v=0")) {
      sfuFail("whep_answer", "WHEP response is not valid SDP");
      return;
    }

    await pc.setRemoteDescription({ type: "answer", sdp: answerSdp });
    log("info", `WHEP: answer set (len=${answerSdp.length}) — waiting for media`);

    if (sfuFirstFrameAt === 0) {
      setStatus("ready", "WHEP connected — waiting for video...");
    }

    sfuStatsInterval = setInterval(updateSfuDiag, 2000);
  } catch (err) {
    sfuFail("startWhepSubscriber", String(err));
  }
}

async function pollForDirectWebrtcOffer(offerUrl: string): Promise<{ sdp: string; type: string; mid: string } | null> {
  // Modal cold start can take 30-40s before the offer is posted. Poll long
  // enough to catch it, then the browser still has Modal's answer timeout
  // window (default 45s) to create and send the answer.
  const deadline = Date.now() + 70000;
  while (Date.now() < deadline) {
    try {
      const resp = await fetch(offerUrl, { method: "GET" });
      if (resp.status === 204) {
        await new Promise((r) => setTimeout(r, 1000));
        continue;
      }
      if (!resp.ok) return null;
      const body = await resp.json() as Record<string, unknown>;
      if (body.sdp && body.type && body.mid) {
        return { sdp: body.sdp as string, type: body.type as string, mid: body.mid as string };
      }
    } catch {
      // ignore and retry
    }
    await new Promise((r) => setTimeout(r, 1000));
  }
  return null;
}

function updateSfuDiag(): void {
  const diag = document.getElementById("sfuDiag");
  if (!diag) return;
  if (!sfuPc) {
    diag.textContent = sfuActive ? "SFU active (PC closed)" : "SFU inactive";
    return;
  }
  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  const dims = video && video.videoWidth ? `${video.videoWidth}x${video.videoHeight}` : "0x0";
  const playing = video ? (!video.paused && (video.currentTime > 0 || video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA)) : false;

  void sfuPc.getStats().then((stats) => {
    const v = Array.from(stats.values()).find(
      (r) => r.type === "inbound-rtp" && (r as Record<string, unknown>).kind === "video",
    ) as Record<string, unknown> | undefined;
    const framesDecoded = (v?.framesDecoded as number) ?? null;
    const fps = (v?.framesPerSecond as number) ?? null;
    const packetsLost = (v?.packetsLost as number) ?? null;
    const jitter = (v?.jitter as number) ?? null;
    const now = Date.now();
    if (video && framesDecoded !== null) {
      if (framesDecoded > sfuLastFramesDecoded) {
        sfuLastFramesDecoded = framesDecoded;
      }
      markSfuVideoLive(video, "stats");
      const decodedButNotAttached = framesDecoded >= 30 && video.videoWidth === 0 && video.videoHeight === 0;
      const attachAgeMs = sfuVideoAttachedAt ? Date.now() - sfuVideoAttachedAt : 0;
      if (decodedButNotAttached && attachAgeMs > 8000 && !sfuRenderAttachFailed) {
        sfuRenderAttachFailed = true;
        sendTelemetry("sfu_render_attach_failed", {
          framesDecoded,
          attachAgeMs,
          iceState: sfuIceState,
          hasSrcObject: Boolean(video.srcObject),
          readyState: video.readyState,
          paused: video.paused,
        });
        sfuFail("render_attach", `WebRTC decoded ${framesDecoded} frames but video element stayed 0x0 after ${Math.round(attachAgeMs)}ms`);
        return;
      }
    }
    if (video && framesDecoded !== null) {
      const currentTime = Number.isFinite(video.currentTime) ? video.currentTime : 0;
      const lost = packetsLost ?? 0;
      if (sfuStatsLogLastAt === 0) {
        sfuStatsLogLastAt = now;
        sfuStatsLogLastCurrentTime = currentTime;
        sfuStatsLogLastFramesDecoded = framesDecoded;
        sfuStatsLogLastPacketsLost = lost;
      }
      const elapsedMs = now - sfuStatsLogLastAt;
      const frameDelta = framesDecoded - sfuStatsLogLastFramesDecoded;
      const lossDelta = lost - sfuStatsLogLastPacketsLost;
      const timeDelta = currentTime - sfuStatsLogLastCurrentTime;
      if (elapsedMs >= 10000) {
        const summary = `SFU stats: ${dims} ${playing ? "playing" : "waiting"} fps=${fps !== null ? fps.toFixed(1) : "—"} decoded+${frameDelta} lost+${lossDelta} jitter=${jitter !== null ? jitter.toFixed(3) : "—"} videoTime+${timeDelta.toFixed(2)}s readyState=${video.readyState}`;
        log("info", summary);
        sfuStatsLogLastAt = now;
        sfuStatsLogLastCurrentTime = currentTime;
        sfuStatsLogLastFramesDecoded = framesDecoded;
        sfuStatsLogLastPacketsLost = lost;
      }
      const poorFps = fps !== null && fps < 20;
      const highLoss = elapsedMs >= 5000 && lossDelta >= 15;
      const playbackStall = elapsedMs >= 5000 && frameDelta >= 60 && timeDelta < 1;
      const hasAnomaly = poorFps || highLoss || playbackStall;
      const shouldEmitSummary = now - sfuStatsTelemetryLastAt >= 10000;
      const shouldEmitAnomaly = hasAnomaly && now - sfuStatsAnomalyLastAt >= 10000;
      if (shouldEmitSummary || shouldEmitAnomaly) {
        sfuStatsTelemetryLastAt = now;
        if (shouldEmitAnomaly) sfuStatsAnomalyLastAt = now;
        sendTelemetry("sfu_browser_stats", {
          width: video.videoWidth,
          height: video.videoHeight,
          playing,
          readyState: video.readyState,
          currentTime,
          framesDecoded,
          fps,
          packetsLost,
          jitter,
          frameDelta,
          lossDelta,
          timeDelta,
          iceState: sfuIceState,
          poorFps,
          highLoss,
          playbackStall,
        });
      }
      if (shouldEmitAnomaly) {
        log("err", `SFU playback anomaly: poorFps=${poorFps} highLoss=${highLoss} playbackStall=${playbackStall} fps=${fps !== null ? fps.toFixed(1) : "—"} lost+${lossDelta} decoded+${frameDelta} videoTime+${timeDelta.toFixed(2)}s`);
      }
    }
    diag.textContent = [
      `transport=webrtc_sfu`,
      `beta_token=${SFU_BETA_TOKEN ? "present" : "missing"}`,
      `ice=${sfuIceState}`,
      `${dims}`,
      playing ? "playing" : "waiting",
      `decoded=${framesDecoded ?? "—"}`,
      `fps=${fps !== null ? fps.toFixed(1) : "—"}`,
      `lost=${packetsLost ?? "—"}`,
      `jitter=${jitter !== null ? jitter.toFixed(3) : "—"}`,
      `fMP4_fallback=NO`,
    ].join(" | ");
  });
}

/**
 * Phase 1 of teardown: mark lifecycle as "ending".
 * Stops accepting new segments but does NOT destroy the media pipeline yet.
 * The actual MediaSource/video teardown happens in fmp4Destroy() after
 * session.ended / ws.close confirms the server has stopped.
 *
 * Called from: stop() button click, before sending session.end.
 */
function fmp4BeginEnding(reason: string): void {
  if (fmp4Lifecycle === "ending" || fmp4Lifecycle === "destroyed" || fmp4Lifecycle === "error") return;
  if (fmp4Lifecycle === "idle") return;
  log("info", `fMP4 begin ending: reason=${reason}`);
  fmp4Lifecycle = "ending";

  // Stop timers that could trigger more appends or seeks
  if (fmp4LiveEdgeCheckInterval !== null) {
    clearInterval(fmp4LiveEdgeCheckInterval);
    fmp4LiveEdgeCheckInterval = null;
  }
  if (fmp4BufferHealthInterval !== null) {
    clearInterval(fmp4BufferHealthInterval);
    fmp4BufferHealthInterval = null;
  }

  // Clear the segment queue so no more data gets appended
  const droppedQueued = fmp4SegmentQueue.length;
  fmp4SegmentQueue = [];
  fmp4PendingInitData = null;

  if (droppedQueued > 0) {
    log("info", `fMP4 ending: dropped ${droppedQueued} queued segments`);
  }
}

/**
 * Phase 2 of teardown: destroy the media pipeline.
 * Idempotent — safe to call from session.ended, ws.close, ws.error, retry.
 * Performs the actual MediaSource detach, video.src clear, and load().
 * Only runs once; subsequent calls are no-ops.
 */
function fmp4Destroy(reason: string): void {
  if (fmp4Lifecycle === "destroyed" || fmp4Lifecycle === "error") return;
  log("info", `fMP4 destroy: reason=${reason} prevState=${fmp4Lifecycle}`);
  fmp4Lifecycle = "destroyed";

  // Stop timers
  if (fmp4LiveEdgeCheckInterval !== null) {
    clearInterval(fmp4LiveEdgeCheckInterval);
    fmp4LiveEdgeCheckInterval = null;
  }
  if (fmp4BufferHealthInterval !== null) {
    clearInterval(fmp4BufferHealthInterval);
    fmp4BufferHealthInterval = null;
  }

  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;

  if (video) {
    try { video.pause(); } catch (_) { /* ignore */ }
  }

  if (fmp4SourceBuffer) {
    try { fmp4SourceBuffer.abort(); } catch (_) { /* ignore */ }
  }

  if (fmp4MediaSource) {
    try {
      if (fmp4MediaSource.readyState === "open" && (!fmp4SourceBuffer || !fmp4SourceBuffer.updating)) {
        fmp4MediaSource.endOfStream();
      }
    } catch (_) { /* ignore */ }
  }

  // Detach media element — this may fire async error events which
  // are suppressed by the lifecycle state check in the error handler.
  if (video) {
    try { video.removeAttribute("src"); } catch (_) { /* ignore */ }
    try { video.src = ""; } catch (_) { /* ignore */ }
    try { video.load(); } catch (_) { /* ignore */ }
  }

  // Reset all fMP4 state
  fmp4InitSeen = false;
  fmp4MseReady = false;
  fmp4FatalError = false;
  fmp4MediaSource = null;
  fmp4SourceBuffer = null;
  fmp4SegmentQueue = [];
  fmp4IsAppending = false;
  fmp4IsTrimming = false;
  fmp4SegmentCount = 0;
  if (video) {
    video.style.display = "none";
  }
  avatarFrame.classList.remove("fmp4-active");
  fmp4ModeExpected = false;
  fmp4DroppedFatalCount = 0;
  fmp4DroppedFatalLastTelemetryAt = 0;
  fmp4ListenersAttached = false;
  fmp4CurrentAppendByteLen = null;
  fmp4LastTimeupdateTelemetryAt = 0;
  fmp4AvMode = false;
  fmp4SpeechActive = false;
  fmp4LastSpeechEndedAt = 0;
  fmp4SeekBlockedUntil = 0;
  hideAvPlayOverlay();
  if (fmp4SpeechPlaybackAckTimerId !== null) {
    clearTimeout(fmp4SpeechPlaybackAckTimerId);
    fmp4SpeechPlaybackAckTimerId = null;
  }
  fmp4SummaryQueuedCount = 0;
  fmp4SummaryAppendedCount = 0;
  fmp4SummaryTotalBytes = 0;
  fmp4SummaryLastSentAt = 0;
  fmp4SessionReadyAt = 0;
  fmp4FirstPlayingAt = 0;
  fmp4WaitingCount = 0;
  fmp4StalledCount = 0;
  fmp4SegmentsReceived = 0;
  fmp4SegmentsQueued = 0;
  fmp4SegmentsAppended = 0;
  fmp4AppendErrors = 0;
  fmp4LastReceivedSequence = null;
  fmp4LastAppendedSequence = null;
  fmp4LastAppendedAt = 0;
  fmp4PendingInitData = null;
  fmp4MseEagerCreatedAt = 0;
  fmp4SourceopenAt = 0;
  fmp4InitAppendedAt = 0;
  if (currentLoadingStage !== "error") {
    const overlay = document.getElementById("avatarLoadingOverlay");
    if (overlay) overlay.remove();
    placeholderShownAt = 0;
    currentLoadingStage = null;
  }
}

function fmp4MaybeSendSummary(): void {
  const now = Date.now();
  const totalSinceLast = fmp4SummaryQueuedCount + fmp4SummaryAppendedCount;
  const elapsedSinceLast = now - fmp4SummaryLastSentAt;
  if (elapsedSinceLast < FMP4_SUMMARY_INTERVAL_MS && totalSinceLast < FMP4_SUMMARY_MIN_SEGMENTS) return;

  const video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  const buf = video ? getVideoBufferInfo(video) : emptyVideoBufferInfo;
  sendTelemetry("avatar_fmp4_segment_summary", {
    queuedCount: fmp4SummaryQueuedCount,
    appendedCount: fmp4SummaryAppendedCount,
    avgByteLength: fmp4SummaryAppendedCount > 0
      ? Math.round(fmp4SummaryTotalBytes / fmp4SummaryAppendedCount)
      : 0,
    queueLength: fmp4SegmentQueue.length,
    currentTime: video?.currentTime?.toFixed(3) ?? "0",
    bufferedEnd: buf.bufferedEnd?.toFixed(3) ?? "0",
    mode: fmp4AvMode ? "av" : "video",
    totalSegments: fmp4SegmentCount,
    waitingCount: fmp4WaitingCount,
    stalledCount: fmp4StalledCount,
    receivedCount: fmp4SegmentsReceived,
    totalQueuedCount: fmp4SegmentsQueued,
    totalAppendedCount: fmp4SegmentsAppended,
    appendErrors: fmp4AppendErrors,
    lastReceivedSequence: fmp4LastReceivedSequence,
    lastAppendedAt: fmp4LastAppendedAt || undefined,
  });
  fmp4SummaryQueuedCount = 0;
  fmp4SummaryAppendedCount = 0;
  fmp4SummaryTotalBytes = 0;
  fmp4SummaryLastSentAt = now;
}

function ensureFmp4VideoElement(): HTMLVideoElement {
  let video = document.getElementById("fmp4Video") as HTMLVideoElement | null;
  if (!video) {
    video = document.createElement("video");
    video.id = "fmp4Video";
    video.autoplay = true;
    video.playsInline = true;
    video.style.display = "none";
    video.style.width = "100%";
    video.style.borderRadius = "12px";
    video.style.background = "#000";
    avatarFrame.appendChild(video);
  }
  video.muted = !fmp4AvMode;
  return video;
}

function getVideoBufferInfo(video: HTMLVideoElement): VideoBufferInfo {
  let bufferedStart = 0;
  let bufferedEnd = 0;
  let bufferAheadMs = 0;
  try {
    if (video.buffered && video.buffered.length > 0) {
      bufferedStart = video.buffered.start(0);
      bufferedEnd = video.buffered.end(video.buffered.length - 1);
      bufferAheadMs = Math.round((bufferedEnd - video.currentTime) * 1000);
    }
  } catch {}
  return { bufferedStart, bufferedEnd, bufferAheadMs };
}

function attachFmp4VideoListeners(video: HTMLVideoElement): void {
  if (fmp4ListenersAttached) return;
  fmp4ListenersAttached = true;

  const emit = (eventName: string) => {
    const buf = getVideoBufferInfo(video);
    sendTelemetry("avatar_fmp4_video_event", {
      eventName,
      currentTime: video.currentTime,
      paused: video.paused,
      readyState: video.readyState,
      networkState: video.networkState,
      videoWidth: video.videoWidth,
      videoHeight: video.videoHeight,
      ...buf,
      segmentCount: fmp4SegmentCount,
    });
  };

  const events = ["loadedmetadata", "canplay", "playing", "pause", "waiting", "stalled", "seeking", "seeked"];
  for (const ev of events) {
    video.addEventListener(ev, () => {
      if (ev === "waiting") fmp4WaitingCount++;
      if (ev === "stalled") fmp4StalledCount++;
      if (ev === "playing" && fmp4FirstPlayingAt === 0 && fmp4SessionReadyAt > 0) {
        fmp4FirstPlayingAt = Date.now();
        const placeholderDurationMs = placeholderShownAt ? Date.now() - placeholderShownAt : undefined;
        setLoadingStage("live");
        hideAvatarLoadingOverlay();
        sendTelemetry("avatar_fmp4_first_frame", {
          timeSinceSessionReadyMs: fmp4FirstPlayingAt - fmp4SessionReadyAt,
          timeSinceConnectClickMs: sessionStartTime ? fmp4FirstPlayingAt - sessionStartTime : undefined,
          segmentCount: fmp4SegmentCount,
        });
        if (placeholderDurationMs !== undefined) {
          sendTelemetry("first_live_frame_after_placeholder_ms", {
            durationMs: placeholderDurationMs,
            sessionId: currentSessionId,
          });
        }
      }
      log("info", `fMP4 video event: ${ev} currentTime=${video.currentTime.toFixed(2)} readyState=${video.readyState} paused=${video.paused} w=${video.videoWidth} h=${video.videoHeight}`);
      emit(ev);
    });
  }

  video.addEventListener("timeupdate", () => {
    const now = Date.now();
    if (now - fmp4LastTimeupdateTelemetryAt < 3000) return;
    fmp4LastTimeupdateTelemetryAt = now;
    emit("timeupdate");
  });

  video.addEventListener("error", () => {
    // Suppress media errors during teardown states — these are caused by
    // src clearing / load() in fmp4Destroy(), not real playback errors.
    // Real errors in "starting" / "playing" states are still logged.
    if (fmp4Lifecycle === "ending" || fmp4Lifecycle === "destroyed" || fmp4Lifecycle === "error") return;
    const err = video.error;
    log("err", `fMP4 video error: code=${err?.code} message=${err?.message}`);
    sendTelemetry("avatar_fmp4_video_event", {
      eventName: "error",
      errorCode: err?.code,
      errorMessage: err?.message,
      currentTime: video.currentTime,
      readyState: video.readyState,
      networkState: video.networkState,
      segmentCount: fmp4SegmentCount,
    });
  });

  log("info", "fMP4 video diagnostics listeners attached");
}

function prepareFmp4Surface(): void {
  // New session: increment generation so stale events from the
  // previous session are ignored by lifecycle-guarded callbacks.
  fmp4Generation++;
  fmp4Lifecycle = "starting";
  const children = Array.from(avatarFrame.children);
  for (const child of children) {
    if (child instanceof HTMLSpanElement) {
      avatarFrame.removeChild(child);
    } else if (child instanceof HTMLImageElement) {
      avatarFrame.removeChild(child);
    }
  }
  const video = ensureFmp4VideoElement();
  video.style.display = "block";
  avatarFrame.classList.add("fmp4-active");
  attachFmp4VideoListeners(video);

  const mimeCodec = fmp4AvMode
    ? 'video/mp4; codecs="avc1.42E01F,mp4a.40.2"'
    : 'video/mp4; codecs="avc1.42E01F"';

  if (!("MediaSource" in window)) {
    log("err", "MSE not supported in this browser");
    fmp4FatalError = true;
    sendTelemetry("avatar_fmp4_mse_unsupported", { reason: "no_MediaSource" });
    showStartupError("Browser does not support MediaSource");
    return;
  }
  if (!MediaSource.isTypeSupported(mimeCodec)) {
    log("err", `MSE codec not supported: ${mimeCodec}`);
    fmp4FatalError = true;
    sendTelemetry("avatar_fmp4_mse_unsupported", { reason: "codec_not_supported", mimeCodec });
    showStartupError("Browser does not support required video codec");
    return;
  }

  try {
    fmp4MseEagerCreatedAt = Date.now();
    fmp4MediaSource = new MediaSource();
    video.src = URL.createObjectURL(fmp4MediaSource);

    const gen = fmp4Generation;
    fmp4MediaSource.addEventListener("sourceopen", () => {
      if (gen !== fmp4Generation) return;
      fmp4SourceopenAt = Date.now();

      try {
        fmp4SourceBuffer = fmp4MediaSource!.addSourceBuffer(mimeCodec);

        fmp4SourceBuffer.addEventListener("updateend", () => {
          if (fmp4IsTrimming) {
            fmp4IsTrimming = false;
            const v = ensureFmp4VideoElement();
            const buf = getVideoBufferInfo(v);
            sendTelemetry("media_buffer_trimmed", {
              currentTime: v.currentTime,
              bufferedStart: buf.bufferedStart,
              bufferedEnd: buf.bufferedEnd,
              bufferAheadMs: buf.bufferAheadMs,
            });
            fmp4DrainQueue();
            return;
          }

          fmp4IsAppending = false;
          const appendByteLen = fmp4CurrentAppendByteLen;
          fmp4CurrentAppendByteLen = null;

          if (!fmp4MseReady) {
            fmp4MseReady = true;
            if (fmp4Lifecycle === "starting") {
              fmp4Lifecycle = "playing";
            }
            log("info", "fMP4 MSE ready — init append complete");
            sendTelemetry("avatar_fmp4_mse_ready", {
              timeSinceSessionReadyMs: fmp4SessionReadyAt > 0 ? Date.now() - fmp4SessionReadyAt : undefined,
              queuedSegments: fmp4SegmentQueue.length,
            });

            avatarFrame.classList.add("fmp4-active");
            video.style.display = "block";
            video.play().then(() => {
              log("info", `fMP4 video.play() resolved after init (avMode=${fmp4AvMode})`);
              sendTelemetry(fmp4AvMode ? "avatar_fmp4_av_mse_ready" : "avatar_fmp4_video_play_attempt", { stage: "after_init", result: "resolved" });
              hideAvPlayOverlay();
            }).catch((playErr) => {
              log("err", `fMP4 video.play() rejected after init: ${playErr}`);
              if (fmp4AvMode) {
                sendTelemetry("avatar_fmp4_av_play_rejected", { error: String(playErr) });
                showAvPlayOverlay();
              } else {
                sendTelemetry("avatar_fmp4_video_play_attempt", { stage: "after_init", result: "rejected", error: String(playErr) });
              }
            });

            fmp4LiveEdgeCheckInterval = window.setInterval(() => {
              const v = ensureFmp4VideoElement();
              if (v.readyState < 2) return;
              if (!v.buffered || v.buffered.length === 0) return;
              const liveEdge = v.buffered.end(v.buffered.length - 1);
              const behind = liveEdge - v.currentTime;
              const behindMs = Math.round(behind * 1000);

              if (fmp4AvMode) {
                if (behind > 5 && VERBOSE_AV_TELEMETRY) {
                  sendTelemetry("avatar_fmp4_live_edge_seek_suppressed_av_paced", {
                    fmp4AvMode,
                    fmp4SpeechActive,
                    behindMs,
                    currentTime: v.currentTime.toFixed(3),
                    liveEdge: liveEdge.toFixed(3),
                    segmentCount: fmp4SegmentCount,
                  });
                }
                return;
              }

              if (fmp4SpeechActive) {
                if (behind > 5) {
                  sendTelemetry("avatar_fmp4_live_edge_seek_blocked_speech_active", {
                    fmp4AvMode,
                    fmp4SpeechActive,
                    behindMs,
                    currentTime: v.currentTime.toFixed(3),
                    liveEdge: liveEdge.toFixed(3),
                    segmentCount: fmp4SegmentCount,
                  });
                }
                return;
              }

              if (Date.now() < fmp4SeekBlockedUntil) {
                if (behind > 5) {
                  sendTelemetry("avatar_fmp4_live_edge_seek_blocked_post_speech_grace", {
                    fmp4AvMode,
                    behindMs,
                    currentTime: v.currentTime.toFixed(3),
                    liveEdge: liveEdge.toFixed(3),
                    segmentCount: fmp4SegmentCount,
                  });
                }
                return;
              }

              if (behind > 5) {
                const target = liveEdge - 0.5;
                log("info", `fMP4 live edge seek: ${behind.toFixed(2)}s behind, seeking to ${target.toFixed(2)}s`);
                sendTelemetry("avatar_fmp4_live_edge_seek", {
                  fmp4AvMode,
                  fmp4SpeechActive,
                  behindMs,
                  currentTime: v.currentTime.toFixed(3),
                  liveEdge: liveEdge.toFixed(3),
                  targetTime: target.toFixed(3),
                  segmentCount: fmp4SegmentCount,
                });
                v.currentTime = target;
              }
            }, 2000);

            fmp4BufferHealthInterval = window.setInterval(() => {
              const v = ensureFmp4VideoElement();
              const buf = getVideoBufferInfo(v);
              sendTelemetry("media_buffer_health", {
                currentTime: v.currentTime,
                bufferedStart: buf.bufferedStart,
                bufferedEnd: buf.bufferedEnd,
                bufferAheadMs: buf.bufferAheadMs,
                sourceBufferUpdating: fmp4SourceBuffer?.updating ?? false,
                queueLength: fmp4SegmentQueue.length,
                fmp4IsTrimming,
              });
            }, 10000);
          } else {
            if (appendByteLen !== null) {
              fmp4SegmentsAppended++;
              fmp4LastAppendedAt = Date.now();
              if (fmp4SegmentCount % 10 === 0 || VERBOSE_DELIVERY_TELEMETRY) {
                const v = ensureFmp4VideoElement();
                const buf = getVideoBufferInfo(v);
                sendTelemetry("avatar_fmp4_browser_segment_appended", {
                  appendedCount: fmp4SegmentsAppended,
                  receivedCount: fmp4SegmentsReceived,
                  queuedCount: fmp4SegmentsQueued,
                  appendErrors: fmp4AppendErrors,
                  queueLength: fmp4SegmentQueue.length,
                  segmentCount: fmp4SegmentCount,
                  currentTime: v.currentTime?.toFixed(3),
                  bufferedEnd: buf.bufferedEnd?.toFixed(3),
                  bufferAheadMs: buf.bufferAheadMs,
                });
              }
              fmp4MaybeSendSummary();
            }

            if (video.paused) {
              video.play().then(() => {
                log("info", "fMP4 video.play() resolved after segment append (was paused)");
                sendTelemetry("avatar_fmp4_video_play_attempt", { stage: "after_segment", result: "resolved" });
              }).catch((playErr) => {
                log("err", `fMP4 video.play() rejected after segment append: ${playErr}`);
                sendTelemetry("avatar_fmp4_video_play_attempt", { stage: "after_segment", result: "rejected", error: String(playErr) });
              });
            }
          }

          fmp4MaybeTrimBuffer();
          if (!fmp4IsTrimming) {
            fmp4DrainQueue();
          }
        });

        fmp4SourceBuffer.addEventListener("error", (e) => {
          log("err", `SourceBuffer error: ${e}`);
          sendTelemetry("avatar_fmp4_mse_error", { error: "sourcebuffer_error" });
        });

        sendTelemetry("avatar_fmp4_mse_sourceopen", {
          timeFromSessionReadyMs: fmp4SessionReadyAt > 0 ? fmp4SourceopenAt - fmp4SessionReadyAt : undefined,
          timeFromEagerCreateMs: fmp4SourceopenAt - fmp4MseEagerCreatedAt,
          hasPendingInit: fmp4PendingInitData !== null,
          queuedSegments: fmp4SegmentQueue.length,
        });

        if (fmp4PendingInitData) {
          const init = fmp4PendingInitData;
          fmp4PendingInitData = null;
          const initData = base64ToArrayBuffer(init.dataBase64);
          fmp4SourceBuffer.appendBuffer(initData);
          fmp4IsAppending = true;
          fmp4InitAppendedAt = Date.now();
          log("info", `fMP4 init append started (eager sourceopen, ${init.byteLength} bytes, timeToInit=${init.timingMs}ms, queued=${fmp4SegmentQueue.length})`);
          sendTelemetry("avatar_fmp4_init_appended", {
            stage: "sourceopen_pending",
            byteLength: init.byteLength,
            timeToInitMs: init.timingMs,
            timeFromInitReceivedMs: fmp4InitAppendedAt - init.receivedAt,
            timeFromSessionReadyMs: fmp4SessionReadyAt > 0 ? fmp4InitAppendedAt - fmp4SessionReadyAt : undefined,
            queuedSegments: fmp4SegmentQueue.length,
          });
        }
      } catch (sourceErr) {
        log("err", `fMP4 sourceopen handler failed: ${sourceErr}`);
        fmp4FatalError = true;
        sendTelemetry("avatar_fmp4_mse_error", { error: String(sourceErr), stage: "sourceopen" });
        showStartupError("Video player initialization failed");
      }
    }, { once: true });

    sendTelemetry("avatar_fmp4_mse_eager_opened", {
      mimeCodec,
      timeFromSessionReadyMs: fmp4MseEagerCreatedAt - fmp4SessionReadyAt,
    });
    log("info", `fMP4 MSE eagerly created at session.ready — codec=${mimeCodec}`);
  } catch (mseErr) {
    log("err", `fMP4 eager MSE setup failed: ${mseErr}`);
    fmp4FatalError = true;
    sendTelemetry("avatar_fmp4_mse_error", { error: String(mseErr), stage: "eager_setup" });
    showStartupError("Video player setup failed");
  }

  log("info", "fMP4 surface prepared — placeholder/img removed, video visible, MSE eager");
}

function sendTelemetry(
  event: string,
  properties?: Record<string, unknown>,
): void {
  const entry = { event, properties };
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    telemetryQueue.push(entry);
    return;
  }
  ws.send(
    JSON.stringify({
      type: "client.telemetry",
      event,
      elapsedMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
      properties,
    }),
  );
}

function flushTelemetryQueue(): void {
  while (telemetryQueue.length > 0) {
    const entry = telemetryQueue.shift()!;
    ws!.send(
      JSON.stringify({
        type: "client.telemetry",
        event: entry.event,
        elapsedMs: sessionStartTime
          ? Date.now() - sessionStartTime
          : undefined,
        properties: entry.properties,
      }),
    );
  }
}

function setStatus(status: Status, detail?: string) {
  statusBar.className = `status-bar status-${status}`;
  statusText.textContent = detail ?? status;
}

function log(direction: "send" | "recv" | "err" | "info", msg: string) {
  const now = new Date();
  const ts = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}.${now.getMilliseconds().toString().padStart(3, "0")}`;
  const elapsed = sessionStartTime
    ? ` +${(Date.now() - sessionStartTime).toString().padStart(5)}ms`
    : "";

  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML =
    `<span class="log-time">${ts}${elapsed}</span>` +
    `<span class="log-dir-${direction}">${direction.toUpperCase().padEnd(4)}</span> ` +
    `<span>${escapeHtml(msg)}</span>`;
  logOutput.appendChild(entry);
  logOutput.scrollTop = logOutput.scrollHeight;
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function setButtons(connected: boolean) {
  connectBtn.disabled = connected;
  stopBtn.disabled = !connected;
  workerUrlInput.disabled = connected;
  startMicBtn.disabled = !connected || micStream !== null;
  stopMicBtn.disabled = !connected || micStream === null;
}

function showAvatarPlaceholder(stage: LoadingStage = "connecting"): void {
  hideAvatarLoadingOverlay();

  const overlay = document.createElement("div");
  overlay.className = "avatar-loading-overlay";
  overlay.id = "avatarLoadingOverlay";

  const img = document.createElement("img");
  img.className = "avatar-placeholder-img";
  img.src = "assets/avatar-placeholder.jpg";
  img.alt = "";

  const content = document.createElement("div");
  content.className = "avatar-loading-content";

  const spinner = document.createElement("div");
  spinner.className = "avatar-loading-spinner";

  const label = document.createElement("div");
  label.className = "avatar-loading-label";
  label.textContent = LOADING_LABELS[stage];

  content.appendChild(spinner);
  content.appendChild(label);
  overlay.appendChild(img);
  overlay.appendChild(content);
  avatarFrame.appendChild(overlay);

  placeholderShownAt = Date.now();
  currentLoadingStage = stage;
  setStatus(LOADING_STATUS_MAP[stage], LOADING_LABELS[stage]);
  sendTelemetry("placeholder_shown", { stage, sessionId: currentSessionId });
  sendTelemetry("loading_state_changed", {
    stage,
    sessionId: currentSessionId,
    elapsedMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
  });
}

function hideAvatarLoadingOverlay(): void {
  const overlay = document.getElementById("avatarLoadingOverlay");
  if (!overlay) return;

  overlay.classList.add("avatar-loading-fade-out");
  const elapsedMs = placeholderShownAt ? Date.now() - placeholderShownAt : undefined;
  sendTelemetry("placeholder_hidden", {
    sessionId: currentSessionId,
    durationMs: elapsedMs,
  });

  setTimeout(() => {
    const el = document.getElementById("avatarLoadingOverlay");
    if (el) el.remove();
  }, 400);
}

function setLoadingStage(stage: LoadingStage): void {
  if (stage !== "live") {
    const existing = document.getElementById("avatarLoadingOverlay");
    if (!existing) {
      showAvatarPlaceholder(stage);
      return;
    }
  }

  currentLoadingStage = stage;
  setStatus(LOADING_STATUS_MAP[stage], LOADING_LABELS[stage]);

  const overlay = document.getElementById("avatarLoadingOverlay");
  if (overlay) {
    const label = overlay.querySelector(".avatar-loading-label");
    if (label) label.textContent = LOADING_LABELS[stage];
  }

  sendTelemetry("loading_state_changed", {
    stage,
    sessionId: currentSessionId,
    elapsedMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
  });
}

function showStartupError(message: string): void {
  const overlay = document.getElementById("avatarLoadingOverlay");
  if (!overlay) {
    showAvatarPlaceholder("error");
  }

  const el = document.getElementById("avatarLoadingOverlay");
  if (el) {
    el.classList.add("avatar-loading-error");
    const spinner = el.querySelector(".avatar-loading-spinner");
    if (spinner) spinner.remove();
    const label = el.querySelector(".avatar-loading-label");
    if (label) label.textContent = message;

    const existingBtn = el.querySelector(".avatar-retry-btn");
    if (!existingBtn) {
      const content = el.querySelector(".avatar-loading-content");
      if (content) {
        const retryBtn = document.createElement("button");
        retryBtn.className = "avatar-retry-btn";
        retryBtn.textContent = "Try again";
        retryBtn.addEventListener("click", () => {
          retryStartup();
        });
        content.appendChild(retryBtn);
      }
    }
  }

  currentLoadingStage = "error";
  setStatus("error", message);
  sendTelemetry("startup_error_visible", {
    message,
    stage: currentLoadingStage,
    sessionId: currentSessionId,
  });
}

function retryStartup(): void {
  log("info", "Retry requested — cleaning up failed session");

  if (ws) {
    try {
      if (ws.readyState === WebSocket.OPEN) {
        const msg = JSON.stringify({ type: "session.end" });
        ws.send(msg);
        log("send", msg);
      }
      ws.onclose = null;
      ws.onerror = null;
      ws.onmessage = null;
      ws.onopen = null;
      ws.close();
    } catch {}
    ws = null;
  }

  currentLoadingStage = null;
  sfuCleanup();
  fmp4Destroy("retry_startup");
  stopMicLocal();
  stopIdleAnimation();
  stopSpeechPlaybackAll();
  stopIdleDiagInterval();
  flushIdleChunkDiag();
  avatarSpeechActive = false;
  idleFrameImages = [];
  idleReady = false;
  idleIsLive = false;
  currentSpeechTurnId = 0;
  activeSpeechTurnId = null;
  fallbackPlayedTurnIds.clear();
  if (avatarFirstChunkTimeoutId !== null) {
    clearTimeout(avatarFirstChunkTimeoutId);
    avatarFirstChunkTimeoutId = null;
  }
  speechStreamEnded = false;
  speechExpectedTotalFrames = 0;
  speechDisplayedFrames = 0;
  speechUnderrunCount = 0;
  speechAudioEnded = false;
  speechVideoEnded = false;
  speechAckSentForTurnId = null;
  speechPlaybackTurnId = null;
  speechAudioPlayback = null;

  avatarFrame.innerHTML = '<span class="placeholder">No frame yet</span>';
  frameMeta.textContent = "";
  placeholderShownAt = 0;
  setButtons(false);
  setStatus("disconnected", "disconnected");
}

function resetAvatar() {
  if (fmp4ModeExpected || fmp4InitSeen || fmp4MseReady) {
    frameMeta.textContent = "fMP4 video mode";
    return;
  }
  avatarFrame.innerHTML = '<span class="placeholder">No frame yet</span>';
  frameMeta.textContent = "";
}

function updateAudioStatsDisplay() {
  const chunks = micChunksSent;
  const bytes = micBytesSent;
  const kb = (bytes / 1024).toFixed(1);
  const seconds = bytes > 0 ? (bytes / 2 / 16000).toFixed(2) : "0.00";
  const avgRms = audioRmsCount > 0 ? (audioRmsSum / audioRmsCount).toFixed(4) : "0";
  const peakMax = audioPeakMax.toFixed(3);
  const nearZeroAvg = audioRmsCount > 0 ? (audioNearZeroSum / audioRmsCount).toFixed(3) : "0";
  const clippedAvg = audioRmsCount > 0 ? (audioClippedSum / audioRmsCount).toFixed(3) : "0";

  const rollingAvgRms = rollingBuffer.length > 0
    ? (rollingBuffer.reduce((s, m) => s + m.rms, 0) / rollingBuffer.length).toFixed(4) : "0";
  const rollingMaxPeak = rollingBuffer.length > 0
    ? Math.max(...rollingBuffer.map(m => m.peak)).toFixed(3) : "0";
  const rollingNearZeroAvg = rollingBuffer.length > 0
    ? (rollingBuffer.reduce((s, m) => s + m.nearZeroRatio, 0) / rollingBuffer.length).toFixed(3) : "0";
  const rollingClippedAvg = rollingBuffer.length > 0
    ? (rollingBuffer.reduce((s, m) => s + m.clippedRatio, 0) / rollingBuffer.length).toFixed(3) : "0";

  audioStats.textContent = `chunks: ${chunks} | bytes: ${kb}KB | audio: ~${seconds}s | rms: ${avgRms} | peak: ${peakMax} | silent: ${nearZeroAvg} | clipped: ${clippedAvg} | Rolling [last ${ROLLING_WINDOW}]: rms=${rollingAvgRms} peak=${rollingMaxPeak} silent=${rollingNearZeroAvg} clipped=${rollingClippedAvg}`;
}

function setSpeechStatus(text: string, color: string) {
  speechStatus.textContent = text;
  speechStatus.style.color = color;
}

let firstAudioChunkCaptured = false;

async function connect() {
  const workerUrl = workerUrlInput.value.trim().replace(/\/+$/, "");
  if (!workerUrl) {
    setStatus("error", "Worker URL is empty");
    return;
  }

  setButtons(true);
  resetAvatar();
  logOutput.innerHTML = "";
  sessionStartTime = Date.now();
  setSpeechStatus("", "");
  telemetryQueue.length = 0;
  firstAudioChunkCaptured = false;
  avatarSpeechActive = false;
  idleReady = false;
  idleIsLive = false;
  idleFrameImages = [];
  currentSpeechTurnId = 0;
  activeSpeechTurnId = null;
  fallbackPlayedTurnIds.clear();
  if (avatarFirstChunkTimeoutId !== null) {
    clearTimeout(avatarFirstChunkTimeoutId);
    avatarFirstChunkTimeoutId = null;
  }
  idleUnderrunCountSinceReport = 0;
  idleUnderrunLastReportAt = 0;
  idleChunkDiag = { count: 0, totalImageCreateMs: 0, totalImageDecodeMs: 0, totalBase64Chars: 0, totalPayloadChars: 0, lastReportAt: 0 };
  speechTimingAssistantAudioAt = 0;
  speechTimingRenderStartAt = 0;
  speechTimingSpeechStartAt = 0;
  speechTimingFirstChunkAt = 0;
  speechTimingPlaybackStartAt = 0;
  speechHadChunksBeforeFallback = false;
  pendingAudioSupersedeState = null;
  speechStreamEnded = false;
  speechExpectedTotalFrames = 0;
  speechDisplayedFrames = 0;
  speechUnderrunCount = 0;
  speechLastUnderrunReportAt = 0;
  speechAudioEnded = false;
  speechVideoEnded = false;
  speechAckSentForTurnId = null;
  speechPlaybackTurnId = null;
  speechAudioPlayback = null;
  speechClockDriftLastReportAt = 0;

  sendTelemetry("connect_clicked", { workerUrl });

  showAvatarPlaceholder("connecting");
  log("info", `POST ${workerUrl}/sessions`);
  if (SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE) {
    const modeLabel = DIRECT_WEBRTC_DEBUG_MODE ? "Direct WebRTC debug" : "SFU";
    log("info", `${modeLabel} beta token present for session create: ${SFU_BETA_TOKEN ? "YES" : "NO"}`);
    sendTelemetry("sfu_beta_token_check", {
      sfuBetaTokenPresent: SFU_BETA_TOKEN !== null,
      sfuBetaTokenDetected: SFU_BETA_TOKEN_DETECTED,
      sfuBetaTokenScrubbed: SFU_BETA_TOKEN_SCRUBBED,
    });
  }

  let sessionRes: { sessionId: string; wsUrl: string };
  try {
    const createUrl = new URL(`${workerUrl}/sessions`);
    if (SFU_BETA_TOKEN) {
      createUrl.searchParams.set("sfu_beta_token", SFU_BETA_TOKEN);
    }
    const res = await fetch(createUrl.toString(), { method: "POST" });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`HTTP ${res.status}: ${text}`);
    }
    sessionRes = await res.json();
  } catch (err) {
    showStartupError("Session creation failed");
    log("err", err instanceof Error ? err.message : String(err));
    sendTelemetry("client_error", {
      context: "session_create",
      error: err instanceof Error ? err.message : String(err),
    });
    setButtons(false);
    return;
  }

  currentSessionId = sessionRes.sessionId;
  log("recv", `Session created: ${sessionRes.sessionId}`);
  log("info", `wsUrl: ${sessionRes.wsUrl}`);

  sendTelemetry("session_created", { sessionId: currentSessionId });

  setStatus("connecting", "Connecting WebSocket...");
  log("info", `Connecting WebSocket...`);

  ws = new WebSocket(sessionRes.wsUrl);
  ws.binaryType = "arraybuffer";

  ws.addEventListener("open", () => {
    log("info", "WebSocket OPEN");

    setLoadingStage("starting");
    sendTelemetry("websocket_open");
    flushTelemetryQueue();

    let startTransport: "webrtc_sfu" | "direct_webrtc_debug" | "whep" | undefined;
    if (SFU_MODE) startTransport = "webrtc_sfu";
    else if (DIRECT_WEBRTC_DEBUG_MODE) startTransport = "direct_webrtc_debug";
    else if (WHEP_MODE) startTransport = "whep";
    const pionDebug = DIRECT_WEBRTC_DEBUG_MODE && new URLSearchParams(window.location.search).get("pion") === "1";
    const startMsg: Record<string, unknown> = startTransport
      ? { type: "session.start", transport: startTransport }
      : { type: "session.start" };
    if (pionDebug) startMsg.pionDebug = true;
    if (WHEP_MODE) {
      const whepUrlParam = new URLSearchParams(window.location.search).get("whep_url");
      if (whepUrlParam) startMsg.whepUrl = whepUrlParam;
    }
    const msg = JSON.stringify(startMsg);
    ws!.send(msg);
    log("send", msg);

    sendTelemetry("session_start_sent");
  });

  ws.addEventListener("message", (event: MessageEvent) => {
    if (typeof event.data === "string") {
      const data = JSON.parse(event.data);
      handleMessage(data);
    }
  });

  ws.addEventListener("close", (event: CloseEvent) => {
    log("info", `WebSocket CLOSED code=${event.code} reason=${event.reason}`);
    const wasLoading = currentLoadingStage !== null && currentLoadingStage !== "live" && currentLoadingStage !== "error";
    sendTelemetry("websocket_closed", {
      code: event.code,
      reason: event.reason,
    });
    ws = null;
    stopMicLocal();
    stopIdleAnimation();
    stopSpeechPlaybackAll();
    stopIdleDiagInterval();
    flushIdleChunkDiag();
    avatarSpeechActive = false;
    idleFrameImages = [];
    idleReady = false;
    idleIsLive = false;
    currentSpeechTurnId = 0;
    activeSpeechTurnId = null;
    fallbackPlayedTurnIds.clear();
    if (avatarFirstChunkTimeoutId !== null) {
      clearTimeout(avatarFirstChunkTimeoutId);
      avatarFirstChunkTimeoutId = null;
    }
    idleUnderrunCountSinceReport = 0;
    idleUnderrunLastReportAt = 0;
    speechTimingAssistantAudioAt = 0;
    speechTimingRenderStartAt = 0;
    speechTimingSpeechStartAt = 0;
    speechTimingFirstChunkAt = 0;
    speechTimingPlaybackStartAt = 0;
    speechStreamEnded = false;
    speechExpectedTotalFrames = 0;
    speechDisplayedFrames = 0;
    speechUnderrunCount = 0;
    speechLastUnderrunReportAt = 0;
    speechAudioEnded = false;
    speechVideoEnded = false;
    speechAckSentForTurnId = null;
    speechPlaybackTurnId = null;
    speechAudioPlayback = null;
    speechClockDriftLastReportAt = 0;
    setButtons(false);
    sfuCleanup();
    fmp4Destroy("ws_closed");
    if (wasLoading) {
      showStartupError("Connection closed");
    } else if (statusBar.className.includes("status-rendering") || statusBar.className.includes("status-ready") || statusBar.className.includes("status-mic_active")) {
      setStatus("stopped", "disconnected");
    }
  });

  ws.addEventListener("error", () => {
    log("err", "WebSocket error");
    sendTelemetry("client_error", { context: "websocket" });
    showStartupError("Connection failed");
    ws = null;
    stopMicLocal();
    stopIdleAnimation();
    stopSpeechPlaybackAll();
    stopIdleDiagInterval();
    flushIdleChunkDiag();
    avatarSpeechActive = false;
    idleFrameImages = [];
    idleReady = false;
    idleIsLive = false;
    currentSpeechTurnId = 0;
    activeSpeechTurnId = null;
    fallbackPlayedTurnIds.clear();
    if (avatarFirstChunkTimeoutId !== null) {
      clearTimeout(avatarFirstChunkTimeoutId);
      avatarFirstChunkTimeoutId = null;
    }
    idleUnderrunCountSinceReport = 0;
    idleUnderrunLastReportAt = 0;
    speechStreamEnded = false;
    speechExpectedTotalFrames = 0;
    speechDisplayedFrames = 0;
    speechUnderrunCount = 0;
    speechLastUnderrunReportAt = 0;
    speechAudioEnded = false;
    speechVideoEnded = false;
    speechAckSentForTurnId = null;
    speechPlaybackTurnId = null;
    speechAudioPlayback = null;
    speechClockDriftLastReportAt = 0;
    setButtons(false);
    sfuCleanup();
    fmp4Destroy("ws_error");
  });
}

function handleMessage(msg: Record<string, unknown>) {
  const type = msg.type as string;

  switch (type) {
    case "session.ready": {
      const transportStr = (msg as Record<string, unknown>).transport as string | undefined;
      if (transportStr === "webrtc_sfu") {
        const sfuMeta = (msg as Record<string, unknown>).sfu as Record<string, unknown> | undefined;
        log("recv", `session.ready — ${msg.avatarId} ${msg.width}x${msg.height} transport=webrtc_sfu sfu.track=${sfuMeta?.publisher_track_name ?? "?"}`);
        setLoadingStage("preparing");
        sendTelemetry("session_ready_received", {
          avatarId: msg.avatarId,
          transport: "webrtc_sfu",
          timeSinceConnectClickMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
        });

        if (!sfuMeta?.token || !sfuMeta?.subscribe_url || !sfuMeta?.answer_url || !sfuMeta?.close_url) {
          sfuFail("session.ready", "Missing sfu_token or sfu URLs in session.ready message");
          break;
        }

        sfuToken = sfuMeta.token as string;
        sfuSubscribeUrl = sfuMeta.subscribe_url as string;
        sfuAnswerUrl = sfuMeta.answer_url as string;
        sfuCloseUrl = sfuMeta.close_url as string;

        setStatus("ready", "SFU session ready — subscribing...");
        void startSfuSubscriber();
        break;
      }

      if (transportStr === "direct_webrtc_debug") {
        const directMeta = (msg as Record<string, unknown>).directWebrtc as Record<string, unknown> | undefined;
        log("recv", `session.ready — ${msg.avatarId} ${msg.width}x${msg.height} transport=direct_webrtc_debug`);
        setLoadingStage("preparing");
        sendTelemetry("session_ready_received", {
          avatarId: msg.avatarId,
          transport: "direct_webrtc_debug",
          timeSinceConnectClickMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
        });

        if (!directMeta?.offer_url || !directMeta?.answer_url || !Array.isArray(directMeta?.ice_servers)) {
          sfuFail("session.ready", "Missing direct WebRTC URLs or ice_servers in session.ready message");
          break;
        }

        void startDirectWebrtcSubscriber(
          directMeta.offer_url as string,
          directMeta.answer_url as string,
          directMeta.ice_servers as RTCIceServer[],
        );
        break;
      }

      if (transportStr === "whep") {
        const whepMeta = (msg as Record<string, unknown>).whep as Record<string, unknown> | undefined;
        log("recv", `session.ready — ${msg.avatarId} ${msg.width}x${msg.height} transport=whep`);
        setLoadingStage("preparing");
        sendTelemetry("session_ready_received", {
          avatarId: msg.avatarId,
          transport: "whep",
          timeSinceConnectClickMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
        });

        if (!whepMeta?.url) {
          sfuFail("session.ready", "Missing whep.url in session.ready message");
          break;
        }

        void startWhepSubscriber(whepMeta.url as string);
        break;
      }

      const renderMode = (msg as Record<string, unknown>).renderMode as string | undefined;
      const short = `session.ready — ${msg.avatarId} ${msg.width}x${msg.height} warmup=${msg.warmupMs}ms renderMode=${renderMode ?? "unknown"}`;
      log("recv", short);
      setLoadingStage("preparing");

      sendTelemetry("session_ready_received", {
        avatarId: msg.avatarId,
        warmupMs: msg.warmupMs,
        renderMode,
        timeSinceConnectClickMs: sessionStartTime ? Date.now() - sessionStartTime : undefined,
      });

      fmp4SessionReadyAt = Date.now();

      if (renderMode === "fmp4_av" || renderMode === "fmp4_video") {
        fmp4ModeExpected = true;
        fmp4AvMode = renderMode === "fmp4_av";
        prepareFmp4Surface();
        log("info", `Skipping render_debug_frame — fMP4 mode (${renderMode})`);
      } else {
        const renderMsg = JSON.stringify({ type: "render_debug_frame" });
        ws!.send(renderMsg);
        log("send", renderMsg);
        setStatus("rendering", "Rendering debug frame...");

        sendTelemetry("render_debug_frame_sent");
      }
      break;
    }

    case "avatar.frame": {
      if (fmp4ModeExpected || fmp4InitSeen || fmp4MseReady) {
        sendTelemetry("avatar_frame_ignored_fmp4_active", {
          width: msg.width as number,
          height: msg.height as number,
        });
        log("info", "avatar.frame ignored — fMP4 mode active");
        break;
      }

      const b64 = msg.dataBase64 as string;
      const mime = msg.mimeType as string;
      const w = msg.width as number;
      const h = msg.height as number;
      const sizeKb = Math.round(b64.length * 0.75 / 1024);

      log("recv", `avatar.frame — ${mime} ${w}x${h} ~${sizeKb}KB`);

      sendTelemetry("avatar_frame_received", {
        width: w,
        height: h,
        sizeKb,
      });

      const img = document.createElement("img");
      img.src = `data:${mime};base64,${b64}`;
      img.alt = "Avatar debug frame";
      avatarFrame.innerHTML = "";
      avatarFrame.appendChild(img);
      frameMeta.textContent = `${mime} | ${w}×${h} | ~${sizeKb}KB`;
      setStatus("ready", `frame_displayed — ${w}×${h}`);
      break;
    }

    // Phase 7.2: live idle rolling buffer. Browser no longer loops a fixed clip.
    // Live idle chunks arrive continuously from DO, browser displays from rolling buffer.
    // Speech interrupts idle and clears buffer. Browser sends playback ack when done.
    // DO waits for ack before resuming idle (with timer fallback).
    case "avatar.speech.start": {
      const turnId = msg.turnId as number;
      const totalFrames = msg.totalFrames as number;
      const fps = msg.fps as number;

      if (fmp4AvMode) {
        if (activeSpeechTurnId !== null && turnId !== activeSpeechTurnId) {
          sendTelemetry("avatar_speech_start_overtaken_av", {
            previousTurnId: activeSpeechTurnId,
            newTurnId: turnId,
          });
        }
      } else {
        if (activeSpeechTurnId !== null && turnId !== activeSpeechTurnId) {
          log("info", `Ignoring stale avatar.speech.start for turnId=${turnId} (active=${activeSpeechTurnId})`);
          break;
        }

        if (fallbackPlayedTurnIds.has(turnId)) {
          log("info", `Ignoring avatar.speech.start for turnId=${turnId} — fallback already played`);
          break;
        }
      }

      currentSpeechTurnId = turnId;
      speechFps = fps;
      speechFrameImages = [];
      speechAnimationFrameIndex = 0;
      speechRenderedDurationMs = (msg.renderedAudioDurationMs as number) ?? 0;
      speechOriginalDurationMs = (msg.originalAudioDurationMs as number) ?? 0;
      speechPlaybackStarted = false;
      speechTimingSpeechStartAt = Date.now();
      speechStreamEnded = false;
      speechExpectedTotalFrames = totalFrames;
      speechDisplayedFrames = 0;
      speechUnderrunCount = 0;
      speechLastUnderrunReportAt = 0;
      speechAudioEnded = false;
      speechVideoEnded = false;
      speechAckSentForTurnId = null;
      speechPlaybackTurnId = null;
      speechAudioPlayback = null;
      speechClockDriftLastReportAt = 0;
      stopSpeechPlaybackAll();

      if (avatarFirstChunkTimeoutId !== null) {
        clearTimeout(avatarFirstChunkTimeoutId);
        avatarFirstChunkTimeoutId = null;
      }

      log("recv", `avatar.speech.start — turnId=${turnId} totalFrames=${totalFrames} fps=${fps} rendered=${speechRenderedDurationMs}ms original=${speechOriginalDurationMs}ms`);
      sendTelemetry("avatar_speech_start_received", { turnId, totalFrames, fps, renderedAudioDurationMs: speechRenderedDurationMs, originalAudioDurationMs: speechOriginalDurationMs });
      setSpeechStatus(`Rendering avatar speech (${totalFrames} frames)...`, "#79c0ff");
      avatarFrame.classList.add("avatar-speaking");
      break;
    }

    case "avatar.speech.chunk": {
      const turnId = msg.turnId as number;
      const chunkIndex = msg.chunkIndex as number;
      const frames = msg.frames as Array<{ dataBase64: string; mimeType: string; width: number; height: number }>;
      const isLast = msg.isLast as boolean;

      if (turnId !== currentSpeechTurnId) {
        log("info", `Discarding stale avatar.speech.chunk for turnId=${turnId} (current=${currentSpeechTurnId})`);
        break;
      }

      if (speechAckSentForTurnId === turnId) {
        const now = Date.now();
        if (now - speechLastUnderrunReportAt > 5000 || speechLastUnderrunReportAt === 0) {
          sendTelemetry("avatar_speech_chunk_after_ack_ignored", {
            turnId,
            chunkIndex,
            framesInChunk: frames.length,
            isLast,
          });
          speechLastUnderrunReportAt = now;
        }
        break;
      }

      if (fallbackPlayedTurnIds.has(turnId)) {
        if (chunkIndex === 0) {
          sendTelemetry("avatar_speech_late_after_fallback", {
            turnId,
            framesArrived: frames.length,
            chunkIndex,
            timeSinceAssistantAudio: speechTimingAssistantAudioAt ? Date.now() - speechTimingAssistantAudioAt : 0,
          });
        }
        break;
      }

      for (const f of frames) {
        const img = new Image();
        img.src = `data:${f.mimeType};base64,${f.dataBase64}`;
        speechFrameImages.push(img);
      }

      if (isLast) {
        speechStreamEnded = true;
      }

      if (speechTimingFirstChunkAt === 0 && chunkIndex === 0) {
        speechTimingFirstChunkAt = Date.now();
        if (avatarFirstChunkTimeoutId !== null) {
          clearTimeout(avatarFirstChunkTimeoutId);
          avatarFirstChunkTimeoutId = null;
        }
      }

      log("recv", `avatar.speech.chunk — turnId=${turnId} chunk=${chunkIndex} +${frames.length} frames total=${speechFrameImages.length} isLast=${isLast} streamEnded=${speechStreamEnded}`);
      sendTelemetry("avatar_speech_chunk_received", {
        turnId,
        chunkIndex,
        framesInChunk: frames.length,
        totalCollected: speechFrameImages.length,
        isLast,
        speechStreamEnded,
      });

      if (!speechPlaybackStarted && speechFrameImages.length >= getSpeechStartBufferThreshold(speechExpectedTotalFrames)) {
        speechPlaybackStarted = true;
        startAvatarSpeechPlayback(turnId);
      } else if (isLast && !speechPlaybackStarted) {
        speechPlaybackStarted = true;
        startAvatarSpeechPlayback(turnId);
      }
      break;
    }

    // Future: true streaming from Modal — forward frames as Modal produces them
    // instead of current batch → early-start approach. Modal supports StreamingResponse.
    case "avatar.idle.start": {
      const totalFrames = msg.totalFrames as number;
      const fps = msg.fps as number;
      const durationMs = msg.durationMs as number | undefined;

      stopIdleAnimation();
      idleFrameImages = [];
      idleFrameIndex = 0;
      idleFps = fps;
      idleReady = false;
      idleIsLive = false;

      log("recv", `avatar.idle.start — totalFrames=${totalFrames} fps=${fps} duration=${durationMs}ms`);
      sendTelemetry("avatar_idle_start_received", { totalFrames, fps, durationMs });
      break;
    }

    case "avatar.idle.chunk": {
      const chunkIndex = msg.chunkIndex as number;
      const frames = msg.frames as Array<{ dataBase64: string; mimeType: string; width: number; height: number }>;
      const isLast = msg.isLast as boolean;
      const isLive = (msg.live as boolean | undefined) === true;

      if (isLive) {
        idleIsLive = true;
        const t0 = performance.now();
        const newImages: HTMLImageElement[] = [];
        let chunkDecodeTotalMs = 0;
        for (const f of frames) {
          const img = new Image();
          img.src = `data:${f.mimeType};base64,${f.dataBase64}`;
          newImages.push(img);
        }
        const imageCreateMs = performance.now() - t0;

        let base64CharsInChunk = 0;
        for (const f of frames) {
          base64CharsInChunk += f.dataBase64.length;
        }

        Promise.all(newImages.map(img => img.decode().catch(() => {}))).then(() => {
          const decodeMs = performance.now() - t0 - imageCreateMs;
          idleChunkDiag.count++;
          idleChunkDiag.totalImageCreateMs += imageCreateMs;
          idleChunkDiag.totalImageDecodeMs += decodeMs;
          idleChunkDiag.totalBase64Chars += base64CharsInChunk;
        });

        for (const img of newImages) {
          idleFrameImages.push(img);
        }
        while (idleFrameImages.length > idleBufferMaxFrames) {
          idleFrameImages.shift();
          if (idleFrameIndex > 0) idleFrameIndex--;
        }

        if (!idleReady && idleFrameImages.length > 0) {
          idleReady = true;
          idleFps = 25;
          log("info", `Live idle animation starting (${idleFrameImages.length} frames buffered)`);
          sendTelemetry("avatar_idle_live_ready", { bufferedFrames: idleFrameImages.length });
          startIdleDiagInterval();
        }

        if (!avatarSpeechActive && idleReady && idleAnimationTimer === null) {
          startIdleAnimation();
        }

        log("recv", `avatar.idle.chunk — LIVE chunk=${chunkIndex} +${frames.length} frames total=${idleFrameImages.length} createMs=${imageCreateMs.toFixed(1)}`);
      } else {
        for (const f of frames) {
          const img = new Image();
          img.src = `data:${f.mimeType};base64,${f.dataBase64}`;
          idleFrameImages.push(img);
        }

        log("recv", `avatar.idle.chunk — BATCH chunk=${chunkIndex} +${frames.length} frames total=${idleFrameImages.length} isLast=${isLast}`);

        if (isLast) {
          idleReady = true;
          idleIsLive = false;
          log("info", `Batch idle animation ready (${idleFrameImages.length} frames)`);
          sendTelemetry("avatar_idle_ready", { totalFrames: idleFrameImages.length });
          if (!avatarSpeechActive) {
            startIdleAnimation();
          }
        }
      }
      break;
    }

    case "session.ended": {
      const modalEndOk = msg.modalEndOk as boolean | undefined;
      const modalEndDurationMs = msg.modalEndDurationMs as number | undefined;
      const modalEndError = msg.modalEndError as string | undefined;
      log("recv", `session.ended — wasActive=${msg.wasActive} modalEndOk=${modalEndOk} modalEndMs=${modalEndDurationMs}${modalEndError ? ` err=${modalEndError}` : ""}`);
      setStatus("stopped", "Session ended");

      sfuCleanup();
      fmp4Destroy("session_ended");

      sendTelemetry("session_ended_received", {
        wasActive: msg.wasActive,
        modalEndOk,
        modalEndDurationMs,
        modalEndError,
      });

      const wsRef = ws;
      ws = null;
      stopMicLocal();
      stopIdleAnimation();
      stopSpeechPlaybackAll();
      stopIdleDiagInterval();
      flushIdleChunkDiag();
      avatarSpeechActive = false;
      idleFrameImages = [];
      idleReady = false;
      idleIsLive = false;
      currentSpeechTurnId = 0;
      activeSpeechTurnId = null;
      fallbackPlayedTurnIds.clear();
      if (avatarFirstChunkTimeoutId !== null) {
        clearTimeout(avatarFirstChunkTimeoutId);
        avatarFirstChunkTimeoutId = null;
      }
      idleUnderrunCountSinceReport = 0;
      idleUnderrunLastReportAt = 0;
      speechTimingAssistantAudioAt = 0;
      speechTimingRenderStartAt = 0;
      speechTimingSpeechStartAt = 0;
      speechTimingFirstChunkAt = 0;
      speechTimingPlaybackStartAt = 0;
      speechStreamEnded = false;
      speechExpectedTotalFrames = 0;
      speechDisplayedFrames = 0;
      speechUnderrunCount = 0;
      speechLastUnderrunReportAt = 0;
      speechAudioEnded = false;
      speechVideoEnded = false;
      speechAckSentForTurnId = null;
      speechPlaybackTurnId = null;
      speechAudioPlayback = null;
      speechClockDriftLastReportAt = 0;
      setButtons(false);

      setTimeout(() => {
        if (wsRef && wsRef.readyState === WebSocket.OPEN) {
          wsRef.close(1000, "client done");
        }
      }, 300);
      break;
    }

    case "error": {
      const code = msg.code as string;
      log("err", `ERROR [${code}] ${msg.message}`);

      if ((SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE) && typeof code === "string" && (code.startsWith("sfu_") || code.startsWith("direct_webrtc_") || code.startsWith("whep_"))) {
        sfuFail("server_error", `[${code}] ${msg.message}`);
        break;
      }

      if (currentLoadingStage !== null && currentLoadingStage !== "live") {
        showStartupError(`${code}`);
      } else {
        setStatus("error", `Error: ${code}`);
      }

      if (code === "avatar_render_failed" && pendingAudio) {
        log("info", "Avatar render failed — playing stored audio as fallback");
        sendTelemetry("avatar_render_failed_fallback", { turnId: pendingAudio.turnId });
        playPendingAudioFallback("render_failed");
      }

      if (msg.fatal) {
        ws?.close();
        ws = null;
        stopMicLocal();
        setButtons(false);
      }
      break;
    }

    case "status": {
      const s = msg.status as string;
      log("recv", `status: ${s}`);
      if (s === "mic_ready") {
        setStatus("mic_active", "Mic active");
      } else if (s === "mic_stopped") {
        setStatus("ready", "Mic stopped");
        setSpeechStatus("", "");
      } else if (s === "listening_same_turn") {
        setSpeechStatus("Waiting for continuation (same turn)", "#a371f7");
      } else if (s === "transcribing") {
        setSpeechStatus("Transcribing...", "#e3b341");
      } else if (s === "thinking") {
        setSpeechStatus("Thinking...", "#79c0ff");
      } else if (s === "synthesizing") {
        setSpeechStatus("Synthesizing voice...", "#f0883e");
      } else if (s === "rendering_avatar_audio") {
        speechTimingRenderStartAt = Date.now();
        setSpeechStatus("Rendering avatar speech...", "#79c0ff");
        setStatus("rendering", "Rendering avatar speech...");
      } else {
        setStatus("rendering", s.replace(/_/g, " "));
      }
      break;
    }

    case "speech.started": {
      const turnId = msg.turnId as number;
      log("recv", `speech.started — turnId=${turnId}`);
      setSpeechStatus(`Speaking (turn ${turnId})`, "#3fb950");
      break;
    }

    case "speech.segment.started": {
      const turnId = msg.turnId as number;
      const segmentIndex = msg.segmentIndex as number;
      log("recv", `speech.segment.started — turnId=${turnId} segment=${segmentIndex}`);
      setSpeechStatus(`Speaking again, same turn ${turnId} segment ${segmentIndex}`, "#3fb950");
      break;
    }

    case "speech.finalized": {
      const turnId = msg.turnId as number;
      const st = msg.smartTurn as { prediction: number; probability: number; rawLogit: number };
      const fallback = msg.smartTurnFallback ? " [fallback]" : "";
      log("recv", `speech.finalized — turnId=${turnId} prob=${st.probability.toFixed(3)}${fallback}`);
      setSpeechStatus(`Turn ${turnId} finalized`, "#58a6ff");
      break;
    }

    case "transcript.final": {
      const turnId = msg.turnId as number;
      const text = msg.text as string;
      const provider = msg.provider as string;
      const latencyMs = msg.latencyMs as number;
      const langCode = msg.languageCode as string | undefined;
      log("recv", `transcript.final — turnId=${turnId} "${text}" (${provider}, ${latencyMs}ms${langCode ? `, lang=${langCode}` : ""})`);
      setSpeechStatus(`Turn ${turnId}: "${text.substring(0, 80)}${text.length > 80 ? "..." : ""}"`, "#d2a8ff");

      const entry = document.createElement("div");
      entry.className = "transcript-entry";
      entry.innerHTML = `<span class="transcript-turn">T${turnId}</span> <span class="transcript-text">${escapeHtml(text)}</span>`;
      transcriptList.appendChild(entry);
      transcriptList.scrollTop = transcriptList.scrollHeight;
      break;
    }

    case "speech.continues": {
      const turnId = msg.turnId as number;
      const st = msg.smartTurn as { prediction: number; probability: number; rawLogit: number };
      log("recv", `speech.continues — turnId=${turnId} prob=${st.probability.toFixed(3)}`);
      setSpeechStatus(`Turn ${turnId} incomplete, waiting for continuation`, "#a371f7");
      break;
    }

    case "assistant.text": {
      const turnId = msg.turnId as number;
      const text = msg.text as string;
      const provider = msg.provider as string;
      const latencyMs = msg.latencyMs as number;
      log("recv", `assistant.text — turnId=${turnId} "${text.substring(0, 100)}${text.length > 100 ? "..." : ""}" (${provider}, ${latencyMs}ms)`);
      setSpeechStatus(`Assistant: "${text.substring(0, 80)}${text.length > 80 ? "..." : ""}"`, "#79c0ff");

      const entry = document.createElement("div");
      entry.className = "assistant-entry";
      entry.innerHTML = `<span class="assistant-turn">A${turnId}</span> <span class="assistant-text">${escapeHtml(text)}</span>`;
      transcriptList.appendChild(entry);
      transcriptList.scrollTop = transcriptList.scrollHeight;
      break;
    }

    case "assistant.audio": {
      const turnId = msg.turnId as number;
      const mimeType = msg.mimeType as string;
      const provider = msg.provider as string;
      const latencyMs = msg.latencyMs as number;
      const sampleRate = msg.sampleRate as number;
      const audioBytes = msg.audioBytes as number;

      if (fmp4AvMode) {
        const sizeKb = Math.round(audioBytes / 1024);
        log("recv", `assistant.audio — turnId=${turnId} ${sizeKb}KB ${sampleRate}Hz (${provider}, ${latencyMs}ms) [AV mode — metadata only]`);
        const entry = document.createElement("div");
        entry.className = "audio-entry";
        entry.innerHTML = `<span class="audio-turn">&#x1f50a;A${turnId}</span> <span class="audio-meta">Voice ready ${sizeKb}KB (${provider}, ${latencyMs}ms) — muxed into fMP4 A/V stream</span>`;
        transcriptList.appendChild(entry);
        transcriptList.scrollTop = transcriptList.scrollHeight;

        sendTelemetry("avatar_fmp4_av_audio_received_browser", { turnId, provider, latencyMs, audioBytes, sampleRate });
        log("info", `fMP4 AV mode — assistant.audio received turnId=${turnId}, NOT playing separately`);
        break;
      }

      // SFU / direct WebRTC guard MUST run before any access to msg.dataBase64.length.
      // In these modes the browser must receive playable bytes; metadata-only/direct-enqueued audio is an error.
      if (SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE) {
        const rawBase64 = msg.dataBase64 as string | undefined;
        const isDirectEnqueued = !!((msg as Record<string, unknown>).directEnqueued) || !rawBase64;

        if (isDirectEnqueued) {
          // Milestone 1 SFU requires browser-playable audio bytes. Direct-enqueued TTS
          // returns no bytes, which is incompatible. Log it clearly and do not pretend success.
          log("err", `SFU direct-enqueued turnId=${turnId} — no browser audio bytes (provider=${provider})`);
          sendTelemetry("sfu_direct_enqueued_without_browser_audio", { turnId, provider, latencyMs });

          const entry = document.createElement("div");
          entry.className = "audio-entry";
          entry.innerHTML = `<span class="audio-turn">&#x1f50a;A${turnId}</span> <span class="audio-meta" style="color:#f85149">Voice unavailable: direct-enqueued audio has no playable bytes (${provider}, ${latencyMs}ms)</span>`;
          transcriptList.appendChild(entry);
          transcriptList.scrollTop = transcriptList.scrollHeight;

          if (pendingAudio) {
            clearTimeout(pendingAudio.timeoutId);
            pendingAudio = null;
          }
          if (avatarFirstChunkTimeoutId !== null) {
            clearTimeout(avatarFirstChunkTimeoutId);
            avatarFirstChunkTimeoutId = null;
          }

          setSpeechStatus("Voice unavailable: direct-enqueued audio", "#f85149");
          break;
        }
      }

      const dataBase64 = msg.dataBase64 as string;
      const wavSizeKb = Math.round(dataBase64.length * 0.75 / 1024);
      log("recv", `assistant.audio — turnId=${turnId} ${mimeType} ${wavSizeKb}KB ${sampleRate}Hz (${provider}, ${latencyMs}ms)`);

      if (fmp4ModeExpected || fmp4InitSeen || fmp4MseReady) {
        const entry = document.createElement("div");
        entry.className = "audio-entry";
        entry.innerHTML = `<span class="audio-turn">&#x1f50a;A${turnId}</span> <span class="audio-meta">Voice ready ${wavSizeKb}KB (${provider}, ${latencyMs}ms) — playing immediately (fMP4 video mode)</span>`;
        transcriptList.appendChild(entry);
        transcriptList.scrollTop = transcriptList.scrollHeight;

        if (pendingAudio) {
          clearTimeout(pendingAudio.timeoutId);
          if (avatarFirstChunkTimeoutId !== null) {
            clearTimeout(avatarFirstChunkTimeoutId);
            avatarFirstChunkTimeoutId = null;
          }
          playPendingAudioFallback("new_audio_superseded_fmp4");
        }

        activeSpeechTurnId = turnId;
        fallbackPlayedTurnIds.delete(turnId);
        avatarSpeechActive = true;
        setSpeechStatus("Playing voice (fMP4 video mode)...", "#3fb950");
        sendTelemetry("avatar_fmp4_audio_play_started", { turnId, provider, latencyMs });

        playAudioBase64WithCallback(dataBase64, sampleRate, () => {
          log("info", `fMP4 video mode audio playback ended for turnId=${turnId}`);
          avatarSpeechActive = false;
          sendTelemetry("avatar_fmp4_audio_play_ended", { turnId });
          if (ws && ws.readyState === WebSocket.OPEN) {
            const ackMsg = JSON.stringify({ type: "avatar.speech.playback.ended", turnId });
            ws.send(ackMsg);
            log("send", ackMsg);
          }
        });
        break;
      }

      if (SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE) {
        const entry = document.createElement("div");
        entry.className = "audio-entry";
        const modeLabel = DIRECT_WEBRTC_DEBUG_MODE ? "direct WebRTC speech queue" : "SFU speech queue";
        entry.innerHTML = `<span class="audio-turn">&#x1f50a;A${turnId}</span> <span class="audio-meta">Voice ready ${wavSizeKb}KB (${provider}, ${latencyMs}ms) — waiting for ${modeLabel}</span>`;
        transcriptList.appendChild(entry);
        transcriptList.scrollTop = transcriptList.scrollHeight;

        if (pendingAudio) {
          clearTimeout(pendingAudio.timeoutId);
          if (avatarFirstChunkTimeoutId !== null) {
            clearTimeout(avatarFirstChunkTimeoutId);
            avatarFirstChunkTimeoutId = null;
          }
          playPendingAudioFallback("new_audio_superseded_sfu");
        }

        activeSpeechTurnId = turnId;
        fallbackPlayedTurnIds.delete(turnId);

        const arrivedAt = Date.now();

        const sfuTimeoutId = window.setTimeout(() => {
          if (pendingAudio && pendingAudio.turnId === turnId) {
            log("info", `SFU speech_queued timeout (${SFU_SPEECH_QUEUED_TIMEOUT_MS}ms) — playing audio fallback`);
            sendTelemetry("sfu_speech_queued_timeout", {
              turnId,
              timeSinceAssistantAudio: Date.now() - arrivedAt,
            });
            fallbackPlayedTurnIds.add(turnId);
            playPendingAudioFallback("sfu_speech_queued_timeout");
          }
        }, SFU_SPEECH_QUEUED_TIMEOUT_MS);

        pendingAudio = { base64: dataBase64, sampleRate, turnId, timeoutId: sfuTimeoutId, arrivedAt };
        speechTimingAssistantAudioAt = arrivedAt;
        speechTimingRenderStartAt = 0;
        speechTimingSpeechStartAt = 0;
        speechTimingFirstChunkAt = 0;
        speechTimingPlaybackStartAt = 0;
        setSpeechStatus("Voice ready, waiting for SFU render...", "#f0883e");
        break;
      }

      const entry = document.createElement("div");
      entry.className = "audio-entry";
      entry.innerHTML = `<span class="audio-turn">&#x1f50a;A${turnId}</span> <span class="audio-meta">Voice ready ${wavSizeKb}KB ${mimeType} (${provider}, ${latencyMs}ms) — waiting for avatar render</span>`;
      transcriptList.appendChild(entry);
      transcriptList.scrollTop = transcriptList.scrollHeight;

      if (pendingAudio) {
        const oldTurnId = pendingAudio.turnId;
        const oldAgeMs = Date.now() - pendingAudio.arrivedAt;
        const oldHadSpeechStart = currentSpeechTurnId === oldTurnId;
        const oldHadSpeechChunks = speechFrameImages.length;
        sendTelemetry("assistant_audio_superseded", {
          oldTurnId,
          newTurnId: turnId,
          oldAgeMs,
          oldHadSpeechStart,
          oldHadSpeechChunks,
        });
        clearTimeout(pendingAudio.timeoutId);
        if (avatarFirstChunkTimeoutId !== null) {
          clearTimeout(avatarFirstChunkTimeoutId);
          avatarFirstChunkTimeoutId = null;
        }
        playPendingAudioFallback("new_audio_superseded");
      }

      activeSpeechTurnId = turnId;
      fallbackPlayedTurnIds.delete(turnId);

      const arrivedAt = Date.now();

      const firstChunkTimeoutId = window.setTimeout(() => {
        if (pendingAudio && pendingAudio.turnId === turnId && speechTimingFirstChunkAt === 0) {
          log("info", `Avatar first-chunk timeout (${AVATAR_FIRST_CHUNK_TIMEOUT_MS}ms) — playing audio fallback`);
          sendTelemetry("avatar_render_timeout_fallback", {
            turnId,
            timeSinceAssistantAudio: Date.now() - speechTimingAssistantAudioAt,
            hadSpeechStart: speechTimingSpeechStartAt > 0,
            hadSpeechChunks: speechFrameImages.length > 0,
            pendingRenderState: "first_chunk_timeout",
            fallbackStage: "first_chunk",
          });
          fallbackPlayedTurnIds.add(turnId);
          playPendingAudioFallback("first_chunk_timeout");
        }
      }, AVATAR_FIRST_CHUNK_TIMEOUT_MS);
      avatarFirstChunkTimeoutId = firstChunkTimeoutId;

      const absoluteTimeoutId = window.setTimeout(() => {
        if (pendingAudio && pendingAudio.turnId === turnId) {
          log("info", `Avatar render absolute timeout (${AVATAR_RENDER_TIMEOUT_MS}ms) — playing audio fallback`);
          sendTelemetry("avatar_render_timeout_fallback", {
            turnId,
            timeSinceAssistantAudio: Date.now() - speechTimingAssistantAudioAt,
            hadSpeechStart: speechTimingSpeechStartAt > 0,
            hadSpeechChunks: speechFrameImages.length > 0,
            pendingRenderState: "absolute_timeout",
            fallbackStage: "absolute",
          });
          fallbackPlayedTurnIds.add(turnId);
          playPendingAudioFallback("absolute_timeout");
        }
      }, AVATAR_RENDER_TIMEOUT_MS);

      pendingAudio = { base64: dataBase64, sampleRate, turnId, timeoutId: absoluteTimeoutId, arrivedAt };
      speechTimingAssistantAudioAt = arrivedAt;
      speechTimingRenderStartAt = 0;
      speechTimingSpeechStartAt = 0;
      speechTimingFirstChunkAt = 0;
      speechTimingPlaybackStartAt = 0;
      speechHadChunksBeforeFallback = false;
      setSpeechStatus("Voice ready, waiting for avatar render...", "#f0883e");
      break;
    }

    case "avatar.sfu.speech_queued": {
      const turnId = msg.turnId as number;
      const chunks = msg.chunks as number;
      log("recv", `avatar.sfu.speech_queued — turnId=${turnId} chunks=${chunks}`);

      if (pendingAudio && pendingAudio.turnId === turnId) {
        clearTimeout(pendingAudio.timeoutId);
        const { base64, sampleRate } = pendingAudio;
        pendingAudio = null;

        if (avatarFirstChunkTimeoutId !== null) {
          clearTimeout(avatarFirstChunkTimeoutId);
          avatarFirstChunkTimeoutId = null;
        }

        avatarSpeechActive = true;
        activeSpeechTurnId = turnId;
        stopIdleAnimation();
        idleFrameImages = [];
        idleReady = false;

        const playbackStartAt = Date.now();
        speechTimingPlaybackStartAt = playbackStartAt;
        speechTimingFirstChunkAt = playbackStartAt;

        setSpeechStatus("Playing voice (SFU mode)...", "#3fb950");
        log("info", `SFU mode — playing audio for turnId=${turnId} (speech queued, ${chunks} chunks)`);

        playAudioBase64WithCallback(base64, sampleRate, () => {
          log("info", `SFU mode audio playback ended for turnId=${turnId}`);
          avatarSpeechActive = false;
          if (ws && ws.readyState === WebSocket.OPEN) {
            const ackMsg = JSON.stringify({ type: "avatar.speech.playback.ended", turnId });
            ws.send(ackMsg);
            log("send", ackMsg);
          }
        });
      } else {
        log("info", `avatar.sfu.speech_queued for turnId=${turnId} but no matching pendingAudio`);
      }
      break;
    }

    case "audio.stats": {
      const chunks = msg.chunksReceived as number;
      const bytes = msg.bytesReceived as number;
      log("recv", `audio.stats — ${chunks} chunks, ${(bytes / 1024).toFixed(1)}KB received`);
      break;
    }

    case "avatar.timeline.started": {
      const mode = (msg as Record<string, unknown>).mode as string | undefined;
      fmp4ModeExpected = mode === "fmp4_video" || mode === "fmp4_av";
      if (mode === "fmp4_av") fmp4AvMode = true;
      log("recv", `avatar.timeline.started — mode=${mode ?? "unknown"}`);
      break;
    }

    case "avatar.timeline.stopped": {
      log("recv", `avatar.timeline.stopped — reason=${msg.reason}`);
      fmp4Destroy("timeline_stopped");
      break;
    }

    case "avatar.fmp4.init": {
      if (!fmp4IsActive()) break;
      const dataBase64 = msg.dataBase64 as string;
      const byteLength = msg.byteLength as number;
      const timingMs = msg.timingMs as number;
      fmp4InitReceivedAt = Date.now();
      fmp4InitSeen = true;

      if (fmp4FatalError) break;

      const receivedAt = Date.now();

      if (fmp4SourceBuffer && !fmp4SourceBuffer.updating && !fmp4IsAppending) {
        try {
          const initData = base64ToArrayBuffer(dataBase64);
          fmp4SourceBuffer.appendBuffer(initData);
          fmp4IsAppending = true;
          fmp4InitAppendedAt = Date.now();

          log("info", `fMP4 init append started (SourceBuffer ready, ${byteLength} bytes, timeToInit=${timingMs}ms, queued=${fmp4SegmentQueue.length})`);
          sendTelemetry("avatar_fmp4_init_appended", {
            stage: "init_direct",
            byteLength,
            timeToInitMs: timingMs,
            timeFromInitReceivedMs: fmp4InitAppendedAt - receivedAt,
            timeFromSessionReadyMs: fmp4SessionReadyAt > 0 ? fmp4InitAppendedAt - fmp4SessionReadyAt : undefined,
            queuedSegments: fmp4SegmentQueue.length,
          });
        } catch (appendErr) {
          log("err", `fMP4 init append failed: ${appendErr}`);
          fmp4FatalError = true;
          sendTelemetry("avatar_fmp4_mse_error", { error: String(appendErr), stage: "init_append" });
          showStartupError("Video stream initialization failed");
        }
      } else {
        fmp4PendingInitData = { dataBase64, byteLength, timingMs, receivedAt };
        log("info", `fMP4 init arrived but SourceBuffer not ready — pending (queued=${fmp4SegmentQueue.length})`);
        sendTelemetry("avatar_fmp4_init_pending", {
          byteLength,
          timeToInitMs: timingMs,
          queuedSegments: fmp4SegmentQueue.length,
          sourceBufferExists: fmp4SourceBuffer !== null,
        });
      }
      break;
    }

    case "avatar.fmp4.segment": {
      const seq = msg.sequence as number;
      const mode = msg.mode as string;
      const byteLen = msg.byteLength as number;

      if (fmp4FatalError) {
        fmp4DroppedFatalCount++;
        const now = Date.now();
        if (now - fmp4DroppedFatalLastTelemetryAt > 5000) {
          sendTelemetry("avatar_fmp4_segment_dropped_fatal", {
            sequence: seq,
            byteLength: byteLen,
            totalDropped: fmp4DroppedFatalCount,
          });
          fmp4DroppedFatalLastTelemetryAt = now;
        }
        break;
      }
      // Drop segments after user stops or pipeline is destroyed
      if (!fmp4IsActive()) {
        break;
      }
      if (!fmp4InitSeen) break;

      const b64 = (msg as Record<string, unknown>).dataBase64 as string;
      if (!b64) break;

      fmp4SegmentQueue.push({ b64, byteLength: byteLen, sequence: seq, mode });
      fmp4SegmentCount++;
      fmp4SegmentsQueued++;
      fmp4SegmentsReceived++;
      fmp4LastReceivedSequence = seq;

      if (VERBOSE_DELIVERY_TELEMETRY || fmp4SegmentsReceived % 10 === 0) {
        const video = ensureFmp4VideoElement();
        sendTelemetry("avatar_fmp4_browser_segment_received", {
          sequence: seq,
          mode,
          byteLength: byteLen,
          receivedCount: fmp4SegmentsReceived,
          queuedCount: fmp4SegmentsQueued,
          appendedCount: fmp4SegmentsAppended,
          appendErrors: fmp4AppendErrors,
          queueLength: fmp4SegmentQueue.length,
          mseReady: fmp4MseReady,
          sourceBufferUpdating: fmp4SourceBuffer?.updating,
          segmentCount: fmp4SegmentCount,
        });
      }

      const video = ensureFmp4VideoElement();
      fmp4SummaryQueuedCount++;
      fmp4MaybeSendSummary();

      if (mode === "speech" && (msg.isSpeechStart as boolean)) {
        fmp4SpeechActive = true;
        avatarFrame.classList.add("avatar-speaking");
        if (fmp4AvMode) {
          setSpeechStatus("Speaking (fMP4 A/V)...", "#79c0ff");
          sendTelemetry("avatar_fmp4_av_speech_video_started_browser", { sequence: seq, turnId: (msg as Record<string, unknown>).turnId ?? null });
        } else {
          setSpeechStatus("Speaking (fMP4 video)...", "#79c0ff");
        }
      }
      if (mode === "speech" && (msg.isSpeechEnd as boolean)) {
        fmp4SpeechActive = false;
        avatarFrame.classList.remove("avatar-speaking");

        const now = Date.now();
        fmp4LastSpeechEndedAt = now;
        const POST_SPEECH_GRACE_MS = 4000;
        let dynamicGraceMs = POST_SPEECH_GRACE_MS;

        const v = ensureFmp4VideoElement();
        if (v.buffered.length > 0) {
          const liveEdge = v.buffered.end(v.buffered.length - 1);
          const behindMs = Math.max(0, Math.round((liveEdge - v.currentTime) * 1000));
          dynamicGraceMs = Math.max(POST_SPEECH_GRACE_MS, behindMs + 1500);
        }
        fmp4SeekBlockedUntil = now + dynamicGraceMs;

        if (fmp4AvMode) {
          setSpeechStatus("Idle (fMP4 A/V)", "#8b949e");
          sendTelemetry("avatar_fmp4_av_speech_video_ended_browser", { sequence: seq, turnId: (msg as Record<string, unknown>).turnId ?? null });
          const ackTurnId = (msg as Record<string, unknown>).turnId as number | null;
          if (ackTurnId != null) {
            if (fmp4SpeechPlaybackAckTimerId !== null) {
              clearTimeout(fmp4SpeechPlaybackAckTimerId);
            }
            fmp4SpeechPlaybackAckTimerId = setTimeout(() => {
              fmp4SpeechPlaybackAckTimerId = null;
              if (ws && ws.readyState === WebSocket.OPEN) {
                const ackMsg = JSON.stringify({ type: "avatar.speech.playback.ended", turnId: ackTurnId });
                ws.send(ackMsg);
                log("send", ackMsg);
              }
            }, 1000);
          }
        } else {
          setSpeechStatus("Idle (fMP4 video)", "#8b949e");
        }
      }

      if (fmp4MseReady) {
        fmp4DrainQueue();
      }

      break;
    }

    default: {
      log("recv", JSON.stringify(msg));
    }
  }
}

async function startMic() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  sendTelemetry("start_mic_clicked");
  log("info", "Start mic clicked");

  try {
    audioContext = new AudioContext({
      sampleRate: 16000,
      latencyHint: "interactive",
    });

    await audioContext.audioWorklet.addModule("./audio-worklet.js");

    sendTelemetry("mic_permission_requested");
    log("info", "Requesting mic permission...");

    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: !rawAudioCheckbox.checked,
        noiseSuppression: !rawAudioCheckbox.checked,
        autoGainControl: !rawAudioCheckbox.checked,
      },
    });

    const track = micStream.getAudioTracks()[0];
    const settings = track.getSettings();
    log("info", `Mic settings: rate=${settings.sampleRate ?? "n/a"} ch=${settings.channelCount ?? "n/a"} echoCancel=${settings.echoCancellation ?? "n/a"} noiseSup=${settings.noiseSuppression ?? "n/a"} agc=${settings.autoGainControl ?? "n/a"} latency=${(settings as Record<string, unknown>).latency ?? "n/a"}ms`);

    sendTelemetry("mic_permission_granted", {
      sampleRate: audioContext.sampleRate,
      trackSettings: {
        sampleRate: settings.sampleRate,
        channelCount: settings.channelCount,
        echoCancellation: settings.echoCancellation,
        noiseSuppression: settings.noiseSuppression,
        autoGainControl: settings.autoGainControl,
        latency: (settings as Record<string, unknown>).latency,
      },
      browserAudioProcessing: !rawAudioCheckbox.checked,
    });
    log("info", "Mic permission granted");

    const source = audioContext.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioContext, "mic-capture-processor");

    silentGain = audioContext.createGain();
    silentGain.gain.value = 0;

    source.connect(workletNode);
    workletNode.connect(silentGain);
    silentGain.connect(audioContext.destination);

    micChunksSent = 0;
    micBytesSent = 0;
    firstAudioChunkCaptured = false;
    audioRmsSum = 0;
    audioRmsCount = 0;
    audioPeakMax = 0;
    audioNearZeroSum = 0;
    audioClippedSum = 0;
    rollingBuffer.length = 0;
    updateAudioStatsDisplay();

    workletNode.port.onmessage = (event: MessageEvent) => {
      const pcm16 = event.data.pcm16 as ArrayBuffer;
      const rms = event.data.rms as number;
      const peak = event.data.peak as number;
      const nearZeroRatio = event.data.nearZeroRatio as number;
      const clippedRatio = event.data.clippedRatio as number;

      audioRmsSum += rms;
      audioRmsCount++;
      if (peak > audioPeakMax) audioPeakMax = peak;
      audioNearZeroSum += nearZeroRatio;
      audioClippedSum += clippedRatio;

      rollingBuffer.push({ rms, peak, nearZeroRatio, clippedRatio });
      if (rollingBuffer.length > ROLLING_WINDOW) rollingBuffer.shift();

      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(pcm16);
        micChunksSent++;
        micBytesSent += pcm16.byteLength;

        if (!firstAudioChunkCaptured) {
          firstAudioChunkCaptured = true;
          sendTelemetry("audio_chunk_sent_first", {
            byteLength: pcm16.byteLength,
            rms,
            peak,
          });
          log("info", `First audio chunk sent: ${pcm16.byteLength} bytes rms=${rms.toFixed(4)} peak=${peak.toFixed(3)}`);
        }

        if (micChunksSent % 50 === 0) {
          updateAudioStatsDisplay();
          const rollingAvgRms = rollingBuffer.length > 0 ? rollingBuffer.reduce((s, m) => s + m.rms, 0) / rollingBuffer.length : 0;
          const rollingMaxPeak = rollingBuffer.length > 0 ? Math.max(...rollingBuffer.map(m => m.peak)) : 0;
          const rollingNearZeroAvg = rollingBuffer.length > 0 ? rollingBuffer.reduce((s, m) => s + m.nearZeroRatio, 0) / rollingBuffer.length : 0;
          const rollingClippedAvg = rollingBuffer.length > 0 ? rollingBuffer.reduce((s, m) => s + m.clippedRatio, 0) / rollingBuffer.length : 0;
          sendTelemetry("audio_chunk_stats_enhanced", {
            chunksSent: micChunksSent,
            bytesSent: micBytesSent,
            avgRms: audioRmsCount > 0 ? audioRmsSum / audioRmsCount : 0,
            maxPeak: audioPeakMax,
            nearZeroRatioAvg: audioRmsCount > 0 ? audioNearZeroSum / audioRmsCount : 0,
            clippedRatioAvg: audioRmsCount > 0 ? audioClippedSum / audioRmsCount : 0,
            rollingAvgRms,
            rollingMaxPeak,
            rollingNearZeroRatioAvg: rollingNearZeroAvg,
            rollingClippedRatioAvg: rollingClippedAvg,
            rollingWindowSize: rollingBuffer.length,
            sampleRate: audioContext?.sampleRate,
          });
          log("info", `Rolling audio stats [last ${ROLLING_WINDOW} chunks]: rms=${rollingAvgRms.toFixed(4)} peak=${rollingMaxPeak.toFixed(3)} silent=${rollingNearZeroAvg.toFixed(3)} clipped=${rollingClippedAvg.toFixed(3)} | Cumulative: avgRms=${(audioRmsCount > 0 ? audioRmsSum / audioRmsCount : 0).toFixed(4)} peak=${audioPeakMax.toFixed(3)} silent=${(audioRmsCount > 0 ? audioNearZeroSum / audioRmsCount : 0).toFixed(3)} clipped=${(audioRmsCount > 0 ? audioClippedSum / audioRmsCount : 0).toFixed(3)}`);
        }
      }
    };

    const micMsg = JSON.stringify({ type: "mic.start" });
    ws.send(micMsg);
    log("send", micMsg);

    setButtons(true);
    setStatus("mic_active", "Mic active");
    log("info", `Mic started — context rate: ${audioContext.sampleRate}Hz`);

    sendTelemetry("mic_started", {
      contextSampleRate: audioContext.sampleRate,
    });
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : String(err);
    log("err", `Mic failed: ${errMsg}`);

    if (errMsg.includes("Permission") || errMsg.includes("NotAllowedError")) {
      sendTelemetry("mic_permission_denied", { error: errMsg });
      log("info", "Mic permission denied");
    } else {
      sendTelemetry("client_error", { context: "mic", error: errMsg });
    }

    stopMicLocal();
    setStatus("error", "Mic failed");
  }
}

function stopMic() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  sendTelemetry("stop_mic_clicked");
  log("info", "Stop mic clicked");

  stopMicLocal();

  const msg = JSON.stringify({ type: "mic.stop" });
  ws.send(msg);
  log("send", msg);
}

function stopMicLocal() {
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (silentGain) {
    silentGain.disconnect();
    silentGain = null;
  }
  if (micStream) {
    for (const track of micStream.getTracks()) {
      track.stop();
    }
    micStream = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  micChunksSent = 0;
  micBytesSent = 0;
  firstAudioChunkCaptured = false;
  audioRmsSum = 0;
  audioRmsCount = 0;
  audioPeakMax = 0;
  audioNearZeroSum = 0;
  audioClippedSum = 0;
  rollingBuffer.length = 0;
  updateAudioStatsDisplay();
  setButtons(ws !== null && ws.readyState === WebSocket.OPEN);

  sendTelemetry("mic_stopped_local");
}

function stop() {
  sfuCleanup();
  fmp4BeginEnding("stop_clicked");

  if (!ws || ws.readyState !== WebSocket.OPEN) {
    fmp4Destroy("stop_no_ws");
    setStatus("stopped", "disconnected");
    ws = null;
    stopMicLocal();
    stopIdleAnimation();
    stopSpeechPlaybackAll();
    stopIdleDiagInterval();
    flushIdleChunkDiag();
    avatarSpeechActive = false;
    idleFrameImages = [];
    idleReady = false;
    idleIsLive = false;
    currentSpeechTurnId = 0;
    activeSpeechTurnId = null;
    fallbackPlayedTurnIds.clear();
    if (avatarFirstChunkTimeoutId !== null) {
      clearTimeout(avatarFirstChunkTimeoutId);
      avatarFirstChunkTimeoutId = null;
    }
    idleUnderrunCountSinceReport = 0;
    idleUnderrunLastReportAt = 0;
    speechStreamEnded = false;
    speechExpectedTotalFrames = 0;
    speechDisplayedFrames = 0;
    speechUnderrunCount = 0;
    speechLastUnderrunReportAt = 0;
    speechAudioEnded = false;
    speechVideoEnded = false;
    speechAckSentForTurnId = null;
    speechPlaybackTurnId = null;
    speechAudioPlayback = null;
    speechClockDriftLastReportAt = 0;
    setButtons(false);
    return;
  }

  sendTelemetry("stop_session_clicked");
  log("info", "Stop session clicked");

  if (micStream !== null) {
    log("info", "Stopping mic before session.end");
    sendTelemetry("stop_mic_before_session_end");
    stopMicLocal();
    const micStopMsg = JSON.stringify({ type: "mic.stop" });
    ws.send(micStopMsg);
    log("send", micStopMsg);
  }

  const msg = JSON.stringify({ type: "session.end" });
  ws.send(msg);
  log("send", msg);
  setStatus("stopped", "Ending session...");

  sendTelemetry("session_end_sent");
}

async function playAudioBase64(base64: string, sampleRate: number): Promise<void> {
  try {
    const binaryStr = atob(base64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    if (!playbackAudioContext || playbackAudioContext.state === "closed") {
      playbackAudioContext = new AudioContext({ sampleRate });
    }

    await playbackAudioContext.resume();

    const audioBuffer = await playbackAudioContext.decodeAudioData(bytes.buffer);
    const source = playbackAudioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(playbackAudioContext.destination);
    source.start(0);

    log("info", `Playing TTS audio: ${audioBuffer.duration.toFixed(2)}s ${audioBuffer.sampleRate}Hz ${audioBuffer.numberOfChannels}ch`);
    setSpeechStatus("Playing assistant voice...", "#3fb950");

    source.onended = () => {
      log("info", "TTS audio playback ended");
    };
  } catch (err) {
    log("err", `Audio playback failed: ${err instanceof Error ? err.message : String(err)}`);
  }
}

function playPendingAudioFallback(reason: string): void {
  if (!pendingAudio) return;
  clearTimeout(pendingAudio.timeoutId);
  const { base64, sampleRate, turnId } = pendingAudio;
  pendingAudio = null;
  fallbackPlayedTurnIds.add(turnId);
  if (avatarFirstChunkTimeoutId !== null) {
    clearTimeout(avatarFirstChunkTimeoutId);
    avatarFirstChunkTimeoutId = null;
  }
  stopSpeechPlaybackAll();
  log("info", `Playing pending audio fallback: reason=${reason} turnId=${turnId}`);
  setSpeechStatus("Playing voice (avatar render fallback)...", "#3fb950");
  avatarSpeechActive = true;
  speechAckSentForTurnId = turnId;
  stopIdleAnimation();
  idleFrameImages = [];
  idleReady = false;
  playAudioBase64WithCallback(base64, sampleRate, () => {
    log("info", "Fallback audio playback ended");
    avatarSpeechActive = false;
    if (ws && ws.readyState === WebSocket.OPEN) {
      const ackMsg = JSON.stringify({ type: "avatar.speech.playback.ended", turnId });
      ws.send(ackMsg);
      log("send", ackMsg);
      sendTelemetry("avatar_speech_playback_ack_sent_fallback", { turnId, reason });
    }
  });
}

function playAudioBase64WithCallback(
  base64: string,
  sampleRate: number,
  onEnded?: () => void,
): void {
  try {
    const binaryStr = atob(base64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }

    if (!playbackAudioContext || playbackAudioContext.state === "closed") {
      playbackAudioContext = new AudioContext({ sampleRate });
    }

    playbackAudioContext.resume().then(() => {
      playbackAudioContext!.decodeAudioData(bytes.buffer).then((audioBuffer) => {
        const source = playbackAudioContext!.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(playbackAudioContext!.destination);
        source.start(0);

        log("info", `Playing TTS audio: ${audioBuffer.duration.toFixed(2)}s ${audioBuffer.sampleRate}Hz ${audioBuffer.numberOfChannels}ch`);
        setSpeechStatus("Playing assistant voice...", "#3fb950");

        source.onended = () => {
          log("info", "TTS audio playback ended");
          onEnded?.();
        };
      });
    });
  } catch (err) {
    log("err", `Audio playback failed: ${err instanceof Error ? err.message : String(err)}`);
    onEnded?.();
  }
}

async function playSpeechAudioBase64WithClock(
  base64: string,
  sampleRate: number,
  onEnded?: () => void,
): Promise<SpeechAudioPlayback> {
  const binaryStr = atob(base64);
  const bytes = new Uint8Array(binaryStr.length);
  for (let i = 0; i < binaryStr.length; i++) {
    bytes[i] = binaryStr.charCodeAt(i);
  }

  if (!playbackAudioContext || playbackAudioContext.state === "closed") {
    playbackAudioContext = new AudioContext({ sampleRate });
  }

  await playbackAudioContext.resume();

  const audioBuffer = await playbackAudioContext.decodeAudioData(bytes.buffer);
  const source = playbackAudioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(playbackAudioContext.destination);

  const audioStartTime = playbackAudioContext.currentTime;
  const audioDurationMs = audioBuffer.duration * 1000;

  source.start(0);

  log("info", `Playing speech audio (clock): ${audioBuffer.duration.toFixed(2)}s ${audioBuffer.sampleRate}Hz ${audioBuffer.numberOfChannels}ch`);

  source.onended = () => {
    log("info", "Speech audio playback ended (clock)");
    onEnded?.();
  };

  return {
    audioContext: playbackAudioContext,
    audioStartTime,
    audioDurationMs,
    stop: () => {
      try { source.stop(); } catch {}
      try { source.disconnect(); } catch {}
    },
  };
}

function startAvatarSpeechPlayback(turnId: number): void {
  if (!pendingAudio || pendingAudio.turnId !== turnId) {
    log("info", `avatar.speech complete but no matching pending audio for turnId=${turnId}`);
    if (speechFrameImages.length > 0 && !pendingAudio) {
      sendTelemetry("avatar_speech_late_after_fallback", {
        turnId,
        framesArrived: speechFrameImages.length,
        timeSinceAssistantAudio: speechTimingAssistantAudioAt ? Date.now() - speechTimingAssistantAudioAt : 0,
      });
    }
    return;
  }

  speechTimingPlaybackStartAt = Date.now();
  sendTelemetry("avatar_speech_browser_timing", {
    turnId,
    timeSinceAssistantAudio: speechTimingAssistantAudioAt ? Date.now() - speechTimingAssistantAudioAt : 0,
    renderStartToSpeechStart: speechTimingSpeechStartAt && speechTimingRenderStartAt ? speechTimingSpeechStartAt - speechTimingRenderStartAt : 0,
    speechStartToFirstChunk: speechTimingFirstChunkAt && speechTimingSpeechStartAt ? speechTimingFirstChunkAt - speechTimingSpeechStartAt : 0,
    firstChunkToPlayback: speechTimingFirstChunkAt ? Date.now() - speechTimingFirstChunkAt : 0,
    assistantAudioToPlayback: speechTimingAssistantAudioAt ? Date.now() - speechTimingAssistantAudioAt : 0,
    hadSpeechStart: speechTimingSpeechStartAt > 0,
    totalFramesCollected: speechFrameImages.length,
  });

  avatarSpeechActive = true;
  stopIdleAnimation();
  idleFrameImages = [];
  idleFrameIndex = 0;
  idleReady = false;

  if (avatarFirstChunkTimeoutId !== null) {
    clearTimeout(avatarFirstChunkTimeoutId);
    avatarFirstChunkTimeoutId = null;
  }

  clearTimeout(pendingAudio.timeoutId);
  const { base64, sampleRate } = pendingAudio;
  pendingAudio = null;
  fallbackPlayedTurnIds.delete(turnId);

  speechPlaybackTurnId = turnId;
  speechAudioEnded = false;
  speechVideoEnded = false;
  speechDisplayedFrames = 0;
  speechUnderrunCount = 0;
  speechAckSentForTurnId = null;
  speechClockDriftLastReportAt = 0;
  speechAudioPlayback = null;

  stopSpeechVideoAnimation();

  const totalFramesAtStart = speechFrameImages.length;

  setSpeechStatus("Avatar speaking...", "#3fb950");

  playSpeechAudioBase64WithClock(base64, sampleRate, () => {
    log("info", "Speech audio playback ended (clock callback)");
    speechAudioEnded = true;

    sendTelemetry("avatar_speech_audio_ended", {
      turnId,
      speechVideoEnded,
      speechStreamEnded,
      speechDisplayedFrames,
      speechUnderrunCount,
      renderedAudioDurationMs: speechRenderedDurationMs,
      originalAudioDurationMs: speechOriginalDurationMs,
    });

    if (speechFrameImages.length > 0 && speechAnimationFrameIndex < speechFrameImages.length - 1) {
      displaySpeechFrame(speechFrameImages.length - 1);
      speechAnimationFrameIndex = speechFrameImages.length - 1;
    }

    if (!speechVideoEnded && speechStreamEnded) {
      speechVideoEnded = true;
      stopSpeechVideoAnimation();
      log("info", `Speech video marked complete at audio end (stream was ended, ${speechFrameImages.length} frames)`);
    }

    maybeSendSpeechPlaybackAck(turnId);
  }).then((playback) => {
    speechAudioPlayback = playback;

    if (speechExpectedTotalFrames <= 0) {
      speechExpectedTotalFrames = Math.ceil(playback.audioDurationMs / (1000 / speechFps));
    }

    const expectedDurationMs = (speechExpectedTotalFrames / speechFps) * 1000;
    log("info", `Starting avatar speech (clock): ${totalFramesAtStart} frames buffered (expecting ${speechExpectedTotalFrames}) at ${speechFps}fps (~${expectedDurationMs.toFixed(0)}ms) audioDur=${playback.audioDurationMs.toFixed(0)}ms streamEnded=${speechStreamEnded}`);

    sendTelemetry("avatar_speech_clock_playback_started", {
      turnId,
      totalFramesAtStart,
      expectedTotalFrames: speechExpectedTotalFrames,
      fps: speechFps,
      expectedDurationMs,
      audioDurationMs: playback.audioDurationMs,
      renderedAudioDurationMs: speechRenderedDurationMs,
      originalAudioDurationMs: speechOriginalDurationMs,
      speechStreamEnded,
    });

    sendTelemetry("avatar_speech_playback_started", {
      turnId,
      totalFramesAtStart,
      expectedTotalFrames: speechExpectedTotalFrames,
      fps: speechFps,
      expectedDurationMs,
      renderedAudioDurationMs: speechRenderedDurationMs,
      originalAudioDurationMs: speechOriginalDurationMs,
      speechStreamEnded,
    });

    if (speechOriginalDurationMs > 0 && speechRenderedDurationMs > 0 && speechOriginalDurationMs > speechRenderedDurationMs) {
      const renderedS = (speechRenderedDurationMs / 1000).toFixed(1);
      const extraS = ((speechOriginalDurationMs - speechRenderedDurationMs) / 1000).toFixed(1);
      log("info", `Avatar render capped at ${renderedS}s; voice continues for ${extraS}s more`);
      sendTelemetry("avatar_render_capped", {
        turnId,
        renderedDurationMs: speechRenderedDurationMs,
        originalDurationMs: speechOriginalDurationMs,
      });
    }

    speechAnimationFrameIndex = 0;
    if (speechFrameImages.length > 0) {
      displaySpeechFrame(0);
      speechDisplayedFrames = 1;
    }

    const frameIntervalMs = 1000 / speechFps;

    function tick() {
      if (speechAckSentForTurnId === turnId) return;

      if (!speechAudioPlayback) return;

      const audioElapsedMs = (speechAudioPlayback.audioContext.currentTime - speechAudioPlayback.audioStartTime) * 1000;
      const targetFrameIndex = Math.floor(audioElapsedMs / frameIntervalMs);

      if (targetFrameIndex >= speechFrameImages.length) {
        if (speechStreamEnded) {
          const finalIndex = speechFrameImages.length - 1;
          if (finalIndex >= 0 && speechAnimationFrameIndex !== finalIndex) {
            displaySpeechFrame(finalIndex);
            speechAnimationFrameIndex = finalIndex;
            speechDisplayedFrames++;
          }
          speechVideoEnded = true;
          stopSpeechVideoAnimation();
          log("info", `Avatar speech video ended: ${speechFrameImages.length} frames (stream complete, clock-synced)`);
          sendTelemetry("avatar_speech_stream_completed_browser", {
            turnId,
            displayedFrames: speechDisplayedFrames,
            totalFrames: speechFrameImages.length,
            expectedTotalFrames: speechExpectedTotalFrames,
            speechUnderrunCount,
            renderedAudioDurationMs: speechRenderedDurationMs,
            originalAudioDurationMs: speechOriginalDurationMs,
          });
          maybeSendSpeechPlaybackAck(turnId);
          return;
        }

        const frameDrift = targetFrameIndex - (speechFrameImages.length - 1);
        speechUnderrunCount++;
        const now = Date.now();
        if (now - speechLastUnderrunReportAt >= 5000) {
          sendTelemetry("avatar_speech_stream_buffer_underrun", {
            turnId,
            displayedFrames: speechDisplayedFrames,
            bufferedFrames: speechFrameImages.length,
            expectedTotalFrames: speechExpectedTotalFrames,
            speechStreamEnded,
            speechUnderrunCount,
            renderedAudioDurationMs: speechRenderedDurationMs,
            originalAudioDurationMs: speechOriginalDurationMs,
          });
          speechLastUnderrunReportAt = now;
          log("info", `Speech stream buffer underrun: displayed=${speechDisplayedFrames} buffered=${speechFrameImages.length} expected=${speechExpectedTotalFrames} underruns=${speechUnderrunCount}`);
        }
        if (frameDrift >= 3 && now - speechClockDriftLastReportAt >= 2000) {
          sendTelemetry("avatar_speech_sync_drift", {
            turnId,
            audioElapsedMs,
            targetFrameIndex,
            displayedFrameIndex: speechFrameImages.length - 1,
            availableFrames: speechFrameImages.length,
            expectedTotalFrames: speechExpectedTotalFrames,
            frameDrift,
            speechStreamEnded,
            speechBufferUnderrunCount: speechUnderrunCount,
            audioDurationMs: speechAudioPlayback ? speechAudioPlayback.audioDurationMs : 0,
            renderedAudioDurationMs: speechRenderedDurationMs,
            originalAudioDurationMs: speechOriginalDurationMs,
          });
          speechClockDriftLastReportAt = now;
        }

        speechAnimationRafId = requestAnimationFrame(tick);
        return;
      }

      if (targetFrameIndex !== speechAnimationFrameIndex) {
        displaySpeechFrame(targetFrameIndex);
        speechAnimationFrameIndex = targetFrameIndex;
        speechDisplayedFrames++;
      }

      speechAnimationRafId = requestAnimationFrame(tick);
    }

    speechAnimationRafId = requestAnimationFrame(tick);
  }).catch((err) => {
    log("err", `Speech audio clock playback failed: ${err instanceof Error ? err.message : String(err)}`);
    playAudioBase64WithCallback(base64, sampleRate, () => {
      speechAudioEnded = true;
      speechVideoEnded = true;
      maybeSendSpeechPlaybackAck(turnId);
    });
  });
}

function displaySpeechFrame(index: number): void {
  if (fmp4ModeExpected || fmp4InitSeen || fmp4MseReady) return;
  const img = speechFrameImages[index];
  if (!img) return;
  avatarFrame.innerHTML = "";
  avatarFrame.appendChild(img);
  const w = img.naturalWidth || 1280;
  const h = img.naturalHeight || 720;
  const sizeKb = Math.round((img.src.length * 0.75) / 1024);
  frameMeta.textContent = `image/jpeg | ${w}×${h} | ~${sizeKb}KB | frame ${index + 1}/${speechFrameImages.length}`;
}

function maybeSendSpeechPlaybackAck(turnId: number): void {
  if (speechAckSentForTurnId === turnId) return;
  if (!speechAudioEnded) return;
  if (!speechVideoEnded && !speechStreamEnded) return;

  speechAckSentForTurnId = turnId;

  sendTelemetry("avatar_speech_ack_sent_final", {
    turnId,
    speechAudioEnded,
    speechVideoEnded,
    speechStreamEnded,
    speechDisplayedFrames,
    speechUnderrunCount,
    totalBufferedFrames: speechFrameImages.length,
    expectedTotalFrames: speechExpectedTotalFrames,
    renderedAudioDurationMs: speechRenderedDurationMs,
    originalAudioDurationMs: speechOriginalDurationMs,
  });

  sendTelemetry("avatar_speech_clock_playback_completed", {
    turnId,
    speechAudioEnded,
    speechVideoEnded,
    speechStreamEnded,
    speechDisplayedFrames,
    speechUnderrunCount,
    totalBufferedFrames: speechFrameImages.length,
    expectedTotalFrames: speechExpectedTotalFrames,
    renderedAudioDurationMs: speechRenderedDurationMs,
    originalAudioDurationMs: speechOriginalDurationMs,
  });

  avatarFrame.classList.remove("avatar-speaking");
  avatarSpeechActive = false;
  setSpeechStatus("Avatar speech complete", "#58a6ff");

  if (ws && ws.readyState === WebSocket.OPEN) {
    const ackMsg = JSON.stringify({ type: "avatar.speech.playback.ended", turnId });
    ws.send(ackMsg);
    log("send", ackMsg);
  }
}

function stopSpeechVideoAnimation(): void {
  if (speechAnimationRafId !== null) {
    cancelAnimationFrame(speechAnimationRafId);
    speechAnimationRafId = null;
  }
}

function stopSpeechPlaybackAll(): void {
  stopSpeechVideoAnimation();
  if (speechAudioPlayback) {
    speechAudioPlayback.stop();
    speechAudioPlayback = null;
  }
}

function startIdleAnimation(): void {
  if (avatarSpeechActive || idleFrameImages.length === 0) return;
  stopIdleAnimation();

  idleFrameIndex = 0;
  avatarFrame.classList.add("avatar-idle");
  avatarFrame.classList.remove("avatar-speaking");

  if (idleIsLive) {
    log("info", `Live idle animation running (${idleFrameImages.length} frames buffered)`);
    sendTelemetry("avatar_idle_live_animation_started", { bufferedFrames: idleFrameImages.length });
  } else {
    log("info", "Batch idle loop running");
    sendTelemetry("avatar_idle_loop_started", { frames: idleFrameImages.length });
  }

  const intervalMs = 1000 / idleFps;
  displayIdleFrame(0);

  idleAnimationTimer = window.setInterval(() => {
    idleFrameIndex++;
    if (idleIsLive) {
      if (idleFrameIndex >= idleFrameImages.length) {
        idleUnderrunCountSinceReport++;
        idleFrameIndex = Math.max(0, idleFrameImages.length - 1);
        return;
      }
      displayIdleFrame(idleFrameIndex);
    } else {
      idleFrameIndex = idleFrameIndex % idleFrameImages.length;
      displayIdleFrame(idleFrameIndex);
    }
  }, intervalMs);
}

function stopIdleAnimation(): void {
  if (idleAnimationTimer !== null) {
    clearInterval(idleAnimationTimer);
    idleAnimationTimer = null;
  }
  avatarFrame.classList.remove("avatar-idle");
}

function displayIdleFrame(index: number): void {
  if (fmp4ModeExpected || fmp4InitSeen || fmp4MseReady) return;
  if (index >= idleFrameImages.length) return;
  const img = idleFrameImages[index];
  if (!img) return;
  avatarFrame.innerHTML = "";
  avatarFrame.appendChild(img);
  const w = img.naturalWidth || 1280;
  const h = img.naturalHeight || 720;
  if (idleIsLive) {
    frameMeta.textContent = `image/jpeg | ${w}×${h} | idle live ${index + 1}/${idleFrameImages.length}`;
  } else {
    frameMeta.textContent = `image/jpeg | ${w}×${h} | idle frame ${(index % idleFrameImages.length) + 1}/${idleFrameImages.length}`;
  }
}

sendTelemetry("page_loaded", {
  userAgent: navigator.userAgent.substring(0, 200),
});

connectBtn.addEventListener("click", connect);
stopBtn.addEventListener("click", stop);
startMicBtn.addEventListener("click", startMic);
stopMicBtn.addEventListener("click", stopMic);

if (SFU_MODE || DIRECT_WEBRTC_DEBUG_MODE || WHEP_MODE) {
  window.addEventListener("beforeunload", () => {
    sfuCleanup();
  });
  const sfuDiag = document.getElementById("sfuDiag");
  if (sfuDiag) sfuDiag.style.display = "block";
  if (SFU_MODE) {
    log("info", "SFU mode active (?transport=webrtc_sfu) — video via WebRTC SFU, audio via WebSocket");
  } else {
    log("info", "Direct WebRTC debug mode active (?transport=direct_webrtc_debug) — video direct to Modal, audio via WebSocket");
  }
}

if (SFU_DEBUG) {
  createSfuDiagPanel();
  (window as unknown as Record<string, unknown>).__sfuDiag = sfuDiagState;
  (window as unknown as Record<string, unknown>).__dumpSfuDiag = (): string =>
    JSON.stringify(sfuDiagState, null, 2);
  log("info", "SFU debug diagnostics active (?sfu_debug=1) — passive observers only");
}
