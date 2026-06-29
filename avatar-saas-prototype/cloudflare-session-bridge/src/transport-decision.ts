// ── Transport Decision (pure, no Workers APIs) ──────────────────────────────
//
// Determines whether a session should attempt the Cloudflare Realtime SFU
// transport or stay on the default fMP4-over-WebSocket path.  This is a
// pure function so it can be unit-tested without instantiating a DO.

export type TransportPreference = "auto" | "webrtc_sfu" | "websocket_fmp4";

export interface TransportEnvLike {
  AVTR_REALTIME_SFU_ENABLED?: string;
  CALLS_APP_ID?: string;
  CALLS_APP_SECRET?: string;
  MODAL_WORKER_SECRET?: string;
  WORKER_BASE_URL?: string;
  AVTR_REALTIME_SFU_DEFAULT?: string;
}

export interface TransportDecision {
  shouldTrySfu: boolean;
  fallbackAllowed: boolean;
}

export function isSfuFullyConfigured(env: TransportEnvLike): boolean {
  return (
    env.AVTR_REALTIME_SFU_ENABLED === "true" &&
    !!env.CALLS_APP_ID &&
    !!env.CALLS_APP_SECRET &&
    !!env.MODAL_WORKER_SECRET &&
    !!env.WORKER_BASE_URL
  );
}

export function resolveTransportDecision(
  env: TransportEnvLike,
  pref: TransportPreference | undefined,
  modal4cAvailable: boolean,
): TransportDecision {
  const transportPref = pref ?? "auto";

  if (transportPref === "websocket_fmp4") {
    return { shouldTrySfu: false, fallbackAllowed: false };
  }

  const sfuReady = isSfuFullyConfigured(env) && modal4cAvailable;

  if (transportPref === "webrtc_sfu") {
    // Explicit request — try SFU if fully ready, but NEVER fallback.
    // Caller must return sfu_not_configured when shouldTrySfu is false.
    return { shouldTrySfu: sfuReady, fallbackAllowed: false };
  }

  // "auto" or omitted
  // Safety guard: AVTR_REALTIME_SFU_DEFAULT=websocket_fmp4 prevents auto→SFU
  // even when fully configured. Only explicit transport:"webrtc_sfu" attempts SFU.
  if (sfuReady && env.AVTR_REALTIME_SFU_DEFAULT !== "websocket_fmp4") {
    return { shouldTrySfu: true, fallbackAllowed: true };
  }
  return { shouldTrySfu: false, fallbackAllowed: false };
}
