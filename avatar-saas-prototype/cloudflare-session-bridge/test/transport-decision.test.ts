import { describe, it, expect } from "vitest";
import {
  isSfuFullyConfigured,
  resolveTransportDecision,
  type TransportEnvLike,
} from "../src/transport-decision";

const FULL_ENV: TransportEnvLike = {
  AVTR_REALTIME_SFU_ENABLED: "true",
  CALLS_APP_ID: "app-123",
  CALLS_APP_SECRET: "secret-abc",
  MODAL_WORKER_SECRET: "modal-secret-xyz",
  WORKER_BASE_URL: "https://test.worker.dev",
};

const FLAG_OFF_ENV: TransportEnvLike = {
  AVTR_REALTIME_SFU_ENABLED: "false",
  CALLS_APP_ID: "app-123",
  CALLS_APP_SECRET: "secret-abc",
  MODAL_WORKER_SECRET: "modal-secret-xyz",
  WORKER_BASE_URL: "https://test.worker.dev",
};

function envWithout(key: keyof TransportEnvLike): TransportEnvLike {
  const copy = { ...FULL_ENV };
  delete copy[key];
  return copy;
}

describe("isSfuFullyConfigured", () => {
  it("returns true when all 5 prerequisites are set", () => {
    expect(isSfuFullyConfigured(FULL_ENV)).toBe(true);
  });

  it("returns false when AVTR_REALTIME_SFU_ENABLED is not 'true'", () => {
    expect(isSfuFullyConfigured(FLAG_OFF_ENV)).toBe(false);
  });

  it("returns false when CALLS_APP_ID is missing", () => {
    expect(isSfuFullyConfigured(envWithout("CALLS_APP_ID"))).toBe(false);
  });

  it("returns false when CALLS_APP_SECRET is missing", () => {
    expect(isSfuFullyConfigured(envWithout("CALLS_APP_SECRET"))).toBe(false);
  });

  it("returns false when MODAL_WORKER_SECRET is missing", () => {
    expect(isSfuFullyConfigured(envWithout("MODAL_WORKER_SECRET"))).toBe(false);
  });

  it("returns false when WORKER_BASE_URL is missing", () => {
    expect(isSfuFullyConfigured(envWithout("WORKER_BASE_URL"))).toBe(false);
  });

  it("returns false when all env vars are empty strings", () => {
    const emptyEnv: TransportEnvLike = {
      AVTR_REALTIME_SFU_ENABLED: "true",
      CALLS_APP_ID: "",
      CALLS_APP_SECRET: "",
      MODAL_WORKER_SECRET: "",
      WORKER_BASE_URL: "",
    };
    expect(isSfuFullyConfigured(emptyEnv)).toBe(false);
  });
});

describe("resolveTransportDecision — websocket_fmp4", () => {
  it("never tries SFU regardless of config", () => {
    expect(resolveTransportDecision(FULL_ENV, "websocket_fmp4", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("never tries SFU even when config is missing", () => {
    expect(resolveTransportDecision(envWithout("WORKER_BASE_URL"), "websocket_fmp4", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });
});

describe("resolveTransportDecision — webrtc_sfu (explicit)", () => {
  it("tries SFU with full config + 4C available", () => {
    expect(resolveTransportDecision(FULL_ENV, "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: true, fallbackAllowed: false });
  });

  it("fails (shouldTrySfu=false) when WORKER_BASE_URL missing — caller returns sfu_not_configured", () => {
    expect(resolveTransportDecision(envWithout("WORKER_BASE_URL"), "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("fails when CALLS_APP_SECRET missing", () => {
    expect(resolveTransportDecision(envWithout("CALLS_APP_SECRET"), "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("fails when MODAL_WORKER_SECRET missing", () => {
    expect(resolveTransportDecision(envWithout("MODAL_WORKER_SECRET"), "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("fails when flag off", () => {
    expect(resolveTransportDecision(FLAG_OFF_ENV, "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("fails when 4C not available", () => {
    expect(resolveTransportDecision(FULL_ENV, "webrtc_sfu", false))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("never allows fallback even with full config", () => {
    const d = resolveTransportDecision(FULL_ENV, "webrtc_sfu", true);
    expect(d.shouldTrySfu).toBe(true);
    expect(d.fallbackAllowed).toBe(false);
  });
});

describe("resolveTransportDecision — auto / omitted", () => {
  it("tries SFU when fully ready + 4C available", () => {
    expect(resolveTransportDecision(FULL_ENV, "auto", true))
      .toEqual({ shouldTrySfu: true, fallbackAllowed: true });
  });

  it("tries SFU when fully ready + 4C available (omitted preference)", () => {
    expect(resolveTransportDecision(FULL_ENV, undefined, true))
      .toEqual({ shouldTrySfu: true, fallbackAllowed: true });
  });

  it("skips SFU silently when WORKER_BASE_URL missing", () => {
    expect(resolveTransportDecision(envWithout("WORKER_BASE_URL"), "auto", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("skips SFU silently when CALLS_APP_ID missing", () => {
    expect(resolveTransportDecision(envWithout("CALLS_APP_ID"), undefined, true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("skips SFU when flag off", () => {
    expect(resolveTransportDecision(FLAG_OFF_ENV, "auto", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("skips SFU when 4C not available", () => {
    expect(resolveTransportDecision(FULL_ENV, "auto", false))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });
});

describe("resolveTransportDecision — AVTR_REALTIME_SFU_DEFAULT safety guard", () => {
  const SAFETY_ON_ENV: TransportEnvLike = {
    ...FULL_ENV,
    AVTR_REALTIME_SFU_DEFAULT: "websocket_fmp4",
  };

  it("auto + fully configured + DEFAULT=websocket_fmp4 → stays fMP4", () => {
    expect(resolveTransportDecision(SAFETY_ON_ENV, "auto", true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("omitted + fully configured + DEFAULT=websocket_fmp4 → stays fMP4", () => {
    expect(resolveTransportDecision(SAFETY_ON_ENV, undefined, true))
      .toEqual({ shouldTrySfu: false, fallbackAllowed: false });
  });

  it("explicit webrtc_sfu + DEFAULT=websocket_fmp4 → still attempts SFU", () => {
    expect(resolveTransportDecision(SAFETY_ON_ENV, "webrtc_sfu", true))
      .toEqual({ shouldTrySfu: true, fallbackAllowed: false });
  });
});
