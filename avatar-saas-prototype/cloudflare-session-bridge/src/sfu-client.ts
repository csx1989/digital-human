// ── Cloudflare Realtime SFU Client ──────────────────────────────────────────
//
// All Cloudflare Realtime SFU HTTPS API calls live here. The Worker/DO is the
// sole component that holds CALLS_APP_SECRET and CALLS_APP_ID. Neither the
// browser nor Modal ever call these functions directly.
//
// API reference: https://developers.cloudflare.com/realtime/sfu/https-api/
//
// Endpoints used:
//   POST /apps/{appId}/sessions/new           — create session
//   POST /apps/{appId}/sessions/{id}/tracks/new — add tracks (with or without SDP)
//   PUT  /apps/{appId}/sessions/{id}/renegotiate — send answer SDP
//   PUT  /apps/{appId}/sessions/{id}/tracks/close — close tracks (force:true)

import type { Env } from "./types";

const SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps";

export class SfuClientError extends Error {
  status: number;
  body: string;
  constructor(message: string, status: number, body: string) {
    super(message);
    this.name = "SfuClientError";
    this.status = status;
    this.body = body;
  }
}

function requireCreds(env: Env): { appId: string; appSecret: string } {
  if (!env.CALLS_APP_ID || !env.CALLS_APP_SECRET) {
    throw new SfuClientError("SFU credentials not configured", 503, "");
  }
  return { appId: env.CALLS_APP_ID, appSecret: env.CALLS_APP_SECRET };
}

async function sfuFetch(
  env: Env,
  method: string,
  path: string,
  body?: unknown,
): Promise<{ status: number; data: Record<string, unknown> }> {
  const { appId, appSecret } = requireCreds(env);
  const url = `${SFU_API_BASE}/${appId}${path}`;
  const resp = await fetch(url, {
    method,
    headers: {
      Authorization: `Bearer ${appSecret}`,
      "Content-Type": "application/json",
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await resp.text();
  let data: Record<string, unknown>;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { raw: text };
  }
  return { status: resp.status, data };
}

// ── Create SFU Session ──────────────────────────────────────────────────────

export async function createSfuSession(env: Env): Promise<string> {
  const { status, data } = await sfuFetch(env, "POST", "/sessions/new");
  if (status !== 200 && status !== 201) {
    throw new SfuClientError(
      `SFU session creation failed: ${status}`,
      status,
      JSON.stringify(data),
    );
  }
  const sessionId = data.sessionId as string;
  if (!sessionId) {
    throw new SfuClientError("SFU session creation returned no sessionId", 502, JSON.stringify(data));
  }
  return sessionId;
}

// ── Tracks New (publisher — local, with SDP offer) ──────────────────────────

export interface PublisherOfferParams {
  sfuSessionId: string;
  sdp: string;
  sdpType: string;
  mid: string;
  trackName: string;
  kind?: string;
}

export interface PublisherOfferResult {
  answerSdp: string;
  answerType: string;
  mid: string;
  requiresRenegotiation: boolean;
}

export async function publisherTracksNew(
  env: Env,
  params: PublisherOfferParams,
): Promise<PublisherOfferResult> {
  const body = {
    sessionDescription: { sdp: params.sdp, type: params.sdpType },
    tracks: [{
      location: "local",
      mid: params.mid,
      trackName: params.trackName,
      ...(params.kind ? { kind: params.kind } : {}),
    }],
  };
  const { status, data } = await sfuFetch(
    env,
    "POST",
    `/sessions/${params.sfuSessionId}/tracks/new`,
    body,
  );
  if (status !== 200) {
    throw new SfuClientError(
      `SFU publisher tracks/new failed: ${status}`,
      status,
      JSON.stringify(data),
    );
  }
  const sd = (data.sessionDescription ?? {}) as Record<string, unknown>;
  return {
    answerSdp: (sd.sdp as string) ?? "",
    answerType: (sd.type as string) ?? "answer",
    mid: params.mid,
    requiresRenegotiation: (data.requiresImmediateRenegotiation as boolean) ?? false,
  };
}

// ── Tracks New (subscriber — remote, no SDP) ────────────────────────────────

export interface SubscriberSubscribeParams {
  sfuSessionId: string;
  publisherSfuSessionId: string;
  trackName: string;
}

export interface SubscriberSubscribeResult {
  offerSdp: string;
  offerType: string;
  requiresRenegotiation: boolean;
  mids: string[];
}

export async function subscriberTracksNew(
  env: Env,
  params: SubscriberSubscribeParams,
): Promise<SubscriberSubscribeResult> {
  const body = {
    tracks: [{
      location: "remote",
      sessionId: params.publisherSfuSessionId,
      trackName: params.trackName,
    }],
  };
  const { status, data } = await sfuFetch(
    env,
    "POST",
    `/sessions/${params.sfuSessionId}/tracks/new`,
    body,
  );
  if (status !== 200) {
    throw new SfuClientError(
      `SFU subscriber tracks/new failed: ${status}`,
      status,
      JSON.stringify(data),
    );
  }
  const sd = (data.sessionDescription ?? {}) as Record<string, unknown>;
  const tracks = (data.tracks ?? []) as Array<Record<string, unknown>>;
  return {
    offerSdp: (sd.sdp as string) ?? "",
    offerType: (sd.type as string) ?? "offer",
    requiresRenegotiation: (data.requiresImmediateRenegotiation as boolean) ?? false,
    mids: tracks.map((t) => (t.mid as string) ?? ""),
  };
}

// ── Renegotiate (subscriber answer) ─────────────────────────────────────────

export async function sfuRenegotiate(
  env: Env,
  sfuSessionId: string,
  sdp: string,
  sdpType: string,
): Promise<void> {
  const body = {
    sessionDescription: { sdp, type: sdpType },
  };
  const { status, data } = await sfuFetch(
    env,
    "PUT",
    `/sessions/${sfuSessionId}/renegotiate`,
    body,
  );
  if (status !== 200) {
    throw new SfuClientError(
      `SFU renegotiate failed: ${status}`,
      status,
      JSON.stringify(data),
    );
  }
}

// ── Tracks Close ────────────────────────────────────────────────────────────

export interface SfuCloseResult {
  status: number;
  requiresRenegotiation: boolean;
}

export async function sfuTracksClose(
  env: Env,
  sfuSessionId: string,
  mids: string[],
  force: boolean = true,
): Promise<SfuCloseResult> {
  if (mids.length === 0) return { status: 200, requiresRenegotiation: false };
  const body = {
    tracks: mids.map((mid) => ({ mid })),
    force,
  };
  const { status, data } = await sfuFetch(
    env,
    "PUT",
    `/sessions/${sfuSessionId}/tracks/close`,
    body,
  );
  if (status !== 200) {
    throw new SfuClientError(
      `SFU tracks/close failed: ${status}`,
      status,
      JSON.stringify(data),
    );
  }
  return {
    status,
    requiresRenegotiation: (data.requiresImmediateRenegotiation as boolean) ?? false,
  };
}

// ── Cleanup: close all known tracks ─────────────────────────────────────────

export async function cleanupSfuTracks(
  env: Env,
  sfuSessionId: string,
  publisherMids: { video: string | null; audio: string | null },
  subscriberMids: string[],
): Promise<void> {
  const allMids = [
    ...Object.values(publisherMids).filter((m): m is string => !!m),
    ...subscriberMids,
  ];
  if (allMids.length === 0) return;
  try {
    await sfuTracksClose(env, sfuSessionId, allMids, true);
  } catch (e) {
    // Best-effort cleanup — log but don't throw
    console.warn("[sfu-client] cleanupSfuTracks best-effort failed:", e instanceof Error ? e.message : String(e));
  }
}
