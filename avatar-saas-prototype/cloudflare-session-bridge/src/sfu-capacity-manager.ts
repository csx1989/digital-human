import { DurableObject } from "cloudflare:workers";

// ── SfuCapacityManager ────────────────────────────────────────────────────
//
// Singleton Durable Object that enforces single-GPU SFU capacity.
// Only one active webrtc_sfu publisher is allowed at a time across all
// session DOs. This prevents GPU contention / CUDA illegal memory access
// when two SFU publishers overlap on the same Modal L40S container.
//
// Lifecycle:
//   acquire(sessionId) → { ok: true } or { ok: false, conflict: sid }
//   release(sessionId) → void
//   getStatus() → { activeCount, activeSessions, leases }
//
// Stale-lock protection:
//   Each lease stores acquiredAt. If a lease is older than LEASE_TTL_MS,
//   acquire() treats it as expired and deletes it before granting the
//   slot to a new session. This self-heals from Worker/DO crashes that
//   leave a lease acquired but never released.
//
// fMP4 sessions never acquire a capacity slot — only webrtc_sfu does.

const STORAGE_KEY_PREFIX = "sfu_active:";
const CAPACITY = 1;

// 15 minutes. Long enough to outlast any Modal /end call, short enough
// to recover from a crashed Worker that holds the slot indefinitely.
// Tunable later via env if needed.
const LEASE_TTL_MS = 15 * 60 * 1000;

interface LeaseRow {
  acquiredAt: number;
}

interface StatusLease {
  sessionId: string;
  acquiredAt: number;
  ageMs: number;
  expiresAt: number;
  expired: boolean;
}

export class SfuCapacityManager extends DurableObject {
  async fetch(request: Request): Promise<Response> {
    const url = new URL(request.url);
    const op = url.searchParams.get("op");

    if (op === "acquire" && request.method === "POST") {
      return this.#handleAcquire(request);
    }
    if (op === "release" && request.method === "POST") {
      return this.#handleRelease(request);
    }
    if (op === "status" && request.method === "GET") {
      return this.#handleStatus();
    }
    if (op === "clear" && request.method === "POST") {
      return this.#handleClear();
    }
    return new Response(JSON.stringify({ error: "unknown op" }), { status: 400 });
  }

  async #handleAcquire(request: Request): Promise<Response> {
    const body = await request.json() as { sessionId?: string };
    const sessionId = body.sessionId;
    if (!sessionId) {
      return new Response(JSON.stringify({ ok: false, error: "sessionId required" }), { status: 400 });
    }

    const now = Date.now();

    // Sweep expired leases before evaluating capacity. This is the
    // self-healing path: a lease whose Worker crashed before releasing
    // is reaped here so a new session can acquire the slot.
    const active = await this.#getActiveLeases(now);
    const live = active.filter((l) => !l.expired);
    const expired = active.filter((l) => l.expired);

    for (const e of expired) {
      await this.ctx.storage.delete(`${STORAGE_KEY_PREFIX}${e.sessionId}`);
    }

    // If this session already holds the slot (live), allow re-acquire (idempotent)
    if (live.some((l) => l.sessionId === sessionId)) {
      return new Response(JSON.stringify({ ok: true, reacquired: true }));
    }

    // Check capacity against live (non-expired) leases only
    if (live.length >= CAPACITY) {
      return new Response(JSON.stringify({
        ok: false,
        error: "sfu_capacity_exceeded",
        conflict: live[0].sessionId,
        capacity: CAPACITY,
      }), { status: 409 });
    }

    // Acquire
    await this.ctx.storage.put(`${STORAGE_KEY_PREFIX}${sessionId}`, {
      acquiredAt: now,
    } satisfies LeaseRow);

    return new Response(JSON.stringify({ ok: true }));
  }

  async #handleRelease(request: Request): Promise<Response> {
    const body = await request.json() as { sessionId?: string };
    const sessionId = body.sessionId;
    if (!sessionId) {
      return new Response(JSON.stringify({ ok: false, error: "sessionId required" }), { status: 400 });
    }

    await this.ctx.storage.delete(`${STORAGE_KEY_PREFIX}${sessionId}`);
    return new Response(JSON.stringify({ ok: true }));
  }

  async #handleStatus(): Promise<Response> {
    const leases = await this.#getActiveLeases(Date.now());
    return new Response(JSON.stringify({
      activeCount: leases.filter((l) => !l.expired).length,
      activeSessions: leases.filter((l) => !l.expired).map((l) => l.sessionId),
      capacity: CAPACITY,
      leaseTtlMs: LEASE_TTL_MS,
      leases: leases,
    }));
  }

  async #getActiveLeases(now: number): Promise<StatusLease[]> {
    const list = await this.ctx.storage.list<LeaseRow>({ prefix: STORAGE_KEY_PREFIX });
    const out: StatusLease[] = [];
    for (const [key, val] of list.entries()) {
      const sessionId = key.slice(STORAGE_KEY_PREFIX.length);
      const acquiredAt = val?.acquiredAt ?? now;
      const ageMs = now - acquiredAt;
      const expiresAt = acquiredAt + LEASE_TTL_MS;
      out.push({
        sessionId,
        acquiredAt,
        ageMs,
        expiresAt,
        expired: ageMs > LEASE_TTL_MS,
      });
    }
    return out;
  }

  async #handleClear(): Promise<Response> {
    const list = await this.ctx.storage.list({ prefix: STORAGE_KEY_PREFIX });
    for (const key of list.keys()) {
      await this.ctx.storage.delete(key);
    }
    return new Response(JSON.stringify({ ok: true, cleared: list.size }));
  }
}
