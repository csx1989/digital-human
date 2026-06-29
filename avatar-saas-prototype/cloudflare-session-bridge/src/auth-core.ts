// ── Pure auth logic ─────────────────────────────────────────────────────────
//
// Zero runtime dependencies.  Importable from both the Worker and offline
// test scripts (npx tsx).  No caching, no HTTP, no Workers APIs.
// api-v1.ts wraps these with a source-keyed cache and HTTP Response
// construction.

export interface MetadataEntry {
  key: string;
  tenant_id: string;
  name: string | null;
  plan: string | null;
  // Phase 3F: per-key SFU private beta authorization.
  sfu_beta?: boolean;
}

export interface AuthResolveResult {
  tenant_id: string;
  key_name: string | null;
  plan: string | null;
  auth_mode: "metadata" | "legacy";
  sfu_beta?: boolean;
}

export function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return diff === 0;
}

export function parseLegacyKeyString(raw: string | undefined): string[] {
  const s = raw?.trim() ?? "";
  if (!s) return [];
  return s
    .split(",")
    .map((k) => k.trim())
    .filter(Boolean);
}

// Returns:
//   null                        — raw is missing/empty (no metadata configured)
//   null                        — raw is present but invalid JSON
//   { entries: [], dropped: N } — valid JSON, zero accepted entries
//   { entries: [...], dropped } — valid JSON with accepted entries
//
// api-v1.ts converts entries.length === 0 to null for auth-config checks
// (zero valid entries === no metadata configured).

export function parseMetadataString(
  raw: string | undefined,
): { entries: MetadataEntry[]; dropped: number } | null {
  const s = raw?.trim() ?? "";
  if (!s) return null;

  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(s) as Record<string, unknown>;
  } catch {
    return null;
  }

  const entries: MetadataEntry[] = [];
  let dropped = 0;

  for (const [key, val] of Object.entries(parsed)) {
    if (typeof val !== "object" || val === null) {
      dropped++;
      continue;
    }
    const meta = val as { tenant_id?: unknown; name?: unknown; plan?: unknown; sfu_beta?: unknown };
    if (typeof meta.tenant_id !== "string" || meta.tenant_id.trim() === "") {
      dropped++;
      continue;
    }
    entries.push({
      key,
      tenant_id: meta.tenant_id.trim(),
      name: typeof meta.name === "string" ? meta.name : null,
      plan: typeof meta.plan === "string" ? meta.plan : null,
      sfu_beta: meta.sfu_beta === true,
    });
  }

  return { entries, dropped };
}

// Metadata checked first — metadata mode wins per design.
// Returns null when the key matches neither source.

export function resolveApiKey(
  requestKey: string,
  legacyKeys: string[],
  metadataEntries: MetadataEntry[] | null,
): AuthResolveResult | null {
  if (metadataEntries) {
    for (const entry of metadataEntries) {
      if (safeEqual(entry.key, requestKey)) {
        return {
          tenant_id: entry.tenant_id,
          key_name: entry.name,
          plan: entry.plan,
          auth_mode: "metadata",
          sfu_beta: entry.sfu_beta === true,
        };
      }
    }
  }

  for (const k of legacyKeys) {
    if (safeEqual(k, requestKey)) {
      return {
        tenant_id: "legacy",
        key_name: null,
        plan: null,
        auth_mode: "legacy",
      };
    }
  }

  return null;
}
