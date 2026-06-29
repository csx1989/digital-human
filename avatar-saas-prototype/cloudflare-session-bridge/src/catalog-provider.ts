import type { Env, V1Avatar, V1Background } from "./types";

// ── Catalog provider abstraction ───────────────────────────────────────────
//
// Public /v1/avatars and /v1/backgrounds endpoints delegate to a
// CatalogProvider.  Providers are layered in a merge/overlay model:
//
//   Layer 4 (lowest):  Hardcoded defaults (arnold, aloe_window)
//   Layer 3:           AVTR_CATALOG_JSON overlay (env secret)
//   Layer 2:           D1 __system__ rows
//   Layer 1 (highest): D1 tenant-owned rows
//
// Higher layers shadow lower layers by avatar_id / bg_id.
// available=false rows are excluded from list results and cause
// getAvatar / getBackground to return null.
//
// If CATALOG_DB is absent, only layers 4+3 are used (StaticCatalogProvider).
// If a D1 query throws, the provider falls back to the static layers only.

export const SYSTEM_OWNER_ID = "__system__";

export interface CatalogProvider {
  readonly name: string;
  listAvatars(tenantId: string): Promise<readonly V1Avatar[]>;
  listBackgrounds(tenantId: string): Promise<readonly V1Background[]>;
  getAvatar(tenantId: string, avatarId: string): Promise<V1Avatar | null>;
  getBackground(tenantId: string, bgId: string): Promise<V1Background | null>;
}

export interface CatalogValidationError {
  ok: false;
  code: "avatar_not_found" | "background_not_found";
  message: string;
}

export interface CatalogValidationSuccess {
  ok: true;
  avatar: V1Avatar;
  background: V1Background | null;
}

export type CatalogValidationResult = CatalogValidationError | CatalogValidationSuccess;

// ── Default static catalog ─────────────────────────────────────────────────

const DEFAULT_AVATARS: readonly V1Avatar[] = [
  {
    avatar_id: "arnold",
    display_name: "Arnold",
    available: true,
    default_bg_id: "aloe_window",
  },
];

const DEFAULT_BACKGROUNDS: readonly V1Background[] = [
  {
    bg_id: "aloe_window",
    display_name: "Aloe Window",
    available: true,
  },
];

// ── StaticCatalogProvider ──────────────────────────────────────────────────
//
// Starts with hardcoded defaults.  Optional overlay entries (from
// AVTR_CATALOG_JSON) merge on top by ID — they extend or shadow defaults
// but never remove them.

export class StaticCatalogProvider implements CatalogProvider {
  readonly name = "static";

  #avatars: Map<string, V1Avatar>;
  #backgrounds: Map<string, V1Background>;

  constructor(
    overlayAvatars?: readonly V1Avatar[],
    overlayBackgrounds?: readonly V1Background[],
  ) {
    this.#avatars = new Map();
    this.#backgrounds = new Map();

    for (const a of DEFAULT_AVATARS) this.#avatars.set(a.avatar_id, a);
    for (const b of DEFAULT_BACKGROUNDS) this.#backgrounds.set(b.bg_id, b);

    if (overlayAvatars) {
      for (const a of overlayAvatars) this.#avatars.set(a.avatar_id, a);
    }
    if (overlayBackgrounds) {
      for (const b of overlayBackgrounds) this.#backgrounds.set(b.bg_id, b);
    }
  }

  async listAvatars(_tenantId: string): Promise<readonly V1Avatar[]> {
    return [...this.#avatars.values()].filter((a) => a.available);
  }

  async listBackgrounds(_tenantId: string): Promise<readonly V1Background[]> {
    return [...this.#backgrounds.values()].filter((b) => b.available);
  }

  async getAvatar(_tenantId: string, avatarId: string): Promise<V1Avatar | null> {
    const a = this.#avatars.get(avatarId);
    return a && a.available ? a : null;
  }

  async getBackground(_tenantId: string, bgId: string): Promise<V1Background | null> {
    const b = this.#backgrounds.get(bgId);
    return b && b.available ? b : null;
  }
}

// ── D1 row mappers ─────────────────────────────────────────────────────────

interface D1AvatarRow {
  avatar_id: string;
  display_name: string;
  available: number;
  default_bg_id: string | null;
  preview_url: string | null;
}

interface D1BackgroundRow {
  bg_id: string;
  display_name: string;
  available: number;
  preview_url: string | null;
}

function d1RowToAvatar(row: D1AvatarRow): V1Avatar {
  const rawBg = row.default_bg_id;
  const avatar: V1Avatar = {
    avatar_id: row.avatar_id,
    display_name: row.display_name,
    available: !!row.available,
    default_bg_id: typeof rawBg === "string" ? rawBg : "",
  };
  if (typeof row.preview_url === "string" && row.preview_url) {
    avatar.preview_url = row.preview_url;
  }
  return avatar;
}

function d1RowToBackground(row: D1BackgroundRow): V1Background {
  const bg: V1Background = {
    bg_id: row.bg_id,
    display_name: row.display_name,
    available: !!row.available,
  };
  if (typeof row.preview_url === "string" && row.preview_url) {
    bg.preview_url = row.preview_url;
  }
  return bg;
}

// ── D1CatalogProvider ──────────────────────────────────────────────────────
//
// Overlays D1 rows (system + tenant) on top of a static fallback provider.
// All D1 queries are wrapped in try/catch — on error the method degrades
// gracefully to static-only results (fail-open).

export class D1CatalogProvider implements CatalogProvider {
  readonly name = "d1";

  readonly #db: D1Database;
  readonly #fallback: CatalogProvider;

  constructor(db: D1Database, fallback: CatalogProvider) {
    this.#db = db;
    this.#fallback = fallback;
  }

  async listAvatars(tenantId: string): Promise<readonly V1Avatar[]> {
    const merged = new Map<string, V1Avatar>();

    for (const a of await this.#fallback.listAvatars(tenantId)) {
      merged.set(a.avatar_id, a);
    }

    try {
      const result = await this.#db
        .prepare(
          `SELECT avatar_id, display_name, available, default_bg_id, preview_url, owner_tenant_id
           FROM catalog_avatars
           WHERE owner_tenant_id = ?1 OR owner_tenant_id = ?2
           ORDER BY CASE owner_tenant_id WHEN ?2 THEN 0 ELSE 1 END`,
        )
        .bind(tenantId, SYSTEM_OWNER_ID)
        .all<D1AvatarRow & { owner_tenant_id: string }>();

      for (const row of result.results) {
        const avatar = d1RowToAvatar(row);
        merged.set(avatar.avatar_id, avatar);
      }
    } catch (err) {
      console.error("[D1Catalog] listAvatars query failed, using static:", err);
    }

    return [...merged.values()].filter((a) => a.available);
  }

  async listBackgrounds(tenantId: string): Promise<readonly V1Background[]> {
    const merged = new Map<string, V1Background>();

    for (const b of await this.#fallback.listBackgrounds(tenantId)) {
      merged.set(b.bg_id, b);
    }

    try {
      const result = await this.#db
        .prepare(
          `SELECT bg_id, display_name, available, preview_url, owner_tenant_id
           FROM catalog_backgrounds
           WHERE owner_tenant_id = ?1 OR owner_tenant_id = ?2
           ORDER BY CASE owner_tenant_id WHEN ?2 THEN 0 ELSE 1 END`,
        )
        .bind(tenantId, SYSTEM_OWNER_ID)
        .all<D1BackgroundRow & { owner_tenant_id: string }>();

      for (const row of result.results) {
        const bg = d1RowToBackground(row);
        merged.set(bg.bg_id, bg);
      }
    } catch (err) {
      console.error("[D1Catalog] listBackgrounds query failed, using static:", err);
    }

    return [...merged.values()].filter((b) => b.available);
  }

  async getAvatar(tenantId: string, avatarId: string): Promise<V1Avatar | null> {
    try {
      const row = await this.#db
        .prepare(
          `SELECT * FROM (
              SELECT *, 0 AS priority FROM catalog_avatars
              WHERE avatar_id = ?1 AND owner_tenant_id = ?2
              UNION ALL
              SELECT *, 1 AS priority FROM catalog_avatars
              WHERE avatar_id = ?1 AND owner_tenant_id = ?3
            )
            ORDER BY priority
            LIMIT 1`,
        )
        .bind(avatarId, tenantId, SYSTEM_OWNER_ID)
        .first<D1AvatarRow>();

      if (row) {
        const avatar = d1RowToAvatar(row);
        return avatar.available ? avatar : null;
      }
    } catch (err) {
      console.error("[D1Catalog] getAvatar query failed, using static:", err);
    }

    return this.#fallback.getAvatar(tenantId, avatarId);
  }

  async getBackground(tenantId: string, bgId: string): Promise<V1Background | null> {
    try {
      const row = await this.#db
        .prepare(
          `SELECT * FROM (
              SELECT *, 0 AS priority FROM catalog_backgrounds
              WHERE bg_id = ?1 AND owner_tenant_id = ?2
              UNION ALL
              SELECT *, 1 AS priority FROM catalog_backgrounds
              WHERE bg_id = ?1 AND owner_tenant_id = ?3
            )
            ORDER BY priority
            LIMIT 1`,
        )
        .bind(bgId, tenantId, SYSTEM_OWNER_ID)
        .first<D1BackgroundRow>();

      if (row) {
        const bg = d1RowToBackground(row);
        return bg.available ? bg : null;
      }
    } catch (err) {
      console.error("[D1Catalog] getBackground query failed, using static:", err);
    }

    return this.#fallback.getBackground(tenantId, bgId);
  }
}

// ── AVTR_CATALOG_JSON parsing (defensive — never throws) ───────────────────

interface ParsedCatalog {
  avatars?: unknown;
  backgrounds?: unknown;
}

function isV1Avatar(v: unknown): v is V1Avatar {
  return (
    typeof v === "object" &&
    v !== null &&
    typeof (v as Record<string, unknown>).avatar_id === "string" &&
    typeof (v as Record<string, unknown>).display_name === "string" &&
    typeof (v as Record<string, unknown>).available === "boolean" &&
    typeof (v as Record<string, unknown>).default_bg_id === "string"
  );
}

function isV1Background(v: unknown): v is V1Background {
  return (
    typeof v === "object" &&
    v !== null &&
    typeof (v as Record<string, unknown>).bg_id === "string" &&
    typeof (v as Record<string, unknown>).display_name === "string" &&
    typeof (v as Record<string, unknown>).available === "boolean"
  );
}

function parseCatalogJson(raw: string): ParsedCatalog {
  const parsed = JSON.parse(raw);
  if (typeof parsed !== "object" || parsed === null) {
    throw new Error("catalog JSON is not an object");
  }
  return parsed as ParsedCatalog;
}

// ── Factory ────────────────────────────────────────────────────────────────

let cachedProvider: CatalogProvider | null = null;
let cachedSource: string | null = null;

function buildStaticProvider(env: Env): StaticCatalogProvider {
  const source = env.AVTR_CATALOG_JSON ?? "";
  if (!source.trim()) return new StaticCatalogProvider();

  try {
    const parsed = parseCatalogJson(source);
    const avatars = Array.isArray(parsed.avatars)
      ? parsed.avatars.filter(isV1Avatar)
      : [];
    const backgrounds = Array.isArray(parsed.backgrounds)
      ? parsed.backgrounds.filter(isV1Background)
      : [];

    if (avatars.length === 0 && backgrounds.length === 0) {
      console.warn(
        "[catalog] AVTR_CATALOG_JSON produced no valid entries — using hardcoded defaults",
      );
      return new StaticCatalogProvider();
    }

    return new StaticCatalogProvider(avatars, backgrounds);
  } catch (err) {
    console.warn(
      "[catalog] Failed to parse AVTR_CATALOG_JSON — using hardcoded defaults:",
      err instanceof Error ? err.message : String(err),
    );
    return new StaticCatalogProvider();
  }
}

export function getCatalogProvider(env: Env): CatalogProvider {
  const hasD1 = !!env.CATALOG_DB;
  const jsonSource = env.AVTR_CATALOG_JSON ?? "";
  const cacheKey = `${hasD1 ? "d1" : "static"}:${jsonSource}`;

  if (cachedProvider && cachedSource === cacheKey) {
    return cachedProvider;
  }

  const staticProvider = buildStaticProvider(env);

  const provider: CatalogProvider = hasD1
    ? new D1CatalogProvider(env.CATALOG_DB!, staticProvider)
    : staticProvider;

  cachedProvider = provider;
  cachedSource = cacheKey;
  return provider;
}

// ── Start-time validation helper ───────────────────────────────────────────

export async function validateAvatarBgForStart(
  env: Env,
  tenantId: string,
  avatarId: string,
  bgId: string | null | undefined,
): Promise<CatalogValidationResult> {
  const provider = getCatalogProvider(env);

  const avatar = await provider.getAvatar(tenantId, avatarId);
  if (!avatar) {
    return {
      ok: false,
      code: "avatar_not_found",
      message: `Avatar "${avatarId}" is not available.`,
    };
  }

  if (!avatar.available) {
    return {
      ok: false,
      code: "avatar_not_found",
      message: `Avatar "${avatarId}" is currently unavailable.`,
    };
  }

  if (bgId !== null && bgId !== undefined && bgId !== "") {
    const background = await provider.getBackground(tenantId, bgId);
    if (!background) {
      return {
        ok: false,
        code: "background_not_found",
        message: `Background "${bgId}" is not available.`,
      };
    }

    if (!background.available) {
      return {
        ok: false,
        code: "background_not_found",
        message: `Background "${bgId}" is currently unavailable.`,
      };
    }

    return { ok: true, avatar, background };
  }

  return { ok: true, avatar, background: null };
}
