import { describe, it, expect, beforeEach } from "vitest";
import { env } from "cloudflare:workers";
import {
  StaticCatalogProvider,
  D1CatalogProvider,
  SYSTEM_OWNER_ID,
  getCatalogProvider,
} from "../src/catalog-provider";
import type { Env } from "../src/types";

// ── Helpers ────────────────────────────────────────────────────────────────

const db = env.CATALOG_DB!;

async function clearTables(): Promise<void> {
  await db.prepare("DELETE FROM catalog_avatars").run();
  await db.prepare("DELETE FROM catalog_backgrounds").run();
}

async function insertAvatar(
  owner: string,
  id: string,
  name: string,
  available = true,
  bgId = "",
): Promise<void> {
  await db
    .prepare(
      "INSERT INTO catalog_avatars (owner_tenant_id, avatar_id, display_name, available, default_bg_id) VALUES (?1, ?2, ?3, ?4, ?5)",
    )
    .bind(owner, id, name, available ? 1 : 0, bgId)
    .run();
}

async function insertBackground(
  owner: string,
  id: string,
  name: string,
  available = true,
): Promise<void> {
  await db
    .prepare(
      "INSERT INTO catalog_backgrounds (owner_tenant_id, bg_id, display_name, available) VALUES (?1, ?2, ?3, ?4)",
    )
    .bind(owner, id, name, available ? 1 : 0)
    .run();
}

// ── StaticCatalogProvider ──────────────────────────────────────────────────

describe("StaticCatalogProvider", () => {
  it("returns hardcoded defaults (arnold + aloe_window)", async () => {
    const p = new StaticCatalogProvider();
    const avatars = await p.listAvatars("any");
    expect(avatars).toHaveLength(1);
    expect(avatars[0].avatar_id).toBe("arnold");

    const backgrounds = await p.listBackgrounds("any");
    expect(backgrounds).toHaveLength(1);
    expect(backgrounds[0].bg_id).toBe("aloe_window");
  });

  it("overlay extends defaults without removing them", async () => {
    const p = new StaticCatalogProvider(
      [{ avatar_id: "sarah", display_name: "Sarah", available: true, default_bg_id: "office" }],
      [{ bg_id: "office", display_name: "Office", available: true }],
    );
    const avatars = await p.listAvatars("any");
    expect(avatars).toHaveLength(2);
    expect(avatars.some((a) => a.avatar_id === "arnold")).toBe(true);
    expect(avatars.some((a) => a.avatar_id === "sarah")).toBe(true);

    const backgrounds = await p.listBackgrounds("any");
    expect(backgrounds).toHaveLength(2);
  });

  it("overlay can shadow default with available=false", async () => {
    const p = new StaticCatalogProvider(
      [{ avatar_id: "arnold", display_name: "Arnold (disabled)", available: false, default_bg_id: "aloe_window" }],
    );
    const avatars = await p.listAvatars("any");
    expect(avatars).toHaveLength(0);

    const detail = await p.getAvatar("any", "arnold");
    expect(detail).toBeNull();
  });

  it("filters unavailable from list and getAvatar", async () => {
    const p = new StaticCatalogProvider(
      [{ avatar_id: "private", display_name: "Private", available: false, default_bg_id: "" }],
    );
    const avatars = await p.listAvatars("any");
    expect(avatars.find((a) => a.avatar_id === "private")).toBeUndefined();

    const detail = await p.getAvatar("any", "private");
    expect(detail).toBeNull();
  });

  it("default_bg_id is always a string", async () => {
    const p = new StaticCatalogProvider();
    const detail = await p.getAvatar("any", "arnold");
    expect(detail).not.toBeNull();
    expect(typeof detail!.default_bg_id).toBe("string");
  });
});

// ── D1CatalogProvider ─────────────────────────────────────────────────────

describe("D1CatalogProvider", () => {
  let provider: D1CatalogProvider;

  beforeEach(async () => {
    await clearTables();
    provider = new D1CatalogProvider(db, new StaticCatalogProvider());
  });

  it("returns static defaults when D1 is empty", async () => {
    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.some((a) => a.avatar_id === "arnold")).toBe(true);

    const detail = await provider.getAvatar("tenant_a", "arnold");
    expect(detail).not.toBeNull();
    expect(detail!.avatar_id).toBe("arnold");
  });

  it("returns system seed after seeding", async () => {
    await insertAvatar(SYSTEM_OWNER_ID, "arnold", "Arnold", true, "aloe_window");
    await insertBackground(SYSTEM_OWNER_ID, "aloe_window", "Aloe Window", true);

    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.some((a) => a.avatar_id === "arnold")).toBe(true);

    const backgrounds = await provider.listBackgrounds("tenant_a");
    expect(backgrounds.some((b) => b.bg_id === "aloe_window")).toBe(true);
  });

  it("tenant-owned asset visible to owning tenant", async () => {
    await insertAvatar("tenant_a", "custom_avatar", "Custom", true, "");

    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.some((a) => a.avatar_id === "custom_avatar")).toBe(true);

    const detail = await provider.getAvatar("tenant_a", "custom_avatar");
    expect(detail).not.toBeNull();
    expect(detail!.display_name).toBe("Custom");
  });

  it("tenant-owned asset not visible to other tenant", async () => {
    await insertAvatar("tenant_a", "custom_avatar", "Custom", true, "");

    const avatars = await provider.listAvatars("tenant_b");
    expect(avatars.find((a) => a.avatar_id === "custom_avatar")).toBeUndefined();

    const detail = await provider.getAvatar("tenant_b", "custom_avatar");
    expect(detail).toBeNull();
  });

  it("tenant row shadows system row in list and detail", async () => {
    await insertAvatar(SYSTEM_OWNER_ID, "shared", "System Shared", true, "");
    await insertAvatar("tenant_a", "shared", "Tenant A Shared", true, "");

    const avatarsA = await provider.listAvatars("tenant_a");
    const sharedA = avatarsA.find((a) => a.avatar_id === "shared");
    expect(sharedA).toBeDefined();
    expect(sharedA!.display_name).toBe("Tenant A Shared");

    const detailA = await provider.getAvatar("tenant_a", "shared");
    expect(detailA!.display_name).toBe("Tenant A Shared");

    const avatarsB = await provider.listAvatars("tenant_b");
    const sharedB = avatarsB.find((a) => a.avatar_id === "shared");
    expect(sharedB).toBeDefined();
    expect(sharedB!.display_name).toBe("System Shared");
  });

  it("unavailable system asset hides default in list and detail", async () => {
    await insertAvatar(SYSTEM_OWNER_ID, "arnold", "Arnold (disabled)", false, "aloe_window");

    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.find((a) => a.avatar_id === "arnold")).toBeUndefined();

    const detail = await provider.getAvatar("tenant_a", "arnold");
    expect(detail).toBeNull();
  });

  it("unavailable tenant asset excluded and returns null", async () => {
    await insertAvatar("tenant_a", "disabled_avatar", "Disabled", false, "");

    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.find((a) => a.avatar_id === "disabled_avatar")).toBeUndefined();

    const detail = await provider.getAvatar("tenant_a", "disabled_avatar");
    expect(detail).toBeNull();
  });

  it("D1 query error falls back to static defaults", async () => {
    await db.prepare("DROP TABLE IF EXISTS catalog_avatars").run();

    const avatars = await provider.listAvatars("tenant_a");
    expect(avatars.some((a) => a.avatar_id === "arnold")).toBe(true);

    const detail = await provider.getAvatar("tenant_a", "arnold");
    expect(detail).not.toBeNull();

    // Recreate for subsequent tests
    await db
      .prepare(
        `CREATE TABLE IF NOT EXISTS catalog_avatars (
          owner_tenant_id TEXT NOT NULL,
          avatar_id       TEXT NOT NULL,
          display_name    TEXT NOT NULL,
          available       INTEGER NOT NULL DEFAULT 1,
          default_bg_id   TEXT NOT NULL DEFAULT '',
          preview_url     TEXT,
          created_at      TEXT NOT NULL DEFAULT (datetime('now')),
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          PRIMARY KEY (owner_tenant_id, avatar_id)
        )`,
      )
      .run();
  });

  it("default_bg_id is always a string from D1", async () => {
    await insertAvatar(SYSTEM_OWNER_ID, "test_bg", "Test", true, "some_bg");

    const detail = await provider.getAvatar("tenant_a", "test_bg");
    expect(detail).not.toBeNull();
    expect(typeof detail!.default_bg_id).toBe("string");
    expect(detail!.default_bg_id).toBe("some_bg");
  });

  it("backgrounds: tenant isolation and shadowing", async () => {
    await insertBackground(SYSTEM_OWNER_ID, "office", "Office", true);
    await insertBackground("tenant_a", "studio", "Studio", true);

    // Tenant A sees system + own
    const bgsA = await provider.listBackgrounds("tenant_a");
    expect(bgsA.some((b) => b.bg_id === "office")).toBe(true);
    expect(bgsA.some((b) => b.bg_id === "studio")).toBe(true);

    // Tenant B sees system only
    const bgsB = await provider.listBackgrounds("tenant_b");
    expect(bgsB.some((b) => b.bg_id === "office")).toBe(true);
    expect(bgsB.find((b) => b.bg_id === "studio")).toBeUndefined();
  });
});

// ── Factory: getCatalogProvider ────────────────────────────────────────────

describe("getCatalogProvider factory", () => {
  it("returns StaticCatalogProvider when CATALOG_DB absent", () => {
    const mockEnv = {} as Env;
    const provider = getCatalogProvider(mockEnv);
    expect(provider.name).toBe("static");
  });

  it("returns D1CatalogProvider when CATALOG_DB present", () => {
    const mockEnv = { CATALOG_DB: db } as Env;
    const provider = getCatalogProvider(mockEnv);
    expect(provider.name).toBe("d1");
  });
});
