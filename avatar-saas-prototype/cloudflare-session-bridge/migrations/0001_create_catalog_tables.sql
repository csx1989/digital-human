-- Phase 6a.2: Catalog metadata tables for D1-backed provider.
--
-- Primary keys use a non-null owner_tenant_id column.
-- System/public assets use the reserved value '__system__'.
-- Tenant-owned assets use the real tenant ID.
-- Tenant IDs must never equal '__system__'.

CREATE TABLE IF NOT EXISTS catalog_avatars (
    owner_tenant_id TEXT NOT NULL,
    avatar_id       TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    available       INTEGER NOT NULL DEFAULT 1,
    default_bg_id   TEXT NOT NULL DEFAULT '',
    preview_url     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (owner_tenant_id, avatar_id)
);

CREATE TABLE IF NOT EXISTS catalog_backgrounds (
    owner_tenant_id TEXT NOT NULL,
    bg_id           TEXT NOT NULL,
    display_name    TEXT NOT NULL,
    available       INTEGER NOT NULL DEFAULT 1,
    preview_url     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (owner_tenant_id, bg_id)
);

CREATE INDEX IF NOT EXISTS idx_avatars_owner    ON catalog_avatars(owner_tenant_id);
CREATE INDEX IF NOT EXISTS idx_backgrounds_owner ON catalog_backgrounds(owner_tenant_id);
