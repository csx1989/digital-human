-- Phase 6a.2: Seed system assets (idempotent).
-- These rows overlay the hardcoded static defaults (arnold, aloe_window).

INSERT OR IGNORE INTO catalog_avatars
    (owner_tenant_id, avatar_id, display_name, available, default_bg_id)
VALUES
    ('__system__', 'arnold', 'Arnold', 1, 'aloe_window');

INSERT OR IGNORE INTO catalog_backgrounds
    (owner_tenant_id, bg_id, display_name, available)
VALUES
    ('__system__', 'aloe_window', 'Aloe Window', 1);
