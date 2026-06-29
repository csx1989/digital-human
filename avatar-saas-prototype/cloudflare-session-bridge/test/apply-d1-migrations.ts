import { applyD1Migrations, env } from "cloudflare:test";

const migrations = (
  env as unknown as {
    TEST_MIGRATIONS: Array<{ name: string; queries: string[] }>;
  }
).TEST_MIGRATIONS;

if (env.CATALOG_DB && migrations) {
  await applyD1Migrations(env.CATALOG_DB, migrations);
}
