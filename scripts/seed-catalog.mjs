#!/usr/bin/env node

import { execSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const bridgeDir = path.resolve(__dirname, "..", "avatar-saas-prototype", "cloudflare-session-bridge");

const isRemote = process.argv.includes("--remote");
const target = isRemote ? "--remote" : "--local";
const dbName = "avtr-catalog";
const seedFile = path.join(bridgeDir, "migrations", "0002_seed_system_assets.sql");

console.log(`[seed-catalog] Seeding ${dbName} (${target})`);
console.log(`[seed-catalog] File: ${seedFile}`);

execSync(
  `npx wrangler d1 execute ${dbName} ${target} --file "${seedFile}"`,
  { stdio: "inherit", cwd: bridgeDir },
);

console.log("[seed-catalog] Done. System assets (arnold, aloe_window) seeded idempotently.");
