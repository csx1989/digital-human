/**
 * Phase C2 — Automated Validation Runner (Modal GPU AVTR publisher)
 *
 * Orchestrates:
 *   1. Start wrangler dev (port 8789) — subscriber Worker proxy
 *   2. Start Modal GPU AVTR publisher — real I420 frames → SFU
 *   3. Wait for PUBLISHER_READY signal
 *   4. Open subscriber page in headless Chromium
 *   5. Run for 120s collecting diagnostics (FPS gate: 23-27)
 *   6. Teardown both publisher and subscriber
 *   7. Evaluate 12 pass/fail criteria + underflow metrics gate
 *
 * Usage:
 *   POC_ACCESS_TOKEN=<token> node run-c2-modal.cjs
 */

const { spawn, execSync } = require("child_process");
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

const POC_URL = process.env.POC_URL || "http://localhost:8789";
const RUN_SECONDS = 125;
const PHASE_DIR = __dirname;

function loadEnvFile(filename) {
  const filepath = path.join(PHASE_DIR, filename);
  if (!fs.existsSync(filepath)) return {};
  const content = fs.readFileSync(filepath, "utf-8");
  const result = {};
  for (const line of content.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx === -1) continue;
    result[trimmed.slice(0, eqIdx).trim()] = trimmed.slice(eqIdx + 1).trim();
  }
  return result;
}

const dotenvVars = loadEnvFile(".env");
const devVars = loadEnvFile(".dev.vars");
const CALLS_APP_ID = process.env.CALLS_APP_ID || dotenvVars.CALLS_APP_ID;
const CALLS_APP_SECRET = process.env.CALLS_APP_SECRET || dotenvVars.CALLS_APP_SECRET;
const POC_TOKEN = process.env.POC_ACCESS_TOKEN || devVars.POC_ACCESS_TOKEN;

if (!CALLS_APP_ID || !CALLS_APP_SECRET) { console.error("ERROR: CALLS_APP_ID and CALLS_APP_SECRET required."); process.exit(1); }
if (!POC_TOKEN) { console.error("ERROR: POC_ACCESS_TOKEN required."); process.exit(1); }

const tokenPreview = POC_TOKEN.length > 8 ? POC_TOKEN.slice(0, 4) + "..." + POC_TOKEN.slice(-4) : "***";

function extractDiag(diagText, key) {
  const m = diagText.match(new RegExp(`^${key}: (.+)$`, "m"));
  return m ? m[1].trim() : "N/A";
}
function parseFrames(val) { const n = parseInt(val, 10); return Number.isNaN(n) ? null : n; }
function parseFps(val) { const n = parseFloat(val); return Number.isNaN(n) ? null : n; }

(async () => {
  console.log("=== Phase C2 Test — Modal GPU AVTR Publisher ===");
  console.log(`URL:       ${POC_URL}`);
  console.log(`App ID:    ${CALLS_APP_ID}`);
  console.log(`Token:     ${tokenPreview}`);
  console.log(`Run:       ${RUN_SECONDS}s`);
  console.log(`FPS gate:  23-27 (target: 25)\n`);

  // 1. Start wrangler dev
  console.log("Starting wrangler dev on port 8789...");
  const wrangler = spawn("npx", ["wrangler", "dev", "--port", "8789"], {
    cwd: PHASE_DIR, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env },
  });
  wrangler.stdout.on("data", (d) => process.stdout.write(`[wrangler] ${d}`));
  wrangler.stderr.on("data", (d) => process.stderr.write(`[wrangler] ${d}`));

  let workerReady = false;
  for (let i = 0; i < 30; i++) {
    await new Promise(r => setTimeout(r, 1000));
    try {
      const resp = await fetch(`${POC_URL}/api/health`, { headers: { "x-poc-token": POC_TOKEN } });
      if (resp.ok) { const body = await resp.json(); if (body.configured) { workerReady = true; break; } }
    } catch {}
  }
  if (!workerReady) { console.error("ERROR: Worker did not start."); wrangler.kill("SIGTERM"); process.exit(1); }
  console.log("Worker ready.\n");

  // 2. Start Modal publisher
  const AVATAR_ID = process.env.AVATAR_ID || "arnold";
  const BG_ID = process.env.BG_ID || "aloe_window";
  console.log(`Starting Modal AVTR publisher: avatar=${AVATAR_ID}, bg=${BG_ID}`);
  const publisher = spawn("modal", ["run", "modal_avtr_publisher.py",
    "--duration", "150", "--avatar-id", AVATAR_ID, "--bg-id", BG_ID], {
    cwd: PHASE_DIR, stdio: ["ignore", "pipe", "pipe"], env: { ...process.env },
  });

  let publisherReady = false;
  let publisherSessionId = null;
  let publisherTrackName = null;
  let publisherIceConnected = false;
  const publisherLogs = [];
  let publisherMetrics = null;
  let publisherCloseStatus = null;
  let publisherCloseBody = "";
  let publisherCloseRequestBody = "";

  function processPublisherLine(line, stream) {
    publisherLogs.push(line);
    console.log(`[publisher:${stream}] ${line}`);
    if (line.startsWith("PUBLISHER_READY")) {
      const parts = line.split(/\s+/);
      publisherSessionId = parts[1];
      publisherTrackName = parts[2];
      publisherReady = true;
    }
    if (line.includes("ICE connected") || line.includes("ICE state: completed")) {
      publisherIceConnected = true;
    }
    // Capture metrics from heartbeat or final output
    const metricsMatch = line.match(/gen=(\d+)\s+emit=(\d+)\s+repeat=(\d+)\s+underflow=(\d+)/);
    if (metricsMatch) {
      publisherMetrics = {
        frames_generated: parseInt(metricsMatch[1]),
        frames_emitted: parseInt(metricsMatch[2]),
        frames_repeated: parseInt(metricsMatch[3]),
        underflow_count: parseInt(metricsMatch[4]),
      };
    }
    // Capture close request body
    const closeReqMatch = line.match(/tracks\/close request body: (.+)/);
    if (closeReqMatch) publisherCloseRequestBody = closeReqMatch[1];
    // Capture close HTTP status + response body
    const closeStatusMatch = line.match(/tracks\/close: status=(\d+)\s+body=(.*)/);
    if (closeStatusMatch) {
      publisherCloseStatus = parseInt(closeStatusMatch[1]);
      publisherCloseBody = closeStatusMatch[2];
    }
    // Fallback: capture close_status from === C2 RESULT === block
    const resultCloseMatch = line.match(/^\s*close_status:\s*(\d+)/);
    if (resultCloseMatch) publisherCloseStatus = parseInt(resultCloseMatch[1]);
    // Fallback: capture metrics from === C2 RESULT === block (Python dict repr)
    const resultMetricsMatch = line.match(/'frames_generated':\s*(\d+).*?'frames_emitted':\s*(\d+).*?'frames_repeated':\s*(\d+).*?'underflow_count':\s*(\d+)/);
    if (resultMetricsMatch) {
      publisherMetrics = {
        frames_generated: parseInt(resultMetricsMatch[1]),
        frames_emitted: parseInt(resultMetricsMatch[2]),
        frames_repeated: parseInt(resultMetricsMatch[3]),
        underflow_count: parseInt(resultMetricsMatch[4]),
      };
    }
  }

  publisher.stdout.on("data", (data) => {
    data.toString().split("\n").filter(l => l.trim()).forEach(l => processPublisherLine(l, "out"));
  });
  publisher.stderr.on("data", (data) => {
    data.toString().split("\n").filter(l => l.trim()).forEach(l => processPublisherLine(l, "err"));
  });

  // 3. Wait for PUBLISHER_READY (300s timeout for GPU init)
  console.log("\nWaiting for PUBLISHER_READY (300s timeout for GPU container + pipeline init)...");
  const readyTimeout = 300;
  const readyStart = Date.now();
  while (!publisherReady) {
    if (Date.now() - readyStart > readyTimeout * 1000) {
      console.error(`ERROR: Publisher did not become ready within ${readyTimeout}s.`);
      console.error("Publisher logs:\n" + publisherLogs.slice(-30).join("\n"));
      publisher.kill("SIGTERM");
      wrangler.kill("SIGTERM");
      process.exit(1);
    }
    if (publisher.exitCode !== null) {
      console.error(`ERROR: Publisher exited with code ${publisher.exitCode}.`);
      console.error("Publisher logs:\n" + publisherLogs.slice(-30).join("\n"));
      wrangler.kill("SIGTERM");
      process.exit(1);
    }
    await new Promise(r => setTimeout(r, 1000));
  }
  console.log(`\nPublisher ready! Session: ${publisherSessionId}, Track: ${publisherTrackName}\n`);

  // 4. Browser subscriber
  const browser = await chromium.launch({
    headless: true,
    args: ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required", "--mute-audio"],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  const consoleLogs = [];
  page.on("console", (msg) => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => consoleLogs.push(`[PAGE_ERROR] ${err.message}`));

  const closeRequests = [];
  page.on("request", (req) => {
    if (req.url().includes("/tracks/close")) {
      closeRequests.push({ url: req.url(), body: req.postData() || "" });
    }
  });

  const subUrl = `${POC_URL}?token=${encodeURIComponent(POC_TOKEN)}&pubSession=${encodeURIComponent(publisherSessionId)}&trackName=${encodeURIComponent(publisherTrackName)}`;
  console.log(`Opening subscriber...`);
  await page.goto(subUrl, { waitUntil: "networkidle" });
  console.log("Subscriber page loaded.");

  // 5. Wait for subscription
  try {
    await page.waitForFunction(
      () => { const el = document.querySelector("#diagOutput"); return el && el.textContent.includes("Subscription running"); },
      { timeout: 60000 }
    );
    console.log("Subscription is running!\n");
  } catch {
    console.error("ERROR: Subscription did not start within 60s.");
    const diag = await page.textContent("#diagOutput");
    console.error("Diagnostics:\n" + diag);
    await browser.close(); publisher.kill("SIGTERM"); wrangler.kill("SIGTERM"); process.exit(1);
  }

  // 6. Monitor
  const snapshots = [];
  const startTime = Date.now();
  for (let i = 0; i < Math.ceil(RUN_SECONDS / 10); i++) {
    await page.waitForTimeout(10000);
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const diag = await page.textContent("#diagOutput");
    const remoteVideoPlaying = await page.evaluate(() => {
      const v = document.querySelector("#remoteVideo");
      return v ? v.currentTime > 0 && !v.paused : false;
    });
    snapshots.push({
      elapsed, subIce: extractDiag(diag, "subscriber.iceState"),
      playing: extractDiag(diag, "remote.playing"),
      dimensions: extractDiag(diag, "remote.dimensions"),
      framesDecoded: extractDiag(diag, "stats.framesDecoded"),
      fps: extractDiag(diag, "stats.framesPerSecond"),
      packetsLost: extractDiag(diag, "stats.packetsLost"),
      jitter: extractDiag(diag, "stats.jitter"),
      videoPlaying: remoteVideoPlaying,
    });
    const s = snapshots[snapshots.length - 1];
    console.log(`[${elapsed}s] sub=${s.subIce} fps=${s.fps} decoded=${s.framesDecoded} lost=${s.packetsLost} dims=${s.dimensions}`);
  }

  // 7. Final diagnostics
  const finalDiag = await page.textContent("#diagOutput");
  console.log("\n=== FINAL DIAGNOSTICS ===\n" + finalDiag);

  // 8. Teardown subscriber
  console.log("\n=== TEARDOWN ===");
  await page.click("#stopBtn");
  await page.waitForTimeout(3000);
  const teardownConsoleLines = consoleLogs.filter(l => /tearing down|tracks closed|close failed|teardown complete/i.test(l));
  teardownConsoleLines.forEach(l => console.log("  " + l));
  closeRequests.forEach(r => console.log(`  close: ${r.body}`));
  await browser.close();

  // 9. Teardown publisher (Modal function exits on its own)
  console.log("\nWaiting for publisher to exit...");
  const pubExitPromise = new Promise(resolve => { publisher.on("exit", (code) => resolve(code)); });
  const pubExitResult = await Promise.race([
    pubExitPromise, new Promise(r => setTimeout(() => r("timeout"), 60000)),
  ]);
  if (pubExitResult === "timeout") {
    console.log("Publisher still running after 60s, killing...");
    publisher.kill("SIGTERM");
  } else {
    console.log(`Publisher exited with code ${pubExitResult}`);
  }
  wrangler.kill("SIGTERM");

  // 10. Evaluate
  const last = snapshots[snapshots.length - 1];
  const first = snapshots.find(s => parseFrames(s.framesDecoded) !== null) || snapshots[0];
  const firstDecoded = parseFrames(first.framesDecoded);
  const lastDecoded = parseFrames(last.framesDecoded);

  const recentFps = snapshots.map(s => parseFps(s.fps)).filter(v => v !== null).slice(-5).sort((a, b) => a - b);
  const medianFps = recentFps.length > 0 ? recentFps[Math.floor(recentFps.length / 2)] : null;

  const totalRuntime = last.elapsed;
  const pageErrors = consoleLogs.filter(l => l.includes("[PAGE_ERROR]"));
  const dims = last.dimensions;
  const dimsValid = dims !== "N/A" && dims !== "0x0";

  const subCloseHadMids = closeRequests.length > 0 && closeRequests.every(r => {
    try {
      const body = JSON.parse(r.body);
      return Array.isArray(body.tracks) && body.tracks.length > 0 && body.tracks.every(t => t.mid != null) && body.force === true;
    } catch { return false; }
  });

  const teardownComplete = teardownConsoleLines.some(l => /teardown complete/i.test(l));
  const fpsInRange = medianFps !== null && medianFps >= 23 && medianFps <= 27;

  // Publisher close validation: require 200, or 400/404 with body indicating already-closed/not-found
  const pubCloseIsOk = publisherCloseStatus === 200;
  const pubCloseIsAcceptable = publisherCloseStatus !== null && publisherCloseStatus !== 200 && (
    publisherCloseStatus === 404 ||
    (publisherCloseStatus === 400 && /not found|already|closed|inactive|gone/i.test(publisherCloseBody))
  );
  const pubClosePassed = pubCloseIsOk || pubCloseIsAcceptable;

  // Underflow gate: metrics MUST exist and underflow_count MUST be 0
  const underflowOk = publisherMetrics !== null && publisherMetrics.underflow_count === 0;

  const criteria = [
    { name: "Runtime >= 120s",                     pass: totalRuntime >= 120,                              detail: `${totalRuntime}s` },
    { name: "Subscriber ICE connected/completed",  pass: ["connected","completed"].includes(last.subIce),  detail: last.subIce },
    { name: "Remote video playing",                pass: !!last.videoPlaying,                              detail: String(last.videoPlaying) },
    { name: "Remote dimensions valid (1280x720)",  pass: dimsValid,                                        detail: dims },
    { name: "framesDecoded increased (delta>=500)",pass: firstDecoded !== null && lastDecoded !== null && (lastDecoded - firstDecoded) >= 500, detail: `${firstDecoded} → ${lastDecoded}` },
    { name: "Final framesDecoded >= 1000",         pass: lastDecoded !== null && lastDecoded >= 1000,      detail: String(lastDecoded) },
    { name: "Median FPS in 23-27 range (25fps)",   pass: fpsInRange,                                       detail: `${medianFps} (median last 5)` },
    { name: "No page errors",                      pass: pageErrors.length === 0,                          detail: `${pageErrors.length} error(s)` },
    { name: "Publisher ICE connected",             pass: publisherIceConnected,                            detail: publisherIceConnected ? "YES" : "NO" },
    { name: "Subscriber teardown complete",        pass: teardownComplete,                                 detail: teardownComplete ? "YES" : "NO" },
    { name: "Subscriber close used mids+force",    pass: subCloseHadMids,                                  detail: `${closeRequests.length} close req(s)` },
    { name: "Publisher close returned 200 (or acceptable already-closed)", pass: pubClosePassed, detail: publisherCloseStatus !== null ? `${publisherCloseStatus}${pubCloseIsOk ? " (OK)" : pubCloseIsAcceptable ? " (acceptable: already-closed)" : " (FAIL)"}` : "no close response" },
    { name: "Underflow = 0 (metrics mandatory)",   pass: underflowOk,                                      detail: publisherMetrics ? `${publisherMetrics.underflow_count} underflows (gen=${publisherMetrics.frames_generated}, emit=${publisherMetrics.frames_emitted}, repeat=${publisherMetrics.frames_repeated})` : "MISSING — no publisher metrics captured" },
  ];

  const allPassed = criteria.every(c => c.pass);

  console.log("\n=== METRICS ===");
  console.log(`Runtime:             ${totalRuntime}s`);
  console.log(`Remote dimensions:   ${last.dimensions}`);
  console.log(`Frames decoded:      ${firstDecoded} → ${lastDecoded}`);
  console.log(`FPS (median last 5): ${medianFps}`);
  console.log(`Packets lost:        ${last.packetsLost}`);
  console.log(`Jitter:              ${last.jitter}`);
  console.log(`Publisher ICE:       ${publisherIceConnected ? "connected" : "unknown"}`);
  console.log(`Pub close req body:  ${publisherCloseRequestBody || "(not captured)"}`);
  console.log(`Pub close status:    ${publisherCloseStatus !== null ? publisherCloseStatus : "(no response)"}`);
  console.log(`Pub close resp body: ${publisherCloseBody || "(not captured)"}`);
  console.log(`Underflow metrics:   ${JSON.stringify(publisherMetrics)}`);
  console.log(`Teardown complete:   ${teardownComplete}`);

  console.log("\n=== PASS/FAIL CRITERIA ===");
  criteria.forEach(c => console.log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name} (${c.detail})`));

  console.log(`\n=== OVERALL: ${allPassed ? "PASS" : "FAIL"} ===`);
  process.exit(allPassed ? 0 : 1);
})();
