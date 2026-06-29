/**
 * Phase A — Realtime SFU Echo Test (Automated)
 *
 * Usage:
 *   POC_ACCESS_TOKEN=<token> node run-echo-test.cjs
 *   POC_URL=http://localhost:8787 POC_ACCESS_TOKEN=<token> node run-echo-test.cjs
 *
 * Requires a running `wrangler dev` server with .dev.vars configured.
 */

const { chromium } = require("playwright");

// ---------------------------------------------------------------------------
// Config from environment
// ---------------------------------------------------------------------------
const POC_URL = process.env.POC_URL || "http://localhost:8787";
const POC_TOKEN = process.env.POC_ACCESS_TOKEN;
const RUN_SECONDS = 125; // 2 min 5 s

if (!POC_TOKEN) {
  console.error("ERROR: POC_ACCESS_TOKEN environment variable is required.\n");
  console.error("Usage:");
  console.error("  POC_ACCESS_TOKEN=<token> node run-echo-test.cjs");
  console.error("  POC_URL=http://localhost:8787 POC_ACCESS_TOKEN=<token> node run-echo-test.cjs");
  process.exit(1);
}

// Never print the full token
const tokenPreview = POC_TOKEN.length > 8
  ? POC_TOKEN.slice(0, 4) + "..." + POC_TOKEN.slice(-4)
  : "***";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function extractDiag(diagText, key) {
  const m = diagText.match(new RegExp(`^${key}: (.+)$`, "m"));
  return m ? m[1].trim() : "N/A";
}

function parseFrames(val) {
  const n = parseInt(val, 10);
  return Number.isNaN(n) ? null : n;
}

function parseFps(val) {
  const n = parseFloat(val);
  return Number.isNaN(n) ? null : n;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
(async () => {
  console.log("=== Phase A Echo Test — Automated Run ===");
  console.log(`URL:   ${POC_URL}`);
  console.log(`Token: ${tokenPreview}`);
  console.log(`Run:   ${RUN_SECONDS}s\n`);

  const browser = await chromium.launch({
    headless: true,
    args: [
      "--use-fake-ui-for-media-stream",
      "--autoplay-policy=no-user-gesture-required",
      "--mute-audio",
    ],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  // Capture ALL console output and page errors
  const consoleLogs = [];
  page.on("console", (msg) => consoleLogs.push(`[${msg.type()}] ${msg.text()}`));
  page.on("pageerror", (err) => consoleLogs.push(`[PAGE_ERROR] ${err.message}`));

  // Intercept close requests to verify body shape
  const closeRequests = [];
  page.on("request", (req) => {
    if (req.url().includes("/tracks/close")) {
      closeRequests.push({ url: req.url(), body: req.postData() || "" });
    }
  });

  // --- 1. Navigate ---
  await page.goto(POC_URL, { waitUntil: "networkidle" });
  console.log("Page loaded.");

  // --- 2. Enter POC token ---
  await page.fill("#pocToken", POC_TOKEN);

  // --- 3. Check Config ---
  await page.click("#checkBtn");
  await page.waitForFunction(
    () => document.querySelector("#configStatus").textContent.includes("configured"),
    { timeout: 10000 }
  );
  const configStatus = await page.textContent("#configStatus");
  console.log("Config:", configStatus);

  // --- 4. Start Echo Test ---
  console.log("\nStarting echo test...");
  await page.click("#startBtn");

  // --- 5. Wait for "Echo test running" ---
  try {
    await page.waitForFunction(
      () => {
        const el = document.querySelector("#diagOutput");
        return el && el.textContent.includes("Echo test running");
      },
      { timeout: 60000 }
    );
    console.log("Echo test is running!\n");
  } catch {
    console.error("ERROR: Echo test did not start within 60s.");
    const diag = await page.textContent("#diagOutput");
    console.error("Diagnostics:\n" + diag);
    console.error("\nConsole logs (last 30):\n" + consoleLogs.slice(-30).join("\n"));
    await browser.close();
    process.exit(1);
  }

  // --- 6. Monitor for RUN_SECONDS ---
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
      elapsed,
      pubIce: extractDiag(diag, "publisher.iceState"),
      subIce: extractDiag(diag, "subscriber.iceState"),
      playing: extractDiag(diag, "remote.playing"),
      uptime: extractDiag(diag, "remote.uptime"),
      dimensions: extractDiag(diag, "remote.dimensions"),
      framesDecoded: extractDiag(diag, "stats.framesDecoded"),
      fps: extractDiag(diag, "stats.framesPerSecond"),
      packetsLost: extractDiag(diag, "stats.packetsLost"),
      jitter: extractDiag(diag, "stats.jitter"),
      videoPlaying: remoteVideoPlaying,
    });

    const s = snapshots[snapshots.length - 1];
    console.log(`[${elapsed}s] pub=${s.pubIce} sub=${s.subIce} fps=${s.fps} decoded=${s.framesDecoded} lost=${s.packetsLost} dims=${s.dimensions}`);
  }

  // --- 7. Final diagnostics ---
  const finalDiag = await page.textContent("#diagOutput");
  console.log("\n=== FINAL DIAGNOSTICS ===");
  console.log(finalDiag);

  // --- 8. Teardown ---
  console.log("\n=== TEARDOWN ===");
  await page.click("#stopBtn");
  await page.waitForTimeout(3000);

  const teardownDiag = await page.textContent("#diagOutput");
  const teardownConsoleLines = consoleLogs.filter((l) =>
    /tearing down|tracks closed|close requested|best-effort close|teardown complete/i.test(l)
  );

  console.log("Teardown console messages:");
  teardownConsoleLines.forEach((l) => console.log("  " + l));

  console.log("\nCaptured close requests:");
  closeRequests.forEach((r) => console.log(`  ${r.url}\n    body: ${r.body}`));

  await browser.close();

  // --- 9. Evaluate pass/fail criteria ---
  const last = snapshots[snapshots.length - 1];
  const first = snapshots.find((s) => parseFrames(s.framesDecoded) !== null) || snapshots[0];

  const firstDecoded = parseFrames(first.framesDecoded);
  const lastDecoded = parseFrames(last.framesDecoded);
  const lastFps = parseFps(last.fps);
  const totalRuntime = last.elapsed;

  // Check teardown specifics from captured request bodies
  const closeHadMids = closeRequests.length > 0 && closeRequests.every((r) => {
    try {
      const body = JSON.parse(r.body);
      return Array.isArray(body.tracks) && body.tracks.length > 0 && body.tracks.every((t) => t.mid != null);
    } catch { return false; }
  });
  const noEmptyTracks = closeRequests.length > 0 && !closeRequests.some((r) => {
    try {
      const body = JSON.parse(r.body);
      return Array.isArray(body.tracks) && body.tracks.length === 0;
    } catch { return false; }
  });
  const teardownComplete = teardownConsoleLines.some((l) => /teardown complete/i.test(l));

  const pageErrors = consoleLogs.filter((l) => l.includes("[PAGE_ERROR]"));

  const dims = last.dimensions;
  const dimsValid = dims !== "N/A" && dims !== "0x0";

  const criteria = [
    { name: "Runtime >= 120s",                    pass: totalRuntime >= 120,                              detail: `${totalRuntime}s` },
    { name: "Publisher ICE connected/completed",  pass: ["connected","completed"].includes(last.pubIce),  detail: last.pubIce },
    { name: "Subscriber ICE connected/completed", pass: ["connected","completed"].includes(last.subIce),  detail: last.subIce },
    { name: "Remote video playing",               pass: !!last.videoPlaying,                              detail: String(last.videoPlaying) },
    { name: "Remote dimensions valid",            pass: dimsValid,                                        detail: dims },
    { name: "framesDecoded increased substantially", pass: firstDecoded !== null && lastDecoded !== null && (lastDecoded - firstDecoded) >= 500, detail: `${firstDecoded} → ${lastDecoded} (Δ${lastDecoded !== null && firstDecoded !== null ? lastDecoded - firstDecoded : "?"})` },
    { name: "Final framesDecoded >= 1000",        pass: lastDecoded !== null && lastDecoded >= 1000,      detail: String(lastDecoded) },
    { name: "Final FPS >= 20",                    pass: lastFps !== null && lastFps >= 20,                detail: String(lastFps) },
    { name: "No page errors",                     pass: pageErrors.length === 0,                          detail: `${pageErrors.length} error(s)` },
    { name: "Teardown logged complete",           pass: teardownComplete,                                 detail: teardownComplete ? "YES" : "NO" },
    { name: "Teardown used specific mids",        pass: closeHadMids,                                     detail: closeRequests.length + " close req(s)" },
    { name: "No empty tracks:[] close",           pass: noEmptyTracks,                                    detail: noEmptyTracks ? "CONFIRMED" : "FOUND EMPTY" },
  ];

  const allPassed = criteria.every((c) => c.pass);

  // --- 10. Summary ---
  console.log("\n=== METRICS ===");
  console.log(`Runtime:             ${totalRuntime}s`);
  console.log(`Remote dimensions:   ${last.dimensions}`);
  console.log(`Frames decoded:      ${firstDecoded} → ${lastDecoded}`);
  console.log(`FPS (final):         ${lastFps}`);
  console.log(`Packets lost:        ${last.packetsLost}`);
  console.log(`Jitter:              ${last.jitter}`);
  console.log(`Publisher ICE:       ${first.pubIce} → ${last.pubIce}`);
  console.log(`Subscriber ICE:      ${first.subIce} → ${last.subIce}`);
  console.log(`Teardown complete:   ${teardownComplete ? "YES" : "NO"}`);
  console.log(`Close requests:      ${closeRequests.length} (all with mids: ${closeHadMids ? "YES" : "NO"})`);

  console.log("\n=== PASS/FAIL CRITERIA ===");
  criteria.forEach((c) => {
    console.log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name} (${c.detail})`);
  });

  console.log(`\n=== OVERALL: ${allPassed ? "PASS" : "FAIL"} ===`);
  process.exit(allPassed ? 0 : 1);
})();
