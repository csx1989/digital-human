// Media pipeline benchmark — fMP4 vs SFU/WebRTC.
//
// Answers: "Where is time and compute spent in each transport path?"
//
// Three modes:
//   --mode fmp4   : single fMP4 session (GPU NVENC, SSE relay)
//   --mode sfu    : single SFU session (CPU PyAV, WebRTC RTP)
//   --mode ab     : sequential A/B (fmp4 then sfu), prints comparison table
//
// Usage:
//   AVTR_API_KEY=... node scripts/media-benchmark.mjs --mode sfu --duration 120
//   AVTR_API_KEY=... node scripts/media-benchmark.mjs --mode fmp4 --duration 120
//   AVTR_API_KEY=... node scripts/media-benchmark.mjs --mode ab  --duration 120
//   AVTR_API_KEY=... node scripts/media-benchmark.mjs --mode ab --json report.json
//
// IMPORTANT: For full timing data, deploy the Modal 5q app with
// AVTR_MEDIA_TIMING_DEBUG=1 in the container environment before running
// benchmarks. Without it, only browser-side + session-level metrics are
// available (server-side per-stage timing columns will be "N/A").
//
// This script does NOT deploy anything. It only creates sessions, monitors
// them, and collects metrics. SFU must be enabled on the Worker via
// AVTR_REALTIME_SFU_ENABLED=true before running --mode sfu or --mode ab.

import { createRequire } from "module";
import { WebSocket } from "ws";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Playwright lives in the POC node_modules
const POC_NM = path.resolve(__dirname, "..", "avatar-saas-prototype", "realtime-sfu-poc", "phase-b-modal-publisher-poc", "node_modules");
const require = createRequire(import.meta.url + "/../");
let chromium;
try {
  chromium = require(path.join(POC_NM, "playwright")).chromium;
} catch {
  chromium = require("playwright").chromium;
}

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;
const POLL_INTERVAL_MS = 10000;

// Parse CLI args
const args = process.argv.slice(2);
function arg(name, def) {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : def;
}
const MODE = arg("mode", "ab");
const DURATION = Number(arg("duration", 120));
const JSON_OUT = arg("json", null);

if (!API_KEY) { console.error("AVTR_API_KEY required."); process.exit(1); }

const auth = { Authorization: `Bearer ${API_KEY}` };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ── API helpers ──────────────────────────────────────────────────────────────

async function apiJson(method, url, body) {
  const resp = await fetch(url, {
    method,
    headers: { ...auth, "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await resp.text();
  let data; try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  return { status: resp.status, data };
}

function wsCollect(ws, timeout, wantType) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error(`WS timeout waiting for ${wantType}`)), timeout);
    ws.on("message", (raw) => {
      let m; try { m = JSON.parse(raw.toString()); } catch { return; }
      if (m.type === wantType || m.type === "error") {
        clearTimeout(to);
        resolve(m);
      }
    });
    ws.on("error", (e) => { clearTimeout(to); reject(e); });
    ws.on("close", (c) => { clearTimeout(to); reject(new Error(`WS closed code=${c}`)); });
  });
}

// ── PostHog query helper (for SFU publisher stop metrics) ────────────────────

function loadPosthogEnv() {
  try {
    const envPath = path.resolve(__dirname, "..", ".env.posthog");
    const text = fs.readFileSync(envPath, "utf8");
    const get = (key) => {
      const m = text.match(new RegExp(`${key}=(.+)`));
      return m ? m[1].trim().replace(/^["']|["']$/g, "") : null;
    };
    return {
      apiKey: get("POSTHOG_PERSONAL_API_KEY"),
      projectId: get("POSTHOG_PROJECT_ID"),
      apiHost: get("POSTHOG_API_HOST") || "https://eu.posthog.com",
    };
  } catch {
    return null;
  }
}

async function queryPosthogTiming(sessionId) {
  const ph = loadPosthogEnv();
  if (!ph?.apiKey) return null;

  const distinctId = `session:${sessionId}`;
  const hogql = `SELECT event, properties FROM events WHERE distinct_id = '${distinctId}' AND event IN ('session_ended', 'modal_end_completed') ORDER BY timestamp DESC LIMIT 5`;

  try {
    const resp = await fetch(`${ph.apiHost}/api/projects/${ph.projectId}/query/`, {
      method: "POST",
      headers: { "Authorization": `Bearer ${ph.apiKey}`, "Content-Type": "application/json" },
      body: JSON.stringify({ query: { kind: "HogQLQuery", query: hogql } }),
    });
    const data = await resp.json();
    const rows = data?.results || [];
    for (const row of rows) {
      const props = typeof row[1] === "string" ? JSON.parse(row[1]) : row[1];
      if (props?.publisherStopMetrics?.timing) {
        return props.publisherStopMetrics;
      }
      if (props?.publisher_stop_metrics?.timing) {
        return props.publisher_stop_metrics;
      }
    }
    return null;
  } catch (e) {
    return null;
  }
}

// ── fMP4 benchmark ───────────────────────────────────────────────────────────

async function runFmp4Benchmark(duration) {
  console.log("\n=== fMP4 Benchmark ===");

  // 1. Create session
  const cr = await apiJson("POST", `${BASE}/v1/sessions`, {});
  if (cr.status !== 200) throw new Error(`create failed: ${cr.status}`);
  const session = cr.data.data;
  const sessionId = session.session_id;
  const wsUrl = session.stream.ws_url;
  console.log(`  session: ${sessionId}`);
  console.log(`  transport: ${session.stream.transport}`);

  // 2. WS connect
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });

  let firstFrameMs = null;
  let segmentCount = 0;
  let totalBytes = 0;
  let errorCount = 0;
  let endEvent = null;
  const t0 = Date.now();

  ws.on("message", (raw) => {
    let m; try { m = JSON.parse(raw.toString()); } catch { return; }
    const type = m.type || "";

    if (type === "avatar.fmp4.segment" || type === "fmp4.init" || type === "fmp4_init") {
      if (!firstFrameMs) {
        firstFrameMs = Date.now() - t0;
        console.log(`  first fMP4 frame at ${firstFrameMs}ms`);
      }
      segmentCount++;
      if (m.byteLength) totalBytes += m.byteLength;
    }

    if (type === "error") {
      errorCount++;
      console.log(`  ERROR: ${m.code || JSON.stringify(m).slice(0, 100)}`);
    }

    if (type === "end") {
      endEvent = m;
    }
  });

  // 3. Start session
  ws.send(JSON.stringify({ type: "session.start", avatarId: "arnold", bgId: "aloe_window" }));
  const ready = await wsCollect(ws, 120000, "session.ready");
  console.log(`  session.ready at ${Date.now() - t0}ms`);

  // 4. Monitor for duration
  console.log(`  monitoring ${duration}s...`);
  await sleep(duration * 1000);

  // 5. Teardown
  console.log("  teardown...");
  try { ws.send(JSON.stringify({ type: "session.end" })); await sleep(2000); } catch {}
  ws.close();
  try { await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`); } catch {}

  const wallMs = Date.now() - t0;

  return {
    mode: "fmp4",
    sessionId,
    wallMs,
    firstFrameMs,
    segmentCount,
    totalBytes,
    errorCount,
    mediaTiming: endEvent?.mediaTiming || null,
    cadenceStats: endEvent?.cadenceStats || null,
  };
}

// ── SFU benchmark ────────────────────────────────────────────────────────────

async function runSfuBenchmark(duration) {
  console.log("\n=== SFU Benchmark ===");

  // 1. Create session
  const cr = await apiJson("POST", `${BASE}/v1/sessions`, {});
  if (cr.status !== 200) throw new Error(`create failed: ${cr.status}`);
  const session = cr.data.data;
  const sessionId = session.session_id;
  const wsUrl = session.stream.ws_url;
  const sfu = session.stream.sfu;
  if (!sfu) throw new Error("SFU not advertised — Worker config incomplete");
  console.log(`  session: ${sessionId}`);

  // 2. WS connect
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });

  ws.send(JSON.stringify({ type: "session.start", avatarId: "arnold", bgId: "aloe_window", transport: "webrtc_sfu" }));
  const ready = await wsCollect(ws, 120000, "session.ready");
  console.log(`  session.ready at ${ready.transport}`);

  // 3. Playwright subscriber
  const browser = await chromium.launch({
    headless: true,
    args: ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required", "--mute-audio"],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));

  await page.setContent(`<!DOCTYPE html><html><body>
    <video id="v" autoplay playsinline muted style="width:640px;height:360px;background:#000"></video>
    <canvas id="fc" width="32" height="18" style="display:none"></canvas>
  </body></html>`);

  // Subscribe via production routes
  const subInfo = await page.evaluate(async (params) => {
    const { subscribeUrl, answerUrl, token } = params;
    const subResp = await fetch(`${subscribeUrl}?token=${encodeURIComponent(token)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    if (!subResp.ok) return { error: `subscribe ${subResp.status}` };
    const sub = await subResp.json();
    if (sub.error) return { error: JSON.stringify(sub.error) };

    const pc = new RTCPeerConnection();
    window._pc = pc;
    let iceState = "new";
    pc.oniceconnectionstatechange = () => { iceState = pc.iceConnectionState; };
    const video = document.getElementById("v");
    pc.ontrack = (e) => { video.srcObject = e.streams[0]; };

    await pc.setRemoteDescription({ type: sub.offer_type || "offer", sdp: sub.offer_sdp });
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    if (pc.iceGatheringState !== "complete") {
      await new Promise((resolve) => {
        const check = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
        pc.addEventListener("icegatheringstatechange", check);
        setTimeout(resolve, 8000);
      });
    }
    const ansResp = await fetch(`${answerUrl}?token=${encodeURIComponent(token)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_sdp: pc.localDescription.sdp, answer_type: pc.localDescription.type }),
    });
    if (!ansResp.ok) return { error: `answer ${ansResp.status}` };

    // Frozen-frame detector
    const fc = document.getElementById("fc");
    const fcx = fc.getContext("2d", { willReadFrequently: true });
    let lastPixels = null;
    let consecutiveIdentical = 0;
    const IDENTICAL_MAX_DELTA = 50;

    window._frozenFingerprint = () => {
      try {
        fcx.drawImage(video, 0, 0, 32, 18);
        const img = fcx.getImageData(0, 0, 32, 18).data;
        const gray = new Uint8ClampedArray(32 * 18);
        for (let i = 0; i < 32 * 18; i++) gray[i] = (img[i*4] + img[i*4+1] + img[i*4+2]) / 3;
        if (lastPixels === null) { lastPixels = gray; return { identical: false, delta: null, consecutive: 0 }; }
        let delta = 0;
        for (let i = 0; i < gray.length; i++) delta += Math.abs(gray[i] - lastPixels[i]);
        const identical = delta < IDENTICAL_MAX_DELTA;
        if (identical) consecutiveIdentical++;
        else consecutiveIdentical = 0;
        lastPixels = gray;
        return { identical, delta, consecutive: consecutiveIdentical };
      } catch { return { identical: false, delta: null, consecutive: 0 }; }
    };

    window._pollStats = async () => {
      const stats = await pc.getStats();
      let v = null, cp = null;
      stats.forEach((r) => {
        if (r.type === "inbound-rtp" && r.kind === "video") v = r;
        if (r.type === "candidate-pair" && r.nominated) cp = r;
      });
      const fp = window._frozenFingerprint();
      return {
        iceState,
        playing: video.currentTime > 0 && !video.paused,
        dimensions: video.videoWidth ? `${video.videoWidth}x${video.videoHeight}` : "0x0",
        framesDecoded: v ? v.framesDecoded : null,
        fps: v ? v.framesPerSecond : null,
        packetsLost: v ? v.packetsLost : null,
        jitter: v ? v.jitter : null,
        rtt: cp ? cp.currentRoundTripTime : null,
        frozenIdentical: fp.identical,
        frozenDelta: fp.delta,
        frozenConsecutive: fp.consecutive,
      };
    };
    return { ok: true, mids: sub.mids };
  }, { subscribeUrl: sfu.subscribe_url, answerUrl: sfu.answer_url, token: sfu.token });

  if (subInfo.error) {
    console.error(`  SUBSCRIBE FAILED: ${subInfo.error}`);
    await browser.close(); ws.close();
    try { await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`); } catch {}
    throw new Error(`SFU subscribe failed: ${subInfo.error}`);
  }

  // 4. Poll metrics
  console.log(`  monitoring ${duration}s...`);
  const snapshots = [];
  const t0 = Date.now();
  let firstFrameMs = null;

  for (let i = 0; i < Math.ceil(duration / (POLL_INTERVAL_MS / 1000)); i++) {
    await sleep(POLL_INTERVAL_MS);
    let s;
    try { s = await page.evaluate(() => window._pollStats()); }
    catch (e) { s = { error: String(e) }; }
    s.elapsed = Math.floor((Date.now() - t0) / 1000);
    if (s.framesDecoded != null && !firstFrameMs) firstFrameMs = Date.now() - t0;
    snapshots.push(s);
    if (!s.error) {
      console.log(
        `  [${s.elapsed}s] decoded=${s.framesDecoded} fps=${s.fps?.toFixed(1)} ` +
        `lost=${s.packetsLost} frozen=${s.frozenIdentical ? `STUCK×${s.frozenConsecutive}` : "live"}(Δ=${s.frozenDelta ?? "?"})`
      );
    }
  }

  // 5. Teardown
  console.log("  teardown...");
  try { await apiJson("POST", `${sfu.close_url}?token=${encodeURIComponent(sfu.token)}`, {}); } catch {}
  await browser.close();
  try { ws.send(JSON.stringify({ type: "session.end" })); await sleep(2000); } catch {}
  ws.close();
  try { await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`); } catch {}

  const wallMs = Date.now() - t0;

  // 6. Query PostHog for publisher stop metrics (timing data)
  console.log("  querying PostHog for publisher timing...");
  await sleep(3000); // allow events to propagate
  const stopMetrics = await queryPosthogTiming(sessionId);

  const last = snapshots.filter(s => !s.error).slice(-1)[0] || {};
  const fpsValues = snapshots.filter(s => s.fps != null).map(s => s.fps).sort((a, b) => a - b);
  const medianFps = fpsValues.length ? fpsValues[Math.floor(fpsValues.length / 2)] : null;
  const frozenMax = Math.max(...snapshots.filter(s => !s.error).map(s => s.frozenConsecutive || 0), 0);

  return {
    mode: "sfu",
    sessionId,
    wallMs,
    firstFrameMs,
    framesDecoded: last.framesDecoded || 0,
    medianFps,
    packetsLost: last.packetsLost || 0,
    jitter: last.jitter || 0,
    rtt: last.rtt || 0,
    frozenMaxConsecutive: frozenMax,
    pageErrors: pageErrors.length,
    stopMetrics,
  };
}

// ── Report formatting ────────────────────────────────────────────────────────

function fmtMs(ms) {
  if (ms == null) return "N/A";
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtPct(stats, key = "p50_ms") {
  if (!stats || stats.count === 0) return "N/A";
  return `${stats[key]}`;
}

function printComparisonTable(fmp4, sfu) {
  console.log("\n" + "=".repeat(70));
  console.log("  MEDIA PIPELINE BENCHMARK — fMP4 vs SFU/WebRTC");
  console.log("=".repeat(70));

  const rows = [
    ["Metric", "fMP4 (NVENC)", "SFU (PyAV CPU)"],
    ["-" .repeat(25), "-".repeat(20), "-".repeat(20)],
    ["Session wall time", fmtMs(fmp4?.wallMs), fmtMs(sfu?.wallMs)],
    ["First frame time", fmtMs(fmp4?.firstFrameMs), fmtMs(sfu?.firstFrameMs)],
    ["Segments / frames", fmp4?.segmentCount ?? "N/A", sfu?.framesDecoded ?? "N/A"],
    ["Total bytes", fmp4?.totalBytes ?? "N/A", "N/A"],
    ["Median FPS", "N/A (MSE)", sfu?.medianFps?.toFixed(1) ?? "N/A"],
    ["Packets lost", "N/A (TCP/SSE)", sfu?.packetsLost ?? "N/A"],
    ["Jitter (s)", "N/A", sfu?.jitter?.toFixed(3) ?? "N/A"],
    ["Frozen max consecutive polls", "N/A (MSE)", sfu?.frozenMaxConsecutive ?? "N/A"],
    ["Page errors", fmp4?.errorCount ?? "N/A", sfu?.pageErrors ?? "N/A"],
    ["-" .repeat(25), "-".repeat(20), "-".repeat(20)],
    ["── Server-side timing (requires TIMING_DEBUG=1) ──", "", ""],
  ];

  // fMP4 timing
  const ft = fmp4?.mediaTiming;
  const st = sfu?.stopMetrics?.timing;

  if (ft || st) {
    rows.push(
      ["process_chunk p50/p95 (ms)",
        ft ? `${ft.renderMs?.p50}/${ft.renderMs?.p95}` : "N/A",
        st ? `${st.process_chunk_ms?.p50_ms}/${st.process_chunk_ms?.p95_ms}` : "N/A"],
      ["process_chunk max (ms)",
        ft ? `${ft.renderMs?.max}` : "N/A",
        st ? `${st.process_chunk_ms?.max_ms}` : "N/A"],
      ["Encode / write p50 (ms)",
        ft ? `${ft.encodeMs?.p50}` : "N/A",
        st ? `${st.queue_put_ms?.p50_ms}` : "N/A"],
      ["from_ndarray p50/p95 (ms)",
        "N/A (FFmpeg pipe)",
        st ? `${st.from_ndarray_ms?.p50_ms}/${st.from_ndarray_ms?.p95_ms}` : "N/A"],
      ["Segment cadence p50 (ms)",
        ft ? `${ft.segmentCadenceMs?.p50}` : "N/A",
        "N/A (RTP)"],
      ["Queue full retries",
        ft ? `${ft.writeQueueFullCount}` : "N/A",
        st ? `${st.queue_full_retries}` : "N/A"],
      ["Queue depth min/max",
        "N/A",
        st ? `${sfu.stopMetrics.queue_depth_min}/${sfu.stopMetrics.queue_depth_max}` : "N/A"],
      ["CPU% p50/max",
        "N/A (NVENC offloaded)",
        st ? `${st.cpu_percent?.p50}/${st.cpu_percent?.max}` : "N/A"],
      ["GPU mem MB p50/max",
        "N/A",
        st ? `${st.gpu_mem_mb?.p50}/${st.gpu_mem_mb?.max}` : "N/A"],
    );
  } else {
    rows.push(["  (no timing data — deploy with AVTR_MEDIA_TIMING_DEBUG=1)", "", ""]);
  }

  rows.push(["=" .repeat(25), "=".repeat(20), "=".repeat(20)]);

  // Print as aligned table
  const colW = [32, 22, 22];
  for (const row of rows) {
    const line = row.map((cell, i) => String(cell).padEnd(colW[i])).join("  ");
    console.log(`  ${line}`);
  }

  // Architecture notes
  console.log("\n  Architecture notes:");
  console.log("  - fMP4: GPU inference → GPU NVENC encode → SSE relay → MSE playback");
  console.log("  - SFU:  GPU inference → CPU PyAV encode (libvpx/libx264) → RTP → WebRTC");
  if (st?.cpu_percent?.p50 > 30) {
    console.log(`  ⚠ SFU CPU encode is significant: ${st.cpu_percent.p50}% (p50) of a core`);
  }
  if (ft && st && ft.renderMs?.p50 > 0 && st.process_chunk_ms?.p50_ms > 0) {
    console.log(`  - GPU render time: fMP4 ${ft.renderMs.p50}ms vs SFU ${st.process_chunk_ms.p50_ms}ms (should be ~equal)`);
  }
}

// ── Main ─────────────────────────────────────────────────────────────────────

(async () => {
  console.log("=== Media Pipeline Benchmark ===");
  console.log(`  mode: ${MODE}`);
  console.log(`  duration: ${DURATION}s`);
  console.log(`  base: ${BASE}`);

  const report = { mode: MODE, duration: DURATION, timestamp: new Date().toISOString(), results: {} };

  if (MODE === "fmp4" || MODE === "ab") {
    const r = await runFmp4Benchmark(DURATION);
    report.results.fmp4 = r;
  }

  if (MODE === "sfu" || MODE === "ab") {
    const r = await runSfuBenchmark(DURATION);
    report.results.sfu = r;
  }

  // Print results
  if (MODE === "ab") {
    printComparisonTable(report.results.fmp4, report.results.sfu);
  } else if (MODE === "fmp4") {
    printComparisonTable(report.results.fmp4, null);
  } else if (MODE === "sfu") {
    printComparisonTable(null, report.results.sfu);
  }

  // Session IDs
  console.log("\n  Session IDs:");
  if (report.results.fmp4) console.log(`    fMP4: ${report.results.fmp4.sessionId}`);
  if (report.results.sfu) console.log(`    SFU:  ${report.results.sfu.sessionId}`);

  // JSON output
  if (JSON_OUT) {
    const outPath = path.resolve(JSON_OUT);
    fs.writeFileSync(outPath, JSON.stringify(report, null, 2));
    console.log(`\n  JSON report: ${outPath}`);
  }

  process.exit(0);
})().catch((e) => {
  console.error("BENCHMARK CRASH:", e);
  process.exit(2);
});
