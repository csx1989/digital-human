// Stage 3A SFU Canary — single-session validation through production SFU routes.
//
// Flow:
//   1. POST /v1/sessions → session_id, ws_url, sfu_token, sfu URLs
//   2. WS connect → session.start {transport:"webrtc_sfu"} → wait session.ready
//   3. Launch Playwright chromium → browser subscribes via /sfu/subscribe|answer
//   4. Poll getStats() every 10s for 130s (ICE, framesDecoded, FPS, loss, jitter)
//   5. Teardown: POST /sfu/close, DELETE session
//
// Usage:
//   AVTR_API_KEY=... node scripts/sfu-canary.mjs
//
// Untracked — validation tooling only.

import { createRequire } from "module";
import { WebSocket } from "ws";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Playwright lives in the POC node_modules
const POC_NM = path.resolve(__dirname, "..", "avatar-saas-prototype", "realtime-sfu-poc", "phase-b-modal-publisher-poc", "node_modules");
const require = createRequire(import.meta.url + "/../"); // resolve from scripts dir fallback
let chromium;
try {
  chromium = require(path.join(POC_NM, "playwright")).chromium;
} catch {
  chromium = require("playwright").chromium;
}

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;
const RUN_SECONDS = Number(process.env.SFU_CANARY_SECONDS || 130);
const POLL_INTERVAL_MS = 10000;

// Threshold env vars (with safe defaults for the 130s single-session case).
// Override per canary class:
//   - Single-session (130s): defaults
//   - A/B cleanup (60s):      MIN_RUNTIME_SECONDS=55, MIN_FRAMES=1000
//   - Concurrent (90s):       MIN_RUNTIME_SECONDS=85, MIN_FRAMES=1800
const MIN_RUNTIME_SECONDS = Number(process.env.SFU_CANARY_MIN_RUNTIME_SECONDS || 120);
const MIN_FRAMES           = Number(process.env.SFU_CANARY_MIN_FRAMES || 2500);
const FPS_MIN              = Number(process.env.SFU_CANARY_FPS_MIN || 23);
const FPS_MAX              = Number(process.env.SFU_CANARY_FPS_MAX || 27);

// Frozen-frame detector (Stage 3B reliability repair).
// After WARMUP_S seconds (allow pipeline init), if the canvas fingerprint
// reports `frozenConsecutive` polls of "identical" frames in a row, that
// equals FROZEN_LIMIT_S of frozen video at POLL_INTERVAL_MS cadence and
// fails the canary even though getStats still reports 25fps and growing
// framesDecoded.
const FROZEN_WARMUP_S = Number(process.env.SFU_CANARY_FROZEN_WARMUP_S || 15);
const FROZEN_LIMIT_S  = Number(process.env.SFU_CANARY_FROZEN_LIMIT_S || 10);
const FROZEN_LIMIT_POLLS = Math.max(1, Math.round((FROZEN_LIMIT_S * 1000) / POLL_INTERVAL_MS));

if (!API_KEY) { console.error("AVTR_API_KEY required."); process.exit(1); }

const auth = { Authorization: `Bearer ${API_KEY}` };

// ── Helpers ─────────────────────────────────────────────────────────────────
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

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
    const msgs = [];
    ws.on("message", (raw) => {
      let m; try { m = JSON.parse(raw.toString()); } catch { return; }
      msgs.push(m);
      process.stdout.write(`  [ws<] ${m.type} ${m.status || m.code || ""}\n`.trimEnd() + "\n");
      if (m.type === wantType || m.type === "error") {
        clearTimeout(to);
        resolve({ msg: m, all: msgs });
      }
    });
    ws.on("error", (e) => { clearTimeout(to); reject(e); });
    ws.on("close", (c) => { clearTimeout(to); reject(new Error(`WS closed early code=${c}`)); });
  });
}

// ── Main ────────────────────────────────────────────────────────────────────
(async () => {
  console.log("=== Stage 3A SFU Canary ===");
  console.log(`Base: ${BASE}`);
  console.log(`Run:  ${RUN_SECONDS}s (thresholds: runtime>=${MIN_RUNTIME_SECONDS}s, frames>=${MIN_FRAMES}, fps ${FPS_MIN}-${FPS_MAX})\n`);

  // 1. Create session
  console.log("[1] Creating session...");
  const cr = await apiJson("POST", `${BASE}/v1/sessions`, {});
  if (cr.status !== 200) { console.error("create failed:", cr.status, cr.data); process.exit(1); }
  const session = cr.data.data;
  const sessionId = session.session_id;
  const wsUrl = session.stream.ws_url;
  const sfu = session.stream.sfu;
  if (!sfu) { console.error("SFU not advertised — config incomplete on Worker"); process.exit(1); }
  console.log(`    session: ${sessionId}`);
  console.log(`    sfu subscribe_url: ${sfu.subscribe_url.split("?")[0]}`);
  console.log(`    available_transports: ${session.stream.available_transports}`);
  console.log(`    default transport: ${session.stream.transport}`);

  // 2. WS connect + session.start with webrtc_sfu
  console.log("\n[2] WS connect + session.start {transport:webrtc_sfu}...");
  const ws = new WebSocket(wsUrl);
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });
  console.log("    WS connected");

  ws.send(JSON.stringify({ type: "session.start", avatarId: "arnold", bgId: "aloe_window", transport: "webrtc_sfu" }));
  console.log('    sent session.start {transport:"webrtc_sfu"}');

  const startResult = await wsCollect(ws, 120000, "session.ready");
  const readyMsg = startResult.msg;
  if (readyMsg.type === "error") {
    console.error("    session.start ERROR:", readyMsg.code, readyMsg.message);
    ws.close(); process.exit(1);
  }
  console.log(`    session.ready: transport=${readyMsg.transport}, sfu.track=${readyMsg.sfu?.publisher_track_name || "?"}`);

  // Stage 3B: Script-side heartbeat is now HARD-DISABLED.
  // The DO alarm-based keepalive (sfu_modal_keepalive_*) owns container
  // keepalive. This canary must prove the DO keepalive works, not the
  // script keepalive. Set SCRIPT_HEARTBEAT_ENABLED=true only for debugging.
  const SCRIPT_HEARTBEAT_ENABLED = false;
  const heartbeat = SCRIPT_HEARTBEAT_ENABLED ? setInterval(async () => {
    try {
      await apiJson("GET", `${BASE}/v1/sessions/${sessionId}/status`);
    } catch { /* best-effort */ }
  }, 15000) : null;
  if (SCRIPT_HEARTBEAT_ENABLED) {
    console.log("    [WARNING] Script heartbeat ENABLED — DO keepalive NOT being tested!");
  } else {
    console.log("    [OK] Script heartbeat DISABLED — DO alarm keepalive owns container survival");
  }

  // 3. Launch Playwright + subscriber
  console.log("\n[3] Launching chromium subscriber...");
  const browser = await chromium.launch({
    headless: true,
    args: ["--use-fake-ui-for-media-stream", "--autoplay-policy=no-user-gesture-required", "--mute-audio"],
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });

  const pageErrors = [];
  page.on("pageerror", (e) => pageErrors.push(String(e)));
  page.on("console", (m) => {
    const t = m.text();
    if (/error|fail|ice|track|subscribe|answer/i.test(t)) console.log(`  [browser] ${t}`);
  });

  // Minimal HTML with a video element + hidden canvas for frozen-frame detection
  await page.setContent(`<!DOCTYPE html><html><body>
    <video id="v" autoplay playsinline muted style="width:640px;height:360px;background:#000"></video>
    <canvas id="fc" width="32" height="18" style="display:none"></canvas>
    <div id="diag"></div>
  </body></html>`);

  // Subscriber setup inside browser — does SDP exchange via production routes
  console.log("    subscribing via /sfu/subscribe ...");
  const subInfo = await page.evaluate(async (params) => {
    const { subscribeUrl, answerUrl, token } = params;
    const diag = (s) => { document.getElementById("diag").textContent = s; };

    // 3a. POST subscribe → SFU offer
    const subResp = await fetch(`${subscribeUrl}?token=${encodeURIComponent(token)}`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: "{}",
    });
    if (!subResp.ok) { diag(`subscribe failed: ${subResp.status}`); return { error: `subscribe ${subResp.status}: ${await subResp.text()}` }; }
    const sub = await subResp.json();
    if (sub.error) return { error: `subscribe: ${JSON.stringify(sub.error)}` };

    // 3b. Create PC + setRemoteDescription(offer)
    const pc = new RTCPeerConnection();
    window._pc = pc;
    let iceState = "new";
    pc.oniceconnectionstatechange = () => { iceState = pc.iceConnectionState; };

    const video = document.getElementById("v");
    let trackEvent = null;
    pc.ontrack = (e) => {
      trackEvent = { kind: e.track.kind, id: e.track.id, streams: e.streams.length };
      video.srcObject = e.streams[0];
    };

    await pc.setRemoteDescription({ type: sub.offer_type || "offer", sdp: sub.offer_sdp });
    diag("remote desc set");

    // 3c. createAnswer + wait for ICE gathering
    const answer = await pc.createAnswer();
    await pc.setLocalDescription(answer);
    diag("waiting ICE gathering...");
    if (pc.iceGatheringState !== "complete") {
      await new Promise((resolve) => {
        const check = () => { if (pc.iceGatheringState === "complete") { pc.removeEventListener("icegatheringstatechange", check); resolve(); } };
        pc.addEventListener("icegatheringstatechange", check);
        setTimeout(resolve, 8000); // cap at 8s
      });
    }
    diag("ICE gathering done");

    // 3d. POST answer back
    const ansResp = await fetch(`${answerUrl}?token=${encodeURIComponent(token)}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer_sdp: pc.localDescription.sdp, answer_type: pc.localDescription.type }),
    });
    if (!ansResp.ok) { diag(`answer failed: ${ansResp.status}`); return { error: `answer ${ansResp.status}: ${await ansResp.text()}` }; }
    const ansBody = await ansResp.json();
    if (ansBody.error) return { error: `answer: ${JSON.stringify(ansBody.error)}` };
    diag("answer accepted — waiting for media");

    // ── Frozen-frame detector (Stage 3B reliability repair) ──
    // Draw the video to a 32x18 grayscale canvas and compute a fingerprint
    // every poll.  The canary main loop (Node) reads `frozen*` fields and
    // fails the run if frames stay identical for > FROZEN_LIMIT_S after
    // warmup.  Browser getStats cannot detect this: the WebRTC track emits
    // a "new" frame at 25fps even when the publisher is repeating the
    // last good buffer (e.g. after a CUDA illegal memory access).
    const fc = document.getElementById("fc");
    const fcx = fc.getContext("2d", { willReadFrequently: true });
    let lastPixels = null;            // Uint8ClampedArray of previous frame
    let consecutiveIdentical = 0;     // polls in a row where frame ≈ previous
    let firstChangedAt = null;        // timestamp of first NON-identical frame
    const IDENTICAL_MAX_DELTA = 50;   // sum-of-abs-diff over 32*18 px below this = identical

    window._frozenFingerprint = () => {
      try {
        fcx.drawImage(video, 0, 0, 32, 18);
        const img = fcx.getImageData(0, 0, 32, 18).data;
        // Convert RGBA → grayscale (luma) bytes
        const gray = new Uint8ClampedArray(32 * 18);
        for (let i = 0; i < 32 * 18; i++) {
          gray[i] = (img[i*4] + img[i*4+1] + img[i*4+2]) / 3;
        }
        if (lastPixels === null) {
          lastPixels = gray;
          return { identical: false, delta: null, consecutive: 0 };
        }
        let delta = 0;
        for (let i = 0; i < gray.length; i++) {
          delta += Math.abs(gray[i] - lastPixels[i]);
        }
        const identical = delta < IDENTICAL_MAX_DELTA;
        if (identical) {
          consecutiveIdentical++;
        } else {
          consecutiveIdentical = 0;
          if (firstChangedAt === null) firstChangedAt = Date.now();
        }
        lastPixels = gray;
        return { identical, delta, consecutive: consecutiveIdentical, firstChangedAt };
      } catch (e) {
        return { identical: false, delta: null, consecutive: 0, error: String(e) };
      }
    };

    // 3e. Expose stats poller on window
    window._pollStats = async () => {
      const stats = await pc.getStats();
      let v = null; // inbound-rtp video
      let cp = null; // candidate-pair
      stats.forEach((r) => {
        if (r.type === "inbound-rtp" && r.kind === "video") v = r;
        if (r.type === "candidate-pair" && r.nominated) cp = r;
      });
      const playing = video.currentTime > 0 && !video.paused;
      const dims = video.videoWidth ? `${video.videoWidth}x${video.videoHeight}` : "0x0";
      // Sample the frozen-frame fingerprint in the same poll cadence.
      const fp = window._frozenFingerprint();
      return {
        iceState,
        playing,
        dimensions: dims,
        framesDecoded: v ? v.framesDecoded : null,
        fps: v ? v.framesPerSecond : null,
        packetsLost: v ? v.packetsLost : null,
        jitter: v ? v.jitter : null,
        nacksSent: v ? v.nackCount : null,
        rtt: cp ? cp.currentRoundTripTime : null,
        frozenIdentical: fp.identical,
        frozenDelta: fp.delta,
        frozenConsecutive: fp.consecutive,
        frozenFirstChangedAt: fp.firstChangedAt,
      };
    };
    return { ok: true, mids: sub.mids, trackEvent };
  }, { subscribeUrl: sfu.subscribe_url, answerUrl: sfu.answer_url, token: sfu.token });

  if (subInfo.error) {
    console.error("    SUBSCRIBE FAILED:", subInfo.error);
    pageErrors.push(subInfo.error);
    if (heartbeat) clearInterval(heartbeat);
    // teardown attempt
    try { await apiJson("POST", `${sfu.close_url}?token=${encodeURIComponent(sfu.token)}`, {}); } catch {}
    try { await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`); } catch {}
    await browser.close(); ws.close();
    process.exit(1);
  }
  console.log(`    subscribed! mids=${JSON.stringify(subInfo.mids)} track=${JSON.stringify(subInfo.trackEvent)}`);

  // 4. Poll metrics
  console.log(`\n[4] Monitoring ${RUN_SECONDS}s (poll every ${POLL_INTERVAL_MS / 1000}s)...`);
  console.log(`    frozen-frame detector: warmup=${FROZEN_WARMUP_S}s, limit=${FROZEN_LIMIT_S}s (${FROZEN_LIMIT_POLLS} consecutive identical polls)`);
  const snapshots = [];
  const t0 = Date.now();
  let frozenViolation = null;  // { atElapsed, consecutivePolls, delta }
  for (let i = 0; i < Math.ceil(RUN_SECONDS / (POLL_INTERVAL_MS / 1000)); i++) {
    await sleep(POLL_INTERVAL_MS);
    const elapsed = Math.floor((Date.now() - t0) / 1000);
    let s;
    try { s = await page.evaluate(() => window._pollStats()); }
    catch (e) { s = { error: String(e) }; }
    s.elapsed = elapsed;
    snapshots.push(s);
    if (s.error) {
      console.log(`  [${elapsed}s] POLL ERROR: ${s.error}`);
    } else {
      console.log(
        `  [${elapsed}s] ice=${s.iceState} play=${s.playing} dims=${s.dimensions} decoded=${s.framesDecoded} ` +
        `fps=${s.fps?.toFixed(1)} lost=${s.packetsLost} jitter=${s.jitter?.toFixed(3)} rtt=${s.rtt?.toFixed(3)} ` +
        `frozen=${s.frozenIdentical ? `STUCK×${s.frozenConsecutive}` : "live"}(Δ=${s.frozenDelta ?? "?"})`
      );
    }
    // Early abort on hard failure
    if (s.iceState === "failed") { console.log("  ICE FAILED — aborting early"); break; }

    // Frozen-frame early abort: only after warmup, only if first frame changed
    // (otherwise we'd false-positive during initial black-frame → real-frame).
    if (
      elapsed > FROZEN_WARMUP_S &&
      s.frozenFirstChangedAt !== null &&
      s.frozenConsecutive >= FROZEN_LIMIT_POLLS &&
      !frozenViolation
    ) {
      frozenViolation = {
        atElapsed: elapsed,
        consecutivePolls: s.frozenConsecutive,
        delta: s.frozenDelta,
      };
      console.log(
        `  FROZEN-FRAME VIOLATION at ${elapsed}s: ${s.frozenConsecutive} consecutive identical polls ` +
        `(Δ=${s.frozenDelta}, limit=${FROZEN_LIMIT_POLLS}). Aborting early — publisher likely repeating stale frames.`
      );
      break;
    }
  }

  // 5. Teardown
  console.log("\n[5] Teardown...");
  if (heartbeat) clearInterval(heartbeat);
  // Subscriber close
  let closeResp = null;
  try {
    closeResp = await apiJson("POST", `${sfu.close_url}?token=${encodeURIComponent(sfu.token)}`, {});
    console.log(`    sfu/close: ${closeResp.status} ${closeResp.data?.status || closeResp.data?.error?.code || ""}`);
  } catch (e) { console.log(`    sfu/close: ERROR ${e.message}`); }
  await browser.close();

  // WS session.end
  try {
    ws.send(JSON.stringify({ type: "session.end" }));
    await sleep(2000);
  } catch { /* best-effort */ }
  ws.close();

  // REST end session
  try {
    const endResp = await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`);
    console.log(`    DELETE session: ${endResp.status} ${endResp.data?.data?.status || ""}`);
    console.log(`    modal_end_ok: ${endResp.data?.data?.modal_end_ok}`);
  } catch (e) { console.log(`    DELETE session: ERROR ${e.message}`); }

  // 6. Evaluate
  const last = snapshots.filter(s => !s.error).slice(-1)[0] || {};
  const first = snapshots.find(s => s.framesDecoded != null) || {};
  const firstDecoded = first.framesDecoded ?? null;
  const lastDecoded = last.framesDecoded ?? null;
  const recentFps = snapshots.filter(s => s.fps != null).slice(-5).map(s => s.fps).sort((a,b)=>a-b);
  const medianFps = recentFps.length ? recentFps[Math.floor(recentFps.length/2)] : null;
  const runtime = last.elapsed || 0;
  const dimsOk = last.dimensions && last.dimensions !== "0x0";

  const closeStatus = closeResp?.data?.status;

  const criteria = [
    { name: `Runtime >= ${MIN_RUNTIME_SECONDS}s`, pass: runtime >= MIN_RUNTIME_SECONDS, detail: `${runtime}s` },
    { name: "ICE connected/completed", pass: ["connected","completed"].includes(last.iceState), detail: last.iceState || "N/A" },
    { name: "Remote video playing", pass: !!last.playing, detail: String(last.playing) },
    { name: "Dimensions valid", pass: !!dimsOk, detail: last.dimensions || "N/A" },
    { name: `Final framesDecoded >= ${MIN_FRAMES}`, pass: lastDecoded != null && lastDecoded >= MIN_FRAMES, detail: String(lastDecoded) },
    { name: `Median FPS between ${FPS_MIN}-${FPS_MAX}`, pass: medianFps != null && medianFps >= FPS_MIN && medianFps <= FPS_MAX, detail: `${medianFps} (median last 5)` },
    { name: "No page errors", pass: pageErrors.length === 0, detail: `${pageErrors.length} error(s)` },
    { name: "packetsLost < 100", pass: (last.packetsLost ?? 9999) < 100, detail: String(last.packetsLost) },
    { name: "sfu/close succeeded", pass: closeStatus === "closed" || closeStatus === "closed_best_effort", detail: closeStatus || "N/A" },
    { name: "Script heartbeat disabled", pass: !SCRIPT_HEARTBEAT_ENABLED, detail: SCRIPT_HEARTBEAT_ENABLED ? "ENABLED (BAD)" : "disabled (OK)" },
    { name: `No frozen frames (limit ${FROZEN_LIMIT_S}s after ${FROZEN_WARMUP_S}s warmup)`, pass: !frozenViolation, detail: frozenViolation ? `STUCK×${frozenViolation.consecutivePolls} at ${frozenViolation.atElapsed}s Δ=${frozenViolation.delta}` : "live" },
  ];
  const allPassed = criteria.every(c => c.pass);

  console.log("\n=== METRICS ===");
  console.log(`Runtime:        ${runtime}s`);
  console.log(`ICE final:      ${last.iceState || "N/A"}`);
  console.log(`Dimensions:     ${last.dimensions || "N/A"}`);
  console.log(`Frames decoded: ${firstDecoded} → ${lastDecoded}`);
  console.log(`FPS (median):   ${medianFps}`);
  console.log(`Packets lost:   ${last.packetsLost}`);
  console.log(`Jitter:         ${last.jitter}`);
  console.log(`RTT:            ${last.rtt}`);
  console.log(`Page errors:    ${pageErrors.length}`);
  console.log(`Frozen frames:  ${frozenViolation ? `VIOLATION at ${frozenViolation.atElapsed}s (consecutive=${frozenViolation.consecutivePolls}, Δ=${frozenViolation.delta})` : "none detected"}`);
  if (pageErrors.length) pageErrors.forEach(e => console.log(`  - ${e}`));

  console.log("\n=== PASS/FAIL ===");
  criteria.forEach(c => console.log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name} (${c.detail})`));
  console.log(`\n=== OVERALL: ${allPassed ? "PASS" : "FAIL"} ===`);
  console.log(`session_id: ${sessionId}`);
  process.exit(allPassed ? 0 : 1);
})().catch((e) => { console.error("CANARY CRASH:", e); process.exit(2); });
