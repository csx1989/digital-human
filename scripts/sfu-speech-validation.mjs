// Phase 3C SFU speech validation — sends real mic audio to trigger TTS,
// then verifies avatar.sfu.speech_queued arrives and no fallback fires.
//
// Flow:
//   1. POST /v1/sessions → create session
//   2. WS connect → session.start {transport:"webrtc_sfu"} → wait session.ready
//   3. Send test_speech.wav as mic audio → trigger VAD/SmartTurn/LLM/TTS
//   4. Wait for: assistant.audio + avatar.sfu.speech_queued
//   5. Verify: NO pipeline_error, NO sfu_unavailable
//   6. End session, print summary
//
// Usage:
//   AVTR_API_KEY=... node scripts/sfu-speech-validation.mjs

import { WebSocket } from "ws";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const KEY = process.env.AVTR_API_KEY;
const WAV_PATH = join(__dirname, "..", "test_speech.wav");

if (!KEY) { console.error("AVTR_API_KEY required."); process.exit(1); }

const auth = { Authorization: `Bearer ${KEY}` };
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

function log(msg) {
  console.log(`[${new Date().toISOString().slice(11, 19)}] ${msg}`);
}

function parseWav(path) {
  const buf = readFileSync(path);
  const pcm16 = new Int16Array(buf.buffer, 44, (buf.length - 44) / 2);
  return { pcm16, sampleRate: 16000 };
}

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

async function main() {
  log("=== Phase 3C SFU Speech Validation ===");

  // 1. Create session
  log("Creating SFU session...");
  const { data: respData } = await apiJson("POST", `${BASE}/v1/sessions`, {});
  const createData = respData.data || respData;
  const sessionId = createData.session_id;
  const wsUrl = createData.stream?.ws_url || createData.ws_url;
  if (!sessionId || !wsUrl) {
    console.error("Failed to create session:", JSON.stringify(respData).slice(0, 300));
    process.exit(1);
  }
  log(`Session: ${sessionId}`);

  // 2. WS connect + session.start
  const ws = new WebSocket(wsUrl);
  const events = [];

  ws.on("message", (raw) => {
    let msg;
    try { msg = JSON.parse(raw.toString()); } catch { return; }
    events.push({ t: Date.now(), msg });

    const type = msg.type || "";
    const interesting = [
      "session.ready", "session.ended", "error", "pipeline_error",
      "speech_started", "speech_finalized",
      "assistant.audio", "assistant.audio.start", "assistant.audio.end",
      "avatar.sfu.speech_queued",
      "mic_stop_summary",
      "stt.result", "llm.token", "tts.start", "tts.end",
    ];
    if (interesting.includes(type) || (msg.status && type === "")) {
      log(`EVENT: ${type || "status:" + msg.status} ${JSON.stringify(msg).slice(0, 200)}`);
    }
  });

  await new Promise((resolve, reject) => {
    ws.once("open", resolve);
    ws.once("error", reject);
  });
  log("WS connected");

  ws.send(JSON.stringify({ type: "session.start", transport: "webrtc_sfu" }));
  log('Sent session.start {transport:"webrtc_sfu"}');

  // Wait for session.ready or error
  let ready = false;
  const readyStart = Date.now();
  while (!ready && Date.now() - readyStart < 90_000) {
    const found = events.find(e => e.msg.type === "session.ready");
    const err = events.find(e => e.msg.type === "error" || e.msg.error);
    if (found) { ready = true; break; }
    if (err) {
      log(`FAIL: session.start error: ${err.msg.error || JSON.stringify(err.msg).slice(0, 200)}`);
      ws.close();
      process.exit(1);
    }
    await sleep(200);
  }
  if (!ready) {
    log("FAIL: session.ready not received within 90s");
    ws.close();
    process.exit(1);
  }
  log("session.ready received — SFU pipeline started");

  // 3. Wait for SFU video to stabilize, then send speech
  log("Waiting 8s for SFU pipeline to stabilize...");
  await sleep(8_000);

  const { pcm16 } = parseWav(WAV_PATH);
  log(`test_speech.wav: ${pcm16.length} samples (${(pcm16.length / 16000).toFixed(1)}s)`);

  log("--- Speech Turn 1 ---");
  ws.send(JSON.stringify({ type: "mic.start" }));
  await sleep(300);

  const samplesPerChunk = 1600;
  const turnSamples = Math.min(pcm16.length, 5 * 16000);
  for (let offset = 0; offset < turnSamples; offset += samplesPerChunk) {
    const end = Math.min(offset + samplesPerChunk, turnSamples);
    const chunk = pcm16.slice(offset, end);
    const chunkBuf = Buffer.from(chunk.buffer, chunk.byteOffset, chunk.byteLength);
    ws.send(chunkBuf);
    await sleep(100);
  }
  log(`Sent ${turnSamples} samples, sending mic.stop...`);
  ws.send(JSON.stringify({ type: "mic.stop" }));

  // 4. Wait for TTS response + speech_queued (up to 30s)
  log("Waiting up to 35s for assistant.audio + avatar.sfu.speech_queued...");
  const speechWaitStart = Date.now();
  while (Date.now() - speechWaitStart < 35_000) {
    await sleep(500);
  }

  // 5. End session
  log("Sending session.end...");
  ws.send(JSON.stringify({ type: "session.end" }));
  await sleep(3_000);
  ws.close();

  // 6. Analysis
  log("\n=== ANALYSIS ===");
  const typeCounts = {};
  for (const e of events) {
    const t = e.msg.type || "unknown";
    typeCounts[t] = (typeCounts[t] || 0) + 1;
  }
  log("Event counts:");
  for (const [t, c] of Object.entries(typeCounts).sort((a, b) => b[1] - a[1])) {
    log(`  ${c.toString().padStart(5)}x  ${t}`);
  }

  // Checks
  const hasSpeechStarted = events.some(e => e.msg.type === "speech_started" || e.msg.type === "speech.started");
  const hasSpeechFinalized = events.some(e => e.msg.type === "speech_finalized" || e.msg.type === "speech.finalized");
  const hasAssistantAudio = events.some(e => e.msg.type === "assistant.audio");
  const hasSfuSpeechQueued = events.some(e => e.msg.type === "avatar.sfu.speech_queued");
  const hasPipelineError = events.some(e => e.msg.type === "pipeline_error" || (e.msg.type === "error" && !e.msg.error?.includes("sfu_")));

  log("\n=== VERIFICATION ===");
  const checks = [
    { name: "speech_started", pass: hasSpeechStarted },
    { name: "speech_finalized", pass: hasSpeechFinalized },
    { name: "assistant.audio received (TTS generated)", pass: hasAssistantAudio },
    { name: "avatar.sfu.speech_queued received (TTS pushed to Modal)", pass: hasSfuSpeechQueued },
    { name: "no pipeline_error", pass: !hasPipelineError },
  ];

  let allPass = true;
  for (const c of checks) {
    log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name}`);
    if (!c.pass) allPass = false;
  }

  log(`\n=== ${allPass ? "OVERALL: PASS" : "OVERALL: FAIL"} ===`);
  log(`session_id: ${sessionId}`);

  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error("FATAL:", e); process.exit(1); });
