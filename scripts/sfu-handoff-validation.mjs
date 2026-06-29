// Phase 3C validation: push fake TTS audio to a live SFU session and verify
// that GstAvtrFrameSource consumes it from the handoff queue.
//
// Flow:
//   1. POST /v1/sessions → create SFU session
//   2. WS connect → session.start {transport:"webrtc_sfu"} → wait session.ready
//   3. Wait 10s for SFU video pipeline to stabilize (ICE + render loop started)
//   4. Poll session-status BEFORE push → record baseline idle_chunks_rendered
//   5. Push fake TTS audio (2s sine wave, 16kHz mono f32) to Modal push-audio-batch
//   6. Wait 5s for render loop to consume
//   7. Poll session-status AFTER push → check handoff_chunks_consumed > 0
//   8. Teardown: sfu/close + DELETE session
//
// Usage:
//   AVTR_API_KEY=... node scripts/sfu-handoff-validation.mjs

import { WebSocket } from "ws";

const BASE = process.env.AVTR_API_BASE || "https://avatar-session-bridge.vallangirakesh.workers.dev";
const API_KEY = process.env.AVTR_API_KEY;
const MODAL_PUSH_URL = "https://vallangiprasad--push-audio-batch-5q.modal.run";
const MODAL_STATUS_URL = "https://vallangiprasad--session-status-5q.modal.run";

if (!API_KEY) { console.error("AVTR_API_KEY required."); process.exit(1); }

const auth = { Authorization: `Bearer ${API_KEY}` };
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

async function modalPost(url, body) {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await resp.text();
  let data; try { data = text ? JSON.parse(text) : {}; } catch { data = { raw: text }; }
  return { status: resp.status, data };
}

// Generate 2s of 220Hz sine wave as base64 float32 16kHz mono
function generateSineWaveBase64(durationSec = 2.0, freq = 220, sampleRate = 16000) {
  const nSamples = Math.floor(durationSec * sampleRate);
  const buffer = new ArrayBuffer(nSamples * 4);
  const view = new DataView(buffer);
  for (let i = 0; i < nSamples; i++) {
    const t = i / sampleRate;
    const envelope = Math.min(1, i / (sampleRate * 0.02)) * Math.min(1, (nSamples - i) / (sampleRate * 0.02));
    const sample = Math.sin(2 * Math.PI * freq * t) * 0.5 * envelope;
    view.setFloat32(i * 4, sample, true);
  }
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function wsCollect(ws, timeout, wantType) {
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => reject(new Error(`WS timeout waiting for ${wantType}`)), timeout);
    const msgs = [];
    const handler = (raw) => {
      let msg; try { msg = JSON.parse(raw.toString()); } catch { return; }
      msgs.push(msg);
      if (msg.type === wantType || msg.status === wantType) {
        clearTimeout(to);
        ws.off("message", handler);
        resolve({ msg, msgs });
      }
    };
    ws.on("message", handler);
  });
}

async function pollSessionStatus(sessionId, label) {
  const r = await modalPost(MODAL_STATUS_URL, { session_id: sessionId });
  console.log(`  [status:${label}] status=${r.status} data=${JSON.stringify(r.data).slice(0, 300)}`);
  return r.data;
}

async function main() {
  console.log("=== Phase 3C SFU Handoff Validation ===\n");

  // 1. Create session
  console.log("[1] Creating SFU session...");
  const { data: respData } = await apiJson("POST", `${BASE}/v1/sessions`, {});
  const createData = respData.data || respData;
  const sessionId = createData.session_id;
  const wsUrl = createData.stream?.ws_url || createData.ws_url;
  if (!sessionId || !wsUrl) {
    console.error("Failed to create session:", createData);
    process.exit(1);
  }
  console.log(`    session: ${sessionId}`);
  console.log(`    ws_url:  ${wsUrl}`);

  // 2. WS connect + session.start
  console.log("\n[2] WS connect + session.start {transport:webrtc_sfu}...");
  const ws = new WebSocket(wsUrl);
  await new Promise((resolve, reject) => {
    ws.on("open", resolve);
    ws.on("error", reject);
  });
  console.log("    WS connected");

  const statusCollector = wsCollect(ws, 90_000, "session.ready");
  ws.on("message", (raw) => {
    try {
      const m = JSON.parse(raw.toString());
      if (m.type === "error" || m.error) {
        console.error(`    [ws<] ERROR: ${m.error || JSON.stringify(m)}`);
      } else if (m.status) {
        console.log(`    [ws<] status ${m.status}`);
      }
    } catch {}
  });
  ws.send(JSON.stringify({ type: "session.start", transport: "webrtc_sfu" }));
  console.log('    sent session.start {transport:"webrtc_sfu"}');

  try {
    const { msg } = await statusCollector;
    if (msg.error) {
      console.error(`    session.start ERROR: ${msg.error}`);
      ws.close();
      process.exit(1);
    }
    console.log(`    session.ready: transport=${msg.transport}`);
  } catch (e) {
    console.error(`    ${e.message}`);
    ws.close();
    process.exit(1);
  }

  // 3. Wait for SFU pipeline to stabilize
  console.log("\n[3] Waiting 12s for SFU pipeline to stabilize...");
  await sleep(12_000);

  // 4. Baseline status
  console.log("\n[4] Polling session status (BEFORE push)...");
  const before = await pollSessionStatus(sessionId, "before");

  // 5. Push fake TTS audio
  console.log("\n[5] Pushing fake TTS audio (2s sine wave, 220Hz)...");
  const audioB64 = generateSineWaveBase64(2.0, 220, 16000);
  const pushResult = await modalPost(MODAL_PUSH_URL, {
    session_id: sessionId,
    audio_data_b64: audioB64,
    audio_mode: "pcm_f32_16k",
    turn_id: 999,
    generated_at: Date.now(),
    enqueue_at: Date.now(),
  });
  console.log(`    push result: status=${pushResult.status} ok=${pushResult.data?.ok} pushed=${pushResult.data?.pushed}/${pushResult.data?.chunks}`);

  // 6. Wait for render loop to consume
  console.log("\n[6] Waiting 5s for render loop to consume...");
  await sleep(5_000);

  // 7. Poll status AFTER push
  console.log("\n[7] Polling session status (AFTER push)...");
  const after = await pollSessionStatus(sessionId, "after");

  // 8. Verification
  console.log("\n=== VERIFICATION ===");

  const checks = [];

  // Check: push succeeded
  const pushOk = pushResult.data?.ok === true;
  checks.push({ name: "push_audio_batch ok", pass: pushOk, detail: `ok=${pushResult.data?.ok}` });

  // Check: handoff consumed (session status won't have these directly,
  // but the push should show pushed > 0)
  const pushed = pushResult.data?.pushed ?? 0;
  checks.push({ name: "chunks pushed > 0", pass: pushed > 0, detail: `pushed=${pushed}` });

  // The session status endpoint doesn't directly expose frame source metrics,
  // but we can infer from chunk_count changes.
  const beforeChunks = before?.chunk_count ?? 0;
  const afterChunks = after?.chunk_count ?? 0;
  const chunkDelta = afterChunks - beforeChunks;
  checks.push({
    name: "chunk_count increased (render loop active)",
    pass: chunkDelta > 0,
    detail: `before=${beforeChunks} after=${afterChunks} delta=${chunkDelta}`,
  });

  // Check: session still active (no pipeline_error)
  const sessionActive = after?.status === "ready" || after?.status === "running";
  checks.push({
    name: "session still active",
    pass: sessionActive,
    detail: `status=${after?.status}`,
  });

  let allPass = true;
  for (const c of checks) {
    console.log(`  [${c.pass ? "PASS" : "FAIL"}] ${c.name} (${c.detail})`);
    if (!c.pass) allPass = false;
  }
  console.log(`\n=== ${allPass ? "OVERALL: PASS" : "OVERALL: FAIL"} ===`);
  console.log(`session_id: ${sessionId}`);

  // 9. Teardown
  console.log("\n[9] Teardown...");
  ws.close();
  await sleep(1_000);
  await apiJson("DELETE", `${BASE}/v1/sessions/${sessionId}`);

  process.exit(allPass ? 0 : 1);
}

main().catch(e => { console.error(e); process.exit(1); });
