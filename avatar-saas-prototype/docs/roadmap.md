# Roadmap

Phased implementation plan for the avatar SaaS prototype.

## Phase 1: Scaffold and Repo Isolation

**Status: Completed**

Create the isolated prototype workspace with documentation, placeholder code,
and a draft protocol. No working integration.

- Directory structure with README files
- Target architecture documentation
- Draft event protocol
- Skeleton code for Modal runtime, Cloudflare bridge, and web client
- Zero modifications to existing production code

## Phase 2: Analyze Avatar Runtime Requirements

**Status: Completed**

Deep-dive into the AVTR-1 renderer to define the exact interface the Modal
service must expose.

- Document the renderer's stateful chunk protocol in detail
- Identify minimum GPU memory and compute requirements
- Profile cold start vs warm latency
- Define the Modal service API contract (input/output format, session lifecycle)
- Decide: wrap the existing HTTP API as-is, or design a new streaming interface

### Deliverables

- `docs/avatar-runtime-analysis.md` — Full analysis of the existing renderer:
  entry points, API endpoints, exact request/response contracts, audio timing,
  stateful behavior, rendering internals, runtime requirements, risks, and open
  questions. 30+ source files inspected.

- `docs/modal-runtime-contract.md` — Proposed Modal service API for Phase 3:
  recommended approach (wrap existing renderer), 4 endpoints with schemas,
  state management recommendation (server-side for prototype), first render
  test plan, and go/no-go criteria.

### Key findings

- The existing `/process-audio-v3` endpoint is a clean HTTP service ready to
  wrap. No redesign needed.
- Audio format: int16 PCM 16kHz mono, 4 fields (current+future x speech+listen),
  exact byte counts: 6400/6560 bytes.
- State is client-carried safetensors (~627 KiB) — server-side state on Modal
  eliminates D2H/H2D overhead.
- AVTR1 encode/decode engines are TRT-only (no ONNX fallback) — this is the
  biggest Modal risk.
- Feasible for prototype on T4, but requires measurement of cold start, VRAM,
  and TRT engine compatibility.

### Open questions (require Phase 3 measurement)

- CUDA/TensorRT version compatibility between local build and Modal T4
- Exact cold start time with all engines + avatar registration
- VRAM footprint with all 7 engines loaded
- Whether pre-built TRT engines work on Modal or must be rebuilt in-container

## Phase 3: Modal GPU Feasibility, Snapshotting, and Runtime Decision

**Status: Completed**

Deploy and benchmark the AVTR-1 renderer on Modal GPUs, then decide the runtime
baseline for the prototype.

- Package the renderer as a Modal app
- Support avatar preload and stateful chunk processing
- Expose a network-callable API (HTTP or WebSocket)
- Measure time-to-first-frame and sustained throughput
- Decide if T4 is sufficient or if upgrade to L4/A10G is needed
- Validate Modal GPU snapshotting for cold-start reduction
- Reject or defer optimization experiments that do not produce meaningful wins

### Phase 3a: Smoke Tests — COMPLETED

Three Modal functions answering: "Can AVTR-1 load and render one chunk on
Modal T4?"

All three tests passed:

- `test_imports()` — CPU, all packages verified (PyTorch 2.7.1+cu128, TensorRT
  10.11, ONNX Runtime 1.26, etc.)
- `test_pipeline_startup()` — T4, pipeline loaded in 13.7s, 0.14 GiB (torch)
- `test_single_chunk_render()` — T4, 5 frames @ 720x1280 in 1339ms (cold start)

All 7 TRT engine files built on T4 (sm75) and persisted in Modal Volume.
Ten bugs found and fixed during the run.

See `docs/modal-smoke-results.md` for full results and caveats.

**Deliverables**:
- `modal-avatar-runtime/modal_smoke.py` — Modal app with smoke-test functions
- `modal-avatar-runtime/README.md` — Commands, troubleshooting, architecture
- `docs/modal-smoke-results.md` — Honest results with limitations documented

**Key findings**:
- AVTR-1 can load and render on Modal T4 (functional proof).
- Cold-start render: 1339ms for 5 frames (6.7x over 200ms realtime budget).
- The current smoke-test path checks/requires all 7 TRT engine files.
  Engine-level runtime verification should be confirmed from logs or code
  instrumentation.
- `torch.cuda.memory_allocated()` under-reports VRAM; need `nvidia-smi`.
- Realtime capability is **not yet proven**.

### Phase 3b: Benchmark — COMPLETED

Warm consecutive chunk benchmark on T4.

- p95 = 367.7ms, FPS = 13.7 (10 chunks, all produced 5 frames)
- **FUNCTIONAL: PASSED**, **REALTIME: FAILED** (p95 > 200ms)
- T4 marginal (200-500ms band). Not close to 200ms.
- VRAM: 3501/15360 MiB (23% usage)

See `docs/modal-smoke-results.md` for full results.

### Phase 3c: Snapshot + Easy Optimizations — COMPLETED

Reduce cold start via GPU memory snapshotting and test easy TRT build
optimizations as an isolated A/B experiment.

- `rebuild_optimized_engines()` — builds engines with
  builder_optimization_level=3 + tiling_optimization_level=FAST into
  separate Volume `avtr1-artifacts-opt` (baseline untouched)
- `benchmark_optimized()` — warm benchmark with optimized engines
- `AVTR1Snapshot` class — GPU memory snapshot with @modal.enter(snap=True)
- Methods: health(), benchmark_first_render(), benchmark_render()
- Strict separation: cold-start metric (snapshot) vs warm render metric (engines)

**Key file**: `modal-avatar-runtime/modal_snapshot_smoke.py`

**Constraints**:
- No modifications to production/root renderer source
- All changes inside `avatar-saas-prototype/`
- GPU snapshots only work for deployed apps (not `modal run`)
- Baseline engines in `avtr1-artifacts` Volume are never modified
- Do not claim realtime unless p95 <= 200ms

#### T4 Optimized Engine Experiment — COMPLETED

- Level 5 on T4: 361 Myelin failures, precision violations, aborted.
- Level 3 + FAST tiling on T4: p95=382.6ms (14.9ms slower than baseline 367.7ms).
- **Result**: TRT optimization flags provide zero benefit on T4 (sm_75).

#### L4 Baseline Experiment — COMPLETED (REALTIME PASSED)

- Separate Volume `avtr1-artifacts-l4`, mount `/data/avtr1-l4`, `gpu="L4"`.
- Default TRT builder settings (no optimization flags).
- L4 engine build: ~11 min (vs T4 ~18 min), zero errors.

| Metric | T4 Baseline | L4 Baseline (100 chunks) |
|--------|-------------|--------------------------|
| p95 | 367.7ms | **188.0ms** |
| p50 | 364.7ms | **184.4ms** |
| FPS | 13.7 | **27.1** |
| VRAM | 3501/15360 MiB (23%) | **3572/23034 MiB (16%)** |

- **FUNCTIONAL: PASSED** (100/100 chunks, all 5 frames)
- **REALTIME: PASSED** (p95=188.0ms <= 200ms, p99=188.9ms)
- No level 5 build needed — L4 default is sufficient.

#### L4 Snapshot Testing — COMPLETED

- Snapshot creation: 79.2s (one-time, includes container start + pipeline load)
- **Snapshot restore: 4.9s** (measured in Phase 3 on warm container; Phase 4A measured 10-15s after scale-down including Modal scheduling and RPC overhead)
- Inference latency identical to non-snapshot (p95 within 2ms, empirically measured)
- TRT engines function after GPU memory snapshot/restore (empirically observed, not formally guaranteed by Modal)
- **Note**: GPU memory snapshots are an Alpha Modal feature. Modal docs state "the underlying checkpoint/restore technology in the device drivers is still quite new." See Phase 4A Snapshot Status below.

#### L4 Stage-Level Profiling — COMPLETED

- 7 engines wrapped with `_TimedEngine` proxy, sync overhead only +1.1%
- **Bottleneck**: decoder=49.1% + warp=24.9% = 74% of chunk time
- decoder: 19.12ms × 5 calls (per-frame, b=1) — biggest optimization target
- warp: 48.49ms × 1 call (already b=5) — single heaviest invocation

#### L4 Decoder Batch Audit — COMPLETED, NOT VIABLE

- B=1×5 mean=99.93ms, B=5×1 mean=102.06ms. Batch penalty: +2.13ms (+2.1%).
- **Verdict**: B=1 is intentionally optimal. Decoder batching rejected.

#### L4 CUDA Graph Decoder Audit — COMPLETED, MARGINAL

- Single-call graph: succeeded, bit-exact. Multi-call graph: succeeded, bit-exact.
- Savings: ~1ms (1.0%). **Not worth production complexity**.

#### L4 Warp Precision Audit — COMPLETED

- Engine uses mixed precision (FP16 internals, GridSample plugin forces FP32).
- 34 Reformat layers prove FP16 is active. FP32 binding I/O is normal TRT behavior.
- **No further precision investigation needed**.

#### L4 Warp Rebuild A/B Experiment — COMPLETED (Variant B REJECTED)

- **Variant B** (static B=5 profile, default builder): p95=194.5ms, **+6.5ms SLOWER** than baseline.
- p99 regressed by +9.7ms. Static profile hurts — TRT benefits from dynamic profile flexibility.
- **Variant C** (static B=5 + level-5): **NOT RUN** — Variant B rejected, static B=5 direction abandoned.
- See `docs/modal-smoke-results.md` for full A/B table.

#### Remaining optimization work

- **L4 level-5 experiment** — code ready, deferred (profiling shows blunt instrument vs targeted bottleneck)
- Multi-avatar testing on L4
- Future model-level optimizations (KV cache, step reduction, distillation) are research-track work, not prototype blockers

### Phase 3 Decision

Use **Modal L4 baseline engines + GPU memory snapshotting** as the prototype
runtime baseline.

- T4 is functional but not realtime.
- L4 baseline is realtime: p95=188.0ms, p99=188.9ms, 27.1 FPS.
- GPU snapshot restore is empirically ~4.9s on warm containers (Phase 4A measured 10-15s after scale-down including scheduling overhead) and does not regress inference in testing. GPU snapshots remain an Alpha Modal feature.
- TRT tactic experiments and static-profile rebuilds did not improve latency.
- Further optimization is deferred until an end-to-end product bottleneck proves
  it is necessary.

**Commands**:
```
# L4 baseline (engines already built, Volume persists)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild-l4
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-l4 --chunks 100

# L4 stage profiling (runs non-profiled + profiled)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test profile-l4 --chunks 10

# Warp rebuild A/B (Variant B REJECTED, Variant C NOT RUN)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild-l4-warp-static-b5
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-l4-warp-static-b5 --chunks 100
# Variant C available but not recommended:
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild-l4-warp-static-b5-level5
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-l4-warp-static-b5-level5 --chunks 100

# L4 level-5 experiment (code ready, not yet run)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild-l4-level5
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-l4-level5 --chunks 100

# T4 optimized (engines already built, Volume persists)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test verify-trt-patch
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-opt --chunks 10

# Snapshot testing — T4 (requires deploy)
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method health
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method startup
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method benchmark --chunks 10

# Snapshot testing — L4 (requires deploy)
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method health --class AVTR1SnapshotL4
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method startup --class AVTR1SnapshotL4
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method benchmark --chunks 20 --class AVTR1SnapshotL4
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method profile --chunks 10 --class AVTR1SnapshotL4

# L4 snapshot via modal run (no snapshot created, just tests methods)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test health-l4
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test startup-l4
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-l4-snap --chunks 20
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test profile-l4-snap --chunks 10
```

## Phase 4: SaaS Lifecycle Proof — Modal Session Runtime + Local Web App

**Status: Phase 4A completed**

Build the smallest working product lifecycle before integrating the full voice
agent stack or Cloudflare bridge.

This phase answers:

> Can a user click Connect in a web app, wake the snapshotted Modal avatar
> runtime, receive an "avatar ready" message, see an avatar/first frame, and
> click Stop to cleanly end the session?

### Scope

- Create a clean Modal avatar session runtime using the chosen L4 snapshot path.
- Expose session lifecycle methods:
  - `start_session`
  - `render_test_frame` or equivalent first-frame proof
  - `end_session`
- Keep per-session avatar state server-side in the Modal container.
- Build a tiny local web app with:
  - Connect button
  - Connecting/loading state
  - Avatar ready state
  - Avatar canvas or placeholder frame view
  - Stop button
- Use a local development server or lightweight local bridge for this phase.
- Do **not** integrate the old toy-agent project yet.
- Do **not** build the full Cloudflare Worker bridge yet.
- Do **not** wire STT/LLM/TTS yet.

### Success criteria

- Connect triggers Modal snapshot restore/start.
- UI receives a clear `avatar_ready` or equivalent readiness response.
- UI transitions from loading to connected.
- A first avatar frame, debug frame, or intentionally documented placeholder is shown.
- Stop ends the Modal session and returns the UI to disconnected state.
- Session cleanup is observable in logs.

### Deliverables

- Clean Modal runtime file for the prototype path, separate from the benchmark
  and optimization harnesses.
- Local web app code for the connect/ready/stop flow.
- Short documentation describing how to run the local lifecycle test.

### Phase 4A Results

**All success criteria met.**

| Criterion | Result |
|-----------|--------|
| Connect triggers Modal snapshot restore | Yes — first call triggered snapshot creation, subsequent calls use restored snapshot¹ |
| UI receives `avatar_ready` | Yes — `start_session` returns `status: "avatar_ready"` (192ms warm container, 1.2-2.2s first render after restore) |
| Debug frame displayed in browser | Yes — YUV I420 → JPEG via OpenCV, 1280x720, ~84KB |
| Stop ends session | Yes — `end_session` clears state, returns `{ ok: true }` |
| No files outside `avatar-saas-prototype/` changed | Confirmed |

¹ GPU memory snapshots are an Alpha Modal feature. First deploy may trigger a cold start
(50-100s) instead of snapshot restore while fresh snapshots are created. Subsequent
calls after snapshot coverage is complete restore successfully. See Snapshot Status below.

**Files created:**
- `modal-avatar-runtime/modal_lifecycle_app.py` — Modal L4 snapshot runtime with `AvatarLifecycleL4` class
- `local-lifecycle-app/server.py` — stdlib HTTP server, zero deps beyond `modal`
- `local-lifecycle-app/static/index.html` — browser UI
- `local-lifecycle-app/static/app.js` — state machine (disconnected→connecting→ready→stopping)
- `local-lifecycle-app/static/styles.css` — dark theme styling
- `local-lifecycle-app/README.md` — run instructions

**Commands:**
```
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py
uv run --with modal python3 avatar-saas-prototype/local-lifecycle-app/server.py
open http://localhost:8765
```

**Known limitations:**
- Single session only (not multi-tenant).
- Debug frame is a single JPEG from silent render, not live video.
- No audio, STT, LLM, TTS, mic, WebRTC, or streaming.
- Browser Stop ends the logical session only; it does not stop the deployed Modal app. Container stays warm for 10s after last call (`scaledown_window=10`). To stop manually: `modal app stop avtr1-lifecycle`.

### Snapshot Status

GPU memory snapshots are an Alpha Modal feature. Modal docs state "the underlying
checkpoint/restore technology in the device drivers is still quite new." The
observations below are empirical and may change as Modal evolves the feature.

- **Functional lifecycle**: Verified — `start_session`, `render_debug_frame`,
  `end_session` all work correctly with snapshot restore.
- **Current deploy snapshot restore**: Verified in latest test run — 5/5 restores
  successful, 0 SIGSEGVs. Modal logs show `Restoring Function from memory snapshot`
  followed by `lifecycle-restore` hook (no `lifecycle-snap` cold start).
- **Fast cold start**: Partially verified. Best case (warm container) ~5s. After
  scale-down, observed 10-15s restore-to-response including Modal scheduling and
  RPC overhead. The earlier 4.9s benchmark was a best-case measurement, not a
  universal guarantee. First render after restore has higher warmupMs (1.2-2.2s)
  compared to warm-container warmupMs (~193ms), likely due to TRT engine
  re-warming.
- **Risk**: Modal docs confirm that snapshots are specific to the underlying worker
  type and that redeploys make previous snapshots obsolete. GPU functions need 2-3
  snapshots per worker type for full coverage. We observed restore failures (SIGSEGV,
  exit code 139) that are consistent with these documented behaviors — obsolete
  snapshots after redeploy and incomplete worker-type coverage — but the exact
  Modal-internal root cause is not proven. Fresh snapshots after redeploy resolved
  the failures in our testing, but this remains empirical evidence, not a formally
  documented guarantee.
- **Operational fallback**: If snapshot restore fails, Modal falls back to a full
  cold start automatically. The app still works, but startup may be 50-100s instead
  of 10-15s. This fallback is handled by Modal, not by application code.

## Phase 5: Cloudflare Bridge Adaptation — New Project Implementation

**Status: Phase 5C completed and live-tested**

Phase 5A deliverable: `docs/phase-5-cloudflare-bridge-adaptation.md`

Key decisions:
- Raw Durable Objects (not `@cloudflare/voice` / `@cloudflare/agents`)
- `@modal.fastapi_endpoint()` on existing `AvatarLifecycleL4` class for HTTP access
- Modal URL is configuration (`MODAL_LIFECYCLE_URL`), not hardcoded
- Reuse existing VAD/Smart Turn workers via service bindings (Phase 6)
- New identity: `avatarSessionId` (not kid/device)
- WebSocket hibernation with storage-backed state (not in-memory only)

Phase 5B validated: Browser → AvatarSessionDO → Modal lifecycle round-trip over
WebSocket confirmed on Cloudflare edge (session start, debug frame, session end).

Phase 5C validated: Minimal browser web client deployed to Cloudflare Pages.
Live at https://avatar-web-client.pages.dev — full round-trip from browser:
POST /sessions → WebSocket → session.start → session.ready → render_debug_frame →
avatar.frame (JPEG displayed) → session.end → session.ended. CORS headers added to
Worker. WebSocket upgrade path passes through to DO unmodified (CORS doesn't apply
to WebSocket).

Known issue: wrangler local dev fetch() to HTTPS URLs fails with "internal error"
(workerd BoringSSL bug #9356); edge deployment works correctly.

Build a new Cloudflare bridge/agent stack **inside this repository**. The
previous toy-agent project is a reference implementation only, not a dependency.

### Scope

- Audit the previous toy-agent implementation for reusable patterns:
  - WebSocket session handling
  - VAD and Smart Turn integration
  - hosted STT/LLM/TTS provider wiring
  - observability and error handling patterns
- Create a new Cloudflare app under this prototype workspace.
- Design the new bridge around browser avatar sessions, not ESP32/toy-agent
  constraints.
- Connect the new Cloudflare bridge to the Modal avatar runtime lifecycle.
- Preserve the previous toy-agent project as-is.

### Explicit constraints

- Old project = reference only.
- Current repo = new implementation.
- No ESP32 assumptions.
- No dependency on old toy-agent Durable Object state model unless intentionally
  copied after review.
- No modifications to the old project as part of this phase.

### Initial responsibilities

- Implement session creation and WebSocket handling.
- Call Modal `start_session` on browser Connect.
- Relay `avatar_ready` to the browser.
- Call Modal `end_session` on Stop/disconnect.
- Add basic observability and failure handling.

## Phase 6: Voice Pipeline and Avatar Streaming Integration

**Status: Not started**

Wire browser speech, the new Cloudflare bridge, hosted AI providers, and the
Modal avatar renderer into one conversational prototype.

- Browser microphone capture.
- Browser audio upload to the new Cloudflare bridge.
- VAD and Smart Turn gating adapted into the new project.
- Hosted STT, LLM, and TTS provider calls.
- TTS audio chunking into Modal renderer-compatible audio windows.
- Modal-rendered avatar frames streamed back to the browser.
- Assistant audio playback in the browser.
- Stop/reconnect behavior.

## Phase 7: Browser Demo Polish and Developer API Shape

**Status: Not started**

Improve the browser demo and begin shaping the external SaaS/API surface.

- Better session UI states and error display.
- Canvas/video rendering improvements.
- Basic transcript display.
- Basic configuration controls for avatar/background.
- Initial public API shape for developers.
- Example integration snippets.

## Phase 8: SaaS/API Hardening

**Status: In progress**

Productionize the platform for external developer use.

- **Internal Modal endpoint authentication (done — enforce mode active)** —
  HMAC-SHA256 Worker→Modal request signing for active 5Q session endpoints
  (`start-session-5q`, `session-status-5q`, `push-audio-batch-5q`,
  `end-session-adapter-5q`, `stream-fmp4-av-adapter-5q`,
  `tts-same-gpu-parallel-5q`). `push-audio-5q` is dead code (signed but
  never called). `list-cached-avatars-5q` was removed (had no Worker caller).
  Enforce mode is active (`AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true`); unsigned
  and tampered requests return 401. OmniVoice `/web` signing is implemented
  but out of scope while config-gated off. SFU paths out of scope while
  disabled. See `docs/MODAL_ENDPOINT_AUTH_CLOSEOUT.md` for details.
- API key authentication and tenant management
- Session concurrency control and quotas
- Usage metering and billing hooks
- Multi-tenant isolation
- Observability dashboards
- Documentation and example integrations
