# Avatar Platform Roadmap

## Current Baseline

- **Phase 5Q**: Low-latency AV pipeline (CPU snapshot restore, eager MSE, adaptive fMP4 relay).
- **API Foundation**: Beta-complete as of `67dee27` (status correction `1d25d0f`).
- **Deployed**: Modal 5Q (`avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s-5q`), Bridge Worker (`avatar-session-bridge`), Web Client (`avatar-web-client.pages.dev`).

## Roadmap Phases

| # | Phase | Status | Summary |
|---|-------|--------|---------|
| 1 | 5Q baseline checkpoint | Done | Working AV pipeline with end-to-end timing telemetry. |
| 2 | API foundation beta | Beta-Complete | `/v1` API shell, auth, ownership, rate limits, lifecycle hardening, docs, drift guard. See `docs/API_FOUNDATION_BETA_CHECKPOINT.md`. |
| 3 | Session UX hardening | Partial | Startup placeholder, loading, and error UX shipped; deeper retry/reconnect behavior remains. |
| 4 | Cloudflare Stream / media delivery canary | Paused | Paused for live interactive avatar sessions. Revisit only for recording/VOD, one-to-many broadcast, or if the renderer emits WHIP/SRT/RTMP/WebRTC natively. |
| 5 | Avatar/background platform APIs | In Progress | Phase 6a.1: catalog read/detail APIs + start-time validation (done). Phase 6a.2: D1 metadata provider (code-only, production activation deferred). CRUD and R2 upload deferred. |
| 6 | Provider layer (STT/LLM/TTS) | Planned | Swappable providers behind a unified interface; config-driven selection. |
| 7 | Multi-avatar cache/LRU hardening | Planned | Production-grade eviction, preloading, and per-session avatar lifecycle. |
| 8 | Observability/benchmarking | Planned | Automated latency benchmarks, PostHog dashboards, alerting. |
| 9 | Scale/reliability | Planned | Concurrency, fault tolerance, retries, graceful shutdown. |
| 10 | SaaS productization | Planned | Billing, dashboard, self-serve keys, production quota accounting. |
| 11 | Internal Modal endpoint authentication | Production-Complete (fMP4 path) | HMAC-SHA256 Worker→Modal request signing (shared secret + timestamp + request-id + body hash). All active Worker→Modal calls signed and verified; enforce mode active (`AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true`). Covers current fMP4 path only. See `docs/MODAL_ENDPOINT_AUTH_CLOSEOUT.md`. |

## Notes

- The API foundation phase is **closed for expansion** unless a security or
  correctness blocker is found. The drift guard
  (`scripts/validate-api-contract.mjs`) should catch spec/runtime/doc drift.
- **Internal Modal endpoint authentication (roadmap #11) is production-complete
  for the current fMP4 Worker→Modal path.** Enforce mode is active
  (`AVTR_ENFORCE_MODAL_REQUEST_SIGNING=true`); all active Worker→Modal
  requests are HMAC-signed and verified on the Modal side. Unsigned/tampered
  requests return 401. `list-cached-avatars-5q` was removed (no Worker caller,
  was the only unsigned public endpoint). OmniVoice `/web` is out of scope
  while config-gated off (`ENABLE_OMNIVOICE_TTS_ON_L4=false`). SFU paths are
  out of scope while disabled (`AVTR_REALTIME_SFU_ENABLED=false`). See
  `docs/MODAL_ENDPOINT_AUTH_CLOSEOUT.md` for full details, rollback procedure,
  and future-work items.
- **Option C SFU POC is closed out as GO** (1280x720 H264 NVENC @ 2 Mbps,
  3/3 canaries) but **SFU remains disabled in production** — production is
  still fMP4 over WebSocket → browser MSE. Enabling SFU is a separate, gated
  decision tracked as a future phase, not part of this closeout.
- **Modal backend/workspace outage** is a known external blocker that may
  prevent full `/start` happy-path validation until Modal infrastructure is
  restored.

## Paused

- **Cold-start / GPU snapshot / Dynamo research** — paused unless explicitly reopened. The 5Q CPU-snapshot restore path (~2.5 s post-restore) is the production baseline. Modal infrastructure scheduling overhead is accepted as platform-level latency.
