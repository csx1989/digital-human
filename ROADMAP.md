# Avatar SaaS Roadmap

## Goal

Build a production-grade, fully self-hosted avatar application and API around
AVTR-1, using:

- Cloudflare Workers + Durable Objects for the control plane
- Modal GPU deployments for the self-hosted avatar renderer
- Hosted providers for `LLM`, `STT`, and `TTS`

We are explicitly **not** using Avaturn's hosted external-engine path. We are
building and owning the full avatar-serving stack ourselves.

## Non-Goals

- Self-hosting `LLM`, `STT`, or `TTS` in the initial roadmap
- Depending on Avaturn hosted sessions or external-engine protocol
- Designing around ESP32-era audio assumptions

## Product Direction

We want two product surfaces:

1. A first-party web application
2. A SaaS/API offering for external developers

The architecture should support both from the beginning.

## Core Architecture

### Control Plane

Cloudflare will own:

- authentication
- tenant and API key management
- session lifecycle
- real-time orchestration
- memory and conversation context
- VAD and end-of-turn detection
- observability, metrics, and tracing
- provider routing for hosted `LLM` / `STT` / `TTS`

### Compute Plane

Modal GPU services will own:

- AVTR-1 renderer deployment
- model warmup and scaling
- renderer health and latency telemetry

### AI Provider Plane

Hosted providers will own:

- `STT`
- `TTS`
- `LLM`

These are swappable integrations behind our own internal interfaces.

## Media Strategy

External clients should not be forced to speak raw PCM unless they want to.

### Public API

Accept web-friendly formats and transports such as:

- WebRTC audio
- WebSocket streaming audio
- browser-native encoded audio
- file uploads where applicable

### Internal Canonical Audio Format

Normalize audio once into:

- `16 kHz`
- `mono`
- `float32` in memory

Use `pcm16` only at exact model or provider boundaries that require it.

This keeps the product API flexible without polluting the full stack with
low-level transport constraints.

## Milestones

## Phase 0: Feasibility and Benchmarking

### Objective

Prove that AVTR-1 can be run as a useful prototype on Modal `T4`, while keeping
an upgrade path open to `L4` or `A10G` if the benchmark fails.

### Tasks

- stand up a reproducible benchmark harness for AVTR-1
- test cold start, warm latency, and sustained real-time throughput on `T4`
- measure time-to-first-frame and chunk latency
- estimate cost per live session minute
- define upgrade criteria for `L4` / `A10G`

### Exit Criteria

- we know whether `T4` is acceptable for prototype use
- we know whether GPU class must be upgraded before customer-facing rollout

## Phase 1: Self-Hosted Renderer Service

### Objective

Deploy AVTR-1 as a self-hosted GPU renderer service on Modal.

### Tasks

- package the renderer around the existing HTTP API boundary
- deploy a dedicated renderer service on Modal
- support avatar preload and stateful chunk processing
- add health checks and latency instrumentation
- define session state handling between chunk calls
- document renderer request and response contracts

### Deliverable

A private renderer service we fully own and can call from Cloudflare.

## Phase 2: Cloudflare Session Orchestrator

### Objective

Build the real-time session layer that coordinates browser/API traffic,
conversation state, hosted providers, and the renderer.

### Tasks

- create a dedicated avatar session Worker
- implement per-session Durable Objects
- handle auth, session creation, and teardown
- manage upstream audio ingestion and downstream event streaming
- connect hosted `STT`, `LLM`, and `TTS` providers behind internal adapters
- define retry, timeout, and provider-fallback behavior

### Deliverable

A Cloudflare-native orchestration layer that owns the live avatar session.

## Phase 3: Turn Detection Stack

### Objective

Use Cloudflare-native turn-taking so the system decides when a user has actually
finished speaking before invoking the assistant.

### Tasks

- keep the lightweight Cloudflare VAD path
- add `SmartTurnDO` using the same isolated ONNX worker pattern as the current
  VAD worker
- buffer active user-turn audio in the session DO
- invoke Smart Turn on silence candidates using the last up to 8 seconds
- finalize turn only when Smart Turn predicts completion
- measure latency and false-cutoff behavior

### Deliverable

Reliable server-side end-of-turn detection under our control.

## Phase 4: Hosted Provider Integration Layer

### Objective

Integrate hosted `STT`, `TTS`, and `LLM` providers cleanly, without coupling the
product architecture to any one vendor.

### Tasks

- define provider interfaces for transcription, synthesis, and response
  generation
- add provider adapters
- standardize audio normalization before provider calls
- add latency, cost, and failure telemetry per provider
- support controlled provider swapping by environment or tenant

### Deliverable

Stable AI-service boundaries that can evolve without changing the rest of the
system.

## Phase 5: Browser Application

### Objective

Ship a first-party web application for interactive avatar conversations.

### Tasks

- build browser mic capture and session bootstrap
- stream audio to the Cloudflare session layer
- render returned avatar media in the browser
- expose connection, latency, and interruption states
- validate cross-browser behavior

### Deliverable

A first-party product experience that validates the full stack end to end.

## Phase 6: Developer API

### Objective

Expose the same platform as a SaaS/API product.

### Tasks

- define session API for external developers
- support API keys and tenant-level configuration
- design web-friendly media contracts
- add documentation and example integrations
- expose events, transcripts, and usage reporting

### Deliverable

A developer-facing avatar platform API.

## Phase 7: Multi-Tenant Production Hardening

### Objective

Prepare the platform for production reliability and customer isolation.

### Tasks

- tenant isolation and quotas
- session concurrency control
- billing and usage metering
- observability dashboards
- failure handling and replay/debug tooling
- rollout and migration strategy for provider changes

### Deliverable

A production-ready SaaS control plane.

## Workstreams

### Workstream A: Renderer

- GPU benchmarking
- Modal deployment
- warm pools
- renderer API stability

### Workstream B: Realtime Orchestration

- Cloudflare session Worker
- Durable Object state
- browser/API transport

### Workstream C: Turn-Taking

- VAD integration
- Smart Turn deployment
- interruption behavior

### Workstream D: Provider Layer

- hosted `STT`
- hosted `TTS`
- hosted `LLM`
- provider abstraction and failover

### Workstream E: Productization

- web app
- API design
- auth
- quotas
- billing

## Immediate Next Steps

1. Benchmark AVTR-1 on Modal `T4`
2. Write the renderer service spec
3. Write the Cloudflare session-DO spec
4. Write the Smart Turn integration spec
5. Choose initial hosted providers for `LLM`, `STT`, and `TTS`

## Decision Log

- We will self-host the avatar stack from day one.
- We will not use Avaturn's hosted external-engine path.
- We will use hosted providers for `LLM`, `STT`, and `TTS`.
- We will prototype the renderer on `T4`, but benchmark-gate that decision.
- We will avoid making raw PCM the primary public API contract.
