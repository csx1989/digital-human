# Modal Avatar Runtime — Smoke Tests

Phase 3 smoke tests for AVTR-1 renderer feasibility on Modal T4 GPU.

**Questions**: Can AVTR-1 load and render on Modal T4? Is it fast enough for
realtime?

## Prerequisites

1. **Modal CLI** installed and authenticated:
   ```bash
   pip install modal
   modal setup
   ```

2. **HuggingFace token** — `avaturn-live/avtr-1` is a gated repo. Create the
   Modal secret before running GPU tests:
   ```bash
   modal secret create hf-secret HF_TOKEN=hf_your_token_here
   ```

3. **Network**: Modal containers must reach `huggingface.co` for artifact
   downloads on first run.

## Commands

Run from the **repo root** (`avtr-1/`):

### Test 1: Import check (CPU only, fast)

```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test imports
```

Verifies Python version, torch, tensorrt, onnxruntime, opencv, kornia, roma,
and attempts a guarded import of `avtr1_renderer`.

### Test 2: Pipeline startup (T4 GPU, slow on first run)

```bash
# Default: fail if TRT engines are missing
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test startup

# Build missing TRT engines on Modal T4 (first run only, ~10-15 min extra)
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test startup --build-missing-engines
```

Downloads HF artifacts, optionally builds AVTR1 TRT engines, then calls
`Pipeline.from_artifacts(avatar_ids=["anya_03_studio"])` (or the first
available portrait).

### Test 3: Single chunk render (T4 GPU, depends on Test 2)

```bash
# Default: fail if TRT engines are missing
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test render

# Build missing TRT engines if needed
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test render --build-missing-engines
```

Full pipeline load + one silence chunk render. Reports frame count, timing,
dimensions, VRAM usage.

> **Note**: Test 3 proves *functional correctness* only — the pipeline can
> produce frames. The cold-start render (1339ms on first run) includes TRT
> engine warmup and CUDA kernel JIT. It does **not** prove realtime capability.
> Run Test 4 (benchmark) for warm consecutive performance.

### Test 4: Render benchmark (T4 GPU, warm consecutive chunks)

```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test benchmark --chunks 10
```

Loads pipeline, runs one warmup chunk, then measures N consecutive chunks
passing state between them. Reports per-chunk timing, statistics (min/p50/
mean/p95/max), and effective FPS.

Two separate verdicts:

- **Functional**: All chunks produced 5 frames (proves state passing works).
- **Realtime**: p95 chunk time <= 200ms (proves T4 can sustain 25fps).

Decision criteria:

| p95 chunk time | Conclusion |
|---|---|
| <= 200ms | T4 viable for realtime prototype. Proceed to Phase 4. |
| 200–500ms | T4 marginal. Test L4 (24 GiB, sm89) next. |
| > 500ms | T4 insufficient. Must test L4 or A10G before Phase 4. |

### Direct function invocation (alternative)

You can also target functions directly:
```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py::test_imports
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py::test_pipeline_startup --build-missing-engines
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py::test_single_chunk_render --build-missing-engines
```

## Phase 3c: Snapshot + Optimized Engines

`modal_snapshot_smoke.py` tests GPU memory snapshotting and TRT build
optimizations as an isolated A/B experiment against baseline engines.

### Prerequisites

Baseline engines must already exist in `avtr1-artifacts` Volume (from Phase 3a/3b).

### Test: Verify TRT optimization patch (cheap, ~30s)

```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test verify-trt-patch
```

Runs on T4 GPU. Applies the monkey-patch, creates a real TRT builder config,
and reads back the actual values. **Must pass before running `--test rebuild`.**

Pass/fail semantics:
- `builder_optimization_level=5` is **REQUIRED**. FAIL if not applied.
- `tiling_optimization_level=FAST` is **optional** (API may not expose it).
  WARN if unavailable.

### Test: Rebuild optimized engines (one-time, 20-40 min)

```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test rebuild
```

Builds all 7 TRT engines with `builder_optimization_level=5` and
`tiling_optimization_level=FAST` into separate Volume `avtr1-artifacts-opt`.
**Baseline engines in `avtr1-artifacts` are never modified.**

The build monkey-patches `trt.Builder.create_builder_config` and prints
verification of actual level/tiling values for each engine build.

### Test: Benchmark optimized engines

```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py --test benchmark-opt --chunks 10
```

Loads pipeline from optimized engines (opt Volume) and runs warm benchmark.
Compare p95 against baseline (367.7ms).

### Test: GPU memory snapshot

```bash
# Deploy (snapshots only work for deployed apps)
modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_snapshot_smoke.py

# Invoke deployed class (NOT modal run — modal run does NOT use snapshots)
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method health
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method startup
python avatar-saas-prototype/modal-avatar-runtime/invoke_snapshot.py --method benchmark --chunks 10
```

The snapshot class (`AVTR1Snapshot`) uses `@modal.enter(snap=True)` to:
1. Ensure HF artifacts + TRT engines
2. Load pipeline
3. Pick avatar/background
4. Run one warmup render chunk (captures CUDA/TRT lazy init)

After deploy, the container is snapshotted. Subsequent invokes typically restore
from snapshot instead of re-initializing. Modal may need 2-3 snapshots per GPU
type for full worker-type coverage — early calls after deploy may still cold-start
while coverage is being built.

**Important**: `modal run` does NOT create/restore snapshots. You MUST use
`modal deploy` first, then invoke via `invoke_snapshot.py` or
`modal.Cls.from_name()`. Look for "Snapshot created" and "Restoring Function
from memory snapshot" in Modal logs.

### Deployed invocation (alternative Python)

```python
import modal
cls = modal.Cls.from_name("avtr1-snapshot-smoke", "AVTR1Snapshot")
inst = cls()
print(inst.health.remote())
print(inst.benchmark_render.remote(chunks=10))
```

### Metrics separation

| Metric | What it measures | Expected |
|--------|-----------------|----------|
| Startup time | Container init + pipeline load + warmup | Baseline: ~24s. Snapshot: faster if GPU snapshot works |
| Warm p95 | Consecutive chunk render latency | Same with/without snapshot. Different with optimized engines |
| Snapshot restore | Overhead of restoring GPU state | Alpha feature — may fail with TRT |

### Snapshot compatibility

GPU memory snapshots are an **Alpha** Modal feature. TRT engine runtime state
(CUDA contexts, driver handles) may not survive checkpoint/restore. If the
snapshot fails, the class reports the error clearly and does not hide it.

Potential outcomes:
1. **GPU snapshot works**: Cold start drops from ~24s to a few seconds.
2. **GPU snapshot fails**: Report failure honestly. Document what happened.
3. **Complete failure**: Snapshot incompatible with TRT runtime state.

## Expected Timing

| Run | test_imports | test_pipeline_startup | test_single_chunk_render | test_render_benchmark |
|-----|-------------|----------------------|--------------------------|----------------------|
| First (no engines) | ~30s | ~5-10 min (HF download) + FAIL | FAIL | FAIL |
| First (with --build-missing-engines) | ~30s | ~20-30 min (download + engine build + load) | ~20-30 min (same) | N/A (use --build-missing-engines on startup first) |
| Subsequent (engines in Volume) | ~30s | ~30-60s (pipeline load only) | ~30-60s (load) + ~1-2s (cold render) | ~30-60s (load) + ~1-2s (warmup) + N x chunk time |

## Architecture

```
modal_smoke.py
├── Image: nvidia/cuda:12.8.1-devel-ubuntu24.04 + Python 3.12
│   ├── pip: torch (cu128 index), tensorrt (nvidia index), + all deps
│   ├── src/avtr1_renderer → /opt/avtr1/avtr1_renderer (copy)
│   └── scripts/           → /opt/avtr1/scripts (copy)
├── Volume: avtr1-artifacts → /data/avtr1-artifacts
│   └── main/                          (HF revision subdirectory)
│       ├── avatars_artifacts/         (from HF)
│       ├── renderer_runtime_artifacts/ (from HF + ditto)
│       ├── build_artifacts/           (from HF)
│       ├── speech2motion_runtime_artifacts_cc/ (built on T4, level=3)
│       │   ├── avtr1_encode_fp16.engine
│       │   ├── avtr1_decode_fp16.engine
│       │   ├── hubert_lbs_fp16.engine
│       │   └── avtr1_normalizer.safetensors
│       └── renderer_runtime_artifacts_cc/ (built on T4, level=3)
│           ├── decoder_b5_fp16.engine
│           ├── warp_network_b5_fp16.engine
│           ├── stitch_network_b5_fp16.engine
│           └── modnet_b5_fp16.engine
└── Functions
    ├── test_imports()            — CPU, package verification
    ├── test_pipeline_startup()   — T4, artifact download + engine check + pipeline load
    ├── test_single_chunk_render() — T4, full pipeline + one render
    └── test_render_benchmark()   — T4, warm consecutive chunk benchmark

modal_snapshot_smoke.py
├── Same Image as modal_smoke.py
├── Volume: avtr1-artifacts-opt → /data/avtr1-opt
│   └── main/                          (same structure, built with level=5 + FAST tiling)
├── Functions
│   ├── verify_trt_patch()       — T4, verify TRT optimization patch works
│   ├── rebuild_optimized_engines() — T4, build optimized engines to opt Volume
│   └── benchmark_optimized()       — T4, benchmark with optimized engines
└── Class: AVTR1Snapshot (gpu="T4", enable_memory_snapshot=True)
    ├── Volume: avtr1-artifacts (baseline, read-only)
    ├── @modal.enter(snap=True)  — load pipeline + warmup (captured in snapshot)
    ├── @modal.enter(snap=False) — post-restore GPU verification
    ├── health()                 — return pipeline status
    ├── benchmark_first_render() — cold start measurement
    └── benchmark_render()       — warm consecutive benchmark
```

## Troubleshooting

### "AVTR1 TRT engines not found"

Run with `--build-missing-engines`:
```bash
modal run avatar-saas-prototype/modal-avatar-runtime/modal_smoke.py --test startup --build-missing-engines
```
This builds the engines on the T4 and persists them in the Volume. Subsequent
runs will use the cached engines.

### "Failed to download artifact" / 403 from HuggingFace

The `avaturn-live/avtr-1` repo requires authentication. Ensure you have created
the Modal secret with a valid HF token:
```bash
modal secret create hf-secret HF_TOKEN=hf_your_token_here
```
You must have accepted the gated repo terms on HuggingFace first.

### "tensorrt" install fails

TensorRT requires CUDA toolkit. The `nvidia/cuda:12.8.1-devel-ubuntu24.04`
base image includes the full toolkit. If pip install fails:
1. Check CUDA version compatibility: TensorRT <=10.12 supports CUDA 12.x
2. The NVIDIA pip index (`https://pypi.nvidia.com`) must be reachable
3. Try pinning a specific version: `tensorrt==10.12.0`

### "torch" is CPU-only

The image installs torch from `https://download.pytorch.org/whl/cu128`. If
`torch.cuda.is_available()` returns False:
1. Verify the function has `gpu="T4"` (or any GPU)
2. Check that the CUDA index URL is correct in the image definition
3. `torch.version.cuda` should report `12.8`

### CUDA mismatch / "cannot dlopen some GPU libraries"

TensorRT engines are compute-capability specific. Engines built on sm80+ (A100,
A10G) will NOT work on T4 (sm75). The smoke test builds engines on T4 when
`--build-missing-engines` is passed, without the `--ampere-plus` flag, so
engines target the T4's sm75.

If you see engine load failures:
1. Delete the Volume contents and rebuild: `modal volume rm avtr1-artifacts /speech2motion_runtime_artifacts_cc --recursive`
2. Re-run with `--build-missing-engines`

### OOM on T4

T4 has 16 GiB VRAM. The smoke-test path checks/requires 7 TRT engine files
(4 renderer + 2 AVTR1 + 1 HuBERT). The current smoke test builds all engines
as TRT on the T4 — the pipeline may have ONNX fallback paths for some engines,
but the warp network (GridSample3D custom op) and HuBERT (dynamic output
shapes) require TRT. If OOM occurs:

1. Check `nvidia-smi` output from the smoke test (authoritative memory usage)
2. `torch.cuda.memory_allocated()` under-reports — it only tracks PyTorch
   tensor allocations, not TRT engine memory, CUDA context, or driver overhead
3. Consider reducing avatar count (load only 1 portrait)
4. If still tight, consider upgrading to L4 (24 GiB) or A10G (24 GiB)

### avtr1_renderer import fails on CPU

This is expected if `tensorrt` requires a GPU device context at import time.
`test_imports()` runs on CPU and reports this as a guarded failure. The
actual pipeline tests (`test_pipeline_startup`, `test_single_chunk_render`)
run on T4 where `tensorrt` is available.

## What This Is Not

- Not a production service
- Not an HTTP/WebSocket endpoint
- Not a session API
- Not the Cloudflare bridge
- Single-chunk render proves functional correctness only, not realtime performance
- VRAM from `torch.cuda.memory_allocated()` is incomplete; nvidia-smi is authoritative
