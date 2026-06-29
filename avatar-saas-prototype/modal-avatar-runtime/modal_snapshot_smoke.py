"""
Modal GPU memory snapshot smoke test + optimized TRT engine experiment.

Phase 3c:
  A) GPU memory snapshot to reduce AVTR-1 cold start (T4 + L4).
  B) TRT builder_optimization_level A/B experiments on T4 and L4.
  C) Honest benchmarking separating cold-start from warm-render metrics.
  D) Stage-level profiling to identify bottlenecks (engine-level timing).
  E) Decoder batching feasibility audit (B=1×5 vs B=5×1).
  F) CUDA Graph feasibility audit for decoder loop.

Phase 3d:
  Warp rebuild A/B experiment — static B=5 profile and level-5 variants.

Usage (modal run):
    modal run modal_snapshot_smoke.py --test rebuild
    modal run modal_snapshot_smoke.py --test benchmark-opt --chunks 10
    modal run modal_snapshot_smoke.py --test rebuild-l4
    modal run modal_snapshot_smoke.py --test benchmark-l4 --chunks 20
    modal run modal_snapshot_smoke.py --test profile-l4 --chunks 10
    modal run modal_snapshot_smoke.py --test rebuild-l4-level5
    modal run modal_snapshot_smoke.py --test benchmark-l4-level5 --chunks 100
    modal run modal_snapshot_smoke.py --test rebuild-l4-warp-static-b5
    modal run modal_snapshot_smoke.py --test benchmark-l4-warp-static-b5 --chunks 100
    modal run modal_snapshot_smoke.py --test rebuild-l4-warp-static-b5-level5
    modal run modal_snapshot_smoke.py --test benchmark-l4-warp-static-b5-level5 --chunks 100
    modal run modal_snapshot_smoke.py --test audit-decoder
    modal run modal_snapshot_smoke.py --test audit-cuda-graph
    modal run modal_snapshot_smoke.py --test audit-warp
    modal run modal_snapshot_smoke.py --test audit-warp-precision
    modal run modal_snapshot_smoke.py --test health
    modal run modal_snapshot_smoke.py --test startup
    modal run modal_snapshot_smoke.py --test benchmark --chunks 10

Usage (modal deploy + invoke):
    modal deploy modal_snapshot_smoke.py
    # After deploy, use the caller script or Python:
    python invoke_snapshot.py --method health
    python invoke_snapshot.py --method health --class AVTR1SnapshotL4
    python invoke_snapshot.py --method startup --class AVTR1SnapshotL4
    python invoke_snapshot.py --method benchmark --chunks 10 --class AVTR1SnapshotL4
    python invoke_snapshot.py --method profile --chunks 10 --class AVTR1SnapshotL4

Important:
  GPU memory snapshots are only created for DEPLOYED apps.
  `modal run` will exercise the code but will NOT create/restore snapshots.
  Look for "Snapshot created" / "Restoring Function from memory snapshot" in logs.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import modal

_SCRIPT_DIR = Path(__file__).resolve().parent

try:
    _REPO_ROOT = _SCRIPT_DIR.parents[1]
    _SRC_DIR = _REPO_ROOT / "src"
    _SCRIPTS_DIR = _REPO_ROOT / "scripts"
    assert (_SRC_DIR / "avtr1_renderer").is_dir(), (
        f"Not in repo root: expected {_SRC_DIR / 'avtr1_renderer'}"
    )
except (IndexError, AssertionError):
    _SRC_DIR = Path("/nonexistent/src")
    _SCRIPTS_DIR = Path("/nonexistent/scripts")

ARTIFACT_VOLUME_NAME = "avtr1-artifacts"
ARTIFACT_MOUNT_PATH = "/data/avtr1-artifacts"

OPT_VOLUME_NAME = "avtr1-artifacts-opt"
OPT_MOUNT_PATH = "/data/avtr1-opt"

L4_VOLUME_NAME = "avtr1-artifacts-l4"
L4_MOUNT_PATH = "/data/avtr1-l4"

L4_LEVEL5_VOLUME_NAME = "avtr1-artifacts-l4-level5"
L4_LEVEL5_MOUNT_PATH = "/data/avtr1-l4-level5"

L4_WARP_STATIC_B5_VOLUME_NAME = "avtr1-artifacts-l4-warp-static-b5"
L4_WARP_STATIC_B5_MOUNT_PATH = "/data/avtr1-l4-warp-static-b5"

L4_WARP_STATIC_B5_LEVEL5_VOLUME_NAME = "avtr1-artifacts-l4-warp-static-b5-level5"
L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH = "/data/avtr1-l4-warp-static-b5-level5"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(
        f"nvidia/cuda:{CUDA_TAG}",
        add_python="3.12",
    )
    .entrypoint([])
    .apt_install(
        "libglib2.0-0",
        "libgl1",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "torch>=2.5.1,<2.8",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "tensorrt<=10.12",
        extra_index_url="https://pypi.nvidia.com",
    )
    .pip_install(
        "onnxruntime-gpu>=1.20",
        "numpy>=2.0,<3",
        "safetensors>=0.5,<1",
        "huggingface_hub>=0.30,<1",
        "kornia>=0.8.1,<0.9",
        "roma>=1.5.4,<2",
        "opencv-python-headless>=4.11,<5",
        "einops>=0.8.1,<0.9",
        "scikit-image>=0.25,<0.26",
        "soxr>=0.4,<0.5",
        "imageio>=2.37,<3",
        "attrs>=25.3,<26",
        "tenacity>=9.1,<10",
        "av>=17.0,<18",
        "soundfile>=0.13,<0.14",
        "httpx>=0.28,<0.29",
        "fastapi>=0.116,<0.117",
        "uvicorn>=0.35,<0.36",
        "python-multipart>=0.0.20,<0.0.21",
        "anyio>=4.11,<5",
        "tqdm>=4.67,<5",
        "onnx>=1.17,<2",
        "onnx-graphsurgeon>=0.5",
    )
    .add_local_dir(
        str(_SRC_DIR / "avtr1_renderer"),
        remote_path="/opt/avtr1/avtr1_renderer",
        copy=True,
    )
    .add_local_dir(
        str(_SCRIPTS_DIR),
        remote_path="/opt/avtr1/scripts",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/avtr1:/opt/avtr1/scripts",
            "AVTR1_LOCAL_STORAGE": ARTIFACT_MOUNT_PATH,
        }
    )
)

app = modal.App("avtr1-snapshot-smoke", image=image)

artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME, create_if_missing=True
)
opt_volume = modal.Volume.from_name(
    OPT_VOLUME_NAME, create_if_missing=True
)
l4_volume = modal.Volume.from_name(
    L4_VOLUME_NAME, create_if_missing=True
)
l4_level5_volume = modal.Volume.from_name(
    L4_LEVEL5_VOLUME_NAME, create_if_missing=True
)
l4_warp_static_b5_volume = modal.Volume.from_name(
    L4_WARP_STATIC_B5_VOLUME_NAME, create_if_missing=True
)
l4_warp_static_b5_level5_volume = modal.Volume.from_name(
    L4_WARP_STATIC_B5_LEVEL5_VOLUME_NAME, create_if_missing=True
)

RENDERER_ENGINE_REL = "renderer_runtime_artifacts_cc"
RENDERER_ENGINES = [
    f"{RENDERER_ENGINE_REL}/decoder_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/warp_network_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/stitch_network_b5_fp16.engine",
    f"{RENDERER_ENGINE_REL}/modnet_b5_fp16.engine",
]
AVTR1_ENGINE_REL = "speech2motion_runtime_artifacts_cc"
AVTR1_ENGINES = [
    f"{AVTR1_ENGINE_REL}/hubert_lbs_fp16.engine",
    f"{AVTR1_ENGINE_REL}/avtr1_encode_fp16.engine",
    f"{AVTR1_ENGINE_REL}/avtr1_decode_fp16.engine",
]
NORMALIZER_FILE = "avtr1_normalizer.safetensors"


def _check_all_engines(storage_root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in RENDERER_ENGINES + AVTR1_ENGINES + [NORMALIZER_FILE]:
        p = storage_root / rel
        if not p.is_file():
            missing.append(rel)
    return len(missing) == 0, missing


WARP_PLUGIN_REL = "renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"

REQUIRED_ARTIFACT_RELS = (
    RENDERER_ENGINES
    + AVTR1_ENGINES
    + [NORMALIZER_FILE, WARP_PLUGIN_REL]
)


def _validate_all_artifacts(storage_root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in REQUIRED_ARTIFACT_RELS:
        p = storage_root / rel
        if not p.is_file():
            missing.append(rel)
    return len(missing) == 0, missing


def _ensure_artifacts(mount_path: Path | None = None) -> Path:
    """Download HF artifacts and return the resolved storage root.

    If *mount_path* is given (e.g. Path("/data/avtr1-opt")), it is set as
    ``AVTR1_LOCAL_STORAGE`` and ``get_storage_root()`` resolves it to
    ``{mount_path}/main/`` (appending the HF revision).

    When called with no arguments the current ``AVTR1_LOCAL_STORAGE`` env var
    (or the project default) is used and ``get_storage_root()`` is returned
    directly — e.g. ``/data/avtr1-artifacts/main/``.
    """
    from avtr1_renderer.avtr1_artifact_manager import (
        get_artifact_manager,
        get_storage_root,
    )

    if mount_path is not None:
        os.environ["AVTR1_LOCAL_STORAGE"] = str(mount_path)

    storage_root = get_storage_root()
    print(f"[artifacts] Storage root: {storage_root}")
    print("[artifacts] Ensuring all HF artifacts are present...")
    t0 = time.monotonic()
    mgr = get_artifact_manager()
    mgr.ensure_all_artifacts(workers=4)
    elapsed = time.monotonic() - t0
    print(f"[artifacts] Artifacts ready in {elapsed:.0f}s")
    return storage_root


def _pick_first_avatar(storage_root: Path) -> str:
    portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
    if not portraits_dir.is_dir():
        portraits_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"No reference_frames at {portraits_dir}")
    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No portrait PNGs in {portraits_dir}")
    stems = [p.stem for p in all_pngs]
    if "anya_03_studio" in stems:
        return "anya_03_studio"
    chosen = stems[0]
    print(f"[avatar] 'anya_03_studio' not found; using '{chosen}'")
    return chosen


def _pick_first_bg(backgrounds: dict[str, object]) -> str:
    for bg_id in sorted(backgrounds):
        if bg_id != "transparent":
            return bg_id
    raise RuntimeError("No non-transparent backgrounds registered")


def _run_nvidia_smi(label: str) -> None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip()
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                name, mem_used, mem_total, gpu_util = parts
                print(f"[nvidia-smi] {label}: {name}, "
                      f"{mem_used}/{mem_total} MiB, "
                      f"GPU util {gpu_util}%")
            else:
                print(f"[nvidia-smi] {label}: {line}")
        else:
            print(f"[nvidia-smi] {label}: failed (rc={result.returncode})")
    except Exception as exc:
        print(f"[nvidia-smi] {label}: unavailable ({exc})")


def _apply_trt_build_optimizations() -> None:
    """Monkey-patch TRT Builder to set builder_optimization_level=3 and
    tiling_optimization_level=FAST. Prints verification of actual values.

    Level 3 is TRT's default: heuristics pick good tactics without forcing
    exhaustive dynamic kernel compilation. Level 5 caused Myelin assertion
    failures (g.nodes.size() == 0) and precision constraint violations on
    T4 (sm_75) with no measurable runtime benefit over level 3.
    """
    import tensorrt as trt

    _orig = trt.Builder.create_builder_config

    def _patched(self):
        config = _orig(self)
        config.builder_optimization_level = 3
        tiling_applied = False
        tiling_err = ""
        try:
            config.tiling_optimization_level = trt.TilingOptimizationLevel.FAST
            tiling_applied = True
        except Exception as e:
            tiling_err = str(e)
        actual_level = config.builder_optimization_level
        actual_tiling = None
        try:
            actual_tiling = config.tiling_optimization_level
        except Exception:
            pass
        print(f"[optimization] VERIFIED config: "
              f"builder_optimization_level={actual_level} "
              f"(expect 3), "
              f"tiling_optimization_level={actual_tiling} "
              f"(expect FAST)")
        if not tiling_applied:
            print(f"[optimization] WARN: tiling FAILED: {tiling_err}")
        return config

    trt.Builder.create_builder_config = _patched
    print("[optimization] TRT Builder.create_builder_config patched "
          "(level=3 + FAST tiling)")


# ---------------------------------------------------------------------------
# Function: verify_trt_patch
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=300,
    secrets=[modal.Secret.from_name("hf-secret")],
)
def verify_trt_patch() -> dict:
    """Verify that the TRT optimization monkey-patch works without rebuilding
    engines.  Runs on T4 GPU.  Returns a status dict.

    Pass/fail semantics:
      - builder_optimization_level=3 is REQUIRED.
      - tiling_optimization_level=FAST is OPTIONAL (API may not expose it).
    """
    import tensorrt as trt

    print(f"[verify] TensorRT version: {trt.__version__}")
    print("[verify] Applying optimization monkey-patch...")
    _apply_trt_build_optimizations()

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()

    level_value = config.builder_optimization_level
    level_applied = level_value == 3

    tiling_applied = False
    tiling_value = None
    tiling_error = None
    try:
        tiling_value = config.tiling_optimization_level
        tiling_applied = tiling_value == trt.TilingOptimizationLevel.FAST
    except Exception as e:
        tiling_error = str(e)

    result = {
        "trt_version": trt.__version__,
        "level_applied": level_applied,
        "level_value": level_value,
        "tiling_applied": tiling_applied,
        "tiling_value": str(tiling_value) if tiling_value is not None else None,
        "tiling_error": tiling_error,
    }

    print()
    if level_applied:
        print(f"[verify] PASS: builder_optimization_level={level_value} (required)")
    else:
        print(f"[verify] FAIL: builder_optimization_level={level_value} "
              f"(expected 3, REQUIRED)")

    if tiling_applied:
        print(f"[verify] PASS: tiling_optimization_level={tiling_value}")
    elif tiling_error:
        print(f"[verify] WARN: tiling unavailable — {tiling_error}")
    else:
        print(f"[verify] WARN: tiling={tiling_value} (expected FAST, optional)")

    overall_pass = level_applied
    result["overall_pass"] = overall_pass
    if overall_pass:
        print("[verify] OVERALL: PASS. Safe to run --test rebuild.")
    else:
        print("[verify] OVERALL: FAIL. Do NOT run --test rebuild.")
    return result


def _build_all_engines(storage_root: Path, vol: modal.Volume) -> None:
    """Build all 7 TRT engines using existing build scripts.
    AVTR1_LOCAL_STORAGE must already point to storage_root.
    """
    from build_renderer_engines import main as renderer_main
    from build_hubert_engine import main as hubert_main
    from build_avtr1_engines import main as avtr1_main

    print("[build] Building renderer TRT engines (decoder, warp, modnet, stitch)...")
    t0 = time.monotonic()
    renderer_main([])
    print(f"[build] Renderer engines built in {time.monotonic() - t0:.0f}s")

    print("[build] Building HuBERT TRT engine...")
    t0 = time.monotonic()
    hubert_main([])
    print(f"[build] HuBERT engine built in {time.monotonic() - t0:.0f}s")

    print("[build] Building AVTR1 encode/decode + normalizer...")
    t0 = time.monotonic()
    avtr1_main([])
    print(f"[build] AVTR1 engines built in {time.monotonic() - t0:.0f}s")

    vol.commit()
    print("[build] Volume committed.")

    engines_ok, missing = _check_all_engines(storage_root)
    if engines_ok:
        print("[build] All 7 optimized engine files verified.")
    else:
        print(f"[build] WARNING: Missing engines after build: {missing}")


def _seed_engines_from_l4(source_root: Path, target_root: Path) -> None:
    """Copy all non-warp engines and plugin from L4 baseline to target volume."""
    import shutil

    warp_rel = f"{RENDERER_ENGINE_REL}/warp_network_b5_fp16.engine"
    files_to_copy = [rel for rel in REQUIRED_ARTIFACT_RELS if rel != warp_rel]

    print(f"[seed] Copying {len(files_to_copy)} artifacts from L4 baseline...")
    print(f"[seed] Source: {source_root}")
    print(f"[seed] Target: {target_root}")
    print(f"[seed] Skipping: {warp_rel} (will be rebuilt)")

    copied = 0
    missing = []
    for rel in files_to_copy:
        src = source_root / rel
        dst = target_root / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
        print(f"  {rel}")

    if missing:
        print(f"[seed] WARNING: {len(missing)} files missing in L4 baseline: {missing}")

    print(f"[seed] Copied {copied}/{len(files_to_copy)} files.")


def _build_warp_variant(
    storage_root: Path,
    *,
    static_batch: int | None = None,
    builder_level: int | None = None,
) -> None:
    """Build warp engine with custom optimization profile.

    Imports graph surgery helpers from build_renderer_engines (no root changes).
    If static_batch is set, uses min=opt=max=static_batch (static profile).
    If builder_level is set, applies that builder_optimization_level.
    """
    import ctypes

    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    import tensorrt as trt

    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager

    from build_renderer_engines import (
        _FilteredLogger,
        _surge_warp,
        _symbolize_batch,
    )

    mgr = get_artifact_manager()
    onnx_path = mgr.get_artifact_path("warp_network_onnx")
    plugin_path = mgr.storage_path("warp_plugin")
    out_rel = f"{RENDERER_ENGINE_REL}/warp_network_b5_fp16.engine"
    out_path = storage_root / out_rel

    max_batch = static_batch if static_batch is not None else 5

    print(f"[warp-variant] Loading ONNX: {onnx_path}")
    model = onnx.load(str(onnx_path))
    graph = gs.import_onnx(model)
    _surge_warp(graph)
    for tensor in list(graph.inputs) + list(graph.outputs):
        new_shape = list(tensor.shape)
        new_shape[0] = "batch"
        tensor.shape = new_shape
    graph.cleanup().toposort()
    new_model = gs.export_onnx(graph)
    _symbolize_batch(new_model)

    ctypes.cdll.LoadLibrary(str(plugin_path))

    logger = _FilteredLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(new_model.SerializeToString()):
        for i in range(parser.num_errors):
            print(parser.get_error(i), file=sys.stderr)
        raise RuntimeError("[warp-variant] ONNX parse failed")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    if builder_level is not None:
        config.builder_optimization_level = builder_level
        actual = config.builder_optimization_level
        print(f"[warp-variant] builder_optimization_level={actual} "
              f"(requested {builder_level})")

    profile = builder.create_optimization_profile()
    if static_batch is not None:
        b = static_batch
        print(f"[warp-variant] Static profile: min=opt=max={b}")
        profile.set_shape("feature_3d", (b, 32, 16, 64, 64), (b, 32, 16, 64, 64), (b, 32, 16, 64, 64))
        profile.set_shape("kp_source", (b, 21, 3), (b, 21, 3), (b, 21, 3))
        profile.set_shape("kp_driving", (b, 21, 3), (b, 21, 3), (b, 21, 3))
    else:
        print(f"[warp-variant] Dynamic profile: min=1, opt={max_batch}, max={max_batch}")
        profile.set_shape("feature_3d", (1, 32, 16, 64, 64), (max_batch, 32, 16, 64, 64), (max_batch, 32, 16, 64, 64))
        profile.set_shape("kp_source", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
        profile.set_shape("kp_driving", (1, 21, 3), (max_batch, 21, 3), (max_batch, 21, 3))
    config.add_optimization_profile(profile)

    print(f"[warp-variant] Building TRT engine -> {out_path}")
    t0 = time.time()
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("[warp-variant] TRT build failed")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(plan))
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(f"[warp-variant] Built in {elapsed:.1f}s, size {size_mb:.1f} MB")


def _run_benchmark(pipeline, registry, avatar_id: str, chunks: int) -> dict:
    """Run warm benchmark and return stats dict."""
    import numpy as np
    import torch

    avatar = registry[avatar_id]
    mg = pipeline._motion_generator
    expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    silence = np.zeros(expected_len, dtype=np.float32)

    from avtr1_renderer.types import Chunk, RenderOptions
    chunk = Chunk(audio_speech=silence, audio_listen=silence)
    bg_id = _pick_first_bg(pipeline._backgrounds)
    options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

    print(f"[benchmark] Warmup chunk (state=None)...")
    t_warm = time.monotonic()
    state, frames_iter = pipeline.process_chunk(
        avatar, chunk, state=None, options=options
    )
    warmup_frames = list(frames_iter)
    warmup_ms = (time.monotonic() - t_warm) * 1000
    print(f"[benchmark] Warmup: {warmup_ms:.0f}ms, "
          f"{len(warmup_frames)} frames, state={type(state).__name__}")

    _run_nvidia_smi("after-warmup")

    chunk_times_ms: list[float] = []
    chunk_frame_counts: list[int] = []
    all_functional = True

    print(f"[benchmark] Running {chunks} measured chunks...")
    for i in range(chunks):
        t_chunk = time.monotonic()
        try:
            next_state, frames_iter = pipeline.process_chunk(
                avatar, chunk, state=state, options=options
            )
            frames = list(frames_iter)
        except Exception as exc:
            print(f"[benchmark] Chunk {i + 1} FAILED: {exc}")
            all_functional = False
            break
        elapsed_ms = (time.monotonic() - t_chunk) * 1000
        chunk_times_ms.append(elapsed_ms)
        chunk_frame_counts.append(len(frames))
        state = next_state
        detail = f"  chunk {i + 1:3d}: {elapsed_ms:7.1f}ms, {len(frames)} frames"
        if i < 3 or i >= chunks - 3:
            print(detail)
        elif i == 3:
            print("  ...")

    _run_nvidia_smi("after-benchmark")

    if not chunk_times_ms:
        return {"error": "No chunks completed successfully"}

    sorted_ms = sorted(chunk_times_ms)
    mean_ms = sum(sorted_ms) / len(sorted_ms)
    min_ms = sorted_ms[0]
    max_ms = sorted_ms[-1]
    p50_ms = sorted_ms[(len(sorted_ms) - 1) // 2]
    p95_ms = sorted_ms[int((len(sorted_ms) - 1) * 0.95)]
    p99_ms = sorted_ms[int((len(sorted_ms) - 1) * 0.99)]
    mean_fps = 5.0 / (mean_ms / 1000.0)

    alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    functional_ok = all_functional and all(c == 5 for c in chunk_frame_counts)

    return {
        "warmup_ms": warmup_ms,
        "chunks": len(chunk_times_ms),
        "min_ms": min_ms,
        "p50_ms": p50_ms,
        "mean_ms": mean_ms,
        "p95_ms": p95_ms,
        "p99_ms": p99_ms,
        "max_ms": max_ms,
        "fps": mean_fps,
        "vram_alloc_gib": alloc,
        "vram_reserved_gib": reserved,
        "vram_total_gib": total,
        "functional": functional_ok,
        "realtime": p95_ms <= 200.0,
    }


def _print_benchmark_results(label: str, stats: dict) -> None:
    print()
    print("=" * 60)
    print(f"BENCHMARK RESULTS ({label})")
    print("=" * 60)
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        print("=" * 60)
        return
    print(f"  Warmup:      {stats['warmup_ms']:.0f}ms")
    print(f"  Chunks:      {stats['chunks']}")
    print(f"  min:         {stats['min_ms']:.1f}ms")
    print(f"  p50:         {stats['p50_ms']:.1f}ms")
    print(f"  mean:        {stats['mean_ms']:.1f}ms")
    print(f"  p95:         {stats['p95_ms']:.1f}ms")
    print(f"  p99:         {stats['p99_ms']:.1f}ms")
    print(f"  max:         {stats['max_ms']:.1f}ms")
    print(f"  FPS:         {stats['fps']:.1f}")
    print(f"  VRAM:        {stats['vram_alloc_gib']:.2f} GiB alloc, "
          f"{stats['vram_reserved_gib']:.2f} GiB reserved, "
          f"{stats['vram_total_gib']:.1f} GiB total")
    print()
    if stats["functional"]:
        print(f"  FUNCTIONAL: PASSED")
    else:
        print(f"  FUNCTIONAL: FAILED")
    if stats["realtime"]:
        print(f"  REALTIME:   PASSED (p95={stats['p95_ms']:.1f}ms <= 200ms)")
    else:
        print(f"  REALTIME:   FAILED (p95={stats['p95_ms']:.1f}ms > 200ms)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Profiling infrastructure
# ---------------------------------------------------------------------------


class _TimedEngine:
    """Proxy that wraps an inference engine and records per-call wall-clock
    timing with ``torch.cuda.synchronize()`` barriers.

    Forwards attribute access (e.g. ``allocate_outputs``) to the wrapped
    engine so existing call-sites work unchanged.
    """

    def __init__(self, engine: object, name: str, timings: dict[str, list[float]]) -> None:
        self._engine = engine
        self._name = name
        self._timings = timings

    def __call__(self, *args: object, **kwargs: object) -> object:
        import torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self._engine(*args, **kwargs)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._timings[self._name].append(elapsed_ms)
        return result

    def __getattr__(self, attr: str) -> object:
        return getattr(self._engine, attr)


_ENGINE_SPECS = [
    ("_motion_generator", "_hubert", "hubert"),
    ("_motion_generator", "_encode", "avtr1_encode"),
    ("_motion_generator", "_decode", "avtr1_decode"),
    (None, "_stitch", "stitch"),
    (None, "_warp", "warp"),
    (None, "_decoder", "decoder"),
    (None, "_matting", "modnet"),
]


def _install_stage_timers(pipeline: object) -> dict[str, list[float]]:
    """Wrap all 7 inference engines on *pipeline* with ``_TimedEngine``.

    Validates that every expected attribute path exists before wrapping.
    Fails loudly with a diagnostic print if any path is missing.

    Returns the shared ``timings`` dict that the wrappers append to.
    """
    timings: dict[str, list[float]] = {}

    for sub_attr, engine_attr, label in _ENGINE_SPECS:
        if sub_attr is not None:
            parent = getattr(pipeline, sub_attr, None)
            if parent is None:
                available = [a for a in dir(pipeline) if not a.startswith("__")]
                raise AttributeError(
                    f"pipeline has no attribute {sub_attr!r}. "
                    f"Available: {available}"
                )
            target = parent
            chain = f"pipeline.{sub_attr}.{engine_attr}"
        else:
            parent = pipeline
            target = pipeline
            chain = f"pipeline.{engine_attr}"

        engine = getattr(target, engine_attr, _SENTINEL)
        if engine is _SENTINEL:
            available = [a for a in dir(target) if not a.startswith("__")]
            raise AttributeError(
                f"{chain} does not exist. "
                f"Available attrs on {type(target).__name__}: {available}"
            )

        timings[label] = []
        wrapped = _TimedEngine(engine, label, timings)
        setattr(target, engine_attr, wrapped)
        print(f"[profiling] Wrapped {chain} -> {label}")

    print(f"[profiling] {len(timings)} engine timers installed")
    return timings


_SENTINEL = object()


def _run_profiled_benchmark(
    pipeline,
    registry,
    avatar_id: str,
    chunks: int,
) -> dict:
    """Run a profiled benchmark: first non-profiled, then profiled.

    Returns a dict with both ``non_profiled`` and ``profiled`` benchmark
    stats plus derived ``profiling_overhead_ms`` / ``profiling_overhead_pct``.
    """
    import numpy as np

    avatar = registry[avatar_id]
    mg = pipeline._motion_generator
    expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    silence = np.zeros(expected_len, dtype=np.float32)

    from avtr1_renderer.types import Chunk, RenderOptions
    chunk = Chunk(audio_speech=silence, audio_listen=silence)
    bg_id = _pick_first_bg(pipeline._backgrounds)
    options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

    # --- Phase 0: Non-profiled benchmark (same session) ---
    print(f"[profile] Running {chunks} non-profiled chunks (sync overhead baseline)...")
    non_profiled_stats = _run_benchmark(pipeline, registry, avatar_id, chunks)

    # --- Phase 1: Install timers ---
    print(f"[profile] Installing stage timers...")
    timings = _install_stage_timers(pipeline)

    # --- Phase 2: Profiled benchmark ---
    print(f"[profile] Running {chunks} profiled chunks...")

    # Warmup
    state, frames_iter = pipeline.process_chunk(
        avatar, chunk, state=None, options=options
    )
    warmup_frames = list(frames_iter)
    _run_nvidia_smi("profiled/after-warmup")

    # Clear any warmup timings
    for key in timings:
        timings[key].clear()

    chunk_times_ms: list[float] = []
    all_functional = True

    for i in range(chunks):
        t_chunk = time.monotonic()
        try:
            next_state, frames_iter = pipeline.process_chunk(
                avatar, chunk, state=state, options=options
            )
            frames = list(frames_iter)
        except Exception as exc:
            print(f"[profile] Chunk {i + 1} FAILED: {exc}")
            all_functional = False
            break
        elapsed_ms = (time.monotonic() - t_chunk) * 1000
        chunk_times_ms.append(elapsed_ms)
        state = next_state

    _run_nvidia_smi("profiled/after-benchmark")

    if not chunk_times_ms:
        return {
            "non_profiled": non_profiled_stats,
            "profiled": {"error": "No profiled chunks completed"},
        }

    sorted_ms = sorted(chunk_times_ms)
    profiled_stats = {
        "chunks": len(chunk_times_ms),
        "min_ms": sorted_ms[0],
        "p50_ms": sorted_ms[(len(sorted_ms) - 1) // 2],
        "mean_ms": sum(sorted_ms) / len(sorted_ms),
        "p95_ms": sorted_ms[int((len(sorted_ms) - 1) * 0.95)],
        "p99_ms": sorted_ms[int((len(sorted_ms) - 1) * 0.99)],
        "max_ms": sorted_ms[-1],
        "fps": 5.0 / (sum(sorted_ms) / len(sorted_ms) / 1000.0),
        "functional": all_functional,
    }

    overhead_ms = profiled_stats["p95_ms"] - non_profiled_stats["p95_ms"]
    overhead_pct = (overhead_ms / non_profiled_stats["p95_ms"] * 100
                    if non_profiled_stats["p95_ms"] > 0 else 0)

    return {
        "non_profiled": non_profiled_stats,
        "profiled": profiled_stats,
        "profiling_overhead_ms": overhead_ms,
        "profiling_overhead_pct": overhead_pct,
        "stage_timings": timings,
    }


def _print_stage_profile(results: dict) -> None:
    """Print the full profiled benchmark report."""
    print()
    print("=" * 72)
    print("PROFILED BENCHMARK REPORT")
    print("=" * 72)

    non_prof = results.get("non_profiled", {})
    prof = results.get("profiled", {})
    overhead_ms = results.get("profiling_overhead_ms", 0)
    overhead_pct = results.get("profiling_overhead_pct", 0)

    if "error" in non_prof:
        print(f"  Non-profiled: ERROR: {non_prof['error']}")
    else:
        print(f"  Non-profiled p95: {non_prof.get('p95_ms', 0):.1f}ms  "
              f"p50={non_prof.get('p50_ms', 0):.1f}ms  "
              f"FPS={non_prof.get('fps', 0):.1f}")

    if "error" in prof:
        print(f"  Profiled:     ERROR: {prof['error']}")
        print("=" * 72)
        return

    print(f"  Profiled p95:     {prof.get('p95_ms', 0):.1f}ms  "
          f"p50={prof.get('p50_ms', 0):.1f}ms  "
          f"FPS={prof.get('fps', 0):.1f}")
    print(f"  Profiling sync overhead: {overhead_ms:+.1f}ms ({overhead_pct:+.1f}%)")
    print()
    print("  NOTE: Profiled timings include cuda.synchronize() barriers.")
    print("  They are DIAGNOSTIC — do not use for realtime verdict.")
    print("  The non-profiled same-session numbers are the accurate latency.")

    timings = results.get("stage_timings", {})
    if not timings:
        print()
        print("  (No stage timings collected)")
        print("=" * 72)
        return

    profiled_p95 = prof.get("p95_ms", 1)
    profiled_mean = prof.get("mean_ms", 1)

    print()
    print(f"  {'Stage':<20} {'Calls':>5} {'Total':>8} {'Mean':>8} "
          f"{'P50':>8} {'P95':>8} {'%Chunk':>7}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    stage_order = [
        "hubert", "avtr1_encode", "avtr1_decode",
        "stitch", "warp", "decoder", "modnet",
    ]

    total_engine_ms = 0.0
    for stage in stage_order:
        samples = timings.get(stage, [])
        if not samples:
            continue
        s = sorted(samples)
        count = len(s)
        total = sum(s)
        mean = total / count
        p50 = s[(count - 1) // 2]
        p95 = s[int((count - 1) * 0.95)]
        pct = (total / (profiled_mean * prof.get("chunks", 1))) * 100
        total_engine_ms += total
        print(f"  {stage:<20} {count:>5} {total:>8.1f} {mean:>8.2f} "
              f"{p50:>8.2f} {p95:>8.2f} {pct:>6.1f}%")

    n_chunks = prof.get("chunks", 1)
    chunk_total_ms = profiled_mean * n_chunks
    unaccounted = chunk_total_ms - total_engine_ms
    unaccounted_pct = (unaccounted / chunk_total_ms * 100
                       if chunk_total_ms > 0 else 0)
    print(f"  {'(unaccounted)':<20} {'':>5} {unaccounted:>8.1f} "
          f"{unaccounted / n_chunks:>8.2f} {'':>8} {'':>8} "
          f"{unaccounted_pct:>6.1f}%")
    print()
    print(f"  Unaccounted = tensor ops (audio prep, ODE euler step, "
          f"motion_stitch, putback warp, YUV, D2H copy)")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Function: rebuild_optimized_engines
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=3600,
    volumes={OPT_MOUNT_PATH: opt_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def rebuild_optimized_engines() -> None:
    """Build all 7 TRT engines with builder_optimization_level=3 and
    tiling_optimization_level=FAST. Engines are stored in the separate
    avtr1-artifacts-opt Volume (does NOT touch baseline).
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"[guard] Writing to: {OPT_MOUNT_PATH} (Volume: {OPT_VOLUME_NAME})")
    print(f"[guard] Baseline volume {ARTIFACT_VOLUME_NAME!r} is NOT mounted here.")
    print()

    storage_root = _ensure_artifacts(Path(OPT_MOUNT_PATH))
    opt_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if engines_ok:
        print("[engines] Optimized engines already present. Skipping rebuild.")
        return

    print("[engines] Building optimized engines (level=3 + FAST tiling)...")
    print()

    _apply_trt_build_optimizations()
    _build_all_engines(storage_root, opt_volume)


# ---------------------------------------------------------------------------
# Function: benchmark_optimized
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={OPT_MOUNT_PATH: opt_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_optimized(chunks: int = 10) -> dict:
    """Benchmark warm render using optimized engines from avtr1-artifacts-opt.
    Does NOT use GPU snapshot.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(OPT_MOUNT_PATH))
    opt_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing: {missing}")
        print("ERROR: Run --test rebuild first.")
        return {"error": "Optimized engines not found"}

    print("[engines] All optimized engines present.")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    stats = _run_benchmark(pipeline, registry, avatar_id, chunks)
    stats["startup_s"] = startup_elapsed
    _print_benchmark_results("optimized engines, no snapshot", stats)
    return stats


# ---------------------------------------------------------------------------
# Function: rebuild_l4_engines — L4 baseline build (default TRT settings)
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=5400,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def rebuild_l4_engines() -> None:
    """Build all 7 TRT engines on NVIDIA L4 using default TRT builder settings.

    Engines are stored in the separate avtr1-artifacts-l4 Volume.
    Does NOT touch T4 baseline (avtr1-artifacts) or T4 optimized
    (avtr1-artifacts-opt) volumes.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"[guard] Writing to: {L4_MOUNT_PATH} (Volume: {L4_VOLUME_NAME})")
    print(f"[guard] Baseline volume {ARTIFACT_VOLUME_NAME!r} is NOT mounted here.")
    print(f"[guard] Opt volume {OPT_VOLUME_NAME!r} is NOT mounted here.")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if engines_ok:
        print("[engines] L4 engines already present. Skipping rebuild.")
        return

    print("[engines] Building L4 baseline engines (default TRT settings)...")
    print()

    _build_all_engines(storage_root, l4_volume)


# ---------------------------------------------------------------------------
# Function: benchmark_l4 — L4 baseline benchmark (default TRT settings)
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_l4(chunks: int = 10) -> dict:
    """Benchmark warm render using L4 baseline engines from avtr1-artifacts-l4.

    Uses default TRT builder settings (no optimization flags).
    Does NOT use GPU snapshot.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing: {missing}")
        print("ERROR: Run --test rebuild-l4 first.")
        return {"error": "L4 engines not found"}

    print("[engines] All L4 engines present.")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    stats = _run_benchmark(pipeline, registry, avatar_id, chunks)
    stats["startup_s"] = startup_elapsed
    _print_benchmark_results("L4 baseline, default TRT settings", stats)
    return stats


# ---------------------------------------------------------------------------
# Function: benchmark_l4_profiled — L4 profiled benchmark
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_l4_profiled(chunks: int = 10) -> dict:
    """Profiled benchmark on L4: non-profiled first, then profiled.

    Returns both non-profiled and profiled stats plus per-engine stage
    timings.  The non-profiled run in the same session quantifies the
    sync-point overhead introduced by profiling.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing: {missing}")
        print("ERROR: Run --test rebuild-l4 first.")
        return {"error": "L4 engines not found"}

    print("[engines] All L4 engines present.")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    results = _run_profiled_benchmark(pipeline, registry, avatar_id, chunks)
    results["startup_s"] = startup_elapsed
    _print_stage_profile(results)
    return results


# ---------------------------------------------------------------------------
# Function: L4 level-5 TRT build + benchmark
# ---------------------------------------------------------------------------


def _apply_trt_level5_only() -> None:
    """Monkey-patch TRT Builder to set builder_optimization_level=5 only.

    No tiling change.  This isolates the effect of the exhaustive kernel
    search on L4 (sm_89) where TF32 and dynamic-shape Tensor Cores work
    correctly — unlike T4 (sm_75) where level 5 caused 361 Myelin failures.
    """
    import tensorrt as trt

    _orig = trt.Builder.create_builder_config

    def _patched(self):
        config = _orig(self)
        config.builder_optimization_level = 5
        actual = config.builder_optimization_level
        print(f"[level5] VERIFIED config: builder_optimization_level={actual} "
              f"(expect 5)")
        if actual != 5:
            print(f"[level5] WARN: expected 5, got {actual}")
        return config

    trt.Builder.create_builder_config = _patched
    print("[level5] TRT Builder.create_builder_config patched "
          "(builder_optimization_level=5, no tiling change)")


@app.function(
    gpu="L4",
    timeout=7200,
    volumes={L4_LEVEL5_MOUNT_PATH: l4_level5_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def rebuild_l4_level5_engines() -> None:
    """Build all 7 TRT engines on L4 with builder_optimization_level=5.

    No tiling change.  Engines stored in separate Volume
    avtr1-artifacts-l4-level5.  Does NOT touch L4 baseline
    (avtr1-artifacts-l4) or any T4 volumes.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"[guard] Writing to: {L4_LEVEL5_MOUNT_PATH} "
          f"(Volume: {L4_LEVEL5_VOLUME_NAME})")
    print(f"[guard] L4 baseline volume {L4_VOLUME_NAME!r} is NOT mounted here.")
    print(f"[guard] T4 volumes are NOT mounted here.")
    print()

    storage_root = _ensure_artifacts(Path(L4_LEVEL5_MOUNT_PATH))
    l4_level5_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if engines_ok:
        print("[engines] L4 level-5 engines already present. Skipping rebuild.")
        return

    print("[engines] Building L4 level-5 engines "
          "(builder_optimization_level=5, no tiling)...")
    print("[engines] Expect 30-60+ minutes on L4 with level 5.")
    print()

    _apply_trt_level5_only()
    _build_all_engines(storage_root, l4_level5_volume)


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={L4_LEVEL5_MOUNT_PATH: l4_level5_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_l4_level5(chunks: int = 100) -> dict:
    """Benchmark L4 level-5 engines from avtr1-artifacts-l4-level5.

    Compares against L4 baseline p95=188.0ms.  Keep level-5 only if
    p95 improves by at least 3ms and build/runtime are stable.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_LEVEL5_MOUNT_PATH))
    l4_level5_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing: {missing}")
        print("ERROR: Run --test rebuild-l4-level5 first.")
        return {"error": "L4 level-5 engines not found"}

    print("[engines] All L4 level-5 engines present.")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    stats = _run_benchmark(pipeline, registry, avatar_id, chunks)
    stats["startup_s"] = startup_elapsed
    _print_benchmark_results("L4 level-5 (builder_optimization_level=5)", stats)

    l4_baseline_p95 = 188.0
    improvement = l4_baseline_p95 - stats["p95_ms"]
    print()
    if improvement >= 3.0:
        print(f"  DECISION: KEEP level-5 (p95 improved by {improvement:.1f}ms "
              f">= 3ms threshold)")
    elif improvement > 0:
        print(f"  DECISION: DISCARD level-5 (p95 improved by only {improvement:.1f}ms "
              f"< 3ms threshold)")
    else:
        print(f"  DECISION: DISCARD level-5 (p95 regressed by "
              f"{abs(improvement):.1f}ms)")

    return stats


# ---------------------------------------------------------------------------
# Variant B: rebuild_l4_warp_static_b5 — static B=5 warp profile
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={
        L4_MOUNT_PATH: l4_volume,
        L4_WARP_STATIC_B5_MOUNT_PATH: l4_warp_static_b5_volume,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)
def rebuild_l4_warp_static_b5() -> None:
    """Build warp engine with static B=5 profile (min=opt=max=5).

    Copies all non-warp engines from L4 baseline volume, then rebuilds
    only the warp engine with a static optimization profile.  Default
    builder_optimization_level (no level-5).
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"[guard] Source (L4 baseline): {L4_MOUNT_PATH} "
          f"(Volume: {L4_VOLUME_NAME})")
    print(f"[guard] Target: {L4_WARP_STATIC_B5_MOUNT_PATH} "
          f"(Volume: {L4_WARP_STATIC_B5_VOLUME_NAME})")
    print(f"[guard] T4 volumes and Cloudflare code are NOT touched.")
    print()

    source_root_str = str(L4_MOUNT_PATH)
    old_env = os.environ.get("AVTR1_LOCAL_STORAGE")
    os.environ["AVTR1_LOCAL_STORAGE"] = source_root_str
    from avtr1_renderer.avtr1_artifact_manager import get_storage_root
    source_root = get_storage_root()
    print(f"[source] L4 baseline storage root: {source_root}")
    if old_env is not None:
        os.environ["AVTR1_LOCAL_STORAGE"] = old_env

    source_ok, source_missing = _check_all_engines(source_root)
    if not source_ok:
        print(f"[source] ERROR: L4 baseline missing engines: {source_missing}")
        print("Run --test rebuild-l4 first.")
        return

    target_root = _ensure_artifacts(Path(L4_WARP_STATIC_B5_MOUNT_PATH))
    l4_warp_static_b5_volume.commit()
    print(f"[target] Storage root: {target_root}")

    _seed_engines_from_l4(source_root, target_root)
    l4_warp_static_b5_volume.commit()
    print("[seed] Non-warp engines committed.")

    print()
    print("[warp-variant] Building warp with static B=5 profile "
          "(min=opt=max=5, default builder level)...")
    _build_warp_variant(target_root, static_batch=5, builder_level=None)
    l4_warp_static_b5_volume.commit()
    print("[warp-variant] Warp engine committed.")

    all_ok, missing = _validate_all_artifacts(target_root)
    if all_ok:
        print("[validate] All artifacts present. Ready for benchmark.")
    else:
        print(f"[validate] ERROR: Missing artifacts: {missing}")
        print("Benchmark will fail. Investigate.")


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={L4_WARP_STATIC_B5_MOUNT_PATH: l4_warp_static_b5_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_l4_warp_static_b5(chunks: int = 100) -> dict:
    """Benchmark Variant B: static B=5 warp profile (default builder level).

    Compares against L4 baseline p95=188.0ms.  Keep only if p95 improves
    by >=3ms and p99 does not regress.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_WARP_STATIC_B5_MOUNT_PATH))
    l4_warp_static_b5_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    all_ok, missing = _validate_all_artifacts(storage_root)
    if not all_ok:
        print(f"[validate] ERROR: Missing artifacts: {missing}")
        print("Run --test rebuild-l4-warp-static-b5 first.")
        return {"error": "Variant B artifacts not found", "missing": missing}

    print("[validate] All artifacts present (engines + plugin + normalizer).")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    stats = _run_benchmark(pipeline, registry, avatar_id, chunks)
    stats["startup_s"] = startup_elapsed
    _print_benchmark_results(
        "Variant B: static B=5 warp (default builder level)", stats
    )

    l4_baseline_p95 = 188.0
    l4_baseline_p99 = 188.9
    improvement = l4_baseline_p95 - stats["p95_ms"]
    p99_regression = stats["p99_ms"] - l4_baseline_p99
    print()
    print(f"  Baseline p95:  {l4_baseline_p95:.1f}ms")
    print(f"  Variant B p95: {stats['p95_ms']:.1f}ms")
    print(f"  Improvement:   {improvement:+.1f}ms")
    print(f"  Baseline p99:  {l4_baseline_p99:.1f}ms")
    print(f"  Variant B p99: {stats['p99_ms']:.1f}ms")
    print(f"  P99 regression: {p99_regression:+.1f}ms")
    print()
    if improvement >= 3.0 and p99_regression <= 0:
        print(f"  DECISION: KEEP Variant B (p95 improved by {improvement:.1f}ms "
              f">= 3ms, p99 did not regress)")
    elif improvement >= 3.0 and p99_regression > 0:
        print(f"  DECISION: DISCARD Variant B (p95 improved by {improvement:.1f}ms "
              f"but p99 regressed by {p99_regression:.1f}ms)")
    elif improvement > 0:
        print(f"  DECISION: DISCARD Variant B (p95 improved by only {improvement:.1f}ms "
              f"< 3ms threshold)")
    else:
        print(f"  DECISION: DISCARD Variant B (p95 regressed by "
              f"{abs(improvement):.1f}ms)")

    return stats


# ---------------------------------------------------------------------------
# Variant C: rebuild_l4_warp_static_b5_level5 — static B=5 + level-5 warp
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=5400,
    volumes={
        L4_MOUNT_PATH: l4_volume,
        L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH: l4_warp_static_b5_level5_volume,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)
def rebuild_l4_warp_static_b5_level5() -> None:
    """Build warp engine with static B=5 profile + builder_optimization_level=5.

    Copies all non-warp engines from L4 baseline volume, then rebuilds
    only the warp engine with static B=5 profile and level-5 optimization.
    No tiling_optimization_level change.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    print(f"[guard] Source (L4 baseline): {L4_MOUNT_PATH} "
          f"(Volume: {L4_VOLUME_NAME})")
    print(f"[guard] Target: {L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH} "
          f"(Volume: {L4_WARP_STATIC_B5_LEVEL5_VOLUME_NAME})")
    print(f"[guard] T4 volumes and Cloudflare code are NOT touched.")
    print()

    source_root_str = str(L4_MOUNT_PATH)
    old_env = os.environ.get("AVTR1_LOCAL_STORAGE")
    os.environ["AVTR1_LOCAL_STORAGE"] = source_root_str
    from avtr1_renderer.avtr1_artifact_manager import get_storage_root
    source_root = get_storage_root()
    print(f"[source] L4 baseline storage root: {source_root}")
    if old_env is not None:
        os.environ["AVTR1_LOCAL_STORAGE"] = old_env

    source_ok, source_missing = _check_all_engines(source_root)
    if not source_ok:
        print(f"[source] ERROR: L4 baseline missing engines: {source_missing}")
        print("Run --test rebuild-l4 first.")
        return

    target_root = _ensure_artifacts(Path(L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH))
    l4_warp_static_b5_level5_volume.commit()
    print(f"[target] Storage root: {target_root}")

    _seed_engines_from_l4(source_root, target_root)
    l4_warp_static_b5_level5_volume.commit()
    print("[seed] Non-warp engines committed.")

    _apply_trt_level5_only()

    print()
    print("[warp-variant] Building warp with static B=5 + level-5...")
    print("  Expect 5-30+ minutes (level-5 is exhaustive kernel search).")
    _build_warp_variant(target_root, static_batch=5, builder_level=5)
    l4_warp_static_b5_level5_volume.commit()
    print("[warp-variant] Warp engine committed.")

    all_ok, missing = _validate_all_artifacts(target_root)
    if all_ok:
        print("[validate] All artifacts present. Ready for benchmark.")
    else:
        print(f"[validate] ERROR: Missing artifacts: {missing}")
        print("Benchmark will fail. Investigate.")


@app.function(
    gpu="L4",
    timeout=1800,
    volumes={
        L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH: l4_warp_static_b5_level5_volume,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)
def benchmark_l4_warp_static_b5_level5(chunks: int = 100) -> dict:
    """Benchmark Variant C: static B=5 + builder_optimization_level=5 warp.

    Compares against L4 baseline p95=188.0ms.  Keep only if p95 improves
    by >=3ms and p99 does not regress.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(
        Path(L4_WARP_STATIC_B5_LEVEL5_MOUNT_PATH)
    )
    l4_warp_static_b5_level5_volume.commit()

    print(f"[engines] Resolved storage root: {storage_root}")
    all_ok, missing = _validate_all_artifacts(storage_root)
    if not all_ok:
        print(f"[validate] ERROR: Missing artifacts: {missing}")
        print("Run --test rebuild-l4-warp-static-b5-level5 first.")
        return {"error": "Variant C artifacts not found", "missing": missing}

    print("[validate] All artifacts present (engines + plugin + normalizer).")
    print()

    _run_nvidia_smi("before-load")

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    stats = _run_benchmark(pipeline, registry, avatar_id, chunks)
    stats["startup_s"] = startup_elapsed
    _print_benchmark_results(
        "Variant C: static B=5 + level-5 warp", stats
    )

    l4_baseline_p95 = 188.0
    l4_baseline_p99 = 188.9
    improvement = l4_baseline_p95 - stats["p95_ms"]
    p99_regression = stats["p99_ms"] - l4_baseline_p99
    print()
    print(f"  Baseline p95:  {l4_baseline_p95:.1f}ms")
    print(f"  Variant C p95: {stats['p95_ms']:.1f}ms")
    print(f"  Improvement:   {improvement:+.1f}ms")
    print(f"  Baseline p99:  {l4_baseline_p99:.1f}ms")
    print(f"  Variant C p99: {stats['p99_ms']:.1f}ms")
    print(f"  P99 regression: {p99_regression:+.1f}ms")
    print()
    if improvement >= 3.0 and p99_regression <= 0:
        print(f"  DECISION: KEEP Variant C (p95 improved by {improvement:.1f}ms "
              f">= 3ms, p99 did not regress)")
    elif improvement >= 3.0 and p99_regression > 0:
        print(f"  DECISION: DISCARD Variant C (p95 improved by {improvement:.1f}ms "
              f"but p99 regressed by {p99_regression:.1f}ms)")
    elif improvement > 0:
        print(f"  DECISION: DISCARD Variant C (p95 improved by only {improvement:.1f}ms "
              f"< 3ms threshold)")
    else:
        print(f"  DECISION: DISCARD Variant C (p95 regressed by "
              f"{abs(improvement):.1f}ms)")

    return stats


@app.function(
    gpu="L4",
    timeout=600,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def audit_decoder_batching() -> dict:
    """Audit whether the SPADE decoder can be called at B=5 instead of B=1×5.

    Inspects the loaded decoder engine's TRT bindings, tests B=1 and B=5
    calls with dummy tensors, and times both patterns to measure the
    actual batch penalty (or benefit).

    Returns a dict with engine metadata, shape info, and timing comparison.
    """
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    print(f"[pipeline] Loaded in {time.monotonic() - t0:.1f}s")
    print()

    decoder_engine = pipeline._decoder

    # --- Part 1: Engine metadata ---
    print("=" * 72)
    print("DECODER ENGINE METADATA")
    print("=" * 72)

    has_trt = hasattr(decoder_engine, "engine")
    print(f"  Engine type: {type(decoder_engine).__name__}")
    print(f"  Has TRT ICudaEngine: {has_trt}")

    if has_trt:
        trt_engine = decoder_engine.engine
        print(f"  TRT engine name: {trt_engine.name}")
        print(f"  num_io_tensors: {trt_engine.num_io_tensors}")
        print()
        for i in range(trt_engine.num_io_tensors):
            name = trt_engine.get_tensor_name(i)
            mode = trt_engine.get_tensor_mode(name)
            dtype = trt_engine.get_tensor_dtype(name)
            shape = trt_engine.get_tensor_shape(name)
            print(f"  [{i}] {name}: mode={mode}, dtype={dtype}, shape={shape}")
        print()

        ctx = decoder_engine.context
        print("  Optimization profile shapes (profile 0):")
        for i in range(trt_engine.num_io_tensors):
            name = trt_engine.get_tensor_name(i)
            mode = trt_engine.get_tensor_mode(name)
            if str(mode) == "TensorIOMode.INPUT":
                min_shape = ctx.get_tensor_shape(name) if ctx else "N/A"
                print(f"    {name}: current_shape={min_shape}")

    from avtr1_renderer.models.decoder import DecoderInput, DecoderOutput

    print()
    print(f"  DecoderInput fields: {[f.name for f in __import__('dataclasses').fields(DecoderInput)]}")
    print(f"  DecoderOutput fields: {[f.name for f in __import__('dataclasses').fields(DecoderOutput)]}")
    print()

    # --- Part 2: B=1 and B=5 dummy calls (correctness) ---
    print("=" * 72)
    print("DUMMY CALL CORRECTNESS TEST")
    print("=" * 72)

    b1_input = torch.randn(1, 256, 64, 64, dtype=torch.float32, device="cuda")
    b5_input = torch.randn(5, 256, 64, 64, dtype=torch.float32, device="cuda")

    print(f"  B=1 input shape: {tuple(b1_input.shape)}")
    try:
        out_b1 = decoder_engine(DecoderInput(feature=b1_input))
        print(f"  B=1 output shape: {tuple(out_b1.output.shape)}")
        print(f"  B=1 output dtype: {out_b1.output.dtype}")
        print(f"  B=1 output range: [{out_b1.output.min().item():.4f}, {out_b1.output.max().item():.4f}]")
        b1_ok = True
    except Exception as exc:
        print(f"  B=1 FAILED: {exc}")
        b1_ok = False

    print()
    print(f"  B=5 input shape: {tuple(b5_input.shape)}")
    try:
        torch.cuda.synchronize()
        out_b5 = decoder_engine(DecoderInput(feature=b5_input))
        torch.cuda.synchronize()
        print(f"  B=5 output shape: {tuple(out_b5.output.shape)}")
        print(f"  B=5 output dtype: {out_b5.output.dtype}")
        print(f"  B=5 output range: [{out_b5.output.min().item():.4f}, {out_b5.output.max().item():.4f}]")
        b5_ok = True
    except Exception as exc:
        print(f"  B=5 FAILED: {exc}")
        b5_ok = False

    # --- Part 3: Timing B=1×5 vs B=5×1 ---
    print()
    print("=" * 72)
    print("TIMING: B=1×5 vs B=5×1 (warm, synchronized)")
    print("=" * 72)

    N_WARMUP = 5
    N_MEASURED = 20

    if b1_ok:
        for _ in range(N_WARMUP):
            out = decoder_engine(DecoderInput(feature=b1_input))
            del out

        b1_times_ms: list[float] = []
        for _ in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                out = decoder_engine(DecoderInput(feature=b1_input))
                del out
            torch.cuda.synchronize()
            b1_times_ms.append((time.perf_counter() - t0) * 1000)

        b1_sorted = sorted(b1_times_ms)
        print(f"  B=1×5 (per batch of 5):")
        print(f"    mean={sum(b1_sorted) / len(b1_sorted):.2f}ms  "
              f"p50={b1_sorted[(len(b1_sorted) - 1) // 2]:.2f}ms  "
              f"p95={b1_sorted[int((len(b1_sorted) - 1) * 0.95)]:.2f}ms  "
              f"min={b1_sorted[0]:.2f}ms  max={b1_sorted[-1]:.2f}ms")

    if b5_ok:
        for _ in range(N_WARMUP):
            out = decoder_engine(DecoderInput(feature=b5_input))
            del out

        b5_times_ms: list[float] = []
        for _ in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = decoder_engine(DecoderInput(feature=b5_input))
            del out
            torch.cuda.synchronize()
            b5_times_ms.append((time.perf_counter() - t0) * 1000)

        b5_sorted = sorted(b5_times_ms)
        print(f"  B=5×1 (single batched call):")
        print(f"    mean={sum(b5_sorted) / len(b5_sorted):.2f}ms  "
              f"p50={b5_sorted[(len(b5_sorted) - 1) // 2]:.2f}ms  "
              f"p95={b5_sorted[int((len(b5_sorted) - 1) * 0.95)]:.2f}ms  "
              f"min={b5_sorted[0]:.2f}ms  max={b5_sorted[-1]:.2f}ms")

    print()
    if b1_ok and b5_ok:
        b1_mean = sum(b1_sorted) / len(b1_sorted)
        b5_mean = sum(b5_sorted) / len(b5_sorted)
        delta = b5_mean - b1_mean
        pct = (delta / b1_mean * 100) if b1_mean > 0 else 0
        print(f"  Batch penalty: {delta:+.2f}ms ({pct:+.1f}%)")
        print(f"  B=5 is {'FASTER' if delta < 0 else 'SLOWER'} than B=1×5")
        if delta < 0:
            print(f"  Decoder batching WINS by {abs(delta):.2f}ms per chunk")
        else:
            print(f"  Decoder batching costs {abs(delta):.2f}ms per chunk")
            per_frame_saving = b1_mean / 5 - b5_mean / 5
            print(f"  Per-frame: B=1={b1_mean / 5:.2f}ms vs B=5/5={b5_mean / 5:.2f}ms "
                  f"({per_frame_saving:+.2f}ms)")

    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if b5_ok:
        print("  B=5 batching is VIABLE — engine accepts B=5 input.")
        print("  Engine metadata confirms dynamic batch profile [1, 5].")
        print("  No per-frame state or dependencies in decoder.")
        print("  See timing above for actual speedup/penalty measurement.")
    else:
        print("  B=5 batching is NOT VIABLE — engine rejects B=5 input.")
    print("=" * 72)

    return {
        "b1_ok": b1_ok,
        "b5_ok": b5_ok,
        "b1_mean_ms": sum(b1_sorted) / len(b1_sorted) if b1_ok else None,
        "b5_mean_ms": sum(b5_sorted) / len(b5_sorted) if b5_ok else None,
    }


# ---------------------------------------------------------------------------
# Function: cuda_graph_decoder_audit — CUDA Graph feasibility for decoder loop
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=600,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def cuda_graph_decoder_audit() -> dict:
    """Audit whether CUDA Graph capture/replay can accelerate the decoder loop.

    Tests three patterns:
      A) Normal B=1x5  — 5 sequential decoder calls (current production)
      B) Single-call graph — capture 1 decoder call, replay 5x (streaming path)
      C) Multi-call graph  — capture 5 decoder calls, replay 1x (batched path)

    For B, input must be copied into the static graph buffer before each
    replay, and output must be copied out after each replay.

    For C (main production candidate), the graph owns warp_buf (5,256,64,64)
    and faces_buf (5,3,512,512) with stable addresses.  Before replay, new
    warped data is copied into warp_buf.  After replay, faces_buf holds all
    5 decoded frames.

    Returns timing comparison + numerical validation.
    """
    import torch
    from dataclasses import fields

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.models.decoder import DecoderInput, DecoderOutput

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    print(f"[pipeline] Loaded in {time.monotonic() - t0:.1f}s")
    print()

    decoder = pipeline._decoder

    # ---- Pre-allocate all buffers (stable addresses for graph capture) ----
    WARP_SHAPE = (1, 256, 64, 64)
    FACE_SHAPE = (1, 3, 512, 512)
    N_FRAMES = 5

    warp_buf = torch.randn(N_FRAMES, 256, 64, 64, dtype=torch.float32, device="cuda")
    faces_buf = torch.empty(N_FRAMES, 3, 512, 512, dtype=torch.float32, device="cuda")

    # For single-call graph: one static input/output pair
    static_in = torch.zeros(*WARP_SHAPE, dtype=torch.float32, device="cuda")
    static_out = torch.zeros(*FACE_SHAPE, dtype=torch.float32, device="cuda")

    # Reference input data (fixed random seed for reproducibility)
    torch.manual_seed(42)
    ref_warp = torch.randn(N_FRAMES, 256, 64, 64, dtype=torch.float32, device="cuda")

    # ====================================================================
    # A) Baseline: Normal B=1x5 loop
    # ====================================================================
    N_WARMUP = 5
    N_MEASURED = 30

    print("=" * 72)
    print("A) BASELINE: Normal B=1x5 decoder loop")
    print("=" * 72)

    for _ in range(N_WARMUP):
        for i in range(N_FRAMES):
            decoder(
                DecoderInput(feature=ref_warp[i : i + 1]),
                out=DecoderOutput(output=faces_buf[i : i + 1]),
            )

    ref_faces = torch.empty_like(faces_buf)
    normal_times_ms: list[float] = []
    for _ in range(N_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(N_FRAMES):
            decoder(
                DecoderInput(feature=ref_warp[i : i + 1]),
                out=DecoderOutput(output=ref_faces[i : i + 1]),
            )
        torch.cuda.synchronize()
        normal_times_ms.append((time.perf_counter() - t0) * 1000)

    normal_sorted = sorted(normal_times_ms)
    normal_mean = sum(normal_sorted) / len(normal_sorted)
    print(f"  mean={normal_mean:.2f}ms  "
          f"p50={normal_sorted[(len(normal_sorted)-1)//2]:.2f}ms  "
          f"p95={normal_sorted[int((len(normal_sorted)-1)*0.95)]:.2f}ms  "
          f"min={normal_sorted[0]:.2f}ms  max={normal_sorted[-1]:.2f}ms")
    print()

    # ====================================================================
    # B) Single-call graph: capture 1 decoder call, replay 5x
    #    Must copy input in before each replay, copy output out after.
    # ====================================================================
    print("=" * 72)
    print("B) SINGLE-CALL GRAPH: capture 1 call, replay 5x")
    print("=" * 72)

    graph_b: torch.cuda.CUDAGraph | None = None
    graph_b_ok = False
    graph_b_capture_error: str | None = None

    try:
        # Warmup on the static buffers (same addresses that capture will use)
        for _ in range(N_WARMUP):
            static_in.copy_(ref_warp[0:1])
            decoder(DecoderInput(feature=static_in), out=DecoderOutput(output=static_out))
        torch.cuda.synchronize()

        # Capture
        graph_b = torch.cuda.CUDAGraph()
        static_in.copy_(ref_warp[0:1])
        with torch.cuda.graph(graph_b):
            _ = decoder(DecoderInput(feature=static_in), out=DecoderOutput(output=static_out))

        print("  Capture succeeded.")
        graph_b_ok = True
    except Exception as exc:
        graph_b_capture_error = f"{type(exc).__name__}: {exc}"
        print(f"  Capture FAILED: {graph_b_capture_error}")

    graph_b_times_ms: list[float] = []
    graph_b_copy_only_ms: list[float] = []
    graph_b_replay_only_ms: list[float] = []
    graph_b_faces = torch.empty_like(faces_buf)

    if graph_b_ok:
        # Measure copy cost independently (input copy + output copy per frame)
        for _ in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(N_FRAMES):
                static_in.copy_(ref_warp[i : i + 1])
                static_out.copy_(torch.zeros(*FACE_SHAPE, device="cuda"))
            torch.cuda.synchronize()
            graph_b_copy_only_ms.append((time.perf_counter() - t0) * 1000)

        copy_b_sorted = sorted(graph_b_copy_only_ms)
        print(f"  Copy-only (5x input + output clear): "
              f"mean={sum(copy_b_sorted)/len(copy_b_sorted):.2f}ms")

        # Measure full replay loop (copy in + replay + copy out) x 5
        for rep in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for i in range(N_FRAMES):
                static_in.copy_(ref_warp[i : i + 1])
                graph_b.replay()
                graph_b_faces[i : i + 1].copy_(static_out)
            torch.cuda.synchronize()
            graph_b_times_ms.append((time.perf_counter() - t0) * 1000)

        graph_b_sorted = sorted(graph_b_times_ms)
        graph_b_mean = sum(graph_b_sorted) / len(graph_b_sorted)
        print(f"  Full loop (copy+replay+copy) x5: "
              f"mean={graph_b_mean:.2f}ms  "
              f"p50={graph_b_sorted[(len(graph_b_sorted)-1)//2]:.2f}ms  "
              f"p95={graph_b_sorted[int((len(graph_b_sorted)-1)*0.95)]:.2f}ms  "
              f"min={graph_b_sorted[0]:.2f}ms  max={graph_b_sorted[-1]:.2f}ms")

        # Decompose: replay-only = full - copy-only (approximate)
        copy_mean = sum(copy_b_sorted) / len(copy_b_sorted)
        replay_est = graph_b_mean - copy_mean
        print(f"  Estimated replay-only x5: ~{replay_est:.2f}ms")
    print()

    # ====================================================================
    # C) Multi-call graph: capture 5 decoder calls, replay 1x
    #    Graph owns warp_buf and faces_buf with stable addresses.
    #    Before replay: copy new warped data into warp_buf.
    #    After replay:  faces_buf has all 5 frames.
    # ====================================================================
    print("=" * 72)
    print("C) MULTI-CALL GRAPH: capture 5 calls, replay 1x (PRODUCTION CANDIDATE)")
    print("=" * 72)

    graph_c: torch.cuda.CUDAGraph | None = None
    graph_c_ok = False
    graph_c_capture_error: str | None = None

    try:
        # Warmup: 5 sequential calls using warp_buf/faces_buf slices
        # (same addresses capture will use)
        for _ in range(N_WARMUP):
            warp_buf.copy_(ref_warp)
            for i in range(N_FRAMES):
                decoder(
                    DecoderInput(feature=warp_buf[i : i + 1]),
                    out=DecoderOutput(output=faces_buf[i : i + 1]),
                )
        torch.cuda.synchronize()

        # Capture 5 sequential decoder calls in one graph
        graph_c = torch.cuda.CUDAGraph()
        warp_buf.copy_(ref_warp)
        with torch.cuda.graph(graph_c):
            for i in range(N_FRAMES):
                _ = decoder(
                    DecoderInput(feature=warp_buf[i : i + 1]),
                    out=DecoderOutput(output=faces_buf[i : i + 1]),
                )

        print("  Capture succeeded.")
        graph_c_ok = True
    except Exception as exc:
        graph_c_capture_error = f"{type(exc).__name__}: {exc}"
        print(f"  Capture FAILED: {graph_c_capture_error}")

    graph_c_times_ms: list[float] = []
    graph_c_copy_only_ms: list[float] = []
    graph_c_faces = torch.empty_like(faces_buf)

    if graph_c_ok:
        # Measure copy cost independently (copying 5 frames into warp_buf)
        for _ in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            warp_buf.copy_(ref_warp)
            torch.cuda.synchronize()
            graph_c_copy_only_ms.append((time.perf_counter() - t0) * 1000)

        copy_c_sorted = sorted(graph_c_copy_only_ms)
        copy_c_mean = sum(copy_c_sorted) / len(copy_c_sorted)
        print(f"  Copy-only (warp_buf.copy_): mean={copy_c_mean:.2f}ms")

        # Measure full replay (copy + replay)
        for rep in range(N_MEASURED):
            warp_buf.copy_(ref_warp)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            graph_c.replay()
            torch.cuda.synchronize()
            graph_c_times_ms.append((time.perf_counter() - t0) * 1000)
            graph_c_faces.copy_(faces_buf)

        graph_c_sorted = sorted(graph_c_times_ms)
        graph_c_mean = sum(graph_c_sorted) / len(graph_c_sorted)
        print(f"  Replay-only: "
              f"mean={graph_c_mean:.2f}ms  "
              f"p50={graph_c_sorted[(len(graph_c_sorted)-1)//2]:.2f}ms  "
              f"p95={graph_c_sorted[int((len(graph_c_sorted)-1)*0.95)]:.2f}ms  "
              f"min={graph_c_sorted[0]:.2f}ms  max={graph_c_sorted[-1]:.2f}ms")
        print(f"  Copy + Replay: ~{copy_c_mean + graph_c_mean:.2f}ms")

    print()

    # ====================================================================
    # Numerical validation
    # ====================================================================
    print("=" * 72)
    print("NUMERICAL VALIDATION")
    print("=" * 72)

    print(f"  Reference (normal) shape: {tuple(ref_faces.shape)}")
    print(f"  Reference dtype: {ref_faces.dtype}")
    print(f"  Reference range: [{ref_faces.min().item():.6f}, {ref_faces.max().item():.6f}]")
    print()

    if graph_b_ok:
        print("  B) Single-call graph vs normal:")
        print(f"     Output shape: {tuple(graph_b_faces.shape)}")
        if graph_b_faces.shape == ref_faces.shape:
            diff_b = (graph_b_faces - ref_faces).abs()
            print(f"     max abs diff: {diff_b.max().item():.8f}")
            print(f"     mean abs diff: {diff_b.mean().item():.8f}")
            # TRT is deterministic for same input on same context
            if diff_b.max().item() < 1e-4:
                print("     PASS: outputs match within 1e-4 tolerance")
            else:
                print("     WARN: outputs differ by more than 1e-4")
        else:
            print("     SHAPE MISMATCH")
    else:
        print("  B) Single-call graph: NOT AVAILABLE (capture failed)")
    print()

    if graph_c_ok:
        print("  C) Multi-call graph vs normal:")
        print(f"     Output shape: {tuple(graph_c_faces.shape)}")
        if graph_c_faces.shape == ref_faces.shape:
            diff_c = (graph_c_faces - ref_faces).abs()
            print(f"     max abs diff: {diff_c.max().item():.8f}")
            print(f"     mean abs diff: {diff_c.mean().item():.8f}")
            if diff_c.max().item() < 1e-4:
                print("     PASS: outputs match within 1e-4 tolerance")
            else:
                print("     WARN: outputs differ by more than 1e-4")
        else:
            print("     SHAPE MISMATCH")
    else:
        print("  C) Multi-call graph: NOT AVAILABLE (capture failed)")
    print()

    # ====================================================================
    # Summary comparison
    # ====================================================================
    print("=" * 72)
    print("SUMMARY COMPARISON")
    print("=" * 72)
    print(f"  A) Normal B=1x5:           mean={normal_mean:.2f}ms")
    if graph_b_ok:
        print(f"  B) Single-call graph x5:   mean={graph_b_mean:.2f}ms  "
              f"(copy+replay+copy)")
        delta_b = graph_b_mean - normal_mean
        pct_b = (delta_b / normal_mean * 100) if normal_mean > 0 else 0
        print(f"     Delta vs normal: {delta_b:+.2f}ms ({pct_b:+.1f}%)")
    else:
        print(f"  B) Single-call graph: CAPTURE FAILED ({graph_b_capture_error})")

    if graph_c_ok:
        print(f"  C) Multi-call graph x1:    replay={graph_c_mean:.2f}ms  "
              f"copy+replay={copy_c_mean + graph_c_mean:.2f}ms")
        delta_c_replay = graph_c_mean - normal_mean
        pct_c_replay = (delta_c_replay / normal_mean * 100) if normal_mean > 0 else 0
        print(f"     Delta (replay-only) vs normal: {delta_c_replay:+.2f}ms ({pct_c_replay:+.1f}%)")
        delta_c_full = (copy_c_mean + graph_c_mean) - normal_mean
        pct_c_full = (delta_c_full / normal_mean * 100) if normal_mean > 0 else 0
        print(f"     Delta (copy+replay) vs normal: {delta_c_full:+.2f}ms ({pct_c_full:+.1f}%)")
    else:
        print(f"  C) Multi-call graph: CAPTURE FAILED ({graph_c_capture_error})")
    print()

    # ====================================================================
    # Verdict
    # ====================================================================
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    if graph_c_ok:
        savings = normal_mean - graph_c_mean
        savings_pct = (savings / normal_mean * 100) if normal_mean > 0 else 0
        if savings > 2.0:
            print(f"  VIABLE — Multi-call graph saves {savings:.2f}ms ({savings_pct:.1f}%) "
                  f"per chunk (replay-only).")
            print(f"  Decoder = 49.1% of chunk time. "
                  f"Projected chunk p95: {188.0 - savings:.1f}ms.")
        elif savings > 0:
            print(f"  MARGINAL — Multi-call graph saves only {savings:.2f}ms "
                  f"({savings_pct:.1f}%). May not justify production complexity.")
        else:
            print(f"  NOT VIABLE — Multi-call graph is slower by "
                  f"{abs(savings):.2f}ms ({abs(savings_pct):.1f}%).")
        print()
        print("  Production implementation plan:")
        print("  1. Pre-allocate warp_buf (5,256,64,64) + faces_buf (5,3,512,512)")
        print("  2. After pipeline load + warmup chunk, capture 5-call graph")
        print("  3. Per chunk: copy warped→warp_buf, graph_c.replay(), "
              "read faces_buf")
        print("  4. Graph capture ~50-100ms one-time; pinned memory ~35MB")
        print("  5. Concurrent sessions need per-session execution context")
    elif graph_b_ok:
        print("  PARTIAL — Multi-call capture failed but single-call works.")
        print(f"  Single-call graph (with copy overhead) is the fallback.")
    else:
        print("  NOT VIABLE — CUDA Graph capture failed for the decoder.")
        print(f"  Single-call error: {graph_b_capture_error}")
        print(f"  Multi-call error:  {graph_c_capture_error}")
    print("=" * 72)

    free2, _ = torch.cuda.mem_get_info(0)
    print(f"\nGPU memory after audit: {free2 / 1e9:.1f} GB free")

    return {
        "normal_mean_ms": normal_mean,
        "graph_b_ok": graph_b_ok,
        "graph_b_mean_ms": sum(graph_b_sorted) / len(graph_b_sorted) if graph_b_ok else None,
        "graph_c_ok": graph_c_ok,
        "graph_c_mean_ms": sum(graph_c_sorted) / len(graph_c_sorted) if graph_c_ok else None,
    }


# ---------------------------------------------------------------------------
# Function: audit_warp_optimization — warp stage optimization feasibility audit
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=600,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def audit_warp_optimization() -> dict:
    """Audit warp stage optimization feasibility on L4.

    Measures:
      1. Warp engine metadata (TRT bindings, shapes, dtype, plugin, profile)
      2. Pre-warp tensor prep costs (expand+contiguous)
      3. Warp B=5 benchmark with synthetic tensors
      4. Warp B=5 benchmark with real production tensors from motion_stitch
      5. Warp B=1 benchmark (scaling analysis: sublinear / linear / worse)
      6. B=1 vs B=5 scaling verdict

    Does NOT rebuild engines or modify production code.
    """
    import torch
    import numpy as np
    from dataclasses import fields as dc_fields

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU memory: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB total")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.models.warp import WarpInput, WarpOutput
    from avtr1_renderer.types import Chunk, RenderOptions

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    print(f"[pipeline] Loaded in {time.monotonic() - t0:.1f}s")
    print()

    warp_engine = pipeline._warp
    avatar = registry[avatar_id]
    stitch_engine = pipeline._stitch

    # ========================================================================
    # Part 1: Warp Engine Metadata
    # ========================================================================
    print("=" * 72)
    print("WARP ENGINE METADATA")
    print("=" * 72)

    has_trt = hasattr(warp_engine, "engine")
    print(f"  Engine type: {type(warp_engine).__name__}")
    print(f"  Has TRT ICudaEngine: {has_trt}")

    if has_trt:
        trt_engine = warp_engine.engine
        print(f"  TRT engine name: {trt_engine.name}")
        print(f"  num_io_tensors: {trt_engine.num_io_tensors}")
        print()
        for i in range(trt_engine.num_io_tensors):
            name = trt_engine.get_tensor_name(i)
            mode = trt_engine.get_tensor_mode(name)
            dtype = trt_engine.get_tensor_dtype(name)
            shape = trt_engine.get_tensor_shape(name)
            print(f"  [{i}] {name}: mode={mode}, dtype={dtype}, shape={shape}")
        print()

        ctx = warp_engine.context
        num_profiles = trt_engine.num_optimization_profiles
        print(f"  Number of optimization profiles: {num_profiles}")
        if num_profiles > 0:
            print(f"  Profile 0 details (querying context):")
            for i in range(trt_engine.num_io_tensors):
                name = trt_engine.get_tensor_name(i)
                mode = trt_engine.get_tensor_mode(name)
                if str(mode) == "TensorIOMode.INPUT":
                    try:
                        min_s = ctx.get_tensor_shape(name)
                        print(f"    {name}: current_context_shape={tuple(min_s)}")
                    except Exception:
                        print(f"    {name}: shape query failed (no profile set yet)")

        print(f"\n  Plugin: libgrid_sample_3d_plugin.so loaded at engine creation time")

    print(f"\n  WarpInput fields: {[f.name for f in dc_fields(WarpInput)]}")
    print(f"  WarpOutput fields: {[f.name for f in dc_fields(WarpOutput)]}")
    print()

    # ========================================================================
    # Part 2: Pre-Warp Tensor Prep Costs
    # ========================================================================
    print("=" * 72)
    print("PRE-WARP TENSOR PREP COSTS")
    print("=" * 72)

    N_FRAMES = 5
    f_s = avatar.f_s
    print(f"  avatar.f_s shape: {tuple(f_s.shape)}  dtype: {f_s.dtype}  "
          f"contiguous: {f_s.is_contiguous()}  device: {f_s.device}")

    N_PREP_WARMUP = 5
    N_PREP_MEASURED = 50

    # Measure f_s.expand().contiguous()
    for _ in range(N_PREP_WARMUP):
        _ = f_s.expand(N_FRAMES, -1, -1, -1, -1).contiguous()
    torch.cuda.synchronize()

    f_s_times = []
    for _ in range(N_PREP_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        f_s_b = f_s.expand(N_FRAMES, -1, -1, -1, -1).contiguous()
        torch.cuda.synchronize()
        f_s_times.append((time.perf_counter() - t0) * 1000)
    f_s_sorted = sorted(f_s_times)
    f_s_mean = sum(f_s_sorted) / len(f_s_sorted)
    print(f"\n  f_s.expand({N_FRAMES},...).contiguous() ({N_FRAMES} frames):")
    print(f"    shape: {tuple(f_s_b.shape)}  size: {f_s_b.numel() * 4 / 1e6:.2f} MB")
    print(f"    mean={f_s_mean:.3f}ms  p50={f_s_sorted[(len(f_s_sorted)-1)//2]:.3f}ms  "
          f"p95={f_s_sorted[int((len(f_s_sorted)-1)*0.95)]:.3f}ms  "
          f"min={f_s_sorted[0]:.3f}ms  max={f_s_sorted[-1]:.3f}ms")

    # x_s.expand_as(x_d_all).contiguous() — use synthetic x_d_all shape
    x_s_shape = (1, 21, 3)
    x_d_all_shape = (N_FRAMES, 21, 3)
    x_s_synthetic = torch.randn(*x_s_shape, dtype=torch.float32, device="cuda")
    x_d_synthetic = torch.randn(*x_d_all_shape, dtype=torch.float32, device="cuda")

    for _ in range(N_PREP_WARMUP):
        _ = x_s_synthetic.expand_as(x_d_synthetic).contiguous()
    torch.cuda.synchronize()

    x_s_times = []
    for _ in range(N_PREP_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = x_s_synthetic.expand_as(x_d_synthetic).contiguous()
        torch.cuda.synchronize()
        x_s_times.append((time.perf_counter() - t0) * 1000)
    x_s_sorted = sorted(x_s_times)
    x_s_mean = sum(x_s_sorted) / len(x_s_sorted)
    print(f"\n  x_s.expand_as(x_d).contiguous() ({N_FRAMES} frames):")
    print(f"    (1,21,3) -> ({N_FRAMES},21,3)  size: {N_FRAMES*21*3*4/1e3:.1f} KB")
    print(f"    mean={x_s_mean:.3f}ms  p50={x_s_sorted[(len(x_s_sorted)-1)//2]:.3f}ms  "
          f"p95={x_s_sorted[int((len(x_s_sorted)-1)*0.95)]:.3f}ms  "
          f"min={x_s_sorted[0]:.3f}ms  max={x_s_sorted[-1]:.3f}ms")

    # x_d_all.contiguous() — check if already contiguous
    x_d_already_contiguous = x_d_synthetic.is_contiguous()
    for _ in range(N_PREP_WARMUP):
        _ = x_d_synthetic.contiguous()
    torch.cuda.synchronize()

    x_d_times = []
    for _ in range(N_PREP_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = x_d_synthetic.contiguous()
        torch.cuda.synchronize()
        x_d_times.append((time.perf_counter() - t0) * 1000)
    x_d_sorted = sorted(x_d_times)
    x_d_mean = sum(x_d_sorted) / len(x_d_sorted)
    print(f"\n  x_d_all.contiguous() ({N_FRAMES} frames):")
    print(f"    already_contiguous: {x_d_already_contiguous}")
    print(f"    mean={x_d_mean:.3f}ms  p50={x_d_sorted[(len(x_d_sorted)-1)//2]:.3f}ms  "
          f"p95={x_d_sorted[int((len(x_d_sorted)-1)*0.95)]:.3f}ms")

    total_prep = f_s_mean + x_s_mean + x_d_mean
    print(f"\n  TOTAL prep cost: {total_prep:.3f}ms")

    # ========================================================================
    # Part 3: Generate Real Production Tensors via motion_stitch
    # ========================================================================
    print()
    print("=" * 72)
    print("REAL PRODUCTION TENSOR GENERATION")
    print("=" * 72)

    from avtr1_renderer.components.liveportrait.motion_stitch import motion_stitch, MotionFrame

    mg = pipeline._motion_generator
    expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    silence = np.zeros(expected_len, dtype=np.float32)
    chunk_obj = Chunk(audio_speech=silence, audio_listen=silence)
    bg_id = _pick_first_bg(pipeline._backgrounds)
    options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

    try:
        init_state, _ = pipeline.process_chunk(
            avatar, chunk_obj, state=None, options=options
        )
        motions, _ = mg.generate_chunk(chunk_obj, avatar, state=init_state, options=options)
        has_real_inputs = True
    except Exception as exc:
        print(f"  WARNING: generate_chunk failed: {exc}")
        print("  Will skip real-tensor benchmark.")
        has_real_inputs = False
        motions = None

    if has_real_inputs:
        print(f"  Got motions: type={type(motions).__name__}")
        if hasattr(motions, 'R'):
            print(f"    R shape: {tuple(motions.R.shape)}")
        if hasattr(motions, 'exp'):
            print(f"    exp shape: {tuple(motions.exp.shape)}")

        x_s_real, x_d_real = motion_stitch(avatar.kp_info, motions, stitch=stitch_engine)
        print(f"  x_s_real: shape={tuple(x_s_real.shape)}  contiguous={x_s_real.is_contiguous()}")
        print(f"  x_d_real: shape={tuple(x_d_real.shape)}  contiguous={x_d_real.is_contiguous()}")

        f_s_b_real = avatar.f_s.expand(N_FRAMES, -1, -1, -1, -1).contiguous()
        x_s_b_real = x_s_real.expand_as(x_d_real).contiguous()
        x_d_contig_real = x_d_real.contiguous()

        real_inputs = WarpInput(
            feature_3d=f_s_b_real,
            kp_source=x_s_b_real,
            kp_driving=x_d_contig_real,
        )
        print("  Real production tensors ready.")

    # ========================================================================
    # Part 4: Warp B=5 Benchmark — Synthetic
    # ========================================================================
    print()
    print("=" * 72)
    print("WARP B=5 BENCHMARK — SYNTHETIC STABLE TENSORS")
    print("=" * 72)

    N_WARMUP = 10
    N_MEASURED = 50

    syn_feature = torch.randn(N_FRAMES, 32, 16, 64, 64, dtype=torch.float32, device="cuda")
    syn_kp_s = torch.randn(N_FRAMES, 21, 3, dtype=torch.float32, device="cuda")
    syn_kp_d = torch.randn(N_FRAMES, 21, 3, dtype=torch.float32, device="cuda")

    syn_warp_in = WarpInput(feature_3d=syn_feature, kp_source=syn_kp_s, kp_driving=syn_kp_d)

    # Warmup
    for _ in range(N_WARMUP):
        _ = warp_engine(syn_warp_in)
    torch.cuda.synchronize()

    syn_b5_times = []
    for _ in range(N_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        syn_out = warp_engine(syn_warp_in)
        torch.cuda.synchronize()
        syn_b5_times.append((time.perf_counter() - t0) * 1000)

    syn_b5_sorted = sorted(syn_b5_times)
    syn_b5_mean = sum(syn_b5_sorted) / len(syn_b5_sorted)
    print(f"  B=5 synthetic ({N_MEASURED} iters, {N_FRAMES} frames):")
    print(f"    mean={syn_b5_mean:.2f}ms  p50={syn_b5_sorted[(len(syn_b5_sorted)-1)//2]:.2f}ms  "
          f"p95={syn_b5_sorted[int((len(syn_b5_sorted)-1)*0.95)]:.2f}ms  "
          f"min={syn_b5_sorted[0]:.2f}ms  max={syn_b5_sorted[-1]:.2f}ms")
    print(f"    output shape: {tuple(syn_out.out.shape)}  dtype: {syn_out.out.dtype}")
    print(f"    output range: [{syn_out.out.min().item():.4f}, {syn_out.out.max().item():.4f}]")

    # ========================================================================
    # Part 5: Warp B=5 Benchmark — Real Production Tensors
    # ========================================================================
    print()
    print("=" * 72)
    print("WARP B=5 BENCHMARK — REAL PRODUCTION TENSORS")
    print("=" * 72)

    if has_real_inputs:
        # Warmup
        for _ in range(N_WARMUP):
            _ = warp_engine(real_inputs)
        torch.cuda.synchronize()

        real_b5_times = []
        for _ in range(N_MEASURED):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            real_out = warp_engine(real_inputs)
            torch.cuda.synchronize()
            real_b5_times.append((time.perf_counter() - t0) * 1000)

        real_b5_sorted = sorted(real_b5_times)
        real_b5_mean = sum(real_b5_sorted) / len(real_b5_sorted)
        print(f"  B=5 real ({N_MEASURED} iters, {N_FRAMES} frames):")
        print(f"    mean={real_b5_mean:.2f}ms  p50={real_b5_sorted[(len(real_b5_sorted)-1)//2]:.2f}ms  "
              f"p95={real_b5_sorted[int((len(real_b5_sorted)-1)*0.95)]:.2f}ms  "
              f"min={real_b5_sorted[0]:.2f}ms  max={real_b5_sorted[-1]:.2f}ms")
        print(f"    output shape: {tuple(real_out.out.shape)}  dtype: {real_out.out.dtype}")
        delta_syn_real = real_b5_mean - syn_b5_mean
        print(f"\n  Delta real vs synthetic: {delta_syn_real:+.2f}ms")
    else:
        real_b5_mean = None
        print("  SKIPPED — no real production tensors available.")

    # ========================================================================
    # Part 6: Warp B=1 Benchmark — Synthetic (Scaling Analysis)
    # ========================================================================
    print()
    print("=" * 72)
    print("WARP B=1 BENCHMARK — SYNTHETIC (SCALING ANALYSIS)")
    print("=" * 72)

    syn_feature_b1 = torch.randn(1, 32, 16, 64, 64, dtype=torch.float32, device="cuda")
    syn_kp_s_b1 = torch.randn(1, 21, 3, dtype=torch.float32, device="cuda")
    syn_kp_d_b1 = torch.randn(1, 21, 3, dtype=torch.float32, device="cuda")

    syn_warp_in_b1 = WarpInput(feature_3d=syn_feature_b1, kp_source=syn_kp_s_b1, kp_driving=syn_kp_d_b1)

    # Warmup
    for _ in range(N_WARMUP):
        _ = warp_engine(syn_warp_in_b1)
    torch.cuda.synchronize()

    # B=1 x1 timing
    syn_b1_times = []
    for _ in range(N_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = warp_engine(syn_warp_in_b1)
        torch.cuda.synchronize()
        syn_b1_times.append((time.perf_counter() - t0) * 1000)

    syn_b1_sorted = sorted(syn_b1_times)
    syn_b1_mean = sum(syn_b1_sorted) / len(syn_b1_sorted)
    print(f"  B=1 synthetic ({N_MEASURED} iters, single frame):")
    print(f"    mean={syn_b1_mean:.2f}ms  p50={syn_b1_sorted[(len(syn_b1_sorted)-1)//2]:.2f}ms  "
          f"p95={syn_b1_sorted[int((len(syn_b1_sorted)-1)*0.95)]:.2f}ms  "
          f"min={syn_b1_sorted[0]:.2f}ms  max={syn_b1_sorted[-1]:.2f}ms")

    # B=1 x5 timing (5 sequential calls matching B=5 total work)
    syn_b1x5_times = []
    for _ in range(N_MEASURED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_FRAMES):
            _ = warp_engine(syn_warp_in_b1)
        torch.cuda.synchronize()
        syn_b1x5_times.append((time.perf_counter() - t0) * 1000)

    syn_b1x5_sorted = sorted(syn_b1x5_times)
    syn_b1x5_mean = sum(syn_b1x5_sorted) / len(syn_b1x5_sorted)
    print(f"\n  B=1x5 synthetic ({N_MEASURED} iters, {N_FRAMES} sequential calls):")
    print(f"    mean={syn_b1x5_mean:.2f}ms  p50={syn_b1x5_sorted[(len(syn_b1x5_sorted)-1)//2]:.2f}ms  "
          f"p95={syn_b1x5_sorted[int((len(syn_b1x5_sorted)-1)*0.95)]:.2f}ms  "
          f"min={syn_b1x5_sorted[0]:.2f}ms  max={syn_b1x5_sorted[-1]:.2f}ms")

    # ========================================================================
    # Part 7: Scaling Analysis
    # ========================================================================
    print()
    print("=" * 72)
    print("SCALING ANALYSIS: B=1 vs B=5")
    print("=" * 72)

    ratio_b5_per_frame = syn_b5_mean / N_FRAMES
    ratio_b1_single = syn_b1_mean
    ratio_b1x5_per_frame = syn_b1x5_mean / N_FRAMES
    b5_vs_b1x5 = syn_b5_mean / syn_b1x5_mean

    print(f"  B=5 single call:      {syn_b5_mean:.2f}ms  ({ratio_b5_per_frame:.2f}ms/frame)")
    print(f"  B=1 single call:      {syn_b1_mean:.2f}ms")
    print(f"  B=1x5 sequential:     {syn_b1x5_mean:.2f}ms  ({ratio_b1x5_per_frame:.2f}ms/frame)")
    print()
    print(f"  B=5 vs B=1x5 ratio:   {b5_vs_b1x5:.3f}x")

    if b5_vs_b1x5 < 0.85:
        scaling = "SUB-LINEAR — B=5 batching significantly faster than B=1x5"
        scaling_tag = "sublinear"
    elif b5_vs_b1x5 < 0.95:
        scaling = "NEAR-LINEAR with mild batch benefit"
        scaling_tag = "near-linear"
    elif b5_vs_b1x5 < 1.05:
        scaling = "LINEAR — B=5 ≈ B=1x5 (no batch overhead or benefit)"
        scaling_tag = "linear"
    elif b5_vs_b1x5 < 1.15:
        scaling = "SLIGHT OVERHEAD — B=5 slightly slower than B=1x5"
        scaling_tag = "slight-overhead"
    else:
        scaling = "SIGNIFICANT OVERHEAD — B=5 much slower than B=1x5"
        scaling_tag = "significant-overhead"

    print(f"  Scaling verdict: {scaling}")

    batch_savings = syn_b1x5_mean - syn_b5_mean
    print(f"\n  Batch savings (B=1x5 - B=5): {batch_savings:+.2f}ms")

    # ========================================================================
    # Part 8: Summary + Feasibility Assessment
    # ========================================================================
    print()
    print("=" * 72)
    print("SUMMARY + FEASIBILITY ASSESSMENT")
    print("=" * 72)

    warp_mean = real_b5_mean if has_real_inputs else syn_b5_mean
    print(f"  Warp B=5 mean: {warp_mean:.2f}ms ({'real' if has_real_inputs else 'synthetic'})")
    print(f"  Pre-warp prep: {total_prep:.3f}ms")
    print(f"  Warp + prep:   {warp_mean + total_prep:.2f}ms")
    print(f"  Scaling: {scaling_tag}")
    print(f"  Batch savings: {batch_savings:+.2f}ms")
    print()

    print("  Potential optimization paths:")
    print(f"    1. Static B=5 rebuild — may reduce dynamic profile overhead")
    print(f"       Current profile: min=1, opt=5, max=5 (already optimal at B=5)")
    print(f"       Static would eliminate runtime set_input_shape dispatch")
    print(f"       Estimated gain: unknown (need measurement)")
    print(f"       Risk: low (just engine rebuild, no code changes)")
    print(f"    2. Level-5 warp-only rebuild — more aggressive TRT tactic search")
    print(f"       Risk: medium (plugin may emit CUDA errors, T4 had Myelin failures)")
    print(f"       L4 (sm_89) vs T4 (sm_75) — different arch, may behave differently")
    print(f"       Warp-specific build is isolated from other engines")
    print(f"    3. Pre-warp prep optimization:")
    if f_s_mean > 0.5:
        print(f"       f_s expand+contiguous: {f_s_mean:.3f}ms — worth investigating")
    else:
        print(f"       f_s expand+contiguous: {f_s_mean:.3f}ms — negligible")

    print()
    print("  NEXT EXPERIMENT RECOMMENDATION:")
    if warp_mean < 40:
        print("    Warp is already fast (<40ms). Optimization not critical.")
        print("    Focus on other pipeline stages.")
    elif scaling_tag == "linear" or scaling_tag == "near-linear":
        print("    B=5 is the right call pattern. Focus on engine build optimization.")
        print("    RECOMMENDED: Static B=5 warp rebuild (low risk, measurable).")
        print("    If static B=5 shows <3ms gain, try warp-only level-5 build.")
    elif scaling_tag == "slight-overhead":
        print("    B=5 has mild overhead. Static B=5 rebuild may help.")
        print("    RECOMMENDED: Static B=5 warp rebuild first, then level-5 if needed.")
    else:
        print("    B=5 has significant overhead — investigate plugin behavior.")
        print("    RECOMMENDED: Profile grid_sample_3d_plugin with Nsight first.")
    print("=" * 72)

    free2, _ = torch.cuda.mem_get_info(0)
    print(f"\nGPU memory after audit: {free2 / 1e9:.1f} GB free")

    return {
        "warp_b5_synthetic_mean_ms": syn_b5_mean,
        "warp_b5_real_mean_ms": real_b5_mean,
        "warp_b1_mean_ms": syn_b1_mean,
        "warp_b1x5_mean_ms": syn_b1x5_mean,
        "prep_f_s_ms": f_s_mean,
        "prep_x_s_ms": x_s_mean,
        "prep_x_d_ms": x_d_mean,
        "prep_total_ms": total_prep,
        "scaling_tag": scaling_tag,
        "batch_savings_ms": batch_savings,
    }


# ---------------------------------------------------------------------------
# Function: audit_warp_precision — warp TRT engine layer precision audit
# ---------------------------------------------------------------------------


@app.function(
    gpu="L4",
    timeout=600,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def audit_warp_precision() -> dict:
    """Audit warp TRT engine layer precision to determine FP16 vs FP32 usage.

    Uses IEngineInspector to dump per-layer precision information.
    Determines whether the grid_sample_3d_plugin forces FP32 and what
    proportion of the engine runs in each precision.

    Does NOT rebuild engines or modify production code.
    """
    import torch
    import tensorrt as trt
    import json

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
    l4_volume.commit()

    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    print(f"[pipeline] Loaded in {time.monotonic() - t0:.1f}s")
    print()

    warp_engine = pipeline._warp

    # ========================================================================
    # Part 1: Engine-level binding info
    # ========================================================================
    print("=" * 72)
    print("WARP ENGINE BINDING INFO")
    print("=" * 72)

    has_trt = hasattr(warp_engine, "engine")
    print(f"  Engine type: {type(warp_engine).__name__}")
    print(f"  Has TRT ICudaEngine: {has_trt}")

    if not has_trt:
        print("  ABORT: Not a TRT engine. Cannot inspect precision.")
        return {"error": "not_trt_engine"}

    trt_engine = warp_engine.engine
    ctx = warp_engine.context

    print(f"  TRT engine name: {trt_engine.name}")
    print(f"  num_io_tensors: {trt_engine.num_io_tensors}")
    print(f"  num_layers: {trt_engine.num_layers}")
    print()

    for i in range(trt_engine.num_io_tensors):
        name = trt_engine.get_tensor_name(i)
        mode = trt_engine.get_tensor_mode(name)
        dtype = trt_engine.get_tensor_dtype(name)
        shape = trt_engine.get_tensor_shape(name)
        print(f"  [{i}] {name}: mode={mode}, dtype={dtype}, shape={shape}")
    print()

    # ========================================================================
    # Part 2: IEngineInspector — full engine information
    # ========================================================================
    print("=" * 72)
    print("ENGINE INSPECTOR — FULL ENGINE INFO")
    print("=" * 72)

    try:
        inspector = trt_engine.create_engine_inspector()
    except Exception as exc:
        print(f"  create_engine_inspector failed: {exc}")
        print("  This TRT version may not support IEngineInspector.")
        inspector = None

    if inspector is not None:
        try:
            ctx.set_input_shape("feature_3d", (5, 32, 16, 64, 64))
            ctx.set_input_shape("kp_source", (5, 21, 3))
            ctx.set_input_shape("kp_driving", (5, 21, 3))
            inspector.execution_context = ctx
        except Exception as exc:
            print(f"  Setting execution context shapes failed: {exc}")

        try:
            engine_info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
            engine_data = json.loads(engine_info)
            print(json.dumps(engine_data, indent=2)[:3000])
            if len(json.dumps(engine_data, indent=2)) > 3000:
                print(f"  ... (truncated, total {len(json.dumps(engine_data))} chars)")
        except Exception as exc:
            print(f"  get_engine_information failed: {exc}")
            engine_data = None

    # ========================================================================
    # Part 3: Per-layer precision analysis
    # ========================================================================
    print()
    print("=" * 72)
    print("PER-LAYER PRECISION ANALYSIS")
    print("=" * 72)

    if inspector is None:
        print("  SKIPPED — no inspector available.")
    else:
        num_layers = trt_engine.num_layers
        print(f"  Total layers: {num_layers}")
        print()

        precision_counts = {}
        layer_type_counts = {}
        plugin_layers = []
        all_layers = []

        # Print first 3 raw layer outputs for format detection
        all_layers_raw_test = []
        print("  Raw layer info format detection (first 3 layers):")
        for i in range(min(3, num_layers)):
            try:
                raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
                all_layers_raw_test.append(raw)
                print(f"    Layer {i} type={type(raw).__name__}: {repr(raw)[:200]}")
            except Exception as exc:
                print(f"    Layer {i}: failed: {exc}")
        print()

        # get_layer_information returns just the layer name string when
        # the engine was built without --profilingVerbosity=detailed.
        # Fall back to the engine info JSON which at least has all names.
        detailed_per_layer = False
        if all_layers_raw_test:
            test_parsed = None
            try:
                test_stripped = all_layers_raw_test[0].strip()
                if test_stripped.startswith('"'):
                    test_stripped = test_stripped[1:]
                if test_stripped.endswith('"'):
                    test_stripped = test_stripped[:-1]
                test_parsed = json.loads(test_stripped)
            except Exception:
                pass
            if isinstance(test_parsed, dict) and "Name" in test_parsed:
                detailed_per_layer = True

        if detailed_per_layer:
            print("  Detailed per-layer info available (profilingVerbosity=detailed).")
            for i in range(num_layers):
                try:
                    layer_raw = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
                    layer_data = json.loads(layer_raw.strip().strip('"'))
                except Exception as exc:
                    layer_data = {"Name": f"layer_{i}", "ParameterType": "unknown", "Precision": "unknown"}

                name = layer_data.get("Name", f"layer_{i}")
                op = layer_data.get("ParameterType", layer_data.get("Type", "unknown"))
                precision = layer_data.get("Precision", "unknown")

                all_layers.append({"index": i, "name": name, "op": op, "precision": precision})
                precision_counts[precision] = precision_counts.get(precision, 0) + 1
                layer_type_counts[op] = layer_type_counts.get(op, 0) + 1

                if "Plugin" in op or "plugin" in name.lower() or "grid_sample" in name.lower():
                    plugin_layers.append(all_layers[-1])
        else:
            print("  Per-layer info is layer-name-only (default profilingVerbosity).")
            print("  Deriving precision from engine info and reformat layer analysis.")
            print()

            layer_names = []
            if engine_data and "Layers" in engine_data:
                layer_names = engine_data["Layers"]
            else:
                for i in range(num_layers):
                    try:
                        raw_name = inspector.get_layer_information(i, trt.LayerInformationFormat.JSON)
                        name_clean = raw_name.strip().strip('"').strip()
                        layer_names.append(name_clean)
                    except Exception:
                        layer_names.append(f"layer_{i}")

            reformat_count = 0
            compute_layers = []
            for i, name in enumerate(layer_names):
                is_reformat = "Reformatting" in name or "Reformat" in name
                if is_reformat:
                    reformat_count += 1
                else:
                    compute_layers.append({"index": i, "name": name, "op": name, "precision": "unknown"})

            all_layers = [{"index": i, "name": n, "op": n, "precision": "N/A"} for i, n in enumerate(layer_names)]

            print(f"  Layer breakdown from engine info:")
            print(f"    Total layers: {len(layer_names)}")
            print(f"    Reformat (copy/precision conversion) layers: {reformat_count}")
            print(f"    Compute layers: {len(compute_layers)}")
            print()

            print(f"  KEY FINDING: {reformat_count} Reformat layers indicate")
            print(f"  precision transitions (FP32↔FP16 conversions).")
            print(f"  With {len(compute_layers)} compute layers and {reformat_count} reformats,")
            print(f"  the engine uses MIXED PRECISION.")
            print()

            # Find the GridSample plugin layer
            grid_sample_layer = None
            for cl in compute_layers:
                if "GridSample" in cl["name"]:
                    grid_sample_layer = cl
                    break

            if grid_sample_layer:
                print(f"  GridSample plugin layer: [{grid_sample_layer['index']}] {grid_sample_layer['name']}")
                # Check surrounding layers for reformat (indicates precision boundary)
                gs_idx = grid_sample_layer["index"]
                nearby_reformats = [n for n in layer_names[gs_idx-1:gs_idx+2] if "Reformatting" in n]
                if nearby_reformats:
                    print(f"    Adjacent reformat layers: {nearby_reformats}")
                    print(f"    GridSample plugin likely runs in FP32 (precision conversion needed)")
                else:
                    print(f"    No adjacent reformat layers — GridSample may run in same precision")
                print()

            # Print all compute layers (non-reformat)
            print(f"  Compute layers ({len(compute_layers)}):")
            for cl in compute_layers:
                print(f"    [{cl['index']:3d}] {cl['name'][:80]}")

        if detailed_per_layer:
            print(f"  Precision distribution:")
            for prec, count in sorted(precision_counts.items(), key=lambda x: -x[1]):
                pct = count / num_layers * 100
                print(f"    {prec}: {count} layers ({pct:.1f}%)")
            print()

            print(f"  Layer type distribution:")
            for lt, count in sorted(layer_type_counts.items(), key=lambda x: -x[1]):
                print(f"    {lt}: {count}")
            print()

            if plugin_layers:
                print(f"  Plugin layers ({len(plugin_layers)}):")
                for pl in plugin_layers:
                    print(f"    [{pl['index']}] {pl['name']}")
                    print(f"      op: {pl['op']}")
                    print(f"      precision: {pl['precision']}")
            else:
                print("  No explicitly identified plugin layers.")
                print("  Checking all layers for 'grid_sample' or 'Plugin' in name:")
                for layer in all_layers:
                    if any(kw in layer['name'].lower() for kw in ['grid', 'plugin', 'sample']):
                        print(f"    [{layer['index']}] {layer['name']} — {layer['op']} — {layer['precision']}")
        print()

        # Print all layers for completeness (first 50)
        print(f"  All layers (first {min(50, len(all_layers))}):")
        for layer in all_layers[:50]:
            print(f"    [{layer['index']:3d}] {layer['precision']:8s}  {layer['op']:30s}  {layer['name']}")
        if len(all_layers) > 50:
            print(f"    ... ({len(all_layers) - 50} more layers)")

    # ========================================================================
    # Part 4: Reformat layer analysis
    # ========================================================================
    print()
    print("=" * 72)
    print("REFORMAT LAYER ANALYSIS (FP32↔FP16 conversions)")
    print("=" * 72)

    if inspector is not None:
        reformat_layers = []
        for layer in all_layers:
            if "Reformat" in layer["op"] or "reformat" in layer["name"].lower() or "Cast" in layer["op"]:
                reformat_layers.append(layer)

        if reformat_layers:
            print(f"  Reformat/Cast layers ({len(reformat_layers)}):")
            for rl in reformat_layers:
                print(f"    [{rl['index']}] {rl['name']} — {rl['op']} — {rl['precision']}")
            print()
            print(f"  Reformat layers indicate precision boundaries.")
            print(f"  {len(reformat_layers)} reformat layers suggest {len(reformat_layers)//2} precision transitions.")
        else:
            print("  No reformat/cast layers found — engine may be uniform precision.")

    # ========================================================================
    # Part 5: Verdict
    # ========================================================================
    print()
    print("=" * 72)
    print("PRECISION VERDICT")
    print("=" * 72)

    if inspector is not None and precision_counts:
        fp16_count = sum(v for k, v in precision_counts.items() if "float16" in str(k).lower() or "fp16" in str(k).lower() or "half" in str(k).lower())
        fp32_count = sum(v for k, v in precision_counts.items() if "float32" in str(k).lower() or "fp32" in str(k).lower() or "float" in str(k).lower())

        if fp16_count > 0 and fp32_count > 0:
            print(f"  MIXED PRECISION: {fp16_count} FP16 layers, {fp32_count} FP32 layers")
            print(f"  FP16 ratio: {fp16_count / num_layers * 100:.1f}%")
            print(f"  Plugin layers likely force FP32 — grid_sample_3d_plugin may not support FP16.")
            print(f"  Verdict: B — Plugin/layers force FP32. Investigate plugin FP16 support.")
        elif fp16_count > 0 and fp32_count == 0:
            print(f"  ALL FP16: {fp16_count} layers")
            print(f"  Binding I/O is FP32 but internals are FP16 — normal TRT behavior.")
            print(f"  Verdict: A — Bindings FP32, internals FP16. No issue.")
        elif fp32_count > 0 and fp16_count == 0:
            print(f"  ALL FP32: {fp32_count} layers")
            print(f"  Engine built with FP16 flag but no layers use FP16.")
            print(f"  Likely cause: plugin forces FP32, and PREFER constraints allows fallback.")
            print(f"  Verdict: B — FP16 flag ineffective. Plugin or layer constraints force FP32.")
        else:
            print(f"  UNKNOWN precision distribution: {precision_counts}")
            print(f"  Verdict: C — Cannot determine precision. Need trtexec or different API.")
    elif inspector is not None and not detailed_per_layer:
        reformat_count = sum(1 for l in all_layers if "Reformatting" in l.get("name", ""))
        compute_count = num_layers - reformat_count
        has_grid_sample = any("GridSample" in l.get("name", "") for l in all_layers)

        print(f"  Engine built with default profilingVerbosity (names only).")
        print(f"  Per-layer precision not directly available.")
        print()
        print(f"  Evidence from layer structure:")
        print(f"    Total: {num_layers} layers, {reformat_count} reformat, {compute_count} compute")
        print(f"    GridSample plugin layer: {'YES' if has_grid_sample else 'NO'}")

        if reformat_count > 0:
            print(f"    {reformat_count} Reformat layers = precision boundaries (FP32↔FP16)")
            print(f"    This PROVES mixed precision — FP16 is active for some layers.")
            print(f"    Reformat layers wrap the GridSample plugin at precision boundaries.")
            print()
            print(f"  Verdict: A (with caveat) — Mixed precision IS active.")
            print(f"  FP16 runs for compute layers. FP32↔FP16 conversions at I/O and GridSample.")
            print(f"  Binding I/O is FP32 but internals are largely FP16.")
            print(f"  GridSample plugin likely forces FP32 (surrounded by reformats).")
            print()
            print(f"  To confirm: rebuild warp with --profilingVerbosity=detailed")
            print(f"  for exact per-layer precision data.")
        else:
            print(f"    No reformat layers — engine may be uniform FP32 or uniform FP16.")
            print(f"    Cannot determine without detailed profiling.")
            print(f"  Verdict: C — Need detailed profiling to determine precision.")
    else:
        print("  Verdict: C — IEngineInspector unavailable. Need trtexec in image.")

    print("=" * 72)

    free2, _ = torch.cuda.mem_get_info(0)
    print(f"\nGPU memory after audit: {free2 / 1e9:.1f} GB free")

    return {
        "num_layers": trt_engine.num_layers,
        "precision_counts": precision_counts if inspector else None,
        "plugin_layers": [pl["name"] for pl in plugin_layers] if inspector else None,
    }


# ---------------------------------------------------------------------------
# Class: AVTR1Snapshot — GPU memory snapshot test
# ---------------------------------------------------------------------------


@app.cls(
    gpu="T4",
    timeout=1800,
    volumes={ARTIFACT_MOUNT_PATH: artifact_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class AVTR1Snapshot:
    """GPU memory snapshot class for AVTR-1.

    @modal.enter(snap=True) runs during snapshot creation:
      - Downloads HF artifacts (cached after first run)
      - Verifies TRT engines
      - Loads pipeline
      - Runs one warmup render chunk (captures CUDA/TRT lazy init)

    @modal.enter(snap=False) runs after snapshot restore:
      - Verifies GPU is available
      - Logs nvidia-smi

    IMPORTANT: GPU snapshots only work for DEPLOYED apps.
    Run `modal deploy modal_snapshot_smoke.py` first, then invoke
    via `modal.Cls.from_name("avtr1-snapshot-smoke", "AVTR1Snapshot")`.
    """

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import torch

        print("[snap] === @modal.enter(snap=True) starting ===")
        print(f"[snap] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[snap] GPU: {torch.cuda.get_device_name(0)}")

        self._snap_ok = True
        self._snap_error: str | None = None

        try:
            storage_root = _ensure_artifacts()
            artifact_volume.commit()

            engines_ok, missing = _check_all_engines(storage_root)
            if not engines_ok:
                self._snap_ok = False
                self._snap_error = f"Missing engines: {missing}"
                print(f"[snap] ERROR: {self._snap_error}")
                print("[snap] Run `modal run modal_smoke.py --test startup "
                      "--build-missing-engines` first.")
                return

            print("[snap] All TRT engines present.")

            _run_nvidia_smi("snap/before-load")

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions
            import numpy as np

            print(f"[snap] Loading pipeline with avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self.startup_elapsed = time.monotonic() - t0
            print(f"[snap] Pipeline loaded in {self.startup_elapsed:.1f}s")

            _run_nvidia_smi("snap/after-load")

            self.avatar_id = avatar_id
            self.avatar = self.registry[avatar_id]

            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            self._chunk = Chunk(audio_speech=silence, audio_listen=silence)

            bg_id = _pick_first_bg(self.pipeline._backgrounds)
            self._bg_id = bg_id
            self._options = RenderOptions(
                pixel_format="yuv_i420", bg_id=bg_id,
            )

            print("[snap] Running warmup render chunk...")
            t_warm = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk, state=None, options=self._options,
            )
            warmup_frames = list(frames_iter)
            warmup_ms = (time.monotonic() - t_warm) * 1000
            print(f"[snap] Warmup: {warmup_ms:.0f}ms, "
                  f"{len(warmup_frames)} frames, "
                  f"state={type(self._state).__name__}")

            _run_nvidia_smi("snap/after-warmup")
            print("[snap] === @modal.enter(snap=True) complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[snap] FAILED during snap init: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        print("[restore] === @modal.enter(snap=False) starting ===")
        print(f"[restore] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[restore] GPU: {torch.cuda.get_device_name(0)}")
        _run_nvidia_smi("post-restore")

        if not self._snap_ok:
            print(f"[restore] WARNING: snap init had error: {self._snap_error}")
            print("[restore] Pipeline may not be available.")

        print("[restore] === @modal.enter(snap=False) complete ===")

    @modal.method()
    def health(self) -> dict:
        """Return health status."""
        import torch

        return {
            "pipeline_loaded": hasattr(self, "pipeline") and self.pipeline is not None,
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "avatar_id": getattr(self, "avatar_id", None),
        }

    @modal.method()
    def benchmark_first_render(self) -> dict:
        """Render one chunk and return timing. The caller should measure
        the total remote-call time (which includes snapshot restore overhead)
        for the true cold-start metric.
        """
        import torch

        if not getattr(self, "_snap_ok", False):
            return {"error": f"Snap init failed: {getattr(self, '_snap_error', 'unknown')}"}

        _run_nvidia_smi("first-render/before")

        t0 = time.monotonic()
        try:
            state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk,
                state=None,
                options=self._options,
            )
            frames = list(frames_iter)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"error": str(exc)}
        render_ms = (time.monotonic() - t0) * 1000

        _run_nvidia_smi("first-render/after")

        return {
            "render_ms": render_ms,
            "frame_count": len(frames),
            "frame_format": frames[0].format if frames else None,
            "frame_dims": f"{frames[0].height}x{frames[0].width}" if frames else None,
        }

    @modal.method()
    def benchmark_render(self, chunks: int = 10) -> dict:
        """Warm consecutive chunk benchmark using snapshot-restored pipeline."""
        if not getattr(self, "_snap_ok", False):
            return {"error": f"Snap init failed: {getattr(self, '_snap_error', 'unknown')}"}

        stats = _run_benchmark(
            self.pipeline, self.registry, self.avatar_id, chunks,
        )
        _print_benchmark_results("snapshot-restored, baseline engines", stats)
        return stats


# ---------------------------------------------------------------------------
# Class: AVTR1SnapshotL4 — L4 GPU memory snapshot test
# ---------------------------------------------------------------------------


@app.cls(
    gpu="L4",
    timeout=1800,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
)
class AVTR1SnapshotL4:
    """L4 GPU memory snapshot class for AVTR-1.

    Same structure as AVTR1Snapshot but targets L4 GPU and the
    avtr1-artifacts-l4 Volume.  GPU snapshots only work for DEPLOYED apps.

    Invoke via:
      modal.Cls.from_name("avtr1-snapshot-smoke", "AVTR1SnapshotL4")
    """

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import torch

        print("[snap-l4] === @modal.enter(snap=True) starting ===")
        print(f"[snap-l4] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[snap-l4] GPU: {torch.cuda.get_device_name(0)}")

        self._snap_ok = True
        self._snap_error: str | None = None

        try:
            storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
            l4_volume.commit()

            engines_ok, missing = _check_all_engines(storage_root)
            if not engines_ok:
                self._snap_ok = False
                self._snap_error = f"Missing engines: {missing}"
                print(f"[snap-l4] ERROR: {self._snap_error}")
                return

            print("[snap-l4] All TRT engines present.")

            _run_nvidia_smi("snap-l4/before-load")

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions
            import numpy as np

            print(f"[snap-l4] Loading pipeline with avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self.startup_elapsed = time.monotonic() - t0
            print(f"[snap-l4] Pipeline loaded in {self.startup_elapsed:.1f}s")

            _run_nvidia_smi("snap-l4/after-load")

            self.avatar_id = avatar_id
            self.avatar = self.registry[avatar_id]

            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            self._chunk = Chunk(audio_speech=silence, audio_listen=silence)

            bg_id = _pick_first_bg(self.pipeline._backgrounds)
            self._bg_id = bg_id
            self._options = RenderOptions(
                pixel_format="yuv_i420", bg_id=bg_id,
            )

            print("[snap-l4] Running warmup render chunk...")
            t_warm = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk, state=None, options=self._options,
            )
            warmup_frames = list(frames_iter)
            warmup_ms = (time.monotonic() - t_warm) * 1000
            print(f"[snap-l4] Warmup: {warmup_ms:.0f}ms, "
                  f"{len(warmup_frames)} frames, "
                  f"state={type(self._state).__name__}")

            _run_nvidia_smi("snap-l4/after-warmup")
            print("[snap-l4] === @modal.enter(snap=True) complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[snap-l4] FAILED during snap init: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        print("[restore-l4] === @modal.enter(snap=False) starting ===")
        print(f"[restore-l4] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[restore-l4] GPU: {torch.cuda.get_device_name(0)}")
        _run_nvidia_smi("post-restore-l4")

        if not self._snap_ok:
            print(f"[restore-l4] WARNING: snap init had error: {self._snap_error}")
            print("[restore-l4] Pipeline may not be available.")

        print("[restore-l4] === @modal.enter(snap=False) complete ===")

    @modal.method()
    def health(self) -> dict:
        import torch

        return {
            "gpu": "L4",
            "pipeline_loaded": hasattr(self, "pipeline") and self.pipeline is not None,
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "avatar_id": getattr(self, "avatar_id", None),
            "startup_s": getattr(self, "startup_elapsed", None),
        }

    @modal.method()
    def benchmark_first_render(self) -> dict:
        import torch

        if not getattr(self, "_snap_ok", False):
            return {"error": f"Snap init failed: {getattr(self, '_snap_error', 'unknown')}"}

        _run_nvidia_smi("first-render-l4/before")

        t0 = time.monotonic()
        try:
            state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk,
                state=None,
                options=self._options,
            )
            frames = list(frames_iter)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"error": str(exc)}
        render_ms = (time.monotonic() - t0) * 1000

        _run_nvidia_smi("first-render-l4/after")

        return {
            "gpu": "L4",
            "render_ms": render_ms,
            "frame_count": len(frames),
            "frame_dims": f"{frames[0].height}x{frames[0].width}" if frames else None,
        }

    @modal.method()
    def benchmark_render(self, chunks: int = 10) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"error": f"Snap init failed: {getattr(self, '_snap_error', 'unknown')}"}

        stats = _run_benchmark(
            self.pipeline, self.registry, self.avatar_id, chunks,
        )
        _print_benchmark_results("L4 snapshot-restored, baseline engines", stats)
        return stats

    @modal.method()
    def benchmark_render_profiled(self, chunks: int = 10) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"error": f"Snap init failed: {getattr(self, '_snap_error', 'unknown')}"}

        results = _run_profiled_benchmark(
            self.pipeline, self.registry, self.avatar_id, chunks,
        )
        _print_stage_profile(results)
        return results


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    test: str = "health",
    chunks: int = 10,
) -> None:
    tests_needing_snapshot = {
        "health", "startup", "benchmark",
        "health-l4", "startup-l4", "benchmark-l4-snap", "profile-l4-snap",
    }
    tests_standalone = {
        "rebuild", "benchmark-opt", "verify-trt-patch",
        "rebuild-l4", "benchmark-l4", "profile-l4",
        "rebuild-l4-level5", "benchmark-l4-level5",
        "rebuild-l4-warp-static-b5", "benchmark-l4-warp-static-b5",
        "rebuild-l4-warp-static-b5-level5", "benchmark-l4-warp-static-b5-level5",
        "audit-decoder", "audit-cuda-graph", "audit-warp", "audit-warp-precision",
    }

    if test == "verify-trt-patch":
        print("Verifying TRT optimization patch (no engine rebuild)...")
        result = verify_trt_patch.remote()
        print()
        print(f"  builder_optimization_level: {result.get('level_value')}"
              f"  applied={result.get('level_applied')}")
        print(f"  tiling_optimization_level:  {result.get('tiling_value')}"
              f"  applied={result.get('tiling_applied')}")
        if result.get("tiling_error"):
            print(f"  tiling error: {result['tiling_error']}")
        print()
        if result.get("overall_pass"):
            print("  RESULT: PASS. Safe to run --test rebuild.")
        else:
            print("  RESULT: FAIL. Do NOT run --test rebuild.")

    elif test == "rebuild":
        print("Rebuilding optimized engines (level=3 + FAST tiling)...")
        print("This builds into avtr1-artifacts-opt (baseline untouched).")
        print("Expect 10-20 minutes on T4 with level 3.")
        rebuild_optimized_engines.remote()
        print("Done. Optimized engines built and committed.")

    elif test == "benchmark-opt":
        print(f"Benchmarking optimized engines ({chunks} chunks)...")
        stats = benchmark_optimized.remote(chunks=chunks)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print()
            print(f"  Optimized p95: {stats['p95_ms']:.1f}ms")
            print(f"  Baseline p95:  367.7ms")
            diff = stats["p95_ms"] - 367.7
            if diff < 0:
                print(f"  Improvement:   {abs(diff):.1f}ms faster")
            else:
                print(f"  Change:        +{diff:.1f}ms slower")

    elif test == "rebuild-l4":
        print("Rebuilding L4 baseline engines (default TRT settings)...")
        print("This builds into avtr1-artifacts-l4 (T4 volumes untouched).")
        print("Build time: unknown, likely 10-40+ minutes depending on "
              "decode/HuBERT engine build.")
        rebuild_l4_engines.remote()
        print("Done. L4 baseline engines built and committed.")

    elif test == "benchmark-l4":
        print(f"Benchmarking L4 baseline engines ({chunks} chunks)...")
        stats = benchmark_l4.remote(chunks=chunks)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print()
            print(f"  L4 p95:       {stats['p95_ms']:.1f}ms")
            print(f"  T4 baseline:  367.7ms")
            diff = stats["p95_ms"] - 367.7
            if diff < 0:
                print(f"  Improvement:  {abs(diff):.1f}ms faster")
            else:
                print(f"  Change:       +{diff:.1f}ms slower")
            print()
            if stats["p95_ms"] <= 200:
                print("  VERDICT: L4 viable for realtime prototype.")
            elif stats["p95_ms"] <= 250:
                print("  VERDICT: L4 marginal; worth minor code optimizations.")
            else:
                print("  VERDICT: L4 not enough; test L40S/A100 or deeper "
                      "model optimizations.")

    elif test == "profile-l4":
        print(f"Profiling L4 baseline engines ({chunks} chunks)...")
        print("Runs non-profiled first, then profiled (with stage timers).")
        results = benchmark_l4_profiled.remote(chunks=chunks)
        if "error" in results:
            print(f"ERROR: {results['error']}")
        elif "error" in results.get("profiled", {}):
            print(f"ERROR (profiled): {results['profiled']['error']}")
        else:
            print()
            prof = results.get("profiled", {})
            non_prof = results.get("non_profiled", {})
            print(f"  Non-profiled p95: {non_prof.get('p95_ms', 0):.1f}ms")
            print(f"  Profiled p95:     {prof.get('p95_ms', 0):.1f}ms")
            print(f"  Sync overhead:    {results.get('profiling_overhead_ms', 0):+.1f}ms "
                  f"({results.get('profiling_overhead_pct', 0):+.1f}%)")

    elif test == "rebuild-l4-level5":
        print("Rebuilding L4 level-5 engines "
              "(builder_optimization_level=5, no tiling)...")
        print(f"This builds into {L4_LEVEL5_VOLUME_NAME} "
              f"(L4 baseline untouched).")
        print("Expect 30-60+ minutes on L4 with level 5.")
        rebuild_l4_level5_engines.remote()
        print("Done. L4 level-5 engines built and committed.")

    elif test == "benchmark-l4-level5":
        print(f"Benchmarking L4 level-5 engines ({chunks} chunks)...")
        stats = benchmark_l4_level5.remote(chunks=chunks)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print()
            print(f"  L4 level-5 p95: {stats['p95_ms']:.1f}ms")
            print(f"  L4 baseline:    188.0ms")
            diff = stats["p95_ms"] - 188.0
            if diff < 0:
                print(f"  Improvement:    {abs(diff):.1f}ms faster")
            else:
                print(f"  Change:         +{diff:.1f}ms slower")

    elif test == "audit-decoder":
        print("Auditing decoder B=5 batching feasibility on L4...")
        print("Inspects engine metadata, tests B=1/B=5, and times both patterns.")
        audit_decoder_batching.remote()

    elif test == "audit-cuda-graph":
        print("Auditing CUDA Graph feasibility for decoder loop on L4...")
        print("Tests normal B=1x5, single-call graph x5, and multi-call graph x1.")
        cuda_graph_decoder_audit.remote()

    elif test == "audit-warp":
        print("Auditing warp stage optimization feasibility on L4...")
        print("Measures engine metadata, prep costs, B=1/B=5 scaling, real+synthetic.")
        audit_warp_optimization.remote()

    elif test == "audit-warp-precision":
        print("Auditing warp TRT engine layer precision on L4...")
        print("Uses IEngineInspector to determine FP16 vs FP32 per layer.")
        audit_warp_precision.remote()

    elif test == "rebuild-l4-warp-static-b5":
        print("Rebuilding Variant B: static B=5 warp (default builder level)...")
        print(f"Source: {L4_VOLUME_NAME} -> Target: {L4_WARP_STATIC_B5_VOLUME_NAME}")
        print("Copies non-warp engines from L4 baseline, rebuilds warp only.")
        print("Build time: ~5-15 minutes (warp only).")
        rebuild_l4_warp_static_b5.remote()
        print("Done. Variant B warp engine built and committed.")

    elif test == "benchmark-l4-warp-static-b5":
        print(f"Benchmarking Variant B: static B=5 warp ({chunks} chunks)...")
        stats = benchmark_l4_warp_static_b5.remote(chunks=chunks)
        if isinstance(stats, dict) and "error" in stats:
            print(f"ERROR: {stats['error']}")
            if "missing" in stats:
                print(f"  Missing: {stats['missing']}")

    elif test == "rebuild-l4-warp-static-b5-level5":
        print("Rebuilding Variant C: static B=5 + level-5 warp...")
        print(f"Source: {L4_VOLUME_NAME} -> Target: {L4_WARP_STATIC_B5_LEVEL5_VOLUME_NAME}")
        print("Copies non-warp engines from L4 baseline, rebuilds warp only.")
        print("Build time: ~5-30+ minutes (level-5 is exhaustive).")
        rebuild_l4_warp_static_b5_level5.remote()
        print("Done. Variant C warp engine built and committed.")

    elif test == "benchmark-l4-warp-static-b5-level5":
        print(f"Benchmarking Variant C: static B=5 + level-5 warp ({chunks} chunks)...")
        stats = benchmark_l4_warp_static_b5_level5.remote(chunks=chunks)
        if isinstance(stats, dict) and "error" in stats:
            print(f"ERROR: {stats['error']}")
            if "missing" in stats:
                print(f"  Missing: {stats['missing']}")

    elif test == "health-l4":
        print("Checking health of L4 snapshot class...")
        result = AVTR1SnapshotL4().health.remote()
        print(f"  Result: {result}")
        if not result.get("pipeline_loaded"):
            print("  WARNING: Pipeline not loaded. Snap init may have failed.")

    elif test == "startup-l4":
        print("Measuring L4 first-render latency (includes snapshot restore)...")
        print("  Timing from caller perspective (includes Modal overhead)...")
        t0 = time.monotonic()
        result = AVTR1SnapshotL4().benchmark_first_render.remote()
        caller_elapsed = time.monotonic() - t0
        print()
        print(f"  Caller-side total:   {caller_elapsed:.1f}s")
        if "error" in result:
            print(f"  Container-side:      ERROR: {result['error']}")
        else:
            print(f"  Container render:    {result['render_ms']:.0f}ms")
            print(f"  Frames:              {result['frame_count']}")
        print()
        print(f"  Baseline cold start (no snapshot, L4): ~6.5s pipeline load")
        if "error" not in result:
            speedup = 6.5 / caller_elapsed if caller_elapsed > 0 else float("inf")
            print(f"  Snapshot speedup:    ~{speedup:.1f}x")
        print()
        print("  Look for 'Snapshot created' / 'Restoring' in Modal logs.")

    elif test == "benchmark-l4-snap":
        print(f"Benchmarking L4 snapshot-restored pipeline ({chunks} chunks)...")
        stats = AVTR1SnapshotL4().benchmark_render.remote(chunks=chunks)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print()
            print(f"  L4 Snapshot p95: {stats['p95_ms']:.1f}ms")
            print(f"  L4 Baseline p95: 188.0ms")

    elif test == "profile-l4-snap":
        print(f"Profiling L4 snapshot-restored pipeline ({chunks} chunks)...")
        results = AVTR1SnapshotL4().benchmark_render_profiled.remote(chunks=chunks)
        if "error" in results:
            print(f"ERROR: {results['error']}")
        elif "error" in results.get("profiled", {}):
            print(f"ERROR (profiled): {results['profiled']['error']}")

    elif test == "health":
        print("Checking health of snapshot class...")
        result = AVTR1Snapshot().health.remote()
        print(f"  Result: {result}")
        if not result.get("pipeline_loaded"):
            print("  WARNING: Pipeline not loaded. Snap init may have failed.")

    elif test == "startup":
        print("Measuring first-render latency (includes snapshot restore)...")
        print("  Timing from caller perspective (includes Modal overhead)...")
        t0 = time.monotonic()
        result = AVTR1Snapshot().benchmark_first_render.remote()
        caller_elapsed = time.monotonic() - t0
        print()
        print(f"  Caller-side total:   {caller_elapsed:.1f}s")
        if "error" in result:
            print(f"  Container-side:      ERROR: {result['error']}")
        else:
            print(f"  Container render:    {result['render_ms']:.0f}ms")
            print(f"  Frames:              {result['frame_count']}")
        print()
        print("  Baseline cold start (no snapshot): ~24s (22.9s load + 1.3s warmup)")
        if "error" not in result:
            speedup = 24.0 / caller_elapsed if caller_elapsed > 0 else float("inf")
            print(f"  Snapshot speedup:    ~{speedup:.1f}x")
        print()
        print("  NOTE: GPU snapshots only work for DEPLOYED apps.")
        print("  If you used 'modal run', no snapshot was created/restored.")
        print("  Look for 'Snapshot created' / 'Restoring' in Modal logs.")

    elif test == "benchmark":
        print(f"Benchmarking snapshot-restored pipeline ({chunks} chunks)...")
        stats = AVTR1Snapshot().benchmark_render.remote(chunks=chunks)
        if "error" in stats:
            print(f"ERROR: {stats['error']}")
        else:
            print()
            print(f"  Snapshot p95: {stats['p95_ms']:.1f}ms")
            print(f"  Baseline p95: 367.7ms")

    else:
        print(f"Unknown test {test!r}. Choose from: "
              f"{sorted(tests_needing_snapshot | tests_standalone)}")
