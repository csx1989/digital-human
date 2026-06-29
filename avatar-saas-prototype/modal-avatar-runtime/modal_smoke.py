"""
Modal smoke tests for AVTR-1 renderer feasibility on T4.

Answers one question: "Can AVTR-1 load and render one chunk on Modal T4?"

Four functions:
  1. test_imports()            - CPU only, verifies package availability
  2. test_pipeline_startup()   - T4, downloads artifacts, loads pipeline
  3. test_single_chunk_render() - T4, renders one silence chunk
  4. test_render_benchmark()   - T4, warm consecutive chunk benchmark

Usage:
  modal run modal_smoke.py --test imports
  modal run modal_smoke.py --test startup
  modal run modal_smoke.py --test startup --build-missing-engines
  modal run modal_smoke.py --test render
  modal run modal_smoke.py --test render --build-missing-engines
  modal run modal_smoke.py --test benchmark --chunks 10
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Paths resolved relative to the repo root.
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------

ARTIFACT_VOLUME_NAME = "avtr1-artifacts"
ARTIFACT_MOUNT_PATH = "/data/avtr1-artifacts"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

# ---------------------------------------------------------------------------
# Modal image.
# ---------------------------------------------------------------------------

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
    # torch from CUDA index first (must match CUDA 12.8)
    .pip_install(
        "torch>=2.5.1,<2.8",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    # tensorrt from NVIDIA index
    .pip_install(
        "tensorrt<=10.12",
        extra_index_url="https://pypi.nvidia.com",
    )
    # everything else from PyPI
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
    # copy the renderer source and build scripts into the image
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

# ---------------------------------------------------------------------------
# Modal app + volume.
# ---------------------------------------------------------------------------

app = modal.App("avtr1-smoke-test", image=image)

artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME, create_if_missing=True
)


# ---------------------------------------------------------------------------
# Helpers (shared between test_pipeline_startup and test_single_chunk_render).
# ---------------------------------------------------------------------------

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
    """Return (all_present, missing_paths)."""
    missing: list[str] = []
    for rel in RENDERER_ENGINES + AVTR1_ENGINES + [NORMALIZER_FILE]:
        p = storage_root / rel
        if not p.is_file():
            missing.append(rel)
    return len(missing) == 0, missing


def _build_avtr1_engines(storage_root: Path) -> None:
    """Build only the missing TRT engines."""
    missing_set = set()
    _, missing = _check_all_engines(storage_root)
    if not missing:
        print("[build] All engines present, nothing to build.")
        return
    missing_set = set(missing)
    print(f"[build] Missing: {missing}")

    renderer_missing = any(r in missing_set for r in RENDERER_ENGINES)
    hubert_missing = f"{AVTR1_ENGINE_REL}/hubert_lbs_fp16.engine" in missing_set
    avtr1_missing = any(
        f"{AVTR1_ENGINE_REL}/avtr1_{e}_fp16.engine" in missing_set for e in ("encode", "decode")
    )

    if renderer_missing:
        from build_renderer_engines import main as renderer_main
        print("[build] Building renderer TRT engines (decoder, warp, modnet, stitch)...")
        t0 = time.monotonic()
        renderer_main([])
        print(f"[build] Renderer engines built in {time.monotonic() - t0:.0f}s")

    if hubert_missing:
        from build_hubert_engine import main as hubert_main
        print("[build] Building HuBERT TRT engine...")
        t0 = time.monotonic()
        hubert_main([])
        print(f"[build] HuBERT engine built in {time.monotonic() - t0:.0f}s")

    if avtr1_missing:
        from build_avtr1_engines import main as avtr1_main
        print("[build] Building AVTR1 encode/decode TRT engines...")
        t0 = time.monotonic()
        avtr1_main([])
        print(f"[build] AVTR1 engines built in {time.monotonic() - t0:.0f}s")

    if NORMALIZER_FILE in missing_set:
        from build_avtr1_engines import main as avtr1_main
        print("[build] Building normalizer (via avtr1 build script)...")
        avtr1_main([])


def _ensure_artifacts() -> Path:
    """Download HF artifacts if not already present. Returns storage root."""
    from avtr1_renderer.avtr1_artifact_manager import (
        get_artifact_manager,
        get_storage_root,
    )

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
    """Return the first available portrait stem, preferring 'anya_03_studio'."""
    portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
    if not portraits_dir.is_dir():
        portraits_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"No reference_frames directory at {portraits_dir}. "
            f"Run test_pipeline_startup first to download artifacts."
        )
    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(
            f"No portrait PNGs found in {portraits_dir}. "
            f"The HF repo may not contain reference_frames, or auth is needed."
        )
    stems = [p.stem for p in all_pngs]
    if "anya_03_studio" in stems:
        return "anya_03_studio"
    chosen = stems[0]
    print(f"[avatar] 'anya_03_studio' not found; using '{chosen}'")
    return chosen


def _pick_first_bg(backgrounds: dict[str, object]) -> str:
    """Return the first non-transparent background id."""
    for bg_id in sorted(backgrounds):
        if bg_id != "transparent":
            return bg_id
    raise RuntimeError("No non-transparent backgrounds registered")


def _run_nvidia_smi(label: str) -> None:
    """Print nvidia-smi query for GPU memory/utilization. Non-fatal."""
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


# ---------------------------------------------------------------------------
# Test 1: imports (CPU, no GPU).
# ---------------------------------------------------------------------------


@app.function(cpu=2, memory=2048, timeout=120)
def test_imports() -> None:
    """Verify Python version and key package availability."""
    import platform

    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print()

    checks = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("tensorrt", "TensorRT"),
        ("onnxruntime", "ONNX Runtime"),
        ("safetensors", "safetensors"),
        ("cv2", "OpenCV (headless)"),
        ("kornia", "Kornia"),
        ("roma", "Roma"),
        ("huggingface_hub", "HuggingFace Hub"),
        ("einops", "einops"),
    ]

    all_ok = True
    for mod_name, label in checks:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  [OK] {label}: {ver}")
        except Exception as exc:
            print(f"  [FAIL] {label}: {exc}")
            all_ok = False

    print()

    # guarded import of avtr1_renderer (may fail without GPU/CUDA)
    try:
        import avtr1_renderer  # noqa: F401

        print("  [OK] avtr1_renderer: imported successfully")
    except Exception as exc:
        print(f"  [FAIL] avtr1_renderer: {exc}")
        print(
            "         This is expected on CPU if tensorrt requires a GPU device."
        )
        all_ok = False

    print()
    if all_ok:
        print("All imports OK.")
    else:
        print("Some imports FAILED (see above).")


# ---------------------------------------------------------------------------
# Test 2: pipeline startup (T4 GPU).
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={ARTIFACT_MOUNT_PATH: artifact_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_pipeline_startup(build_missing_engines: bool = False) -> None:
    """Download artifacts, optionally build TRT engines, load the pipeline."""
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute capability: sm_{cap[0]}{cap[1]}")
    print()

    # --- ensure HF artifacts are present ---
    storage_root = _ensure_artifacts()
    artifact_volume.commit()

    # --- check AVTR1 TRT engines ---
    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing TRT engine files:")
        for m in missing:
            print(f"  - {m}")
        if build_missing_engines:
            _build_avtr1_engines(storage_root)
            artifact_volume.commit()
            # re-check
            engines_ok, missing = _check_all_engines(storage_root)
            if not engines_ok:
                print(f"[engines] Still missing after build: {missing}")
                print("ERROR: Engine build did not produce expected files.")
                return
        else:
            print()
            print("ERROR: TRT engines not found.")
            return

    print("[engines] All TRT engines present.")
    print()

    _run_nvidia_smi("before-load")

    # --- pick avatar ---
    avatar_id = _pick_first_avatar(storage_root)

    # --- load pipeline ---
    from avtr1_renderer.pipeline import Pipeline

    print(f"[pipeline] Loading pipeline with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    try:
        pipeline, registry = Pipeline.from_artifacts(
            avatar_ids=[avatar_id],
        )
    except Exception as exc:
        print(f"[pipeline] FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return
    elapsed = time.monotonic() - t0

    print(f"[pipeline] Loaded in {elapsed:.1f}s")
    print(f"[pipeline] Avatars: {sorted(registry.keys())}")
    bgs = sorted(
        bg_id for bg_id in pipeline._backgrounds if bg_id != "transparent"
    )
    print(f"[pipeline] Backgrounds ({len(bgs)}): {bgs}")
    print(f"[pipeline] Motion generator: {type(pipeline._motion_generator).__name__}")
    mg = pipeline._motion_generator
    print(f"  chunk_size={mg.chunk_size}, past_size={mg.past_size}, "
          f"future_size={mg.future_size}, nfeats={mg.nfeats}")
    print(f"  frame_len={mg.frame_len}, audio_shift={mg.audio_shift}, "
          f"n_ode_steps={mg.n_ode_steps}")

    # --- CUDA memory ---
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print()
        print(f"[gpu] torch.cuda: {alloc:.2f} GiB allocated, "
              f"{reserved:.2f} GiB reserved, "
              f"{total:.1f} GiB total")

    _run_nvidia_smi("after-load")

    print()
    print("Pipeline startup PASSED.")


# ---------------------------------------------------------------------------
# Test 3: single chunk render (T4 GPU).
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={ARTIFACT_MOUNT_PATH: artifact_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_single_chunk_render(build_missing_engines: bool = False) -> None:
    """Load pipeline and render one silence chunk."""
    import numpy as np
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- ensure artifacts + engines ---
    storage_root = _ensure_artifacts()
    artifact_volume.commit()

    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing TRT engine files:")
        for m in missing:
            print(f"  - {m}")
        if build_missing_engines:
            _build_avtr1_engines(storage_root)
            artifact_volume.commit()
            engines_ok, missing = _check_all_engines(storage_root)
            if not engines_ok:
                print(f"[engines] Still missing after build: {missing}")
                print("ERROR: Engine build did not produce expected files.")
                return
        else:
            print()
            print("ERROR: TRT engines not found.")
            return

    print("[engines] All TRT engines present.")
    print()

    _run_nvidia_smi("before-load")

    # --- load pipeline ---
    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, RenderOptions

    print(f"[pipeline] Loading pipeline with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    try:
        pipeline, registry = Pipeline.from_artifacts(
            avatar_ids=[avatar_id],
        )
    except Exception as exc:
        print(f"[pipeline] FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    avatar = registry[avatar_id]

    # --- build silence chunk ---
    # Expected length: (chunk_size + future_size) * frame_len + audio_shift
    # = (5 + 5) * 640 + 80 = 6480 samples
    mg = pipeline._motion_generator
    expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    silence = np.zeros(expected_len, dtype=np.float32)
    chunk = Chunk(audio_speech=silence, audio_listen=silence)
    print(f"[chunk] Audio length: {len(silence)} samples "
          f"({len(silence) / 16000 * 1000:.0f}ms at 16kHz)")

    # --- pick background ---
    bg_id = _pick_first_bg(pipeline._backgrounds)
    print(f"[render] Using bg_id={bg_id!r}")

    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id=bg_id,
    )

    # --- render ---
    print("[render] Running process_chunk (state=None, cold start)...")
    t0 = time.monotonic()
    try:
        next_state, frames_iter = pipeline.process_chunk(
            avatar, chunk, state=None, options=options
        )
    except Exception as exc:
        print(f"[render] FAILED: {exc}")
        import traceback

        traceback.print_exc()
        return

    # consume all frames
    frame_list = list(frames_iter)
    render_elapsed = time.monotonic() - t0

    _run_nvidia_smi("after-render")

    # --- results ---
    print()
    print("=" * 60)
    print("RENDER RESULTS")
    print("=" * 60)
    print(f"  Pipeline startup: {startup_elapsed:.1f}s")
    print(f"  Render time:      {render_elapsed * 1000:.0f}ms")
    print(f"  Frame count:      {len(frame_list)}")
    if frame_list:
        f0 = frame_list[0]
        print(f"  Frame format:     {f0.format}")
        print(f"  Frame height:     {f0.height}")
        print(f"  Frame width:      {f0.width}")
        print(f"  Frame data shape: {f0.data.shape}")
        print(f"  Frame data dtype: {f0.data.dtype}")
    print(f"  State type:       {type(next_state).__name__}")

    # --- CUDA memory ---
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  VRAM allocated:   {alloc:.2f} GiB")
        print(f"  VRAM reserved:    {reserved:.2f} GiB")
        print(f"  VRAM total:       {total:.1f} GiB")
    print("=" * 60)

    # --- success criteria ---
    print()
    ok = True
    if len(frame_list) != 5:
        print(f"FAIL: Expected 5 frames, got {len(frame_list)}")
        ok = False
    if frame_list and frame_list[0].height != 720:
        print(f"FAIL: Expected height 720, got {frame_list[0].height}")
        ok = False
    if frame_list and frame_list[0].width != 1280:
        print(f"FAIL: Expected width 1280, got {frame_list[0].width}")
        ok = False
    if ok:
        print("SUCCESS: 5 frames, 720x1280, no CUDA/OOM/TRT errors.")
    else:
        print("FAILED: One or more success criteria not met.")


# ---------------------------------------------------------------------------
# Test 4: render benchmark (T4 GPU, warm consecutive chunks).
# ---------------------------------------------------------------------------


@app.function(
    gpu="T4",
    timeout=1800,
    volumes={ARTIFACT_MOUNT_PATH: artifact_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
)
def test_render_benchmark(
    build_missing_engines: bool = False,
    chunks: int = 10,
) -> None:
    """Benchmark warm consecutive rendering on T4.

    Separates *functional* success (all chunks produce 5 frames) from
    *realtime* success (p95 chunk time <= 200ms for 5 frames at 25fps).
    """
    import numpy as np
    import torch

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # --- ensure artifacts + engines ---
    storage_root = _ensure_artifacts()
    artifact_volume.commit()

    engines_ok, missing = _check_all_engines(storage_root)
    if not engines_ok:
        print(f"[engines] Missing TRT engine files:")
        for m in missing:
            print(f"  - {m}")
        if build_missing_engines:
            _build_avtr1_engines(storage_root)
            artifact_volume.commit()
            engines_ok, missing = _check_all_engines(storage_root)
            if not engines_ok:
                print(f"[engines] Still missing after build: {missing}")
                print("ERROR: Engine build did not produce expected files.")
                return
        else:
            print()
            print("ERROR: TRT engines not found.")
            return

    print("[engines] All TRT engine files present.")
    print()

    _run_nvidia_smi("before-load")

    # --- load pipeline ---
    avatar_id = _pick_first_avatar(storage_root)

    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, RenderOptions

    print(f"[pipeline] Loading pipeline with avatar_id={avatar_id!r}...")
    t0 = time.monotonic()
    try:
        pipeline, registry = Pipeline.from_artifacts(
            avatar_ids=[avatar_id],
        )
    except Exception as exc:
        print(f"[pipeline] FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return
    startup_elapsed = time.monotonic() - t0
    print(f"[pipeline] Loaded in {startup_elapsed:.1f}s")

    _run_nvidia_smi("after-load")

    avatar = registry[avatar_id]

    # --- build silence chunk ---
    mg = pipeline._motion_generator
    expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
    silence = np.zeros(expected_len, dtype=np.float32)
    chunk = Chunk(audio_speech=silence, audio_listen=silence)
    print(f"[chunk] Audio length: {len(silence)} samples "
          f"({len(silence) / 16000 * 1000:.0f}ms at 16kHz)")

    bg_id = _pick_first_bg(pipeline._backgrounds)
    print(f"[render] Using bg_id={bg_id!r}")
    options = RenderOptions(
        pixel_format="yuv_i420",
        bg_id=bg_id,
    )

    # --- warmup (1 chunk, state=None) ---
    print()
    print("[benchmark] Warmup chunk (state=None)...")
    t_warm = time.monotonic()
    try:
        state, frames_iter = pipeline.process_chunk(
            avatar, chunk, state=None, options=options
        )
        warmup_frames = list(frames_iter)
    except Exception as exc:
        print(f"[benchmark] Warmup FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return
    warmup_ms = (time.monotonic() - t_warm) * 1000
    print(f"[benchmark] Warmup: {warmup_ms:.0f}ms, "
          f"{len(warmup_frames)} frames, "
          f"state={type(state).__name__}")

    _run_nvidia_smi("after-warmup")

    # --- measured chunks ---
    chunk_times_ms: list[float] = []
    chunk_frame_counts: list[int] = []
    all_functional = True

    print()
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

        per_chunk_detail = f"  chunk {i + 1:3d}: {elapsed_ms:7.1f}ms, {len(frames)} frames"
        if i < 3 or i >= chunks - 3:
            print(per_chunk_detail)
        elif i == 3:
            print("  ...")

    _run_nvidia_smi("after-benchmark")

    # --- statistics ---
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"  Pipeline startup:  {startup_elapsed:.1f}s")
    print(f"  Warmup chunk:      {warmup_ms:.0f}ms ({len(warmup_frames)} frames)")
    print(f"  Measured chunks:   {len(chunk_times_ms)}")
    print(f"  Frames per chunk:  {set(chunk_frame_counts)}")

    if not chunk_times_ms:
        print()
        print("ERROR: No chunks completed successfully.")
        return

    sorted_ms = sorted(chunk_times_ms)
    mean_ms = sum(sorted_ms) / len(sorted_ms)
    p50_idx = (len(sorted_ms) - 1) // 2
    p95_idx = int((len(sorted_ms) - 1) * 0.95)
    min_ms = sorted_ms[0]
    max_ms = sorted_ms[-1]
    p50_ms = sorted_ms[p50_idx]
    p95_ms = sorted_ms[p95_idx]

    mean_fps = 5.0 / (mean_ms / 1000.0)

    print()
    print(f"  min:    {min_ms:7.1f}ms")
    print(f"  p50:    {p50_ms:7.1f}ms")
    print(f"  mean:   {mean_ms:7.1f}ms")
    print(f"  p95:    {p95_ms:7.1f}ms")
    print(f"  max:    {max_ms:7.1f}ms")
    print(f"  FPS:    {mean_fps:.1f} (5 frames / mean chunk time)")

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated(0) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  torch.cuda: {alloc:.2f} GiB allocated, "
              f"{reserved:.2f} GiB reserved, "
              f"{total:.1f} GiB total")

    print("=" * 60)

    # --- verdict ---
    print()
    functional_ok = all_functional and all(c == 5 for c in chunk_frame_counts)
    realtime_budget_ms = 200.0
    realtime_ok = p95_ms <= realtime_budget_ms

    if functional_ok:
        print(f"FUNCTIONAL: PASSED ({len(chunk_times_ms)}/{chunks} chunks, "
              f"all produced 5 frames)")
    else:
        print(f"FUNCTIONAL: FAILED (not all chunks produced 5 frames)")

    if realtime_ok:
        print(f"REALTIME:   PASSED (p95={p95_ms:.1f}ms <= {realtime_budget_ms:.0f}ms)")
    else:
        print(f"REALTIME:   FAILED (p95={p95_ms:.1f}ms > {realtime_budget_ms:.0f}ms)")
        if p95_ms <= 500:
            print("  -> T4 marginal. Consider testing L4 (24 GiB, sm89).")
        else:
            print("  -> T4 insufficient. Test L4 or A10G before Phase 4.")

    if functional_ok and realtime_ok:
        print()
        print("VERDICT: T4 viable for realtime prototype. Proceed to Phase 4.")


# ---------------------------------------------------------------------------
# CLI entrypoint.
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    test: str = "imports",
    build_missing_engines: bool = False,
    chunks: int = 10,
) -> None:
    """Run smoke tests.

    Args:
        test: Which test to run: 'imports', 'startup', 'render', or 'benchmark'.
        build_missing_engines: If True, build TRT engines when missing.
        chunks: Number of measured chunks for benchmark (ignored for other tests).
    """
    tests = {
        "imports": test_imports,
        "startup": test_pipeline_startup,
        "render": test_single_chunk_render,
        "benchmark": test_render_benchmark,
    }
    fn = tests.get(test)
    if fn is None:
        print(f"Unknown test {test!r}. Choose from: {sorted(tests)}")
        return

    if test == "imports":
        fn.remote()
    elif test == "benchmark":
        fn.remote(
            build_missing_engines=build_missing_engines,
            chunks=chunks,
        )
    else:
        fn.remote(build_missing_engines=build_missing_engines)
