"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ⚠️  ISOLATED EXPERIMENTAL APP — NOT PRODUCTION                            ║
║  Same-GPU PARALLEL OmniVoice TTS probe — L40S variant.                     ║
║  DO NOT wire into production routing by default.                            ║
║  Gate behind ENABLE_OMNIVOICE_SAME_GPU_PARALLEL_TTS +                      ║
║  OMNIVOICE_SAME_GPU_PARALLEL_TARGET="l40s" in Bridge.                      ║
║  Delete this file when the parallel experiment concludes.                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

Purpose:
  Run AVTR fMP4 A/V timeline + OmniVoice TTS CONCURRENTLY on the same L40S GPU
  using @modal.concurrent(max_inputs=2).  Measures real GPU contention:
  stutter, render slowdown, fMP4 segment gaps — all observed honestly.

  L40S vs L4 vs A10G comparison context:
    - L40S has 48GB VRAM (vs 24GB on L4/A10G).
    - L40S (Ada Lovelace) has newer CUDA cores, higher FP32 throughput,
      and better TensorRT optimization support than L4.
    - L40S should handle AVTR + OmniVoice concurrent with less contention.
    - Therefore L40S parallel performance must be measured empirically.

  Thread A: SSE timeline stream (fMP4 A/V) — held open for session duration.
  Thread B: tts_same_gpu_parallel action — OmniVoice inference on same GPU.

  CUDA serializes GPU kernels at the stream level; both threads contend for
  compute.  No global GPU lock.  No idle pausing.  Contention IS the metric.

  Per-container marker (containerInstanceId) in timeline + TTS responses
  proves both ran in the same Modal container instance.

Image: torch 2.8+cu128 + OmniVoice deps (validated by stress probe).

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_parallel_l40s_app.py

Invoke:
    modal.Cls.from_name("avtr1-lifecycle-parallel-l40s", "AvatarLifecycleL40SParallel")
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import modal

_SENTINEL = object()

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

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

OMNIVOICE_COMMIT = "f7f5040646f4fb8eef84e011adebec0c375ea6ae"
MODEL_ID = "k2-fsa/OmniVoice"

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
        "ffmpeg",
        "git",
        "gcc",
        "g++",
    )
    .pip_install(
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
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
    .pip_install(
        f"git+https://github.com/k2-fsa/OmniVoice.git@{OMNIVOICE_COMMIT}",
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
            "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH,
        }
    )
)

app = modal.App("avtr1-lifecycle-parallel-l40s", image=image)



l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

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
WARP_PLUGIN_REL = "renderer_runtime_artifacts/libgrid_sample_3d_plugin.so"

REQUIRED_ARTIFACT_RELS = (
    RENDERER_ENGINES
    + AVTR1_ENGINES
    + [NORMALIZER_FILE, WARP_PLUGIN_REL]
)


def _ensure_artifacts(mount_path: Path) -> Path:
    from avtr1_renderer.avtr1_artifact_manager import (
        get_artifact_manager,
        get_storage_root,
    )

    os.environ["AVTR1_LOCAL_STORAGE"] = str(mount_path)
    storage_root = get_storage_root()
    print(f"[l40s] Storage root: {storage_root}")
    print("[l40s] Ensuring all HF artifacts are present...")
    t0 = time.monotonic()
    mgr = get_artifact_manager()
    mgr.ensure_all_artifacts(workers=4)
    elapsed = time.monotonic() - t0
    print(f"[l40s] Artifacts ready in {elapsed:.0f}s")
    return storage_root


def _validate_all_artifacts(storage_root: Path) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in REQUIRED_ARTIFACT_RELS:
        p = storage_root / rel
        if not p.is_file():
            missing.append(rel)
    return len(missing) == 0, missing


def _pick_first_avatar(storage_root: Path) -> str:
    portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
    if not portraits_dir.is_dir():
        portraits_dir.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(f"No reference_frames at {portraits_dir}")
    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No portrait PNGs in {portraits_dir}")
    stems = [p.stem for p in all_pngs]
    if "arnold" in stems:
        return "arnold"
    chosen = stems[0]
    print(f"[l40s] Using avatar '{chosen}'")
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
            print(f"[nvidia-smi] {label}: {line}")
    except Exception:
        pass


def _gpu_memory_snapshot(label: str) -> dict:
    snap: dict = {"label": label}
    try:
        import torch
        snap["torchAllocatedMb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        snap["torchPeakMb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    except Exception:
        pass
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                snap["smiMemoryUsedMb"] = int(parts[0])
                snap["smiMemoryTotalMb"] = int(parts[1])
                snap["smiGpuUtilPct"] = int(parts[2])
                snap["smiMemoryUtilPct"] = int(parts[3])
    except Exception:
        pass
    print(f"[gpu-snapshot] {label}: torch_alloc={snap.get('torchAllocatedMb')}MB "
          f"smi_used={snap.get('smiMemoryUsedMb')}MB util={snap.get('smiGpuUtilPct')}%")
    return snap


def _yuv_i420_to_jpeg_b64(
    yuv_data,
    height: int,
    width: int,
    jpeg_quality: int = 85,
) -> str:
    """Convert YUV I420 uint8 array to JPEG base64 string.

    The renderer produces I420 laid out as (3H//2, W) uint8:
      Y plane (H, W), then U+V packed vertically (H/2, W).
    This matches the standard I420 layout that OpenCV expects.
    """
    import base64
    import cv2

    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode JPEG failed")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _yuv_i420_to_jpeg_b64_timed(
    yuv_data,
    height: int,
    width: int,
    jpeg_quality: int = 85,
) -> dict:
    """Same as _yuv_i420_to_jpeg_b64 but returns stage timings.

    Returns dict with:
      dataBase64, encodedBytes, base64Chars,
      convertMs, jpegEncodeMs, base64Ms
    """
    import base64
    import cv2

    t0 = time.monotonic()
    bgr = cv2.cvtColor(yuv_data, cv2.COLOR_YUV2BGR_I420)
    convert_ms = (time.monotonic() - t0) * 1000

    t1 = time.monotonic()
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    success, encoded = cv2.imencode(".jpg", bgr, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode JPEG failed")
    jpeg_ms = (time.monotonic() - t1) * 1000
    encoded_bytes = int(encoded.nbytes)

    t2 = time.monotonic()
    b64_str = base64.b64encode(encoded.tobytes()).decode("ascii")
    b64_ms = (time.monotonic() - t2) * 1000

    return {
        "dataBase64": b64_str,
        "encodedBytes": encoded_bytes,
        "base64Chars": len(b64_str),
        "convertMs": round(convert_ms, 2),
        "jpegEncodeMs": round(jpeg_ms, 2),
        "base64Ms": round(b64_ms, 2),
    }


class _AudioHandoff:
    """Thread-safe per-session audio handoff between TTS thread and timeline generator thread.

    TTS (Thread B) pushes small PCM chunks; timeline stream (Thread A) consumes them.
    Uses closed flag + lock to prevent races with session end.
    """

    def __init__(self, maxsize: int = 300):
        import queue
        import threading

        self.queue: queue.Queue[tuple | None] = queue.Queue(maxsize=maxsize)
        self.closed = False
        self._lock = threading.Lock()

    def push(self, item: tuple, timeout: float = 30.0) -> bool:
        with self._lock:
            if self.closed:
                return False
            try:
                self.queue.put(item, block=True, timeout=timeout)
                return True
            except Exception:
                return False

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.queue.put(None, block=False)
            except Exception:
                pass


@app.cls(
    gpu="L40S",
    timeout=1800,
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    scaledown_window=90,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=2)
class AvatarLifecycleL40SParallel:
    """L40S CPU-only memory snapshot class for PARALLEL AVTR + OmniVoice.

    Uses CPU-only Modal memory snapshots (enable_memory_snapshot=True,
    NO enable_gpu_snapshot). Snapshot lifecycle:
      - snap=True:  CPU-only prep (Python imports, HF artifact download/validation,
                    avatar selection). No CUDA/GPU state.
      - snap=False: GPU init (TRT engine build if needed, Pipeline.from_artifacts,
                    warmup, OmniVoice load, session state reset).

    Uses @modal.concurrent(max_inputs=2) so that:
      - Thread A: SSE timeline stream (fMP4 AV, held open for session)
      - Thread B: tts_same_gpu_parallel (OmniVoice inference)
    run concurrently in the SAME container on the SAME L40S GPU.

    Thread safety:
      - TTS method NEVER touches: _session_state, FFmpeg handles,
        timeline queues, _session_avatar, pipeline process_chunk.
      - TTS ONLY reads: _session_id (for validation), _omnivoice_model,
        _omnivoice_sr, _container_instance_id.
      - _tts_overlap_active is best-effort instrumentation (CPython GIL
        makes bool reads/writes atomic for single-writer/single-reader).
      - NO global GPU lock. NO idle pausing. Contention IS the measurement.

    Deploy:
        modal deploy modal_lifecycle_parallel_l40s_app.py

    Invoke:
        modal.Cls.from_name("avtr1-lifecycle-parallel-l40s", "AvatarLifecycleL40SParallel")
    """

    @modal.enter(snap=True)
    def _cpu_snap_prep(self) -> None:
        print("[l40s-cpu-snap] === @modal.enter(snap=True) CPU-only prep starting ===")

        self._cpu_snapshot_ok = False
        self._snap_error: str | None = None
        self._gpu_artifacts_pending = False
        self._missing_engines_at_cpu_snap: list[str] = []
        self._cpu_snap_prep_s = 0.0

        t_cpu_start = time.monotonic()

        try:
            storage_root = _ensure_artifacts(Path(L40S_MOUNT_PATH))
            l40s_volume.commit()
            self._storage_root = storage_root

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                engine_missing = [m for m in missing if m.endswith(".engine")]
                plugin_missing = [m for m in missing if m.endswith(".so")]
                normalizer_missing = [m for m in missing if "normalizer" in m]
                buildable_missing = engine_missing + plugin_missing + normalizer_missing
                static_missing = [m for m in missing if m not in buildable_missing]

                if static_missing:
                    self._snap_error = f"Missing static HF artifacts: {static_missing}"
                    print(f"[l40s-cpu-snap] ERROR: {self._snap_error}")
                    return

                if buildable_missing:
                    self._gpu_artifacts_pending = True
                    self._missing_engines_at_cpu_snap = buildable_missing
                    print(f"[l40s-cpu-snap] {len(buildable_missing)} buildable artifacts missing "
                          f"(engines={len(engine_missing)}, plugins={len(plugin_missing)}, "
                          f"normalizers={len(normalizer_missing)}). Will build in snap=False with GPU.")
            else:
                print("[l40s-cpu-snap] All artifacts present on volume.")

            self.avatar_id = _pick_first_avatar(storage_root)
            print(f"[l40s-cpu-snap] avatar_id={self.avatar_id!r}")

            self._cpu_snapshot_ok = True
            self._cpu_snap_prep_s = time.monotonic() - t_cpu_start
            print(f"[l40s-cpu-snap] CPU-only prep complete in {self._cpu_snap_prep_s:.1f}s")
            print("[l40s-cpu-snap] === @modal.enter(snap=True) CPU-only prep done ===")

        except Exception as exc:
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[l40s-cpu-snap] FAILED: {exc}")

    @modal.enter(snap=False)
    def _gpu_init_and_restore(self) -> None:
        import torch
        import uuid
        import threading

        print("[l40s-gpu-init] === @modal.enter(snap=False) GPU init + session restore starting ===")

        t_gpu_start = time.monotonic()

        self._snap_ok = False
        self._gpu_init_ok = False
        self._pipeline_loaded = False
        self._omnivoice_loaded = False

        if not getattr(self, "_cpu_snapshot_ok", False):
            self._snap_error = getattr(self, "_snap_error", "CPU snapshot prep failed")
            print(f"[l40s-gpu-init] ABORT: CPU snapshot prep was not ok: {self._snap_error}")
            self._init_session_state()
            return

        cuda_ok = torch.cuda.is_available()
        print(f"[l40s-gpu-init] CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"[l40s-gpu-init] GPU: {torch.cuda.get_device_name(0)}")
        _run_nvidia_smi("gpu-init/start")

        if not cuda_ok:
            self._snap_error = "CUDA not available in snap=False"
            print(f"[l40s-gpu-init] ABORT: {self._snap_error}")
            self._init_session_state()
            return

        try:
            storage_root = getattr(self, "_storage_root", Path(L40S_MOUNT_PATH))

            if getattr(self, "_gpu_artifacts_pending", False):
                print(f"[l40s-gpu-init] Building missing GPU artifacts: "
                      f"{getattr(self, '_missing_engines_at_cpu_snap', [])}")

                ok, missing = _validate_all_artifacts(storage_root)
                if not ok:
                    engine_missing = [m for m in missing if m.endswith(".engine")]
                    if engine_missing:
                        print(f"[l40s-gpu-init] Building {len(engine_missing)} TRT engines on L40S...")
                        print(f"[l40s-gpu-init] Missing: {engine_missing}")
                        t_build = time.monotonic()
                        import build_engines
                        build_engines.main([])
                        build_elapsed = time.monotonic() - t_build
                        print(f"[l40s-gpu-init] Engine build completed in {build_elapsed:.0f}s")
                        l40s_volume.commit()

                    ok, missing = _validate_all_artifacts(storage_root)
                    if not ok:
                        normalizer_missing = [m for m in missing if "normalizer" in m]
                        if normalizer_missing:
                            print(f"[l40s-gpu-init] Building normalizer: {normalizer_missing}")
                            import build_avtr1_engines
                            build_avtr1_engines.main([])
                            l40s_volume.commit()

                    ok, missing = _validate_all_artifacts(storage_root)
                    if not ok:
                        self._snap_error = f"Missing artifacts after GPU build: {missing}"
                        print(f"[l40s-gpu-init] ERROR: {self._snap_error}")
                        self._init_session_state()
                        return

                self._gpu_artifacts_pending = False
                print("[l40s-gpu-init] All GPU artifacts now present.")
            else:
                ok, missing = _validate_all_artifacts(storage_root)
                if not ok:
                    self._snap_error = f"Artifacts missing (no build attempted): {missing}"
                    print(f"[l40s-gpu-init] ERROR: {self._snap_error}")
                    self._init_session_state()
                    return

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions
            import numpy as np

            avatar_id = getattr(self, "avatar_id", None)
            if not avatar_id:
                self._snap_error = "avatar_id not set from CPU snap prep"
                print(f"[l40s-gpu-init] ERROR: {self._snap_error}")
                self._init_session_state()
                return

            print(f"[l40s-gpu-init] Loading pipeline with avatar_id={avatar_id!r}...")
            t_pipeline = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self._pipeline_load_s = time.monotonic() - t_pipeline
            print(f"[l40s-gpu-init] Pipeline loaded in {self._pipeline_load_s:.1f}s")
            _run_nvidia_smi("gpu-init/after-pipeline-load")

            self._pipeline_loaded = True
            self.avatar = self.registry[avatar_id]

            self._bg_id = _pick_first_bg(self.pipeline._backgrounds)
            self._options = RenderOptions(
                pixel_format="yuv_i420",
                bg_id=self._bg_id,
            )

            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            self._chunk = Chunk(audio_speech=silence, audio_listen=silence)

            print("[l40s-gpu-init] Running warmup render chunk...")
            t_warmup = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk, state=None, options=self._options,
            )
            warmup_frames = list(frames_iter)
            self._warmup_s = time.monotonic() - t_warmup
            print(
                f"[l40s-gpu-init] Warmup: {self._warmup_s * 1000:.0f}ms, "
                f"{len(warmup_frames)} frames, "
                f"state={type(self._state).__name__}"
            )
            _run_nvidia_smi("gpu-init/after-warmup")

            print("[l40s-gpu-init] Loading OmniVoice TTS model...")
            t_omni = time.monotonic()
            from omnivoice import OmniVoice

            device = "cuda:0"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16
            self._omnivoice_model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            self._omnivoice_sr = int(getattr(self._omnivoice_model, "sampling_rate", 24000))
            self._omnivoice_device = device
            self._omnivoice_dtype = dtype
            self._omnivoice_load_s = time.monotonic() - t_omni
            print(
                f"[l40s-gpu-init] OmniVoice loaded in {self._omnivoice_load_s:.1f}s, "
                f"sr={self._omnivoice_sr}, dtype={dtype}"
            )
            self._omnivoice_loaded = True
            _run_nvidia_smi("gpu-init/after-omnivoice-load")

            self._gpu_init_ok = True
            self._snap_ok = True
            self._snap_error = None
            self._gpu_init_s = time.monotonic() - t_gpu_start

            self._init_session_state()

            self._total_restore_to_ready_s = time.monotonic() - t_gpu_start
            print(
                f"[l40s-gpu-init] GPU init complete in {self._gpu_init_s:.1f}s "
                f"(pipeline={self._pipeline_load_s:.1f}s, warmup={self._warmup_s:.1f}s, "
                f"omnivoice={self._omnivoice_load_s:.1f}s) "
                f"total_restore_to_ready={self._total_restore_to_ready_s:.1f}s"
            )
            print("[l40s-gpu-init] === @modal.enter(snap=False) GPU init + session restore done ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[l40s-gpu-init] FAILED: {exc}")
            self._init_session_state()

    def _init_session_state(self) -> None:
        import uuid
        import threading

        self._session_active = False
        self._session_id = None

        self._timeline_running = False
        self._timeline_queue = None
        self._timeline_stop_requested = False
        self._timeline_segment_id = 0

        self._container_instance_id = f"{uuid.uuid4().hex[:8]}-{int(time.monotonic()*1000)}"
        print(f"[l40s-session] container_instance_id={self._container_instance_id}")

        self._tts_overlap_active = False
        self._tts_overlap_start_mono = None

        self._audio_handoff_lock = threading.Lock()
        self._audio_handoffs: dict[str, "_AudioHandoff"] = {}

    def _do_health(self) -> dict:
        import torch

        nvidia_smi_raw = None
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                nvidia_smi_raw = result.stdout.strip()
        except Exception:
            pass

        return {
            "gpu": "L40S",
            "app": "avtr1-lifecycle-parallel-l40s",
            "snapshotMode": "cpu_only",
            "gpuSnapshotEnabled": False,
            "cpuSnapshotOk": getattr(self, "_cpu_snapshot_ok", False),
            "gpuInitOk": getattr(self, "_gpu_init_ok", False),
            "pipelineLoaded": getattr(self, "_pipeline_loaded", False),
            "omnivoiceLoaded": getattr(self, "_omnivoice_loaded", False),
            "gpuArtifactsPending": getattr(self, "_gpu_artifacts_pending", False),
            "missingEnginesAtCpuSnap": getattr(self, "_missing_engines_at_cpu_snap", []),
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidiaSmiRaw": nvidia_smi_raw,
            "avatar_id": getattr(self, "avatar_id", None),
            "bg_id": getattr(self, "_bg_id", None),
            "session_active": getattr(self, "_session_active", False),
            "omnivoice_sr": getattr(self, "_omnivoice_sr", None),
            "container_instance_id": getattr(self, "_container_instance_id", None),
            "concurrent_max_inputs": 2,
            "cpuSnapPrep_s": getattr(self, "_cpu_snap_prep_s", None),
            "gpuInit_s": getattr(self, "_gpu_init_s", None),
            "pipelineLoad_s": getattr(self, "_pipeline_load_s", None),
            "warmup_s": getattr(self, "_warmup_s", None),
            "omnivoiceLoad_s": getattr(self, "_omnivoice_load_s", None),
            "totalRestoreToReady_s": getattr(self, "_total_restore_to_ready_s", None),
        }

    def _do_start_session(self, avatar_id: str | None = None, bg_id: str | None = None) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {
                "status": "error",
                "error": f"Pipeline not ready: {getattr(self, '_snap_error', 'unknown')}",
            }

        if getattr(self, "_session_active", False):
            return {
                "status": "error",
                "error": f"Session already active: {self._session_id}",
            }

        import numpy as np
        import uuid
        from avtr1_renderer.types import Chunk

        chosen_avatar = avatar_id or self.avatar_id
        chosen_bg = bg_id or self._bg_id

        if chosen_avatar not in self.registry:
            return {
                "status": "error",
                "error": f"Unknown avatar_id: {chosen_avatar}. Available: {list(self.registry.keys())}",
            }

        avatar = self.registry[chosen_avatar]

        session_id = uuid.uuid4().hex[:12]

        mg = self.pipeline._motion_generator
        expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        silence = np.zeros(expected_len, dtype=np.float32)
        chunk = Chunk(audio_speech=silence, audio_listen=silence)

        print(f"[l40s] Starting session {session_id} with avatar={chosen_avatar}, bg={chosen_bg}")
        t0 = time.monotonic()

        try:
            state, frames_iter = self.pipeline.process_chunk(
                avatar, chunk, state=None, options=self._options,
            )
            frames = list(frames_iter)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": f"Render failed: {exc}"}

        elapsed_ms = (time.monotonic() - t0) * 1000
        print(f"[l40s] Session {session_id} warmup: {elapsed_ms:.0f}ms, {len(frames)} frames")

        self._session_active = True
        self._session_id = session_id
        self._session_state = state
        self._session_avatar = avatar
        self._timeline_queue = modal.Queue.from_name(
            f"avtr1-timeline-{session_id}", create_if_missing=True
        )

        with self._audio_handoff_lock:
            self._audio_handoffs[session_id] = _AudioHandoff(maxsize=300)

        return {
            "sessionId": session_id,
            "status": "avatar_ready",
            "avatarId": chosen_avatar,
            "bgId": chosen_bg,
            "width": 1280,
            "height": 720,
            "fps": 25,
            "framesPerChunk": 5,
            "warmupMs": round(elapsed_ms, 1),
        }

    def _do_render_frame(self, session_id: str | None = None) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"available": False, "note": "Pipeline not ready"}

        if not getattr(self, "_session_active", False):
            return {"available": False, "note": "No active session"}

        if session_id and session_id != self._session_id:
            return {"available": False, "note": f"Session mismatch: got {session_id}, expected {self._session_id}"}

        import numpy as np
        from avtr1_renderer.types import Chunk

        mg = self.pipeline._motion_generator
        expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        silence = np.zeros(expected_len, dtype=np.float32)
        chunk = Chunk(audio_speech=silence, audio_listen=silence)

        try:
            state, frames_iter = self.pipeline.process_chunk(
                self._session_avatar, chunk,
                state=self._session_state,
                options=self._options,
            )
            frames = list(frames_iter)
            self._session_state = state
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"available": False, "note": f"Render failed: {exc}"}

        if not frames:
            return {"available": False, "note": "No frames produced"}

        frame = frames[0]

        try:
            jpeg_b64 = _yuv_i420_to_jpeg_b64(
                frame.data, frame.height, frame.width,
            )
            return {
                "available": True,
                "mimeType": "image/jpeg",
                "dataBase64": jpeg_b64,
                "width": frame.width,
                "height": frame.height,
            }
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {
                "available": False,
                "note": f"JPEG conversion failed: {exc}. Frame was produced ({frame.width}x{frame.height}, format={frame.format}) but could not be converted for browser display.",
            }

    # Future: replace batch render_audio JSON response with Modal StreamingResponse/NDJSON
    # so frames can be forwarded as Modal produces them instead of waiting for full batch.
    # Modal FastAPI endpoints do support StreamingResponse — this is a design choice, not a limitation.
    def _do_render_audio(
        self,
        session_id: str | None = None,
        audio_base64: str | None = None,
        max_duration_s: float = 60.0,
        jpeg_quality: int = 80,
    ) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"available": False, "error": "Pipeline not ready"}

        if not getattr(self, "_session_active", False):
            return {"available": False, "error": "No active session"}

        if session_id and session_id != self._session_id:
            return {"available": False, "error": f"Session mismatch: got {session_id}, expected {self._session_id}"}

        if not audio_base64:
            return {"available": False, "error": "No audio_base64 provided"}

        import base64
        import io
        import numpy as np
        import soundfile as sf
        import soxr
        from avtr1_renderer.types import Chunk

        try:
            wav_bytes = base64.b64decode(audio_base64)
            audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
            audio_f32 = audio_f32.mean(axis=1)

            if sr != 16000:
                print(f"[l40s-render_audio] Resampling {sr}Hz → 16000Hz ({len(audio_f32)} samples)")
                audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                audio_f32 = audio_f32.astype(np.float32)

            total_duration_s = len(audio_f32) / 16000.0
            truncated = total_duration_s > max_duration_s
            if truncated:
                max_samples = int(max_duration_s * 16000)
                audio_f32 = audio_f32[:max_samples]
                print(f"[l40s-render_audio] Truncated {total_duration_s:.2f}s → {max_duration_s:.0f}s")

            audio_duration_ms = len(audio_f32) / 16000.0 * 1000

            mg = self.pipeline._motion_generator
            window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            step = mg.chunk_size * mg.frame_len
            fps = 25.0

            silence = np.zeros(window, dtype=np.float32)

            n_steps = max(1, (len(audio_f32) + step - 1) // step)
            chunks_audio = []
            for i in range(n_steps):
                start = i * step
                piece = audio_f32[start : start + window]
                if len(piece) < window:
                    piece = np.pad(piece, (0, window - len(piece)))
                chunks_audio.append(piece)

            total_frames_expected = n_steps * mg.chunk_size
            print(
                f"[l40s-render_audio] {len(audio_f32)} samples, "
                f"{audio_duration_ms:.0f}ms, "
                f"{n_steps} chunks, "
                f"{total_frames_expected} expected frames"
            )

            handler_start = time.monotonic()

            t0 = time.monotonic()
            state = self._session_state
            all_frames = []

            for i, chunk_audio in enumerate(chunks_audio):
                chunk = Chunk(audio_speech=chunk_audio, audio_listen=silence)
                state, frames_iter = self.pipeline.process_chunk(
                    self._session_avatar, chunk,
                    state=state,
                    options=self._options,
                )
                all_frames.extend(frames_iter)
                if (i + 1) % 10 == 0 or i == n_steps - 1:
                    elapsed = (time.monotonic() - t0) * 1000
                    print(f"[l40s-render_audio] chunk {i + 1}/{n_steps}, {len(all_frames)} frames, {elapsed:.0f}ms")

            self._session_state = state
            model_render_ms = (time.monotonic() - t0) * 1000

            total_convert_ms = 0.0
            total_jpeg_ms = 0.0
            total_b64_ms = 0.0
            total_encoded_bytes = 0
            total_base64_chars = 0

            jpeg_frames = []
            for frame in all_frames:
                timed = _yuv_i420_to_jpeg_b64_timed(
                    frame.data, frame.height, frame.width,
                    jpeg_quality=jpeg_quality,
                )
                total_convert_ms += timed["convertMs"]
                total_jpeg_ms += timed["jpegEncodeMs"]
                total_b64_ms += timed["base64Ms"]
                total_encoded_bytes += timed["encodedBytes"]
                total_base64_chars += timed["base64Chars"]
                jpeg_frames.append({
                    "dataBase64": timed["dataBase64"],
                    "mimeType": "image/jpeg",
                    "width": frame.width,
                    "height": frame.height,
                })

            total_handler_ms = (time.monotonic() - handler_start) * 1000
            overhead_ms = total_handler_ms - model_render_ms - total_convert_ms - total_jpeg_ms - total_b64_ms
            avg_jpeg_ms = total_jpeg_ms / len(jpeg_frames) if jpeg_frames else 0

            result = {
                "available": True,
                "frames": jpeg_frames,
                "audioDurationMs": round(audio_duration_ms, 1),
                "framesRendered": len(jpeg_frames),
                "fps": fps,
                "renderDurationMs": round(model_render_ms, 1),
                "modelRenderDurationMs": round(model_render_ms, 1),
                "jpegEncodeDurationMs": round(total_jpeg_ms, 1),
                "base64EncodeDurationMs": round(total_b64_ms, 1),
                "frameConvertDurationMs": round(total_convert_ms, 1),
                "totalHandlerDurationMs": round(total_handler_ms, 1),
                "overheadMs": round(overhead_ms, 1),
                "avgJpegEncodeMsPerFrame": round(avg_jpeg_ms, 2),
                "encodedBytes": total_encoded_bytes,
                "base64Chars": total_base64_chars,
                "frameWidth": all_frames[0].width if all_frames else 0,
                "frameHeight": all_frames[0].height if all_frames else 0,
                "truncated": truncated,
                "maxDurationS": max_duration_s,
                "originalDurationMs": round(total_duration_s * 1000, 1),
            }
            print(
                f"[l40s-render_audio] timings model={model_render_ms:.0f}ms "
                f"convert={total_convert_ms:.0f}ms jpeg={total_jpeg_ms:.0f}ms "
                f"b64={total_b64_ms:.0f}ms overhead={overhead_ms:.0f}ms "
                f"total={total_handler_ms:.0f}ms frames={len(jpeg_frames)} "
                f"encodedBytes={total_encoded_bytes} base64Chars={total_base64_chars} "
                f"truncated={truncated}"
            )
            return result

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"available": False, "error": f"Render audio failed: {exc}"}

    def _do_render_audio_stream(
        self,
        session_id: str | None = None,
        audio_base64: str | None = None,
        max_duration_s: float = 60.0,
        jpeg_quality: int = 80,
    ):
        import base64
        import io
        import json
        import numpy as np
        import soundfile as sf
        import soxr
        from avtr1_renderer.types import Chunk
        from fastapi.responses import StreamingResponse

        if not getattr(self, "_snap_ok", False):
            err = json.dumps({"type": "error", "error": "Pipeline not ready"}) + "\n"
            yield f"data: {err}\n\n"
            return

        if not getattr(self, "_session_active", False):
            err = json.dumps({"type": "error", "error": "No active session"}) + "\n"
            yield f"data: {err}\n\n"
            return

        if session_id and session_id != self._session_id:
            err = json.dumps({"type": "error", "error": f"Session mismatch: got {session_id}, expected {self._session_id}"}) + "\n"
            yield f"data: {err}\n\n"
            return

        mode = "speech" if audio_base64 else "idle"

        try:
            if audio_base64:
                wav_bytes = base64.b64decode(audio_base64)
                audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                audio_f32 = audio_f32.mean(axis=1)

                if sr != 16000:
                    print(f"[l40s-render_audio_stream] Resampling {sr}Hz -> 16000Hz ({len(audio_f32)} samples)")
                    audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                    audio_f32 = audio_f32.astype(np.float32)

                total_duration_s = len(audio_f32) / 16000.0
                truncated = total_duration_s > max_duration_s
                if truncated:
                    max_samples = int(max_duration_s * 16000)
                    audio_f32 = audio_f32[:max_samples]
                    print(f"[l40s-render_audio_stream] Truncated {total_duration_s:.2f}s -> {max_duration_s:.0f}s")
            else:
                n_samples = int(16000 * max_duration_s)
                audio_f32 = np.zeros(n_samples, dtype=np.float32)
                total_duration_s = max_duration_s
                truncated = False

            audio_duration_ms = len(audio_f32) / 16000.0 * 1000

            mg = self.pipeline._motion_generator
            window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            step = mg.chunk_size * mg.frame_len
            fps = 25.0

            silence = np.zeros(window, dtype=np.float32)

            n_steps = max(1, (len(audio_f32) + step - 1) // step)
            chunks_audio = []
            for i in range(n_steps):
                start = i * step
                piece = audio_f32[start : start + window]
                if len(piece) < window:
                    piece = np.pad(piece, (0, window - len(piece)))
                chunks_audio.append(piece)

            total_frames_expected = n_steps * mg.chunk_size
            print(
                f"[l40s-render_audio_stream] {len(audio_f32)} samples, "
                f"{audio_duration_ms:.0f}ms, "
                f"{n_steps} chunks, "
                f"{total_frames_expected} expected frames"
            )

            handler_start = time.monotonic()

            start_event = {
                "type": "start",
                "mode": mode,
                "sessionId": session_id or self._session_id,
                "fps": fps,
                "originalAudioDurationMs": round(total_duration_s * 1000, 1),
                "renderedAudioDurationMs": round(audio_duration_ms, 1),
                "totalSteps": n_steps,
            }
            yield f"data: {json.dumps(start_event)}\n\n"

            state = self._session_state
            cumulative_frames = 0
            total_encode_ms = 0.0
            first_chunk_ready_ms = 0.0

            for i, chunk_audio in enumerate(chunks_audio):
                render_t0 = time.monotonic()
                chunk_obj = Chunk(audio_speech=chunk_audio, audio_listen=silence)
                state, frames_iter = self.pipeline.process_chunk(
                    self._session_avatar, chunk_obj,
                    state=state,
                    options=self._options,
                )
                frames_list = list(frames_iter)
                render_ms = (time.monotonic() - render_t0) * 1000

                encode_t0 = time.monotonic()
                jpeg_frames = []
                chunk_b64_chars = 0
                for frame in frames_list:
                    timed = _yuv_i420_to_jpeg_b64_timed(
                        frame.data, frame.height, frame.width,
                        jpeg_quality=jpeg_quality,
                    )
                    jpeg_frames.append({
                        "dataBase64": timed["dataBase64"],
                        "mimeType": "image/jpeg",
                        "width": frame.width,
                        "height": frame.height,
                    })
                    chunk_b64_chars += timed["base64Chars"]
                encode_ms = (time.monotonic() - encode_t0) * 1000
                total_encode_ms += encode_ms

                cumulative_frames += len(jpeg_frames)
                is_last = (i == n_steps - 1)

                if first_chunk_ready_ms == 0:
                    first_chunk_ready_ms = (time.monotonic() - handler_start) * 1000

                chunk_event = {
                    "type": "chunk",
                    "mode": mode,
                    "chunkIndex": i,
                    "frames": jpeg_frames,
                    "framesRendered": cumulative_frames,
                    "isLast": is_last,
                    "timing": {
                        "chunkRenderMs": round(render_ms, 1),
                        "chunkEncodeMs": round(encode_ms, 1),
                        "chunkBase64Chars": chunk_b64_chars,
                    },
                }
                yield f"data: {json.dumps(chunk_event)}\n\n"

                if (i + 1) % 10 == 0 or is_last:
                    elapsed = (time.monotonic() - handler_start) * 1000
                    print(
                        f"[l40s-render_audio_stream] chunk {i + 1}/{n_steps}, "
                        f"{cumulative_frames} frames, {elapsed:.0f}ms "
                        f"firstReady={first_chunk_ready_ms:.0f}ms"
                    )

            self._session_state = state
            total_handler_ms = (time.monotonic() - handler_start) * 1000
            model_render_ms = total_handler_ms - total_encode_ms

            end_event = {
                "type": "end",
                "mode": mode,
                "framesRendered": cumulative_frames,
                "totalHandlerDurationMs": round(total_handler_ms, 1),
                "modelRenderDurationMs": round(model_render_ms, 1),
                "totalEncodeMs": round(total_encode_ms, 1),
                "truncated": truncated,
                "originalDurationMs": round(total_duration_s * 1000, 1),
                "renderedAudioDurationMs": round(audio_duration_ms, 1),
                "firstChunkReadyMs": round(first_chunk_ready_ms, 1),
            }
            yield f"data: {json.dumps(end_event)}\n\n"

            print(
                f"[l40s-render_audio_stream] timings model={model_render_ms:.0f}ms "
                f"encode={total_encode_ms:.0f}ms total={total_handler_ms:.0f}ms "
                f"frames={cumulative_frames} firstReady={first_chunk_ready_ms:.0f}ms "
                f"truncated={truncated}"
            )

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"Render audio stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"

    def _do_render_idle(
        self,
        session_id: str | None = None,
        duration_s: float = 3.0,
        jpeg_quality: int = 75,
    ) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"available": False, "error": "Pipeline not ready"}

        if not getattr(self, "_session_active", False):
            return {"available": False, "error": "No active session"}

        if session_id and session_id != self._session_id:
            return {"available": False, "error": f"Session mismatch: got {session_id}, expected {self._session_id}"}

        import numpy as np
        from avtr1_renderer.types import Chunk

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0

        n_samples = int(16000 * duration_s)
        silence_full = np.zeros(n_samples, dtype=np.float32)

        n_steps = max(1, (n_samples + step - 1) // step)
        chunks_audio = []
        for i in range(n_steps):
            start = i * step
            piece = silence_full[start : start + window]
            if len(piece) < window:
                piece = np.pad(piece, (0, window - len(piece)))
            chunks_audio.append(piece)

        total_frames_expected = n_steps * mg.chunk_size
        print(
            f"[l40s-render_idle] {duration_s}s silence, "
            f"{n_steps} chunks, "
            f"{total_frames_expected} expected frames"
        )

        try:
            handler_start = time.monotonic()

            t0 = time.monotonic()
            state = self._session_state
            all_frames = []

            silence_window = np.zeros(window, dtype=np.float32)
            for i, chunk_audio in enumerate(chunks_audio):
                chunk = Chunk(audio_speech=chunk_audio, audio_listen=silence_window)
                state, frames_iter = self.pipeline.process_chunk(
                    self._session_avatar, chunk,
                    state=state,
                    options=self._options,
                )
                all_frames.extend(frames_iter)
                if (i + 1) % 10 == 0 or i == n_steps - 1:
                    elapsed = (time.monotonic() - t0) * 1000
                    print(f"[l40s-render_idle] chunk {i + 1}/{n_steps}, {len(all_frames)} frames, {elapsed:.0f}ms")

            self._session_state = state
            model_render_ms = (time.monotonic() - t0) * 1000

            total_convert_ms = 0.0
            total_jpeg_ms = 0.0
            total_b64_ms = 0.0
            total_encoded_bytes = 0
            total_base64_chars = 0

            jpeg_frames = []
            for frame in all_frames:
                timed = _yuv_i420_to_jpeg_b64_timed(
                    frame.data, frame.height, frame.width,
                    jpeg_quality=jpeg_quality,
                )
                total_convert_ms += timed["convertMs"]
                total_jpeg_ms += timed["jpegEncodeMs"]
                total_b64_ms += timed["base64Ms"]
                total_encoded_bytes += timed["encodedBytes"]
                total_base64_chars += timed["base64Chars"]
                jpeg_frames.append({
                    "dataBase64": timed["dataBase64"],
                    "mimeType": "image/jpeg",
                    "width": frame.width,
                    "height": frame.height,
                })

            total_handler_ms = (time.monotonic() - handler_start) * 1000
            overhead_ms = total_handler_ms - model_render_ms - total_convert_ms - total_jpeg_ms - total_b64_ms
            avg_jpeg_ms = total_jpeg_ms / len(jpeg_frames) if jpeg_frames else 0

            duration_ms = duration_s * 1000
            result = {
                "available": True,
                "frames": jpeg_frames,
                "durationMs": duration_ms,
                "framesRendered": len(jpeg_frames),
                "fps": fps,
                "renderDurationMs": round(model_render_ms, 1),
                "modelRenderDurationMs": round(model_render_ms, 1),
                "jpegEncodeDurationMs": round(total_jpeg_ms, 1),
                "base64EncodeDurationMs": round(total_b64_ms, 1),
                "frameConvertDurationMs": round(total_convert_ms, 1),
                "totalHandlerDurationMs": round(total_handler_ms, 1),
                "overheadMs": round(overhead_ms, 1),
                "avgJpegEncodeMsPerFrame": round(avg_jpeg_ms, 2),
                "encodedBytes": total_encoded_bytes,
                "base64Chars": total_base64_chars,
                "frameWidth": all_frames[0].width if all_frames else 0,
                "frameHeight": all_frames[0].height if all_frames else 0,
            }
            print(
                f"[l40s-render_idle] timings model={model_render_ms:.0f}ms "
                f"convert={total_convert_ms:.0f}ms jpeg={total_jpeg_ms:.0f}ms "
                f"b64={total_b64_ms:.0f}ms overhead={overhead_ms:.0f}ms "
                f"total={total_handler_ms:.0f}ms frames={len(jpeg_frames)} "
                f"encodedBytes={total_encoded_bytes} base64Chars={total_base64_chars}"
            )
            return result

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"available": False, "error": f"Render idle failed: {exc}"}

    def _do_render_idle_chunk(
        self,
        session_id: str | None = None,
        duration_s: float = 0.5,
        jpeg_quality: int = 75,
    ) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"available": False, "error": "Pipeline not ready"}

        if not getattr(self, "_session_active", False):
            return {"available": False, "error": "No active session"}

        if session_id and session_id != self._session_id:
            return {"available": False, "error": f"Session mismatch: got {session_id}, expected {self._session_id}"}

        import numpy as np
        from avtr1_renderer.types import Chunk

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0

        n_samples = int(16000 * duration_s)
        silence_full = np.zeros(n_samples, dtype=np.float32)

        n_steps = max(1, (n_samples + step - 1) // step)
        chunks_audio = []
        for i in range(n_steps):
            start = i * step
            piece = silence_full[start : start + window]
            if len(piece) < window:
                piece = np.pad(piece, (0, window - len(piece)))
            chunks_audio.append(piece)

        try:
            handler_start = time.monotonic()

            t0 = time.monotonic()
            state = self._session_state
            all_frames = []

            silence_window = np.zeros(window, dtype=np.float32)
            for i, chunk_audio in enumerate(chunks_audio):
                chunk = Chunk(audio_speech=chunk_audio, audio_listen=silence_window)
                state, frames_iter = self.pipeline.process_chunk(
                    self._session_avatar, chunk,
                    state=state,
                    options=self._options,
                )
                all_frames.extend(frames_iter)

            self._session_state = state
            model_render_ms = (time.monotonic() - t0) * 1000

            total_convert_ms = 0.0
            total_jpeg_ms = 0.0
            total_b64_ms = 0.0
            total_encoded_bytes = 0
            total_base64_chars = 0

            jpeg_frames = []
            for frame in all_frames:
                timed = _yuv_i420_to_jpeg_b64_timed(
                    frame.data, frame.height, frame.width,
                    jpeg_quality=jpeg_quality,
                )
                total_convert_ms += timed["convertMs"]
                total_jpeg_ms += timed["jpegEncodeMs"]
                total_b64_ms += timed["base64Ms"]
                total_encoded_bytes += timed["encodedBytes"]
                total_base64_chars += timed["base64Chars"]
                jpeg_frames.append({
                    "dataBase64": timed["dataBase64"],
                    "mimeType": "image/jpeg",
                    "width": frame.width,
                    "height": frame.height,
                })

            total_handler_ms = (time.monotonic() - handler_start) * 1000
            overhead_ms = total_handler_ms - model_render_ms - total_convert_ms - total_jpeg_ms - total_b64_ms
            avg_jpeg_ms = total_jpeg_ms / len(jpeg_frames) if jpeg_frames else 0

            duration_ms = duration_s * 1000
            result = {
                "available": True,
                "frames": jpeg_frames,
                "durationMs": duration_ms,
                "framesRendered": len(jpeg_frames),
                "fps": fps,
                "renderDurationMs": round(model_render_ms, 1),
                "modelRenderDurationMs": round(model_render_ms, 1),
                "jpegEncodeDurationMs": round(total_jpeg_ms, 1),
                "base64EncodeDurationMs": round(total_b64_ms, 1),
                "frameConvertDurationMs": round(total_convert_ms, 1),
                "totalHandlerDurationMs": round(total_handler_ms, 1),
                "overheadMs": round(overhead_ms, 1),
                "avgJpegEncodeMsPerFrame": round(avg_jpeg_ms, 2),
                "encodedBytes": total_encoded_bytes,
                "base64Chars": total_base64_chars,
                "frameWidth": all_frames[0].width if all_frames else 0,
                "frameHeight": all_frames[0].height if all_frames else 0,
            }
            print(
                f"[l40s-render_idle_chunk] timings model={model_render_ms:.0f}ms "
                f"convert={total_convert_ms:.0f}ms jpeg={total_jpeg_ms:.0f}ms "
                f"b64={total_b64_ms:.0f}ms overhead={overhead_ms:.0f}ms "
                f"total={total_handler_ms:.0f}ms frames={len(jpeg_frames)} "
                f"encodedBytes={total_encoded_bytes} base64Chars={total_base64_chars}"
            )
            return result

        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {"available": False, "error": f"Render idle chunk failed: {exc}"}

    def _do_end_session(self, session_id: str | None = None) -> dict:
        end_request_mono = time.monotonic()
        ended_id = getattr(self, "_session_id", None)
        was_active = getattr(self, "_session_active", False)
        timeline_running = getattr(self, "_timeline_running", False)
        tts_overlap = getattr(self, "_tts_overlap_active", False)

        if timeline_running:
            print(f"[l40s] end_session while timeline still running "
                  f"(tts_overlap={tts_overlap}), may queue behind stream")

        self._session_active = False
        self._session_id = None
        self._session_state = None
        self._session_avatar = None

        if ended_id:
            with self._audio_handoff_lock:
                handoff = self._audio_handoffs.pop(ended_id, None)
            if handoff is not None:
                handoff.close()
                print(f"[l40s] Audio handoff closed for session {ended_id}")

        end_delay_ms = (time.monotonic() - end_request_mono) * 1000
        print(f"[l40s] Session {ended_id} ended (was_active={was_active}, "
              f"timeline_running={timeline_running}, tts_overlap={tts_overlap}, "
              f"end_delay_ms={end_delay_ms:.1f})")

        return {
            "ok": True,
            "sessionId": ended_id,
            "wasActive": was_active,
            "timelineRunningAtEnd": timeline_running,
            "ttsOverlapAtEnd": tts_overlap,
            "endDelayMs": round(end_delay_ms, 1),
            "containerInstanceId": getattr(self, "_container_instance_id", None),
        }

    def _do_timeline_stream(self, session_id: str | None = None):
        import json
        import numpy as np
        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'Pipeline not ready'})}\n\n"
            return

        if not getattr(self, "_session_active", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'No active session'})}\n\n"
            return

        if session_id and session_id != self._session_id:
            yield f"data: {json.dumps({'type': 'error', 'error': f'Session mismatch: got {session_id}, expected {self._session_id}'})}\n\n"
            return

        self._timeline_running = True
        self._timeline_stop_requested = False
        self._timeline_segment_id = 0

        timeline_queue = self._timeline_queue
        if timeline_queue is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'No timeline queue for session'})}\n\n"
            return

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0
        silence_audio = np.zeros(window, dtype=np.float32)

        EMERGENCY_SPEECH_CAP_S = 300

        start_event = {
            "type": "start",
            "mode": "idle",
            "sessionId": self._session_id,
            "fps": fps,
        }
        yield f"data: {json.dumps(start_event)}\n\n"
        print(f"[l40s-timeline_stream] started session={self._session_id}")

        try:
            idle_chunk_index = 0
            cumulative_chunk_index = 0

            while not self._timeline_stop_requested and getattr(self, "_session_active", False):
                speech_item = timeline_queue.get(block=False)

                if speech_item is not None:
                    if speech_item.get("__stop"):
                        self._timeline_stop_requested = True
                        print("[l40s-timeline_stream] received __stop sentinel")
                        break

                    import base64
                    import io
                    import soundfile as sf
                    import soxr

                    turn_id = speech_item["turnId"]
                    audio_b64 = speech_item["audioBase64"]
                    enqueue_at = speech_item.get("enqueueAt", 0)
                    audio_duration_ms_reported = speech_item.get("audioDurationMs", 0)

                    wav_bytes = base64.b64decode(audio_b64)
                    audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                    audio_f32 = audio_f32.mean(axis=1)

                    if sr != 16000:
                        print(f"[l40s-timeline_stream] Resampling {sr}Hz -> 16000Hz ({len(audio_f32)} samples)")
                        audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                        audio_f32 = audio_f32.astype(np.float32)

                    total_duration_s = len(audio_f32) / 16000.0
                    if total_duration_s > EMERGENCY_SPEECH_CAP_S:
                        print(f"[l40s-timeline_stream] EMERGENCY CAP: truncating {total_duration_s:.1f}s -> {EMERGENCY_SPEECH_CAP_S}s")
                        audio_f32 = audio_f32[:int(EMERGENCY_SPEECH_CAP_S * 16000)]
                        total_duration_s = EMERGENCY_SPEECH_CAP_S

                    audio_duration_ms = len(audio_f32) / 16000.0 * 1000

                    n_steps = max(1, (len(audio_f32) + step - 1) // step)
                    chunks_audio = []
                    for ci in range(n_steps):
                        s = ci * step
                        piece = audio_f32[s : s + window]
                        if len(piece) < window:
                            piece = np.pad(piece, (0, window - len(piece)))
                        chunks_audio.append(piece)

                    total_frames_expected = n_steps * mg.chunk_size
                    print(
                        f"[l40s-timeline_stream] speech turnId={turn_id} "
                        f"{len(audio_f32)} samples, {audio_duration_ms:.0f}ms, "
                        f"{n_steps} chunks, {total_frames_expected} expected frames"
                    )

                    speech_chunk_index = 0
                    first_speech_frame_at = 0.0
                    cumulative_frames = 0

                    for ci, chunk_audio in enumerate(chunks_audio):
                        if self._timeline_stop_requested:
                            break

                        render_t0 = time.monotonic()
                        chunk_obj = Chunk(audio_speech=chunk_audio, audio_listen=silence_audio)
                        self._session_state, frames_iter = self.pipeline.process_chunk(
                            self._session_avatar, chunk_obj,
                            state=self._session_state,
                            options=self._options,
                        )
                        frames_list = list(frames_iter)
                        render_ms = (time.monotonic() - render_t0) * 1000

                        encode_t0 = time.monotonic()
                        jpeg_frames = []
                        chunk_b64_chars = 0
                        for frame in frames_list:
                            timed = _yuv_i420_to_jpeg_b64_timed(
                                frame.data, frame.height, frame.width,
                                jpeg_quality=80,
                            )
                            jpeg_frames.append({
                                "dataBase64": timed["dataBase64"],
                                "mimeType": "image/jpeg",
                                "width": frame.width,
                                "height": frame.height,
                            })
                            chunk_b64_chars += timed["base64Chars"]
                        encode_ms = (time.monotonic() - encode_t0) * 1000

                        cumulative_frames += len(jpeg_frames)

                        if first_speech_frame_at == 0:
                            first_speech_frame_at = time.time()

                        is_speech_start = (ci == 0)
                        is_speech_end = (ci == n_steps - 1)

                        chunk_event = {
                            "type": "frame_chunk",
                            "mode": "speech",
                            "turnId": turn_id,
                            "segmentId": self._timeline_segment_id,
                            "chunkIndex": cumulative_chunk_index,
                            "frames": jpeg_frames,
                            "fps": fps,
                            "isSpeechStart": is_speech_start,
                            "isSpeechEnd": is_speech_end,
                            "timing": {
                                "chunkRenderMs": round(render_ms, 1),
                                "chunkEncodeMs": round(encode_ms, 1),
                                "base64Chars": chunk_b64_chars,
                                "speechRemainingMs": round(max(0, audio_duration_ms - ((ci + 1) * step / 16000.0 * 1000)), 1),
                            },
                        }
                        if is_speech_start and enqueue_at > 0:
                            chunk_event["enqueueToFirstFrameMs"] = round((first_speech_frame_at - enqueue_at) * 1000, 1)

                        yield f"data: {json.dumps(chunk_event)}\n\n"

                        cumulative_chunk_index += 1
                        speech_chunk_index += 1

                    self._timeline_segment_id += 1
                    idle_chunk_index = 0

                    enqueue_to_first = round((first_speech_frame_at - enqueue_at) * 1000, 1) if enqueue_at > 0 and first_speech_frame_at > 0 else None
                    print(
                        f"[l40s-timeline_stream] speech done turnId={turn_id} "
                        f"frames={cumulative_frames} enqueueToFirstFrameMs={enqueue_to_first}"
                    )

                else:
                    render_t0 = time.monotonic()
                    chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
                    self._session_state, frames_iter = self.pipeline.process_chunk(
                        self._session_avatar, chunk_obj,
                        state=self._session_state,
                        options=self._options,
                    )
                    frames_list = list(frames_iter)
                    render_ms = (time.monotonic() - render_t0) * 1000

                    encode_t0 = time.monotonic()
                    jpeg_frames = []
                    chunk_b64_chars = 0
                    for frame in frames_list:
                        timed = _yuv_i420_to_jpeg_b64_timed(
                            frame.data, frame.height, frame.width,
                            jpeg_quality=75,
                        )
                        jpeg_frames.append({
                            "dataBase64": timed["dataBase64"],
                            "mimeType": "image/jpeg",
                            "width": frame.width,
                            "height": frame.height,
                        })
                        chunk_b64_chars += timed["base64Chars"]
                    encode_ms = (time.monotonic() - encode_t0) * 1000

                    chunk_event = {
                        "type": "frame_chunk",
                        "mode": "idle",
                        "turnId": None,
                        "segmentId": self._timeline_segment_id,
                        "chunkIndex": cumulative_chunk_index,
                        "frames": jpeg_frames,
                        "fps": fps,
                        "isSpeechStart": False,
                        "isSpeechEnd": False,
                        "timing": {
                            "chunkRenderMs": round(render_ms, 1),
                            "chunkEncodeMs": round(encode_ms, 1),
                            "base64Chars": chunk_b64_chars,
                            "speechRemainingMs": 0,
                        },
                    }
                    yield f"data: {json.dumps(chunk_event)}\n\n"

                    cumulative_chunk_index += 1
                    idle_chunk_index += 1

            end_event = {
                "type": "end",
                "mode": "idle",
                "chunksRendered": cumulative_chunk_index,
                "stopReason": "requested" if self._timeline_stop_requested else "session_ended",
            }
            yield f"data: {json.dumps(end_event)}\n\n"
            print(f"[l40s-timeline_stream] ended chunks={cumulative_chunk_index} stop_requested={self._timeline_stop_requested}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"Timeline stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"
        finally:
            self._timeline_running = False
            self._timeline_stop_requested = False

    def _do_timeline_stream_fmp4_video(self, session_id: str | None = None):
        import base64
        import io
        import json
        import queue
        import struct
        import threading

        import numpy as np
        import soundfile as sf
        import soxr
        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'Pipeline not ready'})}\n\n"
            return

        if not getattr(self, "_session_active", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'No active session'})}\n\n"
            return

        if session_id and session_id != self._session_id:
            yield f"data: {json.dumps({'type': 'error', 'error': f'Session mismatch: got {session_id}, expected {self._session_id}'})}\n\n"
            return

        self._timeline_running = True
        self._timeline_stop_requested = False
        self._timeline_segment_id = 0

        timeline_queue = self._timeline_queue
        if timeline_queue is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'No timeline queue for session'})}\n\n"
            return

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0
        silence_audio = np.zeros(window, dtype=np.float32)

        W, H = 1280, 720
        FRAME_BYTES = H * W * 3 // 2

        EMERGENCY_SPEECH_CAP_S = 300

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(int(fps)),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(int(fps)), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        ffmpeg_stderr_lines: list[str] = []
        reader_error: list[str] = []

        def _ffmpeg_stderr_reader():
            try:
                while True:
                    line = ffmpeg_proc.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    ffmpeg_stderr_lines.append(decoded)
            except Exception:
                pass

        def _ffmpeg_stdout_reader():
            try:
                buf = bytearray()
                init_collected = bytearray()
                init_sent = False
                media_buf = bytearray()

                while True:
                    chunk = ffmpeg_proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)

                    while len(buf) >= 8:
                        box_size = struct.unpack(">I", bytes(buf[0:4]))[0]
                        box_type = bytes(buf[4:8])

                        if box_size < 8 or len(buf) < box_size:
                            break

                        box_data = bytes(buf[:box_size])
                        buf = buf[box_size:]

                        if not init_sent:
                            init_collected.extend(box_data)
                            if box_type == b"moov":
                                init_sent = True
                                segment_queue.put(("init", bytes(init_collected)))
                                init_collected = bytearray()
                        elif box_type == b"moof":
                            media_buf = bytearray(box_data)
                        elif box_type == b"mdat" and len(media_buf) > 0:
                            media_buf.extend(box_data)
                            segment_queue.put(("media", bytes(media_buf)))
                            media_buf = bytearray()
                        elif box_type == b"mfra":
                            pass

                segment_queue.put(None)
            except Exception as exc:
                reader_error.append(str(exc))
                segment_queue.put(None)

        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stderr_thread.start()

        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        stdout_thread.start()

        start_event = {
            "type": "start",
            "mode": "fmp4_video",
            "sessionId": self._session_id,
            "fps": fps,
        }
        yield f"data: {json.dumps(start_event)}\n\n"
        print(f"[l40s-timeline_fmp4_video] started session={self._session_id} ffmpeg_pid={ffmpeg_proc.pid}")

        stream_start_mono = time.monotonic()
        init_sent_to_client = False
        media_sequence = 0
        total_raw_bytes_written = 0
        total_encoded_bytes_sent = 0
        chunks_rendered = 0
        current_mode = "idle"

        def _drain_segments():
            nonlocal init_sent_to_client, media_sequence, total_encoded_bytes_sent
            drained = []
            while True:
                try:
                    item = segment_queue.get_nowait()
                    if item is None:
                        return drained, True
                    drained.append(item)
                except queue.Empty:
                    break
            return drained, False

        def _yield_segments(segments):
            nonlocal init_sent_to_client, media_sequence, total_encoded_bytes_sent
            events = []
            for seg_type, seg_data in segments:
                if seg_type == "init" and not init_sent_to_client:
                    init_sent_to_client = True
                    t_now = time.monotonic()
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_init",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "timing": {
                            "timeToInitMs": round((t_now - stream_start_mono) * 1000, 1),
                        },
                    }
                    events.append(evt)
                    total_encoded_bytes_sent += len(seg_data)
                    print(f"[l40s-timeline_fmp4_video] init segment {len(seg_data)} bytes, "
                          f"timeToInitMs={evt['timing']['timeToInitMs']}")

                elif seg_type == "media":
                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_media",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "sequence": media_sequence,
                        "mode": current_mode,
                    }
                    events.append(evt)
                    total_encoded_bytes_sent += len(seg_data)
                    media_sequence += 1
            return events

        try:
            idle_chunk_index = 0
            cumulative_chunk_index = 0
            eof_from_reader = False

            while not self._timeline_stop_requested and getattr(self, "_session_active", False) and not eof_from_reader:
                speech_item = timeline_queue.get(block=False)

                if speech_item is not None:
                    if speech_item.get("__stop"):
                        self._timeline_stop_requested = True
                        print("[l40s-timeline_fmp4_video] received __stop sentinel")
                        break

                    turn_id = speech_item["turnId"]
                    audio_b64 = speech_item["audioBase64"]
                    enqueue_at = speech_item.get("enqueueAt", 0)
                    audio_duration_ms_reported = speech_item.get("audioDurationMs", 0)

                    wav_bytes = base64.b64decode(audio_b64)
                    audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                    audio_f32 = audio_f32.mean(axis=1)

                    if sr != 16000:
                        audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                        audio_f32 = audio_f32.astype(np.float32)

                    total_duration_s = len(audio_f32) / 16000.0
                    if total_duration_s > EMERGENCY_SPEECH_CAP_S:
                        audio_f32 = audio_f32[:int(EMERGENCY_SPEECH_CAP_S * 16000)]
                        total_duration_s = EMERGENCY_SPEECH_CAP_S

                    audio_duration_ms = len(audio_f32) / 16000.0 * 1000

                    n_steps = max(1, (len(audio_f32) + step - 1) // step)
                    chunks_audio = []
                    for ci in range(n_steps):
                        s = ci * step
                        piece = audio_f32[s : s + window]
                        if len(piece) < window:
                            piece = np.pad(piece, (0, window - len(piece)))
                        chunks_audio.append(piece)

                    total_frames_expected = n_steps * mg.chunk_size
                    print(
                        f"[l40s-timeline_fmp4_video] speech turnId={turn_id} "
                        f"{len(audio_f32)} samples, {audio_duration_ms:.0f}ms, "
                        f"{n_steps} chunks, {total_frames_expected} expected frames"
                    )

                    current_mode = "speech"

                    speech_first_segment_emitted = False
                    speech_end_emitted = False

                    for ci, chunk_audio in enumerate(chunks_audio):
                        if self._timeline_stop_requested:
                            break

                        render_t0 = time.monotonic()
                        chunk_obj = Chunk(audio_speech=chunk_audio, audio_listen=silence_audio)
                        self._session_state, frames_iter = self.pipeline.process_chunk(
                            self._session_avatar, chunk_obj,
                            state=self._session_state,
                            options=self._options,
                        )
                        frames_list = list(frames_iter)
                        render_ms = (time.monotonic() - render_t0) * 1000

                        encode_t0 = time.monotonic()
                        for frame in frames_list:
                            raw_bytes = frame.data.tobytes()
                            assert len(raw_bytes) == FRAME_BYTES, f"Frame size mismatch: {len(raw_bytes)} != {FRAME_BYTES}"
                            try:
                                ffmpeg_proc.stdin.write(raw_bytes)
                                total_raw_bytes_written += len(raw_bytes)
                            except BrokenPipeError:
                                print(f"[l40s-timeline_fmp4_video] FFmpeg stdin broken pipe")
                                self._timeline_stop_requested = True
                                break
                        try:
                            ffmpeg_proc.stdin.flush()
                        except BrokenPipeError:
                            pass
                        encode_ms = (time.monotonic() - encode_t0) * 1000

                        is_speech_end = (ci == n_steps - 1)

                        segments, eof = _drain_segments()
                        if eof:
                            eof_from_reader = True
                        events = _yield_segments(segments)

                        for evt in events:
                            if evt["type"] == "fmp4_media":
                                evt["mode"] = "speech"
                                evt["turnId"] = turn_id
                                evt["segmentId"] = self._timeline_segment_id
                                evt["chunkIndex"] = cumulative_chunk_index
                                evt["timing"] = {
                                    "chunkRenderMs": round(render_ms, 1),
                                    "feedMs": round(encode_ms, 1),
                                    "framesFed": len(frames_list),
                                }

                                is_first = not speech_first_segment_emitted
                                evt["isSpeechStart"] = is_first
                                if is_first:
                                    speech_first_segment_emitted = True
                                    if enqueue_at > 0:
                                        evt["enqueueToFirstFrameMs"] = round((time.time() - enqueue_at) * 1000, 1)

                                is_end = is_speech_end and not speech_end_emitted
                                evt["isSpeechEnd"] = is_end
                                if is_end:
                                    speech_end_emitted = True

                            yield f"data: {json.dumps(evt)}\n\n"

                        cumulative_chunk_index += 1
                        chunks_rendered += 1

                    if not speech_end_emitted:
                        tail_segments, tail_eof = _drain_segments()
                        if tail_eof:
                            eof_from_reader = True
                        tail_events = _yield_segments(tail_segments)
                        for evt in tail_events:
                            if evt["type"] == "fmp4_media":
                                evt["isSpeechStart"] = False
                                evt["isSpeechEnd"] = True
                                evt["turnId"] = turn_id
                                evt["segmentId"] = self._timeline_segment_id
                                evt["chunkIndex"] = cumulative_chunk_index
                                evt["timing"] = {
                                    "chunkRenderMs": 0,
                                    "feedMs": 0,
                                    "framesFed": 0,
                                }
                            yield f"data: {json.dumps(evt)}\n\n"
                        if tail_events:
                            cumulative_chunk_index += 1
                            chunks_rendered += 1
                        speech_end_emitted = True

                    self._timeline_segment_id += 1
                    idle_chunk_index = 0
                    current_mode = "idle"

                    print(f"[l40s-timeline_fmp4_video] speech done turnId={turn_id}")

                else:
                    current_mode = "idle"

                    render_t0 = time.monotonic()
                    chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
                    self._session_state, frames_iter = self.pipeline.process_chunk(
                        self._session_avatar, chunk_obj,
                        state=self._session_state,
                        options=self._options,
                    )
                    frames_list = list(frames_iter)
                    render_ms = (time.monotonic() - render_t0) * 1000

                    encode_t0 = time.monotonic()
                    for frame in frames_list:
                        raw_bytes = frame.data.tobytes()
                        assert len(raw_bytes) == FRAME_BYTES, f"Frame size mismatch: {len(raw_bytes)} != {FRAME_BYTES}"
                        try:
                            ffmpeg_proc.stdin.write(raw_bytes)
                            total_raw_bytes_written += len(raw_bytes)
                        except BrokenPipeError:
                            print(f"[l40s-timeline_fmp4_video] FFmpeg stdin broken pipe")
                            self._timeline_stop_requested = True
                            break
                    try:
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        pass
                    encode_ms = (time.monotonic() - encode_t0) * 1000

                    segments, eof = _drain_segments()
                    if eof:
                        eof_from_reader = True
                    events = _yield_segments(segments)

                    for evt in events:
                        if evt["type"] == "fmp4_media":
                            evt["mode"] = "idle"
                            evt["isSpeechStart"] = False
                            evt["isSpeechEnd"] = False
                            evt["turnId"] = None
                            evt["segmentId"] = self._timeline_segment_id
                            evt["chunkIndex"] = cumulative_chunk_index
                            evt["timing"] = {
                                "chunkRenderMs": round(render_ms, 1),
                                "feedMs": round(encode_ms, 1),
                                "framesFed": len(frames_list),
                            }
                            yield f"data: {json.dumps(evt)}\n\n"

                        produced_media_s += CHUNK_MEDIA_S
                        cumulative_chunk_index += 1
                    idle_chunk_index += 1
                    chunks_rendered += 1

            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass

            remaining, _ = _drain_segments()
            if remaining:
                events = _yield_segments(remaining)
                for evt in events:
                    yield f"data: {json.dumps(evt)}\n\n"

            ffmpeg_proc.wait(timeout=10)

            stderr_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
            if stderr_errors:
                print(f"[l40s-timeline_fmp4_video] FFmpeg stderr errors: {stderr_errors[:5]}")

            end_event = {
                "type": "end",
                "mode": "fmp4_video",
                "chunksRendered": chunks_rendered,
                "stopReason": "requested" if self._timeline_stop_requested else "session_ended",
                "stats": {
                    "totalRawBytesWritten": total_raw_bytes_written,
                    "totalEncodedBytesSent": total_encoded_bytes_sent,
                    "mediaSegmentsProduced": media_sequence,
                    "ffmpegReturnCode": ffmpeg_proc.returncode,
                    "ffmpegStderrErrors": len(stderr_errors),
                    "readerError": reader_error[0] if reader_error else None,
                },
            }
            yield f"data: {json.dumps(end_event)}\n\n"
            print(f"[l40s-timeline_fmp4_video] ended chunks={chunks_rendered} segments={media_sequence} "
                  f"rawMB={total_raw_bytes_written / 1e6:.1f} encodedMB={total_encoded_bytes_sent / 1e6:.1f}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"fMP4 video stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"
        finally:
            print(f"[l40s-timeline_fmp4_video] cleanup starting ffmpeg_pid={ffmpeg_proc.pid}")

            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass

            try:
                stdout_thread.join(timeout=3)
                if stdout_thread.is_alive():
                    print("[l40s-timeline_fmp4_video] WARNING: stdout_thread still alive after join timeout")
            except Exception:
                pass

            try:
                stderr_thread.join(timeout=3)
                if stderr_thread.is_alive():
                    print("[l40s-timeline_fmp4_video] WARNING: stderr_thread still alive after join timeout")
            except Exception:
                pass

            if ffmpeg_proc.poll() is None:
                print(f"[l40s-timeline_fmp4_video] terminating ffmpeg pid={ffmpeg_proc.pid}")
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            print(f"[l40s-timeline_fmp4_video] ffmpeg exited rc={ffmpeg_proc.returncode}")

            self._timeline_running = False
            self._timeline_stop_requested = False
            print("[l40s-timeline_fmp4_video] cleanup complete")

    def _do_timeline_stream_fmp4_av(self, session_id: str | None = None):
        import base64
        import io
        import json
        import os
        import queue
        import struct
        import threading

        import numpy as np
        import soundfile as sf
        import soxr
        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'Pipeline not ready'})}\n\n"
            return

        if not getattr(self, "_session_active", False):
            yield f"data: {json.dumps({'type': 'error', 'error': 'No active session'})}\n\n"
            return

        if session_id and session_id != self._session_id:
            yield f"data: {json.dumps({'type': 'error', 'error': f'Session mismatch: got {session_id}, expected {self._session_id}'})}\n\n"
            return

        self._timeline_running = True
        self._timeline_stop_requested = False
        self._timeline_segment_id = 0

        timeline_queue = self._timeline_queue
        if timeline_queue is None:
            yield f"data: {json.dumps({'type': 'error', 'error': 'No timeline queue for session'})}\n\n"
            return

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0
        AUDIO_SR = 16000
        silence_audio = np.zeros(window, dtype=np.float32)

        W, H = 1280, 720
        FRAME_BYTES = H * W * 3 // 2

        EMERGENCY_SPEECH_CAP_S = 300
        INIT_TIMEOUT_S = 10

        fifo_path = f"/tmp/avtr1_l40s_audio_{self._session_id}"
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(fifo_path)
        print(f"[l40s-timeline_fmp4_av] FIFO created at {fifo_path}")

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-thread_queue_size", "1024",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(int(fps)),
            "-i", "pipe:0",
            "-thread_queue_size", "1024",
            "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "1",
            "-i", fifo_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(int(fps)), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        print(f"[l40s-timeline_fmp4_av] starting FFmpeg pid pending...")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[l40s-timeline_fmp4_av] FFmpeg started pid={ffmpeg_proc.pid}")

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        audio_write_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)
        video_write_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)
        ffmpeg_stderr_lines: list[str] = []
        reader_error: list[str] = []
        audio_writer_error: list[str] = []
        video_writer_error: list[str] = []
        init_timeout_expired = threading.Event()

        def _ffmpeg_stderr_reader():
            try:
                while True:
                    line = ffmpeg_proc.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    ffmpeg_stderr_lines.append(decoded)
            except Exception:
                pass

        _segment_count = [0]

        def _ffmpeg_stdout_reader():
            try:
                buf = bytearray()
                init_collected = bytearray()
                init_sent = False
                media_buf = bytearray()

                while True:
                    chunk = ffmpeg_proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)

                    while len(buf) >= 8:
                        box_size = struct.unpack(">I", bytes(buf[0:4]))[0]
                        box_type = bytes(buf[4:8])

                        if box_size < 8 or len(buf) < box_size:
                            break

                        box_data = bytes(buf[:box_size])
                        buf = buf[box_size:]

                        if not init_sent:
                            init_collected.extend(box_data)
                            if box_type == b"moov":
                                init_sent = True
                                print(f"[stdout_reader] init complete {len(init_collected)}B")
                                segment_queue.put(("init", bytes(init_collected)))
                                init_collected = bytearray()
                        elif box_type == b"moof":
                            media_buf = bytearray(box_data)
                        elif box_type == b"mdat" and len(media_buf) > 0:
                            media_buf.extend(box_data)
                            _segment_count[0] += 1
                            if _segment_count[0] <= 3 or _segment_count[0] % 100 == 0:
                                print(f"[stdout_reader] media segment #{_segment_count[0]} {len(media_buf)}B")
                            segment_queue.put(("media", bytes(media_buf)))
                            media_buf = bytearray()

                segment_queue.put(None)
                print(f"[stdout_reader] EOF, total segments={_segment_count[0]}")
            except Exception as exc:
                reader_error.append(str(exc))
                print(f"[stdout_reader] exception: {exc}")
                segment_queue.put(None)

        def _audio_writer():
            print(f"[audio_writer] waiting to open FIFO {fifo_path}...")
            if ffmpeg_proc.poll() is not None:
                audio_writer_error.append(
                    f"FFmpeg exited (rc={ffmpeg_proc.returncode}) before audio pipe opened"
                )
                print(f"[audio_writer] FFmpeg already exited rc={ffmpeg_proc.returncode}")
                return
            try:
                with open(fifo_path, "wb") as audio_pipe:
                    print("[audio_writer] Audio FIFO opened successfully")
                    while True:
                        item = audio_write_queue.get()
                        if item is None:
                            print("[audio_writer] Received None sentinel, closing pipe")
                            break
                        try:
                            audio_pipe.write(item)
                            audio_pipe.flush()
                        except BrokenPipeError:
                            print("[audio_writer] Broken pipe writing audio")
                            break
            except OSError as exc:
                audio_writer_error.append(str(exc))
                print(f"[audio_writer] OSError: {exc}")

        def _video_writer():
            print("[video_writer] starting, will write to FFmpeg stdin")
            try:
                while True:
                    item = video_write_queue.get()
                    if item is None:
                        print("[video_writer] Received None sentinel, closing stdin")
                        break
                    try:
                        ffmpeg_proc.stdin.write(item)
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        print("[video_writer] Broken pipe writing video to stdin")
                        break
            except Exception as exc:
                video_writer_error.append(str(exc))
                print(f"[video_writer] exception: {exc}")
            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass
                print("[video_writer] exiting")

        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stderr_thread.start()

        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        stdout_thread.start()

        audio_thread = threading.Thread(target=_audio_writer, daemon=True)
        audio_thread.start()

        video_thread = threading.Thread(target=_video_writer, daemon=True)
        video_thread.start()

        start_event = {
            "type": "start",
            "mode": "fmp4_av",
            "sessionId": self._session_id,
            "fps": fps,
            "audioSampleRate": AUDIO_SR,
            "audioCodec": "aac",
            "videoCodec": "h264_nvenc",
            "samplesPerChunk": step,
            "containerInstanceId": getattr(self, "_container_instance_id", None),
            "omnivoiceLoaded": hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None,
            "concurrentMaxInputs": 2,
        }
        yield f"data: {json.dumps(start_event)}\n\n"
        print(f"[l40s-timeline_fmp4_av] started session={self._session_id} ffmpeg_pid={ffmpeg_proc.pid} "
              f"audio_pipe={fifo_path} step={step} samples")

        stream_start_mono = time.monotonic()
        init_deadline_mono = stream_start_mono + INIT_TIMEOUT_S
        init_sent_to_client = False
        media_sequence = 0
        total_raw_bytes_written = 0
        total_audio_bytes_written = 0
        total_encoded_bytes_sent = 0
        chunks_rendered = 0
        current_mode = "idle"
        init_5s_warned = False
        gpu_snapshots: list[dict] = []
        speech_end_mono: float | None = None
        last_gpu_snapshot_chunk = -25

        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))
        CHUNK_MEDIA_S = 5 / fps
        HANDOFF_WAIT_S = 0.05
        verbose_delivery = os.environ.get("ENABLE_VERBOSE_AV_DELIVERY_TELEMETRY", "").lower() == "true"

        gpu_snapshots.append(_gpu_memory_snapshot("stream_start"))

        import collections

        class _ChunkMeta:
            __slots__ = ("mode", "turn_id", "segment_id", "chunk_index",
                         "is_speech_start", "is_speech_end",
                         "chunk_render_ms", "feed_ms", "frames_fed",
                         "overlap_with_tts", "enqueue_to_first_frame_ms",
                         "audio_samples_per_chunk")

            def __init__(self, **kw):
                for s in self.__slots__:
                    setattr(self, s, kw.get(s))

        pending_chunk_meta: collections.deque[_ChunkMeta] = collections.deque()

        def _push_chunk_meta(**kw):
            pending_chunk_meta.append(_ChunkMeta(**kw))

        class _SegmentSender:
            """Server-side source playout buffer.

            Buffers encoded fMP4 segments from FFmpeg stdout and yields them
            at steady real-time cadence.  Init segment yields immediately.
            During startup, media segments are burst-filled until
            target_lead_s is reached.  After that, segments are yielded at
            ~CHUNK_MEDIA_S intervals.
            """

            def __init__(self):
                self.pending: collections.deque[tuple[str, bytes]] = collections.deque()
                self.produced_media_s: float = 0.0
                self.target_lead_reached: bool = False
                self.last_yield_mono: float = 0.0
                self._inter_yield_gaps: list[float] = []
                self._total_pacing_sleep_ms: float = 0.0
                self._min_queue_depth: int = 999999
                self._max_queue_depth: int = 0
                self._dry_spell_count: int = 0

            def add(self, seg_type: str, seg_data: bytes) -> None:
                self.pending.append((seg_type, seg_data))

            def yield_events(self):
                """Returns list of (event_dict,) ready for SSE."""
                results = []
                now = time.monotonic()

                qd = len(self.pending)
                if qd < self._min_queue_depth:
                    self._min_queue_depth = qd
                if qd > self._max_queue_depth:
                    self._max_queue_depth = qd

                if not self.pending:
                    self._dry_spell_count += 1
                    return results

                while self.pending:
                    seg_type, seg_data = self.pending[0]

                    if seg_type == "init":
                        self.pending.popleft()
                        t_now = time.monotonic()
                        b64 = base64.b64encode(seg_data).decode("ascii")
                        evt = {
                            "type": "fmp4_init",
                            "dataBase64": b64,
                            "byteLength": len(seg_data),
                            "timing": {
                                "timeToInitMs": round((t_now - stream_start_mono) * 1000, 1),
                            },
                        }
                        results.append(evt)
                        self.last_yield_mono = time.monotonic()
                        continue

                    wall_elapsed = time.monotonic() - stream_start_mono
                    media_lead = self.produced_media_s - wall_elapsed

                    if not self.target_lead_reached:
                        self.pending.popleft()
                        self.produced_media_s += CHUNK_MEDIA_S
                        meta = pending_chunk_meta.popleft() if pending_chunk_meta else _ChunkMeta()
                        b64 = base64.b64encode(seg_data).decode("ascii")
                        evt = {
                            "type": "fmp4_media",
                            "dataBase64": b64,
                            "byteLength": len(seg_data),
                            "sequence": media_sequence,
                        }
                        results.append(("media", seg_data, evt, meta))
                        self.last_yield_mono = time.monotonic()

                        wall_after = time.monotonic() - stream_start_mono
                        if self.produced_media_s - wall_after >= TARGET_LEAD_S:
                            self.target_lead_reached = True
                        continue

                    if media_lead > TARGET_LEAD_S + CHUNK_MEDIA_S * 0.5:
                        break

                    desired_interval = CHUNK_MEDIA_S
                    if self.last_yield_mono > 0:
                        elapsed_since = time.monotonic() - self.last_yield_mono
                        if elapsed_since < desired_interval:
                            sleep_s = desired_interval - elapsed_since
                            time.sleep(sleep_s)
                            self._total_pacing_sleep_ms += sleep_s * 1000

                    self.pending.popleft()
                    self.produced_media_s += CHUNK_MEDIA_S
                    meta = pending_chunk_meta.popleft() if pending_chunk_meta else _ChunkMeta()

                    if self.last_yield_mono > 0:
                        gap_ms = (time.monotonic() - self.last_yield_mono) * 1000
                        self._inter_yield_gaps.append(gap_ms)

                    b64 = base64.b64encode(seg_data).decode("ascii")
                    evt = {
                        "type": "fmp4_media",
                        "dataBase64": b64,
                        "byteLength": len(seg_data),
                        "sequence": media_sequence,
                    }
                    results.append(("media", seg_data, evt, meta))
                    self.last_yield_mono = time.monotonic()

                return results

            def drain_all(self):
                results = []
                qd = len(self.pending)
                if qd > self._max_queue_depth:
                    self._max_queue_depth = qd
                if qd < self._min_queue_depth:
                    self._min_queue_depth = qd
                while self.pending:
                    seg_type, seg_data = self.pending.popleft()
                    if seg_type == "init":
                        t_now = time.monotonic()
                        b64 = base64.b64encode(seg_data).decode("ascii")
                        evt = {
                            "type": "fmp4_init",
                            "dataBase64": b64,
                            "byteLength": len(seg_data),
                            "timing": {
                                "timeToInitMs": round((t_now - stream_start_mono) * 1000, 1),
                            },
                        }
                        results.append(evt)
                    else:
                        self.produced_media_s += CHUNK_MEDIA_S
                        meta = pending_chunk_meta.popleft() if pending_chunk_meta else _ChunkMeta()
                        b64 = base64.b64encode(seg_data).decode("ascii")
                        evt = {
                            "type": "fmp4_media",
                            "dataBase64": b64,
                            "byteLength": len(seg_data),
                            "sequence": media_sequence,
                        }
                        results.append(("media", seg_data, evt, meta))
                return results

            @property
            def max_inter_yield_gap_ms(self):
                return round(max(self._inter_yield_gaps), 1) if self._inter_yield_gaps else 0

            @property
            def avg_inter_yield_gap_ms(self):
                return round(sum(self._inter_yield_gaps) / len(self._inter_yield_gaps), 1) if self._inter_yield_gaps else 0

            @property
            def total_pacing_sleep_ms(self):
                return round(self._total_pacing_sleep_ms, 1)

            @property
            def queue_depth(self):
                return len(self.pending)

            @property
            def min_queue_depth(self):
                return self._min_queue_depth if self._min_queue_depth != 999999 else 0

            @property
            def max_queue_depth(self):
                return self._max_queue_depth

            @property
            def dry_spell_count(self):
                return self._dry_spell_count

        sender = _SegmentSender()
        speech_first_segment_emitted = False
        speech_end_emitted = False

        _audio_write_count = [0]
        _video_write_count = [0]

        def _queue_audio(data: bytes, label: str) -> bool:
            nonlocal total_audio_bytes_written
            try:
                audio_write_queue.put(data, timeout=2)
                total_audio_bytes_written += len(data)
                _audio_write_count[0] += 1
                if _audio_write_count[0] <= 3 or _audio_write_count[0] % 100 == 0:
                    print(f"[l40s-timeline_fmp4_av] audio queued: {len(data)}B #{_audio_write_count[0]} ({label})")
                return True
            except queue.Full:
                print(f"[l40s-timeline_fmp4_av] audio_write_queue FULL, cannot queue {label}")
                return False

        def _write_video_frames(frames_list, label: str) -> bool:
            nonlocal total_raw_bytes_written
            raw_chunk = b"".join(frame.data.tobytes() for frame in frames_list)
            assert len(raw_chunk) == len(frames_list) * FRAME_BYTES, (
                f"Frame bytes mismatch: {len(raw_chunk)} != {len(frames_list)} * {FRAME_BYTES}"
            )
            try:
                video_write_queue.put(raw_chunk, timeout=2)
                total_raw_bytes_written += len(raw_chunk)
                _video_write_count[0] += 1
                if _video_write_count[0] <= 3 or _video_write_count[0] % 100 == 0:
                    print(f"[l40s-timeline_fmp4_av] video queued: {len(raw_chunk)}B ({len(frames_list)} frames, {label})")
                return True
            except queue.Full:
                print(f"[l40s-timeline_fmp4_av] video_write_queue FULL, cannot queue {label}")
                return False

        def _drain_into_sender():
            nonlocal init_sent_to_client, eof_from_reader, total_encoded_bytes_sent
            while True:
                try:
                    item = segment_queue.get_nowait()
                    if item is None:
                        eof_from_reader = True
                        return
                    seg_type, seg_data = item
                    if seg_type == "init" and not init_sent_to_client:
                        init_sent_to_client = True
                        sender.add(seg_type, seg_data)
                        total_encoded_bytes_sent += len(seg_data)
                        gpu_snapshots.append(_gpu_memory_snapshot("fmp4_init"))
                        print(f"[l40s-timeline_fmp4_av] init segment {len(seg_data)} bytes")
                    elif seg_type == "media":
                        sender.add(seg_type, seg_data)
                        total_encoded_bytes_sent += len(seg_data)
                except queue.Empty:
                    return

        def _stamp_and_yield_media(evt, meta, pacing_extra=None):
            nonlocal media_sequence, speech_first_segment_emitted, speech_end_emitted
            evt["mode"] = meta.mode if meta.mode else current_mode
            evt["turnId"] = meta.turn_id if meta.turn_id else None
            evt["segmentId"] = meta.segment_id if meta.segment_id is not None else self._timeline_segment_id
            evt["chunkIndex"] = meta.chunk_index if meta.chunk_index is not None else cumulative_chunk_index
            timing = {
                "chunkRenderMs": round(meta.chunk_render_ms, 1) if meta.chunk_render_ms else 0,
                "feedMs": round(meta.feed_ms, 1) if meta.feed_ms else 0,
                "framesFed": meta.frames_fed or 0,
                "audioSamplesPerChunk": step,
            }
            if pacing_extra:
                timing.update(pacing_extra)

            if meta.overlap_with_tts:
                timing["overlapWithTts"] = True
            evt["containerInstanceId"] = getattr(self, "_container_instance_id", None)

            if meta.mode == "speech" and meta.is_speech_start and not speech_first_segment_emitted:
                evt["isSpeechStart"] = True
                speech_first_segment_emitted = True
                if meta.enqueue_to_first_frame_ms and meta.enqueue_to_first_frame_ms > 0:
                    timing["enqueueToFirstFrameMs"] = round(meta.enqueue_to_first_frame_ms, 1)
                if gpu_snapshots:
                    last_snap = gpu_snapshots[-1]
                    evt["gpu"] = {k: v for k, v in last_snap.items() if k != "label"}
            else:
                evt["isSpeechStart"] = False

            if meta.mode == "speech" and meta.is_speech_end and not speech_end_emitted:
                evt["isSpeechEnd"] = True
                speech_end_emitted = True
            else:
                evt["isSpeechEnd"] = False

            if verbose_delivery:
                timing["modalInterYieldGapMs"] = sender.avg_inter_yield_gap_ms
                timing["senderQueueDepth"] = sender.queue_depth
                timing["sourceLeadMs"] = max(0, round((sender.produced_media_s - (time.monotonic() - stream_start_mono)) * 1000))
                timing["targetLeadMs"] = round(TARGET_LEAD_S * 1000)
                timing["segmentMediaDurationMs"] = round(CHUNK_MEDIA_S * 1000)

            evt["timing"] = timing
            media_sequence += 1
            return evt

        def _yield_sender_events():
            """Drain FFmpeg into sender, then yield paced events."""
            _drain_into_sender()
            paced = sender.yield_events()
            for item in paced:
                if isinstance(item, dict):
                    yield f"data: {json.dumps(item)}\n\n"
                else:
                    _, _, evt, meta = item
                    _stamp_and_yield_media(evt, meta)
                    yield f"data: {json.dumps(evt)}\n\n"

        try:
            idle_chunk_index = 0
            cumulative_chunk_index = 0
            eof_from_reader = False

            same_gpu_handoff = self._audio_handoffs.get(self._session_id)
            audio_buffer = np.zeros(0, dtype=np.float32)
            handoff_turn_id = None
            handoff_enqueue_at = None
            handoff_first_segment_emitted = False
            handoff_end_emitted = False
            handoff_chunk_index = 0
            handoff_is_last_received = False

            def _yield_rendered(current_mode_label, turn_id_for_event, is_speech_start, is_speech_end, enqueue_at_val, speech_first_segment, speech_end_segment, extra_timing=None):
                nonlocal cumulative_chunk_index, chunks_rendered, last_gpu_snapshot_chunk, eof_from_reader
                nonlocal speech_first_segment_emitted, speech_end_emitted

                enqueue_to_first = None
                if is_speech_start and not speech_first_segment_emitted and enqueue_at_val and enqueue_at_val > 0:
                    enqueue_to_first = (time.time() - enqueue_at_val) * 1000

                _push_chunk_meta(
                    mode=current_mode_label,
                    turn_id=turn_id_for_event,
                    segment_id=self._timeline_segment_id,
                    chunk_index=cumulative_chunk_index,
                    is_speech_start=is_speech_start and not speech_first_segment_emitted,
                    is_speech_end=is_speech_end and not speech_end_emitted,
                    chunk_render_ms=render_ms,
                    feed_ms=encode_ms,
                    frames_fed=len(frames_list),
                    overlap_with_tts=getattr(self, "_tts_overlap_active", False),
                    enqueue_to_first_frame_ms=enqueue_to_first,
                    audio_samples_per_chunk=step,
                )

                yield from _yield_sender_events()

                cumulative_chunk_index += 1
                chunks_rendered += 1

                if cumulative_chunk_index - last_gpu_snapshot_chunk >= 25:
                    gpu_snapshots.append(_gpu_memory_snapshot(f"periodic_chunk_{cumulative_chunk_index}"))
                    last_gpu_snapshot_chunk = cumulative_chunk_index

            while not self._timeline_stop_requested and getattr(self, "_session_active", False) and not eof_from_reader:
                if not init_sent_to_client and not init_5s_warned and time.monotonic() > stream_start_mono + 5:
                    init_5s_warned = True
                    print("[l40s-timeline_fmp4_av] WARNING: no init after 5s, FFmpeg may be stalled")

                if not init_sent_to_client and time.monotonic() > init_deadline_mono:
                    print(f"[l40s-timeline_fmp4_av] INIT TIMEOUT: no ftyp+moov within {INIT_TIMEOUT_S}s")
                    err_event = {"type": "error", "error": f"fMP4 AV init timeout: no init segment produced within {INIT_TIMEOUT_S}s"}
                    yield f"data: {json.dumps(err_event)}\n\n"
                    break

                if same_gpu_handoff is not None:
                    try:
                        should_wait = (current_mode == "speech" and not handoff_is_last_received)
                        handoff_item = same_gpu_handoff.queue.get(block=should_wait, timeout=HANDOFF_WAIT_S if should_wait else 0)
                    except queue.Empty:
                        handoff_item = None
                else:
                    handoff_item = None

                if handoff_item is not None and handoff_item is not _SENTINEL:
                    chunk_pcm, meta = handoff_item
                    chunk_turn_id = meta["turnId"]
                    chunk_index = meta["chunkIndex"]
                    is_first = meta["isFirst"]
                    is_last = meta["isLast"]

                    if is_first:
                        audio_buffer = np.zeros(0, dtype=np.float32)
                        handoff_turn_id = chunk_turn_id
                        handoff_enqueue_at = meta.get("generatedAt")
                        handoff_first_segment_emitted = False
                        handoff_end_emitted = False
                        handoff_chunk_index = 0
                        handoff_is_last_received = False
                        speech_first_segment_emitted = False
                        speech_end_emitted = False
                        current_mode = "speech"
                        gpu_snapshots.append(_gpu_memory_snapshot("speech_handoff_start"))
                        print(
                            f"[l40s-timeline_fmp4_av] speech handoff start turnId={chunk_turn_id} "
                            f"from in-process queue"
                        )

                    audio_buffer = np.concatenate([audio_buffer, chunk_pcm])

                    while len(audio_buffer) >= window:
                        windowed = audio_buffer[:window].copy()
                        audio_buffer = audio_buffer[step:]

                        ffmpeg_audio = windowed[:step]
                        audio_s16 = (ffmpeg_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                        if not _queue_audio(audio_s16, f"speech handoff {handoff_chunk_index}"):
                            self._timeline_stop_requested = True
                            break

                        render_t0 = time.monotonic()
                        chunk_obj = Chunk(audio_speech=windowed, audio_listen=silence_audio)
                        self._session_state, frames_iter = self.pipeline.process_chunk(
                            self._session_avatar, chunk_obj,
                            state=self._session_state,
                            options=self._options,
                        )
                        frames_list = list(frames_iter)
                        render_ms = (time.monotonic() - render_t0) * 1000

                        encode_t0 = time.monotonic()
                        if not _write_video_frames(frames_list, f"speech handoff {handoff_chunk_index}"):
                            self._timeline_stop_requested = True
                            break
                        encode_ms = (time.monotonic() - encode_t0) * 1000

                        speech_first_segment = not handoff_first_segment_emitted
                        speech_end_segment = False

                        yield from _yield_rendered(
                            "speech", handoff_turn_id,
                            is_speech_start=speech_first_segment,
                            is_speech_end=False,
                            enqueue_at_val=handoff_enqueue_at,
                            speech_first_segment=speech_first_segment,
                            speech_end_segment=False,
                        )
                        handoff_first_segment_emitted = True
                        handoff_chunk_index += 1

                    if is_last:
                        handoff_is_last_received = True

                        if len(audio_buffer) > 0:
                            padded = np.pad(audio_buffer, (0, window - len(audio_buffer)))
                            ffmpeg_audio = audio_buffer[:step]
                            if len(ffmpeg_audio) < step:
                                ffmpeg_audio = np.pad(ffmpeg_audio, (0, step - len(ffmpeg_audio)))
                            audio_s16 = (ffmpeg_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                            _queue_audio(audio_s16, f"speech handoff final")

                            render_t0 = time.monotonic()
                            chunk_obj = Chunk(audio_speech=padded, audio_listen=silence_audio)
                            self._session_state, frames_iter = self.pipeline.process_chunk(
                                self._session_avatar, chunk_obj,
                                state=self._session_state,
                                options=self._options,
                            )
                            frames_list = list(frames_iter)
                            render_ms = (time.monotonic() - render_t0) * 1000

                            encode_t0 = time.monotonic()
                            _write_video_frames(frames_list, f"speech handoff final")
                            encode_ms = (time.monotonic() - encode_t0) * 1000

                            yield from _yield_rendered(
                                "speech", handoff_turn_id,
                                is_speech_start=not handoff_first_segment_emitted,
                                is_speech_end=True,
                                enqueue_at_val=handoff_enqueue_at,
                                speech_first_segment=not handoff_first_segment_emitted,
                                speech_end_segment=True,
                            )
                            handoff_first_segment_emitted = True
                            handoff_chunk_index += 1

                        audio_buffer = np.zeros(0, dtype=np.float32)

                        if not handoff_end_emitted:
                            _push_chunk_meta(
                                mode="speech",
                                turn_id=handoff_turn_id,
                                segment_id=self._timeline_segment_id,
                                chunk_index=cumulative_chunk_index,
                                is_speech_start=False,
                                is_speech_end=True,
                                chunk_render_ms=0,
                                feed_ms=0,
                                frames_fed=0,
                                overlap_with_tts=False,
                                audio_samples_per_chunk=step,
                            )
                            yield from _yield_sender_events()
                            cumulative_chunk_index += 1
                            chunks_rendered += 1
                            handoff_end_emitted = True

                        self._timeline_segment_id += 1
                        idle_chunk_index = 0
                        current_mode = "idle"
                        speech_end_mono = time.monotonic()
                        gpu_snapshots.append(_gpu_memory_snapshot("speech_handoff_end"))
                        print(f"[l40s-timeline_fmp4_av] speech handoff done turnId={handoff_turn_id}")

                    continue

                if handoff_item is _SENTINEL:
                    print("[l40s-timeline_fmp4_av] received handoff sentinel, stopping handoff consumption")
                    same_gpu_handoff = None
                    handoff_item = None

                speech_item = timeline_queue.get(block=False)

                if speech_item is not None:
                    if speech_item.get("__stop"):
                        self._timeline_stop_requested = True
                        print("[l40s-timeline_fmp4_av] received __stop sentinel")
                        break

                    turn_id = speech_item["turnId"]
                    audio_b64 = speech_item["audioBase64"]
                    enqueue_at = speech_item.get("enqueueAt", 0)
                    audio_duration_ms_reported = speech_item.get("audioDurationMs", 0)

                    wav_bytes = base64.b64decode(audio_b64)
                    audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                    audio_f32 = audio_f32.mean(axis=1)

                    if sr != AUDIO_SR:
                        print(f"[l40s-timeline_fmp4_av] resampling {sr}Hz -> {AUDIO_SR}Hz ({len(audio_f32)} samples)")
                        audio_f32 = soxr.resample(audio_f32, sr, AUDIO_SR, quality="HQ")
                        audio_f32 = audio_f32.astype(np.float32)

                    total_duration_s = len(audio_f32) / AUDIO_SR
                    if total_duration_s > EMERGENCY_SPEECH_CAP_S:
                        print(f"[l40s-timeline_fmp4_av] EMERGENCY CAP: truncating {total_duration_s:.1f}s -> {EMERGENCY_SPEECH_CAP_S}s")
                        audio_f32 = audio_f32[:int(EMERGENCY_SPEECH_CAP_S * AUDIO_SR)]
                        total_duration_s = EMERGENCY_SPEECH_CAP_S

                    audio_duration_ms = len(audio_f32) / AUDIO_SR * 1000

                    n_steps = max(1, (len(audio_f32) + step - 1) // step)
                    chunks_audio = []
                    for ci in range(n_steps):
                        s = ci * step
                        piece = audio_f32[s : s + window]
                        if len(piece) < window:
                            piece = np.pad(piece, (0, window - len(piece)))
                        chunks_audio.append(piece)

                    total_frames_expected = n_steps * mg.chunk_size
                    print(
                        f"[l40s-timeline_fmp4_av] speech turnId={turn_id} (modal.Queue) "
                        f"{len(audio_f32)} samples, {audio_duration_ms:.0f}ms, "
                        f"{n_steps} chunks, {total_frames_expected} expected frames"
                    )

                    current_mode = "speech"
                    gpu_snapshots.append(_gpu_memory_snapshot("speech_dequeue"))

                    speech_first_segment_emitted = False
                    speech_end_emitted = False

                    for ci, chunk_audio in enumerate(chunks_audio):
                        if self._timeline_stop_requested:
                            break

                        chunk_audio_start = ci * step
                        chunk_audio_end = min((ci + 1) * step, len(audio_f32))
                        audio_piece = audio_f32[chunk_audio_start:chunk_audio_end]
                        if len(audio_piece) < step:
                            audio_piece = np.pad(audio_piece, (0, step - len(audio_piece)))
                        audio_s16 = (audio_piece * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

                        if not _queue_audio(audio_s16, f"speech chunk {ci}"):
                            self._timeline_stop_requested = True
                            break

                        render_t0 = time.monotonic()
                        chunk_obj = Chunk(audio_speech=chunk_audio, audio_listen=silence_audio)
                        self._session_state, frames_iter = self.pipeline.process_chunk(
                            self._session_avatar, chunk_obj,
                            state=self._session_state,
                            options=self._options,
                        )
                        frames_list = list(frames_iter)
                        render_ms = (time.monotonic() - render_t0) * 1000

                        encode_t0 = time.monotonic()
                        if not _write_video_frames(frames_list, f"speech chunk {ci}"):
                            self._timeline_stop_requested = True
                            break
                        encode_ms = (time.monotonic() - encode_t0) * 1000

                        is_speech_end = (ci == n_steps - 1)

                        yield from _yield_rendered(
                            "speech", turn_id,
                            is_speech_start=not speech_first_segment_emitted,
                            is_speech_end=is_speech_end and not speech_end_emitted,
                            enqueue_at_val=enqueue_at,
                            speech_first_segment=not speech_first_segment_emitted,
                            speech_end_segment=is_speech_end and not speech_end_emitted,
                        )

                    if not speech_end_emitted:
                        _push_chunk_meta(
                            mode="speech",
                            turn_id=turn_id,
                            segment_id=self._timeline_segment_id,
                            chunk_index=cumulative_chunk_index,
                            is_speech_start=False,
                            is_speech_end=True,
                            chunk_render_ms=0,
                            feed_ms=0,
                            frames_fed=0,
                            overlap_with_tts=False,
                            audio_samples_per_chunk=step,
                        )
                        yield from _yield_sender_events()
                        cumulative_chunk_index += 1
                        chunks_rendered += 1
                        speech_end_emitted = True

                    self._timeline_segment_id += 1
                    idle_chunk_index = 0
                    current_mode = "idle"
                    speech_end_mono = time.monotonic()
                    gpu_snapshots.append(_gpu_memory_snapshot("speech_end"))

                    print(f"[l40s-timeline_fmp4_av] speech done turnId={turn_id} (modal.Queue)")

                else:
                    current_mode = "idle"

                    silence_s16 = np.zeros(step, dtype=np.int16).tobytes()
                    if not _queue_audio(silence_s16, f"idle chunk {cumulative_chunk_index}"):
                        self._timeline_stop_requested = True
                        break

                    is_first_idle_after_speech = speech_end_mono is not None
                    if is_first_idle_after_speech:
                        gpu_snapshots.append(_gpu_memory_snapshot("first_idle_after_speech"))
                        speech_end_to_idle_ms = round((time.monotonic() - speech_end_mono) * 1000, 1)
                        speech_end_mono = None
                    else:
                        speech_end_to_idle_ms = None

                    render_t0 = time.monotonic()
                    chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
                    self._session_state, frames_iter = self.pipeline.process_chunk(
                        self._session_avatar, chunk_obj,
                        state=self._session_state,
                        options=self._options,
                    )
                    frames_list = list(frames_iter)
                    render_ms = (time.monotonic() - render_t0) * 1000

                    encode_t0 = time.monotonic()
                    if not _write_video_frames(frames_list, f"idle chunk {cumulative_chunk_index}"):
                        self._timeline_stop_requested = True
                        break
                    encode_ms = (time.monotonic() - encode_t0) * 1000

                    extra_idle_timing = {}
                    if speech_end_to_idle_ms is not None:
                        extra_idle_timing["speechEndToIdleResumeMs"] = speech_end_to_idle_ms

                    yield from _yield_rendered(
                        "idle", None,
                        is_speech_start=False,
                        is_speech_end=False,
                        enqueue_at_val=None,
                        speech_first_segment=False,
                        speech_end_segment=False,
                        extra_timing=extra_idle_timing if extra_idle_timing else None,
                    )

                    idle_chunk_index += 1

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"fMP4 A/V stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"
        finally:
            print(f"[l40s-timeline_fmp4_av] cleanup starting ffmpeg_pid={ffmpeg_proc.pid}")

            try:
                video_write_queue.put(None, timeout=2)
            except Exception:
                pass

            try:
                audio_write_queue.put(None, timeout=2)
            except Exception:
                pass

            try:
                video_thread.join(timeout=5)
                if video_thread.is_alive():
                    print("[l40s-timeline_fmp4_av] WARNING: video_thread still alive after join timeout")
                else:
                    print("[l40s-timeline_fmp4_av] video_thread joined cleanly")
            except Exception:
                pass

            try:
                audio_thread.join(timeout=5)
                if audio_thread.is_alive():
                    print("[l40s-timeline_fmp4_av] WARNING: audio_thread still alive after join timeout")
                else:
                    print("[l40s-timeline_fmp4_av] audio_thread joined cleanly")
            except Exception:
                pass

            try:
                stdout_thread.join(timeout=3)
                if stdout_thread.is_alive():
                    print("[l40s-timeline_fmp4_av] WARNING: stdout_thread still alive after join timeout")
                else:
                    print("[l40s-timeline_fmp4_av] stdout_thread joined cleanly")
            except Exception:
                pass

            try:
                stderr_thread.join(timeout=3)
                if stderr_thread.is_alive():
                    print("[l40s-timeline_fmp4_av] WARNING: stderr_thread still alive after join timeout")
                else:
                    print("[l40s-timeline_fmp4_av] stderr_thread joined cleanly")
            except Exception:
                pass

            if ffmpeg_proc.poll() is None:
                print(f"[l40s-timeline_fmp4_av] terminating ffmpeg pid={ffmpeg_proc.pid}")
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            print(f"[l40s-timeline_fmp4_av] ffmpeg exited rc={ffmpeg_proc.returncode}")

            stderr_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
            if stderr_errors:
                print(f"[l40s-timeline_fmp4_av] FFmpeg stderr errors ({len(stderr_errors)}): {stderr_errors[:5]}")
            stderr_tail = ffmpeg_stderr_lines[-20:] if ffmpeg_stderr_lines else []
            if stderr_tail:
                print(f"[l40s-timeline_fmp4_av] FFmpeg stderr tail: {stderr_tail}")

            if reader_error:
                print(f"[l40s-timeline_fmp4_av] stdout_reader error: {reader_error[0]}")
            if audio_writer_error:
                print(f"[l40s-timeline_fmp4_av] audio_writer error: {audio_writer_error[0]}")
            if video_writer_error:
                print(f"[l40s-timeline_fmp4_av] video_writer error: {video_writer_error[0]}")

            try:
                os.unlink(fifo_path)
            except OSError:
                pass

            gpu_snapshots.append(_gpu_memory_snapshot("stream_end"))

            gpu_summary: dict = {}
            if gpu_snapshots:
                smi_used = [s.get("smiMemoryUsedMb", 0) for s in gpu_snapshots if "smiMemoryUsedMb" in s]
                smi_total = [s.get("smiMemoryTotalMb", 0) for s in gpu_snapshots if "smiMemoryTotalMb" in s]
                torch_alloc = [s.get("torchAllocatedMb", 0) for s in gpu_snapshots if "torchAllocatedMb" in s]
                torch_peak = [s.get("torchPeakMb", 0) for s in gpu_snapshots if "torchPeakMb" in s]
                smi_gpu_util = [s.get("smiGpuUtilPct", 0) for s in gpu_snapshots if "smiGpuUtilPct" in s]
                smi_mem_util = [s.get("smiMemoryUtilPct", 0) for s in gpu_snapshots if "smiMemoryUtilPct" in s]
                gpu_summary = {
                    "smiUsedMinMb": min(smi_used) if smi_used else None,
                    "smiUsedMaxMb": max(smi_used) if smi_used else None,
                    "smiTotalMb": smi_total[-1] if smi_total else None,
                    "torchAllocMinMb": min(torch_alloc) if torch_alloc else None,
                    "torchAllocMaxMb": max(torch_alloc) if torch_alloc else None,
                    "torchPeakMb": max(torch_peak) if torch_peak else None,
                    "smiGpuUtilMinPct": min(smi_gpu_util) if smi_gpu_util else None,
                    "smiGpuUtilMaxPct": max(smi_gpu_util) if smi_gpu_util else None,
                    "smiGpuUtilAvgPct": round(sum(smi_gpu_util) / len(smi_gpu_util), 1) if smi_gpu_util else None,
                    "smiMemoryUtilMinPct": min(smi_mem_util) if smi_mem_util else None,
                    "smiMemoryUtilMaxPct": max(smi_mem_util) if smi_mem_util else None,
                    "smiMemoryUtilAvgPct": round(sum(smi_mem_util) / len(smi_mem_util), 1) if smi_mem_util else None,
                    "snapshotCount": len(gpu_snapshots),
                    "snapshotLabels": [s.get("label") for s in gpu_snapshots],
                }

            stderr_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]

            end_event = {
                "type": "end",
                "chunksRendered": chunks_rendered,
                "mediaSegments": media_sequence,
                "audioBytes": total_audio_bytes_written,
                "videoBytes": total_raw_bytes_written,
                "encodedBytes": total_encoded_bytes_sent,
                "ffmpegReturnCode": ffmpeg_proc.returncode,
                "ffmpegStderrErrorCount": len(stderr_errors),
                "gpu": gpu_summary,
                "containerInstanceId": getattr(self, "_container_instance_id", None),
                "cadenceStats": {
                    "targetLeadS": TARGET_LEAD_S,
                    "chunkMediaS": CHUNK_MEDIA_S,
                    "maxModalInterYieldGapMs": sender.max_inter_yield_gap_ms,
                    "avgModalInterYieldGapMs": sender.avg_inter_yield_gap_ms,
                    "totalPacingSleepMs": sender.total_pacing_sleep_ms,
                    "modalSegmentsYielded": media_sequence,
                    "selectedFfmpegVariant": "current_g25_frag_keyframe",
                    "senderMinQueueDepth": sender.min_queue_depth,
                    "senderMaxQueueDepth": sender.max_queue_depth,
                    "senderDrySpellCount": sender.dry_spell_count,
                },
            }
            yield f"data: {json.dumps(end_event)}\n\n"

            self._timeline_running = False
            self._timeline_stop_requested = False
            print(f"[l40s-timeline_fmp4_av] cleanup complete chunks={chunks_rendered} segments={media_sequence} "
                  f"audioBytes={total_audio_bytes_written} videoBytes={total_raw_bytes_written} "
                  f"encodedBytes={total_encoded_bytes_sent}")

    def _do_timeline_enqueue(self, body: dict) -> dict:
        if not getattr(self, "_snap_ok", False):
            return {"ok": False, "error": "Pipeline not ready"}

        session_id = body.get("session_id")
        if not session_id:
            return {"ok": False, "error": "No session_id provided"}

        turn_id = body.get("turnId")
        audio_base64 = body.get("audio_base64")
        audio_duration_ms = body.get("audio_duration_ms", 0)

        if not audio_base64:
            return {"ok": False, "error": "No audio_base64 provided"}

        q = modal.Queue.from_name(f"avtr1-timeline-{session_id}")
        q.put({
            "turnId": turn_id,
            "audioBase64": audio_base64,
            "audioDurationMs": audio_duration_ms,
            "enqueueAt": time.time(),
        })

        print(f"[timeline_enqueue] turnId={turn_id} durationMs={audio_duration_ms:.0f}")

        return {
            "ok": True,
            "turnId": turn_id,
            "audioDurationMs": audio_duration_ms,
        }

    def _do_timeline_stop(self, body: dict) -> dict:
        session_id = body.get("session_id")
        was_running = getattr(self, "_timeline_running", False)
        tts_overlap = getattr(self, "_tts_overlap_active", False)
        self._timeline_stop_requested = True

        if session_id:
            try:
                q = modal.Queue.from_name(f"avtr1-timeline-{session_id}", create_if_missing=True)
                while True:
                    item = q.get(block=False)
                    if item is None:
                        break
            except Exception:
                pass

        print(f"[timeline_stop] was_running={was_running} tts_overlap={tts_overlap}")
        return {
            "ok": True,
            "wasRunning": was_running,
            "ttsOverlapAtStop": tts_overlap,
            "containerInstanceId": getattr(self, "_container_instance_id", None),
        }

    @modal.method()
    def health(self) -> dict:
        return self._do_health()

    @modal.method()
    def start_session(self, avatar_id: str | None = None, bg_id: str | None = None) -> dict:
        return self._do_start_session(avatar_id, bg_id)

    @modal.method()
    def snapshot_probe(self) -> dict:
        return self._do_snapshot_probe()

    def _do_snapshot_probe(self) -> dict:
        import torch

        return {
            "app": "avtr1-lifecycle-parallel-l40s",
            "snapshotMode": "cpu_only",
            "gpuSnapshotEnabled": False,
            "cpuSnapshotOk": getattr(self, "_cpu_snapshot_ok", False),
            "gpuInitOk": getattr(self, "_gpu_init_ok", False),
            "pipelineLoaded": getattr(self, "_pipeline_loaded", False),
            "omnivoiceLoaded": getattr(self, "_omnivoice_loaded", False),
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "avatar_id": getattr(self, "avatar_id", None),
            "bg_id": getattr(self, "_bg_id", None),
            "session_active": getattr(self, "_session_active", False),
            "has_session_state": getattr(self, "_session_state", None) is not None,
            "has_warmup_state": getattr(self, "_state", None) is not None,
            "omnivoice_loaded": hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None,
            "omnivoice_sr": getattr(self, "_omnivoice_sr", None),
            "container_instance_id": getattr(self, "_container_instance_id", None),
            "gpuArtifactsPending": getattr(self, "_gpu_artifacts_pending", False),
            "missingEnginesAtCpuSnap": getattr(self, "_missing_engines_at_cpu_snap", []),
            "cpuSnapPrep_s": getattr(self, "_cpu_snap_prep_s", None),
            "gpuInit_s": getattr(self, "_gpu_init_s", None),
            "totalRestoreToReady_s": getattr(self, "_total_restore_to_ready_s", None),
        }

    def _do_nvenc_probe(self) -> dict:
        import numpy as np

        result = {
            "timestamp": time.time(),
            "stages": {},
        }

        try:
            t0 = time.monotonic()
            ver_proc = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10,
            )
            result["stages"]["ffmpeg_version"] = {
                "returncode": ver_proc.returncode,
                "stdout_first_line": ver_proc.stdout.split("\n")[0] if ver_proc.stdout else None,
                "available": ver_proc.returncode == 0,
            }
            result["stages"]["ffmpeg_version_ms"] = round((time.monotonic() - t0) * 1000, 1)
        except FileNotFoundError:
            result["stages"]["ffmpeg_version"] = {"available": False, "error": "ffmpeg not found in PATH"}
            return result
        except Exception as exc:
            result["stages"]["ffmpeg_version"] = {"available": False, "error": str(exc)}
            return result

        try:
            t1 = time.monotonic()
            enc_proc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            encoder_lines = [
                line for line in enc_proc.stdout.split("\n")
                if "nvenc" in line.lower()
            ]
            nvenc_encoders = []
            for line in encoder_lines:
                parts = line.split()
                if len(parts) >= 2:
                    nvenc_encoders.append({
                        "name": parts[1].lstrip("."),
                        "type": parts[0].strip(),
                    })
            result["stages"]["nvenc_encoders"] = {
                "available": len(nvenc_encoders) > 0,
                "encoders": nvenc_encoders,
                "raw_lines": encoder_lines,
            }
            result["stages"]["encoder_list_ms"] = round((time.monotonic() - t1) * 1000, 1)
        except Exception as exc:
            result["stages"]["nvenc_encoders"] = {"available": False, "error": str(exc)}
            return result

        if not nvenc_encoders:
            result["stages"]["encode_test"] = {"skipped": True, "reason": "No NVENC encoders found"}
            return result

        W, H = 1280, 720
        FPS = 25
        N_FRAMES = 5
        FRAME_BYTES = H * W * 3 // 2

        grey_y = np.full((H, W), 128, dtype=np.uint8)
        u_plane = np.full((H // 2, W // 2), 128, dtype=np.uint8)
        v_plane = np.full((H // 2, W // 2), 128, dtype=np.uint8)
        yuv_frame = np.concatenate([
            grey_y.ravel(),
            u_plane.ravel(),
            v_plane.ravel(),
        ]).astype(np.uint8)
        assert yuv_frame.nbytes == FRAME_BYTES, f"Frame size mismatch: {yuv_frame.nbytes} != {FRAME_BYTES}"

        raw_input = yuv_frame.tobytes() * N_FRAMES

        encode_tests = {}

        test_configs = [
            {
                "label": "h264_nvenc_default",
                "codec": "h264_nvenc",
                "extra_args": [],
            },
            {
                "label": "h264_nvenc_baseline",
                "codec": "h264_nvenc",
                "extra_args": ["-profile:v", "baseline", "-level", "3.1"],
            },
            {
                "label": "h264_nvenc_low_latency",
                "codec": "h264_nvenc",
                "extra_args": [
                    "-preset", "p4", "-tune", "ull",
                    "-g", "25",
                    "-bf", "0",
                ],
            },
        ]

        for cfg in test_configs:
            label = cfg["label"]
            try:
                cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "verbose",
                    "-f", "rawvideo", "-pix_fmt", "yuv420p",
                    "-s", f"{W}x{H}", "-r", str(FPS),
                    "-i", "pipe:0",
                    "-c:v", cfg["codec"],
                    *cfg["extra_args"],
                    "-f", "mp4",
                    "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                    "-frag_duration", "200000",
                    "pipe:1",
                ]

                t_enc = time.monotonic()
                proc = subprocess.run(
                    cmd,
                    input=raw_input,
                    capture_output=True,
                    timeout=30,
                )
                enc_ms = (time.monotonic() - t_enc) * 1000

                output_bytes = len(proc.stdout) if proc.stdout else 0
                stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

                codec_detail_lines = [
                    line for line in stderr_text.split("\n")
                    if any(kw in line.lower() for kw in ["stream", "codec", "profile", "level", "nvenc", "encoder", "pix_fmt", "output"])
                ]

                encode_tests[label] = {
                    "success": proc.returncode == 0 and output_bytes > 0,
                    "returncode": proc.returncode,
                    "outputBytes": output_bytes,
                    "encodeMs": round(enc_ms, 1),
                    "inputBytes": len(raw_input),
                    "compressionRatio": round(len(raw_input) / output_bytes, 2) if output_bytes > 0 else None,
                    "stderrRelevantLines": codec_detail_lines[:20],
                    "stderrTail": stderr_text[-500:] if stderr_text else None,
                    "cmd": cmd,
                }
            except Exception as exc:
                encode_tests[label] = {
                    "success": False,
                    "error": str(exc),
                }

        result["stages"]["encode_tests"] = encode_tests

        result["stages"]["gpu_info"] = {
            "gpu_name": None,
            "cuda_available": False,
        }
        try:
            import torch
            result["stages"]["gpu_info"]["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                result["stages"]["gpu_info"]["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return result

    def _do_fmp4_real_frame_probe(self) -> dict:
        import base64
        import json
        import struct
        import tempfile

        import numpy as np

        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            return {"success": False, "error": "Pipeline not ready (snap_ok=False)"}

        W, H = 1280, 720
        FPS = 25
        N_CHUNKS = 10
        FRAMES_PER_CHUNK = 5
        N_FRAMES = N_CHUNKS * FRAMES_PER_CHUNK
        FRAME_BYTES = H * W * 3 // 2
        FRAG_DURATION_US = 200000

        result = {
            "success": False,
            "probeConfig": {
                "width": W,
                "height": H,
                "fps": FPS,
                "nChunks": N_CHUNKS,
                "framesPerChunk": FRAMES_PER_CHUNK,
                "nFrames": N_FRAMES,
                "frameBytesI420": FRAME_BYTES,
                "fragDurationUs": FRAG_DURATION_US,
                "fragDurationMs": FRAG_DURATION_US / 1000,
                "expectedFragments": round(N_FRAMES / FPS / (FRAG_DURATION_US / 1_000_000), 1),
            },
        }

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        silence_audio = np.zeros(window, dtype=np.float32)

        t_render = time.monotonic()
        local_state = None
        all_frames = []
        for i in range(N_CHUNKS):
            chunk = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
            local_state, frames_iter = self.pipeline.process_chunk(
                self.avatar, chunk, state=local_state, options=self._options,
            )
            all_frames.extend(frames_iter)
        render_ms = (time.monotonic() - t_render) * 1000

        result["renderPhase"] = {
            "framesRendered": len(all_frames),
            "renderMs": round(render_ms, 1),
            "avgRenderMsPerChunk": round(render_ms / N_CHUNKS, 1),
            "stateSource": "fresh_None_no_shared_state_mutation",
            "sharedStateUntouched": True,
            "note": "state=None creates brand new pipeline state. self._state and self._session_state are not read or written.",
        }

        if len(all_frames) < N_FRAMES:
            result["error"] = f"Expected {N_FRAMES} frames, got {len(all_frames)}"
            return result

        frame0 = all_frames[0]
        result["renderPhase"]["frameShape"] = list(frame0.data.shape)
        result["renderPhase"]["frameDtype"] = str(frame0.data.dtype)
        result["renderPhase"]["frameFormat"] = frame0.format
        result["renderPhase"]["frameReportedSize"] = f"{frame0.width}x{frame0.height}"

        t0 = time.monotonic()
        raw_i420_bytes = b"".join(frame.data.tobytes() for frame in all_frames)
        collect_ms = (time.monotonic() - t0) * 1000

        result["rawData"] = {
            "totalBytes": len(raw_i420_bytes),
            "expectedBytes": N_FRAMES * FRAME_BYTES,
            "bytesMatch": len(raw_i420_bytes) == N_FRAMES * FRAME_BYTES,
            "collectMs": round(collect_ms, 1),
        }

        t0 = time.monotonic()
        jpeg_total_bytes = 0
        jpeg_total_b64_chars = 0
        for frame in all_frames:
            timed = _yuv_i420_to_jpeg_b64_timed(
                frame.data, frame.height, frame.width, jpeg_quality=80,
            )
            jpeg_total_bytes += timed["encodedBytes"]
            jpeg_total_b64_chars += timed["base64Chars"]
        jpeg_encode_ms = (time.monotonic() - t0) * 1000

        result["jpegComparison"] = {
            "jpegQuality": 80,
            "jpegTotalBytes": jpeg_total_bytes,
            "jpegBase64TotalChars": jpeg_total_b64_chars,
            "jpegEncodeMs": round(jpeg_encode_ms, 1),
            "avgJpegBytesPerFrame": round(jpeg_total_bytes / N_FRAMES, 1),
            "avgJpegMsPerFrame": round(jpeg_encode_ms / N_FRAMES, 2),
        }

        def _run_ffmpeg_encode(raw_bytes, label, profile_args):
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "verbose",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{W}x{H}", "-r", str(FPS),
                "-i", "pipe:0",
                "-c:v", "h264_nvenc",
                "-preset", "p4", "-tune", "ull",
                "-g", str(FPS), "-bf", "0",
                *profile_args,
                "-f", "mp4",
                "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
                "-frag_duration", str(FRAG_DURATION_US),
                "pipe:1",
            ]
            t_enc = time.monotonic()
            proc = subprocess.run(
                cmd, input=raw_bytes, capture_output=True, timeout=60,
            )
            enc_ms = (time.monotonic() - t_enc) * 1000

            stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
            relevant = [
                line.strip() for line in stderr_text.split("\n")
                if any(kw in line.lower() for kw in [
                    "stream", "codec", "profile", "level", "nvenc",
                    "encoder", "pix_fmt", "output", "fragment", "moov", "moof",
                ])
            ]

            return {
                "label": label,
                "success": proc.returncode == 0 and len(proc.stdout) > 0,
                "returncode": proc.returncode,
                "outputBytes": len(proc.stdout),
                "encodeWallMs": round(enc_ms, 1),
                "avgEncodeMsPerFrame": round(enc_ms / N_FRAMES, 2) if proc.returncode == 0 else None,
                "inputBytes": len(raw_bytes),
                "compressionRatio": round(len(raw_bytes) / len(proc.stdout), 2) if proc.stdout else None,
                "stderrRelevantLines": relevant[:25],
                "stderrTail": stderr_text[-800:] if stderr_text else None,
                "cmd": cmd,
                "_output": proc.stdout if proc.returncode == 0 else None,
            }

        encode_results = {}
        chosen_label = None
        chosen_output = None
        chosen_profile_args = None

        profile_variants = [
            ("h264_nvenc_baseline_3.1", ["-profile:v", "baseline", "-level", "3.1"]),
            ("h264_nvenc_main_4.0", ["-profile:v", "main", "-level", "4.0"]),
        ]

        for label, pargs in profile_variants:
            cold = _run_ffmpeg_encode(raw_i420_bytes, f"cold_{label}", pargs)
            encode_results[f"cold_{label}"] = {k: v for k, v in cold.items() if k != "_output"}

            if cold["success"] and chosen_output is None:
                chosen_label = label
                chosen_output = cold["_output"]
                chosen_profile_args = pargs

                warm = _run_ffmpeg_encode(raw_i420_bytes, f"warm_{label}", pargs)
                encode_results[f"warm_{label}"] = {k: v for k, v in warm.items() if k != "_output"}
                break

        result["encodeTests"] = encode_results
        result["chosenProfile"] = chosen_label

        if chosen_output is None:
            result["error"] = "All encode profiles failed"
            return result

        result["success"] = True

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as tmp:
            tmp.write(chosen_output)
            tmp_path = tmp.name

        ffprobe_stream = {}
        ffprobe_format = {}
        frames_info = []

        try:
            fp = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams",
                 "-print_format", "json", tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            if fp.returncode == 0:
                fp_data = json.loads(fp.stdout)
                if fp_data.get("streams"):
                    s = fp_data["streams"][0]
                    ffprobe_stream = {
                        "codec_name": s.get("codec_name"),
                        "profile": s.get("profile"),
                        "level": s.get("level"),
                        "pix_fmt": s.get("pix_fmt"),
                        "width": s.get("width"),
                        "height": s.get("height"),
                        "avg_frame_rate": s.get("avg_frame_rate"),
                        "r_frame_rate": s.get("r_frame_rate"),
                        "duration": s.get("duration"),
                        "nb_frames": s.get("nb_frames"),
                        "bit_rate": s.get("bit_rate"),
                    }
                if fp_data.get("format"):
                    f = fp_data["format"]
                    ffprobe_format = {
                        "duration": f.get("duration"),
                        "size": f.get("size"),
                        "bit_rate": f.get("bit_rate"),
                        "nb_streams": f.get("nb_streams"),
                        "format_name": f.get("format_name"),
                    }

            kf = subprocess.run(
                ["ffprobe", "-v", "error", "-show_frames", "-select_streams", "v",
                 "-show_entries", "frame=key_frame,pts_time",
                 "-print_format", "json", tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            if kf.returncode == 0:
                kf_data = json.loads(kf.stdout)
                frames_info = kf_data.get("frames", [])
        except Exception as exc:
            ffprobe_stream = {"error": str(exc)}

        keyframe_pts = [f.get("pts_time") for f in frames_info if f.get("key_frame") == 1 and f.get("pts_time")]
        all_frame_pts = [f.get("pts_time") for f in frames_info if f.get("pts_time")]

        def _parse_boxes(data):
            top_level = []
            moov_count = 0
            moof_count = 0
            mdat_count = 0
            has_mvex = False
            moof_offsets = []

            offset = 0
            while offset + 8 <= len(data):
                size = struct.unpack(">I", data[offset:offset + 4])[0]
                box_type = data[offset + 4:offset + 8].decode("ascii", errors="replace")
                if size == 0:
                    size = len(data) - offset
                elif size == 1:
                    if offset + 16 > len(data):
                        break
                    size = struct.unpack(">Q", data[offset + 8:offset + 16])[0]
                if size < 8:
                    break

                top_level.append({
                    "type": box_type,
                    "offset": offset,
                    "size": size,
                })

                if box_type == "moov":
                    moov_count += 1
                    inner_off = offset + 8
                    inner_end = min(offset + size, len(data))
                    while inner_off + 8 <= inner_end:
                        inner_sz = struct.unpack(">I", data[inner_off:inner_off + 4])[0]
                        inner_tp = data[inner_off + 4:inner_off + 8].decode("ascii", errors="replace")
                        if inner_sz < 8:
                            break
                        if inner_tp == "mvex":
                            has_mvex = True
                        inner_off += inner_sz
                elif box_type == "moof":
                    moof_count += 1
                    moof_offsets.append(offset)
                elif box_type == "mdat":
                    mdat_count += 1

                offset += size

            return {
                "topLevelBoxes": top_level,
                "topLevelBoxCount": len(top_level),
                "moovCount": moov_count,
                "moofCount": moof_count,
                "mdatCount": mdat_count,
                "moofOffsets": moof_offsets,
                "hasMvexInMoov": has_mvex,
            }

        box_analysis = _parse_boxes(chosen_output)
        result["fmp4BoxAnalysis"] = box_analysis

        profile_lower = (ffprobe_stream.get("profile") or "").lower()
        level_val = ffprobe_stream.get("level", 0)
        if isinstance(level_val, str):
            try:
                level_val = int(float(level_val))
            except (ValueError, TypeError):
                level_val = 0

        mse_codec = None
        if "baseline" in profile_lower:
            mse_codec = f"avc1.42E0{level_val:02X}"
        elif "main" in profile_lower:
            mse_codec = f"avc1.4D40{level_val:02X}"
        elif "high" in profile_lower:
            mse_codec = f"avc1.6400{level_val:02X}"

        mse_mime = f'video/mp4; codecs="{mse_codec}"' if mse_codec else None

        frag_count = box_analysis["moofCount"]
        total_dur_s = float(ffprobe_format.get("duration", 0))
        avg_frag_s = total_dur_s / frag_count if frag_count > 0 else 0

        result["ffprobeValidation"] = {
            "stream": ffprobe_stream,
            "format": ffprobe_format,
            "framesEncoded": len(frames_info),
            "keyframeCount": len(keyframe_pts),
            "nonKeyframeCount": len(all_frame_pts) - len(keyframe_pts),
            "keyframePtsTimes": keyframe_pts,
            "gopCadence": f"Every {FPS} frames (1.0s)" if len(keyframe_pts) > 1 else "single_keyframe_or_unknown",
        }

        result["mseCompatibility"] = {
            "candidateMime": mse_mime,
            "candidateCodec": mse_codec,
            "fragmentCount": frag_count,
            "avgFragmentDurationS": round(avg_frag_s, 4),
            "expectedFragmentDurationS": FRAG_DURATION_US / 1_000_000,
            "totalDurationS": round(total_dur_s, 4),
            "hasInitSegment_moov": box_analysis["moovCount"] > 0,
            "hasMediaFragments_moof": box_analysis["moofCount"] > 0,
            "hasMvex": box_analysis["hasMvexInMoov"],
            "isLikelyAppendableToMse": (
                box_analysis["moovCount"] > 0
                and box_analysis["moofCount"] > 0
                and box_analysis["hasMvexInMoov"]
            ),
            "mseAppendStrategy": (
                "1. Append moov box bytes as init segment to SourceBuffer. "
                "2. Append each moof+mdat pair as sequential media segments. "
                "3. mvex in moov signals fragmented track extents to MSE."
                if box_analysis["moovCount"] > 0 and box_analysis["moofCount"] > 0 and box_analysis["hasMvexInMoov"]
                else "Structure may not be suitable for MSE append."
            ),
        }

        fmp4_bytes = len(chosen_output)
        cold_key = f"cold_{chosen_label}"
        warm_key = f"warm_{chosen_label}"
        cold_ms = encode_results.get(cold_key, {}).get("encodeWallMs", 0)
        warm_ms = encode_results.get(warm_key, {}).get("encodeWallMs", 0)

        rec_parts = []
        if fmp4_bytes < jpeg_total_bytes:
            rec_parts.append(f"fMP4 is {round((1 - fmp4_bytes / jpeg_total_bytes) * 100, 0):.0f}% smaller than JPEG")
        if warm_ms and warm_ms < jpeg_encode_ms:
            rec_parts.append(f"fMP4 warm encode {round(warm_ms, 0):.0f}ms vs JPEG {round(jpeg_encode_ms, 0):.0f}ms")
        if box_analysis["hasMvexInMoov"] and box_analysis["moofCount"] > 0:
            rec_parts.append("fMP4 structure is MSE-appendable")

        result["comparisonSummary"] = {
            "jpegTotalBytes": jpeg_total_bytes,
            "jpegBase64TotalChars": jpeg_total_b64_chars,
            "fmp4TotalBytes": fmp4_bytes,
            "fmp4VsJpegBytesRatio": round(fmp4_bytes / jpeg_total_bytes, 3) if jpeg_total_bytes else None,
            "fmp4VsJpegBase64Ratio": round(fmp4_bytes / jpeg_total_b64_chars, 3) if jpeg_total_b64_chars else None,
            "jpegEncodeMs": round(jpeg_encode_ms, 1),
            "fmp4ColdEncodeMs": cold_ms,
            "fmp4WarmEncodeMs": warm_ms,
            "fmp4ColdVsJpegTimeRatio": round(cold_ms / jpeg_encode_ms, 3) if jpeg_encode_ms else None,
            "fmp4WarmVsJpegTimeRatio": round(warm_ms / jpeg_encode_ms, 3) if jpeg_encode_ms else None,
            "bandwidthSavingsVsJpegPct": round((1 - fmp4_bytes / jpeg_total_bytes) * 100, 1) if jpeg_total_bytes else None,
            "bandwidthSavingsVsJpegBase64Pct": round((1 - fmp4_bytes / jpeg_total_b64_chars) * 100, 1) if jpeg_total_b64_chars else None,
            "recommendation": ". ".join(rec_parts) if rec_parts else "See individual metrics",
        }

        artifact_path = "/tmp/avtr1_fmp4_probe_50frames.mp4"
        try:
            with open(artifact_path, "wb") as f:
                f.write(chosen_output)
            artifact_info = {
                "savedTo": artifact_path,
                "sizeBytes": len(chosen_output),
                "sizeKB": round(len(chosen_output) / 1024, 1),
            }
            if len(chosen_output) < 2 * 1024 * 1024:
                artifact_info["base64"] = base64.b64encode(chosen_output).decode("ascii")
                artifact_info["note"] = "Save base64 as .mp4 file and open in browser/player to verify playback"
            else:
                artifact_info["base64Omitted"] = f"File too large ({len(chosen_output)} bytes > 2MB)"
            result["artifact"] = artifact_info
        except Exception as exc:
            result["artifact"] = {"error": str(exc)}

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        return result

    def _do_fmp4_av_probe(self) -> dict:
        import base64
        import json
        import tempfile

        import numpy as np

        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            return {"success": False, "error": "Pipeline not ready (snap_ok=False)"}

        W, H = 1280, 720
        FPS = 25
        N_CHUNKS = 10
        FRAMES_PER_CHUNK = 5
        N_FRAMES = N_CHUNKS * FRAMES_PER_CHUNK
        FRAME_BYTES = H * W * 3 // 2
        FRAG_DURATION_US = 200000
        VIDEO_DURATION_S = N_FRAMES / FPS

        mg = self.pipeline._motion_generator
        step = mg.chunk_size * mg.frame_len
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        AUDIO_SR = 16000
        silence_audio = np.zeros(window, dtype=np.float32)

        result = {
            "success": False,
            "probeConfig": {
                "width": W,
                "height": H,
                "fps": FPS,
                "nChunks": N_CHUNKS,
                "framesPerChunk": FRAMES_PER_CHUNK,
                "nFrames": N_FRAMES,
                "frameBytesI420": FRAME_BYTES,
                "fragDurationUs": FRAG_DURATION_US,
                "videoDurationS": VIDEO_DURATION_S,
                "audioSource": "lavfi_anullsrc",
            },
        }

        t_render = time.monotonic()
        local_state = None
        all_frames = []
        for i in range(N_CHUNKS):
            chunk = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
            local_state, frames_iter = self.pipeline.process_chunk(
                self.avatar, chunk, state=local_state, options=self._options,
            )
            all_frames.extend(frames_iter)
        render_ms = (time.monotonic() - t_render) * 1000

        result["renderPhase"] = {
            "framesRendered": len(all_frames),
            "renderMs": round(render_ms, 1),
            "avgRenderMsPerChunk": round(render_ms / N_CHUNKS, 1),
        }

        if len(all_frames) < N_FRAMES:
            result["error"] = f"Expected {N_FRAMES} frames, got {len(all_frames)}"
            return result

        raw_i420_bytes = b"".join(frame.data.tobytes() for frame in all_frames)

        result["rawData"] = {
            "videoBytes": len(raw_i420_bytes),
            "expectedVideoBytes": N_FRAMES * FRAME_BYTES,
            "videoBytesMatch": len(raw_i420_bytes) == N_FRAMES * FRAME_BYTES,
        }

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "pipe:0",
            "-f", "lavfi", "-i", f"anullsrc=r={AUDIO_SR}:cl=mono",
            "-t", str(VIDEO_DURATION_S),
            "-map", "0:v", "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", str(FRAG_DURATION_US),
            "pipe:1",
        ]

        print(f"[fmp4_av_probe] Starting FFmpeg: video={N_FRAMES} frames ({VIDEO_DURATION_S}s), "
              f"audio=lavfi anullsrc {AUDIO_SR}Hz mono")

        t_enc = time.monotonic()
        try:
            proc = subprocess.run(
                ffmpeg_cmd,
                input=raw_i420_bytes,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired:
            enc_ms = (time.monotonic() - t_enc) * 1000
            result["error"] = f"FFmpeg timed out after {enc_ms:.0f}ms (limit=60s)"
            result["encode"] = {
                "timedOut": True,
                "encodeWallMs": round(enc_ms, 1),
                "cmd": ffmpeg_cmd,
            }
            print(f"[fmp4_av_probe] FFmpeg TIMED OUT after {enc_ms:.0f}ms")
            return result
        enc_ms = (time.monotonic() - t_enc) * 1000

        stdout_data = proc.stdout
        stderr_text = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""

        result["encode"] = {
            "returncode": proc.returncode,
            "outputBytes": len(stdout_data),
            "encodeWallMs": round(enc_ms, 1),
            "avgEncodeMsPerFrame": round(enc_ms / N_FRAMES, 2) if proc.returncode == 0 else None,
            "compressionRatio": round(len(raw_i420_bytes) / len(stdout_data), 2) if stdout_data else None,
            "ffmpegStderrTail": stderr_text[-1200:] if stderr_text else None,
            "cmd": ffmpeg_cmd,
        }

        print(f"[fmp4_av_probe] FFmpeg done: rc={proc.returncode} stdout={len(stdout_data)}B "
              f"wallMs={enc_ms:.0f}")

        if proc.returncode != 0 or len(stdout_data) == 0:
            result["error"] = f"FFmpeg failed: rc={proc.returncode} stdout={len(stdout_data)}B"
            result["encode"]["ffmpegStderrLast10"] = stderr_text.split("\n")[-10:]
            return result

        result["success"] = True

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as tmp:
            tmp.write(stdout_data)
            tmp_path = tmp.name

        ffprobe_streams = []
        ffprobe_format = {}

        try:
            fp = subprocess.run(
                ["ffprobe", "-v", "error", "-show_format", "-show_streams",
                 "-print_format", "json", tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            if fp.returncode == 0:
                fp_data = json.loads(fp.stdout)
                for s in fp_data.get("streams", []):
                    ffprobe_streams.append({
                        "codec_type": s.get("codec_type"),
                        "codec_name": s.get("codec_name"),
                        "profile": s.get("profile"),
                        "level": s.get("level"),
                        "pix_fmt": s.get("pix_fmt"),
                        "width": s.get("width"),
                        "height": s.get("height"),
                        "sample_rate": s.get("sample_rate"),
                        "channels": s.get("channels"),
                        "avg_frame_rate": s.get("avg_frame_rate"),
                        "duration": s.get("duration"),
                        "nb_frames": s.get("nb_frames"),
                        "bit_rate": s.get("bit_rate"),
                    })
                if fp_data.get("format"):
                    f = fp_data["format"]
                    ffprobe_format = {
                        "duration": f.get("duration"),
                        "size": f.get("size"),
                        "bit_rate": f.get("bit_rate"),
                        "nb_streams": f.get("nb_streams"),
                        "format_name": f.get("format_name"),
                    }
        except Exception as exc:
            ffprobe_streams = [{"error": str(exc)}]

        video_stream = next((s for s in ffprobe_streams if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in ffprobe_streams if s.get("codec_type") == "audio"), {})

        result["ffprobe"] = {
            "streams": ffprobe_streams,
            "format": ffprobe_format,
            "videoStream": video_stream,
            "audioStream": audio_stream,
            "trackCount": len(ffprobe_streams),
            "hasVideo": bool(video_stream),
            "hasAudio": bool(audio_stream),
        }

        result["mseCompatibility"] = {
            "codecString": 'video/mp4; codecs="avc1.42E01F,mp4a.40.2"',
            "videoCodec": "avc1.42E01F",
            "audioCodec": "mp4a.40.2",
            "videoCodecMatch": video_stream.get("codec_name") == "h264",
            "audioCodecMatch": audio_stream.get("codec_name") == "aac",
            "readyForAvMode": (
                video_stream.get("codec_name") == "h264"
                and audio_stream.get("codec_name") == "aac"
                and int(video_stream.get("width", 0)) == W
                and int(video_stream.get("height", 0)) == H
                and len(ffprobe_streams) == 2
            ),
        }

        artifact_path = "/tmp/avtr1_fmp4_av_probe_2s.mp4"
        try:
            with open(artifact_path, "wb") as f:
                f.write(stdout_data)
            artifact_info = {
                "savedTo": artifact_path,
                "sizeBytes": len(stdout_data),
                "sizeKB": round(len(stdout_data) / 1024, 1),
            }
            if len(stdout_data) < 4 * 1024 * 1024:
                artifact_info["base64"] = base64.b64encode(stdout_data).decode("ascii")
                artifact_info["note"] = "Save base64 as .mp4 file and open in browser/player to verify A/V playback"
            result["artifact"] = artifact_info
        except Exception as exc:
            result["artifact"] = {"error": str(exc)}

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        return result

    @modal.method()
    def render_debug_frame(self, session_id: str | None = None) -> dict:
        return self._do_render_frame(session_id)

    @modal.method()
    def render_audio(
        self,
        session_id: str | None = None,
        audio_base64: str | None = None,
        max_duration_s: float = 60.0,
        jpeg_quality: int = 80,
    ) -> dict:
        return self._do_render_audio(session_id, audio_base64, max_duration_s, jpeg_quality)

    @modal.method()
    def render_idle(
        self,
        session_id: str | None = None,
        duration_s: float = 3.0,
        jpeg_quality: int = 75,
    ) -> dict:
        return self._do_render_idle(session_id, duration_s, jpeg_quality)

    @modal.method()
    def render_idle_chunk(
        self,
        session_id: str | None = None,
        duration_s: float = 0.5,
        jpeg_quality: int = 75,
    ) -> dict:
        return self._do_render_idle_chunk(session_id, duration_s, jpeg_quality)

    @modal.method()
    def end_session(self, session_id: str | None = None) -> dict:
        return self._do_end_session(session_id)

    @modal.method()
    def fmp4_av_probe(self) -> dict:
        return self._do_fmp4_av_probe()

    @modal.method()
    def fmp4_av_live_smoke(self) -> dict:
        return self._do_fmp4_av_live_smoke()

    def _do_fmp4_av_live_smoke(self) -> dict:
        import json
        import numpy as np

        result = {
            "success": False,
            "timestamp": time.time(),
            "steps": {},
        }

        started_session = False
        if not getattr(self, "_session_active", False):
            print("[l40s-live_smoke] no active session, starting one...")
            start_result = self._do_start_session()
            if start_result.get("status") != "avatar_ready":
                result["error"] = f"Failed to start session: {start_result}"
                return result
            started_session = True
            print(f"[l40s-live_smoke] session started: {start_result['sessionId']}")

        session_id = self._session_id
        result["sessionId"] = session_id
        result["startedSession"] = started_session

        SMOKE_TIMEOUT_S = 15

        t_smoke_start = time.monotonic()
        smoke_deadline = t_smoke_start + SMOKE_TIMEOUT_S

        saw_init = False
        saw_media = False
        time_to_init_ms = None
        first_media_byte_length = None
        events_seen = []
        init_event = None
        first_media_event = None

        try:
            print(f"[l40s-live_smoke] starting timeline_stream_fmp4_av generator, timeout={SMOKE_TIMEOUT_S}s")
            gen = self._do_timeline_stream_fmp4_av(session_id=session_id)

            for raw_sse in gen:
                if time.monotonic() > smoke_deadline:
                    print(f"[l40s-live_smoke] TIMEOUT: exceeded {SMOKE_TIMEOUT_S}s")
                    break

                stripped = raw_sse.strip()
                if not stripped.startswith("data: "):
                    continue
                json_str = stripped[6:]
                try:
                    evt = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                evt_type = evt.get("type", "?")
                events_seen.append(evt_type)
                print(f"[l40s-live_smoke] event: {evt_type}")

                if evt_type == "error":
                    result["error"] = evt.get("error", "unknown error from stream")
                    print(f"[l40s-live_smoke] stream error: {result['error']}")
                    break

                if evt_type == "fmp4_init":
                    saw_init = True
                    time_to_init_ms = evt.get("timing", {}).get("timeToInitMs")
                    first_media_byte_length = evt.get("byteLength")
                    init_event = {k: v for k, v in evt.items() if k != "dataBase64"}
                    print(f"[l40s-live_smoke] INIT received: timeToInitMs={time_to_init_ms} byteLength={evt.get('byteLength')}")

                if evt_type == "fmp4_media":
                    saw_media = True
                    if first_media_event is None:
                        first_media_event = {k: v for k, v in evt.items() if k != "dataBase64"}
                        first_media_byte_length = evt.get("byteLength")
                        print(f"[l40s-live_smoke] FIRST MEDIA received: byteLength={evt.get('byteLength')} sequence={evt.get('sequence')}")
                    if saw_init:
                        print("[l40s-live_smoke] init + first media seen, stopping generator")
                        break

            gen.close()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result["error"] = f"Smoke test exception: {exc}"

        smoke_elapsed_ms = (time.monotonic() - t_smoke_start) * 1000

        self._do_timeline_stop({"session_id": session_id})

        result["steps"]["stream"] = {
            "sawInit": saw_init,
            "sawMedia": saw_media,
            "timeToInitMs": time_to_init_ms,
            "firstMediaByteLength": first_media_byte_length,
            "smokeElapsedMs": round(smoke_elapsed_ms, 1),
            "eventsSeen": events_seen,
            "initEvent": init_event,
            "firstMediaEvent": first_media_event,
        }

        if started_session:
            self._do_end_session(session_id)
            print(f"[l40s-live_smoke] ended session {session_id}")
            result["endedSession"] = True

        result["success"] = saw_init and saw_media
        result["smokeElapsedMs"] = round(smoke_elapsed_ms, 1)

        print(f"[l40s-live_smoke] RESULT: success={result['success']} init={saw_init} media={saw_media} "
              f"timeToInitMs={time_to_init_ms} elapsedMs={result['smokeElapsedMs']}")

        return result

    def _do_ffmpeg_fragment_cadence_probe(self) -> dict:
        """Probe actual FFmpeg fMP4 fragment emission cadence for A/V muxed output.

        Tests multiple FFmpeg flag variants, renders real idle frames from AVTR pipeline,
        feeds them + silence audio through FFmpeg, and measures actual moof+mdat segment
        emission timing from stdout.

        Returns per-variant stats: segment count, inter-segment gaps, bytes per segment,
        init time, and approximate media duration per segment.
        """
        import queue
        import struct
        import threading

        import numpy as np
        from avtr1_renderer.types import Chunk

        if not getattr(self, "_snap_ok", False):
            return {"success": False, "error": "Pipeline not ready (snap_ok=False)"}

        W, H = 1280, 720
        FPS = 25
        FRAME_BYTES = H * W * 3 // 2
        AUDIO_SR = 16000
        N_CHUNKS = 50
        FRAMES_PER_CHUNK = 5

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        silence_audio = np.zeros(window, dtype=np.float32)

        print(f"[frag_probe] Rendering {N_CHUNKS} idle chunks ({N_CHUNKS * FRAMES_PER_CHUNK} frames, "
              f"{N_CHUNKS * FRAMES_PER_CHUNK / FPS:.1f}s)...")

        t_render_start = time.monotonic()
        all_frames = []
        local_state = None
        for i in range(N_CHUNKS):
            chunk = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
            local_state, frames_iter = self.pipeline.process_chunk(
                self.avatar if not getattr(self, '_session_active', False) else self._session_avatar,
                chunk, state=local_state, options=self._options,
            )
            all_frames.extend(frames_iter)
        render_ms = (time.monotonic() - t_render_start) * 1000
        print(f"[frag_probe] Rendered {len(all_frames)} frames in {render_ms:.0f}ms")

        if len(all_frames) < N_CHUNKS * FRAMES_PER_CHUNK:
            return {"success": False, "error": f"Only rendered {len(all_frames)} frames, expected {N_CHUNKS * FRAMES_PER_CHUNK}"}

        raw_i420_bytes = b"".join(frame.data.tobytes() for frame in all_frames)
        video_duration_s = len(all_frames) / FPS

        silence_s16 = np.zeros(step, dtype=np.int16).tobytes()
        audio_total_s = N_CHUNKS * step / AUDIO_SR
        audio_bytes = silence_s16 * N_CHUNKS

        variants = {
            "A_current_g25_frag_keyframe": {
                "g": str(int(FPS)),
                "movflags": "+frag_keyframe+empty_moov+default_base_moof",
                "frag_duration": "200000",
                "label": "Current: -g 25 +frag_keyframe -frag_duration 200000",
            },
            "B_g5_frag_keyframe": {
                "g": str(FRAMES_PER_CHUNK),
                "movflags": "+frag_keyframe+empty_moov+default_base_moof",
                "frag_duration": "200000",
                "label": "Variant B: -g 5 +frag_keyframe -frag_duration 200000",
            },
            "C_g25_frag_duration_only": {
                "g": str(int(FPS)),
                "movflags": "+empty_moov+default_base_moof",
                "frag_duration": "200000",
                "label": "Variant C: -g 25 no frag_keyframe, -frag_duration 200000",
            },
        }

        variant_results = {}

        for variant_key, vcfg in variants.items():
            print(f"\n[frag_probe] === Testing variant {variant_key}: {vcfg['label']} ===")

            fifo_path = f"/tmp/avtr1_frag_probe_{variant_key}_{os.getpid()}"
            try:
                os.unlink(fifo_path)
            except FileNotFoundError:
                pass
            os.mkfifo(fifo_path)

            ffmpeg_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-thread_queue_size", "512",
                "-f", "rawvideo", "-pix_fmt", "yuv420p",
                "-s", f"{W}x{H}", "-r", str(FPS),
                "-i", "pipe:0",
                "-thread_queue_size", "512",
                "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "1",
                "-i", fifo_path,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "h264_nvenc",
                "-preset", "p4", "-tune", "ull",
                "-g", vcfg["g"], "-bf", "0",
                "-profile:v", "baseline", "-level", "3.1",
                "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
                "-f", "mp4",
                "-movflags", vcfg["movflags"],
                "-frag_duration", vcfg["frag_duration"],
                "pipe:1",
            ]

            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            seg_queue: queue.Queue[tuple | None] = queue.Queue(maxsize=200)
            audio_writer_error: list[str] = []

            def _audio_writer():
                try:
                    with open(fifo_path, "wb") as af:
                        chunk_size = step * 2
                        offset = 0
                        while offset < len(audio_bytes):
                            end = min(offset + chunk_size, len(audio_bytes))
                            af.write(audio_bytes[offset:end])
                            af.flush()
                            offset = end
                except OSError as exc:
                    audio_writer_error.append(str(exc))

            def _stdout_reader():
                buf = bytearray()
                init_collected = bytearray()
                init_sent = False
                media_buf = bytearray()
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break
                    buf.extend(chunk)
                    while len(buf) >= 8:
                        box_size = struct.unpack(">I", bytes(buf[0:4]))[0]
                        box_type = bytes(buf[4:8])
                        if box_size < 8 or len(buf) < box_size:
                            break
                        box_data = bytes(buf[:box_size])
                        buf = buf[box_size:]

                        if not init_sent:
                            init_collected.extend(box_data)
                            if box_type == b"moov":
                                init_sent = True
                                seg_queue.put(("init", bytes(init_collected), time.monotonic()))
                                init_collected = bytearray()
                        elif box_type == b"moof":
                            media_buf = bytearray(box_data)
                        elif box_type == b"mdat" and len(media_buf) > 0:
                            media_buf.extend(box_data)
                            seg_queue.put(("media", bytes(media_buf), time.monotonic()))
                            media_buf = bytearray()
                seg_queue.put(None)

            audio_thread = threading.Thread(target=_audio_writer, daemon=True)
            audio_thread.start()
            stdout_thread = threading.Thread(target=_stdout_reader, daemon=True)
            stdout_thread.start()

            t_feed_start = time.monotonic()

            chunk_size_bytes = FRAMES_PER_CHUNK * FRAME_BYTES
            for i in range(0, len(raw_i420_bytes), chunk_size_bytes):
                piece = raw_i420_bytes[i:i + chunk_size_bytes]
                try:
                    proc.stdin.write(piece)
                except BrokenPipeError:
                    break
            try:
                proc.stdin.close()
            except BrokenPipeError:
                pass

            segments_info = []
            init_time = None
            eof = False
            deadline = time.monotonic() + 30

            while not eof and time.monotonic() < deadline:
                try:
                    item = seg_queue.get(timeout=2)
                except queue.Empty:
                    break
                if item is None:
                    eof = True
                    break
                seg_type, seg_data, seg_mono = item
                if seg_type == "init":
                    init_time = seg_mono - t_feed_start
                elif seg_type == "media":
                    segments_info.append({
                        "monoTime": seg_mono,
                        "size": len(seg_data),
                        "wallFromStart": seg_mono - t_feed_start,
                    })

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)

            try:
                os.unlink(fifo_path)
            except OSError:
                pass

            audio_thread.join(timeout=3)
            stdout_thread.join(timeout=3)

            rc = proc.returncode
            stderr_output = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            stderr_errors = [l for l in stderr_output.split("\n") if l.strip() and "error" in l.lower()]

            n_media = len(segments_info)
            inter_gaps_ms = []
            for j in range(1, n_media):
                gap = (segments_info[j]["monoTime"] - segments_info[j - 1]["monoTime"]) * 1000
                inter_gaps_ms.append(gap)

            if n_media > 0:
                estimated_media_duration_ms = (video_duration_s * 1000) / n_media
                avg_size = sum(s["size"] for s in segments_info) / n_media
                first_wall = segments_info[0]["wallFromStart"]
                last_wall = segments_info[-1]["wallFromStart"]
                total_emit_span_ms = (last_wall - first_wall) * 1000 if n_media > 1 else 0
            else:
                estimated_media_duration_ms = 0
                avg_size = 0
                first_wall = 0
                last_wall = 0
                total_emit_span_ms = 0

            variant_results[variant_key] = {
                "label": vcfg["label"],
                "ffmpegCmd": " ".join(ffmpeg_cmd),
                "returnCode": rc,
                "stderrErrorCount": len(stderr_errors),
                "stderrErrors": stderr_errors[:5],
                "inputDurationS": round(video_duration_s, 2),
                "initTimeMs": round(init_time * 1000, 1) if init_time is not None else None,
                "mediaSegmentCount": n_media,
                "estimatedMediaDurationMs": round(estimated_media_duration_ms, 1),
                "avgSegmentBytes": round(avg_size, 0),
                "totalEncodedBytes": sum(s["size"] for s in segments_info),
                "firstSegmentWallMs": round(first_wall * 1000, 1),
                "lastSegmentWallMs": round(last_wall * 1000, 1),
                "totalEmitSpanMs": round(total_emit_span_ms, 1),
                "interSegmentGapMs": {
                    "min": round(min(inter_gaps_ms), 1) if inter_gaps_ms else None,
                    "max": round(max(inter_gaps_ms), 1) if inter_gaps_ms else None,
                    "avg": round(sum(inter_gaps_ms) / len(inter_gaps_ms), 1) if inter_gaps_ms else None,
                    "p50": round(sorted(inter_gaps_ms)[len(inter_gaps_ms) // 2], 1) if inter_gaps_ms else None,
                    "count": len(inter_gaps_ms),
                    "all": [round(g, 1) for g in inter_gaps_ms],
                },
                "expectedChunksPerSegment": round(estimated_media_duration_ms / 200, 2) if estimated_media_duration_ms > 0 else None,
                "oneChunkPerSegment": abs(estimated_media_duration_ms - 200) < 50 if estimated_media_duration_ms > 0 else False,
            }

            print(f"[frag_probe] {variant_key}: segments={n_media} estDurMs={estimated_media_duration_ms:.0f} "
                  f"initMs={init_time * 1000 if init_time else '?'} "
                  f"gapMin={min(inter_gaps_ms):.0f} gapMax={max(inter_gaps_ms):.0f} "
                  f"gapAvg={sum(inter_gaps_ms) / len(inter_gaps_ms):.0f}" if inter_gaps_ms else f"gap=N/A")

        best_key = None
        best_score = -1
        for vk, vr in variant_results.items():
            if vr["returnCode"] != 0:
                continue
            score = 0
            if vr["oneChunkPerSegment"]:
                score += 100
            avg_gap = vr["interSegmentGapMs"].get("avg", 9999) or 9999
            if 150 <= avg_gap <= 300:
                score += 50
            max_gap = vr["interSegmentGapMs"].get("max", 99999) or 99999
            if max_gap < 500:
                score += 30
            score -= vr["stderrErrorCount"] * 10
            vr["score"] = score
            if score > best_score:
                best_score = score
                best_key = vk

        return {
            "success": True,
            "probeConfig": {
                "width": W,
                "height": H,
                "fps": FPS,
                "nChunks": N_CHUNKS,
                "framesPerChunk": FRAMES_PER_CHUNK,
                "videoDurationS": round(video_duration_s, 2),
                "audioDurationS": round(audio_total_s, 2),
                "renderMs": round(render_ms, 1),
            },
            "variants": variant_results,
            "recommendedVariant": best_key,
            "recommendedLabel": variant_results.get(best_key, {}).get("label"),
        }

    def _do_idle_60s_smoke(self) -> dict:
        import json
        import time as _time

        result = {
            "success": False,
            "timestamp": time.time(),
            "steps": {},
        }

        started_session = False
        if not getattr(self, "_session_active", False):
            print("[l40s-idle_60s] no active session, starting one...")
            start_result = self._do_start_session()
            if start_result.get("status") != "avatar_ready":
                result["error"] = f"Failed to start session: {start_result}"
                return result
            started_session = True
            print(f"[l40s-idle_60s] session started: {start_result['sessionId']}")

        session_id = self._session_id
        result["sessionId"] = session_id
        result["startedSession"] = started_session

        SMOKE_TIMEOUT_S = 65
        t_smoke_start = _time.monotonic()
        smoke_deadline = t_smoke_start + SMOKE_TIMEOUT_S

        media_count = 0
        init_seen = False
        error_events = []
        first_media_at = None
        last_media_at = None
        last_sequence = None
        byte_lengths = []

        try:
            print(f"[l40s-idle_60s] starting timeline_stream_fmp4_av, timeout={SMOKE_TIMEOUT_S}s")
            gen = self._do_timeline_stream_fmp4_av(session_id=session_id)

            for raw_sse in gen:
                if _time.monotonic() > smoke_deadline:
                    print(f"[l40s-idle_60s] TIMEOUT: exceeded {SMOKE_TIMEOUT_S}s")
                    break

                stripped = raw_sse.strip()
                if not stripped.startswith("data: "):
                    continue
                json_str = stripped[6:]
                try:
                    evt = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                evt_type = evt.get("type", "?")

                if evt_type == "error":
                    error_events.append(evt.get("error", "unknown"))
                    print(f"[l40s-idle_60s] stream error: {evt.get('error')}")
                    break

                if evt_type == "fmp4_init":
                    init_seen = True
                    print(f"[l40s-idle_60s] INIT received: byteLength={evt.get('byteLength')}")

                if evt_type == "fmp4_media":
                    media_count += 1
                    now_mono = _time.monotonic()
                    if first_media_at is None:
                        first_media_at = now_mono
                    last_media_at = now_mono
                    last_sequence = evt.get("sequence")
                    bl = evt.get("byteLength", 0)
                    byte_lengths.append(bl)
                    if media_count % 50 == 0:
                        print(f"[l40s-idle_60s] media_count={media_count} seq={last_sequence} byteLen={bl}")

            gen.close()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result["error"] = f"Smoke exception: {exc}"

        smoke_elapsed_ms = (_time.monotonic() - t_smoke_start) * 1000

        self._do_timeline_stop({"session_id": session_id})

        result["steps"]["modalStream"] = {
            "initSeen": init_seen,
            "mediaSegments": media_count,
            "lastSequence": last_sequence,
            "errorEvents": error_events[:5] if error_events else [],
            "smokeElapsedMs": round(smoke_elapsed_ms, 1),
            "firstMediaAtMs": round((first_media_at - t_smoke_start) * 1000, 1) if first_media_at else None,
            "lastMediaAtMs": round((last_media_at - t_smoke_start) * 1000, 1) if last_media_at else None,
            "avgByteLength": round(sum(byte_lengths) / len(byte_lengths), 1) if byte_lengths else 0,
            "totalEncodedBytes": sum(byte_lengths),
        }

        if started_session:
            self._do_end_session(session_id)
            print(f"[l40s-idle_60s] ended session {session_id}")
            result["endedSession"] = True

        result["success"] = init_seen and media_count >= 250
        result["smokeElapsedMs"] = round(smoke_elapsed_ms, 1)

        print(f"[l40s-idle_60s] RESULT: success={result['success']} init={init_seen} "
              f"media={media_count} lastSeq={last_sequence} elapsedMs={result['smokeElapsedMs']}")

        return result

    def _do_tts_same_gpu_parallel(self, body: dict) -> dict:
        import io
        import time as _time

        import numpy as np
        import torch

        session_id = body.get("session_id")
        text = body.get("text", "")
        target_sr = int(body.get("target_sample_rate", 16000))
        enqueue_to_timeline = bool(body.get("enqueueToTimeline", False))
        turn_id = body.get("turnId")

        if not text:
            return {"ok": False, "error": "No text provided"}
        if not getattr(self, "_omnivoice_model", None):
            return {"ok": False, "error": "OmniVoice model not loaded"}
        if session_id and session_id != getattr(self, "_session_id", None):
            print(f"[l40s-tts_parallel] WARNING: session mismatch got={session_id} "
                  f"expected={self._session_id}, proceeding anyway")

        container_id = getattr(self, "_container_instance_id", "unknown")

        if enqueue_to_timeline and (not session_id or turn_id is None):
            return {"ok": False, "error": "enqueueToTimeline requires session_id and turnId"}

        self._tts_overlap_active = True
        self._tts_overlap_start_mono = _time.monotonic()

        _run_nvidia_smi("tts_parallel/before_inference")
        torch.cuda.reset_peak_memory_stats()

        try:
            if enqueue_to_timeline:
                print(f"[l40s-tts_parallel] chunked handoff requested session={session_id} turnId={turn_id}")

            t_generate_start = _time.time()
            with torch.inference_mode():
                audio = self._omnivoice_model.generate(
                    text,
                    preprocess_prompt=True,
                    postprocess_output=True,
                    denoise=True,
                    t_shift=0.1,
                    layer_penalty_factor=5.0,
                    position_temperature=5.0,
                    class_temperature=0.0,
                )
            generate_ms = (_time.time() - t_generate_start) * 1000

            audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().float()
            if audio_data.dim() > 1:
                audio_data = audio_data.squeeze(0)
            if audio_data.dim() != 1:
                return {"ok": False, "error": f"Unexpected audio shape: {audio_data.shape}"}

            audio_dur_s = audio_data.shape[0] / self._omnivoice_sr
            audio_dur_ms = audio_dur_s * 1000

            t_resample_start = _time.time()
            enqueue_sr = 16000 if enqueue_to_timeline else target_sr
            if self._omnivoice_sr != enqueue_sr:
                import torchaudio.transforms as T
                resampler = T.Resample(
                    self._omnivoice_sr, enqueue_sr, dtype=audio_data.dtype
                )
                audio_data = resampler(audio_data)
                actual_sr = enqueue_sr
            else:
                actual_sr = self._omnivoice_sr
            resample_ms = (_time.time() - t_resample_start) * 1000

            t_postprocess_start = _time.time()
            audio_np = audio_data.cpu().numpy()
            if not np.all(np.isfinite(audio_np)):
                nan_count = int(np.sum(~np.isfinite(audio_np)))
                print(f"[l40s-tts_parallel] WARNING: {nan_count} non-finite samples, clipping")
                audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)
            postprocess_ms = (_time.time() - t_postprocess_start) * 1000

            validated_dur_ms = (len(audio_np) / actual_sr) * 1000
            rtf = audio_dur_ms / generate_ms if generate_ms > 0 else 0
            peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
            _run_nvidia_smi("tts_parallel/after_inference")
            gpu_snap = _gpu_memory_snapshot("tts_parallel/after")

            if enqueue_to_timeline:
                t_handoff_start = _time.time()
                mg = self.pipeline._motion_generator
                chunk_step = mg.chunk_size * mg.frame_len  # 3200 at 16kHz

                n_chunks = max(1, int(np.ceil(len(audio_np) / chunk_step)))
                first_chunk_available_ms = None

                with self._audio_handoff_lock:
                    handoff = self._audio_handoffs.get(session_id)

                if handoff is None:
                    return {
                        "ok": False,
                        "error": "No audio handoff for session (session may have ended)",
                        "container_instance_id": container_id,
                    }

                for ci in range(n_chunks):
                    s = ci * chunk_step
                    e = min(s + chunk_step, len(audio_np))
                    chunk_pcm = audio_np[s:e].copy()
                    if len(chunk_pcm) < chunk_step:
                        chunk_pcm = np.pad(chunk_pcm, (0, chunk_step - len(chunk_pcm)))

                    if not handoff.push((
                        chunk_pcm,
                        {
                            "turnId": turn_id,
                            "chunkIndex": ci,
                            "isFirst": ci == 0,
                            "isLast": ci == n_chunks - 1,
                            "audioDurationMs": round(len(chunk_pcm) / actual_sr * 1000),
                            "sampleRate": actual_sr,
                            "generatedAt": _time.time(),
                            "textLength": len(text),
                        },
                    ), timeout=30.0):
                        return {
                            "ok": False,
                            "error": "Audio handoff closed or queue full (session may have ended)",
                            "container_instance_id": container_id,
                            "chunksPushed": ci,
                        }

                    if first_chunk_available_ms is None:
                        first_chunk_available_ms = (_time.time() - t_generate_start) * 1000

                handoff_ms = (_time.time() - t_handoff_start) * 1000

                overlap_ms = (_time.monotonic() - self._tts_overlap_start_mono) * 1000
                total_tts_ms = (_time.time() - t_generate_start) * 1000

                print(
                    f"[l40s-tts_parallel] chunked handoff turnId={turn_id} "
                    f"chunks={n_chunks} durationMs={validated_dur_ms:.0f} "
                    f"generate={generate_ms:.0f}ms resample={resample_ms:.0f}ms "
                    f"handoff={handoff_ms:.0f}ms total={total_tts_ms:.0f}ms "
                    f"container={container_id}"
                )

                result = {
                    "ok": True,
                    "enqueued": True,
                    "gpuTarget": "l40s",
                    "streamingHandoff": True,
                    "chunkCount": n_chunks,
                    "sample_rate": actual_sr,
                    "duration_ms": round(audio_dur_ms),
                    "validated_duration_ms": round(validated_dur_ms),
                    "latency_ms": round(generate_ms),
                    "rtf": round(rtf, 2),
                    "text_length": len(text),
                    "provider": "omnivoice_same_gpu_parallel",
                    "peak_vram_gb": round(peak_gb, 3),
                    "overlap_ms": round(overlap_ms, 1),
                    "container_instance_id": container_id,
                    "gpu": {k: v for k, v in gpu_snap.items() if k != "label"},
                    "timing": {
                        "generateMs": round(generate_ms, 1),
                        "resampleMs": round(resample_ms, 1),
                        "postprocessMs": round(postprocess_ms, 1),
                        "handoffPushMs": round(handoff_ms, 1),
                        "firstChunkAvailableMs": round(first_chunk_available_ms or 0, 1),
                        "totalTtsMs": round(total_tts_ms, 1),
                    },
                }
                return result

            import base64
            import soundfile as sf

            buf = io.BytesIO()
            sf.write(buf, audio_np, actual_sr, subtype="PCM_16", format="WAV")
            buf.seek(0)
            wav_bytes = buf.getvalue()
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")

            overlap_ms = (_time.monotonic() - self._tts_overlap_start_mono) * 1000
            total_tts_ms = (_time.time() - t_generate_start) * 1000

            print(
                f"[l40s-tts_parallel] standalone inference={generate_ms:.0f}ms "
                f"audio={audio_dur_ms:.0f}ms RTF={rtf:.2f}x "
                f"wav={len(wav_bytes)}B overlap={overlap_ms:.0f}ms "
                f"container={container_id}"
            )

            result = {
                "ok": True,
                "wav_base64": wav_b64,
                "sample_rate": actual_sr,
                "duration_ms": round(audio_dur_ms),
                "validated_duration_ms": round(validated_dur_ms),
                "latency_ms": round(generate_ms),
                "rtf": round(rtf, 2),
                "output_bytes": len(wav_bytes),
                "text_length": len(text),
                "provider": "omnivoice_same_gpu_parallel",
                "peak_vram_gb": round(peak_gb, 3),
                "overlap_ms": round(overlap_ms, 1),
                "container_instance_id": container_id,
                "gpu": {k: v for k, v in gpu_snap.items() if k != "label"},
                "timing": {
                    "generateMs": round(generate_ms, 1),
                    "resampleMs": round(resample_ms, 1),
                    "postprocessMs": round(postprocess_ms, 1),
                    "handoffPushMs": 0,
                    "firstChunkAvailableMs": round(total_tts_ms, 1),
                    "totalTtsMs": round(total_tts_ms, 1),
                },
            }
            return result
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return {
                "ok": False,
                "error": str(exc),
                "container_instance_id": container_id,
            }
        finally:
            self._tts_overlap_active = False
            self._tts_overlap_start_mono = None

    def _do_tts_handoff_smoke(self, body: dict) -> dict:
        import json
        import threading

        result = {
            "success": False,
            "timestamp": time.time(),
            "steps": {},
        }

        long_text = body.get("text", (
            "The quick brown fox jumps over the lazy dog. "
            "In a world where technology advances at an unprecedented pace, "
            "artificial intelligence continues to reshape every industry from healthcare to finance. "
            "Researchers around the globe are working tirelessly to develop systems that can understand, "
            "reason, and communicate with human-like fluency. "
            "The convergence of large language models, neural rendering, and real-time speech synthesis "
            "has opened up entirely new possibilities for interactive digital avatars. "
            "These avatars can now hold natural conversations, display realistic facial expressions, "
            "and respond in real-time to user input. "
            "The implications for education, entertainment, customer service, and accessibility are profound. "
            "As we continue to push the boundaries of what is possible, "
            "we must also consider the ethical implications and ensure that these technologies "
            "are developed and deployed responsibly. "
            "This is a long text specifically designed to test the chunked in-process audio handoff "
            "without exceeding the one megabyte modal queue item limit. "
            "Each chunk should be approximately twelve kilobytes of raw PCM float32 data, "
            "well within safe limits. "
            "The total audio duration should exceed five seconds to verify multiple chunks flow correctly."
        ))

        started_session = False
        if not getattr(self, "_session_active", False):
            print("[l40s-handoff_smoke] no active session, starting one...")
            start_result = self._do_start_session()
            if start_result.get("status") != "avatar_ready":
                result["error"] = f"Failed to start session: {start_result}"
                return result
            started_session = True
            print(f"[l40s-handoff_smoke] session started: {start_result['sessionId']}")

        session_id = self._session_id
        result["sessionId"] = session_id
        result["startedSession"] = started_session

        speech_segments_seen = 0
        first_speech_segment_latency_ms = None
        pacing_sleep_ms_list = []
        media_lead_ms_list = []
        tts_result_data = None
        stream_error = None

        SMOKE_TIMEOUT_S = 45
        t_smoke_start = time.monotonic()

        def _tts_thread():
            nonlocal tts_result_data
            time.sleep(2.0)
            print(f"[l40s-handoff_smoke] launching TTS in background thread after 2s idle...")
            tts_result_data = self._do_tts_same_gpu_parallel({
                "action": "tts_same_gpu_parallel",
                "text": long_text,
                "session_id": session_id,
                "target_sample_rate": 16000,
                "enqueueToTimeline": True,
                "turnId": 99,
            })
            print(f"[l40s-handoff_smoke] TTS completed: ok={tts_result_data.get('ok')}")

        tts_thread = threading.Thread(target=_tts_thread, daemon=True)
        tts_thread.start()

        try:
            gen = self._do_timeline_stream_fmp4_av(session_id=session_id)
            for raw_sse in gen:
                if time.monotonic() - t_smoke_start > SMOKE_TIMEOUT_S:
                    print(f"[l40s-handoff_smoke] TIMEOUT after {SMOKE_TIMEOUT_S}s")
                    break

                stripped = raw_sse.strip()
                if not stripped.startswith("data: "):
                    continue
                json_str = stripped[6:]
                try:
                    evt = json.loads(json_str)
                except json.JSONDecodeError:
                    continue

                evt_type = evt.get("type", "?")

                if evt_type == "error":
                    stream_error = evt.get("error", "unknown")
                    print(f"[l40s-handoff_smoke] stream error: {stream_error}")
                    break

                if evt_type == "fmp4_media":
                    mode = evt.get("mode", "idle")
                    timing = evt.get("timing", {})

                    if mode == "speech":
                        speech_segments_seen += 1
                        if first_speech_segment_latency_ms is None:
                            first_speech_segment_latency_ms = (time.monotonic() - t_smoke_start) * 1000

                        p_sleep = timing.get("pacingSleepMs", 0)
                        p_lead = timing.get("mediaLeadMs", 0)
                        if p_sleep > 0:
                            pacing_sleep_ms_list.append(p_sleep)
                        if p_lead > 0:
                            media_lead_ms_list.append(p_lead)

                        if evt.get("isSpeechEnd"):
                            print(f"[l40s-handoff_smoke] speech end seen, speech_segments={speech_segments_seen}")
                            break

            gen.close()

        except Exception as exc:
            import traceback
            traceback.print_exc()
            result["error"] = f"Smoke exception: {exc}"

        tts_thread.join(timeout=5)

        self._do_timeline_stop({"session_id": session_id})

        smoke_elapsed_ms = (time.monotonic() - t_smoke_start) * 1000

        result["steps"]["tts"] = tts_result_data
        result["steps"]["stream"] = {
            "speechSegmentsSeen": speech_segments_seen,
            "firstSpeechSegmentLatencyMs": round(first_speech_segment_latency_ms, 1) if first_speech_segment_latency_ms else None,
            "streamError": stream_error,
        }

        if pacing_sleep_ms_list:
            result["steps"]["pacing"] = {
                "avgPacingSleepMs": round(sum(pacing_sleep_ms_list) / len(pacing_sleep_ms_list), 1),
                "maxPacingSleepMs": max(pacing_sleep_ms_list),
                "avgMediaLeadMs": round(sum(media_lead_ms_list) / len(media_lead_ms_list), 1),
                "maxMediaLeadMs": max(media_lead_ms_list),
            }

        if started_session:
            self._do_end_session(session_id)
            result["endedSession"] = True

        result["success"] = (
            speech_segments_seen > 0
            and tts_result_data is not None
            and tts_result_data.get("ok") is True
            and tts_result_data.get("streamingHandoff") is True
            and stream_error is None
        )
        result["smokeElapsedMs"] = round(smoke_elapsed_ms, 1)

        print(f"[l40s-handoff_smoke] RESULT: success={result['success']} "
              f"speech_segments={speech_segments_seen} tts_ok={tts_result_data.get('ok') if tts_result_data else None} "
              f"elapsed={result['smokeElapsedMs']}ms")

        return result

    @modal.fastapi_endpoint(method="POST")
    def handle(self, body: dict):
        from fastapi.responses import StreamingResponse

        action = body.get("action")
        if action == "health":
            return self._do_health()
        elif action == "start":
            return self._do_start_session(
                avatar_id=body.get("avatar_id"),
                bg_id=body.get("bg_id"),
            )
        elif action == "render_frame":
            return self._do_render_frame(session_id=body.get("session_id"))
        elif action == "render_audio":
            return self._do_render_audio(
                session_id=body.get("session_id"),
                audio_base64=body.get("audio_base64"),
                max_duration_s=body.get("max_duration_s", 60.0),
                jpeg_quality=body.get("jpeg_quality", 80),
            )
        elif action == "render_idle":
            return self._do_render_idle(
                session_id=body.get("session_id"),
                duration_s=body.get("duration_s", 3.0),
                jpeg_quality=body.get("jpeg_quality", 75),
            )
        elif action == "render_idle_chunk":
            return self._do_render_idle_chunk(
                session_id=body.get("session_id"),
                duration_s=body.get("duration_s", 0.5),
                jpeg_quality=body.get("jpeg_quality", 75),
            )
        elif action == "render_audio_stream":
            return StreamingResponse(
                self._do_render_audio_stream(
                    session_id=body.get("session_id"),
                    audio_base64=body.get("audio_base64"),
                    max_duration_s=body.get("max_duration_s", 60.0),
                    jpeg_quality=body.get("jpeg_quality", 80),
                ),
                media_type="text/event-stream",
            )
        elif action == "end":
            return self._do_end_session(session_id=body.get("session_id"))
        elif action == "timeline_stream":
            return StreamingResponse(
                self._do_timeline_stream(session_id=body.get("session_id")),
                media_type="text/event-stream",
            )
        elif action == "timeline_stream_fmp4_video":
            return StreamingResponse(
                self._do_timeline_stream_fmp4_video(session_id=body.get("session_id")),
                media_type="text/event-stream",
            )
        elif action == "timeline_stream_fmp4_av":
            return StreamingResponse(
                self._do_timeline_stream_fmp4_av(session_id=body.get("session_id")),
                media_type="text/event-stream",
            )
        elif action == "timeline_enqueue":
            return self._do_timeline_enqueue(body)
        elif action == "timeline_stop":
            return self._do_timeline_stop(body)
        elif action == "snapshot_probe":
            return self._do_snapshot_probe()
        elif action == "nvenc_probe":
            return self._do_nvenc_probe()
        elif action == "fmp4_real_frame_probe":
            return self._do_fmp4_real_frame_probe()
        elif action == "fmp4_av_probe":
            return self._do_fmp4_av_probe()
        elif action == "fmp4_av_live_smoke":
            return self._do_fmp4_av_live_smoke()
        elif action == "ffmpeg_fragment_cadence_probe":
            return self._do_ffmpeg_fragment_cadence_probe()
        elif action == "idle_60s_smoke":
            return self._do_idle_60s_smoke()
        elif action == "tts_same_gpu_parallel":
            return self._do_tts_same_gpu_parallel(body)
        elif action == "tts_handoff_smoke":
            return self._do_tts_handoff_smoke(body)
        else:
            return {"error": f"unknown action: {action}", "valid_actions": ["health", "start", "render_frame", "render_audio", "render_idle", "render_idle_chunk", "render_audio_stream", "end", "timeline_stream", "timeline_stream_fmp4_video", "timeline_stream_fmp4_av", "timeline_enqueue", "timeline_stop", "snapshot_probe", "nvenc_probe", "fmp4_real_frame_probe", "fmp4_av_probe", "fmp4_av_live_smoke", "idle_60s_smoke", "tts_same_gpu_parallel"]}


@app.local_entrypoint()
def main() -> None:
    print("avtr1-lifecycle-parallel-l40s (EXPERIMENTAL): use 'modal deploy' to deploy.")
    inst = AvatarLifecycleL40SParallel()
    health = inst.health.remote()
    print(f"Health: {health}")
