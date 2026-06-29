"""
Modal avatar lifecycle runtime for AVTR-1.

Provides session lifecycle: start_session → render_debug_frame → end_session.
Phase 7: render_audio — renders avatar speech frames from TTS audio.
Phase 7.1: render_idle — renders idle animation clip from silence.
Phase 7.2: render_idle_chunk — renders small idle increments for live idle loop.

Uses L4 GPU + GPU memory snapshot + existing avtr1-artifacts-l4 Volume.

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_app.py

Invoke (from local server):
    import modal
    cls = modal.Cls.from_name("avtr1-lifecycle", "AvatarLifecycleL4")
    inst = cls()
    inst.health.remote()
    inst.start_session.remote()
    inst.render_debug_frame.remote(session_id)
    inst.render_audio.remote(session_id, audio_base64)
    inst.render_idle.remote(session_id)
    inst.render_idle_chunk.remote(session_id)
    inst.end_session.remote(session_id)
"""

from __future__ import annotations

import os
import subprocess
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

L4_VOLUME_NAME = "avtr1-artifacts-l4"
L4_MOUNT_PATH = "/data/avtr1-l4"

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
        "ffmpeg",
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
            "AVTR1_LOCAL_STORAGE": L4_MOUNT_PATH,
        }
    )
)

app = modal.App("avtr1-lifecycle", image=image)

control_image = modal.Image.debian_slim().pip_install("fastapi[standard]")

l4_volume = modal.Volume.from_name(L4_VOLUME_NAME, create_if_missing=True)

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
    print(f"[lifecycle] Storage root: {storage_root}")
    print("[lifecycle] Ensuring all HF artifacts are present...")
    t0 = time.monotonic()
    mgr = get_artifact_manager()
    mgr.ensure_all_artifacts(workers=4)
    elapsed = time.monotonic() - t0
    print(f"[lifecycle] Artifacts ready in {elapsed:.0f}s")
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
    print(f"[lifecycle] Using avatar '{chosen}'")
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


@app.cls(
    gpu="L4",
    timeout=1800,
    volumes={L4_MOUNT_PATH: l4_volume},
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=90,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
)
class AvatarLifecycleL4:
    """L4 GPU snapshot class for avatar session lifecycle.

    scaledown_window=90 keeps the GPU container warm for 90s between calls,
    preventing "No active session" errors during live conversations with
    15-30s gaps between turns. After container restore (snap=False),
    _session_active resets to False — the bridge must detect this via
    modalSessionLost telemetry and skip subsequent render attempts.

    Deploy:
        modal deploy modal_lifecycle_app.py

    Invoke:
        modal.Cls.from_name("avtr1-lifecycle", "AvatarLifecycleL4")
    """

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import torch

        print("[lifecycle-snap] === @modal.enter(snap=True) starting ===")
        print(f"[lifecycle-snap] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[lifecycle-snap] GPU: {torch.cuda.get_device_name(0)}")

        self._snap_ok = True
        self._snap_error: str | None = None

        try:
            storage_root = _ensure_artifacts(Path(L4_MOUNT_PATH))
            l4_volume.commit()

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                self._snap_ok = False
                self._snap_error = f"Missing artifacts: {missing}"
                print(f"[lifecycle-snap] ERROR: {self._snap_error}")
                return

            print("[lifecycle-snap] All artifacts present.")
            _run_nvidia_smi("lifecycle/before-load")

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions
            import numpy as np

            print(f"[lifecycle-snap] Loading pipeline with avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self.startup_elapsed = time.monotonic() - t0
            print(f"[lifecycle-snap] Pipeline loaded in {self.startup_elapsed:.1f}s")

            _run_nvidia_smi("lifecycle/after-load")

            self.avatar_id = avatar_id
            self.avatar = self.registry[avatar_id]

            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            self._chunk = Chunk(audio_speech=silence, audio_listen=silence)

            self._bg_id = _pick_first_bg(self.pipeline._backgrounds)
            self._options = RenderOptions(
                pixel_format="yuv_i420",
                bg_id=self._bg_id,
            )

            print("[lifecycle-snap] Running warmup render chunk...")
            t_warm = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                self.avatar, self._chunk, state=None, options=self._options,
            )
            warmup_frames = list(frames_iter)
            warmup_ms = (time.monotonic() - t_warm) * 1000
            print(
                f"[lifecycle-snap] Warmup: {warmup_ms:.0f}ms, "
                f"{len(warmup_frames)} frames, "
                f"state={type(self._state).__name__}"
            )

            _run_nvidia_smi("lifecycle/after-warmup")
            print("[lifecycle-snap] === @modal.enter(snap=True) complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[lifecycle-snap] FAILED during snap init: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        print("[lifecycle-restore] === @modal.enter(snap=False) starting ===")
        print(f"[lifecycle-restore] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[lifecycle-restore] GPU: {torch.cuda.get_device_name(0)}")
        _run_nvidia_smi("post-restore-l4")

        self._session_active = False
        self._session_id = None

        self._timeline_running = False
        self._timeline_queue = None
        self._timeline_stop_requested = False
        self._timeline_segment_id = 0

        if not self._snap_ok:
            print(f"[lifecycle-restore] WARNING: snap init had error: {self._snap_error}")
        else:
            print("[lifecycle-restore] Pipeline ready from snapshot.")

        print("[lifecycle-restore] === @modal.enter(snap=False) complete ===")

    def _do_health(self) -> dict:
        import torch

        return {
            "gpu": "L4",
            "pipeline_loaded": hasattr(self, "pipeline") and self.pipeline is not None,
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "avatar_id": getattr(self, "avatar_id", None),
            "bg_id": getattr(self, "_bg_id", None),
            "startup_s": getattr(self, "startup_elapsed", None),
            "session_active": getattr(self, "_session_active", False),
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

        print(f"[lifecycle] Starting session {session_id} with avatar={chosen_avatar}, bg={chosen_bg}")
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
        print(f"[lifecycle] Session {session_id} warmup: {elapsed_ms:.0f}ms, {len(frames)} frames")

        self._session_active = True
        self._session_id = session_id
        self._session_state = state
        self._session_avatar = avatar
        self._timeline_queue = modal.Queue.from_name(
            f"avtr1-timeline-{session_id}", create_if_missing=True
        )

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
                print(f"[render_audio] Resampling {sr}Hz → 16000Hz ({len(audio_f32)} samples)")
                audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                audio_f32 = audio_f32.astype(np.float32)

            total_duration_s = len(audio_f32) / 16000.0
            truncated = total_duration_s > max_duration_s
            if truncated:
                max_samples = int(max_duration_s * 16000)
                audio_f32 = audio_f32[:max_samples]
                print(f"[render_audio] Truncated {total_duration_s:.2f}s → {max_duration_s:.0f}s")

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
                f"[render_audio] {len(audio_f32)} samples, "
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
                    print(f"[render_audio] chunk {i + 1}/{n_steps}, {len(all_frames)} frames, {elapsed:.0f}ms")

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
                f"[render_audio] timings model={model_render_ms:.0f}ms "
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
                    print(f"[render_audio_stream] Resampling {sr}Hz -> 16000Hz ({len(audio_f32)} samples)")
                    audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                    audio_f32 = audio_f32.astype(np.float32)

                total_duration_s = len(audio_f32) / 16000.0
                truncated = total_duration_s > max_duration_s
                if truncated:
                    max_samples = int(max_duration_s * 16000)
                    audio_f32 = audio_f32[:max_samples]
                    print(f"[render_audio_stream] Truncated {total_duration_s:.2f}s -> {max_duration_s:.0f}s")
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
                f"[render_audio_stream] {len(audio_f32)} samples, "
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
                        f"[render_audio_stream] chunk {i + 1}/{n_steps}, "
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
                f"[render_audio_stream] timings model={model_render_ms:.0f}ms "
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
            f"[render_idle] {duration_s}s silence, "
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
                    print(f"[render_idle] chunk {i + 1}/{n_steps}, {len(all_frames)} frames, {elapsed:.0f}ms")

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
                f"[render_idle] timings model={model_render_ms:.0f}ms "
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
                f"[render_idle_chunk] timings model={model_render_ms:.0f}ms "
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
        ended_id = getattr(self, "_session_id", None)
        was_active = getattr(self, "_session_active", False)

        self._session_active = False
        self._session_id = None
        self._session_state = None
        self._session_avatar = None

        print(f"[lifecycle] Session {ended_id} ended (was_active={was_active})")

        return {
            "ok": True,
            "sessionId": ended_id,
            "wasActive": was_active,
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
        print(f"[timeline_stream] started session={self._session_id}")

        try:
            idle_chunk_index = 0
            cumulative_chunk_index = 0

            while not self._timeline_stop_requested and getattr(self, "_session_active", False):
                speech_item = timeline_queue.get(block=False)

                if speech_item is not None:
                    if speech_item.get("__stop"):
                        self._timeline_stop_requested = True
                        print("[timeline_stream] received __stop sentinel")
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
                        print(f"[timeline_stream] Resampling {sr}Hz -> 16000Hz ({len(audio_f32)} samples)")
                        audio_f32 = soxr.resample(audio_f32, sr, 16000, quality="HQ")
                        audio_f32 = audio_f32.astype(np.float32)

                    total_duration_s = len(audio_f32) / 16000.0
                    if total_duration_s > EMERGENCY_SPEECH_CAP_S:
                        print(f"[timeline_stream] EMERGENCY CAP: truncating {total_duration_s:.1f}s -> {EMERGENCY_SPEECH_CAP_S}s")
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
                        f"[timeline_stream] speech turnId={turn_id} "
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
                        f"[timeline_stream] speech done turnId={turn_id} "
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
            print(f"[timeline_stream] ended chunks={cumulative_chunk_index} stop_requested={self._timeline_stop_requested}")

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
        print(f"[timeline_fmp4_video] started session={self._session_id} ffmpeg_pid={ffmpeg_proc.pid}")

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
                    print(f"[timeline_fmp4_video] init segment {len(seg_data)} bytes, "
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
                        print("[timeline_fmp4_video] received __stop sentinel")
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
                        f"[timeline_fmp4_video] speech turnId={turn_id} "
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
                                print(f"[timeline_fmp4_video] FFmpeg stdin broken pipe")
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

                    print(f"[timeline_fmp4_video] speech done turnId={turn_id}")

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
                            print(f"[timeline_fmp4_video] FFmpeg stdin broken pipe")
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
                print(f"[timeline_fmp4_video] FFmpeg stderr errors: {stderr_errors[:5]}")

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
            print(f"[timeline_fmp4_video] ended chunks={chunks_rendered} segments={media_sequence} "
                  f"rawMB={total_raw_bytes_written / 1e6:.1f} encodedMB={total_encoded_bytes_sent / 1e6:.1f}")

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"fMP4 video stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"
        finally:
            print(f"[timeline_fmp4_video] cleanup starting ffmpeg_pid={ffmpeg_proc.pid}")

            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass

            try:
                stdout_thread.join(timeout=3)
                if stdout_thread.is_alive():
                    print("[timeline_fmp4_video] WARNING: stdout_thread still alive after join timeout")
            except Exception:
                pass

            try:
                stderr_thread.join(timeout=3)
                if stderr_thread.is_alive():
                    print("[timeline_fmp4_video] WARNING: stderr_thread still alive after join timeout")
            except Exception:
                pass

            if ffmpeg_proc.poll() is None:
                print(f"[timeline_fmp4_video] terminating ffmpeg pid={ffmpeg_proc.pid}")
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            print(f"[timeline_fmp4_video] ffmpeg exited rc={ffmpeg_proc.returncode}")

            self._timeline_running = False
            self._timeline_stop_requested = False
            print("[timeline_fmp4_video] cleanup complete")

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

        fifo_path = f"/tmp/avtr1_audio_{self._session_id}"
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(fifo_path)
        print(f"[timeline_fmp4_av] FIFO created at {fifo_path}")

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

        print(f"[timeline_fmp4_av] starting FFmpeg pid pending...")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[timeline_fmp4_av] FFmpeg started pid={ffmpeg_proc.pid}")

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
                            if box_type == b"ftyp":
                                print(f"[stdout_reader] saw ftyp box {len(box_data)}B")
                            init_collected.extend(box_data)
                            if box_type == b"moov":
                                init_sent = True
                                print(f"[stdout_reader] saw moov box {len(box_data)}B -> init complete, total {len(init_collected)}B")
                                segment_queue.put(("init", bytes(init_collected)))
                                init_collected = bytearray()
                        elif box_type == b"moof":
                            print(f"[stdout_reader] saw moof box {len(box_data)}B seq=?")
                            media_buf = bytearray(box_data)
                        elif box_type == b"mdat" and len(media_buf) > 0:
                            media_buf.extend(box_data)
                            total_media = len(media_buf)
                            print(f"[stdout_reader] saw mdat box {len(box_data)}B -> media segment {total_media}B")
                            segment_queue.put(("media", bytes(media_buf)))
                            media_buf = bytearray()
                        elif box_type == b"mfra":
                            print("[stdout_reader] saw mfra box (end)")

                segment_queue.put(None)
                print("[stdout_reader] EOF from FFmpeg stdout")
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
        }
        yield f"data: {json.dumps(start_event)}\n\n"
        print(f"[timeline_fmp4_av] started session={self._session_id} ffmpeg_pid={ffmpeg_proc.pid} "
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

        gpu_snapshots.append(_gpu_memory_snapshot("stream_start"))

        def _queue_audio(data: bytes, label: str) -> bool:
            nonlocal total_audio_bytes_written
            try:
                audio_write_queue.put(data, timeout=2)
                total_audio_bytes_written += len(data)
                print(f"[timeline_fmp4_av] audio queued: {len(data)}B ({label})")
                return True
            except queue.Full:
                print(f"[timeline_fmp4_av] audio_write_queue FULL, cannot queue {label}")
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
                print(f"[timeline_fmp4_av] video queued: {len(raw_chunk)}B ({len(frames_list)} frames, {label})")
                return True
            except queue.Full:
                print(f"[timeline_fmp4_av] video_write_queue FULL, cannot queue {label}")
                return False

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
                    gpu_snapshots.append(_gpu_memory_snapshot("fmp4_init"))
                    print(f"[timeline_fmp4_av] init segment {len(seg_data)} bytes, "
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
                if not init_sent_to_client and not init_5s_warned and time.monotonic() > stream_start_mono + 5:
                    init_5s_warned = True
                    print("[timeline_fmp4_av] WARNING: no init after 5s, FFmpeg may be stalled")

                if not init_sent_to_client and time.monotonic() > init_deadline_mono:
                    print(f"[timeline_fmp4_av] INIT TIMEOUT: no ftyp+moov within {INIT_TIMEOUT_S}s")
                    err_event = {"type": "error", "error": f"fMP4 AV init timeout: no init segment produced within {INIT_TIMEOUT_S}s"}
                    yield f"data: {json.dumps(err_event)}\n\n"
                    break

                speech_item = timeline_queue.get(block=False)

                if speech_item is not None:
                    if speech_item.get("__stop"):
                        self._timeline_stop_requested = True
                        print("[timeline_fmp4_av] received __stop sentinel")
                        break

                    turn_id = speech_item["turnId"]
                    audio_b64 = speech_item["audioBase64"]
                    enqueue_at = speech_item.get("enqueueAt", 0)
                    audio_duration_ms_reported = speech_item.get("audioDurationMs", 0)

                    wav_bytes = base64.b64decode(audio_b64)
                    audio_f32, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=True)
                    audio_f32 = audio_f32.mean(axis=1)

                    if sr != AUDIO_SR:
                        print(f"[timeline_fmp4_av] resampling {sr}Hz -> {AUDIO_SR}Hz ({len(audio_f32)} samples)")
                        audio_f32 = soxr.resample(audio_f32, sr, AUDIO_SR, quality="HQ")
                        audio_f32 = audio_f32.astype(np.float32)

                    total_duration_s = len(audio_f32) / AUDIO_SR
                    if total_duration_s > EMERGENCY_SPEECH_CAP_S:
                        print(f"[timeline_fmp4_av] EMERGENCY CAP: truncating {total_duration_s:.1f}s -> {EMERGENCY_SPEECH_CAP_S}s")
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
                        f"[timeline_fmp4_av] speech turnId={turn_id} "
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
                                    "audioSamplesPerChunk": step,
                                }

                                is_first = not speech_first_segment_emitted
                                evt["isSpeechStart"] = is_first
                                if is_first:
                                    speech_first_segment_emitted = True
                                    if enqueue_at > 0:
                                        evt["timing"]["enqueueToFirstFrameMs"] = round((time.time() - enqueue_at) * 1000, 1)
                                    if gpu_snapshots:
                                        last_snap = gpu_snapshots[-1]
                                        evt["gpu"] = {k: v for k, v in last_snap.items() if k != "label"}

                                is_end = is_speech_end and not speech_end_emitted
                                evt["isSpeechEnd"] = is_end
                                if is_end:
                                    speech_end_emitted = True

                            yield f"data: {json.dumps(evt)}\n\n"

                        cumulative_chunk_index += 1
                        chunks_rendered += 1

                        if cumulative_chunk_index - last_gpu_snapshot_chunk >= 25:
                            gpu_snapshots.append(_gpu_memory_snapshot(f"periodic_chunk_{cumulative_chunk_index}"))
                            last_gpu_snapshot_chunk = cumulative_chunk_index

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
                                    "audioSamplesPerChunk": step,
                                }
                            yield f"data: {json.dumps(evt)}\n\n"
                        if tail_events:
                            cumulative_chunk_index += 1
                            chunks_rendered += 1
                        speech_end_emitted = True

                    self._timeline_segment_id += 1
                    idle_chunk_index = 0
                    current_mode = "idle"
                    speech_end_mono = time.monotonic()
                    gpu_snapshots.append(_gpu_memory_snapshot("speech_end"))

                    print(f"[timeline_fmp4_av] speech done turnId={turn_id}")

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
                                "audioSamplesPerChunk": step,
                            }
                            if speech_end_to_idle_ms is not None:
                                evt["timing"]["speechEndToIdleResumeMs"] = speech_end_to_idle_ms
                                speech_end_to_idle_ms = None
                        yield f"data: {json.dumps(evt)}\n\n"

                    cumulative_chunk_index += 1
                    idle_chunk_index += 1
                    chunks_rendered += 1

                    if cumulative_chunk_index - last_gpu_snapshot_chunk >= 25:
                        gpu_snapshots.append(_gpu_memory_snapshot(f"periodic_chunk_{cumulative_chunk_index}"))
                        last_gpu_snapshot_chunk = cumulative_chunk_index

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"fMP4 A/V stream failed: {exc}"}
            yield f"data: {json.dumps(err_event)}\n\n"
        finally:
            print(f"[timeline_fmp4_av] cleanup starting ffmpeg_pid={ffmpeg_proc.pid}")

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
                    print("[timeline_fmp4_av] WARNING: video_thread still alive after join timeout")
                else:
                    print("[timeline_fmp4_av] video_thread joined cleanly")
            except Exception:
                pass

            try:
                audio_thread.join(timeout=5)
                if audio_thread.is_alive():
                    print("[timeline_fmp4_av] WARNING: audio_thread still alive after join timeout")
                else:
                    print("[timeline_fmp4_av] audio_thread joined cleanly")
            except Exception:
                pass

            try:
                stdout_thread.join(timeout=3)
                if stdout_thread.is_alive():
                    print("[timeline_fmp4_av] WARNING: stdout_thread still alive after join timeout")
                else:
                    print("[timeline_fmp4_av] stdout_thread joined cleanly")
            except Exception:
                pass

            try:
                stderr_thread.join(timeout=3)
                if stderr_thread.is_alive():
                    print("[timeline_fmp4_av] WARNING: stderr_thread still alive after join timeout")
                else:
                    print("[timeline_fmp4_av] stderr_thread joined cleanly")
            except Exception:
                pass

            if ffmpeg_proc.poll() is None:
                print(f"[timeline_fmp4_av] terminating ffmpeg pid={ffmpeg_proc.pid}")
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            print(f"[timeline_fmp4_av] ffmpeg exited rc={ffmpeg_proc.returncode}")

            stderr_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
            if stderr_errors:
                print(f"[timeline_fmp4_av] FFmpeg stderr errors ({len(stderr_errors)}): {stderr_errors[:5]}")
            stderr_tail = ffmpeg_stderr_lines[-20:] if ffmpeg_stderr_lines else []
            if stderr_tail:
                print(f"[timeline_fmp4_av] FFmpeg stderr tail: {stderr_tail}")

            if reader_error:
                print(f"[timeline_fmp4_av] stdout_reader error: {reader_error[0]}")
            if audio_writer_error:
                print(f"[timeline_fmp4_av] audio_writer error: {audio_writer_error[0]}")
            if video_writer_error:
                print(f"[timeline_fmp4_av] video_writer error: {video_writer_error[0]}")

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
                gpu_summary = {
                    "smiUsedMinMb": min(smi_used) if smi_used else None,
                    "smiUsedMaxMb": max(smi_used) if smi_used else None,
                    "smiTotalMb": smi_total[-1] if smi_total else None,
                    "torchAllocMinMb": min(torch_alloc) if torch_alloc else None,
                    "torchAllocMaxMb": max(torch_alloc) if torch_alloc else None,
                    "torchPeakMb": max(torch_peak) if torch_peak else None,
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
            }
            yield f"data: {json.dumps(end_event)}\n\n"

            self._timeline_running = False
            self._timeline_stop_requested = False
            print(f"[timeline_fmp4_av] cleanup complete chunks={chunks_rendered} segments={media_sequence} "
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
            "enqueueAt": time.monotonic(),
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

        print(f"[timeline_stop] was_running={was_running}")
        return {"ok": True, "wasRunning": was_running}

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
            "pipeline_loaded": hasattr(self, "pipeline") and self.pipeline is not None,
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "avatar_id": getattr(self, "avatar_id", None),
            "bg_id": getattr(self, "_bg_id", None),
            "session_active": getattr(self, "_session_active", False),
            "has_session_state": getattr(self, "_session_state", None) is not None,
            "has_warmup_state": getattr(self, "_state", None) is not None,
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
            print("[live_smoke] no active session, starting one...")
            start_result = self._do_start_session()
            if start_result.get("status") != "avatar_ready":
                result["error"] = f"Failed to start session: {start_result}"
                return result
            started_session = True
            print(f"[live_smoke] session started: {start_result['sessionId']}")

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
            print(f"[live_smoke] starting timeline_stream_fmp4_av generator, timeout={SMOKE_TIMEOUT_S}s")
            gen = self._do_timeline_stream_fmp4_av(session_id=session_id)

            for raw_sse in gen:
                if time.monotonic() > smoke_deadline:
                    print("[live_smoke] TIMEOUT: exceeded {SMOKE_TIMEOUT_S}s")
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
                print(f"[live_smoke] event: {evt_type}")

                if evt_type == "error":
                    result["error"] = evt.get("error", "unknown error from stream")
                    print(f"[live_smoke] stream error: {result['error']}")
                    break

                if evt_type == "fmp4_init":
                    saw_init = True
                    time_to_init_ms = evt.get("timing", {}).get("timeToInitMs")
                    first_media_byte_length = evt.get("byteLength")
                    init_event = {k: v for k, v in evt.items() if k != "dataBase64"}
                    print(f"[live_smoke] INIT received: timeToInitMs={time_to_init_ms} byteLength={evt.get('byteLength')}")

                if evt_type == "fmp4_media":
                    saw_media = True
                    if first_media_event is None:
                        first_media_event = {k: v for k, v in evt.items() if k != "dataBase64"}
                        first_media_byte_length = evt.get("byteLength")
                        print(f"[live_smoke] FIRST MEDIA received: byteLength={evt.get('byteLength')} sequence={evt.get('sequence')}")
                    if saw_init:
                        print("[live_smoke] init + first media seen, stopping generator")
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
            print(f"[live_smoke] ended session {session_id}")
            result["endedSession"] = True

        result["success"] = saw_init and saw_media
        result["smokeElapsedMs"] = round(smoke_elapsed_ms, 1)

        print(f"[live_smoke] RESULT: success={result['success']} init={saw_init} media={saw_media} "
              f"timeToInitMs={time_to_init_ms} elapsedMs={result['smokeElapsedMs']}")

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
        else:
            return {"error": f"unknown action: {action}", "valid_actions": ["health", "start", "render_frame", "render_audio", "render_idle", "render_idle_chunk", "render_audio_stream", "end", "timeline_stream", "timeline_stream_fmp4_video", "timeline_stream_fmp4_av", "timeline_enqueue", "timeline_stop", "snapshot_probe", "nvenc_probe", "fmp4_real_frame_probe", "fmp4_av_probe", "fmp4_av_live_smoke"]}


@app.function(
    image=control_image,
    timeout=30,
    scaledown_window=60,
    min_containers=0,
    buffer_containers=0,
    max_containers=2,
)
@modal.concurrent(max_inputs=20)
@modal.fastapi_endpoint(method="POST")
def timeline_control(body: dict):
    import time as _time

    action = body.get("action")
    if action == "enqueue":
        session_id = body.get("session_id")
        if not session_id:
            return {"ok": False, "error": "No session_id provided"}
        turn_id = body.get("turnId")
        audio_base64 = body.get("audio_base64")
        audio_duration_ms = body.get("audio_duration_ms", 0)
        if not audio_base64:
            return {"ok": False, "error": "No audio_base64 provided"}

        enqueued_at = _time.time()
        q = modal.Queue.from_name(f"avtr1-timeline-{session_id}", create_if_missing=True)
        q.put({
            "turnId": turn_id,
            "audioBase64": audio_base64,
            "audioDurationMs": audio_duration_ms,
            "enqueueAt": enqueued_at,
        })

        print(f"[timeline_control/enqueue] session={session_id} turnId={turn_id} "
              f"durationMs={audio_duration_ms:.0f}")

        return {
            "ok": True,
            "turnId": turn_id,
            "audioDurationMs": audio_duration_ms,
            "enqueuedAt": enqueued_at,
        }

    elif action == "stop":
        session_id = body.get("session_id")
        if not session_id:
            return {"ok": False, "error": "No session_id provided"}

        q = modal.Queue.from_name(f"avtr1-timeline-{session_id}", create_if_missing=True)
        drained = 0
        while True:
            try:
                item = q.get(block=False)
                if item is None:
                    break
                drained += 1
            except Exception:
                break

        q.put({"__stop": True})

        print(f"[timeline_control/stop] session={session_id} drained={drained}")

        return {"ok": True, "drained": drained}

    else:
        return {"ok": False, "error": f"unknown action: {action}", "valid_actions": ["enqueue", "stop"]}


@app.local_entrypoint()
def main() -> None:
    print("avtr1-lifecycle: use 'modal deploy' to deploy, then use the local server.")
    inst = AvatarLifecycleL4()
    health = inst.health.remote()
    print(f"Health: {health}")
