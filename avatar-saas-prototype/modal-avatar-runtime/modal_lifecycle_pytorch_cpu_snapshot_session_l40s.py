"""
Phase 3F — Session Lifecycle Integration
CPU-Only Modal Memory Snapshot on L40S

Built on Phase 3E (OmniVoice same-GPU parallel stress test, proven GO).

Phase 3F adds production-style session lifecycle:
  - start_session(session_id, avatar_id) — creates per-session state
  - push_audio(session_id, ...) — feeds audio into session handoff queue
  - run_session_smoke(session_id, ...) — full session render pipeline
  - get_session_status(session_id) — lifecycle state query
  - end_session(session_id) — cleanup with FFmpeg/thread/tensor teardown
  - @modal.exit() container death finalizer

Guardrails:
  1. max_containers=1 — session state is container-local, not distributed
  2. @modal.exit() cleanup hook for container death without end_session
  3. fMP4 media segments capped at FMP4_MEDIA_CAP entries (counters always tracked)

All Phase 3E smoke tests preserved (render_chunk, fmp4_video_stream, audio_handoff,
omnivoice_tts, omnivoice_parallel_stress).

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_pytorch_cpu_snapshot_session_l40s.py

Invoke (NEVER modal.run):
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s', 'AvatarLifecyclePyTorchCpuSnapshotSessionL40S')
inst = cls()
print(json.dumps(inst.health_check.remote(), indent=2))
    "
"""

from __future__ import annotations

import base64
import collections
import queue
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import modal

from avtr_segment_sender import _SegmentSender, stamp_media_event

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"
MAX_CACHE_ENTRIES = 3
OMNIVOICE_COMMIT = "f7f5040646f4fb8eef84e011adebec0c375ea6ae"
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
FMP4_MEDIA_CAP = 100
# Per-session cap on the in-memory fMP4 media-segment list.
# Counters (fmp4_media_count, fmp4_media_total_bytes) are always tracked
# even after the list is capped, so callers never lose aggregate stats.
SESSION_STATE_NOTE = "container-local (max_containers=1)"

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
        "git",
        "gcc",
        "g++",
    )
    .pip_install(
        "torch>=2.8,<2.9",
        "torchaudio>=2.8,<2.9",
        "torchvision>=0.23,<0.24",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy>=2.0,<3",
        "safetensors>=0.5,<1",
        "huggingface_hub>=1.3",
        "kornia>=0.8.1,<0.9",
        "roma>=1.5.4,<2",
        "opencv-python-headless>=4.11,<5",
        "einops>=0.8.1,<0.9",
        "scikit-image>=0.25,<0.26",
        "soxr>=0.4,<0.5",
        "imageio>=2.37,<3",
        "httpx>=0.28,<0.29",
        "tqdm>=4.67,<5",
        "onnx>=1.17,<2",
        "onnx2torch>=1.5,<2",
        "onnxruntime-gpu>=1.20",
        "tensorrt<=10.12",
        "transformers>=5.3",
        "soundfile>=0.13",
        f"git+https://github.com/k2-fsa/OmniVoice.git@{OMNIVOICE_COMMIT}",
    )
    .pip_install(
        # Phase 4C: required by @modal.fastapi_endpoint() on the class
        # (StreamingResponse, HTTPException, etc.). Pinned <1 to match
        # Modal docs' reference version.
        "fastapi>=0.115,<1",
    )
    .add_local_dir(
        "/Users/ayya/developer/avtr-1/src/avtr1_renderer",
        remote_path="/opt/avtr1/avtr1_renderer",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/avtr_segment_sender.py",
        remote_path="/opt/avtr1/avtr_segment_sender.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/avtr1",
            "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH,
        }
    )
)

app = modal.App("avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s", image=image)

l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)


LIPSYNC_COORDS = (
    3, 4, 5, 6, 7, 8, 18, 19, 20,
    33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62,
)
N_LIPSYNC = len(LIPSYNC_COORDS)
_LIPSYNC_INDEX = list(LIPSYNC_COORDS)
Z_SCORE_CLIP = 5.0
MODNET_INPUT_H = 288
MODNET_INPUT_W = 512


def _read_rss_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


def _run_nvidia_smi(label: str) -> None:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"[nvidia-smi] {label}: {result.stdout.strip()}")
    except Exception:
        pass


def _gpu_memory_snapshot(label: str) -> dict:
    import subprocess
    import torch
    snap: dict = {"label": label}
    snap["torchAllocMb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                snap["smiUsedMb"] = int(parts[0])
                snap["smiTotalMb"] = int(parts[1])
                snap["smiUtilPct"] = int(parts[2])
    except Exception:
        pass
    print(f"[gpu] {label}: torch={snap.get('torchAllocMb')}MB smi={snap.get('smiUsedMb')}MB")
    return snap


def _build_renderer_model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.spectral_norm as spectral_norm

    def _make_coordinate_grid(spatial_size, ref):
        d, h, w = spatial_size
        x = torch.arange(w, dtype=ref.dtype, device=ref.device)
        y = torch.arange(h, dtype=ref.dtype, device=ref.device)
        z = torch.arange(d, dtype=ref.dtype, device=ref.device)
        grid = torch.stack(torch.meshgrid(z, y, x, indexing="ij"), -1)
        return grid

    class StitchingNetwork(nn.Module):
        def __init__(self, in_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv1d(in_features, in_features, kernel_size, padding=padding)

        def forward(self, x):
            return x + self.conv(x)

    class WarpingNetwork(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)

        def forward(self, x):
            return self.conv(x)

    class SPADEDecoder(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)

        def forward(self, x):
            return self.conv(x)

    class FirstOrderMotionModel(nn.Module):
        def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
            super().__init__()

        def forward(self, x):
            return x

    class DownBlock2d(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)

        def forward(self, x):
            return self.conv(x)

    class UpBlock2d(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.ConvTranspose2d(in_features, out_features, kernel_size, padding=padding)

        def forward(self, x):
            return self.conv(x)

    class Generator(nn.Module):
        def __init__(self, block_expansion, num_blocks, max_features, num_kp,
                     feature_channels, num_down_blocks, estimate_jacobian=False,
                     dense_motion_params=None, norm_G="batch", label_nc=0):
            super().__init__()

        def forward(self, x):
            return x

    class LineartGenerator(nn.Module):
        def __init__(self, in_f, out_f, kernel_size=3, padding=1, lrelu=False):
            super().__init__()
            self.conv = nn.Conv2d(in_f, out_f, kernel_size, padding=padding)

        def forward(self, x):
            return self.conv(x)

    class SPADENorm(nn.Module):
        def __init__(self, norm_nc, label_nc):
            super().__init__()

        def forward(self, x, segmap):
            return x

    class SPADEResBlock(nn.Module):
        def __init__(self, fin, fout, norm_G, label_nc):
            super().__init__()

        def forward(self, x, seg1):
            return x

        def _shortcut(self, x, seg1):
            return x

    return {
        "StitchingNetwork": StitchingNetwork,
        "WarpingNetwork": WarpingNetwork,
        "SPADEDecoder": SPADEDecoder,
    }


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


class _AudioHandoff:
    def __init__(self, maxsize: int = 300):
        self.queue: queue.Queue[tuple | None] = queue.Queue(maxsize=maxsize)
        self.closed = False
        self._lock = threading.Lock()
        self._maxsize = maxsize
        self._chunks_pushed = 0
        self._chunks_popped = 0
        self._bytes_pushed = 0
        self._underflow_count = 0
        self._overflow_count = 0
        self._queue_depth_min = maxsize
        self._queue_depth_max = 0
        self._first_audio_mono: float | None = None
        self._last_audio_mono: float | None = None
        self._created_mono = time.monotonic()

    def push(self, chunk, metadata=None, timeout: float = 30.0) -> bool:
        with self._lock:
            if self.closed:
                return False
            try:
                self.queue.put((chunk, metadata), block=True, timeout=timeout)
            except Exception:
                self._overflow_count += 1
                return False
        self._chunks_pushed += 1
        if hasattr(chunk, "nbytes"):
            self._bytes_pushed += chunk.nbytes
        elif isinstance(chunk, bytes):
            self._bytes_pushed += len(chunk)
        now = time.monotonic()
        if self._first_audio_mono is None:
            self._first_audio_mono = now
        self._last_audio_mono = now
        qd = self.queue.qsize()
        if qd < self._queue_depth_min:
            self._queue_depth_min = qd
        if qd > self._queue_depth_max:
            self._queue_depth_max = qd
        return True

    def pop(self, timeout_s: float = 0.5):
        try:
            item = self.queue.get(block=True, timeout=timeout_s)
            if item is None:
                return None
            self._chunks_popped += 1
            qd = self.queue.qsize()
            if qd < self._queue_depth_min:
                self._queue_depth_min = qd
            if qd > self._queue_depth_max:
                self._queue_depth_max = qd
            return item
        except queue.Empty:
            self._underflow_count += 1
            return None

    def drain(self) -> list:
        items = []
        while True:
            try:
                item = self.queue.get_nowait()
                if item is not None:
                    self._chunks_popped += 1
                    items.append(item)
            except queue.Empty:
                break
        return items

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            try:
                self.queue.put(None, block=False)
            except Exception:
                pass

    def stats(self) -> dict:
        return {
            "chunks_pushed": self._chunks_pushed,
            "chunks_popped": self._chunks_popped,
            "bytes_pushed": self._bytes_pushed,
            "queue_maxsize": self._maxsize,
            "queue_depth_min": self._queue_depth_min if self._queue_depth_min != self._maxsize else 0,
            "queue_depth_max": self._queue_depth_max,
            "underflow_count": self._underflow_count,
            "overflow_count": self._overflow_count,
            "first_audio_ms": round((self._first_audio_mono - self._created_mono) * 1000, 1) if self._first_audio_mono is not None else None,
            "last_audio_ms": round((self._last_audio_mono - self._created_mono) * 1000, 1) if self._last_audio_mono is not None else None,
            "closed": self.closed,
        }


def _modnet_forward(modnet, x):
    import torch
    return torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) * 0.99


@app.cls(
    gpu="L40S",
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=3600,
)
@modal.concurrent(max_inputs=2)
class AvatarLifecyclePyTorchCpuSnapshotSessionL40S:

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch
        import numpy as np
        from huggingface_hub import hf_hub_download
        from pathlib import Path

        self._snap_ok = False
        self._full_render_warmed = False

        print("[snap] === CPU-Only Snapshot Init (Phase 3F Session) ===")

        storage_root = Path(L40S_MOUNT_PATH)
        self.avatar_id = _pick_first_avatar(storage_root)
        print(f"[snap] avatar_id={self.avatar_id!r}")

        self._omnivoice_loaded_in_snap = False
        self._omnivoice_snap_error = None
        print("[snap] Loading OmniVoice model to CPU...")
        try:
            from omnivoice import OmniVoice as _OmniVoice
            self._omnivoice_model = _OmniVoice.from_pretrained(
                OMNIVOICE_MODEL_ID,
                device_map="cpu",
                dtype=torch.float32,
            )
            self._omnivoice_sr = int(getattr(
                self._omnivoice_model, "sampling_rate", 24000))
            self._omnivoice_loaded_in_snap = True
            print(f"[snap] OmniVoice CPU loaded, sr={self._omnivoice_sr}")
        except Exception as exc:
            self._omnivoice_snap_error = str(exc)
            self._omnivoice_model = None
            print(f"[snap] OmniVoice CPU load FAILED: {exc}")

        self._cpu_snapshot_ok = True

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch
        import numpy as np
        from avtr1_renderer.pipeline import Pipeline

        print("[restore] Post-restore on GPU")

        if not getattr(self, "_cpu_snapshot_ok", False):
            self._snap_error = getattr(self, "_snap_error", "CPU snapshot prep failed")
            print(f"[restore] ABORT: CPU snapshot prep was not ok: {self._snap_error}")
            self._sessions = {}
            return

        cuda_ok = torch.cuda.is_available()
        print(f"[restore] CUDA available: {cuda_ok}")
        if cuda_ok:
            print(f"[restore] GPU: {torch.cuda.get_device_name(0)}")
        _run_nvidia_smi("gpu-init/start")

        if not cuda_ok:
            self._snap_error = "CUDA not available in snap=False"
            print(f"[restore] ABORT: {self._snap_error}")
            self._sessions = {}
            return

        try:
            storage_root = Path(L40S_MOUNT_PATH)

            print(f"[restore] Loading pipeline with avatar_id={self.avatar_id!r}...")
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[self.avatar_id],
            )
            print(f"[restore] Pipeline loaded")
            _run_nvidia_smi("gpu-init/after-pipeline-load")

            self._pipeline_loaded = True
            self.avatar = self.registry[self.avatar_id]

            mg = self.pipeline._motion_generator
            print(f"[restore] Motion generator attributes: {[a for a in dir(mg) if not a.startswith('_')]}")
            self.chunk_size = mg.chunk_size
            self.future_size = mg.future_size
            self.frame_len = mg.frame_len
            self.audio_shift = getattr(mg, 'audio_shift', 0)
            self.AUDIO_DIM = getattr(mg, 'AUDIO_DIM', 256)
            self.HUBERT_PAST_FRAMES = getattr(mg, 'HUBERT_PAST_FRAMES', 16)
            self.latent_dim = getattr(mg, 'latent_dim', 64)
            self.nfeats = getattr(mg, 'nfeats', 64)
            self.past_size = getattr(mg, 'past_size', 16)
            # Derived values
            self.cond_feature_dim = self.AUDIO_DIM
            self._hubert_audio_len = (self.chunk_size + self.future_size) * self.frame_len + self.audio_shift

            self._bg_id = _pick_first_bg(self.pipeline._backgrounds)
            print(f"[restore] bg_id={self._bg_id!r}")

            self._sessions: dict[str, dict] = {}
            print(f"[restore] self._sessions initialized (container-local, max_containers=1)")

            self._omnivoice_ready = False
            if getattr(self, "_omnivoice_loaded_in_snap", False) \
               and getattr(self, "_omnivoice_model", None) is not None:
                print("[restore] Transferring OmniVoice from CPU snapshot to GPU...")
                try:
                    if hasattr(self._omnivoice_model, "to"):
                        self._omnivoice_model = self._omnivoice_model.to(device="cuda:0")
                    self._omnivoice_model.eval()
                    torch.cuda.synchronize()
                    print("[restore] OmniVoice CPU->GPU transfer done")
                except Exception as exc:
                    print(f"[restore] OmniVoice CPU->GPU transfer FAILED: {exc}")
                    self._omnivoice_model = None

            if getattr(self, "_omnivoice_model", None) is None:
                print("[restore] Loading OmniVoice fresh on GPU (fallback)...")
                try:
                    from omnivoice import OmniVoice as _OmniVoice
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    self._omnivoice_model = _OmniVoice.from_pretrained(
                        OMNIVOICE_MODEL_ID,
                        device_map="cuda:0",
                        dtype=dtype,
                        attn_implementation="eager",
                    )
                    self._omnivoice_sr = int(getattr(
                        self._omnivoice_model, "sampling_rate", 24000))
                    print(f"[restore] OmniVoice fresh loaded, sr={self._omnivoice_sr}")
                except Exception as exc:
                    self._omnivoice_model = None
                    print(f"[restore] OmniVoice fresh load FAILED: {exc}")

            if getattr(self, "_omnivoice_model", None) is not None:
                try:
                    with torch.inference_mode():
                        self._omnivoice_model.generate(
                            "Hi.",
                            preprocess_prompt=True,
                            postprocess_output=True,
                            denoise=True,
                            t_shift=0.1,
                            layer_penalty_factor=5.0,
                            position_temperature=5.0,
                            class_temperature=0.0,
                        )
                    torch.cuda.synchronize()
                    self._omnivoice_ready = True
                    _run_nvidia_smi("gpu-init/after-omnivoice-warmup")
                    print("[restore] OmniVoice warmup OK, _omnivoice_ready=True")
                except Exception as exc:
                    self._omnivoice_model = None
                    self._omnivoice_ready = False
                    print(f"[restore] OmniVoice warmup FAILED: {exc}")

            self._snap_ok = True

        except Exception as exc:
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[restore] FAILED: {exc}")
            self._sessions = {}

    @modal.method()
    def health_check(self) -> dict:
        return {
            "ok": True,
            "chunk_size": getattr(self, "chunk_size", 0),
            "active_sessions": list(getattr(self, "_sessions", {}).keys()),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0,
        }

    def _get_avatar(self, avatar_id: str) -> dict:
        if not getattr(self, "_pipeline_loaded", False):
            raise RuntimeError("Pipeline not loaded yet")
        if avatar_id not in self.registry:
            raise ValueError(f"Unknown avatar_id: {avatar_id}. Available: {list(self.registry.keys())}")
        return self.registry[avatar_id]

    def _fanout_sse_events(self, sess: dict, events: list) -> None:
        import json

        sse_queue = sess.get("sse_queue")
        if sse_queue is None:
            return

        for item in events:
            if isinstance(item, dict):
                evt_type = item.get("type", "unknown")
                sse_frame = f"event: {evt_type}\ndata: {json.dumps(item)}\n\n"
                try:
                    sse_queue.put(sse_frame, timeout=2)
                except Exception:
                    print(f"[sse_fanout] DROPPED {evt_type} event (queue full or closed)")
            else:
                _, _, evt, meta = item
                stamp_media_event(evt, meta, sess)
                evt_type = evt.get("type", "fmp4_media")
                sse_frame = f"event: {evt_type}\ndata: {json.dumps(evt)}\n\n"
                try:
                    sse_queue.put(sse_frame, timeout=2)
                except Exception:
                    print(f"[sse_fanout] DROPPED {evt_type} event (queue full or closed)")

    def _emit_sse_error_and_end(self, sess: dict, message: str) -> None:
        import json

        sse_queue = sess.get("sse_queue")
        if sse_queue is None:
            return

        error_evt = {"type": "error", "error": message}
        try:
            sse_queue.put(f"event: error\ndata: {json.dumps(error_evt)}\n\n", timeout=2)
        except Exception:
            pass

        end_evt = {"type": "end", "mode": "error", "chunksRendered": sess.get("chunk_count", 0), "stopReason": "error"}
        try:
            sse_queue.put(f"event: end\ndata: {json.dumps(end_evt)}\n\n", timeout=2)
        except Exception:
            pass

    def _start_session_impl(self, session_id: str, avatar_id: str) -> dict:
        import torch
        import numpy as np

        t0 = time.monotonic()

        if session_id in self._sessions:
            existing = self._sessions[session_id]
            if existing.get("status") == "running":
                return {
                    "session_id": session_id,
                    "error": f"session {session_id} has a live render thread; call end_session first or wait for it to exit",
                    "active_sessions": list(self._sessions.keys()),
                }

        av = self._get_avatar(avatar_id)
        avatar_load_s = time.monotonic() - t0

        warmup_ms = 0.0
        if not self._full_render_warmed:
            print("[warmup] Running full render warmup (first session on this container)...")
            warmup_t0 = time.monotonic()
            frames = []
            try:
                from avtr1_renderer.types import Chunk
                import numpy as np
                
                mg = self.pipeline._motion_generator
                expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
                silence = np.zeros(expected_len, dtype=np.float32)
                chunk = Chunk(audio_speech=silence, audio_listen=silence)
                
                _bg_id = _pick_first_bg(self.pipeline._backgrounds)
                from avtr1_renderer.types import RenderOptions
                _options = RenderOptions(
                    pixel_format="yuv_i420",
                    bg_id=_bg_id,
                )
                
                with torch.inference_mode():
                    state, frames_iter = self.pipeline.process_chunk(
                        av, chunk, state=None, options=_options,
                    )
                    frames = list(frames_iter)
                
                torch.cuda.synchronize()
                warmup_ms = (time.monotonic() - warmup_t0) * 1000
                self._full_render_warmed = True
                print(f"[warmup] Complete in {warmup_ms:.0f}ms, {len(frames)} frames")
            except Exception as e:
                print(f"[warmup] FAILED (non-fatal): {e}")
                import traceback
                traceback.print_exc()
                warmup_ms = -1

        import queue as _queue_mod
        self._sessions[session_id] = {
            "session_id": session_id,
            "avatar_id": avatar_id,
            "avatar": av,
            "start_time": time.monotonic(),
            "chunk_count": 0,
            "status": "ready",
            "render_state": None,
            "handoff": _AudioHandoff(maxsize=300),
            "render_stop_event": threading.Event(),
            "render_thread": None,
            "sse_queue": _queue_mod.Queue(maxsize=500),
            "fmp4_init_segment": None,
            "fmp4_media_segments": [],
            "fmp4_media_count": 0,
            "fmp4_media_total_bytes": 0,
            "ffmpeg_proc": None,
            "ffmpeg_threads": [],
        }

        total_s = time.monotonic() - t0
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
        print(f"[session] STARTED: {session_id} | avatar={avatar_id} | cache_hit={avatar_load_s < 0.1} "
              f"init={total_s:.3f}s warmup={warmup_ms:.0f}ms | vram={vram_mb}MB")

        return {
            "session_id": session_id,
            "avatar_id": avatar_id,
            "status": "ready",
            "cache_hit": avatar_load_s < 0.1,
            "init_s": round(total_s, 4),
            "warmup_ms": round(warmup_ms, 1),
            "active_sessions": list(self._sessions.keys()),
            "state_note": "container-local (max_containers=1)",
            "vram_mb": vram_mb,
        }

    def _end_session_impl(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            return {
                "session_id": session_id,
                "status": "not_active",
                "active_sessions": list(self._sessions.keys()),
            }

        sess = self._sessions[session_id]
        avatar_id = sess.get("avatar_id", "unknown")
        current_status = sess.get("status", "unknown")

        if current_status == "running":
            sess["status"] = "ending"

            render_stop_event = sess.get("render_stop_event")
            if render_stop_event is not None:
                render_stop_event.set()

            handoff = sess.get("handoff")
            if handoff is not None:
                handoff.close()

            render_thread = sess.get("render_thread")
            if render_thread is not None and render_thread.is_alive():
                sse_queue = sess.get("sse_queue")
                if sse_queue is not None:
                    import json
                    end_evt = {"type": "end", "mode": "server", "chunksRendered": sess.get("chunk_count", 0), "stopReason": "session_end"}
                    try:
                        sse_queue.put(f"event: end\ndata: {json.dumps(end_evt)}\n\n", timeout=2)
                    except Exception:
                        pass
                render_thread.join(timeout=30)

            sess["status"] = "ended"

        handoff = sess.get("handoff")
        handoff_closed = handoff is not None and handoff.closed

        render_thread = sess.get("render_thread")
        render_thread_alive = render_thread is not None and render_thread.is_alive()
        threads_joined = not render_thread_alive

        duration_s = time.monotonic() - sess.get("start_time", time.monotonic())
        chunks = sess.get("chunk_count", 0)
        fmp4_count = sess.get("fmp4_media_count", 0)
        fmp4_bytes = sess.get("fmp4_media_total_bytes", 0)
        ffmpeg_proc = sess.get("ffmpeg_proc")
        ffmpeg_rc = sess.get("ffmpeg_rc")
        if ffmpeg_rc is None and ffmpeg_proc is not None:
            ffmpeg_rc = ffmpeg_proc.returncode

        zombie_status = "n/a"
        if ffmpeg_proc is not None:
            try:
                zc = subprocess.run(["ps", "-p", str(ffmpeg_proc.pid)], capture_output=True, timeout=2)
                zombie_status = "zombie_still_running" if zc.returncode == 0 else "clean"
            except Exception:
                zombie_status = "unknown"

        if render_thread_alive:
            sess["status"] = "ended_with_warnings"
            print(f"[session] WARNING: {session_id} render_thread still alive after 30s join; "
                  f"keeping session in self._sessions")

            import torch
            vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

            sender_ref = sess.get("_sender_ref")

            return {
                "session_id": session_id,
                "avatar_id": avatar_id,
                "status": "ended_with_warnings",
                "chunks_rendered": chunks,
                "fmp4_media_count": fmp4_count,
                "fmp4_media_total_bytes": fmp4_bytes,
                "duration_s": round(duration_s, 2),
                "ffmpeg_rc": ffmpeg_rc,
                "zombie_check": zombie_status,
                "handoff_closed": handoff_closed,
                "threads_joined": False,
                "render_thread_alive": True,
                "active_sessions": list(self._sessions.keys()),
                "vram_mb": vram_mb,
                "sender_stats": {
                    "meta_added_count": sender_ref.meta_added_count if sender_ref else None,
                    "meta_popped_count": sender_ref.meta_popped_count if sender_ref else None,
                    "missing_meta_count": sender_ref.missing_meta_count if sender_ref else None,
                    "speech_media_count": sender_ref.speech_media_count if sender_ref else None,
                    "idle_media_count": sender_ref.idle_media_count if sender_ref else None,
                    "source_lead_ms": sender_ref.source_lead_ms if sender_ref else None,
                    "avg_inter_yield_gap_ms": sender_ref.avg_inter_yield_gap_ms if sender_ref else None,
                    "max_inter_yield_gap_ms": sender_ref.max_inter_yield_gap_ms if sender_ref else None,
                    "total_pacing_sleep_ms": sender_ref.total_pacing_sleep_ms if sender_ref else None,
                },
            }

        del sess["render_state"]

        self._sessions.pop(session_id, None)

        import torch
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        sender_ref = sess.get("_sender_ref")
        sender_stats = ""
        if sender_ref is not None:
            sender_stats = (f" pacing_sleep={sender_ref.total_pacing_sleep_ms}ms"
                            f" speech_media={sender_ref.speech_media_count}"
                            f" idle_media={sender_ref.idle_media_count}"
                            f" missing_meta={sender_ref.missing_meta_count}"
                            f" dry_spells={sender_ref.dry_spell_count}")

        print(f"[session] ENDED: {session_id} | avatar={avatar_id} chunks={chunks} "
              f"fmp4_media={fmp4_count} duration={duration_s:.1f}s{sender_stats}")

        return {
            "session_id": session_id,
            "avatar_id": avatar_id,
            "status": "ended",
            "chunks_rendered": chunks,
            "fmp4_media_count": fmp4_count,
            "fmp4_media_total_bytes": fmp4_bytes,
            "duration_s": round(duration_s, 2),
            "ffmpeg_rc": ffmpeg_rc,
            "zombie_check": zombie_status,
            "handoff_closed": handoff_closed,
            "threads_joined": True,
            "render_thread_alive": False,
            "active_sessions": list(self._sessions.keys()),
            "vram_mb": vram_mb,
            "sender_stats": {
                "meta_added_count": sender_ref.meta_added_count if sender_ref else None,
                "meta_popped_count": sender_ref.meta_popped_count if sender_ref else None,
                "missing_meta_count": sender_ref.missing_meta_count if sender_ref else None,
                "speech_media_count": sender_ref.speech_media_count if sender_ref else None,
                "idle_media_count": sender_ref.idle_media_count if sender_ref else None,
                "source_lead_ms": sender_ref.source_lead_ms if sender_ref else None,
                "avg_inter_yield_gap_ms": sender_ref.avg_inter_yield_gap_ms if sender_ref else None,
                "max_inter_yield_gap_ms": sender_ref.max_inter_yield_gap_ms if sender_ref else None,
                "total_pacing_sleep_ms": sender_ref.total_pacing_sleep_ms if sender_ref else None,
            },
        }

    def _do_push_audio(self, body: dict) -> dict:
        import base64
        import numpy as np

        session_id = body.get("session_id", "")
        if session_id not in self._sessions:
            return {
                "session_id": session_id,
                "pushed": False,
                "samples": 0,
                "queue_depth": 0,
                "session_status": "not_found",
                "error": f"session {session_id} not found",
            }

        sess = self._sessions[session_id]
        handoff = sess.get("handoff")
        if handoff is None or handoff.closed:
            return {
                "session_id": session_id,
                "pushed": False,
                "samples": 0,
                "queue_depth": 0,
                "session_status": sess.get("status", "unknown"),
                "error": "handoff closed or missing",
            }

        audio_data_b64 = body.get("audio_data_b64")
        audio_mode = body.get("audio_mode", "pcm")

        STEP_SAMPLES = self.chunk_size * self.frame_len
        if audio_data_b64:
            raw_bytes = base64.b64decode(audio_data_b64)
            audio_float = np.frombuffer(raw_bytes, dtype=np.float32).astype(np.float32, copy=False)
            audio_float = np.nan_to_num(audio_float, nan=0.0, posinf=0.0, neginf=0.0)
            audio_float = np.clip(audio_float, -1.0, 1.0).astype(np.float32, copy=False)
            if len(audio_float) < STEP_SAMPLES:
                audio_float = np.concatenate(
                    [audio_float, np.zeros(STEP_SAMPLES - len(audio_float), dtype=np.float32)]
                )
            elif len(audio_float) > STEP_SAMPLES:
                audio_float = audio_float[:STEP_SAMPLES]
        else:
            audio_float = np.zeros(STEP_SAMPLES, dtype=np.float32)

        chunk_index_local = sess.get("chunk_count", 0)
        meta = {
            "turnId": body.get("turn_id"),
            "chunkIndex": body.get("chunk_index", chunk_index_local),
            "isFirst": body.get("is_first", chunk_index_local == 0),
            "isLast": body.get("is_last", False),
            "totalChunks": body.get("total_chunks"),
            "generatedAt": body.get("generated_at"),
            "enqueueAt": body.get("enqueue_at"),
            "samples": len(audio_float),
            "audio_mode": audio_mode,
        }

        ok = handoff.push(audio_float, metadata=meta, timeout=5.0)

        push_log_count = sess.get("push_audio_log_count", 0)
        if push_log_count < 10 or push_log_count % 100 == 0:
            print(f"[push_audio] turn={meta['turnId']} chunk={meta['chunkIndex']} isFirst={meta['isFirst']} isLast={meta['isLast']} ok={ok} samples={len(audio_float)} qsize={handoff.queue.qsize()} push_count={push_log_count + 1}")
        sess["push_audio_log_count"] = push_log_count + 1

        return {
            "session_id": session_id,
            "pushed": ok,
            "samples": len(audio_float),
            "queue_depth": handoff.queue.qsize() if handoff else 0,
            "session_status": sess.get("status", "unknown"),
        }

    def _do_push_audio_batch(self, body: dict) -> dict:
        import base64
        import time as _time
        import numpy as np

        session_id = body.get("session_id", "")
        turn_id = body.get("turn_id")
        generated_at = body.get("generated_at")
        enqueue_at = body.get("enqueue_at")

        if session_id not in self._sessions:
            return {
                "ok": False,
                "session_id": session_id,
                "error": f"session {session_id} not found",
                "pushed": 0,
                "chunks": 0,
                "samples": 0,
            }

        sess = self._sessions[session_id]
        handoff = sess.get("handoff")
        if handoff is None or handoff.closed:
            return {
                "ok": False,
                "session_id": session_id,
                "error": "handoff closed or missing",
                "pushed": 0,
                "chunks": 0,
                "samples": 0,
            }

        audio_data_b64 = body.get("audio_data_b64")
        if not audio_data_b64:
            return {
                "ok": False,
                "session_id": session_id,
                "error": "audio_data_b64 is empty",
                "pushed": 0,
                "chunks": 0,
                "samples": 0,
            }

        raw_bytes = base64.b64decode(audio_data_b64)
        audio_float = np.frombuffer(raw_bytes, dtype=np.float32).copy()
        audio_float = np.nan_to_num(audio_float, nan=0.0, posinf=0.0, neginf=0.0)
        audio_float = np.clip(audio_float, -1.0, 1.0).astype(np.float32, copy=False)

        total_samples = len(audio_float)
        STEP_SAMPLES = self.chunk_size * self.frame_len
        n_chunks = max(1, int(np.ceil(total_samples / STEP_SAMPLES)))

        print(f"[push_audio_batch] session={session_id} turnId={turn_id} "
              f"samples={total_samples} chunks={n_chunks}")

        t_enqueue_start = _time.monotonic()
        pushed = 0
        for ci in range(n_chunks):
            s = ci * STEP_SAMPLES
            e = min(s + STEP_SAMPLES, total_samples)
            chunk_pcm = audio_float[s:e].copy()
            if len(chunk_pcm) < STEP_SAMPLES:
                chunk_pcm = np.concatenate(
                    [chunk_pcm, np.zeros(STEP_SAMPLES - len(chunk_pcm), dtype=np.float32)]
                )
            meta = {
                "turnId": turn_id,
                "chunkIndex": ci,
                "isFirst": ci == 0,
                "isLast": ci == n_chunks - 1,
                "totalChunks": n_chunks,
                "generatedAt": generated_at,
                "enqueueAt": enqueue_at,
                "samples": len(chunk_pcm),
            }
            ok = handoff.push(chunk_pcm, metadata=meta, timeout=30.0)
            if ok:
                pushed += 1
            else:
                print(f"[push_audio_batch] chunk {ci}/{n_chunks} push FAILED (handoff closed/full)")

        enqueue_ms = round((_time.monotonic() - t_enqueue_start) * 1000, 1)
        print(f"[push_audio_batch] done: pushed={pushed}/{n_chunks} enqueue_ms={enqueue_ms}")

        return {
            "ok": pushed == n_chunks,
            "session_id": session_id,
            "turn_id": turn_id,
            "samples": total_samples,
            "chunks": n_chunks,
            "pushed": pushed,
            "enqueue_ms": enqueue_ms,
            "queue_depth": handoff.queue.qsize() if handoff else 0,
        }

    def _do_stream_fmp4_av(self, session_id: str):
        import json

        if session_id not in self._sessions:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': f'session {session_id} not found'})}\n\n"
            return

        sess = self._sessions[session_id]
        if sess.get("status") not in ("ready", "running", "ending"):
            status_val = sess.get("status", "unknown")
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': f'session status is {status_val}, cannot stream'})}\n\n"
            return

        existing_thread = sess.get("render_thread")
        if existing_thread is not None and existing_thread.is_alive():
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'render thread already running for this session'})}\n\n"
            yield f"event: end\ndata: {json.dumps({'type': 'end', 'mode': 'error', 'chunksRendered': 0, 'stopReason': 'double_start'})}\n\n"
            return

        render_thread = threading.Thread(
            target=self._run_session_render_impl,
            args=(session_id, 1000000, 200, "sine", "handoff"),
            daemon=True,
        )
        sess["render_thread"] = render_thread
        render_thread.start()

        sse_queue = sess.get("sse_queue")
        if sse_queue is None:
            yield f"event: error\ndata: {json.dumps({'type': 'error', 'error': 'no SSE queue in session'})}\n\n"
            return

        idle_sent = False
        while True:
            if not render_thread.is_alive():
                try:
                    sse_frame = sse_queue.get_nowait()
                    if sse_frame is not None:
                        yield sse_frame
                        continue
                except Exception:
                    pass
                break

            try:
                sse_frame = sse_queue.get(timeout=2.0)
                yield sse_frame
                idle_sent = True
            except Exception:
                if sess.get("status") in ("completed", "ended", "error"):
                    break
                if not idle_sent:
                    yield f"event: heartbeat\ndata: {{}}\n\n"
                    idle_sent = True

        try:
            while True:
                sse_frame = sse_queue.get_nowait()
                yield sse_frame
        except Exception:
            pass

        yield f"event: end\ndata: {json.dumps({'type': 'end', 'mode': 'idle', 'chunksRendered': sess.get('chunk_count', 0), 'stopReason': 'stream_complete'})}\n\n"

    @modal.fastapi_endpoint(method="POST", label="start-session")
    def http_start_session(self, body: dict) -> dict:
        return self._start_session_impl(
            session_id=body.get("session_id", ""),
            avatar_id=body.get("avatar_id", "arnold"),
        )

    @modal.fastapi_endpoint(method="POST", label="session-status")
    def http_session_status(self, body: dict) -> dict:
        session_id = body.get("session_id", "")
        if session_id not in self._sessions:
            return {
                "session_id": session_id,
                "status": "not_active",
                "active_sessions": list(self._sessions.keys()),
            }
        sess = self._sessions[session_id]
        import torch
        return {
            "session_id": session_id,
            "avatar_id": sess.get("avatar_id", ""),
            "status": sess.get("status", "unknown"),
            "chunk_count": sess.get("chunk_count", 0),
            "fmp4_media_count": sess.get("fmp4_media_count", 0),
            "fmp4_media_total_bytes": sess.get("fmp4_media_total_bytes", 0),
            "fmp4_init_segment": sess.get("fmp4_init_segment") is not None,
            "duration_s": round(time.monotonic() - sess.get("start_time", time.monotonic()), 2),
            "render_errors": [],
            "active_sessions": list(self._sessions.keys()),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }

    @modal.fastapi_endpoint(method="POST", label="push-audio")
    def http_push_audio(self, body: dict) -> dict:
        return self._do_push_audio(body)

    def _do_end_session_adapter(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            return {"session_id": session_id, "status": "not_active",
                    "active_sessions": list(self._sessions.keys())}

        sess = self._sessions[session_id]
        was_running = sess.get("adapter_running", False)

        if was_running:
            sess["adapter_stop_requested"] = True
            handoff = sess.get("handoff")
            if handoff is not None and not handoff.closed:
                handoff.close()

        return {
            "session_id": session_id,
            "ok": True,
            "was_running": was_running,
            "adapter_stop_requested": sess.get("adapter_stop_requested", False),
        }

    @modal.fastapi_endpoint(method="POST", label="end-session-adapter")
    def http_end_session_adapter(self, body: dict) -> dict:
        return self._do_end_session_adapter(body.get("session_id", ""))

    @modal.fastapi_endpoint(method="POST", label="stream-fmp4-av-adapter")
    def http_stream_fmp4_av_adapter(self, body: dict):
        from fastapi.responses import StreamingResponse

        session_id = body.get("session_id", "")
        return StreamingResponse(
            self._do_timeline_stream_fmp4_av_adapter(session_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @modal.fastapi_endpoint(method="POST", label="push-audio-batch")
    def http_push_audio_batch(self, body: dict) -> dict:
        return self._do_push_audio_batch(body)

    @modal.fastapi_endpoint(method="POST", label="tts-same-gpu-parallel")
    def http_tts_same_gpu_parallel(self, body: dict) -> dict:
        return self._do_tts_same_gpu_parallel(body)

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
        container_id = "session_l40s"

        if not text:
            return {"ok": False, "error": "No text provided"}
        if not getattr(self, "_omnivoice_ready", False) \
           or getattr(self, "_omnivoice_model", None) is None:
            return {"ok": False, "error": "OmniVoice model not loaded"}

        if enqueue_to_timeline and (not session_id or turn_id is None):
            return {"ok": False, "error": "enqueueToTimeline requires session_id and turnId"}

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

        audio_np = audio_data.cpu().numpy()
        if not np.all(np.isfinite(audio_np)):
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)

        validated_dur_ms = (len(audio_np) / actual_sr) * 1000
        rtf = audio_dur_ms / generate_ms if generate_ms > 0 else 0

        if enqueue_to_timeline:
            t_handoff_start = _time.time()
            STEP_SAMPLES = self.chunk_size * self.frame_len

            n_chunks = max(1, int(np.ceil(len(audio_np) / STEP_SAMPLES)))

            if session_id not in self._sessions:
                return {"ok": False, "error": f"session {session_id} not found"}

            sess = self._sessions[session_id]
            handoff = sess.get("handoff")
            if handoff is None or handoff.closed:
                return {"ok": False, "error": "handoff closed or missing"}

            for ci in range(n_chunks):
                s = ci * STEP_SAMPLES
                e = min(s + STEP_SAMPLES, len(audio_np))
                chunk_pcm = audio_np[s:e].copy()
                if len(chunk_pcm) < STEP_SAMPLES:
                    chunk_pcm = np.pad(chunk_pcm, (0, STEP_SAMPLES - len(chunk_pcm)))
                meta = {
                    "turnId": turn_id,
                    "chunkIndex": ci,
                    "isFirst": ci == 0,
                    "isLast": ci == n_chunks - 1,
                    "totalChunks": n_chunks,
                    "audioDurationMs": round(len(chunk_pcm) / actual_sr * 1000),
                    "sampleRate": actual_sr,
                    "generatedAt": _time.time(),
                    "textLength": len(text),
                }
                if not handoff.push(chunk_pcm, metadata=meta, timeout=30.0):
                    return {
                        "ok": False,
                        "error": "handoff closed or queue full",
                        "container_instance_id": container_id,
                        "chunksPushed": ci,
                    }

            handoff_ms = (_time.time() - t_handoff_start) * 1000
            total_tts_ms = (_time.time() - t_generate_start) * 1000

            print(
                f"[tts_parallel] chunked handoff turnId={turn_id} "
                f"chunks={n_chunks} durMs={validated_dur_ms:.0f} "
                f"gen={generate_ms:.0f}ms resample={resample_ms:.0f}ms "
                f"handoff={handoff_ms:.0f}ms total={total_tts_ms:.0f}ms"
            )

            return {
                "ok": True,
                "enqueued": True,
                "gpuTarget": "l40s_session",
                "streamingHandoff": True,
                "chunkCount": n_chunks,
                "sample_rate": actual_sr,
                "duration_ms": round(audio_dur_ms),
                "validated_duration_ms": round(validated_dur_ms),
                "latency_ms": round(generate_ms),
                "rtf": round(rtf, 2),
                "text_length": len(text),
                "provider": "omnivoice_same_gpu_parallel",
                "container_instance_id": container_id,
                "timing": {
                    "generateMs": round(generate_ms, 1),
                    "resampleMs": round(resample_ms, 1),
                    "handoffPushMs": round(handoff_ms, 1),
                    "totalTtsMs": round(total_tts_ms, 1),
                },
            }

        import base64
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio_np, actual_sr, subtype="PCM_16", format="WAV")
        buf.seek(0)
        wav_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        total_tts_ms = (_time.time() - t_generate_start) * 1000

        return {
            "ok": True,
            "wav_base64": wav_b64,
            "sample_rate": actual_sr,
            "duration_ms": round(audio_dur_ms),
            "validated_duration_ms": round(validated_dur_ms),
            "latency_ms": round(generate_ms),
            "rtf": round(rtf, 2),
            "text_length": len(text),
            "provider": "omnivoice_same_gpu_parallel",
            "container_instance_id": container_id,
            "timing": {
                "generateMs": round(generate_ms, 1),
                "resampleMs": round(resample_ms, 1),
                "totalTtsMs": round(total_tts_ms, 1),
            },
        }

    def _do_timeline_stream_fmp4_av_adapter(self, session_id: str):
        """Phase 5K: old-production sync generator shape + PyTorch inference.

        Faithful copy of modal_lifecycle_parallel_l40s_app.py::
        _do_timeline_stream_fmp4_av (lines 2208-3289), adapted ONLY at:
          - session state via self._sessions[session_id] (not instance vars)
          - inference: self.pipeline.process_chunk() with torch.inference_mode()
          - audio input: _AudioHandoff only (no modal.Queue Path B)
        """
        import collections
        import json
        import os
        import numpy as np
        import torch
        from avtr1_renderer.types import Chunk, RenderOptions

        def _sse(evt):
            t = evt.get("type", "message") if isinstance(evt, dict) else "message"
            return f"event: {t}\ndata: {json.dumps(evt)}\n\n"

        if not getattr(self, "_snap_ok", False):
            yield _sse({"type": "error", "error": "Pipeline not ready"})
            return

        if session_id not in self._sessions:
            yield _sse({"type": "error", "error": f"Session {session_id} not found"})
            return

        sess = self._sessions[session_id]
        if sess.get("adapter_running"):
            yield _sse({"type": "error", "error": "Adapter stream already running"})
            return

        sess["adapter_running"] = True
        sess["adapter_stop_requested"] = False
        adapter_stop_requested = False

        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        step = mg.chunk_size * mg.frame_len
        fps = 25.0
        AUDIO_SR = 16000
        silence_audio = np.zeros(window, dtype=np.float32)

        W, H = 1280, 720
        FRAME_BYTES = H * W * 3 // 2

        INIT_TIMEOUT_S = 10

        render_options = RenderOptions(
            pixel_format="yuv_i420",
            bg_id=self._bg_id,
        )

        fifo_path = f"/tmp/avtr1_adapter_audio_{session_id}"
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(fifo_path)
        print(f"[adapter_fmp4_av] FIFO created at {fifo_path}")

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

        print(f"[adapter_fmp4_av] starting FFmpeg...")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[adapter_fmp4_av] FFmpeg started pid={ffmpeg_proc.pid}")

        segment_queue = queue.Queue(maxsize=100)
        audio_write_queue = queue.Queue(maxsize=50)
        video_write_queue = queue.Queue(maxsize=50)
        ffmpeg_stderr_lines = []
        reader_error = []
        audio_writer_error = []
        video_writer_error = []

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
                                print(f"[adapter_stdout_reader] init complete {len(init_collected)}B")
                                segment_queue.put(("init", bytes(init_collected)))
                                init_collected = bytearray()
                        elif box_type == b"moof":
                            media_buf = bytearray(box_data)
                        elif box_type == b"mdat" and len(media_buf) > 0:
                            media_buf.extend(box_data)
                            _segment_count[0] += 1
                            if _segment_count[0] <= 3 or _segment_count[0] % 100 == 0:
                                print(f"[adapter_stdout_reader] media #{_segment_count[0]} {len(media_buf)}B")
                            segment_queue.put(("media", bytes(media_buf)))
                            media_buf = bytearray()

                segment_queue.put(None)
                print(f"[adapter_stdout_reader] EOF, total segments={_segment_count[0]}")
            except Exception as exc:
                reader_error.append(str(exc))
                print(f"[adapter_stdout_reader] exception: {exc}")
                segment_queue.put(None)

        def _audio_writer():
            print(f"[adapter_audio_writer] waiting to open FIFO {fifo_path}...")
            if ffmpeg_proc.poll() is not None:
                audio_writer_error.append(
                    f"FFmpeg exited (rc={ffmpeg_proc.returncode}) before audio pipe opened"
                )
                return
            try:
                with open(fifo_path, "wb") as audio_pipe:
                    print("[adapter_audio_writer] FIFO opened successfully")
                    while True:
                        item = audio_write_queue.get()
                        if item is None:
                            break
                        try:
                            audio_pipe.write(item)
                            audio_pipe.flush()
                        except BrokenPipeError:
                            break
            except OSError as exc:
                audio_writer_error.append(str(exc))

        def _video_writer():
            try:
                while True:
                    item = video_write_queue.get()
                    if item is None:
                        break
                    try:
                        ffmpeg_proc.stdin.write(item)
                        ffmpeg_proc.stdin.flush()
                    except BrokenPipeError:
                        break
            except Exception as exc:
                video_writer_error.append(str(exc))
            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass

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
            "sessionId": session_id,
            "fps": fps,
            "audioSampleRate": AUDIO_SR,
            "audioCodec": "aac",
            "videoCodec": "h264_nvenc",
            "samplesPerChunk": step,
            "adapter": True,
        }
        yield _sse(start_event)
        print(f"[adapter_fmp4_av] started session={session_id} ffmpeg_pid={ffmpeg_proc.pid}")

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
        speech_end_mono = None
        render_ms = 0.0
        encode_ms = 0.0

        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))
        CHUNK_MEDIA_S = 5 / fps
        HANDOFF_WAIT_S = 0.05

        class _ChunkMeta:
            __slots__ = ("mode", "turn_id", "segment_id", "chunk_index",
                         "is_speech_start", "is_speech_end",
                         "chunk_render_ms", "feed_ms", "frames_fed",
                         "overlap_with_tts", "enqueue_to_first_frame_ms",
                         "audio_samples_per_chunk")

            def __init__(self, **kw):
                for s in self.__slots__:
                    setattr(self, s, kw.get(s))

        pending_chunk_meta = collections.deque()

        def _push_chunk_meta(**kw):
            pending_chunk_meta.append(_ChunkMeta(**kw))

        class _SegmentSender:
            def __init__(self):
                self.pending = collections.deque()
                self.produced_media_s = 0.0
                self.target_lead_reached = False
                self.last_yield_mono = 0.0
                self._inter_yield_gaps = []
                self._total_pacing_sleep_ms = 0.0
                self._min_queue_depth = 999999
                self._max_queue_depth = 0
                self._dry_spell_count = 0

            def add(self, seg_type, seg_data):
                self.pending.append((seg_type, seg_data))

            def yield_events(self):
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
                    break

                return results

            def drain_all(self):
                results = []
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

        sender = _SegmentSender()
        speech_first_segment_emitted = False
        speech_end_emitted = False
        speech_segments = 0
        idle_segments = 0

        _audio_write_count = [0]
        _video_write_count = [0]
        eof_from_reader = False

        def _queue_audio(data, label):
            nonlocal total_audio_bytes_written
            try:
                audio_write_queue.put(data, timeout=2)
                total_audio_bytes_written += len(data)
                _audio_write_count[0] += 1
                if _audio_write_count[0] <= 3 or _audio_write_count[0] % 100 == 0:
                    print(f"[adapter_fmp4_av] audio queued: {len(data)}B #{_audio_write_count[0]} ({label})")
                return True
            except queue.Full:
                print(f"[adapter_fmp4_av] audio_write_queue FULL ({label})")
                return False

        def _write_video_frames(frames_list, label):
            nonlocal total_raw_bytes_written
            raw_chunk = b"".join(frame.data.tobytes() for frame in frames_list)
            if len(frames_list) > 0:
                assert len(raw_chunk) == len(frames_list) * FRAME_BYTES, (
                    f"Frame bytes mismatch: {len(raw_chunk)} != {len(frames_list)} * {FRAME_BYTES}"
                )
            try:
                video_write_queue.put(raw_chunk, timeout=2)
                total_raw_bytes_written += len(raw_chunk)
                _video_write_count[0] += 1
                if _video_write_count[0] <= 3 or _video_write_count[0] % 100 == 0:
                    print(f"[adapter_fmp4_av] video queued: {len(raw_chunk)}B ({len(frames_list)} frames, {label})")
                return True
            except queue.Full:
                print(f"[adapter_fmp4_av] video_write_queue FULL ({label})")
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
                        print(f"[adapter_fmp4_av] init segment {len(seg_data)} bytes")
                    elif seg_type == "media":
                        sender.add(seg_type, seg_data)
                        total_encoded_bytes_sent += len(seg_data)
                except queue.Empty:
                    return

        def _stamp_and_yield_media(evt, meta, pacing_extra=None):
            nonlocal media_sequence, speech_first_segment_emitted, speech_end_emitted
            nonlocal speech_segments, idle_segments
            evt["mode"] = meta.mode if meta.mode else current_mode
            if evt["mode"] == "speech":
                speech_segments += 1
            else:
                idle_segments += 1
            evt["turnId"] = meta.turn_id if meta.turn_id else None
            evt["chunkIndex"] = meta.chunk_index if meta.chunk_index is not None else cumulative_chunk_index
            timing = {
                "chunkRenderMs": round(meta.chunk_render_ms, 1) if meta.chunk_render_ms else 0,
                "feedMs": round(meta.feed_ms, 1) if meta.feed_ms else 0,
                "framesFed": meta.frames_fed or 0,
                "audioSamplesPerChunk": step,
            }
            if pacing_extra:
                timing.update(pacing_extra)

            evt["adapter"] = True

            if meta.mode == "speech" and meta.is_speech_start and not speech_first_segment_emitted:
                evt["isSpeechStart"] = True
                speech_first_segment_emitted = True
                if meta.enqueue_to_first_frame_ms and meta.enqueue_to_first_frame_ms > 0:
                    timing["enqueueToFirstFrameMs"] = round(meta.enqueue_to_first_frame_ms, 1)
            else:
                evt["isSpeechStart"] = False

            if meta.mode == "speech" and meta.is_speech_end and not speech_end_emitted:
                evt["isSpeechEnd"] = True
                speech_end_emitted = True
            else:
                evt["isSpeechEnd"] = False

            evt["timing"] = timing
            media_sequence += 1
            return evt

        def _yield_sender_events():
            _drain_into_sender()
            paced = sender.yield_events()
            for item in paced:
                if isinstance(item, dict):
                    yield _sse(item)
                else:
                    _, _, evt, meta = item
                    _stamp_and_yield_media(evt, meta)
                    yield _sse(evt)

        idle_chunk_index = 0
        cumulative_chunk_index = 0
        timeline_segment_id = 0

        handoff = sess.get("handoff")
        audio_buffer = np.zeros(0, dtype=np.float32)
        handoff_turn_id = None
        handoff_enqueue_at = None
        handoff_first_segment_emitted = False
        handoff_end_emitted = False
        handoff_chunk_index = 0
        handoff_is_last_received = False

        def _yield_rendered(current_mode_label, turn_id_for_event, is_speech_start,
                            is_speech_end, enqueue_at_val, speech_first_segment,
                            speech_end_segment, extra_timing=None):
            nonlocal cumulative_chunk_index, chunks_rendered, eof_from_reader
            nonlocal speech_first_segment_emitted, speech_end_emitted

            enqueue_to_first = None
            if is_speech_start and not speech_first_segment_emitted and enqueue_at_val and enqueue_at_val > 0:
                enqueue_to_first = (time.time() - enqueue_at_val) * 1000

            _push_chunk_meta(
                mode=current_mode_label,
                turn_id=turn_id_for_event,
                segment_id=timeline_segment_id,
                chunk_index=cumulative_chunk_index,
                is_speech_start=is_speech_start and not speech_first_segment_emitted,
                is_speech_end=is_speech_end and not speech_end_emitted,
                chunk_render_ms=render_ms,
                feed_ms=encode_ms,
                frames_fed=0,
                overlap_with_tts=False,
                enqueue_to_first_frame_ms=enqueue_to_first,
                audio_samples_per_chunk=step,
            )

            yield from _yield_sender_events()

            cumulative_chunk_index += 1
            chunks_rendered += 1

        try:
            while not adapter_stop_requested and not eof_from_reader:
                if not init_sent_to_client and not init_5s_warned and time.monotonic() > stream_start_mono + 5:
                    init_5s_warned = True
                    print("[adapter_fmp4_av] WARNING: no init after 5s")

                if not init_sent_to_client and time.monotonic() > init_deadline_mono:
                    print(f"[adapter_fmp4_av] INIT TIMEOUT")
                    err_event = {"type": "error", "error": f"fMP4 AV init timeout ({INIT_TIMEOUT_S}s)"}
                    yield _sse(err_event)
                    break

                handoff_item = None
                if handoff is not None and not handoff.closed:
                    should_wait = (current_mode == "speech" and not handoff_is_last_received)
                    item = handoff.pop(timeout_s=HANDOFF_WAIT_S if should_wait else 0.01)
                    if item is not None:
                        handoff_item = item
                    elif handoff.closed:
                        break

                if handoff_item is not None:
                    chunk_pcm, meta = handoff_item
                    chunk_turn_id = meta.get("turnId")
                    chunk_index = meta.get("chunkIndex", 0)
                    is_first = meta.get("isFirst", False)
                    is_last = meta.get("isLast", False)

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
                        print(f"[adapter_fmp4_av] speech start turnId={chunk_turn_id}")

                    audio_buffer = np.concatenate([audio_buffer, chunk_pcm])

                    while len(audio_buffer) >= window:
                        windowed = audio_buffer[:window].copy()
                        audio_buffer = audio_buffer[step:]

                        ffmpeg_audio = windowed[:step]
                        audio_s16 = (ffmpeg_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
                        if not _queue_audio(audio_s16, f"speech {handoff_chunk_index}"):
                            adapter_stop_requested = True
                            break

                        render_t0 = time.monotonic()
                        chunk_obj = Chunk(audio_speech=windowed, audio_listen=silence_audio)
                        with torch.inference_mode():
                            new_state, frames_iter = self.pipeline.process_chunk(
                                sess["avatar"], chunk_obj,
                                state=sess["render_state"],
                                options=render_options,
                            )
                            frames_list = list(frames_iter)
                        sess["render_state"] = new_state
                        render_ms = (time.monotonic() - render_t0) * 1000

                        encode_t0 = time.monotonic()
                        if not _write_video_frames(frames_list, f"speech {handoff_chunk_index}"):
                            adapter_stop_requested = True
                            break
                        encode_ms = (time.monotonic() - encode_t0) * 1000

                        speech_first_segment = not handoff_first_segment_emitted

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
                            _queue_audio(audio_s16, "speech final")

                            render_t0 = time.monotonic()
                            chunk_obj = Chunk(audio_speech=padded, audio_listen=silence_audio)
                            with torch.inference_mode():
                                new_state, frames_iter = self.pipeline.process_chunk(
                                    sess["avatar"], chunk_obj,
                                    state=sess["render_state"],
                                    options=render_options,
                                )
                                frames_list = list(frames_iter)
                            sess["render_state"] = new_state
                            render_ms = (time.monotonic() - render_t0) * 1000

                            encode_t0 = time.monotonic()
                            _write_video_frames(frames_list, "speech final")
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
                                segment_id=timeline_segment_id,
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

                        timeline_segment_id += 1
                        idle_chunk_index = 0
                        current_mode = "idle"
                        speech_end_mono = time.monotonic()
                        print(f"[adapter_fmp4_av] speech done turnId={handoff_turn_id}")

                    continue

                if handoff is not None and handoff.closed:
                    break

                if sess.get("adapter_stop_requested"):
                    adapter_stop_requested = True
                    break

                current_mode = "idle"

                silence_s16 = np.zeros(step, dtype=np.int16).tobytes()
                if not _queue_audio(silence_s16, f"idle {cumulative_chunk_index}"):
                    adapter_stop_requested = True
                    break

                render_t0 = time.monotonic()
                chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
                with torch.inference_mode():
                    new_state, frames_iter = self.pipeline.process_chunk(
                        sess["avatar"], chunk_obj,
                        state=sess["render_state"],
                        options=render_options,
                    )
                    frames_list = list(frames_iter)
                sess["render_state"] = new_state
                render_ms = (time.monotonic() - render_t0) * 1000

                encode_t0 = time.monotonic()
                if not _write_video_frames(frames_list, f"idle {cumulative_chunk_index}"):
                    adapter_stop_requested = True
                    break
                encode_ms = (time.monotonic() - encode_t0) * 1000

                yield from _yield_rendered(
                    "idle", None,
                    is_speech_start=False,
                    is_speech_end=False,
                    enqueue_at_val=None,
                    speech_first_segment=False,
                    speech_end_segment=False,
                )

                idle_chunk_index += 1

        except Exception as exc:
            import traceback
            traceback.print_exc()
            err_event = {"type": "error", "error": f"Adapter fMP4 A/V stream failed: {exc}"}
            yield _sse(err_event)
        finally:
            print(f"[adapter_fmp4_av] cleanup starting ffmpeg_pid={ffmpeg_proc.pid}")

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
            except Exception:
                pass
            try:
                audio_thread.join(timeout=5)
            except Exception:
                pass
            try:
                stdout_thread.join(timeout=3)
            except Exception:
                pass
            try:
                stderr_thread.join(timeout=3)
            except Exception:
                pass

            if ffmpeg_proc.poll() is None:
                print(f"[adapter_fmp4_av] terminating ffmpeg pid={ffmpeg_proc.pid}")
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            print(f"[adapter_fmp4_av] ffmpeg exited rc={ffmpeg_proc.returncode}")

            try:
                os.unlink(fifo_path)
            except OSError:
                pass

            _drain_into_sender()
            remaining = sender.drain_all()
            for item in remaining:
                if isinstance(item, dict):
                    yield _sse(item)
                else:
                    _, _, evt, meta = item
                    _stamp_and_yield_media(evt, meta)
                    yield _sse(evt)

            stderr_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
            if stderr_errors:
                print(f"[adapter_fmp4_av] FFmpeg stderr errors ({len(stderr_errors)}): {stderr_errors[:5]}")

            _gaps = sorted(sender._inter_yield_gaps) if sender._inter_yield_gaps else []
            _gap_p50 = _gaps[len(_gaps) // 2] if _gaps else 0
            _gap_p95 = _gaps[int(len(_gaps) * 0.95)] if _gaps else 0
            _gap_max = _gaps[-1] if _gaps else 0

            end_event = {
                "type": "end",
                "chunksRendered": chunks_rendered,
                "mediaSegments": media_sequence,
                "speechSegments": speech_segments,
                "idleSegments": idle_segments,
                "audioBytes": total_audio_bytes_written,
                "videoBytes": total_raw_bytes_written,
                "encodedBytes": total_encoded_bytes_sent,
                "ffmpegReturnCode": ffmpeg_proc.returncode,
                "ffmpegStderrErrorCount": len(stderr_errors),
                "adapter": True,
                "cadenceStats": {
                    "targetLeadS": TARGET_LEAD_S,
                    "chunkMediaS": CHUNK_MEDIA_S,
                    "totalPacingSleepMs": round(sender._total_pacing_sleep_ms, 1),
                    "modalSegmentsYielded": media_sequence,
                    "speechSegments": speech_segments,
                    "idleSegments": idle_segments,
                    "interYieldGapMs": {
                        "p50": round(_gap_p50, 1),
                        "p95": round(_gap_p95, 1),
                        "max": round(_gap_max, 1),
                        "sampleCount": len(_gaps),
                    },
                    "senderMinQueueDepth": sender._min_queue_depth if sender._min_queue_depth != 999999 else 0,
                    "senderMaxQueueDepth": sender._max_queue_depth,
                    "senderDrySpellCount": sender._dry_spell_count,
                },
            }
            yield _sse(end_event)

            sess["adapter_running"] = False
            sess["adapter_results"] = {
                "ffmpeg_rc": ffmpeg_proc.returncode,
                "chunks_rendered": chunks_rendered,
                "media_segments": media_sequence,
                "audio_bytes": total_audio_bytes_written,
                "video_bytes": total_raw_bytes_written,
                "encoded_bytes": total_encoded_bytes_sent,
                "ffmpeg_stderr_errors": len(stderr_errors),
            }

            print(f"[adapter_fmp4_av] cleanup complete chunks={chunks_rendered} segments={media_sequence} "
                  f"audioBytes={total_audio_bytes_written} videoBytes={total_raw_bytes_written}")

    @modal.method()
    def fmp4_video_stream_smoke(self, avatar_id: str, num_chunks: int = 5,
                                chunk_ms: int = 200) -> dict:
        import torch
        import numpy as np
        import os

        t_total_start = time.monotonic()
        FPS = 25
        H = 720
        W = 1280
        FRAME_BYTES = H * W * 3 // 2

        av = self._get_avatar(avatar_id)

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        print(f"[fmp4] FFmpeg PID={ffmpeg_proc.pid} spawned")

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        ffmpeg_stderr_lines: list[str] = []
        reader_error_holder: list[str] = []

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
            except Exception as e:
                reader_error_holder.append(str(e))
                try:
                    segment_queue.put(None)
                except Exception:
                    pass

        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        stderr_thread.start()
        stdout_thread.start()

        stream_start_mono = time.monotonic()
        sender = _SegmentSender(
            chunk_media_s=CHUNK_MEDIA_S,
            target_lead_s=TARGET_LEAD_S,
            stream_start_mono=stream_start_mono,
        )
        sess["_speech_first_segment_emitted"] = False
        sess["_speech_end_segment_emitted"] = False
        sess["_current_turn_id"] = None
        sess["_sender_ref"] = sender

        from avtr1_renderer.types import Chunk, RenderOptions
        mg = self.pipeline._motion_generator
        window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        silence_audio = np.zeros(window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)
        render_state = None

        chunks_result: list[dict] = []
        total_raw_bytes_written = 0
        total_encoded_bytes = 0
        all_frames_valid = True
        any_nan = False
        any_inf = False

        for chunk_i in range(num_chunks):
            t_chunk_start = time.monotonic()
            chunk_timings: dict = {"chunk_index": chunk_i}

            chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)
            with torch.inference_mode():
                render_state, frames_iter = self.pipeline.process_chunk(
                    av, chunk_obj, state=render_state, options=_render_options,
                )
                frames_list = list(frames_iter)
            raw_chunk = b"".join(f.data.tobytes() for f in frames_list)
            total_raw_bytes_written += len(raw_chunk)

            try:
                ffmpeg_proc.stdin.write(raw_chunk)
                ffmpeg_proc.stdin.flush()
            except BrokenPipeError:
                pass

            chunk_timings["total_render_ms"] = round((time.monotonic() - t_chunk_start) * 1000, 2)
            chunks_result.append(chunk_timings)

        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass

        while True:
            try:
                item = segment_queue.get(timeout=2.0)
                if item is None:
                    break
                sender.add(item[0], item[1])
                total_encoded_bytes += len(item[1])
            except queue.Empty:
                break

        drained = sender.drain_all()
        if drained:
            for evt in drained:
                total_encoded_bytes += evt.get("byteLength", 0)

        stderr_thread.join(timeout=3)
        stdout_thread.join(timeout=3)

        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait(timeout=2)

        ffmpeg_rc = ffmpeg_proc.returncode
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        init_info = sender.collected_init
        media_events = sender.collected_media

        total_wall_s = time.monotonic() - t_total_start
        result = {
            "ok": True,
            "avatar_id": avatar_id,
            "chunks": num_chunks,
            "total_wall_s": round(total_wall_s, 4),
            "init_generated": init_info is not None,
            "media_count": len(media_events),
            "ffmpeg_rc": ffmpeg_rc,
            "bytes": {
                "raw_written": total_raw_bytes_written,
                "encoded_total": total_encoded_bytes,
            },
            "vram_mb": vram_mb,
            "chunks_detail": chunks_result,
            "reader_errors": reader_error_holder,
        }

        print(f"[fmp4] === fmp4_video_stream_smoke done: {num_chunks} chunks in {total_wall_s:.2f}s ===")
        return result

    @modal.method()
    def fmp4_video_stream_concurrent_smoke(self, avatar_id: str, num_chunks: int = 10,
                                            chunk_ms: int = 200) -> dict:
        import os
        import torch

        t_total_start = time.monotonic()
        FPS = 25
        H = 720
        W = 1280
        N_FRAMES_PER_CHUNK = self.chunk_size
        FRAME_BYTES = H * W * 3 // 2
        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))
        CHUNK_MEDIA_S = N_FRAMES_PER_CHUNK / FPS

        print(f"[fmp4-concurrent] === fmp4_video_stream_concurrent_smoke: avatar={avatar_id} "
              f"chunks={num_chunks} chunk_ms={chunk_ms} ===")
        print(f"[fmp4-concurrent] Video: {W}x{H} @ {FPS}fps, {N_FRAMES_PER_CHUNK} frames/chunk, "
              f"FRAME_BYTES={FRAME_BYTES}, CHUNK_MEDIA_S={CHUNK_MEDIA_S:.3f}s")

        av = self._get_avatar(avatar_id)
        avatar_load_s = time.monotonic() - t_total_start

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        print(f"[fmp4-concurrent] Spawning FFmpeg: {' '.join(ffmpeg_cmd)}")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[fmp4-concurrent] FFmpeg PID={ffmpeg_proc.pid} spawned")

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        ffmpeg_stderr_lines: list[str] = []
        reader_error_holder: list[str] = []

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
                print("[fmp4-concurrent-stdout] EOF, sentinel sent")
            except Exception as e:
                reader_error_holder.append(str(e))
                print(f"[fmp4-concurrent-stdout] exception: {e}")
                try:
                    segment_queue.put(None)
                except Exception:
                    pass

        producer_done_event = threading.Event()
        producer_timings: list[dict] = []
        total_raw_written = [0]
        _producer_start = [0.0]
        _producer_end = [0.0]

        from avtr1_renderer.types import Chunk, RenderOptions
        _mg = self.pipeline._motion_generator
        _window = (_mg.chunk_size + _mg.future_size) * _mg.frame_len + _mg.audio_shift
        _silence_audio = np.zeros(_window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        def _video_producer():
            _producer_start[0] = time.monotonic()
            render_state = None

            for chunk_i in range(num_chunks):
                t_chunk_start = time.monotonic()
                chunk_timings: dict = {"chunk_index": chunk_i}

                chunk_obj = Chunk(audio_speech=_silence_audio, audio_listen=_silence_audio)
                with torch.inference_mode():
                    render_state, frames_iter = self.pipeline.process_chunk(
                        av, chunk_obj, state=render_state, options=_render_options,
                    )
                    frames_list = list(frames_iter)
                raw_chunk = b"".join(f.data.tobytes() for f in frames_list)

                chunk_timings["render_total_ms"] = round((time.monotonic() - t_chunk_start) * 1000, 2)

                t_write_start = time.monotonic()
                try:
                    ffmpeg_proc.stdin.write(raw_chunk)
                    ffmpeg_proc.stdin.flush()
                    total_raw_written[0] += len(raw_chunk)
                except BrokenPipeError:
                    print(f"[fmp4-producer] WARNING: BrokenPipeError chunk {chunk_i}")
                chunk_timings["ffmpeg_write_ms"] = round((time.monotonic() - t_write_start) * 1000, 2)
                chunk_timings["chunk_wall_ms"] = round((time.monotonic() - t_chunk_start) * 1000, 2)

                print(f"[fmp4-producer] chunk {chunk_i}: render={chunk_timings['render_total_ms']}ms "
                      f"write={chunk_timings['ffmpeg_write_ms']}ms "
                      f"wall={chunk_timings['chunk_wall_ms']}ms")

                producer_timings.append(chunk_timings)

            try:
                ffmpeg_proc.stdin.close()
                print("[fmp4-producer] FFmpeg stdin closed")
            except Exception:
                pass

            _producer_end[0] = time.monotonic()
            producer_done_event.set()
            print(f"[fmp4-producer] DONE: {num_chunks} chunks in "
                  f"{_producer_end[0] - _producer_start[0]:.2f}s")

        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        producer_thread = threading.Thread(target=_video_producer, daemon=True)

        stderr_thread.start()
        stdout_thread.start()
        producer_thread.start()

        stream_start_mono = time.monotonic()
        sender = _SegmentSender(
            chunk_media_s=CHUNK_MEDIA_S,
            target_lead_s=TARGET_LEAD_S,
            stream_start_mono=stream_start_mono,
        )

        sentinel_seen = False
        total_encoded_bytes = 0
        consumer_start_mono = time.monotonic()
        max_consumer_s = 120.0
        first_media_wall_ms = 0.0

        print("[fmp4-concurrent] consumer loop starting")

        while True:
            if time.monotonic() - consumer_start_mono > max_consumer_s:
                print("[fmp4-concurrent] CONSUMER TIMEOUT")
                break

            got_any = False
            while True:
                try:
                    item = segment_queue.get_nowait()
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                    if item[0] == "media" and first_media_wall_ms == 0.0:
                        first_media_wall_ms = round((time.monotonic() - stream_start_mono) * 1000, 1)
                    got_any = True
                except queue.Empty:
                    break

            paced = sender.yield_events()

            if sentinel_seen:
                break

            if not got_any and not stdout_thread.is_alive():
                try:
                    item = segment_queue.get(timeout=1.0)
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                    if item[0] == "media" and first_media_wall_ms == 0.0:
                        first_media_wall_ms = round((time.monotonic() - stream_start_mono) * 1000, 1)
                except queue.Empty:
                    print("[fmp4-concurrent] stdout thread dead, queue empty, exiting consumer")
                    break

            if not got_any:
                time.sleep(0.01)

        drained_final = sender.drain_all()
        consumer_end_mono = time.monotonic()

        print(f"[fmp4-concurrent] consumer loop done: sentinel={sentinel_seen} "
              f"sender_media={len(sender.collected_media)} "
              f"drained_final={len(drained_final)} "
              f"consumer_s={consumer_end_mono - consumer_start_mono:.2f}")

        producer_thread.join(timeout=30)
        stderr_thread.join(timeout=3)
        stdout_thread.join(timeout=3)

        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait(timeout=2)

        try:
            zombie_check = subprocess.run(
                ["ps", "-p", str(ffmpeg_proc.pid)],
                capture_output=True, timeout=2,
            )
            zombie_status = "zombie_still_running" if zombie_check.returncode == 0 else "clean"
        except Exception:
            zombie_status = "unknown"

        ffmpeg_rc = ffmpeg_proc.returncode
        n_stderr_lines = len(ffmpeg_stderr_lines)
        ffmpeg_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
        ffmpeg_warnings = [l for l in ffmpeg_stderr_lines if "warning" in l.lower()]

        print(f"[fmp4-concurrent] FFmpeg exit: rc={ffmpeg_rc} stderr_lines={n_stderr_lines} "
              f"errors={len(ffmpeg_errors)} warnings={len(ffmpeg_warnings)} zombie={zombie_status}")

        p_start = _producer_start[0]
        p_end = _producer_end[0]
        c_start = consumer_start_mono
        c_end = consumer_end_mono
        overlap_s = max(0.0, min(p_end, c_end) - max(p_start, c_start))

        init_info = sender.collected_init
        media_events = sender.collected_media
        first_init_latency_ms = init_info.get("timing", {}).get("timeToInitMs", 0) if init_info else 0

        pacing_inter_yield_gaps = sender._inter_yield_gaps

        total_wall_s = time.monotonic() - t_total_start
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        result = {
            "avatar_id": avatar_id,
            "num_chunks": num_chunks,
            "chunk_ms_target": chunk_ms,
            "pacing_mode": "concurrent",
            "avatar_load_s": round(avatar_load_s, 4),
            "total_wall_s": round(total_wall_s, 4),
            "restore_timings": getattr(self, "_restore_timings", {}),
            "video_config": {
                "width": W,
                "height": H,
                "fps": FPS,
                "frames_per_chunk": N_FRAMES_PER_CHUNK,
                "frame_bytes_i420": FRAME_BYTES,
                "chunk_media_s": round(CHUNK_MEDIA_S, 4),
            },
            "init_segment": {
                "generated": init_info is not None,
                "byte_count": init_info.get("byteLength", 0) if init_info else 0,
                "latency_ms": first_init_latency_ms,
            },
            "media_segments": {
                "count": len(media_events),
                "total_bytes": sum(e["byteLength"] for e in media_events),
                "sizes": [e["byteLength"] for e in media_events],
                "first_latency_ms": first_media_wall_ms,
            },
            "pacing": {
                "target_lead_s": TARGET_LEAD_S,
                "chunk_media_s": round(CHUNK_MEDIA_S, 4),
                "total_pacing_sleep_ms": sender.total_pacing_sleep_ms,
                "inter_yield_gap_ms": {
                    "min": min(pacing_inter_yield_gaps) if pacing_inter_yield_gaps else 0,
                    "max": max(pacing_inter_yield_gaps) if pacing_inter_yield_gaps else 0,
                    "avg": round(sum(pacing_inter_yield_gaps) / len(pacing_inter_yield_gaps), 1) if pacing_inter_yield_gaps else 0,
                },
                "queue_depth": {
                    "min": sender.min_queue_depth,
                    "max": sender.max_queue_depth,
                },
                "dry_spell_count": sender.dry_spell_count,
                "target_lead_reached": sender.target_lead_reached,
            },
            "concurrency": {
                "producer_total_s": round(p_end - p_start, 4),
                "consumer_total_s": round(c_end - c_start, 4),
                "overlap_s": round(overlap_s, 4),
                "producer_done_before_consumer_done": p_end <= c_end,
                "parser_sentinel_seen": sentinel_seen,
            },
            "ffmpeg": {
                "return_code": ffmpeg_rc,
                "stderr_line_count": n_stderr_lines,
                "fatal_error_detected": len(ffmpeg_errors) > 0 and ffmpeg_rc != 0,
                "stderr_error_count": len(ffmpeg_errors),
                "stderr_warning_count": len(ffmpeg_warnings),
                "zombie_check": zombie_status,
                "pid": ffmpeg_proc.pid,
            },
            "bytes": {
                "raw_written": total_raw_written[0],
                "encoded_total": total_encoded_bytes,
                "compression_ratio": round(total_raw_written[0] / total_encoded_bytes, 2) if total_encoded_bytes > 0 else 0,
            },
            "vram_mb": vram_mb,
            "chunks": producer_timings,
            "reader_errors": reader_error_holder,
        }

        print(f"[fmp4-concurrent] === fmp4_video_stream_concurrent_smoke done: "
              f"{num_chunks} chunks in {total_wall_s:.2f}s ===")
        print(f"[fmp4-concurrent] init={init_info is not None} media_count={len(media_events)} "
              f"rawMB={total_raw_written[0]/1e6:.1f} encodedMB={total_encoded_bytes/1e6:.1f} "
              f"overlap={overlap_s:.2f}s sentinel={sentinel_seen} vram={vram_mb}MB")

        return result

    @modal.method()
    def audio_handoff_stream_smoke(self, avatar_id: str, num_chunks: int = 10,
                                   chunk_ms: int = 200,
                                   audio_mode: str = "sine") -> dict:
        return self._audio_handoff_stream_impl(avatar_id, num_chunks, chunk_ms, audio_mode, cadence=False)

    def _audio_handoff_stream_impl(self, avatar_id: str, num_chunks: int = 10,
                                   chunk_ms: int = 200,
                                   audio_mode: str = "sine",
                                   cadence: bool = False) -> dict:
        import os
        import torch
        import numpy as np

        t_total_start = time.monotonic()
        FPS = 25
        H = 720
        W = 1280
        N_FRAMES_PER_CHUNK = self.chunk_size
        FRAME_BYTES = H * W * 3 // 2
        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))
        CHUNK_MEDIA_S = N_FRAMES_PER_CHUNK / FPS

        STEP_SAMPLES = self.chunk_size * self.frame_len
        AUDIO_SR = 16000

        from avtr1_renderer.types import Chunk, RenderOptions
        _mg = self.pipeline._motion_generator
        _window = (_mg.chunk_size + _mg.future_size) * _mg.frame_len + _mg.audio_shift
        _silence_audio = np.zeros(_window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        audio_mode_resolved = audio_mode if audio_mode in ("sine", "silence") else "sine"

        HANDOFF_WAIT_S = 0.05

        print(f"[audio-handoff] === audio_handoff_stream_smoke: avatar={avatar_id} "
              f"chunks={num_chunks} chunk_ms={chunk_ms} audio_mode={audio_mode_resolved} ===")
        print(f"[audio-handoff] Video: {W}x{H} @ {FPS}fps, {N_FRAMES_PER_CHUNK} frames/chunk, "
              f"FRAME_BYTES={FRAME_BYTES}, CHUNK_MEDIA_S={CHUNK_MEDIA_S:.3f}s")
        print(f"[audio-handoff] Audio: step={STEP_SAMPLES} samples, window_len={_window}, "
              f"sr={AUDIO_SR}, mode={audio_mode_resolved}")
        assert STEP_SAMPLES == self.chunk_size * self.frame_len

        av = self._get_avatar(avatar_id)
        avatar_load_s = time.monotonic() - t_total_start

        handoff = _AudioHandoff(maxsize=100)

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "pipe:0",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        print(f"[audio-handoff] Spawning FFmpeg: {' '.join(ffmpeg_cmd)}")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[audio-handoff] FFmpeg PID={ffmpeg_proc.pid} spawned")

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        ffmpeg_stderr_lines: list[str] = []
        reader_error_holder: list[str] = []

        audio_producer_start = [0.0]
        audio_producer_end = [0.0]
        audio_push_intervals: list[float] = []
        cadence_push_times: list[float] = []

        def _audio_producer():
            audio_producer_start[0] = time.monotonic()
            loop_start = audio_producer_start[0]
            chunk_interval_s = chunk_ms / 1000.0

            for ci in range(num_chunks):
                if cadence:
                    target_time = loop_start + ci * chunk_interval_s
                    sleep_s = target_time - time.monotonic()
                    if sleep_s > 0.001:
                        time.sleep(sleep_s)

                t_push_start = time.monotonic()

                if audio_mode_resolved == "sine":
                    freq = 200.0
                    t_arr = np.arange(STEP_SAMPLES, dtype=np.float32) / AUDIO_SR
                    chunk_audio = (0.5 * np.sin(2.0 * np.pi * freq * t_arr)).astype(np.float32)
                else:
                    chunk_audio = np.zeros(STEP_SAMPLES, dtype=np.float32)

                meta = {
                    "chunkIndex": ci,
                    "isFirst": ci == 0,
                    "isLast": ci == num_chunks - 1,
                    "samples": len(chunk_audio),
                }

                ok = handoff.push(chunk_audio, metadata=meta, timeout=30.0)
                if not ok:
                    print(f"[audio-producer] push FAILED at chunk {ci}")
                    break

                push_wall_ms = (time.monotonic() - t_push_start) * 1000
                audio_push_intervals.append(round(push_wall_ms, 2))

                if cadence:
                    cadence_push_times.append(round((time.monotonic() - loop_start) * 1000, 2))
                else:
                    if ci < num_chunks - 1:
                        elapsed_in_interval = time.monotonic() - t_push_start
                        sleep_s = chunk_interval_s - elapsed_in_interval
                        if sleep_s > 0.001:
                            time.sleep(sleep_s)

            handoff.close()
            audio_producer_end[0] = time.monotonic()
            total_s = audio_producer_end[0] - audio_producer_start[0]
            cadence_label = " cadence=scheduled" if cadence else ""
            print(f"[audio-producer] DONE: {num_chunks} chunks pushed in {total_s:.2f}s "
                  f"mode={audio_mode_resolved}{cadence_label}")

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
                print("[audio-handoff-stdout] EOF, sentinel sent")
            except Exception as e:
                reader_error_holder.append(str(e))
                print(f"[audio-handoff-stdout] exception: {e}")
                try:
                    segment_queue.put(None)
                except Exception:
                    pass

        producer_done_event = threading.Event()
        producer_timings: list[dict] = []
        total_raw_written = [0]
        _producer_start = [0.0]
        _producer_end = [0.0]

        from avtr1_renderer.types import Chunk, RenderOptions
        _mg = self.pipeline._motion_generator
        _window = (_mg.chunk_size + _mg.future_size) * _mg.frame_len + _mg.audio_shift
        _silence_audio = np.zeros(_window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        def _video_producer():
            _producer_start[0] = time.monotonic()
            render_state = None
            audio_buffer = np.zeros(0, dtype=np.float32)
            is_speech = False

            for chunk_i in range(num_chunks):
                t_chunk_start = time.monotonic()
                chunk_timings: dict = {"chunk_index": chunk_i}

                t_audio_pop_start = time.monotonic()
                pop_result = handoff.pop(timeout_s=HANDOFF_WAIT_S if is_speech else 0)
                audio_wait_ms = round((time.monotonic() - t_audio_pop_start) * 1000, 2)
                chunk_timings["audio_wait_ms"] = audio_wait_ms

                if pop_result is not None:
                    chunk_audio, chunk_meta = pop_result
                    audio_underflow = False
                    is_speech = True
                else:
                    chunk_audio = np.zeros(STEP_SAMPLES, dtype=np.float32)
                    audio_underflow = True
                    is_speech = False

                chunk_timings["audio_underflow"] = audio_underflow
                chunk_timings["audio_chunk_samples"] = len(chunk_audio)

                audio_buffer = np.concatenate([audio_buffer, chunk_audio])

                if len(audio_buffer) >= _window:
                    audio_tail = audio_buffer[-_window:].copy()
                else:
                    audio_tail = np.pad(audio_buffer, (_window - len(audio_buffer), 0))

                t_render_start = time.monotonic()
                chunk_obj = Chunk(audio_speech=audio_tail, audio_listen=_silence_audio)
                with torch.inference_mode():
                    render_state, frames_iter = self.pipeline.process_chunk(
                        av, chunk_obj, state=render_state, options=_render_options,
                    )
                    frames_list = list(frames_iter)
                render_ms = (time.monotonic() - t_render_start) * 1000
                chunk_timings["render_ms"] = round(render_ms, 2)

                raw_chunk = b"".join(f.data.tobytes() for f in frames_list)

                t_write_start = time.monotonic()
                try:
                    ffmpeg_proc.stdin.write(raw_chunk)
                    ffmpeg_proc.stdin.flush()
                    total_raw_written[0] += len(raw_chunk)
                except BrokenPipeError:
                    print(f"[video-producer] WARNING: BrokenPipeError chunk {chunk_i}")
                chunk_timings["ffmpeg_write_ms"] = round((time.monotonic() - t_write_start) * 1000, 2)
                chunk_timings["chunk_wall_ms"] = round((time.monotonic() - t_chunk_start) * 1000, 2)

                label = f"speech chunk {chunk_i}" if not audio_underflow else f"idle chunk {chunk_i}"
                print(f"[video-producer] {label}: audio_wait={audio_wait_ms}ms "
                      f"render={chunk_timings['render_ms']}ms frames={len(frames_list)} "
                      f"wall={chunk_timings['chunk_wall_ms']}ms")

                max_buffer_samples = _window + STEP_SAMPLES * 2
                if len(audio_buffer) > max_buffer_samples:
                    audio_buffer = audio_buffer[-_window:]

                producer_timings.append(chunk_timings)

            try:
                ffmpeg_proc.stdin.close()
                print("[video-producer] FFmpeg stdin closed")
            except Exception:
                pass

            _producer_end[0] = time.monotonic()
            producer_done_event.set()
            total_s = _producer_end[0] - _producer_start[0]
            print(f"[video-producer] DONE: {num_chunks} chunks in {total_s:.2f}s")

        audio_thread = threading.Thread(target=_audio_producer, daemon=True)
        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        producer_thread = threading.Thread(target=_video_producer, daemon=True)

        audio_thread.start()
        stderr_thread.start()
        stdout_thread.start()
        producer_thread.start()

        stream_start_mono = time.monotonic()
        sender = _SegmentSender(
            chunk_media_s=CHUNK_MEDIA_S,
            target_lead_s=TARGET_LEAD_S,
            stream_start_mono=stream_start_mono,
        )

        sentinel_seen = False
        total_encoded_bytes = 0
        consumer_start_mono = time.monotonic()
        max_consumer_s = 120.0
        first_media_wall_ms = 0.0

        print("[audio-handoff] consumer loop starting")

        while True:
            if time.monotonic() - consumer_start_mono > max_consumer_s:
                print("[audio-handoff] CONSUMER TIMEOUT")
                break

            got_any = False
            while True:
                try:
                    item = segment_queue.get_nowait()
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                    if item[0] == "media" and first_media_wall_ms == 0.0:
                        first_media_wall_ms = round((time.monotonic() - stream_start_mono) * 1000, 1)
                    got_any = True
                except queue.Empty:
                    break

            sender.yield_events()

            if sentinel_seen:
                break

            if not got_any and not stdout_thread.is_alive():
                try:
                    item = segment_queue.get(timeout=1.0)
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                    if item[0] == "media" and first_media_wall_ms == 0.0:
                        first_media_wall_ms = round((time.monotonic() - stream_start_mono) * 1000, 1)
                except queue.Empty:
                    print("[audio-handoff] stdout thread dead, queue empty, exiting consumer")
                    break

            if not got_any:
                time.sleep(0.01)

        drained_final = sender.drain_all()
        consumer_end_mono = time.monotonic()

        print(f"[audio-handoff] consumer loop done: sentinel={sentinel_seen} "
              f"sender_media={len(sender.collected_media)} "
              f"drained_final={len(drained_final)} "
              f"consumer_s={consumer_end_mono - consumer_start_mono:.2f}")

        if audio_thread is not None:
            audio_thread.join(timeout=30)
        producer_thread.join(timeout=30)
        stderr_thread.join(timeout=3)
        stdout_thread.join(timeout=3)

        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait(timeout=2)

        try:
            zombie_check = subprocess.run(
                ["ps", "-p", str(ffmpeg_proc.pid)],
                capture_output=True, timeout=2,
            )
            zombie_status = "zombie_still_running" if zombie_check.returncode == 0 else "clean"
        except Exception:
            zombie_status = "unknown"

        ffmpeg_rc = ffmpeg_proc.returncode
        n_stderr_lines = len(ffmpeg_stderr_lines)
        ffmpeg_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
        ffmpeg_warnings = [l for l in ffmpeg_stderr_lines if "warning" in l.lower()]

        print(f"[audio-handoff] FFmpeg exit: rc={ffmpeg_rc} stderr_lines={n_stderr_lines} "
              f"errors={len(ffmpeg_errors)} warnings={len(ffmpeg_warnings)} zombie={zombie_status}")

        p_start = _producer_start[0]
        p_end = _producer_end[0]
        ap_start = audio_producer_start[0]
        ap_end = audio_producer_end[0]
        c_start = consumer_start_mono
        c_end = consumer_end_mono

        video_producer_consumer_overlap_s = max(0.0, min(p_end, c_end) - max(p_start, c_start))
        audio_producer_consumer_overlap_s = max(0.0, min(ap_end, c_end) - max(ap_start, c_start))
        all_three_overlap_s = max(0.0, min(p_end, ap_end, c_end) - max(p_start, ap_start, c_start))

        init_info = sender.collected_init
        media_events = sender.collected_media
        first_init_latency_ms = init_info.get("timing", {}).get("timeToInitMs", 0) if init_info else 0

        pacing_inter_yield_gaps = sender._inter_yield_gaps

        total_wall_s = time.monotonic() - t_total_start
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        handoff_stats = handoff.stats()

        audio_producer_result = {
            "total_s": round(ap_end - ap_start, 4) if ap_start > 0 else 0,
            "push_intervals_ms": audio_push_intervals,
            "mode": audio_mode_resolved,
        }

        if cadence and len(cadence_push_times) >= 2:
            expected_total_s = round((num_chunks - 1) * chunk_ms / 1000, 4)
            cadence_intervals = [round(cadence_push_times[i] - cadence_push_times[i - 1], 2)
                                 for i in range(1, len(cadence_push_times))]
            sorted_intervals = sorted(cadence_intervals)
            n_iv = len(sorted_intervals)
            p50_idx = n_iv // 2
            p95_idx = int(n_iv * 0.95)
            audio_producer_result["cadence"] = {
                "expected_total_s": expected_total_s,
                "actual_push_times_ms_from_start": cadence_push_times,
                "actual_push_intervals_ms": cadence_intervals,
                "interval_stats": {
                    "min": round(min(cadence_intervals), 2),
                    "max": round(max(cadence_intervals), 2),
                    "avg": round(sum(cadence_intervals) / len(cadence_intervals), 2),
                    "p50": round(sorted_intervals[min(p50_idx, n_iv - 1)], 2),
                    "p95": round(sorted_intervals[min(p95_idx, n_iv - 1)], 2),
                },
                "producer_drift_ms": round(cadence_push_times[-1] - (num_chunks - 1) * chunk_ms, 2),
                "chunks_pushed": len(cadence_push_times),
            }

        result = {
            "avatar_id": avatar_id,
            "num_chunks": num_chunks,
            "chunk_ms_target": chunk_ms,
            "audio_mode": audio_mode_resolved,
            "cadence": cadence,
            "pacing_mode": "concurrent_with_audio_handoff",
            "avatar_load_s": round(avatar_load_s, 4),
            "total_wall_s": round(total_wall_s, 4),
            "restore_timings": getattr(self, "_restore_timings", {}),
            "video_config": {
                "width": W,
                "height": H,
                "fps": FPS,
                "frames_per_chunk": N_FRAMES_PER_CHUNK,
                "frame_bytes_i420": FRAME_BYTES,
                "chunk_media_s": round(CHUNK_MEDIA_S, 4),
            },
            "audio_config": {
                "sample_rate": AUDIO_SR,
                "step_samples": STEP_SAMPLES,
                "audio_window_len": _window,
                "hubert_past_frames": self.HUBERT_PAST_FRAMES,
                "audio_mode": audio_mode_resolved,
            },
            "audio_handoff": handoff_stats,
            "audio_producer": audio_producer_result,
            "init_segment": {
                "generated": init_info is not None,
                "byte_count": init_info.get("byteLength", 0) if init_info else 0,
                "latency_ms": first_init_latency_ms,
            },
            "media_segments": {
                "count": len(media_events),
                "total_bytes": sum(e["byteLength"] for e in media_events),
                "sizes": [e["byteLength"] for e in media_events],
                "first_latency_ms": first_media_wall_ms,
            },
            "pacing": {
                "target_lead_s": TARGET_LEAD_S,
                "chunk_media_s": round(CHUNK_MEDIA_S, 4),
                "total_pacing_sleep_ms": sender.total_pacing_sleep_ms,
                "inter_yield_gap_ms": {
                    "min": min(pacing_inter_yield_gaps) if pacing_inter_yield_gaps else 0,
                    "max": max(pacing_inter_yield_gaps) if pacing_inter_yield_gaps else 0,
                    "avg": round(sum(pacing_inter_yield_gaps) / len(pacing_inter_yield_gaps), 1) if pacing_inter_yield_gaps else 0,
                },
                "queue_depth": {
                    "min": sender.min_queue_depth,
                    "max": sender.max_queue_depth,
                },
                "dry_spell_count": sender.dry_spell_count,
                "target_lead_reached": sender.target_lead_reached,
            },
            "concurrency": {
                "audio_producer_total_s": round(ap_end - ap_start, 4) if ap_start > 0 else 0,
                "video_producer_total_s": round(p_end - p_start, 4),
                "consumer_total_s": round(c_end - c_start, 4),
                "video_producer_consumer_overlap_s": round(video_producer_consumer_overlap_s, 4),
                "audio_producer_consumer_overlap_s": round(audio_producer_consumer_overlap_s, 4),
                "all_three_overlap_s": round(all_three_overlap_s, 4),
                "producer_done_before_consumer_done": p_end <= c_end,
                "parser_sentinel_seen": sentinel_seen,
            },
            "ffmpeg": {
                "return_code": ffmpeg_rc,
                "stderr_line_count": n_stderr_lines,
                "fatal_error_detected": len(ffmpeg_errors) > 0 and ffmpeg_rc != 0,
                "stderr_error_count": len(ffmpeg_errors),
                "stderr_warning_count": len(ffmpeg_warnings),
                "zombie_check": zombie_status,
                "pid": ffmpeg_proc.pid,
            },
            "bytes": {
                "raw_written": total_raw_written[0],
                "encoded_total": total_encoded_bytes,
                "compression_ratio": round(total_raw_written[0] / total_encoded_bytes, 2) if total_encoded_bytes > 0 else 0,
            },
            "vram_mb": vram_mb,
            "chunks": producer_timings,
            "reader_errors": reader_error_holder,
        }

        print(f"[audio-handoff] === audio_handoff_stream_smoke done: "
              f"{num_chunks} chunks in {total_wall_s:.2f}s ===")
        print(f"[audio-handoff] init={init_info is not None} media_count={len(media_events)} "
              f"rawMB={total_raw_written[0]/1e6:.1f} encodedMB={total_encoded_bytes/1e6:.1f} "
              f"vram={vram_mb}MB")

        return result

    @modal.method()
    def audio_handoff_stream_cadence_smoke(self, avatar_id: str, num_chunks: int = 20,
                                            chunk_ms: int = 200,
                                            audio_mode: str = "sine") -> dict:
        return self._audio_handoff_stream_impl(avatar_id, num_chunks, chunk_ms, audio_mode, cadence=True)

    def _omnivoice_tts_smoke_impl(self, text: str) -> dict:
        import numpy as np
        import torch

        t_method_start = time.monotonic()
        print(f"[tts-smoke] === omnivoice_tts_smoke: text_len={len(text)} ===")

        if not getattr(self, "_omnivoice_ready", False) or getattr(self, "_omnivoice_model", None) is None:
            return {
                "ok": False,
                "error": "OmniVoice not ready",
                "omnivoice_ready": False,
            }

        vram_before = torch.cuda.memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

        t_gen_start = time.monotonic()
        try:
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
        except Exception as exc:
            return {
                "ok": False,
                "error": f"OmniVoice generate failed: {exc}",
                "generation_s": round(time.monotonic() - t_gen_start, 4),
            }
        t_gen_end = time.monotonic()
        generation_s = t_gen_end - t_gen_start

        audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
        if isinstance(audio_data, np.ndarray):
            audio_data = torch.from_numpy(audio_data).float()
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.detach().float()
        if audio_data.dim() > 1:
            audio_data = audio_data.squeeze(0)

        native_sr = self._omnivoice_sr
        audio_duration_s = audio_data.shape[0] / native_sr

        t_resample_start = time.monotonic()
        if native_sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(native_sr, 16000, dtype=audio_data.dtype)
            audio_data = resampler(audio_data)
        resample_s = time.monotonic() - t_resample_start

        audio_np = audio_data.cpu().numpy()
        has_nan = bool(np.any(np.isnan(audio_np)))
        has_inf = bool(np.any(np.isinf(audio_np)))
        if has_nan or has_inf:
            audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)

        rtf = audio_duration_s / generation_s if generation_s > 0 else 0
        vram_after = torch.cuda.memory_allocated() / 1024**2
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        total_s = time.monotonic() - t_method_start

        print(f"[tts-smoke] generate={generation_s:.3f}s audio={audio_duration_s:.2f}s "
              f"RTF={rtf:.2f}x samples={audio_data.shape[0]} vram_delta={vram_after - vram_before:.1f}MB")

        return {
            "ok": True,
            "generation_s": round(generation_s, 4),
            "resample_s": round(resample_s, 4),
            "total_s": round(total_s, 4),
            "audio_duration_s": round(audio_duration_s, 4),
            "realtime_factor": round(rtf, 2),
            "native_sample_rate": native_sr,
            "resampled_rate": 16000,
            "sample_count": int(audio_data.shape[0]),
            "resampled_sample_count": int(audio_np.shape[0]),
            "dtype": str(audio_data.dtype),
            "value_range": [float(audio_np.min()), float(audio_np.max())],
            "has_nan": has_nan,
            "has_inf": has_inf,
            "vram_before_mb": round(vram_before, 1),
            "vram_after_mb": round(vram_after, 1),
            "vram_delta_mb": round(vram_after - vram_before, 1),
            "peak_vram_gb": round(peak_vram_gb, 3),
            "text_length": len(text),
            "omnivoice_ready": True,
        }

    @modal.method()
    def omnivoice_tts_smoke(self, text: str = (
        "The quick brown fox jumps over the lazy dog. "
        "In a world where technology advances rapidly, digital avatars "
        "are becoming increasingly realistic and interactive."
    )) -> dict:
        return self._omnivoice_tts_smoke_impl(text)

    def _parallel_tts_render_impl(
        self,
        avatar_id: str,
        text: str,
        chunk_ms: int = 200,
        max_render_chunks: int = 20,
    ) -> dict:
        import numpy as np
        import os
        import torch

        t_method_start = time.monotonic()
        print(f"[parallel-stress] === _parallel_tts_render_impl: "
              f"avatar={avatar_id} text_len={len(text)} chunks={max_render_chunks} ===")

        STEP_SAMPLES = self.chunk_size * self.frame_len
        AUDIO_SR = 16000
        FPS = 25
        H, W = 720, 1280
        N_FRAMES = self.chunk_size
        FRAME_BYTES = H * W * 3 // 2
        CHUNK_MEDIA_S = N_FRAMES / FPS
        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))

        av = self._get_avatar(avatar_id)

        from avtr1_renderer.types import Chunk, RenderOptions
        _mg = self.pipeline._motion_generator
        _window = (_mg.chunk_size + _mg.future_size) * _mg.frame_len + _mg.audio_shift
        _silence_audio = np.zeros(_window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        handoff = _AudioHandoff(maxsize=300)

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:0",
            "-c:v", "h264_nvenc", "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", str(chunk_ms * 1000),
            "pipe:1",
        ]

        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[parallel-stress] FFmpeg PID={ffmpeg_proc.pid} spawned")

        segment_queue = queue.Queue(maxsize=100)
        stderr_lines_holder = [None]
        reader_error_holder = [None]
        sentinel_holder = [False]
        producer_done_event = threading.Event()
        init_info_holder = [None]

        tts_start_mono = [0.0]
        tts_end_mono = [0.0]
        tts_result_holder = [None]
        tts_generate_s = [0.0]
        tts_audio_chunk_count = [0]
        tts_audio_total_s = [0.0]
        first_audio_available_s = [None]

        total_raw_written = [0]
        total_encoded_bytes = [0]

        def _ffmpeg_stderr_reader():
            lines = []
            try:
                while True:
                    line = ffmpeg_proc.stderr.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace").rstrip()
                    lines.append(decoded)
            except Exception:
                pass
            stderr_lines_holder[0] = lines

        def _ffmpeg_stdout_reader():
            buf = b""
            try:
                while True:
                    data = ffmpeg_proc.stdout.read(4096)
                    if not data:
                        break
                    buf += data
                    while True:
                        if len(buf) < 8:
                            break
                        box_size = struct.unpack(">I", buf[:4])[0]
                        box_type = buf[4:8]
                        if box_size < 8:
                            buf = buf[1:]
                            continue
                        if len(buf) < box_size:
                            break
                        box_data = buf[:box_size]
                        buf = buf[box_size:]
                        if box_type == b"ftyp":
                            init_info_holder[0] = box_data
                            segment_queue.put(("init", box_data))
                        elif box_type == b"moov":
                            if init_info_holder[0] is not None:
                                init_data = init_info_holder[0] + box_data
                                segment_queue.put(("init", init_data))
                        elif box_type == b"moof":
                            _pending_moof = box_data
                            _pending_mdat = None
                            while True:
                                if len(buf) < 8:
                                    more = ffmpeg_proc.stdout.read(4096)
                                    if not more:
                                        break
                                    buf += more
                                next_size = struct.unpack(">I", buf[:4])[0]
                                next_type = buf[4:8]
                                if next_type == b"mdat":
                                    if len(buf) < next_size:
                                        needed = next_size - len(buf)
                                        while needed > 0:
                                            more = ffmpeg_proc.stdout.read(min(needed, 65536))
                                            if not more:
                                                break
                                            buf += more
                                            needed -= len(more)
                                    if len(buf) >= next_size:
                                        _pending_mdat = buf[:next_size]
                                        buf = buf[next_size:]
                                    break
                                else:
                                    break
                            if _pending_mdat is not None:
                                media_data = _pending_moof + _pending_mdat
                                segment_queue.put(("media", media_data))
                        elif box_type == b"mdat":
                            pass
            except Exception as exc:
                reader_error_holder[0] = str(exc)
            finally:
                segment_queue.put(None)
                sentinel_holder[0] = True

        def _tts_producer():
            import torchaudio.transforms as T

            tts_start_mono[0] = time.monotonic()
            print(f"[tts-producer] Starting OmniVoice generation...")

            try:
                torch.cuda.reset_peak_memory_stats()
                t_gen_start = time.monotonic()
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
                t_gen_end = time.monotonic()
                tts_generate_s[0] = t_gen_end - t_gen_start
                print(f"[tts-producer] generate() done in {tts_generate_s[0]:.3f}s")

                audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
                if isinstance(audio_data, np.ndarray):
                    audio_data = torch.from_numpy(audio_data).float()
                if isinstance(audio_data, torch.Tensor):
                    audio_data = audio_data.detach().float()
                if audio_data.dim() > 1:
                    audio_data = audio_data.squeeze(0)

                native_dur_s = audio_data.shape[0] / self._omnivoice_sr

                t_resample = time.monotonic()
                if self._omnivoice_sr != AUDIO_SR:
                    resampler = T.Resample(self._omnivoice_sr, AUDIO_SR, dtype=audio_data.dtype)
                    audio_data = resampler(audio_data)
                resample_s = time.monotonic() - t_resample
                print(f"[tts-producer] resampled {self._omnivoice_sr}Hz -> {AUDIO_SR}Hz in {resample_s:.4f}s")

                audio_np = audio_data.cpu().numpy()
                if not np.all(np.isfinite(audio_np)):
                    audio_np = np.nan_to_num(audio_np, nan=0.0, posinf=1.0, neginf=-1.0)

                n_chunks = max(1, int(np.ceil(len(audio_np) / STEP_SAMPLES)))
                first_audio_available_s[0] = time.monotonic() - t_method_start
                print(f"[tts-producer] chunking {len(audio_np)} samples into {n_chunks} chunks "
                      f"(step={STEP_SAMPLES}, dur={native_dur_s:.2f}s)")

                for ci in range(n_chunks):
                    s = ci * STEP_SAMPLES
                    e = min(s + STEP_SAMPLES, len(audio_np))
                    chunk_pcm = audio_np[s:e].copy()
                    if len(chunk_pcm) < STEP_SAMPLES:
                        chunk_pcm = np.pad(chunk_pcm, (0, STEP_SAMPLES - len(chunk_pcm)))
                    meta = {
                        "chunkIndex": ci,
                        "isFirst": ci == 0,
                        "isLast": ci == n_chunks - 1,
                        "samples": len(chunk_pcm),
                    }
                    handoff.push(chunk_pcm, metadata=meta, timeout=30.0)

                tts_audio_chunk_count[0] = n_chunks
                tts_audio_total_s[0] = len(audio_np) / AUDIO_SR
                tts_result_holder[0] = {
                    "ok": True,
                    "generate_s": round(tts_generate_s[0], 4),
                    "native_duration_s": round(native_dur_s, 4),
                    "resampled_duration_s": round(len(audio_np) / AUDIO_SR, 4),
                    "n_chunks": n_chunks,
                    "resample_s": round(resample_s, 4),
                    "first_audio_available_s": round(first_audio_available_s[0], 4),
                }

            except Exception as exc:
                import traceback
                traceback.print_exc()
                tts_result_holder[0] = {"ok": False, "error": str(exc)}
            finally:
                handoff.close()
                tts_end_mono[0] = time.monotonic()
                print(f"[tts-producer] DONE in {tts_end_mono[0] - tts_start_mono[0]:.3f}s")

        def _video_producer():
            t_vp_start = time.monotonic()
            chunks_rendered = 0
            underflow_count = 0
            queue_depths = []
            producer_timings = []
            render_state = None
            audio_buffer = np.zeros(0, dtype=np.float32)

            for ci in range(max_render_chunks):
                t_chunk_start = time.monotonic()
                t_pop_start = time.monotonic()
                item = handoff.pop(timeout_s=5.0)
                audio_wait_ms = (time.monotonic() - t_pop_start) * 1000

                if item is None:
                    underflow_count += 1
                    chunk_audio = np.zeros(STEP_SAMPLES, dtype=np.float32)
                else:
                    chunk_audio, _meta = item

                qd = handoff.queue.qsize()
                queue_depths.append(qd)

                audio_buffer = np.concatenate([audio_buffer, chunk_audio])
                if len(audio_buffer) >= _window:
                    audio_window = audio_buffer[-_window:].copy()
                else:
                    audio_window = np.zeros(_window, dtype=np.float32)
                    audio_window[-len(audio_buffer):] = audio_buffer

                t_render_start = time.monotonic()
                chunk_obj = Chunk(audio_speech=audio_window, audio_listen=_silence_audio)
                with torch.inference_mode():
                    render_state, frames_iter = self.pipeline.process_chunk(
                        av, chunk_obj, state=render_state, options=_render_options,
                    )
                    frames_list = list(frames_iter)
                raw_chunk = b"".join(f.data.tobytes() for f in frames_list)
                render_ms = (time.monotonic() - t_render_start) * 1000

                t_write_start = time.monotonic()
                try:
                    ffmpeg_proc.stdin.write(raw_chunk)
                    ffmpeg_proc.stdin.flush()
                    total_raw_written[0] += len(raw_chunk)
                except BrokenPipeError:
                    print(f"[video-producer] FFmpeg stdin broken pipe at chunk {ci}")
                    break
                write_s = time.monotonic() - t_write_start

                t_chunk_end = time.monotonic()
                wall_s = t_chunk_end - t_chunk_start

                chunk_timing = {
                    "ci": ci,
                    "audio_wait_ms": round(audio_wait_ms, 2),
                    "underflow": item is None,
                    "render_ms": round(render_ms, 2),
                    "ffmpeg_write_ms": round(write_s * 1000, 2),
                    "wall_ms": round(wall_s * 1000, 2),
                    "handoff_queue_depth": qd,
                }
                producer_timings.append(chunk_timing)
                chunks_rendered += 1

                max_audio_buf = _window + STEP_SAMPLES * 2
                if len(audio_buffer) > max_audio_buf:
                    audio_buffer = audio_buffer[-_window:]

            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            producer_done_event.set()

            t_vp_end = time.monotonic()
            vp_total = t_vp_end - t_vp_start
            print(f"[video-producer] DONE: {chunks_rendered} chunks in {vp_total:.2f}s "
                  f"underflows={underflow_count}")

            return {
                "chunks_rendered": chunks_rendered,
                "total_s": round(vp_total, 4),
                "underflow_count": underflow_count,
                "queue_depths": queue_depths,
                "timings": producer_timings,
                "audio_path_active": chunks_rendered > 0,
            }

        render_start_mono = [0.0]
        render_end_mono = [0.0]
        video_producer_result = [None]

        def _run_video_producer():
            render_start_mono[0] = time.monotonic()
            video_producer_result[0] = _video_producer()
            render_end_mono[0] = time.monotonic()

        t_stderr = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        t_stdout = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        t_tts = threading.Thread(target=_tts_producer, daemon=True)
        t_video = threading.Thread(target=_run_video_producer, daemon=True)

        t_stderr.start()
        t_stdout.start()
        t_tts.start()
        time.sleep(0.05)
        t_video.start()

        stream_start_mono = time.monotonic()
        sender = _SegmentSender(
            chunk_media_s=CHUNK_MEDIA_S,
            target_lead_s=TARGET_LEAD_S,
            stream_start_mono=stream_start_mono,
        )

        media_events = []
        consumer_timeout_s = 180
        consumer_start = time.monotonic()
        consumer_done = False

        while True:
            if time.monotonic() - consumer_start > consumer_timeout_s:
                print("[parallel-stress] consumer TIMEOUT")
                break

            drained_any = False
            while True:
                try:
                    seg = segment_queue.get_nowait()
                    if seg is None:
                        sentinel_holder[0] = True
                        consumer_done = True
                        break
                    seg_type, seg_data = seg
                    sender.add(seg_type, seg_data)
                    drained_any = True
                except queue.Empty:
                    break

            if consumer_done:
                break

            if drained_any:
                events = sender.yield_events()
                for evt in events:
                    if evt["type"] == "fmp4_media":
                        media_events.append(evt)

            if producer_done_event.is_set() and sentinel_holder[0]:
                break

            if not drained_any:
                try:
                    seg = segment_queue.get(timeout=1.0)
                    if seg is None:
                        sentinel_holder[0] = True
                        break
                    seg_type, seg_data = seg
                    sender.add(seg_type, seg_data)
                except queue.Empty:
                    if producer_done_event.is_set() and not t_stdout.is_alive():
                        break
                    continue

        for evt in sender.drain_all():
            if evt["type"] == "fmp4_media":
                media_events.append(evt)

        t_tts.join(timeout=30)
        t_video.join(timeout=30)
        t_stdout.join(timeout=10)
        t_stderr.join(timeout=5)

        # Phase 4A: canonical Popen cleanup — terminate() before wait() to match all other paths.
        try:
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ffmpeg_proc.kill()
            ffmpeg_proc.wait(timeout=3)

        ffmpeg_rc = ffmpeg_proc.returncode
        zombie = ffmpeg_proc.poll() is None

        stderr_lines = stderr_lines_holder[0] or []
        stderr_errors = [l for l in stderr_lines if "error" in l.lower()]
        stderr_warnings = [l for l in stderr_lines if "warning" in l.lower()]

        vp = video_producer_result[0] or {}
        handoff_stats = handoff.stats()

        total_encoded_bytes = sum(evt.get("byteLength", 0) for evt in media_events)
        total_raw = total_raw_written[0]

        overlap_s = 0.0
        if tts_start_mono[0] > 0 and tts_end_mono[0] > 0 and render_start_mono[0] > 0 and render_end_mono[0] > 0:
            overlap_start = max(tts_start_mono[0], render_start_mono[0])
            overlap_end = min(tts_end_mono[0], render_end_mono[0])
            overlap_s = max(0, overlap_end - overlap_start)

        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)
        total_wall_s = time.monotonic() - t_method_start

        print(f"[parallel-stress] === parallel done in {total_wall_s:.2f}s ===")
        print(f"[parallel-stress] tts_gen={tts_generate_s[0]:.3f}s overlap={overlap_s:.3f}s "
              f"media={len(media_events)} underflows={vp.get('underflow_count', 0)}")

        return {
            "ok": True,
            "mode": "full_utterance_then_chunk",
            "total_wall_s": round(total_wall_s, 4),
            "tts": tts_result_holder[0],
            "tts_timing": {
                "start_mono_offset_s": round(tts_start_mono[0] - t_method_start, 4) if tts_start_mono[0] > 0 else 0,
                "end_mono_offset_s": round(tts_end_mono[0] - t_method_start, 4) if tts_end_mono[0] > 0 else 0,
                "generate_s": round(tts_generate_s[0], 4),
                "audio_chunks": tts_audio_chunk_count[0],
                "audio_total_s": round(tts_audio_total_s[0], 4),
                "first_audio_available_s": first_audio_available_s[0],
            },
            "render": {
                "start_mono_offset_s": round(render_start_mono[0] - t_method_start, 4) if render_start_mono[0] > 0 else 0,
                "end_mono_offset_s": round(render_end_mono[0] - t_method_start, 4) if render_end_mono[0] > 0 else 0,
                "chunks_rendered": vp.get("chunks_rendered", 0),
                "total_s": vp.get("total_s", 0),
                "underflow_count": vp.get("underflow_count", 0),
                "audio_path_active": vp.get("audio_path_active", False),
            },
            "overlap_s": round(overlap_s, 4),
            "handoff": handoff_stats,
            "fmp4": {
                "init_segment": sender.collected_init.get("byteLength", 0) if sender.collected_init else 0,
                "media_segments": len(media_events),
                "total_bytes": total_encoded_bytes,
                "total_raw_bytes": total_raw,
            },
            "ffmpeg": {
                "rc": ffmpeg_rc,
                "zombie": "clean" if not zombie else "LEAK",
                "stderr_lines": len(stderr_lines),
                "stderr_errors": len(stderr_errors),
                "stderr_warnings": len(stderr_warnings),
            },
            "vram_mb": vram_mb,
            "chunks": vp.get("timings", []),
            "queue_depths": vp.get("queue_depths", []),
        }

    @modal.method()
    def omnivoice_parallel_stress_smoke(
        self,
        avatar_id: str = "arnold",
        text: str = "Tell me a short cheerful story about a tiny robot learning to paint.",
        chunk_ms: int = 200,
        max_render_chunks: int = 20,
    ) -> dict:
        import torch

        t_total_start = time.monotonic()
        print(f"[stress-smoke] === omnivoice_parallel_stress_smoke: "
              f"avatar={avatar_id} text_len={len(text)} chunks={max_render_chunks} ===")

        if not getattr(self, "_omnivoice_ready", False) or getattr(self, "_omnivoice_model", None) is None:
            return {
                "ok": False,
                "error": "OmniVoice not ready",
                "omnivoice_ready": False,
            }

        print("[stress-smoke] === Case A: render_only_baseline (synthetic sine) ===")
        render_baseline = self._audio_handoff_stream_impl(
            avatar_id, max_render_chunks, chunk_ms, "sine", cadence=True
        )
        render_baseline_total_s = render_baseline.get("total_wall_s", 0)
        render_baseline_chunks = render_baseline.get("num_chunks", 0)
        render_baseline_avg_wall = 0
        render_timings = render_baseline.get("chunks", [])
        if render_timings:
            render_baseline_avg_wall = sum(c.get("wall_ms", 0) for c in render_timings) / len(render_timings)

        torch.cuda.empty_cache()
        time.sleep(1.0)

        print("[stress-smoke] === Case B: tts_only_baseline ===")
        tts_baseline = self._omnivoice_tts_smoke_impl(text)
        tts_baseline_generate_s = tts_baseline.get("generation_s", 0)
        tts_baseline_duration_s = tts_baseline.get("audio_duration_s", 0)
        tts_baseline_rtf = tts_baseline.get("realtime_factor", 0)

        torch.cuda.empty_cache()
        time.sleep(1.0)

        print("[stress-smoke] === Case C: parallel_tts_render ===")
        parallel = self._parallel_tts_render_impl(avatar_id, text, chunk_ms, max_render_chunks)

        parallel_tts_res = parallel.get("tts", {})
        parallel_render_res = parallel.get("render", {})
        parallel_fmp4_res = parallel.get("fmp4", {})
        parallel_ffmpeg_res = parallel.get("ffmpeg", {})

        parallel_tts_generate_s = parallel.get("tts_timing", {}).get("generate_s", 0)
        parallel_render_chunks = parallel_render_res.get("chunks_rendered", 0)
        parallel_chunks = parallel.get("chunks", [])
        parallel_avg_wall = 0
        if parallel_chunks:
            parallel_avg_wall = sum(c.get("wall_ms", 0) for c in parallel_chunks) / len(parallel_chunks)

        parallel_ok = True
        safety_reasons = []

        if not parallel_tts_res or not parallel_tts_res.get("ok"):
            parallel_ok = False
            safety_reasons.append("tts_failed")
        if parallel_render_chunks == 0:
            parallel_ok = False
            safety_reasons.append("zero_chunks_rendered")
        if parallel_fmp4_res.get("media_segments", 0) == 0:
            parallel_ok = False
            safety_reasons.append("zero_fmp4_segments")
        if parallel_ffmpeg_res.get("rc", -1) != 0:
            parallel_ok = False
            safety_reasons.append(f"ffmpeg_rc={parallel_ffmpeg_res.get('rc', 'unknown')}")
        if not parallel_render_res.get("audio_path_active", False):
            parallel_ok = False
            safety_reasons.append("audio_path_inactive")

        tts_slowdown_pct = 0
        if tts_baseline_generate_s > 0 and parallel_tts_generate_s > 0:
            tts_slowdown_pct = round((parallel_tts_generate_s / tts_baseline_generate_s - 1) * 100, 2)

        render_slowdown_pct = 0
        if render_baseline_avg_wall > 0 and parallel_avg_wall > 0:
            render_slowdown_pct = round((parallel_avg_wall / render_baseline_avg_wall - 1) * 100, 2)

        overlap_s = parallel.get("overlap_s", 0)
        if tts_baseline_generate_s > 0 and overlap_s > 0:
            overlap_pct = round(overlap_s / tts_baseline_generate_s * 100, 1)
        else:
            overlap_pct = 0

        if not parallel_ok:
            verdict = "same_gpu_hold"
        elif tts_slowdown_pct > 100 or render_slowdown_pct > 100:
            verdict = "same_gpu_blocked"
        elif tts_slowdown_pct > 30 or render_slowdown_pct > 30:
            verdict = "same_gpu_degraded"
        else:
            verdict = "same_gpu_viable"

        total_s = time.monotonic() - t_total_start
        print(f"[stress-smoke] === ALL CASES DONE in {total_s:.2f}s ===")
        print(f"[stress-smoke] verdict={verdict} parallel_ok={parallel_ok} "
              f"tts_slowdown={tts_slowdown_pct}% render_slowdown={render_slowdown_pct}% "
              f"overlap={overlap_s:.3f}s")
        if safety_reasons:
            print(f"[stress-smoke] SAFETY REASONS: {safety_reasons}")

        return {
            "ok": parallel_ok,
            "safety_reasons": safety_reasons,
            "total_s": round(total_s, 4),
            "verdict": verdict,
            "case_a_render_only": {
                "total_s": render_baseline_total_s,
                "chunks": render_baseline_chunks,
                "avg_chunk_wall_ms": round(render_baseline_avg_wall, 2),
            },
            "case_b_tts_only": {
                "generate_s": tts_baseline_generate_s,
                "audio_duration_s": tts_baseline_duration_s,
                "rtf": tts_baseline_rtf,
            },
            "case_c_parallel": {
                "total_wall_s": parallel.get("total_wall_s", 0),
                "tts_generate_s": parallel_tts_generate_s,
                "render_chunks": parallel_render_chunks,
                "render_avg_wall_ms": round(parallel_avg_wall, 2),
                "overlap_s": overlap_s,
                "overlap_pct_of_tts_gen": overlap_pct,
                "underflow_count": parallel_render_res.get("underflow_count", 0),
                "audio_path_active": parallel_render_res.get("audio_path_active", False),
                "fmp4_segments": parallel_fmp4_res.get("media_segments", 0),
                "ffmpeg_rc": parallel_ffmpeg_res.get("rc", -1),
                "mode": parallel.get("mode", "unknown"),
            },
            "comparison": {
                "tts_slowdown_pct": tts_slowdown_pct,
                "render_slowdown_pct": render_slowdown_pct,
                "tts_baseline_s": tts_baseline_generate_s,
                "tts_parallel_s": parallel_tts_generate_s,
                "render_baseline_avg_ms": round(render_baseline_avg_wall, 2),
                "render_parallel_avg_ms": round(parallel_avg_wall, 2),
            },
            "parallel_detail": parallel,
            "render_baseline_detail": render_baseline,
            "tts_baseline_detail": tts_baseline,
        }

    def _run_session_render_impl(
        self,
        session_id: str,
        num_chunks: int = 10,
        chunk_ms: int = 200,
        audio_mode: str = "sine",
        audio_source: str = "internal",
    ) -> dict:
        """Phase 4C/4D: session render loop.

        audio_source:
          "internal" — start the internal sine/silence _audio_producer
                       thread (used by run_session_smoke). num_chunks
                       controls the loop bound.
          "handoff"  — do NOT start an internal audio producer. The
                       _video_producer pulls audio from the session's
                       _AudioHandoff queue (populated by
                       http_push_audio from the Bridge). The loop runs
                       until sess["render_stop_event"] is set, the
                       handoff is closed and empty for several
                       iterations, or the consumer times out.

        Status transitions:
          ready -> running (top of function)
          running -> completed (smoke, all chunks rendered)
          running -> ended    (streaming, stop_event set)
          running -> error    (exception in any thread)
        """
        import os
        import numpy as np
        import torch

        t_method_start = time.monotonic()
        audio_source_resolved = audio_source if audio_source in ("internal", "handoff") else "internal"
        print(f"[session-render] === _run_session_render_impl: "
              f"session={session_id} chunks={num_chunks} chunk_ms={chunk_ms} "
              f"audio={audio_mode} audio_source={audio_source_resolved} ===")

        sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": f"session {session_id} not found"}
        if sess.get("status") not in ("ready", "running"):
            return {"ok": False, "error": f"session status is {sess.get('status')}, expected ready/running"}

        sess["status"] = "running"

        av = sess["avatar"]
        render_state = sess.get("render_state")
        handoff = sess["handoff"]
        # Phase 4D: pull the stop event (created in _start_session_impl).
        # For audio_source="handoff" this signals the producer to exit.
        render_stop_event = sess.get("render_stop_event") or threading.Event()

        FPS = 25
        H = 720
        W = 1280
        N_FRAMES_PER_CHUNK = self.chunk_size
        FRAME_BYTES = H * W * 3 // 2
        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "4.0"))
        CHUNK_MEDIA_S = N_FRAMES_PER_CHUNK / FPS

        STEP_SAMPLES = self.chunk_size * self.frame_len
        AUDIO_SR = 16000

        audio_mode_resolved = audio_mode if audio_mode in ("sine", "silence") else "sine"
        HANDOFF_WAIT_S = 0.05

        safe_session_id = session_id.replace("-", "_")[:32]
        fifo_path = f"/tmp/avtr1_pytorch_audio_{safe_session_id}"
        try:
            os.unlink(fifo_path)
        except FileNotFoundError:
            pass
        os.mkfifo(fifo_path)
        print(f"[session-render] Audio FIFO created at {fifo_path}")

        audio_write_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)
        video_write_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=50)
        audio_writer_error: list[str] = []

        ffmpeg_cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "verbose",
            "-thread_queue_size", "1024",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "-s", f"{W}x{H}", "-r", str(FPS),
            "-i", "pipe:0",
            "-thread_queue_size", "1024",
            "-f", "s16le", "-ar", str(AUDIO_SR), "-ac", "1",
            "-i", fifo_path,
            "-map", "0:v", "-map", "1:a",
            "-c:v", "h264_nvenc",
            "-preset", "p4", "-tune", "ull",
            "-g", str(FPS), "-bf", "0",
            "-profile:v", "baseline", "-level", "3.1",
            "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "1",
            "-f", "mp4",
            "-movflags", "+frag_keyframe+empty_moov+default_base_moof",
            "-frag_duration", "200000",
            "pipe:1",
        ]

        print(f"[session-render] Spawning FFmpeg: {' '.join(ffmpeg_cmd)}")
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"[session-render] FFmpeg PID={ffmpeg_proc.pid} spawned")
        sess["ffmpeg_proc"] = ffmpeg_proc

        segment_queue: queue.Queue[tuple[str, bytes] | None] = queue.Queue(maxsize=100)
        ffmpeg_stderr_lines: list[str] = []
        reader_error_holder: list[str] = []
        audio_producer_start = [0.0]
        audio_producer_end = [0.0]

        def _audio_producer():
            audio_producer_start[0] = time.monotonic()
            chunk_interval_s = chunk_ms / 1000.0

            for ci in range(num_chunks):
                target_time = audio_producer_start[0] + ci * chunk_interval_s
                sleep_s = target_time - time.monotonic()
                if sleep_s > 0.001:
                    time.sleep(sleep_s)

                if audio_mode_resolved == "sine":
                    freq = 200.0
                    t_arr = np.arange(STEP_SAMPLES, dtype=np.float32) / AUDIO_SR
                    chunk_audio = (0.5 * np.sin(2.0 * np.pi * freq * t_arr)).astype(np.float32)
                else:
                    chunk_audio = np.zeros(STEP_SAMPLES, dtype=np.float32)

                meta = {
                    "chunkIndex": ci,
                    "isFirst": ci == 0,
                    "isLast": ci == num_chunks - 1,
                    "samples": len(chunk_audio),
                }

                ok = handoff.push(chunk_audio, metadata=meta, timeout=30.0)
                if not ok:
                    print(f"[session-audio] push FAILED at chunk {ci}")
                    break

            handoff.close()
            audio_producer_end[0] = time.monotonic()
            total_s = audio_producer_end[0] - audio_producer_start[0]
            print(f"[session-audio] DONE: {num_chunks} chunks pushed in {total_s:.2f}s mode={audio_mode_resolved}")

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
                print(f"[video_writer] exception: {exc}")
            finally:
                try:
                    ffmpeg_proc.stdin.close()
                except Exception:
                    pass
                print("[video_writer] exiting")

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
                print("[session-stdout] EOF, sentinel sent")
            except Exception as e:
                reader_error_holder.append(str(e))
                print(f"[session-stdout] exception: {e}")
                try:
                    segment_queue.put(None)
                except Exception:
                    pass

        producer_done_event = threading.Event()
        producer_timings: list[dict] = []
        total_raw_written = [0]
        _producer_start = [0.0]
        _producer_end = [0.0]

        from avtr1_renderer.types import Chunk, RenderOptions
        _mg = self.pipeline._motion_generator
        window = (_mg.chunk_size + _mg.future_size) * _mg.frame_len + _mg.audio_shift
        silence_audio = np.zeros(window, dtype=np.float32)
        _bg_id = _pick_first_bg(self.pipeline._backgrounds)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        def _video_producer():
            nonlocal render_state

            _producer_start[0] = time.monotonic()
            audio_output_buffer = np.zeros(0, dtype=np.float32)
            audio_history_buffer = np.zeros(0, dtype=np.float32)
            total_audio_bytes_written = 0
            _audio_write_count = [0]
            total_raw_bytes_written_local = [0]
            _video_write_count = [0]
            is_speech = False

            def _queue_audio(data: bytes, label: str) -> bool:
                nonlocal total_audio_bytes_written
                try:
                    audio_write_queue.put(data, timeout=2)
                    total_audio_bytes_written += len(data)
                    _audio_write_count[0] += 1
                    if _audio_write_count[0] <= 3 or _audio_write_count[0] % 100 == 0:
                        print(f"[session-render] audio queued: {len(data)}B #{_audio_write_count[0]} ({label})")
                    return True
                except queue.Full:
                    print(f"[session-render] audio_write_queue FULL, cannot queue {label}")
                    return False

            def _write_video_frames(frames_bytes: bytes, label: str) -> bool:
                try:
                    video_write_queue.put(frames_bytes, timeout=2)
                    total_raw_bytes_written_local[0] += len(frames_bytes)
                    _video_write_count[0] += 1
                    if _video_write_count[0] <= 3 or _video_write_count[0] % 100 == 0:
                        print(f"[session-render] video queued: {len(frames_bytes)}B ({label})")
                    return True
                except queue.Full:
                    print(f"[session-render] video_write_queue FULL, cannot queue {label}")
                    return False

            chunk_i = 0
            while True:
                if audio_source_resolved == "internal":
                    if chunk_i >= num_chunks:
                        break
                else:
                    if render_stop_event.is_set():
                        print(f"[session-video] STOP EVENT SET at chunk {chunk_i}")
                        break

                t_chunk_start = time.monotonic()
                chunk_timings: dict = {"chunk_index": chunk_i}

                t_audio_pop_start = time.monotonic()
                if audio_source_resolved == "handoff":
                    pop_timeout_s = HANDOFF_WAIT_S
                else:
                    pop_timeout_s = HANDOFF_WAIT_S if is_speech else 0
                pop_result = handoff.pop(timeout_s=pop_timeout_s)
                audio_wait_ms = round((time.monotonic() - t_audio_pop_start) * 1000, 2)
                chunk_timings["audio_wait_ms"] = audio_wait_ms

                if pop_result is not None:
                    chunk_audio, chunk_meta = pop_result
                    audio_underflow = False
                    is_speech = True
                else:
                    chunk_audio = np.zeros(STEP_SAMPLES, dtype=np.float32)
                    audio_underflow = True
                    is_speech = False

                chunk_timings["audio_underflow"] = audio_underflow
                chunk_timings["audio_chunk_samples"] = len(chunk_audio)

                audio_output_buffer = np.concatenate([audio_output_buffer, chunk_audio])
                audio_history_buffer = np.concatenate([audio_history_buffer, chunk_audio])

                if len(audio_output_buffer) >= STEP_SAMPLES:
                    ffmpeg_audio = audio_output_buffer[:STEP_SAMPLES]
                    audio_output_buffer = audio_output_buffer[STEP_SAMPLES:]
                else:
                    ffmpeg_audio = np.pad(audio_output_buffer, (0, STEP_SAMPLES - len(audio_output_buffer)))
                    audio_output_buffer = np.zeros(0, dtype=np.float32)

                label = f"speech chunk {chunk_i}" if not audio_underflow else f"idle chunk {chunk_i}"
                audio_s16 = (ffmpeg_audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()

                if audio_underflow:
                    sender.add_meta({
                        "mode": "idle",
                        "turnId": None,
                        "chunkIndex": chunk_i,
                        "isFirst": False,
                        "isLast": False,
                        "chunkRenderMs": 0,
                        "framesFed": 0,
                        "audioSamplesPerChunk": STEP_SAMPLES,
                    })

                if not _queue_audio(audio_s16, label):
                    print(f"[session-video] audio queue full at chunk {chunk_i}, stopping")
                    break

                if len(audio_history_buffer) >= window:
                    audio_tail = audio_history_buffer[-window:].copy()
                else:
                    audio_tail = np.pad(audio_history_buffer, (window - len(audio_history_buffer), 0))

                t_render_start = time.monotonic()
                chunk_obj = Chunk(audio_speech=audio_tail, audio_listen=silence_audio)
                with torch.inference_mode():
                    new_state, frames_iter = self.pipeline.process_chunk(
                        av, chunk_obj, state=render_state, options=_render_options,
                    )
                    frames_list = list(frames_iter)
                render_state = new_state
                render_ms = (time.monotonic() - t_render_start) * 1000
                chunk_timings["render_ms"] = round(render_ms, 2)

                if not audio_underflow:
                    sender.add_meta({
                        "mode": "speech",
                        "turnId": chunk_meta.get("turnId"),
                        "chunkIndex": chunk_meta.get("chunkIndex", chunk_i),
                        "isFirst": bool(chunk_meta.get("isFirst", False)),
                        "isLast": bool(chunk_meta.get("isLast", False)),
                        "totalChunks": chunk_meta.get("totalChunks"),
                        "generatedAt": chunk_meta.get("generatedAt"),
                        "enqueueAt": chunk_meta.get("enqueueAt"),
                        "chunkRenderMs": render_ms,
                        "framesFed": len(frames_list),
                        "audioSamplesPerChunk": STEP_SAMPLES,
                    })

                raw_chunk = b"".join(f.data.tobytes() for f in frames_list)

                t_write_start = time.monotonic()
                if not _write_video_frames(raw_chunk, label):
                    print(f"[session-video] video queue full at chunk {chunk_i}, stopping")
                    break
                total_raw_written[0] += len(raw_chunk)
                chunk_timings["ffmpeg_write_ms"] = round((time.monotonic() - t_write_start) * 1000, 2)
                chunk_timings["chunk_wall_ms"] = round((time.monotonic() - t_chunk_start) * 1000, 2)

                if chunk_i < 10 or chunk_i % 100 == 0:
                    print(f"[session-video] chunk {chunk_i}: audio_wait={audio_wait_ms}ms "
                          f"underflow={audio_underflow} render={chunk_timings['render_ms']}ms "
                          f"frames={len(frames_list)} wall={chunk_timings['chunk_wall_ms']}ms")

                max_history_samples = window + STEP_SAMPLES * 2
                if len(audio_history_buffer) > max_history_samples:
                    audio_history_buffer = audio_history_buffer[-window:]

                sess["chunk_count"] = chunk_i + 1
                producer_timings.append(chunk_timings)
                chunk_i += 1

            sess["render_state"] = render_state

            _producer_end[0] = time.monotonic()
            producer_done_event.set()
            total_s = _producer_end[0] - _producer_start[0]
            chunks_rendered = chunk_i
            print(f"[session-video] DONE: {chunks_rendered} chunks in {total_s:.2f}s source={audio_source_resolved}")

        # Phase 4D: capture producer thread exceptions so the caller can
        # set status="error" and emit an SSE error event.
        producer_error_holder: list[Exception] = []

        def _video_producer_wrapped():
            try:
                _video_producer()
            except Exception as e:
                producer_error_holder.append(e)
                print(f"[session-video] EXCEPTION: {e}")
                try:
                    self._emit_sse_error_and_end(sess, f"producer_thread_exception: {e}")
                except Exception:
                    pass
                try:
                    segment_queue.put_nowait(None)
                except Exception:
                    pass
                producer_done_event.set()

        # Phase 4D: only start the internal audio producer for smoke tests.
        # Production streaming mode (audio_source="handoff") expects the
        # Bridge to push real audio via http_push_audio into the session's
        # _AudioHandoff queue.
        if audio_source_resolved == "internal":
            audio_thread = threading.Thread(target=_audio_producer, daemon=True)
        else:
            audio_thread = None
        stderr_thread = threading.Thread(target=_ffmpeg_stderr_reader, daemon=True)
        stdout_thread = threading.Thread(target=_ffmpeg_stdout_reader, daemon=True)
        audio_writer_thread = threading.Thread(target=_audio_writer, daemon=True)
        video_writer_thread = threading.Thread(target=_video_writer, daemon=True)
        producer_thread = threading.Thread(target=_video_producer_wrapped, daemon=True)

        def _drain_watcher():
            producer_done_event.wait()
            try:
                video_write_queue.put(None, timeout=2)
            except queue.Full:
                pass
            try:
                audio_write_queue.put(None, timeout=2)
            except queue.Full:
                pass
            video_writer_thread.join(timeout=5)
            audio_writer_thread.join(timeout=5)

        drain_thread = threading.Thread(target=_drain_watcher, daemon=True)

        sess["ffmpeg_threads"] = [
            t for t in (audio_thread, stderr_thread, stdout_thread,
                        audio_writer_thread, video_writer_thread, producer_thread,
                        drain_thread)
            if t is not None
        ]

        stream_start_mono = time.monotonic()
        sender = _SegmentSender(
            chunk_media_s=CHUNK_MEDIA_S,
            target_lead_s=TARGET_LEAD_S,
            stream_start_mono=stream_start_mono,
        )
        sess["_sender_ref"] = sender
        sess["_speech_first_segment_emitted"] = False
        sess["_speech_end_segment_emitted"] = False
        sess["_current_turn_id"] = None

        if audio_thread is not None:
            audio_thread.start()
        stderr_thread.start()
        stdout_thread.start()
        audio_writer_thread.start()
        video_writer_thread.start()
        producer_thread.start()
        drain_thread.start()

        sentinel_seen = False
        total_encoded_bytes = 0
        consumer_start_mono = time.monotonic()
        sess_max_consumer_s = sess.get("max_consumer_s")
        max_consumer_s = float(sess_max_consumer_s) if sess_max_consumer_s else None
        first_media_wall_ms = 0.0

        print("[session-render] consumer loop starting")

        while True:
            if max_consumer_s is not None and time.monotonic() - consumer_start_mono > max_consumer_s:
                print("[session-render] CONSUMER TIMEOUT")
                break

            got_any = False
            while True:
                try:
                    item = segment_queue.get_nowait()
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                    if item[0] == "media" and first_media_wall_ms == 0.0:
                        first_media_wall_ms = round((time.monotonic() - stream_start_mono) * 1000, 1)
                    got_any = True
                except queue.Empty:
                    break

            sse_events_iter = sender.yield_events()
            if sse_events_iter:
                self._fanout_sse_events(sess, sse_events_iter)

            if sentinel_seen:
                sender.set_drain_fast(True)
                break

            if not got_any and not stdout_thread.is_alive():
                try:
                    item = segment_queue.get(timeout=1.0)
                    if item is None:
                        sentinel_seen = True
                        break
                    sender.add(item[0], item[1])
                    total_encoded_bytes += len(item[1])
                except queue.Empty:
                    print("[session-render] stdout thread dead, queue empty, exiting consumer")
                    break

            if not got_any:
                time.sleep(0.01)

        sender.set_drain_fast(True)
        drained_final = sender.drain_all() or []
        if drained_final:
            self._fanout_sse_events(sess, drained_final)
        consumer_end_mono = time.monotonic()

        print(f"[session-render] consumer loop done: sentinel={sentinel_seen} "
              f"sender_media={len(sender.collected_media)} "
              f"drained_final={len(drained_final)}")

        try:
            render_stop_event.set()
            handoff.close()

            producer_thread.join(timeout=10)
            producer_exited = not producer_thread.is_alive()
            if not producer_exited:
                print(f"[session-render] WARNING: producer_thread did not exit within 10s")

            drain_thread.join(timeout=10)

            try:
                video_write_queue.put(None, timeout=2)
            except queue.Full:
                pass
            try:
                audio_write_queue.put(None, timeout=2)
            except queue.Full:
                pass

            video_writer_thread.join(timeout=5)
            audio_writer_thread.join(timeout=5)
            if audio_thread is not None:
                audio_thread.join(timeout=5)
            stderr_thread.join(timeout=3)
            stdout_thread.join(timeout=3)

            if ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.terminate()
                    ffmpeg_proc.wait(timeout=5)
                except Exception:
                    try:
                        ffmpeg_proc.kill()
                        ffmpeg_proc.wait(timeout=2)
                    except Exception:
                        pass

            try:
                os.unlink(fifo_path)
            except OSError:
                pass

            try:
                zombie_check = subprocess.run(
                    ["ps", "-p", str(ffmpeg_proc.pid)],
                    capture_output=True, timeout=2,
                )
                zombie_status = "zombie_still_running" if zombie_check.returncode == 0 else "clean"
            except Exception:
                zombie_status = "unknown"

            ffmpeg_rc = ffmpeg_proc.returncode
            n_stderr_lines = len(ffmpeg_stderr_lines)
            ffmpeg_errors = [l for l in ffmpeg_stderr_lines if "error" in l.lower()]
            ffmpeg_warnings = [l for l in ffmpeg_stderr_lines if "warning" in l.lower()]

            print(f"[session-render] FFmpeg exit: rc={ffmpeg_rc} stderr_lines={n_stderr_lines} "
                  f"errors={len(ffmpeg_errors)} warnings={len(ffmpeg_warnings)} zombie={zombie_status}")

            sess["ffmpeg_rc"] = ffmpeg_rc
            sess["ffmpeg_proc"] = None
            sess["ffmpeg_threads"] = []
        finally:
            pass

        init_info = sender.collected_init
        media_events = sender.collected_media

        if init_info is not None:
            sess["fmp4_init_segment"] = init_info
        for evt in media_events:
            if len(sess["fmp4_media_segments"]) < FMP4_MEDIA_CAP:
                sess["fmp4_media_segments"].append(evt)
            sess["fmp4_media_count"] += 1
            sess["fmp4_media_total_bytes"] += evt.get("byteLength", 0)

        handoff_stats = handoff.stats()

        feats_deltas = []
        motion_means = []
        for pt in producer_timings:
            for ct in pt if isinstance(pt, (list, tuple)) else [pt]:
                if isinstance(ct, dict):
                    feats_deltas.append(ct.get("feats_speech_vs_listen_delta", 0))
                    motion_means.append(ct.get("motion_abs_mean", 0))

        audio_evidence = {
            "audio_mode": audio_mode_resolved,
            "feats_delta_mean": round(sum(feats_deltas) / len(feats_deltas), 6) if feats_deltas else 0,
            "feats_delta_min": round(min(feats_deltas), 6) if feats_deltas else 0,
            "feats_delta_max": round(max(feats_deltas), 6) if feats_deltas else 0,
            "motion_mean": round(sum(motion_means) / len(motion_means), 6) if motion_means else 0,
            "audio_path_active": any(d > 1e-6 for d in feats_deltas) if feats_deltas else False,
        }

        total_wall_s = time.monotonic() - t_method_start
        vram_mb = round(torch.cuda.memory_allocated() / 1024**2, 1)

        # Phase 4D: status transition depends on stop reason.
        if producer_error_holder:
            sess["status"] = "error"
            final_status_reason = f"producer_exception: {producer_error_holder[0]}"
        elif audio_source_resolved == "handoff" and render_stop_event.is_set():
            sess["status"] = "ended"
            final_status_reason = "render_stop_event_set"
        else:
            sess["status"] = "completed"
            final_status_reason = "all_chunks_rendered"

        print(f"[session-render] === session render done: {session_id} "
              f"status={sess['status']} reason={final_status_reason} "
              f"{total_wall_s:.2f}s ===")
        print(f"[session-render] init={init_info is not None} media_count={len(media_events)} "
              f"rawMB={total_raw_written[0]/1e6:.1f} encodedMB={total_encoded_bytes/1e6:.1f} "
              f"audio_path_active={audio_evidence['audio_path_active']} "
              f"vram={vram_mb}MB zombie={zombie_status}")

        if producer_error_holder:
            try:
                self._emit_sse_error_and_end(
                    sess,
                    f"producer_exception: {producer_error_holder[0]}",
                )
            except Exception:
                pass

        return {
            "ok": len(producer_error_holder) == 0,
            "session_id": session_id,
            "status": sess["status"],
            "status_reason": final_status_reason,
            "audio_source": audio_source_resolved,
            "stop_event_set": render_stop_event.is_set() if render_stop_event is not None else False,
            "total_wall_s": round(total_wall_s, 4),
            "audio_evidence": audio_evidence,
            "audio_handoff": handoff_stats,
            "init_segment": {
                "generated": init_info is not None,
                "byte_count": init_info.get("byteLength", 0) if init_info else 0,
            },
            "media_segments": {
                "count": len(media_events),
                "total_bytes": total_encoded_bytes,
                "sizes": [e["byteLength"] for e in media_events],
                "first_latency_ms": first_media_wall_ms,
            },
            "ffmpeg": {
                "return_code": ffmpeg_rc,
                "stderr_line_count": n_stderr_lines,
                "stderr_error_count": len(ffmpeg_errors),
                "stderr_warning_count": len(ffmpeg_warnings),
                "zombie_check": zombie_status,
            },
            "bytes": {
                "raw_written": total_raw_written[0],
                "encoded_total": total_encoded_bytes,
                "compression_ratio": round(total_raw_written[0] / total_encoded_bytes, 2) if total_encoded_bytes > 0 else 0,
            },
            "vram_mb": vram_mb,
            "reader_errors": reader_error_holder,
        }

    def _run_session_smoke_impl(self, session_id: str, num_chunks: int = 10,
                                chunk_ms: int = 200, audio_mode: str = "sine") -> dict:
        """Phase 4C: Pure implementation of run_session_smoke.

        Session-level smoke test: full audio+video pipeline bound to a session.

        Validates the session is in status "ready"/"running", then delegates
        to `_run_session_render_impl` which uses the session's stored state
        (render_state, handoff queue, avatar tensors, FFmpeg proc)
        to run the full pipeline. On completion, session status becomes
        "completed" (use `end_session` to release resources and pop from the
        dict). This is the primary session-level integration smoke test.
        """
        t0 = time.monotonic()
        print(f"[session-smoke] === run_session_smoke: session={session_id} "
              f"chunks={num_chunks} chunk_ms={chunk_ms} audio={audio_mode} ===")

        if session_id not in self._sessions:
            return {
                "ok": False,
                "error": f"session {session_id} not found. Call start_session first.",
                "active_sessions": list(self._sessions.keys()),
            }

        sess = self._sessions[session_id]
        if sess.get("status") not in ("ready", "running"):
            return {
                "ok": False,
                "error": f"session status is {sess.get('status')}, expected ready/running",
                "session_id": session_id,
            }

        result = self._run_session_render_impl(session_id, num_chunks, chunk_ms, audio_mode)
        total_s = time.monotonic() - t0

        sess = self._sessions.get(session_id, {})
        print(f"[session-smoke] === done: session={session_id} total={total_s:.2f}s "
              f"status={sess.get('status')} ===")

        result["smoke_total_s"] = round(total_s, 4)
        result["session_status_after"] = sess.get("status", "unknown")
        result["session_chunk_count"] = sess.get("chunk_count", 0)
        result["session_fmp4_media_count"] = sess.get("fmp4_media_count", 0)

        return result

    @modal.method()
    def run_session_smoke(self, session_id: str, num_chunks: int = 10,
                          chunk_ms: int = 200, audio_mode: str = "sine") -> dict:
        """RPC entrypoint — delegates to _run_session_smoke_impl.

        No @modal.fastapi_endpoint() wrapper: run_session_smoke is a
        diagnostic-only RPC, not a production path. The Bridge does not
        call it.
        """
        return self._run_session_smoke_impl(session_id, num_chunks, chunk_ms, audio_mode)
