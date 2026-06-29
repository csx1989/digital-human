"""
Phase 5Q — Restore Path Optimization (timing instrumentation + optimizations)
CPU-Only Modal Memory Snapshot on L40S

Fork of Phase 5N (proven GO, session_ready ~ 17.5s).

Phase 5Q optimizes the snap=False restore path:
  5Q.1: Detailed timing instrumentation around every step
  5Q.2: Cached avatar fast path (skip AvatarLoader + registration)
  5Q.3: Strict TRT audit (HOLD if ONNX fallback in hot runtime)
  5Q.4: Single-background loading (skip unused backgrounds)
  5Q.5: Remove discarded warmup from start-session
  5Q.6: Validation + browser test

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py

Invoke (NEVER modal.run):
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s-5q', 'AvatarLifecycleCpuSnapshotSession5Q')
inst = cls()
print(json.dumps(inst.health_check.remote(), indent=2))
    "
"""

from __future__ import annotations

# ── Stage 3B reliability repair: gated CUDA_LAUNCH_BLOCKING ──────────────────
# Must be set BEFORE any torch CUDA context creation (which happens in
# @modal.enter(snap=False) _wake_up).  Gated on AVTR_CUDA_LAUNCH_BLOCKING
# env var, typically injected via a Modal Secret for a bounded diagnostic
# deploy.  NEVER leave enabled in production — slows inference 2-10x.
import os as _os
if _os.environ.get("AVTR_CUDA_LAUNCH_BLOCKING", "0") == "1":
    _os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    print("[5q] CUDA_LAUNCH_BLOCKING=1 ENABLED at module load (diagnostic mode)")

# Media timing debug gate — shared with webrtc_publisher.py.
# When AVTR_MEDIA_TIMING_DEBUG=1, per-stage timing counters are collected
# for the fMP4 adapter path and surfaced in the stream-end event +
# adapter_results.  When disabled (default), zero overhead.
_MEDIA_TIMING_DEBUG = _os.environ.get("AVTR_MEDIA_TIMING_DEBUG", "0") == "1"

import base64
import collections
import hashlib
import json
import queue
import struct
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import modal

from avtr_segment_sender import _SegmentSender, stamp_media_event

from fastapi import Request, HTTPException

from request_signing import verify_modal_request

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
        "pkg-config",
        "clang",
        # Phase 3B: GStreamer/NVENC runtime for GstWebRtcPublisher
        # (opt-in via AVTR_SFU_PUBLISHER=gstreamer). These are inert when
        # the aiortc/fMP4 path is used — no import, no runtime cost.
        # Package set verified on Ubuntu 24.04 (nvidia/cuda devel base).
        "gstreamer1.0-tools",           # core CLI + plugin scanner
        "gstreamer1.0-plugins-base",    # appsrc, GstSdp GIR
        "gstreamer1.0-plugins-good",    # rtph264pay (RTP payloader)
        "gstreamer1.0-plugins-bad",     # webrtcbin, nvh264enc, h264parse, GstWebRTC GIR
        "gstreamer1.0-nice",            # libnice ICE plugin (required by webrtcbin, not in -bad)
        "python3-gi",                   # PyGObject (installs into /usr/lib/python3/dist-packages)
        "gir1.2-gstreamer-1.0",         # Gst GObject introspection
        "gir1.2-gst-plugins-base-1.0",  # GstSdp introspection
        "gir1.2-gst-plugins-bad-1.0",   # GstWebRTC introspection
        # Phase D1: PyAV build deps (FFmpeg 7 dev headers for av>=14)
        "libavformat-dev",
        "libavcodec-dev",
        "libavdevice-dev",
        "libavfilter-dev",
        "libavutil-dev",
        "libswscale-dev",
        "libswresample-dev",
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
        "onnx-graphsurgeon>=0.5",
        # onnxruntime-gpu 1.21+ defaults to CUDA 13 on PyPI; 1.20.x is the last
        # CUDA 12.x compatible release. Pin to avoid libcudart.so.13 ImportError.
        "onnxruntime-gpu>=1.20,<1.21",
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
    .pip_install(
        # Phase D1: WebRTC SFU publisher — aiortc installs cleanly from wheel
        "aiortc>=1.9,<2",
    )
    .run_commands(
        # Phase D1: PyAV (av>=14) MUST use pre-built wheel — it bundles its own
        # FFmpeg. Building from source against system FFmpeg 7.x fails due to
        # D3D12VA constant mismatch. --only-binary forces wheel, never source.
        "pip install 'av>=14,<15' --only-binary av",
    )
    .add_local_dir(
        "/Users/ayya/developer/avtr-1/src/avtr1_renderer",
        remote_path="/opt/avtr1/avtr1_renderer",
        copy=True,
    )
    # Phase 3B: TRT/GStreamer coexistence patch.
    # When PyGObject (gi) is loaded in the same process as TensorRT,
    # iterating TRT Dims via tuple() crashes (SIGSEGV/SIGABRT for 4-D+
    # tensors) due to PyGObject allocator hooks + pybind11 __iter__
    # interaction. The patch replaces tuple(shape) with index-based
    # _dims_to_tuple(shape) which produces identical results safely.
    # Applied unconditionally — behaviorally identical for non-GStreamer
    # paths (fMP4, aiortc). Verified by fMP4 regression post-deploy.
    # Ref: https://dev.to/micwarsoft/tensorrt-trtdims-sigsegv-inside-a-gstreamer-python-plugin-root-cause-and-fix-cfl
    .run_commands(
        "python3 -c \""
        "p='/opt/avtr1/avtr1_renderer/runtime/trt.py';"
        "s=open(p).read();"
        "s=s.replace('trt.init_libnvinfer_plugins(_logger, \\\"\\\")',"
        "'trt.init_libnvinfer_plugins(_logger, \\\"\\\")\\n\\n\\n"
        "def _dims_to_tuple(d):\\n    return tuple(d[i] for i in range(len(d)))');"
        "s=s.replace('tuple(self.engine.get_tensor_shape(name))',"
        "'_dims_to_tuple(self.engine.get_tensor_shape(name))');"
        "s=s.replace('tuple(ctx.get_tensor_shape(name))',"
        "'_dims_to_tuple(ctx.get_tensor_shape(name))');"
        "open(p,'w').write(s)"
        "\""
    )
    .run_commands(
        "python3 -c \""
        "import re;"
        "p='/opt/avtr1/avtr1_renderer/runtime/trt.py';"
        "s=open(p).read();"
        "assert '_dims_to_tuple' in s, 'TRT_PATCH_VERIFY FAILED: _dims_to_tuple missing';"
        "raw_ctx=len(re.findall(r'(?<!_dims_to_)tuple\\(ctx\\.get_tensor_shape', s));"
        "raw_eng=len(re.findall(r'(?<!_dims_to_)tuple\\(self\\.engine\\.get_tensor_shape', s));"
        "assert raw_ctx==0 and raw_eng==0, 'TRT_PATCH_VERIFY FAILED: raw_ctx=%d raw_eng=%d' % (raw_ctx, raw_eng);"
        "patched=s.count('_dims_to_tuple(self.engine.')+s.count('_dims_to_tuple(ctx.');"
        "assert patched==2, 'TRT_PATCH_VERIFY FAILED: expected 2 patched calls, found %d' % patched;"
        "print('TRT_PATCH_VERIFY helper=true patched_calls=%d raw_remaining=0' % patched)"
        "\""
    )
    .add_local_dir(
        "/tmp/avtr-1-official/scripts",
        remote_path="/opt/build_scripts",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/avtr_segment_sender.py",
        remote_path="/opt/avtr1/avtr_segment_sender.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/webrtc_publisher.py",
        remote_path="/opt/avtr1/webrtc_publisher.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/_publisher_loop.py",
        remote_path="/opt/avtr1/_publisher_loop.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/request_signing.py",
        remote_path="/opt/avtr1/request_signing.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/gst_webrtc_publisher.py",
        remote_path="/opt/avtr1/gst_webrtc_publisher.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/pion_debug_publisher.py",
        remote_path="/opt/avtr1/pion_debug_publisher.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/whip_publisher.py",
        remote_path="/opt/avtr1/whip_publisher.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/tools/pion-forwarder/pion-forwarder-linux-amd64",
        remote_path="/usr/local/bin/pion-forwarder",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/scripts/build_avtr1_engines.py",
        remote_path="/opt/avtr1/build_avtr1_engines.py",
        copy=True,
    )
    .add_local_file(
        "/Users/ayya/developer/avtr-1/scripts/build_renderer_engines.py",
        remote_path="/opt/avtr1/build_renderer_engines.py",
        copy=True,
    )
    # Make the app module importable from prep helpers / health-check scripts
    # that reuse the production image, volume, and constants.
    .add_local_file(
        "/Users/ayya/developer/avtr-1/avatar-saas-prototype/modal-avatar-runtime/modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py",
        remote_path="/opt/avtr1/modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/usr/lib/python3/dist-packages:/opt/avtr1",
            "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH,
            "AVTR_SFU_PUBLISHER": "gstreamer",
        }
    )
    .apt_install("libcudnn9-cuda-12")
    .run_commands("chmod +x /usr/local/bin/pion-forwarder")
)

app = modal.App("avtr1-lifecycle-pytorch-cpu-snapshot-session-l40s-5q", image=image)

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

SKIP_STARTUP_WARMUP = False
OPT_CACHED_AVATAR = True
OPT_SINGLE_BACKGROUND = True
OPT_PHASE_LABEL = "5Q.5_production_hardening"
DEBUG_STARTUP_DIAGNOSTICS = False
OPT_CPU_PRELOAD = False
OPT_ENGINE_BYTES = True
OPT_PARALLEL_ENGINES = False

MAX_GPU_AVATARS = 3
MAX_BG_CACHE = 3


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
    candidate_dirs = [
        storage_root / "main" / "avatars_artifacts" / "reference_frames",
        storage_root / "avatars_artifacts" / "reference_frames",
    ]

    try:
        from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
        mgr_path = get_artifact_manager().get_artifact_path("reference_frames")
        if mgr_path is not None:
            mgr_path = Path(mgr_path)
            if mgr_path not in candidate_dirs:
                candidate_dirs.append(mgr_path)
    except Exception:
        pass

    for portraits_dir in candidate_dirs:
        if not portraits_dir.is_dir():
            continue
        all_pngs = sorted(portraits_dir.glob("*.png"))
        if all_pngs:
            stems = [p.stem for p in all_pngs]
            print(f"[l40s] Found {len(all_pngs)} portrait(s) in {portraits_dir}")
            if "arnold" in stems:
                return "arnold"
            chosen = stems[0]
            print(f"[l40s] Using avatar '{chosen}'")
            return chosen

    checked = "\n  ".join(str(d) for d in candidate_dirs)
    raise FileNotFoundError(
        f"No portrait PNGs found in any candidate directory:\n  {checked}"
    )


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


# ── Phase 3A: SFU publisher selection (opt-in GStreamer) ─────────────────────
# The GStreamer/NVENC publisher is selected ONLY when AVTR_SFU_PUBLISHER is
# exactly "gstreamer". All other values (unset, empty, "aiortc", typos) fall
# back to the existing aiortc WebRtcPublisher. This is a runtime env flag, not
# a build-time constant, so the same image can serve both paths.

def _resolve_publisher_type() -> str:
    """Return 'gstreamer' or 'aiortc' based on AVTR_SFU_PUBLISHER env."""
    val = _os.environ.get("AVTR_SFU_PUBLISHER", "")
    return "gstreamer" if val == "gstreamer" else "aiortc"


class _GstSourceMetrics:
    """Frame-source metrics compatible with _verify_publisher_healthy access.

    The health check reads metrics.frames_generated and metrics.underflow_count
    as mutable int attributes (same interface as aiortc PublisherMetrics).
    """

    def __init__(self) -> None:
        self.frames_generated: int = 0
        self.frames_emitted: int = 0
        self.underflow_count: int = 0
        self.frames_repeated: int = 0
        self.queue_depth_min: int = 10 ** 9
        self.queue_depth_max: int = 0
        self.handoff_chunks_consumed: int = 0
        self.speech_chunks_rendered: int = 0
        self.idle_chunks_rendered: int = 0
        self.speech_turns_started: int = 0
        self.speech_turns_completed: int = 0
        self.last_turn_id: str | None = None
        self.chunk_latencies_ms: list[float] = []

    def snapshot(self) -> dict:
        return {
            "frames_generated": self.frames_generated,
            "frames_emitted": self.frames_emitted,
            "underflow_count": self.underflow_count,
            "frames_repeated": self.frames_repeated,
            "queue_depth_min": self.queue_depth_min if self.queue_depth_min < 10 ** 9 else 0,
            "queue_depth_max": self.queue_depth_max,
            "handoff_chunks_consumed": self.handoff_chunks_consumed,
            "speech_chunks_rendered": self.speech_chunks_rendered,
            "idle_chunks_rendered": self.idle_chunks_rendered,
            "speech_turns_started": self.speech_turns_started,
            "speech_turns_completed": self.speech_turns_completed,
            "last_turn_id": self.last_turn_id,
        }


# ── Boundary-A capture: raw I420 frame diagnostic ────────────────────────────

_FRAME_CAPTURE_ENABLED = _os.environ.get("AVTR_FRAME_CAPTURE_DEBUG", "0") == "1"
_FRAME_CAPTURE_LABEL = _os.environ.get("AVTR_FRAME_CAPTURE_LABEL", "both")
if _FRAME_CAPTURE_ENABLED:
    print(f"[5q] FRAME_CAPTURE_DEBUG=1 label={_FRAME_CAPTURE_LABEL} — capture hooks ARMED")
else:
    print(f"[5q] FRAME_CAPTURE_DEBUG disabled (env={_os.environ.get('AVTR_FRAME_CAPTURE_DEBUG', '<unset>')})")


class _FrameCaptureDebug:
    """Bounded raw-I420-frame capture for idle-flicker boundary analysis.

    During capture: stores (timestamp, ndarray) references only — no I/O,
    no PNG encoding, no disk writes.  O(1) per frame.
    After capture (finish()): a background thread computes per-frame metrics
    (Y-mean, MAD, hash), saves PNGs + contact sheet + JSON to the volume.

    Dual-gated via AVTR_FRAME_CAPTURE_DEBUG=1 + AVTR_FRAME_CAPTURE_LABEL.
    Zero overhead when disabled (not instantiated).
    """

    MAX_FRAMES = 150
    PNG_SAMPLES = 10

    def __init__(self, label: str, session_id: str):
        import numpy as np  # noqa: already available in container

        self.label = label
        self.session_id = session_id
        self.capture_dir = Path(L40S_MOUNT_PATH) / "debug_captures" / session_id / label
        self._frames: list[tuple[float, object]] = []
        self._start_ts = time.monotonic()
        self._closed = False
        self._writer_thread: threading.Thread | None = None
        print(f"[capture-debug] START label={label} session={session_id} "
              f"max_frames={self.MAX_FRAMES}")

    @property
    def active(self) -> bool:
        return not self._closed and len(self._frames) < self.MAX_FRAMES

    def capture(self, frame_data) -> bool:
        """Store an immutable copy of the frame.  Returns True if captured, False if full."""
        import numpy as np

        if self._closed or len(self._frames) >= self.MAX_FRAMES:
            return False
        self._frames.append(
            (time.monotonic() - self._start_ts, np.array(frame_data, copy=True))
        )
        return True

    def finish(self):
        """Signal capture complete, start background writer."""
        if self._closed:
            return
        self._closed = True
        # Write artifacts SYNCHRONOUSLY (not in background thread).
        # This runs AFTER the render loop — not inside it — so there is
        # no frame-timing perturbation.  Background threads have trouble
        # with Modal volume commit, so we keep it simple and synchronous.
        try:
            self._write_artifacts()
        except Exception as exc:
            print(f"[capture-debug] _write_artifacts FAILED for {self.label}: {exc}")
            import traceback
            traceback.print_exc()
        self._writer_thread = None  # signal that finish() was called

    def wait(self, timeout: float = 30.0) -> bool:
        return True  # synchronous finish — nothing to wait for

    def _write_artifacts(self):
        """Compute metrics and write artifacts synchronously."""
        import os as _os_w
        import numpy as np

        n = len(self._frames)
        if n == 0:
            print(f"[capture-debug] {self.label}: no frames captured")
            return

        H, W = 720, 1280
        _os_w.makedirs(self.capture_dir, exist_ok=True)

        per_frame: list[dict] = []
        prev_y = None
        for i, (ts, data) in enumerate(self._frames):
            arr = np.asarray(data, dtype=np.uint8)
            y_plane = arr[:H, :].astype(np.float64)
            y_mean = float(y_plane.mean())
            y_std = float(y_plane.std())
            mad = 0.0
            if prev_y is not None:
                mad = float(np.abs(y_plane - prev_y).mean())
            h = hashlib.md5(arr.tobytes()).hexdigest()[:16]
            per_frame.append({
                "idx": i, "ts_s": round(ts, 6),
                "y_mean": round(y_mean, 4), "y_std": round(y_std, 4),
                "mad_vs_prev": round(mad, 6), "hash": h,
            })
            prev_y = y_plane

        cadence_ms = [
            round((self._frames[i][0] - self._frames[i - 1][0]) * 1000, 2)
            for i in range(1, n)
        ]
        mads = [f["mad_vs_prev"] for f in per_frame[1:]]
        y_means = [f["y_mean"] for f in per_frame]
        unique_hashes = len(set(f["hash"] for f in per_frame))
        dup_count = sum(
            1 for i in range(1, n)
            if per_frame[i]["hash"] == per_frame[i - 1]["hash"]
        )

        summary = {
            "label": self.label,
            "session_id": self.session_id,
            "frames_captured": n,
            "unique_hashes": unique_hashes,
            "consecutive_duplicate_count": dup_count,
            "y_mean": {
                "avg": round(float(np.mean(y_means)), 4),
                "std": round(float(np.std(y_means)), 4),
                "min": round(min(y_means), 4),
                "max": round(max(y_means), 4),
                "range": round(max(y_means) - min(y_means), 4),
            },
            "mad_vs_prev": {
                "mean": round(float(np.mean(mads)), 6) if mads else 0,
                "p50": round(float(np.median(mads)), 6) if mads else 0,
                "p95": round(float(np.percentile(mads, 95)), 6) if mads else 0,
                "max": round(max(mads), 6) if mads else 0,
            },
            "cadence_ms": {
                "mean": round(float(np.mean(cadence_ms)), 2) if cadence_ms else 0,
                "p50": round(float(np.median(cadence_ms)), 2) if cadence_ms else 0,
                "p95": round(float(np.percentile(cadence_ms, 95)), 2) if cadence_ms else 0,
                "min": round(min(cadence_ms), 2) if cadence_ms else 0,
                "max": round(max(cadence_ms), 2) if cadence_ms else 0,
            },
        }

        # Save evenly-spaced PNGs + contact sheet
        png_indices = list(range(0, n, max(1, n // self.PNG_SAMPLES)))[: self.PNG_SAMPLES]
        try:
            from PIL import Image as _PIL_Image

            thumb_w, thumb_h = 320, 180
            cols = min(5, len(png_indices))
            rows = (len(png_indices) + cols - 1) // cols
            sheet = _PIL_Image.new("L", (thumb_w * cols, thumb_h * rows), 0)
            for slot, idx in enumerate(png_indices):
                arr = np.asarray(self._frames[idx][1], dtype=np.uint8)
                y_plane = arr[:H, :].astype(np.uint8)
                img = _PIL_Image.fromarray(y_plane, mode="L")
                img.save(self.capture_dir / f"frame_{idx:04d}.png")
                thumb = img.resize((thumb_w, thumb_h))
                sheet.paste(thumb, ((slot % cols) * thumb_w, (slot // cols) * thumb_h))
            sheet.save(self.capture_dir / "contact_sheet.png")
        except ImportError:
            for idx in png_indices:
                arr = np.asarray(self._frames[idx][1], dtype=np.uint8)
                arr[:H, :].astype(np.uint8).tofile(
                    self.capture_dir / f"frame_{idx:04d}_y.raw"
                )

        with open(self.capture_dir / "per_frame.json", "w") as f:
            json.dump(per_frame, f, indent=2)
        with open(self.capture_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(self.capture_dir / "manifest.json", "w") as f:
            json.dump({
                "label": self.label,
                "session_id": self.session_id,
                "max_frames": self.MAX_FRAMES,
                "frames_captured": n,
                "capture_duration_s": round(
                    self._frames[-1][0] - self._frames[0][0], 3
                ) if n > 1 else 0,
                "pixel_format": "yuv_i420",
                "width": W, "height": H, "fps": 25,
            }, f, indent=2)

        # Free frame references
        self._frames.clear()
        print(f"[capture-debug] DONE {self.label}: {n} frames, "
              f"y_mean_std={summary['y_mean']['std']}, "
              f"mad_mean={summary['mad_vs_prev']['mean']}, "
              f"unique_hashes={unique_hashes}/{n}, "
              f"dups={dup_count}")

        # Commit volume so artifacts persist after container death
        try:
            l40s_volume.commit()
            print(f"[capture-debug] Volume committed for {self.label}")
        except Exception as e:
            print(f"[capture-debug] Volume commit FAILED for {self.label}: {e}")


def _maybe_create_capture(label: str, session_id: str) -> _FrameCaptureDebug | None:
    """Dual-gated factory for raw-frame capture.

    Gate 1: AVTR_FRAME_CAPTURE_DEBUG=1  (deploy-level Modal secret)
    Gate 2: AVTR_FRAME_CAPTURE_LABEL matches label or 'both'

    Checks env vars at RUNTIME (not module import time) to avoid
    stale values from memory snapshot restoration.
    """
    import os as _os_rt
    enabled = _os_rt.environ.get("AVTR_FRAME_CAPTURE_DEBUG", "0") == "1"
    cap_label = _os_rt.environ.get("AVTR_FRAME_CAPTURE_LABEL", "both")
    if not enabled:
        return None
    if cap_label not in ("both", label):
        return None
    print(f"[capture-debug] Creating capture: label={label} session={session_id}")
    return _FrameCaptureDebug(label, session_id)


class GstAvtrFrameSource:
    """Synchronous I420 frame source for the GStreamer publisher.

    Uses threading.Queue (sync) so GstWebRtcPublisher._push_frames() can call
    next_frame() from its own thread. The render thread mirrors the aiortc
    AvtrFrameSource._producer_loop pattern: silence-driven process_chunk with
    a rolling audio buffer.

    Lifecycle:
      1. Before start(): next_frame() returns cached black I420 frames (pre-roll).
         GstWebRtcPublisher calls source.start() AFTER ICE connects so no GPU
         work happens during SDP negotiation.
      2. After start(): render thread fills the queue. next_frame() pops real
         frames with underflow tracking.

    This class is ONLY instantiated when AVTR_SFU_PUBLISHER=gstreamer.
    Heavy imports (torch, avtr1_renderer) are deferred to the render thread
    body so module import is unaffected.
    """

    FPS = 25
    HANDOFF_POP_TIMEOUT_SPEECH_S = 0.05
    HANDOFF_POP_TIMEOUT_IDLE_S = 0.0

    def __init__(self, pipeline, avatar, bg_id, max_queue: int = 15, handoff=None,
                 frame_capture: _FrameCaptureDebug | None = None,
                 matrix_row: str = ""):
        import queue as _queue_mod
        import numpy as np

        self._pipeline = pipeline
        self._avatar = avatar
        self._bg_id = bg_id
        self._max_queue = max_queue
        self._handoff = handoff
        self._frame_capture = frame_capture
        self.fps = self.FPS
        self._matrix_row = matrix_row

        self.metrics = _GstSourceMetrics()
        self._queue = _queue_mod.Queue(maxsize=max_queue)
        self._stop_event = threading.Event()
        self._activated = False
        self._producer_thread: threading.Thread | None = None
        self._error: Exception | None = None

        h, w = 720, 1280
        self._last_frame = np.zeros((h * 3 // 2, w), dtype=np.uint8)
        self._last_frame[:h, :] = 16
        self._last_frame[h:, :] = 128

        mg = pipeline._motion_generator
        self._chunk_size = mg.chunk_size
        self._future_size = mg.future_size
        self._frame_len = mg.frame_len
        self._audio_shift = getattr(mg, "audio_shift", 0)
        self._window = (self._chunk_size + self._future_size) * self._frame_len + self._audio_shift
        self._step_samples = self._chunk_size * self._frame_len

        self.counter = 0
        self.frames_generated = 0
        self.frames_emitted = 0
        self.underflow_count = 0

    def start(self) -> None:
        if self._activated:
            return
        self._activated = True
        self._producer_thread = threading.Thread(
            target=self._render_loop, daemon=True, name="gst-avtr-render"
        )
        self._producer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=5)

    def get_error(self) -> Exception | None:
        return self._error

    @staticmethod
    def _source_audio_chunk(handoff, was_speech: bool, step_samples: int):
        """Pop one audio chunk from handoff, or return silence on underflow.

        Centralises the TTS-vs-idle decision so it is unit-testable without
        GPU/Modal/GStreamer.  The render loop is responsible for metrics and
        logging; this helper only decides what audio to feed.

        Args:
            handoff: _AudioHandoff instance or None.
            was_speech: whether the previous chunk was speech (controls pop
                timeout — short blocking after speech to bridge inter-chunk
                gaps, non-blocking when idle to avoid render-loop stalls).
            step_samples: number of samples expected per chunk (matches
                chunk_size * frame_len).

        Returns:
            (chunk_audio, is_speech, meta) where chunk_audio is a float32
            ndarray of step_samples, is_speech is True when a real TTS chunk
            was popped, and meta is the chunk metadata dict or None.
        """
        import numpy as np

        if handoff is None:
            return np.zeros(step_samples, dtype=np.float32), False, None

        pop_timeout = (
            GstAvtrFrameSource.HANDOFF_POP_TIMEOUT_SPEECH_S
            if was_speech
            else GstAvtrFrameSource.HANDOFF_POP_TIMEOUT_IDLE_S
        )
        pop_result = handoff.pop(timeout_s=pop_timeout)

        if pop_result is not None:
            chunk_audio, meta = pop_result
            return chunk_audio, True, meta

        return np.zeros(step_samples, dtype=np.float32), False, None

    def _render_loop(self) -> None:
        import time as _time_mod
        import numpy as np
        import torch
        from avtr1_renderer.types import Chunk, RenderOptions

        render_state = None
        silence_audio = np.zeros(self._window, dtype=np.float32)
        audio_history = np.zeros(0, dtype=np.float32)
        options = RenderOptions(pixel_format="yuv_i420", bg_id=self._bg_id)

        chunk_i = 0
        max_hist = self._window + self._step_samples * 2
        was_speech = False

        while not self._stop_event.is_set():
            try:
                chunk_audio, is_speech, chunk_meta = self._source_audio_chunk(
                    self._handoff, was_speech, self._step_samples
                )
                if is_speech:
                    self.metrics.handoff_chunks_consumed += 1
                    self.metrics.speech_chunks_rendered += 1
                    if chunk_meta and chunk_meta.get("isFirst"):
                        self.metrics.speech_turns_started += 1
                        self.metrics.last_turn_id = chunk_meta.get("turnId")
                        print(f"[gst-sfu] speech render started: turnId={chunk_meta.get('turnId')}")
                    if chunk_meta and chunk_meta.get("isLast"):
                        self.metrics.speech_turns_completed += 1
                        print(f"[gst-sfu] speech render completed: turnId={chunk_meta.get('turnId')}")
                else:
                    self.metrics.idle_chunks_rendered += 1
                was_speech = is_speech
                audio_history = np.concatenate([audio_history, chunk_audio])
                if len(audio_history) >= self._window:
                    audio_tail = audio_history[-self._window:].copy()
                else:
                    audio_tail = np.pad(audio_history, (self._window - len(audio_history), 0))
                if len(audio_history) > max_hist:
                    audio_history = audio_history[-self._window:]

                if is_speech:
                    chunk_obj = Chunk(audio_speech=audio_tail, audio_listen=silence_audio)
                else:
                    chunk_obj = Chunk(audio_speech=silence_audio, audio_listen=silence_audio)

                _t_chunk_start = _time_mod.monotonic()

                with torch.inference_mode():
                    new_state, frames_iter = self._pipeline.process_chunk(
                        self._avatar, chunk_obj, state=render_state, options=options,
                    )
                    frames_list = list(frames_iter)

                _chunk_ms = (_time_mod.monotonic() - _t_chunk_start) * 1000
                self.metrics.chunk_latencies_ms.append(_chunk_ms)
                if len(self.metrics.chunk_latencies_ms) > 500:
                    self.metrics.chunk_latencies_ms = self.metrics.chunk_latencies_ms[-500:]

                render_state = new_state

                if self._frame_capture and frames_list:
                    for f in frames_list:
                        self._frame_capture.capture(f.data)
                    if not self._frame_capture.active and not self._frame_capture._closed:
                        self._frame_capture.finish()

                for frame in frames_list:
                    if self._stop_event.is_set():
                        break
                    try:
                        self._queue.put(frame.data.copy(), block=True, timeout=5.0)
                        self.frames_generated += 1
                        self.metrics.frames_generated += 1
                    except Exception:
                        pass

                chunk_i += 1

            except Exception as exc:
                self._error = exc
                import traceback
                traceback.print_exc()
                break

    def next_frame(self):
        import queue as _queue_mod

        if not self._activated:
            self.metrics.frames_repeated += 1
            self.metrics.frames_emitted += 1
            return self._last_frame

        depth = self._queue.qsize()
        if depth < self.metrics.queue_depth_min:
            self.metrics.queue_depth_min = depth
        if depth > self.metrics.queue_depth_max:
            self.metrics.queue_depth_max = depth

        try:
            frame = self._queue.get(timeout=0.2)
            self._last_frame = frame
            self.metrics.frames_emitted += 1
            self.counter += 1
            return frame
        except _queue_mod.Empty:
            self.underflow_count += 1
            self.metrics.underflow_count += 1
            self.metrics.frames_repeated += 1
            return self._last_frame


@app.cls(
    gpu="L40S",
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("hf-secret"),
        modal.Secret.from_name("avtr-modal-worker-secret"),
        modal.Secret.from_name("avtr-modal-request-signing-secret"),
        modal.Secret.from_name("cf-stream-whip"),
        modal.Secret.from_dict({"AVTR_ENFORCE_MODAL_REQUEST_SIGNING": "true"}),
        modal.Secret.from_dict({"AVTR_SFU_RTP_AGGREGATE_MODE": "zero-latency"}),
    ],
    enable_memory_snapshot=True,
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=120,
)
@modal.concurrent(max_inputs=2)
class AvatarLifecycleCpuSnapshotSession5Q:

    @modal.enter(snap=True)
    def _snap_init(self):
        import numpy as np
        from pathlib import Path

        self._snap_ok = False
        self._full_render_warmed = False

        print("[snap] === CPU-Only Snapshot Init (Phase 5Q.3 hardened) ===")

        storage_root = Path(L40S_MOUNT_PATH)
        self.avatar_id = _pick_first_avatar(storage_root)
        print(f"[snap] avatar_id={self.avatar_id!r}")

        # --- CPU preload: cached avatar tensors (safe, no CUDA) ---------------
        self._cached_avatar_cpu_tensors = None
        if OPT_CPU_PRELOAD:
            try:
                import torch
                av_dir = storage_root / "avatars_pytorch_cache" / self.avatar_id
                if av_dir.is_dir():
                    required_files = (
                        "kp_info_kp.pt", "kp_info_exp.pt", "kp_info_scale.pt", "kp_info_t.pt",
                        "kp_info_pitch.pt", "kp_info_yaw.pt", "kp_info_roll.pt", "kp_info_R.pt",
                        "f_s.pt", "M_grid.pt", "mask.pt", "source.pt", "bg.pt", "normalizer.pt",
                    )
                    if all((av_dir / f).is_file() for f in required_files):
                        tensors = {}
                        for name in required_files:
                            tensors[name] = torch.load(
                                av_dir / name, map_location="cpu", weights_only=True
                            )
                        self._cached_avatar_cpu_tensors = tensors
                        print(f"[snap] CPU preloaded avatar tensors for {self.avatar_id}")
                    else:
                        print("[snap] CPU preload: cache incomplete, skipping")
                else:
                    print("[snap] CPU preload: no cache dir, skipping")
            except Exception as exc:
                print(f"[snap] CPU preload FAILED: {exc}")
                self._cached_avatar_cpu_tensors = None

        # --- CPU preload: engine bytes (safe, no TensorRT/CUDA) ---------------
        self._engine_bytes = None
        if OPT_ENGINE_BYTES:
            try:
                from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager, get_trt_engine_path, find_engine_or_onnx
                engine_bytes = {}
                total_mb = 0.0
                # List all hot engines to preload
                engines_to_preload = [
                    ("avtr1_encode", get_trt_engine_path("avtr1_encode")),
                    ("avtr1_decode", get_trt_engine_path("avtr1_decode")),
                    ("hubert", find_engine_or_onnx("hubert_lbs", "hubert_onnx")),
                    ("decoder", find_engine_or_onnx("decoder", "decoder_onnx")),
                    ("stitch", find_engine_or_onnx("stitch_network", "stitch_network_onnx")),
                    ("modnet", find_engine_or_onnx("modnet", "modnet_onnx")),
                ]
                # Warp engine has a plugin — skip byte preload for it
                for name, path in engines_to_preload:
                    if not path.is_file():
                        continue
                    if path.suffix.lower() == ".onnx":
                        print(f"[snap] Skipping {name}: ONNX fallback ({path.name}), not TRT-serializable")
                        continue
                    data = path.read_bytes()
                    engine_bytes[name] = data
                    total_mb += len(data) / (1024 ** 2)
                self._engine_bytes = engine_bytes
                print(f"[snap] CPU preloaded engine bytes: {len(engine_bytes)} engines, {total_mb:.1f} MB")
            except Exception as exc:
                print(f"[snap] Engine bytes preload FAILED: {exc}")
                self._engine_bytes = None

        print("[snap] OmniVoice CPU load SKIPPED (deferred to background GPU load)")

        self._cpu_snapshot_ok = True

    def _load_cached_avatar(self, avatar_id: str) -> Avatar | None:
        """Load an Avatar from the precomputed cache on the L40S volume.

        Returns the Avatar object, or None if the cache is missing/invalid.
        Logs the cache hit/miss and the inferred no_matting value.
        """
        import torch
        import cv2
        from avtr1_renderer.avatar_loader import Avatar
        from avtr1_renderer.types import KPInfo

        av_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache" / avatar_id
        if not av_dir.is_dir():
            print(f"[cache] MISS: no cache dir for {avatar_id}")
            return None

        required_files = (
            "kp_info_kp.pt", "kp_info_exp.pt", "kp_info_scale.pt", "kp_info_t.pt",
            "kp_info_pitch.pt", "kp_info_yaw.pt", "kp_info_roll.pt", "kp_info_R.pt",
            "f_s.pt", "M_grid.pt", "mask.pt", "source.pt", "bg.pt", "normalizer.pt",
        )
        for name in required_files:
            if not (av_dir / name).is_file():
                print(f"[cache] MISS: {name} missing for {avatar_id}")
                return None

        print(f"[cache] HIT: loading {avatar_id} from {av_dir}")

        t0 = time.monotonic()
        device = torch.device("cuda")

        # Fast path: use CPU-preloaded tensors from snap=True if available
        cpu_tensors = getattr(self, "_cached_avatar_cpu_tensors", None)
        if cpu_tensors is not None:
            print(f"[cache] Using CPU-preloaded tensors from snap=True")
            kp_info = KPInfo(
                kp=cpu_tensors["kp_info_kp.pt"].float().to(device),
                exp=cpu_tensors["kp_info_exp.pt"].float().to(device),
                scale=cpu_tensors["kp_info_scale.pt"].float().to(device),
                t=cpu_tensors["kp_info_t.pt"].float().to(device),
                pitch=cpu_tensors["kp_info_pitch.pt"].float().to(device),
                yaw=cpu_tensors["kp_info_yaw.pt"].float().to(device),
                roll=cpu_tensors["kp_info_roll.pt"].float().to(device),
                R=cpu_tensors["kp_info_R.pt"].float().to(device),
            )
            f_s = cpu_tensors["f_s.pt"].float().to(device)
            M_grid = cpu_tensors["M_grid.pt"].float().to(device)
            mask = cpu_tensors["mask.pt"].float().to(device)
            source = cpu_tensors["source.pt"].float().to(device)
        else:
            # Fallback: read from disk
            kp_info = KPInfo(
                kp=torch.load(av_dir / "kp_info_kp.pt", map_location="cpu", weights_only=True).float().to(device),
                exp=torch.load(av_dir / "kp_info_exp.pt", map_location="cpu", weights_only=True).float().to(device),
                scale=torch.load(av_dir / "kp_info_scale.pt", map_location="cpu", weights_only=True).float().to(device),
                t=torch.load(av_dir / "kp_info_t.pt", map_location="cpu", weights_only=True).float().to(device),
                pitch=torch.load(av_dir / "kp_info_pitch.pt", map_location="cpu", weights_only=True).float().to(device),
                yaw=torch.load(av_dir / "kp_info_yaw.pt", map_location="cpu", weights_only=True).float().to(device),
                roll=torch.load(av_dir / "kp_info_roll.pt", map_location="cpu", weights_only=True).float().to(device),
                R=torch.load(av_dir / "kp_info_R.pt", map_location="cpu", weights_only=True).float().to(device),
            )
            f_s = torch.load(av_dir / "f_s.pt", map_location="cpu", weights_only=True).float().to(device)
            M_grid = torch.load(av_dir / "M_grid.pt", map_location="cpu", weights_only=True).float().to(device)
            mask = torch.load(av_dir / "mask.pt", map_location="cpu", weights_only=True).float().to(device)
            source = torch.load(av_dir / "source.pt", map_location="cpu", weights_only=True).float().to(device)

        # Load no_matting: prefer saved json, infer from portrait, fallback to False
        import json
        no_matting_json = av_dir / "no_matting.json"
        if no_matting_json.is_file():
            with open(no_matting_json) as f:
                no_matting = json.load(f).get("no_matting", False)
            no_matting_source = "json"
            print(f"[cache] Loaded no_matting={no_matting} from {no_matting_json}")
        else:
            portraits_dir = Path(L40S_MOUNT_PATH) / "main" / "avatars_artifacts" / "reference_frames"
            if not portraits_dir.is_dir():
                from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
                mgr = get_artifact_manager()
                portraits_dir = mgr.get_artifact_path("reference_frames")
            portrait = portraits_dir / f"{avatar_id}.png"
            if portrait.is_file():
                img = cv2.imread(str(portrait), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    no_matting = img.ndim == 2 or img.shape[2] == 3
                    no_matting_source = "portrait_inference"
                    print(f"[cache] Inferred no_matting={no_matting} from {portrait} (channels={img.shape[2] if img.ndim == 3 else 1})")
                else:
                    print(f"[cache] Could not read portrait, defaulting no_matting=False")
                    no_matting = False
                    no_matting_source = "default"
            else:
                print(f"[cache] Portrait not found, defaulting no_matting=False")
                no_matting = False
                no_matting_source = "default"

        avatar = Avatar(
            id=avatar_id,
            kp_info=kp_info,
            f_s=f_s,
            M_grid=M_grid,
            mask=mask,
            source=source,
            no_matting=no_matting,
            no_matting_source=no_matting_source,
        )
        load_s = time.monotonic() - t0
        print(f"[cache] Avatar loaded in {load_s:.3f}s, shapes: f_s={list(f_s.shape)}, source={list(source.shape)}, mask={list(mask.shape)}")
        return avatar

    @modal.enter(snap=False)
    def _post_restore(self):
        import time as _time

        t_restore_start = _time.monotonic()
        _build_ts = _time.strftime("%Y%m%d-%H%M%S", _time.gmtime())
        _build_git = "unknown"
        try:
            import subprocess as _sp
            _build_git = _sp.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=_sp.DEVNULL, timeout=3,
            ).decode().strip()[:12]
        except Exception:
            pass
        print(f"[build] WHIP_WHEP_PROOF_BUILD={_build_ts} git={_build_git} phase={OPT_PHASE_LABEL}", flush=True)
        self._build_marker = f"{_build_ts}_{_build_git}"
        print(f"[restore] Post-restore on GPU (phase={OPT_PHASE_LABEL})")

        try:
            import faulthandler as _fh
            _fh.enable(all_threads=True)
            import signal as _sig
            try:
                _fh.register(_sig.SIGTERM, chain=True)
            except (ValueError, OSError):
                pass
            print("[restore] faulthandler enabled (all_threads=True)", flush=True)
        except Exception as _fh_exc:
            print(f"[restore] faulthandler enable failed: {_fh_exc}", flush=True)

        import sys as _sys
        import threading as _threading_mod
        _orig_excepthook = _sys.excepthook
        def _process_excepthook(exc_type, exc_value, exc_tb):
            try:
                import json as _json
                _sessions_snapshot = list(getattr(self, "_sessions", {}).keys())
                print(f"[CRASH-FORENSICS] Uncaught exception: {exc_type.__name__ if exc_type else '?'} "
                      f"active_sessions={_sessions_snapshot}", flush=True)
                import traceback as _tb
                _tb.print_exception(exc_type, exc_value, exc_tb)
                _sys.stderr.flush()
            except Exception:
                pass
            _orig_excepthook(exc_type, exc_value, exc_tb)
        _sys.excepthook = _process_excepthook

        _orig_thread_excepthook = _threading_mod.excepthook
        def _process_thread_excepthook(args):
            try:
                _sessions_snapshot = list(getattr(self, "_sessions", {}).keys())
                print(f"[CRASH-FORENSICS] Uncaught thread exception: "
                      f"thread={args.thread.name if args.thread else '?'} "
                      f"exc={args.exc_type.__name__ if args.exc_type else '?'} "
                      f"active_sessions={_sessions_snapshot}", flush=True)
                import traceback as _tb
                _tb.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
                _sys.stderr.flush()
            except Exception:
                pass
        _threading_mod.excepthook = _process_thread_excepthook

        self._timings: dict[str, float] = {}
        self._engine_backends: dict[str, str] = {}

        def _timing_cb(step: str, elapsed_s: float):
            self._timings[step] = round(elapsed_s, 4)

        t0 = _time.monotonic()
        import torch
        _timing_cb("import_torch_s", _time.monotonic() - t0)

        t0 = _time.monotonic()
        import numpy as np
        _timing_cb("import_numpy_s", _time.monotonic() - t0)

        t0 = _time.monotonic()
        from avtr1_renderer.pipeline import Pipeline
        _timing_cb("import_pipeline_s", _time.monotonic() - t0)

        if not getattr(self, "_cpu_snapshot_ok", False):
            self._snap_error = getattr(self, "_snap_error", "CPU snapshot prep failed")
            print(f"[restore] ABORT: CPU snapshot prep was not ok: {self._snap_error}")
            self._sessions = {}
            t_restore_end = _time.monotonic()
            _timing_cb("total_post_restore_s", t_restore_end - t_restore_start)
            return

        t0 = _time.monotonic()
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"[restore] GPU: {torch.cuda.get_device_name(0)}")
        _timing_cb("cuda_probe_s", _time.monotonic() - t0)

        if DEBUG_STARTUP_DIAGNOSTICS:
            t0 = _time.monotonic()
            _run_nvidia_smi("gpu-init/start")
            _timing_cb("nvidia_smi_start_s", _time.monotonic() - t0)

        if not cuda_ok:
            self._snap_error = "CUDA not available in snap=False"
            print(f"[restore] ABORT: {self._snap_error}")
            self._sessions = {}
            t_restore_end = _time.monotonic()
            _timing_cb("total_post_restore_s", t_restore_end - t_restore_start)
            return

        try:
            storage_root = Path(L40S_MOUNT_PATH)

            print(f"[restore] Loading pipeline with avatar_id={self.avatar_id!r}...")

            bg_filter = None
            if OPT_SINGLE_BACKGROUND:
                bg_dir = storage_root / "main" / "avatars_artifacts" / "backgrounds"
                if not bg_dir.is_dir():
                    from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
                    mgr = get_artifact_manager()
                    bg_dir = mgr.get_artifact_path("backgrounds")
                all_bgs = sorted(p.stem for p in bg_dir.glob("*.png")) if bg_dir.is_dir() else []
                if all_bgs:
                    bg_filter = [all_bgs[0]]
                    print(f"[restore] background_filter={bg_filter}")

            t_pipeline_start = _time.monotonic()

            prebuilt_avatars = None
            if OPT_CACHED_AVATAR:
                t0 = _time.monotonic()
                cached_avatar = self._load_cached_avatar(self.avatar_id)
                _timing_cb("cached_avatar_load_s", _time.monotonic() - t0)
                if cached_avatar is not None:
                    prebuilt_avatars = {self.avatar_id: cached_avatar}
                    print(f"[restore] Cached avatar fast path: prebuilt_avatars={self.avatar_id}")
                else:
                    print(f"[restore] Cached avatar MISS, falling back to AvatarLoader")

            # Build prebuilt_engines dict from snap=True byte preload if available
            prebuilt_engines = None
            if OPT_ENGINE_BYTES and getattr(self, "_engine_bytes", None):
                try:
                    from avtr1_renderer.runtime import load_engine_from_bytes
                    from avtr1_renderer.models.hubert import HubertInput, HubertOutput
                    from avtr1_renderer.models.decoder import DecoderInput, DecoderOutput
                    from avtr1_renderer.models.stitch import StitchInput, StitchOutput
                    from avtr1_renderer.models.matting import MODNetInput, MODNetOutput
                    prebuilt_engines = {}
                    engine_bytes = self._engine_bytes
                    if "hubert" in engine_bytes:
                        prebuilt_engines["hubert"] = load_engine_from_bytes(
                            engine_bytes["hubert"], HubertInput, HubertOutput
                        )
                    if "decoder" in engine_bytes:
                        prebuilt_engines["decoder"] = load_engine_from_bytes(
                            engine_bytes["decoder"], DecoderInput, DecoderOutput
                        )
                    if "stitch" in engine_bytes:
                        prebuilt_engines["stitch"] = load_engine_from_bytes(
                            engine_bytes["stitch"], StitchInput, StitchOutput
                        )
                    if "modnet" in engine_bytes:
                        prebuilt_engines["modnet"] = load_engine_from_bytes(
                            engine_bytes["modnet"], MODNetInput, MODNetOutput
                        )
                    # The two engines we always preload as TRT bytes (avtr1_encode/decode)
                    # are not looked at by the ONNX-heavy block above. Deserialize them
                    # here so Pipeline.from_artifacts can skip the disk read.
                    if "avtr1_encode" in engine_bytes:
                        from avtr1_renderer.models.avtr1 import Avtr1EncodeInput, Avtr1EncodeOutput
                        prebuilt_engines["avtr1_encode"] = load_engine_from_bytes(
                            engine_bytes["avtr1_encode"], Avtr1EncodeInput, Avtr1EncodeOutput
                        )
                    if "avtr1_decode" in engine_bytes:
                        from avtr1_renderer.models.avtr1 import Avtr1DecodeInput, Avtr1DecodeOutput
                        prebuilt_engines["avtr1_decode"] = load_engine_from_bytes(
                            engine_bytes["avtr1_decode"], Avtr1DecodeInput, Avtr1DecodeOutput
                        )
                    print(f"[restore] Engine bytes fast path: {len(prebuilt_engines)} engines from snap preload")
                except Exception as exc:
                    print(f"[restore] Engine bytes fast path FAILED: {exc}")
                    prebuilt_engines = None

            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[self.avatar_id] if prebuilt_avatars is None else [],
                background_filter=bg_filter,
                timing_callback=_timing_cb,
                prebuilt_avatars=prebuilt_avatars,
                parallel_engine_loading=OPT_PARALLEL_ENGINES,
                prebuilt_engines=prebuilt_engines,
            )
            t_pipeline_end = _time.monotonic()
            self._timings["pipeline_from_artifacts_s"] = round(t_pipeline_end - t_pipeline_start, 4)
            self._timings["total_pipeline_from_artifacts"] = round(t_pipeline_end - t_pipeline_start, 4)
            self._engine_backends = dict(self.pipeline._engine_backends)

            # ── Fail-closed: all hot-path engines must be TRT ──────────────
            _REQUIRED_TRT = {
                "avtr1_encode", "avtr1_decode", "hubert_lbs",
                "decoder", "stitch_network", "warp_network", "modnet",
            }
            _onnx_leaked = {
                k: v for k, v in self._engine_backends.items()
                if k in _REQUIRED_TRT and v != "TRT"
            }
            _missing = _REQUIRED_TRT - set(self._engine_backends.keys())
            if _onnx_leaked or _missing:
                _detail = f"onnx_leaked={_onnx_leaked} missing={_missing}"
                print(f"[restore] FATAL: Not all hot-path engines are TRT: {_detail}", flush=True)
                raise RuntimeError(
                    f"Official-runtime pivot requires ALL hot-path engines as TRT. "
                    f"Build missing engines with scripts/build_engines.py. {_detail}"
                )
            print(f"[restore] All {len(_REQUIRED_TRT)} hot-path engines confirmed TRT", flush=True)

            print(f"[restore] Pipeline loaded")
            if DEBUG_STARTUP_DIAGNOSTICS:
                t0 = _time.monotonic()
                _run_nvidia_smi("gpu-init/after-pipeline-load")
                _timing_cb("nvidia_smi_after_pipeline_s", _time.monotonic() - t0)

            self._pipeline_loaded = True
            self.avatar = self.registry[self.avatar_id]
            self._cached_avatar_hit = prebuilt_avatars is not None
            self._engine_bytes_hit_count = len(prebuilt_engines) if prebuilt_engines else 0
            self._engine_bytes_total_mb = round(
                sum(len(v) for v in self._engine_bytes.values()) / (1024 ** 2), 1
            ) if getattr(self, "_engine_bytes", None) else 0.0
            self._engine_bytes_preload_enabled = OPT_ENGINE_BYTES

            mg = self.pipeline._motion_generator
            self.chunk_size = mg.chunk_size
            self.future_size = mg.future_size
            self.frame_len = mg.frame_len
            self.audio_shift = getattr(mg, 'audio_shift', 0)
            self.AUDIO_DIM = getattr(mg, 'AUDIO_DIM', 256)
            self.HUBERT_PAST_FRAMES = getattr(mg, 'HUBERT_PAST_FRAMES', 16)
            self.latent_dim = getattr(mg, 'latent_dim', 64)
            self.nfeats = getattr(mg, 'nfeats', 64)
            self.past_size = getattr(mg, 'past_size', 16)
            self.cond_feature_dim = self.AUDIO_DIM
            self._hubert_audio_len = (self.chunk_size + self.future_size) * self.frame_len + self.audio_shift

            self._bg_id = _pick_first_bg(self.pipeline._backgrounds)
            print(f"[restore] bg_id={self._bg_id!r}")

            self._sessions: dict[str, dict] = {}
            print(f"[restore] self._sessions initialized (container-local, max_containers=1)")

            self._omnivoice_status = "not_started"
            self._omnivoice_ready_event = threading.Event()
            self._omnivoice_lock = threading.Lock()
            self._omnivoice_warm_thread = None
            self._omnivoice_error = None
            self._omnivoice_warm_started_at = None
            self._omnivoice_warm_completed_at = None
            self._omnivoice_warm_trigger_point = None
            self._omnivoice_model = None
            self._omnivoice_sr = 24000
            self._omnivoice_ready = False
            print("[restore] OmniVoice state machine initialized: status=not_started")

            # --- Avatar cache (instance-level, initialized post-restore) ---
            self._avatar_cache_lock = threading.Lock()
            self._avatar_loading_in_progress: dict[str, threading.Event] = {}
            self._avatar_lru: OrderedDict[str, float] = OrderedDict()
            self._avatar_cache_stats: dict[str, dict] = {}
            self._avatar_eviction_count = 0

            # --- Background cache (instance-level, initialized post-restore) ---
            self._bg_cache_lock = threading.Lock()
            self._bg_lru: OrderedDict[str, float] = OrderedDict()
            self._bg_cache_stats: dict[str, dict] = {}
            print("[restore] Avatar/background cache state initialized")

            self._snap_ok = True

            t_restore_end = _time.monotonic()
            self._timings["total_post_restore_s"] = round(t_restore_end - t_restore_start, 4)
            print(f"[restore] Timings: {self._timings}")
            print(f"[restore] Engine backends: {self._engine_backends}")

        except Exception as exc:
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[restore] FAILED: {exc}")
            self._sessions = {}
            t_restore_end = _time.monotonic()
            self._timings["total_post_restore_s"] = round(t_restore_end - t_restore_start, 4)

    def _trigger_omnivoice_bg_load(self, trigger_point: str = "unknown"):
        with self._omnivoice_lock:
            if self._omnivoice_status != "not_started":
                return
            self._omnivoice_status = "warming"
            self._omnivoice_warm_started_at = time.monotonic()
            self._omnivoice_warm_trigger_point = trigger_point
        try:
            self._omnivoice_warm_thread = threading.Thread(
                target=self._omnivoice_bg_load_worker,
                daemon=True,
                name="omnivoice-bg-warm",
            )
            self._omnivoice_warm_thread.start()
            print(f"[omnivoice] background load triggered by {trigger_point}")
        except Exception as exc:
            with self._omnivoice_lock:
                self._omnivoice_status = "failed"
                self._omnivoice_error = str(exc)
                self._omnivoice_warm_completed_at = time.monotonic()
                self._omnivoice_ready = False
            self._omnivoice_ready_event.set()
            print(f"[omnivoice] thread start FAILED: {exc}")

    def _omnivoice_bg_load_worker(self):
        try:
            import torch
            print("[omnivoice-bg] starting fresh GPU load...")
            from omnivoice import OmniVoice as _OmniVoice
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = _OmniVoice.from_pretrained(
                OMNIVOICE_MODEL_ID,
                device_map="cuda:0",
                dtype=dtype,
                attn_implementation="eager",
            )
            sr = int(getattr(model, "sampling_rate", 24000))
            print(f"[omnivoice-bg] loaded on GPU, sr={sr}")

            print("[omnivoice-bg] running warmup inference...")
            with torch.inference_mode():
                model.generate(
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

            _run_nvidia_smi("omnivoice-bg/after-warmup")
            with self._omnivoice_lock:
                self._omnivoice_model = model
                self._omnivoice_sr = sr
                self._omnivoice_status = "ready"
                self._omnivoice_ready = True
                self._omnivoice_warm_completed_at = time.monotonic()
            self._omnivoice_ready_event.set()
            warm_dur = self._omnivoice_warm_completed_at - self._omnivoice_warm_started_at
            print(f"[omnivoice-bg] warmup OK, status=ready, dur={warm_dur:.1f}s")

        except Exception as exc:
            with self._omnivoice_lock:
                self._omnivoice_status = "failed"
                self._omnivoice_error = str(exc)
                self._omnivoice_warm_completed_at = time.monotonic()
                self._omnivoice_ready = False
            self._omnivoice_ready_event.set()
            import traceback
            traceback.print_exc()
            warm_dur = self._omnivoice_warm_completed_at - self._omnivoice_warm_started_at
            print(f"[omnivoice-bg] FAILED after {warm_dur:.1f}s: {exc}")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        engine_backends = getattr(self, "_engine_backends", {})
        hot_engines = ["avtr1_encode", "avtr1_decode", "hubert_lbs", "decoder", "warp_network", "stitch_network", "modnet"]
        audit_failures = []
        for name in hot_engines:
            backend = engine_backends.get(name, "unknown")
            if backend != "TRT":
                audit_failures.append(f"{name}={backend}")
        # Calculate engine bytes total if available
        engine_bytes_total_mb = 0.0
        if getattr(self, "_engine_bytes", None):
            engine_bytes_total_mb = sum(len(v) for v in self._engine_bytes.values()) / (1024 ** 2)

        # Avatar cache info
        loaded_avatar_ids = list(getattr(self, "registry", {}).keys())
        active_avatar_ids = list(set(s.get("avatar_id") for s in getattr(self, "_sessions", {}).values()))
        avatar_cache_stats = dict(getattr(self, "_avatar_cache_stats", {}))
        avatar_evictions = getattr(self, "_avatar_eviction_count", 0)

        # Background cache info
        _pipeline = getattr(self, "pipeline", None)
        _bg_dict = getattr(_pipeline, "_backgrounds", {}) if _pipeline is not None else {}
        bg_ids = list(_bg_dict.keys()) if isinstance(_bg_dict, dict) else []
        bg_cache_stats = dict(getattr(self, "_bg_cache_stats", {}))

        return {
            "ok": True,
            "phase": OPT_PHASE_LABEL,
            "skip_startup_warmup": SKIP_STARTUP_WARMUP,
            "opt_cached_avatar": OPT_CACHED_AVATAR,
            "opt_single_background": OPT_SINGLE_BACKGROUND,
            "opt_cpu_preload": OPT_CPU_PRELOAD,
            "opt_engine_bytes": OPT_ENGINE_BYTES,
            "opt_parallel_engines": OPT_PARALLEL_ENGINES,
            "cached_avatar_hit": getattr(self, "_cached_avatar_hit", False),
            "cpu_preload_available": getattr(self, "_cached_avatar_cpu_tensors", None) is not None,
            "engine_bytes_total_mb": round(engine_bytes_total_mb, 1),
            "chunk_size": getattr(self, "chunk_size", 0),
            "active_sessions": list(getattr(self, "_sessions", {}).keys()),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else 0,
            "timings": dict(getattr(self, "_timings", {})),
            "engine_backends": dict(engine_backends),
            "engine_audit_pass": len(audit_failures) == 0,
            "engine_audit_failures": audit_failures,
            "debug_startup_diagnostics": DEBUG_STARTUP_DIAGNOSTICS,
            "no_matting_source": getattr(getattr(self, "avatar", None), "no_matting_source", "unknown"),
            "avatar_cache": {
                "loaded_ids": loaded_avatar_ids,
                "active_ids": active_avatar_ids,
                "max_capacity": MAX_GPU_AVATARS,
                "stats": avatar_cache_stats,
                "eviction_count": avatar_evictions,
            },
            "background_cache": {
                "loaded_ids": bg_ids,
                "stats": bg_cache_stats,
                "max_capacity": MAX_BG_CACHE,
            },
        }

    @modal.method()
    def gst_runtime_probe(self) -> dict:
        """Phase 3B: Verify GStreamer/NVENC runtime is functional inside the container.

        Checks:
          1. import gi (PyGObject) succeeds
          2. gi.require_version for Gst, GstWebRTC, GstSdp
          3. from gi.repository import Gst, GstWebRTC, GstSdp, GLib
          4. Gst is initialized (Gst.init)
          5. nvh264enc element factory exists
          6. TRT coexistence patch (_dims_to_tuple) present in trt.py

        Returns a structured dict — never raises. Each check has a bool
        and an optional error string.
        """
        result: dict = {
            "ok": False,
            "gi_ok": False,
            "gst_ok": False,
            "nvh264enc_ok": False,
            "trt_patch_ok": False,
            "publisher_type": _resolve_publisher_type(),
            "errors": [],
        }

        # 1-3: gi + GStreamer imports
        # Use exec() for 'from gi.repository import' to trigger PyGObject's
        # module-level __getattr__ which dynamically loads typelibs.
        # importlib.import_module + attribute access doesn't trigger it.
        try:
            import importlib
            gi = importlib.import_module("gi")
            gi.require_version("Gst", "1.0")
            gi.require_version("GstWebRTC", "1.0")
            gi.require_version("GstSdp", "1.0")
            namespace = {}
            exec("from gi.repository import Gst, GstWebRTC, GstSdp, GLib", namespace)
            Gst = namespace["Gst"]
            result["gi_ok"] = True
        except Exception as exc:
            result["errors"].append(f"gi_import: {type(exc).__name__}: {exc}")
            return result

        # 4: Gst.init (required before using any GStreamer element)
        try:
            Gst.init(None)
            result["gst_ok"] = True
        except Exception as exc:
            result["errors"].append(f"gst_init: {type(exc).__name__}: {exc}")
            return result

        # 5: nvh264enc element factory
        try:
            factory = Gst.ElementFactory.find("nvh264enc")
            result["nvh264enc_ok"] = factory is not None
            if factory is None:
                result["errors"].append("nvh264enc: ElementFactory.find returned None")
        except Exception as exc:
            result["errors"].append(f"nvh264enc: {type(exc).__name__}: {exc}")

        # 6: TRT patch — _dims_to_tuple exists in trt.py
        try:
            from avtr1_renderer.runtime import trt as _trt_mod
            result["trt_patch_ok"] = hasattr(_trt_mod, "_dims_to_tuple")
            if not result["trt_patch_ok"]:
                result["errors"].append("trt_patch: _dims_to_tuple not found in avtr1_renderer.runtime.trt")
        except Exception as exc:
            result["errors"].append(f"trt_patch: {type(exc).__name__}: {exc}")

        result["ok"] = result["gi_ok"] and result["gst_ok"] and result["nvh264enc_ok"] and result["trt_patch_ok"]
        return result

    @modal.method()
    def probe_gst_element(self, element_name: str) -> dict:
        """Probe GStreamer element availability inside the Modal container.

        Runs gst-inspect-1.0 for the given element and returns the exact output.
        Used for Row F (whipsink probe) — never infers from other elements.
        """
        import subprocess
        try:
            r = subprocess.run(
                ["gst-inspect-1.0", element_name],
                capture_output=True, text=True, timeout=10,
            )
            return {
                "element": element_name,
                "available": r.returncode == 0,
                "returncode": r.returncode,
                "stdout": r.stdout[:3000],
                "stderr": r.stderr[:1000],
            }
        except Exception as exc:
            return {
                "element": element_name,
                "available": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

    def _get_avatar(self, avatar_id: str) -> object:
        if not getattr(self, "_pipeline_loaded", False):
            raise RuntimeError("Pipeline not loaded yet")

        # Fast path: already loaded
        if avatar_id in self.registry:
            with self._avatar_cache_lock:
                self._avatar_lru[avatar_id] = time.monotonic()
                self._avatar_cache_stats.setdefault(avatar_id, {"hits": 0, "misses": 0, "last_load_ms": 0})
                self._avatar_cache_stats[avatar_id]["hits"] += 1
            return self.registry[avatar_id]

        # Miss path: need to load dynamically
        t0 = time.monotonic()
        load_event: threading.Event | None = None
        with self._avatar_cache_lock:
            # Double-check after acquiring lock
            if avatar_id in self.registry:
                self._avatar_lru[avatar_id] = time.monotonic()
                self._avatar_cache_stats.setdefault(avatar_id, {"hits": 0, "misses": 0, "last_load_ms": 0})
                self._avatar_cache_stats[avatar_id]["hits"] += 1
                return self.registry[avatar_id]

            # Another thread is loading this avatar — wait for it
            if avatar_id in self._avatar_loading_in_progress:
                load_event = self._avatar_loading_in_progress[avatar_id]
                # Release lock and wait
                self._avatar_cache_lock.release()
                try:
                    load_event.wait(timeout=30.0)
                finally:
                    self._avatar_cache_lock.acquire()
                # After waiting, re-check
                if avatar_id in self.registry:
                    self._avatar_lru[avatar_id] = time.monotonic()
                    self._avatar_cache_stats.setdefault(avatar_id, {"hits": 0, "misses": 0, "last_load_ms": 0})
                    self._avatar_cache_stats[avatar_id]["hits"] += 1
                    return self.registry[avatar_id]
                else:
                    raise RuntimeError(f"Avatar {avatar_id} failed to load concurrently")

            # We are the loading thread
            load_event = threading.Event()
            self._avatar_loading_in_progress[avatar_id] = load_event

            # Check capacity before loading
            if len(self.registry) >= MAX_GPU_AVATARS:
                # Find LRU avatar not used by active sessions
                active_avatar_ids = set()
                for sess in self._sessions.values():
                    active_avatar_ids.add(sess.get("avatar_id"))

                evict_candidate = None
                for candidate_id in list(self._avatar_lru.keys()):
                    if candidate_id not in active_avatar_ids and candidate_id != self.avatar_id:
                        evict_candidate = candidate_id
                        break

                if evict_candidate is None:
                    del self._avatar_loading_in_progress[avatar_id]
                    load_event.set()
                    raise RuntimeError(
                        f"gpu_avatar_capacity_exceeded: max={MAX_GPU_AVATARS}, "
                        f"loaded={list(self.registry.keys())}, "
                        f"active={list(active_avatar_ids)}"
                    )

                self._evict_avatar(evict_candidate)

        # Load outside lock
        loaded_avatar = None
        try:
            loaded_avatar = self._load_cached_avatar(avatar_id)
            if loaded_avatar is None:
                raise ValueError(f"Avatar cache missing for {avatar_id}")
        except Exception:
            with self._avatar_cache_lock:
                if avatar_id in self._avatar_loading_in_progress:
                    del self._avatar_loading_in_progress[avatar_id]
                if load_event is not None:
                    load_event.set()
            raise

        # Register loaded avatar
        with self._avatar_cache_lock:
            self.registry[avatar_id] = loaded_avatar
            self._avatar_lru[avatar_id] = time.monotonic()
            self._avatar_cache_stats.setdefault(avatar_id, {"hits": 0, "misses": 0, "last_load_ms": 0})
            self._avatar_cache_stats[avatar_id]["misses"] += 1
            self._avatar_cache_stats[avatar_id]["last_load_ms"] = round((time.monotonic() - t0) * 1000, 1)
            if avatar_id in self._avatar_loading_in_progress:
                del self._avatar_loading_in_progress[avatar_id]
            if load_event is not None:
                load_event.set()

        print(f"[avatar_cache] LOADED {avatar_id} in {(time.monotonic()-t0)*1000:.1f}ms")
        return loaded_avatar

    def _evict_avatar(self, avatar_id: str) -> None:
        import torch
        vram_before = round(torch.cuda.memory_allocated() / 1024**2, 1)
        if avatar_id in self.registry:
            del self.registry[avatar_id]
        if avatar_id in self._avatar_lru:
            del self._avatar_lru[avatar_id]
        self._avatar_eviction_count += 1
        vram_after = round(torch.cuda.memory_allocated() / 1024**2, 1)
        print(f"[avatar_cache] EVICTED {avatar_id} vram_before={vram_before}MB vram_after={vram_after}MB")
        # Only empty cache if no active sessions (safe)
        if not self._sessions:
            torch.cuda.empty_cache()
            print("[avatar_cache] empty_cache called (no active sessions)")

    def _get_or_load_background(self, bg_id: str) -> tuple[object, str]:
        """Return (background_tensor, effective_bg_id), loading from disk if needed.

        Falls back to the default background if the requested one is missing.
        Inserts loaded tensors into ``self.pipeline._backgrounds`` so
        ``Pipeline.process_chunk()`` can look them up by id.
        """
        if bg_id in self.pipeline._backgrounds:
            with self._bg_cache_lock:
                self._bg_lru[bg_id] = time.monotonic()
            return self.pipeline._backgrounds[bg_id], bg_id

        # Try to load from disk
        t0 = time.monotonic()
        storage_root = Path(L40S_MOUNT_PATH)
        bg_dir = storage_root / "main" / "avatars_artifacts" / "backgrounds"
        if not bg_dir.is_dir():
            from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
            mgr = get_artifact_manager()
            bg_dir = mgr.get_artifact_path("backgrounds")

        bg_path = bg_dir / f"{bg_id}.png"
        if not bg_path.is_file():
            print(f"[bg_cache] MISSING {bg_id}, fallback to default={self._bg_id}")
            return self.pipeline._backgrounds.get(self._bg_id), self._bg_id

        from avtr1_renderer.backgrounds import load_background
        bg_tensor = load_background(bg_path, 720, 1280)

        with self._bg_cache_lock:
            # Evict if needed (skip transparent)
            non_transparent = [k for k in self.pipeline._backgrounds if k != "transparent"]
            if len(non_transparent) >= MAX_BG_CACHE:
                for candidate_id in list(self._bg_lru.keys()):
                    if candidate_id != "transparent" and candidate_id in self.pipeline._backgrounds:
                        del self.pipeline._backgrounds[candidate_id]
                        if candidate_id in self._bg_lru:
                            del self._bg_lru[candidate_id]
                        break
            self.pipeline._backgrounds[bg_id] = bg_tensor
            self._bg_lru[bg_id] = time.monotonic()
            self._bg_cache_stats.setdefault(bg_id, {"loads": 0})
            self._bg_cache_stats[bg_id]["loads"] += 1

        print(f"[bg_cache] LOADED {bg_id} in {(time.monotonic()-t0)*1000:.1f}ms")
        return bg_tensor, bg_id

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

    def _start_session_impl(self, session_id: str, avatar_id: str, bg_id: str | None = None) -> dict:
        import torch
        import numpy as np

        t0 = time.monotonic()

        # ── Capacity guard: prevent overlapping GPU sessions ─────────────
        # max_containers=1 means all sessions share one L40S. The Worker-side
        # SfuCapacityManager only guards webrtc_sfu sessions; fMP4 sessions
        # bypass it. This guard ensures at most ONE active render session
        # regardless of transport mode, preventing VRAM contention / CUDA
        # crashes from concurrent render threads on the same GPU.
        existing_sessions = {k: v for k, v in self._sessions.items()
                             if v.get("status") in ("ready", "running")}
        if existing_sessions:
            return {
                "session_id": session_id,
                "error": "capacity_full: another session is active on this container",
                "active_sessions": list(self._sessions.keys()),
            }

        if session_id in self._sessions:
            existing = self._sessions[session_id]
            if existing.get("status") == "running":
                return {
                    "session_id": session_id,
                    "error": f"session {session_id} has a live render thread; call end_session first or wait for it to exit",
                    "active_sessions": list(self._sessions.keys()),
                }

        # Resolve avatar with dynamic loading
        try:
            av = self._get_avatar(avatar_id)
        except Exception as e:
            return {
                "session_id": session_id,
                "error": str(e),
                "avatar_id": avatar_id,
                "active_sessions": list(self._sessions.keys()),
            }
        avatar_load_s = time.monotonic() - t0

        # Resolve background
        requested_bg_id = bg_id
        effective_bg_id = self._bg_id
        if requested_bg_id:
            try:
                _, effective_bg_id = self._get_or_load_background(requested_bg_id)
            except Exception as e:
                print(f"[session] bg load failed for {requested_bg_id}: {e}, fallback to {self._bg_id}")

        warmup_ms = 0.0
        if not self._full_render_warmed and not SKIP_STARTUP_WARMUP:
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
                
                from avtr1_renderer.types import RenderOptions
                _options = RenderOptions(
                    pixel_format="yuv_i420",
                    bg_id=effective_bg_id,
                )
                
                with torch.inference_mode():
                    state, frames_iter = self.pipeline.process_chunk(
                        av, chunk, state=None, options=_options,
                    )
                    frames = list(frames_iter)
                
                torch.cuda.synchronize()
                warmup_ms = (time.monotonic() - warmup_t0) * 1000
                if len(frames) == 0:
                    raise RuntimeError("warmup produced 0 frames — pipeline cannot render video")
                self._full_render_warmed = True
                print(f"[warmup] Complete in {warmup_ms:.0f}ms, {len(frames)} frames")
            except Exception as e:
                import os as _os, traceback as _tb
                print(f"[warmup] FAILED: {e}")
                _tb.print_exc()
                if _os.environ.get("AVTR_ALLOW_WARMUP_FAILURE", "") == "1":
                    print("[warmup] Continuing due to AVTR_ALLOW_WARMUP_FAILURE=1 (DEV ONLY)")
                    warmup_ms = -1
                else:
                    self._sessions.pop(session_id, None)
                    return {
                        "session_id": session_id,
                        "avatar_id": avatar_id,
                        "error": f"warmup_failed: {e}",
                        "warmup_ms": -1,
                    }

        import queue as _queue_mod
        self._sessions[session_id] = {
            "session_id": session_id,
            "avatar_id": avatar_id,
            "avatar": av,
            "bg_id": effective_bg_id,
            "requested_bg_id": requested_bg_id,
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
            "omnivoice_status": getattr(self, "_omnivoice_status", "unknown"),
            "omnivoice_ready": getattr(self, "_omnivoice_ready", False),
            "bg_id": effective_bg_id,
            "requested_bg_id": requested_bg_id,
            "loaded_avatar_ids": list(self.registry.keys()),
            "active_avatar_ids": list(set(s.get("avatar_id") for s in self._sessions.values())),
            "timings": dict(getattr(self, "_timings", {})),
            "engine_backends": dict(getattr(self, "_engine_backends", {})),
            "engine_bytes_enabled": getattr(self, "_engine_bytes_preload_enabled", False),
            "engine_bytes_hit": getattr(self, "_engine_bytes_hit_count", 0),
            "engine_bytes_total_mb": getattr(self, "_engine_bytes_total_mb", 0.0),
            "cached_avatar_hit": getattr(self, "_cached_avatar_hit", False),
            "scaledown_window": 30,
            "min_containers": 0,
            "opt_phase": OPT_PHASE_LABEL,
        }

    def _start_session_webrtc(self, body: dict) -> dict:
        """Start session with WebRTC SFU or direct WebRTC debug publisher mode.

        Uses a persistent background event loop (_PublisherLoop) that stays
        alive for the entire session lifetime, keeping the GStreamer webrtcbin
        active for RTP sending and ICE keepalives.

        SDP exchange is brokered through Worker/DO — Modal never holds CALLS_APP_SECRET.
        Modal reads MODAL_WORKER_SECRET from its environment (Modal secret store).

        direct_webrtc_debug variant:
        - The browser subscribes directly to Modal's webrtcbin (no Cloudflare Calls SFU).
        - The publisher-offer callback returns 202; the browser answer arrives later
          via the direct-webrtc-answer-5q endpoint.
        """
        session_id = body.get("session_id", "")
        avatar_id = body.get("avatar_id", "arnold")
        bg_id = body.get("bg_id")
        track_name = f"avatar-video-{session_id[-8:]}"
        publisher_mode = body.get("publisher_mode", "webrtc_sfu")
        publisher_callback_url = body.get("publisher_callback_url", "")
        close_callback_url = body.get("close_callback_url", "")
        answer_callback_url = body.get("answer_callback_url", "")
        ice_servers = body.get("ice_servers", []) or []
        matrix_mode = body.get("matrix_mode", "")
        matrix_row = body.get("matrix_row", "")
        direct_webrtc_debug = publisher_mode == "direct_webrtc_debug"
        pion_debug = publisher_mode == "pion_debug"
        whip_debug = publisher_mode == "whip_debug"

        if not whip_debug and (not publisher_callback_url or not close_callback_url):
            return {
                "session_id": session_id,
                "error": "publisher_callback_url and close_callback_url are required for WebRTC publisher mode",
                "active_sessions": list(self._sessions.keys()),
            }

        if (direct_webrtc_debug or pion_debug) and not answer_callback_url:
            return {
                "session_id": session_id,
                "error": "answer_callback_url is required for direct_webrtc_debug mode",
                "active_sessions": list(self._sessions.keys()),
            }

        # Step 1: Create session (loads avatar, inits pipeline)
        result = self._start_session_impl(session_id, avatar_id, bg_id)
        if "error" in result:
            return result

        sess = self._sessions.get(session_id)
        if not sess:
            return {"session_id": session_id, "error": "session creation failed"}
        avatar = sess.get("avatar")
        effective_bg_id = sess.get("bg_id", bg_id or "aloe_window")

        # Step 2: Create persistent publisher loop + select publisher type
        from _publisher_loop import _PublisherLoop

        publisher_type = _resolve_publisher_type()
        print(f"[webrtc] publisher_type={publisher_type} (AVTR_SFU_PUBLISHER={_os.environ.get('AVTR_SFU_PUBLISHER', '<unset>')})")

        pub_loop = _PublisherLoop()
        sess["publisher_loop"] = pub_loop

        if pion_debug:
            from pion_debug_publisher import PionDebugPublisher

            async def _start_publisher():
                _sfu_frame_capture = _maybe_create_capture("sfu", session_id)
                frame_source = GstAvtrFrameSource(
                    pipeline=self.pipeline,
                    avatar=avatar,
                    bg_id=effective_bg_id,
                    handoff=sess["handoff"],
                    frame_capture=_sfu_frame_capture,
                )
                publisher = PionDebugPublisher(
                    frame_source=frame_source,
                    track_name=track_name,
                    publisher_callback_url=publisher_callback_url,
                    close_callback_url=close_callback_url,
                )
                sess["webrtc_publisher"] = publisher
                sess["frame_source"] = frame_source
                sess["publisher_mode"] = "pion_debug"
                sess["publisher_track_name"] = track_name
                sess["publisher_type"] = "pion"
                sess["_sfu_frame_capture"] = _sfu_frame_capture
                video_mid = await publisher.start()
                return video_mid
        elif whip_debug:
            if matrix_mode:
                print(f"[matrix] mode={matrix_mode} build_marker={getattr(self, '_build_marker', 'unknown')}", flush=True)

            if matrix_mode == "matrix_render_null":
                _row_id = matrix_row or "baseline"

                async def _start_publisher():
                    frame_source = GstAvtrFrameSource(
                        pipeline=self.pipeline,
                        avatar=avatar,
                        bg_id=effective_bg_id,
                        handoff=sess["handoff"],
                        matrix_row=_row_id,
                    )
                    sess["frame_source"] = frame_source
                    sess["publisher_mode"] = "matrix_render_null"
                    sess["publisher_track_name"] = track_name

                    stop_event = sess.get("render_stop_event") or threading.Event()
                    sess["render_stop_event"] = stop_event

                    def _run_render_null(_fs=frame_source, _se=stop_event, _sid=session_id,
                                         _row=_row_id, _engb=getattr(self, "_engine_backends", {}),
                                         _build=getattr(self, "_build_marker", "unknown"),
                                         _cached=getattr(self, "_cached_avatar_hit", False)):
                        import resource as _resource
                        import torch as _torch

                        # ── Runtime fingerprint (one-time) ──
                        try:
                            import sys as _sys
                            import onnxruntime as _ort
                            import tensorrt as _trt
                            _fp = (
                                f"[matrix_fingerprint] row={_row} "
                                f"python={_sys.version.split()[0]} "
                                f"torch={_torch.__version__} "
                                f"cuda_runtime={_torch.version.cuda} "
                                f"ort={_ort.__version__} "
                                f"trt={_trt.__version__} "
                                f"gpu={_torch.cuda.get_device_name(0)} "
                                f"engines={_engb} "
                                f"cache_hit={_cached} "
                                f"build_marker={_build}"
                            )
                            print(_fp, flush=True)
                        except Exception as _e:
                            print(f"[matrix_fingerprint] row={_row} error={_e}", flush=True)

                        # nvidia-smi snapshot
                        try:
                            import subprocess as _sp
                            _r = _sp.run(["nvidia-smi", "--query-gpu=driver_version,memory.total",
                                          "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
                            print(f"[matrix_fingerprint] nvidia_smi={_r.stdout.strip()}", flush=True)
                        except Exception:
                            pass

                        _fs.start()
                        _dur = 90
                        _fps = 25
                        _fdur = 1.0 / _fps
                        _t0 = time.monotonic()
                        _deadline = _t0
                        _hb_last = _t0
                        _telem_last = _t0
                        _hb_last_gen = 0
                        _hb_last_uf = 0
                        _consumed = 0

                        while not _se.is_set():
                            _elapsed = time.monotonic() - _t0
                            if _elapsed >= _dur:
                                break
                            _err = _fs.get_error()
                            if _err is not None:
                                _total = _elapsed
                                print(
                                    f"AVTR_RENDER_MATRIX_RESULT row={_row} status=crashed "
                                    f"duration_ms={int(_total*1000)} "
                                    f"fps={_fs.frames_generated/max(_total,0.1):.1f} "
                                    f"frames={_fs.frames_generated} "
                                    f"error={type(_err).__name__}: {_err}",
                                    flush=True,
                                )
                                return
                            _prev_uf = _fs.underflow_count
                            _fs.next_frame()
                            _curr_uf = _fs.underflow_count
                            _consumed += 1
                            _now = time.monotonic()

                            # ── 5s telemetry ──
                            if _now - _telem_last >= 5.0:
                                _telem_dt = _now - _telem_last
                                _gd = _fs.frames_generated - _hb_last_gen
                                _alloc = _torch.cuda.memory_allocated() / (1024**2)
                                _reserved = _torch.cuda.memory_reserved() / (1024**2)
                                _max_alloc = _torch.cuda.max_memory_allocated() / (1024**2)
                                _rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
                                _threads = threading.active_count()
                                _cl = _fs.metrics.chunk_latencies_ms
                                _cl_sorted = sorted(_cl[-100:]) if _cl else [0]
                                _p50 = _cl_sorted[len(_cl_sorted)//2]
                                _p95 = _cl_sorted[int(len(_cl_sorted)*0.95)] if len(_cl_sorted) > 1 else _cl_sorted[0]
                                _cmax = _cl_sorted[-1]
                                print(
                                    f"[matrix_telemetry] row={_row} "
                                    f"fps={_gd/_telem_dt:.1f} "
                                    f"frames={_fs.frames_generated} "
                                    f"chunk_ms_p50={_p50:.1f} p95={_p95:.1f} max={_cmax:.1f} "
                                    f"alloc_mb={_alloc:.0f} reserved_mb={_reserved:.0f} "
                                    f"max_alloc_mb={_max_alloc:.0f} "
                                    f"rss_mb={_rss:.0f} threads={_threads}",
                                    flush=True,
                                )
                                _telem_last = _now

                            # ── 10s heartbeat ──
                            if _now - _hb_last >= 10.0:
                                _hb = _now - _hb_last
                                _gd = _fs.frames_generated - _hb_last_gen
                                _ud = _curr_uf - _hb_last_uf
                                print(f"[matrix_c] row={_row} phase=live gen_fps={_gd/_hb:.1f} "
                                      f"emit_fps={_consumed/(_now-_t0):.1f} "
                                      f"repeat_delta=0 underflow_delta={_ud}", flush=True)
                                _hb_last = _now
                                _hb_last_gen = _fs.frames_generated
                                _hb_last_uf = _curr_uf

                            _deadline += _fdur
                            _sleep = _deadline - time.monotonic()
                            if _sleep > 0:
                                time.sleep(_sleep)
                            else:
                                _deadline = time.monotonic()

                        _fs.stop()
                        _total = time.monotonic() - _t0
                        _status = "survived" if _total >= _dur - 1 else "stopped_early"
                        _err = _fs.get_error()
                        print(
                            f"AVTR_RENDER_MATRIX_RESULT row={_row} status={_status} "
                            f"duration_ms={int(_total*1000)} "
                            f"fps={_fs.frames_generated/max(_total,0.1):.1f} "
                            f"frames={_fs.frames_generated} "
                            f"error={type(_err).__name__ + ': ' + str(_err) if _err else 'none'}",
                            flush=True,
                        )

                    _t = threading.Thread(target=_run_render_null, daemon=True, name="matrix-render-null")
                    sess["_matrix_test_thread"] = _t
                    _t.start()
                    return "matrix-render-null"

            elif matrix_mode == "matrix_encode_avtr":
                from gst_webrtc_publisher import EncodeOnlyPublisher

                async def _start_publisher():
                    frame_source = GstAvtrFrameSource(
                        pipeline=self.pipeline,
                        avatar=avatar,
                        bg_id=effective_bg_id,
                        handoff=sess["handoff"],
                    )
                    publisher = EncodeOnlyPublisher(
                        frame_source=frame_source,
                        track_name=track_name,
                        encoder="nvh264enc",
                    )
                    sess["webrtc_publisher"] = publisher
                    sess["frame_source"] = frame_source
                    sess["publisher_mode"] = "matrix_encode_avtr"
                    sess["publisher_track_name"] = track_name
                    sess["publisher_type"] = "encode_only"
                    video_mid = await publisher.start()
                    print(f"[matrix_d] EncodeOnlyPublisher started with AVTR source", flush=True)
                    return video_mid

            elif matrix_mode == "matrix_encode_synth":
                from gst_webrtc_publisher import EncodeOnlyPublisher, SyntheticFrameSource

                async def _start_publisher():
                    frame_source = SyntheticFrameSource()
                    publisher = EncodeOnlyPublisher(
                        frame_source=frame_source,
                        track_name=track_name,
                        encoder="nvh264enc",
                    )
                    sess["webrtc_publisher"] = publisher
                    sess["frame_source"] = frame_source
                    sess["publisher_mode"] = "matrix_encode_synth"
                    sess["publisher_track_name"] = track_name
                    sess["publisher_type"] = "encode_only"
                    video_mid = await publisher.start()
                    print(f"[matrix_e] EncodeOnlyPublisher started with synthetic source", flush=True)
                    return video_mid

            else:
                from whip_publisher import WhipPublisher

                _matrix_synthetic = matrix_mode == "matrix_b"
                _matrix_encoder = "nvh264enc"

                async def _start_publisher():
                    _sfu_frame_capture = _maybe_create_capture("sfu", session_id)
                    if _matrix_synthetic:
                        from gst_webrtc_publisher import SyntheticFrameSource
                        frame_source = SyntheticFrameSource()
                        print("[matrix_b] Using SyntheticFrameSource (CPU-only, no GPU)", flush=True)
                    else:
                        frame_source = GstAvtrFrameSource(
                            pipeline=self.pipeline,
                            avatar=avatar,
                            bg_id=effective_bg_id,
                            handoff=sess["handoff"],
                            frame_capture=_sfu_frame_capture,
                        )
                    publisher = WhipPublisher(
                        frame_source=frame_source,
                        track_name=track_name,
                        ice_servers=ice_servers,
                        encoder=_matrix_encoder,
                    )
                    if not publisher._verify_env_visible():
                        raise RuntimeError("CF_STREAM_WHIP_URL not visible in Modal env")
                    print("[whip] CF_STREAM_WHIP_URL env verified (value not logged)", flush=True)
                    sess["webrtc_publisher"] = publisher
                    sess["frame_source"] = frame_source
                    sess["publisher_mode"] = "whip_debug"
                    sess["publisher_track_name"] = track_name
                    sess["publisher_type"] = "whip"
                    sess["_sfu_frame_capture"] = _sfu_frame_capture
                    video_mid = await publisher.start()
                    return video_mid
        elif publisher_type == "gstreamer":
            # Phase 3A: GStreamer/NVENC publisher (opt-in via AVTR_SFU_PUBLISHER=gstreamer)
            from gst_webrtc_publisher import (
                GstWebRtcPublisher,
                start_publisher_with_retry,
                PublisherIceFailedAfterRetries,
                _max_publisher_attempts,
            )

            async def _start_publisher():
                # Factory creates a fresh publisher per retry attempt.
                # Each attempt gets its own frame_source + pipeline + GLib loop.
                _sfu_frame_capture = _maybe_create_capture("sfu", session_id)

                def _factory():
                    frame_source = GstAvtrFrameSource(
                        pipeline=self.pipeline,
                        avatar=avatar,
                        bg_id=effective_bg_id,
                        handoff=sess["handoff"],
                        frame_capture=_sfu_frame_capture,
                    )
                    _h264_cap_dir = None
                    if _sfu_frame_capture is not None:
                        _h264_cap_dir = str(
                            Path(L40S_MOUNT_PATH) / "debug_captures" / session_id / "sfu_h264"
                        )
                    elif _os.environ.get("AVTR_H264_CAPTURE", "0") == "1":
                        _h264_cap_dir = str(
                            Path(L40S_MOUNT_PATH) / "debug_captures" / session_id / "sfu_h264"
                        )
                    publisher = GstWebRtcPublisher(
                        frame_source=frame_source,
                        track_name=track_name,
                        publisher_callback_url=publisher_callback_url,
                        close_callback_url=close_callback_url,
                        ice_servers=ice_servers,
                        h264_capture_dir=_h264_cap_dir,
                        direct_webrtc_debug=direct_webrtc_debug,
                        answer_callback_url=answer_callback_url if direct_webrtc_debug else None,
                    )
                    return publisher

                publisher, video_mid = await start_publisher_with_retry(_factory)
                # Store the SUCCESSFUL publisher + frame_source for session lifecycle.
                # The retry wrapper already cleaned up failed attempts.
                sess["webrtc_publisher"] = publisher
                sess["frame_source"] = publisher._source
                sess["publisher_mode"] = publisher_mode
                sess["publisher_track_name"] = track_name
                sess["publisher_type"] = "gstreamer"
                sess["_sfu_frame_capture"] = _sfu_frame_capture
                return video_mid
        else:
            # Default: aiortc WebRtcPublisher (unchanged behavior)
            from webrtc_publisher import AvtrFrameSource, WebRtcPublisher

            async def _start_publisher():
                frame_source = AvtrFrameSource(
                    pipeline=self.pipeline,
                    avatar=avatar,
                    bg_id=effective_bg_id,
                )
                await frame_source.start()

                publisher = WebRtcPublisher(
                    frame_source=frame_source,
                    track_name=track_name,
                    publisher_callback_url=publisher_callback_url,
                    close_callback_url=close_callback_url,
                )
                # Store BEFORE start() — same rationale as gstreamer branch.
                sess["webrtc_publisher"] = publisher
                sess["frame_source"] = frame_source
                sess["publisher_mode"] = "webrtc_sfu"
                sess["publisher_track_name"] = track_name
                sess["publisher_type"] = "aiortc"

                video_mid = await publisher.start()
                return video_mid

        try:
            video_mid = pub_loop.run(_start_publisher(), timeout=120)
        except Exception as e:
            print(f"[webrtc] publisher start failed: {e}")
            import traceback
            traceback.print_exc()

            # The retry wrapper cleans up failed publishers internally.
            # For PublisherIceFailedAfterRetries, extract attempt_summaries.
            # For other errors, a publisher may be in sess if it failed
            # after being stored (health check, etc.) — cleanup best-effort.
            attempt_summaries = getattr(e, "attempt_summaries", None)
            if attempt_summaries:
                ice_debug = attempt_summaries[-1].get("ice_debug") if attempt_summaries else None
            else:
                ice_debug = getattr(e, "ice_debug", None)

                # Best-effort publisher cleanup (stop() also records cleanup timing).
                publisher_for_cleanup = sess.get("webrtc_publisher")
                if publisher_for_cleanup is not None:
                    try:
                        pub_loop.run(publisher_for_cleanup.stop(), timeout=15)
                    except Exception as stop_exc:
                        print(f"[webrtc] publisher stop during error cleanup: {stop_exc}")

                    recorder = getattr(publisher_for_cleanup, "_recorder", None)
                    if recorder is not None:
                        ice_debug = recorder.snapshot()

            # Print final ice_debug (post-cleanup) for Modal log analysis
            if ice_debug:
                import json as _json_for_debug
                print(f"[ice-debug] {_json_for_debug.dumps(ice_debug, separators=(',', ':'))}", flush=True)

            # GUARANTEED cleanup — each step independently try/excepted so
            # a failure in one does NOT prevent the next from running.
            # This prevents stale sessions from blocking max_containers=1.
            try:
                fs = sess.get("frame_source")
                if fs is not None:
                    fs.stop()
                    print(f"[webrtc] frame_source stopped during cleanup", flush=True)
            except Exception as fs_exc:
                print(f"[webrtc] frame_source stop error: {fs_exc}")
            try:
                pub_loop.shutdown()
            except Exception as shutdown_exc:
                print(f"[webrtc] pub_loop.shutdown error: {shutdown_exc}")
            # Always pop — use pop (not del) so it never raises KeyError
            removed = self._sessions.pop(session_id, None)
            if removed is not None:
                print(f"[webrtc] session {session_id} removed from _sessions after publisher failure", flush=True)
            print(f"[webrtc] active_sessions after cleanup: {list(self._sessions.keys())}", flush=True)
            return {
                "session_id": session_id,
                "error": f"WebRTC publisher failed: {e}",
                "ice_debug": ice_debug,
                "active_sessions": list(self._sessions.keys()),
            }

        # Step 4: Strong health verification — loop alive, ICE connected, frames flowing
        publisher = sess.get("webrtc_publisher")
        frame_source = sess.get("frame_source")

        # Direct WebRTC debug: skip health check — start() returns immediately
        # after posting the offer (ICE connect happens in a background task).
        # The browser polls for the offer and sends the answer asynchronously.
        if direct_webrtc_debug or pion_debug or whip_debug:
            print(f"[webrtc] {publisher_mode}: skipping health check (async ICE handshake)")
            sess["publisher_loop"] = pub_loop
            result["publisher_mode"] = publisher_mode
            result["publisher_track_name"] = track_name
            result["publisher_type"] = "pion" if pion_debug else ("whip" if whip_debug else publisher_type)
            result["publisher_track_name"] = track_name
            return result

        healthy, health_detail = self._verify_publisher_healthy(pub_loop, publisher, frame_source)
        if not healthy:
            print(f"[webrtc] publisher health check FAILED — shutting down. detail={health_detail}")
            try:
                stop_metrics = pub_loop.run(publisher.stop(), timeout=15)
                print(f"[webrtc] publisher stopped after failed health check: {stop_metrics}")
            except Exception as e:
                stop_metrics = {"stop_error": str(e)[:200]}
                print(f"[webrtc] publisher stop error after failed health check: {e}")

            # ── Container poisoning on CUDA-class errors ──
            # If the health check failed due to a producer error (CUDA illegal
            # memory access, etc.) OR a degraded generation rate / excessive
            # underflow, the entire Python process's CUDA context is likely
            # corrupted or degraded and subsequent sessions in this warm
            # container will also fail.  Use Modal's official API to stop
            # fetching new inputs so Modal recycles the container.
            #
            # Stage 3B reliability repair: initially only CUDA error strings
            # triggered poisoning, but diagnostic data showed the most common
            # failure mode is "insufficient_generation_rate" — the producer
            # thread is alive but generating ~4fps instead of 25fps due to
            # partially-corrupted CUDA context (no explicit error string).
            # These degraded contexts are just as unusable as fully-crashed
            # ones and must also trigger poisoning.
            reason = health_detail.get("reason", "unknown")
            producer_error = health_detail.get("producer_error")
            container_poisoned = False
            poison_reasons = (
                "producer_thread_crashed",
                "producer_error_set",
                "producer_error_during_health_window",
                "producer_thread_dead",
                # Degraded-but-not-crashed producer: CUDA context partially
                # corrupted, inference too slow for real-time.  Container must
                # be recycled — subsequent sessions will also be degraded.
                "insufficient_generation_rate",
                "excessive_underflow_during_startup",
            )
            if producer_error or reason in poison_reasons:
                container_poisoned = self._poison_container(
                    source="publisher_health_check",
                    reason=reason,
                    detail=producer_error or reason,
                )

            pub_loop.shutdown()
            if session_id in self._sessions:
                del self._sessions[session_id]
            return {
                "session_id": session_id,
                "error": "WebRTC publisher health check failed",
                "health_check_detail": health_detail,
                "publisher_stop_metrics": stop_metrics,
                "container_poisoned": container_poisoned,
                "active_sessions": list(self._sessions.keys()),
            }

        # Step 5: Store loop handle in session
        sess["publisher_loop"] = pub_loop

        result["publisher_mode"] = "webrtc_sfu"
        result["publisher_track_name"] = track_name
        result["publisher_type"] = publisher_type

        # Include candidate summary for pass-vs-fail forensic comparison.
        # Safe fields only — no IPs, no full SDP.
        if hasattr(publisher, "get_candidate_summary"):
            result["ice_debug_success"] = publisher.get_candidate_summary()

        print(f"[webrtc] session started: {session_id} track={track_name} mid={video_mid} type={publisher_type}")
        return result

    def _verify_publisher_healthy(self, pub_loop, publisher, frame_source) -> tuple[bool, dict]:
        """Verify publisher loop alive, ICE connected, producer thread alive,
        no producer error, and a real minimum generation rate over ~1s.

        Returns (healthy, detail) where detail is a structured dict captured
        in logs + returned to the Worker via the session-start response so
        operators can see *why* a health check failed.

        Stage 3B reliability repair: the previous check only required
        `frames_generated > 0` after 0.5s, which let degraded producers
        (CUDA-poisoned, ~50% frame rate) pass and emit frozen video.
        """
        detail: dict = {}

        # ── 1. Loop alive ──
        if not pub_loop.is_alive():
            detail["reason"] = "publisher_loop_dead"
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']}")
            return False, detail

        # ── 2. ICE connected ──
        # aiortc publisher exposes publisher._pc.iceConnectionState.
        # GStreamer publisher exposes publisher._ice_connected (asyncio.Event).
        async def _check_ice():
            pc = getattr(publisher, "_pc", None)
            if pc is not None:
                return pc.iceConnectionState
            ice_event = getattr(publisher, "_ice_connected", None)
            if ice_event is not None and ice_event.is_set():
                return "connected"
            return "disconnected"

        try:
            ice_state = pub_loop.run(_check_ice(), timeout=5)
        except Exception as e:
            detail["reason"] = "ice_state_unreadable"
            detail["error"] = str(e)[:200]
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} — {detail['error']}")
            return False, detail

        detail["ice_state"] = ice_state
        if ice_state not in ("connected", "completed"):
            detail["reason"] = "ice_not_connected"
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} (state={ice_state})")
            return False, detail

        # ── 3. Producer thread alive ──
        prod_thread = getattr(frame_source, "_producer_thread", None)
        producer_alive = bool(prod_thread and prod_thread.is_alive())
        detail["producer_thread_alive"] = producer_alive
        if not producer_alive:
            err = frame_source.get_error() if hasattr(frame_source, "get_error") else frame_source._error
            detail["reason"] = "producer_thread_dead"
            if err is not None:
                detail["producer_error"] = f"{type(err).__name__}: {str(err)[:300]}"
                detail["reason"] = "producer_thread_crashed"
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} err={detail.get('producer_error')}")
            return False, detail

        # ── 4. No producer error already set (thread may still be alive briefly) ──
        err = frame_source.get_error() if hasattr(frame_source, "get_error") else frame_source._error
        if err is not None:
            detail["reason"] = "producer_error_set"
            detail["producer_error"] = f"{type(err).__name__}: {str(err)[:300]}"
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} — {detail['producer_error']}")
            return False, detail

        # ── 5. Minimum generation rate over ~1.2s ──
        # The old 0.5s/frames>0 gate let degraded producers (10-15fps) pass.
        # At 25fps target, a healthy producer generates >=20 frames in 1.2s.
        # We require >= MIN_FRAMES_IN_WINDOW to prove the pipeline is keeping
        # up, not just limping along with CUDA-throttled inference.
        import time
        MIN_FRAMES_IN_WINDOW = 20          # ~17fps minimum over 1.2s
        MAX_UNDERFLOW_IN_WINDOW = 5        # transient blips allowed; sustained = fail
        HEALTH_WINDOW_S = 1.2

        gen_before = frame_source.metrics.frames_generated
        underflow_before = frame_source.metrics.underflow_count
        time.sleep(HEALTH_WINDOW_S)
        gen_after = frame_source.metrics.frames_generated
        underflow_after = frame_source.metrics.underflow_count

        gen_delta = gen_after - gen_before
        underflow_delta = underflow_after - underflow_before

        detail["health_window_s"] = HEALTH_WINDOW_S
        detail["frames_generated_delta"] = gen_delta
        detail["underflow_delta"] = underflow_delta
        detail["frames_generated_total"] = gen_after

        # Re-check producer error after the sleep — a CUDA fault often surfaces here.
        err = frame_source.get_error() if hasattr(frame_source, "get_error") else frame_source._error
        if err is not None:
            detail["reason"] = "producer_error_during_health_window"
            detail["producer_error"] = f"{type(err).__name__}: {str(err)[:300]}"
            print(f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} — {detail['producer_error']}")
            return False, detail

        print(
            f"[webrtc] HEALTH CHECK: ice={ice_state} producer_alive={producer_alive} "
            f"gen_delta={gen_delta}/{MIN_FRAMES_IN_WINDOW} underflow_delta={underflow_delta} "
            f"gen_total={gen_after}"
        )

        if gen_delta < MIN_FRAMES_IN_WINDOW:
            detail["reason"] = "insufficient_generation_rate"
            detail["min_required"] = MIN_FRAMES_IN_WINDOW
            print(
                f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} "
                f"(got {gen_delta} in {HEALTH_WINDOW_S}s, need >= {MIN_FRAMES_IN_WINDOW})"
            )
            return False, detail

        if underflow_delta > MAX_UNDERFLOW_IN_WINDOW:
            detail["reason"] = "excessive_underflow_during_startup"
            detail["max_allowed"] = MAX_UNDERFLOW_IN_WINDOW
            print(
                f"[webrtc] HEALTH CHECK FAILED: {detail['reason']} "
                f"(got {underflow_delta} underflows in {HEALTH_WINDOW_S}s, "
                f"max allowed {MAX_UNDERFLOW_IN_WINDOW})"
            )
            return False, detail

        detail["reason"] = "healthy"
        return True, detail

    def _poison_container(self, *, source: str, reason: str, detail: str | None = None) -> bool:
        """Mark this Modal container as unworthy of new inputs.

        Once a CUDA illegal memory access (or comparable unrecoverable GPU
        fault) has occurred inside a Python process, the CUDA context is
        corrupted for the process's entire remaining lifetime.  Any
        subsequent session that Modal routes to this warm container will
        also fail.  Modal's official supported remedy is
        ``modal.experimental.stop_fetching_inputs()``: the container stops
        accepting new inputs, drains active inputs, and Modal's autoscaler
        brings up a fresh container for the next request.

        Returns True if the poisoning succeeded, False if the Modal API was
        unavailable (logged but non-fatal — the caller still fails the
        current session, the container just may be reused once more before
        timing out via scaledown_window).
        """
        try:
            import modal.experimental  # Modal-supported public API (1.5.0+)
            modal.experimental.stop_fetching_inputs()
            print(
                f"[webrtc] CONTAINER POISONED via modal.experimental.stop_fetching_inputs() "
                f"source={source} reason={reason} detail={detail or '-'}"
            )
            return True
        except ImportError:
            print(
                f"[webrtc] CONTAINER POISONING FAILED: modal.experimental not available "
                f"(source={source} reason={reason}). Container may be reused — "
                f"upstream should treat next session start as best-effort."
            )
            return False
        except Exception as exc:
            print(
                f"[webrtc] CONTAINER POISONING FAILED: {type(exc).__name__}: {exc} "
                f"(source={source} reason={reason})"
            )
            return False

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

        # ── Phase D1: WebRTC publisher cleanup (persistent loop) ──
        # Stage 3B reliability repair: capture producer_error / fail_fast_reason
        # from the publisher stop metrics and poison the container if a CUDA-
        # class fatal error occurred.  Otherwise the next session routed to
        # this warm container will inherit the corrupted CUDA context.
        pub_loop = sess.get("publisher_loop")
        publisher = sess.get("webrtc_publisher")
        publisher_stop_metrics: dict | None = None
        container_poisoned = False
        poisoning_reason: str | None = None
        if pub_loop is not None and publisher is not None:
            try:
                publisher_stop_metrics = pub_loop.run(publisher.stop(), timeout=15)
                print(f"[webrtc] publisher stopped: {publisher_stop_metrics}")
                sess["webrtc_publisher_metrics"] = publisher_stop_metrics
            except Exception as e:
                publisher_stop_metrics = {"stop_error": str(e)[:300]}
                print(f"[webrtc] publisher stop error: {e}")

            # Detect CUDA-class fatal errors surfaced via the stop metrics.
            if isinstance(publisher_stop_metrics, dict):
                producer_error = publisher_stop_metrics.get("producer_error") or ""
                fail_fast_reason = publisher_stop_metrics.get("fail_fast_reason") or ""
                combined = f"{producer_error} {fail_fast_reason}".lower()
                # CUDA illegal memory access, CUDA out-of-memory, and other
                # torch.AcceleratorError patterns all poison the context.
                cuda_class_patterns = (
                    "cuda error",
                    "illegal memory access",
                    "torch.acceleratorerror",
                    "out of memory",
                    "cuda-launch-blocking",  # in case the debug build surfaces this
                )
                if any(p in combined for p in cuda_class_patterns):
                    poisoning_reason = (
                        f"end_session_producer_error: {producer_error or fail_fast_reason}"
                    )
                    container_poisoned = self._poison_container(
                        source="end_session_publisher_stop_metrics",
                        reason="cuda_class_producer_error",
                        detail=poisoning_reason,
                    )
                elif publisher_stop_metrics.get("track_fail_fast_triggered"):
                    # Track closed due to fail-fast policy but no explicit CUDA
                    # string matched — still poison to be safe; the producer
                    # thread is dead in any case.
                    poisoning_reason = (
                        f"end_session_track_fail_fast: {fail_fast_reason or 'unspecified'}"
                    )
                    container_poisoned = self._poison_container(
                        source="end_session_track_fail_fast",
                        reason="track_fail_fast_no_cuda_string",
                        detail=poisoning_reason,
                    )

            # ── Post-session CUDA context validation ──
            # Even if the publisher stop metrics look clean (no explicit CUDA
            # error string, producer thread was alive), the CUDA context may
            # have been silently corrupted during cleanup (producer thread
            # teardown, PC close, tensor deallocation).  This manifests as
            # the NEXT session in this warm container getting a degraded
            # producer (~4fps instead of 25fps) with no error string.
            #
            # torch.cuda.synchronize() is the definitive single-call test:
            # if the context is corrupted, it raises RuntimeError with the
            # "illegal memory access" message.  If it passes, the context
            # is usable for the next session.
            try:
                import torch
                torch.cuda.synchronize()
                torch.cuda.empty_cache()  # release cached allocator blocks
            except Exception as cuda_exc:
                err_msg = f"{type(cuda_exc).__name__}: {str(cuda_exc)[:300]}"
                print(f"[webrtc] POST-SESSION CUDA CHECK FAILED: {err_msg}")
                if not container_poisoned:  # don't double-poison
                    poisoning_reason = f"post_session_cuda_corrupted: {err_msg}"
                    container_poisoned = self._poison_container(
                        source="post_session_cuda_check",
                        reason="cuda_context_corrupted",
                        detail=poisoning_reason,
                    )
            # ── Always-poison after WebRTC SFU sessions ──
            # Diagnostic data from Stage 3B validation shows the CUDA context
            # degrades across sessions even when the post-session synchronize()
            # check passes: the corruption is latent and only surfaces on the
            # next session's first heavy inference.  Rather than trying to
            # detect latent corruption (impossible without running a full
            # inference cycle), we always recycle the container after SFU
            # sessions.  Cost: ~15s cold-start per session.  Benefit: zero
            # cross-session CUDA contamination.
            #
            # This can be disabled for development by setting the env var
            # AVTR_SFU_CONTAINER_REUSE=1 (keeps the container warm between
            # sessions for faster iteration, at the risk of CUDA degradation).
            if not container_poisoned and _os.environ.get("AVTR_SFU_CONTAINER_REUSE", "0") != "1":
                poisoning_reason = "post_sfu_session_always_recycle"
                container_poisoned = self._poison_container(
                    source="post_sfu_session_always_recycle",
                    reason="preventive_container_recycle_after_sfu",
                    detail=poisoning_reason,
                )

        if pub_loop is not None:
            pub_loop.shutdown()
            sess.pop("publisher_loop", None)

        frame_source = sess.get("frame_source")
        if frame_source is not None:
            frame_source.stop()

        _sfu_cap = sess.pop("_sfu_frame_capture", None)
        if _sfu_cap is not None:
            if not _sfu_cap._closed:
                _sfu_cap.finish()
            _sfu_cap.wait(timeout=30)

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
            # Stage 3B reliability repair: surface publisher stop metrics +
            # container poisoning outcome to the Worker so it can fail
            # closed and emit telemetry on CUDA-class failures.
            "publisher_stop_metrics": publisher_stop_metrics,
            "container_poisoned": container_poisoned,
            "poisoning_reason": poisoning_reason,
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

    @modal.fastapi_endpoint(method="POST",     label="start-session-5q")
    async def http_start_session(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
        publisher_mode = body.get("publisher_mode")

        if publisher_mode in ("webrtc_sfu", "direct_webrtc_debug", "pion_debug", "whip_debug"):
            return self._start_session_webrtc(body)

        return self._start_session_impl(
            session_id=body.get("session_id", ""),
            avatar_id=body.get("avatar_id", "arnold"),
            bg_id=body.get("bg_id"),
        )

    @modal.fastapi_endpoint(method="POST",     label="direct-webrtc-answer-5q")
    async def http_direct_webrtc_answer(self, request: Request) -> dict:
        """Receive the browser's answer SDP and apply it to the waiting publisher.

        Used only by direct_webrtc_debug transport. The Worker signs this request
        with the same HMAC scheme used for all Worker -> Modal internal endpoints.
        """
        await verify_modal_request(request)
        body = json.loads(await request.body())
        session_id = body.get("session_id", "")
        answer_sdp = body.get("answer_sdp", "")
        answer_type = body.get("answer_type", "answer")

        if not session_id or not answer_sdp:
            return {"status": "error", "error": "session_id and answer_sdp required"}

        sess = self._sessions.get(session_id)
        if not sess:
            return {"status": "error", "error": "session_not_found"}

        publisher = sess.get("webrtc_publisher")
        pub_loop = sess.get("publisher_loop")
        if not publisher or not pub_loop:
            return {"status": "error", "error": "publisher_not_ready"}

        try:
            pub_loop.run(publisher.set_remote_answer(answer_sdp, answer_type), timeout=15)
            return {"status": "ok"}
        except Exception as e:
            print(f"[direct-webrtc-answer] failed for {session_id}: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)[:200]}

    @modal.fastapi_endpoint(method="POST",     label="session-status-5q")
    async def http_session_status(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
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
            "omnivoice_status": getattr(self, "_omnivoice_status", "unknown"),
            "omnivoice_ready": getattr(self, "_omnivoice_ready", False),
            "omnivoice_warm_started_at": getattr(self, "_omnivoice_warm_started_at", None),
            "omnivoice_warm_completed_at": getattr(self, "_omnivoice_warm_completed_at", None),
            "omnivoice_warm_error": getattr(self, "_omnivoice_error", None),
            "omnivoice_warm_trigger_point": getattr(self, "_omnivoice_warm_trigger_point", None),
            "bg_id": sess.get("bg_id"),
            "requested_bg_id": sess.get("requested_bg_id"),
            "loaded_avatar_ids": list(self.registry.keys()),
            "active_avatar_ids": list(set(s.get("avatar_id") for s in self._sessions.values())),
        }

    @modal.fastapi_endpoint(method="POST",     label="push-audio-5q")
    async def http_push_audio(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
        return self._do_push_audio(body)

    def _do_end_session_adapter(self, session_id: str) -> dict:
        if session_id not in self._sessions:
            return {"session_id": session_id, "status": "not_active",
                    "active_sessions": list(self._sessions.keys())}

        sess = self._sessions[session_id]

        # ── Stage 3B reliability repair: WebRTC SFU sessions must go through
        # the FULL _end_session_impl, which stops the publisher, runs the
        # post-session CUDA check, and poisons the container.  The lightweight
        # adapter path below was designed for fMP4 SSE and skips all of that,
        # leaving the WebRTC publisher running and the CUDA context degraded
        # for the next session in this warm container.
        publisher_mode = sess.get("publisher_mode")
        has_webrtc_publisher = sess.get("webrtc_publisher") is not None
        if publisher_mode in ("webrtc_sfu", "direct_webrtc_debug", "pion_debug", "whip_debug",
                              "matrix_render_null", "matrix_encode_avtr", "matrix_encode_synth") or has_webrtc_publisher:
            print(f"[end-adapter] delegating to _end_session_impl for WebRTC publisher session {session_id} mode={publisher_mode}")
            return self._end_session_impl(session_id)

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

    @modal.fastapi_endpoint(method="POST",     label="end-session-adapter-5q")
    async def http_end_session_adapter(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
        return self._do_end_session_adapter(body.get("session_id", ""))

    @modal.fastapi_endpoint(method="POST",     label="stream-fmp4-av-adapter-5q")
    async def http_stream_fmp4_av_adapter(self, request: Request):
        await verify_modal_request(request)
        from fastapi.responses import StreamingResponse

        body = json.loads(await request.body())
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

    @modal.fastapi_endpoint(method="POST",     label="push-audio-batch-5q")
    async def http_push_audio_batch(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
        return self._do_push_audio_batch(body)

    @modal.fastapi_endpoint(method="POST",     label="tts-same-gpu-parallel-5q")
    async def http_tts_same_gpu_parallel(self, request: Request) -> dict:
        await verify_modal_request(request)
        body = json.loads(await request.body())
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

        with self._omnivoice_lock:
            ov_status = self._omnivoice_status
            ov_model = self._omnivoice_model
            ov_error = self._omnivoice_error

        if ov_status == "ready":
            if ov_model is None:
                return {"ok": False, "error": "omnivoice_model_none", "omnivoice_status": "failed"}
        elif ov_status == "warming":
            print("[tts_parallel] OmniVoice warming, waiting up to 3s...")
            wait_start = _time.monotonic()
            self._omnivoice_ready_event.wait(timeout=3.0)
            wait_dur_ms = (_time.monotonic() - wait_start) * 1000
            with self._omnivoice_lock:
                ov_status = self._omnivoice_status
            print(f"[tts_parallel] wait completed: status={ov_status}, waitMs={wait_dur_ms:.0f}")
            if ov_status != "ready":
                return {"ok": False, "error": "omnivoice_not_ready_after_wait", "omnivoice_status": ov_status}
        elif ov_status == "failed":
            return {"ok": False, "error": str(ov_error or "omnivoice_failed"), "omnivoice_status": "failed"}
        else:
            self._trigger_omnivoice_bg_load(trigger_point="tts_lazy")
            return {"ok": False, "error": "omnivoice_not_started", "omnivoice_status": "not_started"}

        with self._omnivoice_lock:
            ov_model = self._omnivoice_model
            ov_sr = self._omnivoice_sr
            ov_status = self._omnivoice_status
        if ov_status != "ready" or ov_model is None:
            return {"ok": False, "error": "omnivoice_not_ready_final", "omnivoice_status": ov_status}

        if enqueue_to_timeline and (not session_id or turn_id is None):
            return {"ok": False, "error": "enqueueToTimeline requires session_id and turnId"}

        t_generate_start = _time.time()
        with torch.inference_mode():
            audio = ov_model.generate(
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

        audio_dur_s = audio_data.shape[0] / ov_sr
        audio_dur_ms = audio_dur_s * 1000

        t_resample_start = _time.time()
        enqueue_sr = 16000 if enqueue_to_timeline else target_sr
        if ov_sr != enqueue_sr:
            import torchaudio.transforms as T
            resampler = T.Resample(
                ov_sr, enqueue_sr, dtype=audio_data.dtype
            )
            audio_data = resampler(audio_data)
            actual_sr = enqueue_sr
        else:
            actual_sr = ov_sr
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

        effective_bg_id = sess.get("bg_id", self._bg_id)
        render_options = RenderOptions(
            pixel_format="yuv_i420",
            bg_id=effective_bg_id,
        )

        _adapter_frame_capture = _maybe_create_capture("fmp4", session_id)

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
        _ffmpeg_t0 = time.monotonic() if _MEDIA_TIMING_DEBUG else 0.0
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if _MEDIA_TIMING_DEBUG:
            _ffmpeg_startup_ms = round((time.monotonic() - _ffmpeg_t0) * 1000, 1)
            print(f"[adapter_fmp4_av] FFmpeg started pid={ffmpeg_proc.pid} (startup={_ffmpeg_startup_ms}ms)")
        else:
            _ffmpeg_startup_ms = 0.0
            print(f"[adapter_fmp4_av] FFmpeg started pid={ffmpeg_proc.pid}")

        # Timing collectors (only populated when _MEDIA_TIMING_DEBUG)
        _timing_render_ms: list[float] = []
        _timing_encode_ms: list[float] = []
        _timing_segment_ms: list[float] = []
        _timing_write_queue_full = 0

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
                            if _MEDIA_TIMING_DEBUG:
                                _timing_segment_ms.append(time.monotonic() * 1000)
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
        render_cadence_samples = []

        first_media_emitted_mono = None
        omnivoice_stream_triggered = False
        OMNIVOICE_DEFER_MIN_SEGMENTS = 10
        OMNIVOICE_DEFER_MIN_MEDIA_S = 2.0

        def _maybe_trigger_omnivoice():
            nonlocal first_media_emitted_mono, omnivoice_stream_triggered
            if omnivoice_stream_triggered:
                return
            if first_media_emitted_mono is None:
                first_media_emitted_mono = time.monotonic()
            if (media_sequence >= OMNIVOICE_DEFER_MIN_SEGMENTS
                    and time.monotonic() - first_media_emitted_mono >= OMNIVOICE_DEFER_MIN_MEDIA_S):
                omnivoice_stream_triggered = True
                self._trigger_omnivoice_bg_load(trigger_point="stable_video_delivery")

        TARGET_LEAD_S = float(os.environ.get("AVTR1_TARGET_LEAD_S", "1.0"))
        CHUNK_MEDIA_S = 5 / fps
        HANDOFF_WAIT_S = 0.05
        MAX_BURST_SEGMENTS = 10

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
                self._burst_count = 0

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
                        self._burst_count += 1
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
                        if self.produced_media_s - wall_after >= TARGET_LEAD_S or self._burst_count >= MAX_BURST_SEGMENTS:
                            self.target_lead_reached = True
                            if self._burst_count >= MAX_BURST_SEGMENTS:
                                print(f"[adapter_sender] burst cap hit at {self._burst_count} segments, lead={self.produced_media_s - wall_after:.2f}s")
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
            nonlocal total_raw_bytes_written, _timing_write_queue_full
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
                if _MEDIA_TIMING_DEBUG:
                    _timing_write_queue_full += 1
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

            render_cadence_samples.append(render_ms)
            if len(render_cadence_samples) % 50 == 0:
                last50 = render_cadence_samples[-50:]
                avg_ms = sum(last50) / len(last50)
                max_ms = max(last50)
                min_ms = min(last50)
                ov_status = getattr(self, "_omnivoice_status", "unknown")
                print(f"[cadence] n={len(render_cadence_samples)} avg={avg_ms:.1f}ms min={min_ms:.1f}ms max={max_ms:.1f}ms omnivoice={ov_status}")

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
                        if _MEDIA_TIMING_DEBUG:
                            _timing_render_ms.append(render_ms)

                        encode_t0 = time.monotonic()
                        if not _write_video_frames(frames_list, f"speech {handoff_chunk_index}"):
                            adapter_stop_requested = True
                            break
                        encode_ms = (time.monotonic() - encode_t0) * 1000
                        if _MEDIA_TIMING_DEBUG:
                            _timing_encode_ms.append(encode_ms)

                        speech_first_segment = not handoff_first_segment_emitted

                        _pre_ms = media_sequence
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
                        if media_sequence > _pre_ms:
                            _maybe_trigger_omnivoice()

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

                            _pre_ms = media_sequence
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
                            if media_sequence > _pre_ms:
                                _maybe_trigger_omnivoice()

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

                if _adapter_frame_capture:
                    for f in frames_list:
                        _adapter_frame_capture.capture(f.data)
                    if not _adapter_frame_capture.active and not _adapter_frame_capture._closed:
                        _adapter_frame_capture.finish()

                encode_t0 = time.monotonic()
                if not _write_video_frames(frames_list, f"idle {cumulative_chunk_index}"):
                    adapter_stop_requested = True
                    break
                encode_ms = (time.monotonic() - encode_t0) * 1000

                _pre_ms = media_sequence
                yield from _yield_rendered(
                    "idle", None,
                    is_speech_start=False,
                    is_speech_end=False,
                    enqueue_at_val=None,
                    speech_first_segment=False,
                    speech_end_segment=False,
                )
                if media_sequence > _pre_ms:
                    _maybe_trigger_omnivoice()

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
                    "maxBurstSegments": MAX_BURST_SEGMENTS,
                    "actualBurstCount": sender._burst_count,
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

            # Media timing summary (only when AVTR_MEDIA_TIMING_DEBUG=1)
            if _MEDIA_TIMING_DEBUG:
                def _pct(lst, p):
                    if not lst:
                        return 0.0
                    s = sorted(lst)
                    return round(s[min(len(s) - 1, int(len(s) * p))], 2)

                _seg_intervals = []
                for i in range(1, len(_timing_segment_ms)):
                    _seg_intervals.append(round(_timing_segment_ms[i] - _timing_segment_ms[i - 1], 1))

                end_event["mediaTiming"] = {
                    "ffmpegStartupMs": _ffmpeg_startup_ms,
                    "renderMs": {
                        "count": len(_timing_render_ms),
                        "p50": _pct(_timing_render_ms, 0.50),
                        "p95": _pct(_timing_render_ms, 0.95),
                        "max": _pct(_timing_render_ms, 1.0),
                    },
                    "encodeMs": {  # time to put raw frames into write queue
                        "count": len(_timing_encode_ms),
                        "p50": _pct(_timing_encode_ms, 0.50),
                        "p95": _pct(_timing_encode_ms, 0.95),
                        "max": _pct(_timing_encode_ms, 1.0),
                    },
                    "segmentCadenceMs": {
                        "count": len(_seg_intervals),
                        "p50": _pct(_seg_intervals, 0.50),
                        "p95": _pct(_seg_intervals, 0.95),
                        "max": _pct(_seg_intervals, 1.0),
                    },
                    "writeQueueFullCount": _timing_write_queue_full,
                }
                print(f"[adapter_fmp4_av] MEDIA TIMING: "
                      f"render p50={_pct(_timing_render_ms, 0.50)} p95={_pct(_timing_render_ms, 0.95)} "
                      f"encode p50={_pct(_timing_encode_ms, 0.50)} "
                      f"segCadence p50={_pct(_seg_intervals, 0.50)} "
                      f"writeQueueFull={_timing_write_queue_full}")

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

            if _adapter_frame_capture:
                if not _adapter_frame_capture._closed:
                    _adapter_frame_capture.finish()
                _adapter_frame_capture.wait(timeout=30)

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
        _bg_id = sess.get("bg_id", self._bg_id)
        _render_options = RenderOptions(pixel_format="yuv_i420", bg_id=_bg_id)

        _fmp4_frame_capture = _maybe_create_capture("fmp4", session_id)

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

                if _fmp4_frame_capture:
                    for f in frames_list:
                        _fmp4_frame_capture.capture(f.data)
                    if not _fmp4_frame_capture.active and not _fmp4_frame_capture._closed:
                        _fmp4_frame_capture.finish()

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

            if _fmp4_frame_capture and not _fmp4_frame_capture._closed:
                _fmp4_frame_capture.finish()
            if _fmp4_frame_capture:
                _fmp4_frame_capture.wait(timeout=30)

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

    @modal.method()
    def start_session_admin(self, session_id: str = "", avatar_id: str = "arnold",
                            bg_id: str = "") -> dict:
        """Admin-only RPC session start — bypasses HTTP signing.

        Used for validation without the Worker in the loop.
        """
        if not session_id:
            import time as _t
            session_id = f"admin-{int(_t.time())}"
        _bg = bg_id or None
        return self._start_session_impl(
            session_id=session_id,
            avatar_id=avatar_id,
            bg_id=_bg,
        )

    @modal.method()
    def end_session_admin(self, session_id: str) -> dict:
        """Admin-only RPC session end."""
        return self._do_end_session_adapter(session_id)

    @modal.method()
    def validate_fmp4_admin(self, duration_s: int = 15) -> dict:
        """Validate fMP4 pipeline: start session, render, check segments, cleanup."""
        import time as _time

        # 1. Start session
        session_id = f"fmp4-validate-{int(_time.time())}"
        start_result = self._start_session_impl(
            session_id=session_id, avatar_id="arnold", bg_id="aloe_window",
        )
        if "error" in start_result:
            return {"status": "error", "start_error": start_result.get("error")}

        # 2. Start render thread (same as SSE stream does)
        sess = self._sessions[session_id]
        render_thread = threading.Thread(
            target=self._run_session_render_impl,
            args=(session_id, 1000000, 200, "sine", "handoff"),
            daemon=True,
        )
        sess["render_thread"] = render_thread
        render_thread.start()
        print(f"[fmp4-validate] render thread started for {session_id}", flush=True)

        # 3. Wait for segments
        _time.sleep(duration_s)

        # 4. Stop render thread and WAIT for it to finish
        stop_event = sess.get("render_stop_event")
        if stop_event:
            stop_event.set()
        render_thread.join(timeout=30.0)

        # 5. NOW check results (after render thread has finished counting)
        chunks = sess.get("chunk_count", 0)
        media_count = sess.get("fmp4_media_count", 0)
        media_bytes = sess.get("fmp4_media_total_bytes", 0)
        has_init = sess.get("fmp4_init_segment") is not None

        print(f"[fmp4-validate] chunks={chunks} media_count={media_count} "
              f"media_bytes={media_bytes} has_init={has_init}", flush=True)

        # 6. End session
        self._do_end_session_adapter(session_id)
        _time.sleep(2)

        # 7. Force remove session from registry (admin bypass skips Worker)
        self._sessions.pop(session_id, None)
        import torch
        torch.cuda.empty_cache()

        # 8. Verify cleanup
        active = list(self._sessions.keys())

        return {
            "status": "survived" if media_count > 0 else "no_segments",
            "session_id": session_id,
            "chunk_count": chunks,
            "fmp4_media_count": media_count,
            "fmp4_media_total_bytes": media_bytes,
            "has_init_segment": has_init,
            "active_sessions_after": active,
            "engine_backends": dict(getattr(self, "_engine_backends", {})),
        }

    @modal.method()
    def session_status_admin(self, session_id: str) -> dict:
        """Admin-only RPC session status."""
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
            "bg_id": sess.get("bg_id"),
            "loaded_avatar_ids": list(self.registry.keys()),
        }

    # ── Admin-only render-to-null validation ──────────────────────────────
    # Called directly via Modal RPC (not through Worker). Used for crash
    # validation after the official-runtime pivot.
    @modal.method()
    def render_null_validation(self, duration_s: int = 90, avatar_id: str = "") -> dict:
        """Run the full AVTR pipeline with silence audio for ``duration_s``.

        Returns a dict with status (survived/crashed), fps, frame count,
        and error info.  This is the canonical render-stability test —
        callable directly via Modal RPC without going through the Worker.
        """
        import resource as _resource
        import torch as _torch

        if not getattr(self, "_pipeline_loaded", False):
            return {"status": "error", "error": "Pipeline not loaded — restore failed"}

        _aid = avatar_id or self.avatar_id
        avatar = self.registry.get(_aid)
        if avatar is None:
            return {"status": "error", "error": f"Avatar {_aid!r} not in registry"}

        _bg_id = _pick_first_bg(self.pipeline._backgrounds)

        frame_source = GstAvtrFrameSource(
            pipeline=self.pipeline,
            avatar=avatar,
            bg_id=_bg_id,
            handoff=None,
        )

        # Runtime fingerprint
        import sys as _sys
        try:
            import onnxruntime as _ort
            import tensorrt as _trt
            _fp = (
                f"[render_null] python={_sys.version.split()[0]} "
                f"torch={_torch.__version__} "
                f"cuda_runtime={_torch.version.cuda} "
                f"ort={_ort.__version__} "
                f"trt={_trt.__version__} "
                f"gpu={_torch.cuda.get_device_name(0)} "
                f"engines={self._engine_backends}"
            )
            print(_fp, flush=True)
        except Exception as _e:
            print(f"[render_null] fingerprint error: {_e}", flush=True)

        frame_source.start()
        _fps = 25
        _fdur = 1.0 / _fps
        _t0 = time.monotonic()
        _telem_last = _t0
        _hb_last_gen = 0

        while True:
            _elapsed = time.monotonic() - _t0
            if _elapsed >= duration_s:
                break
            _err = frame_source.get_error()
            if _err is not None:
                _total = _elapsed
                print(
                    f"[render_null] CRASHED duration_ms={int(_total*1000)} "
                    f"fps={frame_source.frames_generated/max(_total,0.1):.1f} "
                    f"frames={frame_source.frames_generated} "
                    f"error={type(_err).__name__}: {_err}",
                    flush=True,
                )
                frame_source.stop()
                return {
                    "status": "crashed",
                    "duration_ms": int(_total * 1000),
                    "fps": round(frame_source.frames_generated / max(_total, 0.1), 1),
                    "frames": frame_source.frames_generated,
                    "error": f"{type(_err).__name__}: {_err}",
                    "engine_backends": dict(self._engine_backends),
                }
            frame_source.next_frame()

            _now = time.monotonic()
            if _now - _telem_last >= 5.0:
                _dt = _now - _telem_last
                _gd = frame_source.frames_generated - _hb_last_gen
                _alloc = _torch.cuda.memory_allocated() / (1024**2)
                _reserved = _torch.cuda.memory_reserved() / (1024**2)
                _max_alloc = _torch.cuda.max_memory_allocated() / (1024**2)
                _rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024
                _threads = threading.active_count()
                _cl = frame_source.metrics.chunk_latencies_ms
                _cl_sorted = sorted(_cl[-100:]) if _cl else [0]
                _p50 = _cl_sorted[len(_cl_sorted)//2]
                _p95 = _cl_sorted[int(len(_cl_sorted)*0.95)] if len(_cl_sorted) > 1 else _cl_sorted[0]
                _cmax = _cl_sorted[-1]
                print(
                    f"[render_null] telemetry "
                    f"fps={_gd/_dt:.1f} "
                    f"frames={frame_source.frames_generated} "
                    f"chunk_ms_p50={_p50:.1f} p95={_p95:.1f} max={_cmax:.1f} "
                    f"alloc_mb={_alloc:.0f} reserved_mb={_reserved:.0f} "
                    f"max_alloc_mb={_max_alloc:.0f} "
                    f"rss_mb={_rss:.0f} threads={_threads}",
                    flush=True,
                )
                _telem_last = _now
                _hb_last_gen = frame_source.frames_generated

            _deadline = _t0 + (frame_source.frames_generated + 1) * _fdur
            _sleep = _deadline - time.monotonic()
            if _sleep > 0:
                time.sleep(_sleep)

        frame_source.stop()
        _total = time.monotonic() - _t0
        _status = "survived" if _total >= duration_s - 1 else "stopped_early"
        _err = frame_source.get_error()
        print(
            f"[render_null] {_status} duration_ms={int(_total*1000)} "
            f"fps={frame_source.frames_generated/max(_total,0.1):.1f} "
            f"frames={frame_source.frames_generated} "
            f"error={type(_err).__name__ + ': ' + str(_err) if _err else 'none'}",
            flush=True,
        )
        _torch.cuda.empty_cache()
        return {
            "status": _status,
            "duration_ms": int(_total * 1000),
            "fps": round(frame_source.frames_generated / max(_total, 0.1), 1),
            "frames": frame_source.frames_generated,
            "error": str(_err) if _err else None,
            "engine_backends": dict(self._engine_backends),
        }

    # _run_session_smoke_impl and run_session_smoke removed — use prep helpers
    # file (modal_prep_helpers_5q.py) for diagnostic sessions.
    # Production class ends here. The standalone @app.function build helpers
    # and smoke probes have been moved to modal_prep_helpers_5q.py to keep
    # the production deployment surface minimal.
    #
    # Removed methods:
    #   - run_session_smoke / _run_session_smoke_impl (RPC diagnostic)
    #   - omnivoice_parallel_stress_smoke (RPC diagnostic)
    #   - omnivoice_tts_smoke / _omnivoice_tts_smoke_impl (RPC diagnostic)
    #   - audio_handoff_stream_cadence_smoke (RPC diagnostic)
    #   - audio_handoff_stream_smoke / _audio_handoff_stream_impl (RPC diagnostic)
    #   - fmp4_video_stream_concurrent_smoke (RPC diagnostic)
    #   - fmp4_video_stream_smoke (RPC diagnostic)
    #   - render_health_check (moved to prep helpers — HTTP-based)
    #   - build_trt_engines stub (use build_trt_engines_standalone in prep)
    #   - build_trt_engines_standalone (moved to prep)
    #   - patch_warp_gridsample_standalone (moved to prep)
    #   - build_warp_trt_engine_standalone (moved to prep)
    #   - _run_warp_build_entrypoint (moved to prep)
    #   - _run_render_health_check (moved to prep)
    # Production surface: 2 RPC methods (health_check, gst_runtime_probe)
    # + 7 HTTP endpoints (start, status, push-audio, end, stream, push-batch, tts-parallel).


# ── Standalone TRT engine builder (admin-only, not part of production class) ──

@app.function(
    image=image,
    gpu="L40S",
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=1800,
)
def build_trt_engines(engines: str = "all"):
    """Build TRT engines using official build scripts.

    Call via Modal RPC after deploy.

    Args:
        engines: comma-separated engine names or "all".
                  Options: decoder, stitch, modnet, warp, hubert
    """
    import sys
    import os

    sys.path.insert(0, "/opt/avtr1")
    sys.path.insert(0, "/opt/build_scripts")

    from avtr1_renderer.avtr1_artifact_manager import (
        get_storage_root, get_trt_engine_path, get_artifact_manager
    )

    storage = get_storage_root()
    print(f"[build] Storage root: {storage}", flush=True)

    # Print current engine status
    print(f"\n[build] Current engine status:", flush=True)
    for name in ["avtr1_encode", "avtr1_decode", "hubert_lbs", "decoder",
                 "stitch_network", "warp_network", "modnet"]:
        try:
            p = get_trt_engine_path(name)
            ex = p.is_file()
            sz = p.stat().st_size / 1e6 if ex else 0
            print(f"  {name:20s} {'OK' if ex else 'MISSING':7s} {sz:>8.1f}MB  {p}", flush=True)
        except Exception as e:
            print(f"  {name:20s} ERROR: {e}", flush=True)

    # Print ONNX source status
    mgr = get_artifact_manager()
    print(f"\n[build] ONNX sources:", flush=True)
    for artifact in ["decoder_onnx", "stitch_network_onnx", "modnet_onnx",
                     "hubert_onnx", "warp_network_onnx"]:
        try:
            p = mgr.get_artifact_path(artifact)
            ex = p.is_file()
            sz = p.stat().st_size / 1e6 if ex else 0
            print(f"  {artifact:25s} {'OK' if ex else 'MISSING':7s} {sz:>8.1f}MB", flush=True)
        except Exception as e:
            print(f"  {artifact:25s} ERROR: {e}", flush=True)

    # Determine what to build
    if engines == "all":
        to_build = ["decoder", "stitch", "modnet", "hubert"]
    else:
        to_build = [e.strip() for e in engines.split(",")]

    print(f"\n[build] Building: {to_build}", flush=True)

    # Build renderer engines (decoder, stitch, modnet, warp)
    renderer_engines = [e for e in to_build if e in ("decoder", "stitch", "modnet", "warp")]
    if renderer_engines:
        print(f"\n{'='*60}", flush=True)
        print(f"[build] Renderer engines: {renderer_engines}", flush=True)
        print(f"{'='*60}", flush=True)
        import build_renderer_engines
        build_renderer_engines.main(renderer_engines)

    # Build HuBERT engine
    if "hubert" in to_build:
        print(f"\n{'='*60}", flush=True)
        print(f"[build] HuBERT engine", flush=True)
        print(f"{'='*60}", flush=True)
        import build_hubert_engine
        build_hubert_engine.main([])

    # Print final status
    print(f"\n[build] Final engine status:", flush=True)
    for name in ["avtr1_encode", "avtr1_decode", "hubert_lbs", "decoder",
                 "stitch_network", "warp_network", "modnet"]:
        try:
            p = get_trt_engine_path(name)
            ex = p.is_file()
            sz = p.stat().st_size / 1e6 if ex else 0
            print(f"  {name:20s} {'OK' if ex else 'MISSING':7s} {sz:>8.1f}MB  {p}", flush=True)
        except Exception as e:
            print(f"  {name:20s} ERROR: {e}", flush=True)

    # Commit volume
    l40s_volume.commit()
    print(f"\n[build] Volume committed successfully.", flush=True)

    return {"status": "done", "built": to_build}
