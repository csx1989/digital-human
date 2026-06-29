"""
Modal L40S GPU Snapshot Isolation Probes.

7 minimal probe classes under one Modal app. Each probe tests a different
combination of model loading/warmup to isolate what causes SIGSEGV 139
on GPU snapshot restore.

DO NOT wire into production routing. This is a diagnostic investigation.
Do NOT modify the live L40S lifecycle app, Bridge, browser, or routing.

Deploy:
    modal deploy modal_l40s_snapshot_isolation_probes.py

Run all probes:
    modal run modal_l40s_snapshot_isolation_probes.py
"""

from __future__ import annotations

import json
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

app = modal.App("avtr1-l40s-snapshot-isolation", image=image)

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
    print(f"[probe] Storage root: {storage_root}")
    print("[probe] Ensuring all HF artifacts are present...")
    t0 = time.monotonic()
    mgr = get_artifact_manager()
    mgr.ensure_all_artifacts(workers=4)
    elapsed = time.monotonic() - t0
    print(f"[probe] Artifacts ready in {elapsed:.0f}s")
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
        raise FileNotFoundError(f"No reference_frames at {portraits_dir}")
    all_pngs = sorted(portraits_dir.glob("*.png"))
    if not all_pngs:
        raise FileNotFoundError(f"No portrait PNGs in {portraits_dir}")
    stems = [p.stem for p in all_pngs]
    if "arnold" in stems:
        return "arnold"
    return stems[0]


def _pick_first_bg(backgrounds: dict[str, object]) -> str:
    for bg_id in sorted(backgrounds):
        if bg_id != "transparent":
            return bg_id
    raise RuntimeError("No non-transparent backgrounds registered")


def _run_nvidia_smi(label: str) -> dict:
    info: dict = {"label": label}
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
                info["gpu_name"] = parts[0]
                info["mem_used_mb"] = int(parts[1])
                info["mem_total_mb"] = int(parts[2])
                info["gpu_util_pct"] = int(parts[3])
                print(f"[nvidia-smi] {label}: {parts[0]}, {parts[1]}/{parts[2]} MiB, util={parts[3]}%")
            else:
                info["raw"] = line
    except Exception as exc:
        info["error"] = str(exc)
    return info


def _version_report() -> dict:
    versions: dict = {}
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda_version"] = torch.version.cuda
        versions["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
            versions["gpu_count"] = torch.cuda.device_count()
    except Exception:
        pass
    try:
        import tensorrt as trt
        versions["tensorrt"] = trt.__version__
    except Exception:
        versions["tensorrt"] = "not importable"
    try:
        import onnxruntime as ort
        versions["onnxruntime"] = ort.__version__
        versions["ort_providers"] = ort.get_available_providers()
    except Exception:
        versions["onnxruntime"] = "not importable"
    return versions


def _vram_snapshot(label: str) -> dict:
    snap: dict = {"label": label}
    try:
        import torch
        if torch.cuda.is_available():
            snap["torch_alloc_mb"] = round(torch.cuda.memory_allocated(0) / 1024 / 1024, 1)
            snap["torch_reserved_mb"] = round(torch.cuda.memory_reserved(0) / 1024 / 1024, 1)
            snap["torch_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024, 1)
    except Exception:
        pass
    smi = _run_nvidia_smi(label)
    snap.update({k: v for k, v in smi.items() if k != "label"})
    return snap


_CLS_KWARGS = dict(
    gpu="L40S",
    timeout=600,
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=90,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)


# ═══════════════════════════════════════════════════════════════════════
# PROBE 1: Minimal CUDA Sanity
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalCudaSnapshotL40S:
    """Probe 1: Bare CUDA tensor ops. No AVTR, no OmniVoice, no TensorRT."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch

        print("[probe-1-cuda] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-1-cuda] versions: {self._versions}")

        self._vram_before = _vram_snapshot("snap-start")

        x = torch.randn(100, 100, device="cuda:0")
        y = x @ x.T
        self._snap_tensor_sum = float(y.sum())
        print(f"[probe-1-cuda] tensor matmul sum: {self._snap_tensor_sum:.4f}")

        self._vram_after = _vram_snapshot("snap-end")
        print("[probe-1-cuda] === snap=True complete ===")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-1-cuda] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            if torch.cuda.is_available():
                z = torch.ones(10, device="cuda:0")
                self._restore_tensor_ok = float(z.sum()) == 10.0
            else:
                self._restore_tensor_ok = False
                self._restore_error = "CUDA not available after restore"
                self._restore_ok = False
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-1-cuda] restore FAILED: {exc}")
        print("[probe-1-cuda] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "1-minimal-cuda",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "tensor_sum": getattr(self, "_snap_tensor_sum", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
            "restore_tensor_ok": getattr(self, "_restore_tensor_ok", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 2: AVTR Load-Only (no warmup)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalAvtrLoadSnapshotL40S:
    """Probe 2: Load AVTR pipeline (TensorRT engines + ONNX) but do NOT call process_chunk."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch

        print("[probe-2-avtr-load] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-2-avtr-load] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-load")

            storage_root = _ensure_artifacts(Path(L40S_MOUNT_PATH))
            l40s_volume.commit()

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                self._snap_ok = False
                self._snap_error = f"Missing artifacts: {missing}"
                print(f"[probe-2-avtr-load] ERROR: {self._snap_error}")
                return

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline

            print(f"[probe-2-avtr-load] Loading pipeline avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self._load_s = time.monotonic() - t0
            print(f"[probe-2-avtr-load] Pipeline loaded in {self._load_s:.1f}s")
            self._avatar_id = avatar_id

            self._vram_after = _vram_snapshot("after-load")
            print("[probe-2-avtr-load] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-2-avtr-load] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-2-avtr-load] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._pipeline_present = hasattr(self, "pipeline") and self.pipeline is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-2-avtr-load] restore FAILED: {exc}")
        print("[probe-2-avtr-load] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "2-avtr-load-only",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "load_s": getattr(self, "_load_s", None),
            "avatar_id": getattr(self, "_avatar_id", None),
            "pipeline_present": getattr(self, "_pipeline_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 3: AVTR Load + Warmup (one process_chunk)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalAvtrWarmupSnapshotL40S:
    """Probe 3: Load AVTR pipeline + run one silence process_chunk. Tests TensorRT/ONNX execution state."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch
        import numpy as np

        print("[probe-3-avtr-warmup] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-3-avtr-warmup] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-load")

            storage_root = _ensure_artifacts(Path(L40S_MOUNT_PATH))
            l40s_volume.commit()

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                self._snap_ok = False
                self._snap_error = f"Missing artifacts: {missing}"
                return

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions

            print(f"[probe-3-avtr-warmup] Loading pipeline avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self._load_s = time.monotonic() - t0
            print(f"[probe-3-avtr-warmup] Pipeline loaded in {self._load_s:.1f}s")

            avatar = self.registry[avatar_id]
            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            chunk = Chunk(audio_speech=silence, audio_listen=silence)
            bg_id = _pick_first_bg(self.pipeline._backgrounds)
            options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

            print("[probe-3-avtr-warmup] Running warmup process_chunk...")
            t_warm = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                avatar, chunk, state=None, options=options,
            )
            warmup_frames = list(frames_iter)
            self._warmup_ms = (time.monotonic() - t_warm) * 1000
            self._warmup_frame_count = len(warmup_frames)
            print(
                f"[probe-3-avtr-warmup] Warmup: {self._warmup_ms:.0f}ms, "
                f"{self._warmup_frame_count} frames"
            )

            self._vram_after = _vram_snapshot("after-warmup")
            self._avatar_id = avatar_id
            print("[probe-3-avtr-warmup] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-3-avtr-warmup] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-3-avtr-warmup] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._pipeline_present = hasattr(self, "pipeline") and self.pipeline is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-3-avtr-warmup] restore FAILED: {exc}")
        print("[probe-3-avtr-warmup] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "3-avtr-warmup",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "load_s": getattr(self, "_load_s", None),
            "warmup_ms": getattr(self, "_warmup_ms", None),
            "warmup_frame_count": getattr(self, "_warmup_frame_count", None),
            "avatar_id": getattr(self, "_avatar_id", None),
            "pipeline_present": getattr(self, "_pipeline_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 4: OmniVoice Load-Only (no generation)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalOmniLoadSnapshotL40S:
    """Probe 4: Load OmniVoice TTS model onto GPU. No AVTR, no generation."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch

        print("[probe-4-omni-load] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-4-omni-load] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-omni-load")

            from omnivoice import OmniVoice

            device = "cuda:0"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16

            print(f"[probe-4-omni-load] Loading OmniVoice dtype={dtype}...")
            t0 = time.monotonic()
            self._omnivoice_model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            self._omni_load_s = time.monotonic() - t0
            self._omnivoice_sr = int(getattr(self._omnivoice_model, "sampling_rate", 24000))
            self._omnivoice_dtype = str(dtype)
            print(
                f"[probe-4-omni-load] OmniVoice loaded in {self._omni_load_s:.1f}s, "
                f"sr={self._omnivoice_sr}, dtype={dtype}"
            )

            self._vram_after = _vram_snapshot("after-omni-load")
            print("[probe-4-omni-load] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-4-omni-load] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-4-omni-load] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._model_present = hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-4-omni-load] restore FAILED: {exc}")
        print("[probe-4-omni-load] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "4-omnivoice-load-only",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "omni_load_s": getattr(self, "_omni_load_s", None),
            "omnivoice_sr": getattr(self, "_omnivoice_sr", None),
            "model_present": getattr(self, "_model_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 5: OmniVoice Load + Generate (one full sentence)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalOmniGenerateSnapshotL40S:
    """Probe 5: Load OmniVoice + run one full sentence generation. Tests PyTorch CUDA execution state."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch

        print("[probe-5-omni-generate] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-5-omni-generate] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-omni-load")

            from omnivoice import OmniVoice

            device = "cuda:0"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16

            print(f"[probe-5-omni-generate] Loading OmniVoice dtype={dtype}...")
            t0 = time.monotonic()
            self._omnivoice_model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            self._omni_load_s = time.monotonic() - t0
            self._omnivoice_sr = int(getattr(self._omnivoice_model, "sampling_rate", 24000))
            print(f"[probe-5-omni-generate] OmniVoice loaded in {self._omni_load_s:.1f}s")

            self._vram_after_load = _vram_snapshot("after-omni-load")

            print("[probe-5-omni-generate] Running one generation...")
            t_gen = time.monotonic()
            with torch.inference_mode():
                audio = self._omnivoice_model.generate(
                    "Hello, this is a test of the text to speech system for snapshot isolation.",
                    preprocess_prompt=True,
                    postprocess_output=True,
                    denoise=True,
                    t_shift=0.1,
                    layer_penalty_factor=5.0,
                    position_temperature=5.0,
                    class_temperature=0.0,
                )
            self._generate_ms = (time.monotonic() - t_gen) * 1000

            import numpy as np
            audio_data = audio[0] if isinstance(audio, (list, tuple)) else audio
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().float()
            if audio_data.dim() > 1:
                audio_data = audio_data.squeeze(0)
            self._audio_dur_s = float(audio_data.shape[0]) / self._omnivoice_sr
            print(
                f"[probe-5-omni-generate] Generated {self._audio_dur_s:.2f}s audio "
                f"in {self._generate_ms:.0f}ms"
            )

            self._vram_after = _vram_snapshot("after-generate")
            print("[probe-5-omni-generate] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-5-omni-generate] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-5-omni-generate] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._model_present = hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-5-omni-generate] restore FAILED: {exc}")
        print("[probe-5-omni-generate] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "5-omnivoice-generate",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after_load": getattr(self, "_vram_after_load", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "omni_load_s": getattr(self, "_omni_load_s", None),
            "generate_ms": getattr(self, "_generate_ms", None),
            "audio_dur_s": getattr(self, "_audio_dur_s", None),
            "omnivoice_sr": getattr(self, "_omnivoice_sr", None),
            "model_present": getattr(self, "_model_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 6: Combined Load-Only (AVTR load + OmniVoice load, no warmup/generation)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalCombinedLoadSnapshotL40S:
    """Probe 6: Load AVTR pipeline + OmniVoice. No warmup, no generation. Tests combined loaded state."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch

        print("[probe-6-combined-load] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-6-combined-load] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-all-load")

            storage_root = _ensure_artifacts(Path(L40S_MOUNT_PATH))
            l40s_volume.commit()

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                self._snap_ok = False
                self._snap_error = f"Missing artifacts: {missing}"
                return

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline

            print(f"[probe-6-combined-load] Loading AVTR pipeline avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self._avtr_load_s = time.monotonic() - t0
            print(f"[probe-6-combined-load] AVTR loaded in {self._avtr_load_s:.1f}s")

            self._vram_after_avtr = _vram_snapshot("after-avtr-load")

            from omnivoice import OmniVoice

            device = "cuda:0"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16

            print(f"[probe-6-combined-load] Loading OmniVoice dtype={dtype}...")
            t_omni = time.monotonic()
            self._omnivoice_model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            self._omni_load_s = time.monotonic() - t_omni
            self._omnivoice_sr = int(getattr(self._omnivoice_model, "sampling_rate", 24000))
            print(f"[probe-6-combined-load] OmniVoice loaded in {self._omni_load_s:.1f}s")

            self._vram_after = _vram_snapshot("after-omni-load")
            self._avatar_id = avatar_id
            print("[probe-6-combined-load] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-6-combined-load] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-6-combined-load] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._pipeline_present = hasattr(self, "pipeline") and self.pipeline is not None
            self._omni_present = hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-6-combined-load] restore FAILED: {exc}")
        print("[probe-6-combined-load] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "6-combined-load-only",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after_avtr": getattr(self, "_vram_after_avtr", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "avtr_load_s": getattr(self, "_avtr_load_s", None),
            "omni_load_s": getattr(self, "_omni_load_s", None),
            "avatar_id": getattr(self, "_avatar_id", None),
            "pipeline_present": getattr(self, "_pipeline_present", None),
            "omni_present": getattr(self, "_omni_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# PROBE 7: Combined Warmup (AVTR load + warmup + OmniVoice load, no OmniVoice generation)
# ═══════════════════════════════════════════════════════════════════════


@app.cls(**_CLS_KWARGS)
class MinimalCombinedWarmupSnapshotL40S:
    """Probe 7: AVTR load + warmup + OmniVoice load. No OmniVoice generation yet.
    If this restore passes, a follow-up probe with OmniVoice generation is warranted."""

    @modal.enter(snap=True)
    def _snap_init(self):
        import torch
        import numpy as np

        print("[probe-7-combined-warmup] === snap=True starting ===")
        self._snap_ok = True
        self._snap_error = None
        self._versions = _version_report()
        print(f"[probe-7-combined-warmup] versions: {self._versions}")

        try:
            self._vram_before = _vram_snapshot("before-all-load")

            storage_root = _ensure_artifacts(Path(L40S_MOUNT_PATH))
            l40s_volume.commit()

            ok, missing = _validate_all_artifacts(storage_root)
            if not ok:
                self._snap_ok = False
                self._snap_error = f"Missing artifacts: {missing}"
                return

            avatar_id = _pick_first_avatar(storage_root)

            from avtr1_renderer.pipeline import Pipeline
            from avtr1_renderer.types import Chunk, RenderOptions

            print(f"[probe-7-combined-warmup] Loading AVTR pipeline avatar_id={avatar_id!r}...")
            t0 = time.monotonic()
            self.pipeline, self.registry = Pipeline.from_artifacts(
                avatar_ids=[avatar_id],
            )
            self._avtr_load_s = time.monotonic() - t0
            print(f"[probe-7-combined-warmup] AVTR loaded in {self._avtr_load_s:.1f}s")

            avatar = self.registry[avatar_id]
            mg = self.pipeline._motion_generator
            expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
            silence = np.zeros(expected_len, dtype=np.float32)
            chunk = Chunk(audio_speech=silence, audio_listen=silence)
            bg_id = _pick_first_bg(self.pipeline._backgrounds)
            options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)

            print("[probe-7-combined-warmup] Running AVTR warmup process_chunk...")
            t_warm = time.monotonic()
            self._state, frames_iter = self.pipeline.process_chunk(
                avatar, chunk, state=None, options=options,
            )
            warmup_frames = list(frames_iter)
            self._warmup_ms = (time.monotonic() - t_warm) * 1000
            self._warmup_frame_count = len(warmup_frames)
            print(
                f"[probe-7-combined-warmup] AVTR warmup: {self._warmup_ms:.0f}ms, "
                f"{self._warmup_frame_count} frames"
            )

            self._vram_after_avtr = _vram_snapshot("after-avtr-warmup")

            from omnivoice import OmniVoice

            device = "cuda:0"
            use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16

            print(f"[probe-7-combined-warmup] Loading OmniVoice dtype={dtype}...")
            t_omni = time.monotonic()
            self._omnivoice_model = OmniVoice.from_pretrained(
                MODEL_ID,
                device_map=device,
                dtype=dtype,
                attn_implementation="eager",
            )
            self._omni_load_s = time.monotonic() - t_omni
            self._omnivoice_sr = int(getattr(self._omnivoice_model, "sampling_rate", 24000))
            print(f"[probe-7-combined-warmup] OmniVoice loaded in {self._omni_load_s:.1f}s")

            self._vram_after = _vram_snapshot("after-omni-load")
            self._avatar_id = avatar_id
            print("[probe-7-combined-warmup] === snap=True complete ===")

        except Exception as exc:
            self._snap_ok = False
            self._snap_error = str(exc)
            import traceback
            traceback.print_exc()
            print(f"[probe-7-combined-warmup] FAILED: {exc}")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        print("[probe-7-combined-warmup] === snap=False (restore) starting ===")
        self._restore_ok = True
        self._restore_error = None
        try:
            self._restore_cuda_avail = torch.cuda.is_available()
            self._restore_gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            self._pipeline_present = hasattr(self, "pipeline") and self.pipeline is not None
            self._omni_present = hasattr(self, "_omnivoice_model") and self._omnivoice_model is not None
            self._restore_vram = _vram_snapshot("post-restore")
        except Exception as exc:
            self._restore_ok = False
            self._restore_error = str(exc)
            print(f"[probe-7-combined-warmup] restore FAILED: {exc}")
        print("[probe-7-combined-warmup] === snap=False complete ===")

    @modal.method()
    def health(self) -> dict:
        return {
            "probe": "7-combined-warmup",
            "snap_ok": self._snap_ok,
            "snap_error": getattr(self, "_snap_error", None),
            "restore_ok": getattr(self, "_restore_ok", None),
            "restore_error": getattr(self, "_restore_error", None),
            "versions": getattr(self, "_versions", {}),
            "vram_before": getattr(self, "_vram_before", {}),
            "vram_after_avtr": getattr(self, "_vram_after_avtr", {}),
            "vram_after": getattr(self, "_vram_after", {}),
            "vram_restore": getattr(self, "_restore_vram", {}),
            "avtr_load_s": getattr(self, "_avtr_load_s", None),
            "warmup_ms": getattr(self, "_warmup_ms", None),
            "warmup_frame_count": getattr(self, "_warmup_frame_count", None),
            "omni_load_s": getattr(self, "_omni_load_s", None),
            "avatar_id": getattr(self, "_avatar_id", None),
            "pipeline_present": getattr(self, "_pipeline_present", None),
            "omni_present": getattr(self, "_omni_present", None),
            "restore_cuda_avail": getattr(self, "_restore_cuda_avail", None),
        }


# ═══════════════════════════════════════════════════════════════════════
# Local Entrypoint: Run all probes and report matrix
# ═══════════════════════════════════════════════════════════════════════


_PROBE_CLASSES = [
    ("1-minimal-cuda", "MinimalCudaSnapshotL40S"),
    ("2-avtr-load-only", "MinimalAvtrLoadSnapshotL40S"),
    ("3-avtr-warmup", "MinimalAvtrWarmupSnapshotL40S"),
    ("4-omnivoice-load-only", "MinimalOmniLoadSnapshotL40S"),
    ("5-omnivoice-generate", "MinimalOmniGenerateSnapshotL40S"),
    ("6-combined-load-only", "MinimalCombinedLoadSnapshotL40S"),
    ("7-combined-warmup", "MinimalCombinedWarmupSnapshotL40S"),
]


@app.local_entrypoint()
def run_probes():
    import json

    results: dict[str, dict] = {}

    print("=" * 72)
    print("PHASE 1: Snapshot creation (first call per probe)")
    print("=" * 72)

    for label, cls_name in _PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        cls = modal.Cls.from_name("avtr1-l40s-snapshot-isolation", cls_name)
        instance = cls()
        try:
            t0 = time.monotonic()
            result = instance.health.remote()
            wall_s = time.monotonic() - t0
            result["_wall_s"] = round(wall_s, 2)
            result["_phase"] = "snapshot-creation"
            print(f"  wall={wall_s:.1f}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
            if result.get("snap_error"):
                print(f"  snap_error: {result['snap_error']}")
            if result.get("restore_error"):
                print(f"  restore_error: {result['restore_error']}")
            results[f"{label}-phase1"] = result
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[f"{label}-phase1"] = {"probe": label, "_phase": "snapshot-creation", "_error": str(exc)}

    print("\n" + "=" * 72)
    print("Waiting 100s for containers to scale to zero...")
    print("=" * 72)
    time.sleep(100)

    print("\n" + "=" * 72)
    print("PHASE 2: Restore test 1")
    print("=" * 72)

    for label, cls_name in _PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        cls = modal.Cls.from_name("avtr1-l40s-snapshot-isolation", cls_name)
        instance = cls()
        try:
            t0 = time.monotonic()
            result = instance.health.remote()
            wall_s = time.monotonic() - t0
            result["_wall_s"] = round(wall_s, 2)
            result["_phase"] = "restore-1"
            print(f"  wall={wall_s:.1f}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
            if result.get("restore_error"):
                print(f"  restore_error: {result['restore_error']}")
            results[f"{label}-phase2"] = result
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[f"{label}-phase2"] = {"probe": label, "_phase": "restore-1", "_error": str(exc)}

    print("\n" + "=" * 72)
    print("Waiting 100s for containers to scale to zero...")
    print("=" * 72)
    time.sleep(100)

    print("\n" + "=" * 72)
    print("PHASE 3: Restore test 2")
    print("=" * 72)

    for label, cls_name in _PROBE_CLASSES:
        print(f"\n--- {label} ({cls_name}) ---")
        cls = modal.Cls.from_name("avtr1-l40s-snapshot-isolation", cls_name)
        instance = cls()
        try:
            t0 = time.monotonic()
            result = instance.health.remote()
            wall_s = time.monotonic() - t0
            result["_wall_s"] = round(wall_s, 2)
            result["_phase"] = "restore-2"
            print(f"  wall={wall_s:.1f}s snap_ok={result.get('snap_ok')} restore_ok={result.get('restore_ok')}")
            if result.get("restore_error"):
                print(f"  restore_error: {result['restore_error']}")
            results[f"{label}-phase3"] = result
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results[f"{label}-phase3"] = {"probe": label, "_phase": "restore-2", "_error": str(exc)}

    print("\n" + "=" * 72)
    print("RESULTS MATRIX")
    print("=" * 72)

    header = (
        f"{'Probe':<30} {'AVTR':>5} {'AVTR':>7} {'Omni':>5} {'Omni':>6} "
        f"{'TRT/':>6} {'Snap':>5} {'Rstr1':>6} {'Rstr2':>6} {'Error':<30} {'Conclusion':<40}"
    )
    subheader = (
        f"{'':>30} {'load':>5} {'warmup':>7} {'load':>5} {'gen':>6} "
        f"{'ONNX':>6} {'ok':>5} {'ok':>6} {'ok':>6} {'':>30} {'':>40}"
    )
    print(header)
    print(subheader)
    print("-" * len(header))

    for label, cls_name in _PROBE_CLASSES:
        probe_num = label.split("-")[0]
        phase1 = results.get(f"{label}-phase1", {})
        phase2 = results.get(f"{label}-phase2", {})
        phase3 = results.get(f"{label}-phase3", {})

        avtr_load = "Y" if label in ("2-avtr-load-only", "3-avtr-warmup", "6-combined-load-only", "7-combined-warmup") else "-"
        avtr_warm = "Y" if label in ("3-avtr-warmup", "7-combined-warmup") else "-"
        omni_load = "Y" if label in ("4-omnivoice-load-only", "5-omnivoice-generate", "6-combined-load-only", "7-combined-warmup") else "-"
        omni_gen = "Y" if label in ("5-omnivoice-generate",) else "-"
        trt_onnx = "Y" if label not in ("1-minimal-cuda", "4-omnivoice-load-only", "5-omnivoice-generate") else "-"

        snap_ok_str = str(phase1.get("snap_ok", "?"))
        rstr1_ok = phase2.get("restore_ok", phase2.get("_error", "FAIL"))
        rstr2_ok = phase3.get("restore_ok", phase3.get("_error", "FAIL"))

        rstr1_str = str(rstr1_ok) if isinstance(rstr1_ok, bool) else "FAIL"
        rstr2_str = str(rstr2_ok) if isinstance(rstr2_ok, bool) else "FAIL"

        error = ""
        for phase_data in [phase1, phase2, phase3]:
            e = phase_data.get("restore_error") or phase_data.get("snap_error") or phase_data.get("_error", "")
            if e:
                error = str(e)[:30]
                break

        conclusion = ""
        if phase2.get("_error") or phase3.get("_error"):
            conclusion = "SIGSEGV/timeout — probe unreachable"
        elif not phase1.get("snap_ok", False):
            conclusion = "Snap init failed — model load issue"
        elif rstr1_str == "True" and rstr2_str == "True":
            conclusion = "PASS — restore works"
        elif rstr1_str == "False" or rstr2_str == "False":
            conclusion = "FAIL — restore error"
        else:
            conclusion = "UNKNOWN"

        print(
            f"{label:<30} {avtr_load:>5} {avtr_warm:>7} {omni_load:>5} {omni_gen:>6} "
            f"{trt_onnx:>6} {snap_ok_str:>5} {rstr1_str:>6} {rstr2_str:>6} "
            f"{error:<30} {conclusion:<40}"
        )

    print("\n" + "=" * 72)
    print("RAW JSON RESULTS")
    print("=" * 72)
    print(json.dumps(results, indent=2, default=str))
