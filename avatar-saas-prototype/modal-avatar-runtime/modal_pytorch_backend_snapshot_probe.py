"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ISOLATED FEASIBILITY PROBE — NOT PRODUCTION                              ║
║  PyTorch Backend + GPU Snapshot Probe for AVTR-1                          ║
║                                                                            ║
║  Goal: Determine whether loading AVTR1 via torch.jit.load() (pure         ║
║  PyTorch, no TensorRT, no ONNX Runtime GPU) enables stable Modal          ║
║  GPU snapshots on L40S.                                                    ║
║                                                                            ║
║  This is a SEPARATE Modal app. It does NOT touch:                         ║
║    - Production L40S lifecycle app                                         ║
║    - Bridge / Web client / wrangler flags / routing                        ║
║    - Any production volumes                                                ║
║                                                                            ║
║  DO NOT wire into production routing.                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Probe 1: AVTR1 Encode + Decode via torch.jit.load()
  - Downloads avtr1.scripted.pt from HuggingFace (gated repo, needs hf-secret)
  - Loads via torch.jit.load() on GPU (no TRT, no ONNX Runtime)
  - Verifies output shapes, dtypes, NaN checks, numeric range
  - Benchmarks eager PyTorch latency (cold + warm)
  - Optionally attempts torch.compile (non-blocking; reports failure and continues)
  - Enables GPU snapshot
  - Waits 150s between restore attempts (scaledown_window=90)
  - 3 restore attempts, checks logs for real restore vs fallback

Artifact availability:
  - avtr1.scripted.pt: HuggingFace avaturn-live/avtr-1, build_artifacts/
  - NOT present locally (LFS pointer only). Downloaded at runtime via HF Hub.
  - Requires HF auth: modal.Secret.from_name("hf-secret")

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_backend_snapshot_probe.py

Invoke:
    modal.Cls.from_name("avtr1-pytorch-snapshot-probe", "PytorchSnapshotProbe")
"""

from __future__ import annotations

import gc
import json
import os
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

PROBE_VOLUME_NAME = "avtr1-pytorch-probe-artifacts"
PROBE_MOUNT_PATH = "/data/avtr1-pytorch-probe"
HF_CACHE_PATH = "/root/.cache/huggingface"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

probe_image = (
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
        "numpy>=2.0,<3",
        "safetensors>=0.5,<1",
        "huggingface_hub>=0.30,<1",
        "httpx>=0.28,<0.29",
        "fastapi>=0.116,<0.117",
        "uvicorn>=0.35,<0.36",
        "tqdm>=4.67,<5",
    )
    .env(
        {
            "PYTHONPATH": "/opt/avtr1",
        }
    )
)

probe_volume = modal.Volume.from_name(PROBE_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-pytorch-snapshot-probe",
    image=probe_image,
    volumes={
        PROBE_MOUNT_PATH: probe_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
)


def _check_nan(tensor, name: str) -> dict:
    import torch
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": round(tensor.min().item(), 6) if not has_nan else "NaN",
        "max": round(tensor.max().item(), 6) if not has_nan else "NaN",
        "mean": round(tensor.mean().item(), 6) if not has_nan else "NaN",
        "std": round(tensor.std().item(), 6) if not has_nan else "NaN",
    }


def _run_nvidia_smi(label: str) -> None:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"[nvidia-smi] {label}: {result.stdout.strip()}")
    except Exception:
        pass


@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=600,
    scaledown_window=90,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class PytorchSnapshotProbe:

    @modal.enter(snap=True)
    def _gpu_snap_init(self) -> None:
        """Load AVTR1 TorchScript model on GPU for snapshot.

        Uses ONLY pure PyTorch. No TensorRT. No ONNX Runtime.
        Downloads avtr1.scripted.pt from HuggingFace via artifact manager.
        If download fails, raises — do NOT fake or bypass.
        """
        import torch

        self._phase = "snap_init"
        self._snap_init_t0 = time.monotonic()

        print("[probe-snap] === GPU Snapshot Init Starting ===")
        print("[probe-snap] Backend: PURE PYTORCH (no TRT, no ONNX Runtime GPU)")
        print("[probe-snap] NOTE: Not importing avtr1_renderer (has onnxruntime dep chain).")
        print("[probe-snap]       Downloading avtr1.scripted.pt directly via huggingface_hub.")

        from huggingface_hub import hf_hub_download

        print("[probe-snap] Downloading avtr1.scripted.pt from avaturn-live/avtr-1...")
        t0 = time.monotonic()
        ckpt_path = hf_hub_download(
            repo_id="avaturn-live/avtr-1",
            filename="build_artifacts/avtr1.scripted.pt",
            repo_type="model",
            cache_dir=str(Path(HF_CACHE_PATH) / "hub"),
        )
        download_s = time.monotonic() - t0
        ckpt_path = Path(ckpt_path)
        print(f"[probe-snap] Downloaded in {download_s:.1f}s")

        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"AVTR1 TorchScript NOT FOUND after download: {ckpt_path}. "
                f"Verify HF auth (hf-secret) and repo access (avaturn-live/avtr-1)."
            )

        file_size_mb = ckpt_path.stat().st_size / 1e6
        print(f"[probe-snap] avtr1.scripted.pt: {file_size_mb:.1f} MB at {ckpt_path}")
        _run_nvidia_smi("snap/before-load")

        print(f"[probe-snap] Loading via torch.jit.load(map_location='cuda')...")
        t0 = time.monotonic()
        self.scripted = torch.jit.load(str(ckpt_path), map_location="cuda").eval()
        load_s = time.monotonic() - t0
        print(f"[probe-snap] TorchScript loaded in {load_s:.1f}s")

        inner = self.scripted.model
        self._chunk_size = int(inner.chunk_size)
        self._past_size = int(inner.past_size)
        self._future_size = int(inner.future_size)
        self._nfeats = int(inner.nfeats)
        self._cond_feature_dim = int(inner.cond_feature_dim)
        self._latent_dim = int(inner.latent_dim)

        print(f"[probe-snap] Shapes: chunk={self._chunk_size}, past={self._past_size}, "
              f"future={self._future_size}, nfeats={self._nfeats}, "
              f"latent={self._latent_dim}, cond_dim={self._cond_feature_dim}")

        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[probe-snap] VRAM after load: {vram:.0f} MB")
        _run_nvidia_smi("snap/after-load")

        self._file_size_mb = file_size_mb
        self._download_s = download_s
        self._load_s = load_s
        self._container_id = id(self)
        self._snap_init_elapsed = time.monotonic() - self._snap_init_t0
        print(f"[probe-snap] Init complete in {self._snap_init_elapsed:.1f}s")
        print("[probe-snap] === GPU Snapshot Init Done ===")

    @modal.enter(snap=False)
    def _post_restore_init(self) -> None:
        """Post-restore hook. Verifies CUDA + model present after snapshot restore."""
        import torch

        self._restore_t0 = time.monotonic()
        print("[probe-restore] === Post-restore init ===")

        cuda_ok = torch.cuda.is_available()
        print(f"[probe-restore] CUDA available: {cuda_ok}")

        if cuda_ok:
            print(f"[probe-restore] GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.memory_allocated() / 1024**2
            print(f"[probe-restore] VRAM: {vram:.0f} MB")
            _run_nvidia_smi("restore/start")

        if not hasattr(self, "scripted") or self.scripted is None:
            raise RuntimeError(
                "TorchScript model NOT found after restore. "
                "GPU snapshot restore FAILED — model not in GPU memory."
            )

        new_id = id(self)
        container_changed = new_id != getattr(self, "_container_id", None)
        self._container_id = new_id

        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[probe-restore] Container ID changed: {container_changed}")
        print(f"[probe-restore] Init elapsed: {self._restore_elapsed:.4f}s")
        print("[probe-restore] === Post-restore init done ===")

    def _make_encode_inputs(self) -> dict:
        """Create synthetic inputs matching AVTR1 encode model shapes."""
        import torch

        B = 1
        chunk = self._chunk_size
        past = self._past_size
        future = self._future_size
        nfeats = self._nfeats
        cond_dim = self._cond_feature_dim
        device = "cuda"
        dtype = torch.float32

        return {
            "past_cond": torch.randn(B, past, nfeats, device=device, dtype=dtype),
            "audio_cond": torch.randn(B, past + future + chunk, cond_dim * 2, device=device, dtype=dtype),
            "kp_cond": torch.randn(B, 1, 3 + 63 + 63, device=device, dtype=dtype),
            "past_times": torch.ones(B, past // chunk, 1, device=device, dtype=dtype),
        }

    def _health(self) -> dict:
        import torch

        result = {
            "ok": True,
            "app": "avtr1-pytorch-snapshot-probe",
            "probe": "avtr1_torchscript_gpu_snapshot",
            "backend": "pytorch_torchscript_pure",
            "gpu_snapshot_enabled": True,
            "has_tensorrt": False,
            "has_onnxruntime_gpu": False,
            "container_instance_id": str(getattr(self, "_container_id", None)),
            "snap_init_elapsed_s": round(getattr(self, "_snap_init_elapsed", -1), 2),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "file_size_mb": round(getattr(self, "_file_size_mb", 0), 1),
            "model_shapes": {
                "chunk_size": self._chunk_size,
                "past_size": self._past_size,
                "future_size": self._future_size,
                "nfeats": self._nfeats,
                "latent_dim": self._latent_dim,
                "cond_feature_dim": self._cond_feature_dim,
            },
        }

        if torch.cuda.is_available():
            result["cuda_available"] = True
            result["gpu_name"] = torch.cuda.get_device_name(0)
            result["vram_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
            result["vram_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**2, 1)

        return result

    @modal.method()
    def health(self) -> dict:
        return self._health()

    def _probe_avtr1_encode(self) -> dict:
        """Benchmark AVTR1 encode. Reports shapes, NaN, numeric range, latency."""
        import torch

        inputs = self._make_encode_inputs()

        print("[probe] Running AVTR1 encode (eager PyTorch)...")

        torch.cuda.synchronize()
        t0 = time.monotonic()

        with torch.no_grad():
            kp_tokens, past_context, past_last, audio_self, audio_other = \
                self.scripted.encode_conditions(
                    past_cond=inputs["past_cond"],
                    audio_cond=inputs["audio_cond"],
                    kp_cond=inputs["kp_cond"],
                    past_times=inputs["past_times"],
                )

        torch.cuda.synchronize()
        latency_ms = (time.monotonic() - t0) * 1000

        outputs = {
            "kp_tokens": kp_tokens,
            "past_context": past_context,
            "past_last": past_last,
            "audio_self": audio_self,
            "audio_other": audio_other,
        }

        checks = [_check_nan(t, n) for n, t in outputs.items()]
        any_nan = any(c["has_nan"] for c in checks)
        any_inf = any(c["has_inf"] for c in checks)

        result = {
            "ok": not any_nan and not any_inf,
            "backend": "pytorch_torchscript_eager",
            "latency_ms": round(latency_ms, 2),
            "output_checks": checks,
            "any_nan": any_nan,
            "any_inf": any_inf,
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "container_instance_id": str(self._container_id),
        }

        print(f"[probe] Encode: {latency_ms:.2f}ms, NaN={any_nan}, Inf={any_inf}")
        return result

    @modal.method()
    def probe_avtr1_encode(self) -> dict:
        return self._probe_avtr1_encode()

    def _probe_avtr1_decode(self) -> dict:
        """Benchmark AVTR1 full ODE decode loop (4 Euler steps). Reports correctness + latency."""
        import torch

        inputs = self._make_encode_inputs()

        with torch.no_grad():
            kp_tokens, past_context, past_last, audio_self, audio_other = \
                self.scripted.encode_conditions(**inputs)

        latent = self._latent_dim
        chunk = self._chunk_size
        nfeats = self._nfeats

        w_self = torch.ones(latent, device="cuda") * 2.0
        w_other = torch.ones(latent, device="cuda") * 2.0
        w_kp = torch.ones(latent, device="cuda") * 3.0

        n_ode_steps = 5
        dt = 1.0 / (n_ode_steps - 1)
        x = torch.randn(1, chunk, nfeats, device="cuda")

        print(f"[probe] Running AVTR1 decode ({n_ode_steps - 1} ODE steps, eager)...")

        torch.cuda.synchronize()
        t0 = time.monotonic()

        with torch.no_grad():
            for step_i in range(n_ode_steps - 1):
                t_val = torch.tensor([[step_i * dt]], device="cuda")
                velocity = self.scripted(
                    x=x, kp=kp_tokens, past_context=past_context,
                    past_last=past_last, audio_self=audio_self,
                    audio_other=audio_other, t=t_val,
                    self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
                )
                x = x + dt * velocity

        torch.cuda.synchronize()
        latency_ms = (time.monotonic() - t0) * 1000

        nan_check = _check_nan(x, "final_motion")
        any_nan = nan_check["has_nan"]
        any_inf = nan_check["has_inf"]

        result = {
            "ok": not any_nan and not any_inf,
            "backend": "pytorch_torchscript_eager",
            "n_ode_steps": n_ode_steps,
            "n_calls": n_ode_steps - 1,
            "latency_ms": round(latency_ms, 2),
            "output_check": nan_check,
            "any_nan": any_nan,
            "any_inf": any_inf,
            "motion_plausibility": {
                "range_ok": abs(x.max().item()) < 100 and abs(x.min().item()) < 100,
                "not_constant": (x.std().item() > 1e-6),
            },
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "container_instance_id": str(self._container_id),
        }

        print(f"[probe] Decode: {latency_ms:.2f}ms ({n_ode_steps - 1} steps), "
              f"NaN={any_nan}, range=[{x.min().item():.4f}, {x.max().item():.4f}]")
        return result

    @modal.method()
    def probe_avtr1_decode(self) -> dict:
        return self._probe_avtr1_decode()

    def _probe_avtr1_encode_warm(self, n_iters: int = 50) -> dict:
        """Warmup + multi-iteration benchmark for AVTR1 encode."""
        import torch

        inputs = self._make_encode_inputs()

        for _ in range(10):
            with torch.no_grad():
                self.scripted.encode_conditions(**inputs)

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.no_grad():
                self.scripted.encode_conditions(**inputs)
            torch.cuda.synchronize()
            latencies.append((time.monotonic() - t0) * 1000)

        latencies.sort()
        result = {
            "ok": True,
            "backend": "pytorch_torchscript_eager",
            "n_iters": n_iters,
            "warmup_iters": 10,
            "latency_mean_ms": round(sum(latencies) / len(latencies), 3),
            "latency_min_ms": round(latencies[0], 3),
            "latency_max_ms": round(latencies[-1], 3),
            "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
            "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
            "container_instance_id": str(self._container_id),
        }
        print(f"[probe] Encode warm({n_iters}): mean={result['latency_mean_ms']:.2f}ms, "
              f"p50={result['latency_p50_ms']:.2f}ms, p95={result['latency_p95_ms']:.2f}ms")
        return result

    @modal.method()
    def probe_avtr1_encode_warm(self, n_iters: int = 50) -> dict:
        return self._probe_avtr1_encode_warm(n_iters)

    def _probe_avtr1_decode_warm(self, n_iters: int = 50) -> dict:
        """Warmup + multi-iteration benchmark for AVTR1 full ODE decode."""
        import torch

        inputs = self._make_encode_inputs()

        with torch.no_grad():
            kp_tokens, past_context, past_last, audio_self, audio_other = \
                self.scripted.encode_conditions(**inputs)

        latent = self._latent_dim
        chunk = self._chunk_size
        nfeats = self._nfeats
        w_self = torch.ones(latent, device="cuda") * 2.0
        w_other = torch.ones(latent, device="cuda") * 2.0
        w_kp = torch.ones(latent, device="cuda") * 3.0
        n_ode_steps = 5
        dt = 1.0 / (n_ode_steps - 1)

        def _run_decode():
            x = torch.randn(1, chunk, nfeats, device="cuda")
            for step_i in range(n_ode_steps - 1):
                t_val = torch.tensor([[step_i * dt]], device="cuda")
                velocity = self.scripted(
                    x=x, kp=kp_tokens, past_context=past_context,
                    past_last=past_last, audio_self=audio_self,
                    audio_other=audio_other, t=t_val,
                    self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
                )
                x = x + dt * velocity
            return x

        for _ in range(10):
            with torch.no_grad():
                _run_decode()

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.monotonic()
            with torch.no_grad():
                _run_decode()
            torch.cuda.synchronize()
            latencies.append((time.monotonic() - t0) * 1000)

        latencies.sort()
        result = {
            "ok": True,
            "backend": "pytorch_torchscript_eager",
            "n_ode_steps": n_ode_steps,
            "n_calls_per_iter": n_ode_steps - 1,
            "n_iters": n_iters,
            "warmup_iters": 10,
            "latency_mean_ms": round(sum(latencies) / len(latencies), 3),
            "latency_min_ms": round(latencies[0], 3),
            "latency_max_ms": round(latencies[-1], 3),
            "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
            "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
            "container_instance_id": str(self._container_id),
        }
        print(f"[probe] Decode warm({n_iters}): mean={result['latency_mean_ms']:.2f}ms, "
              f"p50={result['latency_p50_ms']:.2f}ms, p95={result['latency_p95_ms']:.2f}ms")
        return result

    @modal.method()
    def probe_avtr1_decode_warm(self, n_iters: int = 50) -> dict:
        return self._probe_avtr1_decode_warm(n_iters)

    def _probe_compile_encode(self) -> dict:
        """Attempt torch.compile on AVTR1 encode. Non-blocking; reports failure."""
        import torch

        inputs = self._make_encode_inputs()

        print("[probe] Attempting torch.compile on encode (mode=reduce-overhead)...")

        try:
            compiled_encode = torch.compile(
                self.scripted.encode_conditions,
                mode="reduce-overhead",
            )

            print("[probe] Warmup compiled encode (3 iters)...")
            for _ in range(3):
                with torch.no_grad():
                    compiled_encode(**inputs)

            latencies = []
            for _ in range(50):
                torch.cuda.synchronize()
                t0 = time.monotonic()
                with torch.no_grad():
                    compiled_encode(**inputs)
                torch.cuda.synchronize()
                latencies.append((time.monotonic() - t0) * 1000)

            latencies.sort()
            result = {
                "ok": True,
                "backend": "pytorch_torchscript_compiled",
                "compile_mode": "reduce-overhead",
                "latency_mean_ms": round(sum(latencies) / len(latencies), 3),
                "latency_p50_ms": round(latencies[len(latencies) // 2], 3),
                "latency_p95_ms": round(latencies[int(len(latencies) * 0.95)], 3),
                "container_instance_id": str(self._container_id),
            }
            print(f"[probe] Compiled encode: mean={result['latency_mean_ms']:.2f}ms, "
                  f"p50={result['latency_p50_ms']:.2f}ms")
            return result

        except Exception as e:
            print(f"[probe] torch.compile FAILED: {type(e).__name__}: {e}")
            return {
                "ok": False,
                "backend": "pytorch_torchscript_compiled",
                "error_type": type(e).__name__,
                "error": str(e)[:500],
                "container_instance_id": str(self._container_id),
            }

    @modal.method()
    def probe_compile_encode(self) -> dict:
        return self._probe_compile_encode()

    @modal.method()
    def snapshot_probe(self) -> dict:
        wall_t0 = time.monotonic()

        health_r = self._health()
        encode_cold = self._probe_avtr1_encode()
        decode_cold = self._probe_avtr1_decode()
        encode_warm = self._probe_avtr1_encode_warm(n_iters=50)
        decode_warm = self._probe_avtr1_decode_warm(n_iters=50)
        compile_encode = self._probe_compile_encode()

        wall_elapsed = time.monotonic() - wall_t0

        return {
            "ok": True,
            "wallClock_s": round(wall_elapsed, 2),
            "health": health_r,
            "encode_cold": encode_cold,
            "decode_cold": decode_cold,
            "encode_warm": encode_warm,
            "decode_warm": decode_warm,
            "compile_encode": compile_encode,
            "container_instance_id": str(self._container_id),
        }


@app.local_entrypoint()
def main():
    """Local runner: cold call + 150s wait + 3 restore attempts."""
    import json

    cls = modal.Cls.from_name("avtr1-pytorch-snapshot-probe", "PytorchSnapshotProbe")
    inst = cls()

    results = {"restores": []}

    print("=" * 72)
    print("COLD START — calling snapshot_probe()")
    print("=" * 72)

    cold = inst.snapshot_probe.remote()
    results["cold"] = cold
    cold_cid = cold.get("container_instance_id")

    print(f"\nCold container: {cold_cid}")
    print(f"  Encode cold:  {cold['encode_cold']['latency_ms']}ms (NaN={cold['encode_cold']['any_nan']})")
    print(f"  Decode cold:  {cold['decode_cold']['latency_ms']}ms (NaN={cold['decode_cold']['any_nan']})")
    print(f"  Encode warm:  mean={cold['encode_warm']['latency_mean_ms']}ms, "
          f"p50={cold['encode_warm']['latency_p50_ms']}ms")
    print(f"  Decode warm:  mean={cold['decode_warm']['latency_mean_ms']}ms, "
          f"p50={cold['decode_warm']['latency_p50_ms']}ms")
    print(f"  Compile:      ok={cold['compile_encode']['ok']}, "
          f"latency={cold['compile_encode'].get('latency_mean_ms', 'N/A')}ms")
    print(f"  VRAM:         {cold['health'].get('vram_allocated_mb')} MB")
    print(f"  Artifact:     avtr1.scripted.pt ({cold['health'].get('file_size_mb')} MB)")

    for attempt in range(1, 4):
        wait_s = 150
        print(f"\nWaiting {wait_s}s for scale-to-zero (scaledown_window=90)...")
        time.sleep(wait_s)

        print(f"\n{'=' * 72}")
        print(f"RESTORE ATTEMPT {attempt}/3")
        print("=" * 72)

        restore = inst.snapshot_probe.remote()
        results["restores"].append(restore)
        restore_cid = restore.get("container_instance_id")
        cid_changed = restore_cid != cold_cid

        print(f"\nRestore {attempt}: container_changed={cid_changed}")
        print(f"  Container:    {restore_cid}")
        print(f"  Encode warm:  mean={restore['encode_warm']['latency_mean_ms']}ms, "
              f"p50={restore['encode_warm']['latency_p50_ms']}ms")
        print(f"  Decode warm:  mean={restore['decode_warm']['latency_mean_ms']}ms, "
              f"p50={restore['decode_warm']['latency_p50_ms']}ms")
        print(f"  VRAM:         {restore['health'].get('vram_allocated_mb')} MB")
        print(f"  NaN in encode: {restore['encode_cold']['any_nan']}")
        print(f"  NaN in decode: {restore['decode_cold']['any_nan']}")

    print(f"\n{'=' * 72}")
    print("PROBE COMPLETE")
    print()
    print("CHECK MODAL APP LOGS for evidence:")
    print("  modal app logs avtr1-pytorch-snapshot-probe")
    print()
    print("Look for:")
    print("  POSITIVE: 'Restoring Function from memory snapshot'")
    print("  NEGATIVE: 'SIGSEGV', 'exit code 139', 'Transient snapshot error',")
    print("            'retry with no snapshots', 'failed to restore'")
    print("=" * 72)
