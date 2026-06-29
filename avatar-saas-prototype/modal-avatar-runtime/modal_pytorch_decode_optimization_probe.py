"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ISOLATED FEASIBILITY PROBE — NOT PRODUCTION                              ║
║  Phase 1.5: AVTR1 Decode Optimization + GPU Snapshot Probe                ║
║                                                                            ║
║  Goal: Determine whether optimized PyTorch AVTR1 decode can meet the      ║
║  200ms per-chunk budget, and whether optimizations survive GPU snapshot    ║
║  restore on L40S.                                                          ║
║                                                                            ║
║  This is a SEPARATE Modal app. It does NOT touch:                         ║
║    - Production L40S lifecycle app                                         ║
║    - Phase 1 snapshot probe (avtr1-pytorch-snapshot-probe)                 ║
║    - Bridge / Web client / wrangler flags / routing                        ║
║                                                                            ║
║  DO NOT wire into production routing.                                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Decode variants benchmarked:
  A: baseline_eager        — naive loop, allocations per step
  B: static_prealloc       — production-style, pre-allocated tensors, in-place ops
  C: compile_reduce_overhead — torch.compile(mode="reduce-overhead") on full loop
  D: compile_max_autotune  — torch.compile(mode="max-autotune") on full loop (optional)
  E: cuda_graph            — torch.cuda.CUDAGraph capture of full 4-step loop (optional)

Budget:
  - AVTR decode runs once per 5-frame chunk (~200ms media duration).
  - Target: <=30ms decode = <=15% of chunk budget.

Snapshot verification:
  - Cold start + 3 restores with 60s waits (scaledown_window=30).
  - Modal logs checked for real restore evidence, not just snap_ok=True.

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_decode_optimization_probe.py

Invoke:
    modal run avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_decode_optimization_probe.py
"""

from __future__ import annotations

import gc
import time
from pathlib import Path

import modal

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
        extra_index_url="https://download.pytorch.org/wl/cu128",
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

probe_volume = modal.Volume.from_name("avtr1-pytorch-probe-artifacts", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-decode-optimization-probe",
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
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": round(tensor.min().item(), 6) if not has_nan else "NaN",
        "max": round(tensor.max().item(), 6) if not has_nan else "NaN",
    }


def _stats_ms(latencies_ms: list[float]) -> dict:
    s = sorted(latencies_ms)
    n = len(s)
    return {
        "n": n,
        "mean_ms": round(sum(s) / n, 3),
        "min_ms": round(s[0], 3),
        "max_ms": round(s[-1], 3),
        "p50_ms": round(s[n // 2], 3),
        "p95_ms": round(s[int(n * 0.95)], 3),
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
    timeout=900,
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class DecodeOptProbe:

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import torch

        self._snap_init_t0 = time.monotonic()
        print("[snap-init] === Decode Optimization Probe — Snap Init ===")

        from huggingface_hub import hf_hub_download

        t0 = time.monotonic()
        ckpt_path = hf_hub_download(
            repo_id="avaturn-live/avtr-1",
            filename="build_artifacts/avtr1.scripted.pt",
            repo_type="model",
            cache_dir=str(Path(HF_CACHE_PATH) / "hub"),
        )
        download_s = time.monotonic() - t0
        ckpt_path = Path(ckpt_path)
        print(f"[snap-init] Downloaded in {download_s:.1f}s ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

        t0 = time.monotonic()
        self.scripted = torch.jit.load(str(ckpt_path), map_location="cuda").eval()
        load_s = time.monotonic() - t0
        print(f"[snap-init] TorchScript loaded in {load_s:.1f}s")

        inner = self.scripted.model
        self.chunk_size = int(inner.chunk_size)
        self.past_size = int(inner.past_size)
        self.future_size = int(inner.future_size)
        self.nfeats = int(inner.nfeats)
        self.cond_feature_dim = int(inner.cond_feature_dim)
        self.latent_dim = int(inner.latent_dim)
        self.n_ode_steps = 5

        print(f"[snap-init] chunk={self.chunk_size} past={self.past_size} "
              f"nfeats={self.nfeats} latent={self.latent_dim} ode_steps={self.n_ode_steps}")

        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[snap-init] VRAM: {vram:.0f} MB")
        _run_nvidia_smi("snap-init")

        self._file_size_mb = ckpt_path.stat().st_size / 1e6
        self._download_s = download_s
        self._load_s = load_s
        self._container_id = id(self)
        self._snap_init_elapsed = time.monotonic() - self._snap_init_t0
        print(f"[snap-init] Done in {self._snap_init_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        self._restore_t0 = time.monotonic()
        print("[restore] === Post-restore init ===")
        print(f"[restore] CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM: {vram:.0f} MB")
        _run_nvidia_smi("restore")

        if not hasattr(self, "scripted") or self.scripted is None:
            raise RuntimeError("TorchScript model NOT found after restore — snapshot FAILED")

        self._container_id = id(self)
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    def _encode_inputs(self) -> dict:
        import torch
        B = 1
        return {
            "past_cond": torch.randn(B, self.past_size, self.nfeats, device="cuda"),
            "audio_cond": torch.randn(B, self.past_size + self.future_size + self.chunk_size,
                                       self.cond_feature_dim * 2, device="cuda"),
            "kp_cond": torch.randn(B, 1, 3 + 63 + 63, device="cuda"),
            "past_times": torch.ones(B, self.past_size // self.chunk_size, 1, device="cuda"),
        }

    def _run_encode(self, inputs: dict):
        import torch
        with torch.inference_mode():
            return self.scripted.encode_conditions(
                past_cond=inputs["past_cond"],
                audio_cond=inputs["audio_cond"],
                kp_cond=inputs["kp_cond"],
                past_times=inputs["past_times"],
            )

    def _health(self) -> dict:
        import torch
        return {
            "ok": True,
            "app": "avtr1-decode-optimization-probe",
            "file_size_mb": round(self._file_size_mb, 1),
            "snap_init_s": round(self._snap_init_elapsed, 2),
            "restore_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "container_id": str(self._container_id),
        }

    # ----------------------------------------------------------------
    # Variant A: baseline eager — allocations per step (Phase 1 style)
    # ----------------------------------------------------------------
    def _variant_a_baseline(self, enc, n_warmup: int = 10, n_iters: int = 100) -> dict:
        import torch

        kp_tokens, past_context, past_last, audio_self, audio_other = enc
        latent = self.latent_dim
        chunk = self.chunk_size
        nfeats = self.nfeats
        n_steps = self.n_ode_steps - 1
        dt = 1.0 / (self.n_ode_steps - 1)

        w_self = torch.ones(latent, device="cuda") * 2.0
        w_other = torch.ones(latent, device="cuda") * 2.0
        w_kp = torch.ones(latent, device="cuda") * 3.0

        def _run():
            x = torch.randn(1, chunk, nfeats, device="cuda")
            for i in range(n_steps):
                t_val = torch.tensor([[i * dt]], device="cuda")
                velocity = self.scripted(
                    x=x, kp=kp_tokens, past_context=past_context,
                    past_last=past_last, audio_self=audio_self,
                    audio_other=audio_other, t=t_val,
                    self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
                )
                x = x + dt * velocity
            return x

        for _ in range(n_warmup):
            with torch.inference_mode():
                _run()

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = _run()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        check = _check_nan(out, "variant_a")
        stats = _stats_ms(latencies)
        stats["nan_free"] = not check["has_nan"]
        stats["inf_free"] = not check["has_inf"]
        stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        stats["budget_pct"] = round(stats["p50_ms"] / 200.0 * 100, 1)
        return stats

    # ----------------------------------------------------------------
    # Variant B: static preallocation — production-style
    # ----------------------------------------------------------------
    def _variant_b_static_prealloc(self, enc, n_warmup: int = 10, n_iters: int = 100) -> dict:
        import torch

        kp_tokens, past_context, past_last, audio_self, audio_other = enc
        chunk = self.chunk_size
        nfeats = self.nfeats
        latent = self.latent_dim
        n_steps = self.n_ode_steps - 1
        dt = 1.0 / (self.n_ode_steps - 1)

        times = torch.linspace(0.0, 1.0, self.n_ode_steps, device="cuda").view(-1, 1, 1)
        w_self = torch.full((latent,), 2.0, device="cuda")
        w_other = torch.full((latent,), 2.0, device="cuda")
        w_kp = torch.full((latent,), 3.0, device="cuda")
        x_buf = torch.empty(1, chunk, nfeats, device="cuda")

        def _run():
            x_buf.normal_()
            for i in range(n_steps):
                velocity = self.scripted(
                    x=x_buf, kp=kp_tokens, past_context=past_context,
                    past_last=past_last, audio_self=audio_self,
                    audio_other=audio_other, t=times[i],
                    self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
                )
                x_buf.add_(velocity, alpha=dt)
            return x_buf.clone()

        for _ in range(n_warmup):
            with torch.inference_mode():
                _run()

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = _run()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        check = _check_nan(out, "variant_b")
        stats = _stats_ms(latencies)
        stats["nan_free"] = not check["has_nan"]
        stats["inf_free"] = not check["has_inf"]
        stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        stats["budget_pct"] = round(stats["p50_ms"] / 200.0 * 100, 1)
        return stats

    # ----------------------------------------------------------------
    # Variant C: torch.compile(mode="reduce-overhead") on full ODE loop
    # ----------------------------------------------------------------
    def _variant_c_compile(self, enc, n_warmup: int = 10, n_iters: int = 100) -> dict:
        import torch

        kp_tokens, past_context, past_last, audio_self, audio_other = enc
        chunk = self.chunk_size
        nfeats = self.nfeats
        latent = self.latent_dim
        n_steps = self.n_ode_steps - 1
        dt = 1.0 / (self.n_ode_steps - 1)

        scripted = self.scripted

        def _ode_loop(x, kp, pc, pl, a_s, a_o, w_s, w_o, w_k, times, n, dt_val):
            for i in range(n):
                v = scripted(
                    x=x, kp=kp, past_context=pc, past_last=pl,
                    audio_self=a_s, audio_other=a_o, t=times[i],
                    self_weights=w_s, other_weights=w_o, kp_weights=w_k,
                )
                x = x + dt_val * v
            return x

        print("[variant-c] Compiling with mode=reduce-overhead...")
        compile_t0 = time.monotonic()
        try:
            compiled_loop = torch.compile(_ode_loop, mode="reduce-overhead")
        except Exception as e:
            print(f"[variant-c] compile FAILED: {e}")
            return {"ok": False, "error": str(e)[:300]}
        compile_s = time.monotonic() - compile_t0
        print(f"[variant-c] torch.compile() returned in {compile_s:.1f}s (first call will JIT)")

        times_t = torch.linspace(0.0, 1.0, self.n_ode_steps, device="cuda").view(-1, 1, 1)
        w_self = torch.full((latent,), 2.0, device="cuda")
        w_other = torch.full((latent,), 2.0, device="cuda")
        w_kp = torch.full((latent,), 3.0, device="cuda")

        def _run():
            x = torch.randn(1, chunk, nfeats, device="cuda")
            return compiled_loop(
                x, kp_tokens, past_context, past_last, audio_self, audio_other,
                w_self, w_other, w_kp, times_t, n_steps, dt,
            )

        print("[variant-c] Warming up (includes JIT compilation)...")
        jit_t0 = time.monotonic()
        for i in range(n_warmup):
            with torch.inference_mode():
                _run()
        jit_s = time.monotonic() - jit_t0
        print(f"[variant-c] Warmup done in {jit_s:.1f}s (includes JIT + possible graph breaks)")

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = _run()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        check = _check_nan(out, "variant_c")
        stats = _stats_ms(latencies)
        stats["ok"] = True
        stats["compile_wall_s"] = round(compile_s, 1)
        stats["jit_warmup_s"] = round(jit_s, 1)
        stats["nan_free"] = not check["has_nan"]
        stats["inf_free"] = not check["has_inf"]
        stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        stats["budget_pct"] = round(stats["p50_ms"] / 200.0 * 100, 1)
        return stats

    # ----------------------------------------------------------------
    # Variant D: torch.compile(mode="max-autotune") on full ODE loop
    # ----------------------------------------------------------------
    def _variant_d_max_autotune(self, enc, n_warmup: int = 10, n_iters: int = 100) -> dict:
        import torch

        kp_tokens, past_context, past_last, audio_self, audio_other = enc
        chunk = self.chunk_size
        nfeats = self.nfeats
        latent = self.latent_dim
        n_steps = self.n_ode_steps - 1
        dt = 1.0 / (self.n_ode_steps - 1)

        scripted = self.scripted

        def _ode_loop(x, kp, pc, pl, a_s, a_o, w_s, w_o, w_k, times, n, dt_val):
            for i in range(n):
                v = scripted(
                    x=x, kp=kp, past_context=pc, past_last=pl,
                    audio_self=a_s, audio_other=a_o, t=times[i],
                    self_weights=w_s, other_weights=w_o, kp_weights=w_k,
                )
                x = x + dt_val * v
            return x

        print("[variant-d] Compiling with mode=max-autotune...")
        compile_t0 = time.monotonic()
        try:
            compiled_loop = torch.compile(_ode_loop, mode="max-autotune")
        except Exception as e:
            print(f"[variant-d] compile FAILED: {e}")
            return {"ok": False, "error": str(e)[:300]}
        compile_s = time.monotonic() - compile_t0
        print(f"[variant-d] torch.compile() returned in {compile_s:.1f}s")

        if compile_s > 300:
            print(f"[variant-d] SKIP: compile took {compile_s:.0f}s > 300s budget")
            return {"ok": False, "skipped": True, "reason": f"compile {compile_s:.0f}s > 300s budget"}

        times_t = torch.linspace(0.0, 1.0, self.n_ode_steps, device="cuda").view(-1, 1, 1)
        w_self = torch.full((latent,), 2.0, device="cuda")
        w_other = torch.full((latent,), 2.0, device="cuda")
        w_kp = torch.full((latent,), 3.0, device="cuda")

        def _run():
            x = torch.randn(1, chunk, nfeats, device="cuda")
            return compiled_loop(
                x, kp_tokens, past_context, past_last, audio_self, audio_other,
                w_self, w_other, w_kp, times_t, n_steps, dt,
            )

        print("[variant-d] Warming up (includes autotuning + JIT)...")
        jit_t0 = time.monotonic()
        for i in range(n_warmup):
            with torch.inference_mode():
                try:
                    _run()
                except Exception as e:
                    print(f"[variant-d] warmup iter {i} FAILED: {e}")
                    return {"ok": False, "error": str(e)[:300]}
        jit_s = time.monotonic() - jit_t0
        print(f"[variant-d] Warmup done in {jit_s:.1f}s")

        latencies = []
        for _ in range(n_iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                out = _run()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        check = _check_nan(out, "variant_d")
        stats = _stats_ms(latencies)
        stats["ok"] = True
        stats["compile_wall_s"] = round(compile_s, 1)
        stats["jit_warmup_s"] = round(jit_s, 1)
        stats["nan_free"] = not check["has_nan"]
        stats["inf_free"] = not check["has_inf"]
        stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        stats["budget_pct"] = round(stats["p50_ms"] / 200.0 * 100, 1)
        return stats

    # ----------------------------------------------------------------
    # Variant E: CUDA graph capture of full 4-step loop
    # ----------------------------------------------------------------
    def _variant_e_cuda_graph(self, enc, n_warmup: int = 10, n_iters: int = 100) -> dict:
        import torch

        kp_tokens, past_context, past_last, audio_self, audio_other = enc
        chunk = self.chunk_size
        nfeats = self.nfeats
        latent = self.latent_dim
        n_steps = self.n_ode_steps - 1
        dt = 1.0 / (self.n_ode_steps - 1)

        static_x = torch.randn(1, chunk, nfeats, device="cuda")
        times_t = torch.linspace(0.0, 1.0, self.n_ode_steps, device="cuda").view(-1, 1, 1)
        w_self = torch.full((latent,), 2.0, device="cuda")
        w_other = torch.full((latent,), 2.0, device="cuda")
        w_kp = torch.full((latent,), 3.0, device="cuda")
        scripted = self.scripted

        def _captured_loop():
            for i in range(n_steps):
                v = scripted(
                    x=static_x, kp=kp_tokens, past_context=past_context,
                    past_last=past_last, audio_self=audio_self,
                    audio_other=audio_other, t=times_t[i],
                    self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
                )
                static_x.add_(v, alpha=dt)

        print("[variant-e] Warming up on side stream...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(n_warmup):
                with torch.inference_mode():
                    static_x.normal_()
                    _captured_loop()
        torch.cuda.current_stream().wait_stream(s)

        print("[variant-e] Capturing CUDA graph...")
        try:
            g = torch.cuda.CUDAGraph()
            static_x.normal_()
            with torch.inference_mode(), torch.cuda.graph(g):
                _captured_loop()
            print("[variant-e] Graph captured successfully")
        except Exception as e:
            print(f"[variant-e] CUDA graph capture FAILED: {e}")
            return {"ok": False, "error": str(e)[:300]}

        latencies = []
        for _ in range(n_iters):
            static_x.normal_()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        check = _check_nan(static_x, "variant_e")
        stats = _stats_ms(latencies)
        stats["ok"] = True
        stats["nan_free"] = not check["has_nan"]
        stats["inf_free"] = not check["has_inf"]
        stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
        stats["budget_pct"] = round(stats["p50_ms"] / 200.0 * 100, 1)
        return stats

    # ----------------------------------------------------------------
    # Full benchmark runner
    # ----------------------------------------------------------------
    def _run_all_benchmarks(self) -> dict:
        import torch

        inputs = self._encode_inputs()
        enc = self._run_encode(inputs)
        kp_tokens, past_context, past_last, audio_self, audio_other = enc

        results = {}

        print("\n[bench] === Variant A: baseline eager ===")
        t0 = time.monotonic()
        results["A_baseline"] = self._variant_a_baseline(enc)
        print(f"[bench] A: p50={results['A_baseline']['p50_ms']}ms "
              f"budget={results['A_baseline']['budget_pct']}% "
              f"({time.monotonic()-t0:.1f}s)")

        print("\n[bench] === Variant B: static prealloc ===")
        t0 = time.monotonic()
        results["B_static"] = self._variant_b_static_prealloc(enc)
        print(f"[bench] B: p50={results['B_static']['p50_ms']}ms "
              f"budget={results['B_static']['budget_pct']}% "
              f"({time.monotonic()-t0:.1f}s)")

        print("\n[bench] === Variant C: compile reduce-overhead ===")
        t0 = time.monotonic()
        results["C_compile_ro"] = self._variant_c_compile(enc)
        c_time = time.monotonic() - t0
        if results["C_compile_ro"].get("ok"):
            print(f"[bench] C: p50={results['C_compile_ro']['p50_ms']}ms "
                  f"budget={results['C_compile_ro']['budget_pct']}% ({c_time:.1f}s)")
        else:
            print(f"[bench] C: FAILED — {results['C_compile_ro'].get('error', 'unknown')} ({c_time:.1f}s)")

        print("\n[bench] === Variant D: compile max-autotune ===")
        t0 = time.monotonic()
        results["D_compile_ma"] = self._variant_d_max_autotune(enc)
        d_time = time.monotonic() - t0
        if results["D_compile_ma"].get("ok"):
            print(f"[bench] D: p50={results['D_compile_ma']['p50_ms']}ms "
                  f"budget={results['D_compile_ma']['budget_pct']}% ({d_time:.1f}s)")
        elif results["D_compile_ma"].get("skipped"):
            print(f"[bench] D: SKIPPED — {results['D_compile_ma'].get('reason', '')} ({d_time:.1f}s)")
        else:
            print(f"[bench] D: FAILED — {results['D_compile_ma'].get('error', 'unknown')} ({d_time:.1f}s)")

        print("\n[bench] === Variant E: CUDA graph ===")
        t0 = time.monotonic()
        results["E_cuda_graph"] = self._variant_e_cuda_graph(enc)
        e_time = time.monotonic() - t0
        if results["E_cuda_graph"].get("ok"):
            print(f"[bench] E: p50={results['E_cuda_graph']['p50_ms']}ms "
                  f"budget={results['E_cuda_graph']['budget_pct']}% ({e_time:.1f}s)")
        else:
            print(f"[bench] E: FAILED — {results['E_cuda_graph'].get('error', 'unknown')} ({e_time:.1f}s)")

        gc.collect()
        torch.cuda.empty_cache()

        return results

    @modal.method()
    def run_benchmarks(self) -> dict:
        health = self._health()
        benchmarks = self._run_all_benchmarks()
        return {
            "health": health,
            "benchmarks": benchmarks,
        }

    @modal.method()
    def full_probe(self) -> dict:
        wall_t0 = time.monotonic()
        health = self._health()
        benchmarks = self._run_all_benchmarks()
        wall_s = time.monotonic() - wall_t0
        return {
            "ok": True,
            "wall_s": round(wall_s, 1),
            "health": health,
            "benchmarks": benchmarks,
        }


@app.local_entrypoint()
def main():
    import json

    cls = modal.Cls.from_name("avtr1-decode-optimization-probe", "DecodeOptProbe")
    inst = cls()

    results = {"restores": []}

    print("=" * 72)
    print("COLD START — full_probe()")
    print("=" * 72)
    cold = inst.full_probe.remote()
    results["cold"] = cold
    cold_cid = cold["health"]["container_id"]

    _print_benchmarks("COLD", cold["benchmarks"])

    for attempt in range(1, 4):
        print(f"\nWaiting 60s for scale-to-zero (scaledown_window=30)...")
        time.sleep(60)

        print(f"\n{'=' * 72}")
        print(f"RESTORE {attempt}/3")
        print("=" * 72)

        r = inst.full_probe.remote()
        results["restores"].append(r)
        _print_benchmarks(f"RESTORE {attempt}", r["benchmarks"])

    print(f"\n{'=' * 72}")
    print("PROBE COMPLETE — check Modal logs:")
    print("  modal app logs avtr1-decode-optimization-probe")
    print()
    print("Look for:")
    print("  POSITIVE: 'Restoring Function from memory snapshot'")
    print("  NEGATIVE: 'SIGSEGV', 'exit code 139', 'Transient snapshot error',")
    print("            'retry with no snapshots', 'failed to restore'")
    print("=" * 72)


def _print_benchmarks(label: str, b: dict) -> None:
    print(f"\n  {label} benchmark results:")
    for key, v in b.items():
        if v.get("ok"):
            print(f"    {key}: p50={v['p50_ms']}ms p95={v['p95_ms']}ms "
                  f"min={v['min_ms']}ms max={v['max_ms']}ms "
                  f"budget={v['budget_pct']}% NaN-free={v['nan_free']}")
        elif v.get("skipped"):
            print(f"    {key}: SKIPPED — {v.get('reason', '')}")
        else:
            print(f"    {key}: FAILED — {v.get('error', 'unknown')[:100]}")
