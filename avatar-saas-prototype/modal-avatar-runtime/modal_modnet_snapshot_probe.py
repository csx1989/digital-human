"""
ISOLATED FEASIBILITY PROBE — NOT PRODUCTION
Stage 2B-1: MODNet Pure PyTorch GPU Snapshot Probe (v3)

Strategy:
  Load official MODNet PyTorch model from boopathiraj/MODNet (HuggingFace).
  Native nn.Module with bundled MobileNetV2 backbone — no onnx2torch, no timm.
  pytorch_model.bin (26.3 MB) contains all weights including backbone.
  Forward returns (semantic, detail, matte) — we take matte [-1].

  Previous attempts:
    v1 onnx2torch: 77ms B=1x5 eager — DEAD END (fx.GraphModule 7x slower than TRT)
    v2 hand-crafted + ONNX weight extraction: architecture mismatch (SEBlock 96ch vs 1280ch)
    v3: boopathiraj/MODNet native PyTorch (this file)

MODNet live-path usage:
  - renderer.py:67-72 — per-frame loop in render_chunk_streaming()
  - matting.py:27-51 — matte_chunk() resizes to 288x512, normalizes [-1,1]
  - models/matting.py — (B,3,288,512) -> (B,1,288,512) alpha matte
  - Called 5x/chunk B=1 (streaming) or 1x B=5 (batched)

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_modnet_snapshot_probe.py

Invoke (NEVER modal run):
    python -c "
import modal, time, json
cls = modal.Cls.from_name('avtr1-modnet-snapshot-probe', 'MODNetProbe')
inst = cls()
result = inst.full_probe.remote()
print(json.dumps(result, indent=2, default=str))
    "
"""

from __future__ import annotations

import time

import modal

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
        "libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1",
        "ffmpeg", "git", "gcc", "g++",
    )
    .pip_install(
        "torch>=2.8,<2.9",
        extra_index_url="https://download.pytorch.org/wl/cu128",
    )
    .pip_install(
        "numpy>=2.0,<3",
        "huggingface_hub>=0.30,<1",
        "transformers>=4.45,<5",
    )
)

hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-modnet-snapshot-probe",
    image=probe_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
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
        "mean": round(tensor.mean().item(), 6) if not has_nan else "NaN",
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


def _modnet_forward(model, x):
    _, _, matte = model(x, True)
    return matte


@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class MODNetProbe:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch
        from transformers import AutoModel

        t0 = time.time()
        print("[snap-init] === MODNet Stage 2B-1 v3 (boopathiraj/MODNet) ===")
        _run_nvidia_smi("snap-init start")

        hf_token = os.environ.get("HF_TOKEN")
        print("[snap-init] Loading boopathiraj/MODNet from HuggingFace ...")
        self.modnet = AutoModel.from_pretrained(
            "boopathiraj/MODNet",
            trust_remote_code=True,
            token=hf_token,
        )
        print(f"[snap-init] Model loaded (type: {type(self.modnet).__name__})")

        n_params = sum(p.numel() for p in self.modnet.parameters())
        n_tensors = sum(1 for _ in self.modnet.parameters())
        print(f"[snap-init] Parameters: {n_params:,} in {n_tensors} tensors")

        self.modnet = self.modnet.cuda().eval()
        print(f"[snap-init] Model on CUDA: {next(self.modnet.parameters()).device}")

        with torch.no_grad():
            dummy = torch.randn(1, 3, 288, 512, device="cuda", dtype=torch.float32)
            matte = _modnet_forward(self.modnet, dummy)
        print(f"[snap-init] Warmup: shape={tuple(matte.shape)}, "
              f"min={matte.min().item():.4f}, max={matte.max().item():.4f}")

        _run_nvidia_smi("snap-init end")
        total = time.time() - t0
        self._snap_init_s = round(total, 2)
        print(f"[snap-init] Done in {total:.2f}s")

    @modal.enter(snap=False)
    def _post_restore_init(self):
        import torch
        t0 = time.time()
        print("[restore] === Post-restore init ===")
        print(f"[restore] CUDA: {torch.cuda.is_available()}, "
              f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

        with torch.no_grad():
            dummy = torch.randn(1, 3, 288, 512, device="cuda", dtype=torch.float32)
            matte = _modnet_forward(self.modnet, dummy)
        print(f"[restore] Sanity: shape={tuple(matte.shape)}, "
              f"min={matte.min().item():.4f}, max={matte.max().item():.4f}")

        vram = torch.cuda.memory_allocated() / 1e6
        elapsed = time.time() - t0
        self._restore_vram_mb = round(vram, 1)
        self._restore_s = round(elapsed, 3)
        print(f"[restore] VRAM: {vram:.1f} MB, done in {elapsed:.3f}s")

    def _health_dict(self) -> dict:
        import torch
        return {
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_mb": round(torch.cuda.memory_allocated() / 1e6, 1),
            "snap_init_s": getattr(self, "_snap_init_s", None),
            "restore_s": getattr(self, "_restore_s", None),
        }

    @modal.method()
    def full_probe(self) -> dict:
        import torch

        print("[probe] === Stage 2B-1 MODNet Full Probe (v3 boopathiraj/MODNet) ===")
        print(f"[probe] Health: {self._health_dict()}")
        _run_nvidia_smi("probe start")

        results = {
            "stage": "2B-1",
            "component": "MODNet",
            "approach": "boopathiraj_MODNet_native_pytorch",
            "health": self._health_dict(),
            "checks": {},
            "benchmarks": {},
        }

        H, W = 288, 512
        N_WARMUP = 10
        N_MEASURED = 100

        torch.manual_seed(42)
        inp_b1 = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float32)
        inp_b5 = torch.randn(5, 3, H, W, device="cuda", dtype=torch.float32)

        # --- Correctness ---
        print("[probe] === Correctness checks ===")
        with torch.no_grad():
            out_b1 = _modnet_forward(self.modnet, inp_b1)
            out_b5 = _modnet_forward(self.modnet, inp_b5)

        results["checks"]["b1_fp32"] = _check_nan(out_b1, "modnet_b1_fp32")
        results["checks"]["b5_fp32"] = _check_nan(out_b5, "modnet_b5_fp32")
        print(f"[probe] B=1: {results['checks']['b1_fp32']}")
        print(f"[probe] B=5: {results['checks']['b5_fp32']}")

        assert out_b1.shape == (1, 1, H, W), f"B=1 shape: {out_b1.shape}"
        assert out_b5.shape == (5, 1, H, W), f"B=5 shape: {out_b5.shape}"

        # fp16
        modnet_h = self.modnet.half()
        with torch.no_grad():
            out_b1_h = _modnet_forward(modnet_h, inp_b1.half())
            out_b5_h = _modnet_forward(modnet_h, inp_b5.half())
        results["checks"]["b1_fp16"] = _check_nan(out_b1_h, "modnet_b1_fp16")
        results["checks"]["b5_fp16"] = _check_nan(out_b5_h, "modnet_b5_fp16")
        diff = (out_b1.float() - out_b1_h.float()).abs()
        results["checks"]["fp16_vs_fp32_mae"] = round(diff.mean().item(), 6)
        results["checks"]["fp16_vs_fp32_max"] = round(diff.max().item(), 6)
        print(f"[probe] fp16 MAE={results['checks']['fp16_vs_fp32_mae']}, max={results['checks']['fp16_vs_fp32_max']}")
        self.modnet = self.modnet.float()
        del modnet_h

        # --- Benchmarks ---
        bench_configs = [
            ("b1x5_eager_fp32", inp_b1, 1, 5, torch.float32),
            ("b5_eager_fp32", inp_b5, 5, 1, torch.float32),
            ("b1_eager_fp32", inp_b1, 1, 1, torch.float32),
            ("b1x5_eager_fp16", inp_b1, 1, 5, torch.float16),
            ("b5_eager_fp16", inp_b5, 5, 1, torch.float16),
        ]

        for bench_name, inp, batch, calls, dtype in bench_configs:
            print(f"[probe] === {bench_name} ===")
            model = self.modnet.half() if dtype == torch.float16 else self.modnet
            inp_t = inp.half() if dtype == torch.float16 else inp

            with torch.no_grad():
                for _ in range(N_WARMUP):
                    for _ in range(calls):
                        _modnet_forward(model, inp_t)
                torch.cuda.synchronize()

                lats = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    for _ in range(calls):
                        _modnet_forward(model, inp_t)
                    t1e.record()
                    torch.cuda.synchronize()
                    lats.append(t0e.elapsed_time(t1e))

            results["benchmarks"][bench_name] = _stats_ms(lats)
            per_frame = results["benchmarks"][bench_name]["p50_ms"] / (batch * calls)
            results["benchmarks"][bench_name]["per_frame_ms"] = round(per_frame, 3)
            results["benchmarks"][bench_name]["budget_pct"] = round(
                results["benchmarks"][bench_name]["p50_ms"] / 200 * 100, 1
            )
            print(f"[probe] {bench_name}: p50={results['benchmarks'][bench_name]['p50_ms']:.3f}ms "
                  f"({per_frame:.3f}ms/frame) budget={results['benchmarks'][bench_name]['budget_pct']:.1f}%")

            if dtype == torch.float16:
                self.modnet = self.modnet.float()

        # --- CUDA graph: B=1 replay 5x ---
        print("[probe] === CUDA graph B=1 x5 ===")
        try:
            static_in = inp_b1.clone()
            static_out = torch.empty(1, 1, H, W, device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out.copy_(_modnet_forward(self.modnet, static_in))

            for _ in range(N_WARMUP):
                g.replay()
            torch.cuda.synchronize()

            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                for _ in range(5):
                    g.replay()
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))

            results["benchmarks"]["b1x5_cuda_graph_fp32"] = _stats_ms(lats)
            results["benchmarks"]["b1x5_cuda_graph_fp32"]["per_frame_ms"] = round(
                results["benchmarks"]["b1x5_cuda_graph_fp32"]["p50_ms"] / 5, 3
            )
            results["benchmarks"]["b1x5_cuda_graph_fp32"]["budget_pct"] = round(
                results["benchmarks"]["b1x5_cuda_graph_fp32"]["p50_ms"] / 200 * 100, 1
            )
            print(f"[probe] b1x5_cuda_graph_fp32: p50={results['benchmarks']['b1x5_cuda_graph_fp32']['p50_ms']:.3f}ms "
                  f"budget={results['benchmarks']['b1x5_cuda_graph_fp32']['budget_pct']:.1f}%")
            del g
        except Exception as e:
            print(f"[probe] CUDA graph FAILED: {e}")
            results["benchmarks"]["b1x5_cuda_graph_fp32"] = {"error": str(e)}

        # --- Budget ---
        best_key = min(
            [k for k in ["b1x5_eager_fp32", "b5_eager_fp32", "b1x5_cuda_graph_fp32"]
             if k in results["benchmarks"] and "error" not in results["benchmarks"].get(k, {})],
            key=lambda k: results["benchmarks"][k]["p50_ms"]
        )
        best_ms = results["benchmarks"][best_key]["p50_ms"]
        results["budget"] = {
            "avtr_encode_decode_ms": 27,
            "renderer_phase2a_ms": "122-128",
            "modnet_best_ms": round(best_ms, 3),
            "modnet_best_path": best_key,
            "estimated_total_without_hubert_ms": round(27 + 125 + best_ms, 1),
            "budget_200ms_pct": round((27 + 125 + best_ms) / 200 * 100, 1),
            "headroom_without_hubert_ms": round(200 - 27 - 125 - best_ms, 1),
        }
        print(f"[probe] === Budget: MODNet {best_key} = {best_ms:.3f}ms, "
              f"total (no HuBERT) ~ {results['budget']['estimated_total_without_hubert_ms']}ms ===")

        _run_nvidia_smi("probe end")
        return results

    @modal.method()
    def health_check(self) -> dict:
        return self._health_dict()
