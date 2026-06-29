"""
ISOLATED FEASIBILITY PROBE — NOT PRODUCTION
Stage 2B-2: HuBERT Pure PyTorch GPU Snapshot Probe

Strategy:
  Convert hubert-lbs-avtr1.onnx to PyTorch via onnx2torch.
  ONNX has 149/414 anonymous initializers (onnx::MatMul_*, onnx::Conv_*),
  making weight extraction to native HF HubertModel infeasible.
  onnx2torch preserves exact ONNX computation including anonymous weights.

Architecture (confirmed by ONNX inspection):
  Hubert-Large: 24 layers, 1024 hidden, 16 heads, 4096 FFN, conv_bias=True
  Output: last_hidden_state (B, frames, 1024)
  Input: input_values (B, sequence_length) fp32 audio

Production usage:
  avtr1_motion_generator.py:404-411 — audio_batch (2, 8400), n_motion_frames=10
  hubert.py — run_hubert() → last_hidden_state → 50→25Hz downsample
  Output after downsample: (B, n_motion_frames, 1024)

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_hubert_snapshot_probe.py

Invoke (NEVER modal run):
    python -c "
import modal, time, json
cls = modal.Cls.from_name('avtr1-hubert-snapshot-probe', 'HuBERTProbe')
inst = cls()
result = inst.full_probe.remote()
print(json.dumps(result, indent=2, default=str))
    "

Parity validation (one-shot, separate method):
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-hubert-snapshot-probe', 'HuBERTProbe')
inst = cls()
result = inst.validate_parity.remote()
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
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        "numpy>=2.0,<3",
        "huggingface_hub>=0.30,<1",
        "onnx>=1.17,<2",
        "onnx2torch>=1.5,<2",
        "onnxruntime>=1.17,<2",
    )
)

hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-hubert-snapshot-probe",
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


@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=900,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class HuBERTProbe:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch
        from huggingface_hub import hf_hub_download
        from onnx2torch import convert

        t0 = time.time()
        print("[snap-init] === HuBERT Stage 2B-2 (onnx2torch conversion) ===")
        _run_nvidia_smi("snap-init start")

        HF_REPO = "avaturn-live/avtr-1"
        HF_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"

        hf_token = os.environ.get("HF_TOKEN")
        print(f"[snap-init] Downloading {HF_FILE} from {HF_REPO} ...")
        onnx_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            token=hf_token,
            cache_dir=HF_CACHE_PATH,
        )
        print(f"[snap-init] Downloaded: {onnx_path}")

        file_size = os.path.getsize(onnx_path) / 1e6
        print(f"[snap-init] ONNX file size: {file_size:.1f} MB")

        print("[snap-init] Pre-processing ONNX to fix missing Conv biases ...")
        import onnx
        from onnx import numpy_helper, TensorProto
        import numpy as np

        onnx_model = onnx.load(onnx_path)
        initializer_names = {init.name for init in onnx_model.graph.initializer}
        fixed_count = 0
        for node in onnx_model.graph.node:
            if node.op_type == "Conv" and len(node.input) >= 3:
                bias_name = node.input[2]
                if bias_name and bias_name not in initializer_names:
                    weight_name = node.input[1]
                    weight_init = next(
                        (i for i in onnx_model.graph.initializer if i.name == weight_name), None
                    )
                    if weight_init is not None:
                        out_channels = weight_init.dims[0]
                        zero_bias = numpy_helper.from_array(
                            np.zeros(out_channels, dtype=np.float32), name=bias_name
                        )
                        onnx_model.graph.initializer.append(zero_bias)
                        initializer_names.add(bias_name)
                        fixed_count += 1
                        print(f"[snap-init]   Added zero bias '{bias_name}' shape=[{out_channels}]")
        print(f"[snap-init] Fixed {fixed_count} Conv nodes with missing biases")

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(onnx_model, f.name)
            fixed_onnx_path = f.name
        print(f"[snap-init] Saved fixed ONNX to {fixed_onnx_path}")

        print("[snap-init] Converting ONNX → PyTorch via onnx2torch ...")
        t_conv = time.time()
        self.model = convert(fixed_onnx_path)
        conv_time = time.time() - t_conv
        print(f"[snap-init] Conversion done in {conv_time:.1f}s")
        print(f"[snap-init] Model type: {type(self.model).__name__}")

        n_params = sum(p.numel() for p in self.model.parameters())
        n_buffers = sum(b.numel() for b in self.model.buffers())
        print(f"[snap-init] Parameters: {n_params:,}, Buffers: {n_buffers:,}")

        self.model = self.model.cuda().eval()
        print(f"[snap-init] Model on CUDA")

        with torch.no_grad():
            dummy = torch.randn(2, 8400, device="cuda", dtype=torch.float32)
            output = self.model(dummy)
        if isinstance(output, (tuple, list)):
            output = output[0]
        print(f"[snap-init] Warmup: shape={tuple(output.shape)}, "
              f"min={output.min().item():.4f}, max={output.max().item():.4f}")

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
            dummy = torch.randn(2, 8400, device="cuda", dtype=torch.float32)
            output = self.model(dummy)
        if isinstance(output, (tuple, list)):
            output = output[0]
        print(f"[restore] Sanity: shape={tuple(output.shape)}, "
              f"min={output.min().item():.4f}, max={output.max().item():.4f}")

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
    def validate_parity(self) -> dict:
        import os
        import numpy as np
        import torch
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        HF_REPO = "avaturn-live/avtr-1"
        HF_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"

        print("[parity] === HuBERT Parity Validation ===")

        hf_token = os.environ.get("HF_TOKEN")
        onnx_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            token=hf_token,
            cache_dir=HF_CACHE_PATH,
        )

        print("[parity] Loading ONNX via onnxruntime CPU ...")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_name = sess.get_inputs()[0].name
        onnx_output_name = sess.get_outputs()[0].name

        torch.manual_seed(42)
        np.random.seed(42)

        onnx_batch = 1
        test_audio_b1 = np.random.randn(onnx_batch, 8400).astype(np.float32)
        test_tensor_b2 = torch.from_numpy(
            np.concatenate([test_audio_b1, np.random.randn(1, 8400).astype(np.float32)])
        )

        print("[parity] Running ONNX inference B=1 (CPU) ...")
        onnx_out = sess.run([onnx_output_name], {onnx_input_name: test_audio_b1})[0]
        print(f"[parity] ONNX output: shape={onnx_out.shape}, "
              f"range=[{onnx_out.min():.6f}, {onnx_out.max():.6f}]")

        print("[parity] Running PyTorch inference B=1 (CPU) ...")
        self.model.cpu()
        with torch.no_grad():
            pt_out_b1 = self.model(torch.from_numpy(test_audio_b1))
        if isinstance(pt_out_b1, (tuple, list)):
            pt_out_b1 = pt_out_b1[0]
        pt_np = pt_out_b1.numpy()
        print(f"[parity] PyTorch output: shape={pt_np.shape}, "
              f"range=[{pt_np.min():.6f}, {pt_np.max():.6f}]")

        self.model.cuda()

        abs_err = np.abs(onnx_out - pt_np)
        mae = float(abs_err.mean())
        max_err = float(abs_err.max())
        flat_err = abs_err.flatten()
        p50 = float(np.percentile(flat_err, 50))
        p95 = float(np.percentile(flat_err, 95))
        p99 = float(np.percentile(flat_err, 99))

        onnx_flat = onnx_out.flatten()
        pt_flat = pt_np.flatten()
        cos_sim = float(np.dot(onnx_flat, pt_flat) / (np.linalg.norm(onnx_flat) * np.linalg.norm(pt_flat) + 1e-10))

        onnx_nan = bool(np.isnan(onnx_out).any())
        pt_nan = bool(np.isnan(pt_np).any())

        results = {
            "input_shape": list(test_audio_b1.shape),
            "onnx_output_shape": list(onnx_out.shape),
            "pytorch_output_shape": list(pt_np.shape),
            "shape_match": list(onnx_out.shape) == list(pt_np.shape),
            "mae": round(mae, 6),
            "max_abs_error": round(max_err, 6),
            "p50_error": round(p50, 6),
            "p95_error": round(p95, 6),
            "p99_error": round(p99, 6),
            "cosine_similarity": round(cos_sim, 6),
            "onnx_nan": onnx_nan,
            "pytorch_nan": pt_nan,
            "onnx_range": {"min": float(onnx_out.min()), "max": float(onnx_out.max()),
                           "mean": float(onnx_out.mean()), "std": float(onnx_out.std())},
            "pytorch_range": {"min": float(pt_np.min()), "max": float(pt_np.max()),
                              "mean": float(pt_np.mean()), "std": float(pt_np.std())},
        }

        if cos_sim > 0.999 and mae < 0.01:
            results["verdict"] = "PASS — HuBERT PyTorch is numerically compatible with ONNX"
        elif cos_sim > 0.99:
            results["verdict"] = "MARGINAL — review per-element differences"
        else:
            results["verdict"] = "FAIL — significant difference"

        print(f"[parity] MAE: {mae:.6f}, Max: {max_err:.6f}, Cosine: {cos_sim:.6f}")
        print(f"[parity] VERDICT: {results['verdict']}")
        return results

    @modal.method()
    def full_probe(self) -> dict:
        import torch

        print("[probe] === Stage 2B-2 HuBERT Full Probe ===")
        print(f"[probe] Health: {self._health_dict()}")
        _run_nvidia_smi("probe start")

        results = {
            "stage": "2B-2",
            "component": "HuBERT",
            "approach": "onnx2torch_fx_graphmodule",
            "health": self._health_dict(),
            "checks": {},
            "benchmarks": {},
        }

        N_WARMUP = 10
        N_MEASURED = 100

        torch.manual_seed(42)
        inp = torch.randn(2, 8400, device="cuda", dtype=torch.float32)

        # --- Correctness ---
        print("[probe] === Correctness checks ===")
        with torch.no_grad():
            out = self.model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]

        results["checks"]["b2_fp32"] = _check_nan(out, "hubert_b2_fp32")
        print(f"[probe] B=2: {results['checks']['b2_fp32']}")

        expected_frames = 8400 // 320
        print(f"[probe] Expected frames: {expected_frames} (8400 / 320 stride)")
        print(f"[probe] Actual output shape: {tuple(out.shape)}")

        # --- Benchmark: B=2 eager fp32 ---
        print("[probe] === Benchmark B=2 eager fp32 ===")
        with torch.no_grad():
            for _ in range(N_WARMUP):
                self.model(inp)
                if isinstance(out, (tuple, list)):
                    _ = out[0]
            torch.cuda.synchronize()

            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                out_bench = self.model(inp)
                if isinstance(out_bench, (tuple, list)):
                    out_bench = out_bench[0]
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))

        results["benchmarks"]["b2_eager_fp32"] = _stats_ms(lats)
        results["benchmarks"]["b2_eager_fp32"]["budget_pct"] = round(
            results["benchmarks"]["b2_eager_fp32"]["p50_ms"] / 200 * 100, 1
        )
        print(f"[probe] b2_eager_fp32: p50={results['benchmarks']['b2_eager_fp32']['p50_ms']:.3f}ms "
              f"budget={results['benchmarks']['b2_eager_fp32']['budget_pct']:.1f}%")

        # --- Benchmark: B=1 eager fp32 ---
        print("[probe] === Benchmark B=1 eager fp32 ===")
        inp_b1 = torch.randn(1, 8400, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                self.model(inp_b1)
            torch.cuda.synchronize()

            lats_b1 = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                out_b1 = self.model(inp_b1)
                if isinstance(out_b1, (tuple, list)):
                    out_b1 = out_b1[0]
                t1e.record()
                torch.cuda.synchronize()
                lats_b1.append(t0e.elapsed_time(t1e))

        results["benchmarks"]["b1_eager_fp32"] = _stats_ms(lats_b1)
        results["benchmarks"]["b1_eager_fp32"]["budget_pct"] = round(
            results["benchmarks"]["b1_eager_fp32"]["p50_ms"] / 200 * 100, 1
        )
        print(f"[probe] b1_eager_fp32: p50={results['benchmarks']['b1_eager_fp32']['p50_ms']:.3f}ms "
              f"budget={results['benchmarks']['b1_eager_fp32']['budget_pct']:.1f}%")

        # --- CUDA graph attempt ---
        print("[probe] === CUDA graph B=2 ===")
        try:
            static_in = inp.clone()
            torch.cuda.synchronize()
            with torch.no_grad():
                static_out = self.model(static_in)
            if isinstance(static_out, (tuple, list)):
                static_out = static_out[0]

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    graph_out = self.model(static_in)
                if isinstance(graph_out, (tuple, list)):
                    graph_out = graph_out[0]
                static_out.copy_(graph_out)

            for _ in range(N_WARMUP):
                g.replay()
            torch.cuda.synchronize()

            lats_cg = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                g.replay()
                t1e.record()
                torch.cuda.synchronize()
                lats_cg.append(t0e.elapsed_time(t1e))

            results["benchmarks"]["b2_cuda_graph_fp32"] = _stats_ms(lats_cg)
            results["benchmarks"]["b2_cuda_graph_fp32"]["budget_pct"] = round(
                results["benchmarks"]["b2_cuda_graph_fp32"]["p50_ms"] / 200 * 100, 1
            )
            print(f"[probe] b2_cuda_graph_fp32: p50={results['benchmarks']['b2_cuda_graph_fp32']['p50_ms']:.3f}ms "
                  f"budget={results['benchmarks']['b2_cuda_graph_fp32']['budget_pct']:.1f}%")
            del g
        except Exception as e:
            print(f"[probe] CUDA graph FAILED: {e}")
            results["benchmarks"]["b2_cuda_graph_fp32"] = {"error": str(e)}

        # --- Budget ---
        best_key = min(
            [k for k in ["b2_eager_fp32", "b2_cuda_graph_fp32"]
             if k in results["benchmarks"] and "error" not in results["benchmarks"].get(k, {})],
            key=lambda k: results["benchmarks"][k]["p50_ms"]
        )
        best_ms = results["benchmarks"][best_key]["p50_ms"]

        modnet_ms = 10.6
        results["budget"] = {
            "avtr_encode_decode_ms": 27,
            "renderer_phase2a_ms": "122-128",
            "modnet_cuda_graph_ms": modnet_ms,
            "hubert_best_ms": round(best_ms, 3),
            "hubert_best_path": best_key,
            "estimated_total_ms": round(27 + 125 + modnet_ms + best_ms, 1),
            "budget_200ms_pct": round((27 + 125 + modnet_ms + best_ms) / 200 * 100, 1),
            "headroom_ms": round(200 - 27 - 125 - modnet_ms - best_ms, 1),
        }

        total_est = 27 + 125 + modnet_ms + best_ms
        if total_est <= 180:
            results["budget"]["go_nogo"] = "GO"
        elif total_est <= 200:
            results["budget"]["go_nogo"] = "TIGHT"
        else:
            results["budget"]["go_nogo"] = "NO-GO"

        print(f"[probe] === Budget: HuBERT {best_key} = {best_ms:.3f}ms, "
              f"total ~ {results['budget']['estimated_total_ms']}ms, "
              f"verdict={results['budget']['go_nogo']} ===")

        _run_nvidia_smi("probe end")
        return results

    @modal.method()
    def health_check(self) -> dict:
        return self._health_dict()
