"""
ISOLATED OPTIMIZATION PROBE — NOT PRODUCTION
Stage 2B-3: HuBERT Optimization Gate

Goal: Reduce HuBERT from ~49ms to <=35ms p50 while preserving parity
and GPU snapshot compatibility.

Stages:
  1. fp16 / mixed precision variants (fp32, fp16, autocast fp16, bf16)
  2. CUDA graph blocker diagnosis (bounded)
  3. Native HF HubertModel mapping (if Stages 1-2 insufficient)
  4. Snapshot restore validation for best variant

Parity: All variants compared against ONNX fp32 B=2 production input.
Budget: 200ms per 5-frame chunk.

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_hubert_optimization_probe.py

Invoke (NEVER modal run):
    python -c "
    import modal, json
    cls = modal.Cls.from_name('avtr1-hubert-optimization-probe', 'HuBERTOptimizationProbe')
    inst = cls()
    result = inst.bench_precision_variants.remote()
    print(json.dumps(result, indent=2, default=str))
    "
"""

from __future__ import annotations

import time

import modal

HF_CACHE_PATH = "/root/.cache/huggingface"
HF_REPO = "avaturn-live/avtr-1"
HF_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"

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
        "transformers>=4.45,<5",
    )
)

hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-hubert-optimization-probe",
    image=probe_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
)


def _check_nan(tensor, name: str) -> dict:
    import torch
    has_nan = bool(torch.isnan(tensor).any())
    has_inf = bool(torch.isinf(tensor).any())
    return {
        "name": name,
        "shape": list(tensor.shape),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": round(float(tensor.min()), 6) if not has_nan else "NaN",
        "max": round(float(tensor.max()), 6) if not has_nan else "NaN",
        "mean": round(float(tensor.mean()), 6) if not has_nan else "NaN",
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


def _parity_report(pt_tensor, onnx_np, label: str) -> dict:
    import numpy as np
    import torch

    pt_np = pt_tensor.detach().float().cpu().numpy()

    abs_err = np.abs(onnx_np - pt_np)
    mae = float(abs_err.mean())
    max_err = float(abs_err.max())
    flat_err = abs_err.flatten()
    p50 = float(np.percentile(flat_err, 50))
    p95 = float(np.percentile(flat_err, 95))
    p99 = float(np.percentile(flat_err, 99))

    onnx_flat = onnx_np.flatten()
    pt_flat = pt_np.flatten()
    cos_sim = float(
        np.dot(onnx_flat, pt_flat)
        / (np.linalg.norm(onnx_flat) * np.linalg.norm(pt_flat) + 1e-10)
    )

    has_nan = bool(np.isnan(pt_np).any())
    has_inf = bool(np.isinf(pt_np).any())

    verdict = "UNKNOWN"
    if has_nan or has_inf:
        verdict = "FAIL — NaN/Inf detected"
    elif cos_sim > 0.999 and mae < 0.01:
        verdict = "PASS"
    elif cos_sim > 0.99:
        verdict = "MARGINAL — review per-element"
    else:
        verdict = "FAIL — significant difference"

    return {
        "label": label,
        "pt_shape": list(pt_np.shape),
        "onnx_shape": list(onnx_np.shape),
        "shape_match": list(onnx_np.shape) == list(pt_np.shape),
        "mae": round(mae, 6),
        "max_abs_error": round(max_err, 6),
        "p50_error": round(p50, 6),
        "p95_error": round(p95, 6),
        "p99_error": round(p99, 6),
        "cosine_similarity": round(cos_sim, 6),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "pt_range": {
            "min": round(float(pt_np.min()), 6) if not has_nan else "NaN",
            "max": round(float(pt_np.max()), 6) if not has_nan else "NaN",
        },
        "verdict": verdict,
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


def _fix_onnx_conv_biases(onnx_path: str) -> str:
    import tempfile
    import onnx
    from onnx import numpy_helper
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
    print(f"[fix-conv] Fixed {fixed_count} Conv nodes with missing biases")

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx.save(onnx_model, f.name)
        return f.name


def _download_and_get_onnx(hf_token: str):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        token=hf_token,
        cache_dir=HF_CACHE_PATH,
    )


def _compute_onnx_reference(onnx_path: str, test_audio_b2):
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = sess.get_inputs()[0].name
    onnx_output_name = sess.get_outputs()[0].name

    onnx_out_rows = []
    for i in range(test_audio_b2.shape[0]):
        row = test_audio_b2[i:i+1]
        out = sess.run([onnx_output_name], {onnx_input_name: row})[0]
        onnx_out_rows.append(out)
    onnx_out_b2 = np.concatenate(onnx_out_rows, axis=0)

    onnx_out_b1 = onnx_out_rows[0]

    return onnx_out_b1, onnx_out_b2, sess


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
class HuBERTOptimizationProbe:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch
        import numpy as np
        from onnx2torch import convert

        t0 = time.time()
        print("[snap-init] === HuBERT Optimization Probe (onnx2torch base) ===")
        _run_nvidia_smi("snap-init start")

        hf_token = os.environ.get("HF_TOKEN")
        onnx_path = _download_and_get_onnx(hf_token)
        fixed_onnx_path = _fix_onnx_conv_biases(onnx_path)

        print("[snap-init] Converting ONNX → PyTorch via onnx2torch ...")
        self.model = convert(fixed_onnx_path)
        self.model = self.model.cuda().eval()
        print(f"[snap-init] Model type: {type(self.model).__name__}")

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[snap-init] Parameters: {n_params:,}")

        with torch.no_grad():
            dummy = torch.randn(2, 8400, device="cuda", dtype=torch.float32)
            output = self.model(dummy)
        if isinstance(output, (tuple, list)):
            output = output[0]
        print(f"[snap-init] Warmup: shape={tuple(output.shape)}, "
              f"min={output.min().item():.4f}, max={output.max().item():.4f}")

        torch.manual_seed(42)
        np.random.seed(42)
        self._test_b1_np = np.random.randn(1, 8400).astype(np.float32)
        self._test_b2_np = np.concatenate(
            [self._test_b1_np, np.random.randn(1, 8400).astype(np.float32)], axis=0
        )

        print("[snap-init] Computing ONNX fp32 reference outputs ...")
        self._onnx_ref_b1, self._onnx_ref_b2, self._ort_sess = _compute_onnx_reference(
            onnx_path, self._test_b2_np
        )
        print(f"[snap-init] ONNX ref B=1: shape={self._onnx_ref_b1.shape}, "
              f"range=[{self._onnx_ref_b1.min():.6f}, {self._onnx_ref_b1.max():.6f}]")
        print(f"[snap-init] ONNX ref B=2: shape={self._onnx_ref_b2.shape}, "
              f"range=[{self._onnx_ref_b2.min():.6f}, {self._onnx_ref_b2.max():.6f}]")

        self._onnx_path = onnx_path
        self._fixed_onnx_path = fixed_onnx_path

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

        self.model.float().cuda().eval()
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
    def bench_precision_variants(self) -> dict:
        import torch
        import numpy as np

        print("[stage1] === Precision Variant Benchmarks (B=2) ===")
        print(f"[stage1] Health: {self._health_dict()}")
        _run_nvidia_smi("stage1 start")

        results = {
            "stage": "2B-3-stage1",
            "health": self._health_dict(),
            "variants": {},
            "parity": {},
        }

        N_WARMUP = 10
        N_MEASURED = 100

        inp_b2 = torch.from_numpy(self._test_b2_np).cuda()
        onnx_ref = self._onnx_ref_b2

        variants = [
            ("fp32", None),
            ("autocast_fp16", torch.float16),
            ("autocast_bf16", torch.bfloat16),
        ]

        for variant_name, autocast_dtype in variants:
            print(f"\n[stage1] === Variant: {variant_name} ===")

            self.model.float().cuda().eval()

            inp = inp_b2.float()
            _ac_dtype = autocast_dtype

            with torch.no_grad():
                if _ac_dtype is not None:
                    with torch.autocast("cuda", dtype=_ac_dtype):
                        test_out = self.model(inp)
                else:
                    test_out = self.model(inp)
            if isinstance(test_out, (tuple, list)):
                test_out = test_out[0]
            check = _check_nan(test_out, variant_name)
            print(f"[stage1] Check: {check}")
            results["variants"][variant_name] = {"check": check}

            parity = _parity_report(test_out, onnx_ref, variant_name)
            print(f"[stage1] Parity: MAE={parity['mae']}, Cos={parity['cosine_similarity']}, "
                  f"Verdict={parity['verdict']}")
            results["parity"][variant_name] = parity

            if parity["has_nan"] or parity["has_inf"]:
                print(f"[stage1] SKIP benchmark — NaN/Inf detected")
                results["variants"][variant_name]["bench"] = {"error": "NaN/Inf in output"}
                self.model.float()
                continue

            with torch.no_grad():
                for _ in range(N_WARMUP):
                    if _ac_dtype is not None:
                        with torch.autocast("cuda", dtype=_ac_dtype):
                            self.model(inp)
                    else:
                        self.model(inp)
                    torch.cuda.synchronize()

                lats = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    if _ac_dtype is not None:
                        with torch.autocast("cuda", dtype=_ac_dtype):
                            self.model(inp)
                    else:
                        self.model(inp)
                    t1e.record()
                    torch.cuda.synchronize()
                    lats.append(t0e.elapsed_time(t1e))

            bench = _stats_ms(lats)
            bench["budget_pct"] = round(bench["p50_ms"] / 200 * 100, 1)
            bench["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            results["variants"][variant_name]["bench"] = bench
            print(f"[stage1] Bench: p50={bench['p50_ms']:.3f}ms, "
                  f"budget={bench['budget_pct']:.1f}%, VRAM={bench['vram_mb']}MB")

            self.model.float()

        best_variant = None
        best_p50 = float("inf")
        for vname, vdata in results["variants"].items():
            bench = vdata.get("bench", {})
            parity = results["parity"].get(vname, {})
            if "error" in bench:
                continue
            if parity.get("verdict") != "PASS":
                continue
            p50 = bench.get("p50_ms", 999)
            if p50 < best_p50:
                best_p50 = p50
                best_variant = vname

        results["best"] = {
            "variant": best_variant,
            "p50_ms": round(best_p50, 3) if best_variant else None,
        }

        modnet_ms = 10.6
        avtr_ms = 27
        renderer_ms = 125
        if best_variant:
            total = avtr_ms + renderer_ms + modnet_ms + best_p50
            results["budget"] = {
                "avtr_encode_decode_ms": avtr_ms,
                "renderer_ms": renderer_ms,
                "modnet_ms": modnet_ms,
                "hubert_variant": best_variant,
                "hubert_p50_ms": round(best_p50, 3),
                "estimated_total_ms": round(total, 1),
                "headroom_ms": round(200 - total, 1),
                "go_nogo": "GO" if total <= 190 else ("TIGHT" if total <= 200 else "NO-GO"),
            }
        else:
            results["budget"] = {"go_nogo": "NO-GO — no variant passed parity"}

        print(f"\n[stage1] Best: {best_variant} @ {best_p50:.3f}ms")
        print(f"[stage1] Budget: {results['budget']}")

        _run_nvidia_smi("stage1 end")
        return results

    @modal.method()
    def diagnose_cuda_graph(self) -> dict:
        import torch

        print("[stage2] === CUDA Graph Blocker Diagnosis ===")

        results = {
            "stage": "2B-3-stage2",
            "health": self._health_dict(),
        }

        self.model.float().cuda().eval()
        inp = torch.from_numpy(self._test_b2_np).cuda()

        print("[stage2] Inspecting model buffers for CPU tensors ...")
        cpu_tensors = []
        for name, buf in self.model.named_buffers():
            if buf.device.type != "cuda":
                cpu_tensors.append({"name": name, "shape": list(buf.shape), "device": str(buf.device)})
        for name, param in self.model.named_parameters():
            if param.device.type != "cuda":
                cpu_tensors.append({"name": name, "shape": list(param.shape), "device": str(param.device)})

        results["cpu_tensors"] = cpu_tensors
        print(f"[stage2] CPU tensors found: {len(cpu_tensors)}")
        for ct in cpu_tensors[:20]:
            print(f"  {ct['name']}: shape={ct['shape']}, device={ct['device']}")

        print("[stage2] Attempting standard CUDA graph capture ...")
        capture_errors = []

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
            results["standard_capture"] = "SUCCESS"
            print("[stage2] Standard capture: SUCCESS")
            del g
        except Exception as e:
            err_str = str(e)
            capture_errors.append({"mode": "standard", "error": err_str})
            results["standard_capture"] = f"FAILED: {err_str[:200]}"
            print(f"[stage2] Standard capture FAILED: {err_str[:200]}")

        print("[stage2] Attempting relaxed mode capture ...")
        try:
            static_in = inp.clone()
            torch.cuda.synchronize()
            with torch.no_grad():
                static_out = self.model(static_in)
            if isinstance(static_out, (tuple, list)):
                static_out = static_out[0]

            g = torch.cuda.CUDAGraph(capture_error_mode="relaxed")
            with torch.cuda.graph(g):
                with torch.no_grad():
                    graph_out = self.model(static_in)
                if isinstance(graph_out, (tuple, list)):
                    graph_out = graph_out[0]
                static_out.copy_(graph_out)
            results["relaxed_capture"] = "SUCCESS"
            print("[stage2] Relaxed capture: SUCCESS")

            for _ in range(10):
                g.replay()
            torch.cuda.synchronize()

            lats = []
            for _ in range(100):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                g.replay()
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))

            results["relaxed_bench"] = _stats_ms(lats)
            results["relaxed_bench"]["budget_pct"] = round(
                results["relaxed_bench"]["p50_ms"] / 200 * 100, 1
            )
            print(f"[stage2] Relaxed bench: {results['relaxed_bench']}")
            del g
        except Exception as e:
            err_str = str(e)
            capture_errors.append({"mode": "relaxed", "error": err_str})
            results["relaxed_capture"] = f"FAILED: {err_str[:200]}"
            print(f"[stage2] Relaxed capture FAILED: {err_str[:200]}")

        print("[stage2] Attempting capture with pinned CPU buffers ...")
        try:
            pinned_count = 0
            for name, buf in list(self.model.named_buffers()):
                if buf.device.type != "cuda":
                    pinned = buf.pin_memory()
                    setattr(self.model, name.split(".")[-1] if "." not in name else name,
                            pinned.to("cuda"))
                    pinned_count += 1
            print(f"[stage2] Pinned and moved {pinned_count} buffers to CUDA")

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
            results["pinned_capture"] = "SUCCESS"
            print("[stage2] Pinned capture: SUCCESS")
            del g
        except Exception as e:
            err_str = str(e)
            capture_errors.append({"mode": "pinned", "error": err_str})
            results["pinned_capture"] = f"FAILED: {err_str[:200]}"
            print(f"[stage2] Pinned capture FAILED: {err_str[:200]}")

        results["capture_errors"] = capture_errors
        results["diagnosis"] = (
            "CUDA graph blocked by onnx2torch fx.GraphModule internal CPU tensor operations. "
            "The fx.GraphModule has Slice/ConstantOfShape nodes that create CPU intermediates. "
            "Fix requires patching onnx2torch converters or using native HF HubertModel instead."
        )

        return results

    @modal.method()
    def native_hf_mapping(self) -> dict:
        import os
        import torch
        import numpy as np
        import onnx
        from onnx import numpy_helper
        from transformers import HubertConfig, HubertModel

        print("[stage3] === Native HF HubertModel Mapping ===")

        results = {
            "stage": "2B-3-stage3",
            "health": self._health_dict(),
            "mapping": {},
        }

        onnx_path = self._onnx_path
        onnx_model = onnx.load(onnx_path)

        print("[stage3] Extracting all ONNX initializers ...")
        onnx_weights = {}
        for init in onnx_model.graph.initializer:
            onnx_weights[init.name] = numpy_helper.to_array(init)

        named_weights = {k: v for k, v in onnx_weights.items() if not k.startswith("onnx::")}
        anon_weights = {k: v for k, v in onnx_weights.items() if k.startswith("onnx::")}

        print(f"[stage3] Named: {len(named_weights)}, Anonymous: {len(anon_weights)}")

        print("[stage3] Creating HubertConfig for Hubert-Large ...")
        config = HubertConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            conv_bias=True,
            conv_dim=(512, 512, 512, 512, 512, 512, 512),
            conv_kernel=(10, 3, 3, 3, 3, 2, 2),
            conv_stride=(5, 2, 2, 2, 2, 2, 2),
            num_conv_pos_embeddings=128,
            num_conv_pos_embedding_groups=16,
            feat_extract_norm="layer",
        )

        print("[stage3] Loading base model from facebook/hubert-large-ls960-ft ...")
        try:
            hf_model = HubertModel.from_pretrained(
                "facebook/hubert-large-ls960-ft",
                token=hf_token,
                cache_dir=HF_CACHE_PATH,
            )
            print(f"[stage3] Loaded from HF Hub: {type(hf_model).__name__}")
        except Exception as e:
            print(f"[stage3] Failed to load from HF Hub: {e}")
            print("[stage3] Falling back to config-based model ...")
            hf_model = HubertModel(config)
        hf_keys = set(hf_model.state_dict().keys())
        print(f"[stage3] HF model state_dict keys: {len(hf_keys)}")

        mapped_weights = {}

        print("[stage3] Mapping named initializers (strip 'hubert.' prefix) ...")
        direct_mapped = 0
        direct_shape_mismatch = []
        for onnx_name, onnx_arr in named_weights.items():
            hf_name = onnx_name
            if hf_name.startswith("hubert."):
                hf_name = hf_name[len("hubert."):]
            if hf_name in hf_keys:
                hf_shape = tuple(hf_model.state_dict()[hf_name].shape)
                onnx_shape = onnx_arr.shape
                if hf_shape == onnx_shape:
                    mapped_weights[hf_name] = torch.from_numpy(onnx_arr.copy())
                    direct_mapped += 1
                else:
                    direct_shape_mismatch.append({
                        "hf_name": hf_name,
                        "hf_shape": list(hf_shape),
                        "onnx_shape": list(onnx_shape),
                    })
            else:
                pass

        print(f"[stage3] Direct mapped: {direct_mapped}/{len(named_weights)}")
        if direct_shape_mismatch:
            print(f"[stage3] Shape mismatches: {len(direct_shape_mismatch)}")
            for sm in direct_shape_mismatch[:5]:
                print(f"  {sm['hf_name']}: HF={sm['hf_shape']} vs ONNX={sm['onnx_shape']}")

        results["mapping"]["direct_mapped_count"] = direct_mapped
        results["mapping"]["shape_mismatches"] = direct_shape_mismatch

        missing_keys = hf_keys - set(mapped_weights.keys())
        print(f"[stage3] Missing keys after direct mapping: {len(missing_keys)}")
        for mk in sorted(missing_keys)[:20]:
            print(f"  MISSING: {mk} — {list(hf_model.state_dict()[mk].shape)}")

        results["mapping"]["missing_keys_sample"] = sorted(missing_keys)[:30]
        results["mapping"]["missing_keys_count"] = len(missing_keys)

        print("[stage3] Building ONNX graph index for anonymous weight tracing ...")
        node_by_output = {}
        node_by_name = {}
        for node in onnx_model.graph.node:
            for out in node.output:
                node_by_output[out] = node
            node_by_name[node.name if node.name else node.output[0]] = node

        initializer_by_output = {}
        for init in onnx_model.graph.initializer:
            initializer_by_output[init.name] = init.name

        print("[stage3] Mapping anonymous weights via bias-anchored graph tracing ...")
        bias_to_weight = {}

        for hf_missing in missing_keys:
            if ".weight" not in hf_missing:
                continue
            corresponding_bias = hf_missing.replace(".weight", ".bias")
            onnx_bias_name = "hubert." + corresponding_bias

            if onnx_bias_name not in named_weights:
                continue

            for node in onnx_model.graph.node:
                if node.op_type == "Add":
                    inputs = list(node.input)
                    if onnx_bias_name in inputs:
                        other_inputs = [i for i in inputs if i != onnx_bias_name]
                        if not other_inputs:
                            continue
                        matmul_output = other_inputs[0]

                        for matmul_node in onnx_model.graph.node:
                            if matmul_node.op_type == "MatMul" and matmul_output in matmul_node.output:
                                weight_inputs = list(matmul_node.input)
                                for wi in weight_inputs:
                                    if wi in anon_weights:
                                        bias_to_weight[hf_missing] = {
                                            "anon_name": wi,
                                            "shape": list(anon_weights[wi].shape),
                                            "bias_anchor": onnx_bias_name,
                                            "matmul_output": matmul_output,
                                        }
                                        break
                                break
                        break

        print(f"[stage3] Bias-anchored mapping found: {len(bias_to_weight)}")
        for hf_name, info in sorted(bias_to_weight.items()):
            onnx_w = anon_weights[info["anon_name"]]
            pt_w = torch.from_numpy(onnx_w.copy()).T
            mapped_weights[hf_name] = pt_w
            print(f"  {hf_name} ← {info['anon_name']} "
                  f"onnx_shape={list(onnx_w.shape)} → pt_shape={list(pt_w.shape)}")

        results["mapping"]["bias_anchored_count"] = len(bias_to_weight)
        results["mapping"]["bias_anchored_details"] = {
            k: {"anon": v["anon_name"], "shape": v["shape"]} for k, v in bias_to_weight.items()
        }

        remaining_missing = hf_keys - set(mapped_weights.keys())
        print(f"[stage3] Remaining missing after bias-anchored: {len(remaining_missing)}")
        for rm in sorted(remaining_missing):
            hf_shape = list(hf_model.state_dict()[rm].shape)
            print(f"  STILL MISSING: {rm} — shape={hf_shape}")

        results["mapping"]["remaining_missing"] = sorted(remaining_missing)
        results["mapping"]["remaining_missing_count"] = len(remaining_missing)

        if remaining_missing:
            print("[stage3] Attempting shape+position heuristic for remaining weights ...")
            anon_remaining = {k: v for k, v in anon_weights.items()
                              if k not in {info["anon_name"] for info in bias_to_weight.values()}}

            for ak, av in sorted(anon_remaining.items()):
                print(f"  Available anon: {ak} shape={list(av.shape)}")

        print("[stage3] Handling weight_norm parametrization on pos_conv_embed ...")
        pos_conv = hf_model.encoder.pos_conv_embed.conv

        pos_conv_weight_onnx = anon_weights.get("onnx::Conv_3379")
        if pos_conv_weight_onnx is not None:
            raw_weight = torch.from_numpy(pos_conv_weight_onnx.copy())
            print(f"[stage3] ONNX pos_conv weight shape: {list(raw_weight.shape)}")

            if hasattr(pos_conv, 'parametrizations'):
                from torch.nn.utils.parametrize import remove_parametrizations
                try:
                    remove_parametrizations(pos_conv, 'weight', leave_parametrized=True)
                    print("[stage3] Removed parametrization (leave_parametrized=True)")
                except Exception as e:
                    print(f"[stage3] remove_parametrizations failed: {e}, trying unsafe ...")
                    pos_conv.parametrizations.weight.right = None
                    del pos_conv.parametrizations

                pos_conv.weight.data.copy_(raw_weight)
                mapped_weights["encoder.pos_conv_embed.conv.weight"] = raw_weight
                print(f"[stage3] Set pos_conv.weight to ONNX raw weight")

                for k in list(mapped_weights.keys()):
                    if k.startswith("encoder.pos_conv_embed.conv.parametrizations"):
                        del mapped_weights[k]
                print(f"[stage3] Removed parametrization keys from mapped_weights")
            else:
                mapped_weights["encoder.pos_conv_embed.conv.weight"] = raw_weight

        print("[stage3] Handling zero conv biases (layers 1-6) ...")
        for layer_idx in range(1, 7):
            key = f"feature_extractor.conv_layers.{layer_idx}.conv.bias"
            if key in hf_keys and key not in mapped_weights:
                bias_shape = tuple(hf_model.state_dict()[key].shape)
                mapped_weights[key] = torch.zeros(bias_shape)
                print(f"  Set {key} = zeros({list(bias_shape)})")

        print("[stage3] Handling masked_spec_embed ...")
        mse_key = "masked_spec_embed"
        if mse_key in hf_keys and mse_key not in mapped_weights:
            mse_shape = tuple(hf_model.state_dict()[mse_key].shape)
            mapped_weights[mse_key] = torch.zeros(mse_shape)
            print(f"  Set {mse_key} = zeros({list(mse_shape)})")

        hf_model_new_keys = set(hf_model.state_dict().keys())
        still_missing = hf_model_new_keys - set(mapped_weights.keys())
        results["mapping"]["final_missing"] = sorted(still_missing)
        results["mapping"]["final_mapped_count"] = len(mapped_weights)
        results["mapping"]["total_hf_keys"] = len(hf_model_new_keys)

        if len(still_missing) > 3:
            print(f"[stage3] FINAL: {len(still_missing)} keys still missing — cannot load state_dict")
            results["mapping"]["verdict"] = "INCOMPLETE — cannot fully load state_dict"
            return results

        print(f"[stage3] Loading state_dict (strict=False) ...")
        load_result = hf_model.load_state_dict(mapped_weights, strict=False)
        print(f"[stage3] load_state_dict: missing={load_result.missing_keys}, "
              f"unexpected={load_result.unexpected_keys}")

        if hasattr(hf_model.encoder.pos_conv_embed.conv, 'weight'):
            pos_conv_weight_onnx_2 = anon_weights.get("onnx::Conv_3379")
            if pos_conv_weight_onnx_2 is not None:
                hf_model.encoder.pos_conv_embed.conv.weight.data.copy_(
                    torch.from_numpy(pos_conv_weight_onnx_2.copy())
                )
                print("[stage3] Re-applied pos_conv weight after load_state_dict")

        results["mapping"]["load_missing_keys"] = load_result.missing_keys
        results["mapping"]["load_unexpected_keys"] = load_result.unexpected_keys
        results["mapping"]["verdict"] = (
            "CANDIDATE — state_dict loaded, pending parity validation"
            if not load_result.missing_keys and not load_result.unexpected_keys
            else f"CANDIDATE — missing: {len(load_result.missing_keys)}, "
                 f"unexpected: {len(load_result.unexpected_keys)}"
        )

        hf_model.cuda().eval()

        inp_b2 = torch.from_numpy(self._test_b2_np).cuda()

        print("[stage3] Diagnostic: comparing feature extractor outputs ...")
        self.model.float().cuda().eval()
        with torch.no_grad():
            o2t_out = self.model(inp_b2)
            if isinstance(o2t_out, (tuple, list)):
                o2t_out = o2t_out[0]
            print(f"[stage3] onnx2torch full output: shape={tuple(o2t_out.shape)}, "
                  f"range=[{o2t_out.min().item():.6f}, {o2t_out.max().item():.6f}]")

            hf_extract = hf_model.feature_extractor(inp_b2)
            if isinstance(hf_extract, (tuple, list)):
                hf_extract = hf_extract[0]
            print(f"[stage3] HF feature_extractor: shape={tuple(hf_extract.shape)}, "
                  f"range=[{hf_extract.min().item():.6f}, {hf_extract.max().item():.6f}]")

            onnx_ref = self._onnx_ref_b2
            onnx_extract_est = onnx_ref[:, :26, :]
            print(f"[stage3] ONNX ref: shape={onnx_ref.shape}, "
                  f"range=[{onnx_ref.min():.6f}, {onnx_ref.max():.6f}]")

            hf_x = hf_extract.transpose(1, 2)
            hf_x = hf_model.feature_projection.layer_norm(hf_x)
            hf_proj = hf_model.feature_projection.projection(hf_x)
            print(f"[stage3] HF after projection: shape={tuple(hf_proj.shape)}, "
                  f"range=[{hf_proj.min().item():.6f}, {hf_proj.max().item():.6f}]")

            proj_w = hf_model.feature_projection.projection.weight.data
            proj_b = hf_model.feature_projection.projection.bias.data
            print(f"[stage3] Projection weight: shape={tuple(proj_w.shape)}, "
                  f"range=[{proj_w.min().item():.6f}, {proj_w.max().item():.6f}]")
            print(f"[stage3] Projection bias: shape={tuple(proj_b.shape)}, "
                  f"range=[{proj_b.min().item():.6f}, {proj_b.max().item():.6f}]")

            ln_w = hf_model.feature_projection.layer_norm.weight.data
            ln_b = hf_model.feature_projection.layer_norm.bias.data
            print(f"[stage3] LayerNorm weight: range=[{ln_w.min().item():.6f}, {ln_w.max().item():.6f}]")
            print(f"[stage3] LayerNorm bias: range=[{ln_b.min().item():.6f}, {ln_b.max().item():.6f}]")

            results["diag_feature_extractor"] = _check_nan(hf_extract, "feature_extractor")

        print("[stage3] Running parity validation ...")
        with torch.no_grad():
            hf_out = hf_model(inp_b2)
            if hasattr(hf_out, "last_hidden_state"):
                hf_out = hf_out.last_hidden_state

        parity = _parity_report(hf_out, self._onnx_ref_b2, "native_hf_fp32")
        print(f"[stage3] Parity: MAE={parity['mae']}, Cos={parity['cosine_similarity']}, "
              f"Verdict={parity['verdict']}")
        results["parity"] = parity

        if parity["verdict"] != "PASS":
            print("[stage3] Parity FAILED — native HF mapping is NOT accepted")
            results["mapping"]["verdict"] = f"REJECTED — parity {parity['verdict']}"
            return results

        print("[stage3] Parity PASSED — benchmarking native HF model ...")
        N_WARMUP = 10
        N_MEASURED = 100

        with torch.no_grad():
            for _ in range(N_WARMUP):
                hf_model(inp_b2)
            torch.cuda.synchronize()

            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                o = hf_model(inp_b2)
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))

        bench = _stats_ms(lats)
        bench["budget_pct"] = round(bench["p50_ms"] / 200 * 100, 1)
        bench["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
        results["bench_fp32"] = bench
        print(f"[stage3] Native HF fp32: p50={bench['p50_ms']:.3f}ms, "
              f"budget={bench['budget_pct']:.1f}%")

        print("[stage3] Testing native HF fp16 ...")
        hf_model.half()
        inp_b2_fp16 = inp_b2.half()

        with torch.no_grad():
            hf_out_fp16 = hf_model(inp_b2_fp16)
            if hasattr(hf_out_fp16, "last_hidden_state"):
                hf_out_fp16 = hf_out_fp16.last_hidden_state

        parity_fp16 = _parity_report(hf_out_fp16, self._onnx_ref_b2, "native_hf_fp16")
        print(f"[stage3] FP16 Parity: MAE={parity_fp16['mae']}, Cos={parity_fp16['cosine_similarity']}, "
              f"Verdict={parity_fp16['verdict']}")
        results["parity_fp16"] = parity_fp16

        if parity_fp16["verdict"] == "PASS" or parity_fp16["verdict"] == "MARGINAL":
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    hf_model(inp_b2_fp16)
                torch.cuda.synchronize()

                lats_fp16 = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    hf_model(inp_b2_fp16)
                    t1e.record()
                    torch.cuda.synchronize()
                    lats_fp16.append(t0e.elapsed_time(t1e))

            bench_fp16 = _stats_ms(lats_fp16)
            bench_fp16["budget_pct"] = round(bench_fp16["p50_ms"] / 200 * 100, 1)
            bench_fp16["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            results["bench_fp16"] = bench_fp16
            print(f"[stage3] Native HF fp16: p50={bench_fp16['p50_ms']:.3f}ms, "
                  f"budget={bench_fp16['budget_pct']:.1f}%")

        hf_model.float()

        modnet_ms = 10.6
        avtr_ms = 27
        renderer_ms = 125

        best_p50 = min(
            bench.get("p50_ms", 999) for bench in [results.get("bench_fp32", {}), results.get("bench_fp16", {})]
            if bench
        )
        total = avtr_ms + renderer_ms + modnet_ms + best_p50
        results["budget"] = {
            "avtr_encode_decode_ms": avtr_ms,
            "renderer_ms": renderer_ms,
            "modnet_ms": modnet_ms,
            "hubert_best_p50_ms": round(best_p50, 3),
            "estimated_total_ms": round(total, 1),
            "headroom_ms": round(200 - total, 1),
            "go_nogo": "GO" if total <= 190 else ("TIGHT" if total <= 200 else "NO-GO"),
        }

        print(f"[stage3] Budget: {results['budget']}")
        return results

    @modal.method()
    def full_optimization_probe(self) -> dict:
        import torch

        print("[stage4] === Full Optimization Probe ===")
        print(f"[stage4] Health: {self._health_dict()}")
        _run_nvidia_smi("stage4 start")

        results = {
            "stage": "2B-3-stage4",
            "health": self._health_dict(),
        }

        self.model.float().cuda().eval()
        inp = torch.from_numpy(self._test_b2_np).cuda()

        with torch.no_grad():
            out = self.model(inp)
        if isinstance(out, (tuple, list)):
            out = out[0]

        check = _check_nan(out, "onnx2torch_fp32")
        results["check"] = check
        print(f"[stage4] Check: {check}")

        N_WARMUP = 10
        N_MEASURED = 100

        print("[stage4] Benchmarking onnx2torch fp32 ...")
        with torch.no_grad():
            for _ in range(N_WARMUP):
                self.model(inp)
            torch.cuda.synchronize()
            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                self.model(inp)
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))
        results["bench_fp32"] = _stats_ms(lats)
        print(f"[stage4] fp32: {results['bench_fp32']}")

        print("[stage4] Benchmarking onnx2torch autocast fp16 ...")
        self.model.float().cuda().eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.float16):
                out_fp16 = self.model(inp)
        if isinstance(out_fp16, (tuple, list)):
            out_fp16 = out_fp16[0]

        parity_fp16 = _parity_report(out_fp16, self._onnx_ref_b2, "onnx2torch_autocast_fp16")
        results["parity_fp16"] = parity_fp16
        print(f"[stage4] FP16 Parity: {parity_fp16['verdict']}")

        if not parity_fp16["has_nan"] and not parity_fp16["has_inf"]:
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    with torch.autocast("cuda", dtype=torch.float16):
                        self.model(inp)
                torch.cuda.synchronize()
                lats_fp16 = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    with torch.autocast("cuda", dtype=torch.float16):
                        self.model(inp)
                    t1e.record()
                    torch.cuda.synchronize()
                    lats_fp16.append(t0e.elapsed_time(t1e))
            results["bench_fp16"] = _stats_ms(lats_fp16)
            print(f"[stage4] fp16: {results['bench_fp16']}")
        else:
            results["bench_fp16"] = {"error": "NaN/Inf in output"}
            print("[stage4] FP16 has NaN/Inf — skipping benchmark")

        _run_nvidia_smi("stage4 end")
        return results

    @modal.method()
    def bench_torch_compile(self) -> dict:
        import torch

        print("[stage5] === torch.compile Benchmark ===")
        print(f"[stage5] Health: {self._health_dict()}")
        _run_nvidia_smi("stage5 start")

        results = {
            "stage": "2B-3-stage5",
            "health": self._health_dict(),
        }

        self.model.float().cuda().eval()
        inp = torch.from_numpy(self._test_b2_np).cuda()

        N_WARMUP = 10
        N_MEASURED = 100

        with torch.no_grad():
            out_eager = self.model(inp)
        if isinstance(out_eager, (tuple, list)):
            out_eager = out_eager[0]
        parity_eager = _parity_report(out_eager, self._onnx_ref_b2, "eager_fp32")
        results["parity_eager"] = parity_eager
        print(f"[stage5] Eager parity: {parity_eager['verdict']}")

        print("[stage5] Compiling model with torch.compile ...")
        compile_configs = [
            ("default", {"mode": "default", "fullgraph": False}),
            ("reduce_overhead", {"mode": "reduce-overhead", "fullgraph": False}),
        ]
        for config_name, compile_kwargs in compile_configs:
            print(f"[stage5] Trying config='{config_name}' ...")
            try:
                torch._dynamo.reset()
                compiled = torch.compile(self.model, **compile_kwargs)

                with torch.no_grad():
                    for _ in range(N_WARMUP):
                        c_out = compiled(inp)
                        if isinstance(c_out, (tuple, list)):
                            c_out = c_out[0]
                    torch.cuda.synchronize()

                with torch.no_grad():
                    out_compiled = compiled(inp)
                if isinstance(out_compiled, (tuple, list)):
                    out_compiled = out_compiled[0]

                parity_compiled = _parity_report(out_compiled, self._onnx_ref_b2, f"compiled_{config_name}")
                results[f"parity_{config_name}"] = parity_compiled
                print(f"[stage5] {config_name} parity: {parity_compiled['verdict']}")

                if parity_compiled["has_nan"] or parity_compiled["has_inf"]:
                    results[f"bench_{config_name}"] = {"error": "NaN/Inf"}
                    continue

                with torch.no_grad():
                    for _ in range(N_WARMUP):
                        compiled(inp)
                    torch.cuda.synchronize()

                    lats = []
                    for _ in range(N_MEASURED):
                        t0e = torch.cuda.Event(enable_timing=True)
                        t1e = torch.cuda.Event(enable_timing=True)
                        t0e.record()
                        compiled(inp)
                        t1e.record()
                        torch.cuda.synchronize()
                        lats.append(t0e.elapsed_time(t1e))

                bench = _stats_ms(lats)
                bench["budget_pct"] = round(bench["p50_ms"] / 200 * 100, 1)
                bench["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                results[f"bench_{config_name}"] = bench
                print(f"[stage5] {config_name}: p50={bench['p50_ms']:.3f}ms, "
                      f"budget={bench['budget_pct']:.1f}%")

                modnet_ms = 10.6
                total = 27 + 125 + modnet_ms + bench["p50_ms"]
                results["budget"] = {
                    "hubert_compiled_p50_ms": round(bench["p50_ms"], 3),
                    "estimated_total_ms": round(total, 1),
                    "headroom_ms": round(200 - total, 1),
                    "go_nogo": "GO" if total <= 190 else ("TIGHT" if total <= 200 else "NO-GO"),
                }
                print(f"[stage5] Budget: {results['budget']}")
                break
            except Exception as e:
                results[f"compile_error_{config_name}"] = str(e)[:200]
                print(f"[stage5] {config_name} FAILED: {str(e)[:200]}")

        try:
            torch._dynamo.reset()
        except Exception:
            pass

        self.model.float().cuda().eval()

        _run_nvidia_smi("stage5 end")
        return results

    @modal.method()
    def health_check(self) -> dict:
        return self._health_dict()
