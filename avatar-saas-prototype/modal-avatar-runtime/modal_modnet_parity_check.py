"""
ISOLATED PARITY VALIDATION — NOT PRODUCTION, NOT SNAPSHOT
Part A: MODNet Parity Check

Compares the Stage 2B-1 PyTorch MODNet (boopathiraj/MODNet via transformers.AutoModel)
against the production ONNX modnet.onnx from avaturn-live/avtr-1 HF repo.

Stage 2B-1 probe used:
  model = AutoModel.from_pretrained("boopathiraj/MODNet", trust_remote_code=True)
  forward: model(x, True) -> (semantic, detail, matte), take matte = result[2]

ONNX (modnet.onnx):
  Exported by PyTorch 2.5.1, opset ~17
  Input:  (1, 3, 288, 512) fp32 — fixed B=1
  Output: (1, 1, 288, 512) fp32 alpha matte [0, 1]

This script:
  - Downloads modnet.onnx from avaturn-live/avtr-1 HF repo
  - Loads ONNX via onnxruntime CPU
  - Loads boopathiraj/MODNet via transformers (CPU)
  - Runs identical seeded inputs through both
  - Reports MAE, max error, percentile errors, output ranges

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_modnet_parity_check.py

Invoke (NEVER modal run):
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-modnet-parity-check', 'MODNetParity')
inst = cls()
result = inst.validate.remote()
print(json.dumps(result, indent=2, default=str))
    "
"""

from __future__ import annotations

import modal

HF_CACHE_PATH = "/root/.cache/huggingface"

parity_image = (
    modal.Image.from_registry(
        "python:3.12-slim",
    )
    .apt_install("git", "gcc", "g++")
    .pip_install(
        "numpy>=2.0,<3",
        "onnxruntime>=1.17,<2",
        "onnx>=1.17,<2",
        "torch>=2.5,<3",
        "transformers>=4.45,<5",
        "huggingface_hub>=0.30,<1",
    )
)

hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-modnet-parity-check",
    image=parity_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
)


@app.cls(
    cpu=2,
    memory=4096,
    timeout=300,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class MODNetParity:

    @modal.method()
    def validate(self) -> dict:
        import os
        import numpy as np
        import torch
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel

        HF_REPO = "avaturn-live/avtr-1"
        H, W = 288, 512

        print("[parity] === MODNet Parity Check ===")

        # --- 1. Download modnet.onnx ---
        hf_token = os.environ.get("HF_TOKEN")
        print("[parity] Downloading modnet.onnx from HF ...")
        onnx_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="build_artifacts/modnet.onnx",
            token=hf_token,
            cache_dir=HF_CACHE_PATH,
        )
        print(f"[parity] ONNX path: {onnx_path}")

        # --- 2. Load ONNX model (CPU) ---
        print("[parity] Loading ONNX model (CPU) ...")
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        onnx_input_name = sess.get_inputs()[0].name
        onnx_output_name = sess.get_outputs()[0].name
        onnx_input_shape = sess.get_inputs()[0].shape
        onnx_output_shape = sess.get_outputs()[0].shape
        print(f"[parity] ONNX input: {onnx_input_name} {onnx_input_shape}")
        print(f"[parity] ONNX output: {onnx_output_name} {onnx_output_shape}")

        # --- 3. Load PyTorch MODNet ---
        print("[parity] Loading boopathiraj/MODNet (CPU) ...")
        modnet = AutoModel.from_pretrained(
            "boopathiraj/MODNet",
            trust_remote_code=True,
            token=hf_token,
        )
        modnet.eval()
        print(f"[parity] Model type: {type(modnet).__name__}")

        # --- 4. Inspect ONNX metadata ---
        import onnx
        onnx_model = onnx.load(onnx_path)
        print(f"[parity] ONNX opset: {onnx_model.opset_import[0].version}")
        print(f"[parity] ONNX producer: {onnx_model.producer_name} {onnx_model.producer_version}")
        onnx_meta = {
            "opset": onnx_model.opset_import[0].version,
            "producer": f"{onnx_model.producer_name} {onnx_model.producer_version}",
            "input_name": onnx_input_name,
            "input_shape": str(onnx_input_shape),
            "output_name": onnx_output_name,
            "output_shape": str(onnx_output_shape),
            "initializer_count": len(onnx_model.graph.initializer),
        }

        # --- 5. Generate test inputs ---
        torch.manual_seed(42)
        np.random.seed(42)

        test_cases = {}

        # B=1 test
        inp_b1 = torch.randn(1, 3, H, W, dtype=torch.float32)
        inp_b1_np = inp_b1.numpy()
        test_cases["b1"] = {
            "input_tensor": inp_b1,
            "input_numpy": inp_b1_np,
        }

        # B=5 test (only if ONNX supports dynamic batch)
        batch_dim = onnx_input_shape[0]
        supports_b5 = isinstance(batch_dim, str) or (isinstance(batch_dim, int) and batch_dim != 1)
        if supports_b5:
            inp_b5 = torch.randn(5, 3, H, W, dtype=torch.float32)
            test_cases["b5"] = {
                "input_tensor": inp_b5,
                "input_numpy": inp_b5.numpy(),
            }
            print("[parity] ONNX supports dynamic batch — will test B=5")
        else:
            print("[parity] ONNX is fixed B=1 — skipping B=5")

        # --- 6. Run parity comparison ---
        results = {
            "stage": "2B-parity",
            "component": "MODNet",
            "pytorch_source": "boopathiraj/MODNet",
            "onnx_source": f"{HF_REPO}/build_artifacts/modnet.onnx",
            "onnx_metadata": onnx_meta,
            "supports_b5": supports_b5,
            "parity": {},
        }

        for test_name, tc in test_cases.items():
            print(f"[parity] === Testing {test_name} ===")
            inp_t = tc["input_tensor"]
            inp_np = tc["input_numpy"]
            B = inp_t.shape[0]

            # ONNX forward
            onnx_out = sess.run(
                [onnx_output_name],
                {onnx_input_name: inp_np},
            )[0]
            print(f"[parity] ONNX output: shape={onnx_out.shape}, "
                  f"range=[{onnx_out.min():.6f}, {onnx_out.max():.6f}], "
                  f"mean={onnx_out.mean():.6f}")

            # PyTorch forward
            with torch.no_grad():
                _, _, matte = modnet(inp_t, True)
            pt_out = matte.numpy()
            print(f"[parity] PyTorch output: shape={pt_out.shape}, "
                  f"range=[{pt_out.min():.6f}, {pt_out.max():.6f}], "
                  f"mean={pt_out.mean():.6f}")

            # Shape check
            assert onnx_out.shape == pt_out.shape, \
                f"Shape mismatch: ONNX={onnx_out.shape}, PyTorch={pt_out.shape}"

            # Compute errors
            abs_err = np.abs(onnx_out - pt_out)
            mae = float(abs_err.mean())
            max_err = float(abs_err.max())
            flat_err = abs_err.flatten()
            p50 = float(np.percentile(flat_err, 50))
            p95 = float(np.percentile(flat_err, 95))
            p99 = float(np.percentile(flat_err, 99))

            # NaN/Inf checks
            onnx_nan = bool(np.isnan(onnx_out).any())
            onnx_inf = bool(np.isinf(onnx_out).any())
            pt_nan = bool(np.isnan(pt_out).any())
            pt_inf = bool(np.isinf(pt_out).any())

            # Output ranges
            onnx_range = {"min": float(onnx_out.min()), "max": float(onnx_out.max()),
                          "mean": float(onnx_out.mean()), "std": float(onnx_out.std())}
            pt_range = {"min": float(pt_out.min()), "max": float(pt_out.max()),
                        "mean": float(pt_out.mean()), "std": float(pt_out.std())}

            # Correlation
            onnx_flat = onnx_out.flatten()
            pt_flat = pt_out.flatten()
            correlation = float(np.corrcoef(onnx_flat, pt_flat)[0, 1])

            parity_result = {
                "shape": list(onnx_out.shape),
                "mae": round(mae, 6),
                "max_abs_error": round(max_err, 6),
                "p50_error": round(p50, 6),
                "p95_error": round(p95, 6),
                "p99_error": round(p99, 6),
                "correlation": round(correlation, 6),
                "onnx_nan": onnx_nan,
                "onnx_inf": onnx_inf,
                "pytorch_nan": pt_nan,
                "pytorch_inf": pt_inf,
                "onnx_range": {k: round(v, 6) for k, v in onnx_range.items()},
                "pytorch_range": {k: round(v, 6) for k, v in pt_range.items()},
            }

            results["parity"][test_name] = parity_result

            print(f"[parity] {test_name} results:")
            print(f"  MAE:       {mae:.6f}")
            print(f"  Max error: {max_err:.6f}")
            print(f"  p50:       {p50:.6f}")
            print(f"  p95:       {p95:.6f}")
            print(f"  p99:       {p99:.6f}")
            print(f"  Corr:      {correlation:.6f}")
            print(f"  ONNX NaN:  {onnx_nan}, Inf: {onnx_inf}")
            print(f"  PT NaN:    {pt_nan}, Inf: {pt_inf}")

        # --- 7. Overall verdict ---
        b1 = results["parity"].get("b1", {})
        mae = b1.get("mae", 999)
        max_err = b1.get("max_abs_error", 999)
        p99 = b1.get("p99_error", 999)

        if mae < 0.01 and p99 < 0.1:
            results["verdict"] = "PASS — MODNet PyTorch is numerically compatible with ONNX"
        elif mae < 0.05:
            results["verdict"] = "MARGINAL — some difference, review pixel-level"
        else:
            results["verdict"] = "FAIL — significant difference between PyTorch and ONNX"

        print(f"\n[parity] VERDICT: {results['verdict']}")
        return results
