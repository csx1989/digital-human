"""
ISOLATED INSPECTION — NOT PRODUCTION, NOT SNAPSHOT
Stage B1: HuBERT ONNX Architecture Inspector (v2 — full key dump)

Downloads hubert-lbs-avtr1.onnx from avaturn-live/avtr-1 HF repo and inspects:
- File size, opset, producer
- Input/output names, shapes, dtypes
- Initializer names, count, shape patterns
- Hidden size (from Linear weight shapes)
- Number of transformer layers
- Attention head count (if inferable)
- Whether it looks like HubertModel, HubertForCTC, or custom
- Output dimension (768 or 1024)
- Any projection/head beyond hidden states

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_hubert_inspector.py

Invoke:
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-hubert-inspector', 'HuBERTInspector')
inst = cls()
result = inst.inspect.remote()
print(json.dumps(result, indent=2, default=str))
    "
"""

from __future__ import annotations

import modal

HF_CACHE_PATH = "/root/.cache/huggingface"

inspector_image = (
    modal.Image.from_registry("python:3.12-slim")
    .apt_install("git")
    .pip_install(
        "numpy>=2.0,<3",
        "onnx>=1.17,<2",
        "huggingface_hub>=0.30,<1",
    )
)

hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-hubert-inspector",
    image=inspector_image,
    volumes={HF_CACHE_PATH: hf_cache_vol},
)


@app.cls(
    cpu=4,
    memory=8192,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class HuBERTInspector:

    @modal.method()
    def inspect(self) -> dict:
        import os
        import onnx
        from onnx import numpy_helper
        from huggingface_hub import hf_hub_download

        HF_REPO = "avaturn-live/avtr-1"
        HF_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"

        print("[inspect] === HuBERT ONNX Inspector ===")

        hf_token = os.environ.get("HF_TOKEN")
        print(f"[inspect] Downloading {HF_FILE} from {HF_REPO} ...")
        onnx_path = hf_hub_download(
            repo_id=HF_REPO,
            filename=HF_FILE,
            token=hf_token,
            cache_dir=HF_CACHE_PATH,
        )
        print(f"[inspect] Downloaded: {onnx_path}")

        import os as _os
        file_size = _os.path.getsize(onnx_path)
        print(f"[inspect] File size: {file_size / 1e6:.1f} MB")

        print("[inspect] Loading ONNX model ...")
        # Only need graph structure for inspection, not actual weight data
        model = onnx.load(onnx_path, load_external_data=False)

        # Basic metadata
        opset = model.opset_import[0].version if model.opset_import else "unknown"
        producer = f"{model.producer_name} {model.producer_version}"
        print(f"[inspect] Opset: {opset}, Producer: {producer}")

        # Inputs/outputs
        inputs = []
        for inp in model.graph.input:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(d.dim_value)
                elif d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            inputs.append({"name": inp.name, "shape": dims})
        print(f"[inspect] Inputs: {inputs}")

        outputs = []
        for out in model.graph.output:
            dims = []
            for d in out.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(d.dim_value)
                elif d.dim_param:
                    dims.append(d.dim_param)
                else:
                    dims.append("?")
            outputs.append({"name": out.name, "shape": dims})
        print(f"[inspect] Outputs: {outputs}")

        # Initializer analysis
        initializers = {}
        for init in model.graph.initializer:
            arr = numpy_helper.to_array(init)
            initializers[init.name] = {
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size": arr.size,
            }

        print(f"[inspect] Initializer count: {len(initializers)}")

        # Infer architecture from weight shapes
        hidden_size = None
        num_layers = 0
        output_dim = None
        has_prefix = False
        prefix = ""

        # Check for "hubert." prefix (HubertForCTC or HubertForSequenceClassification)
        sample_keys = list(initializers.keys())
        print(f"[inspect] All {len(sample_keys)} initializer keys:")
        for k in sample_keys:
            print(f"  {k}: {initializers[k]['shape']}")

        if any(k.startswith("hubert.") for k in sample_keys):
            has_prefix = True
            prefix = "hubert."
            print("[inspect] Detected 'hubert.' prefix — likely exported from HubertForCTC/HubertForAudioFrameClassification")

        # Find hidden_size from Linear weights
        p = prefix
        for name, info in initializers.items():
            if "projection.weight" in name and "feature_projection" in name:
                hidden_size = info["shape"][0]
                print(f"[inspect] Hidden size from {name}: {hidden_size}")
                break

        if hidden_size is None:
            for name, info in initializers.items():
                if "q_proj.weight" in name or "in_proj_weight" in name:
                    hidden_size = info["shape"][0]
                    print(f"[inspect] Hidden size from {name}: {hidden_size}")
                    break

        # Count layers
        layer_indices = set()
        for name in initializers:
            if "encoder.layers." in name:
                try:
                    parts = name.split("encoder.layers.")[1].split(".")
                    layer_indices.add(int(parts[0]))
                except (ValueError, IndexError):
                    pass
        num_layers = len(layer_indices) if layer_indices else 0
        print(f"[inspect] Transformer layers: {num_layers}")

        # Check output dim from last output
        if outputs:
            last_output = outputs[-1]
            if len(last_output["shape"]) >= 3:
                output_dim = last_output["shape"][-1]
                if isinstance(output_dim, str):
                    output_dim = "dynamic"
            print(f"[inspect] Output dim: {output_dim}")

        # Check for additional heads
        head_keys = [k for k in initializers if "lm_head" in k or "classifier" in k]
        has_lm_head = len(head_keys) > 0
        if has_lm_head:
            print(f"[inspect] Has lm_head/classifier: True (keys: {head_keys[:5]})")

        # Count conv layers in feature extractor
        conv_layers = set()
        for name in initializers:
            if "conv_layers." in name and "feature_extractor" in name:
                try:
                    parts = name.split("conv_layers.")[1].split(".")
                    conv_layers.add(int(parts[0]))
                except (ValueError, IndexError):
                    pass
        num_conv_layers = len(conv_layers)
        print(f"[inspect] Conv layers in feature extractor: {num_conv_layers}")

        # Attention heads inference
        attention_heads = None
        if hidden_size and num_layers:
            for name, info in initializers.items():
                if f"{p}encoder.layers.0.attention.q_proj.weight" == name:
                    q_weight = info["shape"]
                    if len(q_weight) == 2:
                        attention_heads = hidden_size // 64
                    break
                elif f"{p}encoder.layers.0.attention.k_proj.weight" == name:
                    attention_heads = hidden_size // 64
                    break
            if attention_heads:
                print(f"[inspect] Attention heads: {attention_heads}")

        # Determine model type
        model_type = "unknown"
        if has_lm_head:
            model_type = "HubertForCTC"
        elif has_prefix and not has_lm_head:
            model_type = "HubertModel (prefixed)"
        else:
            model_type = "HubertModel"

        # Determine base vs large
        arch_guess = "unknown"
        if hidden_size == 768 and num_layers == 12:
            arch_guess = "Hubert-Base"
        elif hidden_size == 1024 and num_layers == 24:
            arch_guess = "Hubert-Large"
        elif hidden_size == 1280 and num_layers == 48:
            arch_guess = "Hubert-XLarge"

        print(f"\n[inspect] === SUMMARY ===")
        print(f"  Model type: {model_type}")
        print(f"  Architecture guess: {arch_guess}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Layers: {num_layers}")
        print(f"  Attention heads: {attention_heads}")
        print(f"  Conv layers: {num_conv_layers}")
        print(f"  Output dim: {output_dim}")
        print(f"  Has head: {has_lm_head}")
        print(f"  File size: {file_size / 1e6:.1f} MB")

        # Print all unique key patterns
        print(f"\n[inspect] All unique key name patterns:")
        key_patterns = set()
        for name in initializers:
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                key_patterns.add(parts[0] + ".*")
            else:
                key_patterns.add(name)
        for pattern in sorted(key_patterns):
            count = sum(1 for k in initializers if k.startswith(pattern.replace(".*", ".")))
            print(f"  {pattern} (x{count})")

        return {
            "file_size_mb": round(file_size / 1e6, 1),
            "opset": opset,
            "producer": producer,
            "inputs": inputs,
            "outputs": outputs,
            "initializer_count": len(initializers),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_conv_layers": num_conv_layers,
            "attention_heads": attention_heads,
            "output_dim": output_dim,
            "model_type": model_type,
            "arch_guess": arch_guess,
            "has_lm_head": has_lm_head,
            "has_hubert_prefix": has_prefix,
            "all_keys": {k: initializers[k]["shape"] for k in sample_keys},
        }
