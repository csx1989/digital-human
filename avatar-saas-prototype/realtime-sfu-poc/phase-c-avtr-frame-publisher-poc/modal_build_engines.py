#!/usr/bin/env python3
"""Build ALL AVTR1 TensorRT engines on the L40s volume.

One-time setup for the vallangiprasad workspace. Builds:
  Renderer: decoder, warp, modnet, stitch
  AVTR1: encode, decode, normalizer
  HuBERT: feature extractor

Usage:
  modal run modal_build_engines.py
"""

import modal

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"
CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])
    .apt_install("libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1", "ffmpeg", "git", "gcc", "g++", "pkg-config")
    .pip_install("torch>=2.8,<2.9", "torchaudio>=2.8,<2.9", "torchvision>=0.23,<0.24", extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy>=2.0,<3", "safetensors>=0.5,<1", "huggingface_hub>=1.3",
        "kornia>=0.8.1,<0.9", "roma>=1.5.4,<2", "opencv-python-headless>=4.11,<5",
        "einops>=0.8.1,<0.9", "scikit-image>=0.25,<0.26", "soxr>=0.4,<0.5",
        "imageio>=2.37,<3", "httpx>=0.28,<0.29", "tqdm>=4.67,<5",
        "onnx>=1.17,<2", "onnx2torch>=1.5,<2", "onnxruntime-gpu>=1.20",
        "onnx-graphsurgeon>=0.5,<1",
        "tensorrt<=10.12", "transformers>=5.3", "soundfile>=0.13",
    )
    .add_local_dir("/Users/ayya/developer/avtr-1/src/avtr1_renderer", remote_path="/opt/avtr1/avtr1_renderer", copy=True)
    .add_local_dir("/Users/ayya/developer/avtr-1/scripts", remote_path="/opt/avtr1/scripts", copy=True)
    .env({"PYTHONPATH": "/opt/avtr1", "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH})
)

app = modal.App("avtr-build-engines", image=image)
l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=True)


@app.function(
    gpu="L40S",
    volumes={L40S_MOUNT_PATH: l40s_volume, HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=1800,
)
def build():
    import sys
    sys.path.insert(0, "/opt/avtr1")
    sys.path.insert(0, "/opt/avtr1/scripts")

    # 1. Renderer engines (decoder, warp, modnet, stitch)
    print("=== Building renderer engines ===")
    import build_renderer_engines
    build_renderer_engines.main([])

    # 2. AVTR1 engines (encode, decode, normalizer)
    print("\n=== Building AVTR1 engines ===")
    import build_avtr1_engines
    build_avtr1_engines.main(["--ampere-plus"])

    # 3. HuBERT engine
    print("\n=== Building HuBERT engine ===")
    import build_hubert_engine
    build_hubert_engine.main([])

    l40s_volume.commit()
    print("\nAll engines built and volume committed.")


@app.local_entrypoint()
def main():
    build.remote()
