#!/usr/bin/env python3
"""Phase C1: AVTR frame extraction harness (Modal GPU dry-run).

Validates that AVTR-rendered I420 frames can be wrapped in PyAV
VideoFrame objects for WebRTC — WITHOUT publishing to SFU.

Correction #3: Separates warmup from steady-state latency.
- First chunk: cold start (expect slow)
- Chunks 2-5: warmup (stabilizing)
- Chunks 6+: steady-state (measure p50/p95)
- Real-time target: 5-frame chunk p95 < 200ms

Usage:
  python -m modal run modal_avtr_extractor.py --avatar-id arnold
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import modal

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"
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
        "onnxruntime-gpu>=1.20",
        "tensorrt<=10.12",
        "transformers>=5.3",
        "soundfile>=0.13",
    )
    .pip_install("cython")
    .pip_install("av>=12,<14")
    .add_local_dir(
        "/Users/ayya/developer/avtr-1/src/avtr1_renderer",
        remote_path="/opt/avtr1/avtr1_renderer",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/opt/avtr1",
            "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH,
        }
    )
)

app = modal.App("avtr-phase-c1-extractor", image=image)
l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=True)

WARMUP_CHUNKS = 5
STEADY_STATE_CHUNKS = 20
FPS = 25


@app.function(
    gpu="L40S",
    volumes={L40S_MOUNT_PATH: l40s_volume, HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=600,
)
def extract_frames(avatar_id: str = "arnold", bg_id: str = "aloe_window") -> dict:
    """Extract AVTR frames and validate format for WebRTC.

    Returns dict with:
    - frame_format: str (expected "yuv_i420")
    - frame_shape: tuple
    - frame_dtype: str
    - first_chunk_ms: float
    - warmup_ms: list[float]
    - steady_state_ms: list[float]
    - steady_p50_ms: float
    - steady_p95_ms: float
    - realtime_capable: bool (p95 < 200ms)
    - pyav_accepts: bool
    - pyav_width: int
    - pyav_height: int
    - pyav_format: str
    - error: str | None
    """
    import torch
    import numpy as np
    from avtr1_renderer.pipeline import Pipeline
    from avtr1_renderer.types import Chunk, RenderOptions
    from av import VideoFrame

    result: dict = {
        "avatar_id": avatar_id,
        "bg_id": bg_id,
        "error": None,
    }

    try:
        print(f"[c1] Building pipeline for avatar={avatar_id}...")
        t0 = time.monotonic()
        pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
        print(f"[c1] Pipeline built in {time.monotonic() - t0:.1f}s")

        avatar = registry[avatar_id]
        mg = pipeline._motion_generator
        chunk_size = mg.chunk_size      # 5
        future_size = mg.future_size    # 5
        frame_len = mg.frame_len        # 640
        audio_shift = getattr(mg, "audio_shift", 0)  # 80
        window = (chunk_size + future_size) * frame_len + audio_shift  # 6480
        step_samples = chunk_size * frame_len  # 3200
        silence = np.zeros(window, dtype=np.float32)

        # Pick background
        if bg_id not in pipeline._backgrounds:
            available = sorted(pipeline._backgrounds.keys())
            print(f"[c1] bg_id={bg_id!r} not found. Available: {available[:10]}")
            bg_id = available[0] if available else None
            if not bg_id:
                raise RuntimeError("No backgrounds registered")
        print(f"[c1] Using bg_id={bg_id!r}")

        options = RenderOptions(pixel_format="yuv_i420", bg_id=bg_id)
        state = pipeline.initial_state(avatar)
        audio_history = np.zeros(0, dtype=np.float32)

        total_chunks = 1 + WARMUP_CHUNKS + STEADY_STATE_CHUNKS
        all_latencies: list[float] = []
        sample_frame_data = None

        for chunk_i in range(total_chunks):
            # Build audio window
            audio_history = np.concatenate([audio_history, np.zeros(step_samples, dtype=np.float32)])
            if len(audio_history) >= window:
                audio_tail = audio_history[-window:].copy()
            else:
                audio_tail = np.pad(audio_history, (window - len(audio_history), 0))
            max_hist = window + step_samples * 2
            if len(audio_history) > max_hist:
                audio_history = audio_history[-window:]

            chunk_obj = Chunk(audio_speech=audio_tail, audio_listen=silence)

            t_start = time.monotonic()
            with torch.inference_mode():
                new_state, frames_iter = pipeline.process_chunk(
                    avatar, chunk_obj, state=state, options=options,
                )
                frames_list = list(frames_iter)
            render_ms = (time.monotonic() - t_start) * 1000
            state = new_state

            all_latencies.append(render_ms)

            if chunk_i == 0:
                result["first_chunk_ms"] = round(render_ms, 1)
                print(f"[c1] Chunk 0 (cold): {render_ms:.1f}ms, {len(frames_list)} frames")
            elif chunk_i <= WARMUP_CHUNKS:
                print(f"[c1] Chunk {chunk_i} (warmup): {render_ms:.1f}ms")
            else:
                if chunk_i == WARMUP_CHUNKS + 1:
                    print(f"[c1] --- Steady state begins ---")
                if chunk_i % 5 == 0 or chunk_i == total_chunks - 1:
                    print(f"[c1] Chunk {chunk_i} (steady): {render_ms:.1f}ms")

            # Capture first frame for format validation
            if sample_frame_data is None and frames_list:
                sample_frame_data = frames_list[0].data.copy()
                result["frame_format"] = frames_list[0].format
                result["frame_shape"] = list(frames_list[0].data.shape)
                result["frame_dtype"] = str(frames_list[0].data.dtype)
                result["frame_height"] = frames_list[0].height
                result["frame_width"] = frames_list[0].width
                print(f"[c1] Frame format: {result['frame_format']}")
                print(f"[c1] Frame shape: {result['frame_shape']}")
                print(f"[c1] Frame dtype: {result['frame_dtype']}")
                print(f"[c1] Frame HxW: {result['frame_height']}x{result['frame_width']}")
                print(f"[c1] First 16 bytes (hex): {sample_frame_data.ravel()[:16].tobytes().hex()}")

        # Split latencies
        result["warmup_ms"] = [round(x, 1) for x in all_latencies[1 : 1 + WARMUP_CHUNKS]]
        steady = all_latencies[1 + WARMUP_CHUNKS :]
        result["steady_state_ms"] = [round(x, 1) for x in steady]

        steady_sorted = sorted(steady)
        n = len(steady_sorted)
        result["steady_p50_ms"] = round(steady_sorted[n // 2], 1) if n > 0 else None
        result["steady_p95_ms"] = round(steady_sorted[int(n * 0.95)], 1) if n > 0 else None
        result["realtime_capable"] = result["steady_p95_ms"] is not None and result["steady_p95_ms"] < 200.0

        print(f"\n[c1] === LATENCY SUMMARY ===")
        print(f"[c1] First chunk (cold): {result.get('first_chunk_ms', 'N/A')}ms")
        print(f"[c1] Warmup chunks: {result['warmup_ms']}")
        print(f"[c1] Steady-state p50: {result['steady_p50_ms']}ms")
        print(f"[c1] Steady-state p95: {result['steady_p95_ms']}ms")
        print(f"[c1] Real-time capable (<200ms p95): {result['realtime_capable']}")

        # Validate PyAV acceptance
        if sample_frame_data is not None:
            try:
                vf = VideoFrame.from_ndarray(sample_frame_data, format="yuv420p")
                result["pyav_accepts"] = True
                result["pyav_width"] = vf.width
                result["pyav_height"] = vf.height
                result["pyav_format"] = vf.format.name
                print(f"\n[c1] === PYAV VALIDATION ===")
                print(f"[c1] VideoFrame.from_ndarray: OK")
                print(f"[c1] PyAV width={vf.width}, height={vf.height}, format={vf.format.name}")
                print(f"[c1] Planes: {len(vf.planes)}")
                for i, p in enumerate(vf.planes):
                    print(f"[c1]   plane[{i}]: {p.width}x{p.height} line_size={p.line_size}")
            except Exception as exc:
                result["pyav_accepts"] = False
                result["pyav_error"] = str(exc)
                print(f"[c1] PyAV FAILED: {exc}")

        print(f"\n[c1] === EXTRACTION COMPLETE ===")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        result["error"] = str(exc)

    return result


@app.local_entrypoint()
def main(avatar_id: str = "arnold", bg_id: str = "aloe_window"):
    result = extract_frames.remote(avatar_id=avatar_id, bg_id=bg_id)
    print("\n=== C1 RESULT ===")
    for k, v in sorted(result.items()):
        if isinstance(v, list) and len(v) > 10:
            print(f"  {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} items)")
        else:
            print(f"  {k}: {v}")
