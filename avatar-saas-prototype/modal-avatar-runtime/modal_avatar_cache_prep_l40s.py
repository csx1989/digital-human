"""
ISOLATED ONE-TIME PREP APP — NOT PRODUCTION
Stage 3A: Pre-cache registered avatar data on L40S volume.

Uses the full production runtime deps (TRT, ONNX Runtime) to register
an avatar, then saves the resulting tensors to the volume. The pure-PyTorch
prototype loads these cached tensors without needing ONNX Runtime at all.

This app is deployed ONCE to cache avatar data, then need not be redeployed.

Cache output layout:
    {L40S_MOUNT_PATH}/avatars_pytorch_cache/{avatar_id}/
        kp_info_kp.pt        # (1, 21, 3)  canonical keypoints
        kp_info_exp.pt       # (1, 21, 3)  expression deltas
        kp_info_scale.pt     # (1, 1)      scale
        kp_info_t.pt         # (1, 3)      translation
        kp_info_pitch.pt     # (1, 1)
        kp_info_yaw.pt       # (1, 1)
        kp_info_roll.pt      # (1, 1)
        kp_info_R.pt         # (1, 3, 3)   rotation matrix
        f_s.pt               # (1, 32, 16, 64, 64) appearance feature volume
        M_grid.pt            # (2, 3)      precomputed affine grid for putback
        mask.pt              # (1, H, W)   pasteback mask
        source.pt            # (3, H, W)   source frame RGB [0,1]
        bg.pt                # (1, 3, H, W) background RGB [0,1]
        normalizer.pt        # dict of all normalizer tensors

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_avatar_cache_prep_l40s.py

Invoke:
    python -c "
import modal
cls = modal.Cls.from_name('avtr1-avatar-cache-prep-l40s', 'AvatarCachePrep')
inst = cls()
result = inst.cache_avatar_data.remote()
print(result)
    "
"""

from __future__ import annotations

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

app = modal.App("avtr1-avatar-cache-prep-l40s", image=image)

l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)


@app.cls(
    gpu="L40S",
    timeout=1800,
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
)
class AvatarCachePrep:

    @modal.method()
    def cache_avatar_data(self) -> dict:
        import torch
        from avtr1_renderer.pipeline import Pipeline
        from avtr1_renderer.backgrounds import load_background

        print("[cache] === Avatar Cache Prep ===")

        os.environ["AVTR1_LOCAL_STORAGE"] = L40S_MOUNT_PATH
        storage_root = Path(L40S_MOUNT_PATH)

        from avtr1_renderer.avtr1_artifact_manager import (
            get_artifact_manager,
            get_storage_root,
        )

        storage_root = get_storage_root()
        print(f"[cache] Storage root: {storage_root}")

        mgr = get_artifact_manager()
        mgr.ensure_all_artifacts(workers=4)
        print("[cache] Artifacts ensured")

        portraits_dir = storage_root / "avatars_artifacts" / "reference_frames"
        if not portraits_dir.is_dir():
            return {"ok": False, "error": f"No reference_frames at {portraits_dir}"}

        all_pngs = sorted(portraits_dir.glob("*.png"))
        if not all_pngs:
            return {"ok": False, "error": f"No portrait PNGs in {portraits_dir}"}

        avatar_id = all_pngs[0].stem
        print(f"[cache] Using avatar_id={avatar_id}")

        print("[cache] Loading pipeline...")
        t0 = time.monotonic()
        pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
        print(f"[cache] Pipeline loaded in {time.monotonic() - t0:.1f}s")

        avatar = registry[avatar_id]
        print(f"[cache] Avatar loaded: {avatar.id}")
        print(f"[cache]   f_s shape: {avatar.f_s.shape}")
        print(f"[cache]   M_grid shape: {avatar.M_grid.shape}")
        print(f"[cache]   mask shape: {avatar.mask.shape}")
        print(f"[cache]   source shape: {avatar.source.shape}")
        print(f"[cache]   kp_info.kp shape: {avatar.kp_info.kp.shape}")
        print(f"[cache]   no_matting: {avatar.no_matting}")

        cache_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache" / avatar_id
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(avatar.kp_info.kp.cpu(), cache_dir / "kp_info_kp.pt")
        torch.save(avatar.kp_info.exp.cpu(), cache_dir / "kp_info_exp.pt")
        torch.save(avatar.kp_info.scale.cpu(), cache_dir / "kp_info_scale.pt")
        torch.save(avatar.kp_info.t.cpu(), cache_dir / "kp_info_t.pt")
        torch.save(avatar.kp_info.pitch.cpu(), cache_dir / "kp_info_pitch.pt")
        torch.save(avatar.kp_info.yaw.cpu(), cache_dir / "kp_info_yaw.pt")
        torch.save(avatar.kp_info.roll.cpu(), cache_dir / "kp_info_roll.pt")
        torch.save(avatar.kp_info.R.cpu(), cache_dir / "kp_info_R.pt")
        torch.save(avatar.f_s.cpu(), cache_dir / "f_s.pt")
        torch.save(avatar.M_grid.cpu(), cache_dir / "M_grid.pt")
        torch.save(avatar.mask.cpu(), cache_dir / "mask.pt")
        torch.save(avatar.source.cpu(), cache_dir / "source.pt")

        H, W = avatar.source.shape[-2:]
        bg = load_background(None, H, W)
        torch.save(bg.cpu(), cache_dir / "bg.pt")

        normalizer_path = storage_root / "avtr1_normalizer.safetensors"
        from safetensors.torch import load_file
        normalizer_tensors = load_file(str(normalizer_path), device="cpu")
        torch.save(normalizer_tensors, cache_dir / "normalizer.pt")

        # Save no_matting flag for fast-path reconstruction
        import json
        with open(cache_dir / "no_matting.json", "w") as f:
            json.dump({"no_matting": bool(avatar.no_matting)}, f)

        l40s_volume.commit()

        cached_files = sorted(p.name for p in cache_dir.iterdir())
        total_mb = sum(p.stat().st_size for p in cache_dir.iterdir()) / 1e6

        print(f"[cache] Cached {len(cached_files)} files ({total_mb:.1f} MB)")
        print(f"[cache] Files: {cached_files}")
        print("[cache] Volume committed.")

        return {
            "ok": True,
            "avatar_id": avatar_id,
            "cache_dir": str(cache_dir),
            "files": cached_files,
            "total_mb": round(total_mb, 1),
            "shapes": {
                "kp_info_kp": list(avatar.kp_info.kp.shape),
                "kp_info_exp": list(avatar.kp_info.exp.shape),
                "kp_info_scale": list(avatar.kp_info.scale.shape),
                "kp_info_t": list(avatar.kp_info.t.shape),
                "kp_info_R": list(avatar.kp_info.R.shape),
                "f_s": list(avatar.f_s.shape),
                "M_grid": list(avatar.M_grid.shape),
                "mask": list(avatar.mask.shape),
                "source": list(avatar.source.shape),
                "bg": list(bg.shape),
            },
        }

    @modal.method()
    def verify_cache(self) -> dict:
        import torch

        cache_root = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache"
        if not cache_root.is_dir():
            return {"ok": False, "error": "No cache directory found"}

        avatars = sorted(d.name for d in cache_root.iterdir() if d.is_dir())
        if not avatars:
            return {"ok": False, "error": "No cached avatars found"}

        results = {}
        for av_id in avatars:
            av_dir = cache_root / av_id
            files = {p.stem: p.stat().st_size for p in av_dir.iterdir() if p.suffix == ".pt"}
            shapes = {}
            for p in sorted(av_dir.glob("*.pt")):
                try:
                    data = torch.load(str(p), map_location="cpu", weights_only=False)
                    if isinstance(data, dict):
                        shapes[p.name] = {k: list(v.shape) for k, v in data.items()}
                    elif isinstance(data, torch.Tensor):
                        shapes[p.name] = list(data.shape)
                    else:
                        shapes[p.name] = type(data).__name__
                except Exception as e:
                    shapes[p.name] = f"ERROR: {e}"
            results[av_id] = {"files": files, "shapes": shapes}

        return {"ok": True, "avatars": results}


@app.local_entrypoint()
def main():
    import json
    import modal

    cls = modal.Cls.from_name("avtr1-avatar-cache-prep-l40s", "AvatarCachePrep")
    inst = cls()

    print("=== Caching Avatar Data ===")
    result = inst.cache_avatar_data.remote()
    print(json.dumps(result, indent=2, default=str))

    if result.get("ok"):
        print("\n=== Verifying Cache ===")
        verify = inst.verify_cache.remote()
        print(json.dumps(verify, indent=2, default=str))
