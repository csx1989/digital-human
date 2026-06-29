"""
ISOLATED EXPERIMENTAL APP — NOT PRODUCTION
A100-40GB Stability Test: Same v11 no-CUDA-graph architecture on A100-40GB.

Goal: Test whether Modal GPU snapshot SIGSEGV 139 persists on A100-40GB
when using the same shared-model snapshot architecture that failed on L40S.

Architecture:
  - snap=True: Load only shared models. NO avatar tensors. NO CUDA graph.
  - snap=False (_post_restore): Verify models, init decode buffers, reset avatar cache.
  - load_avatar(avatar_id): Load avatar tensors from Modal Volume on demand.
  - render_smoke(avatar_id): Load + render one chunk to verify correctness.
  - All render methods take avatar_data dict explicitly (no global mutable state).

Avatar cache:
  - OrderedDict-based LRU, max 3 entries.
  - ~35 MB GPU VRAM per avatar (source + bg + f_s + mask + small tensors).
  - Eviction: pop oldest, clear dict, torch.cuda.empty_cache().

Avatar volume layout:
  /data/avtr1-l40s/avatars_pytorch_cache/<avatar_id>/
    kp_info_kp.pt, kp_info_exp.pt, ..., kp_info_R.pt  (8 files, tiny)
    f_s.pt              (~8 MB)
    M_grid.pt           (tiny)
    mask.pt             (~3.5 MB)
    source.pt           (~10.5 MB)
    bg.pt               (~10.5 MB)
    normalizer.pt       (dict, tiny)

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_shared_model_snapshot_a100_40gb.py

Invoke (NEVER modal run):
    python -c "
import modal, json, time
cls = modal.Cls.from_name('avtr1-shared-model-snapshot-a100-40gb', 'SharedModelSnapshotA10040GB')
inst = cls()
print(json.dumps(inst.health_check.remote(), indent=2))
time.sleep(30)
print(json.dumps(inst.health_check.remote(), indent=2))
    "
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import modal

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"
MAX_CACHE_ENTRIES = 3

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
        "torchvision>=0.23,<0.24",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
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
        "httpx>=0.28,<0.29",
        "tqdm>=4.67,<5",
        "onnx>=1.17,<2",
        "onnx2torch>=1.5,<2",
        "transformers>=4.45,<5",
    )
)

app = modal.App("avtr1-shared-model-snapshot-a100-40gb", image=image)

l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)


LIPSYNC_COORDS = (
    3, 4, 5, 6, 7, 8, 18, 19, 20,
    33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 43, 44, 45, 46, 47, 48, 49, 50,
    51, 52, 53, 54, 55, 56, 57, 58, 59,
    60, 61, 62,
)
N_LIPSYNC = len(LIPSYNC_COORDS)
_LIPSYNC_INDEX = list(LIPSYNC_COORDS)
Z_SCORE_CLIP = 5.0
MODNET_INPUT_H = 288
MODNET_INPUT_W = 512


def _run_nvidia_smi(label: str) -> None:
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            print(f"[nvidia-smi] {label}: {result.stdout.strip()}")
    except Exception:
        pass


def _gpu_memory_snapshot(label: str) -> dict:
    import subprocess
    import torch
    snap: dict = {"label": label}
    snap["torchAllocMb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 3:
                snap["smiUsedMb"] = int(parts[0])
                snap["smiTotalMb"] = int(parts[1])
                snap["smiUtilPct"] = int(parts[2])
    except Exception:
        pass
    print(f"[gpu] {label}: torch={snap.get('torchAllocMb')}MB smi={snap.get('smiUsedMb')}MB")
    return snap


def _build_renderer_model_classes():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.spectral_norm as spectral_norm

    def _make_coordinate_grid(spatial_size, ref):
        d, h, w = spatial_size
        x = torch.arange(w, dtype=ref.dtype, device=ref.device)
        y = torch.arange(h, dtype=ref.dtype, device=ref.device)
        z = torch.arange(d, dtype=ref.dtype, device=ref.device)
        x = 2.0 * (x / (w - 1)) - 1.0
        y = 2.0 * (y / (h - 1)) - 1.0
        z = 2.0 * (z / (d - 1)) - 1.0
        yy = y.view(1, -1, 1).repeat(d, 1, w)
        xx = x.view(1, 1, -1).repeat(d, h, 1)
        zz = z.view(-1, 1, 1).repeat(1, h, w)
        return torch.stack([xx, yy, zz], dim=-1)

    def _kp2gaussian(kp, spatial_size, kp_variance):
        mean = kp
        coordinate_grid = _make_coordinate_grid(spatial_size, mean)
        ndim = len(mean.shape) - 1
        shape = (1,) * ndim + coordinate_grid.shape
        coordinate_grid = coordinate_grid.view(*shape)
        repeats = mean.shape[:ndim] + (1, 1, 1, 1)
        coordinate_grid = coordinate_grid.repeat(*repeats)
        shape = mean.shape[:ndim] + (1, 1, 1, 3)
        mean = mean.view(*shape)
        mean_sub = coordinate_grid - mean
        return torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance)

    class ResBlock3d(nn.Module):
        def __init__(self, in_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv1 = nn.Conv3d(in_features, in_features, kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(in_features, in_features, kernel_size, padding=padding)
            self.norm1 = nn.BatchNorm3d(in_features, affine=True)
            self.norm2 = nn.BatchNorm3d(in_features, affine=True)

        def forward(self, x):
            out = F.relu(self.norm1(x))
            out = self.conv1(out)
            out = F.relu(self.norm2(out))
            out = self.conv2(out)
            return out + x

    class UpBlock3d(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv3d(in_features, out_features, kernel_size, padding=padding)
            self.norm = nn.BatchNorm3d(out_features, affine=True)

        def forward(self, x):
            out = F.interpolate(x, scale_factor=(1, 2, 2))
            out = self.conv(out)
            out = self.norm(out)
            return F.relu(out)

    class DownBlock3d(nn.Module):
        def __init__(self, in_features, out_features, kernel_size=3, padding=1):
            super().__init__()
            self.conv = nn.Conv3d(in_features, out_features, kernel_size, padding=padding)
            self.norm = nn.BatchNorm3d(out_features, affine=True)
            self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        def forward(self, x):
            out = self.conv(x)
            out = self.norm(out)
            out = F.relu(out)
            return self.pool(out)

    class HourglassEncoder(nn.Module):
        def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
            super().__init__()
            blocks = []
            for i in range(num_blocks):
                blocks.append(DownBlock3d(
                    in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                    min(max_features, block_expansion * (2 ** (i + 1))),
                ))
            self.down_blocks = nn.ModuleList(blocks)

        def forward(self, x):
            outs = [x]
            for block in self.down_blocks:
                outs.append(block(outs[-1]))
            return outs

    class HourglassDecoder(nn.Module):
        def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
            super().__init__()
            up_blocks = []
            for i in range(num_blocks)[::-1]:
                in_f = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
                out_f = min(max_features, block_expansion * (2 ** i))
                up_blocks.append(UpBlock3d(in_f, out_f))
            self.up_blocks = nn.ModuleList(up_blocks)
            self.out_filters = block_expansion + in_features
            self.conv = nn.Conv3d(self.out_filters, self.out_filters, 3, padding=1)
            self.norm = nn.BatchNorm3d(self.out_filters, affine=True)

        def forward(self, x):
            out = x.pop()
            for block in self.up_blocks:
                out = block(out)
                skip = x.pop()
                out = torch.cat([out, skip], dim=1)
            out = self.conv(out)
            out = self.norm(out)
            return F.relu(out)

    class Hourglass(nn.Module):
        def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
            super().__init__()
            self.encoder = HourglassEncoder(block_expansion, in_features, num_blocks, max_features)
            self.decoder = HourglassDecoder(block_expansion, in_features, num_blocks, max_features)
            self.out_filters = self.decoder.out_filters

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class DenseMotionNetwork(nn.Module):
        def __init__(self, block_expansion, num_blocks, max_features, num_kp,
                     feature_channel, reshape_depth, compress, estimate_occlusion_map=True):
            super().__init__()
            self.hourglass = Hourglass(
                block_expansion=block_expansion,
                in_features=(num_kp + 1) * (compress + 1),
                max_features=max_features, num_blocks=num_blocks,
            )
            self.mask = nn.Conv3d(self.hourglass.out_filters, num_kp + 1, kernel_size=7, padding=3)
            self.compress = nn.Conv3d(feature_channel, compress, kernel_size=1)
            self.norm = nn.BatchNorm3d(compress, affine=True)
            self.num_kp = num_kp
            self.flag_estimate_occlusion_map = estimate_occlusion_map
            if self.flag_estimate_occlusion_map:
                self.occlusion = nn.Conv2d(self.hourglass.out_filters * reshape_depth, 1, kernel_size=7, padding=3)

        def _sparse_motions(self, feature, kp_driving, kp_source):
            bs, _, d, h, w = feature.shape
            identity_grid = _make_coordinate_grid((d, h, w), ref=kp_source)
            identity_grid = identity_grid.view(1, 1, d, h, w, 3)
            coordinate_grid = identity_grid - kp_driving.view(bs, self.num_kp, 1, 1, 1, 3)
            driving_to_source = coordinate_grid + kp_source.view(bs, self.num_kp, 1, 1, 1, 3)
            identity_grid = identity_grid.repeat(bs, 1, 1, 1, 1, 1)
            return torch.cat([identity_grid, driving_to_source], dim=1)

        def _deformed_feature(self, feature, sparse_motions):
            bs, _, d, h, w = feature.shape
            feat_rep = feature.unsqueeze(1).unsqueeze(1).repeat(1, self.num_kp + 1, 1, 1, 1, 1, 1)
            feat_rep = feat_rep.view(bs * (self.num_kp + 1), -1, d, h, w)
            sm_flat = sparse_motions.view(bs * (self.num_kp + 1), d, h, w, -1)
            deformed = F.grid_sample(feat_rep, sm_flat, align_corners=False)
            return deformed.view(bs, self.num_kp + 1, -1, d, h, w)

        def _heatmaps(self, feature, kp_driving, kp_source):
            spatial_size = feature.shape[3:]
            g_d = _kp2gaussian(kp_driving, spatial_size=spatial_size, kp_variance=0.01)
            g_s = _kp2gaussian(kp_source, spatial_size=spatial_size, kp_variance=0.01)
            heatmap = g_d - g_s
            zeros = torch.zeros(heatmap.shape[0], 1, *spatial_size, dtype=heatmap.dtype, device=heatmap.device)
            heatmap = torch.cat([zeros, heatmap], dim=1)
            return heatmap.unsqueeze(2)

        def forward(self, feature, kp_driving, kp_source):
            bs, _, d, h, w = feature.shape
            feature = F.relu(self.norm(self.compress(feature)))
            sparse_motion = self._sparse_motions(feature, kp_driving, kp_source)
            deformed_feature = self._deformed_feature(feature, sparse_motion)
            heatmap = self._heatmaps(deformed_feature, kp_driving, kp_source)
            inp = torch.cat([heatmap, deformed_feature], dim=2).view(bs, -1, d, h, w)
            prediction = self.hourglass(inp)
            mask = F.softmax(self.mask(prediction), dim=1).unsqueeze(2)
            sparse_motion = sparse_motion.permute(0, 1, 5, 2, 3, 4)
            deformation = (sparse_motion * mask).sum(dim=1).permute(0, 2, 3, 4, 1)
            out = {"deformation": deformation}
            if self.flag_estimate_occlusion_map:
                pred_2d = prediction.view(bs, -1, h, w)
                out["occlusion_map"] = torch.sigmoid(self.occlusion(pred_2d))
            return out

    class SameBlock2d(nn.Module):
        def __init__(self, in_f, out_f, kernel_size=3, padding=1, lrelu=False):
            super().__init__()
            self.conv = nn.Conv2d(in_f, out_f, kernel_size, padding=padding)
            self.norm = nn.BatchNorm2d(out_f, affine=True)
            self.ac = nn.LeakyReLU() if lrelu else nn.ReLU()

        def forward(self, x):
            return self.ac(self.norm(self.conv(x)))

    class SPADE(nn.Module):
        def __init__(self, norm_nc, label_nc):
            super().__init__()
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
            nhidden = 128
            self.mlp_shared = nn.Sequential(nn.Conv2d(label_nc, nhidden, 3, padding=1), nn.ReLU())
            self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, 3, padding=1)
            self.mlp_beta = nn.Conv2d(nhidden, norm_nc, 3, padding=1)

        def forward(self, x, segmap):
            normalized = self.param_free_norm(x)
            segmap = F.interpolate(segmap, size=x.shape[2:], mode="nearest")
            actv = self.mlp_shared(segmap)
            return normalized * (1 + self.mlp_gamma(actv)) + self.mlp_beta(actv)

    class SPADEResnetBlock(nn.Module):
        def __init__(self, fin, fout, norm_G, label_nc):
            super().__init__()
            self.learned_shortcut = fin != fout
            fmiddle = min(fin, fout)
            self.conv_0 = nn.Conv2d(fin, fmiddle, 3, padding=1)
            self.conv_1 = nn.Conv2d(fmiddle, fout, 3, padding=1)
            if self.learned_shortcut:
                self.conv_s = nn.Conv2d(fin, fout, 1, bias=False)
            if "spectral" in norm_G:
                self.conv_0 = spectral_norm(self.conv_0)
                self.conv_1 = spectral_norm(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = spectral_norm(self.conv_s)
            self.norm_0 = SPADE(fin, label_nc)
            self.norm_1 = SPADE(fmiddle, label_nc)
            if self.learned_shortcut:
                self.norm_s = SPADE(fin, label_nc)

        def forward(self, x, seg1):
            x_s = self._shortcut(x, seg1)
            dx = self.conv_0(F.leaky_relu(self.norm_0(x, seg1), 2e-1))
            dx = self.conv_1(F.leaky_relu(self.norm_1(dx, seg1), 2e-1))
            return x_s + dx

        def _shortcut(self, x, seg1):
            if self.learned_shortcut:
                return self.conv_s(self.norm_s(x, seg1))
            return x

    class StitchingNetwork(nn.Module):
        def __init__(self, input_size=126, hidden_sizes=None, output_size=65):
            super().__init__()
            if hidden_sizes is None:
                hidden_sizes = [128, 128, 64]
            layers = []
            for i, h in enumerate(hidden_sizes):
                layers.append(nn.Linear(input_size if i == 0 else hidden_sizes[i - 1], h))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(hidden_sizes[-1], output_size))
            self.mlp = nn.Sequential(*layers)

        def forward(self, kp_source, kp_driving):
            bs = kp_source.shape[0]
            delta = self.mlp(torch.cat([kp_source.view(bs, -1), kp_driving.view(bs, -1)], dim=1))
            num_kp = kp_source.shape[1]
            delta_exp = delta[..., :3 * num_kp].reshape(bs, num_kp, 3)
            delta_tx_ty = delta[..., 3 * num_kp:3 * num_kp + 2].reshape(bs, 1, 2)
            out = kp_driving.clone()
            out = out + delta_exp
            out[..., :2] = out[..., :2] + delta_tx_ty
            return out

    class WarpingNetwork(nn.Module):
        def __init__(self, num_kp=21, block_expansion=64, max_features=512,
                     num_down_blocks=2, reshape_channel=32,
                     estimate_occlusion_map=True, flag_use_occlusion_map=True):
            super().__init__()
            self.flag_use_occlusion_map = flag_use_occlusion_map
            self.dense_motion_network = DenseMotionNetwork(
                num_kp=num_kp, feature_channel=reshape_channel,
                estimate_occlusion_map=estimate_occlusion_map,
                block_expansion=32, max_features=1024, num_blocks=5,
                reshape_depth=16, compress=4,
            )
            self.third = SameBlock2d(max_features, block_expansion * (2 ** num_down_blocks), lrelu=True)
            self.fourth = nn.Conv2d(block_expansion * (2 ** num_down_blocks),
                                    block_expansion * (2 ** num_down_blocks), kernel_size=1)

        def forward(self, feature_3d, kp_source, kp_driving):
            dm = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving, kp_source=kp_source)
            occ = dm.get("occlusion_map")
            out = F.grid_sample(feature_3d, dm["deformation"], align_corners=False)
            bs, c, d, h, w = out.shape
            out = out.view(bs, c * d, h, w)
            out = self.third(out)
            out = self.fourth(out)
            if self.flag_use_occlusion_map and occ is not None:
                out = out * occ
            return out

    class SPADEDecoder(nn.Module):
        def __init__(self, upscale=2, max_features=512, block_expansion=64,
                     out_channels=64, num_down_blocks=2):
            super().__init__()
            self.upscale = upscale
            ic = min(max_features, block_expansion * (2 ** num_down_blocks))
            lbl = ic
            nG = "spadespectralinstance"
            self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
            self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, nG, lbl)
            self.up_0 = SPADEResnetBlock(2 * ic, ic, nG, lbl)
            self.up_1 = SPADEResnetBlock(ic, out_channels, nG, lbl)
            self.up = nn.Upsample(scale_factor=2)
            if self.upscale is None or self.upscale <= 1:
                self.conv_img = nn.Conv2d(out_channels, 3, 3, padding=1)
            else:
                self.conv_img = nn.Sequential(
                    nn.Conv2d(out_channels, 3 * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(upscale_factor=2),
                )

        def forward(self, feature):
            seg = feature
            x = self.fc(feature)
            x = self.G_middle_0(x, seg)
            x = self.G_middle_1(x, seg)
            x = self.G_middle_2(x, seg)
            x = self.G_middle_3(x, seg)
            x = self.G_middle_4(x, seg)
            x = self.G_middle_5(x, seg)
            x = self.up(x)
            x = self.up_0(x, seg)
            x = self.up(x)
            x = self.up_1(x, seg)
            return torch.sigmoid(self.conv_img(F.leaky_relu(x, 2e-1)))

    return {
        "StitchingNetwork": StitchingNetwork,
        "WarpingNetwork": WarpingNetwork,
        "SPADEDecoder": SPADEDecoder,
    }


def _remove_ddp_prefix(state_dict):
    cleaned = {}
    for k, v in state_dict.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_k] = v
    return cleaned


def _modnet_forward(model, x):
    _, _, matte = model(x, True)
    return matte


@app.cls(
    gpu="A100-40GB",
    timeout=1800,
    volumes={
        L40S_MOUNT_PATH: l40s_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
)
class SharedModelSnapshotA10040GB:

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import os
        import torch
        import numpy as np
        from huggingface_hub import hf_hub_download

        self._snap_t0 = time.monotonic()
        print("[snap-a100] === Shared Model Snapshot A100-40GB — Snap Init ===")
        self._snap_ok = False
        self._snap_error = None

        device = torch.device("cuda")
        cache_dir = str(Path(HF_CACHE_PATH) / "hub")

        print("[snap-a100] Loading AVTR TorchScript model...")
        t0 = time.monotonic()
        avtr_path = hf_hub_download(
            repo_id="avaturn-live/avtr-1",
            filename="build_artifacts/avtr1.scripted.pt",
            repo_type="model",
            cache_dir=cache_dir,
        )
        self.scripted = torch.jit.load(str(avtr_path), map_location="cuda").eval()
        inner = self.scripted.model
        self.chunk_size = int(inner.chunk_size)
        self.past_size = int(inner.past_size)
        self.future_size = int(inner.future_size)
        self.nfeats = int(inner.nfeats)
        self.cond_feature_dim = int(inner.cond_feature_dim)
        self.latent_dim = int(inner.latent_dim)
        self.n_ode_steps = 5
        print(f"[snap-a100] AVTR loaded in {time.monotonic() - t0:.1f}s "
              f"(chunk={self.chunk_size} past={self.past_size} nfeats={self.nfeats})")

        print("[snap-a100] Loading HuBERT via onnx2torch...")
        t0 = time.monotonic()
        from onnx2torch import convert
        import onnx
        from onnx import numpy_helper
        import tempfile

        hubert_onnx_path = hf_hub_download(
            repo_id="avaturn-live/avtr-1",
            filename="build_artifacts/hubert-lbs-avtr1.onnx",
            repo_type="model",
            cache_dir=cache_dir,
        )
        onnx_model = onnx.load(hubert_onnx_path)
        initializer_names = {init.name for init in onnx_model.graph.initializer}
        for node in onnx_model.graph.node:
            if node.op_type == "Conv" and len(node.input) >= 3:
                bias_name = node.input[2]
                if bias_name and bias_name not in initializer_names:
                    weight_init = next(
                        (i for i in onnx_model.graph.initializer if i.name == node.input[1]), None
                    )
                    if weight_init is not None:
                        out_channels = weight_init.dims[0]
                        zero_bias = numpy_helper.from_array(
                            np.zeros(out_channels, dtype=np.float32), name=bias_name
                        )
                        onnx_model.graph.initializer.append(zero_bias)
                        initializer_names.add(bias_name)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            onnx.save(onnx_model, f.name)
            fixed_path = f.name
        self.hubert_model = convert(fixed_path).cuda().eval()
        self.HUBERT_PAST_FRAMES = 3
        self.AUDIO_DIM = 1024
        self.frame_len = 640
        self.audio_shift = 80
        print(f"[snap-a100] HuBERT loaded in {time.monotonic() - t0:.1f}s")

        print("[snap-a100] Loading renderer models (stitch/warp/decoder)...")
        t0 = time.monotonic()
        renderer_classes = _build_renderer_model_classes()
        HF_REPO = "digital-avatar/ditto-talkinghead"
        WEIGHT_PREFIX = "ditto_pytorch/models"

        self.stitch_model = renderer_classes["StitchingNetwork"]()
        self.warp_model = renderer_classes["WarpingNetwork"]()
        self.decoder_model = renderer_classes["SPADEDecoder"]()

        stitch_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{WEIGHT_PREFIX}/stitch_network.pth",
            repo_type="model", cache_dir=cache_dir,
        )
        stitch_ckpt = torch.load(stitch_path, map_location="cpu", weights_only=False)
        stitch_sd = _remove_ddp_prefix(stitch_ckpt["retarget_shoulder"])
        self.stitch_model.load_state_dict(stitch_sd)
        self.stitch_model.cuda().half().eval()

        warp_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{WEIGHT_PREFIX}/warp_network.pth",
            repo_type="model", cache_dir=cache_dir,
        )
        warp_sd = torch.load(warp_path, map_location="cpu", weights_only=False)
        self.warp_model.load_state_dict(warp_sd)
        self.warp_model.cuda().half().eval()

        decoder_path = hf_hub_download(
            repo_id=HF_REPO, filename=f"{WEIGHT_PREFIX}/decoder.pth",
            repo_type="model", cache_dir=cache_dir,
        )
        decoder_sd = torch.load(decoder_path, map_location="cpu", weights_only=False)
        decoder_sd_float = {k: v.float() for k, v in decoder_sd.items()}
        self.decoder_model.load_state_dict(decoder_sd_float)
        self.decoder_model.cuda().half().eval()

        print(f"[snap-a100] Renderer models loaded in {time.monotonic() - t0:.1f}s")

        print("[snap-a100] Loading MODNet...")
        t0 = time.monotonic()
        from transformers import AutoModel
        hf_token = os.environ.get("HF_TOKEN")
        self.modnet = AutoModel.from_pretrained(
            "boopathiraj/MODNet", trust_remote_code=True, token=hf_token,
        )
        self.modnet.cuda().eval()
        print(f"[snap-a100] MODNet loaded in {time.monotonic() - t0:.1f}s")

        self._past_times = torch.ones(
            (1, self.past_size // self.chunk_size, 1), dtype=torch.float32, device="cuda"
        )

        print("[snap-a100] Warmup...")
        wt0 = time.monotonic()
        with torch.no_grad():
            self.hubert_model(torch.randn(2, 8400, device="cuda"))
            _modnet_forward(self.modnet, torch.randn(1, 3, MODNET_INPUT_H, MODNET_INPUT_W, device="cuda"))
        print(f"[snap-a100] Warmup done in {time.monotonic() - wt0:.1f}s")

        self._avatar_cache: OrderedDict = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

        self._snap_ok = True
        self._snap_elapsed = time.monotonic() - self._snap_t0
        _gpu_memory_snapshot("snap-init-end-a100")
        print(f"[snap-a100] === Snap init done in {self._snap_elapsed:.1f}s ===")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        self._restore_t0 = time.monotonic()
        print("[restore] === Post-restore init ===")
        print(f"[restore] CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM: {vram:.0f} MB")
        _run_nvidia_smi("restore")

        assert hasattr(self, "scripted"), "AVTR TorchScript NOT found after restore"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found after restore"
        assert hasattr(self, "stitch_model"), "Stitch NOT found after restore"
        assert hasattr(self, "warp_model"), "Warp NOT found after restore"
        assert hasattr(self, "decoder_model"), "Decoder NOT found after restore"
        assert hasattr(self, "modnet"), "MODNet NOT found after restore"
        assert hasattr(self, "_past_times"), "_past_times NOT found after restore"

        self._decode_graph = None
        self._decode_path = "eager_static"
        self._dec_static_x = torch.empty(1, self.chunk_size, self.nfeats, device="cuda")
        self._dec_w_self = torch.full((self.latent_dim,), 1.0, device="cuda")
        self._dec_w_other = torch.full((self.latent_dim,), 1.0, device="cuda")
        self._dec_w_kp = torch.full((self.latent_dim,), 1.0, device="cuda")
        self._dec_times = torch.linspace(0, 1, self.n_ode_steps, device="cuda").view(-1, 1, 1)
        print(f"[restore] Decode buffers allocated (eager_static path)")

        self._avatar_cache = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_evictions = 0

        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Cache reset. Done in {self._restore_elapsed:.4f}s")

    def _get_avatar(self, avatar_id: str) -> dict:
        import torch

        if avatar_id in self._avatar_cache:
            self._avatar_cache.move_to_end(avatar_id)
            self._cache_hits += 1
            print(f"[cache] HIT: {avatar_id}")
            return self._avatar_cache[avatar_id]

        self._cache_misses += 1
        print(f"[cache] MISS: {avatar_id} — loading from volume")

        t0 = time.monotonic()
        av_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache" / avatar_id
        if not av_dir.is_dir():
            raise FileNotFoundError(f"Avatar cache not found: {av_dir}")

        device = torch.device("cuda")
        av = {}

        t_load = time.monotonic()
        for name in ("kp_info_kp", "kp_info_exp", "kp_info_scale", "kp_info_t",
                      "kp_info_pitch", "kp_info_yaw", "kp_info_roll", "kp_info_R",
                      "f_s", "M_grid", "mask", "source", "bg"):
            av[name] = torch.load(av_dir / f"{name}.pt", map_location="cpu", weights_only=True).float()
        cpu_load_s = time.monotonic() - t_load

        normalizer_data = torch.load(av_dir / "normalizer.pt", map_location="cpu", weights_only=False)
        av["norm_offset_so3"] = normalizer_data["offset_so3"].float()
        av["norm_scale_so3"] = normalizer_data["scale_so3"].float()
        av["norm_offset_kp"] = normalizer_data["offset_kp"].float()
        av["norm_scale_kp"] = normalizer_data["scale_kp"].float()
        av["norm_offset_exp"] = normalizer_data["offset_exp"].float()
        av["norm_scale_exp"] = normalizer_data["scale_exp"].float()
        motion_offset = normalizer_data["motion_offset"].float()
        motion_scale = normalizer_data["motion_scale"].float()
        av["norm_lipsync_offset"] = motion_offset[3:].contiguous()
        av["norm_lipsync_scale"] = motion_scale[3:].contiguous()

        t_gpu = time.monotonic()
        for k, v in av.items():
            if isinstance(v, torch.Tensor):
                av[k] = v.to(device, non_blocking=True)
        torch.cuda.synchronize()
        gpu_transfer_s = time.monotonic() - t_gpu

        t_precompute = time.monotonic()
        import roma
        R = av["kp_info_R"]
        so3 = roma.rotmat_to_rotvec(R)
        so3_n = (so3 - av["norm_offset_so3"]) / av["norm_scale_so3"]
        kp_n = (av["kp_info_kp"] - av["norm_offset_kp"]) / av["norm_scale_kp"]
        kp_n = kp_n.clamp(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        exp_in_R = av["kp_info_exp"] @ R.transpose(-1, -2)
        exp_n = (exp_in_R - av["norm_offset_exp"]) / av["norm_scale_exp"]
        exp_n = exp_n.clamp(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        av["kp_cond"] = torch.cat([so3_n, kp_n.view(1, -1), exp_n.view(1, -1)], dim=-1)[:, None]

        kp = av["kp_info_kp"]
        scale = av["kp_info_scale"]
        t_kp = av["kp_info_t"]
        exp = av["kp_info_exp"]
        bs_kp, num_kp, _ = kp.shape
        x_s = kp.view(bs_kp, num_kp, 3) @ R + exp.view(bs_kp, num_kp, 3)
        x_s = x_s * scale[..., None]
        x_s[:, :, 0:2] = x_s[:, :, 0:2] + t_kp[:, None, 0:2]
        av["x_s"] = x_s

        f_s = av["f_s"]
        if f_s.dtype != torch.float16:
            av["f_s"] = f_s.half()
        if av["source"].dtype != torch.float16:
            av["source"] = av["source"].half()
        if av["bg"].dtype != torch.float16:
            av["bg"] = av["bg"].half()
        if av["mask"].dtype != torch.float16:
            av["mask"] = av["mask"].half()
        if av["M_grid"].dtype != torch.float16:
            av["M_grid"] = av["M_grid"].half()

        precompute_s = time.monotonic() - t_precompute

        total_s = time.monotonic() - t0

        if len(self._avatar_cache) >= MAX_CACHE_ENTRIES:
            evict_id, evict_data = self._avatar_cache.popitem(last=False)
            self._cache_evictions += 1
            print(f"[cache] EVICT: {evict_id}")
            for v in evict_data.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    del v
            del evict_data
            torch.cuda.empty_cache()

        self._avatar_cache[avatar_id] = av

        vram_after = torch.cuda.memory_allocated() / 1024**2
        print(f"[cache] LOADED: {avatar_id} | cpu={cpu_load_s:.3f}s gpu={gpu_transfer_s:.3f}s "
              f"precompute={precompute_s:.3f}s total={total_s:.3f}s | vram={vram_after:.0f}MB | "
              f"cache_keys={list(self._avatar_cache.keys())}")

        return av

    def _make_encode_inputs(self) -> dict:
        import torch
        return {
            "past_cond": torch.randn(1, self.past_size, self.nfeats, device="cuda"),
            "audio_cond": torch.randn(1, self.past_size + self.future_size + self.chunk_size,
                                       self.cond_feature_dim * 2, device="cuda"),
            "past_times": self._past_times,
        }

    def _run_avtr_encode(self, past_cond, audio_cond, kp_cond):
        import torch
        with torch.inference_mode():
            result = self.scripted.encode_conditions(
                past_cond=past_cond, audio_cond=audio_cond,
                kp_cond=kp_cond, past_times=self._past_times,
            )
            if isinstance(result, tuple):
                from types import SimpleNamespace
                return SimpleNamespace(
                    kp_tokens=result[0],
                    past_context=result[1],
                    past_last=result[2],
                    audio_self=result[3],
                    audio_other=result[4],
                )
            return result

    def _run_avtr_decode_loop(self, enc):
        import torch
        if self._decode_graph is not None:
            with torch.inference_mode():
                self._dec_kp_tokens.copy_(enc.kp_tokens)
                self._dec_past_context.copy_(enc.past_context)
                self._dec_past_last.copy_(enc.past_last)
                self._dec_audio_self.copy_(enc.audio_self)
                self._dec_audio_other.copy_(enc.audio_other)
                self._dec_static_x.normal_()
                self._decode_graph.replay()
            return self._dec_static_x.clone()

        with torch.inference_mode():
            n_steps = self.n_ode_steps - 1
            dt = 1.0 / n_steps
            self._dec_static_x.normal_()
            x_buf = self._dec_static_x
            w_s = self._dec_w_self
            w_o = self._dec_w_other
            w_k = self._dec_w_kp
            times = self._dec_times
            for i in range(n_steps):
                v = self.scripted(
                    x=x_buf, kp=enc.kp_tokens, past_context=enc.past_context,
                    past_last=enc.past_last, audio_self=enc.audio_self,
                    audio_other=enc.audio_other,
                    self_weights=w_s, other_weights=w_o, kp_weights=w_k,
                    t=times[i],
                )
                if isinstance(v, tuple):
                    v = v[0]
                x_buf.add_(v, alpha=dt)
        return x_buf.clone()

    def _motion_to_frames(self, x, av: dict):
        import torch
        import roma
        so3_n = x[0, :, :3]
        exp_lipsync_n = x[0, :, 3:3 + N_LIPSYNC]
        so3 = so3_n * av["norm_scale_so3"] + av["norm_offset_so3"]
        exp_lipsync = exp_lipsync_n * av["norm_lipsync_scale"] + av["norm_lipsync_offset"]
        R = roma.rotvec_to_rotmat(so3)
        return R, exp_lipsync

    def _build_kp_cond(self, av: dict):
        import torch
        import roma
        R = av["kp_info_R"]
        so3 = roma.rotmat_to_rotvec(R)
        so3_n = (so3 - av["norm_offset_so3"]) / av["norm_scale_so3"]
        kp_n = (av["kp_info_kp"] - av["norm_offset_kp"]) / av["norm_scale_kp"]
        kp_n = kp_n.clamp(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        exp_in_R = av["kp_info_exp"] @ R.transpose(-1, -2)
        exp_n = (exp_in_R - av["norm_offset_exp"]) / av["norm_scale_exp"]
        exp_n = exp_n.clamp(-Z_SCORE_CLIP, Z_SCORE_CLIP)
        return torch.cat([so3_n, kp_n.view(1, -1), exp_n.view(1, -1)], dim=-1)[:, None]

    def _transform_source(self, av: dict):
        kp = av["kp_info_kp"]
        R = av["kp_info_R"]
        exp = av["kp_info_exp"]
        scale = av["kp_info_scale"]
        t = av["kp_info_t"]
        bs, num_kp, _ = kp.shape
        kp_t = kp.view(bs, num_kp, 3) @ R + exp.view(bs, num_kp, 3)
        kp_t = kp_t * scale[..., None]
        kp_t[:, :, 0:2] = kp_t[:, :, 0:2] + t[:, None, 0:2]
        return kp_t

    def _build_x_d_delta(self, R, exp, av: dict):
        import torch
        n = R.shape[0]
        delta = av["kp_info_exp"].flatten(1).expand(n, -1).clone()
        delta[:, _LIPSYNC_INDEX] = exp
        return delta.view(n, 21, 3)

    def _transform_driving(self, x_d_delta, R, av: dict):
        scale = av["kp_info_scale"]
        t = av["kp_info_t"]
        x_c = av["kp_info_kp"].view(1, 21, 3)
        x_d = scale * ((x_c + x_d_delta) @ R) + t.view(1, 1, 3)
        return x_d

    def _run_motion_stitch(self, R, exp, av: dict):
        import torch
        x_d_delta = self._build_x_d_delta(R, exp, av)
        x_d_raw = self._transform_driving(x_d_delta, R, av)
        x_s_b = av["x_s"].expand(x_d_raw.shape[0], -1, -1).contiguous().half()
        x_d_half = x_d_raw.contiguous().half()
        x_d = self.stitch_model(x_s_b, x_d_half)
        return av["x_s"], x_d

    def _run_renderer(self, R, exp, av: dict):
        import torch
        import torch.nn.functional as F

        n = R.shape[0]
        x_s, x_d = self._run_motion_stitch(R, exp, av)
        f_s_b = av["f_s"].expand(n, -1, -1, -1, -1).contiguous()
        x_s_b = x_s.expand_as(x_d).contiguous().half()
        x_d_half = x_d.contiguous() if x_d.dtype == torch.float16 else x_d.half()

        warped = self.warp_model(f_s_b, x_s_b, x_d_half)
        faces = torch.empty((n, 3, 512, 512), dtype=torch.float16, device="cuda")
        for i in range(n):
            face = self.decoder_model(warped[i:i+1])
            face.clamp_(0.0, 1.0)
            faces[i] = face[0]

        h, w = av["source"].shape[-2:]
        M_b = av["M_grid"].unsqueeze(0).expand(n, -1, -1)
        grid = F.affine_grid(M_b.float(), [n, 3, h, w], align_corners=False)
        face_warped = F.grid_sample(faces.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
        blended = torch.lerp(av["source"].unsqueeze(0).float(), face_warped, av["mask"].unsqueeze(0).float())

        downsampled = F.interpolate(blended.float(), size=(MODNET_INPUT_H, MODNET_INPUT_W), mode="bilinear")
        normalised = ((downsampled - 0.5) / 0.5).contiguous()
        matte = _modnet_forward(self.modnet, normalised)
        alpha = F.interpolate(matte.float(), size=(h, w), mode="bilinear")

        composited = torch.lerp(av["bg"].unsqueeze(0).float(), blended, alpha)
        return composited, alpha

    def _run_full_chunk(self, av: dict):
        import torch

        with torch.inference_mode():
            enc_inputs = self._make_encode_inputs()
            enc_inputs["kp_cond"] = av["kp_cond"]
            enc = self._run_avtr_encode(
                past_cond=enc_inputs["past_cond"],
                audio_cond=enc_inputs["audio_cond"],
                kp_cond=enc_inputs["kp_cond"],
            )
            x = self._run_avtr_decode_loop(enc)
            R, exp = self._motion_to_frames(x, av)
            composited, alpha = self._run_renderer(R, exp, av)
        return composited

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "snap_ok": getattr(self, "_snap_ok", False),
            "snap_error": getattr(self, "_snap_error", None),
            "snap_elapsed_s": round(getattr(self, "_snap_elapsed", 0), 2),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", 0), 4),
            "avtr_decode_path": getattr(self, "_decode_path", "unknown"),
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "models_loaded": all(hasattr(self, a) for a in [
                "scripted", "hubert_model", "stitch_model", "warp_model",
                "decoder_model", "modnet",
            ]),
            "avatar_cache_keys": list(self._avatar_cache.keys()) if hasattr(self, "_avatar_cache") else [],
            "cache_stats": {
                "hits": getattr(self, "_cache_hits", 0),
                "misses": getattr(self, "_cache_misses", 0),
                "evictions": getattr(self, "_cache_evictions", 0),
            },
        }

    @modal.method()
    def list_avatars(self) -> dict:
        t0 = time.monotonic()
        avatar_cache_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache"
        if not avatar_cache_dir.is_dir():
            return {"avatars": [], "elapsed_s": round(time.monotonic() - t0, 4)}
        avatars = sorted(d.name for d in avatar_cache_dir.iterdir() if d.is_dir())
        return {"avatars": avatars, "elapsed_s": round(time.monotonic() - t0, 4)}

    @modal.method()
    def load_avatar(self, avatar_id: str) -> dict:
        import torch

        t0 = time.monotonic()
        vram_before = torch.cuda.memory_allocated() / 1024**2
        was_hit = avatar_id in self._avatar_cache

        av = self._get_avatar(avatar_id)

        vram_after = torch.cuda.memory_allocated() / 1024**2
        total_s = time.monotonic() - t0

        return {
            "avatar_id": avatar_id,
            "was_cache_hit": was_hit,
            "load_time_s": round(total_s, 4),
            "vram_before_mb": round(vram_before, 1),
            "vram_after_mb": round(vram_after, 1),
            "vram_delta_mb": round(vram_after - vram_before, 1),
            "cache_keys": list(self._avatar_cache.keys()),
            "cache_stats": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
            },
        }

    @modal.method()
    def render_smoke(self, avatar_id: str) -> dict:
        import torch
        import torch.nn.functional as F

        t0 = time.monotonic()
        av = self._get_avatar(avatar_id)
        load_s = time.monotonic() - t0

        def _timed_render(av_inner):
            with torch.inference_mode():
                torch.cuda.synchronize()
                _st = {}

                enc_inputs = self._make_encode_inputs()
                enc_inputs["kp_cond"] = av_inner["kp_cond"]

                ev_start = torch.cuda.Event(enable_timing=True)
                ev_start.record()

                _t = torch.cuda.Event(enable_timing=True)
                enc_ev = torch.cuda.Event(enable_timing=True)
                _t.record()
                enc = self._run_avtr_encode(
                    past_cond=enc_inputs["past_cond"],
                    audio_cond=enc_inputs["audio_cond"],
                    kp_cond=enc_inputs["kp_cond"],
                )
                enc_ev.record()

                dec_ev = torch.cuda.Event(enable_timing=True)
                dec_end = torch.cuda.Event(enable_timing=True)
                dec_ev.record()
                x = self._run_avtr_decode_loop(enc)
                dec_end.record()

                mot_ev = torch.cuda.Event(enable_timing=True)
                mot_end = torch.cuda.Event(enable_timing=True)
                mot_ev.record()
                R, exp = self._motion_to_frames(x, av_inner)
                mot_end.record()

                stitch_ev = torch.cuda.Event(enable_timing=True)
                stitch_end = torch.cuda.Event(enable_timing=True)
                stitch_ev.record()
                n = R.shape[0]
                x_s, x_d = self._run_motion_stitch(R, exp, av_inner)
                stitch_end.record()

                warp_ev = torch.cuda.Event(enable_timing=True)
                warp_end = torch.cuda.Event(enable_timing=True)
                warp_ev.record()
                f_s_b = av_inner["f_s"].expand(n, -1, -1, -1, -1).contiguous()
                x_s_b = x_s.expand_as(x_d).contiguous().half()
                x_d_half = x_d.contiguous() if x_d.dtype == torch.float16 else x_d.half()
                warped = self.warp_model(f_s_b, x_s_b, x_d_half)
                warp_end.record()

                decr_ev = torch.cuda.Event(enable_timing=True)
                decr_end = torch.cuda.Event(enable_timing=True)
                decr_ev.record()
                faces = torch.empty((n, 3, 512, 512), dtype=torch.float16, device="cuda")
                for i in range(n):
                    face = self.decoder_model(warped[i:i+1])
                    face.clamp_(0.0, 1.0)
                    faces[i] = face[0]
                decr_end.record()

                putback_ev = torch.cuda.Event(enable_timing=True)
                putback_end = torch.cuda.Event(enable_timing=True)
                putback_ev.record()
                h, w = av_inner["source"].shape[-2:]
                M_b = av_inner["M_grid"].unsqueeze(0).expand(n, -1, -1)
                grid = F.affine_grid(M_b.float(), [n, 3, h, w], align_corners=False)
                face_warped = F.grid_sample(faces.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
                blended = torch.lerp(av_inner["source"].unsqueeze(0).float(), face_warped, av_inner["mask"].unsqueeze(0).float())
                putback_end.record()

                modnet_ev = torch.cuda.Event(enable_timing=True)
                modnet_end = torch.cuda.Event(enable_timing=True)
                modnet_ev.record()
                downsampled = F.interpolate(blended.float(), size=(MODNET_INPUT_H, MODNET_INPUT_W), mode="bilinear")
                normalised = ((downsampled - 0.5) / 0.5).contiguous()
                matte = _modnet_forward(self.modnet, normalised)
                alpha = F.interpolate(matte.float(), size=(h, w), mode="bilinear")
                modnet_end.record()

                comp_ev = torch.cuda.Event(enable_timing=True)
                comp_end = torch.cuda.Event(enable_timing=True)
                comp_ev.record()
                composited = torch.lerp(av_inner["bg"].unsqueeze(0).float(), blended, alpha)
                comp_end.record()

                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()

                torch.cuda.synchronize()

                _st["encode_ms"] = round(_t.elapsed_time(enc_ev), 3)
                _st["decode_ms"] = round(dec_ev.elapsed_time(dec_end), 3)
                _st["motion_ms"] = round(mot_ev.elapsed_time(mot_end), 3)
                _st["stitch_ms"] = round(stitch_ev.elapsed_time(stitch_end), 3)
                _st["warp_ms"] = round(warp_ev.elapsed_time(warp_end), 3)
                _st["decoder_ms"] = round(decr_ev.elapsed_time(decr_end), 3)
                _st["putback_ms"] = round(putback_ev.elapsed_time(putback_end), 3)
                _st["modnet_ms"] = round(modnet_ev.elapsed_time(modnet_end), 3)
                _st["composite_ms"] = round(comp_ev.elapsed_time(comp_end), 3)
                _st["total_gpu_ms"] = round(ev_start.elapsed_time(ev_end), 3)
                _st["timing_method"] = "cuda_events"

                return composited, _st

        composited, first_st = _timed_render(av)

        composited_w2, warm2_st = _timed_render(av)
        composited_w3, warm3_st = _timed_render(av)

        has_nan = bool(torch.isnan(composited_w3).any().item())
        has_inf = bool(torch.isinf(composited_w3).any().item())
        vmin = round(composited_w3.min().item(), 4)
        vmax = round(composited_w3.max().item(), 4)
        shape = list(composited_w3.shape)

        total_wall_s = time.monotonic() - t0

        return {
            "avatar_id": avatar_id,
            "total_wall_s": round(total_wall_s, 4),
            "avatar_load_s": round(load_s, 4),
            "timing_method": "cuda_events",
            "stages_fp16": ["stitch", "warp", "decoder"],
            "stages_fp32": ["putback_grid", "composite"],
            "eager_stages": ["decode"],
            "first_call": first_st,
            "warm_call_2": warm2_st,
            "warm_call_3": warm3_st,
            "output_shape": shape,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "value_range": [vmin, vmax],
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "cache_stats": {
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "evictions": self._cache_evictions,
            },
        }

    @modal.method()
    def test_cache_eviction(self) -> dict:
        import torch

        avatar_cache_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache"
        available = sorted(d.name for d in avatar_cache_dir.iterdir() if d.is_dir())
        if not available:
            return {"error": "No avatars available on volume"}

        primary_id = available[0]
        results = {}

        self._get_avatar(primary_id)
        results["load_1"] = {
            "avatar_id": primary_id,
            "cache_keys": list(self._avatar_cache.keys()),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

        self._get_avatar(primary_id)
        results["load_2_hit"] = {
            "avatar_id": primary_id,
            "cache_keys": list(self._avatar_cache.keys()),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
        }

        original_max = MAX_CACHE_ENTRIES
        saved_max = self.__class__.__dict__.get("_max_cache_test", original_max)

        for i in range(original_max + 2):
            fake_id = f"__test_fake_{i}__"
            fake_av = {}
            for k, v in self._avatar_cache[primary_id].items():
                if isinstance(v, torch.Tensor):
                    fake_av[k] = v.clone()
                else:
                    fake_av[k] = v
            if len(self._avatar_cache) >= original_max:
                evict_id, evict_data = self._avatar_cache.popitem(last=False)
                self._cache_evictions += 1
                for v in evict_data.values():
                    if isinstance(v, torch.Tensor) and v.is_cuda:
                        del v
                del evict_data
                torch.cuda.empty_cache()
            self._avatar_cache[fake_id] = fake_av
            self._avatar_cache.move_to_end(fake_id)

        results["after_eviction_cycle"] = {
            "cache_keys": list(self._avatar_cache.keys()),
            "evictions": self._cache_evictions,
            "primary_in_cache": primary_id in self._avatar_cache,
        }

        self._get_avatar(primary_id)
        results["reload_primary"] = {
            "avatar_id": primary_id,
            "cache_keys": list(self._avatar_cache.keys()),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "evictions": self._cache_evictions,
        }

        results["vram_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)

        return results
