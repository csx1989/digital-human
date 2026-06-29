"""
Avatar-State Isolation Probe — GPU Snapshot SIGSEGV Root Cause

Goal: Find the smallest avatar/state addition that triggers SIGSEGV 139
on GPU snapshot restore.

Baseline: Combo 5 (AVTR + HuBERT + renderer + MODNet + AVTR decode CUDA graph)
          proven to restore cleanly.

Steps:
  0  — Standalone baseline (no App.include, same models as Combo 5)
  1  — Mount L40S volume, don't read files
  2F — All avatar tensors loaded to CPU only
  3A — Normalizer tensors on CUDA
  3B — M_grid on CUDA
  3C — mask on CUDA
  3D — source on CUDA
  3E — bg on CUDA
  3F — kp_info tensors on CUDA
  3G — f_s on CUDA
  3H — All avatar tensors on CUDA
  4A — Step 3H + _past_times
  4B — Step 3H + _bench_audio
  4C — Step 3H + decode CG static buffers as self attrs
  4D — Step 3H + ALL extra prototype state

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_snapshot_avatar_state_isolation.py

Invoke:
    python -c "
import modal, json
cls = modal.Cls.from_name('avtr1-avatar-state-isolation', 'Step0StandaloneBaseline')
inst = cls()
print(json.dumps(inst.health_check.remote(), indent=2))
    "
"""

from __future__ import annotations

import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import modal

HF_CACHE_PATH = "/root/.cache/huggingface"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
PROBE_MOUNT_PATH = "/data/avtr1-combo-isolation"
RENDERER_REPO = "digital-avatar/ditto-talkinghead"
RENDERER_WEIGHT_PREFIX = "ditto_pytorch/models"
AVTR_REPO = "avaturn-live/avtr-1"
HUBERT_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"
AVTR_FILE = "build_artifacts/avtr1.scripted.pt"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

prototype_image = (
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
        "torchvision>=0.23,<0.24",
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
l40s_volume = modal.Volume.from_name("avtr1-artifacts-l40s", create_if_missing=True)
probe_vol = modal.Volume.from_name("avtr1-pytorch-probe-artifacts", create_if_missing=True)

STEP0_CLS_KWARGS = dict(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=20,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
    image=prototype_image,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        PROBE_MOUNT_PATH: probe_vol,
    },
)

MASTER_CLS_KWARGS = dict(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=20,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
    image=prototype_image,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        L40S_MOUNT_PATH: l40s_volume,
    },
)


def _run_nvidia_smi(label: str) -> dict:
    info = {"label": label}
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.returncode == 0:
            info["smi_output"] = r.stdout.strip()
            print(f"[nvidia-smi] {label}: {r.stdout.strip()}")
    except Exception:
        pass
    return info


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
                weight_init = next(
                    (i for i in onnx_model.graph.initializer if i.name == node.input[1]),
                    None,
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


def _remove_ddp_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


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
            dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
            return x_s + dx

        def actvn(self, x):
            return F.leaky_relu(x, 2e-1)

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


def _load_avtr(cache_dir: str):
    import torch
    from huggingface_hub import hf_hub_download

    avtr_path = hf_hub_download(
        repo_id=AVTR_REPO, filename=AVTR_FILE,
        repo_type="model", cache_dir=cache_dir,
    )
    scripted = torch.jit.load(str(avtr_path), map_location="cuda").eval()
    inner = scripted.model
    chunk_size = int(inner.chunk_size)
    past_size = int(inner.past_size)
    future_size = int(inner.future_size)
    nfeats = int(inner.nfeats)
    cond_feature_dim = int(inner.cond_feature_dim)
    latent_dim = int(inner.latent_dim)
    n_ode_steps = 5

    with torch.no_grad():
        past_cond = torch.randn(1, past_size, nfeats, device="cuda")
        audio_cond = torch.randn(1, past_size + future_size + chunk_size, cond_feature_dim * 2, device="cuda")
        kp_cond = torch.randn(1, 1, 3 + 63 + 63, device="cuda")
        past_times = torch.ones(1, past_size // chunk_size, 1, device="cuda")
        scripted.encode_conditions(
            past_cond=past_cond, audio_cond=audio_cond,
            kp_cond=kp_cond, past_times=past_times,
        )

    print(f"[avtr] Loaded: chunk={chunk_size} past={past_size} nfeats={nfeats} latent={latent_dim}")
    return scripted, {
        "chunk_size": chunk_size, "past_size": past_size,
        "future_size": future_size, "nfeats": nfeats,
        "cond_feature_dim": cond_feature_dim, "latent_dim": latent_dim,
        "n_ode_steps": n_ode_steps,
    }


def _capture_avtr_decode_cg(scripted, params):
    import torch

    chunk_size = params["chunk_size"]
    nfeats = params["nfeats"]
    latent_dim = params["latent_dim"]
    n_ode_steps = params["n_ode_steps"]
    n_steps = n_ode_steps - 1
    dt = 1.0 / n_steps

    past_cond = torch.randn(1, params["past_size"], nfeats, device="cuda")
    audio_cond = torch.randn(1, params["past_size"] + params["future_size"] + chunk_size,
                              params["cond_feature_dim"] * 2, device="cuda")
    kp_cond = torch.randn(1, 1, 3 + 63 + 63, device="cuda")
    past_times = torch.ones(1, params["past_size"] // chunk_size, 1, device="cuda")

    with torch.no_grad():
        enc = scripted.encode_conditions(
            past_cond=past_cond, audio_cond=audio_cond,
            kp_cond=kp_cond, past_times=past_times,
        )
    if isinstance(enc, tuple):
        kp_tokens, past_context, past_last, audio_self, audio_other = enc
    else:
        kp_tokens = enc.kp_tokens
        past_context = enc.past_context
        past_last = enc.past_last
        audio_self = enc.audio_self
        audio_other = enc.audio_other

    static_x = torch.randn(1, chunk_size, nfeats, device="cuda")
    times_t = torch.linspace(0.0, 1.0, n_ode_steps, device="cuda").view(-1, 1, 1)
    w_self = torch.full((latent_dim,), 2.0, device="cuda")
    w_other = torch.full((latent_dim,), 2.0, device="cuda")
    w_kp = torch.full((latent_dim,), 3.0, device="cuda")

    def _captured_loop():
        for i in range(n_steps):
            v = scripted(
                x=static_x, kp=kp_tokens, past_context=past_context,
                past_last=past_last, audio_self=audio_self,
                audio_other=audio_other, t=times_t[i],
                self_weights=w_self, other_weights=w_other, kp_weights=w_kp,
            )
            if isinstance(v, tuple):
                v = v[0]
            static_x.add_(v, alpha=dt)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(10):
            with torch.inference_mode():
                static_x.normal_()
                _captured_loop()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    static_x.normal_()
    with torch.inference_mode(), torch.cuda.graph(g):
        _captured_loop()

    print("[avtr-cg] AVTR decode CUDA graph captured OK (static buffers LOCAL)")
    return g


def _capture_avtr_decode_cg_on_obj(obj, scripted, params):
    import torch

    chunk_size = params["chunk_size"]
    nfeats = params["nfeats"]
    latent_dim = params["latent_dim"]
    n_ode_steps = params["n_ode_steps"]
    n_steps = n_ode_steps - 1
    dt = 1.0 / n_steps

    past_cond = torch.randn(1, params["past_size"], nfeats, device="cuda")
    audio_cond = torch.randn(1, params["past_size"] + params["future_size"] + chunk_size,
                              params["cond_feature_dim"] * 2, device="cuda")
    kp_cond = torch.randn(1, 1, 3 + 63 + 63, device="cuda")
    past_times = torch.ones(1, params["past_size"] // chunk_size, 1, device="cuda")

    with torch.no_grad():
        enc = scripted.encode_conditions(
            past_cond=past_cond, audio_cond=audio_cond,
            kp_cond=kp_cond, past_times=past_times,
        )
    if isinstance(enc, tuple):
        kp_tokens, past_context, past_last, audio_self, audio_other = enc
    else:
        kp_tokens = enc.kp_tokens
        past_context = enc.past_context
        past_last = enc.past_last
        audio_self = enc.audio_self
        audio_other = enc.audio_other

    obj._dec_static_x = torch.randn(1, chunk_size, nfeats, device="cuda")
    obj._dec_kp_tokens = kp_tokens.clone()
    obj._dec_past_context = past_context.clone()
    obj._dec_past_last = past_last.clone()
    obj._dec_audio_self = audio_self.clone()
    obj._dec_audio_other = audio_other.clone()
    obj._dec_w_self = torch.full((latent_dim,), 1.0, device="cuda")
    obj._dec_w_other = torch.full((latent_dim,), 1.0, device="cuda")
    obj._dec_w_kp = torch.full((latent_dim,), 1.0, device="cuda")
    obj._dec_times = torch.linspace(0, 1, n_ode_steps, device="cuda").view(-1, 1, 1)

    def _captured_loop_self():
        for si in range(n_steps):
            v = scripted(
                x=obj._dec_static_x,
                kp=obj._dec_kp_tokens,
                past_context=obj._dec_past_context,
                past_last=obj._dec_past_last,
                audio_self=obj._dec_audio_self,
                audio_other=obj._dec_audio_other,
                self_weights=obj._dec_w_self,
                other_weights=obj._dec_w_other,
                kp_weights=obj._dec_w_kp,
                t=obj._dec_times[si],
            )
            if isinstance(v, tuple):
                v = v[0]
            obj._dec_static_x.add_(v, alpha=dt)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(10):
            with torch.inference_mode():
                obj._dec_static_x.normal_()
                _captured_loop_self()
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    obj._dec_static_x.normal_()
    with torch.inference_mode(), torch.cuda.graph(g):
        _captured_loop_self()

    print("[avtr-cg-self] AVTR decode CUDA graph captured OK (static buffers on obj)")
    return g


def _load_hubert(cache_dir: str):
    import torch
    from huggingface_hub import hf_hub_download
    from onnx2torch import convert

    onnx_path = hf_hub_download(
        repo_id=AVTR_REPO, filename=HUBERT_FILE,
        cache_dir=cache_dir,
    )
    fixed_path = _fix_onnx_conv_biases(onnx_path)
    model = convert(fixed_path).cuda().eval()

    with torch.no_grad():
        model(torch.randn(2, 8400, device="cuda"))

    print("[hubert] Loaded via onnx2torch")
    return model


def _load_renderer(cache_dir: str, hf_token: str):
    import torch
    from huggingface_hub import hf_hub_download

    classes = _build_renderer_model_classes()

    stitch_path = hf_hub_download(
        repo_id=RENDERER_REPO,
        filename=f"{RENDERER_WEIGHT_PREFIX}/stitch_network.pth",
        repo_type="model", cache_dir=cache_dir,
    )
    stitch_ckpt = torch.load(stitch_path, map_location="cpu", weights_only=False)
    stitch_sd = _remove_ddp_prefix(stitch_ckpt["retarget_shoulder"])
    stitch_model = classes["StitchingNetwork"]()
    stitch_model.load_state_dict(stitch_sd)
    stitch_model.cuda().half().eval()

    warp_sd = torch.load(
        hf_hub_download(
            repo_id=RENDERER_REPO,
            filename=f"{RENDERER_WEIGHT_PREFIX}/warp_network.pth",
            repo_type="model", cache_dir=cache_dir,
        ),
        map_location="cpu", weights_only=False,
    )
    warp_model = classes["WarpingNetwork"]()
    warp_model.load_state_dict(warp_sd)
    warp_model.cuda().half().eval()

    decoder_sd = torch.load(
        hf_hub_download(
            repo_id=RENDERER_REPO,
            filename=f"{RENDERER_WEIGHT_PREFIX}/decoder.pth",
            repo_type="model", cache_dir=cache_dir,
        ),
        map_location="cpu", weights_only=False,
    )
    decoder_sd_float = {k: v.float() for k, v in decoder_sd.items()}
    decoder_model = classes["SPADEDecoder"]()
    decoder_model.load_state_dict(decoder_sd_float)
    decoder_model.cuda().half().eval()

    with torch.no_grad():
        stitch_model(
            torch.randn(5, 21, 3, device="cuda", dtype=torch.float16),
            torch.randn(5, 21, 3, device="cuda", dtype=torch.float16),
        )
        warp_model(
            torch.randn(2, 32, 16, 64, 64, device="cuda", dtype=torch.float16),
            torch.randn(2, 21, 3, device="cuda", dtype=torch.float16),
            torch.randn(2, 21, 3, device="cuda", dtype=torch.float16),
        )
        decoder_model(torch.randn(1, 256, 64, 64, device="cuda", dtype=torch.float16))

    print("[renderer] stitch + warp + decoder loaded (fp16)")
    return stitch_model, warp_model, decoder_model


def _load_modnet(hf_token: str):
    import torch
    from transformers import AutoModel

    modnet = AutoModel.from_pretrained(
        "boopathiraj/MODNet", trust_remote_code=True, token=hf_token,
    )
    modnet.cuda().eval()

    with torch.no_grad():
        _, _, matte = modnet(torch.randn(1, 3, 288, 512, device="cuda"), True)

    print("[modnet] Loaded")
    return modnet


def _load_all_models(cache_dir: str, hf_token: str):
    scripted, avtr_params = _load_avtr(cache_dir)
    decode_graph = _capture_avtr_decode_cg(scripted, avtr_params)
    hubert_model = _load_hubert(cache_dir)
    stitch_model, warp_model, decoder_model = _load_renderer(cache_dir, hf_token)
    modnet = _load_modnet(hf_token)
    return {
        "scripted": scripted,
        "avtr_params": avtr_params,
        "decode_graph": decode_graph,
        "hubert_model": hubert_model,
        "stitch_model": stitch_model,
        "warp_model": warp_model,
        "decoder_model": decoder_model,
        "modnet": modnet,
    }


AVATAR_TENSOR_FILES = [
    "kp_info_kp", "kp_info_exp", "kp_info_scale", "kp_info_t",
    "kp_info_pitch", "kp_info_yaw", "kp_info_roll", "kp_info_R",
    "f_s", "M_grid", "mask", "source", "bg",
]

AVATAR_TENSOR_GROUPS = {
    "normalizer": ["normalizer"],
    "M_grid": ["M_grid"],
    "mask": ["mask"],
    "source": ["source"],
    "bg": ["bg"],
    "kp_info": ["kp_info_kp", "kp_info_exp", "kp_info_scale", "kp_info_t",
                "kp_info_pitch", "kp_info_yaw", "kp_info_roll", "kp_info_R"],
    "f_s": ["f_s"],
}

AVATAR_TENSOR_SHAPES = {
    "kp_info_kp": (1, 21, 3),
    "kp_info_exp": (1, 21, 3),
    "kp_info_scale": (1, 1),
    "kp_info_t": (1, 3),
    "kp_info_pitch": (1, 1),
    "kp_info_yaw": (1, 1),
    "kp_info_roll": (1, 1),
    "kp_info_R": (1, 3, 3),
    "f_s": (1, 32, 16, 64, 64),
    "M_grid": (2, 3),
    "mask": (1, 720, 1280),
    "source": (3, 720, 1280),
    "bg": (1, 3, 720, 1280),
}


def _load_avatar_tensors(device: str, groups: list[str] | None = None):
    import torch

    av_dir = Path(L40S_MOUNT_PATH) / "avatars_pytorch_cache"
    avatar_dirs = sorted(d.name for d in av_dir.iterdir() if d.is_dir()) if av_dir.is_dir() else []
    if not avatar_dirs:
        raise RuntimeError(f"No cached avatars found at {av_dir}")
    avatar_id = avatar_dirs[0]
    av_path = av_dir / avatar_id
    print(f"[avatar] Loading from {av_path}, device={device}, groups={groups}")

    tensors = {}

    if groups is None:
        files_to_load = AVATAR_TENSOR_FILES + ["normalizer"]
    else:
        files_to_load = []
        for g in groups:
            files_to_load.extend(AVATAR_TENSOR_GROUPS.get(g, [g]))

    for name in files_to_load:
        fpath = av_path / f"{name}.pt"
        if not fpath.exists():
            print(f"[avatar] SKIP {name} (not found)")
            continue
        data = torch.load(fpath, map_location=device, weights_only=(name != "normalizer"))
        if name == "normalizer":
            for k, v in data.items():
                t = v.float()
                if k in ("motion_offset", "motion_scale"):
                    t = t[3:].contiguous()
                    k = f"norm_lipsync_{k.split('_')[1]}"
                tensors[f"_norm_{k}"] = t.to(device) if device != "cpu" else t
                print(f"[avatar] normalizer: _norm_{k} shape={list(t.shape)} "
                      f"size={t.numel() * t.element_size() / 1024:.1f}KB device={t.device}")
        else:
            t = data.float()
            if device != "cpu":
                t = t.to(device)
            tensors[f"_{name}"] = t
            print(f"[avatar] _{name} shape={list(t.shape)} dtype={t.dtype} "
                  f"size={t.numel() * t.element_size() / 1024 / 1024:.2f}MB device={t.device}")

    return tensors, avatar_id


def _set_attrs(obj, attrs: dict):
    for k, v in attrs.items():
        setattr(obj, k, v)


def _make_health_dict(step: str, extra: dict | None = None) -> dict:
    import torch
    result = {
        "step": step,
        "snap_elapsed_s": round(getattr(_make_health_dict, "_snap_elapsed", -1), 2),
        "snap_vram_mb": round(getattr(_make_health_dict, "_snap_vram", -1), 1),
        "restore_elapsed_s": round(getattr(_make_health_dict, "_restore_elapsed", -1), 4),
        "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
    }
    if extra:
        result.update(extra)
    return result


# =====================================================================
# STEP 0 — Standalone baseline (no App.include)
# =====================================================================

step0_app = modal.App("avtr1-avatar-state-step0-standalone")


@step0_app.cls(**STEP0_CLS_KWARGS)
class Step0StandaloneBaseline:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step0-standalone-baseline"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "decode_graph"), "CUDA graph NOT found"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found"
        assert hasattr(self, "modnet"), "MODNet NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_all_models": all(hasattr(self, a) for a in
                                  ["scripted", "decode_graph", "hubert_model",
                                   "stitch_model", "warp_model", "decoder_model", "modnet"]),
        }


# =====================================================================
# STEP 1 — Mount L40S volume, don't read files
# =====================================================================

step1_app = modal.App("avtr1-avatar-state-step1")


@step1_app.cls(**MASTER_CLS_KWARGS)
class Step1VolumeMount:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step1-volume-mount"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        vol_path = Path(L40S_MOUNT_PATH)
        vol_exists = vol_path.exists()
        print(f"[step1] Volume mount at {L40S_MOUNT_PATH}: exists={vol_exists}")
        if vol_exists:
            entries = sorted(str(p) for p in vol_path.iterdir())
            print(f"[step1] Volume entries: {entries[:20]}")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


# =====================================================================
# STEP 2F — All avatar tensors on CPU
# =====================================================================

step2f_app = modal.App("avtr1-avatar-state-step2f")


@step2f_app.cls(**MASTER_CLS_KWARGS)
class Step2FAllCPU:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step2f-all-cpu"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cpu")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False
        print(f"[step2f] Loaded {len(tensors)} CPU tensors for avatar {avatar_id}")

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === step2f-all-cpu ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_f_s"), "f_s CPU tensor NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "avatar_id": getattr(self, "_avatar_id", None),
        }


# =====================================================================
# STEP 3 — Independent CUDA groups (explicit classes)
# =====================================================================

step3a_app = modal.App("avtr1-avatar-state-step3a")


@step3a_app.cls(**MASTER_CLS_KWARGS)
class Step3ANormalizerCUDA:
    _groups = ["normalizer"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3a-normalizer-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3b_app = modal.App("avtr1-avatar-state-step3b")


@step3b_app.cls(**MASTER_CLS_KWARGS)
class Step3BMGridCUDA:
    _groups = ["M_grid"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3b-M_grid-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3c_app = modal.App("avtr1-avatar-state-step3c")


@step3c_app.cls(**MASTER_CLS_KWARGS)
class Step3CMaskCUDA:
    _groups = ["mask"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3c-mask-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3d_app = modal.App("avtr1-avatar-state-step3d")


@step3d_app.cls(**MASTER_CLS_KWARGS)
class Step3DSourceCUDA:
    _groups = ["source"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3d-source-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3e_app = modal.App("avtr1-avatar-state-step3e")


@step3e_app.cls(**MASTER_CLS_KWARGS)
class Step3EBgCUDA:
    _groups = ["bg"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3e-bg-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3f_app = modal.App("avtr1-avatar-state-step3f")


@step3f_app.cls(**MASTER_CLS_KWARGS)
class Step3FKpInfoCUDA:
    _groups = ["kp_info"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3f-kp_info-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


step3g_app = modal.App("avtr1-avatar-state-step3g")


@step3g_app.cls(**MASTER_CLS_KWARGS)
class Step3GFsCUDA:
    _groups = ["f_s"]

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3g-f_s-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda", groups=self._groups)
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


# =====================================================================
# STEP 3H — All avatar tensors on CUDA
# =====================================================================

step3h_app = modal.App("avtr1-avatar-state-step3h")


@step3h_app.cls(**MASTER_CLS_KWARGS)
class Step3HAllCUDA:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step3h-all-cuda"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False

        total_bytes = sum(t.numel() * t.element_size()
                          for t in tensors.values()
                          if isinstance(t, torch.Tensor))
        print(f"[step3h] {len(tensors)} CUDA tensors for avatar {avatar_id}, "
              f"{total_bytes / 1024 / 1024:.2f}MB total GPU avatar data")

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_f_s"), "f_s CUDA tensor NOT found"
        assert hasattr(self, "_source"), "source CUDA tensor NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "avatar_id": getattr(self, "_avatar_id", None),
        }


# =====================================================================
# STEP 4A — All avatar CUDA + _past_times
# =====================================================================

step4a_app = modal.App("avtr1-avatar-state-step4a")


@step4a_app.cls(**MASTER_CLS_KWARGS)
class Step4APastTimes:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step4a-past-times"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False

        self._past_times = torch.ones(
            (1, self.avtr_params["past_size"] // self.avtr_params["chunk_size"], 1),
            dtype=torch.float32, device="cuda",
        )
        print(f"[step4a] _past_times shape={list(self._past_times.shape)}")

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_past_times"), "_past_times NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


# =====================================================================
# STEP 4B — All avatar CUDA + _bench_audio
# =====================================================================

step4b_app = modal.App("avtr1-avatar-state-step4b")


@step4b_app.cls(**MASTER_CLS_KWARGS)
class Step4BBenchAudio:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step4b-bench-audio"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")
        models = _load_all_models(cache_dir, hf_token)
        for k, v in models.items():
            setattr(self, k, v)

        tensors, avatar_id = _load_avatar_tensors("cuda")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False
        self._past_times = torch.ones(
            (1, self.avtr_params["past_size"] // self.avtr_params["chunk_size"], 1),
            dtype=torch.float32, device="cuda",
        )

        self._bench_audio = torch.randn(2, 8400, device="cuda", dtype=torch.float32)
        print(f"[step4b] _bench_audio shape={list(self._bench_audio.shape)}")

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_bench_audio"), "_bench_audio NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        }


# =====================================================================
# STEP 4C — All avatar CUDA + decode CG static buffers on self
# This is the MOST SUSPICIOUS candidate:
#   Combo 5 captures CG with LOCAL vars → PASS
#   Prototype stores CG buffers as SELF attrs → SIGSEGV
# =====================================================================

step4c_app = modal.App("avtr1-avatar-state-step4c")


@step4c_app.cls(**MASTER_CLS_KWARGS)
class Step4CCGBuffersOnSelf:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step4c-cg-buffers-on-self"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._step} ===")

        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")

        scripted, avtr_params = _load_avtr(cache_dir)
        self.scripted = scripted
        self.avtr_params = avtr_params

        self.hubert_model = _load_hubert(cache_dir)
        self.stitch_model, self.warp_model, self.decoder_model = _load_renderer(cache_dir, hf_token)
        self.modnet = _load_modnet(hf_token)

        tensors, avatar_id = _load_avatar_tensors("cuda")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False
        self._past_times = torch.ones(
            (1, avtr_params["past_size"] // avtr_params["chunk_size"], 1),
            dtype=torch.float32, device="cuda",
        )

        print("[step4c] Capturing AVTR decode CUDA graph with static buffers on SELF...")
        self._decode_graph = _capture_avtr_decode_cg_on_obj(self, scripted, avtr_params)
        self._decode_path = "cuda_graph"

        print(f"[step4c] Self attrs with decode buffers:")
        for attr in ("_dec_static_x", "_dec_kp_tokens", "_dec_past_context",
                      "_dec_past_last", "_dec_audio_self", "_dec_audio_other",
                      "_dec_w_self", "_dec_w_other", "_dec_w_kp", "_dec_times"):
            t = getattr(self, attr, None)
            if t is not None:
                print(f"  {attr}: shape={list(t.shape)} dtype={t.dtype} "
                      f"device={t.device} size={t.numel() * t.element_size() / 1024:.1f}KB")

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        assert hasattr(self, "_dec_static_x"), "CG buffer _dec_static_x NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "cg_buffers_on_self": all(hasattr(self, a) for a in
                ["_dec_static_x", "_dec_kp_tokens", "_dec_past_context",
                 "_dec_past_last", "_dec_audio_self", "_dec_audio_other",
                 "_dec_w_self", "_dec_w_other", "_dec_w_kp", "_dec_times"]),
        }


# =====================================================================
# STEP 4D — All extra prototype state (matches prototype exactly)
# =====================================================================

step4d_app = modal.App("avtr1-avatar-state-step4d")


@step4d_app.cls(**MASTER_CLS_KWARGS)
class Step4DAllExtraState:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._step = "step4d-all-extra-state"
        self._snap_t0 = time.monotonic()
        self._snap_ok = False
        self._snap_error = None
        print(f"[snap-init] === {self._step} ===")

        device = torch.device("cuda")
        cache_dir = str(Path(HF_CACHE_PATH) / "hub")
        hf_token = os.environ.get("HF_TOKEN")

        scripted, avtr_params = _load_avtr(cache_dir)
        self.scripted = scripted
        self.avtr_params = avtr_params
        self.chunk_size = avtr_params["chunk_size"]
        self.past_size = avtr_params["past_size"]
        self.future_size = avtr_params["future_size"]
        self.nfeats = avtr_params["nfeats"]
        self.cond_feature_dim = avtr_params["cond_feature_dim"]
        self.latent_dim = avtr_params["latent_dim"]
        self.n_ode_steps = avtr_params["n_ode_steps"]

        self.hubert_model = _load_hubert(cache_dir)
        self.stitch_model, self.warp_model, self.decoder_model = _load_renderer(cache_dir, hf_token)
        self.modnet = _load_modnet(hf_token)

        tensors, avatar_id = _load_avatar_tensors("cuda")
        _set_attrs(self, tensors)
        self._avatar_id = avatar_id
        self._no_matting = False

        self._past_times = torch.ones(
            (1, self.past_size // self.chunk_size, 1),
            dtype=torch.float32, device="cuda",
        )

        self._decode_graph = _capture_avtr_decode_cg_on_obj(self, scripted, avtr_params)
        self._decode_path = "cuda_graph"

        self._bench_audio = torch.randn(2, 8400, device="cuda", dtype=torch.float32)
        self._snap_ok = True

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._step}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._step} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "_f_s"), "Avatar data NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        assert hasattr(self, "_dec_static_x"), "CG buffer NOT found"
        assert hasattr(self, "_bench_audio"), "bench_audio NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "step": self._step,
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "snap_ok": getattr(self, "_snap_ok", False),
        }


# =====================================================================
# MASTER APP — Steps 1-4D via App.include
# =====================================================================

app = (
    modal.App("avtr1-avatar-state-isolation")
    .include(step0_app)
    .include(step1_app).include(step2f_app)
    .include(step3a_app).include(step3b_app).include(step3c_app)
    .include(step3d_app).include(step3e_app).include(step3f_app).include(step3g_app)
    .include(step3h_app)
    .include(step4a_app).include(step4b_app).include(step4c_app).include(step4d_app)
)
