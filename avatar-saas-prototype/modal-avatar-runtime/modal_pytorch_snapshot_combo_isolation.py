"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  COMPONENT ISOLATION PROBE — NOT PRODUCTION                                ║
║  SIGSEGV 139 on GPU Snapshot Restore — Model Combination Isolation         ║
║                                                                            ║
║  Goal: Find the smallest component combination that triggers SIGSEGV 139   ║
║  on GPU snapshot restore.                                                  ║
║                                                                            ║
║  5 combos tested:                                                          ║
║    1. AVTR only (+ decode CUDA graph)                                      ║
║    2. AVTR + HuBERT (+ decode CUDA graph)                                  ║
║    3. AVTR + renderer core (+ decode CUDA graph)                           ║
║    4. AVTR + HuBERT + renderer core (+ decode CUDA graph)                  ║
║    5. AVTR + HuBERT + renderer core + MODNet (+ decode CUDA graph)         ║
║                                                                            ║
║  Each combo is a separate Modal app/class with its own container.          ║
║  Uses EXACT loading code from proven individual probes.                    ║
║  No avatar data. No prototype-specific state. No benchmarks.               ║
║                                                                            ║
║  Deploy:                                                                    ║
║      modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_snapshot_combo_isolation.py  # noqa
║                                                                            ║
║  Invoke:                                                                    ║
║      python -c "..." (see local_entrypoint)                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import modal

HF_CACHE_PATH = "/root/.cache/huggingface"
PROBE_MOUNT_PATH = "/data/avtr1-combo-isolation"
RENDERER_REPO = "digital-avatar/ditto-talkinghead"
RENDERER_WEIGHT_PREFIX = "ditto_pytorch/models"
AVTR_REPO = "avaturn-live/avtr-1"
HUBERT_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"
AVTR_FILE = "build_artifacts/avtr1.scripted.pt"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

combo_image = (
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
probe_vol = modal.Volume.from_name("avtr1-pytorch-probe-artifacts", create_if_missing=True)

COMMON_CLS_KWARGS = dict(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=20,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=600,
    secrets=[modal.Secret.from_name("hf-secret")],
    image=combo_image,
    volumes={
        HF_CACHE_PATH: hf_cache_vol,
        PROBE_MOUNT_PATH: probe_vol,
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


# =====================================================================
# SHARED MODEL LOADING FUNCTIONS (from proven probes)
# =====================================================================

def _load_avtr(cache_dir: str):
    """Load AVTR TorchScript — from Phase 1.5 probe."""
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
    """Capture AVTR decode CUDA graph — from Phase 1.5 probe."""
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

    print("[avtr-cg] AVTR decode CUDA graph captured OK")
    return g


def _load_hubert(cache_dir: str):
    """Load HuBERT via onnx2torch — from Stage 2B probe."""
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
    """Load stitch + warp + decoder — from Stage 2C probe."""
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
    """Load MODNet — from Stage 2C probe."""
    import torch
    from transformers import AutoModel

    modnet = AutoModel.from_pretrained(
        "boopathiraj/MODNet", trust_remote_code=True, token=hf_token,
    )
    modnet.cuda().eval()

    with torch.no_grad():
        modnet(torch.randn(1, 3, 288, 512, device="cuda"), True)

    print("[modnet] Loaded")
    return modnet


# =====================================================================
# COMBO 1: AVTR only (+ decode CUDA graph)
# =====================================================================

app1 = modal.App(name="avtr1-combo-1-avtr-only")


@app1.cls(**COMMON_CLS_KWARGS)
class Combo1AVTROnly:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._combo = "combo-1-avtr-only"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._combo} ===")

        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        self.scripted, self.avtr_params = _load_avtr(cache_dir)
        self._decode_graph = _capture_avtr_decode_cg(self.scripted, self.avtr_params)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._combo}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._combo} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found after restore"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found after restore"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "combo": self._combo,
            "components": ["avtr_torchscript", "avtr_decode_cuda_graph"],
            "cuda_graphs": ["avtr_decode"],
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_avtr": hasattr(self, "scripted"),
            "has_cg": hasattr(self, "_decode_graph") and self._decode_graph is not None,
        }


# =====================================================================
# COMBO 2: AVTR + HuBERT (+ decode CUDA graph)
# =====================================================================

app2 = modal.App(name="avtr1-combo-2-avtr-hubert")


@app2.cls(**COMMON_CLS_KWARGS)
class Combo2AVTRHuBERT:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._combo = "combo-2-avtr-hubert"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._combo} ===")

        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        self.scripted, self.avtr_params = _load_avtr(cache_dir)
        self._decode_graph = _capture_avtr_decode_cg(self.scripted, self.avtr_params)
        self.hubert_model = _load_hubert(cache_dir)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._combo}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._combo} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "combo": self._combo,
            "components": ["avtr_torchscript", "avtr_decode_cuda_graph", "hubert_onnx2torch"],
            "cuda_graphs": ["avtr_decode"],
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_avtr": hasattr(self, "scripted"),
            "has_hubert": hasattr(self, "hubert_model"),
            "has_cg": hasattr(self, "_decode_graph") and self._decode_graph is not None,
        }


# =====================================================================
# COMBO 3: AVTR + renderer core (+ decode CUDA graph)
# =====================================================================

app3 = modal.App(name="avtr1-combo-3-avtr-renderer")


@app3.cls(**COMMON_CLS_KWARGS)
class Combo3AVTRRenderer:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._combo = "combo-3-avtr-renderer"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._combo} ===")

        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        self.scripted, self.avtr_params = _load_avtr(cache_dir)
        self._decode_graph = _capture_avtr_decode_cg(self.scripted, self.avtr_params)
        self.stitch_model, self.warp_model, self.decoder_model = _load_renderer(cache_dir, hf_token)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._combo}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._combo} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "stitch_model"), "Stitch NOT found"
        assert hasattr(self, "warp_model"), "Warp NOT found"
        assert hasattr(self, "decoder_model"), "Decoder NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "combo": self._combo,
            "components": ["avtr_torchscript", "avtr_decode_cuda_graph",
                          "stitch_nn", "warp_nn", "decoder_nn"],
            "cuda_graphs": ["avtr_decode"],
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_avtr": hasattr(self, "scripted"),
            "has_renderer": all(hasattr(self, a) for a in ["stitch_model", "warp_model", "decoder_model"]),
            "has_cg": hasattr(self, "_decode_graph") and self._decode_graph is not None,
        }


# =====================================================================
# COMBO 4: AVTR + HuBERT + renderer core (+ decode CUDA graph)
# =====================================================================

app4 = modal.App(name="avtr1-combo-4-avtr-hubert-renderer")


@app4.cls(**COMMON_CLS_KWARGS)
class Combo4AVTRHuBERTRenderer:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._combo = "combo-4-avtr-hubert-renderer"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._combo} ===")

        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        self.scripted, self.avtr_params = _load_avtr(cache_dir)
        self._decode_graph = _capture_avtr_decode_cg(self.scripted, self.avtr_params)
        self.hubert_model = _load_hubert(cache_dir)
        self.stitch_model, self.warp_model, self.decoder_model = _load_renderer(cache_dir, hf_token)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._combo}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._combo} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found"
        assert hasattr(self, "stitch_model"), "Stitch NOT found"
        assert hasattr(self, "warp_model"), "Warp NOT found"
        assert hasattr(self, "decoder_model"), "Decoder NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "combo": self._combo,
            "components": ["avtr_torchscript", "avtr_decode_cuda_graph",
                          "hubert_onnx2torch", "stitch_nn", "warp_nn", "decoder_nn"],
            "cuda_graphs": ["avtr_decode"],
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_avtr": hasattr(self, "scripted"),
            "has_hubert": hasattr(self, "hubert_model"),
            "has_renderer": all(hasattr(self, a) for a in ["stitch_model", "warp_model", "decoder_model"]),
            "has_cg": hasattr(self, "_decode_graph") and self._decode_graph is not None,
        }


# =====================================================================
# COMBO 5: Full (AVTR + HuBERT + renderer + MODNet) (+ decode CUDA graph)
# =====================================================================

app5 = modal.App(name="avtr1-combo-5-full")


@app5.cls(**COMMON_CLS_KWARGS)
class Combo5Full:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch

        self._combo = "combo-5-full"
        self._snap_t0 = time.monotonic()
        print(f"[snap-init] === {self._combo} ===")

        hf_token = os.environ.get("HF_TOKEN")
        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")
        self.scripted, self.avtr_params = _load_avtr(cache_dir)
        self._decode_graph = _capture_avtr_decode_cg(self.scripted, self.avtr_params)
        self.hubert_model = _load_hubert(cache_dir)
        self.stitch_model, self.warp_model, self.decoder_model = _load_renderer(cache_dir, hf_token)
        self.modnet = _load_modnet(hf_token)

        vram = torch.cuda.memory_allocated() / 1024**2
        self._snap_elapsed = time.monotonic() - self._snap_t0
        self._snap_vram = vram
        _run_nvidia_smi("snap-init-end")
        print(f"[snap-init] {self._combo}: VRAM={vram:.0f}MB, done in {self._snap_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch

        self._restore_t0 = time.monotonic()
        print(f"[restore] === {self._combo} ===")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM={vram:.0f}MB")
        assert hasattr(self, "scripted"), "AVTR NOT found"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found"
        assert hasattr(self, "stitch_model"), "Stitch NOT found"
        assert hasattr(self, "warp_model"), "Warp NOT found"
        assert hasattr(self, "decoder_model"), "Decoder NOT found"
        assert hasattr(self, "modnet"), "MODNet NOT found"
        assert hasattr(self, "_decode_graph"), "CUDA graph NOT found"
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    @modal.method()
    def health_check(self) -> dict:
        import torch
        return {
            "combo": self._combo,
            "components": ["avtr_torchscript", "avtr_decode_cuda_graph",
                          "hubert_onnx2torch", "stitch_nn", "warp_nn", "decoder_nn",
                          "modnet_transformers"],
            "cuda_graphs": ["avtr_decode"],
            "snap_elapsed_s": round(self._snap_elapsed, 2),
            "snap_vram_mb": round(self._snap_vram, 1),
            "restore_elapsed_s": round(getattr(self, "_restore_elapsed", -1), 4),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "has_avtr": hasattr(self, "scripted"),
            "has_hubert": hasattr(self, "hubert_model"),
            "has_renderer": all(hasattr(self, a) for a in ["stitch_model", "warp_model", "decoder_model"]),
            "has_modnet": hasattr(self, "modnet"),
            "has_cg": hasattr(self, "_decode_graph") and self._decode_graph is not None,
        }


# =====================================================================
# MASTER APP — combines all sub-apps for single `modal deploy`
# =====================================================================

app = modal.App(name="avtr1-snapshot-combo-isolation").include(app1).include(app2).include(app3).include(app4).include(app5)
