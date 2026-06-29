"""
ISOLATED OPTIMIZATION PROBE — NOT PRODUCTION
Stage 2C: Full Pure-PyTorch Budget Rescue on L40S

Goal: Recover >=20ms from renderer/MODNet precision optimization
      to bring total pipeline under 200ms (hard) / 180ms (production-safe).

Budget targets:
  - Hard realtime budget: <=200ms per 5-frame chunk
  - Production-safe target: <=180ms per 5-frame chunk

Current budget (all fp32):
  - AVTR encode+decode CUDA graph: 27ms
  - Renderer stitch+warp+decoder: 122-128ms
  - MODNet CUDA graph B=1x5 fp32: 10.6ms
  - HuBERT onnx2torch fp32 eager: 52.1ms
  - Total: ~214.7ms — NO-GO by 14.7ms

Stages:
  1. Renderer precision: fp32/fp16/autocast/bf16 x eager/CUDA graph
  2. MODNet fp16 sanity check
  3. HuBERT CPU reality check (diagnostic only)
  4. Updated budget report
  5. Snapshot validation for best candidate

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_renderer_precision_probe.py

Invoke (NEVER modal run):
    python -c "
    import modal, json
    cls = modal.Cls.from_name('avtr1-renderer-precision-probe', 'RendererPrecisionProbe')
    inst = cls()
    result = inst.full_budget_report.remote()
    print(json.dumps(result, indent=2, default=str))
    "
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path

import modal

PROBE_MOUNT_PATH = "/data/avtr1-renderer-probe"
HF_CACHE_PATH = "/root/.cache/huggingface"

HF_REPO = "avaturn-live/avtr-1"
HF_HUBERT_FILE = "build_artifacts/hubert-lbs-avtr1.onnx"

CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"

probe_image = (
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
        extra_index_url="https://download.pytorch.org/wl/cu128",
    )
    .pip_install(
        "numpy>=2.0,<3",
        "huggingface_hub>=0.30,<1",
        "httpx>=0.28,<0.29",
        "tqdm>=4.67,<5",
        "transformers>=4.45,<5",
        "onnx>=1.17,<2",
        "onnx2torch>=1.5,<2",
        "onnxruntime>=1.17,<2",
    )
)

probe_volume = modal.Volume.from_name("avtr1-renderer-probe-artifacts", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-renderer-precision-probe",
    image=probe_image,
    volumes={
        PROBE_MOUNT_PATH: probe_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
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


def _parity_report_fp16(tensor_fp16, tensor_fp32, label: str) -> dict:
    import torch
    t16 = tensor_fp16.detach().float()
    t32 = tensor_fp32.detach().float()
    diff = (t16 - t32).abs()
    mae = float(diff.mean())
    max_err = float(diff.max())
    flat = diff.flatten()
    p50 = float(torch.kthvalue(flat, max(1, int(0.5 * flat.numel()))).values)
    p95 = float(torch.kthvalue(flat, max(1, int(0.95 * flat.numel()))).values)
    p99 = float(torch.kthvalue(flat, max(1, int(0.99 * flat.numel()))).values)
    cos = float(torch.nn.functional.cosine_similarity(t16.flatten().unsqueeze(0), t32.flatten().unsqueeze(0)).squeeze())
    has_nan = bool(torch.isnan(t16).any())
    has_inf = bool(torch.isinf(t16).any())
    if has_nan or has_inf:
        verdict = "FAIL"
    elif cos > 0.999 and mae < 0.01:
        verdict = "PASS"
    elif cos > 0.99:
        verdict = "MARGINAL"
    else:
        verdict = "FAIL"
    return {
        "label": label,
        "mae": round(mae, 6),
        "max_abs_error": round(max_err, 6),
        "p50_error": round(p50, 6),
        "p95_error": round(p95, 6),
        "p99_error": round(p99, 6),
        "cosine_similarity": round(cos, 6),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "fp16_range": {"min": round(float(t16.min()), 4), "max": round(float(t16.max()), 4)},
        "fp32_range": {"min": round(float(t32.min()), 4), "max": round(float(t32.max()), 4)},
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


# ---------------------------------------------------------------------------
# Model definitions — same as Phase 2A renderer probe
# Source: https://github.com/antgroup/ditto-talkinghead/tree/main/core/models/modules
# ---------------------------------------------------------------------------

def _build_model_classes():
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


def _remove_ddp_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


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


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    timeout=1800,
    secrets=[modal.Secret.from_name("hf-secret")],
)
class RendererPrecisionProbe:

    @modal.enter(snap=True)
    def _snap_init(self):
        import os
        import torch
        from huggingface_hub import hf_hub_download

        self._snap_init_t0 = time.monotonic()
        print("[snap-init] === Stage 2C Renderer Precision Probe ===")
        _run_nvidia_smi("snap-init start")

        hf_token = os.environ.get("HF_TOKEN")
        classes = _build_model_classes()
        StitchingNetwork = classes["StitchingNetwork"]
        WarpingNetwork = classes["WarpingNetwork"]
        SPADEDecoder = classes["SPADEDecoder"]

        RENDERER_REPO = "digital-avatar/ditto-talkinghead"
        WEIGHT_PREFIX = "ditto_pytorch/models"
        cache_dir = str(Path(PROBE_MOUNT_PATH) / "hub")

        weight_files = {}
        for name in ["stitch_network", "warp_network", "decoder"]:
            path = hf_hub_download(
                repo_id=RENDERER_REPO,
                filename=f"{WEIGHT_PREFIX}/{name}.pth",
                repo_type="model",
                token=hf_token,
                cache_dir=cache_dir,
            )
            weight_files[name] = path
            print(f"[snap-init] Downloaded {name}.pth ({Path(path).stat().st_size / 1e6:.1f} MB)")

        stitch_ckpt = torch.load(weight_files["stitch_network"], map_location="cpu", weights_only=False)
        stitch_sd = _remove_ddp_prefix(stitch_ckpt["retarget_shoulder"])
        self.stitch_model = StitchingNetwork()
        self.stitch_model.load_state_dict(stitch_sd)
        for p in self.stitch_model.parameters():
            p.data = p.data.float()
        for b in self.stitch_model.buffers():
            b.data = b.data.float()
        self.stitch_model.cuda().eval()

        warp_sd = torch.load(weight_files["warp_network"], map_location="cpu", weights_only=False)
        self.warp_model = WarpingNetwork()
        self.warp_model.load_state_dict(warp_sd)
        for p in self.warp_model.parameters():
            p.data = p.data.float()
        for b in self.warp_model.buffers():
            b.data = b.data.float()
        self.warp_model.cuda().eval()

        decoder_sd = torch.load(weight_files["decoder"], map_location="cpu", weights_only=False)
        decoder_sd_float = {k: v.float() if hasattr(v, 'float') else v for k, v in decoder_sd.items()}
        self.decoder_model = SPADEDecoder()
        self.decoder_model.load_state_dict(decoder_sd_float)
        self.decoder_model.float().cuda().eval()

        print("[snap-init] Renderer models loaded")

        from transformers import AutoModel
        self.modnet = AutoModel.from_pretrained(
            "boopathiraj/MODNet",
            trust_remote_code=True,
            token=hf_token,
        )
        self.modnet.cuda().eval()
        print("[snap-init] MODNet loaded")

        from onnx2torch import convert
        hubert_onnx = hf_hub_download(
            repo_id="avaturn-live/avtr-1", filename="build_artifacts/hubert-lbs-avtr1.onnx", token=hf_token, cache_dir=HF_CACHE_PATH,
        )
        fixed_onnx = _fix_onnx_conv_biases(hubert_onnx)
        self.hubert_model = convert(fixed_onnx)
        self.hubert_model.cuda().eval()
        print("[snap-init] HuBERT (onnx2torch) loaded")

        with torch.no_grad():
            self.stitch_model(torch.randn(5, 21, 3, device="cuda"), torch.randn(5, 21, 3, device="cuda"))
            self.warp_model(
                torch.randn(2, 32, 16, 64, 64, device="cuda"),
                torch.randn(2, 21, 3, device="cuda"),
                torch.randn(2, 21, 3, device="cuda"),
            )
            self.decoder_model(torch.randn(1, 256, 64, 64, device="cuda"))
            self.decoder_model(torch.randn(5, 256, 64, 64, device="cuda"))
            _, _, matte = self.modnet(torch.randn(1, 3, 288, 512, device="cuda"), True)
            self.hubert_model(torch.randn(2, 8400, device="cuda"))
        print("[snap-init] All warmup done")

        torch.manual_seed(42)
        self._fp32_refs = {}
        with torch.no_grad():
            kp_s = torch.randn(5, 21, 3, device="cuda")
            kp_d = torch.randn(5, 21, 3, device="cuda")
            feat_3d = torch.randn(5, 32, 16, 64, 64, device="cuda")
            dec_in = torch.randn(1, 256, 64, 64, device="cuda")

            self._fp32_refs["stitch"] = self.stitch_model(kp_s, kp_d)
            kp_d_s = self._fp32_refs["stitch"]
            warp_out = self.warp_model(feat_3d, kp_s, kp_d_s)
            self._fp32_refs["warp"] = warp_out
            dec_out = self.decoder_model(dec_in)
            self._fp32_refs["decoder"] = dec_out

            full_out = None
            for i in range(5):
                full_out = self.decoder_model(warp_out[i:i+1])
            self._fp32_refs["full_renderer"] = full_out

            mod_in = torch.randn(1, 3, 288, 512, device="cuda")
            _, _, mod_out = self.modnet(mod_in, True)
            self._fp32_refs["modnet"] = mod_out

        self.vram_after_load = torch.cuda.memory_allocated() / 1024**2
        print(f"[snap-init] VRAM: {self.vram_after_load:.0f} MB")

        self._snap_init_elapsed = time.monotonic() - self._snap_init_t0
        print(f"[snap-init] Done in {self._snap_init_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self):
        import torch
        self._restore_t0 = time.monotonic()
        print("[restore] === Post-restore init ===")
        print(f"[restore] CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM: {vram:.0f} MB")

        assert hasattr(self, "stitch_model"), "StitchingNetwork NOT found"
        assert hasattr(self, "warp_model"), "WarpingNetwork NOT found"
        assert hasattr(self, "decoder_model"), "SPADEDecoder NOT found"
        assert hasattr(self, "modnet"), "MODNet NOT found"
        assert hasattr(self, "hubert_model"), "HuBERT NOT found"

        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    def _health_dict(self):
        import torch
        return {
            "cuda": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "snap_init_s": round(getattr(self, "_snap_init_elapsed", 0), 2),
            "restore_s": round(getattr(self, "_restore_elapsed", 0), 4),
        }

    @modal.method()
    def health_check(self):
        return self._health_dict()

    # ------------------------------------------------------------------
    # Stage 1: Renderer precision benchmarks
    # ------------------------------------------------------------------

    @modal.method()
    def bench_renderer_precision(self) -> dict:
        return self._run_renderer_precision()

    def _run_renderer_precision(self) -> dict:
        import torch

        print("[stage1] === Renderer Precision Benchmarks ===")
        print(f"[stage1] Health: {self._health_dict()}")
        _run_nvidia_smi("stage1 start")

        N_WARMUP = 10
        N_MEASURED = 100
        B = 5

        results = {"stage": "2C-stage1", "health": self._health_dict(), "benchmarks": {}, "parity": {}}

        torch.manual_seed(42)
        kp_s = torch.randn(B, 21, 3, device="cuda")
        kp_d = torch.randn(B, 21, 3, device="cuda")
        feat_3d = torch.randn(B, 32, 16, 64, 64, device="cuda")
        dec_b1 = torch.randn(1, 256, 64, 64, device="cuda")

        self.stitch_model.float().cuda().eval()
        self.warp_model.float().cuda().eval()
        self.decoder_model.float().cuda().eval()

        def _full_renderer_fp32():
            kp_d_s = self.stitch_model(kp_s, kp_d)
            warped = self.warp_model(feat_3d, kp_s, kp_d_s)
            out = None
            for i in range(5):
                out = self.decoder_model(warped[i:i+1])
            return out

        def _stitch_only():
            return self.stitch_model(kp_s, kp_d)

        def _warp_only():
            return self.warp_model(feat_3d, kp_s, kp_d)

        def _decoder_only_b1x5():
            out = None
            for i in range(5):
                out = self.decoder_model(dec_b1)
            return out

        def _bench_eager(name, fn, inp_fn=None):
            print(f"[bench] {name} ...")
            with torch.no_grad():
                for _ in range(N_WARMUP):
                    fn()
                torch.cuda.synchronize()
                lats = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    fn()
                    t1e.record()
                    torch.cuda.synchronize()
                    lats.append(t0e.elapsed_time(t1e))
            stats = _stats_ms(lats)
            stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
            stats["budget_pct_hard"] = round(stats["p50_ms"] / 200 * 100, 1)
            stats["budget_pct_safe"] = round(stats["p50_ms"] / 180 * 100, 1)
            print(f"[bench] {name}: p50={stats['p50_ms']:.3f}ms")
            return stats

        def _bench_cuda_graph(name, fn):
            print(f"[bench-cg] {name} ...")
            try:
                g = torch.cuda.CUDAGraph()
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    for _ in range(N_WARMUP):
                        with torch.no_grad():
                            fn()
                torch.cuda.current_stream().wait_stream(s)

                with torch.cuda.stream(s):
                    with torch.no_grad():
                        torch.cuda.synchronize()
                        g.capture_begin()
                        out = fn()
                        g.capture_end()
                torch.cuda.current_stream().wait_stream(s)

                for _ in range(N_WARMUP):
                    g.replay()
                torch.cuda.synchronize()

                lats = []
                for _ in range(N_MEASURED):
                    t0e = torch.cuda.Event(enable_timing=True)
                    t1e = torch.cuda.Event(enable_timing=True)
                    t0e.record()
                    g.replay()
                    t1e.record()
                    torch.cuda.synchronize()
                    lats.append(t0e.elapsed_time(t1e))

                stats = _stats_ms(lats)
                stats["vram_mb"] = round(torch.cuda.memory_allocated() / 1e6, 1)
                stats["budget_pct_hard"] = round(stats["p50_ms"] / 200 * 100, 1)
                stats["budget_pct_safe"] = round(stats["p50_ms"] / 180 * 100, 1)
                stats["capture_ok"] = True
                print(f"[bench-cg] {name}: p50={stats['p50_ms']:.3f}ms")
                del g
                return stats
            except Exception as e:
                print(f"[bench-cg] {name}: FAILED — {str(e)[:200]}")
                return {"capture_ok": False, "error": str(e)[:300]}

        print("\n[stage1] === FP32 Eager Baseline ===")
        results["benchmarks"]["fp32_eager_full"] = _bench_eager("fp32_eager_full", _full_renderer_fp32)
        results["benchmarks"]["fp32_eager_stitch"] = _bench_eager("fp32_eager_stitch", _stitch_only)
        results["benchmarks"]["fp32_eager_warp"] = _bench_eager("fp32_eager_warp", _warp_only)
        results["benchmarks"]["fp32_eager_decoder_b1x5"] = _bench_eager("fp32_eager_decoder_b1x5", _decoder_only_b1x5)

        print("\n[stage1] === FP32 CUDA Graph ===")
        results["benchmarks"]["fp32_cg_full"] = _bench_cuda_graph("fp32_cg_full", _full_renderer_fp32)
        results["benchmarks"]["fp32_cg_stitch"] = _bench_cuda_graph("fp32_cg_stitch", _stitch_only)
        results["benchmarks"]["fp32_cg_warp"] = _bench_cuda_graph("fp32_cg_warp", _warp_only)
        results["benchmarks"]["fp32_cg_decoder_b1x5"] = _bench_cuda_graph("fp32_cg_decoder_b1x5", _decoder_only_b1x5)

        with torch.no_grad():
            fp32_full_out = _full_renderer_fp32()
        fp32_dec_out = self._fp32_refs["decoder"].clone()

        print("\n[stage1] === FP16 (model.half()) ===")
        self.stitch_model.half().cuda().eval()
        self.warp_model.half().cuda().eval()
        self.decoder_model.half().cuda().eval()
        kp_s_h = kp_s.half()
        kp_d_h = kp_d.half()
        feat_3d_h = feat_3d.half()
        dec_b1_h = dec_b1.half()

        def _full_renderer_fp16():
            kp_d_s = self.stitch_model(kp_s_h, kp_d_h)
            warped = self.warp_model(feat_3d_h, kp_s_h, kp_d_s)
            out = None
            for i in range(5):
                out = self.decoder_model(warped[i:i+1])
            return out

        def _stitch_only_fp16():
            return self.stitch_model(kp_s_h, kp_d_h)

        def _warp_only_fp16():
            return self.warp_model(feat_3d_h, kp_s_h, kp_d_h)

        def _decoder_only_b1x5_fp16():
            out = None
            for i in range(5):
                out = self.decoder_model(dec_b1_h)
            return out

        results["benchmarks"]["fp16_eager_full"] = _bench_eager("fp16_eager_full", _full_renderer_fp16)
        results["benchmarks"]["fp16_eager_stitch"] = _bench_eager("fp16_eager_stitch", _stitch_only_fp16)
        results["benchmarks"]["fp16_eager_warp"] = _bench_eager("fp16_eager_warp", _warp_only_fp16)
        results["benchmarks"]["fp16_eager_decoder_b1x5"] = _bench_eager("fp16_eager_decoder_b1x5", _decoder_only_b1x5_fp16)

        with torch.no_grad():
            fp16_full_out = _full_renderer_fp16()
            fp16_dec_out = _decoder_only_b1x5_fp16()

        results["parity"]["fp16_full_vs_fp32"] = _parity_report_fp16(fp16_full_out, fp32_full_out, "fp16_full_renderer")
        results["parity"]["fp16_decoder_vs_fp32"] = _parity_report_fp16(fp16_dec_out, fp32_dec_out, "fp16_decoder_b1x5")
        print(f"[parity] fp16 full renderer: {results['parity']['fp16_full_vs_fp32']['verdict']}")
        print(f"[parity] fp16 decoder: {results['parity']['fp16_decoder_vs_fp32']['verdict']}")

        print("\n[stage1] === FP16 CUDA Graph ===")
        results["benchmarks"]["fp16_cg_full"] = _bench_cuda_graph("fp16_cg_full", _full_renderer_fp16)
        results["benchmarks"]["fp16_cg_stitch"] = _bench_cuda_graph("fp16_cg_stitch", _stitch_only_fp16)
        results["benchmarks"]["fp16_cg_warp"] = _bench_cuda_graph("fp16_cg_warp", _warp_only_fp16)
        results["benchmarks"]["fp16_cg_decoder_b1x5"] = _bench_cuda_graph("fp16_cg_decoder_b1x5", _decoder_only_b1x5_fp16)

        print("\n[stage1] === Autocast FP16 ===")
        self.stitch_model.float().cuda().eval()
        self.warp_model.float().cuda().eval()
        self.decoder_model.float().cuda().eval()

        def _full_renderer_autocast_fp16():
            with torch.autocast("cuda", dtype=torch.float16):
                kp_d_s = self.stitch_model(kp_s, kp_d)
                warped = self.warp_model(feat_3d, kp_s, kp_d_s)
                out = None
                for i in range(5):
                    out = self.decoder_model(warped[i:i+1])
                return out

        results["benchmarks"]["autocast_fp16_eager_full"] = _bench_eager("autocast_fp16_eager_full", _full_renderer_autocast_fp16)

        with torch.no_grad():
            autocast_full_out = _full_renderer_autocast_fp16()
        results["parity"]["autocast_fp16_full_vs_fp32"] = _parity_report_fp16(autocast_full_out, fp32_full_out, "autocast_fp16_full")
        print(f"[parity] autocast fp16 full: {results['parity']['autocast_fp16_full_vs_fp32']['verdict']}")

        print("\n[stage1] === Autocast BF16 ===")
        def _full_renderer_autocast_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                kp_d_s = self.stitch_model(kp_s, kp_d)
                warped = self.warp_model(feat_3d, kp_s, kp_d_s)
                out = None
                for i in range(5):
                    out = self.decoder_model(warped[i:i+1])
                return out

        results["benchmarks"]["autocast_bf16_eager_full"] = _bench_eager("autocast_bf16_eager_full", _full_renderer_autocast_bf16)

        with torch.no_grad():
            bf16_full_out = _full_renderer_autocast_bf16()
        results["parity"]["autocast_bf16_full_vs_fp32"] = _parity_report_fp16(bf16_full_out, fp32_full_out, "autocast_bf16_full")
        print(f"[parity] autocast bf16 full: {results['parity']['autocast_bf16_full_vs_fp32']['verdict']}")

        self.stitch_model.float().cuda().eval()
        self.warp_model.float().cuda().eval()
        self.decoder_model.float().cuda().eval()

        _run_nvidia_smi("stage1 end")
        return results

    # ------------------------------------------------------------------
    # Stage 2: MODNet fp16 sanity check
    # ------------------------------------------------------------------

    @modal.method()
    def bench_modnet_fp16(self) -> dict:
        return self._run_modnet_fp16()

    def _run_modnet_fp16(self) -> dict:
        import torch

        print("[stage2] === MODNet fp16 Sanity Check ===")
        N_WARMUP = 10
        N_MEASURED = 100
        H, W = 288, 512

        results = {"stage": "2C-stage2", "benchmarks": {}, "parity": {}}

        def _modnet_forward(model, x):
            _, _, matte = model(x, True)
            return matte

        torch.manual_seed(42)
        inp_b1 = torch.randn(1, 3, H, W, device="cuda", dtype=torch.float32)

        self.modnet.float().cuda().eval()

        with torch.no_grad():
            for _ in range(N_WARMUP):
                for _ in range(5):
                    _modnet_forward(self.modnet, inp_b1)
            torch.cuda.synchronize()

            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                for _ in range(5):
                    _modnet_forward(self.modnet, inp_b1)
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))
        results["benchmarks"]["b1x5_eager_fp32"] = _stats_ms(lats)
        print(f"[stage2] fp32 B=1x5 eager: p50={results['benchmarks']['b1x5_eager_fp32']['p50_ms']:.3f}ms")

        try:
            static_in = inp_b1.clone()
            static_out = torch.empty(1, 1, H, W, device="cuda", dtype=torch.float32)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out.copy_(_modnet_forward(self.modnet, static_in))
            for _ in range(N_WARMUP):
                for _ in range(5):
                    g.replay()
            torch.cuda.synchronize()
            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                for _ in range(5):
                    g.replay()
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))
            results["benchmarks"]["b1x5_cg_fp32"] = _stats_ms(lats)
            print(f"[stage2] fp32 B=1x5 CG: p50={results['benchmarks']['b1x5_cg_fp32']['p50_ms']:.3f}ms")
            del g
        except Exception as e:
            results["benchmarks"]["b1x5_cg_fp32"] = {"error": str(e)[:200]}

        with torch.no_grad():
            fp32_out = _modnet_forward(self.modnet, inp_b1)

        self.modnet.half().cuda().eval()
        inp_b1_h = inp_b1.half()

        with torch.no_grad():
            for _ in range(N_WARMUP):
                for _ in range(5):
                    _modnet_forward(self.modnet, inp_b1_h)
            torch.cuda.synchronize()
            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                for _ in range(5):
                    _modnet_forward(self.modnet, inp_b1_h)
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))
        results["benchmarks"]["b1x5_eager_fp16"] = _stats_ms(lats)
        print(f"[stage2] fp16 B=1x5 eager: p50={results['benchmarks']['b1x5_eager_fp16']['p50_ms']:.3f}ms")

        with torch.no_grad():
            fp16_out = _modnet_forward(self.modnet, inp_b1_h)
        results["parity"]["fp16_vs_fp32"] = _parity_report_fp16(fp16_out, fp32_out, "modnet_fp16")
        print(f"[stage2] fp16 parity: {results['parity']['fp16_vs_fp32']['verdict']}")

        try:
            static_in_h = inp_b1_h.clone()
            static_out_h = torch.empty(1, 1, H, W, device="cuda", dtype=torch.float16)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                static_out_h.copy_(_modnet_forward(self.modnet, static_in_h))
            for _ in range(N_WARMUP):
                for _ in range(5):
                    g.replay()
            torch.cuda.synchronize()
            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                for _ in range(5):
                    g.replay()
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))
            results["benchmarks"]["b1x5_cg_fp16"] = _stats_ms(lats)
            print(f"[stage2] fp16 B=1x5 CG: p50={results['benchmarks']['b1x5_cg_fp16']['p50_ms']:.3f}ms")
            del g
        except Exception as e:
            results["benchmarks"]["b1x5_cg_fp16"] = {"error": str(e)[:200]}

        self.modnet.float().cuda().eval()

        fp16_saves = 0
        if "b1x5_cg_fp16" in results["benchmarks"] and "p50_ms" in results["benchmarks"]["b1x5_cg_fp16"]:
            fp32_best = results["benchmarks"].get("b1x5_cg_fp32", {}).get("p50_ms", 999)
            fp16_best = results["benchmarks"]["b1x5_cg_fp16"]["p50_ms"]
            fp16_saves = round(fp32_best - fp16_best, 3)
        results["fp16_saves_ms"] = fp16_saves
        results["recommendation"] = "use_fp16" if fp16_saves > 2 and results["parity"]["fp16_vs_fp32"]["verdict"] == "PASS" else "keep_fp32"

        return results

    # ------------------------------------------------------------------
    # Stage 3: HuBERT CPU reality check (diagnostic only)
    # ------------------------------------------------------------------

    @modal.method()
    def bench_hubert_cpu(self) -> dict:
        return self._run_hubert_cpu()

    def _run_hubert_cpu(self) -> dict:
        import torch
        import numpy as np

        print("[stage3] === HuBERT CPU Reality Check ===")

        results = {"stage": "2C-stage3", "verdict": "pending"}

        torch.manual_seed(42)
        np.random.seed(42)
        test_b2 = torch.randn(2, 8400)

        from onnx2torch import convert
        from huggingface_hub import hf_hub_download
        import os
        hf_token = os.environ.get("HF_TOKEN")
        hubert_onnx = hf_hub_download(
            repo_id="avaturn-live/avtr-1",
            filename="build_artifacts/hubert-lbs-avtr1.onnx",
            token=hf_token,
            cache_dir=HF_CACHE_PATH,
        )
        fixed_onnx = _fix_onnx_conv_biases(hubert_onnx)
        cpu_model = convert(fixed_onnx)
        cpu_model.float().eval()

        with torch.no_grad():
            for _ in range(2):
                output = cpu_model(test_b2)
                if isinstance(output, (tuple, list)):
                    output = output[0]
            print(f"[stage3] CPU sanity: shape={tuple(output.shape)}")

        import time as _time
        N_WARMUP = 3
        N_MEASURED = 10

        with torch.no_grad():
            for _ in range(N_WARMUP):
                cpu_model(test_b2)

            lats = []
            for _ in range(N_MEASURED):
                t0 = _time.perf_counter()
                out = cpu_model(test_b2)
                lats.append((_time.perf_counter() - t0) * 1000)
            if isinstance(out, (tuple, list)):
                out = out[0]

        stats = _stats_ms(lats)
        results["cpu_stats"] = stats
        results["gpu_reference_p50_ms"] = 52.1
        results["cpu_vs_gpu_ratio"] = round(stats["p50_ms"] / 52.1, 1)
        results["verdict"] = "REJECTED" if stats["p50_ms"] > 52.1 else "SURPRISING"
        print(f"[stage3] CPU p50={stats['p50_ms']:.1f}ms vs GPU 52.1ms → {results['verdict']}")
        print(f"[stage3] CPU is {results['cpu_vs_gpu_ratio']}x slower than GPU — NOT a production path")

        return results

    # ------------------------------------------------------------------
    # Stage 4: Full budget report
    # ------------------------------------------------------------------

    @modal.method()
    def full_budget_report(self) -> dict:
        import torch

        print("[budget] === Stage 2C Full Budget Report ===")
        print(f"[budget] Health: {self._health_dict()}")
        _run_nvidia_smi("budget start")

        renderer = self._run_renderer_precision()
        modnet = self._run_modnet_fp16()
        hubert_cpu = self._run_hubert_cpu()

        avtr_ms = 27.0
        hubert_gpu_ms = 52.1

        renderer_bench = renderer.get("benchmarks", {})
        renderer_best_name = None
        renderer_best_ms = 999
        for name, bench in renderer_bench.items():
            if not isinstance(bench, dict) or "p50_ms" not in bench:
                continue
            parity_key = None
            if "full" in name and "fp16" in name:
                parity_key = "fp16_full_vs_fp32"
            elif "full" in name and "autocast_fp16" in name:
                parity_key = "autocast_fp16_full_vs_fp32"
            elif "full" in name and "autocast_bf16" in name:
                parity_key = "autocast_bf16_full_vs_fp32"
            elif "full" in name:
                parity_key = None

            if parity_key and parity_key in renderer.get("parity", {}):
                if renderer["parity"][parity_key]["verdict"] != "PASS":
                    continue

            if bench["p50_ms"] < renderer_best_ms and "full" in name:
                renderer_best_ms = bench["p50_ms"]
                renderer_best_name = name

        modnet_best_name = None
        modnet_best_ms = 10.6
        if modnet.get("recommendation") == "use_fp16":
            for k in ["b1x5_cg_fp16", "b1x5_eager_fp16"]:
                if k in modnet.get("benchmarks", {}) and "p50_ms" in modnet["benchmarks"][k]:
                    modnet_best_name = k
                    modnet_best_ms = modnet["benchmarks"][k]["p50_ms"]
                    break
        else:
            for k in ["b1x5_cg_fp32", "b1x5_eager_fp32"]:
                if k in modnet.get("benchmarks", {}) and "p50_ms" in modnet["benchmarks"][k]:
                    modnet_best_name = k
                    modnet_best_ms = modnet["benchmarks"][k]["p50_ms"]
                    break

        total_ms = avtr_ms + renderer_best_ms + modnet_best_ms + hubert_gpu_ms

        if total_ms <= 180:
            go_nogo = "GO"
            note = "L40S production-safe with good headroom"
        elif total_ms <= 200:
            go_nogo = "TIGHT"
            note = "L40S viable but tight — limited headroom for OmniVoice contention"
        else:
            go_nogo = "NO-GO"
            note = "L40S exceeds hard budget — recommend higher GPU probe"

        budget = {
            "components": {
                "avtr_encode_decode_ms": avtr_ms,
                "renderer_ms": round(renderer_best_ms, 3),
                "renderer_path": renderer_best_name,
                "modnet_ms": round(modnet_best_ms, 3),
                "modnet_path": modnet_best_name,
                "hubert_ms": hubert_gpu_ms,
                "hubert_path": "onnx2torch_fp32_eager_B=2",
            },
            "total_ms": round(total_ms, 1),
            "hard_budget_ms": 200,
            "production_safe_budget_ms": 180,
            "headroom_hard_ms": round(200 - total_ms, 1),
            "headroom_safe_ms": round(180 - total_ms, 1),
            "go_nogo": go_nogo,
            "note": note,
            "omnivoice_contention": {
                "assessment": (
                    f"Budget headroom vs 180ms target: {round(180 - total_ms, 1)}ms. "
                    f"Same-GPU OmniVoice TTS will contend for GPU compute during rendering. "
                    f"If headroom < 20ms, OmniVoice should be scheduled between render chunks "
                    f"or moved to a separate GPU to avoid deadline misses."
                ),
                "recommendation": (
                    "Schedule OmniVoice between render chunks (not concurrent) "
                    "if headroom < 20ms. Consider separate GPU if concurrent TTS is required."
                ),
            },
        }

        print(f"\n[budget] === FINAL BUDGET ===")
        print(f"[budget] AVTR: {avtr_ms}ms")
        print(f"[budget] Renderer ({renderer_best_name}): {renderer_best_ms:.3f}ms")
        print(f"[budget] MODNet ({modnet_best_name}): {modnet_best_ms:.3f}ms")
        print(f"[budget] HuBERT: {hubert_gpu_ms}ms")
        print(f"[budget] TOTAL: {total_ms:.1f}ms")
        print(f"[budget] Hard budget (200ms): {'PASS' if total_ms <= 200 else 'FAIL'} — headroom {200 - total_ms:.1f}ms")
        print(f"[budget] Safe budget (180ms): {'PASS' if total_ms <= 180 else 'FAIL'} — headroom {180 - total_ms:.1f}ms")
        print(f"[budget] VERDICT: {go_nogo} — {note}")

        _run_nvidia_smi("budget end")

        return {
            "stage": "2C",
            "health": self._health_dict(),
            "renderer_results": renderer,
            "modnet_results": modnet,
            "hubert_cpu_results": hubert_cpu,
            "budget": budget,
        }

    @modal.method()
    def snapshot_validation(self) -> dict:
        import torch
        import time as _time

        print("[snapshot] === Snapshot Validation (fp16 models) ===")

        self.stitch_model.half().cuda().eval()
        self.warp_model.half().cuda().eval()
        self.decoder_model.half().cuda().eval()

        N_WARMUP = 5
        N_MEASURED = 20

        kp_s_h = torch.randn(5, 21, 3, device="cuda", dtype=torch.float16)
        kp_d_h = torch.randn(5, 21, 3, device="cuda", dtype=torch.float16)
        feat_3d_h = torch.randn(5, 32, 16, 64, 64, device="cuda", dtype=torch.float16)

        def _full_fp16():
            kp_d_s = self.stitch_model(kp_s_h, kp_d_h)
            warped = self.warp_model(feat_3d_h, kp_s_h, kp_d_s)
            out = None
            for i in range(5):
                out = self.decoder_model(warped[i:i+1])
            return out

        with torch.no_grad():
            for _ in range(N_WARMUP):
                _full_fp16()
            torch.cuda.synchronize()
            lats = []
            for _ in range(N_MEASURED):
                t0e = torch.cuda.Event(enable_timing=True)
                t1e = torch.cuda.Event(enable_timing=True)
                t0e.record()
                _full_fp16()
                t1e.record()
                torch.cuda.synchronize()
                lats.append(t0e.elapsed_time(t1e))

        stats = _stats_ms(lats)
        health = self._health_dict()
        print(f"[snapshot] fp16 full renderer: p50={stats['p50_ms']:.3f}ms")
        print(f"[snapshot] Health: vram={health['vram_mb']}MB restore_s={health.get('restore_s')}s")

        self.stitch_model.float().cuda().eval()
        self.warp_model.float().cuda().eval()
        self.decoder_model.float().cuda().eval()

        return {
            "stage": "2C-snapshot-validation",
            "health": health,
            "benchmark": stats,
            "restore_ok": health.get("restore_s") is not None and health.get("restore_s", 0) > 0,
        }


@app.local_entrypoint()
def main():
    import json
    import modal

    cls = modal.Cls.from_name("avtr1-renderer-precision-probe", "RendererPrecisionProbe")
    inst = cls()
    result = inst.full_budget_report.remote()
    print(json.dumps(result, indent=2, default=str))
