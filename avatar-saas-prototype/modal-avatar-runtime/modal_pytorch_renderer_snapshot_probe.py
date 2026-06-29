"""
ISOLATED FEASIBILITY PROBE — NOT PRODUCTION
Phase 2A: PyTorch LivePortrait Renderer Snapshot Probe

Goal: Determine whether stitch, warp, and decoder can run as pure PyTorch
on L40S with GPU snapshot compatibility and acceptable latency.

Models tested:
  - StitchingNetwork  (simple MLP, 2.3 MB)
  - WarpingNetwork     (DenseMotionNetwork + Hourglass, 174 MB)
  - SPADEDecoder       (6x SPADEResnetBlock, 212 MB)

Excluded (no PyTorch weights available in repo):
  - MODNet  (only modnet.onnx exists; no .pth weights found)

Budget:
  - Full renderer path per 5-frame chunk: stitch -> warp -> decoder x5
  - 200ms per 5-frame chunk target.
  - Combined with proven AVTR numbers (encode ~2ms, decode ~25ms CUDA graph).

Snapshot verification:
  - Cold start + 3 restores with 60s waits (scaledown_window=30).

Deploy:
    modal deploy avatar-saas-prototype/modal-avatar-runtime/modal_pytorch_renderer_snapshot_probe.py

Invoke (NEVER modal run — snapshots disabled for ephemeral apps):
    python -c "
import modal, time, json
cls = modal.Cls.from_name('avtr1-pytorch-renderer-snapshot-probe', 'RendererProbe')
inst = cls()
result = inst.full_probe.remote()
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
    )
)

probe_volume = modal.Volume.from_name("avtr1-renderer-probe-artifacts", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=False)

app = modal.App(
    name="avtr1-pytorch-renderer-snapshot-probe",
    image=probe_image,
    volumes={
        PROBE_MOUNT_PATH: probe_volume,
        HF_CACHE_PATH: hf_cache_vol,
    },
)


def _check_nan(tensor, name: str) -> dict:
    import torch
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    return {
        "name": name,
        "shape": list(tensor.shape),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "min": round(tensor.min().item(), 6) if not has_nan else "NaN",
        "max": round(tensor.max().item(), 6) if not has_nan else "NaN",
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
# Model definitions — wrapped in a factory function so torch.nn.Module is only
# imported inside the Modal container (no torch on the local machine).
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
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    return new_sd


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L40S",
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    timeout=900,
    scaledown_window=30,
    min_containers=0,
    buffer_containers=0,
    max_containers=1,
    secrets=[],
)
class RendererProbe:

    @modal.enter(snap=True)
    def _snap_init(self) -> None:
        import torch
        from huggingface_hub import hf_hub_download

        self._snap_init_t0 = time.monotonic()
        print("[snap-init] === Phase 2A Renderer Probe — Snap Init ===")

        HF_REPO = "digital-avatar/ditto-talkinghead"
        WEIGHT_PREFIX = "ditto_pytorch/models"
        cache_dir = str(Path(HF_CACHE_PATH) / "hub")

        classes = _build_model_classes()
        self.stitch_model = classes["StitchingNetwork"]()
        self.warp_model = classes["WarpingNetwork"]()
        self.decoder_model = classes["SPADEDecoder"]()

        weight_files = {}
        for name in ["stitch_network", "warp_network", "decoder"]:
            t0 = time.monotonic()
            path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"{WEIGHT_PREFIX}/{name}.pth",
                repo_type="model",
                cache_dir=cache_dir,
            )
            dl_s = time.monotonic() - t0
            sz = Path(path).stat().st_size / 1e6
            weight_files[name] = path
            print(f"[snap-init] Downloaded {name}.pth in {dl_s:.1f}s ({sz:.1f} MB)")

        t0 = time.monotonic()
        stitch_ckpt = torch.load(weight_files["stitch_network"], map_location="cpu", weights_only=False)
        stitch_sd = _remove_ddp_prefix(stitch_ckpt["retarget_shoulder"])
        self.stitch_model.load_state_dict(stitch_sd)
        self.stitch_model.cuda().eval()
        print(f"[snap-init] Stitch loaded in {time.monotonic() - t0:.2f}s")

        t0 = time.monotonic()
        warp_sd = torch.load(weight_files["warp_network"], map_location="cpu", weights_only=False)
        self.warp_model.load_state_dict(warp_sd)
        self.warp_model.cuda().eval()
        print(f"[snap-init] Warp loaded in {time.monotonic() - t0:.2f}s")

        t0 = time.monotonic()
        decoder_sd = torch.load(weight_files["decoder"], map_location="cpu", weights_only=False)
        self.decoder_model.load_state_dict(decoder_sd)
        self.decoder_model.cuda().eval()
        print(f"[snap-init] Decoder loaded in {time.monotonic() - t0:.2f}s")

        self.vram_after_load = torch.cuda.memory_allocated() / 1024**2
        print(f"[snap-init] VRAM after all models: {self.vram_after_load:.0f} MB")
        _run_nvidia_smi("snap-init")

        t0 = time.monotonic()
        with torch.no_grad():
            self.stitch_model(torch.randn(5, 21, 3, device="cuda"), torch.randn(5, 21, 3, device="cuda"))
            self.warp_model(
                torch.randn(2, 32, 16, 64, 64, device="cuda"),
                torch.randn(2, 21, 3, device="cuda"),
                torch.randn(2, 21, 3, device="cuda"),
            )
            self.decoder_model(torch.randn(1, 256, 64, 64, device="cuda"))
            self.decoder_model(torch.randn(5, 256, 64, 64, device="cuda"))
        print(f"[snap-init] Warmup done in {time.monotonic() - t0:.2f}s")

        self._snap_init_elapsed = time.monotonic() - self._snap_init_t0
        self._container_id = id(self)
        self._weight_sizes = {k: Path(v).stat().st_size / 1e6 for k, v in weight_files.items()}
        print(f"[snap-init] Done in {self._snap_init_elapsed:.1f}s")

    @modal.enter(snap=False)
    def _post_restore(self) -> None:
        import torch

        self._restore_t0 = time.monotonic()
        print("[restore] === Post-restore init ===")
        print(f"[restore] CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.memory_allocated() / 1024**2
        print(f"[restore] VRAM: {vram:.0f} MB")
        _run_nvidia_smi("restore")

        assert hasattr(self, "stitch_model"), "StitchingNetwork NOT found after restore"
        assert hasattr(self, "warp_model"), "WarpingNetwork NOT found after restore"
        assert hasattr(self, "decoder_model"), "SPADEDecoder NOT found after restore"

        self._container_id = id(self)
        self._restore_elapsed = time.monotonic() - self._restore_t0
        print(f"[restore] Done in {self._restore_elapsed:.4f}s")

    def _make_inputs(self, B: int = 5):
        import torch
        return {
            "kp_source": torch.randn(B, 21, 3, device="cuda"),
            "kp_driving": torch.randn(B, 21, 3, device="cuda"),
            "feature_3d": torch.randn(B, 32, 16, 64, 64, device="cuda"),
            "decoder_feat_b1": torch.randn(1, 256, 64, 64, device="cuda"),
            "decoder_feat_b5": torch.randn(B, 256, 64, 64, device="cuda"),
        }

    def _bench_stage(self, name: str, fn, warmup: int = 10, measured: int = 100) -> dict:
        import torch

        print(f"[bench] === {name} ===")
        t0 = time.monotonic()

        for _ in range(warmup):
            with torch.no_grad():
                fn()

        latencies = []
        for _ in range(measured):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            with torch.no_grad():
                out = fn()
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t1) * 1000)

        stats = _stats_ms(latencies)

        checks = {}
        if isinstance(out, dict):
            for k, v in out.items():
                if hasattr(v, "shape"):
                    checks[k] = _check_nan(v, f"{name}.{k}")
        elif hasattr(out, "shape"):
            checks = _check_nan(out, name)

        vram = torch.cuda.memory_allocated() / 1024**2
        wall = time.monotonic() - t0
        budget = round(stats["p50_ms"] / 200.0 * 100, 1)

        nan_free, inf_free = self._check_nan_inf(checks)

        result = {
            **stats,
            "budget_pct": budget,
            "vram_mb": round(vram, 1),
            "wall_s": round(wall, 1),
            "nan_free": nan_free,
            "inf_free": inf_free,
            "output_checks": checks,
        }
        print(f"[bench] {name}: p50={stats['p50_ms']}ms budget={budget}% ({wall:.1f}s)")
        return result

    @staticmethod
    def _check_nan_inf(checks):
        if not checks:
            return True, True
        if "has_nan" in checks:
            return not checks.get("has_nan", False), not checks.get("has_inf", False)
        nan_ok = all(not c.get("has_nan", False) for c in checks.values())
        inf_ok = all(not c.get("has_inf", False) for c in checks.values())
        return nan_ok, inf_ok

    def _try_cuda_graph(self, name: str, fn, warmup: int = 10, measured: int = 100) -> dict:
        import torch

        print(f"[cuda-graph] === {name} ===")
        t0 = time.monotonic()

        try:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(warmup):
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

            capture_s = time.monotonic() - t0
            print(f"[cuda-graph] {name}: captured in {capture_s:.2f}s")

            latencies = []
            for _ in range(measured):
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                g.replay()
                torch.cuda.synchronize()
                latencies.append((time.perf_counter() - t1) * 1000)

            stats = _stats_ms(latencies)

            checks = {}
            if isinstance(out, dict):
                for k, v in out.items():
                    if hasattr(v, "shape"):
                        checks[k] = _check_nan(v, f"{name}.{k}")
            elif hasattr(out, "shape"):
                checks = _check_nan(out, name)

            vram = torch.cuda.memory_allocated() / 1024**2
            wall = time.monotonic() - t0
            budget = round(stats["p50_ms"] / 200.0 * 100, 1)

            nan_free, inf_free = self._check_nan_inf(checks)

            result = {
                **stats,
                "budget_pct": budget,
                "vram_mb": round(vram, 1),
                "wall_s": round(wall, 1),
                "capture_s": round(capture_s, 2),
                "capture_ok": True,
                "nan_free": nan_free,
                "inf_free": inf_free,
                "output_checks": checks,
            }
            print(f"[cuda-graph] {name}: p50={stats['p50_ms']}ms budget={budget}% (capture {capture_s:.2f}s)")
            return result

        except Exception as e:
            capture_s = time.monotonic() - t0
            err_str = str(e)
            print(f"[cuda-graph] {name}: FAILED — {err_str[:200]} ({capture_s:.2f}s)")
            return {
                "capture_ok": False,
                "error": err_str[:500],
                "capture_attempt_s": round(capture_s, 2),
            }

    def _health_dict(self) -> dict:
        import torch
        return {
            "ok": True,
            "app": "avtr1-pytorch-renderer-snapshot-probe",
            "gpu": torch.cuda.get_device_name(0),
            "vram_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "snap_init_s": round(self._snap_init_elapsed, 2),
            "restore_s": round(getattr(self, "_restore_elapsed", 0), 4),
            "container_id": self._container_id,
            "weight_sizes_mb": self._weight_sizes,
            "modnet_status": "EXCLUDED — no PyTorch weights available in repo (only modnet.onnx exists)",
        }

    @modal.method()
    def get_health(self) -> dict:
        return self._health_dict()

    @modal.method()
    def full_probe(self) -> dict:
        import torch

        print("[probe] === Phase 2A Full Probe ===")
        probe_t0 = time.monotonic()

        health = self._health_dict()
        print(f"[probe] Health: vram={health['vram_mb']}MB restore_s={health['restore_s']}s")

        B = 5
        inp = self._make_inputs(B)
        kp_s = inp["kp_source"]
        kp_d = inp["kp_driving"]
        feat_3d = inp["feature_3d"]
        dec_b1 = inp["decoder_feat_b1"]
        dec_b5 = inp["decoder_feat_b5"]

        benchmarks = {}

        benchmarks["stitch_eager"] = self._bench_stage(
            "Stitch (eager, B=5)",
            lambda: self.stitch_model(kp_s, kp_d),
        )
        benchmarks["stitch_cuda_graph"] = self._try_cuda_graph(
            "Stitch (cuda_graph, B=5)",
            lambda: self.stitch_model(kp_s, kp_d),
        )

        benchmarks["warp_eager"] = self._bench_stage(
            "Warp (eager, B=5)",
            lambda: self.warp_model(feat_3d, kp_s, kp_d),
        )
        benchmarks["warp_cuda_graph"] = self._try_cuda_graph(
            "Warp (cuda_graph, B=5)",
            lambda: self.warp_model(feat_3d, kp_s, kp_d),
        )

        def _decoder_b1_x5():
            out = None
            for i in range(5):
                out = self.decoder_model(dec_b1)
            return out

        benchmarks["decoder_b1x5_eager"] = self._bench_stage(
            "Decoder (eager, B=1 x5)",
            _decoder_b1_x5,
        )
        benchmarks["decoder_b1x5_cuda_graph"] = self._try_cuda_graph(
            "Decoder (cuda_graph, B=1 x5)",
            _decoder_b1_x5,
        )

        benchmarks["decoder_b5_eager"] = self._bench_stage(
            "Decoder (eager, B=5 batched)",
            lambda: self.decoder_model(dec_b5),
        )
        benchmarks["decoder_b5_cuda_graph"] = self._try_cuda_graph(
            "Decoder (cuda_graph, B=5 batched)",
            lambda: self.decoder_model(dec_b5),
        )

        def _full_renderer_eager():
            kp_d_stitched = self.stitch_model(kp_s, kp_d)
            warped = self.warp_model(feat_3d, kp_s, kp_d_stitched)
            out = None
            for i in range(5):
                out = self.decoder_model(warped[i:i+1])
            return out

        benchmarks["full_renderer_eager"] = self._bench_stage(
            "Full Renderer (eager: stitch+warp+decoder_b1x5)",
            _full_renderer_eager,
        )
        benchmarks["full_renderer_cuda_graph"] = self._try_cuda_graph(
            "Full Renderer (cuda_graph: stitch+warp+decoder_b1x5)",
            _full_renderer_eager,
        )

        wall = time.monotonic() - probe_t0
        print(f"[probe] All benchmarks done in {wall:.1f}s")

        modnet_note = (
            "MODNet is EXCLUDED from this probe. "
            "No PyTorch (.pth) weights exist in the repository — only modnet.onnx (25 MB). "
            "MODNet source: ZHKKKe/MODNet (GitHub). Weights need separate sourcing. "
            "Production TRT estimate: ~2.25ms/frame on L4, ~5ms total on L40S. "
            "Phase 2B should resolve MODNet PyTorch port or accept ONNX/TRT for this stage."
        )

        return {
            "ok": True,
            "wall_s": round(wall, 1),
            "health": health,
            "benchmarks": benchmarks,
            "modnet_status": modnet_note,
            "budget_note": (
                "Budget uses 200ms per 5-frame chunk. "
                "AVTR encode ~2ms + AVTR decode CUDA graph ~25ms = 27ms (proven Phase 1.5). "
                "Renderer numbers are from this probe (stitch+warp+decoder). "
                "MODNet is NOT included — see modnet_status."
            ),
        }
