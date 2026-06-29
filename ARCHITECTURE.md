# AVTR-1 Complete Architecture Document

> Flow-matching-based autoregressive model for live dialogue avatars.
> Given a portrait image and dual-stream audio, renders lip-synced speech
> and active listening at 25 FPS on a single GPU.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [High-Level Pipeline Architecture](#2-high-level-pipeline-architecture)
3. [Phase 1: Avatar Registration](#3-phase-1-avatar-registration-once-per-portrait)
4. [Phase 2: Per-Chunk Rendering Pipeline](#4-phase-2-per-chunk-rendering-pipeline)
5. [AVTR1 Motion Generation (Core Model)](#5-avtr1-motion-generation-core-model)
6. [Rendering Pipeline (Motion to Pixels)](#6-rendering-pipeline-motion-to-pixels)
7. [Autoregressive State Management](#7-autoregressive-state-management)
8. [Runtime Engine Abstraction](#8-runtime-engine-abstraction)
9. [Complete Engine Inventory](#9-complete-engine-inventory)
10. [Live Streaming Architecture](#10-live-streaming-architecture-avaturn_live_streamer)
11. [Conversation Engines](#11-conversation-engines)
12. [Renderer HTTP API](#12-renderer-http-api)
13. [Artifact Management](#13-artifact-management)
14. [Intro Motion System](#14-intro-motion-system)
15. [Dyadic Audio Architecture](#15-dyadic-audio-architecture-the-key-innovation)
16. [Build and Deploy Pipeline](#16-build-and-deploy-pipeline)
17. [End-to-End Timing Budget](#17-end-to-end-timing-budget)
18. [Key Design Decisions and Optimizations](#18-key-design-decisions-and-optimizations)
19. [Normalizer and Z-Score Convention](#19-normalizer-and-z-score-convention)
20. [Configuration Reference](#20-configuration-reference)

---

## 1. System Overview

AVTR-1 is a real-time conversational AI avatar system that animates a single
portrait photo using dual-stream audio (speaker + listener). It generates
lip-synced speech and active listening at **25 FPS** on a single GPU using a
**flow-matching autoregressive model** combined with a **LivePortrait-style
warping renderer**.

The system combines a **Latent Motion Flow Matching (LMFM)** module with a
portrait-warping renderer. Motion is generated in a compact latent space at
25 FPS using continuous flow matching with an Euler ODE solver; rendering is
offloaded to a LivePortrait-style warping pipeline that preserves identity
and produces 512x512 video at interactive latency.

### 1.1 Components and Licensing

| Component | Package | License | Purpose |
|-----------|---------|---------|---------|
| Model and Scripts | `scripts/` + weights | AVTR-1 Community License | Build tools, offline generation |
| Renderer | `src/avtr1_renderer/` | PolyForm Noncommercial 1.0.0 | GPU inference pipeline |
| Streamer | `src/avaturn_live_streamer/` | PolyForm Noncommercial 1.0.0 | Live streaming orchestration |

### 1.2 Key Performance Numbers

| GPU | Latency / 5-frame chunk | Real-time factor |
|-----|--------------------------|------------------|
| L40 | 84 ms | 2.4x |
| A100 | 91 ms | 2.2x |
| RTX 4060 Ti | 166 ms | 1.2x |
| RTX 3070 | 181 ms | 1.1x |
| L4 | 202 ms | 0.99x |
| RTX 3060 Ti | 206 ms | 0.97x |
| RTX 4060 | 232 ms | 0.86x |

Real-time factor = 200 ms / latency. >= 1.0x means the GPU keeps up with 25 FPS.

### 1.3 Prerequisites

- Linux
- NVIDIA GPU (Ampere or later recommended)
- CUDA 12.x + TensorRT 10.x
- pixi package manager

### 1.4 Project Provenance

- The AVTR-1 motion generator was initially prototyped from
  `thuhcsi/S2G-MDDiffusion` (MIT). Broader lineage includes EDGE and lucidrains
  diffusion implementations.
- The renderer pipeline is built on **LivePortrait** (MIT, Kuaishou). LivePortrait
  ONNX checkpoints are pulled from `digital-avatar/ditto-talkinghead` (Apache-2.0).
- Bundled third-party components: HuBERT (Apache-2.0), MODNet (Apache-2.0),
  grid_sample TensorRT plugin (Apache-2.0).
- Non-commercial dependency: InsightFace SCRFD detector + 2D106 landmark model
  (non-commercial research use only).

---

## 2. High-Level Pipeline Architecture

```
+--------------------------------------------------------------------------+
|                         AVTR-1 FULL PIPELINE                            |
|                                                                          |
|  +--------------+    +------------------+    +--------------------+      |
|  |  PORTRAIT PNG |--->| Avatar Registration|--->|     Avatar         |    |
|  |  (reference)  |    | (once per avatar) |    |  (KPInfo + f_s +   |    |
|  +--------------+    +------------------+    |   mask + M_grid)   |    |
|                                               +--------+-----------+    |
|                                                         |                |
|  +--------------+    +----------------------------------+-----------+   |
|  | DUAL-STREAM  |    |    PER-CHUNK PIPELINE (5 frames @ 25fps)    |   |
|  | AUDIO (16kHz)|--->|                                            |   |
|  | speech+listen|    |  +---------+   +--------------+            |   |
|  +--------------+    |  | HuBERT  |-->| AVTR1 Encode |            |   |
|                       |  | (B=2)   |   |  (1x/chunk)  |            |   |
|                       |  +---------+   +------+-------+            |   |
|                       |                       |                     |   |
|                       |          +------------v------------+       |   |
|                       |          |   AVTR1 Decode (ODE)    |       |   |
|                       |          |   5 Euler steps with    |       |   |
|                       |          |   CFG weights           |       |   |
|                       |          +------------+------------+       |   |
|                       |                       | MotionFrame        |   |
|                       |                       | (R + exp_lipsync)  |   |
|                       |          +------------v------------+       |   |
|                       |          |    Motion Stitch         |       |   |
|                       |          |  (splice + transform +   |       |   |
|                       |          |   stitch refinement)     |       |   |
|                       |          +------------+------------+       |   |
|                       |              (x_s,x_d)|                   |   |
|                       |          +------------v------------+       |   |
|                       |          |   Warp Network (B=5)    |       |   |
|                       |          +------------+------------+       |   |
|                       |          +------------v------------+       |   |
|                       |          |  SPADE Decoder (B=1)    |       |   |
|                       |          +------------+------------+       |   |
|                       |          +------------v------------+       |   |
|                       |          |  Putback + MODNet Matte |       |   |
|                       |          +------------+------------+       |   |
|                       |          +------------v------------+       |   |
|                       |          |  Pixel Format Pack      |       |   |
|                       |          |  (YUV I420 + alpha)     |       |   |
|                       |          +------------+------------+       |   |
|                       +-----------------------+--------------------+   |
|                                               |                         |
|                                    +----------v----------+              |
|                                    |  Host Frames (uint8) |              |
|                                    |  25 fps YUV/RGBA     |              |
|                                    +---------------------+              |
+--------------------------------------------------------------------------+
```

---

## 3. Phase 1: Avatar Registration (Once Per Portrait)

Source: `src/avtr1_renderer/avatar_loader.py`

This pipeline runs **once per portrait image** and produces an immutable `Avatar`
object that lives entirely on CUDA. The registration cascade involves 5 neural
engines and multiple geometric transforms.

### 3.1 Registration Cascade

```
Portrait PNG (any resolution)
  |
  +-- Load image (RGB/RGBA via OpenCV, resize to output 720x1280)
  |
  +-- If RGBA: composite over grey-200 background (for MODNet later)
  |     img_bg = rgb * alpha + 200.0 * (1 - alpha)  [float64, matching reference]
  |
  +-- Face Detection (InsightFace SCRFD)
  |     Engine: insightface_det.onnx
  |     Input:  (1, 3, 512, 512) letterbox-resized image
  |     Architecture: Multi-scale FPN with strides {8, 16, 32}
  |     Anchor counts: 8192 / 2048 / 512 (2 anchors per location)
  |     Outputs: 9 tensors (3 FPN levels x {scores, boxes, keypoints})
  |     Post-process: NMS (thresh=0.4), pick largest face bbox
  |     Result: bounding box (x1,y1,x2,y2,score), 5 facial keypoints (x,y) x5
  |
  +-- Landmark106 (InsightFace 2D106)
  |     Engine: landmark106.onnx
  |     Input:  (1, 3, 192, 192) similarity-transform aligned crop from bbox
  |     Output: (106, 2) coarse facial landmarks in original image coords
  |
  +-- Initial 224x224 Crop
  |     crop_image(img_rgb, lmk106_pts, dsize=224, scale=1.5, vy_ratio=-0.1)
  |     Produces: img_crop_224, M_c2o (crop-to-original affine)
  |
  +-- Landmark203 (LivePortrait fine-grained)
  |     Engine: landmark203.onnx
  |     Input:  (1, 3, 224, 224) face crop [0,1] float
  |     Output: (203, 2) refined landmarks mapped to original coords via M_c2o
  |
  +-- Final 512x512 Crop
  |     crop_image(img_rgb, lmk203_pts, dsize=512, scale=2.2, vy_ratio=-0.1,
  |                 flag_do_rot=True)
  |     Produces: img_crop_512, M_c2o (crop-to-original affine 3x3)
  |
  +-- Resize 512 -> 256 for extractors
  |     cv2.INTER_AREA downsampling -> (1, 3, 256, 256) float32 [0,1]
  |
  +-- Appearance Extractor
  |     Engine: appearance_extractor.onnx
  |     Input:  (1, 3, 256, 256) face crop [0,1]
  |     Output: f_s = (1, 32, 16, 64, 64) 3D appearance feature volume
  |     This is the feature_3d input to the Warp network.
  |
  +-- Motion Extractor (LivePortrait)
  |     Engine: motion_extractor.onnx
  |     Input:  (1, 3, 256, 256) face crop [0,1]
  |     Raw outputs:
  |       pitch (1, 66)  -- 66-bin classification logits
  |       yaw   (1, 66)  -- 66-bin classification logits
  |       roll  (1, 66)  -- 66-bin classification logits
  |       t     (1, 3)   -- translation vector
  |       exp   (1, 63)  -- expression coefficients (= 21 keypoints x 3)
  |       scale (1, 1)   -- scale factor
  |       kp    (1, 63)  -- keypoint offsets (= 21 keypoints x 3)
  |     Post-processing:
  |       degree = sum(softmax(logits) * arange(66)) * 3 - 97.5
  |       R = (Rz(roll) @ Ry(yaw) @ Rx(pitch)).T  -> (1, 3, 3)
  |       kp reshaped to (1, 21, 3), exp reshaped to (1, 21, 3)
  |     Result: KPInfo
  |
  +-- Pasteback Mask
  |     Default: elliptical gradient mask (0.9 x 0.9 ratio) at 512x512
  |     Or per-avatar: {portrait_stem}.pbmask.png next to portrait
  |     Pre-warped to original frame coords via kornia.warp_affine + M_c2o
  |     Result: mask = (1, H, W) float32 [0,1] on CUDA
  |
  +-- Precomputed M_grid (cached affine for putback)
        normalize_homography(M_c2o, (512,512), (H,W)) -> torch.linalg.inv()
        -> (2, 3) tensor on CUDA
        Avoids 40-kernel LU factorization + 2 D2H sync points per putback call.
```

### 3.2 Avatar Data Structure

Source: `src/avtr1_renderer/avatar_loader.py:63`

```python
@dataclass(slots=True, frozen=True)
class Avatar:
    id: str
    kp_info: KPInfo            # Identity pose/keypoints/expression
    f_s: torch.Tensor          # (1, 32, 16, 64, 64) 3D appearance features
    M_grid: torch.Tensor       # (2, 3) cached inverse affine for putback
    mask: torch.Tensor         # (1, H, W) pre-warped pasteback mask
    source: torch.Tensor       # (3, H, W) grey-bg portrait, float32 [0,1]
    no_matting: bool = False   # True if 3-ch RGB (bg baked in), skips MODNet
```

### 3.3 KPInfo Data Structure

Source: `src/avtr1_renderer/types.py:34`

```python
@dataclass(slots=True, frozen=True)
class KPInfo:
    kp: torch.Tensor     # (1, 21, 3) canonical 3D keypoints
    exp: torch.Tensor    # (1, 21, 3) expression deltas
    scale: torch.Tensor  # (1, 1)     per-portrait scale
    t: torch.Tensor      # (1, 3)     translation
    pitch: torch.Tensor  # (1, 1)     pitch in degrees
    yaw: torch.Tensor    # (1, 1)     yaw in degrees
    roll: torch.Tensor   # (1, 1)     roll in degrees
    R: torch.Tensor      # (1, 3, 3)  rotation matrix derived from pitch/yaw/roll
```

---

## 4. Phase 2: Per-Chunk Rendering Pipeline

Source: `src/avtr1_renderer/pipeline.py`, `src/avtr1_renderer/avtr1_motion_generator.py`

### 4.1 Constants and Timing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| FPS | 25 | Output frame rate |
| SAMPLE_RATE | 16,000 Hz | Audio sample rate |
| chunk_size | 5 | Motion frames generated per chunk |
| past_size | 75 | Autoregressive context window (15 chunks x 5) |
| future_size | 5 | Lookahead motion frames for audio context |
| frame_len | 640 | Audio samples per motion frame (16000 / 25) |
| audio_shift | 80 | Extra audio samples for alignment offset |
| nfeats | 42 | Motion feature dimension (3 so3 + 39 lipsync_exp) |
| latent_dim | 512 | Internal transformer hidden dimension |
| AUDIO_DIM | 1024 | Per-track HuBERT feature dimension |
| n_ode_steps | 5 | Euler integration steps in flow-matching ODE |
| HUBERT_PAST_FRAMES | 3 | HuBERT left-context frames |

### 4.2 Audio Window per Chunk

Each chunk consumes this much raw audio **per track** (speech and listen):

```
audio_length = (chunk_size + future_size) * frame_len + audio_shift
             = (5 + 5) * 640 + 80
             = 6,480 samples per track at 16 kHz
             = 405 ms of audio per track per chunk
```

Total audio per call (both tracks): 12,960 samples = 810 ms.

### 4.3 Pipeline Entry Point

```python
# Pipeline.process_chunk() -- the single entry point
state, frame_iterator = pipeline.process_chunk(
    avatar=avatar,
    chunk=Chunk(audio_speech=np_array, audio_listen=np_array),
    state=state_or_none,
    options=RenderOptions(bg_id="minimal_office"),
)
for frame in frame_iterator:
    # frame is a Frame(data=uint8_array, format="yuv_i420", height=720, width=1280)
    ...
```

### 4.4 LIPSYNC_COORDS

Source: `src/avtr1_renderer/constants.py`

The motion model predicts **39 of 63** expression coordinates. The 63 values are
the flattened (21 keypoints x 3 coordinates) expression vector. The 39 predicted
indices cover mouth, jaw, brow, and eye regions:

```python
LIPSYNC_COORDS = (
    3, 4, 5,       # keypoint 1 (xyz)
    6, 7, 8,       # keypoint 2 (xyz)
    18, 19, 20,    # keypoint 6 (xyz)
    33, 34, 35,    # keypoint 11 (xyz)
    36, 37, 38,    # keypoint 12 (xyz)
    39, 40, 41,    # keypoint 13 (xyz)
    42, 43, 44,    # keypoint 14 (xyz)
    45, 46, 47,    # keypoint 15 (xyz)
    48, 49, 50,    # keypoint 16 (xyz)
    51, 52, 53,    # keypoint 17 (xyz)
    54, 55, 56,    # keypoint 18 (xyz)
    57, 58, 59,    # keypoint 19 (xyz)
    60, 61, 62,    # keypoint 20 (xyz)
)
```

The remaining 24 coordinates stay at source-portrait values.

---

## 5. AVTR1 Motion Generation (Core Model)

Source: `src/avtr1_renderer/avtr1_motion_generator.py`

This is the **Latent Motion Flow Matching (LMFM)** module -- the heart of AVTR-1.
It converts dual-stream audio into per-frame motion (rotation + lipsync expression).

### 5.1 Per-Chunk Flow (Detailed)

```
Step 1: HuBERT Feature Extraction
  + Concatenate 3-frame left-context tail (from state) + new chunk audio
  + Stack speech || listen as (2, ~8480) batch
  + Run HuBERT engine (B=2) -> (2, frames_50hz, 1024)
  + Pairwise mean (50 Hz -> 25 Hz) -> (2, 10, 1024)
  + Concatenate speech + listen along feature dim -> (1, 10, 2048)

Step 2: Build Full Audio Context
  + Prepend 75-frame history from state -> audio_cond (1, 85, 2048)

Step 3: Build Keypoint Conditioning
  + Normalize avatar's so3, kp, exp with z-score normalizer
  + Concatenate -> kp_cond (1, 1, 129)

Step 4: Encode (run once)
  + Input:  past_cond(1,75,42), audio_cond(1,85,2048), kp_cond(1,1,129),
             past_times(1,15,1)  [all 1.0 at inference]
  + Output: kp_tokens(1,1,512), past_context(1,70,512), past_last(1,5,512),
             audio_self(1,85,512), audio_other(1,85,512)

Step 5: Sample Initial Noise (Progressive AR(1))
  + Generate z ~ truncated normal with AR(1) correlation:
      x[0] = alpha * prev + std * eps  (or pure eps on cold start)
      x[t] = alpha * x[t-1] + std * eps(x[t-1])
  + alpha/noise_alpha=2.0, trunc_z=1.2
  + Result: z = (1, 5, 42) initial noise

Step 6: ODE Euler Integration (5 steps)
  + times = linspace(0, 1, 5) on GPU -> [0.0, 0.25, 0.5, 0.75, 1.0]
  + dt = 1.0 / 4 = 0.25
  + for i in range(4):
      v = decode(x, *conditions, w_self, w_other, w_kp, t=times[i])
      x += dt * v

Step 7: De-Normalize Motion
  + so3 = x[:, :3] * scale_so3 + offset_so3 -> (T, 3) rotation vectors
  + exp_lipsync = x[:, 3:42] * scale + offset -> (T, 39) lipsync expression
  + R = roma.rotvec_to_rotmat(so3) -> (T, 3, 3) rotation matrices
  + Result: MotionFrame(R=(T,3,3), exp=(T,39))

Step 8: Update State
  + past_cond: append 5 new normalized frames, drop oldest 5 (rolling window)
  + audio_features: drop oldest 5, append 5 "current" features
  + audio_prev_*: roll forward by chunk_size * frame_len samples
  + noise_shared: carry last frame of z
```

### 5.2 AVTR1 Encode Engine

Source: `src/avtr1_renderer/models/avtr1.py:29`

Runs **once per chunk**. Converts conditioning into attention-ready tokens.

| Direction | Field | Shape | Dtype | Description |
|-----------|-------|-------|-------|-------------|
| Input | `past_cond` | `(1, 75, 42)` | float32 | Z-score normalized past motion |
| Input | `audio_cond` | `(1, 85, 2048)` | float32 | Speech and listen HuBERT features concatenated |
| Input | `kp_cond` | `(1, 1, 129)` | float32 | Normalized identity: so3(3) + kp(63) + exp(63) |
| Input | `past_times` | `(1, 15, 1)` | float32 | All 1.0 at inference (every past chunk is "clean") |
| Output | `kp_tokens` | `(1, 1, 512)` | float32 | Identity keypoint cross-attention tokens |
| Output | `past_context` | `(1, 70, 512)` | float32 | Bulk past context (75 - 5 = 70 frames) |
| Output | `past_last` | `(1, 5, 512)` | float32 | Most recent chunk context |
| Output | `audio_self` | `(1, 85, 512)` | float32 | Speaker's own audio embedding |
| Output | `audio_other` | `(1, 85, 512)` | float32 | Listener's audio embedding |

Audio sequence length: 85 = past_size(75) + chunk_size(5) + future_size(5).

### 5.3 AVTR1 Decode Engine (Flow-Matching ODE Solver)

Source: `src/avtr1_renderer/models/avtr1.py:46`

Runs **n_ode_steps=5 times** per chunk (Euler integration from t=0 to t=1).

| Direction | Field | Shape | Dtype | Description |
|-----------|-------|-------|-------|-------------|
| Input | `x` | `(1, 5, 42)` | float32 | Current ODE state (normalized motion) |
| Input | `kp_tokens` | `(1, 1, 512)` | float32 | From encode output |
| Input | `past_context` | `(1, 70, 512)` | float32 | From encode output |
| Input | `past_last` | `(1, 5, 512)` | float32 | From encode output |
| Input | `audio_self` | `(1, 85, 512)` | float32 | From encode output |
| Input | `audio_other` | `(1, 85, 512)` | float32 | From encode output |
| Input | `w_self` | `(512,)` | float32 | Per-coordinate CFG weight for self audio |
| Input | `w_other` | `(512,)` | float32 | Per-coordinate CFG weight for other audio |
| Input | `w_kp` | `(512,)` | float32 | Per-coordinate CFG weight for identity/kp |
| Input | `t` | `(1, 1)` | float32 | Current ODE timestep in [0, 1] |
| Output | `output` | `(1, 5, 42)` | float32 | Flow-field velocity v(x, t) |

### 5.4 Classifier-Free Guidance (CFG)

The decode TRT engine internally fans out **4 condition passes**:

1. **past** -- unconditional baseline (past motion only)
2. **self_audio** -- conditioned on speaker's audio
3. **other_audio** -- conditioned on listener's audio
4. **kp** -- conditioned on identity keypoints

The three weight tensors (`w_self`, `w_other`, `w_kp`) control how strongly each
conditional mode pulls away from the unconditional baseline. These are **runtime
inputs** so guidance can be adjusted per request without rebuilding the engine.

Default CFG weights:

| Weight | Default | Controls |
|--------|---------|----------|
| `cfg_self_audio` | 2.0 | Strength of lip-sync from speaker's audio |
| `cfg_other_audio` | 2.0 | Strength of listening reactions from partner's audio |
| `cfg_kp` | 3.0 | Strength of identity/pose conditioning |

### 5.5 Initial Noise (Progressive AR(1))

Source: `src/avtr1_renderer/avtr1_motion_generator.py:183`

The starting noise `z` for each chunk is **correlated** with the previous chunk
via an AR(1) process:

```
coef = alpha / sqrt(1 + alpha^2)
std  = 1 / sqrt(1 + alpha^2)

x[0] = coef * prev_last_frame + std * eps(prev)   # or pure eps on cold start
x[t] = coef * x[t-1] + std * eps(x[t-1])
```

Where `eps()` is drawn from truncated normal N(0,1) clipped to [-trunc_z, trunc_z].

Default parameters: `noise_alpha=2.0`, `noise_trunc_z=1.2`.

The last frame of each chunk's noise is carried as `noise_shared` to the next
chunk for temporal continuity.

### 5.6 Motion De-Normalization

Source: `src/avtr1_renderer/avtr1_motion_generator.py:541`

The ODE output `x` is in **z-score normalized** space. De-normalization:

```
so3 = so3_normalized * scale_so3 + offset_so3              -> (T, 3) rotation vectors
exp_lipsync = exp_n * exp_lipsync_scale + exp_lipsync_offset -> (T, 39) lipsync expression
R = roma.rotvec_to_rotmat(so3)                              -> (T, 3, 3) rotation matrices
```

The `Normalizer` object holds all z-score parameters:

```python
@dataclass(slots=True)
class Normalizer:
    offset_so3: Tensor       # (3,)
    scale_so3: Tensor        # (3,)
    offset_kp: Tensor        # (21, 3)
    scale_kp: Tensor         # (21, 3)
    offset_exp: Tensor       # (21, 3)
    scale_exp: Tensor        # (21, 3)
    exp_lipsync_offset: Tensor  # (39,) -- pre-sliced for de-norm
    exp_lipsync_scale: Tensor   # (39,)
```

Loaded from `avtr1_normalizer.safetensors` sidecar written during TRT engine build.

### 5.7 MotionFrame

Source: `src/avtr1_renderer/components/liveportrait/motion_stitch.py:42`

```python
@dataclass(slots=True, frozen=True)
class MotionFrame:
    R: torch.Tensor    # (N, 3, 3) rotation matrices
    exp: torch.Tensor  # (N, 39)   lipsync expression deltas
```

Supports `__len__`, `__iter__`, `__getitem__` for both int and slice indexing.

---

## 6. Rendering Pipeline (Motion to Pixels)

Source: `src/avtr1_renderer/renderer.py`, `src/avtr1_renderer/components/`

### 6.1 Motion Stitch

Source: `src/avtr1_renderer/components/liveportrait/motion_stitch.py:141`

Combines predicted motion with source identity:

1. **Build expression delta:** Start from source `exp` (1,21,3) flattened to (63,),
   overwrite LIPSYNC_COORDS indices with predicted values -> `(N, 21, 3)` delta
2. **Transform driving keypoints:**
   `x_d = scale * ((kp_canonical + delta) @ R_new) + t` -> `(N, 21, 3)`
3. **Transform source keypoints:**
   `x_s = scale * ((kp @ R_source) + exp_source) + t` -> `(1, 21, 3)`
4. **Stitch network refinement:** Single batched call for B=1..5.
   Input `(x_s, x_d)` -> refined `(x_s, x_d)`. Closes mouth-region seams.

Stitch Engine I/O:

| Direction | Field | Shape | Description |
|-----------|-------|-------|-------------|
| Input | `kp_source` | `(N, 21, 3)` | Source keypoints |
| Input | `kp_driving` | `(N, 21, 3)` | Driving keypoints |
| Output | `out` | `(N, 21, 3)` | Refined driving keypoints |

### 6.2 Warp Network (Batched B=5)

Source: `src/avtr1_renderer/models/warp.py`

Runs **once on the full 5-frame chunk** (shares work across frames).

| Direction | Field | Shape | Description |
|-----------|-------|-------|-------------|
| Input | `feature_3d` | `(5, 32, 16, 64, 64)` | 3D appearance features from avatar |
| Input | `kp_source` | `(5, 21, 3)` | Source keypoints |
| Input | `kp_driving` | `(5, 21, 3)` | Driving keypoints (post-stitch) |
| Output | `out` | `(5, 256, 64, 64)` | Warped feature map |

### 6.3 SPADE Decoder (Per-Frame, B=1)

Source: `src/avtr1_renderer/models/decoder.py`

Runs per-frame because the SPADE stack does not share work across batch elements.
Batched b>1 is actually ~2ms slower.

| Direction | Field | Shape | Description |
|-----------|-------|-------|-------------|
| Input | `feature` | `(1, 256, 64, 64)` | Warped features |
| Output | `output` | `(1, 3, 512, 512)` | RGB face image |

Output is clamped to [0, 1].

### 6.4 Putback + Matting

Source: `src/avtr1_renderer/components/putback.py`, `src/avtr1_renderer/components/matting.py`

For each decoded face frame:

1. **Affine warp** from 512x512 crop -> original portrait coordinates
   using precomputed `M_grid` via `F.affine_grid` + `F.grid_sample`:
   - `M_b = avatar.M_grid.unsqueeze(0).expand(N, -1, -1)`
   - `grid = F.affine_grid(M_b, [N, 3, H, W], align_corners=False)`
   - `face_warped = F.grid_sample(face_crops, grid, ...)`

2. **Feathered blend** with source portrait:
   `blended = torch.lerp(source, face_warped, mask)`
   - avatar.mask: (1, H, W) pre-warped pasteback mask
   - avatar.source: (3, H, W) grey-bg portrait

3. **MODNet matting:**
   - Downsample to (288, 512)
   - Normalize: (img / 255 - 0.5) / 0.5 -> [-1, 1]
   - Run MODNet engine -> alpha at (N, 1, 288, 512)
   - Upsample back to (N, 1, H, W)

4. **Background composite:**
   `final = torch.lerp(bg, blended, alpha)` -> (N, 3, H, W)

MODNet Engine I/O:

| Direction | Field | Shape | Description |
|-----------|-------|-------|-------------|
| Input | `input` | `(N, 3, 288, 512)` | Normalized image [-1, 1] |
| Output | `output` | `(N, 1, 288, 512)` | Alpha matte [0, 1] |

### 6.5 Pixel Format Packing

Source: `src/avtr1_renderer/components/pixel_format.py`, `src/avtr1_renderer/frame_sink.py`

Converts GPU float tensors to host uint8 bytes:

| Format | Layout | Bytes/pixel | Description |
|--------|--------|-------------|-------------|
| `yuv_i420` | Y plane (H, W) + U plane (H/4, W) + V plane (H/4, W) | 1.5 | Standard video, BT.601 limited range |
| `yuv_i420_stacked_alpha` | Y + AY + U + AU + V + AV vertically stacked | 3.0 | Alpha passthrough through H.264/VP9/AV1 |

Color conversion: BT.601 limited range (Y=[16,235], UV=[16,240]) via kornia.
Alpha is round-tripped through a precomputed 256-entry LUT (`alpha_to_luma_lut.npy`)
that maps uint8 alpha to the Y value ffmpeg produces, ensuring alpha survives the
encode/decode pipeline.

For a 720x1280 frame:
- `yuv_i420`: 1,382,400 bytes (1.5 bpp)
- `yuv_i420_stacked_alpha`: 2,764,800 bytes (3.0 bpp)

### 6.6 Streaming vs Batched Modes

The renderer supports two modes controlled by `RenderOptions.stream_frames`:

**Streaming (default):** Warp runs once on the full chunk, then decoder + putback +
matting + pack run per-frame. Each frame is yielded before the next starts. Lower
latency to first frame.

**Batched:** All frames decoded at once, then all putback + matting + pack in one
H2D copy. Lower latency to last frame, higher to first.

---

## 7. Autoregressive State Management

Source: `src/avtr1_renderer/avtr1_motion_generator.py:88`

### 7.1 AVTR1State

All cross-chunk context, lives entirely on CUDA:

```python
@dataclass(slots=True, frozen=True)
class AVTR1State:
    audio_prev_speech: Tensor    # (3 * 640,)     HuBERT 3-frame left context
    audio_prev_listen: Tensor    # (3 * 640,)     parallel
    audio_features: Tensor       # (1, 75, 2048)  75-frame HuBERT history
    past_cond: Tensor            # (1, 75, 42)    normalized motion history
    noise_shared: Tensor | None  # (1, 1, 42)     AR(1) noise carry, None on cold start
```

### 7.2 State Update Per Chunk

| Field | Update Rule |
|-------|-------------|
| `past_cond` | Append 5 new normalized frames, drop oldest 5 (rolling window of 75) |
| `audio_features` | Drop oldest 5, append 5 "current" HuBERT features. Future 5 are dropped -- they re-enter next call as "current" |
| `audio_prev_*` | Roll forward by `chunk_size * frame_len = 3200` samples |
| `noise_shared` | Last frame of the generated noise `(1, 1, 42)` |

### 7.3 Cold Start (Initial State)

```python
# Tile the avatar's normalized neutral motion across 75 frames
so3 = roma.rotmat_to_rotvec(avatar.kp_info.R)
so3_n = (so3 - offset_so3) / scale_so3
exp_n = (exp_in_R - offset_exp) / scale_exp
exp_lipsync_n = exp_n[:, lipsync_coords]
frame = cat([so3_n, exp_lipsync_n], dim=-1)  # (1, 42)
past_cond = frame.repeat(1, 75, 1)           # (1, 75, 42)

audio_prev_speech = zeros(3 * 640)
audio_prev_listen = zeros(3 * 640)
audio_features = zeros(1, 75, 2048)
noise_shared = None
```

### 7.4 State Serialization

Source: `src/avtr1_renderer/avtr1_motion_generator.py:88`

State is serialized via safetensors for cross-request persistence (HTTP API).

```python
def state_to_safetensors(state: AVTR1State) -> bytes
def state_from_safetensors(blob: bytes, *, device="cuda") -> AVTR1State
```

This is the only place state ever crosses the CUDA -> host boundary.

---

## 8. Runtime Engine Abstraction

Source: `src/avtr1_renderer/runtime/`

### 8.1 InferenceEngine Protocol

Source: `src/avtr1_renderer/runtime/inference_engine.py`

```python
class InferenceEngine[InputT, OutputT](Protocol):
    def __call__(self, inputs: InputT, out: OutputT | None = None) -> OutputT: ...
    def allocate_outputs(self, shapes: dict[str, tuple[int, ...]] | None = None) -> OutputT: ...
```

Input/Output are **dataclasses** whose field names match the underlying engine's
tensor names exactly. All tensors are CUDA `torch.Tensor`. No CPU tensors, numpy
arrays, or cross-device transfers.

### 8.2 Backends

| Backend | File | Use Case | Key Features |
|---------|------|----------|--------------|
| **TRTEngine** | `runtime/trt.py` | Production | Zero-copy CUDA, stream-ordered via `execute_async_v3`, FP16, dynamic shapes, custom plugin loading via dlopen |
| **OnnxRTEngine** | `runtime/onnxrt.py` | Dev/Validation | IOBinding zero-copy, CUDA EP stream fencing, auto shape resolution for static shapes |
| **Factory** | `runtime/loader.py` | Both | Picks by file extension: `.engine`/`.trt`/`.plan` -> TRT, `.onnx` -> ORT |

### 8.3 TRTEngine Details

- Validates that dataclass field names match engine tensor names exactly
- Dtype mapping: float32, float16, int8, int32, int64, bool, uint8, bfloat16
- `allocate_outputs()`: Creates fresh CUDA tensors from static engine shapes
- Hot path: Sets input/output data pointers directly, calls `execute_async_v3(stream_handle)`
- Runs on the caller's current CUDA stream, no synchronization

### 8.4 OnnxRTEngine Details

- Uses `IOBinding` to bind input/output tensors by `data_ptr()` (zero copy)
- Cross-stream detection: if caller's stream differs from EP stream, fences via
  CUDA events (device-side, no host blocking)
- Cannot auto-resolve dynamic output shapes (ORT lacks that API)

---

## 9. Complete Engine Inventory

### 9.1 Speech-to-Motion Engines

| Engine | Source Format | Backend | Batch | Precision | TRT Build Notes |
|--------|---------------|---------|-------|-----------|-----------------|
| `avtr1_encode` | TorchScript | TRT only | B=1 | FP16 | LayerNorm ops pinned to FP32 |
| `avtr1_decode` | TorchScript | TRT only | B=1 | FP16 | 4-pass CFG internal, weights as inputs |
| `hubert_lbs` | ONNX | TRT/ONNX | B=1..2 | FP16 | Dynamic sequence length, LayerNorm FP32 |

### 9.2 Renderer Engines

| Engine | Source Format | Backend | Batch | Precision | TRT Build Notes |
|--------|---------------|---------|-------|-----------|-----------------|
| `decoder` | ONNX (Ditto) | TRT/ONNX | B=1..5 | FP16 | SPADE Resize graph surgery, value_info wipe |
| `warp_network` | ONNX (Ditto) | TRT/ONNX | B=1..5 | FP16 | Dense motion Reshape/Concat surgery, grid_sample plugin |
| `stitch_network` | ONNX (Ditto) | TRT/ONNX | B=1..5 | FP16 | ScatterND -> Concat replacement |
| `modnet` | ONNX | TRT/ONNX | B=1..5 | FP16 | SE-block Reshape constants patched |

### 9.3 Avatar Registration Engines (ONNX only, B=1)

| Engine | Input Shape | Output Shape | Description |
|--------|-------------|--------------|-------------|
| `appearance_extractor` | (1, 3, 256, 256) | (1, 32, 16, 64, 64) | 3D appearance features |
| `motion_extractor` | (1, 3, 256, 256) | 7 tensors | Pitch/yaw/roll/t/exp/scale/kp |
| `insightface_det` | (1, 3, 512, 512) | 9 tensors (3 FPN levels) | Face detection |
| `landmark106` | (1, 3, 192, 192) | (1, 212) | 106-point landmarks |
| `landmark203` | (1, 3, 224, 224) | (1, 406) | 203-point landmarks |

### 9.4 HuBERT Engine Build Profile

| Parameter | Min | Opt | Max |
|-----------|-----|-----|-----|
| batch | 1 | 2 | 2 |
| sequence length | 3,240 | 8,400 | 12,960 |

Rationale: B=2 because speech + listen are stacked along batch dim. 8,400 =
(3+5+5 frames) * 640 samples/frame + 80 shift.

---

## 10. Live Streaming Architecture (`avaturn_live_streamer`)

### 10.1 Architecture Overview

The streamer is a **worklet-based async event-driven system** coordinated through
an `EventBus` and `StreamClocks`.

```
Browser <--WebRTC--> LocalRTC (aiortc) <--frames/audio--> LocalRTCWorklet
                            |                               ^
                            |  UserSpeechReceived           |  VideoFrameGenerated
                            v                               |
            RealtimeApiClient / CartesiaApiClient    RenderingWorklet
            (Conversation Engine)              (audio -> video via RendererClient)
                            |                               |
                            v                               v
                  SpeechScheduler -----------> RendererClient
                  (buffer, segment, time)    (HTTP to GPU worker)
                            |
                  DelayedEventWorklet
                  (timed playback events)
                            |
                  TimeoutWorklet
                  (session guard)
```

### 10.2 Stream Runner

Source: `src/avaturn_live_streamer/runner.py`

```python
def run_stream(*worklets: WorkletCallable):
    """Creates EventBus + StreamClocks, spawns all worklets as concurrent tasks."""
```

Each worklet receives a cloned bus (with incremented pending count), subscribes
to events, then signals ready. The bus prevents publishing until all workers are
ready.

### 10.3 EventBus

Source: `src/avaturn_live_streamer/event_bus.py`

- Type-based async pub/sub with worker synchronization
- `subscribe(*event_types, buffer_size)` -> async context manager yielding `Subscription`
- `publish(event)` blocks until all workers ready, routes by `type(event)`
- `publish_nowait(event)` sync variant for callbacks, can buffer pre-ready events
- Backpressure on full queues

### 10.4 StreamClocks

Source: `src/avaturn_live_streamer/clocks.py`

```python
class StreamClocks:
    def start(self)                    # Anchor the clock epoch
    @property now -> float             # Compensated wall time
    async def wakeup_at(time)          # Sleep until specific stream-time
    def measure_delay_after_deadline() # Context manager for latency
    async def wait_until(deadline)     # Context wrapping asyncio.timeout
```

Delay compensation (`_total_delay`) ensures cumulative lag does not cause drift.
Critical for smooth 25 FPS video delivery.

### 10.5 System Constants

Source: `src/avaturn_live_streamer/constant.py`

| Constant | Value |
|----------|-------|
| RENDERER_SPEECH_SAMPLE_RATE | 16,000 Hz |
| NATIVE_SPEECH_SAMPLE_RATE | 24,000 Hz |
| OPENAI_SPEECH_SAMPLE_RATE | 24,000 Hz |
| VIDEO_RESOLUTION | (720, 1280) portrait |
| VIDEO_FPS | 25 |
| FRAMES_PER_RENDER_CHUNK | 5 |
| FRAME_DURATION | 1/25 second (as Fraction) |
| MAX_STREAM_DURATION | 86,399 seconds (~24h) |

### 10.6 Event Taxonomy

Source: `src/avaturn_live_streamer/events.py`

**Internal events** (dataclasses, never cross transport):

| Event | Purpose |
|-------|---------|
| `StreamStarted` / `StreamEnded(duration_seconds)` | Lifecycle |
| `Frame(buffer, timestamp, audio, pixel_format)` | Raw frame data |
| `VideoFrameGenerated(frame)` | Published by rendering worklet |
| `UserSpeechReceived(buffer)` | Inbound user audio chunk |
| `UserSpeechStreamStart` / `UserSpeechStreamEnd` | User audio stream boundaries |
| `ParticipantJoined(participant_id)` / `ParticipantLeft` | RTC presence |
| `SegmentGenerationStarted(segment_id, metadata)` | TTS segment begins |
| `SegmentChunkGenerated(segment_id, buffer)` | TTS audio chunk |
| `SegmentGenerationCompleted(segment_id)` | TTS segment complete |
| `ScheduledEvent(emit_at, event)` | Clock-synchronized delayed emission |
| `Shutdown(reason)` | Graceful termination |
| `DiscardAvatarSpeechBuffer` | Interrupt signal |

**Shutdown reasons:** `"termination_command_received"` | `"user_absent_timeout"` |
`"max_duration_reached"` | `"agent_left"`

### 10.7 Worklets

Each worklet is an async coroutine `(EventBus, StreamClocks) -> None` launched in
a `TaskGroup`.

#### RenderingWorklet

Source: `src/avaturn_live_streamer/worklets/rendering.py`

The **heart of the streaming pipeline**. Bridges speech segments to video frames.

1. Creates two `SpeechScheduler` instances (avatar speech + user speech for listen)
2. Main loop per chunk:
   - `speech_scheduler.do_step()` -> present + future audio chunks
   - Emits `ScheduledEvent` wrappers for playback timing
   - Calls `RendererClient.generate()` -> frame stream
   - Publishes `VideoFrameGenerated` per frame with sliced audio
   - `clocks.wakeup_at()` for 25 FPS pacing

#### SpeechScheduler

Source: `src/avaturn_live_streamer/speech/speech_scheduler.py`

The **audio timing brain**. Manages buffering, segmentation, and timing.

| Parameter | Value |
|-----------|-------|
| present_duration | 5 frames = 200ms |
| future_duration | 5 frames + 80 samples |

Core algorithm (`do_step()`):
1. Calculate needed duration (present + future - buffered)
2. Consume from queue until enough or segment boundary
3. Block if mid-segment with insufficient data
4. Pad with silence at segment start/end
5. Split buffer into present + future
6. Emit timestamped segment lifecycle events

#### SpeechBuffer

Source: `src/avaturn_live_streamer/speech/speech_buffer.py`

Immutable PCM audio buffer:
- `soxr` QQ-quality resampling (16kHz <-> 24kHz)
- `Fraction`-based exact duration arithmetic
- `split_in_two_parts_with_duration()` for present/future splitting
- `silence()` constructor for zero-padding

#### LocalRTCWorklet

Source: `src/avaturn_live_streamer/localrtc/worklet.py`

WebRTC bridge with three sub-workers:
- **Video -> Peer:** Subscribes to `VideoFrameGenerated`, converts raw bytes to
  `av.VideoFrame`, pushes to peer. Hybrid PTS (max of nominal 25fps vs wall-clock).
- **Participant Lifecycle:** Monitors connection state, publishes Joined/Left events.
- **Audio <- Peer:** Reads inbound audio (20ms chunks), publishes `UserSpeechReceived`
  with stream start/end detection (1.5x chunk timeout).

#### DelayedEventWorklet

Source: `src/avaturn_live_streamer/worklets/delayed_event.py`

Clock-synchronized event emission. Waits until `emit_at` via `clocks.wakeup_at()`
then publishes the wrapped event.

#### TimeoutWorklet

Source: `src/avaturn_live_streamer/worklets/timeout.py`

Session guard:
- `user_absent_timeout` (default 60s) -- starts when no participants
- `max_duration` (default 3600s) -- hard session limit
- Cancels/restarts timers on participant join/leave

### 10.8 Renderer Client Layer

Source: `src/avaturn_live_streamer/renderer/`

#### Abstract Interface

```python
class AbstractRendererClient(ABC):
    def generate(self, request: RendererRequest) -> AbstractAsyncContextManager[RenderResponse]
```

```python
@dataclass
class RendererRequest:
    current_chunk: bytes           # Present speech audio (int16 PCM, 16kHz)
    future_chunk: bytes            # Future speech audio
    current_chunk_listen: bytes    # Present listen audio
    future_chunk_listen: bytes     # Future listen audio
    timestamp_global: float
    state: bytes | None            # Safetensors state blob
    config: RenderConfig           # avatar_id, bg_id, pixel_format, params
```

```python
@dataclass
class RenderResponse:
    frame_generator: AsyncGenerator  # Yields (frame_bytes, audio_chunk) tuples
    num_frames: int
    state: Awaitable[bytes]          # New state blob
```

#### RendererClient (HTTP)

Source: `src/avaturn_live_streamer/renderer/client.py`

- Uses **sync httpx on a worker thread** for receiving (avoids event loop contention
  with ~120 TCP chunks per call)
- POSTs to `/process-audio-v3` endpoint
- Streams response: first N bytes = state (safetensors), remaining = rendered frames
- Headers: `X-Frame-Length-Bytes`, `X-Num-Frames`, `X-State-Length-Bytes`
- Retry with exponential backoff (30s deadline for acquire, 60s for rendering)
- Load-balanced or direct instance mode
- OpenTelemetry metrics: TTFF, MTBF, call duration, state size

#### MockRendererClient

Source: `src/avaturn_live_streamer/renderer/mock.py`

Test/development double. Generates synthetic YUV frames using OpenCV with sinusoidal
red channel animation. No GPU required.

#### Renderer Registry

Source: `src/avaturn_live_streamer/renderer/registry.py`

Maps model names to client instances. Environment key normalization:
`avtrn_1` -> `avtrn-1` for Kubernetes compatibility (env vars cannot contain hyphens).

### 10.9 WebRTC/NAT Traversal

Source: `src/avaturn_live_streamer/localrtc/ice.py`

Three-tier ICE resolution:

1. **Manual TURN:** `TURN_URL` + `TURN_USERNAME` + `TURN_CREDENTIAL` env vars
2. **Cloudflare Realtime TURN:** `CLOUDFLARE_TURN_KEY_ID` + `CLOUDFLARE_TURN_KEY_TOKEN`
   (fetches short-lived per-session credentials)
3. **STUN-only fallback:** `stun:stun.l.google.com:19302`

### 10.10 Local Stream CLI

Source: `src/avaturn_live_streamer/local_stream_cli.py`

FastAPI app for local development:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `GET /` | GET | Serves `local_stream_ui.html` |
| `GET /avatars` | GET | Lists available avatars/backgrounds |
| `GET /ice-servers` | GET | ICE/STUN/TURN config for browser |
| `POST /probe-offer` | POST | ICE connectivity probe (15s lifetime) |
| `POST /offer` | POST | Full SDP negotiation, engine build, session start |

Enforces one-at-a-time session constraint via `_SessionSlot`.

---

## 11. Conversation Engines

Source: `src/avaturn_live_streamer/conversation_engines/`

### 11.1 OpenAI Realtime API Client

Source: `conversation_engines/realtime_api_client.py`

- WebSocket to `wss://api.openai.com/v1/realtime`
- Full-duplex audio: sends user PCM (base64 encoded), receives TTS PCM (base64 decoded)
- Handles STT, LLM inference, and TTS output via single WebSocket
- Supports interruption via `input_audio_buffer.speech_started` -> `DiscardAvatarSpeechBuffer`
- Built-in system prompt for demo avatar behavior
- Minted ephemeral client secret (2h TTL) with semantic VAD config

Event flow:
- `response.output_audio.delta` -> `SegmentChunkGenerated`
- `response.output_audio.done` -> `SegmentGenerationCompleted`
- `input_audio_buffer.speech_started` -> `DiscardAvatarSpeechBuffer`
- `conversation.item.input_audio_transcription.completed` -> `InputTranscript`
- `response.output_audio_transcript.done` -> `ResponseTranscript`

### 11.2 Cartesia Client

Source: `conversation_engines/cartesia_client.py`

- WebSocket to `wss://api.cartesia.ai/agents/stream/{agent_id}` with Bearer token
- LLM+TTS managed by Cartesia's platform
- Custom segmentation via drain timer (audio_seconds + grace period)
- Handles `clear` events as interruptions

### 11.3 Builder Layer

Source: `conversation_engines/builders.py`

Factory that handles credential minting:

```python
mint_openai_realtime_secret(api_key, model, prompt, voice, tracing) -> str
# Ephemeral 2h TTL client secret with session config

_mint_cartesia_token(api_key) -> str
# Short-lived access token
```

Engine options:

```python
@dataclass
class OpenAIEngineOptions:
    api_key: str
    model: str
    prompt: str
    voice: OpenaiRealtimeApiVoice  # alloy|ash|ballad|cedar|coral|echo|marin|sage|shimmer|verse

@dataclass
class CartesiaEngineOptions:
    api_key: str
    agent_id: str
```

---

## 12. Renderer HTTP API

Source: `src/avtr1_renderer/api/app.py`

### 12.1 Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/process-audio-v3` | POST | Main streaming audio-to-video endpoint |
| `/avatars` | GET | Lists loaded avatar IDs + background IDs |
| `/health` | GET | CUDA health check |

### 12.2 `/process-audio-v3` Protocol

**Request (multipart form):**

| Field | Type | Description |
|-------|------|-------------|
| `current_chunk` | binary | Raw int16 PCM at 16kHz (speech track, present) |
| `future_chunk` | binary | Raw int16 PCM at 16kHz (speech track, future) |
| `current_chunk_listen` | binary | Raw int16 PCM at 16kHz (listen track, present) |
| `future_chunk_listen` | binary | Raw int16 PCM at 16kHz (listen track, future) |
| `state` | binary (optional) | Safetensors state blob from previous call |
| `avatar_id` | string | Avatar identifier |
| `bg_id` | string | Background identifier |
| `pixel_format` | string | "yuv_i420" or "yuv_i420_stacked_alpha" |
| `cfg_self_audio` | float | CFG weight for self audio (default 2.0) |
| `cfg_other_audio` | float | CFG weight for other audio (default 2.0) |
| `cfg_kp` | float | CFG weight for identity/kp (default 3.0) |
| `noise_alpha` | float | AR(1) noise parameter (default 2.0) |
| `noise_trunc_z` | float | Truncation parameter (default 1.2) |

**Response (streaming body):**

1. First: state safetensors blob (variable length, see header)
2. Then: concatenated rendered frames (each X-Frame-Length-Bytes bytes)

**Response headers:**

| Header | Description |
|--------|-------------|
| `X-State-Length-Bytes` | Length of the state blob at the start of the body |
| `X-Num-Frames` | Number of rendered frames (typically 5) |
| `X-Frame-Height` | Frame height (720) |
| `X-Frame-Width` | Frame width (1280) |
| `X-Frame-Length-Bytes` | Bytes per individual frame |

### 12.3 Lifespan

On startup:
1. Auto-discovers avatar PNGs in `reference_frames` artifact directory
2. Calls `Pipeline.from_artifacts()` to load engines + avatars + backgrounds
3. Computes `current_samples` and `future_samples` from motion generator params
4. Starts the load-balancer keep-alive worker

### 12.4 Load Balancing

Source: `src/avtr1_renderer/api/load_balancing.py`

Workers self-register with an external load balancer via periodic keep-alive POSTs
(0.5-1.0s interval). WorkerId = hostname. On shutdown, sends a `kill` message.

```
LOAD_BALANCER_URL/worker/keep_alive?worker_id={id}&worker_url={url}
LOAD_BALANCER_URL/worker/kill?worker_id={id}
```

When `LOAD_BALANCER_URL=disabled` (local setup), keep-alive is skipped.

---

## 13. Artifact Management

### 13.1 HuggingFace Sources

| Repo ID | Contents |
|---------|----------|
| `avaturn-live/avtr-1` | AVTR1 TorchScript, HuBERT ONNX, MODNet ONNX, avatar references, backgrounds, grid_sample plugin |
| `digital-avatar/ditto-talkinghead` | LivePortrait ONNX models (appearance extractor, motion extractor, warp, decoder, stitch, landmarks) |

### 13.2 Local Storage Layout

```
$AVTR1_LOCAL_STORAGE/v1/                           (or <project_root>/artifacts/v1/)
|
+-- renderer_runtime_artifacts/                     ONNX models (runtime, from HF)
|   +-- appearance_extractor.onnx
|   +-- motion_extractor.onnx
|   +-- landmark106.onnx
|   +-- landmark203.onnx
|   +-- insightface_det.onnx
|   +-- blaze_face.onnx
|   +-- face_mesh.onnx
|   +-- libgrid_sample_3d_plugin.so                Custom TRT plugin (local only)
|
+-- renderer_runtime_artifacts_cc/                  TRT engines (locally built)
|   +-- decoder_b5_fp16.engine
|   +-- warp_network_b5_fp16.engine
|   +-- stitch_network_b5_fp16.engine
|   +-- modnet_b5_fp16.engine
|
+-- speech2motion_runtime_artifacts/                ONNX sources for AVTR1
|   +-- hubert-lbs-avtr1.onnx
|
+-- speech2motion_runtime_artifacts_cc/             TRT engines (locally built)
|   +-- avtr1_encode_fp16.engine
|   +-- avtr1_decode_fp16.engine
|   +-- hubert_lbs_fp16.engine
|   +-- avtr1_normalizer.safetensors
|
+-- avatars_artifacts/
|   +-- reference_frames/                           Portrait PNGs ({id}.png)
|   |   +-- maria.png
|   |   +-- elena.png
|   |   +-- marcus.png
|   |   +-- ...
|   +-- backgrounds/                                Background PNGs
|   |   +-- minimal_office.png
|   |   +-- ...
|   +-- pasteback_mask.png                          Compositing mask template
|
+-- build_artifacts/                                Source models from HF
|   +-- avtr1.scripted.pt                           AVTR1 TorchScript checkpoint
|   +-- hubert-lbs-avtr1.onnx                       Fine-tuned HuBERT encoder
|   +-- modnet.onnx                                  MODNet portrait matting
|
+-- intro_motions/                                  Pre-recorded intro PKL files
    +-- 09_27_v15.pkl                               "with smile" variant
    +-- 09_27_v23.pkl                               "without smile" variant
```

### 13.3 Artifact Manager

Source: `src/avtr1_renderer/avtr1_artifact_manager.py`, `src/avtr1_renderer/artifact_manage_basic.py`

- `get_storage_root()` -> `{AVTR1_LOCAL_STORAGE}/{HF_REVISION}` or `<project_root>/artifacts/{HF_REVISION}`
- `get_artifact_manager()` -> Singleton cached by (config, storage_root)
- `get_trt_engine_path(name)` -> Maps logical name to filesystem path
- `find_engine_or_onnx(trt_name, onnx_name)` -> Returns TRT if built, else ONNX fallback
- `ensure_all_artifacts(workers=4)` -> Pre-fetches all artifacts in parallel

---

## 14. Intro Motion System

Source: `src/avtr1_renderer/intro_motion.py`

Pre-recorded animations played before audio-driven animation begins.

### 14.1 Recordings

| File | Variant | Used For |
|------|---------|----------|
| `09_27_v15.pkl` | "with smile" | Women / avatars starting with "anya" |
| `09_27_v23.pkl` | "without smile" | Everyone else |

Selection: `_select_pkl_for_avatar(avatar_id)` -- "woman" (case-insensitive) or
starts with "anya" -> "with smile"; otherwise "without smile".

### 14.2 Retargeting

```python
def retarget(window, *, anchor, kp_info) -> MotionFrame:
    R_out = window.R
    exp_out = window.exp - anchor.exp + source.exp[lipsync]
```

### 14.3 Normalization

Converts intro motion to normalized `past_cond` rows for state initialization:
- Splice lipsync exp into full 63-d expression
- Rotate by R, re-extract lipsync coords
- Z-score normalize with the same Normalizer used at runtime

---

## 15. Dyadic Audio Architecture (The Key Innovation)

AVTR-1's distinguishing feature is **native duplex conditioning** -- it processes
both sides of the conversation simultaneously.

```
+-------------------------------------------------------------+
|                   AVTR-1 Dual-Stream Audio                  |
|                                                              |
|  Speaker Audio (drives lip sync) ---+                        |
|                                      |                       |
|                                      +--> HuBERT (B=2)       |
|                                      |      |                |
|  Listener Audio (drives nods,       |      +--> audio_self  |
|  brows, gaze) ----------------------+      +--> audio_other |
|                                                             |
|  When avatar is SPEAKING:                                    |
|    audio_self dominates -> lip articulation                  |
|  When avatar is LISTENING:                                   |
|    audio_other dominates -> nodding, brow movement, gaze     |
|                                                             |
|  The split is LEARNED via gated fusion inside the           |
|  transformer, not hand-engineered.                          |
+-------------------------------------------------------------+
```

Both audio streams are always active. The model learns when to attend to each
via gated fusion inside the transformer. This produces convincing "active
listening" behavior -- the avatar nods, raises brows, and shifts gaze in
response to the partner's voice in real-time.

### 15.1 Audio Processing Path

```
For each chunk:
  1. speech_audio = [prev_tail(3 frames) | chunk(5 frames) | future(5 frames)] + shift
  2. listen_audio = [prev_tail(3 frames) | chunk(5 frames) | future(5 frames)] + shift
  3. batch = stack([speech_audio, listen_audio]) -> (2, 6480)
  4. features = HuBERT(batch) -> (2, 10, 1024)
  5. new_feats = cat([features[0], features[1]], dim=-1) -> (1, 10, 2048)
  6. audio_cond = cat([history(75 frames), new_feats(10)], dim=1) -> (1, 85, 2048)
```

---

## 16. Build and Deploy Pipeline

### 16.1 Pixi Environments

| Environment | Features | Purpose |
|-------------|----------|---------|
| `renderer` (default) | dev + build | Heavy deps: torch, tensorrt, kornia, CUDA |
| `streamer` | streamer only | Lightweight: fastapi, aiortc, websockets, openai |

### 16.2 Build Steps

```
pixi install
  |
  +-- pixi run download (scripts/download_artifacts.py)
  |     Downloads all HF artifacts to local storage
  |
  +-- pixi run build-trt-engines (scripts/build_engines.py)
  |     +-- build_renderer_engines.py -> decoder, warp, stitch, modnet (b5 fp16)
  |     +-- build_avtr1_engines.py   -> encode, decode, normalizer (fp16)
  |     +-- build_hubert_engine.py   -> hubert (b2 fp16)
  |
  +-- pixi run generate_offline (scripts/generate_offline.py)
  |     Single-pass offline video generation from audio files
  |
  +-- pixi run interactive-demo (scripts/run_local_stream.py)
        +-- Spawns renderer subprocess (uvicorn)
        +-- Waits for health check (polls GET /health every 1s, timeout 300s)
        +-- Spawns streamer subprocess (FastAPI + WebRTC)
```

### 16.3 TRT Engine Build Details

Source: `scripts/build_avtr1_engines.py`, `scripts/build_renderer_engines.py`,
`scripts/build_hubert_engine.py`

All builds use:
- FP16 with precision constraints preferred
- LayerNorm-flavoured ops (Pow->ReduceMean patterns, "Norm" in name) pinned to FP32
- ONNX opset 17
- Optional AMPERE_PLUS hardware compatibility

Renderer engine graph surgeries:

| Engine | Surgery |
|--------|---------|
| decoder | Patches 4 SPADE Resize nodes with baked batch=1 constants. Wipes 194 value_info entries pinning batch=1 |
| warp_network | Rewrites ~12 Reshape/Concat constants in dense_motion_network for dynamic batch. Requires grid_sample_3d plugin |
| stitch_network | Replaces 2 redundant ScatterND nodes (baked batch=0 indices) with single Concat |
| modnet | Patches 2 SE-block Reshape constants from [1,...] to [B,...] |

### 16.4 Offline Generation

Source: `scripts/generate_offline.py`

```bash
pixi run generate_offline --speech speaker_1.ogg                    # Single speaker
pixi run generate_offline --speech s1.ogg --listen s2.ogg           # Dialogue
pixi run generate_offline --duration 10                             # Idle motion
```

CLI args: `--speech`, `--listen`, `--duration`, `--avatar` (default "maria"),
`--out` (default "demo_output.mp4"), `--bg`, `--no-mux`, `--stream-frames/--no-stream-frames`.

Output: H.264 video (imageio_ffmpeg) at 25fps with optional AAC audio muxed
from the speech track.

Audio preparation: mono 16kHz float32 via soundfile + soxr resampling. Tracks
are padded/trimmed to equal length; missing track defaults to silence.

Chunk slicing:
- window = `(chunk_size + future_size) * frame_len + audio_shift`
- step = `chunk_size * frame_len`

---

## 17. End-to-End Timing Budget

Per 5-frame chunk on an **A100** (91ms total, 2.2x real-time):

| Stage | Approx. Time | Notes |
|-------|-------------|-------|
| Audio upload (H2D) | ~1ms | 6,480 samples x 2 tracks |
| HuBERT inference | ~10ms | B=2, 8480 samples input |
| AVTR1 Encode | ~5ms | Single forward pass |
| AVTR1 Decode x 5 steps | ~25ms | 5 Euler steps with internal CFG |
| Noise sampling | ~1ms | AR(1) truncated normal |
| Warp (B=5) | ~8ms | Batched, once per chunk |
| Decoder x 5 frames | ~25ms | Per-frame SPADE decode |
| Putback + MODNet x 5 | ~12ms | Warp back, matte, composite |
| Pixel format pack + H2D | ~4ms | YUV conversion, host copy |
| **Total** | **~91ms** | **2.2x real-time** |

At 25 FPS, each chunk produces 200ms of video, so any GPU under 200ms per chunk
runs in real-time.

---

## 18. Key Design Decisions and Optimizations

### 18.1 Warp Batched, Decoder Per-Frame

Warp shares work across the chunk (the win is real). The SPADE stack does not
share work across batch elements -- b>1 is actually ~2ms slower. So warp runs
at B=5, decoder runs at B=1.

### 18.2 Zero-Copy CUDA Pipeline

State stays on GPU between chunks. No D2H syncs in the hot path. Only `pack_frames`
does H2D (once per chunk for all 5 frames, batched into a single `.cpu().numpy()`).

### 18.3 Pre-Computed M_grid

Caches the inverse affine for putback at avatar load time. Avoids 40-kernel LU
factorization + 2 D2H sync points per frame (from kornia.warp_affine's
cuBLAS singularity check).

### 18.4 Pre-Allocated Output Buffers

Both encode and decode engines use `allocate_outputs()` + `out=` parameter to
reuse GPU buffers across ODE steps. Saves ~3 allocations per step.

### 18.5 GPU-Side t Schedule

`torch.linspace` on GPU creates the full timestep schedule as a contiguous tensor.
`dt` is a constant host scalar. This avoids ~9 CUDA->host syncs per chunk that
the previous code incurred via `t_buf.fill_(float(time[i]))` and
`(time[i+1] - time[i]).item()`.

### 18.6 CFG as Runtime Inputs

The 4-pass classifier-free guidance batch is internal to the TRT engine, but
the three weight tensors (w_self, w_other, w_kp) are engine inputs. No engine
rebuild needed to retune guidance strength.

### 18.7 Sync HTTP on Worker Thread

The renderer client receives ~120 TCP chunks per call. Using sync httpx on a
dedicated thread avoids event loop contention with frame delivery to WebRTC.

### 18.8 Separate Pixi Environments

Renderer (heavy: torch, tensorrt, CUDA) and streamer (lightweight: fastapi,
aiortc) have independent dependency resolution via pixi solve-groups.

### 18.9 HuBERT 50->25 Hz Downsampling

HuBERT produces features at 50 Hz. Pairwise mean (reshape to pairs, average dim)
produces 25 Hz features that match the motion frame rate. This is done in a single
vectorized operation.

### 18.10 State Never Touches Host

All state tensors live on CUDA. The `state_to_safetensors` / `state_from_safetensors`
functions are the only place state ever crosses the device boundary, used only
for HTTP API persistence.

---

## 19. Normalizer and Z-Score Convention

Source: `src/avtr1_renderer/avtr1_motion_generator.py:219`

The model's input projection is trained on z-score-normalized motion vectors.
The normalizer holds separate stats for each motion component:

```
motion_offset = [so3(3) | exp_lipsync(39)]  -> (42,) combined
motion_scale  = [so3(3) | exp_lipsync(39)]  -> (42,) combined

Normalization:  x_norm = (x - offset) / scale
Denormalization: x = x_norm * scale + offset
```

Separate full tensors are also stored for kp and full exp:

```
offset_so3: (3,)          scale_so3: (3,)
offset_kp:  (21, 3)       scale_kp:  (21, 3)
offset_exp: (21, 3)       scale_exp: (21, 3)
```

The normalizer is loaded from `avtr1_normalizer.safetensors`, which is written
during the TRT engine build process from the TorchScript checkpoint's
normalization buffers.

---

## 20. Configuration Reference

### 20.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AVTR1_LOCAL_STORAGE` | `<project_root>/artifacts/` | Root directory for all downloaded/built artifacts |
| `LOAD_BALANCER_URL` | (none) | Load balancer URL for renderer workers. "disabled" for local |
| `OPENAI__API_KEY` | (none) | OpenAI API key for Realtime API |
| `CARTESIA_API_KEY` | (none) | Cartesia API key |
| `CARTESIA_AGENT_ID` | (none) | Cartesia agent ID |
| `CLOUDFLARE_TURN_KEY_ID` | (none) | Cloudflare TURN key ID |
| `CLOUDFLARE_TURN_KEY_TOKEN` | (none) | Cloudflare TURN API token |
| `TURN_URL` | (none) | Manual TURN server URL (takes precedence over Cloudflare) |
| `TURN_USERNAME` | (none) | Manual TURN username |
| `TURN_CREDENTIAL` | (none) | Manual TURN credential |
| `STUN_URL` | `stun:stun.l.google.com:19302` | STUN server override |
| `STREAMER_HOST` | `127.0.0.1` | Streamer bind host |
| `STREAMER_PORT` | `7860` | Streamer bind port |
| `RENDERER_PORT` | `8000` | Renderer bind port |

### 20.2 RenderOptions Defaults

Source: `src/avtr1_renderer/types.py:113`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pixel_format` | `"yuv_i420"` | Output frame layout |
| `bg_id` | (required) | Background identifier |
| `cfg_self_audio` | 2.0 | CFG weight for speaker audio |
| `cfg_other_audio` | 2.0 | CFG weight for listener audio |
| `cfg_kp` | 3.0 | CFG weight for identity keypoints |
| `noise_alpha` | 2.0 | AR(1) noise correlation strength |
| `noise_trunc_z` | 1.2 | Truncated normal clipping range |
| `stream_frames` | True | Stream frames one-by-one (vs batched) |

### 20.3 AVTR1MotionGenerator Default Parameters

Source: `src/avtr1_renderer/avtr1_motion_generator.py:287`

| Parameter | Default |
|-----------|---------|
| `chunk_size` | 5 |
| `past_size` | 75 |
| `future_size` | 5 |
| `nfeats` | 42 |
| `latent_dim` | 512 |
| `frame_len` | 640 |
| `audio_shift` | 80 |
| `n_ode_steps` | 5 |

### 20.4 Pixi Tasks

| Task | Command | Description |
|------|---------|-------------|
| `download` | `python scripts/download_artifacts.py` | Download all model artifacts from HuggingFace |
| `generate_offline` | `python scripts/generate_offline.py` | Run end-to-end offline generation |
| `build-trt-engines` | `python scripts/build_engines.py` | Build all TRT engines |
| `build-trt-engines-avtr1` | `python scripts/build_avtr1_engines.py` | Build AVTR-1 TRT engines |
| `build-trt-engines-renderer` | `python scripts/build_renderer_engines.py` | Build renderer TRT engines |
| `build-trt-engines-hubert` | `python scripts/build_hubert_engine.py` | Build HuBERT TRT engine |
| `interactive-demo` | `python scripts/run_local_stream.py` | Run interactive WebRTC demo |
| `test` | `pytest` | Run tests |
| `lint` | `ruff check` | Lint with ruff |
| `format` | `ruff format` | Format with ruff |
| `typecheck` | `basedpyright` | Type-check |

---

## File Index

### Renderer (`src/avtr1_renderer/`)

| File | Purpose |
|------|---------|
| `pipeline.py` | Top-level Pipeline orchestrator, from_artifacts factory |
| `renderer.py` | Motion-to-pixel rendering (render_chunk, render_chunk_streaming) |
| `avtr1_motion_generator.py` | AVTR1MotionGenerator, AVTR1State, Normalizer, noise sampling |
| `motion_generator.py` | MotionGenerator Protocol (swappable interface) |
| `avatar_loader.py` | Avatar + AvatarLoader (registration cascade) |
| `frame_sink.py` | Pack GPU RGB(A) into host Frames |
| `types.py` | Chunk, Frame, FrameIterator, KPInfo, RenderOptions |
| `constants.py` | LIPSYNC_COORDS (39 predicted expression indices) |
| `backgrounds.py` | Load background images as CUDA tensors |
| `intro_motion.py` | Pre-recorded intro animations |
| `__init__.py` | Package init |
| `avtr1_artifact_manager.py` | Artifact path resolution for AVTR-1 specific artifacts |
| `artifact_manage_basic.py` | Generic artifact download from HuggingFace |
| `artifact_configs/v1.py` | Version 1 artifact paths and HF repo config |
| `artifact_configs/__init__.py` | Latest config version selector |
| `api/app.py` | FastAPI HTTP server for renderer |
| `api/load_balancing.py` | Load balancer keep-alive client |
| `api/__init__.py` | Package init |
| `runtime/inference_engine.py` | InferenceEngine Protocol definition |
| `runtime/trt.py` | TensorRT engine implementation |
| `runtime/onnxrt.py` | ONNX Runtime engine implementation |
| `runtime/loader.py` | Engine factory (path -> TRT or ORT) |
| `runtime/__init__.py` | Public re-exports |
| `models/avtr1.py` | AVTR1 encode/decode I/O contracts |
| `models/hubert.py` | HuBERT engine I/O contract |
| `models/decoder.py` | SPADE decoder engine I/O contract |
| `models/warp.py` | Warp network engine I/O contract |
| `models/stitch.py` | Stitch network engine I/O contract |
| `models/matting.py` | MODNet matting engine I/O contract |
| `models/appearance_extractor.py` | Appearance extractor engine I/O contract |
| `models/motion_extractor.py` | Motion extractor engine I/O contract |
| `models/face_detection.py` | Face detection engine I/O contract |
| `models/landmark106.py` | Landmark106 engine I/O contract |
| `models/landmark203.py` | Landmark203 engine I/O contract |
| `models/__init__.py` | Package init |
| `components/hubert.py` | HuBERT feature extraction + 50->25 Hz downsampling |
| `components/matting.py` | MODNet matting wrapper (resize + normalize + engine) |
| `components/putback.py` | Face crop -> full-frame composite (warp + blend + matte + bg) |
| `components/source_crop.py` | Aligned face cropping + similarity transform |
| `components/pixel_format.py` | RGB(A) -> YUV I420 packing (BT.601) |
| `components/face_detection.py` | SCRFD face detection + NMS |
| `components/face_landmarks.py` | landmark106 + landmark203 prediction |
| `components/liveportrait/motion_stitch.py` | MotionFrame + motion stitching |
| `components/liveportrait/motion_extractor.py` | Motion extraction + post-processing |
| `components/liveportrait/__init__.py` | Package init |
| `components/__init__.py` | Package init |
| `utils/cuda_health.py` | CUDA health check utility |
| `utils/asyncio.py` | Async utilities |
| `utils/__init__.py` | Package init |

### Streamer (`src/avaturn_live_streamer/`)

| File | Purpose |
|------|---------|
| `runner.py` | Top-level stream orchestrator (run_stream) |
| `settings.py` | Pydantic-settings configuration singleton |
| `config.py` | Per-session stream configuration models |
| `types.py` | Type aliases (RendererAvatarId, BackgroundId, PixelFormat) |
| `events.py` | Complete event taxonomy (internal + transport) |
| `sdk_events.py` | SDK-level avatar speech lifecycle events |
| `event_bus.py` | Type-based async event bus with worker synchronization |
| `clocks.py` | Wall-clock timing infrastructure |
| `constant.py` | System-wide constants (sample rates, FPS, resolution) |
| `local_stream_cli.py` | Local development FastAPI CLI |
| `localrtc/worklet.py` | WebRTC worklet (video/audio bridge) |
| `localrtc/peer.py` | aiortc peer connection wrapper |
| `localrtc/ice.py` | ICE/STUN/TURN server resolution |
| `localrtc/__init__.py` | Package init |
| `worklets/rendering.py` | Core rendering worklet |
| `worklets/delayed_event.py` | Clock-synchronized delayed event emission |
| `worklets/timeout.py` | Session timeout management |
| `worklets/__init__.py` | Package init |
| `renderer/interface.py` | Abstract renderer client interface |
| `renderer/client.py` | HTTP renderer client with load balancing |
| `renderer/registry.py` | Renderer client registry |
| `renderer/models.py` | Model-specific config and defaults |
| `renderer/config.py` | Renderer client factory |
| `renderer/mock.py` | Mock renderer for testing |
| `renderer/__init__.py` | Package init |
| `conversation_engines/realtime_api_client.py` | OpenAI Realtime API client |
| `conversation_engines/cartesia_client.py` | Cartesia Calls API client |
| `conversation_engines/configs.py` | Engine configuration models |
| `conversation_engines/builders.py` | Engine factory with credential minting |
| `conversation_engines/__init__.py` | Package init |
| `speech/speech_scheduler.py` | Audio scheduling engine |
| `speech/speech_buffer.py` | Immutable PCM audio buffer |
| `speech/duration.py` | Duration type alias |
| `speech/__init__.py` | Package init |
| `management/types.py` | Stream/segment ID types and generators |
| `management/__init__.py` | Package init |
| `core/logs.py` | Logging utilities |
| `core/logging_config.py` | Logging configuration |
| `core/__init__.py` | Package init |
| `utils/datetime.py` | DateTime utilities |
| `utils/exceptions.py` | Custom exceptions |
| `utils/async_utils.py` | Async utility functions |
| `utils/__init__.py` | Package init |
| `__init__.py` | Package init |

### Scripts (`scripts/`)

| File | Purpose |
|------|---------|
| `download_artifacts.py` | Download all HuggingFace artifacts |
| `build_engines.py` | Build all TRT engines (orchestrator) |
| `build_avtr1_engines.py` | Build AVTR1 encode + decode TRT engines + normalizer |
| `build_renderer_engines.py` | Build renderer TRT engines (decoder, warp, stitch, modnet) |
| `build_hubert_engine.py` | Build HuBERT TRT engine |
| `generate_offline.py` | Offline video generation from audio files |
| `run_local_stream.py` | Local streaming demo (renderer + streamer subprocess manager) |
| `test_api.py` | API test script |
