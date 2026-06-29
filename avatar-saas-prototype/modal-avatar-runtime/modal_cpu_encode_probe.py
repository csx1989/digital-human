"""Standalone CPU encode probe for aiortc/PyAV.

Measures the CPU encoding cost of the SFU publisher's software encoder
(VP8 via libvpx default, or H264 via libx264 when forced) in isolation,
without GPU inference or network SFU.

Approach: uses PyAV's Codec API directly to encode synthetic I420 frames
at 720p/25fps for a fixed duration, measuring CPU utilization and encode
throughput. This is a MORE DIRECT measurement than an aiortc loopback
because it isolates the encoder itself without RTP/SRTP/network overhead.

Usage:
    modal run modal_cpu_encode_probe.py --codec vp8 --duration-s 30
    modal run modal_cpu_encode_probe.py --codec h264 --duration-s 30
    modal run modal_cpu_encode_probe.py --codec capabilities

Key output:
    - encode_fps: frames the encoder actually produced per second
    - cpu_percent_mean / max: process CPU utilization (100% = 1 core)
    - cpu_cores_used: cpu_percent / 100 (how many cores consumed)
    - bytes_per_frame: encoded size (compression ratio indicator)
    - codec_capabilities: which codecs PyAV wheel actually supports
    - nvenc_available: whether the PyAV wheel includes NVENC

This probe runs on a CPU-only container (no GPU) to isolate the encoder
cost from CUDA inference. The result quantifies the architectural delta
between the fMP4 path (h264_nvenc, ~0% CPU) and the SFU path (libvpx /
libx264, this probe's CPU%).
"""

from __future__ import annotations

import fractions
import time
import threading

import modal

IMAGE = (
    modal.Image.debian_slim()
    .pip_install("av>=14,<15", "numpy", "psutil", extra_options="--only-binary av")
)

app = modal.App("avtr1-cpu-encode-probe")


@app.function(image=IMAGE, cpu=4, memory=4096, timeout=120)
def probe_encode(
    codec: str = "vp8",
    duration_s: int = 30,
    width: int = 1280,
    height: int = 720,
    fps: int = 25,
) -> dict:
    """Measure CPU encoding throughput and CPU cost for a given PyAV codec.

    Creates a PyAV encoder directly (no WebRTC), feeds it shifting-gradient
    I420 frames at the target resolution/fps for duration_s seconds, and
    measures CPU utilization via psutil.

    Args:
        codec: "vp8" (libvpx, aiortc default), "h264" (libx264, forced
               alternative), or "capabilities" (list and return).
        duration_s: Measurement window in seconds (default 30).
        width: Frame width (default 1280).
        height: Frame height (default 720).
        fps: Target frame rate (default 25).
    """
    import av
    import psutil
    import numpy as np

    # ── Step 1: List PyAV codec capabilities ───────────────────────────
    available_codecs = []
    for name in ["libvpx", "libx264", "h264_nvenc", "h264", "vp8", "vp8_nvenc"]:
        try:
            av.Codec(name, "w")
            available_codecs.append(name)
        except Exception:
            pass

    print(f"[probe] PyAV encoders available: {available_codecs}")
    has_nvenc = "h264_nvenc" in available_codecs or "vp8_nvenc" in available_codecs
    print(f"[probe] NVENC available: {'YES' if has_nvenc else 'NO (CPU-only PyAV wheel)'}")

    if codec == "capabilities":
        return {"available_codecs": available_codecs, "nvenc": has_nvenc, "skipped": True}

    # ── Step 2: Map codec name to PyAV encoder name ────────────────────
    codec_map = {
        "vp8": "libvpx",
        "h264": "libx264",
    }
    enc_name = codec_map.get(codec, codec)
    if enc_name not in available_codecs:
        return {
            "codec": codec,
            "error": f"Encoder '{enc_name}' not available in PyAV wheel. Available: {available_codecs}",
        }

    print(f"[probe] Using encoder: {enc_name} ({codec}) at {width}x{height}@{fps}fps")

    # ── Step 3: Create PyAV encoder ────────────────────────────────────
    enc = av.Codec(enc_name, "w").create()
    enc.width = width
    enc.height = height
    enc.pix_fmt = "yuv420p"
    enc.time_base = fractions.Fraction(1, fps)
    enc.open()

    # ── Step 4: Generate + encode frames ───────────────────────────────
    proc = psutil.Process()
    proc.cpu_percent(interval=None)  # prime
    cpu_samples = []

    def cpu_sampler():
        for _ in range(duration_s):
            cpu_samples.append(proc.cpu_percent(interval=1.0))

    sampler_thread = threading.Thread(target=cpu_sampler, daemon=True)
    sampler_thread.start()

    frame_count = 0
    total_bytes = 0
    t_start = time.monotonic()

    # Pre-build the Y-plane template (shifting gradient changes each frame)
    y_base = np.arange(height, dtype=np.int32)[:, None]

    while time.monotonic() - t_start < duration_s:
        # Build shifting-gradient I420 frame
        # Y: height × width, U/V: height/2 × width/2
        fn = frame_count
        y_plane = ((y_base + fn) % 256).astype(np.uint8)
        y_plane = np.broadcast_to(y_plane, (height, width))
        u_plane = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        v_plane = np.full((height // 2, width // 2), 128, dtype=np.uint8)
        # Pack into I420: flatten all planes and reshape to (H*3/2, W)
        i420 = np.concatenate([
            y_plane.ravel(), u_plane.ravel(), v_plane.ravel()
        ]).reshape(height * 3 // 2, width).copy()

        frame = av.VideoFrame.from_ndarray(i420, format="yuv420p")
        frame.pts = frame_count
        packets = enc.encode(frame)
        for pkt in packets:
            total_bytes += pkt.size
        frame_count += 1

        if frame_count % 250 == 0:
            print(f"[probe] encoded {frame_count} frames, {total_bytes} bytes, "
                  f"cpu_sample={cpu_samples[-1] if cpu_samples else '?'}%")

    # Flush encoder
    for pkt in enc.encode(None):
        total_bytes += pkt.size

    t_end = time.monotonic()
    sampler_thread.join(timeout=2)

    wall_s = t_end - t_start
    encode_fps = frame_count / wall_s if wall_s > 0 else 0
    cpu_mean = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
    cpu_max = max(cpu_samples) if cpu_samples else 0
    cpu_cores = cpu_mean / 100.0

    result = {
        "codec": codec,
        "encoder": enc_name,
        "resolution": f"{width}x{height}",
        "target_fps": fps,
        "duration_s": round(wall_s, 2),
        "frames_encoded": frame_count,
        "encode_fps": round(encode_fps, 2),
        "cpu_percent_mean": round(cpu_mean, 1),
        "cpu_percent_max": round(cpu_max, 1),
        "cpu_cores_used": round(cpu_cores, 2),
        "bytes_total": total_bytes,
        "bytes_per_frame": round(total_bytes / frame_count, 1) if frame_count > 0 else 0,
        "nvenc_available": has_nvenc,
        "available_codecs": available_codecs,
        "allocated_cpus": 4,
    }

    print(f"[probe] DONE: {frame_count} frames in {wall_s:.1f}s = {encode_fps:.1f} fps, "
          f"CPU mean={cpu_mean:.1f}% max={cpu_max:.1f}%, "
          f"cores={cpu_cores:.2f}")

    return result


@app.local_entrypoint()
def main(
    codec: str = "vp8",
    duration_s: int = 30,
):
    """Run the CPU encode probe and print results.

    Usage:
        modal run modal_cpu_encode_probe.py --codec vp8
        modal run modal_cpu_encode_probe.py --codec h264
        modal run modal_cpu_encode_probe.py --codec capabilities
    """
    result = probe_encode.remote(codec=codec, duration_s=duration_s)

    if result.get("skipped"):
        print("\n=== PYAV CODEC CAPABILITIES ===")
        for c in result["available_codecs"]:
            print(f"  {c}")
        print(f"\n  NVENC available: {'YES' if result['nvenc'] else 'NO'}")
        return

    if result.get("error"):
        print(f"\n=== PROBE FAILED: {result['error']} ===")
        return

    print("\n" + "=" * 65)
    print("  CPU ENCODE PROBE RESULTS — PyAV Direct Encoding")
    print("=" * 65)
    print(f"  Codec:           {result['codec']} ({result['encoder']})")
    print(f"  Resolution:      {result['resolution']} @ {result['target_fps']}fps target")
    print(f"  Duration:        {result['duration_s']}s")
    print(f"  Frames encoded:  {result['frames_encoded']}")
    print(f"  Encode FPS:      {result['encode_fps']}")
    print(f"  CPU% mean:       {result['cpu_percent_mean']}%")
    print(f"  CPU% max:        {result['cpu_percent_max']}%")
    print(f"  CPU cores used:  {result['cpu_cores_used']}")
    print(f"  Bytes/frame:     {result['bytes_per_frame']}")
    print(f"  NVENC in PyAV:   {'YES' if result['nvenc_available'] else 'NO (CPU-only)'}")
    print(f"  Allocated CPUs:  {result['allocated_cpus']}")
    print("=" * 65)

    # Architecture verdict
    print("\n  VERDICT:")
    if result["cpu_percent_mean"] > 50:
        print(f"  ⚠ {result['codec']} encoding consumes {result['cpu_percent_mean']}% of a CPU core.")
        print(f"    This is a SIGNIFICANT CPU tax absent from the fMP4 NVENC path.")
        print(f"    The SFU path cannot scale to multiple concurrent sessions on one container.")
    elif result["cpu_percent_mean"] > 20:
        print(f"  ⚠ {result['codec']} encoding consumes {result['cpu_percent_mean']}% of a CPU core.")
        print(f"    Moderate CPU cost — limits concurrency on shared containers.")
    else:
        print(f"  ✓ {result['codec']} encoding uses {result['cpu_percent_mean']}% CPU — low overhead.")

    if result["encode_fps"] < result["target_fps"]:
        print(f"  ⚠ Encode throughput {result['encode_fps']} fps < target {result['target_fps']} fps — CANNOT KEEP UP.")
    else:
        print(f"  ✓ Encode throughput {result['encode_fps']} fps >= target {result['target_fps']} fps — can keep up.")

    if not result["nvenc_available"]:
        print(f"  ⚠ PyAV wheel has NO NVENC — CPU encoding is the ONLY option via aiortc/PyAV.")
        print(f"    For GPU-backed WebRTC encoding, consider FFmpeg subprocess (h264_nvenc) or GStreamer (nvh264enc).")

    print("=" * 65)
