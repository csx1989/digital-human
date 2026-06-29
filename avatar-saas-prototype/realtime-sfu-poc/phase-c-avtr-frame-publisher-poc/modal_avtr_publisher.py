#!/usr/bin/env python3
"""Phase C2: Modal GPU AVTR frame publisher — full WebRTC to SFU.

Publishes actual AVTR-rendered I420 frames (1280×720 @ 25fps) from the
L40S GPU through the Cloudflare Realtime SFU.

Correction #1: Reports underflow metrics — pass only if underflow <= 5.
Correction #2: process_chunk runs in a dedicated producer THREAD.
Correction #3: Warmup excluded from steady-state (handled by C1).

Usage:
  modal run modal_avtr_publisher.py --duration 150 --avatar-id arnold
"""

from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import modal

# Guard non-modal imports for local parsing (Modal CLI python lacks aiortc/numpy)
try:
    import numpy as np
except ImportError:
    np = None

try:
    from aiortc import VideoStreamTrack, RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamError, VIDEO_CLOCK_RATE
except ImportError:
    VideoStreamTrack = object
    MediaStreamError = Exception
    VIDEO_CLOCK_RATE = 90000
    RTCPeerConnection = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCSessionDescription = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"
CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"
FPS = 25
TRACK_NAME = "phase-c-avtr-25fps"
STUN_SERVER = "stun:stun.cloudflare.com:3478"
SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"
PTIME_25 = fractions.Fraction(1, FPS)
TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
TICKS_PER_FRAME = int(PTIME_25 * VIDEO_CLOCK_RATE)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [c2] %(message)s", stream=sys.stdout)
logger = logging.getLogger("phase-c2")

# ---------------------------------------------------------------------------
# Modal image + app
# ---------------------------------------------------------------------------

image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])
    .apt_install(
        "libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1",
        "ffmpeg", "git", "gcc", "g++", "pkg-config",
    )
    .pip_install("torch>=2.8,<2.9", "torchaudio>=2.8,<2.9", "torchvision>=0.23,<0.24",
                 extra_index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy>=2.0,<3", "safetensors>=0.5,<1", "huggingface_hub>=1.3",
        "kornia>=0.8.1,<0.9", "roma>=1.5.4,<2", "opencv-python-headless>=4.11,<5",
        "einops>=0.8.1,<0.9", "scikit-image>=0.25,<0.26", "soxr>=0.4,<0.5",
        "imageio>=2.37,<3", "httpx>=0.28,<0.29", "tqdm>=4.67,<5",
        "onnx>=1.17,<2", "onnx2torch>=1.5,<2", "onnxruntime-gpu>=1.20",
        "tensorrt<=10.12", "transformers>=5.3", "soundfile>=0.13",
    )
    .pip_install("cython")
    .pip_install("aiortc", "aiohttp", extra_index_url="https://pypi.org/simple/")
    .add_local_dir("/Users/ayya/developer/avtr-1/src/avtr1_renderer",
                   remote_path="/opt/avtr1/avtr1_renderer", copy=True)
    .env({"PYTHONPATH": "/opt/avtr1", "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH})
)

app = modal.App("avtr-phase-c2-publisher", image=image)
l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Correction #1: Metrics
# ---------------------------------------------------------------------------

@dataclass
class FrameSourceMetrics:
    frames_generated: int = 0
    frames_emitted: int = 0
    frames_repeated: int = 0
    underflow_count: int = 0
    queue_depth_min: int = 10**9
    queue_depth_max: int = 0


# ---------------------------------------------------------------------------
# Correction #2: AvtrFrameSource — producer THREAD
# ---------------------------------------------------------------------------

class AvtrFrameSource:
    def __init__(self, pipeline, avatar, bg_id: str, max_queue: int = 10):
        self._pipeline = pipeline
        self._avatar = avatar
        self._bg_id = bg_id
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._metrics = FrameSourceMetrics()
        self._stop_event = threading.Event()
        self._producer_thread = None
        self._loop = None
        self._last_frame = None

        mg = pipeline._motion_generator
        self._chunk_size = mg.chunk_size
        self._future_size = mg.future_size
        self._frame_len = mg.frame_len
        self._audio_shift = getattr(mg, "audio_shift", 0)
        self._window = (self._chunk_size + self._future_size) * self._frame_len + self._audio_shift
        self._step_samples = self._chunk_size * self._frame_len

    async def start(self):
        self._loop = asyncio.get_running_loop()
        self._producer_thread = threading.Thread(
            target=self._producer_loop, daemon=True, name="avtr-producer")
        self._producer_thread.start()

    def _producer_loop(self):
        import torch
        import numpy as _np
        from avtr1_renderer.types import Chunk, RenderOptions

        render_state = self._pipeline.initial_state(self._avatar)
        silence = _np.zeros(self._window, dtype=_np.float32)
        audio_history = _np.zeros(0, dtype=_np.float32)
        options = RenderOptions(pixel_format="yuv_i420", bg_id=self._bg_id)
        chunk_i = 0

        while not self._stop_event.is_set():
            try:
                t0 = time.monotonic()
                audio_history = _np.concatenate([audio_history, _np.zeros(self._step_samples, dtype=_np.float32)])
                if len(audio_history) >= self._window:
                    audio_tail = audio_history[-self._window:].copy()
                else:
                    audio_tail = _np.pad(audio_history, (self._window - len(audio_history), 0))
                max_hist = self._window + self._step_samples * 2
                if len(audio_history) > max_hist:
                    audio_history = audio_history[-self._window:]

                chunk_obj = Chunk(audio_speech=audio_tail, audio_listen=silence)
                with torch.inference_mode():
                    new_state, frames_iter = self._pipeline.process_chunk(
                        self._avatar, chunk_obj, state=render_state, options=options)
                    frames_list = list(frames_iter)
                render_state = new_state
                render_ms = (time.monotonic() - t0) * 1000

                if chunk_i < 5 or chunk_i % 20 == 0:
                    logger.info(f"producer chunk {chunk_i}: {len(frames_list)} frames in {render_ms:.1f}ms")

                for frame in frames_list:
                    if self._stop_event.is_set():
                        break
                    while not self._stop_event.is_set():
                        try:
                            future = asyncio.run_coroutine_threadsafe(
                                self._queue.put(frame.data.copy()), self._loop)
                            future.result(timeout=5)
                            self._metrics.frames_generated += 1
                            break
                        except asyncio.QueueFull:
                            time.sleep(0.005)
                        except Exception as exc:
                            logger.warning(f"queue put error: {exc}")
                            time.sleep(0.01)

                chunk_i += 1
            except Exception as exc:
                logger.error(f"PRODUCER ERROR: {exc}")
                import traceback; traceback.print_exc()
                break

        logger.info(f"producer exiting: {chunk_i} chunks, gen={self._metrics.frames_generated}")

    async def get_frame(self):
        import numpy as _np
        depth = self._queue.qsize()
        if depth < self._metrics.queue_depth_min:
            self._metrics.queue_depth_min = depth
        if depth > self._metrics.queue_depth_max:
            self._metrics.queue_depth_max = depth
        try:
            frame = self._queue.get_nowait()
            self._last_frame = frame
            self._metrics.frames_emitted += 1
            return frame
        except asyncio.QueueEmpty:
            self._metrics.underflow_count += 1
            if self._last_frame is not None:
                self._metrics.frames_repeated += 1
                return self._last_frame
            h, w = 720, 1280
            black = _np.zeros((h * 3 // 2, w), dtype=_np.uint8)
            black[:h, :] = 16
            black[h:, :] = 128
            self._last_frame = black
            self._metrics.frames_repeated += 1
            return black

    @property
    def metrics(self):
        return self._metrics

    def stop(self):
        self._stop_event.set()


# ---------------------------------------------------------------------------
# PacedVideoTrack — true 25fps (overrides aiortc's hardcoded 30fps)
# ---------------------------------------------------------------------------

class PacedVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, source: AvtrFrameSource):
        super().__init__()
        self._source = source
        self._frame_count = 0

    async def next_timestamp(self):
        if self.readyState != "live":
            raise MediaStreamError
        if hasattr(self, "_timestamp"):
            self._timestamp += TICKS_PER_FRAME
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, TIME_BASE

    async def recv(self):
        from av import VideoFrame
        pts, time_base = await self.next_timestamp()
        frame_data = await self._source.get_frame()
        if frame_data.ndim == 2:
            frame = VideoFrame.from_ndarray(frame_data, format="yuv420p")
        else:
            frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        self._frame_count += 1
        if self._frame_count <= 3 or self._frame_count % 250 == 0:
            logger.info(f"track frame {self._frame_count}: pts={pts} shape={frame_data.shape}")
        return frame


# ---------------------------------------------------------------------------
# Main publisher
# ---------------------------------------------------------------------------

@app.function(
    gpu="L40S",
    volumes={L40S_MOUNT_PATH: l40s_volume, HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret"), modal.Secret.from_name("avtr-calls")],
    timeout=600,
)
def run_publisher(duration: int = 150, avatar_id: str = "arnold", bg_id: str = "aloe_window") -> dict:
    result = {"error": None, "avatar_id": avatar_id, "bg_id": bg_id}

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig, frame):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        result.update(loop.run_until_complete(
            _run_async(duration, avatar_id, bg_id)
        ))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception as exc:
        result["error"] = str(exc)
        import traceback; traceback.print_exc()
    finally:
        loop.close()

    return result


async def _run_async(duration, avatar_id, bg_id) -> dict:
    import torch
    import aiohttp
    import numpy as _np
    from avtr1_renderer.pipeline import Pipeline

    result: dict = {}

    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        return {"error": "CALLS_APP_ID/CALLS_APP_SECRET not set"}

    headers = {"Authorization": f"Bearer {app_secret}", "Content-Type": "application/json"}

    # 1. Pipeline
    logger.info(f"Building pipeline for avatar={avatar_id}...")
    t0 = time.monotonic()
    pipeline, registry = Pipeline.from_artifacts(avatar_ids=[avatar_id])
    logger.info(f"Pipeline built in {time.monotonic() - t0:.1f}s")
    avatar = registry[avatar_id]

    if bg_id not in pipeline._backgrounds:
        available = sorted(pipeline._backgrounds.keys())
        bg_id = available[0] if available else None
    logger.info(f"Using bg_id={bg_id!r}")

    # 2. Frame source + track
    source = AvtrFrameSource(pipeline, avatar, bg_id)
    await source.start()
    await asyncio.sleep(1.0)

    video_track = PacedVideoTrack(source)

    # 3. SFU session
    async with aiohttp.ClientSession() as http:
        async with http.post(f"{SFU_API_BASE}/{app_id}/sessions/new", headers=headers) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"sessions/new: {resp.status} {await resp.text()}")
            session_id = (await resp.json())["sessionId"]
            logger.info(f"SFU session: {session_id}")

        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=[STUN_SERVER])]))

        ice_connected = asyncio.Event()
        ice_failed = asyncio.Event()

        @pc.on("iceconnectionstatechange")
        async def on_ice():
            state = pc.iceConnectionState
            logger.info(f"ICE state: {state}")
            if state in ("connected", "completed"):
                ice_connected.set()
            elif state == "failed":
                ice_failed.set()

        pc.addTrack(video_track)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        gather_start = time.monotonic()
        while True:
            if "a=candidate:" in pc.localDescription.sdp:
                await asyncio.sleep(0.5)
                break
            if time.monotonic() - gather_start > 10:
                break
            await asyncio.sleep(0.2)

        if "a=candidate:" not in pc.localDescription.sdp:
            raise RuntimeError("No ICE candidates")

        video_mid = "0"
        for t in pc.getTransceivers():
            if t.kind == "video":
                video_mid = t.mid or "0"
                break

        tracks_body = {
            "sessionDescription": {"sdp": pc.localDescription.sdp, "type": "offer"},
            "tracks": [{"location": "local", "mid": video_mid, "trackName": TRACK_NAME}],
        }
        async with http.post(f"{SFU_API_BASE}/{app_id}/sessions/{session_id}/tracks/new",
                             headers=headers, json=tracks_body) as resp:
            if resp.status != 200:
                raise RuntimeError(f"tracks/new: {resp.status} {await resp.text()}")
            answer_data = await resp.json()

        answer = RTCSessionDescription(
            sdp=answer_data["sessionDescription"]["sdp"],
            type=answer_data["sessionDescription"]["type"])
        await pc.setRemoteDescription(answer)
        logger.info(f"Remote desc set, mid={video_mid}")

        await asyncio.wait_for(ice_connected.wait(), timeout=30)
        if ice_failed.is_set():
            raise RuntimeError("ICE failed")

        logger.info("ICE connected — publishing AVTR frames")
        print(f"PUBLISHER_READY {session_id} {TRACK_NAME}", flush=True)

        # 4. Run
        start = time.monotonic()
        try:
            while time.monotonic() - start < duration:
                await asyncio.sleep(10)
                elapsed = int(time.monotonic() - start)
                m = source.metrics
                logger.info(
                    f"heartbeat {elapsed}s: gen={m.frames_generated} emit={m.frames_emitted} "
                    f"repeat={m.frames_repeated} underflow={m.underflow_count} "
                    f"qdepth=[{m.queue_depth_min if m.queue_depth_min < 10**9 else 0},{m.queue_depth_max}]")
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutdown signal")

        # 5. Capture metrics
        m = source.metrics
        result["metrics"] = {
            "frames_generated": m.frames_generated,
            "frames_emitted": m.frames_emitted,
            "frames_repeated": m.frames_repeated,
            "underflow_count": m.underflow_count,
            "queue_depth_min": m.queue_depth_min if m.queue_depth_min < 10**9 else 0,
            "queue_depth_max": m.queue_depth_max,
        }

        # 6. Teardown
        source.stop()
        logger.info(f"Closing track (mid={video_mid})...")
        close_status = None
        try:
            close_body = {"tracks": [{"mid": video_mid}], "force": True}
            logger.info(f"tracks/close request body: {json.dumps(close_body)}")
            async with http.put(
                f"{SFU_API_BASE}/{app_id}/sessions/{session_id}/tracks/close",
                headers=headers,
                json=close_body,
            ) as resp:
                close_resp_text = await resp.text()
                close_status = resp.status
                logger.info(f"tracks/close: status={close_status} body={close_resp_text[:300]}")
        except Exception as exc:
            logger.warning(f"tracks/close error: {exc}")

        result["close_status"] = close_status
        result["close_resp_body"] = close_resp_text[:500] if close_status else "no response"

        await pc.close()
        logger.info("PC closed")
        print("PUBLISHER_DONE", flush=True)

    return result


@app.local_entrypoint()
def main(duration: int = 150, avatar_id: str = "arnold", bg_id: str = "aloe_window"):
    result = run_publisher.remote(duration=duration, avatar_id=avatar_id, bg_id=bg_id)
    print("\n=== C2 RESULT ===")
    for k, v in sorted(result.items()):
        print(f"  {k}: {v}")
