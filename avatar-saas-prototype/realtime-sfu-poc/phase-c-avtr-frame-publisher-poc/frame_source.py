"""Frame source abstraction for Phase C WebRTC publisher.

Two implementations:
- SyntheticFrameSource: animated gradient + bouncing circle at 25fps (C0)
- AvtrFrameSource: pulls I420 frames from AVTR pipeline.process_chunk() (C1/C2)

Both produce numpy arrays consumable by PacedVideoTrack.

Correction #2: AvtrFrameSource runs process_chunk in a dedicated thread
(never in the asyncio event loop) to avoid blocking WebRTC pacing/ICE/stats.
"""

from __future__ import annotations

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Metrics (Correction #1: underflow must be tracked and reported)
# ---------------------------------------------------------------------------

@dataclass
class FrameSourceMetrics:
    frames_generated: int = 0
    frames_emitted: int = 0
    frames_repeated: int = 0
    underflow_count: int = 0
    queue_depth_min: int = 10**9
    queue_depth_max: int = 0

    def snapshot(self) -> dict:
        return {
            "frames_generated": self.frames_generated,
            "frames_emitted": self.frames_emitted,
            "frames_repeated": self.frames_repeated,
            "underflow_count": self.underflow_count,
            "queue_depth_min": self.queue_depth_min if self.queue_depth_min < 10**9 else 0,
            "queue_depth_max": self.queue_depth_max,
        }


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class FrameSource(ABC):
    """Produces one frame per call. Must be safe to call from recv() at 25fps."""

    @abstractmethod
    async def get_frame(self) -> np.ndarray:
        ...

    @abstractmethod
    def metrics(self) -> FrameSourceMetrics:
        ...

    @abstractmethod
    def stop(self) -> None:
        ...


# ---------------------------------------------------------------------------
# C0: Synthetic source — animated gradient + bouncing circle at 25fps
# ---------------------------------------------------------------------------

FPS_25 = 25
WIDTH_720P = 1280
HEIGHT_720P = 720


class SyntheticFrameSource(FrameSource):
    """Animated test pattern. Returns RGB uint8 (H, W, 3) arrays.

    Not I420 — the PacedVideoTrack handles format detection and PyAV
    converts RGB to the encoder's preferred pixel format internally.
    Uses 1280×720 to match AVTR's native resolution and test real
    encoder/bandwidth load.
    """

    def __init__(self, width: int = WIDTH_720P, height: int = HEIGHT_720P):
        self._width = width
        self._height = height
        self._counter = 0
        self._metrics = FrameSourceMetrics()

    async def get_frame(self) -> np.ndarray:
        self._counter += 1
        t = self._counter / FPS_25

        img = np.zeros((self._height, self._width, 3), dtype=np.uint8)

        r_vals = ((np.arange(self._width) / self._width * 255 + t * 30) % 256).astype(np.uint8)
        g_vals = ((np.arange(self._height) / self._height * 255 + t * 20) % 256).astype(np.uint8)
        b_val = int((128 + np.sin(t) * 60) % 256)
        img[:, :, 0] = r_vals[np.newaxis, :]
        img[:, :, 1] = g_vals[:, np.newaxis]
        img[:, :, 2] = b_val

        cx = int(self._width / 2 + np.sin(t * 2.1) * (self._width / 3.2))
        cy = int(self._height / 2 + np.cos(t * 1.4) * (self._height / 3.2))
        radius = 60
        yy, xx = np.ogrid[:self._height, :self._width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = [220, 40, 40]

        bar_y = self._height - 30
        bar_width = int(self._width * ((self._counter % (FPS_25 * 5)) / (FPS_25 * 5)))
        img[bar_y : self._height, :bar_width] = [40, 220, 40]

        self._metrics.frames_generated += 1
        self._metrics.frames_emitted += 1
        return img

    def metrics(self) -> FrameSourceMetrics:
        return self._metrics

    def stop(self) -> None:
        pass


# ---------------------------------------------------------------------------
# C1/C2: AVTR frame source — I420 from pipeline.process_chunk()
# ---------------------------------------------------------------------------

class AvtrFrameSource(FrameSource):
    """Pulls I420 frames from the AVTR rendering pipeline.

    Correction #2: process_chunk() runs in a dedicated producer THREAD,
    never inside the asyncio event loop. Frames are pushed into an
    asyncio.Queue that recv() pops from at 25fps.

    Correction #1: Underflow (empty queue) repeats the last frame, but
    every repeat is counted in metrics. The runner gates on these counts.

    Lifecycle:
        source = AvtrFrameSource(pipeline, avatar, bg_id, max_queue=10)
        await source.start()      # spawns producer thread
        frame = await source.get_frame()  # called by PacedVideoTrack.recv()
        source.stop()             # signals producer thread to exit
    """

    def __init__(
        self,
        pipeline,
        avatar,
        bg_id: str,
        max_queue: int = 10,
    ):
        self._pipeline = pipeline
        self._avatar = avatar
        self._bg_id = bg_id

        from avtr1_renderer.types import Chunk, RenderOptions
        self._Chunk = Chunk
        self._RenderOptions = RenderOptions

        mg = pipeline._motion_generator
        self._chunk_size = mg.chunk_size          # 5
        self._future_size = mg.future_size        # 5
        self._frame_len = mg.frame_len            # 640
        self._audio_shift = getattr(mg, "audio_shift", 0)  # 80
        self._window = (self._chunk_size + self._future_size) * self._frame_len + self._audio_shift  # 6480
        self._step_samples = self._chunk_size * self._frame_len  # 3200

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self._metrics = FrameSourceMetrics()
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_frame: np.ndarray | None = None
        self._error: Exception | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="avtr-frame-producer",
        )
        self._producer_thread.start()

    def _producer_loop(self) -> None:
        """Runs in a dedicated thread — calls process_chunk synchronously."""
        import torch

        render_state = self._pipeline.initial_state(self._avatar)
        silence_audio = np.zeros(self._window, dtype=np.float32)
        audio_history = np.zeros(0, dtype=np.float32)
        options = self._RenderOptions(pixel_format="yuv_i420", bg_id=self._bg_id)

        chunk_i = 0
        while not self._stop_event.is_set():
            try:
                t_start = time.monotonic()

                audio_history = np.concatenate([audio_history, np.zeros(self._step_samples, dtype=np.float32)])
                if len(audio_history) >= self._window:
                    audio_tail = audio_history[-self._window:].copy()
                else:
                    audio_tail = np.pad(audio_history, (self._window - len(audio_history), 0))
                max_hist = self._window + self._step_samples * 2
                if len(audio_history) > max_hist:
                    audio_history = audio_history[-self._window:]

                chunk_obj = self._Chunk(audio_speech=audio_tail, audio_listen=silence_audio)
                with torch.inference_mode():
                    new_state, frames_iter = self._pipeline.process_chunk(
                        self._avatar, chunk_obj, state=render_state, options=options,
                    )
                    frames_list = list(frames_iter)
                render_state = new_state

                render_ms = (time.monotonic() - t_start) * 1000
                if chunk_i < 5 or chunk_i % 20 == 0:
                    print(f"[avtr-source] chunk {chunk_i}: {len(frames_list)} frames in {render_ms:.1f}ms")

                for frame in frames_list:
                    if self._stop_event.is_set():
                        break
                    while not self._stop_event.is_set():
                        try:
                            if self._loop and self._loop.is_running():
                                future = asyncio.run_coroutine_threadsafe(
                                    self._queue.put(frame.data.copy()),
                                    self._loop,
                                )
                                future.result(timeout=5)
                                self._metrics.frames_generated += 1
                                break
                            else:
                                return
                        except asyncio.QueueFull:
                            time.sleep(0.01)
                        except Exception as exc:
                            print(f"[avtr-source] queue put error: {exc}")
                            time.sleep(0.01)

                chunk_i += 1

            except Exception as exc:
                self._error = exc
                print(f"[avtr-source] PRODUCER ERROR: {exc}")
                import traceback
                traceback.print_exc()
                break

        print(f"[avtr-source] producer thread exiting after {chunk_i} chunks")

    async def get_frame(self) -> np.ndarray:
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
            # Cold start: queue empty, no last frame — emit black I420
            h, w = 720, 1280
            black = np.zeros((h * 3 // 2, w), dtype=np.uint8)
            black[:h, :] = 16   # Y=16 (BT.601 limited-range black)
            black[h:, :] = 128  # UV=128 (neutral chroma)
            self._last_frame = black
            self._metrics.frames_repeated += 1
            return black

    def metrics(self) -> FrameSourceMetrics:
        return self._metrics

    def stop(self) -> None:
        self._stop_event.set()


__all__ = [
    "FrameSource",
    "FrameSourceMetrics",
    "SyntheticFrameSource",
    "AvtrFrameSource",
    "FPS_25",
    "WIDTH_720P",
    "HEIGHT_720P",
]
