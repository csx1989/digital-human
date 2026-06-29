"""PacedVideoTrack — overrides next_timestamp() for true 25fps pacing.

Root cause of Phase B 30fps bug:
  aiortc's VideoStreamTrack.next_timestamp() hardcodes VIDEO_PTIME = 1/30,
  advancing _timestamp by 3000 ticks (33.3ms) per frame regardless of
  the FPS constant in user code. The asyncio.sleep(wait) inside it DOES
  enforce wall-clock pacing — but at 30fps, not 25fps.

Fix: Override next_timestamp() to use PTIME = 1/25 (3600 ticks = 40ms).

This track consumes any FrameSource and produces PyAV VideoFrame objects.
"""

from __future__ import annotations

import asyncio
import fractions
import time

import numpy as np
from av import VideoFrame
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError, VIDEO_CLOCK_RATE

from frame_source import FrameSource, FPS_25

PTIME_25 = fractions.Fraction(1, FPS_25)
TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
TICKS_PER_FRAME_25 = int(PTIME_25 * VIDEO_CLOCK_RATE)  # 3600


class PacedVideoTrack(VideoStreamTrack):
    """VideoStreamTrack with explicit 25fps wall-clock pacing.

    - next_timestamp() advances by 3600 ticks (40ms) per frame
    - asyncio.sleep enforces real-time pacing anchored to _start
    - recv() pulls from a FrameSource and wraps in a PyAV VideoFrame
    - I420 arrays (2D, shape (3H/2, W)) use format="yuv420p" — zero conversion
    - RGB arrays (3D, shape (H, W, 3)) use format="rgb24"
    """

    kind = "video"

    def __init__(self, frame_source: FrameSource):
        super().__init__()
        self._source = frame_source
        self._frame_count = 0

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += TICKS_PER_FRAME_25
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, TIME_BASE

    async def recv(self) -> VideoFrame:
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
            print(f"[paced-track] frame {self._frame_count}: pts={pts} shape={frame_data.shape} dtype={frame_data.dtype}")

        return frame


__all__ = ["PacedVideoTrack", "FPS_25", "PTIME_25", "TIME_BASE", "TICKS_PER_FRAME_25"]
