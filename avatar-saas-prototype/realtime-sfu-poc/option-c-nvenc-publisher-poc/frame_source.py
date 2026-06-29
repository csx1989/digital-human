"""Frame sources for the Option C GStreamer publisher.

- SyntheticI420Source: generates shifting-gradient I420 frames at 25fps.
- AvtrFrameSource adapter will be added in Phase 3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

FPS_25 = 25
WIDTH_720P = 1280
HEIGHT_720P = 720


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


class SyntheticI420Source:
    """Animated gradient + bouncing circle at 25fps. Returns I420 (H*3/2, W) uint8."""

    def __init__(self, width: int = WIDTH_720P, height: int = HEIGHT_720P, fps: int = FPS_25):
        self._width = width
        self._height = height
        self._fps = fps
        self._counter = 0
        self._metrics = FrameSourceMetrics()
        self._start = time.monotonic()

    def get_frame(self) -> np.ndarray:
        """Return the next frame, blocking slightly to maintain ~25fps if called in a loop."""
        self._counter += 1
        t = self._counter / self._fps

        # Pace loosely around the target frame rate
        target = self._start + self._counter / self._fps
        wait = target - time.monotonic()
        if wait > 0:
            time.sleep(wait)

        h, w = self._height, self._width

        # Y plane: animated gradient
        y_base = np.arange(h, dtype=np.int32)[:, None]
        y_plane = ((y_base + int(t * 60)) % 256).astype(np.uint8)
        y_plane = np.broadcast_to(y_plane, (h, w))

        # U/V planes: neutral + slight motion
        u_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        v_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8)

        # Bouncing red circle drawn on Y plane
        cx = int(w / 2 + np.sin(t * 2.1) * (w / 3.2))
        cy = int(h / 2 + np.cos(t * 1.4) * (h / 3.2))
        radius = 80
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        y_plane[mask] = 220

        # Progress bar at bottom
        bar_y = h - 40
        bar_width = int(w * ((self._counter % (self._fps * 5)) / (self._fps * 5)))
        y_plane[bar_y:h, :bar_width] = 40

        # Pack I420: Y (H,W) + U (H/2,W/2) + V (H/2,W/2)
        i420 = np.concatenate([
            y_plane.ravel(),
            u_plane.ravel(),
            v_plane.ravel(),
        ]).reshape(h * 3 // 2, w).copy()

        self._metrics.frames_generated += 1
        self._metrics.frames_emitted += 1
        return i420

    def metrics(self) -> FrameSourceMetrics:
        return self._metrics

    def stop(self) -> None:
        pass


def black_i420(width: int = WIDTH_720P, height: int = HEIGHT_720P) -> np.ndarray:
    """Return a black I420 frame."""
    frame = np.zeros((height * 3 // 2, width), dtype=np.uint8)
    frame[:height, :] = 16
    frame[height:, :] = 128
    return frame
