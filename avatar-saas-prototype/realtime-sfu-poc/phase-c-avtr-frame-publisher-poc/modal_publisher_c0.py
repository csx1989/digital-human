#!/usr/bin/env python3
"""Phase C0 (Modal): Synthetic 25fps publisher on Modal CPU.

Validates that the PacedVideoTrack 25fps pacing fix works correctly
inside a Modal container (same networking path as C2's GPU container).

Usage:
  python -m modal run modal_publisher_c0.py --duration 150

Secret: modal.Secret.from_name("avtr-calls") with CALLS_APP_ID + CALLS_APP_SECRET.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Any

import modal
import aiohttp
import numpy as np
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer
from av import VideoFrame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [modal-c0] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("phase-c-modal-c0")

FPS = 25
WIDTH = 1280
HEIGHT = 720
TRACK_NAME = "phase-c-modal-synthetic-25fps"
STUN_SERVER = "stun:stun.cloudflare.com:3478"
SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"

# Modal image — CPU only, same as Phase B2
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("aiortc", "numpy", "av", "aiohttp")
)

app = modal.App("avtr-phase-c-c0-modal-publisher", image=image)


class Synthetic25FpsTrack(VideoStreamTrack if False else type(
    "T", (), {}
)):  # placeholder — actual import below
    pass


def _make_track_class():
    """Build the PacedVideoTrack class at runtime (avoids import issues in Modal)."""
    import fractions
    from aiortc import VideoStreamTrack
    from aiortc.mediastreams import MediaStreamError, VIDEO_CLOCK_RATE

    PTIME_25 = fractions.Fraction(1, FPS)
    TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
    TICKS_PER_FRAME = int(PTIME_25 * VIDEO_CLOCK_RATE)

    class PacedVideoTrack(VideoStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self._counter = 0

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

        async def recv(self) -> VideoFrame:
            pts, time_base = await self.next_timestamp()
            self._counter += 1
            t = self._counter / FPS

            img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

            r_vals = ((np.arange(WIDTH) / WIDTH * 255 + t * 30) % 256).astype(np.uint8)
            g_vals = ((np.arange(HEIGHT) / HEIGHT * 255 + t * 20) % 256).astype(np.uint8)
            b_val = int((128 + np.sin(t) * 60) % 256)
            img[:, :, 0] = r_vals[np.newaxis, :]
            img[:, :, 1] = g_vals[:, np.newaxis]
            img[:, :, 2] = b_val

            cx = int(WIDTH / 2 + np.sin(t * 2.1) * (WIDTH / 3.2))
            cy = int(HEIGHT / 2 + np.cos(t * 1.4) * (HEIGHT / 3.2))
            radius = 60
            yy, xx = np.ogrid[:HEIGHT, :WIDTH]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
            img[mask] = [220, 40, 40]

            bar_y = HEIGHT - 30
            bar_width = int(WIDTH * ((self._counter % (FPS * 5)) / (FPS * 5)))
            img[bar_y : HEIGHT, :bar_width] = [40, 220, 40]

            frame = VideoFrame.from_ndarray(img, format="rgb24")
            frame.pts = pts
            frame.time_base = time_base
            return frame

    return PacedVideoTrack


async def _run_publisher(duration: int) -> None:
    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        print("ERROR: CALLS_APP_ID and CALLS_APP_SECRET must be set")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {app_secret}", "Content-Type": "application/json"}
    PacedVideoTrack = _make_track_class()

    async with aiohttp.ClientSession() as http:
        # 1. Create SFU session
        async with http.post(f"{SFU_API_BASE}/{app_id}/sessions/new", headers=headers) as resp:
            if resp.status not in (200, 201):
                body = await resp.text()
                raise RuntimeError(f"sessions/new failed: {resp.status} {body}")
            session_data = await resp.json()
            session_id = session_data["sessionId"]
            logger.info(f"SFU session: {session_id}")

        # 2. Create peer connection
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=[STUN_SERVER])])
        )

        ice_connected = asyncio.Event()
        ice_failed = asyncio.Event()

        @pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            state = pc.iceConnectionState
            logger.info(f"ICE state: {state}")
            if state in ("connected", "completed"):
                ice_connected.set()
            elif state == "failed":
                ice_failed.set()

        # 3. Add paced video track
        video_track = PacedVideoTrack()
        pc.addTrack(video_track)

        # 4. Create offer + ICE gathering
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        gather_start = time.monotonic()
        while True:
            sdp = pc.localDescription.sdp
            if "a=candidate:" in sdp:
                await asyncio.sleep(0.5)
                break
            if time.monotonic() - gather_start > 10:
                logger.warning("ICE gathering timeout")
                break
            await asyncio.sleep(0.2)

        if "a=candidate:" not in pc.localDescription.sdp:
            raise RuntimeError("No ICE candidates")

        # 5. tracks/new
        video_mid = "0"
        for t in pc.getTransceivers():
            if t.kind == "video":
                video_mid = t.mid or "0"
                break

        tracks_body: dict[str, Any] = {
            "sessionDescription": {"sdp": pc.localDescription.sdp, "type": "offer"},
            "tracks": [{"location": "local", "mid": video_mid, "trackName": TRACK_NAME}],
        }

        async with http.post(
            f"{SFU_API_BASE}/{app_id}/sessions/{session_id}/tracks/new",
            headers=headers,
            json=tracks_body,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"tracks/new failed: {resp.status} {body}")
            answer_data = await resp.json()

        await pc.setRemoteDescription(answer_data["sessionDescription"])
        logger.info(f"Remote description set, mid={video_mid}")

        # 6. Wait for ICE
        try:
            await asyncio.wait_for(ice_connected.wait(), timeout=30)
        except asyncio.TimeoutError:
            raise RuntimeError("ICE connection timeout")

        if ice_failed.is_set():
            raise RuntimeError("ICE connection failed")

        logger.info("ICE connected — publishing")
        print(f"PUBLISHER_READY {session_id} {TRACK_NAME}", flush=True)

        # 7. Run
        start = time.monotonic()
        try:
            while time.monotonic() - start < duration:
                await asyncio.sleep(10)
                elapsed = int(time.monotonic() - start)
                logger.info(f"heartbeat {elapsed}s")
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutdown signal")

        # 8. Teardown
        try:
            close_body = {"tracks": [{"mid": video_mid}], "force": True}
            logger.info(f"tracks/close request body: {json.dumps(close_body)}")
            async with http.put(
                f"{SFU_API_BASE}/{app_id}/sessions/{session_id}/tracks/close",
                headers=headers,
                json=close_body,
            ) as resp:
                close_resp_text = await resp.text()
                logger.info(f"tracks/close: status={resp.status} body={close_resp_text[:300]}")
        except Exception as exc:
            logger.warning(f"tracks/close error: {exc}")

        await pc.close()
        logger.info("Peer connection closed")
        print("PUBLISHER_DONE", flush=True)


@app.function(
    secret=modal.Secret.from_name("avtr-calls"),
    cpu=1,
    timeout=600,
)
def run(duration: int = 150) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def handle_signal(sig, frame):
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(_run_publisher(duration=duration))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()


@app.local_entrypoint()
def main(duration: int = 150):
    run.remote(duration=duration)
