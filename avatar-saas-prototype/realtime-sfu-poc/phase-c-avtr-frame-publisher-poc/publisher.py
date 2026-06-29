#!/usr/bin/env python3
"""Phase C0: Local synthetic 25fps publisher.

Validates the PacedVideoTrack fix (true 25fps instead of aiortc's
hardcoded 30fps) by publishing animated test frames to the Cloudflare
Realtime SFU. A browser subscriber (via the Worker proxy) verifies
the received FPS is in the 23-27 range.

Usage:
  python publisher.py --duration 150

Reads CALLS_APP_ID and CALLS_APP_SECRET from environment (.env file).
Prints "PUBLISHER_READY {sessionId} {trackName}" on stdout when ICE
is connected and publishing has started.
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

import aiohttp
from aiortc import RTCPeerConnection, RTCConfiguration, RTCIceServer, RTCSessionDescription

from frame_source import SyntheticFrameSource
from video_track import PacedVideoTrack

logging.basicConfig(level=logging.INFO, format="%(asctime)s [publisher] %(message)s", stream=sys.stderr)
logger = logging.getLogger("phase-c-publisher")

FPS = 25
TRACK_NAME = "phase-c-synthetic-25fps"
STUN_SERVER = "stun:stun.cloudflare.com:3478"
SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"


async def run_publisher(duration: int = 150) -> None:
    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        print("ERROR: CALLS_APP_ID and CALLS_APP_SECRET must be set", file=sys.stderr)
        sys.exit(1)

    headers = {"Authorization": f"Bearer {app_secret}", "Content-Type": "application/json"}

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

        # Track metrics
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

        # 3. Add paced video track (25fps fix)
        source = SyntheticFrameSource()
        video_track = PacedVideoTrack(source)
        pc.addTrack(video_track)

        # 4. Create offer + wait for ICE gathering
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        gather_start = time.monotonic()
        while True:
            sdp = pc.localDescription.sdp
            if "a=candidate:" in sdp:
                if "end-of-candidates" in sdp:
                    logger.info("ICE gathering complete (end-of-candidates)")
                    break
                # Give a short window for more candidates
                await asyncio.sleep(0.5)
                if "end-of-candidates" in pc.localDescription.sdp:
                    break
                break
            if time.monotonic() - gather_start > 10:
                logger.warning("ICE gathering timeout — proceeding with current candidates")
                break
            await asyncio.sleep(0.2)

        has_candidates = "a=candidate:" in pc.localDescription.sdp
        logger.info(f"SDP has candidates: {has_candidates}")
        if not has_candidates:
            raise RuntimeError("No ICE candidates in local SDP")

        # 5. POST tracks/new
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

        logger.info(f"tracks/new OK, mid={video_mid}")

        # 6. Set remote description
        answer_sdp = answer_data["sessionDescription"]["sdp"]
        answer_type = answer_data["sessionDescription"]["type"]
        answer = RTCSessionDescription(sdp=answer_sdp, type=answer_type)
        await pc.setRemoteDescription(answer)
        logger.info("Remote description set")

        # 7. Wait for ICE connected
        try:
            await asyncio.wait_for(ice_connected.wait(), timeout=30)
        except asyncio.TimeoutError:
            logger.error("ICE connection timeout")
            raise

        if ice_failed.is_set():
            raise RuntimeError("ICE connection failed")

        logger.info("ICE connected — publishing started")
        print(f"PUBLISHER_READY {session_id} {TRACK_NAME}", flush=True)

        # 8. Run for duration
        start = time.monotonic()
        try:
            while time.monotonic() - start < duration:
                await asyncio.sleep(10)
                elapsed = int(time.monotonic() - start)
                m = source.metrics()
                logger.info(
                    f"heartbeat {elapsed}s: frames_gen={m.frames_generated} "
                    f"emitted={m.frames_emitted}"
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Shutdown signal received")

        # 9. Teardown — close tracks (force at top level per CF Realtime API)
        logger.info("Closing track...")
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

        # 10. Close PC
        source.stop()
        await pc.close()
        logger.info("Peer connection closed")
        print("PUBLISHER_DONE", flush=True)
        os._exit(0)


def main():
    parser = argparse.ArgumentParser(description="Phase C0 synthetic 25fps publisher")
    parser.add_argument("--duration", type=int, default=150, help="Publish duration in seconds")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        logger.info(f"Signal {sig} received")
        for task in asyncio.all_tasks(loop):
            task.cancel()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(run_publisher(duration=args.duration))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    main()
