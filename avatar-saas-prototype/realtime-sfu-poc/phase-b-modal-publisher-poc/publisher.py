#!/usr/bin/env python3
"""
Phase B1 — Synthetic WebRTC Publisher (aiortc → Cloudflare Realtime SFU)

Generates a 640x480 25fps synthetic video track and publishes it to a
Cloudflare Realtime SFU session.  Prints PUBLISHER_READY to stdout so the
test runner can capture sessionId + trackName for the browser subscriber.

Usage:
  export CALLS_APP_ID=...
  export CALLS_APP_SECRET=...
  python3 publisher.py                       # 150s run
  python3 publisher.py --duration 300        # 5 min run
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time

import aiohttp
import numpy as np
from av import VideoFrame
from aiortc import (
    RTCPeerConnection,
    RTCConfiguration,
    RTCIceServer,
    RTCSessionDescription,
    VideoStreamTrack,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("publisher")

SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"
STUN_SERVER = "stun:stun.cloudflare.com:3478"
TRACK_NAME = "synthetic-video"
WIDTH = 640
HEIGHT = 480
FPS = 25


# ---------------------------------------------------------------------------
# Synthetic video track — animated gradient + bouncing shape + frame counter
# ---------------------------------------------------------------------------
class SyntheticVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._counter = 0

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        self._counter += 1
        t = self._counter / FPS

        img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Animated gradient background (vectorized)
        r_vals = ((np.arange(WIDTH) / WIDTH * 255 + t * 30) % 256).astype(np.uint8)
        g_vals = ((np.arange(HEIGHT) / HEIGHT * 255 + t * 20) % 256).astype(np.uint8)
        b_val = int((128 + np.sin(t) * 60) % 256)
        img[:, :, 0] = r_vals[np.newaxis, :]
        img[:, :, 1] = g_vals[:, np.newaxis]
        img[:, :, 2] = b_val

        # Bouncing circle (red)
        cx = int(WIDTH / 2 + np.sin(t * 2.1) * (WIDTH / 3.2))
        cy = int(HEIGHT / 2 + np.cos(t * 1.4) * (HEIGHT / 3.2))
        radius = 50
        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = [220, 40, 40]

        # Frame counter progress bar
        bar_y = HEIGHT - 20
        bar_width = int(WIDTH * ((self._counter % (FPS * 5)) / (FPS * 5)))
        img[bar_y : HEIGHT, :bar_width] = [40, 220, 40]

        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


# ---------------------------------------------------------------------------
# SFU API helpers
# ---------------------------------------------------------------------------
class SFUClient:
    def __init__(self, session: aiohttp.ClientSession, app_id: str, app_secret: str):
        self.session = session
        self.base = f"{SFU_API_BASE}/{app_id}"
        self.headers = {
            "Authorization": f"Bearer {app_secret}",
            "Content-Type": "application/json",
        }

    async def new_session(self) -> str:
        async with self.session.post(
            f"{self.base}/sessions/new", headers=self.headers
        ) as resp:
            if resp.status not in (200, 201):
                text = await resp.text()
                raise RuntimeError(f"sessions/new → {resp.status}: {text}")
            data = await resp.json()
            return data["sessionId"]

    async def new_tracks(self, session_id: str, body: dict) -> dict:
        async with self.session.post(
            f"{self.base}/sessions/{session_id}/tracks/new",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status not in (200, 201):
                raise RuntimeError(f"tracks/new → {resp.status}: {text}")
            return json.loads(text)

    async def close_tracks(self, session_id: str, body: dict) -> dict:
        async with self.session.put(
            f"{self.base}/sessions/{session_id}/tracks/close",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                log.warning(f"tracks/close → {resp.status}: {text}")
                return {"status": resp.status, "text": text}
            return json.loads(text)


# ---------------------------------------------------------------------------
# Main publisher
# ---------------------------------------------------------------------------
async def run_publisher(duration: int):
    app_id = os.environ.get("CALLS_APP_ID")
    app_secret = os.environ.get("CALLS_APP_SECRET")
    if not app_id or not app_secret:
        log.error("CALLS_APP_ID and CALLS_APP_SECRET must be set in environment.")
        sys.exit(1)

    # Track info to report
    publisher_session_id = None
    publisher_track_mid = None
    track_name = TRACK_NAME

    pc = None
    connector = aiohttp.TCPConnector()

    async with aiohttp.ClientSession(connector=connector) as http_session:
        sfu = SFUClient(http_session, app_id, app_secret)

        # 1. Create SFU session
        publisher_session_id = await sfu.new_session()
        log.info(f"SFU session: {publisher_session_id}")

        # 2. Create PeerConnection with STUN
        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=STUN_SERVER)]
        )
        pc = RTCPeerConnection(configuration=config)

        connection_state = {"value": "new"}
        ice_state = {"value": "new"}

        @pc.on("connectionstatechange")
        async def on_conn_state():
            connection_state["value"] = pc.connectionState
            log.info(f"Connection state: {pc.connectionState}")

        @pc.on("iceconnectionstatechange")
        async def on_ice_state():
            ice_state["value"] = pc.iceConnectionState
            log.info(f"ICE state: {pc.iceConnectionState}")

        # 3. Add synthetic video track
        video_track = SyntheticVideoTrack()
        sender = pc.addTrack(video_track)

        # 4. Create offer
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        # 5. Wait for ICE gathering to complete
        log.info("Waiting for ICE gathering...")
        ice_wait_start = time.time()
        while pc.iceGatheringState != "complete":
            if time.time() - ice_wait_start > 10:
                log.warning(
                    f"ICE gathering not complete after 10s (state={pc.iceGatheringState})"
                )
                break
            await asyncio.sleep(0.2)

        sdp = pc.localDescription.sdp

        # 6. Assert SDP contains candidates
        if "a=candidate:" not in sdp:
            log.error("SDP has no ICE candidates — ICE gathering failed.")
            await pc.close()
            sys.exit(1)
        log.info("SDP contains ICE candidates.")

        if "end-of-candidates" in sdp:
            log.info("ICE gathering marked complete (end-of-candidates present).")
        else:
            log.warning("end-of-candidates marker not present in SDP (non-blocking).")

        # 7. Build tracks/new body — explicit trackName primary, autoDiscover fallback
        # Get mid from transceiver (RTCRtpSender has no .mid in aiortc)
        video_mid = "0"
        for t in pc.getTransceivers():
            if t.kind == "video":
                video_mid = t.mid or "0"
                break

        tracks_body = {
            "sessionDescription": {"sdp": sdp, "type": "offer"},
            "tracks": [
                {
                    "location": "local",
                    "mid": video_mid,
                    "trackName": track_name,
                }
            ],
        }

        log.info(f"Publishing track (mid={video_mid}, name={track_name})...")

        # 8. POST tracks/new
        resp = await sfu.new_tracks(publisher_session_id, tracks_body)

        if "sessionDescription" not in resp:
            log.error(f"tracks/new response missing sessionDescription: {json.dumps(resp)[:500]}")
            log.info("Falling back to autoDiscover...")
            tracks_body = {
                "sessionDescription": {"sdp": sdp, "type": "offer"},
                "autoDiscover": True,
            }
            resp = await sfu.new_tracks(publisher_session_id, tracks_body)
            if "sessionDescription" not in resp:
                log.error(f"autoDiscover also failed: {json.dumps(resp)[:500]}")
                await pc.close()
                sys.exit(1)

        # 9. Set remote description (answer from SFU)
        answer_sdp = resp["sessionDescription"]["sdp"]
        answer_type = resp["sessionDescription"]["type"]
        log.info(f"SFU answer type={answer_type}, ice-lite={'ice-lite' in answer_sdp}")

        answer = RTCSessionDescription(sdp=answer_sdp, type=answer_type)
        await pc.setRemoteDescription(answer)
        log.info("Remote description set.")

        # Extract track info from response
        if resp.get("tracks"):
            publisher_track_mid = resp["tracks"][0].get("mid", video_mid)
            if resp["tracks"][0].get("trackName"):
                track_name = resp["tracks"][0]["trackName"]
        else:
            publisher_track_mid = video_mid

        # 10. Wait for ICE connected
        log.info("Waiting for ICE connected...")
        ice_timeout = 30
        ice_start = time.time()
        while ice_state["value"] not in ("connected", "completed"):
            if ice_state["value"] == "failed":
                log.error("ICE connection failed!")
                await pc.close()
                sys.exit(1)
            if time.time() - ice_start > ice_timeout:
                log.error(f"ICE connection timeout ({ice_timeout}s). State: {ice_state['value']}")
                await pc.close()
                sys.exit(1)
            await asyncio.sleep(0.5)

        log.info(f"ICE connected! State: {ice_state['value']}")

        # 11. Signal readiness
        print(
            f"PUBLISHER_READY {publisher_session_id} {track_name}",
            flush=True,
        )
        log.info(f"Publisher ready: session={publisher_session_id} track={track_name}")

        # 12. Run for specified duration
        log.info(f"Publishing for {duration}s...")
        start = time.time()
        try:
            while time.time() - start < duration:
                elapsed = int(time.time() - start)
                if elapsed % 10 == 0 and elapsed > 0:
                    log.info(
                        f"Heartbeat: {elapsed}s | ICE={ice_state['value']} | "
                        f"Conn={connection_state['value']} | Frames={video_track._counter}"
                    )
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Interrupted — proceeding to teardown.")

        # 13. Teardown — close tracks with mid + force:true
        log.info("Tearing down...")
        if publisher_track_mid:
            try:
                close_resp = await sfu.close_tracks(
                    publisher_session_id,
                    {"tracks": [{"mid": publisher_track_mid}], "force": True},
                )
                log.info(
                    f"Tracks closed (mid={publisher_track_mid}): "
                    f"{json.dumps(close_resp)[:200]}"
                )
            except Exception as e:
                log.warning(f"Close failed (non-blocking): {e}")

        # 14. Close PeerConnection
        await pc.close()
        log.info("PeerConnection closed.")
        print("PUBLISHER_DONE", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Phase B1 Synthetic WebRTC Publisher")
    parser.add_argument(
        "--duration",
        type=int,
        default=150,
        help="Publishing duration in seconds (default: 150)",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run_publisher(args.duration))
    except KeyboardInterrupt:
        log.info("Interrupted.")
    except Exception as e:
        log.error(f"Fatal: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
