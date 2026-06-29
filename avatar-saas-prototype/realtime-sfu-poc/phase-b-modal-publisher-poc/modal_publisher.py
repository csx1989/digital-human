#!/usr/bin/env python3
"""
Phase B2 — Modal-hosted Synthetic WebRTC Publisher (aiortc → Cloudflare SFU)

Runs the same synthetic video publisher as B1 but inside a Modal CPU container.
Proves Modal networking/runtime can publish stable 25fps WebRTC video to the SFU.

Usage:
  modal run modal_publisher.py --duration 150
"""

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

import modal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("modal-publisher")

SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"
STUN_SERVER = "stun:stun.cloudflare.com:3478"
TRACK_NAME = "synthetic-video"
WIDTH = 640
HEIGHT = 480
FPS = 25

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("aiortc", "numpy", "av", "aiohttp")
)

app = modal.App("avtr-phase-b-publisher", image=image)


class SyntheticVideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._counter = 0

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
        radius = 50
        yy, xx = np.ogrid[:HEIGHT, :WIDTH]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = [220, 40, 40]

        bar_y = HEIGHT - 20
        bar_width = int(WIDTH * ((self._counter % (FPS * 5)) / (FPS * 5)))
        img[bar_y : HEIGHT, :bar_width] = [40, 220, 40]

        frame = VideoFrame.from_ndarray(img, format="rgb24")
        frame.pts = pts
        frame.time_base = time_base
        return frame


class SFUClient:
    def __init__(self, session, app_id, app_secret):
        self.session = session
        self.base = f"{SFU_API_BASE}/{app_id}"
        self.headers = {
            "Authorization": f"Bearer {app_secret}",
            "Content-Type": "application/json",
        }

    async def new_session(self):
        async with self.session.post(
            f"{self.base}/sessions/new", headers=self.headers
        ) as resp:
            if resp.status not in (200, 201):
                raise RuntimeError(f"sessions/new → {resp.status}: {await resp.text()}")
            return (await resp.json())["sessionId"]

    async def new_tracks(self, session_id, body):
        async with self.session.post(
            f"{self.base}/sessions/{session_id}/tracks/new",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status not in (200, 201):
                raise RuntimeError(f"tracks/new → {resp.status}: {text}")
            return json.loads(text)

    async def close_tracks(self, session_id, body):
        async with self.session.put(
            f"{self.base}/sessions/{session_id}/tracks/close",
            headers=self.headers,
            json=body,
        ) as resp:
            text = await resp.text()
            if resp.status != 200:
                log.warning(f"tracks/close → {resp.status}: {text}")
                return {"status": resp.status}
            return json.loads(text)


async def _run_publisher(duration):
    app_id = os.environ.get("CALLS_APP_ID")
    app_secret = os.environ.get("CALLS_APP_SECRET")
    if not app_id or not app_secret:
        log.error("CALLS_APP_ID and CALLS_APP_SECRET must be set.")
        return

    publisher_session_id = None
    publisher_track_mid = None
    track_name = TRACK_NAME
    pc = None

    connector = aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as http_session:
        sfu = SFUClient(http_session, app_id, app_secret)

        publisher_session_id = await sfu.new_session()
        log.info(f"SFU session: {publisher_session_id}")

        config = RTCConfiguration(
            iceServers=[RTCIceServer(urls=STUN_SERVER)]
        )
        pc = RTCPeerConnection(configuration=config)

        ice_state = {"value": "new"}
        conn_state = {"value": "new"}

        @pc.on("iceconnectionstatechange")
        async def on_ice():
            ice_state["value"] = pc.iceConnectionState
            log.info(f"ICE state: {pc.iceConnectionState}")

        @pc.on("connectionstatechange")
        async def on_conn():
            conn_state["value"] = pc.connectionState
            log.info(f"Connection state: {pc.connectionState}")

        video_track = SyntheticVideoTrack()
        pc.addTrack(video_track)

        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)

        log.info("Waiting for ICE gathering...")
        ice_wait_start = time.time()
        while pc.iceGatheringState != "complete":
            if time.time() - ice_wait_start > 10:
                log.warning(f"ICE gathering not complete after 10s (state={pc.iceGatheringState})")
                break
            await asyncio.sleep(0.2)

        sdp = pc.localDescription.sdp

        if "a=candidate:" not in sdp:
            log.error("SDP has no ICE candidates — gathering failed.")
            await pc.close()
            return
        log.info("SDP contains ICE candidates.")

        if "end-of-candidates" in sdp:
            log.info("ICE gathering complete (end-of-candidates present).")
        else:
            log.warning("end-of-candidates marker not present (non-blocking).")

        video_mid = "0"
        for t in pc.getTransceivers():
            if t.kind == "video":
                video_mid = t.mid or "0"
                break

        tracks_body = {
            "sessionDescription": {"sdp": sdp, "type": "offer"},
            "tracks": [{"location": "local", "mid": video_mid, "trackName": track_name}],
        }

        log.info(f"Publishing track (mid={video_mid}, name={track_name})...")
        resp = await sfu.new_tracks(publisher_session_id, tracks_body)

        if "sessionDescription" not in resp:
            log.warning("Explicit tracks failed, falling back to autoDiscover...")
            tracks_body = {
                "sessionDescription": {"sdp": sdp, "type": "offer"},
                "autoDiscover": True,
            }
            resp = await sfu.new_tracks(publisher_session_id, tracks_body)
            if "sessionDescription" not in resp:
                log.error(f"autoDiscover failed: {json.dumps(resp)[:500]}")
                await pc.close()
                return

        answer = RTCSessionDescription(
            sdp=resp["sessionDescription"]["sdp"],
            type=resp["sessionDescription"]["type"],
        )
        await pc.setRemoteDescription(answer)
        log.info("Remote description set.")

        if resp.get("tracks"):
            publisher_track_mid = resp["tracks"][0].get("mid", video_mid)
            if resp["tracks"][0].get("trackName"):
                track_name = resp["tracks"][0]["trackName"]
        else:
            publisher_track_mid = video_mid

        log.info("Waiting for ICE connected...")
        ice_start = time.time()
        while ice_state["value"] not in ("connected", "completed"):
            if ice_state["value"] == "failed":
                log.error("ICE failed!")
                await pc.close()
                return
            if time.time() - ice_start > 30:
                log.error(f"ICE timeout. State: {ice_state['value']}")
                await pc.close()
                return
            await asyncio.sleep(0.5)

        log.info(f"ICE connected! State: {ice_state['value']}")
        print(f"PUBLISHER_READY {publisher_session_id} {track_name}", flush=True)

        log.info(f"Publishing for {duration}s...")
        start = time.time()
        try:
            while time.time() - start < duration:
                elapsed = int(time.time() - start)
                if elapsed % 10 == 0 and elapsed > 0:
                    log.info(
                        f"Heartbeat: {elapsed}s | ICE={ice_state['value']} | "
                        f"Conn={conn_state['value']} | Frames={video_track._counter}"
                    )
                await asyncio.sleep(1)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Interrupted — proceeding to teardown.")

        log.info("Tearing down...")
        if publisher_track_mid:
            try:
                close_resp = await sfu.close_tracks(
                    publisher_session_id,
                    {"tracks": [{"mid": publisher_track_mid}], "force": True},
                )
                log.info(f"Tracks closed (mid={publisher_track_mid})")
            except Exception as e:
                log.warning(f"Close failed (non-blocking): {e}")

        await pc.close()
        log.info("PeerConnection closed.")
        print("PUBLISHER_DONE", flush=True)


@app.function(
    secrets=[modal.Secret.from_name("avtr-calls")],
    cpu=1,
    timeout=300,
)
def run_publisher(duration: int = 150):
    log.info(f"=== Modal Publisher starting (duration={duration}s) ===")
    asyncio.run(_run_publisher(duration))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=150)
    args, _ = parser.parse_known_args()
    with app.run():
        run_publisher.remote(args.duration)
