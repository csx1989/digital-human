#!/usr/bin/env python3
"""Option C Phase 1 — Local GStreamer SDP compatibility probe.

On macOS and machines without an NVIDIA GPU, this uses x264enc (CPU) only to
verify that a GStreamer-generated offer SDP is accepted by Cloudflare Realtime
SFU. The real GPU/NVENC gate is Phase 2 on Modal L40S.

Usage:
    source .env
    python3 local_gst_probe.py --duration 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from typing import Any

import aiohttp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [probe] %(message)s")
logger = logging.getLogger("gst-probe")

try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    from gi.repository import Gst, GstWebRTC, GstSdp
except ImportError as exc:
    logger.error("GStreamer Python bindings not found: %s", exc)
    logger.error("macOS: brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav pygobject3")
    logger.error("Linux: sudo apt install gstreamer1.0-* libgirepository1.0-dev python3-gi")
    sys.exit(1)

SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"
STUN_SERVER = "stun://stun.cloudflare.com:3478"
TRACK_NAME = "option-c-probe-video"
WIDTH = 1280
HEIGHT = 720
FPS = 25


# ---------------------------------------------------------------------------
# WebRTC publisher using GStreamer
# ---------------------------------------------------------------------------
class GstProbePublisher:
    def __init__(self, app_id: str, app_secret: str, duration: int, use_nvenc: bool = False):
        self.app_id = app_id
        self.app_secret = app_secret
        self.duration = duration
        self.use_nvenc = use_nvenc

        self.pipeline: Gst.Pipeline | None = None
        self.webrtc: Gst.Element | None = None
        self.session_id: str | None = None
        self.video_mid: str = "0"
        self.ice_connected = asyncio.Event()
        self.ice_failed = asyncio.Event()
        self.offer_ready = asyncio.Event()
        self.local_sdp: str | None = None
        self._ice_candidates: list[str] = []
        self._eoc = False
        self._loop: asyncio.AbstractEventLoop | None = None

    def _check_plugins(self) -> list[str]:
        needed = ["nice", "webrtc", "dtls", "srtp", "rtp", "rtpmanager", "videotestsrc", "x264enc"]
        if self.use_nvenc:
            needed.append("nvh264enc")
        registry = Gst.Registry.get()
        missing = [p for p in needed if registry.find_plugin(p) is None]
        return missing

    def _build_pipeline(self) -> Gst.Pipeline:
        """Construct a single-video sendonly pipeline."""
        # Pick encoder based on availability/flag
        encoder = "nvh264enc" if self.use_nvenc else "x264enc"
        if encoder == "x264enc":
            enc_block = (
                f"x264enc speed-preset=veryfast tune=zerolatency key-int-max=50 "
                f"! video/x-h264,profile=constrained-baseline,level=(string)3.1"
            )
        else:
            enc_block = (
                f"nvh264enc gop-size=50 "
                f"! video/x-h264,profile=constrained-baseline,level=(string)3.1"
            )

        pipe_desc = (
            f"webrtcbin name=sendrecv bundle-policy=max-bundle stun-server={STUN_SERVER} "
            f"videotestsrc is-live=true pattern=ball ! "
            f"video/x-raw,width={WIDTH},height={HEIGHT},format=I420,framerate={FPS}/1 ! "
            f"{enc_block} ! "
            f"h264parse config-interval=-1 ! "
            f"rtph264pay config-interval=-1 aggregate-mode=zero-latency pt=96 ! "
            f"application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000 ! "
            f"sendrecv."
        )
        logger.info("Pipeline:\n%s", pipe_desc)
        pipeline = Gst.parse_launch(pipe_desc)
        self.webrtc = pipeline.get_by_name("sendrecv")
        return pipeline

    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        logger.info("on-negotiation-needed: creating offer")
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, element, None)
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise: Gst.Promise, _webrtc: Gst.Element, _ud: Any) -> None:
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        promise.unref()

        promise = Gst.Promise.new()
        self.webrtc.emit("set-local-description", offer, promise)
        promise.interrupt()

        self.local_sdp = offer.sdp.as_text()
        logger.info("Offer created (mid=0 video)")

    def _on_ice_candidate(self, _webrtc: Gst.Element, mlineindex: int, candidate: str) -> None:
        if not candidate:
            logger.info("ICE end-of-candidates signaled")
            self._eoc = True
            self._maybe_emit_offer()
            return
        self._ice_candidates.append(candidate)
        logger.debug("ICE candidate: %s", candidate)

    def _on_ice_gathering_state(self, webrtc: Gst.Element, _pspec: Any) -> None:
        state = webrtc.get_property("ice-gathering-state")
        logger.info("ICE gathering state: %s", state.value_nick)
        if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            self._eoc = True
            self._maybe_emit_offer()

    def _maybe_emit_offer(self) -> None:
        if self.local_sdp and self._eoc and not self.offer_ready.is_set():
            # Insert gathered candidates into local SDP if not already present
            sdp = self.local_sdp
            if self._ice_candidates and "a=candidate:" not in sdp:
                lines = sdp.split("\r\n")
                # Insert after the m=video section's c= line
                insert_idx = len(lines)
                for i, line in enumerate(lines):
                    if line.startswith("m=video"):
                        insert_idx = i + 2  # after m= and c=
                        break
                for cand in self._ice_candidates:
                    lines.insert(insert_idx, f"a=candidate:{cand}")
                    insert_idx += 1
                lines.insert(insert_idx, "a=end-of-candidates")
                sdp = "\r\n".join(lines)
                self.local_sdp = sdp
            self.offer_ready.set()

    def _on_ice_connection_state(self, webrtc: Gst.Element, _pspec: Any) -> None:
        state = webrtc.get_property("ice-connection-state")
        logger.info("ICE connection state: %s", state.value_nick)
        if state in (GstWebRTC.WebRTCICEConnectionState.CONNECTED,
                     GstWebRTC.WebRTCICEConnectionState.COMPLETED):
            self.ice_connected.set()
        elif state == GstWebRTC.WebRTCICEConnectionState.FAILED:
            self.ice_failed.set()

    async def run(self) -> dict[str, Any]:
        self._loop = asyncio.get_running_loop()

        missing = self._check_plugins()
        if missing:
            logger.error("Missing GStreamer plugins: %s", missing)
            return {"error": f"missing_plugins: {missing}"}

        self.pipeline = self._build_pipeline()
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("notify::ice-gathering-state", self._on_ice_gathering_state)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_connection_state)

        self.pipeline.set_state(Gst.State.PLAYING)

        # Wait for offer + ICE gathering complete
        try:
            await asyncio.wait_for(self.offer_ready.wait(), timeout=15)
        except asyncio.TimeoutError:
            logger.error("Timeout waiting for offer/ICE gathering")
            return {"error": "offer_timeout"}

        if not self.local_sdp or "a=candidate:" not in self.local_sdp:
            logger.error("No ICE candidates in local SDP")
            return {"error": "no_ice_candidates"}

        # Create SFU session
        headers = {"Authorization": f"Bearer {self.app_secret}", "Content-Type": "application/json"}
        async with aiohttp.ClientSession() as http:
            async with http.post(f"{SFU_API_BASE}/{self.app_id}/sessions/new", headers=headers) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    return {"error": f"sessions/new {resp.status}", "detail": text}
                self.session_id = (await resp.json())["sessionId"]
            logger.info("SFU session: %s", self.session_id)

            # POST tracks/new
            tracks_body = {
                "sessionDescription": {"sdp": self.local_sdp, "type": "offer"},
                "tracks": [{
                    "location": "local",
                    "mid": self.video_mid,
                    "trackName": TRACK_NAME,
                    "kind": "video",
                }],
            }
            async with http.post(
                f"{SFU_API_BASE}/{self.app_id}/sessions/{self.session_id}/tracks/new",
                headers=headers,
                json=tracks_body,
            ) as resp:
                text = await resp.text()
                if resp.status != 200:
                    logger.error("tracks/new rejected: %s %s", resp.status, text)
                    return {"error": f"tracks/new {resp.status}", "detail": text, "offer_sdp": self.local_sdp}
                answer_data = json.loads(text)

        answer_sdp = answer_data["sessionDescription"]["sdp"]
        answer_type = answer_data["sessionDescription"]["type"]
        logger.info("SFU answer type=%s", answer_type)

        # Set remote description
        res, sdpmsg = GstSdp.SDPMessage.new()
        GstSdp.sdp_message_parse_buffer(bytes(answer_sdp.encode()), sdpmsg)
        answer = GstWebRTC.WebRTCSessionDescription.new(
            GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
        )
        promise = Gst.Promise.new()
        self.webrtc.emit("set-remote-description", answer, promise)
        promise.interrupt()

        # Wait for ICE connected
        try:
            await asyncio.wait_for(self.ice_connected.wait(), timeout=30)
        except asyncio.TimeoutError:
            return {"error": "ice_connect_timeout"}
        if self.ice_failed.is_set():
            return {"error": "ice_failed"}

        logger.info("ICE connected — streaming for %ss", self.duration)
        start = time.monotonic()
        while time.monotonic() - start < self.duration:
            await asyncio.sleep(1)

        # Teardown
        await self._close_track()
        self.pipeline.set_state(Gst.State.NULL)

        return {
            "session_id": self.session_id,
            "ice_connected": True,
            "duration": self.duration,
            "encoder": "nvh264enc" if self.use_nvenc else "x264enc",
        }

    async def _close_track(self) -> None:
        if not self.session_id:
            return
        headers = {"Authorization": f"Bearer {self.app_secret}", "Content-Type": "application/json"}
        body = {"tracks": [{"mid": self.video_mid}], "force": True}
        async with aiohttp.ClientSession() as http:
            async with http.put(
                f"{SFU_API_BASE}/{self.app_id}/sessions/{self.session_id}/tracks/close",
                headers=headers,
                json=body,
            ) as resp:
                text = await resp.text()
                logger.info("tracks/close: %s %s", resp.status, text[:200])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Option C local GStreamer SDP probe")
    parser.add_argument("--duration", type=int, default=30, help="Stream duration in seconds")
    parser.add_argument("--use-nvenc", action="store_true", help="Try nvh264enc instead of x264enc")
    args = parser.parse_args()

    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        logger.error("CALLS_APP_ID and CALLS_APP_SECRET must be set")
        sys.exit(1)

    Gst.init(None)
    loop = asyncio.new_event_loop()
    publisher = GstProbePublisher(app_id, app_secret, args.duration, args.use_nvenc)
    result = loop.run_until_complete(publisher.run())
    print("\n=== PROBE RESULT ===")
    print(json.dumps(result, indent=2))
    sys.exit(0 if "error" not in result else 1)


if __name__ == "__main__":
    main()
