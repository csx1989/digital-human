#!/usr/bin/env python3
"""Option C Phase 2/3 — Modal GStreamer + NVENC WebRTC publisher.

Modes:
    --mode probe     : Verify nvh264enc is present and can encode a short clip.
    --mode synthetic : 1280x720@25fps synthetic I420 -> nvh264enc -> webrtcbin -> CF SFU.
    --mode avtr      : AVTR pipeline I420 frames -> nvh264enc -> webrtcbin -> CF SFU.

Usage:
    modal run modal_gst_publisher.py --mode probe
    modal run modal_gst_publisher.py --mode synthetic --duration 120
    modal run modal_gst_publisher.py --mode avtr --duration 120 --avatar-id arnold

The synthetic mode must pass a 120s browser subscriber test before AVTR mode is attempted.
"""

from __future__ import annotations

import argparse
import asyncio
import fractions
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import modal

# Guard non-modal imports for local parsing
_gst_available = False
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    from gi.repository import Gst, GstWebRTC, GstSdp, GLib
    _gst_available = True
except Exception:
    Gst = GstWebRTC = GstSdp = GLib = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None

try:
    import psutil
except ImportError:
    psutil = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [option-c] %(message)s", stream=sys.stdout)
logger = logging.getLogger("option-c")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
APP_NAME = "avtr-option-c-gst-publisher"
CUDA_VERSION = "12.8.1"
CUDA_TAG = f"{CUDA_VERSION}-devel-ubuntu24.04"
L40S_VOLUME_NAME = "avtr1-artifacts-l40s"
L40S_MOUNT_PATH = "/data/avtr1-l40s"
HF_CACHE_PATH = "/root/.cache/huggingface"

WIDTH = 1280
HEIGHT = 720
FPS = 25
TRACK_NAME = "option-c-nvenc-video"
STUN_SERVER = "stun://stun.cloudflare.com:3478"
SFU_API_BASE = "https://rtc.live.cloudflare.com/v1/apps"

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(f"nvidia/cuda:{CUDA_TAG}", add_python="3.12")
    .entrypoint([])
    .apt_install(
        # GStreamer runtime + dev
        "gstreamer1.0-tools",
        "gstreamer1.0-plugins-base",
        "gstreamer1.0-plugins-good",
        "gstreamer1.0-plugins-bad",
        "gstreamer1.0-plugins-ugly",
        "gstreamer1.0-libav",
        # libnice GStreamer plugin — required by webrtcbin for ICE.
        # Not bundled in gst-plugins-bad; ships as a separate package.
        "gstreamer1.0-nice",
        "libgstreamer1.0-dev",
        "libgstreamer-plugins-base1.0-dev",
        "libgstreamer-plugins-bad1.0-dev",
        # PyGObject + GObject introspection via apt (avoids fragile source build).
        # These install into the SYSTEM python3.12 site-packages and are made
        # importable from add_python's isolated interpreter via PYTHONPATH below.
        "python3-gi",
        "python3-gi-cairo",
        "gir1.2-gstreamer-1.0",
        "gir1.2-gst-plugins-bad-1.0",
        # General build/runtime deps
        "libglib2.0-0", "libgl1", "libsm6", "libxext6", "libxrender1",
        "ffmpeg", "git", "gcc", "g++", "pkg-config",
    )
    .pip_install("aiohttp>=3.11", "numpy>=2.0,<3", "psutil>=6.0")
    .env({
        "PYTHONPATH": "/usr/lib/python3/dist-packages:/opt/avtr1",
        "AVTR1_LOCAL_STORAGE": L40S_MOUNT_PATH,
    })
    # add_local_file must be last (Modal rebuilds on local file change).
    .add_local_file(
        os.path.join(os.path.dirname(__file__), "gst_sfu_client.py"),
        remote_path="/opt/avtr1/gst_sfu_client.py",
    )
)

app = modal.App(APP_NAME, image=image)
l40s_volume = modal.Volume.from_name(L40S_VOLUME_NAME, create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("omnivoice-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# Metrics collectors
# ---------------------------------------------------------------------------
@dataclass
class RunMetrics:
    cpu_percent_samples: list[float] = field(default_factory=list)
    gpu_util_samples: list[float] = field(default_factory=list)
    mem_util_samples: list[float] = field(default_factory=list)
    nvenc_util_samples: list[float] = field(default_factory=list)
    first_frame_latency_ms: float = 0.0
    offer_time_ms: float = 0.0
    ice_connect_time_ms: float = 0.0
    frames_generated: int = 0
    frames_emitted: int = 0

    def summary(self) -> dict[str, Any]:
        def avg(samples: list[float]) -> float:
            return round(sum(samples) / len(samples), 2) if samples else 0.0
        return {
            "cpu_percent_mean": avg(self.cpu_percent_samples),
            "cpu_percent_max": round(max(self.cpu_percent_samples), 2) if self.cpu_percent_samples else 0.0,
            "gpu_util_mean": avg(self.gpu_util_samples),
            "gpu_util_max": round(max(self.gpu_util_samples), 2) if self.gpu_util_samples else 0.0,
            "mem_util_mean": avg(self.mem_util_samples),
            "nvenc_util_mean": avg(self.nvenc_util_samples),
            "nvenc_util_max": round(max(self.nvenc_util_samples), 2) if self.nvenc_util_samples else 0.0,
            "first_frame_latency_ms": round(self.first_frame_latency_ms, 1),
            "offer_time_ms": round(self.offer_time_ms, 1),
            "ice_connect_time_ms": round(self.ice_connect_time_ms, 1),
            "frames_generated": self.frames_generated,
            "frames_emitted": self.frames_emitted,
        }


class MetricsCollector:
    """Sample CPU and GPU metrics in a background thread."""

    def __init__(self, interval_s: float = 1.0):
        self.interval = interval_s
        self.metrics = RunMetrics()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True, name="metrics-collector")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        proc = psutil.Process() if psutil else None
        while not self._stop.is_set():
            if proc:
                try:
                    self.metrics.cpu_percent_samples.append(proc.cpu_percent(interval=None))
                except Exception:
                    pass
            gpu_util, mem_util, nvenc_util = self._query_gpu()
            if gpu_util is not None:
                self.metrics.gpu_util_samples.append(gpu_util)
            if mem_util is not None:
                self.metrics.mem_util_samples.append(mem_util)
            if nvenc_util is not None:
                self.metrics.nvenc_util_samples.append(nvenc_util)
            time.sleep(self.interval)

    def _query_gpu(self) -> tuple[float | None, float | None, float | None]:
        """Return (gpu_util%, mem_util%, nvenc_util%) from nvidia-smi."""
        gpu_util = mem_util = nvenc_util = None
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,utilization.memory", "--format=csv,noheader,nounits"],
                text=True,
                timeout=5,
            ).strip()
            parts = out.split(",")
            if len(parts) >= 2:
                gpu_util = float(parts[0].strip())
                mem_util = float(parts[1].strip())
        except Exception:
            pass

        # Encoder utilization: try dmon in one-shot mode
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "dmon", "-s", "u", "-c", "1"],
                text=True,
                timeout=5,
            )
            for line in out.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        nvenc_util = float(parts[2].strip())
                        break
                    except ValueError:
                        continue
        except Exception:
            pass

        return gpu_util, mem_util, nvenc_util


# ---------------------------------------------------------------------------
# Synthetic I420 frame source
# ---------------------------------------------------------------------------
class SyntheticI420Source:
    """Generates animated I420 frames at 25fps for appsrc."""

    def __init__(self, width: int = WIDTH, height: int = HEIGHT, fps: int = FPS):
        self.width = width
        self.height = height
        self.fps = fps
        self.counter = 0
        self.frames_generated = 0
        self.frames_emitted = 0
        self._start = time.monotonic()

    def next_frame(self) -> np.ndarray:
        self.counter += 1
        t = self.counter / self.fps

        h, w = self.height, self.width
        y_base = np.arange(h, dtype=np.int32)[:, None]
        y_plane = ((y_base + int(t * 60)) % 256).astype(np.uint8)
        y_plane = np.broadcast_to(y_plane.copy(), (h, w)).copy()

        u_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8)
        v_plane = np.full((h // 2, w // 2), 128, dtype=np.uint8)

        cx = int(w / 2 + np.sin(t * 2.1) * (w / 3.2))
        cy = int(h / 2 + np.cos(t * 1.4) * (h / 3.2))
        radius = 80
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        y_plane[mask] = 220

        bar_y = h - 40
        bar_width = int(w * ((self.counter % (self.fps * 5)) / (self.fps * 5)))
        y_plane[bar_y:h, :bar_width] = 40

        i420 = np.concatenate([
            y_plane.ravel(),
            u_plane.ravel(),
            v_plane.ravel(),
        ]).reshape(h * 3 // 2, w).copy()

        self.frames_generated += 1
        return i420


# ---------------------------------------------------------------------------
# GStreamer synthetic publisher
# ---------------------------------------------------------------------------
class GstSyntheticPublisher:
    def __init__(self, app_id: str, app_secret: str, duration: int):
        self.app_id = app_id
        self.app_secret = app_secret
        self.duration = duration

        self.pipeline: Gst.Pipeline | None = None
        self.webrtc: Gst.Element | None = None
        self.appsrc: Gst.Element | None = None
        self.session_id: str | None = None
        self.video_mid: str = "0"

        self.ice_connected = asyncio.Event()
        self.ice_failed = asyncio.Event()
        self.offer_ready = asyncio.Event()
        self.local_sdp: str | None = None
        self._eoc = False
        self._ice_candidates: list[str] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._source = SyntheticI420Source()
        self._metrics_collector = MetricsCollector()
        self._push_thread: threading.Thread | None = None
        self._stop_push = threading.Event()
        # GLib MainLoop must run in a background thread to dispatch GStreamer
        # signals (on-negotiation-needed, on-ice-candidate, state changes).
        # Without it, asyncio callbacks never fire.
        self._glib_loop: GLib.MainLoop | None = None
        self._glib_thread: threading.Thread | None = None

    def _ts_set(self, event: asyncio.Event) -> None:
        """Thread-safe Event.set() — GStreamer callbacks fire on the GLib thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(event.set)
        else:
            event.set()

    def _on_negotiation_needed(self, element: Gst.Element) -> None:
        logger.info("on-negotiation-needed")
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, element, None)
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise: Gst.Promise, _webrtc: Gst.Element, _ud: Any) -> None:
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")

        set_promise = Gst.Promise.new()
        self.webrtc.emit("set-local-description", offer, set_promise)
        set_promise.interrupt()

        self.local_sdp = offer.sdp.as_text()
        logger.info("Offer created (local SDP: %d chars)", len(self.local_sdp))

    def _on_ice_candidate(self, _webrtc: Gst.Element, _mlineindex: int, candidate: str) -> None:
        if not candidate:
            logger.info("ICE end-of-candidates")
            self._eoc = True
            self._maybe_emit_offer()
            return
        # Collect trickle candidates for non-trickle injection into SDP.
        self._ice_candidates.append(candidate)
        logger.debug("ICE candidate: %s", candidate)

    def _on_ice_gathering_state(self, webrtc: Gst.Element, _pspec: Any) -> None:
        state = webrtc.get_property("ice-gathering-state")
        logger.info("ICE gathering state: %s", state.value_nick)
        if state == GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            self._eoc = True
            self._maybe_emit_offer()

    def _maybe_emit_offer(self) -> None:
        """When offer exists and gathering is complete, inject candidates into
        the SDP (CF Realtime SFU needs non-trickle), then signal offer_ready."""
        if not (self.local_sdp and self._eoc) or self.offer_ready.is_set():
            return
        # Extract the video mid from the SDP (webrtcbin uses e.g. "video0").
        mid_match = re.search(r"^a=mid:(.+)$", self.local_sdp, re.MULTILINE)
        if mid_match:
            self.video_mid = mid_match.group(1).strip()
            logger.info("Extracted video mid from SDP: %s", self.video_mid)
        # Inject candidates into the SDP if not already present.
        if self._ice_candidates and "a=candidate:" not in self.local_sdp:
            lines = self.local_sdp.split("\r\n")
            insert_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("m=video"):
                    insert_idx = i + 2  # after m= and c= lines
                    break
            for cand in self._ice_candidates:
                lines.insert(insert_idx, f"a=candidate:{cand}")
                insert_idx += 1
            lines.insert(insert_idx, "a=end-of-candidates")
            self.local_sdp = "\r\n".join(lines)
            logger.info("Injected %d ICE candidates into SDP", len(self._ice_candidates))
        self._ts_set(self.offer_ready)

    def _on_ice_connection_state(self, webrtc: Gst.Element, _pspec: Any) -> None:
        state = webrtc.get_property("ice-connection-state")
        logger.info("ICE connection state: %s", state.value_nick)
        if state in (GstWebRTC.WebRTCICEConnectionState.CONNECTED,
                     GstWebRTC.WebRTCICEConnectionState.COMPLETED):
            self._ts_set(self.ice_connected)
        elif state == GstWebRTC.WebRTCICEConnectionState.FAILED:
            self._ts_set(self.ice_failed)

    def _build_pipeline(self) -> Gst.Pipeline:
        # Full parse_launch pipeline. This pattern is confirmed working with
        # webrtcbin + appsrc (gstreamer1.0-nice must be installed for ICE).
        # Source: gstwebrtc-demos + SO "webrtcbin not connecting with appsrc".
        pipe_desc = (
            f"webrtcbin name=sendrecv bundle-policy=max-bundle stun-server={STUN_SERVER} "
            "appsrc name=src is-live=true format=time do-timestamp=true ! "
            f"video/x-raw,format=I420,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
            "queue max-size-buffers=1 leaky=downstream ! "
            "nvh264enc name=nvenc gop-size=50 preset=low-latency-hq ! "
            "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
            "h264parse config-interval=-1 ! "
            "rtph264pay config-interval=-1 aggregate-mode=zero-latency pt=96 ! "
            "queue ! "
            "sendrecv."
        )
        logger.info("Pipeline:\n%s", pipe_desc)
        pipeline = Gst.parse_launch(pipe_desc)
        self.webrtc = pipeline.get_by_name("sendrecv")
        self.appsrc = pipeline.get_by_name("src")
        return pipeline

    def _on_need_data(self, appsrc: Gst.Element, _length: int) -> None:
        """No-op: frame pushing is driven by the producer thread in push mode."""
        return

    def _on_enough_data(self, _appsrc: Gst.Element) -> None:
        pass

    def _push_frames(self) -> None:
        """Producer thread: feed I420 frames into appsrc at 25fps (push mode).

        Canonical appsrc push pattern: the app paces via wall-clock and calls
        push-buffer. The first pushed frame kicks off WebRTC negotiation.
        """
        frame_duration_ns = int(1_000_000_000 / self._source.fps)
        next_deadline = time.monotonic()
        while not self._stop_push.is_set():
            frame_data = self._source.next_frame()
            buf = Gst.Buffer.new_wrapped(frame_data.tobytes())
            buf.pts = self._source.counter * frame_duration_ns
            buf.duration = frame_duration_ns
            ret = self.appsrc.emit("push-buffer", buf)
            if ret == Gst.FlowReturn.OK:
                self._source.frames_emitted += 1
            else:
                logger.warning("appsrc push-buffer returned %s", ret)
            # Wall-clock pacing to true 25fps
            next_deadline += 1.0 / self._source.fps
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_deadline = time.monotonic()  # fell behind, reset

    async def run(self) -> dict[str, Any]:
        self._loop = asyncio.get_running_loop()

        self.pipeline = self._build_pipeline()
        self.webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("notify::ice-gathering-state", self._on_ice_gathering_state)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_connection_state)
        # need-data drives frame pushing; pushing the first frame also kicks
        # off WebRTC negotiation (appsrc live source).
        self.appsrc.connect("need-data", self._on_need_data)
        self.appsrc.connect("enough-data", self._on_enough_data)

        t0 = time.monotonic()
        # Start GLib MainLoop in a background thread so GStreamer signals fire.
        self._glib_loop = GLib.MainLoop()
        self._glib_thread = threading.Thread(target=self._glib_loop.run, daemon=True, name="glib-mainloop")
        self._glib_thread.start()
        logger.info("GLib MainLoop started in background thread")

        self.pipeline.set_state(Gst.State.PLAYING)
        self._metrics_collector.start()

        # Start the producer thread immediately — the first pushed frame kicks
        # off WebRTC negotiation (appsrc live source needs data to trigger it).
        self._push_thread = threading.Thread(target=self._push_frames, daemon=True, name="frame-pusher")
        self._push_thread.start()

        try:
            await asyncio.wait_for(self.offer_ready.wait(), timeout=30)
        except asyncio.TimeoutError:
            return {"error": "offer_timeout", "sdp": self.local_sdp}

        if not self.local_sdp or "a=candidate:" not in self.local_sdp:
            return {"error": "no_ice_candidates", "sdp": self.local_sdp}

        offer_time_ms = (time.monotonic() - t0) * 1000
        self._metrics_collector.metrics.offer_time_ms = offer_time_ms
        logger.info("Offer ready in %.1fms", offer_time_ms)

        # SDP exchange
        from gst_sfu_client import SfuClient, require_sfu_creds
        app_id, app_secret = require_sfu_creds()
        async with aiohttp.ClientSession() as http:
            sfu = SfuClient(http, app_id, app_secret)
            self.session_id = await sfu.new_session()
            logger.info("SFU session: %s", self.session_id)

            tracks_body = {
                "sessionDescription": {"sdp": self.local_sdp, "type": "offer"},
                "tracks": [{
                    "location": "local",
                    "mid": self.video_mid,
                    "trackName": TRACK_NAME,
                    "kind": "video",
                }],
            }
            try:
                answer_data = await sfu.new_tracks(self.session_id, tracks_body)
            except Exception as exc:
                logger.error("tracks/new rejected: %s", exc)
                return {
                    "error": f"tracks/new rejected: {exc}",
                    "offer_sdp": self.local_sdp,
                }

        answer_sdp = answer_data["sessionDescription"]["sdp"]
        answer_type = answer_data["sessionDescription"]["type"]
        logger.info("SFU answer type=%s", answer_type)

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

        ice_connect_time_ms = (time.monotonic() - t0) * 1000
        self._metrics_collector.metrics.ice_connect_time_ms = ice_connect_time_ms
        self._metrics_collector.metrics.first_frame_latency_ms = ice_connect_time_ms
        logger.info("ICE connected in %.1fms", ice_connect_time_ms)
        print(f"PUBLISHER_READY {self.session_id} {TRACK_NAME}", flush=True)

        # Stream
        start = time.monotonic()
        try:
            while time.monotonic() - start < self.duration:
                await asyncio.sleep(1)
                elapsed = int(time.monotonic() - start)
                if elapsed % 10 == 0:
                    logger.info(
                        "heartbeat %ss: gen=%s emit=%s",
                        elapsed,
                        self._source.frames_generated,
                        self._source.frames_emitted,
                    )
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Stream interrupted, proceeding to clean teardown")

        # Teardown
        self._stop_push.set()
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=3)
        self._metrics_collector.stop()

        await self._close_track()
        self.pipeline.set_state(Gst.State.NULL)

        # Stop GLib MainLoop
        if self._glib_loop:
            self._glib_loop.quit()
        if self._glib_thread and self._glib_thread.is_alive():
            self._glib_thread.join(timeout=2)

        metrics = self._metrics_collector.metrics
        metrics.frames_generated = self._source.frames_generated
        metrics.frames_emitted = self._source.frames_emitted
        return {
            "session_id": self.session_id,
            "encoder": "nvh264enc",
            "duration": self.duration,
            **metrics.summary(),
        }

    async def _close_track(self) -> None:
        if not self.session_id:
            return
        from gst_sfu_client import SfuClient
        async with aiohttp.ClientSession() as http:
            sfu = SfuClient(http, self.app_id, self.app_secret)
            try:
                await sfu.close_tracks(self.session_id, {"tracks": [{"mid": self.video_mid}], "force": True})
                logger.info("tracks/close OK")
            except Exception as exc:
                logger.warning("tracks/close failed: %s", exc)


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------
@app.function(
    gpu="L40S",
    timeout=600,
)
def probe_nvenc() -> dict[str, Any]:
    """Gate A: verify nvh264enc plugin exists and can encode.
    Also reports webrtcbin + dependency plugin status."""
    Gst.init(None)
    result: dict[str, Any] = {"plugin_found": False, "encode_smoke_test": False}

    registry = Gst.Registry.get()
    plugin = registry.find_plugin("nvcodec")
    element = registry.find_feature("nvh264enc", Gst.ElementFactory.__gtype__)
    result["plugin_found"] = plugin is not None
    result["element_found"] = element is not None

    # WebRTC dependency diagnostics
    webrtc_deps = ["nice", "webrtc", "dtls", "srtp", "sctp", "rtp", "rtpmanager"]
    missing_webrtc = [p for p in webrtc_deps if registry.find_plugin(p) is None]
    result["missing_webrtc_deps"] = missing_webrtc

    wb_factory = registry.find_feature("webrtcbin", Gst.ElementFactory.__gtype__)
    result["webrtcbin_found"] = wb_factory is not None
    # Count sink pad templates to confirm webrtcbin accepts RTP input
    if wb_factory is not None:
        n_sink_templates = 0
        for pt in wb_factory.get_static_pad_templates():
            try:
                if pt.direction == Gst.PadDirection.SINK:
                    n_sink_templates += 1
            except Exception:
                n_sink_templates += 1
        result["webrtcbin_sink_pad_templates"] = n_sink_templates

    if not element:
        return result

    # Run a short encode smoke test
    pipe_desc = (
        "videotestsrc num-buffers=250 ! "
        f"video/x-raw,format=I420,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
        "nvh264enc ! "
        "video/x-h264,profile=constrained-baseline ! "
        "h264parse ! "
        "fakesink"
    )
    pipeline = Gst.parse_launch(pipe_desc)
    t0 = time.monotonic()
    pipeline.set_state(Gst.State.PLAYING)
    bus = pipeline.get_bus()
    msg = bus.timed_pop_filtered(
        Gst.CLOCK_TIME_NONE,
        Gst.MessageType.ERROR | Gst.MessageType.EOS,
    )
    elapsed = time.monotonic() - t0
    pipeline.set_state(Gst.State.NULL)

    if msg and msg.type == Gst.MessageType.EOS:
        result["encode_smoke_test"] = True
        result["encode_smoke_time_s"] = round(elapsed, 2)
    elif msg and msg.type == Gst.MessageType.ERROR:
        err, dbg = msg.parse_error()
        result["error"] = f"{err}: {dbg}"
    else:
        result["error"] = "no EOS/ERROR message"

    return result


@app.function(
    gpu="L40S",
    timeout=600,
    secrets=[modal.Secret.from_name("avtr-calls")],
)
def run_synthetic(duration: int = 120) -> dict[str, Any]:
    Gst.init(None)
    app_id = os.environ.get("CALLS_APP_ID", "")
    app_secret = os.environ.get("CALLS_APP_SECRET", "")
    if not app_id or not app_secret:
        return {"error": "CALLS_APP_ID/CALLS_APP_SECRET not set"}

    loop = asyncio.new_event_loop()
    publisher = GstSyntheticPublisher(app_id, app_secret, duration)
    try:
        result = loop.run_until_complete(publisher.run())
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — running best-effort teardown")
        try:
            result = loop.run_until_complete(publisher._close_track())
            result = {**result, "interrupted": True} if isinstance(result, dict) else {"interrupted": True}
        except Exception:
            result = {"error": "interrupted", "close_failed": True}
    except Exception as exc:
        logger.exception("run_synthetic failed")
        result = {"error": str(exc)}
    finally:
        loop.close()
    return result


# ---------------------------------------------------------------------------
# AVTR frame handoff placeholder — Phase 3
# ---------------------------------------------------------------------------
@app.function(
    gpu="L40S",
    timeout=600,
    volumes={L40S_MOUNT_PATH: l40s_volume, HF_CACHE_PATH: hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret"), modal.Secret.from_name("avtr-calls")],
)
def run_avtr(duration: int = 120, avatar_id: str = "arnold", bg_id: str = "aloe_window") -> dict[str, Any]:
    """Phase 3: AVTR-rendered I420 frames -> GStreamer -> CF SFU.

    Implemented only after synthetic mode passes Gate C.
    """
    return {
        "error": "not_yet_implemented",
        "note": "Synthetic NVENC WebRTC must pass 120s before AVTR handoff is wired.",
        "duration": duration,
        "avatar_id": avatar_id,
        "bg_id": bg_id,
    }


@app.local_entrypoint()
def main(
    mode: str = "probe",
    duration: int = 120,
    avatar_id: str = "arnold",
    bg_id: str = "aloe_window",
):
    if mode == "probe":
        result = probe_nvenc.remote()
        print("\n=== NVENC PROBE ===")
        print(json.dumps(result, indent=2))
    elif mode == "synthetic":
        result = run_synthetic.remote(duration=duration)
        print("\n=== SYNTHETIC RUN ===")
        print(json.dumps(result, indent=2))
    elif mode == "avtr":
        result = run_avtr.remote(duration=duration, avatar_id=avatar_id, bg_id=bg_id)
        print("\n=== AVTR RUN ===")
        print(json.dumps(result, indent=2))
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
