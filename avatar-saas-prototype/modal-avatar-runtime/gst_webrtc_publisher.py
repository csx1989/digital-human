"""GStreamer NVENC WebRTC publisher for Modal — production SFU integration.

Phase 2 standalone module. NOT imported by production runtime yet (Phase 3
will wire it into modal_lifecycle_..._5q.py behind AVTR_SFU_PUBLISHER=gstreamer).

Replaces the aiortc-based WebRtcPublisher with a GStreamer/NVENC pipeline that
delivers ~38x lower CPU (11% vs 420%). The interface matches WebRtcPublisher
so the 5Q session handler can switch publishers with minimal change.

Key design principles:
  - SDP exchange is brokered through Worker callbacks (publisher-offer,
    publisher-close). NEVER calls Cloudflare SFU directly — no SFU secrets.
  - gi/GStreamer imported lazily inside _import_gstreamer(), so this module
    is importable and py_compile-able on machines without GStreamer installed.
  - Video-only (H264 NVENC). Audio stays on WebSocket for Milestone 1.
  - Explicit pre-roll: before ICE connects, the publisher pushes cached black
    I420 frames so GPU rendering work does not start during SDP negotiation.
  - Fail-fast: sustained underflow or producer error trips a policy that
    stops pushing and signals EOS — never emits frozen frames forever.

Encoder profile (validated in Option C POC, 3/3 canaries passed;
  hardened for SFU startup stability — CBR-ld-hq rate control eliminates
  the VBR startup bitrate spike that caused front-loaded packet loss):
  1280x720, H264 NVENC, bitrate=4000 kbit/sec (overridable via
  AVTR_SFU_BITRATE), gop-size=50 (overridable via AVTR_SFU_GOP_SIZE),
  constrained-baseline, level 3.1, preset=low-latency-hq,
  rc-mode=cbr-ld-hq, strict-gop=true, zerolatency=true.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Protocol, runtime_checkable
from urllib.parse import quote as _url_quote

import httpx

logger = logging.getLogger("gst-webrtc-pub")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[gst-pub] %(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


# ── Crash forensics ──────────────────────────────────────────────────────────

_CRASH_FORENSICS_INSTALLED = False
_CRASH_CONTEXT: dict[str, Any] = {
    "session_id": "(unknown)",
    "phase": "(not started)",
    "frames_generated": 0,
    "frames_emitted": 0,
    "last_heartbeat": "",
    "publisher_mode": "",
    "transport": "",
    "ice_connection_state": "",
    "connection_state": "",
}


def _update_crash_context(**kwargs: Any) -> None:
    _CRASH_CONTEXT.update(kwargs)


def _install_crash_forensics() -> None:
    global _CRASH_FORENSICS_INSTALLED
    if _CRASH_FORENSICS_INSTALLED:
        return
    _CRASH_FORENSICS_INSTALLED = True

    import faulthandler
    import signal
    import sys
    import traceback as _tb_mod

    faulthandler.enable(all_threads=True)
    try:
        faulthandler.register(signal.SIGTERM, chain=True)
    except (ValueError, OSError):
        pass
    try:
        signal.SIGUSR1
        faulthandler.register(signal.SIGUSR1, chain=False)
    except (AttributeError, ValueError, OSError):
        pass

    _orig_excepthook = sys.excepthook

    def _crash_excepthook(exc_type, exc_value, exc_tb):
        try:
            logger.error(
                "[CRASH-FORENSICS] Uncaught exception: %s | context=%s",
                exc_type.__name__ if exc_type else "?",
                json.dumps(_CRASH_CONTEXT, default=str),
            )
            _tb_mod.print_exception(exc_type, exc_value, exc_tb)
            sys.stderr.flush()
        except Exception:
            pass
        _orig_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_excepthook

    _orig_thread_excepthook = threading.excepthook

    def _crash_thread_excepthook(args):
        try:
            logger.error(
                "[CRASH-FORENSICS] Uncaught thread exception: thread=%s exc=%s | context=%s",
                args.thread.name if args.thread else "?",
                args.exc_type.__name__ if args.exc_type else "?",
                json.dumps(_CRASH_CONTEXT, default=str),
            )
            _tb_mod.print_exception(args.exc_type, args.exc_value, args.exc_traceback)
            sys.stderr.flush()
        except Exception:
            pass

    threading.excepthook = _crash_thread_excepthook

    logger.info(
        "[CRASH-FORENSICS] Installed faulthandler=%s",
        faulthandler.is_enabled(),
    )

# ── Constants ────────────────────────────────────────────────────────────────

def _env_int(name: str, default: int, *, min_val: int | None = None) -> int:
    """Read an integer env var with validation and fallback."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (ValueError, TypeError):
        value = default
    if min_val is not None:
        value = max(min_val, value)
    return value


WIDTH = 1280
HEIGHT = 720
FPS = 25
BITRATE_DEFAULT = _env_int("AVTR_SFU_BITRATE", 4000, min_val=500)
GOP_SIZE = _env_int("AVTR_SFU_GOP_SIZE", 50, min_val=1)
RTP_AGGREGATE_MODE = os.environ.get("AVTR_SFU_RTP_AGGREGATE_MODE", "zero-latency")
STUN_SERVER = "stun://stun.cloudflare.com:3478"

ICE_GATHER_TIMEOUT_S = 30


def _ice_connect_timeout_s() -> int:
    """Per-attempt ICE connect timeout, configurable via env.

    Production default is 10s (not 30s) because the retry wrapper handles
    transient failures. Lower per-attempt budget + retry = faster recovery.
    """
    try:
        return max(5, int(os.environ.get("AVTR_SFU_ICE_CONNECT_TIMEOUT_S", "10")))
    except (ValueError, TypeError):
        return 10


def _max_publisher_attempts() -> int:
    """Max publisher startup attempts, configurable via env. Default 2."""
    try:
        return max(1, int(os.environ.get("AVTR_SFU_PUBLISHER_MAX_ATTEMPTS", "2")))
    except (ValueError, TypeError):
        return 2


def _direct_webrtc_answer_timeout_s() -> float:
    """Seconds to wait for the browser's answer SDP in direct WebRTC debug mode.

    Must be long enough for the browser to poll through a Modal cold start
    (which can take 30-40s) and still create/send an answer.
    """
    try:
        return max(15, float(os.environ.get("AVTR_DIRECT_WEBRTC_ANSWER_TIMEOUT_S", "45")))
    except (ValueError, TypeError):
        return 45

_ENCODER_INFO: dict[str, Any] = {
    "encoder": "nvh264enc",
    "profile": "constrained-baseline",
    "level": "3.1",
    "width": WIDTH,
    "height": HEIGHT,
    "fps": FPS,
    "gop_size": GOP_SIZE,
    "rtp_aggregate_mode": RTP_AGGREGATE_MODE,
    "direct_webrtc_debug": False,
}

_PIPELINE_TEMPLATE = (
    "webrtcbin name=sendrecv bundle-policy=max-bundle "
    "stun-server={stun} "
    "appsrc name=src is-live=true format=time do-timestamp=false ! "
    "video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
    "queue max-size-buffers=3 leaky=downstream ! "
    "nvh264enc name=nvenc gop-size={gop} preset=low-latency-hq bitrate={bitrate} "
    "rc-mode=cbr-ld-hq strict-gop=true zerolatency=true ! "
    "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
    "h264parse name=h264p config-interval=-1 ! "
    "rtph264pay name=rtppay config-interval=-1 aggregate-mode={agg_mode} pt=96 ! "
    "queue ! "
    "sendrecv."
)

_PIPELINE_TEMPLATE_X264 = (
    "webrtcbin name=sendrecv bundle-policy=max-bundle "
    "stun-server={stun} "
    "appsrc name=src is-live=true format=time do-timestamp=false ! "
    "video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
    "queue max-size-buffers=3 leaky=downstream ! "
    "x264enc name=enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate} "
    "key-int-max={gop} ref=1 ! "
    "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
    "h264parse name=h264p config-interval=-1 ! "
    "rtph264pay name=rtppay config-interval=-1 aggregate-mode={agg_mode} pt=96 ! "
    "queue ! "
    "sendrecv."
)

_PIPELINE_TEMPLATE_ENCODE_ONLY = (
    "appsrc name=src is-live=true format=time do-timestamp=false ! "
    "video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
    "queue max-size-buffers=3 leaky=downstream ! "
    "nvh264enc name=nvenc gop-size={gop} preset=low-latency-hq bitrate={bitrate} "
    "rc-mode=cbr-ld-hq strict-gop=true zerolatency=true ! "
    "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
    "h264parse name=h264p config-interval=-1 ! "
    "fakesink sync=false"
)

_PIPELINE_TEMPLATE_ENCODE_ONLY_X264 = (
    "appsrc name=src is-live=true format=time do-timestamp=false ! "
    "video/x-raw,format=I420,width={width},height={height},framerate={fps}/1 ! "
    "queue max-size-buffers=3 leaky=downstream ! "
    "x264enc name=enc speed-preset=ultrafast tune=zerolatency bitrate={bitrate} "
    "key-int-max={gop} ref=1 ! "
    "video/x-h264,profile=constrained-baseline,level=(string)3.1 ! "
    "h264parse name=h264p config-interval=-1 ! "
    "fakesink sync=false"
)


# ── Pre-roll black frame ─────────────────────────────────────────────────────

def _black_i420_bytes(width: int = WIDTH, height: int = HEIGHT) -> bytes:
    """Return a black I420 frame as raw bytes (Y=16, U=128, V=128).

    No numpy dependency — constructed from raw bytes for the pre-roll phase.
    Layout: Y plane (width*height, all 0x10) + U plane (w/2*h/2, all 0x80)
    + V plane (w/2*h/2, all 0x80).
    """
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    return b"\x10" * y_size + b"\x80" * (uv_size * 2)


# ── Lazy GStreamer import ────────────────────────────────────────────────────

def _import_gstreamer() -> tuple[Any, Any, Any, Any]:
    """Lazy import of gi/GStreamer modules.

    Called from start() only. Uses importlib so the literal strings 'import gi'
    and 'from gi' never appear at module top level. This keeps the module
    importable on machines without GStreamer (fMP4-only environments).

    The authoritative guard is the AST-based test_no_top_level_gi_import().
    """
    import importlib

    if os.environ.get("AVTR_WHEP_CRASH_DEBUG", "0") == "1":
        categories = (
            "webrtcbin:5,nvh264enc:4,nice:5,libnice:5,"
            "GST_PADS:4,GST_STATES:4,dtls:5,appsrc:4,rtph264pay:4"
        )
        existing = os.environ.get("GST_DEBUG", "")
        os.environ["GST_DEBUG"] = f"{existing},{categories}" if existing else categories
        logger.info("[CRASH-DEBUG] GST_DEBUG enabled: %s", categories)

    gi = importlib.import_module("gi")
    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    namespace: dict[str, Any] = {}
    exec("from gi.repository import Gst, GstWebRTC, GstSdp, GLib", namespace)
    Gst = namespace["Gst"]
    GstWebRTC = namespace["GstWebRTC"]
    GstSdp = namespace["GstSdp"]
    GLib = namespace["GLib"]
    Gst.init(None)
    return Gst, GstWebRTC, GstSdp, GLib


# ── Frame Source Protocol ────────────────────────────────────────────────────

@runtime_checkable
class FrameSource(Protocol):
    """Structural protocol for frame sources (AVTR render, synthetic, test fake).

    Before start() is called, the source MUST NOT do GPU work.
    After start() is called, next_frame() yields real frames.
    The publisher calls start() only after ICE connects (delayed activation).
    """

    fps: int
    counter: int
    frames_generated: int
    frames_emitted: int
    underflow_count: int

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def next_frame(self) -> Any: ...
    def get_error(self) -> Exception | None: ...


# ── Synthetic frame source (crash isolation matrix) ──────────────────────────

_SYNTHETIC_RING_SIZE = 30


class SyntheticFrameSource:
    """Deterministic CPU-only I420 frame source for crash isolation.

    Precomputes a ring of scrolling color-bar I420 frames at start() using
    NumPy vectorized operations. next_frame() returns prebuilt bytes — no
    per-frame computation. Designed to sustain true 25fps at 1280x720 with
    zero GPU dependency.

    Implements the FrameSource protocol for matrix rows B/E.
    """

    fps: int = FPS
    counter: int = 0
    frames_generated: int = 0
    frames_emitted: int = 0
    underflow_count: int = 0

    def __init__(self, width: int = WIDTH, height: int = HEIGHT) -> None:
        self._w = width
        self._h = height
        self._stop_event = False
        self._error: Exception | None = None
        self._y_size = width * height
        self._uv_h = height // 2
        self._uv_w = width // 2
        self._uv_size = self._uv_h * self._uv_w
        self._frame_size = self._y_size + 2 * self._uv_size
        self._ring: list[bytes] = []
        self._started = False

    def _build_ring(self) -> None:
        import numpy as np

        w = self._w
        h = self._h
        uv_w = self._uv_w
        uv_h = self._uv_h
        bar_width = max(1, w // 8)

        colors_y = np.array([235, 170, 82, 65, 30, 160, 41, 90], dtype=np.uint8)
        colors_u = np.array([128, 166, 90, 100, 120, 70, 220, 90], dtype=np.uint8)
        colors_v = np.array([128, 16, 240, 210, 210, 100, 110, 240], dtype=np.uint8)

        x_indices = np.arange(w)
        bar_indices = (x_indices // bar_width) % 8
        base_y_row = colors_y[bar_indices]

        x_uv_indices = np.arange(uv_w)
        uv_bar_width = max(1, uv_w // 8)
        bar_uv_indices = (x_uv_indices // uv_bar_width) % 8
        base_u_row = colors_u[bar_uv_indices]
        base_v_row = colors_v[bar_uv_indices]

        base_y_plane = np.tile(base_y_row, (h, 1))
        base_u_plane = np.tile(base_u_row, (uv_h, 1))
        base_v_plane = np.tile(base_v_row, (uv_h, 1))

        self._ring = []
        for i in range(_SYNTHETIC_RING_SIZE):
            scroll = (i * 4) % w
            y_plane = np.roll(base_y_plane, scroll, axis=1)
            u_plane = np.roll(base_u_plane, scroll // 2, axis=1)
            v_plane = np.roll(base_v_plane, scroll // 2, axis=1)
            frame_bytes = y_plane.tobytes() + u_plane.tobytes() + v_plane.tobytes()
            self._ring.append(frame_bytes)

    def start(self) -> None:
        if self._started:
            return
        self._build_ring()
        self._started = True
        self._stop_event = False
        self.counter = 0
        self.frames_generated = 0
        self.frames_emitted = 0
        self.underflow_count = 0
        logger.info(
            "[synthetic] ring precomputed: %d frames, %d bytes each",
            len(self._ring), self._frame_size,
        )

    def stop(self) -> None:
        self._stop_event = True

    def get_error(self) -> Exception | None:
        return self._error

    @property
    def metrics(self) -> Any:
        class _M:
            queue_depth_min = 0
            queue_depth_max = 0
        return _M()

    def next_frame(self) -> bytes:
        if not self._ring:
            return b"\x10" * self._frame_size
        frame = self._ring[self.counter % len(self._ring)]
        self.counter += 1
        self.frames_generated += 1
        return frame


# ── Fail-Fast Policy ─────────────────────────────────────────────────────────

# After the producer thread dies (e.g. CUDA illegal memory access), the source
# may keep returning the last good frame *without* incrementing underflow_count.
# Without fail-fast, this produces 25fps of frozen video forever — the browser
# sees it as "healthy".  These bounds make the track fail closed instead.
#
#   MAX_REPEATS_AFTER_ERROR: once a producer error is observed, EVERY frame
#       (underflow or not) counts toward the grace drain. After this many
#       frames the policy trips with 'producer_error_drained'. ~1s at 25fps.
#       This handles sources that return cached/repeated frames without
#       incrementing underflow_count.
#
#   MAX_CONSECUTIVE_UNDERFLOWS: even without an explicit error, this many
#       consecutive underflow frames (~10s at 25fps) is a silent stall.

MAX_REPEATS_AFTER_ERROR = 25
MAX_CONSECUTIVE_UNDERFLOWS = 250


class _FailFastPolicy:
    """Tracks underflow/error conditions and trips when fail-fast is needed.

    Pure Python — no GStreamer dependency. Fully unit-testable in isolation.

    Two INDEPENDENT trip conditions:
      1. Producer error + grace drain: once record_producer_error() is called,
         EVERY frame (underflow or not) counts toward the grace drain. After
         max_repeats frames, trips with 'producer_error_drained'. This catches
         sources that return cached/repeated frames without incrementing
         underflow_count — the publisher stops rather than emitting frozen
         video forever.
      2. Sustained underflow: max_underflows consecutive underflow frames
         trips with 'sustained_underflow'. The counter resets on any
         non-underflow frame.

    Transient underflow (intermittent, counter resets) never trips.
    """

    def __init__(
        self,
        *,
        max_repeats: int = MAX_REPEATS_AFTER_ERROR,
        max_underflows: int = MAX_CONSECUTIVE_UNDERFLOWS,
    ) -> None:
        self._max_repeats = max_repeats
        self._max_underflows = max_underflows
        self._consecutive_underflows: int = 0
        self._frames_since_error: int = 0
        self._producer_error: Exception | None = None
        self._tripped: bool = False
        self.fail_fast_reason: str | None = None

    def record_producer_error(self, exc: Exception) -> None:
        """Record the first producer error. Subsequent calls are ignored."""
        if self._producer_error is None:
            self._producer_error = exc

    def record_frame(self, *, underflow: bool) -> str | None:
        """Record a live-phase frame attempt. Returns trip reason or None.

        Must be called once per frame in the live phase.

        Args:
            underflow: True if the source's underflow_count increased for
                this frame read, indicating a queue stall.

        Fail-fast logic:
          1. If a producer error has been recorded, EVERY frame counts toward
             the grace drain (max_repeats) — regardless of underflow status.
             This catches sources that return cached/repeated frames without
             incrementing underflow_count.
          2. Consecutive underflows are tracked independently. max_underflows
             in a row trips with 'sustained_underflow'. The counter resets on
             any non-underflow frame (regardless of producer error state).
        """
        if self._tripped:
            return self.fail_fast_reason

        # Policy 1: producer error → grace drain on ALL frames (not just underflows)
        if self._producer_error is not None:
            self._frames_since_error += 1
            if self._frames_since_error > self._max_repeats:
                self._tripped = True
                self.fail_fast_reason = (
                    f"producer_error_drained: error="
                    f"{type(self._producer_error).__name__}: "
                    f"{str(self._producer_error)[:200]}"
                )
                return self.fail_fast_reason

        # Policy 2: sustained consecutive underflow (independent of producer error)
        if underflow:
            self._consecutive_underflows += 1
        else:
            self._consecutive_underflows = 0

        if self._consecutive_underflows > self._max_underflows:
            self._tripped = True
            self.fail_fast_reason = (
                f"sustained_underflow: {self._consecutive_underflows} "
                f"consecutive underflows "
                f"(~{self._consecutive_underflows / FPS:.1f}s frozen)"
            )
            return self.fail_fast_reason

        return None

    def is_tripped(self) -> bool:
        return self._tripped

    def snapshot(self) -> dict[str, Any]:
        return {
            "consecutive_underflows": self._consecutive_underflows,
            "frames_since_error": self._frames_since_error,
            "fail_fast_reason": self.fail_fast_reason,
        }


# ── Publisher Metrics ────────────────────────────────────────────────────────

class _GstPublisherMetrics:
    """Counters for the GStreamer publisher's push pipeline."""

    def __init__(self) -> None:
        self.frames_generated: int = 0
        self.frames_emitted: int = 0
        self.frames_repeated: int = 0
        self.underflow_count: int = 0
        self.queue_depth_min: int = 10 ** 9
        self.queue_depth_max: int = 0
        self.producer_error: str | None = None
        self.fail_fast_reason: str | None = None
        self.pipeline_error: str | None = None
        self.source_underflow_count: int = 0
        self.source_frames_repeated: int = 0

    def snapshot(self, encoder_info: dict[str, Any] | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "frames_generated": self.frames_generated,
            "frames_emitted": self.frames_emitted,
            "frames_repeated": self.frames_repeated,
            "underflow_count": self.underflow_count,
            "queue_depth_min": self.queue_depth_min if self.queue_depth_min < 10 ** 9 else 0,
            "queue_depth_max": self.queue_depth_max,
            "producer_error": self.producer_error,
            "fail_fast_reason": self.fail_fast_reason,
            "pipeline_error": self.pipeline_error,
            "source_underflow_count": self.source_underflow_count,
            "source_frames_repeated": self.source_frames_repeated,
        }
        if encoder_info:
            result["encoder_info"] = encoder_info
        return result


# ── ICE Forensic Instrumentation ─────────────────────────────────────────────
#
# Pure-Python forensic recorder for ICE/WebRTC state transitions. Captures every
# signal/callback with elapsed-ms from recorder creation. Thread-safe via a Lock
# (GStreamer callbacks fire on the GLib thread). Designed to be JSON-serializable
# for Modal prints + PostHog. No GStreamer dependency — fully unit-testable.
#
# Key design rules:
#   - No raw IP addresses, no full SDP, no full candidate strings.
#   - All timestamps are monotonic elapsed_ms from t0 (never wall-clock).
#   - Bounded arrays (max 20 entries) to prevent unbounded growth.
#   - Thread-safe: every public method acquires self._lock.

import json as _json

_MAX_BUS_ENTRIES = 20

# IP redaction patterns for bus/error/warning debug text.
# GStreamer and libnice debug strings can contain IP addresses (e.g., in
# candidate descriptions or connection error messages). These patterns
# redact IPv4 and full IPv6 addresses to [redacted-ip].
_IPV4_RE = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_IPV6_RE = re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b')


def _redact_debug_text(text: str, max_length: int = 300) -> str:
    """Redact IP addresses from debug text for safe logging/telemetry.

    Replaces IPv4 and full IPv6 addresses with [redacted-ip].
    Truncates to max_length. Applied to all bus error/warning/element messages
    before storage in the forensic recorder.
    """
    if not text:
        return ""
    redacted = _IPV4_RE.sub('[redacted-ip]', text)
    redacted = _IPV6_RE.sub('[redacted-ip]', redacted)
    return redacted[:max_length]


# Element message type prefixes to capture (filtered — not all bus messages).
# This keeps the forensic arrays clean and focused on ICE/DTLS/WebRTC issues.
# Only ICE / libnice / DTLS / WebRTC-relevant messages are captured.
_ELEMENT_MSG_FILTER_PREFIXES = (
    "ice",       # GstWebRTCICE messages
    "nice",      # libnice ICE agent messages
    "dtls",      # DTLS handshake messages
    "GstWebRTC", # GstWebRTC* element messages
)


def _sanitize_candidate(candidate: str) -> dict[str, str]:
    """Parse an ICE candidate line, return type/protocol/foundation without IP.

    Candidate format (RFC 8445):
        <foundation> <component> <protocol> <priority> <addr> <port> typ <type> [raddr <rel-addr>] [rport <rel-port>]

    Example input:  "1 1 UDP 2122252543 192.168.1.100 12345 typ host"
    Example output: {"type": "host", "protocol": "udp", "foundation": "1"}
    """
    if not candidate:
        return {"type": "unknown", "protocol": "unknown", "foundation": ""}
    parts = candidate.split()
    result: dict[str, str] = {
        "type": "unknown",
        "protocol": "unknown",
        "foundation": parts[0] if parts else "",
    }
    if len(parts) >= 3:
        result["protocol"] = parts[2].lower()
    # Find typ=<type> or "typ <type>"
    for i, token in enumerate(parts):
        if token == "typ" and i + 1 < len(parts):
            result["type"] = parts[i + 1]
            break
        if token.startswith("typ="):
            result["type"] = token[4:]
            break
    return result


class _IceDebugRecorder:
    """Thread-safe forensic recorder for ICE/WebRTC state transitions.

    Captures timing checkpoints, state transition histories, sanitized candidate
    metadata, and bounded bus message summaries. All timestamps are elapsed_ms
    from creation (t0). Pure Python — no GStreamer dependency.
    """

    def __init__(self, *, attempt_id: str = "", track_name: str = "", publisher_type: str = "") -> None:
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._attempt_id = attempt_id
        self._track_name = track_name
        self._publisher_type = publisher_type

        # Phase timing checkpoints (elapsed_ms or None)
        self._checkpoints: dict[str, float] = {}

        # SDP/answer info
        self._answer_received: bool = False
        self._answer_sdp_length: int = 0

        # Candidate metadata (sanitized — no IPs)
        self._candidate_types: set[str] = set()
        self._candidate_protocols: set[str] = set()
        self._candidate_foundations: set[str] = set()
        self._candidate_count: int = 0

        # State transition histories
        self._ice_connection_transitions: list[dict[str, Any]] = []
        self._ice_gathering_transitions: list[dict[str, Any]] = []
        self._signaling_state_transitions: list[dict[str, Any]] = []
        self._connection_state_transitions: list[dict[str, Any]] = []

        # Final states (read at timeout)
        self._final_ice_state: str | None = None
        self._final_gathering_state: str | None = None
        self._final_signaling_state: str | None = None
        self._final_connection_state: str | None = None

        # Stats snapshot (from get-stats at timeout)
        self._stats_snapshot: dict[str, Any] | None = None
        self._stats_collection_timed_out: bool = False

        # Bus messages (bounded arrays)
        self._bus_errors: list[str] = []
        self._bus_warnings: list[str] = []
        self._bus_element_msgs: list[str] = []
        self._webrtcbin_errors: list[str] = []

        # Timeout/cleanup
        self._timeout_phase: str | None = None
        self._cleanup_started_ms: float | None = None
        self._cleanup_completed_ms: float | None = None
        self._cleanup_ok: bool | None = None

        # TURN relay metadata (safe fields only — no credentials)
        self._turn_configured: bool = False
        self._turn_url_count: int = 0
        self._turn_transports: list[str] = []
        self._selected_local_candidate_type: str | None = None
        self._selected_remote_candidate_type: str | None = None

    @property
    def t0(self) -> float:
        return self._t0

    def _elapsed_ms(self) -> float:
        return round((time.monotonic() - self._t0) * 1000.0, 1)

    def record_checkpoint(self, name: str) -> None:
        """Record a phase timing checkpoint (elapsed_ms from t0)."""
        with self._lock:
            if name not in self._checkpoints:
                self._checkpoints[name] = self._elapsed_ms()

    def record_answer(self, received: bool, sdp_length: int) -> None:
        with self._lock:
            self._answer_received = received
            self._answer_sdp_length = sdp_length

    def record_candidate(self, candidate: str) -> None:
        """Record a sanitized ICE candidate (no IP stored)."""
        parsed = _sanitize_candidate(candidate)
        with self._lock:
            self._candidate_count += 1
            if self._candidate_count == 1:
                self._checkpoints["first_candidate"] = self._elapsed_ms()
            if parsed["type"] != "unknown":
                self._candidate_types.add(parsed["type"])
            if parsed["protocol"] != "unknown":
                self._candidate_protocols.add(parsed["protocol"])
            if parsed["foundation"]:
                self._candidate_foundations.add(parsed["foundation"])

    def record_state_transition(self, category: str, state: str) -> None:
        """Record a webrtcbin state transition with elapsed_ms."""
        entry = {"state": state, "elapsed_ms": self._elapsed_ms()}
        with self._lock:
            if category == "ice_connection":
                self._ice_connection_transitions.append(entry)
            elif category == "ice_gathering":
                self._ice_gathering_transitions.append(entry)
            elif category == "signaling":
                self._signaling_state_transitions.append(entry)
            elif category == "connection":
                self._connection_state_transitions.append(entry)

    def record_final_states(
        self,
        *,
        ice_connection: str | None,
        ice_gathering: str | None,
        signaling: str | None,
        connection: str | None,
    ) -> None:
        with self._lock:
            self._final_ice_state = ice_connection
            self._final_gathering_state = ice_gathering
            self._final_signaling_state = signaling
            self._final_connection_state = connection

    def record_stats_snapshot(self, stats: dict[str, Any] | None, timed_out: bool) -> None:
        with self._lock:
            self._stats_snapshot = stats
            self._stats_collection_timed_out = timed_out

    def record_bus_error(self, message: str) -> None:
        with self._lock:
            if len(self._bus_errors) < _MAX_BUS_ENTRIES:
                self._bus_errors.append(_redact_debug_text(message))

    def record_bus_warning(self, message: str) -> None:
        with self._lock:
            if len(self._bus_warnings) < _MAX_BUS_ENTRIES:
                self._bus_warnings.append(_redact_debug_text(message))

    def record_bus_element(self, message: str) -> None:
        with self._lock:
            if len(self._bus_element_msgs) < _MAX_BUS_ENTRIES:
                self._bus_element_msgs.append(_redact_debug_text(message))

    def record_webrtcbin_error(self, message: str) -> None:
        with self._lock:
            if len(self._webrtcbin_errors) < _MAX_BUS_ENTRIES:
                self._webrtcbin_errors.append(_redact_debug_text(message))

    def record_timeout(self, phase: str) -> None:
        with self._lock:
            self._timeout_phase = phase
            self._checkpoints["ice_timeout"] = self._elapsed_ms()

    def record_cleanup_start(self) -> None:
        with self._lock:
            self._cleanup_started_ms = self._elapsed_ms()

    def record_cleanup_complete(self, ok: bool) -> None:
        with self._lock:
            self._cleanup_completed_ms = self._elapsed_ms()
            self._cleanup_ok = ok

    def record_turn_config(self, *, configured: bool, url_count: int, transports: list[str]) -> None:
        with self._lock:
            self._turn_configured = configured
            self._turn_url_count = url_count
            self._turn_transports = sorted(set(transports))

    def record_selected_pair(self, *, local_type: str | None, remote_type: str | None) -> None:
        with self._lock:
            self._selected_local_candidate_type = local_type
            self._selected_remote_candidate_type = remote_type

    def snapshot(self) -> dict[str, Any]:
        """Produce a JSON-serializable snapshot of all forensic data."""
        with self._lock:
            return {
                "attempt_id": self._attempt_id,
                "track_name": self._track_name,
                "publisher_type": self._publisher_type,
                "checkpoints": dict(sorted(self._checkpoints.items())),
                "answer_received": self._answer_received,
                "answer_sdp_length": self._answer_sdp_length,
                "local_candidate_count": self._candidate_count,
                "local_candidate_types": sorted(self._candidate_types),
                "local_candidate_protocols": sorted(self._candidate_protocols),
                "local_candidate_foundation_count": len(self._candidate_foundations),
                "ice_connection_transitions": list(self._ice_connection_transitions),
                "ice_gathering_transitions": list(self._ice_gathering_transitions),
                "signaling_state_transitions": list(self._signaling_state_transitions),
                "connection_state_transitions": list(self._connection_state_transitions),
                "final_ice_state": self._final_ice_state,
                "final_gathering_state": self._final_gathering_state,
                "final_signaling_state": self._final_signaling_state,
                "final_connection_state": self._final_connection_state,
                "stats_snapshot": self._stats_snapshot,
                "stats_collection_timed_out": self._stats_collection_timed_out,
                "webrtcbin_error_messages": list(self._webrtcbin_errors),
                "gstreamer_bus_errors": list(self._bus_errors),
                "gstreamer_bus_warnings": list(self._bus_warnings),
                "gstreamer_bus_element_msgs": list(self._bus_element_msgs),
                "timeout_phase": self._timeout_phase,
                "cleanup_started_ms": self._cleanup_started_ms,
                "cleanup_completed_ms": self._cleanup_completed_ms,
                "cleanup_ok": self._cleanup_ok,
                "turn_configured": self._turn_configured,
                "turn_url_count": self._turn_url_count,
                "turn_transports": list(self._turn_transports),
                "selected_local_candidate_type": self._selected_local_candidate_type,
                "selected_remote_candidate_type": self._selected_remote_candidate_type,
            }

    def to_json(self) -> str:
        """Compact JSON for Modal structured prints."""
        return _json.dumps(self.snapshot(), separators=(",", ":"))


def _should_capture_element_msg(msg_src_name: str) -> bool:
    """Filter element bus messages — only capture ICE/libnice/DTLS/WebRTC-relevant."""
    name_lower = msg_src_name.lower()
    return any(name_lower.startswith(prefix.lower()) or prefix.lower() in name_lower
               for prefix in _ELEMENT_MSG_FILTER_PREFIXES)


class _IceDebugError(RuntimeError):
    """RuntimeError carrying forensic ice_debug data for ICE/offer failures.

    The ice_debug dict is produced by _IceDebugRecorder.snapshot() and contains
    only sanitized fields (no IPs, no full SDP).
    """

    def __init__(self, message: str, ice_debug: dict[str, Any]) -> None:
        super().__init__(message)
        self.ice_debug = ice_debug


class PublisherIceFailedAfterRetries(RuntimeError):
    """Raised when all publisher startup attempts are exhausted.

    Carries sanitized per-attempt summaries (ice_debug snapshots). No
    credentials, no IPs, no full SDP — only allowlisted forensic fields.
    """

    def __init__(self, message: str, attempt_summaries: list[dict[str, Any]]) -> None:
        super().__init__(message)
        self.attempt_summaries = attempt_summaries


# ── TURN Support (pure Python — no GStreamer dependency) ──────────────────────

# Regex to match the userinfo portion of a TURN URI: scheme://USERINFO@host
_TURN_USERINFO_RE = re.compile(r'^(turns?)://([^@]+)@', re.IGNORECASE)


def _ice_servers_to_gstreamer_uris(ice_servers: list[dict[str, Any]]) -> list[str]:
    """Convert WebRTC iceServers format to GStreamer add-turn-server URIs.

    Input format (Cloudflare TURN API response):
        [{"urls": ["turn:host:3478?transport=udp", "turns:host:5349?transport=tcp"],
          "username": "...", "credential": "..."}]

    Output format (GStreamer webrtcbin add-turn-server signal):
        ["turn://user:pass@host:3478?transport=udp",
         "turns://user:pass@host:5349?transport=tcp"]

    Rules:
      - Handles ``urls`` as a string or array.
      - URL-encodes username and credential (future-proof against special chars).
      - Skips STUN-only entries (no username/credential).
      - Skips non-turn/turns URLs (e.g., stun:).
      - Deduplicates by the resulting URI.
    """
    turn_uris: list[str] = []
    seen: set[str] = set()
    for server in ice_servers or []:
        username = server.get("username") or ""
        credential = server.get("credential") or ""
        if not username or not credential:
            continue

        urls = server.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]

        encoded_user = _url_quote(username, safe="")
        encoded_pass = _url_quote(credential, safe="")

        for url in urls:
            if not isinstance(url, str):
                continue
            stripped = url.strip()
            if stripped.startswith("turn:"):
                rest = stripped[5:]
                gst_uri = f"turn://{encoded_user}:{encoded_pass}@{rest}"
            elif stripped.startswith("turns:"):
                rest = stripped[6:]
                gst_uri = f"turns://{encoded_user}:{encoded_pass}@{rest}"
            else:
                continue

            if gst_uri not in seen:
                seen.add(gst_uri)
                turn_uris.append(gst_uri)

    return turn_uris


def _redact_turn_uri(uri: str) -> str:
    """Redact credentials from a TURN URI for safe logging/telemetry.

    "turn://user:pass@host:3478?transport=udp" → "turn://[redacted]:[redacted]@host:3478?transport=udp"
    "turns://user:pass@host:5349" → "turns://[redacted]:[redacted]@host:5349"

    If the URI has no userinfo, returns it unchanged.
    """
    if not uri:
        return ""

    def _redact_replacer(match: re.Match[str]) -> str:
        return f"{match.group(1)}://[redacted]:[redacted]@"

    return _TURN_USERINFO_RE.sub(_redact_replacer, uri)


def _extract_turn_transports(turn_uris: list[str]) -> list[str]:
    """Extract relay transport types from GStreamer TURN URIs.

    GStreamer source code behavior (gstwebrtcice.c _add_turn_server):
      - turn:// without ?transport= → adds BOTH udp and tcp relays
      - turn://...?transport=udp     → udp only
      - turn://...?transport=tcp     → tcp only
      - turns://                     → tls only

    Returns sorted unique list, e.g. ["tcp", "tls", "udp"].
    """
    transports: set[str] = set()
    for uri in turn_uris:
        if uri.startswith("turns://"):
            transports.add("tls")
        elif uri.startswith("turn://"):
            if "?transport=udp" in uri:
                transports.add("udp")
            elif "?transport=tcp" in uri:
                transports.add("tcp")
            else:
                transports.add("udp")
                transports.add("tcp")
    return sorted(transports)


# ── Retry error classifier ───────────────────────────────────────────────────

_RETRYABLE_SIGNALS = (
    "ice_connect_timeout",
    "ice_failed",
    "offer_timeout",
    "no_ice_candidates",
    "no_selected_pair",
    "publisher-offer callback failed",
)

_NON_RETRYABLE_SIGNALS = (
    "cuda",
    "nvh264enc",
    "nvenc",
    "moda_worker_secret not set",
    "modal_worker_secret not set",
    "moda_worker_secret",
    "fail_fast",
    "frame_source",
    "sustained_underflow",
    "producer_error",
    "gstsampler",  # GStreamer sampler/allocator errors
    "turn_configuration_failed",  # TURN URI rejected by webrtcbin
)


def _is_retryable_ice_error(exc: Exception) -> bool:
    """Classify whether an exception warrants a retry attempt.

    Only ICE/connectivity/transient SFU-API failures are retryable.
    CUDA, encoder, pipeline, frame source, and auth errors are NOT retried.

    This function is intentionally conservative — retrying a CUDA error
    wastes time and risks container instability.
    """
    msg = str(exc).lower()

    for signal in _NON_RETRYABLE_SIGNALS:
        if signal in msg:
            return False

    for signal in _RETRYABLE_SIGNALS:
        if signal in msg:
            return True

    # If it carries ice_debug (is _IceDebugError), treat as retryable by default.
    if hasattr(exc, "ice_debug"):
        return True

    return False


# ── GStreamer WebRTC Publisher ───────────────────────────────────────────────

class GstWebRtcPublisher:
    """Manages a GStreamer NVENC WebRTC publisher lifecycle for SFU publishing.

    SDP exchange is brokered through Worker/DO — never calls the Cloudflare
    SFU directly. Modal authenticates to Worker with the x-modal-secret header.

    Interface matches webrtc_publisher.WebRtcPublisher:
        async def start(self) -> str   # returns video mid
        async def stop(self) -> dict   # returns metrics, idempotent

    Pre-roll behavior:
        The push thread starts immediately in start() (GStreamer's webrtcbin
        needs the first appsrc frame to trigger on-negotiation-needed). Before
        ICE connects, the publisher pushes cached black I420 frames — NO GPU
        work happens. After ICE connects, frame_source.start() is called and
        real frames flow. This prevents GPU rendering during SDP negotiation.

    Fail-fast behavior:
        The push thread consults _FailFastPolicy on every frame read in the
        live phase. If the policy trips (sustained underflow or producer
        error + grace drain), the push thread signals EOS to appsrc and stops.
        The browser sees stream end, not frozen video.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        track_name: str,
        publisher_callback_url: str,
        close_callback_url: str,
        *,
        bitrate: int = BITRATE_DEFAULT,
        ice_servers: list[dict[str, Any]] | None = None,
        h264_capture_dir: str | None = None,
        direct_webrtc_debug: bool = False,
        answer_callback_url: str | None = None,
        encoder: str = "nvh264enc",
    ) -> None:
        self._source = frame_source
        self._track_name = track_name
        self._callback_url = publisher_callback_url
        self._close_url = close_callback_url
        self._bitrate = bitrate
        self._h264_capture_dir = h264_capture_dir
        self._direct_webrtc_debug = direct_webrtc_debug
        self._answer_callback_url = answer_callback_url
        self._encoder = encoder

        # TURN relay configuration (Cloudflare ICE servers format).
        # Converted to GStreamer URIs in _configure_turn() during start().
        # If None or empty, only STUN + direct ICE is used (dev/diagnostics only).
        self._ice_servers: list[dict[str, Any]] = list(ice_servers) if ice_servers else []
        self._turn_uris: list[str] = _ice_servers_to_gstreamer_uris(self._ice_servers)
        if self._ice_servers and not self._turn_uris:
            raise RuntimeError(
                "turn_configuration_failed: ice_servers was non-empty but "
                "yielded zero usable TURN URIs (all entries lacked credentials "
                "or had no turn:/turns: URLs). Cannot proceed without TURN."
            )

        self._modal_secret: str | None = os.environ.get("MODAL_WORKER_SECRET")

        # GStreamer objects (set in start(), None until then)
        self._Gst: Any = None
        self._GstWebRTC: Any = None
        self._GstSdp: Any = None
        self._GLib: Any = None
        self._pipeline: Any = None
        self._webrtc: Any = None
        self._appsrc: Any = None
        self._bus: Any = None

        # Threading
        self._glib_loop: Any = None
        self._glib_thread: threading.Thread | None = None
        self._push_thread: threading.Thread | None = None
        self._stop_push = threading.Event()

        # SDP / ICE state
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ice_connected = asyncio.Event()
        self._ice_failed = asyncio.Event()
        self._offer_ready = asyncio.Event()
        self._local_sdp: str | None = None
        self._eoc = False
        self._ice_candidates: list[str] = []
        self._video_mid: str = "0"
        self._pipeline_error: str | None = None

        # Direct WebRTC debug: answer SDP arrives asynchronously via a separate
        # Modal endpoint instead of the synchronous publisher-offer response.
        self._answer_event = asyncio.Event()
        self._answer_data: dict[str, Any] | None = None

        # Pre-roll black frame (cached, no per-frame allocation)
        self._black_frame: bytes = _black_i420_bytes()
        self._push_counter: int = 0

        # Pacing telemetry
        self._push_intervals_ms: list[float] = []
        self._push_pts_deltas_ms: list[float] = []
        self._push_late_count: int = 0
        self._push_last_wall: float = 0.0

        # Metrics + fail-fast
        self._metrics = _GstPublisherMetrics()
        self._fail_fast = _FailFastPolicy()

        # H264 capture (Boundary B diagnostic, None unless h264_capture_dir set)
        self._h264_capture_file = None
        self._h264_capture_count = 0
        self._H264_CAPTURE_MAX = 750  # ~30s at 25fps

        # RTP validation after rtph264pay (direct WebRTC / SFU transport-control).
        self._rtp_pay_probe_state: dict[str, Any] | None = None
        self._rtp_pay_validation_max_packets = 10_000

        # ICE forensic recorder (created in start(), None until then)
        self._recorder: _IceDebugRecorder | None = None

        # Idempotent stop
        self._stopped = False
        self._last_metrics: dict[str, Any] | None = None

    # ── Thread-safe Event setter ─────────────────────────────────────────

    def _ts_set(self, event: asyncio.Event) -> None:
        """Thread-safe Event.set() — GStreamer callbacks fire on GLib thread."""
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(event.set)
        else:
            event.set()

    # ── GStreamer signal callbacks (fire on GLib thread) ─────────────────

    def _on_negotiation_needed(self, element: Any) -> None:
        logger.info("on-negotiation-needed")
        promise = self._Gst.Promise.new_with_change_func(
            self._on_offer_created, element, None
        )
        element.emit("create-offer", None, promise)

    def _on_offer_created(self, promise: Any, _webrtc: Any, _ud: Any) -> None:
        promise.wait()
        reply = promise.get_reply()
        offer = reply.get_value("offer")

        set_promise = self._Gst.Promise.new()
        self._webrtc.emit("set-local-description", offer, set_promise)
        set_promise.interrupt()

        self._local_sdp = offer.sdp.as_text()
        logger.info("Offer created (local SDP: %d chars)", len(self._local_sdp))
        if self._recorder:
            self._recorder.record_checkpoint("offer_created")
            self._recorder.record_checkpoint("local_description_set")

    def _on_ice_candidate(self, _webrtc: Any, _mlineindex: int, candidate: str) -> None:
        if not candidate:
            logger.info("ICE end-of-candidates")
            self._eoc = True
            self._maybe_emit_offer()
            return
        self._ice_candidates.append(candidate)
        parsed = _sanitize_candidate(candidate)
        if self._recorder:
            self._recorder.record_candidate(candidate)
        logger.debug(
            "ICE candidate: type=%s protocol=%s foundation=%s",
            parsed["type"], parsed["protocol"], parsed["foundation"],
        )

    def _on_ice_gathering_state(self, webrtc: Any, _pspec: Any) -> None:
        state = webrtc.get_property("ice-gathering-state")
        state_nick = state.value_nick
        logger.info("ICE gathering state: %s", state_nick)
        if self._recorder:
            self._recorder.record_state_transition("ice_gathering", state_nick)
            if state_nick != "new":
                self._recorder.record_checkpoint("ice_gathering_started")
        if state == self._GstWebRTC.WebRTCICEGatheringState.COMPLETE:
            if self._recorder:
                self._recorder.record_checkpoint("ice_gathering_completed")
            self._eoc = True
            self._maybe_emit_offer()

    def _on_ice_connection_state(self, webrtc: Any, _pspec: Any) -> None:
        state = webrtc.get_property("ice-connection-state")
        state_nick = state.value_nick
        logger.info("ICE connection state: %s", state_nick)
        _update_crash_context(ice_connection_state=state_nick)
        if self._recorder:
            self._recorder.record_state_transition("ice_connection", state_nick)
            if state_nick in ("connected", "completed"):
                self._recorder.record_checkpoint("ice_connected")
        if state in (
            self._GstWebRTC.WebRTCICEConnectionState.CONNECTED,
            self._GstWebRTC.WebRTCICEConnectionState.COMPLETED,
        ):
            self._ts_set(self._ice_connected)
        elif state == self._GstWebRTC.WebRTCICEConnectionState.FAILED:
            self._ts_set(self._ice_failed)

    def _on_connection_state(self, webrtc: Any, _pspec: Any) -> None:
        """WebRTC peer connection state (NEW/CONNECTING/CONNECTED/DISCONNECTED/FAILED/CLOSED)."""
        state = webrtc.get_property("connection-state")
        state_nick = state.value_nick if hasattr(state, "value_nick") else str(state)
        logger.info("Connection state: %s", state_nick)
        _update_crash_context(connection_state=state_nick)
        if self._recorder:
            self._recorder.record_state_transition("connection", state_nick)

    def _on_signaling_state(self, webrtc: Any, _pspec: Any) -> None:
        """SDP signaling state (stable/have-local-offer/have-remote-pranswer/etc.)."""
        state = webrtc.get_property("signaling-state")
        state_nick = state.value_nick if hasattr(state, "value_nick") else str(state)
        logger.info("Signaling state: %s", state_nick)
        if self._recorder:
            self._recorder.record_state_transition("signaling", state_nick)

    def _on_bus_message(self, _bus: Any, msg: Any) -> None:
        if msg.type == self._Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            err_str = f"GstError: {err}: {dbg}"
            # Redact IPs before storing anywhere — GStreamer debug text can
            # contain addresses from libnice/ICE candidate descriptions.
            self._pipeline_error = _redact_debug_text(err_str)
            logger.error("Pipeline bus ERROR: %s | %s", err, dbg)
            if self._recorder:
                self._recorder.record_bus_error(err_str)

    def _on_bus_warning(self, _bus: Any, msg: Any) -> None:
        """Capture GStreamer bus WARNING messages (bounded, fire-and-forget)."""
        try:
            err, dbg = msg.parse_warning()
            warn_str = f"GstWarning: {err}: {dbg}"
            logger.warning("Pipeline bus WARNING: %s | %s", err, dbg)
            if self._recorder:
                self._recorder.record_bus_warning(warn_str)
        except Exception as exc:
            logger.debug("Could not parse bus warning: %s", exc)

    def _on_bus_element(self, _bus: Any, msg: Any) -> None:
        """Capture filtered element messages (ICE/libnice/DTLS/WebRTC only)."""
        try:
            src = msg.src
            src_name = src.get_name() if hasattr(src, "get_name") else str(src)
            structure = msg.get_structure()
            struct_name = structure.get_name() if structure else "unknown"
            combined = f"{src_name}:{struct_name}"
            if not _should_capture_element_msg(combined):
                return
            summary = f"{src_name} [{struct_name}]"
            logger.debug("Bus element msg: %s", summary)
            if self._recorder:
                self._recorder.record_bus_element(summary)
        except Exception as exc:
            logger.debug("Could not parse bus element message: %s", exc)

    # ── Offer assembly ───────────────────────────────────────────────────

    def _maybe_emit_offer(self) -> None:
        """When offer exists and ICE gathering is complete, inject candidates
        into the SDP (CF Realtime SFU needs non-trickle), then signal offer_ready.
        """
        if not (self._local_sdp and self._eoc) or self._offer_ready.is_set():
            return

        mid_match = re.search(r"^a=mid:(.+)$", self._local_sdp, re.MULTILINE)
        if mid_match:
            self._video_mid = mid_match.group(1).strip()
            logger.info("Extracted video mid from SDP: %s", self._video_mid)

        if self._ice_candidates and "a=candidate:" not in self._local_sdp:
            lines = self._local_sdp.split("\r\n")
            insert_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith("m=video"):
                    insert_idx = i + 2
                    break
            for cand in self._ice_candidates:
                lines.insert(insert_idx, f"a=candidate:{cand}")
                insert_idx += 1
            lines.insert(insert_idx, "a=end-of-candidates")
            self._local_sdp = "\r\n".join(lines)
            logger.info(
                "Injected %d ICE candidates into SDP", len(self._ice_candidates)
            )

        self._ts_set(self._offer_ready)

    # ── Pipeline construction ────────────────────────────────────────────

    def _build_pipeline(self) -> Any:
        template = _PIPELINE_TEMPLATE_X264 if self._encoder == "x264enc" else _PIPELINE_TEMPLATE
        pipe_desc = template.format(
            stun=STUN_SERVER,
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            gop=GOP_SIZE,
            bitrate=self._bitrate,
            agg_mode=RTP_AGGREGATE_MODE,
        )
        logger.info("Pipeline:\n%s", pipe_desc)
        logger.info(
            "Encoder: %s %dx%d bitrate=%d gop=%d",
            self._encoder, WIDTH, HEIGHT, self._bitrate, GOP_SIZE,
        )
        pipeline = self._Gst.parse_launch(pipe_desc)
        self._webrtc = pipeline.get_by_name("sendrecv")
        self._appsrc = pipeline.get_by_name("src")

        # Pad-probe: verify buffer PTS/DTS/duration deltas are stable at frame_duration.
        # Logs aggregate stats every 10s. Proves publisher hands GStreamer a paced stream.
        _appsrc_src_pad = self._appsrc.get_static_pad("src")
        if _appsrc_src_pad is not None:
            _appsrc_src_pad.add_probe(
                self._Gst.PadProbeType.BUFFER,
                self._make_pts_probe_callback(),
            )
            logger.info("PTS sanity probe installed on appsrc:src")

        # Boundary B diagnostic: passive H264 capture after h264parse, before rtph264pay
        if self._h264_capture_dir:
            import os as _os
            _os.makedirs(self._h264_capture_dir, exist_ok=True)
            h264_path = _os.path.join(self._h264_capture_dir, "boundary_b.h264")
            self._h264_capture_file = open(h264_path, "wb")
            h264parse = pipeline.get_by_name("h264p")
            if h264parse is not None:
                src_pad = h264parse.get_static_pad("src")
                src_pad.add_probe(
                    self._Gst.PadProbeType.BUFFER,
                    self._h264_capture_probe,
                )
                logger.info("H264 capture probe installed: %s (max %d buffers)",
                            h264_path, self._H264_CAPTURE_MAX)
            else:
                logger.warning("h264parse element not found — H264 capture disabled")

        # RTP-after-pay validation probe — OFF by default for stability.
        # The probe registers BUFFER | BUFFER_LIST. Calling get_buffer() on a
        # BUFFER_LIST event triggers a GStreamer-CRITICAL assertion
        # (gst_pad_probe_info_get_buffer) that spam-crashes the container
        # (SIGABRT). Enable ONLY for RTP-level debugging.
        if os.environ.get("AVTR_RTP_VALIDATION", "0") == "1":
            rtph264pay = pipeline.get_by_name("rtppay")
            if rtph264pay is not None:
                pay_src = rtph264pay.get_static_pad("src")
                if pay_src is not None:
                    pay_src.add_probe(
                        self._Gst.PadProbeType.BUFFER | self._Gst.PadProbeType.BUFFER_LIST,
                        self._make_rtp_pay_probe_callback(),
                    )
                    logger.info("RTP validation probe installed on rtph264pay:src (AVTR_RTP_VALIDATION=1)")
            else:
                logger.warning("rtph264pay element not found — RTP validation disabled")
        else:
            self._rtp_pay_probe_state = None
            logger.info("RTP validation probe SKIPPED (AVTR_RTP_VALIDATION not set — OFF by default)")

        # Configure TURN relay servers via add-turn-server action signal.
        # Must be called after parse_launch (webrtcbin exists) and BEFORE
        # set_state(PLAYING) so TURN candidates are gathered during ICE.
        self._configure_turn()

        return pipeline

    def _configure_turn(self) -> None:
        """Emit add-turn-server for each TURN URI on the webrtcbin element.

        GStreamer source (gstwebrtcice.c _add_turn_server):
          - turn:// without ?transport= → adds BOTH udp and tcp relays
          - turn://...?transport=udp     → udp relay only
          - turn://...?transport=tcp     → tcp relay only
          - turns://                     → tls relay only

        This is a no-op if no TURN URIs are configured (dev/diagnostics mode).
        """
        if not self._turn_uris or self._webrtc is None:
            if self._recorder:
                transports = _extract_turn_transports(self._turn_uris)
                self._recorder.record_turn_config(
                    configured=False, url_count=0, transports=transports,
                )
            return

        failed_uris: list[str] = []
        for uri in self._turn_uris:
            redacted = _redact_turn_uri(uri)
            try:
                ret = self._webrtc.emit("add-turn-server", uri)
                if ret is False:
                    logger.error("TURN server REJECTED by webrtcbin: %s", redacted)
                    failed_uris.append(redacted)
                else:
                    logger.info("TURN server configured: %s (ret=%s)", redacted, ret)
            except Exception as exc:
                logger.error("Failed to configure TURN server %s: %s", redacted, exc)
                failed_uris.append(redacted)

        if failed_uris:
            raise RuntimeError(
                f"turn_configuration_failed: webrtcbin rejected {len(failed_uris)} "
                f"TURN server(s): {failed_uris}. Cannot proceed without TURN — "
                f"production-grade SFU requires TURN relay."
            )

        transports = _extract_turn_transports(self._turn_uris)
        if self._recorder:
            self._recorder.record_turn_config(
                configured=True,
                url_count=len(self._turn_uris),
                transports=transports,
            )
        logger.info(
            "TURN relay configured: %d URIs, transports=%s",
            len(self._turn_uris), transports,
        )

    def _h264_capture_probe(self, pad, info):
        """Passive pad probe: copies H264 NALUs to file without modifying flow."""
        if self._h264_capture_count >= self._H264_CAPTURE_MAX:
            if self._h264_capture_file:
                self._h264_capture_file.close()
                self._h264_capture_file = None
            return self._Gst.PadProbeReturn.OK
        if self._h264_capture_file is None:
            return self._Gst.PadProbeReturn.OK
        if info.type & self._Gst.PadProbeType.BUFFER:
            buf = info.get_buffer()
            if buf is not None:
                success, map_info = buf.map(self._Gst.MapFlags.READ)
                if success:
                    self._h264_capture_file.write(map_info.data)
                    self._h264_capture_count += 1
                    buf.unmap(map_info)
        elif info.type & self._Gst.PadProbeType.BUFFER_LIST:
            buf_list = info.get_buffer_list()
            if buf_list is not None:
                for i in range(buf_list.length()):
                    b = buf_list.get(i)
                    if b is not None:
                        success, map_info = b.map(self._Gst.MapFlags.READ)
                        if success:
                            self._h264_capture_file.write(map_info.data)
                            self._h264_capture_count += 1
                            b.unmap(map_info)
        return self._Gst.PadProbeReturn.OK

    def _make_pts_probe_callback(self):
        """Create a pad-probe closure that verifies PTS/duration stability.

        Logs aggregate stats every 10s. No per-buffer logging.
        """
        expected_ns = int(1_000_000_000 / FPS)
        state: dict[str, Any] = {
            "last_pts": None,
            "deltas_ns": [],
            "bad_count": 0,
            "total_count": 0,
            "last_log": time.monotonic(),
        }

        def _probe(pad, info):
            if not (info.type & self._Gst.PadProbeType.BUFFER):
                return self._Gst.PadProbeReturn.OK
            buf = info.get_buffer()
            if buf is None:
                return self._Gst.PadProbeReturn.OK
            pts = buf.pts
            duration = buf.duration
            state["total_count"] += 1
            if state["last_pts"] is not None and pts != self._Gst.CLOCK_TIME_NONE:
                delta = pts - state["last_pts"]
                state["deltas_ns"].append(delta)
                if len(state["deltas_ns"]) > 300:
                    state["deltas_ns"] = state["deltas_ns"][-300:]
                if abs(delta - expected_ns) > 2_000_000:  # >2ms drift = bad
                    state["bad_count"] += 1
            state["last_pts"] = pts if pts != self._Gst.CLOCK_TIME_NONE else None

            now = time.monotonic()
            if now - state["last_log"] >= 10.0:
                deltas = state["deltas_ns"]
                if deltas:
                    sd = sorted(deltas)
                    p50 = sd[len(sd) // 2] / 1_000_000.0
                    p95 = sd[int(len(sd) * 0.95)] if len(sd) >= 20 else max(sd)
                    p95 = p95 / 1_000_000.0
                    mx = max(sd) / 1_000_000.0
                    mn = min(sd) / 1_000_000.0
                    bad = state["bad_count"]
                    total = state["total_count"]
                else:
                    p50 = p95 = mx = mn = 0.0
                    bad = state["bad_count"]
                    total = state["total_count"]
                logger.info(
                    "[pts_probe] pts_delta p50=%.2fms p95=%.2fms min=%.2fms max=%.2fms "
                    "bad=%d/%d duration=%sns",
                    p50, p95, mn, mx, bad, total,
                    duration / 1_000_000.0 if duration != self._Gst.CLOCK_TIME_NONE else "?",
                )
                state["deltas_ns"] = []
                state["bad_count"] = 0
                state["total_count"] = 0
                state["last_log"] = now
            return self._Gst.PadProbeReturn.OK

        return _probe

    def _make_rtp_pay_probe_callback(self):
        """Create a thread-safe pad-probe recorder on RTP leaving rtph264pay.

        Only records every packet's seq/timestamp/marker/SSRC/PT/NAL types into
        a bounded ring buffer. All validation (seq gaps, timestamp cadence,
        marker bits, SPS/PPS/IDR counts) is done offline in
        _dump_rtp_pay_validation() so race conditions in real-time counting are
        avoided.
        """
        lock = threading.Lock()
        state: dict[str, Any] = {
            "packets": [],
            "start_monotonic": time.monotonic(),
            "first_packets_logged": 0,
            "last_log": time.monotonic(),
        }
        self._rtp_pay_probe_state = state

        def _scan_payload_nals(payload: bytes) -> list[int]:
            """Return all NAL unit types found in this RTP payload.

            Handles single NAL unit, FU-A fragmentation, and STAP-A aggregation.
            """
            if len(payload) < 1:
                return []
            first = payload[0]
            nal_type = first & 0x1F
            nals: list[int] = []

            if nal_type < 24:
                # Single NAL unit packet
                nals.append(nal_type)
            elif nal_type == 24:
                # STAP-A: one or more NAL units, each prefixed by 16-bit length
                off = 1
                while off + 2 <= len(payload):
                    nal_len = (payload[off] << 8) | payload[off + 1]
                    off += 2
                    if off + nal_len > len(payload) or nal_len < 1:
                        break
                    nals.append(payload[off] & 0x1F)
                    off += nal_len
            elif nal_type == 28:
                # FU-A: original NAL type in second byte, bits 0-4
                if len(payload) >= 2:
                    nals.append(payload[1] & 0x1F)
            else:
                # MTAP16/MTAP24/FU-B/etc — count payload first byte as-is
                nals.append(nal_type)
            return nals

        def _process_single_buf(buf) -> None:
            """Extract RTP header fields from one GstBuffer and record it."""
            success, map_info = buf.map(self._Gst.MapFlags.READ)
            if not success:
                return
            try:
                data = bytes(map_info.data)
                if len(data) < 12:
                    return
                b0 = data[0]
                version = (b0 >> 6) & 0x03
                if version != 2:
                    return

                b1 = data[1]
                marker = bool(b1 & 0x80)
                pt = b1 & 0x7F
                seq = (data[2] << 8) | data[3]
                ts = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
                ssrc = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]

                csrc_count = b0 & 0x0F
                header_len = 12 + 4 * csrc_count
                if b1 & 0x10 and len(data) >= header_len + 4:
                    ext_len_words = (data[header_len + 2] << 8) | data[header_len + 3]
                    header_len += 4 + 4 * ext_len_words

                payload = data[header_len:]
                nal_types = _scan_payload_nals(payload)

                with lock:
                    packets = state["packets"]
                    if len(packets) >= self._rtp_pay_validation_max_packets:
                        packets.pop(0)
                    packets.append({
                        "seq": seq,
                        "ts": ts,
                        "marker": marker,
                        "pt": pt,
                        "ssrc": ssrc,
                        "nal_types": nal_types,
                        "payload_len": len(payload),
                        "t": time.monotonic(),
                    })

                    if state["first_packets_logged"] < 10:
                        state["first_packets_logged"] += 1
                        logger.info(
                            "[rtp_pay_probe_sample #%d] seq=%d ts=%d marker=%s pt=%d "
                            "ssrc=%s nal_types=%s payload_len=%d",
                            state["first_packets_logged"], seq, ts, marker, pt,
                            ssrc, nal_types, len(payload),
                        )

                    now = time.monotonic()
                    if now - state["last_log"] >= 10.0:
                        logger.info(
                            "[rtp_pay_probe] packets=%d start_seq=%d end_seq=%d",
                            len(packets),
                            packets[0]["seq"] if packets else None,
                            packets[-1]["seq"] if packets else None,
                        )
                        state["last_log"] = now
            finally:
                buf.unmap(map_info)

        def _probe(_pad, info):
            # Guard: check info.type BEFORE calling get_buffer()/get_buffer_list().
            # Calling get_buffer() on a BUFFER_LIST event triggers a
            # GStreamer-CRITICAL assertion that can SIGABRT the container.
            if info.type & self._Gst.PadProbeType.BUFFER:
                buf = info.get_buffer()
                if buf is not None:
                    _process_single_buf(buf)
                    return self._Gst.PadProbeReturn.OK
            if info.type & self._Gst.PadProbeType.BUFFER_LIST:
                buf_list = info.get_buffer_list()
                if buf_list is not None:
                    length = buf_list.length()
                    for i in range(length):
                        b = buf_list.get(i)
                        if b is not None:
                            _process_single_buf(b)
                    return self._Gst.PadProbeReturn.OK

            return self._Gst.PadProbeReturn.OK

        return _probe

    def _dump_rtp_pay_validation(self) -> dict[str, Any] | None:
        """Write the RTP-after-pay validation snapshot to a JSON file.

        Performs offline validation on the recorded packets so the probe itself
        does no real-time counting. Returns the summary dict for final metrics.
        """
        state = self._rtp_pay_probe_state
        if state is None:
            return None

        try:
            packets = state.get("packets", [])
            total = len(packets)
            duration = time.monotonic() - state.get("start_monotonic", time.monotonic())
            ts_per_frame = 90_000 // FPS

            # Offline validation
            seq_gaps = 0
            seq_dups = 0
            ts_anomalies = 0
            marker_anomalies = 0
            sps_count = 0
            pps_count = 0
            idr_count = 0
            fu_a_count = 0
            stap_a_count = 0
            nal_other_count = 0
            frames = 0
            idr_frame_indices: list[int] = []
            last_ts = None
            last_seq = None
            marker_expected = False

            for i, p in enumerate(packets):
                seq = p["seq"]
                ts = p["ts"]
                marker = p["marker"]
                nal_types = p.get("nal_types", [])

                # NAL counts
                first_byte_nal = None
                if p.get("payload_len", 0) >= 1 and packets:
                    # We didn't store raw payload; infer STAP-A from nal_types length
                    pass
                for nt in nal_types:
                    if nt == 7:
                        sps_count += 1
                    elif nt == 8:
                        pps_count += 1
                    elif nt == 5:
                        idr_count += 1
                    else:
                        nal_other_count += 1

                # Detect packetization type from first NAL header byte. Since we
                # don't store the raw byte, infer from nal_types multiplicity:
                # - Single nal_types entry could be single NAL or FU-A
                # - Multiple nal_types entries means STAP-A
                if len(nal_types) > 1:
                    stap_a_count += 1
                # FU-A was expanded to the original NAL type; we can't distinguish
                # it from a single NAL here, which is fine for validation.

                # Sequence continuity with wraparound
                if last_seq is not None:
                    expected = (last_seq + 1) & 0xFFFF
                    if seq != expected:
                        diff = (seq - last_seq) & 0xFFFF
                        if diff == 0:
                            seq_dups += 1
                        else:
                            seq_gaps += 1
                last_seq = seq

                # Frame / timestamp / marker validation
                new_frame = last_ts is None or ts != last_ts
                if new_frame:
                    if marker_expected:
                        marker_anomalies += 1
                    frames += 1
                    marker_expected = True
                    if last_ts is not None:
                        ts_delta = (ts - last_ts) & 0xFFFFFFFF
                        if ts_delta != 0 and ts_delta % ts_per_frame != 0:
                            ts_anomalies += 1
                    last_ts = ts

                if marker:
                    if not marker_expected:
                        marker_anomalies += 1
                    marker_expected = False

                if 5 in nal_types and new_frame:
                    idr_frame_indices.append(frames)

            # Catch trailing unterminated frame
            if marker_expected:
                marker_anomalies += 1

            summary = {
                "duration_s": round(duration, 3),
                "total_packets": total,
                "frames_inferred": frames,
                "seq_gaps": seq_gaps,
                "seq_dups": seq_dups,
                "ts_anomalies": ts_anomalies,
                "marker_anomalies": marker_anomalies,
                "sps_count": sps_count,
                "pps_count": pps_count,
                "idr_count": idr_count,
                "fu_a_count": fu_a_count,
                "stap_a_count": stap_a_count,
                "nal_other_count": nal_other_count,
                "first_seq": packets[0]["seq"] if packets else None,
                "last_seq": packets[-1]["seq"] if packets else None,
                "first_ts": packets[0]["ts"] if packets else None,
                "last_ts": packets[-1]["ts"] if packets else None,
                "ssrc": packets[0]["ssrc"] if packets else None,
                "idr_frame_indices": idr_frame_indices,
            }

            dump = {
                "summary": summary,
                "local_sdp": self._local_sdp,
                "answer_data": self._answer_data,
                "encoder_info": dict(_ENCODER_INFO, bitrate=self._bitrate, direct_webrtc_debug=self._direct_webrtc_debug),
                "packets": packets,
            }

            import os as _os
            _os.makedirs("/tmp/avtr-rtp-validation", exist_ok=True)
            path = _os.path.join(
                "/tmp/avtr-rtp-validation",
                f"rtp_validation_{self._track_name}_{int(time.monotonic() * 1000)}.json",
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(dump, f, indent=None, separators=(",", ":"))
            logger.info("RTP validation dump written: %s", path)
            return summary
        except Exception as exc:
            logger.warning("RTP validation dump failed: %s", exc)
            return None

    # ── Push thread (frame source → appsrc) ──────────────────────────────

    def _push_frames(self) -> None:
        """Producer thread: feed frames into appsrc at FPS (push mode).

        Two phases:
          - Pre-roll (before ICE connected): push cached black I420 frames.
            No GPU work, no fail-fast. GStreamer needs the first frame to
            trigger on-negotiation-needed.
          - Live (after ICE connected): pull real frames from source.next_frame().
            Fail-fast policy is active.

        Emits a low-noise heartbeat every 10s with cadence diagnostics.
        """
        frame_duration_ns = int(1_000_000_000 / FPS)
        next_deadline = time.monotonic()

        # Heartbeat tracking (10s interval, fire-and-forget)
        hb_interval = 10.0
        hb_last_time = time.monotonic()
        hb_last_gen = 0
        hb_last_emit = 0
        hb_last_repeat = 0
        hb_last_underflow = 0

        while not self._stop_push.is_set():
            if self._ice_connected.is_set():
                # ── Live phase: real frames from source ──
                # Poll producer error BEFORE reading the frame.
                # Once error is observed, record_frame() counts EVERY frame
                # toward the grace drain — even if underflow_count doesn't
                # increase (source returns cached/repeated frames without
                # underflowing). This prevents frozen-frame emission forever.
                err = self._source.get_error()
                if err is not None:
                    self._fail_fast.record_producer_error(err)

                prev_underflow = getattr(self._source, "underflow_count", 0)
                frame = self._source.next_frame()
                curr_underflow = getattr(self._source, "underflow_count", 0)

                is_underflow = curr_underflow > prev_underflow
                if is_underflow:
                    self._metrics.underflow_count += 1

                trip_reason = self._fail_fast.record_frame(underflow=is_underflow)

                if trip_reason or self._fail_fast.is_tripped():
                    reason = self._fail_fast.fail_fast_reason or trip_reason
                    self._signal_fail_fast(reason)
                    break

                self._metrics.frames_generated += 1
            else:
                # ── Pre-roll phase: cached black frames ──
                frame = self._black_frame
                self._metrics.frames_repeated += 1

            # Normalize to bytes and create GStreamer buffer
            raw = frame.tobytes() if hasattr(frame, "tobytes") else frame
            buf = self._Gst.Buffer.new_wrapped(raw)
            buf.pts = self._push_counter * frame_duration_ns
            buf.dts = buf.pts
            buf.duration = frame_duration_ns
            buf.offset = self._push_counter
            buf.offset_end = self._push_counter + 1
            self._push_counter += 1

            # Pacing telemetry: track push wall-clock intervals
            _push_now = time.monotonic()
            if self._push_last_wall > 0:
                _delta_ms = (_push_now - self._push_last_wall) * 1000.0
                self._push_intervals_ms.append(_delta_ms)
                if len(self._push_intervals_ms) > 300:
                    self._push_intervals_ms = self._push_intervals_ms[-300:]
                if _delta_ms > 60.0:
                    self._push_late_count += 1
            self._push_last_wall = _push_now
            self._push_pts_deltas_ms.append(frame_duration_ns / 1_000_000.0)
            if len(self._push_pts_deltas_ms) > 300:
                self._push_pts_deltas_ms = self._push_pts_deltas_ms[-300:]

            ret = self._appsrc.emit("push-buffer", buf)
            if ret == self._Gst.FlowReturn.OK:
                self._metrics.frames_emitted += 1
            else:
                logger.warning("appsrc push-buffer returned %s", ret)

            # ── Heartbeat (every 10s, fire-and-forget) ──
            now = time.monotonic()
            if now - hb_last_time >= hb_interval:
                elapsed = now - hb_last_time
                gen_delta = self._metrics.frames_generated - hb_last_gen
                emit_delta = self._metrics.frames_emitted - hb_last_emit
                repeat_delta = self._metrics.frames_repeated - hb_last_repeat
                underflow_delta = self._metrics.underflow_count - hb_last_underflow
                gen_fps = gen_delta / elapsed if elapsed > 0 else 0.0
                emit_fps = emit_delta / elapsed if elapsed > 0 else 0.0
                phase = "live" if self._ice_connected.is_set() else "pre-roll"

                _update_crash_context(
                    phase=phase,
                    frames_generated=self._metrics.frames_generated,
                    frames_emitted=self._metrics.frames_emitted,
                    last_heartbeat=f"gen_fps={gen_fps:.1f} emit_fps={emit_fps:.1f}",
                )

                # Source-level queue depth (if available)
                qmin = qmax = None
                src_metrics = getattr(self._source, "metrics", None)
                if src_metrics and hasattr(src_metrics, "queue_depth_min"):
                    qmin = src_metrics.queue_depth_min if src_metrics.queue_depth_min < 10 ** 9 else 0
                    qmax = src_metrics.queue_depth_max

                logger.info(
                    "[heartbeat] phase=%s gen_fps=%.1f emit_fps=%.1f "
                    "repeat_delta=%d underflow_delta=%d qdepth=%s/%s",
                    phase, gen_fps, emit_fps, repeat_delta, underflow_delta,
                    qmin, qmax,
                )

                # Pacing telemetry
                _intervals = self._push_intervals_ms
                _pts_deltas = self._push_pts_deltas_ms
                if _intervals:
                    _si = sorted(_intervals)
                    _p50 = _si[len(_si) // 2]
                    _p95 = _si[int(len(_si) * 0.95)] if len(_si) >= 20 else max(_si)
                    _max = max(_si)
                    _late = self._push_late_count
                else:
                    _p50 = _p95 = _max = 0.0
                    _late = 0
                if _pts_deltas:
                    _spi = sorted(_pts_deltas)
                    _pts_p50 = _spi[len(_spi) // 2]
                    _pts_p95 = _spi[int(len(_spi) * 0.95)] if len(_spi) >= 20 else max(_spi)
                    _pts_max = max(_spi)
                else:
                    _pts_p50 = _pts_p95 = _pts_max = 0.0

                # appsrc current level
                _appsrc_buffers = _appsrc_time = None
                try:
                    _cur = self._appsrc.get_property("current-level-buffers")
                    _appsrc_buffers = int(_cur) if _cur is not None else None
                    _cur_t = self._appsrc.get_property("current-level-time")
                    if _cur_t is not None:
                        _appsrc_time = int(_cur_t) / 1_000_000_000.0
                except Exception:
                    pass

                logger.info(
                    "[pacing] push_wall_p50=%.1fms p95=%.1fms max=%.1fms late=%d "
                    "pts_delta_p50=%.1fms p95=%.1fms max=%.1fms "
                    "appsrc_level_buf=%s time=%ss frames_gen=%d emit=%d repeat_delta=%d underflow=%d",
                    _p50, _p95, _max, _late,
                    _pts_p50, _pts_p95, _pts_max,
                    _appsrc_buffers, _appsrc_time,
                    gen_delta, emit_delta, repeat_delta, underflow_delta,
                )
                # Reset pacing window
                self._push_intervals_ms = []
                self._push_pts_deltas_ms = []
                self._push_late_count = 0

                hb_last_time = now
                hb_last_gen = self._metrics.frames_generated
                hb_last_emit = self._metrics.frames_emitted
                hb_last_repeat = self._metrics.frames_repeated
                hb_last_underflow = self._metrics.underflow_count

            # Wall-clock pacing to true FPS
            next_deadline += 1.0 / FPS
            sleep_for = next_deadline - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_deadline = time.monotonic()

    def _signal_fail_fast(self, reason: str) -> None:
        """Signal pipeline that no more frames will come (fail-fast tripped).

        Sends EOS to appsrc so the pipeline finishes naturally instead of
        emitting frozen video forever.
        """
        self._metrics.fail_fast_reason = reason
        logger.error("[gst-webrtc-pub] FAIL-FAST: %s", reason)
        if self._appsrc is not None and self._Gst is not None:
            try:
                self._appsrc.emit("end-of-stream")
            except Exception as exc:
                logger.warning("EOS signal failed: %s", exc)

    # ── Testable helpers (no GStreamer dependency) ────────────────────────

    def _build_offer_body(self, sdp: str, mid: str) -> dict[str, Any]:
        """Construct the publisher-offer callback JSON body."""
        return {
            "sdp": sdp,
            "type": "offer",
            "mid": mid,
            "track_name": self._track_name,
            "kind": "video",
        }

    def _build_close_body(self) -> dict[str, Any]:
        """Construct the publisher-close callback JSON body."""
        return {
            "tracks": [{"mid": self._video_mid}],
            "force": True,
        }

    def _build_callback_headers(self) -> dict[str, str]:
        """Construct HTTP headers for Worker callbacks."""
        return {
            "x-modal-secret": self._modal_secret or "",
            "content-type": "application/json",
        }

    async def _post_offer(self, sdp: str, mid: str) -> dict[str, Any]:
        """POST the publisher-offer to the Worker callback URL."""
        body = self._build_offer_body(sdp, mid)
        headers = self._build_callback_headers()
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._callback_url, headers=headers, json=body
            )
            # Direct WebRTC debug: Worker accepts the offer and returns 202
            # while it waits for the browser's answer SDP.
            if self._direct_webrtc_debug and resp.status_code == 202:
                return {"status": "pending"}
            if resp.status_code != 200:
                raise RuntimeError(
                    f"publisher-offer callback failed: "
                    f"{resp.status_code} {resp.text}"
                )
            return resp.json()

    async def set_remote_answer(self, answer_sdp: str, answer_type: str) -> None:
        """Receive the browser's answer SDP asynchronously (direct WebRTC debug).

        Called from the Modal HTTP answer endpoint via the publisher loop.
        Stores the answer and unblocks the startup flow.
        """
        logger.info("set_remote_answer called (type=%s, len=%d)", answer_type, len(answer_sdp))
        self._answer_data = {"answer_sdp": answer_sdp, "answer_type": answer_type}
        self._ts_set(self._answer_event)

    async def _set_remote_description(self, answer_sdp: str, answer_type: str) -> None:
        """Set the remote (answer) SDP on the webrtcbin element."""
        res, sdpmsg = self._GstSdp.SDPMessage.new()
        self._GstSdp.sdp_message_parse_buffer(bytes(answer_sdp.encode()), sdpmsg)
        answer = self._GstWebRTC.WebRTCSessionDescription.new(
            self._GstWebRTC.WebRTCSDPType.ANSWER, sdpmsg
        )
        promise = self._Gst.Promise.new()
        self._webrtc.emit("set-remote-description", answer, promise)
        promise.interrupt()
        logger.info("Remote description set (type=%s)", answer_type)
        if self._recorder:
            self._recorder.record_checkpoint("remote_description_set")

    # ── ICE forensic helpers ─────────────────────────────────────────────

    def _read_final_states(self) -> dict[str, str | None]:
        """Read current webrtcbin property values for final-state snapshot."""
        result: dict[str, str | None] = {
            "ice_connection": None,
            "ice_gathering": None,
            "signaling": None,
            "connection": None,
        }
        try:
            if self._webrtc is not None:
                for prop, key in (
                    ("ice-connection-state", "ice_connection"),
                    ("ice-gathering-state", "ice_gathering"),
                    ("signaling-state", "signaling"),
                    ("connection-state", "connection"),
                ):
                    try:
                        val = self._webrtc.get_property(prop)
                        result[key] = val.value_nick if hasattr(val, "value_nick") else str(val)
                    except Exception:
                        pass
        except Exception:
            pass
        return result

    def _collect_stats_snapshot(self) -> tuple[dict[str, Any] | None, bool]:
        """Fire-and-forget get-stats introspection. Strict 250ms timeout.

        Returns (stats_dict, timed_out). Uses a Gst Promise with a safety
        timeout. Wrapped in try/except — never raises, never blocks beyond 250ms.

        Note: webrtcbin's get-stats signal expects (pad, promise) in GStreamer
        1.20+. Older versions accept just (promise). We feature-detect safely.
        """
        if self._webrtc is None or self._Gst is None:
            return None, False
        try:
            stats_box: dict[str, Any] = {}
            done = threading.Event()
            stats_structure: list[Any] = []

            def _on_stats(promise: Any) -> None:
                try:
                    reply = promise.get_reply()
                    stats_structure.append(reply)
                except Exception:
                    pass
                finally:
                    done.set()

            promise = self._Gst.Promise.new_with_change_func(_on_stats)
            # webrtcbin get-stats expects (pad, promise) in newer GStreamer.
            # Pass None for pad to request stats for all pads.
            try:
                self._webrtc.emit("get-stats", None, promise)
            except Exception:
                # Fallback: older GStreamer may accept just (promise)
                try:
                    self._webrtc.emit("get-stats", promise)
                except Exception:
                    return {"error": "get-stats emit failed (both signatures)"}, False
            completed = done.wait(timeout=0.250)
            if not completed:
                return None, True

            if not stats_structure:
                return {}, False

            reply = stats_structure[0]
            # Extract sanitized candidate-pair info (no IPs)
            try:
                for i in range(reply.n_fields()):
                    field_name = reply.nth_field_name(i)
                    field = reply.get_value(field_name)
                    if field is None:
                        continue
                    if hasattr(field, "get_value"):
                        try:
                            sub = field
                            if hasattr(sub, "n_fields"):
                                for j in range(sub.n_fields()):
                                    sub_name = sub.nth_field_name(j)
                                    if sub_name in (
                                        "selected-candidate-pair-id",
                                        "local-candidate-id",
                                        "remote-candidate-id",
                                        "nominated",
                                        "state",
                                        "bytes-sent",
                                        "requests-sent",
                                        "responses-received",
                                    ):
                                        stats_box[sub_name] = str(sub.get_value(sub_name))
                        except Exception:
                            pass
            except Exception:
                pass
            return stats_box, False
        except Exception as exc:
            logger.debug("get-stats collection error: %s", exc)
            return {"error": str(exc)[:200]}, False

    # ── Lifecycle: start ─────────────────────────────────────────────────

    async def start(self) -> str:
        """Create pipeline, exchange SDP via Worker, wait for ICE.

        Returns the video mid string.

        Raises _IceDebugError (RuntimeError subclass) if MODAL_WORKER_SECRET is
        missing, ICE gathering or connection times out, or the callback fails.
        The ice_debug dict on _IceDebugError carries sanitized forensic data.
        """
        _install_crash_forensics()
        _update_crash_context(phase="start", session_id=self._track_name)

        if not self._modal_secret:
            raise RuntimeError("MODAL_WORKER_SECRET not set in environment")

        self._loop = asyncio.get_running_loop()

        # Initialize ICE forensic recorder BEFORE any GStreamer work
        self._recorder = _IceDebugRecorder(
            attempt_id=self._track_name,
            track_name=self._track_name,
            publisher_type="gstreamer",
        )

        # Lazy import GStreamer (only needed when SFU publisher is actually used)
        self._Gst, self._GstWebRTC, self._GstSdp, self._GLib = _import_gstreamer()

        # Build pipeline and connect signals
        self._pipeline = self._build_pipeline()
        self._recorder.record_checkpoint("pipeline_created")
        self._webrtc.connect("on-negotiation-needed", self._on_negotiation_needed)
        self._webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self._webrtc.connect(
            "notify::ice-gathering-state", self._on_ice_gathering_state
        )
        self._webrtc.connect(
            "notify::ice-connection-state", self._on_ice_connection_state
        )
        # Forensic: capture peer connection + signaling state transitions
        self._webrtc.connect("notify::connection-state", self._on_connection_state)
        self._webrtc.connect("notify::signaling-state", self._on_signaling_state)
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_bus_message)
        # Forensic: capture warnings + filtered element messages
        self._bus.connect("message::warning", self._on_bus_warning)
        self._bus.connect("message::element", self._on_bus_element)

        # Start GLib MainLoop in background thread for GStreamer signal dispatch
        self._glib_loop = self._GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run, daemon=True, name="glib-mainloop"
        )
        self._glib_thread.start()
        logger.info("GLib MainLoop started in background thread")

        # Start pipeline — PLAYING state
        self._pipeline.set_state(self._Gst.State.PLAYING)
        self._recorder.record_checkpoint("pipeline_playing")

        # Start push thread immediately.
        # Pre-roll: pushes black frames until ICE connects.
        # The first pushed frame triggers on-negotiation-needed (appsrc live).
        self._push_thread = threading.Thread(
            target=self._push_frames, daemon=True, name="frame-pusher"
        )
        self._push_thread.start()

        # Wait for offer to be ready (SDP created + ICE candidates gathered)
        try:
            await asyncio.wait_for(
                self._offer_ready.wait(), timeout=ICE_GATHER_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            self._recorder.record_timeout("ice_gather")
            self._finalize_and_raise(
                f"offer_timeout: ICE gathering did not complete in "
                f"{ICE_GATHER_TIMEOUT_S}s"
            )

        if not self._local_sdp or "a=candidate:" not in self._local_sdp:
            self._recorder.record_timeout("no_candidates")
            self._finalize_and_raise("no_ice_candidates: offer has no ICE candidates")

        logger.info("Offer ready, posting to Worker callback")
        self._recorder.record_checkpoint("offer_ready")

        # Post offer to Worker (brokered — Worker proxies to SFU)
        self._recorder.record_checkpoint("offer_callback_started")
        try:
            answer_data = await self._post_offer(self._local_sdp, self._video_mid)
        except Exception as exc:
            self._recorder.record_answer(False, 0)
            self._recorder.record_checkpoint("offer_callback_failed")
            self._finalize_and_raise(f"publisher-offer callback failed: {exc}")
        self._recorder.record_checkpoint("offer_callback_completed")

        # Direct WebRTC debug: return immediately after posting the offer so the
        # Modal HTTP response can reach the Worker and the Worker can send
        # session.ready to the browser. The answer/ICE handshake continues in a
        # background task on the publisher loop.
        if self._direct_webrtc_debug and answer_data.get("status") == "pending":
            logger.info("Direct WebRTC: offer posted, returning early to unblock session.ready")
            asyncio.create_task(self._complete_direct_webrtc_startup())
            return self._video_mid

        answer_sdp = answer_data["answer_sdp"]
        answer_type = answer_data.get("answer_type", "answer")
        await self._apply_remote_answer_and_connect(answer_sdp, answer_type)

        logger.info("Publishing AVTR frames via GStreamer/NVENC, mid=%s", self._video_mid)
        return self._video_mid

    async def _complete_direct_webrtc_startup(self) -> None:
        """Background completion for direct WebRTC debug mode.

        Waits for the browser's answer SDP, applies it, waits for ICE to connect,
        and activates the real frame source. Errors are logged but do NOT propagate
        to the caller because start() has already returned.
        """
        answer_timeout = _direct_webrtc_answer_timeout_s()
        logger.info("Direct WebRTC: background wait for browser answer (%.1fs)", answer_timeout)
        try:
            await asyncio.wait_for(self._answer_event.wait(), timeout=answer_timeout)
        except asyncio.TimeoutError:
            self._recorder.record_timeout("direct_webrtc_answer")
            logger.error("direct_webrtc_answer_timeout: browser answer not received within %.1fs", answer_timeout)
            return

        answer_data = self._answer_data
        if not answer_data:
            logger.error("direct_webrtc_answer_missing: answer event fired but no SDP")
            return

        answer_sdp = answer_data["answer_sdp"]
        answer_type = answer_data.get("answer_type", "answer")
        try:
            await self._apply_remote_answer_and_connect(answer_sdp, answer_type)
        except Exception as exc:
            logger.error("Direct WebRTC background startup failed: %s", exc)

    async def _apply_remote_answer_and_connect(self, answer_sdp: str, answer_type: str) -> None:
        """Set remote answer, wait for ICE connected, then start frame source."""
        self._recorder.record_answer(True, len(answer_sdp))
        logger.info("Worker returned answer type=%s (len=%d)", answer_type, len(answer_sdp))

        # Set remote description on webrtcbin
        await self._set_remote_description(answer_sdp, answer_type)
        self._recorder.record_checkpoint("ice_connect_started")

        # Wait for ICE connected
        ice_timeout = _ice_connect_timeout_s()
        try:
            await asyncio.wait_for(
                self._ice_connected.wait(), timeout=ice_timeout
            )
        except asyncio.TimeoutError:
            self._recorder.record_timeout("ice_connect")
            self._finalize_and_raise(
                f"ice_connect_timeout: ICE did not connect in "
                f"{ice_timeout}s"
            )
        if self._ice_failed.is_set():
            self._recorder.record_timeout("ice_failed")
            self._finalize_and_raise("ice_failed: ICE connection entered FAILED state")

        logger.info(
            "ICE connected — activating frame source (GPU rendering starts)"
        )

        # Activate real frame source NOW (delayed start).
        # Before this point, the push thread pushed cheap black frames so
        # no GPU rendering work happened during SDP/ICE negotiation.
        self._source.start()

    def _finalize_and_raise(self, message: str) -> None:
        """Read final states, collect stats snapshot, then raise _IceDebugError.

        This is the decisive-failure path: every timeout/error captures the full
        forensic picture so the next failure explains itself. Never returns.
        """
        final_states = self._read_final_states()
        self._recorder.record_final_states(**final_states)
        stats, stats_timed_out = self._collect_stats_snapshot()
        self._recorder.record_stats_snapshot(stats, stats_timed_out)
        ice_debug = self._recorder.snapshot()
        logger.error("[ice-debug] %s", self._recorder.to_json())
        raise _IceDebugError(message, ice_debug)

    # ── Lifecycle: stop (idempotent, production-grade teardown) ──────────

    async def stop(self) -> dict[str, Any]:
        """Tear down the publisher in the correct order.

        Returns a metrics dict (always, even on double-call).

        Teardown order (critical for avoiding C++ SIGABRT and resource leaks):
          1. Signal push thread to stop
          2. Stop frame source (stops render/producer thread)
          3. Join push thread (wait for it to exit)
          4. Set pipeline to NULL
          5. Stop GLib MainLoop
          6. Call publisher-close callback (best-effort)
          7. Return metrics

        Calling stop() twice is safe — the second call returns cached metrics.
        """
        if self._stopped:
            return self._last_metrics or {}
        self._stopped = True
        _update_crash_context(phase="stop")

        if self._recorder:
            self._recorder.record_cleanup_start()

        cleanup_errors: list[str] = []
        try:
            self._signal_push_stop()
            logger.info("[stop] 1/7 push_stop signaled")
        except Exception as exc:
            cleanup_errors.append(f"push_stop: {exc}")
        try:
            self._stop_frame_source()
            logger.info("[stop] 2/7 frame_source stopped")
        except Exception as exc:
            cleanup_errors.append(f"source_stop: {exc}")
        try:
            self._join_push_thread()
            logger.info("[stop] 3/7 push_thread joined")
        except Exception as exc:
            cleanup_errors.append(f"push_join: {exc}")
        try:
            self._set_pipeline_null()
            logger.info("[stop] 4/7 pipeline set to NULL")
        except Exception as exc:
            cleanup_errors.append(f"pipeline_null: {exc}")
        try:
            if self._h264_capture_file:
                self._h264_capture_file.close()
                self._h264_capture_file = None
                logger.info("H264 capture closed: %d buffers written", self._h264_capture_count)
        except Exception as exc:
            cleanup_errors.append(f"h264_capture_close: {exc}")
        try:
            rtp_summary = self._dump_rtp_pay_validation()
            if rtp_summary:
                logger.info("RTP validation summary: %s", rtp_summary)
        except Exception as exc:
            cleanup_errors.append(f"rtp_validation_dump: {exc}")
        try:
            self._stop_glib_loop()
            logger.info("[stop] 5/7 glib_loop stopped")
        except Exception as exc:
            cleanup_errors.append(f"glib_stop: {exc}")
        try:
            await self._call_close_callback()
            logger.info("[stop] 6/7 close_callback done")
        except Exception as exc:
            cleanup_errors.append(f"close_callback: {exc}")

        if self._recorder:
            self._recorder.record_cleanup_complete(ok=len(cleanup_errors) == 0)

        self._last_metrics = self._collect_metrics()
        if cleanup_errors:
            self._last_metrics["cleanup_errors"] = cleanup_errors
        return self._last_metrics

    def get_candidate_summary(self) -> dict[str, Any]:
        """Return sanitized candidate summary for success-path telemetry.

        Used to compare pass-vs-fail candidate types without deploying the
        full ice_debug recorder snapshot. Safe fields only — no IPs, no creds.
        """
        if self._recorder is None:
            return {}
        snap = self._recorder.snapshot()
        return {
            "local_candidate_count": snap["local_candidate_count"],
            "local_candidate_types": snap["local_candidate_types"],
            "local_candidate_protocols": snap["local_candidate_protocols"],
            "local_candidate_foundation_count": snap["local_candidate_foundation_count"],
            "answer_received": snap["answer_received"],
            "answer_sdp_length": snap["answer_sdp_length"],
            "ice_connection_transition_count": len(snap["ice_connection_transitions"]),
            "ice_gathering_transition_count": len(snap["ice_gathering_transitions"]),
            "turn_configured": snap.get("turn_configured", False),
            "turn_url_count": snap.get("turn_url_count", 0),
            "turn_transports": snap.get("turn_transports", []),
            "selected_local_candidate_type": snap.get("selected_local_candidate_type"),
            "selected_remote_candidate_type": snap.get("selected_remote_candidate_type"),
            "checkpoints": snap["checkpoints"],
        }

    def _signal_push_stop(self) -> None:
        self._stop_push.set()

    def _stop_frame_source(self) -> None:
        try:
            self._source.stop()
        except Exception as exc:
            logger.warning("frame source stop error: %s", exc)

    def _join_push_thread(self) -> None:
        if self._push_thread and self._push_thread.is_alive():
            self._push_thread.join(timeout=5)
            if self._push_thread.is_alive():
                logger.warning("push thread did not exit within 5s")

    def _set_pipeline_null(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.set_state(self._Gst.State.NULL)
                logger.info("Pipeline set to NULL")
            except Exception as exc:
                logger.warning("pipeline NULL error: %s", exc)

    def _stop_glib_loop(self) -> None:
        if self._glib_loop is not None:
            try:
                self._glib_loop.quit()
            except Exception as exc:
                logger.warning("GLib loop quit error: %s", exc)
        if self._glib_thread and self._glib_thread.is_alive():
            self._glib_thread.join(timeout=2)
            if self._glib_thread.is_alive():
                logger.warning("GLib thread did not exit within 2s")

    async def _call_close_callback(self) -> None:
        if not self._close_url or not self._video_mid:
            return
        body = self._build_close_body()
        headers = self._build_callback_headers()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.put(
                    self._close_url, headers=headers, json=body
                )
                logger.info(
                    "publisher-close: status=%s body=%s",
                    resp.status_code,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.warning("publisher-close error: %s", exc)

    # ── Metrics collection ───────────────────────────────────────────────

    def _collect_metrics(self) -> dict[str, Any]:
        """Assemble the final metrics dict from publisher + source + fail-fast."""
        # Surface producer-side fatal errors
        err = self._source.get_error()
        if err is not None:
            self._metrics.producer_error = (
                f"{type(err).__name__}: {str(err)[:300]}"
            )

        # Surface fail-fast reason
        if self._fail_fast.fail_fast_reason:
            self._metrics.fail_fast_reason = self._fail_fast.fail_fast_reason

        # Surface pipeline bus error
        if self._pipeline_error:
            self._metrics.pipeline_error = self._pipeline_error

        # Merge source-side metrics if available
        source_metrics = getattr(self._source, "metrics", None)
        if source_metrics and hasattr(source_metrics, "snapshot"):
            snap = source_metrics.snapshot()
            qmin = snap.get("queue_depth_min")
            qmax = snap.get("queue_depth_max")
            if qmin is not None:
                self._metrics.queue_depth_min = qmin
            if qmax is not None:
                self._metrics.queue_depth_max = qmax
            # Surface source-level underflow/repeat counts so final metrics
            # distinguish pre-roll repeats from live-phase starvation.
            src_underflow = snap.get("underflow_count", 0)
            src_repeat = snap.get("frames_repeated", 0)
            self._metrics.source_underflow_count = src_underflow
            self._metrics.source_frames_repeated = src_repeat

        encoder_info = dict(_ENCODER_INFO, bitrate=self._bitrate, direct_webrtc_debug=self._direct_webrtc_debug)
        return self._metrics.snapshot(encoder_info=encoder_info)


# ── Retry wrapper ────────────────────────────────────────────────────────────

async def start_publisher_with_retry(
    factory: Callable[[], "GstWebRtcPublisher"],
    *,
    max_attempts: int | None = None,
) -> tuple["GstWebRtcPublisher", str]:
    """Start a WebRTC publisher with clean retry on ICE failures.

    Creates a fresh publisher per attempt via *factory*. On retryable ICE
    failures (timeout, ICE failed), performs full cleanup and retries.
    Non-retryable errors (CUDA, pipeline, auth) are re-raised immediately.

    Args:
        factory: Callable that returns a NEW GstWebRtcPublisher instance.
            Must construct the frame_source, set URLs, and pass ice_servers.
            Each call MUST create a completely fresh publisher — no reuse.
        max_attempts: Override max attempts (default: env or 2).

    Returns:
        (publisher, video_mid) — the successful publisher and its video mid.
        The caller MUST store the returned publisher for session lifecycle.

    Raises:
        PublisherIceFailedAfterRetries: All attempts exhausted (ICE only).
        Exception: Non-retryable error from any attempt (re-raised as-is).
    """
    attempts = max_attempts if max_attempts is not None else _max_publisher_attempts()
    summaries: list[dict[str, Any]] = []

    for attempt_num in range(1, attempts + 1):
        publisher = factory()
        logger.info(
            "[publisher-retry] attempt %d/%d (ice_timeout=%ds)",
            attempt_num, attempts, _ice_connect_timeout_s(),
        )
        try:
            video_mid = await publisher.start()
            if attempt_num > 1:
                logger.info(
                    "[publisher-retry] succeeded on attempt %d/%d",
                    attempt_num, attempts,
                )
            return publisher, video_mid
        except Exception as exc:
            ice_debug = getattr(exc, "ice_debug", None)

            # Non-retryable: re-raise immediately after cleanup.
            if not _is_retryable_ice_error(exc):
                logger.warning(
                    "[publisher-retry] non-retryable error on attempt %d: %s",
                    attempt_num, exc,
                )
                try:
                    await publisher.stop()
                except Exception as stop_exc:
                    logger.warning(
                        "[publisher-retry] cleanup error (non-retryable): %s",
                        stop_exc,
                    )
                raise

            # Retryable: record summary, cleanup, continue.
            logger.warning(
                "[publisher-retry] retryable ICE failure on attempt %d/%d: %s",
                attempt_num, attempts, exc,
            )

            # Re-read ice_debug AFTER cleanup to include cleanup_ok timing.
            recorder = getattr(publisher, "_recorder", None)
            try:
                await publisher.stop()
            except Exception as stop_exc:
                logger.warning(
                    "[publisher-retry] cleanup error (retryable): %s",
                    stop_exc,
                )
            if recorder is not None:
                ice_debug = recorder.snapshot()

            summaries.append({
                "attempt": attempt_num,
                "error": str(exc)[:300],
                "ice_debug": ice_debug,
            })

    raise PublisherIceFailedAfterRetries(
        f"sfu_publisher_ice_failed_after_retries: {len(summaries)} attempts exhausted",
        summaries,
    )


# ── Encode-Only Publisher (crash isolation matrix rows D/E) ───────────────────

class EncodeOnlyPublisher(GstWebRtcPublisher):
    """NVENC encode-only publisher — no WebRTC, no WHIP, no ICE, no TURN.

    Inherits the push thread, heartbeat, fail-fast, pacing, crash forensics,
    and teardown from GstWebRtcPublisher. Overrides:

      - __init__: No callback URLs, no ice_servers, no MODAL_WORKER_SECRET.
      - start(): No SDP, no offer, no ICE wait. Pipeline → PLAYING, source
        activated immediately, push thread enters live phase via the
        _ice_connected trick.
      - _build_encode_pipeline(): Uses encode-only template ending in fakesink.

    stop() is inherited unchanged. It does NOT touch webrtcbin/ICE/TURN/WHIP
    state — it only signals push stop, stops source, joins thread, sets
    pipeline NULL, stops GLib, and skips the empty close callback URL.
    """

    def __init__(
        self,
        frame_source: FrameSource,
        track_name: str,
        *,
        encoder: str = "nvh264enc",
        bitrate: int = BITRATE_DEFAULT,
    ) -> None:
        super().__init__(
            frame_source=frame_source,
            track_name=track_name,
            publisher_callback_url="",
            close_callback_url="",
            bitrate=bitrate,
            ice_servers=None,
            encoder=encoder,
        )

    def _build_encode_pipeline(self) -> Any:
        template = (
            _PIPELINE_TEMPLATE_ENCODE_ONLY_X264
            if self._encoder == "x264enc"
            else _PIPELINE_TEMPLATE_ENCODE_ONLY
        )
        pipe_desc = template.format(
            width=WIDTH,
            height=HEIGHT,
            fps=FPS,
            gop=GOP_SIZE,
            bitrate=self._bitrate,
        )
        logger.info("[encode-only] Pipeline:\n%s", pipe_desc)
        logger.info(
            "[encode-only] encoder=%s %dx%d bitrate=%d gop=%d",
            self._encoder, WIDTH, HEIGHT, self._bitrate, GOP_SIZE,
        )
        pipeline = self._Gst.parse_launch(pipe_desc)
        self._appsrc = pipeline.get_by_name("src")

        _appsrc_src_pad = self._appsrc.get_static_pad("src")
        if _appsrc_src_pad is not None:
            _appsrc_src_pad.add_probe(
                self._Gst.PadProbeType.BUFFER,
                self._make_pts_probe_callback(),
            )
            logger.info("[encode-only] PTS sanity probe installed on appsrc:src")

        return pipeline

    async def start(self) -> str:
        _install_crash_forensics()
        _update_crash_context(
            phase="start",
            session_id=self._track_name,
            publisher_mode="encode_only",
        )

        self._loop = asyncio.get_running_loop()
        self._Gst, self._GstWebRTC, self._GstSdp, self._GLib = _import_gstreamer()

        self._pipeline = self._build_encode_pipeline()
        self._bus = self._pipeline.get_bus()
        self._bus.add_signal_watch()
        self._bus.connect("message::error", self._on_bus_message)
        self._bus.connect("message::warning", self._on_bus_warning)

        self._glib_loop = self._GLib.MainLoop()
        self._glib_thread = threading.Thread(
            target=self._glib_loop.run, daemon=True, name="glib-mainloop"
        )
        self._glib_thread.start()
        logger.info("[encode-only] GLib MainLoop started")

        self._pipeline.set_state(self._Gst.State.PLAYING)
        logger.info("[encode-only] Pipeline set to PLAYING")

        self._source.start()
        logger.info("[encode-only] Frame source started")

        self._ice_connected.set()

        self._push_thread = threading.Thread(
            target=self._push_frames, daemon=True, name="frame-pusher"
        )
        self._push_thread.start()
        logger.info("[encode-only] Push thread started — live phase immediate")

        return "encode-only"


__all__ = [
    "GstWebRtcPublisher",
    "EncodeOnlyPublisher",
    "FrameSource",
    "SyntheticFrameSource",
    "_FailFastPolicy",
    "_IceDebugRecorder",
    "_IceDebugError",
    "PublisherIceFailedAfterRetries",
    "_sanitize_candidate",
    "_redact_debug_text",
    "_redact_turn_uri",
    "_ice_servers_to_gstreamer_uris",
    "_extract_turn_transports",
    "_is_retryable_ice_error",
    "_ice_connect_timeout_s",
    "_max_publisher_attempts",
    "start_publisher_with_retry",
    "_black_i420_bytes",
]
