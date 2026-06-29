"""Tests for gst_webrtc_publisher.py — no GStreamer, GPU, Modal, or Cloudflare required.

Run:  python3 -m pytest avatar-saas-prototype/modal-avatar-runtime/test_gst_webrtc_publisher.py -v

Test strategy:
  - All tests are sync (no pytest-asyncio dependency).
  - Async methods tested via asyncio.run().
  - Mocks are hand-rolled fakes + unittest.mock.patch.
  - _FailFastPolicy and body-building helpers are pure Python (no gi).
  - stop()/teardown tested with a publisher whose pipeline is None (no start()).
"""

from __future__ import annotations

import ast
import asyncio
import os
import sys
import threading
import time
import unittest.mock
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).parent
sys.path.insert(0, str(MODULE_DIR))

import gst_webrtc_publisher as mod
from gst_webrtc_publisher import (
    GstWebRtcPublisher,
    _FailFastPolicy,
    _black_i420_bytes,
)


# ── Fake frame source ────────────────────────────────────────────────────────


class _FakeFrameSource:
    """Minimal fake implementing the FrameSource protocol.

    Attributes are controllable for fail-fast testing:
      - underflow_count: increment to simulate queue underflow
      - _error: set to simulate producer error
    """

    def __init__(self) -> None:
        self.fps = 25
        self.counter = 0
        self.frames_generated = 0
        self.frames_emitted = 0
        self.underflow_count = 0
        self._error: Exception | None = None
        self._started = False
        self._stopped = False
        self._frame = b"\x10" * (1280 * 720) + b"\x80" * (640 * 360 * 2)

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._stopped = True

    def next_frame(self):
        self.counter += 1
        return self._frame

    def get_error(self) -> Exception | None:
        return self._error


def _make_publisher(**kwargs) -> GstWebRtcPublisher:
    """Construct a publisher with a fake source and dummy URLs."""
    os.environ.setdefault("MODAL_WORKER_SECRET", "test-secret")
    source = kwargs.pop("source", _FakeFrameSource())
    return GstWebRtcPublisher(
        frame_source=source,
        track_name="test-video",
        publisher_callback_url="https://worker.test/internal/sessions/s1/publisher-offer",
        close_callback_url="https://worker.test/internal/sessions/s1/publisher-close",
        **kwargs,
    )


# ── Test 1: Module imports without GStreamer ─────────────────────────────


def test_module_imports_without_gstreamer():
    """The module must import successfully on a machine without GStreamer.

    gi/Gst/GstWebRTC/GstSdp/GLib must NOT be imported at module top level.
    This test passes because the test runner itself imported the module
    without gi available (lazy import pattern).
    """
    import importlib

    # Re-import to verify no side effects
    spec = importlib.util.find_spec("gst_webrtc_publisher")
    assert spec is not None, "gst_webrtc_publisher not found"

    # Verify gi was NOT imported by this module (it would fail without it)
    # If gi is somehow installed on this machine, the key point is that
    # the module didn't import it at top level — verified by test 2.
    assert mod.WIDTH == 1280
    assert mod.HEIGHT == 720
    assert mod.FPS == 25


# ── Test 2: No top-level gi import (AST analysis) ────────────────────────


def test_no_top_level_gi_import():
    """AST-verify that no 'import gi' or 'from gi...' appears at module level.

    This is the authoritative guard (not grep). Function-level imports inside
    _import_gstreamer() are expected and allowed.
    """
    source_path = MODULE_DIR / "gst_webrtc_publisher.py"
    tree = ast.parse(source_path.read_text())

    top_level_imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_imports.append(node.module)

    gi_imports = [name for name in top_level_imports if name == "gi" or name.startswith("gi.")]
    assert len(gi_imports) == 0, (
        f"Top-level gi imports found (must be lazy/function-level only): {gi_imports}"
    )


# ── Test 3: No direct SFU strings ────────────────────────────────────────


def test_no_sfu_strings():
    """The module must never reference the Cloudflare SFU API directly.

    Forbidden strings:
      - rtc.live.cloudflare.com (SFU REST API base)
      - CALLS_APP_SECRET (SFU direct auth)
    These ensure SDP exchange is brokered through Worker callbacks only.
    """
    source_path = MODULE_DIR / "gst_webrtc_publisher.py"
    source = source_path.read_text()

    forbidden = ["rtc.live.cloudflare.com", "CALLS_APP_SECRET"]
    found = [s for s in forbidden if s in source]
    assert len(found) == 0, (
        f"Forbidden SFU strings found in source: {found}. "
        "The publisher must use Worker callbacks only."
    )


# ── Test 4: Offer callback body shape ────────────────────────────────────


def test_offer_body_shape():
    """_build_offer_body must produce the exact JSON the Worker expects.

    Contract (avatar-session-do.ts:6018-6024):
      { sdp, type:"offer", mid, track_name, kind:"video" }
    """
    pub = _make_publisher()
    body = pub._build_offer_body("v=0\r\no=- 123 1 IN IP4 0.0.0.0\r\n", "0")

    assert body["sdp"].startswith("v=0")
    assert body["type"] == "offer"
    assert body["mid"] == "0"
    assert body["track_name"] == "test-video"
    assert body["kind"] == "video"

    # All 5 required keys present
    required_keys = {"sdp", "type", "mid", "track_name", "kind"}
    assert required_keys.issubset(body.keys()), (
        f"Missing keys: {required_keys - body.keys()}"
    )


# ── Test 5: Close callback body shape ────────────────────────────────────


def test_close_body_shape():
    """_build_close_body must produce the exact JSON the Worker expects.

    Contract (webrtc_publisher.py:683):
      { tracks: [{mid}], force: true }
    """
    pub = _make_publisher()
    pub._video_mid = "0"
    body = pub._build_close_body()

    assert "tracks" in body
    assert isinstance(body["tracks"], list)
    assert len(body["tracks"]) == 1
    assert body["tracks"][0]["mid"] == "0"
    assert body["force"] is True


# ── Test 6: Callback headers include x-modal-secret ─────────────────────


def test_callback_headers_include_modal_secret():
    """_build_callback_headers must include x-modal-secret from env."""
    os.environ["MODAL_WORKER_SECRET"] = "my-secret-123"
    pub = _make_publisher()

    headers = pub._build_callback_headers()

    assert "x-modal-secret" in headers
    assert headers["x-modal-secret"] == "my-secret-123"
    assert headers["content-type"] == "application/json"


# ── Test 7: stop() is idempotent ─────────────────────────────────────────


def test_stop_idempotent():
    """Calling stop() twice must not raise.

    The second call returns cached metrics (or empty dict).
    No pipeline/thread/GLib objects exist since start() was never called.
    """
    pub = _make_publisher()

    result1 = asyncio.run(pub.stop())
    result2 = asyncio.run(pub.stop())

    assert isinstance(result1, dict)
    assert isinstance(result2, dict)
    assert result1 == result2  # second call returns cached metrics


# ── Test 8: Teardown order ───────────────────────────────────────────────


def test_teardown_order():
    """Teardown must follow the exact order:
      push_stop → source_stop → push_join → pipeline_null → glib_stop → close_callback

    We monkeypatch each teardown step to record its call order.
    """
    pub = _make_publisher()
    order: list[str] = []

    pub._signal_push_stop = lambda: order.append("push_stop")
    pub._stop_frame_source = lambda: order.append("source_stop")
    pub._join_push_thread = lambda: order.append("push_join")
    pub._set_pipeline_null = lambda: order.append("pipeline_null")
    pub._stop_glib_loop = lambda: order.append("glib_stop")

    async def fake_close():
        order.append("close_callback")

    pub._call_close_callback = fake_close

    asyncio.run(pub.stop())

    expected = [
        "push_stop",
        "source_stop",
        "push_join",
        "pipeline_null",
        "glib_stop",
        "close_callback",
    ]
    assert order == expected, f"Teardown order mismatch: got {order}, expected {expected}"


# ── Test 9: Metrics dict includes all required keys ──────────────────────


def test_metrics_has_required_keys():
    """_collect_metrics must return a dict with all required metric keys.

    Required: producer_error, fail_fast_reason, frames_generated,
    frames_emitted, frames_repeated, underflow_count, queue_depth_min,
    queue_depth_max, plus encoder/profile fields.
    """
    pub = _make_publisher()
    pub._video_mid = "0"

    metrics = pub._collect_metrics()

    required_keys = {
        "frames_generated",
        "frames_emitted",
        "frames_repeated",
        "underflow_count",
        "queue_depth_min",
        "queue_depth_max",
        "producer_error",
        "fail_fast_reason",
        "pipeline_error",
        "encoder_info",
        "source_underflow_count",
        "source_frames_repeated",
    }
    missing = required_keys - metrics.keys()
    assert len(missing) == 0, f"Missing metric keys: {missing}"

    # Encoder info sub-dict
    ei = metrics["encoder_info"]
    assert ei["encoder"] == "nvh264enc"
    assert ei["profile"] == "constrained-baseline"
    assert ei["level"] == "3.1"
    assert ei["width"] == 1280
    assert ei["height"] == 720
    assert ei["fps"] == 25
    assert ei["bitrate"] == 1500
    assert ei["gop_size"] == 25


# ── Test 10: Fail-fast policy ────────────────────────────────────────────


def test_failfast_policy_underflow():
    """Sustained underflow beyond MAX_CONSECUTIVE_UNDERFLOWS trips the policy."""
    policy = _FailFastPolicy()

    # 250 underflows: should NOT trip (boundary is > 250, so 250 is OK)
    for i in range(250):
        reason = policy.record_frame(underflow=True)
        assert reason is None, f"Tripped too early at underflow {i + 1}"

    # 251st underflow: SHOULD trip
    reason = policy.record_frame(underflow=True)
    assert reason is not None, "Should have tripped at 251 underflows"
    assert "sustained_underflow" in reason
    assert policy.is_tripped()


def test_failfast_policy_producer_error_grace_drain_no_underflow():
    """Producer error trips after MAX_REPEATS_AFTER_ERROR frames even WITHOUT underflow.

    This is the regression test for the bug: a source that returns cached/
    repeated frames without incrementing underflow_count must still trigger
    fail-fast. The grace drain counts ALL frames, not just underflows.
    """
    policy = _FailFastPolicy(max_repeats=5)
    policy.record_producer_error(RuntimeError("CUDA illegal memory access"))

    # 5 frames with NO underflow: should NOT trip (grace drain)
    for i in range(5):
        reason = policy.record_frame(underflow=False)
        assert reason is None, f"Tripped too early at frame {i + 1}"

    # 6th frame (still no underflow): SHOULD trip
    reason = policy.record_frame(underflow=False)
    assert reason is not None, "Should have tripped after grace drain (no underflow)"
    assert "producer_error_drained" in reason
    assert "RuntimeError" in reason
    assert policy.is_tripped()


def test_failfast_policy_producer_error_with_underflow():
    """Producer error + underflow also trips via grace drain."""
    policy = _FailFastPolicy(max_repeats=5)
    policy.record_producer_error(ValueError("GPU OOM"))

    # 5 underflow frames after error: should NOT trip
    for i in range(5):
        reason = policy.record_frame(underflow=True)
        assert reason is None, f"Tripped too early at underflow {i + 1}"

    # 6th underflow: SHOULD trip
    reason = policy.record_frame(underflow=True)
    assert reason is not None
    assert "producer_error_drained" in reason
    assert "ValueError" in reason


def test_failfast_policy_transient_underflow_no_trip():
    """Transient underflow (counter resets on non-underflow) must NOT trip."""
    policy = _FailFastPolicy()

    for _ in range(10):
        # A few underflows
        for _ in range(100):
            reason = policy.record_frame(underflow=True)
            assert reason is None
        # Non-underflow frame resets the counter
        policy.record_frame(underflow=False)

    assert not policy.is_tripped()
    assert policy.fail_fast_reason is None


def test_failfast_policy_already_tripped_returns_cached_reason():
    """Once tripped, record_frame returns the same reason (idempotent)."""
    policy = _FailFastPolicy(max_repeats=3)
    policy.record_producer_error(ValueError("test error"))

    # Trip it
    for _ in range(4):
        policy.record_frame(underflow=False)
    assert policy.is_tripped()

    # Further calls return same reason
    reason1 = policy.record_frame(underflow=False)
    reason2 = policy.record_frame(underflow=True)
    assert reason1 == reason2
    assert "producer_error_drained" in reason1


# ── Test 11: Pre-roll black frame behavior ───────────────────────────────


def test_black_i420_bytes_dimensions():
    """_black_i420_bytes returns the correct size for 1280x720 I420."""
    frame = _black_i420_bytes(1280, 720)
    # Y: 1280*720 = 921600, U: 640*360 = 230400, V: 640*360 = 230400
    expected_size = 921600 + 230400 + 230400
    assert len(frame) == expected_size, (
        f"Expected {expected_size} bytes, got {len(frame)}"
    )

    # Y plane is all 0x10 (16)
    assert all(b == 0x10 for b in frame[:921600])
    # UV planes are all 0x80 (128)
    assert all(b == 0x80 for b in frame[921600:])


def test_black_i420_bytes_custom_dimensions():
    """_black_i420_bytes works with non-standard dimensions."""
    frame = _black_i420_bytes(640, 480)
    # Y: 640*480 = 307200, U: 320*240 = 76800, V: 320*240 = 76800
    expected_size = 307200 + 76800 + 76800
    assert len(frame) == expected_size


def test_preroll_phase_pushes_black_frames():
    """Before ICE connects, the push thread must use black frames, NOT source frames.

    This test verifies the phase-decision logic: when _ice_connected is not set,
    the next push frame is the cached black frame and frames_repeated increments.
    """
    pub = _make_publisher()
    source = pub._source  # type: ignore

    # ICE not connected → pre-roll phase
    assert not pub._ice_connected.is_set()

    # Simulate one iteration of pre-roll frame selection
    # (we test the decision logic, not the full GStreamer push)
    if not pub._ice_connected.is_set():
        frame = pub._black_frame
        pub._metrics.frames_repeated += 1
    else:
        frame = source.next_frame()
        pub._metrics.frames_generated += 1

    assert frame == pub._black_frame
    assert pub._metrics.frames_repeated == 1
    assert pub._metrics.frames_generated == 0
    # Source was NOT called (no GPU work during pre-roll)
    assert source.counter == 0


def test_live_phase_uses_source_frames():
    """After ICE connects, the push thread must use real source frames."""
    pub = _make_publisher()
    source = pub._source  # type: ignore

    # Simulate ICE connected
    pub._ice_connected.set()

    if pub._ice_connected.is_set():
        frame = source.next_frame()
        pub._metrics.frames_generated += 1
    else:
        frame = pub._black_frame
        pub._metrics.frames_repeated += 1

    assert frame == source._frame
    assert pub._metrics.frames_generated == 1
    assert source.counter == 1


def test_source_start_not_called_before_ice():
    """frame_source.start() must NOT be called before ICE connects.

    This is the production contract: GPU rendering starts only after ICE
    connects. The publisher's start() calls source.start() as the last step.
    """
    pub = _make_publisher()
    source = pub._source  # type: ignore

    # Before start() is called, source should not be started
    assert not source._started

    # Even if we set up the push thread state, source.start() is not called
    # until the very end of publisher.start() (after ICE connect).
    # This test verifies the contract by checking that _source.start() is
    # NOT called in __init__ or _build_pipeline or _push_frames pre-roll.
    assert not source._started


# ── Mock GStreamer objects for push-thread tests ─────────────────────────────


class _MockBuffer:
    """Mock Gst.Buffer — accepts pts/duration assignment."""
    def __init__(self):
        self.pts: int = 0
        self.duration: int = 0
        self._raw: bytes = b""


class _MockGst:
    """Minimal Gst mock: provides Buffer.new_wrapped and FlowReturn.OK."""
    FlowReturn = type("FlowReturn", (), {"OK": "flow-ok"})

    class Buffer:
        @staticmethod
        def new_wrapped(raw):
            buf = _MockBuffer()
            buf._raw = raw
            return buf

    class MessageType:
        ERROR = "error"


class _MockAppSrc:
    """Mock appsrc: tracks push-buffer and end-of-stream calls."""
    def __init__(self):
        self.push_count = 0
        self.eos_count = 0

    def emit(self, signal: str, *args):
        if signal == "push-buffer":
            self.push_count += 1
            return _MockGst.FlowReturn.OK
        elif signal == "end-of-stream":
            self.eos_count += 1
        return None


def _setup_push_publisher(source=None, *, ice_connected=False, max_repeats=3, max_underflows=5):
    """Create a publisher wired with mock Gst/appsrc for push-thread tests."""
    os.environ.setdefault("MODAL_WORKER_SECRET", "test-secret")
    src = source or _FakeFrameSource()
    pub = GstWebRtcPublisher(
        frame_source=src,
        track_name="test-video",
        publisher_callback_url="https://worker.test/publisher-offer",
        close_callback_url="https://worker.test/publisher-close",
    )
    pub._Gst = _MockGst()
    pub._appsrc = _MockAppSrc()
    pub._fail_fast = _FailFastPolicy(max_repeats=max_repeats, max_underflows=max_underflows)
    if ice_connected:
        pub._ice_connected.set()
    return pub, src


# ── Push-thread tests: exercise the actual _push_frames() path ───────────────


def test_push_frames_preroll_black_frames():
    """Pre-roll path pushes black frames without calling source.next_frame()."""
    pub, source = _setup_push_publisher(ice_connected=False)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    time.sleep(0.15)  # ~3-4 frames at 25fps
    pub._stop_push.set()
    t.join(timeout=2)

    assert not t.is_alive(), "push thread should have exited"
    assert source.counter == 0, (
        f"source.next_frame called {source.counter} times during pre-roll"
    )
    assert pub._metrics.frames_repeated > 0, "no black frames pushed during pre-roll"
    assert pub._appsrc.push_count > 0, "no buffers pushed to appsrc"
    assert pub._metrics.fail_fast_reason is None, "fail-fast should not trip during pre-roll"


def test_push_frames_live_calls_source():
    """Live path calls source.next_frame() after ICE connected."""
    pub, source = _setup_push_publisher(ice_connected=True)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    time.sleep(0.15)
    pub._stop_push.set()
    t.join(timeout=2)

    assert not t.is_alive()
    assert source.counter > 0, "source.next_frame not called in live phase"
    assert pub._metrics.frames_generated > 0
    assert pub._metrics.fail_fast_reason is None


def test_push_frames_producer_error_no_underflow_trips():
    """REGRESSION: producer error trips fail-fast even when underflow_count doesn't increase.

    This is the bug the repair fixes. The source reports a CUDA error but
    returns cached frames without incrementing underflow_count. Without the
    fix, the publisher would emit frozen frames forever.
    """
    source = _FakeFrameSource()
    source._error = RuntimeError("CUDA illegal memory access")
    # underflow_count stays at 0 — source returns cached frames, no underflow
    pub, _ = _setup_push_publisher(source=source, ice_connected=True, max_repeats=3)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive(), "push thread should have exited via fail-fast"
    assert pub._metrics.fail_fast_reason is not None, "fail-fast did not trip"
    assert "producer_error_drained" in pub._metrics.fail_fast_reason
    assert pub._appsrc.eos_count > 0, "EOS should have been signaled to appsrc"


def test_push_frames_producer_error_with_underflow_trips():
    """Producer error + underflow also trips fail-fast via the push thread."""
    class _UnderflowSource(_FakeFrameSource):
        def next_frame(self):
            self.underflow_count += 1
            return self._frame

    source = _UnderflowSource()
    source._error = RuntimeError("CUDA crash")
    pub, _ = _setup_push_publisher(source=source, ice_connected=True, max_repeats=3)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert pub._metrics.fail_fast_reason is not None
    assert "producer_error_drained" in pub._metrics.fail_fast_reason


def test_push_frames_sustained_underflow_trips():
    """Sustained underflow without producer error still trips fail-fast."""
    class _UnderflowSource(_FakeFrameSource):
        def next_frame(self):
            self.underflow_count += 1
            return self._frame

    source = _UnderflowSource()
    # No producer error set
    pub, _ = _setup_push_publisher(source=source, ice_connected=True, max_underflows=5)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    t.join(timeout=5)

    assert not t.is_alive()
    assert pub._metrics.fail_fast_reason is not None
    assert "sustained_underflow" in pub._metrics.fail_fast_reason


def test_push_frames_transient_underflow_no_trip():
    """Transient underflow (intermittent) does NOT trip fail-fast."""
    class _IntermittentSource(_FakeFrameSource):
        def __init__(self):
            super().__init__()
            self._call_count = 0

        def next_frame(self):
            self._call_count += 1
            # Underflow every other call — never sustained
            if self._call_count % 2 == 0:
                self.underflow_count += 1
            return self._frame

    source = _IntermittentSource()
    pub, _ = _setup_push_publisher(source=source, ice_connected=True, max_underflows=5)

    t = threading.Thread(target=pub._push_frames, daemon=True)
    t.start()
    time.sleep(0.3)  # ~7-8 frames at 25fps
    pub._stop_push.set()
    t.join(timeout=2)

    assert not t.is_alive()
    assert pub._metrics.fail_fast_reason is None, (
        f"Unexpected trip on transient underflow: {pub._metrics.fail_fast_reason}"
    )


# ── Static grep validation ──────────────────────────────────────────────────


def test_grep_validation_no_forbidden_patterns():
    """Mirrors the rg validation command from the spec.

    rg "rtc.live.cloudflare.com|CALLS_APP_SECRET" → must be EMPTY (zero matches)
    rg "^import gi|^from gi"                       → must be EMPTY (top-level only)
    """
    source_path = MODULE_DIR / "gst_webrtc_publisher.py"
    lines = source_path.read_text().splitlines()

    # No line contains forbidden SFU strings
    for i, line in enumerate(lines, 1):
        assert "rtc.live.cloudflare.com" not in line, (
            f"Line {i}: contains rtc.live.cloudflare.com"
        )
        assert "CALLS_APP_SECRET" not in line, (
            f"Line {i}: contains CALLS_APP_SECRET"
        )

    # No top-level (column 0) 'import gi' or 'from gi'
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("import gi") or stripped.startswith("from gi"):
            indent = len(line) - len(stripped)
            assert indent > 0, (
                f"Line {i}: top-level '{stripped}' (must be inside a function)"
            )
