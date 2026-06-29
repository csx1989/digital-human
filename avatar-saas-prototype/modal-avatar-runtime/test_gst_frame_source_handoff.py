"""Tests for GstAvtrFrameSource handoff-driven audio sourcing (Phase 3C).

Verifies that the SFU GStreamer frame source consumes real TTS audio from the
session's _AudioHandoff queue instead of rendering silence forever.

Run:  python3 -m pytest avatar-saas-prototype/modal-avatar-runtime/test_gst_frame_source_handoff.py -v

Strategy:
  - The helper _source_audio_chunk is extracted via AST and exec'd with a
    fake GstAvtrFrameSurface namespace (for class-level constants).  No
    GPU/Modal/GStreamer imports required.
  - AST/string assertions prove the production code actually wires the
    handoff through __init__, _start_session_webrtc, and _render_loop.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = Path(__file__).parent
FIVE_Q_PATH = MODULE_DIR / "modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_method_source(source_path: Path, class_name: str, method_name: str) -> str:
    """Extract a method's source text (without decorators) from a class."""
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    seg = ast.get_source_segment(source, item)
                    assert seg is not None
                    lines = seg.strip().split("\n")
                    if lines and lines[0].strip().startswith("@"):
                        lines = lines[1:]
                    return "\n".join(lines)
    pytest.fail(f"{class_name}.{method_name} not found")


def _extract_class_constants(source_path: Path, class_name: str, prefix: str) -> dict:
    """Extract class-level constants whose name starts with prefix."""
    source = source_path.read_text()
    tree = ast.parse(source)
    constants: dict[str, object] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id.startswith(prefix)
                        ):
                            constants[target.id] = ast.literal_eval(item.value)
    return constants


def _load_helper():
    """Exec _source_audio_chunk into a callable, wiring class constants."""
    method_src = _extract_method_source(
        FIVE_Q_PATH, "GstAvtrFrameSource", "_source_audio_chunk"
    )
    constants = _extract_class_constants(
        FIVE_Q_PATH, "GstAvtrFrameSource", "HANDOFF_POP_TIMEOUT"
    )
    gst_cls = type("GstAvtrFrameSource", (), constants)
    ns: dict = {"GstAvtrFrameSource": gst_cls}
    exec(method_src, ns)
    return ns["_source_audio_chunk"]


class _FakeHandoff:
    """Minimal handoff stub: pop() returns queued (chunk, meta) or None."""

    def __init__(self, queued=None):
        self._queued = list(queued) if queued else []

    def pop(self, timeout_s: float = 0.5):
        if self._queued:
            return self._queued.pop(0)
        return None


STEP = 800  # arbitrary step_samples for tests


# ── Test 1: Real TTS chunk → real PCM + is_speech=True ─────────────────────


def test_helper_returns_real_pcm_when_handoff_has_audio():
    """When handoff has a chunk, the helper returns that PCM and is_speech=True."""
    helper = _load_helper()
    pcm = np.ones(STEP, dtype=np.float32) * 0.42
    meta = {"turnId": "turn-1", "isFirst": True, "isLast": False, "chunkIndex": 0}
    handoff = _FakeHandoff([(pcm, meta)])

    chunk_audio, is_speech, returned_meta = helper(handoff, False, STEP)

    assert is_speech is True
    np.testing.assert_array_equal(chunk_audio, pcm)
    assert returned_meta == meta


# ── Test 2a: Empty handoff → zero PCM + is_speech=False ────────────────────


def test_helper_returns_silence_when_handoff_empty():
    """When handoff.pop() returns None, the helper returns zeros and is_speech=False."""
    helper = _load_helper()
    handoff = _FakeHandoff([])

    chunk_audio, is_speech, returned_meta = helper(handoff, False, STEP)

    assert is_speech is False
    assert returned_meta is None
    assert chunk_audio.dtype == np.float32
    np.testing.assert_array_equal(chunk_audio, np.zeros(STEP, dtype=np.float32))


# ── Test 2b: None handoff → zero PCM + is_speech=False ─────────────────────


def test_helper_returns_silence_when_handoff_none():
    """When handoff is None (no handoff wired), the helper returns zeros."""
    helper = _load_helper()

    chunk_audio, is_speech, returned_meta = helper(None, False, STEP)

    assert is_speech is False
    assert returned_meta is None
    np.testing.assert_array_equal(chunk_audio, np.zeros(STEP, dtype=np.float32))


# ── Test 2c: Consecutive chunks drain correctly ────────────────────────────


def test_helper_drains_multiple_chunks():
    """Multiple queued chunks are returned in order across successive calls."""
    helper = _load_helper()
    pcm_a = np.full(STEP, 0.1, dtype=np.float32)
    pcm_b = np.full(STEP, 0.2, dtype=np.float32)
    meta_a = {"turnId": "t", "isFirst": True, "isLast": False, "chunkIndex": 0}
    meta_b = {"turnId": "t", "isFirst": False, "isLast": True, "chunkIndex": 1}
    handoff = _FakeHandoff([(pcm_a, meta_a), (pcm_b, meta_b)])

    a, speech_a, ma = helper(handoff, False, STEP)
    b, speech_b, mb = helper(handoff, True, STEP)
    c, speech_c, mc = helper(handoff, True, STEP)

    assert speech_a and speech_b
    assert not speech_c
    np.testing.assert_array_equal(a, pcm_a)
    np.testing.assert_array_equal(b, pcm_b)
    assert ma["isFirst"] is True
    assert mb["isLast"] is True


# ── Test 3: GstAvtrFrameSource.__init__ accepts handoff ───────────────────


def test_init_accepts_handoff_parameter():
    """GstAvtrFrameSource.__init__ must accept a 'handoff' argument."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GstAvtrFrameSource":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    arg_names = [a.arg for a in item.args.args]
                    assert "handoff" in arg_names, (
                        f"__init__ args are {arg_names}, missing 'handoff'"
                    )
                    return

    pytest.fail("GstAvtrFrameSource.__init__ not found")


# ── Test 4: _start_session_webrtc passes handoff=sess["handoff"] ──────────


def test_start_session_webrtc_passes_handoff():
    """The gstreamer branch must construct GstAvtrFrameSource with handoff=sess["handoff"]."""
    methods = _extract_class_methods(FIVE_Q_PATH, "AvatarLifecycleCpuSnapshotSession5Q")
    func_src = methods.get("_start_session_webrtc")
    assert func_src is not None, "_start_session_webrtc not found"

    assert "GstAvtrFrameSource(" in func_src, (
        "_start_session_webrtc does not construct GstAvtrFrameSource"
    )
    assert 'handoff=sess["handoff"]' in func_src, (
        "_start_session_webrtc does not pass handoff=sess[\"handoff\"] "
        "to GstAvtrFrameSource"
    )


# ── Test 5: _render_loop calls _source_audio_chunk ─────────────────────────


def test_render_loop_calls_source_audio_chunk():
    """_render_loop must call _source_audio_chunk so the helper is not dead code."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GstAvtrFrameSource":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_render_loop":
                    func_src = ast.get_source_segment(source, item)
                    assert func_src is not None
                    assert "_source_audio_chunk" in func_src, (
                        "_render_loop does not call _source_audio_chunk"
                    )
                    return

    pytest.fail("GstAvtrFrameSource._render_loop not found")


# ── Test 6: _GstSourceMetrics has handoff counters ────────────────────────


def test_gst_source_metrics_has_handoff_counters():
    """_GstSourceMetrics must include the 6 new handoff/speech counters."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_GstSourceMetrics":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "snapshot":
                    func_src = ast.get_source_segment(source, item)
                    assert func_src is not None
                    for key in (
                        "handoff_chunks_consumed",
                        "speech_chunks_rendered",
                        "idle_chunks_rendered",
                        "speech_turns_started",
                        "speech_turns_completed",
                        "last_turn_id",
                    ):
                        assert key in func_src, (
                            f"_GstSourceMetrics.snapshot() missing key '{key}'"
                        )
                    return

    pytest.fail("_GstSourceMetrics.snapshot() not found")


# ── Shared helper for class method extraction ─────────────────────────────


def _extract_class_methods(source_path: Path, class_name: str) -> dict[str, str]:
    """Extract all method source from a class via AST. Returns {name: source}."""
    source = source_path.read_text()
    tree = ast.parse(source)
    methods: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seg = ast.get_source_segment(source, item)
                    if seg:
                        methods[item.name] = seg
    return methods
