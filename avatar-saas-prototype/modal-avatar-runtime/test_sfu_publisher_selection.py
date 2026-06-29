"""Tests for Phase 3A SFU publisher selection integration.

These tests verify the AVTR_SFU_PUBLISHER flag behavior and structural
constraints WITHOUT importing the 5Q file (which requires modal/torch/GPU).
Instead they use AST extraction and source analysis.

Run:  python3 -m pytest avatar-saas-prototype/modal-avatar-runtime/test_sfu_publisher_selection.py -v
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

MODULE_DIR = Path(__file__).parent
FIVE_Q_PATH = MODULE_DIR / "modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py"
GST_MODULE_PATH = MODULE_DIR / "gst_webrtc_publisher.py"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_func_source(source_path: Path, func_name: str) -> str | None:
    """Extract a function's source text from a Python file via AST."""
    source = source_path.read_text()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(source, node)
    return None


def _extract_class_methods(source_path: Path, class_name: str) -> dict[str, str]:
    """Extract all method source from a class via AST. Returns {name: source}."""
    source = source_path.read_text()
    tree = ast.parse(source)
    methods: dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    methods[item.name] = ast.get_source_segment(source, item)
    return methods


# ── Test 1: _resolve_publisher_type — default/unset → aiortc ─────────────


def test_resolve_publisher_type_unset():
    """When AVTR_SFU_PUBLISHER is unset, the default is 'aiortc'."""
    func_src = _extract_func_source(FIVE_Q_PATH, "_resolve_publisher_type")
    assert func_src is not None, "_resolve_publisher_type not found in 5Q file"

    ns = {"_os": os}
    exec(func_src, ns)
    resolve = ns["_resolve_publisher_type"]

    os.environ.pop("AVTR_SFU_PUBLISHER", None)
    assert resolve() == "aiortc"


# ── Test 2: _resolve_publisher_type — gstreamer → gstreamer ──────────────


def test_resolve_publisher_type_gstreamer():
    """When AVTR_SFU_PUBLISHER=gstreamer, the result is 'gstreamer'."""
    func_src = _extract_func_source(FIVE_Q_PATH, "_resolve_publisher_type")
    ns = {"_os": os}
    exec(func_src, ns)
    resolve = ns["_resolve_publisher_type"]

    os.environ["AVTR_SFU_PUBLISHER"] = "gstreamer"
    assert resolve() == "gstreamer"


# ── Test 3: _resolve_publisher_type — invalid → aiortc fallback ──────────


def test_resolve_publisher_type_invalid_falls_back():
    """Invalid/empty/typo values all fall back to 'aiortc'."""
    func_src = _extract_func_source(FIVE_Q_PATH, "_resolve_publisher_type")
    ns = {"_os": os}
    exec(func_src, ns)
    resolve = ns["_resolve_publisher_type"]

    for val in ("", "aiortc", "invalid", "gst", "GStreamer", "none", "false"):
        os.environ["AVTR_SFU_PUBLISHER"] = val
        assert resolve() == "aiortc", f"AVTR_SFU_PUBLISHER='{val}' should give 'aiortc'"


# ── Test 4: No top-level gst_webrtc_publisher import ──────────────────────


def test_no_top_level_gst_import_in_5q():
    """5Q file must NOT import gst_webrtc_publisher at module level.

    The import must be lazy (inside _start_session_webrtc only) so the
    fMP4-only path never touches GStreamer.
    """
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "gst_webrtc_publisher" not in alias.name, (
                    f"Top-level import of {alias.name} — must be lazy"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module and "gst_webrtc_publisher" in node.module:
                pytest.fail(
                    f"Top-level 'from {node.module} import' — must be inside a function"
                )


# ── Test 5: No top-level gi import in 5Q or gst module ───────────────────


def test_no_top_level_gi_import_5q():
    """5Q file must not import gi at module level."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "gi", "Top-level 'import gi' in 5Q file"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("gi"):
                pytest.fail(f"Top-level 'from {node.module} import' in 5Q file")


def test_no_top_level_gi_import_gst_module():
    """gst_webrtc_publisher.py must not import gi at module level."""
    source = GST_MODULE_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name != "gi", "Top-level 'import gi' in gst module"
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.startswith("gi"):
                pytest.fail(f"Top-level 'from {node.module} import' in gst module")


# ── Test 6: gst_webrtc_publisher only referenced in _start_session_webrtc ──


def test_gst_import_only_in_start_session_webrtc():
    """gst_webrtc_publisher must only appear inside _start_session_webrtc.

    No other method (especially the fMP4 path) should reference it.
    """
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_src = ast.get_source_segment(source, node)
            if func_src and "gst_webrtc_publisher" in func_src:
                assert node.name == "_start_session_webrtc", (
                    f"gst_webrtc_publisher referenced in {node.name}, "
                    f"should only be in _start_session_webrtc"
                )


# ── Test 7: fMP4 path does not touch gst_webrtc_publisher ────────────────


def test_fmp4_path_no_gst_reference():
    """The fMP4 startup path must not reference gst_webrtc_publisher."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    # Find _start_session_impl (the fMP4 session startup method)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "_start_session_impl":
                func_src = ast.get_source_segment(source, node)
                assert func_src is not None
                assert "gst_webrtc_publisher" not in func_src, (
                    "_start_session_impl references gst_webrtc_publisher"
                )
                assert "GstAvtrFrameSource" not in func_src, (
                    "_start_session_impl references GstAvtrFrameSource"
                )
                return

    pytest.fail("_start_session_impl not found in 5Q file")


def test_http_start_session_fmp4_branch_no_gst():
    """The http_start_session fMP4 branch must not import gst_webrtc_publisher."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "http_start_session":
            func_src = ast.get_source_segment(source, node)
            assert func_src is not None
            assert "gst_webrtc_publisher" not in func_src, (
                "http_start_session references gst_webrtc_publisher"
            )
            return

    pytest.fail("http_start_session not found")


# ── Test 8: _start_session_webrtc branches on _resolve_publisher_type ─────


def test_start_session_webrtc_uses_resolve_publisher_type():
    """_start_session_webrtc must call _resolve_publisher_type() and branch."""
    methods = _extract_class_methods(FIVE_Q_PATH, "AvatarLifecycleCpuSnapshotSession5Q")
    func_src = methods.get("_start_session_webrtc")

    assert func_src is not None, "_start_session_webrtc not found"
    assert "_resolve_publisher_type" in func_src, (
        "_start_session_webrtc does not call _resolve_publisher_type"
    )
    assert "gstreamer" in func_src, "No gstreamer branch in _start_session_webrtc"
    assert "aiortc" in func_src or "webrtc_publisher" in func_src, (
        "No aiortc fallback branch in _start_session_webrtc"
    )


# ── Test 9: _verify_publisher_healthy handles both publisher types ────────


def test_verify_publisher_healthy_handles_both_types():
    """_verify_publisher_healthy must check both _pc (aiortc) and _ice_connected (gst)."""
    methods = _extract_class_methods(FIVE_Q_PATH, "AvatarLifecycleCpuSnapshotSession5Q")
    func_src = methods.get("_verify_publisher_healthy")
    assert func_src is not None, "_verify_publisher_healthy not found"

    assert "_pc" in func_src, "Health check does not reference publisher._pc (aiortc)"
    assert "_ice_connected" in func_src, (
        "Health check does not reference publisher._ice_connected (GStreamer)"
    )
    assert "getattr" in func_src, (
        "Health check must use getattr for publisher-type-agnostic ICE check"
    )


# ── Test 10: _end_session_impl reads metrics keys from both publishers ────


def test_end_session_reads_producer_error_and_fail_fast():
    """_end_session_impl must read producer_error and fail_fast_reason from
    publisher stop metrics — keys returned by BOTH aiortc and GStreamer publishers."""
    methods = _extract_class_methods(FIVE_Q_PATH, "AvatarLifecycleCpuSnapshotSession5Q")
    func_src = methods.get("_end_session_impl")
    assert func_src is not None, "_end_session_impl not found"

    assert "producer_error" in func_src, (
        "_end_session_impl does not read producer_error from stop metrics"
    )
    assert "fail_fast_reason" in func_src, (
        "_end_session_impl does not read fail_fast_reason from stop metrics"
    )
    assert "publisher.stop()" in func_src, (
        "_end_session_impl does not call publisher.stop()"
    )


# ── Test 11: GstAvtrFrameSource defined at module level ──────────────────


def test_gst_avtr_frame_source_exists():
    """GstAvtrFrameSource class must be defined at module level in 5Q file."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    found = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GstAvtrFrameSource":
            found = True
            break

    assert found, "GstAvtrFrameSource class not found at module level in 5Q file"


def test_gst_avtr_frame_source_has_required_interface():
    """GstAvtrFrameSource must have start(), stop(), next_frame(), get_error()."""
    source = FIVE_Q_PATH.read_text()
    tree = ast.parse(source)

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GstAvtrFrameSource":
            method_names = {
                item.name for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            required = {"start", "stop", "next_frame", "get_error", "_render_loop"}
            missing = required - method_names
            assert len(missing) == 0, (
                f"GstAvtrFrameSource missing methods: {missing}"
            )
            return

    pytest.fail("GstAvtrFrameSource not found")


# ── Test 12: Static grep validation ─────────────────────────────────────


def test_no_forbidden_strings_in_gst_module():
    """gst_webrtc_publisher.py must not contain SFU-direct strings."""
    source = GST_MODULE_PATH.read_text()
    assert "rtc.live.cloudflare.com" not in source
    assert "CALLS_APP_SECRET" not in source


def test_no_top_level_gi_regex_5q():
    """Regex check: no top-level 'import gi' or 'from gi' in 5Q file."""
    lines = FIVE_Q_PATH.read_text().splitlines()
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith("import gi") or stripped.startswith("from gi"):
            indent = len(line) - len(stripped)
            assert indent > 0, f"Line {i}: top-level '{stripped}'"
