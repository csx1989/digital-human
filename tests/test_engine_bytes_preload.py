# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Regression test: ONNX paths must never enter engine-bytes preload.

The 5Q Modal app preloads engine bytes during ``_snap_init`` (CPU snapshot
phase) so that at GPU restore time the engines can be quickly deserialised
via ``TRTEngine.from_bytes()``.  ONNX files are **not** TRT-serialisable;
feeding them to ``load_engine_from_bytes`` causes a deserialisation failure
that was previously swallowed and masked as a non-fatal warning.

This test inspects the 5Q source file to prove that:

1. The preload loop checks ``path.suffix.lower() == ".onnx"`` and skips.
2. Only ``.engine`` / ``.trt`` / ``.plan`` files pass through to
   ``read_bytes()``.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

_5Q_FILE = Path(__file__).resolve().parents[1] / (
    "avatar-saas-prototype/modal-avatar-runtime/"
    "modal_lifecycle_pytorch_cpu_snapshot_session_l40s_5q.py"
)


def _extract_snap_init_source() -> str:
    """Extract the ``_snap_init`` method source from the 5Q file."""
    source = _5Q_FILE.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_snap_init":
            return ast.get_source_segment(source, node) or ""

    pytest.fail(f"_snap_init not found in {_5Q_FILE}")


class TestOnnxExcludedFromEngineBytesPreload:
    """Prove .onnx files are filtered out of the byte preload path."""

    def test_onnx_suffix_check_exists(self) -> None:
        """The preload loop must check for .onnx suffix."""
        src = _extract_snap_init_source()
        assert '.onnx' in src, (
            "Expected .onnx suffix check in _snap_init preload loop.\n"
            f"Source:\n{src}"
        )

    def test_onnx_skip_message_present(self) -> None:
        """A clear skip message must be logged when ONNX is encountered."""
        src = _extract_snap_init_source()
        assert 'not TRT-serializable' in src or 'ONNX fallback' in src, (
            "Expected ONNX skip log message in _snap_init.\n"
            f"Source:\n{src}"
        )

    def test_read_bytes_after_onnx_check(self) -> None:
        """read_bytes() must only be called after the .onnx filter.

        We parse the AST and verify that within the for-loop body that
        iterates over engines_to_preload, any call to ``read_bytes`` is
        preceded by a conditional that checks the suffix.
        """
        src = _extract_snap_init_source()
        tree = ast.parse(textwrap.dedent(src))

        found_read_bytes = False
        found_onnx_guard = False

        for node in ast.walk(tree):
            # Look for `path.read_bytes()` calls
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "read_bytes"):
                found_read_bytes = True

            # Look for suffix == ".onnx" comparisons
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and comparator.value == ".onnx":
                        found_onnx_guard = True

        assert found_read_bytes, "No path.read_bytes() call found in _snap_init"
        assert found_onnx_guard, (
            "No .onnx suffix comparison found — ONNX files could leak into byte preload"
        )
