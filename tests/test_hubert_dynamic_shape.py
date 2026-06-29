# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Tests for HuBERT speech-feature extraction with dynamic ONNX output shapes.

Verifies that ``run_hubert()`` pre-allocates the ``last_hidden_state``
output with the correct ``(B, N // 320, 1024)`` shape so that ONNX
engines — which cannot auto-resolve dynamic output dimensions — work
correctly, while remaining compatible with TRT engines that can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from avtr1_renderer.components.hubert import run_hubert
from avtr1_renderer.models.hubert import HubertInput, HubertOutput


# ---------------------------------------------------------------------------
# Fake engines
# ---------------------------------------------------------------------------

class _RecordingOnnxEngine:
    """Simulates OnnxRTEngine: raises if shapes not provided for dynamic output.

    Records the shapes passed to ``allocate_outputs`` and the inputs
    passed to ``__call__`` so tests can assert on them.
    """

    def __init__(self) -> None:
        self.allocated_shapes: dict[str, tuple[int, ...]] | None = None
        self.last_inputs: HubertInput | None = None
        self.last_out: HubertOutput | None = None

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> HubertOutput:
        if shapes is None:
            raise ValueError(
                "Output 'last_hidden_state' has a dynamic shape; pass it via "
                "shapes={ 'last_hidden_state': ... } or supply out=... to __call__"
            )
        self.allocated_shapes = shapes
        return HubertOutput(
            last_hidden_state=torch.empty(
                shapes["last_hidden_state"], dtype=torch.float32
            )
        )

    def __call__(
        self, inputs: HubertInput, out: HubertOutput | None = None
    ) -> HubertOutput:
        if out is None:
            out = self.allocate_outputs()
        self.last_inputs = inputs
        self.last_out = out
        # Simulate HuBERT output: fill with recognizable values
        out.last_hidden_state.fill_(0.5)
        return out


class _FakeTrtEngine:
    """Simulates TRTEngine: auto-resolves shapes, accepts out= but doesn't require it.

    Mirrors ``TRTEngine.__call__`` which resolves output shapes via
    ``ctx.get_tensor_shape`` and can work with or without pre-allocated outputs.
    """

    def __init__(self) -> None:
        self.allocate_called = False
        self.allocated_shapes: dict[str, tuple[int, ...]] | None = None

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> HubertOutput:
        self.allocate_called = True
        self.allocated_shapes = shapes
        if shapes is not None:
            return HubertOutput(
                last_hidden_state=torch.empty(
                    shapes["last_hidden_state"], dtype=torch.float32
                )
            )
        # TRT would resolve from execution context; simulate with a dummy
        raise ValueError("TRT would resolve shapes internally — not reached in this test")

    def __call__(
        self, inputs: HubertInput, out: HubertOutput | None = None
    ) -> HubertOutput:
        if out is None:
            # TRT resolves shapes internally — simulate
            B, N = inputs.input_values.shape
            n_frames = N // 320
            out = HubertOutput(
                last_hidden_state=torch.empty(B, n_frames, 1024, dtype=torch.float32)
            )
        out.last_hidden_state.fill_(0.7)
        return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunHubertOnnxDynamicShape:
    """Verify run_hubert passes explicit output shapes for ONNX engines."""

    def test_allocates_dynamic_onnx_output(self) -> None:
        """ONNX engine receives explicit (B, N//320, 1024) shape via allocate_outputs."""
        engine = _RecordingOnnxEngine()
        B, N = 2, 6480  # typical audio length
        audio = torch.zeros(B, N, dtype=torch.float32)

        result = run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert engine.allocated_shapes is not None, (
            "allocate_outputs was never called with shapes"
        )
        expected_frames = N // 320  # 20
        assert engine.allocated_shapes["last_hidden_state"] == (B, expected_frames, 1024)

    def test_onnx_engine_does_not_raise(self) -> None:
        """The ONNX path that previously raised ValueError now works."""
        engine = _RecordingOnnxEngine()
        audio = torch.zeros(2, 6480, dtype=torch.float32)

        # Before the fix this raised:
        #   ValueError: Output 'last_hidden_state' has a dynamic shape
        result = run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert result.shape == (2, 5, 1024)

    def test_correct_frame_count(self) -> None:
        """N=6480 → 6480//320 = 20 frames at 50 Hz."""
        engine = _RecordingOnnxEngine()
        audio = torch.zeros(1, 6480, dtype=torch.float32)

        run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert engine.allocated_shapes is not None
        shape = engine.allocated_shapes["last_hidden_state"]
        assert shape[1] == 20, f"Expected 20 frames, got {shape[1]}"


class TestRunHubertTrtCompatible:
    """Verify run_hubert still works with TRT-like engines."""

    def test_works_with_trt_like_engine(self) -> None:
        """TRT engine accepts pre-allocated output without issues."""
        engine = _FakeTrtEngine()
        audio = torch.zeros(2, 6480, dtype=torch.float32)

        result = run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert result.shape == (2, 5, 1024)
        assert engine.allocate_called, "allocate_outputs should be called"
        assert engine.allocated_shapes is not None, "shapes should be passed"


class TestRunHubertGuardrails:
    """Verify guardrails prevent silent bad-reshape."""

    def test_too_short_audio_raises(self) -> None:
        """Audio shorter than 2*n_motion_frames*320 must raise."""
        engine = _RecordingOnnxEngine()
        # n_motion_frames=5 → need ≥ 10 frames → ≥ 3200 samples
        # Give only 1000 samples → 3 frames → too few
        audio = torch.zeros(1, 1000, dtype=torch.float32)

        with pytest.raises(ValueError, match="Audio too short for HuBERT"):
            run_hubert(audio, n_motion_frames=5, hubert=engine)

    def test_exact_minimum_works(self) -> None:
        """Audio exactly at minimum boundary (10*320=3200) should work."""
        engine = _RecordingOnnxEngine()
        N = 10 * 320  # exactly 10 frames
        audio = torch.zeros(1, N, dtype=torch.float32)

        result = run_hubert(audio, n_motion_frames=5, hubert=engine)
        assert result.shape == (1, 5, 1024)

    def test_1d_audio_raises(self) -> None:
        """1D audio (missing batch dim) should raise."""
        engine = _RecordingOnnxEngine()
        audio = torch.zeros(6480, dtype=torch.float32)

        with pytest.raises(ValueError):
            run_hubert(audio, n_motion_frames=5, hubert=engine)


class _FixedBatchOnnxEngine:
    """Simulates OnnxRTEngine with fixed batch=1 — falls back to per-element.

    Mirrors the real OnnxRTEngine.__call__ batch-fallback: tries full
    batch, catches RuntimeError, falls back to per-element inference.
    """

    def __init__(self) -> None:
        self.call_count = 0

    def allocate_outputs(
        self, shapes: dict[str, tuple[int, ...]] | None = None
    ) -> HubertOutput:
        if shapes is None:
            raise ValueError("dynamic shape")
        return HubertOutput(
            last_hidden_state=torch.empty(
                shapes["last_hidden_state"], dtype=torch.float32
            )
        )

    def _run_single(self, inputs: HubertInput, out: HubertOutput) -> HubertOutput:
        self.call_count += 1
        out.last_hidden_state.fill_(0.5)
        return out

    def __call__(
        self, inputs: HubertInput, out: HubertOutput | None = None
    ) -> HubertOutput:
        B = inputs.input_values.shape[0]
        if B == 1:
            if out is None:
                out = self.allocate_outputs()
            return self._run_single(inputs, out)

        # Try full batch first
        try:
            self.call_count += 1
            raise RuntimeError(
                f"Got invalid dimensions for input: input_values "
                f"index: 0 Got: {B} Expected: 1"
            )
        except RuntimeError:
            # Fall back to per-element (simulates OnnxRTEngine behavior)
            if out is not None:
                for i in range(B):
                    single_in = HubertInput(
                        input_values=inputs.input_values[i : i + 1].contiguous()
                    )
                    single_out = HubertOutput(
                        last_hidden_state=out.last_hidden_state[i : i + 1]
                    )
                    self._run_single(single_in, single_out)
                return out
            else:
                results = []
                for i in range(B):
                    single_in = HubertInput(
                        input_values=inputs.input_values[i : i + 1].contiguous()
                    )
                    results.append(self._run_single(single_in, self.allocate_outputs(
                        shapes={"last_hidden_state": (1, single_in.input_values.shape[1] // 320, 1024)}
                    )))
                return HubertOutput(
                    last_hidden_state=torch.cat(
                        [r.last_hidden_state for r in results], dim=0
                    )
                )


class TestRunHubertBatchFallback:
    """Verify run_hubert falls back to per-element when batch > 1 fails."""

    def test_batch2_falls_back_to_single(self) -> None:
        """ONNX engine with batch=1: batch=2 triggers per-element fallback."""
        engine = _FixedBatchOnnxEngine()
        audio = torch.zeros(2, 6480, dtype=torch.float32)

        result = run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert result.shape == (2, 5, 1024)
        # 1 failed batch attempt (caught internally by engine) + 2 single calls
        assert engine.call_count == 3

    def test_batch1_no_fallback(self) -> None:
        """Batch=1 with fixed-batch engine: no fallback needed."""
        engine = _FixedBatchOnnxEngine()
        audio = torch.zeros(1, 6480, dtype=torch.float32)

        result = run_hubert(audio, n_motion_frames=5, hubert=engine)

        assert result.shape == (1, 5, 1024)
        assert engine.call_count == 1
