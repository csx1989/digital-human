# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Streaming face animation orchestrator.

Thin coordinator wiring three stages: a ``MotionGenerator`` produces
per-frame motion from audio, ``render_chunk_streaming`` yields per-frame
GPU ``(rgb, alpha)`` slices (warp runs once on the full chunk, every
downstream step runs per frame), and ``pack_frames`` converts each
slice to the requested pixel format and copies to host.

    motions, next_state = motion_generator.generate_chunk(chunk, avatar, state)
    for rgb_1, alpha_1 in render_chunk_streaming(motions, avatar, bg, ...):
        yield pack_frames(rgb_1, alpha_1, pixel_format=...)[0]

``Pipeline.process_chunk`` returns ``(state, frame_iterator)``; with
``stream_frames=True`` (default) the iterator is truly per-frame —
each ``next()`` runs decode + putback + matting + pack + H2D for
exactly one frame and yields it before the next frame starts.

TRT vs ONNX
-----------
Build TRT engines once with the ``scripts/build_*.py`` helpers; they're
written under the artifact storage root (``<project_root>/artifacts/{HF_REVISION}/``
by default, or ``$AVTR1_LOCAL_STORAGE/{HF_REVISION}/`` when set).
``Pipeline.from_artifacts`` checks for TRT engines first and falls back to
ONNX Runtime automatically for every model except the AVTR1 decoder, which
requires TRT (run ``scripts/build_avtr1_engines.py`` before first use).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from avtr1_renderer.avtr1_artifact_manager import (
    find_engine_or_onnx,
    get_artifact_manager,
    get_trt_engine_path, get_storage_root,
)
from avtr1_renderer.avatar_loader import Avatar
from avtr1_renderer.backgrounds import load_background
from avtr1_renderer.frame_sink import pack_frames
from avtr1_renderer.avtr1_motion_generator import (
    AVTR1MotionGenerator,
    Normalizer,
)
from avtr1_renderer.models.decoder import DecoderEngine, DecoderInput, DecoderOutput
from avtr1_renderer.models.hubert import HubertInput, HubertOutput
from avtr1_renderer.models.avtr1 import (
    Avtr1DecodeInput,
    Avtr1DecodeOutput,
    Avtr1EncodeInput,
    Avtr1EncodeOutput,
)
from avtr1_renderer.models.matting import MODNetEngine, MODNetInput, MODNetOutput
from avtr1_renderer.models.stitch import StitchEngine, StitchInput, StitchOutput
from avtr1_renderer.models.warp import WarpEngine, WarpInput, WarpOutput
from avtr1_renderer.motion_generator import MotionGenerator
from avtr1_renderer.renderer import render_chunk, render_chunk_streaming
from avtr1_renderer.runtime import load_engine
from avtr1_renderer.runtime.inference_engine import InferenceEngine

from avtr1_renderer.types import Chunk, FrameIterator, RenderOptions

# Reserved sentinel: skip bg compositing so callers receive raw foreground + matte.
# Pair with ``yuv_i420_stacked_alpha`` for clean transparency.
TRANSPARENT_BG_ID = "transparent"


class Pipeline[StateT]:
    """Holds the motion generator + renderer engines + bg registry."""

    _engine_backends: dict[str, str]

    def __init__(
        self,
        *,
        motion_generator: MotionGenerator[StateT],
        stitch: StitchEngine,
        warp: WarpEngine,
        decoder: DecoderEngine,
        matting: MODNetEngine,
        backgrounds: dict[str, torch.Tensor],
        engine_backends: dict[str, str] | None = None,
    ) -> None:
        self._motion_generator = motion_generator
        self._stitch = stitch
        self._warp = warp
        self._decoder = decoder
        self._matting = matting
        if not backgrounds:
            raise ValueError("backgrounds registry is empty")
        self._backgrounds = backgrounds
        self._engine_backends = engine_backends or {}

    @classmethod
    def from_artifacts(
        cls,
        *,
        avatar_ids: list[str] | None = None,
        portraits_dir: Path | str | None = None,
        background_paths: dict[str, Path | str] | None = None,
        background_filter: list[str] | None = None,
        out_size: tuple[int, int] = (720, 1280),
        download_workers: int = 4,
        prebuilt_avatars: dict[str, Avatar] | None = None,
        timing_callback: Callable[[str, float], None] | None = None,
        parallel_engine_loading: bool = False,
        prebuilt_engines: dict[str, InferenceEngine] | None = None,
    ) -> tuple[Pipeline, dict[str, Avatar]]:
        """Build the Pipeline + avatar registry from downloaded artifacts.

        Downloads any missing artifacts from HuggingFace, then wires the full
        speech-to-motion + renderer stack.

        TRT engines built by ``scripts/build_*.py`` are used automatically when
        present in ``get_local_engine_dir()``.  For every renderer model the
        pipeline falls back to ONNX Runtime if the TRT engine is absent.

        AVTR1 (speech decoder) requires TRT engines — run
        ``scripts/build_avtr1_engines.py`` before calling this method.

        Args:
            avatar_ids:       Portrait IDs to pre-load (stem of ``{id}.png`` in
                              ``portraits_dir``). Pass ``None`` to skip.
            portraits_dir:    Directory of ``{avatar_id}.png`` files. Defaults to
                              the ``reference_frames`` artifact from HuggingFace.
            background_paths: ``{bg_id: png_path}`` mapping. When ``None`` all
                              PNGs from the ``backgrounds`` artifact are used.
            background_filter: When set, only load backgrounds whose IDs are in
                              this list (plus the transparent sentinel).
            out_size:         ``(H, W)`` the pipeline operates at.
            download_workers: Parallel HuggingFace download threads per group.
            prebuilt_avatars: Pre-built ``{avatar_id: Avatar}`` dict. When
                              provided, skip AvatarLoader construction entirely
                              and use these avatars as the registry.
            timing_callback:  Called as ``callback(step_name, elapsed_s)`` after
                              each major step for timing instrumentation.
            parallel_engine_loading:  When ``True``, load ``hubert_lbs``,
                              ``decoder``, ``stitch_network``, and ``modnet``
                              engines concurrently via a thread pool.
            prebuilt_engines: Pre-built ``{name: InferenceEngine}`` dict.  When
                              provided, skip engine loading for those names and
                              use the supplied engines directly.
        """
        import time

        def _tc(name: str, start: float) -> None:
            if timing_callback is not None:
                timing_callback(name, time.monotonic() - start)

        engine_backends: dict[str, str] = {}

        def _backend(path: Path) -> str:
            return "TRT" if path.suffix.lower() in {".engine", ".trt", ".plan"} else "ONNX"

        if prebuilt_engines is None:
            prebuilt_engines = {}

        out_h, out_w = out_size

        # --- Download missing artifacts from HuggingFace ---------------------
        t0 = time.monotonic()
        mgr = get_artifact_manager()
        mgr.ensure_all_artifacts(workers=download_workers)
        _tc("artifacts_ensure", t0)

        # --- Backgrounds -----------------------------------------------------
        t0 = time.monotonic()
        if background_paths is None:
            bg_dir = mgr.get_artifact_path("backgrounds")
            background_paths = {p.stem: p for p in sorted(bg_dir.glob("*.png"))}
        if background_filter is not None:
            bg_set = set(background_filter) | {TRANSPARENT_BG_ID}
            background_paths = {
                k: v for k, v in background_paths.items() if k in bg_set
            }
        backgrounds: dict[str, torch.Tensor] = {
            bg_id: load_background(path, out_h, out_w)
            for bg_id, path in background_paths.items()
        }
        backgrounds[TRANSPARENT_BG_ID] = torch.zeros(
            (1, 3, out_h, out_w), dtype=torch.float32, device="cuda"
        )
        _tc("backgrounds_load", t0)

        # --- AVTR1 TRT engines (required) ------------------------------------
        t0 = time.monotonic()
        encode_path = get_trt_engine_path("avtr1_encode")
        decode_path = get_trt_engine_path("avtr1_decode")
        if not encode_path.is_file() or not decode_path.is_file():
            raise RuntimeError(
                f"AVTR1 TRT engines not found.\n"
                f"Run: python scripts/build_avtr1_engines.py\n"
                f"Expected files:\n"
                f"  {encode_path}\n"
                f"  {decode_path}"
            )
        if "avtr1_encode" in prebuilt_engines:
            encode = prebuilt_engines["avtr1_encode"]
            engine_backends["avtr1_encode"] = "TRT"
            if timing_callback is not None:
                timing_callback("avtr1_encode_trt", 0.0)
        else:
            encode = load_engine(encode_path, Avtr1EncodeInput, Avtr1EncodeOutput)
            engine_backends["avtr1_encode"] = _backend(encode_path)
        _tc("avtr1_encode_trt", t0)

        t0 = time.monotonic()
        if "avtr1_decode" in prebuilt_engines:
            decode = prebuilt_engines["avtr1_decode"]
            engine_backends["avtr1_decode"] = "TRT"
            if timing_callback is not None:
                timing_callback("avtr1_decode_trt", 0.0)
        else:
            decode = load_engine(decode_path, Avtr1DecodeInput, Avtr1DecodeOutput)
            engine_backends["avtr1_decode"] = _backend(decode_path)
        _tc("avtr1_decode_trt", t0)

        # AVTR1 normaliser stats — downloaded from HF (CC-independent)
        t0 = time.monotonic()
        normalizer = Normalizer.from_safetensors(
            get_storage_root() / "avtr1_normalizer.safetensors"
        )
        _tc("normalizer", t0)

        # --- Renderer engines: TRT if built, else ONNX ----------------------
        if parallel_engine_loading:
            t0 = time.monotonic()
            with ThreadPoolExecutor(max_workers=4) as executor:
                def _load_hubert():
                    if "hubert" in prebuilt_engines:
                        return prebuilt_engines["hubert"], "hubert_lbs", "TRT"
                    path = find_engine_or_onnx("hubert_lbs", "hubert_onnx")
                    return load_engine(path, HubertInput, HubertOutput), "hubert_lbs", _backend(path)

                def _load_decoder():
                    if "decoder" in prebuilt_engines:
                        return prebuilt_engines["decoder"], "decoder", "TRT"
                    path = find_engine_or_onnx("decoder", "decoder_onnx")
                    return load_engine(path, DecoderInput, DecoderOutput), "decoder", _backend(path)

                def _load_stitch():
                    if "stitch" in prebuilt_engines:
                        return prebuilt_engines["stitch"], "stitch_network", "TRT"
                    path = find_engine_or_onnx("stitch_network", "stitch_network_onnx")
                    return load_engine(path, StitchInput, StitchOutput), "stitch_network", _backend(path)

                def _load_modnet():
                    if "modnet" in prebuilt_engines:
                        return prebuilt_engines["modnet"], "modnet", "TRT"
                    path = find_engine_or_onnx("modnet", "modnet_onnx")
                    return load_engine(path, MODNetInput, MODNetOutput), "modnet", _backend(path)

                futures = {
                    executor.submit(_load_hubert): "hubert",
                    executor.submit(_load_decoder): "decoder",
                    executor.submit(_load_stitch): "stitch",
                    executor.submit(_load_modnet): "modnet",
                }
                results = {}
                for future, name in futures.items():
                    engine, backend_name, backend = future.result()
                    results[name] = engine
                    engine_backends[backend_name] = backend

            hubert = results["hubert"]
            decoder = results["decoder"]
            stitch = results["stitch"]
            matting = results["modnet"]
            _tc("parallel_engine_load", t0)
        else:
            t0 = time.monotonic()
            if "hubert" in prebuilt_engines:
                hubert = prebuilt_engines["hubert"]
                engine_backends["hubert_lbs"] = "TRT"
                if timing_callback is not None:
                    timing_callback("hubert_engine", 0.0)
            else:
                hubert_path = find_engine_or_onnx("hubert_lbs", "hubert_onnx")
                hubert = load_engine(hubert_path, HubertInput, HubertOutput)
                engine_backends["hubert_lbs"] = _backend(hubert_path)
                _tc("hubert_engine", t0)

            t0 = time.monotonic()
            if "decoder" in prebuilt_engines:
                decoder = prebuilt_engines["decoder"]
                engine_backends["decoder"] = "TRT"
                if timing_callback is not None:
                    timing_callback("decoder_engine", 0.0)
            else:
                decoder_path = find_engine_or_onnx("decoder", "decoder_onnx")
                decoder = load_engine(decoder_path, DecoderInput, DecoderOutput)
                engine_backends["decoder"] = _backend(decoder_path)
                _tc("decoder_engine", t0)

            t0 = time.monotonic()
            if "stitch" in prebuilt_engines:
                stitch = prebuilt_engines["stitch"]
                engine_backends["stitch_network"] = "TRT"
                if timing_callback is not None:
                    timing_callback("stitch_engine", 0.0)
            else:
                stitch_path = find_engine_or_onnx("stitch_network", "stitch_network_onnx")
                stitch = load_engine(stitch_path, StitchInput, StitchOutput)
                engine_backends["stitch_network"] = _backend(stitch_path)
            _tc("stitch_engine", t0)

            t0 = time.monotonic()
            if "modnet" in prebuilt_engines:
                matting = prebuilt_engines["modnet"]
                engine_backends["modnet"] = "TRT"
                if timing_callback is not None:
                    timing_callback("modnet_engine", 0.0)
            else:
                modnet_path = find_engine_or_onnx("modnet", "modnet_onnx")
                matting = load_engine(modnet_path, MODNetInput, MODNetOutput)
                engine_backends["modnet"] = _backend(modnet_path)
            _tc("modnet_engine", t0)

        # Warp needs the grid-sample plugin when running in TRT mode.
        # Always sequential — plugin loading is not thread-safe.
        t0 = time.monotonic()
        if "warp" in prebuilt_engines:
            warp = prebuilt_engines["warp"]
            engine_backends["warp_network"] = "TRT"
        else:
            warp_trt = get_trt_engine_path("warp_network")
            if warp_trt.is_file():
                plugin_path = mgr.storage_path("warp_plugin")
                warp = load_engine(
                    warp_trt, WarpInput, WarpOutput,
                    plugin_files=[str(plugin_path)] if plugin_path.is_file() else [],
                )
                engine_backends["warp_network"] = _backend(warp_trt)
            else:
                warp_path = find_engine_or_onnx("warp_network", "warp_network_onnx")
                warp = load_engine(warp_path, WarpInput, WarpOutput)
                engine_backends["warp_network"] = _backend(warp_path)
        _tc("warp_engine", t0)

        # --- Avatar registry -------------------------------------------------
        if prebuilt_avatars is not None:
            registry = dict(prebuilt_avatars)
            if timing_callback is not None:
                timing_callback("avatar_loader", 0.0)
                timing_callback("avatar_registration", 0.0)
        else:
            from avtr1_renderer.avatar_loader import AvatarLoader

            t0 = time.monotonic()
            mask_path = (
                mgr.get_artifact_path("pasteback_mask")
                if "pasteback_mask" in mgr._artifacts
                else None
            )
            loader = AvatarLoader(
                engine_files={
                    "insightface_det": mgr.get_artifact_path("insightface_det"),
                    "landmark106": mgr.get_artifact_path("landmark106"),
                    "landmark203": mgr.get_artifact_path("landmark203"),
                    "appearance_extractor": mgr.get_artifact_path("appearance_extractor"),
                    "motion_extractor": mgr.get_artifact_path("motion_extractor"),
                },
                mask_template_path=mask_path,
                out_h=out_h,
                out_w=out_w,
                max_dim=max(out_h, out_w),
            )
            _tc("avatar_loader", t0)

            t0 = time.monotonic()
            if portraits_dir is None:
                portraits_dir = mgr.get_artifact_path("reference_frames")
            portraits_dir = Path(portraits_dir)

            registry: dict[str, Avatar] = {}
            for avatar_id in (avatar_ids or []):
                portrait = portraits_dir / f"{avatar_id}.png"
                if not portrait.is_file():
                    raise FileNotFoundError(f"No portrait at {portrait}")
                registry[avatar_id] = loader.load(portrait, avatar_id=avatar_id)
            _tc("avatar_registration", t0)

        # --- Motion generator ------------------------------------------------
        t0 = time.monotonic()
        motion_generator = AVTR1MotionGenerator(
            hubert=hubert,
            encode_engine=encode,
            decode_engine=decode,
            normalizer=normalizer,
        )
        _tc("motion_generator", t0)

        return (
            cls(
                motion_generator=motion_generator,
                stitch=stitch,
                warp=warp,
                decoder=decoder,
                matting=matting,
                backgrounds=backgrounds,
                engine_backends=engine_backends,
            ),
            registry,
        )

    def initial_state(self, avatar: Avatar) -> StateT:
        return self._motion_generator.initial_state(avatar)

    def process_chunk(
        self,
        avatar: Avatar,
        chunk: Chunk,
        state: StateT | None,
        options: RenderOptions | None = None,
    ) -> tuple[StateT, FrameIterator]:
        """Run one streaming chunk end-to-end.

        Pass ``state=None`` for the cold-start call — the pipeline builds
        the initial state for ``avatar`` itself.
        """
        if options is None:
            options = RenderOptions()
        if state is None:
            state = self._motion_generator.initial_state(avatar)
        mg = self._motion_generator
        expected_len = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        got_len = len(chunk.audio_speech)
        if got_len != expected_len:
            raise ValueError(
                f"Chunk audio length mismatch: expected {expected_len} samples "
                f"({expected_len / 16000 * 1000:.0f} ms at 16 kHz), got {got_len}. "
                f"Use Pipeline helpers (_chunk_window / _chunk_step) or the "
                f"slice_chunks utility in scripts/generate_offline.py."
            )
        if options.bg_id is None:
            raise ValueError("RenderOptions.bg_id is required (no implicit default)")
        try:
            bg = self._backgrounds[options.bg_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown bg_id {options.bg_id!r}; registered: {sorted(self._backgrounds)}"
            ) from exc
        motions, next_state = self._motion_generator.generate_chunk(
            chunk, avatar, state, options
        )

        def frames_streaming() -> FrameIterator:
            stream = render_chunk_streaming(
                motions, avatar, bg,
                stitch=self._stitch,
                warp=self._warp,
                decoder=self._decoder,
                matting=self._matting,
            )
            for rgb, alpha in stream:
                packed = pack_frames(rgb, alpha, pixel_format=options.pixel_format)
                yield packed[0]

        def frames_batched() -> FrameIterator:
            rgb, alpha = render_chunk(
                motions, avatar, bg,
                stitch=self._stitch,
                warp=self._warp,
                decoder=self._decoder,
                matting=self._matting,
            )
            yield from pack_frames(rgb, alpha, pixel_format=options.pixel_format)

        frames = frames_streaming if options.stream_frames else frames_batched
        return next_state, frames()


__all__ = ["Pipeline", "TRANSPARENT_BG_ID"]
