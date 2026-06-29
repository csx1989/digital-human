"""WebRTC publisher for Modal — Phase D1 production integration.

Replaces FFmpeg→fMP4→SSE output path with aiortc→SFU→browser WebRTC path.

Key differences from Phase C POC:
- NO CALLS_APP_SECRET — SDP exchange is brokered through Worker/DO
- Modal sends offer SDP to Worker's /internal/sessions/:id/publisher-offer
- Worker proxies to SFU tracks/new and returns the answer SDP
- Non-trickle ICE: waits for ALL candidates before sending offer
- MODAL_WORKER_SECRET read from Modal environment (never from request body)
- Integrates with existing _AudioHandoff for producer thread audio source
- Frame source adapted to use the existing session's pipeline + avatar

This module is imported by the 5Q lifecycle class when publisher_mode="webrtc_sfu".
"""

from __future__ import annotations

import asyncio
import collections
import fractions
import gc
import json
import os
import threading
import time

import httpx
import numpy as np
from av import VideoFrame
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc import RTCConfiguration, RTCIceServer
from aiortc.mediastreams import MediaStreamError, VIDEO_CLOCK_RATE

# ── Media timing debug gate ──────────────────────────────────────────────────
# When AVTR_MEDIA_TIMING_DEBUG=1, low-overhead ring-buffered timing counters
# are collected and surfaced in PublisherMetrics.snapshot()["timing"].
# When disabled (default), zero timing overhead — no time.monotonic() calls
# beyond the ones that already exist, no ring-buffer writes.
TIMING_DEBUG = os.environ.get("AVTR_MEDIA_TIMING_DEBUG", "0") == "1"


class _RingStats:
    """Fixed-size ring buffer with p50/p95/max/mean stats.

    Thread-safe: collections.deque.append is atomic under the GIL, which
    is all we need (producer thread writes, snapshot reads at stop time).
    """

    __slots__ = ("_buf",)

    def __init__(self, maxlen: int = 256):
        self._buf: collections.deque[float] = collections.deque(maxlen=maxlen)

    def add(self, ms: float) -> None:
        self._buf.append(ms)

    def stats(self) -> dict:
        if not self._buf:
            return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "mean_ms": 0.0}
        s = sorted(self._buf)
        n = len(s)
        return {
            "count": n,
            "p50_ms": round(s[n // 2], 3),
            "p95_ms": round(s[min(n - 1, int(n * 0.95))], 3),
            "max_ms": round(s[-1], 3),
            "mean_ms": round(sum(s) / n, 3),
        }


def _list_stats(samples: list[float]) -> dict:
    """Stats for a plain list of scalar samples (CPU%, GPU mem)."""
    if not samples:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    s = sorted(samples)
    n = len(s)
    return {
        "count": n,
        "p50": round(s[n // 2], 3),
        "p95": round(s[min(n - 1, int(n * 0.95))], 3),
        "max": round(s[-1], 3),
        "mean": round(sum(s) / n, 3),
    }


# ── Constants ────────────────────────────────────────────────────────────────

FPS_25 = 25
PTIME_25 = fractions.Fraction(1, FPS_25)
TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
TICKS_PER_FRAME_25 = int(PTIME_25 * VIDEO_CLOCK_RATE)  # 3600

ICE_GATHER_TIMEOUT_S = 15
ICE_CONNECT_TIMEOUT_S = 30
STUN_SERVER = "stun:stun.cloudflare.com:3478"

# ── Fail-fast policy for sustained frame repeats ────────────────────────────
# After the producer thread dies (e.g. CUDA illegal memory access), the WebRTC
# track used to repeat the last good frame *forever* — producing 25fps of
# frozen video that the browser decoded as "healthy".  These bounds make the
# track fail closed instead, surfacing the error to aiortc/Modal/Worker.
#
#   MAX_REPEATS_AFTER_ERROR: once `_error` is set, allow at most this many
#       repeated frames before raising MediaStreamError (closes the track).
#       Set generously so the last good frame still drains to the SFU.
#
#   MAX_CONSECUTIVE_UNDERFLOWS: even without an explicit `_error`, this many
#       consecutive queue underflows (= `MAX_CONSECUTIVE_UNDERFLOWS / 25`
#       seconds of frozen video) is treated as a silent producer stall and
#       also raises MediaStreamError.
MAX_REPEATS_AFTER_ERROR = 25          # 1 second of grace at 25fps
MAX_CONSECUTIVE_UNDERFLOWS = 250      # 10 seconds of sustained stall


# ── Metrics ──────────────────────────────────────────────────────────────────

class PublisherMetrics:
    """Counters + optional timing for the WebRTC frame pipeline.

    Timing fields (ring buffers) are only populated when
    AVTR_MEDIA_TIMING_DEBUG=1.  When disabled, snapshot() omits the
    "timing" key entirely and the ring buffers stay empty.
    """

    # Counters (always active — incrementing an int is ~50ns)
    frames_generated: int = 0
    frames_emitted: int = 0
    frames_repeated: int = 0
    underflow_count: int = 0
    queue_depth_min: int = 10**9
    queue_depth_max: int = 0
    # Fail-fast diagnostics (populated by get_frame when the policy trips).
    consecutive_underflows: int = 0
    repeats_after_error: int = 0
    fail_fast_reason: str | None = None

    def __init__(self):
        # Timing ring buffers (only written when TIMING_DEBUG=True)
        self.process_chunk_ms = _RingStats()
        self.frames_per_chunk = _RingStats()
        self.gpu_to_cpu_copy_ms = _RingStats()
        self.queue_put_ms = _RingStats()
        self.from_ndarray_ms = _RingStats()
        self.recv_pacing_ms = _RingStats()
        self.queue_full_retries: int = 0
        self.cpu_percent_samples: list[float] = []
        self.gpu_mem_mb_samples: list[float] = []

    def snapshot(self) -> dict:
        base = {
            "frames_generated": self.frames_generated,
            "frames_emitted": self.frames_emitted,
            "frames_repeated": self.frames_repeated,
            "underflow_count": self.underflow_count,
            "queue_depth_min": self.queue_depth_min if self.queue_depth_min < 10**9 else 0,
            "queue_depth_max": self.queue_depth_max,
            "consecutive_underflows": self.consecutive_underflows,
            "repeats_after_error": self.repeats_after_error,
            "fail_fast_reason": self.fail_fast_reason,
        }
        if TIMING_DEBUG:
            base["timing"] = {
                "process_chunk_ms": self.process_chunk_ms.stats(),
                "frames_per_chunk": self.frames_per_chunk.stats(),
                "gpu_to_cpu_copy_ms": self.gpu_to_cpu_copy_ms.stats(),
                "queue_put_ms": self.queue_put_ms.stats(),
                "from_ndarray_ms": self.from_ndarray_ms.stats(),
                "recv_pacing_ms": self.recv_pacing_ms.stats(),
                "queue_full_retries": self.queue_full_retries,
                "cpu_percent": _list_stats(self.cpu_percent_samples),
                "gpu_mem_mb": _list_stats(self.gpu_mem_mb_samples),
            }
        return base


# ── Frame Source ─────────────────────────────────────────────────────────────

class AvtrFrameSource:
    """Pulls I420 frames from the AVTR rendering pipeline.

    process_chunk() runs in a dedicated producer THREAD (never in the
    asyncio event loop) to avoid blocking WebRTC pacing/ICE/stats.
    """

    def __init__(
        self,
        pipeline,
        avatar,
        bg_id: str,
        max_queue: int = 10,
    ):
        self._pipeline = pipeline
        self._avatar = avatar
        self._bg_id = bg_id

        from avtr1_renderer.types import Chunk, RenderOptions
        self._Chunk = Chunk
        self._RenderOptions = RenderOptions

        mg = pipeline._motion_generator
        self._chunk_size = mg.chunk_size
        self._future_size = mg.future_size
        self._frame_len = mg.frame_len
        self._audio_shift = getattr(mg, "audio_shift", 0)
        self._window = (self._chunk_size + self._future_size) * self._frame_len + self._audio_shift
        self._step_samples = self._chunk_size * self._frame_len

        self._queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.metrics = PublisherMetrics()
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_frame: np.ndarray | None = None
        self._error: Exception | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True,
            name="avtr-frame-producer",
        )
        self._producer_thread.start()

    def _producer_loop(self) -> None:
        import torch

        # ── Stage 3B reliability repair: optional CUDA_LAUNCH_BLOCKING ──
        # CUDA errors are asynchronous: the traceback surfaces at an
        # unrelated later API call, making root-causing impossible.
        # Setting CUDA_LAUNCH_BLOCKING=1 makes all CUDA calls synchronous
        # so the traceback points at the exact failing kernel.
        #
        # Gated behind AVTR_CUDA_LAUNCH_BLOCKING env var, default OFF.
        # Set to "1" only for a bounded diagnostic window — it slows
        # inference by 2-10x.  Never enable in normal production.
        if os.environ.get("AVTR_CUDA_LAUNCH_BLOCKING", "0") == "1":
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            print("[webrtc-pub] CUDA_LAUNCH_BLOCKING=1 ENABLED (diagnostic mode — inference will be slower)")

        render_state = self._pipeline.initial_state(self._avatar)
        silence_audio = np.zeros(self._window, dtype=np.float32)
        audio_history = np.zeros(0, dtype=np.float32)
        options = self._RenderOptions(pixel_format="yuv_i420", bg_id=self._bg_id)

        chunk_i = 0
        while not self._stop_event.is_set():
            try:
                t_start = time.monotonic()

                audio_history = np.concatenate([audio_history, np.zeros(self._step_samples, dtype=np.float32)])
                if len(audio_history) >= self._window:
                    audio_tail = audio_history[-self._window:].copy()
                else:
                    audio_tail = np.pad(audio_history, (self._window - len(audio_history), 0))
                max_hist = self._window + self._step_samples * 2
                if len(audio_history) > max_hist:
                    audio_history = audio_history[-self._window:]

                chunk_obj = self._Chunk(audio_speech=audio_tail, audio_listen=silence_audio)
                with torch.inference_mode():
                    new_state, frames_iter = self._pipeline.process_chunk(
                        self._avatar, chunk_obj, state=render_state, options=options,
                    )
                    frames_list = list(frames_iter)
                render_state = new_state

                render_ms = (time.monotonic() - t_start) * 1000
                if TIMING_DEBUG:
                    self.metrics.process_chunk_ms.add(render_ms)
                    self.metrics.frames_per_chunk.add(len(frames_list))
                if chunk_i < 5 or chunk_i % 20 == 0:
                    print(f"[webrtc-pub] producer chunk {chunk_i}: {len(frames_list)} frames in {render_ms:.1f}ms")

                for frame in frames_list:
                    if self._stop_event.is_set():
                        break
                    while not self._stop_event.is_set():
                        try:
                            if self._loop and self._loop.is_running():
                                if TIMING_DEBUG:
                                    _t_copy = time.monotonic()
                                _frame_copy = frame.data.copy()
                                if TIMING_DEBUG:
                                    self.metrics.gpu_to_cpu_copy_ms.add(
                                        (time.monotonic() - _t_copy) * 1000
                                    )
                                    _t_put = time.monotonic()
                                future = asyncio.run_coroutine_threadsafe(
                                    self._queue.put(_frame_copy),
                                    self._loop,
                                )
                                future.result(timeout=5)
                                if TIMING_DEBUG:
                                    self.metrics.queue_put_ms.add(
                                        (time.monotonic() - _t_put) * 1000
                                    )
                                self.metrics.frames_generated += 1
                                break
                            else:
                                return
                        except asyncio.QueueFull:
                            if TIMING_DEBUG:
                                self.metrics.queue_full_retries += 1
                            time.sleep(0.01)
                        except Exception as exc:
                            print(f"[webrtc-pub] queue put error: {exc}")
                            time.sleep(0.01)

                chunk_i += 1

            except Exception as exc:
                self._error = exc
                print(f"[webrtc-pub] PRODUCER ERROR: {exc}")
                import traceback
                traceback.print_exc()
                break

        print(f"[webrtc-pub] producer thread exiting after {chunk_i} chunks")

    async def get_frame(self) -> np.ndarray:
        """Return the next frame, or raise MediaStreamError to close the track.

        Fail-fast policy (Stage 3B reliability repair):

        * If the producer thread set `_error` (e.g. CUDA illegal memory access),
          we allow at most `MAX_REPEATS_AFTER_ERROR` repeated frames so the last
          good frame still drains, then raise MediaStreamError.  This prevents
          the WebRTC track from emitting 25fps of frozen video forever.

        * Even without an explicit error, sustained queue underflow
          (`MAX_CONSECUTIVE_UNDERFLOWS` in a row, ≈10s at 25fps) is treated as
          a silent producer stall and also raises MediaStreamError.

        * Legitimate transient backpressure (queue briefly empty between chunks)
          is unaffected — we reset `consecutive_underflows` on every successful
          queue.pop().
        """
        depth = self._queue.qsize()
        if depth < self.metrics.queue_depth_min:
            self.metrics.queue_depth_min = depth
        if depth > self.metrics.queue_depth_max:
            self.metrics.queue_depth_max = depth

        try:
            frame = self._queue.get_nowait()
            self._last_frame = frame
            self.metrics.frames_emitted += 1
            # Transient backpressure cleared — reset the stall counter.
            self.metrics.consecutive_underflows = 0
            return frame
        except asyncio.QueueEmpty:
            self.metrics.underflow_count += 1
            self.metrics.consecutive_underflows += 1

            # ── Policy 1: producer thread set a hard error ──
            if self._error is not None:
                self.metrics.repeats_after_error += 1
                if self.metrics.repeats_after_error > MAX_REPEATS_AFTER_ERROR:
                    reason = (
                        f"producer_error_drained: error={type(self._error).__name__}: "
                        f"{str(self._error)[:200]}"
                    )
                    self.metrics.fail_fast_reason = reason
                    print(f"[webrtc-pub] FAIL-FAST: {reason}")
                    raise MediaStreamError()
                # Grace drain: emit the last good frame a few more times.
                if self._last_frame is not None:
                    self.metrics.frames_repeated += 1
                    return self._last_frame

            # ── Policy 2: silent sustained stall (no explicit error) ──
            if self.metrics.consecutive_underflows > MAX_CONSECUTIVE_UNDERFLOWS:
                reason = (
                    f"sustained_underflow: {self.metrics.consecutive_underflows} consecutive "
                    f"empty-queue reads (~{self.metrics.consecutive_underflows / FPS_25:.1f}s "
                    f"of frozen video) without producer error"
                )
                self.metrics.fail_fast_reason = reason
                print(f"[webrtc-pub] FAIL-FAST: {reason}")
                raise MediaStreamError()

            # ── Default: transient underflow, repeat last frame ──
            if self._last_frame is not None:
                self.metrics.frames_repeated += 1
                return self._last_frame
            h, w = 720, 1280
            black = np.zeros((h * 3 // 2, w), dtype=np.uint8)
            black[:h, :] = 16
            black[h:, :] = 128
            self._last_frame = black
            self.metrics.frames_repeated += 1
            return black

    def get_error(self) -> Exception | None:
        """Return the producer thread's fatal error, if any.

        Polled by the 5q session handler to surface CUDA/runtime errors to the
        Worker via session status, and to decide whether the container must be
        poisoned via modal.experimental.stop_fetching_inputs().
        """
        return self._error

    def stop(self) -> None:
        self._stop_event.set()


# ── Paced Video Track ────────────────────────────────────────────────────────

class PacedVideoTrack(VideoStreamTrack):
    """VideoStreamTrack with explicit 25fps wall-clock pacing.

    Overrides next_timestamp() to advance by 3600 ticks (40ms = 1/25s)
    instead of aiortc's hardcoded 3000 ticks (1/30s).
    """

    kind = "video"

    def __init__(self, frame_source: AvtrFrameSource, pc_ref=None):
        super().__init__()
        self._source = frame_source
        self._frame_count = 0
        self._pc_ref = pc_ref  # weak ref for diagnostics (RTCPeerConnection)
        self._loop_tid = threading.get_ident()
        # Set to True when get_frame() raises MediaStreamError due to the
        # fail-fast policy.  Polled by WebRtcPublisher.stop() so the stop
        # response can flag the failure mode for the 5q handler.
        self._fail_fast_triggered: bool = False

    async def next_timestamp(self) -> tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise MediaStreamError

        if hasattr(self, "_timestamp"):
            self._timestamp += TICKS_PER_FRAME_25
            wait = self._start + (self._timestamp / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
        else:
            self._start = time.time()
            self._timestamp = 0
        return self._timestamp, TIME_BASE

    async def recv(self) -> VideoFrame:
        _t_recv = time.monotonic() if TIMING_DEBUG else 0.0
        pts, time_base = await self.next_timestamp()
        try:
            frame_data = await self._source.get_frame()
        except MediaStreamError:
            # Fail-fast policy tripped — surface to aiortc which closes the
            # track and eventually the PC.  Flag for the stop() metrics.
            self._fail_fast_triggered = True
            print(
                f"[webrtc-pub] TRACK FAIL-FAST at frame={self._frame_count} "
                f"(producer_error={'YES' if self._source._error else 'no'})"
            )
            raise

        # Ensure the numpy array is contiguous and independent before handing
        # to PyAV.  This prevents use-after-free if the source buffer is reused.
        frame_data = np.ascontiguousarray(frame_data)

        if TIMING_DEBUG:
            _t_wrap = time.monotonic()
        if frame_data.ndim == 2:
            frame = VideoFrame.from_ndarray(frame_data, format="yuv420p")
        else:
            frame = VideoFrame.from_ndarray(frame_data, format="rgb24")
        if TIMING_DEBUG:
            self._source.metrics.from_ndarray_ms.add(
                (time.monotonic() - _t_wrap) * 1000
            )

        frame.pts = pts
        frame.time_base = time_base

        self._frame_count += 1

        if TIMING_DEBUG:
            self._source.metrics.recv_pacing_ms.add(
                (time.monotonic() - _t_recv) * 1000
            )

        # Periodic diagnostics + GC every 100 frames to prevent C++ object
        # accumulation in PyAV/av (suspected cause of SIGABRT after ~1000 frames).
        if self._frame_count % 100 == 0:
            m = self._source.metrics
            if TIMING_DEBUG:
                try:
                    import psutil
                    m.cpu_percent_samples.append(
                        psutil.Process().cpu_percent(interval=None)
                    )
                except Exception:
                    pass
                try:
                    import torch
                    m.gpu_mem_mb_samples.append(
                        round(torch.cuda.memory_allocated() / 1048576, 1)
                    )
                except Exception:
                    pass
            ice = "?"
            pcs = "?"
            if self._pc_ref is not None:
                try:
                    ice = self._pc_ref.iceConnectionState
                    pcs = self._pc_ref.connectionState
                except Exception:
                    pass
            qd = self._source._queue.qsize()
            prod_err = self._source._error
            print(
                f"[webrtc-pub] diag frame={self._frame_count} pts={pts} "
                f"qdepth={qd} gen={m.frames_generated} emit={m.frames_emitted} "
                f"repeat={m.frames_repeated} underflow={m.underflow_count} "
                f"ice={ice} pcs={pcs} loop_tid={self._loop_tid} "
                f"prod_tid={self._source._producer_thread.ident if self._source._producer_thread else '?'} "
                f"prod_err={'NONE' if prod_err is None else str(prod_err)[:80]}"
            )
            gc.collect()

        if self._frame_count <= 3 or self._frame_count % 250 == 0:
            print(f"[webrtc-pub] track frame {self._frame_count}: pts={pts} shape={frame_data.shape}")

        return frame


# ── Non-trickle ICE helper ───────────────────────────────────────────────────

async def wait_for_ice_gathering(pc: RTCPeerConnection, timeout_s: float = ICE_GATHER_TIMEOUT_S) -> None:
    """Wait for ICE gathering to complete (non-trickle).

    Blocks until iceGatheringState becomes "complete" or timeout expires.
    """
    if pc.iceGatheringState == "complete":
        return

    future = asyncio.get_running_loop().create_future()

    def on_state_change():
        if pc.iceGatheringState == "complete":
            if not future.done():
                future.set_result(None)

    pc.addListener("icegatheringstatechange", on_state_change)

    try:
        await asyncio.wait_for(future, timeout=timeout_s)
    except asyncio.TimeoutError:
        print(f"[webrtc-pub] ICE gathering timeout after {timeout_s}s (state={pc.iceGatheringState})")
    finally:
        pc.removeListener("icegatheringstatechange", on_state_change)


# ── Publisher — creates PC, exchanges SDP via Worker callback ────────────────

class WebRtcPublisher:
    """Manages the aiortc RTCPeerConnection lifecycle for SFU publishing.

    SDP exchange is brokered through the Worker/DO — never calls SFU directly.
    Modal authenticates to Worker with x-modal-secret header.
    """

    def __init__(
        self,
        frame_source: AvtrFrameSource,
        track_name: str,
        publisher_callback_url: str,
        close_callback_url: str,
    ):
        self._source = frame_source
        self._track_name = track_name
        self._callback_url = publisher_callback_url
        self._close_url = close_callback_url
        self._pc: RTCPeerConnection | None = None
        self._video_track: PacedVideoTrack | None = None
        self._video_mid: str | None = None
        self._modal_secret: str | None = os.environ.get("MODAL_WORKER_SECRET")

    async def start(self) -> str:
        """Create PC, exchange SDP, wait for ICE. Returns video mid."""
        if not self._modal_secret:
            raise RuntimeError("MODAL_WORKER_SECRET not set in environment")

        self._pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[RTCIceServer(urls=[STUN_SERVER])])
        )
        self._video_track = PacedVideoTrack(self._source, pc_ref=self._pc)
        self._pc.addTrack(self._video_track)

        # ICE state tracking
        ice_connected = asyncio.Event()
        ice_failed = asyncio.Event()

        @self._pc.on("iceconnectionstatechange")
        async def on_ice_state():
            state = self._pc.iceConnectionState
            print(f"[webrtc-pub] ICE state: {state}")
            if state in ("connected", "completed"):
                ice_connected.set()
            elif state == "failed":
                ice_failed.set()

        # Create offer
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)

        # Non-trickle: wait for ALL ICE candidates
        await wait_for_ice_gathering(self._pc)
        print(f"[webrtc-pub] ICE gathering complete, sending offer to Worker")

        # Find video mid
        self._video_mid = "0"
        for t in self._pc.getTransceivers():
            if t.kind == "video":
                self._video_mid = t.mid or "0"
                break

        if "a=candidate:" not in self._pc.localDescription.sdp:
            raise RuntimeError("No ICE candidates gathered")

        # Send offer to Worker (brokered — Worker proxies to SFU)
        offer_body = {
            "sdp": self._pc.localDescription.sdp,
            "type": "offer",
            "mid": self._video_mid,
            "track_name": self._track_name,
            "kind": "video",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self._callback_url,
                headers={
                    "x-modal-secret": self._modal_secret,
                    "content-type": "application/json",
                },
                json=offer_body,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"publisher-offer callback failed: {resp.status_code} {resp.text}")
            answer_data = resp.json()

        # Set remote description (answer from SFU, relayed by Worker)
        answer = RTCSessionDescription(
            sdp=answer_data["answer_sdp"],
            type=answer_data["answer_type"],
        )
        await self._pc.setRemoteDescription(answer)
        print(f"[webrtc-pub] Remote desc set, mid={self._video_mid}")

        # Wait for ICE connected
        await asyncio.wait_for(ice_connected.wait(), timeout=ICE_CONNECT_TIMEOUT_S)
        if ice_failed.is_set():
            raise RuntimeError("ICE failed")

        print(f"[webrtc-pub] ICE connected — publishing AVTR frames")
        return self._video_mid

    async def stop(self) -> dict:
        """Close publisher track via Worker callback, then close PC.

        Returns a metrics dict that includes any producer-side fatal error
        (`producer_error`) and the fail-fast reason (`fail_fast_reason`) so
        the 5q session handler can poison the container and surface the
        cause to the Worker via session-end telemetry.
        """
        metrics = self._source.metrics.snapshot()

        # Surface producer-side fatal errors in the stop response so the
        # 5q handler can poison the container (modal stop_fetching_inputs).
        producer_error = self._source.get_error()
        if producer_error is not None:
            metrics["producer_error"] = f"{type(producer_error).__name__}: {str(producer_error)[:300]}"
        track_fail_fast = getattr(self._video_track, "_fail_fast_triggered", False) if self._video_track else False
        if track_fail_fast:
            metrics["track_fail_fast_triggered"] = True

        # Close track via Worker (brokered)
        if self._video_mid and self._close_url:
            close_body = {"tracks": [{"mid": self._video_mid}], "force": True}
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.put(
                        self._close_url,
                        headers={
                            "x-modal-secret": self._modal_secret or "",
                            "content-type": "application/json",
                        },
                        json=close_body,
                    )
                    print(f"[webrtc-pub] publisher-close: status={resp.status_code} body={resp.text[:200]}")
            except Exception as exc:
                print(f"[webrtc-pub] publisher-close error: {exc}")

        # Stop frame source and WAIT for producer thread to exit before closing PC.
        # This prevents the C++ SIGABRT ("terminate called without an active
        # exception") that occurs when the RTCPeerConnection is closed while the
        # producer thread is still bridging frames via run_coroutine_threadsafe().
        self._source.stop()
        if self._source._producer_thread and self._source._producer_thread.is_alive():
            self._source._producer_thread.join(timeout=5)
            if self._source._producer_thread.is_alive():
                print("[webrtc-pub] WARNING: producer thread did not exit within 5s")

        # Close PC (now safe — producer thread is stopped)
        if self._pc:
            await self._pc.close()
            print(f"[webrtc-pub] PC closed")

        return metrics


def loop_alive(frame_source) -> bool:
    """Returns True if the publisher's event loop is still running.

    Convenience function for smoke/health checks.  The frame_source captures
    the loop during AvtrFrameSource.start() — if the loop is dead, the producer
    thread has exited and no more frames will be generated.
    """
    loop = getattr(frame_source, "_loop", None)
    return loop is not None and loop.is_running()


__all__ = [
    "AvtrFrameSource",
    "PacedVideoTrack",
    "WebRtcPublisher",
    "PublisherMetrics",
    "loop_alive",
]
